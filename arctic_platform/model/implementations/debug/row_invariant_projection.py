# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Make a projection's rows independent of how many rows share the call.

A bfloat16 ``[M, K] x [K, N]`` product on H200 can return a different answer for a row depending on how many
rows travelled with it. cuBLASLt splits the contraction and sums the partial products in a separate float32
reduction when the workspace can hold them, and that reduction groups the dot product differently from the
unsplit kernel a taller call selects.

Zero rows contribute nothing to a matmul, so running a short call at a row count where the kernel selection is
already pinned and slicing the wanted rows back out returns exactly what a tall call returns for them. That is
the whole mechanism, and it is why this is a no-op in arithmetic rather than a second approximation.

**Where that mechanism applies is decided by the alignment of the contraction, not by its size.** Two
different effects produce a row-count dependence and only one of them is a split-K reduction over the output:

* A contraction that is a multiple of :data:`CONTRACTION_ALIGNMENT` is exposed only through the split-K
  reduction, whose eligibility is bounded by the output. Padding rows up to
  :func:`invariant_row_target` removes it exactly.
* A contraction that is not selects a different kernel per row count for a reason the output size does not
  bound, and no row count removes it. Padding such a call spends the arithmetic and delivers nothing, so this
  module refuses the model rather than reporting it as covered.

Both figures below are measured on H200 under CUDA 13 in bfloat16, against the same rows computed inside a
4096-row call, with every same-shape repeat floor ``0.000e+00``. At ``CUBLAS_WORKSPACE_CONFIG=":16:8"``,
``K`` 2500 / ``N`` 1024 deviates by 3.906e-02 unpadded and by 3.906e-02 after padding to its derived target --
the pad moves it nowhere. At the same workspace ``K`` 4096 / ``N`` 1024 deviates by 7.812e-03 unpadded and by
0.000e+00 padded. A ``[1, tokens, K]`` activation and a ``[tokens, K]`` one give the same bits at every
shape and row count measured, so the leading batch dimension is not a third case.

Where the boundary sits for an aligned contraction is not a table. A split-K kernel holds one float32 partial
product per output element, so it is eligible only while that buffer fits the workspace, and the call is
row-count dependent while

    ``rows * out_features < workspace_bytes / 8``

with ``workspace_bytes`` read from the environment cuBLAS actually initialized with, and from
:data:`DRIVER_DEFAULT_WORKSPACE` when the variable is absent. :func:`invariant_row_target` evaluates that
condition rather than looking up a width, so the target follows the workspace instead of a table that would
silently become wrong.

Note which direction absence points. An unset variable is the *larger* workspace, and the exposed band is
therefore wider and not narrower: with the variable unset, ``K`` 12288 / ``N`` 1024 follows the row count
through 1024 rows where at ``":16:8"`` it is pinned from 16. The pad therefore grows when the workspace is
absent and never stands down on account of it. It is also why a string in the environment is not taken as
proof: :func:`verify_targets_cover_the_runtime` measures the boundary on the device and raises if the device
demonstrates more workspace than the targets assume.

A wide output clears the boundary on its own, so the condition spares it without being told to. A *narrow*
output does the opposite: the target rises as the width falls, and at one column it reaches the whole bound,
which is a pad no call can afford. Those are excluded by :data:`MIN_PADDABLE_OUTPUT_WIDTH` and stay exposed,
which is a decision with a cost behind it rather than an oversight.

Where this does not reach: a sampler in its own process, and the two backward products, whose output widths
are ``in_features`` and so cross their boundaries at different row counts from the forward. A trainer-against-
sampler comparison needs a tolerance rather than an equality; see ``docs/determinism.md``.
"""

from __future__ import annotations

import types
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional

import torch
import torch.nn.functional as F

from .determinism import full_determinism_enabled

# The contraction quantum that decides whether padding rows can deliver anything at all. This is the invariant
# the whole module rests on:
#
#     a contraction that is a multiple of CONTRACTION_ALIGNMENT is row-count invariant at and above
#     invariant_row_target(out_features), and one that is not is row-count dependent at row counts a pad
#     cannot reach
#
# Eight is the measured quantum rather than a guess at a tile size: on H200 under CUDA 13 in bfloat16, ``K``
# 1032 -- a multiple of eight and not of 64 -- is bit-identical to a 4096-row call at every row count from 1
# to 2048 at every workspace tried, while ``K`` 1025, 1026 and 1028 deviate by up to 3.125e-02 there. Sixty-four
# would therefore refuse widths that need nothing: 1032, 1056, 2504 and 2568 are all invariant or paddable.
CONTRACTION_ALIGNMENT = 8

# Aligned contractions below this are invariant at every row count measured, because cuBLASLt only considers
# splitting a contraction this short across blocks when the split can pay for itself. Padding them would add
# arithmetic where there is no dependence to remove.
#
# The boundary is sharp and sits between two adjacent multiples of CONTRACTION_ALIGNMENT. Sweeping every
# multiple of eight from 2496 to 2656 at ``N`` 1024 on H200 under CUDA 13 in bfloat16, over every row count
# from 1 to 64 and then 96, 128, 192, 256, 512 and 1024 against a 4096-row call, floor 0.000e+00 throughout:
# every width to 2560 deviates 0.000e+00 at every row count, and every width from 2568 up deviates, by
# 1.953e-03 to 1.562e-02, until it reaches its own row target. Identical at ``":16:8"`` and ``":4096:8"``,
# which differ only in where the target lands -- 16 rows against 192.
#
# Two things this bound does not say. It is only meaningful for an aligned contraction, since an unaligned
# one moves far below it -- ``K`` 1028 at ``N`` 1024 deviates by 1.562e-02 -- which is why the alignment test
# comes first and is not a refinement of this one. And it is measured at output widths of 256 and above; at
# ``N`` 2 a contraction of 1024 already follows its row count, at 1.562e-02 against a 0.000e+00 floor. An
# output that narrow has a target of 8192 rows at ``":16:8"``, which no call reaches, so it is unprotected
# either way -- the tests that enumerate projection widths require any such width a configuration issues to
# be declared as an accepted exposure rather than left to this floor.
MIN_DEPENDENT_CONTRACTION = 2568

# Workspace bytes per output element at which the boundary is observed. A float32 partial product is four
# bytes, so eight is the measured eligibility threshold rather than the buffer's own size: cuBLASLt stops
# selecting the split while the buffer would still fit, leaving a factor of two in hand. At ``":16:8"`` this
# reproduces the measured boundary exactly -- 16 rows at ``N`` 1024 and 4 at ``N`` 4096, for every aligned
# contraction from 2592 to 12288. At ``":4096:8"`` it over-predicts, asking 4096 rows at ``N`` 1024 where 256
# already measures invariant, which costs a larger pad and stays exact.
WORKSPACE_BYTES_PER_OUTPUT_ELEMENT = 8

# What cuBLAS uses when CUBLAS_WORKSPACE_CONFIG is absent from the environment. Measured rather than taken from
# documentation: a sweep of first-invariant row counts over 19 contraction widths by two output widths is
# bit-identical with the variable unset and with it set to this value, so the driver's own workspace behaves as
# 32 MiB on H200 under CUDA 13. This value matters because an absent variable is the *larger* workspace and
# therefore the wider exposure: a rule that treated absence as "nothing to do" would switch the pad off exactly
# where the band is widest.
DRIVER_DEFAULT_WORKSPACE = ":4096:8"

# The narrowest output the pad will cover. The target rises as the output narrows -- the eligibility bound is
# on ``rows * N``, so halving the width doubles the rows -- and at a single column it reaches the whole bound:
# 16384 rows at ``":16:8"`` and 4194304 with the variable unset. Materialising that operand on every call is a
# cost no call can carry, and a single column cannot feed a ``topk`` over columns, so no categorical change can
# follow from the movement left in place.
#
# The rule is on the width rather than on the target for a reason a row ceiling cannot satisfy. A fixed ceiling
# of a few hundred rows holds only while ``CUBLAS_WORKSPACE_CONFIG`` is small: with the variable unset the
# bound is far larger, a router gate's own target grows past any such ceiling, and the ceiling would exclude
# *it* -- reopening a categorical failure in exactly the configuration where the exposure is widest. Only a
# statement about the output width is stable across workspaces.
MIN_PADDABLE_OUTPUT_WIDTH = 2

# The target is rounded up to a power of two once the condition has been evaluated. Kept separate from the
# condition so the two can be read apart -- at ``N`` 1536 the boundary is 11 rows and the target 16 -- but not
# optional: the condition is necessary and not sufficient, and at a width that is not a power of two the call
# stays dependent above the boundary. Padding to the bare boundary would leave those widths exposed.
POWER_OF_TWO_MARGIN = True


def workspace_bytes(workspace_config: str) -> int:
    """Total bytes cuBLAS may use for workspace under a ``CUBLAS_WORKSPACE_CONFIG`` value.

    The format is one or more ``:size:count`` pairs, ``size`` in KiB -- ``":16:8"`` is eight buffers of 16 KiB.
    Rejects a value it cannot parse rather than falling back to a default, because a misparsed workspace yields
    a target that is too small and a pad that quietly does not deliver invariance. Use
    :func:`effective_workspace_config` to resolve an absent value, which is a different question.
    """
    fields = [field for field in workspace_config.split(":") if field != ""]
    if not fields or len(fields) % 2 != 0:
        raise ValueError(f"CUBLAS_WORKSPACE_CONFIG must be one or more ':size:count' pairs, got {workspace_config!r}")
    total = 0
    for size_kib, count in zip(fields[::2], fields[1::2]):
        total += int(size_kib) * 1024 * int(count)
    return total


def invariant_row_target(out_features: int, workspace_config: Optional[str] = None) -> int:
    """Rows a call to a projection of this width must carry for its result to equal a tall call's.

    A split-K kernel is eligible while its float32 partial-product buffer for the whole output fits the
    workspace, so the dependence holds while ``rows * out_features`` is under
    ``workspace_bytes / WORKSPACE_BYTES_PER_OUTPUT_ELEMENT`` and the target is the first row count at or above
    that. Derived on every call rather than tabulated, so no table can disagree with the workspace actually in
    force. ``workspace_config`` defaults to :func:`effective_workspace_config`, which is the workspace the
    process really has and not the one some configuration intended it to have.

    The target is only a statement about a contraction that is a multiple of :data:`CONTRACTION_ALIGNMENT`;
    :func:`contraction_is_paddable` is what decides whether it means anything for a given projection.
    """
    if out_features <= 0:
        raise ValueError(f"out_features must be positive, got {out_features}")
    if workspace_config is None:
        workspace_config = effective_workspace_config()
    elements = workspace_bytes(workspace_config) // WORKSPACE_BYTES_PER_OUTPUT_ELEMENT
    boundary = -(-elements // out_features)
    if not POWER_OF_TWO_MARGIN:
        return max(boundary, 1)
    target = 1
    while target < boundary:
        target *= 2
    return target


def effective_workspace_config(environ: Optional[Mapping[str, str]] = None) -> str:
    """The cuBLAS workspace this process actually has, which is what the row targets have to be derived from.

    Reads the environment rather than any configuration value, because cuBLAS reads
    ``CUBLAS_WORKSPACE_CONFIG`` once when it creates its handle: a value placed anywhere else, or set after
    that point, has no effect on the kernels this process will select. An absent or empty variable resolves to
    :data:`DRIVER_DEFAULT_WORKSPACE`, so the targets widen to cover the larger workspace. Treating absence as
    "no padding needed" would be backwards -- absence is the widest exposure there is.

    A string here is a claim about the runtime, not proof of one. :func:`verify_targets_cover_the_runtime`
    checks the claim against the device.
    """
    if environ is None:
        import os

        environ = os.environ
    configured = environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not configured:
        return DRIVER_DEFAULT_WORKSPACE
    return configured


def contraction_is_paddable(in_features: int) -> bool:
    """Whether padding rows can make a projection over this contraction row-count invariant.

    True exactly when the contraction is a multiple of :data:`CONTRACTION_ALIGNMENT`. That is the condition
    under which the only row-count dependence is cuBLASLt's split-K reduction over the output, which a taller
    call escapes. An unaligned contraction is dependent at row counts no affordable pad reaches, so padding it
    is arithmetic spent for nothing and, worse, reports coverage that does not exist.
    """
    return in_features % CONTRACTION_ALIGNMENT == 0


class RowInvarianceUnverified(RuntimeError):
    """The workspace the row targets were derived from is smaller than the one the device is really using.

    Raised rather than logged. A pad sized against a workspace the process does not have is not a partial fix:
    it pads to a row count that is itself still row-count dependent, and it does so silently, which is worse
    than not padding at all because nothing downstream can tell.
    """


class RowInvariancePadCannotCover(RuntimeError):
    """A projection contracts over a width that padding rows cannot make row-count invariant.

    Raised rather than skipped for the same reason as :class:`RowInvarianceUnverified`. Skipping would leave
    the model with a projection whose answer for a token depends on how many tokens shared the call, larger
    than the deviation the pad removes elsewhere, and with nothing in the patch count or the log to say so.
    """


# Shape of the probe that measures where the boundary really is. Narrow and short on purpose: the contraction
# is a multiple of CONTRACTION_ALIGNMENT and reaches past MIN_DEPENDENT_CONTRACTION so the split-K dependence
# exists at all, and a 1024-wide output puts the boundary at the highest row count of any production width,
# which is what makes two workspaces easy to tell apart -- 16 rows at ``":16:8"`` against 256 with the variable
# unset.
PROBE_CONTRACTION = 4096
PROBE_OUTPUT_WIDTH = 1024
PROBE_REFERENCE_ROWS = 4096
PROBE_ROW_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def measure_workspace_bytes(device: torch.device, *, dtype: torch.dtype = torch.bfloat16) -> int:
    """Workspace the device's kernel selection behaves as though it has, in bytes.

    Sweeps row counts on one probe shape against the same rows inside a tall call and finds the first that
    agrees. Inverting the condition, ``rows * out_features * WORKSPACE_BYTES_PER_OUTPUT_ELEMENT`` is the
    workspace that row count implies. Returns a lower bound when the sweep does not reach an invariant row
    count, which is still enough to reject a workspace claim that is far too small.
    """
    generator = torch.Generator(device=device).manual_seed(0x5150)
    weight = (
        torch.randn(PROBE_OUTPUT_WIDTH, PROBE_CONTRACTION, generator=generator, device=device, dtype=torch.float32)
        * 0.02
    ).to(dtype)
    activation = torch.randn(PROBE_REFERENCE_ROWS, PROBE_CONTRACTION, generator=generator, device=device, dtype=dtype)
    with torch.no_grad():
        reference = F.linear(activation, weight)
        for rows in PROBE_ROW_COUNTS:
            if rows >= PROBE_REFERENCE_ROWS:
                break
            deviation = F.linear(activation[:rows].contiguous(), weight) - reference[:rows]
            if not bool(deviation.any()):
                return rows * PROBE_OUTPUT_WIDTH * WORKSPACE_BYTES_PER_OUTPUT_ELEMENT
    return PROBE_ROW_COUNTS[-1] * PROBE_OUTPUT_WIDTH * WORKSPACE_BYTES_PER_OUTPUT_ELEMENT


def verify_targets_cover_the_runtime(
    workspace_config: str, device: torch.device, *, dtype: torch.dtype = torch.bfloat16
) -> int:
    """Check the workspace the targets assume against the one the device demonstrates, and raise if it is less.

    One-sided by design. A device that behaves as though it has *less* workspace than assumed only means the
    targets are larger than they need to be, which costs a little and stays exact. A device that behaves as
    though it has *more* means the targets are too small and the pad does not deliver invariance, which is the
    failure this exists to make impossible to miss. Returns the measured figure so a caller can log it.
    """
    assumed = workspace_bytes(workspace_config)
    measured = measure_workspace_bytes(device, dtype=dtype)
    if measured > assumed:
        raise RowInvarianceUnverified(
            f"row targets were derived from {workspace_config!r} ({assumed} bytes) but this device's kernel "
            f"selection behaves as though it has at least {measured} bytes, so every target is too small and "
            "padding to it would not make a projection row-count invariant. cuBLAS reads "
            "CUBLAS_WORKSPACE_CONFIG once when it creates its handle, so the usual cause is that the variable "
            "was set after that point, or set in a parent process that did not pass it to this one, or placed "
            "in a configuration rather than the environment. Set it in the environment the process is spawned "
            "with."
        )
    return measured


def row_invariant_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    target: Optional[int] = None,
) -> torch.Tensor:
    """``F.linear``, with a short call run at the row count where the kernel selection is already pinned.

    All leading dimensions of ``input`` are rows of the product, which is what cuBLAS sees, so a
    ``[1, tokens, K]`` activation is a call of ``tokens`` rows and not of one. Returns ``F.linear``'s own
    result untouched whenever the call already carries enough rows, so the region that is invariant without
    this pays one integer comparison and keeps its kernel.

    ``target`` is the row count to pad up to, derived from the weight when not supplied. The patched forward
    supplies it, because deriving it parses the workspace configuration and every training call would pay for
    that to be told the row count it already has is high enough.

    Raises :class:`RowInvariancePadCannotCover` on a contraction padding cannot cover. The check is first,
    before the row count is even looked at, because an unaligned contraction has no row count at which it is
    safe: at ``K`` 2500 / ``N`` 1024 in bfloat16 on H200 under CUDA 13, a 256-row call still disagrees with
    the same rows inside a 4096-row call by 3.125e-02 against a 0.000e+00 floor, well above the 16-row target
    that workspace gives. A refusal conditioned on the row count would therefore pass exactly the calls it
    cannot vouch for.
    """
    in_features = weight.shape[-1]
    if not contraction_is_paddable(in_features):
        raise RowInvariancePadCannotCover(
            f"a projection contracting over {in_features} is not a multiple of {CONTRACTION_ALIGNMENT}, so "
            "its result follows the row count of the call at row counts padding cannot reach and no row "
            "target removes it. Align the contraction, or exclude this module from the pad and record what "
            "protects it."
        )
    rows = input.numel() // in_features
    if target is None:
        target = invariant_row_target(weight.shape[0])
    if rows >= target or rows == 0 or in_features < MIN_DEPENDENT_CONTRACTION:
        return F.linear(input, weight, bias)
    flat = input.reshape(rows, in_features)
    # Concatenation rather than an in-place write into a zero buffer: backward then slices the pad away by
    # construction, and the zero rows carry no gradient to the weight because they carry no value to the
    # output.
    padded = torch.cat([flat, flat.new_zeros((target - rows, in_features))], dim=0)
    projected = F.linear(padded, weight, bias)[:rows]
    return projected.reshape(*input.shape[:-1], weight.shape[0])


def make_row_invariant_forward(target: int) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]:
    """A ``forward`` for one output width, holding that width's row target so no call re-derives it."""

    def forward(self: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
        return row_invariant_linear(input, self.weight, self.bias, target)

    return forward


def is_row_count_dependent(module: torch.nn.Module) -> bool:
    """Whether this module is a projection the pad both needs to cover and can afford to cover.

    Four conditions. The forward has to be the one this pad reimplements: a subclass of :class:`torch.nn.Linear`
    that overrides ``forward`` does something other than the projection, and replacing it would either raise on
    the arguments it expects or, worse, quietly drop what it does. The contraction has to be a multiple of
    :data:`CONTRACTION_ALIGNMENT`, without which padding rows removes nothing -- an unaligned width is not
    excluded here as safe but refused by :func:`apply_row_invariant_projections`. It has to reach
    :data:`MIN_DEPENDENT_CONTRACTION`, below which there is no dependence to remove. And the output has to be
    at least :data:`MIN_PADDABLE_OUTPUT_WIDTH` columns wide, below which the pad is larger than the activations
    it would protect.

    Beyond that the output width does not exclude a module, it sets its target: a wide output is pinned by four
    rows and a narrow one needs sixteen, and both are exposed below their own target.
    """
    return (
        isinstance(module, torch.nn.Linear)
        and type(module).forward is torch.nn.Linear.forward
        and contraction_is_paddable(module.in_features)
        and module.in_features >= MIN_DEPENDENT_CONTRACTION
        and module.out_features >= MIN_PADDABLE_OUTPUT_WIDTH
    )


def row_invariant_projections_requested(training_config: Mapping[str, Any]) -> bool:
    """Whether this run asked for row-invariant projections.

    The pad runs every covered projection at a taller row count than its call asked for, on every call. What it
    buys -- a token's result not moving with the height of the batch it arrived in -- is read by a parity
    comparison, so it follows ``debug.full_determinism`` rather than every run. A production step keeps the
    projection it had, and stays exposed to the deviation ``docs/determinism.md`` measures.
    """
    return full_determinism_enabled(training_config)


def maybe_apply_row_invariant_projections(
    model: torch.nn.Module,
    training_config: Mapping[str, Any],
    *,
    logger=None,
    **kwargs,
) -> int:
    """Install the pad on ``model`` if this run asked for it, and leave the model alone if it did not.

    The entry point a model loader calls once the model is built. Whether the pad belongs in a given run is
    decided here rather than at the loader, so a caller neither reads the flag nor carries the reason: a run
    that did not ask returns 0 with nothing patched, and the projections keep the row-count dependence the
    module docstring measures. Remaining keyword arguments reach
    :func:`apply_row_invariant_projections` unchanged.
    """
    if not row_invariant_projections_requested(training_config):
        return 0
    if logger is None:
        from arctic_platform.model.implementations.moe.logging_utils import get_logger

        logger = get_logger()
    return apply_row_invariant_projections(model, logger=logger, **kwargs)


def apply_row_invariant_projections(
    model: torch.nn.Module,
    *,
    is_target: Callable[[torch.nn.Module], bool] = is_row_count_dependent,
    environ: Optional[Mapping[str, str]] = None,
    workspace_config: Optional[str] = None,
    verify: bool = True,
    logger=None,
) -> int:
    """Pad-and-slice every linear projection whose result would otherwise follow its row count.

    Applies to every projection the predicate both needs to cover and can afford to; a width narrower than
    :data:`MIN_PADDABLE_OUTPUT_WIDTH` is left exposed and named in the log. There is no configuration in which
    the dependence is absent and nothing to do: the workspace only moves where the boundary sits, and an unset
    one moves it *up*, so the targets are derived from :func:`effective_workspace_config` and grow when the
    workspace is larger. Returns the number of modules patched. Safe before or after activation-checkpoint
    wrapping, and a no-op in results at every row count: a call at or above its target keeps the code path it
    had.

    Refuses a model containing a projection whose contraction is not a multiple of
    :data:`CONTRACTION_ALIGNMENT`, because nothing here can make it row-count invariant. The check runs over
    every :class:`torch.nn.Linear` in the tree and before anything is patched, so a model is either wholly
    covered or wholly untouched.

    With ``verify`` and a CUDA device present, the workspace the targets assume is checked against the one the
    device demonstrates, and :class:`RowInvarianceUnverified` is raised if the device has more. Padding to a
    target that does not reach the real boundary is the one failure mode that produces no symptom, so it is
    made loud here rather than left to be discovered in a log line.
    """
    if workspace_config is None:
        workspace_config = effective_workspace_config(environ)

    unpaddable = [
        f"{name or type(module).__name__} (K={module.in_features}, N={module.out_features})"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and not contraction_is_paddable(module.in_features)
    ]
    if unpaddable:
        raise RowInvariancePadCannotCover(
            f"{len(unpaddable)} projection(s) contract over a width that is not a multiple of "
            f"{CONTRACTION_ALIGNMENT}. Such a projection returns a different answer for a token depending on "
            "how many tokens shared the call, at row counts no affordable pad reaches, so padding rows would "
            "spend the arithmetic and remove none of it. The remedy is to align the contraction -- an "
            "intermediate size, a head dimension or an expert width -- rather than to widen any threshold "
            "here: "
            + ", ".join(unpaddable)
        )

    patched = 0
    widths: dict[int, int] = {}
    left_exposed: dict[int, int] = {}
    forwards: dict[int, Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = {}
    for module in model.modules():
        if is_target(module):
            width = module.out_features
            if width not in forwards:
                widths[width] = invariant_row_target(width, workspace_config)
                forwards[width] = make_row_invariant_forward(widths[width])
            module.forward = types.MethodType(forwards[width], module)
            patched += 1
        elif isinstance(module, torch.nn.Linear) and module.in_features >= MIN_DEPENDENT_CONTRACTION:
            left_exposed[module.out_features] = invariant_row_target(module.out_features, workspace_config)

    measured = None
    if verify and patched and torch.cuda.is_available():
        device = next(
            (parameter.device for parameter in model.parameters() if parameter.is_cuda),
            torch.device("cuda"),
        )
        measured = verify_targets_cover_the_runtime(workspace_config, device)

    if logger is not None and patched:
        logger.info(
            "Row-invariant projections applied to %d linear module(s) with contraction >= %d and a multiple "
            "of %d; effective workspace %s (%d bytes)%s; row targets by output width: %s",
            patched,
            MIN_DEPENDENT_CONTRACTION,
            CONTRACTION_ALIGNMENT,
            workspace_config,
            workspace_bytes(workspace_config),
            "" if measured is None else f", device behaves as {measured} bytes",
            ", ".join(f"N={width}:{target}" for width, target in sorted(widths.items())),
        )
    if logger is not None and left_exposed:
        logger.info(
            "Row-invariant projections left %d output width(s) unpatched, so their result still follows the "
            "row count; each is narrower than %d columns and would need the row target shown: %s",
            len(left_exposed),
            MIN_PADDABLE_OUTPUT_WIDTH,
            ", ".join(f"N={width}:{target}" for width, target in sorted(left_exposed.items())),
        )
    return patched
