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

"""The MoE combine-side row sum folds a token's expert rows in an order fixed by the program.

Routing sends a token to ``top_k`` experts, so several rows of expert output fold onto one row of the layer's
output. Folding them with ``scatter_add`` issues one atomic per repeated destination, and atomics are applied in
whatever order the hardware delivers them, which is not a property of the program: two runs of the same
configuration then disagree. Two addends commute, so the effect begins at three, which makes the multiplicity of
the reduction -- how many of a token's experts one rank holds -- the quantity that decides whether a
configuration is affected. The tests below record it and refuse to pass on a case that never reaches three.

The delivery-order effect exists only where atomics contend, so the tests that observe it need CUDA and carry
``integration``. What runs without a GPU is everything the order-fixing construction has to satisfy regardless of
device: that it computes the same sum the atomic form does, that its association does not depend on how many
other tokens share the batch, that its refusals fire on the inputs they name, and that its backward carries the
same linear map.

The fixed-order fold is what ``debug.full_determinism`` selects, through the one call a model loader makes to
``debug.token_combine.maybe_use_fixed_order_row_sum``: it reads the routing width off the model and replaces the
shared production fold in the modules that fold rows, so no combine site names the debug package or carries the
flag. A step without that request folds with atomics, because the fixed order synchronizes with the device to
check the widest group and writes a padded layout of ``num_tokens * top_k`` rows. Most cases below reach the
construction directly; installing it is held by its own cases.

``torch.use_deterministic_algorithms`` also removes the delivery-order effect, and none of these tests may enable
it: the fold has to fix its own order, without a global flag that pins every kernel in the process. Every
comparison here is therefore ``rtol=0, atol=0`` with the flag left alone.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.debug.token_combine import fixed_order_sum_rows_by_token
from arctic_platform.model.implementations.debug.token_combine import maybe_use_fixed_order_row_sum
from arctic_platform.model.implementations.debug.token_combine import routing_width
from arctic_platform.model.implementations.moe.token_combine import sum_rows_by_token

# Enough rows that the comparison covers many independent reductions rather than a handful.
UNIT_TOKENS = 4096
UNIT_HIDDEN = 512
UNIT_TOP_K = 4
REPEATS = 3

# The expert-parallel arm shards eight experts over two ranks, so a token's four experts split between the two
# and a rank holding three or four of them is common rather than contrived.
EP_NUM_EXPERTS = 8
EP_TOP_K = 4
EP_DIM = 256
EP_HIDDEN = 512
EP_TOKENS = 512

# Token counts and neighbouring-group widths for the batch-composition sweep. The widths stop at ``UNIT_TOP_K``
# because a wider group is what the refusal below rejects. The observed token carries three rows -- the fewest
# that make reduction order matter at all -- so the widest group in the batch varies with the neighbour rather
# than being pinned by the observed token.
SWEEP_TOKEN_COUNTS = (128, 256, 384, 512)
SWEEP_NEIGHBOUR_WIDTHS = (1, 2, 3, 4)
SWEEP_TARGET_ROWS = 3

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.integration)]


def _free_port() -> int:
    from arctic_platform.testing_utils import get_unique_port_number
    from arctic_platform.testing_utils import reserve_free_port

    return reserve_free_port(get_unique_port_number() + 1, span=7)


def _rounding_sensitive_rows(seed: int, dtype: torch.dtype, device: str):
    """``top_k`` rows per token, with values chosen so exact arithmetic is cheap but rounding order shows.

    A bfloat16 value in the binade [1, 2) is a multiple of 2^-8, so a sum of four of them is a multiple of 2^-8
    with magnitude below 2^3: eleven significant bits. That is exact in float32 and in float64, and it is not
    representable in bfloat16. The pair of properties is what the comparison needs -- the reference can be
    computed exactly, while the result still depends on whether rounding happens once at the end or after every
    addend. Widening the range would break the first property and shrinking it would break the second.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_indices = torch.arange(UNIT_TOKENS).repeat(UNIT_TOP_K).to(device)
    shape = (token_indices.numel(), UNIT_HIDDEN)
    magnitude = 1.0 + torch.rand(shape, generator=generator)
    sign = torch.where(torch.rand(shape, generator=generator) < 0.5, -1.0, 1.0)
    return (sign * magnitude).to(device=device, dtype=dtype), token_indices


def _exact_row_sum(rows, token_indices, num_tokens):
    """The sum in float64, then rounded once. Order-independent because every partial sum is exact."""
    exact = torch.zeros(num_tokens, rows.shape[1], dtype=torch.float64, device=rows.device)
    exact.index_add_(0, token_indices, rows.double())
    return exact.to(rows.dtype)


def _atomic_row_sum(rows, token_indices, num_tokens):
    """The reduction this replaces, kept here as the thing the fixed-order form has to agree with."""
    destination = token_indices.reshape(-1, 1).expand(-1, rows.shape[1])
    return torch.zeros(num_tokens, rows.shape[1], dtype=rows.dtype, device=rows.device).scatter_add(
        0, destination, rows
    )


def _sweep_case(num_tokens: int, neighbour_width: int, target_rows: torch.Tensor):
    """One batch containing ``target_rows`` as token 1, a neighbour of ``neighbour_width`` rows, and filler.

    Token 1 keeps the same rows in the same relative order in every case, so any change in its output comes from
    the rest of the batch. The filler values are drawn per case, which is what makes bit-identity of token 1 a
    statement about the reduction rather than about the inputs happening to match.
    """
    hidden = target_rows.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(1000 * num_tokens + neighbour_width)
    neighbour = torch.rand((neighbour_width, hidden), generator=generator) + 1.0
    filler = torch.rand((num_tokens - 2, hidden), generator=generator) + 1.0

    rows = torch.cat(
        [
            neighbour.to(device=target_rows.device, dtype=target_rows.dtype),
            target_rows,
            filler.to(device=target_rows.device, dtype=target_rows.dtype),
        ]
    )
    token_indices = torch.cat(
        [
            torch.zeros(neighbour_width, dtype=torch.int64),
            torch.ones(target_rows.shape[0], dtype=torch.int64),
            torch.arange(2, num_tokens, dtype=torch.int64),
        ]
    ).to(target_rows.device)
    return rows, token_indices


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_row_sum_agrees_with_the_atomic_form_it_replaces(dtype):
    """Same value, not merely a close one: on the CPU the atomic form is sequential, so both bit patterns exist.

    The floor for this comparison is zero -- the fixed-order sum called twice on one input is bit-identical, and
    so is ``scatter_add`` here -- which is why the assertion is exact rather than tolerant.
    """
    rows, token_indices = _rounding_sensitive_rows(seed=7, dtype=dtype, device="cpu")
    multiplicity = torch.bincount(token_indices, minlength=UNIT_TOKENS)
    assert int(multiplicity.max()) >= 3, (
        f"two addends commute, so a case whose largest multiplicity is {int(multiplicity.max())} says nothing "
        "about reduction order"
    )

    first = fixed_order_sum_rows_by_token(rows, token_indices, UNIT_TOKENS, UNIT_TOP_K)
    repeated = fixed_order_sum_rows_by_token(rows, token_indices, UNIT_TOKENS, UNIT_TOP_K)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0, msg="the fixed-order sum is not repeating itself")

    atomic = _atomic_row_sum(rows, token_indices, UNIT_TOKENS)
    torch.testing.assert_close(
        first,
        atomic,
        rtol=0,
        atol=0,
        msg="the fixed-order sum returns a different value than the reduction it replaces",
    )


def test_bfloat16_row_sum_is_the_exact_sum_rounded_once():
    """Rounding once at the end, not after every addend, which is what the padded fold buys over an accumulator."""
    rows, token_indices = _rounding_sensitive_rows(seed=7, dtype=torch.bfloat16, device="cpu")
    torch.testing.assert_close(
        fixed_order_sum_rows_by_token(rows, token_indices, UNIT_TOKENS, UNIT_TOP_K),
        _exact_row_sum(rows, token_indices, UNIT_TOKENS),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_one_token_output_does_not_depend_on_the_rest_of_the_batch(dtype, device):
    """A token's rows fold to the same bits whatever else shares the batch, over sixteen compositions.

    Two ways of losing this are worth separating. The slot arithmetic can leak: a wrong group offset puts a
    token's rows in slots that move when a neighbouring group changes width, and that shows up on any device. The
    fold's association can also follow the batch, which only shows up where the reduction's split follows the
    shape it is given -- a host reduction over a handful of addends is sequential, so the CPU parametrization
    cannot see that one and the CUDA parametrization is what covers it.

    An extent taken from the widest group the batch contains, rather than from ``top_k``, is the second kind. It
    survives any number of repeats of one configuration and moves a token's gradient when a packer changes how
    many tokens share a microbatch, so the sweep is over composition rather than over repeats.
    """
    if device == "cuda" and torch.cuda.device_count() < 1:
        pytest.skip("the CUDA parametrization of the batch-composition sweep requires a GPU")

    generator = torch.Generator(device="cpu").manual_seed(31)
    target_rows = (1.0 + torch.rand((SWEEP_TARGET_ROWS, UNIT_HIDDEN), generator=generator)).to(
        device=device, dtype=dtype
    )

    reference = None
    widest_seen = set()
    for num_tokens in SWEEP_TOKEN_COUNTS:
        for neighbour_width in SWEEP_NEIGHBOUR_WIDTHS:
            rows, token_indices = _sweep_case(num_tokens, neighbour_width, target_rows)
            widest_seen.add(int(torch.bincount(token_indices, minlength=num_tokens).max()))
            target = fixed_order_sum_rows_by_token(rows, token_indices, num_tokens, UNIT_TOP_K)[1]
            if reference is None:
                reference = target
                continue
            torch.testing.assert_close(
                target,
                reference,
                rtol=0,
                atol=0,
                msg=f"token 1 moved at {num_tokens} tokens with a neighbour of {neighbour_width} rows",
            )

    assert len(widest_seen) > 1, (
        f"the widest group was {sorted(widest_seen)} in every case, so the sweep cannot tell an extent taken "
        "from the batch apart from an extent taken from top_k"
    )


def test_row_sum_gradient_matches_the_atomic_form():
    """The row sum is on the training forward path, so its backward has to exist and carry the same linear map.

    Both forms send a token's upstream gradient to each of that token's rows, and that scatter of one value to
    several destinations involves no accumulation, so the two gradients are equal exactly rather than closely.
    """
    generator = torch.Generator(device="cpu").manual_seed(13)
    token_indices = torch.arange(UNIT_TOKENS).repeat(UNIT_TOP_K)
    rows = torch.randn(token_indices.numel(), UNIT_HIDDEN, generator=generator)
    upstream = torch.linspace(-1.0, 1.0, UNIT_TOKENS * UNIT_HIDDEN).reshape(UNIT_TOKENS, UNIT_HIDDEN)

    fixed_order_input = rows.detach().clone().requires_grad_(True)
    fixed_order = fixed_order_sum_rows_by_token(fixed_order_input, token_indices, UNIT_TOKENS, UNIT_TOP_K)
    (fixed_order * upstream).sum().backward()

    atomic_input = rows.detach().clone().requires_grad_(True)
    (_atomic_row_sum(atomic_input, token_indices, UNIT_TOKENS) * upstream).sum().backward()

    torch.testing.assert_close(fixed_order_input.grad, atomic_input.grad, rtol=0, atol=0)


def test_row_sum_refuses_a_token_wider_than_top_k():
    """A group wider than the reserved slots would land in the next token's slots, so it has to be rejected."""
    rows = torch.ones((UNIT_TOP_K + 1, UNIT_HIDDEN))
    token_indices = torch.zeros(UNIT_TOP_K + 1, dtype=torch.int64)

    with pytest.raises(AssertionError, match=rf"routed to {UNIT_TOP_K + 1} of this rank's experts"):
        fixed_order_sum_rows_by_token(rows, token_indices, 2, UNIT_TOP_K)

    # One row fewer is the widest group the layout holds, and it must be accepted rather than merely not crash.
    accepted = fixed_order_sum_rows_by_token(rows[:UNIT_TOP_K], token_indices[:UNIT_TOP_K], 2, UNIT_TOP_K)
    torch.testing.assert_close(accepted[0], torch.full((UNIT_HIDDEN,), float(UNIT_TOP_K)), rtol=0, atol=0)
    torch.testing.assert_close(accepted[1], torch.zeros(UNIT_HIDDEN), rtol=0, atol=0)


def test_row_sum_of_no_rows_is_zeros_of_the_output_shape():
    """A rank can receive no routed rows at all, and the combine still owes its caller a full output block."""
    rows = torch.ones((0, UNIT_HIDDEN), dtype=torch.bfloat16)
    empty = fixed_order_sum_rows_by_token(rows, torch.zeros(0, dtype=torch.int64), 5, UNIT_TOP_K)

    assert empty.shape == (5, UNIT_HIDDEN)
    assert empty.dtype is torch.bfloat16
    assert not empty.any()


def test_row_sum_drops_rows_the_index_does_not_name():
    """A padded receive buffer keeps its tail, and those rows belong to no token."""
    token_indices = torch.tensor([0, 0, 1], dtype=torch.int64)
    rows = torch.cat([torch.ones((3, UNIT_HIDDEN)), torch.full((4, UNIT_HIDDEN), 99.0)])

    summed = fixed_order_sum_rows_by_token(rows, token_indices, 2, UNIT_TOP_K)
    torch.testing.assert_close(summed[0], torch.full((UNIT_HIDDEN,), 2.0), rtol=0, atol=0)
    torch.testing.assert_close(summed[1], torch.ones(UNIT_HIDDEN), rtol=0, atol=0)


class _RoutedModel(torch.nn.Module):
    """Stand-in for the model a loader hands over: the fold's extent is read off it rather than passed to it."""

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k


@pytest.fixture
def row_sum_bindings_restored():
    """Put every module's fold back after a case installs one, so an install does not leak into the next case."""
    import sys

    saved = {
        name: module.sum_rows_by_token
        for name, module in list(sys.modules.items())
        if name.startswith("arctic_platform.model.implementations.") and hasattr(module, "sum_rows_by_token")
    }
    yield
    for name, fold in saved.items():
        setattr(sys.modules[name], "sum_rows_by_token", fold)


def test_the_loaders_one_call_is_what_installs_the_fold(row_sum_bindings_restored):
    """A run that did not ask keeps the atomic fold, and a run that asked folds in a fixed order at every site.

    The two folds are told apart by the slot layout rather than by a value. On a host CPU the atomic scatter is
    sequential, so both return the same bits; the fixed-order fold reserves ``top_k`` rows per token and refuses
    a group wider than that, while the atomic form has no layout to overrun and sums the group as it stands.
    """
    from arctic_platform.model.implementations.moe.distributed import token_permute

    rows = torch.ones((UNIT_TOP_K + 1, UNIT_HIDDEN))
    token_indices = torch.zeros(UNIT_TOP_K + 1, dtype=torch.long)
    model = _RoutedModel(UNIT_TOP_K)

    assert maybe_use_fixed_order_row_sum(model, {}) == 0
    assert token_permute.sum_rows_by_token is sum_rows_by_token
    torch.testing.assert_close(
        token_permute.unpermute_tokens(rows, token_indices, 1)[0],
        torch.full((UNIT_HIDDEN,), float(UNIT_TOP_K + 1)),
        rtol=0,
        atol=0,
    )

    assert maybe_use_fixed_order_row_sum(model, {"debug": {"full_determinism": True}}) >= 1
    assert token_permute.sum_rows_by_token is not sum_rows_by_token
    with pytest.raises(AssertionError, match="but top_k is"):
        token_permute.unpermute_tokens(rows, token_indices, 1)


def test_the_installed_fold_is_the_construction_this_file_characterizes(row_sum_bindings_restored):
    """What the install puts at a combine site is the fixed-order fold itself, bit for bit."""
    from arctic_platform.model.implementations.moe.distributed import token_permute

    rows, token_indices = _rounding_sensitive_rows(seed=11, dtype=torch.float32, device="cpu")
    maybe_use_fixed_order_row_sum(_RoutedModel(UNIT_TOP_K), {"debug": {"full_determinism": True}})

    torch.testing.assert_close(
        token_permute.unpermute_tokens(rows, token_indices, UNIT_TOKENS),
        fixed_order_sum_rows_by_token(rows, token_indices, UNIT_TOKENS, UNIT_TOP_K),
        rtol=0,
        atol=0,
    )


def test_the_extent_is_read_off_the_model_and_a_disagreement_is_refused():
    """``top_k`` comes from the model, because every quantity in scope at a combine is a function of the batch."""
    assert routing_width(_RoutedModel(UNIT_TOP_K)) == UNIT_TOP_K

    with pytest.raises(ValueError, match="no module on this model exposes top_k"):
        routing_width(torch.nn.Sequential(torch.nn.Linear(4, 4)))

    with pytest.raises(ValueError, match="different layers"):
        routing_width(torch.nn.Sequential(_RoutedModel(2), _RoutedModel(4)))


def test_a_request_that_reaches_no_fold_is_refused_rather_than_dropped(monkeypatch):
    """An install that patched nothing has to raise, or the run reports a determinism it did not deliver."""
    monkeypatch.setattr(
        "arctic_platform.model.implementations.debug.token_combine.bind_row_sum",
        lambda fold: 0,
    )

    with pytest.raises(RuntimeError, match="no module was found folding rows"):
        maybe_use_fixed_order_row_sum(_RoutedModel(UNIT_TOP_K), {"debug": {"full_determinism": True}})


@pytest.mark.integration
def test_row_sum_returns_the_correctly_rounded_sum_on_every_repeat():
    """Three calls agree bit for bit, and each is the exact sum rounded once, with the global flag untouched."""
    if torch.cuda.device_count() < 1:
        pytest.skip("the row-sum determinism test requires a GPU")
    assert (
        not torch.are_deterministic_algorithms_enabled()
    ), "this test measures determinism that must hold without the global flag, so it cannot run with the flag on"

    rows, token_indices = _rounding_sensitive_rows(seed=7, dtype=torch.bfloat16, device="cuda")
    multiplicity = torch.bincount(token_indices, minlength=UNIT_TOKENS)
    assert int(multiplicity.max()) >= 3, (
        f"two addends commute, so a case whose largest multiplicity is {int(multiplicity.max())} says nothing "
        "about reduction order"
    )

    expected = _exact_row_sum(rows, token_indices, UNIT_TOKENS)
    for repeat in range(REPEATS):
        torch.testing.assert_close(
            fixed_order_sum_rows_by_token(rows, token_indices, UNIT_TOKENS, UNIT_TOP_K),
            expected,
            rtol=0,
            atol=0,
            msg=f"call {repeat} is not the exact sum rounded once",
        )


@pytest.mark.integration
def test_atomic_row_sum_is_the_form_this_replaces_and_does_not_round_once():
    """The comparison above has teeth only if the reduction it replaces fails it on the same input.

    Atomic accumulation rounds at every step, so it does not return the correctly rounded sum. That is a
    deterministic consequence of the implementation and holds on any device, which makes it a check on the
    measurement rather than on the hardware; a device where it passes is a device where the input stopped being
    rounding-sensitive.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("the row-sum determinism test requires a GPU")

    rows, token_indices = _rounding_sensitive_rows(seed=7, dtype=torch.bfloat16, device="cuda")
    expected = _exact_row_sum(rows, token_indices, UNIT_TOKENS)
    atomic = _atomic_row_sum(rows, token_indices, UNIT_TOKENS)

    differing = int((atomic != expected).sum())
    assert differing > 0, (
        "the atomic reduction returned the correctly rounded sum on every element, so this input no longer "
        "distinguishes rounding once from rounding at every addend and the repeat test above proves nothing"
    )


@pytest.mark.integration
def test_row_sum_gradient_matches_the_atomic_form_on_device():
    """The same exact gradient agreement as on the host, where the backward's scatter runs as a CUDA kernel."""
    if torch.cuda.device_count() < 1:
        pytest.skip("the row-sum gradient test requires a GPU")

    generator = torch.Generator(device="cpu").manual_seed(13)
    token_indices = torch.arange(UNIT_TOKENS).repeat(UNIT_TOP_K).cuda()
    rows = torch.randn(token_indices.numel(), UNIT_HIDDEN, generator=generator).cuda()
    upstream = torch.linspace(-1.0, 1.0, UNIT_TOKENS * UNIT_HIDDEN, device="cuda").reshape(UNIT_TOKENS, UNIT_HIDDEN)

    fixed_order_input = rows.detach().clone().requires_grad_(True)
    fixed_order = fixed_order_sum_rows_by_token(fixed_order_input, token_indices, UNIT_TOKENS, UNIT_TOP_K)
    (fixed_order * upstream).sum().backward()

    atomic_input = rows.detach().clone().requires_grad_(True)
    (_atomic_row_sum(atomic_input, token_indices, UNIT_TOKENS) * upstream).sum().backward()

    torch.testing.assert_close(fixed_order_input.grad, atomic_input.grad, rtol=0, atol=0)


def _expert_parallel_repeat_worker(rank: int, world_size: int, port: int) -> None:
    """Build one expert-parallel MoE block, run its forward ``REPEATS`` times, require the bits to agree."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    from arctic_platform.model.implementations.moe.distributed import deepep
    from arctic_platform.model.implementations.moe.distributed.expert_parallel import DeepEPExpertParallel
    from arctic_platform.model.implementations.moe.layers.moe import MoE
    from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        assert (
            not torch.are_deterministic_algorithms_enabled()
        ), "the expert-parallel repeat has to hold without the global determinism flag, which is on"
        device = torch.device("cuda", rank)
        # One seed on every rank keeps the router identical across the expert-parallel group, which is what a
        # replicated router means; the tokens differ per rank because each rank forwards its own shard.
        torch.manual_seed(4242)
        args = MoEArgs(num_experts=EP_NUM_EXPERTS, top_k=EP_TOP_K, num_shared_experts=1, use_grouped_mm=True)
        block = MoE(args, dim=EP_DIM, hidden_dim=EP_HIDDEN).to(device)
        block.init_weights(init_std=0.02, buffer_device=device)
        block = block.to(dtype=torch.bfloat16).eval()

        mesh = DeviceMesh.from_group(dist.group.WORLD, "cuda")
        parallelize_module(block.experts, device_mesh=mesh, parallelize_plan=DeepEPExpertParallel())

        # What a model loader does with the recipe this run would carry, at the point the loader does it.
        installed = maybe_use_fixed_order_row_sum(block, {"debug": {"full_determinism": True}})
        assert installed > 0, "the fixed-order fold was not installed, so the repeat below is not evidence"

        # Largest number of rows the combine folds onto one token, over every call this rank makes. The reduction
        # order can only matter from three addends up, so a run that never reaches three proves nothing and the
        # assertion below rejects it.
        largest_multiplicity = 0
        original_unpermute = deepep.unpermute_tokens

        def recording_unpermute(permuted_hidden_states, permuted_indices, num_tokens):
            nonlocal largest_multiplicity
            counts = torch.bincount(permuted_indices, minlength=num_tokens)
            largest_multiplicity = max(largest_multiplicity, int(counts.max()) if counts.numel() else 0)
            return original_unpermute(permuted_hidden_states, permuted_indices, num_tokens)

        deepep.unpermute_tokens = recording_unpermute
        try:
            hidden = torch.randn(1, EP_TOKENS, EP_DIM, device=device, dtype=torch.bfloat16) * 0.1
            outputs = []
            for _ in range(REPEATS):
                with torch.no_grad():
                    outputs.append(block(hidden))
        finally:
            deepep.unpermute_tokens = original_unpermute

        witness = torch.tensor([largest_multiplicity], device=device, dtype=torch.int64)
        dist.all_reduce(witness, op=dist.ReduceOp.MAX)
        assert int(witness.item()) >= 3, (
            f"no rank folded three or more rows onto a token (largest was {int(witness.item())}), so this "
            "configuration cannot exhibit the ordering effect and the repeat below is not evidence"
        )
        assert torch.isfinite(outputs[0]).all(), "the forward produced non-finite output, so its repeat means nothing"

        for repeat in range(1, REPEATS):
            torch.testing.assert_close(
                outputs[repeat],
                outputs[0],
                rtol=0,
                atol=0,
                msg=f"rank {rank}: forward repeat {repeat} differs from the first",
            )

        # A gradient has to flow through the same reduction, since it sits on the training forward path.
        trainable = hidden.detach().clone().requires_grad_(True)
        block(trainable).sum().backward()
        assert (
            trainable.grad is not None and torch.isfinite(trainable.grad).all()
        ), "the combine-side row sum has to differentiate; the input gradient is missing or non-finite"
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
def test_expert_parallel_moe_forward_repeats_its_bits():
    """Two ranks, eight experts, top_k 4: the production dispatch/combine path, repeated on one input."""
    import torch.multiprocessing as mp

    pytest.importorskip("deep_ep", reason="the expert-parallel MoE forward requires DeepEP")
    if torch.cuda.device_count() < 2:
        pytest.skip("the expert-parallel repeat test requires at least two GPUs")
    mp.spawn(_expert_parallel_repeat_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_every_expert_parallel_backend_folds_through_the_shared_row_sum(row_sum_bindings_restored):
    """Each expert-parallel backend's combine folds a token's rows through the one symbol an install replaces.

    Which backend a job runs is a configuration choice, so a fold that is order-fixed in one of them and atomic
    in another makes determinism a property of that choice. The requirement is checked by putting a recording
    fold in each backend module and calling that backend's own unpermute: a backend that kept an inline scatter
    returns the same value without the recording fold ever being reached, which is the case this rejects.

    A backend whose transport is not installed on this host is skipped rather than asserted over, because
    importing it imports that transport. At least one has to be checked, or the requirement is stated over an
    empty set and guards nothing.
    """
    import importlib

    reached = {}
    for module_name, fold_name in (
        ("arctic_platform.model.implementations.moe.distributed.token_permute", "unpermute_tokens"),
        ("arctic_platform.model.implementations.moe.distributed.ucclep", "_unpermute_tokens"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        calls = []

        def recording(rows, token_indices, num_tokens, calls=calls):
            calls.append(num_tokens)
            return sum_rows_by_token(rows, token_indices, num_tokens)

        module.sum_rows_by_token = recording
        getattr(module, fold_name)(torch.ones((2, 4)), torch.zeros(2, dtype=torch.long), 1)
        reached[module_name] = calls

    assert reached, "no expert-parallel backend module could be imported, so nothing was checked"
    for module_name, calls in sorted(reached.items()):
        assert calls == [1], (
            f"{module_name}'s unpermute does not fold through models.moe.token_combine.sum_rows_by_token, so "
            "the single install a determinism run performs does not reach it. A combine that accumulates with "
            "scatter_add applies one atomic per repeated destination, so a token holding three or more of its "
            "experts on one rank sums them in delivery order and two runs of this backend disagree"
        )
