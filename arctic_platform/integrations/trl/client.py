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

"""Arctic Platform backend for TRL's ``TrainingClientProtocol``.

TRL's protocol is a single fused call::

    forward_backward(model, input_ids, position_ids, completion_mask, loss_fn, aux_loss_coef) -> ForwardBackwardOutput

``loss_fn`` is a Python callable mapping per-token log probs to a scalar. It closes over everything
algorithm-shaped (advantages, old log probs, the mask, clipping bounds), so nothing algorithm-shaped reaches
this file and nothing here changes when TRL adds a loss variant.

The adapter runs in the trainer's process even though the model does not, so ``loss_fn`` is called here, on
tensors the server returned. Only tensors cross the wire. That is what lets a TRL user write a new objective
as a plain Python function and run it against Arctic on day one, with no change on this side.

Two wire calls, ``fwd_no_grad`` then ``fwd_bwd``, because the per-token weights are a function of the log
probs and cannot be known before the first pass. Whether that costs a second forward is a server-side choice:
retaining the graph between the two calls trades the extra forward for activation memory held across one round
trip. A co-located deployment pays neither, since the graph never leaves the process.
"""

from typing import Any

import torch
from trl.experimental.api import ForwardBackwardOutput


def _meta_dict(
    *,
    temperature: float,
    rollout_n: int,
    pad_token_id: int,
    max_token_len_per_gpu: int,
    calculate_entropy: bool,
    logits_optimization: str = "none",
    logits_optimization_peak_mem_size_in_gib: int = 4,
    logits_compute_in_fp32: bool = False,
) -> dict:
    """The ``meta`` block the server's ``run_pipeline`` requires (mirrors the verl-GRPO ``base_meta``).

    Shared by :class:`ArcticTrainingClient` (new log-probs during the update) and :func:`engine_old_log_probs`
    (old log-probs at rollout time), so both go through the *same* forward contract and therefore the same
    per-token log-prob frame -- the whole point of computing ``old_log_probs`` on the training engine.
    """
    return dict(
        zorro_train_enable=False,
        zorro_train_max_rollouts=rollout_n,
        rollout_n=rollout_n,
        max_prompt_len=0,
        max_response_len=0,
        max_token_len_per_gpu=max_token_len_per_gpu,
        temperature=temperature,
        calculate_entropy=calculate_entropy,
        pad_token_id=pad_token_id,
        drop_position_ids=True,
        logits_optimization=logits_optimization,
        logits_optimization_peak_mem_size_in_gib=logits_optimization_peak_mem_size_in_gib,
        logits_compute_in_fp32=logits_compute_in_fp32,
    )


class ArcticTrainingClient:
    """Runs the model on an Arctic RL server while TRL keeps the loss.

    Configuration lives on this object rather than in TRL's config: the trainer takes the client as a
    constructed instance, so endpoint, temperature and loss name never become TRL config fields.

    Args:
        client: A :class:`~arctic_platform.client.client.SyncArcticRLClient`.
        temperature: Sampling temperature applied by the ``apply_temperature`` post-processor.
        loss_fn: Name of the server-side surrogate. ``pipeline._resolve_fn`` falls back to a dotted-path
            import, so ``"arctic_platform.integrations.trl.loss.weighted_logprob_sum"`` also resolves when the
            registry entry is not present. Unrelated to TRL's ``loss_fn`` argument, which is a callable and
            never leaves this process.
    """

    def __init__(
        self,
        client: Any,
        temperature: float = 1.0,
        loss_fn: str = "arctic_platform.integrations.trl.loss.weighted_logprob_sum",
        *,
        pad_token_id: int = 0,
        rollout_n: int = 1,
        max_token_len_per_gpu: int = 4096,
        logits_optimization: str = "none",
        logits_optimization_peak_mem_size_in_gib: int = 4,
        logits_compute_in_fp32: bool = False,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.loss_fn = loss_fn
        self.pad_token_id = pad_token_id
        self.rollout_n = rollout_n
        self.max_token_len_per_gpu = max_token_len_per_gpu
        self.logits_optimization = logits_optimization
        self.logits_optimization_peak_mem_size_in_gib = logits_optimization_peak_mem_size_in_gib
        self.logits_compute_in_fp32 = logits_compute_in_fp32

    def _meta(self, *, calculate_entropy: bool) -> dict:
        """This client's ``meta`` block; see :func:`_meta_dict`.

        The whole dict is splatted as ``**meta`` into the model forward and read by the ``apply_temperature`` /
        ``compute_entropy_and_logprobs`` post-processors, so every key the proven verl path sends is present.
        """
        return _meta_dict(
            temperature=self.temperature,
            rollout_n=self.rollout_n,
            pad_token_id=self.pad_token_id,
            max_token_len_per_gpu=self.max_token_len_per_gpu,
            calculate_entropy=calculate_entropy,
            logits_optimization=self.logits_optimization,
            logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
            logits_compute_in_fp32=self.logits_compute_in_fp32,
        )

    def forward_backward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_fn: Any,
        aux_loss_coef: float = 0.0,
    ) -> ForwardBackwardOutput:
        # `model` is the trainer's local module. Arctic owns the weights, so it is unused here.
        #
        # AsyncGRPOTrainer packs a rank's samples into one padding-free row and marks sequence boundaries by
        # position_ids resets. The server wants padded [B, S] rows, so unpack first. This conversion is the
        # bulk of the adapter and is the main reason it belongs on this side of the wire.
        seq_lens = _segment_lengths(position_ids)
        batch = _unpack_to_padded_rows(input_ids, position_ids, completion_mask, seq_lens)

        # The on-prem server unwraps the body as `{"batch", "meta", "processing"}` (see `unpack_batch`), so both
        # calls wrap the padded tensors in that envelope. `apply_temperature` scales the logits and
        # `compute_entropy_and_logprobs` turns them into per-token log probs (roll(-1) convention).
        response = self.client.fwd_no_grad(
            {
                "batch": batch,
                "meta": self._meta(calculate_entropy=True),
                "processing": {
                    "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                    "loss_fn": None,
                },
            }
        )
        out = response["batch"]

        # The server returns per-row log probs [B, S] under the roll(-1) convention: entry j holds
        # log p(token_{j+1} | <=j), i.e. the log prob of the token predicted at position j. Repack the valid
        # (unpadded) entries into the packed [1, T] row, then drop the trailing roll-around slot (`_shift_for_trl`)
        # so entry k lines up with `shifted_old_log_probs[k]` / `shifted_completion_mask[k]` (the token at k+1).
        # The server returns log probs / entropy as CPU tensors; TRL's loss closure holds its tensors
        # (`shifted_old_log_probs`, advantages, ...) on the trainer's device, so realign before `loss_fn`.
        device = input_ids.device
        log_probs = _shift_for_trl(_repack_to_row(out["logprobs"], seq_lens)).to(device)
        entropy = _shift_for_trl(_repack_to_row(out["entropy"], seq_lens)).to(device)

        # Evaluate the trainer's loss here, locally, on a leaf. This is the whole trick: the loss is a Python
        # callable in this process, so it never has to exist on the server or be named on the wire.
        leaf = log_probs.detach().requires_grad_(True)
        loss = loss_fn(leaf)
        (grad_log_probs,) = torch.autograd.grad(loss, leaf)

        def send_backward(grad_loss: torch.Tensor) -> None:
            # Fires from the trainer's `accelerator.backward`. `grad_loss` carries whatever scaling that
            # backward applies, so folding it in here keeps gradient accumulation correct.
            #
            # grad_log_probs is [1, T-1] in TRL's shifted frame. Undo the shift back to the packed [1, T]
            # frame, then unpack to padded [B, S] so it lines up with the server's roll(-1) log-prob grid.
            weights = _unpack_to_padded(_unshift_from_trl(grad_log_probs * grad_loss), seq_lens)
            # `_shifted` marks the roll(-1) convention, matching `model_outputs["logprobs"]`. The loss surrogate
            # `weighted_logprob_sum` reads `logprobs`, so `compute_entropy_and_logprobs` must run in `post`
            # (before the loss) to populate it -- `apply_temperature` alone would leave it unset.
            self.client.fwd_bwd(
                {
                    "batch": {**batch, "logprob_weights_shifted": weights},
                    "meta": self._meta(calculate_entropy=False),
                    "processing": {
                        "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                        "loss_fn": self.loss_fn,
                    },
                }
            )

        # Detached, because nothing on this side is connected to the model. The hook is what reaches it.
        reported_loss = loss.detach().requires_grad_(True)
        reported_loss.register_hook(send_backward)

        return ForwardBackwardOutput(
            loss=reported_loss,
            log_probs=log_probs.detach(),
            entropy=entropy,
            aux_loss=None,  # Arctic reports the MoE aux loss as a metric, not as a differentiable tensor
        )


class ArcticOptimizer(torch.optim.Optimizer):
    """Drives ``client.step()``. Passed to the trainer as ``optimizers=(ArcticOptimizer(...), scheduler)``.

    TRL needs nothing for this. ``transformers.Trainer`` already accepts a user-supplied optimizer and calls
    ``.step()`` on it, so the optimizer leg needs no protocol of its own and gradient clipping stays inside
    ``step()`` where Arctic already does it.
    """

    def __init__(self, client: Any, params: Any, lr: float = 1e-6) -> None:
        # `lr` is cosmetic here -- the real learning rate lives in the server's `ds_config` optimizer -- but a
        # param-group `lr` is required so `transformers`' LR scheduler (and any logging) can read it.
        super().__init__(params, {"lr": lr})
        self.client = client
        self.last_grad_norm: float | None = None

    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        metrics = self.client.step().get("metrics", {})
        norm = metrics.get("grad_norm")
        # grad_norm comes back per DP rank as a flat list. Under ZeRO-3 the value is already globally
        # reduced, so the first entry is the value.
        self.last_grad_norm = norm[0] if isinstance(norm, list) else norm

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        pass  # gradients live on the server and are cleared by step()


# ------------------------------------------------------------------------------------------------------- #
# Batch layout helpers
# ------------------------------------------------------------------------------------------------------- #
# Padding-free row <-> padded rows. Mechanical but fiddly, and where a first integration spends its time.
# Note what is absent compared with `integrations/verl/adapter.py`: no `old_log_probs`, no `advantages`, no
# `ref_log_prob`. TRL's `loss_fn` consumed those before the gradient was formed, so they never reach the wire.
#
# AsyncGRPOTrainer hands the client one padding-free row per rank: `input_ids`/`position_ids`/`completion_mask`
# of shape `[1, T]` with N sequences concatenated end to end. `position_ids` restarts at 0 at each sequence
# boundary (`DataCollatorForRollout`), which is the only signal for where one sequence ends and the next
# begins. The server instead wants dense right-padded `[B, S]` rows (B = N sequences, S = longest sequence).


def _segment_lengths(position_ids: torch.Tensor) -> torch.Tensor:
    """Per-sequence real-token lengths from a packed row.

    `position_ids` is `[1, T]` and restarts at 0 at each sequence start, so the starts are the zero
    positions and the lengths are the gaps between consecutive starts (last one runs to `T`).
    """
    pos = position_ids.reshape(-1)
    total = pos.numel()
    starts = torch.nonzero(pos == 0, as_tuple=False).reshape(-1)
    # A well-formed packed row always starts a sequence at index 0.
    if starts.numel() == 0 or starts[0].item() != 0:
        starts = torch.cat([torch.zeros(1, dtype=starts.dtype, device=starts.device), starts])
    ends = torch.cat([starts[1:], torch.tensor([total], dtype=starts.dtype, device=starts.device)])
    return (ends - starts).to(torch.long)


def _unpack_to_padded_rows(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    seq_lens: torch.Tensor,
) -> dict:
    """Packed `[1, T]` row -> dense right-padded `[B, S]` server batch.

    Returns the minimal contract the server needs to run a forward and produce per-token log probs:
    `input_ids`, `attention_mask` (1 real / 0 pad), and `position_ids` (per-row `arange`, 0 on pad).
    """
    ids = input_ids.reshape(-1)
    device = ids.device
    b = int(seq_lens.numel())
    s = int(seq_lens.max().item()) if b else 0

    padded_ids = torch.zeros((b, s), dtype=input_ids.dtype, device=device)
    attention_mask = torch.zeros((b, s), dtype=torch.long, device=device)
    padded_pos = torch.zeros((b, s), dtype=torch.long, device=device)

    offset = 0
    for i, length in enumerate(seq_lens.tolist()):
        padded_ids[i, :length] = ids[offset : offset + length]
        attention_mask[i, :length] = 1
        padded_pos[i, :length] = torch.arange(length, device=device)
        offset += length

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "position_ids": padded_pos,
        # The server's `compute_packing_info_for_batch` derives `response_lens` from
        # `attention_mask[:, prompts.shape[1]:]`. A TRL packed row is a single sequence with a per-sequence
        # completion boundary (no uniform prompt width), and the client masks prompt tokens itself via
        # `completion_mask`, so a zero-width `prompts` (whole row counts as response) is the right mapping.
        "prompts": padded_ids[:, :0],
    }


def _unpack_to_padded(row: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Packed `[1, T]` (or `[T]`) row -> dense right-padded `[B, S]`, zeros on pad. Inverse of `_repack_to_row`."""
    flat = row.reshape(-1)
    device = flat.device
    b = int(seq_lens.numel())
    s = int(seq_lens.max().item()) if b else 0

    padded = torch.zeros((b, s), dtype=flat.dtype, device=device)
    offset = 0
    for i, length in enumerate(seq_lens.tolist()):
        padded[i, :length] = flat[offset : offset + length]
        offset += length
    return padded


def _repack_to_row(padded: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Dense `[B, S]` -> packed `[1, T]` by concatenating each row's real (unpadded) entries. Inverse of
    `_unpack_to_padded`."""
    segments = [padded[i, :length] for i, length in enumerate(seq_lens.tolist())]
    if not segments:
        return padded.reshape(1, 0)
    return torch.cat(segments).reshape(1, -1)


def _shift_for_trl(row: torch.Tensor) -> torch.Tensor:
    """Packed `[1, T]` -> `[1, T-1]`, dropping the **last** packed position.

    The server already computes log probs with roll(-1) labels (``labels = torch.roll(input_ids, -1)``), so entry
    ``j`` is ``log p(token at j+1)`` -- i.e. the log prob of the token *predicted at* position ``j``. TRL's loss
    consumes ``log_probs[k]`` alongside ``old_log_probs[:, 1:][k]`` and ``completion_mask[:, 1:][k]``, all of which
    describe the token at packed position ``k+1``. Since ``engine[k]`` is exactly that (log prob of token ``k+1``),
    the correct alignment is ``engine[:, :-1]`` (drop the trailing roll-around slot), **not** ``engine[:, 1:]``:
    the latter would return ``engine[k+1]`` (log prob of token ``k+2``), a one-token overshoot that de-aligns the
    ratio from ``old_log_probs`` (and from the ``completion_mask``). The dropped trailing slot is the per-row
    roll-around (label wraps to pad), which is never a valid completion position."""
    return row[:, :-1]


def _unshift_from_trl(row: torch.Tensor) -> torch.Tensor:
    """Inverse of `_shift_for_trl`: `[1, T-1]` -> `[1, T]`, restoring the dropped **trailing** position as 0 so the
    gradient weights realign with the packed/roll(-1) grid (entry ``k`` -> ``engine[k]``) before `_unpack_to_padded`."""
    pad = torch.zeros((row.shape[0], 1), dtype=row.dtype, device=row.device)
    return torch.cat([row, pad], dim=1)


# ------------------------------------------------------------------------------------------------------- #
# old_log_probs from the training engine (verl-style)
# ------------------------------------------------------------------------------------------------------- #
# verl computes ``old_log_prob`` by running ``compute_log_prob`` on the *actor* (the training engine) over the
# rollout, not by trusting the sampler's logprobs. The importance ratio then compares two log-probs produced by
# the *same* subsystem, so an on-policy step gives ``ratio ~ 1`` regardless of any absolute convention. The TRL
# adapter mirrors that here: the rollout worker calls this at generation time to fill ``RolloutSample.old_log_probs``
# from the Arctic training engine instead of vLLM, so it shares the exact forward path (``fwd_no_grad`` +
# ``apply_temperature`` + ``compute_entropy_and_logprobs``) that :meth:`ArcticTrainingClient.forward_backward`
# uses for the *new* log-probs.


def _pad_rows(input_ids_rows: list[list[int]], pad_token_id: int) -> tuple[dict, list[int]]:
    """Variable-length token rows -> dense right-padded `[B, S]` server batch (same contract as
    :func:`_unpack_to_padded_rows`, but built from per-sequence id lists rather than a packed row)."""
    b = len(input_ids_rows)
    lens = [len(ids) for ids in input_ids_rows]
    s = max(lens) if b else 0

    padded_ids = torch.full((b, s), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((b, s), dtype=torch.long)
    position_ids = torch.zeros((b, s), dtype=torch.long)
    for i, ids in enumerate(input_ids_rows):
        length = lens[i]
        padded_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1
        position_ids[i, :length] = torch.arange(length)

    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        # Zero-width `prompts`: the whole row counts as response (the client masks prompt tokens itself), matching
        # `_unpack_to_padded_rows`.
        "prompts": padded_ids[:, :0],
    }, lens


def engine_old_log_probs(
    client: Any,
    input_ids_rows: list[list[int]],
    *,
    temperature: float,
    pad_token_id: int,
    rollout_n: int,
    max_token_len_per_gpu: int,
    logits_optimization: str = "none",
    logits_optimization_peak_mem_size_in_gib: int = 4,
    logits_compute_in_fp32: bool = False,
) -> list[list[float]]:
    """Per-sequence policy log-probs from the Arctic *training* engine (verl-style ``old_log_prob``).

    Runs one forward-only (`fwd_no_grad`, `torch.no_grad`) pass over the padded rollout sequences on the training
    job -- the NCCL root that holds the live policy weights -- through the same post-processors the trainer uses
    for its *new* log-probs. The server emits roll(-1) log probs (entry ``j`` = ``log p(token at j+1)``); this
    returns them in the **current-token frame** ``old[p] = log p(token at position p)`` (``old[0] = 0``, since the
    first token is unconditioned and always a masked prompt token). That is exactly the frame vLLM's sampled
    logprobs use and the frame TRL expects for ``old_log_probs`` (it applies ``[:, 1:]`` itself). Sourcing ``old``
    here rather than from vLLM makes ``old`` and the trainer's ``new`` come from the same subsystem, so the
    on-policy ``ratio`` sits near 1 (the small residual is genuine staleness from weight updates between rollout
    and consumption, not a sampler/engine gap).

    Returns one ``list[float]`` per input row, each the same length as its ``input_ids``.
    """
    if not input_ids_rows:
        return []

    # NOTE: this submits all rollout sequences as one multi-row [B, S] batch, which the server packs varlen-style
    # (concatenate to [1, T] with per-sequence position_ids resets) exactly like the trainer's `new` log-prob pass
    # in `forward_backward`. Sequence separation therefore requires a varlen-capable attention backend
    # (`attn_implementation="flash_attention_2"`), where HF derives block-diagonal `cu_seqlens` from the reset
    # position_ids. Under `sdpa` the packed rows attend across sequence boundaries and corrupt per-token log probs
    # for every sequence after the first (verified: batched vs per-row diverges by tens of nats under sdpa, but is
    # bit-for-bit identical under flash_attention_2). `old` (here) and `new` (forward_backward) share this path, so
    # both are correct together under FA2 and the on-policy ratio sits at ~1.
    batch, lens = _pad_rows(input_ids_rows, pad_token_id)
    response = client.fwd_no_grad(
        {
            "batch": batch,
            "meta": _meta_dict(
                temperature=temperature,
                rollout_n=rollout_n,
                pad_token_id=pad_token_id,
                max_token_len_per_gpu=max_token_len_per_gpu,
                calculate_entropy=False,
                logits_optimization=logits_optimization,
                logits_optimization_peak_mem_size_in_gib=logits_optimization_peak_mem_size_in_gib,
                logits_compute_in_fp32=logits_compute_in_fp32,
            ),
            "processing": {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": None},
        }
    )
    log_probs = response["batch"]["logprobs"]
    if not torch.is_tensor(log_probs):
        log_probs = torch.as_tensor(log_probs)
    log_probs = log_probs.detach().to("cpu", dtype=torch.float32)
    # roll(-1) (entry j = token j+1) -> current-token (entry p = token p): shift right by one, lead with 0.
    out: list[list[float]] = []
    for i, length in enumerate(lens):
        roll = log_probs[i, :length]
        current = torch.cat([roll.new_zeros(1), roll[: length - 1]]) if length else roll
        out.append(current.tolist())
    return out
