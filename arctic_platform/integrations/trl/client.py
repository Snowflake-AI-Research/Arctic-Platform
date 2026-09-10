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

"""Arctic backend for TRL's ``TrainingClientProtocol``.

TRL's fused ``forward_backward(..., loss_fn)`` stays in-process: Arctic runs the
model, this adapter unpacks packed rows and evaluates ``loss_fn`` on returned logprobs.
"""

from typing import Any

import torch
from trl.experimental.api import ForwardBackwardOutput

from arctic_platform.integrations.trl.loss import _CLIENT_LOSS_ENCODINGS
from arctic_platform.integrations.trl.loss import _surrogate_payload


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
    zorro_train_enable: bool = False,
) -> dict:
    """``meta`` for ``run_pipeline`` (same keys as verl-GRPO ``base_meta``)."""
    return dict(
        zorro_train_enable=zorro_train_enable,
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
    """Arctic-hosted model, TRL-hosted loss.

    Args:
        client: :class:`~arctic_platform.client.ArcticRLClient`.
        temperature: Applied by the ``apply_temperature`` post-processor.
        loss_fn: Server surrogate name (or dotted import). Not TRL's local ``loss_fn`` callable.
        server_side_loss: When ``True``, ship GRPO ingredients to the server and run
            forward+loss+backward in one fused ``fwd_bwd`` via ``server_loss_fn``
            (perf + parity with verl/SkyRL). When ``False`` (default), keep the
            two-pass client-side surrogate path.
        server_loss_fn: Server GRPO loss name/dotted-path used when ``server_side_loss``.
        client_loss_encoding: How the two-pass path expresses ``dL/dlogprobs`` to the
            server. ``"weighted_logprob_sum"`` (default) names this package's own
            surrogate, which the server must have registered. ``"grpo"`` encodes the
            same quantity in the stock GRPO loss instead, so the path runs on a server
            that ships nothing from this package -- see :func:`_surrogate_payload`.
    """

    def __init__(
        self,
        client: Any,
        temperature: float = 1.0,
        loss_fn: str = "arctic_platform.rl.processors.weighted_logprob.weighted_logprob_sum",
        *,
        pad_token_id: int = 0,
        rollout_n: int = 1,
        max_token_len_per_gpu: int = 4096,
        logits_optimization: str = "none",
        logits_optimization_peak_mem_size_in_gib: int = 4,
        logits_compute_in_fp32: bool = False,
        server_side_loss: bool = False,
        server_loss_fn: str = "arctic_platform.integrations.trl.loss.trl_grpo",
        client_loss_encoding: str = "weighted_logprob_sum",
        zorro_train_enable: bool = False,
        response_len: int | None = None,
        zorro_load_balancer: bool = False,
        grad_accum_steps: int | None = None,
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
        self.server_side_loss = server_side_loss
        self.server_loss_fn = server_loss_fn
        if client_loss_encoding not in _CLIENT_LOSS_ENCODINGS:
            raise ValueError(
                f"client_loss_encoding must be one of {sorted(_CLIENT_LOSS_ENCODINGS)}, got {client_loss_encoding!r}"
            )
        self.client_loss_encoding = client_loss_encoding
        self.zorro_train_enable = zorro_train_enable
        # Padded response width (== configured max_completion_length == server ds_worker_config.response_len).
        # Zorro emits verl-style structured [B, max_prompt_len + response_len] batches, so this must be set.
        self.response_len = response_len
        # Zorro-group load balancer: server reorgs the global batch across DP workers (reorg_global_batch) so
        # same-prompt rollouts land together for better dedup, then restore_batch_order undoes it before the
        # response. EXPERIMENTAL/off by default here: reorg's bin-packer assumes a verl-style global batch that
        # divides evenly into world_size bins with fittable prompt groups; TRL's per-forward_backward microbatch
        # does not guarantee that, so it raises "shouldn't reach here". Opt in only for compatible batch shapes.
        self.zorro_load_balancer = zorro_load_balancer
        # DeepSpeed engine GAS. None → TRL's ``current_gradient_accumulation_steps``.
        self.grad_accum_steps = grad_accum_steps
        if zorro_train_enable:
            if not server_side_loss:
                raise ValueError(
                    "zorro_train_enable requires server_side_loss=True (the zorro path is server-side only; "
                    "the two-pass client-side surrogate is not zorro-shaped)."
                )
            if not response_len or response_len <= 0:
                raise ValueError(
                    "zorro_train_enable requires response_len (== max_completion_length) so the adapter can build "
                    "the fixed-width [B, max_prompt_len + response_len] zorro batch."
                )

    def _meta(self, *, calculate_entropy: bool) -> dict:
        """``meta`` for this client's forwards; see :func:`_meta_dict`."""
        return _meta_dict(
            temperature=self.temperature,
            rollout_n=self.rollout_n,
            pad_token_id=self.pad_token_id,
            max_token_len_per_gpu=self.max_token_len_per_gpu,
            calculate_entropy=calculate_entropy,
            logits_optimization=self.logits_optimization,
            logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
            logits_compute_in_fp32=self.logits_compute_in_fp32,
            zorro_train_enable=self.zorro_train_enable,
        )

    def _grad_accum_meta(self, ing: dict) -> int:
        return self.grad_accum_steps if self.grad_accum_steps is not None else ing["grad_accum_steps"]

    def forward_backward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_fn: Any,
        aux_loss_coef: float = 0.0,
    ) -> ForwardBackwardOutput:
        del model  # trainer-local; Arctic owns the weights
        seq_lens = _segment_lengths(position_ids)

        if self.zorro_train_enable:
            # Zorro needs verl-style structured [B, max_prompt_len + response_len] batches (built from the
            # packed rows + completion_mask), not the plain right-padded [B, S] the other paths use.
            return self._forward_backward_server_loss_zorro(
                loss_fn, input_ids, completion_mask, seq_lens, input_ids.device
            )

        batch = _unpack_to_padded_rows(input_ids, position_ids, completion_mask, seq_lens)

        if self.server_side_loss:
            return self._forward_backward_server_loss(loss_fn, batch, seq_lens, input_ids.device)

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

        # Server [B, S] roll(-1) -> packed TRL [1, T-1].
        device = input_ids.device
        log_probs = _shift_for_trl(_repack_to_row(out["logprobs"], seq_lens)).to(device)
        entropy = _shift_for_trl(_repack_to_row(out["entropy"], seq_lens)).to(device)

        leaf = log_probs.detach().requires_grad_(True)
        loss = loss_fn(leaf)
        (grad_log_probs,) = torch.autograd.grad(loss, leaf)

        def send_backward(grad_loss: torch.Tensor) -> None:
            # Scale, unshift to [1, T], unpack to [B, S].
            weights = _unpack_to_padded(_unshift_from_trl(grad_log_probs * grad_loss), seq_lens)
            back_batch, back_loss_fn, loss_config = _surrogate_payload(
                self.client_loss_encoding, batch, weights, out["logprobs"], self.loss_fn
            )
            processing = {
                "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                "loss_fn": back_loss_fn,
            }
            if loss_config:
                processing["config"] = loss_config
            self.client.fwd_bwd(
                {
                    "batch": back_batch,
                    "meta": self._meta(calculate_entropy=False),
                    "processing": processing,
                }
            )

        reported_loss = loss.detach().requires_grad_(True)
        reported_loss.register_hook(send_backward)

        return ForwardBackwardOutput(
            loss=reported_loss,
            log_probs=log_probs.detach(),
            entropy=entropy,
            aux_loss=None,  # MoE aux is a server metric, not a tensor
        )

    def _forward_backward_server_loss(
        self,
        loss_fn: Any,
        batch: dict,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> ForwardBackwardOutput:
        """Single fused ``fwd_bwd``: server runs forward + GRPO loss + backward.

        Recovers GRPO ingredients from TRL's ``loss_fn`` closure, re-aligns them
        into the server's roll(-1) ``[B, S]`` frame, and hands them to the
        ``server_loss_fn``. The returned loss leaf is a no-op backward (the real
        backward already ran on the server).
        """
        ing = _extract_grpo_ingredients(loss_fn)

        old_lp_bs = _to_server_bs(ing["old_log_probs"], seq_lens)
        adv_bs = _to_server_bs(ing["advantages"], seq_lens)
        mask_bs = _to_server_bs(ing["completion_mask"].to(torch.float32), seq_lens)

        tokens_per_rank = ing["tokens_per_rank"]
        if torch.is_tensor(tokens_per_rank):
            tokens_per_rank = float(tokens_per_rank.item())
        batch_num_tokens = max(1, int(round(tokens_per_rank)))

        meta = {
            **self._meta(calculate_entropy=True),
            "epsilon_low": ing["epsilon_low"],
            "epsilon_high": ing["epsilon_high"],
            "batch_num_tokens": batch_num_tokens,
            "grad_accum_steps": self._grad_accum_meta(ing),
            "return_fwd_batch": True,  # server omits fwd_bwd batch by default
        }

        response = self.client.fwd_bwd(
            {
                "batch": {
                    **batch,
                    "old_log_probs": old_lp_bs,
                    "advantages": adv_bs,
                    "loss_mask": mask_bs,
                },
                "meta": meta,
                "processing": {
                    "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                    "loss_fn": self.server_loss_fn,
                },
            }
        )

        out = response["batch"]
        log_probs = _shift_for_trl(_repack_to_row(out["logprobs"], seq_lens)).to(device)
        entropy = _shift_for_trl(_repack_to_row(out["entropy"], seq_lens)).to(device)

        # Backward already happened server-side; hand TRL a leaf so its
        # ``accelerator.backward(loss)`` is a harmless no-op.
        loss_val = float(response.get("avg_loss", 0.0))
        loss = torch.tensor(loss_val, device=device, dtype=torch.float32, requires_grad=True)

        return ForwardBackwardOutput(
            loss=loss,
            log_probs=log_probs.detach(),
            entropy=entropy,
            aux_loss=None,
        )

    def _forward_backward_server_loss_zorro(
        self,
        loss_fn: Any,
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> ForwardBackwardOutput:
        """Server-side GRPO loss over a ZoRRo-shaped batch (prompt-prefix dedup in the training forward).

        Zorro assumes verl's structured layout: a fixed-width ``[B, max_prompt_len + response_len]`` batch
        with left-padded prompts (so a rollout group's shared prefix is byte-identical for ``find_prompt_groups``)
        and right-padded responses. This builds that batch from TRL's packed rows + ``completion_mask``, ships
        the GRPO ingredients (recovered from ``loss_fn``'s closure) in the response window exactly like verl's
        ``_send_update_actor`` left-pad, and maps the response-aligned logprobs/entropy back to TRL's shifted
        ``[1, T-1]`` packed frame.
        """
        ing = _extract_grpo_ingredients(loss_fn)

        batch, layout = _zorro_structured_batch(
            input_ids, completion_mask, seq_lens, self.pad_token_id, self.response_len
        )
        if self.zorro_load_balancer:
            # Fail fast on ragged groups before the opaque server-side "shouldn't reach here" (see helper).
            _assert_uniform_prompt_groups(batch["prompts"], self.rollout_n)
        # GRPO ingredients: TRL-shifted [1, T-1] -> response window of [B, max_prompt_len + response_len]
        # (matches verl's response-only tensors left-padded into the full sequence).
        old_lp = _place_response_window(ing["old_log_probs"], layout, self.response_len, torch.float32)
        adv = _place_response_window(ing["advantages"], layout, self.response_len, torch.float32)
        mask = _place_response_window(
            ing["completion_mask"].to(torch.float32), layout, self.response_len, torch.float32
        )

        tokens_per_rank = ing["tokens_per_rank"]
        if torch.is_tensor(tokens_per_rank):
            tokens_per_rank = float(tokens_per_rank.item())
        batch_num_tokens = max(1, int(round(tokens_per_rank)))

        meta = {
            **self._meta(calculate_entropy=True),
            "max_prompt_len": layout["max_prompt_len"],
            "max_response_len": self.response_len,
            # reorg same-prompt rollouts across DP workers for better dedup (undone by restore_batch_order).
            "load_balancer": self.zorro_load_balancer,
            "epsilon_low": ing["epsilon_low"],
            "epsilon_high": ing["epsilon_high"],
            "batch_num_tokens": batch_num_tokens,
            "grad_accum_steps": self._grad_accum_meta(ing),
            "return_fwd_batch": True,
        }

        response = self.client.fwd_bwd(
            {
                "batch": {
                    **batch,
                    "old_log_probs": old_lp,
                    "advantages": adv,
                    "loss_mask": mask,
                },
                "meta": meta,
                "processing": {
                    "post": ["apply_temperature", "compute_entropy_and_logprobs"],
                    "loss_fn": self.server_loss_fn,
                },
            }
        )

        out = response["batch"]
        # Zorro returns response-aligned logprobs/entropy scattered into [B, max_prompt_len + response_len];
        # map them back to TRL's shifted [1, T-1] packed frame at the completion indices.
        log_probs = _response_window_to_shifted(out["logprobs"], layout, self.response_len).to(device)
        entropy = _response_window_to_shifted(out["entropy"], layout, self.response_len).to(device)

        loss_val = float(response.get("avg_loss", 0.0))
        loss = torch.tensor(loss_val, device=device, dtype=torch.float32, requires_grad=True)

        return ForwardBackwardOutput(
            loss=loss,
            log_probs=log_probs.detach(),
            entropy=entropy,
            aux_loss=None,
        )


class ArcticOptimizer(torch.optim.Optimizer):
    """Calls ``client.step()``; clip/LR live in the server's ``ds_config``."""

    def __init__(self, client: Any, params: Any, lr: float = 1e-6) -> None:
        super().__init__(params, {"lr": lr})  # scheduler/logging only; real LR is on the server
        self.client = client
        self.last_grad_norm: float | None = None

    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        metrics = self.client.step().get("metrics", {})
        norm = metrics.get("grad_norm")
        self.last_grad_norm = norm[0] if isinstance(norm, list) else norm

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        pass  # server clears grads in step()


# Packed [1, T] <-> padded [B, S]. Client-side path keeps advantages /
# old_log_probs inside TRL's loss_fn; server_side_loss extracts them from
# the closure and ships them on the fused fwd_bwd.


def _segment_lengths(position_ids: torch.Tensor) -> torch.Tensor:
    """Lengths of each packed sequence (gaps between ``position_ids == 0``)."""
    pos = position_ids.reshape(-1)
    total = pos.numel()
    starts = torch.nonzero(pos == 0, as_tuple=False).reshape(-1)
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
    """Packed ``[1, T]`` -> padded ``[B, S]`` server batch."""
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
        "prompts": padded_ids[:, :0],  # completion_mask is applied client-side
    }


def _unpack_to_padded(row: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Packed ``[1, T]`` -> padded ``[B, S]``. Inverse of ``_repack_to_row``."""
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
    """Padded ``[B, S]`` -> packed ``[1, T]``. Inverse of ``_unpack_to_padded``."""
    segments = [padded[i, :length] for i, length in enumerate(seq_lens.tolist())]
    if not segments:
        return padded.reshape(1, 0)
    return torch.cat(segments).reshape(1, -1)


def _shift_for_trl(row: torch.Tensor) -> torch.Tensor:
    """Packed ``[1, T]`` -> ``[1, T-1]`` (drop roll(-1) wraparound). Matches TRL ``old_log_probs[:, 1:]``."""
    return row[:, :-1]


def _unshift_from_trl(row: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_shift_for_trl``: pad a trailing 0 onto the roll(-1) grid."""
    pad = torch.zeros((row.shape[0], 1), dtype=row.dtype, device=row.device)
    return torch.cat([row, pad], dim=1)


def _to_server_bs(shifted: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """TRL-frame ``[1, T-1]`` -> server roll(-1) ``[B, S]``.

    Inverts ``_shift_for_trl(_repack_to_row(...))`` so a shifted per-token TRL
    tensor lands aligned with the server's logprobs frame.
    """
    return _unpack_to_padded(_unshift_from_trl(shifted), seq_lens)


# --- ZoRRo Train: verl-style structured [B, max_prompt_len + response_len] batches ---
#
# Zorro (prompt-prefix dedup in the training forward) assumes verl's dense layout, NOT TRL's varlen packing:
#   * prompts LEFT-padded so a rollout group's shared prefix is byte-identical (find_prompt_groups uses
#     torch.equal on input_ids[:, :max_prompt_len]);
#   * responses RIGHT-padded to a FIXED width == configured max_completion_length (== ds_worker_config.response_len);
#   * attention_mask 0 on both pad regions; position_ids omitted (drop_position_ids reconstructs from the mask).
#
# The response-token index correspondence (both directions) is:
#   server cell [i, max_prompt_len + j]  <->  TRL shifted-packed index  q = start_i + prompt_len_i + j - 1
# for response token j in [0, resp_len_i). This is the same roll(-1)/global-drop frame the non-zorro adapter
# uses (see _shift_for_trl), so it is consistent with TRL's shifted_old_log_probs. Per-row it is a contiguous
# slice starting at q0 = start_i + prompt_len_i - 1. These helpers must NOT be replaced by _repack_to_row /
# _shift_for_trl / _to_server_bs: those assume the plain right-padded [B, S] roll frame, not the response window.


def _assert_uniform_prompt_groups(prompts_2d: torch.Tensor, rollout_n: int) -> None:
    """Fail fast when the load-balancer batch holds ragged prompt groups.

    ZoRRo's ``load_balancer`` (``reorg_global_batch``) packs *atomic* prompt groups into ``world_size`` bins and
    assumes every group has exactly ``rollout_n`` rollouts. TRL's async buffer drops *individual* stale samples
    (``RolloutQueueDataset``), so a fixed-count micro-batch can straddle group boundaries (e.g. sizes ``8,8,7,7,2``),
    which the server bin-packer rejects with an opaque ``"shouldn't reach here"``. Prompts are left-padded, so a
    group's rows are byte-identical; group by row identity and require every group to have ``rollout_n`` members.
    """
    from collections import Counter

    sizes = Counter(tuple(row.tolist()) for row in prompts_2d.detach().cpu())
    if any(n != rollout_n for n in sizes.values()):
        counts = sorted(sizes.values(), reverse=True)
        raise ValueError(
            f"zorro_load_balancer: forward batch has ragged prompt groups (sizes={counts}); every group must have "
            f"exactly rollout_n={rollout_n} rollouts so reorg_global_batch can tile whole groups across DP workers. "
            "This usually means the async rollout buffer dropped stale samples mid-group. Fixes: raise max_staleness "
            "so groups aren't split, keep per_device_bsz a multiple of rollout_n*training_gpus, or disable "
            "zorro_load_balancer."
        )


def _zorro_layout(seq_lens: torch.Tensor, completion_mask: torch.Tensor) -> dict:
    """Per-sequence packed offsets and prompt/response split (completion tokens are the contiguous tail)."""
    cm = completion_mask.reshape(-1)
    starts: list[int] = []
    prompt_lens: list[int] = []
    resp_lens: list[int] = []
    offset = 0
    for length in seq_lens.tolist():
        seg = cm[offset : offset + length]
        r = int(seg.sum().item())
        starts.append(offset)
        resp_lens.append(r)
        prompt_lens.append(length - r)
        offset += length
    return {
        "starts": starts,
        "prompt_lens": prompt_lens,
        "resp_lens": resp_lens,
        "max_prompt_len": max(prompt_lens) if prompt_lens else 0,
        "T": offset,
        "B": len(starts),
    }


def _zorro_structured_batch(
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    pad_token_id: int,
    response_len: int,
) -> tuple[dict, dict]:
    """Packed ``[1, T]`` -> verl-style structured ``[B, max_prompt_len + response_len]`` zorro batch."""
    layout = _zorro_layout(seq_lens, completion_mask)
    ids = input_ids.reshape(-1)
    device = ids.device
    b = layout["B"]
    mp = layout["max_prompt_len"]
    s = mp + response_len

    max_resp = max(layout["resp_lens"]) if layout["resp_lens"] else 0
    if max_resp > response_len:
        raise ValueError(
            f"zorro: a completion has {max_resp} tokens > response_len={response_len}; the fixed-width response "
            "window cannot hold it. Set response_len (max_completion_length) >= the longest completion."
        )

    padded_ids = torch.full((b, s), pad_token_id, dtype=input_ids.dtype, device=device)
    attention_mask = torch.zeros((b, s), dtype=torch.long, device=device)
    response_mask = torch.zeros((b, s), dtype=torch.long, device=device)

    for i in range(b):
        start = layout["starts"][i]
        pl = layout["prompt_lens"][i]
        rl = layout["resp_lens"][i]
        seq = ids[start : start + pl + rl]
        # left-pad prompt into [mp - pl, mp); right-pad response into [mp, mp + rl).
        padded_ids[i, mp - pl : mp] = seq[:pl]
        attention_mask[i, mp - pl : mp] = 1
        padded_ids[i, mp : mp + rl] = seq[pl : pl + rl]
        attention_mask[i, mp : mp + rl] = 1
        response_mask[i, mp : mp + rl] = 1

    batch = {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "prompts": padded_ids[:, :mp],  # width == max_prompt_len (compute_packing_info_for_batch reads .shape[1])
        "response_mask": response_mask,
    }
    return batch, layout


def _place_response_window(shifted: torch.Tensor, layout: dict, response_len: int, dtype: torch.dtype) -> torch.Tensor:
    """TRL-shifted ``[1, T-1]`` -> response window of ``[B, max_prompt_len + response_len]`` (zeros elsewhere)."""
    b = layout["B"]
    mp = layout["max_prompt_len"]
    flat = shifted.reshape(-1)
    out = torch.zeros((b, mp + response_len), dtype=dtype, device=flat.device)
    for i in range(b):
        pl = layout["prompt_lens"][i]
        rl = layout["resp_lens"][i]
        q0 = layout["starts"][i] + pl - 1  # packed index of response token j=0 (predicted from last prompt token)
        out[i, mp : mp + rl] = flat[q0 : q0 + rl].to(dtype)
    return out


def _response_window_to_shifted(full_2d: torch.Tensor, layout: dict, response_len: int) -> torch.Tensor:
    """Response window of ``[B, max_prompt_len + response_len]`` -> TRL-shifted ``[1, T-1]`` (zeros elsewhere)."""
    mp = layout["max_prompt_len"]
    t = layout["T"]
    out = torch.zeros((1, max(t - 1, 0)), dtype=full_2d.dtype, device=full_2d.device)
    flat = out.reshape(-1)
    for i in range(layout["B"]):
        pl = layout["prompt_lens"][i]
        rl = layout["resp_lens"][i]
        q0 = layout["starts"][i] + pl - 1
        flat[q0 : q0 + rl] = full_2d[i, mp : mp + rl]
    return out


# GRPO ingredients that TRL captures in its ``compute_loss.loss_fn`` closure
# (trl/experimental/async_grpo/async_grpo_trainer.py). Recovered by name so the
# server can run the loss instead of the CPU client.
_TRL_LOSS_FREEVARS = ("shifted_old_log_probs", "shifted_advantages", "shifted_completion_mask", "tokens_per_rank")


def _extract_grpo_ingredients(loss_fn: Any) -> dict:
    """Recover GRPO tensors/scalars from TRL's ``loss_fn`` closure.

    TRL only hands the adapter ``forward_backward(..., loss_fn)``; the
    advantages/old_log_probs/mask/epsilons live inside ``loss_fn``'s closure.
    We read them by ``__code__.co_freevars`` + ``__closure__`` and fail loudly
    (pointing at ``loss_placement=client``) if TRL's variable names drift.
    """
    code = getattr(loss_fn, "__code__", None)
    closure = getattr(loss_fn, "__closure__", None)
    if code is None or closure is None:
        raise RuntimeError(
            "server_side_loss requires TRL's closure-based loss_fn; got a plain callable. Use loss_placement=client."
        )
    cells = dict(zip(code.co_freevars, (cell.cell_contents for cell in closure)))

    missing = [name for name in (*_TRL_LOSS_FREEVARS, "self") if name not in cells]
    if missing:
        raise RuntimeError(
            f"server_side_loss could not find TRL loss_fn freevars {missing}; TRL's "
            f"compute_loss may have changed (saw {tuple(cells)}). Use loss_placement=client."
        )

    trainer = cells["self"]
    # Single-process TRL trainer: tokens_per_rank == global completion tokens, so
    # server-side dp_size normalization (in trl_grpo) is the only DP correction.
    world_size = int(getattr(trainer.accelerator, "num_processes", 1))
    if world_size != 1:
        raise RuntimeError(
            "server_side_loss assumes a single-process TRL trainer, but "
            f"accelerator.num_processes={world_size}. Use loss_placement=client."
        )

    return {
        "old_log_probs": cells["shifted_old_log_probs"],
        "advantages": cells["shifted_advantages"],
        "completion_mask": cells["shifted_completion_mask"],
        "tokens_per_rank": cells["tokens_per_rank"],
        "epsilon_low": float(trainer.epsilon_low),
        "epsilon_high": float(trainer.epsilon_high),
        "grad_accum_steps": int(trainer.current_gradient_accumulation_steps),
    }


def _pad_rows(input_ids_rows: list[list[int]], pad_token_id: int) -> tuple[dict, list[int]]:
    """Per-sequence ids -> padded server batch (same contract as ``_unpack_to_padded_rows``)."""
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
    """Training-engine logprobs in the current-token frame (``old[0]=0``). Requires FA2 packing."""
    if not input_ids_rows:
        return []

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
    # roll(-1) -> current-token frame (lead 0).
    out: list[list[float]] = []
    for i, length in enumerate(lens):
        roll = log_probs[i, :length]
        current = torch.cat([roll.new_zeros(1), roll[: length - 1]]) if length else roll
        out.append(current.tolist())
    return out
