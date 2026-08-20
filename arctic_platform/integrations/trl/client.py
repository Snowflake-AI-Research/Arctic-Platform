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
    """``meta`` for ``run_pipeline`` (same keys as verl-GRPO ``base_meta``)."""
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
    """Arctic-hosted model, TRL-hosted loss.

    Args:
        client: :class:`~arctic_platform.client.client.SyncArcticRLClient`.
        temperature: Applied by the ``apply_temperature`` post-processor.
        loss_fn: Server surrogate name (or dotted import). Not TRL's local ``loss_fn`` callable.
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
        del model  # trainer-local; Arctic owns the weights
        seq_lens = _segment_lengths(position_ids)
        batch = _unpack_to_padded_rows(input_ids, position_ids, completion_mask, seq_lens)

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

        reported_loss = loss.detach().requires_grad_(True)
        reported_loss.register_hook(send_backward)

        return ForwardBackwardOutput(
            loss=reported_loss,
            log_probs=log_probs.detach(),
            entropy=entropy,
            aux_loss=None,  # MoE aux is a server metric, not a tensor
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


# Packed [1, T] <-> padded [B, S]. Advantages / old_log_probs stay in TRL.


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
