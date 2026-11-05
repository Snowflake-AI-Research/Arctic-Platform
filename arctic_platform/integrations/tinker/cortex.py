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
"""Run a Tinker forward-backward, optimizer step, or sample on a Cortex job.

This module lowers loss names, aligns padded rows for Cortex, and restores
returned log-probs to the datum layout from :mod:`arctic_platform.integrations.tinker.convert`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

if TYPE_CHECKING:
    import torch

    from arctic_platform.client import AsyncArcticRLClient

__all__ = ["CortexTinkerBackend", "build_handlers"]

# Cortex registers only `identity` and `compute_logprobs`. convert.py's default
# (`compute_entropy_and_logprobs`) does not exist there, and the zone refuses
# the request before any model call.
_POST_PROCESSORS = ["compute_logprobs"]

# The loss convert.py names for Tinker's ``cross_entropy``, and the batch key
# carrying its per-token weights.
_WEIGHTED_LOGPROB_SUM = "weighted_logprob_sum"
_LOGPROB_WEIGHTS = "logprob_weights_shifted"

_SAMPLER_LOGPROBS = "old_log_probs_shifted"


def _clip_config(ratio_clip: tuple[float, float]) -> dict[str, float]:
    """Lower Tinker's ``(low, high)`` ratio bounds onto ``grpo``'s clip range.

    ``grpo`` always clamps the ratio to ``[1 - eps_clip, 1 + eps_clip_higher]``
    and its config travels as JSON, which has no infinity, so an unbounded side
    becomes float32's max: no finite ratio reaches it.
    """
    import torch

    low, high = ratio_clip
    return {
        "eps_clip": 1.0 - low,
        "eps_clip_higher": high - 1.0 if math.isfinite(high) else float(torch.finfo(torch.float32).max),
    }


def _grpo_surrogate(body: dict, meta: dict) -> tuple[dict, dict]:
    """Encode Tinker's weighted log-prob gradient with stock ``grpo``.

    Token-mean divides by ``batch_num_tokens``. The server writes that same
    count into the loss context and rejects a config value that differs, so
    the count stays the real mask sum. Scaling the advantages by it cancels
    the division and leaves Tinker's unnormalized sum, ``dL/dlogprobs = -weights``.
    """
    if _LOGPROB_WEIGHTS not in body:
        raise ValueError(
            f"loss_fn={_WEIGHTED_LOGPROB_SUM!r} needs {_LOGPROB_WEIGHTS!r} in the "
            f"batch to encode as grpo; got keys {sorted(body)}. Without it the "
            "advantages would carry no signal and the step would be a no-op."
        )
    weights = body.pop(_LOGPROB_WEIGHTS)
    token_count = meta.get("batch_num_tokens")
    if not token_count:
        raise ValueError("weighted cross-entropy needs a positive meta batch_num_tokens")
    return {**body, "advantages": -weights * float(token_count)}, meta


def _pad_rows(body: dict, min_rows: int) -> dict:
    """Append copies of row 0's tokens until the batch has ``min_rows`` rows.

    Cortex shards rows across its training ranks and refuses a batch with fewer
    rows than ranks, which RL hits once most groups have equal rewards and are
    dropped. Only the model inputs are copied; every other tensor is zero on the
    added rows, so they carry no loss and leave ``batch_num_tokens`` unchanged.
    """
    import torch

    rows = body["attention_mask"].shape[0]
    if rows >= min_rows:
        return body
    extra = min_rows - rows

    def pad(name: str, t: Any) -> Any:
        if not torch.is_tensor(t) or t.dim() == 0 or t.shape[0] != rows:
            return t
        filler = t[:1].expand(extra, *t.shape[1:])
        if name not in ("input_ids", "attention_mask", "position_ids"):
            filler = torch.full_like(filler, -100 if name == "labels" else 0)
        return torch.cat([t, filler])

    return {k: pad(k, v) for k, v in body.items()}


def _align_plan(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(order, valid)`` moving each row's real tokens to its leading columns.

    Returned rather than applied so the caller can invert it. ``order`` is a
    full permutation of the width, which makes the inverse an exact scatter.
    """
    import torch

    mask = attention_mask.to(torch.bool)
    width = mask.shape[-1]
    lengths = mask.sum(dim=1)
    valid = torch.arange(width, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
    return order, valid


def _align(batch: dict, order: torch.Tensor, valid: torch.Tensor) -> dict:
    """Gather every full-width 2-D tensor through ``order``.

    One index for all of them, so ``advantages`` and ``response_mask`` stay on
    the tokens they scored.
    """
    import torch

    width = valid.shape[-1]
    pad_for = {"labels": -100}

    def move(name: str, t: Any) -> Any:
        if not torch.is_tensor(t) or t.dim() != 2 or t.shape[-1] != width:
            return t
        pad = pad_for.get(name, False if t.dtype == torch.bool else 0)
        return torch.where(valid, t.gather(1, order), torch.full_like(t, pad))

    out = {k: move(k, v) for k, v in batch.items()}
    out["attention_mask"] = valid.to(batch["attention_mask"].dtype)
    return out


def _unalign_rows(aligned: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Invert :func:`_align_plan` for one ``[B, width]`` tensor.

    ``aligned[i, j] == original[i, order[i, j]]``, so a scatter along ``order``
    is the exact inverse.
    """
    import torch

    if aligned.dim() != 2 or aligned.shape != order.shape:
        raise ValueError(
            f"cannot un-align log-probs of shape {tuple(aligned.shape)} against an "
            f"alignment plan of shape {tuple(order.shape)}; Cortex returned a "
            "frame that does not match the batch that was sent"
        )
    out = torch.zeros_like(aligned)
    return out.scatter(1, order, aligned)


def _require_logprobs(response: dict, op: str) -> torch.Tensor:
    """Return aligned per-token log-probs or fail if the response omitted them."""
    import torch

    logprobs = None
    if isinstance(response, dict):
        for container in (response.get("post_process_outputs"), response.get("batch"), response):
            if isinstance(container, dict) and container.get("logprobs") is not None:
                logprobs = container["logprobs"]
                break

    if logprobs is None:
        raise RuntimeError(
            f"cortex {op} returned no per-token log-probs. Requested post-processors: "
            f"{_POST_PROCESSORS}; response keys: "
            f"{sorted(response) if isinstance(response, dict) else type(response).__name__}. "
            "Tinker's forward_backward contract requires them, so this cannot be "
            "defaulted -- they feed the sampler-vs-trainer KL check."
        )

    if not torch.is_tensor(logprobs):
        logprobs = torch.as_tensor(logprobs, dtype=torch.float32)
    return logprobs.to(torch.float32)


def _sampled_logprobs(result: dict) -> list[float] | None:
    """Per-position log-prob of the token actually sampled.

    Each position is a dict keyed by token id -- the sampled token plus any
    top-k extras -- so it has to be looked up by id, never positionally. These
    become ``old_log_probs``, where a misaligned list would bias the importance
    ratio without looking wrong, so a gap raises.
    """
    per_position = result.get("logprobs")
    if not per_position:
        return None
    token_ids = list(result.get("token_ids") or [])
    if len(per_position) != len(token_ids):
        raise RuntimeError(
            f"cortex returned {len(per_position)} log-prob positions for "
            f"{len(token_ids)} sampled tokens; these become old_log_probs, so a "
            "mismatched pairing would bias the importance ratio"
        )

    out: list[float] = []
    for position, token_id in zip(per_position, token_ids):
        entry = position.get(str(token_id), position.get(token_id))
        if entry is None:
            raise RuntimeError(
                f"cortex omitted the log-prob of sampled token {token_id}; ask for "
                "`logprobs` in sampling_params so the sampled token is always included"
            )
        out.append(float(entry["logprob"] if isinstance(entry, dict) else entry))
    return out


def _prompt_logprobs(result: dict, prompt_tokens: list[int]) -> list[float | None]:
    """Per-position log-prob of each prompt token, ``None`` for the first.

    Same layout as the sampled ones -- a dict per position keyed by token id --
    and the same rule: on-policy distillation subtracts these from the
    student's log-probs position by position, so a gap raises.
    """
    per_position = result.get("prompt_logprobs")
    if per_position is None or len(per_position) != len(prompt_tokens):
        raise RuntimeError(
            f"cortex returned {None if per_position is None else len(per_position)} prompt log-prob "
            f"positions for {len(prompt_tokens)} prompt tokens"
        )

    out: list[float | None] = []
    for index, (position, token_id) in enumerate(zip(per_position, prompt_tokens)):
        if position is None:
            if index != 0:
                raise RuntimeError(f"cortex omitted the prompt log-prob at position {index}")
            out.append(None)
            continue
        entry = position.get(str(token_id), position.get(token_id))
        if entry is None:
            raise RuntimeError(f"cortex omitted the log-prob of prompt token {token_id} at position {index}")
        out.append(float(entry["logprob"] if isinstance(entry, dict) else entry))
    return out


def _extend_rows(aligned: dict, min_len: int) -> dict:
    """Lengthen every left-aligned row to at least ``min_len`` attended tokens.

    The filler follows the row's last real token, so a causal model scores the
    real tokens exactly as before; every loss input is already zero there.
    """
    import torch

    mask = aligned["attention_mask"]
    if mask.shape[-1] < min_len:
        raise ValueError(f"cannot extend rows of width {mask.shape[-1]} to {min_len} tokens")
    filler = torch.arange(mask.shape[-1], device=mask.device).unsqueeze(0) < min_len
    filler = filler & ~mask.to(torch.bool)
    out = dict(aligned)
    out["attention_mask"] = torch.where(filler, torch.ones_like(mask), mask)
    out["input_ids"] = torch.where(filler, aligned["input_ids"][:, :1].expand_as(mask), aligned["input_ids"])
    return out


class CortexTinkerBackend:
    """Tinker's verbs, lowered onto a Cortex-backed unified client.

    ``isolate_capacity`` keeps Cortex from packing two sequences into one
    micro-batch. Cortex's Hugging Face model path does not reset
    linear-attention state (Qwen3.5's GatedDeltaNet) at packed boundaries, so
    every sequence after the first in a pack is scored with its predecessor's
    state. The job is provisioned with ``max_tokens_per_mb = isolate_capacity``
    and every row is lengthened past half of it, so no two rows fit together.
    """

    def __init__(self, client: AsyncArcticRLClient, min_rows: int = 1, isolate_capacity: int | None = None) -> None:
        self.client = client
        self.min_rows = min_rows
        self.isolate_capacity = isolate_capacity

    async def fwd_bwd(self, batch: dict) -> dict:
        import torch

        body = dict(batch.get("batch") or {})
        meta = dict(batch.get("meta") or {})
        attention_mask = body.get("attention_mask")
        if not torch.is_tensor(attention_mask):
            body = {k: torch.as_tensor(v) if not torch.is_tensor(v) else v for k, v in body.items()}
            attention_mask = body.get("attention_mask")
        if attention_mask is None:
            raise ValueError("tinker fwd_bwd batch is missing 'attention_mask'")

        processing = dict(batch.get("processing") or {})
        if processing.get("loss_fn") == _WEIGHTED_LOGPROB_SUM:
            body, meta = _grpo_surrogate(body, meta)
        else:
            body.pop(_LOGPROB_WEIGHTS, None)

        # Tinker's ratio is against the sampler's log-probs. Without them Cortex
        # uses the trainer's own, so the ratio is 1 and any sampler/trainer
        # mismatch goes uncorrected.
        ratio_clip = processing.pop("ratio_clip", None)
        if ratio_clip is not None:
            if _SAMPLER_LOGPROBS not in body:
                raise ValueError(f"a ratio loss needs the sampler's log-probs as {_SAMPLER_LOGPROBS!r}")
            processing["config"] = {**(processing.get("config") or {}), **_clip_config(ratio_clip)}

        rows = attention_mask.shape[0]
        body = _pad_rows(body, self.min_rows)
        order, valid = _align_plan(body["attention_mask"])
        aligned = _align(body, order, valid)
        if self.isolate_capacity is not None:
            aligned = _extend_rows(aligned, self.isolate_capacity // 2 + 1)
        sampler_logprobs = aligned.pop(_SAMPLER_LOGPROBS, None)
        payload = to_cortex_fwd_bwd_payload(
            {"batch": aligned, "meta": meta},
            processing=processing,
            old_log_probs_shifted=sampler_logprobs if ratio_clip is not None else None,
        )
        response = await self.client.fwd_bwd(payload)
        logprobs = _unalign_rows(_require_logprobs(response, "forward-backward"), order)[:rows]
        return {"batch": {"logprobs": logprobs}, "metrics": response.get("metrics") or {}}

    async def step(self, overrides: dict | None) -> dict:
        # Cortex step takes a learning rate. Other Adam settings stay as provisioned.
        learning_rate = (overrides or {}).get("lr") or (overrides or {}).get("learning_rate")
        return await self.client.step(learning_rate=learning_rate)

    async def sync_weights(self) -> Any:
        # A LoRA run's sampler holds the base weights plus an adapter, so only
        # the adapter is broadcast.
        if self.client.config.training.peft:
            return await self.client.sync_weights(weight_format="lora")
        return await self.client.sync_weights()

    async def generate(self, prompt_tokens: list[int], sampling_params: dict) -> dict:
        """``num_samples`` rollouts of one prompt.

        Cortex takes no ``n`` and returns exactly one completion per prompt, so
        N samples means sending the prompt N times.
        """
        params = dict(sampling_params)
        num_samples = max(int(params.pop("n", 1) or 1), 1)
        results = await self.client.generate([list(prompt_tokens)] * num_samples, sampling_params=params)
        if len(results) != num_samples:
            raise RuntimeError(
                f"asked cortex for {num_samples} rollouts and got {len(results)}; "
                "sampling would silently return the wrong group size"
            )
        out: dict[str, Any] = {
            "outputs": [
                {
                    "token_ids": list(result.get("token_ids") or []),
                    "logprobs": _sampled_logprobs(result),
                    "finish_reason": result.get("finish_reason"),
                }
                for result in results
            ]
        }
        if params.get("prompt_logprobs") is not None:
            out["prompt_logprobs"] = _prompt_logprobs(results[0], list(prompt_tokens))
        return out


def build_handlers(client: AsyncArcticRLClient, isolate_capacity: int | None = None) -> dict[str, Any]:
    """Handlers the in-process client calls. There is no forward handler."""
    backend = CortexTinkerBackend(
        client, min_rows=max(int(client.config.training_gpus), 1), isolate_capacity=isolate_capacity
    )
    return {
        "fwd_bwd_handler": backend.fwd_bwd,
        "step_handler": backend.step,
        "sync_weights_handler": backend.sync_weights,
        "generate_handler": backend.generate,
    }
