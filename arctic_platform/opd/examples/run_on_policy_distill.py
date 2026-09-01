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

"""Minimal synchronous single-logit on-policy distillation loop.

Use ``--dry-run`` to validate causal alignment without launching jobs.
Use ``--live`` to run student/teacher sampling plus train steps.

3-node on-prem packing matching xyu ``yak/opd-qwen35/446z0mnb``
(8 train + 8 student-infer + 8 teacher-infer, teacher TP=2). Student Ray
cluster is nodes 0-1; teacher Ray cluster is node 2 (head has no local
GPUs). Neutrino / ``--image-tag`` are omitted::

    python -m arctic_platform.opd.examples.run_on_policy_distill \\
      --live \\
      --student-model Qwen/Qwen3.5-4B \\
      --teacher-model Qwen/Qwen3.6-27B \\
      --training-gpus 8 --sampling-gpus 8 --teacher-sampling-gpus 8 \\
      --student-tp 1 --teacher-tp 2 \\
      --no-colocate \\
      --server-cuda-visible-devices 0,1,2,3,4,5,6,7 \\
      --teacher-server-cuda-visible-devices '' \\
      --student-ray-hostfile /tmp/opd_student_hostfile \\
      --teacher-ray-hostfile /tmp/opd_teacher_hostfile \\
      --steps 152 --batch-size 16 \\
      --max-prompt-len 30144 --max-new-tokens 16384 --seq-len 46592 \\
      --prompts-file /data/xyu/important/opd_prompts_30720.jsonl \\
      --lr 2e-5 --warmup-steps 8 --lr-schedule cosine \\
      --distill-estimator low_var_kl \\
      --max-num-batched-tokens 2048 --max-tokens-per-mb 46592 \\
      --gpu-memory-utilization 0.85 \\
      --shuffle --seed 0 \\
      --save-every 25 --checkpoint-dir /data-fast/truwase/opd_qwen35_4b27b_3node_ckpt \\
      --metrics-jsonl opd_qwen35_4b27b_3node_metrics.jsonl \\
      --wandb-project arctic-opd-3node \\
      --wandb-run-name opd3n_qwen35_4b_27b_lr2e-5 \\
      --attn flash_attention_3

1-node 8-GPU pack (train 2 + student-infer 2 + teacher-infer 4), used when
only one box is available::

    python -m arctic_platform.opd.examples.run_on_policy_distill \\
      --live \\
      --student-model Qwen/Qwen3.5-4B \\
      --teacher-model Qwen/Qwen3.6-27B \\
      --training-gpus 2 --sampling-gpus 2 --teacher-sampling-gpus 4 \\
      --student-tp 1 --teacher-tp 2 \\
      --no-colocate \\
      --server-cuda-visible-devices 0,1,2,3 \\
      --teacher-server-cuda-visible-devices 4,5,6,7 \\
      --steps 152 --batch-size 16 \\
      --max-prompt-len 30144 --max-new-tokens 16384 --seq-len 46592 \\
      --prompts-file /data/xyu/important/opd_prompts_30720.jsonl \\
      --lr 2e-5 --warmup-steps 8 --lr-schedule cosine \\
      --distill-estimator low_var_kl \\
      --max-num-batched-tokens 2048 --max-tokens-per-mb 46592 \\
      --gpu-memory-utilization 0.85 \\
      --shuffle --seed 0 \\
      --save-every 25 --checkpoint-dir /data-fast/truwase/opd_qwen35_4b27b_1node_ckpt \\
      --metrics-jsonl opd_qwen35_4b27b_1node_metrics.jsonl \\
      --wandb-project arctic-opd-1node \\
      --wandb-run-name opd3n_20260812_194823_lr2e-5-1node \\
      --attn flash_attention_3

1+1+1 debug pack (one node, three GPUs; teacher TP=1)::

    python -m arctic_platform.opd.examples.run_on_policy_distill \\
      --live \\
      --student-model Qwen/Qwen3.5-4B \\
      --teacher-model Qwen/Qwen3.6-27B \\
      --training-gpus 1 --sampling-gpus 1 --teacher-sampling-gpus 1 \\
      --student-tp 1 --teacher-tp 1 \\
      --no-colocate \\
      --server-cuda-visible-devices 0,1 \\
      --teacher-server-cuda-visible-devices 2 \\
      --steps 5 --batch-size 2 \\
      --max-prompt-len 30144 --max-new-tokens 16384 --seq-len 46592 \\
      --prompts-file /data/xyu/important/opd_prompts_30720.jsonl \\
      --lr 2e-5 --warmup-steps 8 --lr-schedule cosine \\
      --distill-estimator low_var_kl \\
      --max-num-batched-tokens 2048 --max-tokens-per-mb 46592 \\
      --gpu-memory-utilization 0.85 \\
      --shuffle --seed 0 --save-every 0 \\
      --attn flash_attention_3
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from collections.abc import Mapping
from typing import Any

import torch
from transformers import AutoTokenizer

from arctic_platform.client.config import OnPremConfig
from arctic_platform.client.config import SamplingConfig
from arctic_platform.client.config import TrainingConfig
from arctic_platform.opd import ArcticOPDClient
from arctic_platform.opd import ArcticOPDClientConfig
from arctic_platform.opd import DEFAULT_PROCESSING
from arctic_platform.opd import score_teacher


logger = logging.getLogger(__name__)

# A genuine sync stall is permanent; bf16 rounding produces isolated zeros, which
# is why the threshold is a run length rather than a single step.
MAX_CONSECUTIVE_ZERO_SYNCS = 5
_SYNC_ZERO_RUN = [0]


DEFAULT_TRIVIA_PROMPTS = [
    "In one sentence, what is the capital of France?",
    "In one sentence, what is 2 + 2?",
    "In one sentence, why is the sky blue?",
    "In one sentence, what is the largest planet in the solar system?",
    "In one sentence, who wrote Hamlet?",
    "In one sentence, what is water made of?",
    "In one sentence, what year did the Apollo 11 moon landing happen?",
    "In one sentence, what is the boiling point of water at sea level?",
]


def _walk_sync_scalars(result: Any) -> tuple[list[float], list[float], list[float]]:
    """Collect (model_l2_sq_before, model_l2_sq_after, params_loaded) from a sync response."""
    before: list[float] = []
    after: list[float] = []
    loaded: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("model_l2_sq_before", "l2_sq_before") and isinstance(value, (int, float)):
                    before.append(float(value))
                elif key in ("model_l2_sq_after", "l2_sq_after") and isinstance(value, (int, float)):
                    after.append(float(value))
                elif key in ("params_loaded", "num_params_loaded") and isinstance(value, (int, float)):
                    loaded.append(float(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return before, after, loaded


def check_sync_result(result: Any) -> tuple[list[str], str]:
    """Confirm a weight sync actually landed.

    ``params_loaded=0`` is the unambiguous name-mismatch failure. An isolated
    zero L2 delta is normal at small lr (bf16 rounding); a consecutive run of
    zeros is a stall. On-prem CUDA-IPC often reports only ``status=ok`` — that
    is unverifiable from L2, so sampler/trainer parity is the real gate.
    """
    warnings: list[str] = []
    before, after, loaded = _walk_sync_scalars(result)

    if loaded and all(count == 0 for count in loaded):
        warnings.append("weight sync reported params_loaded=0 on every worker")

    if before and after:
        unchanged = sum(1 for b, a in zip(before, after) if b == a)
        if unchanged == len(before):
            _SYNC_ZERO_RUN[0] += 1
        else:
            _SYNC_ZERO_RUN[0] = 0
        if _SYNC_ZERO_RUN[0] >= MAX_CONSECUTIVE_ZERO_SYNCS:
            warnings.append(
                f"weight sync has changed nothing on all {len(before)} inference "
                f"worker(s) for {_SYNC_ZERO_RUN[0]} consecutive steps (model_l2_sq "
                f"identical before and after every time). An isolated zero is normal "
                f"at small lr -- a run this long is not, so the tensors are likely "
                f"being received and dropped and the sampler is serving stale weights"
            )
        deltas = [a - b for b, a in zip(before, after)]
        zero_note = f" [zero-delta run {_SYNC_ZERO_RUN[0]}]" if _SYNC_ZERO_RUN[0] else ""
        summary = (
            f"sync: {len(before)} worker(s), params_loaded="
            f"{sorted({int(c) for c in loaded}) if loaded else 'n/a'}, "
            f"L2^2 {before[0]:.6f} -> {after[0]:.6f} "
            f"(delta {min(deltas):+.3g}..{max(deltas):+.3g}){zero_note}"
        )
        return warnings, summary

    if loaded:
        return warnings, f"sync: params_loaded={sorted({int(c) for c in loaded})}"
    return warnings, "sync: unverifiable (no L2 / params_loaded reported)"


def check_metrics(metrics: dict[str, Any], step: int) -> list[str]:
    """Sanity-check fwd-bwd metrics. Returns human-readable warnings."""
    warnings: list[str] = []

    for key in ("distill_kl_coef", "distill_estimator_is_k3"):
        if key not in metrics:
            warnings.append(
                f"expected metric {key} missing: is processing.loss_fn actually "
                "resolving to the distillation loss?"
            )

    clamped = float(metrics.get("distill_delta_clamped_count", 0.0) or 0.0)
    if clamped > 0:
        warnings.append(
            f"distill_delta_clamped_count={clamped:g}: teacher and student "
            "disagree by more than delta_clamp somewhere; the k3 gradient is "
            "truncated for those tokens"
        )
    out_clamped = float(metrics.get("distill_kl_output_clamped_count", 0.0) or 0.0)
    if out_clamped > 0:
        warnings.append(
            f"distill_kl_output_clamped_count={out_clamped:g}: the KL output clamp "
            "fired, which zeroes the gradient for those tokens. It is off by "
            "default -- something enabled it"
        )

    if "sampler_train_kl_sum" not in metrics and "sampler_train_abs_delta_max" not in metrics:
        warnings.append(
            "sampler_train_* metrics absent: the trainer did not compare its "
            "logprobs against old_log_probs_shifted, so the sampler/trainer "
            "parity gate did NOT run"
        )
    else:
        kl_sum = float(metrics.get("sampler_train_kl_sum", 0.0) or 0.0)
        n = float(metrics.get("distill_kl_count", 0.0) or 0.0)
        mean_lowvar = kl_sum / n if n else 0.0
        gap = math.sqrt(2.0 * max(mean_lowvar, 0.0))
        abs_max = metrics.get("sampler_train_abs_delta_max")
        threshold = 0.05 if step == 0 else 0.25
        abs_threshold = 0.1 if step == 0 else 0.5
        if abs_max is not None and float(abs_max) > abs_threshold:
            warnings.append(
                f"step {step} sampler/trainer abs_delta_max = {float(abs_max):.4f} > "
                f"{abs_threshold} (RMS={gap:.4f}, n={n}). "
                + (
                    "Before any update these must match; bf16 lm-head noise or a "
                    "loss-mask / prompt-completion split bug."
                    if step == 0
                    else "Weights are synced every step, so a growing max gap points "
                    "at sampler/trainer precision drift or /sync-weights not landing."
                )
            )
        if gap > threshold:
            warnings.append(
                f"step {step} sampler/trainer RMS delta = {gap:.4f} > "
                f"{threshold} (mean_lowvar={mean_lowvar:.6g}, "
                f"abs_delta_max={abs_max}, n={n}). "
                + (
                    "Before any update these must match; check the loss-mask "
                    "shift and the prompt/completion split."
                    if step == 0
                    else "Weights are synced every step, so a growing gap points "
                    "at /sync-weights not landing."
                )
            )
        elif not (abs_max is not None and float(abs_max) > abs_threshold):
            logger.info(
                "step %d parity OK: sampler/trainer RMS delta=%.5f "
                "(mean_lowvar=%.3g, abs_delta_max=%s) over n=%s tokens",
                step,
                gap,
                mean_lowvar,
                abs_max,
                n,
            )

    kl_sum = metrics.get("distill_kl_sum")
    kl_count = metrics.get("distill_kl_count")
    if kl_sum is not None and kl_count:
        per_tok = float(kl_sum) / float(kl_count)
        if per_tok < 0:
            warnings.append(
                f"mean per-token KL is negative ({per_tok:.6g}); k3 is "
                "non-negative by construction, so this indicates a bad estimator"
            )
    return warnings


def build_batch(rollouts: list[dict[str, Any]], pad_token_id: int, max_seq_len: int) -> dict[str, torch.Tensor]:
    lengths = [len(row["prompt_ids"]) + len(row["completion_ids"]) for row in rollouts]
    width = max(lengths)
    if width > max_seq_len:
        raise ValueError(f"batch width {width} exceeds max_seq_len={max_seq_len}")
    batch_size = len(rollouts)
    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    teacher = torch.zeros((batch_size, width), dtype=torch.float32)
    sampler = torch.zeros((batch_size, width), dtype=torch.float32)

    prompt_width = max(len(row["prompt_ids"]) for row in rollouts)
    prompts = torch.full((batch_size, prompt_width), pad_token_id, dtype=torch.long)

    for row_index, row in enumerate(rollouts):
        prompt = list(row["prompt_ids"])
        completion = list(row["completion_ids"])
        full_ids = prompt + completion
        start, stop = len(prompt) - 1, len(full_ids) - 1
        if len(row["teacher_logprobs"]) != len(completion):
            raise ValueError("teacher logprobs are not aligned with completion tokens")
        if len(row["sampler_logprobs"]) != len(completion):
            raise ValueError("sampler logprobs are not aligned with completion tokens")
        input_ids[row_index, : len(full_ids)] = torch.tensor(full_ids)
        attention_mask[row_index, : len(full_ids)] = True
        loss_mask[row_index, start:stop] = True
        teacher[row_index, start:stop] = torch.tensor(row["teacher_logprobs"])
        sampler[row_index, start:stop] = torch.tensor(row["sampler_logprobs"])
        prompts[row_index, : len(prompt)] = torch.tensor(prompt)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "teacher_log_probs_shifted": teacher,
        "old_log_probs_shifted": sampler,
        "prompts": prompts,
    }


def lr_at(
    step: int,
    peak_lr: float,
    warmup_steps: int,
    *,
    schedule: str = "linear",
    total_steps: int | None = None,
) -> float:
    """xyu ``lr_at``: warmup then cosine floored at 10% of peak. Applied via ``step(lr)``."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if schedule != "cosine":
        return peak_lr
    span = max(1, (total_steps or 0) - max(warmup_steps, 0))
    frac = min(1.0, max(0.0, float(step - warmup_steps) / float(span)))
    return peak_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))


def prompt_content_from_record(record: dict[str, Any]) -> str | list[dict[str, Any]]:
    messages = record.get("messages")
    if messages:
        return list(messages)
    for key in ("prompt", "text", "content"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"jsonl record is missing prompt/text/content/messages: {sorted(record)}")


def load_prompt_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(payload)
    if not records:
        raise ValueError(f"{path} contained no prompt records")
    return records


def _as_token_ids(ids: Any) -> list[int]:
    """Normalize chat-template output to a 1-D ``list[int]``.

    Qwen3 tokenizers return ``list[int]``. Qwen3.5 processors return a
    ``BatchEncoding`` / mapping with ``input_ids`` (sometimes one batch dim).
    """
    if isinstance(ids, Mapping):
        if "input_ids" not in ids:
            raise TypeError(f"chat template mapping has no input_ids: {type(ids).__name__}")
        ids = ids["input_ids"]
    if hasattr(ids, "tolist") and not isinstance(ids, (list, tuple)):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError(f"expected a single prompt, got batch of {len(ids)}")
        ids = list(ids[0])
    try:
        out = [int(x) for x in ids]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"chat template did not return token ids: {type(ids).__name__} {ids!r}") from exc
    if not out:
        raise TypeError("chat template returned empty token ids")
    return out


def tokenize_prompt(
    tokenizer: Any,
    prompt: str | list[dict[str, Any]],
    enable_thinking: bool = False,
) -> list[int]:
    """Chat-template tokenize one prompt the way xyu's Neutrino client does.

    xyu documents the Qwen3.5 student template as non-thinking
    (``<think>\\n\\n</think>\\n\\n``). transformers 5.16 defaults to an open
    ``<think>`` block when ``enable_thinking`` is omitted, so we pin it here.
    ``enable_thinking=True`` matches xyu's omit-the-kwarg default on 5.16 and is
    used by the Phase-1 thinking smoke.
    """
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "enable_thinking": enable_thinking,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    return _as_token_ids(ids)


def student_sampling_params(max_tokens: int) -> dict[str, Any]:
    """vLLM sampling dict matching xyu ``446z0mnb``, minus keys this engine rejects."""
    params: dict[str, Any] = {
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "logprobs": 0,
        "top_k": -1,
        "min_p": 0.0,
    }
    try:
        from vllm.sampling_params import SamplingParams
    except Exception:
        params["return_sampled_logprobs_only"] = True
        return params
    try:
        SamplingParams(**params)
    except TypeError:
        params.pop("min_p", None)
        params.pop("top_k", None)
    try:
        SamplingParams(**{**params, "return_sampled_logprobs_only": True})
    except TypeError:
        return params
    params["return_sampled_logprobs_only"] = True
    return params


def _vllm_engine_fields() -> set[str]:
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:
        return set()
    fields = getattr(EngineArgs, "model_fields", None)
    if fields is None:
        fields = getattr(EngineArgs, "__dataclass_fields__", {})
    return set(fields)


def _vllm_version_tuple() -> tuple[int, ...]:
    try:
        import vllm
    except Exception:
        return ()
    parts: list[int] = []
    for token in str(getattr(vllm, "__version__", "")).split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def vllm_supports_fp32_lm_head() -> bool:
    """True when this vLLM build accepts Neutrino's ``fp32_lm_head`` flag."""
    return "fp32_lm_head" in _vllm_engine_fields()


def vllm_fp32_engine_kwargs() -> dict[str, Any]:
    """Engine kwargs that put the vLLM LM head in fp32.

    Neutrino exposes ``fp32_lm_head``. Stock vLLM 0.26 uses
    ``hf_overrides={"head_dtype": "float32"}`` (PR #48390). 0.18 generation
    models ignore ``head_dtype``, so do not send it there.
    """
    if vllm_supports_fp32_lm_head():
        return {"fp32_lm_head": True}
    if _vllm_version_tuple() >= (0, 26):
        return {"hf_overrides": {"head_dtype": "float32"}}
    return {}


def vllm_fa2_engine_kwargs() -> dict[str, Any]:
    """Force vLLM softmax attention to FA2 so it matches ``--attn flash_attention_2``.

    Stock vLLM 0.26 picks FA3 on Hopper. The trainer still uses FA2, and the
    step-8 length cliff survived fp32 heads with a healthy ``abs_delta_max``.
    """
    if _vllm_version_tuple() >= (0, 26):
        return {"attention_config": {"flash_attn_version": 2}}
    return {}


def tokenize_and_filter(
    tokenizer: Any,
    prompts: list[str | list[dict[str, Any]] | dict[str, Any]],
    max_prompt_len: int | None,
    enable_thinking: bool = False,
) -> list[list[int]]:
    kept: list[list[int]] = []
    for prompt in prompts:
        content = prompt_content_from_record(prompt) if isinstance(prompt, dict) else prompt
        ids = tokenize_prompt(tokenizer, content, enable_thinking=enable_thinking)
        if max_prompt_len is not None and len(ids) > max_prompt_len:
            continue
        kept.append(ids)
    if not kept:
        raise ValueError("no prompts left after max-prompt-len filter")
    return kept


def iter_batches(
    prompt_ids: list[list[int]],
    batch_size: int,
    steps: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[list[list[int]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    ids = list(prompt_ids)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)
    n = len(ids)
    batches: list[list[list[int]]] = []
    for step in range(steps):
        start = (step * batch_size) % n
        batches.append([ids[(start + offset) % n] for offset in range(batch_size)])
    return batches


def sampled_logprobs(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    token_ids = list(output["token_ids"])
    positions = output.get("logprobs")
    if positions is None:
        raise RuntimeError("generate did not return logprobs; pass logprobs=0")
    values: list[float] = []
    for token_id, position in zip(token_ids, positions):
        if isinstance(position, dict):
            entry = position.get(token_id, position.get(str(token_id)))
            if entry is None:
                raise RuntimeError(f"generated token {token_id} missing from logprob keys")
            values.append(float(entry["logprob"] if isinstance(entry, dict) else entry))
        else:
            values.append(float(position))
    return token_ids, values


def train_step(
    client: ArcticOPDClient,
    prompt_ids: list[list[int]],
    pad_token_id: int,
    max_seq_len: int,
    learning_rate: float,
    *,
    max_tokens: int = 32,
    processing: dict[str, Any] | None = None,
) -> dict:
    t0 = time.monotonic()
    t_phase = time.monotonic()
    outputs = client.generate(prompt_ids, student_sampling_params(max_tokens))
    gen_s = time.monotonic() - t_phase
    rollouts = []
    for prompt, output in zip(prompt_ids, outputs):
        token_ids, sampler_logprobs = sampled_logprobs(output)
        if not token_ids:
            continue
        rollouts.append(
            {
                "prompt_ids": prompt,
                "completion_ids": token_ids,
                "sampler_logprobs": sampler_logprobs,
            }
        )
    if not rollouts:
        raise RuntimeError("student generated empty completions for every prompt")
    t_phase = time.monotonic()
    scored = score_teacher(client, rollouts)
    score_s = time.monotonic() - t_phase
    batch = build_batch(scored, pad_token_id, max_seq_len)
    t_phase = time.monotonic()
    fwd = client.fwd_bwd(
        batch,
        processing=processing,
        meta={"pad_token_id": pad_token_id, "zorro_train_enable": False},
    )
    fwdbwd_s = time.monotonic() - t_phase
    t_phase = time.monotonic()
    step_result = client.step(learning_rate)
    sync_result = client.sync_weights()
    step_sync_s = time.monotonic() - t_phase
    tokens_scored = sum(len(row["completion_ids"]) for row in scored)
    return {
        "forward_backward": fwd,
        "step": step_result,
        "sync": sync_result,
        "n_rollouts": len(scored),
        "tokens_scored": tokens_scored,
        "times": {
            "gen_s": gen_s,
            "score_s": score_s,
            "fwdbwd_s": fwdbwd_s,
            "step_sync_s": step_sync_s,
            "total_s": time.monotonic() - t0,
        },
    }


def _metric(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _maybe_metric(value: Any) -> float | None:
    if value is None:
        return None
    return _metric(value)


def kl_per_token_len_adj(per_token: float, tokens_per_rollout: float, len_ref: float, exponent: float) -> float:
    """xyu length adjustment: ``kl * (tokens_per_rollout / len_ref) ** exponent``."""
    if tokens_per_rollout <= 0 or len_ref <= 0:
        return per_token
    return per_token * (tokens_per_rollout / len_ref) ** exponent


def build_step_record(
    *,
    step: int,
    learning_rate: float,
    result: dict[str, Any],
    tokens_cumulative: int,
    elapsed_s: float,
    wandb_len_ref: float,
    wandb_len_exponent: float,
) -> dict[str, Any]:
    """xyu-style namespaced metrics plus the previous flat aliases."""
    metrics = (result.get("forward_backward") or {}).get("metrics") or {}
    step_metrics = (result.get("step") or {}).get("metrics") or {}
    times = result.get("times") or {}
    n_rollouts = int(result.get("n_rollouts") or 0)
    tokens_scored = int(result.get("tokens_scored") or 0)
    tokens_per_rollout = tokens_scored / n_rollouts if n_rollouts else 0.0
    loss = _metric(metrics.get("loss", float("nan")))
    kl_count = _maybe_metric(metrics.get("distill_kl.tokens")) or _maybe_metric(metrics.get("distill_kl_count"))
    paired_kl_sum = _maybe_metric(metrics.get("distill_kl.sum"))
    if paired_kl_sum is not None and kl_count:
        kl_per_token = paired_kl_sum / kl_count
    elif "distill_kl" in metrics:
        kl_per_token = _metric(metrics["distill_kl"])
    elif metrics.get("distill_kl_sum") is not None and kl_count:
        kl_per_token = _metric(metrics["distill_kl_sum"]) / kl_count
    else:
        kl_per_token = float("nan")
    if "distill_k1" in metrics:
        k1_per_token: float | None = _metric(metrics["distill_k1"])
    elif metrics.get("distill_k1_sum") is not None and kl_count:
        k1_per_token = _metric(metrics["distill_k1_sum"]) / kl_count
    else:
        k1_per_token = None
    record: dict[str, Any] = {
        "step": step,
        "loss/avg": loss,
        "kl/per_token": kl_per_token,
        "kl/per_token_len_adj": kl_per_token_len_adj(
            kl_per_token, tokens_per_rollout, wandb_len_ref, wandb_len_exponent
        ),
        "optim/lr": learning_rate,
        "optim/update_successful": 1.0,
        "tokens/scored": tokens_scored,
        "tokens/per_rollout": tokens_per_rollout,
        "tokens/cumulative": tokens_cumulative,
        "n_rollouts": n_rollouts,
        "lr": learning_rate,
        "loss": loss,
        "distill_kl": kl_per_token,
        "elapsed_s": elapsed_s,
    }
    for dest, source in (
        ("kl/k1_per_token", None),
        ("distill_kl_count", "distill_kl_count"),
        ("distill_kl_max", "distill_kl_max"),
        ("distill_delta_max", "distill_delta_max"),
        ("distill_delta_min", "distill_delta_min"),
        ("distill_delta_clamped_count", "distill_delta_clamped_count"),
        ("sampler_train_abs_delta_max", "sampler_train_abs_delta_max"),
        ("sampler_train_abs_delta_mean", "sampler_train_abs_delta_mean"),
        ("sampler_train_kl_sum", "sampler_train_kl_sum"),
        ("distill_batch_num_tokens", "distill_batch_num_tokens"),
        ("distill_dp_size", "distill_dp_size"),
        ("distill_kl.sum", "distill_kl.sum"),
        ("distill_kl.tokens", "distill_kl.tokens"),
        ("loss.sum", "loss.sum"),
        ("loss.tokens", "loss.tokens"),
    ):
        if dest == "kl/k1_per_token":
            value = k1_per_token
        else:
            value = _maybe_metric(metrics.get(source))
        if value is not None:
            record[dest] = value
    if "sampler_train_abs_delta_max" in record:
        record["sync/abs_delta_max"] = record["sampler_train_abs_delta_max"]
    grad_norm = _maybe_metric(step_metrics.get("grad_norm"))
    if grad_norm is not None:
        record["optim/grad_norm"] = grad_norm
        record["grad_norm"] = grad_norm
    for key in ("gen_s", "score_s", "fwdbwd_s", "step_sync_s", "total_s"):
        if key in times:
            record[f"time/{key}"] = float(times[key])
    return record


def _student_gpu_memory_utilization(args: argparse.Namespace) -> float:
    if args.student_gpu_memory_utilization is not None:
        return args.student_gpu_memory_utilization
    if args.colocate:
        return 0.35
    return args.gpu_memory_utilization


def resolve_train_attn(requested: str) -> str:
    """xyu trains with FA3. Stock transformers needs the FA3 package; fall back to FA2."""
    if requested != "flash_attention_3":
        return requested
    try:
        from transformers.utils import is_flash_attn_3_available
    except Exception:
        print("train_attn_fallback flash_attention_3 -> flash_attention_2 (import check failed)")
        return "flash_attention_2"
    if is_flash_attn_3_available():
        return requested
    print("train_attn_fallback flash_attention_3 -> flash_attention_2 (HF FlashAttention3 not installed)")
    return "flash_attention_2"


def _ds_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size % args.training_gpus != 0:
        raise SystemExit(
            f"--batch-size {args.batch_size} must be divisible by --training-gpus {args.training_gpus}"
        )
    gas = args.batch_size // args.training_gpus
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": gas,
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        },
    }
    if getattr(args, "max_tokens_per_mb", None):
        per_gpu = max(1, args.batch_size // args.training_gpus)
        ds_config["train_micro_batch_size_per_gpu"] = per_gpu
        ds_config["gradient_accumulation_steps"] = 1
        ds_config["train_batch_size"] = per_gpu * args.training_gpus
    # LR schedule is applied per step via ``step(lr)`` (xyu ``lr_at``), not a
    # DeepSpeed scheduler that would overwrite the client LR.
    return ds_config


def build_live_client(args: argparse.Namespace):
    """Construct the live client and its inputs.

    Shared by ``live_main`` and the offline kl/per_token probe so both exercise
    the exact same student/teacher bring-up. Returns
    ``(client, tokenizer, pad_token_id, prompt_ids, batches, processing, checkpoint_path)``.
    """
    print(f"client CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)
    if args.prompts_file:
        raw_prompts: list[str | list[dict[str, Any]] | dict[str, Any]] = load_prompt_records(args.prompts_file)
    else:
        raw_prompts = list(DEFAULT_TRIVIA_PROMPTS)
    enable_thinking = bool(getattr(args, "enable_thinking", False))
    print(f"enable_thinking={enable_thinking}")
    prompt_ids = tokenize_and_filter(
        tokenizer, raw_prompts, args.max_prompt_len, enable_thinking=enable_thinking
    )
    batches = iter_batches(
        prompt_ids,
        args.batch_size,
        args.steps,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"n_prompts={len(prompt_ids)} batch_size={args.batch_size} steps={args.steps}")

    checkpoint_path = args.checkpoint_dir or f"/tmp/arctic_opd_ckpt_{os.getpid()}"
    print(f"checkpoint_path={checkpoint_path}")
    print(f"student={args.student_model} teacher={args.teacher_model}")

    if args.sampling_gpus % args.student_tp != 0:
        raise SystemExit(
            f"--student-infer-gpus {args.sampling_gpus} must be divisible by --student-tp {args.student_tp}"
        )
    if args.teacher_sampling_gpus % args.teacher_tp != 0:
        raise SystemExit(
            f"--teacher-infer-gpus {args.teacher_sampling_gpus} must be divisible by --teacher-tp {args.teacher_tp}"
        )

    student_util = _student_gpu_memory_utilization(args)
    student_vllm: dict[str, Any] = {
        "gpu_memory_utilization": student_util,
        "max_model_len": args.max_seq_len,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "tensor_parallel_size": args.student_tp,
    }
    teacher_vllm: dict[str, Any] = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_seq_len,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "tensor_parallel_size": args.teacher_tp,
    }
    if args.max_num_batched_tokens is not None:
        student_vllm["max_num_batched_tokens"] = args.max_num_batched_tokens
        teacher_vllm["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.vllm_fp32_lm_head:
        fp32_engine = vllm_fp32_engine_kwargs()
        if fp32_engine:
            student_vllm.update(fp32_engine)
            teacher_vllm.update(fp32_engine)
            print(f"vllm_fp32_lm_head={fp32_engine}")
        else:
            print("vllm_fp32_lm_head=skipped (need Neutrino fp32_lm_head or vLLM>=0.26 head_dtype)")
    else:
        print("vllm_fp32_lm_head=disabled")
    print(f"vllm_enforce_eager={args.enforce_eager}")
    fa_version = args.vllm_flash_attn_version
    if fa_version == 2:
        fa2_engine = vllm_fa2_engine_kwargs()
        if fa2_engine:
            student_vllm.update(fa2_engine)
            teacher_vllm.update(fa2_engine)
        print(f"vllm_flash_attn={fa2_engine or 'env VLLM_FLASH_ATTN_VERSION=2'}")
    elif fa_version is not None:
        print(f"vllm_flash_attn=pin {fa_version}")
    else:
        print("vllm_flash_attn=default (FA3 on Hopper)")

    processing = copy.deepcopy(DEFAULT_PROCESSING)
    processing["config"]["distill_estimator"] = args.distill_estimator
    train_attn = resolve_train_attn(args.attn)
    print(f"train_attn={train_attn} requested={args.attn}")
    print(f"train_fp32_lm_head={args.train_fp32_lm_head}")

    config = ArcticOPDClientConfig(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        training_gpus=args.training_gpus,
        sampling_gpus=args.sampling_gpus,
        teacher_sampling_gpus=args.teacher_sampling_gpus,
        training=TrainingConfig(
            checkpoint_path=checkpoint_path,
            cuda_ipc=True,
            ds_config=_ds_config(args),
            ds_worker_config={
                "attn_implementation": train_attn,
                "enable_gradient_checkpointing": args.max_seq_len > 2048,
                "zorro_train_enable": False,
                "fp32_lm_head": args.train_fp32_lm_head,
                "fused_cross_entropy": False,
                "fla_tilelang": False,
                **({"fused_lm_head_token_chunk_size": 2048} if args.train_fp32_lm_head else {}),
                **(
                    {"max_tokens_per_mb": args.max_tokens_per_mb}
                    if args.max_tokens_per_mb
                    else {}
                ),
            },
        ),
        sampling=SamplingConfig(vllm=student_vllm),
        teacher_sampling=SamplingConfig(vllm=teacher_vllm),
        backend=OnPremConfig(
            protocol="http",
            host="localhost",
            port=args.port,
            colocate=args.colocate,
            launch_local_server=True,
            server_cuda_visible_devices=args.server_cuda_visible_devices,
            startup_timeout=args.startup_timeout,
            server_extra_env={
                "FLA_TILELANG": "0",
                "FLA_DISABLE_BACKEND_DISPATCH": "1",
                **(
                    {"VLLM_FLASH_ATTN_VERSION": str(args.vllm_flash_attn_version)}
                    if args.vllm_flash_attn_version is not None
                    else {}
                ),
            },
        ),
        teacher_port=args.teacher_port,
        teacher_server_cuda_visible_devices=args.teacher_server_cuda_visible_devices,
        student_ray_hostfile=args.student_ray_hostfile,
        teacher_ray_hostfile=args.teacher_ray_hostfile,
        job_ready_timeout=args.job_ready_timeout,
        request_timeout=args.request_timeout,
    )

    client = ArcticOPDClient(config)
    print(
        "composed "
        f"student={type(client.student).__name__}(train_gpus={client.student.config.training_gpus},"
        f" sample_gpus={client.student.config.sampling_gpus}) "
        f"teacher={type(client.teacher).__name__}(train_gpus={client.teacher.config.training_gpus},"
        f" sample_gpus={client.teacher.config.sampling_gpus})"
    )
    print(
        "jobs "
        f"train={client.student_jobs.training} "
        f"sample={client.student_jobs.sampling} "
        f"teacher={client.teacher_jobs.sampling}"
    )
    print(
        f"colocate={args.colocate} student_tp={args.student_tp} teacher_tp={args.teacher_tp} "
        f"student_gpu_mem={student_util} teacher_gpu_mem={args.gpu_memory_utilization} "
        f"max_tokens_per_mb={args.max_tokens_per_mb} "
        f"student_hostfile={args.student_ray_hostfile} teacher_hostfile={args.teacher_ray_hostfile}"
    )
    return client, tokenizer, pad_token_id, prompt_ids, batches, processing, checkpoint_path


def live_main(args: argparse.Namespace) -> None:
    (
        client,
        tokenizer,
        pad_token_id,
        prompt_ids,
        batches,
        processing,
        checkpoint_path,
    ) = build_live_client(args)
    metrics_path = args.metrics_jsonl or f"/tmp/arctic_opd_metrics_{os.getpid()}.jsonl"
    print(f"metrics_jsonl={metrics_path}")
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(" --wandb-project requires the wandb package") from exc
        wandb_kwargs: dict[str, Any] = {"project": args.wandb_project, "config": vars(args)}
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        wandb_run = wandb.init(**wandb_kwargs)
    started = time.monotonic()
    tokens_cumulative = 0
    try:
        for step, batch_prompts in enumerate(batches):
            learning_rate = lr_at(
                step,
                args.lr,
                args.warmup_steps,
                schedule=args.lr_schedule,
                total_steps=args.steps,
            )
            result = train_step(
                client,
                batch_prompts,
                pad_token_id,
                args.max_seq_len,
                learning_rate,
                max_tokens=args.max_tokens,
                processing=processing,
            )
            tokens_cumulative += int(result.get("tokens_scored") or 0)
            record = build_step_record(
                step=step + 1,
                learning_rate=learning_rate,
                result=result,
                tokens_cumulative=tokens_cumulative,
                elapsed_s=time.monotonic() - started,
                wandb_len_ref=args.wandb_len_ref,
                wandb_len_exponent=args.wandb_len_exponent,
            )
            metrics = (result.get("forward_backward") or {}).get("metrics") or {}
            metric_warnings = check_metrics(metrics, step)
            sync_warnings, sync_summary = check_sync_result(result.get("sync"))
            record["sync/summary"] = sync_summary
            for warning in metric_warnings + sync_warnings:
                print("WARNING " + warning, flush=True)
            print(sync_summary, flush=True)
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                wandb_run.log(record, step=record["step"])
            print("METRICS " + json.dumps(record), flush=True)
            line = (
                f"step {record['step']}/{args.steps} "
                f"lr={learning_rate:.3g} "
                f"loss={record['loss']:.4g} "
                f"distill_kl={record['distill_kl']:.4g} "
                f"n={record['n_rollouts']} "
                f"tokens={record['tokens/scored']}"
            )
            if "time/total_s" in record:
                line += (
                    f" gen={record['time/gen_s']:.1f}s"
                    f" score={record['time/score_s']:.1f}s"
                    f" fwdbwd={record['time/fwdbwd_s']:.1f}s"
                    f" step_sync={record['time/step_sync_s']:.1f}s"
                    f" total={record['time/total_s']:.1f}s"
                )
            if record.get("sampler_train_abs_delta_max") is not None:
                line += f" sampler_train_abs_delta_max={_metric(record['sampler_train_abs_delta_max']):.4g}"
            if record.get("grad_norm") is not None:
                line += f" grad_norm={_metric(record['grad_norm']):.4g}"
            print(line, flush=True)
            if args.save_every and (step + 1) % args.save_every == 0:
                save_result = client.save_checkpoint(step=step + 1, path=checkpoint_path)
                print(f"checkpoint step={step + 1} {save_result}", flush=True)
        if args.save_every and args.steps % args.save_every != 0:
            save_result = client.save_checkpoint(step=args.steps, path=checkpoint_path)
            print(f"checkpoint step={args.steps} {save_result}", flush=True)
        print("live OPD run complete")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        client.shutdown()
        print("shutdown complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--student-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-5)
    parser.add_argument("--lr-schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", "--seq-len", dest="max_seq_len", type=int, default=512)
    parser.add_argument("--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=32)
    parser.add_argument("--max-prompt-len", type=int, default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill-estimator", default="low_var_kl")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--max-tokens-per-mb",
        type=int,
        default=None,
        help="Token budget per packed train microbatch (xyu mb_spec). "
        "When set, the worker packs instead of splitting by DeepSpeed GAS.",
    )
    parser.add_argument("--training-gpus", "--train-gpus", dest="training_gpus", type=int, default=1)
    parser.add_argument("--sampling-gpus", "--student-infer-gpus", dest="sampling_gpus", type=int, default=1)
    parser.add_argument(
        "--teacher-sampling-gpus", "--teacher-infer-gpus", dest="teacher_sampling_gpus", type=int, default=1
    )
    parser.add_argument("--student-tp", type=int, default=1)
    parser.add_argument("--teacher-tp", type=int, default=1)
    parser.add_argument("--colocate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--teacher-port", type=int, default=18101)
    parser.add_argument("--server-cuda-visible-devices", default="0")
    parser.add_argument("--teacher-server-cuda-visible-devices", default="1")
    parser.add_argument(
        "--student-ray-hostfile",
        default=None,
        help="Hostfile for the student Ray cluster (ARL_RAY_HOSTFILE). "
        "Use with --teacher-ray-hostfile to pin 8+8+8 roles onto disjoint nodes.",
    )
    parser.add_argument(
        "--teacher-ray-hostfile",
        default=None,
        help="Hostfile for the teacher Ray cluster (ARL_RAY_HOSTFILE).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--student-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--attn", default="flash_attention_3")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Student chat template thinking block. Default off (pinned "
        "non-thinking). --enable-thinking matches xyu's omit-the-kwarg default "
        "on transformers 5.16 (thinking ON); used by the Phase-1 thinking smoke.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="vLLM enforce_eager. Default off so CUDA graphs can run (v6). "
        "v5 used --enforce-eager and paid ~2x generate vs xyu.",
    )
    parser.add_argument(
        "--vllm-flash-attn-version",
        type=int,
        default=None,
        choices=(2, 3, 4),
        help="Pin vLLM FlashAttention version. Default: let vLLM pick (FA3 on Hopper).",
    )
    parser.add_argument(
        "--vllm-fp32-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send stock vLLM head_dtype=float32 (or Neutrino fp32_lm_head). "
        "Use --no-vllm-fp32-lm-head to isolate bf16 infer head vs v9.",
    )
    parser.add_argument(
        "--train-fp32-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject chunked fp32 train LM head. Use --no-train-fp32-lm-head to isolate.",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-len-ref",
        type=float,
        default=4096.0,
        help="Reference completion length for kl/per_token_len_adj (xyu default 4096).",
    )
    parser.add_argument(
        "--wandb-len-exponent",
        type=float,
        default=0.486,
        help="Exponent for kl/per_token_len_adj (xyu default 0.486).",
    )
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--job-ready-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        batch = build_batch(
            [
                {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4],
                    "sampler_logprobs": [-0.2, -0.3],
                    "teacher_logprobs": [-0.1, -0.25],
                }
            ],
            pad_token_id=0,
            max_seq_len=8,
        )
        assert batch["loss_mask"].tolist() == [[False, True, True, False]]
        print("dry-run OK")
        return
    if args.live:
        live_main(args)
        return
    raise SystemExit("Pass --dry-run or --live.")


if __name__ == "__main__":
    main()
