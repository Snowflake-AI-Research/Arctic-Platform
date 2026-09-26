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

"""Weighted NLL and grouped KL/JSD class losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import torch
from torch.utils.checkpoint import checkpoint

from arctic_platform.common.registry import LOSS_FNS

from .base_loss import BaseLoss
from .causal_cross_entropy import _connected_zero
from .causal_cross_entropy import _validate_global_normalization
from .compute_logprobs import _LOGPROB_SLICE_BYTES
from .functional import _resolve_dp_size
from .functional import canonicalize_loss_mask
from .functional import resolve_global_loss_scale

_DISTILLATION_METRICS = (
    "kd_weight_sum",
    "kd_sum",
    "teacher_tail_mass_sum",
    "student_tail_mass_sum",
)
_GROUPED_DISTILLATION_METRICS = (*_DISTILLATION_METRICS, "sft_nll_sum")
_GRPO_DISTILLATION_METRICS = (*_DISTILLATION_METRICS, "loss_term_kd")
_GROUPED_DISTILLATION_CONFIG_KEYS = frozenset(
    {"kd_coef", "kd_divergence", "kd_beta", "kd_batch_num_tokens", "dp_size"}
)
_GRPO_DISTILLATION_CONFIG_KEYS = frozenset({"kd_coef", "kd_divergence", "kd_beta", "kd_batch_num_tokens"})
_TEACHER_TOKEN_ID_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
)
_TEACHER_LOG_PROB_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})


class Divergence(str, Enum):
    KL = "kl"
    JSD = "jsd"


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return result


def _resolve_divergence(config: dict, prefix: str) -> tuple[Divergence, float]:
    try:
        divergence = Divergence(config.get(f"{prefix}divergence", "kl"))
    except ValueError as exc:
        raise ValueError(
            f"{prefix}divergence must be 'kl' or 'jsd', got {config.get(f'{prefix}divergence')!r}"
        ) from exc
    beta = _finite_real(config.get(f"{prefix}beta", 1.0), f"{prefix}beta")
    valid = 0 < beta < 1 if divergence is Divergence.JSD else 0 <= beta <= 1
    if not valid:
        interval = "(0, 1)" if divergence is Divergence.JSD else "[0, 1]"
        raise ValueError(f"{prefix}beta must lie in {interval} for {divergence.value!r}, got {beta}")
    return divergence, beta


def _kl(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """KL(a || b) per row, including the conventional ``0 log 0 = 0``."""
    return (log_a.exp() * (log_a - log_b).masked_fill(torch.isneginf(log_a), 0.0)).sum(-1)


def _divergence_values(
    log_teacher: torch.Tensor,
    log_student: torch.Tensor,
    divergence: Divergence,
    beta: float,
) -> torch.Tensor:
    references = (log_student, log_teacher)
    if divergence is Divergence.JSD:
        empty = torch.isneginf(log_teacher) & torch.isneginf(log_student)
        mixture = torch.logaddexp(
            log_teacher.masked_fill(empty, 0) + math.log(beta),
            log_student.masked_fill(empty, 0) + math.log1p(-beta),
        )
        references = (mixture, mixture)
    sides = (
        (beta, log_teacher, references[0]),
        (1.0 - beta, log_student, references[1]),
    )
    return sum(
        (weight * _kl(distribution, reference) for weight, distribution, reference in sides if weight > 0),
        torch.zeros_like(log_teacher[:, 0]),
    )


def _student_groups_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    inactive: torch.Tensor,
) -> torch.Tensor:
    """Build candidate buckets and a complement bucket without ``1 - sum(p)``."""
    work = logits.to(torch.promote_types(logits.dtype, torch.float32), copy=True).masked_fill_(inactive[:, None], 0)
    vocab_size = work.shape[-1]
    slots = torch.where((token_ids >= 0) & (token_ids < vocab_size), token_ids, vocab_size)
    work = torch.cat([work, work.new_full((work.shape[0], 1), -math.inf)], -1)
    complement = work.scatter(-1, slots, -math.inf)
    empty = torch.isneginf(complement.amax(-1, keepdim=True))
    complement[:, -1] = torch.where(empty.squeeze(-1), 0.0, complement[:, -1])
    tail = complement.logsumexp(-1, keepdim=True).masked_fill(empty, -math.inf)
    return torch.cat([work.gather(-1, slots), tail], -1).log_softmax(-1)


def _selected_rows(
    tensor: torch.Tensor,
    index: torch.Tensor,
    positions: int,
    width: int,
    name: str,
) -> torch.Tensor:
    if tensor.numel() != positions * width:
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} must hold {width} value(s) at each of {positions} positions"
        )
    return tensor.reshape(positions, width).to(index.device).index_select(0, index)


def _temperature_values(temperature) -> list:
    if torch.is_tensor(temperature):
        return temperature.detach().reshape(-1)[:8].cpu().tolist()
    return [temperature]


def _validate_neutral_temperature(context: dict) -> None:
    temperature = context.get("temperature")
    if temperature is None:
        return
    if torch.is_tensor(temperature):
        neutral = (
            not torch.is_complex(temperature)
            and bool(torch.isfinite(temperature).all().item())
            and bool((temperature == 1).all().item())
        )
    else:
        neutral = (
            isinstance(temperature, Real)
            and not isinstance(temperature, bool)
            and math.isfinite(float(temperature))
            and float(temperature) == 1.0
        )
    if not neutral:
        raise ValueError(
            "grouped distillation requires untempered model probabilities; "
            f"omit temperature or set every value to 1, received {_temperature_values(temperature)!r}"
        )


def _require_tensor(context: dict, name: str) -> torch.Tensor:
    value = context.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"grouped distillation requires tensor context[{name!r}]")
    return value


def _signed_teacher_token_ids(context: dict, weights: torch.Tensor) -> torch.Tensor:
    token_ids = _require_tensor(context, "teacher_token_ids")
    if token_ids.dtype not in _TEACHER_TOKEN_ID_DTYPES:
        raise ValueError(f"teacher_token_ids must contain integer ids (int32 recommended), got {token_ids.dtype}")
    if token_ids.ndim == 0:
        raise ValueError("teacher_token_ids must have a final candidate dimension")
    if token_ids.shape[-1] < 1:
        raise ValueError("teacher_token_ids candidate dimension M must be at least 1")
    if tuple(token_ids.shape[:-1]) != tuple(weights.shape):
        raise ValueError(
            f"teacher_token_ids leading shape {tuple(token_ids.shape[:-1])} must match "
            f"grouped-distillation weights shape {tuple(weights.shape)}"
        )

    signed = token_ids.to(dtype=torch.int64)
    if token_ids.dtype == torch.uint64:
        active = weights.reshape(-1) > 0
        active_ids = signed.reshape(active.numel(), signed.shape[-1])[active.to(signed.device)]
        if bool((active_ids < 0).any().item()):
            raise ValueError("active uint64 teacher_token_ids must fit in signed int64")
    return signed


def _validate_action_masked_labels(context: dict, weights: torch.Tensor) -> None:
    action_masks = context.get("action_masks")
    labels = context.get("labels")
    if action_masks is None or not torch.is_tensor(labels):
        return
    if not isinstance(action_masks, dict):
        raise ValueError("grouped distillation action_masks must be a dictionary")
    positions = action_masks.get("positions")
    if positions is None:
        return
    sources = torch.as_tensor(positions, device=labels.device, dtype=torch.long) - 1
    sources = sources[sources >= 0]
    if not sources.numel():
        return
    labels_flat = labels.reshape(-1)
    weights_flat = weights.reshape(-1).to(labels.device)
    if int(sources.max().item()) >= labels_flat.numel():
        raise ValueError("grouped distillation action-mask source positions exceed label width")
    invalid = (labels_flat.index_select(0, sources) == -100) & (weights_flat.index_select(0, sources) > 0)
    if bool(invalid.any().item()):
        raise ValueError(
            "grouped-distillation weights must be zero at action-masked positions whose label is "
            "IGNORE_INDEX (-100); otherwise the model head drops a constraint used by the objective"
        )


def _validate_teacher_context(context: dict, weights: torch.Tensor) -> None:
    token_ids = _signed_teacher_token_ids(context, weights)
    teacher_log_probs = _require_tensor(context, "teacher_log_probs")
    teacher_tail_log_prob = _require_tensor(context, "teacher_tail_log_prob")
    for name, values in (
        ("teacher_log_probs", teacher_log_probs),
        ("teacher_tail_log_prob", teacher_tail_log_prob),
    ):
        if values.dtype not in _TEACHER_LOG_PROB_DTYPES:
            raise ValueError(
                f"{name} must use a real floating-point dtype "
                f"(float16, bfloat16, float32, or float64), got {values.dtype}"
            )

    if tuple(teacher_log_probs.shape) != tuple(token_ids.shape):
        raise ValueError(
            f"teacher_log_probs shape {tuple(teacher_log_probs.shape)} must match teacher_token_ids "
            f"shape {tuple(token_ids.shape)}"
        )
    if tuple(teacher_tail_log_prob.shape) != tuple(weights.shape):
        raise ValueError(
            f"teacher_tail_log_prob shape {tuple(teacher_tail_log_prob.shape)} must match "
            f"grouped-distillation weights shape {tuple(weights.shape)}"
        )

    positions = weights.numel()
    width = token_ids.shape[-1]
    flat_weights = weights.reshape(-1).to(torch.float64)
    active = flat_weights > 0
    if not bool(torch.isfinite(flat_weights).all().item()) or bool((flat_weights < 0).any().item()):
        raise ValueError("grouped-distillation weights must be finite and non-negative")
    if not bool(active.any().item()):
        return

    ids = token_ids.reshape(positions, width)[active.to(token_ids.device)]
    if bool((ids < -1).any().item()):
        raise ValueError("teacher_token_ids uses -1 for padding; values below -1 are invalid")
    sorted_ids = ids.sort(-1).values
    if width > 1 and not bool(((sorted_ids[:, 1:] != sorted_ids[:, :-1]) | (sorted_ids[:, 1:] < 0)).all().item()):
        raise ValueError("teacher_token_ids contains repeated ids at a positive-weight position")

    padding = ids < 0
    head = teacher_log_probs.reshape(positions, width)[active.to(teacher_log_probs.device)].double()
    tail = teacher_tail_log_prob.reshape(positions, 1)[active.to(teacher_tail_log_prob.device)].double()
    teacher = torch.cat([head.masked_fill(padding.to(head.device), -math.inf), tail], -1)
    mass_error = teacher.logsumexp(-1).expm1().abs()
    if not bool(torch.isfinite(mass_error).all().item()) or not bool((mass_error <= 1e-2).all().item()):
        raise ValueError(
            "teacher_log_probs and teacher_tail_log_prob must define one distribution whose masses "
            "sum to one within 1% at every positive-weight position"
        )
    _validate_action_masked_labels(context, weights)


def grouped_divergence(
    model_outputs: dict,
    context: dict,
    weights: torch.Tensor,
    divergence: Divergence,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return weighted grouped divergence and additive metric totals."""
    student = model_outputs.get("group_log_probs")
    grouped = student is not None
    student = student if grouped else model_outputs.get("logits")
    if student is None:
        raise ValueError(
            "grouped distillation requires model output 'group_log_probs' or full logits; configure a "
            "group-capable head or retain full logits for the objective"
        )

    token_ids = _signed_teacher_token_ids(context, weights)
    flat_weights = weights.reshape(-1).to(student.device, torch.float64)
    positions, width = flat_weights.numel(), token_ids.shape[-1]
    index = flat_weights.ne(0).nonzero().squeeze(-1)
    if not index.numel():
        return flat_weights.new_zeros(()), dict.fromkeys(_DISTILLATION_METRICS, 0.0)

    selected_ids = _selected_rows(token_ids, index, positions, width, "teacher_token_ids")
    padding = torch.cat(
        [selected_ids < 0, torch.zeros_like(selected_ids[:, :1], dtype=torch.bool)],
        -1,
    )
    log_teacher = torch.cat(
        [
            _selected_rows(context["teacher_log_probs"], index, positions, width, "teacher_log_probs"),
            _selected_rows(context["teacher_tail_log_prob"], index, positions, 1, "teacher_tail_log_prob"),
        ],
        -1,
    )
    if not grouped:
        invalid = selected_ids >= student.shape[-1]
        if bool(invalid.any().item()):
            first_invalid = int(selected_ids[invalid][0].item())
            raise ValueError(
                f"active teacher token id {first_invalid} is outside the student vocabulary [0, {student.shape[-1]})"
            )
        if student.numel() != positions * student.shape[-1]:
            raise ValueError(f"logits shape {tuple(student.shape)} does not cover {positions} positions")
        flat_logits = student.reshape(positions, -1)
        flat_ids = token_ids.reshape(positions, width).to(student.device)
        shard_count = max(1, math.ceil(flat_logits.numel() * 16 / _LOGPROB_SLICE_BYTES))
        slices = zip(
            flat_logits.chunk(shard_count),
            flat_ids.chunk(shard_count),
            flat_weights.eq(0).chunk(shard_count),
        )
        student = torch.cat([checkpoint(_student_groups_from_logits, *rows, use_reentrant=False) for rows in slices])
    log_student = _selected_rows(student, index, positions, width + 1, "group_log_probs")
    log_teacher, log_student = (
        value.double().masked_fill(padding.to(value.device), -math.inf) for value in (log_teacher, log_student)
    )

    teacher_mass = log_teacher.logsumexp(-1, keepdim=True)
    student_mass = log_student.detach().logsumexp(-1, keepdim=True)
    log_teacher = log_teacher - teacher_mass
    values = _divergence_values(log_teacher, log_student, divergence, beta)
    selected_weights = flat_weights[index]
    kd_sum = (selected_weights * values).sum()

    student_ok = (student_mass.expm1().abs() <= 1e-2).all() & torch.isfinite(values).all()
    stats = torch.stack(
        [
            selected_weights.sum(),
            kd_sum.detach(),
            (selected_weights * log_teacher[:, -1].exp()).sum(),
            (selected_weights * log_student[:, -1].detach().exp()).sum(),
            student_ok.double(),
        ]
    ).tolist()
    *totals, valid_student = stats
    if not valid_student:
        raise ValueError(
            "the student's candidate buckets plus tail must form one distribution with finite divergence; "
            "check out-of-vocabulary candidates and zero-mass buckets required by reverse KL"
        )
    return kd_sum, dict(zip(_DISTILLATION_METRICS, totals))


@dataclass(frozen=True)
class KDTerm:
    coef: float
    divergence: Divergence
    beta: float
    weight_sum: float
    dp_size: int


def _kd_coefficient(config: dict) -> float:
    unknown = {key for key in config if key.startswith("kd_")} - _GRPO_DISTILLATION_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown KD config keys for loss_fn 'grpo': {sorted(unknown)}")
    if "kd_coef" not in config:
        return 0.0
    raw = config["kd_coef"]
    coefficient = _finite_real(raw, "kd_coef")
    if coefficient < 0:
        raise ValueError(f"kd_coef must be a finite non-negative number, got {raw!r}")
    return coefficient


def resolve_kd_term(config: dict) -> KDTerm | None:
    coefficient = _kd_coefficient(config)
    divergence, beta = _resolve_divergence(config, "kd_")
    if coefficient == 0:
        return None
    weight_sum = config.get("kd_batch_num_tokens")
    if (
        isinstance(weight_sum, bool)
        or not isinstance(weight_sum, Real)
        or not math.isfinite(float(weight_sum))
        or float(weight_sum) < 0
    ):
        raise ValueError(
            "kd_coef > 0 requires config 'kd_batch_num_tokens' to be the finite non-negative "
            f"request-global sum of kd_mask; got kd_coef={coefficient}, kd_batch_num_tokens={weight_sum!r}"
        )
    weight_sum = float(weight_sum)
    return KDTerm(
        coefficient,
        divergence,
        beta,
        weight_sum,
        _resolve_dp_size(config.get("dp_size"), weight_sum),
    )


def _kd_term(
    model_outputs: dict,
    context: dict,
    kd: KDTerm,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    kd_sum, metrics = grouped_divergence(
        model_outputs,
        context,
        weights,
        kd.divergence,
        kd.beta,
    )
    if metrics["kd_weight_sum"] > kd.weight_sum * (1 + 1e-6):
        raise ValueError(
            f"this worker's kd_mask sums to {metrics['kd_weight_sum']}, above kd_batch_num_tokens={kd.weight_sum}"
        )
    share = kd.coef / kd.weight_sum if kd.weight_sum else 0.0
    return share * kd.dp_size * kd_sum, {
        "kd_coef": kd.coef,
        "loss_term_kd": share * metrics["kd_sum"],
        **metrics,
    }


def _grouped_distillation_config(config: dict) -> tuple[float, Divergence, float, dict]:
    unknown = set(config) - _GROUPED_DISTILLATION_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys for loss_fn 'grouped_distillation': {sorted(unknown)}")
    raw_coefficient = config.get("kd_coef", 0.25)
    coefficient = _finite_real(raw_coefficient, "kd_coef")
    if not 0 <= coefficient <= 1:
        raise ValueError(f"kd_coef must be a finite number in [0, 1], got {raw_coefficient!r}")
    divergence, beta = _resolve_divergence(config, "kd_")
    normalization = {}
    if "kd_batch_num_tokens" in config:
        normalization["batch_num_tokens"] = config["kd_batch_num_tokens"]
    if "dp_size" in config:
        normalization["dp_size"] = config["dp_size"]
    return coefficient, divergence, beta, normalization


def _request_kd_masks(request: dict) -> tuple[torch.Tensor, ...]:
    batch = request.get("batch")
    if isinstance(batch, list):
        masks = tuple(microbatch.get("kd_mask") for microbatch in batch if isinstance(microbatch, dict))
        if masks and all(torch.is_tensor(mask) for mask in masks):
            return masks
        if any(mask is not None for mask in masks):
            raise ValueError("every grouped-distillation microbatch must contain tensor 'kd_mask'")

    containers = []
    for name in ("kwargs", "batch", "context", "meta"):
        value = request.get(name)
        if isinstance(value, dict):
            containers.append(value)
    containers.append(request)
    for container in containers:
        kd_mask = container.get("kd_mask")
        if kd_mask is not None:
            return (kd_mask,)
    return ()


def _set_request_kd_weight_sum(request: dict, config: dict, *, objective: str) -> None:
    masks = _request_kd_masks(request)
    if not masks or any(not torch.is_tensor(mask) for mask in masks):
        raise ValueError(f"{objective} requires tensor context['kd_mask']")
    total = 0.0
    for kd_mask in masks:
        weights = canonicalize_loss_mask(
            kd_mask,
            kd_mask,
            objective=f"{objective} kd_mask",
            binary=False,
        )
        total += float(weights.sum(dtype=torch.float64).item())
    config["kd_batch_num_tokens"] = total


def _context(batch: dict, meta: dict) -> dict:
    return {**meta, **batch}


def _remove_packed_batch_dim(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and tensor.ndim >= 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _validation_context(context: dict) -> dict:
    if context.get("cu_seqlens") is None:
        return context
    input_ids = context.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim < 2 or input_ids.shape[0] != 1:
        return context
    return {
        **context,
        **{
            name: _remove_packed_batch_dim(context.get(name))
            for name in (
                "input_ids",
                "labels",
                "loss_mask",
                "kd_mask",
                "teacher_token_ids",
                "teacher_log_probs",
                "teacher_tail_log_prob",
            )
            if name in context
        },
    }


def _sum_metrics(worker_metrics: Sequence[dict], metrics: dict, names: Sequence[str]) -> None:
    for name in names:
        if any(name in worker for worker in worker_metrics):
            metrics[name] = sum(float(worker.get(name, 0.0)) for worker in worker_metrics)


class _GroupedLossCallbacks:
    """Callbacks shared by standalone and policy-plus-distillation losses."""

    metric_names: tuple[str, ...] = _DISTILLATION_METRICS

    def _distillation_enabled(self, config: dict) -> bool:
        raise NotImplementedError

    def _weights(self, context: dict) -> torch.Tensor:
        raise NotImplementedError

    def validation_callback(self, context: dict, config: dict) -> None:
        if not self._distillation_enabled(config):
            return
        context = _validation_context(context)
        _validate_neutral_temperature(context)
        weights = self._weights(context)
        _validate_teacher_context(context, weights)

    def model_forward_callback(
        self,
        model_kwargs: dict,
        context: dict,
        config: dict,
        output_keys: list[str],
    ) -> None:
        if not self._distillation_enabled(config) or not model_kwargs.get("dss_compute_logprobs"):
            return
        normalized = _validation_context(context)
        weights = self._weights(normalized)
        token_ids = _signed_teacher_token_ids(normalized, weights)
        group_token_ids = torch.where(
            weights.to(token_ids.device)[..., None] > 0,
            token_ids,
            token_ids.new_full((), -1),
        )
        reference = context["input_ids"]
        model_kwargs["group_token_ids"] = group_token_ids.reshape(
            *reference.shape,
            group_token_ids.shape[-1],
        )
        if "group_log_probs" not in output_keys:
            output_keys.append("group_log_probs")

    def metrics_callback(self, worker_metrics: Sequence[dict], metrics: dict) -> None:
        _sum_metrics(worker_metrics, metrics, self.metric_names)

    def output_callback(self, model_outputs: dict) -> None:
        model_outputs.pop("logits", None)
        model_outputs.pop("group_log_probs", None)


class GroupedDistillationLoss(_GroupedLossCallbacks, BaseLoss):
    """Standalone weighted NLL mixed with grouped teacher divergence."""

    name = "grouped_distillation"
    metric_names = _GROUPED_DISTILLATION_METRICS

    def _distillation_enabled(self, config: dict) -> bool:
        return _grouped_distillation_config(config)[0] > 0

    def _weights(self, context: dict) -> torch.Tensor:
        reference = context.get("input_ids")
        if not torch.is_tensor(reference):
            raise ValueError("grouped_distillation requires tensor context['input_ids']")
        kd_mask = context.get("kd_mask")
        if kd_mask is None:
            raise ValueError("grouped_distillation requires context['kd_mask']")
        return canonicalize_loss_mask(
            kd_mask,
            reference,
            objective="grouped_distillation kd_mask",
            binary=False,
        )

    def batching_callback(self, request: dict) -> None:
        processing = request.get("processing")
        if not isinstance(processing, dict):
            return
        config = processing.get("config")
        if config is None:
            config = {}
            processing["config"] = config
        if not isinstance(config, dict):
            raise ValueError("processing.config must be a dictionary")
        _grouped_distillation_config(config)
        _set_request_kd_weight_sum(
            request,
            config,
            objective="grouped_distillation",
        )

    def validation_callback(self, context: dict, config: dict) -> None:
        context = _validation_context(context)
        kd_coef, _, _, normalization = _grouped_distillation_config(config)
        weights = self._weights(context)
        labels = context.get("labels")
        if torch.is_tensor(labels):
            if tuple(labels.shape) != tuple(weights.shape):
                raise ValueError("grouped_distillation labels must match kd_mask when labels are present")
            if bool(((labels.to(weights.device) == -100) & (weights > 0)).any().item()):
                raise ValueError("grouped_distillation kd_mask must be zero where labels use IGNORE_INDEX (-100)")
        scale = resolve_global_loss_scale(context, normalization)
        _validate_global_normalization(scale, float(weights.sum(dtype=torch.float32).item()))
        if kd_coef > 0:
            _validate_neutral_temperature(context)
            _validate_teacher_context(context, weights)

    def packed_reduction_callback(
        self,
        microbatches: Sequence[dict],
        config: dict,
        loss_fn_name: str,
    ):
        del loss_fn_name
        _, _, _, normalization = _grouped_distillation_config(config)
        local_weights = []
        for microbatch in microbatches:
            weights = self._weights(microbatch)
            local_weights.append(float(weights.sum(dtype=torch.float32).item()))
        scale = resolve_global_loss_scale(microbatches[0], normalization)
        global_weight_sum, _ = _validate_global_normalization(scale, sum(local_weights))
        from .packed_reduction import additive_packed_loss_reduction
        from .packed_reduction import local_mean_packed_loss_reduction

        if global_weight_sum is None:
            return local_mean_packed_loss_reduction(local_weights)
        return additive_packed_loss_reduction(local_weights)

    def loss(
        self,
        model_outputs: dict,
        batch: dict,
        meta: dict,
        config: dict,
        device: str,
    ) -> tuple[torch.Tensor, dict]:
        del device
        kd_coef, divergence, beta, normalization = _grouped_distillation_config(config)
        context = _context(batch, meta)
        if context.get("cu_seqlens") is not None:
            context = _validation_context(context)
            model_outputs = {
                key: (
                    _remove_packed_batch_dim(value)
                    if key in {"logits", "logprobs", "group_log_probs"} and torch.is_tensor(value)
                    else value
                )
                for key, value in model_outputs.items()
            }
        logprobs = model_outputs.get("logprobs")
        if logprobs is None:
            raise ValueError(
                "grouped_distillation requires model output 'logprobs'; "
                "configure processing.post=['compute_logprobs'] or a chunked logprob head"
            )
        weights = canonicalize_loss_mask(
            context["kd_mask"],
            logprobs,
            objective="grouped_distillation kd_mask",
            binary=False,
        )
        scale = resolve_global_loss_scale(context, normalization)
        local_weight_sum_tensor = weights.sum(dtype=torch.float32)
        local_weight_sum = float(local_weight_sum_tensor.detach().item())
        global_weight_sum, dp_size = _validate_global_normalization(scale, local_weight_sum)
        active = weights > 0
        if bool((active & ~torch.isfinite(logprobs)).any().item()):
            raise ValueError("grouped_distillation found non-finite logprobs at a positive-weight position")
        safe_logprobs = logprobs.float().masked_fill(~active, 0.0)
        nll_sum = -(safe_logprobs * weights).sum()
        if global_weight_sum is None:
            nll = nll_sum / local_weight_sum_tensor if bool(active.any().item()) else _connected_zero(logprobs)
        elif global_weight_sum == 0:
            nll = _connected_zero(logprobs)
        else:
            nll = nll_sum * dp_size / global_weight_sum
        if kd_coef == 0:
            return nll, {}

        kd_sum, metrics = grouped_divergence(model_outputs, context, weights, divergence, beta)
        if global_weight_sum is None:
            denominator = weights.sum(dtype=torch.float64)
            kd = kd_sum / denominator.clamp_min(torch.finfo(torch.float64).tiny)
        elif global_weight_sum == 0:
            kd = kd_sum * 0
        else:
            kd = kd_sum * dp_size / global_weight_sum
        metrics["sft_nll_sum"] = float(nll_sum.detach())
        return ((1 - kd_coef) * nll + kd_coef * kd).to(nll.dtype), metrics


class GRPOGroupedDistillationLoss(_GroupedLossCallbacks, BaseLoss):
    """Existing ``grpo`` policy behavior plus an optional grouped KD term."""

    name = "grpo"
    metric_names = _GRPO_DISTILLATION_METRICS

    def _distillation_enabled(self, config: dict) -> bool:
        return _kd_coefficient(config) > 0

    def _weights(self, context: dict) -> torch.Tensor:
        kd_mask = _require_tensor(context, "kd_mask")
        reference = context.get("input_ids")
        if not torch.is_tensor(reference):
            raise ValueError("grpo grouped distillation requires tensor context['input_ids']")
        return canonicalize_loss_mask(
            kd_mask,
            reference,
            objective="grpo kd_mask",
            binary=False,
        )

    def batching_callback(self, request: dict) -> None:
        processing = request.get("processing")
        if not isinstance(processing, dict):
            return
        config = processing.get("config")
        if config is None:
            config = {}
            processing["config"] = config
        if not isinstance(config, dict):
            raise ValueError("processing.config must be a dictionary")
        if not self._distillation_enabled(config):
            return
        _set_request_kd_weight_sum(
            request,
            config,
            objective="kd_coef > 0",
        )

    def validation_callback(self, context: dict, config: dict) -> None:
        kd = resolve_kd_term(config)
        if kd is None:
            return
        context = _validation_context(context)
        super().validation_callback(context, config)
        local_weight_sum = float(self._weights(context).sum(dtype=torch.float64).item())
        if local_weight_sum > kd.weight_sum and not math.isclose(
            local_weight_sum, kd.weight_sum, rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError(
                f"this worker's kd_mask sum {local_weight_sum} exceeds kd_batch_num_tokens={kd.weight_sum}"
            )

    def packed_reduction_callback(
        self,
        microbatches: Sequence[dict],
        config: dict,
        loss_fn_name: str,
    ):
        legacy = LOSS_FNS[self.name]
        resolver = getattr(legacy, "_arctic_packed_loss_reduction")
        reduction = resolver(microbatches, config, loss_fn_name)
        if len(microbatches) > 1 and not reduction.loss_is_additive and resolve_kd_term(config) is not None:
            raise ValueError(
                "kd_coef > 0 across multiple packed microbatches requires an additive policy reduction "
                "(for example, batch_num_tokens with token-mean)"
            )
        return reduction

    def loss(
        self,
        model_outputs: dict,
        batch: dict,
        meta: dict,
        config: dict,
        device: str,
    ) -> tuple[torch.Tensor, dict]:
        loss, metrics = LOSS_FNS[self.name](model_outputs, batch, meta, config, device)
        kd = resolve_kd_term(config)
        if kd is None:
            return loss, metrics
        context = _validation_context(_context(batch, meta))
        kd_loss, kd_metrics = _kd_term(model_outputs, context, kd, self._weights(context))
        return loss + kd_loss.to(loss.dtype), {**metrics, **kd_metrics}
