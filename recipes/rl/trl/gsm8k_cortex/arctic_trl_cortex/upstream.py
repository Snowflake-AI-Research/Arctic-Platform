"""Upstream-TRL bridge for the CPU-only Cortex GSM8K recipe.

Upstream ``AsyncGRPOTrainer`` owns the policy loss.  This module replaces only
the model forward: Cortex supplies log-probabilities and the autograd backward
hook submits the corresponding server-side GRPO backward.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

from .client import ArcticOptimizer
from .client import _meta_dict
from .client import _repack_to_row
from .client import _segment_lengths
from .client import _unpack_to_padded
from .client import _unpack_to_padded_rows
from .cortex_forward import normalize_forward_result
from .cortex_forward import to_cortex_fwd_payload
from .cortex_fwd_bwd import to_cortex_fwd_bwd_payload


class _CortexBackward(torch.autograd.Function):
    """Make Cortex's remote backward part of the local autograd graph."""

    @staticmethod
    def forward(ctx: Any, anchor: torch.Tensor, values: torch.Tensor, bridge: "CortexModelBridge") -> torch.Tensor:
        ctx.bridge = bridge
        return values

    @staticmethod
    def backward(ctx: Any, grad_values: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        # The server owns the model and applies its GRPO loss.  The upstream
        # loss still drives logging, but its local derivative must not change
        # the server update from the validated Cortex recipe.
        del grad_values
        ctx.bridge.backward()
        return torch.zeros_like(ctx.bridge.anchor), None, None


class CortexModelBridge:
    """Translate upstream's flat packed row into Cortex forward/backward RPCs."""

    def __init__(
        self,
        client: Any,
        *,
        temperature: float,
        pad_token_id: int,
        rollout_n: int,
        max_token_len_per_gpu: int,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.pad_token_id = pad_token_id
        self.rollout_n = rollout_n
        self.max_token_len_per_gpu = max_token_len_per_gpu
        self.anchor: torch.Tensor | None = None
        self._pending: tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float] | None = None

    def set_anchor(self, anchor: torch.Tensor) -> None:
        self.anchor = anchor

    def stage(self, inputs: dict[str, torch.Tensor], epsilon: float) -> None:
        """Capture advantages and target positions before upstream calls model.forward."""
        if self._pending is not None:
            raise RuntimeError("Cortex forward was staged twice before its backward completed.")
        kept = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"][kept].unsqueeze(0)
        position_ids = inputs["position_ids"][kept].unsqueeze(0)
        seq_lens = _segment_lengths(position_ids)
        completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        advantages = torch.zeros_like(input_ids, dtype=torch.float32)

        token_index = inputs["token_index"][inputs["token_index"] >= 0].to(torch.long)
        selected_advantages = inputs["advantages"][inputs["token_index"] >= 0].to(torch.float32)
        target_index = token_index + 1
        if target_index.numel() and int(target_index.max()) >= input_ids.numel():
            raise ValueError("A selected TRL token_index points beyond the flattened Cortex row.")
        completion_mask.reshape(-1)[target_index] = True
        advantages.reshape(-1)[target_index] = selected_advantages

        batch = _unpack_to_padded_rows(input_ids, position_ids, completion_mask, seq_lens)
        response_mask = _unpack_to_padded(completion_mask, seq_lens).bool()
        advantages = _unpack_to_padded(advantages, seq_lens)
        self._pending = (batch, advantages, response_mask, epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        shift_labels: torch.Tensor | None = None,
        support_ids: torch.Tensor | None = None,
        score_ids: torch.Tensor | None = None,
        fused_lm_head: bool = False,
        **tree_kwargs: Any,
    ) -> SimpleNamespace:
        del shift_labels, tree_kwargs
        if not fused_lm_head:
            raise ValueError("The Cortex bridge requires upstream's fused log-probability forward.")
        if support_ids is not None or score_ids is not None:
            raise ValueError("Cortex recipe requires score_centering=False and unrestricted sampling.")
        if self._pending is None:
            raise RuntimeError("Cortex model.forward was called without staged training inputs.")
        if self.anchor is None:
            raise RuntimeError("Cortex model bridge has no autograd anchor.")

        batch, _advantages, _response_mask, _epsilon = self._pending
        expected = int(batch["attention_mask"].sum())
        if input_ids.numel() != expected or position_ids.numel() != expected:
            raise ValueError("Upstream model inputs no longer match the staged Cortex batch.")
        response = normalize_forward_result(
            self.client.fwd_no_grad(
                to_cortex_fwd_payload(
                    {
                        "batch": batch,
                        "meta": _meta_dict(
                            temperature=self.temperature,
                            rollout_n=self.rollout_n,
                            pad_token_id=self.pad_token_id,
                            max_token_len_per_gpu=self.max_token_len_per_gpu,
                            calculate_entropy=True,
                        ),
                    }
                )
            )
        )
        out = response["batch"]
        seq_lens = batch["attention_mask"].sum(dim=1).to(torch.long)
        log_probs = _repack_to_row(torch.as_tensor(out["logprobs"]), seq_lens).to(input_ids.device)
        entropy = _repack_to_row(torch.as_tensor(out["entropy"]), seq_lens).to(input_ids.device)
        # ``token_index`` addresses a predictor position. Cortex already
        # returns its next-token logprob at that predictor position.
        log_probs = _CortexBackward.apply(self.anchor, log_probs, self)
        return SimpleNamespace(log_probs=log_probs, entropy=entropy, aux_loss=None)

    def backward(self) -> None:
        if self._pending is None:
            raise RuntimeError("Cortex backward ran without a staged forward.")
        batch, advantages, response_mask, epsilon = self._pending
        self._pending = None
        self.client.fwd_bwd(
            to_cortex_fwd_bwd_payload(
                {
                    "batch": {**batch, "advantages": advantages, "response_mask": response_mask},
                    "meta": _meta_dict(
                        temperature=self.temperature,
                        rollout_n=self.rollout_n,
                        pad_token_id=self.pad_token_id,
                        max_token_len_per_gpu=self.max_token_len_per_gpu,
                        calculate_entropy=True,
                    ),
                    "processing": {
                        "config": {
                            "eps_clip": epsilon,
                            "loss_agg_mode": "token-mean",
                            "entropy_coeff": 0.0,
                        }
                    },
                }
            )
        )


class FlatRolloutCollator:
    """Use upstream's queue/planner but send Cortex one complete row per sample."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, examples: list[Any]) -> dict[str, torch.Tensor]:
        # Retain upstream's group/reward accounting and queue bookkeeping.
        self.inner.torch_call(examples)
        (rows,) = examples
        flat = [_flatten_row(row, self.inner.pad_token_id) for row in rows]
        return _pad_rows(flat, self.inner.num_processes, self.inner.pad_token_id)


def _flatten_row(row: Any, pad_token_id: int) -> dict[str, torch.Tensor]:
    ids: list[int] = []
    positions: list[int] = []
    completion: list[int] = []
    labels: list[int] = []
    token_index: list[int] = []
    old_log_probs: list[float] = []
    advantages: list[float] = []
    segment_id: list[int] = []

    for segment, sample in enumerate(row.samples):
        sample_ids = sample["input_ids"].tolist()
        sample_mask = sample["completion_mask"].tolist()
        sample_old = sample["old_log_probs"].tolist()
        offset = len(ids)
        ids.extend(int(token) for token in sample_ids)
        positions.extend(range(len(sample_ids)))
        completion.extend(int(mask) for mask in sample_mask)
        labels.extend([-100] * len(sample_ids))
        for position in range(1, len(sample_ids)):
            if sample_mask[position]:
                predictor = offset + position - 1
                labels[predictor] = int(sample_ids[position])
                token_index.append(predictor)
                old_log_probs.append(float(sample_old[position]))
                advantages.append(float(sample["advantage"]))
                segment_id.append(segment)

    n_tokens = len(ids)
    if not n_tokens:
        raise ValueError("Cortex GRPO received an empty rollout row.")
    # These satisfy unpad_tree(), but the replacement forward intentionally
    # does not use the tree: Cortex receives one padded sequence per sample.
    chunks = math.ceil(n_tokens / 64)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "position_ids": torch.tensor(positions, dtype=torch.long),
        "completion_mask": torch.tensor(completion, dtype=torch.long),
        "shift_labels": torch.tensor(labels, dtype=torch.long),
        "token_index": torch.tensor(token_index, dtype=torch.long),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "segment_id": torch.tensor(segment_id, dtype=torch.long),
        "sampling_mask": torch.empty(0, dtype=torch.long),
        "sampling_mask_counts": torch.zeros(n_tokens, dtype=torch.long),
        "sampler_ids": torch.empty((n_tokens, 0), dtype=torch.long),
        "sampler_log_probs": torch.empty((n_tokens, 0), dtype=torch.float32),
        "tree_kv_index": torch.arange(n_tokens, dtype=torch.long),
        "tree_cu_seqlens_q": torch.tensor([0, n_tokens], dtype=torch.int32),
        "tree_cu_seqlens_k": torch.tensor([0, n_tokens], dtype=torch.int32),
        "tree_segment_parents": torch.tensor([-1], dtype=torch.int32),
        "tree_segment_depths": torch.tensor([0], dtype=torch.int32),
        "tree_sizes": torch.tensor([n_tokens, n_tokens, n_tokens, chunks, 1], dtype=torch.long),
        "global_n_tokens": torch.tensor(float(len(token_index))),
        "global_n_forward_tokens": torch.tensor(float(n_tokens)),
        "mean_seq_len": torch.tensor(float(n_tokens) / len(row.samples)),
    }


def _pad_rows(rows: list[dict[str, torch.Tensor]], num_processes: int, pad_token_id: int) -> dict[str, torch.Tensor]:
    if len(rows) != num_processes:
        raise ValueError(f"Expected {num_processes} rollout rows, got {len(rows)}.")
    max_tokens = max(int(row["input_ids"].numel()) for row in rows)
    padded: dict[str, list[torch.Tensor]] = {key: [] for key in rows[0]}
    max_lengths = {
        key: max(int(row[key].numel()) for row in rows)
        for key, value in rows[0].items()
        if value.ndim == 1 and key not in {"global_n_tokens", "global_n_forward_tokens", "mean_seq_len"}
    }
    pad_values = {
        "input_ids": pad_token_id,
        "shift_labels": -100,
        "token_index": -1,
        "sampler_ids": -1,
        "sampler_log_probs": -math.inf,
    }
    for row in rows:
        n_tokens = int(row["input_ids"].numel())
        for key, value in row.items():
            if key in {"global_n_tokens", "global_n_forward_tokens", "mean_seq_len"}:
                padded[key].append(value)
                continue
            if value.ndim == 1:
                fill = pad_values.get(key, 0)
                tail = torch.full((max_lengths[key] - value.numel(),), fill, dtype=value.dtype)
                padded[key].append(torch.cat([value, tail]))
            elif key in {"sampler_ids", "sampler_log_probs"}:
                fill = pad_values[key]
                tail = torch.full((max_tokens - value.shape[0], value.shape[1]), fill, dtype=value.dtype)
                padded[key].append(torch.cat([value, tail]))
            else:
                padded[key].append(value)
        row["attention_mask"] = torch.cat(
            [torch.ones(n_tokens, dtype=torch.long), torch.zeros(max_tokens - n_tokens, dtype=torch.long)]
        )
    batch = {key: torch.stack(values) for key, values in padded.items()}
    batch["attention_mask"] = torch.stack([row["attention_mask"] for row in rows])
    # The trainer expects global values repeated once per rank before the
    # dispatcher slices the batch.
    for key in ("global_n_tokens", "global_n_forward_tokens", "mean_seq_len"):
        batch[key] = torch.full((num_processes,), float(batch[key].sum()))
    return batch


def make_trainer_class() -> type:
    """Build lazily so importing this recipe does not import TRL at install time."""
    from trl.experimental.async_grpo import AsyncGRPOTrainer
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
    from trl.experimental.async_grpo.async_grpo_trainer import FixedCountBatcher
    from trl.experimental.async_grpo.async_grpo_trainer import RolloutQueueDataset
    from trl.experimental.async_grpo.async_grpo_trainer import TokenBudgetBatcher
    from trl.experimental.async_grpo.async_grpo_trainer import _EmptyIterableDataset
    import trl.experimental.async_grpo.async_grpo_trainer as trainer_module

    class CortexAsyncGRPOTrainer(AsyncGRPOTrainer):
        def __init__(self, *args: Any, arctic_client: Any, cortex_bridge: CortexModelBridge, **kwargs: Any) -> None:
            self._arctic_client = arctic_client
            self._cortex_bridge = cortex_bridge
            original_peak_flops = trainer_module.get_peak_flops_per_device

            def cpu_safe_peak_flops(accelerator: Any, dtype: str) -> float | None:
                device = getattr(accelerator, "device", None)
                if device is None or getattr(device, "type", None) != "cuda":
                    return None
                return original_peak_flops(accelerator, dtype)

            trainer_module.get_peak_flops_per_device = cpu_safe_peak_flops
            try:
                super().__init__(*args, **kwargs)
            finally:
                trainer_module.get_peak_flops_per_device = original_peak_flops
            anchor = next(self.model.parameters())
            self._cortex_bridge.set_anchor(anchor)
            self.model.forward = self._cortex_bridge.forward

        def create_optimizer(self, *args: Any, **kwargs: Any) -> ArcticOptimizer:
            del args, kwargs
            if self.optimizer is None:
                self.optimizer = ArcticOptimizer(
                    self._arctic_client, self.model.parameters(), lr=self.args.learning_rate
                )
            return self.optimizer

        def get_train_dataloader(self) -> DataLoader:
            num_processes = self.accelerator.num_processes
            if self.accelerator.is_main_process:
                dataset = RolloutQueueDataset(
                    rollout_queue=self.rollout_queue,
                    model_version_fn=lambda: self.model_version,
                    check_health_fn=self.rollout_worker.check_health,
                    stale_after_s=self.args.heartbeat_stale_after_s,
                    metrics=self._metrics["train"],
                    dropped_groups=self._dropped_groups,
                    max_staleness=self.args.max_staleness,
                    report_to=self.args.report_to,
                )
                self._rollout_dataset = dataset
                if self.args.token_budget is None:
                    raise ValueError("token_budget must be set for the Cortex recipe.")
                if self.args.token_budget > 0:
                    dataset = TokenBudgetBatcher(
                        dataset, num_processes, self.args.token_budget, self._metrics["train"]
                    )
                else:
                    dataset = FixedCountBatcher(
                        dataset, num_processes, self.args.per_device_train_batch_size * num_processes
                    )
            else:
                dataset = _EmptyIterableDataset()

            collator = FlatRolloutCollator(
                DataCollatorForRollout(
                    self.processing_class.pad_token_id,
                    num_processes,
                    groups_trained=self._trained_groups,
                    metrics=self._metrics["train"],
                    token_budget=max(self.args.token_budget or 0, 0),
                )
            )
            return self.accelerator.prepare(DataLoader(dataset, batch_size=1, collate_fn=collator, num_workers=0))

        def _set_signature_columns_if_needed(self) -> None:
            if self._signature_columns is None:
                self._signature_columns = [
                    "input_ids", "attention_mask", "position_ids", "completion_mask", "shift_labels",
                    "token_index", "old_log_probs", "advantages", "segment_id", "sampling_mask",
                    "sampling_mask_counts", "sampler_ids", "sampler_log_probs", "tree_kv_index",
                    "tree_cu_seqlens_q", "tree_cu_seqlens_k", "tree_segment_parents",
                    "tree_segment_depths", "tree_sizes", "global_n_tokens", "global_n_forward_tokens",
                    "mean_seq_len",
                ]

        def compute_loss(self, model: Any, inputs: dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
            del model
            self._cortex_bridge.stage(inputs, float(self.epsilon_low))
            return super().compute_loss(self.model, inputs, *args, **kwargs)

    return CortexAsyncGRPOTrainer
