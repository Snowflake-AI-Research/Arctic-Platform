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

"""TRL ``AsyncGRPOTrainer`` wired to an Arctic server.

The trainer process holds a one-layer ``PreTrainedModel`` so TRL can initialize.
Forwards go to Arctic. Rows stay packed ``[1, T]``; shared prefixes are not
folded into a tree on this side.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from arctic_platform.integrations.trl.client import ArcticOptimizer


def build_stub_model(config: PretrainedConfig) -> PreTrainedModel:
    """One-layer fp32 model with the original vocabulary, hidden size, and model id.

    The full config is kept on ``model._arctic_full_config`` so checkpoints do not
    record the one-layer stub.
    """
    full = copy.deepcopy(config)
    stub_config = copy.deepcopy(config)
    text = stub_config.get_text_config() if hasattr(stub_config, "get_text_config") else stub_config
    if hasattr(text, "num_hidden_layers"):
        text.num_hidden_layers = 1
    model_id = _model_id(full) or _model_id(text)
    model = AutoModelForCausalLM.from_config(stub_config, dtype=torch.float32)
    model._arctic_full_config = full
    if model_id:
        _set_model_id(model.config, model_id)
        _set_model_id(full, model_id)
        inner = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        if inner is not model.config:
            _set_model_id(inner, model_id)
    return model


def _model_id(config: Any) -> str | None:
    for name in ("name_or_path", "_name_or_path"):
        value = getattr(config, name, None)
        if value:
            return str(value)
    return None


def _set_model_id(config: Any, model_id: str) -> None:
    config._name_or_path = model_id
    try:
        config.name_or_path = model_id
    except AttributeError:
        pass


class _FlatRolloutCollator:
    """Call TRL's tree collator for bookkeeping, then pack each row's raw samples."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, examples: list[Any]) -> dict[str, torch.Tensor]:
        tree_batch = self.inner.torch_call(examples)
        (rows,) = examples
        flat = [_flatten_row(row) for row in rows]
        width = max((int(row["input_ids"].numel()) for row in flat), default=0)
        batch = _pad_rows(flat, width, int(self.inner.pad_token_id))
        n_forward = sum(len(sample["input_ids"]) for row in rows for sample in row.samples)
        n_samples = sum(len(row.samples) for row in rows)
        repeats = int(self.inner.num_processes)
        batch["global_n_tokens"] = tree_batch["global_n_tokens"]
        batch["global_n_forward_tokens"] = torch.full((repeats,), float(n_forward))
        batch["mean_seq_len"] = torch.full((repeats,), float(n_forward) / max(n_samples, 1))
        return batch


def _as_list(value: Any) -> list:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _flatten_row(row: Any) -> dict[str, torch.Tensor]:
    ids: list[int] = []
    positions: list[int] = []
    completion: list[int] = []
    old: list[float] = []
    advantages: list[float] = []
    segments: list[int] = []
    for index, sample in enumerate(row.samples):
        tokens = [int(token) for token in _as_list(sample["input_ids"])]
        length = len(tokens)
        ids.extend(tokens)
        positions.extend(range(length))
        completion.extend(int(token) for token in _as_list(sample["completion_mask"]))
        old.extend(float(token) for token in _as_list(sample["old_log_probs"]))
        advantage = float(sample["advantage"])
        advantages.extend([advantage] * length)
        segments.extend([index] * length)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "position_ids": torch.tensor(positions, dtype=torch.long),
        "completion_mask": torch.tensor(completion, dtype=torch.long),
        "old_log_probs": torch.tensor(old, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "segment_id": torch.tensor(segments, dtype=torch.long),
    }


def _pad_rows(rows: list[dict[str, torch.Tensor]], width: int, pad_token_id: int) -> dict[str, torch.Tensor]:
    padded: dict[str, list[torch.Tensor]] = {key: [] for key in rows[0]}
    attention = []
    fills = {
        "input_ids": pad_token_id,
        "position_ids": 0,
        "completion_mask": 0,
        "old_log_probs": 0.0,
        "advantages": 0.0,
        "segment_id": -1,
    }
    dtypes = {key: rows[0][key].dtype for key in rows[0]}
    for row in rows:
        length = int(row["input_ids"].numel())
        attention.append(
            torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(width - length, dtype=torch.long)])
        )
        for key, values in row.items():
            pad_value = fills[key]
            tail = torch.full((width - length,), pad_value, dtype=dtypes[key])
            padded[key].append(torch.cat([values, tail]) if width else values)
    batch = {key: torch.stack(chunks) for key, chunks in padded.items()}
    batch["attention_mask"] = torch.stack(attention) if attention else torch.zeros((0, width), dtype=torch.long)
    return batch


_TRAINER_CLS = None


def _trainer_cls():
    global _TRAINER_CLS
    if _TRAINER_CLS is not None:
        return _TRAINER_CLS
    import trl.experimental.async_grpo.async_grpo_trainer as trainer_mod
    from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
    from trl.experimental.async_grpo.async_grpo_trainer import FixedCountBatcher
    from trl.experimental.async_grpo.async_grpo_trainer import RolloutQueueDataset
    from trl.experimental.async_grpo.async_grpo_trainer import TokenBudgetBatcher
    from trl.experimental.async_grpo.async_grpo_trainer import _EmptyIterableDataset
    from trl.trainer.utils import nanmax
    from trl.trainer.utils import nanmin

    class ArcticAsyncGRPOTrainer(AsyncGRPOTrainer):
        def __init__(self, *args: Any, arctic_client: Any, training_client: Any, **kwargs: Any) -> None:
            if kwargs.get("peft_config") is not None:
                raise ValueError("ArcticAsyncGRPOTrainer does not support peft_config.")
            grpo_args = kwargs.get("args")
            if grpo_args is not None:
                if float(getattr(grpo_args, "top_p", 1.0)) < 1.0:
                    raise ValueError("top_p < 1 is not supported without a sampling mask.")
                if int(getattr(grpo_args, "top_k", 0) or 0) > 0:
                    raise ValueError("top_k > 0 is not supported without a sampling mask.")
                min_p = getattr(grpo_args, "min_p", None)
                if min_p not in (None, 0, 0.0):
                    raise ValueError("min_p is not supported.")
            model = kwargs.get("model", args[0] if args else None)
            if isinstance(model, str):
                raise TypeError("Pass the module from build_stub_model(config); a model id would load the checkpoint.")
            self.arctic_client = arctic_client
            self.training_client = training_client
            full_config = getattr(model, "_arctic_full_config", model.config)
            self._original_config = copy.deepcopy(full_config)
            original = trainer_mod.get_peak_flops_per_device

            def _peak_flops(accelerator: Any, dtype: str):
                device = getattr(accelerator, "device", None)
                if device is None or getattr(device, "type", None) != "cuda":
                    return None
                return original(accelerator, dtype)

            trainer_mod.get_peak_flops_per_device = _peak_flops
            try:
                super().__init__(*args, **kwargs)
            finally:
                trainer_mod.get_peak_flops_per_device = original

        def create_optimizer(self, *args: Any, **kwargs: Any):
            del args, kwargs
            if self.optimizer is None:
                self.optimizer = ArcticOptimizer(
                    self.arctic_client, self.model.parameters(), lr=self.args.learning_rate
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
                    max_staleness=self.args.max_staleness,
                    report_to=self.args.report_to,
                )
                self._rollout_dataset = dataset
                if self.args.token_budget is None:
                    raise ValueError("token_budget must be set; Arctic training does not query a local vLLM server.")
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

            collator = _FlatRolloutCollator(
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
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "completion_mask",
                    "old_log_probs",
                    "advantages",
                    "segment_id",
                    "global_n_tokens",
                    "global_n_forward_tokens",
                    "mean_seq_len",
                ]

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del model, return_outputs, num_items_in_batch
            kept = inputs["attention_mask"].bool()
            input_ids = inputs["input_ids"][kept].unsqueeze(0)
            position_ids = inputs["position_ids"][kept].unsqueeze(0)
            completion_mask = inputs["completion_mask"][kept].unsqueeze(0)
            old_log_probs = inputs["old_log_probs"][kept].unsqueeze(0)
            advantages = inputs["advantages"][kept].unsqueeze(0)
            segment_id = inputs["segment_id"][kept].unsqueeze(0)

            shifted_old = old_log_probs[:, 1:]
            shifted_adv = advantages[:, 1:]
            loss_mask = completion_mask[:, 1:].to(dtype=torch.float32)
            global_n_tokens = inputs["global_n_tokens"].reshape(-1)[0]
            world_size = self.accelerator.num_processes

            forward_start = time.time()
            out = self.training_client.forward_backward(
                input_ids,
                position_ids,
                completion_mask,
                old_log_probs=shifted_old,
                advantages=shifted_adv,
                loss_mask=loss_mask,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                tokens_per_rank=global_n_tokens,
                world_size=world_size,
            )
            self._last_forward_time_s = time.time() - forward_start

            loss = out.loss
            if not self.training_client.server_side_loss:
                tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
                loss = loss / tokens_per_rank.to(dtype=torch.float32)
            # ``trl_grpo`` already divides by the token count and does not divide by GAS.
            loss = loss / self.current_gradient_accumulation_steps

            with torch.no_grad():
                selected = loss_mask.bool()
                log_probs = out.log_probs[selected]
                entropy = out.entropy[selected]
                old = shifted_old[selected]
                adv = shifted_adv[selected]
                seg = segment_id[:, 1:][selected]
                log_ratio = log_probs - old
                coef_1 = torch.exp(log_ratio)
                local_count = torch.full_like(coef_1.sum(), float(log_probs.numel()))
                stats = torch.stack(
                    [
                        coef_1.sum(),
                        ((coef_1 - 1) - log_ratio).sum(),
                        entropy.sum(),
                        ((coef_1 < 1 - self.epsilon_low) & (adv < 0)).float().sum(),
                        ((coef_1 > 1 + self.epsilon_high) & (adv > 0)).float().sum(),
                        (
                            ((coef_1 < 1 - self.epsilon_low) & (adv < 0))
                            | ((coef_1 > 1 + self.epsilon_high) & (adv > 0))
                        )
                        .float()
                        .sum(),
                        local_count,
                    ]
                )
                stats = self.accelerator.reduce(stats, reduction="sum")
                ratio, kl, ent, low, high, region, count = stats.unbind(0)
                sink = self._metrics["train"]
                sink["ratio"].append((ratio / count).item())
                sink["kl"].append((kl / count).item())
                sink["entropy"].append((ent / count).item())
                sink["clip_ratio/low_mean"].append((low / count).item())
                sink["clip_ratio/high_mean"].append((high / count).item())
                sink["clip_ratio/region_mean"].append((region / count).item())
                if seg.numel():
                    num_seq = int(seg.max()) + 1

                    def seg_sum(vals: torch.Tensor) -> torch.Tensor:
                        return torch.zeros(num_seq, device=vals.device).index_add_(0, seg, vals)

                    seq_tokens = seg_sum(torch.ones_like(coef_1))
                    per_seq_low = seg_sum(((coef_1 < 1 - self.epsilon_low) & (adv < 0)).float()) / seq_tokens
                    per_seq_high = seg_sum(((coef_1 > 1 + self.epsilon_high) & (adv > 0)).float()) / seq_tokens
                    sink["clip_ratio/low_min"].append(nanmin(self.accelerator.gather(nanmin(per_seq_low))).item())
                    sink["clip_ratio/high_max"].append(nanmax(self.accelerator.gather(nanmax(per_seq_high))).item())

            n_forward_tokens = float(inputs["global_n_forward_tokens"].reshape(-1)[0])
            mean_seq_len = float(inputs["mean_seq_len"].reshape(-1)[0])
            self._step_forward_tokens += n_forward_tokens
            self._step_trained_tokens += float(global_n_tokens)
            self._step_seq_len_weighted += mean_seq_len * n_forward_tokens
            self._step_samples += n_forward_tokens / mean_seq_len if mean_seq_len else 0.0
            self._step_forward_s += self._last_forward_time_s
            return loss

        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            del state_dict
            if not self.accelerator.is_main_process:
                return
            output_dir = output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            self.training_client.save(output_dir)
            if hasattr(self.processing_class, "save_pretrained"):
                self.processing_class.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            self._original_config.save_pretrained(output_dir)

        def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:
            del model
            self.training_client.load(resume_from_checkpoint)

    _TRAINER_CLS = ArcticAsyncGRPOTrainer
    return _TRAINER_CLS


# Rebind the exported name once TRL is importable. Callers that import the class
# through ``__getattr__`` pay the import; tests can also call ``_trainer_cls``.
def __getattr__(name: str):
    if name == "ArcticAsyncGRPOTrainer":
        return _trainer_cls()
    raise AttributeError(name)
