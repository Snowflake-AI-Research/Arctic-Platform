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
"""Build the Cortex job a Tinker client trains and samples on.

Tinker has no verb for GPU count, ZeRO, LoRA targets, or sequence isolation.
:class:`TinkerJobConfig` holds those choices. :func:`client_config` turns one
into the Cortex client config. :func:`isolation` decides whether each
micro-batch holds a single sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

__all__ = ["TinkerJobConfig", "client_config", "isolation"]


@dataclass
class TinkerJobConfig:
    # None reads the connection from ARCTIC_CORTEX_* instead, which is how the
    # other Cortex integrations are configured.
    config: str | None = None
    model: str = "Qwen/Qwen3-0.6B"
    training_gpus: int = 1
    sampling_gpus: int = 1
    max_prompt_length: int = 512
    max_response_length: int = 512
    learning_rate: float = 1e-6
    # DeepSpeed needs a batch size at provisioning time; Tinker has no verb that
    # declares one. These only have to satisfy DeepSpeed's own invariant, since
    # Cortex chunks each forward-backward to fit whatever actually arrives.
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dtype: str = "bfloat16"
    seed: int = 7
    # The Cortex image ships FA3 only; FA2 dies at model load.
    attn_implementation: str = "flash_attention_3"
    # auto | on | off. Keep Cortex from packing two sequences into one
    # micro-batch; `auto` turns it on for models with linear-attention layers,
    # whose state Cortex leaks across a pack (see CortexTinkerBackend). It
    # provisions max_tokens_per_mb = max_seq_len, overriding the flag below.
    isolate_sequences: str = "auto"
    max_tokens_per_mb: int = 8192
    gpu_memory_utilization: float = 0.8
    zero_stage: int = 2
    # 0 is full fine-tuning. Otherwise a LoRA adapter shaped like Tinker's:
    # alpha 32 scaled by alpha/rank, on the comma-separated module groups of
    # `_LORA_MODULE_GROUPS` -- Tinker's train_mlp / train_attn / train_unembed.
    lora_rank: int = 0
    lora_alpha: int = 32
    lora_modules: str = "mlp,attn,unembed"
    # Adam is provisioned once; only the learning rate varies per step. The
    # defaults are what the cookbook's RL and SL loops send.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    # 0 disables clipping, as in Tinker's AdamParams.
    grad_clip_norm: float = 0.0

    @property
    def max_seq_len(self) -> int:
        return self.max_prompt_length + self.max_response_length

    @property
    def fixed_adam(self) -> dict[str, float]:
        return {
            "beta1": self.adam_beta1,
            "beta2": self.adam_beta2,
            "eps": self.adam_eps,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }


# Module names per Tinker LoRA group, covering Qwen3 / Qwen3.5 (including its
# linear-attention layers) and Llama. PEFT matches each by name suffix.
_LORA_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attn": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
    ),
    "unembed": ("lm_head",),
}


def _lora_groups(cfg: TinkerJobConfig) -> list[str]:
    groups = [g.strip() for g in cfg.lora_modules.split(",") if g.strip()]
    unknown = sorted(set(groups) - set(_LORA_MODULE_GROUPS))
    if unknown or not groups:
        raise ValueError(f"--lora-modules must name some of {sorted(_LORA_MODULE_GROUPS)}, got {cfg.lora_modules!r}")
    return groups


def _peft_config(cfg: TinkerJobConfig) -> dict[str, Any] | None:
    if cfg.lora_rank <= 0:
        return None
    return {
        "peft_type": "Lora",
        "r": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [m for g in _lora_groups(cfg) for m in _LORA_MODULE_GROUPS[g]],
    }


def _has_linear_attention(model_config: Any) -> bool:
    text_config = getattr(model_config, "text_config", None) or model_config
    return "linear_attention" in (getattr(text_config, "layer_types", None) or [])


def isolation(cfg: TinkerJobConfig, model_config: Any) -> tuple[TinkerJobConfig, int | None]:
    """The config to provision and the backend's ``isolate_capacity``.

    Isolating caps a micro-batch at one full-length sequence, so any two rows
    lengthened past half of it cannot share one.
    """
    if cfg.isolate_sequences not in ("auto", "on", "off"):
        raise ValueError(f"--isolate-sequences must be auto, on or off, got {cfg.isolate_sequences!r}")
    isolate = cfg.isolate_sequences == "on" or (
        cfg.isolate_sequences == "auto" and _has_linear_attention(model_config)
    )
    if not isolate:
        return cfg, None
    return replace(cfg, max_tokens_per_mb=cfg.max_seq_len), cfg.max_seq_len


def client_config(cfg: TinkerJobConfig) -> Any:
    from arctic_platform.client import ArcticClientConfig
    from arctic_platform.client import CortexConfig
    from arctic_platform.client import SamplingConfig
    from arctic_platform.client import TrainingConfig

    if cfg.config:
        parsed = json.loads(Path(cfg.config).expanduser().read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"connection config {cfg.config} must be a JSON object")
        connection = parsed.get("connection", parsed)
        backend_keys = ("base_url", "host", "pat", "database", "schema", "endpoint", "max_retries")
        backend = CortexConfig(**{key: connection[key] for key in backend_keys if key in connection})
    else:
        backend = CortexConfig()

    return ArcticClientConfig(
        backend=backend,
        model_name=cfg.model,
        max_seq_len=cfg.max_seq_len,
        seed=cfg.seed,
        dtype=cfg.dtype,
        training_gpus=cfg.training_gpus,
        sampling_gpus=cfg.sampling_gpus,
        job_ready_timeout=3600.0,
        training=TrainingConfig(
            ds_config={
                "train_batch_size": cfg.micro_batch_size * cfg.training_gpus * cfg.gradient_accumulation_steps,
                "train_micro_batch_size_per_gpu": cfg.micro_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "bf16": {"enabled": cfg.dtype == "bfloat16"},
                "zero_optimization": {"stage": cfg.zero_stage},
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": cfg.learning_rate,
                        "betas": [cfg.adam_beta1, cfg.adam_beta2],
                        "eps": cfg.adam_eps,
                        "weight_decay": cfg.weight_decay,
                    },
                },
                "gradient_clipping": cfg.grad_clip_norm,
            },
            ds_worker_config={
                "attn_implementation": cfg.attn_implementation,
                "model_provider": "huggingface",
                "mb_spec": {"max_tokens_per_mb": cfg.max_tokens_per_mb},
            },
            peft=_peft_config(cfg),
        ),
        sampling=SamplingConfig(vllm={"gpu_memory_utilization": cfg.gpu_memory_utilization}),
    )
