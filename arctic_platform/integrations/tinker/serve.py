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
"""Provision a Cortex job for the in-process Tinker client.

Recipes enter through ``python -m arctic_platform.tinker.run``. That module
builds a :class:`TinkerServeConfig` from ``--training-gpus`` and
``--sampling-gpus`` and calls :func:`_client_config`. ``TINKER_BASE_URL`` is
not part of that path.

Provisioning is not expressible in Tinker's protocol -- there is no verb for
"give me four GPUs with ZeRO-2 and FA3" -- so the job is created here from
those counts and the Tinker calls are bound onto it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from arctic_platform.integrations.tinker.cortex import CortexTinkerBackend
from arctic_platform.integrations.tinker.cortex import build_handlers
from arctic_platform.integrations.tinker.router import init_tinker_state
from arctic_platform.integrations.tinker.router import router as tinker_router

logger = logging.getLogger(__name__)

__all__ = ["TinkerServeConfig", "create_app", "main"]


@dataclass
class TinkerServeConfig:
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
    job_id: str | None = None
    # On-policy distillation's teacher: served from its base weights by a
    # sampling-only job of its own, created and released with this server.
    teacher_model: str | None = None
    teacher_sampling_gpus: int = 1
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
    host: str = "127.0.0.1"
    port: int = 8000

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


def _lora_groups(cfg: TinkerServeConfig) -> list[str]:
    groups = [g.strip() for g in cfg.lora_modules.split(",") if g.strip()]
    unknown = sorted(set(groups) - set(_LORA_MODULE_GROUPS))
    if unknown or not groups:
        raise ValueError(f"--lora-modules must name some of {sorted(_LORA_MODULE_GROUPS)}, got {cfg.lora_modules!r}")
    return groups


def _peft_config(cfg: TinkerServeConfig) -> dict[str, Any] | None:
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


def _served_lora(cfg: TinkerServeConfig) -> Any:
    from arctic_platform.integrations.tinker.router import LoraConfig

    if cfg.lora_rank <= 0:
        return None
    groups = set(_lora_groups(cfg))
    return LoraConfig(
        rank=cfg.lora_rank,
        train_mlp="mlp" in groups,
        train_attn="attn" in groups,
        train_unembed="unembed" in groups,
    )


def _has_linear_attention(model_config: Any) -> bool:
    text_config = getattr(model_config, "text_config", None) or model_config
    return "linear_attention" in (getattr(text_config, "layer_types", None) or [])


def _isolation(cfg: TinkerServeConfig, model_config: Any) -> tuple[TinkerServeConfig, int | None]:
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


def _client_config(cfg: TinkerServeConfig) -> Any:
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

    job_ids = {}
    if cfg.job_id is not None:
        job_ids = {
            "training_job_id": f"{cfg.job_id}:training:0",
            "sampling_job_id": f"{cfg.job_id}:sampling:0",
        }

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
        **job_ids,
    )


def _teacher_config(cfg: TinkerServeConfig) -> Any:
    # The teacher scores a whole student sequence and then has to sample one
    # token to do it, so it needs one position more than the student.
    return _client_config(
        replace(
            cfg,
            model=cfg.teacher_model,
            training_gpus=0,
            sampling_gpus=cfg.teacher_sampling_gpus,
            max_response_length=cfg.max_response_length + 1,
            job_id=None,
            lora_rank=0,
        )
    )


def create_app(cfg: TinkerServeConfig):
    """A FastAPI app serving Tinker's protocol, bound to a Cortex job.

    The job is created on startup and released on shutdown unless ``job_id``
    attached this server to someone else's, in which case it is left running.
    """
    from fastapi import FastAPI

    from arctic_platform.client import AsyncArcticRLClient

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        from transformers import AutoConfig
        from transformers import AutoTokenizer

        job_cfg, isolate_capacity = _isolation(cfg, AutoConfig.from_pretrained(cfg.model))
        if isolate_capacity is not None:
            logger.info("one sequence per micro-batch of %d tokens (linear attention)", isolate_capacity)
        client_cfg = _client_config(job_cfg)
        attached = client_cfg.training_job_id is not None
        client = AsyncArcticRLClient(client_cfg)
        logger.info("training job %s is running", client.jobs.training)
        teacher = None
        try:
            teacher_handlers = {}
            if cfg.teacher_model:
                teacher = AsyncArcticRLClient(_teacher_config(cfg))
                logger.info("teacher %s is running as job %s", cfg.teacher_model, teacher.jobs.sampling)
                teacher_handlers[cfg.teacher_model] = CortexTinkerBackend(teacher).generate

            tokenizer = AutoTokenizer.from_pretrained(cfg.model)
            init_tinker_state(
                app,
                base_model=cfg.model,
                max_prompt_length=cfg.max_prompt_length,
                max_response_length=cfg.max_response_length,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                # Cortex registers no `apply_temperature`, so the trainer always
                # scores at 1.0 and `sample` refuses any other temperature.
                supports_temperature_scaling=False,
                teacher_generate_handlers=teacher_handlers,
                lora=_served_lora(cfg),
                fixed_adam=cfg.fixed_adam,
                **build_handlers(client, isolate_capacity=isolate_capacity),
            )
            app.state.arctic_client = client
            yield
        finally:
            if teacher is not None:
                logger.info("releasing teacher job %s", teacher.jobs.sampling)
                await teacher.shutdown()
            if attached:
                logger.info("leaving pre-existing job %s running", client.jobs.training)
            else:
                logger.info("releasing job %s", client.jobs.training)
                await client.shutdown()

    app = FastAPI(title="Tinker over Cortex Training", lifespan=lifespan)
    app.include_router(tinker_router)
    return app


def _parse_args(argv: list[str] | None = None) -> TinkerServeConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="Cortex connection JSON; default reads ARCTIC_CORTEX_*")
    defaults = TinkerServeConfig()
    for name, value in vars(defaults).items():
        if name == "config":
            continue
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            p.add_argument(flag, action="store_true", default=value)
        else:
            p.add_argument(flag, type=type(value) if value is not None else str, default=value)
    return TinkerServeConfig(**vars(p.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = _parse_args(argv)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
