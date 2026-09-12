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
"""Serve Tinker's HTTP API against Cortex Training.

Run this, point ``TINKER_BASE_URL`` at it, and an unmodified ``tinker-cookbook``
recipe trains on Cortex::

    python -m arctic_platform.integrations.tinker.serve --config conn.json \\
        --model Qwen/Qwen3-0.6B --training-gpus 1 --sampling-gpus 1

Provisioning is not expressible in Tinker's protocol -- there is no verb for
"give me four GPUs with ZeRO-2 and FA3" -- so the job is created here from
flags and the Tinker surface is bound onto it.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from dataclasses import dataclass
from typing import Any

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
    lora_rank: int = 0
    # DeepSpeed needs a batch size at provisioning time; Tinker has no verb that
    # declares one. These only have to satisfy DeepSpeed's own invariant, since
    # Cortex chunks each forward-backward to fit whatever actually arrives.
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dtype: str = "bfloat16"
    seed: int = 7
    # The Cortex image ships FA3 only; FA2 dies at model load.
    attn_implementation: str = "flash_attention_3"
    max_tokens_per_mb: int = 8192
    gpu_memory_utilization: float = 0.8
    zero_stage: int = 2
    job_id: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def max_seq_len(self) -> int:
        return self.max_prompt_length + self.max_response_length


def _client_config(cfg: TinkerServeConfig) -> Any:
    from arctic_platform.client import CortexConfig
    from recipes.recipe_utils import client_config
    from recipes.recipe_utils import load_backend

    return client_config(
        backend=load_backend(cfg.config) if cfg.config else CortexConfig(),
        model_name=cfg.model,
        max_seq_len=cfg.max_seq_len,
        seed=cfg.seed,
        dtype=cfg.dtype,
        training_gpus=cfg.training_gpus,
        sampling_gpus=cfg.sampling_gpus,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        lora_rank=cfg.lora_rank,
        # `offload_optimizer` is omitted rather than set to `{"device": "none"}`:
        # that is still enough for DeepSpeed to instantiate CPUAdam, which then
        # asserts the params are on cuda.
        ds_config={
            "train_batch_size": (
                cfg.micro_batch_size * cfg.training_gpus * cfg.gradient_accumulation_steps
            ),
            "train_micro_batch_size_per_gpu": cfg.micro_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "bf16": {"enabled": cfg.dtype == "bfloat16"},
            "zero_optimization": {"stage": cfg.zero_stage},
            "optimizer": {"type": "AdamW", "params": {"lr": cfg.learning_rate}},
        },
        ds_worker_config={
            "attn_implementation": cfg.attn_implementation,
            "model_provider": "huggingface",
            "mb_spec": {"max_tokens_per_mb": cfg.max_tokens_per_mb},
        },
        job_id=cfg.job_id,
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
        from transformers import AutoTokenizer

        client_cfg = _client_config(cfg)
        attached = client_cfg.training_job_id is not None
        client = AsyncArcticRLClient(client_cfg)
        logger.info("training job %s is running", client.jobs.training)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        init_tinker_state(
            app,
            base_model=cfg.model,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            **build_handlers(client),
        )
        app.state.arctic_client = client
        try:
            yield
        finally:
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
