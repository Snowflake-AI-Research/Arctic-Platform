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
"""SkyRL entrypoint that trains an OpenHands localization agent on Cortex.

Select it with::

    trainer.override_entrypoint=arctic_platform.integrations.openhands.entrypoint

SkyRL parses the config and owns the GRPO loop. This module swaps in the
Cortex client, the OpenHands generator, and the dispatch that keeps the
rollout's loss mask. The driver is CPU-only.

Adapted from CodeScout's ``src/train.py`` at
https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2
and from ``arctic_platform.integrations.skyrl.entrypoint``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from arctic_platform.integrations._cortex_dispatch import create_cortex_client
from arctic_platform.integrations.openhands.config import validate_openhands_cfg
from arctic_platform.integrations.openhands.proxy import ChatCompletionsProxy


def _client_config(cfg: Any):
    """SkyRL's client config, with the placement timeout taken from ``startup_timeout``."""
    from integrations.arctic_rl.config import build_rl_config

    rl_config = build_rl_config(cfg)
    return rl_config.model_copy(update=dict(job_ready_timeout=float(cfg.trainer.arctic_rl.startup_timeout)))


def _forwarded_env(cfg: Any) -> dict[str, str]:
    from skyrl.train.utils.utils import prepare_runtime_environment

    env_vars = prepare_runtime_environment(cfg)
    for key, value in os.environ.items():
        if key.startswith(("ARCTIC_", "WANDB_", "GIT_")):
            env_vars[key] = value
    for key in ("PATH", "HF_HOME", "HF_HUB_CACHE"):
        value = os.environ.get(key)
        if value is not None and value != "":
            env_vars[key] = value
    return env_vars


def _max_model_len(cfg: Any) -> int:
    arctic = cfg.trainer.arctic_rl
    if arctic.vllm_max_model_len is not None:
        return int(arctic.vllm_max_model_len)
    return int(cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length)


def _experiment_cls():
    from integrations.arctic_rl.entrypoint import ArcticRLExp

    from arctic_platform.integrations.openhands.rollout import OpenHandsGenerator
    from arctic_platform.integrations.openhands.trainer import openhands_trainer_cls

    trainer_cls = openhands_trainer_cls()

    class _Experiment(ArcticRLExp):
        def make_generate_fn(self):
            reconnect = self.arctic_client.reconnect_config()

            async def generate(prompt_ids, sampling_params, routing_key):
                client = getattr(generate, "_client", None)
                if client is None:
                    client = await asyncio.to_thread(create_cortex_client, reconnect)
                    generate._client = client
                results = await client.generate(
                    prompts=[list(prompt_ids)],
                    sampling_params=sampling_params,
                    routing_key=routing_key,
                )
                return results[0]

            return generate

        def get_generator(self, cfg, tokenizer, inference_engine_client):
            max_model_len = _max_model_len(cfg)
            self.sampling_proxy = ChatCompletionsProxy(
                generate_fn=self.make_generate_fn(),
                tokenizer=tokenizer,
                model_name=cfg.trainer.policy.model.path,
                max_model_len=max_model_len,
                default_max_tokens=cfg.generator.sampling_params.max_generate_length,
            )
            self.sampling_proxy.start()
            workspace = os.environ.get("OPENHANDS_WORKSPACE", "/tmp/testbed")
            return OpenHandsGenerator(
                generator_cfg=cfg.generator,
                tokenizer=tokenizer,
                model_name=cfg.trainer.policy.model.path,
                proxy=self.sampling_proxy,
                workspace_root=workspace,
                traj_dir=str(Path(cfg.trainer.ckpt_path) / "trajectories"),
            )

        def get_trainer(
            self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
        ):
            return trainer_cls(
                cfg=cfg,
                tracker=tracker,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client,
                generator=generator,
                colocate_pg=colocate_pg,
                arctic_client=self.arctic_client,
            )

    return _Experiment


def _run_remote(remote_cfg: Any, reconnect_config: Any) -> None:
    """Ray worker. Rebuilds the experiment here so the class is not pickled."""
    from arctic_platform.integrations.skyrl.entrypoint import _patch_upstream

    # _patch_upstream installs the two-argument factory SkyRL calls
    # (config, server_state). Cortex has no in-process server; the extra
    # argument is ignored. Replacing that factory with the one-argument
    # client constructor breaks experiment startup.
    _patch_upstream()
    experiment = _experiment_cls()(remote_cfg, reconnect_config=reconnect_config)
    experiment.run()


def main() -> None:
    """Parse, check the Cortex constraints, provision, then run SkyRL's loop."""
    import ray
    from integrations.arctic_rl.config import ArcticRLTrainerConfig
    from integrations.arctic_rl.config import ArcticSkyRLConfig
    from loguru import logger
    from skyrl.train.utils import validate_cfg

    from arctic_platform.integrations.skyrl.entrypoint import _patch_upstream

    argv = [arg for arg in sys.argv[1:] if not arg.startswith("trainer.override_entrypoint=")]
    cfg = ArcticSkyRLConfig.from_cli_overrides(argv)
    if cfg.trainer.arctic_rl is None:
        cfg.trainer.arctic_rl = ArcticRLTrainerConfig()
    validate_openhands_cfg(cfg)
    validate_cfg(cfg)

    upstream = _patch_upstream()
    rl_config = _client_config(cfg)
    logger.info("Provisioning Cortex training and sampling jobs")
    pre_client = create_cortex_client(rl_config)
    reconnect = pre_client.reconnect_config()
    logger.info(f"Cortex jobs ready: training={pre_client.training_job_id} sampling={pre_client.sampling_job_id}")

    env_vars = _forwarded_env(cfg)
    skyrl_root = Path(upstream.__file__).resolve().parents[2]
    python_path = [str(skyrl_root), os.environ.get("PYTHONPATH", "")]
    env_vars["PYTHONPATH"] = os.pathsep.join(part for part in python_path if part != "")
    runtime_env = dict(env_vars=env_vars)
    # num_cpus=0: this task is the coordinator. It ray.get()s the rollouts,
    # and a reserved CPU here can deadlock those children on a small machine.
    remote = ray.remote(num_cpus=0)(_run_remote)
    ray.init(num_gpus=0, runtime_env=runtime_env, ignore_reinit_error=True)
    ray.get(remote.options(runtime_env=runtime_env).remote(cfg, reconnect))


if __name__ == "__main__":
    main()
