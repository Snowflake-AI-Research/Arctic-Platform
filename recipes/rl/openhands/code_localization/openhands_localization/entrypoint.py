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
"""Localization training entrypoint.

Select it with ``trainer.override_entrypoint=openhands_localization.entrypoint``.
The directory that contains this package has to be on ``PYTHONPATH``. Job setup
is :func:`arctic_platform.integrations.skyrl.entrypoint.launch_driver`.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from openhands_localization.config import validate_openhands_cfg

from arctic_platform.integrations._cortex_dispatch import create_cortex_client
from arctic_platform.integrations.harness.generator import HarnessGenerator
from arctic_platform.integrations.harness.proxy import ChatCompletionsProxy
from arctic_platform.integrations.skyrl.entrypoint import launch_driver
from arctic_platform.integrations.skyrl.entrypoint import patch_upstream

_ERROR_DETAIL = dict(file_reward=0.0, module_reward=0.0, entity_reward=0.0, reward=0.0)


def _max_model_len(cfg: Any) -> int:
    arctic = cfg.trainer.arctic_rl
    if arctic.vllm_max_model_len is not None:
        return int(arctic.vllm_max_model_len)
    return int(cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length)


def _experiment_cls():
    from integrations.arctic_rl.entrypoint import ArcticRLExp
    from openhands_localization.rollout import OpenHandsAdapter
    from openhands_localization.rollout import grade_localization
    from openhands_localization.trainer import openhands_trainer_cls

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
            model_name = cfg.trainer.policy.model.path
            max_turns = int(cfg.generator.max_turns)
            workspace = os.environ.get("OPENHANDS_WORKSPACE", "/tmp/testbed")
            adapter = OpenHandsAdapter("openai/" + model_name, max_turns, workspace)

            def grader(episodes, artifacts, request):
                return grade_localization(episodes, artifacts, request.env_extra, tokenizer, max_turns)

            return HarnessGenerator(
                tokenizer=tokenizer,
                proxy=self.sampling_proxy,
                traj_dir=str(Path(cfg.trainer.ckpt_path) / "trajectories"),
                adapter=adapter,
                grader=grader,
                error_detail=_ERROR_DETAIL,
                step_wise=bool(cfg.generator.step_wise_trajectories),
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
    """Ray worker. The experiment class is built here so it is not pickled.

    SkyRL calls ``create_arctic_rl_client(config, server_state)``. The patched
    factory has to keep that second argument.
    """
    patch_upstream()
    experiment = _experiment_cls()(remote_cfg, reconnect_config=reconnect_config)
    experiment.run()


def _require_source_checkout() -> None:
    """Fail when a cached wheel is the module Python imported."""
    root = os.environ.get("AP_ROOT")
    if not root:
        return
    loaded = Path(launch_driver.__code__.co_filename).resolve()
    expected = Path(root).resolve()
    if not loaded.is_relative_to(expected):
        raise SystemExit(
            f"ERROR: launch_driver loaded from {loaded}, not from {expected}. A cached wheel is hiding the checkout."
        )


def main() -> None:
    from integrations.arctic_rl.config import ArcticRLTrainerConfig
    from integrations.arctic_rl.config import ArcticSkyRLConfig
    from skyrl.train.utils import validate_cfg

    _require_source_checkout()

    argv = [arg for arg in sys.argv[1:] if not arg.startswith("trainer.override_entrypoint=")]
    cfg = ArcticSkyRLConfig.from_cli_overrides(argv)
    if cfg.trainer.arctic_rl is None:
        cfg.trainer.arctic_rl = ArcticRLTrainerConfig()
    validate_openhands_cfg(cfg)
    validate_cfg(cfg)
    launch_driver(cfg, _run_remote, num_cpus=0, env_prefixes=("ARCTIC_", "WANDB_", "GIT_", "OPENHANDS_"))


if __name__ == "__main__":
    main()
