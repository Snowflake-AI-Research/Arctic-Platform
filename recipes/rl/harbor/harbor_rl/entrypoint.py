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
"""Harbor RL training entrypoint on Cortex Training.

Select with ``trainer.override_entrypoint=harbor_rl.entrypoint``.
Uses :class:`arctic_platform.integrations.harbor.adapter.HarborTrialAdapter`
to run every rollout through Harbor's :class:`harbor.trial.trial.Trial`
against :class:`arctic_platform.integrations.harness.proxy.ChatCompletionsProxy`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from openhands_localization import reward as localization_reward_module
from openhands_localization.config import validate_openhands_cfg
from openhands_localization.events import last_step_is_one_tool_call
from openhands_localization.trainer import openhands_trainer_cls

from arctic_platform.integrations.harbor.adapter import DEFAULT_HARBOR_ENV
from arctic_platform.integrations.harbor.adapter import HarborTrialAdapter
from arctic_platform.integrations.harbor.adapter import grade_harbor
from arctic_platform.integrations.harbor.adapter import resolve_env
from arctic_platform.integrations.harbor.env import contained_path
from arctic_platform.integrations.harness.generator import Grade
from arctic_platform.integrations.harness.generator import HarnessGenerator
from arctic_platform.integrations.harness.proxy import ChatCompletionsProxy

DEFAULT_HARBOR_AGENT = "harbor_rl.agent:OpenHandsHarborAgent"
_ERROR_DETAIL = dict(file_reward=0.0, module_reward=0.0, entity_reward=0.0, reward=0.0)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    return str(value)


_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _require(pattern: re.Pattern[str], value: str, field: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"Dataset row has an invalid {field}: {value!r}")
    return value


_LOCALIZATION_VERIFIER_SH = """#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
python3 - <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, "/tests")
from localization_reward import localization_reward

sub = pathlib.Path("/logs/agent/localization_submission.json")
truth = pathlib.Path("/tests/ground_truth.json")
locations = json.loads(sub.read_text()) if sub.is_file() else None
instance = json.loads(truth.read_text()) if truth.is_file() else {}
total, detail = localization_reward(locations, instance)
pathlib.Path("/logs/verifier/reward.txt").write_text(f"{total:.6f}\\n")
pathlib.Path("/logs/verifier/reward.json").write_text(json.dumps(detail) + "\\n")
PY
"""


def materialize_harbor_task_from_row(instance: dict[str, Any], tasks_root: Path) -> Path:
    """Materialize a Harbor task directory for a SWE-smith code localization row on demand.

    ``instance_id``, ``repo``, and ``base_commit`` come from the dataset, so they
    are validated before they name a directory or reach a Dockerfile line.
    """
    instance_id = _require(_INSTANCE_ID, str(instance["instance_id"]), "instance_id")
    repo = _require(_REPO, str(instance["repo"]), "repo")
    base_commit = instance.get("base_commit")
    if base_commit is not None and str(base_commit) in ("None", "nan", ""):
        base_commit = None
    if base_commit is not None:
        base_commit = _require(_COMMIT, str(base_commit), "base_commit")

    tasks_root.mkdir(parents=True, exist_ok=True)
    task_dir = contained_path(tasks_root, tasks_root / instance_id, "Task directory")
    test_sh = task_dir / "tests" / "test.sh"
    if all(p.is_file() for p in (test_sh, task_dir / "instruction.md", task_dir / "tests" / "localization_reward.py")):
        return task_dir

    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    use_patch = bool(instance.get("use_patch", True))
    patch_text = str(instance.get("patch") or "") if use_patch else ""
    problem = str(instance.get("problem_statement") or "")
    file_changes = _to_jsonable(
        instance.get("file_changes") if instance.get("file_changes") is not None else instance.get("target") or []
    )

    (task_dir / "task.toml").write_text(
        f'name = "{instance_id}"\nversion = "1.0"\n',
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text(problem, encoding="utf-8")

    repo_dir = f"{repo.replace('/', '_')}_{instance_id}"
    df_lines = [
        "FROM python:3.11-slim",
        "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*",
        "WORKDIR /app",
        f"RUN git clone {shlex.quote(f'https://github.com/{repo}.git')} {shlex.quote(f'/app/{repo_dir}')}",
        f"WORKDIR /app/{repo_dir}",
    ]
    if base_commit:
        df_lines.append(f"RUN git checkout {shlex.quote(base_commit)}")
    if patch_text:
        (env_dir / "bug.patch").write_text(patch_text, encoding="utf-8")
        df_lines.append("COPY bug.patch /app/bug.patch")
        df_lines.append("RUN git apply /app/bug.patch && rm -f /app/bug.patch")
    (env_dir / "Dockerfile").write_text("\n".join(df_lines) + "\n", encoding="utf-8")

    (tests_dir / "ground_truth.json").write_text(
        json.dumps(dict(instance_id=instance_id, file_changes=file_changes)),
        encoding="utf-8",
    )
    shutil.copyfile(localization_reward_module.__file__, tests_dir / "localization_reward.py")
    test_sh.write_text(_LOCALIZATION_VERIFIER_SH, encoding="utf-8")
    test_sh.chmod(0o755)
    return task_dir


def harbor_env_kwargs() -> dict[str, Any]:
    """``HARBOR_ENV_KWARGS`` (a JSON object), passed to the Harbor environment like ``harbor run --ek``."""
    raw = os.environ.get("HARBOR_ENV_KWARGS", "").strip()
    if not raw:
        return {}
    kwargs = json.loads(raw)
    if not isinstance(kwargs, dict):
        raise ValueError(f"HARBOR_ENV_KWARGS must be a JSON object, got {raw!r}")
    return kwargs


def _max_model_len(cfg: Any) -> int:
    arctic = cfg.trainer.arctic_rl
    if arctic.vllm_max_model_len is not None:
        return int(arctic.vllm_max_model_len)
    return int(cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length)


def _experiment_cls():
    from integrations.arctic_rl.entrypoint import ArcticRLExp

    from arctic_platform.integrations._cortex_dispatch import create_cortex_client

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
            harbor_agent = os.environ.get("HARBOR_AGENT", DEFAULT_HARBOR_AGENT)
            harbor_env = resolve_env(os.environ.get("HARBOR_ENV") or DEFAULT_HARBOR_ENV)
            skill_env = os.environ.get("HARBOR_SKILL", "")
            skills = [s.strip() for s in skill_env.split(":") if s.strip()] if skill_env else []

            adapter = HarborTrialAdapter(
                "openai/" + model_name,
                agent=harbor_agent,
                env=harbor_env,
                trials_root=workspace,
                max_turns=max_turns,
                skills=skills,
                env_kwargs=harbor_env_kwargs(),
                prepare_task_fn=materialize_harbor_task_from_row,
            )

            def grader(episodes, artifacts, request):
                grade = grade_harbor(episodes, artifacts, request, error_detail=_ERROR_DETAIL)
                is_loc = "file_reward" in grade.reward_detail
                if is_loc and grade.reward > 0 and len(episodes) > 0:
                    last_ids = list(episodes[-1].response_token_ids)
                    last_text = tokenizer.decode(last_ids, skip_special_tokens=False)
                    if not last_step_is_one_tool_call(last_text):
                        return Grade(reward=0.0, reward_detail=dict(_ERROR_DETAIL), exhausted=False)
                if is_loc and grade.reward == 0.0 and len(episodes) >= max_turns and artifacts.get("exhausted"):
                    return Grade(reward=0.0, reward_detail=dict(_ERROR_DETAIL), exhausted=True)
                return grade

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
    from arctic_platform.integrations.skyrl.entrypoint import patch_upstream

    patch_upstream()
    experiment = _experiment_cls()(remote_cfg, reconnect_config=reconnect_config)
    experiment.run()


def _require_source_checkout() -> None:
    from arctic_platform.integrations.skyrl.entrypoint import launch_driver

    root = os.environ.get("AP_ROOT")
    if not root:
        return
    loaded = Path(launch_driver.__code__.co_filename).resolve()
    expected = Path(root).resolve()
    if not loaded.is_relative_to(expected):
        raise SystemExit(
            f"ERROR: launch_driver loaded from {loaded}, not from {expected}. A cached wheel is hiding the checkout."
        )


def check_harbor_env() -> None:
    """Fail on the driver, before any GPUs are provisioned, if ``HARBOR_ENV`` can't run tasks."""
    from arctic_platform.integrations.harbor.adapter import HOST_HARBOR_ENV
    from arctic_platform.integrations.harbor.adapter import SNOWFLAKE_HARBOR_ENV
    from arctic_platform.integrations.harbor.env import ALLOW_HOST_EXEC_VAR
    from arctic_platform.integrations.harbor.env import host_exec_allowed

    env = resolve_env(os.environ.get("HARBOR_ENV") or DEFAULT_HARBOR_ENV)
    kwargs = harbor_env_kwargs()
    if env == SNOWFLAKE_HARBOR_ENV and not (kwargs.get("image_repo") or kwargs.get("prebuilt_image")):
        raise SystemExit(
            "ERROR: HARBOR_ENV=snowflake needs somewhere to mirror task images. Set, for example,\n        "
            ' HARBOR_ENV_KWARGS=\'{"image_repo": "DB.SCHEMA.REPO", "connection": "MY_CONN", "role": "MY_ROLE"}\''
        )
    if env == HOST_HARBOR_ENV and not host_exec_allowed(kwargs.get("allow_host_exec")):
        raise SystemExit(
            "ERROR: HARBOR_ENV=host runs task and agent commands on this machine without isolation.\n"
            f"       Use a sandbox, or set {ALLOW_HOST_EXEC_VAR}=1 if every task is trusted."
        )


def main() -> None:
    from integrations.arctic_rl.config import ArcticRLTrainerConfig
    from integrations.arctic_rl.config import ArcticSkyRLConfig
    from skyrl.train.utils import validate_cfg

    from arctic_platform.integrations.skyrl.entrypoint import launch_driver

    _require_source_checkout()
    check_harbor_env()

    argv = [arg for arg in sys.argv[1:] if not arg.startswith("trainer.override_entrypoint=")]
    cfg = ArcticSkyRLConfig.from_cli_overrides(argv)
    if cfg.trainer.arctic_rl is None:
        cfg.trainer.arctic_rl = ArcticRLTrainerConfig()
    validate_openhands_cfg(cfg)
    validate_cfg(cfg)
    launch_driver(
        cfg,
        _run_remote,
        num_cpus=0,
        env_prefixes=("ARCTIC_", "WANDB_", "GIT_", "OPENHANDS_", "HARBOR_"),
    )


if __name__ == "__main__":
    main()
