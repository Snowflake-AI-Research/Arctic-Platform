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
"""Harbor ``HarnessAdapter`` and ``Grader`` for SkyRL on Cortex Training.

Drives each rollout through Harbor's own :class:`harbor.trial.trial.Trial`
orchestrator:
1. ``submit(request)`` builds a :class:`harbor.models.trial.config.TrialConfig`
   pointed at ``request.base_url`` and ``request.api_key`` (plus any Harbor
   agent, environment, and OpenHands skills) and starts ``Trial.run()`` in a
   background thread.
2. ``get_status(rollout_id)`` returns ``"running"``, ``"completed"``, or
   ``"errored"``.
3. ``collect_output(rollout_id)`` returns the Harbor ``TrialResult`` rewards
   produced by Harbor's ``Verifier`` (``tests/test.sh`` -> ``reward.txt`` /
   ``reward.json``), and ``grade_harbor`` converts them into a
   :class:`arctic_platform.integrations.harness.generator.Grade`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import traceback
from pathlib import Path
from typing import Any

from arctic_platform.integrations.harness.generator import Grade
from arctic_platform.integrations.harness.generator import RolloutRequest

DEFAULT_HARBOR_ENV = "arctic_platform.integrations.harbor.env:HostEnvironment"


def _build_trial_config(
    task_dir: Path,
    trials_dir: Path,
    trial_name: str,
    *,
    agent: str,
    env: str,
    model_name: str,
    base_url: str,
    api_key: str,
    skills: list[str] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    from harbor.agents.factory import AgentFactory
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import AgentConfig
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.models.trial.config import TaskConfig
    from harbor.models.trial.config import TrialConfig

    agent_name: str | None = None
    agent_import_path: str | None = None
    if ":" in agent:
        agent_import_path = agent
    else:
        agent_name = agent

    env_type: EnvironmentType | None = None
    env_import_path: str | None = None
    if ":" in env:
        env_import_path = env
    else:
        env_type = EnvironmentType(env)

    merged_agent_env = {
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": api_key,
        "LLM_BASE_URL": base_url,
        "LLM_API_KEY": api_key,
        "LLM_MODEL": model_name,
    }
    kw = dict(agent_kwargs or {})
    kw.setdefault("api_key", api_key)
    if agent_import_path is not None:
        kw.setdefault("base_url", base_url)

    agent_cfg_kwargs: dict[str, Any] = dict(
        name=agent_name,
        import_path=agent_import_path,
        model_name=model_name,
        skills=[Path(s) for s in (skills or [])],
        kwargs=kw,
        env=merged_agent_env,
    )
    if "base_url" in AgentConfig.model_fields and hasattr(AgentFactory, "get_agent_class_from_config"):
        probe_cfg = AgentConfig(name=agent_name, import_path=agent_import_path)
        agent_cls = AgentFactory.get_agent_class_from_config(probe_cfg)
        if getattr(agent_cls, "MODEL_CONNECTION", None) is not None:
            agent_cfg_kwargs["base_url"] = base_url

    return TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name=trial_name,
        trials_dir=trials_dir,
        agent=AgentConfig(**agent_cfg_kwargs),
        environment=EnvironmentConfig(
            type=env_type,
            import_path=env_import_path,
            delete=True,
            kwargs=dict(env_kwargs or {}),
        ),
    )


async def run_harbor_trial(
    task_dir: Path,
    trials_dir: Path,
    trial_name: str,
    *,
    agent: str,
    env: str,
    model_name: str,
    base_url: str,
    api_key: str,
    skills: list[str] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one Harbor ``Trial`` and return its verifier rewards and metadata."""
    from harbor.trial.trial import Trial

    trial_cfg = _build_trial_config(
        task_dir=task_dir,
        trials_dir=trials_dir,
        trial_name=trial_name,
        agent=agent,
        env=env,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        skills=skills,
        agent_kwargs=agent_kwargs,
        env_kwargs=env_kwargs,
    )
    trial_dir = trials_dir / trial_name
    try:
        trial = await Trial.create(trial_cfg)
        result = await trial.run()
        rewards = dict((result.verifier_result.rewards if result.verifier_result else None) or {})
        reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0))) if rewards else 0.0

        messages: list[Any] = []
        exhausted = False
        agent_meta_path = trial_dir / "agent" / "rollout_meta.json"
        if agent_meta_path.is_file():
            try:
                meta = json.loads(agent_meta_path.read_text(encoding="utf-8"))
                messages = meta.get("messages") or []
                exhausted = bool(meta.get("exhausted", False))
            except Exception:
                pass

        return dict(
            reward=reward,
            reward_detail=rewards,
            exhausted=exhausted,
            messages=messages,
            exception=str(result.exception_info.exception_message) if result.exception_info else None,
        )
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


class HarborTrialAdapter:
    """``HarnessAdapter`` that runs each rollout through Harbor's ``Trial``."""

    def __init__(
        self,
        model_name: str,
        *,
        agent: str,
        env: str = DEFAULT_HARBOR_ENV,
        trials_root: str = "/tmp/harbor-trials",
        max_turns: int = 10,
        skills: list[str] | None = None,
        agent_kwargs: dict[str, Any] | None = None,
        env_kwargs: dict[str, Any] | None = None,
        prepare_task_fn: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._agent = agent
        self._env = env
        self._trials_root = Path(trials_root)
        self._max_turns = max_turns
        self._skills = list(skills or [])
        self._agent_kwargs = dict(agent_kwargs or {})
        self._env_kwargs = dict(env_kwargs or {})
        self._prepare_task_fn = prepare_task_fn
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}

    def submit(self, request: RolloutRequest) -> str:
        rollout_id = request.routing_key
        with self._lock:
            self._runs[rollout_id] = dict(status="running", artifacts=None)

        def work() -> None:
            try:
                instance = request.env_extra
                if "harbor_task_dir" in instance and instance["harbor_task_dir"]:
                    task_dir = Path(str(instance["harbor_task_dir"]))
                elif self._prepare_task_fn is not None:
                    task_dir = Path(self._prepare_task_fn(instance, self._trials_root / "tasks"))
                else:
                    raise ValueError("Dataset row is missing 'harbor_task_dir'")

                kw = dict(self._agent_kwargs)
                kw.setdefault("max_turns", self._max_turns)
                kw.setdefault("training_phase", request.phase)
                kw["instance"] = instance

                trial_name = f"trial-{rollout_id}-{os.urandom(3).hex()}"
                artifacts = asyncio.run(
                    run_harbor_trial(
                        task_dir=task_dir,
                        trials_dir=self._trials_root,
                        trial_name=trial_name,
                        agent=self._agent,
                        env=self._env,
                        model_name=self._model_name,
                        base_url=request.base_url,
                        api_key=request.api_key,
                        skills=self._skills,
                        agent_kwargs=kw,
                        env_kwargs=self._env_kwargs,
                    )
                )
            except Exception:
                artifacts = dict(
                    error=traceback.format_exc(),
                    reward=0.0,
                    reward_detail=dict(reward=0.0),
                    exhausted=False,
                    messages=[],
                )
                self._store(rollout_id, "errored", artifacts)
                return
            self._store(rollout_id, "completed", artifacts)

        threading.Thread(target=work, name=f"harbor-{rollout_id}", daemon=True).start()
        return rollout_id

    def _store(self, rollout_id: str, status: str, artifacts: dict[str, Any]) -> None:
        with self._lock:
            self._runs[rollout_id] = dict(status=status, artifacts=artifacts)

    def get_status(self, rollout_id: str) -> str:
        with self._lock:
            return self._runs[rollout_id]["status"]

    def collect_output(self, rollout_id: str) -> dict[str, Any]:
        with self._lock:
            artifacts = self._runs[rollout_id]["artifacts"]
        return dict(artifacts or {})


def grade_harbor(
    episodes: list[Any],
    artifacts: dict[str, Any],
    request: RolloutRequest | None = None,
    *,
    error_detail: dict[str, float] | None = None,
) -> Grade:
    """Convert Harbor's verifier output in ``artifacts`` into a :class:`Grade`."""
    del episodes, request
    default_detail = dict(error_detail or dict(reward=0.0))
    if artifacts.get("error"):
        return Grade(reward=0.0, reward_detail=default_detail, exhausted=False)
    reward = float(artifacts.get("reward") or 0.0)
    raw_detail = dict(artifacts.get("reward_detail") or {})
    if not raw_detail:
        raw_detail = dict(default_detail, reward=reward)
    else:
        raw_detail.setdefault("reward", reward)
    exhausted = bool(artifacts.get("exhausted", False))
    return Grade(reward=reward, reward_detail=raw_detail, exhausted=exhausted)
