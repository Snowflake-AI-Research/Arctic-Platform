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
   pointed at ``request.base_url`` and ``request.api_key`` (plus any driver-side
   Harbor agent, environment, and OpenHands skills) and starts ``Trial.run()`` in a
   background thread.
2. ``get_status(rollout_id)`` returns ``"running"``, ``"completed"``, or
   ``"errored"``. A trial that fails before its agent starts (the sandbox never
   came up, agent setup failed) is ``"errored"``, not a reward of 0.
3. ``collect_output(rollout_id)`` returns the Harbor ``TrialResult`` rewards
   produced by Harbor's ``Verifier`` (``tests/test.sh`` -> ``reward.txt`` /
   ``reward.json``), and ``grade_harbor`` converts them into a
   :class:`arctic_platform.integrations.harness.generator.Grade`.

Every rollout of an adapter runs on one persistent event loop, so asyncio
locks and semaphores that environments share across trials (Snowflake
sessions, image registries, live-sandbox caps) are only ever used from the
loop they were created on.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from arctic_platform.integrations.harness.generator import Grade
from arctic_platform.integrations.harness.generator import RolloutRequest

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_S = 120

HOST_HARBOR_ENV = "arctic_platform.integrations.harbor.env:HostEnvironment"
SNOWFLAKE_HARBOR_ENV = "arctic_platform.integrations.harbor.snowflake_sandbox:SnowflakeSandboxEnvironment"
DEFAULT_HARBOR_ENV = SNOWFLAKE_HARBOR_ENV
ENV_ALIASES = {"host": HOST_HARBOR_ENV, "snowflake": SNOWFLAKE_HARBOR_ENV}


def resolve_env(env: str) -> str:
    """Map ``host`` and ``snowflake`` to their import paths; pass anything else through."""
    return ENV_ALIASES.get(env.strip().lower(), env.strip())


def require_driver_resident_agent(agent: str) -> Any:
    """The Harbor agent class for ``agent``, if it runs on the driver rather than inside the environment.

    Harbor exports an installed agent's credentials into every command it runs
    in the environment, so a task or prompt-injected command could read the
    rollout's proxy key. Driver-resident agents receive the key as a
    constructor argument that their commands never see.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.models.trial.config import AgentConfig

    if ":" in agent:
        agent_cls = AgentFactory.get_agent_class_from_config(AgentConfig(import_path=agent))
    else:
        agent_cls = AgentFactory.get_agent_class_from_config(AgentConfig(name=agent))
    if issubclass(agent_cls, BaseInstalledAgent):
        raise ValueError(
            f"HARBOR_AGENT={agent!r} is installed and run inside the environment, where its commands could read "
            "the rollout's API key. Use an agent that runs on the driver, such as the default "
            "harbor_rl.agent:OpenHandsHarborAgent or your own harbor BaseAgent subclass (module:Class)."
        )
    return agent_cls


def require_task_dir(path: Path) -> Path:
    """``path`` resolved, if it is a Harbor task directory (``instruction.md`` and ``tests/test.sh``)."""
    task_dir = path.expanduser().resolve()
    if not (task_dir / "instruction.md").is_file() or not (task_dir / "tests" / "test.sh").is_file():
        raise ValueError(f"{path} is not a Harbor task directory (needs instruction.md and tests/test.sh)")
    return task_dir


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

    env = resolve_env(env)
    env_type: EnvironmentType | None = None
    env_import_path: str | None = None
    if ":" in env:
        env_import_path = env
    else:
        env_type = EnvironmentType(env)

    agent_cls = require_driver_resident_agent(agent)
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
    )
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
        if result.exception_info is not None and result.agent_execution is None:
            info = result.exception_info
            raise HarborTrialError(
                f"Harbor trial failed before the agent started: {info.exception_type}: {info.exception_message}\n"
                f"{info.exception_traceback}"
            )
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


class HarborTrialError(RuntimeError):
    """A Harbor trial failed before its agent started, so it measures nothing about the policy."""


class _LoopThread:
    """An event loop running on a daemon thread until :meth:`close`."""

    def __init__(self, max_workers: int) -> None:
        self.loop = asyncio.new_event_loop()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="harbor-io")
        self.loop.set_default_executor(self._executor)
        self._thread = threading.Thread(target=self.loop.run_forever, name="harbor-loop", daemon=True)
        self._thread.start()

    def submit(self, coro: Any) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> bool:
        """Cancel every pending rollout, close the loop, and wait for its worker threads.

        Cancelling a task doesn't stop a thread it is waiting on, so each agent
        has to end its own blocking work when cancelled (``OpenHandsHarborAgent``
        pauses its conversation). Closing the loop first makes any further
        ``environment.exec`` from such a thread fail at once instead of running.
        Returns whether every worker thread finished within ``timeout_s``.
        """
        deadline = time.monotonic() + timeout_s

        async def cancel_pending() -> None:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(cancel_pending(), self.loop).result(timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning("Harbor rollouts did not finish cleaning up within %ss", timeout_s)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(max(0.0, deadline - time.monotonic()))
            if not self._thread.is_alive():
                self.loop.close()
        waiter = threading.Thread(target=self._executor.shutdown, kwargs=dict(cancel_futures=True), daemon=True)
        waiter.start()
        waiter.join(max(0.0, deadline - time.monotonic()))
        if waiter.is_alive():
            logger.warning("Harbor worker threads were still running %ss after close; leaving them", timeout_s)
            return False
        return True


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
        max_concurrency: int = 256,
    ) -> None:
        self._model_name = model_name
        self._agent = agent
        self._env = resolve_env(env)
        self._trials_root = Path(trials_root)
        self._max_turns = max_turns
        self._skills = list(skills or [])
        self._agent_kwargs = dict(agent_kwargs or {})
        self._env_kwargs = dict(env_kwargs or {})
        self._prepare_task_fn = prepare_task_fn
        self._max_concurrency = int(max_concurrency)
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._loop_thread: _LoopThread | None = None

    def _loop(self) -> _LoopThread:
        with self._lock:
            if self._loop_thread is None:
                # Each rollout holds one worker thread for its whole agent conversation.
                self._loop_thread = _LoopThread(self._max_concurrency)
            return self._loop_thread

    def submit(self, request: RolloutRequest) -> str:
        rollout_id = request.routing_key
        with self._lock:
            self._runs[rollout_id] = dict(status="running", artifacts=None)
        future = self._loop().submit(self._run_rollout(rollout_id, request))
        future.add_done_callback(lambda f: self._finish(rollout_id, f))
        return rollout_id

    def _task_dir(self, instance: dict[str, Any]) -> Path:
        if instance.get("harbor_task_dir"):
            return require_task_dir(Path(str(instance["harbor_task_dir"])))
        if self._prepare_task_fn is not None:
            return require_task_dir(Path(self._prepare_task_fn(instance, self._trials_root / "tasks")))
        raise ValueError("Dataset row is missing 'harbor_task_dir'")

    async def _run_rollout(self, rollout_id: str, request: RolloutRequest) -> dict[str, Any]:
        instance = request.env_extra
        task_dir = await asyncio.to_thread(self._task_dir, instance)
        kw = dict(self._agent_kwargs)
        kw.setdefault("max_turns", self._max_turns)
        kw.setdefault("training_phase", request.phase)
        kw["instance"] = instance
        return await run_harbor_trial(
            task_dir=task_dir,
            trials_dir=self._trials_root,
            trial_name=f"trial-{rollout_id}-{os.urandom(3).hex()}",
            agent=self._agent,
            env=self._env,
            model_name=self._model_name,
            base_url=request.base_url,
            api_key=request.api_key,
            skills=self._skills,
            agent_kwargs=kw,
            env_kwargs=self._env_kwargs,
        )

    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> None:
        """Cancel in-flight rollouts, stop the rollout loop, and release what the environment shares across trials.

        Environments loaded by import path may define a ``close_shared_resources()``
        classmethod for state that outlives a single trial (sessions, credentials).
        It runs after the worker threads finish, or after ``timeout_s`` if one is stuck.
        """
        with self._lock:
            loop_thread, self._loop_thread = self._loop_thread, None
        try:
            if loop_thread is not None:
                loop_thread.close(timeout_s)
        finally:
            module_name, _, class_name = self._env.partition(":")
            env_cls = getattr(sys.modules.get(module_name), class_name, None) if class_name else None
            close_shared = getattr(env_cls, "close_shared_resources", None)
            if close_shared is not None:
                close_shared()

    def _finish(self, rollout_id: str, future: concurrent.futures.Future) -> None:
        exc = asyncio.CancelledError("Harbor adapter closed") if future.cancelled() else future.exception()
        if exc is None:
            self._store(rollout_id, "completed", future.result())
            return
        self._store(
            rollout_id,
            "errored",
            dict(
                error="".join(traceback.format_exception(exc)),
                reward=0.0,
                reward_detail=dict(reward=0.0),
                exhausted=False,
                messages=[],
            ),
        )

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
