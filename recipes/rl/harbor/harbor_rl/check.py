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
"""Run the first training task once, with an agent that does nothing, before any GPUs are provisioned.

A wrong connection, a missing external access integration, or a Dockerfile step
that fails on the sandbox image otherwise shows up only after the GPUs are up,
as a step where every rollout errored.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from arctic_platform.integrations.harbor.adapter import SNOWFLAKE_HARBOR_ENV
from arctic_platform.integrations.harbor.adapter import HarborTrialError
from arctic_platform.integrations.harbor.adapter import close_shared_resources
from arctic_platform.integrations.harbor.adapter import require_task_dir
from arctic_platform.integrations.harbor.adapter import run_harbor_trial

SKIP_CHECK_VAR = "HARBOR_SKIP_TASK_CHECK"
SANDBOX_LIMIT = 100


def task_check_skipped() -> bool:
    return os.environ.get(SKIP_CHECK_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


class CheckAgent(BaseAgent):
    """Runs ``pwd`` in the environment and nothing else."""

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs: Any) -> None:
        forward = {k: v for k, v in kwargs.items() if k in ("logger", "mcp_servers", "skills_dir")}
        super().__init__(logs_dir=Path(logs_dir), model_name=model_name, **forward)

    @staticmethod
    def name() -> str:
        return "harbor-rl-check"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:  # noqa: ARG002
        return

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:  # noqa: ARG002
        result = await environment.exec("pwd")
        if result.return_code != 0:
            raise RuntimeError(
                f"`pwd` exited {result.return_code} in the environment: {result.stderr or result.stdout!r}"
            )


def first_task_dir(train_file: str, tasks_root: Path, prepare_task_fn: Callable[[dict, Path], Any]) -> Path:
    import pyarrow.parquet as pq

    rows = pq.read_table(train_file).slice(0, 1).to_pylist()
    if not rows:
        raise SystemExit(f"ERROR: {train_file} has no rows.")
    row = rows[0]
    if row.get("harbor_task_dir"):
        return require_task_dir(Path(str(row["harbor_task_dir"])))
    return require_task_dir(Path(prepare_task_fn(row, tasks_root)))


def warn_live_sandboxes(env_kwargs: dict[str, Any], rollouts: int) -> None:
    from arctic_platform.integrations.harbor.snowflake_sandbox.environment import SnowflakeSandboxEnvironment

    try:
        live = SnowflakeSandboxEnvironment.live_sandboxes(**env_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: couldn't list Snowflake sandboxes: {exc}", file=sys.stderr)
        return
    if live and len(live) + rollouts > SANDBOX_LIMIT:
        print(
            f"WARNING: {len(live)} sandboxes from an earlier Harbor run are still live on this connection. With this"
            f" run's {rollouts} rollouts per step that is over Snowflake's {SANDBOX_LIMIT}-sandbox limit, so creates"
            " will wait or fail. If no other run uses them, terminate them as recipes/rl/harbor/README.md shows.",
            file=sys.stderr,
        )


def check_first_task(
    train_file: str,
    *,
    env: str,
    env_kwargs: dict[str, Any],
    rollouts: int,
    prepare_task_fn: Callable[[dict, Path], Any],
) -> None:
    """Raise ``SystemExit`` if the first task in ``train_file`` can't start, run a command, and be graded on ``env``."""
    with tempfile.TemporaryDirectory(prefix="harbor-rl-check-") as tmp:
        task_dir = first_task_dir(train_file, Path(tmp) / "tasks", prepare_task_fn)
        if env == SNOWFLAKE_HARBOR_ENV:
            warn_live_sandboxes(env_kwargs, rollouts)
        where = env.rpartition(":")[2]
        print(f"Checking that task {task_dir.name} runs on {where} before provisioning GPUs ...", flush=True)
        start = time.monotonic()
        try:
            result = asyncio.run(
                run_harbor_trial(
                    task_dir=task_dir,
                    trials_dir=Path(tmp) / "trials",
                    trial_name="check",
                    agent=f"{__name__}:CheckAgent",
                    env=env,
                    # CheckAgent never calls a model.
                    model_name="check",
                    base_url="http://127.0.0.1:9",
                    api_key="check",
                    env_kwargs=env_kwargs,
                )
            )
            error = result.get("exception")
        except HarborTrialError as exc:
            error = exc.reason
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            close_shared_resources(env)
    if error:
        raise SystemExit(
            f"ERROR: task {task_dir.name} failed on {where}, so every rollout would too:\n{error}\n"
            f"       Fix that, or set {SKIP_CHECK_VAR}=1 to skip this check."
        )
    print(f"Task {task_dir.name} ran and was graded in {time.monotonic() - start:.0f}s.", flush=True)
