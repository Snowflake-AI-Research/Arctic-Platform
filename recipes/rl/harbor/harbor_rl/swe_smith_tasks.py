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
"""Write SWE-smith code-localization rows as Harbor task directories.

Standard library only: ``prepare_data.py`` runs this as a ``uv run`` script,
without Arctic Platform or the training dependencies installed.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

# Copied into each task's tests/ and run there by plain python3, so it must stay standard-library only.
LOCALIZATION_REWARD_FILE = (
    Path(__file__).resolve().parents[2] / "openhands" / "code_localization" / "openhands_localization" / "reward.py"
)

_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,40}$")

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


def _require(pattern: re.Pattern[str], value: str, field: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"Dataset row has an invalid {field}: {value!r}")
    return value


def materialize_harbor_task_from_row(
    instance: dict[str, Any], tasks_root: Path, *, reward_file: Path = LOCALIZATION_REWARD_FILE
) -> Path:
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
    task_dir = (tasks_root / instance_id).resolve()
    if not task_dir.is_relative_to(tasks_root.resolve()):
        raise ValueError(f"Task directory {tasks_root / instance_id} resolves outside {tasks_root}")
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
        # Images that already ship git (such as the Snowflake sandbox catalog image) have no apt-get.
        (
            "RUN command -v git >/dev/null 2>&1"
            " || (apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*)"
        ),
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
    shutil.copyfile(reward_file, tests_dir / "localization_reward.py")
    test_sh.write_text(_LOCALIZATION_VERIFIER_SH, encoding="utf-8")
    test_sh.chmod(0o755)
    return task_dir
