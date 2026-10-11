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
"""Clone one training instance onto the driver.

Adapted from ``src/utils/instance.py`` in
https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2

The sandbox is a local checkout. OpenHands is given that directory. It is not
a Docker or Modal workspace. ``git clone`` does not create missing parents, so
the per-rollout directory is created first.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(args: list[str], *, input_text: str | None = None) -> None:
    result = subprocess.run(args, input=input_text, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{' '.join(args)} failed ({result.returncode}): {detail}")


def clone_instance(
    repo: str,
    commit: str | None,
    instance_id: str,
    workspace: Path,
    patch: str | None = None,
) -> Path:
    """Clone ``repo`` at ``commit`` and apply ``patch`` when the row has one."""
    workspace.mkdir(parents=True, exist_ok=True)
    destination = workspace / f"{repo.replace('/', '_')}_{instance_id}"
    if not destination.exists():
        _git(["git", "clone", f"https://github.com/{repo}.git", str(destination)])
    if commit is not None and commit != "":
        _git(["git", "-C", str(destination), "checkout", commit])
    if patch is not None and patch != "":
        _git(["git", "-C", str(destination), "apply"], input_text=patch)
    return destination
