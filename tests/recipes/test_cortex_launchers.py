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

from __future__ import annotations

import re
from pathlib import Path

import pytest

_RECIPES = Path(__file__).resolve().parents[2] / "recipes"
_SPEC = re.compile(r'ARCTIC_PLATFORM_SPEC="\$\{ARCTIC_PLATFORM_SPEC:-\$\{AP_ROOT\}\[([^\]]*)\]\}"')


def _cortex_launchers() -> list[tuple[Path, list[str]]]:
    found = []
    for script in sorted(_RECIPES.rglob("*.sh")):
        match = _SPEC.search(script.read_text(encoding="utf-8"))
        if match:
            extras = [e.strip() for e in match.group(1).split(",")]
            if "cortex" in extras:
                found.append((script, extras))
    return found


def test_cortex_launchers_are_found() -> None:
    assert len(_cortex_launchers()) >= 3


@pytest.mark.parametrize(
    "script,extras", _cortex_launchers(), ids=lambda v: v.parent.name if isinstance(v, Path) else ""
)
def test_cortex_launchers_do_not_install_the_local_sampler(script: Path, extras: list[str]) -> None:
    # A Cortex driver samples remotely; [rl] adds kvcached, which fails to build on a machine without torch.
    assert "rl" not in extras, f"{script.relative_to(_RECIPES.parent)} installs [rl]; use [sft,cortex]"
