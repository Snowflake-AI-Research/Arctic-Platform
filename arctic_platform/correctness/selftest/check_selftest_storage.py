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

"""The public selftest command keeps even tiny checkpoint fixtures on fast storage."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.correctness.harness import execution as cli


def test_selftest_default_uses_unique_fast_scratch(monkeypatch):
    commands = []
    monkeypatch.setattr(cli, "correctness_workdir", lambda prefix: Path("/data-fast/tmp/unique-selftests"))
    monkeypatch.setattr(pytest, "main", lambda command: commands.append(command) or 0)
    assert cli.cmd_selftest(SimpleNamespace(runner_args=[])) == 0
    assert commands[0][-2:] == ["--basetemp", "/data-fast/tmp/unique-selftests/pytest"]
    assert commands[0][0] == "-sv"


@pytest.mark.parametrize(
    "arguments",
    [["--basetemp=/tmp/tests"], ["--basetemp", "/code/tests"], ["--basetemp", "/data-fast"], ["--basetemp"]],
)
def test_selftest_rejects_off_fast_override_before_pytest(monkeypatch, arguments):
    def unexpected(*args, **kwargs):
        pytest.fail("unsafe storage must be rejected before allocation or pytest")

    monkeypatch.setattr(cli, "correctness_workdir", unexpected)
    monkeypatch.setattr(pytest, "main", unexpected)
    with pytest.raises(ValueError, match="/data-fast"):
        cli.cmd_selftest(SimpleNamespace(runner_args=arguments))


@pytest.mark.parametrize("joined", [False, True])
def test_selftest_accepts_explicit_fast_override(tmp_path, monkeypatch, joined):
    commands = []
    arguments = [f"--basetemp={tmp_path / 'tests'}"] if joined else ["--basetemp", str(tmp_path / "tests")]
    monkeypatch.setattr(pytest, "main", lambda command: commands.append(command) or 0)
    assert cli.cmd_selftest(SimpleNamespace(runner_args=arguments)) == 0
    assert commands[0][-len(arguments) :] == arguments
    assert not (tmp_path / "tests").exists()
