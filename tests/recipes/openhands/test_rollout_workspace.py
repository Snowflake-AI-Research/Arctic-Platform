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
"""Each rollout's clone is removed once its conversation ends."""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
from openhands_localization import rollout

INSTANCE = dict(repo="org/repo", base_commit="abc", instance_id="org__repo-1", problem_statement="bug")


@pytest.fixture
def fake_openhands(monkeypatch):
    modules = {
        "openhands": ModuleType("openhands"),
        "openhands.sdk": ModuleType("openhands.sdk"),
        "openhands.sdk.tool": ModuleType("openhands.sdk.tool"),
        "openhands.tools": ModuleType("openhands.tools"),
        "openhands.tools.terminal": ModuleType("openhands.tools.terminal"),
        "openhands_localization.finish": ModuleType("openhands_localization.finish"),
    }
    sdk = modules["openhands.sdk"]
    sdk.LLM = sdk.Agent = sdk.Conversation = sdk.Tool = lambda *args, **kwargs: SimpleNamespace(**kwargs)
    modules["openhands.sdk.tool"].register_tool = lambda name, cls: None
    modules["openhands.tools.terminal"].TerminalTool = SimpleNamespace(name="terminal")
    modules["openhands_localization.finish"].LocalizationFinishTool = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _clone_into(clones):
    def clone(repo, commit, instance_id, destination, patch):
        working_dir = destination / instance_id
        working_dir.mkdir(parents=True)
        (working_dir / "a.py").write_text("x = 1\n")
        clones.append(destination)
        return working_dir

    return clone


def _run(tmp_path):
    return rollout.run_conversation(INSTANCE, "model", "http://proxy", "key", 10, str(tmp_path), "train")


def test_clone_is_removed_after_the_conversation(fake_openhands, monkeypatch, tmp_path):
    clones = []
    monkeypatch.setattr(rollout, "clone_instance", _clone_into(clones))
    monkeypatch.setattr(rollout, "run_and_close", lambda conversation, prompt: [])
    result = _run(tmp_path)
    assert result == dict(messages=[], locations=None)
    assert len(clones) == 1
    assert not clones[0].exists()


def test_clone_is_removed_when_the_conversation_fails(fake_openhands, monkeypatch, tmp_path):
    clones = []

    def fail(conversation, prompt):
        raise RuntimeError("agent crashed")

    monkeypatch.setattr(rollout, "clone_instance", _clone_into(clones))
    monkeypatch.setattr(rollout, "run_and_close", fail)
    with pytest.raises(RuntimeError, match="agent crashed"):
        _run(tmp_path)
    assert not clones[0].exists()
