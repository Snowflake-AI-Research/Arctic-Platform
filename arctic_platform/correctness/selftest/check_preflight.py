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

"""The multi-node preflight finds an interrupted extension build and names it."""

from __future__ import annotations

import subprocess

import pytest

from arctic_platform.correctness.harness import preflight


def _run_node_script(root) -> str:
    return subprocess.run(
        ["bash", "-c", preflight.node_script(str(root))], capture_output=True, text=True, check=True
    ).stdout


def test_a_lock_without_a_shared_object_is_reported(tmp_path) -> None:
    build = tmp_path / "py312_cu130" / "fused_adam"
    build.mkdir(parents=True)
    (build / "lock").touch()
    (build / "fused_adam_frontend.o").touch()

    found = preflight.parse(_run_node_script(tmp_path))

    assert [directory for _, directory in found] == [str(build)]


def test_a_linked_build_and_a_clean_build_are_not_reported(tmp_path) -> None:
    linked = tmp_path / "py312_cu130" / "fused_adam"
    linked.mkdir(parents=True)
    (linked / "lock").touch()
    (linked / "fused_adam.so").touch()
    (tmp_path / "py312_cu130" / "cpu_adam").mkdir()

    assert preflight.parse(_run_node_script(tmp_path)) == []


def test_parse_reads_hosts_behind_the_pdsh_prefix() -> None:
    output = (
        "hostfile=/data-fast/hostfile\n"
        f"10.0.0.2: {preflight.MARKER} node-1 /home/u/.cache/torch_extensions/py312_cu130/fused_adam\n"
    )

    assert preflight.parse(output) == [("node-1", "/home/u/.cache/torch_extensions/py312_cu130/fused_adam")]


def _hostfile(tmp_path, lines: int):
    path = tmp_path / "hostfile"
    path.write_text("".join(f"10.0.0.{i} slots=8\n" for i in range(1, lines + 1)))
    return str(path)


def test_the_run_stops_with_every_interrupted_directory_named(monkeypatch, tmp_path) -> None:
    out = (
        f"10.0.0.1: {preflight.MARKER} node-0 /c/py312_cu130/fused_adam\n"
        f"10.0.0.2: {preflight.MARKER} node-1 /c/py312_cu130/cpu_adam\n"
    )
    monkeypatch.setattr(
        preflight.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr="")
    )

    with pytest.raises(SystemExit) as stopped:
        preflight.ensure_no_interrupted_builds(_hostfile(tmp_path, 2))

    assert "node-0: /c/py312_cu130/fused_adam" in str(stopped.value)
    assert "node-1: /c/py312_cu130/cpu_adam" in str(stopped.value)


def test_a_single_node_allocation_is_not_inspected(monkeypatch, tmp_path) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a one-node allocation must not be inspected over ssh")

    monkeypatch.setattr(preflight.subprocess, "run", refuse)

    preflight.ensure_no_interrupted_builds(_hostfile(tmp_path, 1))


def test_without_a_named_gateway_nothing_is_inspected(monkeypatch, tmp_path) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a gateway the harness starts itself serves this node alone")

    monkeypatch.delenv("DSS_GATEWAY_URL", raising=False)
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", _hostfile(tmp_path, 2))
    monkeypatch.setattr(preflight.subprocess, "run", refuse)

    preflight.ensure_no_interrupted_builds()


def test_an_inspection_that_could_not_run_stops_the_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="ssh: timeout"),
    )

    with pytest.raises(SystemExit, match="ssh: timeout"):
        preflight.ensure_no_interrupted_builds(_hostfile(tmp_path, 2))
