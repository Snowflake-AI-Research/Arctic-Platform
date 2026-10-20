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

"""CLI selection for accelerator-specific correctness configs."""

from __future__ import annotations

import json
import os
import runpy
from argparse import Namespace
from pathlib import Path

import pytest

from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.selftest.config_factory import native_config


def _write_config(root: Path, gpu_type: str) -> Path:
    path = root / "model" / gpu_type / "train.config"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(native_config({"n_gpus": 8, "sp_size": 8})))
    return path


def _args(config_dir: Path, *, any_gpu: bool = False, config: str | None = None) -> Namespace:
    return Namespace(all_configs=config is None, any_gpu=any_gpu, config_dir=str(config_dir), config=config)


@pytest.mark.parametrize("auth_mode", [None, "disabled", "token"])
def test_cli_entry_preserves_ray_auth_mode(monkeypatch, auth_mode) -> None:
    if auth_mode is None:
        monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("RAY_AUTH_MODE", auth_mode)
    monkeypatch.setattr(cli.sys, "argv", ["correctness", "--help"])

    def forbidden_restart(*args):
        pytest.fail("the correctness CLI must not reexec to change Ray authentication")

    monkeypatch.setattr(os, "execv", forbidden_restart)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("arctic_platform.correctness", run_name="__main__")

    assert exited.value.code == 0
    assert os.environ.get("RAY_AUTH_MODE") == auth_mode


def test_all_configs_selects_only_the_local_gpu_family(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    h200 = _write_config(config_dir, "h200")
    _write_config(config_dir, "b200")
    monkeypatch.setattr(cli, "local_gpu_type", lambda: "h200")

    assert cli._resolve_configs(_args(config_dir)) == [h200]


def test_all_configs_any_gpu_keeps_every_gpu_family(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    h200 = _write_config(config_dir, "h200")
    b200 = _write_config(config_dir, "b200")
    monkeypatch.setattr(cli, "local_gpu_type", lambda: "h200")

    assert cli._resolve_configs(_args(config_dir, any_gpu=True)) == sorted([h200, b200])


def test_explicit_config_is_not_filtered_before_validation(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    b200 = _write_config(config_dir, "b200")
    monkeypatch.setattr(cli, "local_gpu_type", lambda: "h200")

    assert cli._resolve_configs(_args(config_dir, config=str(b200))) == [b200]
