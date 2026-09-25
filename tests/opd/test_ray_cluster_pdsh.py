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

import subprocess
from pathlib import Path

import pytest

import arctic_platform.common.ray_cluster as ray_cluster


def test_pdsh_failure_after_head_start_tears_down_cluster(monkeypatch, tmp_path: Path):
    temp = tmp_path / "ray_arctic_pdsh_fail"
    temp.mkdir()
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_cluster.ray, "shutdown", lambda: None)
    monkeypatch.setattr(ray_cluster, "_ray_bin", lambda: "ray")
    monkeypatch.setattr(ray_cluster, "_peer_hosts", lambda: ["10.0.0.2"])
    monkeypatch.setattr(ray_cluster, "read_ray_address", lambda _d: "10.0.0.1:6379")
    monkeypatch.setenv("ARL_RAY_TEMP_DIR", str(temp))
    monkeypatch.setattr(
        ray_cluster.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0] if args else [], 0),
    )
    monkeypatch.setattr(
        ray_cluster,
        "_pdsh",
        lambda *args, **kwargs: subprocess.CompletedProcess(["pdsh"], 1),
    )

    seen: dict[str, bool] = {}
    orig_shutdown = ray_cluster._shutdown

    def wrapped_shutdown() -> None:
        seen["owned_head"] = ray_cluster._spawned_cluster
        orig_shutdown()

    monkeypatch.setattr(ray_cluster, "_shutdown", wrapped_shutdown)

    with pytest.raises(RuntimeError, match="pdsh ray start"):
        ray_cluster.init_ray_cluster(auto_attach=False)

    assert seen["owned_head"] is True
    assert ray_cluster._spawned_cluster is False
    assert ray_cluster._spawned_temp_dir is None
