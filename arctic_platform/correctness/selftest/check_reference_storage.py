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

"""Direct reference entrypoints reject unsafe tensor storage before allocation."""

from pathlib import Path

import pytest

from arctic_platform.correctness.reference import hf_single_gpu
from arctic_platform.correctness.reference import optimizer_step


class AllocationReached(RuntimeError):
    pass


@pytest.mark.parametrize("entrypoint", ["reference", "optimizer"])
@pytest.mark.parametrize(
    "destination", ["/code/optimizer", "/tmp/optimizer", "/data-fast-other/optimizer", "/data-fast"]
)
def test_reference_tensor_output_rejected_before_allocation(monkeypatch, entrypoint, destination):
    def allocate(*args, **kwargs):
        raise AllocationReached("unsafe output reached model or optimizer allocation")

    monkeypatch.setattr(hf_single_gpu, "build_engine", allocate)
    monkeypatch.setattr(optimizer_step, "master_dtype", allocate)
    with pytest.raises(ValueError, match="/data-fast"):
        if entrypoint == "reference":
            hf_single_gpu.run("unused", None, optimizer_output_dir=destination)
        else:
            optimizer_step.run_adamw_step(
                None,
                {},
                {},
                learning_rate=1e-3,
                gradient_clipping=None,
                optimizer_dtype="float32",
                output_dir=Path(destination),
            )


@pytest.mark.parametrize("entrypoint", ["reference", "optimizer"])
def test_reference_tensor_output_rejects_fast_symlink_escape(tmp_path, monkeypatch, entrypoint):
    alias = tmp_path / "escape"
    alias.symlink_to("/tmp", target_is_directory=True)
    test_reference_tensor_output_rejected_before_allocation(monkeypatch, entrypoint, str(alias / "optimizer"))


def test_reference_without_optimizer_output_still_reaches_engine(monkeypatch):
    def allocate(*args, **kwargs):
        raise AllocationReached("reference engine reached")

    monkeypatch.setattr(hf_single_gpu, "build_engine", allocate)
    with pytest.raises(AllocationReached, match="reference engine"):
        hf_single_gpu.run("unused", None)


def test_reference_fast_optimizer_output_still_reaches_engine(tmp_path, monkeypatch):
    def allocate(*args, **kwargs):
        raise AllocationReached("reference engine reached")

    monkeypatch.setattr(hf_single_gpu, "build_engine", allocate)
    with pytest.raises(AllocationReached, match="reference engine"):
        hf_single_gpu.run("unused", None, optimizer_output_dir=str(tmp_path / "optimizer"))
    assert not (tmp_path / "optimizer").exists()
