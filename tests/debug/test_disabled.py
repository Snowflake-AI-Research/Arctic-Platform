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

"""Disabled essential hooks do not import training stacks or inspect native objects."""

import json
import subprocess
import sys
from pathlib import Path

from arctic_platform.debug import deepspeed_zero_initialization
from arctic_platform.debug import weight_sync


def test_debug_imports_do_not_load_optional_training_or_comparison_dependencies():
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib
import importlib.abc
import json
import sys

sys.path[:] = json.loads(sys.argv[1])
forbidden = {"torch", "deepspeed", "ray", "vllm", "dss", "transformers"}

class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden or fullname.startswith("arctic_platform.correctness"):
            raise AssertionError(f"eager optional import: {fullname}")

sys.meta_path.insert(0, RejectTrainingImports())
for name in ("weight_sync", "deepspeed_zero_initialization"):
    importlib.import_module(f"arctic_platform.debug.{name}")
assert not forbidden.intersection(sys.modules)
"""
    paths = [str(root), *sys.path]
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, json.dumps(paths)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_disabled_initialization_keeps_config_identity():
    config = {"zero_optimization": {"stage": 2}}
    with deepspeed_zero_initialization.maybe_trace_zero_initialization(config, 0) as selected:
        assert selected is config


def test_disabled_send_observer_neither_inspects_worker_nor_consumes_weights(monkeypatch):
    monkeypatch.delenv(weight_sync.VERIFY_ENV, raising=False)

    def stream():
        raise AssertionError("a disabled observer consumed the stream")
        yield

    weights = stream()
    with weight_sync.maybe_observe_weight_send(object(), weights) as evidence:
        assert evidence == {}
    result = {"sent": 1}
    assert weight_sync.maybe_annotate_weight_sync_result(result, evidence) is result
    weights.close()
