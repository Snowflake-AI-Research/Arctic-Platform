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

"""Disabled debug hooks neither load training stacks nor alter native objects."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.debug import deepspeed_zero_initialization
from arctic_platform.debug import inference
from arctic_platform.debug import lora
from arctic_platform.debug import training
from arctic_platform.debug import weight_sync


@pytest.fixture(autouse=True)
def disabled_debug(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("ARCTIC_PLATFORM_", "ARCTIC_CORRECTNESS_", "ARCTIC_INFERENCE_WEIGHT_SYNC_")):
            monkeypatch.delenv(name)


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
for name in ("inference", "lora", "training", "weight_sync", "deepspeed_zero_initialization"):
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


def test_disabled_initialization_and_router_hooks_keep_config_identity():
    config = {"zero_optimization": {"stage": 2}}
    with deepspeed_zero_initialization.maybe_trace_zero_initialization(config, 0) as selected:
        assert selected is config
    assert training.maybe_configure_router_replay_trace(config, object(), 0) is config
    assert weight_sync.maybe_configure_broadcast_training(config) is config
    training.maybe_record_router_replay_received("sample", object())
    training.maybe_begin_router_replay_call(object(), {}, backward=True)
    training.maybe_complete_router_replay_call(object())


def test_disabled_broadcast_observer_preserves_a_lazy_stream_and_native_result():
    def stream():
        raise AssertionError("a disabled observer consumed the stream")
        yield

    weights = stream()
    with weight_sync.maybe_observe_broadcast_send(object(), weights, object()) as (observed, evidence):
        assert observed is weights
        assert evidence == {}
    result = {"sent": 1}
    assert weight_sync.maybe_annotate_weight_sync_result(result, evidence) is result
    weights.close()


def test_disabled_worker_send_observer_does_not_require_an_engine():
    worker = SimpleNamespace(_weight_sender=None)
    with weight_sync.maybe_observe_weight_send(worker, object()) as evidence:
        assert evidence == {}


def test_disabled_lora_hooks_do_not_require_worker_or_sampler_attributes():
    sentinel = object()
    assert lora.maybe_admit_model_construction(sentinel, sentinel) is None
    assert lora.maybe_lora_training_worker(sentinel) is sentinel
    assert lora.maybe_lora_replica_pool(sentinel) is sentinel
    assert lora.maybe_project_lora_transfer_schedule(sentinel) is sentinel
    assert lora.maybe_lora_max_param_bytes(sentinel, 128) == 128
    lora.maybe_release_lora_worker(sentinel)


def test_disabled_startup_diagnostics_do_not_create_a_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert inference.maybe_begin_inference_startup_attempt(1) is None
    inference.maybe_finish_inference_startup_attempt(None, "error", RuntimeError("native"))
    inference.maybe_install_inference_startup_error_producer()
    inference.maybe_install_inference_warmup_observer()
    assert list(tmp_path.iterdir()) == []


def test_debug_config_consumption_does_not_mutate_the_callers_dictionary(monkeypatch):
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "0")
    config = {"zero_optimization": {"stage": 2}, weight_sync.TRAINING_OPTION: True}
    selected = weight_sync.maybe_configure_broadcast_training(config)
    assert selected == {"zero_optimization": {"stage": 2}}
    assert config[weight_sync.TRAINING_OPTION] is True
    assert selected["zero_optimization"] is config["zero_optimization"]
    assert os.environ[weight_sync.VERIFY_ENV] == "1"
