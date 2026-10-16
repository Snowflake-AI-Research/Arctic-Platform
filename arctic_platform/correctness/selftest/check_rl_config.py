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

"""A job that both trains and samples is reinforcement learning, and the reference cannot serve it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import reference_tests
from arctic_platform.correctness.harness.registry import registered_tests
from arctic_platform.correctness.harness.spec import TestSpec as Spec
from arctic_platform.correctness.selftest.config_factory import native_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3.6-35b-a3b/h200/train-rl-16gpus-64k.config"
SPEC = ROOT / "specs/qwen3.6-35b-a3b-h200-train-rl-16gpus-64k.json"

TRAINING_CONFIG = {
    "max_seq_len": 4096,
    "n_gpus": 8,
    "attn_implementation": "flash_attention_3",
}
SAMPLING_CONFIG = {"max_seq_len": 4096, "n_gpus": 8}


def _write(tmp_path, training: dict, sampling: dict | None = None):
    path = tmp_path / "job.config"
    path.write_text(json.dumps(native_config(training, sampling=sampling)))
    return load_config(path)


def test_training_alone_is_not_rl(tmp_path):
    assert _write(tmp_path, TRAINING_CONFIG).is_rl is False


def test_training_with_sampling_is_rl(tmp_path):
    assert _write(tmp_path, TRAINING_CONFIG, SAMPLING_CONFIG).is_rl is True


def test_prime_rl_provider_alone_is_not_rl(tmp_path):
    """The training sub-job's own options never decide this; a supervised job may select PrimeRL."""
    training = json.loads(json.dumps(TRAINING_CONFIG))
    training["model_provider"] = "prime_rl"
    training["prime_rl"] = {"fused_cross_entropy": False}
    assert _write(tmp_path, training).is_rl is False


def test_every_reference_check_is_registered():
    """The checks an RL config is refused for are exactly the registered ones that need the reference.

    Not an equality: a check that compares Arctic Platform against Arctic Platform declares ``compares_to_reference=False`` and is
    registered without belonging to this set.
    """
    assert set(reference_tests()) <= set(registered_tests())


@pytest.mark.parametrize("check", ["rl-weight-sync", "rl-router-replay"])
def test_frozen_rl_checks_do_not_require_the_reduced_reference(check):
    cfg = load_config(CONFIG)
    spec = Spec.read(SPEC)
    assert cfg.is_rl
    assert cfg.model_name == spec.model.source_checkpoint == "Qwen/Qwen3.6-35B-A3B"
    assert spec.model.cache_path == "/data-fast/base-models/synthetic/Qwen3.6-35B-A3B-4L"
    assert spec.model.num_hidden_layers == 4
    assert check in spec.applicable_tests
    assert registered_tests()[check].compares_to_reference is False
    assert check not in reference_tests()


@pytest.mark.parametrize("check", ["rl-weight-sync", "rl-router-replay"])
def test_frozen_rl_cortex_capability_guard_precedes_check_setup(tmp_path, monkeypatch, check):
    cfg = load_config(CONFIG)
    spec = Spec.read(SPEC)
    entry = registered_tests()[check]
    monkeypatch.setattr(entry, "fn", lambda ctx: pytest.fail("unsupported check must not load a model or client"))
    results = runner.execute_rl(
        cfg,
        spec,
        vocab_size=128,
        select=[check],
        workdir=tmp_path,
        client_backend=CortexConfig(base_url="http://unused"),
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.FAIL
    assert results[0].summary == "check not supported on Cortex"
    assert results[0].reason == runner.cortex_unavailable_reason(entry)
    assert "worker-local" in results[0].reason
