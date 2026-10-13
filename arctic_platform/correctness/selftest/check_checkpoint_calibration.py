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

"""Calibration uses regression's per-arm accumulation and request packing."""

import contextlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.harness.arms import ArmDefinition
from arctic_platform.correctness.harness.batches import build_batch
from arctic_platform.correctness.harness.seeds import SEED
from arctic_platform.correctness.harness.spec import ArmSpec


@pytest.mark.parametrize("slots,sp_size", [(8, 1), (16, 1), (8, 8), (16, 8)])
@pytest.mark.parametrize("provider", ["huggingface", "prime_rl"])
@pytest.mark.parametrize("reviewed_arms", [False, True])
def test_calibration_uses_arm_specific_accumulation_and_identical_work(
    monkeypatch, tmp_path, slots, sp_size, provider, reviewed_arms
):
    training = {
        "n_gpus": slots,
        "sp_size": sp_size,
        "model_provider": provider,
        "mb_spec": {"max_tokens_per_mb": 64},
        "ds_config": {"gradient_accumulation_steps": 1},
    }
    original = deepcopy(training)
    definitions = [ArmDefinition(name, slots * gas, 64) for name, gas in (("gas1", 1), ("gas4", 4))]
    arms = [
        (
            ArmSpec(arm.name, arm.global_batch_size, arm.max_seq_len, arm.total_tokens, 1, 1, 0.0)
            if reviewed_arms
            else arm
        )
        for arm in definitions
    ]
    transports = []

    class Transport:
        def __init__(self, workdir, config, model, seed, **kwargs):
            assert config == original
            assert model == "/models/reduced" and seed == SEED
            self.workdir, self.options, self.jobs = workdir, kwargs, []
            transports.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextlib.contextmanager
        def job(self):
            calls = []
            self.jobs.append(calls)

            def step(body, *, learning_rate):
                calls.append((body, learning_rate))
                return float(len(calls))

            yield SimpleNamespace(fwd_bwd_step=step)

    monkeypatch.setattr(checkpoint_resume, "GatewayTransport", Transport)
    raw, where = checkpoint_resume.measure_undeterministic_loss_range(
        training,
        "/models/reduced",
        slots=slots,
        vocab_size=128,
        arms=arms,
        workdir=tmp_path,
        attn_implementation="flash_attention_3",
    )

    assert len(transports) == 2
    assert raw == 0.0 and where == "gas1 loss at iteration 1"
    for arm, transport in zip(definitions, transports, strict=True):
        gas = arm.microbatches(64, slots // sp_size)
        assert transport.workdir == tmp_path / arm.name
        assert transport.options == {
            "slots": slots,
            "attn_implementation": "flash_attention_3",
            "determinism": "off",
            "gradient_accumulation_steps": gas,
        }
        assert len(transport.jobs) == 3
        for job in transport.jobs:
            assert len(job) == 10
            for iteration, (body, rate) in enumerate(job, start=1):
                batch = build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, 128, seed=SEED + iteration)
                wires = body["batch"] if isinstance(body["batch"], list) else [body["batch"]]
                assert len(wires) == gas
                for field in ("input_ids", "position_ids"):
                    assert torch.equal(torch.cat([wire[field] for wire in wires]), getattr(batch, field))
                labels = batch.shifted_labels() if provider == "prime_rl" else batch.labels
                assert torch.equal(torch.cat([wire["labels"] for wire in wires]), labels)
                assert body["processing"] == {"loss_fn": "sft"}
                assert rate == checkpoint_resume.LEARNING_RATE
        for run in transport.jobs[1:]:
            assert all(body is prior for (body, _), (prior, _) in zip(run, transport.jobs[0], strict=True))
    assert training == original


def test_qwen38_calibration_requires_four_gas4_microbatches():
    for slots in (8, 16):
        assert ArmDefinition("gas1", slots, 8192).microbatches(10240, slots) == 1
        assert ArmDefinition("gas4", slots * 4, 8192).microbatches(10240, slots) == 4
