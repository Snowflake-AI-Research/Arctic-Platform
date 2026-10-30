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

"""The unpacked gas loop stays on DeepSpeed's micro-step counter."""

from __future__ import annotations

import torch

from arctic_platform.common.unpacked_gas import assert_ready_for_final_step
from arctic_platform.common.unpacked_gas import run_unpacked_gas


class _Engine:
    def __init__(self, gas: int):
        self.device = torch.device("cpu")
        self.micro_steps = 0
        self._is_gradient_accumulation_boundary = None
        self._gas = gas
        self.steps = 0
        self.trained = False

    def gradient_accumulation_steps(self):
        return self._gas

    def is_gradient_accumulation_boundary(self):
        if self._is_gradient_accumulation_boundary is not None:
            return self._is_gradient_accumulation_boundary
        return (self.micro_steps + 1) % self._gas == 0

    def train(self):
        self.trained = True

    def step(self):
        self.steps += 1
        self.micro_steps += 1


def test_unpacked_gas_steps_between_chunks_and_leaves_the_last_step(monkeypatch):
    import arctic_platform.rl.processors as processors

    seen = []

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing
        seen.append((int(batch["input_ids"].shape[0]), kwargs["pack"], kwargs["backward"]))
        return {"avg_loss": float(batch["input_ids"].shape[0]), "metrics": {}}

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(processors, "resolve_packed_loss_reduction", lambda *args, **kwargs: None)
    engine = _Engine(gas=2)
    batch = {"input_ids": torch.arange(4).reshape(4, 1)}
    outputs = run_unpacked_gas(
        engine,
        gas=2,
        batch_data=batch,
        meta_data={},
        processing={},
        backward=True,
        device="cpu",
    )
    assert seen == [(2, False, True), (2, False, True)]
    assert engine.steps == 1
    assert engine.micro_steps == 1
    assert engine._is_gradient_accumulation_boundary is None
    assert outputs["avg_loss"] == 2.0
    assert_ready_for_final_step(engine)
    engine.step()
    assert engine.micro_steps == 2


def test_unpacked_gas_rejects_a_sticky_boundary_override():
    engine = _Engine(gas=1)
    engine._is_gradient_accumulation_boundary = True
    batch = {"input_ids": torch.ones(1, 1)}
    try:
        run_unpacked_gas(
            engine,
            gas=1,
            batch_data=batch,
            meta_data={},
            processing={},
            backward=True,
            device="cpu",
        )
    except RuntimeError as err:
        assert "boundary override" in str(err)
    else:
        raise AssertionError("sticky override was accepted")


def test_a_failure_after_a_between_chunk_step_marks_the_cycle_broken(monkeypatch):
    import arctic_platform.rl.processors as processors

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing, kwargs
        if int(batch["input_ids"][0]) == 2:
            raise RuntimeError("nonfinite")
        return {"avg_loss": 1.0, "metrics": {}}

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(processors, "resolve_packed_loss_reduction", lambda *args, **kwargs: None)
    engine = _Engine(gas=2)
    batch = {"input_ids": torch.tensor([[0], [1], [2], [3]])}
    try:
        run_unpacked_gas(
            engine,
            gas=2,
            batch_data=batch,
            meta_data={},
            processing={},
            backward=True,
            device="cpu",
        )
    except RuntimeError as err:
        assert "nonfinite" in str(err)
    else:
        raise AssertionError("the second chunk was accepted")
    assert engine.steps == 1
    try:
        assert_ready_for_final_step(engine)
    except RuntimeError as err:
        assert "interrupted" in str(err)
    else:
        raise AssertionError("a broken cycle was treated as ready")


def test_forward_only_does_not_step(monkeypatch):
    import arctic_platform.rl.processors as processors

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing, kwargs
        return {"batch": {"logprobs": batch["input_ids"].float()}}

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    engine = _Engine(gas=2)
    batch = {"input_ids": torch.arange(4).reshape(4, 1)}
    outputs = run_unpacked_gas(
        engine,
        gas=2,
        batch_data=batch,
        meta_data={},
        processing={},
        backward=False,
        device="cpu",
        return_tensors=True,
    )
    assert engine.steps == 0
    assert engine.trained is False
    assert outputs["batch"]["logprobs"].shape == (4, 1)
