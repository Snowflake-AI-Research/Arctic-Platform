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

import pytest
import torch

from arctic_platform.common.unpacked_gas import assert_ready_for_final_step
from arctic_platform.common.unpacked_gas import run_unpacked_gas
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


class _Engine:
    def __init__(self, gas: int, *, managed: bool = True):
        self.device = torch.device("cpu")
        self.micro_steps = 0
        self._is_gradient_accumulation_boundary = None
        self._gas = gas
        self._managed = managed
        self.steps = 0
        self.trained = False
        self.backward_losses = []

    def gradient_accumulation_steps(self):
        return self._gas

    def managed_gradient_accumulation(self):
        return self._managed

    def is_gradient_accumulation_boundary(self):
        if self._is_gradient_accumulation_boundary is not None:
            return self._is_gradient_accumulation_boundary
        return (self.micro_steps + 1) % self._gas == 0

    def train(self):
        self.trained = True

    def step(self):
        self.steps += 1
        self.micro_steps += 1

    def backward(self, loss, scale_wrt_gas=True):
        assert scale_wrt_gas is False
        self.backward_losses.append(float(loss.detach()))


def test_unpacked_gas_steps_between_chunks_and_leaves_the_last_step(monkeypatch):
    import arctic_platform.rl.processors as processors

    seen = []

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing
        assert "pack" not in kwargs
        seen.append((int(batch["input_ids"].shape[0]), kwargs["backward"]))
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
    assert seen == [(2, True), (2, True)]
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


def test_unmanaged_loop_count_varies_and_leaves_the_only_step_to_the_caller(monkeypatch):
    import arctic_platform.rl.processors as processors

    seen = []

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing
        seen.append((int(batch["input_ids"].item()), kwargs["backward"]))
        return {"avg_loss": float(batch["input_ids"].item()), "metrics": {}}

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(processors, "resolve_packed_loss_reduction", lambda *args, **kwargs: None)
    engine = _Engine(gas=1, managed=False)
    outputs = run_unpacked_gas(
        engine,
        loop_count=3,
        batch_data={"input_ids": torch.arange(3).reshape(3, 1)},
        meta_data={},
        processing={},
        backward=True,
        device="cpu",
    )
    assert seen == [(0, True), (1, True), (2, True)]
    assert outputs["avg_loss"] == 1.0
    assert engine.steps == 0
    assert_ready_for_final_step(engine)
    engine.step()
    assert engine.steps == 1


def test_forward_only_resolves_objective_reporting_weights(monkeypatch):
    import arctic_platform.rl.processors as processors
    from arctic_platform.rl.processors import PackedLossReduction

    reduction = PackedLossReduction(
        loss_scales=(1.0, 1.0),
        reporting_weights=(1.0, 3.0),
        loss_is_additive=False,
    )
    monkeypatch.setattr(processors, "resolve_packed_loss_reduction", lambda *args, **kwargs: reduction)

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing, kwargs
        return {"avg_loss": float(batch["input_ids"].item()), "metrics": {}}

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    engine = _Engine(gas=2)
    outputs = run_unpacked_gas(
        engine,
        gas=2,
        batch_data={"input_ids": torch.tensor([[1], [3]])},
        meta_data={},
        processing={"loss_fn": "grpo"},
        backward=False,
        device="cpu",
    )
    assert outputs["avg_loss"] == 2.5


def test_execute_payload_owns_only_the_shared_five_fields(monkeypatch):
    from arctic_platform.common.payload import APPayload
    from arctic_platform.common.payload import execute_payload

    payload = APPayload(
        batch_data={"input_ids": torch.arange(2).reshape(2, 1)},
        meta_data={"source": "test"},
        processing={"loss_fn": None},
        loss_object=None,
        args=(),
    )
    captured = {}

    def fake_run_unpacked_gas(engine, **kwargs):
        captured.update(kwargs)
        return {"batch": {}}

    monkeypatch.setattr("arctic_platform.common.unpacked_gas.run_unpacked_gas", fake_run_unpacked_gas)
    engine = _Engine(gas=2)
    assert execute_payload(
        engine,
        payload,
        loop_count=2,
        backward=False,
        device="cpu",
    ) == {"batch": {}}
    assert captured["batch_data"] is payload.batch_data
    assert captured["meta_data"] is payload.meta_data
    assert captured["processing"] is payload.processing
    assert captured["loss_object"] is None
    assert captured["args"] == ()
    assert captured["loop_count"] == 2


def test_scheduled_payloads_apply_source_scales_and_zero_padding(monkeypatch):
    import arctic_platform.rl.processors as processors
    from arctic_platform.common.payload import APPayload
    from arctic_platform.common.payload import ScheduledAPPayload
    from arctic_platform.common.payload import execute_payload_schedule
    from arctic_platform.rl.processors import PackedLossReduction

    reduction = PackedLossReduction(
        loss_scales=(0.25, 0.75),
        reporting_weights=(1.0, 3.0),
        loss_is_additive=False,
    )
    monkeypatch.setattr(
        processors,
        "resolve_packed_loss_reduction",
        lambda *args, **kwargs: reduction,
    )
    applied = []
    monkeypatch.setattr(
        processors,
        "apply_packed_loss_reduction",
        lambda engine, loss_tensor, scale, *, backward: applied.append(
            (float(loss_tensor), scale, backward)
        ),
    )
    validations = []

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing
        validations.append(kwargs["validate_loss_callback"])
        value = float(batch["input_ids"].item())
        return {
            "avg_loss": value,
            "metrics": {},
            "loss_tensor": torch.tensor(value),
        }

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    processing = {"loss_fn": "grpo"}
    loss_object = object()
    payloads = [
        APPayload(
            batch_data={"input_ids": torch.tensor([[value]])},
            meta_data={},
            processing=processing,
            loss_object=loss_object,
            args=(),
        )
        for value in (1, 2)
    ]
    calls = [
        ScheduledAPPayload(payloads[0], source_index=0),
        ScheduledAPPayload(payloads[1], source_index=1),
        ScheduledAPPayload(payloads[1], source_index=1, is_padding=True),
    ]
    engine = _Engine(gas=1, managed=False)
    execution = execute_payload_schedule(
        engine,
        calls,
        backward=True,
        device="cpu",
        return_tensors=True,
    )
    assert [output["avg_loss"] for output in execution.outputs] == [1.0, 2.0, 2.0]
    assert execution.loss_reduction is reduction
    assert applied == [(1.0, 0.25, True), (2.0, 0.75, True), (2.0, 0.0, True)]
    assert validations == [True, True, False]
    assert engine.steps == 0
    assert_ready_for_final_step(engine)


@pytest.mark.parametrize("managed", [True, False])
@pytest.mark.parametrize("gas", [1, 2])
@pytest.mark.parametrize("zorro", [False, True])
def test_payload_execution_parity_matrix(monkeypatch, managed, gas, zorro):
    """Both accumulation modes preserve the same AP calls and objective math."""
    import arctic_platform.rl.processors as processors
    from arctic_platform.common.payload import APPayload
    from arctic_platform.common.payload import execute_payload
    from arctic_platform.rl.processors import PackedLossReduction

    scales = tuple(1.0 / gas for _ in range(gas))
    reduction = PackedLossReduction(
        loss_scales=scales,
        reporting_weights=scales,
        loss_is_additive=True,
    )
    monkeypatch.setattr(
        processors,
        "resolve_packed_loss_reduction",
        lambda *args, **kwargs: reduction,
    )
    seen = []

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, processing
        value = float(batch["input_ids"].item())
        seen.append((value, meta.get("zorro_train_enable", False), kwargs["backward"]))
        return {
            "avg_loss": value,
            "metrics": {},
            "loss_tensor": torch.tensor(value),
        }

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    engine = _Engine(gas=gas if managed else 1, managed=managed)
    payload = APPayload(
        batch_data={"input_ids": torch.arange(1, gas + 1).reshape(gas, 1)},
        meta_data={"zorro_train_enable": zorro},
        processing={"loss_fn": "grpo"},
        loss_object=object(),
        args=(),
    )

    result = execute_payload(
        engine,
        payload,
        loop_count=gas,
        backward=True,
        device="cpu",
    )

    assert seen == [(float(index), zorro, "loss_only") for index in range(1, gas + 1)]
    assert engine.backward_losses == pytest.approx(
        [float(index) / gas for index in range(1, gas + 1)]
    )
    assert result["avg_loss"] == pytest.approx(sum(range(1, gas + 1)))
    assert engine.steps == (gas - 1 if managed else 0)
    assert_ready_for_final_step(engine)


def test_packed_schedule_matches_unpacked_reduction_and_zeroes_padding(monkeypatch):
    import arctic_platform.rl.processors as processors
    from arctic_platform.common.payload import APPayload
    from arctic_platform.common.payload import ScheduledAPPayload
    from arctic_platform.common.payload import execute_payload
    from arctic_platform.common.payload import execute_payload_schedule
    from arctic_platform.rl.processors import PackedLossReduction

    reduction = PackedLossReduction(
        loss_scales=(0.25, 0.75),
        reporting_weights=(0.25, 0.75),
        loss_is_additive=False,
    )
    monkeypatch.setattr(
        processors,
        "resolve_packed_loss_reduction",
        lambda *args, **kwargs: reduction,
    )

    def fake_run_pipeline(engine, args, batch, meta, processing, **kwargs):
        del engine, args, meta, processing, kwargs
        value = float(batch["input_ids"].item())
        return {
            "avg_loss": value,
            "metrics": {},
            "loss_tensor": torch.tensor(value),
        }

    monkeypatch.setattr(processors, "run_pipeline", fake_run_pipeline)
    processing = {"loss_fn": "grpo"}
    unpacked_engine = _Engine(gas=1, managed=False)
    unpacked = APPayload(
        batch_data={"input_ids": torch.tensor([[1], [3]])},
        meta_data={},
        processing=processing,
        loss_object=object(),
        args=(),
    )
    unpacked_result = execute_payload(
        unpacked_engine,
        unpacked,
        loop_count=2,
        backward=True,
        device="cpu",
    )

    packed_payloads = [
        APPayload(
            batch_data={"input_ids": torch.tensor([[value]])},
            meta_data={"cu_seqlens": torch.tensor([0, 1])},
            processing=processing,
            loss_object=unpacked.loss_object,
            args=(),
        )
        for value in (1, 3)
    ]
    packed_engine = _Engine(gas=1, managed=False)
    packed_result = execute_payload_schedule(
        packed_engine,
        [
            ScheduledAPPayload(packed_payloads[0], source_index=0),
            ScheduledAPPayload(packed_payloads[1], source_index=1),
            ScheduledAPPayload(
                packed_payloads[1], source_index=1, is_padding=True
            ),
        ],
        backward=True,
        device="cpu",
    )

    assert unpacked_result["avg_loss"] == pytest.approx(2.5)
    assert [output["avg_loss"] for output in packed_result.outputs] == [1.0, 3.0, 3.0]
    assert unpacked_engine.backward_losses == pytest.approx([0.25, 2.25])
    assert packed_engine.backward_losses == pytest.approx([0.25, 2.25, 0.0])


@require_torch_gpu
def test_real_deepspeed_unmanaged_requests_take_exactly_one_step():
    """Variable backward counts are each committed by one unmanaged step."""
    import deepspeed

    set_seed(17)
    model = torch.nn.Linear(1, 1, bias=False, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 1,
            "managed_gradient_accumulation": False,
            "gradient_clipping": 0.0,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {"stage": 2, "overlap_comm": False},
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
        },
        dist_init_required=False,
    )
    assert engine.managed_gradient_accumulation() is False

    expected_weight = 1.0
    for expected_global_step, backward_count in enumerate((2, 3), start=1):
        starting_global_step = engine.global_steps
        for _ in range(backward_count):
            loss = engine(torch.ones(1, 1, device=engine.device)).sum()
            engine.backward(loss, scale_wrt_gas=False)
        assert engine.global_steps == starting_global_step

        engine.step()
        expected_weight -= 0.1 * backward_count
        assert engine.global_steps == expected_global_step
        torch_assert_close(
            engine.module.weight.detach().cpu(),
            torch.tensor([[expected_weight]]),
            rtol=0,
            atol=1e-6,
        )
