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

"""Exact AdamW trajectory parity and bounded optimizer-storage regressions."""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import pytest
import torch

from arctic_platform.correctness.reference.adamw_trajectory import AdamWTrajectory


@pytest.fixture(scope="module")
def trajectory_class():
    return AdamWTrajectory


class SmallModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-0.7, 0.9, 21, dtype=dtype).reshape(3, 7))
        self.bias = torch.nn.Parameter(torch.linspace(-0.2, 0.4, 7, dtype=dtype))
        self.norm = torch.nn.LayerNorm(7, dtype=dtype)
        self.scalar = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))
        self.dormant = torch.nn.Parameter(torch.ones(5, dtype=dtype))
        self.frozen = torch.nn.Parameter(torch.ones(4, dtype=dtype), requires_grad=False)
        self.to(os.environ.get("ADAMW_TEST_DEVICE", "cpu"))


OWNED_NAMES = ("weight", "bias", "scalar", "norm.weight", "norm.bias")
DECAY_NAMES = {"weight", "scalar"}
CONFIG = {"name": "AdamW", "weight_decay": 0.17, "betas": [0.73, 0.91], "eps": 0.003}
LEARNING_RATE = 0.007


def gradients_for(model, step):
    gradients = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if name == "dormant":
            continue
        values = torch.arange(parameter.numel(), dtype=torch.float32, device=parameter.device).reshape(parameter.shape)
        gradient = torch.sin(values + step * 0.7 + index) * (step + 1)
        if step == 2:
            gradient.zero_()
        if gradient.ndim == 2:
            gradient = gradient.T.contiguous().T
        gradients[name] = gradient.requires_grad_()
    return gradients


class OrdinaryAdamW:
    """Independent whole-parameter oracle with explicitly declared decay membership."""

    def __init__(self, model, master_type, clipping):
        sources = dict(model.named_parameters())
        self.masters = {
            name: torch.nn.Parameter(sources[name].detach().to(master_type).clone()) for name in OWNED_NAMES
        }
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [self.masters[name] for name in OWNED_NAMES if name in DECAY_NAMES],
                    "weight_decay": CONFIG["weight_decay"],
                },
                {
                    "params": [self.masters[name] for name in OWNED_NAMES if name not in DECAY_NAMES],
                    "weight_decay": 0.0,
                },
            ],
            lr=LEARNING_RATE,
            betas=tuple(CONFIG["betas"]),
            eps=CONFIG["eps"],
            foreach=False,
            fused=False,
        )
        self.model = model
        self.clipping = clipping

    def step(self, gradients):
        for name, master in self.masters.items():
            master.grad = gradients[name].detach().to(master.dtype).clone()
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(master.grad) for master in self.masters.values()])
        )
        if self.clipping is not None and self.clipping > 0:
            torch.nn.utils.clip_grad_norm_(list(self.masters.values()), self.clipping)
        self.optimizer.step()
        with torch.no_grad():
            for name, source in self.model.named_parameters():
                if name in self.masters:
                    source.copy_(self.masters[name].to(source.dtype))
        for master in self.masters.values():
            master.grad = None
        return float(norm)


def make_trajectory(trajectory_class, model, *, chunk=4, clipping=None, master="float32"):
    return trajectory_class(
        model,
        copy.deepcopy(CONFIG),
        learning_rate=LEARNING_RATE,
        gradient_clipping=clipping,
        optimizer_dtype=master,
        optimizer_chunk_elements=chunk,
    )


def assert_same_tensor(actual, expected, *, context):
    assert torch.equal(actual, expected), (
        f"{context}: max absolute difference "
        f"{float((actual.to(torch.float32) - expected.to(torch.float32)).abs().max())}"
    )


def assert_oracle_state(trajectory, oracle, step):
    assert trajectory.steps == step
    assert tuple(trajectory.trainable_names) == OWNED_NAMES
    assert set(trajectory._masters) == set(OWNED_NAMES)
    assert len(trajectory._optimizer.state) == len(OWNED_NAMES)
    assert len(trajectory._optimizer.param_groups) == 2
    assert trajectory._optimizer.defaults["foreach"] is False
    assert trajectory._optimizer.defaults["fused"] is False
    diagnostics = trajectory.state_norms()
    assert set(diagnostics) == set(OWNED_NAMES)
    for name in OWNED_NAMES:
        master = trajectory._masters[name]
        expected_master = oracle.masters[name]
        assert_same_tensor(master, expected_master, context=f"step {step} {name} master")
        actual = trajectory._optimizer.state[master]
        expected = oracle.optimizer.state[expected_master]
        assert set(actual) == set(expected) == {"step", "exp_avg", "exp_avg_sq"}
        for key in expected:
            assert_same_tensor(actual[key], expected[key], context=f"step {step} {name} {key}")
        assert actual["step"].item() == step
        assert actual["exp_avg"].shape == master.shape
        assert actual["exp_avg_sq"].shape == master.shape
        assert master.grad is None
        # Diagnostic reductions follow storage layout; elementwise parity above is exact.
        diagnostic_moments = {
            key: expected[key].contiguous() if actual[key].is_contiguous() else expected[key]
            for key in ("exp_avg", "exp_avg_sq")
        }
        assert diagnostics[name] == {
            "exp_avg": float(torch.linalg.vector_norm(diagnostic_moments["exp_avg"])),
            "exp_avg_sq": float(torch.linalg.vector_norm(diagnostic_moments["exp_avg_sq"])),
            "step": float(step),
        }
    for name, actual in trajectory._model.named_parameters():
        assert_same_tensor(
            actual,
            dict(oracle.model.named_parameters())[name],
            context=f"step {step} {name} source",
        )
    for group in trajectory._optimizer.param_groups:
        names = {name for name, master in trajectory._masters.items() if any(master is p for p in group["params"])}
        assert names == (DECAY_NAMES if group["weight_decay"] else set(OWNED_NAMES) - DECAY_NAMES)
        assert group["betas"] == tuple(CONFIG["betas"])
        assert group["eps"] == CONFIG["eps"]
        assert group["lr"] == LEARNING_RATE


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("master", ["float32", "bfloat16"])
@pytest.mark.parametrize("clipping", [None, 0.35, 100.0])
@pytest.mark.parametrize("chunk", [1, 4, 64, None])
def test_multistep_exact_master_moments_source_and_global_clip_parity(
    trajectory_class, source_dtype, master, clipping, chunk
):
    model = SmallModel(source_dtype)
    oracle = OrdinaryAdamW(copy.deepcopy(model), getattr(torch, master), clipping)
    trajectory = make_trajectory(trajectory_class, model, chunk=chunk, clipping=clipping, master=master)
    assert trajectory.trainable_names == []
    assert trajectory.state_norms() == {}
    persistent_storage = None
    for step in range(1, 7):
        gradients = gradients_for(model, step)
        inputs_before = {name: tensor.detach().clone() for name, tensor in gradients.items()}
        assert trajectory.step(gradients) == oracle.step(gradients)
        assert_oracle_state(trajectory, oracle, step)
        storage = {
            name: (
                master_tensor.data_ptr(),
                trajectory._optimizer.state[master_tensor]["exp_avg"].data_ptr(),
                trajectory._optimizer.state[master_tensor]["exp_avg_sq"].data_ptr(),
            )
            for name, master_tensor in trajectory._masters.items()
        }
        if persistent_storage is None:
            persistent_storage = storage
        assert storage == persistent_storage
        for name, tensor in gradients.items():
            assert_same_tensor(tensor, inputs_before[name], context=f"step {step} {name} input")
            assert tensor.grad is None
            assert tensor.requires_grad


@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_functional_calls_are_bounded_views_with_one_counter_per_original_parameter(
    trajectory_class, monkeypatch, chunk
):
    import torch.optim._functional as functional

    model = SmallModel(torch.bfloat16)
    trajectory = make_trajectory(trajectory_class, model, chunk=chunk, clipping=0.35)
    original = functional.adamw
    calls = []

    def capture(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps, **kwargs):
        assert len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs) == len(steps) == 1
        assert max_exp_avg_sqs == []
        assert kwargs["foreach"] is False and kwargs["fused"] is False
        assert kwargs["amsgrad"] is False
        parameter, gradient, first, second, counter = (
            params[0],
            grads[0],
            exp_avgs[0],
            exp_avg_sqs[0],
            steps[0],
        )
        assert 0 < parameter.numel() == gradient.numel() == first.numel() == second.numel() <= chunk
        owners = [
            (name, master)
            for name, master in trajectory._masters.items()
            if master.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
        ]
        assert len(owners) == 1
        name, master = owners[0]
        state = trajectory._optimizer.state[master]
        assert first.untyped_storage().data_ptr() == state["exp_avg"].untyped_storage().data_ptr()
        assert second.untyped_storage().data_ptr() == state["exp_avg_sq"].untyped_storage().data_ptr()
        assert parameter.storage_offset() == first.storage_offset() == second.storage_offset()
        assert counter.data_ptr() != state["step"].data_ptr()
        assert counter.item() == trajectory.steps
        assert all(owned.grad is None for owned in trajectory._masters.values())
        calls.append((name, parameter.storage_offset(), parameter.numel()))
        return original(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps, **kwargs)

    monkeypatch.setattr(functional, "adamw", capture)
    for step in range(1, 4):
        calls.clear()
        trajectory.step(gradients_for(model, step))
        assert len(calls) == sum(math.ceil(master.numel() / chunk) for master in trajectory._masters.values())
        for name, master in trajectory._masters.items():
            actual = [(offset, size) for owner, offset, size in calls if owner == name]
            assert actual == [
                (offset, min(chunk, master.numel() - offset)) for offset in range(0, master.numel(), chunk)
            ]
            assert trajectory._optimizer.state[master]["step"].item() == step


@pytest.mark.parametrize("chunk", [4, None])
def test_missing_owned_gradient_fails_before_mutation_and_can_resume(trajectory_class, chunk):
    model = SmallModel(torch.bfloat16)
    trajectory = make_trajectory(trajectory_class, model, chunk=chunk, clipping=0.35)
    oracle = OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    gradients = gradients_for(model, 1)
    trajectory.step(gradients)
    oracle.step(gradients)
    sources_before = copy.deepcopy(model.state_dict())
    states_before = {
        name: {key: tensor.clone() for key, tensor in trajectory._optimizer.state[master].items()}
        for name, master in trajectory._masters.items()
    }
    masters_before = {name: master.detach().clone() for name, master in trajectory._masters.items()}
    incomplete = gradients_for(model, 2)
    del incomplete["norm.weight"]
    with pytest.raises(
        ValueError,
        match="step 2 received no gradient.*norm.weight.*trainable set changed",
    ):
        trajectory.step(incomplete)
    assert trajectory.steps == 1
    for name, value in model.state_dict().items():
        assert_same_tensor(value, sources_before[name], context=f"failed step {name} source")
    for name, master in trajectory._masters.items():
        assert_same_tensor(master, masters_before[name], context=f"failed step {name} master")
        for key, tensor in trajectory._optimizer.state[master].items():
            assert_same_tensor(tensor, states_before[name][key], context=f"failed step {name} {key}")
        assert master.grad is None
    gradients = gradients_for(model, 2)
    trajectory.step(gradients)
    oracle.step(gradients)
    assert_oracle_state(trajectory, oracle, 2)


@pytest.mark.parametrize("chunk", [4, None])
@pytest.mark.parametrize("name", ["weight", OWNED_NAMES[-1]])
@pytest.mark.parametrize("malformed", ["same-numel", "short", "long"])
def test_owned_gradient_shape_rejected_atomically_and_can_resume(
    trajectory_class, monkeypatch, chunk, name, malformed
):
    model = SmallModel(torch.bfloat16)
    trajectory = make_trajectory(trajectory_class, model, chunk=chunk, clipping=0.35)
    oracle = OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    gradients = gradients_for(model, 1)
    trajectory.step(gradients)
    oracle.step(gradients)
    sources_before = copy.deepcopy(model.state_dict())
    masters_before = {key: master.detach().clone() for key, master in trajectory._masters.items()}
    states_before = {
        key: {field: tensor.clone() for field, tensor in trajectory._optimizer.state[master].items()}
        for key, master in trajectory._masters.items()
    }
    storage_before = {
        key: (
            master.data_ptr(),
            tuple(tensor.data_ptr() for tensor in trajectory._optimizer.state[master].values()),
        )
        for key, master in trajectory._masters.items()
    }
    invalid = gradients_for(model, 2)
    expected_shape = tuple(invalid[name].shape)
    count = invalid[name].numel()
    shape = {
        "same-numel": (count,) if len(expected_shape) > 1 else (1, count),
        "short": (count - 1,),
        "long": (count + 1,),
    }[malformed]
    invalid[name] = (
        torch.arange(math.prod(shape), device=model.weight.device, dtype=torch.float32).reshape(shape).requires_grad_()
    )
    inputs_before = {key: tensor.detach().clone() for key, tensor in invalid.items()}
    with monkeypatch.context() as guard:
        guard.setattr(
            trajectory,
            "_bounded_step",
            lambda gradients: pytest.fail("invalid gradient reached chunked update"),
        )
        guard.setattr(
            trajectory._optimizer,
            "step",
            lambda: pytest.fail("invalid gradient reached native optimizer update"),
        )
        with pytest.raises(ValueError) as error:
            trajectory.step(invalid)
    assert str(error.value) == f"step 2 gradient {name!r} has shape {shape}, expected {expected_shape}"
    assert trajectory.steps == 1
    assert tuple(trajectory.trainable_names) == OWNED_NAMES
    assert len(trajectory._optimizer.state) == len(OWNED_NAMES)
    for key, source in model.state_dict().items():
        assert_same_tensor(source, sources_before[key], context=f"rejected shape {key} source")
    for key, master in trajectory._masters.items():
        assert_same_tensor(master, masters_before[key], context=f"rejected shape {key} master")
        state = trajectory._optimizer.state[master]
        assert set(state) == set(states_before[key])
        for field, tensor in state.items():
            assert_same_tensor(
                tensor,
                states_before[key][field],
                context=f"rejected shape {key} {field}",
            )
        assert (
            master.data_ptr(),
            tuple(tensor.data_ptr() for tensor in state.values()),
        ) == storage_before[key]
        assert master.grad is None
    for key, tensor in invalid.items():
        assert_same_tensor(tensor, inputs_before[key], context=f"rejected shape {key} input")
        assert tensor.grad is None and tensor.requires_grad
    gradients = gradients_for(model, 2)
    assert trajectory.step(gradients) == oracle.step(gradients)
    assert_oracle_state(trajectory, oracle, 2)


@pytest.mark.parametrize("chunk", [4, None])
def test_no_trainable_gradient_fails_without_adoption(trajectory_class, chunk):
    model = SmallModel(torch.float32)
    trajectory = make_trajectory(trajectory_class, model, chunk=chunk)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="no trainable parameters with gradients"):
        trajectory.step({"frozen": torch.zeros_like(model.frozen)})
    assert trajectory.steps == 0
    assert trajectory.trainable_names == []
    assert trajectory._masters == {}
    assert trajectory._optimizer is None
    for name, source in model.state_dict().items():
        assert_same_tensor(source, before[name], context=f"no adoption {name}")


@pytest.mark.parametrize("chunk", [0, -1, True, False, 1.5, "4"])
def test_invalid_chunk_size_fails_before_adoption(trajectory_class, chunk):
    with pytest.raises(ValueError, match="optimizer_chunk_elements must be a positive integer"):
        make_trajectory(trajectory_class, SmallModel(torch.float32), chunk=chunk)


@pytest.mark.parametrize("clipping", [0.0, -1.0])
def test_nonpositive_clipping_keeps_native_disabled_semantics(trajectory_class, clipping):
    model = SmallModel(torch.float32)
    trajectory = make_trajectory(trajectory_class, model, clipping=clipping)
    oracle = OrdinaryAdamW(copy.deepcopy(model), torch.float32, clipping)
    for step in range(1, 4):
        gradients = gradients_for(model, step)
        assert trajectory.step(gradients) == oracle.step(gradients)
        assert_oracle_state(trajectory, oracle, step)


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 4, None])
def test_noncontiguous_parameter_parity_and_default_layout_preservation(trajectory_class, source_dtype, chunk):
    model = SmallModel(source_dtype)
    model.weight = torch.nn.Parameter(model.weight.detach().T.contiguous().T)
    assert not model.weight.is_contiguous()
    original_stride = model.weight.stride()
    trajectory = make_trajectory(trajectory_class, model, chunk=chunk, clipping=0.35)
    oracle = OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    for step in range(1, 4):
        gradients = gradients_for(model, step)
        assert trajectory.step(gradients) == oracle.step(gradients)
        assert_oracle_state(trajectory, oracle, step)
        assert model.weight.stride() == original_stride
        if chunk is None:
            assert trajectory._masters["weight"].stride() == original_stride
        else:
            assert trajectory._masters["weight"].is_contiguous()


def test_omitted_chunk_option_remains_native_whole_parameter_path(trajectory_class, monkeypatch):
    model = SmallModel(torch.bfloat16)
    configuration = copy.deepcopy(CONFIG)
    trajectory = trajectory_class(
        model,
        configuration,
        learning_rate=LEARNING_RATE,
        gradient_clipping=0.35,
        optimizer_dtype="float32",
    )
    monkeypatch.setattr(
        trajectory,
        "_bounded_step",
        lambda gradients: pytest.fail("default unexpectedly used chunking"),
    )
    oracle = OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    for step in range(1, 4):
        gradients = gradients_for(model, step)
        assert trajectory.step(gradients) == oracle.step(gradients)
        assert_oracle_state(trajectory, oracle, step)
        assert configuration == CONFIG


@pytest.mark.parametrize("chunk", [None, 4, 16 * 1024 * 1024])
def test_sft_cli_forwards_explicit_chunk_size_and_preserves_default(tmp_path, monkeypatch, chunk):
    from arctic_platform.correctness.harness import batches
    from arctic_platform.correctness.reference import sft_trajectory

    replay = tmp_path / "replay.json"
    train = tmp_path / "train.pt"
    validation = tmp_path / "validation.pt"
    output = tmp_path / "result.json"
    replay.write_text(json.dumps({"train_batches": [str(train)], "validation_batch": str(validation)}))
    monkeypatch.setattr(batches, "load", lambda path: path)
    calls = []

    def capture_run(model, train_batches, validation_batch, **kwargs):
        calls.append((model, list(train_batches), validation_batch, kwargs))
        return sft_trajectory.TrajectoryResult([], [], [])

    monkeypatch.setattr(sft_trajectory, "run", capture_run)
    argv = [
        "sft_trajectory",
        "--model",
        str(tmp_path / "model"),
        "--replay",
        str(replay),
        "--out",
        str(output),
        "--seed",
        "3",
        "--optimizer-config",
        json.dumps(CONFIG),
        "--learning-rate",
        str(LEARNING_RATE),
    ]
    if chunk is not None:
        argv.extend(["--optimizer-chunk-elements", str(chunk)])
    monkeypatch.setattr(sys, "argv", argv)
    sft_trajectory.main()
    assert len(calls) == 1
    model, train_batches, validation_batch, kwargs = calls[0]
    assert model == str(tmp_path / "model")
    assert train_batches == [train]
    assert validation_batch == validation
    assert kwargs["optimizer_chunk_elements"] == chunk
    assert kwargs["optimizer_config"] == CONFIG
    assert kwargs["learning_rate"] == LEARNING_RATE
    assert kwargs["optimizer_dtype"] == "float32"
    assert json.loads(output.read_text())["optimizer_steps"] == 0
