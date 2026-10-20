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

"""Static correctness selftests: optimizer math."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import fwd_bwd_step
from arctic_platform.correctness.checks.optimizer_state import COMPARISON_STATE_KEYS
from arctic_platform.correctness.checks.optimizer_state import OPTIMIZER_NORM_CRITERION
from arctic_platform.correctness.checks.optimizer_state import STATE_KEYS
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_artifacts
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_norms
from arctic_platform.correctness.checks.optimizer_state import max_pairwise_optimizer_norm_delta
from arctic_platform.correctness.checks.optimizer_state import summarize_optimizer_artifacts
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.runner import OPTIMIZER_LEARNING_RATE
from arctic_platform.correctness.reference import optimizer_step
from arctic_platform.correctness.reference.adamw_trajectory import AdamWTrajectory
from arctic_platform.correctness.reference.optimizer_step import run_adamw_step

# Adamw chunked.


@pytest.fixture(scope="module")
def _adamw_chunked_trajectory_class():
    return AdamWTrajectory


class _adamw_chunked_SmallModel(torch.nn.Module):

    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-0.7, 0.9, 21, dtype=dtype).reshape(3, 7))
        self.bias = torch.nn.Parameter(torch.linspace(-0.2, 0.4, 7, dtype=dtype))
        self.norm = torch.nn.LayerNorm(7, dtype=dtype)
        self.scalar = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))
        self.dormant = torch.nn.Parameter(torch.ones(5, dtype=dtype))
        self.frozen = torch.nn.Parameter(torch.ones(4, dtype=dtype), requires_grad=False)
        self.to(os.environ.get("ADAMW_TEST_DEVICE", "cpu"))


_adamw_chunked_OWNED_NAMES = ("weight", "bias", "scalar", "norm.weight", "norm.bias")

_adamw_chunked_DECAY_NAMES = {"weight", "scalar"}

_adamw_chunked_CONFIG = {"name": "AdamW", "weight_decay": 0.17, "betas": [0.73, 0.91], "eps": 0.003}

_adamw_chunked_LEARNING_RATE = 0.007


def _adamw_chunked_gradients_for(model, step):
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


class _adamw_chunked_OrdinaryAdamW:
    """Independent whole-parameter oracle with explicitly declared decay membership."""

    def __init__(self, model, master_type, clipping):
        sources = dict(model.named_parameters())
        self.masters = {
            name: torch.nn.Parameter(sources[name].detach().to(master_type).clone())
            for name in _adamw_chunked_OWNED_NAMES
        }
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        self.masters[name] for name in _adamw_chunked_OWNED_NAMES if name in _adamw_chunked_DECAY_NAMES
                    ],
                    "weight_decay": _adamw_chunked_CONFIG["weight_decay"],
                },
                {
                    "params": [
                        self.masters[name]
                        for name in _adamw_chunked_OWNED_NAMES
                        if name not in _adamw_chunked_DECAY_NAMES
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=_adamw_chunked_LEARNING_RATE,
            betas=tuple(_adamw_chunked_CONFIG["betas"]),
            eps=_adamw_chunked_CONFIG["eps"],
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


def _adamw_chunked_make_trajectory(trajectory_class, model, *, chunk=4, clipping=None, master="float32"):
    return trajectory_class(
        model,
        copy.deepcopy(_adamw_chunked_CONFIG),
        learning_rate=_adamw_chunked_LEARNING_RATE,
        gradient_clipping=clipping,
        optimizer_dtype=master,
        optimizer_chunk_elements=chunk,
    )


def _adamw_chunked_assert_same_tensor(actual, expected, *, context):
    assert torch.equal(actual, expected), (
        f"{context}: max absolute difference"
        f" {float((actual.to(torch.float32) - expected.to(torch.float32)).abs().max())}"
    )


def _adamw_chunked_assert_oracle_state(trajectory, oracle, step):
    assert trajectory.steps == step
    assert tuple(trajectory.trainable_names) == _adamw_chunked_OWNED_NAMES
    assert set(trajectory._masters) == set(_adamw_chunked_OWNED_NAMES)
    assert len(trajectory._optimizer.state) == len(_adamw_chunked_OWNED_NAMES)
    assert len(trajectory._optimizer.param_groups) == 2
    assert trajectory._optimizer.defaults["foreach"] is False
    assert trajectory._optimizer.defaults["fused"] is False
    diagnostics = trajectory.state_norms()
    assert set(diagnostics) == set(_adamw_chunked_OWNED_NAMES)
    for name in _adamw_chunked_OWNED_NAMES:
        master = trajectory._masters[name]
        expected_master = oracle.masters[name]
        _adamw_chunked_assert_same_tensor(master, expected_master, context=f"step {step} {name} master")
        actual = trajectory._optimizer.state[master]
        expected = oracle.optimizer.state[expected_master]
        assert set(actual) == set(expected) == {"step", "exp_avg", "exp_avg_sq"}
        for key in expected:
            _adamw_chunked_assert_same_tensor(actual[key], expected[key], context=f"step {step} {name} {key}")
        assert actual["step"].item() == step
        assert actual["exp_avg"].shape == master.shape
        assert actual["exp_avg_sq"].shape == master.shape
        assert master.grad is None
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
        _adamw_chunked_assert_same_tensor(
            actual, dict(oracle.model.named_parameters())[name], context=f"step {step} {name} source"
        )
    for group in trajectory._optimizer.param_groups:
        names = {name for name, master in trajectory._masters.items() if any((master is p for p in group["params"]))}
        assert names == (
            _adamw_chunked_DECAY_NAMES
            if group["weight_decay"]
            else set(_adamw_chunked_OWNED_NAMES) - _adamw_chunked_DECAY_NAMES
        )
        assert group["betas"] == tuple(_adamw_chunked_CONFIG["betas"])
        assert group["eps"] == _adamw_chunked_CONFIG["eps"]
        assert group["lr"] == _adamw_chunked_LEARNING_RATE


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("master", ["float32", "bfloat16"])
@pytest.mark.parametrize("clipping", [None, 0.35, 100.0])
@pytest.mark.parametrize("chunk", [1, 4, 64, None])
def test_adamw_chunked_multistep_exact_master_moments_source_and_global_clip_parity(
    _adamw_chunked_trajectory_class, source_dtype, master, clipping, chunk
):
    model = _adamw_chunked_SmallModel(source_dtype)
    oracle = _adamw_chunked_OrdinaryAdamW(copy.deepcopy(model), getattr(torch, master), clipping)
    trajectory = _adamw_chunked_make_trajectory(
        _adamw_chunked_trajectory_class, model, chunk=chunk, clipping=clipping, master=master
    )
    assert trajectory.trainable_names == []
    assert trajectory.state_norms() == {}
    persistent_storage = None
    for step in range(1, 7):
        gradients = _adamw_chunked_gradients_for(model, step)
        inputs_before = {name: tensor.detach().clone() for name, tensor in gradients.items()}
        assert trajectory.step(gradients) == oracle.step(gradients)
        _adamw_chunked_assert_oracle_state(trajectory, oracle, step)
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
            _adamw_chunked_assert_same_tensor(tensor, inputs_before[name], context=f"step {step} {name} input")
            assert tensor.grad is None
            assert tensor.requires_grad


@pytest.mark.parametrize("chunk", [1, 4, 64])
def test_adamw_chunked_functional_calls_are_bounded_views_with_one_counter_per_original_parameter(
    _adamw_chunked_trajectory_class, monkeypatch, chunk
):
    import torch.optim._functional as functional

    model = _adamw_chunked_SmallModel(torch.bfloat16)
    trajectory = _adamw_chunked_make_trajectory(_adamw_chunked_trajectory_class, model, chunk=chunk, clipping=0.35)
    original = functional.adamw
    calls = []

    def capture(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps, **kwargs):
        assert len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs) == len(steps) == 1
        assert max_exp_avg_sqs == []
        assert kwargs["foreach"] is False and kwargs["fused"] is False
        assert kwargs["amsgrad"] is False
        parameter, gradient, first, second, counter = (params[0], grads[0], exp_avgs[0], exp_avg_sqs[0], steps[0])
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
        assert all((owned.grad is None for owned in trajectory._masters.values()))
        calls.append((name, parameter.storage_offset(), parameter.numel()))
        return original(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps, **kwargs)

    monkeypatch.setattr(functional, "adamw", capture)
    for step in range(1, 4):
        calls.clear()
        trajectory.step(_adamw_chunked_gradients_for(model, step))
        assert len(calls) == sum((math.ceil(master.numel() / chunk) for master in trajectory._masters.values()))
        for name, master in trajectory._masters.items():
            actual = [(offset, size) for owner, offset, size in calls if owner == name]
            assert actual == [
                (offset, min(chunk, master.numel() - offset)) for offset in range(0, master.numel(), chunk)
            ]
            assert trajectory._optimizer.state[master]["step"].item() == step


@pytest.mark.parametrize("chunk", [4, None])
def test_adamw_chunked_missing_owned_gradient_fails_before_mutation_and_can_resume(
    _adamw_chunked_trajectory_class, chunk
):
    model = _adamw_chunked_SmallModel(torch.bfloat16)
    trajectory = _adamw_chunked_make_trajectory(_adamw_chunked_trajectory_class, model, chunk=chunk, clipping=0.35)
    oracle = _adamw_chunked_OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    gradients = _adamw_chunked_gradients_for(model, 1)
    trajectory.step(gradients)
    oracle.step(gradients)
    sources_before = copy.deepcopy(model.state_dict())
    states_before = {
        name: {key: tensor.clone() for key, tensor in trajectory._optimizer.state[master].items()}
        for name, master in trajectory._masters.items()
    }
    masters_before = {name: master.detach().clone() for name, master in trajectory._masters.items()}
    incomplete = _adamw_chunked_gradients_for(model, 2)
    del incomplete["norm.weight"]
    with pytest.raises(ValueError, match="step 2 received no gradient.*norm.weight.*trainable set changed"):
        trajectory.step(incomplete)
    assert trajectory.steps == 1
    for name, value in model.state_dict().items():
        _adamw_chunked_assert_same_tensor(value, sources_before[name], context=f"failed step {name} source")
    for name, master in trajectory._masters.items():
        _adamw_chunked_assert_same_tensor(master, masters_before[name], context=f"failed step {name} master")
        for key, tensor in trajectory._optimizer.state[master].items():
            _adamw_chunked_assert_same_tensor(tensor, states_before[name][key], context=f"failed step {name} {key}")
        assert master.grad is None
    gradients = _adamw_chunked_gradients_for(model, 2)
    trajectory.step(gradients)
    oracle.step(gradients)
    _adamw_chunked_assert_oracle_state(trajectory, oracle, 2)


@pytest.mark.parametrize("chunk", [4, None])
@pytest.mark.parametrize("name", ["weight", _adamw_chunked_OWNED_NAMES[-1]])
@pytest.mark.parametrize("malformed", ["same-numel", "short", "long"])
def test_adamw_chunked_owned_gradient_shape_rejected_atomically_and_can_resume(
    _adamw_chunked_trajectory_class, monkeypatch, chunk, name, malformed
):
    model = _adamw_chunked_SmallModel(torch.bfloat16)
    trajectory = _adamw_chunked_make_trajectory(_adamw_chunked_trajectory_class, model, chunk=chunk, clipping=0.35)
    oracle = _adamw_chunked_OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    gradients = _adamw_chunked_gradients_for(model, 1)
    trajectory.step(gradients)
    oracle.step(gradients)
    sources_before = copy.deepcopy(model.state_dict())
    masters_before = {key: master.detach().clone() for key, master in trajectory._masters.items()}
    states_before = {
        key: {field: tensor.clone() for field, tensor in trajectory._optimizer.state[master].items()}
        for key, master in trajectory._masters.items()
    }
    storage_before = {
        key: (master.data_ptr(), tuple((tensor.data_ptr() for tensor in trajectory._optimizer.state[master].values())))
        for key, master in trajectory._masters.items()
    }
    invalid = _adamw_chunked_gradients_for(model, 2)
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
            trajectory, "_bounded_step", lambda gradients: pytest.fail("invalid gradient reached chunked update")
        )
        guard.setattr(
            trajectory._optimizer, "step", lambda: pytest.fail("invalid gradient reached native optimizer update")
        )
        with pytest.raises(ValueError) as error:
            trajectory.step(invalid)
    assert str(error.value) == f"step 2 gradient {name!r} has shape {shape}, expected {expected_shape}"
    assert trajectory.steps == 1
    assert tuple(trajectory.trainable_names) == _adamw_chunked_OWNED_NAMES
    assert len(trajectory._optimizer.state) == len(_adamw_chunked_OWNED_NAMES)
    for key, source in model.state_dict().items():
        _adamw_chunked_assert_same_tensor(source, sources_before[key], context=f"rejected shape {key} source")
    for key, master in trajectory._masters.items():
        _adamw_chunked_assert_same_tensor(master, masters_before[key], context=f"rejected shape {key} master")
        state = trajectory._optimizer.state[master]
        assert set(state) == set(states_before[key])
        for field, tensor in state.items():
            _adamw_chunked_assert_same_tensor(
                tensor, states_before[key][field], context=f"rejected shape {key} {field}"
            )
        assert (master.data_ptr(), tuple((tensor.data_ptr() for tensor in state.values()))) == storage_before[key]
        assert master.grad is None
    for key, tensor in invalid.items():
        _adamw_chunked_assert_same_tensor(tensor, inputs_before[key], context=f"rejected shape {key} input")
        assert tensor.grad is None and tensor.requires_grad
    gradients = _adamw_chunked_gradients_for(model, 2)
    assert trajectory.step(gradients) == oracle.step(gradients)
    _adamw_chunked_assert_oracle_state(trajectory, oracle, 2)


@pytest.mark.parametrize("chunk", [4, None])
def test_adamw_chunked_no_trainable_gradient_fails_without_adoption(_adamw_chunked_trajectory_class, chunk):
    model = _adamw_chunked_SmallModel(torch.float32)
    trajectory = _adamw_chunked_make_trajectory(_adamw_chunked_trajectory_class, model, chunk=chunk)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="no trainable parameters with gradients"):
        trajectory.step({"frozen": torch.zeros_like(model.frozen)})
    assert trajectory.steps == 0
    assert trajectory.trainable_names == []
    assert trajectory._masters == {}
    assert trajectory._optimizer is None
    for name, source in model.state_dict().items():
        _adamw_chunked_assert_same_tensor(source, before[name], context=f"no adoption {name}")


@pytest.mark.parametrize("chunk", [0, -1, True, False, 1.5, "4"])
def test_adamw_chunked_invalid_chunk_size_fails_before_adoption(_adamw_chunked_trajectory_class, chunk):
    with pytest.raises(ValueError, match="optimizer_chunk_elements must be a positive integer"):
        _adamw_chunked_make_trajectory(
            _adamw_chunked_trajectory_class, _adamw_chunked_SmallModel(torch.float32), chunk=chunk
        )


@pytest.mark.parametrize("clipping", [0.0, -1.0])
def test_adamw_chunked_nonpositive_clipping_keeps_native_disabled_semantics(_adamw_chunked_trajectory_class, clipping):
    model = _adamw_chunked_SmallModel(torch.float32)
    trajectory = _adamw_chunked_make_trajectory(_adamw_chunked_trajectory_class, model, clipping=clipping)
    oracle = _adamw_chunked_OrdinaryAdamW(copy.deepcopy(model), torch.float32, clipping)
    for step in range(1, 4):
        gradients = _adamw_chunked_gradients_for(model, step)
        assert trajectory.step(gradients) == oracle.step(gradients)
        _adamw_chunked_assert_oracle_state(trajectory, oracle, step)


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 4, None])
def test_adamw_chunked_noncontiguous_parameter_parity_and_default_layout_preservation(
    _adamw_chunked_trajectory_class, source_dtype, chunk
):
    model = _adamw_chunked_SmallModel(source_dtype)
    model.weight = torch.nn.Parameter(model.weight.detach().T.contiguous().T)
    assert not model.weight.is_contiguous()
    original_stride = model.weight.stride()
    trajectory = _adamw_chunked_make_trajectory(_adamw_chunked_trajectory_class, model, chunk=chunk, clipping=0.35)
    oracle = _adamw_chunked_OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    for step in range(1, 4):
        gradients = _adamw_chunked_gradients_for(model, step)
        assert trajectory.step(gradients) == oracle.step(gradients)
        _adamw_chunked_assert_oracle_state(trajectory, oracle, step)
        assert model.weight.stride() == original_stride
        if chunk is None:
            assert trajectory._masters["weight"].stride() == original_stride
        else:
            assert trajectory._masters["weight"].is_contiguous()


def test_adamw_chunked_omitted_chunk_option_remains_native_whole_parameter_path(
    _adamw_chunked_trajectory_class, monkeypatch
):
    model = _adamw_chunked_SmallModel(torch.bfloat16)
    configuration = copy.deepcopy(_adamw_chunked_CONFIG)
    trajectory = _adamw_chunked_trajectory_class(
        model,
        configuration,
        learning_rate=_adamw_chunked_LEARNING_RATE,
        gradient_clipping=0.35,
        optimizer_dtype="float32",
    )
    monkeypatch.setattr(
        trajectory, "_bounded_step", lambda gradients: pytest.fail("default unexpectedly used chunking")
    )
    oracle = _adamw_chunked_OrdinaryAdamW(copy.deepcopy(model), torch.float32, 0.35)
    for step in range(1, 4):
        gradients = _adamw_chunked_gradients_for(model, step)
        assert trajectory.step(gradients) == oracle.step(gradients)
        _adamw_chunked_assert_oracle_state(trajectory, oracle, step)
        assert configuration == _adamw_chunked_CONFIG


@pytest.mark.parametrize("chunk", [None, 4, 16 * 1024 * 1024])
def test_adamw_chunked_sft_cli_forwards_explicit_chunk_size_and_preserves_default(tmp_path, monkeypatch, chunk):
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
        json.dumps(_adamw_chunked_CONFIG),
        "--learning-rate",
        str(_adamw_chunked_LEARNING_RATE),
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
    assert kwargs["optimizer_config"] == _adamw_chunked_CONFIG
    assert kwargs["learning_rate"] == _adamw_chunked_LEARNING_RATE
    assert kwargs["optimizer_dtype"] == "float32"
    assert json.loads(output.read_text())["optimizer_steps"] == 0


# Optimizer norms.


_optimizer_norms_KWARGS = {"optimizer_config": {}, "learning_rate": OPTIMIZER_LEARNING_RATE}


@pytest.fixture
def _optimizer_norms_artifact_root(tmp_path):
    assert tmp_path.resolve().is_relative_to(
        Path("/data-fast")
    ), "run optimizer controls with --basetemp=/data-fast/..."
    return tmp_path


def _optimizer_norms_manifest(root, parameters, *, step=1):
    root.mkdir()
    records = []
    for index, (name, states) in enumerate(parameters.items()):
        filename = f"{index:06d}.pt"
        torch.save(states, root / filename)
        records.append({"name": name, "file": filename})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"step": step, "parameters": records}))
    return manifest


def _optimizer_norms_states(value):
    value = torch.as_tensor(value)
    return {
        "parameter_update": value.clone(),
        "exp_avg": torch.zeros_like(value),
        "exp_avg_sq": torch.zeros_like(value),
    }


def _optimizer_norms_summary(parameters=None):
    if parameters is None:
        parameters = {
            "layer.weight": {"shape": [2], "parameter_update_residual": 0.0, "exp_avg": 0.0, "exp_avg_sq": 0.0}
        }
    return {
        "criterion": OPTIMIZER_NORM_CRITERION,
        "step": 1,
        "state_keys": list(COMPARISON_STATE_KEYS),
        "parameter_count": len(parameters),
        "optimizer_values_count": 3 * len(parameters),
        "parameters": parameters,
    }


def _optimizer_norms_adam_states(gradient, backend):
    beta1, beta2, epsilon, lr = (0.9, 0.999, 1e-08, OPTIMIZER_LEARNING_RATE)
    if backend == "dss":
        beta1, beta2, epsilon, lr = [
            float(torch.tensor(value, dtype=torch.float32)) for value in (beta1, beta2, epsilon, lr)
        ]
        correction1, correction2 = [float(torch.tensor(1.0 - value, dtype=torch.float32)) for value in (beta1, beta2)]
    else:
        correction1, correction2 = (1.0 - beta1, 1.0 - beta2)
    first = (1.0 - beta1) * gradient
    second = (1.0 - beta2) * gradient.square()
    denominator = (second / correction2).sqrt() + epsilon
    update = (
        -(first / correction1 / denominator) * lr if backend == "dss" else -lr * (first / correction1) / denominator
    )
    return {"parameter_update": update, "exp_avg": first, "exp_avg_sq": second}


def test_optimizer_norms_summary_is_compact_serializable_float32_l2(_optimizer_norms_artifact_root):
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "ref", {"model.layer.weight": _optimizer_norms_states(value)}
    )
    result = summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result == _optimizer_norms_summary(
        {
            "layer.weight": {
                "shape": [2, 2],
                "parameter_update_residual": float(torch.linalg.vector_norm(value.float())),
                "exp_avg": 0.0,
                "exp_avg_sq": 0.0,
            }
        }
    )


@pytest.mark.parametrize("backend", ["reference", "dss"])
def test_optimizer_norms_norms_keep_backend_specific_moment_update_residual(_optimizer_norms_artifact_root, backend):
    states = _optimizer_norms_adam_states(torch.tensor([0.2, -0.4, 2.0]), backend)
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / backend, {"layer.weight": states})
    result = summarize_optimizer_artifacts(manifest, backend=backend, **_optimizer_norms_KWARGS)["parameters"][
        "layer.weight"
    ]
    assert result["parameter_update_residual"] < 1e-08
    assert result["exp_avg"] == float(torch.linalg.vector_norm(states["exp_avg"]))
    assert result["exp_avg_sq"] == float(torch.linalg.vector_norm(states["exp_avg_sq"]))


@pytest.mark.parametrize("backend", ["reference", "dss"])
def test_optimizer_norms_nonzero_initial_parameters_do_not_enter_update_norm(_optimizer_norms_artifact_root, backend):
    initial = torch.tensor([2.0, -3.0])
    states = _optimizer_norms_adam_states(torch.tensor([1.0, -1.0]), backend)
    after = initial + states["parameter_update"]
    states["parameter_update"] = after - initial
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / backend, {"layer.weight": states})
    result = summarize_optimizer_artifacts(manifest, backend=backend, **_optimizer_norms_KWARGS)["parameters"][
        "layer.weight"
    ]
    assert result["parameter_update_residual"] < 5e-07
    assert result["parameter_update_residual"] < float(torch.linalg.vector_norm(initial))


def test_optimizer_norms_unexplained_update_is_not_removed_by_norm_reduction(_optimizer_norms_artifact_root):
    states = _optimizer_norms_adam_states(torch.tensor([1.0, -1.0]), "dss")
    states["parameter_update"] += torch.tensor([0.003, 0.004])
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "corrupt", {"module.layer.weight": states})
    result = summarize_optimizer_artifacts(manifest, backend="dss", **_optimizer_norms_KWARGS)
    assert result["parameters"]["layer.weight"]["parameter_update_residual"] == pytest.approx(0.005)


def test_optimizer_norms_moe_tensors_are_concatenated_before_residual_and_norm(_optimizer_norms_artifact_root):
    w1 = _optimizer_norms_adam_states(torch.tensor([[[0.2, 0.4]]]), "dss")
    w3 = _optimizer_norms_adam_states(torch.tensor([[[0.6, 0.8]]]), "dss")
    dss = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "moe",
        {"module.layers.0.mlp.experts.w1": w1, "module.layers.0.mlp.experts.w3": w3},
    )
    joined = {state: torch.cat([w1[state], w3[state]], dim=1) for state in STATE_KEYS}
    canonical = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "canonical", {"layers.0.mlp.experts.gate_up_proj": joined}
    )
    actual = summarize_optimizer_artifacts(dss, backend="dss", **_optimizer_norms_KWARGS)
    expected = summarize_optimizer_artifacts(canonical, backend="dss", **_optimizer_norms_KWARGS)
    assert actual == expected
    entry = actual["parameters"]["layers.0.mlp.experts.gate_up_proj"]
    assert entry["shape"] == [1, 2, 2]
    assert entry["exp_avg"] == float(torch.linalg.vector_norm(joined["exp_avg"]))
    assert entry["parameter_update_residual"] < 1e-08


def test_optimizer_norms_canonical_collision_cannot_hide_a_parameter(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "collision",
        {
            "layers.0.mlp.experts.w1": _optimizer_norms_states(torch.zeros(1, 1, 1)),
            "layers.0.mlp.experts.w3": _optimizer_norms_states(torch.zeros(1, 1, 1)),
            "layers.0.mlp.experts.gate_up_proj": _optimizer_norms_states(torch.zeros(1, 2, 1)),
        },
    )
    with pytest.raises(ValueError, match="canonical parameter name collision"):
        summarize_optimizer_artifacts(manifest, backend="dss", **_optimizer_norms_KWARGS)


def test_optimizer_norms_normalization_collision_is_rejected(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "aliases",
        {
            "model.layer.weight": _optimizer_norms_states(torch.zeros(2)),
            "module.layer.weight": _optimizer_norms_states(torch.zeros(2)),
        },
    )
    with pytest.raises(ValueError, match="duplicate normalized parameter"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("step", [0, 2, 1.0, True, "1"])
def test_optimizer_norms_summary_requires_exact_step_one(_optimizer_norms_artifact_root, step):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "step", {"layer.weight": _optimizer_norms_states(torch.zeros(2))}, step=step
    )
    with pytest.raises(ValueError, match="step 1"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_rejects_empty_capture(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "empty", {})
    with pytest.raises(ValueError, match="no parameters"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("state", STATE_KEYS)
def test_optimizer_norms_summary_requires_every_artifact_state(_optimizer_norms_artifact_root, state):
    states = _optimizer_norms_states(torch.zeros(2))
    states.pop(state)
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "missing", {"layer.weight": states})
    with pytest.raises(ValueError, match=state):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_requires_identical_state_shapes(_optimizer_norms_artifact_root):
    states = _optimizer_norms_states(torch.zeros(2))
    states["exp_avg_sq"] = torch.zeros(1)
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "shape", {"layer.weight": states})
    with pytest.raises(ValueError, match="shape mismatch"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_optimizer_norms_summary_rejects_nonfinite_or_float32_overflow(_optimizer_norms_artifact_root, value):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "invalid",
        {"layer.weight": _optimizer_norms_states(torch.tensor([value], dtype=torch.float64))},
    )
    with pytest.raises(ValueError, match="not finite"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_rejects_norm_reduction_overflow(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "overflow",
        {"layer.weight": _optimizer_norms_states(torch.tensor([3e38, 3e38]))},
    )
    with pytest.raises(ValueError, match="optimizer norm .* not finite"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_summary_rejects_invalid_second_moment(_optimizer_norms_artifact_root):
    states = _optimizer_norms_states(torch.zeros(2))
    states["exp_avg_sq"] = torch.tensor([-1.0, 0.0])
    manifest = _optimizer_norms_manifest(_optimizer_norms_artifact_root / "negative", {"layer.weight": states})
    with pytest.raises(ValueError, match="comparison tensor .* not finite"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_optimizer_norms_summary_rejects_nonfloating_artifacts(_optimizer_norms_artifact_root, dtype):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dtype",
        {"layer.weight": _optimizer_norms_states(torch.zeros(2, dtype=dtype))},
    )
    with pytest.raises(ValueError, match="floating tensor"):
        summarize_optimizer_artifacts(manifest, **_optimizer_norms_KWARGS)


def test_optimizer_norms_compare_uses_scalar_norm_difference_not_tensor_difference(_optimizer_norms_artifact_root):
    ref = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "ref", {"layer.weight": _optimizer_norms_states(torch.tensor([1.0, 2.0]))}
    )
    dss = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dss", {"layer.weight": _optimizer_norms_states(torch.tensor([4.0, 6.0]))}
    )
    comparison = compare_optimizer_norms(
        summarize_optimizer_artifacts(dss, backend="dss", **_optimizer_norms_KWARGS),
        summarize_optimizer_artifacts(ref, **_optimizer_norms_KWARGS),
    )
    assert comparison.worst_delta == pytest.approx(math.sqrt(52) - math.sqrt(5))
    assert compare_optimizer_artifacts(dss, ref, **_optimizer_norms_KWARGS).worst_delta == 5.0


def test_optimizer_norms_equal_norm_different_direction_is_intentionally_undetected(_optimizer_norms_artifact_root):
    ref = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "ref", {"layer.weight": _optimizer_norms_states(torch.tensor([3.0, 4.0]))}
    )
    dss = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dss", {"layer.weight": _optimizer_norms_states(torch.tensor([-3.0, -4.0]))}
    )
    comparison = compare_optimizer_norms(
        summarize_optimizer_artifacts(dss, backend="dss", **_optimizer_norms_KWARGS),
        summarize_optimizer_artifacts(ref, **_optimizer_norms_KWARGS),
    )
    assert comparison.worst_delta == 0.0
    assert compare_optimizer_artifacts(dss, ref, **_optimizer_norms_KWARGS).worst_delta == 10.0


@pytest.mark.parametrize("value", [0.0, 1e-30, 0.25])
def test_optimizer_norms_zero_and_near_zero_have_absolute_difference_without_exemption(value):
    reference = _optimizer_norms_summary()
    target = copy.deepcopy(reference)
    target["parameters"]["layer.weight"]["exp_avg"] = value
    comparison = compare_optimizer_norms(target, reference)
    assert comparison.worst_delta == value


def test_optimizer_norms_missing_zero_vision_parameter_is_still_a_coverage_failure():
    reference = _optimizer_norms_summary()
    target = copy.deepcopy(reference["parameters"])
    target["visual.unused.weight"] = copy.deepcopy(target["layer.weight"])
    comparison = compare_optimizer_norms(_optimizer_norms_summary(target), reference)
    assert comparison.only_dss == ["visual.unused.weight"]
    assert comparison.only_reference == []
    assert len(comparison.deltas) == 3


def test_optimizer_norms_comparison_reports_both_unmatched_parameter_sets():
    target = _optimizer_norms_summary({"target.weight": _optimizer_norms_summary()["parameters"]["layer.weight"]})
    reference = _optimizer_norms_summary({"ref.weight": _optimizer_norms_summary()["parameters"]["layer.weight"]})
    comparison = compare_optimizer_norms(target, reference)
    assert comparison.only_dss == ["target.weight"]
    assert comparison.only_reference == ["ref.weight"]
    assert comparison.deltas == []


def test_optimizer_norms_comparison_requires_matching_shapes():
    target = _optimizer_norms_summary()
    reference = copy.deepcopy(target)
    reference["parameters"]["layer.weight"]["shape"] = [1, 2]
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_optimizer_norms(target, reference)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("criterion", "full-tensor"),
        ("step", 2),
        ("step", True),
        ("state_keys", ["exp_avg"]),
        ("parameter_count", 2),
        ("parameter_count", True),
        ("optimizer_values_count", 2),
        ("parameters", {}),
    ],
)
@pytest.mark.parametrize("side", ["target", "reference"])
def test_optimizer_norms_comparison_validates_both_summary_envelopes(key, value, side):
    target, reference = (_optimizer_norms_summary(), _optimizer_norms_summary())
    (target if side == "target" else reference)[key] = value
    with pytest.raises(ValueError):
        compare_optimizer_norms(target, reference)


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True, "0.0"])
def test_optimizer_norms_comparison_rejects_nonfinite_negative_or_nonnumeric_norm(state, value):
    target = _optimizer_norms_summary()
    target["parameters"]["layer.weight"][state] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
def test_optimizer_norms_comparison_requires_complete_per_parameter_state_coverage(state):
    target = _optimizer_norms_summary()
    target["parameters"]["layer.weight"].pop(state)
    with pytest.raises(ValueError, match="state coverage"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


@pytest.mark.parametrize("shape", [[True], [-1], [1.0], "2"])
def test_optimizer_norms_comparison_rejects_invalid_shape_metadata(shape):
    target = _optimizer_norms_summary()
    target["parameters"]["layer.weight"]["shape"] = shape
    with pytest.raises(ValueError, match="invalid shape"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


@pytest.mark.parametrize("name", ["", "model.layer.weight", "module.layer.weight"])
def test_optimizer_norms_comparison_rejects_noncanonical_names(name):
    target = _optimizer_norms_summary({name: _optimizer_norms_summary()["parameters"]["layer.weight"]})
    with pytest.raises(ValueError, match="canonical parameter"):
        compare_optimizer_norms(target, _optimizer_norms_summary())


def test_optimizer_norms_pairwise_norm_calibration_checks_every_pair(_optimizer_norms_artifact_root):
    manifests = [
        _optimizer_norms_manifest(
            _optimizer_norms_artifact_root / f"run-{index}",
            {"layer.weight": _optimizer_norms_states(torch.tensor([value]))},
        )
        for index, value in enumerate((2.0, 0.0, 4.0))
    ]
    updates = []
    progress = SimpleNamespace(update=lambda: updates.append(1))
    worst = max_pairwise_optimizer_norm_delta(manifests, progress=progress, **_optimizer_norms_KWARGS)
    assert (worst.name, worst.state, worst.delta_norm) == ("layer.weight", "parameter_update_residual", 4.0)
    assert updates == [1]


def test_optimizer_norms_pairwise_norm_calibration_sign_change_has_no_magnitude_variation(
    _optimizer_norms_artifact_root,
):
    manifests = [
        _optimizer_norms_manifest(
            _optimizer_norms_artifact_root / f"run-{index}",
            {"layer.weight": _optimizer_norms_states(torch.tensor([value]))},
        )
        for index, value in enumerate((1.0, -1.0))
    ]
    assert max_pairwise_optimizer_norm_delta(manifests, **_optimizer_norms_KWARGS).delta_norm == 0.0


def test_optimizer_norms_pairwise_norm_calibration_empty_or_single_capture(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "single", {"layer.weight": _optimizer_norms_states(torch.zeros(2))}
    )
    assert max_pairwise_optimizer_norm_delta([], **_optimizer_norms_KWARGS) is None
    assert max_pairwise_optimizer_norm_delta([manifest], **_optimizer_norms_KWARGS) is None


@pytest.mark.parametrize("kind", ["names", "shapes"])
def test_optimizer_norms_pairwise_norm_calibration_requires_identical_coverage(_optimizer_norms_artifact_root, kind):
    first = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "first", {"layer.weight": _optimizer_norms_states(torch.zeros(2))}
    )
    second = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "second",
        {
            "other.weight" if kind == "names" else "layer.weight": _optimizer_norms_states(
                torch.zeros(3 if kind == "shapes" else 2)
            )
        },
    )
    with pytest.raises(ValueError, match="parameter sets" if kind == "names" else "shape mismatch"):
        max_pairwise_optimizer_norm_delta([first, second], **_optimizer_norms_KWARGS)


def _optimizer_norms_context_and_results(manifest, reference_norms=None, gate=0.0):

    def forbid_frozen_gate(*args):
        raise AssertionError("must not read a frozen full-tensor gate")

    ctx = SimpleNamespace(
        config_id="config",
        cfg=SimpleNamespace(training={"optimizer": {}}),
        spec=SimpleNamespace(tolerance_for=forbid_frozen_gate),
    )
    reference = {
        "optimizer_norms": reference_norms or _optimizer_norms_summary(),
        "optimizer_norm_gate": {
            "criterion": OPTIMIZER_NORM_CRITERION,
            "absolute": gate,
            "status": "reviewed",
            "provenance": {"review": "explicit-control-review"},
        },
        "loss": 2.0,
        "microbatches": 1,
    }
    target = SimpleNamespace(optimizer_state_manifest=manifest, avg_loss=2.0, model_calls=1)
    return (ctx, SimpleNamespace(name="gas1"), reference, target)


def test_optimizer_norms_arm_consumes_only_stored_norms_and_explicit_reviewed_gate(_optimizer_norms_artifact_root):
    manifest = _optimizer_norms_manifest(
        _optimizer_norms_artifact_root / "dss", {"layer.weight": _optimizer_norms_states(torch.tensor([0.01, 0.0]))}
    )
    ctx, arm, reference, target = _optimizer_norms_context_and_results(manifest, gate=0.02)
    reference["optimizer_state_manifest"] = "do-not-open-reference-tensors"
    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    assert result.outcome is Outcome.PASS
    assert result.metrics["stated_criterion_abs"] == 0.02
    assert result.metrics["max_optimizer_norm_delta"] == pytest.approx(0.01)
    assert OPTIMIZER_NORM_CRITERION in result.reason
    assert "explicit-control-review" in result.reason
    assert "Equal norms can conceal" in result.reason


@pytest.mark.parametrize("state", COMPARISON_STATE_KEYS)
def test_optimizer_norms_arm_gates_every_scalar_and_reports_actual_norms(monkeypatch, state):
    target_summary = _optimizer_norms_summary()
    reference_summary = _optimizer_norms_summary()
    target_summary["parameters"]["layer.weight"][state] = 0.125
    reference_summary["parameters"]["layer.weight"][state] = 0.1
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: target_summary)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target", reference_summary, gate=0.01)
    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["optimizer_values_compared"] == 3
    assert result.metrics["optimizer_values_over_criterion"] == 1
    assert result.worst_name == f"layer.weight::{state}"
    (mismatch,) = result.mismatches
    assert mismatch.target == 0.125
    assert mismatch.reference == 0.1
    assert mismatch.abs_diff == pytest.approx(0.025)


def test_optimizer_norms_arm_gate_boundary_is_inclusive(monkeypatch):
    summary = _optimizer_norms_summary()
    summary["parameters"]["layer.weight"]["exp_avg"] = 0.125
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: summary)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target", gate=0.125)
    assert fwd_bwd_step._compare_arm(ctx, arm, reference, target).outcome is Outcome.PASS


def test_optimizer_norms_arm_coverage_failure_is_not_a_zero_norm_pass(monkeypatch):
    summary = _optimizer_norms_summary(
        {"visual.unused.weight": _optimizer_norms_summary()["parameters"]["layer.weight"]}
    )
    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", lambda *args, **kwargs: summary)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target", gate=100)
    result = fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["unmatched_dss"] == 1
    assert result.metrics["unmatched_reference"] == 1


def test_optimizer_norms_arm_missing_norms_has_no_manifest_fallback(monkeypatch):

    def forbid_tensor_read(*args, **kwargs):
        raise AssertionError("must reject missing compact reference before reading artifacts")

    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", forbid_tensor_read)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target")
    reference.pop("optimizer_norms")
    reference["optimizer_state_manifest"] = "legacy-manifest"
    with pytest.raises(ValueError, match="stored optimizer_norms"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("criterion", "full-tensor"),
        ("status", "proposed"),
        ("absolute", -1.0),
        ("absolute", float("nan")),
        ("absolute", float("inf")),
        ("absolute", True),
        ("absolute", "0.01"),
        ("provenance", None),
        ("provenance", {}),
        ("provenance", ""),
        ("provenance", "   "),
    ],
)
def test_optimizer_norms_arm_rejects_invalid_or_unreviewed_gate_before_artifact_read(monkeypatch, key, value):

    def forbid_tensor_read(*args, **kwargs):
        raise AssertionError("must reject invalid gate before reading artifacts")

    monkeypatch.setattr(fwd_bwd_step, "summarize_optimizer_artifacts", forbid_tensor_read)
    ctx, arm, reference, target = _optimizer_norms_context_and_results("target")
    reference["optimizer_norm_gate"][key] = value
    with pytest.raises(ValueError, match="optimizer_norm_gate"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)


def test_optimizer_norms_arm_requires_gate_and_target_capture():
    ctx, arm, reference, target = _optimizer_norms_context_and_results(None)
    with pytest.raises(ValueError, match="target optimizer-state manifest"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)
    reference.pop("optimizer_norm_gate")
    with pytest.raises(ValueError, match="reviewed optimizer_norm_gate"):
        fwd_bwd_step._compare_arm(ctx, arm, reference, target)


# Optimizer reference.


@pytest.fixture
def _optimizer_reference_fast_output():
    with tempfile.TemporaryDirectory(prefix="optimizer-reference-selftest-", dir="/data-fast/tmp") as directory:
        yield Path(directory)


def test_optimizer_reference_adamw_reference_writes_named_fp32_moments_and_updated_parameter(
    _optimizer_reference_fast_output: Path,
) -> None:
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16))
    gradient = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    config = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.1}
    initial = model.weight.detach().float().clone()
    expected = torch.nn.Parameter(initial.clone())
    expected.grad = gradient.clone()
    optimizer = torch.optim.AdamW(
        [{"params": [expected], "weight_decay": 0.1}],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-08,
        foreach=False,
        fused=False,
    )
    optimizer.step()
    artifact = run_adamw_step(
        model,
        {"weight": gradient},
        config,
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=_optimizer_reference_fast_output,
    )
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["step"] == 1
    assert [entry["name"] for entry in manifest["parameters"]] == ["weight"]
    tensors = torch.load(_optimizer_reference_fast_output / manifest["parameters"][0]["file"], weights_only=True)
    expected_state = optimizer.state[expected]
    torch.testing.assert_close(tensors["parameter_update"], expected.detach() - initial)
    torch.testing.assert_close(tensors["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(tensors["exp_avg_sq"], expected_state["exp_avg_sq"])


def test_optimizer_reference_adamw_reference_excludes_norm_weight_from_weight_decay(
    _optimizer_reference_fast_output: Path,
) -> None:

    class Model(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(2, elementwise_affine=True, bias=False, dtype=torch.bfloat16)

    model = Model()
    gradient = torch.tensor([0.25, -0.5], dtype=torch.float32)
    run_adamw_step(
        model,
        {"norm.weight": gradient},
        {"name": "AdamW", "weight_decay": 1.0},
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=_optimizer_reference_fast_output,
    )
    manifest = json.loads((_optimizer_reference_fast_output / "manifest.json").read_text())
    tensors = torch.load(_optimizer_reference_fast_output / manifest["parameters"][0]["file"], weights_only=True)
    initial = model.norm.weight.detach().float().clone()
    expected = torch.nn.Parameter(initial.clone())
    expected.grad = gradient.clone()
    optimizer = torch.optim.AdamW([{"params": [expected], "weight_decay": 0.0}], lr=0.01, foreach=False, fused=False)
    optimizer.step()
    torch.testing.assert_close(tensors["parameter_update"], expected.detach() - initial)


def test_optimizer_reference_adamw_reference_treats_zero_clipping_as_disabled(
    _optimizer_reference_fast_output: Path,
) -> None:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    gradient = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    artifact = run_adamw_step(
        model,
        {"weight": gradient},
        {"name": "AdamW", "weight_decay": 0.0},
        learning_rate=0.01,
        gradient_clipping=0,
        optimizer_dtype="float32",
        output_dir=_optimizer_reference_fast_output,
    )
    manifest = json.loads(artifact.manifest_path.read_text())
    tensors = torch.load(_optimizer_reference_fast_output / manifest["parameters"][0]["file"], weights_only=True)
    assert torch.count_nonzero(tensors["parameter_update"]) > 0


@pytest.mark.parametrize("output", ("/code/optimizer", "/tmp/optimizer", "/data-fast", "/data-fast-sibling/optimizer"))
def test_optimizer_reference_adamw_reference_rejects_unsafe_storage_before_model_or_optimizer(
    output, monkeypatch, _optimizer_reference_fast_output
):
    before = list(_optimizer_reference_fast_output.iterdir())
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("model/optimizer/tensor callbacks must not run for invalid output")

    model = SimpleNamespace(named_parameters=forbidden)
    monkeypatch.setattr(optimizer_step, "master_dtype", forbidden)
    monkeypatch.setattr(optimizer_step, "build_adamw", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    with pytest.raises(ValueError, match="reference optimizer output.* /data-fast"):
        run_adamw_step(
            model,
            {},
            {"name": "AdamW"},
            learning_rate=0.01,
            gradient_clipping=None,
            optimizer_dtype="float32",
            output_dir=Path(output),
        )
    assert not calls
    assert list(_optimizer_reference_fast_output.iterdir()) == before


def test_optimizer_reference_adamw_reference_rejects_symlink_escape_before_callbacks(
    monkeypatch, _optimizer_reference_fast_output
):
    link = _optimizer_reference_fast_output / "escaped"
    link.symlink_to("/tmp", target_is_directory=True)
    before = list(_optimizer_reference_fast_output.iterdir())
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("model/optimizer/tensor callbacks must not run for escaped output")

    monkeypatch.setattr(optimizer_step, "master_dtype", forbidden)
    monkeypatch.setattr(optimizer_step, "build_adamw", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    with pytest.raises(ValueError, match="/data-fast"):
        run_adamw_step(
            SimpleNamespace(named_parameters=forbidden),
            {},
            {"name": "AdamW"},
            learning_rate=0.01,
            gradient_clipping=None,
            optimizer_dtype="float32",
            output_dir=link / "optimizer",
        )
    assert not calls
    assert list(_optimizer_reference_fast_output.iterdir()) == before


def test_optimizer_reference_adamw_reference_returns_canonical_fast_manifest(_optimizer_reference_fast_output):
    actual = _optimizer_reference_fast_output / "actual"
    actual.mkdir()
    link = _optimizer_reference_fast_output / "link"
    link.symlink_to(actual, target_is_directory=True)
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    artifact = run_adamw_step(
        model,
        {"weight": torch.tensor([[0.25, -0.5]], dtype=torch.float32)},
        {"name": "AdamW"},
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
        output_dir=link / "optimizer",
    )
    assert artifact.manifest_path == actual / "optimizer/manifest.json"
    assert artifact.manifest_path.is_file()
