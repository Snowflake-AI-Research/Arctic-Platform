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
"""Stochastic rounding of ZeRO-1/2's bf16 parameters, through a real two-rank DeepSpeed engine on CPU (gloo).

Each element's update is ``lr * sign(g)`` (AdamW on a constant gradient) and stays below half a bf16 spacing for all
``_STEPS`` steps, so round-to-nearest never moves the bf16 copy while the fp32 master moves on every step.
"""

from __future__ import annotations

import json
import math
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.common.deepspeed_engine import OptimizerSettings
from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import build_deepspeed_config
from arctic_platform.common.deepspeed_engine import initialize_training_engine
from arctic_platform.common.stochastic_rounding import stochastic_round_to_bf16
from arctic_platform.model.weight_export import iter_model_weights
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import single_rank_process_group

_WORLD = 2
_NUMEL = (1 << 16) + 1
_STEPS = 8
_LR = 1e-4  # 8 steps move a master 8e-4: below half the bf16 spacing (9.8e-4) of every |w| in [0.25, 1).


def _bf16_spacing(q: torch.Tensor) -> torch.Tensor:
    """The distance from bf16 ``q`` to its neighbour away from zero, in float64."""
    away = (q.view(torch.int16).to(torch.int32) + 1).to(torch.int16).view(torch.bfloat16)
    return (away.double() - q.double()).abs()


def _bit_neighbours(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The bf16 values on either side of fp32 ``x``: truncated toward zero, and one spacing further from zero."""
    toward_zero = x.view(torch.int32) & -(1 << 16)
    return toward_zero.view(torch.float32).double(), (toward_zero + (1 << 16)).view(torch.float32).double()


def test_stochastic_rounding_is_unbiased_for_either_sign_and_subnormals():
    tiny = 2.0**-133  # the smallest bf16 subnormal
    values = torch.tensor(
        [
            1.0 + 0.25 * 2**-7,
            1.0 + 0.5 * 2**-7,
            -(1.0 + 0.75 * 2**-7),
            0.3,
            -0.3,
            1e-3,
            -7e5,
            2.5 * tiny,
            -1.25 * tiny,
        ],
        dtype=torch.float32,
    )
    draws = 1 << 18
    x = values.repeat_interleave(draws)
    generator = torch.Generator().manual_seed(0)
    q = stochastic_round_to_bf16(x, generator).double().view(len(values), draws)
    low, high = (bound.view(len(values), draws)[:, 0] for bound in _bit_neighbours(x))
    fraction = (values.double() - low) / (high - low)
    assert ((q == low[:, None]) | (q == high[:, None])).all()
    up = (q == high[:, None]).double().mean(dim=1)
    standard_error = (fraction * (1 - fraction) / draws).sqrt()
    assert ((up - fraction).abs() < 5 * standard_error).all(), (up, fraction)


def test_stochastic_rounding_keeps_bf16_values_and_rounds_non_finite_to_nearest():
    exact = torch.tensor([0.0, -0.0, 1.0, -0.5, 2.0**-133, -(2.0**-126), 3.3895e38], dtype=torch.float32)
    exact = exact.to(torch.bfloat16).float().repeat(1000)
    out = stochastic_round_to_bf16(exact, torch.Generator().manual_seed(1))
    assert torch.equal(out.view(torch.int16), exact.to(torch.bfloat16).view(torch.int16))
    non_finite = torch.tensor([float("inf"), float("-inf"), float("nan")])
    out = stochastic_round_to_bf16(non_finite, torch.Generator().manual_seed(2))
    assert torch.equal(out.view(torch.int16), non_finite.to(torch.bfloat16).view(torch.int16))


def test_stochastic_rounding_repeats_a_seeded_draw():
    x = torch.randn(1 << 12)
    first = stochastic_round_to_bf16(x, torch.Generator().manual_seed(3))
    assert torch.equal(first, stochastic_round_to_bf16(x, torch.Generator().manual_seed(3)))
    assert not torch.equal(first, stochastic_round_to_bf16(x, torch.Generator().manual_seed(4)))


class _Weights(torch.nn.Module):
    """One bf16-trained vector with |w| in [0.25, 1) on the bf16 grid, and one parameter kept in fp32."""

    @staticmethod
    def keeps_fp32_parameter(name: str) -> bool:
        return name == "kept"

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(5)
        magnitude = (torch.rand(_NUMEL, generator=generator) * 0.75 + 0.25).to(torch.bfloat16)
        # No powers of two, whose spacing toward zero is half the spacing away from it.
        bits = magnitude.view(torch.int16)
        magnitude = torch.where(bits & 0x7F == 0, bits + 1, bits).view(torch.bfloat16)
        sign = torch.randint(0, 2, (_NUMEL,), generator=generator) * 2 - 1
        self.weight = torch.nn.Parameter(sign * magnitude.float())
        self.kept = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 64))
        self.gradient = torch.randint(0, 2, (_NUMEL,), generator=generator).float() * 2 - 1


def _engine_worker(rank, init_file, out, ds_overrides, stochastic):
    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE=str(_WORLD))
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=_WORLD, rank=rank, timeout=timedelta(seconds=120)
    )
    try:
        from deepspeed.utils import safe_get_full_fp32_param

        model = _Weights()
        spec = TrainingEngineSpec(world_size=_WORLD, fp32_masters=True, seed=11, ds_overrides=ds_overrides)
        engine, _, _ = initialize_training_engine(
            model,
            build_deepspeed_config(spec),
            optimizer=OptimizerSettings(name="torch_adamw", lr=_LR, eps=1e-15),
            ep_size=1,
            mpu=None,
            model_path="",
            rank=rank,
            stochastic_round_bf16_params=stochastic,
        )
        weight, kept = engine.module.weight, engine.module.kept
        served = [weight.detach().clone()]
        masters = [safe_get_full_fp32_param(weight).double()]
        for _ in range(_STEPS):
            engine.backward((weight.float() * model.gradient).sum() + kept.sum())
            engine.step()
            served.append(weight.detach().clone())
            masters.append(safe_get_full_fp32_param(weight).double())
        gathered = [torch.empty(_NUMEL) for _ in range(_WORLD)]
        dist.all_gather(gathered, weight.detach().float())
        kept_matches_master = torch.equal(kept.detach(), safe_get_full_fp32_param(kept))
        if rank == 0:
            q, theta = [s.double() for s in served], masters
            spacing = _bf16_spacing(served[0])
            drift = torch.sign(theta[-1] - theta[0])
            lag = (q[-1] - theta[-1]) * drift / spacing
            served_move = (q[-1] - q[0]) * drift / spacing
            # Each master's fractional position between q[0] and its neighbour in the drift direction.
            f = [(t - q[0]).abs() / spacing for t in theta]
            # Draws are independent across steps: q changes with probability f_K (1 - f_{K-1}) + f_{K-1} (1 - f_K).
            flips = (f[-1] * (1 - f[-2]) + f[-2] * (1 - f[-1])).mean()
            summary = dict(
                cpu_offload=bool(engine.optimizer.cpu_offload),
                dtype=str(weight.dtype),
                kept_dtype=str(kept.dtype),
                kept_equals_master=bool(kept_matches_master),
                ranks_identical=all(torch.equal(g, gathered[0]) for g in gathered),
                q0_on_master=bool(torch.equal(q[0], theta[0])),
                served_is_nearest=bool(torch.equal(served[-1], theta[-1].float().to(torch.bfloat16))),
                max_master_move_over_half_spacing=float(((theta[-1] - theta[0]).abs() / (spacing / 2)).max()),
                served_changed_fraction=float((served[-1] != served[0]).double().mean()),
                last_step_changed_fraction=float((served[-1] != served[-2]).double().mean()),
                last_step_changed_expected=float(flips),
                lag_mean=float(lag.mean()),
                lag_standard_error=float(lag.std() / math.sqrt(_NUMEL)),
                lag_negative_fraction=float((lag < 0).double().mean()),
                master_move_mean=float(((theta[-1] - theta[0]) * drift / spacing).mean()),
                served_move_mean=float(served_move.mean()),
                served_move_standard_error=float(served_move.std() / math.sqrt(_NUMEL)),
            )
            with open(out, "w") as file:
                json.dump(summary, file)
        for weight_format in ("hf", "vllm"):
            exported = dict(iter_model_weights(engine.module, weight_format, is_master=True, is_zero3=False))
            for name, parameter in engine.module.named_parameters():
                assert torch.equal(exported[name], parameter), (weight_format, name)
        if stochastic:
            checkpoint_dir = init_file + "_checkpoint"
            engine.save_checkpoint(checkpoint_dir, tag="saved")
            engine.backward((weight.float() * model.gradient).sum() + kept.sum())
            engine.step()
            uninterrupted = weight.detach().clone()
            engine.load_checkpoint(checkpoint_dir, tag="saved")
            assert engine.global_steps == _STEPS
            assert torch.equal(weight, served[-1])
            engine.backward((weight.float() * model.gradient).sum() + kept.sum())
            engine.step()
            assert torch.equal(weight, uninterrupted)
    finally:
        dist.destroy_process_group()


_OFFLOAD = {
    "no_offload": {"zero_optimization": {"stage": 2}, "zero_force_ds_cpu_optimizer": False},
    "cpu_offload": {
        "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": False}},
        "zero_force_ds_cpu_optimizer": False,
    },
}


def _run_engine(tmp_path, offload, stochastic) -> dict:
    out = tmp_path / "summary.json"
    mp.spawn(_engine_worker, args=(str(tmp_path / "group"), str(out), _OFFLOAD[offload], stochastic), nprocs=_WORLD)
    summary = json.loads(out.read_text())
    assert summary["cpu_offload"] == (offload == "cpu_offload")
    assert summary["dtype"] == "torch.bfloat16" and summary["kept_dtype"] == "torch.float32"
    assert summary["kept_equals_master"] and summary["ranks_identical"] and summary["q0_on_master"]
    assert summary["max_master_move_over_half_spacing"] < 1
    return summary


@require_deepspeed
@pytest.mark.parametrize("offload", sorted(_OFFLOAD))
def test_round_to_nearest_leaves_the_bf16_copy_behind_its_master(tmp_path, offload):
    summary = _run_engine(tmp_path, offload, stochastic=False)
    assert summary["served_is_nearest"] and summary["served_changed_fraction"] == 0.0
    assert summary["lag_negative_fraction"] == 1.0
    assert summary["lag_mean"] == pytest.approx(-summary["master_move_mean"], rel=1e-6)


@require_deepspeed
@pytest.mark.parametrize("offload", sorted(_OFFLOAD))
def test_stochastic_rounding_keeps_the_bf16_copy_unbiased(tmp_path, offload):
    summary = _run_engine(tmp_path, offload, stochastic=True)
    assert abs(summary["lag_mean"]) < 5 * summary["lag_standard_error"], summary
    served, master = summary["served_move_mean"], summary["master_move_mean"]
    assert abs(served - master) < 5 * summary["served_move_standard_error"], summary
    changed, expected = summary["last_step_changed_fraction"], summary["last_step_changed_expected"]
    assert abs(changed - expected) < 5 * math.sqrt(expected * (1 - expected) / _NUMEL), summary


@require_deepspeed
def test_stochastic_rounding_refuses_bf16_masters(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    with single_rank_process_group():
        ds_config = build_deepspeed_config(
            TrainingEngineSpec(world_size=1, ds_overrides={"zero_force_ds_cpu_optimizer": False})
        )
        assert ds_config["bf16"]["bf16_master_weights_and_grads"]
        with pytest.raises(ValueError, match="fp32 master weights"):
            initialize_training_engine(
                _Weights(),
                ds_config,
                optimizer=OptimizerSettings(name="torch_adamw"),
                ep_size=1,
                mpu=None,
                model_path="",
                rank=0,
                stochastic_round_bf16_params=True,
            )
