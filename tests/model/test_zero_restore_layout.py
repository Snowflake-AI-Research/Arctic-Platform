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
"""Real ZeRO-2 save/resume preserves each GLM-5.3 parameter's master and moments, or refuses a reordered checkpoint."""

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common import deepspeed_engine
from arctic_platform.common.deepspeed_engine import OptimizerSettings
from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import build_deepspeed_config
from arctic_platform.common.deepspeed_engine import initialize_training_engine
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_glm5_next import _tiny_config

# ZeRO-2 with the optimizer offloaded, and ZeRO-1 with fp32 gradient accumulation (DeepSpeed's
# BF16_Optimizer); both restore masters and moments by position.
ZERO = {
    "zero2-offload": {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": False}},
    "zero1-bf16-optimizer": {"stage": 1},
}


def _engine(checkpoint_wrapped_layers=0, zero="zero2-offload", *, parameter_alignment=False):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    torch.manual_seed(0)
    config = _tiny_config()
    config.use_grouped_mm = False
    model = Glm5NextForConditionalGenerationPrimeRL(config)
    with torch.no_grad():  # the tiny model leaves its MoE weights uninitialized, and NaN never equals NaN
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.02)
    layers = model.model.language_model.layers
    for name, layer in list(layers.named_children())[:checkpoint_wrapped_layers]:
        layers.register_module(name, checkpoint_wrapper(layer))  # renames its parameters, moves none
    spec = TrainingEngineSpec(
        dtype="bfloat16",
        world_size=1,
        fp32_masters=True,
        ds_overrides={
            "zero_optimization": {**ZERO[zero], "parameter_alignment": parameter_alignment},
            "zero_force_ds_cpu_optimizer": False,
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    )
    engine, _, _ = initialize_training_engine(
        model,
        build_deepspeed_config(spec),
        optimizer=OptimizerSettings(name="torch_adamw", lr=1e-3),
        ep_size=1,
        mpu=None,
        model_path="zai-org/GLM-5.3-Flash-BF16",
        rank=0,
    )
    return engine


def _train_and_save(engine, path):
    """One step with a different gradient for every parameter, so each has its own master and moments."""
    generator = torch.Generator().manual_seed(1)
    parameters = sorted(engine.module.named_parameters())
    engine.backward(sum((p.float() * torch.randn(p.shape, generator=generator)).sum() for _, p in parameters))
    engine.step()
    engine.save_checkpoint(path, tag="step")
    return _optimizer_state(engine)


def _optimizer_state(engine):
    """Each parameter's fp32 master and Adam moments, read where the engine's own mapping places it (one rank)."""
    zero = engine.optimizer
    state = {}
    groups = getattr(zero, "bit16_groups", None) or zero.bf16_groups  # BF16_Optimizer's names
    flats = getattr(zero, "single_partition_of_fp32_groups", None) or zero.fp32_groups_flat_partition
    for params, flat in zip(groups, flats):
        moments = zero.optimizer.state[flat]
        for p in params:
            address = p._hp_mapping.get_hp_fragment_address()
            tensors = [flat, moments["exp_avg"], moments["exp_avg_sq"]]
            name = engine.param_names[p].replace("._checkpoint_wrapped_module", "")
            state[name] = [t.narrow(0, address.start, address.numel).clone() for t in tensors]
    return state


def _layout(engine):
    groups = getattr(engine.optimizer, "bit16_groups", None) or engine.optimizer.bf16_groups
    return [[engine.param_names[p] for p in group] for group in groups]


@require_deepspeed
@pytest.mark.parametrize("checkpoint_wrapped_layers", [0, 2], ids=["same-module", "more-layers-checkpointed"])
def test_resume_restores_every_parameters_own_master_and_moments(
    single_rank_group, tmp_path, checkpoint_wrapped_layers
):
    expected = _train_and_save(_engine(), tmp_path)
    resumed = _engine(checkpoint_wrapped_layers)
    resumed.load_checkpoint(tmp_path, tag="step")

    for name, state in _optimizer_state(resumed).items():
        for key, value, saved in zip(("master", "exp_avg", "exp_avg_sq"), state, expected[name]):
            torch_assert_equal(value, saved, msg=lambda default, name=name, key=key: f"{name} {key}: {default}")


@require_deepspeed
def test_legacy_compact_layout_restores_into_aligned_zero_groups(single_rank_group, tmp_path):
    from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT
    from deepspeed.checkpoint.constants import PARAM_ALIGNMENT_PADDINGS

    expected = _train_and_save(_engine(), tmp_path)
    (path,) = tmp_path.glob("step/*_optim_states.pt")
    state = torch.load(path, weights_only=False)
    del state[OPTIMIZER_STATE_DICT][PARAM_ALIGNMENT_PADDINGS]
    torch.save(state, path)
    resumed = _engine(parameter_alignment=True)
    resumed.load_checkpoint(tmp_path, tag="step")
    for name, values in _optimizer_state(resumed).items():
        for actual, saved in zip(values, expected[name]):
            torch_assert_equal(actual, saved)


@require_deepspeed
@pytest.mark.parametrize("zero", list(ZERO))
def test_resume_across_a_parameter_order_change_refuses_or_restores_every_parameter(
    single_rank_group, tmp_path, monkeypatch, zero
):
    with monkeypatch.context() as patch:
        patch.setattr(deepspeed_engine, "_aligned_first", list)  # the order before _aligned_first: the model's
        saved = _engine(zero=zero)
    expected = _train_and_save(saved, tmp_path)
    resumed = _engine(zero=zero)
    assert _layout(saved) != _layout(resumed)

    try:
        resumed.load_checkpoint(tmp_path, tag="step")
    except RuntimeError as error:
        assert "saved with the parameters at other positions" in str(error)
        return
    wrong = [
        name
        for name, state in _optimizer_state(resumed).items()
        if not all(torch.equal(value, saved) for value, saved in zip(state, expected[name]))
    ]
    assert not wrong, f"{len(wrong)} of {len(expected)} parameters resumed with another one's state: {wrong[:4]}"


@require_deepspeed
@pytest.mark.parametrize(
    ("saved_dp", "current_dp", "bf16"),
    [(1, 2, False), (2, 1, False), (2, 2, False), (2, 2, True)],
    ids=["expand", "shrink", "rigid-zero", "rigid-bf16"],
)
def test_restore_checks_the_saved_partitions_deepspeed_reads(monkeypatch, saved_dp, current_dp, bf16):
    from deepspeed import comm as dist
    from deepspeed.checkpoint.constants import GROUP_PADDINGS
    from deepspeed.checkpoint.constants import PARAM_SLICE_MAPPINGS
    from deepspeed.checkpoint.constants import SINGLE_PARTITION_OF_FP32_GROUPS
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.utils.tensor_fragment import fragment_address

    cls = BF16_Optimizer if bf16 else DeepSpeedZeroOptimizer
    zero = cls.__new__(cls)
    zero.dp_process_group = None
    zero.real_dp_process_group = [None]
    zero.optimizer = SimpleNamespace(param_groups=[{}])
    a, b = torch.zeros(5), torch.zeros(2)
    zero.param_names = {a: "a", b: "b"}
    if bf16:
        zero.bf16_groups = [[a, b]]
    else:
        zero.round_robin_bit16_groups = [[a, b]]
    zero._load_legacy_checkpoint = lambda states, *_: states
    rank = current_dp - 1
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: current_dp)

    def state(mapping, width, padding):
        return {
            PARAM_SLICE_MAPPINGS: [mapping],
            SINGLE_PARTITION_OF_FP32_GROUPS: [torch.zeros(width if bf16 else width - padding)],
            GROUP_PADDINGS: [padding],
        }

    if saved_dp == 1:
        saved = [state({"a": fragment_address(5, 0), "b": fragment_address(2, 5)}, 8, 1)]
    else:
        saved = [
            state({"a": fragment_address(4, 0)}, 4, 0) if saved_dp != current_dp else None,
            state({"a": fragment_address(1, 0), "b": fragment_address(2, 1)}, 4, 1),
        ]
    deepspeed_engine._refuse_restoring_another_parameter_layout(zero)
    assert zero._load_legacy_checkpoint(saved, False, True) is saved
    mapping = saved[-1][PARAM_SLICE_MAPPINGS][0]
    mapping["other"] = mapping.pop("b")
    with pytest.raises(RuntimeError, match="saved with the parameters at other positions"):
        zero._load_legacy_checkpoint(saved, False, True)
