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
"""GLM-5.3-Flash's FP32 checkpoint parameters (mHC base/scale, KDA A_log/dt_bias) stay FP32 through a real
``deepspeed.initialize`` with the production precision (bf16, FP32 or BF16 masters, ZeRO-2), and an optimizer step
moves them by the master's update instead of leaving their bf16 copy unchanged. Their gradients reach the optimizer
unrounded, and BF16 masters round neither them nor their updates.
"""

import pytest
import torch

from arctic_platform.common.deepspeed_engine import OptimizerSettings
from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import build_deepspeed_config
from arctic_platform.common.deepspeed_engine import initialize_training_engine
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_glm5_next import _tiny_config

LR = 1e-3


_SUFFIXES = (
    "attn_hc.base",
    "attn_hc.scale",
    "ffn_hc.base",
    "ffn_hc.scale",
    "forget_gate.A_log",
    "forget_gate.dt_bias",
)


def _model_with_unrepresentable_fp32_parameters():
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    config = _tiny_config()
    config.use_grouped_mm = False
    model = Glm5NextForConditionalGenerationPrimeRL(config)
    exact = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(_SUFFIXES):
                # Not bf16-representable, and a bf16 ulp at these magnitudes is far larger than one LR step.
                parameter.copy_(torch.linspace(-8.0123, 7.98765, parameter.numel()).view_as(parameter))
                exact[name] = parameter.detach().clone()
    assert all(any(name.endswith(suffix) for name in exact) for suffix in _SUFFIXES)
    return model, exact


@require_deepspeed
@pytest.mark.parametrize("fp32_masters", [True, False], ids=["fp32_masters", "bf16_masters"])
def test_glm53_fp32_parameters_survive_deepspeed_and_train(single_rank_group, fp32_masters):
    model, exact = _model_with_unrepresentable_fp32_parameters()

    spec = TrainingEngineSpec(
        dtype="bfloat16",
        world_size=1,
        fp32_masters=fp32_masters,
        ds_overrides={
            "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": False}},
            "zero_force_ds_cpu_optimizer": False,
        },
    )
    engine, _, _ = initialize_training_engine(
        model,
        build_deepspeed_config(spec),
        optimizer=OptimizerSettings(name="torch_adamw", lr=LR, eps=1e-15),
        ep_size=1,
        mpu=None,
        model_path="zai-org/GLM-5.3-Flash-BF16",
        rank=0,
    )

    parameters = dict(engine.module.named_parameters())
    assert parameters["lm_head.weight"].dtype == torch.bfloat16
    for name, value in exact.items():
        torch_assert_equal(parameters[name].cpu(), value, msg=lambda default, name=name: f"{name}: {default}")

    engine.backward(sum(parameter.float().sum() for parameter in engine.module.parameters()))
    engine.step()

    for name, value in exact.items():  # Adam's first step with a unit gradient is -LR
        torch_assert_close(
            parameters[name].cpu(), value - LR, rtol=0, atol=1e-6, msg=lambda default, name=name: f"{name}: {default}"
        )


@require_deepspeed
@require_torch_gpu
@pytest.mark.parametrize("fp32_masters", [True, False], ids=["fp32_masters", "bf16_masters"])
def test_glm53_fp32_parameters_keep_their_gradient_and_master_in_a_zero2_engine(single_rank_group, fp32_masters):
    from deepspeed.utils import safe_get_full_grad

    model, exact = _model_with_unrepresentable_fp32_parameters()
    spec = TrainingEngineSpec(dtype="bfloat16", world_size=1, fp32_masters=fp32_masters)
    engine, _, _ = initialize_training_engine(
        model.cuda(),
        build_deepspeed_config(spec),
        optimizer=OptimizerSettings(name="fused_adam", lr=0.0),
        ep_size=1,
        mpu=None,
        model_path="zai-org/GLM-5.3-Flash-BF16",
        rank=0,
    )
    parameters = dict(engine.module.named_parameters())
    fp32_names = [name for name in exact if parameters[name].dtype == torch.float32]
    assert set(fp32_names) == set(exact)

    # 1.001 is not bf16-representable; a bf16 staging bucket turns it into 1.0.
    engine.backward(
        sum(1.001 * parameters[name].sum() for name in exact)
        + sum(parameter.float().sum() for name, parameter in parameters.items() if name not in exact)
    )
    for name in exact:
        torch_assert_equal(
            safe_get_full_grad(parameters[name]).cpu(),
            torch.full(exact[name].shape, 1.001),
            msg=lambda default, name=name: f"{name} gradient: {default}",
        )
    engine.step()

    for name, value in exact.items():  # lr=0, so a bf16 master would still round the parameter
        torch_assert_equal(parameters[name].cpu(), value, msg=lambda default, name=name: f"{name}: {default}")
