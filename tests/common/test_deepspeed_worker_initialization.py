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

"""The on-prem worker hands DeepSpeed a TrainingEngineSpec."""

from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import build_deepspeed_config


def test_onprem_engine_adapter_translates_optimizer_and_preserves_scheduler_and_gas():
    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 3,
        "gradient_clipping": 0.5,
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}},
        "optimizer": {"type": "AdamW", "params": {"lr": 3e-4, "torch_adam": True}},
        "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 10}},
    }

    spec = TrainingEngineSpec.from_deepspeed_config(ds_config, dtype="bfloat16", world_size=4, seed=7)
    resolved = build_deepspeed_config(spec)

    assert spec.optimizer.name == "torch_adamw"
    assert spec.optimizer.lr == 3e-4
    assert spec.optimizer.fused is False
    assert resolved["train_batch_size"] == 24
    assert resolved["zero_optimization"] == ds_config["zero_optimization"]
    assert "optimizer" not in resolved
    assert resolved["scheduler"] == ds_config["scheduler"]
    assert resolved["seed"] == 7
