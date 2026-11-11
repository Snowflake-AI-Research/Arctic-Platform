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
"""Per-job settings of the MoE runtime that would otherwise be worker environment variables.

``MoeRuntimeOptions`` is a base of the MoE loaders' ``loader_options`` models (glm_moe_dsa, qwen3_5_moe, which the
deepseek_v4 / deepseek_v41 and flash_moe loaders validate with too, and generic_moe), so a job sets these fields like
any other loader option (DSS: ``training_config.model_options.moe.*``, or the ``prime_rl`` block). Each field steers
one module that keeps its environment variable as the fallback: a field left unset (None) leaves that variable in
charge, so a job that sets none of them runs exactly as before. ``apply_moe_runtime_options`` hands the fields to
their modules once per model load, before the model is built.
"""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# field -> (module that reads it, relative to arctic_platform.model.implementations; the setter it takes it through)
_SETTERS = {
    "w4a8_experts": ("kernels.w4a8", "configure_from_options"),
    "w8a8_experts": ("kernels.fp8_grouped_w8a8", "configure_from_options"),
}


class MoeRuntimeOptions(BaseModel):
    """Settings of the MoE runtime, one field per module setting. None: the environment variable decides."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    w4a8_experts: bool | None = Field(
        None,
        description=(
            "Run INT4 routed experts with FP8 activations (W4A8, the checkpoint's static activation scales) instead of "
            "decoding them to BF16 (kernels/w4a8). Unset: ARL_W4A8_EXPERTS."
        ),
    )
    w8a8_experts: bool | None = Field(
        None,
        description=(
            "Run the frozen block-FP8 routed experts' forward GEMMs as W8A8 (activations quantized to E4M3 per token "
            "per 128 inputs) through DeepGEMM on Hopper, instead of decoding the weights to BF16; input gradients "
            "stay BF16 GEMMs on the decoded weights (kernels/fp8_grouped_w8a8.py). Refused for experts that are not "
            "block FP8. Unset: ARL_W8A8_EXPERTS."
        ),
    )


def apply_moe_runtime_options(options: Any) -> None:
    """Hand ``options``' runtime fields to the modules that read them; a field the options model lacks counts as
    unset. Every load sets every field, so a value from an earlier load in the same process never leaks into this
    one."""
    for field, (module_name, setter) in _SETTERS.items():
        module = importlib.import_module(f"arctic_platform.model.implementations.{module_name}")
        getattr(module, setter)(getattr(options, field, None))
