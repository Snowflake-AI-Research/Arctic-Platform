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
"""Per-job settings of the MoE runtime that were first worker environment variables.

``MoeRuntimeOptions`` is a base of the MoE loaders' ``loader_options`` models, so a job sets these fields like any
other loader option (DSS: ``training_config.model_options.moe.*``, or the ``prime_rl`` block). Each field steers one
module that keeps its environment variable as the fallback: a field left unset (None) leaves that variable in charge,
so a job that sets none of them runs exactly as before. ``apply_moe_runtime_options`` hands the fields to their
modules once per model load, before the model is built. A module a branch does not have (W4A8, UltraEP for packed
experts) refuses a field set to anything but its off value.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# field -> (module that reads it, relative to arctic_platform.model.implementations)
_MODULES = {
    "frozen_expert_offload": "moe.frozen_offload",
    "shard_frozen_dense": "moe.frozen_shard",
    "w4a8_experts": "kernels.w4a8",
    "ultraep_spares": "moe.ultraep_packed4",
}
_ULTRA_EP_IPC_SYMM_ENV = "ULTRA_EP_IPC_SYMM"


class MoeRuntimeOptions(BaseModel):
    """Memory and expert-balance settings shared by the MoE loaders' options. None: the environment variable decides."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    frozen_expert_offload: bool | None = Field(
        None,
        description=(
            "Keep the frozen routed experts' weights (codes) in pinned host memory, copied into two GPU buffers per "
            "experts call with the next layer prefetched (moe/frozen_offload.py; layers that are not recomputed keep "
            "theirs on the GPU). Unset: ARL_OFFLOAD_FROZEN_EXPERTS."
        ),
    )
    shard_frozen_dense: bool | None = Field(
        None,
        description=(
            "Hold each recomputed layer's frozen replicated weights as a 1/world shard per rank, all-gathered for each "
            "call of the layer (moe/frozen_shard.py). Unset: ARL_SHARD_FROZEN_DENSE."
        ),
    )
    w4a8_experts: bool | None = Field(
        None,
        description=(
            "Run INT4 routed experts with FP8 activations (W4A8, the checkpoint's static activation scales) instead of "
            "decoding them to BF16 (kernels/w4a8). Unset: ARL_W4A8_EXPERTS."
        ),
    )
    ultraep_spares: int | None = Field(
        None,
        ge=0,
        description=(
            "Spare expert slots per GPU that UltraEP fills, in every MoE call, with copies of that call's hottest "
            "frozen packed experts and their LoRA inside the NVLink domain (moe/ultraep_packed4.py). 0 turns UltraEP "
            "off. Unset: ARL_ULTRAEP_SPARES."
        ),
    )
    ultraep_ipc_symm: bool | None = Field(
        None,
        description=(
            "Map UltraEP's peer buffers through CUDA IPC handles instead of torch symmetric memory, whose rendezvous "
            "refuses ranks that each see only their own GPU (as DSS runs them). Written to ULTRA_EP_IPC_SYMM, which "
            "UltraEP reads, before its buffers are allocated. Unset: ULTRA_EP_IPC_SYMM."
        ),
    )


def apply_moe_runtime_options(options: Any) -> None:
    """Hand ``options``' runtime fields to the modules that read them; a field the options model lacks counts as
    unset. Every load sets every field, so a value from an earlier load in the same process never leaks into this
    one."""
    for field, module_name in _MODULES.items():
        value = getattr(options, field, None)
        try:
            module = importlib.import_module(f"arctic_platform.model.implementations.{module_name}")
        except ImportError:
            if value:
                raise ValueError(f"loader option {field}={value!r}: this branch has no {module_name}") from None
            continue
        module.configure_from_options(value)
    ipc_symm = getattr(options, "ultraep_ipc_symm", None)
    if ipc_symm is not None:
        os.environ[_ULTRA_EP_IPC_SYMM_ENV] = "1" if ipc_symm else "0"
