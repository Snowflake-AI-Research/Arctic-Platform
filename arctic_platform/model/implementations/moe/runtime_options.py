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

``MoeRuntimeOptions`` is a base of the ``glm_moe_dsa`` loader's options (``GlmMoeDsaOptions``), the loader that
installs these modules, so a job sets its fields like any other loader option (DSS:
``training_config.model_options.moe.*``, or the ``prime_rl`` block). Each field steers one module that keeps its
environment variable as the fallback: a field left unset (None) leaves that variable in charge, so a job that sets
none of them runs exactly as before. ``apply_moe_runtime_options`` hands the fields to their modules once per model
load, before the model is built; the other MoE loaders have no such fields, which leaves every module to its variable.
"""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# field -> module that reads it, relative to arctic_platform.model.implementations
_MODULES = {
    "frozen_expert_offload": "moe.frozen_offload",
}


class MoeRuntimeOptions(BaseModel):
    """Memory settings of the MoE runtime. None: the environment variable decides."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    frozen_expert_offload: bool | None = Field(
        None,
        description=(
            "Keep the frozen routed experts' weights (codes) in pinned host memory, copied into two GPU buffers per "
            "experts call with the next layer prefetched (moe/frozen_offload.py). Every layer with routed experts "
            "must be recomputed. Unset: ARL_OFFLOAD_FROZEN_EXPERTS."
        ),
    )


def apply_moe_runtime_options(options: Any) -> None:
    """Hand ``options``' runtime fields to the modules that read them; a field the options model lacks counts as
    unset. Every load sets every field, so a value from an earlier load in the same process never leaks into this
    one."""
    for field, module_name in _MODULES.items():
        module = importlib.import_module(f"arctic_platform.model.implementations.{module_name}")
        module.configure_from_options(getattr(options, field, None))
