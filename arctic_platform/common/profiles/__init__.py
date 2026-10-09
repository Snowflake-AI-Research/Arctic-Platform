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

"""Model profiles, one per registered loader and named after it. ``register`` adds them all to a registry.

A cell is ``supports`` only when the loader's own options model (``Patches`` and ``ModelSpec`` for the
``huggingface`` loader) has a field for the option at this commit. ``tests/model/test_option_profiles.py``
checks that against the loaders.
"""

from __future__ import annotations

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.profiles import generic_moe
from arctic_platform.common.profiles import glm5_next
from arctic_platform.common.profiles import glm_moe_dsa
from arctic_platform.common.profiles import huggingface
from arctic_platform.common.profiles import qwen3_5_moe
from arctic_platform.common.profiles import qwen4_exp

_PROFILE_MODULES = (huggingface, generic_moe, qwen3_5_moe, glm_moe_dsa, glm5_next, qwen4_exp)


def register(registry: Registry) -> None:
    """Register every built-in profile into ``registry``."""
    for module in _PROFILE_MODULES:
        module.register(registry)
