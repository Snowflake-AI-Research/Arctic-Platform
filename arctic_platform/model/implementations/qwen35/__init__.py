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
"""Self-contained Qwen3.5-MoE loading path carved out of prime-rl.

This package replicates ``prime_rl``'s DeepSpeed + expert-parallel + DeepEP
loading path for the Qwen3.5-MoE model family with no imports from ``prime_rl``.
The primary entry point is :func:`load_moe_model_for_dss`.
"""

__all__ = [
    "load_moe_model_for_dss",
    "load_moe_model_for_deepspeed",
    "patch_deepspeed_moe_detection",
]


def __getattr__(name):
    if name in __all__:
        from . import deepspeed_integration

        return getattr(deepspeed_integration, name)
    raise AttributeError(name)
