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

"""LoRA on the DeepSeek-V4.1 entry point.

V4.1 checkpoints declare the vision wrapper, which vLLM ships without
``SupportsLoRA``; its text backbone has it. Adapters go on the language model
only, as for V4.
"""

ARCHITECTURE = "DeepseekV41ForCausalLM"

_REGISTERED = False


def ensure_deepseek_v41_lora() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm import ModelRegistry

    if ARCHITECTURE in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCHITECTURE,
            "arctic_platform.inference.vllm.deepseek_v41_lora_model:DeepseekV41LoRAForCausalLM",
        )
    _REGISTERED = True
