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

"""
ZoRRO Train Prompt deduplication optimization for RL training.

This package provides utilities to deduplicate shared prompts across batch samples
during forward and backward passes, reducing computation while maintaining
gradient correctness.
"""

from .actor import DeduplicatedActor
from .module_patcher import ModuleReconstructionPatcher
from .qwen_attention_patcher import QwenAttentionPatcher
from .qwen_model_patcher import Qwen3ModelPatcher
from .qwen_model_patcher import ReconstructionInfo
from .zorro_train import ZoRRoTrain

__all__ = [
    "DeduplicatedActor",
    "ModuleReconstructionPatcher",
    "Qwen3ModelPatcher",
    "QwenAttentionPatcher",
    "ReconstructionInfoZoRRoTrain",
]
