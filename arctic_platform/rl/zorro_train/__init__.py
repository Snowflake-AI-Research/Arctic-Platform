"""
ZoRRO Train Prompt deduplication optimization for RL training.

This package provides utilities to deduplicate shared prompts across batch samples
during forward and backward passes, reducing computation while maintaining
gradient correctness.
"""

from .zorro_train import ZoRRoTrain
from .module_patcher import ModuleReconstructionPatcher
from .qwen_attention_patcher import QwenAttentionPatcher
from .qwen_model_patcher import Qwen3ModelPatcher, ReconstructionInfo
from .actor import DeduplicatedActor

__all__ = [
    "DeduplicatedActor",
    "ModuleReconstructionPatcher",
    "Qwen3ModelPatcher",
    "QwenAttentionPatcher",
    "ReconstructionInfo"
    "ZoRRoTrain",
]

