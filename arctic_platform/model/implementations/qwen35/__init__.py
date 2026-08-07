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
