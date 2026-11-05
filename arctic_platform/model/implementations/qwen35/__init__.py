"""Self-contained Qwen3.5-MoE loading path carved out of prime-rl.

This package contains the DeepSpeed and expert-parallel implementation behind
the public ``qwen3_5_moe`` model loader.
"""

__all__ = [
    "load_qwen3_5_moe_model",
    "load_moe_model_for_deepspeed",
    "patch_deepspeed_moe_detection",
]


def __getattr__(name):
    if name in __all__:
        from . import deepspeed_integration

        return getattr(deepspeed_integration, name)
    raise AttributeError(name)
