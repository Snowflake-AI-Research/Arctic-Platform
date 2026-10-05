"""Self-contained GLM MoE DSA loading path carved out of prime-rl.

Vendored from arctic-primerl ``trainer/models/glm_moe_dsa`` at e7f66ccb.
"""

__all__ = [
    "load_glm_moe_dsa_model",
]


def __getattr__(name):
    if name in __all__:
        from . import deepspeed_integration

        return getattr(deepspeed_integration, name)
    raise AttributeError(name)
