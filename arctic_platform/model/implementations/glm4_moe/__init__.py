from arctic_platform.model.implementations.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

__all__ = [
    "Glm4MoeConfig",
    "Glm4MoeForCausalLM",
    "Glm4MoeModel",
    "Glm4MoePreTrainedModel",
]


def __getattr__(name: str):
    if name in {"Glm4MoeForCausalLM", "Glm4MoeModel", "Glm4MoePreTrainedModel"}:
        from arctic_platform.model.implementations.glm4_moe.modeling_glm4_moe import (
            Glm4MoeForCausalLM,
            Glm4MoeModel,
            Glm4MoePreTrainedModel,
        )

        return {
            "Glm4MoeForCausalLM": Glm4MoeForCausalLM,
            "Glm4MoeModel": Glm4MoeModel,
            "Glm4MoePreTrainedModel": Glm4MoePreTrainedModel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
