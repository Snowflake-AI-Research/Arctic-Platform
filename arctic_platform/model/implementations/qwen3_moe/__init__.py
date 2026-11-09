from arctic_platform.model.implementations.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

__all__ = [
    "Qwen3MoeConfig",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
]


def __getattr__(name: str):
    if name in {"Qwen3MoeForCausalLM", "Qwen3MoeModel", "Qwen3MoePreTrainedModel"}:
        from arctic_platform.model.implementations.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeForCausalLM,
            Qwen3MoeModel,
            Qwen3MoePreTrainedModel,
        )

        return {
            "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
            "Qwen3MoeModel": Qwen3MoeModel,
            "Qwen3MoePreTrainedModel": Qwen3MoePreTrainedModel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
