from arctic_platform.model.implementations.afmoe.configuration_afmoe import AfmoeConfig

__all__ = [
    "AfmoeConfig",
    "AfmoeForCausalLM",
    "AfmoeModel",
    "AfmoePreTrainedModel",
]


def __getattr__(name: str):
    if name in {"AfmoeForCausalLM", "AfmoeModel", "AfmoePreTrainedModel"}:
        from arctic_platform.model.implementations.afmoe.modeling_afmoe import (
            AfmoeForCausalLM,
            AfmoeModel,
            AfmoePreTrainedModel,
        )

        return {
            "AfmoeForCausalLM": AfmoeForCausalLM,
            "AfmoeModel": AfmoeModel,
            "AfmoePreTrainedModel": AfmoePreTrainedModel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
