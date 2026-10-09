from arctic_platform.model.implementations.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config

__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2Model",
    "MiniMaxM2PreTrainedModel",
]


def __getattr__(name: str):
    if name in {"MiniMaxM2ForCausalLM", "MiniMaxM2Model", "MiniMaxM2PreTrainedModel"}:
        from arctic_platform.model.implementations.minimax_m2.modeling_minimax_m2 import (
            MiniMaxM2ForCausalLM,
            MiniMaxM2Model,
            MiniMaxM2PreTrainedModel,
        )

        return {
            "MiniMaxM2ForCausalLM": MiniMaxM2ForCausalLM,
            "MiniMaxM2Model": MiniMaxM2Model,
            "MiniMaxM2PreTrainedModel": MiniMaxM2PreTrainedModel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
