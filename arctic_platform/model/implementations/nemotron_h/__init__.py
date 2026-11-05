from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig

__all__ = [
    "NemotronHConfig",
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHPreTrainedModel",
]


def __getattr__(name: str):
    if name in {"NemotronHForCausalLM", "NemotronHModel", "NemotronHPreTrainedModel"}:
        from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import (
            NemotronHForCausalLM,
            NemotronHModel,
            NemotronHPreTrainedModel,
        )

        return {
            "NemotronHForCausalLM": NemotronHForCausalLM,
            "NemotronHModel": NemotronHModel,
            "NemotronHPreTrainedModel": NemotronHPreTrainedModel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
