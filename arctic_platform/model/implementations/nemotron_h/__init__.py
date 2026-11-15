from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHOmniTextConfig

__all__ = [
    "NemotronHConfig",
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHOmniTextConfig",
    "NemotronHOmniTextForCausalLMPrimeRL",
    "NemotronHPreTrainedModel",
]

_MODELING_NAMES = {
    "NemotronHForCausalLM",
    "NemotronHModel",
    "NemotronHOmniTextForCausalLMPrimeRL",
    "NemotronHPreTrainedModel",
}


def __getattr__(name: str):
    if name in _MODELING_NAMES:
        from arctic_platform.model.implementations.nemotron_h import modeling_nemotron_h

        return getattr(modeling_nemotron_h, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
