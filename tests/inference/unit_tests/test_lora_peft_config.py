import pytest

from arctic_platform.inference.server.weight_sync.peft import (
    normalize_lora_peft_config,
)


def test_normalize_lora_peft_config_fills_defaults():
    config = normalize_lora_peft_config(
        {"peft_type": "Lora", "target_modules": ["q_proj"]},
        location="lora_config",
    )
    assert config == {
        "peft_type": "Lora",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj"],
        "target_parameters": [],
    }


def test_normalize_lora_peft_config_rejects_unsupported_bias():
    with pytest.raises(ValueError, match="bias='all'"):
        normalize_lora_peft_config(
            {"peft_type": "Lora", "target_modules": ["q_proj"], "bias": "all"}
        )
