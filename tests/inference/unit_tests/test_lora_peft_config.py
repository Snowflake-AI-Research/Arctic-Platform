import subprocess
import sys

import pytest

from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.debug.lora import maybe_initialize_lora_factors
from arctic_platform.debug.lora import maybe_prepare_lora_initialization
from arctic_platform.debug.lora import maybe_validate_lora_parallelism


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
        normalize_lora_peft_config({"peft_type": "Lora", "target_modules": ["q_proj"], "bias": "all"})


def test_disabled_lora_debug_helpers_leave_production_objects_untouched():
    config = {"peft_type": "Lora", "target_modules": ["q_proj"]}
    model = object()
    adapter_config, initialization_config = maybe_prepare_lora_initialization(model, config)
    assert adapter_config is config
    assert initialization_config is None
    assert maybe_initialize_lora_factors(model, initialization_config) == 0
    maybe_validate_lora_parallelism(config, object())
    maybe_validate_lora_parallelism(None, object())


def test_lora_debug_config_imports_without_training_dependencies():
    code = """
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "arctic_platform.model" or fullname.split(".")[0] in {"torch", "peft", "transformers", "deepspeed"}:
            raise AssertionError(f"unexpected training import: {fullname}")

sys.meta_path.insert(0, BlockTrainingImports())
from arctic_platform.common.peft import normalize_lora_peft_config
config = normalize_lora_peft_config({
    "peft_type": "Lora",
    "target_modules": ["q_proj"],
    "deterministic_init": {
        "recipe_version": "arctic-correctness-lora-init-v1",
        "global_seed": 42,
        "dtype": "float32",
    },
})
assert config["deterministic_init"]["global_seed"] == 42
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30)
