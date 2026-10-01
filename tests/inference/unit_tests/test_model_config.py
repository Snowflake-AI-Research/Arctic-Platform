import importlib.util
import sys
from pathlib import Path

import pytest


def _model_config_cls():
    path = Path(__file__).parents[3] / "inference" / "arctic_inference" / "server" / "config.py"
    spec = importlib.util.spec_from_file_location("_test_arctic_server_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ModelConfig


def test_model_config_accepts_gdn_prefill_backend():
    ModelConfig = _model_config_cls()
    config = ModelConfig(model="m", gdn_prefill_backend="triton")

    assert config.gdn_prefill_backend == "triton"
    assert config.to_engine_kwargs()["gdn_prefill_backend"] == "triton"


def test_model_config_preserves_structured_outputs_config():
    ModelConfig = _model_config_cls()
    config = ModelConfig(
        model="m",
        structured_outputs_config={"enable_in_reasoning": True},
    )

    assert config.to_engine_kwargs()["structured_outputs_config"] == {
        "enable_in_reasoning": True,
    }


def test_model_config_preserves_speculative_config():
    ModelConfig = _model_config_cls()
    speculative_config = {
        "method": "dflash",
        "model": "z-lab/Qwen3.8-27B-DFlash2",
        "num_speculative_tokens": 7,
    }
    config = ModelConfig(model="m", speculative_config=speculative_config)

    assert config.to_engine_kwargs()["speculative_config"] == speculative_config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ulysses_sequence_parallel_size": 2},
        {
            "extra_engine_kwargs": {
                "ulysses_sequence_parallel_size": 2,
            }
        },
        {
            "extra_engine_kwargs": {
                "ulysses_sequence_parallel_size": 1.5,
            }
        },
    ],
)
def test_model_config_rejects_sequence_parallelism(kwargs):
    ModelConfig = _model_config_cls()

    with pytest.raises(ValueError, match="sequence parallelism must be 1"):
        ModelConfig(model="m", **kwargs)


def test_model_config_accepts_sequence_parallelism_one_without_forwarding_field():
    ModelConfig = _model_config_cls()
    config = ModelConfig(model="m", ulysses_sequence_parallel_size=1)

    assert "ulysses_sequence_parallel_size" not in config.to_engine_kwargs()


def test_model_config_preserves_distributed_executor_backend():
    ModelConfig = _model_config_cls()
    config = ModelConfig(model="m", distributed_executor_backend="mp")

    assert config.to_engine_kwargs()["distributed_executor_backend"] == "mp"


@pytest.mark.parametrize("staging", ["cpu", "gpu"])
def test_model_config_accepts_lora_sync_staging_without_forwarding(staging):
    ModelConfig = _model_config_cls()
    config = ModelConfig(model="m", lora_sync_staging=staging)

    assert config.lora_sync_staging == staging
    assert "lora_sync_staging" not in config.to_engine_kwargs()


def test_model_config_rejects_invalid_lora_sync_staging():
    ModelConfig = _model_config_cls()

    with pytest.raises(ValueError, match="lora_sync_staging"):
        ModelConfig(model="m", lora_sync_staging="auto")
