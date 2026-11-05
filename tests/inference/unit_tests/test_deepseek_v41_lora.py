import sys
from types import ModuleType

import pytest

from arctic_platform.inference.vllm import deepseek_v41_lora


class FakeRegistry:
    def __init__(self, archs):
        self.archs = set(archs)
        self.registered = {}

    def get_supported_archs(self):
        return self.archs

    def register_model(self, arch, target):
        self.registered[arch] = target


@pytest.fixture
def registry(monkeypatch, request):
    fake = FakeRegistry(request.param)
    vllm = ModuleType("vllm")
    vllm.ModelRegistry = fake
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(deepseek_v41_lora, "_REGISTERED", False)
    return fake


@pytest.mark.parametrize("registry", [["DeepseekV41ForCausalLM"]], indirect=True)
def test_registers_lora_wrapper_once(registry):
    deepseek_v41_lora.ensure_deepseek_v41_lora()
    registry.registered.clear()
    deepseek_v41_lora.ensure_deepseek_v41_lora()

    assert registry.registered == {}


@pytest.mark.parametrize("registry", [["DeepseekV41ForCausalLM"]], indirect=True)
def test_overrides_vllm_v41_architecture(registry):
    deepseek_v41_lora.ensure_deepseek_v41_lora()

    assert registry.registered == {
        "DeepseekV41ForCausalLM": (
            "arctic_platform.inference.vllm.deepseek_v41_lora_model:DeepseekV41LoRAForCausalLM"
        )
    }


@pytest.mark.parametrize("registry", [["DeepseekV4ForCausalLM"]], indirect=True)
def test_skips_vllm_without_v41(registry):
    deepseek_v41_lora.ensure_deepseek_v41_lora()

    assert registry.registered == {}
