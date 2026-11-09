import importlib
import sys
from types import ModuleType
from unittest.mock import Mock


def test_registration_preserves_mapping_and_adds_mla_shards(monkeypatch):
    from arctic_platform.inference.vllm import glm5next_lora

    registry = Mock()
    vllm = ModuleType("vllm")
    vllm.ModelRegistry = registry
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(glm5next_lora, "_REGISTERED", False)

    glm5next_lora.ensure_glm5next_lora()
    glm5next_lora.ensure_glm5next_lora()
    registry.register_model.assert_called_once()
    arch, target = registry.register_model.call_args.args
    assert arch == "Glm5NextForConditionalGeneration"

    class BaseModel:
        packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    upstream = ModuleType("vllm.models.glm5next")
    upstream.Glm5NextForConditionalGeneration = BaseModel
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    module_name, class_name = target.split(":")
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    model = getattr(importlib.import_module(module_name), class_name)
    assert issubclass(model, BaseModel)
    assert model.packed_modules_mapping == {
        **BaseModel.packed_modules_mapping,
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    assert "fused_qkv_a_proj" not in BaseModel.packed_modules_mapping
    monkeypatch.delitem(sys.modules, module_name)
