from types import SimpleNamespace

import pytest


def _pack_fused_expert_loras():
    from arctic_inference.server.weight_sync.receiver import (
        _pack_fused_expert_loras,
    )

    return _pack_fused_expert_loras


def test_pack_fused_expert_loras_stacks_w1_w2_w3():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.lora.lora_weights import LoRALayerWeights

    pack = _pack_fused_expert_loras()
    num_experts, rank, hidden, intermediate = 4, 8, 16, 32
    loras = {}
    for proj, out_dim in (("w1", intermediate), ("w2", hidden), ("w3", intermediate)):
        name = f"model.layers.0.mlp.experts.{proj}"
        loras[name] = LoRALayerWeights(
            name,
            rank,
            16,
            torch.ones(num_experts, rank, hidden if proj != "w2" else intermediate),
            torch.ones(num_experts, out_dim, rank),
        )
    loras["model.layers.0.self_attn.o_proj"] = LoRALayerWeights(
        "model.layers.0.self_attn.o_proj",
        rank,
        16,
        torch.ones(rank, hidden),
        torch.ones(hidden, rank),
    )
    model = SimpleNamespace(loras=loras)

    assert pack(model) == 1
    assert "model.layers.0.mlp.experts" in model.loras
    assert "model.layers.0.self_attn.o_proj" in model.loras
    for proj in ("w1", "w2", "w3"):
        assert f"model.layers.0.mlp.experts.{proj}" not in model.loras
    packed = model.loras["model.layers.0.mlp.experts"]
    assert packed.is_packed
    assert len(packed.lora_a) == 3
    assert packed.lora_a[0].shape == (num_experts, rank, hidden)
    assert packed.lora_a[1].shape == (num_experts, rank, intermediate)
    assert packed.lora_b[1].shape == (num_experts, hidden, rank)


def test_pack_fused_expert_loras_rejects_incomplete_layer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.lora.lora_weights import LoRALayerWeights

    pack = _pack_fused_expert_loras()
    name = "model.layers.0.mlp.experts.w1"
    model = SimpleNamespace(
        loras={
            name: LoRALayerWeights(
                name,
                8,
                16,
                torch.ones(2, 8, 4),
                torch.ones(2, 4, 8),
            )
        }
    )
    with pytest.raises(RuntimeError, match="missing"):
        pack(model)


def test_pack_fused_expert_loras_noop_without_experts():
    pack = _pack_fused_expert_loras()
    model = SimpleNamespace(loras={"model.layers.0.self_attn.o_proj": object()})
    assert pack(model) == 0
    assert list(model.loras) == ["model.layers.0.self_attn.o_proj"]
