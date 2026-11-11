from types import SimpleNamespace

import pytest


def _pack_fused_expert_loras():
    from arctic_platform.inference.server.weight_sync.receiver import (
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


def _stacked_lora(torch, name, num_experts, rank, in_dim, out_dim):
    from vllm.lora.lora_weights import LoRALayerWeights

    return LoRALayerWeights(
        name,
        rank,
        16,
        torch.ones(num_experts, rank, in_dim),
        torch.ones(num_experts, out_dim, rank),
    )


def test_pack_fused_expert_loras_non_gated_reuses_w1_for_the_third_slot():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")

    pack = _pack_fused_expert_loras()
    num_experts, rank, hidden, intermediate = 4, 8, 16, 32
    up = _stacked_lora(torch, "model.layers.1.mlp.experts.w1", num_experts, rank, hidden, intermediate)
    down = _stacked_lora(torch, "model.layers.1.mlp.experts.w2", num_experts, rank, intermediate, hidden)
    model = SimpleNamespace(loras={up.module_name: up, down.module_name: down})

    assert pack(model, non_gated=True) == 1
    packed = model.loras["model.layers.1.mlp.experts"]
    assert list(model.loras) == ["model.layers.1.mlp.experts"]
    assert packed.lora_a[0] is up.lora_a and packed.lora_b[0] is up.lora_b
    assert packed.lora_a[1] is down.lora_a and packed.lora_b[1] is down.lora_b
    assert packed.lora_a[2] is up.lora_a


def test_pack_fused_expert_loras_gated_still_requires_w3():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")

    pack = _pack_fused_expert_loras()
    up = _stacked_lora(torch, "model.layers.1.mlp.experts.w1", 2, 8, 16, 32)
    down = _stacked_lora(torch, "model.layers.1.mlp.experts.w2", 2, 8, 32, 16)
    model = SimpleNamespace(loras={up.module_name: up, down.module_name: down})
    with pytest.raises(RuntimeError, match=r"missing \['w3'\]"):
        pack(model)


def test_has_non_gated_experts_reads_the_served_moe_layers():
    from arctic_platform.inference.server.weight_sync.receiver import (
        _has_non_gated_experts,
    )

    def served(*gated):
        modules = [("model.embed_tokens", SimpleNamespace())] + [
            (f"model.layers.{i}.mlp.experts", SimpleNamespace(moe_config=SimpleNamespace(is_act_and_mul=g)))
            for i, g in enumerate(gated)
        ]
        return SimpleNamespace(named_modules=lambda: iter(modules))

    assert _has_non_gated_experts(served(False, False)) is True
    assert _has_non_gated_experts(served(True, True)) is False
    assert _has_non_gated_experts(served()) is False
    assert _has_non_gated_experts(None) is False
