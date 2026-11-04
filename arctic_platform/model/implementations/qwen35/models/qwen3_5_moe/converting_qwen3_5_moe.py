import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single layer from HF to PrimeRL format in-place.

    Handles:
    - Router: mlp.gate.weight -> mlp.router.gate.weight
    - Routed experts: fused gate_up_proj/down_proj or per-expert -> w1/w2/w3
    - Shared expert: mlp.shared_expert.{gate,up,down}_proj.weight -> shared_expert.{w1,w3,w2}.weight
    - Shared expert gate: mlp.shared_expert_gate.weight -> shared_expert_gate.weight
    """
    i = layer_idx

    # Router: mlp.gate.weight -> mlp.router.gate.weight
    gate_key = f"model.layers.{i}.mlp.gate.weight"
    if gate_key not in state_dict:
        return

    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict.pop(gate_key)

    # Routed experts: convert to fused w1/w2/w3 format
    if f"model.layers.{i}.mlp.experts.gate_up_proj" in state_dict:
        # New fused format (transformers 5.0+): gate_up_proj shape (num_experts, 2*moe_dim, dim)
        gate_up_proj = state_dict.pop(f"model.layers.{i}.mlp.experts.gate_up_proj")
        down_proj = state_dict.pop(f"model.layers.{i}.mlp.experts.down_proj")

        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]  # gate
        w3 = gate_up_proj[:, moe_dim:, :]  # up
        w2 = down_proj  # down
    else:
        # Old per-expert format; block-FP8 checkpoints add a ``weight_scale_inv`` next to every weight.
        prefix = f"model.layers.{i}.mlp.experts."
        num_experts = len([k for k in state_dict.keys() if k.startswith(prefix) and k.endswith(".gate_proj.weight")])
        if num_experts == 0:
            return

        for name, hf_name in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            for suffix, key in (("", "weight"), ("_scale_inv", "weight_scale_inv")):
                if f"{prefix}0.{hf_name}.{key}" not in state_dict:
                    continue
                stacked = torch.stack([state_dict.pop(f"{prefix}{j}.{hf_name}.{key}") for j in range(num_experts)])
                state_dict[f"model.layers.{i}.mlp.experts.{name}{suffix}"] = stacked
        w1 = state_dict.pop(f"model.layers.{i}.mlp.experts.w1")
        w2 = state_dict.pop(f"model.layers.{i}.mlp.experts.w2")
        w3 = state_dict.pop(f"model.layers.{i}.mlp.experts.w3")

    state_dict[f"model.layers.{i}.mlp.experts.w1"] = w1
    state_dict[f"model.layers.{i}.mlp.experts.w2"] = w2
    state_dict[f"model.layers.{i}.mlp.experts.w3"] = w3

    # Shared expert: mlp.shared_expert.{gate,up,down}_proj -> shared_expert.{w1,w3,w2} (with FP8 scales if present)
    se_gate_key = f"model.layers.{i}.mlp.shared_expert.gate_proj.weight"
    if se_gate_key in state_dict:
        for name, hf_name in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            for key in ("weight", "weight_scale_inv"):
                hf_key = f"model.layers.{i}.mlp.shared_expert.{hf_name}.{key}"
                if hf_key in state_dict:
                    state_dict[f"model.layers.{i}.shared_expert.{name}.{key}"] = state_dict.pop(hf_key)

    # Shared expert gate: mlp.shared_expert_gate.weight -> shared_expert_gate.weight
    seg_key = f"model.layers.{i}.mlp.shared_expert_gate.weight"
    if seg_key in state_dict:
        state_dict[f"model.layers.{i}.shared_expert_gate.weight"] = state_dict.pop(seg_key)


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single layer from PrimeRL to HF format in-place."""
    i = layer_idx

    # Router
    router_key = f"model.layers.{i}.mlp.router.gate.weight"
    if router_key not in state_dict:
        return

    state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict.pop(router_key)

    w1 = state_dict.pop(f"model.layers.{i}.mlp.experts.w1")  # gate
    w2 = state_dict.pop(f"model.layers.{i}.mlp.experts.w2")  # down
    w3 = state_dict.pop(f"model.layers.{i}.mlp.experts.w3")  # up
    state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"] = torch.cat([w1, w3], dim=1)
    state_dict[f"model.layers.{i}.mlp.experts.down_proj"] = w2

    # Shared expert: shared_expert.{w1,w2,w3} -> mlp.shared_expert.{gate,down,up}_proj
    se_w1_key = f"model.layers.{i}.shared_expert.w1.weight"
    if se_w1_key in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_expert.gate_proj.weight"] = state_dict.pop(se_w1_key)
        state_dict[f"model.layers.{i}.mlp.shared_expert.down_proj.weight"] = state_dict.pop(
            f"model.layers.{i}.shared_expert.w2.weight"
        )
        state_dict[f"model.layers.{i}.mlp.shared_expert.up_proj.weight"] = state_dict.pop(
            f"model.layers.{i}.shared_expert.w3.weight"
        )

    # Shared expert gate
    seg_key = f"model.layers.{i}.shared_expert_gate.weight"
    if seg_key in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_expert_gate.weight"] = state_dict.pop(seg_key)


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert all MoE weights from HF to PrimeRL format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert all MoE weights from PrimeRL to HF format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
