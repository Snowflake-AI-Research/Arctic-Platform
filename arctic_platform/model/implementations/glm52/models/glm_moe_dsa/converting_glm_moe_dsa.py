import torch
from torch import Tensor

from arctic_platform.model.implementations.glm52.models.fp8 import quantize_to_fp8_blockwise


def _scale_key_candidates(weight_key: str) -> list[str]:
    stems = [weight_key]
    if weight_key.endswith(".weight"):
        stems.append(weight_key[: -len(".weight")])
    candidates: list[str] = []
    for stem in stems:
        candidates.append(f"{stem}.weight_scale_inv")
        candidates.append(f"{stem}_scale_inv")
    return candidates


def _get_scale(state_dict: dict[str, Tensor], weight_key: str) -> Tensor | None:
    """Return the HF finegrained-FP8 scale sitting next to ``weight_key``."""
    for key in _scale_key_candidates(weight_key):
        if key in state_dict:
            return state_dict[key]
    return None


def _pop_scale(state_dict: dict[str, Tensor], weight_key: str) -> Tensor | None:
    """Pop the HF finegrained-FP8 scale sitting next to ``weight_key``."""
    for key in _scale_key_candidates(weight_key):
        if key in state_dict:
            return state_dict.pop(key)
    return None


def _cat_fp8_with_scales(
    weights: list[Tensor],
    scales: list[Tensor | None],
    dim: int,
) -> tuple[Tensor, Tensor | None]:
    """Concatenate native-FP8 weights and the matching block scales on the same axis."""
    fused = torch.cat(weights, dim=dim)
    if fused.dtype != torch.float8_e4m3fn:
        return fused, None
    if any(scale is None for scale in scales):
        missing = sum(scale is None for scale in scales)
        raise ValueError(
            f"native FP8 concat is missing {missing} weight_scale_inv tensor(s) "
            f"for shapes {[tuple(w.shape) for w in weights]}"
        )
    return fused, torch.cat(scales, dim=dim)  # type: ignore[arg-type]


def _split_fused_scale(scale: Tensor, moe_dim: int, fused_dim: int) -> tuple[Tensor, Tensor]:
    """Split a fused gate_up scale along the intermediate (out) axis for w1/w3.

    The tile size comes from how many row blocks the checkpoint used for
    ``fused_dim`` rather than a fixed 128, so a checkpoint quantized at another
    ``weight_block_size`` fails instead of mis-slicing gate into up.
    """
    if scale.ndim not in (2, 3):
        raise ValueError(f"Unexpected fused FP8 scale ndim={scale.ndim} shape={tuple(scale.shape)}")
    n_row_blocks = scale.shape[-2]
    block_size, remainder = divmod(fused_dim, n_row_blocks)
    if remainder or moe_dim % block_size:
        raise ValueError(
            f"fused FP8 gate_up scale with {n_row_blocks} row blocks does not tile "
            f"fused_dim={fused_dim} evenly into gate/up halves at moe_dim={moe_dim}"
        )
    split_at = moe_dim // block_size
    if scale.ndim == 2:
        return scale[:split_at], scale[split_at:]
    return scale[:, :split_at], scale[:, split_at:]


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def _is_moe_layer(state_dict: dict[str, Tensor], layer_idx: int) -> bool:
    """Check if a layer is an MoE layer by looking for the router gate weight."""
    return f"model.layers.{layer_idx}.mlp.gate.weight" in state_dict


def _routed_expert_indices(state_dict: dict[str, Tensor], layer_idx: int) -> list[int]:
    """Return routed-expert ids for one layer.

    Keys must start with ``model.layers.{idx}.mlp.experts.{{id}}.`` so layer 3
    does not also match layer 13 / 23 / 30. Unique ids are used rather than
    ``len(keys) // 3`` because native FP8 checkpoints also store
    ``weight_scale_inv``.
    """
    prefix = f"model.layers.{layer_idx}.mlp.experts."
    ids: set[int] = set()
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        head = key[len(prefix) :].split(".", 1)[0]
        if head.isdigit():
            ids.add(int(head))
    return sorted(ids)


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    i = layer_idx

    if not _is_moe_layer(state_dict, i):
        return

    # Router: gate.weight -> router.gate.weight
    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
    del state_dict[f"model.layers.{i}.mlp.gate.weight"]

    # Routed experts: fused or per-expert format -> stacked w1/w2/w3
    fused_key = f"model.layers.{i}.mlp.experts.gate_up_proj"
    fused_weight_key = f"{fused_key}.weight"
    if fused_key in state_dict or fused_weight_key in state_dict:
        weight_key = fused_weight_key if fused_weight_key in state_dict else fused_key
        down_key = (
            f"model.layers.{i}.mlp.experts.down_proj.weight"
            if f"model.layers.{i}.mlp.experts.down_proj.weight" in state_dict
            else f"model.layers.{i}.mlp.experts.down_proj"
        )
        gate_up_proj = state_dict.pop(weight_key)
        down_proj = state_dict.pop(down_key)
        gate_up_scale = _pop_scale(state_dict, weight_key)
        down_scale = _pop_scale(state_dict, down_key)

        num_experts, fused_dim, dim = gate_up_proj.shape
        moe_dim = fused_dim // 2

        w1 = gate_up_proj[:, :moe_dim, :]
        w3 = gate_up_proj[:, moe_dim:, :]
        w2 = down_proj
        if gate_up_scale is not None:
            w1_scale, w3_scale = _split_fused_scale(gate_up_scale, moe_dim, fused_dim)
            state_dict[f"model.layers.{i}.mlp.experts.w1_scale_inv"] = w1_scale
            state_dict[f"model.layers.{i}.mlp.experts.w3_scale_inv"] = w3_scale
        if down_scale is not None:
            state_dict[f"model.layers.{i}.mlp.experts.w2_scale_inv"] = down_scale
    else:
        expert_ids = _routed_expert_indices(state_dict, i)
        if not expert_ids:
            return
        if expert_ids != list(range(len(expert_ids))):
            raise KeyError(
                f"Non-contiguous routed experts on layer {i}: {expert_ids[:8]}...{expert_ids[-3:]}"
            )
        num_experts = len(expert_ids)

        dim, moe_dim = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].shape
        dtype = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        w1_scales: list[Tensor] = []
        w2_scales: list[Tensor] = []
        w3_scales: list[Tensor] = []
        for j in range(num_experts):
            g_key = f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"
            d_key = f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"
            u_key = f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"
            w1[j].copy_(state_dict.pop(g_key))
            w2[j].copy_(state_dict.pop(d_key))
            w3[j].copy_(state_dict.pop(u_key))
            g_scale = _pop_scale(state_dict, g_key)
            d_scale = _pop_scale(state_dict, d_key)
            u_scale = _pop_scale(state_dict, u_key)
            if g_scale is not None:
                w1_scales.append(g_scale)
            if d_scale is not None:
                w2_scales.append(d_scale)
            if u_scale is not None:
                w3_scales.append(u_scale)
        if w1_scales or w2_scales or w3_scales:
            if len(w1_scales) != num_experts or len(w2_scales) != num_experts or len(w3_scales) != num_experts:
                raise ValueError(
                    f"layer {i}: incomplete FP8 expert scales "
                    f"(w1={len(w1_scales)} w2={len(w2_scales)} w3={len(w3_scales)} experts={num_experts})"
                )
            state_dict[f"model.layers.{i}.mlp.experts.w1_scale_inv"] = torch.stack(w1_scales)
            state_dict[f"model.layers.{i}.mlp.experts.w2_scale_inv"] = torch.stack(w2_scales)
            state_dict[f"model.layers.{i}.mlp.experts.w3_scale_inv"] = torch.stack(w3_scales)

    state_dict[f"model.layers.{i}.mlp.experts.w1"] = w1
    state_dict[f"model.layers.{i}.mlp.experts.w2"] = w2
    state_dict[f"model.layers.{i}.mlp.experts.w3"] = w3

    # Shared experts
    shared_g = f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
    shared_d = f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
    shared_u = f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
    state_dict[f"model.layers.{i}.mlp.shared_expert.w1"] = state_dict.pop(shared_g)
    state_dict[f"model.layers.{i}.mlp.shared_expert.w2"] = state_dict.pop(shared_d)
    state_dict[f"model.layers.{i}.mlp.shared_expert.w3"] = state_dict.pop(shared_u)
    shared_g_scale = _pop_scale(state_dict, shared_g)
    shared_d_scale = _pop_scale(state_dict, shared_d)
    shared_u_scale = _pop_scale(state_dict, shared_u)
    if shared_g_scale is not None or shared_d_scale is not None or shared_u_scale is not None:
        if shared_g_scale is None or shared_d_scale is None or shared_u_scale is None:
            raise ValueError(f"layer {i}: incomplete FP8 shared-expert scales")
        state_dict[f"model.layers.{i}.mlp.shared_expert.w1_scale_inv"] = shared_g_scale
        state_dict[f"model.layers.{i}.mlp.shared_expert.w2_scale_inv"] = shared_d_scale
        state_dict[f"model.layers.{i}.mlp.shared_expert.w3_scale_inv"] = shared_u_scale

    # Expert bias for load balancing
    state_dict[f"model.layers.{i}.mlp.expert_bias"] = state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]
    del state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    i = layer_index

    # Expert bias
    if f"model.layers.{i}.mlp.expert_bias" in state_dict:
        state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = state_dict[
            f"model.layers.{i}.mlp.expert_bias"
        ]
        del state_dict[f"model.layers.{i}.mlp.expert_bias"]
    if f"model.layers.{i}.mlp.tokens_per_expert" in state_dict:
        del state_dict[f"model.layers.{i}.mlp.tokens_per_expert"]

    # Shared experts
    if f"model.layers.{i}.mlp.shared_expert.w1" in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w1"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w2"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w3"
        ]
        sw1_scale = state_dict.pop(f"model.layers.{i}.mlp.shared_expert.w1_scale_inv", None)
        sw2_scale = state_dict.pop(f"model.layers.{i}.mlp.shared_expert.w2_scale_inv", None)
        sw3_scale = state_dict.pop(f"model.layers.{i}.mlp.shared_expert.w3_scale_inv", None)

        if state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"].shape[0] == 1:
            state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
            ][0]
            if sw1_scale is not None and sw1_scale.shape[0] == 1:
                sw1_scale = sw1_scale[0]
                sw2_scale = sw2_scale[0] if sw2_scale is not None else None
                sw3_scale = sw3_scale[0] if sw3_scale is not None else None
        if sw1_scale is not None or sw2_scale is not None or sw3_scale is not None:
            if sw1_scale is None or sw2_scale is None or sw3_scale is None:
                raise ValueError(f"layer {i}: incomplete FP8 shared-expert scales on HF export")
            state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight_scale_inv"] = sw1_scale
            state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight_scale_inv"] = sw2_scale
            state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight_scale_inv"] = sw3_scale
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w1"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w2"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w3"]

    # Router
    if f"model.layers.{i}.mlp.router.gate.weight" in state_dict:
        state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict[f"model.layers.{i}.mlp.router.gate.weight"]
        del state_dict[f"model.layers.{i}.mlp.router.gate.weight"]

        # Routed experts - convert to per-expert format (compatible with vLLM and transformers)
        w1 = state_dict.pop(f"model.layers.{i}.mlp.experts.w1")  # (num_experts, moe_dim, dim)
        w2 = state_dict.pop(f"model.layers.{i}.mlp.experts.w2")  # (num_experts, dim, moe_dim)
        w3 = state_dict.pop(f"model.layers.{i}.mlp.experts.w3")  # (num_experts, moe_dim, dim)
        w1_scale = state_dict.pop(f"model.layers.{i}.mlp.experts.w1_scale_inv", None)
        w2_scale = state_dict.pop(f"model.layers.{i}.mlp.experts.w2_scale_inv", None)
        w3_scale = state_dict.pop(f"model.layers.{i}.mlp.experts.w3_scale_inv", None)

        num_experts = w1.shape[0]
        for j in range(num_experts):
            state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = w1[j]
            state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = w2[j]
            state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = w3[j]
            if w1_scale is not None:
                if w2_scale is None or w3_scale is None:
                    raise ValueError(f"layer {i}: incomplete FP8 expert scales on HF export")
                state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight_scale_inv"] = w1_scale[j]
                state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight_scale_inv"] = w2_scale[j]
                state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight_scale_inv"] = w3_scale[j]


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)


def convert_tt_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    """Convert a single GLM layer from PrimeRL format to vLLM kernel format."""
    out: dict[str, Tensor] = {}
    prefix = f"model.layers.{layer_idx}"

    def add(name: str, tensor: Tensor) -> None:
        out[name] = tensor

    def add_maybe_fp8(name: str, tensor: Tensor, scale: Tensor | None = None) -> None:
        if tensor.dtype == torch.float8_e4m3fn:
            if scale is None:
                raise ValueError(f"native FP8 tensor {name} is missing weight_scale_inv")
            out[name] = tensor
            out[name.removesuffix(".weight") + ".weight_scale_inv"] = scale
            return
        if quantize_fp8 and tensor.ndim == 2:
            fp8_weight, scale = quantize_to_fp8_blockwise(tensor)
            out[name] = fp8_weight
            scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
            out[scale_name] = scale
            return
        out[name] = tensor

    for name in [f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"]:
        if name in state_dict:
            add(name, state_dict[name])

    q_a_key = f"{prefix}.self_attn.q_a_proj.weight"
    kv_a_key = f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"
    if q_a_key in state_dict and kv_a_key in state_dict:
        fused_qkv, fused_qkv_scale = _cat_fp8_with_scales(
            [state_dict[q_a_key], state_dict[kv_a_key]],
            [_get_scale(state_dict, q_a_key), _get_scale(state_dict, kv_a_key)],
            dim=0,
        )
        add_maybe_fp8(f"{prefix}.self_attn.fused_qkv_a_proj.weight", fused_qkv, fused_qkv_scale)

    for suffix in ["q_a_layernorm.weight", "kv_a_layernorm.weight"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    for suffix in ["q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key], _get_scale(state_dict, key))

    for suffix in ["indexer.wq_b.weight", "indexer.wk.weight", "indexer.weights_proj.weight"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add_maybe_fp8(key, state_dict[key], _get_scale(state_dict, key))
    for suffix in ["indexer.k_norm.weight", "indexer.k_norm.bias"]:
        key = f"{prefix}.self_attn.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    gate_key = f"{prefix}.mlp.gate_proj.weight"
    up_key = f"{prefix}.mlp.up_proj.weight"
    down_key = f"{prefix}.mlp.down_proj.weight"
    if gate_key in state_dict and up_key in state_dict:
        gate_up, gate_up_scale = _cat_fp8_with_scales(
            [state_dict[gate_key], state_dict[up_key]],
            [_get_scale(state_dict, gate_key), _get_scale(state_dict, up_key)],
            dim=0,
        )
        add_maybe_fp8(f"{prefix}.mlp.gate_up_proj.weight", gate_up, gate_up_scale)
        if down_key in state_dict:
            add_maybe_fp8(down_key, state_dict[down_key], _get_scale(state_dict, down_key))

    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key in state_dict:
        add(f"{prefix}.mlp.gate.weight", state_dict[router_key])
    expert_bias_key = f"{prefix}.mlp.expert_bias"
    if expert_bias_key in state_dict:
        add(f"{prefix}.mlp.gate.e_score_correction_bias", state_dict[expert_bias_key])

    w1_key = f"{prefix}.mlp.experts.w1"
    w2_key = f"{prefix}.mlp.experts.w2"
    w3_key = f"{prefix}.mlp.experts.w3"
    if w1_key in state_dict and w2_key in state_dict and w3_key in state_dict:
        w1 = state_dict[w1_key]
        w2 = state_dict[w2_key]
        w3 = state_dict[w3_key]
        w13 = torch.cat([w1, w3], dim=1)
        w1_scale = state_dict.get(f"{prefix}.mlp.experts.w1_scale_inv")
        w2_scale = state_dict.get(f"{prefix}.mlp.experts.w2_scale_inv")
        w3_scale = state_dict.get(f"{prefix}.mlp.experts.w3_scale_inv")

        if w1.dtype == torch.float8_e4m3fn:
            if w1_scale is None or w3_scale is None or w2_scale is None:
                raise ValueError(
                    f"{prefix}: native FP8 experts require w1/w2/w3 weight_scale_inv"
                )
            out[f"{prefix}.mlp.experts.w13_weight"] = w13
            out[f"{prefix}.mlp.experts.w2_weight"] = w2
            out[f"{prefix}.mlp.experts.w13_weight_scale_inv"] = torch.cat([w1_scale, w3_scale], dim=1)
            out[f"{prefix}.mlp.experts.w2_weight_scale_inv"] = w2_scale
        elif quantize_fp8:
            w13_fp8: list[Tensor] = []
            w13_scales: list[Tensor] = []
            w2_fp8: list[Tensor] = []
            w2_scales: list[Tensor] = []
            for expert_idx in range(w1.shape[0]):
                expert_w13_fp8, expert_w13_scales = quantize_to_fp8_blockwise(w13[expert_idx])
                expert_w2_fp8, expert_w2_scales = quantize_to_fp8_blockwise(w2[expert_idx])
                w13_fp8.append(expert_w13_fp8)
                w13_scales.append(expert_w13_scales)
                w2_fp8.append(expert_w2_fp8)
                w2_scales.append(expert_w2_scales)

            out[f"{prefix}.mlp.experts.w13_weight"] = torch.stack(w13_fp8)
            out[f"{prefix}.mlp.experts.w13_weight_scale_inv"] = torch.stack(w13_scales)
            out[f"{prefix}.mlp.experts.w2_weight"] = torch.stack(w2_fp8)
            out[f"{prefix}.mlp.experts.w2_weight_scale_inv"] = torch.stack(w2_scales)
        else:
            out[f"{prefix}.mlp.experts.w13_weight"] = w13
            out[f"{prefix}.mlp.experts.w2_weight"] = w2

    sw1_key = f"{prefix}.mlp.shared_expert.w1"
    sw2_key = f"{prefix}.mlp.shared_expert.w2"
    sw3_key = f"{prefix}.mlp.shared_expert.w3"
    if sw1_key in state_dict and sw2_key in state_dict and sw3_key in state_dict:
        sw1 = state_dict[sw1_key]
        sw2 = state_dict[sw2_key]
        sw3 = state_dict[sw3_key]
        if sw1.ndim == 3:
            sw1 = sw1.squeeze(0)
            sw2 = sw2.squeeze(0)
            sw3 = sw3.squeeze(0)
        sw1_scale = state_dict.get(f"{prefix}.mlp.shared_expert.w1_scale_inv")
        sw2_scale = state_dict.get(f"{prefix}.mlp.shared_expert.w2_scale_inv")
        sw3_scale = state_dict.get(f"{prefix}.mlp.shared_expert.w3_scale_inv")
        if sw1_scale is not None and sw1_scale.ndim == 3:
            sw1_scale = sw1_scale.squeeze(0)
            sw2_scale = None if sw2_scale is None else sw2_scale.squeeze(0)
            sw3_scale = None if sw3_scale is None else sw3_scale.squeeze(0)
        gate_up, gate_up_scale = _cat_fp8_with_scales([sw1, sw3], [sw1_scale, sw3_scale], dim=0)
        add_maybe_fp8(f"{prefix}.mlp.shared_experts.gate_up_proj.weight", gate_up, gate_up_scale)
        add_maybe_fp8(f"{prefix}.mlp.shared_experts.down_proj.weight", sw2, sw2_scale)

    return out
