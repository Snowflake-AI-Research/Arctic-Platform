"""PrimeRL MiniMax M2 → vLLM ``/sync-weights`` packing."""

from __future__ import annotations

import torch

from arctic_platform.model.implementations.moe.vllm_weights import raise_unmapped_mlp_keys


def convert_minimax_m2_layer_to_vllm(
    layer_sd: dict,
    layer_idx: int,
    *,
    layer_prefix: str | None = None,
) -> dict:
    """PrimeRL ``mlp`` stacked experts → vLLM MiniMax ``RoutedExperts`` params.

    Runtime parameters live at ``block_sparse_moe.experts.routed_experts.w13_weight``
    (not ``experts.w13_weight``). ArcticInference-internal #149 aliases the short
    ``load_weights`` name onto that parameter; merge that PR before shipping this
    packer, or MiniMax ``/sync-weights`` fails destination validation.
    """
    if layer_idx < 0:
        return layer_sd
    prefix = layer_prefix if layer_prefix is not None else f"model.layers.{layer_idx}"
    dst = f"{prefix}.block_sparse_moe"
    bias_key = f"{prefix}.mlp.expert_bias"
    if bias_key in layer_sd:
        layer_sd[f"{dst}.e_score_correction_bias"] = layer_sd.pop(bias_key)
    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key in layer_sd:
        layer_sd[f"{dst}.gate.weight"] = layer_sd.pop(router_key)

    def _pop_expert(leaf: str):
        for key in (f"{prefix}.mlp.experts.{leaf}", f"{prefix}.mlp.experts.{leaf}.weight"):
            if key in layer_sd:
                return layer_sd.pop(key)
        return None

    w1 = _pop_expert("w1")
    w2 = _pop_expert("w2")
    w3 = _pop_expert("w3")
    experts = f"{dst}.experts.routed_experts"
    if w1 is not None and w3 is not None:
        layer_sd[f"{experts}.w13_weight"] = torch.cat([w1, w3], dim=1)
    else:
        if w1 is not None:
            layer_sd[f"{prefix}.mlp.experts.w1"] = w1
        if w3 is not None:
            layer_sd[f"{prefix}.mlp.experts.w3"] = w3
    if w2 is not None:
        layer_sd[f"{experts}.w2_weight"] = w2

    layer_sd.pop(f"{prefix}.mlp.tokens_per_expert", None)
    leftover = [key for key in layer_sd if key.startswith(f"{prefix}.mlp.")]
    raise_unmapped_mlp_keys("MiniMax M2", leftover)
    return layer_sd
