"""PrimeRL Nemotron-H → vLLM ``/sync-weights`` packing."""

from __future__ import annotations

from arctic_platform.model.implementations.moe.vllm_weights import raise_unmapped_mlp_keys


def _rename_prefix(state_dict: dict, old_prefix: str, new_prefix: str) -> None:
    for key in [k for k in state_dict if k.startswith(old_prefix)]:
        state_dict[new_prefix + key[len(old_prefix) :]] = state_dict.pop(key)


def convert_nemotron_h_layer_to_vllm(
    layer_sd: dict,
    layer_idx: int,
    *,
    layer_prefix: str | None = None,
) -> dict:
    """PrimeRL Nemotron-H names → vLLM runtime views (mixer, packed experts, A)."""
    if layer_idx < 0:
        if "model.norm.weight" in layer_sd:
            layer_sd["model.norm_f.weight"] = layer_sd.pop("model.norm.weight")
        return layer_sd

    prefix = layer_prefix if layer_prefix is not None else f"model.layers.{layer_idx}"

    if any(key.startswith(f"{prefix}.mamba.") for key in layer_sd):
        _rename_prefix(layer_sd, f"{prefix}.mamba.", f"{prefix}.mixer.")
        a_log = f"{prefix}.mixer.A_log"
        if a_log in layer_sd:
            layer_sd[f"{prefix}.mixer.A"] = layer_sd.pop(a_log)
        return layer_sd

    if any(key.startswith(f"{prefix}.self_attn.") for key in layer_sd):
        _rename_prefix(layer_sd, f"{prefix}.self_attn.", f"{prefix}.mixer.")
        return layer_sd

    if not any(key.startswith(f"{prefix}.mlp.") for key in layer_sd):
        return layer_sd

    router_key = f"{prefix}.mlp.router.gate"
    if router_key in layer_sd:
        layer_sd[f"{prefix}.mixer.gate.weight"] = layer_sd.pop(router_key)
    bias_key = f"{prefix}.mlp.router.e_score_correction_bias"
    if bias_key in layer_sd:
        layer_sd[f"{prefix}.mixer.gate.e_score_correction_bias"] = layer_sd.pop(bias_key)

    w1_key = f"{prefix}.mlp.experts.w1"
    w2_key = f"{prefix}.mlp.experts.w2"
    if w1_key in layer_sd:
        layer_sd[f"{prefix}.mixer.experts.w13_weight"] = layer_sd.pop(w1_key)
    if w2_key in layer_sd:
        layer_sd[f"{prefix}.mixer.experts.w2_weight"] = layer_sd.pop(w2_key)
    layer_sd.pop(f"{prefix}.mlp.experts.w3", None)

    _rename_prefix(
        layer_sd, f"{prefix}.mlp.shared_expert.", f"{prefix}.mixer.shared_experts."
    )
    _rename_prefix(
        layer_sd, f"{prefix}.mlp.fc1_latent_proj.", f"{prefix}.mixer.fc1_latent_proj."
    )
    _rename_prefix(
        layer_sd, f"{prefix}.mlp.fc2_latent_proj.", f"{prefix}.mixer.fc2_latent_proj."
    )

    for buf_key in (f"{prefix}.mlp.expert_bias", f"{prefix}.mlp.tokens_per_expert"):
        layer_sd.pop(buf_key, None)

    leftover = [key for key in layer_sd if key.startswith(f"{prefix}.mlp.")]
    raise_unmapped_mlp_keys("Nemotron-H", leftover)
    return layer_sd
