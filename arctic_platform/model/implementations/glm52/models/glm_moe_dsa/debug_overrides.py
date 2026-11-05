"""Apply trainer debug overrides to GLM MoE DSA (GLM-5 / GLM-5.2) configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arctic_platform.model.implementations.glm52.config import DebugModelConfig


def _round_down_to(n: int, multiple: int) -> int:
    return max(multiple, (n // multiple) * multiple)


def _scale_mla_dims_for_hidden(target, new_hidden: int, old_hidden: int) -> None:
    """Scale MLA / indexer dims when hidden_size shrinks (keeps o_proj geometry sane).

    kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, and index_head_dim are
    left unchanged: TileLang sparse MLA kernels hardcode dim+tail=576 (=512+64).
    """
    if new_hidden >= old_hidden:
        return
    ratio = new_hidden / old_hidden
    target.num_attention_heads = _round_down_to(int(target.num_attention_heads * ratio), 4)
    target.q_lora_rank = _round_down_to(int(target.q_lora_rank * ratio), 64)
    target.index_n_heads = _round_down_to(int(target.index_n_heads * ratio), 4)


def _rebuild_mlp_layer_types(target) -> None:
    n = target.num_hidden_layers
    k = target.first_k_dense_replace
    target.mlp_layer_types = ["dense"] * min(k, n) + ["sparse"] * max(0, n - k)


def _extend_indexer_types(target, new_layers: int) -> None:
    types = list(getattr(target, "indexer_types", None) or [])
    if types:
        if new_layers <= len(types):
            target.indexer_types = types[:new_layers]
        else:
            # Cycle the tail pattern when going deeper than the HF config.
            tail = types[-4:] if len(types) >= 4 else types
            while len(types) < new_layers:
                types.append(tail[len(types) % len(tail)])
            target.indexer_types = types

    pattern = getattr(target, "index_topk_pattern", None)
    if isinstance(pattern, str) and pattern:
        if new_layers <= len(pattern):
            target.index_topk_pattern = pattern[:new_layers]
        else:
            tail = pattern[-4:] if len(pattern) >= 4 else pattern
            while len(pattern) < new_layers:
                pattern += tail[len(pattern) % len(tail)]
            target.index_topk_pattern = pattern


def apply_glm_moe_dsa_debug_overrides(target, debug: DebugModelConfig, *, logger) -> None:
    """Mutate a GlmMoeDsaConfig (or text_config) from DebugModelConfig fields."""
    if getattr(target, "model_type", None) not in (None, "glm_moe_dsa"):
        return

    old_hidden = target.hidden_size

    if debug.hidden_size is not None:
        target.hidden_size = debug.hidden_size
    if debug.intermediate_size is not None:
        target.intermediate_size = debug.intermediate_size
    if debug.moe_intermediate_size is not None:
        target.moe_intermediate_size = debug.moe_intermediate_size

    if debug.hidden_size is not None and debug.hidden_size != old_hidden:
        if debug.intermediate_size is None:
            target.intermediate_size = debug.hidden_size * 2
        if debug.moe_intermediate_size is None:
            target.moe_intermediate_size = max(256, int(target.moe_intermediate_size * debug.hidden_size / old_hidden))
        _scale_mla_dims_for_hidden(target, debug.hidden_size, old_hidden)

    if debug.num_layers is not None:
        base_layers = target.num_hidden_layers
        if debug.num_layers > base_layers and not debug.random_init:
            raise ValueError(
                f"debug.num_layers={debug.num_layers} exceeds config depth {base_layers}; "
                "set debug.random_init=true to build a deeper random-init model."
            )
        if debug.num_layers < base_layers and not debug.random_init:
            logger.warning(
                f"Truncating model from {base_layers} to {debug.num_layers} layers "
                f"({base_layers - debug.num_layers} layers will not be loaded from checkpoint)."
            )
        elif debug.num_layers > base_layers:
            logger.warning(
                f"Expanding model from {base_layers} to {debug.num_layers} layers (random_init only)."
            )
        target.num_hidden_layers = debug.num_layers
        _extend_indexer_types(target, debug.num_layers)

    if debug.first_k_dense_replace is not None:
        target.first_k_dense_replace = debug.first_k_dense_replace

    if debug.num_layers is not None or debug.first_k_dense_replace is not None:
        _rebuild_mlp_layer_types(target)

    if debug.hidden_size is not None or debug.intermediate_size is not None:
        logger.warning(
            "GLM debug shape override: "
            f"hidden={target.hidden_size}, intermediate={target.intermediate_size}, "
            f"moe_intermediate={target.moe_intermediate_size}, layers={target.num_hidden_layers}, "
            f"heads={target.num_attention_heads}, q_lora={target.q_lora_rank}, kv_lora={target.kv_lora_rank}"
        )


def estimate_glm_moe_dsa_params(config) -> int:
    """Rough parameter count for logging (MoE-dominated)."""
    h = config.hidden_size
    n = config.num_hidden_layers
    k_dense = min(config.first_k_dense_replace, n)
    k_moe = n - k_dense
    vocab = config.vocab_size
    embed = vocab * h * 2  # embed + lm_head
    dense_mlp = k_dense * (3 * h * config.intermediate_size + config.intermediate_size * h)
    experts = config.n_routed_experts + config.n_shared_experts
    moe_per_layer = 3 * h * config.moe_intermediate_size * experts + h * config.moe_intermediate_size * experts
    moe = k_moe * moe_per_layer
    # MLA is small vs MoE; add ~1% fudge
    return int(embed + dense_mlp + moe * 1.01)
