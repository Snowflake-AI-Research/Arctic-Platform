from collections.abc import Mapping
from typing import Any


def effective_fused_cross_entropy(options: Mapping[str, Any] | None):
    """Return the configured fused-CE backend or the MoE default."""
    return (options or {}).get("fused_cross_entropy", "liger")


def validate_lm_head_fused_ce_config(options: Mapping[str, Any]) -> None:
    """Reject ``fused_cross_entropy`` combined with ``fused_lm_head_token_chunk_size``.

    Both settings select an lm_head implementation and only one head can be installed. ``fp32_lm_head`` selects
    the arithmetic inside whichever head is installed, so it combines with either.
    """
    fused_cross_entropy = effective_fused_cross_entropy(options)
    if fused_cross_entropy and isinstance(options.get("fused_lm_head_token_chunk_size"), int):
        raise ValueError("cannot combine fused_cross_entropy with fused_lm_head_token_chunk_size")


def validate_deepep_v2_recompute(ep_comm_backend: str, ac_config: Any) -> None:
    """Refuse ``deepep_v2`` under whole-block checkpointing without the combine replay.

    DeepEP V2's unordered kernels deliver each expert's rows in arrival order, so a whole-block recompute must
    dispatch with its forward's handle, which the layer's ``CombineReplay`` carries (``distributed/deepep_v2.py``);
    ``ac_config.combine_replay`` installs it. Without it the recompute's dispatch raises, at the first backward.
    """
    if ep_comm_backend != "deepep_v2" or ac_config is None:
        return
    if ac_config.mode == "full" and not ac_config.combine_replay:
        raise ValueError(
            "ep_comm_backend='deepep_v2' recomputes a checkpointed block's dispatch with its forward's handle, "
            "which ac_config.combine_replay carries: set ac_config.combine_replay=true"
        )
