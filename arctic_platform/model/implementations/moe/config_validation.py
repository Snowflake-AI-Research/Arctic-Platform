from collections.abc import Mapping
from typing import Any


def validate_lm_head_fused_ce_config(options: Mapping[str, Any]) -> None:
    """Reject ``fused_cross_entropy`` combined with ``fused_lm_head_token_chunk_size``.

    Both settings select an lm_head implementation and only one head can be installed. ``fp32_lm_head`` selects
    the arithmetic inside whichever head is installed, so it combines with either.
    """
    fused_cross_entropy = options.get("fused_cross_entropy", "liger")
    if fused_cross_entropy and isinstance(options.get("fused_lm_head_token_chunk_size"), int):
        raise ValueError(
            "qwen3_5_moe cannot combine fused_cross_entropy with fused_lm_head_token_chunk_size."
        )
