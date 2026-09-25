def validate_lm_head_fused_ce_config(prl_config: dict) -> None:
    """Reject ``fused_cross_entropy`` combined with ``fused_lm_head_token_chunk_size``.

    Both settings select an lm_head implementation and only one head can be installed. ``fp32_lm_head`` selects
    the arithmetic inside whichever head is installed, so it combines with either.
    """
    fused_cross_entropy = prl_config.get("fused_cross_entropy", "liger")
    if fused_cross_entropy and isinstance(prl_config.get("fused_lm_head_token_chunk_size"), int):
        raise ValueError(
            "PrimeRL MoE DSS config cannot combine fused_cross_entropy with fused_lm_head_token_chunk_size."
        )
