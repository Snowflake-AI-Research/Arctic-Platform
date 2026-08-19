def validate_lm_head_fused_ce_config(prl_config: dict) -> None:
    fused_cross_entropy = prl_config.get("fused_cross_entropy", "liger")
    if fused_cross_entropy and (
        prl_config.get("fp32_lm_head", False)
        or isinstance(prl_config.get("fused_lm_head_token_chunk_size"), int)
    ):
        raise ValueError(
            "PrimeRL MoE DSS config cannot combine fused_cross_entropy with "
            "fp32_lm_head or fused_lm_head_token_chunk_size."
        )
