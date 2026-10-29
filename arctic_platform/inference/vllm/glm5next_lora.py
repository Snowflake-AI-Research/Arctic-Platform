# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Register the GLM-5.3-Flash packed MLA adapter mapping."""

_REGISTERED = False


def ensure_glm5next_lora() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration",
        "arctic_platform.inference.vllm.glm5next_lora_model:Glm5NextLoRAForConditionalGeneration",
    )
    _REGISTERED = True
