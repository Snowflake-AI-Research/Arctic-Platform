# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from vllm.models.glm5next import Glm5NextForConditionalGeneration


class Glm5NextLoRAForConditionalGeneration(Glm5NextForConditionalGeneration):
    packed_modules_mapping = {
        **Glm5NextForConditionalGeneration.packed_modules_mapping,
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
