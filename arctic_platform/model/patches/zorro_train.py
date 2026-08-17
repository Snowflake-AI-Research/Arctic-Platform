# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ZoRRo Train forward patch (RL deduplication/reconstruction)."""

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


@register_patch("zorro_train")
def apply_zorro_train(model: nn.Module, ctx: LoaderContext) -> None:
    # Lazy: keep `import arctic_platform.model` free of RL deps.
    from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelOncePatcher
    from arctic_platform.rl.zorro_train.qwen_model_patcher import get_supported_model_type

    # Fail fast: ZoRRo Train only supports Qwen3-family model_type values.
    get_supported_model_type(model)

    settings = ctx.spec.patches.zorro_train

    patcher = Qwen3ModelOncePatcher(
        model,
        response_len=settings.response_len,
        max_token_len=settings.max_token_len,
        rollout_n=settings.rollout_n,
        temperature=settings.temperature,
        logits_optimization=settings.logits_optimization,
        logits_optimization_peak_mem_size_in_gib=settings.logits_optimization_peak_mem_size_in_gib,
        logits_compute_from_fp32_inputs=settings.logits_compute_from_fp32_inputs,
        logits_compute_in_fp32=settings.logits_compute_in_fp32,
        use_unpad=settings.use_unpad,
        world_size=settings.world_size,
    )
    patcher.patch_forward()
    # Pin patcher so attention sub-patcher / closures are not GC'd.
    model._arctic_zorro_once_patcher = patcher
