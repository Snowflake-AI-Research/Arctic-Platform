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
"""Causal LM-head precision and chunking patch."""

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


@register_patch("lm_head")
def apply_lm_head(model: nn.Module, ctx: LoaderContext) -> None:
    from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs
    from arctic_platform.model.implementations.gpu.lm_head import enable_fp32_lm_head

    settings = ctx.spec.patches.lm_head
    assert settings is not None
    if settings.fp32:
        enable_fp32_lm_head(model)
    if settings.token_chunk_size is not None:
        enable_chunked_lm_head_logprobs(
            model,
            token_chunk_size=settings.token_chunk_size,
            vocab_chunk_size=settings.vocab_chunk_size,
            fp32_lm_head=settings.fp32,
        )
