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
"""Loader integration for GLM-5.3-Flash."""

from __future__ import annotations

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader
from arctic_platform.model.loader import resolve_spec_with_defaults
from arctic_platform.model.loaders.flash_moe import load_flash_moe
from arctic_platform.model.loaders.flash_moe import matches_model_type
from arctic_platform.model.loaders.flash_moe import validate_flash_moe_spec
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions


def _validate_spec(spec: ModelSpec) -> None:
    from arctic_platform.model.implementations.glm53.deepspeed_integration import GLM53_ATTN_BACKEND

    validate_flash_moe_spec(
        spec,
        "GLM-5.3-Flash",
        allow_sequence_parallel=True,
    )
    if spec.attn_implementation is not None and spec.attn_implementation not in (
        GLM53_ATTN_BACKEND,
        "flashmla",
    ):
        raise ValueError(
            f"GLM-5.3-Flash training requires sparse MLA; got attn_implementation={spec.attn_implementation!r}"
        )


def _resolve_spec(spec, platform):
    return resolve_spec_with_defaults(
        spec,
        platform,
        attention="sparse_mla",
        ep_comm_backend="uccl",
        sp_strategy="native",
        label_contract="logit_aligned",
        requires_weight_conversion=True,
        model_forward_requires_labels=True,
    )


@register_loader(
    "glm5_next",
    resolve_spec=_resolve_spec,
    matches=matches_model_type("glm5_next"),
    options=GenericMoeOptions,
    validate_spec=_validate_spec,
)
def load_glm5_next(ctx: LoaderContext) -> LoadedModel:
    from arctic_platform.model.implementations.glm53.deepspeed_integration import load_glm5_next_model

    return load_flash_moe(ctx, load_glm5_next_model)
