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
"""Loader integration for Qwen3.8-Flash-Next."""

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
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import QWEN38_ATTN_BACKEND
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import QWEN38_NUM_EXPERTS

    validate_flash_moe_spec(
        spec,
        "Qwen3.8-Flash-Next",
        allow_sequence_parallel=True,
    )
    if spec.attn_implementation is not None and spec.attn_implementation not in (
        QWEN38_ATTN_BACKEND,
        "flex_attention",
    ):
        raise ValueError(
            "Qwen3.8-Flash-Next training requires QSA FlexAttention; "
            f"got attn_implementation={spec.attn_implementation!r}"
        )
    ep_size = spec.parallelism.expert_parallel
    if QWEN38_NUM_EXPERTS % ep_size:
        raise ValueError(
            f"Qwen3.8-Flash-Next has {QWEN38_NUM_EXPERTS} experts, "
            f"so ep_size={ep_size} must divide {QWEN38_NUM_EXPERTS}."
        )


def _resolve_spec(spec, platform):
    return resolve_spec_with_defaults(
        spec,
        platform,
        attention="qsa_flex",
        ep_comm_backend="uccl",
        sp_strategy="native",
        sp_requires_head_divisibility=False,
        label_contract="logit_aligned",
        requires_weight_conversion=True,
        model_forward_requires_labels=True,
    )


@register_loader(
    "qwen4_exp",
    resolve_spec=_resolve_spec,
    matches=matches_model_type("qwen4_exp"),
    options=GenericMoeOptions,
    validate_spec=_validate_spec,
)
def load_qwen4_exp(ctx: LoaderContext) -> LoadedModel:
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import load_qwen4_exp_model

    return load_flash_moe(ctx, load_qwen4_exp_model)
