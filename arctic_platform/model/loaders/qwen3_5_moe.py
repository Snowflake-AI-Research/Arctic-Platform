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
"""Loader for the carved-out Qwen3.5 MoE implementation (meta-init, EP/DeepEP, DeepSpeed)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader

# These options mirror the qwen35 loader's ``prl_config`` keys (see
# implementations/qwen35/deepspeed_integration.load_moe_model_for_dss and config.py).
# The internal implementation is under active development; keep these in sync with it.


class ActivationOffloadOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    enabled: bool = Field(False, description="Stream checkpointed block boundaries to CPU.")
    keep_last_n: int = Field(1, description="Boundaries to leave resident on GPU.")
    use_streams: bool = Field(True, description="Overlap offload copies on side streams.")
    tensor_size_threshold: int | None = Field(None, description="Min bytes to offload; None uses the default.")


class ActivationCheckpointOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    mode: Literal["full", "selective"] = Field("full", description="Recompute whole blocks or selected targets.")
    freq: int = Field(1, description="Checkpoint every Nth block.")
    targets: list[str] = Field(default_factory=lambda: ["norm"], description="Submodules to checkpoint (selective).")
    # Matches the internal ActivationCheckpointConfig default_factory (an offload config with enabled=False),
    # not None: apply_ac reads ``offload_config.enabled`` unconditionally.
    offload_config: ActivationOffloadOptions = Field(
        default_factory=ActivationOffloadOptions, description="CPU offload of checkpointed boundaries."
    )
    router_replay_recompute: bool = Field(True, description="Deterministic MoE routing across recompute.")


class Qwen3_5MoeOptions(BaseModel):
    """Validated ``loader_options`` for the qwen3_5_moe loader (passed through as prl_config)."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    seq_len: int = Field(4096, description="Training sequence length.")
    attn: str = Field("flash_attention_3", description="Attention implementation.")
    ep_comm_backend: Literal["deepep"] = Field("deepep", description="Expert-parallel comm backend.")
    optimization_dtype: Literal["bfloat16", "float32"] = Field("bfloat16", description="Compute/param dtype.")
    reduce_dtype: Literal["bfloat16", "float32"] = Field("float32", description="Gradient reduction dtype.")
    moe_use_grouped_mm: bool = Field(True, description="Use grouped matmul for experts.")
    fused_cross_entropy: bool | Literal["liger"] = Field("liger", description="LM-head fused CE backend.")
    fused_lm_head_token_chunk_size: int | Literal["auto", "disabled"] = Field(
        "disabled", description="Chunked LM-head logprobs token size."
    )
    fp32_lm_head: bool = Field(False, description="Compute the LM head in fp32.")
    tiled_mlp_token_chunk_size: int | None = Field(None, description="ALST tiled shared-expert MLP token chunk.")
    deepep_token_chunk_size: int | None = Field(None, description="DeepEP dispatch token chunk size.")
    # Mirrors ModelConfig.weight_conversion_cache_dir (the implementation's effective default).
    weight_conversion_cache_dir: str = Field(
        "/data-fast/prime-rl-weight-cache", description="Dir for the one-time HF<->Prime weight-conversion cache."
    )
    ac_config: ActivationCheckpointOptions | None = Field(None, description="Activation checkpointing config.")
    debug: dict | None = Field(None, description="Test-only tiny-model overrides.")

    @model_validator(mode="after")
    def _check_lm_head(self) -> Self:
        if self.fused_cross_entropy and (self.fp32_lm_head or isinstance(self.fused_lm_head_token_chunk_size, int)):
            raise ValueError(
                "cannot combine fused_cross_entropy with fp32_lm_head or an integer fused_lm_head_token_chunk_size"
            )
        return self


def _matches(ctx: LoaderContext) -> bool:
    if ctx.spec.parallelism.expert_parallel <= 1:
        return False
    model_type = getattr(ctx.hf_config, "model_type", "") or ""
    return model_type == "qwen3_5_moe_text"


@register_loader("qwen3_5_moe", matches=_matches, options=Qwen3_5MoeOptions)
def load_qwen3_5_moe(ctx: LoaderContext) -> LoadedModel:
    # The Liger patch dispatches on model_type and targets a standard HF layout; it
    # does not match this custom architecture (custom RMSNorm/RoPE, grouped-mm MoE,
    # GatedDeltaNet), so it would silently no-op or apply wrong kernels. The LM-head
    # Liger fusion is available separately via loader_options["fused_cross_entropy"].
    if ctx.spec.patches.liger:
        raise ValueError(
            "the qwen3_5_moe loader does not support the liger patch; "
            'use loader_options={"fused_cross_entropy": "liger"} for the LM head instead'
        )

    from arctic_platform.model.implementations.qwen35 import load_moe_model_for_dss

    parallelism = ctx.spec.parallelism
    groups = ctx.parallel_groups or {}
    model = load_moe_model_for_dss(
        model_name=ctx.spec.model_path_or_name,
        ep_size=parallelism.expert_parallel,
        sp_size=parallelism.sequence_parallel,
        sp_group=groups.get("sp_group"),
        prl_config=ctx.spec.loader_options,
    )
    return LoadedModel(model=model)
