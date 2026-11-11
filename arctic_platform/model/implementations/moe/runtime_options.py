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
"""Per-job settings of the MoE runtime that would otherwise be worker environment variables.

``MoeRuntimeOptions`` is a base of the MoE loaders' ``loader_options`` models (glm_moe_dsa, qwen3_5_moe, which the
deepseek_v4 / deepseek_v41 and flash_moe loaders validate with too, and generic_moe), so a job sets these fields like
any other loader option (DSS: ``training_config.model_options.moe.*``, or the ``prime_rl`` block). Each field steers
one module that keeps its environment variable as the fallback: a field left unset (None) leaves that variable in
charge, so a job that sets none of them runs exactly as before. ``apply_moe_runtime_options`` hands the fields to
their modules once per model load, before the model is built.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

# field -> (module that reads it, relative to arctic_platform.model.implementations; the setter it takes it through)
_SETTERS = {
    "frozen_expert_offload": ("moe.frozen_offload", "configure_from_options"),
    "shard_frozen_dense": ("moe.frozen_shard", "configure_from_options"),
    "w4a8_experts": ("kernels.w4a8", "configure_from_options"),
    "w8a8_experts": ("kernels.fp8_grouped_w8a8", "configure_from_options"),
    # One spare count for UltraEP's three paths (moe/ultraep_replicas.py picks one from the experts).
    "ultraep_spares": ("moe.ultraep_replicas", "configure_from_options"),
    "expert_replicas": ("moe.expert_replicas", "configure_spares"),
    "expert_replicas_ema": ("moe.expert_replicas", "configure_ema"),
    "expert_placement": ("moe.expert_placement", "configure_from_options"),
    # glm_moe_dsa's own fields (GlmMoeDsaOptions): the other loaders lack them, so they count as unset there.
    "shard_dense": ("moe.dense_shard", "configure_from_options"),
    "shard_dense_group_size": ("moe.dense_shard", "configure_group_size"),
}
_ULTRA_EP_IPC_SYMM_ENV = "ULTRA_EP_IPC_SYMM"


class MoeRuntimeOptions(BaseModel):
    """Memory, expert-balance, expert-GEMM and deepep_v2 settings of the MoE runtime. None: the environment variable
    decides. UltraEP's spares and the online replicas both fill the spare expert slots, so at most one of them is
    on."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    frozen_expert_offload: bool | None = Field(
        None,
        description=(
            "Keep the frozen routed experts' weights (codes) in pinned host memory, copied into two GPU buffers per "
            "experts call with the next layer prefetched (moe/frozen_offload.py); layers that are not recomputed keep "
            "theirs on the GPU. Unset: ARL_OFFLOAD_FROZEN_EXPERTS."
        ),
    )
    shard_frozen_dense: bool | None = Field(
        None,
        description=(
            "Hold each recomputed layer's frozen replicated weights as a 1/world shard per rank, all-gathered for each "
            "call of the layer (moe/frozen_shard.py). Unset: ARL_SHARD_FROZEN_DENSE."
        ),
    )
    w4a8_experts: bool | None = Field(
        None,
        description=(
            "Run INT4 routed experts with FP8 activations (W4A8, the checkpoint's static activation scales) instead of "
            "decoding them to BF16 (kernels/w4a8). Unset: ARL_W4A8_EXPERTS."
        ),
    )
    ultraep_spares: int | None = Field(
        None,
        ge=0,
        description=(
            "Spare expert slots per GPU that UltraEP fills, in every MoE call, with copies of that call's hottest "
            "experts inside the owner's NVLink domain. The experts pick the path: BF16 experts as stored, trained in "
            "full or frozen without an adapter (moe/ultraep_replicas.py); frozen block-FP8 or packed 4-bit (INT4, "
            "NVFP4, MXFP4) experts with their LoRA (moe/ultraep_frozen.py). 0 turns UltraEP off. Unset: "
            "ARL_ULTRAEP_SPARES."
        ),
    )
    w8a8_experts: bool | None = Field(
        None,
        description=(
            "Run the frozen block-FP8 routed experts' forward GEMMs as W8A8 (activations quantized to E4M3 per token "
            "per 128 inputs) through DeepGEMM on Hopper, instead of decoding the weights to BF16; input gradients "
            "stay BF16 GEMMs on the decoded weights (kernels/fp8_grouped_w8a8.py). Refused for experts that are not "
            "block FP8. Unset: ARL_W8A8_EXPERTS."
        ),
    )
    ultraep_ipc_symm: bool | None = Field(
        None,
        description=(
            "Map UltraEP's peer buffers through CUDA IPC handles instead of torch symmetric memory, whose rendezvous "
            "refuses ranks that each see only their own GPU (as DSS runs them). Written to ULTRA_EP_IPC_SYMM, which "
            "UltraEP reads, before its buffers are allocated. Unset: ULTRA_EP_IPC_SYMM."
        ),
    )
    deepep_v2_num_sms: int | None = Field(
        None,
        ge=0,
        description=(
            "SMs of the deepep_v2 backend's dispatch and combine kernels, in place of deepep_num_sms; 0 lets DeepEP "
            "V2 choose. Unset: ARL_DEEPEP_V2_NUM_SMS, else deepep_num_sms."
        ),
    )
    deepep_v2_max_tokens: int | None = Field(
        None,
        gt=0,
        description=(
            "Tokens per rank the deepep_v2 buffer is sized for, at least the largest per-rank token count of any MoE "
            "call. Unset: ARL_DEEPEP_V2_MAX_TOKENS, else the first call's count, agreed over the EP group."
        ),
    )
    expert_replicas: int | None = Field(
        None,
        ge=0,
        description=(
            "Spare expert slots per GPU for Arctic's own online replicas of BF16 experts, planned once per "
            "micro-batch from an EMA of the routed counts and kept on their owner's node (moe/expert_replicas.py). 0 "
            "turns them off. Unset: ARL_EXPERT_REPLICAS."
        ),
    )
    expert_replicas_ema: float | None = Field(
        None,
        gt=0,
        le=1,
        description=(
            "Weight of the newest micro-batch's counts in the online replicas' load estimate (1 plans from that "
            "micro-batch alone). Unset: ARL_EXPERT_REPLICAS_EMA, else 0.5."
        ),
    )
    expert_placement: str | dict[str, Any] | None = Field(
        None,
        description=(
            "A static placement of each MoE layer's experts over the EP ranks, fitted offline from router histograms "
            "(moe/expert_placement.py): a JSON file path on the workers, the JSON text, or the object itself, "
            '{"ep_size": ep, "gpus_per_node": g, "layers": {"<layer>": [expert per slot, ...]}}. Unset: '
            "ARL_EXPERT_PLACEMENT."
        ),
    )

    @field_validator("expert_placement")
    @classmethod
    def _check_placement(cls, value: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        """Refuse what can be refused without the workers' files: an empty path, malformed inline JSON, or an
        object without its ``layers`` map. The permutations are checked against the model when it loads."""
        if value is None:
            return None
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("expert_placement must be a file path, the placement JSON, or its object")
            if not value.lstrip().startswith("{"):
                return value
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"expert_placement is not valid JSON: {error}") from None
        else:
            parsed = value
        if not isinstance(parsed, dict) or not isinstance(parsed.get("layers"), dict):
            raise ValueError("expert_placement must hold {'layers': {layer: [expert per slot, ...]}}")
        return value

    @model_validator(mode="after")
    def _one_spare_planner(self) -> Self:
        if (self.ultraep_spares or 0) > 0 and (self.expert_replicas or 0) > 0:
            raise ValueError(
                "ultraep_spares and expert_replicas both plan the spare expert slots; set at most one of them"
            )
        return self


# The deepep_v2 settings, read by distributed/deepep_v2.py, which cannot be imported without DeepEP V2.
DEEPEP_V2_NUM_SMS_ENV = "ARL_DEEPEP_V2_NUM_SMS"
DEEPEP_V2_MAX_TOKENS_ENV = "ARL_DEEPEP_V2_MAX_TOKENS"
_deepep_v2_num_sms: int | None = None
_deepep_v2_max_tokens: int | None = None


def deepep_v2_num_sms(default: int) -> int:
    """The deepep_v2 SM count: the job's ``deepep_v2_num_sms``, else ``ARL_DEEPEP_V2_NUM_SMS``, else ``default``."""
    if _deepep_v2_num_sms is not None:
        return _deepep_v2_num_sms
    return int(os.environ.get(DEEPEP_V2_NUM_SMS_ENV, default))


def deepep_v2_max_tokens(default: int) -> int:
    """The deepep_v2 buffer's tokens per rank: the job's ``deepep_v2_max_tokens``, else
    ``ARL_DEEPEP_V2_MAX_TOKENS``, else ``default``."""
    if _deepep_v2_max_tokens is not None:
        return _deepep_v2_max_tokens
    return int(os.environ.get(DEEPEP_V2_MAX_TOKENS_ENV, default))


def apply_moe_runtime_options(options: Any) -> None:
    """Hand ``options``' runtime fields to the modules that read them; a field the options model lacks counts as
    unset. Every load sets every field, so a value from an earlier load in the same process never leaks into this
    one."""
    global _deepep_v2_num_sms, _deepep_v2_max_tokens
    for field, (module_name, setter) in _SETTERS.items():
        module = importlib.import_module(f"arctic_platform.model.implementations.{module_name}")
        getattr(module, setter)(getattr(options, field, None))
    _deepep_v2_num_sms = getattr(options, "deepep_v2_num_sms", None)
    _deepep_v2_max_tokens = getattr(options, "deepep_v2_max_tokens", None)
    ipc_symm = getattr(options, "ultraep_ipc_symm", None)
    if ipc_symm is not None:
        os.environ[_ULTRA_EP_IPC_SYMM_ENV] = "1" if ipc_symm else "0"
