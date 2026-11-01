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
"""Per-job settings of the MoE runtime that were first worker environment variables.

``MoeRuntimeOptions`` is a base of the MoE loaders' ``loader_options`` models, so a job sets these fields like any
other loader option (DSS: ``training_config.model_options.moe.*``, or the ``prime_rl`` block). Each field steers one
module that kept its environment variable: a field left unset (None) leaves that variable in charge, so a job that
sets none of them runs exactly as before. ``apply_moe_runtime_options`` hands the fields to their modules once per
model load, before the model is built.
"""

from __future__ import annotations

from os import environ
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self


class MoeRuntimeOptions(BaseModel):
    """Expert-balance settings shared by the MoE loaders' options models. None: the environment variable decides."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    ultraep_spares: int | None = Field(
        None,
        ge=0,
        description=(
            "Spare expert slots per GPU that UltraEP fills, in every MoE call, with copies of that call's hottest "
            "experts inside the owner's NVLink domain (moe/ultraep_replicas.py). 0 turns UltraEP off. Unset: "
            "ARL_ULTRAEP_SPARES."
        ),
    )
    ultraep_ipc_symm: bool | None = Field(
        None,
        description=(
            "Map UltraEP's peer buffers through CUDA IPC handles instead of torch symmetric memory, whose rendezvous "
            "refuses ranks that each see only their own GPU (one GPU per process, as DSS runs them). Written to "
            "ULTRA_EP_IPC_SYMM, which UltraEP reads, before its buffers are allocated. Unset: ULTRA_EP_IPC_SYMM."
        ),
    )
    deepep_v2_num_sms: int | None = Field(
        None,
        ge=0,
        description=(
            "SMs of the deepep_v2 backend's dispatch and combine kernels, in place of deepep_num_sms; 0 lets DeepEP V2 "
            "choose. Unset: ARL_DEEPEP_V2_NUM_SMS, else deepep_num_sms."
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
            "Spare expert slots per GPU for Arctic's own online replicas, planned once per micro-batch from an EMA of "
            "the routed counts (moe/expert_replicas.py). 0 turns them off. Unset: ARL_EXPERT_REPLICAS."
        ),
    )
    expert_replicas_ema: float | None = Field(
        None,
        gt=0,
        le=1,
        description=(
            "Weight of the newest micro-batch's counts in the replicas' load estimate (1 plans from that micro-batch "
            "alone). Unset: ARL_EXPERT_REPLICAS_EMA, else 0.5."
        ),
    )

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
    return int(environ.get(DEEPEP_V2_NUM_SMS_ENV, default))


def deepep_v2_max_tokens(default: int) -> int:
    """The deepep_v2 buffer's tokens per rank: the job's ``deepep_v2_max_tokens``, else
    ``ARL_DEEPEP_V2_MAX_TOKENS``, else ``default``."""
    if _deepep_v2_max_tokens is not None:
        return _deepep_v2_max_tokens
    return int(environ.get(DEEPEP_V2_MAX_TOKENS_ENV, default))


def apply_moe_runtime_options(options: Any) -> None:
    """Hand ``options``' runtime fields to the modules that read them; a field the options model lacks counts as
    unset. Every load sets every field, so a value from an earlier load in the same process never leaks into this
    one."""
    from . import expert_replicas
    from . import ultraep_replicas

    global _deepep_v2_num_sms, _deepep_v2_max_tokens
    _deepep_v2_num_sms = getattr(options, "deepep_v2_num_sms", None)
    _deepep_v2_max_tokens = getattr(options, "deepep_v2_max_tokens", None)
    ultraep_replicas.configure(
        spares=getattr(options, "ultraep_spares", None),
        ipc_symm=getattr(options, "ultraep_ipc_symm", None),
    )
    expert_replicas.configure(
        spares=getattr(options, "expert_replicas", None),
        ema=getattr(options, "expert_replicas_ema", None),
    )
