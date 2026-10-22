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
"""The optimizer block a training job hands to DeepSpeed.

Imports nothing beyond Pydantic, so config validation can use it without loading model code.
"""

from __future__ import annotations

from typing import Annotated
from typing import Any

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictStr
from pydantic import model_validator

_DEFAULT_BETAS = (0.9, 0.999)
_NonNegative = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
_Beta = Annotated[float, Field(ge=0, lt=1, strict=True, allow_inf_nan=False)]


class OptimizerSettings(BaseModel):
    """Optimizer block handed to DeepSpeed.

    Also accepts DeepSpeed's ``{"type": ..., "params": {...}}`` shape and ``beta1`` / ``beta2``.
    Closed, because a dropped key would silently train with the default instead.
    """

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field("AdamW", pattern=r"\S", validation_alias=AliasChoices("name", "type"))
    lr: _NonNegative = 1e-5
    weight_decay: _NonNegative = 0.0
    betas: list[_Beta] = Field(default_factory=lambda: list(_DEFAULT_BETAS), min_length=2, max_length=2)
    eps: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)] = 1e-8
    # ``None`` lets the model family decide. GLM-MoE leaves FusedAdam off.
    fused: StrictBool | None = None
    # Older clients send it here, but it applies to the whole job: the job's training config owns it.
    gradient_clipping: _NonNegative | None = Field(None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _flatten_deepspeed_block(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        params = data.pop("params", None) or {}
        if not isinstance(params, dict):
            raise ValueError("optimizer.params must be an object")
        if overlap := sorted(data.keys() & params.keys()):
            raise ValueError(f"optimizer sets {overlap} both at the top level and in params")
        data |= params
        beta1, beta2 = data.pop("beta1", None), data.pop("beta2", None)
        if beta1 is not None or beta2 is not None:
            if "betas" in data:
                raise ValueError("optimizer sets betas and beta1/beta2. Use betas.")
            data["betas"] = [
                _DEFAULT_BETAS[0] if beta1 is None else beta1,
                _DEFAULT_BETAS[1] if beta2 is None else beta2,
            ]
        return data
