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

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictStr
from pydantic import model_validator

_DEFAULT_BETAS = (0.9, 0.999)
_NULL_IS_ABSENT = frozenset({"betas", "beta1", "beta2", "gradient_clipping"})
_NonNegative = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
_Beta = Annotated[float, Field(ge=0, lt=1, strict=True, allow_inf_nan=False)]


class OptimizerSettings(BaseModel):
    """Optimizer block handed to DeepSpeed.

    Also accepts DeepSpeed's ``{"type": ..., "params": {...}}`` shape and ``beta1`` / ``beta2``.
    Closed, because a dropped key would silently train with the default instead.
    """

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field("AdamW", pattern=r"\S")
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
    def _resolve_aliases(cls, data: Any) -> Any:
        """A key given twice (top level and ``params``, or ``type`` and ``name``) is an error only when the values
        differ. ``None`` reads as absent for the keys older clients send as null."""
        if not isinstance(data, dict):
            return data
        params = {} if data.get("params") is None else data["params"]
        if not isinstance(params, dict):
            raise ValueError("optimizer.params must be an object")
        flat: dict[str, Any] = {}
        for key, value in [*data.items(), *params.items()]:
            key = "name" if key == "type" else key
            if key == "params" or (value is None and key in _NULL_IS_ABSENT):
                continue
            if key in flat and flat[key] != value:
                raise ValueError(f"optimizer sets {key} to both {flat[key]!r} and {value!r}. Remove one.")
            flat[key] = value

        aliases = [flat.pop("beta1", None), flat.pop("beta2", None)]
        if aliases == [None, None]:
            return flat
        merged = [default if alias is None else alias for alias, default in zip(aliases, _DEFAULT_BETAS)]
        if "betas" not in flat:
            flat["betas"] = merged
            return flat
        betas = flat["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError(f"optimizer sets betas={betas!r} and beta1/beta2. Use betas.")
        if any(alias is not None and alias != beta for alias, beta in zip(aliases, betas)):
            raise ValueError(f"optimizer sets betas={list(betas)!r} and beta1/beta2={aliases!r}. Use betas.")
        return flat
