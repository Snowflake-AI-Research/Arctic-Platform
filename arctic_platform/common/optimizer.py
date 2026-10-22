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
"""The training optimizer block: its fields, defaults, limits, and accepted aliases.

The canonical form is flat and uses ``betas``:

    {"name": "AdamW", "lr": 1e-5, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8}

``OptimizerSettings`` also accepts what older clients send and rewrites it: a
DeepSpeed ``{"type": ..., "params": {...}}`` block, ``type`` for ``name``, and
``beta1`` / ``beta2`` for ``betas`` (a lone one takes its partner from
:data:`DEFAULT_BETAS`). Scheduler names are rejected with a pointer to what
works. Fields the caller did not send stay unset, so
``model_dump(exclude_unset=True)`` tells an explicit value from a default.
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

# Accepted on input, rewritten to the canonical spelling. ``type`` is the DeepSpeed spelling.
OPTIMIZER_KEY_ALIASES = {"type": "name"}

DEFAULT_BETAS = (0.9, 0.999)

# Look like optimizer fields but nothing consumes them. Rejected with an actionable
# message rather than accepted and ignored.
UNSUPPORTED_SCHEDULER_KEYS = frozenset(
    {
        "lr_scheduler",
        "lr_scheduler_type",
        "num_warmup_steps",
        "training_horizon",
        "warmup_ratio",
        "warmup_steps",
        "warmup_steps_proportion",
    }
)

_SCHEDULER_GUIDANCE = (
    "The server does not consume these names. For a client-driven learning rate, pass the per-step value "
    'to the client\'s step() call. For a server-side DeepSpeed scheduler, set ds_config["scheduler"].'
)

_NonNegative = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
_Beta = Annotated[float, Field(ge=0, lt=1, strict=True, allow_inf_nan=False)]


class OptimizerSettings(BaseModel):
    """The optimizer block handed to DeepSpeed. Closed: a dropped key would silently change training."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field("AdamW", pattern=r"\S")
    # lr=0 is a legitimate control run; lr<0 is the only invalid value.
    lr: _NonNegative = 1e-5
    weight_decay: _NonNegative = 0.0
    betas: list[_Beta] = Field(default_factory=lambda: list(DEFAULT_BETAS), min_length=2, max_length=2)
    eps: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)] = 1e-8
    # ``None`` lets the model family decide. GLM-MoE leaves FusedAdam off.
    fused: StrictBool | None = None
    # Older clients put it here, but it applies to the whole job: the training config
    # moves it to its own ``gradient_clipping``, so it is never dumped with the block.
    gradient_clipping: _NonNegative | None = Field(None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _resolve_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        flat = _flatten_params(data)
        if flat.get("betas", ()) is None:
            del flat["betas"]

        scheduler_keys = sorted(flat.keys() & UNSUPPORTED_SCHEDULER_KEYS)
        if scheduler_keys:
            raise ValueError(
                f"optimizer contains unsupported scheduler field(s): {scheduler_keys}. {_SCHEDULER_GUIDANCE}"
            )

        _merge_betas(flat)
        for alias, target in OPTIMIZER_KEY_ALIASES.items():
            if alias not in flat:
                continue
            value = flat.pop(alias)
            if target in flat and flat[target] != value:
                raise ValueError(f"optimizer sets both {target}={flat[target]!r} and {alias}={value!r}. Use {target}.")
            flat[target] = value

        unsupported = sorted(flat.keys() - cls.model_fields.keys())
        if unsupported:
            raise ValueError(
                f"optimizer contains unsupported field(s): {unsupported}. Supported fields: {sorted(cls.model_fields)}"
            )
        return flat


def _flatten_params(optimizer: dict[str, Any]) -> dict[str, Any]:
    """Merge a DeepSpeed ``params`` block into the top level, where the worker reads it."""
    flat = {key: value for key, value in optimizer.items() if key != "params"}
    params = optimizer.get("params")
    if params is None:
        return flat
    if not isinstance(params, dict):
        raise ValueError(f"optimizer.params must be an object, got {params!r}")
    for key, value in params.items():
        if key in flat and flat[key] != value:
            raise ValueError(f"optimizer sets {key}={flat[key]!r} and params.{key}={value!r}. Remove one.")
        flat[key] = value
    return flat


def _merge_betas(flat: dict[str, Any]) -> None:
    """Fold ``beta1`` / ``beta2`` into ``betas``; an alias that disagrees with ``betas`` is an error."""
    aliases = {alias: value for alias in ("beta1", "beta2") if (value := flat.pop(alias, None)) is not None}
    if len(aliases) == 0:
        return
    if "betas" not in flat:
        flat["betas"] = [aliases.get("beta1", DEFAULT_BETAS[0]), aliases.get("beta2", DEFAULT_BETAS[1])]
        return
    betas = flat["betas"]
    for index, alias in enumerate(("beta1", "beta2")):
        if alias in aliases and not (
            isinstance(betas, (list, tuple)) and len(betas) == 2 and betas[index] == aliases[alias]
        ):
            raise ValueError(f"optimizer sets betas={betas!r} and {alias}={aliases[alias]!r}. Use betas.")
