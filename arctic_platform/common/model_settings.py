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
"""The model settings a training job sets, shared by the on-prem worker and Cortex.

``model_options`` names each option by its registry key and is rewritten onto the flat fields
(``model_option_aliases``); the loaders read only the flat fields. Imports nothing beyond Pydantic and the
option registry, so config validation can use it without loading model code.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Mapping

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt
from pydantic import StrictBool
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from arctic_platform.common import model_option_aliases as aliases
from arctic_platform.common.config import validate_peft_config

logger = logging.getLogger(__name__)

MODEL_PROVIDERS = ("huggingface", "arctic", aliases.DFLASH2_PROVIDER)


def _provenance(info: ValidationInfo | None) -> Any:
    """The ``provenance`` a caller passed in the validation context, which collects what validation renamed."""
    context = info.context if info is not None else None
    return context.get("provenance") if isinstance(context, dict) else None


def _require_strict_positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


class ModelSettings(BaseModel):
    """How a training job builds and patches its model, under the field names Cortex has always read.

    ``ep_size`` above 1 selects the expert-parallel (MoE) route; the ``model_options`` rewrite reads it to pick
    which fields an option writes.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_provider: str = "huggingface"
    use_liger: StrictBool | None = Field(
        None, description="Liger kernels on a huggingface model; the arctic provider uses lm_head.cross_entropy."
    )
    attn_impl: str | None = Field(None, description="Overrides attn_implementation when set.")
    attn_implementation: str | None = None
    activation_checkpointing: StrictBool | PositiveInt = Field(
        True, validation_alias=AliasChoices(*aliases.ACTIVATION_CHECKPOINTING_NAMES)
    )
    ac_config: dict[str, Any] | None = None
    compile: dict[str, Any] | None = None
    fp32_lm_head: bool = False
    fused_lm_head_token_chunk_size: int | None = Field(None, ge=1)
    fused_lm_head_vocab_chunk_size: int = Field(8192, ge=1)
    tiled_mlp_token_chunk_size: int | None = Field(None, ge=1)
    fp32_parity: StrictBool | None = Field(
        None, description="The huggingface loader's fp32_parity on the dense route; unset keeps its default."
    )
    fp32_residual: StrictBool | None = Field(
        None, description="The huggingface loader's experimental fp32_residual; it requires fp32_parity."
    )
    text_config_overrides: dict[str, Any] | None = Field(
        None,
        description=(
            "Recursively merge into the text config (text_config in composite config.json, otherwise the "
            "whole config). Training and sampling sub-jobs must use equal values; agreement is not cross-checked."
        ),
    )
    ep_size: int | None = Field(None, ge=1)
    prime_rl: dict[str, Any] | None = Field(None, description="The arctic MoE loader's options.")
    peft_config: dict[str, Any] | None = None
    model_options: dict[str, Any] | None = Field(
        None, description="Model options by registry key, rewritten onto the fields above."
    )
    # The prime_rl entries the request itself carried, recorded before the model_options rewrite adds its own,
    # for the deprecation warning. A field so it survives a dump and a second validation.
    prime_rl_request_entries: SkipJsonSchema[list[str] | None] = None

    @field_validator(
        "fused_lm_head_token_chunk_size", "fused_lm_head_vocab_chunk_size", "tiled_mlp_token_chunk_size", mode="before"
    )
    @classmethod
    def _require_strict_positive_chunk_size(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return None
        return _require_strict_positive_integer(value, field_name=str(info.field_name))

    @field_validator("prime_rl", mode="before")
    @classmethod
    def _reject_nested_tiled_mlp_setting(cls, value: Any) -> Any:
        if isinstance(value, dict) and "tiled_mlp_token_chunk_size" in value:
            raise ValueError(
                "prime_rl.tiled_mlp_token_chunk_size is not supported; "
                "use training_config.tiled_mlp_token_chunk_size for every model provider"
            )
        return value

    _validate_peft_config = field_validator("peft_config")(validate_peft_config)

    @model_validator(mode="after")
    def _check_text_config_overrides(self) -> Self:
        if self.text_config_overrides and (self.ep_size or 1) > 1:
            raise ValueError("text_config_overrides is unsupported with training_config.ep_size > 1")
        return self

    @model_validator(mode="after")
    def _reject_fullgraph_with_tiled_mlp(self) -> Self:
        if self.tiled_mlp_token_chunk_size is not None and bool((self.compile or {}).get("fullgraph", False)):
            raise ValueError("training_config.compile.fullgraph cannot be combined with tiled_mlp_token_chunk_size")
        return self

    @model_validator(mode="after")
    def _reject_liger_on_arctic(self) -> Self:
        if self.use_liger and self.model_provider == "arctic":
            raise ValueError(
                "model_options.liger applies to huggingface models. On the arctic provider, "
                "set model_options.lm_head.cross_entropy."
            )
        return self

    # Pydantic runs ``mode="before"`` model validators in reverse order of definition, so these four run
    # bottom-up: the provider alias, the model_options rewrite, then the checks that must see rewritten fields.
    @model_validator(mode="before")
    @classmethod
    def _normalize_activation_checkpointing_aliases(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        names = aliases.ACTIVATION_CHECKPOINTING_NAMES
        normalized = dict(data)
        present = [key for key in names if key in normalized]
        if present:
            normalized["activation_checkpointing"] = normalized[present[0]]
        for key in names[1:]:
            normalized.pop(key, None)

        provenance = _provenance(info)
        if provenance is not None and present and present[0] != names[0]:
            provenance.normalized.append(
                {
                    "field": f"training_config.{present[0]}",
                    "submitted_value": data[present[0]],
                    "effective_field": f"training_config.{names[0]}",
                    "reason": f"'{present[0]}' is another name for '{names[0]}'",
                }
            )
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _validate_model_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "model_provider" not in data:
            return data
        provider = data["model_provider"]
        if provider == "liger":
            raise ValueError(
                "training_config.model_provider='liger' is not accepted. "
                "Set training_config.model_options.liger=true to apply Liger kernels on a huggingface model."
            )
        if provider not in MODEL_PROVIDERS:
            raise ValueError(
                "training_config.model_provider must be 'huggingface', 'arctic' or "
                f"'{aliases.DFLASH2_PROVIDER}', got {provider!r}."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _apply_model_options(cls, data: Any, info: ValidationInfo) -> Any:
        data = aliases.record_prime_rl_request_entries(data)
        return aliases.apply(data, location=aliases.TRAINING, provenance=_provenance(info))

    @model_validator(mode="before")
    @classmethod
    def _alias_deprecated_model_provider(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict) or data.get("model_provider") != "prime_rl":
            return data
        if _provenance(info) is None:
            logger.warning("training_config.model_provider='prime_rl' is deprecated and accepted as 'arctic'.")
        return {**data, "model_provider": "arctic"}

    def resolved_attn_implementation(self) -> str | None:
        """The attention implementation the job asks for: ``prime_rl.attn``, else ``attn_impl``, else
        ``attn_implementation``. The top-level and ``prime_rl`` names must agree."""
        top_level = self.attn_impl or self.attn_implementation
        prime_rl = (self.prime_rl or {}).get("attn")
        if top_level is not None and prime_rl is not None and top_level != prime_rl:
            raise ValueError(
                f"the job asks for attention implementation {top_level!r} at the top level and {prime_rl!r} in "
                "its prime_rl block; name it once"
            )
        return prime_rl or top_level


# ``ds_worker_config`` keys the on-prem worker once read as model settings, by the field that replaced each.
_RENAMED_WORKER_KEYS = {
    "enable_gradient_checkpointing": "activation_checkpointing",
    **{name: "activation_checkpointing" for name in aliases.ACTIVATION_CHECKPOINTING_NAMES},
}


def worker_config_model_keys(ds_worker_config: Mapping[str, Any] | None) -> dict[str, str]:
    """The model settings a ``ds_worker_config`` still carries, each mapped to the ``ModelSettings`` field that
    replaced it. A ``ds_worker_config`` may set none of them."""
    found = {}
    for key in ds_worker_config or {}:
        if key in _RENAMED_WORKER_KEYS:
            found[key] = _RENAMED_WORKER_KEYS[key]
        elif key in ModelSettings.model_fields:
            found[key] = key
    return found
