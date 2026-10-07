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
"""Loaders: how a base module is built and its weights materialized."""

from __future__ import annotations

import functools
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Mapping

import torch
import torch.nn as nn

from arctic_platform.model.config import ModelSpec

if TYPE_CHECKING:
    from pydantic import BaseModel
    from transformers import PretrainedConfig


@dataclass(frozen=True)
class ModelParallelismMetadata:
    """Model dimensions that constrain sequence/head parallel exchanges."""

    num_attention_heads: int
    num_key_value_heads: int | None
    has_linear_attention: bool
    linear_num_key_heads: int | None
    linear_num_value_heads: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelParallelismMetadata:
        return cls(
            num_attention_heads=int(value["num_attention_heads"]),
            num_key_value_heads=(
                int(value["num_key_value_heads"]) if value.get("num_key_value_heads") is not None else None
            ),
            has_linear_attention=bool(value.get("has_linear_attention", False)),
            linear_num_key_heads=(
                int(value["linear_num_key_heads"]) if value.get("linear_num_key_heads") is not None else None
            ),
            linear_num_value_heads=(
                int(value["linear_num_value_heads"]) if value.get("linear_num_value_heads") is not None else None
            ),
        )


def model_parallelism_metadata_from_config(
    config: Mapping[str, Any],
    *,
    source: str = "model config",
) -> ModelParallelismMetadata:
    """Extract parallelism-relevant dimensions from a text or composite model config."""
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        text_config = config

    linear_attn_config = text_config.get("linear_attn_config")

    def positive_int(
        name: str,
        *,
        required: bool,
        fallback: Any = None,
    ) -> int | None:
        value = text_config.get(name)
        if value is None and text_config is not config:
            value = config.get(name)
        if value is None:
            value = fallback
        if value is None and not required:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            requirement = "a positive integer" if required else "a positive integer when set"
            raise ValueError(f"{name} must be {requirement} in {source}; got {value!r}")
        return value

    layer_types = text_config.get("layer_types", config.get("layer_types", ()))
    if layer_types is None:
        layer_types = ()
    if not isinstance(layer_types, (list, tuple)):
        raise ValueError(f"layer_types must be a list in {source}; got {layer_types!r}")
    has_linear_attention = (
        "linear_attention" in layer_types
        or "linear_num_key_heads" in text_config
        or "linear_num_value_heads" in text_config
    )
    num_attention_heads = positive_int("num_attention_heads", required=True)
    assert num_attention_heads is not None

    return ModelParallelismMetadata(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=positive_int("num_key_value_heads", required=False),
        has_linear_attention=has_linear_attention,
        linear_num_key_heads=positive_int(
            "linear_num_key_heads",
            required=has_linear_attention,
            fallback=(linear_attn_config.get("num_heads") if isinstance(linear_attn_config, Mapping) else None),
        ),
        linear_num_value_heads=positive_int(
            "linear_num_value_heads",
            required=has_linear_attention,
            fallback=(linear_attn_config.get("num_heads") if isinstance(linear_attn_config, Mapping) else None),
        ),
    )


def canonical_parameter_name(name: str) -> str:
    """Remove activation-checkpoint wrapper segments from a parameter name."""
    return ".".join(segment for segment in name.split(".") if segment != "_checkpoint_wrapped_module")


def _float8_dtypes() -> frozenset[torch.dtype]:
    return frozenset(
        dtype
        for name in dir(torch)
        if name.startswith("float8_") and isinstance((dtype := getattr(torch, name)), torch.dtype)
    )


def finalize_model_for_training(model: nn.Module) -> int:
    """Freeze FP8 storage and inverse-scale parameters after model transformations."""
    float8_dtypes = _float8_dtypes()
    frozen = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.dtype in float8_dtypes or name.endswith("_scale_inv")):
            parameter.requires_grad_(False)
            frozen += 1
    return frozen


@functools.lru_cache(maxsize=None)
def _load_hf_config(model_path_or_name: str) -> PretrainedConfig | None:
    """Load the HuggingFace config, or None when no config file exists.

    An unrecognized ``model_type`` still returns the raw config so a family matcher
    can select a loader. A config file that cannot be parsed raises. Treating that
    as a missing model would select the default loader with an empty model type.
    """
    from transformers import AutoConfig
    from transformers import PretrainedConfig

    try:
        return AutoConfig.from_pretrained(model_path_or_name)
    except ValueError:
        config, _ = PretrainedConfig.get_config_dict(model_path_or_name)
        if not isinstance(config, dict) or not config.get("model_type"):
            return None
        return PretrainedConfig.from_dict(config)
    except OSError:
        from pathlib import Path

        if Path(model_path_or_name, "config.json").is_file():
            raise
        return None


def _model_type(config: Any) -> str:
    if isinstance(config, dict):
        return str(config.get("model_type") or "")
    return str(getattr(config, "model_type", "") or "")


@dataclass
class LoaderContext:
    """Everything a loader needs to build a model."""

    spec: ModelSpec
    parallel_groups: Any | None = None

    @property
    def hf_config(self) -> PretrainedConfig | None:
        """The model's HuggingFace config, or None if the name/path isn't an HF model. Parsed once."""
        return _load_hf_config(self.spec.model_path_or_name)

    @property
    def hf_model_type(self) -> str:
        return _model_type(self.hf_config)

    @property
    def hf_text_model_type(self) -> str:
        config = self.hf_config
        text_config = config.get("text_config") if isinstance(config, dict) else getattr(config, "text_config", None)
        return _model_type(text_config)


@dataclass
class LoadedModel:
    """A built model and the patches already applied to it."""

    model: nn.Module
    applied_patches: frozenset[str] = field(default_factory=frozenset)


Loader = Callable[[LoaderContext], LoadedModel]
Matcher = Callable[[LoaderContext], bool]
SpecValidator = Callable[[ModelSpec], None]


@dataclass
class _LoaderEntry:
    fn: Loader
    matches: Matcher | None
    options: type[BaseModel] | None = None
    validate_spec: SpecValidator | None = None


_LOADERS: dict[str, _LoaderEntry] = {}
_DEFAULT_LOADER: str | None = None


def register_loader(
    name: str,
    matches: Matcher | None = None,
    default: bool = False,
    options: type[BaseModel] | None = None,
    validate_spec: SpecValidator | None = None,
) -> Callable[[Loader], Loader]:
    """Register a loader by name.

    Optionally give it a ``matches`` predicate, mark it the ``default``, or attach an
    ``options`` pydantic model used to validate ``ModelSpec.loader_options``.
    """

    def decorator(fn: Loader) -> Loader:
        global _DEFAULT_LOADER
        assert name not in _LOADERS, f"loader {name!r} already registered"
        if default:
            assert _DEFAULT_LOADER is None, f"default loader already registered: {_DEFAULT_LOADER!r}"
            _DEFAULT_LOADER = name
        _LOADERS[name] = _LoaderEntry(
            fn=fn,
            matches=matches,
            options=options,
            validate_spec=validate_spec,
        )
        return fn

    return decorator


def is_registered_loader(name: str) -> bool:
    return name in _LOADERS


def get_loader_options_model(name: str) -> type[BaseModel] | None:
    """Return the pydantic options model registered for a loader, if any."""
    return _LOADERS[name].options


def validate_loader_spec(name: str, spec: ModelSpec) -> None:
    validator = _LOADERS[name].validate_spec
    if validator is not None:
        validator(spec)


def resolve_loader_name(spec: ModelSpec) -> str:
    """Resolve which loader a spec should use: single matching predicate, else the default."""
    ctx = LoaderContext(spec=spec)
    matched = [name for name, entry in _LOADERS.items() if entry.matches is not None and entry.matches(ctx)]
    assert len(matched) <= 1, f"multiple loaders match: {matched}; set spec.loader to disambiguate"
    if len(matched) == 1:
        return matched[0]

    assert _DEFAULT_LOADER is not None, "no default loader registered"
    return _DEFAULT_LOADER


def select_loader(ctx: LoaderContext) -> Loader:
    """Return the loader for a spec whose ``loader`` has already been resolved."""
    name = ctx.spec.loader
    assert name is not None, "spec.loader must be resolved before select_loader"
    return _LOADERS[name].fn
