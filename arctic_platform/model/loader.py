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
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal

import torch.nn as nn
from pydantic import BaseModel
from pydantic import ConfigDict

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.platform import PlatformCapabilities

if TYPE_CHECKING:
    from transformers import PretrainedConfig


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


@dataclass(frozen=True)
class LoaderRuntimePolicy:
    attention: Literal["platform"] | str = "platform"
    ep_comm_backend: Literal["deepep", "uccl"] | None = None
    sp_strategy: Literal["transformers_ulysses", "native"] = "transformers_ulysses"
    sp_requires_head_divisibility: bool = True
    label_contract: Literal["causal_labels", "logit_aligned"] = "causal_labels"
    requires_weight_conversion: bool = False
    model_forward_requires_labels: bool = False


class ModelRuntimeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    loader: str
    attn_implementation: str
    ep_comm_backend: Literal["deepep", "uccl"] | None = None
    sp_strategy: Literal["transformers_ulysses", "native"]
    sp_requires_head_divisibility: bool = True
    label_contract: Literal["causal_labels", "logit_aligned"]
    requires_weight_conversion: bool
    model_forward_requires_labels: bool
    fused_cross_entropy: bool | str | None = None


@dataclass
class _LoaderEntry:
    fn: Loader
    matches: Matcher | None
    options: type[BaseModel] | None = None
    validate_spec: SpecValidator | None = None
    runtime_policy: LoaderRuntimePolicy = field(default_factory=LoaderRuntimePolicy)


_LOADERS: dict[str, _LoaderEntry] = {}
_DEFAULT_LOADER: str | None = None


def register_loader(
    name: str,
    matches: Matcher | None = None,
    default: bool = False,
    options: type[BaseModel] | None = None,
    validate_spec: SpecValidator | None = None,
    runtime_policy: LoaderRuntimePolicy | None = None,
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
            runtime_policy=runtime_policy or LoaderRuntimePolicy(),
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


def _platform_attention_default(platform: PlatformCapabilities) -> str:
    return {
        "blackwell": "flash_attention_4",
        "hopper": "flash_attention_3",
        "ampere": "sdpa",
    }.get(platform.accelerator, "sdpa")


def resolve_model_profile(
    spec: ModelSpec,
    platform: PlatformCapabilities | None = None,
) -> ModelRuntimeProfile:
    if spec.loader is None:
        raise ValueError("ModelSpec.loader must be resolved before runtime defaults")
    platform = platform or PlatformCapabilities.detect()
    entry = _LOADERS[spec.loader]
    policy = entry.runtime_policy

    attention = spec.attn_implementation
    if attention is None:
        attention = _platform_attention_default(platform) if policy.attention == "platform" else policy.attention
    if attention.startswith("flash_attention_") and attention not in platform.attention_backends:
        raise ValueError(
            f"{attention} is the default for loader {spec.loader!r} on {platform.accelerator}, "
            f"but the backend is unavailable; available={sorted(platform.attention_backends)}"
        )

    ep_comm_backend = None
    if spec.parallelism.expert_parallel > 1:
        requested_backend = spec.loader_options.get("ep_comm_backend")
        ep_comm_backend = requested_backend or policy.ep_comm_backend
        if ep_comm_backend is None:
            raise ValueError(f"loader {spec.loader!r} has no expert-parallel communication policy")
        if ep_comm_backend not in platform.ep_comm_backends:
            raise ValueError(
                f"{ep_comm_backend} is required by loader {spec.loader!r}, "
                f"but the backend is unavailable; available={sorted(platform.ep_comm_backends)}"
            )
        spec.loader_options["ep_comm_backend"] = ep_comm_backend

    spec.attn_implementation = attention
    if entry.options is not None:
        spec.loader_options = entry.options.model_validate(spec.loader_options).model_dump()

    fused_cross_entropy = spec.loader_options.get("fused_cross_entropy")
    if spec.patches.liger:
        fused_cross_entropy = "liger"
    return ModelRuntimeProfile(
        loader=spec.loader,
        attn_implementation=attention,
        ep_comm_backend=ep_comm_backend,
        sp_strategy=policy.sp_strategy,
        sp_requires_head_divisibility=policy.sp_requires_head_divisibility,
        label_contract=policy.label_contract,
        requires_weight_conversion=policy.requires_weight_conversion,
        model_forward_requires_labels=policy.model_forward_requires_labels,
        fused_cross_entropy=fused_cross_entropy,
    )


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
