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

"""The plugin manifest: what a model plugin declares about itself before any of its code runs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

CONTRACT_VERSION = "1.0"

PLATFORM_ATTENTION = "platform:attention"
PLUGIN_MIXER = "plugin"


class Capabilities(BaseModel):
    """Features the plugin supports. A job asking for an undeclared feature is rejected at admission."""

    model_config = ConfigDict(extra="forbid")

    training: bool = True
    sampling: bool = False
    packing: bool = True
    sequence_parallel: bool = False
    expert_parallel: bool = False
    max_expert_parallel: int | None = Field(None, ge=1)
    router_replay: bool = False
    peft: bool = False

    @model_validator(mode="after")
    def _check(self) -> Capabilities:
        if self.max_expert_parallel is not None and not self.expert_parallel:
            raise ValueError("max_expert_parallel requires expert_parallel = true")
        if not self.training and not self.sampling:
            raise ValueError("a plugin must support training, sampling or both")
        return self


class SamplerSpec(BaseModel):
    """How the sampler (vLLM) serves this architecture."""

    model_config = ConfigDict(extra="forbid")

    vllm_architecture: str
    vllm_plugin: str | None = Field(None, description="'module:function' that registers the vLLM model class.")
    tensor_parallel: bool = False
    speculative_decoding: bool = False
    prefix_caching: bool = False
    enforce_eager: bool = Field(
        False, description="Run without CUDA graphs: required when a stateful layer is not a registered custom op."
    )

    @model_validator(mode="after")
    def _check(self) -> SamplerSpec:
        if self.vllm_plugin is not None and ":" not in self.vllm_plugin:
            raise ValueError(f"vllm_plugin must be 'module:function', got {self.vllm_plugin!r}")
        return self


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^custom/[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9._-]*$")
    version: str
    contract: str = Field(pattern=r"^\d+\.\d+$")
    runtime_release: str
    hf_model_types: list[str] = Field(min_length=1)
    layer_types: dict[str, str] = Field(min_length=1)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    sampler: SamplerSpec | None = None
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> PluginManifest:
        for layer_type, source in self.layer_types.items():
            if source not in (PLATFORM_ATTENTION, PLUGIN_MIXER):
                raise ValueError(
                    f"layer_types[{layer_type!r}] must be {PLATFORM_ATTENTION!r} or {PLUGIN_MIXER!r}, got {source!r}"
                )
        if self.capabilities.sampling and self.sampler is None:
            raise ValueError("capabilities.sampling = true requires a [sampler] table")
        if not self.capabilities.sampling and self.sampler is not None:
            raise ValueError("a [sampler] table requires capabilities.sampling = true")
        return self

    def check_contract(self, supported: str = CONTRACT_VERSION) -> None:
        """Refuse a manifest written against a different major contract version."""
        if self.contract.split(".")[0] != supported.split(".")[0]:
            raise ValueError(
                f"plugin {self.name} targets contract {self.contract}; this platform supports {supported}"
            )


def load_manifest(path: str | Path) -> PluginManifest:
    with open(path, "rb") as f:
        return PluginManifest.model_validate(tomllib.load(f))
