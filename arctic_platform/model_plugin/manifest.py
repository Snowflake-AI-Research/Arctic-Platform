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
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

CONTRACT_VERSION = "1.1"

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


# Sampler settings the manifest may require. A job's inference config may not set them to anything else; every
# other engine setting (tensor_parallel_size, max_num_seqs, gpu_memory_utilization, kernel flags, max_model_len)
# is the job's to tune. There is deliberately no way to set raw environment variables.
SAMPLER_REQUIREMENTS = (
    "tokenizer_mode",
    "reasoning_parser",
    "tool_call_parser",
    "model_runner",
    "enforce_eager",
    "chat_parser",
)


class SamplerSpec(BaseModel):
    """How the sampler (vLLM) serves this architecture: its capabilities, and the settings it requires."""

    model_config = ConfigDict(extra="forbid")

    vllm_architecture: str
    vllm_plugin: str | None = Field(None, description="'module:function' that registers the vLLM model class.")
    tensor_parallel: bool = False
    speculative_decoding: bool = False
    prefix_caching: bool = Field(
        False, description="The sampler is correct with prefix caching on; off, the platform launches it off."
    )
    enforce_eager: bool = Field(
        False, description="Run without CUDA graphs: required when a stateful layer is not a registered custom op."
    )
    tokenizer_mode: str | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    model_runner: Literal["v1", "v2"] | None = Field(None, description="The vLLM model runner the model needs.")
    chat_parser: str | None = Field(None, description="'module:Class' that parses this model's chat output.")

    @model_validator(mode="after")
    def _check(self) -> SamplerSpec:
        for name in ("vllm_plugin", "chat_parser"):
            value = getattr(self, name)
            if value is not None and ":" not in value:
                raise ValueError(f"{name} must be 'module:attribute', got {value!r}")
        return self

    def requirements(self) -> dict[str, Any]:
        """The engine settings this sampler requires, by name; ``enforce_eager`` only when it is required."""
        required = {name: getattr(self, name) for name in SAMPLER_REQUIREMENTS if getattr(self, name) is not None}
        if not self.enforce_eager:
            required.pop("enforce_eager")
        return required


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^custom/[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9._-]*$")
    version: str
    contract: str = Field(pattern=r"^\d+\.\d+$")
    runtime_release: str
    hf_model_types: list[str] = Field(min_length=1)
    layer_types: dict[str, str] = Field(min_length=1)
    sublayer_types: dict[str, str] = Field(default_factory=dict)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    sampler: SamplerSpec | None = None
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> PluginManifest:
        for table, entries in (("layer_types", self.layer_types), ("sublayer_types", self.sublayer_types)):
            for layer_type, source in entries.items():
                if source not in (PLATFORM_ATTENTION, PLUGIN_MIXER):
                    raise ValueError(
                        f"{table}[{layer_type!r}] must be {PLATFORM_ATTENTION!r} or {PLUGIN_MIXER!r}, got {source!r}"
                    )
        shared = set(self.layer_types) & set(self.sublayer_types)
        if len(shared) > 0:
            raise ValueError(f"types {sorted(shared)} are in both layer_types and sublayer_types")
        for name, profile in self.profiles.items():
            inference = profile.get("inference", {})
            overridden = set(inference) & set(SAMPLER_REQUIREMENTS)
            if len(overridden) > 0:
                raise ValueError(
                    f"profile {name!r} sets sampler requirements {sorted(overridden)}; set them in [sampler]"
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
