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
"""Declarative, JSON-serializable spec that describes a model to build."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self


class ParallelismConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    expert_parallel: int = Field(1, description="Expert-parallel degree.")
    sequence_parallel: int = Field(1, description="Ulysses sequence-parallel degree.")


class Optimizations(BaseModel):
    """Optional features applied to the model after it is loaded."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    liger: bool = Field(False, description="Apply Liger kernels.")


class ModelSpec(BaseModel):
    """Everything needed to build a model, fully JSON-serializable.

    Non-serializable runtime handles (process groups, device mesh) are NOT in the
    spec; they are passed separately to ``build_model(spec, parallel_groups=...)``.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    model_path: str | None = Field(None, description="HF model id/path. None for fully-custom loaders.")
    dtype: str = Field("bfloat16", description="Parameter dtype.")
    attn_implementation: str | None = Field(None, description="Attention implementation to request from HF.")
    loader: str | None = Field(None, description="Loader name; auto-resolved at construction when not set.")
    parallelism: ParallelismConfig = Field(
        default_factory=ParallelismConfig, description="Loader-specific parallelism."
    )
    optimizations: Optimizations = Field(default_factory=Optimizations, description="Post-load optimizations.")
    loader_options: dict[str, Any] = Field(default_factory=dict, description="JSON-only loader-specific extras.")

    @model_validator(mode="after")
    def _resolve_loader(self) -> Self:
        from arctic_platform.model.loader import is_registered_loader
        from arctic_platform.model.loader import resolve_loader_name

        if self.loader is None:
            self.loader = resolve_loader_name(self)
        else:
            assert is_registered_loader(self.loader), f"unknown loader {self.loader!r}"
        return self
