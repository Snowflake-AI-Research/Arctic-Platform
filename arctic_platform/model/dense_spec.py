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
"""Build the ModelSpec used to load a dense Hugging Face model."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches


class DenseModelSettings(BaseModel):
    """Typed dense-model input shared by runtime adapters."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    model_path: str
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    sequence_parallel: int = 1
    patches: Patches = Field(default_factory=Patches)

    @field_validator("dtype")
    @classmethod
    def _normalize_dtype(cls, value: str) -> str:
        return {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }.get(value.lower(), value.lower())


def dense_model_spec(settings: DenseModelSettings) -> ModelSpec:
    """Build a Hugging Face ``ModelSpec`` from typed dense settings."""
    import arctic_platform.model.loaders  # noqa: F401  # registers the huggingface loader

    return ModelSpec(
        model_path_or_name=settings.model_path,
        dtype=settings.dtype,
        attn_implementation=settings.attn_implementation,
        loader="huggingface",
        parallelism=ParallelismConfig(sequence_parallel=settings.sequence_parallel),
        patches=settings.patches,
    )
