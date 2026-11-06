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
"""The training-engine settings a job sets, shared by the on-prem worker and Cortex.

Imports nothing beyond Pydantic and the option registry, so config validation can use it without loading model
code.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from arctic_platform.common.model_settings import ModelSettings
from arctic_platform.common.optimizer import OptimizerSettings

# The engine steps once per ``step`` call with one microbatch per accumulation boundary, so these
# DeepSpeed keys are owned by typed fields or fixed by the engine.
_ENGINE_OWNED_DS_KEYS = {
    "optimizer": "set training.optimizer",
    "gradient_clipping": "set training.gradient_clipping",
    "gradient_accumulation_steps": "set training.mb_spec.num_micro_batches",
    "train_micro_batch_size_per_gpu": "set training.mb_spec.num_micro_batches",
    "train_batch_size": "the engine derives it from the data-parallel size",
    "bf16": "set dtype, and training.fp32_masters for fp32 master weights",
    "fp16": "set dtype",
}


class TrainingEngineSettings(BaseModel):
    """Optimizer and DeepSpeed settings that ``training_engine_spec`` turns into a ``TrainingEngineSpec``.

    ``optimizer=None`` is a frozen model. ``ds_config`` is merged over the built DeepSpeed config, one level
    deep for ``zero_optimization`` and replacing every other top-level key.
    """

    model_config = ConfigDict(extra="forbid")

    optimizer: OptimizerSettings | None = Field(default_factory=OptimizerSettings)
    gradient_clipping: float = Field(1.0, ge=0.0)
    fp32_masters: bool = Field(
        False, description="Keep fp32 master weights and optimizer states for a bf16 job; the default is bf16."
    )
    zero_bucket_size: PositiveInt | None = Field(
        None, description="ZeRO reduce and allgather bucket size, in elements."
    )
    ds_config: dict[str, Any] | None = Field(
        None,
        description="DeepSpeed config-json merged over the built config (zero_optimization, scheduler, ...).",
    )


class MicroBatchSpec(BaseModel):
    """How a worker lowers one ``fwd-bwd`` batch into model calls.

    The on-prem worker splits each rank's shard into ``num_micro_batches`` equal parts. Cortex packs
    by the ``max_tokens_per_mb`` token budget.
    """

    model_config = ConfigDict(extra="forbid")

    num_micro_batches: PositiveInt | None = None
    max_tokens_per_mb: PositiveInt | None = None

    @model_validator(mode="after")
    def _one_policy(self) -> Self:
        if self.num_micro_batches is not None and self.max_tokens_per_mb is not None:
            raise ValueError("mb_spec sets both num_micro_batches and max_tokens_per_mb. Use one.")
        return self


class WorkerTrainingSettings(TrainingEngineSettings, ModelSettings):
    """``TrainingEngineSettings`` and ``ModelSettings`` plus the worker's microbatching and debug options.

    ``ds_config`` may not set what a typed field or the engine owns.
    """

    mb_spec: MicroBatchSpec | None = None
    debug: dict[str, Any] | None = Field(None, description="Worker debug options (gpu_mem_preallocate_gib, ...).")

    @field_validator("ds_config")
    @classmethod
    def _reject_engine_owned_keys(cls, ds_config: dict[str, Any] | None) -> dict[str, Any] | None:
        owned = [f"{key} ({_ENGINE_OWNED_DS_KEYS[key]})" for key in _ENGINE_OWNED_DS_KEYS if key in (ds_config or {})]
        if owned:
            raise ValueError(f"training.ds_config may not set {', '.join(owned)}")
        return ds_config
