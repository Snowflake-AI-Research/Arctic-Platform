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

"""DSS job-creation passthrough for correctness, using the public Cortex data plane."""

from __future__ import annotations

from copy import deepcopy

from pydantic import model_validator
from typing_extensions import Self

from arctic_platform._dependency_groups import require_any_dep_group
from arctic_platform.client import ArcticClient
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client.transport import initialize_or_cleanup


class CortexTrainingConfig(ArcticClientConfig):
    """Keep DSS's extra creation fields outside the production config adapter."""

    hardware: str | None = None
    source_checkpoint_info: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        if self.source_checkpoint_info is not None:
            if not isinstance(self.backend, CortexConfig):
                raise ValueError("source_checkpoint_info requires CortexConfig")
            source = self.source_checkpoint_info
            if set(source) != {"checkpoint_id", "source_job_id"} or not all(source.values()):
                raise ValueError("source_checkpoint_info requires checkpoint_id and source_job_id")
        return self

    def _cortex_training_sub_job(self) -> dict:
        sub = super()._cortex_training_sub_job()
        extra = deepcopy(self.training.ds_worker_config or {})
        activation = extra.pop("enable_gradient_checkpointing", None)
        if activation is not None:
            extra["activation_checkpointing"] = activation
        debug = dict(extra.get("debug") or {})
        debug.setdefault("full_determinism", self.training.full_determinism)
        extra["debug"] = debug
        sub["training_config"] = {**extra, **sub["training_config"]}
        if self.source_checkpoint_info is not None:
            sub["source_checkpoint_info"] = dict(self.source_checkpoint_info)
        return sub

    def _cortex_inference_sub_job(self, job_type: str, n_gpus: int) -> dict:
        sub = super()._cortex_inference_sub_job(job_type, n_gpus)
        if self.source_checkpoint_info is not None:
            sub["source_checkpoint_info"] = dict(self.source_checkpoint_info)
        return sub


class CorrectnessClient(ArcticClient):
    def __init__(self, config: ArcticClientConfig) -> None:
        if not isinstance(config, CortexTrainingConfig) or not isinstance(config.backend, CortexConfig):
            super().__init__(config)
            return
        require_any_dep_group("cortex")
        from .cortex_transport import CorrectnessCortexTransport

        self.config = config
        self.transport = CorrectnessCortexTransport(config.backend)
        self.jobs = initialize_or_cleanup(self.transport, client_config=config)
