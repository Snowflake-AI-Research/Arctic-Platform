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
"""The SFT frontend.

SFT is blocking-only, so `ArcticSFTClient` extends `ArcticClient` (there is
no async twin, unlike RL). Only two things are genuinely SFT-specific: the loss
contract on the two forward ops, and the stricter job requirements on
`ArcticSFTClientConfig`. Everything else is inherited.
"""

from __future__ import annotations

from typing import Any

from pydantic import model_validator
from typing_extensions import Self

from arctic_platform.client.base import ArcticClient
from arctic_platform.client.config import ArcticClientConfig


class ArcticSFTClientConfig(ArcticClientConfig):
    """The shared config plus SFT's job requirements. Adds no fields.

    SFT always trains, so unlike RL it cannot run with `training_gpus=0` (RL uses
    that for sampling-only clients) and the server needs a checkpoint dir up
    front. Both are enforced here so a bad config fails before any job or GPU is
    claimed, rather than at first op or at save time.
    """

    @model_validator(mode="after")
    def _require_training_job(self) -> Self:
        if self.training_job_id is not None:  # reconnecting; the job already exists
            return self
        if self.training_gpus <= 0:
            raise ValueError("training_gpus must be > 0 (or set training_job_id to reconnect)")
        if not self.training.checkpoint_path:
            raise ValueError(
                "training.checkpoint_path is required to start a new training job "
                "(the server needs it to save checkpoints); set training_job_id to "
                "reconnect to an existing job instead."
            )
        return self


def _sft_body(batch: dict) -> dict:
    """The SFT loss contract: bodies carry ``processing`` + ``meta`` unless the caller set them."""
    body = dict(batch)
    body.setdefault("processing", {"loss_fn": "sft"})
    body.setdefault("meta", {})
    return body


class ArcticSFTClient(ArcticClient):
    """SFT frontend: forward bodies default to the ``sft`` loss unless the caller overrides."""

    def fwd_bwd(self, batch: dict, processing: dict | None = None, router_replay: Any = None) -> dict:
        return super().fwd_bwd(_sft_body(batch), processing, router_replay)

    def fwd_no_grad(self, batch: dict, reference_model: bool = False) -> dict:
        return super().fwd_no_grad(_sft_body(batch), reference_model)
