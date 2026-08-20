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

SFT is blocking-only, so `ArcticSFTClient` extends `SyncArcticClient` (there is
no async twin, unlike RL). The only thing genuinely SFT-specific is the loss
contract on the two forward ops; everything else is inherited.
"""

from __future__ import annotations

from typing import Any

from arctic_platform.client.base import SyncArcticClient
from arctic_platform.client.config import ArcticClientConfig


def _sft_body(batch: dict) -> dict:
    """The SFT loss contract: bodies carry ``processing`` + ``meta`` unless the caller set them."""
    body = dict(batch)
    body.setdefault("processing", {"loss_fn": "sft"})
    body.setdefault("meta", {})
    return body


class ArcticSFTClient(SyncArcticClient):
    """SFT frontend: forward bodies default to the ``sft`` loss unless the caller overrides."""

    def fwd_bwd(self, batch: dict, processing: dict | None = None, router_replay: Any = None) -> dict:
        return super().fwd_bwd(_sft_body(batch), processing, router_replay)

    def fwd_no_grad(self, batch: dict, reference_model: bool = False) -> dict:
        return super().fwd_no_grad(_sft_body(batch), reference_model)


def create_arctic_sft_client(config: ArcticClientConfig) -> ArcticSFTClient:
    return ArcticSFTClient(config)
