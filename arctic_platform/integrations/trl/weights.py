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

"""Train→sampler sync via ``client.sync_weights`` (trainer params are unused)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


class ArcticWeightTransfer:
    """``client.sync_weights``; ``cuda_ipc=None`` uses IPC when colocated."""

    def __init__(self, client: Any, cuda_ipc: bool | None = None, low_memory: bool = False) -> None:
        self.client = client
        colocate = bool(getattr(client.config.backend, "colocate", False))
        self.cuda_ipc = colocate if cuda_ipc is None else cuda_ipc
        self.low_memory = low_memory

    def init_weight_transfer(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def send_weights(self, iterator: Iterator[tuple[str, torch.Tensor]]) -> None:
        del iterator  # trained weights live on the Arctic training engine
        self.client.sync_weights(cuda_ipc=self.cuda_ipc, low_memory=self.low_memory)

    def resume(self) -> None:
        pass

    def destroy(self) -> None:
        pass
