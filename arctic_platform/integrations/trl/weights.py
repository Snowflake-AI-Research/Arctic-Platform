# Copyright 2026 Snowflake Inc.
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

"""Arctic-backed weight sync for TRL's ``AsyncGRPOTrainer`` (``WeightTransferProtocol``).

The default ``WeightTransferClient`` streams the *trainer's* local parameters into a vLLM server over NCCL. With
Arctic that source is wrong: the trained weights live on Arctic's training engine, not in the trainer process
(whose local module the ``ArcticTrainingClient`` never updates). So this backend ignores the trainer's parameter
iterator and triggers the server-side train -> sampler sync via ``SyncArcticRLClient.sync_weights``, which stages
the sampler wake, copies the weights, and resets the prefix cache itself. ``pause`` / ``resume`` / ``init`` /
``destroy`` are therefore no-ops -- the whole transaction is that one call.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


class ArcticWeightTransfer:
    """Delegates trainer-driven weight sync to the Arctic server.

    Args:
        client: A :class:`~arctic_platform.client.client.SyncArcticRLClient` with both a training and a sampling
            job (the sync copies train -> sampler on the server).
        cuda_ipc: Force the zero-copy CUDA-IPC path (training weights must be on GPU). ``None`` auto-selects it
            under colocation, where it is the cheap path, and leaves it off (NCCL) otherwise.
        low_memory: Stream one parameter at a time on the server to bound peak GPU memory during the sync.
    """

    def __init__(self, client: Any, cuda_ipc: bool | None = None, low_memory: bool = False) -> None:
        self.client = client
        colocate = bool(getattr(client.config.backend, "colocate", False))
        self.cuda_ipc = colocate if cuda_ipc is None else cuda_ipc
        self.low_memory = low_memory

    def init_weight_transfer(self) -> None:
        # Arctic sets up its own transport; nothing to establish trainer-side.
        pass

    def pause(self) -> None:
        # sync_weights() stages the sampler wake/sleep internally.
        pass

    def send_weights(self, iterator: Iterator[tuple[str, torch.Tensor]]) -> None:
        # The trainer's local params are not the source of truth (Arctic's training engine holds the trained
        # weights), so the iterator is deliberately not consumed. On rank 0 that costs nothing; non-main ranks
        # never reach here (AsyncGRPOTrainer only holds a weight_transfer on the main process).
        del iterator
        self.client.sync_weights(cuda_ipc=self.cuda_ipc, low_memory=self.low_memory)

    def resume(self) -> None:
        pass

    def destroy(self) -> None:
        pass
