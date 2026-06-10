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

"""WeightSyncCoordinator -- shared NCCL topology for training-to-inference weight transfer.

This is the single source of truth for sender/receiver IP mapping, GPU IDs,
and NCCL ports.  Both training and inference clients reference the same
coordinator instance so they can coordinate weight sync without duplicating
topology state.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

import torch
from arctic_inference.server.weight_sync.schedule import TransferSchedule
from arctic_inference.server.weight_sync.sender import WeightSender

from arctic_platform.rl.config import WeightSyncConfig

if TYPE_CHECKING:
    from arctic_platform.rl.client import ArcticRLClient

logger = logging.getLogger(__name__)


class WeightSyncCoordinator:
    """Manages the NCCL weight-transfer topology between training GPUs and
    inference replicas.

    Reuses :class:`TransferSchedule` for static assignment of TP workers to
    sender GPUs and :class:`WeightSender` for the actual NCCL data path.

    Parameters
    ----------
    config : WeightSyncConfig
        Topology and connection parameters.
    """

    def __init__(self, config: WeightSyncConfig) -> None:
        self.config = config
        self.schedule = TransferSchedule.build(
            training_sharding=config.training_sharding,
            training_gpus=config.training_gpus,
            inference_replicas=config.inference_replicas,
            inference_tp=config.inference_tp,
        )
        self._senders: dict[int, WeightSender] = {}
        self._server_ready = False
        self._master_addr: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        n_active = min(config.training_gpus, config.inference_replicas * config.inference_tp)
        self._sync_lock = threading.Lock()
        self._sync_http_future: Future | None = None
        self._sync_arrive_count = 0
        self._sync_done_count = 0
        self._n_active = n_active

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    @property
    def sender_ranks(self) -> list[int]:
        """Training GPU ranks that are actively sending."""
        return self.schedule.active_sender_ranks

    @property
    def num_groups(self) -> int:
        return self.schedule.num_groups

    # ------------------------------------------------------------------
    # Sender management
    # ------------------------------------------------------------------

    def get_or_create_sender(
        self,
        rank: int,
        master_addr: str,
        device: torch.device,
        *,
        reverse: bool = False,
    ) -> WeightSender:
        """Get or lazily create a :class:`WeightSender` for *rank*."""
        if rank not in self._senders:
            group = self.schedule.groups[rank]
            self._senders[rank] = WeightSender(
                group=group,
                schedule=self.schedule,
                master_addr=master_addr,
                base_port=self.config.base_port,
                device=device,
                bucket_size=self.config.bucket_size,
                reverse=reverse,
            )
        return self._senders[rank]

    def sync_weights(
        self,
        rank: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        client: ArcticRLClient,
        master_addr: str | None = None,
        device: torch.device | None = None,
        direct: bool = False,
    ) -> dict[str, Any]:
        """Send *weights* from training *rank* to its assigned inference targets.

        Thread-safe: all active ranks should call this concurrently.  The
        first arrival fires ONE HTTP request to the server (which dispatches
        all receivers); every rank then pushes its own data through NCCL.
        The last rank to finish resets the shared state for the next round.

        On the first call for a given *rank*, the NCCL sender is created
        automatically (lazy setup).  *master_addr* and *device* are required
        on that first call; subsequent calls for the same rank can omit them.
        """
        if rank not in self._senders:
            if master_addr is None:
                raise ValueError("master_addr is required on the first sync_weights call for each rank")
            if device is None:
                device = torch.device("cuda", rank)
            self.get_or_create_sender(rank, master_addr, device)

        if self._master_addr is None:
            if master_addr is None:
                raise ValueError("master_addr is required on the first sync_weights call")
            self._master_addr = master_addr

        sender = self._senders[rank]

        with self._sync_lock:
            self._sync_arrive_count += 1
            if self._sync_arrive_count == 1:
                groups = self._build_groups()
                self._sync_http_future = self._executor.submit(
                    client.update_weights,
                    groups,
                    bucket_size=self.config.bucket_size,
                    direct_mode=direct,
                )
            http_future = self._sync_http_future

        try:
            result = sender.send(weights, direct=direct)
            http_future.result()
        finally:
            with self._sync_lock:
                self._sync_done_count += 1
                if self._sync_done_count == self._n_active:
                    self._sync_arrive_count = 0
                    self._sync_done_count = 0
                    self._sync_http_future = None

        return result

    # ------------------------------------------------------------------
    # Inference-side NCCL setup
    # ------------------------------------------------------------------

    def ensure_server_ready(
        self,
        client: ArcticRLClient,
        master_addr: str,
    ) -> None:
        """Idempotent: call :meth:`prepare_inference_server` at most once.

        Safe to call from every training rank -- only the first call
        performs the HTTP request; subsequent calls are no-ops.
        """
        if self._server_ready:
            return
        self.prepare_inference_server(client, master_addr)
        self._server_ready = True

    def prepare_inference_server(
        self,
        client: ArcticRLClient,
        master_addr: str,
    ) -> dict[str, Any]:
        """Tell the inference server to create NCCL receiver engines.

        Calls ``/sync_weights`` with ``engine_only=True`` so the server
        performs the NCCL rendezvous without actually receiving data.
        """
        self._master_addr = master_addr
        groups = self._build_groups()
        return client.update_weights(groups, engine_only=True)

    def _build_groups(self) -> list[dict[str, Any]]:
        """Build the group descriptor list used by ``/sync_weights``."""
        return [
            {
                "group_id": g.group_id,
                "master_addr": self._master_addr,
                "master_port": self.config.base_port + g.group_id * self.config.inference_tp,
                "world_size": g.world_size,
                "replica_ids": g.replica_ids,
            }
            for g in self.schedule.groups
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy all NCCL sender connections and reset state."""
        for sender in self._senders.values():
            try:
                sender.destroy()
            except Exception:
                logger.warning("Failed to destroy sender", exc_info=True)
        self._senders.clear()
        self._server_ready = False
        self._master_addr = None
        self._executor.shutdown(wait=False)
