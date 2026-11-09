"""Broadcast weight transfer: rank 0 sends, all other ranks receive via NCCL broadcast."""

from __future__ import annotations

import torch

from arctic_platform.inference.server.weight_sync.engine import NCCLEngine


class BroadcastNCCLEngine(NCCLEngine):
    """NCCLEngine variant for world_size > 2: send/recv primitives become broadcast(src=0)."""

    def _send_pair(self, idx: int, stream: torch.cuda.Stream) -> None:
        self.nccl.broadcast(self._meta_bufs[idx], src=0, stream=stream)
        self.nccl.broadcast(self._data_bufs[idx], src=0, stream=stream)

    def _recv_pair(self, idx: int, stream: torch.cuda.Stream) -> None:
        self.nccl.broadcast(self._meta_bufs[idx], src=0, stream=stream)
        self.nccl.broadcast(self._data_bufs[idx], src=0, stream=stream)

    def _send_buf(self, buf: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self.nccl.broadcast(buf, src=0, stream=stream)

    def _recv_buf(self, buf: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self.nccl.broadcast(buf, src=0, stream=stream)
