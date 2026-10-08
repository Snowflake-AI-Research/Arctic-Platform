from __future__ import annotations

import torch

from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids


def get_cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Packed-sequence boundaries plus max segment length, matching PrimeRL's helper."""
    cu_seqlens = cu_seqlens_from_position_ids(position_ids)
    if cu_seqlens.numel() <= 1:
        return cu_seqlens, 0
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return cu_seqlens, max_seqlen
