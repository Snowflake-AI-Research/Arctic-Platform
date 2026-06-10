"""Sequence packing utilities for flash-attention (world_size-free)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

N_TOKENS_PER_PAGE = 256


def _align(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def pack_sequences(data: dict[str, Any]) -> dict[str, Any]:
    """Pack padded [B, S] tensor dict into [1, T] with position_ids for flash-attn.

    No world_size dependency — pack whatever B sequences are given.
    The server splits the batch across DP ranks first, then calls this per shard.
    """
    assert "attention_mask" in data, "Input data must contain 'attention_mask' key."
    attention_mask = data["attention_mask"]
    assert attention_mask.ndim == 2, "Attention mask must be 2D [B, S]."
    B, S = attention_mask.shape
    lens = attention_mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(lens, dim=0, dtype=torch.int32), (1, 0), value=0)
    T = int(cu_seqlens[-1].item())
    position_ids = torch.cat([torch.arange(int(l), device=attention_mask.device) for l in lens.tolist()])
    packed = {}
    for key, value in data.items():
        if key == "attention_mask":
            continue
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == B and value.shape[1] == S:
            packed_tensor = torch.empty((T, *value.shape[2:]), dtype=value.dtype, device=value.device)
            for i in range(B):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                packed_tensor[start:end] = value[i, : end - start]
            packed[key] = packed_tensor.unsqueeze(0)
        else:
            packed[key] = value
    packed["position_ids"] = position_ids.unsqueeze(0)
    packed["cu_seqlens"] = cu_seqlens
    packed["_pack_meta"] = {"B": B, "S": S, "lens": lens}
    return packed


def pad_packed_for_model(packed: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Pad a packed [1, T] dict to the next 256-token boundary for model forward."""
    cu_seqlens = packed["cu_seqlens"]
    T = int(cu_seqlens[-1].item())
    padded_T = _align(T, N_TOKENS_PER_PAGE)
    pad_length = padded_T - T
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    model_kwargs = {}
    for key in ("input_ids", "position_ids"):
        value = packed[key]
        if pad_length > 0 and torch.is_tensor(value) and value.ndim >= 2:
            pad_spec = [0, 0] * (value.ndim - 2) + [0, pad_length]
            value = F.pad(value, list(reversed(pad_spec)), value=0)
        model_kwargs[key] = value
    model_kwargs["cu_seq_lens_q"] = cu_seqlens
    model_kwargs["cu_seq_lens_k"] = cu_seqlens
    model_kwargs["max_length_q"] = max_seqlen
    model_kwargs["max_length_k"] = max_seqlen
    model_kwargs["attention_mask"] = dict(full_attention=None, sliding_attention=None)
    model_kwargs["use_cache"] = False
    return model_kwargs, pad_length


def unpack_sequences(
    packed_tensor: torch.Tensor,
    pack_meta: dict[str, Any],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Unpack a packed [1, T, ...] tensor back to padded [B, S, ...]."""
    B = pack_meta["B"]
    S = pack_meta["S"]
    lens = pack_meta["lens"]
    if packed_tensor.shape[0] == 1 and packed_tensor.ndim >= 2:
        packed_tensor = packed_tensor.squeeze(0)
    cu_seqlens = F.pad(torch.cumsum(lens, dim=0, dtype=torch.int32), (1, 0), value=0)
    out = torch.full((B, S, *packed_tensor.shape[1:]), pad_value, dtype=packed_tensor.dtype, device=packed_tensor.device)
    for i in range(B):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        out[i, : end - start] = packed_tensor[start:end]
    return out
