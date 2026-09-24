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
"""Model-agnostic native-FP8 helpers shared by Qwen and GLM MoE loaders.

Keep this module free of qwen35/glm52 package imports at module level so
either family can use FP8 LoRA without a layering inversion. The CUDA 1x128
activation-quant kernel is imported lazily from the shared kernels package.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

_FP8_BLOCK = 128
_deep_gemm_mod = None
_deep_gemm_checked = False

# Frozen FP8 block scales must stay FP32 through DeepSpeed's mixed-precision
# cast. They are the only thing restoring magnitude for e4m3 codes, and they
# cannot become buffers: DeepSpeed tags EP sharding on parameters, and weight
# sync gathers them via ``named_parameters()``.
KEEP_FP32_ATTR = "_dss_keep_fp32"


def mark_keep_fp32(param: nn.Parameter) -> nn.Parameter:
    """Exempt ``param`` from the bf16/fp16 cast applied at DeepSpeed init."""
    setattr(param, KEEP_FP32_ATTR, True)
    return param


def keeps_fp32(tensor: Tensor) -> bool:
    return bool(getattr(tensor, KEEP_FP32_ATTR, False))


def carry_keep_fp32(src: Tensor, dst: nn.Parameter) -> nn.Parameter:
    """Re-mark ``dst`` when it replaces ``src``."""
    if keeps_fp32(src):
        mark_keep_fp32(dst)
    return dst


def _deep_gemm():
    """Standalone DeepGEMM wheel, or vLLM's vendored 2.5.0 if the wheel was uninstalled.

    Training wants ``deep_gemm`` from ``dss-pre-built-wheels``
    (``deep_gemm-2.5.0+891d57b``). Sampling uninstalls that top-level package so
    it cannot shadow ``vllm.third_party.deep_gemm``; fall back to the vendored copy.
    """
    global _deep_gemm_mod, _deep_gemm_checked
    if _deep_gemm_checked:
        return _deep_gemm_mod
    _deep_gemm_checked = True
    try:
        import deep_gemm as dg  # type: ignore[import-not-found]
    except ImportError:
        try:
            from vllm.third_party import deep_gemm as dg  # type: ignore[import-not-found]
        except ImportError:
            dg = None
    _deep_gemm_mod = dg
    return _deep_gemm_mod


def _require_deep_gemm():
    """DeepGEMM is not optional on CUDA.

    Dequantizing instead would silently restore the full BF16 weight this path
    exists to avoid -- 42 GiB/GPU of experts on GLM-5.3 at EP=16 -- so a broken
    install would surface as an OOM rather than a missing dependency.
    """
    dg = _deep_gemm()
    if dg is None:
        raise RuntimeError(
            "FP8 training requires DeepGEMM: install `deep_gemm` from "
            "dss-pre-built-wheels, or run where `vllm.third_party.deep_gemm` "
            "is importable"
        )
    return dg


def _require_fp8_state(weight: Tensor, scale: Tensor, what: str) -> None:
    """A block scale means ``weight`` holds e4m3 codes and ``scale`` is FP32.

    Repairing either here would paper over the DeepSpeed/PEFT dtype cast that
    ``_preserve_fp8_params_through_ds_cast`` exists to prevent: recasting the
    weight doubles frozen FP8 storage, and the scale keeps only 8 mantissa bits
    once cast, which upcasting at use cannot restore.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"{what} FP8 weight arrived as {weight.dtype} instead of "
            "float8_e4m3fn; frozen FP8 parameters were cast (see "
            "_preserve_fp8_params_through_ds_cast)"
        )
    if scale.dtype != torch.float32:
        raise TypeError(
            f"{what} FP8 block scale arrived as {scale.dtype} instead of "
            "float32; the scale is the only factor restoring magnitude for an "
            "e4m3 code (see _preserve_fp8_params_through_ds_cast)"
        )


def _disable_ue8m0_cast() -> bool:
    # Hopper (SM90) uses FP32 scaling factors; Blackwell uses UE8M0.
    return torch.cuda.get_device_capability()[0] < 10


def _quantize_act_fp8_1x128(x: Tensor) -> tuple[Tensor, Tensor]:
    """Dynamic 1×128 e4m3 activation quant (HF ``activation_scheme=dynamic``)."""
    x = x.contiguous()
    group = _FP8_BLOCK
    if x.shape[-1] % group != 0:
        raise ValueError(f"FP8 activation quant expects K divisible by {group}, got {tuple(x.shape)}")
    if x.is_cuda:
        from arctic_platform.model.implementations.kernels.fp8_quant import per_token_group_quant_fp8

        q, scale = per_token_group_quant_fp8(x, group_size=group, use_ue8m0=False)
    else:
        rows, cols = x.shape
        blocks = x.view(rows, cols // group, group)
        amax = blocks.float().abs().amax(dim=-1).clamp(min=1e-4)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = (amax / fp8_max).contiguous()
        q = (blocks.float() / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).view(rows, cols)
    dg = _deep_gemm()
    if dg is not None and q.is_cuda:
        scale = dg.get_mn_major_tma_aligned_tensor(scale)
    return q, scale


def original_param_and_lora_delta(module: nn.Module, name: str) -> tuple[Tensor, Tensor | None]:
    """Frozen FP8 parameter plus PEFT ``ParamWrapper`` delta, if any.

    PEFT's ``_LoraParameterProxy`` does ``fp8_codes.to(bf16) + dW`` and recasts to
    e4m3. That adds a true-weight LoRA delta onto quantized codes and leaves
    ``weight_scale_inv`` unchanged, so the next DeepGEMM saturates. Use the
    original e4m3 tensor for the GEMM and add ``dW @ x`` in BF16 instead.
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or name not in parametrizations:
        return getattr(module, name), None
    plist = parametrizations[name]
    original = plist.original
    delta = None
    for proxy in plist:
        dw = getattr(proxy, "delta_weight", None)
        if dw is None:
            continue
        delta = dw if delta is None else delta + dw
    return original, delta


def _fp8_linear_fwd(x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
    """``x @ weight.T`` with block-scaled FP8 weights. ``x`` is 2D."""
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[0]))
    _require_fp8_state(weight, scale, "attention/dense")
    if not x.is_cuda:
        restored = dequantize_from_fp8_blockwise(weight, scale, dtype=x.dtype)
        return F.linear(x, restored)
    dg = _require_deep_gemm()
    act, act_scale = _quantize_act_fp8_1x128(x)
    out = torch.empty(x.shape[0], weight.shape[0], device=x.device, dtype=torch.bfloat16)
    dg.fp8_gemm_nt(
        (act, act_scale),
        (weight, scale),
        out,
        disable_ue8m0_cast=_disable_ue8m0_cast(),
    )
    return out if x.dtype == torch.bfloat16 else out.to(dtype=x.dtype)


def _fp8_linear_bwd_input(
    grad_output: Tensor,
    weight: Tensor,
    scale: Tensor,
    block_size: int,
    row_chunk_blocks: int = 4,
) -> Tensor:
    """Compute ``grad_output @ dequant(weight)`` without a full FP32 weight."""
    rows, cols = weight.shape
    if grad_output.shape[0] == 0:
        return grad_output.new_zeros((0, cols), dtype=torch.float32)
    row_chunk = block_size * row_chunk_blocks
    dx = torch.zeros(
        grad_output.shape[0],
        cols,
        device=grad_output.device,
        dtype=torch.float32,
    )
    for row0 in range(0, rows, row_chunk):
        row1 = min(row0 + row_chunk, rows)
        scale0 = row0 // block_size
        scale1 = (row1 + block_size - 1) // block_size
        restored = dequantize_from_fp8_blockwise(
            weight[row0:row1],
            scale[scale0:scale1],
            block_size=block_size,
            dtype=torch.float32,
        )
        dx.addmm_(grad_output[:, row0:row1], restored)
    return dx


class _BlockFp8LinearFn(torch.autograd.Function):
    """FP8 GEMM in forward; FP32 ``dX = dY @ W`` in backward (frozen weights)."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, scale: Tensor, block_size: int) -> Tensor:
        ctx.save_for_backward(weight, scale)
        ctx.x_shape = x.shape
        ctx.go_dtype = x.dtype
        ctx.block_size = block_size
        y = _fp8_linear_fwd(x.reshape(-1, x.shape[-1]), weight, scale)
        return y.view(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        weight, scale = ctx.saved_tensors
        go = grad_output.reshape(-1, grad_output.shape[-1]).float()
        dx = _fp8_linear_bwd_input(go, weight, scale, ctx.block_size)
        return dx.to(dtype=grad_output.dtype).view(*ctx.x_shape), None, None, None


def fp8_linear(x: Tensor, weight: Tensor, scale: Tensor | None, block_size: int | None) -> Tensor:
    """Linear with optional block-FP8 weights. Gradients flow to ``x`` only."""
    if scale is None or block_size is None:
        return F.linear(x, weight)
    # Dequant stays out of the autograd graph: a full SwiGLU dequant graph OOMs
    # a 78-layer model.
    return _BlockFp8LinearFn.apply(x, weight, scale, block_size)


def fp8_linear_with_lora_delta(
    x: Tensor,
    weight: Tensor,
    scale: Tensor | None,
    block_size: int | None,
    delta: Tensor | None,
) -> Tensor:
    """FP8 GEMM on frozen ``weight``, plus BF16 LoRA ``x @ delta.T`` when ``delta`` is set."""
    y = fp8_linear(x, weight, scale, block_size)
    extra = expert_lora_output(x, fused_delta=delta, ab=None, expert_idx=0)
    return y if extra is None else y + extra


def expert_lora_output(
    x: Tensor,
    *,
    fused_delta: Tensor | None,
    ab: tuple[Tensor, Tensor, int, float] | None,
    expert_idx: int,
) -> Tensor | None:
    """LoRA contribution for one expert without allocating fused ``[E, out, in]``.

    PEFT ``ParamWrapper.get_delta_weight`` materializes ``E * 2048 * 6144 * 2`` bytes
    (384 MiB at EP=16). That allocation is what OOMs GLM-5.3 Search-R1 backward.
    The A/B path is ``x @ A_e.T @ B_e.T`` with ``A_e`` ``[r, in]`` and ``B_e``
    ``[out, r]`` (views into the existing adapter parameters).
    """
    if fused_delta is not None:
        return F.linear(x, fused_delta.to(dtype=x.dtype))
    if ab is None:
        return None
    weight_a, weight_b, rank, scaling = ab
    if x.shape[0] == 0:
        return x.new_zeros(0, weight_b.shape[0])
    # PEFT ParamWrapper layout (``_did_swap_in_out_features``): A is
    # ``[E * r, in]``, B is ``[out, E * r]``. Same views as
    # ``get_delta_weight``'s ``eoi`` einsum, without allocating ``[E, out, in]``.
    a_e = weight_a.reshape(weight_a.shape[0] // rank, rank, weight_a.shape[1])[expert_idx]
    b_e = weight_b.reshape(weight_b.shape[0], rank, weight_b.shape[1] // rank)[:, :, expert_idx]
    return F.linear(F.linear(x, a_e), b_e) * scaling


def fp8_linear_with_expert_lora(
    x: Tensor,
    weight: Tensor,
    scale: Tensor | None,
    block_size: int | None,
    fused_delta: Tensor | None,
    ab: tuple[Tensor, Tensor, int, float] | None,
    expert_idx: int,
) -> Tensor:
    """FP8 GEMM plus unfused expert LoRA (or a pre-sliced fused delta)."""
    y = fp8_linear(x, weight, scale, block_size)
    extra = expert_lora_output(x, fused_delta=fused_delta, ab=ab, expert_idx=expert_idx)
    return y if extra is None else y + extra


def _pad_tokens_for_deepgemm(
    x: Tensor,
    num_tokens_per_expert: Tensor,
    align: int,
) -> tuple[Tensor, Tensor, list[int], int, list[int]]:
    counts = num_tokens_per_expert.tolist()
    n_real = int(sum(counts))
    n_tail = x.shape[0] - n_real
    pieces: list[Tensor] = []
    ids: list[Tensor] = []
    padded_counts: list[int] = []
    offset = 0
    for expert_idx, n in enumerate(counts):
        chunk = x[offset : offset + n]
        pad = (align - n % align) % align if n else 0
        if pad:
            chunk = torch.cat([chunk, chunk.new_zeros(pad, chunk.shape[-1])], dim=0)
        pieces.append(chunk)
        ids.append(torch.full((chunk.shape[0],), expert_idx, device=x.device, dtype=torch.int32))
        padded_counts.append(chunk.shape[0])
        offset += n
    if not pieces:
        empty = x.new_empty(0, x.shape[-1])
        return empty, x.new_empty(0, dtype=torch.int32), counts, n_tail, padded_counts
    return torch.cat(pieces, dim=0), torch.cat(ids, dim=0), counts, n_tail, padded_counts


def _unpad_tokens_from_deepgemm(
    y: Tensor,
    counts: list[int],
    padded_counts: list[int],
    n_tail: int,
) -> Tensor:
    pieces = []
    offset = 0
    for n, padded in zip(counts, padded_counts):
        pieces.append(y[offset : offset + n])
        offset += padded
    out = torch.cat(pieces, dim=0) if pieces else y.new_empty(0, y.shape[-1])
    if n_tail:
        out = torch.vstack((out, out.new_zeros((n_tail, out.shape[-1]))))
    return out


class _GroupedBlockFp8MmFn(torch.autograd.Function):
    """Grouped FP8 NT GEMM (DeepGEMM contiguous layout). Gradients flow to ``x``."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        scale: Tensor,
        num_tokens_per_expert: Tensor,
        block_size: int,
    ) -> Tensor:
        ctx.save_for_backward(weight, scale, num_tokens_per_expert)
        ctx.x_shape = x.shape
        ctx.block_size = block_size
        _require_fp8_state(weight, scale, "expert")
        if not x.is_cuda:
            restored = dequantize_from_fp8_blockwise(weight, scale, dtype=x.dtype)
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            ctx.counts = num_tokens_per_expert.tolist()
            ctx.n_tail = 0
            return torch._grouped_mm(x.bfloat16(), restored.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
        dg = _require_deep_gemm()
        align = int(dg.get_mk_alignment_for_contiguous_layout())
        xp, m_indices, counts, n_tail, padded_counts = _pad_tokens_for_deepgemm(x, num_tokens_per_expert, align)
        ctx.counts = counts
        ctx.n_tail = n_tail
        if xp.numel() == 0:
            return x.new_zeros(x.shape[0], weight.shape[1])
        act, act_scale = _quantize_act_fp8_1x128(xp)
        out = torch.empty(xp.shape[0], weight.shape[1], device=x.device, dtype=torch.bfloat16)
        dg.m_grouped_fp8_gemm_nt_contiguous(
            (act, act_scale),
            (weight, scale),
            out,
            m_indices,
            disable_ue8m0_cast=_disable_ue8m0_cast(),
        )
        if x.dtype != torch.bfloat16:
            out = out.to(dtype=x.dtype)
        return _unpad_tokens_from_deepgemm(out, counts, padded_counts, n_tail)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        weight, scale, num_tokens_per_expert = ctx.saved_tensors
        counts = num_tokens_per_expert.tolist()
        n_real = int(sum(counts))
        go = grad_output[:n_real].float()
        chunks = torch.split(go, counts, dim=0) if n_real else []
        dx_parts = [
            _fp8_linear_bwd_input(chunk, weight[i], scale[i], ctx.block_size) for i, chunk in enumerate(chunks)
        ]
        dx = torch.cat(dx_parts, dim=0) if dx_parts else grad_output.new_zeros(0, weight.shape[-1])
        n_tail = grad_output.shape[0] - n_real
        if n_tail:
            dx = torch.vstack((dx, dx.new_zeros((n_tail, dx.shape[-1]))))
        return dx.to(dtype=grad_output.dtype).view(*ctx.x_shape), None, None, None, None


def grouped_fp8_mm(
    x: Tensor,
    weight: Tensor,
    scale: Tensor | None,
    num_tokens_per_expert: Tensor,
    block_size: int | None,
) -> Tensor:
    """Grouped ``x_e @ W_e.T`` for tokens packed by expert.

    Uses DeepGEMM ``m_grouped_fp8_gemm_nt_contiguous`` on CUDA when weights are
    FP8; otherwise dequantizes and falls back to ``torch._grouped_mm``.
    """
    if scale is None or block_size is None:
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        return torch._grouped_mm(x.bfloat16(), weight.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return _GroupedBlockFp8MmFn.apply(x, weight, scale, num_tokens_per_expert, block_size)


def fp8_weight_block_size(config: Any) -> int | None:
    """Return the HF block size when ``config`` is a finegrained FP8 checkpoint."""
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return None
    if not isinstance(qc, dict):
        qc = getattr(qc, "to_dict", lambda: {})()
    if qc.get("quant_method") != "fp8":
        return None
    weight_block_size = qc.get("weight_block_size") or [128, 128]
    return int(weight_block_size[0])


def quantize_to_fp8_blockwise(weight: Tensor, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize a 2D tensor to FP8 e4m3 with per-block scales."""
    if weight.ndim != 2:
        raise ValueError(f"FP8 quantization expects a 2D tensor, got shape={tuple(weight.shape)}")

    rows, cols = weight.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size

    if pad_rows or pad_cols:
        padded = torch.zeros(
            rows + pad_rows,
            cols + pad_cols,
            dtype=weight.dtype,
            device=weight.device,
        )
        padded[:rows, :cols] = weight
    else:
        padded = weight.contiguous()

    padded_rows, padded_cols = padded.shape
    blocks = padded.view(
        padded_rows // block_size,
        block_size,
        padded_cols // block_size,
        block_size,
    ).permute(0, 2, 1, 3)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    max_abs = blocks.float().abs().amax(dim=(2, 3))
    scales = (max_abs / fp8_max).clamp(min=1e-12)
    blocks_fp8 = (blocks.float() / scales[:, :, None, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    quantized = blocks_fp8.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols].contiguous()
    return quantized, scales.float().contiguous()


def fp8_scale_shape(weight_shape: tuple[int, ...], block_size: int | None) -> tuple[int, ...]:
    """Scale-tensor shape for a 2D or stacked-expert 3D FP8 weight."""
    bs = block_size or 1
    if len(weight_shape) == 2:
        rows, cols = weight_shape
        return ((rows + bs - 1) // bs, (cols + bs - 1) // bs)
    if len(weight_shape) == 3:
        experts, rows, cols = weight_shape
        return (experts, (rows + bs - 1) // bs, (cols + bs - 1) // bs)
    raise ValueError(f"FP8 scale shape expects 2D or 3D weights, got {weight_shape}")


def dequantize_from_fp8_blockwise(
    weight: Tensor,
    scale: Tensor,
    block_size: int = 128,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Undo ``quantize_to_fp8_blockwise``. ``scale`` is the multiply-to-restore factor (HF ``weight_scale_inv``)."""
    if weight.ndim not in (2, 3):
        raise ValueError(f"FP8 dequantization expects a 2D or 3D tensor, got shape={tuple(weight.shape)}")

    leading = (weight.shape[0],) if weight.ndim == 3 else ()
    rows, cols = weight.shape[-2:]
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    padded_rows = rows + pad_rows
    padded_cols = cols + pad_cols
    if pad_rows or pad_cols:
        padded = weight.new_zeros((*leading, padded_rows, padded_cols))
        padded[..., :rows, :cols] = weight
    else:
        padded = weight.contiguous()

    n_row = padded_rows // block_size
    n_col = padded_cols // block_size
    if tuple(scale.shape[-2:]) != (n_row, n_col):
        raise ValueError(
            f"FP8 scale shape {tuple(scale.shape)} does not match "
            f"weight {tuple(weight.shape)} with block_size={block_size} (expected (*, {n_row}, {n_col}))"
        )
    if leading and scale.shape[0] != leading[0]:
        raise ValueError(f"FP8 scale expert dim {scale.shape[0]} does not match weight expert dim {leading[0]}")

    if leading:
        blocks = padded.view(*leading, n_row, block_size, n_col, block_size).permute(0, 1, 3, 2, 4)
        restored = (blocks.float() * scale.float()[..., None, None]).permute(0, 1, 3, 2, 4)
        restored = restored.reshape(*leading, padded_rows, padded_cols)[..., :rows, :cols]
    else:
        blocks = padded.view(n_row, block_size, n_col, block_size).permute(0, 2, 1, 3)
        restored = (
            (blocks.float() * scale.float()[:, :, None, None])
            .permute(0, 2, 1, 3)
            .reshape(padded_rows, padded_cols)[:rows, :cols]
        )
    return restored.to(dtype or torch.bfloat16)


def _peft_lora_delta(module: nn.Module) -> Tensor | None:
    """PEFT ``LoraLinear.get_delta_weight`` lives on the wrapper, not ``base_layer``."""
    get_delta = getattr(module, "get_delta_weight", None)
    if get_delta is None or not hasattr(module, "lora_A"):
        return None
    adapter = getattr(module, "active_adapter", "default")
    names = adapter if isinstance(adapter, (list, tuple)) else (adapter,)
    delta = None
    for name in names:
        extra = get_delta(name)
        delta = extra if delta is None else delta + extra
    return delta


def compute_weight(linear: nn.Module, dtype: torch.dtype | None = None) -> Tensor:
    """Weight tensor used for GEMM / views. Dequantizes ``BlockFp8Linear`` in place of ``.weight``."""
    peft_delta = _peft_lora_delta(linear)
    while hasattr(linear, "base_layer"):
        linear = linear.base_layer
        if peft_delta is None:
            peft_delta = _peft_lora_delta(linear)
    if isinstance(linear, BlockFp8Linear):
        original, delta = original_param_and_lora_delta(linear, "weight")
        restored = dequantize_from_fp8_blockwise(
            original, linear.weight_scale_inv, block_size=linear.block_size, dtype=dtype
        )
        if delta is not None:
            restored = restored + delta.to(dtype=restored.dtype)
        if peft_delta is not None:
            restored = restored + peft_delta.to(dtype=restored.dtype)
        return restored
    weight = linear.weight
    if peft_delta is not None:
        weight = weight + peft_delta.to(dtype=weight.dtype)
    return weight if dtype is None else weight.to(dtype)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    fp8_block_size: int | None = None,
) -> nn.Module:
    if fp8_block_size is None:
        return nn.Linear(in_features, out_features, bias=bias)
    if bias:
        raise NotImplementedError("BlockFp8Linear does not support bias")
    return BlockFp8Linear(in_features, out_features, block_size=fp8_block_size)


class BlockFp8Linear(nn.Linear):
    """``nn.Linear`` that stores block-scaled FP8 weights and runs an FP8 GEMM.

    Subclasses ``nn.Linear`` so PEFT LoRA still wraps attention projections.
    Frozen e4m3 weights stay in HBM. Forward is DeepGEMM ``fp8_gemm_nt``
    (1×128 dynamic activations × 128×128 weights) on Hopper; backward dequantizes
    only to form ``dX``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: int = 128,
        device=None,
    ) -> None:
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.bias = None
        n_row = (out_features + block_size - 1) // block_size
        n_col = (in_features + block_size - 1) // block_size
        self.weight_scale_inv = mark_keep_fp32(
            nn.Parameter(
                torch.empty(n_row, n_col, device=device, dtype=torch.float32),
                requires_grad=False,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        weight, delta = original_param_and_lora_delta(self, "weight")
        return fp8_linear_with_lora_delta(x, weight, self.weight_scale_inv, self.block_size, delta)
