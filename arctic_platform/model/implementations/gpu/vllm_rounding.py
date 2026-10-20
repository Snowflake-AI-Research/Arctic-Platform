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
"""Operations evaluated in fp32 the way vLLM evaluates them, shared by the Qwen3.5 trainer models.

``FusedAddRMSNorm`` is a residual add followed by a (1 + weight) RMSNorm, both in fp32;
``FusedAddRMSNormFp32Residual`` returns the sum unrounded. ``rope_rotary_slice_fp32`` applies a partial rotary
embedding in fp32. ``rms_norm_gated_fp32`` is the GatedDeltaNet output norm in fp32, rounded once, and
``round_gated_norms_once`` makes the stock transformers output norms of a model use it.
"""

from __future__ import annotations

import importlib
import types

import torch
import torch.nn.functional as F
from torch import nn

try:
    from fla.modules import FusedRMSNormGated
    from fla.modules.fused_norm_gate import rms_norm_gated as fla_rms_norm_gated
except ImportError:
    FusedRMSNormGated = None  # type: ignore
    fla_rms_norm_gated = None  # type: ignore

# The transformers GatedDeltaNet output norms whose forward normalizes in fp32, rounds to the input dtype, multiplies by
# the weight in the input dtype and rounds again; and the GatedDeltaNet layers that hold them as ``norm``.
_STOCK_GATED_NORMS = (
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNormGated"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeRMSNormGated"),
    ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextRMSNormGated"),
    ("transformers.models.qwen4_exp.modeling_qwen4_exp", "Qwen4ExpTextRMSNormGated"),
)
_STOCK_GATED_DELTA_NETS = (
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5GatedDeltaNet"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeGatedDeltaNet"),
    ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextGatedDeltaNet"),
)


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Return fp32 for the low-precision training dtypes and ``x``'s own dtype otherwise.

    Keeping fp64 inputs in fp64 lets ``torch.autograd.gradcheck`` exercise :class:`FusedAddRMSNorm`.
    """
    if x.dtype in (torch.bfloat16, torch.float16):
        return torch.float32
    return x.dtype


def _fused_add_rms_norm_forward(ctx, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    compute_dtype = _compute_dtype(x)
    summed = x.to(compute_dtype) + residual.to(compute_dtype)
    rstd = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps)
    out = summed * rstd * (1.0 + weight.to(compute_dtype))
    ctx.save_for_backward(x, residual, weight, rstd)
    return out.to(x.dtype), summed


def _fused_add_rms_norm_backward(ctx, grad_out: torch.Tensor, grad_residual: torch.Tensor):
    x, residual, weight, rstd = ctx.saved_tensors
    compute_dtype = _compute_dtype(x)
    summed = x.to(compute_dtype) + residual.to(compute_dtype)
    normed = summed * rstd
    scale = 1.0 + weight.to(compute_dtype)

    grad = grad_out.to(compute_dtype)
    grad_weight = (grad * normed).reshape(-1, weight.shape[-1]).sum(0).to(weight.dtype)

    # Gradient through normed = summed * rstd(summed), with rstd = (mean(summed**2) + eps) ** -0.5.
    grad_normed = grad * scale
    hidden_size = summed.shape[-1]
    grad_summed = rstd * (grad_normed - normed * (grad_normed * normed).sum(-1, keepdim=True) / hidden_size)
    grad_summed = grad_summed + grad_residual.to(compute_dtype)
    return grad_summed.to(x.dtype), grad_summed.to(residual.dtype), grad_weight, None


class FusedAddRMSNorm(torch.autograd.Function):
    """Residual add followed by (1+weight) RMSNorm, both evaluated in fp32.

    This follows vLLM's ``fused_add_rms_norm``: the sum of ``x`` and ``residual`` is formed in fp32, and the
    variance and the normalized value are computed from that fp32 sum rather than from a sum already rounded to
    the input dtype. Both outputs, the normalized value and the sum that becomes the next residual, are returned
    in the input dtype.

    For backward it saves only the inputs as given (``x``, ``residual`` and ``weight``) and the per-row
    ``rstd``, and recomputes the fp32 sum from them, instead of keeping the fp32 sum and the fp32 normalized
    value alive until backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
        out, summed = _fused_add_rms_norm_forward(ctx, x, residual, weight, eps)
        return out, summed.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_residual: torch.Tensor):
        return _fused_add_rms_norm_backward(ctx, grad_out, grad_residual)


class FusedAddRMSNormFp32Residual(torch.autograd.Function):
    """``FusedAddRMSNorm`` that returns the sum, the next residual, unrounded: in fp32 for 16-bit inputs.

    The normalized value is still returned in the input dtype. Used to carry the residual stream in fp32 between
    decoder layers.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
        return _fused_add_rms_norm_forward(ctx, x, residual, weight, eps)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_residual: torch.Tensor):
        return _fused_add_rms_norm_backward(ctx, grad_out, grad_residual)


def rope_rotary_slice_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply partial RoPE in fp32 and cast the rotated channels back to ``x``'s dtype once.

    ``x`` is ``[batch, heads, seq, head_dim]`` and ``cos``/``sin`` are ``[batch, seq, rotary_dim]``; they are
    unsqueezed on the head axis. Only the first ``rotary_dim`` channels are rotated; the remaining channels are
    returned unchanged. The result is bitwise equal to upcasting all of ``x``, rotating in fp32 and casting back,
    because the pass-through channels survive the round trip through fp32 exactly. Upcasting only the rotary
    slice avoids materializing an fp32 copy of the whole head dimension.
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot_fp32 = x_rot.float()
    half = rotary_dim // 2
    rotated = torch.cat((-x_rot_fp32[..., half:], x_rot_fp32[..., :half]), dim=-1)
    embed = (x_rot_fp32 * cos.unsqueeze(1).float()) + (rotated * sin.unsqueeze(1).float())
    return torch.cat((embed.to(x.dtype), x_pass), dim=-1)


def _uses_fla_kernel(x: torch.Tensor) -> bool:
    # Under torch.compile the torch formula runs, so a compiled layer never traces fla's Triton autograd function.
    return fla_rms_norm_gated is not None and x.is_cuda and not torch.compiler.is_compiling()


_GATE_ACTIVATIONS = ("silu", "swish", "sigmoid")


def _gate_activation(gate: torch.Tensor, activation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``activation(gate)`` and its derivative, for the gate activations fla's kernel supports."""
    sigmoid = torch.sigmoid(gate)
    if activation == "sigmoid":
        return sigmoid, sigmoid * (1.0 - sigmoid)
    return F.silu(gate), sigmoid * (1.0 + gate * (1.0 - sigmoid))


class RMSNormGatedFp32(torch.autograd.Function):
    """RMSNorm of ``x`` times ``weight`` times ``activation(gate)``, in fp32, rounded once to ``x``'s dtype.

    For backward it saves only the inputs as given (``x``, ``gate`` and ``weight``) and the per-row ``rstd``, and
    recomputes the fp32 values from them, instead of keeping full-size fp32 activations alive until backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float, activation: str):
        compute_dtype = _compute_dtype(x)
        hidden = x.to(compute_dtype)
        rstd = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
        activated, _ = _gate_activation(gate.to(compute_dtype), activation)
        out = hidden * rstd * weight.to(compute_dtype) * activated
        ctx.save_for_backward(x, gate, weight, rstd)
        ctx.activation = activation
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, gate, weight, rstd = ctx.saved_tensors
        compute_dtype = _compute_dtype(x)
        normed = x.to(compute_dtype) * rstd
        scale = weight.to(compute_dtype)
        activated, activation_grad = _gate_activation(gate.to(compute_dtype), ctx.activation)

        grad = grad_out.to(compute_dtype)
        grad_gate = grad * normed * scale * activation_grad
        grad_weight = (grad * normed * activated).reshape(-1, weight.shape[-1]).sum(0)
        # Gradient through normed = x * rstd(x), with rstd = (mean(x**2) + eps) ** -0.5.
        grad_normed = grad * scale * activated
        grad_x = rstd * (grad_normed - normed * (grad_normed * normed).mean(-1, keepdim=True))
        return grad_x.to(x.dtype), grad_gate.to(gate.dtype), grad_weight.to(weight.dtype), None, None


def rms_norm_gated_fp32(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float, activation: str
) -> torch.Tensor:
    """RMSNorm of ``x`` times ``weight`` times ``activation(gate)``, evaluated in fp32 and rounded once to ``x``'s dtype.

    This is vLLM's ``RMSNormGated`` with ``norm_before_gate=True``, the GatedDeltaNet output norm of the sampler. On
    CUDA it runs fla's fused ``rms_norm_gated`` kernel, which computes the same way and has a fused backward; elsewhere
    ``RMSNormGatedFp32``, which saves the same tensors for backward.
    """
    if activation not in _GATE_ACTIVATIONS:
        raise ValueError(f"unsupported gate activation {activation!r}; expected one of {_GATE_ACTIVATIONS}")
    if _uses_fla_kernel(x):
        return fla_rms_norm_gated(x, gate, weight, None, activation, eps=eps)
    return RMSNormGatedFp32.apply(x, gate, weight, eps, activation)


def _rms_norm_gated_once_forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return rms_norm_gated_fp32(hidden_states, gate, self.weight, self.variance_epsilon, self.activation)


def _resolve(names: tuple[tuple[str, str], ...]) -> tuple[type[nn.Module], ...]:
    """The classes of ``names`` that the installed transformers has."""
    classes: list[type[nn.Module]] = []
    for module_name, class_name in names:
        try:
            classes.append(getattr(importlib.import_module(module_name), class_name))
        except (ImportError, AttributeError):
            continue
    return tuple(classes)


def round_gated_norm_once(norm: nn.Module) -> None:
    """Make one stock transformers GatedDeltaNet output norm evaluate in fp32 and round once, as vLLM's does.

    The norm keeps its class, parameters and state-dict keys; only its forward is replaced. Any other module raises,
    so a transformers release that changes the norm fails here instead of keeping the old rounding.
    """
    if not isinstance(norm, _resolve(_STOCK_GATED_NORMS)):
        raise RuntimeError(
            f"expected a stock transformers GatedDeltaNet output norm, got {type(norm).__module__}."
            f"{type(norm).__qualname__}; its rounding is unknown, so it cannot be made to round once"
        )
    if norm.activation not in _GATE_ACTIVATIONS:
        raise RuntimeError(f"unsupported GatedDeltaNet output-norm gate activation {norm.activation!r}")
    norm.forward = types.MethodType(_rms_norm_gated_once_forward, norm)


def round_gated_norms_once(model: nn.Module) -> int:
    """Make the output norm of every stock transformers Qwen GatedDeltaNet layer in ``model`` round once.

    Returns how many norms were changed. A norm that is fla's ``FusedRMSNormGated`` already rounds once and is left
    as it is. Models without these GatedDeltaNet layers are left alone.
    """
    gated_delta_net_classes = _resolve(_STOCK_GATED_DELTA_NETS)
    changed = 0
    for module in model.modules():
        if not isinstance(module, gated_delta_net_classes):
            continue
        norm = module.get_submodule("norm")
        if FusedRMSNormGated is not None and isinstance(norm, FusedRMSNormGated):
            continue
        round_gated_norm_once(norm)
        changed += 1
    return changed
