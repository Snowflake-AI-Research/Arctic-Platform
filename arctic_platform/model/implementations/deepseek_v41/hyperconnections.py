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

"""Shifted manifold-constrained hyper-connections for DeepSeek-V4.1.

V4 collapses a sublayer with the ``pre`` that sublayer just computed. V4.1 applies ``post`` and
``comb`` immediately and hands ``pre`` to the next sublayer. The stack starts from a one-hot
``pre`` on stream 0, and the last FFN ``pre`` is the collapse in front of the final norm.

On CUDA the four-stream logits, gates, collapse and write-back run through GLM-5.3-Flash's fused mHC kernels,
which compute the same functions (unweighted RMS, sigmoid gates, the same Sinkhorn steps) and read the streams
once per direction instead of making FP32 copies of them.
"""

import torch
import torch.nn.functional as F
from torch import nn

from arctic_platform.model.implementations.deepseek_v4.hyperconnections import DeepseekV4UnweightedRMSNorm
from arctic_platform.model.implementations.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config
from arctic_platform.model.implementations.glm53.hyper_connections import collapse_streams
from arctic_platform.model.implementations.glm53.hyper_connections import hyper_connection_gates
from arctic_platform.model.implementations.glm53.hyper_connections import mix_streams


def _fused(mhc_states: torch.Tensor) -> bool:
    """The fused kernels take CUDA tensors and are written for four streams."""
    return mhc_states.is_cuda and mhc_states.shape[-2] == 4


def sinkhorn(logits: torch.Tensor, num_iterations: int, eps: float) -> torch.Tensor:
    """Project ``logits`` onto the doubly-stochastic manifold.

    Matches the reference ``hc_split_sinkhorn`` recurrence: the first row normalization is a
    softmax plus ``eps``, and later row normalizations put ``eps`` in the denominator.
    """
    row_max = logits.amax(dim=-1, keepdim=True)
    matrix = torch.exp(logits - row_max)
    matrix = matrix / matrix.sum(dim=-1, keepdim=True) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iterations - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def identity_pre(mhc_states: torch.Tensor) -> torch.Tensor:
    """One-hot mix on stream 0, the mix the embedding is collapsed with."""
    pre = torch.zeros(*mhc_states.shape[:-1], device=mhc_states.device, dtype=torch.float32)
    pre[..., 0] = 1.0
    return pre


def collapse(mhc_states: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Weighted sum of the residual streams. ``pre`` is ``(batch, seq, hc_mult)``."""
    if _fused(mhc_states):
        return collapse_streams(mhc_states, pre)
    collapsed = (pre.unsqueeze(-1) * mhc_states.float()).sum(dim=2)
    return collapsed.to(dtype=mhc_states.dtype)


def write_back(
    residual: torch.Tensor, sublayer_out: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Broadcast ``sublayer_out`` with ``post`` and remix ``residual`` with ``comb``.

    ``comb`` is summed over its source axis, the same contraction as ``comb^T @ residual``.
    """
    if _fused(residual):
        return mix_streams(post, comb, sublayer_out, residual)
    mixed = torch.einsum("bsij,bsid->bsjd", comb.float(), residual.float())
    written = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2).float() + mixed
    return written.to(dtype=sublayer_out.dtype)


class DeepseekV41HyperConnection(nn.Module):
    """One sublayer's ``pre`` / ``post`` / ``comb`` projection.

    The caller applies an incoming ``pre``. This module only produces the mix the next sublayer
    consumes, plus the ``post`` and ``comb`` written back around the current sublayer.
    """

    def __init__(self, config: DeepseekV41Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps, out_dtype=torch.float32)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def mixes(self, mhc_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _fused(mhc_states):
            return hyper_connection_gates(self, mhc_states)
        hc = self.hc_mult
        flat_states = mhc_states.flatten(start_dim=2)
        # Quack's RMSNorm is selected whenever a GPU is present, and it rejects CPU tensors.
        if flat_states.is_cuda:
            flat = self.input_norm(flat_states)
        else:
            flat = flat_states.float()
            flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.input_norm.eps)
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.unflatten(-1, (hc, hc)) * comb_scale + comb_b.view(hc, hc)
        comb = sinkhorn(comb_logits, self.hc_sinkhorn_iters, self.hc_eps)
        return pre, post, comb

    def init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.fn, mean=0.0, std=init_std)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)


__all__ = [
    "DeepseekV41HyperConnection",
    "collapse",
    "identity_pre",
    "sinkhorn",
    "write_back",
]
