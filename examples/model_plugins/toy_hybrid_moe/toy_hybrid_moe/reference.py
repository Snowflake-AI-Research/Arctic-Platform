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

"""A slow, independent ToyHybridMoE forward over one unpacked sequence: the conformance oracle.

It reads the trainer model's weights but none of its code: attention is an explicit masked softmax, rotary
embedding rotates pairs by angle, the convolution and recurrence are explicit loops, and every token visits its
experts one at a time.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import FULL_ATTENTION
from .config import MOE
from .config import SHORT_CONV


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * weight


def _rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """``x`` is ``[L, H, D]``; dimension ``j`` pairs with ``j + D/2`` and rotates by ``position * theta^(-2j/D)``."""
    length, _, dim = x.shape
    half = dim // 2
    out = torch.empty_like(x)
    for position in range(length):
        for j in range(half):
            angle = position * theta ** (-2 * j / dim)
            c, s = math.cos(angle), math.sin(angle)
            a, b = x[position, :, j], x[position, :, j + half]
            out[position, :, j] = a * c - b * s
            out[position, :, j + half] = b * c + a * s
    return out


def _swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def reference_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits ``[L, vocab]`` of one sequence, positions ``0..L-1``."""
    cfg = model.config
    p = {name: param.detach().double() for name, param in model.named_parameters()}
    eps = cfg.rms_norm_eps
    h = p["model.embed_tokens.weight"][input_ids]
    length = h.shape[0]
    for i in range(cfg.num_hidden_layers):
        pre = f"model.layers.{i}."
        x = _rms(h, p[pre + "input_norm.weight"], eps)
        if cfg.layer_types[i] == FULL_ATTENTION:
            heads, kv_heads, dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
            q = _rope((x @ p[pre + "self_attn.q_proj.weight"].T).view(length, heads, dim), cfg.rope_theta)
            k = _rope((x @ p[pre + "self_attn.k_proj.weight"].T).view(length, kv_heads, dim), cfg.rope_theta)
            v = (x @ p[pre + "self_attn.v_proj.weight"].T).view(length, kv_heads, dim)
            mixed = torch.zeros(length, heads, dim, dtype=h.dtype)
            for head in range(heads):
                kv = head // (heads // kv_heads)
                scores = (q[:, head] @ k[:, kv].T) / math.sqrt(dim)
                allowed = torch.ones(length, length, dtype=torch.bool).tril()
                if cfg.sliding_window is not None:
                    allowed &= ~torch.ones(length, length, dtype=torch.bool).tril(-cfg.sliding_window)
                mixed[:, head] = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ v[:, kv]
            h = h + mixed.reshape(length, -1) @ p[pre + "self_attn.o_proj.weight"].T
        elif cfg.layer_types[i] == SHORT_CONV:
            u = x @ p[pre + "conv.in_proj.weight"].T
            gate = x @ p[pre + "conv.gate_proj.weight"].T
            weight = p[pre + "conv.conv_weight"]
            conv = torch.zeros_like(u)
            for t in range(length):
                for tap in range(cfg.conv_kernel):
                    source = t - (cfg.conv_kernel - 1 - tap)
                    if source >= 0:
                        conv[t] += weight[:, tap] * u[source]
            h = h + (conv * F.silu(gate)) @ p[pre + "conv.out_proj.weight"].T
        else:
            heads, dim, kernel = cfg.recurrence_heads, cfg.recurrence_head_dim, cfg.conv_kernel
            u = (x @ p[pre + "recurrence.in_proj.weight"].T).view(length, heads, dim)
            g = (x @ p[pre + "recurrence.gate_proj.weight"].T).view(length, heads, dim)
            z = (x @ p[pre + "recurrence.z_proj.weight"].T).view(length, heads, dim)
            weight, bias = p[pre + "recurrence.conv_weight"], p[pre + "recurrence.decay_bias"]
            conv = torch.zeros_like(u)
            for t in range(length):
                for tap in range(kernel):
                    source = t - (kernel - 1 - tap)
                    if source >= 0:
                        conv[t] += weight[:, :, tap] * u[source]
            conv = F.silu(conv)
            state = torch.zeros(heads, dim, dtype=h.dtype)
            outputs = torch.zeros_like(u)
            for t in range(length):
                a = torch.sigmoid(g[t] + bias[:, None])
                state = a * state + (1 - a) * conv[t]
                outputs[t] = state
            y = _rms(outputs, p[pre + "recurrence.norm.weight"], eps) * F.silu(z)
            h = h + y.reshape(length, -1) @ p[pre + "recurrence.out_proj.weight"].T

        x = _rms(h, p[pre + "post_norm.weight"], eps)
        if cfg.mlp_types[i] == MOE:
            out = _swiglu(
                x,
                p[pre + "mlp.shared_expert.gate_proj.weight"],
                p[pre + "mlp.shared_expert.up_proj.weight"],
                p[pre + "mlp.shared_expert.down_proj.weight"],
            )
            for t in range(length):
                scores = torch.softmax(x[t] @ p[pre + "mlp.router.weight"].T, dim=-1)
                top = torch.topk(scores, cfg.num_experts_per_tok).indices
                gates = scores[top] / scores[top].sum() * cfg.routed_scaling_factor
                for gate, expert in zip(gates, top.tolist()):
                    out[t] += (
                        gate
                        * _swiglu(
                            x[t : t + 1],
                            p[pre + "mlp.experts.w1"][expert],
                            p[pre + "mlp.experts.w3"][expert],
                            p[pre + "mlp.experts.w2"][expert],
                        )[0]
                    )
        else:
            out = _swiglu(
                x, p[pre + "mlp.gate_proj.weight"], p[pre + "mlp.up_proj.weight"], p[pre + "mlp.down_proj.weight"]
            )
        h = h + out
    return (_rms(h, p["model.norm.weight"], eps) @ p["lm_head.weight"].T).float()
