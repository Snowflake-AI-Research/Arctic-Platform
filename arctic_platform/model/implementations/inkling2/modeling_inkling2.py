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

"""Inkling2 text model with AP's expert-parallel MoE and query-chunked FlexAttention.

Module and parameter names follow the official checkpoint (``model.llm.`` becomes ``model.``), except the routed
experts, which use AP's ``w1``/``w3``/``w2`` for the checkpoint's interleaved ``w13_weight`` and ``w2_weight``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from arctic_platform.model.implementations.fp8 import expert_lora_output
from arctic_platform.model.implementations.fp8 import grouped_expert_lora_rows
from arctic_platform.model.implementations.gpu.sp.collectives import gather_sequence
from arctic_platform.model.implementations.gpu.sp.collectives import sequence_head_all_to_all
from arctic_platform.model.implementations.inkling2.configuration_inkling2 import Inkling2TextConfig
from arctic_platform.model.implementations.inkling.attention import relative_attention
from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep
from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs
from arctic_platform.model.implementations.moe.layers.moe import _maybe_to_local

_EXPERT_LORA_NAME = re.compile(r"(.*\.experts)\.(w[123])\.lora_([AB])\.weight")


def rms_norm(x, weight, eps):
    """FP32 ``x / rms(x) * weight``; ``weight=None`` normalizes only."""
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x if weight is None else x * weight.float()


def l2_clip(x, threshold, eps=1e-6):
    """Scale each last-dimension vector down to an L2 norm of at most ``threshold``."""
    return x * (threshold / (x.norm(dim=-1, keepdim=True) + eps)).clamp(max=1.0)


def softcap(x, cap):
    return cap * torch.tanh(x / cap)


@dataclass
class Inkling2Batch:
    """Per-forward token metadata shared by every layer.

    Under sequence parallelism ``tau`` and ``conv_masks`` cover this rank's shard, while ``padding_mask`` and
    ``sequence_ids`` cover the whole sequence that the head-sharded attention sees. Attention block masks depend
    only on these, so every layer and the checkpoint recompute share ``mask_cache``.
    """

    tau: torch.Tensor | None
    conv_masks: list[torch.Tensor] | None
    padding_mask: torch.Tensor | None
    sequence_ids: torch.Tensor | None
    query_chunk_size: int
    sp_group: dist.ProcessGroup | None = None
    mask_cache: dict = field(default_factory=dict, repr=False)


class Inkling2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps).to(x.dtype)


class Inkling2L2Clip(nn.Module):
    def __init__(self, num_heads, dim, threshold):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, dim))
        self.threshold = threshold

    def forward(self, x):
        return l2_clip(x.float() * self.scale.float(), self.threshold).to(x.dtype)


class Inkling2RelativeProjection(nn.Module):
    def __init__(self, d_rel, extent):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(d_rel, extent))


def previous_tokens(x, width, group):
    """The ``width`` tokens before this rank's shard of ``x`` ``[B, L, C]``, zeros before the first token."""
    if group is None:
        return x.new_zeros(x.shape[0], width, x.shape[2])
    halo = min(width, x.shape[1])
    tails = gather_sequence(x[:, -halo:], group)
    end = dist.get_rank(group) * halo
    previous = tails[:, max(0, end - width) : end]
    # A zero dependency keeps every rank in the gather's backward.
    zeros = tails[:, :1] * 0
    return torch.cat((zeros.expand(-1, width - previous.shape[1], -1), previous), dim=1)


def short_conv(x, lag_weights, batch, scale):
    """FP32 ``scale * (x + sum_lag lag_weights[lag] * x[t - lag])`` within each sequence of ``x`` ``[B, L, C]``."""
    x = x.float()
    length, width = x.shape[1], len(lag_weights) - 1
    extended = torch.cat((previous_tokens(x, width, batch.sp_group), x), dim=1)
    out = x + x * lag_weights[0]
    for lag in range(1, width + 1):
        shifted = extended[:, width - lag : width - lag + length]
        if batch.conv_masks is not None:
            shifted = shifted * batch.conv_masks[lag - 1]
        out = out + shifted * lag_weights[lag]
    return out * scale


class Inkling2ShortConv(nn.Module):
    """Causal depthwise convolution with an identity tap; ``weight`` is ``[dim, 1, width]``, newest tap last."""

    def __init__(self, dim, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size))

    def forward(self, x, batch):
        width = self.weight.shape[-1]
        taps = self.weight.float().squeeze(1)
        return short_conv(x, [taps[:, width - 1 - lag] for lag in range(width)], batch, (width + 1) ** -0.5)


class Inkling2DynamicShortConv(nn.Module):
    """Per-token taps ``static_w[lag] + weight_proj(dynamic_input)[head, lag]``, shared within each head."""

    def __init__(self, dim, kernel_size, num_heads, input_size):
        super().__init__()
        self.static_w = nn.Parameter(torch.empty(kernel_size, dim))
        self.weight_proj = nn.Parameter(torch.empty(num_heads * kernel_size, input_size))

    def forward(self, x, dynamic_input, batch):
        width, dim = self.static_w.shape
        heads = self.weight_proj.shape[0] // width
        taps = F.linear(dynamic_input, self.weight_proj.to(dynamic_input.dtype)).float()
        taps = taps.view(*taps.shape[:-1], heads, 1, width)
        static = self.static_w.float().view(width, heads, dim // heads)
        lag_weights = [(static[lag] + taps[..., lag]).flatten(-2) for lag in range(width)]
        return short_conv(x, lag_weights, batch, (2 * width + 1) ** -0.5)


def attn_res(blocks, last, addend, query, norms, normalize, eps, token_chunk_size=None):
    """Block attention residual: per query row, softmax-mix the RMS-normalized ``[*blocks, last (+ addend)]`` over
    blocks, then apply that row's norm. Mixes in FP32, like the sampler's fused kernel.

    The mix is per token, so long inputs run in checkpointed token chunks: the FP32 working set then scales with
    ``token_chunk_size`` rather than the sequence.
    """
    length = last.shape[1]
    if token_chunk_size is None or length <= token_chunk_size:
        return _attn_res(blocks, last, addend, query, norms, normalize, eps)
    count = len(blocks)

    def mix(*tensors):
        return tuple(
            _attn_res(list(tensors[:count]), tensors[count], tensors[count + 1], query, norms, normalize, eps)
        )

    chunks = []
    for start in range(0, length, token_chunk_size):
        window = slice(start, start + token_chunk_size)
        args = (
            *(block[:, window] for block in blocks),
            last[:, window],
            None if addend is None else addend[:, window],
        )
        chunks.append(checkpoint(mix, *args, use_reentrant=False) if torch.is_grad_enabled() else mix(*args))
    return [torch.cat(parts, dim=1) for parts in zip(*chunks)]


def _attn_res(blocks, last, addend, query, norms, normalize, eps):
    dtype = last.dtype
    last = last.float() if addend is None else last.float() + addend.float()
    values = torch.stack([*(block.float() for block in blocks), last])
    hidden = values.shape[-1]
    inv_rms = torch.rsqrt(values.square().mean(-1) + eps)
    queries = query.float().view(-1, hidden) * hidden**-0.5
    logits = torch.einsum("nblc,qc->qbln", values, queries) * inv_rms.permute(1, 2, 0)
    probs = logits.softmax(-1)
    if normalize:
        values = values * inv_rms[..., None]
    mixes = torch.einsum("qbln,nblc->qblc", probs, values)
    return [rms_norm(mix, norm.weight, norm.eps).to(dtype) for norm, mix in zip(norms, mixes, strict=True)]


class Inkling2Attention(nn.Module):
    """Sliding-window (local) or full attention with a learned relative bias, an output gate and a per-token sink.

    Full layers project q, r, gate and sink from ``x_q``, k from ``x_k`` and v from ``x_v``, convolve k/v with
    input-dependent taps, and scale q/r by the log-scaling ``tau``.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_local = layer_idx in config.local_layer_ids
        hidden = config.hidden_size
        if self.is_local:
            heads, kv_heads, dim = (
                config.swa_num_attention_heads,
                config.swa_num_key_value_heads,
                config.swa_head_dim,
            )
            extent = config.sliding_window_size
        else:
            heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
            extent = config.rel_extent
        self.num_heads, self.num_key_value_heads, self.head_dim = heads, kv_heads, dim
        self.d_rel = config.d_rel
        self.window = config.sliding_window_size if self.is_local else None
        self.qk_clip = config.qk_l2_clip_threshold
        self.eps = config.rms_norm_eps
        self.wq_du = nn.Linear(hidden, heads * dim, bias=False)
        self.wr_du = nn.Linear(hidden, heads * config.d_rel, bias=False)
        self.gate_attn = nn.Linear(hidden, heads * dim, bias=False)
        self.attn_sink_proj = nn.Linear(hidden, heads, bias=False)
        self.wk_dv = nn.Linear(hidden, kv_heads * dim, bias=False)
        self.wv_dv = nn.Linear(hidden, kv_heads * dim, bias=False)
        self.wo_ud = nn.Linear(heads * dim, hidden, bias=False)
        self.q_l2_clip = Inkling2L2Clip(heads, dim, config.qk_l2_clip_threshold)
        self.rel_l2_clip = Inkling2L2Clip(heads, config.d_rel, config.rel_l2_clip_threshold)
        self.rel_logits_proj = Inkling2RelativeProjection(config.d_rel, extent)
        if self.is_local:
            self.k_sconv = Inkling2ShortConv(kv_heads * dim, config.sconv_kernel_size)
            self.v_sconv = Inkling2ShortConv(kv_heads * dim, config.sconv_kernel_size)
        else:
            self.k_sconv = Inkling2DynamicShortConv(kv_heads * dim, config.sconv_kernel_size, kv_heads, hidden)
            self.v_sconv = Inkling2DynamicShortConv(kv_heads * dim, config.sconv_kernel_size, kv_heads, hidden)

    def forward(self, x_q, x_k, x_v, batch):
        size, length, _ = x_q.shape
        dtype = x_q.dtype
        heads, kv_heads, dim = self.num_heads, self.num_key_value_heads, self.head_dim
        query = self.q_l2_clip(self.wq_du(x_q).view(size, length, heads, dim))
        relative = self.rel_l2_clip(self.wr_du(x_q).view(size, length, heads, self.d_rel))
        if not self.is_local and batch.tau is not None:
            tau = batch.tau[..., None, None]
            query = (query.float() * tau).to(dtype)
            relative = (relative.float() * tau).to(dtype)
        if self.is_local:
            key = self.k_sconv(self.wk_dv(x_k), batch)
            value = self.v_sconv(self.wv_dv(x_v), batch)
        else:
            key = self.k_sconv(self.wk_dv(x_k), x_k, batch)
            value = self.v_sconv(self.wv_dv(x_v), x_v, batch)
        key = l2_clip(key.to(dtype).float().view(size, length, kv_heads, dim), self.qk_clip).to(dtype)
        value = rms_norm(value.to(dtype).view(size, length, kv_heads, dim), None, self.eps).to(dtype)
        gate = self.gate_attn(x_q).view(size, length, heads, dim)
        sink = self.attn_sink_proj(x_q)
        group = batch.sp_group
        if group is not None:
            # Ulysses: attend over the whole sequence with this rank's share of the heads.
            ranks = dist.get_world_size(group)
            if ranks > kv_heads:
                key, value = (tensor.repeat_interleave(ranks // kv_heads, dim=2) for tensor in (key, value))
            query, relative, key, value, gate, sink = (
                sequence_head_all_to_all(group, tensor, scatter_dim=2, gather_dim=1)
                for tensor in (query, relative, key, value, gate, sink)
            )
            length = query.shape[1]
        query, key, value = (tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
        chunk_size = batch.query_chunk_size
        chunks = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)

            def attend(q, k, v, r, projection, g, s, start=start):
                output, lse = relative_attention(
                    q.contiguous(),
                    k,
                    v,
                    r,
                    projection,
                    query_offset=start,
                    padding_mask=batch.padding_mask,
                    sequence_ids=batch.sequence_ids,
                    window=self.window,
                    scaling=dim**-0.5,
                    return_lse=True,
                    mask_cache=batch.mask_cache,
                )
                output = output.transpose(1, 2).float()
                gated = output * torch.sigmoid(g.float().view(output.shape))
                gated = gated * torch.sigmoid(lse.transpose(1, 2) - s.float())[..., None]
                return gated.to(dtype).flatten(2)

            args = (
                query[:, :, start:end],
                key,
                value,
                relative[:, start:end].contiguous(),
                self.rel_logits_proj.proj,
                gate[:, start:end],
                sink[:, start:end],
            )
            if torch.is_grad_enabled():
                chunks.append(checkpoint(attend, *args, use_reentrant=False, preserve_rng_state=False))
            else:
                chunks.append(attend(*args))
        output = torch.cat(chunks, dim=1)
        if group is not None:
            output = output.unflatten(2, (-1, dim))
            output = sequence_head_all_to_all(group, output, scatter_dim=1, gather_dim=2).flatten(2)
        return self.wo_ud(output)


class Inkling2Router(nn.Module):
    """Sigmoid scores plus a constant selection bias pick the experts; the weights are the selected sigmoids,
    renormalized and scaled by ``route_scale``. Routing runs in FP32."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, flat):
        logits = F.linear(flat.float(), self.weight.float())
        indices = torch.topk(logits.sigmoid() + self.bias.float(), self.top_k, dim=-1, sorted=False).indices
        weights = F.logsigmoid(logits.gather(-1, indices)).softmax(-1) * self.route_scale
        return weights, indices


class Inkling2Experts(GroupedExperts):
    """Routed experts with ``softcap(gate) * sigmoid(gate) * softcap(up)``; each row's routing weight scales the
    activation before the down projection, as in the sampler.

    PEFT expert LoRA (``target_parameters`` on ``w1``/``w2``/``w3``) is added to each GEMM's output from the
    attached A/B factors (``_ap_lora_ab``), without a merged ``[experts, out, in]`` delta.
    """

    gate_softcap: float
    up_softcap: float
    unfused_expert_lora = True

    def _activation(self, gate, up, scores, dtype):
        gate, up = gate.float(), up.float()
        hidden = softcap(gate, self.gate_softcap) * torch.sigmoid(gate) * softcap(up, self.up_softcap)
        if scores is not None:
            hidden = hidden * scores[: hidden.shape[0], None].float()
        return hidden.to(dtype)

    def _forward_deepep(self, x, num_tokens_per_expert, scores=None):
        if self.fp8_block_size is not None:
            raise ValueError("Inkling2 experts train in BF16 or FP32")
        weights = dict(zip(("w1", "w2", "w3"), (_maybe_to_local(w).to(x.dtype) for w in (self.w1, self.w2, self.w3))))
        lora = {
            name: (a.to(x.dtype), b.to(x.dtype), rank, scaling)
            for name, (a, b, rank, scaling) in getattr(self, "_ap_lora_ab", {}).items()
        }
        if self.use_grouped_mm and x.is_cuda and x.dtype == torch.bfloat16:
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

            def project(inputs, name):
                out = torch._grouped_mm(inputs, weights[name].transpose(-2, -1), offs=offsets)
                if name in lora:
                    # Rows past the routed ones stay unused, as in the base GEMM.
                    out = out + grouped_expert_lora_rows(inputs, lora[name], offsets, inputs.shape[0])
                return out

            hidden = self._activation(project(x, "w1"), project(x, "w3"), scores, x.dtype)
            return project(hidden, "w2")

        def project_expert(inputs, name, expert):
            out = inputs @ weights[name][expert].T
            if name in lora:
                out = out + expert_lora_output(inputs, fused_delta=None, ab=lora[name], expert_idx=expert)
            return out

        counts = num_tokens_per_expert.tolist()
        rows = sum(counts)
        outputs = []
        start = 0
        for expert, count in enumerate(counts):
            tokens = x[start : start + count]
            gate, up = project_expert(tokens, "w1", expert), project_expert(tokens, "w3", expert)
            expert_scores = None if scores is None else scores[start : start + count]
            outputs.append(project_expert(self._activation(gate, up, expert_scores, x.dtype), "w2", expert))
            start += count
        outputs.append(x.new_zeros(x.shape[0] - rows, weights["w2"].shape[1]))
        return torch.cat(outputs)


class Inkling2MoE(MoE):
    def __init__(self, config):
        super().__init__(
            MoEArgs(
                num_experts=config.n_routed_experts,
                num_shared_experts=0,
                top_k=config.num_experts_per_tok,
                score_before_experts=False,
                load_balance_coeff=None,
                use_grouped_mm=getattr(config, "use_grouped_mm", True),
                fp32_routed_weighting=True,
            ),
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
        )
        del self.router
        self.gate = Inkling2Router(config)
        self.experts.__class__ = Inkling2Experts
        self.experts.gate_softcap = config.mlp_gate_softcap
        self.experts.up_softcap = config.mlp_up_softcap
        # UCCL combine events retain their receive buffers until the stream wait.
        self.sync_combine_per_chunk = True
        self.checkpoint_dispatch_chunks = True

    def forward(self, hidden_states):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        weights, indices = self.gate(flat)
        if uses_dispatch_ep(self.ep_comm_backend):
            routed = self._run_deepep_routed_experts(flat, indices, weights)
        else:
            order = torch.argsort(indices.reshape(-1), stable=True)
            token_indices = order // indices.shape[-1]
            counts = torch.bincount(indices.reshape(-1), minlength=self.gate.num_experts)
            outputs = self.experts(flat[token_indices], counts, weights.reshape(-1)[order])
            routed = torch.zeros_like(flat).index_add(0, token_indices, outputs)
        return routed.view(shape)


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_local = layer_idx in config.local_layer_ids
        self.starts_block = layer_idx % config.n_layers_per_attn_res_block == 0
        self.normalize_values = config.attn_res_normalize_values
        self.eps = config.rms_norm_eps
        hidden = config.hidden_size
        self.attn = Inkling2Attention(config, layer_idx)
        if self.is_local:
            self.attn_norm = Inkling2RMSNorm(hidden, config.rms_norm_eps)
        else:
            self.q_input_norm = Inkling2RMSNorm(hidden, config.rms_norm_eps)
            self.k_input_norm = Inkling2RMSNorm(hidden, config.rms_norm_eps)
            self.v_input_norm = Inkling2RMSNorm(hidden, config.rms_norm_eps)
        self.mlp_norm = Inkling2RMSNorm(hidden, config.rms_norm_eps)
        self.mlp = Inkling2MoE(config)
        self.attn_sconv = Inkling2ShortConv(hidden, config.sconv_kernel_size)
        self.mlp_sconv = Inkling2ShortConv(hidden, config.sconv_kernel_size)

    def _mix(self, blocks, last, addend, query, norms, batch):
        return attn_res(blocks, last, addend, query, norms, self.normalize_values, self.eps, batch.query_chunk_size)

    def forward(self, blocks, partial, attn_query, mlp_query, batch):
        """Return the completed blocks and the running block after this layer.

        ``blocks`` are the completed block-residual sums (the embeddings first) and ``partial`` the running one.
        A layer that starts a block closes ``partial`` before its MLP input mix.
        """
        norms = [self.attn_norm] if self.is_local else [self.q_input_norm, self.k_input_norm, self.v_input_norm]
        if len(blocks) == 0:
            inputs = [norm(partial) for norm in norms]
        else:
            inputs = self._mix(blocks, partial, None, attn_query, norms, batch)
        x_q, x_k, x_v = inputs if len(inputs) == 3 else inputs * 3
        attn_out = self.attn_sconv(self.attn(x_q, x_k, x_v, batch), batch).to(partial.dtype)
        if self.starts_block:
            blocks = [*blocks, partial]
            (mlp_in,) = self._mix(blocks, attn_out, None, mlp_query, [self.mlp_norm], batch)
            residual = attn_out.float()
        else:
            (mlp_in,) = self._mix(blocks, partial, attn_out, mlp_query, [self.mlp_norm], batch)
            residual = partial.float() + attn_out.float()
        partial = (residual + self.mlp_sconv(self.mlp(mlp_in), batch)).to(partial.dtype)
        return blocks, partial


class Inkling2AttnRes(nn.Module):
    """FP32 pseudo-queries of the block attention residuals: one per input of each layer's attention (three for
    full layers) and MLP, and one for the final norm."""

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        local = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden if index in local else (3, hidden), dtype=torch.float32))
            for index in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden, dtype=torch.float32)) for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))


class Inkling2TextModel(nn.Module):
    _sp_group = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(Inkling2DecoderLayer(config, index) for index in range(config.num_hidden_layers))
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def _batch(self, hidden, attention_mask, position_ids):
        size, length = hidden.shape[:2]
        config, group = self.config, self._sp_group
        offset = dist.get_rank(group) * length if group is not None else 0
        padding = attention_mask.bool() if attention_mask is not None else None
        positions = position_ids.expand(size, -1).contiguous() if position_ids is not None else None
        global_positions = positions
        if group is not None:
            padding = gather_sequence(padding, group) if padding is not None else None
            global_positions = gather_sequence(positions, group) if positions is not None else None
        sequence_ids = None
        if global_positions is not None:
            resets = global_positions[:, 1:] != global_positions[:, :-1] + 1
            sequence_ids = F.pad(resets.to(torch.int64).cumsum(-1), (1, 0))
        conv_masks = None
        if padding is not None or sequence_ids is not None:
            total = (padding if padding is not None else sequence_ids).shape[1]
            conv_masks = []
            for lag in range(1, config.sconv_kernel_size):
                valid = torch.ones(size, total, dtype=torch.bool, device=hidden.device)
                if sequence_ids is not None:
                    valid[:, lag:] &= sequence_ids[:, lag:] == sequence_ids[:, :-lag]
                if padding is not None:
                    valid[:, lag:] &= padding[:, :-lag]
                conv_masks.append(valid[:, offset : offset + length, None].float())
        tau = None
        if config.log_scaling_n_floor is not None:
            if positions is None:
                positions = torch.arange(offset, offset + length, device=hidden.device).expand(size, -1)
            ratio = ((positions + 1).float() / config.log_scaling_n_floor).clamp(min=1.0)
            tau = 1.0 + config.log_scaling_alpha * torch.log(ratio)
        return Inkling2Batch(
            tau=tau,
            conv_masks=conv_masks,
            padding_mask=padding,
            sequence_ids=sequence_ids,
            query_chunk_size=getattr(config, "inkling2_attention_query_chunk_size", 2048),
            sp_group=group,
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is not None and (not torch.is_tensor(attention_mask) or attention_mask.ndim != 2):
            raise ValueError("Inkling2 requires a two-dimensional padding mask")
        hidden = self.embed_norm(inputs_embeds)
        batch = self._batch(hidden, attention_mask, position_ids)
        queries = self.attn_res
        blocks, partial = [], hidden
        for index, layer in enumerate(self.layers):
            attn_query = queries.attn_queries[index - 1] if index > 0 else None
            blocks, partial = layer(blocks, partial, attn_query, queries.mlp_queries[index], batch)
        config = self.config
        return attn_res(
            blocks,
            partial,
            None,
            queries.output_query,
            [self.norm],
            config.attn_res_normalize_values,
            config.rms_norm_eps,
            batch.query_chunk_size,
        )[0]


class Inkling2ForCausalLMPrimeRL(PreTrainedModelPrimeRL, GenerationMixin):
    config_class = Inkling2TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Inkling2DecoderLayer"]
    _inkling2_lm_head_token_chunk_size = None
    _inkling2_fp32_lm_head = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Inkling2TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        for name, parameter in self.named_parameters():
            if self.keeps_fp32_parameter(name):
                parameter.data = parameter.data.float()

    @torch.no_grad()
    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, (Inkling2RMSNorm, Inkling2L2Clip)):
            for parameter in module.parameters(recurse=False):
                nn.init.ones_(parameter)
        elif isinstance(module, Inkling2RelativeProjection):
            nn.init.normal_(module.proj, std=std)
        elif isinstance(module, Inkling2ShortConv):
            nn.init.normal_(module.weight, std=0.1)
        elif isinstance(module, Inkling2DynamicShortConv):
            nn.init.normal_(module.static_w, std=0.1)
            nn.init.normal_(module.weight_proj, std=std)
        elif isinstance(module, Inkling2AttnRes):
            for parameter in module.parameters():
                nn.init.normal_(parameter, std=std)
        elif isinstance(module, Inkling2Router):
            nn.init.normal_(module.weight, std=std)
            nn.init.zeros_(module.bias)
        elif isinstance(module, GroupedExperts):
            for parameter in (module.w1, module.w2, module.w3):
                nn.init.normal_(parameter, std=std)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def _logits(self, hidden):
        vocab = self.config.unpadded_vocab_size or self.vocab_size
        weight = self.lm_head.weight[:vocab]
        if self._inkling2_fp32_lm_head:
            hidden, weight = hidden.float(), weight.float()
        return softcap(F.linear(hidden, weight).float(), self.config.final_logit_softcapping)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        dss_compute_logprobs=False,
        temperature=None,
        action_masks=None,
        group_token_ids=None,
        **kwargs,
    ):
        if past_key_values is not None or use_cache:
            raise ValueError("Inkling2 training does not support KV caching")
        num_items = kwargs.pop("num_items_in_batch", None)
        shifted = kwargs.pop("shift_labels", None)
        if labels is not None and shifted is None and not dss_compute_logprobs and self.model._sp_group is not None:
            raise ValueError("Inkling2 sequence parallelism requires shift_labels; a shard cannot shift its labels")
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        chunk_size = self._inkling2_lm_head_token_chunk_size
        if chunk_size is None:
            if dss_compute_logprobs:
                raise ValueError("Inkling2 logprobs require an enabled chunked head and explicit labels")
            columns = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self._logits(hidden[:, columns])
            loss = None
            if labels is not None:
                labels = shifted if shifted is not None else F.pad(labels, (0, 1), value=-100)[:, 1:]
                labels = labels[:, columns]
                valid = labels != -100
                denominator = valid.sum() if num_items is None else num_items
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum") / denominator
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        if labels is None:
            raise ValueError("Inkling2 chunked logprobs require explicit next-token labels")
        from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
        from arctic_platform.model.implementations.gpu.lm_head import slice_action_masks_for_logits_to_keep
        from arctic_platform.model.implementations.gpu.lm_head import slice_temperature_for_logits_to_keep

        if not dss_compute_logprobs:
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("Inkling2 chunked SFT requires logits_to_keep=0")
            labels = shifted if shifted is not None else F.pad(labels, (0, 1), value=-100)[:, 1:]
        else:
            action_masks = slice_action_masks_for_logits_to_keep(
                action_masks,
                batch_size=hidden.shape[0],
                original_seq_len=hidden.shape[1],
                logits_to_keep=logits_to_keep,
            )
            columns = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int) and logits_to_keep > 0
                else (slice(None) if isinstance(logits_to_keep, int) else logits_to_keep)
            )
            hidden, labels = (hidden[:, columns], labels.reshape(hidden.shape[:2])[:, columns])
            temperature = slice_temperature_for_logits_to_keep(temperature, columns)
            if group_token_ids is not None:
                group_token_ids = group_token_ids[:, columns]
        vocab = self.config.unpadded_vocab_size or self.vocab_size
        logprobs = chunked_lm_head_logprobs(
            hidden,
            self.lm_head.weight[:vocab],
            labels,
            token_chunk_size=chunk_size,
            vocab_chunk_size=8192,
            fp32_lm_head=self._inkling2_fp32_lm_head,
            temperature=temperature if dss_compute_logprobs else None,
            action_masks=action_masks if dss_compute_logprobs else None,
            group_token_ids=group_token_ids if dss_compute_logprobs else None,
            softcap=self.config.final_logit_softcapping,
        )
        if dss_compute_logprobs:
            if group_token_ids is not None:
                return {"logprobs": logprobs[0], "group_log_probs": logprobs[1]}
            return {"logprobs": logprobs}
        valid = labels != -100
        denominator = valid.sum() if num_items is None else num_items
        loss = -logprobs.masked_fill(~valid, 0).sum() / denominator
        return CausalLMOutputWithPast(loss=loss, logits=None)

    @staticmethod
    def keeps_fp32_parameter(name):
        return (
            name.endswith(".gate.bias")
            or ".attn_res." in name
            or any(
                f".{part}." in name for part in ("attn_sconv", "mlp_sconv", "k_sconv", "v_sconv", "lora_A", "lora_B")
            )
        )

    def init_buffers_post_meta(self):
        for module in self.modules():
            if isinstance(module, Inkling2MoE):
                module.tokens_per_expert.zero_()

    @staticmethod
    def vllm_lora_tensors(tensors):
        """Synced LoRA tensors in the served checkpoint's adapter layout.

        Expert adapters are 3D, ``lora_A`` ``[E, r, in]`` and ``lora_B`` ``[E, out, r]``, on ``w13_weight`` and
        ``w2_weight``. The ``w1`` and ``w3`` adapters merge into one ``w13_weight`` adapter of twice the rank whose
        output rows interleave gate and up like the base weight.
        """
        pending = {}
        for name, tensor in tensors:
            match = _EXPERT_LORA_NAME.fullmatch(name)
            if match is None:
                yield name, tensor
                continue
            prefix, projection, kind = match.groups()
            if projection == "w2":
                yield f"{prefix}.w2_weight.lora_{kind}.weight", tensor
                continue
            factors = pending.setdefault(prefix, {})
            factors[projection, kind] = tensor
            if len(factors) == 4:
                factors = pending.pop(prefix)
                rank = factors["w1", "A"].shape[1]
                gate, up = F.pad(factors["w1", "B"], (0, rank)), F.pad(factors["w3", "B"], (rank, 0))
                yield f"{prefix}.w13_weight.lora_A.weight", torch.cat((factors["w1", "A"], factors["w3", "A"]), 1)
                yield f"{prefix}.w13_weight.lora_B.weight", torch.stack((gate, up), 2).flatten(1, 2)
        if pending:
            raise ValueError(f"Inkling2 expert LoRA needs adapters on both w1 and w3: {sorted(pending)}")

    @classmethod
    def convert_layer_to_hf(cls, state_dict, layer_idx):
        """Checkpoint-format routed experts: gate and up rows interleaved in ``w13_weight``."""
        prefix = f"model.layers.{layer_idx}.mlp.experts."
        if prefix + "w1" in state_dict:
            gate, up = state_dict.pop(prefix + "w1"), state_dict.pop(prefix + "w3")
            state_dict[prefix + "w13_weight"] = torch.stack((gate, up), dim=2).flatten(1, 2)
            state_dict[prefix + "w2_weight"] = state_dict.pop(prefix + "w2")
        return state_dict
