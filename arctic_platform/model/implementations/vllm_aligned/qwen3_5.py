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
"""vLLM-aligned logprobs for the dense Qwen3.5 family, the GatedDeltaNet hybrids such as Qwen3.8-27B.

The logprobs the forward returns are computed by the functions vLLM 0.32 runs for ``Qwen3_5ForConditionalGeneration``
on text, on CUDA, with ``VLLM_BATCH_INVARIANT=1``, ``enforce_eager=True``, tensor parallelism 1, the FLASH_ATTN
backend, ``additional_config={"gdn_prefill_backend": "triton"}``, ``VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0``, no
prefix caching, no speculative decoding, every prompt prefilled in one step, and
:mod:`arctic_platform.inference.vllm.gdn_batch_invariance` enabled (``ARCTIC_GDN_BATCH_INVARIANT=1``). The trainer
pins the same GDN kernel configs. Per decoder layer:

* norms: ``GemmaRMSNorm`` as vLLM's native IR op, with the residual carried as vLLM carries it;
* full attention: the concatenated q/k/v projection, ``fused_qk_rmsnorm_rope_gate``, FlashAttention over a paged KV
  cache whose page size is the one vLLM picks for the hybrid (:func:`hybrid_kv_block_size`), ``sigmoid`` gating and
  the output projection;
* GatedDeltaNet: the concatenated ``[q, k, v, z]`` and ``[b, a]`` projections, ``causal_conv1d_fn``, then the
  chunked delta rule (``fused_post_conv_prep`` + ``chunk_gated_delta_rule``) over each prompt and the recurrent update
  (``fused_sigmoid_gating_delta_rule_update``) over each decoded token, the gated RMSNorm and the output projection;
* MLP: the concatenated gate/up projection, ``silu_and_mul`` and the down projection;
* the sampled token's logprob: Model Runner V2's ``compute_token_logprobs``.

A GatedDeltaNet layer does not compute a prompt token and a decoded token the same way, so the forward has to know
which tokens the sampler decoded. They are the tokens after the first position the loss trains on, the prompt's last
token, whose logits come from the prefill. The trained positions come from ``vllm_aligned_loss_mask`` when the
trainer passes it, and from ``labels != -100`` otherwise. A later prompt turn that the sampler prefilled after a
completion is computed here as decoded, so a multi-turn sample does not match from that turn on.

None of vLLM's GatedDeltaNet kernels has a backward. When gradients are enabled the model also runs its stock forward
and LM-head logprobs, and the returned logprobs take the aligned values with the stock forward's gradient. That
gradient belongs to a forward that differs from the returned values in the last bits.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

import torch

from arctic_platform.model.implementations.vllm_aligned import ops

SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5_text"})
LOSS_MASK_KWARG = "vllm_aligned_loss_mask"
_MLP_TOKEN_CHUNK = 8192
_KERNEL_BLOCK_ALIGNMENT = 16


def _rope_parameters(config) -> dict:
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    return dict(rope)


def check_supported_config(config) -> None:
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"the vLLM-aligned Qwen3.5 forward covers model types {sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}"
        )
    if not getattr(config, "attn_output_gate", True):
        raise ValueError("the vLLM-aligned Qwen3.5 forward expects gated attention (attn_output_gate)")
    if getattr(config, "hidden_act", None) not in ("silu", "swish"):
        raise ValueError(f"the vLLM-aligned Qwen3.5 forward expects silu, got {getattr(config, 'hidden_act', None)!r}")
    rope_type = _rope_parameters(config).get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"the vLLM-aligned Qwen3.5 forward covers the default rotary embedding, got {rope_type!r}")
    if getattr(config, "layer_types", None) is None:
        raise ValueError("the vLLM-aligned Qwen3.5 forward needs config.layer_types")


def hybrid_kv_block_size(config) -> int:
    """The attention page size vLLM 0.32 sets for this hybrid, so that one page holds a GatedDeltaNet state.

    vLLM's ``_align_hybrid_block_size`` with the FLASH_ATTN backend, a bf16 KV cache and the ``mamba_ssm_dtype`` of the
    config: 16 tokens times the number of 16-token attention pages a conv state plus a recurrent state fills. The
    sampler logs the value as "Setting attention block size to N tokens".
    """
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    attn_bytes_per_token = 2 * config.num_key_value_heads * head_dim * 2
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_bytes = (2 * key_dim + value_dim) * (config.linear_conv_kernel_dim - 1) * 2
    ssm_itemsize = 4 if str(getattr(config, "mamba_ssm_dtype", "float32")).endswith("float32") else 2
    ssm_bytes = (
        config.linear_num_value_heads * config.linear_value_head_dim * config.linear_key_head_dim * ssm_itemsize
    )
    pages = math.ceil((conv_bytes + ssm_bytes) / (_KERNEL_BLOCK_ALIGNMENT * attn_bytes_per_token))
    return _KERNEL_BLOCK_ALIGNMENT * pages


@dataclass(frozen=True)
class _Geometry:
    eps: float
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float | int
    mrope_section: tuple[int, ...] | None
    max_position: int
    kv_block_size: int

    @classmethod
    def from_config(cls, config, *, kv_block_size: int) -> _Geometry:
        rope = _rope_parameters(config)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial = rope.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
        mrope_section = rope.get("mrope_section")
        return cls(
            eps=float(config.rms_norm_eps),
            num_heads=int(config.num_attention_heads),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(head_dim),
            rotary_dim=int(head_dim * partial),
            rope_theta=rope.get("rope_theta", getattr(config, "rope_theta", 10000)),
            mrope_section=tuple(mrope_section) if mrope_section else None,
            max_position=int(config.max_position_embeddings),
            kv_block_size=int(kv_block_size),
        )


def _fused_qk_rmsnorm_rope_gate():
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate

    return fused_qk_rmsnorm_rope_gate


def _causal_conv1d_fn():
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    return causal_conv1d_fn


def _fla_ops():
    from vllm.third_party.flash_linear_attention import ops as fla_ops

    return fla_ops


def _gated_rms_norm():
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

    return rmsnorm_fn


def _pin_gdn_kernels() -> None:
    from arctic_platform.inference.vllm.gdn_batch_invariance import pin_gdn_kernels

    pin_gdn_kernels()


@dataclass(frozen=True)
class GdnLayout:
    """Which tokens of each packed sequence the sampler prefilled and which it decoded, and the state slots it used.

    A sequence is prefilled up to and including its first trained position and decoded after it. Slot 0 of each state
    buffer is a scratch slot; sequence ``i`` uses slot ``i + 1``.
    """

    seq_cu: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    prefill_index: torch.Tensor
    prefill_cu: torch.Tensor
    decode_index: torch.Tensor
    decode_cu: torch.Tensor
    decode_state_indices: torch.Tensor

    @property
    def num_seqs(self) -> int:
        return int(self.seq_cu.numel()) - 1

    @classmethod
    def build(cls, cu_seqlens: torch.Tensor, trained: torch.Tensor) -> GdnLayout:
        device = trained.device
        seq_cu = cu_seqlens.to(device=device, dtype=torch.int32)
        num_tokens = int(trained.numel())
        num_seqs = int(seq_cu.numel()) - 1
        starts = seq_cu[:-1].to(torch.long)
        ends = seq_cu[1:].to(torch.long)
        seq_of_token = torch.repeat_interleave(torch.arange(num_seqs, device=device), ends - starts)
        token = torch.arange(num_tokens, device=device)
        trained = trained.reshape(-1).to(torch.bool)
        first = torch.full((num_seqs,), num_tokens, dtype=torch.long, device=device)
        first = first.scatter_reduce(0, seq_of_token[trained], token[trained], reduce="amin", include_self=True)
        boundary = torch.minimum(first + 1, ends)
        decoded = token >= boundary[seq_of_token]
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        prefill_lens = (boundary - starts).to(torch.int32)
        decode_lens = (ends - boundary).to(torch.int32)
        max_decode = max(int(decode_lens.max().item()) if num_seqs else 0, 1)
        decode_state_indices = torch.zeros((num_seqs, max_decode), dtype=torch.int32, device=device)
        decode_state_indices[:, 0] = torch.arange(1, num_seqs + 1, dtype=torch.int32, device=device)
        return cls(
            seq_cu=seq_cu,
            cache_indices=torch.arange(1, num_seqs + 1, dtype=torch.int32, device=device),
            has_initial_state=torch.zeros(num_seqs, dtype=torch.bool, device=device),
            prefill_index=torch.nonzero(~decoded, as_tuple=False).flatten(),
            prefill_cu=torch.cat((zero, torch.cumsum(prefill_lens, 0, dtype=torch.int32))),
            decode_index=torch.nonzero(decoded, as_tuple=False).flatten(),
            decode_cu=torch.cat((zero, torch.cumsum(decode_lens, 0, dtype=torch.int32))),
            decode_state_indices=decode_state_indices,
        )


def _split_qkv(
    mixed_qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_k_heads: int,
    num_v_heads: int,
):
    """vLLM's ``rearrange_mixed_qkv``: contiguous ``(1, tokens, heads, dim)`` query, key and value."""
    tokens = mixed_qkv.shape[0]
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)
    q_size, v_size = tokens * key_dim, tokens * value_dim
    query = fused[:q_size].view(1, tokens, num_k_heads, key_dim // num_k_heads)
    key = fused[q_size : 2 * q_size].view(1, tokens, num_k_heads, key_dim // num_k_heads)
    value = fused[2 * q_size : 2 * q_size + v_size].view(1, tokens, num_v_heads, value_dim // num_v_heads)
    return query, key, value


def _gated_delta_net(layer: Any, hidden: torch.Tensor, gdn: GdnLayout) -> torch.Tensor:
    num_k_heads, num_v_heads = layer.num_k_heads, layer.num_v_heads
    head_k_dim, head_v_dim = layer.head_k_dim, layer.head_v_dim
    key_dim, value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim
    tokens = hidden.shape[0]
    fla_ops = _fla_ops()

    mixed_qkvz = ops.linear_kernel(hidden, torch.cat((layer.in_proj_qkv.weight, layer.in_proj_z.weight)))
    ba = ops.linear_kernel(hidden, torch.cat((layer.in_proj_b.weight, layer.in_proj_a.weight)))
    mixed_qkv, z = mixed_qkvz.split([conv_dim, value_dim], dim=-1)
    z = z.reshape(tokens, num_v_heads, head_v_dim)
    b, a = ba.chunk(2, dim=-1)
    b, a = b.contiguous(), a.contiguous()
    a_log = layer.A_log.detach().to(torch.float32)

    conv_weight = layer.conv1d.weight.view(conv_dim, layer.conv1d.weight.shape[-1])
    conv_states = hidden.new_zeros((gdn.num_seqs + 1, conv_dim, conv_weight.shape[-1] - 1))
    conv_out = _causal_conv1d_fn()(
        mixed_qkv.transpose(0, 1),
        conv_weight,
        layer.conv1d.bias,
        activation=layer.activation,
        conv_states=conv_states,
        has_initial_state=gdn.has_initial_state,
        cache_indices=gdn.cache_indices,
        query_start_loc=gdn.seq_cu,
    ).transpose(0, 1)

    core = torch.zeros((tokens, num_v_heads, head_v_dim), dtype=hidden.dtype, device=hidden.device)
    query, key, value, g, beta = fla_ops.fused_post_conv_prep(
        conv_output=conv_out[gdn.prefill_index],
        a=a[gdn.prefill_index],
        b=b[gdn.prefill_index],
        A_log=a_log,
        dt_bias=layer.dt_bias,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    )
    initial_state = torch.zeros(
        (gdn.num_seqs, num_v_heads, head_v_dim, head_k_dim),
        dtype=torch.float32,
        device=hidden.device,
    )
    prefill_out, final_state = fla_ops.chunk_gated_delta_rule(
        q=query.unsqueeze(0),
        k=key.unsqueeze(0),
        v=value.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=gdn.prefill_cu,
        use_qk_l2norm_in_kernel=False,
    )
    core[gdn.prefill_index] = prefill_out.squeeze(0).to(core.dtype)

    if gdn.decode_index.numel() > 0:
        state = torch.zeros(
            (gdn.num_seqs + 1, num_v_heads, head_v_dim, head_k_dim),
            dtype=torch.float32,
            device=hidden.device,
        )
        state[1:] = final_state.to(torch.float32)
        query, key, value = _split_qkv(conv_out[gdn.decode_index], key_dim, value_dim, num_k_heads, num_v_heads)
        decode_out, _ = fla_ops.fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a[gdn.decode_index],
            b=b[gdn.decode_index],
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=gdn.decode_cu,
            ssm_state_indices=gdn.decode_state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        core[gdn.decode_index] = decode_out.reshape(-1, num_v_heads, head_v_dim).to(core.dtype)

    normed = _gated_rms_norm()(
        core,
        layer.norm.weight,
        None,
        z=z,
        eps=layer.layer_norm_epsilon,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    )
    return ops.linear_kernel(normed.flatten(-2), layer.out_proj.weight)


def _full_attention(
    attn: Any,
    hidden: torch.Tensor,
    geometry: _Geometry,
    layout: ops.PackedLayout,
    rope_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    fa_version: int,
) -> torch.Tensor:
    tokens = hidden.shape[0]
    q_size = geometry.num_heads * geometry.head_dim
    kv_size = geometry.num_kv_heads * geometry.head_dim
    qkv = ops.linear_kernel(hidden, torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight)))
    q_gate, key, value = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
    query, key, gate = _fused_qk_rmsnorm_rope_gate()(
        q_gate,
        key,
        attn.q_norm.weight,
        attn.k_norm.weight,
        cos_sin_cache,
        rope_positions,
        geometry.eps,
        geometry.num_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        geometry.rotary_dim,
        norm_beta=1.0,
        mrope_section=list(geometry.mrope_section) if geometry.mrope_section and rope_positions.dim() == 2 else None,
    )
    out = ops.paged_attention_kernel(
        query.view(tokens, geometry.num_heads, geometry.head_dim),
        key.view(tokens, geometry.num_kv_heads, geometry.head_dim),
        value.view(tokens, geometry.num_kv_heads, geometry.head_dim),
        layout,
        geometry.head_dim**-0.5,
        fa_version,
    ).view(tokens, q_size)
    out = out * torch.sigmoid(gate)
    return ops.linear_kernel(out, attn.o_proj.weight)


def _mlp(mlp: Any, hidden: torch.Tensor) -> torch.Tensor:
    gate_up_weight = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight))
    out = torch.empty(
        (hidden.shape[0], mlp.down_proj.weight.shape[0]),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    for start in range(0, hidden.shape[0], _MLP_TOKEN_CHUNK):
        stop = min(start + _MLP_TOKEN_CHUNK, hidden.shape[0])
        gate_up = ops.linear_kernel(hidden[start:stop], gate_up_weight)
        out[start:stop] = ops.linear_kernel(ops.silu_and_mul_kernel(gate_up), mlp.down_proj.weight)
    return out


def _cos_sin_cache(backbone: Any, geometry: _Geometry, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    cache = getattr(backbone, "_vllm_aligned_cos_sin_cache", None)
    if cache is None or cache.device != device or cache.dtype != dtype:
        cache = ops.rotary_cos_sin_cache(
            rotary_dim=geometry.rotary_dim,
            base=geometry.rope_theta,
            max_position=geometry.max_position,
            dtype=dtype,
            device=device,
        )
        backbone._vllm_aligned_cos_sin_cache = cache
    return cache


@torch.no_grad()
def aligned_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    labels: torch.Tensor,
    trained: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """The sampler's logprob of each ``labels`` token, ``[tokens]`` in fp32; -100 scores 0."""
    backbone = model.model
    geometry: _Geometry = model._vllm_aligned_geometry
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    layout = ops.PackedLayout.build(position_ids, block_size=geometry.kv_block_size, cu_seqlens=cu_seqlens)
    gdn = GdnLayout.build(layout.cu_seqlens, trained)
    positions = layout.positions
    rope_positions = positions.unsqueeze(0).expand(3, -1) if geometry.mrope_section else positions
    hidden = backbone.embed_tokens(input_ids.reshape(-1))
    cos_sin_cache = _cos_sin_cache(backbone, geometry, hidden.dtype, positions.device)
    fa_version = ops.flash_attn_version(geometry.head_dim)

    residual = None
    for layer in backbone.layers:
        if residual is None:
            residual = hidden
            hidden, _ = ops.gemma_rms_norm_kernel(hidden, layer.input_layernorm.weight, geometry.eps)
        else:
            hidden, residual = ops.gemma_rms_norm_kernel(hidden, layer.input_layernorm.weight, geometry.eps, residual)
        if layer.block_type == "linear_attention":
            mixer_out = _gated_delta_net(layer.linear_attn, hidden, gdn)
        else:
            mixer_out = _full_attention(
                layer.self_attn,
                hidden,
                geometry,
                layout,
                rope_positions,
                cos_sin_cache,
                fa_version,
            )
        hidden, residual = ops.gemma_rms_norm_kernel(
            mixer_out, layer.post_attention_layernorm.weight, geometry.eps, residual
        )
        hidden = _mlp(layer.mlp, hidden)
    hidden, _ = ops.gemma_rms_norm_kernel(hidden, backbone.norm.weight, geometry.eps, residual)

    labels = labels.reshape(-1)
    ignored = labels == -100
    logprobs = ops.lm_head_logprobs_kernel(
        hidden,
        model.lm_head.weight,
        labels.masked_fill(ignored, 0).to(torch.long),
        float(temperature),
        model._vllm_aligned_token_chunk_size,
        64,
    )
    return logprobs.masked_fill(ignored, 0.0)


class _AlignedValues(torch.autograd.Function):
    """Return ``aligned`` and pass the incoming gradient to ``reference``."""

    @staticmethod
    def forward(ctx, reference, aligned):
        return aligned.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature=None,
    action_masks=None,
    dss_compute_logprobs: bool = False,
    group_token_ids=None,
    **kwargs,
):
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import _scalar_temperature

    loss_mask = kwargs.pop(LOSS_MASK_KWARG, None)
    stock_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        logits_to_keep=logits_to_keep,
        temperature=temperature,
        action_masks=action_masks,
        dss_compute_logprobs=dss_compute_logprobs,
        group_token_ids=group_token_ids,
        **kwargs,
    )
    if not dss_compute_logprobs:
        return self._vllm_aligned_stock_forward(**stock_kwargs)
    if inputs_embeds is not None or action_masks is not None or group_token_ids is not None:
        raise ValueError("the vLLM-aligned Qwen3.5 logprobs take input_ids, without action_masks or group_token_ids")
    if not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
        raise ValueError("the vLLM-aligned Qwen3.5 logprobs score every position; logits_to_keep is not supported")
    if labels is None or input_ids is None:
        raise ValueError("dss_compute_logprobs requires input_ids and labels")
    if position_ids is None:
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).expand_as(input_ids)
    trained = loss_mask if loss_mask is not None else labels != -100
    aligned = aligned_logprobs(
        self,
        input_ids,
        position_ids,
        labels.to(input_ids.device),
        trained.to(input_ids.device),
        cu_seqlens=kwargs.get("cu_seq_lens_q"),
        temperature=_scalar_temperature(temperature),
    ).view(input_ids.shape)
    if not torch.is_grad_enabled():
        return {"logprobs": aligned}
    reference = self._vllm_aligned_stock_forward(**stock_kwargs)["logprobs"]
    return {"logprobs": _AlignedValues.apply(reference, aligned.to(reference.dtype))}


def apply_vllm_aligned_qwen3_5(model: Any, *, token_chunk_size: int, kv_block_size: int | None = None) -> None:
    """Make a transformers ``Qwen3_5ForCausalLM``'s ``dss_compute_logprobs`` forward return vLLM's logprobs.

    Call it after the chunked LM-head patch, whose forward supplies the gradient. ``kv_block_size`` defaults to
    :func:`hybrid_kv_block_size`.
    """
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import _text_config

    config = _text_config(model)
    check_supported_config(config)
    if getattr(model, "model", None) is None or getattr(model, "lm_head", None) is None:
        raise ValueError("the vLLM-aligned forward needs a causal LM with .model and .lm_head")
    if not getattr(model, "_dss_chunked_lm_head_logprobs", False):
        raise ValueError("apply the chunked LM-head patch before the vLLM-aligned Qwen3.5 forward")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    ops.require_batch_invariant_env()
    _pin_gdn_kernels()
    block_size = hybrid_kv_block_size(config) if kv_block_size is None else int(kv_block_size)
    model._vllm_aligned_geometry = _Geometry.from_config(config, kv_block_size=block_size)
    model._vllm_aligned_token_chunk_size = int(token_chunk_size)
    model._vllm_aligned_stock_forward = model.forward
    model._vllm_aligned_takes_loss_mask = True
    model.forward = types.MethodType(_causal_lm_forward, model)
    model._vllm_aligned = True
