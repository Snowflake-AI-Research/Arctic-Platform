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
on text, on CUDA, with ``VLLM_BATCH_INVARIANT=1``, the FLASH_ATTN backend,
``additional_config={"gdn_prefill_backend": "triton"}``, ``VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`` and
:mod:`arctic_platform.inference.vllm.gdn_batch_invariance` enabled (``ARCTIC_GDN_BATCH_INVARIANT=1``); the trainer
pins the same GDN kernel configs. Tensor parallelism, chunked prefill and prefix caching are reproduced as
:func:`apply_vllm_aligned_qwen3_5` and :mod:`~arctic_platform.model.implementations.vllm_aligned.turn_plan` describe.
Per decoder layer:

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
completion is computed that way as decoded, so a multi-turn row needs ``vllm_aligned_plan``
(:mod:`~arctic_platform.model.implementations.vllm_aligned.turn_plan`): each turn is then replayed as the request
that computed it, resuming from the prefix-cache pages and state the sampler reused.

The backward follows the Qwen3 path: each operator's forward is vLLM's kernel, and its backward is the trainer's
math for the same function at the saved inputs. vLLM's GatedDeltaNet kernels have no backward, so that layer's
backward recomputes it with the trainer's chunked gated delta rule (``fla``) over whole sequences and
differentiates that; the prompt/decode split does not change the function, only its rounding. The decoder layers
after the first are checkpointed in pairs when gradients are enabled.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any

import torch

from arctic_platform.model.implementations.gpu.sp.collectives import gather_sequence
from arctic_platform.model.implementations.gpu.sp.collectives import sequence_head_all_to_all
from arctic_platform.model.implementations.vllm_aligned import ops
from arctic_platform.model.implementations.vllm_aligned.turn_plan import PLAN_KWARG
from arctic_platform.model.implementations.vllm_aligned.turn_plan import build_turn_layout

SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5_text"})
LOSS_MASK_KWARG = "vllm_aligned_loss_mask"
# A checkpoint boundary holds both the hidden stream and the residual, so layers are checkpointed in pairs to keep the
# saved activations at the size of a stock per-layer checkpoint.
_LAYERS_PER_CHECKPOINT = 2
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


def _check_tensor_parallel(config, tensor_parallel: int) -> None:
    """Refuse a tensor-parallel size the sampler could not shard this config into."""
    if tensor_parallel < 1:
        raise ValueError("tensor_parallel must be positive")
    names = ("num_attention_heads", "intermediate_size", "linear_num_key_heads", "linear_num_value_heads")
    for name in names:
        if getattr(config, name) % tensor_parallel:
            raise ValueError(f"config.{name}={getattr(config, name)} does not split across {tensor_parallel} ranks")


def _check_sequence_parallel(config, sp_size: int) -> None:
    """Refuse a sequence-parallel size whose head shards the mixers cannot use."""
    for name in ("num_attention_heads", "linear_num_key_heads", "linear_num_value_heads"):
        if getattr(config, name) % sp_size:
            raise ValueError(f"config.{name}={getattr(config, name)} does not split across {sp_size} SP ranks")
    kv_heads = config.num_key_value_heads
    if kv_heads % sp_size and sp_size % kv_heads:
        raise ValueError(f"{kv_heads} key/value heads neither split across nor divide {sp_size} SP ranks")


def hybrid_kv_block_size(config, tensor_parallel: int = 1) -> int:
    """The attention page size vLLM 0.32 sets for this hybrid, so that one page holds a GatedDeltaNet state.

    vLLM's ``_align_hybrid_block_size`` with the FLASH_ATTN backend, a bf16 KV cache and the ``mamba_ssm_dtype`` of the
    config, on one rank's shard: 16 tokens times the number of 16-token attention pages a conv state plus a recurrent
    state fills. The sampler logs the value as "Setting attention block size to N tokens".
    """
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_heads = max(config.num_key_value_heads // tensor_parallel, 1)
    attn_bytes_per_token = 2 * kv_heads * head_dim * 2
    key_dim = config.linear_num_key_heads // tensor_parallel * config.linear_key_head_dim
    value_heads = config.linear_num_value_heads // tensor_parallel
    value_dim = value_heads * config.linear_value_head_dim
    conv_bytes = (2 * key_dim + value_dim) * (config.linear_conv_kernel_dim - 1) * 2
    ssm_itemsize = 4 if str(getattr(config, "mamba_ssm_dtype", "float32")).endswith("float32") else 2
    ssm_bytes = value_heads * config.linear_value_head_dim * config.linear_key_head_dim * ssm_itemsize
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
    tensor_parallel: int = 1
    prefill_chunk: int = 0
    conv_width: int = 4

    @classmethod
    def from_config(cls, config, *, kv_block_size: int, tensor_parallel: int = 1, prefill_chunk: int = 0) -> _Geometry:
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
            tensor_parallel=int(tensor_parallel),
            prefill_chunk=int(prefill_chunk),
            conv_width=int(config.linear_conv_kernel_dim),
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
class GdnSpan:
    """One sequence of GatedDeltaNet tokens as the sampler computed it, for :meth:`GdnLayout.from_spans`.

    The span holds packed tokens ``[start, start + length)``. ``segments`` are its kernel calls in order, as
    ``(decode, lo, hi)`` offsets into the span: a prefill chunk runs the chunked scan and a decode run the recurrent
    update, each resuming from the state the previous one left. ``resume = (span, offset)`` starts the span from
    the conv and recurrent state another span had after its first ``offset`` tokens, as a request resumes from a
    prefix-cache hit; that offset must end one of the other span's segments.
    """

    start: int
    length: int
    segments: tuple[tuple[bool, int, int], ...]
    resume: tuple[int, int] | None = None


@dataclass(frozen=True)
class GdnStep:
    """One kernel call: a segment of each of several spans.

    ``index`` holds the segments' tokens, ``cu_seqlens`` their boundaries and ``spans`` the span of each. A span's
    state is in slot ``span + 1``; a decode call reads it from its run's first state index and writes the final
    state back through its last. After the call the states of ``export_src`` are copied to ``export_dst``.
    """

    decode: bool
    index: torch.Tensor
    cu_seqlens: torch.Tensor
    spans: torch.Tensor
    state_indices: torch.Tensor | None = None
    export_src: torch.Tensor | None = None
    export_dst: torch.Tensor | None = None


@dataclass(frozen=True)
class GdnLayout:
    """How the sampler computed the GatedDeltaNet layers of a packed batch, and the state slots it used.

    ``steps`` are the sampler's kernel calls in dependency order. ``reference_steps`` compute the same function in
    the trainer's chunked rule, one call per stretch between the positions other spans resume from. ``prev_index``
    gives, for each token and each of the conv's ``width - 1`` earlier taps, the token whose input that tap reads
    (``-1`` for zero), across a resume into the span it resumes from. ``conv_init_spans`` start from those
    earlier inputs: their conv states are ``conv_init_index``'s tokens.
    """

    seq_cu: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    steps: tuple[GdnStep, ...]
    reference_steps: tuple[GdnStep, ...]
    prev_index: torch.Tensor
    conv_init_spans: torch.Tensor
    conv_init_index: torch.Tensor

    @property
    def num_seqs(self) -> int:
        return int(self.seq_cu.numel()) - 1

    @property
    def resumes(self) -> bool:
        return bool(self.has_initial_state.any().item())

    @classmethod
    def build(
        cls, cu_seqlens: torch.Tensor, trained: torch.Tensor, prefill_chunk: int = 0, conv_width: int = 4
    ) -> GdnLayout:
        """One span per packed sequence, prefilled through its first trained position and decoded after it.

        With a prefill chunk size ``C`` the prompt is prefilled ``C`` tokens at a time from the sequence start. A
        one-token chunk is a decode step in vLLM (its query is one token long), so a one-token tail of the prompt
        is decoded too.
        """
        device = trained.device
        num_tokens = int(trained.numel())
        bounds = cu_seqlens.to(torch.long).tolist()
        trained_positions = torch.nonzero(trained.reshape(-1).to(torch.bool), as_tuple=False).flatten().tolist()
        spans = []
        cursor = 0
        for start, end in zip(bounds[:-1], bounds[1:]):
            while cursor < len(trained_positions) and trained_positions[cursor] < start:
                cursor += 1
            first = trained_positions[cursor] if cursor < len(trained_positions) else num_tokens
            prompt_end = min(first + 1, end) - start
            chunk = prefill_chunk if prefill_chunk > 0 else max(prompt_end, 1)
            cuts = list(range(0, prompt_end, chunk)) + [prompt_end]
            segments = [(False, lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])]
            spans.append(GdnSpan(start, end - start, tuple(segments) + ((True, prompt_end, end - start),)))
        return cls.from_spans(spans, num_tokens=num_tokens, conv_width=conv_width, device=device)

    @classmethod
    def from_spans(cls, spans: list[GdnSpan], *, num_tokens: int, conv_width: int, device: torch.device) -> GdnLayout:
        spans = [_normalized_span(span) for span in spans]
        levels = _span_levels(spans)
        exports: dict[tuple[int, int], list[int]] = {}
        for index, span in enumerate(spans):
            if span.resume is not None:
                provider, offset = span.resume
                if offset < conv_width - 1 or offset not in {hi for _, _, hi in spans[provider].segments}:
                    raise ValueError(f"span {index} resumes at offset {offset}, which no segment of {provider} ends")
                exports.setdefault((provider, offset), []).append(index)

        steps: list[GdnStep] = []
        reference_steps: list[GdnStep] = []
        for level in range(max(levels, default=-1) + 1):
            members = [i for i, value in enumerate(levels) if value == level]
            for k in range(max(len(spans[i].segments) for i in members)):
                for decode in (False, True):
                    pieces = [
                        (i, lo, hi) for i in members if k < len(spans[i].segments)
                        for seg_decode, lo, hi in (spans[i].segments[k],) if seg_decode == decode
                    ]  # fmt: skip
                    if pieces:
                        steps.append(_gdn_step(spans, pieces, exports, decode, device))
            reference_pieces = {i: _reference_pieces(i, spans[i], exports) for i in members}
            for k in range(max(len(pieces) for pieces in reference_pieces.values())):
                pieces = [(i, *reference_pieces[i][k]) for i in members if k < len(reference_pieces[i])]
                reference_steps.append(_gdn_step(spans, pieces, exports, False, device))

        token_span = torch.empty(num_tokens, dtype=torch.long)
        offset = torch.empty(num_tokens, dtype=torch.long)
        for index, span in enumerate(spans):
            token_span[span.start : span.start + span.length] = index
            offset[span.start : span.start + span.length] = torch.arange(span.length)
        prev_index = torch.full((num_tokens, max(conv_width - 1, 0)), -1, dtype=torch.long)
        for shift in range(1, conv_width):
            prev_index[:, shift - 1] = _earlier_token(spans, token_span, offset, shift)
        init_spans = [i for i, span in enumerate(spans) if span.resume is not None]
        init_index = [
            [spans[spans[i].resume[0]].start + spans[i].resume[1] - conv_width + 1 + tap for tap in range(conv_width - 1)]
            for i in init_spans
        ]  # fmt: skip
        lengths = torch.tensor([span.length for span in spans], dtype=torch.int32)
        return cls(
            seq_cu=torch.cat((torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, 0, dtype=torch.int32))).to(
                device
            ),
            cache_indices=torch.arange(1, len(spans) + 1, dtype=torch.int32, device=device),
            has_initial_state=torch.tensor(
                [span.resume is not None for span in spans], dtype=torch.bool, device=device
            ),
            steps=tuple(steps),
            reference_steps=tuple(reference_steps),
            prev_index=prev_index.to(device),
            conv_init_spans=torch.tensor(init_spans, dtype=torch.long, device=device),
            conv_init_index=torch.tensor(init_index, dtype=torch.long, device=device).reshape(
                -1, max(conv_width - 1, 0)
            ),
        )


def _normalized_span(span: GdnSpan) -> GdnSpan:
    """Drop empty segments, run a one-token prefill chunk as the decode step vLLM makes of it, merge decode runs."""
    segments: list[tuple[bool, int, int]] = []
    for decode, lo, hi in span.segments:
        if hi <= lo:
            continue
        decode = decode or hi - lo == 1
        if segments and segments[-1][0] and decode and segments[-1][2] == lo:
            segments[-1] = (True, segments[-1][1], hi)
        else:
            segments.append((decode, lo, hi))
    covered = [lo for _, lo, _ in segments] + [span.length]
    if (segments and segments[0][1] != 0) or any(hi != nxt for (_, _, hi), nxt in zip(segments, covered[1:])):
        raise ValueError(f"segments {span.segments} do not tile a span of {span.length} tokens")
    return GdnSpan(span.start, span.length, tuple(segments), span.resume)


def _span_levels(spans: list[GdnSpan]) -> list[int]:
    """Each span's distance from a span that starts fresh, following ``resume``."""
    levels: list[int | None] = [None] * len(spans)

    def level(index: int, seen: frozenset) -> int:
        if levels[index] is None:
            resume = spans[index].resume
            if resume is not None and resume[0] in seen | {index}:
                raise ValueError(f"span {index} resumes in a cycle")
            levels[index] = 0 if resume is None else level(resume[0], seen | {index}) + 1
        return levels[index]

    return [level(i, frozenset()) for i in range(len(spans))]


def _reference_pieces(index: int, span: GdnSpan, exports: dict) -> list[tuple[int, int]]:
    cuts = sorted({0, span.length} | {offset for provider, offset in exports if provider == index})
    return [(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:]) if hi > lo]


def _gdn_step(spans, pieces, exports, decode: bool, device) -> GdnStep:
    lengths = torch.tensor([hi - lo for _, lo, hi in pieces], dtype=torch.int32)
    index = torch.cat([torch.arange(spans[i].start + lo, spans[i].start + hi) for i, lo, hi in pieces])
    state_indices = None
    if decode:
        state_indices = torch.zeros((len(pieces), int(lengths.max())), dtype=torch.int32)
        for row, ((i, _, _), length) in enumerate(zip(pieces, lengths.tolist())):
            state_indices[row, 0] = state_indices[row, length - 1] = i + 1
    pairs = [(i, dst) for i, _, hi in pieces for dst in exports.get((i, hi), ())]
    return GdnStep(
        decode=decode,
        index=index.to(device),
        cu_seqlens=torch.cat((torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, 0, dtype=torch.int32))).to(
            device
        ),
        spans=torch.tensor([i for i, _, _ in pieces], dtype=torch.long, device=device),
        state_indices=None if state_indices is None else state_indices.to(device),
        export_src=torch.tensor([s for s, _ in pairs], dtype=torch.long, device=device) if pairs else None,
        export_dst=torch.tensor([d for _, d in pairs], dtype=torch.long, device=device) if pairs else None,
    )


def _earlier_token(spans: list[GdnSpan], token_span: torch.Tensor, offset: torch.Tensor, shift: int) -> torch.Tensor:
    """The token ``shift`` places before each token in its lineage, or ``-1`` before the lineage starts."""
    out = torch.full_like(token_span, -1)
    for index, span in enumerate(spans):
        rows = slice(span.start, span.start + span.length)
        back = offset[rows] - shift
        own = back >= 0
        out[rows][own] = span.start + back[own]
        if span.resume is not None and bool((~own).any()):
            provider, resume_offset = span.resume
            out[rows][~own] = spans[provider].start + resume_offset + back[~own]
    return out


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


def _reference_gated_delta_rule():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def _reference_causal_conv(x: torch.Tensor, weight: torch.Tensor, prev_index: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution and SiLU over packed sequences, differentiable.

    ``x`` is ``[tokens, channels]``, ``weight`` ``[channels, width]``; tap ``s`` of a token reads the token
    ``prev_index[:, s - 1]``, or zero where that is ``-1``.
    """
    width = weight.shape[1]
    out = x * weight[:, width - 1]
    for shift in range(1, width):
        source = prev_index[:, shift - 1]
        valid = (source >= 0).to(x.dtype)[:, None]
        out = out + x[source.clamp(min=0)] * valid * weight[:, width - 1 - shift]
    return torch.nn.functional.silu(out)


@dataclass(frozen=True)
class _GdnDims:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    activation: str

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim


def _gated_delta_net_core_kernel(mixed_qkv, a, b, A_log, dt_bias, conv_weight, dims: _GdnDims, gdn: GdnLayout):
    """The sampler's conv and delta rule, ``[tokens, value heads, head_v_dim]``; ``mixed_qkv`` keeps its strides."""
    fla_ops = _fla_ops()
    tokens = mixed_qkv.shape[0]
    conv_dim = 2 * dims.key_dim + dims.value_dim
    a_log = A_log.to(torch.float32)
    conv_weight = conv_weight.view(conv_dim, conv_weight.shape[-1])
    conv_states = mixed_qkv.new_zeros((gdn.num_seqs + 1, conv_dim, conv_weight.shape[-1] - 1))
    if gdn.conv_init_spans.numel() > 0:
        conv_states[gdn.conv_init_spans + 1] = mixed_qkv[gdn.conv_init_index].transpose(1, 2)
    conv_out = _causal_conv1d_fn()(
        mixed_qkv.transpose(0, 1),
        conv_weight,
        None,
        activation=dims.activation,
        conv_states=conv_states,
        has_initial_state=gdn.has_initial_state,
        cache_indices=gdn.cache_indices,
        query_start_loc=gdn.seq_cu,
    ).transpose(0, 1)

    core = torch.zeros((tokens, dims.num_v_heads, dims.head_v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    state_shape = (dims.num_v_heads, dims.head_v_dim, dims.head_k_dim)
    state = torch.zeros((gdn.num_seqs + 1, *state_shape), dtype=torch.float32, device=mixed_qkv.device)
    for step in gdn.steps:
        slots = step.spans + 1
        if not step.decode:
            query, key, value, g, beta = fla_ops.fused_post_conv_prep(
                conv_output=conv_out[step.index],
                a=a[step.index],
                b=b[step.index],
                A_log=a_log,
                dt_bias=dt_bias,
                num_k_heads=dims.num_k_heads,
                head_k_dim=dims.head_k_dim,
                head_v_dim=dims.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            out, step_state = fla_ops.chunk_gated_delta_rule(
                q=query.unsqueeze(0),
                k=key.unsqueeze(0),
                v=value.unsqueeze(0),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=state[slots].contiguous(),
                output_final_state=True,
                cu_seqlens=step.cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
            core[step.index] = out.squeeze(0).to(core.dtype)
            state[slots] = step_state.to(torch.float32)
        else:
            query, key, value = _split_qkv(
                conv_out[step.index], dims.key_dim, dims.value_dim, dims.num_k_heads, dims.num_v_heads
            )
            out, _ = fla_ops.fused_sigmoid_gating_delta_rule_update(
                A_log=a_log,
                a=a[step.index],
                b=b[step.index],
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                initial_state=state,
                inplace_final_state=True,
                cu_seqlens=step.cu_seqlens,
                ssm_state_indices=step.state_indices,
                use_qk_l2norm_in_kernel=True,
            )
            core[step.index] = out.reshape(-1, dims.num_v_heads, dims.head_v_dim).to(core.dtype)
        if step.export_src is not None:
            state[step.export_dst + 1] = state[step.export_src + 1]
    return core


def _gated_delta_net_core_reference(mixed_qkv, a, b, A_log, dt_bias, conv_weight, dims: _GdnDims, gdn: GdnLayout):
    """The same function as :func:`_gated_delta_net_core_kernel`, in the trainer's differentiable kernels."""
    tokens = mixed_qkv.shape[0]
    conv_weight = conv_weight.reshape(conv_weight.shape[0], conv_weight.shape[-1])
    conv_out = _reference_causal_conv(mixed_qkv.float(), conv_weight.float(), gdn.prev_index).to(mixed_qkv.dtype)
    query, key, value = conv_out.split([dims.key_dim, dims.key_dim, dims.value_dim], dim=-1)
    groups = dims.num_v_heads // dims.num_k_heads
    query = query.reshape(tokens, dims.num_k_heads, dims.head_k_dim).repeat_interleave(groups, dim=1)
    key = key.reshape(tokens, dims.num_k_heads, dims.head_k_dim).repeat_interleave(groups, dim=1)
    value = value.reshape(tokens, dims.num_v_heads, dims.head_v_dim)
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = b.sigmoid()
    rule = _reference_gated_delta_rule()
    if not gdn.resumes and len(gdn.reference_steps) == 1:
        out, _ = rule(
            query[None],
            key[None],
            value[None],
            g=g[None],
            beta=beta[None],
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=gdn.seq_cu.to(torch.long),
        )
        return out.reshape(tokens, dims.num_v_heads, dims.head_v_dim)

    # fla's state is [value heads, head_k_dim, head_v_dim]; it only flows between the reference's own calls.
    states: list[torch.Tensor | None] = [None] * gdn.num_seqs
    zero = mixed_qkv.new_zeros((dims.num_v_heads, dims.head_k_dim, dims.head_v_dim), dtype=torch.float32)
    pieces, rows = [], []
    for step in gdn.reference_steps:
        spans = step.spans.tolist()
        index = step.index
        out, final = rule(
            query[index][None],
            key[index][None],
            value[index][None],
            g=g[index][None],
            beta=beta[index][None],
            initial_state=torch.stack([zero if states[i] is None else states[i] for i in spans]),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=step.cu_seqlens.to(torch.long),
        )
        pieces.append(out[0])
        rows.append(index)
        for position, i in enumerate(spans):
            states[i] = final[position]
        if step.export_src is not None:
            for src, dst in zip(step.export_src.tolist(), step.export_dst.tolist()):
                states[dst] = states[src]
    core = torch.cat(pieces)[torch.argsort(torch.cat(rows))]
    return core.reshape(tokens, dims.num_v_heads, dims.head_v_dim)


class _GatedDeltaNetCore(torch.autograd.Function):
    """The GatedDeltaNet conv and recurrence with the sampler's values and the trainer's gradient.

    vLLM's kernels have no backward, and the recurrent decode overwrites its state in place, so the backward
    recomputes the layer at the saved inputs with the trainer's chunked gated delta rule over whole sequences, the
    same function as the sampler's prompt and decode kernels, and differentiates that.
    """

    @staticmethod
    def forward(ctx, mixed_qkv, a, b, A_log, dt_bias, conv_weight, dims, gdn):
        ctx.save_for_backward(mixed_qkv, a, b, A_log, dt_bias, conv_weight)
        ctx.dims = dims
        ctx.gdn = gdn
        return _gated_delta_net_core_kernel(mixed_qkv, a, b, A_log, dt_bias, conv_weight, dims, gdn)

    @staticmethod
    def backward(ctx, grad_core):
        with torch.enable_grad():
            leaves = [tensor.detach().requires_grad_(True) for tensor in ctx.saved_tensors]
            core = _gated_delta_net_core_reference(*leaves, ctx.dims, ctx.gdn)
            grads = torch.autograd.grad(core, leaves, grad_core, allow_unused=True)
        return (*grads, None, None)


def _gated_rms_norm_reference(core, z, weight, eps):
    hidden = core.float()
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return (hidden * torch.nn.functional.silu(z.float())).to(core.dtype)


class _GatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, core, z, weight, eps):
        ctx.save_for_backward(core, z, weight)
        ctx.eps = eps
        return _gated_rms_norm()(
            core, weight, None, z=z, eps=eps, group_size=None, norm_before_gate=True, activation="silu"
        )

    @staticmethod
    def backward(ctx, grad_out):
        with torch.enable_grad():
            leaves = [tensor.detach().requires_grad_(True) for tensor in ctx.saved_tensors]
            out = _gated_rms_norm_reference(*leaves, ctx.eps)
            grads = torch.autograd.grad(out, leaves, grad_out)
        return (*grads, None)


def _qk_norm_rope_gate_reference(q_gate, key, q_weight, k_weight, cos_sin_cache, positions, geometry: _Geometry):
    tokens, heads, dim, rotary = q_gate.shape[0], geometry.num_heads, geometry.head_dim, geometry.rotary_dim
    q_gate = q_gate.view(tokens, heads, 2 * dim)
    query, gate = q_gate[..., :dim], q_gate[..., dim:]

    def gemma_norm(x, weight):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + geometry.eps) * (weight.float() + 1.0)

    query = gemma_norm(query, q_weight)
    key = gemma_norm(key.view(tokens, geometry.num_kv_heads, dim), k_weight)
    rows = positions[0] if positions.dim() == 2 else positions
    cos, sin = cos_sin_cache[rows].float().chunk(2, dim=-1)
    cos, sin, half = cos[:, None, :], sin[:, None, :], rotary // 2

    def rotate(x):
        first, second = x[..., :half], x[..., half:rotary]
        return torch.cat((first * cos - second * sin, second * cos + first * sin, x[..., rotary:]), dim=-1)

    dtype = q_gate.dtype
    return (
        rotate(query).to(dtype).reshape(tokens, -1),
        rotate(key).to(dtype).reshape(tokens, -1),
        gate.reshape(tokens, -1),
    )


class _QKNormRopeGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_gate, key, q_weight, k_weight, cos_sin_cache, positions, geometry):
        ctx.save_for_backward(q_gate, key, q_weight, k_weight, cos_sin_cache, positions)
        ctx.geometry = geometry
        return _fused_qk_rmsnorm_rope_gate()(
            q_gate,
            key,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            geometry.eps,
            geometry.num_heads,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.rotary_dim,
            norm_beta=1.0,
            mrope_section=list(geometry.mrope_section) if geometry.mrope_section and positions.dim() == 2 else None,
        )

    @staticmethod
    def backward(ctx, grad_query, grad_key, grad_gate):
        q_gate, key, q_weight, k_weight, cos_sin_cache, positions = ctx.saved_tensors
        with torch.enable_grad():
            leaves = [tensor.detach().requires_grad_(True) for tensor in (q_gate, key, q_weight, k_weight)]
            outputs = _qk_norm_rope_gate_reference(*leaves, cos_sin_cache, positions, ctx.geometry)
            pairs = [(o, g) for o, g in zip(outputs, (grad_query, grad_key, grad_gate)) if g is not None]
            grads = torch.autograd.grad([o for o, _ in pairs], leaves, [g for _, g in pairs], allow_unused=True)
        return (*grads, None, None, None)


@dataclass(frozen=True)
class _ForwardContext:
    geometry: _Geometry
    layout: ops.PackedLayout
    gdn: GdnLayout
    rope_positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    fa_version: int
    sp_group: Any = None


def _to_heads(group, x: torch.Tensor) -> torch.Tensor:
    """``[shard tokens, heads, ...]`` to ``[all tokens, heads / sp, ...]`` across the sequence-parallel group."""
    return x if group is None else sequence_head_all_to_all(group, x[None], scatter_dim=2, gather_dim=1)[0]


def _to_tokens(group, x: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`_to_heads`."""
    return x if group is None else sequence_head_all_to_all(group, x[None], scatter_dim=1, gather_dim=2)[0]


def _gated_delta_net(layer: Any, hidden: torch.Tensor, ctx: _ForwardContext) -> torch.Tensor:
    """The GatedDeltaNet mixer; under sequence parallelism the conv and recurrence see all tokens and a head shard.

    A rank's shard holds the same heads and channels a tensor-parallel sampler rank holds: query, key and value
    channels of its key heads and value heads, in that order, with their conv channels, ``A_log`` and ``dt_bias``.
    """
    dims = _GdnDims(layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, layer.activation)
    tokens = hidden.shape[0]
    mixed_qkvz = ops.linear(hidden, torch.cat((layer.in_proj_qkv.weight, layer.in_proj_z.weight)))
    ba = ops.linear(hidden, torch.cat((layer.in_proj_b.weight, layer.in_proj_a.weight)))
    mixed_qkv, z = mixed_qkvz.split([2 * dims.key_dim + dims.value_dim, dims.value_dim], dim=-1)
    z = z.reshape(tokens, dims.num_v_heads, dims.head_v_dim)
    b, a = ba.chunk(2, dim=-1)
    a, b = a.contiguous(), b.contiguous()
    A_log, dt_bias, conv_weight = layer.A_log, layer.dt_bias, layer.conv1d.weight
    group = ctx.sp_group
    if group is not None:
        sp_size, rank = torch.distributed.get_world_size(group), torch.distributed.get_rank(group)
        query, key, value = mixed_qkv.split([dims.key_dim, dims.key_dim, dims.value_dim], dim=-1)
        query = _to_heads(group, query.reshape(tokens, dims.num_k_heads, dims.head_k_dim))
        key = _to_heads(group, key.reshape(tokens, dims.num_k_heads, dims.head_k_dim))
        value = _to_heads(group, value.reshape(tokens, dims.num_v_heads, dims.head_v_dim))
        mixed_qkv = torch.cat((query.flatten(1), key.flatten(1), value.flatten(1)), dim=-1)
        a, b = _to_heads(group, a[..., None])[..., 0].contiguous(), _to_heads(group, b[..., None])[..., 0].contiguous()
        k_dim, v_dim = dims.key_dim // sp_size, dims.value_dim // sp_size
        heads = slice(rank * (dims.num_v_heads // sp_size), (rank + 1) * (dims.num_v_heads // sp_size))
        conv_weight = torch.cat(
            (
                conv_weight[rank * k_dim : (rank + 1) * k_dim],
                conv_weight[dims.key_dim + rank * k_dim : dims.key_dim + (rank + 1) * k_dim],
                conv_weight[2 * dims.key_dim + rank * v_dim : 2 * dims.key_dim + (rank + 1) * v_dim],
            )
        )
        A_log, dt_bias = A_log[heads], dt_bias[heads]
        dims = _GdnDims(
            dims.num_k_heads // sp_size, dims.num_v_heads // sp_size, dims.head_k_dim, dims.head_v_dim, dims.activation
        )
    core = _GatedDeltaNetCore.apply(mixed_qkv, a, b, A_log, dt_bias, conv_weight, dims, ctx.gdn)
    core = _to_tokens(group, core)
    normed = _GatedRMSNorm.apply(core, z, layer.norm.weight, layer.layer_norm_epsilon)
    return ops.row_parallel_linear(normed.flatten(-2), layer.out_proj.weight, ctx.geometry.tensor_parallel)


def _full_attention(attn: Any, hidden: torch.Tensor, ctx: _ForwardContext) -> torch.Tensor:
    geometry = ctx.geometry
    tokens = hidden.shape[0]
    q_size = geometry.num_heads * geometry.head_dim
    kv_size = geometry.num_kv_heads * geometry.head_dim
    qkv = ops.linear(hidden, torch.cat((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight)))
    q_gate, key, value = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
    query, key, gate = _QKNormRopeGate.apply(
        q_gate, key, attn.q_norm.weight, attn.k_norm.weight, ctx.cos_sin_cache, ctx.rope_positions, geometry
    )
    query = query.view(tokens, geometry.num_heads, geometry.head_dim)
    key = key.view(tokens, geometry.num_kv_heads, geometry.head_dim)
    value = value.view(tokens, geometry.num_kv_heads, geometry.head_dim)
    if ctx.sp_group is not None:
        sp_size = torch.distributed.get_world_size(ctx.sp_group)
        if geometry.num_kv_heads < sp_size:
            # Each rank needs the key/value head its query heads read; the gradient sums the copies back.
            key = key.repeat_interleave(sp_size // geometry.num_kv_heads, dim=1)
            value = value.repeat_interleave(sp_size // geometry.num_kv_heads, dim=1)
        query, key, value = (_to_heads(ctx.sp_group, x) for x in (query, key, value))
    out = ops.paged_attention(query, key, value, ctx.layout, scale=geometry.head_dim**-0.5, fa_version=ctx.fa_version)
    out = _to_tokens(ctx.sp_group, out).reshape(tokens, q_size)
    out = out * torch.sigmoid(gate)
    return ops.row_parallel_linear(out, attn.o_proj.weight, geometry.tensor_parallel)


def _mlp(mlp: Any, hidden: torch.Tensor, tensor_parallel: int) -> torch.Tensor:
    gate_up = ops.linear(hidden, torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)))
    return ops.row_parallel_linear(ops.silu_and_mul(gate_up), mlp.down_proj.weight, tensor_parallel)


def _decoder_layer(layer: Any, hidden: torch.Tensor, residual: torch.Tensor | None, ctx: _ForwardContext):
    """One decoder layer on ``[tokens, hidden]``, carrying ``(hidden, residual)`` as vLLM does."""
    eps = ctx.geometry.eps
    if residual is None:
        residual = hidden
        hidden, _ = ops.gemma_rms_norm(hidden, layer.input_layernorm.weight, eps)
    else:
        hidden, residual = ops.gemma_rms_norm(hidden, layer.input_layernorm.weight, eps, residual)
    if layer.block_type == "linear_attention":
        mixer_out = _gated_delta_net(layer.linear_attn, hidden, ctx)
    else:
        mixer_out = _full_attention(layer.self_attn, hidden, ctx)
    hidden, residual = ops.gemma_rms_norm(mixer_out, layer.post_attention_layernorm.weight, eps, residual)
    return _mlp(layer.mlp, hidden, ctx.geometry.tensor_parallel), residual


def _decoder_layers(layers: list, hidden: torch.Tensor, residual: torch.Tensor, ctx: _ForwardContext):
    for layer in layers:
        hidden, residual = _decoder_layer(layer, hidden, residual, ctx)
    return hidden, residual


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


def aligned_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    labels: torch.Tensor,
    trained: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    temperature: float,
    plan: torch.Tensor | None = None,
) -> torch.Tensor:
    """The sampler's logprob of each ``labels`` token, ``[tokens]`` in fp32, differentiable; -100 scores 0.

    ``plan`` is a :mod:`~arctic_platform.model.implementations.vllm_aligned.turn_plan` plan, ``[..., tokens, K,
    3]``: each row is then replayed request by request, and ``trained`` is not used. With gradients enabled the
    decoder layers after the first are checkpointed in pairs; a recompute reruns the same kernels, so it
    reproduces the forward exactly.
    """
    backbone = model.model
    geometry: _Geometry = model._vllm_aligned_geometry
    sp_group = getattr(model, "_vllm_aligned_sp_group", None)
    sp_size = 1 if sp_group is None else torch.distributed.get_world_size(sp_group)
    sp_rank = 0 if sp_group is None else torch.distributed.get_rank(sp_group)
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    local_tokens = int(input_ids.numel())
    if sp_size > 1:
        input_ids, position_ids, labels, trained = (
            _gathered(sp_group, tensor.reshape(1, -1)) for tensor in (input_ids, position_ids, labels, trained)
        )
        plan = None if plan is None else _gathered(sp_group, plan.reshape(1, -1, *plan.shape[-2:]))
        cu_seqlens = None
    layout = ops.PackedLayout.build(position_ids, block_size=geometry.kv_block_size, cu_seqlens=cu_seqlens)
    token_ids, labels = input_ids.reshape(-1), labels.reshape(-1)
    output_index = None
    if plan is not None:
        turns = build_turn_layout(
            plan.reshape(-1, *plan.shape[-2:]),
            layout.cu_seqlens,
            block_size=geometry.kv_block_size,
            conv_width=geometry.conv_width,
            device=token_ids.device,
            pad_multiple=sp_size,
        )
        output_index = turns.output_index
        scored = torch.zeros(turns.expand_index.numel(), dtype=torch.bool, device=token_ids.device)
        scored[output_index[output_index >= 0]] = True
        token_ids = token_ids[turns.expand_index]
        labels = torch.where(scored, labels[turns.expand_index], -100)
        layout, gdn = turns.attention, turns.gdn
    else:
        gdn = GdnLayout.build(layout.cu_seqlens, trained, geometry.prefill_chunk, geometry.conv_width)
    if token_ids.numel() % sp_size:
        raise ValueError(f"{token_ids.numel()} tokens do not split across {sp_size} sequence-parallel ranks")
    shard = slice(sp_rank * (token_ids.numel() // sp_size), (sp_rank + 1) * (token_ids.numel() // sp_size))
    positions = layout.positions[shard]
    hidden = ops.tensor_parallel_embedding(backbone.embed_tokens(token_ids[shard]), geometry.tensor_parallel)
    ctx = _ForwardContext(
        geometry=geometry,
        layout=layout,
        gdn=gdn,
        rope_positions=positions.unsqueeze(0).expand(3, -1) if geometry.mrope_section else positions,
        cos_sin_cache=_cos_sin_cache(backbone, geometry, hidden.dtype, positions.device),
        fa_version=ops.flash_attn_version(geometry.head_dim),
        sp_group=sp_group if sp_size > 1 else None,
    )
    layers = list(backbone.layers)
    hidden, residual = _decoder_layer(layers[0], hidden, None, ctx)
    for start in range(1, len(layers), _LAYERS_PER_CHECKPOINT):
        group = layers[start : start + _LAYERS_PER_CHECKPOINT]
        if torch.is_grad_enabled():
            hidden, residual = torch.utils.checkpoint.checkpoint(
                _decoder_layers, group, hidden, residual, ctx, use_reentrant=False
            )
        else:
            hidden, residual = _decoder_layers(group, hidden, residual, ctx)
    hidden, _ = ops.gemma_rms_norm(hidden, backbone.norm.weight, geometry.eps, residual)
    logprobs = ops.lm_head_logprobs(
        hidden,
        model.lm_head.weight,
        labels[shard],
        temperature=temperature,
        token_chunk_size=model._vllm_aligned_token_chunk_size,
        fp32=model._vllm_aligned_lm_head_fp32,
    )
    if sp_size > 1:
        logprobs = gather_sequence(logprobs[None], sp_group)[0]
    if output_index is not None:
        logprobs = torch.where(output_index >= 0, logprobs[output_index.clamp(min=0)], 0.0)
    return logprobs[sp_rank * local_tokens : (sp_rank + 1) * local_tokens]


def _gathered(group, tensor: torch.Tensor) -> torch.Tensor:
    """The sequence-parallel ranks' ``[1, shard, ...]`` pieces of an input, concatenated in rank order."""
    dtype = tensor.dtype
    tensor = tensor.to(torch.uint8) if dtype == torch.bool else tensor
    parts = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=1).to(dtype)


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
    plan = kwargs.pop(PLAN_KWARG, None)
    if not dss_compute_logprobs:
        return self._vllm_aligned_stock_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
    if inputs_embeds is not None or action_masks is not None or group_token_ids is not None:
        raise ValueError("the vLLM-aligned Qwen3.5 logprobs take input_ids, without action_masks or group_token_ids")
    if not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
        raise ValueError("the vLLM-aligned Qwen3.5 logprobs score every position; logits_to_keep is not supported")
    if labels is None or input_ids is None:
        raise ValueError("dss_compute_logprobs requires input_ids and labels")
    if position_ids is None:
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).expand_as(input_ids)
    trained = loss_mask if loss_mask is not None else labels != -100
    logprobs = aligned_logprobs(
        self,
        input_ids,
        position_ids,
        labels.to(input_ids.device),
        trained.to(input_ids.device),
        cu_seqlens=kwargs.get("cu_seq_lens_q"),
        temperature=_scalar_temperature(temperature),
        plan=plan,
    )
    return {"logprobs": logprobs.view(input_ids.shape)}


def apply_vllm_aligned_qwen3_5(
    model: Any,
    *,
    token_chunk_size: int,
    kv_block_size: int | None = None,
    lm_head_fp32: bool = False,
    tensor_parallel: int = 1,
    prefill_chunk: int = 0,
    sp_group: Any = None,
) -> None:
    """Make a transformers ``Qwen3_5ForCausalLM``'s ``dss_compute_logprobs`` forward compute vLLM's logprobs.

    ``sp_group`` runs the forward sequence-parallel (Ulysses): each rank gets a contiguous shard of the packed
    tokens, runs the per-token operators on its share of them and the attention and GatedDeltaNet mixers on all
    tokens for a shard of the heads. vLLM's per-head kernels give the same bits on a head shard (the sampler's own
    tensor parallelism relies on it) and the batch-invariant matmul gives each row the same bits however many rows
    a call has, so the logprobs do not depend on the sequence-parallel size.

    ``kv_block_size`` defaults to :func:`hybrid_kv_block_size`. ``lm_head_fp32`` computes fp32
    logits as the sampler's batch-invariant fp32 LM head does, with an fp32 gradient. ``tensor_parallel`` is the
    sampler's tensor-parallel size: the row-parallel projections then add per-rank partials as its all-reduce does.
    The column-parallel projections and the per-head kernels give the same bits sharded or not, with the GDN
    autotuners pinned. ``prefill_chunk`` is the sampler's ``ARCTIC_PREFILL_CHUNK_ALIGN``: prompts are then
    prefilled that many tokens at a time from the sequence start, as the sampler's chunked prefill cuts them.
    """
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import _text_config

    config = _text_config(model)
    check_supported_config(config)
    if getattr(model, "model", None) is None or getattr(model, "lm_head", None) is None:
        raise ValueError("the vLLM-aligned forward needs a causal LM with .model and .lm_head")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    for layer in model.model.layers:
        if layer.block_type == "linear_attention" and layer.linear_attn.conv1d.bias is not None:
            raise ValueError("the vLLM-aligned Qwen3.5 forward expects a GatedDeltaNet conv without a bias")
    ops.require_batch_invariant_env()
    _pin_gdn_kernels()
    block_size = hybrid_kv_block_size(config, tensor_parallel) if kv_block_size is None else int(kv_block_size)
    _check_tensor_parallel(config, tensor_parallel)
    if prefill_chunk < 0:
        raise ValueError("prefill_chunk must be non-negative")
    model._vllm_aligned_geometry = _Geometry.from_config(
        config, kv_block_size=block_size, tensor_parallel=tensor_parallel, prefill_chunk=prefill_chunk
    )
    if sp_group is not None and torch.distributed.get_world_size(sp_group) > 1:
        _check_sequence_parallel(config, torch.distributed.get_world_size(sp_group))
        model._vllm_aligned_sp_group = sp_group
    model._vllm_aligned_token_chunk_size = int(token_chunk_size)
    model._vllm_aligned_lm_head_fp32 = bool(lm_head_fp32)
    model._vllm_aligned_stock_forward = model.forward
    model._vllm_aligned_takes_loss_mask = True
    model.forward = types.MethodType(_causal_lm_forward, model)
    model._vllm_aligned = True
