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
"""Checks for the vLLM-aligned Qwen3.5 (GatedDeltaNet hybrid) logprobs.

Bitwise agreement with a sampler can only be shown on a GPU against vLLM. These CPU tests replace each vLLM kernel
with its formula, so the aligned logprobs of packed sequences have to match stock transformers Qwen3.5 run one
sequence at a time. That checks the fused projection orders, the q/gate interleave, the partial and multimodal rotary
inputs, the residual carry, the conv input layout, the prefill/decode split with its state handoff, the gated norm and
the LM head. The gradient test checks that the per-operator backward gives the stock model's gradient.
"""

from __future__ import annotations

import itertools
import types

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from arctic_platform.model.implementations.vllm_aligned import ops
from arctic_platform.model.implementations.vllm_aligned import qwen3_5


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _causal_attention(query, key, value, scale):
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query, key) * scale
    queries, keys = query.shape[0], key.shape[0]
    mask = torch.ones(queries, keys, dtype=torch.bool).triu(keys - queries + 1)
    probs = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, value)


def _gating(A_log, a, dt_bias):
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


def _recurrence(query, key, value, g, beta, state):
    """The delta rule one token at a time; ``query``/``key`` already expanded to the value heads, state ``[H, V, K]``."""
    scale = query.shape[-1] ** -0.5
    outputs = []
    for t in range(query.shape[0]):
        state = state * g[t].exp()[:, None, None]
        memory = (state * key[t][:, None, :]).sum(-1)
        delta = (value[t] - memory) * beta[t][:, None]
        state = state + delta[:, :, None] * key[t][:, None, :]
        outputs.append((state * (query[t] * scale)[:, None, :]).sum(-1))
    return torch.stack(outputs) if outputs else value.new_zeros((0,) + value.shape[1:]), state


class _FakeBatchInvariant:
    @staticmethod
    def linear_batch_invariant(x, weight, bias=None):
        return F.linear(x, weight, bias)

    @staticmethod
    def mean_batch_invariant(x, dims, keepdim=False):
        return x.mean(dim=dims, keepdim=keepdim)


class _FakeFaUtils:
    @staticmethod
    def flash_attn_supports_quant_query_input():
        return True

    @staticmethod
    def get_flash_attn_version(head_size=None, **_):
        return 3

    @staticmethod
    def flash_attn_varlen_func(
        *,
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        softmax_scale,
        causal,
        block_table,
        **_,
    ):
        assert causal
        block_size = k.shape[1]
        for index in range(seqused_k.numel()):
            start, stop = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
            length = int(seqused_k[index])
            blocks = block_table[index, : -(-length // block_size)].long()
            keys = k[blocks].reshape(-1, k.shape[2], k.shape[3])[:length]
            values = v[blocks].reshape(-1, v.shape[2], v.shape[3])[:length]
            out[start:stop] = _causal_attention(q[start:stop], keys, values, softmax_scale)
        return out


def _fake_trainer_flash_attn(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=None, causal=False):
    assert causal
    q_bounds, k_bounds = cu_q.tolist(), cu_k.tolist()
    return torch.cat(
        [
            _causal_attention(q[a:b], k[c:d], v[c:d], softmax_scale)
            for a, b, c, d in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:])
        ]
    )


def _fake_silu_and_mul(out, x):
    half = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :half]) * x[..., half:])


def _fake_token_logprobs(logits, token_ids):
    return torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids[:, None]).squeeze(1)


def _fake_fused_qk_rmsnorm_rope_gate(
    q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps, num_q_heads, num_kv_heads, head_dim, rotary_dim,
    mrope_section=None, norm_beta=0.0,
):  # fmt: skip
    tokens = q_gate.shape[0]
    if positions.dim() == 2:
        assert mrope_section is not None and sum(mrope_section) == rotary_dim // 2
        assert bool((positions == positions[0]).all())
        positions = positions[0]
    q_gate = q_gate.reshape(tokens, num_q_heads, 2 * head_dim)
    query, gate = q_gate[..., :head_dim], q_gate[..., head_dim:]
    query = _rms(query, q_weight + norm_beta, eps)
    key = _rms(k.reshape(tokens, num_kv_heads, head_dim), k_weight + norm_beta, eps)
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    half = rotary_dim // 2

    def rotate(x):
        first, second = x[..., :half], x[..., half:rotary_dim]
        return torch.cat(
            (
                first * cos - second * sin,
                second * cos + first * sin,
                x[..., rotary_dim:],
            ),
            dim=-1,
        )

    return (
        rotate(query).reshape(tokens, -1),
        rotate(key).reshape(tokens, -1),
        gate.reshape(tokens, -1),
    )


def _fake_causal_conv1d_fn(
    x,
    weight,
    bias,
    *,
    activation,
    conv_states,
    has_initial_state,
    cache_indices,
    query_start_loc,
    **_,
):
    assert activation == "silu"
    assert x.stride(0) == 1, "the sampler passes the projection transposed, channel-last"
    width = weight.shape[1]
    out = torch.empty_like(x)
    bounds = query_start_loc.tolist()
    for seq, (start, stop) in enumerate(itertools.pairwise(bounds)):
        if bool(has_initial_state[seq]):
            padded = torch.cat((conv_states[int(cache_indices[seq])], x[:, start:stop]), dim=1)
        else:
            padded = F.pad(x[:, start:stop], (width - 1, 0))
        out[:, start:stop] = F.silu(F.conv1d(padded[None], weight[:, None, :], bias, groups=weight.shape[0])[0])
    return out


class _FakeFlaOps:
    decoded_tokens = 0

    @staticmethod
    def fused_post_conv_prep(
        *,
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm,
        output_g_exp,
    ):
        assert apply_l2norm and not output_g_exp and A_log.dtype == torch.float32
        tokens, key_dim = conv_output.shape[0], num_k_heads * head_k_dim
        query, key, value = conv_output.split([key_dim, key_dim, conv_output.shape[1] - 2 * key_dim], dim=-1)
        query = _l2norm(query.reshape(tokens, num_k_heads, head_k_dim))
        key = _l2norm(key.reshape(tokens, num_k_heads, head_k_dim))
        value = value.reshape(tokens, -1, head_v_dim)
        return query, key, value, _gating(A_log, a, dt_bias), b.float().sigmoid()

    @staticmethod
    def chunk_gated_delta_rule(
        *,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    ):
        assert output_final_state and not use_qk_l2norm_in_kernel
        groups = v.shape[2] // q.shape[2]
        query, key = (
            q[0].repeat_interleave(groups, 1),
            k[0].repeat_interleave(groups, 1),
        )
        out = torch.zeros_like(v[0])
        final = torch.empty_like(initial_state)
        bounds = cu_seqlens.tolist()
        for index, (start, stop) in enumerate(itertools.pairwise(bounds)):
            out[start:stop], final[index] = _recurrence(
                query[start:stop], key[start:stop], v[0, start:stop], g[0, start:stop], beta[0, start:stop],
                initial_state[index],
            )  # fmt: skip
        return out[None], final

    @staticmethod
    def fused_sigmoid_gating_delta_rule_update(
        *, A_log, a, b, dt_bias, q, k, v, initial_state, inplace_final_state, cu_seqlens, ssm_state_indices,
        use_qk_l2norm_in_kernel,
    ):  # fmt: skip
        assert inplace_final_state and use_qk_l2norm_in_kernel and initial_state.dtype == torch.float32
        _FakeFlaOps.decoded_tokens += int(q.shape[1])
        groups = v.shape[2] // q.shape[2]
        query = _l2norm(q[0].float()).repeat_interleave(groups, 1)
        key = _l2norm(k[0].float()).repeat_interleave(groups, 1)
        g, beta = _gating(A_log, a, dt_bias), b.float().sigmoid()
        out = torch.zeros_like(v[0])
        bounds = cu_seqlens.tolist()
        for index, (start, stop) in enumerate(itertools.pairwise(bounds)):
            slot, final_slot = int(ssm_state_indices[index, 0]), int(ssm_state_indices[index, stop - start - 1])
            out[start:stop], final = _recurrence(
                query[start:stop], key[start:stop], v[0, start:stop], g[start:stop], beta[start:stop],
                initial_state[slot],
            )  # fmt: skip
            if final_slot > 0:
                initial_state[final_slot] = final
        return out[None], initial_state


def _fake_reference_gated_delta_rule(
    q, k, v, *, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens
):
    """fla's chunked rule as a differentiable recurrence; ``q``/``k`` already carry the value heads.

    The state is fla's ``[heads, head_k_dim, head_v_dim]``.
    """
    assert use_qk_l2norm_in_kernel
    query, key = _l2norm(q[0].float()), _l2norm(k[0].float())
    bounds = cu_seqlens.tolist()
    outputs, finals = [], []
    for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        if initial_state is None:
            state = torch.zeros(v.shape[2], v.shape[3], q.shape[3], dtype=torch.float32)
        else:
            state = initial_state[index].transpose(-1, -2)
        out, final = _recurrence(
            query[start:stop], key[start:stop], v[0, start:stop].float(), g[0, start:stop], beta[0, start:stop].float(),
            state,
        )  # fmt: skip
        outputs.append(out)
        finals.append(final.transpose(-1, -2))
    return torch.cat(outputs)[None].to(v.dtype), torch.stack(finals) if output_final_state else None


def _fake_rmsnorm_fn(x, weight, bias, *, z, eps, group_size, norm_before_gate, activation):
    assert bias is None and group_size is None and norm_before_gate and activation == "silu"
    return (_rms(x.float(), weight.float(), eps) * F.silu(z.float())).to(x.dtype)


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setattr(ops, "_batch_invariant", lambda: _FakeBatchInvariant)
    monkeypatch.setattr(ops, "_fa_utils", lambda: _FakeFaUtils)
    monkeypatch.setattr(ops, "_silu_and_mul_kernel", _fake_silu_and_mul)
    monkeypatch.setattr(ops, "_trainer_flash_attn_varlen_func", lambda: _fake_trainer_flash_attn)
    monkeypatch.setattr(ops, "_token_logprobs_kernel", _fake_token_logprobs)
    monkeypatch.setattr(ops, "_fp32_logits_linear", lambda: lambda x, w: x.float() @ w.float().t())
    monkeypatch.setattr(qwen3_5, "_fused_qk_rmsnorm_rope_gate", lambda: _fake_fused_qk_rmsnorm_rope_gate)
    monkeypatch.setattr(qwen3_5, "_causal_conv1d_fn", lambda: _fake_causal_conv1d_fn)
    monkeypatch.setattr(qwen3_5, "_fla_ops", lambda: _FakeFlaOps)
    monkeypatch.setattr(qwen3_5, "_gated_rms_norm", lambda: _fake_rmsnorm_fn)
    monkeypatch.setattr(qwen3_5, "_pin_gdn_kernels", lambda: None)
    monkeypatch.setattr(qwen3_5, "_reference_gated_delta_rule", lambda: _fake_reference_gated_delta_rule)
    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "1")
    _FakeFlaOps.decoded_tokens = 0


def _config(**overrides):
    values = {
        "vocab_size": 97,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 256,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return Qwen3_5TextConfig(**values)


_LINEAR_LAYERS = 3


def _model(seed=0):
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(_config()).float().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") and "linear_attn" not in name:
                param.normal_(0.0, 0.2)
    model.config._attn_implementation = "eager"
    return model


def _stock_logprobs(model, ids):
    with torch.no_grad():
        logits = model(input_ids=ids[None]).logits[0].float()
    return torch.log_softmax(logits[:-1], dim=-1).gather(1, ids[1:, None]).squeeze(1)


def _enable_aligned(model, monkeypatch, **kwargs):
    qwen3_5.apply_vllm_aligned_qwen3_5(model, token_chunk_size=8, **kwargs)


def _packed(sequences, first_trained):
    """Packed input, labels, positions, cu_seqlens and loss mask; ``first_trained[i]`` is a position or None."""
    input_ids = torch.cat(sequences)
    labels, positions, mask = [], [], []
    for seq, first in zip(sequences, first_trained):
        labels.append(torch.cat((seq[1:], torch.tensor([-100]))))
        positions.append(torch.arange(seq.numel()))
        row = torch.zeros(seq.numel(), dtype=torch.bool)
        if first is not None:
            row[first:-1] = True
        mask.append(row)
    lengths = torch.tensor([0] + [seq.numel() for seq in sequences])
    cu_seqlens = torch.cumsum(lengths, 0).to(torch.int32)
    return (
        input_ids[None],
        torch.cat(labels)[None],
        torch.cat(positions)[None],
        cu_seqlens,
        torch.cat(mask)[None],
    )


def test_hybrid_kv_block_size_matches_vllm_for_qwen3_8_27b():
    config = types.SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
    )
    assert qwen3_5.hybrid_kv_block_size(config) == 784


def test_layout_prefills_through_the_first_trained_position():
    cu_seqlens = torch.tensor([0, 5, 9, 12], dtype=torch.int32)
    trained = torch.zeros(12, dtype=torch.bool)
    trained[2:4] = True  # sequence 0: prompt 0..2, decode 3..4
    trained[11] = True  # sequence 2: its last token is the first trained one, so nothing is decoded
    layout = qwen3_5.GdnLayout.build(cu_seqlens, trained)
    assert _steps(layout) == [
        (False, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11], [0, 3, 7, 10], [0, 1, 2]),
        (True, [3, 4], [0, 2], [0]),
    ]
    assert layout.steps[1].state_indices.tolist() == [[1, 1]]
    assert layout.cache_indices.tolist() == [1, 2, 3]
    assert not layout.resumes and len(layout.reference_steps) == 1


def _steps(layout):
    return [(s.decode, s.index.tolist(), s.cu_seqlens.tolist(), s.spans.tolist()) for s in layout.steps]


def test_layout_cuts_prompts_into_chunks_and_decodes_a_one_token_tail():
    cu_seqlens = torch.tensor([0, 10, 17], dtype=torch.int32)
    trained = torch.zeros(17, dtype=torch.bool)
    trained[6:9] = True  # sequence 0: prompt 0..6 (7 tokens), decode 7..9
    trained[15] = True  # sequence 1: prompt 10..15 (6 tokens), decode 16
    layout = qwen3_5.GdnLayout.build(cu_seqlens, trained, prefill_chunk=3)
    # Sequence 0: chunks [0,3) and [3,6); its tail [6,7) is one token, so vLLM decodes it.
    # Sequence 1: chunks [10,13) and [13,16).
    assert _steps(layout) == [
        (False, [0, 1, 2, 10, 11, 12], [0, 3, 6], [0, 1]),
        (False, [3, 4, 5, 13, 14, 15], [0, 3, 6], [0, 1]),
        (True, [6, 7, 8, 9, 16], [0, 4, 5], [0, 1]),
    ]


def test_a_one_token_prompt_is_decoded():
    layout = qwen3_5.GdnLayout.build(torch.tensor([0, 4], dtype=torch.int32), torch.tensor([1, 1, 1, 0]).bool())
    assert _steps(layout) == [(True, [0, 1, 2, 3], [0, 4], [0])]


def test_resumed_spans_take_state_and_conv_inputs_from_their_provider():
    spans = [
        qwen3_5.GdnSpan(0, 10, ((False, 0, 4), (False, 4, 8), (True, 8, 10))),
        qwen3_5.GdnSpan(10, 5, ((False, 0, 3), (True, 3, 5)), resume=(0, 8)),
        qwen3_5.GdnSpan(15, 3, ((False, 0, 3),), resume=(1, 3)),
    ]
    layout = qwen3_5.GdnLayout.from_spans(spans, num_tokens=18, conv_width=4, device=torch.device("cpu"))
    steps = [(s.decode, s.index.tolist(), s.spans.tolist()) for s in layout.steps]
    assert steps == [
        (False, list(range(0, 4)), [0]),
        (False, list(range(4, 8)), [0]),
        (True, [8, 9], [0]),
        (False, [10, 11, 12], [1]),
        (True, [13, 14], [1]),
        (False, [15, 16, 17], [2]),
    ]
    assert layout.steps[1].export_src.tolist() == [0] and layout.steps[1].export_dst.tolist() == [1]
    assert layout.steps[3].export_src.tolist() == [1] and layout.steps[3].export_dst.tolist() == [2]
    assert layout.conv_init_spans.tolist() == [1, 2]
    assert layout.conv_init_index.tolist() == [[5, 6, 7], [10, 11, 12]]
    assert layout.prev_index[10].tolist() == [7, 6, 5] and layout.prev_index[11].tolist() == [10, 7, 6]
    assert layout.prev_index[15].tolist() == [12, 11, 10] and layout.prev_index[2].tolist() == [1, 0, -1]
    reference = [(s.index.tolist(), s.spans.tolist()) for s in layout.reference_steps]
    assert reference == [(list(range(0, 8)), [0]), ([8, 9], [0]), (list(range(10, 13)), [1]), ([13, 14], [1]),
                         ([15, 16, 17], [2])]  # fmt: skip


def test_a_resume_must_land_on_a_segment_end():
    spans = [
        qwen3_5.GdnSpan(0, 10, ((False, 0, 10),)),
        qwen3_5.GdnSpan(10, 4, ((False, 0, 4),), resume=(0, 8)),
    ]
    with pytest.raises(ValueError, match="no segment"):
        qwen3_5.GdnLayout.from_spans(spans, num_tokens=14, conv_width=4, device=torch.device("cpu"))


def test_chunked_prefill_keeps_the_stock_function_and_gradient(fake_vllm, monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", modeling.torch_recurrent_gated_delta_rule)
    stock = _model()
    aligned = _model()
    _enable_aligned(aligned, monkeypatch, prefill_chunk=3)
    torch.manual_seed(8)
    sequences = [torch.randint(0, 97, (n,)) for n in (12, 9)]
    weights = [torch.linspace(0.5, 1.5, seq.numel()) for seq in sequences]
    input_ids, labels, positions, cu_seqlens, loss_mask = _packed(sequences, [7, 6])
    _stock_weighted_loss(stock, sequences, weights).backward()
    out = aligned(
        input_ids=input_ids,
        position_ids=positions,
        labels=labels,
        dss_compute_logprobs=True,
        cu_seq_lens_q=cu_seqlens,
        vllm_aligned_loss_mask=loss_mask,
    )["logprobs"][0]
    (out * torch.cat(weights)).sum().backward()
    for start, seq in zip(cu_seqlens.tolist()[:-1], sequences):
        torch.testing.assert_close(
            out[start : start + seq.numel() - 1].detach(), _stock_logprobs(stock, seq), rtol=1e-4, atol=1e-4
        )
    for name, param in stock.named_parameters():
        grad, other = param.grad, dict(aligned.named_parameters())[name].grad
        assert float((other - grad).abs().max() / grad.abs().max().clamp_min(1e-12)) < 1e-4, name


def test_aligned_logprobs_match_stock_per_sequence(fake_vllm, monkeypatch):
    model = _model()
    _enable_aligned(model, monkeypatch)
    torch.manual_seed(1)
    sequences = [torch.randint(0, 97, (n,)) for n in (11, 7, 9)]
    input_ids, labels, positions, cu_seqlens, loss_mask = _packed(sequences, [4, None, 1])
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=positions,
            labels=labels,
            dss_compute_logprobs=True,
            cu_seq_lens_q=cu_seqlens,
            vllm_aligned_loss_mask=loss_mask,
        )["logprobs"][0]
    assert _FakeFlaOps.decoded_tokens == _LINEAR_LAYERS * ((11 - 5) + (9 - 2))
    bounds = cu_seqlens.tolist()
    for seq, start in zip(sequences, bounds[:-1]):
        expected = _stock_logprobs(model, seq)
        torch.testing.assert_close(out[start : start + seq.numel() - 1], expected, rtol=1e-4, atol=1e-4)
        assert float(out[start + seq.numel() - 1]) == 0.0


def test_labels_decide_the_decoded_tokens_without_a_loss_mask(fake_vllm, monkeypatch):
    model = _model()
    _enable_aligned(model, monkeypatch)
    torch.manual_seed(2)
    seq = torch.randint(0, 97, (10,))
    labels = torch.cat((seq[1:], torch.tensor([-100])))
    labels[:3] = -100
    with torch.no_grad():
        out = model(
            input_ids=seq[None],
            position_ids=torch.arange(10)[None],
            labels=labels[None],
            dss_compute_logprobs=True,
        )["logprobs"][0]
    assert _FakeFlaOps.decoded_tokens == _LINEAR_LAYERS * (10 - 4)
    torch.testing.assert_close(out[3:9], _stock_logprobs(model, seq)[3:], rtol=1e-4, atol=1e-4)
    assert out[:3].abs().max().item() == 0.0


def _stock_weighted_loss(model, sequences, weights):
    total = 0.0
    for seq, w in zip(sequences, weights):
        logits = model(input_ids=seq[None]).logits[0].float()
        logprobs = torch.log_softmax(logits[:-1], dim=-1).gather(1, seq[1:, None]).squeeze(1)
        total = total + (logprobs * w[: seq.numel() - 1]).sum()
    return total


def test_gradient_is_the_stock_models_gradient_on_packed_sequences(fake_vllm, monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    # The stock chunked rule's fp32 algorithm moves the tiny A_log/dt_bias gradients by ~0.2%; its recurrent form
    # computes the same function the way the fake kernels do.
    monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", modeling.torch_recurrent_gated_delta_rule)
    stock = _model()
    aligned = _model()
    _enable_aligned(aligned, monkeypatch)
    torch.manual_seed(3)
    sequences = [torch.randint(0, 97, (n,)) for n in (11, 7, 9)]
    weights = [torch.linspace(0.5, 1.5, seq.numel()) for seq in sequences]
    input_ids, labels, positions, cu_seqlens, loss_mask = _packed(sequences, [4, None, 1])

    _stock_weighted_loss(stock, sequences, weights).backward()
    out = aligned(
        input_ids=input_ids,
        position_ids=positions,
        labels=labels,
        dss_compute_logprobs=True,
        cu_seq_lens_q=cu_seqlens,
        vllm_aligned_loss_mask=loss_mask,
    )["logprobs"][0]
    assert out.requires_grad
    (out * torch.cat(weights)).sum().backward()

    stock_grads = {name: p.grad for name, p in stock.named_parameters() if p.grad is not None}
    aligned_grads = {name: p.grad for name, p in aligned.named_parameters() if p.grad is not None}
    assert aligned_grads.keys() == stock_grads.keys() and len(stock_grads) > 20
    worst = {
        name: float((aligned_grads[name] - grad).abs().max() / grad.abs().max().clamp_min(1e-12))
        for name, grad in stock_grads.items()
    }
    assert max(worst.values()) < 1e-4, sorted(worst.items(), key=lambda kv: -kv[1])[:5]


def test_no_grad_forward_builds_no_graph(fake_vllm, monkeypatch):
    model = _model()
    _enable_aligned(model, monkeypatch)
    seq = torch.randint(0, 97, (6,))
    with torch.no_grad():
        out = model(
            input_ids=seq[None],
            position_ids=torch.arange(6)[None],
            labels=torch.cat((seq[1:], torch.tensor([-100])))[None],
            dss_compute_logprobs=True,
        )["logprobs"]
    assert not out.requires_grad


def test_lm_head_fp32_computes_fp32_logits_and_their_gradient(fake_vllm):
    torch.manual_seed(4)
    hidden = torch.randn(10, 16, dtype=torch.bfloat16)
    weight = torch.randn(32, 16, dtype=torch.bfloat16)
    labels = torch.randint(0, 32, (10,))
    h, w = hidden.clone().requires_grad_(True), weight.clone().requires_grad_(True)
    out = ops.lm_head_logprobs(h, w, labels, token_chunk_size=4, vocab_padding=8, fp32=True)
    out.sum().backward()
    h32, w32 = hidden.float().requires_grad_(True), weight.float().requires_grad_(True)
    expected = torch.log_softmax(h32 @ w32.t(), dim=-1).gather(1, labels[:, None]).squeeze(1)
    expected.sum().backward()
    torch.testing.assert_close(out, expected.detach())
    torch.testing.assert_close(h.grad.float(), h32.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(w.grad.float(), w32.grad, rtol=1e-2, atol=1e-2)


def test_without_dss_compute_logprobs_the_stock_forward_runs(fake_vllm, monkeypatch):
    model = _model()
    _enable_aligned(model, monkeypatch)
    seq = torch.randint(0, 97, (6,))
    with torch.no_grad():
        logits = model(input_ids=seq[None]).logits
    assert logits.shape == (1, 6, 97)
    assert _FakeFlaOps.decoded_tokens == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_act": "gelu"}, "silu"),
        (
            {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 2.0,
                    "rope_theta": 10000,
                }
            },
            "default rotary",
        ),
    ],
)
def test_check_supported_config_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        qwen3_5.check_supported_config(_config(**overrides))


def test_check_supported_config_rejects_other_model_types():
    with pytest.raises(ValueError, match="qwen3_5_text"):
        qwen3_5.check_supported_config(types.SimpleNamespace(model_type="qwen3"))


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_lm_head_backward_over_vocabulary_slices_matches_autograd(fake_vllm, monkeypatch, temperature):
    monkeypatch.setattr(ops, "_LM_HEAD_VOCAB_CHUNK", 7)
    torch.manual_seed(5)
    hidden = torch.randn(10, 16, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(30, 16, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 30, (10,))
    labels[3] = -100
    upstream = torch.linspace(-1.0, 2.0, 10, dtype=torch.float64)
    out = ops.lm_head_logprobs(hidden, weight, labels, temperature=temperature, token_chunk_size=4, vocab_padding=8)
    (out * upstream).sum().backward()

    h, w = hidden.detach().clone().requires_grad_(True), weight.detach().clone().requires_grad_(True)
    logprobs = torch.log_softmax((h @ w.t()) / temperature, dim=-1).gather(1, labels.clamp_min(0)[:, None]).squeeze(1)
    (logprobs.masked_fill(labels == -100, 0.0) * upstream).sum().backward()
    # The backward forms its logits in fp32 whatever the input dtype.
    torch.testing.assert_close(hidden.grad, h.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weight.grad, w.grad, rtol=1e-4, atol=1e-5)


def test_row_parallel_linear_adds_rounded_rank_partials_in_rank_order(fake_vllm):
    torch.manual_seed(6)
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    weight = torch.randn(7, 16, dtype=torch.bfloat16)
    partials = [F.linear(x[:, 4 * r : 4 * r + 4], weight[:, 4 * r : 4 * r + 4]).float() for r in range(4)]
    expected = (((partials[0] + partials[1]) + partials[2]) + partials[3]).to(torch.bfloat16)
    assert torch.equal(ops.row_parallel_linear_kernel(x, weight, 4), expected)

    xg, wg = x.float().requires_grad_(True), weight.float().requires_grad_(True)
    ops.row_parallel_linear(xg, wg, 4).sum().backward()
    xr, wr = x.float().requires_grad_(True), weight.float().requires_grad_(True)
    F.linear(xr, wr).sum().backward()
    torch.testing.assert_close(xg.grad, xr.grad)
    torch.testing.assert_close(wg.grad, wr.grad)


def test_tensor_parallel_emulation_keeps_the_stock_function_and_gradient(fake_vllm, monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", modeling.torch_recurrent_gated_delta_rule)
    stock = _model()
    aligned = _model()
    _enable_aligned(aligned, monkeypatch, tensor_parallel=2)
    assert aligned._vllm_aligned_geometry.tensor_parallel == 2
    torch.manual_seed(7)
    sequences = [torch.randint(0, 97, (n,)) for n in (9, 6)]
    weights = [torch.linspace(0.5, 1.5, seq.numel()) for seq in sequences]
    input_ids, labels, positions, cu_seqlens, loss_mask = _packed(sequences, [3, 1])
    _stock_weighted_loss(stock, sequences, weights).backward()
    out = aligned(
        input_ids=input_ids,
        position_ids=positions,
        labels=labels,
        dss_compute_logprobs=True,
        cu_seq_lens_q=cu_seqlens,
        vllm_aligned_loss_mask=loss_mask,
    )["logprobs"][0]
    (out * torch.cat(weights)).sum().backward()
    for start, seq in zip(cu_seqlens.tolist()[:-1], sequences):
        torch.testing.assert_close(
            out[start : start + seq.numel() - 1].detach(), _stock_logprobs(stock, seq), rtol=1e-4, atol=1e-4
        )
    for name, param in stock.named_parameters():
        grad, other = param.grad, dict(aligned.named_parameters())[name].grad
        assert float((other - grad).abs().max() / grad.abs().max().clamp_min(1e-12)) < 1e-4, name


def test_tensor_parallel_must_divide_the_heads():
    with pytest.raises(ValueError, match="does not split"):
        qwen3_5._check_tensor_parallel(_config(), 3)


def test_hybrid_kv_block_size_per_rank_for_qwen3_8_27b_tp4():
    config = types.SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
    )
    assert qwen3_5.hybrid_kv_block_size(config, tensor_parallel=4) == 784


_PAGE = 4

# A three-turn conversation of 30 tokens with 4-token pages. Turn 0 prefills [0, 9) in chunks ending at 4, 8 and 9
# (the last one token long, so decoded) and decodes to 14. Turn 1 resumes at 8 from turn 0's state and pages,
# prefills its prompt [8, 20) in chunks ending at 12, 16 and 20 and decodes to 25. Turn 2 resumes at 16 from turn 1's
# state, reads pages 0-2 from turn 0 (page 2 holds tokens turn 0 decoded) and page 3 from turn 1, and decodes to 30.
_TURNS = [
    dict(start=0, chunk_ends=[4, 8, 9], prompt_end=9, end=14, outputs=range(8, 13)),
    dict(start=8, chunk_ends=[12, 16, 20], prompt_end=20, end=25, outputs=range(19, 24)),
    dict(start=16, chunk_ends=[20, 24, 27], prompt_end=27, end=30, outputs=range(26, 29)),
]


def _turn_inputs(seed):
    from arctic_platform.model.implementations.vllm_aligned import turn_plan

    torch.manual_seed(seed)
    seq = torch.randint(0, 97, (30,))
    labels = torch.cat((seq[1:], torch.tensor([-100])))
    plan = turn_plan.encode_turns(30, _TURNS, block_size=_PAGE)
    return seq, labels, plan


def test_turn_plan_lays_out_each_request_with_the_pages_and_state_it_read():
    from arctic_platform.model.implementations.vllm_aligned import turn_plan

    _, _, plan = _turn_inputs(0)
    spans = turn_plan.parse_row(plan, row=0, block_size=_PAGE)
    assert [(s.start, s.end, s.page_columns, s.resume_column) for s in spans] == [
        (0, 14, (), None),
        (8, 25, (0, 0), 0),
        (16, 30, (0, 0, 0, 1), 1),
    ]
    assert spans[0].segments == ((False, 0, 4), (False, 4, 8), (False, 8, 9), (True, 9, 14))
    layout = turn_plan.build_turn_layout(
        plan, torch.tensor([0, 30]), block_size=_PAGE, conv_width=4, device=torch.device("cpu")
    )
    # Spans hold 14, 17 and 14 tokens: pages 0-3, 4-8 and 9-12.
    assert layout.attention.block_table.tolist() == [
        [0, 1, 2, 3, 0, 0, 0, 0],
        [0, 1, 4, 5, 6, 7, 8, 0],
        [0, 1, 2, 5, 9, 10, 11, 12],
    ]
    assert layout.attention.key_lens.tolist() == [14, 25, 30]
    # Turn 2's keys follow turn 0's 14 and turn 1's 25; its positions 12-13 are turn 1's tokens 4-5.
    assert layout.attention.key_index[39 + 12 : 39 + 14].tolist() == [14 + 4, 14 + 5]
    assert layout.attention.key_index[39 + 8 : 39 + 10].tolist() == [8, 9]
    assert layout.output_index[[8, 19, 26, 13, 14]].tolist() == [8, 14 + 11, 31 + 10, 13, 14 + 6]
    steps = [(s.decode, s.spans.tolist(), s.export_dst) for s in layout.gdn.steps]
    assert [d for d, _, _ in steps] == [False, False, True, False, False, False, True, False, False, False, True]


def test_turn_plan_rejects_a_page_its_writer_did_not_compute():
    from arctic_platform.model.implementations.vllm_aligned import turn_plan

    turns = [
        dict(start=0, chunk_ends=[4, 6], prompt_end=6, end=7),
        dict(start=8, prompt_end=10, end=12, pages=[0, 0], resume=0),
    ]
    plan = turn_plan.encode_turns(12, turns, block_size=_PAGE)
    with pytest.raises(ValueError, match="did not compute page 1"):
        turn_plan.parse_row(plan, row=0, block_size=_PAGE)


def test_multi_turn_replay_keeps_the_stock_function_and_gradient(fake_vllm, monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", modeling.torch_recurrent_gated_delta_rule)
    stock = _model()
    aligned = _model()
    _enable_aligned(aligned, monkeypatch, kv_block_size=_PAGE)
    seq, labels, plan = _turn_inputs(9)
    other = torch.randint(0, 97, (7,))
    weights = torch.linspace(0.5, 1.5, 30)
    _stock_weighted_loss(stock, [seq, other], [weights, torch.ones(7)]).backward()
    other_plan = torch.zeros((7, 3, 3), dtype=torch.int32)
    other_plan[:7, 0, 0] = 1
    out = aligned(
        input_ids=torch.cat((seq, other))[None],
        position_ids=torch.cat((torch.arange(30), torch.arange(7)))[None],
        labels=torch.cat((labels, other[1:], torch.tensor([-100])))[None],
        dss_compute_logprobs=True,
        cu_seq_lens_q=torch.tensor([0, 30, 37], dtype=torch.int32),
        vllm_aligned_plan=torch.cat((plan, other_plan))[None],
    )["logprobs"][0]
    assert _FakeFlaOps.decoded_tokens == _LINEAR_LAYERS * (6 + 5 + 3)
    torch.testing.assert_close(out[:29].detach(), _stock_logprobs(stock, seq), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[30:36].detach(), _stock_logprobs(stock, other), rtol=1e-4, atol=1e-4)
    (out * torch.cat((weights, torch.ones(7)))).sum().backward()
    for name, param in stock.named_parameters():
        grad, mine = param.grad, dict(aligned.named_parameters())[name].grad
        assert float((mine - grad).abs().max() / grad.abs().max().clamp_min(1e-12)) < 1e-4, name
