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
the LM head. The gradient tests check that the returned values carry the stock forward's gradient.
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
    length = query.shape[0]
    mask = torch.ones(length, length, dtype=torch.bool).triu(1)
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
    assert activation == "silu" and not bool(has_initial_state.any())
    assert x.stride(0) == 1, "the sampler passes the projection transposed, channel-last"
    width = weight.shape[1]
    out = torch.empty_like(x)
    bounds = query_start_loc.tolist()
    for start, stop in itertools.pairwise(bounds):
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
            slot = int(ssm_state_indices[index, 0])
            out[start:stop], initial_state[slot] = _recurrence(
                query[start:stop], key[start:stop], v[0, start:stop], g[start:stop], beta[start:stop],
                initial_state[slot],
            )  # fmt: skip
        return out[None], initial_state


def _fake_rmsnorm_fn(x, weight, bias, *, z, eps, group_size, norm_before_gate, activation):
    assert bias is None and group_size is None and norm_before_gate and activation == "silu"
    return (_rms(x.float(), weight.float(), eps) * F.silu(z.float())).to(x.dtype)


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setattr(ops, "_batch_invariant", lambda: _FakeBatchInvariant)
    monkeypatch.setattr(ops, "_fa_utils", lambda: _FakeFaUtils)
    monkeypatch.setattr(ops, "_silu_and_mul_kernel", _fake_silu_and_mul)
    monkeypatch.setattr(ops, "_token_logprobs_kernel", _fake_token_logprobs)
    monkeypatch.setattr(qwen3_5, "_fused_qk_rmsnorm_rope_gate", lambda: _fake_fused_qk_rmsnorm_rope_gate)
    monkeypatch.setattr(qwen3_5, "_causal_conv1d_fn", lambda: _fake_causal_conv1d_fn)
    monkeypatch.setattr(qwen3_5, "_fla_ops", lambda: _FakeFlaOps)
    monkeypatch.setattr(qwen3_5, "_gated_rms_norm", lambda: _fake_rmsnorm_fn)
    monkeypatch.setattr(qwen3_5, "_pin_gdn_kernels", lambda: None)
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


def _enable_aligned(model, monkeypatch):
    from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs

    enable_chunked_lm_head_logprobs(model, token_chunk_size=8, vocab_chunk_size=32, fp32_lm_head=False)
    qwen3_5.apply_vllm_aligned_qwen3_5(model, token_chunk_size=8)


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
    assert layout.prefill_index.tolist() == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    assert layout.prefill_cu.tolist() == [0, 3, 7, 10]
    assert layout.decode_index.tolist() == [3, 4]
    assert layout.decode_cu.tolist() == [0, 2, 2, 2]
    assert layout.decode_state_indices[:, 0].tolist() == [1, 2, 3]
    assert layout.cache_indices.tolist() == [1, 2, 3]


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


def test_values_are_aligned_and_gradient_is_stock(fake_vllm, monkeypatch):
    model = _model()
    _enable_aligned(model, monkeypatch)
    torch.manual_seed(3)
    seq = torch.randint(0, 97, (12,))
    input_ids, labels, positions, _cu_seqlens, loss_mask = _packed([seq], [5])
    kwargs = {
        "input_ids": input_ids,
        "position_ids": positions,
        "labels": labels,
        "dss_compute_logprobs": True,
    }

    sentinel = torch.full_like(labels, 7.0, dtype=torch.float32)
    monkeypatch.setattr(qwen3_5, "aligned_logprobs", lambda *args, **kw: sentinel.reshape(-1))
    weights = torch.linspace(0.5, 1.5, labels.numel()).view_as(labels)
    out = model(**kwargs, vllm_aligned_loss_mask=loss_mask)["logprobs"]
    assert torch.equal(out, sentinel)
    (out * weights).sum().backward()
    aligned_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    stock = model._vllm_aligned_stock_forward(**kwargs)["logprobs"]
    (stock * weights).sum().backward()
    stock_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert aligned_grads.keys() == stock_grads.keys() and len(stock_grads) > 0
    for name, grad in stock_grads.items():
        assert torch.equal(aligned_grads[name], grad), name


def test_no_grad_skips_the_stock_forward(fake_vllm, monkeypatch):
    model = _model()
    _enable_aligned(model, monkeypatch)

    def fail(**_):
        raise AssertionError("the stock forward ran without gradients")

    model._vllm_aligned_stock_forward = fail
    seq = torch.randint(0, 97, (6,))
    with torch.no_grad():
        model(
            input_ids=seq[None],
            position_ids=torch.arange(6)[None],
            labels=torch.cat((seq[1:], torch.tensor([-100])))[None],
            dss_compute_logprobs=True,
        )


def test_without_dss_compute_logprobs_the_stock_forward_runs(fake_vllm, monkeypatch):
    model = _model()
    _enable_aligned(model, monkeypatch)
    seq = torch.randint(0, 97, (6,))
    with torch.no_grad():
        logits = model(input_ids=seq[None]).logits
    assert logits.shape == (1, 6, 97)
    assert _FakeFlaOps.decoded_tokens == 0


def test_apply_requires_the_chunked_lm_head(fake_vllm):
    with pytest.raises(ValueError, match="chunked LM-head patch"):
        qwen3_5.apply_vllm_aligned_qwen3_5(_model(), token_chunk_size=8)


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
