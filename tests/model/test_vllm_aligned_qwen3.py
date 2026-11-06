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
"""Checks for the vLLM-aligned Qwen3 training forward.

Bitwise agreement with a sampler can only be shown on a GPU against vLLM itself. What these tests can show without one
is that the aligned forward computes the stock model's function and that its hand-written backward is that function's
gradient. The CPU tests replace each vLLM kernel with its exact float64 formula, so the aligned forward has to match
stock transformers Qwen3 run one sequence at a time; that checks the fused projection order, the norm placement, the
rotary convention, the packed layout and the paged cache it reads, the LM head and every custom backward. The GPU test
runs the real kernels and checks the property the sampler relies on: a sequence's logprobs do not depend on what else
is packed into the call.
"""

from __future__ import annotations

import copy
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.model.implementations.vllm_aligned import ops


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _causal_attention(query, key, value, scale):
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query, key) * scale
    length = query.shape[0]
    mask = torch.ones(length, length, dtype=torch.bool, device=query.device).triu(1)
    probs = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, value)


class _FakeBatchInvariant:
    @staticmethod
    def linear_batch_invariant(x, weight, bias=None):
        return F.linear(x, weight, bias)

    @staticmethod
    def rms_norm_batch_invariant(x, weight, eps):
        return _rms(x, weight, eps)


class _FakeCustomOps:
    @staticmethod
    def fused_add_rms_norm(inp, residual, weight, eps):
        residual.add_(inp)
        inp.copy_(_rms(residual, weight, eps))

    @staticmethod
    def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox, rope_dim_offset=0, inverse=False):
        assert is_neox and rope_dim_offset == 0
        cos, sin = cos_sin_cache[positions].to(query.dtype).chunk(2, dim=-1)
        if inverse:
            sin = -sin
        half = head_size // 2
        for tensor in (query, key):
            heads = tensor.view(tensor.shape[0], -1, head_size)
            first, second = heads[..., :half].clone(), heads[..., half:].clone()
            heads[..., :half] = first * cos[:, None] - second * sin[:, None]
            heads[..., half:] = second * cos[:, None] + first * sin[:, None]


class _FakeFaUtils:
    calls: list[dict] = []

    @staticmethod
    def flash_attn_supports_quant_query_input():
        return True

    @staticmethod
    def get_flash_attn_version(head_size=None, **_):
        return 3

    @staticmethod
    def flash_attn_varlen_func(
        *, q, k, v, out, cu_seqlens_q, seqused_k, softmax_scale, causal, block_table, num_splits, **kwargs
    ):
        _FakeFaUtils.calls.append({"num_splits": num_splits, "scheduler_metadata": kwargs["scheduler_metadata"]})
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
    assert causal and torch.equal(cu_q, cu_k)
    bounds = cu_q.tolist()
    return torch.cat(
        [_causal_attention(q[a:b], k[a:b], v[a:b], softmax_scale) for a, b in zip(bounds[:-1], bounds[1:])]
    )


def _fake_token_logprobs(logits, token_ids):
    return torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids[:, None]).squeeze(1)


def _fake_silu_and_mul(out, x):
    half = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :half]) * x[..., half:])


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setattr(ops, "_batch_invariant", lambda: _FakeBatchInvariant)
    monkeypatch.setattr(ops, "_custom_ops", lambda: _FakeCustomOps)
    monkeypatch.setattr(ops, "_fa_utils", lambda: _FakeFaUtils)
    monkeypatch.setattr(ops, "_silu_and_mul_kernel", _fake_silu_and_mul)
    monkeypatch.setattr(ops, "_token_logprobs_kernel", _fake_token_logprobs)
    monkeypatch.setattr(ops, "_trainer_flash_attn_varlen_func", lambda: _fake_trainer_flash_attn)
    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "1")
    _FakeFaUtils.calls.clear()


def _tiny_qwen3(*, vocab_size=100, dtype=torch.float64, device="cpu", tie_word_embeddings=False):
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(dtype=dtype, device=device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.add_(0.1 * torch.randn_like(parameter))
    return model


def _packed(sequences, *, device="cpu"):
    """Packed ``[1, T]`` input ids, positions and next-token labels; the last token of each sequence is ignored."""
    input_ids = torch.cat(sequences).unsqueeze(0).to(device)
    position_ids = torch.cat([torch.arange(len(sequence)) for sequence in sequences]).unsqueeze(0).to(device)
    labels = (
        torch.cat([torch.cat((sequence[1:], torch.tensor([-100]))) for sequence in sequences]).unsqueeze(0).to(device)
    )
    return input_ids, position_ids, labels


def _stock_logprobs(model, sequences):
    """Per-token logprobs of each sequence run on its own through the stock forward, concatenated."""
    pieces = []
    for sequence in sequences:
        logits = model(input_ids=sequence.unsqueeze(0)).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        targets = torch.cat((sequence[1:], sequence.new_tensor([0])))
        picked = logprobs.gather(1, targets[:, None]).squeeze(1)
        pieces.append(torch.cat((picked[:-1], picked.new_zeros(1))))
    return torch.cat(pieces)


def _sequences(vocab_size):
    torch.manual_seed(1)
    return [torch.randint(0, vocab_size, (length,)) for length in (5, 17, 1, 9)]


def _relative_error(actual, expected):
    return float((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-12))


@pytest.mark.parametrize("vocab_size, tie", [(100, False), (128, True)])
def test_aligned_forward_computes_the_stock_model_on_packed_sequences(fake_vllm, vocab_size, tie):
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import apply_vllm_aligned

    stock = _tiny_qwen3(vocab_size=vocab_size, tie_word_embeddings=tie)
    aligned = copy.deepcopy(stock)
    apply_vllm_aligned(aligned, kv_block_size=4, token_chunk_size=7)
    sequences = _sequences(vocab_size)
    input_ids, position_ids, labels = _packed(sequences)

    logprobs = aligned(input_ids=input_ids, position_ids=position_ids, labels=labels, dss_compute_logprobs=True)
    expected = _stock_logprobs(stock, sequences)

    assert logprobs["logprobs"].shape == (1, input_ids.shape[1])
    assert _relative_error(logprobs["logprobs"][0].double(), expected.double()) < 1e-5
    assert _FakeFaUtils.calls and all(call["num_splits"] == 1 for call in _FakeFaUtils.calls)
    assert all(call["scheduler_metadata"] is None for call in _FakeFaUtils.calls)


def test_aligned_backward_is_the_gradient_of_the_stock_model(fake_vllm):
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import apply_vllm_aligned

    stock = _tiny_qwen3()
    aligned = copy.deepcopy(stock)
    apply_vllm_aligned(aligned, kv_block_size=4, token_chunk_size=5)
    sequences = _sequences(stock.config.vocab_size)
    input_ids, position_ids, labels = _packed(sequences)
    torch.manual_seed(2)
    token_weights = torch.randn(input_ids.shape[1], dtype=torch.float64)

    out = aligned(input_ids=input_ids, position_ids=position_ids, labels=labels, dss_compute_logprobs=True)
    (out["logprobs"][0].double() * token_weights).sum().backward()
    (_stock_logprobs(stock, sequences).double() * token_weights).sum().backward()

    stock_grads = dict(stock.named_parameters())
    for name, parameter in aligned.named_parameters():
        expected = stock_grads[name].grad
        assert parameter.grad is not None, name
        assert _relative_error(parameter.grad, expected) < 1e-4, name


def test_a_sequence_scores_the_same_alone_and_packed(fake_vllm):
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import apply_vllm_aligned

    model = _tiny_qwen3()
    apply_vllm_aligned(model, kv_block_size=4)
    sequences = _sequences(model.config.vocab_size)
    alone_ids, alone_positions, alone_labels = _packed(sequences[1:2])
    packed_ids, packed_positions, packed_labels = _packed(sequences)
    start = len(sequences[0])

    alone = model(input_ids=alone_ids, position_ids=alone_positions, labels=alone_labels, dss_compute_logprobs=True)
    packed = model(
        input_ids=packed_ids, position_ids=packed_positions, labels=packed_labels, dss_compute_logprobs=True
    )

    torch.testing.assert_close(
        packed["logprobs"][0, start : start + len(sequences[1])], alone["logprobs"][0], rtol=1e-12, atol=1e-12
    )


def test_packed_layout_gives_each_sequence_its_own_pages():
    position_ids = torch.cat([torch.arange(n) for n in (5, 17, 1)]).unsqueeze(0)

    layout = ops.PackedLayout.build(position_ids, block_size=4)

    assert layout.cu_seqlens.tolist() == [0, 5, 22, 23]
    assert layout.seq_lens.tolist() == [5, 17, 1]
    assert layout.max_seqlen == 17
    assert layout.num_blocks == 2 + 5 + 1
    assert layout.block_table.tolist() == [[0, 1, 0, 0, 0], [2, 3, 4, 5, 6], [7, 0, 0, 0, 0]]
    expected_slots = list(range(0, 5)) + list(range(8, 25)) + [28]
    assert layout.slots.tolist() == expected_slots


def test_packed_layout_starts_a_sequence_at_each_row_of_a_batch():
    position_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])

    layout = ops.PackedLayout.build(position_ids, block_size=16)

    assert layout.cu_seqlens.tolist() == [0, 3, 6]


def test_the_trainer_environment_must_not_disable_batch_invariance(monkeypatch):
    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "0")
    with pytest.raises(RuntimeError, match="needs it set to 1"):
        ops.require_batch_invariant_env()

    monkeypatch.delenv(ops.BATCH_INVARIANT_ENV)
    ops.require_batch_invariant_env()
    assert ops.os.environ[ops.BATCH_INVARIANT_ENV] == "1"


def _saved_tiny_qwen3(path, *, vocab_size=128):
    model = _tiny_qwen3(vocab_size=vocab_size, dtype=torch.bfloat16)
    model.save_pretrained(path)
    return path


def _spec(path, **patches):
    from arctic_platform.model.config import ModelSpec
    from arctic_platform.model.config import Patches

    lm_head = patches.pop("lm_head", {"token_chunk_size": 8, "vocab_chunk_size": 64})
    return ModelSpec(
        model_path_or_name=str(path),
        loader="huggingface",
        dtype=patches.pop("dtype", "bfloat16"),
        attn_implementation="eager",
        loader_options={"vllm_aligned": True},
        patches=Patches(lm_head=lm_head, **patches),
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"liger": True}, "remove liger"),
        ({"compile": {}}, "remove compile"),
        ({"lm_head": None}, "needs patches.lm_head.token_chunk_size"),
        ({"dtype": "float32"}, "requires dtype bfloat16"),
    ],
)
def test_the_loader_refuses_what_the_aligned_forward_cannot_honour(tmp_path, monkeypatch, overrides, message):
    from arctic_platform.model.loader import LoaderContext
    from arctic_platform.model.loaders.huggingface import load_huggingface

    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "1")
    path = _saved_tiny_qwen3(tmp_path)
    with pytest.raises(ValueError, match=message):
        load_huggingface(LoaderContext(spec=_spec(path, **overrides)))


def test_the_loader_refuses_a_model_type_it_does_not_reproduce(tmp_path, monkeypatch):
    from transformers import LlamaConfig

    from arctic_platform.model.loader import LoaderContext
    from arctic_platform.model.loaders.huggingface import load_huggingface

    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "1")
    LlamaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=1, intermediate_size=48).save_pretrained(
        tmp_path
    )
    with pytest.raises(ValueError, match="supports model types"):
        load_huggingface(LoaderContext(spec=_spec(tmp_path)))


def test_the_loader_installs_the_aligned_forward_and_owns_the_lm_head_patch(tmp_path, monkeypatch):
    from arctic_platform.model.loader import LoaderContext
    from arctic_platform.model.loaders.huggingface import load_huggingface

    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "1")
    path = _saved_tiny_qwen3(tmp_path)
    loaded = load_huggingface(LoaderContext(spec=_spec(path)))

    assert loaded.applied_patches == frozenset({"lm_head"})
    assert loaded.model._vllm_aligned is True
    assert loaded.model._vllm_aligned_token_chunk_size == 8
    assert loaded.model._vllm_aligned_lm_head_fp32_backward is False
    stock_keys = set(_tiny_qwen3(vocab_size=128).state_dict())
    assert set(loaded.model.state_dict()) == stock_keys


def test_lm_head_fp32_makes_only_the_backward_fp32(tmp_path, monkeypatch):
    from arctic_platform.model.loader import LoaderContext
    from arctic_platform.model.loaders.huggingface import load_huggingface

    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "1")
    path = _saved_tiny_qwen3(tmp_path)
    lm_head = {"fp32": True, "token_chunk_size": 8, "vocab_chunk_size": 64}
    loaded = load_huggingface(LoaderContext(spec=_spec(path, lm_head=lm_head)))
    assert loaded.model._vllm_aligned_lm_head_fp32_backward is True


_HAS_CUDA_AND_VLLM = torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None


@pytest.mark.integration
@pytest.mark.gpu_serial
@pytest.mark.skipif(not _HAS_CUDA_AND_VLLM, reason="needs CUDA and vLLM")
def test_real_kernels_score_a_sequence_bitwise_the_same_alone_and_packed(monkeypatch):
    """With vLLM's kernels, a sequence's logprobs and the stock model's agree, and packing changes no bit."""
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM

    from arctic_platform.model.implementations.vllm_aligned.qwen3 import apply_vllm_aligned

    monkeypatch.setenv(ops.BATCH_INVARIANT_ENV, "1")
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
    )
    config._attn_implementation = "eager"
    stock = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.float32)
    aligned = copy.deepcopy(stock).to(torch.bfloat16)
    apply_vllm_aligned(aligned)
    sequences = [torch.randint(0, 512, (length,)) for length in (33, 200, 7)]
    alone = _packed(sequences[1:2], device="cuda")
    packed = _packed(sequences, device="cuda")

    def score(inputs):
        input_ids, position_ids, labels = inputs
        return aligned(input_ids=input_ids, position_ids=position_ids, labels=labels, dss_compute_logprobs=True)[
            "logprobs"
        ][0]

    alone_logprobs = score(alone)
    packed_logprobs = score(packed)
    start = len(sequences[0])
    assert torch.equal(packed_logprobs[start : start + len(sequences[1])], alone_logprobs)

    with torch.no_grad():
        reference = _stock_logprobs(stock, [sequence.cuda() for sequence in sequences[1:2]])
    assert float((alone_logprobs - reference).abs().max()) < 5e-2

    alone_logprobs.sum().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in aligned.parameters()
    )
