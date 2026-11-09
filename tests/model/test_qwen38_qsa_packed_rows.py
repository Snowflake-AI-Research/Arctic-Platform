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

import pytest
import torch

pytest.importorskip("transformers.models.qwen4_exp")


# Row-B positions restart at 0, restart at an offset, or continue row A (only varlen metadata can split it).
@pytest.mark.parametrize("layout", ["zero", "offset", "continuous_with_cu_seqlens"])
def test_qsa_packed_rows_match_isolated_rows(monkeypatch, layout):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    from arctic_platform.model.implementations.qwen38 import qsa_flex as qsa

    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        layer_types=["qwen_sparse_attention"],
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [1, 1, 0]},
    )
    qsa.register_qsa_flex_backend()
    config._attn_implementation = "qsa_flex"
    model = Qwen4ExpForCausalLM(config).to(torch.bfloat16).eval()
    qsa.apply_qsa_flex(model)
    # Exercise the real attention implementation on CPU; replace only its CUDA/BF16 entry guard.
    monkeypatch.setattr(
        qsa,
        "flex_sparse_gqa_attention",
        lambda q, k, v, ids, softmax_scale: qsa._SparseGQAAttention.apply(q, k, v, ids, softmax_scale, 128),
    )

    for lengths in ((2, 4), (3, 5), (1, 3)):
        a, b = lengths
        ids = torch.arange(a + b)[None] + 1
        if layout == "zero":
            positions = torch.cat((torch.arange(a), torch.arange(b)))[None]
        elif layout == "offset":
            positions = torch.cat((torch.arange(a) + 7, torch.arange(b) + 7))[None]
        else:
            positions = torch.arange(a + b)[None] + 7
        varlen = (
            {"cu_seq_lens_q": torch.tensor([0, a, a + b], dtype=torch.int32)} if layout.endswith("cu_seqlens") else {}
        )
        for mask in (None, torch.ones_like(ids)):

            def run(tokens, pos, amask, **kwargs):
                x = model.model.embed_tokens(tokens).detach().requires_grad_(True)
                y = model(inputs_embeds=x, position_ids=pos, attention_mask=amask, use_cache=False, **kwargs).logits
                return x, y

            x, packed = run(ids, positions, mask, **varlen)
            bx, isolated = run(ids[:, a:], positions[:, a:], None if mask is None else mask[:, a:])
            torch.testing.assert_close(packed[:, a:], isolated, rtol=1e-5, atol=1e-6)
            packed[:, a:].square().sum().backward()
            isolated.square().sum().backward()
            torch.testing.assert_close(x.grad[:, a:], bx.grad, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(x.grad[:, :a], torch.zeros_like(x.grad[:, :a]))
            changed = ids.clone()
            changed[:, :a] += 10
            _, perturbed = run(changed, positions, mask, **varlen)
            torch.testing.assert_close(perturbed[:, a:], isolated, rtol=1e-5, atol=1e-6)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        implicit = model(input_ids=ids, use_cache=False).logits
        broadcast = model(input_ids=ids, position_ids=torch.arange(3)[None], use_cache=False).logits
        explicit = model(input_ids=ids, position_ids=torch.arange(3)[None].expand(2, -1), use_cache=False).logits
        torch.testing.assert_close(implicit, explicit)
        torch.testing.assert_close(broadcast, explicit)


def test_qwen38_packed_boundaries_isolate_padded_tail():
    from arctic_platform.model.implementations.qwen38.qsa_flex import _normalize_packed_cu_seqlens

    normalized = _normalize_packed_cu_seqlens(
        torch.tensor([0, 3, 5], dtype=torch.int32),
        sequence_length=8,
        device=torch.device("cpu"),
    )
    assert normalized.tolist() == [0, 3, 5, 6, 7, 8]


@pytest.mark.parametrize(
    "boundaries",
    [
        [1, 3],
        [0, 3, 2],
        [0, 2, 2],
        [0, 9],
    ],
)
def test_qwen38_rejects_malformed_packed_boundaries(boundaries):
    from arctic_platform.model.implementations.qwen38.qsa_flex import _normalize_packed_cu_seqlens

    with pytest.raises(ValueError, match="packed boundaries"):
        _normalize_packed_cu_seqlens(
            torch.tensor(boundaries, dtype=torch.int32),
            sequence_length=8,
            device=torch.device("cpu"),
        )


def test_qwen38_cp_uses_explicit_global_packed_boundaries(monkeypatch):
    from arctic_platform.model.implementations.qwen38 import context_parallel

    class Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    backbone = Backbone()
    linear = torch.nn.Module()
    linear.conv_kernel_size = 4
    group = object()
    monkeypatch.setattr(context_parallel.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(
        context_parallel.dist,
        "all_gather",
        lambda *_args, **_kwargs: pytest.fail("explicit boundaries must not be reconstructed from positions"),
    )
    observed = {}

    def build_context(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(context_parallel, "build_gated_delta_net_cp_context", build_context)
    context_parallel._wrap_backbone_forward(backbone, [linear], group)
    boundaries = torch.tensor([0, 3, 6], dtype=torch.int32)
    backbone(
        input_ids=torch.ones(1, 3, dtype=torch.long),
        position_ids=torch.tensor([[7, 8, 9]]),
        cu_seq_lens_q=boundaries,
    )

    assert observed["global_cu_seqlens"] is boundaries
    assert getattr(linear, context_parallel._GLOBAL_CU_SEQLENS) is boundaries
