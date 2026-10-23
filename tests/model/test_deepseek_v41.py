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
import json

import torch
from torch import nn


def _tiny_config(**overrides):
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config

    config = dict(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        max_position_embeddings=16,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=4,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        hc_mult=2,
        vision_n_layers=0,
        image_token_id=1,
        dspark_noise_token_id=1,
        pad_token_id=0,
        swiglu_limit=10.0,
    )
    config.update(overrides)
    return DeepseekV41Config(**config)


def test_defaults_match_v41_flash():
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config

    config = DeepseekV41Config()
    assert config.model_type == "deepseek_v41"
    assert config.num_hidden_layers == 40
    assert config.hidden_size == 5120
    assert config.moe_intermediate_size == 2304
    assert config.n_routed_experts == 384
    assert config.num_experts_per_tok == 6
    assert config.scoring_func == "sqrtsoftplus"
    assert config.rms_norm_eps == 1e-20
    assert config.qk_rope_head_dim == 64
    assert config.compress_ratios == [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    assert config.kv_source_layer_ids == [2, 8, 14, 20]
    assert config.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36]
    assert config.candidate_source_layer_id == 20
    assert config.hc_mult == 4
    assert config.vision_n_layers == 32
    assert config.engram_quantized is True
    assert config.rope_parameters["main"]["rope_type"] == "default"
    assert config.rope_parameters["compress"]["rope_type"] == "yarn"
    assert config.rope_parameters["compress"]["original_max_position_embeddings"] == 65536
    assert not hasattr(config, "num_hash_layers")


def test_nested_text_config_flattens_and_keeps_outer_type():
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config

    config = DeepseekV41Config(
        text_config={
            "model_type": "deepseek_v41_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 0,
            "vocab_size": 48,
            "compress_ratios": [0, 2],
            "kv_source_layer_ids": [1],
            "index_source_layer_ids": [],
            "candidate_source_layer_id": -1,
            "engram_layer_ids": [],
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 64,
            },
        },
        vision_config={"model_type": "deepseek_v41_vision", "num_hidden_layers": 4, "hidden_size": 128},
        image_token_id=3,
    )
    assert config.model_type == "deepseek_v41"
    assert config.hidden_size == 32
    assert config.num_hidden_layers == 2
    assert config.compress_ratios == [0, 2]
    assert config.kv_source_layer_ids == [1]
    assert config.vision_n_layers == 4
    assert config.vision_dim == 128
    assert config.image_token_id == 3
    assert config.rope_parameters["compress"]["rope_type"] == "yarn"
    assert config.rope_parameters["main"]["rope_type"] == "default"
    explicit = DeepseekV41Config(text_config={"hidden_size": 32, "num_hidden_layers": 2}, hidden_size=16)
    assert explicit.hidden_size == 16


def test_autoconfig_returns_the_ported_class(tmp_path):
    from transformers import AutoConfig

    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config
    from arctic_platform.model.implementations.qwen35.models import supports_custom_impl

    path = tmp_path / "deepseek_v41"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "image_token_id": 3,
                "text_config": {
                    "model_type": "deepseek_v41_text",
                    "vocab_size": 48,
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "num_nextn_predict_layers": 0,
                    "compress_ratios": [0, 0],
                    "kv_source_layer_ids": [],
                    "index_source_layer_ids": [],
                    "candidate_source_layer_id": -1,
                    "engram_layer_ids": [],
                    "qk_rope_head_dim": 8,
                    "head_dim": 16,
                },
                "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 0},
            }
        )
    )
    config = AutoConfig.from_pretrained(path)
    assert isinstance(config, DeepseekV41Config)
    assert config.model_type == "deepseek_v41"
    assert config.hidden_size == 32
    assert config.num_hidden_layers == 2
    assert config.vision_n_layers == 0
    assert supports_custom_impl(config)


def test_shifted_mhc_feeds_each_pre_to_the_next_sublayer():
    from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
    from arctic_platform.model.implementations.deepseek_v41.hyperconnections import identity_pre
    from arctic_platform.model.implementations.deepseek_v41.modeling_deepseek_v41 import DeepseekV41DecoderLayer

    config = _tiny_config()
    layer = DeepseekV41DecoderLayer(config, 0, DeepseekV4RotaryEmbedding(config))
    hc = config.hc_mult
    for connection, pre_index in ((layer.attn_hc, 1), (layer.ffn_hc, 0)):
        connection.fn.data.zero_()
        connection.scale.data.fill_(1.0)
        base = torch.full((connection.base.numel(),), -30.0)
        base[pre_index] = 30.0
        comb = base[2 * hc :].view(hc, hc)
        comb.fill_(-30.0)
        comb.fill_diagonal_(30.0)
        connection.base.data.copy_(base)

    class _Record(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, hidden, packed=None, shared=None, routed_experts=None):
            del packed, routed_experts
            self.seen = hidden.detach().clone()
            zeros = torch.zeros_like(hidden)
            if shared is None:
                return zeros
            return zeros, shared

    layer.self_attn = _Record()
    layer.mlp = _Record()
    layer.input_layernorm = nn.Identity()
    layer.post_attention_layernorm = nn.Identity()

    states = torch.zeros(1, 1, hc, config.hidden_size)
    states[..., 0, 0] = 1.0
    states[..., 1, 1] = 1.0
    pre = identity_pre(states)
    layer(states, pre, packed=None)

    assert torch.allclose(layer.self_attn.seen, states[:, :, 0, :])
    assert torch.allclose(layer.mlp.seen, states[:, :, 1, :], atol=1e-4)


def test_final_pre_replaces_hyper_head():
    from arctic_platform.model.implementations.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Model

    config = _tiny_config()
    model = DeepseekV41Model(config)
    assert not hasattr(model, "hc_head")

    class _Stub(nn.Module):
        def forward(self, states, pre, shared=None, routed_experts=None, *, packed):
            del states, routed_experts, packed
            out = torch.zeros(1, 2, config.hc_mult, config.hidden_size)
            out[..., 0, 0] = 1.0
            out[..., 1, 1] = 1.0
            nxt = torch.zeros_like(pre)
            nxt[..., 1] = 1.0
            self.incoming = pre.detach().clone()
            return out, nxt, shared

    stub = _Stub()
    model.layers = nn.ModuleList([stub])
    hidden = model(
        inputs_embeds=torch.zeros(1, 2, config.hidden_size),
        seq_lens=torch.tensor([2]),
        position_ids=torch.arange(2).view(1, -1),
    ).last_hidden_state
    assert torch.equal(stub.incoming[..., 0], torch.ones(1, 2))
    stream = torch.zeros(1, 2, config.hidden_size)
    stream[..., 1] = 1.0
    assert torch.allclose(hidden, model.norm(stream))


def test_moe_has_no_hash_router():
    from arctic_platform.model.implementations.deepseek_v41.moe import DeepseekV41MoE

    config = _tiny_config(vision_n_layers=4)
    moe = DeepseekV41MoE(config, 0)
    assert moe.is_hash is False
    assert not hasattr(moe.router, "tid2eid")
    assert moe.expert_bias.shape == (4,)
    assert moe.bias_vl.shape == (4,)
    assert moe.router.score_func == "sqrtsoftplus"
    assert moe.experts.swiglu_limit == 10.0
    assert moe.score_before_experts is False


def test_sliding_window_respects_packed_documents():
    from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
    from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Attention
    from arctic_platform.model.implementations.deepseek_v41.attention import build_packed

    torch.manual_seed(0)
    config = _tiny_config(num_hidden_layers=1)
    attn = DeepseekV41Attention(config, 0, DeepseekV4RotaryEmbedding(config))
    hidden = torch.randn(1, 4, config.hidden_size)
    one_doc = build_packed(config, torch.tensor([4]), hidden.device)
    two_docs = build_packed(config, torch.tensor([2, 2]), hidden.device)
    single, _ = attn(hidden, one_doc)
    split, _ = attn(hidden, two_docs)
    assert torch.allclose(single[:, 0], split[:, 0])
    assert not torch.allclose(single[:, 2], split[:, 2], atol=1e-5)


def test_tiny_forward_backward():
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM

    torch.manual_seed(0)
    config = _tiny_config(num_hidden_layers=1)
    model = DeepseekV41ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    output = model(
        input_ids=input_ids,
        seq_lens=torch.tensor([4]),
        position_ids=torch.arange(4).view(1, -1),
    )
    logits = output["logits"]
    assert logits.shape == (1, 4, config.vocab_size)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), input_ids.view(-1))
    loss.backward()
    graded = [param.grad for param in model.parameters() if param.grad is not None]
    assert graded
    assert all(torch.isfinite(grad).all() for grad in graded)
    assert any(grad.abs().sum().item() > 0 for grad in graded)


def _sparse_config(**overrides):
    config = dict(
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
        compress_ratios=[2],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=-1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        candidate_block_size=2,
        candidate_topk_blocks=1,
    )
    config.update(overrides)
    return _tiny_config(**config)


def test_compressor_ratio_math_and_group_start_rope():
    from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
    from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Attention
    from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Compressor
    from arctic_platform.model.implementations.deepseek_v41.attention import apply_interleaved_rope
    from arctic_platform.model.implementations.deepseek_v41.attention import build_packed

    config = _sparse_config()
    one_config = _sparse_config(compress_ratios=[1])
    ratio_one = DeepseekV41Compressor(one_config, 1)
    assert not hasattr(ratio_one, "wgate")
    hidden = torch.randn(1, 4, config.hidden_size)
    one_packed = build_packed(one_config, torch.tensor([4]), hidden.device)
    assert torch.allclose(ratio_one(hidden, one_packed, lambda tensor: tensor), ratio_one.norm(ratio_one.wkv(hidden)))
    packed = build_packed(config, torch.tensor([4]), hidden.device)

    compressor = DeepseekV41Compressor(config, 2)
    assert compressor.wkv.weight.dtype == torch.float32
    assert compressor.wgate.weight.dtype == torch.float32
    split = build_packed(config, torch.tensor([3, 3]), hidden.device)
    assert split.compression_layouts[2].entry_tok_idx.tolist() == [[0, 1], [3, 4]]
    six = torch.randn(1, 6, config.hidden_size)
    latent = compressor(six, split, lambda tensor: tensor)
    kv = torch.nn.functional.linear(six.float(), compressor.wkv.weight.float())
    gate = torch.nn.functional.linear(six.float(), compressor.wgate.weight.float())
    pairs = torch.stack(
        (torch.stack((kv[:, 0], kv[:, 1]), dim=1), torch.stack((kv[:, 3], kv[:, 4]), dim=1)),
        dim=1,
    )
    gates = torch.stack(
        (torch.stack((gate[:, 0], gate[:, 1]), dim=1), torch.stack((gate[:, 3], gate[:, 4]), dim=1)),
        dim=1,
    )
    pooled = (pairs * gates.softmax(dim=2)).sum(dim=2)
    assert torch.allclose(latent, compressor.norm(pooled.to(dtype=six.dtype)))

    torch.manual_seed(0)
    rotary = DeepseekV4RotaryEmbedding(config)
    attention = DeepseekV41Attention(config, 0, rotary)
    attention.compressor.wkv.weight.data.zero_()
    attention.compressor.wgate.weight.data.zero_()
    attention.compressor.norm.weight.data.fill_(1)
    rope_channel = config.head_dim - config.qk_rope_head_dim
    attention.compressor.wkv.weight.data[rope_channel, 0] = 1
    row = torch.zeros(1, 4, config.hidden_size)
    row[:, :, 0] = 1
    _, state = attention(row, packed)
    produced = attention.compressor(row, packed, lambda tensor: tensor)
    cache = rotary.cos_sin_cache("compress")
    rotated = apply_interleaved_rope(produced.unsqueeze(2), cache, torch.tensor([[0, 2]])).squeeze(2)
    shifted = apply_interleaved_rope(produced.unsqueeze(2), cache, torch.tensor([[1, 3]])).squeeze(2)
    assert torch.allclose(state.compress_kv[0], rotated, atol=1e-5)
    assert not torch.allclose(state.compress_kv[0][:, 1], shifted[:, 1], atol=1e-5)


def test_sources_are_reused_and_candidates_mask_later_indexers():
    from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
    from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Attention
    from arctic_platform.model.implementations.deepseek_v41.attention import SharedAttentionState
    from arctic_platform.model.implementations.deepseek_v41.attention import build_packed
    from arctic_platform.model.implementations.deepseek_v41.attention import select_candidate_blocks

    pinned = select_candidate_blocks(torch.tensor([[0.0, 0.0, -10.0, -10.0]]), torch.tensor([4]), 1, 2)
    assert pinned.tolist() == [[False, False, True, True]]

    config = _sparse_config(
        num_hidden_layers=3,
        compress_ratios=[2, 2, 2],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 2],
    )
    rotary = DeepseekV4RotaryEmbedding(config)
    layers = [DeepseekV41Attention(config, layer_idx, rotary) for layer_idx in range(3)]
    assert layers[1].compressor is None and layers[1].indexer is None
    assert layers[2].indexer is not None and not hasattr(layers[2].indexer, "wk")
    assert hasattr(layers[0].compressor, "wkv") and hasattr(layers[0].compressor, "wgate")
    hidden = torch.randn(1, 4, config.hidden_size)
    packed = build_packed(config, torch.tensor([4]), hidden.device)
    _, source_state = layers[0](hidden, packed, SharedAttentionState.empty())
    consumer_out, reused = layers[1](hidden, packed, source_state)
    _, published = layers[2](hidden, packed, reused)
    assert reused.compress_kv[0] is source_state.compress_kv[0]
    assert set(published.compress_kv) == {0}
    assert set(published.index_k) == {0}
    assert set(published.topk) == {0, 2}
    layers[0].compressor.wkv.weight.data.add_(1)
    _, changed = layers[0](hidden, packed, SharedAttentionState.empty())
    changed_out, _ = layers[1](hidden, packed, changed)
    assert not torch.allclose(consumer_out, changed_out, atol=1e-5)

    masked = _sparse_config(
        num_hidden_layers=2,
        compress_ratios=[1, 1],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0, 1],
        candidate_source_layer_id=0,
        index_topk=1,
    )
    rotary = DeepseekV4RotaryEmbedding(masked)
    source = DeepseekV41Attention(masked, 0, rotary)
    later = DeepseekV41Attention(masked, 1, rotary)
    assert not hasattr(source.compressor, "wgate")
    assert not hasattr(later.indexer, "wk")
    packed = build_packed(masked, torch.tensor([4]), hidden.device)
    _, state = source(hidden, packed, SharedAttentionState.empty())
    assert state.candidates.shape == (1, 4, 4)
    assert not bool(state.candidates[0, 3, 0])
    assert bool(state.candidates[0, 3, 2])
    _, state = later(hidden, packed, state)
    assert int(state.topk[1][0, 3, 0]) in (2, 3)
    cache_names = [name for name, _ in source.named_buffers()]
    assert not any(name.endswith(("compress_kv_cache", "k_cache")) or "kv_state" in name for name in cache_names)


def test_consumer_grad_reaches_source_compressor_under_checkpointing():
    from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM
    from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Attention
    from arctic_platform.model.implementations.deepseek_v41.attention import SharedAttentionState
    from arctic_platform.model.implementations.deepseek_v41.attention import build_packed

    torch.manual_seed(0)
    config = _sparse_config(
        num_hidden_layers=2,
        compress_ratios=[2, 2],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
    )
    rotary = DeepseekV4RotaryEmbedding(config)
    source = DeepseekV41Attention(config, 0, rotary)
    consumer = DeepseekV41Attention(config, 1, rotary)
    hidden = torch.randn(1, 4, config.hidden_size)
    packed = build_packed(config, torch.tensor([4]), hidden.device)
    _, state = source(hidden, packed, SharedAttentionState.empty())
    output, _ = consumer(hidden.detach(), packed, state)
    output.sum().backward()
    direct = source.compressor.wkv.weight.grad
    assert direct is not None and torch.isfinite(direct).all() and direct.abs().sum().item() > 0

    model = DeepseekV41ForCausalLM(config)
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    logits = model(input_ids=input_ids, seq_lens=torch.tensor([4]), position_ids=torch.arange(4).view(1, -1))["logits"]
    torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), input_ids.view(-1)).backward()
    graded = model.model.layers[0].self_attn.compressor.wkv.weight.grad
    assert model.model.layers[1].self_attn.compressor is None
    assert graded is not None and torch.isfinite(graded).all() and graded.abs().sum().item() > 0


def test_cp_shard_uses_global_entries(monkeypatch):
    from arctic_platform.model.implementations.deepseek_v4.cp import CPContext
    from arctic_platform.model.implementations.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
    from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Attention
    from arctic_platform.model.implementations.deepseek_v41.attention import build_packed

    config = _sparse_config(max_position_embeddings=16, sliding_window=4)
    rotary = DeepseekV4RotaryEmbedding(config)
    attention = DeepseekV41Attention(config, 0, rotary)
    group = object()
    attention.cp_context = CPContext(cp_group=group, cp_rank=0, cp_world_size=2)

    def gather(tensor, process_group):
        assert process_group is group
        return torch.cat((tensor, torch.zeros_like(tensor)), dim=1)

    monkeypatch.setattr(
        "arctic_platform.model.implementations.deepseek_v41.attention.gather_for_cp",
        gather,
    )
    packed = build_packed(config, torch.tensor([8]), torch.device("cpu"), cp_rank=0, cp_world_size=2)
    other = build_packed(config, torch.tensor([8]), torch.device("cpu"), cp_rank=1, cp_world_size=2)
    assert packed.position_ids.shape == (1, 4)
    assert int(packed.position_ids[0, 0]) == 0
    assert int(other.position_ids[0, 0]) == 4
    assert other.window_indices.shape == (4, config.sliding_window)
    assert int(other.window_indices.max()) >= 4
    assert packed.compression_layouts[2].entry_tok_idx.shape[0] == 4
    hidden = torch.randn(1, 4, config.hidden_size)
    output, state = attention(hidden, packed)
    assert output.shape == (1, 4, config.hidden_size)
    assert state.compress_kv[0].shape == (1, 4, config.head_dim)
    assert state.topk[0].shape == (1, 4, config.index_topk)
    assert state.index_k[0].shape == (1, 4, config.index_head_dim)


def _engram_config(**overrides):
    config = dict(
        num_hidden_layers=2,
        engram_layer_ids=[0],
        engram_num_embeddings=[2048],
        engram_vocab_size=8,
        engram_n_heads=8,
        engram_head_dim=4,
        engram_max_ngram_size=4,
        engram_compressed_vocab_size=32,
        engram_pad_token_id=0,
        engram_quantized=False,
    )
    config.update(overrides)
    return _tiny_config(**config)


def _compressed_map(vocab_size=32):
    return torch.arange(vocab_size)


def test_engram_layout_primes_and_odd_multipliers():
    import numpy as np

    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config
    from arctic_platform.model.implementations.deepseek_v41.engram import EngramLayout
    from arctic_platform.model.implementations.deepseek_v41.engram import compute_hash_multipliers

    config = DeepseekV41Config()
    layout = EngramLayout.from_config(config)
    assert layout.n_hash_cols == 24
    assert layout.layer_ids == (1, 14)
    assert layout.primes[0][0][0] > config.engram_vocab_size - 1
    drawn = [prime for layer in layout.primes for per_ngram in layer for prime in per_ngram]
    assert drawn == sorted(drawn)
    assert len(drawn) == len(set(drawn)) == 2 * 3 * 8
    assert layout.offsets[0][0] == 0 and layout.offsets[1][0] == 0
    flat_layer = [prime for per_ngram in layout.primes[0] for prime in per_ngram]
    assert layout.offsets[0] == tuple([0, *np.cumsum(flat_layer[:-1]).tolist()])

    multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size)
    assert multipliers.shape == (2, 4)
    assert torch.all(multipliers % 2 == 1)
    assert not torch.equal(multipliers[0], multipliers[1])
    bound = max(1, (np.iinfo(np.int64).max // config.engram_compressed_vocab_size) // 2)
    generated = np.random.default_rng(10007 * 14).integers(0, bound, size=(4,), dtype=np.int64) * 2 + 1
    assert torch.equal(multipliers[1], torch.tensor(generated))


def test_engram_hash_matches_xor_and_blocks_boundaries():
    import numpy as np

    from arctic_platform.model.implementations.deepseek_v41.engram import EngramHasher

    config = _engram_config(
        num_hidden_layers=2,
        engram_layer_ids=[0, 1],
        engram_num_embeddings=[2048, 2048],
        engram_n_heads=1,
    )
    hasher = EngramHasher(config)
    mapping = _compressed_map()
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    hashes, mask = hasher(ids, compressed_token_map=mapping, seq_lens=torch.tensor([6]))
    assert hashes.shape == (1, 6, 2, 3)
    assert torch.all(mask)
    assert not torch.equal(hashes[:, :, 0], hashes[:, :, 1])

    tokens = np.array(
        [
            [4, 0, 0, 0],
            [5, 4, 0, 0],
            [6, 5, 4, 0],
            [7, 6, 5, 4],
            [8, 7, 6, 5],
            [9, 8, 7, 6],
        ],
        dtype=np.int64,
    )
    products = tokens[:, None, :] * hasher.multipliers.numpy()[None, :, :]
    rolling = products[..., 0].copy()
    columns = []
    primes = hasher.primes.numpy()
    for index in range(1, 4):
        rolling = np.bitwise_xor(rolling, products[..., index])
        columns.append(rolling[:, :, None] % primes[:, index - 1][None, :, :])
    expected = np.concatenate(columns, axis=-1) + hasher.offsets.numpy()
    assert np.array_equal(hashes[0].numpy(), expected)

    packed, _ = hasher(ids, compressed_token_map=mapping, seq_lens=torch.tensor([3, 3]))
    first, _ = hasher(ids[:, :3], compressed_token_map=mapping, seq_lens=torch.tensor([3]))
    second, _ = hasher(ids[:, 3:], compressed_token_map=mapping, seq_lens=torch.tensor([3]))
    whole, _ = hasher(ids, compressed_token_map=mapping, seq_lens=torch.tensor([6]))
    assert torch.equal(packed[:, :3], first)
    assert torch.equal(packed[:, 3:], second)
    assert not torch.equal(packed[:, 3:], whole[:, 3:])

    image = torch.tensor([[4, 1, 6, 7]])
    pad = torch.tensor([[4, 0, 6, 7]])
    suffix = torch.tensor([[6, 7]])
    image_hash, image_mask = hasher(image, compressed_token_map=mapping, seq_lens=torch.tensor([4]))
    pad_hash, pad_mask = hasher(pad, compressed_token_map=mapping, seq_lens=torch.tensor([4]))
    suffix_hash, _ = hasher(suffix, compressed_token_map=mapping, seq_lens=torch.tensor([2]))
    assert image_mask.tolist() == [[True, False, True, True]]
    assert pad_mask.tolist() == [[True, False, True, True]]
    assert torch.equal(image_hash[:, 2:], suffix_hash)
    assert torch.equal(pad_hash[:, 2:], suffix_hash)
    live, _ = hasher(torch.tensor([[4, 2, 6, 7]]), compressed_token_map=mapping, seq_lens=torch.tensor([4]))
    assert not torch.equal(image_hash[:, 2:], live[:, 2:])


def test_engram_gate_and_frozen_table():
    import pytest

    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM
    from arctic_platform.model.implementations.deepseek_v41.engram import DeepseekV41Engram
    from arctic_platform.model.implementations.deepseek_v41.engram import EngramEmbedding
    from arctic_platform.model.implementations.deepseek_v41.engram import EngramLayout
    from arctic_platform.model.implementations.deepseek_v41.engram import dequantize_block_fp8
    from arctic_platform.model.implementations.deepseek_v41.engram import ep_row_shard

    assert ep_row_shard(7, 2, 0) == (0, 4)
    assert ep_row_shard(7, 2, 1) == (4, 7)
    assert [ep_row_shard(7, 3, rank) for rank in range(3)] == [(0, 3), (3, 6), (6, 7)]
    scales = torch.tensor([0, 127, 128, 126], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert torch.allclose(scales.float(), torch.tensor([2.0**-127, 1.0, 2.0, 0.5]))

    config = _engram_config(num_hidden_layers=1)
    layout = EngramLayout.from_config(config)
    engram = DeepseekV41Engram(config, 0, layout)
    torch.manual_seed(0)
    with torch.no_grad():
        engram.embed.weight.normal_()
        engram.wkv.weight.normal_()
    hidden = torch.randn(1, 3, config.hc_mult, config.hidden_size)
    hash_ids = torch.randint(0, 32, (1, 3, layout.n_hash_cols))
    mask = torch.tensor([[True, False, True]])
    output = engram(hidden, hash_ids, mask)
    rows = engram.embed(hash_ids)
    kv = torch.nn.functional.linear(rows.flatten(-2), engram.wkv.weight)
    key, value = kv.split([config.hc_mult * config.hidden_size, config.hidden_size], dim=-1)
    key = key.float().unflatten(-1, (config.hc_mult, config.hidden_size))
    weight = engram.q_weight.float() * engram.k_weight.float()
    stream = hidden.float()
    rstd = torch.rsqrt(stream.square().mean(-1) + engram.eps) * torch.rsqrt(key.square().mean(-1) + engram.eps)
    dot = (stream * weight * key).sum(-1) * rstd * config.hidden_size**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(engram.clamp_value).sqrt(), dot))
    gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    expected = (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(dtype=hidden.dtype)
    assert torch.allclose(output, expected)
    assert torch.equal(output[:, 1], hidden[:, 1])
    assert not torch.equal(output[:, 0], hidden[:, 0])

    codes = torch.tensor([2.0, 1.0, 1.5, 1.0]).view(4, 1).expand(4, 32).contiguous().to(torch.float8_e4m3fn)
    code_scales = torch.tensor([127, 128, 126, 0], dtype=torch.uint8).view(torch.float8_e8m0fnu).view(4, 1)
    table = EngramEmbedding(4, 32, quantized=True)
    table.weight.data.copy_(codes)
    table.scale.data.copy_(code_scales)
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    assert table._dss_shard_on_ep and table._dss_skip_weight_sync
    assert table.weight.requires_grad is False and table.scale.requires_grad is False
    assert table.weight._dss_skip_weight_sync and table.scale._dss_skip_weight_sync
    looked = table(torch.tensor([3, 0, 1, 3]))
    assert torch.equal(looked, dequantize_block_fp8(codes, code_scales)[[3, 0, 1, 3]])
    assert looked[1, 0].item() == 2.0
    assert looked[2, 0].item() == 2.0
    table.to(dtype=torch.bfloat16)
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    assert table.weight._dss_skip_weight_sync and table.scale._dss_skip_weight_sync
    with pytest.raises(ValueError, match="engram ids"):
        table(torch.tensor([4]))

    model = DeepseekV41ForCausalLM(config)
    layer = model.model.layers[0]
    assert layer.engram is not None
    assert torch.equal(layer.engram.q_weight, torch.ones_like(layer.engram.q_weight))
    assert torch.equal(layer.engram.k_weight, torch.ones_like(layer.engram.k_weight))
    assert layer.engram.embed.weight.requires_grad is False
    with torch.no_grad():
        layer.engram.embed.weight.normal_()
    mapping = _compressed_map()
    ids = torch.tensor([[4, 5, 6, 7]])
    with pytest.raises(RuntimeError, match="compressed_token_map"):
        model(input_ids=ids, seq_lens=torch.tensor([4]))
    with pytest.raises(ValueError, match="input_ids"):
        model(inputs_embeds=torch.randn(1, 4, config.hidden_size), seq_lens=torch.tensor([4]))
    with pytest.raises(ValueError, match="engram_compressed_vocab_size"):
        model(input_ids=ids, seq_lens=torch.tensor([4]), compressed_token_map=torch.arange(8))
    with pytest.raises(RuntimeError, match="compressed_pad_id"):
        model(input_ids=ids, seq_lens=torch.tensor([4]), compressed_token_ids=ids)

    seen = {}
    original = layer.attn_hc.mixes

    def mixes(states):
        seen["states"] = states.detach().clone()
        return original(states)

    layer.attn_hc.mixes = mixes
    logits = model(input_ids=ids, seq_lens=torch.tensor([4]), compressed_token_map=mapping)["logits"]
    hashes, token_mask = model.model.engram_hasher(ids, compressed_token_map=mapping, seq_lens=torch.tensor([4]))
    stream = model.model.embed_tokens(ids).unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
    assert torch.allclose(seen["states"], layer.engram(stream, hashes[:, :, 0], token_mask))
    via_ids, _ = model.model.engram_hasher(
        ids,
        compressed_token_ids=mapping[ids],
        compressed_pad_id=int(mapping[config.engram_pad_token_id]),
        seq_lens=torch.tensor([4]),
    )
    assert torch.equal(hashes, via_ids)
    torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), ids.view(-1)).backward()
    assert layer.engram.embed.weight.grad is None
    for param in (layer.engram.wkv.weight, layer.engram.q_weight, layer.engram.k_weight):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0


def test_bf16_export_keeps_engram_quantized_and_drops_deferred_keys():
    import pytest

    from arctic_platform.model.implementations.deepseek_v4.conversion_ops import apply_hf_to_prime
    from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import conversion_chain
    from arctic_platform.model.implementations.deepseek_v41.dequantize import dequantize_state_dict_

    config = _engram_config(
        num_hidden_layers=1,
        engram_layer_ids=[0],
        engram_num_embeddings=[4],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        compress_ratios=[0],
        engram_quantized=True,
    )
    dense = torch.randn(8, 16, dtype=torch.bfloat16)
    engram = torch.zeros(4, 32, dtype=torch.float8_e4m3fn)
    scale = torch.full((4, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    state = {
        "embed.weight": dense.clone(),
        "layers.0.attn.wq_a.weight": dense.clone(),
        "layers.0.engram.embed.weight": engram,
        "layers.0.engram.embed.scale": scale,
        "layers.0.engram.wkv.weight": dense.clone(),
        "vision.encoder.weight": dense.clone(),
        "mtp.0.hc_attn_fn": torch.zeros(1),
    }
    dequantize_state_dict_(state)
    converted = apply_hf_to_prime(state, conversion_chain(config))
    assert torch.equal(converted["model.embed_tokens.weight"], dense)
    assert torch.equal(converted["model.layers.0.self_attn.q_a_proj.weight"], dense)
    assert converted["model.layers.0.engram.embed.weight"].dtype == torch.float8_e4m3fn
    assert converted["model.layers.0.engram.embed.scale"].dtype == torch.float8_e8m0fnu
    assert "vision.encoder.weight" not in converted
    assert not any(key.startswith("mtp.") for key in converted)

    with pytest.raises(ValueError, match="unexplained"):
        apply_hf_to_prime({"layers.0.mystery": dense.clone()}, conversion_chain(config))
