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
import copy
import json
import logging

import pytest
import torch
from pydantic import ValidationError

from arctic_platform.common.option_registry import NOT_APPLICABLE
from arctic_platform.common.option_registry import settle
from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_weight
from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config
from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM
from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import DEFERRED_CHECKPOINT_NAMESPACES
from arctic_platform.model.implementations.deepseek_v41.dequantize import dequantize_state_dict_
from arctic_platform.model.implementations.deepseek_v41.engram import EngramHasher
from arctic_platform.model.implementations.deepseek_v41.engram import build_compressed_token_map
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loaders.deepseek_v41 import load_deepseek_v41


def _tiny_config(**overrides):
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


def _convert_config():
    return _tiny_config(
        num_hidden_layers=4,
        compress_ratios=[0, 2, 1, 1],
        kv_source_layer_ids=[1, 2],
        index_source_layer_ids=[1, 2, 3],
        candidate_source_layer_id=-1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        engram_layer_ids=[0],
        engram_num_embeddings=[32],
        engram_vocab_size=8,
        engram_n_heads=1,
        engram_head_dim=4,
        engram_max_ngram_size=2,
        engram_compressed_vocab_size=32,
        engram_quantized=False,
        vision_n_layers=1,
    )


def _e8m0(values, shape):
    return torch.tensor(values, dtype=torch.uint8).view(torch.float8_e8m0fnu).reshape(shape)


def _checkpoint(tmp_path, payload: dict) -> str:
    path = tmp_path / "deepseek_v41"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(payload))
    return str(path)


class _Backend:
    def __init__(self, tokens, raw):
        self.tokens = tokens
        self.raw = raw

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return self.tokens[ids[0]]

    def id_to_token(self, token_id):
        return self.raw[token_id]


class _Tokenizer:
    def __init__(self, tokens, raw=None):
        self.tokens = tokens
        self.backend_tokenizer = _Backend(tokens, tokens if raw is None else raw)

    def __len__(self):
        return len(self.tokens)


def _published_state():
    ones = torch.ones(32, 32, dtype=torch.float8_e4m3fn)
    scale_two = _e8m0([128], (1, 1))
    packed = torch.full((2, 16), 0x11, dtype=torch.int8)
    expert_scale = _e8m0([127, 127], (2, 1))
    state = {
        "embed.weight": torch.zeros(4, 4),
        "norm.weight": torch.ones(4),
        "head.weight": torch.zeros(4, 4),
        "vision.blocks.0.weight": torch.zeros(1),
        "aligner.w1.weight": torch.zeros(1),
        "image_start": torch.zeros(1),
        "image_newline": torch.zeros(1),
        "mtp.0.norm.weight": torch.zeros(1),
        "layers.0.attn.wq_a.weight": ones.clone(),
        "layers.0.attn.wq_a.scale": scale_two.clone(),
        "layers.0.attn.q_norm.weight": torch.ones(4),
        "layers.0.attn.wq_b.weight": torch.zeros(4, 4),
        "layers.0.attn.wkv.weight": torch.zeros(4, 4),
        "layers.0.attn.kv_norm.weight": torch.ones(4),
        "layers.0.attn.wo_a.weight": torch.zeros(4, 4),
        "layers.0.attn.wo_b.weight": torch.zeros(4, 4),
        "layers.0.attn.attn_sink": torch.zeros(4),
        "layers.0.attn_norm.weight": torch.ones(4),
        "layers.0.ffn_norm.weight": torch.ones(4),
        "layers.0.hc_attn_fn": torch.zeros(2, 4),
        "layers.0.hc_attn_base": torch.zeros(2),
        "layers.0.hc_attn_scale": torch.ones(3),
        "layers.0.hc_ffn_fn": torch.zeros(2, 4),
        "layers.0.hc_ffn_base": torch.zeros(2),
        "layers.0.hc_ffn_scale": torch.ones(3),
        "layers.0.ffn.gate.weight": torch.zeros(4, 4),
        "layers.0.ffn.gate.bias": torch.ones(4),
        "layers.0.ffn.gate.bias_vl": torch.full((4,), 0.25),
        "layers.0.ffn.shared_experts.w1.weight": ones.clone(),
        "layers.0.ffn.shared_experts.w1.scale": scale_two.clone(),
        "layers.0.ffn.shared_experts.w2.weight": torch.zeros(4, 4),
        "layers.0.ffn.shared_experts.w3.weight": torch.zeros(4, 4),
        "layers.0.engram.embed.weight": torch.ones(4, 32, dtype=torch.float8_e4m3fn),
        "layers.0.engram.embed.scale": _e8m0([127, 128, 126, 0], (4, 1)),
        "layers.0.engram.wkv.weight": ones.clone(),
        "layers.0.engram.wkv.scale": scale_two.clone(),
        "layers.0.engram.q_weight": torch.ones(2, 4),
        "layers.0.engram.k_weight": torch.ones(2, 4),
        "layers.1.attn.compressor.wkv.weight": torch.ones(4, 4),
        "layers.1.attn.compressor.wgate.weight": torch.full((4, 4), 2.0),
        "layers.1.attn.compressor.norm.weight": torch.ones(4),
        "layers.1.attn.indexer.wq_b.weight": torch.zeros(4, 4),
        "layers.1.attn.indexer.weights_proj.weight": torch.zeros(4, 4),
        "layers.1.attn.indexer.wk.weight": torch.ones(4, 4),
        "layers.1.attn.indexer.k_norm.weight": torch.ones(4),
        "layers.2.attn.compressor.wkv.weight": torch.ones(4, 4),
        "layers.2.attn.compressor.norm.weight": torch.ones(4),
        "layers.2.attn.indexer.wk.weight": torch.ones(4, 4),
        "layers.2.attn.indexer.k_norm.weight": torch.ones(4),
        "layers.3.attn.indexer.wq_b.weight": torch.zeros(4, 4),
        "layers.3.attn.indexer.weights_proj.weight": torch.zeros(4, 4),
        "layers.3.attn.attn_sink": torch.zeros(2),
    }
    for expert in (0, 1):
        state[f"layers.0.ffn.experts.{expert}.w1.weight"] = packed.clone()
        state[f"layers.0.ffn.experts.{expert}.w1.scale"] = expert_scale.clone()
        state[f"layers.0.ffn.experts.{expert}.w2.weight"] = torch.zeros(4, 4)
        state[f"layers.0.ffn.experts.{expert}.w3.weight"] = torch.zeros(4, 4)
    return state


def test_dequantize_fp8_mxfp4_and_keeps_engram_table():
    state = _published_state()
    embed = state["layers.0.engram.embed.weight"].clone()
    embed_scale = state["layers.0.engram.embed.scale"].clone()
    dequantize_state_dict_(state)
    expected_dense = dequantize_weight(
        torch.ones(32, 32, dtype=torch.float8_e4m3fn),
        _e8m0([128], (1, 1)),
    )
    expected_expert = dequantize_weight(torch.full((2, 16), 0x11, dtype=torch.int8), _e8m0([127, 127], (2, 1)))
    assert state["layers.0.attn.wq_a.weight"].dtype == torch.bfloat16
    assert torch.equal(state["layers.0.attn.wq_a.weight"], expected_dense)
    assert "layers.0.attn.wq_a.scale" not in state
    assert torch.equal(state["layers.0.ffn.shared_experts.w1.weight"], expected_dense)
    assert torch.equal(state["layers.0.ffn.experts.0.w1.weight"], expected_expert)
    assert expected_expert.shape == (2, 32)
    assert "layers.0.ffn.experts.0.w1.scale" not in state
    assert state["layers.0.engram.embed.weight"].dtype == torch.float8_e4m3fn
    assert state["layers.0.engram.embed.scale"].dtype == torch.float8_e8m0fnu
    assert torch.equal(state["layers.0.engram.embed.weight"], embed)
    assert torch.equal(state["layers.0.engram.embed.scale"], embed_scale)
    assert state["layers.0.engram.wkv.weight"].dtype == torch.bfloat16
    assert "layers.0.engram.wkv.scale" not in state


def test_published_keys_convert_and_round_trip(caplog):
    caplog.set_level(logging.INFO)
    config = _convert_config()
    model = DeepseekV41ForCausalLM(config)
    raw = _published_state()
    assert model.is_hf_state_dict(raw)
    assert not model.is_prime_state_dict(raw)
    assert model.is_hf_state_dict({"embed.weight": raw["embed.weight"]})
    assert model.is_prime_state_dict({"model.embed_tokens.weight": raw["embed.weight"]})
    assert model.is_prime_state_dict({"lm_head.weight": raw["head.weight"]})
    assert not model.is_hf_state_dict({"lm_head.weight": raw["head.weight"]})

    dequantized = copy.deepcopy(raw)
    dequantize_state_dict_(dequantized)
    prime = model.convert_to_prime(copy.deepcopy(raw))
    assert model.is_prime_state_dict(prime)
    assert not model.is_hf_state_dict(prime)
    assert DEFERRED_CHECKPOINT_NAMESPACES == ("vision.", "aligner.", "image_", "mtp.")
    assert "DeepSeek V4.1 text conversion defers namespaces vision., aligner., image_, mtp." in caplog.text
    assert "present in this checkpoint: vision., aligner., image_, mtp." in caplog.text
    for key in ("vision.blocks.0.weight", "aligner.w1.weight", "image_start", "image_newline", "mtp.0.norm.weight"):
        assert key not in prime

    layer = "model.layers.0"
    assert torch.equal(prime[f"{layer}.self_attn.q_a_proj.weight"], dequantized["layers.0.attn.wq_a.weight"])
    assert prime[f"{layer}.self_attn.q_a_norm.weight"].shape == (4,)
    assert f"{layer}.self_attn.q_b_proj.weight" in prime
    assert f"{layer}.self_attn.kv_proj.weight" in prime
    assert f"{layer}.self_attn.kv_norm.weight" in prime
    assert f"{layer}.self_attn.o_a_proj.weight" in prime
    assert f"{layer}.self_attn.o_b_proj.weight" in prime
    assert f"{layer}.self_attn.sinks" in prime
    assert f"{layer}.input_layernorm.weight" in prime
    assert f"{layer}.post_attention_layernorm.weight" in prime
    assert torch.equal(prime[f"{layer}.attn_hc.fn"], raw["layers.0.hc_attn_fn"])
    assert f"{layer}.attn_hc.base" in prime and f"{layer}.ffn_hc.scale" in prime
    assert torch.equal(prime[f"{layer}.mlp.expert_bias"], raw["layers.0.ffn.gate.bias"])
    assert torch.equal(prime[f"{layer}.mlp.bias_vl"], raw["layers.0.ffn.gate.bias_vl"])
    assert torch.equal(prime[f"{layer}.mlp.shared_expert.w1"], dequantized["layers.0.ffn.shared_experts.w1.weight"])
    assert prime[f"{layer}.mlp.experts.w1"].shape == (2, 2, 32)
    assert prime[f"{layer}.mlp.experts.w2"].shape == (2, 4, 4)
    assert prime[f"{layer}.engram.embed.weight"].dtype == torch.float8_e4m3fn
    assert prime[f"{layer}.engram.embed.scale"].dtype == torch.float8_e8m0fnu
    assert prime[f"{layer}.engram.wkv.weight"].dtype == torch.bfloat16
    assert f"{layer}.engram.wkv.scale" not in prime
    assert f"{layer}.engram.q_weight" in prime and f"{layer}.engram.k_weight" in prime
    assert "model.embed_tokens.weight" in prime and "lm_head.weight" in prime and "model.norm.weight" in prime

    assert "model.layers.1.self_attn.compressor.wgate.weight" in prime
    assert "model.layers.1.self_attn.compressor.wkv.weight" in prime
    assert "model.layers.1.self_attn.indexer.wk.weight" in prime
    assert "model.layers.1.self_attn.indexer.k_norm.weight" in prime
    assert "model.layers.2.self_attn.compressor.wkv.weight" in prime
    assert "model.layers.2.self_attn.compressor.wgate.weight" not in prime
    assert "model.layers.2.self_attn.indexer.wk.weight" in prime
    assert "model.layers.3.self_attn.indexer.wq_b.weight" in prime
    assert "model.layers.3.self_attn.indexer.wk.weight" not in prime
    assert "model.layers.3.self_attn.compressor.wkv.weight" not in prime
    assert "model.layers.0.self_attn.compressor.wkv.weight" not in prime
    assert "model.layers.3.self_attn.sinks" in prime

    restored = model.convert_to_hf(copy.deepcopy(prime))
    assert torch.equal(restored["layers.0.ffn.experts.0.w1.weight"], dequantized["layers.0.ffn.experts.0.w1.weight"])
    shared = "layers.0.ffn.shared_experts.w1.weight"
    assert torch.equal(restored[shared], dequantized[shared])
    assert torch.equal(restored["layers.0.attn.wq_a.weight"], dequantized["layers.0.attn.wq_a.weight"])
    assert torch.equal(restored["head.weight"], raw["head.weight"])
    assert "layers.0.ffn.experts.0.w1.scale" not in restored
    assert restored["layers.0.engram.embed.scale"].dtype == torch.float8_e8m0fnu

    saved = model.model.engram_hasher.token_map.clone()
    model.init_buffers_post_meta()
    assert torch.equal(model.model.engram_hasher.token_map, saved)
    counts = model.model.layers[0].mlp.tokens_per_expert
    assert torch.equal(counts, torch.zeros_like(counts))


def test_source_only_and_unknown_keys_are_rejected():
    model = DeepseekV41ForCausalLM(_convert_config())
    surprises = {
        "ratio-1 wgate": {"layers.2.attn.compressor.wgate.weight": torch.zeros(1)},
        "non-source compressor": {"layers.0.attn.compressor.wkv.weight": torch.zeros(1)},
        "index consumer wk": {"layers.3.attn.indexer.wk.weight": torch.zeros(1)},
        "unknown": {"layers.0.tid2eid": torch.zeros(1)},
    }
    for name, extra in surprises.items():
        with pytest.raises(ValueError, match="unexplained"):
            model.convert_to_prime(extra)
        assert name


@pytest.mark.parametrize("peft", [None, {"peft_type": "Lora", "r": 8}])
def test_only_lora_keeps_the_official_experts_packed(tmp_path, monkeypatch, peft):
    from arctic_platform.model.implementations.deepseek_v41 import deepspeed_integration

    calls = []
    monkeypatch.setattr(deepspeed_integration, "load_deepseek_v41_model", lambda **kwargs: calls.append(kwargs))
    path = _checkpoint(tmp_path, {"model_type": "deepseek_v41", "n_routed_experts": 384})
    spec = ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=8), patches={"peft": peft})
    load_deepseek_v41(LoaderContext(spec=spec, parallel_groups={"ep_group": object()}))
    assert calls[0]["dequantize_checkpoint"] is (peft is None)


def test_loader_selection_options_and_groups(tmp_path):
    path = _checkpoint(tmp_path, {"model_type": "deepseek_v41", "n_routed_experts": 384})
    spec = ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=8))
    assert spec.loader == "deepseek_v41"
    assert spec.attn_implementation == "eager"

    nested = tmp_path / "nested"
    nested.mkdir()
    text = {"model_type": "deepseek_v41_text", "n_routed_experts": 384}
    (nested / "config.json").write_text(json.dumps({"model_type": "deepseek_v41_vl", "text_config": text}))
    nested_spec = ModelSpec(model_path_or_name=str(nested), parallelism=ParallelismConfig(expert_parallel=8))
    assert nested_spec.loader == "deepseek_v41"

    plain = ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=1))
    assert plain.loader == "huggingface"
    parallel = ModelSpec(
        model_path_or_name=path,
        parallelism=ParallelismConfig(expert_parallel=8, sequence_parallel=2),
    )
    assert parallel.loader == "deepseek_v41"
    assert parallel.parallelism.sequence_parallel == 2
    with pytest.raises(ValueError, match="must divide"):
        ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=7))
    with pytest.raises(ValueError, match="liger"):
        ModelSpec(model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=8), patches={"liger": True})
    with pytest.raises(ValueError, match="generic forward patches"):
        ModelSpec(
            model_path_or_name=path,
            parallelism=ParallelismConfig(expert_parallel=8),
            patches={"gradient_checkpointing": True},
        )
    with pytest.raises(ValueError, match="fused sparse"):
        ModelSpec(
            model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=8), attn_implementation="sdpa"
        )
    with pytest.raises(ValidationError):
        ModelSpec(
            model_path_or_name=path, parallelism=ParallelismConfig(expert_parallel=8), loader_options={"nope": 1}
        )
    with pytest.raises(ValueError, match="ep_group"):
        load_deepseek_v41(LoaderContext(spec=spec, parallel_groups={}))
    assert settle("deepseek_v41")["attention.sparse_mla"].status == NOT_APPLICABLE


def test_cp_contexts_land_on_model_and_attention(monkeypatch):
    import torch.distributed as dist

    from arctic_platform.model.implementations.deepseek_v41.deepspeed_integration import _install_context_parallel

    model = DeepseekV41ForCausalLM(_tiny_config())
    group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda process_group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda process_group: 1)
    with pytest.raises(ValueError, match="SP group"):
        _install_context_parallel(model, None, 2)
    _install_context_parallel(model, group, 2)
    assert model.model.cp_context.cp_group is group
    assert model.model.cp_context.cp_rank == 1
    assert model.model.cp_context.cp_world_size == 2
    assert model.model.layers[0].self_attn.cp_context is model.model.cp_context


def test_compressed_token_map_normalizes_and_defaults_on_the_model(tmp_path, monkeypatch):
    tokens = ["The", "the", "THE", "café", "cafe", "a  b", "a b", " ", "\ufffd", "\ufffd"]
    raw = tokens[:-2] + ["<byte8>", "<byte9>"]
    lookup, size = build_compressed_token_map(_Tokenizer(tokens, raw))
    assert lookup[0] == lookup[1] == lookup[2]
    assert lookup[3] == lookup[4]
    assert lookup[5] == lookup[6]
    assert lookup[7] not in {lookup[0], lookup[3], lookup[5]}
    assert lookup[8] != lookup[9]
    assert size == len(set(lookup))

    config = _tiny_config(
        vocab_size=16,
        num_hidden_layers=1,
        engram_layer_ids=[0],
        engram_num_embeddings=[2048],
        engram_vocab_size=8,
        engram_n_heads=1,
        engram_head_dim=4,
        engram_max_ngram_size=2,
        engram_compressed_vocab_size=size,
        engram_quantized=False,
    )
    model = DeepseekV41ForCausalLM(config)
    ids = torch.tensor([[0, 3, 7, 8]])
    with pytest.raises(RuntimeError, match="compressed_token_map"):
        model(input_ids=ids, seq_lens=torch.tensor([4]))
    mapping = torch.tensor(lookup, dtype=torch.long)
    model.install_compressed_token_map(mapping)
    assert "token_map" not in model.model.engram_hasher.state_dict()
    installed = model.model.engram_hasher.token_map
    assert installed.dtype == torch.long and installed.numel() == len(lookup)
    default = model(input_ids=ids, seq_lens=torch.tensor([4]))["logits"]
    explicit = model(input_ids=ids, seq_lens=torch.tensor([4]), compressed_token_map=mapping)["logits"]
    via_ids = model(
        input_ids=ids,
        seq_lens=torch.tensor([4]),
        compressed_token_ids=mapping[ids],
        compressed_pad_id=int(mapping[config.engram_pad_token_id]),
    )["logits"]
    assert torch.equal(default, explicit)
    assert torch.equal(default, via_ids)

    bare = EngramHasher(config)
    assert bare.token_map.numel() == 0
    bare.install_compressed_token_map(mapping)
    hashed, _ = bare(ids, seq_lens=torch.tensor([4]))
    hashed_explicit, _ = bare(ids, compressed_token_map=mapping, seq_lens=torch.tensor([4]))
    assert torch.equal(hashed, hashed_explicit)

    from arctic_platform.model.implementations.deepseek_v41.deepspeed_integration import (
        install_tokenizer_compressed_map,
    )

    assert install_tokenizer_compressed_map(model, str(tmp_path)) is False
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(["only"]),
    )
    with pytest.raises(ValueError, match="compressed vocab"):
        install_tokenizer_compressed_map(model, str(tmp_path))
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(tokens, raw),
    )
    assert install_tokenizer_compressed_map(model, str(tmp_path)) is True
