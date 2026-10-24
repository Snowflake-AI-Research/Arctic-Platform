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

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config
from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM
from arctic_platform.model.implementations.deepseek_v41 import streaming_conversion
from arctic_platform.model.implementations.deepseek_v41.streaming_conversion import write_streaming_prime_cache
from arctic_platform.model.implementations.moe.conversion_cache import _write_conversion_cache
from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_ready

ENGRAM_ROWS = 4096


def _config():
    return DeepseekV41Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=4,
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
        image_token_id=1,
        dspark_noise_token_id=1,
        pad_token_id=0,
        swiglu_limit=10.0,
        compress_ratios=[0, 2, 1, 1],
        kv_source_layer_ids=[1, 2],
        index_source_layer_ids=[1, 2, 3],
        candidate_source_layer_id=-1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        engram_layer_ids=[0],
        engram_num_embeddings=[ENGRAM_ROWS],
        engram_vocab_size=8,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_max_ngram_size=2,
        engram_compressed_vocab_size=32,
        engram_quantized=True,
        vision_n_layers=1,
    )


def _fp8_pair(weight):
    return weight.to(torch.float8_e4m3fn), torch.tensor([[128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)


def _mxfp4_pair(rows, cols, seed):
    generator = torch.Generator().manual_seed(seed)
    packed = torch.randint(-128, 128, (rows, cols // 2), dtype=torch.int8, generator=generator)
    scale = torch.randint(124, 130, (rows, 1), dtype=torch.uint8, generator=generator).view(torch.float8_e8m0fnu)
    return packed, scale


def _published_source(model):
    torch.manual_seed(0)
    prime = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    for name, tensor in prime.items():
        if tensor.is_floating_point() and tensor.dtype not in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            prime[name] = torch.randn_like(tensor)
    table = torch.randn(ENGRAM_ROWS, 32).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 135, (ENGRAM_ROWS, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    prime["model.layers.0.engram.embed.weight"] = table
    prime["model.layers.0.engram.embed.scale"] = scale
    source = model.convert_to_hf(prime)
    source = {name: tensor.contiguous() for name, tensor in source.items()}
    weight, wscale = _fp8_pair(source["layers.0.attn.wq_a.weight"])
    source["layers.0.attn.wq_a.weight"], source["layers.0.attn.wq_a.scale"] = weight, wscale
    weight, wscale = _fp8_pair(source["layers.0.engram.wkv.weight"])
    source["layers.0.engram.wkv.weight"], source["layers.0.engram.wkv.scale"] = weight, wscale
    for expert in range(4):
        for proj, (rows, cols) in (("w1", (8, 16)), ("w3", (8, 16)), ("w2", (16, 8))):
            packed, pscale = _mxfp4_pair(rows, cols, seed=expert * 10 + len(proj))
            source[f"layers.1.ffn.experts.{expert}.{proj}.weight"] = packed
            source[f"layers.1.ffn.experts.{expert}.{proj}.scale"] = pscale
    source["vision.blocks.0.attn.wo.weight"] = torch.zeros(2, 2)
    source["aligner.w1.weight"] = torch.zeros(2, 2)
    source["image_newline"] = torch.zeros(16)
    source["mtp.0.ffn.experts.0.w1.weight"] = torch.zeros(2, 2)
    return source


def _write_source(path, source):
    path.mkdir()
    groups = {
        "model-00001-of-00004.safetensors": lambda k: k.startswith("layers.0."),
        "model-00002-of-00004.safetensors": lambda k: k.startswith(("layers.1.", "layers.2.", "layers.3.")),
        "model-00003-of-00004.safetensors": lambda k: ".engram.embed." in k,
    }
    weight_map = {}
    for key in source:
        target = next((name for name, match in groups.items() if match(key)), "model-00004-of-00004.safetensors")
        weight_map[key] = target
    engram = {"layers.0.engram.embed.weight", "layers.0.engram.embed.scale"}
    for name in sorted(set(weight_map.values())):
        keys = [key for key, target in weight_map.items() if target == name]
        if name == "model-00003-of-00004.safetensors":
            assert set(keys) == engram
        save_file({key: source[key] for key in keys}, path / name, metadata={"format": "pt"})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def _read_all(path):
    out = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                out[key] = handle.get_tensor(key)
    return out


def test_streaming_cache_matches_in_memory_and_loads_through_dcp(tmp_path, monkeypatch):
    config = _config()
    model = DeepseekV41ForCausalLM(config)
    source = _published_source(model)
    source_dir = tmp_path / "source"
    _write_source(source_dir, source)
    expected = model.convert_to_prime(copy.deepcopy(source))

    loaded_keys = []
    real_safe_open = streaming_conversion.safe_open

    class RecordingOpen:
        def __init__(self, *args, **kwargs):
            self.inner = real_safe_open(*args, **kwargs)

        def __enter__(self):
            handle = self.inner.__enter__()
            original = handle.get_tensor

            class Wrapped:
                def get_tensor(self, key):
                    loaded_keys.append(key)
                    return original(key)

            return Wrapped()

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

    monkeypatch.setattr(streaming_conversion, "safe_open", RecordingOpen)
    writes = []
    real_copy = streaming_conversion._copy_range

    def recording_copy(destination, source_tensor, handles, chunk_bytes):
        class Probe:
            def write(self, data):
                writes.append((source_tensor.key, len(data)))
                return destination.write(data)

        return real_copy(Probe(), source_tensor, handles, chunk_bytes)

    monkeypatch.setattr(streaming_conversion, "_copy_range", recording_copy)
    out_dir = tmp_path / "prime"
    plan = write_streaming_prime_cache(
        source_dir,
        out_dir,
        model.conversion_chain(config),
        chunk_bytes=4096,
        shard_target_bytes=64 * 1024,
        workers=3,
    )

    assert all(".engram.embed." not in key for key in loaded_keys)
    experts = [f"layers.1.ffn.experts.{e}.{p}" for e in range(4) for p in ("w1", "w2", "w3")]
    assert set(loaded_keys) == {
        *(f"{expert}.{suffix}" for expert in experts for suffix in ("weight", "scale")),
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.engram.wkv.weight",
        "layers.0.engram.wkv.scale",
    }
    table_writes = [size for key, size in writes if key == "layers.0.engram.embed.weight"]
    assert len(table_writes) == ENGRAM_ROWS * 32 // 4096 and max(table_writes) == 4096
    assert len(plan["model.layers.1.mlp.experts.w1"].pieces) == 4

    index = json.loads((out_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(expected)
    assert len(set(index["weight_map"].values())) > 3
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in expected.values())
    assert conversion_cache_ready(out_dir)

    streamed = _read_all(out_dir)
    assert set(streamed) == set(expected)
    for key, tensor in expected.items():
        assert streamed[key].dtype == tensor.dtype, key
        assert torch.equal(streamed[key].view(torch.uint8), tensor.contiguous().view(torch.uint8)), key
    assert streamed["model.layers.0.engram.embed.weight"].dtype == torch.float8_e4m3fn
    assert streamed["model.layers.0.engram.embed.scale"].dtype == torch.float8_e8m0fnu
    assert "model.layers.0.engram.wkv.scale" not in streamed
    assert streamed["model.layers.1.mlp.experts.w1"].shape == (4, 8, 16)
    assert streamed["model.layers.1.mlp.experts.w1"].dtype == torch.bfloat16
    assert not any(key.startswith(("vision.", "aligner.", "image_", "mtp.")) for key in streamed)

    fresh = DeepseekV41ForCausalLM(config)
    state = fresh.state_dict()
    dcp_load(state, storage_reader=HuggingFaceStorageReader(path=out_dir.as_posix()), no_dist=True)
    for key, tensor in state.items():
        torch.testing.assert_close(tensor, expected[key].to(tensor.dtype), rtol=0, atol=0, msg=key)


def test_cache_writer_streams_without_loading_the_snapshot(tmp_path):
    target = tmp_path / "cache" / "prime"
    target.parent.mkdir()
    calls = []

    def stream(source, output):
        calls.append((source, output))
        output.mkdir()
        save_file({"a": torch.zeros(1)}, output / "model-00001-of-00001.safetensors")
        (output / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": {"a": "model-00001-of-00001.safetensors"}})
        )

    def refuse(*args, **kwargs):
        raise AssertionError("snapshot must not be loaded into memory")

    _write_conversion_cache(
        tmp_path, target, refuse, "HF", "PrimeRL", refuse, refuse, rank=0, local_rank=0, stream_convert_fn=stream
    )
    assert calls and calls[0][0] == tmp_path
    assert conversion_cache_ready(target)

    def broken(source, output):
        output.mkdir()
        raise RuntimeError("disk full")

    other = tmp_path / "cache" / "other"
    with pytest.raises(RuntimeError, match="disk full"):
        _write_conversion_cache(
            tmp_path, other, refuse, "HF", "PrimeRL", refuse, refuse, rank=0, local_rank=0, stream_convert_fn=broken
        )
    assert not other.exists()
    assert not any(path.name.startswith("other.tmp") for path in other.parent.iterdir())


def test_quantized_engram_table_survives_to_empty():
    from arctic_platform.model.implementations.deepseek_v41.engram import EngramEmbedding

    with torch.device("meta"):
        table = EngramEmbedding(64, 32, quantized=True)
    table.to_empty(device="cpu")
    assert table.weight.device.type == "cpu" and table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    table.to(dtype=torch.bfloat16)
    assert table.weight.dtype == torch.float8_e4m3fn


_OFFICIAL_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def test_lora_keeps_official_experts_packed_through_streaming_conversion(tmp_path):
    from arctic_platform.model.implementations.deepseek_v41.mxfp4_experts import Mxfp4GroupedExperts

    config = _config()
    config.hidden_size = 32
    config.moe_intermediate_size = 32
    source = {name: tensor.contiguous() for name, tensor in _published_source(DeepseekV41ForCausalLM(config)).items()}
    for layer in range(config.num_hidden_layers):
        for expert in range(4):
            for proj in ("w1", "w2", "w3"):
                packed, pscale = _mxfp4_pair(32, 32, seed=layer * 100 + expert * 10 + int(proj[1]))
                source[f"layers.{layer}.ffn.experts.{expert}.{proj}.weight"] = packed
                source[f"layers.{layer}.ffn.experts.{expert}.{proj}.scale"] = pscale
    source_dir = tmp_path / "source"
    _write_source(source_dir, source)

    config.quantization_config = dict(_OFFICIAL_QUANTIZATION)
    model = DeepseekV41ForCausalLM(config)
    experts = model.model.layers[1].mlp.experts
    assert isinstance(experts, Mxfp4GroupedExperts)
    assert not any(param.requires_grad for param in experts.parameters())

    expected = model.convert_to_prime(copy.deepcopy(source))
    assert expected["model.layers.1.mlp.experts.w1"].dtype == torch.int8
    assert expected["model.layers.1.mlp.experts.w1"].shape == (4, 32, 16)
    assert expected["model.layers.1.mlp.experts.w1_scale_inv"].dtype == torch.float32
    assert expected["model.layers.1.mlp.experts.w1_scale_inv"].shape == (4, 32, 1)
    assert expected["model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.bfloat16

    out_dir = tmp_path / "prime"
    model.stream_convert_to_prime(source_dir, out_dir)
    streamed = _read_all(out_dir)
    assert set(streamed) == set(expected) == set(model.state_dict())
    for key, tensor in expected.items():
        assert streamed[key].dtype == tensor.dtype, key
        assert torch.equal(streamed[key].view(torch.uint8), tensor.contiguous().view(torch.uint8)), key

    state = DeepseekV41ForCausalLM(config).state_dict()
    dcp_load(state, storage_reader=HuggingFaceStorageReader(path=out_dir.as_posix()), no_dist=True)
    for key, tensor in state.items():
        torch.testing.assert_close(tensor, expected[key].to(tensor.dtype), rtol=0, atol=0, msg=key)


def test_mxfp4_experts_match_dequantized_bf16_and_train_only_activations():
    from arctic_platform.model.implementations.deepseek_v41.mxfp4_experts import Mxfp4GroupedExperts
    from arctic_platform.model.implementations.deepseek_v41.mxfp4_experts import dequantize_mxfp4
    from arctic_platform.model.implementations.moe.layers.moe import _run_experts_for_loop_impl

    torch.manual_seed(0)
    experts = Mxfp4GroupedExperts(dim=64, hidden_dim=32, num_experts=2, swiglu_limit=10.0)
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            weight = getattr(experts, name)
            weight.copy_(torch.randint(-128, 128, weight.shape, dtype=torch.int8))
            getattr(experts, f"{name}_scale_inv").copy_(2.0 ** torch.randint(-3, 2, weight.shape[:-1] + (1,)).float())
    x = torch.randn(5, 64, requires_grad=True)
    counts = torch.tensor([2, 3], dtype=torch.int64)
    output = experts(x, counts)
    restored = [dequantize_mxfp4(getattr(experts, n), getattr(experts, f"{n}_scale_inv")) for n in ("w1", "w2", "w3")]
    x_ref = x.detach().clone().requires_grad_(True)
    reference = _run_experts_for_loop_impl(*restored, x_ref.bfloat16(), counts, 10.0).float()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    output.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    assert all(param.grad is None for param in experts.parameters())
