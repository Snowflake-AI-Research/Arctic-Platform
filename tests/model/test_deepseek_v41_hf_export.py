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
"""DeepSeek-V4.1 weights-only HF export: a published checkpoint -> the PrimeRL module tree -> the export's per-layer
groups (``convert_layer_to_hf``) -> the published names again, bit for bit.

The checkpoint is the full fine-tuning one (dense and routed experts in BF16, the Engram table FP8 with UE8M0 scales),
or the LoRA one (packed MXFP4 experts). The 4-rank test (gloo over a file store, CPU) runs the export as an EP job
does: experts sharded over EP under a static placement, the Engram table row-sharded by DTensor with a short last
shard, through ``build_iter_full_hf_weights``.
"""

import copy
import io
import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config
from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM

EP = 4
E = 8
ENGRAM_ROWS = 30  # 4 EP ranks hold 8, 8, 8 and 6 rows
# perm[slot] = logical expert, for every MoE layer.
PERMS = {0: [5, 2, 7, 0, 6, 1, 3, 4], 2: [1, 6, 0, 7, 3, 2, 4, 5], 3: [7, 6, 5, 4, 3, 2, 1, 0]}


def _config(**overrides) -> DeepseekV41Config:
    config = dict(
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
        n_routed_experts=E,
        n_shared_experts=1,
        hc_mult=2,
        # Layer 1 is a ratio-2 KV and index source, 2 a ratio-1 one, 3 an index source reusing 2's keys.
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
        # A vision tower in the config gives every MoE layer its image-token routing bias, as in the published model.
        vision_n_layers=1,
        image_token_id=1,
        dspark_noise_token_id=1,
        pad_token_id=0,
        swiglu_limit=10.0,
    )
    config.update(overrides)
    return DeepseekV41Config(**config)


def _published_keys(config: DeepseekV41Config, packed: bool) -> set[str]:
    """The published checkpoint's text keys, written out from DeepSeek's naming (not from the conversion chain)."""
    keys = {"embed.weight", "norm.weight", "head.weight"}
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        keys |= {
            f"{p}.attn_norm.weight",
            f"{p}.ffn_norm.weight",
            *(f"{p}.hc_{sub}_{part}" for sub in ("attn", "ffn") for part in ("fn", "base", "scale")),
            *(f"{p}.attn.{name}.weight" for name in ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_a", "wo_b")),
            f"{p}.attn.attn_sink",
            f"{p}.ffn.gate.weight",
            f"{p}.ffn.gate.bias",
            f"{p}.ffn.gate.bias_vl",
            *(f"{p}.ffn.shared_experts.{w}.weight" for w in ("w1", "w2", "w3")),
            *(f"{p}.ffn.experts.{e}.{w}.weight" for e in range(E) for w in ("w1", "w2", "w3")),
        }
        if packed:
            keys |= {f"{p}.ffn.experts.{e}.{w}.scale" for e in range(E) for w in ("w1", "w2", "w3")}
        if i in config.kv_source_layer_ids:
            keys |= {f"{p}.attn.compressor.wkv.weight", f"{p}.attn.compressor.norm.weight"}
            if config.compress_ratios[i] > 1:
                keys.add(f"{p}.attn.compressor.wgate.weight")
        if i in config.index_source_layer_ids:
            keys |= {f"{p}.attn.indexer.wq_b.weight", f"{p}.attn.indexer.weights_proj.weight"}
            if i in config.kv_source_layer_ids:
                keys |= {f"{p}.attn.indexer.wk.weight", f"{p}.attn.indexer.k_norm.weight"}
        if i in config.engram_layer_ids:
            keys |= {
                f"{p}.engram.embed.weight",
                f"{p}.engram.embed.scale",
                f"{p}.engram.wkv.weight",
                f"{p}.engram.q_weight",
                f"{p}.engram.k_weight",
            }
    return keys


def _random_like(name: str, tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if tensor.dtype == torch.float8_e8m0fnu:  # powers of two around 1
        codes = torch.randint(120, 135, tensor.shape, generator=generator, dtype=torch.uint8)
        return codes.view(torch.float8_e8m0fnu)
    if tensor.dtype == torch.float8_e4m3fn:
        return torch.randn(tensor.shape, generator=generator).to(torch.float8_e4m3fn)
    if tensor.dtype == torch.int8:  # packed MXFP4 nibbles
        return torch.randint(-128, 128, tensor.shape, generator=generator, dtype=torch.int8)
    dtype = torch.float32 if tensor.dtype == torch.float32 else torch.bfloat16
    return torch.randn(tensor.shape, generator=generator).to(dtype)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8) if tensor.element_size() == 1 else tensor


def _assert_same(got: dict, want: dict) -> None:
    assert sorted(got) == sorted(want), (sorted(set(got) - set(want)), sorted(set(want) - set(got)))
    for name in want:
        assert got[name].dtype == want[name].dtype, (name, got[name].dtype, want[name].dtype)
        assert got[name].shape == want[name].shape, (name, got[name].shape, want[name].shape)
        assert torch.equal(_bytes(got[name]), _bytes(want[name])), name


def _published_checkpoint(config: DeepseekV41Config, packed: bool) -> dict[str, torch.Tensor]:
    """A random published checkpoint for ``config``: names from DeepSeek's naming, shapes and dtypes from the model.

    The full fine-tuning snapshot is BF16 except the FP32 tensors the model keeps FP32 and the FP8 Engram table; a
    LoRA snapshot keeps the experts packed (int8 nibbles + UE8M0 scales)."""
    model = DeepseekV41ForCausalLM(config)
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors

    shapes = {name: tensor.detach() for name, tensor in named_weight_sync_tensors(model)}
    template = model.convert_to_hf({name: t for name, t in shapes.items()})
    if packed:
        for name in list(template):
            if ".ffn.experts." in name and name.endswith("_scale_inv"):
                raise AssertionError(name)
            if ".ffn.experts." in name and name.endswith(".scale"):
                template[name] = template[name].to(torch.float8_e8m0fnu)
    assert set(template) == _published_keys(config, packed), sorted(set(template) ^ _published_keys(config, packed))
    generator = torch.Generator().manual_seed(7)
    return {name: _random_like(name, tensor, generator) for name, tensor in sorted(template.items())}


def _export_groups(model: DeepseekV41ForCausalLM, include_buffers: bool = True) -> dict[str, torch.Tensor]:
    """The single-rank export: ``build_iter_full_hf_weights``'s grouping, without the EP gathers."""
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors
    from arctic_platform.model.implementations.moe.weights import hf_export_param_name

    groups: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in named_weight_sync_tensors(model, include_export_buffers=include_buffers):
        hf_name = hf_export_param_name(name)
        parts = hf_name.split(".")
        index = int(parts[2]) if parts[:2] == ["model", "layers"] else -1
        groups.setdefault(index, {})[hf_name] = tensor.detach().clone()
    out: dict[str, torch.Tensor] = {}
    for index, state in sorted(groups.items()):
        out.update(model.convert_layer_to_hf(state, index))
    return out


@pytest.mark.parametrize("packed", [False, True], ids=["full-ft-bf16", "lora-packed-experts"])
def test_published_checkpoint_round_trips_through_the_layer_export(packed):
    overrides = {}
    if packed:
        overrides["quantization_config"] = {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "expert_dtype": "fp4",
        }
        overrides.update(hidden_size=64, moe_intermediate_size=32)  # MXFP4 scales 32-column blocks
    config = _config(**overrides)
    hf = _published_checkpoint(config, packed)
    torch.manual_seed(0)
    model = DeepseekV41ForCausalLM(config)
    missing, unexpected = model.load_state_dict(model.convert_to_prime(copy.deepcopy(hf)), strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    _assert_same(_export_groups(model), hf)


def test_weight_sync_groups_leave_the_routing_bias_out():
    config = _config()
    torch.manual_seed(0)
    model = DeepseekV41ForCausalLM(config)
    synced = _export_groups(model, include_buffers=False)
    assert not any(name.endswith(".bias_vl") for name in synced)
    assert all(f"layers.{i}.ffn.gate.bias" in synced for i in range(config.num_hidden_layers))
    assert "layers.1.ffn.gate.bias_vl" in _export_groups(model)


# --- 4 ranks (gloo) --------------------------------------------------------------------------------------------------


def _entry(rank, worker, store, args, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group("gloo", store=dist.FileStore(store, EP), rank=rank, world_size=EP)
    try:
        results[rank] = worker(rank, *args)
    finally:
        dist.destroy_process_group()


def _run(worker, tmp_path, *args):
    results = mp.Manager().dict()
    mp.spawn(_entry, args=(worker, str(tmp_path / "store"), args, results), nprocs=EP, join=True)
    return dict(results)


def _shard_rows_on_ep(module, rank):
    """What ``DeepEPShardParallel`` leaves after the DCP load: this rank's ``ep_row_shard`` rows of every direct
    parameter. (Distributing the loaded table instead would pad its short shard with zero UE8M0 scales, which do not
    exist; the loader shards the meta-device model before the load.)"""
    from torch import nn

    from arctic_platform.model.implementations.deepseek_v41.engram import ep_row_shard

    for name, parameter in list(module.named_parameters(recurse=False)):
        start, end = ep_row_shard(parameter.shape[0], EP, rank)
        local = nn.Parameter(parameter.data[start:end].clone(), requires_grad=parameter.requires_grad)
        if getattr(parameter, "_dss_skip_weight_sync", False):
            local._dss_skip_weight_sync = True
        module.register_parameter(name, local)
    module._ep_group, module._ep_rank, module._ep_world_size = dist.group.WORLD, rank, EP
    module._dss_ep_sharded = True


def _ep_export_worker(rank, stream, placed):
    import deepspeed.utils.groups as ds_groups
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    from arctic_platform.model.implementations.moe import deepspeed_integration
    from arctic_platform.model.implementations.moe import expert_placement as placement
    from arctic_platform.model.implementations.moe.distributed.expert_parallel import DeepEPExpertParallel

    ds_groups._get_expert_parallel_group = lambda name: dist.group.WORLD
    if stream:
        deepspeed_integration._EXPORT_GATHER_BYTES = 64  # streamed: shards a row, EP weights 64 / ep bytes at a time
    config = _config()
    hf = _published_checkpoint(config, packed=False)
    torch.manual_seed(0)
    model = DeepseekV41ForCausalLM(config)
    model.load_state_dict(model.convert_to_prime(copy.deepcopy(hf)), strict=True)

    # The loader's order (moe/deepspeed_integration.py): EP sharding, DTensors to local tensors, the placement, the
    # DeepSpeed tags.
    mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("ep",))
    for layer in model.model.layers:
        parallelize_module(layer.mlp.experts, device_mesh=mesh, parallelize_plan=DeepEPExpertParallel())
    deepspeed_integration.convert_dtensors_to_local(model)
    for module in model.modules():
        if getattr(module, "_dss_shard_on_ep", False):
            _shard_rows_on_ep(module, rank)
    table = model.model.layers[0].engram.embed
    assert table.weight.shape[0] == (6 if rank == EP - 1 else 8)
    if placed:
        os.environ[placement.PLACEMENT_ENV] = json.dumps(
            {"ep_size": EP, "layers": {str(k): v for k, v in PERMS.items()}}
        )
        assert placement.apply_expert_placement(model) == len(PERMS)
        held = model.model.layers[0].mlp.experts.w1.data
        want = model.convert_to_prime(copy.deepcopy(hf))["model.layers.0.mlp.experts.w1"]
        assert torch.equal(held, want[torch.tensor(PERMS[0][rank * 2 : rank * 2 + 2])])
    else:
        os.environ.pop(placement.PLACEMENT_ENV, None)
    deepspeed_integration.tag_expert_params_for_deepspeed(model, "ep_size_4")

    exported = dict(deepspeed_integration.build_iter_full_hf_weights(model)(include_skipped=True))
    if rank != 0:
        assert not exported
        return True
    _assert_same(exported, hf)
    return True


@pytest.mark.parametrize("placed", [False, True], ids=["contiguous", "placed"])
@pytest.mark.parametrize("stream", [False, True], ids=["device-gather", "host-stream"])
def test_ep_export_restores_the_published_checkpoint(tmp_path, stream, placed):
    assert all(_run(_ep_export_worker, tmp_path, stream, placed).values())


def test_hf_export_config_follows_the_models_own_quantization():
    from arctic_platform.model import hf_export_config

    quantization = {"quant_method": "fp8", "weight_block_size": [32, 32], "expert_dtype": "fp4"}
    source = {"model_type": "deepseek_v41", "quantization_config": quantization, "text_config": {}}
    source["text_config"]["quantization_config"] = quantization
    # Full FT dequantizes at load (the builder deletes the config's quantization), so the export is BF16.
    full_ft = DeepseekV41ForCausalLM(_config())
    assert hf_export_config(full_ft, source) == {"model_type": "deepseek_v41", "text_config": {}}
    assert source["quantization_config"] == quantization  # not modified
    # Packed experts frozen under LoRA keep the published MXFP4 layout, so the export keeps the source's config.
    packed = DeepseekV41ForCausalLM(
        _config(quantization_config=quantization, hidden_size=64, moe_intermediate_size=32)
    )
    assert hf_export_config(packed, source) == source


def test_hf_export_copies_the_vision_tower_the_text_model_drops(tmp_path):
    from safetensors.torch import save_file

    from arctic_platform.model import iter_hf_source_tensors
    from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import (
        DEFERRED_CHECKPOINT_NAMESPACES,
    )
    from arctic_platform.model.implementations.deepseek_v41.converting_deepseek_v41 import EXPORT_SOURCE_NAMESPACES

    assert set(EXPORT_SOURCE_NAMESPACES) == set(DEFERRED_CHECKPOINT_NAMESPACES) - {"mtp."}
    generator = torch.Generator().manual_seed(0)
    shards = {
        "model-00001-of-00002.safetensors": {
            "vision.blocks.0.attn.qkv.weight": torch.randn(6, 4, generator=generator).bfloat16(),
            "layers.0.attn_norm.weight": torch.ones(4).bfloat16(),
            "mtp.0.attn_norm.weight": torch.ones(4).bfloat16(),
        },
        "model-00002-of-00002.safetensors": {
            "aligner.fc1.weight": torch.randn(4, 4, generator=generator).bfloat16(),
            "image_newline": torch.randn(4, generator=generator).bfloat16(),
        },
    }
    weight_map = {}
    for file, tensors in shards.items():
        save_file(tensors, str(tmp_path / file))
        weight_map.update(dict.fromkeys(tensors, file))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    model = DeepseekV41ForCausalLM(_config())

    got = dict(iter_hf_source_tensors(model, str(tmp_path)))
    assert sorted(got) == ["aligner.fc1.weight", "image_newline", "vision.blocks.0.attn.qkv.weight"]
    for name, tensor in got.items():
        file = weight_map[name]
        assert tensor.dtype == torch.bfloat16 and torch.equal(tensor, shards[file][name]), name
    assert sorted(dict(iter_hf_source_tensors(model, str(tmp_path), skip={"image_newline"}))) == [
        "aligner.fc1.weight",
        "vision.blocks.0.attn.qkv.weight",
    ]
    assert list(iter_hf_source_tensors(nn.Linear(1, 1), str(tmp_path))) == []  # families without the attribute


def _as_trained(model) -> None:
    """The dtypes a LoRA job trains the model in: BF16, with the norms, sinks, mHC and gate biases kept FP32 (and the
    FP8 Engram table), so a layer's frozen weights mix BF16 and FP32."""
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point() or parameter.element_size() == 1:
            continue
        keep = any(part in name for part in ("norm", "attn_sink", ".hc_", "gate.bias"))
        parameter.data = parameter.data.to(torch.float32 if keep else torch.bfloat16)


def _frozen_ep_model(rank: int, config, hf, shard: bool):
    """The published checkpoint loaded as a LoRA job's frozen DeepSeek-V4.1 over the 4 EP ranks, its frozen
    replicated weights sharded (moe.shard_frozen_dense) when ``shard``: each layer as after its first call."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    from arctic_platform.model.implementations.moe import deepspeed_integration
    from arctic_platform.model.implementations.moe.distributed.expert_parallel import DeepEPExpertParallel
    from arctic_platform.model.implementations.moe.frozen_shard import install_frozen_layer_shards

    torch.manual_seed(0)
    model = DeepseekV41ForCausalLM(config)
    model.load_state_dict(model.convert_to_prime(copy.deepcopy(hf)), strict=True)
    _as_trained(model)
    mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("ep",))
    for layer in model.model.layers:
        parallelize_module(layer.mlp.experts, device_mesh=mesh, parallelize_plan=DeepEPExpertParallel())
    deepspeed_integration.convert_dtensors_to_local(model)
    for module in model.modules():
        if getattr(module, "_dss_shard_on_ep", False):
            _shard_rows_on_ep(module, rank)
    deepspeed_integration.tag_expert_params_for_deepspeed(model, "ep_size_4")
    model.requires_grad_(False)  # the base of a LoRA job
    for i, layer in enumerate(model.model.layers):  # as the loader's whole-block activation checkpointing wraps them
        model.model.layers[i] = checkpoint_wrapper(layer, early_stop=False)
    if shard:
        for held in install_frozen_layer_shards(model.model.layers, root=model):
            held.gather()
            held.release()
    return model


def _frozen_shard_export_worker(rank):
    """A LoRA job's frozen DeepSeek-V4.1 (moe.shard_frozen_dense): every layer's frozen replicated weights (BF16 and
    FP32) sharded over the 4 ranks, the experts and the Engram table expert-parallel. The weights-only export equals
    the unsharded model's under the published names, and the state dict inside whole_frozen_weights, saved and loaded
    as DeepSpeed's checkpoint does, is the unsharded model's."""
    import deepspeed.utils.groups as ds_groups

    from arctic_platform.model.implementations.moe import deepspeed_integration
    from arctic_platform.model.implementations.moe.frozen_held import whole_frozen_weights

    ds_groups._get_expert_parallel_group = lambda name: dist.group.WORLD
    config = _config()
    hf = _published_checkpoint(config, packed=False)
    unsharded = _frozen_ep_model(rank, config, hf, shard=False)
    model = _frozen_ep_model(rank, config, hf, shard=True)
    attn = model.model.layers[1]._checkpoint_wrapped_module.self_attn
    assert all(p.numel() == 0 for p in attn.parameters()), "the frozen attention weights are sharded"
    dtypes = {p.dtype for p in unsharded.model.layers[1]._checkpoint_wrapped_module.parameters()}
    assert {torch.bfloat16, torch.float32} <= dtypes, dtypes

    want = dict(deepspeed_integration.build_iter_full_hf_weights(unsharded)(include_skipped=True))
    exported = dict(deepspeed_integration.build_iter_full_hf_weights(model)(include_skipped=True))
    with whole_frozen_weights(model):
        # As DeepSpeed's checkpoint saves it: one torch.save of the module state and of the frozen parameters, which
        # refuses a storage viewed under several dtypes. Then loaded back.
        saved = io.BytesIO()
        torch.save({"module": model.state_dict(), "frozen": [p.detach().cpu() for p in model.parameters()]}, saved)
        saved.seek(0)
        state = torch.load(saved, weights_only=True)["module"]
    assert all(p.numel() == 0 for p in attn.parameters())
    want_state = unsharded.state_dict()
    assert list(state) == list(want_state)  # the checkpoint wrappers leave the state dict's names alone
    for name, tensor in want_state.items():
        assert state[name].dtype == tensor.dtype and torch.equal(_bytes(state[name]), _bytes(tensor)), name
    if rank != 0:
        assert not exported
        return True
    assert sorted(exported) == sorted(hf), "the published names"
    _assert_same(exported, want)
    return True


def test_ep_export_of_a_frozen_sharded_model_restores_the_published_checkpoint(tmp_path):
    assert all(_run(_frozen_shard_export_worker, tmp_path).values())
