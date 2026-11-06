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

"""Checkpoint export of Qwen3.8's EP-sharded frozen n-gram table: exact bytes, bounded per-rank memory (gloo, CPU)."""

import copy
import json
import os
import tempfile
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

pytest.importorskip("transformers.models.qwen4_exp")
pytest.importorskip("deepspeed")

_WORLD, _EP = 4, 2  # two EP groups, {0, 1} and {2, 3}, as with expert data parallelism
_NGRAM = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight"


class _PeakAllocatedBytes(TorchDispatchMode):
    """Peak bytes of storages allocated inside the mode and still alive: max_memory_allocated() for CPU tensors."""

    def __init__(self):
        super().__init__()
        self.live = self.peak = 0
        self.tracked: set[int] = set()

    def _release(self, key: int, nbytes: int) -> None:
        self.tracked.discard(key)
        self.live -= nbytes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        inputs = {id(t.untyped_storage()) for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)}
        for tensor in tree_leaves(out):
            if not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            key = id(storage)
            if key in inputs or key in self.tracked:
                continue
            self.tracked.add(key)
            self.live += storage.nbytes()
            self.peak = max(self.peak, self.live)
            weakref.finalize(storage, self._release, key, storage.nbytes())
        return out


def _config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    text_config = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1],
        ple_embed_dim=64,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=1,
        ngram_vocab_size_base=100_000,  # the table dwarfs every other tensor, as in Flash-Next (51.2B of 180B params)
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=2,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=4,
        indexer_compress_ratio=2,
        output_gate_type="sigmoid",
        eos_token_id=1,
        pad_token_id=0,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=10_000.0,
            partial_rotary_factor=0.5,
            mrope_section=[1, 1, 0],
            mrope_interleaved=True,
        ),
    )
    vision_config = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    config = Qwen4ExpConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    config.use_grouped_mm = False
    return config


def _worker(rank: int, rendezvous: str, out_dir: str) -> None:
    import deepspeed.utils.groups as ds_groups
    from torch.distributed.device_mesh import init_device_mesh

    from arctic_platform.model import iter_model_weights
    from arctic_platform.model.implementations.moe.deepspeed_integration import apply_ep_with_mesh
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.moe.deepspeed_integration import convert_dtensors_to_local
    from arctic_platform.model.implementations.moe.deepspeed_integration import tag_expert_params_for_deepspeed
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL
    from arctic_platform.model.weight_export import WeightExportContract
    from arctic_platform.model.weight_export import register_weight_export

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=_WORLD, timeout=timedelta(seconds=120)
    )
    torch.manual_seed(0)
    model = Qwen4ExpForConditionalGenerationPrimeRL(_config())
    with torch.no_grad():
        for parameter in model.parameters():  # DeepSpeed's bf16 cast
            parameter.data = parameter.data.to(torch.bfloat16)
    reference = copy.deepcopy(model)
    contract = WeightExportContract(hf=build_iter_full_hf_weights, vllm=None)
    register_weight_export(reference, contract)

    mesh = init_device_mesh("cpu", (_WORLD // _EP, _EP), mesh_dim_names=("edp", "ep"))
    apply_ep_with_mesh(model, SimpleNamespace(ep_comm_backend="deepep"), mesh["ep"])
    convert_dtensors_to_local(model)
    ds_groups._EXPERT_PARALLEL_GROUP[f"ep_size_{_EP}"] = mesh["ep"].get_group()
    tag_expert_params_for_deepspeed(model, f"ep_size_{_EP}")
    register_weight_export(model, contract)
    table = reference.get_parameter(_NGRAM)
    table_bytes = table.numel() * table.element_size()
    assert model.get_parameter(_NGRAM).shape[0] * _EP == table.shape[0]

    expected = dict(iter_model_weights(reference, "hf", is_master=True, is_zero3=False, include_skipped=True))
    exported_names = []
    table_ref = None
    # Consume as the streaming checkpointer does, releasing each tensor after writing it.
    with _PeakAllocatedBytes() as memory:
        for name, actual in iter_model_weights(model, "hf", is_master=rank == 0, is_zero3=False, include_skipped=True):
            exported_names.append(name)
            wanted = expected[name]
            assert actual.dtype == wanted.dtype and actual.shape == wanted.shape, name
            assert torch.equal(actual.view(torch.uint8), wanted.contiguous().view(torch.uint8)), name
            if name == _NGRAM:
                table_ref = weakref.ref(actual)
            elif table_ref is not None and ".layers.1." in name:
                assert table_ref() is None, "export iterator retained the previous layer's n-gram table"
    if rank == 0:
        assert exported_names == list(expected)
        assert _NGRAM in exported_names
    else:
        assert exported_names == []
    # Rank 0's exported table is the output; anything else is transient.
    transient = memory.peak - (table_bytes if rank == 0 else 0)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(dict(transient_tables=transient / table_bytes), f)
    dist.destroy_process_group()


def test_ngram_table_export_is_exact_and_rank_memory_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(f"file://{tmp}/rendezvous", tmp), nprocs=_WORLD, join=True)
        transient = []
        for rank in range(_WORLD):
            with open(os.path.join(tmp, f"rank{rank}.json")) as f:
                transient.append(json.load(f)["transient_tables"])
    report = ", ".join(f"rank {rank} {value:.2f}" for rank, value in enumerate(transient))
    # Each rank keeps one EP-sized piece; rank 0 additionally keeps the exported table.
    assert max(transient) <= 1.1 / (_EP * _EP), f"transient memory in tables: {report}"


def _checkpoint_roundtrip_worker(rank: int, rendezvous: str, directory: str, fault: str) -> None:
    import deepspeed
    import deepspeed.utils.groups as ds_groups
    from torch.distributed.device_mesh import init_device_mesh

    from arctic_platform.model import iter_model_weights
    from arctic_platform.model.implementations.moe.deepspeed_integration import apply_ep_with_mesh
    from arctic_platform.model.implementations.moe.deepspeed_integration import convert_dtensors_to_local
    from arctic_platform.model.implementations.moe.deepspeed_integration import tag_expert_params_for_deepspeed
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors
    from arctic_platform.model.implementations.moe.weights import hf_export_param_name
    from arctic_platform.model.implementations.moe.weights import load_state_dict
    from arctic_platform.model.implementations.qwen35.mtp_weights import _configured_names
    from arctic_platform.model.implementations.qwen35.mtp_weights import capture_mtp_weights
    from arctic_platform.model.implementations.qwen38.deepspeed_integration import _adapter
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL
    from arctic_platform.model.weight_export import WeightExportContract
    from arctic_platform.model.weight_export import register_weight_export
    from arctic_platform.model.weight_export import save_exported_state_dict

    set_seed(0)
    config = _config()
    config.text_config.ngram_vocab_size_base = 29
    config.text_config.split_ngram_parts = 3
    config.text_config.mtp_num_hidden_layers = 1
    model = Qwen4ExpForConditionalGenerationPrimeRL(config)
    # Bare routed-expert modules defer initialization to the loader; use finite exact checkpoint values.
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
            parameter.copy_(values.remainder(97).div(97) + index / 1000)
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    source_state = model.convert_to_hf(
        {
            hf_export_param_name(name): value.detach().clone()
            for name, value in named_weight_sync_tensors(model, include_export_buffers=True)
            if hf_export_param_name(name) is not None
        }
    )
    # Build canonical shards independently of the exporter under test, including unequal row counts.
    for name in list(source_state):
        if name.endswith(".ple.ple_embedding.ngram_embedding.weight"):
            table = source_state.pop(name)
            quotient, remainder = divmod(table.shape[0], 3)
            offset = 0
            for index in range(3):
                rows = quotient + (index < remainder)
                source_state[f"{name.removesuffix('.weight')}.shard_{index}.weight"] = table[
                    offset : offset + rows
                ].clone()
                offset += rows
    sidecar = {
        name: (
            (torch.arange(6, dtype=torch.float32).reshape(2, 3) + index).to(
                (torch.float32, torch.bfloat16, torch.float16)[index % 3]
            )
        )
        for index, name in enumerate(sorted(_configured_names(model)))
    }
    source_state.update(sidecar)
    source = Path(directory) / "source"
    destination = Path(directory) / "exported"
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    deepspeed.init_distributed(dist_backend="gloo", auto_mpi_discovery=False)
    try:
        if rank == 0:
            save_exported_state_dict(dict(source_state), str(source))
            config.save_pretrained(source)
        dist.barrier()
        capture_mtp_weights(model, source if rank == 0 else Path(directory) / "absent-on-peer")
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        apply_ep_with_mesh(model, SimpleNamespace(ep_comm_backend="deepep"), mesh)
        convert_dtensors_to_local(model)
        ds_groups._EXPERT_PARALLEL_GROUP["ep_size_2"] = mesh.get_group()
        tag_expert_params_for_deepspeed(model, "ep_size_2")
        contract = WeightExportContract(hf=_adapter().hf_weight_export)
        register_weight_export(model, contract)
        synced = dict(iter_model_weights(model, "hf", is_master=rank == 0, is_zero3=False))
        assert not set(synced) & set(sidecar)
        if rank == 0 and fault == "source-removed":
            for shard in source.glob("*.safetensors"):
                shard.unlink()
        dist.barrier()
        try:
            exported = dict(iter_model_weights(model, "hf", is_master=rank == 0, is_zero3=False, include_skipped=True))
        except RuntimeError as error:
            assert fault == "source-removed", str(error)
            message = str(error)
        else:
            assert fault == "none", "changed source was accepted"
            message = None
            if rank == 0:
                assert set(exported) == set(source_state), {
                    "missing": sorted(set(source_state) - set(exported)),
                    "extra": sorted(set(exported) - set(source_state)),
                }
                for name, wanted in source_state.items():
                    assert exported[name].shape == wanted.shape and exported[name].dtype == wanted.dtype, name
                    torch_assert_equal(exported[name], wanted, msg=name)
                save_exported_state_dict(dict(exported), str(destination))
                config.save_pretrained(destination)
            else:
                assert exported == {}
        statuses = [None, None]
        dist.all_gather_object(statuses, message)
        assert statuses[0] == statuses[1], statuses
        if fault != "none":
            assert message is not None
            return
        dist.barrier()
        # Reload canonical HF shards through the real family converter into a real tiny Qwen3.8 module.
        saved = load_state_dict(destination)
        restored = Qwen4ExpForConditionalGenerationPrimeRL(config)
        result = restored.load_state_dict(restored.convert_to_prime(dict(saved)), strict=False)
        assert result.missing_keys == []
        assert set(result.unexpected_keys) == set(sidecar)
        for name, wanted in original_state.items():
            torch_assert_equal(restored.state_dict()[name], wanted)
        capture_mtp_weights(restored, destination if rank == 0 else Path(directory) / "absent-on-peer")
        register_weight_export(restored, contract)
        reexported = dict(
            iter_model_weights(restored, "hf", is_master=rank == 0, is_zero3=False, include_skipped=True)
        )
        if rank == 0:
            assert set(reexported) == set(source_state)
            for name, wanted in source_state.items():
                torch_assert_equal(reexported[name], wanted)
        else:
            assert reexported == {}
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestQwen38CheckpointRoundtrip(TestCasePlus):
    def _run(self, fault):
        directory = self.get_auto_remove_tmp_dir()
        mp.spawn(
            _checkpoint_roundtrip_worker,
            args=(f"file://{directory}/rendezvous", str(directory), fault),
            nprocs=2,
            join=True,
        )

    def test_registered_exporter_preserves_shards_and_mtp_across_reload(self):
        self._run("none")

    def test_removed_source_fails_on_both_real_ranks(self):
        self._run("source-removed")
