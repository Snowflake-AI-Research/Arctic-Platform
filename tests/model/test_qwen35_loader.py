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

"""Qwen loader configuration and runtime ownership boundaries."""

import json
import sys

import pytest
from torch import nn

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import Patches
from arctic_platform.model import build_model
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async
from arctic_platform.testing_utils import torch_assert_equal


@pytest.mark.parametrize("composite", [False, True])
def test_selection_uses_text_config_not_checkpoint_name(tmp_path, composite):
    config = {"model_type": "qwen3_5_moe_text"}
    if composite:
        config = {
            "model_type": "qwen3_5_moe",
            "text_config": config,
        }
    (tmp_path / "config.json").write_text(json.dumps(config))
    spec = ModelSpec(model_path_or_name=str(tmp_path), parallelism=ParallelismConfig(expert_parallel=2))
    assert spec.loader == "qwen3_5_moe"


@pytest.mark.parametrize("backend", ["deepep", "uccl"])
def test_loader_preserves_options_and_process_groups(monkeypatch, backend):
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    seen = {}
    model = nn.Identity()

    def load(**kwargs):
        seen.update(kwargs)
        return model

    monkeypatch.setattr(qwen, "load_qwen3_5_moe_model", load)
    ep_group, sp_group = object(), object()
    spec = ModelSpec(
        model_path_or_name="local-checkpoint",
        loader="qwen3_5_moe",
        parallelism=ParallelismConfig(expert_parallel=2, sequence_parallel=2),
        loader_options={
            "ep_comm_backend": backend,
            "deepep_num_sms": 24,
            "tiled_mlp_token_chunk_size": 32,
            "weight_conversion_cache_dir": "",
            "trust_remote_code": True,
            "ac_config": {"offload_config": {"pin_memory_enabled": False, "pin_memory_max_size_gib": 0}},
        },
    )
    result = build_model(spec, parallel_groups={"ep_group": ep_group, "sp_group": sp_group})
    assert result.model is model
    assert seen["ep_group"] is ep_group and seen["sp_group"] is sp_group
    assert seen["ep_size"] == seen["sp_size"] == 2
    assert seen["optimization_dtype"] == "bfloat16"
    assert seen["attn_implementation"] == "flash_attention_3"
    assert seen["options"] == Qwen3_5MoeOptions.model_validate(spec.loader_options)
    assert seen["options"].tiled_mlp_token_chunk_size == 32
    assert seen["options"].ac_config.offload_config.pin_memory_enabled is False
    assert ModelSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize(
    "options",
    [
        {"ep_comm_backend": "unknown"},
        {"unsupported_option": True},
        {"debug": {"unknown": True}},
        {"tiled_mlp_token_chunk_size": 0},
        {"ac_config": {"freq": 0}},
        {"ac_config": {"offload_config": {"pin_memory_max_size_gib": -1}}},
        {"ac_config": {"mode": "selective", "offload_config": {"enabled": True}}},
        {"fused_cross_entropy": "liger", "fused_lm_head_token_chunk_size": 128},
    ],
)
def test_unsupported_options_are_rejected(options):
    with pytest.raises(ValueError):
        Qwen3_5MoeOptions(**options)


def test_liger_fused_cross_entropy_allows_fp32_lm_head():
    options = Qwen3_5MoeOptions(fused_cross_entropy="liger", fp32_lm_head=True)
    assert options.fused_cross_entropy == "liger"
    assert options.fp32_lm_head is True


def test_spec_cannot_silently_ignore_dtype():
    with pytest.raises(ValueError):
        ModelSpec(model_path_or_name="local", loader="qwen3_5_moe", dtype="float16")


def test_spec_accepts_peft():
    spec = ModelSpec(
        model_path_or_name="local",
        loader="qwen3_5_moe",
        patches=Patches(peft={"peft_type": "Lora"}),
    )
    assert spec.patches.peft == {"peft_type": "Lora"}


def test_runtime_groups_are_required():
    spec = ModelSpec(model_path_or_name="local", loader="qwen3_5_moe")
    with pytest.raises(ValueError, match="ep_group"):
        build_model(spec)


def test_sequence_parallel_group_is_required():
    spec = ModelSpec(
        model_path_or_name="local",
        loader="qwen3_5_moe",
        parallelism=ParallelismConfig(expert_parallel=2, sequence_parallel=2),
    )
    with pytest.raises(ValueError, match="sp_group"):
        build_model(spec, parallel_groups={"ep_group": object()})


@pytest.mark.parametrize("patches", [Patches(gradient_checkpointing=True), Patches(zorro_train={})])
def test_generic_forward_patches_are_rejected(patches):
    with pytest.raises(ValueError, match="generic forward patches"):
        ModelSpec(model_path_or_name="local", loader="qwen3_5_moe", patches=patches)


def test_activation_offload_defaults_to_disabled():
    options = Qwen3_5MoeOptions(ac_config={"offload_config": {}})
    assert options.ac_config is not None
    assert options.ac_config.offload_config.enabled is False
    assert options.ac_config.offload_config.tensor_size_threshold == 1 << 20


def test_runtime_config_is_derived_from_validated_options():
    from arctic_platform.model.implementations.qwen35.deepspeed_integration import _build_model_config

    options = Qwen3_5MoeOptions(
        seq_len=1024,
        ep_comm_backend="uccl",
        ac_config={
            "freq": 2,
            "offload_config": {
                "enabled": True,
                "keep_last_n": 3,
                "pin_memory_max_size_gib": 0,
            },
        },
        debug={
            "num_layers": 4,
            "random_init": True,
        },
    )
    config = _build_model_config(
        "local",
        4,
        2,
        "float32",
        "sdpa",
        options,
    )

    assert config.seq_len == 1024
    assert config.ep == 4
    assert config.dp_replicate == 2
    assert config.optimization_dtype == "float32"
    assert config.attn == "sdpa"
    assert config.ep_comm_backend == "uccl"
    assert config.ac is options.ac_config
    assert config.ac.offload_config.enabled is True
    assert config.ac.offload_config.keep_last_n == 3
    assert config.debug.num_layers == 4
    assert config.debug.random_init is True


def test_legacy_qwen_types_are_shared():
    from arctic_platform.model.implementations.moe.layers.moe import MoE
    from arctic_platform.model.implementations.qwen35.models.layers.moe import MoE as LegacyMoE

    assert LegacyMoE is MoE


class TestStandaloneImports(TestCasePlus):
    def test_model_implementation_has_no_service_imports(self):
        code = """
import importlib.abc
import sys
class BlockService(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('dss', 'dss_client'):
            raise AssertionError(f'service import: {fullname}')
sys.meta_path.insert(0, BlockService())
from arctic_platform.model.implementations.qwen35 import deepspeed_integration
from arctic_platform.model.implementations import fp8
print('Standalone MoE and FP8 imports passed')
"""
        execute_subprocess_async([sys.executable, "-c", code], env=self.get_env(), timeout=60)


def _mtp_export_model(layer_count=1):
    from types import SimpleNamespace

    model = nn.Identity()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            mtp_num_hidden_layers=layer_count,
        )
    )
    return model


def _glm_prediction_model():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    model = nn.Identity()
    model.config = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=0,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_kpool=2,
        linear_num_heads=2,
        linear_head_dim=4,
        hc_mult=2,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["sparse"],
    )
    return model


def _glm_prediction_tensors(model):
    import torch

    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    names = auxiliary_weights._required_layer_names(model.config, "model.language_model.layers.1.", set())
    return {name: torch.ones(2) for name in names}


def test_prediction_weight_export_builders_are_family_scoped():
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as glm
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.implementations.qwen38 import deepspeed_integration as flash

    assert qwen._adapter().hf_weight_export is qwen._build_iter_full_hf_weights_with_mtp
    assert flash._adapter().hf_weight_export is flash._build_iter_full_hf_weights
    assert glm._adapter().hf_weight_export is glm._build_iter_full_hf_weights_with_auxiliary
    assert qwen._generic_adapter().hf_weight_export is qwen._build_iter_full_hf_weights_with_mtp


@pytest.mark.parametrize("family", ["qwen35", "glm53", "qwen38", "qwen38-split"])
def test_prediction_export_wrappers_release_tensors_before_next_gather(monkeypatch, family):
    import gc
    import weakref

    import torch

    from arctic_platform.model.implementations.glm53 import deepspeed_integration as glm
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.implementations.qwen38 import deepspeed_integration as flash

    model = _mtp_export_model()
    model.config.text_config.mtp_num_hidden_layers = 0
    model.config.text_config.split_ngram_parts = 3 if family == "qwen38-split" else 1
    references = []
    gathered = []
    name = "model.ple.ple_embedding.ngram_embedding.weight" if family == "qwen38-split" else "model.weight"

    def ordinary(**kwargs):
        for index in range(2):
            gc.collect()
            assert all(reference() is None for reference in references), "wrapper retained the previous gather"
            tensor = torch.ones(6, 2)
            references.append(weakref.ref(tensor))
            gathered.append(index)
            yield name, tensor
            del tensor

    ordinary.weight_names = lambda **kwargs: frozenset({name})
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", lambda model: ordinary)
    if family == "glm53":
        exporter = glm._build_iter_full_hf_weights_with_auxiliary(model)
    else:
        exporter = (
            qwen._build_iter_full_hf_weights_with_mtp(model)
            if family == "qwen35"
            else flash._build_iter_full_hf_weights(model)
        )
    count = 0
    for _, tensor in exporter(include_skipped=family == "qwen38-split"):
        assert tensor.shape[1] == 2
        count += 1
        del tensor
    assert gathered == [0, 1]
    assert count == (6 if family == "qwen38-split" else 2)
    gc.collect()
    assert all(reference() is None for reference in references)


def test_preserved_sidecar_releases_tensor_before_reading_next(monkeypatch):
    import gc
    import weakref
    from types import SimpleNamespace

    import torch

    from arctic_platform.model.implementations.qwen35 import mtp_weights

    references = []

    def weights():
        for index in range(2):
            gc.collect()
            assert all(reference() is None for reference in references), "sidecar retained the previous tensor"
            tensor = torch.ones(4)
            references.append(weakref.ref(tensor))
            yield f"mtp.{index}", tensor
            del tensor

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    preserved = SimpleNamespace(names=frozenset({"mtp.0", "mtp.1"}), weights=weights())
    iterator = mtp_weights.finish_preserved_weight_export(preserved, set(), description="MTP")
    for _, tensor in iterator:
        del tensor
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("family", ["generic", "qwen35", "qwen38"])
def test_configured_mtp_is_exported_by_each_qwen_adapter(tmp_path, monkeypatch, family):
    import torch
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.implementations.qwen35 import mtp_weights
    from arctic_platform.model.implementations.qwen38 import deepspeed_integration as flash

    model = _mtp_export_model()
    if family != "qwen35":
        model.config.text_config.model_type = "qwen4_exp_text"
    payload = {
        name: torch.tensor([index], dtype=torch.bfloat16)
        for index, name in enumerate(sorted(mtp_weights._configured_names(model)))
    }
    save_file(payload, tmp_path / "model.safetensors")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    mtp_weights.capture_mtp_weights(model, tmp_path)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def ordinary(**kwargs):
        yield "model.weight", torch.ones(1)

    ordinary.weight_names = lambda **kwargs: frozenset({"model.weight"})
    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", lambda model: ordinary)
    adapter = (
        qwen._generic_adapter() if family == "generic" else qwen._adapter() if family == "qwen35" else flash._adapter()
    )
    exporter = adapter.hf_weight_export(model)
    assert exporter.weight_names(include_skipped=True) == {"model.weight", *payload}
    assert exporter.weight_names(include_skipped=False) == {"model.weight"}
    exported = dict(exporter(include_skipped=True))
    assert set(exported) == {"model.weight", *payload}
    for name, tensor in payload.items():
        torch_assert_equal(exported[name], tensor)


def test_qwen_hf_export_without_include_skipped_never_reads_mtp_sidecar(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    model = _mtp_export_model()
    ordinary_tensor = torch.tensor([1.0])
    calls = []

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            calls.append(("ordinary", include_skipped))
            yield "model.weight", ordinary_tensor

        iterator.weight_names = lambda **kwargs: frozenset({"model.weight"})
        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(
        qwen,
        "iter_preserved_mtp_weights",
        lambda model: (_ for _ in ()).throw(AssertionError("sidecar must not be read")),
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda: (_ for _ in ()).throw(AssertionError("rank is irrelevant without include_skipped")),
    )

    exported = list(qwen._build_iter_full_hf_weights_with_mtp(model)())

    assert exported == [("model.weight", ordinary_tensor)]
    assert calls == [("ordinary", False)]


def test_qwen_full_hf_export_validates_mtp_first_and_appends_after_ordinary_stream(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model = _mtp_export_model()
    ordinary_tensor = torch.tensor([1.0])
    mtp_tensor = torch.tensor([2.0], dtype=torch.bfloat16)
    events = []

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            events.append(("ordinary-start", include_skipped))
            yield "model.weight", ordinary_tensor
            events.append(("ordinary-exhausted",))

        iterator.weight_names = lambda **kwargs: frozenset({"model.weight"})
        return iterator

    def iter_sidecar(built_model):
        assert built_model is model
        events.append(("sidecar-start",))
        yield "mtp.fc.weight", mtp_tensor

    def validate_sidecar(built_model):
        assert built_model is model
        events.append(("sidecar-validated",))

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(qwen, "iter_preserved_mtp_weights", iter_sidecar)
    monkeypatch.setattr(mtp_weights, "validate_preserved_mtp_weights", validate_sidecar)
    monkeypatch.setattr(mtp_weights, "preserved_mtp_names", lambda built_model: ("mtp.fc.weight",))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    exported = list(qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True))

    assert exported == [("model.weight", ordinary_tensor), ("mtp.fc.weight", mtp_tensor)]
    assert events == [
        ("sidecar-validated",),
        ("ordinary-start", True),
        ("ordinary-exhausted",),
        ("sidecar-start",),
    ]


def test_qwen_full_hf_export_nonmaster_finishes_ordinary_stream_without_sidecar(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    model = _mtp_export_model()
    events = []

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            events.append(("ordinary-start", include_skipped))
            if False:
                yield
            events.append(("ordinary-exhausted",))

        iterator.weight_names = lambda **kwargs: frozenset({"model.weight"})
        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(
        qwen,
        "iter_preserved_mtp_weights",
        lambda model: (_ for _ in ()).throw(AssertionError("nonmaster must not hold or export payload")),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    assert list(qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)) == []
    assert events == [("ordinary-start", True), ("ordinary-exhausted",)]


def test_qwen_full_hf_export_rejects_live_name_collision(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model = _mtp_export_model()
    live = torch.tensor([1.0])
    preserved = torch.tensor([2.0])

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            assert include_skipped is True
            yield "mtp.fc.weight", live

        iterator.weight_names = lambda **kwargs: frozenset({"mtp.fc.weight"})
        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    sidecar_reads = []

    def sidecar(built_model):
        sidecar_reads.append(built_model)
        yield "mtp.fc.weight", preserved

    monkeypatch.setattr(qwen, "iter_preserved_mtp_weights", sidecar)
    monkeypatch.setattr(mtp_weights, "validate_preserved_mtp_weights", lambda built_model: None)
    monkeypatch.setattr(mtp_weights, "preserved_mtp_names", lambda built_model: ("mtp.fc.weight",))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    iterator = qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)
    with pytest.raises(RuntimeError, match="collides with live HF export"):
        next(iterator)
    assert sidecar_reads == []


@pytest.mark.parametrize("family", ["qwen", "glm"])
@pytest.mark.parametrize("replacement", [False, True])
def test_preserved_prediction_export_rechecks_shards_between_tensors(tmp_path, monkeypatch, family, replacement):
    import os
    import time

    import torch
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.glm53 import auxiliary_weights as glm
    from arctic_platform.model.implementations.qwen35 import mtp_weights as qwen

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    if family == "glm":
        model = _glm_prediction_model()
        tensors = _glm_prediction_tensors(model)
        capture, iterator = glm.capture_auxiliary_weights, glm.iter_preserved_auxiliary_weights
    else:
        model = _mtp_export_model()
        tensors = {name: torch.ones(2) for name in qwen._configured_names(model)}
        capture, iterator = qwen.capture_mtp_weights, qwen.iter_preserved_mtp_weights
    shard = tmp_path / "model.safetensors"
    save_file(tensors, shard)
    original_metadata = shard.stat()
    capture(model, tmp_path)
    export = iterator(model)
    next(export)
    if replacement:
        substitute = tmp_path / "replacement.safetensors"
        save_file({name: tensor + 1 for name, tensor in tensors.items()}, substitute)
        substitute.replace(shard)
    else:
        save_file({name: tensor + 1 for name, tensor in tensors.items()}, shard)
        # An immediate rewrite can share the original filesystem timestamp tick.
        for _ in range(100):
            os.utime(shard, ns=(original_metadata.st_atime_ns, original_metadata.st_mtime_ns))
            if shard.stat().st_ctime_ns != original_metadata.st_ctime_ns:
                break
            time.sleep(0.01)
        assert shard.stat().st_mtime_ns == original_metadata.st_mtime_ns
        assert shard.stat().st_ctime_ns != original_metadata.st_ctime_ns
    with pytest.raises(RuntimeError, match="shard changed"):
        next(export)


def test_glm_prediction_capture_rejects_nonregular_shard_without_blocking(tmp_path, monkeypatch):
    import os

    import torch

    from arctic_platform.model.implementations.glm53 import auxiliary_weights as glm

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    model = _glm_prediction_model()
    names = _glm_prediction_tensors(model)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=dict.fromkeys(names, "fifo.safetensors")))
    )
    os.mkfifo(tmp_path / "fifo.safetensors")
    glm.capture_auxiliary_weights(model, tmp_path)
    with pytest.raises(RuntimeError, match="not a regular file"):
        glm.validate_preserved_auxiliary_weights(model)


@pytest.mark.parametrize("family", ["qwen", "glm"])
def test_prediction_export_rechecks_source_after_tensor_read(tmp_path, monkeypatch, family):
    import torch
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.glm53 import auxiliary_weights as glm
    from arctic_platform.model.implementations.moe import preserved_weights
    from arctic_platform.model.implementations.qwen35 import mtp_weights as qwen

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    if family == "glm":
        model = _glm_prediction_model()
        tensors = _glm_prediction_tensors(model)
        capture, iterator = glm.capture_auxiliary_weights, glm.iter_preserved_auxiliary_weights
    else:
        model = _mtp_export_model()
        tensors = {name: torch.ones(2) for name in qwen._configured_names(model)}
        capture, iterator = qwen.capture_mtp_weights, qwen.iter_preserved_mtp_weights
    shard = tmp_path / "model.safetensors"
    save_file(tensors, shard)
    capture(model, tmp_path)
    replacement = tmp_path / "replacement.safetensors"
    save_file({name: torch.zeros_like(tensor) for name, tensor in tensors.items()}, replacement)
    native = preserved_weights.safe_open

    class Reader:
        def __init__(self, *args, **kwargs):
            self.context = native(*args, **kwargs)

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_slice(self, name):
            return self.reader.get_slice(name)

        def get_tensor(self, name):
            tensor = self.reader.get_tensor(name)
            replacement.replace(shard)
            return tensor

    monkeypatch.setattr(preserved_weights, "safe_open", Reader)
    with pytest.raises(RuntimeError, match="shard changed"):
        next(iterator(model))


def test_glm_prediction_capture_checks_indexed_inventory(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.glm53 import auxiliary_weights

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    model = _glm_prediction_model()
    prefix = "model.language_model.layers.1."
    tensors = _glm_prediction_tensors(model)
    save_file({**tensors, prefix + "unindexed.weight": torch.ones(2)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=dict.fromkeys(tensors, "model.safetensors")))
    )
    auxiliary_weights.capture_auxiliary_weights(model, tmp_path)
    with pytest.raises(RuntimeError, match="does not match checkpoint index"):
        auxiliary_weights.validate_preserved_auxiliary_weights(model)


@pytest.mark.parametrize("family,fault", [("qwen", "collision"), ("glm", "collision"), ("glm", "read")])
def test_preserved_export_completion_failure_reaches_both_ranks(monkeypatch, family, fault):
    import torch

    from arctic_platform.model.implementations.glm53 import deepspeed_integration as glm
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    rank = [0]
    broadcasts = []
    sidecar_reads = []
    name = "inactive.weight"
    live_name = name if fault == "collision" else "active.weight"
    value = torch.ones(1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank[0])

    def broadcast(message, src):
        if rank[0] == 0:
            broadcasts.append(message[0])
        else:
            message[0] = broadcasts.pop(0)

    def sidecar(model):
        sidecar_reads.append(rank[0])
        if fault == "read":
            raise OSError("source disappeared during export")
        yield name, value

    def ordinary(**kwargs):
        return iter([(live_name, value)] if rank[0] == 0 else [])

    ordinary.weight_names = lambda **kwargs: frozenset({live_name})
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", lambda model: ordinary)
    if family == "qwen":
        monkeypatch.setattr(qwen, "iter_preserved_mtp_weights", sidecar)
        monkeypatch.setattr(mtp_weights, "validate_preserved_mtp_weights", lambda model: None)
        monkeypatch.setattr(mtp_weights, "preserved_mtp_names", lambda model: (name,))
        exporter = qwen._build_iter_full_hf_weights_with_mtp
    else:
        monkeypatch.setattr(glm, "iter_preserved_auxiliary_weights", sidecar)
        monkeypatch.setattr(glm, "validate_preserved_auxiliary_weights", lambda model: None)
        monkeypatch.setattr(glm, "preserved_auxiliary_names", lambda model: (name,))
        exporter = glm._build_iter_full_hf_weights_with_auxiliary
    model = _mtp_export_model()
    for current in (0, 1):
        rank[0] = current
        with pytest.raises(RuntimeError, match="collides|source disappeared"):
            list(exporter(model)(include_skipped=True))
    assert broadcasts == []
    assert sidecar_reads == ([] if fault == "collision" else [0])


def test_qwen_full_hf_export_requires_capture_for_configured_mtp(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    model = _mtp_export_model()
    ordinary_tensor = torch.tensor([1.0])
    ordinary_calls = []

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            assert include_skipped is True
            ordinary_calls.append(include_skipped)
            yield "model.weight", ordinary_tensor

        iterator.weight_names = lambda **kwargs: frozenset({"model.weight"})
        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    iterator = qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)
    with pytest.raises(RuntimeError, match="requires captured original MTP weights"):
        next(iterator)
    assert ordinary_calls == []


def test_qwen_full_hf_export_without_configured_mtp_needs_no_capture(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    model = _mtp_export_model(layer_count=0)
    ordinary_tensor = torch.tensor([1.0])

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            assert include_skipped is True
            yield "model.weight", ordinary_tensor

        iterator.weight_names = lambda **kwargs: frozenset({"model.weight"})
        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    assert list(qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)) == [
        ("model.weight", ordinary_tensor)
    ]
