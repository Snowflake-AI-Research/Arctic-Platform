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

import hashlib
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


def test_reduced_fixture_mtp_roundtrips_prime_rl_worker_options_to_qwen_runtime_config(tmp_path):
    from arctic_platform.debug.reduced_fixture_mtp import ReducedFixtureMTP
    from arctic_platform.model.implementations.qwen35.deepspeed_integration import _build_model_config

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe_text"}))
    files = [
        {"path": "config.json", "size_bytes": 1, "sha256": "1" * 64},
        {"path": "model.safetensors.index.json", "size_bytes": 2, "sha256": "2" * 64},
    ]
    descriptor = {
        "schema_version": 1,
        "config_id": "qwen35-reduced-fixture",
        "fixture_locator": "ap-correctness/qwen35-reduced",
        "reference_path": "/data-fast/references/qwen35-reduced",
        "reviewed_hash": "0123456789abcdef",
        "fixture_sha256": (
            hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ),
        "files": files,
        "source_checkpoint": "Qwen/Qwen3.5-Source",
        "num_hidden_layers": 2,
        "vision_depth": 2,
        "materialization": "pretrained-prefix-reduction",
    }
    worker_config = {
        "attn_implementation": "sdpa",
        "ep_size": 2,
        "prime_rl": {"debug": {"reduced_fixture_mtp": descriptor}},
    }

    spec = ModelSpec.from_ds_worker_config(str(tmp_path), worker_config)
    options = Qwen3_5MoeOptions.model_validate(spec.loader_options)
    runtime = _build_model_config(str(tmp_path), 2, 1, "bfloat16", "sdpa", options)

    assert spec.loader_options["debug"]["reduced_fixture_mtp"] == descriptor
    assert options.debug is not None
    assert options.debug.reduced_fixture_mtp == ReducedFixtureMTP.model_validate(descriptor)
    assert ReducedFixtureMTP.model_validate(runtime.debug.reduced_fixture_mtp).model_dump() == descriptor


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


def test_qwen_full_hf_export_appends_mtp_only_after_ordinary_stream_exhausts(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

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

        return iterator

    def iter_sidecar(built_model):
        assert built_model is model
        events.append(("sidecar-start",))
        yield "mtp.fc.weight", mtp_tensor

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(qwen, "iter_preserved_mtp_weights", iter_sidecar)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    exported = list(qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True))

    assert exported == [("model.weight", ordinary_tensor), ("mtp.fc.weight", mtp_tensor)]
    assert events == [("ordinary-start", True), ("ordinary-exhausted",), ("sidecar-start",)]


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

    model = _mtp_export_model()
    live = torch.tensor([1.0])
    preserved = torch.tensor([2.0])

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            assert include_skipped is True
            yield "mtp.fc.weight", live

        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(qwen, "iter_preserved_mtp_weights", lambda built_model: iter([("mtp.fc.weight", preserved)]))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    iterator = qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)
    assert next(iterator) == ("mtp.fc.weight", live)
    with pytest.raises(RuntimeError, match="collides with live HF export"):
        next(iterator)


def test_qwen_full_hf_export_requires_capture_for_configured_mtp(monkeypatch):
    import torch

    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    model = _mtp_export_model()
    ordinary_tensor = torch.tensor([1.0])

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            assert include_skipped is True
            yield "model.weight", ordinary_tensor

        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    iterator = qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)
    assert next(iterator) == ("model.weight", ordinary_tensor)
    with pytest.raises(RuntimeError, match="requires captured original MTP weights"):
        next(iterator)


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

        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    assert list(qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)) == [
        ("model.weight", ordinary_tensor)
    ]
