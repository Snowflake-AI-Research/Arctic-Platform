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

"""LoRA correctness controls for gradient and optimizer-step testing."""


from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import peft
import pytest
import safetensors
import torch
import transformers
from safetensors.torch import load_file
from safetensors.torch import save
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from arctic_platform.client import CortexConfig
from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.correctness.harness import lora as deterministic_lora
from arctic_platform.correctness.harness import lora as lora_init
from arctic_platform.correctness.harness import lora as lora_save0
from arctic_platform.correctness.harness import shared_lora_adapter as shared
from arctic_platform.correctness.harness import storage
from arctic_platform.correctness.harness import storage as workdir
from arctic_platform.correctness.harness.lora import canonical_name
from arctic_platform.correctness.harness.registry import TestOutcome
from arctic_platform.correctness.reference import process_determinism
from arctic_platform.correctness.reference.hf_single_gpu import build_engine
from arctic_platform.correctness.reference.qwen4_reference import load_qwen4_reference
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.model.weight_export import save_peft_adapters
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

from . import check_reference_records as _reference_selftests

_prepared_reference_runner_prepared_runner = _reference_selftests._prepared_reference_runner_prepared_runner


def test_gdn_determinism_reference_reduction_pinning_does_not_mutate_fla_autotuners() -> None:
    assert "pin_fla_gdn_autotuners" not in inspect.getsource(process_determinism)


_lora_init_A = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"


_lora_init_B = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"


def _lora_init_specs(dtype="bfloat16"):
    return [
        lora_init.LoraTensorSpec(_lora_init_A, (4, 16), dtype),
        lora_init.LoraTensorSpec(_lora_init_B, (12, 4), dtype),
    ]


def _lora_init_tensors(specs, seed=1234):
    return {spec.name: lora_init.generate_tensor(spec, seed) for spec in specs}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_lora_init_private_cpu_generation_matches_declared_kaiming_and_zero_distribution(
    dtype,
):
    a, b = _lora_init_specs(dtype)
    expected = torch.empty(a.shape, device="cpu", dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(lora_init.tensor_seed(1234, _lora_init_A))
    torch.nn.init.kaiming_uniform_(
        expected,
        a=math.sqrt(5),
        mode="fan_in",
        nonlinearity="leaky_relu",
        generator=generator,
    )
    actual = lora_init.generate_tensor(a, 1234)
    assert actual.device.type == "cpu"
    assert actual.dtype == getattr(torch, dtype)
    assert torch.equal(actual, expected.to(dtype=actual.dtype))
    assert float(actual.abs().max()) <= 1 / math.sqrt(a.shape[1])
    assert torch.count_nonzero(lora_init.generate_tensor(b, 1234)).item() == 0


def test_lora_init_unrelated_draws_and_traversal_order_do_not_change_values_or_global_rng():
    specs = _lora_init_specs()
    state = torch.random.get_rng_state().clone()
    first = _lora_init_tensors(specs)
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.rand(257)
    state = torch.random.get_rng_state().clone()
    second = _lora_init_tensors(list(reversed(specs)))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert all((torch.equal(first[name], second[name]) for name in first))
    assert lora_init.build_manifest(specs, 1234) == lora_init.build_manifest(reversed(specs), 1234)


def test_lora_init_sha256_seed_is_stable_and_changes_with_name_or_global_seed():
    recipe = {
        "domain": "arctic-correctness-lora-tensor-seed-v1",
        "global_seed": 1234,
        "name": _lora_init_A,
    }
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).digest()
    assert lora_init.tensor_seed(1234, _lora_init_A) == int.from_bytes(digest[:8], "big") & 2**63 - 1
    assert lora_init.tensor_seed(1234, _lora_init_A) != lora_init.tensor_seed(
        1234, _lora_init_A.replace("q_proj", "k_proj")
    )
    assert lora_init.tensor_seed(1234, _lora_init_A) != lora_init.tensor_seed(1235, _lora_init_A)


@pytest.mark.parametrize("seed", [-1, 2**63, True, "1234"])
def test_lora_init_invalid_global_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="global seed"):
        lora_init.generate_tensor(_lora_init_specs()[0], seed)


def test_lora_init_fresh_process_reconstruction_ignores_global_random_draws():
    expected = lora_init.build_manifest(_lora_init_specs(), 1234)
    code = (
        "import json, torch; from arctic_platform.correctness.harness.lora import LoraTensorSpec, build_manifest;"
        f" torch.manual_seed(98); torch.rand(299); specs=[LoraTensorSpec({_lora_init_A!r}, (4,16), 'bfloat16'),"
        f" LoraTensorSpec({_lora_init_B!r}, (12,4), 'bfloat16')];"
        " print(json.dumps(build_manifest(specs,1234),sort_keys=True))"
    )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == expected


def test_lora_init_canonical_names_retain_containers_and_strip_only_supported_segments():
    raw = _lora_init_A.replace(".layers.0.", ".layers.0._checkpoint_wrapped_module.").replace(
        ".lora_A.weight", ".lora_A.default.weight"
    )
    assert lora_init.canonical_name(raw) == _lora_init_A
    assert lora_init.canonical_name(_lora_init_A.replace("base_model.model.", "", 1)) != _lora_init_A
    assert lora_init.canonical_name(_lora_init_A.replace(".model.layers.", ".language_model.layers.")) != _lora_init_A


@pytest.mark.parametrize(
    "name",
    [
        "",
        "model..lora_A.weight",
        "model.lora_A.other.weight",
        "model.weight",
        "model x.lora_A.weight",
    ],
)
def test_lora_init_unsupported_or_ambiguous_names_are_rejected(name):
    with pytest.raises(ValueError):
        lora_init.canonical_name(name)


@pytest.mark.parametrize("shape", [(0, 16), (4,), (4, 16, 2), (True, 16), [4, 16]])
def test_lora_init_invalid_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="two-dimensional"):
        lora_init.LoraTensorSpec(_lora_init_A, shape, "float32")


def test_lora_init_unsupported_dtype_and_incomplete_or_colliding_pairs_are_rejected():
    with pytest.raises(ValueError, match="dtype"):
        lora_init.LoraTensorSpec(_lora_init_A, (4, 16), "float64")
    a, b = _lora_init_specs()
    for specs, message in [
        ([], "complete A/B"),
        ([a], "partner"),
        ([a, b, a], "duplicate"),
        (
            [a, lora_init.LoraTensorSpec(_lora_init_B, (12, 5), b.dtype)],
            "rank or dtype",
        ),
        (
            [a, lora_init.LoraTensorSpec(_lora_init_B, b.shape, "float32")],
            "rank or dtype",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            lora_init.build_manifest(specs, 1234)


def test_lora_init_manifest_contains_only_compact_json_recipe_and_exact_hashes():
    manifest = lora_init.build_manifest(_lora_init_specs(), 1234)
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["generation_dtype"] == "float32"
    assert manifest["generator"]["device"] == "cpu"
    assert manifest["generator"]["torch_version"] == str(torch.__version__)
    assert manifest["tensors"][0]["distribution"]["fan_in"] == 16
    assert all((len(entry["sha256"]) == 64 for entry in manifest["tensors"]))
    assert lora_init.manifest_specs(manifest) == _lora_init_specs()


@pytest.mark.parametrize(
    "field",
    [
        "recipe_version",
        "global_seed",
        "generation_dtype",
        "identity_sha256",
        "tensor_hash",
        "torch_version",
    ],
)
def test_lora_init_corrupt_or_incompatible_manifest_cannot_be_reused(field):
    manifest = lora_init.build_manifest(_lora_init_specs(), 1234)
    if field == "tensor_hash":
        manifest["tensors"][0]["sha256"] = "0" * 64
    elif field == "torch_version":
        manifest["generator"]["torch_version"] = "different-runtime"
    else:
        manifest[field] = "changed"
    with pytest.raises(ValueError):
        lora_init.manifest_specs(manifest)


def test_lora_init_hash_covers_name_shape_dtype_and_signed_zero_bytes():
    zeros = torch.zeros((4, 16), dtype=torch.float32)
    digest = lora_init.tensor_sha256(_lora_init_A, zeros)
    assert digest != lora_init.tensor_sha256(_lora_init_A.replace("q_proj", "k_proj"), zeros)
    assert digest != lora_init.tensor_sha256(_lora_init_A, zeros.reshape(8, 8))
    assert digest != lora_init.tensor_sha256(_lora_init_A, zeros.bfloat16())
    negative_zero = zeros.clone()
    negative_zero[0, 0] = -0.0
    assert torch.equal(zeros, negative_zero)
    assert digest != lora_init.tensor_sha256(_lora_init_A, negative_zero)


def test_lora_init_exact_verification_accepts_only_wrapper_default_aliases_and_full_global_shapes():
    specs = _lora_init_specs()
    manifest = lora_init.build_manifest(specs, 1234)
    tensors = _lora_init_tensors(specs)
    raw = {name.replace(".weight", ".default.weight"): tensor for name, tensor in tensors.items()}
    receipt = lora_init.verify_tensors(raw, manifest)
    assert receipt["identity_sha256"] == manifest["identity_sha256"]
    shards = list(tensors[_lora_init_A].chunk(2, dim=1))
    reconstructed = {**tensors, _lora_init_A: torch.cat(shards, dim=1)}
    lora_init.verify_tensors(reconstructed, manifest)
    with pytest.raises(ValueError, match="shape differs"):
        lora_init.verify_tensors({**tensors, _lora_init_A: shards[0]}, manifest)
    with pytest.raises(ValueError, match="coverage"):
        lora_init.verify_tensors(
            {name.removeprefix("base_model.model."): value for name, value in tensors.items()},
            manifest,
        )


@pytest.mark.parametrize(
    "kind",
    ["missing", "extra", "collision", "shape", "nonfinite", "different", "signed_zero"],
)
def test_lora_init_verification_rejects_coverage_shape_and_value_changes(kind):
    specs = _lora_init_specs()
    manifest = lora_init.build_manifest(specs, 1234)
    tensors = _lora_init_tensors(specs)
    if kind == "missing":
        del tensors[_lora_init_B]
    elif kind == "extra":
        tensors[_lora_init_A.replace("q_proj", "k_proj")] = tensors[_lora_init_A]
    elif kind == "collision":
        tensors[_lora_init_A.replace(".weight", ".default.weight")] = tensors[_lora_init_A]
    elif kind == "shape":
        tensors[_lora_init_A] = tensors[_lora_init_A].reshape(8, 8)
    elif kind == "nonfinite":
        tensors[_lora_init_A][0, 0] = float("nan")
    elif kind == "different":
        tensors[_lora_init_A][0, 0] = 0.5
    else:
        tensors[_lora_init_B][0, 0] = -0.0
    with pytest.raises(ValueError):
        lora_init.verify_tensors(tensors, manifest)


def test_lora_init_only_explicit_exact_float32_widening_is_accepted():
    specs = _lora_init_specs()
    manifest = lora_init.build_manifest(specs, 1234)
    tensors = {name: tensor.float() for name, tensor in _lora_init_tensors(specs).items()}
    with pytest.raises(ValueError, match="dtype"):
        lora_init.verify_tensors(tensors, manifest)
    lora_init.verify_tensors(tensors, manifest, allow_exact_float32_widening=True)
    tensors[_lora_init_A][0, 0] += 2 ** (-20)
    with pytest.raises(ValueError, match="bytes"):
        lora_init.verify_tensors(tensors, manifest, allow_exact_float32_widening=True)
    fp32_specs = _lora_init_specs("float32")
    with pytest.raises(ValueError, match="dtype"):
        lora_init.verify_tensors(
            {name: tensor.bfloat16() for name, tensor in _lora_init_tensors(fp32_specs).items()},
            lora_init.build_manifest(fp32_specs, 1234),
            allow_exact_float32_widening=True,
        )


def test_lora_init_non_lora_is_noop_without_model_or_storage_access(monkeypatch):
    cfg = SimpleNamespace(training={})
    monkeypatch.setattr(
        deterministic_lora,
        "require_data_fast_path",
        lambda *a, **k: pytest.fail("non-LoRA storage accessed"),
    )
    assert deterministic_lora.materialize_adapter(cfg, "/absent/model", "/code/not-fast") is None
    payload = {"training_config": {}, "model_name": "Qwen/Qwen3-8B"}
    assert deterministic_lora.bind_payload(payload, None) is payload


def test_lora_init_target_binding_never_invents_public_adapter_or_resume_fields():
    payload = {
        "training_config": {"peft_config": {"r": 4}},
        "model_name": "Qwen/Qwen3-8B",
    }
    before = deepcopy(payload)
    with pytest.raises(
        deterministic_lora.LoraInitializationPrerequisiteError,
        match="public training adapter",
    ):
        deterministic_lora.bind_payload(payload, Path("/data-fast/tmp/adapter"))
    assert payload == before
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="requires"):
        deterministic_lora.bind_payload(payload, None)


@pytest.fixture
def _lora_init_local_adapter_fixture(tmp_path, monkeypatch):
    root = tmp_path / "data-fast"
    root.mkdir()
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", root)
    source = root / "model"
    source.mkdir()
    (source / "config.json").write_text('{"architectures":["SyntheticModel"]}')

    class SyntheticModel(torch.nn.Module):

        def __init__(self, config):
            super().__init__()
            self.base = torch.nn.Linear(16, 12, bias=False)
            self.base.requires_grad_(False)

    class SyntheticLoraConfig:

        def __init__(self, **kwargs):
            self.values = kwargs

        def save_pretrained(self, destination):
            (Path(destination) / "adapter_config.json").write_text(json.dumps(self.values, sort_keys=True))

    def get_peft_model(model, config):
        model.q_proj = torch.nn.Module()
        model.q_proj.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(16, config.values["r"], bias=False)})
        model.q_proj.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(config.values["r"], 12, bias=False)})
        model.peft_config = {"default": config}
        return model

    def from_pretrained(path, **kwargs):
        assert Path(path) == source
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return SimpleNamespace(architectures=["SyntheticModel"])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=from_pretrained),
            SyntheticModel=SyntheticModel,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=SyntheticLoraConfig,
            get_peft_model=get_peft_model,
            get_peft_model_state_dict=lambda model, state_dict, adapter_name: {
                name.replace(".default.weight", ".weight"): tensor for name, tensor in state_dict.items()
            },
        ),
    )
    cfg = SimpleNamespace(
        training={"peft_config": {"peft_type": "Lora", "r": 4, "target_modules": ["q_proj"]}},
        sub_job={"dtype": "bfloat16", "seed": 1234},
        model_name="Qwen/Qwen3-8B",
    )
    return (cfg, source, root / "runtime")


def test_lora_init_temporary_pack_uses_meta_shapes_and_preserves_rng_and_identity(
    _lora_init_local_adapter_fixture,
):
    cfg, source, scratch = _lora_init_local_adapter_fixture
    rng = torch.random.get_rng_state().clone()
    first = deterministic_lora.materialize_adapter(cfg, source, scratch)
    assert torch.equal(torch.random.get_rng_state(), rng)
    second = deterministic_lora.materialize_adapter(cfg, source, scratch)
    assert first != second
    assert first.parent == scratch
    assert {path.name for path in first.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "initialization.json",
    }
    assert deterministic_lora.identity_metadata(first) == deterministic_lora.identity_metadata(second)
    assert not list(source.glob("*.safetensors"))
    assert not list(source.glob("*.pt"))


def test_lora_init_temporary_pack_refuses_non_fast_storage_before_topology_import(
    _lora_init_local_adapter_fixture,
):
    cfg, source, scratch = _lora_init_local_adapter_fixture
    with pytest.raises(ValueError, match="/data-fast"):
        deterministic_lora.materialize_adapter(cfg, source, scratch.parent.parent / "outside")


@pytest.mark.parametrize("kind", ["metadata", "adapter_config", "tensor"])
def test_lora_init_pack_identity_refuses_mutation(_lora_init_local_adapter_fixture, kind):
    from safetensors.torch import load_file
    from safetensors.torch import save_file

    cfg, source, scratch = _lora_init_local_adapter_fixture
    output = deterministic_lora.materialize_adapter(cfg, source, scratch)
    if kind == "metadata":
        path = output / "initialization.json"
        metadata = json.loads(path.read_text())
        metadata["model_name"] = "another-model"
        path.write_text(json.dumps(metadata))
    elif kind == "adapter_config":
        (output / "adapter_config.json").write_text("{}")
    else:
        path = output / "adapter_model.safetensors"
        tensors = load_file(str(path))
        tensors[sorted(tensors)[0]][0, 0] += 1
        save_file(tensors, str(path))
    with pytest.raises(ValueError):
        deterministic_lora.identity_metadata(output)


def test_lora_init_real_peft_tiny_qwen_serialization_and_loaded_initial_values(tmp_path, monkeypatch):
    import peft
    from safetensors.torch import load_file
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM

    root = tmp_path / "data-fast"
    root.mkdir()
    monkeypatch.setattr(workdir, "_DATA_FAST_ROOT", root)
    source = root / "model"
    source.mkdir()
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        architectures=["Qwen3ForCausalLM"],
    )
    config.save_pretrained(source)
    cfg = SimpleNamespace(
        training={
            "peft_config": {
                "peft_type": "Lora",
                "r": 4,
                "target_modules": ["q_proj", "v_proj"],
            }
        },
        sub_job={"dtype": "bfloat16", "seed": 1234},
        model_name="Qwen/Qwen3-8B",
    )
    rng = torch.random.get_rng_state().clone()
    output = deterministic_lora.materialize_adapter(cfg, source, root / "runtime")
    assert torch.equal(torch.random.get_rng_state(), rng)
    identity = deterministic_lora.adapter_identity(output)
    assert identity == deterministic_lora.identity_metadata(output)
    state = load_file(str(output / "adapter_model.safetensors"))
    assert len(state) == 4
    assert all((".default." not in name for name in state))
    assert all((name.startswith("base_model.model.model.layers.0.") for name in state))
    with torch.random.fork_rng(devices=[]):
        base = Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.bfloat16)
        loaded = peft.PeftModel.from_pretrained(base, output, is_trainable=True, local_files_only=True)
    actual = {name: parameter for name, parameter in loaded.named_parameters() if parameter.requires_grad}
    receipt = lora_init.verify_tensors(actual, identity["initialization"], allow_exact_float32_widening=True)
    assert len(receipt["verified_tensors"]) == 4
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError):
        deterministic_lora.bind_payload({"training_config": cfg.training}, output)


_lora_save0_TOPOLOGY = normalize_lora_peft_config({"peft_type": "Lora", "r": 2, "target_modules": ["q_proj"]})


def _lora_save0_tiny_model():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
        )
    )


def _lora_save0_metadata(model):
    specs = [
        lora_init.LoraTensorSpec(lora_init.canonical_name(name), tuple(parameter.shape), "float32")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "model_name": "Qwen/Qwen3-8B",
        "initialization": lora_init.build_manifest(specs, 42),
        "peft_config": deepcopy(_lora_save0_TOPOLOGY),
    }


def _lora_save0_saved(tmp_path):
    from peft import LoraConfig
    from peft import get_peft_model

    model = get_peft_model(
        _lora_save0_tiny_model(),
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=8, target_modules=["q_proj"]),
    )
    metadata = _lora_save0_metadata(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                spec = lora_init.LoraTensorSpec(lora_init.canonical_name(name), tuple(parameter.shape), "float32")
                parameter.copy_(lora_init.generate_tensor(spec, 42))
    root = tmp_path / "checkpoint"
    destination = root / "global_step0"
    save_peft_adapters(model, str(destination))
    response = {
        "checkpoint_path": str(root),
        "checkpoint_tag": "global_step0",
        "version": 0,
        "stage_path": None,
    }
    return (root, response, metadata, model)


def test_lora_save0_absent_option_preserves_normalized_topology_and_default_peft_behavior():
    assert "deterministic_init" not in normalize_lora_peft_config(_lora_save0_TOPOLOGY)
    config = {"peft_type": "Lora", "r": 2, "target_modules": ["q_proj"]}
    set_seed(17)
    baseline = apply_peft(_lora_save0_tiny_model(), config)
    set_seed(17)
    repeated = apply_peft(_lora_save0_tiny_model(), config)
    for (name, actual), (other_name, expected) in zip(
        baseline.named_parameters(), repeated.named_parameters(), strict=True
    ):
        assert name == other_name
        torch_assert_equal(actual, expected)


def test_lora_save0_real_peft_save_pack_proves_complete_exact_initialization(tmp_path):
    root, response, metadata, _ = _lora_save0_saved(tmp_path)
    proof = lora_save0.verify_saved_initialization(root, response, metadata)
    assert proof["identity_sha256"] == metadata["initialization"]["identity_sha256"]
    assert len(proof["verified_tensors"]) == len(metadata["initialization"]["tensors"])
    assert proof["proof_scope"] == "exported-adapter-pack-not-all-rank-equality"


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 1),
        ("version", False),
        ("checkpoint_tag", "global_step1"),
        ("checkpoint_path", "/data-fast/tmp/not-owned"),
        ("stage_path", "remote"),
    ],
)
def test_lora_save0_save0_rejects_wrong_step_path_or_stage(tmp_path, field, value):
    root, response, metadata, _ = _lora_save0_saved(tmp_path)
    response[field] = value
    with pytest.raises(ValueError, match="step/version zero"):
        lora_save0.verify_saved_initialization(root, response, metadata)


@pytest.mark.parametrize("failure", ["missing", "extra", "bytes", "dtype", "shape", "symlink", "config"])
def test_lora_save0_save0_rejects_changed_coverage_identity_and_files(tmp_path, failure):
    root, response, metadata, _ = _lora_save0_saved(tmp_path)
    directory = root / "global_step0" / "default"
    path = directory / "adapter_model.safetensors"
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    name = next(iter(tensors))
    if failure == "missing":
        del tensors[name]
    elif failure == "extra":
        tensors[name.replace("q_proj", "k_proj")] = tensors[name].clone()
    elif failure == "bytes":
        tensors[name] = tensors[name] + 1
    elif failure == "dtype":
        tensors[name] = tensors[name].to(torch.bfloat16)
    elif failure == "shape":
        tensors[name] = tensors[name][:1]
    elif failure == "symlink":
        target = directory / "hidden.safetensors"
        path.rename(target)
        path.symlink_to(target)
    elif failure == "config":
        config_path = directory / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["r"] = 99
        config_path.write_text(json.dumps(config))
    if failure not in {"symlink", "config"}:
        save_file(tensors, str(path))
    with pytest.raises(ValueError):
        lora_save0.verify_saved_initialization(root, response, metadata)


class _qwen4_reference_TinyQwen4(nn.Module):

    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.ple = nn.Module()
        layer.ple.ple_embedding = nn.Module()
        layer.ple.ple_embedding.ngram_embedding = nn.Embedding(5, 2)
        layer.ple.key_proj = nn.Linear(2, 2, bias=False)
        layer.self_attn = nn.Module()
        layer.self_attn.indexer = nn.Linear(2, 2, bias=False)
        layer.self_attn.q_proj = nn.Linear(2, 2, bias=False)
        layer.register_buffer("positions", torch.arange(2), persistent=False)
        self.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(2, 5, bias=False)
        self.config = SimpleNamespace(hidden_size=2)
        self.checkpointing = False

    def get_decoder(self):
        return self

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self):
        self.checkpointing = True


def _qwen4_reference_loader(model, info=None):
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return (model, info or {})

    return (SimpleNamespace(from_pretrained=from_pretrained), calls)


@pytest.mark.parametrize("attention,expected", [("qsa_flex", "sdpa"), ("sdpa", "sdpa"), ("eager", "eager")])
def test_qwen4_reference_upstream_backend_and_cpu_checkpoint_load(attention, expected):
    model = _qwen4_reference_TinyQwen4()
    cls, calls = _qwen4_reference_loader(model)
    result = load_qwen4_reference(
        cls,
        "fixture",
        dtype=torch.bfloat16,
        attn_implementation=attention,
        device="cpu",
    )
    assert result is model
    assert calls == [
        (
            "fixture",
            {
                "dtype": torch.bfloat16,
                "attn_implementation": expected,
                "output_loading_info": True,
            },
        )
    ]
    layer = model.layers[0]
    assert not layer.ple.ple_embedding.ngram_embedding.weight.requires_grad
    assert not layer.self_attn.indexer.weight.requires_grad
    assert layer.ple.key_proj.weight.requires_grad
    assert layer.self_attn.q_proj.weight.requires_grad


def test_qwen4_reference_table_never_reaches_accelerator_placement(monkeypatch):
    model = _qwen4_reference_TinyQwen4()
    table = model.layers[0].ple.ple_embedding.ngram_embedding.weight
    original = table.detach().clone()
    parameter_ids = {id(parameter) for parameter in model.parameters()} - {id(table)}
    moved = []
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            assert tensor is not table
            moved.append(id(tensor))
            return original_to(tensor, device="cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    cls, _ = _qwen4_reference_loader(model)
    load_qwen4_reference(
        cls,
        "fixture",
        dtype=torch.bfloat16,
        attn_implementation="qsa_flex",
        device="cuda",
    )
    assert parameter_ids <= set(moved)
    assert id(model.layers[0].positions) in moved
    assert model.layers[0].ple.ple_embedding.ngram_embedding.weight is table
    assert table.device.type == "cpu"
    assert torch.equal(table, original)


@pytest.mark.parametrize("attention", ["flash_attention_3", "flash_attention_4", "flex_attention", "unknown"])
def test_qwen4_reference_unsupported_attention_rejected_before_loading(attention):
    cls, calls = _qwen4_reference_loader(_qwen4_reference_TinyQwen4())
    with pytest.raises(ValueError, match="upstream QSA"):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation=attention, device="cpu")
    assert calls == []


@pytest.mark.parametrize("key", ["missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs"])
def test_qwen4_reference_incomplete_checkpoint_rejected_before_placement(key, monkeypatch):
    model = _qwen4_reference_TinyQwen4()
    cls, _ = _qwen4_reference_loader(model, {key: ["ngram_embedding.weight"]})
    monkeypatch.setattr(model, "_apply", lambda *args, **kwargs: pytest.fail("placement reached"))
    with pytest.raises(ValueError, match=key):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation="qsa_flex", device="cpu")


def test_qwen4_reference_meta_checkpoint_rejected_before_placement():
    model = _qwen4_reference_TinyQwen4().to("meta")
    cls, _ = _qwen4_reference_loader(model)
    with pytest.raises(ValueError, match="load on CPU"):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation="qsa_flex", device="cpu")


@pytest.mark.parametrize("missing", ["ple", "self_attn"])
def test_qwen4_reference_missing_layer_period_rejected(missing):
    model = _qwen4_reference_TinyQwen4()
    delattr(model.layers[0], missing)
    cls, _ = _qwen4_reference_loader(model)
    with pytest.raises(ValueError, match="complete PLE/QSA"):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation="qsa_flex", device="cpu")


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text", "qwen3"])
def test_qwen4_reference_actual_build_engine_family_dispatch(monkeypatch, model_type):
    model = _qwen4_reference_TinyQwen4()
    calls = []
    cfg = SimpleNamespace(model_type=model_type, architectures=["Tiny"], hidden_size=2)

    def from_pretrained(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("output_loading_info"):
            return (model, {})
        return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: cfg),
            Tiny=SimpleNamespace(from_pretrained=from_pretrained),
        ),
    )
    engine = build_engine("fixture", attn_implementation="qsa_flex", device="cpu")
    assert engine.model is model
    assert model.checkpointing and model.training
    if model_type.startswith("qwen4"):
        assert calls[0]["attn_implementation"] == "sdpa"
        assert calls[0]["output_loading_info"] is True
        assert not model.layers[0].ple.ple_embedding.ngram_embedding.weight.requires_grad
    else:
        assert calls[0] == {"dtype": torch.bfloat16, "attn_implementation": "qsa_flex"}
        assert model.layers[0].ple.ple_embedding.ngram_embedding.weight.requires_grad


# Shared lora adapter.


def _model(*, dtype=torch.float32, rank=2, targets=("q_proj", "v_proj")):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    with torch.random.fork_rng(devices=[]):
        model = peft.get_peft_model(
            Qwen3ForCausalLM(config).to(device="cpu", dtype=dtype),
            peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=rank,
                lora_alpha=4,
                target_modules=list(targets),
                lora_dropout=0.0,
            ),
        )
    return model.to(dtype=dtype)


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _installed(model):
    return {canonical_name(name): parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


@pytest.fixture
def adapter_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DATA_FAST_ROOT", tmp_path)
    model = _model()
    with torch.no_grad():
        for index, parameter in enumerate(_installed(model).values()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape) / 16 - index
            values.reshape(-1)[0] = -0.0
            parameter.copy_(values)
    path = tmp_path / "adapter"
    model.save_pretrained(path, safe_serialization=True)
    state = {name: tensor.clone() for name, tensor in load_file(str(path / "adapter_model.safetensors")).items()}
    return SimpleNamespace(path=path, state=state)


def test_shared_adapter_genuine_peft_identity_binds_config_storage_and_every_tensor(
    adapter_pack,
):
    pack = adapter_pack
    identity = shared.adapter_identity(pack.path)
    config = json.loads((pack.path / "adapter_config.json").read_text())
    assert identity["schema"] == "shared-peft-adapter-v1"
    assert len(pack.state) == len(identity["tensors"]) == 4
    assert all(".default." not in name for name in pack.state)
    assert (
        identity["adapter_config_sha256"]
        == hashlib.sha256((pack.path / "adapter_config.json").read_bytes()).hexdigest()
    )
    assert (
        identity["adapter_model_sha256"]
        == hashlib.sha256((pack.path / "adapter_model.safetensors").read_bytes()).hexdigest()
    )
    for entry in identity["tensors"]:
        tensor = pack.state[entry["name"]]
        assert entry == {
            "name": entry["name"],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(_bytes(tensor)).hexdigest(),
        }
    reordered = {**config, "target_modules": list(reversed(config["target_modules"]))}
    assert shared.adapter_identity(pack.path, reordered) == identity
    relocated = pack.path.parent / "relocated"
    shutil.copytree(pack.path, relocated)
    assert shared.adapter_identity(relocated) == identity


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_shared_adapter_reload_installs_exact_real_peft_values_before_optimizer(adapter_pack, dtype):
    state = {name: tensor.to(dtype) for name, tensor in adapter_pack.state.items()}
    save_file(state, str(adapter_pack.path / "adapter_model.safetensors"))
    identity = shared.adapter_identity(adapter_pack.path)
    model = _model(dtype=dtype)
    parameters = dict(model.named_parameters())
    frozen = {name: _bytes(value) for name, value in parameters.items() if not value.requires_grad}
    rng = torch.random.get_rng_state().clone()
    assert shared.initialize_adapter(model, adapter_pack.path, identity) is None
    assert torch.equal(torch.random.get_rng_state(), rng)
    installed = _installed(model)
    assert set(installed) == set(state)
    for name, parameter in installed.items():
        assert _bytes(parameter) == _bytes(state[name])
        assert torch.signbit(parameter.reshape(-1)[0])
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in parameters.items())
    assert all(_bytes(parameters[name]) == raw for name, raw in frozen.items())
    optimizer = torch.optim.AdamW(installed.values(), lr=0.01)
    assert not optimizer.state


@pytest.mark.parametrize("changed", ["config", "tensor", "signed-zero"])
def test_shared_adapter_changed_identity_rejects_before_target_mutation(adapter_pack, changed):
    identity = shared.adapter_identity(adapter_pack.path)
    if changed == "config":
        path = adapter_pack.path / "adapter_config.json"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        state = adapter_pack.state
        name = sorted(state)[0]
        state[name].reshape(-1)[0] = 0.0 if changed == "signed-zero" else 1.0
        save_file(state, str(adapter_pack.path / "adapter_model.safetensors"))
    model = _model()
    before = {name: _bytes(parameter) for name, parameter in _installed(model).items()}
    with pytest.raises(ValueError, match="changed after reference admission"):
        shared.initialize_adapter(model, adapter_pack.path, identity)
    assert {name: _bytes(parameter) for name, parameter in _installed(model).items()} == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("peft_type", "IA3"),
        ("r", 3),
        ("target_modules", ["k_proj"]),
        ("target_parameters", ["experts.gate_up_proj"]),
        ("modules_to_save", ["lm_head"]),
        ("use_dora", True),
        ("use_rslora", True),
        ("lora_bias", True),
        ("rank_pattern", {"q_proj": 3}),
        ("alpha_pattern", {"q_proj": 3}),
    ],
)
def test_shared_adapter_rejects_different_or_unsupported_topology(adapter_pack, field, value):
    config = json.loads((adapter_pack.path / "adapter_config.json").read_text())
    config[field] = value
    with pytest.raises(ValueError):
        shared.adapter_identity(adapter_pack.path, config)


@pytest.mark.parametrize(
    "failure",
    [
        "empty",
        "missing",
        "extra",
        "collision",
        "shape",
        "rank",
        "dtype",
        "mixed-dtype",
        "nan",
        "inf",
        "key",
    ],
)
def test_shared_adapter_rejects_invalid_serialized_tensor_inventory(adapter_pack, failure):
    state = dict(adapter_pack.state)
    name = sorted(state)[0]
    partner = name.replace("lora_A", "lora_B")
    if failure == "empty":
        state.clear()
    elif failure == "missing":
        del state[partner]
    elif failure == "extra":
        state["lm_head.weight"] = state[name].clone()
    elif failure == "collision":
        state[name.replace(".lora_A.", "._checkpoint_wrapped_module.lora_A.")] = state[name].clone()
    elif failure == "shape":
        state[name] = state[name].reshape(-1)
    elif failure == "rank":
        state[name] = state[name][:1].clone()
    elif failure == "dtype":
        state[name] = state[name].to(torch.float64)
    elif failure == "mixed-dtype":
        state[name] = state[name].to(torch.bfloat16)
    elif failure in {"nan", "inf"}:
        state[name].reshape(-1)[0] = float(failure)
    else:
        state[name.replace(".lora_A.", ".lora_C.")] = state.pop(name)
    save_file(state, str(adapter_pack.path / "adapter_model.safetensors"))
    with pytest.raises(ValueError):
        shared.adapter_identity(adapter_pack.path)


@pytest.mark.parametrize("failure", ["rank", "targets", "dtype", "shape", "adapter-name"])
def test_shared_adapter_rejects_incompatible_installed_model_without_loading(adapter_pack, failure):
    identity = shared.adapter_identity(adapter_pack.path)
    model = _model(
        rank=3 if failure == "rank" else 2,
        targets=("k_proj",) if failure == "targets" else ("q_proj", "v_proj"),
        dtype=torch.bfloat16 if failure == "dtype" else torch.float32,
    )
    if failure == "shape":
        module = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"]
        module.weight = torch.nn.Parameter(module.weight.detach()[:, :8].clone())
    elif failure == "adapter-name":
        model.add_adapter(
            "second",
            peft.LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj"]),
        )
    before = {name: _bytes(parameter) for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="topology|coverage"):
        shared.initialize_adapter(model, adapter_pack.path, identity)
    assert {name: _bytes(parameter) for name, parameter in model.named_parameters()} == before


@pytest.mark.parametrize("failure", ["unexpected-keys", "inexact-copy", "identity-drift"])
def test_shared_adapter_rejects_failed_load_or_source_drift(adapter_pack, monkeypatch, failure):
    identity = shared.adapter_identity(adapter_pack.path)
    model = _model()
    original = peft.set_peft_model_state_dict

    def load(target, state, *, adapter_name):
        assert target is model and adapter_name == "default"
        if failure == "unexpected-keys":
            return SimpleNamespace(unexpected_keys=["foreign.weight"])
        result = original(target, state, adapter_name=adapter_name)
        if failure == "inexact-copy":
            with torch.no_grad():
                next(iter(_installed(target).values())).reshape(-1)[0] = 0.0
        else:
            config = adapter_pack.path / "adapter_config.json"
            config.write_bytes(config.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(peft, "set_peft_model_state_dict", load)
    with pytest.raises(ValueError, match="unexpected keys|differ after loading|changed during loading"):
        shared.initialize_adapter(model, adapter_pack.path, identity)


@pytest.mark.parametrize("failure", [None, "build", "load", "body"])
def test_shared_adapter_build_scope_restores_worker_builder_on_all_exits(adapter_pack, monkeypatch, failure):
    from arctic_platform import common

    worker = ModuleType("arctic_platform.common.deepspeed_worker")
    loaded = SimpleNamespace(model=_model())
    calls = []

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        if failure == "build":
            raise RuntimeError("builder failed")
        return loaded

    worker.build_model = build
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    monkeypatch.setattr(common, "deepspeed_worker", worker, raising=False)
    identity = shared.adapter_identity(adapter_pack.path)
    if failure == "load":
        identity = copy.deepcopy(identity)
        identity["schema"] = "foreign"

    def execute():
        with shared.adapter_build_scope(adapter_pack.path, identity):
            assert worker.build_model is not build
            result = worker.build_model("spec", parallel_groups="groups")
            assert result is loaded
            assert calls == [(("spec",), {"parallel_groups": "groups"})]
            assert {_name: _bytes(value) for _name, value in _installed(loaded.model).items()} == {
                canonical_name(name): _bytes(value) for name, value in adapter_pack.state.items()
            }
            if failure == "body":
                raise RuntimeError("body failed")

    if failure:
        with pytest.raises(ValueError if failure == "load" else RuntimeError):
            execute()
    else:
        execute()
    assert worker.build_model is build


def test_shared_adapter_nested_scope_restores_outer_then_original_builder(adapter_pack, monkeypatch):
    from arctic_platform import common

    worker = ModuleType("arctic_platform.common.deepspeed_worker")
    worker.build_model = lambda *args, **kwargs: SimpleNamespace(model=_model())
    original = worker.build_model
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    monkeypatch.setattr(common, "deepspeed_worker", worker, raising=False)
    identity = shared.adapter_identity(adapter_pack.path)
    with shared.adapter_build_scope(adapter_pack.path, identity):
        outer = worker.build_model
        with shared.adapter_build_scope(adapter_pack.path, identity):
            assert worker.build_model is not outer
            worker.build_model("spec")
        assert worker.build_model is outer
    assert worker.build_model is original


@pytest.mark.parametrize("order", ["batched", "arm"])
def test_shared_adapter_runner_binds_identical_pack_to_both_cached_reference_arms(
    adapter_pack, _prepared_reference_runner_prepared_runner, monkeypatch, order
):
    from arctic_platform.correctness.harness import runner

    control = _prepared_reference_runner_prepared_runner
    identity = shared.adapter_identity(adapter_pack.path)
    control.cfg.training["peft_config"] = copy.deepcopy(identity["peft_config"])
    payloads = []

    @contextlib.contextmanager
    def running(session, payload):
        assert session == "mock-target-session"
        payloads.append(copy.deepcopy(payload))
        yield "mock-target-job"

    monkeypatch.setattr(runner, "running_job", running)
    results = control.execute(peft_adapter=adapter_pack.path, order=order)
    assert len(results) == 4
    assert all(result.outcome is TestOutcome.PASS for result in results)
    assert len(payloads) == 2
    for payload in payloads:
        assert payload["training_config"]["peft_config"] == identity["peft_config"]
        assert payload["training_config"]["debug"]["shared_peft_adapter"] == {
            "path": str(adapter_pack.path),
            "identity": identity,
        }
        assert "deterministic_init" not in payload["training_config"]["peft_config"]
    assert len(control.recipes) == 4
    assert all(recipe["initialization"] == identity for _, _, recipe in control.recipes)
    assert not control.published
    assert [event for event in control.events if event[0] == "target"] == [
        ("target", "gas1"),
        ("target", "gas4"),
    ]


@pytest.mark.parametrize("failure", ["dense", "rl", "cortex", "selection", "retired"])
def test_shared_adapter_runner_refuses_unadmitted_inputs_before_reference_or_target_work(
    adapter_pack, _prepared_reference_runner_prepared_runner, failure
):
    control = _prepared_reference_runner_prepared_runner
    control.cfg.training["peft_config"] = shared.adapter_identity(adapter_pack.path)["peft_config"]
    options = {"peft_adapter": adapter_pack.path}
    if failure == "dense":
        control.cfg.training.pop("peft_config")
    elif failure == "rl":
        control.cfg.native = control.cfg.native.model_copy(update={"sampling_gpus": 4})
    elif failure == "cortex":
        options["client_backend"] = CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev")
    elif failure == "selection":
        options["selected"] = ["e2e-sft-train-validate"]
    else:
        options["local_gateway_lora_save0"] = True
    with pytest.raises(ValueError):
        control.execute(**options)
    assert not control.events and not control.recipes and not control.published


def test_shared_adapter_onboard_freezes_supplied_identity_and_rejects_later_mutation(
    adapter_pack,
):
    from arctic_platform.correctness.onboarding import onboard

    config = SimpleNamespace(training={"peft_config": shared.adapter_identity(adapter_pack.path)["peft_config"]})
    args = SimpleNamespace(peft_adapter=adapter_pack.path)
    assert onboard._shared_adapter(args, config) == adapter_pack.path
    assert args._admitted_shared_adapter_identity == shared.adapter_identity(adapter_pack.path)
    assert onboard._shared_adapter(args, config) == adapter_pack.path
    path = adapter_pack.path / "adapter_config.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed during onboarding"):
        onboard._shared_adapter(args, config)


@pytest.fixture
def generated_adapter_config(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DATA_FAST_ROOT", tmp_path)
    model_path = tmp_path / "config-only-base"
    Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        architectures=["Qwen3ForCausalLM"],
    ).save_pretrained(model_path)
    config = {
        "peft_type": "Lora",
        "task_type": "CAUSAL_LM",
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
        "init_lora_weights": True,
    }
    return SimpleNamespace(path=model_path, config=config)


@pytest.mark.parametrize("seed", [0, 42])
def test_generated_adapter_reproduces_bytes_across_base_output_paths_and_ambient_rng(
    generated_adapter_config, tmp_path, seed
):
    control = generated_adapter_config
    second_source = tmp_path / "other-node" / "config-only-base"
    shutil.copytree(control.path, second_source)
    identities = []
    packs = []
    config_before = copy.deepcopy(control.config)
    with torch.random.fork_rng(devices=[]):
        for index, source in enumerate((control.path, second_source)):
            torch.random.default_generator.manual_seed(1000 + index)
            before = torch.random.get_rng_state().clone()
            path = shared.generate_adapter(source, control.config, tmp_path / f"generated-{index}", seed=seed)
            assert torch.equal(torch.random.get_rng_state(), before)
            identities.append(shared.adapter_identity(path, control.config))
            packs.append(shared.adapter_files(path, identities[-1]))
    assert identities[0] == identities[1]
    assert packs[0] == packs[1]
    assert control.config == config_before
    assert set(packs[0]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "correctness_recipe.json",
    }
    assert json.loads(packs[0]["adapter_config.json"])["base_model_name_or_path"] == ""
    for source in (control.path, second_source):
        assert {path.name for path in source.iterdir()} == {"config.json"}


def test_generated_adapter_seed_changes_a_only_and_binds_upstream_recipe(generated_adapter_config, tmp_path):
    control = generated_adapter_config
    states = []
    identities = []
    for seed in (41, 42):
        path = shared.generate_adapter(control.path, control.config, tmp_path / f"seed-{seed}", seed=seed)
        states.append(load_file(str(path / "adapter_model.safetensors")))
        identities.append(shared.adapter_identity(path))
        assert identities[-1]["generation"] == {
            "schema": "upstream-default-lora-v1",
            "seed": seed,
            "initializer": True,
            "peft_config": identities[-1]["peft_config"],
            "dependencies": {
                "torch": str(torch.__version__),
                "peft": peft.__version__,
                "transformers": transformers.__version__,
                "safetensors": safetensors.__version__,
            },
        }
    assert identities[0]["adapter_model_sha256"] != identities[1]["adapter_model_sha256"]
    assert set(states[0]) == set(states[1])
    assert len(states[0]) == 4
    for name, tensor in states[0].items():
        assert tensor.dtype == torch.float32 and tensor.ndim == 2
        if ".lora_A." in name:
            assert tensor.shape == (2, 16)
            assert torch.count_nonzero(tensor) > 0
            assert torch.max(torch.abs(tensor)).item() <= tensor.shape[1] ** -0.5
            assert _bytes(tensor) != _bytes(states[1][name])
        else:
            assert ".lora_B." in name and tensor.shape == (16, 2)
            assert _bytes(tensor) == _bytes(states[1][name]) == bytes(tensor.numel() * tensor.element_size())


def test_generated_adapter_constructs_meta_base_and_allocates_only_cpu_factors(
    generated_adapter_config, tmp_path, monkeypatch
):
    control = generated_adapter_config
    original = peft.get_peft_model
    observed = []
    resets = []
    reset = peft.tuners.lora.layer.LoraLayer.reset_lora_parameters

    def wrap(model, config, *args, **kwargs):
        assert all(parameter.device.type == "meta" for parameter in model.parameters())
        result = original(model, config, *args, **kwargs)
        assert all(parameter.device.type == "meta" for parameter in result.parameters())
        observed.append(result)
        return result

    def record_reset(module, adapter_name, init_lora_weights):
        if module.lora_A[adapter_name].weight.device.type == "cpu":
            resets.append((adapter_name, init_lora_weights))
        return reset(module, adapter_name, init_lora_weights)

    def reject_weight_load(*args, **kwargs):
        pytest.fail("generation must not load pretrained base weights")

    monkeypatch.setattr(peft, "get_peft_model", wrap)
    monkeypatch.setattr(peft.tuners.lora.layer.LoraLayer, "reset_lora_parameters", record_reset)
    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", reject_weight_load)
    destination = shared.generate_adapter(control.path, control.config, tmp_path / "meta-generated")
    assert resets == [("default", True), ("default", True)]
    assert len(observed) == 1
    base = [parameter for name, parameter in observed[0].named_parameters() if ".lora_" not in name]
    factors = list(_installed(observed[0]).values())
    assert base and all(parameter.device.type == "meta" for parameter in base)
    assert len(factors) == 4 and all(parameter.device.type == "cpu" for parameter in factors)
    assert {path.name for path in control.path.iterdir()} == {"config.json"}
    assert not (destination / "model.safetensors").exists()
    assert not (destination / "pytorch_model.bin").exists()


@pytest.mark.parametrize("initializer", [False, "gaussian", "pissa"])
def test_generated_adapter_refuses_nondefault_upstream_initializer(generated_adapter_config, tmp_path, initializer):
    control = generated_adapter_config
    config = {**control.config, "init_lora_weights": initializer}
    with pytest.raises(ValueError, match="initializer|init_lora_weights"):
        shared.generate_adapter(control.path, config, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("seed", [-1, True, 1.5, "42"])
def test_generated_adapter_refuses_invalid_seed(generated_adapter_config, tmp_path, seed):
    control = generated_adapter_config
    with pytest.raises(ValueError, match="seed"):
        shared.generate_adapter(control.path, control.config, tmp_path / "rejected", seed=seed)
    assert not (tmp_path / "rejected").exists()


def test_generated_adapter_restores_rng_on_upstream_save_failure(generated_adapter_config, tmp_path, monkeypatch):
    control = generated_adapter_config

    def fail_save(*args, **kwargs):
        torch.rand(7)
        raise RuntimeError("upstream save failed")

    monkeypatch.setattr(peft.PeftModel, "save_pretrained", fail_save)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="upstream save failed"):
        shared.generate_adapter(control.path, control.config, tmp_path / "failed-save")
    assert torch.equal(torch.random.get_rng_state(), before)


def test_generated_adapter_dependency_recipe_drift_invalidates_admitted_identity(generated_adapter_config, tmp_path):
    control = generated_adapter_config
    path = shared.generate_adapter(control.path, control.config, tmp_path / "generated")
    admitted = shared.adapter_identity(path)
    recipe_path = path / "correctness_recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["dependencies"]["peft"] = "different-version"
    recipe_path.write_text(json.dumps(recipe))
    changed = shared.adapter_identity(path)
    assert changed["tensors"] == admitted["tensors"]
    assert changed["adapter_model_sha256"] == admitted["adapter_model_sha256"]
    assert changed != admitted
    with pytest.raises(ValueError, match="changed before transfer"):
        shared.adapter_files(path, admitted)
    with pytest.raises(ValueError, match="changed after reference admission"):
        shared.initialize_adapter(_model(), path, admitted)


@pytest.fixture
def worker_adapter_transfer(generated_adapter_config, tmp_path, monkeypatch):
    from arctic_platform import common

    control = generated_adapter_config
    pack = shared.generate_adapter(control.path, control.config, tmp_path / "coordinator-pack")
    identity = shared.adapter_identity(pack)
    files = shared.adapter_files(pack, identity)
    state = {name: tensor.clone() for name, tensor in load_file(str(pack / "adapter_model.safetensors")).items()}
    shutil.rmtree(pack)
    assert not pack.exists()
    worker_root = tmp_path / "worker-node"
    monkeypatch.setenv(storage.CORRECTNESS_TMPDIR_ENV, str(worker_root))
    ref = object()
    gets = []
    ray = ModuleType("ray")

    def get(actual_ref):
        assert actual_ref is ref
        gets.append(actual_ref)
        return files

    ray.get = get
    monkeypatch.setitem(sys.modules, "ray", ray)
    scratch = []
    staged = []
    original_workdir = shared.correctness_workdir
    original_scope = shared.adapter_build_scope

    def workdir(prefix):
        path = original_workdir(prefix)
        assert path.is_relative_to(worker_root)
        scratch.append(path)
        return path

    @contextlib.contextmanager
    def build_scope(path, expected):
        staged.append(Path(path))
        assert Path(path).is_relative_to(worker_root)
        assert {file.name: file.read_bytes() for file in Path(path).iterdir()} == files
        with original_scope(path, expected):
            yield

    monkeypatch.setattr(shared, "correctness_workdir", workdir)
    monkeypatch.setattr(shared, "adapter_build_scope", build_scope)
    worker = ModuleType("arctic_platform.common.deepspeed_worker")
    loaded = SimpleNamespace(model=_model())
    builds = []

    def build(*args, **kwargs):
        builds.append((args, kwargs))
        return loaded

    worker.build_model = build
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    monkeypatch.setattr(common, "deepspeed_worker", worker, raising=False)
    return SimpleNamespace(
        ref=ref,
        identity=identity,
        files=files,
        state=state,
        ray=ray,
        gets=gets,
        scratch=scratch,
        staged=staged,
        root=worker_root,
        worker=worker,
        loaded=loaded,
        builds=builds,
        original=build,
    )


@pytest.mark.parametrize("failure", [None, "build", "load", "body"])
def test_worker_adapter_transfer_stages_without_coordinator_path_and_always_cleans(worker_adapter_transfer, failure):
    control = worker_adapter_transfer
    original = control.original
    if failure == "build":

        def failed_build(*args, **kwargs):
            raise RuntimeError("worker builder failed")

        original = failed_build
        control.worker.build_model = original
    elif failure == "load":
        control.loaded.model = _model(dtype=torch.bfloat16)

    def execute():
        with shared.worker_adapter_scope(control.ref, control.identity) as yielded:
            assert yielded is None
            assert control.worker.build_model is not original
            result = control.worker.build_model("spec", parallel_groups="groups")
            assert result is control.loaded
            assert control.builds == [(("spec",), {"parallel_groups": "groups"})]
            assert {name: _bytes(value) for name, value in _installed(result.model).items()} == {
                canonical_name(name): _bytes(value) for name, value in control.state.items()
            }
            assert all(path.is_dir() for path in control.staged + control.scratch)
            if failure == "body":
                raise RuntimeError("worker body failed")

    if failure:
        with pytest.raises(ValueError if failure == "load" else RuntimeError, match="dtypes|worker"):
            execute()
    else:
        execute()
    assert control.gets == [control.ref]
    assert len(control.scratch) == len(control.staged) == 1
    assert not any(path.exists() for path in control.scratch + control.staged)
    assert not list(control.root.iterdir())
    assert control.worker.build_model is original


@pytest.mark.parametrize("failure", ["not-dict", "missing", "extra", "traversal", "nonbytes"])
def test_worker_adapter_transfer_refuses_invalid_envelope_before_staging(worker_adapter_transfer, failure):
    control = worker_adapter_transfer
    files = dict(control.files)
    if failure == "not-dict":
        files = list(files.items())
    elif failure == "missing":
        del files["adapter_model.safetensors"]
    elif failure in {"extra", "traversal"}:
        files["unexpected.bin" if failure == "extra" else "../escape"] = b"foreign"
    else:
        files["adapter_config.json"] = bytearray(files["adapter_config.json"])
    control.ray.get = lambda ref: files
    with pytest.raises(ValueError, match="only standard pack bytes"):
        with shared.worker_adapter_scope(control.ref, control.identity):
            pytest.fail("invalid transport must not enter worker scope")
    assert not control.scratch and not control.staged and not control.builds
    assert not control.root.exists()
    assert control.worker.build_model is control.original


@pytest.mark.parametrize(
    "failure",
    [
        "config",
        "tensor",
        "recipe",
        "recipe-missing",
        "invalid-config",
        "invalid-tensor",
    ],
)
def test_worker_adapter_transfer_rejects_corrupted_identity_and_cleans_staging(worker_adapter_transfer, failure):
    control = worker_adapter_transfer
    if failure == "config":
        control.files["adapter_config.json"] += b"\n"
    elif failure == "tensor":
        state = {name: tensor.clone() for name, tensor in control.state.items()}
        next(iter(state.values())).reshape(-1)[0] += 1
        control.files["adapter_model.safetensors"] = save(state)
    elif failure == "recipe":
        recipe = json.loads(control.files["correctness_recipe.json"])
        recipe["seed"] += 1
        control.files["correctness_recipe.json"] = json.dumps(recipe).encode()
    elif failure == "recipe-missing":
        del control.files["correctness_recipe.json"]
    elif failure == "invalid-config":
        control.files["adapter_config.json"] = b"{"
    else:
        control.files["adapter_model.safetensors"] = b"not-safetensors"
    with pytest.raises((ValueError, safetensors.SafetensorError)):
        with shared.worker_adapter_scope(control.ref, control.identity):
            pytest.fail("corrupted transport must not enter worker scope")
    assert len(control.scratch) == 1 and not control.scratch[0].exists()
    assert not control.staged and not control.builds
    assert not list(control.root.iterdir())
    assert control.worker.build_model is control.original


def test_worker_adapter_transfer_get_failure_propagates_without_staging(
    worker_adapter_transfer,
):
    control = worker_adapter_transfer

    def fail_get(ref):
        assert ref is control.ref
        raise RuntimeError("Ray transport failed")

    control.ray.get = fail_get
    with pytest.raises(RuntimeError, match="Ray transport failed"):
        with shared.worker_adapter_scope(control.ref, control.identity):
            pytest.fail("failed get must not enter worker scope")
    assert not control.scratch and not control.staged and not control.builds
    assert not control.root.exists()
    assert control.worker.build_model is control.original


@pytest.mark.parametrize("order", ["batched", "arm"])
def test_shared_adapter_autogenerated_runner_uses_one_real_pack_for_both_arms(
    generated_adapter_config,
    _prepared_reference_runner_prepared_runner,
    tmp_path,
    monkeypatch,
    order,
):
    from arctic_platform.correctness.harness import runner

    source = generated_adapter_config
    control = _prepared_reference_runner_prepared_runner
    control.spec.model.cache_path = str(source.path)
    control.cfg.training["peft_config"] = copy.deepcopy(source.config)
    original_generate = shared.generate_adapter
    original_running = runner.running_job
    generated = []
    payloads = []

    def generate(model_path, config, destination, *, seed):
        assert model_path == str(source.path)
        assert config == source.config and seed == runner.SEED
        assert destination == tmp_path / "run" / "initial-adapter"
        path = original_generate(model_path, config, destination, seed=seed)
        generated.append(path)
        return path

    @contextlib.contextmanager
    def running(session, payload):
        assert len(generated) == 1
        payloads.append(copy.deepcopy(payload))
        with original_running(session, payload) as job:
            yield job

    monkeypatch.setattr(shared, "generate_adapter", generate)
    monkeypatch.setattr(runner, "running_job", running)
    results = control.execute(order=order)
    assert len(results) == 4 and all(result.outcome is TestOutcome.PASS for result in results)
    assert len(generated) == 1 and len(payloads) == 2
    identity = shared.adapter_identity(generated[0], source.config)
    assert identity["generation"]["seed"] == runner.SEED
    assert identity["generation"]["initializer"] is True
    for payload in payloads:
        assert payload["model_name"] == str(source.path)
        assert payload["training_config"]["debug"]["shared_peft_adapter"] == {
            "path": str(generated[0]),
            "identity": identity,
        }
    assert len(control.recipes) == 4
    assert {arm for _, arm, _ in control.recipes} == {"gas1", "gas4"}
    assert all(recipe["initialization"] == identity for _, _, recipe in control.recipes)
    assert all(ctx.reference_adapter_path == generated[0] for ctx in control.contexts)
    assert [event for event in control.events if event[0] == "target"] == [
        ("target", "gas1"),
        ("target", "gas4"),
    ]
    assert not control.published


@pytest.mark.parametrize("changed", ["config", "tensor", "recipe"])
def test_shared_adapter_generated_onboard_cache_reuses_exact_pack_and_refuses_mutation(
    generated_adapter_config, tmp_path, monkeypatch, changed
):
    from arctic_platform.correctness.onboarding import onboard

    source = generated_adapter_config
    config = SimpleNamespace(training={"peft_config": copy.deepcopy(source.config)})
    args = SimpleNamespace(peft_adapter=None)
    monkeypatch.setenv(storage.CORRECTNESS_TMPDIR_ENV, str(tmp_path / "onboard-cache"))
    original_generate = shared.generate_adapter
    generated = []

    def generate(model_path, peft_config, destination):
        assert model_path == source.path and peft_config == source.config
        path = original_generate(model_path, peft_config, destination)
        generated.append(path)
        return path

    monkeypatch.setattr(shared, "generate_adapter", generate)
    assert onboard._shared_adapter(args, config) is None
    assert not generated
    path = onboard._shared_adapter(args, config, source.path)
    identity = shared.adapter_identity(path, source.config)
    before = shared.adapter_files(path, identity)
    assert onboard._shared_adapter(args, config, source.path) == path
    alias = source.path / ".." / source.path.name
    assert onboard._shared_adapter(args, config, alias) == path
    assert generated == [path]
    key = str(source.path.resolve())
    assert args._generated_shared_adapters == {key: path}
    assert args._generated_shared_adapter_identities == {key: identity}
    assert shared.adapter_files(path, identity) == before
    if changed == "config":
        file = path / "adapter_config.json"
        file.write_bytes(file.read_bytes() + b"\n")
    elif changed == "tensor":
        tensors = {name: tensor.clone() for name, tensor in load_file(str(path / "adapter_model.safetensors")).items()}
        next(iter(tensors.values())).reshape(-1)[0] += 1
        save_file(tensors, str(path / "adapter_model.safetensors"))
    else:
        file = path / "correctness_recipe.json"
        recipe = json.loads(file.read_text())
        recipe["dependencies"]["peft"] = "changed-after-onboarding"
        file.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="shared adapter changed during onboarding"):
        onboard._shared_adapter(args, config, source.path)
    assert generated == [path]
    assert args._generated_shared_adapter_identities == {key: identity}


def test_shared_adapter_generated_hf_autocast_loads_fp32_factors_on_bfloat16_cpu_base(
    generated_adapter_config, tmp_path, monkeypatch
):
    from arctic_platform.correctness.reference.hf_single_gpu import build_engine

    source = generated_adapter_config
    pack = shared.generate_adapter(source.path, source.config, tmp_path / "hf-adapter")
    identity = shared.adapter_identity(pack, source.config)
    state = load_file(str(pack / "adapter_model.safetensors"))
    assert all(tensor.dtype == torch.float32 for tensor in state.values())
    calls = []
    with torch.random.fork_rng(devices=[]):
        base_config = Qwen3Config.from_pretrained(source.path)
        base = Qwen3ForCausalLM(base_config).to(dtype=torch.bfloat16, device="cpu")
        frozen = [(parameter, _bytes(parameter)) for parameter in base.parameters()]
        target_base = Qwen3ForCausalLM(base_config).to(dtype=torch.bfloat16, device="cpu")
        target = peft.get_peft_model(
            target_base,
            peft.LoraConfig(**{key: value for key, value in source.config.items() if key != "peft_type"}),
        )
        shared.initialize_adapter(target, pack, identity)
        original_load = peft.PeftModel.from_pretrained

        def load_base(model_path, *, dtype, attn_implementation):
            assert model_path == str(source.path)
            assert dtype == torch.bfloat16 and attn_implementation == "sdpa"
            calls.append("base")
            return base

        def load_adapter(model, model_path, **kwargs):
            assert model is base and model_path == str(pack)
            assert kwargs == {"is_trainable": True}
            calls.append("adapter-default-autocast")
            return original_load(model, model_path, **kwargs)

        monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", load_base)
        monkeypatch.setattr(peft.PeftModel, "from_pretrained", load_adapter)
        precision = torch.get_float32_matmul_precision()
        try:
            engine = build_engine(
                str(source.path),
                dtype="bfloat16",
                device="cpu",
                peft_config=source.config,
                peft_adapter_path=str(pack),
            )
        finally:
            torch.set_float32_matmul_precision(precision)
    assert calls == ["base", "adapter-default-autocast"]
    assert engine.device == "cpu" and engine.hidden_size == 16
    assert engine.model.training and engine.model.is_gradient_checkpointing
    assert all(parameter.dtype == torch.bfloat16 and _bytes(parameter) == raw for parameter, raw in frozen)
    installed = _installed(engine.model)
    assert set(installed) == {canonical_name(name) for name in state} == set(_installed(target))
    for name, parameter in installed.items():
        assert parameter.device.type == "cpu" and parameter.dtype == torch.float32
        assert _bytes(parameter) == _bytes(state[name]) == _bytes(_installed(target)[name])
    assert shared.adapter_identity(pack, source.config) == identity
    assert {path.name for path in source.path.iterdir()} == {"config.json"}


@pytest.mark.parametrize(
    "initialization,binds",
    [
        ("external-model", False),
        ({"schema": "arctic-correctness-lora-init-v1"}, False),
        ({"schema": "shared-peft-adapter-v1"}, True),
    ],
)
def test_shared_adapter_initialization_source_binding_leaves_other_record_recipes_unchanged(
    monkeypatch, initialization, binds
):
    from arctic_platform.correctness.harness import reference_records

    ctx = SimpleNamespace(
        config_id="synthetic-shared-adapter",
        cfg=SimpleNamespace(gpu_type="H200"),
        spec=SimpleNamespace(config_checksum="c" * 64),
    )
    original_file_identity = reference_records.file_identity
    helper_pin = {"sha256": "a" * 64, "size": 1}
    reads = []

    def file_identity(path):
        if path.name == "shared_lora_adapter.py":
            reads.append(path)
            return copy.deepcopy(helper_pin)
        return original_file_identity(path)

    monkeypatch.setattr(reference_records, "file_identity", file_identity)
    recipe = {"initialization": initialization}
    before = reference_records.workload_identity(ctx, "single-step-optimizer", recipe)
    helper_pin["sha256"] = "b" * 64
    after = reference_records.workload_identity(ctx, "single-step-optimizer", recipe)
    assert ("harness/shared_lora_adapter.py" in before["reference_sources"]) is binds
    assert len(reads) == (2 if binds else 0)
    if binds:
        assert before != after
        assert before["reference_sources"]["harness/shared_lora_adapter.py"]["sha256"] == "a" * 64
        assert after["reference_sources"]["harness/shared_lora_adapter.py"]["sha256"] == "b" * 64
        del before["reference_sources"]["harness/shared_lora_adapter.py"]
        del after["reference_sources"]["harness/shared_lora_adapter.py"]
    assert before == after


@pytest.mark.parametrize("failure", [None, "client", "body", "shutdown", "put", "admission"])
def test_shared_adapter_running_job_ray_handoff_precedes_client_and_restores_overlay(
    generated_adapter_config, tmp_path, monkeypatch, failure
):
    from arctic_platform import common
    from arctic_platform.correctness.harness import dss_driver
    from arctic_platform.correctness.harness import optimizer_capture

    source = generated_adapter_config
    pack = shared.generate_adapter(source.path, source.config, tmp_path / "head-pack")
    identity = shared.adapter_identity(pack)
    files = shared.adapter_files(pack, identity)
    events = []
    active = []
    ref = object()
    raised = RuntimeError(f"injected {failure} failure")
    worker = ModuleType("arctic_platform.common.deepspeed_worker")
    original_worker = object()
    capture_worker = object()
    worker.DeepSpeedWorker = original_worker
    server = ModuleType("arctic_platform.common.ray_server")
    compatibility = ModuleType("arctic_platform.rl.ray_server")

    class ServerState:
        pass

    class Server:
        pass

    for module in (server, compatibility):
        module.DeepSpeedWorker = original_worker
        module.ArcticRLRayServerState = ServerState
        module.ArcticRLRayServer = Server
    ray = ModuleType("ray")
    for module in (worker, server, compatibility, ray):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(common, "deepspeed_worker", worker, raising=False)
    monkeypatch.setattr(common, "ray_server", server, raising=False)
    monkeypatch.setattr(optimizer_capture, "_optimizer_capture_worker_class", lambda: capture_worker)

    def require_overlay():
        assert active == ["runtime"]
        assert worker.DeepSpeedWorker is capture_worker
        for module in (server, compatibility):
            assert module.DeepSpeedWorker is capture_worker
            assert module.ArcticRLRayServerState is not ServerState
            assert module.ArcticRLRayServer is not Server

    @contextlib.contextmanager
    def runtime(slots):
        assert slots == 1
        events.append("runtime-enter")
        active.append("runtime")
        try:
            yield
        finally:
            assert worker.DeepSpeedWorker is original_worker
            active.pop()
            events.append("runtime-exit")

    def put(transferred):
        require_overlay()
        events.append("put")
        assert transferred == files and all(isinstance(value, bytes) for value in transferred.values())
        if failure == "put":
            raise raised
        return ref

    ray.put = put
    client_configs = []

    class Client:
        def __init__(self, config):
            require_overlay()
            events.append("client-start")
            client_configs.append(config)
            expected = {"object_ref": ref, "identity": identity}
            assert config.training.ds_worker_config["debug"]["shared_peft_adapter"] == expected
            converted = config.to_onprem("training")
            assert converted["ds_worker_config"]["debug"]["shared_peft_adapter"] == expected
            assert "path" not in converted["ds_worker_config"]["debug"]["shared_peft_adapter"]
            assert "training_config" not in converted
            if failure == "client":
                raise raised
            self.jobs = SimpleNamespace(training="owned-training-child")

        def shutdown(self):
            require_overlay()
            events.append("client-shutdown")
            if failure == "shutdown":
                raise raised

    monkeypatch.setattr(dss_driver, "_gateway_runtime", runtime)
    monkeypatch.setattr(dss_driver, "ArcticClient", Client)
    payload = dss_driver.build_payload({"n_gpus": 1, "peft_config": source.config}, str(source.path), 42)
    payload["training_config"]["debug"]["shared_peft_adapter"] = {
        "path": str(pack),
        "identity": identity,
    }
    before = copy.deepcopy(payload)
    if failure == "admission":
        config_path = pack / "adapter_config.json"
        config_path.write_bytes(config_path.read_bytes() + b"\n")

    def execute():
        with dss_driver.running_job(dss_driver.GatewaySession(tmp_path / "run", 1), payload) as job:
            require_overlay()
            events.append("body")
            assert job.job_id == "owned-training-child"
            if failure == "body":
                raise raised

    if failure == "admission":
        with pytest.raises(ValueError, match="changed before transfer"):
            execute()
    elif failure:
        error_type = dss_driver.UnsafeCleanupError if failure == "shutdown" else RuntimeError
        with pytest.raises(error_type) as caught:
            execute()
        if failure == "shutdown":
            assert caught.value.__cause__ is raised
        else:
            assert caught.value is raised
    else:
        execute()
    expected_events = ["runtime-enter"]
    if failure != "admission":
        expected_events.append("put")
    if failure not in {"admission", "put"}:
        expected_events.append("client-start")
    if failure not in {"admission", "put", "client"}:
        expected_events.extend(["body", "client-shutdown"])
    assert events == [*expected_events, "runtime-exit"]
    assert not active
    assert payload == before
    assert len(client_configs) == (0 if failure in {"admission", "put"} else 1)
    assert worker.DeepSpeedWorker is original_worker
    for module in (server, compatibility):
        assert module.DeepSpeedWorker is original_worker
        assert module.ArcticRLRayServerState is ServerState
        assert module.ArcticRLRayServer is Server


@pytest.mark.parametrize("failure", [None, "initialize", "load"])
def test_shared_adapter_capture_initializer_enters_worker_scope_before_base_and_restores(
    worker_adapter_transfer, tmp_path, monkeypatch, failure
):
    from arctic_platform.correctness.harness import dss_driver
    from arctic_platform.correctness.harness import optimizer_capture

    control = worker_adapter_transfer
    events = []
    raised = RuntimeError("base initializer failed")
    if failure == "load":
        control.loaded.model = _model(dtype=torch.bfloat16)
    payload = dss_driver.build_payload({"n_gpus": 1}, "config-only-base", 42)
    job_config = dss_driver._training_client_config(dss_driver.GatewaySession(tmp_path / "run", 1), payload).to_onprem(
        "training"
    )
    job_config["ds_worker_config"]["debug"]["shared_peft_adapter"] = {
        "object_ref": control.ref,
        "identity": copy.deepcopy(control.identity),
    }
    expected_identity = copy.deepcopy(job_config["ds_worker_config"]["debug"]["shared_peft_adapter"]["identity"])

    class Worker:
        def initialize(self, master_addr, actual_config):
            assert master_addr == "owned-master" and actual_config is job_config
            assert control.gets == [control.ref]
            assert len(control.scratch) == len(control.staged) == 1
            assert control.worker.build_model is not control.original
            events.append("base-initialize")
            loaded = control.worker.build_model("spec", parallel_groups="groups")
            assert loaded is control.loaded
            assert {name: _bytes(value) for name, value in _installed(loaded.model).items()} == {
                canonical_name(name): _bytes(value) for name, value in control.state.items()
            }
            events.append("exact-loaded")
            if failure == "initialize":
                raise raised
            return True

    control.worker.DeepSpeedWorker = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Worker))
    control.ray.remote = lambda cls: cls
    monkeypatch.setattr(optimizer_capture, "_OPTIMIZER_CAPTURE_WORKER", None)
    worker = optimizer_capture._optimizer_capture_worker_class()()
    if failure:
        with pytest.raises(ValueError if failure == "load" else RuntimeError) as caught:
            worker.initialize("owned-master", job_config)
        if failure == "initialize":
            assert caught.value is raised
    else:
        assert worker.initialize("owned-master", job_config) is True
    assert events == (["base-initialize"] if failure == "load" else ["base-initialize", "exact-loaded"])
    assert control.worker.build_model is control.original
    assert control.gets == [control.ref] and len(control.scratch) == len(control.staged) == 1
    assert not any(path.exists() for path in control.scratch + control.staged)
    assert not list(control.root.iterdir())
    assert job_config["ds_worker_config"]["debug"]["shared_peft_adapter"]["object_ref"] is control.ref
    assert job_config["ds_worker_config"]["debug"]["shared_peft_adapter"]["identity"] == expected_identity
