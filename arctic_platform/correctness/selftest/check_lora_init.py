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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only deterministic LoRA recipe controls; no gateway, Ray, or base weights."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.harness import deterministic_lora
from arctic_platform.correctness.harness import lora_init
from arctic_platform.correctness.harness import workdir

A = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
B = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"


def _specs(dtype="bfloat16"):
    return [lora_init.LoraTensorSpec(A, (4, 16), dtype), lora_init.LoraTensorSpec(B, (12, 4), dtype)]


def _tensors(specs, seed=1234):
    return {spec.name: lora_init.generate_tensor(spec, seed) for spec in specs}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_private_cpu_generation_matches_declared_kaiming_and_zero_distribution(dtype):
    a, b = _specs(dtype)
    expected = torch.empty(a.shape, device="cpu", dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(lora_init.tensor_seed(1234, A))
    torch.nn.init.kaiming_uniform_(
        expected, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu", generator=generator
    )
    actual = lora_init.generate_tensor(a, 1234)
    assert actual.device.type == "cpu"
    assert actual.dtype == getattr(torch, dtype)
    assert torch.equal(actual, expected.to(dtype=actual.dtype))
    assert float(actual.abs().max()) <= 1 / math.sqrt(a.shape[1])
    assert torch.count_nonzero(lora_init.generate_tensor(b, 1234)).item() == 0


def test_unrelated_draws_and_traversal_order_do_not_change_values_or_global_rng():
    specs = _specs()
    state = torch.random.get_rng_state().clone()
    first = _tensors(specs)
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.rand(257)
    state = torch.random.get_rng_state().clone()
    second = _tensors(list(reversed(specs)))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert lora_init.build_manifest(specs, 1234) == lora_init.build_manifest(reversed(specs), 1234)


def test_sha256_seed_is_stable_and_changes_with_name_or_global_seed():
    recipe = {"domain": "arctic-correctness-lora-tensor-seed-v1", "global_seed": 1234, "name": A}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).digest()
    assert lora_init.tensor_seed(1234, A) == int.from_bytes(digest[:8], "big") & (2**63 - 1)
    assert lora_init.tensor_seed(1234, A) != lora_init.tensor_seed(1234, A.replace("q_proj", "k_proj"))
    assert lora_init.tensor_seed(1234, A) != lora_init.tensor_seed(1235, A)


@pytest.mark.parametrize("seed", [-1, 2**63, True, "1234"])
def test_invalid_global_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="global seed"):
        lora_init.generate_tensor(_specs()[0], seed)


def test_fresh_process_reconstruction_ignores_global_random_draws():
    expected = lora_init.build_manifest(_specs(), 1234)
    code = (
        "import json, torch; "
        "from arctic_platform.correctness.harness.lora_init import LoraTensorSpec, build_manifest; "
        "torch.manual_seed(98); torch.rand(299); "
        f"specs=[LoraTensorSpec({A!r}, (4,16), 'bfloat16'), LoraTensorSpec({B!r}, (12,4), 'bfloat16')]; "
        "print(json.dumps(build_manifest(specs,1234),sort_keys=True))"
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
        [sys.executable, "-c", code], env=environment, text=True, capture_output=True, check=True, timeout=60
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == expected


def test_canonical_names_retain_containers_and_strip_only_supported_segments():
    raw = A.replace(".layers.0.", ".layers.0._checkpoint_wrapped_module.").replace(
        ".lora_A.weight", ".lora_A.default.weight"
    )
    assert lora_init.canonical_name(raw) == A
    assert lora_init.canonical_name(A.replace("base_model.model.", "", 1)) != A
    assert lora_init.canonical_name(A.replace(".model.layers.", ".language_model.layers.")) != A


@pytest.mark.parametrize(
    "name", ["", "model..lora_A.weight", "model.lora_A.other.weight", "model.weight", "model x.lora_A.weight"]
)
def test_unsupported_or_ambiguous_names_are_rejected(name):
    with pytest.raises(ValueError):
        lora_init.canonical_name(name)


@pytest.mark.parametrize("shape", [(0, 16), (4,), (4, 16, 2), (True, 16), [4, 16]])
def test_invalid_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="two-dimensional"):
        lora_init.LoraTensorSpec(A, shape, "float32")


def test_unsupported_dtype_and_incomplete_or_colliding_pairs_are_rejected():
    with pytest.raises(ValueError, match="dtype"):
        lora_init.LoraTensorSpec(A, (4, 16), "float64")
    a, b = _specs()
    for specs, message in [
        ([], "complete A/B"),
        ([a], "partner"),
        ([a, b, a], "duplicate"),
        ([a, lora_init.LoraTensorSpec(B, (12, 5), b.dtype)], "rank or dtype"),
        ([a, lora_init.LoraTensorSpec(B, b.shape, "float32")], "rank or dtype"),
    ]:
        with pytest.raises(ValueError, match=message):
            lora_init.build_manifest(specs, 1234)


def test_manifest_contains_only_compact_json_recipe_and_exact_hashes():
    manifest = lora_init.build_manifest(_specs(), 1234)
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["generation_dtype"] == "float32"
    assert manifest["generator"]["device"] == "cpu"
    assert manifest["generator"]["torch_version"] == str(torch.__version__)
    assert manifest["tensors"][0]["distribution"]["fan_in"] == 16
    assert all(len(entry["sha256"]) == 64 for entry in manifest["tensors"])
    assert lora_init.manifest_specs(manifest) == _specs()


@pytest.mark.parametrize(
    "field", ["recipe_version", "global_seed", "generation_dtype", "identity_sha256", "tensor_hash", "torch_version"]
)
def test_corrupt_or_incompatible_manifest_cannot_be_reused(field):
    manifest = lora_init.build_manifest(_specs(), 1234)
    if field == "tensor_hash":
        manifest["tensors"][0]["sha256"] = "0" * 64
    elif field == "torch_version":
        manifest["generator"]["torch_version"] = "different-runtime"
    else:
        manifest[field] = "changed"
    with pytest.raises(ValueError):
        lora_init.manifest_specs(manifest)


def test_hash_covers_name_shape_dtype_and_signed_zero_bytes():
    zeros = torch.zeros((4, 16), dtype=torch.float32)
    digest = lora_init.tensor_sha256(A, zeros)
    assert digest != lora_init.tensor_sha256(A.replace("q_proj", "k_proj"), zeros)
    assert digest != lora_init.tensor_sha256(A, zeros.reshape(8, 8))
    assert digest != lora_init.tensor_sha256(A, zeros.bfloat16())
    negative_zero = zeros.clone()
    negative_zero[0, 0] = -0.0
    assert torch.equal(zeros, negative_zero)
    assert digest != lora_init.tensor_sha256(A, negative_zero)


def test_exact_verification_accepts_only_wrapper_default_aliases_and_full_global_shapes():
    specs = _specs()
    manifest = lora_init.build_manifest(specs, 1234)
    tensors = _tensors(specs)
    raw = {name.replace(".weight", ".default.weight"): tensor for name, tensor in tensors.items()}
    receipt = lora_init.verify_tensors(raw, manifest)
    assert receipt["identity_sha256"] == manifest["identity_sha256"]
    shards = list(tensors[A].chunk(2, dim=1))
    reconstructed = {**tensors, A: torch.cat(shards, dim=1)}
    lora_init.verify_tensors(reconstructed, manifest)
    with pytest.raises(ValueError, match="shape differs"):
        lora_init.verify_tensors({**tensors, A: shards[0]}, manifest)
    with pytest.raises(ValueError, match="coverage"):
        lora_init.verify_tensors(
            {name.removeprefix("base_model.model."): value for name, value in tensors.items()}, manifest
        )


@pytest.mark.parametrize("kind", ["missing", "extra", "collision", "shape", "nonfinite", "different", "signed_zero"])
def test_verification_rejects_coverage_shape_and_value_changes(kind):
    specs = _specs()
    manifest = lora_init.build_manifest(specs, 1234)
    tensors = _tensors(specs)
    if kind == "missing":
        del tensors[B]
    elif kind == "extra":
        tensors[A.replace("q_proj", "k_proj")] = tensors[A]
    elif kind == "collision":
        tensors[A.replace(".weight", ".default.weight")] = tensors[A]
    elif kind == "shape":
        tensors[A] = tensors[A].reshape(8, 8)
    elif kind == "nonfinite":
        tensors[A][0, 0] = float("nan")
    elif kind == "different":
        tensors[A][0, 0] = 0.5
    else:
        tensors[B][0, 0] = -0.0
    with pytest.raises(ValueError):
        lora_init.verify_tensors(tensors, manifest)


def test_only_explicit_exact_float32_widening_is_accepted():
    specs = _specs()
    manifest = lora_init.build_manifest(specs, 1234)
    tensors = {name: tensor.float() for name, tensor in _tensors(specs).items()}
    with pytest.raises(ValueError, match="dtype"):
        lora_init.verify_tensors(tensors, manifest)
    lora_init.verify_tensors(tensors, manifest, allow_exact_float32_widening=True)
    tensors[A][0, 0] += 2**-20
    with pytest.raises(ValueError, match="bytes"):
        lora_init.verify_tensors(tensors, manifest, allow_exact_float32_widening=True)
    fp32_specs = _specs("float32")
    with pytest.raises(ValueError, match="dtype"):
        lora_init.verify_tensors(
            {name: tensor.bfloat16() for name, tensor in _tensors(fp32_specs).items()},
            lora_init.build_manifest(fp32_specs, 1234),
            allow_exact_float32_widening=True,
        )


def test_non_lora_is_noop_without_model_or_storage_access(monkeypatch):
    cfg = SimpleNamespace(training={})
    monkeypatch.setattr(
        deterministic_lora, "require_data_fast_path", lambda *a, **k: pytest.fail("non-LoRA storage accessed")
    )
    assert deterministic_lora.materialize_adapter(cfg, "/absent/model", "/code/not-fast") is None
    payload = {"training_config": {}, "model_name": "Qwen/Qwen3-8B"}
    assert deterministic_lora.bind_payload(payload, None) is payload


def test_target_binding_never_invents_public_adapter_or_resume_fields():
    payload = {"training_config": {"peft_config": {"r": 4}}, "model_name": "Qwen/Qwen3-8B"}
    before = deepcopy(payload)
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="public training adapter"):
        deterministic_lora.bind_payload(payload, Path("/data-fast/tmp/adapter"))
    assert payload == before
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="requires"):
        deterministic_lora.bind_payload(payload, None)


@pytest.fixture
def local_adapter_fixture(tmp_path, monkeypatch):
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
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=from_pretrained), SyntheticModel=SyntheticModel),
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
    return cfg, source, root / "runtime"


def test_temporary_pack_uses_meta_shapes_and_preserves_rng_and_identity(local_adapter_fixture):
    cfg, source, scratch = local_adapter_fixture
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


def test_temporary_pack_refuses_non_fast_storage_before_topology_import(local_adapter_fixture):
    cfg, source, scratch = local_adapter_fixture
    with pytest.raises(ValueError, match="/data-fast"):
        deterministic_lora.materialize_adapter(cfg, source, scratch.parent.parent / "outside")


@pytest.mark.parametrize("kind", ["metadata", "adapter_config", "tensor"])
def test_pack_identity_refuses_mutation(local_adapter_fixture, kind):
    from safetensors.torch import load_file
    from safetensors.torch import save_file

    cfg, source, scratch = local_adapter_fixture
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


def test_real_peft_tiny_qwen_serialization_and_loaded_initial_values(tmp_path, monkeypatch):
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
        training={"peft_config": {"peft_type": "Lora", "r": 4, "target_modules": ["q_proj", "v_proj"]}},
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
    assert all(".default." not in name for name in state)
    assert all(name.startswith("base_model.model.model.layers.0.") for name in state)
    with torch.random.fork_rng(devices=[]):
        base = Qwen3ForCausalLM(config).to(device="cpu", dtype=torch.bfloat16)
        loaded = peft.PeftModel.from_pretrained(base, output, is_trainable=True, local_files_only=True)
    actual = {name: parameter for name, parameter in loaded.named_parameters() if parameter.requires_grad}
    receipt = lora_init.verify_tensors(actual, identity["initialization"], allow_exact_float32_widening=True)
    assert len(receipt["verified_tensors"]) == 4
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError):
        deterministic_lora.bind_payload({"training_config": cfg.training}, output)
