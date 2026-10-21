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

"""Static correctness selftests: lora training."""


from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.correctness.harness import cortex as cortex_fixture
from arctic_platform.correctness.harness import cortex_optimizer as cortex_optimizer_capture
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness import lora as deterministic_lora
from arctic_platform.correctness.harness import lora as lora_init
from arctic_platform.correctness.harness import lora as lora_save0
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import storage as workdir
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.reference import process_determinism
from arctic_platform.correctness.reference.hf_single_gpu import build_engine
from arctic_platform.correctness.reference.qwen4_reference import load_qwen4_reference
from arctic_platform.debug import lora as lora_initialization
from arctic_platform.debug.lora import normalize_lora_initialization
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.model.patches.peft import apply_peft_patch
from arctic_platform.model.weight_export import save_peft_adapters
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

from . import check_reference_records as reference_selftests
from .check_reference_records import _prepared_reference_runner_STEP_CHECKS
from .helpers import _synthetic_config_spec

_prepared_reference_runner_prepared_runner = reference_selftests._prepared_reference_runner_prepared_runner


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
def test_lora_init_private_cpu_generation_matches_declared_kaiming_and_zero_distribution(dtype):
    a, b = _lora_init_specs(dtype)
    expected = torch.empty(a.shape, device="cpu", dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(lora_init.tensor_seed(1234, _lora_init_A))
    torch.nn.init.kaiming_uniform_(
        expected, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu", generator=generator
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
    recipe = {"domain": "arctic-correctness-lora-tensor-seed-v1", "global_seed": 1234, "name": _lora_init_A}
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
        [sys.executable, "-c", code], env=environment, text=True, capture_output=True, check=True, timeout=60
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
    "name", ["", "model..lora_A.weight", "model.lora_A.other.weight", "model.weight", "model x.lora_A.weight"]
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
        ([a, lora_init.LoraTensorSpec(_lora_init_B, (12, 5), b.dtype)], "rank or dtype"),
        ([a, lora_init.LoraTensorSpec(_lora_init_B, b.shape, "float32")], "rank or dtype"),
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
    "field", ["recipe_version", "global_seed", "generation_dtype", "identity_sha256", "tensor_hash", "torch_version"]
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
            {name.removeprefix("base_model.model."): value for name, value in tensors.items()}, manifest
        )


@pytest.mark.parametrize("kind", ["missing", "extra", "collision", "shape", "nonfinite", "different", "signed_zero"])
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
        deterministic_lora, "require_data_fast_path", lambda *a, **k: pytest.fail("non-LoRA storage accessed")
    )
    assert deterministic_lora.materialize_adapter(cfg, "/absent/model", "/code/not-fast") is None
    payload = {"training_config": {}, "model_name": "Qwen/Qwen3-8B"}
    assert deterministic_lora.bind_payload(payload, None) is payload


def test_lora_init_target_binding_never_invents_public_adapter_or_resume_fields():
    payload = {"training_config": {"peft_config": {"r": 4}}, "model_name": "Qwen/Qwen3-8B"}
    before = deepcopy(payload)
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="public training adapter"):
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
    return (cfg, source, root / "runtime")


def test_lora_init_temporary_pack_uses_meta_shapes_and_preserves_rng_and_identity(_lora_init_local_adapter_fixture):
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


def test_lora_init_temporary_pack_refuses_non_fast_storage_before_topology_import(_lora_init_local_adapter_fixture):
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


_lora_save0_OPTIONS = {"recipe_version": "arctic-correctness-lora-init-v1", "global_seed": 42, "dtype": "float32"}


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
    model = apply_peft(_lora_save0_tiny_model(), {**_lora_save0_TOPOLOGY, "deterministic_init": _lora_save0_OPTIONS})
    root = tmp_path / "checkpoint"
    destination = root / "global_step0"
    save_peft_adapters(model, str(destination))
    metadata = _lora_save0_metadata(model)
    response = {"checkpoint_path": str(root), "checkpoint_tag": "global_step0", "version": 0, "stage_path": None}
    return (root, response, metadata, model)


def _lora_save0_payload():
    return {
        "job_type": "training",
        "model_name": "/data-fast/base-models/synthetic/Qwen3-8B-4L",
        "seed": 42,
        "training_config": {
            "n_gpus": 4,
            "sp_size": 1,
            "ep_size": 1,
            "ds_config": {"zero_optimization": {"stage": 2}},
            "peft_config": deepcopy(_lora_save0_TOPOLOGY),
        },
    }


def test_lora_save0_absent_option_preserves_normalized_topology_and_default_peft_behavior():
    assert "deterministic_init" not in normalize_lora_peft_config(_lora_save0_TOPOLOGY)
    set_seed(17)
    baseline = apply_peft(_lora_save0_tiny_model(), _lora_save0_TOPOLOGY)
    set_seed(17)
    repeated = apply_peft(_lora_save0_tiny_model(), _lora_save0_TOPOLOGY)
    for (name, actual), (other_name, expected) in zip(
        baseline.named_parameters(), repeated.named_parameters(), strict=True
    ):
        assert name == other_name
        torch_assert_equal(actual, expected)


@pytest.mark.parametrize(
    "options",
    [
        None,
        {},
        {**_lora_save0_OPTIONS, "unexpected": True},
        {**_lora_save0_OPTIONS, "recipe_version": "unknown"},
        {**_lora_save0_OPTIONS, "global_seed": True},
        {**_lora_save0_OPTIONS, "global_seed": -1},
        {**_lora_save0_OPTIONS, "global_seed": 2**63},
        {**_lora_save0_OPTIONS, "dtype": []},
        {**_lora_save0_OPTIONS, "dtype": "float64"},
    ],
)
def test_lora_save0_opt_in_schema_rejects_unsupported_values(options):
    with pytest.raises(ValueError):
        normalize_lora_initialization(options)


def test_lora_save0_normalizer_preserves_validated_option_without_mutating_input():
    original = {**_lora_save0_TOPOLOGY, "deterministic_init": deepcopy(_lora_save0_OPTIONS)}
    expected = deepcopy(original)
    normalized = normalize_lora_peft_config(original)
    assert original == expected == normalized
    assert normalized["deterministic_init"] is not original["deterministic_init"]
    with pytest.raises(ValueError, match="dense"):
        normalize_lora_peft_config({**original, "target_parameters": ["w1"]})


def test_lora_save0_real_peft_constructor_strips_option_and_matches_independent_reference_before_optimizer():
    model = apply_peft(_lora_save0_tiny_model(), {**_lora_save0_TOPOLOGY, "deterministic_init": _lora_save0_OPTIONS})
    assert "deterministic_init" not in model.peft_config["default"].to_dict()
    metadata = _lora_save0_metadata(model)
    tensors = {name: value for name, value in model.named_parameters() if value.requires_grad}
    lora_init.verify_tensors(tensors, metadata["initialization"])
    optimizer = torch.optim.AdamW(tensors.values(), lr=0.01)
    assert not optimizer.state
    lora_init.verify_tensors(tensors, metadata["initialization"])


def test_lora_save0_public_sdk_and_ap_loader_forward_initialization_option():
    from arctic_platform.model.dense_spec import dense_model_spec

    model = _lora_save0_tiny_model()
    expected = _lora_save0_metadata(
        apply_peft(_lora_save0_tiny_model(), {**_lora_save0_TOPOLOGY, "deterministic_init": _lora_save0_OPTIONS})
    )
    payload = lora_save0.bind_initialization(_lora_save0_payload(), expected)
    session = dss_driver.GatewaySession(
        workdir=Path("/data-fast/tmp/not-used-sdk-control"),
        slots=4,
        backend=CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev"),
    )
    config = dss_driver._training_client_config(session, payload)
    sub_jobs = config.to_cortex()
    assert len(sub_jobs) == 1 and sub_jobs[0]["job_type"] == "training"
    public_config = sub_jobs[0]["training_config"]
    assert public_config["peft_config"]["deterministic_init"] == _lora_save0_OPTIONS
    spec = dense_model_spec(
        model_path=payload["model_name"],
        dtype="float32",
        attn_implementation="eager",
        sequence_parallel=1,
        liger=False,
        training_options=public_config,
    )
    assert spec.patches.peft["deterministic_init"] == _lora_save0_OPTIONS
    actual = apply_peft_patch(model, LoaderContext(spec=spec))
    lora_init.verify_tensors(
        [(name, parameter) for name, parameter in actual.named_parameters() if parameter.requires_grad],
        expected["initialization"],
    )


def test_lora_save0_initializer_preserves_parameter_objects_and_rng_and_ignores_traversal_order():
    model = apply_peft(_lora_save0_tiny_model(), _lora_save0_TOPOLOGY)
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = torch.random.get_rng_state().clone()
    lora_initialization.initialize_lora_factors(model, _lora_save0_OPTIONS)
    torch_assert_equal(state, torch.random.get_rng_state())
    expected = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    proxy = SimpleNamespace(named_parameters=lambda **kwargs: reversed(list(parameters.items())))
    lora_initialization.initialize_lora_factors(proxy, _lora_save0_OPTIONS)
    for name, parameter in model.named_parameters():
        if name in parameters:
            assert parameter is parameters[name]
            torch_assert_equal(parameter, expected[name])
    tree = ast.parse(Path(lora_initialization.__file__).read_text())
    assert not any(
        (isinstance(node, ast.ImportFrom) and "correctness" in (node.module or "") for node in ast.walk(tree))
    )


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_lora_save0_private_initializer_matches_exact_reference_generation_dtype_and_cast(dtype):
    model = apply_peft(_lora_save0_tiny_model(), _lora_save0_TOPOLOGY)
    model.to(dtype=getattr(torch, dtype))
    lora_initialization.initialize_lora_factors(model, {**_lora_save0_OPTIONS, "dtype": dtype})
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    manifest = lora_init.build_manifest(
        [
            lora_init.LoraTensorSpec(lora_init.canonical_name(name), tuple(value.shape), dtype)
            for name, value in parameters
        ],
        42,
    )
    lora_init.verify_tensors(parameters, manifest)


@pytest.mark.parametrize("ep,sp", [(2, 1), (1, 2)])
def test_lora_save0_patcher_rejects_unreviewed_parallelism_before_wrapping(ep, sp):
    ctx = SimpleNamespace(
        spec=SimpleNamespace(
            patches=SimpleNamespace(peft={**_lora_save0_TOPOLOGY, "deterministic_init": _lora_save0_OPTIONS}),
            parallelism=SimpleNamespace(expert_parallel=ep, sequence_parallel=sp),
        )
    )
    with pytest.raises(ValueError, match="EP1/SP1"):
        apply_peft_patch(_lora_save0_tiny_model(), ctx)


@pytest.mark.parametrize("failure", ["dtype", "sharded", "missing", "alias", "tied"])
def test_lora_save0_initializer_rejects_unsupported_storage_before_any_parameter_is_changed(failure):
    model = apply_peft(_lora_save0_tiny_model(), _lora_save0_TOPOLOGY)
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    original = [parameter.detach().clone() for _, parameter in parameters]
    options = {**_lora_save0_OPTIONS, "dtype": "bfloat16"} if failure == "dtype" else _lora_save0_OPTIONS
    if failure == "sharded":
        parameters[0][1].ds_id = 0
    if failure == "missing":
        parameters = parameters[:1]
    if failure == "alias":
        parameters.append((parameters[0][0], parameters[0][1]))
    if failure == "tied":
        parameters.append((parameters[0][0].replace(".q_proj.", ".k_proj."), parameters[0][1]))
    proxy = SimpleNamespace(named_parameters=lambda **kwargs: parameters)
    with pytest.raises(ValueError):
        lora_initialization.initialize_lora_factors(proxy, options)
    for (_, parameter), before in zip(parameters, original):
        torch_assert_equal(parameter, before)


def test_lora_save0_real_peft_save_pack_proves_complete_exact_initialization(tmp_path):
    root, response, metadata, _ = _lora_save0_saved(tmp_path)
    proof = lora_save0.verify_saved_initialization(root, response, metadata)
    assert proof["identity_sha256"] == metadata["initialization"]["identity_sha256"]
    assert len(proof["verified_tensors"]) == len(metadata["initialization"]["tensors"])
    assert proof["proof_scope"] == "exported-adapter-pack-not-all-rank-equality"
    payload = _lora_save0_payload()
    bound = lora_save0.bind_initialization(payload, metadata)
    assert "deterministic_init" not in payload["training_config"]["peft_config"]
    assert bound["training_config"]["peft_config"]["deterministic_init"] == _lora_save0_OPTIONS


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


@pytest.mark.parametrize("failure", ["width", "ep", "sp", "zero3", "model", "topology"])
def test_lora_save0_binding_rejects_unreviewed_target_scope(tmp_path, failure):
    _, _, metadata, _ = _lora_save0_saved(tmp_path)
    payload = _lora_save0_payload()
    training = payload["training_config"]
    if failure == "width":
        training["n_gpus"] = 8
    elif failure in {"ep", "sp"}:
        training[f"{failure}_size"] = 8
    elif failure == "zero3":
        training["ds_config"]["zero_optimization"]["stage"] = 3
    elif failure == "model":
        metadata["model_name"] = "another-model"
    elif failure == "topology":
        training["peft_config"]["r"] = 99
    with pytest.raises(ValueError):
        lora_save0.bind_initialization(payload, metadata)


@pytest.mark.parametrize("failure", [None, "save", "identity"])
def test_lora_save0_client_is_not_exposed_before_proof_and_owned_shutdown_always_occurs(
    tmp_path, monkeypatch, failure
):
    root, response, metadata, _ = _lora_save0_saved(tmp_path)
    events = []
    job = SimpleNamespace(client=object())

    @contextlib.contextmanager
    def running(session, payload):
        events.append("initialized")
        try:
            yield job
        finally:
            events.append("closed")

    def save(client, kind):
        assert client is job.client and kind == "weights-only"
        events.append("save0")
        if failure == "save":
            raise RuntimeError("actual save failure")
        if failure == "identity":
            return (root, {**response, "version": 1})
        return (root, response)

    monkeypatch.setattr(dss_driver, "running_job", running)
    monkeypatch.setattr(hosted, "_save_local_checkpoint", save)
    monkeypatch.setattr(hosted, "require_local_cortex_gateway", lambda backend: None)
    proof_path = tmp_path / "proof.json"
    if failure is not None:
        error = RuntimeError if failure == "save" else ValueError
        with pytest.raises(error):
            with lora_save0.verified_running_job(
                SimpleNamespace(backend=object()), _lora_save0_payload(), metadata, proof_path=proof_path
            ):
                pytest.fail("client exposed after failed proof")
        assert events == ["initialized", "save0", "closed"] and (not proof_path.exists())
    else:
        with lora_save0.verified_running_job(
            SimpleNamespace(backend=object()), _lora_save0_payload(), metadata, proof_path=proof_path
        ) as (exposed, proof):
            assert events == ["initialized", "save0"]
            assert exposed is job and proof_path.is_file()
            events.append("first-forward")
        assert events == ["initialized", "save0", "first-forward", "closed"]


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
    result = load_qwen4_reference(cls, "fixture", dtype=torch.bfloat16, attn_implementation=attention, device="cpu")
    assert result is model
    assert calls == [
        ("fixture", {"dtype": torch.bfloat16, "attn_implementation": expected, "output_loading_info": True})
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
    load_qwen4_reference(cls, "fixture", dtype=torch.bfloat16, attn_implementation="qsa_flex", device="cuda")
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


@pytest.fixture
def _runner_lora_save0_save0_runner(_prepared_reference_runner_prepared_runner, tmp_path, monkeypatch):
    control = _prepared_reference_runner_prepared_runner
    lora_cfg = load_config(_synthetic_config_spec(tmp_path / "lora-inputs", lora=True)[0])
    control.cfg.__dict__.update(lora_cfg.__dict__)
    control.spec.config_id = control.cfg.config_id
    control.spec.model.source_checkpoint = control.cfg.model_name
    control.spec.model.num_hidden_layers = 4
    control.backend = CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev", host=None, pat=None)
    control.payloads = []
    control.proofs = []
    control.failure = None
    control.adapter_path = tmp_path / "adapter"
    control.metadata = {
        "model_name": control.cfg.model_name,
        "peft_config": normalize_lora_peft_config(control.cfg.training["peft_config"]),
    }

    def set_manifest(pairs):
        rank = control.metadata["peft_config"]["r"]
        specs = []
        for index in range(pairs):
            prefix = f"base_model.model.model.layers.{index}.self_attn.q_proj"
            specs.extend(
                [
                    lora_init.LoraTensorSpec(f"{prefix}.lora_A.weight", (rank, 3 + index), "float32"),
                    lora_init.LoraTensorSpec(f"{prefix}.lora_B.weight", (5 + index, rank), "float32"),
                ]
            )
        control.metadata["initialization"] = lora_init.build_manifest(specs, runner.SEED)

    set_manifest(1)
    control.set_manifest = set_manifest

    def materialize(cfg, model, workdir):
        assert cfg.config_id == control.cfg.config_id
        assert model == control.spec.model.cache_path
        assert workdir == tmp_path / "run" / "reference-adapter-init"
        return control.adapter_path

    def identity(path):
        assert path == control.adapter_path
        return copy.deepcopy(control.metadata)

    def fixture(backend, spec):
        assert backend is control.backend and spec is control.spec
        return {"model_name": control.spec.model.cache_path}

    @contextmanager
    def gateway(workdir, gpus, *, backend):
        assert workdir == tmp_path / "run"
        assert gpus == 4 and backend is control.backend
        control.events.append(("gateway-enter",))
        try:
            yield SimpleNamespace(backend=backend)
        finally:
            control.events.append(("gateway-exit",))

    @contextmanager
    def running_job(session, payload):
        assert session.backend is control.backend
        assert payload["model_name"] == control.spec.model.cache_path
        assert payload["training_config"]["peft_config"]["deterministic_init"] == {
            "recipe_version": control.metadata["initialization"]["recipe_version"],
            "global_seed": runner.SEED,
            "dtype": "float32",
        }
        control.payloads.append(copy.deepcopy(payload))
        control.events.append(("initialized",))
        try:
            yield SimpleNamespace(client=object())
        finally:
            control.events.append(("closed",))

    root = tmp_path / "saved"
    response = {"checkpoint_path": str(root), "checkpoint_tag": "global_step0", "version": 0, "stage_path": None}

    def save(client, kind):
        assert client is not None and kind == "weights-only"
        control.events.append(("save0",))
        if control.failure == "save":
            raise RuntimeError("public SAVE0 failed")
        return (root, response)

    def verify(saved_root, saved_response, metadata):
        assert saved_root == root and saved_response == response
        assert metadata == control.metadata
        control.events.append(("proof",))
        if control.failure == "proof":
            raise ValueError("saved tensor identity differs")
        proof = {
            "identity_sha256": metadata["initialization"]["identity_sha256"],
            "verified_tensors": [
                {key: entry[key] for key in ("name", "dtype", "sha256")}
                for entry in metadata["initialization"]["tensors"]
            ],
            **response,
            "proof_scope": "exported-adapter-pack-not-all-rank-equality",
        }
        control.proofs.append(copy.deepcopy(proof))
        return proof

    def target(session, job, body, *, learning_rate, **kwargs):
        assert session.backend is control.backend and job.client is not None
        rows = body["batch"]["input_ids"].shape[0]
        arm = next((arm for arm in control.arms if arm.global_batch_size == rows))
        proof_path = tmp_path / "run" / "lora-save0" / f"{arm.name}.json"
        assert json.loads(proof_path.read_text()) == control.proofs[-1]
        assert control.events[-1] == ("proof",)
        if kwargs:
            assert learning_rate == runner.OPTIMIZER_LEARNING_RATE
            assert kwargs["local_checkpoint"] is True
            assert kwargs["expected_dp_world_size"] == 4
            assert kwargs["output_dir"] == tmp_path / "run" / "optimizer" / "dss" / arm.name
        else:
            assert learning_rate == 0.0
        control.events.append(("target", arm.name))
        return SimpleNamespace(
            optimizer_state_manifest=str(control.target_manifest),
            avg_loss=2.0,
            grad_norms={"module.layer.weight": 1.0},
            model_calls=None,
        )

    def execute(*, selected=_prepared_reference_runner_STEP_CHECKS, selected_arms=None, **overrides):
        options = {
            "vocab_size": 128,
            "select": selected,
            "workdir": tmp_path / "run",
            "reference_records_dir": tmp_path / "records",
            "client_backend": control.backend,
            "local_cortex_fixture": True,
            "local_gateway_lora_save0": True,
        }
        options.update(overrides)
        return runner.execute(
            control.cfg, control.spec, control.arms if selected_arms is None else selected_arms, **options
        )

    monkeypatch.setattr(deterministic_lora, "materialize_adapter", materialize)
    monkeypatch.setattr(deterministic_lora, "adapter_identity", identity)
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", fixture)
    monkeypatch.setattr(runner, "gateway", gateway)
    monkeypatch.setattr(dss_driver, "running_job", running_job)
    monkeypatch.setattr(hosted, "_save_local_checkpoint", save)
    monkeypatch.setattr(lora_save0, "verify_saved_initialization", verify)
    monkeypatch.setattr(runner, "running_job", lambda *a, **k: pytest.fail("must enter verified SAVE0 wrapper"))
    monkeypatch.setattr(runner, "fwd_bwd_step", target)
    monkeypatch.setattr(cortex_optimizer_capture, "capture_optimizer_step", target)
    control.execute = execute
    return control


@pytest.mark.parametrize("order", ["batched", "arm"])
@pytest.mark.parametrize(
    "selected",
    [
        _prepared_reference_runner_STEP_CHECKS[:1],
        _prepared_reference_runner_STEP_CHECKS[1:],
        _prepared_reference_runner_STEP_CHECKS,
    ],
)
@pytest.mark.parametrize("pairs", [1, 2])
def test_runner_lora_save0_cached_records_then_save0_proof_before_each_measurement(
    _runner_lora_save0_save0_runner, order, selected, pairs, tmp_path
):
    control = _runner_lora_save0_save0_runner
    control.set_manifest(pairs)
    before = copy.deepcopy((control.cfg.sub_job, control.metadata, control.record))
    results = control.execute(order=order, selected=selected)
    assert [(result.test_id, result.arm, result.outcome, result.gpus) for result in results] == [
        (check, arm.name, Outcome.PASS, 4) for arm in control.arms for check in selected
    ]
    records = [("record", check, arm.name) for arm in control.arms for check in selected]
    assert control.events[: len(records)] == records
    assert [event for event in control.events if event[0] not in {"record", "gateway-enter", "gateway-exit"}] == [
        event
        for arm in control.arms
        for event in [("initialized",), ("save0",), ("proof",), ("target", arm.name), ("closed",)]
    ]
    assert (
        control.events.count(("gateway-enter",))
        == control.events.count(("gateway-exit",))
        == (1 if order == "batched" else 2)
    )
    assert (control.cfg.sub_job, control.metadata, control.record) == before
    assert not control.published
    assert len(control.payloads) == len(control.arms)
    for _, _, recipe in control.recipes:
        assert recipe["initialization"] == control.metadata
        assert "deterministic_init" not in recipe["training"]["peft_config"]
    for arm in control.arms:
        proof = json.loads((tmp_path / "run" / "lora-save0" / f"{arm.name}.json").read_text())
        assert proof["identity_sha256"] == control.metadata["initialization"]["identity_sha256"]
        assert len(proof["verified_tensors"]) == len(control.metadata["initialization"]["tensors"]) == 2 * pairs
    for result in results:
        if result.test_id == "single-step-optimizer":
            assert result.metrics["optimizer_values_compared"] == 3
            assert result.metrics["stated_criterion_abs"] == 0.001


def test_runner_lora_save0_default_lora_cortex_rejection_is_unchanged(_runner_lora_save0_save0_runner):
    control = _runner_lora_save0_save0_runner
    results = control.execute(local_gateway_lora_save0=False)
    assert len(results) == 2
    assert all((result.outcome is Outcome.FAIL for result in results))
    assert all(("product-initialized LoRA adapter" in result.reason for result in results))
    assert not control.events and (not control.payloads) and (not control.published)


def test_runner_lora_save0_opt_in_does_not_bypass_fixture_capability_rejection(
    _runner_lora_save0_save0_runner, monkeypatch
):
    control = _runner_lora_save0_save0_runner
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", lambda *a: None)
    results = control.execute()
    assert len(results) == 2
    assert all((result.outcome is Outcome.FAIL and "published" in result.reason for result in results))
    assert not control.events and (not control.payloads) and (not control.published)


@pytest.mark.parametrize("order", ["batched", "arm"])
def test_runner_lora_save0_missing_cached_record_stops_before_allocation_or_save0(
    _runner_lora_save0_save0_runner, order
):
    control = _runner_lora_save0_save0_runner
    control.missing.add(("single-step-optimizer", "gas4"))
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="single-step-optimizer/gas4"):
        control.execute(order=order)
    assert control.events == [
        ("record", check, arm.name) for arm in control.arms for check in _prepared_reference_runner_STEP_CHECKS
    ]
    assert not control.payloads and (not control.proofs) and (not control.published)


@pytest.mark.parametrize("failure", ["save", "proof"])
def test_runner_lora_save0_save0_failure_closes_each_job_and_never_measures_or_passes(
    _runner_lora_save0_save0_runner, failure, tmp_path
):
    control = _runner_lora_save0_save0_runner
    control.failure = failure
    results = control.execute()
    assert len(results) == 4
    assert all((result.outcome is Outcome.FAIL and "execution error:" in result.summary for result in results))
    assert control.events.count(("initialized",)) == control.events.count(("closed",)) == 2
    assert not any((event[0] == "target" for event in control.events))
    assert not control.proofs and (not control.published)
    assert not (tmp_path / "run" / "lora-save0").exists()


@pytest.mark.parametrize(
    "failure",
    [
        "width",
        "config",
        "spec",
        "model",
        "source",
        "layers",
        "sp",
        "ep",
        "rl",
        "peft",
        "onprem",
        "remote",
        "auth",
        "prepare",
        "records",
        "empty",
        "unknown-check",
        "publication",
        "fixture",
    ],
)
def test_runner_lora_save0_opt_in_refuses_outside_reviewed_scope_before_any_work(
    _runner_lora_save0_save0_runner, failure
):
    control = _runner_lora_save0_save0_runner
    overrides = {}
    if failure == "width":
        control.cfg.native = control.cfg.native.model_copy(update={"training_gpus": 8})
        control.cfg.training["n_gpus"] = 8
    elif failure == "config":
        control.cfg.config_id = "another-config"
    elif failure == "spec":
        control.spec.config_id = "another-config"
    elif failure == "model":
        control.cfg.sub_job["model_name"] = "another-model"
    elif failure == "source":
        control.spec.model.source_checkpoint = "another-model"
    elif failure == "layers":
        control.spec.model.num_hidden_layers = 32
    elif failure in {"sp", "ep"}:
        control.cfg.training[f"{failure}_size"] = 2
    elif failure == "rl":
        control.cfg.native = control.cfg.native.model_copy(update={"sampling_gpus": 4})
    elif failure == "peft":
        control.cfg.training.pop("peft_config")
    elif failure == "onprem":
        overrides["client_backend"] = OnPremConfig()
    elif failure in {"remote", "auth"}:
        overrides["client_backend"] = CortexConfig(
            base_url="https://remote.invalid" if failure == "remote" else "http://127.0.0.1:8003",
            database="dev",
            schema="dev",
            host=None,
            pat="not-a-real-token" if failure == "auth" else None,
        )
    elif failure == "prepare":
        overrides["prepare_references"] = True
    elif failure == "records":
        overrides["reference_records_dir"] = None
    elif failure in {"empty", "unknown-check"}:
        overrides["selected"] = {"empty": [], "unknown-check": ["unknown-check"]}[failure]
    elif failure == "publication":
        overrides["cortex_fixture"] = Path("unread-publication.json")
    elif failure == "fixture":
        overrides["local_cortex_fixture"] = False
    with pytest.raises(ValueError):
        control.execute(**overrides)
    assert not control.events and (not control.payloads) and (not control.published)


def test_runner_lora_save0_default_selection_runs_both_checks(_runner_lora_save0_save0_runner):
    control = _runner_lora_save0_save0_runner
    results = control.execute(selected=None)
    assert [(result.test_id, result.arm, result.outcome) for result in results] == [
        (check, arm.name, Outcome.PASS) for arm in control.arms for check in _prepared_reference_runner_STEP_CHECKS
    ]


@pytest.mark.parametrize("failure", ["missing-adapter", "topology", "zero3", "sp-alias", "ep-alias"])
def test_runner_lora_save0_metadata_or_binding_failure_stops_before_records_and_gateway(
    _runner_lora_save0_save0_runner, monkeypatch, failure
):
    control = _runner_lora_save0_save0_runner
    if failure == "missing-adapter":
        monkeypatch.setattr(deterministic_lora, "materialize_adapter", lambda *a: None)
    elif failure == "topology":
        control.metadata["peft_config"]["r"] += 1
    elif failure == "zero3":
        control.cfg.training["ds_config"]["zero_optimization"]["stage"] = 3
    else:
        key = "sequence_parallel_size" if failure == "sp-alias" else "expert_parallel_size"
        control.cfg.training[key] = 2
    with pytest.raises(ValueError):
        control.execute()
    assert not control.events and (not control.payloads) and (not control.published)


def test_runner_lora_save0_default_onprem_binding_still_fails_closed(_runner_lora_save0_save0_runner):
    control = _runner_lora_save0_save0_runner
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="no supported public"):
        control.execute(client_backend=OnPremConfig(), local_cortex_fixture=False, local_gateway_lora_save0=False)
    assert not control.events and (not control.payloads) and (not control.published)
