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
"""New CPU controls for opt-in AP initialization and failclosed public SAVE0."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from arctic_platform.client import CortexConfig
from arctic_platform.common.peft import normalize_lora_initialization
from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness import lora_init
from arctic_platform.correctness.harness import lora_save0
from arctic_platform.model import lora_initialization
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.model.patches.peft import apply_peft_patch
from arctic_platform.model.weight_export import save_peft_adapters
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

OPTIONS = {
    "recipe_version": "arctic-correctness-lora-init-v1",
    "global_seed": 42,
    "dtype": "float32",
}
TOPOLOGY = normalize_lora_peft_config({"peft_type": "Lora", "r": 2, "target_modules": ["q_proj"]})


def _tiny_model():
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


def _metadata(model):
    specs = [
        lora_init.LoraTensorSpec(lora_init.canonical_name(name), tuple(parameter.shape), "float32")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "model_name": "Qwen/Qwen3-8B",
        "initialization": lora_init.build_manifest(specs, 42),
        "peft_config": deepcopy(TOPOLOGY),
    }


def _saved(tmp_path):
    model = apply_peft(_tiny_model(), {**TOPOLOGY, "deterministic_init": OPTIONS})
    root = tmp_path / "checkpoint"
    destination = root / "global_step0"
    save_peft_adapters(model, str(destination))
    metadata = _metadata(model)
    response = {"checkpoint_path": str(root), "checkpoint_tag": "global_step0", "version": 0, "stage_path": None}
    return root, response, metadata, model


def _payload():
    return {
        "job_type": "training",
        "model_name": "/data-fast/base-models/synthetic/Qwen3-8B-4L",
        "seed": 42,
        "training_config": {
            "n_gpus": 4,
            "sp_size": 1,
            "ep_size": 1,
            "ds_config": {"zero_optimization": {"stage": 2}},
            "peft_config": deepcopy(TOPOLOGY),
        },
    }


def test_absent_option_preserves_normalized_topology_and_default_peft_behavior():
    assert "deterministic_init" not in normalize_lora_peft_config(TOPOLOGY)
    set_seed(17)
    baseline = apply_peft(_tiny_model(), TOPOLOGY)
    set_seed(17)
    repeated = apply_peft(_tiny_model(), TOPOLOGY)
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
        {**OPTIONS, "unexpected": True},
        {**OPTIONS, "recipe_version": "unknown"},
        {**OPTIONS, "global_seed": True},
        {**OPTIONS, "global_seed": -1},
        {**OPTIONS, "global_seed": 2**63},
        {**OPTIONS, "dtype": []},
        {**OPTIONS, "dtype": "float64"},
    ],
)
def test_opt_in_schema_rejects_unsupported_values(options):
    with pytest.raises(ValueError):
        normalize_lora_initialization(options)


def test_normalizer_preserves_validated_option_without_mutating_input():
    original = {**TOPOLOGY, "deterministic_init": deepcopy(OPTIONS)}
    expected = deepcopy(original)
    normalized = normalize_lora_peft_config(original)
    assert original == expected == normalized
    assert normalized["deterministic_init"] is not original["deterministic_init"]
    with pytest.raises(ValueError, match="dense"):
        normalize_lora_peft_config({**original, "target_parameters": ["w1"]})


def test_real_peft_constructor_strips_option_and_matches_independent_reference_before_optimizer():
    model = apply_peft(_tiny_model(), {**TOPOLOGY, "deterministic_init": OPTIONS})
    assert "deterministic_init" not in model.peft_config["default"].to_dict()
    metadata = _metadata(model)
    tensors = {name: value for name, value in model.named_parameters() if value.requires_grad}
    lora_init.verify_tensors(tensors, metadata["initialization"])
    optimizer = torch.optim.AdamW(tensors.values(), lr=0.01)
    assert not optimizer.state
    lora_init.verify_tensors(tensors, metadata["initialization"])


def test_actual_public_sdk_main_validation_and_loader_forward_option_to_ap():
    from dss.ray_dss import models as main_models

    assert (
        hashlib.sha256(Path(main_models.__file__).read_bytes()).hexdigest()
        == "f408556316da701d58e7abf51090c7d8708577c7f58e491b2b54ddb30908718e"
    )
    loader_path = Path(main_models.__file__).parent / "jobs" / "gpu" / "model_loader.py"
    assert (
        hashlib.sha256(loader_path.read_bytes()).hexdigest()
        == "fcfc39f21601db6d025c8cd00e3c36d1e2ba4f3d161ee5b4d294da5984b93094"
    )
    # Exercise the real pure loader without importing the unrelated Ray actor package.
    loader_spec = importlib.util.spec_from_file_location("_correctness_main_model_loader", loader_path)
    assert loader_spec is not None and loader_spec.loader is not None
    main_loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(main_loader)
    model = _tiny_model()
    expected = _metadata(apply_peft(_tiny_model(), {**TOPOLOGY, "deterministic_init": OPTIONS}))
    payload = lora_save0.bind_initialization(_payload(), expected)
    session = dss_driver.GatewaySession(
        workdir=Path("/data-fast/tmp/not-used-sdk-control"),
        slots=4,
        backend=CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev"),
    )
    config = dss_driver._training_client_config(session, payload)
    sub_jobs = config.to_cortex()
    assert len(sub_jobs) == 1 and sub_jobs[0]["job_type"] == "training"
    public_config = sub_jobs[0]["training_config"]
    assert public_config["peft_config"]["deterministic_init"] == OPTIONS
    validated = main_models.TrainingConfig.model_validate(public_config)
    assert validated.peft_config["deterministic_init"] == OPTIONS
    spec = main_loader.dense_model_spec(
        model_path=payload["model_name"],
        torch_dtype=torch.float32,
        attn_impl="eager",
        model_provider="huggingface",
        sp_size=1,
        training_config=validated.model_dump(exclude_none=True),
    )
    assert spec.patches.peft["deterministic_init"] == OPTIONS
    actual = apply_peft_patch(model, LoaderContext(spec=spec))
    lora_init.verify_tensors(
        [(name, parameter) for name, parameter in actual.named_parameters() if parameter.requires_grad],
        expected["initialization"],
    )


def test_initializer_preserves_parameter_objects_and_rng_and_ignores_traversal_order():
    model = apply_peft(_tiny_model(), TOPOLOGY)
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = torch.random.get_rng_state().clone()
    lora_initialization.initialize_lora_factors(model, OPTIONS)
    torch_assert_equal(state, torch.random.get_rng_state())
    expected = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    proxy = SimpleNamespace(named_parameters=lambda **kwargs: reversed(list(parameters.items())))
    lora_initialization.initialize_lora_factors(proxy, OPTIONS)
    for name, parameter in model.named_parameters():
        if name in parameters:
            assert parameter is parameters[name]
            torch_assert_equal(parameter, expected[name])
    tree = ast.parse(Path(lora_initialization.__file__).read_text())
    assert not any(
        isinstance(node, ast.ImportFrom) and "correctness" in (node.module or "") for node in ast.walk(tree)
    )


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_private_initializer_matches_exact_reference_generation_dtype_and_cast(dtype):
    model = apply_peft(_tiny_model(), TOPOLOGY)
    model.to(dtype=getattr(torch, dtype))
    lora_initialization.initialize_lora_factors(model, {**OPTIONS, "dtype": dtype})
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
def test_patcher_rejects_unreviewed_parallelism_before_wrapping(ep, sp):
    ctx = SimpleNamespace(
        spec=SimpleNamespace(
            patches=SimpleNamespace(peft={**TOPOLOGY, "deterministic_init": OPTIONS}),
            parallelism=SimpleNamespace(expert_parallel=ep, sequence_parallel=sp),
        )
    )
    with pytest.raises(ValueError, match="EP1/SP1"):
        apply_peft_patch(_tiny_model(), ctx)


@pytest.mark.parametrize("failure", ["dtype", "sharded", "missing", "alias"])
def test_initializer_rejects_unsupported_storage_before_any_parameter_is_changed(failure):
    model = apply_peft(_tiny_model(), TOPOLOGY)
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    original = [parameter.detach().clone() for _, parameter in parameters]
    options = {**OPTIONS, "dtype": "bfloat16"} if failure == "dtype" else OPTIONS
    if failure == "sharded":
        parameters[0][1].ds_id = 0
    if failure == "missing":
        parameters = parameters[:1]
    if failure == "alias":
        parameters.append((parameters[0][0], parameters[0][1]))
    proxy = SimpleNamespace(named_parameters=lambda **kwargs: parameters)
    with pytest.raises(ValueError):
        lora_initialization.initialize_lora_factors(proxy, options)
    for (_, parameter), before in zip(parameters, original):
        torch_assert_equal(parameter, before)


def test_real_peft_save_pack_proves_complete_exact_initialization(tmp_path):
    root, response, metadata, _ = _saved(tmp_path)
    proof = lora_save0.verify_saved_initialization(root, response, metadata)
    assert proof["identity_sha256"] == metadata["initialization"]["identity_sha256"]
    assert len(proof["verified_tensors"]) == len(metadata["initialization"]["tensors"])
    assert proof["proof_scope"] == "exported-adapter-pack-not-all-rank-equality"
    payload = _payload()
    bound = lora_save0.bind_initialization(payload, metadata)
    assert "deterministic_init" not in payload["training_config"]["peft_config"]
    assert bound["training_config"]["peft_config"]["deterministic_init"] == OPTIONS


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
def test_save0_rejects_wrong_step_path_or_stage(tmp_path, field, value):
    root, response, metadata, _ = _saved(tmp_path)
    response[field] = value
    with pytest.raises(ValueError, match="step/version zero"):
        lora_save0.verify_saved_initialization(root, response, metadata)


@pytest.mark.parametrize("failure", ["missing", "extra", "bytes", "dtype", "shape", "symlink", "config"])
def test_save0_rejects_changed_coverage_identity_and_files(tmp_path, failure):
    root, response, metadata, _ = _saved(tmp_path)
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
def test_binding_rejects_unreviewed_target_scope(tmp_path, failure):
    _, _, metadata, _ = _saved(tmp_path)
    payload = _payload()
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
def test_client_is_not_exposed_before_proof_and_owned_shutdown_always_occurs(tmp_path, monkeypatch, failure):
    root, response, metadata, _ = _saved(tmp_path)
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
            return root, {**response, "version": 1}
        return root, response

    monkeypatch.setattr(dss_driver, "running_job", running)
    monkeypatch.setattr(hosted, "_save_local_checkpoint", save)
    monkeypatch.setattr(hosted, "require_local_cortex_gateway", lambda backend: None)
    proof_path = tmp_path / "proof.json"
    if failure is not None:
        error = RuntimeError if failure == "save" else ValueError
        with pytest.raises(error):
            with lora_save0.verified_running_job(
                SimpleNamespace(backend=object()), _payload(), metadata, proof_path=proof_path
            ):
                pytest.fail("client exposed after failed proof")
        assert events == ["initialized", "save0", "closed"] and not proof_path.exists()
    else:
        with lora_save0.verified_running_job(
            SimpleNamespace(backend=object()), _payload(), metadata, proof_path=proof_path
        ) as (exposed, proof):
            assert events == ["initialized", "save0"]
            assert exposed is job and proof_path.is_file()
            events.append("first-forward")
        assert events == ["initialized", "save0", "first-forward", "closed"]
