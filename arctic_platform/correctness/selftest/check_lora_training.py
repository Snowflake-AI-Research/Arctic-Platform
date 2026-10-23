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

"""LoRA correctness controls: training, adapters, synchronization, and sampling."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import hashlib
import inspect
import json
import logging
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
import torch.distributed as dist
import transformers
from safetensors.torch import load_file
from safetensors.torch import save
from safetensors.torch import save_file
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.checks import e2e_sft_train_validate as check
from arctic_platform.correctness.checks import e2e_sft_train_validate as e2e
from arctic_platform.correctness.checks import inference_checkpoint_loss as inference
from arctic_platform.correctness.harness import lora as deterministic_lora
from arctic_platform.correctness.harness import lora as lora_init
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import sft
from arctic_platform.correctness.harness import sft as gsm8k
from arctic_platform.correctness.harness import shared_lora_adapter as shared
from arctic_platform.correctness.harness import storage
from arctic_platform.correctness.harness import storage as workdir
from arctic_platform.correctness.harness.batches import IGNORE_INDEX
from arctic_platform.correctness.harness.batches import load
from arctic_platform.correctness.harness.dss_driver import pack
from arctic_platform.correctness.harness.dss_driver import pack_logit_aligned
from arctic_platform.correctness.harness.lora import canonical_name
from arctic_platform.correctness.harness.registry import TestOutcome
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.reference import process_determinism
from arctic_platform.correctness.reference.adamw_trajectory import AdamWTrajectory
from arctic_platform.correctness.reference.hf_single_gpu import build_engine
from arctic_platform.correctness.reference.optimizer_step import run_adamw_step
from arctic_platform.correctness.reference.qwen4_reference import load_qwen4_reference
from arctic_platform.debug import lora_adapter_storage_evidence as evidence
from arctic_platform.debug import lora_adapter_tensor_binding as _lora_adapter_tensor_binding_evidence
from arctic_platform.debug import lora_receiver_bindings as bindings
from arctic_platform.debug import lora_receiver_preparation as helper
from arctic_platform.debug import lora_replica_pool_routing as routing
from arctic_platform.debug import lora_request_staging as _lora_request_staging_helper
from arctic_platform.debug import lora_sampler_evidence as bridge
from arctic_platform.debug import lora_sampler_evidence as receiver
from arctic_platform.debug import lora_sampler_worker as _lora_sampler_worker_bridge
from arctic_platform.debug import lora_selected_sampler as selected
from arctic_platform.debug import lora_sync_hooks as hooks
from arctic_platform.debug import lora_sync_hooks as legacy
from arctic_platform.debug import lora_training_protocol as protocol
from arctic_platform.debug import lora_weight_sync_evidence as _lora_weight_sync_evidence_evidence
from arctic_platform.debug import rl_sampler_evidence
from arctic_platform.debug.ep_weight_sync_evidence import global_inventory
from arctic_platform.debug.lora_adapter_storage_evidence import ObservedLoRAOperation
from arctic_platform.debug.rl_sampler_evidence import SamplerEvidenceWorker
from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.model import weight_export as native
from arctic_platform.model.patches.peft import is_peft_lora_param

from . import check_reference_records as _reference_selftests

_prepared_reference_runner_prepared_runner = _reference_selftests._prepared_reference_runner_prepared_runner


# E2e reference records.


class _e2e_reference_records_Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    padding_side = "right"
    offset = 0

    def __len__(self):
        return 1000

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is True
        return {"input_ids": [1, *[ord(value) + 10 + self.offset for value in text]]}


def _e2e_reference_records_forbidden(*args, **kwargs):
    raise AssertionError("HF, product initialization, or an extra allocation was reached")


@pytest.fixture
def _e2e_reference_records_ctx(tmp_path, monkeypatch):
    model = tmp_path / "full-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"vocab_size": 1000, "layer_types": ["full_attention"]}))
    (model / "model.safetensors").write_bytes(b"mock full-model content")
    (model / "tokenizer.json").write_text('{"mock":true}')
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.parquet").write_bytes(b"ordered train source")
    (dataset / "test.parquet").write_bytes(b"ordered held-out source")
    monkeypatch.setenv(check.GSM8K_DATA_ROOT_ENV, str(dataset))
    tokenizer = _e2e_reference_records_Tokenizer()
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: tokenizer)

    def rows(path, *, start=0, count=None):
        values = [(f"train-{index}", str(index)) for index in range(200)]
        if Path(path).name == "test.parquet":
            values = [(f"held-{index}", str(index)) for index in range(8)]
        return values[start:] if count is None else values[start : start + count]

    monkeypatch.setattr(check.gsm8k, "read_rows", rows)
    training = {"train_batch_size": 2, "optimizer": {"name": "AdamW", "lr": 0.0001}}
    cfg = SimpleNamespace(
        training=training,
        effective_training=training,
        max_tokens_per_mb=512,
        n_gpus=4,
        dp_size=4,
        gpu_type="h200",
        sub_job={"dtype": "bfloat16"},
        optimizer_dtype="float32",
        fused_cross_entropy=False,
        lm_head_token_chunk_size=None,
    )
    context = SimpleNamespace(
        cfg=cfg,
        config_id="full-model-h200-e2e",
        spec=SimpleNamespace(
            model=SimpleNamespace(cache_path=str(model)),
            config_checksum="frozen-config",
            config_path=tmp_path / "config",
        ),
        workdir=tmp_path / "work",
        attn_implementation="sdpa",
        prepare_references=False,
        reference_records_dir=tmp_path / "records",
        reference_record_paths={},
        tokenizer=tokenizer,
        dataset=dataset,
    )
    return context


def _e2e_reference_records_measurements(inputs):
    return {
        "train_losses": [2.0] * 100,
        "train_gradient_norms": [1.0] * 100,
        "optimizer_steps": 100,
        "validation_cross_entropy": 1.5,
        "scored_answer_tokens": inputs.recipe["scored_answer_tokens"],
        "active_tokens": [batch["active_tokens"] for batch in inputs.recipe["train_batches"]],
        "peak_gib": 3.0,
        "trainable_parameters": 8,
    }


def _e2e_reference_records_publish(ctx, inputs=None):
    inputs = check.prepare_inputs(ctx) if inputs is None else inputs
    identity = reference_records.workload_identity(ctx, check.TEST_ID, inputs.recipe)
    reference = _e2e_reference_records_measurements(inputs)
    path = reference_records.publish_record(ctx, identity, reference)
    return (inputs, identity, reference, path)


def _e2e_reference_records_guard_launches(monkeypatch):
    monkeypatch.setattr(check, "run_reference_trajectory", _e2e_reference_records_forbidden)
    monkeypatch.setattr(runner, "materialize_dss_peft_adapter", _e2e_reference_records_forbidden)
    monkeypatch.setattr(check.subprocess, "Popen", _e2e_reference_records_forbidden)
    monkeypatch.setattr(check, "gateway", _e2e_reference_records_forbidden)
    monkeypatch.setattr(check, "running_job", _e2e_reference_records_forbidden)


def test_e2e_reference_records_prepared_regression_runs_one_target_and_zero_hf(
    _e2e_reference_records_ctx, monkeypatch
):
    inputs, _, reference, _ = _e2e_reference_records_publish(_e2e_reference_records_ctx)
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(reference_records, "publish_record", _e2e_reference_records_forbidden)
    calls = []

    def target(context, replay, directory, adapter):
        assert context is _e2e_reference_records_ctx
        assert adapter is None
        check.validate_replay(inputs)
        calls.append(replay.steps)
        return (
            [2.0] * 100,
            torch.full(
                (
                    len(inputs.validation_examples),
                    int(inputs.measured["validation_padded_width"]),
                ),
                -1.5,
            ),
        )

    monkeypatch.setattr(check, "run_dss_trajectory", target)
    result = check.run(_e2e_reference_records_ctx)[0]
    assert result.outcome is Outcome.PASS
    assert calls == [100]
    assert result.metrics["gated_quantities"] == 2
    assert result.metrics["train_steps_recorded"] == 100
    assert result.metrics["scored_answer_tokens"] == reference["scored_answer_tokens"]
    diagnostic = json.loads((inputs.workdir / "trajectories.json").read_text())
    assert diagnostic["reference_train_losses"] == reference["train_losses"]
    assert diagnostic["reference_train_gradient_norms"] == reference["train_gradient_norms"]
    assert not any(("logprobs" in key for key in diagnostic))
    assert len([key for key in result.metrics if key.startswith("train_loss_abs_diff_step_")]) == 100


def test_e2e_reference_records_missing_record_stops_before_target_and_prints_full_traceback(
    _e2e_reference_records_ctx, monkeypatch, caplog
):
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    with caplog.at_level(logging.ERROR):
        result = check.run(_e2e_reference_records_ctx)[0]
    assert result.outcome is Outcome.FAIL
    assert "prepared HF reference missing" in result.summary
    assert "--prepare-references" in result.summary
    assert "Traceback (most recent call last)" in caplog.text


def test_e2e_reference_records_corrupt_record_cannot_fall_back_to_hf(_e2e_reference_records_ctx, monkeypatch):
    _, _, _, path = _e2e_reference_records_publish(_e2e_reference_records_ctx)
    record = json.loads(path.read_text())
    record["measurements"]["train_losses"][0] = 123.0
    path.write_text(json.dumps(record))
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="corrupt or incompatible"):
        check._run(_e2e_reference_records_ctx)


def test_e2e_reference_records_shared_sft_source_is_bound_and_drift_requires_explicit_preparation(
    _e2e_reference_records_ctx, monkeypatch
):
    _, identity, _, _ = _e2e_reference_records_publish(_e2e_reference_records_ctx)
    source = reference_records.file_identity(Path(check.__file__).parents[1] / "harness" / "sft.py")
    assert identity["sft_source"] == source
    original = reference_records.file_identity

    def changed(path):
        value = original(path)
        return {**value, "sha256": "0" * 64} if Path(path).name == "sft.py" else value

    monkeypatch.setattr(reference_records, "file_identity", changed)
    _e2e_reference_records_guard_launches(monkeypatch)
    with pytest.raises(
        reference_records.ReferencePrerequisiteError,
        match="prepared HF reference missing",
    ):
        check._run(_e2e_reference_records_ctx)


def test_e2e_reference_records_old_unbound_e2e_record_is_not_silently_relabelled(
    _e2e_reference_records_ctx, monkeypatch
):
    inputs = check.prepare_inputs(_e2e_reference_records_ctx)
    identity = reference_records.workload_identity(_e2e_reference_records_ctx, check.TEST_ID, inputs.recipe)
    old = {key: value for key, value in identity.items() if key != "sft_source"}
    path = reference_records.publish_record(
        _e2e_reference_records_ctx, old, _e2e_reference_records_measurements(inputs)
    )
    before = path.read_bytes()
    _e2e_reference_records_guard_launches(monkeypatch)
    with pytest.raises(
        reference_records.ReferencePrerequisiteError,
        match="prepared HF reference missing",
    ):
        check._run(_e2e_reference_records_ctx)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("train_losses", [2.0] * 99),
        ("train_losses", [float("nan")] * 100),
        ("train_gradient_norms", [1.0] * 99),
        ("train_gradient_norms", [-1.0] * 100),
        ("train_gradient_norms", [True] * 100),
        ("optimizer_steps", 99),
        ("validation_cross_entropy", float("inf")),
        ("scored_answer_tokens", 999999),
        ("scored_answer_tokens", True),
        ("active_tokens", [0] * 100),
        ("validation_logprobs", [[-1.5]]),
    ],
)
def test_e2e_reference_records_incomplete_or_nonfinite_scalars_stop_before_target(
    _e2e_reference_records_ctx, monkeypatch, key, value
):
    inputs = check.prepare_inputs(_e2e_reference_records_ctx)
    reference = _e2e_reference_records_measurements(inputs)
    reference[key] = value
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(reference_records, "load_record", lambda context, identity: reference)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    with pytest.raises(reference_records.ReferencePrerequisiteError):
        check._run(_e2e_reference_records_ctx)


@pytest.mark.parametrize("change", ["dataset", "tokenizer", "model", "grouping", "budget", "gate"])
def test_e2e_reference_records_input_or_workload_drift_requires_a_new_record_before_target(
    _e2e_reference_records_ctx, monkeypatch, change
):
    _e2e_reference_records_publish(_e2e_reference_records_ctx)
    if change == "dataset":
        (_e2e_reference_records_ctx.dataset / "train.parquet").write_bytes(b"different ordered source")
    elif change == "tokenizer":
        _e2e_reference_records_ctx.tokenizer.offset = 1
    elif change == "model":
        (Path(_e2e_reference_records_ctx.spec.model.cache_path) / "model.safetensors").write_bytes(
            b"different full-model content"
        )
    elif change == "grouping":
        _e2e_reference_records_ctx.cfg.training["train_batch_size"] = 1
    elif change == "budget":
        _e2e_reference_records_ctx.cfg.max_tokens_per_mb = 256
    else:
        monkeypatch.setattr(check, "TOLERANCE", 0.04)
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    with pytest.raises(
        reference_records.ReferencePrerequisiteError,
        match="prepared HF reference missing",
    ):
        check._run(_e2e_reference_records_ctx)


def test_e2e_reference_records_relocated_source_paths_preserve_identity_and_sequential_selection(
    _e2e_reference_records_ctx, tmp_path, monkeypatch
):
    first = check.prepare_inputs(_e2e_reference_records_ctx)
    staged = tmp_path / "relocated-dataset"
    staged.mkdir()
    for name in ("train.parquet", "test.parquet"):
        (staged / name).write_bytes((_e2e_reference_records_ctx.dataset / name).read_bytes())
    monkeypatch.setenv(check.GSM8K_DATA_ROOT_ENV, str(staged))
    second = check.prepare_inputs(_e2e_reference_records_ctx)
    assert first.recipe == second.recipe
    assert first.replay.steps == 100
    assert first.recipe["dataset"]["train"] == {
        "start": 0,
        "count": 200,
        "rows_per_step": 2,
    }
    assert first.recipe["dataset"]["train_groups"] == [{"start": step * 2, "count": 2} for step in range(100)]
    assert str(tmp_path) not in json.dumps(first.recipe)
    assert first.recipe["preprocessing"]["labels"] == "answer-only-huggingface"
    assert first.recipe["dataset"]["validation"]["minimum_rows"] == 4


def test_e2e_reference_records_replay_tampering_after_record_load_stops_before_target(
    _e2e_reference_records_ctx, monkeypatch
):
    inputs, _, reference, _ = _e2e_reference_records_publish(_e2e_reference_records_ctx)
    monkeypatch.setattr(check, "prepare_inputs", lambda context: inputs)

    def load_record(context, identity):
        batch = check.load(inputs.replay.train_batch_paths[0])
        batch.labels[0, -1] += 1
        check.save(batch, inputs.replay.train_batch_paths[0])
        return reference

    monkeypatch.setattr(reference_records, "load_record", load_record)
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="replay differs"):
        check._run(_e2e_reference_records_ctx)


@pytest.mark.parametrize("lora", [False, True])
def test_e2e_reference_records_explicit_preparation_runs_hf_once_publishes_scalars_and_never_runs_target(
    _e2e_reference_records_ctx, monkeypatch, lora
):
    _e2e_reference_records_ctx.prepare_references = True
    adapter = None
    if lora:
        _e2e_reference_records_ctx.cfg.training["peft_config"] = {"r": 4}
        adapter = _e2e_reference_records_ctx.workdir / "adapter"
        monkeypatch.setattr(deterministic_lora, "materialize_adapter", lambda *args: adapter)
        monkeypatch.setattr(deterministic_lora, "identity_metadata", lambda path: {"mock": "exact-init"})
    monkeypatch.setattr(deterministic_lora, "bind_payload", _e2e_reference_records_forbidden)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    monkeypatch.setattr(runner, "materialize_dss_peft_adapter", _e2e_reference_records_forbidden)
    calls = []

    def reference(context, replay, directory, prepared_adapter):
        assert prepared_adapter == adapter
        inputs = check.prepare_inputs(context)
        values = _e2e_reference_records_measurements(inputs)
        calls.append(replay.steps)
        values.pop("validation_cross_entropy")
        values.pop("scored_answer_tokens")
        values["validation_logprobs"] = torch.full(
            (
                len(inputs.validation_examples),
                int(inputs.measured["validation_padded_width"]),
            ),
            -1.5,
        ).tolist()
        return values

    monkeypatch.setattr(check, "run_reference_trajectory", reference)
    assert check.run(_e2e_reference_records_ctx) == []
    assert calls == [100]
    paths = list(_e2e_reference_records_ctx.reference_records_dir.rglob("*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["measurements"]["validation_cross_entropy"] == 1.5
    assert len(record["measurements"]["train_losses"]) == 100
    assert len(record["measurements"]["train_gradient_norms"]) == 100
    assert "validation_logprobs" not in record["measurements"]
    assert list(_e2e_reference_records_ctx.reference_records_dir.rglob("*.pt")) == []
    assert list(_e2e_reference_records_ctx.reference_records_dir.rglob("*.safetensors")) == []
    monkeypatch.setattr(check, "run_reference_trajectory", _e2e_reference_records_forbidden)
    assert check.run(_e2e_reference_records_ctx) == []
    assert calls == [100]


def test_e2e_reference_records_preparation_exception_propagates_instead_of_becoming_empty_success(
    _e2e_reference_records_ctx, monkeypatch
):
    _e2e_reference_records_ctx.prepare_references = True

    def fail(context):
        raise RuntimeError("actual preparation traceback")

    monkeypatch.setattr(check, "prepare_reference_record", fail)
    with pytest.raises(RuntimeError, match="actual preparation traceback"):
        check.run(_e2e_reference_records_ctx)


def test_e2e_reference_records_preparation_requires_explicit_mode(_e2e_reference_records_ctx, monkeypatch):
    monkeypatch.setattr(check, "prepare_inputs", _e2e_reference_records_forbidden)
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="explicit"):
        check.prepare_reference_record(_e2e_reference_records_ctx)


def test_e2e_reference_records_corrupt_existing_record_is_not_reprepared(_e2e_reference_records_ctx, monkeypatch):
    _, _, _, path = _e2e_reference_records_publish(_e2e_reference_records_ctx)
    path.write_text("invalid JSON")
    _e2e_reference_records_ctx.prepare_references = True
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    with pytest.raises(
        reference_records.ReferencePrerequisiteError,
        match="prepared HF reference corrupt",
    ):
        check.run(_e2e_reference_records_ctx)


def test_e2e_reference_records_unsupported_lora_init_fails_before_hf_or_target(
    _e2e_reference_records_ctx, monkeypatch
):
    _e2e_reference_records_ctx.cfg.training["peft_config"] = {"r": 4}
    monkeypatch.setattr(
        deterministic_lora,
        "materialize_adapter",
        lambda *args: _e2e_reference_records_ctx.workdir / "adapter",
    )
    monkeypatch.setattr(deterministic_lora, "identity_metadata", _e2e_reference_records_forbidden)
    _e2e_reference_records_guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", _e2e_reference_records_forbidden)
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="public"):
        check._run(_e2e_reference_records_ctx)


def test_e2e_reference_records_staged_dataset_root_must_be_fast_storage(_e2e_reference_records_ctx, monkeypatch):
    monkeypatch.setenv(check.GSM8K_DATA_ROOT_ENV, "/code/staged-gsm8k")
    with pytest.raises(ValueError, match="/data-fast"):
        check.dataset_paths()


def test_e2e_reference_records_reference_options_are_shared_with_hf_command(
    _e2e_reference_records_ctx,
):
    options = check.reference_options(_e2e_reference_records_ctx, _e2e_reference_records_ctx.spec.model.cache_path)
    command = check.reference_command(_e2e_reference_records_ctx, None, Path("replay"), Path("output"), None)
    assert command[command.index("--ce-chunk") + 1] == str(options["ce_chunk"])
    assert command[command.index("--token-budget") + 1] == str(options["token_budget"])
    assert command[command.index("--learning-rate") + 1] == str(options["learning_rate"])
    assert json.loads(command[command.index("--optimizer-config") + 1]) == options["optimizer_config"]


# E2e sft train validate.


def _e2e_sft_train_validate_assert_nested_equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _e2e_sft_train_validate_assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _e2e_sft_train_validate_assert_nested_equal(left_item, right_item)
    else:
        assert left == right


_e2e_sft_train_validate_STEPS = 4

_e2e_sft_train_validate_ROWS_PER_STEP = 2

_e2e_sft_train_validate_ADAMW = {
    "name": "AdamW",
    "lr": 2e-05,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1e-08,
}

_e2e_sft_train_validate_OVER_GATE = check.TOLERANCE * 3


def _e2e_sft_train_validate_examples(count: int, *, first: int = 0):
    """Rows of unequal length, each with a distinct token stream, so no two steps share a batch."""
    return [
        gsm8k.Example(
            question=f"q{first + index}",
            prompt="Question: q\nAnswer:",
            completion=" a",
            input_ids=tuple(range(10 * (first + index) + 1, 10 * (first + index) + 4 + index % 3)),
            n_prompt=2,
        )
        for index in range(count)
    ]


def _e2e_sft_train_validate_a_replay(tmp_path: Path, steps: int = _e2e_sft_train_validate_STEPS) -> check.Replay:
    rows = _e2e_sft_train_validate_examples(steps * _e2e_sft_train_validate_ROWS_PER_STEP)
    return check.materialize_replay(
        check.replay_batches(rows, _e2e_sft_train_validate_ROWS_PER_STEP, steps, pad_id=0),
        gsm8k.build_batch("validation", _e2e_sft_train_validate_examples(3, first=100), pad_id=0),
        tmp_path / "replay",
    )


def test_e2e_sft_train_validate_mixer_packing_is_selected_from_checkpoint_layers(
    tmp_path,
) -> None:
    from arctic_platform.correctness.reference.model_features import uses_mixer_packing

    (tmp_path / "config.json").write_text(json.dumps({"layer_types": ["linear_attention", "full_attention"]}))
    assert uses_mixer_packing(tmp_path) is True
    (tmp_path / "config.json").write_text(json.dumps({"layer_types": ["full_attention"]}))
    assert uses_mixer_packing(tmp_path) is False


def test_e2e_sft_train_validate_mixer_packing_keeps_each_sequence_in_its_own_reference_call() -> None:
    from arctic_platform.correctness.reference.hf_single_gpu import reference_row_groups

    lengths = [4, 4]
    assert reference_row_groups(lengths, 4, 64, mixer_packing=True) == [[0], [1]]
    assert reference_row_groups(lengths, 4, 64, mixer_packing=False) == [[0, 1]]


def test_e2e_sft_train_validate_every_step_gets_its_own_batch_from_the_training_slice() -> None:
    rows = _e2e_sft_train_validate_examples(_e2e_sft_train_validate_STEPS * _e2e_sft_train_validate_ROWS_PER_STEP)
    batches = check.replay_batches(
        rows,
        _e2e_sft_train_validate_ROWS_PER_STEP,
        _e2e_sft_train_validate_STEPS,
        pad_id=0,
    )
    assert len(batches) == _e2e_sft_train_validate_STEPS
    assert all((batch.rows == _e2e_sft_train_validate_ROWS_PER_STEP for batch in batches))
    seen = [tuple(row) for batch in batches for row in batch.input_ids.tolist()]
    assert len(set(seen)) == len(rows)


def test_e2e_sft_train_validate_a_slice_that_does_not_fill_the_steps_is_refused() -> None:
    with pytest.raises(ValueError, match="need 8 examples"):
        check.replay_batches(
            _e2e_sft_train_validate_examples(7),
            _e2e_sft_train_validate_ROWS_PER_STEP,
            _e2e_sft_train_validate_STEPS,
            pad_id=0,
        )


def test_e2e_sft_train_validate_both_engines_are_handed_the_same_replay_files_in_the_same_order(
    tmp_path,
) -> None:
    """The one property that makes the comparison a comparison: one list of files, read by both sides."""
    replay = _e2e_sft_train_validate_a_replay(tmp_path)
    manifest = json.loads(check.write_replay_manifest(replay, tmp_path / "replay.json").read_text())
    training_bodies, _ = check.request_bodies(replay, model_provider="huggingface")
    assert manifest["train_batches"] == [str(path) for path in replay.train_batch_paths]
    assert manifest["validation_batch"] == str(replay.validation_batch_path)
    assert len(training_bodies) == replay.steps
    expected = [pack(load(path), model_provider="huggingface") for path in replay.train_batch_paths]
    _e2e_sft_train_validate_assert_nested_equal(training_bodies, expected)
    assert (
        len({body["batch"]["input_ids"].detach().cpu().numpy().tobytes() for body in training_bodies}) == replay.steps
    )


def test_e2e_sft_train_validate_the_held_out_request_carries_logit_aligned_labels(
    tmp_path,
) -> None:
    import torch

    "The forward-only route shifts nothing, so the same bytes ``/fwd-bwd`` takes would score one off."
    replay = _e2e_sft_train_validate_a_replay(tmp_path)
    validation_batch = load(replay.validation_batch_path)
    _, validation_body = check.request_bodies(replay, model_provider="huggingface")
    _e2e_sft_train_validate_assert_nested_equal(validation_body, pack_logit_aligned(validation_batch))
    assert not torch.equal(
        validation_body["batch"]["labels"],
        pack(validation_batch, model_provider="huggingface")["batch"]["labels"],
    )


def test_e2e_sft_train_validate_the_training_requests_carry_the_convention_their_dispatch_expects(
    tmp_path,
) -> None:
    """``/fwd-bwd`` shifts HuggingFace-convention labels itself; pre-shifting them would double it."""
    replay = _e2e_sft_train_validate_a_replay(tmp_path)
    packed = {"train": [], "validation": []}

    def record_train(batch, *, model_provider):
        packed["train"].append(batch.name)
        return f"train:{batch.name}".encode()

    def record_validation(batch):
        packed["validation"].append(batch.name)
        return f"validation:{batch.name}".encode()

    original = (check.pack, check.pack_logit_aligned)
    check.pack, check.pack_logit_aligned = (record_train, record_validation)
    try:
        training_bodies, validation_body = check.request_bodies(replay, model_provider="huggingface")
    finally:
        check.pack, check.pack_logit_aligned = original
    assert packed["train"] == [f"train-step-{step:03d}" for step in range(1, replay.steps + 1)]
    assert packed["validation"] == ["validation"]
    assert validation_body == b"validation:validation"
    assert len(training_bodies) == replay.steps


def _e2e_sft_train_validate_a_verdict(reference, target, reference_validation=1.5, target_validation=1.5):
    return check.verdict(
        SimpleNamespace(config_id="example"),
        reference,
        target,
        reference_validation,
        target_validation,
    )


def test_e2e_sft_train_validate_the_verdict_gates_on_the_final_training_loss_and_the_validation_loss() -> None:
    assert check.gated_training_steps(100) == (100,)
    result = _e2e_sft_train_validate_a_verdict([2.5] * 100, [2.5] * 100)
    assert result.metrics["gated_quantities"] == 2.0
    assert result.metrics["train_steps_recorded"] == 100.0


def test_e2e_sft_train_validate_two_identical_trajectories_pass_and_still_record_every_step() -> None:
    losses = [2.5 - 0.01 * step for step in range(10)]
    result = _e2e_sft_train_validate_a_verdict(losses, list(losses))
    assert result.outcome is Outcome.PASS
    assert result.mismatches == []
    assert result.reason == ""
    assert result.metrics["stated_criterion_abs"] == check.TOLERANCE
    assert result.metrics["first_disagreeing_train_step"] == 0.0
    assert [f"train_loss_abs_diff_step_{step:03d}" in result.metrics for step in range(1, 11)] == [True] * 10


def test_e2e_sft_train_validate_a_mid_run_disagreement_that_closes_by_the_final_step_does_not_decide_the_outcome() -> (
    None
):
    """The trajectory is reported and not gated, so the widest step does not reach ``max_abs_diff``."""
    reference = [2.5] * 10
    target = [
        2.5,
        2.5,
        2.5 + _e2e_sft_train_validate_OVER_GATE * 2,
        2.5 + _e2e_sft_train_validate_OVER_GATE,
    ] + [2.5] * 6
    result = _e2e_sft_train_validate_a_verdict(reference, target)
    assert result.outcome is Outcome.PASS
    assert result.metrics["final_train_loss_abs_diff"] == pytest.approx(0.0)
    assert result.metrics["max_abs_diff"] == pytest.approx(0.0)
    assert result.metrics["trajectory_steps_over_gate"] == 2.0
    assert result.metrics["worst_train_loss_step"] == 3.0
    assert result.metrics["worst_train_loss_abs_diff"] == pytest.approx(_e2e_sft_train_validate_OVER_GATE * 2)
    assert result.metrics["first_disagreeing_train_step"] == 3.0


def test_e2e_sft_train_validate_a_disagreement_at_step_one_is_read_as_the_forward_and_not_as_accumulation() -> None:
    result = _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5 + _e2e_sft_train_validate_OVER_GATE] * 10)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["first_disagreeing_train_step"] == 1.0
    assert "step 1" in result.reason
    assert "not one that accumulated" in result.reason


def test_e2e_sft_train_validate_a_disagreement_that_appears_later_is_read_as_accumulation() -> None:
    """The distinction the trajectory exists for: same final magnitude, opposite cause."""
    result = _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] * 5 + [2.5 + _e2e_sft_train_validate_OVER_GATE] * 5)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["first_disagreeing_train_step"] == 6.0
    assert "step 6 of 10" in result.reason
    assert "accumulated over 5 optimizer steps" in result.reason


def test_e2e_sft_train_validate_an_onset_at_the_second_step_counts_one_elapsed_optimizer_step() -> None:
    """The sentence counts optimizer steps, and one of them is a step rather than steps."""
    result = _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] + [2.5 + _e2e_sft_train_validate_OVER_GATE] * 9)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["first_disagreeing_train_step"] == 2.0
    assert "first disagree at step 2 of 10" in result.reason
    assert "accumulated over 1 optimizer step rather" in result.reason


def test_e2e_sft_train_validate_the_final_training_loss_fails_on_its_own() -> None:
    result = _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] * 9 + [2.5 + _e2e_sft_train_validate_OVER_GATE])
    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [check.TRAIN_LOSS_QUANTITY.format(step=10)]
    assert result.metrics["final_train_loss_abs_diff"] == pytest.approx(_e2e_sft_train_validate_OVER_GATE)


def test_e2e_sft_train_validate_a_validation_disagreement_fails_on_its_own() -> None:
    result = _e2e_sft_train_validate_a_verdict(
        [2.5] * 10,
        [2.5] * 10,
        reference_validation=1.75,
        target_validation=1.75 + _e2e_sft_train_validate_OVER_GATE,
    )
    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [check.VALIDATION_QUANTITY]
    assert result.reason == ""
    assert result.metrics["validation_abs_diff"] == pytest.approx(_e2e_sft_train_validate_OVER_GATE)


def test_e2e_sft_train_validate_the_first_and_widest_step_are_reported_whatever_the_verdict() -> None:
    """A passing run still states where the two runs started and how far apart they ever got."""
    under = check.TOLERANCE / 2
    result = _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] + [2.5 + under] * 8 + [2.5])
    assert result.outcome is Outcome.PASS
    assert result.metrics["first_train_loss_abs_diff"] == 0.0
    assert result.metrics["worst_train_loss_abs_diff"] == pytest.approx(under)
    assert "step 1 0.000e-03" in result.summary
    assert "neither gated" in result.summary


def test_e2e_sft_train_validate_the_stated_gate_decides_the_outcome() -> None:
    """The stated gate, asserted where it acts: either side of ``TOLERANCE`` decides the outcome."""
    inside, outside = (check.TOLERANCE * 0.9, check.TOLERANCE * 1.1)
    assert _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] * 9 + [2.5 + inside]).outcome is Outcome.PASS
    assert _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] * 9 + [2.5 + outside]).outcome is Outcome.FAIL
    assert _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] * 10, 1.75, 1.75 + inside).outcome is Outcome.PASS
    assert _e2e_sft_train_validate_a_verdict([2.5] * 10, [2.5] * 10, 1.75, 1.75 + outside).outcome is Outcome.FAIL


def test_e2e_sft_train_validate_reference_fused_cross_entropy_matches_product_selection(
    tmp_path,
) -> None:
    from arctic_platform.correctness.harness.config import LoadedConfig

    def cfg(training):
        return LoadedConfig(
            "example",
            tmp_path / "example.config",
            {"model_name": "example", "training_config": training},
        )

    assert cfg({"model_provider": "prime_rl"}).fused_cross_entropy == "liger"
    assert cfg({"model_provider": "prime_rl", "fused_cross_entropy": False}).fused_cross_entropy is False
    assert cfg({"model_provider": "huggingface"}).fused_cross_entropy is False


@pytest.mark.parametrize("name", ["AdamW", "FusedAdam", "torch_adamw"])
def test_e2e_sft_train_validate_reference_uses_independent_unfused_pytorch_adamw(
    name,
) -> None:
    trajectory = AdamWTrajectory(
        _e2e_sft_train_validate_model(),
        {**_e2e_sft_train_validate_ADAMW, "name": name},
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
    )
    trajectory.step(_e2e_sft_train_validate_gradients(trajectory._model))
    assert type(trajectory._optimizer) is torch.optim.AdamW
    assert trajectory._optimizer.defaults["fused"] is False
    assert trajectory._optimizer.defaults["foreach"] is False


def test_e2e_sft_train_validate_trajectories_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="one loss per step"):
        _e2e_sft_train_validate_a_verdict([2.5] * 5, [2.5] * 4)


def _e2e_sft_train_validate_model() -> torch.nn.Module:
    """A weight and a bias, so the no-decay group is populated as it is for a real model."""
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3, bias=True)


def _e2e_sft_train_validate_gradients(model, value: float = 0.1) -> dict:
    return {name: torch.full_like(parameter, value) for name, parameter in model.named_parameters()}


def _e2e_sft_train_validate_trajectory(model) -> AdamWTrajectory:
    return AdamWTrajectory(
        model,
        _e2e_sft_train_validate_ADAMW,
        learning_rate=0.01,
        gradient_clipping=None,
        optimizer_dtype="float32",
    )


def test_e2e_sft_train_validate_the_optimizer_moments_persist_across_steps() -> None:
    """Adam's step counter and both moments carry forward; a rebuilt optimizer would report step 1."""
    model = _e2e_sft_train_validate_model()
    trajectory = _e2e_sft_train_validate_trajectory(model)
    gradients = _e2e_sft_train_validate_gradients(model)
    trajectory.step(gradients)
    after_one = trajectory.state_norms()
    trajectory.step(gradients)
    after_two = trajectory.state_norms()
    assert trajectory.steps == 2
    assert {entry["step"] for entry in after_one.values()} == {1.0}
    assert {entry["step"] for entry in after_two.values()} == {2.0}
    assert after_two["weight"]["exp_avg_sq"] > after_one["weight"]["exp_avg_sq"]


def test_e2e_sft_train_validate_a_step_reads_the_moments_the_step_before_it_wrote() -> None:
    """One gradient, two optimizers: the update differs by exactly the history one of them carries.

    An Adam update at zero weight decay depends on the gradient and on the moments, not on the parameter
    value, so the two deltas below differ for one reason only. A trajectory that rebuilt its optimizer per
    step, or cleared its state, would produce the fresh delta at every step of a hundred-step run.
    """
    carried = _e2e_sft_train_validate_model()
    trajectory = _e2e_sft_train_validate_trajectory(carried)
    trajectory.step(_e2e_sft_train_validate_gradients(carried, 0.1))
    before = carried.weight.detach().clone()
    trajectory.step(_e2e_sft_train_validate_gradients(carried, 0.5))
    carried_delta = carried.weight.detach() - before
    fresh = _e2e_sft_train_validate_model()
    start = fresh.weight.detach().clone()
    _e2e_sft_train_validate_trajectory(fresh).step(_e2e_sft_train_validate_gradients(fresh, 0.5))
    fresh_delta = fresh.weight.detach() - start
    assert not torch.allclose(carried_delta, fresh_delta, atol=1e-09)
    assert carried_delta.abs().max().item() < fresh_delta.abs().max().item()


def test_e2e_sft_train_validate_the_step_is_written_back_into_the_model() -> None:
    """The next step's forward reads the model, so a step that only moved the master copy is lost."""
    model = _e2e_sft_train_validate_model()
    trajectory = _e2e_sft_train_validate_trajectory(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trajectory.step(_e2e_sft_train_validate_gradients(model))
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter.detach(), before[name])


def test_e2e_sft_train_validate_the_single_step_reference_cannot_be_looped(
    tmp_path,
) -> None:
    """Why the trajectory exists: ``run_adamw_step`` clears its state, so twice is once, twice over."""
    model = _e2e_sft_train_validate_model()
    gradients = _e2e_sft_train_validate_gradients(model)
    updates = []
    manifests = []
    for attempt in (1, 2):
        artifact = run_adamw_step(
            model,
            gradients,
            _e2e_sft_train_validate_ADAMW,
            learning_rate=0.01,
            gradient_clipping=None,
            optimizer_dtype="float32",
            output_dir=tmp_path / f"attempt-{attempt}",
        )
        manifests.append(json.loads(artifact.manifest_path.read_text()))
        updates.append({entry["name"]: entry["parameter_update_norm"] for entry in manifests[-1]["parameters"]})
    assert updates[0] == updates[1]
    assert [manifest["step"] for manifest in manifests] == [1, 1]


def test_e2e_sft_train_validate_a_gradient_missing_for_an_owned_parameter_is_refused() -> None:
    model = _e2e_sft_train_validate_model()
    trajectory = _e2e_sft_train_validate_trajectory(model)
    gradients = _e2e_sft_train_validate_gradients(model)
    trajectory.step(gradients)
    with pytest.raises(ValueError, match="the trainable set changed mid-run"):
        trajectory.step({"weight": gradients["weight"]})


def test_e2e_sft_train_validate_an_optimizer_that_is_not_adamw_is_refused() -> None:
    with pytest.raises(ValueError, match="AdamW semantics"):
        AdamWTrajectory(
            _e2e_sft_train_validate_model(),
            {"name": "SGD"},
            learning_rate=0.01,
            gradient_clipping=None,
            optimizer_dtype="float32",
        )


# Gdn determinism.


def test_gdn_determinism_reference_reduction_pinning_does_not_mutate_fla_autotuners() -> None:
    assert "pin_fla_gdn_autotuners" not in inspect.getsource(process_determinism)


# Gsm8k.


def _gsm8k_example(prompt: str, completion: str, input_ids, n_prompt: int) -> gsm8k.Example:
    return gsm8k.Example(
        question=prompt,
        prompt=prompt,
        completion=completion,
        input_ids=tuple(input_ids),
        n_prompt=n_prompt,
    )


def test_gsm8k_the_boundary_is_the_common_prefix_not_the_prompt_length() -> None:
    """A prompt whose last token merges with the completion's first tokenizes one way alone and another
    way joined. Taking ``len(prompt_ids)`` would score a position the joined sequence does not have there.
    """
    prompt_ids = [9, 4, 7, 88]
    full_ids = [9, 4, 7, 91, 12, 13]
    assert gsm8k.answer_boundary(prompt_ids, full_ids) == 3
    assert gsm8k.answer_boundary(prompt_ids, full_ids) != len(prompt_ids)


def test_gsm8k_a_prompt_that_survives_whole_gives_its_own_length() -> None:
    assert gsm8k.answer_boundary([9, 4, 7], [9, 4, 7, 12, 13]) == 3


def test_gsm8k_labels_cover_the_answer_span_only() -> None:
    rows = [
        _gsm8k_example("q1", " a1", [5, 6, 7, 8], 2),
        _gsm8k_example("q2", " a2", [1, 2, 3], 1),
    ]
    batch = gsm8k.build_batch("validation", rows, pad_id=0)
    assert batch.input_ids.tolist() == [[5, 6, 7, 8], [1, 2, 3, 0]]
    assert batch.labels.tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 7, 8],
        [IGNORE_INDEX, 2, 3, IGNORE_INDEX],
    ]
    assert batch.active_tokens == sum((row.answer_tokens for row in rows))


def test_gsm8k_the_shift_puts_each_answer_token_on_the_logit_that_predicts_it() -> None:
    """``shifted_labels()[r, s]`` is the token at ``s + 1``, which is what a logit at ``s`` predicts."""
    rows = [_gsm8k_example("q1", " a1", [5, 6, 7, 8], 2)]
    shifted = gsm8k.build_batch("validation", rows, pad_id=0).shifted_labels()
    assert shifted.tolist() == [[IGNORE_INDEX, 7, 8, IGNORE_INDEX]]


def test_gsm8k_positions_are_row_local_and_padding_is_not_numbered() -> None:
    rows = [
        _gsm8k_example("q1", " a1", [5, 6, 7, 8], 2),
        _gsm8k_example("q2", " a2", [1, 2, 3], 1),
    ]
    batch = gsm8k.build_batch("validation", rows, pad_id=0)
    assert batch.position_ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 0]]


def test_gsm8k_the_slice_is_the_largest_that_fits_one_forward() -> None:
    """Row count times padded width is what the forward holds, and the width is the longest row taken."""
    rows = [_gsm8k_example(f"q{index}", " a", list(range(length)), 1) for index, length in enumerate([4, 6, 5, 40])]
    chosen, width = gsm8k.select_by_token_budget(rows, token_budget=17, minimum_rows=1)
    assert [row.length for row in chosen] == [4, 6]
    assert width == 6
    assert len(chosen) * width == 12
    assert 3 * width > 17


def test_gsm8k_a_slice_too_narrow_for_the_data_parallel_shards_is_refused() -> None:
    rows = [
        _gsm8k_example("q0", " a", list(range(10)), 1),
        _gsm8k_example("q1", " a", list(range(10)), 1),
    ]
    with pytest.raises(ValueError, match="data-parallel"):
        gsm8k.select_by_token_budget(rows, token_budget=10, minimum_rows=2)


def test_gsm8k_an_empty_tokenizer_is_a_loud_failure() -> None:
    """The observed silent failure: no sidecars, one vocabulary entry, every string encodes to nothing."""
    with pytest.raises(ValueError, match="cannot represent"):
        gsm8k.assert_vocabulary_matches_model(1, 151936, "/models/reduced")


def test_gsm8k_a_tokenizer_wider_than_the_embedding_table_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot look up"):
        gsm8k.assert_vocabulary_matches_model(200000, 151936, "/models/reduced")


def test_gsm8k_a_tokenizer_inside_the_embedding_table_is_accepted() -> None:
    gsm8k.assert_vocabulary_matches_model(151669, 151936, "/models/reduced")


def test_gsm8k_the_padded_columns_fall_back_to_the_end_of_sequence_token() -> None:
    assert gsm8k.padding_token_id(SimpleNamespace(pad_token_id=7, eos_token_id=9)) == 7
    assert gsm8k.padding_token_id(SimpleNamespace(pad_token_id=None, eos_token_id=9)) == 9
    with pytest.raises(ValueError):
        gsm8k.padding_token_id(SimpleNamespace(pad_token_id=None, eos_token_id=None))


def test_gsm8k_a_shared_question_is_refused() -> None:
    shared = _gsm8k_example("same question", " a", [1, 2, 3], 1)
    with pytest.raises(ValueError, match="both"):
        gsm8k.assert_disjoint([shared], [shared])


def test_gsm8k_the_validation_split_shares_no_question_with_the_training_split() -> None:
    """The two slices the check reads, from the dataset's own files, with nothing in common.

    A validation loss measured on rows the run trained on would agree between the two engines for a reason
    that has nothing to do with the checkpoint, so the disjointness is read off the data rather than
    inferred from the file names.
    """
    from arctic_platform.correctness.checks import inference_checkpoint_loss as check

    training = gsm8k.read_rows(check.TRAIN_SPLIT)
    held_out = gsm8k.read_rows(check.VALIDATION_SPLIT)
    assert training and held_out
    assert not {question for question, _ in training} & {question for question, _ in held_out}


# Lora init.


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


# Qwen4 reference.


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


# Sft.


@pytest.mark.parametrize(
    "name",
    [
        "TRAIN_SPLIT",
        "VALIDATION_SPLIT",
        "configured_learning_rate",
        "configured_rows_per_step",
        "cross_entropy",
        "reference_answer_logprobs",
    ],
)
def test_sft_check_compatibility_exports_are_shared_objects(name):
    assert getattr(e2e, name) is getattr(sft, name)
    assert getattr(inference, name) is getattr(sft, name)


def test_sft_e2e_shared_helpers_do_not_import_another_check():
    tree = ast.parse(Path(e2e.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any((node.level == 1 or "inference_checkpoint_loss" in (node.module or "") for node in imports))
    names = {alias.name for node in imports if node.module == "harness.sft" for alias in node.names}
    assert names == {
        "TRAIN_SPLIT",
        "VALIDATION_SPLIT",
        "configured_learning_rate",
        "configured_rows_per_step",
        "cross_entropy",
        "reference_answer_logprobs",
    }


def test_sft_shared_sft_has_no_check_dependency():
    tree = ast.parse(Path(sft.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in imports:
        assert "checks" not in (getattr(node, "module", None) or "").split(".")
        assert all(("checks" not in alias.name.split(".") for alias in node.names))


def test_sft_shared_dataset_locations_are_unchanged():
    assert sft.TRAIN_SPLIT == Path("/code/shared/gsm8k/train.parquet")
    assert sft.VALIDATION_SPLIT == Path("/code/shared/gsm8k/test.parquet")


@pytest.mark.parametrize("rate,rows", [(2e-05, 16), ("0.00002", "16"), (0.0, 1)])
def test_sft_configured_settings_are_read_without_defaults(rate, rows):
    cfg = SimpleNamespace(config_id="sft", training={"optimizer": {"lr": rate}, "train_batch_size": rows})
    assert sft.configured_learning_rate(cfg) == float(rate)
    assert sft.configured_rows_per_step(cfg) == int(rows)


@pytest.mark.parametrize("optimizer", [None, "AdamW", {}, {"lr": None}])
def test_sft_missing_configured_learning_rate_keeps_the_original_error(optimizer):
    cfg = SimpleNamespace(config_id="sft", training={"optimizer": optimizer})
    with pytest.raises(ValueError, match="sft: the training config declares no optimizer learning rate"):
        sft.configured_learning_rate(cfg)


@pytest.mark.parametrize("training", [{}, {"train_batch_size": None}])
def test_sft_missing_configured_rows_keeps_the_original_error(training):
    cfg = SimpleNamespace(config_id="sft", training=training)
    with pytest.raises(ValueError, match="sft: the training config declares no train_batch_size"):
        sft.configured_rows_per_step(cfg)


def test_sft_cross_entropy_preserves_scalar_accumulation_order():
    assert sft.cross_entropy([[-1e16, -1.0], [1e16, -2.0]]) == (0.5, 4)


def test_sft_cross_entropy_pools_tokens_and_accepts_empty_individual_rows():
    assert sft.cross_entropy([[-1.0] * 4, [], [-5.0]]) == (1.8, 5)


def test_sft_cross_entropy_rejects_no_scored_tokens():
    with pytest.raises(ValueError, match="not a measurement"):
        sft.cross_entropy([[], []])


def _sft_example(n_prompt=2):
    return gsm8k.Example(
        question="q",
        prompt="p",
        completion=" a",
        input_ids=(1, 2, 3, 4),
        n_prompt=n_prompt,
    )


def test_sft_reference_span_excludes_prompt_padding_and_unpredicted_position():
    logprobs = torch.tensor([[float("nan"), -1.0, -2.0, float("inf"), float("nan")]])
    assert sft.reference_answer_logprobs(logprobs, [_sft_example()]) == [[-1.0, -2.0]]


@pytest.mark.parametrize("shape", [(5,), (2, 5), (1, 1, 5)])
def test_sft_reference_scoring_rejects_non_row_aligned_tensors(shape):
    with pytest.raises(RuntimeError, match="two-dimensional row"):
        sft.reference_answer_logprobs(torch.zeros(shape), [_sft_example()])


@pytest.mark.parametrize("n_prompt", [0, 4, 5])
def test_sft_reference_scoring_rejects_invalid_answer_spans(n_prompt):
    with pytest.raises(RuntimeError, match="answer span must follow a prompt token"):
        sft.reference_answer_logprobs(torch.zeros(1, 5), [_sft_example(n_prompt)])


def test_sft_reference_scoring_rejects_truncated_answer_response():
    with pytest.raises(RuntimeError, match="covers 1 of the row's 2 answer tokens"):
        sft.reference_answer_logprobs(torch.zeros(1, 2), [_sft_example()])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_sft_reference_scoring_rejects_nonfinite_answer_tokens(value):
    with pytest.raises(RuntimeError, match="answer log-probabilities must be finite"):
        sft.reference_answer_logprobs(torch.tensor([[0.0, value, -1.0, 0.0]]), [_sft_example()])


# Lora adapter storage evidence.


class Factors:
    def __init__(self, module_name, *, stacked=False):
        self.module_name, self.rank, self.lora_alpha, self.scaling = (
            module_name,
            2,
            3,
            1.5,
        )
        self.lora_a = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        self.lora_b = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
        if stacked:
            self.lora_a = self.lora_a.unsqueeze(0)
            self.lora_b = self.lora_b.unsqueeze(0)
        self.calls = 0

    def optimize(self):
        self.calls += 1
        if self.scaling != 1:
            self.lora_b *= self.scaling
            self.scaling = 1
        return self


class Packed:
    calls = 0
    clone_projection = False

    @classmethod
    def pack_moe_stacked(cls, loras, module_name):
        cls.calls += 1
        result = cls()
        result.module_name = module_name
        result.rank, result.lora_alpha = loras[0].rank, 0
        result.lora_alphas = [loras[0].lora_alpha] * 3
        result.scaling = [loras[0].scaling] * 3
        result.lora_a = [weights.lora_a for weights in loras]
        result.lora_b = [weights.lora_b for weights in loras]
        if cls.clone_projection:
            result.lora_b[1] = result.lora_b[1].clone()
        result.calls = 0
        return result

    def optimize(self):
        self.calls += 1
        for index in range(len(self.lora_b)):
            if self.scaling[index] != 1:
                self.lora_b[index] *= self.scaling[index]
                self.scaling[index] = 1
        return self


class WrongScaling(Factors):
    def optimize(self):
        self.calls += 1
        self.lora_b += self.scaling
        self.scaling = 1
        return self


class NativeSlotFixture:
    def __init__(self):
        self.lora_a_stacked = [torch.zeros((2, 4, 3), dtype=torch.bfloat16)]
        self.lora_b_stacked = [torch.zeros((2, 3, 4), dtype=torch.bfloat16)]
        self.calls = 0
        self.failure = None
        self.changed_source = None
        self.foreign = torch.ones((2, 3), dtype=torch.bfloat16)

    def set_lora(self, index, lora_a, lora_b):
        self.calls += 1
        if self.failure == "factor_mutation":
            lora_a.add_(1)
        if self.failure == "unbound":
            self.lora_a_stacked[0][index, :2].copy_(self.foreign)
        else:
            self.lora_a_stacked[0][index].zero_()
            self.lora_b_stacked[0][index].zero_()
            values = lora_a[:, 3:6]
            if self.failure == "nonfinite":
                values = values / 0
            self.lora_a_stacked[0][index, :2].copy_(values)
            if self.failure != "ignored":
                self.lora_b_stacked[0][index, :, :2].copy_(lora_b[3:6])
        if self.failure == "identity":
            self.lora_a_stacked[0] = self.lora_a_stacked[0].clone()
        if self.changed_source is not None:
            self.changed_source.write_text("changed while native setter was active\n")
        return {"actual_native_return": index}


@pytest.fixture
def admitted(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    monkeypatch.setattr(Packed, "calls", 0)
    monkeypatch.setattr(Packed, "clone_projection", False)
    return source_files(
        evidence.observe_stacked_moe_packing,
        evidence.observe_lora_scaling,
        evidence.observe_lora_storage_write,
        Factors,
        Packed,
        WrongScaling,
        NativeSlotFixture,
        TorchDispatchMode,
        tree_flatten,
        tree_map,
    )


def _pack(admitted, loras=None, **updates):
    if loras is None:
        loras = [Factors(f"layer.experts.w{index}", stacked=True) for index in (1, 2, 3)]
    request = {
        "pack": Packed.pack_moe_stacked,
        "max_shadow_bytes": 4096,
        "admitted_sources": admitted,
    }
    request.update(updates)
    return evidence.observe_stacked_moe_packing(loras, "layer.experts", **request)


def _write(admitted, layer=None, **updates):
    layer = layer or NativeSlotFixture()
    a = torch.arange(12, dtype=torch.bfloat16).reshape(2, 6)
    b = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    request = {
        "storage_paths": {"A": ("lora_a_stacked", 0), "B": ("lora_b_stacked", 0)},
        "factor_tensors": {"A": a, "B": b},
        "max_shadow_bytes": 4096,
        "admitted_sources": admitted,
    }
    request.update(updates)
    return (
        layer,
        a,
        b,
        evidence.observe_lora_storage_write(layer.set_lora, 1, a, b, **request),
    )


def _nonadmissions(record):
    for name in (
        "complete_adapter_inventory_verified",
        "tensor_parallel_slot_routing_verified",
        "adapter_storage_verified",
        "adapter_application_verified",
        "full_parameter_coverage",
        "capacity_admitted",
    ):
        assert record[name] is False
    assert record["numerical"] is None
    assert record["native_python_calls"] == 1


def test_adapter_stacked_pack_retains_projection_order_aliases_and_exact_bytes(
    admitted,
):
    loras = [Factors(f"layer.experts.w{index}", stacked=True) for index in (1, 2, 3)]
    before = [tensor_manifest(weights.lora_b) for weights in loras]
    observed = _pack(admitted, loras)
    assert Packed.calls == 1
    assert all(observed.value.lora_b[index] is weights.lora_b for index, weights in enumerate(loras))
    assert [tensor_manifest(weights.lora_b) for weights in loras] == before
    record = observed.evidence
    assert record["projection_order"] == ["w1", "w2", "w3"]
    assert record["factor_bytes"] == 84 and record["private_shadow_bytes"] == 0
    assert record["matched_retained_native_weights_source"] is False
    assert record["ordered_aliases_verified"] is True
    _nonadmissions(record)


@pytest.mark.parametrize("cause", ["order", "rank", "alpha", "scale", "alias", "budget"])
def test_adapter_stacked_pack_rejects_incomplete_native_domain_before_call(admitted, cause):
    loras = [Factors(f"layer.experts.w{index}", stacked=True) for index in (1, 2, 3)]
    request = {}
    if cause == "order":
        loras[1], loras[2] = loras[2], loras[1]
    elif cause == "rank":
        loras[1].rank = 3
    elif cause == "alpha":
        loras[1].lora_alpha = 4
    elif cause == "scale":
        loras[1].scaling = 2.0
    elif cause == "alias":
        loras[1].lora_a = loras[0].lora_a
    else:
        request["max_shadow_bytes"] = 1
    with pytest.raises(ValueError):
        _pack(admitted, loras, **request)
    assert Packed.calls == 0


def test_adapter_stacked_pack_rejects_equal_bytes_with_wrong_tensor_alias(admitted, monkeypatch):
    monkeypatch.setattr(Packed, "clone_projection", True)
    with pytest.raises(ValueError, match="ordered factor aliases"):
        _pack(admitted)
    assert Packed.calls == 1


@pytest.mark.parametrize("packed", [False, True])
def test_adapter_optimize_observes_one_same_dtype_scaling_call(admitted, packed):
    weights = _pack(admitted).value if packed else Factors("layer.proj")
    a_values = weights.lora_a if packed else [weights.lora_a]
    b_values = weights.lora_b if packed else [weights.lora_b]
    initial_a = [tensor_manifest(value) for value in a_values]
    expected_b = [value.clone().mul_(1.5) for value in b_values]
    observed = evidence.observe_lora_scaling(weights, max_shadow_bytes=4096, admitted_sources=admitted)
    assert observed.value is weights and weights.calls == 1
    assert [tensor_manifest(value) for value in a_values] == initial_a
    assert all(tensor_manifest(actual) == tensor_manifest(expected) for actual, expected in zip(b_values, expected_b))
    assert observed.evidence["private_shadow_bytes"] == (48 if packed else 16)
    assert observed.evidence["same_dtype_scaling_bytes_verified"] is True
    assert observed.evidence["matched_retained_native_weights_source"] is False
    _nonadmissions(observed.evidence)


def test_adapter_optimize_rejects_native_arithmetic_other_than_recorded_scaling(
    admitted,
):
    weights = WrongScaling("layer.proj")
    with pytest.raises(ValueError, match="same-dtype scaled B"):
        evidence.observe_lora_scaling(weights, max_shadow_bytes=4096, admitted_sources=admitted)
    assert weights.calls == 1


def test_adapter_optimize_shadow_budget_is_checked_before_actual_native_call(admitted):
    weights = Factors("layer.proj")
    with pytest.raises(ValueError, match="explicit limit"):
        evidence.observe_lora_scaling(weights, max_shadow_bytes=1, admitted_sources=admitted)
    assert weights.calls == 0


def test_adapter_actual_slot_writes_mirror_tp_slices_without_replaying_python(admitted):
    layer, a, b, observed = _write(admitted)
    assert layer.calls == 1 and observed.value == {"actual_native_return": 1}
    assert torch.equal(layer.lora_a_stacked[0][1, :2], a[:, 3:6])
    assert torch.equal(layer.lora_b_stacked[0][1, :, :2], b[3:6])
    assert torch.count_nonzero(layer.lora_a_stacked[0][0]) == 0
    record = observed.evidence
    assert record["observed_write_shadow_verified"] is True
    assert record["elementwise_factor_origin_verified"] is False
    assert record["native_python_call_replayed"] is False
    assert record["factor_names_reaching_observed_writes"] == ["A", "B"]
    assert record["private_shadow_bytes"] == 240
    assert len(record["writes"]) == 4
    assert [entry["logical_write_bytes"] for entry in record["writes"]] == [
        24,
        24,
        12,
        12,
    ]
    for entry in record["destinations"].values():
        assert entry["storage_bytes"] == 48
        assert entry["covered_elements"] == entry["uncovered_elements"] == 12
        assert entry["mismatched_covered_elements"] == 0
        assert entry["changed_uncovered_elements"] == 0
    _nonadmissions(record)


@pytest.mark.parametrize("cause", ["factor_mutation", "unbound"])
def test_adapter_native_write_rejects_unowned_side_effect_before_dispatch(admitted, cause):
    layer = NativeSlotFixture()
    layer.failure = cause
    before = tensor_manifest(layer.lora_a_stacked[0])
    with pytest.raises(ValueError, match="overwrite bound|unbound tensor"):
        _write(admitted, layer)
    assert layer.calls == 1
    assert tensor_manifest(layer.lora_a_stacked[0]) == before


def test_adapter_native_write_cannot_ignore_one_supplied_factor(admitted):
    layer = NativeSlotFixture()
    layer.failure = "ignored"
    with pytest.raises(ValueError, match="every bound adapter factor"):
        _write(admitted, layer)
    assert layer.calls == 1


def test_adapter_native_write_cannot_swap_native_storage_after_mirrored_writes(
    admitted,
):
    layer = NativeSlotFixture()
    layer.failure = "identity"
    with pytest.raises(ValueError, match="storage identities"):
        _write(admitted, layer)
    assert layer.calls == 1


def test_adapter_native_write_cannot_count_nonfinite_values_as_uncovered_padding(
    admitted,
):
    layer = NativeSlotFixture()
    layer.failure = "nonfinite"
    with pytest.raises(ValueError, match="finite input"):
        _write(admitted, layer)
    assert layer.calls == 1


def test_adapter_source_drift_rejected_before_actual_setter(admitted, tmp_path):
    path = tmp_path / "selected_source.py"
    path.write_text("original source\n")
    admitted[str(path)] = "0" * 64
    layer = NativeSlotFixture()
    with pytest.raises(RuntimeError, match="source changed"):
        _write(admitted, layer)
    assert layer.calls == 0


def test_adapter_source_drift_during_actual_setter_never_yields_a_positive_receipt(admitted, tmp_path):
    import hashlib

    path = tmp_path / "selected_source.py"
    path.write_text("original source\n")
    admitted[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    layer = NativeSlotFixture()
    layer.changed_source = path
    with pytest.raises(RuntimeError, match="source changed"):
        _write(admitted, layer)
    assert layer.calls == 1


# Lora adapter tensor binding.


class Mapper:
    def __init__(self, prefix="mapped."):
        self.prefix = prefix
        self.orig_to_new_renaming = []
        self.orig_to_new_stacked = {}

    def get_rename_mapper(self):
        return Mapper(self.prefix)

    def map_name(self, name):
        return self._map_name_with_shard(name)[0]

    def _map_name(self, name):
        return self.map_name(name)

    def _map_name_with_shard(self, name):
        return self.prefix + name, None


class Served:
    def __init__(self, *, mapper=False):
        self.hf_to_vllm_mapper = Mapper() if mapper else None

    def named_modules(self):
        return [("language_model.model.layers.0", self)]


class LoRALayerWeights:
    def __init__(self, name, config):
        self.module_name, self.rank = name, config["r"]
        self.lora_alpha = config["lora_alpha"]
        self.scaling = self.lora_alpha / self.rank
        self.lora_a = self.lora_b = None


class PEFTHelper:
    @classmethod
    def from_dict(cls, config):
        return dict(config)


def parse_fine_tuned_lora_name(name, mapper):
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    if mapper is not None:
        name = mapper._map_name(name)
    module, factor, _ = name.rsplit(".", 2)
    return module, factor == "lora_A"


class LoRAModel:
    calls = 0
    failure = None
    source_path = None

    @classmethod
    def from_lora_tensors(cls, *, lora_model_id, tensors, peft_helper, device, dtype, weights_mapper):
        parts = {}
        for name, tensor in tensors.items():
            module, is_a = parse_fine_tuned_lora_name(name, weights_mapper)
            if module not in parts:
                parts[module] = LoRALayerWeights(module, peft_helper)
            value = tensor.to(device=device, dtype=dtype)
            if cls.failure == "copy":
                value = value.clone()
            setattr(parts[module], "lora_a" if is_a else "lora_b", value)
        result = cls()
        result.id, result.rank, result.loras = lora_model_id, peft_helper["r"], parts
        result.modules_to_save, result.is_3d_lora_weight = {}, False
        return result


def _served_lm_module_prefix(model):
    return "language_model.model." if model is not None else "model."


def _remap_lora_key(name, prefix):
    name = name.replace("._checkpoint_wrapped_module", "")
    if prefix == "model.":
        return name
    head = "base_model.model." if name.startswith("base_model.model.") else ""
    body = name[len(head) :]
    for root in ("model.language_model.", "model."):
        if body.startswith(root):
            body = prefix + body[len(root) :]
            break
    return head + body


def _pack_fused_expert_loras(result):
    groups = {}
    for name, weights in list(result.loras.items()):
        for index in (1, 2, 3):
            if name.endswith(f".experts.w{index}") or name == f"experts.w{index}":
                groups.setdefault(name[:-3], {})[index] = weights
    for name, parts in groups.items():
        ordered = [parts[index] for index in (1, 2, 3)]
        weights = LoRALayerWeights(name, {"r": ordered[0].rank, "lora_alpha": 0})
        weights.lora_a, weights.lora_b = [part.lora_a for part in ordered], [part.lora_b for part in ordered]
        weights.lora_alphas = [part.lora_alpha for part in ordered]
        weights.scaling = [part.scaling for part in ordered]
        result.loras[name] = weights
        for part in ordered:
            del result.loras[part.module_name]


def _build_lora_model(*, lora_int_id, tensors, lora_config, device, dtype, model):
    LoRAModel.calls += 1
    prefix = _served_lm_module_prefix(model)
    mapper = model.hf_to_vllm_mapper.get_rename_mapper() if model and model.hf_to_vllm_mapper else None
    result = LoRAModel.from_lora_tensors(
        lora_model_id=lora_int_id,
        tensors={_remap_lora_key(name, prefix): value for name, value in tensors.items()},
        peft_helper=PEFTHelper.from_dict(lora_config),
        device=device,
        dtype=dtype,
        weights_mapper=mapper,
    )
    _pack_fused_expert_loras(result)
    failure = LoRAModel.failure
    weights = next(iter(result.loras.values()))
    if failure == "type":
        return SimpleNamespace(**vars(result))
    if failure == "identity":
        result.id += 1
    elif failure == "drop":
        result.loras.pop(next(iter(result.loras)))
    elif failure == "extra":
        result.loras["unsupported"] = weights
    elif failure == "order":
        result.loras = dict(reversed(list(result.loras.items())))
    elif failure == "bytes":
        weights.lora_a = weights.lora_a.clone().add_(1)
    elif failure == "rank":
        weights.rank += 1
    elif failure == "scaling":
        weights.scaling = 1
    elif failure == "packed_order":
        weights.lora_b = list(reversed(weights.lora_b))
    elif failure == "input":
        next(iter(tensors.values())).add_(1)
    elif failure == "config":
        lora_config["r"] += 1
    elif failure == "mapper":
        model.hf_to_vllm_mapper.prefix = "changed."
    elif failure == "source":
        LoRAModel.source_path.write_text("changed source during native build\n")
    elif failure == "callee":
        globals()["_remap_lora_key"] = lambda name, prefix: name
    return result


def _wire(mode):
    modules = ["model.layers.0.q_proj"]
    if mode == "packed":
        modules = [f"model.layers.0.experts.w{index}" for index in (3, 1, 2)]
    elif mode == "mixed":
        modules += [f"model.layers.0.experts.w{index}" for index in (3, 1, 2)]
        modules += [f"model.layers.1.experts.w{index}" for index in (2, 3, 1)]
    tensors = {}
    for index, name in enumerate(modules):
        stacked = ".experts." in name
        for factor, shape in (("A", (2, 3)), ("B", (4, 2))):
            if stacked:
                shape = (1, *shape)
            tensors[f"base_model.model.{name}.lora_{factor}.weight"] = torch.full(
                shape,
                index + (1 if factor == "A" else 11),
                dtype=torch.bfloat16,
            )
    return tensors


@pytest.fixture
def _lora_adapter_tensor_binding_admitted(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    monkeypatch.setattr(LoRAModel, "calls", 0)
    monkeypatch.setattr(LoRAModel, "failure", None)
    monkeypatch.setattr(LoRAModel, "source_path", None)
    # Register the original binding with monkeypatch so a deliberately replaced
    # builder-global callable is restored after the selected negative control.
    monkeypatch.setitem(globals(), "_remap_lora_key", _remap_lora_key)
    return source_files(
        _lora_adapter_tensor_binding_evidence.observe_lora_build,
        ObservedLoRAOperation,
        _build_lora_model,
    )


def _observe(admitted, *, mode="dense", model=None, **updates):
    request = {
        "lora_int_id": 7,
        "tensors": _wire(mode),
        "lora_config": {"r": 2, "lora_alpha": 3, "bias": "none", "peft_type": "LORA"},
        "device": "cpu",
        "dtype": torch.bfloat16,
        "model": model,
        "native_model_class": LoRAModel,
        "parse_name": parse_fine_tuned_lora_name,
        "remap_key": _remap_lora_key,
        "served_prefix_for_model": _served_lm_module_prefix,
        "max_binding_bytes": 4096,
        "admitted_sources": admitted,
    }
    request.update(updates)
    return _lora_adapter_tensor_binding_evidence.observe_lora_build(_build_lora_model, **request)


@pytest.mark.parametrize("mode", ["dense", "packed", "mixed", "mapper", "copied"])
def test_lora_builder_binds_every_received_factor_once_before_installation(
    _lora_adapter_tensor_binding_admitted, monkeypatch, mode
):
    if mode == "copied":
        monkeypatch.setattr(LoRAModel, "failure", "copy")
    model = Served(mapper=True) if mode == "mapper" else None
    tensors = _wire("dense" if mode == "mapper" else mode)
    before = {name: tensor_manifest(value) for name, value in tensors.items()}
    observed = _observe(_lora_adapter_tensor_binding_admitted, mode=mode, model=model, tensors=tensors)
    assert type(observed.value) is LoRAModel and LoRAModel.calls == 1
    record = observed.evidence
    assert record["original_received"] == before
    assert {name: tensor_manifest(value) for name, value in tensors.items()} == before
    assert record["factor_binding"].keys() == tensors.keys()
    assert record["received_bytes"] == sum(value.numel() * value.element_size() for value in tensors.values())
    assert record["all_received_adapter_factors_bound"] is True
    assert record["built_factor_byte_correspondence_verified"] is True
    assert record["ignored_received_tensors"] == record["extra_built_tensors"] == []
    assert record["matched_retained_model_parser_mapper_sources"] is False
    assert record["built_factor_alias_identity_required"] is False
    assert record["mapper_projection_unchanged"] is True
    assert record["native_builder_calls"] == 1
    if mode == "copied":
        assert next(iter(observed.value.loras.values())).lora_a is not next(iter(tensors.values()))
    if mode in ("packed", "mixed"):
        for entry in record["factor_binding"].values():
            if entry["packed_projection"] is not None:
                assert entry["packed_projection"] in ("w1", "w2", "w3")
                assert f".{entry['packed_projection']}." in next(
                    name for name, bound in record["factor_binding"].items() if bound is entry
                )
    for name in (
        "runtime_admitted",
        "manager_layer_coverage_verified",
        "tensor_parallel_slot_routing_verified",
        "adapter_storage_verified",
        "adapter_application_verified",
        "full_parameter_coverage",
        "capacity_admitted",
    ):
        assert record[name] is False
    assert record["numerical"] is None


@pytest.mark.parametrize(
    "cause",
    [
        "empty",
        "missing_pair",
        "missing_projection",
        "remap_collision",
        "parser_collision",
        "config",
        "dtype",
        "budget",
        "layout",
        "nonfinite",
        "disabled",
        "source",
    ],
)
def test_lora_builder_rejects_unsupported_received_domain_before_native_call(
    _lora_adapter_tensor_binding_admitted, monkeypatch, tmp_path, cause
):
    tensors, updates = _wire("packed" if cause == "missing_projection" else "dense"), {}
    first = next(iter(tensors))
    if cause == "empty":
        tensors.clear()
    elif cause == "missing_pair":
        tensors.pop(first)
    elif cause == "missing_projection":
        tensors = {name: value for name, value in tensors.items() if ".w2." not in name}
    elif cause == "remap_collision":
        tensors[first.replace("model.layers", "model._checkpoint_wrapped_module.layers")] = tensors[first].clone()
    elif cause == "parser_collision":
        tensors[first.removeprefix("base_model.model.")] = tensors[first].clone()
    elif cause == "config":
        updates["lora_config"] = {
            "r": 2,
            "lora_alpha": 3,
            "bias": "none",
            "use_rslora": True,
        }
    elif cause == "dtype":
        updates["dtype"] = torch.float32
    elif cause == "budget":
        updates["max_binding_bytes"] = 1
    elif cause == "layout":
        tensors[first] = torch.zeros((2, 6), dtype=torch.bfloat16)[:, ::2]
    elif cause == "nonfinite":
        tensors[first].fill_(float("nan"))
    elif cause == "disabled":
        monkeypatch.delenv(VERIFY_ENV)
    else:
        sentinel = tmp_path / "source.py"
        sentinel.write_text("old\n")
        _lora_adapter_tensor_binding_admitted[str(sentinel)] = "0" * 64
    with pytest.raises((ValueError, RuntimeError)):
        _observe(_lora_adapter_tensor_binding_admitted, tensors=tensors, **updates)
    assert LoRAModel.calls == 0


@pytest.mark.parametrize(
    "cause",
    [
        "type",
        "identity",
        "drop",
        "extra",
        "order",
        "bytes",
        "rank",
        "scaling",
        "packed_order",
        "input",
        "config",
        "mapper",
        "source",
        "callee",
    ],
)
def test_lora_builder_rejects_drift_ignored_or_changed_factors_after_exactly_one_call(
    _lora_adapter_tensor_binding_admitted,
    monkeypatch,
    tmp_path,
    cause,
):
    monkeypatch.setattr(LoRAModel, "failure", cause)
    mode = "mixed" if cause == "order" else "packed" if cause == "packed_order" else "dense"
    model = Served(mapper=True) if cause == "mapper" else None
    if cause == "source":
        sentinel = tmp_path / "source.py"
        sentinel.write_text("original\n")
        _lora_adapter_tensor_binding_admitted[str(sentinel)] = hashlib.sha256(sentinel.read_bytes()).hexdigest()
        monkeypatch.setattr(LoRAModel, "source_path", sentinel)
    with pytest.raises((ValueError, RuntimeError)):
        _observe(_lora_adapter_tensor_binding_admitted, mode=mode, model=model)
    assert LoRAModel.calls == 1


# Lora receiver bindings.


class Manager:
    def __init__(self):
        self.adapters = {}
        self.calls = []
        self.remove_error = None

    def add_adapter(self, model):
        self.calls.append("add")
        self.adapters[model.id] = model
        return True

    def activate_adapter(self, identity):
        self.calls.append("activate")
        return identity in self.adapters

    def get_adapter(self, identity):
        return self.adapters.get(identity)

    def remove_adapter(self, identity):
        self.calls.append("remove")
        if self.remove_error is not None:
            raise self.remove_error
        self.adapters.pop(identity, None)
        return True


class Teardown:
    def __init__(self):
        self.calls = 0
        self.error = None

    def close(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


def installer(manager, model, identity):
    return manager.add_adapter(model), manager.activate_adapter(identity)


def rejected_installer(manager, model, identity):
    manager.add_adapter(model)
    raise RuntimeError("native activation rejected")


def drift_installer(manager, model, identity):
    result = installer(manager, model, identity)
    manager.remove_adapter = lambda identity: False
    return result


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    manager, teardown = Manager(), Teardown()
    model = SimpleNamespace(id=31)
    receiver = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=object(),
            lora_manager=SimpleNamespace(_adapter_manager=manager),
        )
    )
    sources = source_files(
        bindings.capture_receiver_manager,
        Manager,
        Teardown,
        installer,
        rejected_installer,
        drift_installer,
    )
    return receiver, manager, model, teardown, sources


def scope(inputs, **changes):
    receiver, manager, model, teardown, sources = inputs
    request = {
        "adapter_id": 31,
        "admitted_sources": sources,
        "teardown": teardown.close,
    }
    request.update(changes)
    return bindings.owned_receiver_binding(receiver, **request)


def test_receiver_binding_native_install_once_then_captured_cleanup(inputs):
    receiver, manager, model, teardown, sources = inputs
    with scope(inputs) as binding:
        assert binding.manager is manager
        record = binding.install(model, installer=installer)
        assert receiver._ap_owned_lora_installations[31][1] is model
        assert record["added"] is record["activated"] is True
        for name in (
            "native_slot_binding_admitted",
            "adapter_storage_verified",
            "adapter_application_verified",
            "capacity_admitted",
        ):
            assert record[name] is False
        assert record["numerical"] is None
        with pytest.raises(ValueError, match="once"):
            binding.install(model, installer=installer)
    assert manager.calls == ["add", "activate", "remove"]
    assert receiver._ap_owned_lora_installations == {}
    assert teardown.calls == 1


@pytest.mark.parametrize("field", ["runner", "model", "wrapper", "manager", "callback"])
def test_receiver_binding_identity_drift_rejects_then_cleans_actual_owner(inputs, field):
    receiver, manager, model, teardown, sources = inputs
    with pytest.raises(ValueError, match="identity|callbacks"):
        with scope(inputs) as binding:
            binding.install(model, installer=installer)
            if field == "runner":
                receiver.model_runner = object()
            elif field == "model":
                receiver.model_runner.model = object()
            elif field == "wrapper":
                receiver.model_runner.lora_manager = object()
            elif field == "manager":
                receiver.model_runner.lora_manager._adapter_manager = Manager()
            else:
                manager.remove_adapter = lambda identity: False
            binding.guard()
    assert manager.get_adapter(31) is None
    assert manager.calls == ["add", "activate", "remove"]
    assert teardown.calls == 1


def test_receiver_binding_owns_partial_native_installation_on_error(inputs):
    receiver, manager, model, teardown, sources = inputs
    with pytest.raises(RuntimeError, match="activation rejected"):
        with scope(inputs) as binding:
            binding.install(model, installer=rejected_installer)
    assert manager.calls == ["add", "remove"]
    assert receiver._ap_owned_lora_installations == {}
    assert teardown.calls == 1


def test_receiver_binding_install_callback_drift_uses_original_remove(inputs):
    receiver, manager, model, teardown, sources = inputs
    with pytest.raises(ValueError, match="callbacks"):
        with scope(inputs) as binding:
            binding.install(model, installer=drift_installer)
    assert manager.get_adapter(31) is None
    assert manager.calls == ["add", "activate", "remove"]
    assert teardown.calls == 1


def test_receiver_binding_preserves_primary_and_both_cleanup_failures(inputs):
    receiver, manager, model, teardown, sources = inputs
    manager.remove_error, teardown.error = RuntimeError("remove failed"), RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="original operation") as caught:
        with scope(inputs) as binding:
            binding.install(model, installer=installer)
            raise RuntimeError("original operation")
    assert len(caught.value.__notes__) == 2
    assert "remove failed" in caught.value.__notes__[0]
    assert "close failed" in caught.value.__notes__[1]
    assert manager.get_adapter(31) is model
    assert teardown.calls == 1


def test_receiver_binding_cleanup_error_does_not_skip_teardown(inputs):
    receiver, manager, model, teardown, sources = inputs
    manager.remove_error = RuntimeError("remove failed")
    with pytest.raises(RuntimeError, match="remove failed"):
        with scope(inputs) as binding:
            binding.install(model, installer=installer)
    assert teardown.calls == 1


def test_receiver_binding_never_removes_foreign_replacement(inputs):
    receiver, manager, model, teardown, sources = inputs
    foreign = SimpleNamespace(id=31)
    with pytest.raises(ValueError, match="different owner"):
        with scope(inputs) as binding:
            binding.install(model, installer=installer)
            manager.adapters[31] = foreign
    assert manager.get_adapter(31) is foreign
    assert manager.calls == ["add", "activate"]
    assert teardown.calls == 1


def test_receiver_binding_missing_source_still_calls_captured_teardown(inputs):
    with pytest.raises(ValueError, match="source identities"):
        with scope(inputs, admitted_sources={}):
            pytest.fail("source rejection must precede entry")
    assert inputs[3].calls == 1
    assert inputs[1].calls == []


def test_receiver_binding_source_drift_does_not_gate_owned_cleanup(inputs, tmp_path):
    receiver, manager, model, teardown, sources = inputs
    extra = tmp_path / "selected-source.py"
    extra.write_text("before")
    sources[str(extra)] = hashlib.sha256(b"before").hexdigest()
    with pytest.raises(RuntimeError, match="source"):
        with scope(inputs) as binding:
            binding.install(model, installer=installer)
            extra.write_text("after")
            binding.guard()
    assert manager.get_adapter(31) is None
    assert teardown.calls == 1


def test_receiver_binding_existing_adapter_rejects_without_eviction(inputs):
    receiver, manager, model, teardown, sources = inputs
    manager.adapters[31] = model
    with pytest.raises(ValueError, match="already installed"):
        with scope(inputs):
            pytest.fail("foreign installation cannot enter")
    assert manager.get_adapter(31) is model
    assert manager.calls == []
    assert teardown.calls == 1


def test_receiver_binding_model_id_mismatch_never_reserves(inputs):
    receiver, manager, model, teardown, sources = inputs
    with pytest.raises(ValueError, match="adapter id"):
        with scope(inputs) as binding:
            binding.install(SimpleNamespace(id=32), installer=installer)
    assert not hasattr(receiver, "_ap_owned_lora_installations")
    assert manager.calls == []
    assert teardown.calls == 1


# Lora receiver preparation.


def test_receiver_preparation_resolves_exact_optional_native_components(monkeypatch):
    world_group = object()
    model_class = object()
    native = SimpleNamespace(
        _build_lora_model=build,
        _install_lora_adapter=install,
        _remap_lora_key=selected_symbol,
        _served_lm_module_prefix=selected_symbol,
    )
    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(get_world_group=lambda: world_group))
    monkeypatch.setitem(sys.modules, "vllm.lora.lora_model", SimpleNamespace(LoRAModel=model_class))
    monkeypatch.setitem(sys.modules, "vllm.lora.utils", SimpleNamespace(parse_fine_tuned_lora_name=selected_symbol))
    monkeypatch.setitem(sys.modules, "arctic_platform.inference.server.weight_sync.receiver", native)

    components, get_world_group = helper._actual_components()

    assert get_world_group() is world_group
    assert components["builder"] is native._build_lora_model
    assert components["installer"] is native._install_lora_adapter
    assert components["remap_key"] is native._remap_lora_key
    assert components["served_prefix_for_model"] is native._served_lm_module_prefix
    assert components["native_model_class"] is model_class
    assert components["parse_name"] is selected_symbol
    assert components["observer"] is _lora_adapter_tensor_binding_evidence.observe_lora_build


def selected_symbol():
    raise AssertionError("preparation must not invoke builder/parser/installer")


@pytest.fixture
def _lora_receiver_preparation_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY", "1")
    monkeypatch.setenv(helper.PATH_ENV, str(tmp_path / "bootstrap.json"))
    monkeypatch.setenv(helper.SHA_ENV, "a" * 64)
    settings = {
        "schema": "job3-selected-lora-preparation-v1",
        "publication_root": "/data-fast/selected-private-operation",
        "distribution_names": ["selected-test-distribution"],
        "max_source_bytes": max(200000, Path(__file__).stat().st_size),
        "max_layers": 100,
        "max_total_source_bytes": 1000000,
    }
    path = tmp_path / helper.PREPARATION_FILENAME
    path.write_text(json.dumps(settings))
    sources = {
        **source_files(selected_symbol, helper.prepare_selected_lora_receiver),
        str(path): hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(helper, "bootstrap_descriptor", lambda pointer: {"admitted_sources": sources})
    components = {
        key: selected_symbol
        for key in (
            "observer",
            "builder",
            "installer",
            "normalize_config",
            "native_model_class",
            "parse_name",
            "remap_key",
            "served_prefix_for_model",
        )
    }
    group = SimpleNamespace(rank=3, world_size=8)

    def get_group():
        return group

    monkeypatch.setattr(helper, "_actual_components", lambda: (components, get_group))
    distribution = object()
    selected = []

    def distribution_for(name):
        selected.append(name)
        return distribution

    monkeypatch.setattr(helper.importlib.metadata, "distribution", distribution_for)
    output_calls = []

    def rank_output(root, rank):
        output_calls.append((root, rank))
        return Path(root) / f"tp-rank-{rank}"

    monkeypatch.setattr(helper, "_rank_output", rank_output)
    receiver = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=object(),
            lora_manager=SimpleNamespace(_adapter_manager=object()),
        )
    )
    return (
        receiver,
        settings,
        path,
        sources,
        group,
        distribution,
        selected,
        output_calls,
    )


def rewrite(inputs, **changes):
    _, settings, path, sources, *_ = inputs
    settings.update(changes)
    path.write_text(json.dumps(settings))
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_preparation_resolves_exact_callable_options_without_invoking_them(
    _lora_receiver_preparation_inputs,
):
    receiver, _, _, _, _, distribution, selected, output_calls = _lora_receiver_preparation_inputs
    options = helper.prepare_selected_lora_receiver(receiver)
    assert len(options) == 13
    assert options["builder"] is options["installer"] is selected_symbol
    assert options["distributions"] == (distribution,)
    assert selected == ["selected-test-distribution"]
    assert output_calls == [("/data-fast/selected-private-operation", 3)]
    assert options["output"].name == "tp-rank-3"
    assert "capacity_admitted" not in options


def test_preparation_missing_optin_rejects_before_private_output(_lora_receiver_preparation_inputs, monkeypatch):
    monkeypatch.delenv("ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY")
    with pytest.raises(ValueError, match="explicit opt-in"):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_preparation_unbound_settings_rejects_before_native_resolution(
    _lora_receiver_preparation_inputs,
):
    _lora_receiver_preparation_inputs[3].pop(str(_lora_receiver_preparation_inputs[2]))
    with pytest.raises(ValueError, match="bootstrap-bound"):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_preparation_cannot_accept_changed_settings_bytes(
    _lora_receiver_preparation_inputs,
):
    _lora_receiver_preparation_inputs[2].write_text("{}")
    with pytest.raises(ValueError):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_preparation_original_eight_rank_required(_lora_receiver_preparation_inputs):
    _lora_receiver_preparation_inputs[4].world_size = 16
    with pytest.raises(ValueError, match="SAMPLE8"):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_preparation_unbound_actual_builder_rejects(_lora_receiver_preparation_inputs):
    _lora_receiver_preparation_inputs[3].pop(str(Path(__file__).resolve()))
    with pytest.raises(ValueError, match="builder/parser/rank"):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_preparation_does_not_supply_missing_selected_manager(
    _lora_receiver_preparation_inputs,
):
    _lora_receiver_preparation_inputs[0].model_runner.lora_manager._adapter_manager = None
    with pytest.raises(ValueError, match="already-loaded"):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_preparation_boolean_bound_is_not_positive_integer(
    _lora_receiver_preparation_inputs,
):
    rewrite(_lora_receiver_preparation_inputs, max_layers=True)
    with pytest.raises(ValueError, match="positive publication bounds"):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_preparation_no_unrequested_distribution_discovery(
    _lora_receiver_preparation_inputs,
):
    rewrite(_lora_receiver_preparation_inputs, distribution_names=["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="distinct selected distributions"):
        helper.prepare_selected_lora_receiver(_lora_receiver_preparation_inputs[0])
    assert not _lora_receiver_preparation_inputs[-1]


def test_rank_publication_rejects_nonfast_root_without_creation():
    with pytest.raises(ValueError, match="private FAST"):
        helper._rank_output("/tmp", 0)


def test_rank_publication_distinct_private_directories_preserve_other_rank(tmp_path):
    tmp_path.chmod(0o700)
    first = helper._rank_output(str(tmp_path), 0)
    second = helper._rank_output(str(tmp_path), 7)
    assert first != second
    assert first.stat().st_mode & 0o7777 == 0o700
    assert second.stat().st_mode & 0o7777 == 0o700
    with pytest.raises(FileExistsError):
        helper._rank_output(str(tmp_path), 0)
    assert first.is_dir() and second.is_dir()


# Lora replica pool routing.


class ActualPool:
    pass


@pytest.fixture
def selected_route(tmp_path, monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    calls = []
    state = SimpleNamespace(change=None)
    expected = object()
    pointer_path = tmp_path / "bootstrap.json"
    pointer_path.write_text("{}")
    pointer = {
        "path": str(pointer_path),
        "sha256": hashlib.sha256(pointer_path.read_bytes()).hexdigest(),
    }

    def maker(pool, sampling, received_pointer):
        calls.append((pool, deepcopy(sampling), deepcopy(received_pointer)))
        if state.change is not None:
            state.change(sampling, received_pointer)
        return expected

    monkeypatch.setattr(selected, "make_selected_lora_replica_pool", maker)
    pins = source_files(
        routing._bind_sources,
        routing.maybe_lora_replica_pool,
        routing._pointer,
        legacy.selected_lora_request,
        maker,
    )
    request = {
        "schema": "explicit-native-lora-sync-request-v1",
        "training": {"admitted_sources": dict(pins)},
        "sampling": {
            "lora_int_id": 7,
            "lora_name": "owned",
            "lora_config": {"r": 32},
            "max_received_bytes": 4096,
            "admitted_sources": dict(pins),
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    monkeypatch.setenv(legacy.REQUEST_ENV, str(request_path))
    monkeypatch.setenv(legacy.DIGEST_ENV, hashlib.sha256(request_path.read_bytes()).hexdigest())
    monkeypatch.setenv(selected.PATH_ENV, pointer["path"])
    monkeypatch.setenv(selected.SHA_ENV, pointer["sha256"])
    monkeypatch.setattr(
        legacy,
        "maybe_lora_replica_pool",
        lambda *args: pytest.fail("selected bootstrap used frozen legacy route"),
    )
    return SimpleNamespace(**locals())


def test_selected_pool_route_passes_actual_class_exact_sampling_and_pointer(
    selected_route,
):
    case = selected_route
    assert routing.maybe_lora_replica_pool(ActualPool) is case.expected
    assert case.calls == [(ActualPool, case.request["sampling"], case.pointer)]


def test_selected_pool_route_copies_request_and_pointer_before_factory_mutation(
    selected_route,
):
    case = selected_route

    def change(sampling, pointer):
        sampling["lora_config"]["r"] = 99
        sampling["admitted_sources"].clear()
        pointer.clear()

    case.state.change = change
    assert routing.maybe_lora_replica_pool(ActualPool) is case.expected
    assert legacy.selected_lora_request() == case.request


@pytest.mark.parametrize("explicit_request", [False, True])
def test_selected_pool_route_without_pointer_preserves_exact_legacy_route(tmp_path, monkeypatch, explicit_request):
    for name in (
        legacy.REQUEST_ENV,
        legacy.DIGEST_ENV,
        selected.PATH_ENV,
        selected.SHA_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    if explicit_request:
        monkeypatch.setenv(VERIFY_ENV, "1")
        path = tmp_path / "request.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "explicit-native-lora-sync-request-v1",
                    "training": {"admitted_sources": {}},
                    "sampling": {"admitted_sources": {}},
                }
            )
        )
        monkeypatch.setenv(legacy.REQUEST_ENV, str(path))
        monkeypatch.setenv(legacy.DIGEST_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    expected, seen = object(), []

    def legacy_pool(pool):
        seen.append(pool)
        return expected

    monkeypatch.setattr(legacy, "maybe_lora_replica_pool", legacy_pool)
    monkeypatch.setattr(
        selected,
        "make_selected_lora_replica_pool",
        lambda *a: pytest.fail("unexpected selected route"),
    )
    assert routing.maybe_lora_replica_pool(ActualPool) is expected
    assert seen == [ActualPool]


@pytest.mark.parametrize(
    "damage",
    ["missing-path", "missing-sha", "bad-sha", "relative", "symlink", "no-request"],
)
def test_selected_pool_route_invalid_pointer_cannot_fall_back_or_construct(selected_route, monkeypatch, damage):
    case = selected_route
    if damage == "missing-path":
        monkeypatch.delenv(selected.PATH_ENV)
    elif damage == "missing-sha":
        monkeypatch.delenv(selected.SHA_ENV)
    elif damage == "bad-sha":
        monkeypatch.setenv(selected.SHA_ENV, "invalid")
    elif damage == "relative":
        monkeypatch.setenv(selected.PATH_ENV, "bootstrap.json")
    elif damage == "symlink":
        link = case.tmp_path / "linked.json"
        link.symlink_to(case.pointer_path)
        monkeypatch.setenv(selected.PATH_ENV, str(link))
    else:
        monkeypatch.delenv(legacy.REQUEST_ENV)
        monkeypatch.delenv(legacy.DIGEST_ENV)
    with pytest.raises(ValueError):
        routing.maybe_lora_replica_pool(ActualPool)
    assert case.calls == []


@pytest.mark.parametrize("damage", ["request", "pointer", "factory", "source"])
def test_selected_pool_route_rechecks_staged_request_pointer_factory_and_source(selected_route, monkeypatch, damage):
    case = selected_route

    def change(sampling, pointer):
        if damage == "request":
            case.request_path.write_text("changed")
        elif damage == "pointer":
            monkeypatch.setenv(selected.SHA_ENV, "0" * 64)
        elif damage == "factory":
            monkeypatch.setattr(selected, "make_selected_lora_replica_pool", lambda *args: None)
        else:
            # Add an admitted file before selection, then change only that private fixture.
            case.extra_source.write_text("# source changed\n")

    if damage == "source":
        case.extra_source = case.tmp_path / "extra.py"
        case.extra_source.write_text("# source before\n")
        pin = hashlib.sha256(case.extra_source.read_bytes()).hexdigest()
        for section in ("training", "sampling"):
            case.request[section]["admitted_sources"][str(case.extra_source)] = pin
        case.request_path.write_text(json.dumps(case.request))
        monkeypatch.setenv(
            legacy.DIGEST_ENV,
            hashlib.sha256(case.request_path.read_bytes()).hexdigest(),
        )
    case.state.change = change
    with pytest.raises(RuntimeError if damage == "source" else ValueError):
        routing.maybe_lora_replica_pool(ActualPool)
    assert len(case.calls) == 1


def test_selected_pool_route_requires_own_source_in_admitted_closure(selected_route, monkeypatch):
    case = selected_route
    for section in ("training", "sampling"):
        case.request[section]["admitted_sources"].pop(str(Path(routing.__file__)))
    case.request_path.write_text(json.dumps(case.request))
    monkeypatch.setenv(legacy.DIGEST_ENV, hashlib.sha256(case.request_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        routing.maybe_lora_replica_pool(ActualPool)
    assert case.calls == []


def test_selected_pool_route_server_hook_selects_new_helper_without_importing_server():
    path = Path(__file__).resolve().parents[2] / "common/ray_server.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_replica_pool_cls"
    )
    assert [
        node.module
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom) and any(name.name == "maybe_lora_replica_pool" for name in node.names)
    ] == ["arctic_platform.debug.lora_replica_pool_routing"]


# Lora request staging.


@pytest.fixture
def _lora_request_staging_inputs(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    monkeypatch.setenv("ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY", "1")
    config = tmp_path / "canonical.config"
    config.write_text("actual-test-config")
    directory = tmp_path / "inputs"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    cfg = SimpleNamespace(
        config_id=_lora_request_staging_helper.CONFIG_ID,
        is_rl=True,
        model_name=_lora_request_staging_helper.MODEL,
        n_gpus=8,
        native=SimpleNamespace(
            model_name=_lora_request_staging_helper.MODEL,
            training_gpus=8,
            sampling_gpus=8,
            log_prob_gpus=0,
        ),
        training={"sp_size": 8, "ep_size": 8, "peft_config": {"r": 32}},
        path=config,
    )
    prepare = Path(_lora_request_staging_helper.__file__).with_name("lora_receiver_preparation.py")
    sources = {
        **source_files(
            _lora_request_staging_helper.stage_selected_lora_request,
            _lora_request_staging_helper.bootstrap_descriptor,
        ),
        str(prepare): hashlib.sha256(prepare.read_bytes()).hexdigest(),
        str(config): hashlib.sha256(config.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def bootstrap(pointer):
        raw = Path(pointer["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pointer["sha256"]
        return json.loads(raw)

    monkeypatch.setattr(_lora_request_staging_helper, "bootstrap_descriptor", bootstrap)
    # Admit the actual fixture function origin after replacing bootstrap only in this test.
    sources.update(source_files(bootstrap))
    options = {
        "directory": directory,
        "receiver": {
            "module": "selected.fixture",
            "name": "ActualReceiver",
            "source": str(Path(__file__).resolve()),
        },
        "admitted_sources": sources,
        "adapter_id": 3,
        "adapter_name": "owned-test-adapter",
        "max_weight_bytes": 4096,
        "max_received_bytes": 8192,
        "publication_bounds": {
            "max_source_bytes": max(200000, Path(__file__).stat().st_size),
            "max_layers": 100,
            "max_total_source_bytes": 1000000,
        },
        "distribution_names": ["test-selected-distribution"],
    }
    return cfg, options


def test_staging_binds_same_sources_and_settings_without_native_claim(
    _lora_request_staging_inputs,
):
    cfg, options = _lora_request_staging_inputs
    result = _lora_request_staging_helper.stage_selected_lora_request(cfg, **options)
    request = json.loads(Path(result["request"]["path"]).read_bytes())
    bootstrap = json.loads(Path(result["bootstrap"]["path"]).read_bytes())
    assert request["training"]["admitted_sources"] == request["sampling"]["admitted_sources"]
    assert bootstrap["admitted_sources"] == result["admitted_sources"]
    assert result["admitted_sources"][result["settings"]["path"]] == result["settings"]["sha256"]
    assert result["capacity_admitted"] is result["runtime_admitted"] is False
    assert result["limits_are_capacity_proof"] is False and result["numerical"] is None
    assert result["environment"][_lora_request_staging_helper.PATH_ENV] == result["bootstrap"]["path"]
    assert result["environment"][_lora_request_staging_helper.DIGEST_ENV] == result["request"]["sha256"]
    assert str(result["settings"]["path"]) not in options["admitted_sources"]
    assert request["sampling"]["lora_config"] == cfg.training["peft_config"]
    for key in ("settings", "bootstrap", "request"):
        assert Path(result[key]["path"]).stat().st_mode & 0o7777 == 0o600


def test_staging_no_optin_leaves_fresh_directory_empty(_lora_request_staging_inputs, monkeypatch):
    monkeypatch.delenv("ARCTIC_INFERENCE_WEIGHT_SYNC_VERIFY")
    with pytest.raises(ValueError, match="opt-in"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_refuses_nonoriginal_sampling_width(_lora_request_staging_inputs):
    _lora_request_staging_inputs[0].native.sampling_gpus = 16
    with pytest.raises(ValueError, match="TRAIN8"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_refuses_unbound_actual_receiver_source(_lora_request_staging_inputs):
    _lora_request_staging_inputs[1]["receiver"]["source"] = "/not/admitted/receiver.py"
    with pytest.raises(ValueError, match="native receiver"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_refuses_changed_config_bytes_before_outputs(
    _lora_request_staging_inputs,
):
    _lora_request_staging_inputs[0].path.write_text("changed-config")
    with pytest.raises(RuntimeError, match="weight-sync source changed:"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_refuses_boolean_byte_limit(_lora_request_staging_inputs):
    _lora_request_staging_inputs[1]["max_weight_bytes"] = True
    with pytest.raises(ValueError, match="byte limits"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_refuses_smaller_receive_limit(_lora_request_staging_inputs):
    _lora_request_staging_inputs[1]["max_received_bytes"] = 2048
    with pytest.raises(ValueError, match="receive byte"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_refuses_reuse_and_preserves_existing_bytes(
    _lora_request_staging_inputs,
):
    result = _lora_request_staging_helper.stage_selected_lora_request(
        _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
    )
    before = Path(result["request"]["path"]).read_bytes()
    with pytest.raises(ValueError, match="fresh and empty"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert Path(result["request"]["path"]).read_bytes() == before


def test_staging_refuses_nonprivate_fast_directory(_lora_request_staging_inputs):
    _lora_request_staging_inputs[1]["directory"].chmod(0o755)
    with pytest.raises(ValueError, match="private FAST"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_refuses_nonstring_receiver_source_before_outputs(
    _lora_request_staging_inputs,
):
    _lora_request_staging_inputs[1]["receiver"]["source"] = []
    with pytest.raises(ValueError, match="native receiver"):
        _lora_request_staging_helper.stage_selected_lora_request(
            _lora_request_staging_inputs[0], **_lora_request_staging_inputs[1]
        )
    assert not list(_lora_request_staging_inputs[1]["directory"].iterdir())


def test_staging_actual_config_and_consumers_agree_without_native_import(_lora_request_staging_inputs, monkeypatch):
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.debug import lora_selected_sampler as selected
    from arctic_platform.debug.lora_receiver_preparation import preparation_settings
    from arctic_platform.debug.lora_sync_hooks import selected_lora_request

    _, options = _lora_request_staging_inputs
    root = Path(__file__).resolve().parents[3]
    cfg = load_config(
        root / "arctic_platform/correctness/configs/glm5.3-flash/h200/train-rl-lora-16gpus-sp8-256k.config"
    )
    monkeypatch.setattr(
        _lora_request_staging_helper,
        "bootstrap_descriptor",
        selected.bootstrap_descriptor,
    )
    options["admitted_sources"].update(
        source_files(
            selected.bootstrap_descriptor,
            selected.LoRASamplerEvidenceWorker,
            selected.engine_kwargs_for_lora,
            preparation_settings,
            selected_lora_request,
            load_config,
        )
    )
    for path in (cfg.path, selected.RPC_PATH):
        options["admitted_sources"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = _lora_request_staging_helper.stage_selected_lora_request(cfg, **options)
    for name, value in result["environment"].items():
        monkeypatch.setenv(name, value)
    bootstrap = selected.bootstrap_descriptor(result["bootstrap"])
    settings, sources = preparation_settings()
    request = selected_lora_request()
    assert sources == bootstrap["admitted_sources"] == result["admitted_sources"]
    assert request["sampling"]["admitted_sources"] == sources
    assert request["training"]["admitted_sources"] == sources
    assert request["sampling"]["lora_config"] == cfg.training["peft_config"]
    assert cfg.native.training_gpus == cfg.native.sampling_gpus == 8
    assert settings["distribution_names"] == options["distribution_names"]
    assert not result["native_engine_constructed"] and result["numerical"] is None


# Lora sampler evidence.


class _lora_sampler_evidence_Manager:
    def __init__(self):
        self.adapters = {}
        self.active = None

    def add_adapter(self, model):
        self.adapters[model.id] = model
        return True

    def activate_adapter(self, identity):
        self.active = identity
        return True

    def get_adapter(self, identity):
        return self.adapters.get(identity)

    def remove_adapter(self, identity):
        if self.active == identity:
            self.active = None
        return self.adapters.pop(identity, None) is not None


class Engine:
    def __init__(self, rows):
        self.rows = rows

    def receive_weights(self):
        yield from self.rows


def build(*, lora_int_id, tensors, lora_config, device, dtype, model):
    return SimpleNamespace(id=lora_int_id, tensors=tensors, is_3d_lora_weight=False)


def install(manager, model, identity):
    return manager.add_adapter(model), manager.activate_adapter(identity)


def normalize(config, *, location):
    return dict(config)


@pytest.fixture
def native_fixture(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    monkeypatch.setitem(
        sys.modules,
        "arctic_platform.inference.server.weight_sync.receiver",
        SimpleNamespace(
            _build_lora_model=build,
            _install_lora_adapter=install,
            normalize_lora_peft_config=normalize,
        ),
    )
    manager = _lora_sampler_evidence_Manager()
    receiver = SimpleNamespace(
        model_runner=SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=manager), model=object()),
        model_config=SimpleNamespace(quantization=None, dtype=torch.float32),
    )
    pins = source_files(
        bridge.receive_and_install_lora,
        build,
        install,
        normalize,
        _lora_sampler_evidence_Manager,
        Engine,
    )
    options = {
        "lora_int_id": 1,
        "lora_name": "policy",
        "lora_config": {"r": 32},
        "max_received_bytes": 32,
        "admitted_sources": pins,
    }
    return receiver, manager, options


def test_lora_receiver_native_build_install_and_honest_nonadmissions(native_fixture):
    receiver, manager, options = native_fixture
    value = torch.tensor([1.0, 2.0])
    record = bridge.receive_and_install_lora(receiver, Engine([("layer.lora_A.weight", value)]), **options)
    assert manager.active == 1
    assert torch.equal(manager.get_adapter(1).tensors["layer.lora_A.weight"], value)
    assert record["received"] == {"layer.lora_A.weight": tensor_manifest(value)}
    assert record["received_bytes"] == 8
    assert record["status"] == "adapter_installed"
    assert record["adapter_storage_verified"] is False
    assert record["adapter_application_verified"] is False
    assert record["full_parameter_coverage"] is False
    assert record["capacity_admitted"] is False and record["numerical"] is None
    assert "weight_sync_verification" not in record and "trainer_parameters" not in record


def test_lora_receiver_disabled_evidence_never_resolves_optional_backend(native_fixture, monkeypatch):
    receiver, _, options = native_fixture
    monkeypatch.delenv(VERIFY_ENV)

    def unexpected_import(name):
        raise AssertionError(f"disabled receiver imported {name}")

    monkeypatch.setattr(bridge.importlib, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="explicitly enabled"):
        bridge.receive_and_install_lora(receiver, Engine([]), **options)


def test_lora_receiver_clones_before_reusable_bucket_advances(native_fixture):
    class ReusingEngine:
        def receive_weights(self):
            value = torch.tensor([1.0, 2.0])
            yield "layer.lora_A.weight", value
            value.fill_(7)
            yield "layer.lora_B.weight", value
            value.fill_(9)

    receiver, manager, options = native_fixture
    options["admitted_sources"].update(source_files(ReusingEngine))
    result = bridge.receive_and_install_lora(receiver, ReusingEngine(), **options)
    actual = manager.get_adapter(1).tensors
    assert actual["layer.lora_A.weight"].tolist() == [1.0, 2.0]
    assert actual["layer.lora_B.weight"].tolist() == [7.0, 7.0]
    assert result["received"]["layer.lora_A.weight"] == tensor_manifest(actual["layer.lora_A.weight"])


@pytest.mark.parametrize("cause", ["duplicate", "base", "budget", "existing", "empty"])
def test_lora_receiver_invalid_transfer_rejected_before_install(native_fixture, cause):
    receiver, manager, options = native_fixture
    value = torch.ones(2)
    rows = [("layer.lora_A.weight", value)]
    if cause == "duplicate":
        rows += rows
    elif cause == "base":
        rows = [("base.weight", value)]
    elif cause == "budget":
        options["max_received_bytes"] = 4
    elif cause == "existing":
        manager.adapters[1] = object()
    else:
        rows = []
    with pytest.raises(ValueError):
        bridge.receive_and_install_lora(receiver, Engine(rows), **options)
    assert manager.active is None


def test_lora_receiver_requires_actual_callable_source_before_receive(native_fixture):
    receiver, manager, options = native_fixture
    options["admitted_sources"] = {}
    with pytest.raises(ValueError, match="actual adapter callable"):
        bridge.receive_and_install_lora(receiver, Engine([("layer.lora_A.weight", torch.ones(2))]), **options)
    assert manager.adapters == {}


def test_lora_receiver_changed_operation_source_rejected_before_install(native_fixture, tmp_path):
    receiver, manager, options = native_fixture
    source = tmp_path / "changed.py"
    source.write_text("changed source\n")
    options["admitted_sources"][str(source)] = "0" * 64
    with pytest.raises(RuntimeError, match="source changed"):
        bridge.receive_and_install_lora(receiver, Engine([("layer.lora_A.weight", torch.ones(2))]), **options)
    assert manager.adapters == {}


def test_lora_serving_selects_actual_installed_id_without_disk_reload(native_fixture, monkeypatch):
    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=Request))
    receiver, _, options = native_fixture
    installation = bridge.receive_and_install_lora(
        receiver, Engine([("layer.lora_A.weight", torch.ones(2))]), **options
    )
    pins = {**options["admitted_sources"], **source_files(Request)}
    request = bridge.serving_lora_request(installation, admitted_sources=pins)
    assert request.lora_int_id == 1 and request.lora_name == "policy"
    assert request.lora_path == "arctic-nccl://policy" and request.load_inplace is False
    assert installation["adapter_application_verified"] is False


@pytest.mark.parametrize("field", ["lora_added", "lora_activated"])
def test_lora_serving_rejects_incomplete_installation(native_fixture, field):
    receiver, _, options = native_fixture
    installation = bridge.receive_and_install_lora(
        receiver, Engine([("layer.lora_A.weight", torch.ones(2))]), **options
    )
    installation[field] = False
    with pytest.raises(ValueError, match="genuine installation"):
        bridge.serving_lora_request(installation, admitted_sources=options["admitted_sources"])


def test_lora_engine_binding_preserves_recipe_fields(native_fixture):
    original = {
        "enable_lora": True,
        "max_lora_rank": 32,
        "tensor_parallel_size": 8,
        "attention_config": {"backend": "FLASHINFER_MLA_SPARSE_SM90"},
    }
    result = bridge.engine_kwargs_for_lora(original, rank=32)
    assert {key: result[key] for key in original} == original
    assert result["worker_extension_cls"] == bridge.EXTENSION
    assert "worker_extension_cls" not in original


@pytest.mark.parametrize(
    "change",
    [
        {"enable_lora": False},
        {"max_lora_rank": 16},
        {"worker_extension_cls": "other.Extension"},
    ],
)
def test_lora_engine_binding_rejects_unadmitted_adapter_engine(native_fixture, change):
    kwargs = {"enable_lora": True, "max_lora_rank": 32, **change}
    with pytest.raises(ValueError):
        bridge.engine_kwargs_for_lora(kwargs, rank=32)


def test_lora_explicit_extension_keeps_external_receiver_and_engine(native_fixture, monkeypatch):
    class External:
        def _get_or_create_engine(self, *args, **kwargs):
            self.engine_arguments = (args, kwargs)
            return object()

    monkeypatch.setitem(
        sys.modules,
        "arctic_inference.server.weight_sync.receiver",
        SimpleNamespace(WeightSyncExtension=External),
    )
    bridge.receiver_extension.cache_clear()
    try:
        cls = bridge.receiver_extension()
        assert issubclass(cls, External)
        receiver = cls()
        pins = source_files(cls._get_or_create_engine, bridge.receive_and_install_lora)
        result = receiver.sync_lora_weights(
            "127.0.0.1",
            7777,
            1,
            9,
            lora_int_id=1,
            lora_name="policy",
            lora_config={"r": 32},
            max_received_bytes=32,
            admitted_sources=pins,
            engine_only=True,
        )
        assert receiver.engine_arguments == (
            ("127.0.0.1", 7777, 256 * 1024 * 1024, False),
            {"engine_attr": "_ws_engine", "key_attr": "_ws_engine_key"},
        )
        assert result["status"] == "engine_ready"
        assert result["rank"] == 1 and result["world_size"] == 2
        assert result["adapter_storage_verified"] is False and result["capacity_admitted"] is False
    finally:
        bridge.receiver_extension.cache_clear()


def test_lora_cleanup_after_activation_failure_uses_owned_manager(native_fixture):
    receiver, manager, options = native_fixture

    def reject_activation(identity):
        raise RuntimeError("actual activation failed")

    manager.activate_adapter = reject_activation
    options["admitted_sources"].update(source_files(reject_activation))
    with pytest.raises(RuntimeError, match="activation failed"):
        bridge.receive_and_install_lora(receiver, Engine([("layer.lora_A.weight", torch.ones(2))]), **options)
    assert manager.adapters == {} and receiver._ap_owned_lora_installations == {}


def test_lora_cleanup_retains_captured_remover_without_new_source_guard(native_fixture, monkeypatch):
    receiver, manager, options = native_fixture
    bridge.receive_and_install_lora(receiver, Engine([("layer.lora_A.weight", torch.ones(2))]), **options)

    def forbidden(*args, **kwargs):
        raise AssertionError("cleanup must use the captured route")

    manager.remove_adapter = forbidden
    monkeypatch.setattr(bridge, "verify_sources", forbidden)
    assert bridge.cleanup_owned_loras(receiver) == {"owned_adapters_empty": True}
    assert manager.adapters == {} and manager.active is None


def test_lora_cleanup_nested_external_engine_teardown_despite_foreign_adapter(native_fixture, monkeypatch):
    class External:
        def close_weight_sync(self):
            self.engine_closed = True

    monkeypatch.setitem(
        sys.modules,
        "arctic_inference.server.weight_sync.receiver",
        SimpleNamespace(WeightSyncExtension=External),
    )
    bridge.receiver_extension.cache_clear()
    try:
        receiver = bridge.receiver_extension()()
        manager = _lora_sampler_evidence_Manager()
        owned = SimpleNamespace(id=1)
        foreign = SimpleNamespace(id=1)
        manager.adapters[1] = foreign
        receiver._ap_owned_lora_installations = {
            1: (manager, owned, manager.get_adapter, manager.remove_adapter),
        }
        with pytest.raises(RuntimeError, match="different owner's adapter"):
            receiver.close_weight_sync()
        assert receiver.engine_closed is True and manager.get_adapter(1) is foreign
    finally:
        bridge.receiver_extension.cache_clear()


def test_lora_cleanup_preserves_adapter_error_when_engine_teardown_also_fails(native_fixture, monkeypatch):
    class External:
        def close_weight_sync(self):
            raise RuntimeError("external engine also failed")

    monkeypatch.setitem(
        sys.modules,
        "arctic_inference.server.weight_sync.receiver",
        SimpleNamespace(WeightSyncExtension=External),
    )
    bridge.receiver_extension.cache_clear()
    try:
        receiver = bridge.receiver_extension()()
        manager = _lora_sampler_evidence_Manager()
        manager.adapters[1] = object()
        receiver._ap_owned_lora_installations = {
            1: (manager, object(), manager.get_adapter, manager.remove_adapter),
        }
        with pytest.raises(RuntimeError, match="different owner's adapter") as caught:
            receiver.close_weight_sync()
        assert "external engine also failed" in caught.value.__notes__[0]
    finally:
        bridge.receiver_extension.cache_clear()


# Lora sampler worker.


class EngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tensor_parallel_size = kwargs["tensor_parallel_size"]

    def create_engine_config(self):
        return self.kwargs


class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class LoRARequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LLM:
    constructed = []

    def __init__(self):
        self.calls = []
        self.ranks = []
        self.fail_generate = False
        self.fail_close = False
        self.fail_shutdown = False

    @classmethod
    def from_vllm_config(cls, config, *, stat_loggers):
        instance = cls()
        instance.config = config
        cls.constructed.append(instance)
        return instance

    async def collective_rpc(self, method, *, args=(), kwargs=None):
        self.calls.append((method, args, kwargs))
        if method == "close_weight_sync" and self.fail_close:
            raise RuntimeError("fixture engine close failed")
        return self.ranks

    async def generate(self, prompt, params, *, request_id, lora_request):
        self.calls.append(("generate", prompt, lora_request))
        if self.fail_generate:
            raise RuntimeError("fixture native generation failed")
        yield SimpleNamespace(
            prompt_token_ids=[1, 2],
            prompt_logprobs=None,
            num_cached_tokens=1,
            outputs=[
                SimpleNamespace(
                    text="sample",
                    token_ids=[3],
                    finish_reason="stop",
                    logprobs=[{3: SimpleNamespace(logprob=-0.1, rank=1)}],
                )
            ],
        )

    async def abort(self, request_id):
        self.calls.append(("abort", request_id))

    def shutdown(self):
        self.calls.append(("shutdown",))
        if self.fail_shutdown:
            raise RuntimeError("fixture native shutdown failed")


def external_enabled():
    return False


@pytest.fixture
def native_modules(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    monkeypatch.setitem(
        sys.modules,
        "arctic_inference.envs",
        SimpleNamespace(arctic_inference_effective_enabled=external_enabled),
    )
    monkeypatch.setitem(
        sys.modules,
        "arctic_inference.server.metrics",
        SimpleNamespace(RingStatLogger=object),
    )
    plugins = SimpleNamespace(load_general_plugins=lambda: None)
    vllm = SimpleNamespace(plugins=plugins, SamplingParams=SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.plugins", plugins)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", SimpleNamespace(AsyncLLM=LLM))
    monkeypatch.setitem(
        sys.modules,
        "vllm.engine.arg_utils",
        SimpleNamespace(AsyncEngineArgs=EngineArgs),
    )
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=LoRARequest))
    LLM.constructed.clear()
    pins = source_files(
        _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker,
        SamplerEvidenceWorker,
        receiver.receive_and_install_lora,
        EngineArgs,
        SamplingParams,
        LLM,
        LoRARequest,
    )
    return {
        "lora_int_id": 7,
        "lora_name": "owned-policy",
        "lora_config": {"r": 32},
        "max_received_bytes": 4096,
        "admitted_sources": pins,
    }


def installation(request):
    return {
        "schema": "explicit-external-lora-installation-v1",
        "status": "adapter_installed",
        "weight_format": "lora",
        "lora_int_id": request["lora_int_id"],
        "lora_name": request["lora_name"],
        "lora_added": True,
        "lora_activated": True,
        "is_3d_lora_weight": False,
        "received": {
            "experts.w1.lora_A.weight": {
                "shape": [8, 32, 3],
                "numel": 768,
                "dtype": "bfloat16",
                "sha256": "a" * 64,
            }
        },
        "received_bytes": 1536,
        "evidence_sources": request["admitted_sources"],
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
        "numerical": None,
    }


def worker(request):
    selected = _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker(request)
    selected.llm = LLM()
    selected._origins = deepcopy(request["admitted_sources"])
    selected._tp = 8
    selected.state = "ready"
    selected.llm.ranks = [deepcopy(installation(request)) for _ in range(8)]
    return selected


def sync(selected, **kwargs):
    return asyncio.run(selected.sync_weights("same-allocation-head", 30000, 1, 2, **kwargs))


def test_lora_worker_native_engine_has_explicit_extension_and_unchanged_recipe(
    native_modules,
):
    selected = _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker(native_modules)
    options = {
        "enable_lora": True,
        "max_lora_rank": 32,
        "tensor_parallel_size": 8,
        "model": "exact-original",
        "dtype": "bfloat16",
    }
    asyncio.run(selected.initialize(options))
    assert selected.llm.config == {
        **options,
        "worker_extension_cls": receiver.EXTENSION,
    }
    assert options.get("worker_extension_cls") is None
    assert selected.state == "ready" and selected._tp == 8
    assert selected._origins == native_modules["admitted_sources"]


@pytest.mark.parametrize("tp", [1, 4])
def test_lora_worker_requires_original_sample8_before_constructor(native_modules, tp):
    selected = _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker(native_modules)
    with pytest.raises(ValueError, match="SAMPLE8"):
        asyncio.run(selected.initialize({"enable_lora": True, "max_lora_rank": 32, "tensor_parallel_size": tp}))
    assert LLM.constructed == []


def test_lora_worker_public_pool_factory_keeps_external_class_and_owned_request(native_modules, monkeypatch):
    class Pool:
        def __init__(self, *, worker_cls):
            self.worker_cls = worker_cls

    class Ray:
        @staticmethod
        def remote(**kwargs):
            assert kwargs == {"runtime_env": {"env_vars": {VERIFY_ENV: "1"}}}
            return lambda cls: cls

    monkeypatch.setitem(sys.modules, "ray", Ray)
    native_modules["admitted_sources"].update(source_files(Pool))
    factory = _lora_sampler_worker_bridge.make_lora_replica_pool(Pool, native_modules)
    native_modules["lora_name"] = "changed-caller"
    pool = factory()
    selected = pool.worker_cls()
    assert isinstance(pool, Pool) and isinstance(selected, _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker)
    assert selected._request["lora_name"] == "owned-policy"


def test_lora_worker_preserves_all_actual_tp_installations_without_fullweight_proof(
    native_modules,
):
    selected = worker(native_modules)
    result = sync(selected)
    assert result["rank_results"] is selected.llm.ranks
    assert result["all_rank_adapter_installation"] is selected.llm.ranks
    assert "all_rank_weight_sync_verification" not in result
    assert result["full_parameter_coverage"] is False and result["numerical"] is None
    method, args, kwargs = selected.llm.calls[0]
    assert method == "sync_lora_weights" and args == (
        "same-allocation-head",
        30000,
        1,
        2,
    )
    assert kwargs["lora_config"] == {"r": 32} and kwargs["max_received_bytes"] == 4096
    assert selected._lora_request.lora_int_id == 7 and selected._lora_request.load_inplace is False


@pytest.mark.parametrize("cause", ["rank", "identity", "bytes", "false-proof", "source"])
def test_lora_worker_incomplete_or_conflicting_rank_evidence_cannot_select_adapter(native_modules, cause):
    selected = worker(native_modules)
    if cause == "rank":
        selected.llm.ranks.pop()
    elif cause == "identity":
        selected.llm.ranks[-1]["lora_int_id"] = 9
    elif cause == "bytes":
        selected.llm.ranks[-1]["received"]["experts.w1.lora_A.weight"]["sha256"] = "b" * 64
    elif cause == "false-proof":
        selected.llm.ranks[-1]["full_parameter_coverage"] = True
    else:
        selected.llm.ranks[-1]["evidence_sources"] = {"unadmitted": "c" * 64}
    with pytest.raises(ValueError):
        sync(selected)
    assert selected._lora_request is None and selected._installation is None


def test_lora_worker_engine_initialization_is_not_adapter_installation(native_modules):
    selected = worker(native_modules)
    actual = deepcopy(selected.llm.ranks)
    selected.llm.ranks = [{"status": "engine_ready", "weight_format": "lora"} for _ in range(8)]
    assert sync(selected, engine_only=True)["status"] == "engine_ready"
    assert selected._lora_request is None and selected._sync_attempted is False
    selected.llm.ranks = actual
    assert sync(selected)["status"] == "done"
    with pytest.raises(ValueError, match="single-use"):
        sync(selected)
    assert len(selected.llm.calls) == 2


@pytest.mark.parametrize("options", [{"direct_mode": True}, {"reverse": True}])
def test_lora_worker_rejects_nonselected_transport_before_rpc(native_modules, options):
    selected = worker(native_modules)
    with pytest.raises(ValueError, match="pairwise"):
        sync(selected, **options)
    assert selected.llm.calls == []


def test_lora_worker_source_drift_prevents_actual_transfer(native_modules, tmp_path):
    selected = worker(native_modules)
    changed = tmp_path / "changed.py"
    changed.write_text("new bytes")
    selected._origins[str(changed)] = "0" * 64
    with pytest.raises(RuntimeError, match="source changed"):
        sync(selected)
    assert selected.llm.calls == []


def test_lora_worker_native_generate_receives_exact_installed_request(native_modules):
    selected = worker(native_modules)
    sync(selected)
    request = selected._lora_request
    result = asyncio.run(selected.generate([1, 2], {"temperature": 0.0}))
    assert selected.llm.calls[-1] == ("generate", {"prompt_token_ids": [1, 2]}, request)
    assert result["token_ids"] == [3] and result["prefix_cache_len"] == 1
    assert result["logprobs"] == [{3: {"logprob": -0.1, "rank": 1}}]


def test_lora_worker_no_adapter_means_no_base_model_generation(native_modules):
    selected = worker(native_modules)
    with pytest.raises(ValueError, match="installed adapter"):
        asyncio.run(selected.generate("prompt", {}))
    assert selected.llm.calls == []


def test_lora_worker_does_not_fake_native_router_protocol(native_modules):
    selected = worker(native_modules)
    with pytest.raises(ValueError, match="router"):
        asyncio.run(selected.generate("prompt", {"dss_router_replay": True}))
    assert selected.llm.calls == []


def test_lora_worker_generation_error_aborts_actual_request(native_modules):
    selected = worker(native_modules)
    sync(selected)
    selected.llm.fail_generate = True
    with pytest.raises(RuntimeError, match="native generation"):
        asyncio.run(selected.generate("prompt", {}))
    assert selected.llm.calls[-1][0] == "abort"


def test_lora_worker_close_does_not_gate_owned_cleanup_on_source_drift(native_modules):
    selected = worker(native_modules)
    sync(selected)
    selected._origins = {__file__: "0" * 64}
    selected.llm.fail_close = True
    with pytest.raises(RuntimeError, match="engine close"):
        asyncio.run(selected.close_weight_sync())
    assert selected.llm.calls[-1][0] == "close_weight_sync"
    assert selected._installation is None and selected._lora_request is None
    selected.llm.fail_shutdown = True
    with pytest.raises(RuntimeError, match="shutdown"):
        selected.shutdown()
    assert selected.llm.calls[-1][0] == "shutdown" and selected._lora_request is None


def test_lora_worker_preserves_native_generation_error_when_abort_also_fails(native_modules, monkeypatch):
    selected = worker(native_modules)
    sync(selected)
    selected.llm.fail_generate = True

    async def fail_abort(request_id):
        selected.llm.calls.append(("abort", request_id))
        raise RuntimeError("fixture abort also failed")

    monkeypatch.setattr(selected.llm, "abort", fail_abort)
    with pytest.raises(RuntimeError, match="native generation") as caught:
        asyncio.run(selected.generate("prompt", {}))
    assert any("abort also failed" in note for note in caught.value.__notes__)
    assert selected.llm.calls[-1][0] == "abort"


def test_lora_worker_preserves_post_constructor_source_error_when_shutdown_fails(native_modules, monkeypatch):
    selected = _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker(native_modules)
    actual_verify = _lora_sampler_worker_bridge.verify_sources

    def drift_after_constructor(pins):
        if LLM.constructed:
            LLM.constructed[-1].fail_shutdown = True
            raise RuntimeError("fixture source drift after constructor")
        actual_verify(pins)

    monkeypatch.setattr(_lora_sampler_worker_bridge, "verify_sources", drift_after_constructor)
    with pytest.raises(RuntimeError, match="source drift after constructor") as caught:
        asyncio.run(selected.initialize({"enable_lora": True, "max_lora_rank": 32, "tensor_parallel_size": 8}))
    assert LLM.constructed[-1].calls[-1][0] == "shutdown"
    assert any("shutdown also failed" in note for note in caught.value.__notes__)


# Lora sync hooks.


class Server:
    weight_sync_ready = False


class Pool:
    num_replicas = 1
    tp_size = 8

    async def sync_weights(self):
        return {}


class Schedule:
    def __init__(self):
        self.groups = [SimpleNamespace(sender_train_rank=0, world_size=2, replica_ids=[0])]


@pytest.fixture
def staged_request(tmp_path, monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    pins = source_files(hooks._bind_sources, hooks.maybe_prepare_lora_training, Server, Pool, Schedule)
    request = {
        "schema": "explicit-native-lora-sync-request-v1",
        "training": {"max_weight_bytes": 4096, "admitted_sources": pins},
        "sampling": {
            "lora_int_id": 7,
            "lora_name": "owned",
            "lora_config": {"r": 32},
            "max_received_bytes": 4096,
            "admitted_sources": pins,
        },
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setenv(hooks.REQUEST_ENV, str(path))
    monkeypatch.setenv(hooks.DIGEST_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    return request, path


def test_lora_hooks_pinned_fast_request_rejects_later_byte_change(staged_request):
    request, path = staged_request
    assert hooks.selected_lora_request() == request
    path.write_text(json.dumps(request) + " ")
    with pytest.raises(ValueError, match="changed"):
        hooks.selected_lora_request()


def test_lora_hooks_no_request_preserves_external_debug_pool_selection(monkeypatch):
    monkeypatch.delenv(hooks.REQUEST_ENV, raising=False)
    monkeypatch.delenv(hooks.DIGEST_ENV, raising=False)
    expected, pool = object(), object()
    monkeypatch.setattr(
        rl_sampler_evidence,
        "maybe_replica_pool",
        lambda actual: expected if actual is pool else None,
    )
    assert hooks.maybe_lora_replica_pool(pool) is expected


def test_lora_hooks_digest_without_path_is_not_silent_fallback(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    monkeypatch.delenv(hooks.REQUEST_ENV, raising=False)
    monkeypatch.setenv(hooks.DIGEST_ENV, "a" * 64)
    with pytest.raises(ValueError, match="path and SHA256"):
        hooks.selected_lora_request()


def test_lora_hooks_explicit_request_cannot_bypass_optin(staged_request, monkeypatch):
    monkeypatch.delenv(VERIFY_ENV)
    with pytest.raises(ValueError, match="opt-in"):
        hooks.selected_lora_request()


def test_lora_hooks_actor_request_must_match_actual_staged_bytes(staged_request, monkeypatch):
    request, _ = staged_request
    seen = []
    monkeypatch.setattr(hooks, "prepare_lora_worker", lambda worker, value: seen.append((worker, value)))
    worker = object()
    hooks.maybe_prepare_lora_worker(worker, request["training"])
    assert seen == [(worker, request["training"])]
    with pytest.raises(ValueError, match="staged actor"):
        hooks.maybe_prepare_lora_worker(worker, {**request["training"], "max_weight_bytes": 8192})
    assert len(seen) == 1


def test_lora_hooks_selected_send_does_not_use_base_get_weights(staged_request, monkeypatch):
    request, _ = staged_request

    def forbid_base():
        pytest.fail("base get_weights is not the selected adapter exporter")

    worker = SimpleNamespace(
        get_weights=forbid_base,
        _ap_owned_lora_preparation={"request": request["training"]},
        _weight_sender=SimpleNamespace(
            group_id=0,
            targets=[(0, rank) for rank in range(8)],
            ports=list(range(30000, 30008)),
        ),
    )
    worker._ap_owned_lora_preparation["sender_binding"] = {
        "sender": worker._weight_sender,
        "targets": list(worker._weight_sender.targets),
        "ports": list(worker._weight_sender.ports),
    }
    expected = {"weight_format": "lora"}
    monkeypatch.setattr(hooks, "send_lora_worker", lambda actual: expected if actual is worker else None)
    assert hooks.maybe_send_lora_worker(worker) is expected


def test_lora_hooks_cleanup_ignores_request_drift_and_does_not_destroy_sender(
    staged_request,
):
    _, path = staged_request
    slot = {"owned": True}
    worker = SimpleNamespace(_ap_owned_lora_preparation=slot)
    path.write_text("changed")
    result = hooks.maybe_release_lora_worker(worker)
    assert result["owned_preparation_released"] is True and result["native_sender_destroyed"] is False


def test_lora_hooks_all8_barrier_occurs_before_returning_selected_send_readiness(staged_request, monkeypatch):
    request, _ = staged_request
    server, pool, schedule, workers = (
        Server(),
        Pool(),
        Schedule(),
        [object() for _ in range(8)],
    )
    events = []

    async def prepare(actual_workers, actual_request):
        assert actual_workers is workers and actual_request == request["training"]
        events.append("prepare-all8")
        await asyncio.sleep(0)
        events.append("prepare-all8-complete")
        return {"all_training_ranks_prepared": True}

    monkeypatch.setattr(hooks, "prepare_lora_training_workers", prepare)
    result = asyncio.run(hooks.maybe_prepare_lora_training(server, workers, pool, schedule))
    assert events == ["prepare-all8", "prepare-all8-complete"]
    assert server._ap_lora_training_preparation is result
    with pytest.raises(ValueError, match="single-use"):
        asyncio.run(hooks.maybe_prepare_lora_training(server, workers, pool, schedule))


def test_lora_hooks_failed_barrier_cannot_be_retried(staged_request, monkeypatch):
    async def fail(*args):
        raise RuntimeError("actual collective failed")

    monkeypatch.setattr(hooks, "prepare_lora_training_workers", fail)
    server, pool, schedule = Server(), Pool(), Schedule()
    with pytest.raises(RuntimeError, match="collective failed"):
        asyncio.run(hooks.maybe_prepare_lora_training(server, [], pool, schedule))
    with pytest.raises(ValueError, match="single-use"):
        asyncio.run(hooks.maybe_prepare_lora_training(server, [], pool, schedule))


@pytest.mark.parametrize("cause", ["sender", "world", "replicas", "tp"])
def test_lora_hooks_nonoriginal_schedule_never_enters_collective(staged_request, monkeypatch, cause):
    def forbid_prepare(*args):
        pytest.fail("invalid native topology entered collective preparation")

    monkeypatch.setattr(hooks, "prepare_lora_training_workers", forbid_prepare)
    pool, schedule = Pool(), Schedule()
    if cause == "sender":
        schedule.groups[0].sender_train_rank = 1
    elif cause == "world":
        schedule.groups[0].world_size = 9
    elif cause == "replicas":
        pool.num_replicas = 2
    else:
        pool.tp_size = 4
    with pytest.raises(ValueError, match="pairwise schedule"):
        asyncio.run(hooks.maybe_prepare_lora_training(Server(), [], pool, schedule))


def test_lora_hooks_bucket_geometry_comes_from_actual_adapter_not_base_manifest(
    staged_request,
):
    server = Server()
    server._ap_lora_training_preparation = {
        "all_training_ranks_prepared": True,
        "rank_preparations": [
            {
                "emitted_manifest": {
                    "native.adapter.lora_A.weight": {"dtype": "bfloat16", "numel": 512},
                    "native.adapter.lora_B.weight": {"dtype": "bfloat16", "numel": 768},
                }
            }
        ],
    }
    assert asyncio.run(hooks.maybe_lora_max_param_bytes(server, object())) == 1536


def test_lora_hooks_installed_adapter_receipts_keep_judge_and_capacity_nonadmitting(
    staged_request,
):
    request, _ = staged_request
    server = Server()
    server._ap_lora_training_preparation = {"all_training_ranks_prepared": True}
    worker = {
        "weight_format": "lora",
        "all_rank_adapter_installation": [{"rank": rank} for rank in range(8)],
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
    }

    async def receive():
        return {"workers": [worker]}

    async def send():
        return {"weight_format": "lora"}

    result = asyncio.run(hooks.maybe_collect_lora_sync(server, receive(), [send()]))
    assert result["targets"][0]["recv"]["workers"] == [worker]
    assert result["training_preparation"] is server._ap_lora_training_preparation
    assert result["parameter_only_judge_admitted"] is result["capacity_admitted"] is False
    assert result["numerical"] is None


def test_lora_hooks_transfer_drains_peer_failure_and_retains_both_causes(
    staged_request,
):
    async def receive():
        await asyncio.sleep(0)
        raise RuntimeError("receiver failed")

    async def send():
        raise RuntimeError("sender failed")

    with pytest.raises(RuntimeError, match="sender or receiver") as caught:
        asyncio.run(hooks.maybe_collect_lora_sync(Server(), receive(), [send()]))
    assert any("receiver failed" in note for note in caught.value.__notes__)
    assert any("sender failed" in note for note in caught.value.__notes__)


def test_lora_hooks_production_calls_keep_prepare_before_init_and_sender_dispatch():
    root = Path(__file__).resolve().parents[2]
    source = (root / "common/ray_server.py").read_text()
    tree = ast.parse(source)
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_sync_weights_nccl"
    )
    statements = ast.unparse(method)
    assert statements.index("maybe_prepare_lora_training") < statements.index("init_weight_sender.remote")
    assert statements.index("init_weight_sender.remote") < statements.index("send_weights.remote")
    assert "maybe_lora_max_param_bytes" in statements
    worker = ast.parse((root / "common/deepspeed_worker.py").read_text())
    prepare = next(
        node for node in ast.walk(worker) if isinstance(node, ast.FunctionDef) and node.name == "prepare_lora_weights"
    )
    assert "maybe_prepare_lora_worker(self, request)" in ast.unparse(prepare)


class TransferGroup:
    def __init__(self, *, group_id, sender_train_rank, targets):
        self.group_id, self.sender_train_rank, self.targets = (
            group_id,
            sender_train_rank,
            targets,
        )
        self.world_size = 2


class NativeSchedule:
    @classmethod
    def build(cls, **kwargs):
        selected = cls()
        selected.__dict__.update(kwargs)
        selected.groups = [
            TransferGroup(group_id=rank, sender_train_rank=rank, targets=[(0, rank)]) for rank in range(8)
        ]
        return selected

    def port_for_target(self, replica_id, tp_rank, base_port):
        return base_port + replica_id * self.inference_tp + tp_rank


def test_lora_hooks_projects_only_sender_policy_preserving_native8_geometry_targets_and_ports(
    staged_request, monkeypatch
):
    request, path = staged_request
    request["training"]["admitted_sources"].update(source_files(NativeSchedule, TransferGroup))
    request["sampling"]["admitted_sources"] = request["training"]["admitted_sources"]
    path.write_text(json.dumps(request))
    monkeypatch.setenv(hooks.DIGEST_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    schedule = hooks.maybe_lora_transfer_schedule(
        NativeSchedule,
        training_sharding="dp",
        training_gpus=8,
        inference_replicas=1,
        inference_tp=8,
    )
    assert schedule.training_gpus == schedule.inference_tp == 8 and schedule.inference_replicas == 1
    assert len(schedule.groups) == 1 and schedule.groups[0].sender_train_rank == 0
    assert schedule.groups[0].targets == [(0, rank) for rank in range(8)]
    assert [schedule.port_for_target(0, rank, 30000) for rank in range(8)] == list(range(30000, 30008))


def test_lora_hooks_default_schedule_remains_original_when_unselected(monkeypatch):
    monkeypatch.delenv(hooks.REQUEST_ENV, raising=False)
    monkeypatch.delenv(hooks.DIGEST_ENV, raising=False)
    schedule = hooks.maybe_lora_transfer_schedule(
        NativeSchedule,
        training_sharding="dp",
        training_gpus=8,
        inference_replicas=1,
        inference_tp=8,
    )
    assert len(schedule.groups) == 8 and [group.sender_train_rank for group in schedule.groups] == list(range(8))


def test_lora_hooks_schedule_projection_cannot_relabel_training8_as1(staged_request):
    with pytest.raises(ValueError, match="TRAIN8"):
        hooks.maybe_lora_transfer_schedule(
            NativeSchedule,
            training_sharding="dp",
            training_gpus=1,
            inference_replicas=1,
            inference_tp=8,
        )


class NativeSender:
    def __init__(self, group, schedule, base_port):
        self.group_id = group.group_id
        self.targets = group.targets
        self.ports = [schedule.port_for_target(replica, tp, base_port) for replica, tp in group.targets]


def test_lora_hooks_captures_actual_initialized_sender_and_rejects_port_drift(staged_request, monkeypatch):
    request, path = staged_request
    request["training"]["admitted_sources"].update(source_files(NativeSender, NativeSchedule, TransferGroup))
    request["sampling"]["admitted_sources"] = request["training"]["admitted_sources"]
    path.write_text(json.dumps(request))
    monkeypatch.setenv(hooks.DIGEST_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    schedule = hooks.maybe_lora_transfer_schedule(
        NativeSchedule,
        training_sharding="dp",
        training_gpus=8,
        inference_replicas=1,
        inference_tp=8,
    )
    worker = SimpleNamespace(rank=0, _ap_owned_lora_preparation={"request": request["training"]})
    worker._weight_sender = NativeSender(schedule.groups[0], schedule, 30000)
    binding = hooks.maybe_record_lora_sender(worker, schedule.groups[0], schedule, 30000)
    assert binding["training_gpus"] == binding["inference_tp"] == 8
    assert binding["ports"] == list(range(30000, 30008))
    monkeypatch.setattr(hooks, "send_lora_worker", lambda actual: {"weight_format": "lora"})
    assert hooks.maybe_send_lora_worker(worker)["weight_format"] == "lora"
    worker._weight_sender.ports[-1] += 1
    with pytest.raises(ValueError, match="targets or ports"):
        hooks.maybe_send_lora_worker(worker)


def test_lora_hooks_native_sender_missing_initialization_binding_never_dispatches(
    staged_request,
):
    request, _ = staged_request
    worker = SimpleNamespace(
        _ap_owned_lora_preparation={"request": request["training"]},
        _weight_sender=object(),
    )
    with pytest.raises(ValueError, match="captured initialization"):
        hooks.maybe_send_lora_worker(worker)


def test_lora_hooks_unselected_send_preserves_existing_debug_sender_callback(
    monkeypatch,
):
    from arctic_platform.debug import weight_sync_evidence

    monkeypatch.delenv(hooks.REQUEST_ENV, raising=False)
    monkeypatch.delenv(hooks.DIGEST_ENV, raising=False)
    weights, expected, seen = [("original", object())], object(), []
    worker = SimpleNamespace(get_weights=lambda: weights)

    def original_send(actual, values):
        seen.append((actual, values))
        return expected

    monkeypatch.setattr(weight_sync_evidence, "maybe_send_weights", original_send)
    assert hooks.maybe_send_lora_worker(worker) is expected
    assert seen == [(worker, weights)]


def test_lora_hooks_unselected_bucket_preserves_actual_native_max_parameter_rpc(
    monkeypatch,
):
    monkeypatch.delenv(hooks.REQUEST_ENV, raising=False)
    monkeypatch.delenv(hooks.DIGEST_ENV, raising=False)
    seen = []

    async def native_max():
        seen.append("original-rpc")
        return 16384

    worker = SimpleNamespace(max_param_bytes=SimpleNamespace(remote=native_max))
    assert asyncio.run(hooks.maybe_lora_max_param_bytes(object(), worker)) == 16384
    assert seen == ["original-rpc"]


# Lora training protocol.


class Model:
    def named_parameters(self):
        return iter(())

    def named_buffers(self):
        return iter(())


class Sender:
    def __init__(self):
        self.calls = []
        self.fail = False

    def send(self, weights):
        self.calls.append(weights)
        if self.fail:
            raise RuntimeError("fixture native send failed")
        return {"status": "done", "weight_format": "lora"}


class Prepared:
    def __init__(self, model, request, rank=0):
        self.model = model
        self.rank = rank
        self.sources = request["admitted_sources"]
        self.weights = [("native.adapter.lora_A.weight", object())] if rank == 0 else []
        self.verified = 0

    def receipt(self):
        self.verified += 1
        return receipt(self.rank, self.sources)


def receipt(rank, sources):
    return {
        "rank": rank,
        "weight_format": "lora",
        "native_iterator_completed": True,
        "unchanged_native_iterator": True,
        "native_shard_bytes_checked": rank == 0,
        "rank_records": [{"rank": index, "world_size": 8} for index in range(8)],
        "global_inventory": {
            "parameters": {"base": {}, "adapter": {}},
            "buffers": {"router_bias": {}},
        },
        "adapter_parameter_names": ["adapter"],
        "non_wire_parameter_names": ["base"],
        "non_wire_buffers": {"router_bias": {}},
        "source_files": sources,
        "receiver_installation_admitted": False,
        "serving_request_admitted": False,
        "router_application_admitted": False,
        "runtime_admitted": False,
        "capacity_admitted": False,
        "parameter_only_judge_admitted": False,
        "recipe_admitted": False,
    }


@pytest.fixture
def native_functions(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    calls = []

    def prepare(model, **kwargs):
        calls.append((model, kwargs))
        return Prepared(model, kwargs)

    def send(sender, prepared):
        calls.append(("send", sender, prepared))
        return sender.send(prepared.weights)

    monkeypatch.setattr(protocol, "prepare_collective_lora_weights", prepare)
    monkeypatch.setattr(protocol, "send_prepared_lora_weights", send)
    monkeypatch.setattr(protocol, "PreparedLoRAWeights", Prepared)
    request = {
        "max_weight_bytes": 4096,
        "admitted_sources": source_files(
            protocol._bind_sources,
            protocol.prepare_lora_worker,
            Model,
            Sender,
            prepare,
            send,
        ),
    }
    worker = SimpleNamespace(
        rank=0,
        world_size=8,
        engine=SimpleNamespace(module=Model()),
        _weight_sender=Sender(),
    )
    return request, worker, calls


def test_lora_training_prepares_same_native_module_and_keeps_private_stream_actor_local(
    native_functions,
):
    request, worker, calls = native_functions
    result = protocol.prepare_lora_worker(worker, request)
    model, args = calls[0]
    assert model is worker.engine.module
    assert args["weight_format"] == "lora" and args["is_zero3"] is False
    assert args["expected_world_size"] == args["expected_ep_size"] == 8
    assert args["max_weight_bytes"] == 4096
    assert worker._ap_owned_lora_preparation["prepared"].model is model
    assert "weights" not in result and result["parameter_only_judge_admitted"] is False


def test_lora_training_uses_genuine_current_sender_once_without_get_weights(
    native_functions,
):
    request, worker, calls = native_functions
    protocol.prepare_lora_worker(worker, request)
    slot = worker._ap_owned_lora_preparation
    result = protocol.send_lora_worker(worker)
    assert result["weight_format"] == "lora"
    assert calls[-1] == ("send", worker._weight_sender, slot["prepared"])
    assert worker._weight_sender.calls == [slot["prepared"].weights]
    with pytest.raises(ValueError, match="single-use"):
        protocol.send_lora_worker(worker)


def test_lora_training_failed_send_is_not_retryable(native_functions):
    request, worker, _ = native_functions
    protocol.prepare_lora_worker(worker, request)
    worker._weight_sender.fail = True
    with pytest.raises(RuntimeError, match="native send"):
        protocol.send_lora_worker(worker)
    with pytest.raises(ValueError, match="single-use"):
        protocol.send_lora_worker(worker)
    assert len(worker._weight_sender.calls) == 1


@pytest.mark.parametrize("cause", ["engine", "model", "sender-missing", "rank"])
def test_lora_training_drift_or_missing_native_sender_prevents_dispatch(native_functions, cause):
    request, worker, calls = native_functions
    protocol.prepare_lora_worker(worker, request)
    if cause == "engine":
        worker.engine = SimpleNamespace(module=worker.engine.module)
    elif cause == "model":
        worker.engine.module = Model()
    elif cause == "sender-missing":
        worker._weight_sender = None
    else:
        worker.rank = 1
    with pytest.raises(ValueError):
        protocol.send_lora_worker(worker)
    assert len(calls) == 1


def test_lora_training_preparation_attempt_is_single_use_even_after_owned_release(
    native_functions,
):
    request, worker, calls = native_functions
    protocol.prepare_lora_worker(worker, request)
    slot = worker._ap_owned_lora_preparation
    result = protocol.release_lora_worker(worker, slot)
    assert result["owned_preparation_released"] is True
    assert result["native_sender_destroyed"] is result["native_process_group_destroyed"] is False
    with pytest.raises(ValueError, match="single-use"):
        protocol.prepare_lora_worker(worker, request)
    assert len(calls) == 1


def test_lora_training_cleanup_cannot_delete_a_different_captured_slot(
    native_functions,
):
    request, worker, _ = native_functions
    protocol.prepare_lora_worker(worker, request)
    captured = worker._ap_owned_lora_preparation
    replacement = dict(captured)
    worker._ap_owned_lora_preparation = replacement
    with pytest.raises(ValueError, match="another owner"):
        protocol.release_lora_worker(worker, captured)
    assert worker._ap_owned_lora_preparation is replacement


def test_lora_training_owned_release_does_not_recheck_drifted_sources(native_functions):
    request, worker, _ = native_functions
    protocol.prepare_lora_worker(worker, request)
    slot = worker._ap_owned_lora_preparation
    slot["prepared"].sources = {__file__: "0" * 64}
    assert protocol.release_lora_worker(worker, slot)["owned_preparation_released"] is True


@pytest.mark.parametrize("limit", [0, True])
def test_lora_training_invalid_budget_never_enters_native_collective(native_functions, limit):
    request, worker, calls = native_functions
    request["max_weight_bytes"] = limit
    with pytest.raises(ValueError, match="byte limit"):
        protocol.prepare_lora_worker(worker, request)
    assert calls == [] and not hasattr(worker, "_ap_owned_lora_preparation")


class RemotePreparation:
    def __init__(self, rank, events, request):
        self.rank, self.events, self.request = rank, events, request
        self.change = None
        self.fail = False

    def remote(self, request):
        self.events.append(("dispatched", self.rank))
        assert request == self.request and request is not self.request

        async def complete():
            await asyncio.sleep(0)
            self.events.append(("completed", self.rank))
            if self.fail:
                raise RuntimeError("fixture native collective failed")
            row = receipt(self.rank, deepcopy(request["admitted_sources"]))
            if self.change:
                self.change(row)
            return row

        return complete()


def actors(request):
    events = []
    workers = [SimpleNamespace(prepare_lora_weights=RemotePreparation(rank, events, request)) for rank in range(8)]
    return workers, events


def test_lora_training_barrier_dispatches_all8_before_wait_and_preserves_honest_counts(
    native_functions,
):
    request, _, _ = native_functions
    workers, events = actors(request)
    result = asyncio.run(protocol.prepare_lora_training_workers(workers, request))
    assert events[:8] == [("dispatched", rank) for rank in range(8)]
    assert events[8:] == [("completed", rank) for rank in range(8)]
    assert result["all_training_ranks_prepared"] is True and result["native_send_started"] is False
    assert result["rank_preparations"][0]["non_wire_parameter_names"] == ["base"]
    assert result["rank_preparations"][0]["non_wire_buffers"] == {"router_bias": {}}
    assert result["capacity_admitted"] is result["parameter_only_judge_admitted"] is False
    assert result["numerical"] is None


def test_lora_training_barrier_drains_other_rank_results_before_reporting_error(
    native_functions,
):
    request, _, _ = native_functions
    workers, events = actors(request)
    workers[2].prepare_lora_weights.fail = True
    with pytest.raises(RuntimeError, match="preparations failed") as caught:
        asyncio.run(protocol.prepare_lora_training_workers(workers, request))
    assert len(events) == 16
    assert any("native collective failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("cause", ["rank", "sources", "inventory", "records", "admission", "completion"])
def test_lora_training_barrier_rejects_nonmatching_rank_or_fabricated_admission(native_functions, cause):
    request, _, _ = native_functions
    workers, _ = actors(request)

    def change(row):
        if cause == "rank":
            row["rank"] = 0
        elif cause == "sources":
            row["source_files"] = {"unadmitted": "f" * 64}
        elif cause == "inventory":
            row["non_wire_parameter_names"] = []
        elif cause == "records":
            row["rank_records"].pop()
        elif cause == "admission":
            row["parameter_only_judge_admitted"] = True
        else:
            row["native_iterator_completed"] = False

    workers[-1].prepare_lora_weights.change = change
    with pytest.raises(ValueError):
        asyncio.run(protocol.prepare_lora_training_workers(workers, request))


def test_lora_training_barrier_rejects_duplicate_handles_before_any_dispatch(
    native_functions,
):
    request, _, _ = native_functions
    workers, events = actors(request)
    workers[-1] = workers[0]
    with pytest.raises(ValueError, match="distinct"):
        asyncio.run(protocol.prepare_lora_training_workers(workers, request))
    assert events == []


def test_lora_training_omitted_verifier_source_prevents_native_preparation(
    native_functions,
):
    request, worker, calls = native_functions
    for path in source_files(protocol._bind_sources):
        del request["admitted_sources"][path]
    with pytest.raises(ValueError, match="source identity"):
        protocol.prepare_lora_worker(worker, request)
    assert calls == [] and not getattr(worker, "_ap_lora_prepare_attempted", False)


# Lora weight sync evidence.


class SyntheticParamWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter_name = "w1"
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(torch.arange(6).reshape(2, 3))
            self.lora_B["default"].weight.copy_(torch.arange(8).reshape(4, 2))
        for parameter in self.parameters():
            parameter.allreduce = False
            parameter.group_name = "ep8"


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.peft_config = {"default": SimpleNamespace(r=2)}
        self.base = torch.nn.Parameter(torch.ones((3, 3)), requires_grad=False)
        self.register_buffer("expert_bias", torch.ones(2))
        self.experts = SyntheticParamWrapper()
        self.proj = torch.nn.Module()
        self.proj.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        self.proj.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 3, bias=False)})


class SyntheticSender:
    def __init__(self):
        self.calls = []
        self.change_bytes = False

    def send(self, weights):
        self.calls.append(weights)
        if self.change_bytes:
            weights[0][1].add_(1)
        return {"status": "done", "actually_sent": len(weights)}


class SyntheticCollective:
    """Simulate peer bytes while leaving native iterator code unchanged."""

    def __init__(self, model):
        self.model = model
        self.rank = 0
        self.calls = []
        self.reverse_shards = False
        self.change_replica = False
        self.change_base = False
        self.omit_rank = False

    def is_initialized(self):
        return True

    def get_rank(self, group=None):
        return self.rank

    def get_world_size(self, group=None):
        return 8

    def get_process_group_ranks(self, group):
        assert group == "ep8"
        return list(range(8))

    def expert_group(self, group_name):
        assert group_name == "ep8"
        return group_name

    def all_gather(self, shards, local, group):
        assert group == "ep8"
        self.calls.append("gather")
        for rank, shard in enumerate(shards):
            offset = (7 - rank if self.reverse_shards else rank) - self.rank
            shard.copy_(local + offset)
        if self.change_base:
            self.model.base.add_(1)

    def all_gather_object(self, records, record):
        self.calls.append("inventory")
        parameters = dict(self.model.named_parameters())
        normalized = {}
        for (
            prefix,
            weight_name,
            kind,
            parameter,
        ) in native._iter_param_wrapper_expert_loras(self.model):
            name = native._expert_lora_vllm_key(prefix, weight_name, kind)
            normalized[name] = native._reshape_expert_lora_for_ep(parameter.data.contiguous(), kind, 2, label=name)
        for wire_name, entry in record["wire_inputs"].items():
            normalized.setdefault(wire_name, parameters[entry["parameter"]].data)
        for rank in range(8):
            current = deepcopy(record)
            current["rank"] = rank
            for name, parameter in parameters.items():
                if current["parameters"][name]["ep_ranks"] is not None:
                    current["parameters"][name].update(tensor_manifest(parameter + rank - self.rank))
            for name, entry in current["wire_inputs"].items():
                local = normalized[name]
                if entry["ep_ranks"] is not None:
                    local = local + rank - self.rank
                elif self.change_replica and rank == 1:
                    local = local + 1
                entry["manifest"] = tensor_manifest(local)
            records[rank] = current
        if self.omit_rank:
            records[7]["rank"] = 6


@pytest.fixture
def synthetic_native(monkeypatch):
    model = SyntheticModel()
    collective = SyntheticCollective(model)
    for name in (
        "is_initialized",
        "get_rank",
        "get_world_size",
        "get_process_group_ranks",
        "all_gather",
        "all_gather_object",
    ):
        monkeypatch.setattr(dist, name, getattr(collective, name))
    peft = ModuleType("peft")
    tuners = ModuleType("peft.tuners")
    lora = ModuleType("peft.tuners.lora")
    layer = ModuleType("peft.tuners.lora.layer")
    layer.ParamWrapper = SyntheticParamWrapper
    peft.tuners, tuners.lora, lora.layer = tuners, lora, layer
    for module in (peft, tuners, lora, layer):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    deepspeed = ModuleType("deepspeed")
    utils = ModuleType("deepspeed.utils")
    groups = ModuleType("deepspeed.utils.groups")
    groups._get_expert_parallel_group = collective.expert_group
    deepspeed.utils, utils.groups = utils, groups
    for module in (deepspeed, utils, groups):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    pins = source_files(
        _lora_weight_sync_evidence_evidence.prepare_collective_lora_weights,
        global_inventory,
        SyntheticModel,
        SyntheticParamWrapper,
        SyntheticSender.send,
        model.named_parameters,
        model.named_buffers,
        model.named_modules,
        model.modules,
        native.iter_lora_weights,
        native._iter_param_wrapper_expert_loras,
        native._reshape_expert_lora_for_ep,
        native._expert_lora_vllm_key,
        native._peft_lora_parameter_to_vllm_name,
        native._peft_adapter_rank,
        native._is_ep_sharded_parameter,
        native._expert_parallel_group,
        native.validate_lora_sync_trainable_parameters,
        native.is_peft_model,
        is_peft_lora_param,
        collective.is_initialized,
        collective.get_rank,
        collective.get_world_size,
        collective.all_gather,
        collective.all_gather_object,
        collective.get_process_group_ranks,
        collective.expert_group,
    )
    return model, collective, pins


def _prepare(synthetic_native, **updates):
    model, _, pins = synthetic_native
    request = {
        "weight_format": "lora",
        "is_zero3": False,
        "max_weight_bytes": 4096,
        "admitted_sources": pins,
    }
    request.update(updates)
    return _lora_weight_sync_evidence_evidence.prepare_collective_lora_weights(model, **request)


def test_lora_native_iterator_retains_expert_and_dense_bytes_without_fullweight_relabel(
    synthetic_native,
):
    _, collective, _ = synthetic_native
    prepared = _prepare(synthetic_native)
    assert collective.calls == ["inventory", "gather", "gather"]
    assert list(prepared.manifest) == [
        "experts.w1.lora_A.weight",
        "experts.w1.lora_B.weight",
        "proj.lora_A.weight",
        "proj.lora_B.weight",
    ]
    receipt = prepared.receipt()
    assert receipt["unchanged_native_iterator"] is True
    assert receipt["non_wire_parameter_names"] == ["base"]
    assert set(receipt["non_wire_buffers"]) == {"expert_bias"}
    assert receipt["receiver_installation_admitted"] is False
    assert receipt["serving_request_admitted"] is False
    assert receipt["parameter_only_judge_admitted"] is False
    sender = SyntheticSender()
    sent = _lora_weight_sync_evidence_evidence.send_prepared_lora_weights(sender, prepared)
    assert sender.calls == [prepared.weights]
    assert sent["sent_manifest"] == prepared.manifest
    assert sent["trainer_parameters"]["numel"] == sent["trainer_adapter_parameters"]["numel"] + 9
    assert sent["trainer_non_wire_parameters"] == {"numel": 9, "count": 1}
    assert sent["trainer_buffers"] == {"numel": 2, "count": 1, "wire_numel": 0}


def test_lora_native_expert_b_permutation_and_gather_order_are_retained(
    synthetic_native,
):
    model, _, _ = synthetic_native
    prepared = _prepare(synthetic_native)
    output = dict(prepared.weights)["experts.w1.lora_B.weight"]
    local = model.experts.lora_B["default"].weight.detach().reshape(4, 2, 1).permute(2, 0, 1)
    assert output.shape == (8, 4, 2)
    for rank in range(8):
        assert torch.equal(output[rank : rank + 1], local + rank)


def test_lora_native_nonmaster_participates_but_emits_nothing(synthetic_native):
    _, collective, _ = synthetic_native
    collective.rank = 1
    prepared = _prepare(synthetic_native)
    assert prepared.rank == 1
    assert prepared.weights == []
    assert prepared.manifest == {}
    assert prepared.receipt()["native_shard_bytes_checked"] is False
    assert collective.calls == ["inventory", "gather", "gather"]
    with pytest.raises(ValueError, match="rank-zero"):
        _lora_weight_sync_evidence_evidence.send_prepared_lora_weights(SyntheticSender(), prepared)


@pytest.mark.parametrize(
    "updates",
    [
        {"weight_format": "vllm"},
        {"is_zero3": True},
        {"expected_world_size": 16},
        {"expected_ep_size": 4},
    ],
)
def test_lora_native_rejects_changed_format_zero_or_topology_before_collectives(synthetic_native, updates):
    _, collective, _ = synthetic_native
    with pytest.raises(ValueError):
        _prepare(synthetic_native, **updates)
    assert collective.calls == []


def test_lora_native_source_mismatch_rejects_before_collectives(synthetic_native):
    _, collective, pins = synthetic_native
    bad = dict(pins)
    path = next(iter(source_files(native.iter_lora_weights).keys() - source_files().keys()))
    bad[path] = "0" * 64
    with pytest.raises((ValueError, RuntimeError), match="source"):
        _prepare(synthetic_native, admitted_sources=bad)
    assert collective.calls == []


def test_lora_native_byte_budget_rejects_before_native_tensor_gathers(synthetic_native):
    _, collective, _ = synthetic_native
    with pytest.raises(ValueError, match="byte limit"):
        _prepare(synthetic_native, max_weight_bytes=1)
    assert collective.calls == ["inventory"]


def test_lora_native_unsupported_trainable_base_not_silently_ignored(synthetic_native):
    model, collective, _ = synthetic_native
    model.base.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="Unsupported trainable"):
        _prepare(synthetic_native)
    assert collective.calls == []


def test_lora_native_multiple_expert_adapter_buckets_rejected(synthetic_native):
    model, collective, _ = synthetic_native
    model.experts.lora_A["other"] = torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(ValueError, match="bucket"):
        _prepare(synthetic_native)
    assert collective.calls == []


def test_lora_native_dense_wire_name_collision_rejected(synthetic_native):
    model, collective, _ = synthetic_native
    model.proj.lora_A["other"] = torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(ValueError, match="ambiguous"):
        _prepare(synthetic_native)
    assert collective.calls == []


def test_lora_native_replica_byte_disagreement_rejected(synthetic_native):
    _, collective, _ = synthetic_native
    collective.change_replica = True
    with pytest.raises(ValueError, match="replicated"):
        _prepare(synthetic_native)
    assert collective.calls == ["inventory"]


def test_lora_native_missing_rank_rejected(synthetic_native):
    _, collective, _ = synthetic_native
    collective.omit_rank = True
    with pytest.raises(ValueError, match="rank identities"):
        _prepare(synthetic_native)
    assert collective.calls == ["inventory"]


def test_lora_native_reordered_gathered_bytes_rejected(synthetic_native):
    _, collective, _ = synthetic_native
    collective.reverse_shards = True
    with pytest.raises(ValueError, match="ordered shard"):
        _prepare(synthetic_native)


def test_lora_native_nonwire_base_drift_rejected(synthetic_native):
    _, collective, _ = synthetic_native
    collective.change_base = True
    with pytest.raises(RuntimeError, match="changed during export"):
        _prepare(synthetic_native)


@pytest.mark.parametrize("defect", ["tensor", "metadata", "order"])
def test_lora_native_retained_stream_drift_rejected_before_sender(synthetic_native, defect):
    prepared = _prepare(synthetic_native)
    if defect == "tensor":
        prepared.weights[0][1].add_(1)
    elif defect == "metadata":
        prepared.inventory["parameters"]["base"]["global_numel"] += 1
    else:
        prepared.weights.reverse()
    sender = SyntheticSender()
    with pytest.raises(RuntimeError, match="changed"):
        _lora_weight_sync_evidence_evidence.send_prepared_lora_weights(sender, prepared)
    assert sender.calls == []


def test_lora_native_actual_sender_byte_mutation_is_retained_as_failure(
    synthetic_native,
):
    prepared = _prepare(synthetic_native)
    sender = SyntheticSender()
    sender.change_bytes = True
    with pytest.raises(RuntimeError, match="changed"):
        _lora_weight_sync_evidence_evidence.send_prepared_lora_weights(sender, prepared)
    assert len(sender.calls) == 1


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


# Source-bound initialization hooks; fixtures are not native/capacity evidence.

_INITIALIZATION_EVENTS = []


class _InitializationRuntime:
    def __init__(self, release_bound, *, wrapper_sources):
        self.release_bound = release_bound
        self.wrapper_sources = wrapper_sources

    def guard(self):
        _INITIALIZATION_EVENTS.append(("guard", self))

    def verify_engine(self, engine, *, sp_info, sp_group):
        _INITIALIZATION_EVENTS.append(("engine", engine, sp_info, sp_group))
        return {"synthetic_engine_verified": True}


def _initialization_capacity(request, *, phase, subject, runtime):
    _INITIALIZATION_EVENTS.append(("capacity", phase, subject, runtime))
    if request.get("fail_phase") == phase:
        raise RuntimeError("synthetic capacity supplier rejected " + phase)
    if request.get("empty"):
        return {"phase": phase, "request": request, "evidence": {}}
    if request.get("mutate_recipe"):
        subject["ds_config"]["optimizer"]["lr"] = 0.0
    if request.get("mutate_args"):
        subject["engine_args"].tensor_parallel_size = 1
    return {"phase": phase, "request": request, "evidence": {"synthetic_only": True}}


class _InitializationWorker:
    def __init__(self):
        self.engine = SimpleNamespace(global_steps=0)
        self.sp_group = object()
        self.fail_cleanup = False

    def initialize(self):
        pass

    def destroy(self):
        _INITIALIZATION_EVENTS.append(("destroy", self.engine))
        self.engine = None
        if self.fail_cleanup:
            raise RuntimeError("synthetic cleanup failed")

    def get_sequence_parallel_info(self):
        return {"rank": 0, "world_size": 8, "sp_size": 8, "sp_rank": 0, "sp_ranks": list(range(8))}


def _initialization_build_model(spec):
    return spec


def _initialization_initialize(**kwargs):
    return kwargs


@pytest.fixture
def initialization_hooks(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(sys.modules[__name__], "_INITIALIZATION_EVENTS", events)
    monkeypatch.setenv(VERIFY_ENV, "1")
    release = tmp_path / "synthetic-source-release.json"
    release.write_text('{"synthetic_only":true}')
    sources = source_files(
        hooks,
        selected,
        receiver,
        _lora_sampler_worker_bridge,
        SamplerEvidenceWorker,
        _InitializationRuntime,
    )
    sources[str(release)] = hashlib.sha256(release.read_bytes()).hexdigest()

    def symbol(name):
        return {"module": __name__, "name": name, "source": str(Path(__file__).resolve())}

    request = {
        "schema": "explicit-native-initialization-request-v1",
        "admitted_sources": sources,
        "runtime": symbol("_InitializationRuntime"),
        "runtime_options": {
            "release_bound": {"path": str(release), "sha256": sources[str(release)]},
            "wrapper_sources": {},
        },
        "verify_capacity": symbol("_initialization_capacity"),
        "capacity_request": {"synthetic_only": True},
    }
    path = tmp_path / "initialization-request.json"

    def publish():
        path.write_text(json.dumps(request, sort_keys=True))
        monkeypatch.setenv(hooks.INITIALIZATION_ENV, str(path))
        monkeypatch.setenv(hooks.INITIALIZATION_SHA_ENV, hashlib.sha256(path.read_bytes()).hexdigest())

    publish()
    return SimpleNamespace(
        events=events,
        request=request,
        path=path,
        release=release,
        publish=publish,
        worker=_InitializationWorker(),
        spec=object(),
        groups=object(),
        config={"optimizer": {"lr": 0.001}},
    )


def _arm_initialization(control):
    return hooks.maybe_admit_worker_construction(
        control.worker,
        control.spec,
        control.config,
        control.groups,
        _initialization_build_model,
        _initialization_initialize,
    )


def test_initialization_hooks_disabled_path_does_not_read_request_or_change_worker(monkeypatch):
    monkeypatch.delenv(VERIFY_ENV, raising=False)
    monkeypatch.delenv(hooks.INITIALIZATION_ENV, raising=False)
    monkeypatch.delenv(hooks.INITIALIZATION_SHA_ENV, raising=False)
    worker = _InitializationWorker()
    engine = worker.engine
    assert hooks.maybe_admit_worker_construction(worker, None, None, None, None, None) is None
    assert hooks.maybe_verify_worker_engine(worker) is None
    assert hooks.maybe_admit_sampler_construction(worker, None, None, None) is None
    assert hooks.maybe_verify_sampler_construction(worker) is None
    assert worker.engine is engine and not hasattr(worker, "_ap_initialization_admission")


def test_initialization_hooks_weight_sync_optin_alone_preserves_unselected_initialization(monkeypatch):
    monkeypatch.setenv(VERIFY_ENV, "1")
    monkeypatch.delenv(hooks.INITIALIZATION_ENV, raising=False)
    monkeypatch.delenv(hooks.INITIALIZATION_SHA_ENV, raising=False)
    worker = _InitializationWorker()
    engine = worker.engine
    assert hooks.maybe_admit_worker_construction(worker, None, None, None, None, None) is None
    assert hooks.maybe_verify_worker_engine(worker) is None
    assert hooks.maybe_admit_sampler_construction(worker, None, None, None) is None
    assert hooks.maybe_verify_sampler_construction(worker) is None
    assert worker.engine is engine and not hasattr(worker, "_ap_initialization_admission")


@pytest.mark.parametrize("key", [hooks.INITIALIZATION_ENV, hooks.INITIALIZATION_SHA_ENV])
def test_initialization_hooks_partial_pair_without_evidence_optin_never_falls_back(monkeypatch, key):
    monkeypatch.delenv(VERIFY_ENV, raising=False)
    monkeypatch.delenv(hooks.INITIALIZATION_ENV, raising=False)
    monkeypatch.delenv(hooks.INITIALIZATION_SHA_ENV, raising=False)
    monkeypatch.setenv(key, "/data-fast/not-a-request" if key == hooks.INITIALIZATION_ENV else "a" * 64)
    with pytest.raises(ValueError, match="requires explicit evidence opt-in"):
        hooks.maybe_admit_worker_construction(_InitializationWorker(), None, None, None, None, None)


@pytest.mark.parametrize("missing", ["pointer", "digest", "source", "runtime", "capacity", "plan"])
def test_initialization_hooks_missing_required_inputs_fail_before_constructor(
    initialization_hooks, monkeypatch, missing
):
    control = initialization_hooks
    if missing in {"pointer", "digest"}:
        key = hooks.INITIALIZATION_ENV if missing == "pointer" else hooks.INITIALIZATION_SHA_ENV
        monkeypatch.delenv(key)
    else:
        key = {"source": "admitted_sources", "runtime": "runtime", "capacity": "verify_capacity"}.get(missing)
        if key:
            control.request.pop(key)
        else:
            control.request["capacity_request"] = {}
        control.publish()
    with pytest.raises(ValueError):
        _arm_initialization(control)
    assert control.events == [] and not hasattr(control.worker, "_ap_initialization_admission")


def test_initialization_hooks_worker_binds_actual_recipe_groups_and_engine_before_ready(initialization_hooks):
    control = initialization_hooks
    result = _arm_initialization(control)
    assert result["capacity_admitted"] is False and result["numerical"] is None
    engine = control.worker.engine
    result = hooks.maybe_verify_worker_engine(control.worker)
    calls = [event for event in control.events if event[0] == "capacity"]
    assert [event[1] for event in calls] == ["before-build-model", "after-engine"]
    assert calls[0][2]["worker"] is control.worker and calls[0][2]["model_spec"] is control.spec
    assert calls[0][2]["parallel_groups"] is control.groups and calls[0][2]["ds_config"] is control.config
    assert calls[1][2]["engine"] is engine
    assert [event for event in control.events if event[0] == "engine"][0][1] is engine
    assert control.worker._ap_initialization_admission["status"] == "verified"
    assert control.config == {"optimizer": {"lr": 0.001}}
    assert result["capacity_admitted"] is False and result["numerical"] is None
    with pytest.raises(ValueError, match="single-use"):
        _arm_initialization(control)


@pytest.mark.parametrize("cause", ["provider", "empty", "recipe"])
def test_initialization_hooks_supplier_failure_cannot_arm_constructor(initialization_hooks, cause):
    control = initialization_hooks
    control.request["capacity_request"].update(
        {"fail_phase": "before-build-model"}
        if cause == "provider"
        else {"empty": True} if cause == "empty" else {"mutate_recipe": True}
    )
    control.publish()
    with pytest.raises(RuntimeError if cause == "provider" else ValueError):
        _arm_initialization(control)
    assert not hasattr(control.worker, "_ap_initialization_admission")


@pytest.mark.parametrize("cause", ["request", "source", "disabled"])
def test_initialization_hooks_postengine_drift_restores_only_owned_worker(initialization_hooks, monkeypatch, cause):
    control = initialization_hooks
    _arm_initialization(control)
    engine = control.worker.engine
    if cause == "request":
        control.path.write_text("{}")
    elif cause == "source":
        control.release.write_text("changed")
    else:
        monkeypatch.delenv(VERIFY_ENV)
        monkeypatch.delenv(hooks.INITIALIZATION_ENV)
        monkeypatch.delenv(hooks.INITIALIZATION_SHA_ENV)
    with pytest.raises(ValueError):
        hooks.maybe_verify_worker_engine(control.worker)
    assert ("destroy", engine) in control.events and control.worker.engine is None
    assert control.worker._ap_initialization_admission["status"] == "failed"


def test_initialization_hooks_afterengine_provider_error_survives_cleanup_failure(initialization_hooks):
    control = initialization_hooks
    control.request["capacity_request"]["fail_phase"] = "after-engine"
    control.publish()
    _arm_initialization(control)
    control.worker.fail_cleanup = True
    with pytest.raises(RuntimeError, match="supplier rejected") as caught:
        hooks.maybe_verify_worker_engine(control.worker)
    assert control.worker.engine is None
    assert any("initialization cleanup also failed" in note for note in caught.value.__notes__)


def test_initialization_hooks_wrong_worker_cannot_claim_or_destroy_another_owner(initialization_hooks):
    control = initialization_hooks
    _arm_initialization(control)
    other = _InitializationWorker()
    other._ap_initialization_admission = control.worker._ap_initialization_admission
    with pytest.raises(ValueError, match="actual armed worker"):
        hooks.maybe_verify_worker_engine(other)
    assert not [event for event in control.events if event[0] == "destroy"]
    assert control.worker.engine is not None and other.engine is not None


def test_initialization_hooks_no_prebuild_slot_cannot_validate_selected_engine(initialization_hooks):
    with pytest.raises(ValueError, match="no armed pre-build"):
        hooks.maybe_verify_worker_engine(initialization_hooks.worker)


def test_initialization_hooks_provider_code_replacement_is_not_admitted_source(initialization_hooks, monkeypatch):
    def replacement(request, *, phase, subject, runtime):
        return {"phase": phase, "request": request, "evidence": {"fabricated": True}}

    monkeypatch.setattr(_initialization_capacity, "__code__", replacement.__code__)
    with pytest.raises(ValueError, match="differs from admitted source"):
        _arm_initialization(initialization_hooks)
    assert initialization_hooks.events == []


def test_initialization_hooks_sampler_positive_native_boundary_retains_recipe(initialization_hooks, native_modules):
    control = initialization_hooks
    worker = _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker(native_modules)
    options = {"enable_lora": True, "max_lora_rank": 32, "tensor_parallel_size": 8, "model": "original"}
    asyncio.run(worker.initialize(options))
    assert worker.state == "ready" and len(LLM.constructed) == 1
    calls = [event for event in control.events if event[0] == "capacity"]
    assert len(calls) == 1 and calls[0][1] == "before-sampler-constructor"
    assert calls[0][2]["worker"] is worker and calls[0][2]["engine_kwargs"] is options
    assert worker._ap_initialization_admission["status"] == "verified"
    assert options == {"enable_lora": True, "max_lora_rank": 32, "tensor_parallel_size": 8, "model": "original"}


@pytest.mark.parametrize("cause", ["missing", "provider", "args"])
def test_initialization_hooks_sampler_rejection_precedes_actual_constructor(
    initialization_hooks, native_modules, monkeypatch, cause
):
    control = initialization_hooks
    if cause == "missing":
        monkeypatch.delenv(hooks.INITIALIZATION_ENV)
    else:
        control.request["capacity_request"].update(
            {"fail_phase": "before-sampler-constructor"} if cause == "provider" else {"mutate_args": True}
        )
        control.publish()
    worker = _lora_sampler_worker_bridge.LoRASamplerEvidenceWorker(native_modules)
    expected = (
        "requires an explicit FAST request"
        if cause == "missing"
        else "supplier rejected" if cause == "provider" else "changed the native sampler recipe"
    )
    with pytest.raises(RuntimeError if cause == "provider" else ValueError, match=expected):
        asyncio.run(worker.initialize({"enable_lora": True, "max_lora_rank": 32, "tensor_parallel_size": 8}))
    assert LLM.constructed == [] and worker.state == "uninitialized"


def test_initialization_hooks_production_calls_bracket_native_worker_and_sampler_boundaries():
    root = Path(hooks.__file__).resolve().parents[1]
    tree = ast.parse((root / "common/deepspeed_worker.py").read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepSpeedWorker")
    initialize = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "initialize")
    source = ast.unparse(initialize)
    assert source.index("maybe_admit_worker_construction(") < source.index("build_model(spec")
    assert source.index("deepspeed.initialize(**init_kwargs)") < source.index("maybe_verify_worker_engine(self)")
    assert source.index("maybe_verify_worker_engine(self)") < source.index("self._device =")
    for name in ("rl_sampler_evidence", "lora_sampler_worker", "lora_selected_sampler"):
        source = (root / f"debug/{name}.py").read_text()
        assert source.index("maybe_admit_sampler_construction(self,") < source.index("self.llm = AsyncLLM")
        assert source.index("self.llm = AsyncLLM") < source.index("maybe_verify_sampler_construction(self)")


@contextlib.contextmanager
def _initialization_bootstrap_base(staging, *, cluster, guard):
    guard()
    yield {"settings": {"SYNTHETIC_BASE_ENV": "unchanged"}}


def _initialization_peer_launch(hosts, argv, **kwargs):
    _INITIALIZATION_EVENTS.append(("peer", hosts, argv, kwargs))
    return SimpleNamespace(returncode=0)


@pytest.fixture
def initialization_hooks_bootstrap(initialization_hooks, _lora_request_staging_inputs, monkeypatch):
    from arctic_platform.debug import lora_bootstrap_environment as bootstrap

    control = initialization_hooks
    cfg, options = _lora_request_staging_inputs
    monkeypatch.setattr(_lora_request_staging_helper, "bootstrap_descriptor", selected.bootstrap_descriptor)
    options["admitted_sources"].update(control.request["admitted_sources"])
    options["admitted_sources"].update(
        source_files(
            bootstrap,
            selected.bootstrap_descriptor,
            selected.engine_kwargs_for_lora,
            _initialization_bootstrap_base,
        ),
    )
    options["admitted_sources"][str(selected.RPC_PATH)] = hashlib.sha256(selected.RPC_PATH.read_bytes()).hexdigest()
    staged = _lora_request_staging_helper.stage_selected_lora_request(cfg, **options)
    control.request["admitted_sources"] = staged["admitted_sources"]
    control.publish()
    pointer = {
        "path": str(control.path),
        "sha256": hashlib.sha256(control.path.read_bytes()).hexdigest(),
    }
    return SimpleNamespace(
        bootstrap=bootstrap,
        control=control,
        staged=staged,
        pointer=pointer,
        cluster=SimpleNamespace(_pdsh=_initialization_peer_launch, _spawned_temp_dir="/tmp/j3rl-" + "a" * 24),
    )


@pytest.mark.parametrize("selected_initialization", [False, True])
@pytest.mark.parametrize("prior_environment", [False, True])
@pytest.mark.parametrize("body_failure", [False, True])
def test_initialization_hooks_bootstrap_forwards_only_selected_pair_and_restores_owner(
    initialization_hooks_bootstrap, monkeypatch, selected_initialization, prior_environment, body_failure
):
    case = initialization_hooks_bootstrap
    staged = copy.deepcopy(case.staged)
    if selected_initialization:
        staged["initialization"] = case.pointer
        staged["environment"].update(
            {
                hooks.INITIALIZATION_ENV: case.pointer["path"],
                hooks.INITIALIZATION_SHA_ENV: case.pointer["sha256"],
            },
        )
    keys = {*staged["environment"], hooks.INITIALIZATION_ENV, hooks.INITIALIZATION_SHA_ENV}
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    if prior_environment:
        for key, value in staged["environment"].items():
            monkeypatch.setenv(key, value)
    before = {key: os.environ.get(key) for key in keys}
    original = case.cluster._pdsh
    failure = RuntimeError("synthetic bootstrap body failed")
    receipt = None

    def execute():
        nonlocal receipt
        with case.bootstrap.owned_selected_lora_bootstrap(
            staged,
            cluster=case.cluster,
            base_environment=_initialization_bootstrap_base,
            staging=object(),
            guard=lambda: None,
        ) as receipt:
            assert {key: os.environ.get(key) for key in staged["environment"]} == staged["environment"]
            result = case.cluster._pdsh(
                [case.bootstrap.PEER],
                [
                    "env",
                    "SYNTHETIC_BASE_ENV=unchanged",
                    case.bootstrap.RAY,
                    "start",
                    "--address=10.4.166.121:23456",
                    f"--temp-dir={case.cluster._spawned_temp_dir}",
                    "--disable-usage-stats",
                ],
                check=False,
                timeout=600,
            )
            assert result.returncode == 0
            forwarded = receipt["peer_launches"][0]["argv"]
            for key, value in staged["environment"].items():
                assert f"{key}={value}" in forwarded
            if not selected_initialization:
                assert not any(item.startswith(hooks.INITIALIZATION_ENV + "=") for item in forwarded)
                assert not any(item.startswith(hooks.INITIALIZATION_SHA_ENV + "=") for item in forwarded)
            if body_failure:
                raise failure

    if body_failure:
        with pytest.raises(RuntimeError) as caught:
            execute()
        assert caught.value is failure
    else:
        execute()
    assert case.cluster._pdsh is original
    assert {key: os.environ.get(key) for key in keys} == before
    assert receipt["callback_restored"] and receipt["environment_restored"]
    assert receipt["capacity_admitted"] is False and receipt["numerical"] is None
    assert len(case.control.events) == 1 and case.control.events[0][0] == "peer"


def test_initialization_hooks_bootstrap_rejects_mismatched_source_closure_before_environment(
    initialization_hooks_bootstrap,
):
    case = initialization_hooks_bootstrap
    case.control.request["admitted_sources"] = copy.deepcopy(case.control.request["admitted_sources"])
    case.control.request["admitted_sources"].pop(str(Path(case.bootstrap.__file__).resolve()))
    case.control.publish()
    staged = copy.deepcopy(case.staged)
    staged["initialization"] = {
        "path": str(case.control.path),
        "sha256": hashlib.sha256(case.control.path.read_bytes()).hexdigest(),
    }
    staged["environment"].update(
        {
            hooks.INITIALIZATION_ENV: staged["initialization"]["path"],
            hooks.INITIALIZATION_SHA_ENV: staged["initialization"]["sha256"],
        },
    )
    before = dict(os.environ)
    with pytest.raises(ValueError, match="same complete source closure"):
        with case.bootstrap.owned_selected_lora_bootstrap(
            staged,
            cluster=case.cluster,
            base_environment=_initialization_bootstrap_base,
            staging=object(),
            guard=lambda: None,
        ):
            pytest.fail("mismatched sources reached the bootstrap body")
    assert dict(os.environ) == before and case.control.events == []
    assert case.cluster._pdsh is _initialization_peer_launch


@contextlib.contextmanager
def _initialization_bootstrap_ownership_base(staging, *, cluster, guard):
    guard()
    yield {"settings": {"SYNTHETIC_BASE_ENV": "unchanged"}}
    if cluster.teardown_failure is not None:
        raise cluster.teardown_failure


@pytest.mark.parametrize("changed_owner", ["callback", "environment", "both"])
@pytest.mark.parametrize("primary_failure", ["none", "body", "teardown"])
def test_bootstrap_restoration_preserves_foreign_owner(
    initialization_hooks_bootstrap, monkeypatch, changed_owner, primary_failure
):
    case = initialization_hooks_bootstrap
    keys = {*case.staged["environment"], hooks.INITIALIZATION_ENV, hooks.INITIALIZATION_SHA_ENV}
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    original = case.cluster._pdsh
    foreign = object()
    key = next(iter(case.staged["environment"]))
    failure = RuntimeError("synthetic primary " + primary_failure)
    case.cluster.teardown_failure = failure if primary_failure == "teardown" else None
    receipt = None
    with pytest.raises(ValueError if primary_failure == "none" else RuntimeError) as caught:
        with case.bootstrap.owned_selected_lora_bootstrap(
            case.staged,
            cluster=case.cluster,
            base_environment=_initialization_bootstrap_ownership_base,
            staging=object(),
            guard=lambda: None,
        ) as receipt:
            if changed_owner in ("callback", "both"):
                case.cluster._pdsh = foreign
            if changed_owner in ("environment", "both"):
                monkeypatch.setenv(key, "foreign owner")
            if primary_failure == "body":
                raise failure
    if primary_failure == "none":
        assert "captured bootstrap ownership changed" in str(caught.value)
    else:
        assert caught.value is failure
        assert any("captured bootstrap ownership changed" in note for note in failure.__notes__)
    assert case.cluster._pdsh is (foreign if changed_owner in ("callback", "both") else original)
    for name in case.staged["environment"]:
        expected = "foreign owner" if name == key and changed_owner in ("environment", "both") else None
        assert os.environ.get(name) == expected
    assert receipt["callback_restored"] is (changed_owner == "environment")
    assert receipt["environment_restored"] is (changed_owner == "callback")
    assert receipt["peer_runtime_admitted"] is False
    assert receipt["capacity_admitted"] is False and receipt["numerical"] is None


def test_bootstrap_restoration_does_not_claim_state_before_installation(initialization_hooks_bootstrap, monkeypatch):
    case = initialization_hooks_bootstrap
    for key in {*case.staged["environment"], hooks.INITIALIZATION_ENV, hooks.INITIALIZATION_SHA_ENV}:
        monkeypatch.delenv(key, raising=False)
    key = next(iter(case.staged["environment"]))
    foreign = object()
    failure = RuntimeError("synthetic initial guard failure")

    def guard():
        case.cluster._pdsh = foreign
        monkeypatch.setenv(key, "foreign guard owner")
        raise failure

    with pytest.raises(RuntimeError) as caught:
        with case.bootstrap.owned_selected_lora_bootstrap(
            case.staged,
            cluster=case.cluster,
            base_environment=_initialization_bootstrap_base,
            staging=object(),
            guard=guard,
        ):
            pytest.fail("failed initial guard reached bootstrap")
    assert caught.value is failure
    assert case.cluster._pdsh is foreign
    assert os.environ[key] == "foreign guard owner"
    assert all(os.environ.get(name) is None for name in case.staged["environment"] if name != key)


@pytest.mark.parametrize("changed_owner", ["callback", "environment", "both"])
def test_bootstrap_installation_preserves_foreign_owner_after_guard(
    initialization_hooks_bootstrap, monkeypatch, changed_owner
):
    case = initialization_hooks_bootstrap
    for key in {*case.staged["environment"], hooks.INITIALIZATION_ENV, hooks.INITIALIZATION_SHA_ENV}:
        monkeypatch.delenv(key, raising=False)
    key = next(iter(case.staged["environment"]))
    original = case.cluster._pdsh
    foreign = object()

    def guard():
        if changed_owner in ("callback", "both"):
            case.cluster._pdsh = foreign
        if changed_owner in ("environment", "both"):
            monkeypatch.setenv(key, "foreign guard owner")

    with pytest.raises(ValueError, match="ownership changed before installation"):
        with case.bootstrap.owned_selected_lora_bootstrap(
            case.staged,
            cluster=case.cluster,
            base_environment=_initialization_bootstrap_base,
            staging=object(),
            guard=guard,
        ):
            pytest.fail("foreign ownership reached bootstrap")
    assert case.cluster._pdsh is (foreign if changed_owner in ("callback", "both") else original)
    for name in case.staged["environment"]:
        expected = "foreign guard owner" if name == key and changed_owner in ("environment", "both") else None
        assert os.environ.get(name) == expected
    assert case.control.events == []


@pytest.mark.parametrize("change", ["code", "defaults", "keyword_defaults", "binding"])
def test_initialization_cleanup_does_not_invoke_changed_destroy(initialization_hooks, monkeypatch, change):
    control = initialization_hooks
    control.request["capacity_request"]["fail_phase"] = "after-engine"
    control.publish()
    _arm_initialization(control)
    engine = control.worker.engine
    destroy = type(control.worker).destroy

    def foreign_destroy(self):
        _INITIALIZATION_EVENTS.append(("foreign_destroy", self.engine))
        self.engine = None

    if change == "code":
        monkeypatch.setattr(destroy, "__code__", foreign_destroy.__code__)
    elif change == "defaults":
        monkeypatch.setattr(destroy, "__defaults__", (None,))
    elif change == "keyword_defaults":
        monkeypatch.setattr(destroy, "__kwdefaults__", {"foreign": True})
    else:
        monkeypatch.setattr(control.worker, "destroy", foreign_destroy.__get__(control.worker))
    with pytest.raises((ValueError, RuntimeError)) as caught:
        hooks.maybe_verify_worker_engine(control.worker)
    assert control.worker.engine is engine
    assert control.worker._ap_initialization_admission["status"] == "failed"
    assert not [event for event in control.events if event[0] in ("destroy", "foreign_destroy")]
    assert any("owned cleanup not attempted" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("phase", ["runtime", "capacity"])
def test_initialization_cleanup_drift_during_verification_preserves_primary(initialization_hooks, monkeypatch, phase):
    control = initialization_hooks
    _arm_initialization(control)
    engine = control.worker.engine
    failure = KeyboardInterrupt("synthetic engine verification interrupted")

    def foreign_destroy(self):
        _INITIALIZATION_EVENTS.append(("foreign_destroy", self.engine))
        self.engine = None

    class VerificationEvents(list):
        def append(self, event):
            super().append(event)
            selected = event[0] == "engine" if phase == "runtime" else event[:2] == ("capacity", "after-engine")
            if selected:
                monkeypatch.setattr(type(control.worker).destroy, "__code__", foreign_destroy.__code__)
                if phase == "runtime":
                    raise failure

    events = VerificationEvents(control.events)
    monkeypatch.setattr(sys.modules[__name__], "_INITIALIZATION_EVENTS", events)
    with pytest.raises(KeyboardInterrupt if phase == "runtime" else ValueError) as caught:
        hooks.maybe_verify_worker_engine(control.worker)
    if phase == "runtime":
        assert caught.value is failure
    else:
        assert "capacity verifier changed worker cleanup" in str(caught.value)
    assert control.worker.engine is engine
    assert control.worker._ap_initialization_admission["status"] == "failed"
    assert not [event for event in events if event[0] in ("destroy", "foreign_destroy")]
    assert any("owned cleanup not attempted" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("method", ["guard", "verify_engine"])
@pytest.mark.parametrize("attribute", ["__defaults__", "__kwdefaults__"])
def test_initialization_runtime_method_default_drift_refuses_engine(
    initialization_hooks, monkeypatch, method, attribute
):
    control = initialization_hooks
    _arm_initialization(control)
    function = getattr(_InitializationRuntime, method)
    value = (None,) if attribute == "__defaults__" else {"foreign": True}
    monkeypatch.setattr(function, attribute, value)
    with pytest.raises(ValueError, match="runtime verification method defaults changed"):
        hooks.maybe_verify_worker_engine(control.worker)
    assert not [event for event in control.events if event[0] == "engine"]
    assert control.worker._ap_initialization_admission["status"] == "failed"
    assert control.worker.engine is None
    assert len([event for event in control.events if event[0] == "destroy"]) == 1


@pytest.mark.parametrize("method", ["guard", "verify_engine"])
def test_initialization_runtime_defaults_changed_by_capacity_cannot_arm(initialization_hooks, monkeypatch, method):
    control = initialization_hooks

    class MutatingEvents(list):
        def append(self, event):
            super().append(event)
            if event[0] == "capacity":
                monkeypatch.setattr(getattr(_InitializationRuntime, method), "__kwdefaults__", {"foreign": True})

    events = MutatingEvents()
    monkeypatch.setattr(sys.modules[__name__], "_INITIALIZATION_EVENTS", events)
    with pytest.raises(ValueError, match="runtime verification method defaults changed"):
        _arm_initialization(control)
    assert not hasattr(control.worker, "_ap_initialization_admission")
    assert not [event for event in events if event[0] in ("engine", "destroy")]


def test_initialization_runtime_unchanged_nonempty_defaults_preserve_verification(initialization_hooks, monkeypatch):
    control = initialization_hooks
    monkeypatch.setattr(_InitializationRuntime.guard, "__defaults__", (None,))
    monkeypatch.setattr(_InitializationRuntime.verify_engine, "__kwdefaults__", {"sp_info": None, "sp_group": None})
    _arm_initialization(control)
    engine = control.worker.engine
    result = hooks.maybe_verify_worker_engine(control.worker)
    assert result["capacity_admitted"] is False and result["numerical"] is None
    assert control.worker.engine is engine
    assert control.worker._ap_initialization_admission["status"] == "verified"
    assert len([event for event in control.events if event[0] == "engine"]) == 1
    assert not [event for event in control.events if event[0] == "destroy"]


def test_initialization_runtime_default_guards_leave_unselected_worker_unchanged(initialization_hooks, monkeypatch):
    control = initialization_hooks
    monkeypatch.delenv(hooks.INITIALIZATION_ENV)
    monkeypatch.delenv(hooks.INITIALIZATION_SHA_ENV)
    monkeypatch.setattr(_InitializationRuntime.guard, "__defaults__", (None,))
    monkeypatch.setattr(_InitializationRuntime.verify_engine, "__kwdefaults__", {"foreign": True})
    engine = control.worker.engine
    assert _arm_initialization(control) is None
    assert hooks.maybe_verify_worker_engine(control.worker) is None
    assert control.worker.engine is engine and control.events == []
    assert not hasattr(control.worker, "_ap_initialization_admission")


@pytest.mark.parametrize("method", ["guard", "verify_engine"])
def test_initialization_runtime_guard_default_mutation_stops_before_engine(initialization_hooks, monkeypatch, method):
    control = initialization_hooks
    _arm_initialization(control)

    class MutatingGuardEvents(list):
        def append(self, event):
            super().append(event)
            if event[0] == "guard":
                monkeypatch.setattr(getattr(_InitializationRuntime, method), "__kwdefaults__", {"foreign": True})

    events = MutatingGuardEvents()
    monkeypatch.setattr(sys.modules[__name__], "_INITIALIZATION_EVENTS", events)
    with pytest.raises(ValueError, match="runtime verification method defaults changed"):
        hooks.maybe_verify_worker_engine(control.worker)
    assert not [event for event in events if event[0] == "engine"]
    assert control.worker._ap_initialization_admission["status"] == "failed"
    assert control.worker.engine is None
    assert len([event for event in events if event[0] == "destroy"]) == 1


def _model_factory_load(spec, groups):
    from arctic_platform.model.factory import build_model

    return build_model(spec, parallel_groups=groups)


class _ModelFactoryWorker:
    def __init__(self, cfg, spec, groups):
        self.cfg = cfg
        self.loaded = _model_factory_load(spec, groups)


def _model_factory_capacity(request, *, phase, subject, runtime):
    result = _initialization_capacity(request, phase=phase, subject=subject, runtime=runtime)
    damage = request.get("damage")
    if damage == "spec":
        subject["model_spec"].dtype = "float32"
    elif damage == "config":
        subject["config"]["original_recipe"] = False
    elif damage == "groups":
        subject["parallel_groups"]["sp_group"] = object()
    elif damage == "source":
        Path(request["source_path"]).write_text("synthetic source drift")
    return result


@pytest.fixture
def model_factory_initialization(initialization_hooks, monkeypatch):
    from arctic_platform.model import factory
    from arctic_platform.model.config import ModelSpec

    control = initialization_hooks
    control.request = copy.deepcopy(control.request)
    control.request["schema"] = "explicit-native-model-construction-request-v1"
    control.request["admitted_sources"].update(source_files(factory, ModelSpec))
    control.request["verify_capacity"]["name"] = "_model_factory_capacity"
    source = str(Path(__file__).resolve())
    control.request["factory_route"] = {
        "factory": {"module": factory.__name__, "qualname": "build_model", "source": factory.__file__},
        "caller": {"module": __name__, "qualname": "_model_factory_load", "source": source},
        "constructor": {"module": __name__, "qualname": "_ModelFactoryWorker.__init__", "source": source},
    }
    monkeypatch.delenv(hooks.INITIALIZATION_ENV)
    monkeypatch.delenv(hooks.INITIALIZATION_SHA_ENV)

    def publish():
        control.path.write_text(json.dumps(control.request, sort_keys=True))
        monkeypatch.setenv(hooks.MODEL_CONSTRUCTION_ENV, str(control.path))
        monkeypatch.setenv(
            hooks.MODEL_CONSTRUCTION_SHA_ENV,
            hashlib.sha256(control.path.read_bytes()).hexdigest(),
        )

    publish()
    spec = ModelSpec(model_path_or_name="synthetic-no-weights")
    groups = {"ep_group": object(), "sp_group": object()}
    config = {"original_recipe": True}
    loaded = SimpleNamespace(model=object())
    operations = []

    def allocate(debug):
        if os.environ.get(hooks.MODEL_CONSTRUCTION_ENV) is not None:
            calls = [event for event in control.events if event[0] == "capacity"]
            assert len(calls) == 1 and calls[0][1] == "before-build-model"
        operations.append(("allocate", debug))
        return None

    def select_loader(ctx):
        operations.append(("select", ctx))

        def load(actual):
            assert actual is ctx and actual.spec is spec and actual.parallel_groups is groups
            operations.append(("load", actual))
            return loaded

        return load

    monkeypatch.setattr(factory, "allocate_gpu_mem_preallocation", allocate)
    monkeypatch.setattr(factory, "select_loader", select_loader)
    monkeypatch.setattr(factory, "apply_patches", lambda actual, ctx: operations.append(("patch", actual)))
    monkeypatch.setattr(factory, "freeze_fp8_weights_and_scales", lambda model: operations.append(("freeze", model)))
    return SimpleNamespace(
        control=control,
        factory=factory,
        spec=spec,
        groups=groups,
        config=config,
        loaded=loaded,
        operations=operations,
        publish=publish,
    )


@pytest.mark.parametrize("evidence_optin", [False, True])
def test_model_factory_initialization_unselected_route_preserves_normal_factory(
    model_factory_initialization,
    monkeypatch,
    evidence_optin,
):
    case = model_factory_initialization
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_ENV)
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_SHA_ENV)
    if evidence_optin:
        monkeypatch.setenv(VERIFY_ENV, "1")
    else:
        monkeypatch.delenv(VERIFY_ENV)
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert worker.loaded is case.loaded
    assert case.control.events == []
    assert [event[0] for event in case.operations] == ["allocate", "select", "load", "patch", "freeze"]


def test_model_factory_initialization_actual_spec_config_groups_and_route_precede_allocation(
    model_factory_initialization,
):
    case = model_factory_initialization
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    calls = [event for event in case.control.events if event[0] == "capacity"]
    assert len(calls) == 1 and calls[0][1] == "before-build-model"
    subject = calls[0][2]
    assert subject["worker"] is worker and subject["config"] is case.config
    assert subject["model_spec"] is case.spec and subject["parallel_groups"] is case.groups
    assert subject["build_model"] is case.factory.build_model
    assert subject["factory_route"] == case.control.request["factory_route"]
    assert worker.loaded is case.loaded
    assert case.spec.dtype == "bfloat16" and case.config == {"original_recipe": True}
    assert [event[0] for event in case.operations] == ["allocate", "select", "load", "patch", "freeze"]
    assert not hasattr(worker, "_ap_initialization_admission")


@pytest.mark.parametrize("missing", ["path", "digest", "provider", "route", "factory_source"])
def test_model_factory_initialization_missing_input_stops_before_any_allocation(
    model_factory_initialization,
    monkeypatch,
    missing,
):
    case = model_factory_initialization
    if missing in {"path", "digest"}:
        monkeypatch.delenv(
            hooks.MODEL_CONSTRUCTION_ENV if missing == "path" else hooks.MODEL_CONSTRUCTION_SHA_ENV,
        )
    else:
        if missing == "factory_source":
            case.control.request["admitted_sources"].pop(case.factory.__file__)
        else:
            case.control.request.pop("verify_capacity" if missing == "provider" else "factory_route")
        case.publish()
    with pytest.raises(ValueError):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert case.operations == [] and case.control.events == []


@pytest.mark.parametrize("field", ["module", "qualname", "source"])
def test_model_factory_initialization_selected_caller_must_match_actual_body(
    model_factory_initialization,
    field,
):
    case = model_factory_initialization
    case.control.request["factory_route"]["caller"][field] = "not.the.selected.caller"
    case.publish()
    with pytest.raises(ValueError, match="differs from selected caller"):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert case.operations == [] and case.control.events == []


@pytest.mark.parametrize("damage", ["provider", "empty", "spec", "config", "groups", "source"])
def test_model_factory_initialization_provider_or_input_drift_cannot_reach_loader(
    model_factory_initialization,
    damage,
):
    case = model_factory_initialization
    request = case.control.request["capacity_request"]
    if damage == "provider":
        request["fail_phase"] = "before-build-model"
    elif damage == "empty":
        request["empty"] = True
    else:
        request["damage"] = damage
        request["source_path"] = str(case.control.release)
    case.publish()
    with pytest.raises(RuntimeError if damage == "provider" else ValueError):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert case.operations == []
    assert len([event for event in case.control.events if event[0] == "capacity"]) == 1


def test_model_factory_initialization_rejects_unselected_direct_factory_caller(model_factory_initialization):
    case = model_factory_initialization
    with pytest.raises(ValueError, match="differs from selected caller"):
        case.factory.build_model(case.spec, parallel_groups=case.groups)
    assert case.operations == [] and case.control.events == []


def test_model_factory_initialization_worker_request_alone_does_not_select_factory(
    model_factory_initialization,
    monkeypatch,
):
    case = model_factory_initialization
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_ENV)
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_SHA_ENV)
    monkeypatch.setenv(hooks.INITIALIZATION_ENV, str(case.control.path))
    monkeypatch.setenv(hooks.INITIALIZATION_SHA_ENV, "a" * 64)
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert worker.loaded is case.loaded and case.control.events == []


def test_model_factory_initialization_reader_keeps_worker_selection_independent(initialization_hooks, monkeypatch):
    control = initialization_hooks
    monkeypatch.setenv(hooks.MODEL_CONSTRUCTION_ENV, "/data-fast/not-a-factory-request")
    monkeypatch.setenv(hooks.MODEL_CONSTRUCTION_SHA_ENV, "b" * 64)
    _arm_initialization(control)
    result = hooks.maybe_verify_worker_engine(control.worker)
    assert control.worker._ap_initialization_admission["request_reader"] is hooks._initialization_request
    assert result["capacity_admitted"] is False and result["numerical"] is None
    assert [event[1] for event in control.events if event[0] == "capacity"] == [
        "before-build-model",
        "after-engine",
    ]


def _model_constructor_frame_foreign_init(self, cfg, spec, groups):
    raise AssertionError("a foreign constructor must never run")


def _model_constructor_frame_capacity(request, *, phase, subject, runtime):
    result = _model_factory_capacity(request, phase=phase, subject=subject, runtime=runtime)
    damage = request.get("wrapper_damage")
    installed = type(subject["worker"]).__init__
    if damage == "binding":
        type(subject["worker"]).__init__ = _model_constructor_frame_foreign_init
    elif damage == "original_code":
        installed.__wrapped__.__code__ = _model_constructor_frame_foreign_init.__code__
    elif damage == "wrapped_pointer":
        installed.__wrapped__ = _model_constructor_frame_foreign_init
    elif damage == "closure":
        cells = dict(zip(installed.__code__.co_freevars, installed.__closure__, strict=True))
        cells["original"].cell_contents = _model_constructor_frame_foreign_init
    elif damage == "source":
        Path(request["wrapper_source"]).write_text("synthetic wrapper source drift")
    return result


@pytest.fixture
def model_constructor_frame_binding(model_factory_initialization, tmp_path, monkeypatch):
    case = model_factory_initialization
    path = tmp_path / "synthetic_owned_constructor.py"
    raw = """from functools import wraps
from arctic_platform.debug import lora_sync_hooks as hooks

def indirect(original, worker, args, kwargs):
    return original(worker, *args, **kwargs)

def make_wrapper(worker_class, original, sources, mode):
    @wraps(original)
    def wrapped(worker, *args, **kwargs):
        assert type(worker) is worker_class
        if mode == "unregistered":
            return original(worker, *args, **kwargs)
        selected_sources = dict(sources)
        if mode == "incomplete_sources":
            selected_sources.pop(__file__)
        with hooks.owned_model_constructor_frame(
            worker, constructor=original, wrapper=wrapped, admitted_sources=selected_sources,
        ):
            if mode == "nested":
                with hooks.owned_model_constructor_frame(
                    worker, constructor=original, wrapper=wrapped, admitted_sources=selected_sources,
                ):
                    return original(worker, *args, **kwargs)
            if mode == "indirect":
                return indirect(original, worker, args, kwargs)
            if mode == "wrong_receiver":
                return original(object.__new__(worker_class), *args, **kwargs)
            return original(worker, *args, **kwargs)
    return wrapped
"""
    path.write_text(raw)
    module = ModuleType("_synthetic_owned_model_constructor")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    case.control.request["admitted_sources"].update(source_files(module))
    case.control.request["verify_capacity"]["name"] = "_model_constructor_frame_capacity"
    case.publish()
    original = _ModelFactoryWorker.__init__
    monkeypatch.setattr(original, "__code__", original.__code__)

    def install(mode="owned"):
        wrapper = module.make_wrapper(
            _ModelFactoryWorker,
            original,
            case.control.request["admitted_sources"],
            mode,
        )
        monkeypatch.setattr(_ModelFactoryWorker, "__init__", wrapper)
        return wrapper

    return SimpleNamespace(case=case, original=original, install=install, source=path)


def test_model_constructor_frame_binding_actual_wrapper_allows_only_original_frame_before_allocation(
    model_constructor_frame_binding,
    monkeypatch,
):
    control = model_constructor_frame_binding
    case = control.case
    wrapper = control.install()
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    calls = [event for event in case.control.events if event[0] == "capacity"]
    assert len(calls) == 1 and calls[0][2]["worker"] is worker
    assert wrapper.__wrapped__ is control.original
    assert worker.loaded is case.loaded
    assert case.config == {"original_recipe": True} and case.spec.dtype == "bfloat16"
    assert [event[0] for event in case.operations] == ["allocate", "select", "load", "patch", "freeze"]
    assert hooks._MODEL_CONSTRUCTOR_FRAME.get() is None
    monkeypatch.setattr(_ModelFactoryWorker, "__init__", control.original)
    case.control.events.clear()
    case.operations.clear()
    assert _ModelFactoryWorker(case.config, case.spec, case.groups).loaded is case.loaded
    assert hooks._MODEL_CONSTRUCTOR_FRAME.get() is None


def test_model_constructor_frame_binding_absent_selection_preserves_wrapped_normal_route(
    model_constructor_frame_binding,
    monkeypatch,
):
    control = model_constructor_frame_binding
    case = control.case
    control.install()
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_ENV)
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_SHA_ENV)
    assert _ModelFactoryWorker(case.config, case.spec, case.groups).loaded is case.loaded
    assert case.control.events == [] and hooks._MODEL_CONSTRUCTOR_FRAME.get() is None


@pytest.mark.parametrize("mode", ["unregistered", "incomplete_sources", "nested", "indirect", "wrong_receiver"])
def test_model_constructor_frame_binding_cannot_admit_unowned_or_indirect_routes(
    model_constructor_frame_binding,
    mode,
):
    control = model_constructor_frame_binding
    case = control.case
    control.install(mode)
    with pytest.raises(ValueError):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert case.operations == [] and case.control.events == []
    assert hooks._MODEL_CONSTRUCTOR_FRAME.get() is None


def test_model_constructor_frame_binding_detached_wrapper_cannot_claim_class(
    model_constructor_frame_binding,
    monkeypatch,
):
    control = model_constructor_frame_binding
    case = control.case
    wrapper = control.install()
    monkeypatch.setattr(_ModelFactoryWorker, "__init__", control.original)
    worker = object.__new__(_ModelFactoryWorker)
    with pytest.raises(ValueError, match="constructor binding changed"):
        wrapper(worker, case.config, case.spec, case.groups)
    assert case.operations == [] and case.control.events == []
    assert hooks._MODEL_CONSTRUCTOR_FRAME.get() is None


@pytest.mark.parametrize("damage", ["binding", "original_code", "wrapped_pointer", "closure", "source"])
def test_model_constructor_frame_binding_provider_drift_never_reaches_loader(
    model_constructor_frame_binding,
    damage,
):
    control = model_constructor_frame_binding
    case = control.case
    control.install()
    request = case.control.request["capacity_request"]
    request["wrapper_damage"] = damage
    request["wrapper_source"] = str(control.source)
    case.publish()
    with pytest.raises(ValueError):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert case.operations == []
    assert len([event for event in case.control.events if event[0] == "capacity"]) == 1
    assert hooks._MODEL_CONSTRUCTOR_FRAME.get() is None


# The real post-sharding inventory is supplied before the original loader runs.


class _MaterializationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((2, 3), device="meta"))
        self.register_buffer("scratch", torch.zeros(2), persistent=False)


def _materialization_checkpoint_load(model, config, parallel_dims):
    _INITIALIZATION_EVENTS.append(("checkpoint_load", model, config, parallel_dims))


def _materialization_load(model, adapter, config, parallel_dims, ep_mesh):
    hooks.maybe_observe_model_materialization(model, adapter, config, parallel_dims, ep_mesh)
    adapter.load_dcp_from_hf(model, config, parallel_dims)


def _materialization_capacity(request, *, phase, subject, runtime):
    result = _initialization_capacity(request, phase=phase, subject=subject, runtime=runtime)
    if phase != "before-model-materialization":
        return result
    damage = request.get("materialization_damage")
    if damage == "buffer":
        subject["model"].register_buffer("unexpected", torch.zeros(1))
    elif damage == "recipe":
        subject["config"].original_recipe = False
    elif damage == "loader_defaults":
        subject["adapter"].load_dcp_from_hf.__defaults__ = (None,)
    elif damage == "empty_inventory":
        subject["native_inventory"].clear()
    elif damage == "parameter_identity":
        subject["model"].weight = nn.Parameter(torch.empty_like(subject["model"].weight))
    elif damage == "buffer_identity":
        subject["model"].scratch = torch.zeros_like(subject["model"].scratch)
    elif damage == "factory_binding":
        subject["build_model"].__globals__["build_model"] = _model_factory_load
    elif damage == "worker_config":
        subject["worker"].cfg["original_recipe"] = False
    elif damage == "model_spec":
        subject["build_model"].__globals__["_materialization_selected_spec"].dtype = "float32"
    return result


@pytest.fixture
def model_materialization(model_factory_initialization, monkeypatch):
    from arctic_platform.model.implementations.debug import gpu_memory

    case = model_factory_initialization
    model = _MaterializationModel()
    adapter = SimpleNamespace(load_dcp_from_hf=_materialization_checkpoint_load)
    config = SimpleNamespace(original_recipe=True)
    dims = SimpleNamespace(ep=2)
    mesh = object()
    case.loaded.model = model
    case.control.request["admitted_sources"].update(source_files(gpu_memory, _MaterializationModel))
    case.control.request["verify_capacity"]["name"] = "_materialization_capacity"
    monkeypatch.setattr(_materialization_checkpoint_load, "__defaults__", None)
    case.publish()

    def select_loader(ctx):
        def load(actual):
            assert actual is ctx
            _materialization_load(model, adapter, config, dims, mesh)
            return case.loaded

        return load

    monkeypatch.setattr(case.factory, "select_loader", select_loader)
    return SimpleNamespace(case=case, model=model, adapter=adapter, config=config, dims=dims, mesh=mesh)


def test_model_materialization_observes_actual_model_before_checkpoint_load(model_materialization):
    control = model_materialization
    case = control.case
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    events = [event for event in case.control.events if event[0] in ("capacity", "checkpoint_load")]
    assert [event[1] for event in events[:2]] == ["before-build-model", "before-model-materialization"]
    subject = events[1][2]
    assert subject["worker"] is worker and subject["model"] is control.model
    assert subject["ep_mesh"] is control.mesh and subject["parallel_dims"] is control.dims
    assert subject["native_inventory"]
    assert events[2] == ("checkpoint_load", control.model, control.config, control.dims)
    assert worker.loaded.model is control.model
    assert control.model.weight.device.type == "meta"


def test_model_materialization_disabled_does_not_inspect_or_require_factory(model_materialization, monkeypatch):
    control = model_materialization
    case = control.case
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_ENV)
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_SHA_ENV)
    _materialization_load(control.model, control.adapter, control.config, control.dims, control.mesh)
    assert case.control.events == [("checkpoint_load", control.model, control.config, control.dims)]


def test_model_materialization_selected_without_owning_factory_refuses_load(model_materialization):
    control = model_materialization
    with pytest.raises(ValueError, match="active owning model factory"):
        _materialization_load(control.model, control.adapter, control.config, control.dims, control.mesh)
    assert control.case.control.events == []


@pytest.mark.parametrize("damage", ["buffer", "recipe", "loader_defaults"])
def test_model_materialization_provider_drift_never_calls_checkpoint_loader(model_materialization, damage):
    case = model_materialization.case
    case.control.request["capacity_request"]["materialization_damage"] = damage
    case.publish()
    with pytest.raises(ValueError, match="capacity verifier changed"):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert not [event for event in case.control.events if event[0] == "checkpoint_load"]


def test_model_materialization_capacity_rejection_never_calls_checkpoint_loader(model_materialization):
    case = model_materialization.case
    case.control.request["capacity_request"]["fail_phase"] = "before-model-materialization"
    case.publish()
    with pytest.raises(RuntimeError, match="before-model-materialization"):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert not [event for event in case.control.events if event[0] == "checkpoint_load"]


def test_model_materialization_observation_cannot_change_inventory_snapshot(model_materialization):
    case = model_materialization.case
    case.control.request["capacity_request"]["materialization_damage"] = "empty_inventory"
    case.publish()
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert worker.loaded.model is model_materialization.model
    assert len([event for event in case.control.events if event[0] == "checkpoint_load"]) == 1


@pytest.mark.parametrize(
    "damage", ["parameter_identity", "buffer_identity", "factory_binding", "worker_config", "model_spec"]
)
def test_model_materialization_identity_drift_refuses_checkpoint_load(model_materialization, monkeypatch, damage):
    case = model_materialization.case
    monkeypatch.setattr(case.factory, "build_model", case.factory.build_model)
    monkeypatch.setattr(case.factory, "_materialization_selected_spec", case.spec, raising=False)
    case.control.request["capacity_request"]["materialization_damage"] = damage
    case.publish()
    with pytest.raises(ValueError, match="capacity verifier changed"):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert not [event for event in case.control.events if event[0] == "checkpoint_load"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_model_materialization_retained_identities_preserve_selected_recipe(
    model_materialization, model_constructor_frame_binding, wrapped
):
    control = model_materialization
    case = control.case
    if wrapped:
        model_constructor_frame_binding.install()
    case.control.request["verify_capacity"]["name"] = "_materialization_capacity"
    case.publish()
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    subject = next(
        event[2] for event in case.control.events if event[:2] == ("capacity", "before-model-materialization")
    )
    inventory = subject["native_inventory"]
    assert inventory["parameters"][0]["tensor"]["object_id"] == f"tensor:{id(control.model.weight):x}"
    assert inventory["buffers"][0]["tensor"]["object_id"] == f"tensor:{id(control.model.scratch):x}"
    assert inventory["modules"][0]["object_id"] == f"module:{id(control.model):x}"
    assert inventory["capacity_admitted"] is inventory["runtime_admitted"] is False
    assert subject["model_spec"] is case.spec and subject["native_config"] is case.config
    assert subject["parallel_groups"] is case.groups
    assert worker.loaded.model is control.model and control.model.weight.device.type == "meta"
    assert case.spec.dtype == "bfloat16" and case.config == {"original_recipe": True}
    assert [event for event in case.control.events if event[0] == "checkpoint_load"] == [
        ("checkpoint_load", control.model, control.config, control.dims)
    ]
    assert hooks._MODEL_CONSTRUCTOR_FRAME.get() is None


def test_model_materialization_production_hook_precedes_adapter_load():
    path = Path(__file__).resolve().parents[2] / "model/implementations/moe/deepspeed_integration.py"
    module = ast.parse(path.read_text())
    setup = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "setup_model_local_no_train"
    )
    calls = [node.value for node in setup.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    observation = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "maybe_observe_model_materialization"
    )
    load = next(
        call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == "load_dcp_from_hf"
    )
    assert calls.index(observation) + 1 == calls.index(load)
    assert [argument.id for argument in observation.args] == ["model", "adapter", "config", "parallel_dims", "ep_mesh"]


# Selected checkpoint reader observation.


class _CheckpointObservationReader:
    def __init__(self):
        self.path = "/synthetic/selected-cache"
        self.thread_count = 1
        self.storage_data = {"weight": {"file": "model.safetensors", "shape": [2, 3]}}
        self.result = object()
        self.failure = None

    def read_data(self, plan, planner):
        _INITIALIZATION_EVENTS.append(("checkpoint_read", self, plan, planner))
        if self.failure is not None:
            raise self.failure
        return self.result


class _CheckpointObservationPlanner:
    def __init__(self, state_dict):
        self.state_dict = dict(state_dict)

    def resolve_tensor(self, item):
        raise AssertionError("observation must not resolve destination tensors")

    def commit_tensor(self, item, tensor):
        raise AssertionError("observation must not commit destination tensors")

    def finish_plan(self, plan):
        return plan


def _checkpoint_observation_load(model, state_dict, config, parallel_dims):
    reader = _CheckpointObservationReader()
    if config.reader_failure:
        reader.failure = KeyboardInterrupt("synthetic checkpoint payload interruption")
    reader = hooks.maybe_observe_checkpoint_reader(reader, model, state_dict, config, parallel_dims)
    plan = SimpleNamespace(items=[SimpleNamespace(source="weight", offsets=[0, 0], lengths=[2, 3])])
    planner = _CheckpointObservationPlanner(state_dict)
    config.read_result = reader.read_data(plan, planner)
    config.actual_reader = reader
    if config.repeat:
        reader.read_data(plan, planner)


def _checkpoint_observation_capacity(request, *, phase, subject, runtime):
    result = _initialization_capacity(request, phase=phase, subject=subject, runtime=runtime)
    if phase != "before-checkpoint-read":
        return result
    damage = request.get("reader_damage")
    if damage == "plan":
        subject["load_plan"].items.clear()
    elif damage == "files":
        subject["reader"].storage_data.clear()
    elif damage == "threads":
        subject["reader"].thread_count = 2
    elif damage == "model":
        subject["model"].weight = nn.Parameter(torch.empty_like(subject["model"].weight))
    elif damage == "destinations":
        subject["checkpoint_destinations"]["weight"] = torch.zeros(2, 3)
    elif damage == "planner":
        subject["planner"].state_dict = dict(subject["planner"].state_dict)
    elif damage == "inventory":
        subject["native_inventory"].clear()
    elif damage == "planner_defaults":
        subject["planner"].resolve_tensor.__func__.__defaults__ = (None,)
    elif damage == "planner_keyword_defaults":
        subject["planner"].commit_tensor.__func__.__kwdefaults__ = {"tensor": None}
    elif damage == "reader_defaults":
        type(subject["reader"]).read_data.__defaults__ = (None,)
    return result


@pytest.fixture
def checkpoint_observation(model_factory_initialization, monkeypatch):
    from arctic_platform.model.implementations.debug import gpu_memory

    case = model_factory_initialization
    model = _MaterializationModel()
    model.to_empty(device="cpu")
    state_dict = {"weight": model.weight.detach()}
    config = SimpleNamespace(reader_failure=False, repeat=False)
    dims = SimpleNamespace(ep=2)
    case.loaded.model = model
    case.control.request["admitted_sources"].update(source_files(gpu_memory, _MaterializationModel))
    case.control.request["verify_capacity"]["name"] = "_checkpoint_observation_capacity"
    case.publish()

    def select_loader(ctx):
        def load(actual):
            assert actual is ctx
            _checkpoint_observation_load(model, state_dict, config, dims)
            return case.loaded

        return load

    monkeypatch.setattr(case.factory, "select_loader", select_loader)
    return SimpleNamespace(case=case, model=model, state_dict=state_dict, config=config, dims=dims)


def test_checkpoint_observation_supplies_materialized_storage_and_preserves_original_result(checkpoint_observation):
    control = checkpoint_observation
    case = control.case
    worker = _ModelFactoryWorker(case.config, case.spec, case.groups)
    events = [event for event in case.control.events if event[0] in ("capacity", "checkpoint_read")]
    assert [event[1] for event in events[:2]] == ["before-build-model", "before-checkpoint-read"]
    subject = events[1][2]
    assert subject["worker"] is worker and subject["model"] is control.model
    assert subject["load_plan"] is events[2][2] and subject["planner"] is events[2][3]
    assert control.config.read_result is control.config.actual_reader.result
    inventory = subject["native_inventory"]
    parameter = inventory["parameters"][0]["tensor"]
    destination = inventory["checkpoint_destinations"]["weight"]
    assert parameter["object_id"] != destination["object_id"]
    assert parameter["storage"]["storage_id"] == destination["storage"]["storage_id"]
    assert inventory["buffers"] and inventory["capacity_admitted"] is False
    assert subject["reader"].read_data.__func__ is not _CheckpointObservationReader.read_data


def test_checkpoint_observation_disabled_preserves_instance_without_factory(checkpoint_observation, monkeypatch):
    control = checkpoint_observation
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_ENV)
    monkeypatch.delenv(hooks.MODEL_CONSTRUCTION_SHA_ENV)
    reader = _CheckpointObservationReader()
    assert hooks.maybe_observe_checkpoint_reader(reader, object(), {}, object(), object()) is reader
    assert "read_data" not in vars(reader) and control.case.control.events == []


@pytest.mark.parametrize("damage", ["plan", "files", "threads", "model", "destinations", "planner"])
def test_checkpoint_observation_callback_drift_refuses_original_read(checkpoint_observation, damage):
    case = checkpoint_observation.case
    case.control.request["capacity_request"]["reader_damage"] = damage
    case.publish()
    with pytest.raises(ValueError):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert not [event for event in case.control.events if event[0] == "checkpoint_read"]


def test_checkpoint_observation_callback_inventory_is_an_independent_copy(checkpoint_observation):
    control = checkpoint_observation
    case = control.case
    case.control.request["capacity_request"]["reader_damage"] = "inventory"
    case.publish()
    _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert control.config.read_result is control.config.actual_reader.result
    assert len([event for event in case.control.events if event[0] == "checkpoint_read"]) == 1


def test_checkpoint_observation_rejection_does_not_read_payload(checkpoint_observation):
    case = checkpoint_observation.case
    case.control.request["capacity_request"]["fail_phase"] = "before-checkpoint-read"
    case.publish()
    with pytest.raises(RuntimeError, match="before-checkpoint-read"):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert not [event for event in case.control.events if event[0] == "checkpoint_read"]


def test_checkpoint_observation_original_exception_identity_is_preserved(checkpoint_observation):
    control = checkpoint_observation
    case = control.case
    control.config.reader_failure = True
    with pytest.raises(KeyboardInterrupt) as caught:
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    reads = [event for event in case.control.events if event[0] == "checkpoint_read"]
    assert len(reads) == 1 and caught.value is reads[0][1].failure


def test_checkpoint_observation_reader_is_single_use(checkpoint_observation):
    control = checkpoint_observation
    case = control.case
    control.config.repeat = True
    with pytest.raises(ValueError, match="single-use"):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert len([event for event in case.control.events if event[0] == "checkpoint_read"]) == 1


def test_checkpoint_observation_production_call_wraps_only_created_reader():
    path = Path(__file__).resolve().parents[2] / "model/implementations/qwen35/model_builder.py"
    module = ast.parse(path.read_text())
    load = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "load_dcp_from_hf")
    call = next(
        node for node in ast.walk(load) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dcp_load"
    )
    reader = next(keyword.value for keyword in call.keywords if keyword.arg == "storage_reader")
    assert reader.func.id == "maybe_observe_checkpoint_reader"
    assert reader.args[0].func.id == "HuggingFaceStorageReader"
    assert [argument.id for argument in reader.args[1:]] == ["model", "state_dict", "config", "parallel_dims"]


@pytest.mark.parametrize("damage", ["planner_defaults", "planner_keyword_defaults", "reader_defaults"])
def test_checkpoint_observation_method_default_drift_refuses_read(checkpoint_observation, monkeypatch, damage):
    control = checkpoint_observation
    case = control.case
    monkeypatch.setattr(_CheckpointObservationPlanner.resolve_tensor, "__defaults__", None)
    monkeypatch.setattr(_CheckpointObservationPlanner.commit_tensor, "__kwdefaults__", None)
    monkeypatch.setattr(_CheckpointObservationReader.read_data, "__defaults__", None)
    case.control.request["capacity_request"]["reader_damage"] = damage
    case.publish()
    with pytest.raises(ValueError, match="checkpoint (planner|reader)"):
        _ModelFactoryWorker(case.config, case.spec, case.groups)
    assert not [event for event in case.control.events if event[0] == "checkpoint_read"]


def test_checkpoint_observation_existing_instance_wrapper_is_not_replaced(checkpoint_observation):
    control = checkpoint_observation
    case = control.case
    reader = _CheckpointObservationReader()
    foreign = object()
    reader.read_data = foreign

    def load(model, state_dict, config, parallel_dims):
        return hooks.maybe_observe_checkpoint_reader(reader, model, state_dict, config, parallel_dims)

    with pytest.raises(ValueError, match="unwrapped selected reader"):
        load(control.model, control.state_dict, control.config, control.dims)
    assert reader.read_data is foreign and case.control.events == []
