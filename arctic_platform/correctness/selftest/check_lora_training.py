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
from arctic_platform.correctness.harness import execution
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

    def changed(path, *, refresh=False):
        value = original(path, refresh=refresh)
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

    async def _await_training_refs(self, submit_refs):
        return await asyncio.gather(*submit_refs())

    async def _await_weight_sync_refs(self, receive_task, send_tasks):
        received, *sent = await asyncio.gather(receive_task, *send_tasks)
        return received, sent


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


def test_lora_hooks_transfer_preserves_native_awaiter_failure(
    staged_request,
    monkeypatch,
):
    server, receive_task, send_tasks = Server(), object(), [object()]
    error = RuntimeError("sender failed")

    async def collect(actual_receive, actual_send):
        assert actual_receive is receive_task and actual_send is send_tasks
        raise error

    monkeypatch.setattr(server, "_await_weight_sync_refs", collect)
    with pytest.raises(RuntimeError, match="sender failed") as caught:
        asyncio.run(hooks.maybe_collect_lora_sync(server, receive_task, send_tasks))
    assert caught.value is error


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
    server = Server()
    original_await = server._await_training_refs

    async def native_await(submit_refs):
        seen.append("native-awaiter")
        return await original_await(submit_refs)

    monkeypatch.setattr(server, "_await_training_refs", native_await)
    assert asyncio.run(hooks.maybe_lora_max_param_bytes(server, worker)) == 16384
    assert seen == ["native-awaiter", "original-rpc"]


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


@pytest.mark.parametrize(
    "hash_seeds,targets",
    [
        (("1", "7"), ["q_proj", "v_proj"]),
        (("9", "23"), ["q_proj", "k_proj", "v_proj", "o_proj"]),
        (("27", "42"), ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    ],
    ids=["attention-pair", "all-attention", "attention-and-mlp"],
)
def test_generated_adapter_reproduces_identity_across_process_hash_seeds(
    generated_adapter_config, tmp_path, hash_seeds, targets
):
    control = generated_adapter_config
    config = {**control.config, "target_modules": targets}
    child = """
import json
import sys
from pathlib import Path
from arctic_platform.correctness.harness import shared_lora_adapter, storage

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
storage._DATA_FAST_ROOT = destination.parent
adapter = shared_lora_adapter.generate_adapter(source, json.loads(sys.argv[3]), destination, seed=42)
identity = shared_lora_adapter.adapter_identity(adapter)
print("GENERATED_ADAPTER_IDENTITY " + json.dumps(identity, sort_keys=True), flush=True)
"""
    identities = []
    for index, hash_seed in enumerate(hash_seeds):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                child,
                str(control.path),
                str(tmp_path / f"hash-seed-{index}"),
                json.dumps(config),
            ],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        prefix = "GENERATED_ADAPTER_IDENTITY "
        rows = [
            json.loads(line.removeprefix(prefix)) for line in completed.stdout.splitlines() if line.startswith(prefix)
        ]
        assert len(rows) == 1, completed.stdout
        identities.append(rows[0])
    assert identities[0] == identities[1]


@pytest.mark.parametrize(
    "targets",
    [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r".*\.(q_proj|v_proj)$",
    ],
    ids=["module-list", "module-regex"],
)
def test_generated_adapter_canonicalizes_only_serialized_module_order(
    generated_adapter_config, tmp_path, monkeypatch, targets
):
    control = generated_adapter_config
    config = {**control.config, "target_modules": targets}
    config_before = copy.deepcopy(config)
    original_save = peft.PeftModel.save_pretrained
    upstream = {}

    def save_with_unordered_modules(model, destination, *args, **kwargs):
        original_save(model, destination, *args, **kwargs)
        destination = Path(destination)
        config_path = destination / "adapter_config.json"
        serialized = json.loads(config_path.read_bytes())
        if isinstance(serialized["target_modules"], list):
            serialized["target_modules"] = sorted(serialized["target_modules"], reverse=True)
            config_path.write_text(json.dumps(serialized) + "\n")
        upstream["config"] = copy.deepcopy(serialized)
        upstream["tensors"] = (destination / "adapter_model.safetensors").read_bytes()

    monkeypatch.setattr(peft.PeftModel, "save_pretrained", save_with_unordered_modules)
    path = shared.generate_adapter(control.path, config, tmp_path / "generated")
    expected = copy.deepcopy(upstream["config"])
    expected["base_model_name_or_path"] = ""
    if isinstance(targets, list):
        expected["target_modules"] = sorted(targets)
    else:
        assert expected["target_modules"] == targets
    assert json.loads((path / "adapter_config.json").read_bytes()) == expected
    assert (path / "adapter_model.safetensors").read_bytes() == upstream["tensors"]
    assert config == config_before


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
def initialization_hooks_bootstrap(initialization_hooks, _lora_request_staging_inputs, monkeypatch, tmp_path):
    from arctic_platform.debug import lora_bootstrap_environment as bootstrap

    control = initialization_hooks
    cfg, options = _lora_request_staging_inputs
    # Descriptor validation needs an admitted source, not the retained fleet experiment.
    rpc_path = tmp_path / "selected_rpc.py"
    rpc_path.write_text("# Synthetic RPC source for bootstrap descriptor validation.\n")
    monkeypatch.setattr(selected, "RPC_PATH", rpc_path)
    monkeypatch.setattr(selected, "RPC_SHA", hashlib.sha256(rpc_path.read_bytes()).hexdigest())
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


# Original deterministic adapter initialization in the native worker constructor.


class _NativeOriginalAdapterBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"model_type": "synthetic"}
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise AssertionError("native adapter CPU controls must not run generation")

    def forward(self, *args, **kwargs):
        raise AssertionError("native adapter CPU controls must not execute a model")


def _native_original_adapter_model(dtype=torch.bfloat16, targets=("q_proj", "v_proj")):
    with torch.random.fork_rng(devices=[]):
        return peft.get_peft_model(
            _NativeOriginalAdapterBase(),
            peft.LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=list(targets)),
        ).to(dtype=dtype)


@pytest.fixture
def native_original_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DATA_FAST_ROOT", tmp_path)
    worker_root = tmp_path / "worker-scratch"
    monkeypatch.setenv(storage.CORRECTNESS_TMPDIR_ENV, str(worker_root))
    source = tmp_path / "original-adapter"
    source.mkdir()
    model = _native_original_adapter_model()
    model.peft_config["default"].save_pretrained(source)
    specs = [
        lora_init.LoraTensorSpec(name, tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."))
        for name, tensor in _installed(model).items()
    ]
    manifest = lora_init.build_manifest(specs, 73)
    state = {spec.name: lora_init.generate_tensor(spec, 73) for spec in specs}
    save_file(state, str(source / "adapter_model.safetensors"))
    identity = shared.adapter_identity(source)
    metadata = {
        "schema_version": 1,
        "initialization": manifest,
        "model_name": "synthetic-native-adapter-control",
        "model_config_sha256": hashlib.sha256(b"synthetic native topology").hexdigest(),
        "adapter_config_sha256": identity["adapter_config_sha256"],
        "peft_config": identity["peft_config"],
    }
    metadata["identity_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    (source / "initialization.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    assert lora_init.identity_metadata(source) == metadata
    files = shared.original_adapter_files(source, identity, metadata)
    training = ModuleType("selected_native_training_control")
    calls = []
    result = (object(), object(), object())

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        calls.append((model, ds_config, optimizer, ep_size, mpu, model_path, rank))
        lora_init.verify_tensors(peft.get_peft_model_state_dict(model), manifest)
        return result

    training.initialize_training_engine = engine
    return SimpleNamespace(
        path=source,
        files=files,
        identity=identity,
        metadata=metadata,
        manifest=manifest,
        worker_root=worker_root,
        training=training,
        engine=engine,
        calls=calls,
        result=result,
        model=_native_original_adapter_model(),
        ds_config={},
        optimizer=object(),
        mpu=object(),
    )


def _native_original_adapter_scope(control, **kwargs):
    return shared.native_worker_adapter_scope(
        control.training,
        control.files,
        control.identity,
        control.metadata,
        job_id=kwargs.get("job_id", "selected-job"),
        rank=kwargs.get("rank", 3),
    )


def _native_original_adapter_call(control, *, rank=3):
    return control.training.initialize_training_engine(
        control.model,
        control.ds_config,
        optimizer=control.optimizer,
        ep_size=4,
        mpu=control.mpu,
        model_path="unchanged-model-path",
        rank=rank,
    )


def _native_original_adapter_restored(control):
    assert control.training.initialize_training_engine is control.engine
    assert not control.worker_root.exists() or list(control.worker_root.iterdir()) == []


def test_native_original_adapter_metadata_transit_and_pre_optimizer_receipt(native_original_adapter, monkeypatch):
    control = native_original_adapter
    assert control.files == {path.name: path.read_bytes() for path in control.path.iterdir()}
    assert json.loads(control.files["initialization.json"]) == control.metadata
    events = []
    load = shared.initialize_adapter
    verify_metadata = lora_init.identity_metadata
    verify_manifest = lora_init.verify_tensors

    def initialize(model, path, identity):
        assert {item.name: item.read_bytes() for item in path.iterdir()} == control.files
        load(model, path, identity)
        events.append("loaded-original")

    def metadata(path):
        events.append("metadata")
        return verify_metadata(path)

    def manifest(*args, **kwargs):
        assert not kwargs.get("allow_exact_float32_widening", False)
        events.append("manifest")
        return verify_manifest(*args, **kwargs)

    original = control.engine

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        events.append("optimizer-boundary")
        return original(
            model, ds_config, optimizer=optimizer, ep_size=ep_size, mpu=mpu, model_path=model_path, rank=rank
        )

    monkeypatch.setattr(shared, "initialize_adapter", initialize)
    monkeypatch.setattr(lora_init, "identity_metadata", metadata)
    monkeypatch.setattr(lora_init, "verify_tensors", manifest)
    monkeypatch.setattr(control.training, "initialize_training_engine", engine)
    control.engine = engine
    shutil.rmtree(control.path)
    with _native_original_adapter_scope(control) as receipt:
        assert receipt["engine_initializations"] == 0
        assert _native_original_adapter_call(control) is control.result
    load_index = events.index("loaded-original")
    engine_index = events.index("optimizer-boundary")
    assert "metadata" in events[load_index + 1 : engine_index]
    assert "manifest" in events[load_index + 1 : engine_index]
    assert len(control.calls) == 1
    actual = control.calls[0]
    assert actual[0] is control.model and actual[1] is control.ds_config
    assert actual[2] is control.optimizer and actual[4] is control.mpu
    assert actual[3:] == (4, control.mpu, "unchanged-model-path", 3)
    assert receipt == {
        "schema": "original-peft-worker-initialization-v1",
        "job_id": "selected-job",
        "rank": 3,
        "engine_initializations": 1,
        "original_identity_sha256": control.metadata["identity_sha256"],
        "shared_identity": control.identity,
        "initialization": verify_manifest(peft.get_peft_model_state_dict(control.model), control.manifest),
    }
    assert json.loads(json.dumps(receipt)) == receipt
    _native_original_adapter_restored(control)


@pytest.mark.parametrize(
    "damage",
    ["metadata", "config", "bytes", "dtype", "coverage", "missing-metadata", "extra-file", "not-bytes"],
)
def test_native_original_adapter_pack_rejection_precedes_hook(native_original_adapter, damage):
    control = native_original_adapter
    if damage == "metadata":
        altered = copy.deepcopy(control.metadata)
        altered["model_name"] = "different-original"
        control.files["initialization.json"] = json.dumps(altered).encode()
    elif damage == "config":
        control.files["adapter_config.json"] += b" "
    elif damage in {"bytes", "dtype", "coverage"}:
        state = safetensors.torch.load(control.files["adapter_model.safetensors"])
        if damage == "bytes":
            state[sorted(state)[0]][0, 0] += 1
        elif damage == "dtype":
            state = {name: tensor.float() for name, tensor in state.items()}
        else:
            state.pop(sorted(state)[0])
        control.files["adapter_model.safetensors"] = save(state)
    elif damage == "missing-metadata":
        del control.files["initialization.json"]
    elif damage == "extra-file":
        control.files["../foreign-file"] = b"untrusted"
    else:
        control.files["initialization.json"] = "{}"
    before = {name: _bytes(value) for name, value in _installed(control.model).items()}
    with pytest.raises(ValueError):
        with _native_original_adapter_scope(control):
            pytest.fail("invalid original pack reached the constructor")
    assert before == {name: _bytes(value) for name, value in _installed(control.model).items()}
    assert control.calls == []
    _native_original_adapter_restored(control)


def test_native_original_adapter_rejects_forged_shared_identity_for_changed_original_bytes(native_original_adapter):
    control = native_original_adapter
    state = load_file(str(control.path / "adapter_model.safetensors"))
    state[sorted(state)[0]][0, 0] += 1
    save_file(state, str(control.path / "adapter_model.safetensors"))
    control.identity = shared.adapter_identity(control.path)
    control.files["adapter_model.safetensors"] = (control.path / "adapter_model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="bytes differ"):
        with _native_original_adapter_scope(control):
            pytest.fail("a replacement shared hash admitted different original factors")
    assert control.calls == []
    _native_original_adapter_restored(control)


@pytest.mark.parametrize("damage", ["dtype", "topology", "shape", "coverage"])
def test_native_original_adapter_target_rejection_precedes_optimizer(native_original_adapter, monkeypatch, damage):
    control = native_original_adapter
    if damage == "dtype":
        control.model = control.model.float()
    elif damage == "topology":
        control.model = _native_original_adapter_model(targets=("q_proj",))
    elif damage == "shape":
        control.model.base_model.model.q_proj.lora_A["default"].weight = nn.Parameter(
            torch.zeros(3, 4, dtype=torch.bfloat16)
        )
    else:
        state_dict = peft.get_peft_model_state_dict

        def missing(model, *args, **kwargs):
            state = state_dict(model, *args, **kwargs)
            state.pop(sorted(state)[0])
            return state

        monkeypatch.setattr(peft, "get_peft_model_state_dict", missing)
    before = {name: _bytes(value) for name, value in _installed(control.model).items()}
    with pytest.raises(ValueError):
        with _native_original_adapter_scope(control):
            _native_original_adapter_call(control)
    assert before == {name: _bytes(value) for name, value in _installed(control.model).items()}
    assert control.calls == []
    _native_original_adapter_restored(control)


@pytest.mark.parametrize("rank", [2, True, "3"])
def test_native_original_adapter_observed_rank_must_match_receipt(native_original_adapter, rank):
    control = native_original_adapter
    with pytest.raises(ValueError, match="rank differs"):
        with _native_original_adapter_scope(control):
            _native_original_adapter_call(control, rank=rank)
    assert control.calls == []
    _native_original_adapter_restored(control)


@pytest.mark.parametrize("job_id,rank", [("", 3), (" ", 3), (None, 3), ("selected-job", -1), ("selected-job", True)])
def test_native_original_adapter_receipt_requires_explicit_worker_identity(native_original_adapter, job_id, rank):
    control = native_original_adapter
    with pytest.raises(ValueError, match="job identity"):
        with _native_original_adapter_scope(control, job_id=job_id, rank=rank):
            pytest.fail("invalid worker identity reached the constructor")
    assert control.calls == []
    _native_original_adapter_restored(control)


def test_native_original_adapter_constructor_requires_one_hook(native_original_adapter):
    control = native_original_adapter
    with pytest.raises(ValueError, match="exactly one successful"):
        with _native_original_adapter_scope(control):
            pass
    _native_original_adapter_restored(control)


def test_native_original_adapter_multiple_hooks_fail_even_if_constructor_catches_second(native_original_adapter):
    control = native_original_adapter
    with pytest.raises(ValueError, match="exactly one successful"):
        with _native_original_adapter_scope(control):
            _native_original_adapter_call(control)
            with pytest.raises(ValueError, match="single-use"):
                _native_original_adapter_call(control)
    assert len(control.calls) == 1
    _native_original_adapter_restored(control)


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("drift", [False, True])
def test_native_original_adapter_engine_exception_identity_and_restoration(
    native_original_adapter, failure_type, drift
):
    control = native_original_adapter
    failure = failure_type("original engine interruption")

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        lora_init.verify_tensors(peft.get_peft_model_state_dict(model), control.manifest)
        if drift:
            control.training.initialize_training_engine = object()
        raise failure

    control.training.initialize_training_engine = control.engine = engine
    with pytest.raises(failure_type) as caught:
        with _native_original_adapter_scope(control) as receipt:
            _native_original_adapter_call(control)
    assert caught.value is failure
    assert receipt["engine_initializations"] == 0
    _native_original_adapter_restored(control)


def test_native_original_adapter_constructor_exception_is_not_replaced_by_missing_hook(native_original_adapter):
    control = native_original_adapter
    failure = KeyboardInterrupt("constructor interruption before engine")
    with pytest.raises(KeyboardInterrupt) as caught:
        with _native_original_adapter_scope(control):
            raise failure
    assert caught.value is failure
    _native_original_adapter_restored(control)


@pytest.mark.parametrize("phase", ["constructor", "loader", "engine"])
def test_native_original_adapter_binding_drift_is_rejected_and_restored(native_original_adapter, monkeypatch, phase):
    control = native_original_adapter
    original_load = shared.initialize_adapter
    original_engine = control.engine

    def initialize(*args, **kwargs):
        original_load(*args, **kwargs)
        control.training.initialize_training_engine = object()

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        result = original_engine(
            model, ds_config, optimizer=optimizer, ep_size=ep_size, mpu=mpu, model_path=model_path, rank=rank
        )
        control.training.initialize_training_engine = object()
        return result

    if phase == "loader":
        monkeypatch.setattr(shared, "initialize_adapter", initialize)
    elif phase == "engine":
        control.training.initialize_training_engine = control.engine = engine
    with pytest.raises(ValueError, match="binding changed"):
        with _native_original_adapter_scope(control):
            if phase == "constructor":
                control.training.initialize_training_engine = object()
            else:
                _native_original_adapter_call(control)
    assert len(control.calls) == int(phase == "engine")
    _native_original_adapter_restored(control)


def test_native_original_adapter_metadata_drift_after_loading_prevents_engine(native_original_adapter, monkeypatch):
    control = native_original_adapter
    original_load = shared.initialize_adapter

    def initialize(model, path, identity):
        original_load(model, path, identity)
        (path / "initialization.json").write_text("{}")

    monkeypatch.setattr(shared, "initialize_adapter", initialize)
    with pytest.raises(ValueError, match="initialization metadata"):
        with _native_original_adapter_scope(control):
            _native_original_adapter_call(control)
    assert control.calls == []
    _native_original_adapter_restored(control)


def test_native_original_adapter_single_worker_scope_rejects_nested_initialization(native_original_adapter):
    control = native_original_adapter
    with _native_original_adapter_scope(control):
        installed = control.training.initialize_training_engine
        with pytest.raises(ValueError, match="already active"):
            with _native_original_adapter_scope(control):
                pytest.fail("a second scope was installed")
        assert control.training.initialize_training_engine is installed
        _native_original_adapter_call(control)
    _native_original_adapter_restored(control)


def test_native_original_adapter_receipt_drift_rejects_normal_constructor_exit(native_original_adapter):
    control = native_original_adapter
    with pytest.raises(ValueError, match="receipt changed"):
        with _native_original_adapter_scope(control) as receipt:
            _native_original_adapter_call(control)
            receipt["rank"] = 8
    _native_original_adapter_restored(control)


def test_native_original_adapter_swallowed_engine_failure_cannot_produce_receipt(native_original_adapter):
    control = native_original_adapter

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        raise RuntimeError("original engine failure")

    control.training.initialize_training_engine = control.engine = engine
    with pytest.raises(ValueError, match="exactly one successful"):
        with _native_original_adapter_scope(control):
            with pytest.raises(RuntimeError, match="original engine failure"):
                _native_original_adapter_call(control)
    _native_original_adapter_restored(control)


def test_native_original_adapter_sender_rejects_metadata_drift_during_pack_read(native_original_adapter, monkeypatch):
    control = native_original_adapter
    original_files = shared.adapter_files

    def files(*args, **kwargs):
        transferred = original_files(*args, **kwargs)
        (control.path / "initialization.json").write_text("{}")
        return transferred

    monkeypatch.setattr(shared, "adapter_files", files)
    with pytest.raises(ValueError, match="metadata changed during transfer"):
        shared.original_adapter_files(control.path, control.identity, control.metadata)


def _native_original_composition_modules(monkeypatch, **leaves):
    modules = {}
    for name in (
        "dss",
        "dss.ray_dss",
        "dss.ray_dss.jobs",
        "dss.ray_dss.jobs.gpu",
        "dss.ray_dss.placement",
        "dss.ray_dss.zones",
    ):
        module = modules[name] = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, _, attribute = name.rpartition(".")
            setattr(modules[parent], attribute, module)
    for name, module in leaves.items():
        parent, _, attribute = name.rpartition(".")
        monkeypatch.setitem(sys.modules, name, module)
        setattr(modules[parent], attribute, module)


@pytest.mark.parametrize(
    "module,zone,backend,selected",
    [
        ("dss.ray_dss.zones.server", "training", "gpu", True),
        ("dss.ray_dss.zones.server", "sampling", "gpu", False),
        ("dss.ray_dss.zones.server", "training", "cpu", False),
        ("dss.ray_dss.zones.server_head", "training", "gpu", False),
        ("unrelated.module", "training", "gpu", False),
    ],
)
def test_native_original_composition_zone_command_selects_only_training_gpu(module, zone, backend, selected):
    import shlex

    from arctic_platform.correctness.harness import native_lora

    argv = [
        "exec",
        sys.executable,
        "-m",
        module,
        "--zone-type",
        zone,
        "--backend",
        backend,
        "--workdir",
        "/synthetic/path with spaces",
    ]
    command = shlex.join(argv)
    actual = native_lora._zone_command(command)
    if selected:
        argv[3:4] = ["arctic_platform.correctness.harness.native_lora", "zone"]
        assert shlex.split(actual) == argv
    else:
        assert actual == command


@pytest.mark.parametrize("command", ["kill -TERM -- 123", "pkill -f selected-owned-zone", "true", ""])
def test_native_original_composition_zone_command_leaves_cleanup_unchanged(command):
    from arctic_platform.correctness.harness import native_lora

    assert native_lora._zone_command(command) == command


def _native_original_composition_receipts(control):
    request = {
        "training_gpus": 2,
        "identity": control.identity,
        "original_metadata": control.metadata,
    }
    verified = lora_init.verify_tensors(
        safetensors.torch.load(control.files["adapter_model.safetensors"]), control.manifest
    )
    receipts = [
        {
            "schema": "original-peft-worker-initialization-v1",
            "job_id": "selected-child",
            "rank": rank,
            "engine_initializations": 1,
            "original_identity_sha256": control.metadata["identity_sha256"],
            "shared_identity": copy.deepcopy(control.identity),
            "initialization": copy.deepcopy(verified),
        }
        for rank in range(2)
    ]
    return request, receipts


def test_native_original_composition_verifies_every_rank_receipt(native_original_adapter):
    from arctic_platform.correctness.harness import native_lora

    request, receipts = _native_original_composition_receipts(native_original_adapter)
    assert native_lora._verify_receipts(receipts, request, "selected-child") is None


@pytest.mark.parametrize(
    "damage",
    [
        "missing-rank",
        "duplicate-rank",
        "rank-order",
        "bool-rank",
        "job",
        "engine-count",
        "bool-engine-count",
        "original-identity",
        "shared-identity",
        "manifest-identity",
        "missing-tensor",
        "tensor-name",
        "tensor-dtype",
        "tensor-bytes",
        "extra-tensor",
    ],
)
def test_native_original_composition_rejects_missing_duplicate_and_drifted_receipts(native_original_adapter, damage):
    from arctic_platform.correctness.harness import native_lora

    request, receipts = _native_original_composition_receipts(native_original_adapter)
    if damage == "missing-rank":
        receipts.pop()
    elif damage == "duplicate-rank":
        receipts[1]["rank"] = 0
    elif damage == "rank-order":
        receipts.reverse()
    elif damage == "bool-rank":
        receipts[0]["rank"] = False
    elif damage == "job":
        receipts[0]["job_id"] = "other-child"
    elif damage == "engine-count":
        receipts[0]["engine_initializations"] = 2
    elif damage == "bool-engine-count":
        receipts[0]["engine_initializations"] = True
    elif damage == "original-identity":
        receipts[0]["original_identity_sha256"] = "0" * 64
    elif damage == "shared-identity":
        receipts[0]["shared_identity"]["adapter_model_sha256"] = "0" * 64
    elif damage == "manifest-identity":
        receipts[0]["initialization"]["identity_sha256"] = "0" * 64
    elif damage == "missing-tensor":
        receipts[0]["initialization"]["verified_tensors"].pop()
    elif damage == "extra-tensor":
        receipts[0]["initialization"]["verified_tensors"].append(
            copy.deepcopy(receipts[0]["initialization"]["verified_tensors"][0])
        )
    else:
        field, value = {
            "tensor-name": ("name", "other.lora_A.weight"),
            "tensor-dtype": ("dtype", "float32"),
            "tensor-bytes": ("sha256", "0" * 64),
        }[damage]
        receipts[0]["initialization"]["verified_tensors"][0][field] = value
    with pytest.raises(ValueError, match="receipt|evidence"):
        native_lora._verify_receipts(receipts, request, "selected-child")


# Optional adapter capabilities must not gate an unrelated full-model gateway.


@pytest.fixture
def gateway_optional_adapter_control(monkeypatch):
    from arctic_platform.correctness.harness import native_lora

    training = ModuleType("dss.ray_dss.jobs.gpu.training")

    def constructor(self, cfg):
        raise AssertionError("no worker execution in capability discovery")

    def engine(model, *, rank):
        raise AssertionError("no engine execution in capability discovery")

    training._DeepSpeedWorker = type("SyntheticCapabilityWorker", (), {"__init__": constructor})
    training.initialize_training_engine = engine
    local = ModuleType("dss.ray_dss.placement.local")
    local.run_cmd = lambda command: command
    local.get_free_port = lambda: 9001
    _native_original_composition_modules(
        monkeypatch, **{"dss.ray_dss.placement.local": local, "dss.ray_dss.jobs.gpu.training": training}
    )
    document = {"openapi": "3.1", "paths": {"/unrelated": {}}}

    def original_openapi():
        return document

    schema = {"original": "cached"}
    gateway = SimpleNamespace(app=SimpleNamespace(openapi=original_openapi, openapi_schema=schema))
    return SimpleNamespace(
        native=native_lora,
        training=training,
        local=local,
        gateway=gateway,
        original_run=local.run_cmd,
        original_openapi=original_openapi,
        document=document,
        schema=schema,
    )


@pytest.mark.parametrize("missing", ["_DeepSpeedWorker", "initialize_training_engine"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_gateway_optional_adapter_missing_binding_preserves_full_model_route(
    gateway_optional_adapter_control, missing, interrupted, caplog
):
    control = gateway_optional_adapter_control
    delattr(control.training, missing)
    primary = RuntimeError("independent gateway interrupted") if interrupted else None
    captured = None
    try:
        with control.native.gateway_scope(control.gateway):
            assert control.gateway.app.openapi() == control.document
            assert control.native.OPENAPI_KEY not in control.gateway.app.openapi()
            assert control.native.SHARED_OPENAPI_KEY not in control.gateway.app.openapi()
            if primary:
                raise primary
    except BaseException as error:
        captured = error
    assert captured is primary
    assert control.local.run_cmd is control.original_run
    assert control.gateway.app.openapi is control.original_openapi
    assert control.gateway.app.openapi_schema is control.schema
    assert missing in caplog.text and "unavailable" in caplog.text


@pytest.mark.parametrize("kind", ["source", "type", "binding", "nested-attribute", "other-training-attribute"])
def test_gateway_optional_adapter_does_not_hide_invalid_sources(gateway_optional_adapter_control, monkeypatch, kind):
    control = gateway_optional_adapter_control
    if kind == "nested-attribute":
        error = AttributeError("nested binding missing", name="initialize_training_engine", obj=object())
    elif kind == "other-training-attribute":
        error = AttributeError("unrelated training field missing", name="unrelated", obj=control.training)
    else:
        error = {"source": OSError, "type": TypeError, "binding": ValueError}[kind]("invalid adapter source")

    def fail(training):
        assert training is control.training
        raise error

    monkeypatch.setattr(control.native, "_runtime_sources", fail)
    with pytest.raises(type(error)) as caught:
        with control.native.gateway_scope(control.gateway):
            raise AssertionError("invalid sources cannot enter gateway scope")
    assert caught.value is error
    assert control.local.run_cmd is control.original_run
    assert control.gateway.app.openapi is control.original_openapi
    assert control.gateway.app.openapi_schema is control.schema


def test_gateway_optional_adapter_available_capabilities_keep_exact_source_descriptors(
    gateway_optional_adapter_control,
):
    control = gateway_optional_adapter_control
    sources = control.native._runtime_sources(control.training)
    shared_sources = control.native._shared_runtime_sources(control.training)
    with control.native.gateway_scope(control.gateway):
        document = control.gateway.app.openapi()
        assert document[control.native.OPENAPI_KEY] == {"schema": control.native.SCHEMA, "sources": sources}
        assert document[control.native.SHARED_OPENAPI_KEY] == {
            "schema": control.native.SHARED_SCHEMA,
            "sources": shared_sources,
        }
        assert control.native.OPENAPI_KEY not in control.document
    assert control.local.run_cmd is control.original_run
    assert control.gateway.app.openapi is control.original_openapi


@pytest.mark.parametrize("shared_capability", [False, True])
def test_gateway_optional_adapter_absent_capability_still_rejects_adapter_client(monkeypatch, shared_capability):
    import requests

    from arctic_platform.correctness.harness import hosted
    from arctic_platform.correctness.harness import native_lora

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            assert url == "http://127.0.0.1:12345/openapi.json"
            assert kwargs == {"timeout": 30, "allow_redirects": False}
            assert self.trust_env is False
            return SimpleNamespace(status_code=200, json=lambda: {"openapi": "3.1"})

    monkeypatch.setattr(hosted, "require_local_cortex_gateway", lambda backend: None)
    monkeypatch.setattr(requests, "Session", Session)
    with pytest.raises(ValueError, match="does not support verified original PEFT initialization"):
        native_lora.gateway_capability(SimpleNamespace(base_url="http://127.0.0.1:12345"), shared=shared_capability)


@pytest.mark.parametrize("missing", ["_DeepSpeedWorker", "initialize_training_engine"])
def test_gateway_optional_adapter_missing_binding_still_rejects_worker_runtime(
    gateway_optional_adapter_control, missing
):
    control = gateway_optional_adapter_control
    sources = control.native._runtime_sources(control.training)
    delattr(control.training, missing)
    with pytest.raises(ValueError, match="runtime differs"):
        control.native._verify_runtime(control.training, sources)


@pytest.mark.parametrize("stale", ["original", "shared", "both"])
def test_gateway_optional_adapter_final_guard_removes_stale_capabilities(gateway_optional_adapter_control, stale):
    control = gateway_optional_adapter_control
    del control.training.initialize_training_engine
    keys = {
        "original": [control.native.OPENAPI_KEY],
        "shared": [control.native.SHARED_OPENAPI_KEY],
        "both": [control.native.OPENAPI_KEY, control.native.SHARED_OPENAPI_KEY],
    }[stale]
    for key in keys:
        control.document[key] = {"schema": "unverified-old-capability", "sources": {}}
    with control.native.gateway_scope(control.gateway):
        published = control.gateway.app.openapi()
        assert published == {"openapi": "3.1", "paths": {"/unrelated": {}}}
        assert all(key in control.document for key in keys)


def test_gateway_optional_adapter_final_guard_present_binding_attribute_error_is_not_absence(
    gateway_optional_adapter_control, monkeypatch
):
    control = gateway_optional_adapter_control
    error = AttributeError("existing source binding failed", name="initialize_training_engine", obj=control.training)

    def fail(training):
        raise error

    monkeypatch.setattr(control.native, "_runtime_sources", fail)
    with pytest.raises(AttributeError) as caught:
        with control.native.gateway_scope(control.gateway):
            raise AssertionError("present binding failure must not become absence")
    assert caught.value is error
    assert control.local.run_cmd is control.original_run
    assert control.gateway.app.openapi is control.original_openapi


@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt])
def test_native_original_composition_gateway_scope_restores_captured_bindings(monkeypatch, failure_type):
    from arctic_platform.correctness.harness import native_lora

    calls = []
    result = object()

    def run(command, *args, **kwargs):
        calls.append((command, args, kwargs))
        return result

    local = ModuleType("dss.ray_dss.placement.local")
    local.run_cmd = run
    local.get_free_port = lambda: 9001
    training = ModuleType("dss.ray_dss.jobs.gpu.training")
    _native_original_composition_modules(
        monkeypatch, **{"dss.ray_dss.placement.local": local, "dss.ray_dss.jobs.gpu.training": training}
    )
    schema = {"original": "cached"}
    document = {"openapi": "3.1"}

    def openapi():
        return document

    gateway = SimpleNamespace(app=SimpleNamespace(openapi=openapi, openapi_schema=schema))
    sources = {"selected": "runtime-sources"}
    monkeypatch.setattr(native_lora, "_runtime_sources", lambda actual: sources if actual is training else None)
    failure = failure_type("gateway scope interrupted") if failure_type else None
    captured = None
    try:
        with native_lora.gateway_scope(gateway):
            assert gateway.app.openapi is not openapi and local.run_cmd is not run
            published = gateway.app.openapi()
            assert published[native_lora.OPENAPI_KEY] == {"schema": native_lora.SCHEMA, "sources": sources}
            assert document == {"openapi": "3.1"}
            assert local.run_cmd("kill -TERM -- 123", result, unchanged=True) is result
            gateway.app.openapi_schema = {"temporary": True}
            if failure:
                local.run_cmd = object()
                gateway.app.openapi = object()
                raise failure
    except BaseException as error:
        captured = error
    assert captured is failure
    assert calls == [("kill -TERM -- 123", (result,), {"unchanged": True})]
    assert local.run_cmd is run and gateway.app.openapi is openapi and gateway.app.openapi_schema is schema


@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt])
def test_native_original_composition_manager_actor_table_restores_after_spawn(
    native_original_adapter, monkeypatch, failure_type
):
    from enum import Enum

    from arctic_platform.correctness.harness import native_lora

    control = native_original_adapter

    class ZoneType(Enum):
        TRAINING = "training"

    class Backend(Enum):
        GPU = "gpu"

    key = (ZoneType.TRAINING, Backend.GPU)
    actor = object()
    selected = object()
    result = object()
    failure = failure_type("original native spawn interrupted") if failure_type else None
    events = []
    manager_module = ModuleType("dss.ray_dss.zones.manager")
    manager_module._ACTOR_CLS = {key: actor}
    training = ModuleType("dss.ray_dss.jobs.gpu.training")
    training.GPUTrainingJob = actor

    class ZoneManager:
        zone_type = ZoneType.TRAINING
        backend = Backend.GPU

        def _spawn_actor(self, info):
            events.append(("spawn", self, info, manager_module._ACTOR_CLS[key]))
            if failure:
                raise failure
            return result

    manager_module.ZoneManager = ZoneManager
    _native_original_composition_modules(
        monkeypatch, **{"dss.ray_dss.jobs.gpu.training": training, "dss.ray_dss.zones.manager": manager_module}
    )
    ray = ModuleType("ray")
    transferred = []

    def put(files):
        transferred.append(files)
        return "owned-existing-cluster-ref"

    ray.put = put
    monkeypatch.setitem(sys.modules, "ray", ray)
    request = {
        "training_gpus": 2,
        "path": str(control.path),
        "identity": control.identity,
        "original_metadata": control.metadata,
        "sources": {"selected": "runtime"},
    }
    monkeypatch.setattr(native_lora, "_request", lambda cfg: request)
    monkeypatch.setattr(native_lora, "_verify_runtime", lambda actual, sources: events.append(("verify", actual)))

    def coordinator(actual, actual_request, ref):
        assert actual is training and actual_request is request and ref == "owned-existing-cluster-ref"
        return selected

    monkeypatch.setattr(native_lora, "_coordinator_class", coordinator)
    manager = ZoneManager()
    info = SimpleNamespace(config=object(), layout=[SimpleNamespace(gpu_ids=[0, 1])])
    native_lora.install_manager(manager)
    captured = None
    try:
        assert manager._spawn_actor(info) is result
    except BaseException as error:
        captured = error
    assert captured is failure
    assert events == [("verify", training), ("spawn", manager, info, selected)]
    assert transferred == [control.files]
    assert manager_module._ACTOR_CLS[key] is actor


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "spawn-runtime",
        "spawn-interrupt",
        "missing-receipt",
        "drifted-receipt",
        "spawn-cleanup-runtime",
        "spawn-cleanup-interrupt",
    ],
)
def test_native_original_composition_coordinator_restores_worker_and_owns_failure_teardown(
    native_original_adapter, monkeypatch, outcome, capsys
):
    from arctic_platform.correctness.harness import native_lora

    control = native_original_adapter
    request, receipts = _native_original_composition_receipts(control)
    request["sources"] = {"selected": "runtime"}
    ranks = [("owned-host-a", 0), ("owned-host-b", 1)]
    events = []
    failure = None
    if outcome in {"spawn-runtime", "spawn-cleanup-runtime"}:
        failure = RuntimeError("original native worker spawn")
    elif outcome in {"spawn-interrupt", "spawn-cleanup-interrupt"}:
        failure = KeyboardInterrupt("original native worker spawn")
    cleanup_failure = RuntimeError("synthetic cleanup failure") if "cleanup" in outcome else None
    training = ModuleType("dss.ray_dss.jobs.gpu.training")

    class NativeWorker:
        pass

    async def receive(receipt):
        return receipt

    class NativeCoordinator:
        def __init__(self):
            self._workers = []
            self._info = SimpleNamespace(config=object(), job_id="selected-child")

        async def _spawn_workers(self, cfg, actual_ranks, address, port, source):
            assert actual_ranks is ranks and cfg is config and source is source_stage
            assert address == "owned-master" and port == 12345
            selected = training._DeepSpeedWorker
            assert selected is not NativeWorker and issubclass(selected, NativeWorker)
            events.append("native-spawn")
            selected_receipts = copy.deepcopy(receipts)
            if outcome == "missing-receipt":
                selected_receipts.pop()
            elif outcome == "drifted-receipt":
                selected_receipts[0]["initialization"]["verified_tensors"][0]["sha256"] = "0" * 64
            self._workers = [
                SimpleNamespace(
                    original_adapter_receipt=SimpleNamespace(remote=lambda receipt=receipt: receive(receipt))
                )
                for receipt in selected_receipts
            ]
            if failure:
                raise failure

        async def _kill_workers(self, hosts):
            assert hosts == ["owned-host-a", "owned-host-b"]
            assert training._DeepSpeedWorker is NativeWorker
            events.append("owned-kill")
            if cleanup_failure:
                raise cleanup_failure
            self._workers = []

        def info(self):
            return {"native_info": "preserved"}

    training._DeepSpeedWorker = NativeWorker
    training.GPUTrainingJob = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=NativeCoordinator))
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.jobs.gpu.training": training})
    ray = ModuleType("ray")

    def remote(**options):
        assert options == {"num_cpus": 1, "num_gpus": 0}
        return lambda cls: cls

    ray.remote = remote
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(native_lora, "_verify_runtime", lambda actual, sources: None)
    monkeypatch.setattr(native_lora, "_request", lambda config: copy.deepcopy(request))
    coordinator = native_lora._coordinator_class(training, request, "existing-owned-cluster-ref")()
    config = {}
    source_stage = object()
    captured = None
    try:
        asyncio.run(coordinator._spawn_workers(config, ranks, "owned-master", 12345, source_stage))
    except BaseException as error:
        captured = error
    assert training._DeepSpeedWorker is NativeWorker
    if outcome == "success":
        assert captured is None and events == ["native-spawn"]
        assert coordinator.info() == {
            "native_info": "preserved",
            "correctness_original_adapter": {"schema": native_lora.SCHEMA, "workers": receipts},
        }
        assert json.loads(capsys.readouterr().out)["workers"] == receipts
    else:
        assert captured is failure if failure is not None else isinstance(captured, ValueError)
        assert events == ["native-spawn", "owned-kill"]
        if cleanup_failure:
            assert any("synthetic cleanup failure" in note for note in getattr(captured, "__notes__", []))
        else:
            assert coordinator._workers == []


@pytest.mark.parametrize("role", ["zone", "head"])
@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt])
def test_native_original_composition_bootstrap_restores_original_callback(monkeypatch, role, failure_type):
    from arctic_platform.correctness.harness import native_lora

    module_name = f"dss.ray_dss.zones.{'server' if role == 'zone' else 'server_head'}"
    server = ModuleType(module_name)
    owner = object()
    server.app = SimpleNamespace(state=SimpleNamespace(manager=owner))
    calls = []
    marker = object()
    failure = failure_type("bootstrap interruption") if failure_type else None
    listener = SimpleNamespace(fileno=lambda: 43)
    monkeypatch.setattr(native_lora, "_zone_listener", lambda: contextlib.nullcontext(listener))
    if role == "zone":

        def original(*args, **kwargs):
            calls.append(("head-argv", args, kwargs))
            return [sys.executable, "-m", "dss.ray_dss.zones.server_head", "--unchanged-option"]

        server._head_argv = original
        server.run_cmd = lambda *args, **kwargs: None
        binding = "_head_argv"
    else:

        def original(*args, **kwargs):
            calls.append(("setup-state", args, kwargs))

        server._setup_state = original
        server._serve = lambda args: None
        binding = "_setup_state"
        monkeypatch.setattr(native_lora, "install_manager", lambda manager: calls.append(("install", manager)))

    def main():
        callback = getattr(server, binding)
        assert callback is not original
        actual = callback(marker, unchanged=True)
        if role == "zone":
            assert actual == [
                sys.executable,
                "-m",
                "arctic_platform.correctness.harness.native_lora",
                "head",
                "--unchanged-option",
                "--correctness-listener-fd",
                "43",
            ]
        if failure:
            setattr(server, binding, object())
            raise failure

    server.main = main
    _native_original_composition_modules(monkeypatch, **{module_name: server})
    monkeypatch.setattr(sys, "argv", ["native_lora.py", role, "--remaining"])
    captured = None
    try:
        native_lora.main()
    except BaseException as error:
        captured = error
    assert captured is failure and getattr(server, binding) is original
    expected = [("head-argv" if role == "zone" else "setup-state", (marker,), {"unchanged": True})]
    if role == "head":
        expected.append(("install", owner))
    assert calls == expected and sys.argv == ["native_lora.py", "--remaining"]


@pytest.fixture
def native_reserved_listener_control(monkeypatch):
    from arctic_platform.correctness.harness import native_lora

    events = []
    listeners = []

    class Listener:
        family = native_lora.socket.AF_INET

        def __init__(self, host, port):
            self.host = host
            self.port = port or 43000 + len(listeners)
            self.descriptor = 70 + len(listeners)
            self.closed = False
            self.accepting = 1
            self.kind = native_lora.socket.SOCK_STREAM
            listeners.append(self)

        def fileno(self):
            return -1 if self.closed else self.descriptor

        def getsockname(self):
            return self.host, self.port

        def getsockopt(self, level, option):
            assert level == native_lora.socket.SOL_SOCKET
            return self.kind if option == native_lora.socket.SO_TYPE else self.accepting

        def setblocking(self, value):
            events.append(("blocking", value))

        def set_inheritable(self, value):
            events.append(("inheritable", value))

        def close(self):
            self.closed = True
            events.append(("closed", self.descriptor))

    monkeypatch.setattr(native_lora, "_reserve_listener", Listener)
    return SimpleNamespace(native=native_lora, Listener=Listener, listeners=listeners, events=events)


@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt])
def test_native_reserved_listener_gateway_handoff_and_restoration(native_reserved_listener_control, failure_type):
    import shlex

    control = native_reserved_listener_control
    launches = []

    def run(command, host, **kwargs):
        argv = shlex.split(command)
        listener, health = control.listeners
        assert not listener.closed and health.closed
        assert host == "owned-head"
        assert argv[-2:] == [
            control.native._LISTENER_FD_OPTION,
            str(listener.descriptor),
        ]
        assert kwargs == {"pass_fds": (12, listener.descriptor), "stdout": "owned-log"}
        launches.append(argv)
        if failure_type:
            raise failure_type("spawn failed")
        return "owned-child"

    def original_port():
        pytest.fail("must reserve rather than free the HTTP socket")

    local = SimpleNamespace(run_cmd=run, get_free_port=original_port, is_local=lambda host: True)
    with control.native._gateway_listener_scope(local):
        port = local.get_free_port()
        local.get_free_port()
        command = shlex.join(
            [
                "exec",
                sys.executable,
                "-m",
                "dss.ray_dss.zones.server",
                "--zone-type",
                "training",
                "--backend",
                "gpu",
                "--port",
                str(port),
            ]
        )
        if failure_type:
            with pytest.raises(failure_type, match="spawn failed"):
                local.run_cmd(command, "owned-head", pass_fds=(12,), stdout="owned-log")
        else:
            assert local.run_cmd(command, "owned-head", pass_fds=(12,), stdout="owned-log") == "owned-child"
        assert all(listener.closed for listener in control.listeners)
    assert len(launches) == 1 and local.run_cmd is run and local.get_free_port is original_port


@pytest.mark.parametrize("command", ["true", "kill -TERM -- 123", "exec python -m unrelated"])
def test_native_reserved_listener_unrelated_launch_unchanged(native_reserved_listener_control, command):
    control = native_reserved_listener_control
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))

    local = SimpleNamespace(run_cmd=original, get_free_port=lambda: 9001, is_local=lambda host: True)
    with control.native._gateway_listener_scope(local):
        local.get_free_port()
        local.run_cmd(command, "owned-head", unchanged=True)
        assert control.listeners[0].closed
    assert calls == [((command, "owned-head"), {"unchanged": True})]


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_native_reserved_listener_unused_reservation_closed(native_reserved_listener_control, failure_type):
    control = native_reserved_listener_control
    local = SimpleNamespace(run_cmd=lambda *args: None, get_free_port=lambda: 9001)
    original_port = local.get_free_port
    with pytest.raises(failure_type):
        with control.native._gateway_listener_scope(local):
            local.get_free_port()
            raise failure_type("scope interrupted")
    assert control.listeners[0].closed and local.get_free_port is original_port


def test_native_reserved_listener_missing_local_reservation_prevents_spawn(
    native_reserved_listener_control,
):
    import shlex

    control = native_reserved_listener_control
    local = SimpleNamespace(
        run_cmd=lambda *args, **kwargs: pytest.fail("missing reservation must prevent spawn"),
        get_free_port=lambda: 9001,
        is_local=lambda host: True,
    )
    with control.native._gateway_listener_scope(local):
        with pytest.raises(ValueError, match="no reserved"):
            local.run_cmd(
                shlex.join(
                    [
                        "exec",
                        sys.executable,
                        "-m",
                        "dss.ray_dss.zones.server",
                        "--zone-type",
                        "training",
                        "--backend",
                        "gpu",
                        "--port",
                        "9001",
                    ]
                ),
                "owned-head",
            )


@pytest.mark.parametrize("damage", [None, "port", "host", "kind", "listening", "family"])
def test_native_reserved_listener_inherited_identity_and_close(native_reserved_listener_control, monkeypatch, damage):
    control = native_reserved_listener_control
    listener = control.Listener("0.0.0.0", 43303)
    if damage == "port":
        listener.port += 1
    elif damage == "host":
        listener.host = "127.0.0.1"
    elif damage == "kind":
        listener.kind = control.native.socket.SOCK_DGRAM
    elif damage == "listening":
        listener.accepting = 0
    elif damage == "family":
        listener.family = control.native.socket.AF_INET6
    monkeypatch.setattr(
        control.native.socket,
        "socket",
        lambda *, fileno: listener if fileno == 70 else None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "owned",
            "--host",
            "0.0.0.0",
            "--port",
            "43303",
            control.native._LISTENER_FD_OPTION,
            "70",
            "--native-flag",
        ],
    )
    if damage:
        with pytest.raises(ValueError, match="does not match"):
            with control.native._zone_listener():
                pytest.fail("invalid listener admitted")
    else:
        with control.native._zone_listener() as actual:
            assert actual is listener and not listener.closed
            assert control.events == [("inheritable", False), ("blocking", False)]
    assert listener.closed and control.native._LISTENER_FD_OPTION not in sys.argv
    assert sys.argv[-1] == "--native-flag"


def test_native_reserved_listener_remote_bootstrap_reserves_before_ray(native_reserved_listener_control, monkeypatch):
    control = native_reserved_listener_control
    monkeypatch.setattr(sys, "argv", ["owned", "--host", "0.0.0.0", "--port", "43303"])
    with control.native._zone_listener() as listener:
        assert listener.getsockname() == ("0.0.0.0", 43303) and not listener.closed
    assert listener.closed


@pytest.mark.parametrize("phase", ["bind", "listen", "setblocking"])
def test_native_reserved_listener_reservation_failure_closes_socket(monkeypatch, phase):
    from arctic_platform.correctness.harness import native_lora

    events = []
    failure = OSError("reserved listener failed")

    class Socket:
        def bind(self, address):
            assert address == ("0.0.0.0", 43303)
            events.append("bind")
            if phase == "bind":
                raise failure

        def listen(self, backlog):
            assert backlog == 2048
            events.append("listen")
            if phase == "listen":
                raise failure

        def setblocking(self, blocking):
            assert blocking is False
            events.append("setblocking")
            raise failure

        def close(self):
            events.append("close")

    monkeypatch.setattr(native_lora.socket, "socket", lambda *args: Socket())
    with pytest.raises(OSError, match="reserved listener failed") as captured:
        native_lora._reserve_listener("0.0.0.0", 43303)
    assert captured.value is failure and events[-1] == "close"
    assert events[:-1] == ["bind", "listen", "setblocking"][: ["bind", "listen", "setblocking"].index(phase) + 1]


@pytest.mark.parametrize("outcome", ["success", "spawn-failure", "remote-head", "changed-descriptor"])
def test_native_reserved_listener_zone_head_handoff_and_worker_isolation(
    native_reserved_listener_control, monkeypatch, outcome
):
    import shlex

    control = native_reserved_listener_control
    listener = control.Listener("0.0.0.0", 43303)
    server = ModuleType("dss.ray_dss.zones.server")
    launches = []

    def original_argv(*args, **kwargs):
        return [
            sys.executable,
            "-m",
            "dss.ray_dss.zones.server_head",
            "--host",
            "0.0.0.0",
            "--port",
            "43303",
        ]

    failure = RuntimeError("head spawn failed")

    def original_run(command, host, **kwargs):
        argv = shlex.split(command)
        launches.append((argv, host, kwargs))
        if argv[4] == "head":
            assert not listener.closed
            assert kwargs == {
                "pass_fds": (12, listener.descriptor),
                "stdout": "owned-log",
            }
            if outcome == "spawn-failure":
                raise failure
        else:
            assert argv[:5] == ["exec", sys.executable, "-m", control.native._MODULE, "worker"]
            assert listener.closed and kwargs == {"stdout": "worker-log", "stdin": subprocess.PIPE}
        return SimpleNamespace(pid=123, stdin=SimpleNamespace(close=lambda: None), wait=lambda **kwargs: 0)

    server._head_argv = original_argv
    server.run_cmd = original_run
    server.is_local = lambda host: outcome != "remote-head"

    def main():
        argv = server._head_argv()
        assert argv[:4] == [sys.executable, "-m", control.native._MODULE, "head"]
        if outcome == "changed-descriptor":
            argv[-1] = "999"
        if outcome == "success":
            child = server.run_cmd(shlex.join(argv), "owned-head", pass_fds=(12,), stdout="owned-log")
            assert child.process.pid == 123
            assert listener.closed
            server.run_cmd(
                shlex.join([sys.executable, "-m", "dss.ray_dss.zones.server_worker"]),
                "owned-peer",
                stdout="worker-log",
            )
        else:
            server.run_cmd(shlex.join(argv), "owned-head", pass_fds=(12,), stdout="owned-log")

    server.main = main
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.zones.server": server})
    monkeypatch.setattr(
        control.native.socket,
        "socket",
        lambda *, fileno: listener if fileno == 70 else None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "owned",
            "zone",
            "--host",
            "0.0.0.0",
            "--port",
            "43303",
            control.native._LISTENER_FD_OPTION,
            "70",
        ],
    )
    if outcome == "success":
        control.native.main()
    else:
        expected = RuntimeError if outcome == "spawn-failure" else ValueError
        with pytest.raises(expected, match="spawn failed|cannot inherit"):
            control.native.main()
    assert listener.closed
    assert server._head_argv is original_argv and server.run_cmd is original_run
    assert len(launches) == (2 if outcome == "success" else 1 if outcome == "spawn-failure" else 0)


@pytest.mark.parametrize("failure_type", [None, RuntimeError, asyncio.CancelledError])
def test_native_reserved_listener_http_uses_socket_through_startup(native_reserved_listener_control, failure_type):
    control = native_reserved_listener_control
    listener = control.Listener("0.0.0.0", 43303)
    captured = []
    failure = failure_type("serve failed") if failure_type else None

    class Server:
        def __init__(self, config):
            self.config = config

        async def serve(self, sockets=None):
            assert sockets == [listener] and not listener.closed
            captured.append(sockets)
            if failure:
                raise failure
            return "served"

    original_uvicorn = SimpleNamespace(Server=Server, Config=SimpleNamespace, untouched=object())
    server_head = SimpleNamespace(uvicorn=original_uvicorn)

    async def original_serve(args):
        assert not listener.closed  # The DSS Ray-start call happens at this boundary.
        config = server_head.uvicorn.Config(host=args.host, port=args.port)
        assert server_head.uvicorn.untouched is original_uvicorn.untouched
        return await server_head.uvicorn.Server(config).serve()

    args = SimpleNamespace(host="0.0.0.0", port=43303, http_workers=1)
    if failure:
        with pytest.raises(failure_type, match="serve failed"):
            asyncio.run(control.native._serve_with_listener(server_head, original_serve, args, listener))
    else:
        assert (
            asyncio.run(control.native._serve_with_listener(server_head, original_serve, args, listener)) == "served"
        )
    assert captured == [[listener]] and server_head.uvicorn is original_uvicorn


@pytest.mark.parametrize("damage", ["address", "sockets", "workers"])
def test_native_reserved_listener_changed_server_contract_fails_closed(native_reserved_listener_control, damage):
    control = native_reserved_listener_control
    listener = control.Listener("0.0.0.0", 43303)

    class Server:
        def __init__(self, config):
            self.config = config

        async def serve(self, sockets=None):
            pytest.fail("changed listener contract must not reach native server")

    original_uvicorn = SimpleNamespace(Server=Server)
    server_head = SimpleNamespace(uvicorn=original_uvicorn)

    async def original_serve(args):
        if damage == "workers":
            pytest.fail("unsupported worker mode must stop before Ray")
        config = SimpleNamespace(host="0.0.0.0", port=43304 if damage == "address" else 43303)
        return await server_head.uvicorn.Server(config).serve(sockets=[] if damage == "sockets" else None)

    args = SimpleNamespace(host="0.0.0.0", port=43303, http_workers=2 if damage == "workers" else 1)
    with pytest.raises(ValueError, match="single-process|changed"):
        asyncio.run(control.native._serve_with_listener(server_head, original_serve, args, listener))
    assert server_head.uvicorn is original_uvicorn


def test_native_reserved_listener_changed_address_rejected_before_ray(
    native_reserved_listener_control,
):
    control = native_reserved_listener_control
    listener = control.Listener("0.0.0.0", 43303)

    async def original_serve(args):
        pytest.fail("listener mismatch must prevent Ray startup")

    args = SimpleNamespace(host="0.0.0.0", port=43304, http_workers=1)
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(control.native._serve_with_listener(SimpleNamespace(), original_serve, args, listener))


@pytest.fixture
def native_original_capability(native_original_adapter, monkeypatch):
    import requests

    from arctic_platform.correctness.harness import native_lora

    control = native_original_adapter
    pins = {
        key: {"path": f"/synthetic/{key}.py", "sha256": hashlib.sha256(key.encode()).hexdigest()}
        for key in ("training", "engine", "adapter", "lora", "composition", "native_adapter")
    }
    capability = {"schema": native_lora.SCHEMA, "sources": pins}
    response = SimpleNamespace(status_code=200, document={native_lora.OPENAPI_KEY: capability})
    gets = []

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert self.trust_env is False
            gets.append((url, kwargs))
            return SimpleNamespace(status_code=response.status_code, json=lambda: response.document)

        def post(self, *args, **kwargs):
            pytest.fail("capability admission must not create a training job")

    monkeypatch.setattr(requests, "Session", Session)
    backend = SimpleNamespace(base_url="http://127.0.0.1:8000", host=None, pat=None)
    payload = {
        "model_name": control.metadata["model_name"],
        "training_config": {"peft_config": control.identity["peft_config"], "n_gpus": 2},
    }
    return SimpleNamespace(control=control, pins=pins, response=response, gets=gets, backend=backend, payload=payload)


def test_native_original_composition_capability_explicit_bind_preserves_original_identity(native_original_capability):
    from arctic_platform.correctness.harness import native_lora

    case = native_original_capability
    before = copy.deepcopy(case.payload)
    bound = lora_init.bind_payload(case.payload, case.control.path, native_backend=case.backend)
    assert case.payload == before
    assert bound["training_config"]["debug"][native_lora.REQUEST_KEY] == {
        "schema": native_lora.SCHEMA,
        "path": str(case.control.path),
        "identity": case.control.identity,
        "original_metadata": case.control.metadata,
        "sources": case.pins,
        "training_gpus": 2,
        "model_name": case.control.metadata["model_name"],
    }
    assert case.gets == [
        ("http://127.0.0.1:8000/openapi.json", {"timeout": 30, "allow_redirects": False}),
    ]
    assert (
        shared.original_adapter_files(case.control.path, case.control.identity, case.control.metadata)
        == case.control.files
    )


@pytest.mark.parametrize(
    "damage",
    [
        "plain-route",
        "external",
        "https-loopback",
        "redirect",
        "no-capability",
        "schema",
        "missing-source",
        "relative-path",
        "short-hash",
    ],
)
def test_native_original_composition_capability_rejects_unadmitted_routes_without_create(
    native_original_capability, damage
):
    from arctic_platform.correctness.harness import native_lora

    case = native_original_capability
    if damage == "external":
        case.backend.base_url = "http://external.invalid:8000"
    elif damage == "https-loopback":
        case.backend.base_url = "https://127.0.0.1:8000"
    elif damage == "redirect":
        case.response.status_code = 302
    elif damage == "no-capability":
        case.response.document = {}
    elif damage == "schema":
        case.response.document[native_lora.OPENAPI_KEY]["schema"] = "unsupported"
    elif damage == "missing-source":
        del case.pins["training"]
    elif damage == "relative-path":
        case.pins["training"]["path"] = "relative/training.py"
    elif damage == "short-hash":
        case.pins["training"]["sha256"] = "bad"
    before = copy.deepcopy(case.payload)
    with pytest.raises((ValueError, lora_init.LoraInitializationPrerequisiteError)):
        lora_init.bind_payload(
            case.payload, case.control.path, native_backend=None if damage == "plain-route" else case.backend
        )
    assert case.payload == before
    assert len(case.gets) == int(damage not in {"plain-route", "external", "https-loopback"})


@pytest.mark.parametrize(
    "failure_type,runtime_mode",
    [
        (None, "mocked"),
        (RuntimeError, "mocked"),
        (KeyboardInterrupt, "mocked"),
        (None, "actual"),
        (None, "before-worker"),
        (None, "after-worker"),
        (None, "after-spawn"),
        (None, "after-receipt"),
    ],
)
def test_native_original_composition_worker_delegates_native_constructor_inside_preengine_scope(
    native_original_adapter, monkeypatch, failure_type, runtime_mode, capsys
):
    from types import FunctionType

    from arctic_platform.correctness.harness import native_adapter
    from arctic_platform.correctness.harness import native_lora

    control = native_original_adapter
    request, _ = _native_original_composition_receipts(control)
    request.update(training_gpus=1, sources={"selected": "runtime"})
    training = control.training
    events = []
    source_stage = object()
    config = {"job_id": "selected-child", "original_config": object()}
    failure = failure_type("original native constructor") if failure_type else None
    runtime_drift = {"observed": False}
    captured_worker = []
    original_load = native_adapter.initialize_adapter
    original_engine = control.engine

    def initialize(model, path, identity):
        assert model is control.model and identity == control.identity
        original_load(model, path, identity)
        events.append("loaded-original")

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        events.append("native-engine")
        return original_engine(
            model, ds_config, optimizer=optimizer, ep_size=ep_size, mpu=mpu, model_path=model_path, rank=rank
        )

    control.engine = training.initialize_training_engine = engine

    class NativeWorker:
        def __init__(self, cfg, source_stage_info=None):
            assert cfg is config and source_stage_info is source_stage
            assert training.initialize_training_engine is not engine
            captured_worker.append(self)
            events.append("native-constructor")
            assert (
                globals()["initialize_training_engine"](
                    control.model,
                    control.ds_config,
                    optimizer=control.optimizer,
                    ep_size=4,
                    mpu=control.mpu,
                    model_path="unchanged-model-path",
                    rank=0,
                )
                is control.result
            )
            if runtime_mode == "after-worker":
                runtime_drift["observed"] = True
            if failure:
                raise failure

    class NativeCoordinator:
        def __init__(self):
            self._workers = []
            self._info = SimpleNamespace(config=object(), job_id="selected-child")

        async def _spawn_workers(self, cfg, ranks, address, port, source_stage_info):
            selected_worker = training._DeepSpeedWorker
            # Emulate the original training-module import in the separate Ray worker process.
            training._DeepSpeedWorker = NativeWorker
            try:
                if runtime_mode == "before-worker":
                    runtime_drift["observed"] = True
                worker = selected_worker(cfg, source_stage_info)
            finally:
                training._DeepSpeedWorker = selected_worker

            async def receipt():
                if runtime_mode == "after-receipt":
                    runtime_drift["observed"] = True
                return worker.original_adapter_receipt()

            self._workers = [SimpleNamespace(original_adapter_receipt=SimpleNamespace(remote=receipt))]
            if runtime_mode == "after-spawn":
                runtime_drift["observed"] = True

        async def _kill_workers(self, hosts):
            assert runtime_mode in {"after-spawn", "after-receipt"}
            assert hosts == ["owned-host"] and training._DeepSpeedWorker is NativeWorker
            events.append("owned-kill")
            self._workers = []

    constructor = NativeWorker.__init__
    bound_constructor = FunctionType(
        constructor.__code__,
        training.__dict__,
        constructor.__name__,
        constructor.__defaults__,
        constructor.__closure__,
    )
    bound_constructor.__qualname__ = constructor.__qualname__
    NativeWorker.__init__ = bound_constructor
    training._DeepSpeedWorker = NativeWorker
    training.GPUTrainingJob = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=NativeCoordinator))
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.jobs.gpu.training": training})
    ray = ModuleType("ray")
    ref = object()
    gets = []

    def get(actual):
        assert actual is ref
        gets.append(actual)
        return control.files

    ray.get = get
    ray.remote = lambda **options: lambda cls: cls
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(native_adapter, "initialize_adapter", initialize)
    if runtime_mode == "mocked":
        monkeypatch.setattr(native_lora, "_verify_runtime", lambda actual, sources: None)
    else:
        request["sources"] = native_lora._runtime_sources(training)
        original_source = native_lora._source

        def observed_source(callback):
            source = original_source(callback)
            if runtime_drift["observed"]:
                source["sha256"] = "0" * 64
            return source

        monkeypatch.setattr(native_lora, "_source", observed_source)
    monkeypatch.setattr(native_lora, "_request", lambda cfg: copy.deepcopy(request))
    coordinator = native_lora._coordinator_class(training, request, ref)()
    caught = None
    try:
        asyncio.run(coordinator._spawn_workers(config, [("owned-host", 0)], "owned-master", 12345, source_stage))
    except BaseException as error:
        caught = error
    drifted = runtime_mode in {"before-worker", "after-worker", "after-spawn", "after-receipt"}
    if drifted:
        assert isinstance(caught, ValueError) and "runtime differs" in str(caught)
    else:
        assert caught is failure
    expected_events = (
        [] if runtime_mode == "before-worker" else ["native-constructor", "loaded-original", "native-engine"]
    )
    if runtime_mode in {"after-spawn", "after-receipt"}:
        expected_events.append("owned-kill")
        assert coordinator._workers == []
    assert events == expected_events
    assert len(captured_worker) == int(runtime_mode != "before-worker")
    assert gets == ([] if runtime_mode == "before-worker" else [ref])
    assert training._DeepSpeedWorker is NativeWorker
    if failure is None and not drifted:
        receipt = captured_worker[0].original_adapter_receipt()
        assert receipt["job_id"] == config["job_id"] and receipt["rank"] == 0
        assert receipt["original_identity_sha256"] == control.metadata["identity_sha256"]
        receipt["rank"] = 10
        assert captured_worker[0].original_adapter_receipt()["rank"] == 0
        assert json.loads(capsys.readouterr().out)["workers"][0]["rank"] == 0
    elif captured_worker and runtime_mode not in {"after-spawn", "after-receipt"}:
        assert not hasattr(captured_worker[0], "_correctness_original_adapter_receipt")
    _native_original_adapter_restored(control)


def test_native_original_composition_source_pin_unwraps_adapter_scope():
    from arctic_platform.correctness.harness import native_lora

    path = Path(shared.__file__)
    assert native_lora._source(shared.native_worker_adapter_scope) == {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    "test_id,local_fixture,has_fixture,expected_rejection",
    [
        ("e2e-sft-train-validate", True, True, False),
        ("e2e-sft-train-validate", True, False, True),
        ("e2e-sft-train-validate", False, True, True),
        ("e2e-sft-train-validate", False, False, True),
        ("single-step-grads", True, True, True),
        ("single-step-optimizer", True, True, True),
    ],
)
def test_native_e2e_admission_lora_exception_is_only_for_verified_local_e2e(
    test_id, local_fixture, has_fixture, expected_rejection
):
    cfg = SimpleNamespace(config_id="original-lora", training={"peft_config": {"r": 32}})
    test = SimpleNamespace(test_id=test_id, compares_to_reference=True)
    fixture = {"model_name": "verified-reduced-fixture"} if has_fixture else None
    result = runner._cortex_rejection(test, cfg, fixture, local_cortex_fixture=local_fixture)
    assert (result is not None) == expected_rejection
    if result is not None:
        assert result.outcome is Outcome.FAIL


@pytest.mark.parametrize(
    "selected,accepted",
    [
        (["e2e-sft-train-validate"], True),
        (["single-step-grads", "e2e-sft-train-validate"], True),
        (["inference-checkpoint-loss"], False),
        (["checkpoint-resume-loss", "e2e-sft-train-validate"], False),
    ],
)
def test_native_e2e_admission_cli_preserves_explicit_selection_and_local_boundary(
    tmp_path, monkeypatch, selected, accepted
):
    args = SimpleNamespace(
        config=tmp_path / "original.config",
        all_configs=False,
        test=selected,
        local_cortex_fixture=True,
        out=tmp_path / "report",
    )
    backend = CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev")
    monkeypatch.setattr(execution, "_configure_regression_target", lambda args: args)
    monkeypatch.setattr(execution, "_client_backend", lambda args: backend)
    reached = []

    class SelectionReached(RuntimeError):
        pass

    def resolve(actual, *, filter_by_local_gpu):
        assert actual is args and actual.test == selected and not filter_by_local_gpu
        reached.append(True)
        raise SelectionReached()

    monkeypatch.setattr(execution, "_resolve_configs", resolve)
    with pytest.raises(SelectionReached if accepted else SystemExit):
        execution._run_configs(args)
    assert bool(reached) is accepted


def test_native_e2e_admission_runner_still_requires_fixture_identity(monkeypatch):
    from arctic_platform.correctness.harness import cortex

    cfg = SimpleNamespace(config_id="original-lora", n_gpus=8, is_rl=False, training={"peft_config": {"r": 32}})
    cfg.at_gpu_width = lambda width: cfg
    spec = SimpleNamespace(applicable_tests=["e2e-sft-train-validate"])
    verified = []

    def missing_fixture(backend, actual_spec):
        assert backend.base_url == "http://127.0.0.1:8003" and actual_spec is spec
        verified.append(True)
        raise ValueError("original fixture bytes unavailable")

    monkeypatch.setattr(cortex, "load_local_fixture", missing_fixture)
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=256,
        select=["e2e-sft-train-validate"],
        local_cortex_fixture=True,
        client_backend=CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev"),
    )
    assert verified == [True]
    assert len(results) == 1 and results[0].outcome is Outcome.FAIL
    assert results[0].reason == "original fixture bytes unavailable"
    assert results[0].gpus == 8


# PEFT config serialization order; retained config bodies, synthetic CPU factors only.


@pytest.fixture
def lora_config_serialization_control(_lora_init_local_adapter_fixture, monkeypatch):
    cfg, source, scratch = _lora_init_local_adapter_fixture
    cfg.training["peft_config"] = {
        "bias": "none",
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "peft_type": "Lora",
        "r": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "target_parameters": [],
        "task_type": "CAUSAL_LM",
    }
    cfg.sub_job.update(dtype="float32", seed=42)
    cfg.model_name = "Qwen/Qwen3.6-35B-A3B"
    # Exact PEFT 0.21.2 config body from the retained ordering-only identity receipt.
    body = {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": None,
        "base_model_name_or_path": "/data-fast/base-models/synthetic/Qwen3.6-35B-A3B-4L",
        "bias": "none",
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "kasa_config": None,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 32,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "monteclora_config": None,
        "peft_type": "LORA",
        "peft_version": "0.21.2",
        "qalora_group_size": 16,
        "r": 32,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["o_proj", "q_proj", "v_proj", "k_proj"],
        "target_parameters": None,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
        "velora_config": None,
    }
    control = SimpleNamespace(cfg=cfg, source=source, scratch=scratch, body=body, saved=[])
    config_class = sys.modules["peft"].LoraConfig

    def save_pretrained(config, destination):
        raw = json.dumps(control.body, indent=2, sort_keys=True).encode()
        control.saved.append((config, copy.deepcopy(config.values), raw))
        (Path(destination) / "adapter_config.json").write_bytes(raw)

    monkeypatch.setattr(config_class, "save_pretrained", save_pretrained)
    return control


def test_lora_config_serialization_retained_orders_produce_exact_pack_identity(lora_config_serialization_control):
    control = lora_config_serialization_control
    before = copy.deepcopy(control.cfg.training)
    rng = torch.random.get_rng_state().clone()
    paths = []
    for modules, expected_raw_sha in (
        (
            ["o_proj", "q_proj", "v_proj", "k_proj"],
            "07c42f95b85a8cdc46b8363fc609c9368c0af3a90eb93f7db0e991014e510898",
        ),
        (
            ["q_proj", "o_proj", "v_proj", "k_proj"],
            "bff00c6f0665c17e8e95fe61bb1d7314efcb04ada251c27e45daa68f493ad345",
        ),
    ):
        control.body["target_modules"] = modules
        paths.append(deterministic_lora.materialize_adapter(control.cfg, control.source, control.scratch))
        config, values, raw = control.saved[-1]
        assert hashlib.sha256(raw).hexdigest() == expected_raw_sha
        assert config.values == values
        assert control.body["target_modules"] == modules
    assert paths[0] != paths[1]
    expected = copy.deepcopy(control.body)
    expected["target_modules"] = sorted(expected["target_modules"])
    config_bytes = (json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    for path in paths:
        assert (path / "adapter_config.json").read_bytes() == config_bytes
    first, second = map(deterministic_lora.identity_metadata, paths)
    assert first == second
    assert first["adapter_config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert first["peft_config"] == before["peft_config"]
    for name in ("initialization.json", "adapter_model.safetensors"):
        assert (paths[0] / name).read_bytes() == (paths[1] / name).read_bytes()
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert control.cfg.training == before
    assert {path.name for path in control.source.iterdir()} == {"config.json"}


@pytest.mark.parametrize("modules", [r".*\.(q_proj|v_proj)", [], ["q_proj", "q_proj", "k_proj"]])
def test_lora_config_serialization_helper_preserves_regex_empty_targets_and_other_fields(
    lora_config_serialization_control, tmp_path, modules
):
    body = copy.deepcopy(lora_config_serialization_control.body)
    body["target_modules"] = modules
    body["target_parameters"] = ["experts.up_proj", "experts.gate_proj"]
    body["future_field"] = {"ordered": ["z", "a"], "enabled": True}
    path = tmp_path / "adapter_config.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True))
    deterministic_lora._canonicalize_adapter_config(path)
    expected = copy.deepcopy(body)
    if isinstance(modules, list):
        expected["target_modules"] = sorted(modules)
    assert json.loads(path.read_bytes()) == expected
    assert json.loads(path.read_bytes())["target_parameters"] == ["experts.up_proj", "experts.gate_proj"]
    first = path.read_bytes()
    deterministic_lora._canonicalize_adapter_config(path)
    assert path.read_bytes() == first


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_modules", ["q_proj", "k_proj", "v_proj", "other_proj"]),
        ("lora_alpha", 64),
        ("lora_dropout", 0.25),
        ("r", 16),
        ("bias", "all"),
        ("inference_mode", True),
        ("target_parameters", ["experts.gate_proj", "experts.up_proj"]),
    ],
)
def test_lora_config_serialization_keeps_semantic_config_changes_in_identity(
    lora_config_serialization_control, field, value
):
    control = lora_config_serialization_control
    control.body["target_parameters"] = ["experts.up_proj", "experts.gate_proj"]
    first_path = deterministic_lora.materialize_adapter(control.cfg, control.source, control.scratch)
    control.body[field] = value
    second_path = deterministic_lora.materialize_adapter(control.cfg, control.source, control.scratch)
    first, second = map(deterministic_lora.identity_metadata, (first_path, second_path))
    assert json.loads((second_path / "adapter_config.json").read_bytes())[field] == (
        sorted(value) if field == "target_modules" else value
    )
    assert first["adapter_config_sha256"] != second["adapter_config_sha256"]
    assert first["identity_sha256"] != second["identity_sha256"]
    assert first["initialization"] == second["initialization"]
    assert first["peft_config"] == second["peft_config"]
    assert (first_path / "adapter_model.safetensors").read_bytes() == (
        second_path / "adapter_model.safetensors"
    ).read_bytes()
    assert {key: value for key, value in first.items() if key not in {"adapter_config_sha256", "identity_sha256"}} == {
        key: value for key, value in second.items() if key not in {"adapter_config_sha256", "identity_sha256"}
    }


def test_lora_config_serialization_keeps_tensor_changes_in_identity(lora_config_serialization_control):
    control = lora_config_serialization_control
    first_path = deterministic_lora.materialize_adapter(control.cfg, control.source, control.scratch)
    control.cfg.sub_job["seed"] = 43
    second_path = deterministic_lora.materialize_adapter(control.cfg, control.source, control.scratch)
    first, second = map(deterministic_lora.identity_metadata, (first_path, second_path))
    assert first["adapter_config_sha256"] == second["adapter_config_sha256"]
    assert first["peft_config"] == second["peft_config"]
    assert first["initialization"] != second["initialization"]
    assert first["identity_sha256"] != second["identity_sha256"]
    assert (first_path / "adapter_model.safetensors").read_bytes() != (
        second_path / "adapter_model.safetensors"
    ).read_bytes()


@pytest.mark.parametrize("modules", [None, True, 1, {}, "", ["q_proj", ""], ["q_proj", 1], ["q_proj", None]])
def test_lora_config_serialization_helper_rejects_malformed_modules_without_rewriting(
    lora_config_serialization_control, tmp_path, modules
):
    body = copy.deepcopy(lora_config_serialization_control.body)
    body["target_modules"] = modules
    path = tmp_path / "adapter_config.json"
    raw = json.dumps(body, sort_keys=True).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="target_modules"):
        deterministic_lora._canonicalize_adapter_config(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("raw", [b"{", b"[]", b"null", b'{"r": 32}', b'{"target_modules": ["q_proj"], "r": NaN}'])
def test_lora_config_serialization_helper_rejects_malformed_config_without_rewriting(tmp_path, raw):
    path = tmp_path / "adapter_config.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, TypeError, KeyError)):
        deterministic_lora._canonicalize_adapter_config(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_dora", True),
        ("rank_pattern", {"q_proj": 4}),
        ("peft_type", "LORA"),
        ("r", True),
        ("lora_alpha", -1),
        ("lora_dropout", float("nan")),
        ("bias", "all"),
        ("target_modules", ["q_proj", 1]),
        ("target_parameters", ["experts.up_proj", ""]),
    ],
)
def test_lora_config_serialization_materializer_preserves_input_rejection(
    lora_config_serialization_control, field, value
):
    control = lora_config_serialization_control
    control.cfg.training["peft_config"][field] = value
    before = copy.deepcopy(control.cfg.training)
    with pytest.raises(ValueError):
        deterministic_lora.materialize_adapter(control.cfg, control.source, control.scratch)
    assert control.saved == []
    assert not control.scratch.exists()
    assert control.cfg.training == before


@pytest.mark.parametrize("mutation", ["target_order", "scalar", "parameter_order", "tensor"])
def test_lora_config_serialization_pack_identity_still_rejects_postpublication_mutation(
    lora_config_serialization_control, mutation
):
    control = lora_config_serialization_control
    control.body["target_parameters"] = ["experts.up_proj", "experts.gate_proj"]
    path = deterministic_lora.materialize_adapter(control.cfg, control.source, control.scratch)
    deterministic_lora.identity_metadata(path)
    if mutation == "tensor":
        tensors = load_file(str(path / "adapter_model.safetensors"))
        tensors[sorted(tensors)[0]][0, 0] += 1
        save_file(tensors, str(path / "adapter_model.safetensors"))
        expected = "initialization bytes differ"
    else:
        config_path = path / "adapter_config.json"
        body = json.loads(config_path.read_bytes())
        if mutation == "target_order":
            body["target_modules"].reverse()
        elif mutation == "scalar":
            body["lora_alpha"] += 1
        else:
            body["target_parameters"].reverse()
        config_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        expected = "adapter configuration changed"
    with pytest.raises(ValueError, match=expected):
        deterministic_lora.identity_metadata(path)


# Native Ray spawn serialization: actual worker lookup in copied function globals.

_native_spawn_globals_control: SimpleNamespace
_DeepSpeedWorker: type


async def _native_spawn_globals_template(self, cfg, ranks, address, port, source_stage_info):
    control = _native_spawn_globals_control
    assert cfg is control.config and ranks is control.ranks and source_stage_info is control.source_stage
    assert address == "owned-master" and port == 12345
    control.events.append("native-spawn")
    selected = _DeepSpeedWorker
    control.selected.append(selected)
    self._workers = [control.construct(selected, cfg, source_stage_info)]
    if control.drift == "globals":
        control.spawn_globals["_DeepSpeedWorker"] = control.foreign
    if control.spawn_failure is not None:
        raise control.spawn_failure
    return control.spawn_result


@pytest.fixture
def native_spawn_globals_control(native_original_adapter, monkeypatch):
    from types import FunctionType

    from arctic_platform.correctness.harness import native_adapter
    from arctic_platform.correctness.harness import native_lora

    adapter = native_original_adapter
    training = adapter.training
    training.__name__ = "dss.ray_dss.jobs.gpu.training"
    request, receipts = _native_original_composition_receipts(adapter)
    request.update(
        schema=native_lora.SCHEMA,
        path=str(adapter.path),
        model_name=adapter.metadata["model_name"],
        training_gpus=1,
    )
    config_body = {
        "job_type": "training",
        "model_name": request["model_name"],
        "training_config": {"n_gpus": 1, "debug": {native_lora.REQUEST_KEY: request}},
    }
    control = SimpleNamespace(
        adapter=adapter,
        training=training,
        request=request,
        expected_receipts=receipts[:1],
        config={"job_id": "selected-child", "unchanged-native-option": object()},
        ranks=[("owned-host", 0)],
        source_stage=object(),
        ref=object(),
        events=[],
        selected=[],
        workers=[],
        gets=[],
        spawn_result=object(),
        spawn_failure=None,
        constructor_failure=None,
        receipt_failure=None,
        cleanup_failure=None,
        drift=None,
        receipt_drift=False,
        ordinary=False,
        namespace="detached",
        supplier=None,
        foreign=object(),
        config_object=SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(config_body)),
    )
    original_load = native_adapter.initialize_adapter
    original_engine = adapter.engine

    def initialize(model, path, identity):
        assert model is adapter.model and identity == adapter.identity
        original_load(model, path, identity)
        control.events.append("loaded-original")

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        control.events.append("native-engine")
        return original_engine(
            model, ds_config, optimizer=optimizer, ep_size=ep_size, mpu=mpu, model_path=model_path, rank=rank
        )

    adapter.engine = training.initialize_training_engine = engine
    monkeypatch.setattr(native_adapter, "initialize_adapter", initialize)

    class NativeWorker:
        def __init__(self, cfg, source_stage_info=None):
            assert cfg is control.config and source_stage_info is control.source_stage
            control.workers.append(self)
            control.events.append("native-constructor")
            if control.ordinary:
                assert training.initialize_training_engine is engine
                return
            assert training.initialize_training_engine is not engine
            assert (
                globals()["initialize_training_engine"](
                    adapter.model,
                    adapter.ds_config,
                    optimizer=adapter.optimizer,
                    ep_size=4,
                    mpu=adapter.mpu,
                    model_path="unchanged-model-path",
                    rank=0,
                )
                is adapter.result
            )
            if control.constructor_failure is not None:
                raise control.constructor_failure

    def restored():
        assert training._DeepSpeedWorker is NativeWorker
        assert control.spawn_globals["_DeepSpeedWorker"] is control.spawn_worker_original
        _native_original_adapter_restored(adapter)

    def construct(selected, cfg, source_stage_info):
        assert issubclass(selected, NativeWorker)
        if control.ordinary:
            assert selected is NativeWorker
        else:
            assert selected is not NativeWorker
            assert training._DeepSpeedWorker is (selected if control.namespace == "shared" else NativeWorker)
        # A Ray worker imports its own ordinary training module before subclass construction.
        coordinator_binding = training._DeepSpeedWorker
        training._DeepSpeedWorker = NativeWorker
        try:
            worker = selected(cfg, source_stage_info)
        finally:
            training._DeepSpeedWorker = coordinator_binding
        if control.ordinary:
            assert not hasattr(worker, "original_adapter_receipt")
            return SimpleNamespace(native_worker=worker)

        async def receipt():
            restored()
            control.events.append("remote-receipt")
            if control.receipt_failure is not None:
                raise control.receipt_failure
            reported = worker.original_adapter_receipt()
            if control.receipt_drift:
                reported["initialization"]["verified_tensors"][0]["sha256"] = "0" * 64
            return reported

        return SimpleNamespace(original_adapter_receipt=SimpleNamespace(remote=receipt))

    control.construct = construct
    control.restored = restored
    constructor = NativeWorker.__init__
    bound_constructor = FunctionType(
        constructor.__code__,
        training.__dict__,
        constructor.__name__,
        constructor.__defaults__,
        constructor.__closure__,
    )
    bound_constructor.__qualname__ = constructor.__qualname__
    NativeWorker.__init__ = bound_constructor
    training._DeepSpeedWorker = NativeWorker
    control.worker_base = NativeWorker
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.jobs.gpu.training": training})
    ray = ModuleType("ray")

    def get(ref):
        assert ref is control.ref
        control.gets.append(ref)
        return adapter.files

    def remote(**options):
        assert options == {"num_cpus": 1, "num_gpus": 0}
        return lambda cls: cls

    ray.get = get
    ray.remote = remote
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(subprocess, "Popen", _e2e_reference_records_forbidden)
    monkeypatch.setattr(torch.cuda, "init", _e2e_reference_records_forbidden)
    import socket

    monkeypatch.setattr(socket, "create_connection", _e2e_reference_records_forbidden)

    def coordinator(namespace="detached"):
        control.namespace = namespace
        spawn_globals = dict(training.__dict__) if namespace == "detached" else training.__dict__
        if control.supplier is not None:
            assert namespace == "detached"
            spawn_globals["_DeepSpeedWorker"] = control.supplier
        spawn_globals["_native_spawn_globals_control"] = control
        template = _native_spawn_globals_template
        spawn = FunctionType(
            template.__code__, spawn_globals, "_spawn_workers", template.__defaults__, template.__closure__
        )
        assert spawn.__module__ == training.__name__
        assert (spawn.__globals__ is training.__dict__) == (namespace == "shared")
        assert "_DeepSpeedWorker" in spawn.__code__.co_names
        control.spawn_globals = spawn_globals
        control.spawn_worker_original = spawn_globals["_DeepSpeedWorker"]
        control.spawn = spawn

        class NativeCoordinator:
            def __init__(self):
                self._workers = []
                self._info = SimpleNamespace(config=control.config_object, job_id="selected-child")

            async def _kill_workers(self, hosts):
                assert hosts == ["owned-host"]
                restored()
                control.events.append("owned-kill")
                if control.cleanup_failure is not None:
                    raise control.cleanup_failure
                self._workers = []

            def info(self):
                return {"native_info": "preserved"}

        NativeCoordinator._spawn_workers = spawn
        training.GPUTrainingJob = SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=NativeCoordinator))
        request["sources"] = native_lora._runtime_sources(training)
        control.coordinator_base = NativeCoordinator
        return native_lora._coordinator_class(training, request, control.ref)()

    control.coordinator = coordinator
    return control


@pytest.mark.parametrize("namespace", ["detached", "shared"])
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "spawn-runtime",
        "spawn-interrupt",
        "constructor-runtime",
        "constructor-interrupt",
        "receipt-runtime",
        "receipt-drift",
        "spawn-cleanup-runtime",
        "spawn-cleanup-interrupt",
        "receipt-cleanup-runtime",
        "globals-drift",
    ],
)
def test_native_spawn_globals_selected_subclass_preengine_receipt_and_all_exit_restoration(
    native_spawn_globals_control, namespace, outcome, capsys
):
    from arctic_platform.correctness.harness import native_lora

    control = native_spawn_globals_control
    if outcome.startswith("spawn"):
        error_type = KeyboardInterrupt if outcome == "spawn-interrupt" else RuntimeError
        control.spawn_failure = error_type("copied-globals native spawn failed")
    elif outcome.startswith("constructor"):
        error_type = KeyboardInterrupt if outcome == "constructor-interrupt" else RuntimeError
        control.constructor_failure = error_type("copied-globals native constructor failed")
    elif outcome in {"receipt-runtime", "receipt-cleanup-runtime"}:
        control.receipt_failure = RuntimeError("copied-globals remote receipt failed")
    elif outcome == "receipt-drift":
        control.receipt_drift = True
    elif outcome.endswith("-drift"):
        control.drift = outcome.split("-")[0]
    if "cleanup" in outcome:
        error_type = KeyboardInterrupt if outcome == "spawn-cleanup-interrupt" else RuntimeError
        control.cleanup_failure = error_type("copied-globals owned cleanup failed")
    coordinator = control.coordinator(namespace)
    original_module = dict(control.training.__dict__)
    original_globals = dict(control.spawn_globals)
    failure = None
    try:
        asyncio.run(
            coordinator._spawn_workers(control.config, control.ranks, "owned-master", 12345, control.source_stage)
        )
    except BaseException as error:
        failure = error
    control.restored()
    assert control.training.__dict__ == original_module
    assert control.spawn.__globals__ == original_globals
    assert len(control.selected) == len(control.workers) == 1
    assert type(control.workers[0]) is control.selected[0]
    assert control.gets == [control.ref]
    assert len(control.adapter.calls) == 1
    expected_events = ["native-spawn", "native-constructor", "loaded-original", "native-engine"]
    if outcome in {"success", "receipt-runtime", "receipt-drift", "receipt-cleanup-runtime"}:
        expected_events.append("remote-receipt")
    if outcome != "success" and not outcome.startswith("constructor"):
        expected_events.append("owned-kill")
    assert control.events == expected_events
    output = capsys.readouterr().out
    if outcome == "success":
        assert failure is None
        assert coordinator._correctness_original_adapter_receipts == control.expected_receipts
        assert coordinator.info() == {
            "native_info": "preserved",
            "correctness_original_adapter": {
                "schema": native_lora.SCHEMA,
                "workers": control.expected_receipts,
            },
        }
        assert json.loads(output) == {
            "schema": native_lora.SCHEMA,
            "job_id": "selected-child",
            "workers": control.expected_receipts,
        }
        receipt = control.workers[0].original_adapter_receipt()
        receipt["rank"] = 99
        assert control.workers[0].original_adapter_receipt() == control.expected_receipts[0]
    else:
        primary = control.spawn_failure or control.constructor_failure or control.receipt_failure
        assert failure is primary if primary is not None else isinstance(failure, ValueError)
        assert output == "" and not hasattr(coordinator, "_correctness_original_adapter_receipts")
        if control.cleanup_failure is not None:
            assert any(
                f"{type(control.cleanup_failure).__name__}: {control.cleanup_failure}" in note
                for note in getattr(failure, "__notes__", [])
            )
        else:
            assert coordinator._workers == []
        if outcome.startswith("constructor"):
            assert not hasattr(control.workers[0], "_correctness_original_adapter_receipt")


@pytest.mark.parametrize("scope", ["globals", "module"])
@pytest.mark.parametrize("damage", ["foreign", "stale-subclass", "missing"])
def test_native_spawn_globals_rejects_preexisting_foreign_stale_or_missing_worker_before_spawn(
    native_spawn_globals_control, scope, damage
):
    control = native_spawn_globals_control
    coordinator = control.coordinator()
    bindings = control.spawn_globals if scope == "globals" else control.training.__dict__
    if damage == "missing":
        del bindings["_DeepSpeedWorker"]
    elif damage == "stale-subclass":
        bindings["_DeepSpeedWorker"] = type("StaleOriginalAdapterWorker", (control.worker_base,), {})
    else:
        bindings["_DeepSpeedWorker"] = control.foreign
    before_module = dict(control.training.__dict__)
    before_globals = dict(control.spawn_globals)
    with pytest.raises(ValueError):
        asyncio.run(
            coordinator._spawn_workers(control.config, control.ranks, "owned-master", 12345, control.source_stage)
        )
    assert control.events == control.selected == control.workers == control.gets == control.adapter.calls == []
    assert coordinator._workers == []
    assert control.training.__dict__ == before_module
    assert control.spawn.__globals__ == before_globals
    _native_original_adapter_restored(control.adapter)


@pytest.mark.parametrize("namespace", ["detached", "shared"])
def test_native_spawn_globals_ordinary_native_route_remains_unwrapped(native_spawn_globals_control, namespace, capsys):
    control = native_spawn_globals_control
    control.coordinator(namespace)
    control.ordinary = True
    coordinator = control.coordinator_base()
    before_module = dict(control.training.__dict__)
    before_globals = dict(control.spawn_globals)
    assert (
        asyncio.run(
            coordinator._spawn_workers(control.config, control.ranks, "owned-master", 12345, control.source_stage)
        )
        is control.spawn_result
    )
    control.restored()
    assert control.events == ["native-spawn", "native-constructor"]
    assert control.selected == [control.worker_base]
    assert control.gets == control.adapter.calls == []
    assert control.training.__dict__ == before_module and control.spawn.__globals__ == before_globals
    assert not hasattr(coordinator._workers[0], "original_adapter_receipt")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("namespace", ["detached", "shared"])
def test_native_spawn_globals_installed_manager_leaves_unselected_training_dispatch_ordinary(
    native_spawn_globals_control, monkeypatch, namespace
):
    from enum import Enum

    from arctic_platform.correctness.harness import native_lora

    control = native_spawn_globals_control
    control.coordinator(namespace)
    control.ordinary = True
    actor = control.training.GPUTrainingJob
    ordinary_config = SimpleNamespace(
        model_dump=lambda **kwargs: {
            "job_type": "training",
            "model_name": "ordinary-model",
            "training_config": {"n_gpus": 1},
        }
    )
    info = SimpleNamespace(config=ordinary_config, layout=[SimpleNamespace(gpu_ids=[0])])

    class ZoneType(Enum):
        TRAINING = "training"

    class Backend(Enum):
        GPU = "gpu"

    actor_key = (ZoneType.TRAINING, Backend.GPU)
    manager_module = ModuleType("dss.ray_dss.zones.manager")
    manager_module._ACTOR_CLS = {actor_key: actor}
    spawned = []

    class ZoneManager:
        zone_type = ZoneType.TRAINING
        backend = Backend.GPU

        def _spawn_actor(self, actual_info):
            assert actual_info is info and manager_module._ACTOR_CLS[actor_key] is actor
            coordinator = actor.__ray_metadata__.modified_class()
            spawned.append(coordinator)
            return asyncio.run(
                coordinator._spawn_workers(control.config, control.ranks, "owned-master", 12345, control.source_stage)
            )

    manager_module.ZoneManager = ZoneManager
    monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
    monkeypatch.setattr(sys.modules["dss.ray_dss.zones"], "manager", manager_module, raising=False)
    before_module = dict(control.training.__dict__)
    before_globals = dict(control.spawn_globals)
    manager = ZoneManager()
    native_lora.install_manager(manager)
    assert manager._spawn_actor(info) is control.spawn_result
    control.restored()
    assert control.events == ["native-spawn", "native-constructor"]
    assert control.selected == [control.worker_base]
    assert control.gets == control.adapter.calls == []
    assert not hasattr(spawned[0]._workers[0], "original_adapter_receipt")
    assert manager_module._ACTOR_CLS == {actor_key: actor}
    assert control.training.__dict__ == before_module and control.spawn.__globals__ == before_globals


@pytest.mark.parametrize("failure_type", [None, RuntimeError, KeyboardInterrupt])
def test_native_spawn_globals_authenticated_deserialized_supplier_restores_exact_class(
    native_spawn_globals_control, failure_type, capsys
):
    from types import FunctionType

    control = native_spawn_globals_control
    original = control.worker_base.__init__
    constructor = FunctionType(
        original.__code__,
        dict(original.__globals__),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    constructor.__qualname__ = original.__qualname__
    control.supplier = type(
        control.worker_base.__name__,
        (),
        {
            "__module__": control.worker_base.__module__,
            "__qualname__": control.worker_base.__qualname__,
            "__init__": constructor,
        },
    )
    assert control.supplier is not control.worker_base
    assert constructor is not original and constructor.__globals__ is not original.__globals__
    assert inspect.getsourcefile(constructor) == inspect.getsourcefile(original)
    control.spawn_failure = failure_type("serialized supplier native spawn failed") if failure_type else None
    coordinator = control.coordinator()
    failure = None
    try:
        asyncio.run(
            coordinator._spawn_workers(control.config, control.ranks, "owned-master", 12345, control.source_stage)
        )
    except BaseException as error:
        failure = error
    assert failure is control.spawn_failure
    control.restored()
    assert control.spawn.__globals__["_DeepSpeedWorker"] is control.supplier
    assert control.training._DeepSpeedWorker is control.worker_base
    assert control.selected[0] is not control.supplier
    assert issubclass(control.selected[0], control.worker_base)
    assert control.gets == [control.ref] and len(control.adapter.calls) == 1
    events = ["native-spawn", "native-constructor", "loaded-original", "native-engine"]
    if failure_type is None:
        events.append("remote-receipt")
        assert json.loads(capsys.readouterr().out)["workers"] == control.expected_receipts
    else:
        events.append("owned-kill")
        assert coordinator._workers == [] and capsys.readouterr().out == ""
    assert control.events == events


# Native-local first-two composition; target/reference calls are mocked.


@pytest.fixture
def native_first_two_control(_prepared_reference_runner_prepared_runner, tmp_path, monkeypatch):
    from arctic_platform.correctness.harness import cortex
    from arctic_platform.correctness.harness import cortex_optimizer

    case = _prepared_reference_runner_prepared_runner
    monkeypatch.setattr(storage, "_DATA_FAST_ROOT", tmp_path)
    peft_config = {"peft_type": "Lora", "r": 32, "target_modules": ["q_proj"]}
    case.cfg.training["peft_config"] = deepcopy(peft_config)
    adapter = tmp_path / "original-pack"
    adapter.mkdir()
    identity = {"schema": "mock-shared-original", "tensor_sha256": "a" * 64}
    backend = CortexConfig(base_url="http://127.0.0.1:8003", database="dev", schema="dev")
    generated = []
    bound = []
    targets = []
    fixtures = []

    def generate(model, config, destination, *, seed):
        assert model == case.spec.model.cache_path
        assert config == peft_config and seed == runner.SEED
        generated.append((model, deepcopy(config), Path(destination), seed))
        return adapter

    def identify(path, config=None):
        assert Path(path) == adapter
        assert config is None or config == peft_config
        return deepcopy(identity)

    def bind(payload, path, *, native_backend=None):
        assert Path(path) == adapter and native_backend is backend
        assert payload["training_config"]["peft_config"] == peft_config
        assert "shared_peft_adapter" not in payload["training_config"]["debug"]
        before = deepcopy(payload)
        bound.append((before, Path(path), native_backend, deepcopy(identity)))
        result = deepcopy(payload)
        result["training_config"]["debug"]["mock_native_original_adapter"] = deepcopy(identity)
        assert payload == before
        return result

    def fixture(actual_backend, spec):
        assert actual_backend is backend and spec is case.spec
        fixtures.append(True)
        return {"model_name": case.spec.model.cache_path}

    def target(session, job, body, **kwargs):
        targets.append((session, job, deepcopy(kwargs)))
        return SimpleNamespace(
            optimizer_state_manifest=str(case.target_manifest),
            avg_loss=2.0,
            grad_norms={"module.layer.weight": 1.0},
            model_calls=None,
        )

    monkeypatch.setattr(shared, "generate_adapter", generate)
    monkeypatch.setattr(shared, "adapter_identity", identify)
    monkeypatch.setattr(deterministic_lora, "materialize_adapter", _e2e_reference_records_forbidden)
    monkeypatch.setattr(deterministic_lora, "bind_payload", bind)
    monkeypatch.setattr(cortex, "load_local_fixture", fixture)
    monkeypatch.setattr(cortex, "load_fixture", lambda path, spec: {"model_name": case.spec.model.cache_path})
    monkeypatch.setattr(runner, "fwd_bwd_step", target)
    monkeypatch.setattr(cortex_optimizer, "capture_optimizer_step", target)

    @contextlib.contextmanager
    def running_job(session, payload):
        assert session == "mock-target-session"
        assert payload["model_name"] == case.spec.model.cache_path
        if "mock_native_original_adapter" in payload["training_config"]["debug"]:
            assert payload["training_config"]["debug"]["mock_native_original_adapter"] == identity
            assert "shared_peft_adapter" not in payload["training_config"]["debug"]
        else:
            assert payload["training_config"]["debug"]["shared_peft_adapter"] == {
                "path": str(adapter),
                "identity": identity,
            }
        case.events.append(("job-enter",))
        yield "mock-target-job"
        case.events.append(("job-exit",))

    monkeypatch.setattr(runner, "running_job", running_job)

    def execute(*, selected=None, supplied=False, **kwargs):
        options = {
            "client_backend": backend,
            "local_cortex_fixture": True,
            **({"peft_adapter": adapter} if supplied else {}),
            **kwargs,
        }
        return case.execute(
            selected=["single-step-grads"] if selected is None else selected,
            **options,
        )

    return SimpleNamespace(
        case=case,
        adapter=adapter,
        identity=identity,
        backend=backend,
        generated=generated,
        bound=bound,
        targets=targets,
        fixtures=fixtures,
        execute=execute,
    )


@pytest.mark.parametrize("test_id", ["single-step-grads", "single-step-optimizer"])
@pytest.mark.parametrize(
    "local,fixture,selected,accepted",
    [
        (True, True, True, True),
        (True, True, False, False),
        (True, False, True, False),
        (False, True, True, False),
    ],
)
def test_native_first_two_admission_requires_explicit_route_and_fixture(test_id, local, fixture, selected, accepted):
    test = SimpleNamespace(test_id=test_id, compares_to_reference=True)
    cfg = SimpleNamespace(config_id="original-lora", training={"peft_config": {"r": 32}})
    result = runner._cortex_rejection(
        test,
        cfg,
        {"model_name": "verified-fixture"} if fixture else None,
        local_cortex_fixture=local,
        native_adapter_initialization=selected,
    )
    assert (result is None) is accepted


@pytest.mark.parametrize(
    "selected",
    [
        ["single-step-grads"],
        ["single-step-optimizer"],
        ["single-step-grads", "single-step-optimizer"],
    ],
)
@pytest.mark.parametrize("supplied", [False, True])
def test_native_first_two_composition_keeps_exact_adapter_recipe_and_native_payload(
    native_first_two_control, selected, supplied
):
    control = native_first_two_control
    training = deepcopy(control.case.cfg.training)
    results = control.execute(selected=selected, supplied=supplied)
    assert len(results) == 2 * len(selected) and all(result.outcome is Outcome.PASS for result in results)
    assert len(control.generated) == (0 if supplied else 1)
    assert control.fixtures == [True]
    assert len(control.bound) == 3 and len(control.targets) == 2
    assert control.case.cfg.training == training
    assert all(row[1:] == (control.adapter, control.backend, control.identity) for row in control.bound)
    assert all(recipe["initialization"] == control.identity for _, _, recipe in control.case.recipes)
    assert [event[2] for event in control.case.events if event[0] == "record"] == [
        arm for arm in ("gas1", "gas4") for _ in selected
    ]


@pytest.mark.parametrize("supplied", [False, True])
def test_native_first_two_unsupported_binding_stops_before_records_or_target(
    native_first_two_control, monkeypatch, supplied
):
    control = native_first_two_control
    calls = []

    def reject(payload, path, *, native_backend):
        assert native_backend is control.backend and Path(path) == control.adapter
        calls.append(deepcopy(payload))
        raise ValueError("owned gateway shared-pack capability unavailable")

    monkeypatch.setattr(deterministic_lora, "bind_payload", reject)
    with pytest.raises(ValueError, match="shared-pack capability unavailable"):
        control.execute(supplied=supplied)
    assert len(calls) == 1 and not control.case.events and not control.targets
    assert not control.case.published


@pytest.mark.parametrize("supplied", [False, True])
def test_native_first_two_reference_preparation_never_binds_or_starts_target(
    native_first_two_control, monkeypatch, supplied
):
    control = native_first_two_control
    control.case.missing.update({("single-step-grads", "gas1"), ("single-step-grads", "gas4")})
    measurements = []

    def reference(model, batch, output, **kwargs):
        assert model == control.case.spec.model.cache_path
        assert kwargs["peft_adapter_path"] == control.adapter
        assert kwargs["peft_config"] == control.case.cfg.training["peft_config"]
        measurements.append((model, batch, deepcopy(kwargs)))
        return {**deepcopy(control.case.record), "active_tokens": load(batch).active_tokens}

    monkeypatch.setattr(deterministic_lora, "bind_payload", _e2e_reference_records_forbidden)
    monkeypatch.setattr(runner, "run_reference", reference)
    assert control.execute(supplied=supplied, prepare_references=True) == []
    assert len(measurements) == 2 and len(control.case.published) == 2
    assert len(control.generated) == (0 if supplied else 1)
    assert not control.bound and not control.targets and not control.fixtures
    assert not any(event[0].startswith(("job-", "gateway-")) for event in control.case.events)
    assert all(recipe["initialization"] == control.identity for _, _, recipe in control.case.recipes)


def test_native_first_two_published_nonlocal_route_stays_rejected(native_first_two_control):
    control = native_first_two_control
    results = control.execute(local_cortex_fixture=False, cortex_fixture=Path("mock-publication.json"))
    assert len(results) == 1 and results[0].outcome is Outcome.FAIL
    assert "product-initialized LoRA adapter" in results[0].reason
    assert not control.generated and not control.bound and not control.targets and not control.case.events


def test_native_first_two_supplied_nonlocal_route_fails_before_adapter_read(native_first_two_control):
    control = native_first_two_control
    with pytest.raises(ValueError, match="explicit local Cortex"):
        control.execute(supplied=True, local_cortex_fixture=False)
    assert not control.generated and not control.bound and not control.targets and not control.case.events


def test_native_first_two_onprem_preserves_legacy_shared_payload(native_first_two_control):
    from arctic_platform.client import OnPremConfig

    control = native_first_two_control
    results = control.execute(client_backend=OnPremConfig(), local_cortex_fixture=False)
    assert len(results) == 2 and all(result.outcome is Outcome.PASS for result in results)
    assert len(control.generated) == 1 and not control.bound and len(control.targets) == 2
    assert all(recipe["initialization"] == control.identity for _, _, recipe in control.case.recipes)


def test_native_first_two_adapter_mutation_after_reference_admission_cannot_rebind(
    native_first_two_control, monkeypatch
):
    control = native_first_two_control
    original_load = reference_records.load_record

    def changed(ctx, identity):
        record = original_load(ctx, identity)
        control.identity["tensor_sha256"] = "b" * 64
        return record

    monkeypatch.setattr(reference_records, "load_record", changed)
    results = control.execute()
    assert len(results) == 2 and all(result.outcome is Outcome.FAIL for result in results)
    assert all("shared adapter changed before native target binding" in result.reason for result in results)
    assert len(control.bound) == 1 and not control.targets


# Native shared-pack composition: real CPU PEFT factors with synthetic runtime boundaries.


def _native_shared_pack_engine_template(model, ds_config, *, rank):
    control = globals()["_native_shared_pack_control"]
    assert model is control.model and ds_config is control.ds_config
    assert type(rank) is int and rank == control.rank
    assert isinstance(model, peft.PeftModel)
    assert shared._tensor_identity(peft.get_peft_model_state_dict(model)) == control.identity["tensors"]
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu" and parameter.dtype == torch.float32
        if name in control.base:
            assert _bytes(parameter) == control.base[name]
    control.events.append(("stub-engine", rank))
    if control.engine_drift is not None:
        control.engine_drift()
    if control.engine_failure is not None:
        raise control.engine_failure
    return control.engine_result


def _native_shared_pack_constructor_template(self, cfg, source_stage_info=None):
    control = globals()["_native_shared_pack_control"]
    assert cfg is control.cfg and source_stage_info is control.source_stage
    control.events.append(("synthetic-constructor", control.rank))
    self.engine = globals()["initialize_training_engine"](control.model, control.ds_config, rank=control.rank)
    if control.constructor_failure is not None:
        raise control.constructor_failure


async def _native_shared_pack_spawn_template(self, cfg, ranks, master_addr, master_port, source_stage_info):
    control = globals()["_native_shared_pack_control"]
    assert cfg is control.cfg and ranks is control.ranks and source_stage_info is control.source_stage
    assert master_addr == "synthetic-master" and master_port == 12345
    control.events.append(("synthetic-spawn", len(ranks)))
    selected = _DeepSpeedWorker
    assert selected is not control.worker_base and issubclass(selected, control.worker_base)
    control.selected.append(selected)
    for rank, _ in enumerate(ranks):
        self._workers.append(control.construct(selected, rank))
    if control.spawn_drift:
        control.spawn_globals["_DeepSpeedWorker"] = object()
    if control.spawn_failure is not None:
        raise control.spawn_failure


@pytest.fixture
def native_shared_pack_control(generated_adapter_config, tmp_path, monkeypatch):
    import socket
    from types import FunctionType

    from arctic_platform.correctness.harness import native_lora

    source = generated_adapter_config
    monkeypatch.setenv(storage.CORRECTNESS_TMPDIR_ENV, str(tmp_path / "worker-scratch"))
    monkeypatch.setattr(subprocess, "Popen", _e2e_reference_records_forbidden)
    monkeypatch.setattr(socket.socket, "connect", _e2e_reference_records_forbidden)
    monkeypatch.setattr(socket, "create_connection", _e2e_reference_records_forbidden)
    monkeypatch.setattr(torch.cuda, "init", _e2e_reference_records_forbidden)
    monkeypatch.setattr(dist, "init_process_group", _e2e_reference_records_forbidden)
    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", _e2e_reference_records_forbidden)
    monkeypatch.setattr(Qwen3ForCausalLM, "forward", _e2e_reference_records_forbidden)
    monkeypatch.setattr(peft.PeftModel, "forward", _e2e_reference_records_forbidden)
    monkeypatch.setattr(transformers.GenerationMixin, "generate", _e2e_reference_records_forbidden)
    path = shared.generate_adapter(source.path, source.config, tmp_path / "seed42-pack", seed=42)
    identity = shared.adapter_identity(path, source.config)
    files = shared.adapter_files(path, identity)
    control = SimpleNamespace(
        native=native_lora,
        source=source,
        path=path,
        identity=identity,
        files=files,
        events=[],
        scratch=[],
        staged=[],
        cfg={"job_id": "synthetic-shared-child", "native-option": object()},
        ds_config={"native-engine-option": object()},
        source_stage=object(),
        engine_result=object(),
        engine_failure=None,
        engine_drift=None,
        constructor_failure=None,
        rank=0,
    )
    original_workdir = storage.correctness_workdir

    def workdir(prefix):
        directory = original_workdir(prefix)
        control.scratch.append(directory)
        return directory

    monkeypatch.setattr(storage, "correctness_workdir", workdir)
    original_load = shared.initialize_adapter

    def initialize(model, directory, expected):
        control.staged.append(Path(directory))
        assert {item.name: item.read_bytes() for item in Path(directory).iterdir()} == control.files
        original_load(model, directory, expected)
        control.events.append(("real-peft-load", control.rank))

    monkeypatch.setattr(shared, "initialize_adapter", initialize)

    def reset_model(rank=0):
        control.rank = rank
        control.model = _model()
        control.base = {
            name: _bytes(parameter)
            for name, parameter in control.model.named_parameters()
            if not parameter.requires_grad
        }
        with torch.no_grad():
            for parameter in _installed(control.model).values():
                parameter.fill_(9.0)
        assert shared._tensor_identity(peft.get_peft_model_state_dict(control.model)) != identity["tensors"]

    def namespace(mode="detached"):
        training = ModuleType("dss.ray_dss.jobs.gpu.training")
        training.__dict__.update(
            _native_shared_pack_control=control, peft=peft, shared=shared, torch=torch, _bytes=_bytes
        )
        template = _native_shared_pack_engine_template
        engine = FunctionType(template.__code__, training.__dict__, template.__name__)
        engine.__qualname__ = template.__qualname__
        training.initialize_training_engine = engine
        constructor_globals = dict(training.__dict__) if mode == "detached" else training.__dict__
        template = _native_shared_pack_constructor_template
        constructor = FunctionType(template.__code__, constructor_globals, "__init__", template.__defaults__)
        constructor.__qualname__ = "SyntheticNativeWorker.__init__"
        worker = type("SyntheticNativeWorker", (), {"__module__": training.__name__, "__init__": constructor})
        training._DeepSpeedWorker = worker
        constructor_globals["_DeepSpeedWorker"] = worker
        control.training = training
        control.constructor = constructor
        control.constructor_globals = constructor_globals
        control.worker_base = worker
        control.engine = engine
        control.namespace_mode = mode
        control.original_namespace = dict(constructor_globals)
        control.original_module = dict(training.__dict__)
        return constructor_globals

    def restored():
        assert control.constructor_globals == control.original_namespace
        assert control.training.__dict__ == control.original_module
        assert not shared._NATIVE_ADAPTER_SCOPE_LOCK.locked()
        assert all(not directory.exists() for directory in control.scratch)
        assert all(not directory.exists() for directory in control.staged)

    def scope(**kwargs):
        options = dict(job_id=control.cfg["job_id"], rank=control.rank)
        options.update(kwargs)
        return native_lora.native_shared_worker_adapter_scope(
            control.training, control.constructor, control.files, control.identity, **options
        )

    control.reset_model = reset_model
    control.namespace = namespace
    control.restored = restored
    control.scope = scope
    reset_model()
    namespace()
    return control


@pytest.mark.parametrize("namespace", ["detached", "shared"])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_native_shared_pack_real_seed42_fp32_load_precedes_stub_engine_and_restores_globals(
    native_shared_pack_control, namespace, rank
):
    control = native_shared_pack_control
    control.reset_model(rank)
    globals_ = control.namespace(namespace)
    assert (globals_ is control.training.__dict__) == (namespace == "shared")
    assert control.identity["generation"] == {
        "schema": "upstream-default-lora-v1",
        "seed": 42,
        "initializer": True,
        "peft_config": control.identity["peft_config"],
        "dependencies": {
            "torch": str(torch.__version__),
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
    }
    assert not (control.path / "initialization.json").exists()
    assert len(control.identity["tensors"]) == 4
    assert all(entry["dtype"] == "torch.float32" for entry in control.identity["tensors"])
    for name, tensor in load_file(str(control.path / "adapter_model.safetensors")).items():
        assert bool(torch.count_nonzero(tensor)) == (".lora_A." in name)
    with control.scope() as receipt:
        assert globals_["initialize_training_engine"] is not control.engine
        if namespace == "detached":
            assert control.training.initialize_training_engine is control.engine
        worker = control.worker_base(control.cfg, control.source_stage)
        assert worker.engine is control.engine_result
    assert receipt == {
        "schema": "shared-peft-worker-initialization-v1",
        "job_id": control.cfg["job_id"],
        "rank": rank,
        "engine_initializations": 1,
        "shared_identity": control.identity,
    }
    assert control.events == [("synthetic-constructor", rank), ("real-peft-load", rank), ("stub-engine", rank)]
    control.restored()


@pytest.mark.parametrize("namespace", ["detached", "shared"])
@pytest.mark.parametrize(
    "failure",
    [
        "missing-init",
        "duplicate-init",
        "wrong-rank",
        "boolean-rank",
        "engine-runtime",
        "engine-interrupt",
        "constructor-runtime",
        "constructor-interrupt",
        "swallowed-engine",
        "receipt-drift",
        "binding-drift",
        "pack-before-engine",
        "pack-after-engine",
        "pack-after-load",
    ],
)
def test_native_shared_pack_worker_failure_restores_exact_binding_and_rejects_receipt(
    native_shared_pack_control, namespace, failure, monkeypatch
):
    control = native_shared_pack_control
    globals_ = control.namespace(namespace)
    primary = RuntimeError("synthetic primary")
    error = ValueError
    match = "native shared adapter"
    if failure.startswith("engine-") or failure == "swallowed-engine":
        primary = KeyboardInterrupt("synthetic primary") if failure == "engine-interrupt" else primary
        control.engine_failure = primary
    if failure.startswith("constructor-"):
        primary = KeyboardInterrupt("synthetic primary") if failure == "constructor-interrupt" else primary
        control.constructor_failure = primary
    if failure in {"engine-runtime", "engine-interrupt", "constructor-runtime", "constructor-interrupt"}:
        error, match = type(primary), "synthetic primary"

    def drift():
        recipe = control.staged[-1] / "correctness_recipe.json"
        changed = json.loads(recipe.read_bytes())
        changed["seed"] = 43
        recipe.write_text(json.dumps(changed))

    if failure == "pack-after-engine":
        control.engine_drift = drift
    if failure == "pack-after-load":
        original = shared.initialize_adapter

        def load(*args):
            original(*args)
            drift()

        monkeypatch.setattr(shared, "initialize_adapter", load)
    with pytest.raises(error, match=match) as caught:
        with control.scope() as receipt:
            wrapped = globals_["initialize_training_engine"]
            if failure == "missing-init":
                pass
            elif failure == "binding-drift":
                globals_["initialize_training_engine"] = object()
                wrapped(control.model, control.ds_config, rank=0)
            elif failure == "pack-before-engine":
                directory = next(control.scratch[-1].iterdir())
                recipe = directory / "correctness_recipe.json"
                recipe.write_text(json.dumps({**json.loads(recipe.read_bytes()), "seed": 43}))
                wrapped(control.model, control.ds_config, rank=0)
            elif failure in {"wrong-rank", "boolean-rank"}:
                wrapped(control.model, control.ds_config, rank=1 if failure == "wrong-rank" else False)
            elif failure == "swallowed-engine":
                with pytest.raises(RuntimeError) as swallowed:
                    control.worker_base(control.cfg, control.source_stage)
                assert swallowed.value is primary
            else:
                control.worker_base(control.cfg, control.source_stage)
                if failure == "duplicate-init":
                    wrapped(control.model, control.ds_config, rank=0)
                elif failure == "receipt-drift":
                    receipt["engine_initializations"] = 2
    if error is not ValueError:
        assert caught.value is primary
    if failure in {"missing-init", "wrong-rank", "boolean-rank", "binding-drift", "pack-before-engine"}:
        assert not any(event[0] in {"real-peft-load", "stub-engine"} for event in control.events)
    control.restored()


@pytest.mark.parametrize(
    "damage",
    [
        "namespace",
        "module",
        "qualname",
        "code",
        "source",
        "missing-engine",
        "foreign-engine",
        "engine-module",
        "engine-qualname",
        "engine-code",
        "engine-source",
        "prewrapped-engine",
    ],
)
def test_native_shared_pack_constructor_namespace_code_source_and_engine_guards(native_shared_pack_control, damage):
    from types import FunctionType

    control = native_shared_pack_control
    namespace = control.constructor_globals
    original_constructor = control.constructor
    constructor = FunctionType(original_constructor.__code__, namespace, "__init__", original_constructor.__defaults__)
    constructor.__qualname__ = original_constructor.__qualname__
    if damage == "namespace":
        namespace["__name__"] = "foreign.training"
    elif damage == "module":
        constructor.__module__ = "foreign.training"
    elif damage == "qualname":
        constructor.__qualname__ = "Foreign.__init__"
    elif damage == "code":
        constructor.__code__ = constructor.__code__.replace(co_name="changed_constructor")
    elif damage == "source":
        constructor.__code__ = constructor.__code__.replace(co_filename=shared.__file__)
    elif damage == "missing-engine":
        del namespace["initialize_training_engine"]
    elif damage == "foreign-engine":
        namespace["initialize_training_engine"] = object()
    else:
        engine = FunctionType(control.engine.__code__, namespace, control.engine.__name__)
        engine.__qualname__ = control.engine.__qualname__
        namespace["initialize_training_engine"] = engine
        if damage == "engine-module":
            engine.__module__ = "foreign.training"
        elif damage == "engine-qualname":
            engine.__qualname__ = "foreign_engine"
        elif damage == "engine-code":
            engine.__code__ = engine.__code__.replace(co_name="changed_engine")
        elif damage == "engine-source":
            engine.__code__ = engine.__code__.replace(co_filename=shared.__file__)
        else:
            engine._correctness_shared_native_adapter = True
    before = dict(namespace)
    with pytest.raises(ValueError, match="constructor (namespace|engine binding) differs"):
        with control.native.native_shared_worker_adapter_scope(
            control.training, constructor, control.files, control.identity, job_id="child", rank=0
        ):
            pytest.fail("mismatched constructor must not enter the scope")
    assert namespace == before and not control.events and not control.scratch
    assert not shared._NATIVE_ADAPTER_SCOPE_LOCK.locked()


@pytest.mark.parametrize("job_id,rank", [("", 0), (" ", 0), (None, 0), ("child", -1), ("child", True), ("child", "0")])
def test_native_shared_pack_worker_requires_explicit_job_and_integer_rank(native_shared_pack_control, job_id, rank):
    control = native_shared_pack_control
    with pytest.raises(ValueError, match="explicit job identity"):
        with control.scope(job_id=job_id, rank=rank):
            pytest.fail("invalid worker identity entered scope")
    assert not control.scratch and not control.events
    control.restored()


@pytest.mark.parametrize(
    "damage",
    [
        "missing-config",
        "missing-tensors",
        "missing-recipe",
        "extra-original",
        "extra-path",
        "nonbytes",
        "config-drift",
        "tensor-drift",
        "recipe-drift",
    ],
)
def test_native_shared_pack_worker_transfer_requires_exclusive_authenticated_files(native_shared_pack_control, damage):
    control = native_shared_pack_control
    files = dict(control.files)
    if damage.startswith("missing-"):
        name = {
            "missing-config": "adapter_config.json",
            "missing-tensors": "adapter_model.safetensors",
            "missing-recipe": "correctness_recipe.json",
        }[damage]
        del files[name]
    elif damage == "extra-original":
        files["initialization.json"] = b"{}"
    elif damage == "extra-path":
        files["../escaped.json"] = b"{}"
    elif damage == "nonbytes":
        files["adapter_config.json"] = bytearray(files["adapter_config.json"])
    elif damage == "config-drift":
        files["adapter_config.json"] += b"\n"
    elif damage == "tensor-drift":
        state = load_file(str(control.path / "adapter_model.safetensors"))
        state[next(iter(state))] = state[next(iter(state))] + 1
        files["adapter_model.safetensors"] = save(state)
    else:
        recipe = json.loads(files["correctness_recipe.json"])
        files["correctness_recipe.json"] = json.dumps({**recipe, "seed": 43}).encode()
    with pytest.raises(ValueError, match="transfer|identity changed"):
        with control.native.native_shared_worker_adapter_scope(
            control.training, control.constructor, files, control.identity, job_id="child", rank=0
        ):
            pytest.fail("invalid transferred pack entered scope")
    assert not control.events
    control.restored()


@pytest.mark.parametrize(
    "key", ["schema", "peft_config", "adapter_config_sha256", "adapter_model_sha256", "tensors", "generation"]
)
def test_native_shared_pack_worker_authenticates_entire_identity(native_shared_pack_control, key):
    control = native_shared_pack_control
    identity = copy.deepcopy(control.identity)
    del identity[key]
    with pytest.raises(ValueError, match="generation recipe|identity changed"):
        with control.native.native_shared_worker_adapter_scope(
            control.training, control.constructor, control.files, identity, job_id="child", rank=0
        ):
            pytest.fail("incomplete identity entered scope")
    assert not control.events
    control.restored()


def test_native_shared_pack_scope_is_nonreentrant_and_snapshots_inputs(native_shared_pack_control):
    from types import FunctionType

    control = native_shared_pack_control
    before_identity, before_files = copy.deepcopy(control.identity), dict(control.files)
    nested = FunctionType(
        control.constructor.__code__, dict(control.original_namespace), "__init__", control.constructor.__defaults__
    )
    nested.__qualname__ = control.constructor.__qualname__
    with control.scope() as receipt:
        with pytest.raises(ValueError, match="already active"):
            with control.native.native_shared_worker_adapter_scope(
                control.training, nested, control.files, control.identity, job_id="other-child", rank=0
            ):
                pytest.fail("nested scope entered")
        control.worker_base(control.cfg, control.source_stage)
        control.identity["generation"]["seed"] = 99
        control.files["correctness_recipe.json"] = b"{}"
    assert receipt["shared_identity"] == before_identity
    control.identity.clear()
    control.identity.update(before_identity)
    control.files.clear()
    control.files.update(before_files)
    control.restored()


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup", ["temporary", "scratch", "both"])
@pytest.mark.parametrize("primary_present", [True, False])
def test_native_shared_pack_staging_cleanup_preserves_primary_and_restores_namespace_and_lock(
    native_shared_pack_control, monkeypatch, failure_type, cleanup, primary_present
):
    control = native_shared_pack_control
    primary = failure_type("primary engine failure")
    cleanup_errors = []
    original_temporary = control.native.tempfile.TemporaryDirectory
    original_rmdir = Path.rmdir

    class Temporary:
        def __init__(self, *args, **kwargs):
            self.delegate = original_temporary(*args, **kwargs)
            self.name = self.delegate.name

        def cleanup(self):
            self.delegate.cleanup()
            if cleanup in {"temporary", "both"}:
                error = failure_type("temporary cleanup failure")
                cleanup_errors.append(error)
                raise error

    def rmdir(path):
        original_rmdir(path)
        if path in control.scratch and cleanup in {"scratch", "both"}:
            error = failure_type("scratch cleanup failure")
            cleanup_errors.append(error)
            raise error

    monkeypatch.setattr(control.native.tempfile, "TemporaryDirectory", Temporary)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    if primary_present:
        control.engine_failure = primary
    with pytest.raises(failure_type) as caught:
        with control.scope():
            control.worker_base(control.cfg, control.source_stage)
    assert caught.value is (primary if primary_present else cleanup_errors[0])
    notes = getattr(caught.value, "__notes__", [])
    assert len(notes) == len(cleanup_errors) - int(not primary_present)
    assert all("Shared adapter scratch cleanup also failed" in note for note in notes)
    control.restored()


@pytest.fixture
def native_shared_pack_capability(native_shared_pack_control, monkeypatch):
    import requests

    control = native_shared_pack_control
    pins = control.native._shared_runtime_sources(control.training)
    capability = {"schema": control.native.SHARED_SCHEMA, "sources": pins}
    response = SimpleNamespace(
        status_code=200,
        document={
            control.native.SHARED_OPENAPI_KEY: capability,
            control.native.OPENAPI_KEY: {"schema": control.native.SCHEMA, "sources": copy.deepcopy(pins)},
        },
    )
    gets = []

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert self.trust_env is False
            gets.append((url, kwargs))
            return SimpleNamespace(status_code=response.status_code, json=lambda: copy.deepcopy(response.document))

        def post(self, *args, **kwargs):
            pytest.fail("admission must not create a target job")

    monkeypatch.setattr(requests, "Session", Session)
    payload = {
        "job_type": "training",
        "model_name": "synthetic-config-only-qwen3",
        "training_config": {
            "peft_config": copy.deepcopy(control.source.config),
            "n_gpus": 3,
            "debug": {"unrelated-debug-option": True},
        },
        "unchanged-option": {"keep": "exact"},
    }
    return SimpleNamespace(
        control=control,
        pins=pins,
        capability=capability,
        response=response,
        gets=gets,
        payload=payload,
        backend=SimpleNamespace(base_url="http://127.0.0.1:8000", host=None, pat=None),
    )


def test_native_shared_pack_bind_and_request_preserve_pack_payload_and_distinct_schema(native_shared_pack_capability):
    case = native_shared_pack_capability
    native = case.control.native
    before = copy.deepcopy(case.payload)
    bound = native.bind_shared_adapter(case.payload, case.control.path, case.backend)
    assert case.payload == before and bound is not case.payload
    assert bound["training_config"] is not case.payload["training_config"]
    assert bound["training_config"]["debug"]["unrelated-debug-option"] is True
    assert native.REQUEST_KEY not in bound["training_config"]["debug"]
    assert "shared_peft_adapter" not in bound["training_config"]["debug"]
    expected = {
        "schema": native.SHARED_SCHEMA,
        "path": str(case.control.path),
        "identity": case.control.identity,
        "sources": case.pins,
        "training_gpus": 3,
        "model_name": case.payload["model_name"],
    }
    assert bound["training_config"]["debug"][native.SHARED_REQUEST_KEY] == expected
    config = SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(bound))
    received = native._shared_request(config)
    assert received == expected and received is not expected
    received["identity"]["generation"]["seed"] = 99
    assert native._shared_request(config) == expected
    assert native._request(config) is None
    assert case.gets == [("http://127.0.0.1:8000/openapi.json", {"timeout": 30, "allow_redirects": False})]
    assert shared.adapter_files(case.control.path, case.control.identity) == case.control.files
    assert not (case.control.path / "initialization.json").exists()
    assert not case.control.events


@pytest.mark.parametrize(
    "damage",
    [
        "external",
        "https",
        "authenticated",
        "decorated",
        "redirect",
        "missing-shared",
        "original-only",
        "original-schema",
        "extra-key",
        "missing-source",
        "extra-source",
        "relative-path",
        "short-hash",
        "source-extra-key",
        "source-not-dict",
    ],
)
def test_native_shared_pack_capability_admission_is_local_explicit_and_source_complete(
    native_shared_pack_capability, damage
):
    case = native_shared_pack_capability
    native = case.control.native
    if damage == "external":
        case.backend.base_url = "http://external.invalid:8000"
    elif damage == "https":
        case.backend.base_url = "https://127.0.0.1:8000"
    elif damage == "authenticated":
        case.backend.pat = "synthetic-not-a-credential"
    elif damage == "decorated":
        case.backend.base_url += "/extra?query=value"
    elif damage == "redirect":
        case.response.status_code = 302
    elif damage in {"missing-shared", "original-only"}:
        del case.response.document[native.SHARED_OPENAPI_KEY]
    elif damage == "original-schema":
        case.capability["schema"] = native.SCHEMA
    elif damage == "extra-key":
        case.capability["unexpected"] = True
    elif damage == "missing-source":
        del case.pins["adapter"]
    elif damage == "extra-source":
        case.pins["unexpected"] = case.pins["engine"]
    elif damage == "relative-path":
        case.pins["engine"]["path"] = "relative.py"
    elif damage == "short-hash":
        case.pins["engine"]["sha256"] = "bad"
    elif damage == "source-extra-key":
        case.pins["engine"]["unexpected"] = True
    else:
        case.pins["engine"] = None
    before = copy.deepcopy(case.payload)
    with pytest.raises(ValueError):
        native.bind_shared_adapter(case.payload, case.control.path, case.backend)
    assert case.payload == before and not case.control.events
    assert len(case.gets) == int(damage not in {"external", "https", "authenticated", "decorated"})


@pytest.mark.parametrize(
    "key", ["correctness_original_peft_adapter", "correctness_shared_peft_adapter", "shared_peft_adapter"]
)
def test_native_shared_pack_sender_never_overwrites_another_initialization(native_shared_pack_capability, key):
    case = native_shared_pack_capability
    case.payload["training_config"]["debug"][key] = {}
    before = copy.deepcopy(case.payload)
    with pytest.raises(ValueError, match="cannot replace another initialization"):
        case.control.native.bind_shared_adapter(case.payload, case.control.path, case.backend)
    assert case.payload == before and not case.control.events


def test_native_shared_pack_sender_rejects_deterministic_metadata(native_shared_pack_capability):
    case = native_shared_pack_capability
    (case.control.path / "initialization.json").write_text("{}")
    with pytest.raises(ValueError, match="deterministic initialization metadata"):
        case.control.native.bind_shared_adapter(case.payload, case.control.path, case.backend)
    assert not case.control.events


@pytest.mark.parametrize(
    "damage",
    [
        "missing-schema",
        "missing-seed",
        "missing-initializer",
        "missing-peft_config",
        "missing-dependencies",
        "schema",
        "seed-bool",
        "seed-negative",
        "seed-string",
        "initializer-false",
        "initializer-string",
        "topology",
        "extra-key",
        "dependencies-not-dict",
        "missing-torch",
        "missing-peft",
        "missing-transformers",
        "missing-safetensors",
        "extra-dependency",
        "empty-dependency",
        "nonstring-dependency",
    ],
)
def test_native_shared_pack_generation_recipe_requires_every_authenticated_field(
    native_shared_pack_capability, damage
):
    case = native_shared_pack_capability
    recipe_path = case.control.path / "correctness_recipe.json"
    recipe = json.loads(recipe_path.read_bytes())
    if damage.startswith("missing-"):
        key = damage[len("missing-") :]
        del (recipe["dependencies"] if key in {"torch", "peft", "transformers", "safetensors"} else recipe)[key]
    elif damage == "schema":
        recipe["schema"] = "original-peft-initialization-v1"
    elif damage.startswith("seed-"):
        recipe["seed"] = {"seed-bool": True, "seed-negative": -1, "seed-string": "42"}[damage]
    elif damage == "initializer-false":
        recipe["initializer"] = False
    elif damage == "initializer-string":
        recipe["initializer"] = "gaussian"
    elif damage == "topology":
        recipe["peft_config"]["r"] = 3
    elif damage == "extra-key":
        recipe["unexpected"] = True
    elif damage == "dependencies-not-dict":
        recipe["dependencies"] = []
    elif damage == "extra-dependency":
        recipe["dependencies"]["unexpected"] = "v1"
    elif damage == "empty-dependency":
        recipe["dependencies"]["peft"] = ""
    else:
        recipe["dependencies"]["peft"] = 1
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="authenticated upstream-default generation recipe"):
        case.control.native.bind_shared_adapter(case.payload, case.control.path, case.backend)
    assert not case.control.events


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "extra-key",
        "missing-schema",
        "missing-path",
        "missing-identity",
        "missing-sources",
        "missing-training_gpus",
        "missing-model_name",
        "job-type",
        "model-name",
        "gpu-zero",
        "gpu-bool",
        "gpu-mismatch",
        "source-checkpoint",
        "resume-checkpoint",
        "resume-tag",
        "original-conflict",
        "legacy-conflict",
        "outside-storage",
        "identity-recipe",
    ],
)
def test_native_shared_pack_request_requires_fresh_exact_job_and_exclusive_schema(
    native_shared_pack_capability, damage
):
    case = native_shared_pack_capability
    native = case.control.native
    bound = native.bind_shared_adapter(case.payload, case.control.path, case.backend)
    request = bound["training_config"]["debug"][native.SHARED_REQUEST_KEY]
    if damage.startswith("missing-"):
        del request[damage[len("missing-") :]]
    elif damage == "schema":
        request["schema"] = native.SCHEMA
    elif damage == "extra-key":
        request["original_metadata"] = {}
    elif damage == "job-type":
        bound["job_type"] = "inference"
    elif damage == "model-name":
        bound["model_name"] = "other-model"
    elif damage.startswith("gpu-"):
        request["training_gpus"] = {"gpu-zero": 0, "gpu-bool": True, "gpu-mismatch": 2}[damage]
    elif damage == "source-checkpoint":
        bound["source_checkpoint_info"] = {}
    elif damage == "resume-checkpoint":
        bound["resume_from_checkpoint"] = "other"
    elif damage == "resume-tag":
        bound["resume_checkpoint_tag"] = "other"
    elif damage == "original-conflict":
        bound["training_config"]["debug"][native.REQUEST_KEY] = {}
    elif damage == "legacy-conflict":
        bound["training_config"]["debug"]["shared_peft_adapter"] = {}
    elif damage == "outside-storage":
        request["path"] = "/tmp/not-admitted"
    else:
        request["identity"]["generation"]["initializer"] = False
    before = copy.deepcopy(bound)
    with pytest.raises(ValueError):
        native._shared_request(SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(bound)))
    assert bound == before and not case.control.events


def test_native_shared_pack_absent_request_is_ordinary_and_source_pins_cover_loader_and_scope(
    native_shared_pack_control,
):
    control = native_shared_pack_control
    assert control.native._shared_request(SimpleNamespace(model_dump=lambda **kwargs: {})) is None
    pins = control.native._shared_runtime_sources(control.training)
    assert pins == control.native._runtime_sources(control.training)
    assert pins["adapter"]["path"] == str(Path(shared.__file__))
    assert pins["composition"]["path"] == str(Path(control.native.__file__))
    assert pins["adapter"]["sha256"] == hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest()
    assert pins["composition"]["sha256"] == hashlib.sha256(Path(control.native.__file__).read_bytes()).hexdigest()
    control.native._verify_shared_runtime(control.training, pins)
    for key in pins:
        changed = copy.deepcopy(pins)
        changed[key]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="runtime differs"):
            control.native._verify_shared_runtime(control.training, changed)


@pytest.fixture
def native_shared_pack_coordinator(native_shared_pack_capability, monkeypatch):
    from types import FunctionType

    case = native_shared_pack_capability
    control = case.control
    control.ranks = [("synthetic-host-a", 0), ("synthetic-host-a", 1), ("synthetic-host-b", 0)]
    control.selected = []
    control.receipts = []
    control.gets = []
    control.kills = []
    control.ref = object()
    control.spawn_failure = None
    control.spawn_drift = False
    control.receipt_failure = None
    control.receipt_damage = None
    control.cleanup_failure = None
    control.constructor_failure_rank = None
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.jobs.gpu.training": control.training})
    ray = ModuleType("ray")

    def get(ref):
        assert ref is control.ref
        control.gets.append(ref)
        return control.files

    def remote(**options):
        assert options == {"num_cpus": 1, "num_gpus": 0}
        return lambda cls: cls

    ray.get = get
    ray.remote = remote
    monkeypatch.setitem(sys.modules, "ray", ray)

    def restored():
        assert control.training._DeepSpeedWorker is control.worker_base
        assert control.spawn_globals["_DeepSpeedWorker"] is control.worker_base
        control.restored()

    def construct(selected, rank):
        control.reset_model(rank)
        monkeypatch.setenv("RANK", str(rank))
        # Separate Ray processes have ordinary module bindings during worker construction.
        coordinator_worker = control.training._DeepSpeedWorker
        control.training._DeepSpeedWorker = control.worker_base
        try:
            saved_failure = control.constructor_failure
            if control.constructor_failure_rank != rank:
                control.constructor_failure = None
            try:
                worker = selected(control.cfg, control.source_stage)
            finally:
                control.constructor_failure = saved_failure
        finally:
            control.training._DeepSpeedWorker = coordinator_worker
        receipt = worker.shared_adapter_receipt()
        assert receipt["rank"] == rank
        second = worker.shared_adapter_receipt()
        second["engine_initializations"] = 99
        assert worker.shared_adapter_receipt() == receipt
        control.receipts.append(receipt)

        async def reported():
            restored()
            control.events.append(("synthetic-receipt", rank))
            if rank == 2 and control.receipt_failure is not None:
                raise control.receipt_failure
            result = worker.shared_adapter_receipt()
            if control.receipt_damage == rank:
                result["rank"] = 0 if rank else 1
            return result

        return SimpleNamespace(shared_adapter_receipt=SimpleNamespace(remote=reported))

    control.construct = construct

    def coordinator(spawn_namespace="detached"):
        namespace = dict(control.training.__dict__) if spawn_namespace == "detached" else control.training.__dict__
        template = _native_shared_pack_spawn_template
        spawn = FunctionType(template.__code__, namespace, "_spawn_workers")
        spawn.__qualname__ = "SyntheticNativeCoordinator._spawn_workers"
        control.spawn_globals = namespace

        class NativeCoordinator:
            def __init__(self):
                self._workers = []
                self._info = SimpleNamespace(config=None, job_id=control.cfg["job_id"])

            async def _kill_workers(self, hosts):
                restored()
                assert hosts == [host for host, _ in control.ranks]
                control.kills.append(list(hosts))
                self._workers.clear()
                if control.cleanup_failure is not None:
                    raise control.cleanup_failure

            def info(self):
                return {"ordinary-native-info": "preserved"}

        NativeCoordinator._spawn_workers = spawn
        control.training.GPUTrainingJob = SimpleNamespace(
            __ray_metadata__=SimpleNamespace(modified_class=NativeCoordinator)
        )
        case.pins.clear()
        case.pins.update(control.native._shared_runtime_sources(control.training))
        bound = control.native.bind_shared_adapter(case.payload, control.path, case.backend)
        request = bound["training_config"]["debug"][control.native.SHARED_REQUEST_KEY]
        control.request = request
        control.bound = bound
        control.original_module = dict(control.training.__dict__)
        if control.namespace_mode == "shared":
            control.original_namespace = dict(control.training.__dict__)
        selected = control.native._shared_coordinator_class(control.training, request, control.ref)
        instance = selected()
        instance._info.config = SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(bound))
        return instance

    control.coordinator = coordinator
    control.coordinator_restored = restored
    return control


@pytest.mark.parametrize("spawn_namespace", ["detached", "shared"])
def test_native_shared_pack_coordinator_constructs_every_rank_reports_receipts_and_restores_spawn(
    native_shared_pack_coordinator, spawn_namespace, capsys
):
    control = native_shared_pack_coordinator
    coordinator = control.coordinator(spawn_namespace)
    asyncio.run(
        coordinator._spawn_workers(control.cfg, control.ranks, "synthetic-master", 12345, control.source_stage)
    )
    control.coordinator_restored()
    assert len(control.gets) == len(control.receipts) == 3 and not control.kills
    assert [receipt["rank"] for receipt in control.receipts] == [0, 1, 2]
    control.native._verify_shared_receipts(control.receipts, control.request, control.cfg["job_id"])
    assert [event[0] for event in control.events] == [
        "synthetic-spawn",
        "synthetic-constructor",
        "real-peft-load",
        "stub-engine",
        "synthetic-constructor",
        "real-peft-load",
        "stub-engine",
        "synthetic-constructor",
        "real-peft-load",
        "stub-engine",
        "synthetic-receipt",
        "synthetic-receipt",
        "synthetic-receipt",
    ]
    report = coordinator.info()
    assert report == {
        "ordinary-native-info": "preserved",
        "correctness_shared_adapter": {"schema": control.native.SHARED_SCHEMA, "workers": control.receipts},
    }
    report["correctness_shared_adapter"]["workers"][0]["rank"] = 99
    assert coordinator.info()["correctness_shared_adapter"]["workers"] == control.receipts
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == {
        "schema": control.native.SHARED_SCHEMA,
        "job_id": control.cfg["job_id"],
        "workers": control.receipts,
    }


@pytest.mark.parametrize("spawn_namespace", ["detached", "shared"])
@pytest.mark.parametrize(
    "failure",
    [
        "spawn-runtime",
        "spawn-interrupt",
        "constructor-runtime",
        "constructor-interrupt",
        "receipt-runtime",
        "receipt-interrupt",
        "receipt-rank0",
        "receipt-rank1",
        "receipt-rank2",
        "spawn-binding",
        "spawn-cleanup-runtime",
        "receipt-cleanup-interrupt",
    ],
)
def test_native_shared_pack_coordinator_failure_owns_teardown_and_preserves_primary(
    native_shared_pack_coordinator, spawn_namespace, failure, capsys
):
    control = native_shared_pack_coordinator
    # Interrupt the cleanup, not a receipt task that bypasses coordinator exception delivery in asyncio.
    interrupt_primary = failure in {"spawn-interrupt", "constructor-interrupt", "receipt-interrupt"}
    primary = (
        KeyboardInterrupt("primary native failure") if interrupt_primary else RuntimeError("primary native failure")
    )
    expected = type(primary)
    if failure.startswith("constructor-"):
        control.constructor_failure = primary
        control.constructor_failure_rank = 1
    elif failure.startswith("receipt-rank"):
        control.receipt_damage = int(failure[-1])
        expected = ValueError
    elif failure == "spawn-binding":
        control.spawn_drift = True
        expected = ValueError
    elif failure.startswith("receipt-"):
        control.receipt_failure = primary
    else:
        control.spawn_failure = primary
    if "cleanup" in failure:
        control.cleanup_failure = (
            KeyboardInterrupt("owned cleanup failure")
            if "interrupt" in failure
            else RuntimeError("owned cleanup failure")
        )
    coordinator = control.coordinator(spawn_namespace)
    with pytest.raises(expected) as caught:
        asyncio.run(
            coordinator._spawn_workers(control.cfg, control.ranks, "synthetic-master", 12345, control.source_stage)
        )
    if expected is not ValueError:
        assert caught.value is primary
    control.coordinator_restored()
    assert len(control.kills) == 1 and not coordinator._workers
    assert coordinator.info()["correctness_shared_adapter"]["workers"] == []
    assert not capsys.readouterr().out.strip()
    notes = getattr(caught.value, "__notes__", [])
    if control.cleanup_failure is not None:
        assert len(notes) == 1 and "Owned shared-adapter worker teardown also failed" in notes[0]
        assert "owned cleanup failure" in notes[0]
    else:
        assert not notes


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "schema",
        "job",
        "rank-bool",
        "rank-order",
        "init-zero",
        "init-two",
        "init-bool",
        "identity",
        "missing-field",
        "extra-field",
        "not-dict",
    ],
)
def test_native_shared_pack_receipts_authenticate_every_rank_without_partial_acceptance(
    native_shared_pack_capability, rank, damage
):
    case = native_shared_pack_capability
    native = case.control.native
    request = {"training_gpus": 3, "identity": case.control.identity}
    receipts = [
        {
            "schema": "shared-peft-worker-initialization-v1",
            "job_id": "child",
            "rank": index,
            "engine_initializations": 1,
            "shared_identity": copy.deepcopy(case.control.identity),
        }
        for index in range(3)
    ]
    native._verify_shared_receipts(receipts, request, "child")
    if damage == "missing":
        del receipts[rank]
    elif damage == "duplicate":
        receipts[rank] = copy.deepcopy(receipts[(rank + 1) % 3])
    elif damage == "schema":
        receipts[rank]["schema"] = "original-peft-worker-initialization-v1"
    elif damage == "job":
        receipts[rank]["job_id"] = "other-child"
    elif damage == "rank-bool":
        receipts[rank]["rank"] = bool(rank)
    elif damage == "rank-order":
        receipts[rank]["rank"] = (rank + 1) % 3
    elif damage.startswith("init-"):
        receipts[rank]["engine_initializations"] = {"init-zero": 0, "init-two": 2, "init-bool": True}[damage]
    elif damage == "identity":
        receipts[rank]["shared_identity"]["generation"]["seed"] = 43
    elif damage == "missing-field":
        del receipts[rank]["shared_identity"]
    elif damage == "extra-field":
        receipts[rank]["initialization"] = {}
    else:
        receipts[rank] = None
    with pytest.raises(ValueError, match="every worker|evidence differs"):
        native._verify_shared_receipts(receipts, request, "child")


@pytest.mark.parametrize("damage", ["existing-workers", "missing-rank", "config-drift", "source-drift"])
def test_native_shared_pack_coordinator_rejects_layout_configuration_and_source_before_spawn(
    native_shared_pack_coordinator, damage
):
    control = native_shared_pack_coordinator
    coordinator = control.coordinator()
    if damage == "existing-workers":
        coordinator._workers = [object()]
    elif damage == "missing-rank":
        control.ranks = control.ranks[:-1]
    elif damage == "config-drift":
        control.bound["model_name"] = "other"
    else:
        control.request["sources"]["composition"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        asyncio.run(
            coordinator._spawn_workers(control.cfg, control.ranks, "synthetic-master", 12345, control.source_stage)
        )
    assert not control.events and not control.gets and not control.kills
    control.coordinator_restored()


# Shared Qwen4 reference residency policy (new controls only).


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_qwen4_residency_new_policy_is_structural_pure_and_identity_based(device):
    from dataclasses import FrozenInstanceError

    from arctic_platform.correctness.reference.qwen4_reference import qwen4_parameter_policy

    with torch.device(device):
        model = _qwen4_reference_TinyQwen4()
        model.misleading_ngram_embedding = nn.Linear(2, 3, bias=False)
        model.unrelated_indexer = nn.Linear(2, 3, bias=False)
    layer = model.layers[0]
    table = layer.ple.ple_embedding.ngram_embedding.weight
    indexer = layer.self_attn.indexer.weight
    before = {
        id(parameter): (parameter.requires_grad, parameter.shape, parameter.dtype, parameter.device)
        for parameter in model.parameters()
    }
    values = {id(parameter): parameter.detach().clone() for parameter in model.parameters()} if device == "cpu" else {}
    config = vars(model.config).copy()
    policy = qwen4_parameter_policy(model)
    assert policy.cpu_parameter_ids == frozenset({id(table)})
    assert policy.frozen_parameter_ids == frozenset({id(table), id(indexer)})
    assert policy.cpu_parameter_ids <= policy.frozen_parameter_ids
    assert before == {
        id(parameter): (parameter.requires_grad, parameter.shape, parameter.dtype, parameter.device)
        for parameter in model.parameters()
    }
    assert vars(model.config) == config
    for parameter in model.parameters():
        if device == "cpu":
            assert torch.equal(parameter, values[id(parameter)])
    with pytest.raises(FrozenInstanceError):
        policy.cpu_parameter_ids = frozenset()


def test_qwen4_residency_new_shared_table_and_indexer_are_deduplicated():
    from arctic_platform.correctness.reference.qwen4_reference import qwen4_parameter_policy

    model = _qwen4_reference_TinyQwen4()
    model.layers.append(model.layers[0])
    model.lm_head.weight = model.layers[0].ple.ple_embedding.ngram_embedding.weight
    policy = qwen4_parameter_policy(model)
    assert len(policy.cpu_parameter_ids) == 1
    assert len(policy.frozen_parameter_ids) == 2
    assert sum(parameter.numel() for parameter in model.parameters()) == 22


@pytest.mark.parametrize(
    "damage",
    [
        "ple",
        "self_attn",
        "ngram_embedding",
        "indexer",
        "tensor-table",
        "unregistered-table",
        "unregistered-indexer",
        "empty-indexer",
        "nonmodule-indexer",
        "decoder",
    ],
)
def test_qwen4_residency_new_malformed_policy_rejects_before_mutation(damage):
    from arctic_platform.correctness.reference.qwen4_reference import qwen4_parameter_policy

    model = _qwen4_reference_TinyQwen4()
    layer = model.layers[0]
    if damage in {"ple", "self_attn"}:
        delattr(layer, damage)
    elif damage == "ngram_embedding":
        del layer.ple.ple_embedding.ngram_embedding
    elif damage == "indexer":
        del layer.self_attn.indexer
    elif damage in {"tensor-table", "unregistered-table"}:
        embedding = layer.ple.ple_embedding.ngram_embedding
        table = embedding.weight
        del embedding.weight
        object.__setattr__(embedding, "weight", table.detach() if damage == "tensor-table" else table)
    elif damage == "unregistered-indexer":
        indexer = layer.self_attn.indexer
        del layer.self_attn.indexer
        object.__setattr__(layer.self_attn, "indexer", indexer)
    elif damage == "empty-indexer":
        layer.self_attn.indexer = nn.Identity()
    elif damage == "nonmodule-indexer":
        del layer.self_attn.indexer
        layer.self_attn.indexer = object()
    else:
        model.get_decoder = lambda: SimpleNamespace()
    before = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
    with pytest.raises(ValueError, match="PLE/QSA|registered"):
        qwen4_parameter_policy(model)
    assert before == {id(parameter): parameter.requires_grad for parameter in model.parameters()}


def test_qwen4_residency_new_loader_uses_shared_policy_and_preserves_table_bytes(monkeypatch):
    from arctic_platform.correctness.reference import qwen4_reference

    model = _qwen4_reference_TinyQwen4()
    table = model.layers[0].ple.ple_embedding.ngram_embedding.weight
    values = table.detach().clone()
    original_policy = qwen4_reference.qwen4_parameter_policy
    observed = []
    moved = []
    original_to = torch.Tensor.to

    def policy(loaded):
        assert loaded is model
        result = original_policy(loaded)
        observed.append(result)
        return result

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            assert tensor is not table
            moved.append(id(tensor))
            return original_to(tensor, device="cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(qwen4_reference, "qwen4_parameter_policy", policy)
    monkeypatch.setattr(torch.Tensor, "to", to)
    cls, calls = _qwen4_reference_loader(model)
    result = qwen4_reference.load_qwen4_reference(
        cls, "unloaded-synthetic-fixture", dtype=torch.bfloat16, attn_implementation="qsa_flex", device="cuda"
    )
    assert result is model and len(observed) == 1
    assert calls[0][1]["attn_implementation"] == "sdpa"
    assert calls[0][1]["output_loading_info"] is True
    assert not table.requires_grad and torch.equal(table, values)
    assert id(model.layers[0].self_attn.indexer.weight) in moved
    assert not model.layers[0].self_attn.indexer.weight.requires_grad
    assert id(model.layers[0].positions) in moved
    assert model.layers[0].ple.key_proj.weight.requires_grad
    assert model.layers[0].self_attn.q_proj.weight.requires_grad


@pytest.mark.parametrize("failure", ["load-info", "non-cpu", "attention"])
def test_qwen4_residency_new_loader_rejects_before_shared_policy(failure, monkeypatch):
    from arctic_platform.correctness.reference import qwen4_reference

    model = _qwen4_reference_TinyQwen4()
    if failure == "non-cpu":
        model = model.to("meta")
    cls, calls = _qwen4_reference_loader(model, {"missing_keys": ["required"]} if failure == "load-info" else None)
    monkeypatch.setattr(
        qwen4_reference, "qwen4_parameter_policy", lambda *args: pytest.fail("shared policy reached before admission")
    )
    with pytest.raises(ValueError, match="missing_keys|load on CPU|upstream QSA"):
        qwen4_reference.load_qwen4_reference(
            cls,
            "unloaded-synthetic-fixture",
            dtype=torch.bfloat16,
            attn_implementation="unsupported" if failure == "attention" else "sdpa",
            device="cpu",
        )
    assert len(calls) == (0 if failure == "attention" else 1)


# Ray tracing wrappers retain invocation while worker lookup belongs to the native leaf.


def _native_ray_tracing_wrap(method, events):
    from functools import wraps
    from types import FunctionType

    async def tracing(self, *args, _ray_trace_ctx=None, **kwargs):
        events.append("trace-enter")
        try:
            return await method(self, *args, **kwargs)
        finally:
            events.append("trace-exit")

    namespace = {"__name__": "ray.util.tracing.tracing_helper", "_DeepSpeedWorker": object()}
    wrapped = FunctionType(tracing.__code__, namespace, tracing.__name__, tracing.__defaults__, tracing.__closure__)
    wrapped.__kwdefaults__ = tracing.__kwdefaults__
    return wraps(method)(wrapped)


def _native_ray_tracing_copy_chain(callback):
    from types import CellType
    from types import FunctionType

    child = getattr(callback, "__wrapped__", None)
    copied_child = _native_ray_tracing_copy_chain(child) if child is not None else None
    closure = callback.__closure__
    if closure is not None and child is not None:
        closure = tuple(CellType(copied_child) if cell.cell_contents is child else cell for cell in closure)
    copied = FunctionType(
        callback.__code__, dict(callback.__globals__), callback.__name__, callback.__defaults__, closure
    )
    copied.__dict__.update(callback.__dict__)
    copied.__module__ = callback.__module__
    copied.__qualname__ = callback.__qualname__
    copied.__kwdefaults__ = callback.__kwdefaults__
    if child is not None:
        copied.__wrapped__ = copied_child
    return copied


def _native_ray_tracing_engine_template(model, ds_config, *, rank):
    control = globals()["_native_ray_tracing_control"]
    assert model is control.model and ds_config is control.ds_config and rank == control.rank
    control.events.append(("synthetic-engine", rank))
    control.active_receipt["engine_initializations"] += 1
    return control.engine_result


def _native_ray_tracing_constructor_template(self, cfg, source_stage_info=None):
    control = globals()["_native_ray_tracing_control"]
    assert cfg is control.cfg and source_stage_info is control.source_stage
    control.events.append(("synthetic-constructor", control.rank))
    self.engine = globals()["initialize_training_engine"](control.model, control.ds_config, rank=control.rank)
    if control.constructor_failure is not None:
        raise control.constructor_failure


@pytest.fixture
def native_ray_tracing_controls(tmp_path, monkeypatch):
    import importlib.metadata
    import importlib.util
    import socket
    from contextlib import contextmanager
    from types import FunctionType

    from arctic_platform.correctness.harness import native_lora

    def forbidden(*args, **kwargs):
        pytest.fail("Ray tracing controls attempted a cluster, subprocess, network, or model operation")

    class ForbiddenPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            forbidden(*args, **kwargs)

    monkeypatch.setenv(storage.CORRECTNESS_TMPDIR_ENV, str(tmp_path / "no-model-scratch"))
    monkeypatch.setattr(subprocess, "Popen", ForbiddenPopen)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(dist, "init_process_group", forbidden)
    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(Qwen3ForCausalLM, "forward", forbidden)
    monkeypatch.setattr(peft.PeftModel, "forward", forbidden)
    monkeypatch.setattr(transformers.GenerationMixin, "generate", forbidden)
    monkeypatch.setattr(sys.modules[__name__], "_model", forbidden)
    monkeypatch.setattr(sys.modules[__name__], "_native_original_adapter_model", forbidden)
    monkeypatch.setattr(shared, "generate_adapter", forbidden)

    installed = SimpleNamespace(inject=None, version=None, source=None, error=None, runtime="inert imports")
    ray_site = Path("/home/yak/miniconda3/envs/vllm/lib/python3.12/site-packages")
    helper_path = ray_site / "ray/util/tracing/tracing_helper.py"
    if helper_path.is_file():
        # Execute installed tracing code, not Ray's eager hardware/worker inventory.
        ray = ModuleType("ray")
        ray.__path__ = []
        private = ModuleType("ray._private")
        private.__path__ = []
        worker = ModuleType("ray._private.worker")
        runtime = ModuleType("ray.runtime_context")
        ray.init = ray.shutdown = ray.remote = runtime.get_runtime_context = forbidden
        ray._private, private.worker = private, worker
        for module in (ray, private, worker, runtime):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        inspect_spec = importlib.util.spec_from_file_location(
            "ray._private.inspect_util", ray_site / "ray/_private/inspect_util.py"
        )
        inspect_module = importlib.util.module_from_spec(inspect_spec)
        monkeypatch.setitem(sys.modules, inspect_spec.name, inspect_module)
        inspect_spec.loader.exec_module(inspect_module)
        helper_spec = importlib.util.spec_from_file_location("ray.util.tracing.tracing_helper", helper_path)
        helper = importlib.util.module_from_spec(helper_spec)
        monkeypatch.setitem(sys.modules, helper_spec.name, helper)
        helper_spec.loader.exec_module(helper)
        assert helper._is_tracing_enabled() is False
        installed.inject = helper._inject_tracing_into_class
        installed.version = next(
            distribution.version
            for distribution in importlib.metadata.distributions(path=[str(ray_site)])
            if distribution.metadata["Name"].lower() == "ray"
        )
        installed.source = native_lora._source(installed.inject)
        assert installed.source["sha256"] == hashlib.sha256(helper_path.read_bytes()).hexdigest()
    else:
        installed.error = "installed tracing helper file absent"

    def original_control():
        from arctic_platform.correctness.harness import native_adapter

        training = ModuleType("selected_native_tracing_control")
        tensor_specs = [
            lora_init.LoraTensorSpec(f"tiny.lora_{factor}.weight", (1, 1), "float32") for factor in ("A", "B")
        ]
        manifest = lora_init.build_manifest(tensor_specs, 42)
        data = save({spec.name: lora_init.generate_tensor(spec, 42) for spec in tensor_specs})
        adapter = SimpleNamespace(
            training=training,
            path=tmp_path / "synthetic-original-pack",
            worker_root=tmp_path / "unused-original",
            files={"adapter_model.safetensors": data},
            identity={"synthetic": "original-tracing"},
            metadata={
                "model_name": "synthetic-tracing-original",
                "identity_sha256": "synthetic-original-identity",
                "initialization": manifest,
            },
            manifest=manifest,
            model=object(),
            calls=[],
            result=(object(), object(), object()),
            ds_config={},
            optimizer=object(),
            mpu=object(),
        )

        def engine(model, ds_config, **kwargs):
            assert model is adapter.model and ds_config is adapter.ds_config
            adapter.calls.append((model, ds_config, kwargs))
            return adapter.result

        adapter.engine = training.initialize_training_engine = engine
        monkeypatch.setattr(native_adapter, "initialize_adapter", lambda *args: None)

        @contextmanager
        def scope(native_training, worker_constructor, files, identity, metadata, *, job_id, rank):
            assert native_training is training and files == adapter.files
            assert worker_constructor is training._DeepSpeedWorker.__init__
            assert identity == adapter.identity and metadata == adapter.metadata
            ordinary_engine = training.initialize_training_engine
            _, prototypes = _native_original_composition_receipts(adapter)
            receipt = copy.deepcopy(prototypes[rank])
            receipt["job_id"] = job_id
            receipt["engine_initializations"] = 0

            def scoped_engine(model, ds_config, **kwargs):
                from arctic_platform.correctness.harness import native_adapter

                native_adapter.initialize_adapter(model, adapter.path, identity)
                receipt["engine_initializations"] += 1
                return ordinary_engine(model, ds_config, **kwargs)

            training.initialize_training_engine = scoped_engine
            try:
                yield receipt
            finally:
                training.initialize_training_engine = ordinary_engine

        monkeypatch.setattr(native_lora, "native_original_worker_adapter_scope", scope)
        # Reuse coordinator construction without the fixture's model-producing dependency.
        return native_spawn_globals_control.__wrapped__(adapter, monkeypatch)

    def shared_control():
        peft_config = {"init_lora_weights": True}
        identity = {
            "schema": "shared-peft-adapter-v1",
            "peft_config": peft_config,
            "generation": {
                "schema": "upstream-default-lora-v1",
                "seed": 42,
                "initializer": True,
                "peft_config": peft_config,
                "dependencies": {
                    key: "synthetic-no-model" for key in ("torch", "peft", "transformers", "safetensors")
                },
            },
        }
        training = ModuleType("dss.ray_dss.jobs.gpu.training")
        control = SimpleNamespace(
            native=native_lora,
            training=training,
            identity=identity,
            files={"synthetic-pack": b"no-model"},
            path=tmp_path / "synthetic-shared-pack",
            cfg={"job_id": "synthetic-shared-child"},
            source_stage=object(),
            events=[],
            model=object(),
            ds_config={},
            rank=0,
            engine_result=object(),
            constructor_failure=None,
            active_receipt=None,
            namespace_mode="shared",
        )
        training.__dict__.update(_native_ray_tracing_control=control, _native_shared_pack_control=control)
        engine = FunctionType(_native_ray_tracing_engine_template.__code__, training.__dict__)
        engine.__qualname__ = _native_ray_tracing_engine_template.__qualname__
        training.initialize_training_engine = engine
        constructor = FunctionType(_native_ray_tracing_constructor_template.__code__, training.__dict__, "__init__")
        constructor.__qualname__ = "SyntheticNativeWorker.__init__"
        worker = type("SyntheticNativeWorker", (), {"__module__": training.__name__, "__init__": constructor})
        control.engine, control.worker_base = engine, worker
        training._DeepSpeedWorker = worker

        def reset_model(rank):
            control.rank = rank

        def restored():
            assert training._DeepSpeedWorker is worker and training.initialize_training_engine is engine
            assert control.active_receipt is None

        control.reset_model, control.restored = reset_model, restored
        pins = {}
        payload = {
            "job_type": "training",
            "model_name": "synthetic-tracing-shared",
            "training_config": {"n_gpus": 3, "peft_config": peft_config},
        }

        def bind(actual, path, backend):
            assert actual == payload and path == control.path
            bound = copy.deepcopy(actual)
            bound["training_config"]["debug"] = {
                native_lora.SHARED_REQUEST_KEY: {
                    "schema": native_lora.SHARED_SCHEMA,
                    "path": str(path),
                    "identity": copy.deepcopy(identity),
                    "sources": copy.deepcopy(pins),
                    "training_gpus": 3,
                    "model_name": actual["model_name"],
                }
            }
            return bound

        @contextmanager
        def scope(native_training, worker_constructor, files, expected_identity, *, job_id, rank):
            assert native_training is training and worker_constructor is worker.__init__
            assert files == control.files and expected_identity == identity
            receipt = {
                "schema": "shared-peft-worker-initialization-v1",
                "job_id": job_id,
                "rank": rank,
                "engine_initializations": 0,
                "shared_identity": copy.deepcopy(identity),
            }
            control.active_receipt = receipt
            try:
                yield receipt
            finally:
                control.active_receipt = None

        monkeypatch.setattr(native_lora, "bind_shared_adapter", bind)
        monkeypatch.setattr(native_lora, "native_shared_worker_adapter_scope", scope)
        case = SimpleNamespace(control=control, pins=pins, payload=payload, backend=object())
        return native_shared_pack_coordinator.__wrapped__(case, monkeypatch)

    def create(route, namespace="detached", wrapper="faithful"):
        control = original_control() if route == "original" else shared_control()
        coordinator = control.coordinator(namespace)
        training = control.training
        base = training.GPUTrainingJob.__ray_metadata__.modified_class
        leaf = base._spawn_workers
        traces = []
        if wrapper == "installed":
            if installed.inject is None:
                pytest.skip(f"installed Ray tracing helper unavailable: {installed.error}")

            class TraceOnly:
                pass

            TraceOnly._spawn_workers = leaf
            installed.inject(TraceOnly)
            wrapped = TraceOnly._spawn_workers
        else:
            wrapped = _native_ray_tracing_wrap(leaf, traces)
            if wrapper in {"nested", "copied"}:
                wrapped = _native_ray_tracing_wrap(wrapped, traces)
            if wrapper == "copied":
                wrapped = _native_ray_tracing_copy_chain(wrapped)
                leaf = inspect.unwrap(wrapped)
                control.spawn_globals = leaf.__globals__
                if route == "original":
                    control.spawn = leaf
                    control.namespace = "detached"
        base._spawn_workers = wrapped
        assert wrapped.__globals__["__name__"] == "ray.util.tracing.tracing_helper"
        assert inspect.unwrap(wrapped) is leaf
        assert inspect.iscoroutinefunction(wrapped) and inspect.iscoroutinefunction(leaf)
        control.tracing_route = route
        control.tracing_wrapped = wrapped
        control.tracing_leaf = leaf
        control.tracing_events = traces
        control.tracing_kind = wrapper
        control.tracing_installed = installed
        return control, coordinator

    return create


def _native_ray_tracing_snapshot(control):
    chain = []
    callback = control.tracing_wrapped
    while True:
        chain.append((callback.__globals__, dict(callback.__globals__)))
        callback = getattr(callback, "__wrapped__", None)
        if callback is None:
            break
    return dict(control.training.__dict__), chain


def _native_ray_tracing_assert_restored(control, snapshot):
    module, chain = snapshot
    assert control.training.__dict__ == module
    for namespace, before in chain:
        assert namespace == before
    if control.tracing_route == "original":
        control.restored()
    else:
        control.coordinator_restored()


def _native_ray_tracing_invoke(control, coordinator):
    cfg = control.config if control.tracing_route == "original" else control.cfg
    address = "owned-master" if control.tracing_route == "original" else "synthetic-master"
    return asyncio.run(coordinator._spawn_workers(cfg, control.ranks, address, 12345, control.source_stage))


@pytest.mark.parametrize("route", ["original", "shared"])
@pytest.mark.parametrize("namespace", ["detached", "shared"])
@pytest.mark.parametrize("wrapper", ["faithful", "nested", "copied", "installed"])
def test_native_ray_tracing_wrapped_spawn_patches_native_leaf_and_retains_wrapped_invocation_and_receipts(
    native_ray_tracing_controls, route, namespace, wrapper, capsys
):
    control, coordinator = native_ray_tracing_controls(route, namespace, wrapper)
    snapshot = _native_ray_tracing_snapshot(control)
    from arctic_platform.correctness.harness import native_lora

    actual_namespace, actual_worker = native_lora._spawn_worker_binding(
        control.tracing_wrapped, control.training, control.request["sources"]
    )
    assert actual_namespace is control.tracing_leaf.__globals__
    assert actual_namespace is not control.tracing_wrapped.__globals__
    assert actual_worker is control.worker_base
    _native_ray_tracing_invoke(control, coordinator)
    _native_ray_tracing_assert_restored(control, snapshot)
    assert len(control.selected) == 1 and control.selected[0] is not control.worker_base
    assert issubclass(control.selected[0], control.worker_base)
    if wrapper == "installed":
        assert control.tracing_installed.version and control.tracing_installed.source
        assert "_ray_trace_ctx" in inspect.signature(control.tracing_wrapped).parameters
    else:
        depth = 1 if wrapper == "faithful" else 2
        assert control.tracing_events == ["trace-enter"] * depth + ["trace-exit"] * depth
    printed = json.loads(capsys.readouterr().out.strip())
    if route == "original":
        assert control.gets == [control.ref]
        assert printed["workers"] == control.expected_receipts
        assert coordinator.info()["correctness_original_adapter"]["workers"] == control.expected_receipts
    else:
        assert len(control.gets) == len(control.receipts) == 3
        assert [row["rank"] for row in printed["workers"]] == [0, 1, 2]
        native_lora._verify_shared_receipts(printed["workers"], control.request, control.cfg["job_id"])


@pytest.mark.parametrize("route", ["original", "shared"])
@pytest.mark.parametrize(
    "failure",
    [
        "spawn-runtime",
        "spawn-interrupt",
        "receipt-runtime",
        "receipt-cleanup",
        "spawn-cleanup-runtime",
        "spawn-cleanup-interrupt",
    ],
)
def test_native_ray_tracing_wrapper_failure_restores_exact_globals_and_preserves_primary_and_owned_cleanup(
    native_ray_tracing_controls, route, failure, capsys
):
    # Each negative first proves the same wrapped/coordinator shape succeeds.
    valid, baseline = native_ray_tracing_controls(route, "detached", "copied")
    _native_ray_tracing_invoke(valid, baseline)
    assert json.loads(capsys.readouterr().out)["workers"]
    control, coordinator = native_ray_tracing_controls(route, "detached", "copied")
    primary = (
        KeyboardInterrupt("wrapped native primary")
        if failure == "spawn-interrupt"
        else RuntimeError("wrapped native primary")
    )
    if failure.startswith("receipt"):
        control.receipt_failure = primary
    else:
        control.spawn_failure = primary
    if "cleanup" in failure:
        control.cleanup_failure = (
            KeyboardInterrupt("wrapped owned cleanup")
            if failure.endswith("interrupt")
            else RuntimeError("wrapped owned cleanup")
        )
    snapshot = _native_ray_tracing_snapshot(control)
    with pytest.raises(type(primary)) as caught:
        _native_ray_tracing_invoke(control, coordinator)
    assert caught.value is primary
    _native_ray_tracing_assert_restored(control, snapshot)
    assert control.tracing_events == ["trace-enter", "trace-enter", "trace-exit", "trace-exit"]
    assert not capsys.readouterr().out.strip()
    cleanup_events = control.events if route == "original" else control.kills
    assert ("owned-kill" in cleanup_events) if route == "original" else len(cleanup_events) == 1
    notes = getattr(primary, "__notes__", [])
    if control.cleanup_failure is None:
        assert not notes and not coordinator._workers
    else:
        assert any("wrapped owned cleanup" in note for note in notes)
        assert any(type(control.cleanup_failure).__name__ in note for note in notes)


@pytest.mark.parametrize("route", ["original", "shared"])
@pytest.mark.parametrize("damage", ["leaf-namespace", "source", "foreign", "stale", "missing", "module-foreign"])
def test_native_ray_tracing_authenticated_leaf_and_worker_guards_reject_without_touching_wrapper_globals(
    native_ray_tracing_controls, route, damage, capsys
):
    from arctic_platform.correctness.harness import native_lora

    valid, baseline = native_ray_tracing_controls(route, "detached", "nested")
    _native_ray_tracing_invoke(valid, baseline)
    assert json.loads(capsys.readouterr().out)["workers"]
    control, coordinator = native_ray_tracing_controls(route, "detached", "nested")
    native_lora._spawn_worker_binding(control.tracing_wrapped, control.training, control.request["sources"])
    namespace = control.tracing_leaf.__globals__
    if damage == "leaf-namespace":
        namespace["__name__"] = "foreign.native.training"
    elif damage == "source":
        control.request["sources"]["training"]["sha256"] = "0" * 64
    elif damage == "foreign":
        namespace["_DeepSpeedWorker"] = object()
    elif damage == "stale":
        namespace["_DeepSpeedWorker"] = type("StaleAdapterWorker", (control.worker_base,), {})
    elif damage == "missing":
        namespace.pop("_DeepSpeedWorker")
    else:
        control.training._DeepSpeedWorker = type("ForeignModuleWorker", (), {"__init__": lambda self: None})
    snapshot = _native_ray_tracing_snapshot(control)
    with pytest.raises(ValueError):
        _native_ray_tracing_invoke(control, coordinator)
    module, chain = snapshot
    assert control.training.__dict__ == module
    assert all(actual == before for actual, before in chain)
    assert control.tracing_events == [] and control.selected == [] and control.gets == []
    assert not coordinator._workers and not capsys.readouterr().out.strip()


@pytest.mark.parametrize("route", ["original", "shared"])
@pytest.mark.parametrize("damage", ["sync-outer", "sync-leaf"])
def test_native_ray_tracing_requires_both_wrapper_and_leaf_to_be_coroutines(
    native_ray_tracing_controls, route, damage
):
    from functools import wraps

    from arctic_platform.correctness.harness import native_lora

    control, coordinator = native_ray_tracing_controls(route, "detached", "faithful")
    native_lora._spawn_worker_binding(control.tracing_wrapped, control.training, control.request["sources"])
    if damage == "sync-outer":

        @wraps(control.tracing_leaf)
        def invalid(*args, **kwargs):
            pytest.fail("synchronous outer wrapper invoked")

    else:

        def synchronous_leaf(*args, **kwargs):
            pytest.fail("synchronous leaf invoked")

        invalid = _native_ray_tracing_wrap(synchronous_leaf, [])
    snapshot = _native_ray_tracing_snapshot(control)
    with pytest.raises(ValueError, match="namespace differs"):
        native_lora._spawn_worker_binding(invalid, control.training, control.request["sources"])
    _native_ray_tracing_assert_restored(control, snapshot)
    assert not control.selected and not control.gets and not coordinator._workers


@pytest.mark.parametrize("wrapper", ["faithful", "nested", "installed"])
def test_native_ray_tracing_ordinary_native_dispatch_remains_wrapped_and_does_not_select_adapter(
    native_ray_tracing_controls, wrapper, capsys
):
    control, _ = native_ray_tracing_controls("original", "detached", wrapper)
    control.ordinary = True
    coordinator = control.coordinator_base()
    snapshot = _native_ray_tracing_snapshot(control)
    assert _native_ray_tracing_invoke(control, coordinator) is control.spawn_result
    _native_ray_tracing_assert_restored(control, snapshot)
    assert control.selected == [control.worker_base]
    assert control.events == ["native-spawn", "native-constructor"]
    assert control.gets == control.adapter.calls == []
    assert not capsys.readouterr().out.strip()
    if wrapper != "installed":
        depth = 1 if wrapper == "faithful" else 2
        assert control.tracing_events == ["trace-enter"] * depth + ["trace-exit"] * depth


# Registered adapter factors in real checkpoint-wrapped PEFT layouts, without a forward.


def _native_adapter_layout_factors(model):
    result = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name:
            parts = [part for part in name.split(".") if part != "_checkpoint_wrapped_module"]
            assert parts[-2] == "default"
            del parts[-2]
            canonical = ".".join(parts)
            assert canonical not in result
            result[canonical] = parameter
    return result


def _native_adapter_layout_snapshot(model):
    parameters = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameters[name] = (
            id(parameter),
            parameter.shape,
            parameter.dtype,
            parameter.device,
            parameter.requires_grad,
            (
                None
                if parameter.device.type == "meta"
                else (
                    (
                        parameter.untyped_storage().data_ptr(),
                        parameter.storage_offset(),
                        parameter.stride(),
                        _bytes(parameter),
                    )
                    if parameter.layout == torch.strided
                    else (parameter.layout, _bytes(parameter.to_dense()))
                )
            ),
        )
    return parameters, torch.random.get_rng_state().clone()


def _native_adapter_layout_unchanged(model, snapshot):
    actual, rng = _native_adapter_layout_snapshot(model)
    assert actual == snapshot[0]
    assert torch.equal(rng, snapshot[1])


def _native_adapter_layout_engine_template(model, *, rank):
    control = globals()["_native_adapter_layout_control"]
    assert model is control.model and rank == 0
    control.verify(model)
    control.events.append("stub-engine-after-verified-load")
    if control.engine_failure is not None:
        raise control.engine_failure
    return control.result


def _native_adapter_layout_constructor_template(self, cfg):
    self.engine = globals()["initialize_training_engine"](cfg["model"], rank=0)


@pytest.fixture
def native_adapter_layout_control(tmp_path, monkeypatch):
    import socket
    from types import FunctionType

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.correctness.harness import native_lora

    assert peft.__version__ == "0.21.2"

    def forbidden(*args, **kwargs):
        pytest.fail("native adapter layout controls attempted HF, forward, GPU, Ray, network, or subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(dist, "init_process_group", forbidden)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(Qwen3ForCausalLM, "forward", forbidden)
    monkeypatch.setattr(peft.PeftModel, "forward", forbidden)
    monkeypatch.setattr(nn.Linear, "forward", forbidden)
    monkeypatch.setattr(transformers.GenerationMixin, "generate", forbidden)
    monkeypatch.setattr(shared, "generate_adapter", forbidden)
    ray = ModuleType("ray")
    ray.init = ray.remote = ray.get = forbidden
    monkeypatch.setitem(sys.modules, "ray", ray)
    scratch_root = tmp_path / "native-worker"
    monkeypatch.setenv(storage.CORRECTNESS_TMPDIR_ENV, str(scratch_root))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "native_layout_tiny"}
            self.block = nn.Module()
            self.block.q_proj = nn.Linear(3, 4, bias=False)
            self.block.v_proj = nn.Linear(3, 4, bias=False)

        def forward(self, *args, **kwargs):
            forbidden(*args, **kwargs)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            forbidden(*args, **kwargs)

    def make(wrapped):
        base = Tiny()
        if wrapped:
            base.block = checkpoint_wrapper(base.block)
        model = peft.get_peft_model(
            base,
            peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                init_lora_weights=True,
            ),
        )
        assert len(_native_adapter_layout_factors(model)) == 4
        return model

    # Genuine local upstream-default factors, generated by PEFT under seed42.
    # This is not the maintained HF/meta generation path or an HF reference.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        supplier = make(False)
        path = tmp_path / "upstream-seed42-pack"
        supplier.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_bytes())
    config["base_model_name_or_path"] = ""
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    topology = shared._topology(config)
    recipe = {
        "schema": "upstream-default-lora-v1",
        "seed": 42,
        "initializer": True,
        "peft_config": topology,
        "dependencies": {
            "torch": str(torch.__version__),
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
    }
    (path / "correctness_recipe.json").write_text(json.dumps(recipe, sort_keys=True) + "\n")
    identity = shared.adapter_identity(path)
    files = shared.adapter_files(path, identity)
    expected = safetensors.torch.load(files["adapter_model.safetensors"])
    assert set(expected) == set(_native_adapter_layout_factors(supplier))
    assert all(
        _bytes(parameter) == _bytes(expected[name])
        for name, parameter in _native_adapter_layout_factors(supplier).items()
    )
    control = SimpleNamespace(
        make=make,
        path=path,
        identity=identity,
        files=files,
        expected=expected,
        native=native_lora,
        events=[],
        model=None,
        result=object(),
        engine_failure=None,
        scratch_root=scratch_root,
    )

    def verify(model):
        factors = _native_adapter_layout_factors(model)
        assert set(factors) == set(control.expected)
        for name, parameter in factors.items():
            assert _bytes(parameter) == _bytes(control.expected[name])

    control.verify = verify

    def scope(model, namespace_mode="detached", received=None):
        training = ModuleType("dss.ray_dss.jobs.gpu.training")
        training.__dict__["_native_adapter_layout_control"] = control
        engine = FunctionType(_native_adapter_layout_engine_template.__code__, training.__dict__)
        engine.__qualname__ = _native_adapter_layout_engine_template.__qualname__
        training.initialize_training_engine = engine
        namespace = dict(training.__dict__) if namespace_mode == "detached" else training.__dict__
        constructor = FunctionType(_native_adapter_layout_constructor_template.__code__, namespace, "__init__")
        constructor.__qualname__ = "LayoutNativeWorker.__init__"
        worker = type("LayoutNativeWorker", (), {"__module__": training.__name__, "__init__": constructor})
        training._DeepSpeedWorker = worker
        control.model = model
        before_module, before_namespace = dict(training.__dict__), dict(namespace)
        context = native_lora.native_shared_worker_adapter_scope(
            training,
            constructor,
            control.files if received is None else received,
            control.identity,
            job_id="layout-child",
            rank=0,
        )
        return SimpleNamespace(
            context=context,
            training=training,
            namespace=namespace,
            worker=worker,
            engine=engine,
            before_module=before_module,
            before_namespace=before_namespace,
        )

    control.scope = scope
    return control


def _native_adapter_layout_scope_restored(control, scope):
    assert scope.training.__dict__ == scope.before_module
    assert scope.namespace == scope.before_namespace
    assert scope.training.initialize_training_engine is scope.engine
    assert not control.scratch_root.exists() or not list(control.scratch_root.iterdir())
    assert shared._NATIVE_ADAPTER_SCOPE_LOCK.acquire(blocking=False)
    shared._NATIVE_ADAPTER_SCOPE_LOCK.release()


def _native_adapter_layout_forbid_state_apis(model, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("registered-factor loader called a full state_dict or PEFT getter/setter")

    monkeypatch.setattr(model, "state_dict", forbidden)
    monkeypatch.setattr(nn.Module, "state_dict", forbidden)
    for module in (peft, peft.utils, peft.utils.save_and_load):
        monkeypatch.setattr(module, "get_peft_model_state_dict", forbidden)
        monkeypatch.setattr(module, "set_peft_model_state_dict", forbidden)


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_adapter_layout_real_peft_checkpoint_mechanism_and_registered_coverage(
    native_adapter_layout_control, wrapped
):
    control = native_adapter_layout_control
    model = control.make(wrapped)
    snapshot = _native_adapter_layout_snapshot(model)
    names = [name for name, _ in model.named_parameters() if ".lora_" in name]
    serialized = [name for name in model.state_dict() if ".lora_" in name]
    getter = peft.get_peft_model_state_dict(model, adapter_name="default")
    assert len(names) == len(serialized) == 4
    assert any("_checkpoint_wrapped_module" in name for name in names) is wrapped
    assert not any("_checkpoint_wrapped_module" in name for name in serialized)
    assert len(getter) == (0 if wrapped else 4)
    _native_adapter_layout_unchanged(model, snapshot)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("signed_zero", [False, True])
def test_native_adapter_layout_exact_factor_bytes_existing_storage_base_and_rng_without_state_apis(
    native_adapter_layout_control, monkeypatch, wrapped, signed_zero
):
    control = native_adapter_layout_control
    model = control.make(wrapped)
    if signed_zero:
        state = {name: tensor.clone() for name, tensor in control.expected.items()}
        for tensor in state.values():
            tensor.flatten()[0] = -0.0
            tensor.flatten()[1] = 0.0
        save_file(state, control.path / "adapter_model.safetensors")
        control.identity = shared.adapter_identity(control.path)
        control.expected = state
    before, rng = _native_adapter_layout_snapshot(model)
    factors = _native_adapter_layout_factors(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    shared.initialize_adapter(model, control.path, control.identity)
    control.verify(model)
    after, actual_rng = _native_adapter_layout_snapshot(model)
    assert torch.equal(rng, actual_rng)
    for name, original in before.items():
        current = after[name]
        assert current[:5] == original[:5]
        assert current[5][:3] == original[5][:3]
        if ".lora_" not in name:
            assert not current[4] and current == original
    assert all(_native_adapter_layout_factors(model)[name] is parameter for name, parameter in factors.items())
    if signed_zero:
        assert all(
            torch.signbit(parameter.flatten()[0]) and not torch.signbit(parameter.flatten()[1])
            for parameter in factors.values()
        )
    assert not control.events


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "missing-A",
        "missing-B",
        "extra",
        "module-alias",
        "parameter-alias",
        "storage-alias",
        "foreign",
        "dtype",
        "shape",
        "meta",
        "non-strided",
        "nan",
        "inf",
        "topology",
        "unowned-factor",
    ],
)
def test_native_adapter_layout_invalid_registered_layout_rejects_before_mutation_and_engine(
    native_adapter_layout_control, monkeypatch, wrapped, damage
):
    control = native_adapter_layout_control
    valid = control.make(wrapped)
    shared.initialize_adapter(valid, control.path, control.identity)
    control.verify(valid)
    model = control.make(wrapped)
    block = model.base_model.model.block
    q, v = block.q_proj, block.v_proj
    if damage in {"missing-A", "missing-B"}:
        del getattr(q, "lora_" + damage[-1])["default"]
    elif damage == "extra":
        block.extra_proj = copy.deepcopy(q)
    elif damage == "module-alias":
        block.alias_proj = q
    elif damage == "parameter-alias":
        v.lora_A["default"].weight = q.lora_A["default"].weight
    elif damage == "storage-alias":
        v.lora_A["default"].weight = nn.Parameter(q.lora_A["default"].weight.detach())
    elif damage == "foreign":
        model.add_adapter("foreign", copy.deepcopy(model.peft_config["default"]))
    elif damage in {"dtype", "shape", "meta", "non-strided"}:
        old = q.lora_A["default"].weight
        replacement = (
            old.detach().double()
            if damage == "dtype"
            else (
                torch.empty((1, 3), dtype=old.dtype)
                if damage == "shape"
                else old.detach().to_sparse() if damage == "non-strided" else torch.empty_like(old, device="meta")
            )
        )
        q.lora_A["default"].weight = nn.Parameter(replacement)
    elif damage in {"nan", "inf"}:
        with torch.no_grad():
            q.lora_A["default"].weight.flatten()[0] = float(damage)
    elif damage == "topology":
        model.peft_config["default"].r = 3
    else:
        block.unowned = nn.Module()
        block.unowned.lora_A = nn.ModuleDict({"default": nn.Linear(3, 2, bias=False)})
        block.unowned.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
    snapshot = _native_adapter_layout_snapshot(model)
    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    with pytest.raises(ValueError):
        with scope.context:
            scope.worker({"model": model})
    assert not control.events
    _native_adapter_layout_unchanged(model, snapshot)
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("namespace", ["detached", "shared"])
@pytest.mark.parametrize(
    "outcome", ["success", "engine-runtime", "engine-interrupt", "body-runtime", "body-interrupt"]
)
def test_native_adapter_layout_generated_pack_native_scope_receipt_and_all_exit_restoration(
    native_adapter_layout_control, monkeypatch, namespace, outcome
):
    control = native_adapter_layout_control
    model = control.make(True)
    before, rng = _native_adapter_layout_snapshot(model)
    scope = control.scope(model, namespace)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    primary = KeyboardInterrupt("layout primary") if outcome.endswith("interrupt") else RuntimeError("layout primary")
    if outcome.startswith("engine"):
        control.engine_failure = primary

    def exercise():
        with scope.context as receipt:
            worker = scope.worker({"model": model})
            assert worker.engine is control.result
            assert receipt == {
                "schema": "shared-peft-worker-initialization-v1",
                "job_id": "layout-child",
                "rank": 0,
                "engine_initializations": 1,
                "shared_identity": control.identity,
            }
            if outcome.startswith("body"):
                raise primary
        return receipt

    if outcome == "success":
        exercise()
    else:
        with pytest.raises(type(primary)) as caught:
            exercise()
        assert caught.value is primary
    assert control.events == ["stub-engine-after-verified-load"]
    control.verify(model)
    after, actual_rng = _native_adapter_layout_snapshot(model)
    assert torch.equal(actual_rng, rng)
    for name, original in before.items():
        assert after[name][:5] == original[:5]
        assert after[name][5][:3] == original[5][:3]
        if ".lora_" not in name:
            assert after[name] == original
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize(
    "damage", ["missing-pair", "extra-pair", "duplicate-canonical", "dtype", "shape", "nan", "topology"]
)
def test_native_adapter_layout_pack_rejection_restores_without_parameter_mutation_or_engine(
    native_adapter_layout_control, monkeypatch, damage
):
    control = native_adapter_layout_control
    model = control.make(True)
    state = {name: tensor.clone() for name, tensor in control.expected.items()}
    name = sorted(state)[0]
    if damage == "missing-pair":
        del state[name]
    elif damage == "extra-pair":
        state.update(
            {
                key.replace(".q_proj.", ".extra_proj."): value.clone()
                for key, value in list(state.items())
                if ".q_proj." in key
            }
        )
    elif damage == "duplicate-canonical":
        state[name.replace(".block.", ".block._checkpoint_wrapped_module.")] = state[name].clone()
    elif damage == "dtype":
        state[name] = state[name].double()
    elif damage == "shape":
        state[name] = state[name][:1].clone()
    elif damage == "nan":
        state[name].flatten()[0] = float("nan")
    else:
        config = json.loads(control.files["adapter_config.json"])
        config["lora_alpha"] = 5
        (control.path / "adapter_config.json").write_text(json.dumps(config))
    if damage != "topology":
        save_file(state, control.path / "adapter_model.safetensors")
    received = {name: (control.path / name).read_bytes() for name in control.files}
    snapshot = _native_adapter_layout_snapshot(model)
    scope = control.scope(model, received=received)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    with pytest.raises(ValueError):
        with scope.context:
            scope.worker({"model": model})
    assert not control.events
    _native_adapter_layout_unchanged(model, snapshot)
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "damage", ["base-parameter", "base-storage", "new-base-parameter", "new-base-storage", "base-buffer"]
)
def test_native_adapter_layout_base_aliases_reject_without_base_or_factor_mutation(
    native_adapter_layout_control, monkeypatch, wrapped, damage
):
    control = native_adapter_layout_control
    valid = control.make(wrapped)
    shared.initialize_adapter(valid, control.path, control.identity)
    control.verify(valid)
    model = control.make(wrapped)
    block = model.base_model.model.block
    q = block.q_proj
    factor = q.lora_A["default"].weight
    if damage == "base-parameter":
        q.base_layer.weight = factor
    elif damage == "base-storage":
        q.lora_A["default"].weight = nn.Parameter(q.base_layer.weight[:2].detach())
    elif damage == "new-base-parameter":
        block.register_parameter("ordinary_alias", factor)
    elif damage == "new-base-storage":
        block.register_parameter("ordinary_alias", nn.Parameter(factor.detach(), requires_grad=False))
    else:
        block.register_buffer("ordinary_alias", factor.detach())
    buffers = {name: (id(value), _bytes(value)) for name, value in model.named_buffers(remove_duplicate=False)}
    snapshot = _native_adapter_layout_snapshot(model)
    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    with pytest.raises(ValueError):
        with scope.context:
            scope.worker({"model": model})
    assert not control.events
    _native_adapter_layout_unchanged(model, snapshot)
    assert buffers == {name: (id(value), _bytes(value)) for name, value in model.named_buffers(remove_duplicate=False)}
    _native_adapter_layout_scope_restored(control, scope)


@pytest.fixture
def native_adapter_layout_four_block_control(native_adapter_layout_control, tmp_path):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    control = native_adapter_layout_control

    class FourBlock(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.config = {"model_type": "native_layout_four_block"}
            self.blocks = nn.ModuleList()
            for _ in range(4):
                block = nn.Module()
                block.q_proj = nn.Linear(3, 3, bias=False)
                block.v_proj = nn.Linear(3, 3, bias=False)
                self.blocks.append(checkpoint_wrapper(block) if wrapped else block)

        def forward(self, *args, **kwargs):
            pytest.fail("four-block layout control invoked a forward")

        def prepare_inputs_for_generation(self, *args, **kwargs):
            pytest.fail("four-block layout control invoked generation")

    def make(wrapped):
        model = peft.get_peft_model(
            FourBlock(wrapped),
            peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                init_lora_weights=True,
            ),
        )
        assert len(_native_adapter_layout_factors(model)) == 16
        return model

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        supplier = make(False)
        path = tmp_path / "four-block-seed42-pack"
        supplier.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_bytes())
    config["base_model_name_or_path"] = ""
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    recipe = json.loads(control.files["correctness_recipe.json"])
    recipe["peft_config"] = shared._topology(config)
    (path / "correctness_recipe.json").write_text(json.dumps(recipe, sort_keys=True) + "\n")
    control.make, control.path = make, path
    control.identity = shared.adapter_identity(path)
    control.files = shared.adapter_files(path, control.identity)
    control.expected = safetensors.torch.load(control.files["adapter_model.safetensors"])
    assert len(control.expected) == 16
    control.verify(supplier)
    return control


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_adapter_layout_four_blocks_all_sixteen_factors_copy_before_stub_engine(
    native_adapter_layout_four_block_control, monkeypatch, wrapped
):
    control = native_adapter_layout_four_block_control
    model = control.make(wrapped)
    snapshot = _native_adapter_layout_snapshot(model)
    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    with scope.context as receipt:
        scope.worker({"model": model})
        assert receipt["engine_initializations"] == 1
    control.verify(model)
    after = _native_adapter_layout_snapshot(model)
    assert torch.equal(after[1], snapshot[1])
    for name, original in snapshot[0].items():
        assert after[0][name][:5] == original[:5]
        assert after[0][name][5][:3] == original[5][:3]
        if ".lora_" not in name:
            assert after[0][name] == original
    assert len(_native_adapter_layout_factors(model)) == 16
    assert control.events == ["stub-engine-after-verified-load"]
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("damage", ["missing-last", "nan-last", "inf-last", "dtype-last", "shape-last"])
def test_native_adapter_layout_last_of_sixteen_invalid_factors_rejects_before_any_earlier_write(
    native_adapter_layout_four_block_control, monkeypatch, wrapped, damage
):
    control = native_adapter_layout_four_block_control
    valid = control.make(wrapped)
    shared.initialize_adapter(valid, control.path, control.identity)
    control.verify(valid)
    model = control.make(wrapped)
    last = model.base_model.model.blocks[3].v_proj.lora_B["default"]
    if damage == "missing-last":
        del last.weight
    elif damage in {"nan-last", "inf-last"}:
        with torch.no_grad():
            last.weight.flatten()[0] = float(damage.split("-")[0])
    else:
        value = last.weight.detach().double() if damage == "dtype-last" else last.weight.detach()[:1].clone()
        last.weight = nn.Parameter(value)
    snapshot = _native_adapter_layout_snapshot(model)
    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    with pytest.raises(ValueError):
        with scope.context:
            scope.worker({"model": model})
    assert not control.events
    _native_adapter_layout_unchanged(model, snapshot)
    _native_adapter_layout_scope_restored(control, scope)


# Native-only factor transactions; reference supplier source remains immutable.


def _native_adapter_layout_rollback_buffers(model):
    return {
        name: (
            id(value),
            value.shape,
            value.dtype,
            value.device,
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            value.stride(),
            _bytes(value),
        )
        for name, value in model.named_buffers(remove_duplicate=False)
    }


def _native_adapter_layout_rollback_model(control, wrapped, native_initializer):
    valid = control.make(wrapped)
    native_initializer.initialize_adapter(valid, control.path, control.identity)
    control.verify(valid)
    model = control.make(wrapped)
    model.base_model.model.register_buffer("rollback_frozen_buffer", torch.tensor([0.0, -0.0, 7.0]))
    assert any(
        _bytes(parameter) != _bytes(control.expected[name])
        for name, parameter in _native_adapter_layout_factors(model).items()
    )
    return model


@pytest.fixture
def native_adapter_layout_rollback_control(native_adapter_layout_control):
    from arctic_platform.correctness.harness import native_adapter

    return SimpleNamespace(control=native_adapter_layout_control, initializer=native_adapter)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("nth", [2, 4])
@pytest.mark.parametrize("interrupt", [False, True])
def test_native_adapter_layout_rollback_nth_partial_copy_retains_primary_and_exact_originals(
    native_adapter_layout_rollback_control, monkeypatch, wrapped, nth, interrupt
):
    case = native_adapter_layout_rollback_control
    control = case.control
    model = _native_adapter_layout_rollback_model(control, wrapped, case.initializer)
    factors = _native_adapter_layout_factors(model)
    tracked = {id(parameter) for parameter in factors.values()}
    snapshot = _native_adapter_layout_snapshot(model)
    buffers = _native_adapter_layout_rollback_buffers(model)
    before_bytes = {name: _bytes(parameter) for name, parameter in factors.items()}
    original_copy = torch.Tensor.copy_
    primary = KeyboardInterrupt("nth native copy interrupted") if interrupt else RuntimeError("nth native copy failed")
    attempts = []
    raised = False
    changed_before_failure = []

    def copy_(parameter, value, *args, **kwargs):
        nonlocal raised
        if id(parameter) not in tracked:
            return original_copy(parameter, value, *args, **kwargs)
        attempts.append(id(parameter))
        result = original_copy(parameter, value, *args, **kwargs)
        if not raised and len(attempts) == nth:
            changed_before_failure.extend(
                name for name, current in factors.items() if _bytes(current) != before_bytes[name]
            )
            raised = True
            raise primary
        return result

    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    with pytest.raises(type(primary)) as caught:
        with scope.context:
            scope.worker({"model": model})
    assert caught.value is primary and raised and changed_before_failure
    assert len(attempts) == nth + len(factors)
    assert set(attempts[nth:]) == tracked
    assert not control.events
    _native_adapter_layout_unchanged(model, snapshot)
    assert _native_adapter_layout_rollback_buffers(model) == buffers
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("outcome", ["raises", "rejects"])
def test_native_adapter_layout_rollback_final_only_identity_postflight_restores_every_factor(
    native_adapter_layout_rollback_control, monkeypatch, wrapped, outcome
):
    case = native_adapter_layout_rollback_control
    control = case.control
    model = _native_adapter_layout_rollback_model(control, wrapped, case.initializer)
    factors = _native_adapter_layout_factors(model)
    snapshot = _native_adapter_layout_snapshot(model)
    buffers = _native_adapter_layout_rollback_buffers(model)
    original_identity = case.initializer.adapter_identity
    primary = RuntimeError("native final identity recheck failed")
    postflight = []

    def identity(path):
        actual = original_identity(path)
        if all(_bytes(parameter) == _bytes(control.expected[name]) for name, parameter in factors.items()):
            postflight.append(actual)
            if outcome == "raises":
                raise primary
            altered = copy.deepcopy(actual)
            altered["generation"]["seed"] += 1
            return altered
        return actual

    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    monkeypatch.setattr(case.initializer, "adapter_identity", identity)
    with pytest.raises(RuntimeError if outcome == "raises" else ValueError) as caught:
        with scope.context:
            scope.worker({"model": model})
    if outcome == "raises":
        assert caught.value is primary
    else:
        assert "changed during loading" in str(caught.value)
    assert postflight == [control.identity]
    assert not control.events
    _native_adapter_layout_unchanged(model, snapshot)
    assert _native_adapter_layout_rollback_buffers(model) == buffers
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
def test_native_adapter_layout_rollback_restore_failure_notes_preserve_primary_and_continue_cleanup(
    native_adapter_layout_rollback_control, monkeypatch, wrapped, interrupt
):
    case = native_adapter_layout_rollback_control
    control = case.control
    model = _native_adapter_layout_rollback_model(control, wrapped, case.initializer)
    factors = _native_adapter_layout_factors(model)
    tracked = {id(parameter) for parameter in factors.values()}
    first = next(iter(factors.values()))
    snapshot = _native_adapter_layout_snapshot(model)
    buffers = _native_adapter_layout_rollback_buffers(model)
    original_copy = torch.Tensor.copy_
    primary = KeyboardInterrupt("native primary interruption") if interrupt else RuntimeError("native primary failure")
    secondary = RuntimeError("injected factor restoration failure")
    writes = []
    restores = []
    rollback_started = False

    def copy_(parameter, value, *args, **kwargs):
        nonlocal rollback_started
        if id(parameter) not in tracked:
            return original_copy(parameter, value, *args, **kwargs)
        if rollback_started:
            restores.append(id(parameter))
            if parameter is first:
                raise secondary
            return original_copy(parameter, value, *args, **kwargs)
        writes.append(id(parameter))
        result = original_copy(parameter, value, *args, **kwargs)
        if len(writes) == 2:
            rollback_started = True
            raise primary
        return result

    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    with pytest.raises(type(primary)) as caught:
        with scope.context:
            scope.worker({"model": model})
    assert caught.value is primary
    notes = getattr(primary, "__notes__", [])
    assert any(str(secondary) in note for note in notes)
    assert any("verification" in note and "rollback" in note for note in notes)
    assert len(restores) == len(factors) and set(restores) == tracked
    after, rng = _native_adapter_layout_snapshot(model)
    assert torch.equal(rng, snapshot[1])
    for name, original in snapshot[0].items():
        assert after[name][:5] == original[:5]
        assert after[name][5][:3] == original[5][:3]
        if original[0] != id(first):
            assert after[name] == original
    assert any(after[name] != original for name, original in snapshot[0].items() if original[0] == id(first))
    assert _native_adapter_layout_rollback_buffers(model) == buffers
    assert not control.events
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_adapter_layout_rollback_expanded_last_of_sixteen_rejects_before_any_copy(
    native_adapter_layout_four_block_control, monkeypatch, wrapped
):
    from arctic_platform.correctness.harness import native_adapter

    control = native_adapter_layout_four_block_control
    model = _native_adapter_layout_rollback_model(control, wrapped, native_adapter)
    last = model.base_model.model.blocks[3].v_proj.lora_B["default"]
    last.weight = nn.Parameter(last.weight.detach()[:1].clone().expand_as(last.weight))
    assert not last.weight.is_contiguous() and last.weight.stride()[0] == 0
    snapshot = _native_adapter_layout_snapshot(model)
    buffers = _native_adapter_layout_rollback_buffers(model)
    tracked = {id(parameter) for parameter in _native_adapter_layout_factors(model).values()}
    original_copy = torch.Tensor.copy_
    attempts = []

    def copy_(parameter, value, *args, **kwargs):
        if id(parameter) in tracked:
            attempts.append(id(parameter))
        return original_copy(parameter, value, *args, **kwargs)

    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    with pytest.raises(ValueError, match="materialized contiguous"):
        with scope.context:
            scope.worker({"model": model})
    assert not attempts and not control.events
    _native_adapter_layout_unchanged(model, snapshot)
    assert _native_adapter_layout_rollback_buffers(model) == buffers
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_adapter_layout_rollback_scope_routes_only_native_initializer_before_stub_engine(
    native_adapter_layout_rollback_control, monkeypatch, wrapped
):
    case = native_adapter_layout_rollback_control
    control = case.control
    model = _native_adapter_layout_rollback_model(control, wrapped, case.initializer)
    snapshot = _native_adapter_layout_snapshot(model)
    buffers = _native_adapter_layout_rollback_buffers(model)
    original = case.initializer.initialize_adapter
    calls = []

    def initialize(actual_model, path, identity):
        assert actual_model is model and identity == control.identity
        assert not control.events
        calls.append((actual_model, Path(path), identity))
        original(actual_model, path, identity)
        control.verify(actual_model)

    def forbidden(*args, **kwargs):
        pytest.fail("native shared scope routed through the frozen reference loader")

    scope = control.scope(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    monkeypatch.setattr(case.initializer, "initialize_adapter", initialize)
    monkeypatch.setattr(shared, "initialize_adapter", forbidden)
    with scope.context as receipt:
        scope.worker({"model": model})
        assert receipt["engine_initializations"] == 1 and receipt["shared_identity"] == control.identity
    assert len(calls) == 1 and control.events == ["stub-engine-after-verified-load"]
    control.verify(model)
    after, rng = _native_adapter_layout_snapshot(model)
    assert torch.equal(rng, snapshot[1])
    for name, original_value in snapshot[0].items():
        assert after[name][:5] == original_value[:5]
        assert after[name][5][:3] == original_value[5][:3]
        if ".lora_" not in name:
            assert after[name] == original_value
    assert _native_adapter_layout_rollback_buffers(model) == buffers
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("damage", ["none", "missing-native", "extra-original", "malformed-native"])
def test_native_adapter_layout_rollback_source_compatibility_and_distinct_gateway_capability(
    native_adapter_layout_rollback_control, monkeypatch, damage
):
    import requests

    from arctic_platform.correctness.harness import hosted

    case = native_adapter_layout_rollback_control
    control = case.control
    scope = control.scope(control.make(False))
    native = control.native
    original = native._runtime_sources(scope.training)
    generated = native._shared_runtime_sources(scope.training)
    assert set(original) == {"training", "engine", "adapter", "lora", "composition", "native_adapter"}
    assert generated == original
    assert original["native_adapter"] == native._source(case.initializer.initialize_adapter)
    shared_path = Path(shared.__file__).resolve()
    assert (
        hashlib.sha256(shared_path.read_bytes()).hexdigest()
        == "fcbedc59fb4ddb79aacb47708d9bd753d0edd77082b0e301ab4263bbbf39aa2a"
    )
    assert original["adapter"] == {
        "path": str(shared_path),
        "sha256": hashlib.sha256(shared_path.read_bytes()).hexdigest(),
    }
    native._verify_shared_runtime(scope.training, generated)
    drifted = copy.deepcopy(generated)
    drifted["native_adapter"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime differs"):
        native._verify_shared_runtime(scope.training, drifted)
    document = {
        native.OPENAPI_KEY: {"schema": native.SCHEMA, "sources": copy.deepcopy(original)},
        native.SHARED_OPENAPI_KEY: {"schema": native.SHARED_SCHEMA, "sources": copy.deepcopy(generated)},
    }

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert self.trust_env is False
            assert kwargs == {"timeout": 30, "allow_redirects": False}
            return SimpleNamespace(status_code=200, json=lambda: copy.deepcopy(document))

    monkeypatch.setattr(requests, "Session", Session)
    backend = SimpleNamespace(base_url="http://127.0.0.1:8000")
    monkeypatch.setattr(
        hosted,
        "require_local_cortex_gateway",
        lambda selected: None if selected is backend else pytest.fail("foreign backend"),
    )
    assert native.gateway_capability(backend)["sources"] == original
    assert native.gateway_capability(backend, shared=True)["sources"] == generated
    if damage == "missing-native":
        del document[native.SHARED_OPENAPI_KEY]["sources"]["native_adapter"]
    elif damage == "extra-original":
        document[native.OPENAPI_KEY]["sources"]["unexpected"] = generated["native_adapter"]
    elif damage == "malformed-native":
        document[native.SHARED_OPENAPI_KEY]["sources"]["native_adapter"]["path"] = "relative-native.py"
    if damage != "none":
        with pytest.raises(ValueError):
            native.gateway_capability(backend, shared=damage != "extra-original")
    assert _native_adapter_layout_rollback_buffers(scope.training._native_adapter_layout_control.model) == {}
    _native_adapter_layout_scope_restored(control, scope)


# Original packs use registered factors and the actual constructor namespace too.


def _native_original_registered_engine_template(model, *, rank):
    control = globals()["_native_original_registered_control"]
    assert model is control.model and rank == 0
    control.verify(model)
    control.calls.append(model)
    if control.engine_failure is not None:
        raise control.engine_failure
    return control.result


@pytest.fixture
def native_original_registered_control(native_original_adapter, monkeypatch):
    import socket
    from types import FunctionType

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.correctness.harness import native_adapter
    from arctic_platform.correctness.harness import native_lora

    control = native_original_adapter
    control.calls.clear()
    control.scratch_root = control.worker_root
    control.engine_failure = None

    def forbidden(*args, **kwargs):
        pytest.fail("original native controls attempted forward, HF, GPU, Ray, network, or subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(dist, "init_process_group", forbidden)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(shared, "generate_adapter", forbidden)
    ray = ModuleType("ray")
    ray.init = ray.remote = ray.get = forbidden
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(shared, "initialize_adapter", forbidden)

    def verify(model):
        return lora_init.verify_tensors(native_adapter._adapter_parameters(model), control.manifest)

    def scope(wrapped=True, namespace_mode="detached", files=None, rank=0):
        model = _native_original_adapter_model()
        if wrapped:
            model.base_model.model = checkpoint_wrapper(model.base_model.model)
        control.model = model
        training = ModuleType("dss.ray_dss.jobs.gpu.training")
        training.__dict__["_native_original_registered_control"] = control
        engine = FunctionType(_native_original_registered_engine_template.__code__, training.__dict__)
        engine.__qualname__ = _native_original_registered_engine_template.__qualname__
        training.initialize_training_engine = engine
        namespace = dict(training.__dict__) if namespace_mode == "detached" else training.__dict__
        constructor = FunctionType(_native_adapter_layout_constructor_template.__code__, namespace, "__init__")
        constructor.__qualname__ = "OriginalRegisteredWorker.__init__"
        worker = type("OriginalRegisteredWorker", (), {"__module__": training.__name__, "__init__": constructor})
        training._DeepSpeedWorker = worker
        return SimpleNamespace(
            training=training,
            namespace=namespace,
            worker=worker,
            engine=engine,
            before_module=dict(training.__dict__),
            before_namespace=dict(namespace),
            context=native_lora.native_original_worker_adapter_scope(
                training,
                constructor,
                control.files if files is None else files,
                control.identity,
                control.metadata,
                job_id="original-child",
                rank=rank,
            ),
        )

    control.verify = verify
    control.scope = scope
    control.native = native_lora
    return control


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("namespace_mode", ["shared", "detached"])
def test_native_original_registered_factors_load_exact_bytes_before_engine(
    native_original_registered_control, monkeypatch, wrapped, namespace_mode
):
    control = native_original_registered_control
    scope = control.scope(wrapped, namespace_mode)
    model = control.model
    assert len(peft.get_peft_model_state_dict(model)) == (0 if wrapped else 4)
    snapshot, rng = _native_adapter_layout_snapshot(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    with scope.context as receipt:
        worker = scope.worker({"model": model})
        assert worker.engine is control.result
    verified = control.verify(model)
    assert receipt == {
        "schema": "original-peft-worker-initialization-v1",
        "job_id": "original-child",
        "rank": 0,
        "engine_initializations": 1,
        "shared_identity": control.identity,
        "original_identity_sha256": control.metadata["identity_sha256"],
        "initialization": verified,
    }
    control.native._verify_receipts(
        [receipt],
        {"training_gpus": 1, "identity": control.identity, "original_metadata": control.metadata},
        "original-child",
    )
    after, actual_rng = _native_adapter_layout_snapshot(model)
    assert torch.equal(rng, actual_rng)
    for name, original in snapshot.items():
        assert after[name][:5] == original[:5]
        assert after[name][5][:3] == original[5][:3]
        if ".lora_" not in name:
            assert after[name] == original
    assert control.calls == [model]
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("damage", ["metadata", "factor-bytes", "missing-metadata", "extra-file", "dtype", "alias"])
def test_native_original_registered_rejection_keeps_parameters_and_skips_engine(
    native_original_registered_control, monkeypatch, damage
):
    control = native_original_registered_control
    files = dict(control.files)
    if damage == "metadata":
        files["initialization.json"] = b"{}"
    elif damage == "factor-bytes":
        tensors = safetensors.torch.load(files["adapter_model.safetensors"])
        tensors[sorted(tensors)[0]][0, 0] += 1
        files["adapter_model.safetensors"] = save(tensors)
    elif damage == "missing-metadata":
        del files["initialization.json"]
    elif damage == "extra-file":
        files["unowned.json"] = b"{}"
    scope = control.scope(files=files)
    model = control.model
    layer = model.base_model.model.q_proj
    if damage == "dtype":
        layer.lora_A["default"].to(dtype=torch.float32)
    elif damage == "alias":
        layer.lora_A["default"].weight = layer.lora_B["default"].weight
    snapshot = _native_adapter_layout_snapshot(model)
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    with pytest.raises((ValueError, KeyError)):
        with scope.context:
            scope.worker({"model": model})
    assert not control.calls
    _native_adapter_layout_unchanged(model, snapshot)
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("outcome", ["engine-error", "interrupt", "rank", "receipt", "metadata-after-load"])
def test_native_original_registered_failure_preserves_primary_and_restores_binding(
    native_original_registered_control, monkeypatch, outcome
):
    from arctic_platform.correctness.harness import native_adapter

    control = native_original_registered_control
    scope = control.scope(rank=1 if outcome == "rank" else 0)
    model = control.model
    _native_adapter_layout_forbid_state_apis(model, monkeypatch)
    primary = KeyboardInterrupt("owned engine interrupted") if outcome == "interrupt" else RuntimeError("owned engine")
    if outcome in {"engine-error", "interrupt"}:
        control.engine_failure = primary
    if outcome == "metadata-after-load":
        original = native_adapter.initialize_adapter

        def initialize(*args):
            original(*args)
            (Path(args[1]) / "initialization.json").write_text("{}")

        monkeypatch.setattr(native_adapter, "initialize_adapter", initialize)
    expected = type(primary) if control.engine_failure is not None else ValueError
    with pytest.raises(expected) as caught:
        with scope.context as receipt:
            scope.worker({"model": model})
            if outcome == "receipt":
                receipt["rank"] = 19
    if control.engine_failure is not None:
        assert caught.value is primary
    assert len(control.calls) == int(outcome in {"engine-error", "interrupt", "receipt"})
    _native_adapter_layout_scope_restored(control, scope)


def test_native_original_registered_source_guard_requires_native_loader(native_original_registered_control):
    control = native_original_registered_control
    scope = control.scope()
    sources = control.native._runtime_sources(scope.training)
    assert set(sources) == {"training", "engine", "adapter", "lora", "composition", "native_adapter"}
    control.native._verify_runtime(scope.training, sources)
    for damage in ("missing", "changed"):
        altered = copy.deepcopy(sources)
        if damage == "missing":
            del altered["native_adapter"]
        else:
            altered["native_adapter"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="runtime differs"):
            control.native._verify_runtime(scope.training, altered)
    _native_adapter_layout_scope_restored(control, scope)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("namespace", ["shared", "detached"])
@pytest.mark.parametrize("receipt_failure", [False, True])
def test_native_original_registered_coordinator_preserves_factor_receipts_and_owned_teardown(
    native_original_registered_control, monkeypatch, wrapped, namespace, receipt_failure
):
    adapter = native_original_registered_control
    scope = adapter.scope(wrapped=wrapped)

    def engine(model, ds_config, *, optimizer, ep_size, mpu, model_path, rank):
        assert model is adapter.model and rank == 0
        adapter.verify(model)
        adapter.calls.append(model)
        return adapter.result

    adapter.engine = scope.training.initialize_training_engine = engine
    _native_adapter_layout_forbid_state_apis(adapter.model, monkeypatch)
    control = native_spawn_globals_control.__wrapped__(adapter, monkeypatch)
    primary = RuntimeError("registered-factor receipt delivery failed") if receipt_failure else None
    control.receipt_failure = primary
    coordinator = control.coordinator(namespace)
    with pytest.raises(RuntimeError) if primary is not None else contextlib.nullcontext():
        asyncio.run(
            coordinator._spawn_workers(control.config, control.ranks, "owned-master", 12345, control.source_stage)
        )
    control.restored()
    assert adapter.calls == [adapter.model]
    adapter.verify(adapter.model)
    assert control.events == [
        "native-spawn",
        "native-constructor",
        "loaded-original",
        "native-engine",
        "remote-receipt",
    ] + (["owned-kill"] if primary is not None else [])
    assert control.gets == [control.ref]
    if primary is not None:
        assert coordinator._workers == []
        assert not hasattr(coordinator, "_correctness_original_adapter_receipts")
    else:
        assert coordinator._correctness_original_adapter_receipts == control.expected_receipts
        receipt = control.workers[0].original_adapter_receipt()
        receipt["rank"] = 99
        assert control.workers[0].original_adapter_receipt() == control.expected_receipts[0]


# GLM sparse attention must retain LoRA gradients when absorbing the KV projection.


@pytest.mark.parametrize("adapter", ["initial-zero-b", "trained-b", "plain"])
def test_glm_sparse_absorbed_peft_projection_matches_explicit_gradients_and_adamw(adapter):
    from peft.tuners.lora.layer import Linear

    from arctic_platform.model.implementations.glm53 import context_parallel

    with torch.random.fork_rng():
        torch.manual_seed(47)
        attention = nn.Module()
        attention.num_heads = 2
        attention.qk_nope_head_dim = 3
        attention.qk_rope_head_dim = 0
        attention.qk_head_dim = 3
        attention.v_head_dim = 2
        attention.kv_lora_rank = 4
        attention.scaling = 3**-0.5
        attention.q_a_proj = nn.Linear(6, 4, bias=False)
        attention.q_a_layernorm = nn.Identity()
        attention.q_b_proj = nn.Linear(4, 6, bias=False)
        attention.kv_a_proj_with_mqa = nn.Linear(6, 4, bias=False)
        attention.kv_a_layernorm = nn.Identity()
        attention.kv_b_proj = nn.Linear(4, 10, bias=False)
        attention.o_proj = nn.Linear(4, 6, bias=False)
        attention.requires_grad_(False)
        if adapter == "plain":
            attention.kv_b_proj.requires_grad_(True)
        else:
            attention.kv_b_proj = Linear(
                attention.kv_b_proj,
                adapter_name="default",
                r=2,
                lora_alpha=2,
                lora_dropout=0.0,
                init_lora_weights=True,
            )
            if adapter == "trained-b":
                with torch.no_grad():
                    attention.kv_b_proj.lora_B["default"].weight.normal_(std=0.1)
        attention.indexer = None
        attention.next_skip_topk = False
        attention._cp_group = None
        attention._cp_world_size = 1
        attention._cp_rank = 0
        attention._ap_sparse_mla_backend = "sparse_mla"
        reference = copy.deepcopy(attention)
        hidden = torch.randn(1, 5, 6, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        upstream_grad = torch.randn(1, 5, 6)

    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    indices = torch.arange(5).expand(5, -1).masked_fill(~causal, -1)[None]
    actual, _, _ = context_parallel._sparse_attention_forward(attention, hidden, None, prev_topk_indices=indices)
    query = reference.q_b_proj(reference.q_a_layernorm(reference.q_a_proj(reference_hidden)))
    query = query.view(1, 5, 2, 3)
    latent = reference.kv_a_layernorm(reference.kv_a_proj_with_mqa(reference_hidden))
    expanded = reference.kv_b_proj(latent).view(1, 5, 2, 5)
    key, value = expanded.split([3, 2], dim=-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * reference.scaling
    scores = scores.masked_fill(~causal, float("-inf"))
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), value)
    expected = reference.o_proj(expected.reshape(1, 5, 4))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.backward(upstream_grad)
    expected.backward(upstream_grad)

    projection = attention.kv_b_proj
    reference_projection = reference.kv_b_proj
    if adapter == "plain":
        torch.testing.assert_close(projection.weight.grad, reference_projection.weight.grad, rtol=1e-5, atol=1e-6)
    else:
        for factor in ("A", "B"):
            parameter = getattr(projection, "lora_" + factor)["default"].weight
            reference_parameter = getattr(reference_projection, "lora_" + factor)["default"].weight
            assert parameter.grad is not None, f"absorbed KV projection disconnected LoRA {factor}"
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-5, atol=1e-6)
        assert reference_projection.lora_B["default"].weight.grad.norm() > 0
        assert projection.base_layer.weight.grad is None
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)

    parameters = [parameter for parameter in attention.parameters() if parameter.requires_grad]
    reference_parameters = [parameter for parameter in reference.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    reference_optimizer = torch.optim.AdamW(reference_parameters, lr=1e-4, weight_decay=0.0)
    optimizer.step()
    reference_optimizer.step()
    for parameter, reference_parameter in zip(parameters, reference_parameters, strict=True):
        torch.testing.assert_close(parameter, reference_parameter, rtol=1e-5, atol=1e-6)
        for moment in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[parameter][moment],
                reference_optimizer.state[reference_parameter][moment],
                rtol=1e-5,
                atol=1e-6,
            )


@pytest.fixture
def owned_worker_lease_new_control():
    from arctic_platform.correctness.harness import native_lora

    events = []

    class Process:
        def __init__(self, pid, *, status=0, wait_error=None, close_error=None):
            self.pid = pid
            self.status = status
            self.wait_error = wait_error
            self.close_error = close_error
            self.returncode = None
            self.timeouts = []
            self.stdin = SimpleNamespace(close=self.close)

        def close(self):
            events.append(("close", self.pid))
            if self.close_error:
                raise self.close_error

        def wait(self, *, timeout):
            events.append(("wait", self.pid))
            self.timeouts.append(timeout)
            if self.wait_error:
                raise self.wait_error
            self.returncode = self.status
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append(("terminate", self.pid))
            self.returncode = -15

        def kill(self):
            events.append(("kill", self.pid))
            self.returncode = -9

    return SimpleNamespace(native=native_lora, Process=Process, events=events)


@pytest.mark.parametrize("method", ["terminate", "kill"])
def test_owned_worker_lease_new_all_workers_exit_before_head_signal(owned_worker_lease_new_control, method):
    case = owned_worker_lease_new_control
    leases = case.native._OwnedWorkerLeases()
    leases.workers = [case.Process(11), case.Process(12)]
    head = case.native._OwnedZoneProcess(case.Process(10), leases, head=True)
    getattr(head, method)()
    leases.close()
    assert case.events == [("close", 11), ("close", 12), ("wait", 11), ("wait", 12), (method, 10)]
    assert leases.errors == [] and head.pid == 10 and head.poll() in (-15, -9)


@pytest.mark.parametrize("failure", ["timeout", "ssh-exit", "close", "interrupt"])
def test_owned_worker_lease_new_failures_never_signal_head_or_worker(owned_worker_lease_new_control, failure):
    case = owned_worker_lease_new_control
    wait_error = subprocess.TimeoutExpired("owned ssh", 30) if failure == "timeout" else None
    if failure == "interrupt":
        wait_error = KeyboardInterrupt("shutdown interrupted")
    worker = case.Process(
        11,
        status=255 if failure == "ssh-exit" else 0,
        wait_error=wait_error,
        close_error=OSError("lease close failed") if failure == "close" else None,
    )
    leases = case.native._OwnedWorkerLeases()
    leases.workers = [worker, case.Process(12)]
    head = case.native._OwnedZoneProcess(case.Process(10), leases, head=True)
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt, match="shutdown interrupted"):
            head.terminate()
    else:
        head.terminate()
    head.kill()
    case.native._OwnedZoneProcess(worker, leases).terminate()
    case.native._OwnedZoneProcess(worker, leases).kill()
    assert case.events[:2] == [("close", 11), ("close", 12)]
    assert not any(event[0] in ("terminate", "kill") for event in case.events)
    assert leases.errors and leases.closed
    if failure != "interrupt":
        assert ("wait", 12) in case.events


def test_owned_worker_lease_new_one_shared_deadline(owned_worker_lease_new_control, monkeypatch):
    case = owned_worker_lease_new_control
    times = iter([100.0, 110.0, 150.0])
    monkeypatch.setattr(case.native.time, "monotonic", lambda: next(times))
    leases = case.native._OwnedWorkerLeases()
    leases.workers = [case.Process(11), case.Process(12)]
    leases.close()
    assert [worker.timeouts for worker in leases.workers] == [[20.0], [0.0]]


@pytest.mark.parametrize("outcome", ["normal", "primary", "interrupt", "spawn", "timeout", "ssh-exit", "head-exit"])
def test_owned_worker_lease_new_zone_launch_and_primary_failure(owned_worker_lease_new_control, monkeypatch, outcome):
    import shlex

    case = owned_worker_lease_new_control
    server = ModuleType("dss.ray_dss.zones.server")
    listener = SimpleNamespace(fileno=lambda: 43, close=lambda: case.events.append(("listener-close",)))
    monkeypatch.setattr(case.native, "_zone_listener", lambda: contextlib.nullcontext(listener))

    def original_head_argv():
        return [sys.executable, "-m", "dss.ray_dss.zones.server_head", "--port", "9001"]

    server._head_argv = original_head_argv
    server.is_local = lambda host: host == "owned-head"
    failure = KeyboardInterrupt("primary") if outcome == "interrupt" else RuntimeError("primary")
    head_process = case.Process(10)
    worker_process = case.Process(
        11,
        status=255 if outcome == "ssh-exit" else 0,
        wait_error=subprocess.TimeoutExpired("owned worker", 30) if outcome == "timeout" else None,
    )
    launches = []

    def original_run(command, host, **kwargs):
        argv = shlex.split(command)
        launches.append((argv, host, kwargs))
        if argv[4] == "head":
            assert argv[0:4] == ["exec", sys.executable, "-m", case.native._MODULE]
            assert kwargs == {"pass_fds": (43,)}
            return head_process
        assert argv[:4] == ["exec", "setsid", "--wait", "env"]
        assert argv[4].startswith("DSS_CORRECTNESS_GATEWAY_OWNER=") and len(argv[4].split("=", 1)[1]) == 32
        assert argv[5:10] == ["SPT_NOENV=1", sys.executable, "-m", case.native._MODULE, "worker"]
        assert argv[10:] == ["--zone-id", "owned-zone", "--server-url", "owned-head:1234"]
        assert kwargs == {"stdin": subprocess.PIPE, "stdout": "retained-log"}
        if outcome == "spawn":
            raise failure
        return worker_process

    server.run_cmd = original_run

    def main():
        head = server.run_cmd(shlex.join(server._head_argv()), "owned-head")
        worker = None
        try:
            command = shlex.join(
                [
                    sys.executable,
                    "-m",
                    "dss.ray_dss.zones.server_worker",
                    "--zone-id",
                    "owned-zone",
                    "--server-url",
                    "owned-head:1234",
                ]
            )
            worker = server.run_cmd(command, "owned-peer", stdout="retained-log")
            if outcome == "head-exit":
                head_process.returncode = 1
            if outcome in ("primary", "interrupt"):
                raise failure
        finally:
            if head.poll() is None:
                head.terminate()
            if worker is not None and worker.poll() is None:
                worker.terminate()

    server.main = main
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.zones.server": server})
    monkeypatch.setattr(sys, "argv", ["owned", "zone"])
    if outcome in ("primary", "interrupt", "spawn"):
        with pytest.raises(type(failure)) as caught:
            case.native.main()
        assert caught.value is failure
    elif outcome in ("timeout", "ssh-exit"):
        with pytest.raises(RuntimeError, match="Owned worker shutdown failed"):
            case.native.main()
        assert ("terminate", 10) not in case.events
    else:
        case.native.main()
    assert server.run_cmd is original_run and server._head_argv is original_head_argv
    assert len(launches) == 2
    if outcome != "spawn":
        assert ("close", 11) in case.events
        assert not any(event in case.events for event in [("terminate", 11), ("kill", 11)])
    if outcome == "head-exit":
        assert case.events[-2:] == [("close", 11), ("wait", 11)]


def test_owned_worker_lease_new_primary_keeps_cleanup_failure_note(owned_worker_lease_new_control, monkeypatch):
    case = owned_worker_lease_new_control
    server = ModuleType("dss.ray_dss.zones.server")

    def original_argv():
        return []

    def original_run(*args, **kwargs):
        return None

    server._head_argv, server.run_cmd = original_argv, original_run
    primary = SystemExit(7)
    leases = case.native._OwnedWorkerLeases()
    leases.workers = [case.Process(11, status=255)]
    monkeypatch.setattr(case.native, "_OwnedWorkerLeases", lambda: leases)
    monkeypatch.setattr(case.native, "_zone_listener", lambda: contextlib.nullcontext(object()))

    def main():
        raise primary

    server.main = main
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.zones.server": server})
    monkeypatch.setattr(sys, "argv", ["owned", "zone"])
    with pytest.raises(SystemExit) as caught:
        case.native.main()
    assert caught.value is primary and caught.value.code == 7
    assert len(primary.__notes__) == 1 and "exit 255" in primary.__notes__[0]
    assert server.run_cmd is original_run and server._head_argv is original_argv


@pytest.mark.parametrize("early_eof", [False, True])
def test_owned_worker_lease_new_eof_requests_registered_shutdown_without_fd_inheritance(monkeypatch, early_eof):
    import signal
    import threading

    from arctic_platform.correctness.harness import native_lora

    read_fd, write_fd = os.pipe()
    original_dup = os.dup
    original_dup2 = os.dup2
    original_set_inheritable = os.set_inheritable
    events = []
    stop = threading.Event()
    previous = object()
    signals = SimpleNamespace(signal=lambda *args: previous)
    worker = SimpleNamespace(signal=signals)
    monkeypatch.setattr(native_lora.os, "dup", lambda descriptor: original_dup(read_fd))
    monkeypatch.setattr(native_lora.os, "dup2", lambda source, target: events.append(("redirect", target)))

    def set_inheritable(descriptor, value):
        original_set_inheritable(descriptor, value)
        events.append(("inherit", os.get_inheritable(descriptor)))

    monkeypatch.setattr(native_lora.os, "set_inheritable", set_inheritable)
    if early_eof:
        os.close(write_fd)

    def main():
        assert worker.signal.signal(signal.SIGTERM, lambda *args: stop.set()) is previous
        if not early_eof:
            os.write(write_fd, b"ignored lease data")
            os.close(write_fd)
        assert stop.wait(2.0), "EOF did not reach the upstream stop callback"

    worker.main = main
    try:
        native_lora._owned_worker_main(worker)
    finally:
        os.close(read_fd)
        monkeypatch.setattr(native_lora.os, "dup2", original_dup2)
    assert worker.signal is signals
    assert events == [("inherit", False), ("redirect", 0)]


@pytest.mark.parametrize("primary_failure", [False, True])
def test_owned_worker_lease_new_reader_failure_preserves_primary(monkeypatch, primary_failure):
    import threading

    from arctic_platform.correctness.harness import native_lora

    read_fd, write_fd = os.pipe()
    original_dup = os.dup
    failed = threading.Event()
    failure = RuntimeError("worker primary")
    signals = SimpleNamespace(signal=lambda *args: None)
    worker = SimpleNamespace(signal=signals)
    monkeypatch.setattr(native_lora.os, "dup", lambda descriptor: original_dup(read_fd))
    monkeypatch.setattr(native_lora.os, "dup2", lambda *args: None)

    def select_failure(*args):
        failed.set()
        raise OSError("lease reader broken")

    monkeypatch.setattr(native_lora.select, "select", select_failure)

    def main():
        assert failed.wait(2.0)
        if primary_failure:
            raise failure

    worker.main = main
    try:
        with pytest.raises(RuntimeError) as caught:
            native_lora._owned_worker_main(worker)
        if primary_failure:
            assert caught.value is failure and "lease reader broken" in failure.__notes__[0]
        else:
            assert "lease reader broken" in str(caught.value)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert worker.signal is signals


def test_owned_worker_lease_new_worker_role_dispatches_without_changing_args(monkeypatch):
    from arctic_platform.correctness.harness import native_lora

    worker = ModuleType("dss.ray_dss.zones.server_worker")
    calls = []
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.zones.server_worker": worker})
    monkeypatch.setattr(native_lora, "_owned_worker_session", lambda actual: calls.append(actual))
    monkeypatch.setattr(sys, "argv", ["owned", "worker", "--zone-id", "owned-zone"])
    native_lora.main()
    assert calls == [worker] and sys.argv == ["owned", "--zone-id", "owned-zone"]


@pytest.mark.parametrize("primary_failure", [False, True])
def test_owned_worker_session_cleanup_new_waits_for_descendants(monkeypatch, primary_failure):
    from arctic_platform.correctness.harness import dss_driver
    from arctic_platform.correctness.harness import native_lora

    token = "a" * 32
    identity = {"pid": os.getpid(), "uid": os.getuid(), "start_time": "123"}
    events = []
    worker = object()
    primary = RuntimeError("worker primary")
    monkeypatch.setenv("DSS_CORRECTNESS_GATEWAY_OWNER", token)
    monkeypatch.setenv("SPT_NOENV", "1")
    monkeypatch.setattr(native_lora.os, "getsid", lambda pid: os.getpid())
    monkeypatch.setattr(dss_driver, "_gateway_process_identity", lambda pid: identity)

    def main(actual):
        assert actual is worker
        events.append("upstream-exit")
        if primary_failure:
            raise primary

    monkeypatch.setattr(native_lora, "_owned_worker_main", main)
    monkeypatch.setattr(
        dss_driver,
        "_reap_owned_gateway_descendants",
        lambda actual, marker: events.append(("descendants-exit", actual, marker)),
    )
    if primary_failure:
        with pytest.raises(RuntimeError) as caught:
            native_lora._owned_worker_session(worker)
        assert caught.value is primary
    else:
        native_lora._owned_worker_session(worker)
    assert events == ["upstream-exit", ("descendants-exit", identity, token)]


@pytest.mark.parametrize("primary_failure", [False, True])
def test_owned_worker_session_cleanup_new_retains_cleanup_failure(monkeypatch, primary_failure):
    from arctic_platform.correctness.harness import dss_driver
    from arctic_platform.correctness.harness import native_lora

    monkeypatch.setenv("DSS_CORRECTNESS_GATEWAY_OWNER", "a" * 32)
    monkeypatch.setenv("SPT_NOENV", "1")
    monkeypatch.setattr(native_lora.os, "getsid", lambda pid: os.getpid())
    monkeypatch.setattr(
        dss_driver,
        "_gateway_process_identity",
        lambda pid: {"pid": pid, "uid": os.getuid(), "start_time": "123"},
    )
    primary = RuntimeError("worker primary")
    cleanup = dss_driver.UnsafeCleanupError("unverified descendant")

    def main(worker):
        if primary_failure:
            raise primary

    def reap(*args):
        raise cleanup

    monkeypatch.setattr(native_lora, "_owned_worker_main", main)
    monkeypatch.setattr(dss_driver, "_reap_owned_gateway_descendants", reap)
    with pytest.raises(RuntimeError) as caught:
        native_lora._owned_worker_session(object())
    if primary_failure:
        assert caught.value is primary
        assert "unverified descendant" in primary.__notes__[0]
    else:
        assert caught.value is cleanup


@pytest.mark.parametrize("invalid", ["token", "retitle", "session", "identity"])
def test_owned_worker_session_cleanup_new_rejects_unowned_launch(monkeypatch, invalid):
    from arctic_platform.correctness.harness import dss_driver
    from arctic_platform.correctness.harness import native_lora

    monkeypatch.setenv("DSS_CORRECTNESS_GATEWAY_OWNER", "a" * 32 if invalid != "token" else "")
    monkeypatch.setenv("SPT_NOENV", "1" if invalid != "retitle" else "0")
    monkeypatch.setattr(native_lora.os, "getsid", lambda pid: os.getpid() + (invalid == "session"))
    monkeypatch.setattr(
        dss_driver,
        "_gateway_process_identity",
        lambda pid: None if invalid == "identity" else {"pid": pid, "uid": os.getuid(), "start_time": "123"},
    )
    monkeypatch.setattr(native_lora, "_owned_worker_main", lambda worker: pytest.fail("unowned worker executed"))
    monkeypatch.setattr(
        dss_driver,
        "_reap_owned_gateway_descendants",
        lambda *args: pytest.fail("unowned process signalled"),
    )
    with pytest.raises(RuntimeError):
        native_lora._owned_worker_session(object())


@pytest.mark.parametrize("primary_failure", [False, True])
def test_owned_worker_session_cleanup_new_real_retitled_child_exits_before_worker(primary_failure):
    import textwrap

    source = textwrap.dedent(
        """
        import signal
        import subprocess
        import sys
        from types import SimpleNamespace
        from arctic_platform.correctness.harness import native_lora

        children = []
        child_source = (
            "import signal, time, setproctitle;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "setproctitle.setproctitle('ray::OwnedWorkerCleanupTest');"
            "print('ready', flush=True);"
            "time.sleep(120)"
        )
        worker = SimpleNamespace(signal=signal)
        primary = RuntimeError("worker primary")

        def main():
            child = subprocess.Popen([sys.executable, "-c", child_source], stdout=subprocess.PIPE, text=True)
            children.append(child)
            assert child.stdout.readline().strip() == "ready"
            print("OWNED_WORKER_CHILD " + str(child.pid), flush=True)
            if sys.argv[1] == "primary":
                raise primary

        worker.main = main
        try:
            try:
                native_lora._owned_worker_session(worker)
            except RuntimeError as error:
                assert sys.argv[1] == "primary" and error is primary
            assert len(children) == 1 and children[0].wait(timeout=2) == -signal.SIGKILL
            print("OWNED_WORKER_DESCENDANTS_EXITED", flush=True)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
    """
    )
    completed = subprocess.run(
        [
            "setsid",
            "--wait",
            "env",
            "DSS_CORRECTNESS_GATEWAY_OWNER=" + "b" * 32,
            "SPT_NOENV=1",
            sys.executable,
            "-c",
            source,
            "primary" if primary_failure else "normal",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=45,
    )
    print(completed.stdout, end="")
    print(completed.stderr, end="")
    assert completed.returncode == 0
    assert "OWNED_WORKER_DESCENDANTS_EXITED" in completed.stdout


@pytest.mark.parametrize("phase", ["inherit", "redirect", "thread-start"])
def test_owned_worker_lease_setup_new_preserves_error_and_closes_descriptor(monkeypatch, phase):
    from arctic_platform.correctness.harness import native_lora

    primary = OSError(f"setup {phase}")
    signals = object()
    worker = SimpleNamespace(signal=signals, main=lambda: pytest.fail("failed setup invoked worker"))
    closed = []
    monkeypatch.setattr(native_lora.os, "dup", lambda descriptor: 990)
    monkeypatch.setattr(native_lora.os, "close", lambda descriptor: closed.append(descriptor))

    def setup_step(selected):
        if selected == phase:
            raise primary

    monkeypatch.setattr(native_lora.os, "set_inheritable", lambda *args: setup_step("inherit"))
    monkeypatch.setattr(native_lora.os, "dup2", lambda *args: setup_step("redirect"))
    monkeypatch.setattr(
        native_lora.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(
            start=lambda: setup_step("thread-start"),
            join=lambda **kwargs: pytest.fail("unstarted lease thread was joined"),
        ),
    )
    with pytest.raises(OSError) as caught:
        native_lora._owned_worker_main(worker)
    assert caught.value is primary and closed == [990] and worker.signal is signals


@pytest.mark.parametrize("primary_type", [None, RuntimeError, SystemExit, KeyboardInterrupt])
def test_owned_worker_lease_setup_new_close_failure_never_replaces_primary(monkeypatch, primary_type):
    from arctic_platform.correctness.harness import native_lora

    primary = primary_type("worker primary") if primary_type is not None else None
    cleanup = OSError("descriptor close failed")
    signals = object()
    worker = SimpleNamespace(signal=signals)
    monkeypatch.setattr(native_lora.os, "dup", lambda descriptor: 990)
    monkeypatch.setattr(native_lora.os, "set_inheritable", lambda *args: None)
    monkeypatch.setattr(native_lora.os, "dup2", lambda *args: None)
    monkeypatch.setattr(
        native_lora.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kwargs: None, is_alive=lambda: False),
    )

    def close(descriptor):
        raise cleanup

    def main():
        if primary is not None:
            raise primary

    worker.main = main
    monkeypatch.setattr(native_lora.os, "close", close)
    with pytest.raises(type(primary) if primary is not None else OSError) as caught:
        native_lora._owned_worker_main(worker)
    assert caught.value is (primary if primary is not None else cleanup)
    if primary is not None:
        assert primary.__notes__ == ["Owned worker lease cleanup also failed: descriptor close failed"]
    assert worker.signal is signals


def test_owned_worker_lease_setup_new_read_error_before_registration_stops_worker(monkeypatch):
    import signal
    import threading

    from arctic_platform.correctness.harness import native_lora

    read_fd, write_fd = os.pipe()
    original_dup = os.dup
    failed = threading.Event()
    stop = threading.Event()
    signals = SimpleNamespace(signal=lambda *args: None)
    worker = SimpleNamespace(signal=signals)
    monkeypatch.setattr(native_lora.os, "dup", lambda descriptor: original_dup(read_fd))
    monkeypatch.setattr(native_lora.os, "dup2", lambda *args: None)

    def broken_reader(*args):
        failed.set()
        raise OSError("early lease read failure")

    def main():
        assert failed.wait(2.0)
        worker.signal.signal(signal.SIGTERM, lambda *args: stop.set())
        assert stop.wait(2.0)

    monkeypatch.setattr(native_lora.select, "select", broken_reader)
    worker.main = main
    try:
        with pytest.raises(RuntimeError, match="early lease read failure"):
            native_lora._owned_worker_main(worker)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert worker.signal is signals


def test_owned_worker_lease_setup_new_unfinished_reader_is_cleanup_failure(monkeypatch):
    from arctic_platform.correctness.harness import native_lora

    signals = object()
    worker = SimpleNamespace(signal=signals, main=lambda: None)
    closed = []
    monkeypatch.setattr(native_lora.os, "dup", lambda descriptor: 990)
    monkeypatch.setattr(native_lora.os, "set_inheritable", lambda *args: None)
    monkeypatch.setattr(native_lora.os, "dup2", lambda *args: None)
    monkeypatch.setattr(native_lora.os, "close", lambda descriptor: closed.append(descriptor))
    monkeypatch.setattr(
        native_lora.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kwargs: None, is_alive=lambda: True),
    )
    with pytest.raises(RuntimeError, match="lease reader did not stop"):
        native_lora._owned_worker_main(worker)
    assert closed == [990] and worker.signal is signals


def test_owned_worker_lease_interrupt_new_closes_other_writers_and_never_signals_head(
    owned_worker_lease_new_control,
):
    case = owned_worker_lease_new_control
    interruption = KeyboardInterrupt("close interrupted")
    leases = case.native._OwnedWorkerLeases()
    leases.workers = [case.Process(11, close_error=interruption), case.Process(12)]
    head = case.native._OwnedZoneProcess(case.Process(10), leases, head=True)
    with pytest.raises(KeyboardInterrupt) as caught:
        head.terminate()
    assert caught.value is interruption
    assert case.events == [("close", 11), ("close", 12), ("wait", 11), ("wait", 12)]
    head.kill()
    assert ("kill", 10) not in case.events and len(leases.errors) == 1


@pytest.mark.parametrize("close_interrupted", [False, True])
def test_owned_worker_lease_wait_interrupt_new_attempts_every_worker_and_preserves_first(
    owned_worker_lease_new_control, close_interrupted
):
    case = owned_worker_lease_new_control
    close_error = KeyboardInterrupt("close interrupted") if close_interrupted else None
    wait_error = SystemExit("wait interrupted")
    leases = case.native._OwnedWorkerLeases()
    leases.workers = [
        case.Process(11, close_error=close_error, wait_error=wait_error),
        case.Process(12, wait_error=KeyboardInterrupt("second wait interrupted")),
        case.Process(13),
    ]
    head = case.native._OwnedZoneProcess(case.Process(10), leases, head=True)
    first_error = close_error if close_interrupted else wait_error
    with pytest.raises(type(first_error)) as caught:
        head.terminate()
    assert caught.value is first_error
    assert case.events == [
        ("close", 11),
        ("close", 12),
        ("close", 13),
        ("wait", 11),
        ("wait", 12),
        ("wait", 13),
    ]
    assert len(leases.errors) == (3 if close_interrupted else 2)
    head.kill()
    assert not any(event[0] in ("terminate", "kill") for event in case.events)


@pytest.mark.parametrize("primary_present", [False, True])
def test_owned_worker_lease_zone_interrupt_new_restores_hooks_and_retains_primary(
    owned_worker_lease_new_control, monkeypatch, primary_present
):
    case = owned_worker_lease_new_control
    server = ModuleType("dss.ray_dss.zones.server")

    def original_argv():
        return []

    def original_run(*args, **kwargs):
        return None

    server._head_argv, server.run_cmd = original_argv, original_run
    primary = RuntimeError("zone primary")
    interruption = KeyboardInterrupt("lease close interrupted")
    leases = case.native._OwnedWorkerLeases()
    leases.workers = [case.Process(11, close_error=interruption), case.Process(12)]
    monkeypatch.setattr(case.native, "_OwnedWorkerLeases", lambda: leases)
    monkeypatch.setattr(case.native, "_zone_listener", lambda: contextlib.nullcontext(object()))

    def main():
        if primary_present:
            raise primary

    server.main = main
    _native_original_composition_modules(monkeypatch, **{"dss.ray_dss.zones.server": server})
    monkeypatch.setattr(sys, "argv", ["owned", "zone"])
    with pytest.raises(RuntimeError if primary_present else KeyboardInterrupt) as caught:
        case.native.main()
    assert caught.value is (primary if primary_present else interruption)
    if primary_present:
        assert len(primary.__notes__) == 2
        assert "lease close interrupted" in primary.__notes__[0]
        assert "KeyboardInterrupt" in primary.__notes__[1]
    assert case.events == [("close", 11), ("close", 12), ("wait", 11), ("wait", 12)]
    assert server.run_cmd is original_run and server._head_argv is original_argv
