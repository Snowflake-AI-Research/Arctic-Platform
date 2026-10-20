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
import importlib.util
import inspect
import json
import logging
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
from arctic_platform.correctness.checks import e2e_sft_train_validate as check
from arctic_platform.correctness.checks import e2e_sft_train_validate as e2e
from arctic_platform.correctness.checks import inference_checkpoint_loss as inference
from arctic_platform.correctness.harness import cortex as cortex_fixture
from arctic_platform.correctness.harness import cortex_optimizer as cortex_optimizer_capture
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness import lora as deterministic_lora
from arctic_platform.correctness.harness import lora as lora_init
from arctic_platform.correctness.harness import lora as lora_save0
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import sft
from arctic_platform.correctness.harness import sft as gsm8k
from arctic_platform.correctness.harness import storage as workdir
from arctic_platform.correctness.harness.batches import IGNORE_INDEX
from arctic_platform.correctness.harness.batches import load
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.dss_driver import pack
from arctic_platform.correctness.harness.dss_driver import pack_logit_aligned
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.reference import process_determinism
from arctic_platform.correctness.reference.adamw_trajectory import AdamWTrajectory
from arctic_platform.correctness.reference.hf_single_gpu import build_engine
from arctic_platform.correctness.reference.optimizer_step import run_adamw_step
from arctic_platform.correctness.reference.qwen4_reference import load_qwen4_reference
from arctic_platform.debug import lora as lora_initialization
from arctic_platform.debug.lora import normalize_lora_initialization
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.model.patches.peft import apply_peft_patch
from arctic_platform.model.weight_export import save_peft_adapters
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

from .check_reference_records import _prepared_reference_runner_prepared_runner  # noqa: F401
from .check_reference_records import _prepared_reference_runner_STEP_CHECKS

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
            torch.full((len(inputs.validation_examples), int(inputs.measured["validation_padded_width"])), -1.5),
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
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="prepared HF reference missing"):
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
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="prepared HF reference missing"):
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
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="prepared HF reference missing"):
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
    assert first.recipe["dataset"]["train"] == {"start": 0, "count": 200, "rows_per_step": 2}
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
            (len(inputs.validation_examples), int(inputs.measured["validation_padded_width"])), -1.5
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
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="prepared HF reference corrupt"):
        check.run(_e2e_reference_records_ctx)


def test_e2e_reference_records_unsupported_lora_init_fails_before_hf_or_target(
    _e2e_reference_records_ctx, monkeypatch
):
    _e2e_reference_records_ctx.cfg.training["peft_config"] = {"r": 4}
    monkeypatch.setattr(
        deterministic_lora, "materialize_adapter", lambda *args: _e2e_reference_records_ctx.workdir / "adapter"
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


def test_e2e_reference_records_reference_options_are_shared_with_hf_command(_e2e_reference_records_ctx):
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


def test_e2e_sft_train_validate_mixer_packing_is_selected_from_checkpoint_layers(tmp_path) -> None:
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
        rows, _e2e_sft_train_validate_ROWS_PER_STEP, _e2e_sft_train_validate_STEPS, pad_id=0
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


def test_e2e_sft_train_validate_both_engines_are_handed_the_same_replay_files_in_the_same_order(tmp_path) -> None:
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


def test_e2e_sft_train_validate_the_held_out_request_carries_logit_aligned_labels(tmp_path) -> None:
    import torch

    "The forward-only route shifts nothing, so the same bytes ``/fwd-bwd`` takes would score one off."
    replay = _e2e_sft_train_validate_a_replay(tmp_path)
    validation_batch = load(replay.validation_batch_path)
    _, validation_body = check.request_bodies(replay, model_provider="huggingface")
    _e2e_sft_train_validate_assert_nested_equal(validation_body, pack_logit_aligned(validation_batch))
    assert not torch.equal(
        validation_body["batch"]["labels"], pack(validation_batch, model_provider="huggingface")["batch"]["labels"]
    )


def test_e2e_sft_train_validate_the_training_requests_carry_the_convention_their_dispatch_expects(tmp_path) -> None:
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
        SimpleNamespace(config_id="example"), reference, target, reference_validation, target_validation
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
    target = [2.5, 2.5, 2.5 + _e2e_sft_train_validate_OVER_GATE * 2, 2.5 + _e2e_sft_train_validate_OVER_GATE] + [
        2.5
    ] * 6
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
        [2.5] * 10, [2.5] * 10, reference_validation=1.75, target_validation=1.75 + _e2e_sft_train_validate_OVER_GATE
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


def test_e2e_sft_train_validate_reference_fused_cross_entropy_matches_product_selection(tmp_path) -> None:
    from arctic_platform.correctness.harness.config import LoadedConfig

    def cfg(training):
        return LoadedConfig(
            "example", tmp_path / "example.config", {"model_name": "example", "training_config": training}
        )

    assert cfg({"model_provider": "prime_rl"}).fused_cross_entropy == "liger"
    assert cfg({"model_provider": "prime_rl", "fused_cross_entropy": False}).fused_cross_entropy is False
    assert cfg({"model_provider": "huggingface"}).fused_cross_entropy is False


@pytest.mark.parametrize("name", ["AdamW", "FusedAdam", "torch_adamw"])
def test_e2e_sft_train_validate_reference_uses_independent_unfused_pytorch_adamw(name) -> None:
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
        model, _e2e_sft_train_validate_ADAMW, learning_rate=0.01, gradient_clipping=None, optimizer_dtype="float32"
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


def test_e2e_sft_train_validate_the_single_step_reference_cannot_be_looped(tmp_path) -> None:
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
        question=prompt, prompt=prompt, completion=completion, input_ids=tuple(input_ids), n_prompt=n_prompt
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
    rows = [_gsm8k_example("q1", " a1", [5, 6, 7, 8], 2), _gsm8k_example("q2", " a2", [1, 2, 3], 1)]
    batch = gsm8k.build_batch("validation", rows, pad_id=0)
    assert batch.input_ids.tolist() == [[5, 6, 7, 8], [1, 2, 3, 0]]
    assert batch.labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 7, 8], [IGNORE_INDEX, 2, 3, IGNORE_INDEX]]
    assert batch.active_tokens == sum((row.answer_tokens for row in rows))


def test_gsm8k_the_shift_puts_each_answer_token_on_the_logit_that_predicts_it() -> None:
    """``shifted_labels()[r, s]`` is the token at ``s + 1``, which is what a logit at ``s`` predicts."""
    rows = [_gsm8k_example("q1", " a1", [5, 6, 7, 8], 2)]
    shifted = gsm8k.build_batch("validation", rows, pad_id=0).shifted_labels()
    assert shifted.tolist() == [[IGNORE_INDEX, 7, 8, IGNORE_INDEX]]


def test_gsm8k_positions_are_row_local_and_padding_is_not_numbered() -> None:
    rows = [_gsm8k_example("q1", " a1", [5, 6, 7, 8], 2), _gsm8k_example("q2", " a2", [1, 2, 3], 1)]
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
    rows = [_gsm8k_example("q0", " a", list(range(10)), 1), _gsm8k_example("q1", " a", list(range(10)), 1)]
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


# Lora save0.


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


def test_lora_save0_actual_public_sdk_main_validation_and_loader_forward_option_to_ap():
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
    loader_spec = importlib.util.spec_from_file_location("_correctness_main_model_loader", loader_path)
    assert loader_spec is not None and loader_spec.loader is not None
    main_loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(main_loader)
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
    validated = main_models.TrainingConfig.model_validate(public_config)
    assert validated.peft_config["deterministic_init"] == _lora_save0_OPTIONS
    spec = main_loader.dense_model_spec(
        model_path=payload["model_name"],
        torch_dtype=torch.float32,
        attn_impl="eager",
        model_provider="huggingface",
        sp_size=1,
        training_config=validated.model_dump(exclude_none=True),
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


# Runner lora save0.


@pytest.fixture
def _runner_lora_save0_save0_runner(_prepared_reference_runner_prepared_runner, tmp_path, monkeypatch):  # noqa: F811
    control = _prepared_reference_runner_prepared_runner
    lora_cfg = load_config(Path("arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-lora-4gpus-64k.config"))
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
        "fullmodel",
        "prepare",
        "records",
        "selection",
        "empty",
        "e2e",
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
    elif failure == "fullmodel":
        overrides["local_gateway_full_model"] = True
    elif failure == "prepare":
        overrides["prepare_references"] = True
    elif failure == "records":
        overrides["reference_records_dir"] = None
    elif failure in {"selection", "empty", "e2e"}:
        overrides["selected"] = {"selection": None, "empty": [], "e2e": ["e2e-sft-train-validate"]}[failure]
    elif failure == "publication":
        overrides["cortex_fixture"] = Path("unread-publication.json")
    elif failure == "fixture":
        overrides["local_cortex_fixture"] = False
    with pytest.raises(ValueError):
        control.execute(**overrides)
    assert not control.events and (not control.payloads) and (not control.published)


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
    return gsm8k.Example(question="q", prompt="p", completion=" a", input_ids=(1, 2, 3, 4), n_prompt=n_prompt)


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
