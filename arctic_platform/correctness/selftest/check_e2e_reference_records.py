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

"""Prepared E2E scalars reconstruct inputs and never launch HF during regression."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import e2e_sft_train_validate as check
from arctic_platform.correctness.harness import deterministic_lora
from arctic_platform.correctness.harness import reference_records
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome


class Tokenizer:
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


def forbidden(*args, **kwargs):
    raise AssertionError("HF, product initialization, or an extra allocation was reached")


@pytest.fixture
def ctx(tmp_path, monkeypatch):
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
    tokenizer = Tokenizer()
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


def measurements(inputs):
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


def publish(ctx, inputs=None):
    inputs = check.prepare_inputs(ctx) if inputs is None else inputs
    identity = reference_records.workload_identity(ctx, check.TEST_ID, inputs.recipe)
    reference = measurements(inputs)
    path = reference_records.publish_record(ctx, identity, reference)
    return inputs, identity, reference, path


def guard_launches(monkeypatch):
    monkeypatch.setattr(check, "run_reference_trajectory", forbidden)
    monkeypatch.setattr(runner, "materialize_dss_peft_adapter", forbidden)
    monkeypatch.setattr(check.subprocess, "Popen", forbidden)
    monkeypatch.setattr(check, "gateway", forbidden)
    monkeypatch.setattr(check, "running_job", forbidden)


def test_prepared_regression_runs_one_target_and_zero_hf(ctx, monkeypatch):
    inputs, _, reference, _ = publish(ctx)
    guard_launches(monkeypatch)
    monkeypatch.setattr(reference_records, "publish_record", forbidden)
    calls = []

    def target(context, replay, directory, adapter):
        assert context is ctx
        assert adapter is None
        check.validate_replay(inputs)
        calls.append(replay.steps)
        return [2.0] * 100, torch.full(
            (len(inputs.validation_examples), int(inputs.measured["validation_padded_width"])), -1.5
        )

    monkeypatch.setattr(check, "run_dss_trajectory", target)
    result = check.run(ctx)[0]

    assert result.outcome is Outcome.PASS
    assert calls == [100]
    assert result.metrics["gated_quantities"] == 2
    assert result.metrics["train_steps_recorded"] == 100
    assert result.metrics["scored_answer_tokens"] == reference["scored_answer_tokens"]
    diagnostic = json.loads((inputs.workdir / "trajectories.json").read_text())
    assert diagnostic["reference_train_losses"] == reference["train_losses"]
    assert diagnostic["reference_train_gradient_norms"] == reference["train_gradient_norms"]
    assert not any("logprobs" in key for key in diagnostic)
    assert len([key for key in result.metrics if key.startswith("train_loss_abs_diff_step_")]) == 100


def test_missing_record_stops_before_target_and_prints_full_traceback(ctx, monkeypatch, caplog):
    guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)

    with caplog.at_level(logging.ERROR):
        result = check.run(ctx)[0]

    assert result.outcome is Outcome.FAIL
    assert "prepared HF reference missing" in result.summary
    assert "--prepare-references" in result.summary
    assert "Traceback (most recent call last)" in caplog.text


def test_corrupt_record_cannot_fall_back_to_hf(ctx, monkeypatch):
    _, _, _, path = publish(ctx)
    record = json.loads(path.read_text())
    record["measurements"]["train_losses"][0] = 123.0
    path.write_text(json.dumps(record))
    guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)

    with pytest.raises(reference_records.ReferencePrerequisiteError, match="corrupt or incompatible"):
        check._run(ctx)


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
def test_incomplete_or_nonfinite_scalars_stop_before_target(ctx, monkeypatch, key, value):
    inputs = check.prepare_inputs(ctx)
    reference = measurements(inputs)
    reference[key] = value
    guard_launches(monkeypatch)
    monkeypatch.setattr(reference_records, "load_record", lambda context, identity: reference)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)

    with pytest.raises(reference_records.ReferencePrerequisiteError):
        check._run(ctx)


@pytest.mark.parametrize("change", ["dataset", "tokenizer", "model", "grouping", "budget", "gate"])
def test_input_or_workload_drift_requires_a_new_record_before_target(ctx, monkeypatch, change):
    publish(ctx)
    if change == "dataset":
        (ctx.dataset / "train.parquet").write_bytes(b"different ordered source")
    elif change == "tokenizer":
        ctx.tokenizer.offset = 1
    elif change == "model":
        (Path(ctx.spec.model.cache_path) / "model.safetensors").write_bytes(b"different full-model content")
    elif change == "grouping":
        ctx.cfg.training["train_batch_size"] = 1
    elif change == "budget":
        ctx.cfg.max_tokens_per_mb = 256
    else:
        monkeypatch.setattr(check, "TOLERANCE", 0.04)
    guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)

    with pytest.raises(reference_records.ReferencePrerequisiteError, match="prepared HF reference missing"):
        check._run(ctx)


def test_relocated_source_paths_preserve_identity_and_sequential_selection(ctx, tmp_path, monkeypatch):
    first = check.prepare_inputs(ctx)
    staged = tmp_path / "relocated-dataset"
    staged.mkdir()
    for name in ("train.parquet", "test.parquet"):
        (staged / name).write_bytes((ctx.dataset / name).read_bytes())
    monkeypatch.setenv(check.GSM8K_DATA_ROOT_ENV, str(staged))
    second = check.prepare_inputs(ctx)

    assert first.recipe == second.recipe
    assert first.replay.steps == 100
    assert first.recipe["dataset"]["train"] == {"start": 0, "count": 200, "rows_per_step": 2}
    assert first.recipe["dataset"]["train_groups"] == [{"start": step * 2, "count": 2} for step in range(100)]
    assert str(tmp_path) not in json.dumps(first.recipe)
    assert first.recipe["preprocessing"]["labels"] == "answer-only-huggingface"
    assert first.recipe["dataset"]["validation"]["minimum_rows"] == 4


def test_replay_tampering_after_record_load_stops_before_target(ctx, monkeypatch):
    inputs, _, reference, _ = publish(ctx)
    monkeypatch.setattr(check, "prepare_inputs", lambda context: inputs)

    def load_record(context, identity):
        batch = check.load(inputs.replay.train_batch_paths[0])
        batch.labels[0, -1] += 1
        check.save(batch, inputs.replay.train_batch_paths[0])
        return reference

    monkeypatch.setattr(reference_records, "load_record", load_record)
    guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)

    with pytest.raises(reference_records.ReferencePrerequisiteError, match="replay differs"):
        check._run(ctx)


@pytest.mark.parametrize("lora", [False, True])
def test_explicit_preparation_runs_hf_once_publishes_scalars_and_never_runs_target(ctx, monkeypatch, lora):
    ctx.prepare_references = True
    adapter = None
    if lora:
        ctx.cfg.training["peft_config"] = {"r": 4}
        adapter = ctx.workdir / "adapter"
        monkeypatch.setattr(deterministic_lora, "materialize_adapter", lambda *args: adapter)
        monkeypatch.setattr(deterministic_lora, "identity_metadata", lambda path: {"mock": "exact-init"})
    monkeypatch.setattr(deterministic_lora, "bind_payload", forbidden)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)
    monkeypatch.setattr(runner, "materialize_dss_peft_adapter", forbidden)
    calls = []

    def reference(context, replay, directory, prepared_adapter):
        assert prepared_adapter == adapter
        inputs = check.prepare_inputs(context)
        values = measurements(inputs)
        calls.append(replay.steps)
        values.pop("validation_cross_entropy")
        values.pop("scored_answer_tokens")
        values["validation_logprobs"] = torch.full(
            (len(inputs.validation_examples), int(inputs.measured["validation_padded_width"])), -1.5
        ).tolist()
        return values

    monkeypatch.setattr(check, "run_reference_trajectory", reference)

    assert check.run(ctx) == []
    assert calls == [100]
    paths = list(ctx.reference_records_dir.rglob("*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["measurements"]["validation_cross_entropy"] == 1.5
    assert len(record["measurements"]["train_losses"]) == 100
    assert len(record["measurements"]["train_gradient_norms"]) == 100
    assert "validation_logprobs" not in record["measurements"]
    assert list(ctx.reference_records_dir.rglob("*.pt")) == []
    assert list(ctx.reference_records_dir.rglob("*.safetensors")) == []
    monkeypatch.setattr(check, "run_reference_trajectory", forbidden)
    assert check.run(ctx) == []
    assert calls == [100]


def test_preparation_exception_propagates_instead_of_becoming_empty_success(ctx, monkeypatch):
    ctx.prepare_references = True

    def fail(context):
        raise RuntimeError("actual preparation traceback")

    monkeypatch.setattr(check, "prepare_reference_record", fail)
    with pytest.raises(RuntimeError, match="actual preparation traceback"):
        check.run(ctx)


def test_preparation_requires_explicit_mode(ctx, monkeypatch):
    monkeypatch.setattr(check, "prepare_inputs", forbidden)
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="explicit"):
        check.prepare_reference_record(ctx)


def test_corrupt_existing_record_is_not_reprepared(ctx, monkeypatch):
    _, _, _, path = publish(ctx)
    path.write_text("invalid JSON")
    ctx.prepare_references = True
    guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)
    with pytest.raises(reference_records.ReferencePrerequisiteError, match="prepared HF reference corrupt"):
        check.run(ctx)


def test_unsupported_lora_init_fails_before_hf_or_target(ctx, monkeypatch):
    ctx.cfg.training["peft_config"] = {"r": 4}
    monkeypatch.setattr(deterministic_lora, "materialize_adapter", lambda *args: ctx.workdir / "adapter")
    monkeypatch.setattr(deterministic_lora, "identity_metadata", forbidden)
    guard_launches(monkeypatch)
    monkeypatch.setattr(check, "run_dss_trajectory", forbidden)
    with pytest.raises(deterministic_lora.LoraInitializationPrerequisiteError, match="public"):
        check._run(ctx)


def test_staged_dataset_root_must_be_fast_storage(ctx, monkeypatch):
    monkeypatch.setenv(check.GSM8K_DATA_ROOT_ENV, "/code/staged-gsm8k")
    with pytest.raises(ValueError, match="/data-fast"):
        check.dataset_paths()


def test_reference_options_are_shared_with_hf_command(ctx):
    options = check.reference_options(ctx, ctx.spec.model.cache_path)
    command = check.reference_command(ctx, None, Path("replay"), Path("output"), None)
    assert command[command.index("--ce-chunk") + 1] == str(options["ce_chunk"])
    assert command[command.index("--token-budget") + 1] == str(options["token_budget"])
    assert command[command.index("--learning-rate") + 1] == str(options["learning_rate"])
    assert json.loads(command[command.index("--optimizer-config") + 1]) == options["optimizer_config"]
