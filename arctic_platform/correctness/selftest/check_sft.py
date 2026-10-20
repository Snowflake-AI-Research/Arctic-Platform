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

"""Shared SFT helpers preserve compatibility, scoring alignment, and configuration values."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import e2e_sft_train_validate as e2e
from arctic_platform.correctness.checks import inference_checkpoint_loss as inference
from arctic_platform.correctness.harness import gsm8k
from arctic_platform.correctness.harness import sft


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
def test_check_compatibility_exports_are_shared_objects(name):
    assert getattr(e2e, name) is getattr(sft, name)
    assert getattr(inference, name) is getattr(sft, name)


def test_e2e_shared_helpers_do_not_import_another_check():
    tree = ast.parse(Path(e2e.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(node.level == 1 or "inference_checkpoint_loss" in (node.module or "") for node in imports)
    names = {alias.name for node in imports if node.module == "harness.sft" for alias in node.names}
    assert names == {
        "TRAIN_SPLIT",
        "VALIDATION_SPLIT",
        "configured_learning_rate",
        "configured_rows_per_step",
        "cross_entropy",
        "reference_answer_logprobs",
    }


def test_shared_sft_has_no_check_dependency():
    tree = ast.parse(Path(sft.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in imports:
        assert "checks" not in (getattr(node, "module", None) or "").split(".")
        assert all("checks" not in alias.name.split(".") for alias in node.names)


def test_shared_dataset_locations_are_unchanged():
    assert sft.TRAIN_SPLIT == Path("/code/shared/gsm8k/train.parquet")
    assert sft.VALIDATION_SPLIT == Path("/code/shared/gsm8k/test.parquet")


@pytest.mark.parametrize("rate,rows", [(0.00002, 16), ("0.00002", "16"), (0.0, 1)])
def test_configured_settings_are_read_without_defaults(rate, rows):
    cfg = SimpleNamespace(config_id="sft", training={"optimizer": {"lr": rate}, "train_batch_size": rows})
    assert sft.configured_learning_rate(cfg) == float(rate)
    assert sft.configured_rows_per_step(cfg) == int(rows)


@pytest.mark.parametrize("optimizer", [None, "AdamW", {}, {"lr": None}])
def test_missing_configured_learning_rate_keeps_the_original_error(optimizer):
    cfg = SimpleNamespace(config_id="sft", training={"optimizer": optimizer})
    with pytest.raises(ValueError, match="sft: the training config declares no optimizer learning rate"):
        sft.configured_learning_rate(cfg)


@pytest.mark.parametrize("training", [{}, {"train_batch_size": None}])
def test_missing_configured_rows_keeps_the_original_error(training):
    cfg = SimpleNamespace(config_id="sft", training=training)
    with pytest.raises(ValueError, match="sft: the training config declares no train_batch_size"):
        sft.configured_rows_per_step(cfg)


def test_cross_entropy_preserves_scalar_accumulation_order():
    assert sft.cross_entropy([[-1e16, -1.0], [1e16, -2.0]]) == (0.5, 4)


def test_cross_entropy_pools_tokens_and_accepts_empty_individual_rows():
    assert sft.cross_entropy([[-1.0] * 4, [], [-5.0]]) == (1.8, 5)


def test_cross_entropy_rejects_no_scored_tokens():
    with pytest.raises(ValueError, match="not a measurement"):
        sft.cross_entropy([[], []])


def example(n_prompt=2):
    return gsm8k.Example(question="q", prompt="p", completion=" a", input_ids=(1, 2, 3, 4), n_prompt=n_prompt)


def test_reference_span_excludes_prompt_padding_and_unpredicted_position():
    logprobs = torch.tensor([[float("nan"), -1.0, -2.0, float("inf"), float("nan")]])
    assert sft.reference_answer_logprobs(logprobs, [example()]) == [[-1.0, -2.0]]


@pytest.mark.parametrize("shape", [(5,), (2, 5), (1, 1, 5)])
def test_reference_scoring_rejects_non_row_aligned_tensors(shape):
    with pytest.raises(RuntimeError, match="two-dimensional row"):
        sft.reference_answer_logprobs(torch.zeros(shape), [example()])


@pytest.mark.parametrize("n_prompt", [0, 4, 5])
def test_reference_scoring_rejects_invalid_answer_spans(n_prompt):
    with pytest.raises(RuntimeError, match="answer span must follow a prompt token"):
        sft.reference_answer_logprobs(torch.zeros(1, 5), [example(n_prompt)])


def test_reference_scoring_rejects_truncated_answer_response():
    with pytest.raises(RuntimeError, match="covers 1 of the row's 2 answer tokens"):
        sft.reference_answer_logprobs(torch.zeros(1, 2), [example()])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reference_scoring_rejects_nonfinite_answer_tokens(value):
    with pytest.raises(RuntimeError, match="answer log-probabilities must be finite"):
        sft.reference_answer_logprobs(torch.tensor([[0.0, value, -1.0, 0.0]]), [example()])
