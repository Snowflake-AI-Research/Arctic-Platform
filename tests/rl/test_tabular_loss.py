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
"""CPU tests of the tabular losses: values against the reference fine-tuning loop, and request validation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors.tabular import TabularCrossEntropyLoss
from arctic_platform.rl.processors.tabular import TabularPinballLoss
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close

ROWS, CONTEXT, COLUMNS = 12, 8, 5
QUERY = ROWS - CONTEXT


def request(loss_fn, *, tables=None, classification=True, config=None):
    lead = () if tables is None else (tables,)
    x = torch.randn(*lead, ROWS, COLUMNS)
    if classification:
        y, y_query = torch.randint(0, 3, (*lead, CONTEXT)), torch.randint(0, 3, (*lead, QUERY))
    else:
        y, y_query = torch.randn(*lead, CONTEXT), torch.randn(*lead, QUERY)
    return {
        "x": x,
        "y": y,
        "categorical_mask": torch.zeros(COLUMNS, dtype=torch.bool),
        "context": {"y_query": y_query},
        "processing": {"loss_fn": loss_fn, "config": {"num_classes": 3} if config is None else config},
    }


def test_tabular_losses_are_registered():
    assert isinstance(resolve_loss("tabular_cross_entropy"), TabularCrossEntropyLoss)
    assert isinstance(resolve_loss("tabular_pinball"), TabularPinballLoss)


@pytest.mark.parametrize("tables", [None, 2])
def test_cross_entropy_uses_the_first_num_classes_logits(tables):
    set_seed(0)
    req = request("tabular_cross_entropy", tables=tables)
    lead = () if tables is None else (tables,)
    outputs = torch.randn(*lead, QUERY, 10, requires_grad=True)
    loss_object = resolve_loss("tabular_cross_entropy")
    loss_object.batching_callback(req)

    loss, metrics = loss_object.loss({"outputs": outputs}, req["context"], {}, {"num_classes": 3}, "cpu")

    y_query = req["context"]["y_query"].reshape(-1)
    expected = F.cross_entropy(outputs[..., :3].reshape(-1, 3), y_query)
    torch_assert_close(loss, expected)
    correct = float((outputs[..., :3].reshape(-1, 3).argmax(-1) == y_query).sum().item())
    assert metrics == {
        "tabular_accuracy": pytest.approx(correct / y_query.numel()),
        "tabular_correct_count": correct,
        "tabular_query_row_count": float(y_query.numel()),
    }
    loss.backward()
    assert outputs.grad[..., 3:].abs().sum() == 0
    assert outputs.grad[..., :3].abs().sum() > 0


def test_accuracy_is_rebuilt_from_the_counts_summed_over_workers():
    # Worker metrics arrive merged: additive *_count metrics summed, the rest from worker 0.
    worker_metrics = [
        {"tabular_accuracy": 1.0, "tabular_correct_count": 4.0, "tabular_query_row_count": 4.0},
        {"tabular_accuracy": 0.0, "tabular_correct_count": 0.0, "tabular_query_row_count": 4.0},
    ]
    merged = {"tabular_accuracy": 1.0, "tabular_correct_count": 4.0, "tabular_query_row_count": 8.0}
    resolve_loss("tabular_cross_entropy").metrics_callback(worker_metrics, merged)
    assert merged["tabular_accuracy"] == 0.5


def test_pinball_matches_the_reference_loop():
    set_seed(0)
    req = request("tabular_pinball", classification=False, config={})
    quantiles = torch.randn(QUERY, 999, requires_grad=True)
    loss_object = resolve_loss("tabular_pinball")
    loss_object.batching_callback(req)

    loss, metrics = loss_object.loss({"outputs": quantiles}, req["context"], {}, {}, "cpu")

    # examples/tabular/finetune.py in NVIDIA/structured-data-models.
    diff = req["context"]["y_query"].unsqueeze(-1) - quantiles
    level = torch.linspace(0.001, 0.999, 999)
    torch_assert_close(loss, torch.maximum(level * diff, (level - 1) * diff).mean())
    assert metrics == {"tabular_query_row_count": float(QUERY)}


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["context"].pop("y_query"), "y_query"),
        (lambda r: r["context"].update(y_query=torch.zeros(QUERY + 1, dtype=torch.long)), r"shape \(4,\)"),
        (lambda r: r.update(y=r["y"].float()), "class indices"),
        (lambda r: r["context"].update(y_query=torch.full((QUERY,), 3)), r"\[0, 3\)"),
        (lambda r: r["processing"].update(config={}), "num_classes"),
        (lambda r: r["processing"]["config"].update(temperature=1.0), "Unknown config keys"),
        (lambda r: r.update(y=torch.zeros(ROWS, dtype=torch.long)), "fewer rows than x"),
    ],
)
def test_cross_entropy_rejects_a_malformed_request(mutate, match):
    req = request("tabular_cross_entropy")
    mutate(req)
    with pytest.raises(ValueError, match=match):
        resolve_loss("tabular_cross_entropy").batching_callback(req)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["context"].update(y_query=torch.zeros(QUERY, dtype=torch.long)), "float targets"),
        (lambda r: r["context"].update(y_query=torch.full((QUERY,), float("nan"))), "finite"),
        (lambda r: r["processing"].update(config={"num_classes": 3}), "takes none"),
    ],
)
def test_pinball_rejects_a_malformed_request(mutate, match):
    req = request("tabular_pinball", classification=False, config={})
    mutate(req)
    with pytest.raises(ValueError, match=match):
        resolve_loss("tabular_pinball").batching_callback(req)
