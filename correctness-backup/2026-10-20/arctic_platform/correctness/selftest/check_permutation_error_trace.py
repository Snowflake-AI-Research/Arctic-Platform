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

"""Permutation error observation must leave successful calls and original errors unchanged."""

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.diagnostics import deepep_trace


def test_successful_permutation_is_not_observed(monkeypatch):
    result = object()
    arguments = []

    def permute(*args, **kwargs):
        arguments.append((args, kwargs))
        return result

    module = SimpleNamespace(permute_tokens=permute)
    records = []
    indices = torch.tensor([[0, -1]])
    monkeypatch.setattr(torch.Tensor, "cpu", lambda _: pytest.fail("successful path reads tensor"))
    with deepep_trace.permutation_error_trace(module, records.append):
        assert module.permute_tokens(torch.ones(1, 2), indices, torch.ones(1, 2), 1) is result
    assert module.permute_tokens is permute
    assert arguments[0][0][1] is indices
    assert records == []


@pytest.mark.parametrize(
    "indices",
    [torch.tensor([[0, -1], [-2, 1]]), torch.tensor([[0.0, -1.0]]), torch.empty((0, 2), dtype=torch.long)],
)
def test_failed_permutation_records_indices_without_replacing_error(indices):
    error = RuntimeError("bincount only supports 1-d non-negative integral inputs.")

    def permute(*args):
        raise error

    module = SimpleNamespace(permute_tokens=permute)
    records = []
    with deepep_trace.permutation_error_trace(module, records.append):
        with pytest.raises(RuntimeError) as caught:
            module.permute_tokens(torch.ones(len(indices), 3), indices, torch.ones_like(indices), 2)
    assert caught.value is error
    assert module.permute_tokens is permute
    assert len(records) == 1
    record = records[0]
    assert record["operation"] == "permute_tokens"
    assert record["phase"] == "error"
    assert record["indices"]["shape"] == list(indices.shape)
    assert record["indices"]["dtype"] == str(indices.dtype)
    assert record["num_local_experts"] == 2
    assert record["index_values"]["count"] == indices.numel()
    assert record["index_values"]["other_negative_count"] == int((indices < -1).sum())
    assert record["index_values"]["padding_count"] == int((indices == -1).sum())
    if indices.numel():
        assert record["index_values"]["min"] == indices.min().item()
        assert record["index_values"]["max"] == indices.max().item()
    else:
        assert record["index_values"]["min"] is None


@pytest.mark.parametrize("fault", ["read", "emit", "bound"])
def test_failed_capture_or_logging_preserves_primary_exception(monkeypatch, fault):
    original_error = RuntimeError("original permutation error")

    def permute(*args):
        raise original_error

    module = SimpleNamespace(permute_tokens=permute)
    records = []
    indices = torch.tensor([[-2, -1]])
    emit = records.append
    if fault == "read":

        def bad_read(_):
            raise RuntimeError("CUDA unavailable")

        monkeypatch.setattr(torch.Tensor, "cpu", bad_read)
    elif fault == "emit":

        def bad_emit(_):
            raise OSError("log unavailable")

        emit = bad_emit
    else:
        indices = torch.empty((262145, 1), dtype=torch.long)
        monkeypatch.setattr(torch.Tensor, "cpu", lambda _: pytest.fail("oversized capture reads tensor"))
    with deepep_trace.permutation_error_trace(module, emit):
        with pytest.raises(RuntimeError) as caught:
            module.permute_tokens(torch.ones(1, 3), indices, indices, 1)
    assert caught.value is original_error
    assert module.permute_tokens is permute
    if fault != "emit":
        assert len(records) == 1
        assert "capture_error" in records[0]


def test_error_sample_is_bounded_and_indices_are_unchanged():
    indices = torch.arange(-80, 80).reshape(40, 4)
    original = indices.clone()
    error = RuntimeError("failure")
    records = []

    def permute(*args):
        raise error

    module = SimpleNamespace(permute_tokens=permute)
    with deepep_trace.permutation_error_trace(module, records.append):
        with pytest.raises(RuntimeError):
            module.permute_tokens(torch.ones(40, 3), indices, torch.ones(40, 4), 80)
    assert len(records[0]["index_values"]["other_negative_sample"]) == 32
    torch.testing.assert_close(indices, original, rtol=0, atol=0)
