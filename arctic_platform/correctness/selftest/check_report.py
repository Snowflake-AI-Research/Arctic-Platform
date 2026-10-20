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

import json
from pathlib import Path

import pytest

from arctic_platform.correctness.harness import report
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.report import write
from arctic_platform.correctness.selftest.check_applicable_tests import _spec


def test_reports_are_timestamped_and_latest_links_move_without_deleting_history(tmp_path: Path) -> None:
    write([], {}, tmp_path, timestamp="2026-09-24-15-50")
    write([], {}, tmp_path, timestamp="2026-09-24-15-51")

    assert (tmp_path / "report-2026-09-24-15-50.md").is_file()
    assert (tmp_path / "report-2026-09-24-15-50.json").is_file()
    assert (tmp_path / "report-2026-09-24-15-51.md").is_file()
    assert (tmp_path / "report-2026-09-24-15-51.json").is_file()
    assert (tmp_path / "report.md").is_symlink()
    assert (tmp_path / "report.json").is_symlink()
    assert (tmp_path / "report.md").readlink() == Path("report-2026-09-24-15-51.md")
    assert (tmp_path / "report.json").readlink() == Path("report-2026-09-24-15-51.json")


@pytest.mark.parametrize(
    "selected",
    [
        ["inference-checkpoint-loss"],
        ["inference-checkpoint-loss", "e2e-sft-train-validate"],
        ["e2e-sft-train-validate"],
    ],
)
def test_cortex_fixture_report_distinguishes_full_inference_model(tmp_path, monkeypatch, selected):
    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _spec(selected)
    metadata = {
        "backend": "cortex",
        "requested_model_name": "published-reduced-model",
        "fixture_sha256": "fixture-hash",
        "fixture_stage_path": "snow://fixture-stage/exact",
    }
    inference_selected = "inference-checkpoint-loss" in selected
    if inference_selected:
        metadata["inference_model_name"] = "catalog-full-model"
    execution = {"unit": metadata}
    results = [Result(test_id=test_id, config_id="unit", outcome=Outcome.PASS) for test_id in selected]

    write(results, {"unit": spec}, tmp_path, execution=execution)

    markdown = (tmp_path / "report.md").read_text()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["execution"] == execution
    assert [result["test_id"] for result in payload["results"]] == selected
    assert "local model/tokenizer SHA-256 binding: `fixture-hash`" in markdown
    assert "publication stage: `snow://fixture-stage/exact`" in markdown
    if inference_selected:
        assert "`inference-checkpoint-loss` requested full model: `catalog-full-model`" in markdown
        assert "requested published reduced model (fixture-based checks only): `published-reduced-model`" in markdown
        assert "- requested published reduced model: `published-reduced-model`" not in markdown
        assert "reviewed reference fixture (fixture-based checks only; not used by inference)" in markdown
        assert "reviewed reference fixture (not loaded on Cortex)" not in markdown
    else:
        assert "requested published reduced model: `published-reduced-model`" in markdown
        assert "requested full model" not in markdown


def test_local_cortex_fixture_report_does_not_require_publication_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _spec(["single-step-grads"])
    execution = {
        "unit": {
            "backend": "cortex",
            "requested_model_name": "/data-fast/base-models/synthetic/unit",
            "fixture_sha256": "fixture-hash",
            "fixture_distribution": "local-gateway",
            "gateway_url": "http://127.0.0.1:8003",
        }
    }
    results = [Result(test_id="single-step-grads", config_id="unit", outcome=Outcome.PASS)]

    write(results, {"unit": spec}, tmp_path, execution=execution)

    markdown = (tmp_path / "report.md").read_text()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["execution"] == execution
    assert "requested local reduced model: `/data-fast/base-models/synthetic/unit`" in markdown
    assert "local gateway: `http://127.0.0.1:8003`" in markdown
    assert "local model/tokenizer SHA-256 binding: `fixture-hash`" in markdown
    assert "publication stage:" not in markdown
    assert "reviewed reference fixture (not loaded on Cortex)" not in markdown


@pytest.mark.parametrize(
    ("metadata", "expected_backend"),
    [
        ({"gpus": 16}, "unknown"),
        ({"gpus": 16, "backend": None}, "unknown"),
        ({"gpus": 16, "backend": ""}, "unknown"),
        ({"gpus": 16, "backend": "onprem"}, "onprem"),
    ],
    ids=["missing-backend", "null-backend", "empty-backend", "known-backend"],
)
def test_execution_error_report_preserves_original_failure_with_partial_metadata(
    tmp_path, monkeypatch, metadata, expected_backend
):
    monkeypatch.setattr(report, "collect", lambda: {"host": "client-host"})
    spec = _spec(["inference-checkpoint-loss"])
    reason = "Traceback (most recent call last):\nRuntimeError: exported tensor inventory is incomplete"
    result = Result(
        test_id="inference-checkpoint-loss",
        config_id="unit",
        outcome=Outcome.FAIL,
        summary="Checkpoint export failed before comparison",
        reason=reason,
    )
    execution = {"unit": metadata}

    write([result], {"unit": spec}, tmp_path, execution=execution)

    markdown = (tmp_path / "report.md").read_text()
    payload = json.loads((tmp_path / "report.json").read_text())
    assert f"execution backend: `{expected_backend}`" in markdown
    assert reason in markdown
    assert payload["execution"] == execution
    assert payload["results"][0]["reason"] == reason
    assert payload["results"][0]["outcome"] == "fail"
    assert payload["totals"]["fail"] == 1
    assert payload["totals"]["pass"] == 0
