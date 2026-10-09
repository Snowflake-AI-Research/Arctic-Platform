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

"""Parity between the reviewed DSS contracts and their native Arctic Platform forms."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import registered_tests
from arctic_platform.correctness.harness.spec import TestSpec as ConfigSpec

PROJECT_ROOT = Path(__file__).parents[2]
CORRECTNESS_ROOT = PROJECT_ROOT / "arctic_platform" / "correctness"
REVIEWED = json.loads((Path(__file__).parent / "fixtures" / "dss_reviewed_contracts.json").read_text())


def _native_config_from_dss(raw: dict) -> dict:
    """Translate a reviewed DSS job without choosing or defaulting any contract value."""
    sub_jobs = raw.get("sub_job_configs") or [raw]
    training_jobs = [job for job in sub_jobs if job.get("job_type") == "training"]
    assert len(training_jobs) == 1
    job = training_jobs[0]
    legacy = deepcopy(job["training_config"])
    ds_config = deepcopy(legacy["ds_config"])
    for key in ("train_batch_size", "gradient_clipping"):
        if key in legacy:
            ds_config[key] = deepcopy(legacy[key])
    optimizer = legacy.get("optimizer")
    if optimizer is not None:
        params = deepcopy(optimizer)
        name = params.pop("name")
        ds_config["optimizer"] = {"type": name, "params": params}
    excluded = {
        "ds_config",
        "gradient_clipping",
        "max_seq_len",
        "n_gpus",
        "optimizer",
        "peft_config",
        "train_batch_size",
    }
    native = {
        "model_name": job["model_name"],
        "seed": job["seed"],
        "dtype": job["dtype"],
        "max_seq_len": legacy["max_seq_len"],
        "training_gpus": legacy["n_gpus"],
        "sampling_gpus": 0,
        "log_prob_gpus": 0,
        "training": {
            "ds_config": ds_config,
            "ds_worker_config": {key: deepcopy(value) for key, value in legacy.items() if key not in excluded},
        },
        "sampling": {},
        "backend": {"type": "onprem", "protocol": "ray"},
    }
    if "peft_config" in legacy:
        native["training"]["peft"] = deepcopy(legacy["peft_config"])

    sampling_jobs = [job for job in sub_jobs if job.get("job_type") == "sampling"]
    assert len(sampling_jobs) <= 1
    if sampling_jobs:
        inference = deepcopy(sampling_jobs[0]["inference_config"])
        assert inference.pop("max_seq_len") == native["max_seq_len"]
        native["sampling_gpus"] = inference.pop("n_gpus")
        native["sampling"] = {
            "vllm": inference.pop("vllm_config", {}),
            "arctic_inference_config": inference or None,
        }
    return native


def test_all_eight_reviewed_specs_preserve_the_dss_contract() -> None:
    spec_paths = sorted((CORRECTNESS_ROOT / "specs").glob("*.json"))

    assert {path.stem for path in spec_paths} == set(REVIEWED["specs"])
    for path in spec_paths:
        actual = asdict(ConfigSpec.read(path))
        actual.pop("config_path")
        actual.pop("config_checksum")
        assert actual == REVIEWED["specs"][path.stem]


def test_all_eight_native_configs_preserve_the_dss_job_semantics() -> None:
    config_paths = sorted((CORRECTNESS_ROOT / "configs").rglob("*.config"))
    reviewed_paths = {CORRECTNESS_ROOT / "configs" / relative for relative in REVIEWED["configs"]}

    assert reviewed_paths <= set(config_paths)
    for relative, legacy in REVIEWED["configs"].items():
        expected = ArcticClientConfig.model_validate(_native_config_from_dss(legacy))
        actual = load_config(CORRECTNESS_ROOT / "configs" / relative).native
        assert actual == expected


def test_every_check_listed_by_a_reviewed_ap_spec_is_registered() -> None:
    listed = {test_id for contract in REVIEWED["specs"].values() for test_id in contract["applicable_tests"]}

    assert listed <= set(registered_tests())
