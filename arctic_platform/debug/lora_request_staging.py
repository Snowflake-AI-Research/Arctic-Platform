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

"""Stage one explicit LoRA request without inferring runtime or memory admission."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from copy import deepcopy
from pathlib import Path

from arctic_platform.debug.lora_receiver_preparation import BOUND_KEYS
from arctic_platform.debug.lora_receiver_preparation import PREPARATION_FILENAME
from arctic_platform.debug.lora_selected_sampler import PATH_ENV
from arctic_platform.debug.lora_selected_sampler import SHA_ENV
from arctic_platform.debug.lora_selected_sampler import bootstrap_descriptor
from arctic_platform.debug.lora_sync_hooks import DIGEST_ENV
from arctic_platform.debug.lora_sync_hooks import REQUEST_ENV
from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import verify_sources

CONFIG_ID = "glm5.3-flash-h200-train-rl-lora-16gpus-sp8-256k"
MODEL = "zai-org/GLM-5.3-Flash"
PREPARE_MODULE = "arctic_platform.debug.lora_receiver_preparation"
PREPARE_NAME = "prepare_selected_lora_receiver"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _private_directory(path):
    path = Path(path)
    _require(
        path.is_absolute()
        and path.is_relative_to("/data-fast")
        and path.resolve(strict=True) == path
        and path.is_dir()
        and not path.is_symlink()
        and stat.S_IMODE(path.stat().st_mode) == 0o700,
        "canonical existing private FAST staging directory required",
    )
    return path


def _new_directory(path):
    path.mkdir(mode=0o700, exist_ok=False)
    path.chmod(0o700)
    return _private_directory(path)


def _write_json(path, body):
    raw = (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _require(
        path.read_bytes() == raw and stat.S_IMODE(path.stat().st_mode) == 0o600,
        "private staged bytes changed",
    )
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _config(cfg, sources):
    _require(
        cfg.config_id == CONFIG_ID
        and cfg.is_rl is True
        and cfg.model_name == cfg.native.model_name == MODEL
        and cfg.n_gpus == cfg.native.training_gpus == cfg.native.sampling_gpus == 8
        and cfg.native.log_prob_gpus == 0
        and cfg.training["sp_size"] == cfg.training["ep_size"] == 8
        and cfg.training["peft_config"]["r"] == 32,
        "actual canonical LoRA TRAIN8+SAMPLE8 LoadedConfig required",
    )
    config_path = str(Path(cfg.path).resolve(strict=True))
    _require(config_path in sources, "actual LoadedConfig bytes must be source-bound")
    verify_sources(sources)
    return deepcopy(cfg.training["peft_config"])


def stage_selected_lora_request(
    cfg,
    *,
    directory,
    receiver,
    admitted_sources,
    adapter_id,
    adapter_name,
    max_weight_bytes,
    max_received_bytes,
    publication_bounds,
    distribution_names,
):
    """Return pinned environment after caller-owned inputs are staged exactly once.

    Byte limits are attempt limits, not a constructor or transient-capacity proof.
    The useful caller still verifies these exact files on every consuming node.
    """
    _require(enabled(), "explicit LoRA staging opt-in required")
    sources = deepcopy(admitted_sources)
    _require(type(sources) is dict and sources, "actual selected source closure required")
    observed = source_files(stage_selected_lora_request, bootstrap_descriptor)
    _require(
        observed.items() <= sources.items(),
        "staging/bootstrap sources must be admitted",
    )
    lora_config = _config(cfg, sources)
    receiver = deepcopy(receiver)
    _require(
        type(receiver) is dict
        and set(receiver) == {"module", "name", "source"}
        and type(receiver["module"]) is str
        and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", receiver["module"])
        and type(receiver["name"]) is str
        and re.fullmatch(r"[A-Za-z_]\w*", receiver["name"])
        and type(receiver["source"]) is str
        and receiver["source"] in sources,
        "explicit selected native receiver descriptor required",
    )
    _require(
        type(adapter_id) is int
        and adapter_id > 0
        and type(adapter_name) is str
        and adapter_name
        and all(type(value) is int and value > 0 for value in (max_weight_bytes, max_received_bytes)),
        "positive owned adapter identity and explicit byte limits required",
    )
    _require(
        max_received_bytes >= max_weight_bytes,
        "receive byte limit cannot be smaller than prepared byte limit",
    )
    bounds = deepcopy(publication_bounds)
    _require(
        type(bounds) is dict
        and set(bounds) == set(BOUND_KEYS)
        and all(type(value) is int and value > 0 for value in bounds.values()),
        "exact explicit positive private-publication bounds required",
    )
    _require(
        bounds["max_total_source_bytes"] >= bounds["max_source_bytes"],
        "total publication limit cannot be smaller than one source limit",
    )
    names = deepcopy(distribution_names)
    _require(
        type(names) is list
        and names
        and all(type(name) is str and name for name in names)
        and len(names) == len(set(names)),
        "exact distinct caller-selected distributions required",
    )
    directory = _private_directory(directory)
    _require(not any(directory.iterdir()), "staging directory must be fresh and empty")
    prepare_source = str(Path(__file__).with_name("lora_receiver_preparation.py").resolve(strict=True))
    _require(prepare_source in sources, "actual preparation callback source must be admitted")
    verify_sources(sources)
    publication = _new_directory(directory / "callback-sources")
    settings = _write_json(
        directory / PREPARATION_FILENAME,
        {
            "schema": "job3-selected-lora-preparation-v1",
            "publication_root": str(publication),
            "distribution_names": names,
            **bounds,
        },
    )
    _require(settings["path"] not in sources, "cannot replace an admitted settings identity")
    sources[settings["path"]] = settings["sha256"]
    bootstrap_body = {
        "schema": "job3-selected-lora-bootstrap-v1",
        "receiver": receiver,
        "prepare": {
            "module": PREPARE_MODULE,
            "name": PREPARE_NAME,
            "source": prepare_source,
        },
        "admitted_sources": sources,
    }
    bootstrap = _write_json(directory / "bootstrap.json", bootstrap_body)
    _require(bootstrap_descriptor(bootstrap) == bootstrap_body, "staged bootstrap rejected")
    request_body = {
        "schema": "explicit-native-lora-sync-request-v1",
        "training": {"max_weight_bytes": max_weight_bytes, "admitted_sources": sources},
        "sampling": {
            "lora_int_id": adapter_id,
            "lora_name": adapter_name,
            "lora_config": lora_config,
            "max_received_bytes": max_received_bytes,
            "admitted_sources": sources,
        },
    }
    request = _write_json(directory / "request.json", request_body)
    verify_sources(sources)
    return {
        "schema": "job3-selected-lora-staged-inputs-v1",
        "settings": settings,
        "bootstrap": bootstrap,
        "request": request,
        "admitted_sources": sources,
        "environment": {
            VERIFY_ENV: "1",
            REQUEST_ENV: request["path"],
            DIGEST_ENV: request["sha256"],
            PATH_ENV: bootstrap["path"],
            SHA_ENV: bootstrap["sha256"],
        },
        "training_gpus": 8,
        "sampling_gpus": 8,
        "limits_are_capacity_proof": False,
        "runtime_admitted": False,
        "capacity_admitted": False,
        "native_engine_constructed": False,
        "numerical": None,
    }
