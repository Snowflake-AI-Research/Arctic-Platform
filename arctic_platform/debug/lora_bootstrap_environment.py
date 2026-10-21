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

"""Forward sealed LoRA inputs through the captured same-allocation bootstrap."""

from __future__ import annotations

import hashlib
import inspect
import os
import re
from contextlib import contextmanager
from copy import deepcopy

from arctic_platform.debug.lora_selected_sampler import PATH_ENV
from arctic_platform.debug.lora_selected_sampler import SHA_ENV
from arctic_platform.debug.lora_selected_sampler import _bytes
from arctic_platform.debug.lora_selected_sampler import bootstrap_descriptor
from arctic_platform.debug.lora_sync_hooks import DIGEST_ENV
from arctic_platform.debug.lora_sync_hooks import REQUEST_ENV
from arctic_platform.debug.lora_sync_hooks import selected_lora_request
from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import verify_sources

PEER = "10.4.171.50"
PYTHON = "/home/yak/miniconda3/envs/dss/bin/python"
RAY = "/home/yak/miniconda3/envs/dss/bin/ray"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _environment(staged):
    _require(
        staged["schema"] == "job3-selected-lora-staged-inputs-v1"
        and staged["training_gpus"] == staged["sampling_gpus"] == 8
        and staged["runtime_admitted"] is staged["capacity_admitted"] is False
        and staged["native_engine_constructed"] is False
        and staged["limits_are_capacity_proof"] is False
        and staged["numerical"] is None,
        "original source-only TRAIN8+SAMPLE8 staged inputs required",
    )
    settings = {
        VERIFY_ENV: "1",
        REQUEST_ENV: staged["request"]["path"],
        DIGEST_ENV: staged["request"]["sha256"],
        PATH_ENV: staged["bootstrap"]["path"],
        SHA_ENV: staged["bootstrap"]["sha256"],
    }
    _require(staged["environment"] == settings, "staged environment differs from pointers")
    body = bootstrap_descriptor(staged["bootstrap"])
    sources = staged["admitted_sources"]
    _require(body["admitted_sources"] == sources, "bootstrap source closure differs")
    for name in ("request", "settings"):
        pointer = staged[name]
        raw = _bytes(pointer["path"])
        _require(
            hashlib.sha256(raw).hexdigest() == pointer["sha256"],
            "staged request or settings bytes changed",
        )
    _require(
        sources.get(staged["settings"]["path"]) == staged["settings"]["sha256"],
        "settings must belong to the shared source closure",
    )
    verify_sources(sources)
    return settings, dict(sources)


@contextmanager
def owned_selected_lora_bootstrap(staged, *, cluster, base_environment, staging, guard):
    """Wrap the supplied frozen context; do not start a cluster or construct a model."""
    staged = deepcopy(staged)
    settings, sources = _environment(staged)
    original = cluster._pdsh
    observed = source_files(
        owned_selected_lora_bootstrap.__wrapped__,
        _environment,
        inspect.unwrap(base_environment),
        original,
        selected_lora_request,
    )
    _require(observed.items() <= sources.items(), "captured bootstrap sources must be admitted")
    previous = {key: os.environ.get(key) for key in settings}
    _require(
        all(value is None or value == settings[key] for key, value in previous.items()),
        "cannot replace another owner's request environment",
    )
    report = {
        "schema": "job3-selected-lora-bootstrap-forwarding-v1",
        "environment": dict(settings),
        "peer_launches": [],
        "peer_runtime_admitted": False,
        "capacity_admitted": False,
        "native_engine_constructed": False,
        "numerical": None,
        "callback_restored": False,
        "environment_restored": False,
    }
    base_settings = None
    primary = None

    def launch_peer(hosts, argv, **kwargs):
        _require(list(hosts) == [PEER], "only the current same-allocation peer is allowed")
        if len(argv) == 4 and argv[:1] == [PYTHON] and argv[2] == "--cleanup-root":
            # The frozen context already validates its exact captured cleanup path.
            return original(hosts, argv, **kwargs)
        root = cluster._spawned_temp_dir
        _require(
            type(base_settings) is dict
            and type(root) is str
            and re.fullmatch(r"/tmp/j3rl-[0-9a-f]{24}", root)
            and argv[:1] == ["env"],
            "captured native peer bootstrap command required",
        )
        prefix = ["env", *(f"{key}={value}" for key, value in sorted(base_settings.items()))]
        command = argv[len(prefix) :]
        _require(
            argv[: len(prefix)] == prefix
            and len(command) == 5
            and command[:2] == [RAY, "start"]
            and re.fullmatch(r"--address=10\.4\.166\.121:[0-9]{1,5}", command[2])
            and command[3:] == [f"--temp-dir={root}", "--disable-usage-stats"]
            and 1 <= int(command[2].rsplit(":", 1)[1]) <= 65535
            and kwargs == {"check": False, "timeout": 600}
            and all(key not in base_settings or base_settings[key] == value for key, value in settings.items()),
            "exact retained native peer recipe and unchanged base environment required",
        )
        guard()
        _require(_environment(staged) == (settings, sources), "staged inputs changed before peer launch")
        merged = {**base_settings, **settings}
        forwarded = ["env", *(f"{key}={value}" for key, value in sorted(merged.items())), *command]
        result = original(hosts, forwarded, **kwargs)
        report["peer_launches"].append(
            {
                "hosts": list(hosts),
                "argv": forwarded,
                "exit": result.returncode,
                "runtime_admitted": False,
            }
        )
        guard()
        _require(_environment(staged) == (settings, sources), "staged inputs changed during peer launch")
        return result

    context = None
    entered = False
    try:
        guard()
        os.environ.update(settings)
        request = selected_lora_request()
        _require(
            request["training"]["admitted_sources"] == request["sampling"]["admitted_sources"] == sources,
            "actual selected request differs from staged sources",
        )
        cluster._pdsh = launch_peer
        context = base_environment(staging, cluster=cluster, guard=guard)
        base_receipt = context.__enter__()
        entered = True
        base_settings = dict(base_receipt["settings"])
        report["base"] = base_receipt
        try:
            yield report
        except BaseException as error:
            primary = error
            try:
                context.__exit__(type(error), error, error.__traceback__)
            except BaseException as cleanup_error:
                error.add_note("captured bootstrap teardown also failed: " + repr(cleanup_error))
            entered = False
            raise
        else:
            entered = False
            context.__exit__(None, None, None)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if entered and context is not None:
                try:
                    context.__exit__(None, None, None)
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    primary.add_note("captured bootstrap entry cleanup also failed: " + repr(cleanup_error))
        finally:
            cluster._pdsh = original
            report["callback_restored"] = cluster._pdsh is original
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            report["environment_restored"] = all(os.environ.get(key) == value for key, value in previous.items())
            try:
                verify_sources(sources)
                guard()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note("final bootstrap source guard also failed: " + repr(cleanup_error))
