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

"""Explicit, pinned LoRA protocol selection at existing native call boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from arctic_platform.debug.lora_sampler_evidence import _bind_sources
from arctic_platform.debug.lora_sampler_evidence import _require
from arctic_platform.debug.lora_training_protocol import prepare_lora_training_workers
from arctic_platform.debug.lora_training_protocol import prepare_lora_worker
from arctic_platform.debug.lora_training_protocol import release_lora_worker
from arctic_platform.debug.lora_training_protocol import send_lora_worker
from arctic_platform.debug.weight_sync_evidence import enabled

REQUEST_ENV = "ARCTIC_PLATFORM_LORA_SYNC_REQUEST"
DIGEST_ENV = "ARCTIC_PLATFORM_LORA_SYNC_REQUEST_SHA256"


def selected_lora_request():
    """Read the caller-staged request, not a discovered or fabricated recipe."""
    path, expected = os.environ.get(REQUEST_ENV), os.environ.get(DIGEST_ENV)
    if path is None and expected is None:
        return None
    _require(enabled(), "pinned LoRA sync request requires explicit evidence opt-in")
    _require(
        type(path) is str
        and type(expected) is str
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected),
        "exact caller-staged request path and SHA256 required",
    )
    selected = Path(path)
    _require(
        selected.is_absolute()
        and selected.is_relative_to("/data-fast")
        and selected.is_file()
        and not selected.is_symlink()
        and selected.resolve(strict=True) == selected,
        "canonical regular FAST LoRA request required",
    )
    before = selected.stat()
    raw = selected.read_bytes()
    after = selected.stat()

    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    _require(
        identity(before) == identity(after) and hashlib.sha256(raw).hexdigest() == expected,
        "caller-staged LoRA request changed",
    )
    request = json.loads(raw)
    _require(
        type(request) is dict
        and set(request) == {"schema", "training", "sampling"}
        and request["schema"] == "explicit-native-lora-sync-request-v1"
        and type(request["training"]) is dict
        and type(request["sampling"]) is dict
        and request["training"].get("admitted_sources") == request["sampling"].get("admitted_sources"),
        "exact shared source-bound training and sampling request required",
    )
    return request


def maybe_lora_replica_pool(pool_class):
    request = selected_lora_request()
    if request is None:
        from arctic_platform.debug.rl_sampler_evidence import maybe_replica_pool

        return maybe_replica_pool(pool_class)
    from arctic_platform.debug.lora_sampler_worker import make_lora_replica_pool

    return make_lora_replica_pool(pool_class, request["sampling"])


def maybe_prepare_lora_worker(worker, request):
    selected = selected_lora_request()
    _require(
        selected is not None and selected["training"] == request, "actual staged actor preparation request required"
    )
    return prepare_lora_worker(worker, request)


def maybe_send_lora_worker(worker):
    selected = selected_lora_request()
    if selected is None:
        from arctic_platform.debug.weight_sync_evidence import maybe_send_weights

        return maybe_send_weights(worker, worker.get_weights())
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _require(
        type(slot) is dict and slot["request"] == selected["training"],
        "native preparation differs from current staged request",
    )
    sender = worker._weight_sender
    binding = slot.get("sender_binding")
    _require(
        type(binding) is dict
        and binding["sender"] is sender
        and sender.group_id == 0
        and sender.targets == binding["targets"]
        and sender.ports == binding["ports"],
        "actual native sender targets or ports differ from captured initialization",
    )
    result = send_lora_worker(worker)
    _require(
        worker._weight_sender is sender and sender.targets == binding["targets"] and sender.ports == binding["ports"],
        "actual native sender routing changed during transfer",
    )
    return result


def maybe_record_lora_sender(worker, group, schedule, base_port):
    request = selected_lora_request()
    if request is None:
        return None
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _require(
        type(slot) is dict and slot["request"] == request["training"] and worker.rank == 0,
        "actual prepared global-rank-zero actor required before sender initialization",
    )
    _require(
        group.group_id == 0
        and group.sender_train_rank == 0
        and group.targets == [(0, rank) for rank in range(8)]
        and schedule.training_gpus == 8
        and schedule.inference_tp == 8
        and schedule.inference_replicas == 1
        and type(base_port) is int
        and 0 < base_port <= 65528,
        "actual full8 sender schedule and finite native ports required",
    )
    sender = worker._weight_sender
    _require(sender is not None and "sender_binding" not in slot, "actual single initialized native sender required")
    _bind_sources(
        request["training"]["admitted_sources"],
        maybe_record_lora_sender,
        type(sender),
        type(group),
        type(schedule),
        schedule.port_for_target,
    )
    targets = list(group.targets)
    ports = [schedule.port_for_target(replica, tp, base_port) for replica, tp in targets]
    _require(
        sender.group_id == group.group_id
        and sender.targets == targets
        and sender.ports == ports
        and ports == list(range(base_port, base_port + 8)),
        "actual initialized sender differs from the retained native target and port recipe",
    )
    slot["sender_binding"] = {"sender": sender, "targets": targets, "ports": ports}
    return {"sender_train_rank": 0, "targets": targets, "ports": ports, "training_gpus": 8, "inference_tp": 8}


def maybe_release_lora_worker(worker):
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    if slot is not None:
        return release_lora_worker(worker, slot)
    return None


def maybe_lora_transfer_schedule(schedule_class, **kwargs):
    native = schedule_class.build(**kwargs)
    request = selected_lora_request()
    if request is None:
        return native
    _require(
        kwargs
        == {
            "training_sharding": "dp",
            "training_gpus": 8,
            "inference_replicas": 1,
            "inference_tp": 8,
        },
        "actual original TRAIN8 and SAMPLE8 schedule geometry required",
    )
    _require(
        len(native.groups) == 8
        and [group.sender_train_rank for group in native.groups] == list(range(8))
        and all(group.world_size == 2 for group in native.groups),
        "actual native eight-sender default schedule required before LoRA projection",
    )
    targets = [target for group in native.groups for target in group.targets]
    _require(
        targets == [(0, rank) for rank in range(8)],
        "every native TP target must occur exactly once in retained native order",
    )
    origins = request["training"]["admitted_sources"]
    group_class = type(native.groups[0])
    _bind_sources(origins, maybe_lora_transfer_schedule, schedule_class.build, schedule_class, group_class)
    # A prepared actor-local list is replayable; the native sender consumes it for each target.
    native.groups = [group_class(group_id=0, sender_train_rank=0, targets=targets)]
    _require(
        native.training_gpus == 8
        and native.inference_tp == 8
        and native.inference_replicas == 1
        and native.groups[0].targets == targets
        and native.groups[0].world_size == 2,
        "projected sender policy must retain actual native allocation geometry and targets",
    )
    _bind_sources(origins, maybe_lora_transfer_schedule, schedule_class, group_class)
    return native


async def maybe_prepare_lora_training(server, workers, pool, schedule):
    request = selected_lora_request()
    if request is None:
        return None
    _require(
        server.weight_sync_ready is False
        and not getattr(server, "_ap_lora_prepare_attempted", False)
        and not hasattr(server, "_ap_lora_training_preparation"),
        "native LoRA transfer is single-use; no unchanged collective or target retry",
    )
    _require(
        pool.num_replicas == 1
        and pool.tp_size == 8
        and len(schedule.groups) == 1
        and schedule.groups[0].sender_train_rank == 0
        and schedule.groups[0].world_size == 2
        and schedule.groups[0].replica_ids == [0],
        "actual original one-replica SAMPLE8 rank-zero pairwise schedule required",
    )
    origins = request["training"]["admitted_sources"]
    _bind_sources(origins, maybe_prepare_lora_training, type(server), type(pool), pool.sync_weights, type(schedule))
    server._ap_lora_prepare_attempted = True
    prepared = await prepare_lora_training_workers(workers, request["training"])
    _require(selected_lora_request() == request, "staged request changed during native preparation")
    _bind_sources(origins, maybe_prepare_lora_training, type(server), type(pool), pool.sync_weights, type(schedule))
    server._ap_lora_training_preparation = prepared
    return prepared


async def maybe_lora_max_param_bytes(server, worker):
    if selected_lora_request() is None:
        return await worker.max_param_bytes.remote()
    prepared = getattr(server, "_ap_lora_training_preparation", None)
    _require(type(prepared) is dict and prepared["all_training_ranks_prepared"] is True, "all8 preparation required")
    manifest = prepared["rank_preparations"][0].get("emitted_manifest")
    _require(type(manifest) is dict and manifest, "actual retained rank-zero adapter manifest required")
    widths = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
    sizes = []
    for value in manifest.values():
        _require(
            type(value) is dict
            and value.get("dtype") in widths
            and type(value.get("numel")) is int
            and value["numel"] > 0,
            "actual positive floating adapter tensor geometry required",
        )
        sizes.append(value["numel"] * widths[value["dtype"]])
    return max(sizes)


async def maybe_collect_lora_sync(server, receive_task, send_tasks):
    request = selected_lora_request()
    if request is None:
        from arctic_platform.debug.rl_sampler_evidence import maybe_sync_weights

        return await maybe_sync_weights(receive_task, send_tasks)
    results = await asyncio.gather(receive_task, *send_tasks, return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        error = RuntimeError("native LoRA sender or receiver failed")
        for cause in errors:
            error.add_note(repr(cause))
        raise error from errors[0]
    received, *sent = results
    _require(
        len(sent) == 1
        and type(sent[0]) is dict
        and sent[0].get("weight_format") == "lora"
        and type(received) is dict
        and type(received.get("workers")) is list
        and len(received["workers"]) == 1,
        "actual single native LoRA sender and sampler receipt required",
    )
    worker = received["workers"][0]
    _require(
        type(worker) is dict
        and worker.get("weight_format") == "lora"
        and type(worker.get("all_rank_adapter_installation")) is list
        and len(worker["all_rank_adapter_installation"]) == 8
        and worker.get("adapter_storage_verified") is False
        and worker.get("adapter_application_verified") is False
        and worker.get("full_parameter_coverage") is False
        and worker.get("capacity_admitted") is False,
        "allTP adapter installation is not storage/application/full-model/capacity admission",
    )
    _require(selected_lora_request() == request, "staged request changed during native transfer")
    _bind_sources(request["training"]["admitted_sources"], maybe_collect_lora_sync)
    return {
        "status": "ok",
        "weight_format": "lora",
        "targets": [{"send": sent, "recv": received}],
        "training_preparation": server._ap_lora_training_preparation,
        "adapter_storage_verified": False,
        "adapter_application_verified": False,
        "full_parameter_coverage": False,
        "capacity_admitted": False,
        "parameter_only_judge_admitted": False,
        "numerical": None,
    }
