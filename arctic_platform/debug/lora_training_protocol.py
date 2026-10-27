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

"""Actor-local LoRA preparation and the all-training-rank dispatch barrier."""

from __future__ import annotations

import asyncio
from copy import deepcopy

from arctic_platform.debug.lora_sampler_evidence import _bind_sources
from arctic_platform.debug.lora_sampler_evidence import _require
from arctic_platform.debug.lora_weight_sync_evidence import PreparedLoRAWeights
from arctic_platform.debug.lora_weight_sync_evidence import prepare_collective_lora_weights
from arctic_platform.debug.lora_weight_sync_evidence import send_prepared_lora_weights
from arctic_platform.debug.weight_sync_evidence import enabled


def _request(request):
    _require(
        type(request) is dict and set(request) == {"max_weight_bytes", "admitted_sources"},
        "exact collective LoRA preparation request required",
    )
    _require(
        type(request["max_weight_bytes"]) is int and request["max_weight_bytes"] > 0,
        "finite positive native adapter byte limit required",
    )
    _require(
        type(request["admitted_sources"]) is dict and request["admitted_sources"],
        "actual operation source closure required before collective preparation",
    )
    return deepcopy(request)


def prepare_lora_worker(worker, request):
    """Run on every live native training actor; tensors never leave that actor."""
    _require(enabled(), "collective LoRA preparation must be explicitly enabled")
    request = _request(request)
    _require(
        type(worker.rank) is int and 0 <= worker.rank < 8 and worker.world_size == 8,
        "actual original eight-rank training actor required",
    )
    engine = worker.engine
    _require(engine is not None and engine.module is not None, "actual initialized native engine.module required")
    _require(
        not getattr(worker, "_ap_lora_prepare_attempted", False) and not hasattr(worker, "_ap_owned_lora_preparation"),
        "native adapter preparation is single-use; no unchanged collective retry",
    )
    model = engine.module
    origins = _bind_sources(
        request["admitted_sources"],
        prepare_lora_worker,
        prepare_collective_lora_weights,
        send_prepared_lora_weights,
        type(model),
        model.named_parameters,
        model.named_buffers,
    )
    worker._ap_lora_prepare_attempted = True
    prepared = prepare_collective_lora_weights(
        model,
        weight_format="lora",
        is_zero3=False,
        max_weight_bytes=request["max_weight_bytes"],
        admitted_sources=origins,
        expected_world_size=8,
        expected_ep_size=8,
    )
    _require(
        isinstance(prepared, PreparedLoRAWeights) and prepared.rank == worker.rank,
        "actual native preparation returned a different training rank",
    )
    _require(worker.engine is engine and engine.module is model, "native model changed during preparation")
    receipt = deepcopy(prepared.receipt())
    _bind_sources(origins, prepare_lora_worker, prepare_collective_lora_weights, send_prepared_lora_weights)
    worker._ap_owned_lora_preparation = {
        "engine": engine,
        "model": model,
        "prepared": prepared,
        "request": request,
        "send_attempted": False,
    }
    return receipt


def send_lora_worker(worker):
    """Send the captured stream with the current owned native sender, once."""
    _require(enabled(), "prepared LoRA send must be explicitly enabled")
    slot = getattr(worker, "_ap_owned_lora_preparation", None)
    _require(type(slot) is dict, "actual actor-local collective preparation required before send")
    _require(
        worker.rank == 0 and slot["prepared"].rank == 0,
        "only actual global rank zero may send the native adapter stream",
    )
    _require(
        worker.engine is slot["engine"] and worker.engine.module is slot["model"],
        "prepared stream belongs to a different native engine or model",
    )
    _require(not slot["send_attempted"], "prepared native send is single-use")
    sender = worker._weight_sender
    _require(sender is not None, "actual initialized native weight sender required")
    origins = _bind_sources(slot["prepared"].sources, send_lora_worker, send_prepared_lora_weights, sender.send)
    slot["send_attempted"] = True
    result = send_prepared_lora_weights(sender, slot["prepared"])
    _require(
        worker._weight_sender is sender and worker._ap_owned_lora_preparation is slot,
        "owned sender or preparation changed during native send",
    )
    _require(
        worker.engine is slot["engine"] and worker.engine.module is slot["model"],
        "native model changed during send",
    )
    _bind_sources(origins, send_lora_worker, send_prepared_lora_weights, sender.send)
    return result


def release_lora_worker(worker, slot):
    """Drop only the captured private tensors; native actor teardown stays separate."""
    _require(
        slot is not None and getattr(worker, "_ap_owned_lora_preparation", None) is slot,
        "refusing to release another owner's preparation",
    )
    del worker._ap_owned_lora_preparation
    return {
        "owned_preparation_released": True,
        "native_sender_destroyed": False,
        "native_process_group_destroyed": False,
    }


async def prepare_lora_training_workers(workers, request):
    """Await all eight public actor preparations before returning send readiness."""
    _require(enabled(), "collective LoRA preparation must be explicitly enabled")
    request = _request(request)
    _require(
        type(workers) is list and len(workers) == 8 and len({id(worker) for worker in workers}) == 8,
        "exact eight distinct current training actor handles required",
    )
    methods = [worker.prepare_lora_weights.remote for worker in workers]
    _bind_sources(request["admitted_sources"], prepare_lora_training_workers, prepare_lora_worker)
    pending = []
    errors = []
    for method in methods:
        try:
            pending.append(method(deepcopy(request)))
        except BaseException as error:
            errors.append(error)
    completed = await asyncio.gather(*pending, return_exceptions=True)
    errors.extend(value for value in completed if isinstance(value, BaseException))
    if errors:
        preparation_error = RuntimeError("one or more native training-rank preparations failed")
        for cause in errors:
            preparation_error.add_note(repr(cause))
        raise preparation_error from errors[0]
    _require(len(completed) == 8, "all eight actual training-rank preparations must complete")
    for rank, receipt in enumerate(completed):
        _require(
            type(receipt) is dict
            and type(receipt.get("rank")) is int
            and receipt["rank"] == rank
            and receipt.get("weight_format") == "lora"
            and receipt.get("native_iterator_completed") is True
            and receipt.get("unchanged_native_iterator") is True
            and receipt.get("native_shard_bytes_checked") is (rank == 0),
            "actual completed native preparation from each ordered training rank required",
        )
        _require(
            type(receipt.get("rank_records")) is list
            and len(receipt["rank_records"]) == 8
            and [row.get("rank") for row in receipt["rank_records"]] == list(range(8)),
            "every preparation must retain all eight actual rank records",
        )
        _require(
            receipt.get("source_files") == request["admitted_sources"],
            "training preparation sources differ from the current operation",
        )
        for key in (
            "global_inventory",
            "rank_records",
            "adapter_parameter_names",
            "non_wire_parameter_names",
            "non_wire_buffers",
        ):
            _require(
                key in receipt and receipt[key] == completed[0].get(key),
                "collective native inventory or adapter selection differs across training ranks",
            )
        for key in (
            "receiver_installation_admitted",
            "serving_request_admitted",
            "router_application_admitted",
            "runtime_admitted",
            "capacity_admitted",
            "parameter_only_judge_admitted",
            "recipe_admitted",
        ):
            _require(receipt.get(key) is False, "preparation cannot promote native or judge admission")
    _bind_sources(request["admitted_sources"], prepare_lora_training_workers, prepare_lora_worker)
    return {
        "schema": "all-training-ranks-lora-prepared-v1",
        "rank_preparations": completed,
        "all_training_ranks_prepared": True,
        "private_streams_remain_actor_local": True,
        "native_send_started": False,
        "capacity_admitted": False,
        "parameter_only_judge_admitted": False,
        "numerical": None,
    }
