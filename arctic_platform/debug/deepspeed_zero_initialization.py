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

"""Request-scoped observation of installed ZeRO initialization allocations."""

from __future__ import annotations

TRAINING_OPTION = "arctic_platform_zero_initialization_trace"


def zero_optimizer_class():
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    return DeepSpeedZeroOptimizer


def initialize_with_allocation_trace(original, optimizer, record):
    import types

    torch = original.__globals__["torch"]
    allocation_index = 0
    partitions = [
        {
            "partition_index": index,
            "partition_size": int(size),
            "master_numel": tensor.numel(),
            "master_dtype": str(tensor.dtype),
            "master_device": str(tensor.device),
        }
        for index, (size, tensor) in enumerate(
            zip(optimizer.partition_size, optimizer.single_partition_of_fp32_groups, strict=True)
        )
    ]
    record(
        "INITIALIZATION_START",
        partitions=partitions,
        device=str(optimizer.device),
        cpu_offload=optimizer.cpu_offload,
        cpu_offload_pin_memory=optimizer.cpu_offload_pin_memory,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
    )

    class TorchAllocationObserver:
        def __getattr__(self, name):
            return getattr(torch, name)

        def zeros(self, *args, **kwargs):
            nonlocal allocation_index
            index = allocation_index
            allocation_index += 1
            context = {
                "allocation_index": index,
                "shape": args,
                "dtype": str(kwargs.get("dtype")),
                "device": str(kwargs.get("device")),
            }
            record("ALLOCATION_START", **context)
            try:
                tensor = torch.zeros(*args, **kwargs)
            except BaseException as error:
                record("ALLOCATION_ERROR", **context, error=repr(error))
                raise
            record(
                "ALLOCATION_COMPLETE",
                allocation_index=index,
                numel=tensor.numel(),
                bytes=tensor.numel() * tensor.element_size(),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
            )
            return tensor

    # Only this installed method's globals change; torch.zeros stays untouched.
    traced = types.FunctionType(
        original.__code__,
        {**original.__globals__, "torch": TorchAllocationObserver()},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    traced.__kwdefaults__ = original.__kwdefaults__
    try:
        result = traced(optimizer)
    except BaseException as error:
        record("INITIALIZATION_ERROR", error=repr(error))
        raise
    record("INITIALIZATION_COMPLETE", allocations=allocation_index)
    return result


def open_trace(directory, rank):
    import os
    import socket
    import time

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{socket.gethostname()}-rank-{rank}-pid-{os.getpid()}-{time.time_ns()}.jsonl"
    return path.open("x")


def maybe_initialize_with_zero_allocation_trace(initialize, ds_config, rank, **kwargs):
    if TRAINING_OPTION not in ds_config:
        return initialize(config=ds_config, **kwargs)

    import hashlib
    import inspect
    import json
    import re
    import threading
    import time
    from pathlib import Path

    option = ds_config[TRAINING_OPTION]
    if not isinstance(option, dict) or set(option) != {"request_id", "output_dir"}:
        raise ValueError(f"{TRAINING_OPTION} requires request_id and output_dir")
    request_id = option["request_id"]
    if not isinstance(request_id, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", request_id) is None:
        raise ValueError("ZeRO initialization request_id must be a nonempty safe identifier")
    if not isinstance(option["output_dir"], str):
        raise ValueError("ZeRO initialization output_dir must be a path string")
    directory = Path(option["output_dir"]).resolve()
    if not directory.is_relative_to("/data-fast"):
        raise ValueError("ZeRO initialization evidence must stay under /data-fast")
    if type(rank) is not int or rank < 0:
        raise ValueError("ZeRO initialization rank must be a nonnegative integer")

    optimizer_class = zero_optimizer_class()
    original = optimizer_class.initialize_optimizer_states
    if getattr(original, "_arctic_zero_trace", False):
        raise RuntimeError("ZeRO initialization trace already active")
    source = Path(inspect.getsourcefile(original))
    identity = {
        "request_id": request_id,
        "rank": rank,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    caller_thread = threading.get_ident()
    with open_trace(directory / request_id, rank) as stream:

        def record(event, **fields):
            row = {**identity, "event": event, "time_ns": time.time_ns(), **fields}
            line = json.dumps(row)
            stream.write(line + "\n")
            stream.flush()
            print("ZERO_INITIALIZATION_TRACE " + line, flush=True)

        def observed(optimizer):
            if threading.get_ident() != caller_thread:
                return original(optimizer)
            return initialize_with_allocation_trace(original, optimizer, record)

        observed._arctic_zero_trace = True
        optimizer_class.initialize_optimizer_states = observed
        try:
            return initialize(
                config={key: value for key, value in ds_config.items() if key != TRAINING_OPTION},
                **kwargs,
            )
        finally:
            optimizer_class.initialize_optimizer_states = original
