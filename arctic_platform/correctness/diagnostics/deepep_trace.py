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

"""Observe DeepEP calls, with optional bounded tensor reads only after permutation errors."""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


def tensor_metadata(tensor):
    if isinstance(tensor, tuple):
        return [tensor_metadata(value) for value in tensor]
    if not isinstance(tensor, torch.Tensor):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
        "requires_grad": tensor.requires_grad,
    }


@contextlib.contextmanager
def trace_exceptions(emit, operation):
    try:
        yield
    except BaseException as error:
        emit(
            {
                "operation": operation,
                "phase": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


@contextlib.contextmanager
def permutation_error_trace(module, emit):
    original = module.permute_tokens

    @functools.wraps(original)
    def observed(hidden_states, dispatched_indices, dispatched_scores, num_local_experts):
        try:
            return original(hidden_states, dispatched_indices, dispatched_scores, num_local_experts)
        except RuntimeError as error:
            record = {
                "operation": "permute_tokens",
                "phase": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "indices": tensor_metadata(dispatched_indices),
                "scores": tensor_metadata(dispatched_scores),
                "hidden_states": tensor_metadata(hidden_states),
                "num_local_experts": num_local_experts,
            }
            try:
                if dispatched_indices.numel() > 262144:
                    raise ValueError("Receive indices exceed diagnostic bound")
                indices = dispatched_indices.detach().cpu()
                other_negative = indices[indices < -1]
                record["index_values"] = {
                    "count": indices.numel(),
                    "min": indices.min().item() if indices.numel() else None,
                    "max": indices.max().item() if indices.numel() else None,
                    "padding_count": int((indices == -1).sum()),
                    "other_negative_count": other_negative.numel(),
                    "other_negative_sample": other_negative[:32].tolist(),
                    "out_of_range_count": int((indices >= num_local_experts).sum()),
                }
            except Exception as capture_error:
                record["capture_error"] = f"{type(capture_error).__name__}: {capture_error}"
            try:
                emit(record)
            except Exception:
                # A diagnostic write must not replace the original model exception.
                pass
            raise

    module.permute_tokens = observed
    try:
        yield
    finally:
        module.permute_tokens = original


@contextlib.contextmanager
def buffer_call_trace(buffer_class, emit, *, operations=("dispatch", "combine")):
    originals = {name: getattr(buffer_class, name) for name in operations}
    call_number = 0

    def instrument(operation, original):
        @functools.wraps(original)
        def traced(buffer, *args, **kwargs):
            nonlocal call_number
            call_number += 1
            call = call_number
            record = {
                "operation": operation,
                "call": call,
                "caller": sys._getframe(1).f_code.co_name,
                "cached": kwargs.get("handle", args[1] if len(args) > 1 else None) is not None,
                "buffer": id(buffer),
                "nvl_bytes": getattr(buffer, "num_nvl_bytes", None),
                "rdma_bytes": getattr(buffer, "num_rdma_bytes", None),
                "grad_enabled": torch.is_grad_enabled(),
            }
            emit(
                dict(
                    record,
                    phase="enter",
                    x=tensor_metadata(kwargs.get("x", args[0] if args else None)),
                )
            )
            try:
                result = original(buffer, *args, **kwargs)
            except BaseException as error:
                emit(
                    dict(
                        record,
                        phase="error",
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
                )
                raise
            emit(dict(record, phase="return", x=tensor_metadata(result[0])))
            return result

        return traced

    for name, original in originals.items():
        setattr(buffer_class, name, instrument(name, original))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(buffer_class, name, original)


@contextlib.contextmanager
def traced_workers(directory: Path, *, permutation_errors: bool = False):
    import ray

    from arctic_platform.common import deepspeed_worker
    from arctic_platform.common import ray_server
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    base = deepspeed_worker.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class DeepEPTraceWorker(base):
        def initialize(self, master_addr, job_config):
            from deep_ep import Buffer

            directory.mkdir(parents=True, exist_ok=True)
            self._trace_path = directory / f"worker-{os.getpid()}.jsonl"
            self._trace_context = contextlib.ExitStack()
            self._trace_context.enter_context(
                buffer_call_trace(Buffer, self._emit_trace, operations=("get_dispatch_layout", "dispatch", "combine"))
            )
            if permutation_errors:
                from arctic_platform.model.implementations.moe.distributed import deepep

                self._trace_context.enter_context(permutation_error_trace(deepep, self._emit_trace))
            try:
                with trace_exceptions(self._emit_trace, "initialize"):
                    result = super().initialize(master_addr, job_config)
            except BaseException:
                self._trace_context.close()
                raise
            self._emit_trace({"operation": "initialize", "phase": "return"})
            return result

        def _emit_trace(self, record):
            engine = getattr(self, "engine", None)
            record = dict(
                record,
                rank=getattr(self, "rank", None),
                pid=os.getpid(),
                time_ns=time.time_ns(),
                step=getattr(engine, "global_steps", None),
                micro_step=getattr(engine, "micro_steps", None),
            )
            with self._trace_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

        def forward_backward(self, batch):
            self._emit_trace({"operation": "forward_backward", "phase": "enter"})
            with trace_exceptions(self._emit_trace, "forward_backward"):
                result = super().forward_backward(batch)
            self._emit_trace({"operation": "forward_backward", "phase": "return"})
            return result

    original_server = ray_server.ArcticRLRayServerState

    class DeepEPTraceServer(original_server):
        async def initialize(self, job_config_dict):
            with _patched_deepspeed_worker_bindings(DeepEPTraceWorker):
                return await super().initialize(job_config_dict)

    with _patched_deepspeed_worker_bindings(DeepEPTraceWorker):
        ray_server.ArcticRLRayServerState = DeepEPTraceServer
        try:
            yield
        finally:
            ray_server.ArcticRLRayServerState = original_server


def main():
    from arctic_platform.correctness import __main__ as cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with traced_workers(args.out / "trace"):
        status = cli.main(
            [
                "run",
                "--config",
                args.config,
                "--test",
                "checkpoint-resume-loss",
                "--case",
                "gas4",
                "--out",
                str(args.out / "report"),
            ]
        )
    for path in sorted((args.out / "trace").glob("worker-*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        print("DEEPEP_TRACE_TAIL", path, json.dumps(records[-16:]), flush=True)
    for report in sorted((args.out / "report").glob("report-*.json")):
        print("REPORT_JSON_BEGIN", report, report.read_text(), "REPORT_JSON_END", flush=True)
    print("DEEPEP_TRACE_COMPLETE", status, args.out, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
