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

"""Public external-pool injection and all-rank results for opt-in sync checks."""

from __future__ import annotations

import asyncio
import functools
import os
from uuid import uuid4

from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import verify_sources

_EXTENSION = "arctic_platform.debug.weight_sync_evidence.WeightSyncEvidenceExtension"


def maybe_replica_pool(pool_class):
    """Keep the actual external pool/scheduler; inject only its public worker protocol."""
    if not enabled():
        return pool_class
    return functools.partial(pool_class, worker_cls=_worker_actor())


def _worker_actor():
    import ray

    environment = {VERIFY_ENV: "1"}
    limit_key = "ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES"
    if limit_key in os.environ:
        environment[limit_key] = os.environ[limit_key]
    return ray.remote(runtime_env={"env_vars": environment})(SamplerEvidenceWorker)


async def maybe_sync_weights(receive_task, send_tasks):
    results = await asyncio.gather(receive_task, *send_tasks)
    if not enabled():
        return {"status": "ok"}
    received, *sent = results
    if not isinstance(received, dict) or not isinstance(received.get("workers"), list) or not received["workers"]:
        raise RuntimeError("external pool did not return actual worker sync results")
    for worker in received["workers"]:
        if not isinstance(worker, dict) or not worker.get("all_rank_weight_sync_verification"):
            raise RuntimeError("a sampling worker returned no all-rank byte verification")
    return {
        "status": "ok",
        "weight_format": "nccl",
        "targets": [{"send": sent, "recv": received}],
    }


class SamplerEvidenceWorker:
    """External ReplicaPool worker protocol, without unwrapping its Ray actor.

    The external TransferSchedule, sender, NCCL engine, receiver superclass and
    scheduler remain selected. A per-actor lock excludes generate/sleep during
    sync. This worker does not implement native router replay or speculative sync.
    """

    def __init__(self):
        self.llm = None
        self.state = "uninitialized"
        self._lock = asyncio.Lock()
        self._origins = None
        self._tp = None

    async def initialize(self, engine_kwargs, extra_env=None):
        if self.llm is not None:
            raise RuntimeError("sampler evidence worker is already initialized")
        if extra_env:
            os.environ.update(extra_env)
        if not enabled():
            raise RuntimeError("sampler evidence opt-in must propagate into the actual worker")
        import vllm.plugins
        from arctic_inference.envs import arctic_inference_effective_enabled
        from arctic_inference.server.metrics import RingStatLogger

        vllm.plugins.load_general_plugins()
        from vllm.v1.engine.async_llm import AsyncLLM

        if arctic_inference_effective_enabled():
            from arctic_inference.vllm.args import ArcticAsyncEngineArgs as EngineArgs
        else:
            from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs

        kwargs = dict(engine_kwargs)
        existing = kwargs.get("worker_extension_cls")
        if existing not in (None, _EXTENSION, "arctic_inference.server.weight_sync.WeightSyncExtension"):
            raise ValueError("sampler evidence cannot replace another selected worker extension")
        if kwargs.get("speculative_config"):
            raise ValueError("speculative weight-sync evidence is not implemented")
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
            kwargs.setdefault("attention_config", {"backend": "FLASH_ATTN"})
        kwargs["worker_extension_cls"] = _EXTENSION
        # Unlike a compatibility fallback, never silently remove engine recipe fields.
        args = EngineArgs(**kwargs)
        self._tp = args.tensor_parallel_size
        if type(self._tp) is not int or self._tp < 1:
            raise ValueError("native engine tensor-parallel size must be positive")
        if self._tp == 1:
            raise ValueError("native TP1 selects an unsupported direct weight loader")
        self._origins = source_files(type(self), EngineArgs, AsyncLLM.from_vllm_config)
        verify_sources(self._origins)
        self.llm = AsyncLLM.from_vllm_config(args.create_engine_config(), stat_loggers=[RingStatLogger])
        try:
            verify_sources(self._origins)
        except BaseException:
            self.shutdown()
            raise
        self.state = "ready"

    def _ready(self):
        if self.llm is None or self.state != "ready":
            raise RuntimeError(f"sampler not ready: {self.state}")
        verify_sources(self._origins)

    async def generate(self, prompt, sampling_params):
        # Native routing/cache/application must be supplied, not simulated by a marker.
        if any(key.startswith("dss_") for key in sampling_params):
            raise ValueError("external native router replay producer/transport/application is not implemented")
        from vllm import SamplingParams

        async with self._lock:
            self._ready()
            params = SamplingParams(**dict(sampling_params))
            request_id = str(uuid4())
            prompt_input = {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
            final = None
            try:
                async for output in self.llm.generate(prompt_input, params, request_id=request_id):
                    final = output
            except BaseException:
                await self.llm.abort(request_id)
                raise
            verify_sources(self._origins)
            if final is None or not final.outputs:
                raise RuntimeError("empty generation output")
            choice = final.outputs[0]
            result = {
                "text": choice.text,
                "token_ids": list(choice.token_ids),
                "finish_reason": choice.finish_reason,
                "prompt_len": len(final.prompt_token_ids or []),
                "generation_len": len(choice.token_ids),
                "prefix_cache_len": int(getattr(final, "num_cached_tokens", None) or 0),
            }
            for key, positions in (("prompt_logprobs", final.prompt_logprobs), ("logprobs", choice.logprobs)):
                if positions is not None:
                    result[key] = [
                        (
                            None
                            if position is None
                            else {
                                token: (
                                    {"logprob": value.logprob, "rank": value.rank}
                                    if hasattr(value, "logprob")
                                    else {"logprob": float(value), "rank": None}
                                )
                                for token, value in position.items()
                            }
                        )
                        for position in positions
                    ]
            return result

    async def sync_weights(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        bucket_size=256 * 1024 * 1024,
        engine_only=False,
        direct_mode=False,
        reverse=False,
    ):
        async with self._lock:
            self._ready()
            results = await self.llm.collective_rpc(
                "sync_weights",
                args=(
                    master_addr,
                    master_port,
                    rank_offset,
                    world_size,
                    bucket_size,
                    engine_only,
                    direct_mode,
                    reverse,
                ),
            )
            verify_sources(self._origins)
            if not isinstance(results, list) or len(results) != self._tp:
                raise RuntimeError("collective RPC did not return every selected TP rank")
            if engine_only:
                if any(not isinstance(row, dict) or row.get("status") != "engine_ready" for row in results):
                    raise RuntimeError("native receiver engine initialization failed")
                return {"status": "engine_ready", "rank_results": results}
            if any(
                not isinstance(row, dict)
                or row.get("status") != "done"
                or not isinstance(row.get("weight_sync_verification"), dict)
                for row in results
            ):
                raise RuntimeError("a native TP rank did not return receiver byte/shadow evidence")
            return {
                "status": "done",
                "rank_results": results,
                "all_rank_weight_sync_verification": [row["weight_sync_verification"] for row in results],
            }

    async def sleep(self, level=1, offload_weights=False):
        async with self._lock:
            self._ready()
            if offload_weights:
                await self.llm.collective_rpc("offload_model_weights")
            await self.llm.collective_rpc("sleep", kwargs={"level": level})
            verify_sources(self._origins)
            self.state = "sleeping"
            return {"status": "sleeping", "level": level}

    async def wake_up(self, tags=None, restore_weights=False):
        async with self._lock:
            if self.state == "ready":
                return {"status": "already_ready"}
            if self.state not in {"sleeping", "weights_ready"}:
                raise RuntimeError(f"cannot wake sampler in {self.state}")
            verify_sources(self._origins)
            await self.llm.collective_rpc("wake_up", kwargs={"tags": tags})
            if restore_weights:
                await self.llm.collective_rpc("backload_model_weights")
            verify_sources(self._origins)
            self.state = "weights_ready" if set(tags or []) == {"weights"} else "ready"
            return {"status": self.state}

    async def reset_prefix_cache(self):
        async with self._lock:
            if self.llm is None:
                raise RuntimeError("sampler is not initialized")
            verify_sources(self._origins)
            await self.llm.reset_prefix_cache()
            verify_sources(self._origins)
            return {"status": "prefix_cache_reset"}

    async def close_weight_sync(self):
        async with self._lock:
            if self.llm is None:
                raise RuntimeError("sampler is not initialized")
            # Resource cleanup is not conditional on a fresh source-admission check.
            results = await self.llm.collective_rpc("close_weight_sync")
            return {"status": "ok", "rank_results": results}

    async def compute_weight_norm(self):
        async with self._lock:
            self._ready()
            results = await self.llm.collective_rpc("compute_weight_norm")
            verify_sources(self._origins)
            squared = sum(row["sq_sum"] for row in results)
            return {
                "norm": squared**0.5,
                "sq_sum": squared,
                "num_params": sum(row["num_params"] for row in results),
            }

    def is_healthy(self):
        return self.llm is not None and self.state in {"ready", "sleeping", "weights_ready"}

    def get_state(self):
        return self.state

    def get_stats(self):
        from arctic_inference.server.metrics import get_collector

        return {"state": self.state, "pid": os.getpid(), **get_collector().latest()}

    def drain_metrics(self):
        from arctic_inference.server.metrics import get_collector

        return {"pid": os.getpid(), "snapshots": get_collector().drain_snapshots()}

    def set_replica_id(self, replica_id):
        from arctic_inference.server.metrics import get_collector

        get_collector().set_replica_id(int(replica_id))

    def pid(self):
        return os.getpid()

    def shutdown(self):
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None
        self.state = "uninitialized"
