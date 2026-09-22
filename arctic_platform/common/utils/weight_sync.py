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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared training→sampling weight-sync helpers (HTTP + Ray servers).

Colocated CUDA-IPC / CPU-file paths live here so both servers call one
implementation. Staged ``wake_inference`` (weights → load → kv_cache) is the
caller's job: Ray does it inside its wrappers; HTTP leaves it to the client
(see legacy ``rl/http_client.sync_weights`` and unified ``AsyncArcticRLClient``).
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import time
from typing import Any
from typing import Awaitable
from typing import Callable

import ray

from arctic_platform.common.utils.cuda_ipc import merge_cuda_ipc_payloads

logger = logging.getLogger(__name__)

# Optional async callback: await wake(["weights"]) / await wake(["kv_cache"]).
# Extra kwargs (e.g. restore_weights=True) are forwarded for Ray's staged wake.
WakeFn = Callable[..., Awaitable[Any]]


async def sync_weights_cuda_ipc(
    workers,
    pool,
    lp_pool=None,
    *,
    wake_inference: WakeFn | None = None,
) -> dict:
    """Colocated weight sync via CUDA IPC (zero-copy, same GPU)."""
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    gather_refs = [w.gather_cuda_ipc_handles.remote() for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in gather_refs])
    ipc_payload = merge_cuda_ipc_payloads(results)
    num_params = ipc_payload.get("num_params", 0)

    if wake_inference is not None:
        await wake_inference(["weights"])

    recv_tasks = []
    total_replicas = 0
    for p in [pool, lp_pool]:
        if p is None or getattr(p, "_config", None) is None:
            continue
        for rid in range(p.num_replicas):
            w = p._workers[rid]
            recv_tasks.append(loop.run_in_executor(None, ray.get, w.load_weights_cuda_ipc.remote(ipc_payload)))
            total_replicas += 1
    await asyncio.gather(*recv_tasks)

    await asyncio.gather(*[loop.run_in_executor(None, ray.get, w.release_ipc_handles.remote()) for w in workers])

    if wake_inference is not None:
        await wake_inference(["kv_cache"])

    elapsed = time.monotonic() - t0
    logger.info(
        "Weight sync (CUDA IPC) complete in %.3fs (%d replica(s), %d params)",
        elapsed,
        total_replicas,
        num_params,
    )
    return {"status": "ok"}


async def sync_weights_cuda_ipc_low_mem(
    workers,
    pool,
    lp_pool=None,
    *,
    wake_inference: WakeFn | None = None,
) -> dict:
    """Memory-efficient colocated CUDA IPC: one gathered param at a time."""
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    param_names = ray.get(workers[0].get_parameter_names.remote())
    num_params = len(param_names)

    replicas = []
    for p in [pool, lp_pool]:
        if p is None or getattr(p, "_config", None) is None:
            continue
        for rid in range(p.num_replicas):
            replicas.append(p._workers[rid])

    if wake_inference is not None:
        # restore_weights=True: sleep_inference may have replaced param.data with
        # [1] stubs; backload full-shape storage before the IPC stream copies in.
        await wake_inference(["weights"], restore_weights=True)

    all_names: list = []
    for idx, name in enumerate(param_names):
        gather_refs = [w.get_cuda_ipc_handle.remote(name) for w in workers]
        results = ray.get(gather_refs)
        payload = merge_cuda_ipc_payloads(results)
        all_names.extend(payload["names"])
        validate = all_names if idx == num_params - 1 else None
        recv_tasks = [
            loop.run_in_executor(None, ray.get, w.load_weights_cuda_ipc_chunk.remote(payload, validate))
            for w in replicas
        ]
        await asyncio.gather(*recv_tasks)
        await asyncio.gather(*[loop.run_in_executor(None, ray.get, w.release_ipc_handles.remote()) for w in workers])

    if wake_inference is not None:
        await wake_inference(["kv_cache"])

    elapsed = time.monotonic() - t0
    logger.info(
        "Weight sync (CUDA IPC, low-mem) complete in %.3fs (%d replica(s), %d params)",
        elapsed,
        len(replicas),
        num_params,
    )
    return {"status": "ok"}


async def sync_weights_cpu_file(
    sync_path: str,
    workers,
    pool,
    lp_pool=None,
    *,
    wake_inference: WakeFn | None = None,
) -> dict:
    """Colocated weight sync via a shared CPU file (works when training is offloaded)."""
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    save_refs = [w.gather_and_save_state_dict.remote(sync_path) for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in save_refs])
    num_params = results[0].get("num_params", 0)

    if wake_inference is not None:
        await wake_inference(["weights"])

    recv_tasks = []
    total_replicas = 0
    for p in [pool, lp_pool]:
        if p is None or getattr(p, "_config", None) is None:
            continue
        for rid in range(p.num_replicas):
            w = p._workers[rid]
            recv_tasks.append(loop.run_in_executor(None, ray.get, w.load_weights_from_shm_path.remote(sync_path)))
            total_replicas += 1
    await asyncio.gather(*recv_tasks)

    pathlib.Path(sync_path).unlink(missing_ok=True)

    if wake_inference is not None:
        await wake_inference(["kv_cache"])

    elapsed = time.monotonic() - t0
    logger.info(
        "Weight sync (CPU→GPU) complete in %.3fs (%d replica(s), %d params)",
        elapsed,
        total_replicas,
        num_params,
    )
    return {"status": "ok"}
