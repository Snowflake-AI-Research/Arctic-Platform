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
"""Training→sampling weight-sync helpers for the Ray server.

Colocated CUDA-IPC / CPU-file paths live here. Staged ``wake_inference``
(weights → load → kv_cache) is the caller's job: the Ray server does it inside
its wrappers.
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
from arctic_platform.common.weight_checksum import compare_weight_checksums

logger = logging.getLogger(__name__)

# Optional async callback: await wake(["weights"]) / await wake(["kv_cache"]).
WakeFn = Callable[..., Awaitable[Any]]


def _receivers(pool, lp_pool) -> list[tuple[str, Any]]:
    """``(label, worker)`` for every replica of the sampling and log-prob pools that are up."""
    receivers = []
    for kind, p in (("sampling", pool), ("log_prob", lp_pool)):
        if p is None or getattr(p, "_config", None) is None:
            continue
        for rid in range(p.num_replicas):
            receivers.append((f"{kind} replica {rid}", p._workers[rid]))
    return receivers


def received_checksums(labels: list[str], results: list[dict]) -> dict[str, float | None]:
    """Each worker's ``recv_l2_sq``, keyed ``"<label> rank <i>"``.

    The index is the worker's place in ``collective_rpc``, which is the tensor-parallel rank only when there is no
    pipeline parallelism.
    """
    received = {}
    for label, result in zip(labels, results):
        ranks = result.get("rank_recv_l2_sq") or [result.get("recv_l2_sq")]
        for rank, value in enumerate(ranks):
            received[f"{label} rank {rank}"] = value
    return received


def _accumulate(totals: dict[str, float | None], values: dict[str, float | None]) -> None:
    """Add one chunk's checksums into running totals. A chunk without a value leaves that total unavailable."""
    for label, value in values.items():
        if label in totals and totals[label] is None:
            continue
        totals[label] = None if value is None else totals.get(label, 0.0) + value


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

    receivers = _receivers(pool, lp_pool)
    recv_results = await asyncio.gather(
        *[loop.run_in_executor(None, ray.get, w.load_weights_cuda_ipc.remote(ipc_payload)) for _, w in receivers]
    )
    total_replicas = len(receivers)

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
    checksum = compare_weight_checksums(
        {f"training rank {rank}": r.get("sent_l2_sq") for rank, r in enumerate(results)},
        received_checksums([label for label, _ in receivers], recv_results),
    )
    return {"status": "ok", **checksum}


async def sync_weights_cuda_ipc_low_mem(
    workers,
    pool,
    lp_pool=None,
    *,
    wake_inference: WakeFn | None = None,
) -> dict:
    """Memory-efficient colocated CUDA IPC: one gathered param at a time.

    Each param's ``sent_l2_sq`` / ``recv_l2_sq`` covers that param only, so both are summed here, per training rank
    and per sampler TP rank, into the sync's totals.
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    param_names = ray.get(workers[0].get_parameter_names.remote())
    num_params = len(param_names)

    receivers = _receivers(pool, lp_pool)
    replicas = [w for _, w in receivers]
    labels = [label for label, _ in receivers]

    if wake_inference is not None:
        await wake_inference(["weights"])

    all_names: list = []
    sent: dict[str, float | None] = {}
    received: dict[str, float | None] = {}
    for idx, name in enumerate(param_names):
        gather_refs = [w.get_cuda_ipc_handle.remote(name) for w in workers]
        results = ray.get(gather_refs)
        _accumulate(sent, {f"training rank {rank}": r.get("sent_l2_sq") for rank, r in enumerate(results)})
        payload = merge_cuda_ipc_payloads(results)
        all_names.extend(payload["names"])
        validate = all_names if idx == num_params - 1 else None
        recv_tasks = [
            loop.run_in_executor(None, ray.get, w.load_weights_cuda_ipc_chunk.remote(payload, validate))
            for w in replicas
        ]
        _accumulate(received, received_checksums(labels, await asyncio.gather(*recv_tasks)))
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
    return {"status": "ok", **compare_weight_checksums(sent, received)}


async def sync_weights_cpu_file(
    sync_path: str,
    workers,
    pool,
    lp_pool=None,
    *,
    wake_inference: WakeFn | None = None,
) -> dict:
    """Colocated weight sync via a shared CPU file (works when training is offloaded).

    Only rank 0 writes the file, so only its ``sent_l2_sq`` is the reference.
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    save_refs = [w.gather_and_save_state_dict.remote(sync_path) for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in save_refs])
    num_params = results[0].get("num_params", 0)

    if wake_inference is not None:
        await wake_inference(["weights"])

    receivers = _receivers(pool, lp_pool)
    recv_results = await asyncio.gather(
        *[loop.run_in_executor(None, ray.get, w.load_weights_from_shm_path.remote(sync_path)) for _, w in receivers]
    )
    total_replicas = len(receivers)

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
    checksum = compare_weight_checksums(
        {"training rank 0": results[0].get("sent_l2_sq")},
        received_checksums([label for label, _ in receivers], recv_results),
    )
    return {"status": "ok", **checksum}
