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

"""Local RL server matching the dss-platform sftp_server HTTP API.

Uses Ray to manage DeepSpeed workers and ArcticInference ReplicaPools.

Usage::

    python -m arctic_platform.common.http_server \\
        --training-gpus 4 --sampling-gpus 2 --log-prob-gpus 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

import ray
import torch
import uvicorn
from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from transformers import AutoTokenizer

from arctic_platform import wire
from arctic_platform._dependency_groups import require_any_dep_group
from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.common.ray_cluster import init_ray_cluster
from arctic_platform.common.server import ArcticRLServerState
from arctic_platform.common.utils import http_split_batch
from arctic_platform.common.utils import merge_dict_shards
from arctic_platform.common.utils.batch import finalize_fwd_bwd_metrics
from arctic_platform.common.utils.batch import restore_batch_order
from arctic_platform.common.utils.checkpoint import resolve_checkpoint_save_paths
from arctic_platform.common.utils.debug import pr0
from arctic_platform.common.utils.ray_pg import ColocatePlacement
from arctic_platform.common.utils.ray_pg import create_colocate_placement
from arctic_platform.common.utils.ray_pg import pg_scheduling_options
from arctic_platform.common.utils.server_models import GenerateRequest
from arctic_platform.common.utils.server_models import JobConfig
from arctic_platform.common.utils.server_models import LoadCheckpointRequest
from arctic_platform.common.utils.server_models import LogProbsRequest
from arctic_platform.common.utils.server_models import OperationRequest
from arctic_platform.common.utils.server_models import ResetPrefixCacheRequest
from arctic_platform.common.utils.server_models import SaveRequest
from arctic_platform.common.utils.server_models import StepRequest
from arctic_platform.common.utils.server_models import WeightNormRequest
from arctic_platform.common.utils.server_models import WeightSyncRequest
from arctic_platform.common.utils.server_models import build_model_config

if TYPE_CHECKING:
    from arctic_inference.server.replica_pool import ReplicaPool


def _replica_pool_cls():
    """Lazy import — training-only servers do not need arctic_inference / vLLM."""
    require_any_dep_group("rl")
    from arctic_inference.server.replica_pool import ReplicaPool

    return ReplicaPool


def _transfer_schedule_cls():
    require_any_dep_group("rl")
    from arctic_inference.server.weight_sync.schedule import TransferSchedule

    return TransferSchedule


logger = logging.getLogger(__name__)

app = FastAPI(title="Arctic RL Local Server")

# ---------------------------------------------------------------------------
# Request / response models (mirrors dss-platform sftp_server)
# ---------------------------------------------------------------------------

ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


class ArcticRLHTTPServerState(ArcticRLServerState):
    def __init__(self, **kwargs):
        pass


# Honor ARL_WEIGHT_SYNC_PORT when set so back-to-back / concurrent training jobs on one host (e.g. repeated
# pytest-flakefinder iterations or parallel xdist workers) get a fresh NCCL rendezvous port instead of all reusing
# 29600, where a SIGKILL-reaped sender from a prior job can still squat the port and deadlock the next sync.
_WEIGHT_SYNC_BASE_PORT = int(os.environ.get("ARL_WEIGHT_SYNC_PORT", 29600))
_WEIGHT_SYNC_BUCKET_SIZE = 256 * 1024 * 1024


def _verify_job(job_id: int, expected_types: Union[str, list[str]]) -> None:
    info = app.state.jobs.get(job_id)
    if isinstance(expected_types, str):
        expected_types = [expected_types]
    if info is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if info["job_type"] not in expected_types:
        raise HTTPException(400, f"Job {job_id} is not a {', '.join(expected_types)} job")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.post("/initialize")
async def initialize(job_config: JobConfig = Body(...)):
    job_type = job_config.job_type
    job_id = app.state.next_job_id
    app.state.next_job_id += 1

    colocate = app.state.colocate
    placement: ColocatePlacement = getattr(app.state, "placement", ColocatePlacement())

    # Fractional GPU fractions within each PG bundle.  Each bundle owns 1
    # physical GPU; fractions let multiple actors share that bundle while
    # Ray still sets CUDA_VISIBLE_DEVICES so each actor can see the GPU.
    # These are *Ray scheduling accounting* only (not memory caps): real
    # VRAM is time-shared via sleep/wake/offload.  All actors that share a
    # bundle must sum to <= 1.0, so with full 3-way colocation:
    #   training (0.34) + sampling (0.33) + log_prob (0.33) = 1.0
    _COLOCATE_GPU_FRACTIONS = {"sampling": 0.33, "log_prob": 0.33, "training": 0.34}

    def _pg_options(bundle_index: int, fraction_key: str) -> dict:
        """PG-pinned scheduling: fractional GPU claim inside a specific (global) bundle."""
        return pg_scheduling_options(
            placement,
            bundle_index,
            _COLOCATE_GPU_FRACTIONS[fraction_key],
        )

    # n_bundles = getattr(app.state, "n_bundles", 0)

    # Bundle layout (deterministic), full 3-way colocation:
    #   training pins rank r        → bundle r            [0 .. training_gpus-1]
    #   sampling replica r (TP=tp)  → bundles [r*tp .. r*tp+tp-1]
    #   log_prob pins rank r        → bundle r            [0 .. log_prob_gpus-1]
    # All three overlap on the same bundles (offset 0), so each physical GPU
    # hosts a training rank, a sampling worker, and a log_prob rank.
    # n_bundles = max(training_gpus, sampling_gpus, log_prob_gpus).

    if job_type == "training":
        gpus = app.state.training_gpus
        if gpus == 0:
            raise HTTPException(400, "No training GPUs configured")
        if app.state.training_workers:
            raise HTTPException(409, "Training job already running")

        workers = []
        config_dict = job_config.model_dump()
        # Honor MASTER_PORT when set so concurrent training jobs on one host (e.g.
        # parallel pytest-xdist workers) don't collide on the rendezvous port.
        master_port = int(os.environ.get("MASTER_PORT", 29500))
        for rank in range(gpus):
            if colocate and placement:
                opts = _pg_options(bundle_index=rank, fraction_key="training")
            else:
                opts = dict(num_gpus=1)
            w = DeepSpeedWorker.options(**opts).remote(rank, gpus, master_port)
            workers.append(w)

        # Use rank 0's host as the distributed rendezvous master. Passing None
        # falls back to "localhost" in the worker, which only works when every
        # rank is on the same node; on multi-node clusters the off-node ranks
        # would rendezvous against their own localhost and init_distributed()
        # hangs forever.
        master_addr = await workers[0].get_ip.remote()
        await asyncio.gather(*[w.initialize.remote(master_addr, config_dict) for w in workers])
        app.state.training_workers = workers

    elif job_type == "sampling":
        gpus = app.state.sampling_gpus
        if gpus == 0:
            raise HTTPException(400, "No sampling GPUs configured")
        pool: ReplicaPool = app.state.sampling_pool
        if pool._config is not None:
            raise HTTPException(409, "Sampling job already running")
        vllm_cfg = dict(job_config.vllm_config or {})
        if colocate:
            vllm_cfg["enable_sleep_mode"] = True
        model_cfg = build_model_config(
            job_config.model_name, vllm_cfg, arctic_inference_config=job_config.arctic_inference_config
        )
        tp = model_cfg.tensor_parallel_size
        num_replicas = gpus // tp
        if colocate and placement:
            per_replica_pgs, bundle_indices = placement.tp_layout(num_replicas, tp)
            extra_env = {}
            if tp > 1:
                extra_env["VLLM_RAY_PER_WORKER_GPUS"] = str(_COLOCATE_GPU_FRACTIONS["sampling"])
                vllm_cfg["distributed_executor_backend"] = "ray"
                model_cfg = build_model_config(
                    job_config.model_name, vllm_cfg, arctic_inference_config=job_config.arctic_inference_config
                )
            if job_config.arctic_inference_config:
                extra_env["ARCTIC_INFERENCE_ENABLED"] = "1"
                # vllm-project/vllm#31199 was fixed in 0.18.0 (vllm-project/vllm#35420);
                # override the global VLLM_DISABLE_COMPILE_CACHE=1 set in the verl runtime_env.
                extra_env["VLLM_DISABLE_COMPILE_CACHE"] = "0"
                # capture ARCTIC_INFERENCE_ENABLED from the client to the Ray TP workers
                extra_env["VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY"] = "ARCTIC_INFERENCE_"
            # set env variables before initialize
            model_cfg.extra_env = dict(extra_env)
            await pool.initialize(
                model_cfg,
                num_replicas=num_replicas,
                ray_num_gpus=_COLOCATE_GPU_FRACTIONS["sampling"],
                placement_groups=per_replica_pgs,
                bundle_indices=bundle_indices,
                extra_env=extra_env if extra_env else None,
            )
        else:
            await pool.initialize(model_cfg, num_replicas=num_replicas)

    elif job_type == "log_prob":
        gpus = app.state.log_prob_gpus
        if gpus == 0:
            raise HTTPException(400, "No log-prob GPUs configured")

        # Full 3-way colocation: log_prob ranks share the same bundles as
        # training (and sampling), so offset 0. The reference engine is
        # offloaded right after init and only woken for the ref-logprob pass,
        # so it does not contend for VRAM with training/sampling.
        # n_bundles = max(training_gpus, sampling_gpus, log_prob_gpus), so a
        # non-zero offset would push log_prob bundles out of range.
        lp_bundle_offset = 0

        if job_config.ds_config is not None:
            if app.state.log_prob_workers:
                raise HTTPException(409, "Log-prob job already running")
            # Honor MASTER_PORT when set (mirrors ray_server / training branch) so concurrent
            # log-prob jobs on one host don't collide on a hardcoded rendezvous port.
            master_port = int(os.environ.get("MASTER_PORT", 29501))
            workers = []
            config_dict = job_config.model_dump()
            for rank in range(gpus):
                if colocate and placement:
                    opts = _pg_options(bundle_index=lp_bundle_offset + rank, fraction_key="log_prob")
                else:
                    opts = dict(num_gpus=1)
                w = DeepSpeedWorker.options(**opts).remote(rank, gpus, master_port)
                workers.append(w)
            master_addr = await workers[0].get_ip.remote()
            await asyncio.gather(*[w.initialize.remote(master_addr, config_dict) for w in workers])
            app.state.log_prob_workers = workers
            app.state.log_prob_tokenizer = AutoTokenizer.from_pretrained(job_config.model_name)
            engine = "deepspeed"
        else:
            pool: ReplicaPool = app.state.log_prob_pool
            if pool._config is not None:
                raise HTTPException(409, "Log-prob job already running")
            lp_vllm_cfg = dict(job_config.vllm_config or {})
            if colocate:
                lp_vllm_cfg["enable_sleep_mode"] = True
            model_cfg = build_model_config(
                job_config.model_name, lp_vllm_cfg, arctic_inference_config=job_config.arctic_inference_config
            )
            lp_tp = model_cfg.tensor_parallel_size
            num_replicas = gpus // lp_tp
            if colocate and placement:
                per_replica_pgs, bundle_indices = placement.tp_layout(
                    num_replicas,
                    lp_tp,
                    bundle_offset=lp_bundle_offset,
                )
                lp_extra_env = {}
                if lp_tp > 1:
                    lp_extra_env["VLLM_RAY_PER_WORKER_GPUS"] = str(_COLOCATE_GPU_FRACTIONS["log_prob"])
                    # NOTE: ReplicaPool overrides VLLM_RAY_BUNDLE_INDICES
                    # per-worker using bundle_indices[i]*tp+t, so it doesn't
                    # need to be set here.
                    lp_extra_env.pop("CUDA_VISIBLE_DEVICES", None)
                    lp_vllm_cfg["distributed_executor_backend"] = "ray"
                    model_cfg = build_model_config(
                        job_config.model_name, lp_vllm_cfg, arctic_inference_config=job_config.arctic_inference_config
                    )
                if job_config.arctic_inference_config:
                    lp_extra_env["ARCTIC_INFERENCE_ENABLED"] = "1"
                    # vllm-project/vllm#31199 was fixed in 0.18.0 (vllm-project/vllm#35420);
                    # override the global VLLM_DISABLE_COMPILE_CACHE=1 set in the verl runtime_env.
                    lp_extra_env["VLLM_DISABLE_COMPILE_CACHE"] = "0"
                    # capture ARCTIC_INFERENCE_ENABLED from the client to the Ray TP workers
                    lp_extra_env["VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY"] = "ARCTIC_INFERENCE_"
                # set env variables before initialize
                model_cfg.extra_env = dict(lp_extra_env)
                await pool.initialize(
                    model_cfg,
                    num_replicas=num_replicas,
                    ray_num_gpus=_COLOCATE_GPU_FRACTIONS["log_prob"],
                    placement_groups=per_replica_pgs,
                    bundle_indices=bundle_indices,
                    extra_env=lp_extra_env if lp_extra_env else None,
                )
            else:
                await pool.initialize(model_cfg, num_replicas=num_replicas)
            engine = "vllm"

    else:
        raise HTTPException(400, f"Unknown job type: {job_type}")

    job_info: dict[str, Any] = {
        "job_id": job_id,
        "job_type": job_type,
        "model_name": job_config.model_name,
        "status": "RUNNING",
        "checkpoint_path": None,
        "sync_path": None,
    }
    if job_type == "log_prob":
        job_info["engine"] = engine
    if job_type == "training":
        assert job_config.checkpoint_path is not None, "checkpoint_path is required for training jobs"
        ckpt_dir = pathlib.Path(job_config.checkpoint_path) / f"arctic_rl_job_{job_id}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        job_info["checkpoint_path"] = str(ckpt_dir)
        job_info["sync_path"] = str(ckpt_dir / "weight_sync.pt")
        # Weight-sync strategy is static per run: record it on the job so /weight-sync need not resend it each call
        # (per-call override still wins).
        job_info["cuda_ipc"] = job_config.cuda_ipc
        job_info["low_memory"] = job_config.low_memory
    app.state.jobs[job_id] = job_info
    return {"job_id": job_id, "job_type": job_type, "running": True}


@app.post("/destroy")
async def destroy(job_id: int, job_type: str = Body(..., embed=True)):
    info = app.state.jobs.pop(job_id, None)
    if info is None:
        raise HTTPException(404, f"Job {job_id} not found")

    if info["job_type"] == "training":
        await asyncio.gather(*[w.destroy.remote() for w in app.state.training_workers])
        app.state.training_workers.clear()
    elif info["job_type"] == "sampling":
        await app.state.sampling_pool.shutdown()
    elif info["job_type"] == "log_prob":
        if info.get("engine") == "deepspeed":
            await asyncio.gather(*[w.destroy.remote() for w in app.state.log_prob_workers])
            app.state.log_prob_workers.clear()
            app.state.log_prob_tokenizer = None
        else:
            await app.state.log_prob_pool.shutdown()

    return {"job_id": job_id}


@app.post("/forward-backward")
async def forward_backward(
    job_id: int,
    body: bytes = Body(..., media_type="application/octet-stream"),
):
    tname_e2e = timers.start("xyz fwd_bwd e2e")

    tname = timers.start("xyz fwd_bwd: _verify_job")
    _verify_job(job_id, "training")
    workers = app.state.training_workers
    timers.stop_and_print_elapsed(tname)

    # tname = timers.start("xyz fwd_bwd: decompress")
    # import zlib
    # body = zlib.decompress(body)
    # timers.stop_and_print_elapsed(tname)

    tname = timers.start("xyz fwd_bwd: split_batch")
    shards, _ = http_split_batch(body, len(workers))
    # The verl driver's ``update_actor`` only consumes ``metrics`` from the
    # fwd_bwd response (see arctic_rl_client.update_actor) -- the per-token
    # ``batch`` (logprobs/entropy) is never read. Keep the worker output as
    # tensors so ``run_pipeline`` skips the per-microbatch detensorize()
    # ``.tolist()``, and omit ``batch`` from the response so it is never
    # serialized over the wire.
    shards[0]["meta"]["worker_return_tensors"] = True
    timers.stop_and_print_elapsed(tname)

    tname = timers.start("xyz fwd_bwd: gather + forward_backward")
    results = await asyncio.gather(*[w.forward_backward.remote(s) for w, s in zip(workers, shards)])
    timers.stop_and_print_elapsed(tname)
    pr0(f"[DeepSpeedWorker] fwd_bwd: {len(results)=}")

    tname = timers.start("xyz fwd_bwd: epilogue")
    metrics, avg_loss = finalize_fwd_bwd_metrics(results)
    # ``batch`` is intentionally omitted -- the driver does not consume it.
    merged = dict(
        job_id=job_id,
        metrics=metrics,
        avg_loss=avg_loss,
    )
    timers.stop_and_print_elapsed(tname)

    timers.stop_and_print_elapsed(tname_e2e)

    return Response(content=wire.dumps(merged), media_type="application/octet-stream")


@app.post("/forward")
async def forward(
    job_id: int,
    body: bytes = Body(..., media_type="application/octet-stream"),
):
    info = app.state.jobs[job_id]
    _verify_job(job_id, ["training", "log_prob"])
    job_type = info["job_type"]
    if job_type == "log_prob":
        workers = app.state.log_prob_workers
    else:
        workers = app.state.training_workers
    if not workers:
        raise HTTPException(400, f"Job {job_id} ({job_type}) has no DeepSpeed workers")

    shards, reorder_indices = http_split_batch(body, len(workers))
    shards[0]["meta"]["worker_return_tensors"] = True
    results = await asyncio.gather(*[w.forward_no_grad.remote(s) for w, s in zip(workers, shards)])
    pr0(f"[DeepSpeedWorker] fwd_no_grad: {len(results)=}")

    batch = merge_dict_shards([r["batch"] for r in results])
    if reorder_indices is not None:
        batch = restore_batch_order(batch, reorder_indices)

    metrics, avg_loss = finalize_fwd_bwd_metrics(results)
    merged = dict(
        job_id=job_id,
        batch=batch,
        metrics=metrics,
        avg_loss=avg_loss,
    )

    return Response(content=wire.dumps(merged), media_type="application/octet-stream")


@app.post("/step")
async def step(job_id: int, request: StepRequest = Body(default=StepRequest())):
    # `learning_rate` is accepted for Cortex parity; on-prem uses the DeepSpeed
    # scheduler's LR, so it is not applied here.
    _verify_job(job_id, "training")
    results = await asyncio.gather(*[w.step.remote() for w in app.state.training_workers])
    merged = dict(
        job_id=job_id,
        metrics=merge_dict_shards([r["metrics"] for r in results]),
        batch=merge_dict_shards([r["batch"] for r in results]),
    )
    return merged


@app.post("/empty-training-cache")
async def empty_training_cache(job_id: int):
    """Release ZeRO partition cache and PyTorch cached memory on all workers."""
    _verify_job(job_id, "training")
    workers = app.state.training_workers
    loop = asyncio.get_running_loop()
    refs = [w.empty_cache.remote() for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
    logger.info("Empty training cache: %s", results)
    return {"job_id": job_id, "workers": results}


@app.post("/save")
async def save(job_id: int, request: SaveRequest = Body(default=SaveRequest())):
    # `checkpoint_id`/`checkpoint_type` are accepted for Cortex parity; on-prem
    # saves to the job's configured checkpoint_path (or SFT path/step override).
    _verify_job(job_id, "training")
    info = app.state.jobs[job_id]
    # Client path overrides the job dir; optional step → .../checkpoint-{step}/.
    root = request.path or info.get("checkpoint_path", None)
    assert root is not None, f"checkpoint_path is required for training job {job_id}"
    step = request.step
    path, prune_root = resolve_checkpoint_save_paths(root, step)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    export_hf = bool(request.export_hf)
    results = await asyncio.gather(
        *[w.save_checkpoint.remote(path, export_hf) for w in app.state.training_workers],
    )
    if step is not None:
        (pathlib.Path(prune_root) / "latest").write_text(str(int(step)))
    limit = request.save_total_limit
    if limit is not None and int(limit) > 0 and app.state.training_workers:
        await app.state.training_workers[0].prune_checkpoint_dirs.remote(prune_root, int(limit))
    hf_path = None
    global_step = None
    if results and isinstance(results[0], dict):
        hf_path = results[0].get("hf_path")
        global_step = results[0].get("global_step")
    return {"job_id": job_id, "path": path, "hf_path": hf_path, "global_step": global_step}


@app.post("/load-checkpoint")
async def load_checkpoint(job_id: int, request: LoadCheckpointRequest = Body(default=LoadCheckpointRequest())):
    _verify_job(job_id, "training")
    info = app.state.jobs[job_id]
    path = request.path or info.get("checkpoint_path", None)
    assert path is not None, f"checkpoint_path is required for training job {job_id}"
    step = request.step
    if step is not None:
        path = str(pathlib.Path(path) / f"checkpoint-{int(step)}")
    elif request.path is None:
        # No explicit path/step: prefer the ``latest`` pointer under the job dir.
        latest = pathlib.Path(path) / "latest"
        if latest.is_file():
            try:
                path = str(pathlib.Path(path) / f"checkpoint-{int(latest.read_text().strip())}")
            except ValueError:
                pass
    steps = await asyncio.gather(
        *[w.load_checkpoint.remote(path) for w in app.state.training_workers],
    )
    return {"job_id": job_id, "path": path, "global_step": int(steps[0]) if steps else 0}


@app.post("/sleep-inference")
async def sleep_inference(job_id: int, level: int = 1, body: dict | None = Body(None)):
    """Put all inference engines to sleep, freeing GPU memory."""
    if isinstance(body, dict) and body.get("level") is not None:
        level = int(body["level"])
    _verify_job(job_id, "sampling")
    colocate = app.state.colocate
    results = {}
    pool: ReplicaPool = app.state.sampling_pool
    # Let vLLM's CuMemAllocator free the weights (offload_weights=False) instead
    # of the legacy manual offload, which reallocated param.data on each wake and
    # changed weight addresses -> stale rollout CUDA graphs (compile on) ->
    # grad-norm explosion. cumem keeps addresses stable.
    offload_weights = False
    results["sampling"] = await pool.sleep(level=level, offload_weights=offload_weights)
    lp_pool: ReplicaPool | None = app.state.log_prob_pool
    if lp_pool is not None and lp_pool._config is not None and not lp_pool.sleeping:
        results["log_prob"] = await lp_pool.sleep(level=level, offload_weights=offload_weights)
    if colocate:
        await pool.close_weight_sync()
        if lp_pool is not None and lp_pool._config is not None:
            await lp_pool.close_weight_sync()
    return {"job_id": job_id, **results}


@app.post("/wake-inference")
async def wake_inference(job_id: int, body: Any = Body(None)):
    """Wake all inference engines, restoring GPU memory.

    Body may be a tag list (legacy HTTP client) or ``{"tags": [...]}``.
    """
    tags = body.get("tags") if isinstance(body, dict) else body
    _verify_job(job_id, "sampling")
    colocate = app.state.colocate
    restore = colocate and (tags is None or "weights" in tags)
    results = {}
    pool: ReplicaPool = app.state.sampling_pool
    results["sampling"] = await pool.wake_up(tags=tags, restore_weights=restore)
    lp_pool: ReplicaPool | None = app.state.log_prob_pool
    if lp_pool is not None and lp_pool._config is not None and lp_pool.sleeping:
        results["log_prob"] = await lp_pool.wake_up(tags=tags, restore_weights=restore)
    return {"job_id": job_id, **results}


@app.post("/reset-prefix-cache")
async def reset_prefix_cache(job_id: int, request: ResetPrefixCacheRequest = Body(default=ResetPrefixCacheRequest())):
    """Reset the prefix cache on the sampling inference engines.

    `drain`/`timeout_s`/`retry_interval_s` are accepted for Cortex parity.
    """
    _verify_job(job_id, "sampling")
    results = {}
    pool: ReplicaPool = app.state.sampling_pool
    results["sampling"] = await pool.reset_prefix_cache()
    lp_pool: ReplicaPool | None = app.state.log_prob_pool
    if lp_pool is not None and lp_pool._config is not None:
        results["log_prob"] = await lp_pool.reset_prefix_cache()
    return {"job_id": job_id, **results}


@app.post("/operation")
async def operation(job_id: int, request: OperationRequest = Body(...)):
    """Generic data-plane operation, matching Cortex's /{job_id}/operation envelope.

    Dispatches on operation_type so the unified client routes control ops through
    one payload shape. Reuses the typed handlers below; no logic is duplicated.
    """
    payload = request.payload or {}
    if request.operation_type == "weight-sync":
        return await weight_sync(job_id, WeightSyncRequest(**payload))
    if request.operation_type == "reset-prefix-cache":
        return await reset_prefix_cache(job_id, ResetPrefixCacheRequest(**payload))
    if request.operation_type == "sleep-inference":
        return await sleep_inference(job_id, payload.get("level", 1))
    if request.operation_type == "wake-inference":
        return await wake_inference(job_id, payload.get("tags"))
    raise HTTPException(status_code=400, detail=f"unknown operation_type {request.operation_type!r}")


@app.post("/sleep-training")
async def sleep_training(job_id: int, mode: str = "all", body: dict | None = Body(None)):
    """Offload training state to CPU (sleep training workers).

    mode='all':       Offload everything (for training → inference transition)
    mode='non_lp':    Keep bf16 params on GPU, offload rest (before CUDA IPC sync)
    mode='lp_params': Offload bf16 params only (after CUDA IPC sync)
    """
    if isinstance(body, dict) and body.get("mode") is not None:
        mode = str(body["mode"])
    _verify_job(job_id, "training")
    workers = app.state.training_workers
    loop = asyncio.get_running_loop()
    if mode == "non_lp":
        refs = [w.offload_non_lp_states.remote() for w in workers]
    elif mode == "lp_params":
        refs = [w.offload_lp_params.remote() for w in workers]
    else:
        refs = [w.offload_to_cpu.remote() for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
    logger.info("Offload training (mode=%s): %s", mode, results)
    return {"job_id": job_id, "workers": results}


@app.post("/wake-training")
async def wake_training(job_id: int, body: dict | None = Body(None)):
    """Reload all training state to GPU (wake training workers)."""
    _verify_job(job_id, "training")
    workers = app.state.training_workers
    loop = asyncio.get_running_loop()
    refs = [w.backload_to_gpu.remote() for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
    logger.info("Wake training: %s", results)
    return {"job_id": job_id, "workers": results}


@app.post("/sleep-log-prob")
async def sleep_log_prob(job_id: int):
    """Offload the reference (log-prob) DeepSpeed engine to CPU.

    No-op when the log-prob engine is vLLM or no separate log-prob job exists.
    """
    _verify_job(job_id, "log_prob")
    workers = app.state.log_prob_workers
    if not workers:
        return {"job_id": job_id, "workers": []}
    loop = asyncio.get_running_loop()
    refs = [w.offload_to_cpu.remote() for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
    logger.info("Offload log_prob: %s", results)
    return {"job_id": job_id, "workers": results}


@app.post("/wake-log-prob")
async def wake_log_prob(job_id: int):
    """Reload the reference (log-prob) DeepSpeed engine to GPU.

    No-op when the log-prob engine is vLLM or no separate log-prob job exists.
    """
    _verify_job(job_id, "log_prob")
    workers = app.state.log_prob_workers
    if not workers:
        return {"job_id": job_id, "workers": []}
    loop = asyncio.get_running_loop()
    refs = [w.backload_to_gpu.remote() for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
    logger.info("Wake log_prob: %s", results)
    return {"job_id": job_id, "workers": results}


@app.post("/generate")
async def generate(job_id: int, body: bytes = Body(..., media_type="application/octet-stream")):
    # DSSST1 octet in/out, matching Cortex/SnowAPI's generate wire (no tensors in
    # the request, but the endpoint speaks the shared binary wire either way).
    _verify_job(job_id, "sampling")
    request = GenerateRequest(**wire.loads(body))
    pool: ReplicaPool = app.state.sampling_pool
    results = await pool.generate(request.prompts, request.sampling_params, strict=request.strict)
    return Response(content=wire.dumps({"job_id": job_id, "results": results}), media_type="application/octet-stream")


@app.post("/weight-norm")
async def weight_norm(request: WeightNormRequest = Body(...)):
    """Global L2 weight norm of the training (DeepSpeed) and sampling (vLLM) engines.

    Both are sqrt of the sum of squares over all params -- invariant to each engine's sharding/fusion -- so after a
    weight sync the two values must match. Used by tests to verify sync correctness.
    """
    _verify_job(request.training_job_id, "training")
    _verify_job(request.sampling_job_id, "sampling")
    workers = app.state.training_workers
    pool: ReplicaPool = app.state.sampling_pool
    loop = asyncio.get_running_loop()
    training = await loop.run_in_executor(None, ray.get, workers[0].weight_norm.remote())
    sampling = await pool.compute_weight_norm()
    return {
        "training_norm": training["norm"],
        "sampling_norm": sampling["norm"],
        "training_num_params": training["num_params"],
        "sampling_num_params": sampling["num_params"],
    }


@app.post("/weight-sync")
async def weight_sync(job_id: int, request: WeightSyncRequest = Body(...)):
    """Sync training model weights to the sampling engine.

    Matches Cortex's `weight_sync(job_id, source_sub_job_id, target_sub_job_ids)`;
    on-prem treats a sub_job_id as its plain job id (source == training, target == sampling).

    Uses NCCL for non-colocated mode (separate GPUs).  In colocated mode:
    - cuda_ipc=True: CUDA IPC (zero-copy, requires training weights on GPU)
    - cuda_ipc=False: CPU file path (slower, works when offloaded)
    """
    training_job_id = request.source_sub_job_id
    sampling_job_id = request.target_sub_job_ids[0]
    _verify_job(training_job_id, "training")
    _verify_job(sampling_job_id, "sampling")

    workers = app.state.training_workers
    pool: ReplicaPool = app.state.sampling_pool
    # colocate is a server-launch property, never a per-call arg.
    colocate = app.state.colocate
    # Strategy comes from the training job (set at init); a non-None request value overrides it.
    training_job_info = app.state.jobs.get(training_job_id) or {}
    cuda_ipc = request.cuda_ipc if request.cuda_ipc is not None else training_job_info.get("cuda_ipc", False)
    low_memory = request.low_memory if request.low_memory is not None else training_job_info.get("low_memory", False)

    if colocate:
        lp_pool = app.state.log_prob_pool
        if cuda_ipc:
            if low_memory:
                print("colo _sync_weights_cuda_ipc_low_mem")
                results = await _sync_weights_cuda_ipc_low_mem(workers, pool, lp_pool)
            else:
                print("colo _sync_weights_cuda_ipc")
                results = await _sync_weights_cuda_ipc(workers, pool, lp_pool)
        else:
            print("colo _sync_weights_ipc")
            sync_path = training_job_info.get("sync_path", None)
            assert sync_path is not None, f"sync_path is required for training job {training_job_id}"
            results = await _sync_weights_ipc(sync_path, workers, pool, lp_pool)
    else:
        print("colo _sync_weights_nccl")
        results = await _sync_weights_nccl(workers, pool)

    return {"job_id": training_job_id, **results}

    # if colocate:
    #     lp_pool = app.state.log_prob_pool
    #     if request.cuda_ipc:
    #         return await _sync_weights_cuda_ipc(workers, pool, lp_pool)
    #     return await _sync_weights_ipc(workers, pool, lp_pool)
    # return await _sync_weights_nccl(workers, pool)

    # schedule = TransferSchedule.build(
    #     training_sharding="dp",
    #     training_gpus=len(workers),
    #     inference_replicas=pool.num_replicas,
    #     inference_tp=pool.tp_size,
    # )

    # sender_ranks = [g.sender_train_rank for g in schedule.groups]
    # sender_ips = await asyncio.gather(
    #     *[workers[r].get_ip.remote() for r in sender_ranks]
    # )
    # group_master_addrs = {g.group_id: ip for g, ip in zip(schedule.groups, sender_ips)}

    # if not app.state.weight_sync_ready:
    #     max_param = await workers[0].max_param_bytes.remote()
    #     bucket_size = max(max_param, _WEIGHT_SYNC_BUCKET_SIZE)
    #     app.state.weight_sync_bucket_size = bucket_size

    #     await asyncio.gather(
    #         *[
    #             workers[g.sender_train_rank].init_weight_sender.remote(
    #                 g,
    #                 schedule,
    #                 group_master_addrs[g.group_id],
    #                 _WEIGHT_SYNC_BASE_PORT,
    #                 bucket_size,
    #             )
    #             for g in schedule.groups
    #         ]
    #     )
    #     app.state.weight_sync_ready = True
    #     logger.info(
    #         "Weight sync initialized: %d training GPUs -> %d replicas (tp=%d), %d NCCL group(s); sender IPs=%s",
    #         len(workers),
    #         pool.num_replicas,
    #         pool.tp_size,
    #         len(schedule.groups),
    #         group_master_addrs,
    #     )

    # bucket_size = app.state.weight_sync_bucket_size

    # groups = [
    #     {
    #         "group_id": g.group_id,
    #         "master_addr": group_master_addrs[g.group_id],
    #         "master_port": _WEIGHT_SYNC_BASE_PORT,
    #         "world_size": g.world_size,
    #         "replica_ids": g.replica_ids,
    #     }
    #     for g in schedule.groups
    # ]

    # send_tasks = [workers[g.sender_train_rank].send_weights.remote() for g in schedule.groups]
    # receive_task = pool.sync_weights(
    #     groups=groups,
    #     bucket_size=bucket_size,
    # )

    # t0 = time.monotonic()
    # await asyncio.gather(receive_task, *send_tasks)
    # logger.info("Weight sync complete in %.3fs (%d group(s))", time.monotonic() - t0, len(schedule.groups))
    # return {"status": "ok"}


async def _sync_weights_cuda_ipc(workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None) -> dict:
    """Colocated weight sync via CUDA IPC (zero-copy, same GPU)."""
    from arctic_platform.common.utils.weight_sync import sync_weights_cuda_ipc

    return await sync_weights_cuda_ipc(workers, pool, lp_pool)


async def _sync_weights_cuda_ipc_low_mem(workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None) -> dict:
    """Memory-efficient (slower) colocated weight sync via CUDA IPC."""
    from arctic_platform.common.utils.weight_sync import sync_weights_cuda_ipc_low_mem

    return await sync_weights_cuda_ipc_low_mem(workers, pool, lp_pool)


async def _sync_weights_ipc(sync_path: str, workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None) -> dict:
    """Colocated weight sync via CPU file."""
    from arctic_platform.common.utils.weight_sync import sync_weights_cpu_file

    return await sync_weights_cpu_file(sync_path, workers, pool, lp_pool)


async def _sync_weights_nccl(workers, pool: ReplicaPool) -> dict:
    """Non-colocated weight sync via NCCL (original path)."""
    schedule = _transfer_schedule_cls().build(
        training_sharding="dp",
        training_gpus=len(workers),
        inference_replicas=pool.num_replicas,
        inference_tp=pool.tp_size,
    )

    sender_ranks = [g.sender_train_rank for g in schedule.groups]
    sender_ips = await asyncio.gather(*[workers[r].get_ip.remote() for r in sender_ranks])
    group_master_addrs = {g.group_id: ip for g, ip in zip(schedule.groups, sender_ips)}

    if not app.state.weight_sync_ready:
        max_param = await workers[0].max_param_bytes.remote()
        bucket_size = max(max_param, _WEIGHT_SYNC_BUCKET_SIZE)
        app.state.weight_sync_bucket_size = bucket_size

        await asyncio.gather(
            *[
                workers[g.sender_train_rank].init_weight_sender.remote(
                    g,
                    schedule,
                    group_master_addrs[g.group_id],
                    _WEIGHT_SYNC_BASE_PORT,
                    bucket_size,
                )
                for g in schedule.groups
            ]
        )
        app.state.weight_sync_ready = True
        logger.info(
            "Weight sync initialized: %d training GPUs -> %d replicas (tp=%d), %d NCCL group(s); sender IPs=%s",
            len(workers),
            pool.num_replicas,
            pool.tp_size,
            len(schedule.groups),
            group_master_addrs,
        )

    bucket_size = app.state.weight_sync_bucket_size

    groups = [
        {
            "group_id": g.group_id,
            "master_addr": group_master_addrs[g.group_id],
            "master_port": _WEIGHT_SYNC_BASE_PORT,
            "world_size": g.world_size,
            "replica_ids": g.replica_ids,
        }
        for g in schedule.groups
    ]

    send_tasks = [workers[g.sender_train_rank].send_weights.remote() for g in schedule.groups]
    receive_task = pool.sync_weights(
        groups=groups,
        bucket_size=bucket_size,
    )

    t0 = time.monotonic()
    await asyncio.gather(receive_task, *send_tasks)
    logger.info("Weight sync complete in %.3fs (%d group(s))", time.monotonic() - t0, len(schedule.groups))
    return {"status": "ok"}


@app.post("/log-probs")
async def log_probs(job_id: int, request: LogProbsRequest = Body(...)):
    _verify_job(job_id, "log_prob")
    info = app.state.jobs[job_id]

    if request.completions is not None:
        full_texts = [p + c for p, c in zip(request.prompts, request.completions)]
    else:
        full_texts = request.prompts

    if info.get("engine") == "deepspeed":
        tokenizer = app.state.log_prob_tokenizer
        encoded = tokenizer(full_texts, return_tensors="pt", padding=True)
        workers = app.state.log_prob_workers
        # Wrap the encoded batch as the {"batch","meta","processing"} payload unpack_batch expects (the same shape
        # fwd_no_grad sends), split it across DP workers, and forward each dict shard. Empty meta -> no ZoRRO/
        # position-id rewrites, so chunk order is preserved and a plain cat reassembles the global batch.
        batch_bytes = wire.dumps(dict(batch=dict(encoded), meta={}, processing={}))
        shards, _ = http_split_batch(batch_bytes, len(workers))
        raw = await asyncio.gather(*[w.compute_log_probs.remote(s) for w, s in zip(workers, shards)])
        results = torch.cat([r.cpu() for r in raw], dim=0)
    else:
        pool: ReplicaPool = app.state.log_prob_pool
        results = await pool.generate(
            full_texts,
            {"max_tokens": 1, "temperature": 0, "prompt_logprobs": request.top_k},
        )

    return Response(content=wire.dumps({"job_id": job_id, "results": results}), media_type="application/octet-stream")


@app.get("/status")
async def status():
    return {
        "training_gpus": app.state.training_gpus,
        "sampling_gpus": app.state.sampling_gpus,
        "log_prob_gpus": app.state.log_prob_gpus,
        "jobs": {jid: info for jid, info in app.state.jobs.items()},
    }


@app.get("/job/{job_id}")
async def get_job_status(job_id: int):
    info = app.state.jobs.get(job_id)
    if info is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return info


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Arctic RL Local Server")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", "-p", type=int, default=7000)
    parser.add_argument(
        "--training-gpus",
        type=int,
        default=0,
        help="Number of GPUs for DeepSpeed training",
    )
    parser.add_argument(
        "--sampling-gpus",
        type=int,
        default=0,
        help="Number of GPUs for vLLM sampling",
    )
    parser.add_argument(
        "--log-prob-gpus",
        type=int,
        default=0,
        help="Number of GPUs for log-prob engine",
    )
    parser.add_argument(
        "--log-prob-engine",
        type=str,
        default="vllm",
        choices=["vllm", "deepspeed"],
        help="Engine backend for log-prob jobs",
    )
    parser.add_argument(
        "--colocate",
        action="store_true",
        help="Colocate all workers on the same GPUs using fractional Ray resources",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable uvicorn access logs and INFO-level server banner output (quiet by default)",
    )
    parser.add_argument(
        "--no-ray-auto-attach",
        dest="ray_auto_attach",
        action="store_false",
        help="Always start a fresh Ray cluster instead of attempting to attach to an existing one",
    )
    args = parser.parse_args()

    total = args.training_gpus + args.sampling_gpus + args.log_prob_gpus
    if total == 0:
        pr0("At least one of --training-gpus, --sampling-gpus, --log-prob-gpus must be > 0")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    init_ray_cluster(auto_attach=args.ray_auto_attach)

    app.state.training_gpus = args.training_gpus
    app.state.sampling_gpus = args.sampling_gpus
    app.state.log_prob_gpus = args.log_prob_gpus
    app.state.log_prob_engine = args.log_prob_engine
    app.state.colocate = args.colocate

    # In colocated mode, create one STRICT_PACK placement group *per Ray node*
    # (one bundle per physical GPU) rather than a single STRICT_PACK group
    # spanning every GPU in the cluster. A single cluster-wide STRICT_PACK
    # group requires all bundles to land on one node, which is unsatisfiable
    # on multi-node clusters (e.g. 16 GPUs across 2x8-GPU nodes) -- the
    # autoscaler can never fulfill {"GPU": n_bundles} on a single node, so
    # pg.ready() blocks forever and the server never becomes healthy.
    # See arctic_platform.common.utils.ray_pg for the per-node layout.
    app.state.placement = ColocatePlacement()
    app.state.placement_group = None
    app.state.n_bundles = 0
    if args.colocate:
        n_bundles = max(args.training_gpus, args.sampling_gpus, args.log_prob_gpus)
        app.state.placement = create_colocate_placement(n_bundles)
        # Back-compat views for callers that still read these attributes.
        app.state.n_bundles = app.state.placement.n_bundles
        app.state.placement_group = (
            app.state.placement.placement_groups[0] if len(app.state.placement.placement_groups) == 1 else None
        )

    if args.colocate:
        assert app.state.placement, "Placement groups must be created when colocate=True"

    app.state.training_workers = []
    # ReplicaPool pulls in arctic_inference → vLLM; only create when needed so
    # training-only SFT servers can start without the inference stack.
    app.state.sampling_pool = _replica_pool_cls()() if args.sampling_gpus > 0 else None
    if args.log_prob_gpus > 0 and args.log_prob_engine == "vllm":
        app.state.log_prob_pool = _replica_pool_cls()()
    else:
        app.state.log_prob_pool = None
    app.state.log_prob_workers = []
    app.state.log_prob_tokenizer = None
    app.state.jobs = {}
    app.state.next_job_id = 1
    app.state.weight_sync_ready = False
    app.state.weight_sync_bucket_size = _WEIGHT_SYNC_BUCKET_SIZE

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=args.verbose,
        log_level="info" if args.verbose else "warning",
    )


if __name__ == "__main__":
    main()
