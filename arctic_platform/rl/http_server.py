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

    python -m arctic_platform.rl.server \\
        --training-gpus 4 --sampling-gpus 2 --log-prob-gpus 2
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import pathlib
import sys
import time
from typing import Any
from typing import Union

import ray
import torch
import uvicorn
from arctic_inference.server.replica_pool import ReplicaPool
from arctic_inference.server.weight_sync.schedule import TransferSchedule
from fastapi import Body
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from pydantic import BaseModel
from transformers import AutoTokenizer

from arctic_platform.rl.deepspeed_worker import DeepSpeedWorker
from arctic_platform.rl.ray_cluster import init_ray_cluster
from arctic_platform.rl.server import ArcticRLServerState
from arctic_platform.rl.utils import http_split_batch
from arctic_platform.rl.utils import merge_cuda_ipc_payloads
from arctic_platform.rl.utils import merge_dict_shards
from arctic_platform.rl.utils.batch import combine_metric_shards
from arctic_platform.rl.utils.batch import restore_batch_order
from arctic_platform.rl.utils.debug import pr0
from arctic_platform.rl.utils.ray_pg import ColocatePlacement
from arctic_platform.rl.utils.ray_pg import create_colocate_placement
from arctic_platform.rl.utils.ray_pg import pg_scheduling_options
from arctic_platform.rl.tinker_server import init_tinker_state
from arctic_platform.rl.tinker_server import router as _tinker_router
from arctic_platform.rl.utils.server_models import GenerateRequest
from arctic_platform.rl.utils.server_models import JobConfig
from arctic_platform.rl.utils.server_models import LogProbsRequest
from arctic_platform.rl.utils.server_models import StepRequest
from arctic_platform.rl.utils.server_models import SyncWeightsRequest
from arctic_platform.rl.utils.server_models import WeightNormRequest
from arctic_platform.rl.utils.server_models import build_model_config

logger = logging.getLogger(__name__)

app = FastAPI(title="Arctic RL Local Server")
# Tinker HTTP layer. Routes are mounted eagerly (small Pydantic-only cost);
# in-process handlers are bound lazily in ``initialize`` once a training job
# id is known. See ``tinker_server.init_tinker_state``.
app.include_router(_tinker_router)

# ---------------------------------------------------------------------------
# Request / response models (mirrors dss-platform sftp_server)
# ---------------------------------------------------------------------------

ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy

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
            workers = []
            config_dict = job_config.model_dump()
            for rank in range(gpus):
                if colocate and placement:
                    opts = _pg_options(bundle_index=lp_bundle_offset + rank, fraction_key="log_prob")
                else:
                    opts = dict(num_gpus=1)
                w = DeepSpeedWorker.options(**opts).remote(rank, gpus, 29501)
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


@app.post("/fwd-bwd")
async def fwd_bwd(
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
    losses = [r["avg_loss"] for r in results]
    avg_loss = sum(losses)  # / len(losses)

    # See ray_server.fwd_bwd for the rationale: collapse the per-DP-rank
    # paired ``.sum`` / ``.tokens`` metric scalars into one global
    # token-mean scalar per metric per mini-batch. ``batch`` is intentionally
    # omitted -- the driver does not consume it (see note above).
    merged = dict(
        job_id=job_id,
        metrics=combine_metric_shards([r["metrics"] for r in results]),
        avg_loss=avg_loss,
    )
    timers.stop_and_print_elapsed(tname)

    timers.stop_and_print_elapsed(tname_e2e)

    buffer = io.BytesIO()
    torch.save(merged, buffer)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


@app.post("/fwd-no-grad")
async def fwd_no_grad(
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

    merged = dict(
        job_id=job_id,
        batch=batch,
        metrics=merge_dict_shards([r["metrics"] for r in results]),
    )

    buffer = io.BytesIO()
    torch.save(merged, buffer)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


@app.post("/step")
async def step(job_id: int, request: StepRequest | None = Body(default=None)):
    _verify_job(job_id, "training")
    optim_overrides = request.optim_overrides if request is not None else None
    results = await asyncio.gather(
        *[w.step.remote(optim_overrides) for w in app.state.training_workers]
    )
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


@app.post("/save-checkpoint")
async def save_checkpoint(job_id: int):
    _verify_job(job_id, "training")
    info = app.state.jobs[job_id]
    path = info.get("checkpoint_path", None)
    assert path is not None, f"checkpoint_path is required for training job {job_id}"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    await asyncio.gather(
        *[w.save_checkpoint.remote(path) for w in app.state.training_workers],
    )
    return {"job_id": job_id, "path": path}


@app.post("/sleep-inference")
async def sleep_inference(job_id: int, level: int):
    """Put all inference engines to sleep, freeing GPU memory."""
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
async def wake_inference(job_id: int, tags: list[str] | None = None):
    """Wake all inference engines, restoring GPU memory."""
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
async def reset_prefix_cache(job_id: int):
    """Reset the prefix cache on the sampling inference engines."""
    _verify_job(job_id, "sampling")
    results = {}
    pool: ReplicaPool = app.state.sampling_pool
    results["sampling"] = await pool.reset_prefix_cache()
    lp_pool: ReplicaPool | None = app.state.log_prob_pool
    if lp_pool is not None and lp_pool._config is not None:
        results["log_prob"] = await lp_pool.reset_prefix_cache()
    return {"job_id": job_id, **results}


@app.post("/sleep-training")
async def sleep_training(job_id: int, mode: str = "all"):
    """Offload training state to CPU (sleep training workers).

    mode='all':       Offload everything (for training → inference transition)
    mode='non_lp':    Keep bf16 params on GPU, offload rest (before CUDA IPC sync)
    mode='lp_params': Offload bf16 params only (after CUDA IPC sync)
    """
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
async def wake_training(job_id: int):
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
async def generate(job_id: int, request: GenerateRequest = Body(...)):
    _verify_job(job_id, "sampling")
    pool: ReplicaPool = app.state.sampling_pool
    results = await pool.generate(request.prompts, request.sampling_params)
    return {"job_id": job_id, "results": results}


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


@app.post("/sync-weights")
async def sync_weights(request: SyncWeightsRequest = Body(...)):
    """Sync training model weights to the sampling engine.

    Uses NCCL for non-colocated mode (separate GPUs).  In colocated mode:
    - cuda_ipc=True: CUDA IPC (zero-copy, requires training weights on GPU)
    - cuda_ipc=False: CPU file path (slower, works when offloaded)
    """
    _verify_job(request.training_job_id, "training")
    _verify_job(request.sampling_job_id, "sampling")

    workers = app.state.training_workers
    pool: ReplicaPool = app.state.sampling_pool
    colocate = request.colocate or app.state.colocate

    if colocate:
        lp_pool = app.state.log_prob_pool
        if request.cuda_ipc:
            if request.low_memory:
                print("colo _sync_weights_cuda_ipc_low_mem")
                results = await _sync_weights_cuda_ipc_low_mem(workers, pool, lp_pool)
            else:
                print("colo _sync_weights_cuda_ipc")
                results = await _sync_weights_cuda_ipc(workers, pool, lp_pool)
        else:
            print("colo _sync_weights_ipc")
            training_job_info = app.state.jobs.get(request.training_job_id)
            sync_path = training_job_info.get("sync_path", None)
            assert sync_path is not None, f"sync_path is required for training job {request.training_job_id}"
            results = await _sync_weights_ipc(sync_path, workers, pool, lp_pool)
    else:
        print("colo _sync_weights_nccl")
        results = await _sync_weights_nccl(workers, pool)

    # await self.arctic_rl_ray_server_state.reset_prefix_cache.remote(request.sampling_job_id)

    return {"job_id": request.training_job_id, **results}

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
    """Colocated weight sync via CUDA IPC (zero-copy, same GPU).

    For ZeRO-3: all workers call gather_cuda_ipc_handles collectively
    (GatheredParameters is a collective op).  Every rank produces IPC
    handles for the full weight on its own physical GPU; we merge the
    per-rank, per-parameter handle dicts so each colocated inference
    replica (on a distinct GPU) finds a handle for its GPU.

    For ZeRO-2: falls back to get_cuda_ipc_handles on each rank
    (params are already full on every rank, no collective needed).
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    # All workers must participate for ZeRO-3 collective gather.
    # gather_cuda_ipc_handles is safe for ZeRO-2 too (no ds_id → no gather).
    gather_refs = [w.gather_cuda_ipc_handles.remote() for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in gather_refs])
    ipc_payload = merge_cuda_ipc_payloads(results)
    num_params = ipc_payload.get("num_params", 0)

    # Staged wake: weights only → IPC load → KV cache (reduces peak GPU memory).
    # unlike ray_server, all these are done from http_client

    recv_tasks = []
    total_replicas = 0
    for p in [pool, lp_pool]:
        if p is None or p._config is None:
            continue
        for rid in range(p.num_replicas):
            w = p._workers[rid]
            recv_tasks.append(loop.run_in_executor(None, ray.get, w.load_weights_cuda_ipc.remote(ipc_payload)))
            total_replicas += 1
    await asyncio.gather(*recv_tasks)

    # Release IPC tensor refs on every rank (each rank holds its own GPU's
    # cloned weights alive for the duration of the sync).
    await asyncio.gather(*[loop.run_in_executor(None, ray.get, w.release_ipc_handles.remote()) for w in workers])

    elapsed = time.monotonic() - t0
    logger.info(
        "Weight sync (CUDA IPC) complete in %.3fs (%d replica(s), %d params)", elapsed, total_replicas, num_params
    )
    return {"status": "ok"}


async def _sync_weights_cuda_ipc_low_mem(workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None) -> dict:
    """Memory-efficient (slower) colocated weight sync via CUDA IPC.

    Streams one parameter at a time: all training ranks collectively gather a
    single ZeRO-3 param onto their own GPU, the colocated inference replicas
    copy it in, then the source IPC ref is released before moving on. Peak extra
    GPU memory is one full parameter per GPU (instead of the whole model as in
    ``_sync_weights_cuda_ipc``), at the cost of many more round-trips.

    Selected via ``arctic_rl.low_memory_weight_sync=True``.

    The staged wake (weights with restore_weights → kv_cache) is driven by the
    http_client around the /sync-weights call (same as ``_sync_weights_cuda_ipc``);
    the weights wake restores full-shape vLLM params so the per-param chunk copy
    lands on real storage rather than offloaded [1] stubs.
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    # Enumerate parameter names once (only names cross the Ray boundary; each
    # worker resolves its own live param by name inside get_cuda_ipc_handle so
    # the ZeRO-3 gather stays correct).
    param_names = ray.get(workers[0].get_parameter_names.remote())
    num_params = len(param_names)

    # Flatten the colocated inference replicas (sampling + optional log-prob).
    replicas = []
    for p in [pool, lp_pool]:
        if p is None or p._config is None:
            continue
        for rid in range(p.num_replicas):
            replicas.append(p._workers[rid])

    all_names: list = []
    for idx, name in enumerate(param_names):
        # Collective single-param gather across all training ranks.
        gather_refs = [w.get_cuda_ipc_handle.remote(name) for w in workers]
        results = ray.get(gather_refs)
        payload = merge_cuda_ipc_payloads(results)
        all_names.extend(payload["names"])

        # On the final param pass the full name list so each receiver can
        # validate the complete architecture match in one shot.
        validate = all_names if idx == num_params - 1 else None

        recv_tasks = [
            loop.run_in_executor(None, ray.get, w.load_weights_cuda_ipc_chunk.remote(payload, validate))
            for w in replicas
        ]
        await asyncio.gather(*recv_tasks)

        # Release this param's source ref on every training rank before the next
        # gather, bounding peak memory to one full param per GPU.
        await asyncio.gather(*[loop.run_in_executor(None, ray.get, w.release_ipc_handles.remote()) for w in workers])

    elapsed = time.monotonic() - t0
    logger.info(
        "Weight sync (CUDA IPC, low-mem) complete in %.3fs (%d replica(s), %d params)",
        elapsed,
        len(replicas),
        num_params,
    )
    return {"status": "ok"}


async def _sync_weights_ipc(sync_path: str, workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None) -> dict:
    """Colocated weight sync via CPU file.

    For ZeRO-3: all workers call gather_and_save_state_dict collectively
    (GatheredParameters requires all ranks).  Only rank 0 writes the file.
    For ZeRO-2: only rank 0 saves (params are already full).
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    # All workers must participate for ZeRO-3 collective gather
    save_refs = [w.gather_and_save_state_dict.remote(sync_path) for w in workers]
    results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in save_refs])
    num_params = results[0].get("num_params", 0)

    recv_tasks = []
    total_replicas = 0
    for p in [pool, lp_pool]:
        if p is None or p._config is None:
            continue
        for rid in range(p.num_replicas):
            w = p._workers[rid]
            recv_tasks.append(loop.run_in_executor(None, ray.get, w.load_weights_from_shm_path.remote(sync_path)))
            total_replicas += 1
    await asyncio.gather(*recv_tasks)

    pathlib.Path(sync_path).unlink(missing_ok=True)

    elapsed = time.monotonic() - t0
    logger.info(
        "Weight sync (CPU→GPU) complete in %.3fs (%d replica(s), %d params)", elapsed, total_replicas, num_params
    )
    return {"status": "ok"}


async def _sync_weights_nccl(workers, pool: ReplicaPool) -> dict:
    """Non-colocated weight sync via NCCL (original path)."""
    schedule = TransferSchedule.build(
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
        batch_buf = io.BytesIO()
        torch.save(dict(batch=dict(encoded), meta={}, processing={}), batch_buf)
        shards, _ = http_split_batch(batch_buf.getvalue(), len(workers))
        raw = await asyncio.gather(*[w.compute_log_probs.remote(s) for w, s in zip(workers, shards)])
        results = torch.cat([r.cpu() for r in raw], dim=0)
    else:
        pool: ReplicaPool = app.state.log_prob_pool
        results = await pool.generate(
            full_texts,
            {"max_tokens": 1, "temperature": 0, "prompt_logprobs": request.top_k},
        )

    buffer = io.BytesIO()
    torch.save({"job_id": job_id, "results": results}, buffer)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


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


class TinkerBindRequest(BaseModel):
    """Bind the Tinker HTTP surface onto two already-provisioned jobs.

    Provisioning (ZoRRo, ZeRO stage, offload, vLLM knobs) is Arctic's
    normal ``/initialize`` path; this endpoint is a pure adapter."""
    training_job_id: int
    sampling_job_id: int
    base_model: str
    max_prompt_length: int = 1024
    max_response_length: int = 512


@app.post("/tinker/bind")
async def tinker_bind(request: TinkerBindRequest = Body(...)):
    _verify_job(request.training_job_id, "training")
    _verify_job(request.sampling_job_id, "sampling")
    if getattr(app.state, "tinker_base_model", None) is not None:
        raise HTTPException(409, "Tinker layer already bound")

    tokenizer = AutoTokenizer.from_pretrained(request.base_model)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    training_job_id = request.training_job_id
    sampling_job_id = request.sampling_job_id

    def _batch_np_to_torch(batch: dict) -> dict:
        """Numpy → torch so Arctic's ``http_split_batch``
        (``torch.load(weights_only=True)``) can round-trip the payload."""
        import numpy as np
        out = dict(batch)
        b = dict(batch.get("batch", {}))
        for k, v in b.items():
            if isinstance(v, np.ndarray):
                b[k] = torch.from_numpy(v)
        out["batch"] = b
        return out

    async def _fwd_bwd_handler(batch: dict) -> dict:
        buf = io.BytesIO()
        torch.save(_batch_np_to_torch(batch), buf)
        resp = await fwd_bwd(training_job_id, buf.getvalue())
        return torch.load(io.BytesIO(resp.body), weights_only=False)

    async def _fwd_no_grad_handler(batch: dict) -> dict:
        buf = io.BytesIO()
        torch.save(_batch_np_to_torch(batch), buf)
        resp = await fwd_no_grad(training_job_id, buf.getvalue())
        return torch.load(io.BytesIO(resp.body), weights_only=False)

    async def _step_handler(overrides: dict | None) -> dict:
        return await step(training_job_id, StepRequest(optim_overrides=overrides))

    async def _sync_weights_handler() -> Any:
        return await sync_weights(SyncWeightsRequest(
            training_job_id=training_job_id,
            sampling_job_id=sampling_job_id,
            colocate=app.state.colocate,
            cuda_ipc=app.state.colocate,
        ))

    async def _generate_handler(prompt_tokens: list[int], sampling_params: dict) -> dict:
        # Fan the group into N single-sample calls (Arctic's replica-pool
        # worker returns ``outputs[0]``) and let vLLM's prefix cache dedupe
        # the shared prompt KV. Passes token-id prompts straight through.
        params = dict(sampling_params)
        n = int(params.pop("n", 1))
        prompts: list[Any] = [list(prompt_tokens)] * n
        pool: ReplicaPool = app.state.sampling_pool
        results = await pool.generate(prompts, params)
        outputs: list[dict[str, Any]] = []
        for res in results:
            per_pos = res.get("logprobs")
            flat_lp: list[float] | None = None
            if per_pos is not None:
                flat_lp = []
                for tok, pos in zip(res.get("token_ids", []), per_pos):
                    if not isinstance(pos, dict):
                        continue
                    entry = pos.get(tok, pos.get(str(tok)))
                    if isinstance(entry, dict):
                        flat_lp.append(float(entry.get("logprob", 0.0)))
                    elif isinstance(entry, (int, float)):
                        flat_lp.append(float(entry))
            outputs.append({
                "token_ids": list(res.get("token_ids", [])),
                "logprobs": flat_lp,
                "finish_reason": res.get("finish_reason", "length"),
            })
        return {"outputs": outputs}

    init_tinker_state(
        app,
        base_model=request.base_model,
        max_prompt_length=int(request.max_prompt_length),
        max_response_length=int(request.max_response_length),
        pad_token_id=int(pad_token_id),
        fwd_bwd_handler=_fwd_bwd_handler,
        fwd_no_grad_handler=_fwd_no_grad_handler,
        step_handler=_step_handler,
        sync_weights_handler=_sync_weights_handler,
        generate_handler=_generate_handler,
    )
    logger.info(
        "Tinker layer bound: training_job_id=%d sampling_job_id=%d base_model=%s",
        training_job_id, sampling_job_id, request.base_model,
    )
    return {
        "training_job_id": training_job_id,
        "sampling_job_id": sampling_job_id,
        "base_model": request.base_model,
        "max_prompt_length": int(request.max_prompt_length),
        "max_response_length": int(request.max_response_length),
    }


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
    # See arctic_platform.rl.utils.ray_pg for the per-node layout.
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
    app.state.sampling_pool = ReplicaPool()
    if args.log_prob_engine == "vllm":
        app.state.log_prob_pool = ReplicaPool()
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
