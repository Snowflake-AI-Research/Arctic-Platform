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

"""Local RL server using Ray to manage DeepSpeed workers and ArcticInference ReplicaPools.

Uses Ray to manage DeepSpeed workers and ArcticInference ReplicaPools.

Usage::

    python -m arctic_platform.rl.server \\
        --training-gpus 4 --sampling-gpus 2 --log-prob-gpus 2
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import pathlib
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import ray
import torch
from arctic_inference.server.config import ModelConfig
from arctic_inference.server.replica_pool import ReplicaPool
from arctic_inference.server.weight_sync.schedule import TransferSchedule
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer

from arctic_platform.rl.deepspeed_worker import DeepSpeedWorker
from arctic_platform.rl.ray_cluster import init_ray_cluster
from arctic_platform.rl.server import ArcticRLServerState
from arctic_platform.rl.utils import combine_metric_shards
from arctic_platform.rl.utils import log_dp_shard_tokens
from arctic_platform.rl.utils import merge_cuda_ipc_payloads
from arctic_platform.rl.utils import merge_dict_shards
from arctic_platform.rl.utils import ray_split_batch
from arctic_platform.rl.utils import unpack_batch
from arctic_platform.rl.utils.batch import restore_batch_order
from arctic_platform.rl.utils.debug import ProfilerContext
from arctic_platform.rl.utils.debug import pr0
from arctic_platform.rl.utils.ray_pg import ColocatePlacement
from arctic_platform.rl.utils.ray_pg import create_colocate_placement
from arctic_platform.rl.utils.ray_pg import pg_scheduling_options

logger = logging.getLogger(__name__)

# PROFILER_TYPE = "c"
# PROFILER_TYPE = "torch"
PROFILER_TYPE = "none"

ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


class JobConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str
    job_type: str = Field(default="training")
    num_devices: Optional[int] = None
    ds_config: Optional[dict] = None
    training_config: Optional[dict] = None
    log_prob_config: Optional[dict] = None
    ds_worker_config: Optional[dict] = None
    vllm_config: Optional[dict] = None
    checkpoint_path: Optional[str] = None
    use_arctic_inference: bool = False
    full_determinism: bool = False
    seed: int = 42


class GenerateRequest(BaseModel):
    prompts: List[str]
    sampling_params: Optional[Dict[str, Any]] = None


class LogProbsRequest(BaseModel):
    prompts: List[str]
    completions: Optional[List[str]] = None
    top_k: int = 1


class SyncWeightsRequest(BaseModel):
    training_job_id: int
    sampling_job_id: int
    colocate: bool = False
    cuda_ipc: bool = False
    low_memory: bool = False


def _build_model_config(model_name: str, vllm_config: dict | None) -> ModelConfig:
    """Construct a :class:`ModelConfig` from user-supplied vllm_config dict."""
    cfg = dict(vllm_config or {})
    cfg["model"] = model_name
    known_fields = set(ModelConfig.model_fields.keys())
    extra = {k: v for k, v in cfg.items() if k not in known_fields}
    base = {k: v for k, v in cfg.items() if k in known_fields}
    if extra:
        base["extra_engine_kwargs"] = extra
    return ModelConfig(**base)


# Honor ARL_WEIGHT_SYNC_PORT when set so back-to-back / concurrent training jobs on one host (e.g. repeated
# pytest-flakefinder iterations or parallel xdist workers) get a fresh NCCL rendezvous port instead of all reusing
# 29600, where a SIGKILL-reaped sender from a prior job can still squat the port and deadlock the next sync.
_WEIGHT_SYNC_BASE_PORT = int(os.environ.get("ARL_WEIGHT_SYNC_PORT", 29600))
_WEIGHT_SYNC_BUCKET_SIZE = 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def create_arctic_rl_ray_server_state(**kwargs):
    sched_pg = placement_group([{"GPU": 0, "CPU": 1}])
    return ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=sched_pg,
            # Must be False: InferenceWorker / DeepSpeed actors need real GPU
            # bundles; inheriting this CPU-only PG would raise ValueError at spawn.
            placement_group_capture_child_tasks=False,
        ),
    )(ArcticRLRayServerState).remote(**kwargs)


# TODO: add remote decorator
class ArcticRLRayServerState(ArcticRLServerState):
    def __init__(
        self,
        training_gpus: int,
        sampling_gpus: int,
        log_prob_gpus: int,
        log_prob_engine: str,
        colocate: bool,
    ):
        total_gpus = training_gpus + sampling_gpus + log_prob_gpus
        if total_gpus == 0:
            raise ValueError("At least one of --training-gpus, --sampling-gpus, --log-prob-gpus must be > 0")

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

        pr0("[ArcticRLRayServer] initializing ray cluster")
        # This server-state object runs as a Ray actor *inside* the cluster the
        # driver already created, so it must attach to that cluster (auto_attach
        # resolves to its own cluster), not start a fresh head. (Previously this
        # passed auto_attach=False but the flag was ignored by a hardcoded branch
        # in init_ray_cluster, so the effective behavior was always to attach.)
        init_ray_cluster(auto_attach=True)
        pr0("[ArcticRLRayServer] ray cluster initialized")

        self.training_gpus = training_gpus
        self.sampling_gpus = sampling_gpus
        self.log_prob_gpus = log_prob_gpus
        self.log_prob_engine = log_prob_engine
        self.colocate = colocate

        # In colocated mode, create one STRICT_PACK placement group per Ray
        # node so each TP=tp group is guaranteed to live on a single physical
        # node (see :mod:`arctic_platform.rl.utils.ray_pg`).
        self.placement: ColocatePlacement = ColocatePlacement()
        if self.colocate:
            n_bundles = max(self.training_gpus, self.sampling_gpus, self.log_prob_gpus)
            self.placement = create_colocate_placement(n_bundles)
            # Back-compat: callers that still read `self.placement_group` /
            # `self.placement_groups` / `self.n_bundles` get a meaningful view
            # of the new layout.
            self.placement_groups = self.placement.placement_groups
            self.gpus_per_node = self.placement.gpus_per_node
            self.n_bundles = self.placement.n_bundles
            self.placement_group = (
                self.placement.placement_groups[0] if len(self.placement.placement_groups) == 1 else None
            )
        else:
            self.placement_groups = []
            self.gpus_per_node = 0
            self.n_bundles = 0
            self.placement_group = None

        if self.colocate:
            assert self.placement, "Placement groups must be created when colocate=True"

        self.training_workers = []
        self.sampling_pool = ReplicaPool()
        if self.log_prob_engine == "vllm":
            self.log_prob_pool = ReplicaPool()
        else:
            self.log_prob_pool = None
        self.log_prob_workers = []
        self.log_prob_tokenizer = None
        self.jobs = {}
        self.next_job_id = 1
        self.weight_sync_ready = False
        self.weight_sync_bucket_size = _WEIGHT_SYNC_BUCKET_SIZE

        pr0("[ArcticRLRayServerState] initialized")

    async def get_jobs(self) -> dict[str, Any]:
        return self.jobs

    async def get_training_workers(self) -> list[DeepSpeedWorker]:
        return self.training_workers

    async def get_sampling_pool(self) -> ReplicaPool:
        return self.sampling_pool

    async def get_log_prob_pool(self) -> ReplicaPool:
        return self.log_prob_pool

    async def get_log_prob_workers(self) -> list[DeepSpeedWorker]:
        return self.log_prob_workers

    async def get_log_prob_tokenizer(self) -> AutoTokenizer:
        return self.log_prob_tokenizer

    async def get_next_job_id(self) -> int:
        return self.next_job_id

    async def get_weight_sync_ready(self) -> bool:
        return self.weight_sync_ready

    async def get_weight_sync_bucket_size(self) -> int:
        return self.weight_sync_bucket_size

    async def get_colocate(self) -> bool:
        return self.colocate

    async def sleep_inference(self, job_id: int, level: int) -> dict[str, Any]:
        colocate = self.colocate
        results = {}
        pool: ReplicaPool = self.sampling_pool
        # Let vLLM's CuMemAllocator free the weights (offload_weights=False)
        # instead of the legacy manual offload, which reallocated param.data on
        # each wake and changed weight addresses -> stale rollout CUDA graphs
        # (compile on) -> grad-norm explosion. cumem keeps addresses stable.
        offload_weights = False
        results["sampling"] = await pool.sleep(level=level, offload_weights=offload_weights)
        lp_pool: ReplicaPool | None = self.log_prob_pool
        if lp_pool is not None and lp_pool._config is not None and not lp_pool.sleeping:
            results["log_prob"] = await lp_pool.sleep(level=level, offload_weights=offload_weights)
        if colocate:
            await pool.close_weight_sync()
            if lp_pool is not None and lp_pool._config is not None:
                await lp_pool.close_weight_sync()
        return {"job_id": job_id, **results}

    async def _empty_training_cache(self, workers: list[DeepSpeedWorker]):
        loop = asyncio.get_running_loop()
        refs = [w.empty_cache.remote() for w in workers]
        results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
        return results

    async def wake_inference(self, tags: list[str] | None = None, restore_weights: bool | None = None):
        """Wake all inference engines, restoring GPU memory.

        ``restore_weights`` overrides the default of restoring the CPU-offloaded
        weights on a weights wake. Callers about to overwrite every weight (e.g.
        the CUDA IPC sync) pass ``False`` to skip a redundant full-model CPU->GPU
        copy that would otherwise double peak GPU memory at the wake.
        """
        if self.colocate:
            # Release cached GPU pages so cumem can remap.
            await self._empty_training_cache(self.training_workers)
        colocate = self.colocate
        if restore_weights is None:
            restore = colocate and (tags is None or "weights" in tags)
        else:
            restore = restore_weights
        results = {}
        pool: ReplicaPool = self.sampling_pool
        results["sampling"] = await pool.wake_up(tags=tags, restore_weights=restore)
        lp_pool: ReplicaPool | None = self.log_prob_pool
        if lp_pool is not None and lp_pool._config is not None and lp_pool.sleeping:
            results["log_prob"] = await lp_pool.wake_up(tags=tags, restore_weights=restore)
        return results

    async def reset_prefix_cache(self, job_id: int) -> dict[str, Any]:
        """Reset the prefix cache on the sampling inference engines."""
        results = {}
        pool: ReplicaPool = self.sampling_pool
        results["sampling"] = await pool.reset_prefix_cache()
        lp_pool: ReplicaPool | None = self.log_prob_pool
        if lp_pool is not None and lp_pool._config is not None:
            results["log_prob"] = await lp_pool.reset_prefix_cache()
        return {"job_id": job_id, **results}

    async def initialize(self, job_config: dict[str, Any]) -> dict[str, Any]:
        job_config = JobConfig(**job_config)
        pr0(f"[ArcticRLRayServer] initialize: {job_config=}")
        job_type = job_config.job_type
        job_id = self.next_job_id
        self.next_job_id += 1

        colocate = self.colocate
        placement: ColocatePlacement = getattr(self, "placement", ColocatePlacement())

        # Fractional GPU fractions within each PG bundle.  Each bundle owns 1
        # physical GPU; fractions let multiple actors share that bundle while
        # Ray still sets CUDA_VISIBLE_DEVICES so each actor can see the GPU.
        # These are *Ray scheduling accounting* only (not memory caps): real
        # VRAM is time-shared via sleep/wake/offload.  All actors that share a
        # bundle must sum to <= 1.0, so with full 3-way colocation:
        #   training (0.34) + sampling (0.33) + log_prob (0.33) = 1.0
        _COLOCATE_GPU_FRACTIONS = {"sampling": 0.33, "log_prob": 0.33, "training": 0.34}

        def _pg_options(bundle_index: int, fraction_key: str) -> dict:
            """PG-pinned scheduling: fractional GPU claim inside a specific bundle."""
            return pg_scheduling_options(
                placement,
                bundle_index,
                _COLOCATE_GPU_FRACTIONS[fraction_key],
            )

        # n_bundles = placement.n_bundles
        # n_sample = self.sampling_gpus
        # n_logprob = self.log_prob_gpus

        # Bundle layout (deterministic), full 3-way colocation:
        #   training pins rank r        → bundle r            [0 .. training_gpus-1]
        #   sampling replica r (TP=tp)  → bundles [r*tp .. r*tp+tp-1]
        #   log_prob pins rank r        → bundle r            [0 .. log_prob_gpus-1]
        # All three overlap on the same bundles (offset 0), so each physical
        # GPU hosts a training rank, a sampling worker, and a log_prob rank.
        # n_bundles = max(training_gpus, sampling_gpus, log_prob_gpus).

        if job_type == "training":
            gpus = self.training_gpus
            if gpus == 0:
                raise ValueError("No training GPUs configured")
            if self.training_workers:
                raise ValueError("Training job already running")
            # Honor MASTER_PORT when set so concurrent training jobs on one host
            # (e.g. parallel pytest-xdist workers, each with its own cluster) don't
            # collide on the rendezvous port. All ranks of THIS job are handed the
            # same value below, so multi-node rendezvous is unaffected.
            master_port = int(os.environ.get("MASTER_PORT", 29500))
            workers = []
            config_dict = job_config.model_dump()
            for rank in range(gpus):
                if colocate and placement:
                    opts = _pg_options(bundle_index=rank, fraction_key="training")
                else:
                    opts = dict(num_gpus=1)
                w = DeepSpeedWorker.options(**opts).remote(rank, gpus, master_port)
                workers.append(w)
            master_addr = await workers[0].get_ip.remote()
            await asyncio.gather(*[w.initialize.remote(master_addr, config_dict) for w in workers])
            self.training_workers = workers

        elif job_type == "sampling":
            gpus = self.sampling_gpus
            if gpus == 0:
                raise ValueError("No sampling GPUs configured")
            pool: ReplicaPool = self.sampling_pool
            if pool._config is not None:
                raise ValueError("Sampling job already running")
            vllm_cfg = dict(job_config.vllm_config or {})
            if colocate:
                vllm_cfg["enable_sleep_mode"] = True
            model_cfg = _build_model_config(job_config.model_name, vllm_cfg)
            tp = model_cfg.tensor_parallel_size
            num_replicas = gpus // tp
            if colocate and placement:
                per_replica_pgs, bundle_indices = placement.tp_layout(num_replicas, tp)
                extra_env = {}
                if tp > 1:
                    extra_env["VLLM_RAY_PER_WORKER_GPUS"] = str(_COLOCATE_GPU_FRACTIONS["sampling"])
                    vllm_cfg["distributed_executor_backend"] = "ray"
                    model_cfg = _build_model_config(job_config.model_name, vllm_cfg)
                if job_config.use_arctic_inference:
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
            gpus = self.log_prob_gpus
            if gpus == 0:
                raise ValueError("No log-prob GPUs configured")

            # Full 3-way colocation: log_prob ranks share the same bundles as
            # training (and sampling), so offset 0. The reference engine is
            # offloaded right after init and only woken for the ref-logprob
            # pass, so it does not contend for VRAM with training/sampling.
            lp_bundle_offset = 0

            if job_config.ds_config is not None:
                if self.log_prob_workers:
                    raise ValueError("Log-prob job already running")
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
                self.log_prob_workers = workers
                self.log_prob_tokenizer = AutoTokenizer.from_pretrained(job_config.model_name)
                engine = "deepspeed"
                # The reference engine starts resident on GPU. Whether it is
                # offloaded between ref-logprob passes is decided by the client
                # (ArcticRLClientWrapper, via ref.fsdp_config.param_offload),
                # which drives wake_log_prob/sleep_log_prob on demand.
            else:
                pool: ReplicaPool = self.log_prob_pool
                if pool._config is not None:
                    raise ValueError("Log-prob job already running")
                lp_vllm_cfg = dict(job_config.vllm_config or {})
                if colocate:
                    lp_vllm_cfg["enable_sleep_mode"] = True
                model_cfg = _build_model_config(job_config.model_name, lp_vllm_cfg)
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
                        model_cfg = _build_model_config(job_config.model_name, lp_vllm_cfg)
                    if job_config.use_arctic_inference:
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
            raise ValueError(f"Unknown job type: {job_type}")

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
            job_info["checkpoint_path"] = os.path.join(job_config.checkpoint_path, f"arctic_rl_job_{job_id}")
            os.makedirs(job_info["checkpoint_path"], exist_ok=True)
            job_info["sync_path"] = os.path.join(job_info["checkpoint_path"], "weight_sync.pt")

        self.jobs[job_id] = job_info
        return {"job_id": job_id, "job_type": job_type, "running": True}

    async def destroy(self, job_id: int, job_type: str) -> dict[str, Any]:
        info = self.jobs.pop(job_id, None)
        if info is None:
            raise ValueError(f"Job {job_id} not found")

        if info["job_type"] == "training":
            await asyncio.gather(*[w.destroy.remote() for w in self.training_workers])
            # refs = [w.destroy.remote() for w in self.training_workers]
            # _ = ray.get(refs)
            self.training_workers.clear()
        elif info["job_type"] == "sampling":
            await self.sampling_pool.shutdown()
            # asyncio.run(self.sampling_pool.shutdown())
        elif info["job_type"] == "log_prob":
            if info.get("engine") == "deepspeed":
                await asyncio.gather(*[w.destroy.remote() for w in self.log_prob_workers])
                # refs = [w.destroy.remote() for w in self.log_prob_workers]
                # _ = ray.get(refs)
                self.log_prob_workers.clear()
                self.log_prob_tokenizer = None
            else:
                await self.log_prob_pool.shutdown()
                # asyncio.run(self.log_prob_pool.shutdown())

        return {"job_id": job_id}


class ArcticRLRayServer:
    # def __init__(self,
    #             training_gpus: int,
    #             sampling_gpus: int,
    #             log_prob_gpus: int,
    #             log_prob_engine: str,
    #             colocate: bool,
    #             ):
    #     total_gpus = training_gpus + sampling_gpus + log_prob_gpus
    #     if total_gpus == 0:
    #         raise ValueError("At least one of --training-gpus, --sampling-gpus, --log-prob-gpus must be > 0")

    #     logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    #     pr0(f"[ArcticRLRayServer] initializing ray cluster")
    #     init_ray_cluster(auto_attach=False)
    #     pr0(f"[ArcticRLRayServer] ray cluster initialized")

    #     self.training_gpus = training_gpus
    #     self.sampling_gpus = sampling_gpus
    #     self.log_prob_gpus = log_prob_gpus
    #     self.log_prob_engine = log_prob_engine
    #     self.colocate = colocate

    #     # In colocated mode, create a placement group with one bundle per
    #     # physical GPU (like native SkyRL).  All workers are pinned to specific
    #     # bundles so training ranks get unique GPUs while still sharing them
    #     # with inference workers.
    #     self.placement_group = None
    #     if self.colocate:
    #         n_bundles = max(self.training_gpus, self.sampling_gpus, self.log_prob_gpus)  # num_ray_gpus
    #         pg = placement_group(
    #             [{"GPU": 1, "CPU": 4}] * n_bundles,
    #             strategy="STRICT_PACK",
    #         )
    #         ray.get(pg.ready())
    #         logger.info("Created colocate placement group with %d bundles", n_bundles)
    #         self.placement_group = pg
    #         self.n_bundles = n_bundles

    #     if self.colocate:
    #         assert self.placement_group is not None, \
    #             "Placement group must be created when colocate=True"

    #     self.training_workers = []
    #     self.sampling_pool = ReplicaPool()
    #     if self.log_prob_engine == "vllm":
    #         self.log_prob_pool = ReplicaPool()
    #     else:
    #         self.log_prob_pool = None
    #     self.log_prob_workers = []
    #     self.log_prob_tokenizer = None
    #     self.jobs = {}
    #     self.next_job_id = 1
    #     self.weight_sync_ready = False
    #     self.weight_sync_bucket_size = _WEIGHT_SYNC_BUCKET_SIZE

    #     pr0(f"[ArcticRLRayServer] initialized")

    def __init__(self, arctic_rl_ray_server_state: ArcticRLRayServerState):
        self.arctic_rl_ray_server_state = arctic_rl_ray_server_state

        # Alias shared state via ray remote object
        pr0(f"[ArcticRLRayServer] initializing ray cluster {type(arctic_rl_ray_server_state)=}")

        self.jobs = ray.get(arctic_rl_ray_server_state.get_jobs.remote())  # type: ignore
        self.training_workers = ray.get(arctic_rl_ray_server_state.get_training_workers.remote())  # type: ignore
        self.log_prob_workers = ray.get(arctic_rl_ray_server_state.get_log_prob_workers.remote())  # type: ignore
        self.log_prob_tokenizer = ray.get(arctic_rl_ray_server_state.get_log_prob_tokenizer.remote())  # type: ignore
        self.next_job_id = ray.get(arctic_rl_ray_server_state.get_next_job_id.remote())  # type: ignore
        self.weight_sync_ready = ray.get(arctic_rl_ray_server_state.get_weight_sync_ready.remote())  # type: ignore
        self.weight_sync_bucket_size = ray.get(arctic_rl_ray_server_state.get_weight_sync_bucket_size.remote())  # type: ignore
        self.sampling_pool = ray.get(arctic_rl_ray_server_state.get_sampling_pool.remote())  # type: ignore
        self.log_prob_pool = ray.get(arctic_rl_ray_server_state.get_log_prob_pool.remote())  # type: ignore
        self.colocate = ray.get(arctic_rl_ray_server_state.get_colocate.remote())  # type: ignore

    def _verify_job(self, job_id: int, expected_types: Union[str, list[str]]) -> None:
        info = self.jobs.get(job_id)
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if info is None:
            raise ValueError(f"Job {job_id} not found")
        if info["job_type"] not in expected_types:
            raise ValueError(f"Job {job_id} is not a {', '.join(expected_types)} job")

    async def health(self):
        return {"status": "OK"}

    # async def initialize(self, job_config: dict[str, Any]) -> dict[str, Any]:
    #     job_config = JobConfig(**job_config)
    #     pr0(f"[ArcticRLRayServer] initialize: {job_config=}")
    #     job_type = job_config.job_type
    #     job_id = self.next_job_id
    #     self.next_job_id += 1

    #     colocate = self.colocate
    #     pg = self.placement_group

    #     # Fractional GPU fractions within each PG bundle.  Each bundle owns 1
    #     # physical GPU; fractions let multiple actors share that bundle while
    #     # Ray still sets CUDA_VISIBLE_DEVICES so each actor can see the GPU.
    #     #   training (0.4) + sampling (0.6)  = 1.0  — share a bundle
    #     #   training (0.4) + log_prob (0.3)  = 0.7  — share a bundle
    #     _COLOCATE_GPU_FRACTIONS = {"sampling": 0.6, "log_prob": 0.3, "training": 0.4}

    #     def _pg_options(bundle_index: int, fraction_key: str) -> dict:
    #         """PG-pinned scheduling: fractional GPU claim inside a specific bundle."""
    #         return dict(
    #             num_gpus=_COLOCATE_GPU_FRACTIONS[fraction_key],
    #             scheduling_strategy=PlacementGroupSchedulingStrategy(
    #                 placement_group=pg,
    #                 placement_group_bundle_index=bundle_index,
    #             ),
    #         )

    #     n_bundles = getattr(self, "n_bundles", 0)
    #     n_sample = self.sampling_gpus
    #     n_logprob = self.log_prob_gpus

    #     # Bundle layout (deterministic):
    #     #   bundles [0 .. n_sample-1]           → training + sampling
    #     #   bundles [n_sample .. n_sample+n_lp] → training + log_prob
    #     #   remaining bundles                   → training only

    #     if job_type == "training":
    #         gpus = self.training_gpus
    #         if gpus == 0:
    #             raise ValueError("No training GPUs configured")
    #         if self.training_workers:
    #             raise ValueError("Training job already running")

    #         workers = []
    #         config_dict = job_config.model_dump()
    #         for rank in range(gpus):
    #             if colocate and pg is not None:
    #                 opts = _pg_options(bundle_index=rank, fraction_key="training")
    #             else:
    #                 opts = dict(num_gpus=1)
    #             w = DeepSpeedWorker.options(**opts).remote(rank, gpus, 29500)
    #             workers.append(w)
    #         await asyncio.gather(*[w.initialize.remote(config_dict) for w in workers])
    #         self.training_workers = workers

    #     elif job_type == "sampling":
    #         gpus = self.sampling_gpus
    #         if gpus == 0:
    #             raise ValueError("No sampling GPUs configured")
    #         pool: ReplicaPool = self.sampling_pool
    #         if pool._config is not None:
    #             raise ValueError("Sampling job already running")
    #         vllm_cfg = dict(job_config.vllm_config or {})
    #         if colocate:
    #             vllm_cfg["enable_sleep_mode"] = True
    #         model_cfg = _build_model_config(job_config.model_name, vllm_cfg)
    #         tp = model_cfg.tensor_parallel_size
    #         num_replicas = gpus // tp
    #         if colocate and pg is not None:
    #             bundle_indices = list(range(num_replicas))
    #             extra_env = {}
    #             if tp > 1:
    #                 extra_env["VLLM_RAY_PER_WORKER_GPUS"] = str(_COLOCATE_GPU_FRACTIONS["sampling"])
    #                 vllm_cfg["distributed_executor_backend"] = "ray"
    #                 model_cfg = _build_model_config(job_config.model_name, vllm_cfg)
    #             await pool.initialize(
    #                 model_cfg,
    #                 num_replicas=num_replicas,
    #                 ray_num_gpus=_COLOCATE_GPU_FRACTIONS["sampling"],
    #                 placement_group=pg,
    #                 bundle_indices=bundle_indices,
    #                 extra_env=extra_env if extra_env else None,
    #             )
    #         else:
    #             await pool.initialize(model_cfg, num_replicas=num_replicas)

    #     elif job_type == "log_prob":
    #         gpus = self.log_prob_gpus
    #         if gpus == 0:
    #             raise ValueError("No log-prob GPUs configured")

    #         # Log-prob bundles start after sampling bundles to avoid oversubscription.
    #         lp_bundle_offset = n_sample

    #         if job_config.ds_config is not None:
    #             if self.log_prob_workers:
    #                 raise ValueError("Log-prob job already running")
    #             workers = []
    #             config_dict = job_config.model_dump()
    #             for rank in range(gpus):
    #                 if colocate and pg is not None:
    #                     opts = _pg_options(bundle_index=lp_bundle_offset + rank, fraction_key="log_prob")
    #                 else:
    #                     opts = dict(num_gpus=1)
    #                 w = DeepSpeedWorker.options(**opts).remote(rank, gpus, 29501)
    #                 workers.append(w)
    #             await asyncio.gather(*[w.initialize.remote(config_dict) for w in workers])
    #             self.log_prob_workers = workers
    #             self.log_prob_tokenizer = AutoTokenizer.from_pretrained(job_config.model_name)
    #             engine = "deepspeed"
    #         else:
    #             pool: ReplicaPool = self.log_prob_pool
    #             if pool._config is not None:
    #                 raise ValueError("Log-prob job already running")
    #             lp_vllm_cfg = dict(job_config.vllm_config or {})
    #             if colocate:
    #                 lp_vllm_cfg["enable_sleep_mode"] = True
    #             model_cfg = _build_model_config(job_config.model_name, lp_vllm_cfg)
    #             lp_tp = model_cfg.tensor_parallel_size
    #             num_replicas = gpus // lp_tp
    #             if colocate and pg is not None:
    #                 bundle_indices = [lp_bundle_offset + i for i in range(num_replicas)]
    #                 lp_extra_env = {}
    #                 if lp_tp > 1:
    #                     lp_bundles = [lp_bundle_offset + i for i in range(num_replicas * lp_tp)]
    #                     lp_extra_env["VLLM_RAY_PER_WORKER_GPUS"] = str(_COLOCATE_GPU_FRACTIONS["log_prob"])
    #                     lp_extra_env["VLLM_RAY_BUNDLE_INDICES"] = ",".join(str(b) for b in lp_bundles[:lp_tp])
    #                     lp_extra_env.pop("CUDA_VISIBLE_DEVICES", None)
    #                     lp_vllm_cfg["distributed_executor_backend"] = "ray"
    #                     model_cfg = _build_model_config(job_config.model_name, lp_vllm_cfg)
    #                 await pool.initialize(
    #                     model_cfg,
    #                     num_replicas=num_replicas,
    #                     ray_num_gpus=_COLOCATE_GPU_FRACTIONS["log_prob"],
    #                     placement_group=pg,
    #                     bundle_indices=bundle_indices,
    #                     extra_env=lp_extra_env if lp_extra_env else None,
    #                 )
    #             else:
    #                 await pool.initialize(model_cfg, num_replicas=num_replicas)
    #             engine = "vllm"

    #     else:
    #         raise ValueError(f"Unknown job type: {job_type}")

    #     job_info: dict[str, Any] = {
    #         "job_id": job_id,
    #         "job_type": job_type,
    #         "model_name": job_config.model_name,
    #         "status": "RUNNING",
    #     }
    #     if job_type == "log_prob":
    #         job_info["engine"] = engine
    #     self.jobs[job_id] = job_info
    #     return {"job_id": job_id, "job_type": job_type, "running": True}

    async def destroy(self, job_id: int, job_type: str) -> dict[str, Any]:
        return await self.arctic_rl_ray_server_state.destroy.remote(job_id, job_type)

    async def fwd_bwd(self, job_id: int, batch: dict) -> dict[str, Any]:
        tname_e2e = timers.start("xyz fwd_bwd e2e")

        tname = timers.start("xyz fwd_bwd: _verify_job")
        self._verify_job(job_id, "training")
        workers = self.training_workers
        timers.stop_and_print_elapsed(tname)

        # tname = timers.start("xyz fwd_bwd: decompress")
        # import zlib
        # body = zlib.decompress(body)
        # timers.stop_and_print_elapsed(tname)

        tname = timers.start("xyz fwd_bwd: ray_split_batch")
        shards, _ = ray_split_batch(batch, len(workers))
        # The verl driver's ``update_actor`` only consumes ``metrics`` from the
        # fwd_bwd response (see arctic_rl_client.update_actor) -- the per-token
        # ``batch`` (logprobs/entropy) is never read. Keep the worker output as
        # tensors so ``run_pipeline`` skips the per-microbatch detensorize()
        # ``.tolist()``, and omit ``batch`` from the response so it is never
        # passed back through the Ray object store.
        shards[0]["meta"]["worker_return_tensors"] = True
        timers.stop_and_print_elapsed(tname)
        for shard_rank, shard in enumerate(shards):
            _, shard_batch, shard_meta, _ = unpack_batch(shard)
            log_dp_shard_tokens(shard_rank, "ray_split_batch", shard_batch, shard_meta)

        tname = timers.start("xyz fwd_bwd: gather + forward_backward")
        # results = await asyncio.gather(*[
        #     w.forward_backward.remote(s) for w, s in zip(workers, shards)
        # ])

        prof = ProfilerContext(type=PROFILER_TYPE, name="GATHER")
        with prof():
            refs = [w.forward_backward.remote(s) for w, s in zip(workers, shards)]
            results = ray.get(refs)

        timers.stop_and_print_elapsed(tname)
        prof.report()
        pr0(f"[DeepSpeedWorker] fwd_bwd: {len(results)=}")

        tname = timers.start("xyz fwd_bwd: epilogue")
        losses = [r["avg_loss"] for r in results]
        avg_loss = sum(losses)  # / len(losses)
        print(f"new loss {avg_loss=}")
        print(f"new loss {len(losses)=}")
        print(f"new loss {sum(losses) / len(losses)=}")

        # Each rank-shard's ``metrics`` dict is already collapsed across its
        # microbatches by ``deepspeed_worker._forward_maybe_backward`` (paired
        # ``{name}.sum`` / ``{name}.tokens`` scalars summed across microbatches).
        # ``combine_metric_shards`` sums those across DP ranks and divides
        # ``Σ sum / Σ tokens`` per metric, so the controller receives a single
        # global token-mean scalar per metric per mini-batch (one ``fwd_bwd``
        # call). That matches the baseline VeRL semantics of one scalar per
        # PPO mini-batch update and replaces the prior per-(rank × microbatch)
        # list shape that ``merge_dict_shards`` produced. ``batch`` is
        # intentionally omitted -- the driver does not consume it (see note above).
        merged = dict(
            job_id=job_id,
            metrics=combine_metric_shards([r["metrics"] for r in results]),
            avg_loss=avg_loss,
        )
        timers.stop_and_print_elapsed(tname)

        timers.stop_and_print_elapsed(tname_e2e)

        return merged

    # losses = [r["avg_loss"] for r in results]
    # avg_loss = sum(losses) / len(losses)
    # post_process_outputs = merge_metrics(
    #     [r.get("post_process_outputs", {}) for r in results]
    # )
    # return {"job_id": job_id, "avg_loss": avg_loss, "post_process_outputs": post_process_outputs}

    async def fwd_no_grad(self, job_id: int, batch: dict) -> dict[str, Any]:
        info = self.jobs[job_id]
        self._verify_job(job_id, ["training", "log_prob"])
        job_type = info["job_type"]
        if job_type == "log_prob":
            workers = self.log_prob_workers
        else:
            workers = self.training_workers
        if not workers:
            raise ValueError(f"Job {job_id} ({job_type}) has no DeepSpeed workers")
        batch["meta"]["worker_return_tensors"] = True

        # import zlib
        # body = zlib.decompress(body)

        # shards = ray_split_batch(batch, len(workers))
        # results = await asyncio.gather(*[
        #     w.forward_no_grad.remote(s) for w, s in zip(workers, shards)
        # ])

        shards, reorder_indices = ray_split_batch(batch, len(workers))
        refs = [w.forward_no_grad.remote(s) for w, s in zip(workers, shards)]
        results = ray.get(refs)

        pr0(f"[ArcticRLRayServer] fwd_no_grad: {len(results)=}")

        batch = merge_dict_shards([r["batch"] for r in results])
        if reorder_indices is not None:
            batch = restore_batch_order(batch, reorder_indices)

        merged = dict(
            job_id=job_id,
            batch=batch,
            metrics=merge_dict_shards([r["metrics"] for r in results]),
        )

        return merged

    async def step(self, job_id: int) -> dict[str, Any]:
        self._verify_job(job_id, "training")
        # results = await asyncio.gather(*[w.step.remote() for w in self.training_workers])
        refs = [w.step.remote() for w in self.training_workers]
        results = ray.get(refs)
        merged = dict(
            job_id=job_id,
            metrics=merge_dict_shards([r["metrics"] for r in results]),
            batch=merge_dict_shards([r["batch"] for r in results]),
        )
        return merged

    async def empty_training_cache(self, job_id: int):
        """Release ZeRO partition cache and PyTorch cached memory on all workers."""
        self._verify_job(job_id, "training")
        workers = self.training_workers
        results = await self.arctic_rl_ray_server_state._empty_training_cache.remote(workers)
        logger.info("Empty training cache: %s", results)
        return {"job_id": job_id, "workers": results}

    async def save_checkpoint(self, job_id: int):
        self._verify_job(job_id, "training")
        info = self.jobs[job_id]
        checkpoint_path = info.get("checkpoint_path", None)
        assert checkpoint_path is not None, f"checkpoint_path is required for training jobs {job_id}"
        os.makedirs(checkpoint_path, exist_ok=True)
        # await asyncio.gather(
        #     *[w.save_checkpoint.remote(path) for w in self.training_workers],
        # )
        refs = [w.save_checkpoint.remote(checkpoint_path) for w in self.training_workers]
        _ = ray.get(refs)
        return {"job_id": job_id, "path": checkpoint_path}

    async def sleep_inference(self, job_id: int, level: int):
        """Put all inference engines to sleep, freeing GPU memory."""
        self._verify_job(job_id, "sampling")
        return await self.arctic_rl_ray_server_state.sleep_inference.remote(job_id, level)

    async def wake_inference(self, job_id: int, tags: list[str] | None = None):
        """Wake all inference engines, restoring GPU memory."""
        self._verify_job(job_id, "sampling")
        results = await self.arctic_rl_ray_server_state.wake_inference.remote(tags)
        return {"job_id": job_id, **results}

    async def reset_prefix_cache(self, job_id: int):
        """Reset the prefix cache on the sampling inference engines."""
        self._verify_job(job_id, "sampling")
        return await self.arctic_rl_ray_server_state.reset_prefix_cache.remote(job_id)

    async def sleep_training(self, job_id: int, mode: str = "all"):
        """Offload training state to CPU (sleep training workers).

        mode='all':       Offload everything (for training → inference transition)
        mode='non_lp':    Keep bf16 params on GPU, offload rest (before CUDA IPC sync)
        mode='lp_params': Offload bf16 params only (after CUDA IPC sync)
        """
        self._verify_job(job_id, "training")
        workers = self.training_workers
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

    async def wake_training(self, job_id: int):
        """Reload all training state to GPU (wake training workers)."""
        self._verify_job(job_id, "training")
        workers = self.training_workers
        loop = asyncio.get_running_loop()
        refs = [w.backload_to_gpu.remote() for w in workers]
        results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
        logger.info("Wake training: %s", results)
        return {"job_id": job_id, "workers": results}

    async def sleep_log_prob(self, job_id: int):
        """Offload the reference (log-prob) DeepSpeed engine to CPU.

        No-op when the log-prob engine is vLLM (those replicas are handled by
        sleep_inference) or when no separate log-prob job exists.
        """
        self._verify_job(job_id, "log_prob")
        workers = self.log_prob_workers
        if not workers:
            return {"job_id": job_id, "workers": []}
        loop = asyncio.get_running_loop()
        refs = [w.offload_to_cpu.remote() for w in workers]
        results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
        logger.info("Offload log_prob: %s", results)
        return {"job_id": job_id, "workers": results}

    async def wake_log_prob(self, job_id: int):
        """Reload the reference (log-prob) DeepSpeed engine to GPU.

        No-op when the log-prob engine is vLLM or no separate log-prob job
        exists (see sleep_log_prob).
        """
        self._verify_job(job_id, "log_prob")
        workers = self.log_prob_workers
        if not workers:
            return {"job_id": job_id, "workers": []}
        loop = asyncio.get_running_loop()
        refs = [w.backload_to_gpu.remote() for w in workers]
        results = await asyncio.gather(*[loop.run_in_executor(None, ray.get, ref) for ref in refs])
        logger.info("Wake log_prob: %s", results)
        return {"job_id": job_id, "workers": results}

    async def generate(self, job_id: int, request: dict[str, Any]) -> dict[str, Any]:
        request = GenerateRequest(**request)
        self._verify_job(job_id, "sampling")
        pool: ReplicaPool = self.sampling_pool
        # Strict routing keys group affinity on the prompt hash and balances
        # rollout groups across engines via round-robin.
        results = await pool.generate(
            request.prompts,
            request.sampling_params,
            strict=True,
        )

        return {"job_id": job_id, "results": results}

    async def sync_weights(self, request: dict[str, Any]) -> dict[str, Any]:
        """Sync training model weights to the sampling engine.

        Uses NCCL for non-colocated mode (separate GPUs).  In colocated mode:
        - cuda_ipc=True: CUDA IPC (zero-copy, requires training weights on GPU)
        - cuda_ipc=False: CPU file path (slower, works when offloaded)
        """
        request = SyncWeightsRequest(**request)
        self._verify_job(request.training_job_id, "training")
        self._verify_job(request.sampling_job_id, "sampling")

        workers = self.training_workers
        pool: ReplicaPool = self.sampling_pool
        colocate = request.colocate or self.colocate

        if colocate:
            lp_pool = self.log_prob_pool
            if request.cuda_ipc:
                if request.low_memory:
                    # Slower, memory-efficient path: stream one gathered param
                    # at a time so peak extra GPU memory is one full param per
                    # GPU instead of the whole model (avoids OOM on big models).
                    results = await self._sync_weights_cuda_ipc_low_mem(workers, pool, lp_pool)
                else:
                    results = await self._sync_weights_cuda_ipc(workers, pool, lp_pool)
            else:
                training_job_info = self.jobs[request.training_job_id]
                sync_path = training_job_info.get("sync_path", None)
                assert sync_path is not None, f"sync_path is required for training job {request.training_job_id}"
                results = await self._sync_weights_ipc(sync_path, workers, pool, lp_pool)
        else:
            results = await self._sync_weights_nccl(workers, pool)

        await self.arctic_rl_ray_server_state.reset_prefix_cache.remote(request.sampling_job_id)

        return {"job_id": request.training_job_id, **results}

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

        # if not self.weight_sync_ready:
        #     max_param = await workers[0].max_param_bytes.remote()
        #     bucket_size = max(max_param, _WEIGHT_SYNC_BUCKET_SIZE)
        #     self.weight_sync_bucket_size = bucket_size

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
        #     self.weight_sync_ready = True
        #     logger.info(
        #         "Weight sync initialized: %d training GPUs -> %d replicas (tp=%d), %d NCCL group(s); sender IPs=%s",
        #         len(workers),
        #         pool.num_replicas,
        #         pool.tp_size,
        #         len(schedule.groups),
        #         group_master_addrs,
        #     )

        # bucket_size = self.weight_sync_bucket_size

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

    async def _sync_weights_cuda_ipc(self, workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None) -> dict:
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
        # await pool.sleep(level=2, offload_weights=True)
        # if lp_pool is not None:
        #     await lp_pool.sleep(level=2, offload_weights=True)

        # All workers must participate for ZeRO-3 collective gather.
        # gather_cuda_ipc_handles is safe for ZeRO-2 too (no ds_id → no gather).
        gather_refs = [w.gather_cuda_ipc_handles.remote() for w in workers]
        # results = await asyncio.gather(*[
        #     loop.run_in_executor(None, ray.get, ref) for ref in gather_refs
        # ])
        results = ray.get(gather_refs)
        ipc_payload = merge_cuda_ipc_payloads(results)
        num_params = ipc_payload.get("num_params", 0)

        # Staged wake: weights only → IPC load → KV cache (reduces peak GPU memory).
        await self.arctic_rl_ray_server_state.wake_inference.remote(tags=["weights"])

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

        await self.arctic_rl_ray_server_state.wake_inference.remote(tags=["kv_cache"])

        elapsed = time.monotonic() - t0
        logger.info(
            "Weight sync (CUDA IPC) complete in %.3fs (%d replica(s), %d params)", elapsed, total_replicas, num_params
        )
        return {"status": "ok"}

    async def _sync_weights_cuda_ipc_low_mem(
        self, workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None
    ) -> dict:
        """Memory-efficient (slower) colocated weight sync via CUDA IPC.

        Streams one parameter at a time: all training ranks collectively
        gather a single ZeRO-3 param onto their own GPU, the colocated
        inference replicas copy it in, then the source IPC ref is released
        before moving on. Peak extra GPU memory is one full parameter per GPU
        (instead of the whole model as in ``_sync_weights_cuda_ipc``), at the
        cost of many more round-trips.

        Selected via ``arctic_rl.low_memory_weight_sync=True``.
        """
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()

        # Enumerate parameter names once (only names cross the Ray boundary;
        # each worker resolves its own live param by name inside
        # get_cuda_ipc_handle so the ZeRO-3 gather stays correct).
        param_names = ray.get(workers[0].get_parameter_names.remote())
        num_params = len(param_names)

        # Flatten the colocated inference replicas (sampling + optional log-prob).
        replicas = []
        for p in [pool, lp_pool]:
            if p is None or p._config is None:
                continue
            for rid in range(p.num_replicas):
                replicas.append(p._workers[rid])

        # Staged wake: weights only → per-param IPC load → KV cache.
        # restore_weights=True: sleep_inference manually offloads the weights
        # (offload_weights=colocate), replacing every vLLM param.data with a [1]
        # CPU stub. The manual backload must run on the weights wake to restore
        # full-shape params before the IPC stream copies into them; otherwise the
        # load lands on [1] stubs and dies with "output with shape [1] doesn't
        # match the broadcast shape". (Skipping the backload is only valid if
        # cumem owns the weights, i.e. offload_weights=False.)
        await self.arctic_rl_ray_server_state.wake_inference.remote(tags=["weights"], restore_weights=True)

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

            # Release this param's source ref on every training rank before the
            # next gather, bounding peak memory to one full param per GPU.
            await asyncio.gather(
                *[loop.run_in_executor(None, ray.get, w.release_ipc_handles.remote()) for w in workers]
            )

        await self.arctic_rl_ray_server_state.wake_inference.remote(tags=["kv_cache"])

        elapsed = time.monotonic() - t0
        logger.info(
            "Weight sync (CUDA IPC, low-mem) complete in %.3fs (%d replica(s), %d params)",
            elapsed,
            len(replicas),
            num_params,
        )
        return {"status": "ok"}

    async def _sync_weights_ipc(
        self, sync_path: str, workers, pool: ReplicaPool, lp_pool: ReplicaPool | None = None
    ) -> dict:
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

        await self.arctic_rl_ray_server_state.wake_inference.remote(tags=["weights"])

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

        await self.arctic_rl_ray_server_state.wake_inference.remote(tags=["kv_cache"])

        elapsed = time.monotonic() - t0
        logger.info(
            "Weight sync (CPU→GPU) complete in %.3fs (%d replica(s), %d params)", elapsed, total_replicas, num_params
        )
        return {"status": "ok"}

    async def _sync_weights_nccl(self, workers, pool: ReplicaPool) -> dict:
        """Non-colocated weight sync via NCCL (original path)."""
        await self.arctic_rl_ray_server_state.wake_inference.remote()
        schedule = TransferSchedule.build(
            training_sharding="dp",
            training_gpus=len(workers),
            inference_replicas=pool.num_replicas,
            inference_tp=pool.tp_size,
        )
        sender_ranks = [g.sender_train_rank for g in schedule.groups]
        sender_ips = await asyncio.gather(*[workers[r].get_ip.remote() for r in sender_ranks])
        group_master_addrs = {g.group_id: ip for g, ip in zip(schedule.groups, sender_ips)}

        if not self.weight_sync_ready:
            max_param = await workers[0].max_param_bytes.remote()
            bucket_size = max(max_param, _WEIGHT_SYNC_BUCKET_SIZE)
            self.weight_sync_bucket_size = bucket_size

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
            self.weight_sync_ready = True
            logger.info(
                "Weight sync initialized: %d training GPUs -> %d replicas (tp=%d), %d NCCL group(s); sender IPs=%s",
                len(workers),
                pool.num_replicas,
                pool.tp_size,
                len(schedule.groups),
                group_master_addrs,
            )

        bucket_size = self.weight_sync_bucket_size

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

        # One sender rank per NCCL group (as assigned by TransferSchedule).
        # Broadcasting send_weights from every DP rank breaks the intended
        # topology and can hang / duplicate / corrupt transfers for
        # non-colocated runs. Mirrors the HTTP server's sender-rank-only sends.
        send_tasks = [workers[g.sender_train_rank].send_weights.remote() for g in schedule.groups]
        receive_task = pool.sync_weights(
            groups=groups,
            bucket_size=bucket_size,
        )

        t0 = time.monotonic()
        await asyncio.gather(receive_task, *send_tasks)
        logger.info("Weight sync complete in %.3fs (%d group(s))", time.monotonic() - t0, len(schedule.groups))
        return {"status": "ok"}

    async def log_probs(self, job_id: int, request: dict[str, Any]) -> dict[str, Any]:
        self._verify_job(job_id, "log_prob")
        request = LogProbsRequest(**request)
        info = self.jobs[job_id]

        if request.completions is not None:
            full_texts = [p + c for p, c in zip(request.prompts, request.completions)]
        else:
            full_texts = request.prompts

        if info.get("engine") == "deepspeed":
            tokenizer = self.log_prob_tokenizer
            encoded = tokenizer(full_texts, return_tensors="pt", padding=True)
            batch_buf = io.BytesIO()
            torch.save(dict(encoded), batch_buf)
            batch_data = batch_buf.getvalue()

            workers = self.log_prob_workers
            shards, _ = ray_split_batch(batch_data, len(workers))
            raw = await asyncio.gather(*[w.compute_log_probs.remote(s) for w, s in zip(workers, shards)])
            shard_tensors = [torch.load(io.BytesIO(r), map_location="cpu") for r in raw]
            results = torch.cat(shard_tensors, dim=0)
        else:
            pool: ReplicaPool = self.log_prob_pool
            results = await pool.generate(
                full_texts,
                {"max_tokens": 1, "temperature": 0, "prompt_logprobs": request.top_k},
            )

        return {"job_id": job_id, "results": results}

    async def status(self):
        return {
            "training_gpus": self.training_gpus,
            "sampling_gpus": self.sampling_gpus,
            "log_prob_gpus": self.log_prob_gpus,
            "jobs": {jid: info for jid, info in self.jobs.items()},
        }

    async def get_job_status(self, job_id: int):
        info = self.jobs.get(job_id)
        if info is None:
            raise ValueError(f"Job {job_id} not found")
        return info


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
