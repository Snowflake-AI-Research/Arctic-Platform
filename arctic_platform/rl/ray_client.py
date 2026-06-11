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

"""ArcticRLClient -- unified HTTP client for RL training.

Works identically against a remote dss-platform deployment or a local
``server.py`` instance -- the only differences are ``base_url`` and whether the
client launches the server.

All jobs (training, sampling, log-prob) are initialized automatically at
construction time.
"""

from __future__ import annotations

import atexit
import io
import logging
import signal
import subprocess
import sys
import time
import ray
import aiohttp
import requests
import torch
from typing import Any

from arctic_platform.rl.config import ArcticRLClientConfig
from arctic_platform.rl.utils.batch import tensorize
from arctic_platform.rl.ray_server import create_arctic_rl_ray_server_state, ArcticRLRayServerState, ArcticRLRayServer
from arctic_platform.rl.utils.debug import see_memory_usage, pr, pr0

ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple
    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy
    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)

logger = logging.getLogger(__name__)

class ArcticRLRayClient:
    """HTTP client for RL training against dss-platform or a local server.

    Jobs are created automatically during ``__init__`` for each engine type.

    Parameters
    ----------
    config : ArcticRLClientConfig
        Connection parameters, model name, and GPU allocation.
    """

    def __init__(self, config: ArcticRLClientConfig, rl_server_state: ArcticRLRayServerState) -> None:
        self.config = config

        assert config.backend == "local", "ArcticRLRayClient only supports local backend"

        pr0(f"[ArcticRLRayClient] entry: config: {config=} {rl_server_state=}")

        if config.training_job_id is not None:
            # Reconnect mode: attach to pre-existing jobs without calling /initialize.
            # Populated by reconnect_config(); used when passing a client across Ray
            # process boundaries where the live object cannot be serialized.
            self._training_job_id = config.training_job_id
            self._sampling_job_id = config.sampling_job_id
            self._log_prob_job_id = config.log_prob_job_id

            self._arctic_rl_ray_server_state = rl_server_state
        else:
            self._training_job_id: int | None = None
            self._sampling_job_id: int | None = None
            self._log_prob_job_id: int | None = None
            self._arctic_rl_ray_server_state = create_arctic_rl_ray_server_state(
                training_gpus=config.training_gpus,
                sampling_gpus=config.sampling_gpus,
                log_prob_gpus=config.log_prob_gpus,
                log_prob_engine=config.log_prob_engine,
                colocate=config.colocate,
            )
            self._initialize_jobs(config)

        self._arctic_rl_ray_server = ArcticRLRayServer(self._arctic_rl_ray_server_state)
        pr0(f"[ArcticRLRayClient] exit: created arctic_rl_ray_server: {self._arctic_rl_ray_server=} {self._arctic_rl_ray_server_state=}")


    def reconnect_config(self) -> ArcticRLClientConfig:
        """Return a serializable config that reconnects to the same pre-existing jobs.

        The returned config can be passed to ``ArcticRLClient()`` in another
        process (e.g. a Ray actor) to connect to the same jobs without calling
        ``/initialize`` again.  ``backend`` is always ``"dss-platform"`` since
        the server is already running.

        Example::

            # driver
            client = ArcticRLClient(config)
            rc = client.reconnect_config()   # plain Pydantic model — Ray-serializable

            # Ray actor
            actor_client = ArcticRLClient(rc)   # reconnects, no /initialize
        """
        return ArcticRLClientConfig(
            backend="local",
            model_name=self.config.model_name,
            training_job_id=self._training_job_id,
            sampling_job_id=self._sampling_job_id,
            log_prob_job_id=self._log_prob_job_id,
            comm_protocol="ray",
        )


    def get_server_state(self) -> ArcticRLRayServerState:
        return self._arctic_rl_ray_server_state


    # ------------------------------------------------------------------
    # Job initialization
    # ------------------------------------------------------------------

    def _initialize_jobs(self, config: ArcticRLClientConfig) -> None:
        data = self._post_initialize("training")
        self._training_job_id = data["job_id"]

        data = self._post_initialize("sampling")
        self._sampling_job_id = data["job_id"]

        if self.config.log_prob_gpus > 0:
            data = self._post_initialize("log_prob")
            self._log_prob_job_id = data["job_id"]


    def _post_initialize(self, job_type: str) -> dict[str, Any]:
        job_config: dict[str, Any] = {
            "model_name": self.config.model_name,
            "job_type": job_type,
            "use_arctic_inference": self.config.use_arctic_inference,
            "full_determinism": self.config.full_determinism,
            "seed": self.config.seed,
        }
        use_deepspeed = job_type == "training" or (
            job_type == "log_prob" and self.config.log_prob_engine == "deepspeed"
        )

        if use_deepspeed:
            if self.config.ds_config:
                job_config["ds_config"] = self.config.ds_config
            if job_type == "training" and self.config.training_config is not None:
                job_config["training_config"] = self.config.training_config
                job_config["ds_config"] = self.config.ds_config
            elif job_type == "log_prob":
                # Treat log_prob like training: send the base ds_config plus a
                # dedicated log_prob_config. The worker builds a forward-only
                # (no-optimizer) engine from log_prob_config.
                job_config["ds_config"] = self.config.ds_config
                if self.config.log_prob_ds_config is not None:
                    job_config["log_prob_config"] = self.config.log_prob_ds_config
            else:
                job_config["ds_config"] = self.config.ds_config
            # this config is usually the same for any DeepspeedWorker job type - e.g. activating zorro - but it could be made job-type-specific as well, similar to do `ds_config`
            job_config["ds_worker_config"] = self.config.ds_worker_config
            if job_type == "training":
                job_config["checkpoint_path"] = self.config.checkpoint_path
        else:
            if self.config.vllm_config:
                job_config["vllm_config"] = self.config.vllm_config

        resp = ray.get(self._arctic_rl_ray_server_state.initialize.remote(job_config))
        return resp



    async def _destroy_job(self, job_id: int, job_type: str) -> None:
        response = await self._arctic_rl_ray_server.destroy(job_id, job_type)
        print(f"[ArcticRLRayClient] destroy_job OUTPUT: {response.keys()=}")

    # ------------------------------------------------------------------
    # Job ID properties
    # ------------------------------------------------------------------

    @property
    def training_job_id(self) -> int:
        if self._training_job_id is None:
            raise ValueError("No training job initialized.")
        return self._training_job_id

    @property
    def sampling_job_id(self) -> int:
        if self._sampling_job_id is None:
            raise ValueError("No sampling job initialized.")
        return self._sampling_job_id

    @property
    def log_prob_job_id(self) -> int | None:
        return self._log_prob_job_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def fwd_bwd(
        self,
        batch: dict,
        processing: dict | None = None,
    ) -> dict[str, Any]:
        """Forward-backward pass on the training engine.

        Parameters
        ----------
        batch:
            Dict with ``args`` and ``kwargs`` (model inputs). When using the
            pluggable loss pattern, also include ``context`` (RL tensors) here
            or let ``processing`` drive the dispatch.

            **Log-prob convention (``_shifted`` suffix contract):**
            All log-prob tensors in ``context`` must follow the roll(-1)
            convention — ``tensor[i]`` is the log-prob of *the next token*
            (``input_ids[i+1]``), matching how the server computes current
            log-probs (``labels = torch.roll(input_ids, shifts=-1)``).
            The ``_shifted`` suffix in context key names encodes this:

            - ``old_log_probs_shifted`` — behavioral policy log-probs after roll
            - ``prox_logp_shifted``     — proximal log-probs after roll
            - ``ref_log_probs_shifted`` — reference policy log-probs after roll

            Framework adapters are responsible for applying the roll before
            setting these keys.  Adapters that get logprobs from
            ``fwd_no_grad`` (VERL, SkyRL) receive them pre-shifted from the
            server.  Adapters using vLLM rollout logprobs directly (AReaL)
            must apply ``torch.roll(logprobs, shifts=-1)`` client-side first.

        processing:
            Optional processing descriptor::

                {
                    "loss_fn": "arctic_platform.rl.processors.grpo_loss",
                    "config": {"eps_clip": 0.2, ...},   # per-step, can change
                    "post": [],                          # post-forward processors
                }

            When provided the server uses ``run_pipeline`` with the named loss
            function instead of the hardcoded loss_config baked in at job init.
            Pass ``None`` (default) to use the server's legacy loss_config path.
        """
        pr0(f"[ArcticRLRayClient] fwd_bwd INPUT: {batch.keys()=} {processing=}")
        tname_e2e = timers.start("xyz arctic_rl.ray_client fwd_bwd")
        tname = timers.start("xyz arctic_rl.ray_client incoming processing 1")
        if processing is not None:
            batch = {**batch, "processing": processing}
        timers.stop_and_print_elapsed(tname)
        response = await self._arctic_rl_ray_server.fwd_bwd(self.training_job_id, batch)
        # tname = timers.start("xyz arctic_rl.ray_client outgoing processing 2")
        # response["batch"] = tensorize(response["batch"])
        # timers.stop_and_print_elapsed(tname)
        pr0(f"[ArcticRLRayClient] fwd_bwd OUTPUT: {response.keys()=}")
        timers.stop_and_print_elapsed(tname_e2e)
        return response


    async def fwd_no_grad(self, batch: dict, reference_model: bool) -> dict[str, Any]:
        """Forward-only pass (no gradient) on the training engine.
        """
        job_id = self.log_prob_job_id if reference_model else self.training_job_id
        pr0(f"[ArcticRLRayClient] fwd_no_grad INPUT: {batch.keys()=} {reference_model=} {job_id=}")
        response = await self._arctic_rl_ray_server.fwd_no_grad(job_id, batch)
        pr0(f"[ArcticRLRayClient] fwd_no_grad OUTPUT: {response.keys()=}")
        # response["batch"] = tensorize(response["batch"])
        return response


    async def step(self) -> dict[str, Any]:
        """Optimizer step on the training engine."""
        job_id = self.training_job_id
        pr0(f"[ArcticRLRayClient] step INPUT: {job_id=}")
        response = await self._arctic_rl_ray_server.step(job_id)
        pr0(f"[ArcticRLRayClient] step OUTPUT: {response.keys()=}")
        return response

    async def save_checkpoint(self) -> dict[str, Any]:
        """Save training checkpoint."""
        response = await self._arctic_rl_ray_server.save_checkpoint(self.training_job_id)
        pr0(f"[ArcticRLClient] save_checkpoint OUTPUT: {response.keys()=}")
        return response

    # TODO: implement this
    async def save_weights(self, path: str) -> dict[str, Any]:
        """Tell the sampling engine to reload weights from a checkpoint path.

        Note: disk-based weight sync is not yet fully implemented on the server
        side — this call will warn if the endpoint returns an error.
        """
        raise NotImplementedError("save_weights is not implemented")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    async def generate(
        self,
        prompts: list[str],
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        request = dict(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        resp = await self._arctic_rl_ray_server.generate(self.sampling_job_id, request)

        # import torch
        # path = "/code/users/stas/github/sf/arctic-verl/generate.pickle"
        # torch.save(resp, path)
        # exit()
        # #resp = torch.load(path, weights_only=False)

        return resp["results"]


    # ------------------------------------------------------------------
    # Colocated lifecycle
    # ------------------------------------------------------------------
    # TODO: implement this
    async def sleep_inference(self, level: int = 1) -> dict[str, Any]:
        """Put the sampling inference engine to sleep (free GPU memory)."""
        job_id = self.sampling_job_id
        print(f"ArcticRLRayClient: sleep_inference INPUT: {job_id=} {level=}")
        response = await self._arctic_rl_ray_server.sleep_inference(job_id, level)
        print(f"ArcticRLRayClient: sleep_inference OUTPUT: {response.keys()=}")
        return response

    async def wake_inference(self, tags: list[str] | None = None) -> dict[str, Any]:
        """Wake the sampling inference engine."""
        job_id = self.sampling_job_id
        print(f"ArcticRLRayClient: wake_inference INPUT: {job_id=} {tags=}")
        response = await self._arctic_rl_ray_server.wake_inference(job_id, tags)
        print(f"ArcticRLRayClient: wake_inference OUTPUT: {response.keys()=}")
        return response


    async def reset_prefix_cache(self) -> dict[str, Any]:
        """Reset the prefix cache on the sampling inference engine."""
        job_id = self.sampling_job_id
        print(f"ArcticRLRayClient: reset_prefix_cache INPUT: {job_id=}")
        response = await self._arctic_rl_ray_server.reset_prefix_cache(job_id)
        print(f"ArcticRLRayClient: reset_prefix_cache OUTPUT: {response.keys()=}")
        return response


    async def sleep_training(self, mode: str = "all") -> dict[str, Any]:
        """Offload training state to CPU (sleep training workers).

        mode='all':       Everything (training → inference transition)
        mode='non_lp':    Keep bf16 params, offload rest (before CUDA IPC)
        mode='lp_params': Offload bf16 params only (after CUDA IPC)
        """
        job_id = self.training_job_id
        print(f"ArcticRLRayClient: sleep_training INPUT: {job_id=} {mode=}")
        response = await self._arctic_rl_ray_server.sleep_training(job_id, mode)
        print(f"ArcticRLRayClient: sleep_training OUTPUT: {response.keys()}")
        return response


    async def wake_training(self) -> dict[str, Any]:
        """Reload all training state to GPU (wake training workers)."""
        job_id = self.training_job_id
        print(f"ArcticRLRayClient: wake_training INPUT: {job_id=}")
        response = await self._arctic_rl_ray_server.wake_training(job_id)
        print(f"ArcticRLRayClient: wake_training OUTPUT: {response.keys()=}")
        return response


    async def sleep_log_prob(self) -> dict[str, Any]:
        """Offload the reference (log-prob) DeepSpeed engine to CPU."""
        job_id = self.log_prob_job_id
        if job_id is None:
            return {"status": "no_log_prob_job"}
        print(f"ArcticRLRayClient: sleep_log_prob INPUT: {job_id=}")
        response = await self._arctic_rl_ray_server.sleep_log_prob(job_id)
        print(f"ArcticRLRayClient: sleep_log_prob OUTPUT: {response.keys()=}")
        return response


    async def wake_log_prob(self) -> dict[str, Any]:
        """Reload the reference (log-prob) DeepSpeed engine to GPU."""
        job_id = self.log_prob_job_id
        if job_id is None:
            return {"status": "no_log_prob_job"}
        print(f"ArcticRLRayClient: wake_log_prob INPUT: {job_id=}")
        response = await self._arctic_rl_ray_server.wake_log_prob(job_id)
        print(f"ArcticRLRayClient: wake_log_prob OUTPUT: {response.keys()=}")
        return response



    async def empty_training_cache(self) -> dict[str, Any]:
        """Release PyTorch cached GPU memory on all training workers."""
        job_id = self.training_job_id
        print(f"ArcticRLRayClient: empty_training_cache INPUT: {job_id=}")
        response = await self._arctic_rl_ray_server.empty_training_cache(job_id)
        print(f"ArcticRLRayClient: empty_training_cache OUTPUT: {response.keys()=}")
        return response

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    async def sync_weights(self, cuda_ipc: bool = False,
                           low_memory: bool = False) -> dict[str, Any]:
        """Sync training model weights to the sampling engine.

        In non-colocated mode, uses NCCL.  In colocated mode:
        - cuda_ipc=True: zero-copy CUDA IPC (training weights must be on GPU)
        - cuda_ipc=False: CPU file path (works when training is offloaded)

        low_memory only applies to the cuda_ipc path: stream one gathered param
        at a time so peak extra GPU memory is one full param per GPU instead of
        the whole model (avoids OOM on big models, at the cost of round-trips).
        """
        request = dict(
            training_job_id=self.training_job_id,
            sampling_job_id=self.sampling_job_id,
            colocate=self.config.colocate,
            cuda_ipc=cuda_ipc,
            low_memory=low_memory,
        )
        response = await self._arctic_rl_ray_server.sync_weights(request)
        pr0(f"[ArcticRLClient] sync_weights OUTPUT: {response.keys()=}")
        return response

    # ------------------------------------------------------------------
    # Log probabilities
    # ------------------------------------------------------------------

    async def log_probs(
        self,
        prompts: list[str],
        completions: list[str] | None = None,
        top_k: int = 1,
    ) -> dict[str, Any]:
        """Compute per-token log probabilities via the log-prob engine."""
        job_id = self.log_prob_job_id
        print(f"[ArcticRLClient] log_probs INPUT: {job_id=} {prompts=} {completions=} {top_k=}")
        request = dict(
            prompts=prompts,
            completions=completions,
            top_k=top_k,
        )
        response = await self._arctic_rl_ray_server.log_probs(job_id, request)
        print(f"[ArcticRLClient] log_probs OUTPUT: {response.keys()=}")
        return response

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Destroy all jobs and stop the local server if applicable."""
        for job_type, job_id in [
            ("training", self._training_job_id),
            ("sampling", self._sampling_job_id),
            ("log_prob", self._log_prob_job_id),
        ]:
            if job_id is not None:
                await self._destroy_job(job_id, job_type)

        self._training_job_id = None
        self._sampling_job_id = None
        self._log_prob_job_id = None
