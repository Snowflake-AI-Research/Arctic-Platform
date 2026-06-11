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
from typing import Any

import aiohttp
import requests
import torch

from arctic_platform.rl.config import ArcticRLClientConfig
from arctic_platform.rl.http_server import ArcticRLHTTPServerState
from arctic_platform.rl.utils.debug import pr0

ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)

logger = logging.getLogger(__name__)


class ArcticRLHTTPClient:
    """HTTP client for RL training against dss-platform or a local server.

    Jobs are created automatically during ``__init__`` for each engine type.

    Parameters
    ----------
    config : ArcticRLClientConfig
        Connection parameters, model name, and GPU allocation.
    """

    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self._base_url = f"http://{config.host}:{config.port}"
        self._session = requests.Session()
        self._server_process: subprocess.Popen | None = None

        if config.training_job_id is not None:
            # Reconnect mode: attach to pre-existing jobs without calling /initialize.
            # Populated by reconnect_config(); used when passing a client across Ray
            # process boundaries where the live object cannot be serialized.
            self._training_job_id = config.training_job_id
            self._sampling_job_id = config.sampling_job_id
            self._log_prob_job_id = config.log_prob_job_id
        else:
            self._training_job_id: int | None = None
            self._sampling_job_id: int | None = None
            self._log_prob_job_id: int | None = None
            if config.backend == "local":
                self._launch_local_server()
            self._initialize_jobs()

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
            host=self.config.host,
            port=self.config.port,
            backend="dss-platform",
            model_name=self.config.model_name,
            training_job_id=self._training_job_id,
            sampling_job_id=self._sampling_job_id,
            log_prob_job_id=self._log_prob_job_id,
            comm_protocol="http",
        )

    def get_server_state(self) -> ArcticRLHTTPServerState:
        return ArcticRLHTTPServerState()

    # ------------------------------------------------------------------
    # Internal: server lifecycle
    # ------------------------------------------------------------------

    def _launch_local_server(self) -> None:
        cfg = self.config
        cmd = [
            sys.executable,
            "-m",
            "arctic_platform.rl.http_server",
            "--host",
            "0.0.0.0",  # bind all interfaces; client connects via cfg.host
            "--port",
            str(cfg.port),
            "--training-gpus",
            str(cfg.training_gpus),
            "--sampling-gpus",
            str(cfg.sampling_gpus),
            "--log-prob-gpus",
            str(cfg.log_prob_gpus),
            "--log-prob-engine",
            cfg.log_prob_engine,
        ]
        if cfg.colocate:
            cmd.append("--colocate")
        if not cfg.ray_auto_attach:
            cmd.append("--no-ray-auto-attach")
        logger.info("Launching local server: %s", " ".join(cmd))
        stdio = None if cfg.server_logs else subprocess.DEVNULL
        self._server_process = subprocess.Popen(cmd, stdout=stdio, stderr=stdio)
        atexit.register(self._kill_server)
        try:
            self._wait_for_healthy()
        except Exception:
            self._stop_server()
            raise

    def _wait_for_healthy(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            if self._server_process and self._server_process.poll() is not None:
                raise RuntimeError(f"Server exited with code {self._server_process.returncode}")
            try:
                resp = requests.get(f"{self._base_url}/health", timeout=3)
                if resp.ok:
                    logger.info("Server ready at %s", self._base_url)
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(self.config.health_check_interval)
        raise TimeoutError(f"Server not healthy within {self.config.startup_timeout}s at {self._base_url}")

    def _stop_server(self) -> None:
        proc = self._server_process
        if proc is None:
            return
        self._server_process = None
        if proc.poll() is not None:
            return
        logger.info("Stopping local server (pid=%d)", proc.pid)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not exit after SIGTERM, sending SIGKILL")
            proc.kill()
            proc.wait(timeout=5)

    def _kill_server(self) -> None:
        try:
            self._stop_server()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Job initialization
    # ------------------------------------------------------------------

    def _initialize_jobs(self) -> None:
        self._cleanup_stale_jobs()
        data = self._post_initialize("sampling")
        self._sampling_job_id = data["job_id"]
        if self.config.log_prob_gpus > 0:
            data = self._post_initialize("log_prob")
            self._log_prob_job_id = data["job_id"]
        data = self._post_initialize("training")
        self._training_job_id = data["job_id"]
        self._wait_for_jobs_running()

    def _cleanup_stale_jobs(self) -> None:
        """Destroy any leftover jobs from a previous run before creating new ones."""
        try:
            resp = self._session.get(f"{self._base_url}/status", timeout=5)
            if not resp.ok:
                return
            for job_id_str, info in resp.json().get("jobs", {}).items():
                try:
                    self._session.post(
                        f"{self._base_url}/destroy",
                        params={"job_id": int(job_id_str)},
                        json={"job_type": info["job_type"]},
                        timeout=10,
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _wait_for_jobs_running(self) -> None:
        """Poll until all three jobs reach RUNNING status."""
        jobs = [
            (self._training_job_id, "training"),
            (self._sampling_job_id, "sampling"),
            (self._log_prob_job_id, "log_prob"),
        ]
        timeout = self.config.job_ready_timeout
        for job_id, label in jobs:
            if job_id is None:
                continue
            logger.info("Waiting for %s job %s to be RUNNING...", label, job_id)
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    resp = self._session.get(f"{self._base_url}/job/{job_id}", timeout=5)
                    if resp.ok and resp.json().get("status") == "RUNNING":
                        logger.info("%s job %s is RUNNING", label, job_id)
                        break
                except Exception:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError(f"{label} job {job_id} did not become RUNNING within {timeout}s")

    def _post_initialize(self, job_type: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
                payload["ds_config"] = self.config.ds_config
            if job_type == "training" and self.config.training_config is not None:
                payload["training_config"] = self.config.training_config
                payload["ds_config"] = self.config.ds_config
            elif job_type == "log_prob":
                # Treat log_prob like training: send the base ds_config plus a
                # dedicated log_prob_config. The worker builds a forward-only
                # (no-optimizer) engine from log_prob_config.
                payload["ds_config"] = self.config.ds_config
                if self.config.log_prob_ds_config is not None:
                    payload["log_prob_config"] = self.config.log_prob_ds_config
            else:
                payload["ds_config"] = self.config.ds_config
            # this config is usually the same for any DeepspeedWorker job type - e.g. activating zorro - but it could be made job-type-specific as well, similar to do `ds_config`
            payload["ds_worker_config"] = self.config.ds_worker_config
            if job_type == "training":
                payload["checkpoint_path"] = self.config.checkpoint_path
        else:
            if self.config.vllm_config:
                payload["vllm_config"] = self.config.vllm_config

        resp = self._session.post(f"{self._base_url}/initialize", json=payload)
        if not resp.ok:
            logger.error(f"Failed to initialize {job_type} job: {resp.status_code} {resp.text}")
            resp.raise_for_status()
        return resp.json()

    def _destroy_job(self, job_id: int, job_type: str) -> None:
        try:
            self._session.post(
                f"{self._base_url}/destroy",
                params={"job_id": job_id},
                json={"job_type": job_type},
            )
        except Exception:
            logger.warning(f"Failed to destroy {job_type} job {job_id}", exc_info=True)

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
        tname_e2e = timers.start("xyz arctic_rl.client fwd_bwd")
        tname = timers.start("xyz arctic_rl.client incoming processing 1")
        if processing is not None:
            batch = {**batch, "processing": processing}
        buffer = io.BytesIO()
        timers.stop_and_print_elapsed(tname)

        tname = timers.start("xyz arctic_rl.client incoming processing 2")
        torch.save(batch, buffer)
        timers.stop_and_print_elapsed(tname)

        request_body = buffer.getvalue()
        # import zlib
        # tname = timers.start("xyz arctic_rl.client compress")
        # request_body = zlib.compress(request_body)
        # timers.stop_and_print_elapsed(tname)

        tname = timers.start("xyz arctic_rl.client http post")
        resp = self._session.post(
            f"{self._base_url}/fwd-bwd",
            params={"job_id": self.training_job_id},
            data=request_body,
            headers={
                "Content-Type": "application/octet-stream",
                #                "Content-Encoding": "gzip",
            },
        )
        timers.stop_and_print_elapsed(tname)

        tname = timers.start("xyz arctic_rl.client outgoing processing 1")
        resp.raise_for_status()
        response = torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=False)
        timers.stop_and_print_elapsed(tname)
        pr0(f"[ArcticRLClient] fwd_bwd OUTPUT: {response.keys()=}")
        timers.stop_and_print_elapsed(tname_e2e)

        return response

    async def fwd_no_grad(self, batch: dict, reference_model: bool) -> dict[str, Any]:
        """Forward-only pass (no gradient) on the training engine.

        Returns a binary (torch.save) response containing ``{"logprobs": Tensor}``.
        Matches the ``/fwd-no-grad`` endpoint added in dss-platform PR #41.
        """
        buffer = io.BytesIO()
        torch.save(batch, buffer)

        request_body = buffer.getvalue()

        # import zlib
        # request_body = zlib.compress(buffer.getvalue())

        # len() is the actual payload size; sys.getsizeof measures Python object overhead.
        pr0(f"fwd_no_grad size of outgoing buffer {len(request_body)}")

        job_id = self.log_prob_job_id if reference_model else self.training_job_id
        pr0(f"[ArcticRLClient] fwd_no_grad INPUT: {batch.keys()=} {job_id=}")
        resp = self._session.post(
            f"{self._base_url}/fwd-no-grad",
            params={"job_id": job_id},
            data=request_body,
            # data=buffer.getvalue(),
            headers={
                "Content-Type": "application/octet-stream",
                #                "Content-Encoding": "gzip",
            },
        )
        resp.raise_for_status()
        response = torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=False)
        pr0(f"[ArcticRLClient] fwd_no_grad OUTPUT: {response.keys()=}")
        return response

    async def step(self) -> dict[str, Any]:
        """Optimizer step on the training engine."""
        resp = self._session.post(
            f"{self._base_url}/step",
            params={"job_id": self.training_job_id},
        )
        resp.raise_for_status()
        response = resp.json()
        pr0(f"[ArcticRLClient] step OUTPUT: {response.keys()=}")
        return response

    async def save_checkpoint(self) -> dict[str, Any]:
        """Save training checkpoint."""
        resp = self._session.post(
            f"{self._base_url}/save-checkpoint",
            params={"job_id": self.training_job_id},
        )
        resp.raise_for_status()
        response = resp.json()
        pr0(f"[ArcticRLClient] save_checkpoint OUTPUT: {response.keys()=}")
        return response

    async def save_weights(self, path: str) -> dict[str, Any]:
        """Tell the sampling engine to reload weights from a checkpoint path.

        Note: disk-based weight sync is not yet fully implemented on the server
        side — this call will warn if the endpoint returns an error.
        """
        resp = self._session.post(
            f"{self._base_url}/sync-weights",
            params={"job_id": self.sampling_job_id},
            json={"checkpoint_path": path},
        )
        try:
            resp.raise_for_status()
        except Exception as e:
            logger.warning("reload_weights failed (disk-based reload not yet implemented): %s", e)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    # def generate(
    #     self,
    #     prompts: list[str],
    #     sampling_params: dict[str, Any] | None = None,
    # ) -> list[dict[str, Any]]:
    #     """Generate text completions from the sampling engine."""
    #     resp = self._session.post(
    #         f"{self._base_url}/generate",
    #         params={"job_id": self.sampling_job_id},
    #         json={"prompts": prompts, "sampling_params": sampling_params},
    #     )
    #     resp.raise_for_status()
    #     return resp.json()["results"]

    async def generate(
        self,
        prompts: list[str],
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async version of :meth:`generate`.

        Uses a per-call ``aiohttp.ClientSession`` so the coroutine is safe to
        run from any event loop (driver, Ray actor, etc.) without sharing
        connector state across loops.
        """
        # Per-call session avoids cross-event-loop reuse issues; localhost
        # keep-alive gains are negligible here.
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/generate",
                params={"job_id": self.sampling_job_id},
                json={"prompts": prompts, "sampling_params": sampling_params},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return data["results"]

    # ------------------------------------------------------------------
    # Colocated lifecycle
    # ------------------------------------------------------------------

    async def sleep_inference(self, level: int = 1) -> dict[str, Any]:
        """Put the sampling inference engine to sleep (free GPU memory)."""
        resp = self._session.post(
            f"{self._base_url}/sleep-inference",
            params={"job_id": self.sampling_job_id, "level": level},
        )
        resp.raise_for_status()
        return resp.json()

    async def wake_inference(self, tags: list[str] | None = None) -> dict[str, Any]:
        """Wake the sampling inference engine."""
        resp = self._session.post(
            f"{self._base_url}/wake-inference",
            params={"job_id": self.sampling_job_id},
            json=tags,
        )
        resp.raise_for_status()
        return resp.json()

    async def reset_prefix_cache(self) -> dict[str, Any]:
        """Reset the prefix cache on the sampling inference engine."""
        resp = self._session.post(
            f"{self._base_url}/reset-prefix-cache",
            params={"job_id": self.sampling_job_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def sleep_training(self, mode: str = "all") -> dict[str, Any]:
        """Offload training state to CPU (sleep training workers).

        mode='all':       Everything (training → inference transition)
        mode='non_lp':    Keep bf16 params, offload rest (before CUDA IPC)
        mode='lp_params': Offload bf16 params only (after CUDA IPC)
        """
        resp = self._session.post(
            f"{self._base_url}/sleep-training",
            params={"job_id": self.training_job_id, "mode": mode},
        )
        resp.raise_for_status()
        return resp.json()

    async def wake_training(self) -> dict[str, Any]:
        """Reload all training state to GPU (wake training workers)."""
        resp = self._session.post(
            f"{self._base_url}/wake-training",
            params={"job_id": self.training_job_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def sleep_log_prob(self) -> dict[str, Any]:
        """Offload the reference (log-prob) DeepSpeed engine to CPU."""
        job_id = self.log_prob_job_id
        if job_id is None:
            return {"status": "no_log_prob_job"}
        resp = self._session.post(
            f"{self._base_url}/sleep-log-prob",
            params={"job_id": job_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def wake_log_prob(self) -> dict[str, Any]:
        """Reload the reference (log-prob) DeepSpeed engine to GPU."""
        job_id = self.log_prob_job_id
        if job_id is None:
            return {"status": "no_log_prob_job"}
        resp = self._session.post(
            f"{self._base_url}/wake-log-prob",
            params={"job_id": job_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def empty_training_cache(self) -> dict[str, Any]:
        """Release PyTorch cached GPU memory on all training workers."""
        resp = self._session.post(
            f"{self._base_url}/empty-training-cache",
            params={"job_id": self.training_job_id},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    async def sync_weights(self, cuda_ipc: bool = False, low_memory: bool = False) -> dict[str, Any]:
        """Sync training model weights to the sampling engine.

        3 modes:
        1. In non-colocated mode, uses NCCL.
        2. In colocated mode:
          - cuda_ipc=True: zero-copy CUDA IPC (training weights must be on GPU)
          - cuda_ipc=False: CPU file path (works when training is offloaded)

        low_memory only applies to the cuda_ipc path (stream one gathered param
        at a time). Not yet implemented on the HTTP path -- see the server-side
        guard in /sync-weights; use the ray protocol for low_memory_weight_sync.
        """

        resp = self._session.post(
            f"{self._base_url}/wake-inference",
            params={"job_id": self.sampling_job_id},
            json=["weights"],
        )
        resp.raise_for_status()

        resp = self._session.post(
            f"{self._base_url}/sync-weights",
            json={
                "training_job_id": self.training_job_id,
                "sampling_job_id": self.sampling_job_id,
                "colocate": self.config.colocate,
                "cuda_ipc": cuda_ipc,
                "low_memory": low_memory,
            },
        )
        resp.raise_for_status()
        response = resp.json()

        resp = self._session.post(
            f"{self._base_url}/wake-inference",
            params={"job_id": self.sampling_job_id},
            json=["kv_cache"],
        )
        resp.raise_for_status()

        resp = self._session.post(
            f"{self._base_url}/reset-prefix-cache",
            params={"job_id": self.sampling_job_id},
        )
        resp.raise_for_status()

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
        resp = self._session.post(
            f"{self._base_url}/log-probs",
            params={"job_id": self.log_prob_job_id},
            json={"prompts": prompts, "completions": completions, "top_k": top_k},
        )
        resp.raise_for_status()
        return torch.load(io.BytesIO(resp.content), map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Destroy all jobs and stop the local server if applicable."""
        for job_type, job_id in [
            ("training", self._training_job_id),
            ("sampling", self._sampling_job_id),
            ("log_prob", self._log_prob_job_id),
        ]:
            if job_id is not None:
                self._destroy_job(job_id, job_type)

        self._training_job_id = None
        self._sampling_job_id = None
        self._log_prob_job_id = None
        self._stop_server()
