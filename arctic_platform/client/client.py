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
"""The single frontend + the single definition of the op API.

Every op lives here exactly once: friendly signature -> canonical `Request`
(op name, target job, user-data body). Job ids and wire mechanics belong to the
transport, so the client never mentions them. Switching deployment == changing
`config.backend`.
"""

from __future__ import annotations

from typing import Any

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport


def make_transport(config: ArcticRLClientConfig) -> Transport:
    if config.backend == "cortex":
        from arctic_platform.client.transports.cortex import CortexTransport

        return CortexTransport(config)
    from arctic_platform.client.transports.onprem_http import HttpTransport
    from arctic_platform.client.transports.onprem_ray import RayTransport

    if config.backend == "onprem" and config.comm_protocol == "ray":
        return RayTransport(config)
    return HttpTransport(config)  # onprem (HTTP)


def _flatten_metrics(result: Any) -> Any:
    """Bring ``result["metrics"]`` keys to the top level for legacy readers.

    SkyRL reads ``client.step().get("grad_norm")`` and verl reads
    ``client.fwd_bwd()["avg_loss"]`` at the top level, but the on-prem server
    nests scalars under ``metrics`` and Cortex returns loss-only. Both callers
    predate the unified schema. Flattening once here means the transports can
    keep returning their native shape and callers see a superset dict:

        {"loss": ..., "avg_loss": ..., "grad_norm": ..., "metrics": {...}, ...}

    Existing top-level keys always win over ``metrics`` keys of the same name.
    """
    if not isinstance(result, dict):
        return result
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return result
    merged = {**metrics, **result}
    if "avg_loss" not in merged and "loss" in merged:
        merged["avg_loss"] = merged["loss"]
    return merged


class ArcticRLClient:
    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.transport = make_transport(config)
        self.jobs = self.transport.initialize()

    # ── legacy job-id attributes ─────────────────────────────────────────
    # SkyRL's entrypoint reads pre_client.training_job_id / sampling_job_id /
    # log_prob_job_id after initialize(). Preserve that surface as attributes
    # backed by JobHandles rather than forcing the integration to reach into
    # `client.jobs.*`.
    @property
    def training_job_id(self) -> Any:
        return self.jobs.training

    @property
    def sampling_job_id(self) -> Any:
        return self.jobs.sampling

    @property
    def log_prob_job_id(self) -> Any:
        return self.jobs.log_prob

    def get_server_state(self) -> Any:
        """Expose the transport's server state to callers that reconnect Ray.

        The SkyRL entrypoint reads this after creating the driver-side client
        and forwards it to the Ray worker so the worker can reattach to the
        same on-prem Ray server. Non-Ray transports return ``None``.
        """
        getter = getattr(self.transport, "get_server_state", None)
        return getter() if callable(getter) else None

    # ── training ─────────────────────────────────────────────────────────
    def fwd_bwd(
        self,
        batch: dict,
        processing: dict | None = None,
        router_replay: Any = None,
        **legacy_kwargs: Any,
    ) -> dict:
        # NOTE: the call *signature* is unified across backends, but `batch`'s
        # *content* is not: Cortex expects an RPC-style
        # {"args": [...], "kwargs": {...}} that the server tokenizes, while on-prem
        # expects a pre-tokenized verl-GRPO {"batch", "meta", "processing"}. The
        # client forwards `batch` verbatim, so today the caller must still match
        # the target backend's data contract.
        # TODO(unify-backends): converge the server-side fwd_bwd on ONE batch
        # contract (ideally Cortex's) so this frontend is truly backend-agnostic
        # and callers stop branching on backend. Until then `processing` is folded
        # into the body here, which is why callers can leave it inside `batch`.
        # Legacy callers (verl `compute_log_prob`, SkyRL wire packer) pass
        # `post_processors=[...]`, `context={...}` or `kwargs={...}` alongside
        # `batch`; we fold them into the body so the transports can forward one
        # uniform envelope.
        body = dict(batch)
        body.update({k: v for k, v in (("processing", processing), ("router_replay", router_replay)) if v is not None})
        body.update({k: v for k, v in legacy_kwargs.items() if v is not None})
        result = self.transport.call(Request("fwd-bwd", self.jobs.require("training"), body, binary=True))
        return _flatten_metrics(result)

    def fwd_no_grad(self, batch: dict, **legacy_kwargs: Any) -> dict:
        # SkyRL's arctic_generator / _ArcticDispatch calls this with
        # `post_processors=["logprobs"]`; verl calls it with `context=...` and
        # `processing=...`. Fold every extra kwarg into the body so transports
        # receive one uniform envelope (see `CortexTransport._fwd_no_grad`).
        body = dict(batch)
        body.update({k: v for k, v in legacy_kwargs.items() if v is not None})
        result = self.transport.call(Request("fwd-no-grad", self.jobs.require("training"), body, binary=True))
        return _flatten_metrics(result)

    def step(self, learning_rate: float | None = None) -> dict:
        # NOTE: response *shapes* also diverge across backends -- on-prem returns
        # {"metrics": {"grad_norm", ...}, ...} (scalars can be per-DP-rank lists),
        # while Cortex returns just the loss with no metrics dict. Callers cope
        # via graceful key lookups today.
        # TODO(unify-backends): converge the server-side fwd_bwd/step *responses*
        # on ONE schema (loss + metrics) so callers never key on backend.
        result = self.transport.call(Request("step", self.jobs.require("training"), {"learning_rate": learning_rate}))
        return _flatten_metrics(result)

    def save_checkpoint(self, stage_info: dict | None = None, path: str | None = None) -> dict:
        # `path` is this call's destination; when None the server uses the job's
        # config.checkpoint_path. An explicit `path` here wins over the config.
        body = {"stage_info": stage_info, "path": path}
        return self.transport.call(Request("save-checkpoint", self.jobs.require("training"), body))

    # ── sampling / log-prob ──────────────────────────────────────────────
    def generate(
        self,
        prompts: list,
        sampling_params: dict | None = None,
        routing_key: Any = None,
    ) -> list:
        body = {"prompts": prompts, "sampling_params": sampling_params, "routing_key": routing_key}
        return self.transport.call(Request("generate", self.jobs.require("sampling"), body))["results"]

    def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        body = {"prompts": prompts, "completions": completions, "top_k": top_k}
        return self.transport.call(Request("log-probs", self.jobs.require("log_prob"), body))

    # ── weight sync + cache ──────────────────────────────────────────────
    def sync_weights(self, cuda_ipc: bool = False, low_memory: bool = False) -> dict:
        # SkyRL's colocated path calls `sync_weights(cuda_ipc=True)`. On-prem
        # honors the flag (same-GPU IPC handoff); Cortex has no colocation
        # concept and silently ignores it. Both flags travel in the body so the
        # transport decides what's meaningful.
        tid, sid = self.jobs.require("training"), self.jobs.require("sampling")
        body = {
            "training_job_id": tid,
            "sampling_job_id": sid,
            "cuda_ipc": cuda_ipc,
            "low_memory": low_memory,
        }
        return self.transport.call(Request("sync-weights", None, body))

    def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0) -> dict:
        body = {"drain": drain, "timeout_s": timeout_s}
        return self.transport.call(Request("reset-prefix-cache", self.jobs.require("sampling"), body))

    # ── colocation lifecycle (legacy SkyRL / verl surface) ───────────────
    # SkyRL's `_ArcticDispatch.save_weights_for_sampler` and the on-prem
    # colocated path call these unconditionally when `colocate=True`. On-prem
    # implements each op server-side; Cortex has no colocation concept and
    # returns an empty dict (its transport table registers them as no-ops).
    # Exposing them here (rather than raising `AttributeError`) is what lets
    # SkyRL / verl run against `backend=cortex` unmodified.
    #
    # verl passes `tags=...` on wake and `level=...` on sleep — those are
    # engine hints (e.g. vLLM sleep level). We fold every extra kwarg into
    # the body so on-prem can honor them and Cortex can safely ignore.
    def wake_training(self, **kwargs: Any) -> dict:
        return self._colo("wake-training", "training", body=kwargs)

    def sleep_training(self, **kwargs: Any) -> dict:
        return self._colo("sleep-training", "training", body=kwargs)

    def wake_inference(self, **kwargs: Any) -> dict:
        return self._colo("wake-inference", "sampling", body=kwargs)

    def sleep_inference(self, **kwargs: Any) -> dict:
        return self._colo("sleep-inference", "sampling", body=kwargs)

    def wake_log_prob(self, **kwargs: Any) -> dict:
        return self._colo("wake-log-prob", "log_prob", body=kwargs)

    def sleep_log_prob(self, **kwargs: Any) -> dict:
        return self._colo("sleep-log-prob", "log_prob", body=kwargs)

    def empty_training_cache(self, **kwargs: Any) -> dict:
        return self._colo("empty-training-cache", "training", body=kwargs)

    def weight_norm(self, **kwargs: Any) -> dict:
        return self._colo("weight-norm", "training", body=kwargs)

    def save_weights(self, path: str | None = None) -> dict:
        # Disk-based weight reload; deliberately optional (see UNIFICATION_NOTES).
        body = {"path": path}
        return self.transport.call(Request("save-weights", self.jobs.require("training"), body))

    def _colo(self, op: str, job_type: str, *, body: dict[str, Any] | None = None) -> dict:
        """Dispatch a colocation-lifecycle op if the target job exists.

        The primary target is the requested job type; when it isn't set (e.g.
        Cortex without a log-prob sub-job) the op is a no-op so callers don't
        have to branch on backend. Extra kwargs from the caller (e.g. verl's
        `tags`, `level`) travel in `body` so the transport can decide what's
        meaningful.
        """
        job_id = getattr(self.jobs, job_type, None)
        if job_id is None:
            return {}
        return self.transport.call(Request(op, job_id, dict(body or {})))

    # ── lifecycle ────────────────────────────────────────────────────────
    def reconnect_config(self) -> ArcticRLClientConfig:
        """A serializable config that reattaches to these jobs in another process."""
        return self.config.model_copy(
            update={
                "training_job_id": self.jobs.training,
                "sampling_job_id": self.jobs.sampling,
                "log_prob_job_id": self.jobs.log_prob,
            }
        )

    def shutdown(self) -> None:
        self.transport.shutdown()

    def __enter__(self) -> ArcticRLClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def create_arctic_rl_client(config: ArcticRLClientConfig) -> ArcticRLClient:
    """Factory matching the current OSS entrypoint shape."""
    return ArcticRLClient(config)
