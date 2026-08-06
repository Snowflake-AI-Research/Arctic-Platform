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
`config.backend`. Sync and async frontends share the Request builders; they
differ only at `transport.call` vs `transport.acall`.
"""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import overload

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport


def make_transport(config: ArcticRLClientConfig) -> Transport:
    from arctic_platform.client.transports.onprem_http import HttpTransport
    from arctic_platform.client.transports.onprem_ray import RayTransport

    if config.backend == "onprem" and config.backend_config.comm_protocol == "ray":
        return RayTransport(config)
    return HttpTransport(config)  # onprem (HTTP)


# ── Request builders (op vocabulary defined once) ───────────────────────────


def _fwd_bwd_request(
    jobs: JobHandles, batch: dict, processing: dict | None = None, router_replay: Any = None
) -> Request:
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
    body = dict(batch)
    body.update({k: v for k, v in (("processing", processing), ("router_replay", router_replay)) if v is not None})
    return Request("forward-backward", jobs.require("training"), body, binary=True)


def _fwd_no_grad_request(jobs: JobHandles, batch: dict) -> Request:
    return Request("forward", jobs.require("training"), dict(batch), binary=True)


def _step_request(jobs: JobHandles, learning_rate: float | None = None) -> Request:
    # NOTE: response *shapes* also diverge across backends -- on-prem returns
    # {"metrics": {"grad_norm", ...}, ...} (scalars can be per-DP-rank lists),
    # while Cortex returns just the loss with no metrics dict. Callers cope
    # via graceful key lookups today.
    # TODO(unify-backends): converge the server-side fwd_bwd/step *responses*
    # on ONE schema (loss + metrics) so callers never key on backend.
    return Request("step", jobs.require("training"), {"learning_rate": learning_rate})


def _save_checkpoint_request(
    jobs: JobHandles, checkpoint_id: str | None = None, checkpoint_type: str = "resumable"
) -> Request:
    # Matches Cortex's `save(job_id, checkpoint_id=None, checkpoint_type=None)`.
    body = {"checkpoint_id": checkpoint_id, "checkpoint_type": checkpoint_type}
    return Request("save", jobs.require("training"), body)


def _generate_request(
    jobs: JobHandles,
    prompts: list,
    sampling_params: dict | None = None,
    routing_key: Any = None,
    strict: bool = False,
) -> Request:
    body = {"prompts": prompts, "sampling_params": sampling_params, "routing_key": routing_key, "strict": strict}
    return Request("generate", jobs.require("sampling"), body)


def _log_probs_request(jobs: JobHandles, prompts: list, completions: list | None = None, top_k: int = 1) -> Request:
    body = {"prompts": prompts, "completions": completions, "top_k": top_k}
    return Request("log-probs", jobs.require("log_prob"), body)


def _sync_weights_request(jobs: JobHandles) -> Request:
    # The client assembles the full Cortex `/operation` envelope here so transports
    # just forward it: SnowAPI reads the `sub_job_*` routing hints, on-prem accepts
    # the same shape and ignores them (it addresses jobs by job id). On-prem treats a
    # sub_job_id as its plain job id, so source/target ids double as its job ids.
    tid, sid = jobs.require("training"), jobs.require("sampling")
    body = {
        "operation_type": "weight-sync",
        "sub_job_id": tid,
        "sub_job_type": "training",
        "payload": {"source_sub_job_id": tid, "target_sub_job_ids": [sid]},
    }
    return Request("operation", tid, body)


def _reset_prefix_cache_request(
    jobs: JobHandles, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1
) -> Request:
    sid = jobs.require("sampling")
    body = {
        "operation_type": "reset-prefix-cache",
        "sub_job_type": "sampling",
        "payload": {"drain": drain, "timeout_s": timeout_s, "retry_interval_s": retry_interval_s},
    }
    return Request("operation", sid, body)


def _reconnect_config(config: ArcticRLClientConfig, jobs: JobHandles) -> ArcticRLClientConfig:
    """A serializable config that reattaches to these jobs in another process."""
    return config.model_copy(
        update={
            "training_job_id": jobs.training,
            "sampling_job_id": jobs.sampling,
            "log_prob_job_id": jobs.log_prob,
        }
    )


class SyncArcticRLClient:
    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.transport = make_transport(config)
        self.jobs = self.transport.initialize()

    # ── training ─────────────────────────────────────────────────────────
    def fwd_bwd(self, batch: dict, processing: dict | None = None, router_replay: Any = None) -> dict:
        return self.transport.call(_fwd_bwd_request(self.jobs, batch, processing, router_replay))

    def fwd_no_grad(self, batch: dict) -> dict:
        return self.transport.call(_fwd_no_grad_request(self.jobs, batch))

    def step(self, learning_rate: float | None = None) -> dict:
        return self.transport.call(_step_request(self.jobs, learning_rate))

    def save_checkpoint(self, checkpoint_id: str | None = None, checkpoint_type: str = "resumable") -> dict:
        return self.transport.call(_save_checkpoint_request(self.jobs, checkpoint_id, checkpoint_type))

    # ── sampling / log-prob ──────────────────────────────────────────────
    def generate(
        self, prompts: list, sampling_params: dict | None = None, routing_key: Any = None, strict: bool = False
    ) -> list:
        return self.transport.call(_generate_request(self.jobs, prompts, sampling_params, routing_key, strict))[
            "results"
        ]

    def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        return self.transport.call(_log_probs_request(self.jobs, prompts, completions, top_k))

    # ── weight sync + cache ──────────────────────────────────────────────
    def sync_weights(self) -> dict:
        return self.transport.call(_sync_weights_request(self.jobs))

    def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1) -> dict:
        return self.transport.call(_reset_prefix_cache_request(self.jobs, drain, timeout_s, retry_interval_s))

    # ── lifecycle ────────────────────────────────────────────────────────
    def reconnect_config(self) -> ArcticRLClientConfig:
        return _reconnect_config(self.config, self.jobs)

    def shutdown(self) -> None:
        self.transport.shutdown()

    def __enter__(self) -> SyncArcticRLClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


class ArcticRLClient:
    """The async client. Use `SyncArcticRLClient` for blocking calls."""

    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.transport = make_transport(config)
        self.jobs = self.transport.initialize()

    # ── training ─────────────────────────────────────────────────────────
    async def fwd_bwd(self, batch: dict, processing: dict | None = None, router_replay: Any = None) -> dict:
        return await self.transport.acall(_fwd_bwd_request(self.jobs, batch, processing, router_replay))

    async def fwd_no_grad(self, batch: dict) -> dict:
        return await self.transport.acall(_fwd_no_grad_request(self.jobs, batch))

    async def step(self, learning_rate: float | None = None) -> dict:
        return await self.transport.acall(_step_request(self.jobs, learning_rate))

    async def save_checkpoint(self, checkpoint_id: str | None = None, checkpoint_type: str = "resumable") -> dict:
        return await self.transport.acall(_save_checkpoint_request(self.jobs, checkpoint_id, checkpoint_type))

    # ── sampling / log-prob ──────────────────────────────────────────────
    async def generate(
        self, prompts: list, sampling_params: dict | None = None, routing_key: Any = None, strict: bool = False
    ) -> list:
        return (
            await self.transport.acall(_generate_request(self.jobs, prompts, sampling_params, routing_key, strict))
        )["results"]

    async def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        return await self.transport.acall(_log_probs_request(self.jobs, prompts, completions, top_k))

    # ── weight sync + cache ──────────────────────────────────────────────
    async def sync_weights(self) -> dict:
        return await self.transport.acall(_sync_weights_request(self.jobs))

    async def reset_prefix_cache(
        self, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1
    ) -> dict:
        return await self.transport.acall(_reset_prefix_cache_request(self.jobs, drain, timeout_s, retry_interval_s))

    # ── lifecycle ────────────────────────────────────────────────────────
    def reconnect_config(self) -> ArcticRLClientConfig:
        return _reconnect_config(self.config, self.jobs)

    async def shutdown(self) -> None:
        # aiohttp's session must be closed with await; the sync shutdown() that
        # tears down jobs can't do that from inside a running event loop.
        await self.transport.aclose()
        self.transport.shutdown()

    async def __aenter__(self) -> ArcticRLClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.shutdown()


@overload
def create_arctic_rl_client(
    config: ArcticRLClientConfig, *, blocking_calls: Literal[False] = ...
) -> ArcticRLClient: ...


@overload
def create_arctic_rl_client(config: ArcticRLClientConfig, *, blocking_calls: Literal[True]) -> SyncArcticRLClient: ...


def create_arctic_rl_client(
    config: ArcticRLClientConfig, *, blocking_calls: bool = False
) -> ArcticRLClient | SyncArcticRLClient:
    """Async client by default; pass blocking_calls=True for the sync client."""
    return SyncArcticRLClient(config) if blocking_calls else ArcticRLClient(config)
