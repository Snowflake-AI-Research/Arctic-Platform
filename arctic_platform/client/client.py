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


class ArcticRLClient:
    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.transport = make_transport(config)
        self.jobs = self.transport.initialize()

    # ── training ─────────────────────────────────────────────────────────
    def fwd_bwd(self, batch: dict, processing: dict | None = None, router_replay: Any = None) -> dict:
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
        return self.transport.call(Request("forward-backward", self.jobs.require("training"), body, binary=True))

    def fwd_no_grad(self, batch: dict) -> dict:
        return self.transport.call(Request("forward", self.jobs.require("training"), dict(batch), binary=True))

    def step(self, learning_rate: float | None = None) -> dict:
        # NOTE: response *shapes* also diverge across backends -- on-prem returns
        # {"metrics": {"grad_norm", ...}, ...} (scalars can be per-DP-rank lists),
        # while Cortex returns just the loss with no metrics dict. Callers cope
        # via graceful key lookups today.
        # TODO(unify-backends): converge the server-side fwd_bwd/step *responses*
        # on ONE schema (loss + metrics) so callers never key on backend.
        return self.transport.call(Request("step", self.jobs.require("training"), {"learning_rate": learning_rate}))

    def save_checkpoint(self, checkpoint_id: str | None = None, checkpoint_type: str = "resumable") -> dict:
        # Matches Cortex's `save(job_id, checkpoint_id=None, checkpoint_type=None)`.
        body = {"checkpoint_id": checkpoint_id, "checkpoint_type": checkpoint_type}
        return self.transport.call(Request("save", self.jobs.require("training"), body))

    # ── sampling / log-prob ──────────────────────────────────────────────
    def generate(
        self, prompts: list, sampling_params: dict | None = None, routing_key: Any = None, strict: bool = False
    ) -> list:
        body = {"prompts": prompts, "sampling_params": sampling_params, "routing_key": routing_key, "strict": strict}
        return self.transport.call(Request("generate", self.jobs.require("sampling"), body))["results"]

    def log_probs(self, prompts: list, completions: list | None = None, top_k: int = 1) -> dict:
        body = {"prompts": prompts, "completions": completions, "top_k": top_k}
        return self.transport.call(Request("log-probs", self.jobs.require("log_prob"), body))

    # ── weight sync + cache (control-plane ops -> the /operation envelope) ──
    # The client assembles the full Cortex `/operation` envelope here so transports
    # just forward it: SnowAPI reads the `sub_job_*` routing hints, on-prem accepts
    # the same shape and ignores them (it addresses jobs by job id). On-prem treats a
    # sub_job_id as its plain job id, so source/target ids double as its job ids.
    def sync_weights(self) -> dict:
        tid, sid = self.jobs.require("training"), self.jobs.require("sampling")
        body = {
            "operation_type": "weight-sync",
            "sub_job_id": tid,
            "sub_job_type": "training",
            "payload": {"source_sub_job_id": tid, "target_sub_job_ids": [sid]},
        }
        return self.transport.call(Request("operation", tid, body))

    def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1) -> dict:
        sid = self.jobs.require("sampling")
        body = {
            "operation_type": "reset-prefix-cache",
            "sub_job_type": "sampling",
            "payload": {"drain": drain, "timeout_s": timeout_s, "retry_interval_s": retry_interval_s},
        }
        return self.transport.call(Request("operation", sid, body))

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
