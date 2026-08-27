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
"""The shared client frontends: transport plumbing plus the whole op surface.

Three layers, so nothing is written twice:

- `_ArcticClientCore` — call-style agnostic. Owns the transport, the job handles, and
  the ops-free lifecycle (`reconnect_config`, `get_server_state`).
- `ArcticClient` — blocking op surface.
- `AsyncArcticClient` — awaitable op surface.

Both op surfaces lower calls with the same builders from `requests.py`, so they
differ only at `transport.call` vs `transport.acall`. Workload clients live in
`sft.py` and `rl.py` and should only add what is genuinely theirs; anything
shared belongs here.

Switching deployment == changing `config.backend`; job ids and wire mechanics
belong to the transport, so a client never mentions them.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import Self

from arctic_platform._dependency_groups import require_any_dep_group
from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.requests import fwd_bwd_request
from arctic_platform.client.requests import fwd_no_grad_request
from arctic_platform.client.requests import generate_request
from arctic_platform.client.requests import load_checkpoint_request
from arctic_platform.client.requests import reset_prefix_cache_request
from arctic_platform.client.requests import save_checkpoint_request
from arctic_platform.client.requests import sleep_inference_request
from arctic_platform.client.requests import sleep_training_request
from arctic_platform.client.requests import step_request
from arctic_platform.client.requests import sync_weights_request
from arctic_platform.client.requests import wake_inference_request
from arctic_platform.client.requests import wake_training_request
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport
from arctic_platform.client.transport import initialize_or_cleanup


def make_transport(config: ArcticClientConfig, server_state: Any = None) -> Transport:
    protocol = config.backend.protocol
    if config.backend.type == "remote" and protocol == "cortex":
        require_any_dep_group("cortex")
        from arctic_platform.client.transports.cortex import CortexTransport

        return CortexTransport(config)

    require_any_dep_group("sft", "rl")
    from arctic_platform.client.transports.onprem_http import HttpTransport
    from arctic_platform.client.transports.onprem_ray import RayTransport

    if protocol == "ray":
        return RayTransport(config, server_state=server_state)
    if server_state is not None:
        raise ValueError("server_state reconnect is only supported by the in-process Ray transport.")
    return HttpTransport(config)


def _check_weight_format(config: ArcticClientConfig, weight_format: str | None) -> None:
    """Refuse a weight_format the deployment would drop on the floor.

    On-prem's ``WeightSyncRequest`` ignores unknown fields, so an unsupported
    format would silently full-sync dense weights instead of the adapter.
    """
    if weight_format is not None and config.backend.type == "onprem":
        raise ValueError(
            f"weight_format={weight_format!r} is only supported by the remote Cortex backend; "
            "the on-prem server always syncs full weights."
        )


def _maybe_print_server_profile(op: str, out: dict | None) -> None:
    """Echo the server's per-op timings when ARL_SFT_PROFILE is set; a no-op otherwise."""
    # TODO(generalize-profiling): extend profiling to every transport and workload in a
    # follow-up PR; nothing here is SFT-specific but the names.
    from arctic_platform import sft_profile

    if not sft_profile.enabled() or not isinstance(out, dict):
        return
    prof = (out.get("metrics") or {}).get("_profile_ms")
    if isinstance(prof, dict) and prof:
        sft_profile.maybe_print(f"server {op}", prof)


class _ArcticClientCore:
    """Transport + job plumbing shared by every frontend.

    Not usable on its own: the op surface lives on `ArcticClient` /
    `AsyncArcticClient`. Only put things here that read the same whether calls
    block or are awaited.
    """

    def __init__(self, config: ArcticClientConfig, server_state: Any = None) -> None:
        self.config = config
        self.transport = make_transport(config, server_state=server_state)
        self.jobs = initialize_or_cleanup(self.transport)

    def reconnect_config(self) -> ArcticClientConfig:
        """A serializable config that reattaches to these jobs in another process."""
        return self.config.model_copy(
            update={
                "training_job_id": self.jobs.training,
                "sampling_job_id": self.jobs.sampling,
                "log_prob_job_id": self.jobs.log_prob,
            }
        )

    def get_server_state(self) -> Any:
        """The transport's server-state handle for cross-process reattach.

        Only meaningful for the in-process Ray transport; other transports raise.
        """
        get_state = getattr(self.transport, "get_server_state", None)
        if get_state is None:
            raise NotImplementedError(f"{type(self.transport).__name__} has no server state to share.")
        return get_state()


class ArcticClient(_ArcticClientCore):
    """The blocking frontend: every op shared by SFT and RL."""

    def _call(self, request: Request) -> dict:
        out = self.transport.call(request)
        _maybe_print_server_profile(request.op, out)
        return out

    # ── training ─────────────────────────────────────────────────────────
    def fwd_bwd(self, batch: dict, processing: dict | None = None, router_replay: Any = None) -> dict:
        return self._call(fwd_bwd_request(self.jobs, batch, processing, router_replay))

    def fwd_no_grad(self, batch: dict, processing: dict | None = None, reference_model: bool = False) -> dict:
        return self._call(fwd_no_grad_request(self.jobs, batch, processing, reference_model))

    def step(self, learning_rate: float | None = None) -> dict:
        return self._call(step_request(self.jobs, learning_rate))

    def save_checkpoint(
        self,
        checkpoint_id: str | None = None,
        checkpoint_type: str = "resumable",
        path: str | None = None,
        *,
        step: int | None = None,
        export_hf: bool = False,
        save_total_limit: int | None = None,
        stage_info: dict | None = None,
    ) -> dict:
        return self._call(
            save_checkpoint_request(
                self.jobs,
                checkpoint_id,
                checkpoint_type,
                path=path,
                step=step,
                export_hf=export_hf,
                save_total_limit=save_total_limit,
                stage_info=stage_info,
            )
        )

    def load_checkpoint(self, path: str | None = None, *, step: int | None = None) -> dict:
        """Restore weights/optimizer/LR/step. Returns ``{"global_step", ...}``."""
        return self._call(load_checkpoint_request(self.jobs, path, step))

    # ── sampling ─────────────────────────────────────────────────────────
    def generate(
        self, prompts: list, sampling_params: dict | None = None, routing_key: Any = None, strict: bool = False
    ) -> list:
        return self._call(generate_request(self.jobs, prompts, sampling_params, routing_key, strict))["results"]

    # ── weight sync + cache ──────────────────────────────────────────────
    def sync_weights(
        self, cuda_ipc: bool | None = None, low_memory: bool | None = None, weight_format: str | None = None
    ) -> dict:
        """Sync training weights to sampling (staged wake → operation → wake → reset).

        ``cuda_ipc`` / ``low_memory`` default to the training job's ``TrainingConfig``; pass a value to override this
        call. ``weight_format="lora"`` broadcasts only the adapter tensors (Cortex only).
        """
        _check_weight_format(self.config, weight_format)
        self.wake_inference(tags=["weights"])
        out = self._call(
            sync_weights_request(self.jobs, cuda_ipc=cuda_ipc, low_memory=low_memory, weight_format=weight_format)
        )
        self.wake_inference(tags=["kv_cache"])
        self.reset_prefix_cache()
        return out

    def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1) -> dict:
        return self._call(reset_prefix_cache_request(self.jobs, drain, timeout_s, retry_interval_s))

    def sleep_inference(self, level: int = 1) -> dict:
        return self._call(sleep_inference_request(self.jobs, level))

    def wake_inference(self, tags: list | None = None) -> dict:
        return self._call(wake_inference_request(self.jobs, tags))

    def sleep_training(self, mode: str = "all") -> dict:
        return self._call(sleep_training_request(self.jobs, mode))

    def wake_training(self) -> dict:
        return self._call(wake_training_request(self.jobs))

    # ── lifecycle ────────────────────────────────────────────────────────
    def shutdown(self) -> None:
        self.transport.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


class AsyncArcticClient(_ArcticClientCore):
    """The awaitable frontend; the async twin of `ArcticClient`."""

    async def _acall(self, request: Request) -> dict:
        out = await self.transport.acall(request)
        _maybe_print_server_profile(request.op, out)
        return out

    # ── training ─────────────────────────────────────────────────────────
    async def fwd_bwd(self, batch: dict, processing: dict | None = None, router_replay: Any = None) -> dict:
        return await self._acall(fwd_bwd_request(self.jobs, batch, processing, router_replay))

    async def fwd_no_grad(self, batch: dict, processing: dict | None = None, reference_model: bool = False) -> dict:
        return await self._acall(fwd_no_grad_request(self.jobs, batch, processing, reference_model))

    async def step(self, learning_rate: float | None = None) -> dict:
        return await self._acall(step_request(self.jobs, learning_rate))

    async def save_checkpoint(
        self,
        checkpoint_id: str | None = None,
        checkpoint_type: str = "resumable",
        path: str | None = None,
        *,
        step: int | None = None,
        export_hf: bool = False,
        save_total_limit: int | None = None,
        stage_info: dict | None = None,
    ) -> dict:
        return await self._acall(
            save_checkpoint_request(
                self.jobs,
                checkpoint_id,
                checkpoint_type,
                path=path,
                step=step,
                export_hf=export_hf,
                save_total_limit=save_total_limit,
                stage_info=stage_info,
            )
        )

    async def load_checkpoint(self, path: str | None = None, *, step: int | None = None) -> dict:
        return await self._acall(load_checkpoint_request(self.jobs, path, step))

    # ── sampling ─────────────────────────────────────────────────────────
    async def generate(
        self, prompts: list, sampling_params: dict | None = None, routing_key: Any = None, strict: bool = False
    ) -> list:
        return (await self._acall(generate_request(self.jobs, prompts, sampling_params, routing_key, strict)))[
            "results"
        ]

    # ── weight sync + cache ──────────────────────────────────────────────
    async def sync_weights(
        self, cuda_ipc: bool | None = None, low_memory: bool | None = None, weight_format: str | None = None
    ) -> dict:
        """Async twin of ArcticClient.sync_weights (staged wake → operation → wake → reset)."""
        _check_weight_format(self.config, weight_format)
        await self.wake_inference(tags=["weights"])
        out = await self._acall(
            sync_weights_request(self.jobs, cuda_ipc=cuda_ipc, low_memory=low_memory, weight_format=weight_format)
        )
        await self.wake_inference(tags=["kv_cache"])
        await self.reset_prefix_cache()
        return out

    async def reset_prefix_cache(
        self, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1
    ) -> dict:
        return await self._acall(reset_prefix_cache_request(self.jobs, drain, timeout_s, retry_interval_s))

    async def sleep_inference(self, level: int = 1) -> dict:
        return await self._acall(sleep_inference_request(self.jobs, level))

    async def wake_inference(self, tags: list | None = None) -> dict:
        return await self._acall(wake_inference_request(self.jobs, tags))

    async def sleep_training(self, mode: str = "all") -> dict:
        return await self._acall(sleep_training_request(self.jobs, mode))

    async def wake_training(self) -> dict:
        return await self._acall(wake_training_request(self.jobs))

    # ── lifecycle ────────────────────────────────────────────────────────
    async def shutdown(self) -> None:
        # aiohttp's session must be closed with await; the sync shutdown() that
        # tears down jobs can't do that from inside a running event loop.
        await self.transport.aclose()
        self.transport.shutdown()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.shutdown()
