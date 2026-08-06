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
"""Arctic SFT client — HTTP-first, CPU-safe; optional sampling for generate_samples.

Training ops always available. Sampling ops (``generate`` / ``sync_weights`` / …)
require ``sampling_gpus > 0``. Client process: ``CUDA_VISIBLE_DEVICES=`` empty.
Batch contract::

    {
        "batch": [                         # list of GAS microbatches (H3)
            {
                "input_ids": LongTensor[B, S_i],
                "attention_mask": LongTensor[B, S_i],  # optional when packed
                "labels": LongTensor[B, S_i],
                "position_ids": LongTensor[B, S_i],    # required for sample packing
            },
            ...
        ],
        # Legacy demos may still send a single concatenated dict instead of a list.
        "meta": {
            "pad_token_id": int,
            "gas_microbatches": True,       # list form
            "sample_packing": bool,         # optional
        },
        "processing": {"loss_fn": "sft"},
    }

When ``sample_packing`` / ``position_ids`` are set, the server FA2 path uses
HF varlen attention (boundaries from position-id resets) instead of a dense
padded rectangle. The GAS list form avoids client concat → server re-split.
"""

from __future__ import annotations

from typing import Any

from arctic_platform.sft.config import ArcticSFTClientConfig
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport
from arctic_platform.client.transport import initialize_or_cleanup


def _make_transport(config: ArcticSFTClientConfig) -> Transport:
    from arctic_platform.client.transports.onprem_http import HttpTransport
    from arctic_platform.client.transports.onprem_ray import RayTransport

    rl_cfg = config.to_rl_config()
    if config.backend == "onprem" and config.comm_protocol == "ray":
        return RayTransport(rl_cfg)
    return HttpTransport(rl_cfg)


def _training_body(batch: dict, processing: dict | None) -> dict:
    """Wire envelope: explicit ``processing`` wins; else keep/default ``loss_fn: sft``."""
    body = dict(batch)
    if processing is not None:
        body["processing"] = processing
    else:
        body.setdefault("processing", {"loss_fn": "sft"})
    body.setdefault("meta", {})
    return body


def _maybe_print_server_profile(op: str, out: dict | None) -> None:
    from arctic_platform.common.utils import sft_profile

    if not sft_profile.enabled() or not isinstance(out, dict):
        return
    prof = (out.get("metrics") or {}).get("_profile_ms")
    if isinstance(prof, dict) and prof:
        sft_profile.maybe_print(f"server {op}", prof)


class ArcticSFTClient:
    """Training-only SFT client over the on-prem HTTP (or Ray) transport."""

    def __init__(self, config: ArcticSFTClientConfig) -> None:
        self.config = config
        self.transport = _make_transport(config)
        self.jobs = initialize_or_cleanup(self.transport)

    def fwd_bwd(self, batch: dict, processing: dict | None = None) -> dict:
        """Forward + loss + backward. ``batch`` is ``{"batch","meta","processing"}``."""
        body = _training_body(batch, processing)
        out = self.transport.call(Request("fwd-bwd", self.jobs.require("training"), body, binary=True))
        _maybe_print_server_profile("fwd-bwd", out)
        return out

    def fwd_no_grad(self, batch: dict, processing: dict | None = None) -> dict:
        """Forward + loss, no backward (eval)."""
        body = _training_body(batch, processing)
        out = self.transport.call(Request("fwd-no-grad", self.jobs.require("training"), body, binary=True))
        _maybe_print_server_profile("fwd-no-grad", out)
        return out

    def step(self) -> dict:
        """One optimizer update. LR is server-side (DeepSpeed schedule at init); no client override."""
        out = self.transport.call(Request("step", self.jobs.require("training"), {}))
        _maybe_print_server_profile("step", out)
        return out

    def save_checkpoint(
        self,
        path: str | None = None,
        *,
        step: int | None = None,
        export_hf: bool = False,
        save_total_limit: int | None = None,
    ) -> dict:
        """Save. Optional ``step`` → ``…/checkpoint-{step}/``; ``export_hf`` writes ``…/hf/``."""
        body = {
            "path": path,
            "step": step,
            "export_hf": export_hf,
            "save_total_limit": save_total_limit,
        }
        return self.transport.call(Request("save-checkpoint", self.jobs.require("training"), body))

    def load_checkpoint(self, path: str | None = None, *, step: int | None = None) -> dict:
        """Restore weights/optimizer/LR/step. Returns ``{"global_step", ...}`` (0 if none found)."""
        return self.transport.call(
            Request("load-checkpoint", self.jobs.require("training"), {"path": path, "step": step})
        )

    # ── sampling (optional; requires sampling_gpus > 0) ───────────────────
    def generate(self, prompts: list, sampling_params: dict | None = None, routing_key: Any = None) -> list:
        body = {"prompts": prompts, "sampling_params": sampling_params, "routing_key": routing_key}
        return self.transport.call(Request("generate", self.jobs.require("sampling"), body))["results"]

    def sync_weights(self, cuda_ipc: bool = False, low_memory: bool = False) -> dict:
        """Push training weights to the sampling engine (same staging as RL client)."""
        tid, sid = self.jobs.require("training"), self.jobs.require("sampling")
        self.wake_inference(tags=["weights"])
        out = self.transport.call(
            Request(
                "sync-weights",
                None,
                {
                    "training_job_id": tid,
                    "sampling_job_id": sid,
                    "colocate": self.config.colocate,
                    "cuda_ipc": cuda_ipc,
                    "low_memory": low_memory,
                },
            )
        )
        self.wake_inference(tags=["kv_cache"])
        self.reset_prefix_cache()
        return out

    def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0) -> dict:
        body = {"drain": drain, "timeout_s": timeout_s}
        return self.transport.call(Request("reset-prefix-cache", self.jobs.require("sampling"), body))

    def sleep_inference(self, level: int = 1) -> dict:
        return self.transport.call(
            Request("sleep-inference", self.jobs.require("sampling"), {"level": level})
        )

    def wake_inference(self, tags: list | None = None) -> dict:
        return self.transport.call(Request("wake-inference", self.jobs.require("sampling"), tags))

    def sleep_training(self, mode: str = "all") -> dict:
        return self.transport.call(
            Request("sleep-training", self.jobs.require("training"), {"mode": mode})
        )

    def wake_training(self) -> dict:
        return self.transport.call(Request("wake-training", self.jobs.require("training"), {}))

    def reconnect_config(self) -> ArcticSFTClientConfig:
        """Config that reattaches to this training (+ sampling) job from another process."""
        return self.config.model_copy(
            update={
                "training_job_id": self.jobs.training,
                "sampling_job_id": self.jobs.sampling,
            }
        )

    def shutdown(self) -> None:
        self.transport.shutdown()

    def __enter__(self) -> ArcticSFTClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def create_arctic_sft_client(config: ArcticSFTClientConfig) -> ArcticSFTClient:
    return ArcticSFTClient(config)
