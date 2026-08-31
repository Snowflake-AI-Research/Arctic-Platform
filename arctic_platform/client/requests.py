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
"""The op vocabulary, defined exactly once.

Each builder lowers a friendly signature to a canonical `Request`: op name,
target job, user-data body. Nothing here touches a transport or an event loop,
so the sync and async frontends in `base.py` share every builder and differ only
at `transport.call` vs `transport.acall`.

The set of ops built here must match `transport.OPS`; `test_client_ops.py`
asserts that in both directions.
"""

from __future__ import annotations

from typing import Any

from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request


def fwd_bwd_request(
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


def fwd_no_grad_request(
    jobs: JobHandles, batch: dict, processing: dict | None = None, reference_model: bool = False
) -> Request:
    # Forward-only (no grad). Reference log-probs run on the log_prob engine;
    # current-policy log-probs run on the training engine (mirrors the old
    # ray_client.fwd_no_grad reference_model routing).
    #
    # `processing` is folded into the body as in fwd_bwd_request, which is why
    # callers can equivalently leave it inside `batch`.
    job = "log_prob" if reference_model else "training"
    body = dict(batch)
    if processing is not None:
        body["processing"] = processing
    return Request("forward", jobs.require(job), body, binary=True)


def step_request(jobs: JobHandles, learning_rate: float | None = None) -> Request:
    # NOTE: response *shapes* also diverge across backends -- on-prem returns
    # {"metrics": {"grad_norm", ...}, ...} (scalars can be per-DP-rank lists),
    # while Cortex returns just the loss with no metrics dict. Callers cope
    # via graceful key lookups today.
    # TODO(unify-backends): converge the server-side fwd_bwd/step *responses*
    # on ONE schema (loss + metrics) so callers never key on backend.
    #
    # LR is server-authoritative (the DeepSpeed schedule set at init), so an
    # unset learning_rate is omitted rather than sent as an explicit null.
    body = {} if learning_rate is None else {"learning_rate": learning_rate}
    return Request("step", jobs.require("training"), body)


def save_checkpoint_request(
    jobs: JobHandles,
    checkpoint_id: str | None = None,
    checkpoint_type: str = "resumable",
    *,
    path: str | None = None,
    step: int | None = None,
    export_hf: bool = False,
    save_total_limit: int | None = None,
    stage_info: dict | None = None,
) -> Request:
    # Cortex: checkpoint_id/checkpoint_type. On-prem SFT also accepts
    # path/step/export_hf/save_total_limit/stage_info.
    body = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_type": checkpoint_type,
        "path": path,
        "step": step,
        "export_hf": export_hf,
        "save_total_limit": save_total_limit,
        "stage_info": stage_info,
    }
    return Request("save", jobs.require("training"), body)


def load_checkpoint_request(jobs: JobHandles, path: str | None = None, step: int | None = None) -> Request:
    return Request("load-checkpoint", jobs.require("training"), {"path": path, "step": step})


def generate_request(
    jobs: JobHandles,
    prompts: list,
    sampling_params: dict | None = None,
    routing_key: Any = None,
    strict: bool = False,
) -> Request:
    body = {"prompts": prompts, "sampling_params": sampling_params, "routing_key": routing_key, "strict": strict}
    return Request("generate", jobs.require("sampling"), body)


def log_probs_request(jobs: JobHandles, prompts: list, completions: list | None = None, top_k: int = 1) -> Request:
    body = {"prompts": prompts, "completions": completions, "top_k": top_k}
    return Request("log-probs", jobs.require("log_prob"), body)


def sync_weights_request(
    jobs: JobHandles,
    *,
    cuda_ipc: bool | None = None,
    low_memory: bool | None = None,
    weight_format: str | None = None,
    source_sub_job_id: Any = None,
    target_sub_job_ids: list | None = None,
) -> Request:
    # The client assembles the full Cortex `/operation` envelope here so transports
    # just forward it: SnowAPI reads the `sub_job_*` routing hints, on-prem accepts
    # the same shape and ignores them (it addresses jobs by job id). On-prem treats a
    # sub_job_id as its plain job id, so source/target ids double as its job ids.
    #
    # In non-colocated mode the server uses NCCL. In colocated mode, cuda_ipc=True is
    # zero-copy (training weights must be on GPU) and low_memory streams one param at
    # a time to bound peak GPU memory. weight_format="lora" broadcasts only the
    # trained adapter tensors instead of the full model.
    #
    # source/target default to this session's training and sampling jobs. They are
    # overridable for the multi-sub-job Cortex topologies the CLI exposes, where one
    # trainer broadcasts to several samplers that `JobHandles` (one id per role)
    # cannot name.
    tid, sid = jobs.require("training"), jobs.require("sampling")
    source = source_sub_job_id or tid
    payload = {"source_sub_job_id": source, "target_sub_job_ids": list(target_sub_job_ids or [sid])}
    if cuda_ipc is not None:
        payload["cuda_ipc"] = cuda_ipc
    if low_memory is not None:
        payload["low_memory"] = low_memory
    if weight_format is not None:
        payload["weight_format"] = weight_format
    body = {
        "operation_type": "weight-sync",
        "sub_job_id": source,
        "sub_job_type": "training",
        "payload": payload,
    }
    return Request("operation", source, body)


def reset_prefix_cache_request(
    jobs: JobHandles, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1
) -> Request:
    sid = jobs.require("sampling")
    body = {
        "operation_type": "reset-prefix-cache",
        "sub_job_type": "sampling",
        "payload": {"drain": drain, "timeout_s": timeout_s, "retry_interval_s": retry_interval_s},
    }
    return Request("operation", sid, body)


def sleep_inference_request(jobs: JobHandles, level: int = 1) -> Request:
    return Request("sleep-inference", jobs.require("sampling"), {"level": level})


def wake_inference_request(jobs: JobHandles, tags: list | None = None) -> Request:
    # Unlike the other ops the body is the bare tag list, matching the server route.
    return Request("wake-inference", jobs.require("sampling"), tags)


def sleep_training_request(jobs: JobHandles, mode: str = "all") -> Request:
    return Request("sleep-training", jobs.require("training"), {"mode": mode})


def wake_training_request(jobs: JobHandles) -> Request:
    return Request("wake-training", jobs.require("training"), {})
