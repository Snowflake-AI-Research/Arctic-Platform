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

"""On-policy distillation as two ``SyncArcticRLClient``s: student + frozen teacher."""

from __future__ import annotations

from typing import Any

from arctic_platform.client.client import SyncArcticRLClient
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.transport import Request
from arctic_platform.opd.config import ArcticOPDClientConfig

DEFAULT_PROCESSING = {
    "loss_fn": "on_policy_distill",
    "post": ["compute_entropy_and_logprobs"],
    "config": {
        "distill_estimator": "low_var_kl",
        "kl_coef": 1.0,
        "loss_agg_mode": "token-mean",
    },
    "return_post_outputs": False,
}


def _fwd_bwd_body(
    config: ArcticOPDClientConfig,
    batch: dict,
    processing: dict | None,
    meta: dict | None = None,
) -> dict:
    descriptor = processing or DEFAULT_PROCESSING
    meta_payload = {"zorro_train_enable": False, **(meta or {})}
    if isinstance(config.backend, CortexConfig):
        model_keys = {"input_ids", "attention_mask", "position_ids", "use_cache"}
        kwargs = {key: value for key, value in batch.items() if key in model_keys}
        return {
            "args": (),
            "kwargs": kwargs,
            "context": dict(batch),
            "processing": descriptor,
        }
    return {"batch": dict(batch), "meta": meta_payload, "processing": descriptor}


class ArcticOPDClient:
    """Student train+sample plus a sampling-only teacher, each a ``SyncArcticRLClient``.

    The teacher has ``training_gpus=0``; only ``generate`` is used. Weight sync
    runs on the student (train → student sampler) and never targets the teacher.
    """

    def __init__(self, config: ArcticOPDClientConfig) -> None:
        self.config = config
        self.student = SyncArcticRLClient(config.student_transport_config())
        try:
            self.teacher = SyncArcticRLClient(config.teacher_transport_config())
        except Exception:
            self.student.shutdown()
            raise

    @property
    def student_jobs(self):
        return self.student.jobs

    @property
    def teacher_jobs(self):
        return self.teacher.jobs

    def generate(
        self,
        prompts: list,
        sampling_params: dict | None = None,
        routing_key: Any = None,
        strict: bool = False,
    ) -> list:
        return self.student.generate(prompts, sampling_params, routing_key, strict)

    def generate_teacher(
        self,
        prompts: list,
        sampling_params: dict | None = None,
        routing_key: Any = None,
        strict: bool = False,
    ) -> list:
        return self.teacher.generate(prompts, sampling_params, routing_key, strict)

    def fwd_bwd(self, batch: dict, processing: dict | None = None, meta: dict | None = None) -> dict:
        return self.student.fwd_bwd(_fwd_bwd_body(self.config, batch, processing, meta))

    def step(self, learning_rate: float | None = None) -> dict:
        return self.student.step(learning_rate)

    def sync_weights(self, cuda_ipc: bool | None = None, low_memory: bool | None = None) -> dict:
        """Student train → student sampler. Cortex has no wake-inference op."""
        if isinstance(self.config.backend, CortexConfig):
            tid = self.student.jobs.require("training")
            sid = self.student.jobs.require("sampling")
            payload: dict[str, Any] = {
                "source_sub_job_id": tid,
                "target_sub_job_ids": [sid],
                "weight_format": "hf",
            }
            if cuda_ipc is not None:
                payload["cuda_ipc"] = cuda_ipc
            if low_memory is not None:
                payload["low_memory"] = low_memory
            out = self.student.transport.call(
                Request(
                    "operation",
                    tid,
                    {
                        "operation_type": "weight-sync",
                        "sub_job_id": tid,
                        "sub_job_type": "training",
                        "payload": payload,
                    },
                )
            )
            self.student.reset_prefix_cache()
            return out
        return self.student.sync_weights(cuda_ipc=cuda_ipc, low_memory=low_memory)

    def reset_student_prefix_cache(
        self, drain: bool = True, timeout_s: float = 60.0, retry_interval_s: float = 0.1
    ) -> dict:
        return self.student.reset_prefix_cache(drain, timeout_s, retry_interval_s)

    def wake_student_inference(self, tags: list | None = None) -> dict:
        return self.student.wake_inference(tags)

    def save_checkpoint(
        self,
        checkpoint_id: str | None = None,
        checkpoint_type: str = "weights-only",
        path: str | None = None,
        *,
        step: int | None = None,
        export_hf: bool = False,
        save_total_limit: int | None = None,
    ) -> dict:
        return self.student.save_checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            path=path,
            step=step,
            export_hf=export_hf,
            save_total_limit=save_total_limit,
        )

    def reconnect_config(self) -> ArcticOPDClientConfig:
        return self.config.model_copy(
            update={
                "training_job_id": self.student.jobs.training,
                "sampling_job_id": self.student.jobs.sampling,
                "teacher_job_id": self.teacher.jobs.sampling,
            }
        )

    def shutdown(self) -> None:
        teacher_error: Exception | None = None
        try:
            self.teacher.shutdown()
        except Exception as exc:  # teardown must still reach the student
            teacher_error = exc
        try:
            self.student.shutdown()
        finally:
            if teacher_error is not None:
                raise teacher_error

    def __enter__(self) -> ArcticOPDClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def create_arctic_opd_client(config: ArcticOPDClientConfig) -> ArcticOPDClient:
    return ArcticOPDClient(config)
