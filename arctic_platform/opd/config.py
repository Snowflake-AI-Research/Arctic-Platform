# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the first-class on-policy distillation client."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.config import JobId
from arctic_platform.client.config import OnPremConfig
from arctic_platform.client.config import SamplingConfig
from arctic_platform.client.config import TrainingConfig


def _local_cluster_env(http_port: int) -> dict[str, str]:
    """Disjoint Ray / DeepSpeed ports derived from the HTTP listen port.

    Two HTTP servers on one host cannot share Ray GCS, worker-port ranges, or
    the DeepSpeed rendezvous port.
    """
    slot = http_port % 10
    worker_base = 40000 + slot * 10000
    return {
        "PYTHONUNBUFFERED": "1",
        "RAY_PORT": str(25000 + slot),
        "RAY_DASHBOARD_PORT": str(26000 + slot),
        "RAY_CLIENT_SERVER_PORT": str(10010 + slot),
        "RAY_DASHBOARD_AGENT_LISTEN_PORT": str(23000 + slot),
        "MASTER_PORT": str(27000 + slot),
        "ARL_WEIGHT_SYNC_PORT": str(28000 + slot),
        "ARL_RAY_MIN_WORKER_PORT": str(worker_base),
        "ARL_RAY_MAX_WORKER_PORT": str(worker_base + 9999),
    }


class ArcticOPDClientConfig(BaseModel):
    """Student training/sampling plus a fixed teacher sampling engine.

    ``sampling_gpus`` follows the ArcticRLClient convention and belongs to the
    student rollout engine. ``teacher_sampling_gpus`` is the teacher scorer.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    student_model: str
    teacher_model: str
    seed: int | None = None
    dtype: str | None = None
    max_seq_len: int = 8192

    training_gpus: int = Field(..., gt=0)
    sampling_gpus: int = Field(..., gt=0)
    teacher_sampling_gpus: int = Field(..., gt=0)

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    teacher_sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    backend: OnPremConfig | CortexConfig = Field(default_factory=OnPremConfig, discriminator="type")

    # A local OPD deployment uses a second server for the teacher because the
    # current on-prem server owns one sampling pool. Cortex similarly needs a
    # second parent job because generate has no sub-job selector.
    teacher_port: int | None = None
    teacher_server_cuda_visible_devices: str | None = None

    job_ready_timeout: float = 1800.0
    request_timeout: float = 1800.0

    training_job_id: JobId | None = None
    sampling_job_id: JobId | None = None
    teacher_job_id: JobId | None = None

    @model_validator(mode="after")
    def _validate_reconnect(self) -> ArcticOPDClientConfig:
        ids = (self.training_job_id, self.sampling_job_id, self.teacher_job_id)
        if any(job_id is not None for job_id in ids) and not all(job_id is not None for job_id in ids):
            raise ValueError("OPD reconnect requires training_job_id, sampling_job_id, and teacher_job_id together")
        if isinstance(self.backend, OnPremConfig) and self.backend.launch_local_server:
            teacher_port = self.teacher_port or self.backend.port + 1
            if teacher_port == self.backend.port:
                raise ValueError("teacher_port must differ from the student server port")
            student_devices = self.backend.server_cuda_visible_devices
            teacher_devices = self.teacher_server_cuda_visible_devices
            if student_devices is None or teacher_devices is None:
                raise ValueError(
                    "local OPD launch requires server_cuda_visible_devices and "
                    "teacher_server_cuda_visible_devices"
                )
            if set(student_devices.split(",")) & set(teacher_devices.split(",")):
                raise ValueError("student and teacher server CUDA device lists must be disjoint")
        return self

    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dtype": self.dtype,
            "max_seq_len": self.max_seq_len,
            "job_ready_timeout": self.job_ready_timeout,
            "request_timeout": self.request_timeout,
        }

    def student_transport_config(self) -> ArcticRLClientConfig:
        """Internal adapter for the shared transport; not an ArcticRLClient."""
        backend = self.backend
        if isinstance(backend, OnPremConfig) and backend.launch_local_server:
            backend = backend.model_copy(
                update={
                    "server_extra_env": {
                        **_local_cluster_env(backend.port),
                        **backend.server_extra_env,
                    }
                }
            )
        return ArcticRLClientConfig(
            model_name=self.student_model,
            training_gpus=self.training_gpus,
            sampling_gpus=self.sampling_gpus,
            log_prob_gpus=0,
            training=self.training,
            sampling=self.sampling,
            backend=backend,
            training_job_id=self.training_job_id,
            sampling_job_id=self.sampling_job_id,
            **self._base_kwargs(),
        )

    def teacher_transport_config(self) -> ArcticRLClientConfig:
        """Internal sampling-only transport config for the fixed teacher."""
        backend = self.backend
        if isinstance(backend, OnPremConfig):
            teacher_port = self.teacher_port or backend.port + 1
            extra_env = dict(backend.server_extra_env)
            if backend.launch_local_server:
                extra_env = {**_local_cluster_env(teacher_port), **backend.server_extra_env}
            backend = backend.model_copy(
                update={
                    "port": teacher_port,
                    "colocate": False,
                    "server_cuda_visible_devices": (
                        self.teacher_server_cuda_visible_devices
                        if self.teacher_server_cuda_visible_devices is not None
                        else backend.server_cuda_visible_devices
                    ),
                    "server_extra_env": extra_env,
                }
            )
        return ArcticRLClientConfig(
            model_name=self.teacher_model,
            training_gpus=0,
            sampling_gpus=self.teacher_sampling_gpus,
            log_prob_gpus=0,
            sampling=self.teacher_sampling,
            backend=backend,
            sampling_job_id=self.teacher_job_id,
            **self._base_kwargs(),
        )
