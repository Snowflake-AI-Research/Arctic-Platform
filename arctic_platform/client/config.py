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
"""One config for every backend. Switching deployment == changing `backend`."""

from __future__ import annotations

import warnings
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

JobId = int | str

# Legacy backend labels accepted for compatibility with the ArcticTraining and
# older arctic_platform clients that SkyRL / verl integrations were built
# against. They collapse into the two canonical labels below at validation
# time; downstream code only ever sees `onprem` or `cortex`.
_BACKEND_ALIASES = {
    "local": "onprem",
    "dss-platform": "onprem",
    "dss_platform": "onprem",
    "neutrino": "cortex",
}

# Legacy field names accepted for compatibility. Mapped into the canonical
# field before pydantic validation so downstream code only sees the canonical
# name. Kept as (legacy_name, canonical_name) pairs.
_FIELD_ALIASES = (
    ("sample_gpus", "sampling_gpus"),
    ("sampling_engine", "vllm_config"),  # legacy verl passes engine name; ignored below
)

# Legacy fields accepted-and-ignored. These exist on old ArcticTraining /
# dss-client configs but have no analogue on the unified client. Silently
# dropping them keeps SkyRL / verl integrations config-compatible.
_LEGACY_IGNORED_FIELDS = frozenset({
    "log_prob_engine",
    "sampling_engine",
    "reference_model",
    "job_name",
    "experiment_name",
})


class ArcticRLClientConfig(BaseModel):
    # NOTE: extra="ignore" (was "forbid") so legacy fields from ArcticTraining /
    # verl configs pass through without erroring. Legacy fields we care about
    # are mapped into canonical fields by `_apply_legacy_aliases` below; the
    # rest are dropped silently.
    model_config = ConfigDict(extra="ignore", validate_default=True)

    backend: Literal["onprem", "cortex"] = Field("onprem", description="Deployment target.")
    comm_protocol: Literal["http", "ray"] = Field("http", description="onprem transport: HTTP or in-process Ray.")
    model_name: str = Field(..., description="HF model id to train/serve.")
    seed: int | None = Field(None, description="Global seed.")
    dtype: str | None = Field(None, description="Parameter dtype override.")
    max_seq_len: int = Field(8192, description="Max sequence length (cortex sub-jobs).")

    # GPU allocation — a job type is created only when its count is > 0.
    training_gpus: int = 0
    sampling_gpus: int = 0
    log_prob_gpus: int = 0

    # onprem (HTTP + Ray)
    host: str = "localhost"
    port: int = 8000
    colocate: bool = False
    launch_local_server: bool = Field(False, description="onprem: spawn a local server before connecting.")
    startup_timeout: float = 600.0
    job_ready_timeout: float = 1800.0
    request_timeout: float = Field(
        1800.0, description="onprem HTTP: per-request timeout (seconds) applied to every call. Generous for long ops."
    )
    ds_config: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    vllm_config: dict[str, Any] | None = None
    checkpoint_path: str | None = Field(
        None,
        description=(
            "onprem: training job's base checkpoint dir, set at init (resume-from + weight sync). "
            "This is the default destination; a per-call save_checkpoint(path=...) overrides it for "
            "that call and falls back to this dir when path is None."
        ),
    )

    # cortex (SnowAPI)
    cortex_base_url: str | None = Field(None, description="Mock/direct GS URL; bypasses PAT auth.")
    cortex_host: str | None = None
    cortex_pat_env_var: str = "CORTEX_PAT"
    cortex_database: str = ""
    cortex_schema: str = ""
    cortex_endpoint: str = "cortex-training"

    # Reconnect: attach to pre-existing jobs instead of creating new ones.
    training_job_id: JobId | None = None
    sampling_job_id: JobId | None = None
    log_prob_job_id: JobId | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_aliases(cls, data: Any) -> Any:
        """Map legacy backend labels and field names to canonical ones.

        Runs before field validation so SkyRL and verl configs authored against
        the old ArcticTraining / dss-client contract validate unchanged:

        - ``backend="local"`` / ``"dss-platform"`` -> ``"onprem"``
        - ``backend="neutrino"``                    -> ``"cortex"``
        - ``sample_gpus=`` alias for ``sampling_gpus=``
        - ``log_prob_engine``/``sampling_engine`` are dropped (no analogue).
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        raw = data.get("backend")
        if isinstance(raw, str) and raw in _BACKEND_ALIASES:
            canonical = _BACKEND_ALIASES[raw]
            warnings.warn(
                f"ArcticRLClientConfig(backend={raw!r}) is a legacy alias for "
                f"{canonical!r}; migrate to {canonical!r}.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["backend"] = canonical
        for legacy, canonical in _FIELD_ALIASES:
            if legacy in data and data.get(canonical) in (None, 0, "", {}):
                if canonical in {name for name, _ in _FIELD_ALIASES}:
                    continue
                data[canonical] = data.pop(legacy)
            elif legacy in data:
                data.pop(legacy)
        for name in list(data):
            if name in _LEGACY_IGNORED_FIELDS:
                data.pop(name)
        return data

    def gpus_for(self, job_type: str) -> int:
        """GPU count allocated to a job type (0 == the job type is disabled)."""
        return getattr(self, f"{job_type}_gpus")
