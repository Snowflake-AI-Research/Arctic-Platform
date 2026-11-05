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
"""Shared configuration validation for clients and model code."""

from __future__ import annotations

from numbers import Integral
from typing import Any
from typing import Mapping

_DS_WORKER_SP_PATHS: tuple[tuple[str, ...], ...] = (
    ("ds_worker_config", "sp_size"),
    ("ds_worker_config", "sequence_parallel"),
    ("ds_worker_config", "sequence_parallel_size"),
    ("ds_worker_config", "parallelism", "sequence_parallel"),
)
_TRAINING_SP_PATHS: tuple[tuple[str, ...], ...] = (
    ("training_config", "sp_size"),
    ("ds_config", "sequence_parallel_size"),
)
_LOG_PROB_SP_PATH: tuple[str, ...] = ("log_prob_config", "sequence_parallel_size")
_DS_CONFIG_SP_PATH: tuple[str, ...] = ("ds_config", "sequence_parallel_size")

SP_SIZE_PATHS: tuple[tuple[str, ...], ...] = _TRAINING_SP_PATHS + (_LOG_PROB_SP_PATH,) + _DS_WORKER_SP_PATHS


def _nested(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = mapping
    for part in path:
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


def _sp_size_paths_for_job(job_config: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Log-prob jobs use their explicit SP degree instead of the training degree."""
    job_type = job_config.get("job_type") or "training"
    if job_type == "log_prob":
        if _nested(job_config, _LOG_PROB_SP_PATH) is not None:
            return (_LOG_PROB_SP_PATH,) + _DS_WORKER_SP_PATHS
        return (_DS_CONFIG_SP_PATH, _LOG_PROB_SP_PATH) + _DS_WORKER_SP_PATHS
    if job_type == "sampling":
        return _DS_WORKER_SP_PATHS
    return _TRAINING_SP_PATHS + _DS_WORKER_SP_PATHS


def resolve_parallelism_degree(value: Any, name: str) -> int:
    """Unset means disabled; supplied degrees must be non-bool integers >= 1."""
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= 1 (1 = disabled); got {value!r}")
    parallelism_degree = int(value)
    if parallelism_degree < 1:
        raise ValueError(f"{name} must be an integer >= 1 (1 = disabled); got {value!r}")
    return parallelism_degree


def resolve_sp_size(job_config: Mapping[str, Any]) -> int:
    """Resolve live sequence-parallel settings and reject conflicting aliases."""
    candidates: dict[str, int] = {}
    for path in _sp_size_paths_for_job(job_config):
        node = _nested(job_config, path)
        if node is not None:
            candidates[".".join(path)] = resolve_parallelism_degree(node, path[-1])
    values = set(candidates.values())
    if len(values) > 1:
        raise ValueError(f"conflicting sequence-parallel degrees: {candidates}")
    return values.pop() if values else 1


def validate_peft_config(peft_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """None disables PEFT; a supplied config must name its adapter type."""
    if peft_config is not None:
        if not isinstance(peft_config, dict):
            raise ValueError("peft_config must be a dict or None")
        peft_type = peft_config.get("peft_type")
        if not isinstance(peft_type, str) or not peft_type.strip():
            raise ValueError("peft_config.peft_type must be a non-empty string; use None to disable PEFT")
    return peft_config
