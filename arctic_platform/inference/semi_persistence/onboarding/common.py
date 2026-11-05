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

"""Helpers shared by the onboarding checks: generate calls, timing, log text."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

DONE_STATES = ("completed", "done", "succeeded")


@dataclass
class Pair:
    """The two running jobs being compared, plus the facts the checks need."""

    on: str
    off: str
    replicas: int
    max_seq_len: int


def parse_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def generate(
    client: Any,
    job_id: str,
    prompts: list,
    sampling_params: dict,
    routing_key: str | None = None,
) -> list[dict]:
    """One generate request; returns its results, or one ``{"error": ...}`` per prompt."""
    try:
        request_id = client.generate(
            job_id, prompts=prompts, sampling_params=sampling_params, routing_key=routing_key
        )
        results = client.poll_request(job_id, request_id).get("results") or []
    except Exception as exc:  # noqa: BLE001 - a failed request is a check result, not a crash
        return [{"error": f"{type(exc).__name__}: {exc}"} for _ in prompts]
    if len(results) != len(prompts):
        return [{"error": f"asked for {len(prompts)} results, got {len(results)}"} for _ in prompts]
    return [r if isinstance(r, dict) else {"text": str(r)} for r in results]


def timed_generate(client: Any, job_id: str, prompt: list[int], sampling_params: dict) -> dict:
    """One single-prompt request with server-side and client-side latency.

    Server latency is the request's ``updated_at - created_at``; client latency
    also covers submission and polling, so the two differ by client overhead.
    """
    t0 = time.time()
    request_id = client.generate(job_id, prompts=[prompt], sampling_params=sampling_params)
    result = client.poll_request(job_id, request_id)["results"][0]
    client_s = time.time() - t0
    status = client.get_request_status(job_id, request_id)
    return {
        "server_s": parse_ts(status["updated_at"]) - parse_ts(status["created_at"]),
        "client_s": client_s,
        "gen": int(result.get("generation_len", 0)),
        "prompt": int(result.get("prompt_len", 0)),
        "zero_ids": bool(result.get("token_ids")) and all(t == 0 for t in result["token_ids"]),
    }


def prompt_ids(length: int, salt: int) -> list[int]:
    """Ordinary-word token ids, varied by ``salt`` so prefix caching cannot help."""
    return [1000 + (salt * 7919 + i * 104729) % 20000 for i in range(length)]


def log_text(entry: Any) -> str:
    """The text of one log entry as returned by ``tail_logs``."""
    if not isinstance(entry, dict):
        return str(entry)
    for key in ("_raw", "msg", "message"):
        if entry.get(key) is not None:
            return str(entry[key])
    return json.dumps(entry, sort_keys=True)


def parse_kv_args(argv: list[str]) -> dict[str, str]:
    """``key=value`` command-line arguments."""
    args: dict[str, str] = {}
    for item in argv:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"expected key=value, got {item!r}")
        args[key] = value
    return args


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"expected a boolean, got {value!r}")
