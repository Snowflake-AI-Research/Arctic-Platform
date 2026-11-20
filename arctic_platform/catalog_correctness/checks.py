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

"""Catalog-based comparisons using the Cortex Training API."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
import time
from pathlib import Path
from typing import Any


def connect(args):
    from cortex_training import CortexTrainingClient

    if args.host:
        if not args.database or not args.schema or not os.environ.get(args.pat_env):
            raise ValueError("--host requires --database, --schema and a PAT in the selected environment variable")
        return CortexTrainingClient.from_pat(
            host=args.host,
            pat=os.environ[args.pat_env],
            database=args.database,
            schema=args.schema,
            poll_timeout=args.poll_timeout,
        )
    return CortexTrainingClient.from_configs(
        connection=args.connection,
        database=args.database,
        schema=args.schema,
        poll_timeout=args.poll_timeout,
    )


def read_recipe(path: Path, kind: str) -> tuple[dict, str]:
    import json

    raw = path.read_bytes()
    recipe = json.loads(raw)
    subs = recipe.get("sub_job_configs")
    if not isinstance(subs, list) or len(subs) != 1 or subs[0].get("job_type") != kind:
        raise ValueError(f"{path}: expected one {kind} sub-job")
    return recipe, hashlib.sha256(raw).hexdigest()


def validate_recipes(training: dict, sampling: dict) -> None:
    train, sample = training["sub_job_configs"][0], sampling["sub_job_configs"][0]
    if train.get("model_name") != sample.get("model_name"):
        raise ValueError("training and sampling recipes must name the same model")
    if training.get("hardware") != sampling.get("hardware"):
        raise ValueError("training and sampling recipes must use the same hardware")
    train_peft = (train.get("training_config") or {}).get("peft_config")
    sample_peft = (sample.get("inference_config") or {}).get("peft_config")
    if train_peft or sample_peft:
        raise ValueError("this suite requires full-parameter training and sampling recipes")


def batches(tokenizer: Any, rows: int, width: int, count: int, seed: int) -> list[list[list[int]]]:
    if rows < 1 or width < 2 or count < 1:
        raise ValueError("batch size and count must be positive and sequence length at least 2")
    special = set(tokenizer.all_special_ids)
    vocabulary = sorted({value for value in tokenizer.get_vocab().values() if value not in special and value >= 0})
    if not vocabulary:
        raise ValueError("tokenizer has no non-special tokens")
    randomizer = random.Random(seed)
    return [
        [
            randomizer.choices(vocabulary, k=width - (row % 4) * min(8, max(0, (width - 16) // 4)))
            for row in range(rows)
        ]
        for _ in range(count)
    ]


def loss(logprobs: list[list[float]]) -> float:
    values = [float(value) for row in logprobs for value in row]
    if not values or any(not math.isfinite(value) or value > 1e-6 for value in values):
        raise ValueError("log probabilities must be finite, nonpositive and nonempty")
    return -sum(values) / len(values)


def train_step(
    client, job: str, tokens: list[list[int]], *, expected_model_calls: int | None = None
) -> tuple[float, dict]:
    import torch
    from cortex_training.client import custom_loss

    n_tokens = sum(len(row) - 1 for row in tokens)
    gradients = [torch.full((len(row) - 1,), -1.0 / n_tokens) for row in tokens]
    payload = custom_loss.build_custom_loss_backward_payload(tokens, gradients)
    result = client.poll_request(job, client.forward_backward(job, payload))
    if (
        expected_model_calls is not None
        and (result.get("metrics") or {}).get("rl_model_calls") != expected_model_calls
    ):
        raise ValueError("forward-backward did not report the expected number of model calls")
    logprobs = custom_loss.logprobs_from_backward_result(result, tokens)
    if logprobs is None:
        raise ValueError("forward-backward returned no per-token log probabilities")
    training_loss = loss([row.tolist() for row in logprobs])
    step_result = client.poll_request(job, client.step(job))
    step_number(step_result)
    metrics = step_result.get("metrics") or {}
    updated = step_result.get("update_successful", metrics.get("update_successful"))
    if updated is not None and (not isinstance(updated, (int, float)) or not math.isfinite(updated) or updated != 1):
        raise ValueError("optimizer step did not report a successful update")
    return training_loss, step_result


def step_number(result: dict) -> int:
    metrics = result.get("metrics") or {}
    value = result.get("global_steps", metrics.get("global_steps"))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("optimizer step did not report a finite global step")
    if value < 1 or value != int(value):
        raise ValueError("optimizer step reported an invalid global step")
    return int(value)


def training_logprobs(client, job: str, tokens: list[list[int]]) -> list[list[float]]:
    from cortex_training.client import custom_loss

    payload = custom_loss.build_forward_logprobs_payload(tokens)
    submitted = client.forward(job, payload, sub_job_type="training")
    result = client.poll_request(job, submitted["request_id"])
    return [row.tolist() for row in custom_loss.logprobs_from_forward_result(result, tokens)]


def sampling_logprobs(client, job: str, tokens: list[list[int]]) -> list[list[float]]:
    request = client.generate(
        job,
        prompts=tokens,
        sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
        sub_job_id=f"{job}:sampling:0",
    )
    results = client.poll_request(job, request)["results"]
    if len(results) != len(tokens):
        raise ValueError("sampler returned a different number of sequences")
    output = []
    for row, result in zip(tokens, results, strict=True):
        positions = result.get("prompt_logprobs")
        if not isinstance(positions, list) or len(positions) != len(row) or positions[0] is not None:
            raise ValueError("sampler did not return one aligned log probability per prompt token")
        values = []
        for token, position in zip(row[1:], positions[1:], strict=True):
            if not isinstance(position, dict):
                raise ValueError("sampler log probability entry is not a token mapping")
            entry = position.get(str(token), position.get(token))
            if isinstance(entry, dict):
                entry = entry.get("logprob")
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise ValueError("sampler omitted the requested token's log probability")
            values.append(float(entry))
        output.append(values)
    loss(output)
    return output


def compare_logprobs(
    target: list[list[float]],
    reference: list[list[float]],
    tolerance: float | None,
    token_tolerance: float | None = None,
) -> dict:
    if len(target) != len(reference) or any(len(a) != len(b) for a, b in zip(target, reference, strict=True)):
        raise ValueError("training and sampling token coverage differs")
    reference_loss, target_loss = loss(reference), loss(target)
    differences = [abs(a - b) for a, b in zip(sum(target, []), sum(reference, []), strict=True)]
    delta = abs(target_loss - reference_loss)
    loss_status = verdict(delta, tolerance)
    token_status = verdict(max(differences), token_tolerance)
    status = (
        "fail"
        if "fail" in (loss_status, token_status)
        else ("pass" if loss_status == token_status == "pass" else "measured")
    )
    return {
        "status": status,
        "training_loss": reference_loss,
        "sampling_loss": target_loss,
        "loss_abs_diff": delta,
        "mean_token_abs_diff": sum(differences) / len(differences),
        "max_token_abs_diff": max(differences),
        "tokens": len(differences),
        "absolute_tolerance": tolerance,
        "token_absolute_tolerance": token_tolerance,
    }


def verdict(difference: float, tolerance: float | None) -> str:
    if not math.isfinite(difference):
        raise ValueError("comparison produced a non-finite difference")
    if tolerance is None:
        return "measured"
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    return "pass" if difference <= tolerance else "fail"


def changed_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        paths = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(changed_paths(before[key], after[key], path))
        return paths
    if isinstance(before, list) and isinstance(after, list):
        paths = []
        for index in range(max(len(before), len(after))):
            path = f"{prefix}[{index}]"
            if index >= len(before) or index >= len(after):
                paths.append(path)
            else:
                paths.extend(changed_paths(before[index], after[index], path))
        return paths
    return [prefix] if before != after else []


class Jobs:
    def __init__(self, client, record):
        self.client = client
        self.record = record
        self.jobs: list[str] = []
        self.active: list[str] = []
        self.runtime: dict[str, dict] = {}

    def start(self, body: dict) -> str:
        import json

        request = copy.deepcopy(body)
        original = copy.deepcopy(request)
        created = self.client.create_job_from_body(request)
        job = created.get("job_id")
        if not isinstance(job, str) or not job:
            raise RuntimeError("service did not return a job id")
        self.jobs.append(job)
        self.active.append(job)
        submitted = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.runtime[job] = {
            "submitted_config_sha256": hashlib.sha256(submitted).hexdigest(),
            "client_normalized_fields": changed_paths(original, request),
        }
        self.record()
        print("job submitted; waiting for workers", flush=True)
        running = self.client.wait_for_job(job)
        self.runtime[job]["image_tag"] = running.get("image_tag")
        self.record()
        return job

    def stop(self, job: str) -> None:
        print("canceling test job", flush=True)
        try:
            self.client.cancel_job(job)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                status = str(self.client.get_job(job).get("status", "")).lower()
                if status in {"cancelled", "canceled", "terminated", "failed", "done", "completed", "succeeded"}:
                    break
                time.sleep(2)
            else:
                raise TimeoutError("job did not terminate after cancellation")
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) != 404:
                raise
        self.active.remove(job)
        self.record()

    def cleanup(self) -> list[str]:
        errors = []
        for job in reversed(self.active[:]):
            try:
                self.stop(job)
            except Exception as exc:
                errors.append(f"{job}: {type(exc).__name__}: {exc}")
        return errors


def checkpoint_id(result: dict) -> str:
    import re

    value = result.get("checkpoint_id")
    if isinstance(value, str) and value:
        return value
    match = re.search(r"/checkpoints/(cp_[0-9a-fA-F-]+)(?:/|$)", str(result.get("stage_path", "")))
    if match:
        return match.group(1)
    raise ValueError("save did not return a durable checkpoint identifier")
