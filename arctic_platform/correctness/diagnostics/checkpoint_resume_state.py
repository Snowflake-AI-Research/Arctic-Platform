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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Locate the first inexact component between two uninterrupted training trajectories.

Each trajectory is one fresh job on the same batches. Comparison starts at iteration 1 and stops at the
first inexact pre-step parameter, loss, or gradient. A one-step pair runs first; an exact one-step pair
continues through the checkpoint iteration plus one.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from arctic_platform.correctness.checks.checkpoint_resume import CHECKPOINT_ITERATION
from arctic_platform.correctness.checks.checkpoint_resume import ITERATIONS
from arctic_platform.correctness.checks.checkpoint_resume import LEARNING_RATE
from arctic_platform.correctness.harness.batches import build_batch
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step
from arctic_platform.correctness.harness.dss_driver import gateway
from arctic_platform.correctness.harness.dss_driver import pack
from arctic_platform.correctness.harness.dss_driver import running_job
from arctic_platform.correctness.harness.optimizer_capture import optimizer_capture_worker
from arctic_platform.correctness.harness.seeds import SEED
from arctic_platform.correctness.harness.spec import TestSpec
from arctic_platform.correctness.harness.workdir import correctness_workdir

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-lora-8gpus-sp8-64k.config"
SPEC = ROOT / "arctic_platform/correctness/specs/qwen3.6-35b-a3b-h200-train-sft-lora-8gpus-sp8-64k.json"
TENSOR_COMPONENTS = (
    ("lora_parameters", "parameter"),
    ("fp32_masters", "fp32_master"),
    ("adam_exp_avg_before", "exp_avg_before"),
    ("adam_exp_avg_sq_before", "exp_avg_sq_before"),
    ("gradients", "gradient"),
    ("parameter_updates", "parameter_update"),
    ("adam_exp_avg_after", "exp_avg"),
    ("adam_exp_avg_sq_after", "exp_avg_sq"),
)


@dataclass(frozen=True)
class Comparison:
    component: str
    exact: bool
    max_abs: float
    l2: float
    worst: str


def _snapshot(capture: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(capture, destination)


def _run_pair(limit: int, work: Path, cfg, spec, arm, bodies: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
    losses = {"uninterrupted": {}, "resumed": {}}
    runtime = work / "runtime"
    with optimizer_capture_worker(), gateway(runtime, cfg.n_gpus) as session:
        capture = work / "capture-uninterrupted"
        payload = build_payload(
            cfg.training,
            spec.model.cache_path,
            SEED,
            attn_implementation=cfg.attention_implementation,
            optimizer_state_output_dir=capture,
        )
        with running_job(session, payload) as job:
            for iteration, body in enumerate(bodies[:limit], start=1):
                result = fwd_bwd_step(session, job, body, learning_rate=LEARNING_RATE)
                if iteration > CHECKPOINT_ITERATION:
                    losses["uninterrupted"][iteration] = result.avg_loss
                    _snapshot(capture, work / "uninterrupted" / f"step-{iteration}")

        first_capture = work / "capture-before-checkpoint"
        first_payload = build_payload(
            cfg.training,
            spec.model.cache_path,
            SEED,
            attn_implementation=cfg.attention_implementation,
            optimizer_state_output_dir=first_capture,
        )
        with running_job(session, first_payload) as first_job:
            for body in bodies[:CHECKPOINT_ITERATION]:
                fwd_bwd_step(session, first_job, body, learning_rate=LEARNING_RATE)
            checkpoint = first_job.client.save_checkpoint()["path"]

        resumed_capture = work / "capture-resumed"
        resumed_payload = build_payload(
            cfg.training,
            spec.model.cache_path,
            SEED,
            attn_implementation=cfg.attention_implementation,
            optimizer_state_output_dir=resumed_capture,
        )
        with running_job(session, resumed_payload) as resumed_job:
            restored_step = int(resumed_job.client.load_checkpoint(path=checkpoint)["global_step"])
            print(f"RESTORE requested_step={CHECKPOINT_ITERATION} restored_step={restored_step}", flush=True)
            for iteration, body in enumerate(bodies[CHECKPOINT_ITERATION:limit], start=CHECKPOINT_ITERATION + 1):
                result = fwd_bwd_step(session, resumed_job, body, learning_rate=LEARNING_RATE)
                losses["resumed"][iteration] = result.avg_loss
                _snapshot(resumed_capture, work / "resumed" / f"step-{iteration}")
    return losses["uninterrupted"], losses["resumed"]


def _manifest(snapshot: Path) -> dict:
    return json.loads((snapshot / "manifest.json").read_text())


def _entries(snapshot: Path, manifest: dict) -> dict[str, Path]:
    entries = {item["name"]: snapshot / item["file"] for item in manifest["parameters"] if "lora_" in item["name"]}
    if not entries:
        raise RuntimeError(f"checkpoint probe found no LoRA parameters in {snapshot}")
    return entries


def _compare_tensor_component(left: Path, right: Path, component: str, field: str) -> Comparison:
    left_manifest = _manifest(left)
    right_manifest = _manifest(right)
    left_entries = _entries(left, left_manifest)
    right_entries = _entries(right, right_manifest)
    if set(left_entries) != set(right_entries):
        missing = sorted(set(left_entries) ^ set(right_entries))
        return Comparison(component, False, float("inf"), float("inf"), f"parameter-set:{missing[0]}")
    exact = True
    max_abs = 0.0
    squared = 0.0
    worst = ""
    for name in sorted(left_entries):
        left_value = torch.load(left_entries[name], map_location="cpu", weights_only=True)[field]
        right_value = torch.load(right_entries[name], map_location="cpu", weights_only=True)[field]
        if left_value is None or right_value is None:
            same = left_value is None and right_value is None
            exact = exact and same
            if not same and not worst:
                worst = name
            continue
        if left_value.shape != right_value.shape:
            return Comparison(component, False, float("inf"), float("inf"), f"shape:{name}")
        delta = left_value.float() - right_value.float()
        local_max = float(delta.abs().max()) if delta.numel() else 0.0
        local_squared = float(torch.sum(delta.double().square()))
        if local_max > max_abs:
            max_abs = local_max
            worst = name
        squared += local_squared
        exact = exact and torch.equal(left_value, right_value)
    return Comparison(component, exact, max_abs, squared**0.5, worst or "none")


def _compare_metadata(left: dict, right: dict, section: str, key: str, component: str) -> Comparison:
    left_value = left[section][key]
    right_value = right[section][key]
    exact = left_value == right_value
    return Comparison(
        component,
        exact,
        0.0 if exact else float("inf"),
        0.0 if exact else float("inf"),
        str((left_value, right_value)),
    )


def _compare_iteration(
    work: Path,
    iteration: int,
    left_loss: float,
    right_loss: float,
    left_name: str = "uninterrupted",
    right_name: str = "resumed",
) -> list[Comparison]:
    left = work / left_name / f"step-{iteration}"
    right = work / right_name / f"step-{iteration}"
    left_manifest = _manifest(left)
    right_manifest = _manifest(right)
    comparisons = [
        _compare_tensor_component(left, right, "lora_parameters", "parameter"),
        _compare_tensor_component(left, right, "fp32_masters", "fp32_master"),
        _compare_tensor_component(left, right, "adam_exp_avg_before", "exp_avg_before"),
        _compare_tensor_component(left, right, "adam_exp_avg_sq_before", "exp_avg_sq_before"),
        _compare_metadata(left_manifest, right_manifest, "before", "optimizer_steps", "adam_step_before"),
        _compare_metadata(left_manifest, right_manifest, "before", "engine_global_step", "engine_global_step_before"),
        _compare_metadata(left_manifest, right_manifest, "before", "learning_rates", "learning_rates_before"),
        Comparison(
            "batch_loss", left_loss == right_loss, abs(left_loss - right_loss), abs(left_loss - right_loss), "loss"
        ),
        _compare_tensor_component(left, right, "gradients", "gradient"),
        _compare_tensor_component(left, right, "parameter_updates", "parameter_update"),
        _compare_tensor_component(left, right, "adam_exp_avg_after", "exp_avg"),
        _compare_tensor_component(left, right, "adam_exp_avg_sq_after", "exp_avg_sq"),
        _compare_metadata(left_manifest, right_manifest, "after", "optimizer_steps", "adam_step_after"),
        _compare_metadata(left_manifest, right_manifest, "after", "engine_global_step", "engine_global_step_after"),
        _compare_metadata(left_manifest, right_manifest, "after", "learning_rates", "learning_rates_after"),
    ]
    for item in comparisons:
        print(
            f"COMPARE iteration={iteration} component={item.component} exact={item.exact} "
            f"max_abs={item.max_abs:.17g} l2={item.l2:.17g} worst={item.worst}",
            flush=True,
        )
    return comparisons


def _first_difference(
    work: Path, uninterrupted: dict[int, float], resumed: dict[int, float]
) -> tuple[int, Comparison] | None:
    for iteration in sorted(uninterrupted):
        for comparison in _compare_iteration(work, iteration, uninterrupted[iteration], resumed[iteration]):
            if not comparison.exact:
                return iteration, comparison
    return None


def _train_uninterrupted(
    session, label: str, limit: int, work: Path, cfg, spec, bodies: list[dict]
) -> dict[int, float]:
    """Train one fresh job and keep each step's pre-update snapshot."""
    print(f"TRAJECTORY label={label} limit={limit}", flush=True)
    losses: dict[int, float] = {}
    capture = work / f"capture-{label}"
    payload = build_payload(
        cfg.training,
        spec.model.cache_path,
        SEED,
        attn_implementation=cfg.attention_implementation,
        optimizer_state_output_dir=capture,
    )
    with running_job(session, payload) as job:
        for iteration, body in enumerate(bodies[:limit], start=1):
            result = fwd_bwd_step(session, job, body, learning_rate=LEARNING_RATE)
            losses[iteration] = result.avg_loss
            _snapshot(capture, work / label / f"step-{iteration}")
    return losses


def _run_independent(limit: int, work: Path, cfg, spec, bodies: list[dict]) -> tuple[int, Comparison] | None:
    """Train two fresh jobs on the same batches and return the first inexact component."""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    runtime = work / "runtime"
    with optimizer_capture_worker(), gateway(runtime, cfg.n_gpus) as session:
        left = _train_uninterrupted(session, "trajectory-a", limit, work, cfg, spec, bodies)
        capture = work / "capture-trajectory-b"
        payload = build_payload(
            cfg.training,
            spec.model.cache_path,
            SEED,
            attn_implementation=cfg.attention_implementation,
            optimizer_state_output_dir=capture,
        )
        print(f"TRAJECTORY label=trajectory-b limit={limit}", flush=True)
        with running_job(session, payload) as job:
            for iteration, body in enumerate(bodies[:limit], start=1):
                result = fwd_bwd_step(session, job, body, learning_rate=LEARNING_RATE)
                _snapshot(capture, work / "trajectory-b" / f"step-{iteration}")
                comparisons = _compare_iteration(
                    work,
                    iteration,
                    left[iteration],
                    result.avg_loss,
                    left_name="trajectory-a",
                    right_name="trajectory-b",
                )
                for comparison in comparisons:
                    if not comparison.exact:
                        return iteration, comparison
    return None


def main() -> int:
    cfg = load_config(CONFIG)
    spec = TestSpec.read(SPEC)
    arm = next(item for item in spec.arms if item.name == "gas1")
    provider = str(cfg.training.get("model_provider", "huggingface"))
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab_size = int(getattr(model_config, "text_config", model_config).vocab_size)
    bodies = [
        pack(
            build_batch(
                arm.name,
                arm.global_batch_size,
                arm.max_seq_len,
                vocab_size,
                seed=SEED + iteration,
            ),
            model_provider=provider,
        )
        for iteration in range(1, ITERATIONS + 1)
    ]
    horizon = CHECKPOINT_ITERATION + 1
    output = correctness_workdir("ap-host4-qwen36-lora-sp8-checkpoint-state-")
    base = output / "independent-uninterrupted"
    base.mkdir(parents=True, exist_ok=True)
    print(f"INDEPENDENT_TRAJECTORIES horizon={horizon}", flush=True)
    difference = _run_independent(1, base / "step-1", cfg, spec, bodies)
    if difference is None:
        print("STEP_1_EXACT extending_to_checkpoint_horizon", flush=True)
        difference = _run_independent(horizon, base / "through-horizon", cfg, spec, bodies)
    if difference is None:
        print(f"DIAGNOSIS exact_through_iteration={horizon}", flush=True)
        return 0
    iteration, comparison = difference
    print(
        f"FIRST_DIFFERENCE iteration={iteration} component={comparison.component} "
        f"max_abs={comparison.max_abs:.17g} l2={comparison.l2:.17g} worst={comparison.worst}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
