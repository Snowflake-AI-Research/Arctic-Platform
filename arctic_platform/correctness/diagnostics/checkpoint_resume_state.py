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

"""Locate the first inexact component in fresh trajectories or shared-prefix checkpoint resume.

Each trajectory is one fresh job on the same batches. Comparison starts at iteration 1 and stops at the
first inexact pre-step parameter, loss, or gradient. A one-step pair runs first; an exact one-step pair
continues through the checkpoint iteration plus one.

The optional loss control runs two complete trajectories without checkpoint operations or capture hooks.
The checkpoint control saves after ten steps, continues the same job, and then resumes the saved state
in a fresh job. Both branches share their exact pre-checkpoint training history.
The checkpoint-state mode captures the first post-boundary step from that shared prefix.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from arctic_platform.correctness.checks.checkpoint_resume import CHECKPOINT_ITERATION
from arctic_platform.correctness.checks.checkpoint_resume import LEARNING_RATE
from arctic_platform.correctness.checks.checkpoint_resume import RECORDED_ITERATIONS
from arctic_platform.correctness.checks.checkpoint_resume import TEST_ID
from arctic_platform.correctness.checks.checkpoint_resume import _iteration_bodies
from arctic_platform.correctness.checks.checkpoint_resume import _trajectory
from arctic_platform.correctness.checks.checkpoint_resume import _uninterrupted
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.dss_driver import GatewayTransport
from arctic_platform.correctness.harness.dss_driver import build_payload
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step
from arctic_platform.correctness.harness.dss_driver import gateway
from arctic_platform.correctness.harness.dss_driver import running_job
from arctic_platform.correctness.harness.model_state_capture import compare_model_state
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


def _capture_payload(cfg, spec, arm, capture: Path, model_state: Path | None = None) -> dict:
    payload = build_payload(
        cfg.training,
        spec.model.cache_path,
        SEED,
        attn_implementation=cfg.attention_implementation,
        optimizer_state_output_dir=capture,
    )
    if model_state is not None:
        payload["training_config"]["debug"]["model_state_output_dir"] = str(model_state.resolve())
    return payload


def _run_pair(limit: int, work: Path, cfg, spec, arm, bodies: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
    losses = {"uninterrupted": {}, "resumed": {}}
    runtime = work / "runtime"
    with optimizer_capture_worker(), gateway(runtime, cfg.n_gpus) as session:
        capture = work / "capture-uninterrupted"
        payload = _capture_payload(cfg, spec, arm, capture)
        with running_job(session, payload) as job:
            for iteration, body in enumerate(bodies[:limit], start=1):
                result = fwd_bwd_step(session, job, body, learning_rate=LEARNING_RATE)
                if iteration > CHECKPOINT_ITERATION:
                    losses["uninterrupted"][iteration] = result.avg_loss
                    _snapshot(capture, work / "uninterrupted" / f"step-{iteration}")

        first_capture = work / "capture-before-checkpoint"
        first_payload = _capture_payload(cfg, spec, arm, first_capture)
        with running_job(session, first_payload) as first_job:
            for body in bodies[:CHECKPOINT_ITERATION]:
                fwd_bwd_step(session, first_job, body, learning_rate=LEARNING_RATE)
            checkpoint = first_job.client.save_checkpoint()["path"]

        resumed_capture = work / "capture-resumed"
        resumed_payload = _capture_payload(cfg, spec, arm, resumed_capture)
        with running_job(session, resumed_payload) as resumed_job:
            restored_step = int(resumed_job.client.load_checkpoint(path=checkpoint)["global_step"])
            print(f"RESTORE requested_step={CHECKPOINT_ITERATION} restored_step={restored_step}", flush=True)
            for iteration, body in enumerate(bodies[CHECKPOINT_ITERATION:limit], start=CHECKPOINT_ITERATION + 1):
                result = fwd_bwd_step(session, resumed_job, body, learning_rate=LEARNING_RATE)
                losses["resumed"][iteration] = result.avg_loss
                _snapshot(resumed_capture, work / "resumed" / f"step-{iteration}")
    return losses["uninterrupted"], losses["resumed"]


def _run_shared_prefix_pair(
    limit: int, work: Path, cfg, spec, arm, bodies: list[dict], *, model_state: bool = False
) -> tuple[dict[int, float], dict[int, float]]:
    """Capture continuation and resume after one identical saved prefix."""
    if not CHECKPOINT_ITERATION < limit <= len(bodies):
        raise ValueError("shared-prefix state capture requires post-checkpoint batches")
    left, right = {}, {}
    model_state_directory = work / "model-state" if model_state else None
    with optimizer_capture_worker(), gateway(work / "runtime", cfg.n_gpus) as session:
        capture = work / "capture-uninterrupted"
        with running_job(session, _capture_payload(cfg, spec, arm, capture, model_state_directory)) as job:
            for iteration, body in enumerate(bodies[:limit], start=1):
                result = fwd_bwd_step(session, job, body, learning_rate=LEARNING_RATE)
                if iteration == CHECKPOINT_ITERATION:
                    checkpoint = job.client.save_checkpoint()["path"]
                if iteration > CHECKPOINT_ITERATION:
                    left[iteration] = result.avg_loss
                    _snapshot(capture, work / "uninterrupted" / f"step-{iteration}")

        capture = work / "capture-resumed"
        with running_job(session, _capture_payload(cfg, spec, arm, capture, model_state_directory)) as job:
            restored_step = int(job.client.load_checkpoint(path=checkpoint)["global_step"])
            print(f"RESTORE requested_step={CHECKPOINT_ITERATION} restored_step={restored_step}", flush=True)
            if restored_step != CHECKPOINT_ITERATION:
                raise RuntimeError(f"restored step {restored_step}, expected {CHECKPOINT_ITERATION}")
            for iteration, body in enumerate(bodies[CHECKPOINT_ITERATION:limit], start=CHECKPOINT_ITERATION + 1):
                result = fwd_bwd_step(session, job, body, learning_rate=LEARNING_RATE)
                right[iteration] = result.avg_loss
                _snapshot(capture, work / "resumed" / f"step-{iteration}")
    return left, right


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
    session, label: str, limit: int, work: Path, cfg, spec, arm, bodies: list[dict]
) -> dict[int, float]:
    """Train one fresh job and keep each step's pre-update snapshot."""
    print(f"TRAJECTORY label={label} limit={limit}", flush=True)
    losses: dict[int, float] = {}
    capture = work / f"capture-{label}"
    payload = _capture_payload(cfg, spec, arm, capture)
    with running_job(session, payload) as job:
        for iteration, body in enumerate(bodies[:limit], start=1):
            result = fwd_bwd_step(session, job, body, learning_rate=LEARNING_RATE)
            losses[iteration] = result.avg_loss
            _snapshot(capture, work / label / f"step-{iteration}")
    return losses


def _run_independent(limit: int, work: Path, cfg, spec, bodies: list[dict], *, arm) -> tuple[int, Comparison] | None:
    """Train two fresh jobs on the same batches and return the first inexact component."""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    runtime = work / "runtime"
    with optimizer_capture_worker(), gateway(runtime, cfg.n_gpus) as session:
        left = _train_uninterrupted(session, "trajectory-a", limit, work, cfg, spec, arm, bodies)
        capture = work / "capture-trajectory-b"
        payload = _capture_payload(cfg, spec, arm, capture)
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


def _run_loss_control(work: Path, cfg, spec, arm, bodies: list[dict], *, checkpoint_control: bool = False) -> dict:
    """Compare fresh trajectories or a shared-prefix resume, without capture hooks."""
    with GatewayTransport(
        work / "runtime",
        cfg.training,
        spec.model.cache_path,
        SEED,
        slots=cfg.n_gpus,
        attn_implementation=cfg.attention_implementation,
    ) as transport:
        if checkpoint_control:
            with transport.job() as first_job:
                left = _trajectory(first_job, bodies[:CHECKPOINT_ITERATION], first_iteration=1)
                checkpoint = first_job.save_resumable()
                right = dict(left)
                left.update(
                    _trajectory(first_job, bodies[CHECKPOINT_ITERATION:], first_iteration=CHECKPOINT_ITERATION + 1)
                )
            with transport.job(resume_from=checkpoint) as resumed_job:
                right.update(
                    _trajectory(resumed_job, bodies[CHECKPOINT_ITERATION:], first_iteration=CHECKPOINT_ITERATION + 1)
                )
        else:
            left = _uninterrupted(transport, bodies)
            right = _uninterrupted(transport, bodies)
    gate = spec.tolerance_for(TEST_ID)
    return {
        "case": arm.name,
        "gate": gate,
        "comparisons": [
            {
                "iteration": iteration,
                "left_loss": left[iteration],
                "right_loss": right[iteration],
                "abs_diff": abs(left[iteration] - right[iteration]),
                "within_gate": abs(left[iteration] - right[iteration]) <= gate,
            }
            for iteration in RECORDED_ITERATIONS
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("gas1", "gas4"), default="gas1")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument(
        "--loss-control", action="store_true", help="Compare two full runs without checkpoints or hooks"
    )
    controls.add_argument(
        "--checkpoint-control", action="store_true", help="Compare continuation and resume from one shared prefix"
    )
    controls.add_argument(
        "--checkpoint-state", action="store_true", help="Capture the first post-resume step from one shared prefix"
    )
    controls.add_argument(
        "--checkpoint-model-state",
        action="store_true",
        help="Also fingerprint every rank's parameters and buffers immediately around save and load",
    )
    args = parser.parse_args(argv)
    cfg = load_config(CONFIG)
    spec = TestSpec.read(SPEC)
    arm = next(item for item in spec.arms if item.name == args.case)
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab_size = int(getattr(model_config, "text_config", model_config).vocab_size)
    ctx = SimpleNamespace(cfg=cfg, config_id=cfg.config_id, vocab_size=vocab_size)
    bodies = _iteration_bodies(ctx, arm)
    horizon = CHECKPOINT_ITERATION + 1
    output = correctness_workdir(f"ap-qwen36-lora-sp8-checkpoint-state-{arm.name}-")
    if args.loss_control or args.checkpoint_control:
        if args.checkpoint_control:
            result = _run_loss_control(output, cfg, spec, arm, bodies, checkpoint_control=True)
            filename = "checkpoint-control.json"
            label = "SHARED_PREFIX_CHECKPOINT_CONTROL"
        else:
            result = _run_loss_control(output, cfg, spec, arm, bodies)
            filename = "loss-control.json"
            label = "UNINTERRUPTED_LOSS_CONTROL"
        (output / filename).write_text(json.dumps(result, indent=2) + "\n")
        print(label, json.dumps(result), "OUTPUT", output, flush=True)
        return 0
    if args.checkpoint_state or args.checkpoint_model_state:
        options = {"model_state": True} if args.checkpoint_model_state else {}
        left, right = _run_shared_prefix_pair(horizon, output, cfg, spec, arm, bodies, **options)
        if args.checkpoint_model_state:
            directory = output / "model-state"
            result = compare_model_state(directory / "before-save", directory / "after-load", ranks=cfg.n_gpus)
            (directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
            print("MODEL_STATE_COMPARISON", json.dumps(result), flush=True)
        difference = _first_difference(output, left, right)
        if difference is None:
            print(f"DIAGNOSIS shared_prefix_exact_through_iteration={horizon}", flush=True)
        else:
            iteration, comparison = difference
            print(
                f"FIRST_DIFFERENCE iteration={iteration} component={comparison.component} "
                f"max_abs={comparison.max_abs:.17g} l2={comparison.l2:.17g} worst={comparison.worst}",
                flush=True,
            )
        print("SHARED_PREFIX_STATE_OUTPUT", output, flush=True)
        return 0
    base = output / "independent-uninterrupted"
    base.mkdir(parents=True, exist_ok=True)
    print(
        f"INDEPENDENT_TRAJECTORIES case={arm.name} microbatches={arm.dss_microbatches} "
        f"horizon={horizon} output={output}",
        flush=True,
    )
    difference = _run_independent(1, base / "step-1", cfg, spec, bodies, arm=arm)
    if difference is None:
        print("STEP_1_EXACT extending_to_checkpoint_horizon", flush=True)
        difference = _run_independent(horizon, base / "through-horizon", cfg, spec, bodies, arm=arm)
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
