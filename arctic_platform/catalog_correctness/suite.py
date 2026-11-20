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

"""Run the five catalog correctness checks against a Cortex Training service."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
from pathlib import Path

from . import checks
from .reference import _accumulation_dtype
from .reference import _tensor_hash


def _comparison(target: dict, reference: dict, tolerance: float | None) -> dict:
    if not target or set(target) != set(reference):
        raise ValueError("comparison parameter coverage differs")
    differences = {}
    relative = {}
    for name in target:
        a, b = float(target[name]), float(reference[name])
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < 0:
            raise ValueError(f"nonfinite or negative measurement for {name}")
        differences[name] = abs(a - b)
        relative[name] = differences[name] / max(abs(b), 1e-12)
    worst = max(differences, key=differences.get)
    maximum = differences[worst]
    return {
        "status": checks.verdict(maximum, tolerance),
        "max_absolute_difference": maximum,
        "max_relative_difference": max(relative.values()),
        "worst_parameter": worst,
        "parameters": len(target),
        "absolute_tolerance": tolerance,
    }


def _initial_identity(root: Path, reference: dict) -> None:
    import deepspeed
    from deepspeed.runtime.constants import GRAD_ACCUM_DTYPE_DEFAULT

    from arctic_platform.correctness.harness.optimizer import load_dense_checkpoint_model

    checkpoint = load_dense_checkpoint_model(root)
    if checkpoint.get("global_steps") != 0:
        raise ValueError("initial checkpoint is not at step zero")
    if checkpoint.get("ds_version") != deepspeed.__version__:
        raise ValueError("checkpoint and local DeepSpeed versions differ; accumulation default is not verified")
    dtype = _accumulation_dtype(checkpoint["ds_config"], reference["dtype"], GRAD_ACCUM_DTYPE_DEFAULT)
    if dtype != reference["gas4"]["gradient_accumulation_dtype"]:
        raise ValueError("checkpoint and reference gradient accumulation dtypes differ")
    module, shared = checkpoint.get("module"), checkpoint.get("shared_params") or {}
    expected = reference["initial_parameters"]
    if not isinstance(module, dict) or not isinstance(shared, dict) or set(module) != set(expected) | set(shared):
        raise ValueError("initial checkpoint parameter coverage differs")
    for name, item in expected.items():
        value = module[name]
        if list(value.shape) != item["shape"] or _tensor_hash(value) != item["sha256"]:
            raise ValueError(f"initial checkpoint tensor differs: {name}")
    for alias, source in shared.items():
        if source not in expected or _tensor_hash(module[alias]) != expected[source]["sha256"]:
            raise ValueError(f"initial shared parameter differs: {alias}")


def _download(client, job: str, checkpoint: str, root: Path) -> None:
    import requests

    export = client.export_checkpoint(job, checkpoint)
    files = export.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint export contains no files")
    root.mkdir(parents=True, exist_ok=True)
    for entry in files:
        name = Path(entry["filename"])
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise ValueError("invalid checkpoint filename")
        size = int(entry["size_bytes"])
        if size < 0:
            raise ValueError("invalid checkpoint size")
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        try:
            with requests.get(entry["url"], stream=True, timeout=(30, 300)) as response:
                if response.status_code != 200:
                    raise ValueError(f"checkpoint download failed with HTTP {response.status_code}")
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        stream.write(chunk)
        except requests.RequestException:
            raise ValueError("checkpoint download failed") from None
        if partial.stat().st_size != size:
            raise ValueError(f"checkpoint file size mismatch: {name}")
        partial.replace(destination)


def _optimizer_comparison(before: Path, after: Path, output: Path, reference: dict, tolerance: float | None) -> dict:
    from arctic_platform.correctness.harness.optimizer import decode_optimizer_capture

    optimizer = reference["optimizer"]
    manifest = decode_optimizer_capture(
        before,
        after,
        output,
        expected_optimizer_config=optimizer,
        expected_learning_rate=optimizer["lr"],
        expected_gradient_clipping=reference["gradient_clipping"] or 0.0,
        allow_external_optimizer_config=True,
        output_root=output.parent,
    )
    return _optimizer_manifest_comparison(manifest, reference, tolerance)


def _optimizer_manifest_comparison(manifest: Path, reference: dict, tolerance: float | None) -> dict:
    import torch

    from arctic_platform.correctness.checks.optimizer_state import optimizer_comparison_tensors
    from arctic_platform.correctness.checks.optimizer_state import optimizer_tensor_norm

    document = json.loads(manifest.read_bytes())
    if document.get("step") != 1:
        raise ValueError("optimizer comparison requires a first-step checkpoint")
    optimizer = reference["optimizer"]
    actual = {"parameter_update_residual_norm": {}, "exp_avg_norm": {}, "exp_avg_sq_norm": {}}
    for entry in document["parameters"]:
        name = entry["name"]
        if name in actual["parameter_update_residual_norm"]:
            raise ValueError(f"duplicate optimizer parameter {name}")
        expected = reference["initial_parameters"].get(name)
        if expected is None or entry["shape"] != expected["shape"]:
            raise ValueError(f"optimizer parameter shape differs: {name}")
        tensors = torch.load(manifest.parent / entry["file"], map_location="cpu", weights_only=True)
        values = optimizer_comparison_tensors(
            tensors, step=1, optimizer_config=optimizer, learning_rate=optimizer["lr"], backend="ap"
        )
        for key, value in values.items():
            actual[f"{key}_norm"][name] = optimizer_tensor_norm(value)
    results = {
        key: _comparison(
            values, {name: values[key] for name, values in reference["optimizer_step"].items()}, tolerance
        )
        for key, values in actual.items()
    }
    return {"status": _combined_status([item["status"] for item in results.values()]), "measurements": results}


def _combined_status(statuses: list[str]) -> str:
    if "error" in statuses:
        return "error"
    return "fail" if "fail" in statuses else ("pass" if all(status == "pass" for status in statuses) else "measured")


def _read_calibration(path: Path, reference: dict, reference_sha256: str) -> dict:
    calibration = json.loads(path.read_bytes())
    workload = calibration.get("workload")
    expected_keys = {"model", "recipe_sha256", "inputs_sha256", "model_files_sha256", "reference_source_sha256"}
    if (
        calibration.get("policy") != "reference-repeatability-4x-upward-1e-3-v1"
        or calibration.get("runs", 0) < 16
        or reference_sha256 not in calibration.get("reference_sha256", [])
        or not isinstance(workload, dict)
        or set(workload) != expected_keys
        or any(reference.get(key) != value for key, value in workload.items())
    ):
        raise ValueError("calibration does not match this reference")
    gates = {}
    for name in ("gradient", "optimizer"):
        value = calibration.get("gates", {}).get(name, {}).get("absolute_tolerance")
        if value is None:
            raise ValueError(f"calibration has no {name} tolerance")
        checks.verdict(0, value)
        gates[name] = value
    return gates


def _save(client, job: str, kind: str) -> str:
    return checks.checkpoint_id(client.poll_request(job, client.save(job, checkpoint_type=kind)))


def _same_image(jobs, first: str, second: str) -> None:
    first_image = jobs.runtime[first].get("image_tag")
    if not first_image or first_image != jobs.runtime[second].get("image_tag"):
        raise ValueError("training jobs must report the same service image")


def _resume(client, jobs, training, job, checkpoint, inputs, losses, baseline_post, tolerance):
    control_job = jobs.start(training)
    _same_image(jobs, job, control_job)
    control = []
    for index, batch in enumerate(inputs[:10], 1):
        value, step = checks.train_step(client, control_job, batch)
        if checks.step_number(step) != index:
            raise ValueError("control step number mismatch")
        control.append(value)
    restored = client.poll_request(control_job, client.load(control_job, checkpoint, source_job_id=job))
    if restored.get("checkpoint_id") not in (None, checkpoint):
        raise ValueError("service restored a different checkpoint")
    resumed = []
    for index, batch in enumerate(inputs[10:20], 11):
        value, step = checks.train_step(client, control_job, batch)
        if checks.step_number(step) != index:
            raise ValueError("resumed step number mismatch")
        resumed.append(value)
    resumed_post = checks.loss(checks.training_logprobs(client, control_job, inputs[19]))
    difference = max(
        max(abs(x - y) for x, y in zip(control, losses[:10], strict=True)),
        max(abs(x - y) for x, y in zip(resumed, losses[10:20], strict=True)),
        abs(resumed_post - baseline_post),
    )
    return {
        "status": checks.verdict(difference, tolerance),
        "max_loss_difference": difference,
        "control_losses": control,
        "resumed_losses": resumed,
        "post_update_difference": abs(resumed_post - baseline_post),
        "absolute_tolerance": tolerance,
    }


def _sample(client, jobs, sampling, job, checkpoint, tokens, training_logprobs, tolerance):
    sampler = copy.deepcopy(sampling)
    sampler["sub_job_configs"][0]["source_checkpoint_info"] = {"source_job_id": job, "checkpoint_id": checkpoint}
    sample_job = jobs.start(sampler)
    sampled = checks.sampling_logprobs(client, sample_job, tokens)
    result = checks.compare_logprobs(sampled, training_logprobs, tolerance, None)
    result["status"] = checks.verdict(result["loss_abs_diff"], tolerance)
    return result


def _gas4(client, jobs, training, inputs, *, source_job=None):
    body = copy.deepcopy(training)
    settings = body["sub_job_configs"][0]["training_config"]
    rows = [batch[0] for batch in inputs[:4]]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("four-microbatch input rows must have equal lengths")
    settings.setdefault("mb_spec", {})["max_tokens_per_mb"] = len(rows[0])
    job = jobs.start(body)
    if source_job is not None:
        _same_image(jobs, source_job, job)
    initial = _save(client, job, "resumable")
    _, step = checks.train_step(client, job, rows, expected_model_calls=4)
    if checks.step_number(step) != 1:
        raise ValueError("four-microbatch target step number mismatch")
    gradients = step.get("gradient_norms_per_param") or (step.get("metrics") or {}).get("gradient_norms_per_param")
    if not isinstance(gradients, dict):
        raise ValueError("four-microbatch target did not return named gradient norms")
    return {
        "job": job,
        "initial": initial,
        "step1": _save(client, job, "resumable"),
        "gradient_norms": gradients,
        "microbatch_tokens": len(rows[0]),
    }


def _gas4_comparisons(client, capture, scratch, reference, tolerances):
    initial, step1 = scratch / "gas4-initial", scratch / "gas4-step1"
    _download(client, capture["job"], capture["initial"], initial)
    _download(client, capture["job"], capture["step1"], step1)
    _initial_identity(initial, reference)
    gradient = _comparison(capture["gradient_norms"], reference["gas4"]["gradient_norms"], tolerances["gradient"])
    gas4_reference = {**reference, "optimizer_step": reference["gas4"]["optimizer_step"]}
    optimizer = _optimizer_comparison(
        initial, step1, scratch / "gas4-optimizer", gas4_reference, tolerances["optimizer"]
    )
    return gradient, optimizer


def run(
    client,
    training: dict,
    sampling: dict,
    inputs: list,
    reference: dict,
    scratch: Path,
    report: dict,
    tolerances: dict,
):
    scratch = scratch.expanduser()
    if scratch.exists() or scratch.is_symlink():
        raise ValueError("scratch must be a new directory")
    scratch.mkdir(parents=True, exist_ok=False)
    scratch = scratch.resolve()
    report_path = Path(report["report_path"])

    def write():
        report["created_job_ids"] = list(jobs.jobs)
        report["runtime"] = jobs.runtime
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)

    jobs = checks.Jobs(client, write)

    def attempt(work):
        existing = set(jobs.active)
        try:
            return work()
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            for job in reversed(jobs.active[:]):
                if job not in existing:
                    jobs.stop(job)
            write()

    train = copy.deepcopy(training)
    debug = train["sub_job_configs"][0]["training_config"].setdefault("debug", {})
    debug["gradient_norms_per_param"] = True
    result = report["results"]
    try:
        a = jobs.start(train)
        report["checkpoints"] = {"initial": _save(client, a, "resumable")}
        losses = []
        for index, batch in enumerate(inputs[:100], 1):
            value, step = checks.train_step(client, a, batch)
            if checks.step_number(step) != index:
                raise ValueError("target step number mismatch")
            losses.append(value)
            if index == 1:
                gradients = step.get("gradient_norms_per_param") or (step.get("metrics") or {}).get(
                    "gradient_norms_per_param"
                )
                if not isinstance(gradients, dict):
                    raise ValueError("target did not return named gradient norms")
                report["checkpoints"]["step1"] = _save(client, a, "resumable")
            if index == 10:
                report["checkpoints"]["resume"] = _save(client, a, "resumable")
                report["checkpoints"]["sampling"] = _save(client, a, "weights-only")
                checkpoint_logprobs = checks.training_logprobs(client, a, inputs[100])
            if index == 20:
                baseline_post = checks.loss(checks.training_logprobs(client, a, inputs[19]))
            if index == 1 or index % 10 == 0:
                print(f"QA training step {index}/100", flush=True)
                report["training_losses"] = losses
                write()
        validation = checks.loss(checks.training_logprobs(client, a, inputs[100]))
        report["validation_loss"] = validation
        write()

        print("running independent checkpoint control", flush=True)
        result["checkpoint-resume-loss"] = attempt(
            lambda: _resume(
                client,
                jobs,
                training,
                a,
                report["checkpoints"]["resume"],
                inputs,
                losses,
                baseline_post,
                tolerances["resume"],
            )
        )
        jobs.stop(a)
        write()

        print("running checkpoint-backed sampler", flush=True)
        result["inference-checkpoint-loss"] = attempt(
            lambda: _sample(
                client,
                jobs,
                sampling,
                a,
                report["checkpoints"]["sampling"],
                inputs[100],
                checkpoint_logprobs,
                tolerances["sampler"],
            )
        )
        write()

        print("running four-microbatch gradient and optimizer checks", flush=True)
        gas4 = attempt(lambda: _gas4(client, jobs, train, inputs, source_job=a))
        if gas4.get("status") != "error":
            report["gas4_capture"] = {key: value for key, value in gas4.items() if key != "gradient_norms"}
        write()

        print("downloading initial and step-one checkpoints", flush=True)
        initial, step1 = scratch / "initial", scratch / "step1"
        _download(client, a, report["checkpoints"]["initial"], initial)
        _download(client, a, report["checkpoints"]["step1"], step1)
        _initial_identity(initial, reference)
        report["initial_parameters_match"] = True
        gas1_grad = _comparison(gradients, reference["gradient_norms"], tolerances["gradient"])
        gas1_optimizer = attempt(
            lambda: _optimizer_comparison(initial, step1, scratch / "optimizer", reference, tolerances["optimizer"])
        )
        gas4_result = (
            gas4
            if gas4.get("status") == "error"
            else attempt(lambda: _gas4_comparisons(client, gas4, scratch, reference, tolerances))
        )
        gas4_grad, gas4_optimizer = (gas4_result, gas4_result) if isinstance(gas4_result, dict) else gas4_result
        result["single-step-grads"] = {
            "status": _combined_status([gas1_grad["status"], gas4_grad["status"]]),
            "gas1": gas1_grad,
            "gas4": gas4_grad,
        }
        result["single-step-optimizer"] = {
            "status": _combined_status([gas1_optimizer["status"], gas4_optimizer["status"]]),
            "gas1": gas1_optimizer,
            "gas4": gas4_optimizer,
        }
        train_diff = abs(losses[-1] - reference["losses"][-1])
        validation_diff = abs(validation - reference["validation_loss"])
        result["e2e-sft-train-validate"] = {
            "status": checks.verdict(max(train_diff, validation_diff), tolerances["e2e"]),
            "training_loss_difference": train_diff,
            "validation_loss_difference": validation_diff,
            "absolute_tolerance": tolerances["e2e"],
        }
        write()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["cleanup_errors"] = jobs.cleanup()
        write()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--connection")
    parser.add_argument("--host")
    parser.add_argument("--database")
    parser.add_argument("--schema")
    parser.add_argument("--pat-env", default="CORTEX_TRAINING_PAT")
    parser.add_argument("--poll-timeout", type=float, default=3600)
    for name in ("gradient", "optimizer", "resume", "sampler", "e2e"):
        parser.add_argument(f"--{name}-tolerance", type=float)
    args = parser.parse_args()
    training, digest = checks.read_recipe(args.training_config, "training")
    sampling, _ = checks.read_recipe(args.sampling_config, "sampling")
    checks.validate_recipes(training, sampling)
    reference = json.loads(args.reference.read_bytes())
    inputs = json.loads(args.batches.read_bytes())[:101]
    input_hash = hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest()
    if (
        reference.get("schema") != "catalog-hf-reference-v2"
        or reference.get("recipe_sha256") != digest
        or reference.get("inputs_sha256") != input_hash
        or len(inputs) != 101
        or len(reference.get("losses", [])) != 100
        or not isinstance(reference.get("gas4"), dict)
    ):
        raise ValueError("reference does not match the training recipe and 100-step workload")
    names = set(reference.get("initial_parameters", {}))
    required_states = {"parameter_update_residual_norm", "exp_avg_norm", "exp_avg_sq_norm"}
    for case in (reference, reference["gas4"]):
        if (
            not names
            or set(case.get("gradient_norms", {})) != names
            or set(case.get("optimizer_step", {})) != names
            or any(set(values) != required_states for values in case["optimizer_step"].values())
        ):
            raise ValueError("reference has incomplete gradient or optimizer measurements")
    tolerances = {
        name: getattr(args, f"{name}_tolerance") for name in ("gradient", "optimizer", "resume", "sampler", "e2e")
    }
    reference_sha256 = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    if args.calibration:
        for name, value in _read_calibration(args.calibration, reference, reference_sha256).items():
            if tolerances[name] is None:
                tolerances[name] = value
    for value in tolerances.values():
        checks.verdict(0, value)
    report = {
        "report_path": str(args.report),
        "model": reference["model"],
        "original_suite_equivalent": False,
        "coverage_limitations": [
            "No worker participation receipts; accumulation precision follows the recipe.",
            "Resume restores into a job that has already trained, rather than an untouched job.",
            "The workload scores all tokens and does not cover answer masks or padding.",
            "Distributed training, LoRA, MoE and RL checks are unsupported.",
        ],
        "reference_sha256": reference_sha256,
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
        "inputs_sha256": input_hash,
        "recipe_sha256": digest,
        "results": {},
    }
    client = checks.connect(args)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise SystemExit("received SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        run(client, training, sampling, inputs, reference, args.scratch, report, tolerances)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        client.close()
    statuses = [item["status"] for item in report["results"].values()]
    if report.get("cleanup_errors") or len(statuses) != 5 or any(status in {"error", "fail"} for status in statuses):
        return 1
    return 0 if all(status == "pass" for status in statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
