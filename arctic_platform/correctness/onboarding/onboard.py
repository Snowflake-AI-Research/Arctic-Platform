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

# !/usr/bin/env python3
"""Onboard one Arctic Platform config into the fixed correctness harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

# Sixteen reference executions over an eight-GPU node: two waves, so more samples cost the same wall
# clock as two sequential runs.
CALIBRATION_RUNS = 16
# How far above the run-to-run variation the calibrated gate sits. Sixteen calibration runs measure the
# tail imperfectly: on the CPU-offloaded Qwen3.8-27B config a factor of two produced a 2e-03 gate that the
# validation run then exceeded at 2.929e-03, so a factor that only just covers the observed spread does not
# survive the next run. Four keeps that measurement inside the gate with room, and stays far below the
# disagreement a real defect produces -- a dropped microbatch gradient is a fraction of the correct norm,
# not a per-mille difference.
GATE_SAFETY_FACTOR = 4.0

OOM_EXIT = 42


def _checkout_root(config_path: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=config_path.resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _resolve_source(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists() or value.startswith(("/", ".", "~")):
        return str(path.resolve())
    return value


def _select_source_checkpoint(override: str | None, config) -> str:
    """Use an explicit checkpoint override or the model named by the training config."""
    return _resolve_source(override or config.model_name)


def _model_shape(source_checkpoint: str) -> tuple[int, int, int]:
    """Return source depth, repeating period, and the smallest representative prefix.

    The prefix ends after every attention-layer type has appeared once and is never shorter than four
    layers. It deliberately does not search for the largest single-GPU model: onboarding needs a stable,
    representative regression fixture, not the maximum memory load.
    """
    from transformers import AutoConfig

    from arctic_platform.correctness.onboarding.synth_model import detect_block_period

    config = AutoConfig.from_pretrained(source_checkpoint, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    full_layers = int(text_config.num_hidden_layers)
    layer_types = list(getattr(text_config, "layer_types", []))
    period = detect_block_period(layer_types) if layer_types else 1
    if full_layers < 4:
        raise ValueError(f"source model has {full_layers} layers; onboarding requires at least 4")
    if not layer_types:
        return full_layers, period, 4

    required = set(layer_types)
    seen = set()
    representative_layers = None
    for index, layer_type in enumerate(layer_types, start=1):
        seen.add(layer_type)
        if seen == required:
            representative_layers = index
            break
    if representative_layers is None:
        raise RuntimeError("could not find a prefix containing every attention-layer type")
    return full_layers, period, max(4, representative_layers)


def _optimizer_reference_model(spec, cache_root: Path) -> tuple[Path, int]:
    """Choose Test 2's minimum representative reference without changing Test 1's frozen model."""
    _, _, layers = _model_shape(spec.model.source_checkpoint)
    return cache_root / f"{Path(spec.model.source_checkpoint).name}-{layers}L", layers


def _probe_command(args: argparse.Namespace, layers: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "arctic_platform.correctness",
        "onboard",
        "--config",
        str(args.config.resolve()),
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--cache-root",
        str(args.cache_root),
        "--tokens",
        str(args.tokens),
        "--reference-token-budget",
        str(args.reference_token_budget),
        "--probe-layers",
        str(layers),
        "--probe-output",
        str(output),
    ]
    return command


def _run_probe(args: argparse.Namespace, layers: int) -> dict:
    output = args.output_dir / "sizing" / f"{layers}-layers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    print(f"phase 1/3 probe: layers={layers}", flush=True)
    completed = subprocess.run(_probe_command(args, layers, output), cwd=args.checkout_root, env=env)
    if not output.exists():
        raise RuntimeError(f"{layers}-layer sizing worker exited {completed.returncode} without writing {output}")
    result = json.loads(output.read_text())
    if completed.returncode == 0 and result.get("status") == "fit":
        return result
    if completed.returncode == OOM_EXIT and result.get("status") == "oom":
        return result
    raise RuntimeError(
        f"{layers}-layer sizing worker exited {completed.returncode}: {result.get('error', 'no error recorded')}"
    )


def _is_cuda_oom(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            return True
    except ImportError:
        pass
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def _memory_snapshot() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device": device,
        "allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "total_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
    }


def _probe_worker(args: argparse.Namespace) -> int:
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"

    import torch
    from transformers import AutoConfig

    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.seeds import SEED
    from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained
    from arctic_platform.correctness.reference.hf_single_gpu import run as run_reference

    print(f"interpreter {sys.prefix}", flush=True)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    config = load_config(args.config)
    training = config.effective_training
    model_path = args.cache_root / f"{Path(args.source_checkpoint).name}-{args.probe_layers}L"
    cache_preexisting = (model_path / "config.json").exists()
    payload = {"layers": args.probe_layers, "model_path": str(model_path)}
    try:
        model = materialize_pretrained(
            str(args.source_checkpoint),
            str(model_path),
            args.probe_layers,
        )
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text_config = getattr(model_config, "text_config", model_config)
        batch = build_batch("gas1", 1, args.tokens, text_config.vocab_size, seed=SEED)
        result = run_reference(
            str(model_path),
            batch,
            token_budget=args.reference_token_budget,
            attn_implementation=config.attention_implementation,
            fp32_lm_head=bool(training.get("fp32_lm_head", False)),
            fused_cross_entropy=training.get("fused_cross_entropy", False),
            matmul_precision=training.get("matmul_precision", "highest"),
            peft_config=training.get("peft_config"),
            seed=SEED,
            lm_head_token_chunk_size=config.lm_head_token_chunk_size,
            lm_head_vocab_chunk_size=training.get("fused_lm_head_vocab_chunk_size", 8192),
        )
        payload.update(
            status="fit",
            peak_gib=result.peak_gib,
            param_count=model.param_count,
            model_hash=model.content_hash,
        )
        exit_code = 0
    except Exception as error:  # CUDA OOM is an expected sizing outcome; other failures stay fatal.
        payload.update(memory=_memory_snapshot(), error=f"{type(error).__name__}: {error}")
        if _is_cuda_oom(error):
            payload["status"] = "oom"
            if not cache_preexisting:
                shutil.rmtree(model_path, ignore_errors=True)
                payload["model_cache_removed"] = True
            exit_code = OOM_EXIT
        else:
            payload["status"] = "error"
            exit_code = 1
    args.probe_output.parent.mkdir(parents=True, exist_ok=True)
    args.probe_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"sizing layers={args.probe_layers} status={payload['status']} "
        f"peak_gib={payload.get('peak_gib', payload.get('memory', {}).get('max_allocated_gib'))}",
        flush=True,
    )
    return exit_code


def build_spec(
    cfg,
    source_checkpoint: str,
    cache_root: str,
    arm_defs,
    *,
    attn_implementations=None,
    num_layers: int | None = None,
    workdir: Path | None = None,
):
    """Materialize the reduced reference model and freeze the generated cases."""
    from transformers import AutoConfig

    from arctic_platform.correctness.harness.arms import correctness_microbatch_tokens
    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.seeds import SEED
    from arctic_platform.correctness.harness.spec import ArmSpec
    from arctic_platform.correctness.harness.spec import TestSpec
    from arctic_platform.correctness.harness.workdir import correctness_workdir
    from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained
    from arctic_platform.correctness.onboarding.synth_model import size_for_single_gpu

    temporary = Path(workdir) if workdir else correctness_workdir("dss-onboard-")
    temporary.mkdir(parents=True, exist_ok=True)
    if num_layers is None:
        num_layers, _ = size_for_single_gpu(source_checkpoint)
    destination = f"{cache_root}/{Path(source_checkpoint).name}-{num_layers}L"
    model = materialize_pretrained(source_checkpoint, destination, num_layers)
    print(f"[onboard] model {model.param_count:,} params, {num_layers} layers -> {destination}", flush=True)

    model_config = AutoConfig.from_pretrained(destination, trust_remote_code=True)
    vocab_size = getattr(model_config, "text_config", model_config).vocab_size
    arms = []
    for definition in arm_defs:
        batch = build_batch(
            definition.name,
            definition.global_batch_size,
            definition.max_seq_len,
            vocab_size,
            seed=SEED,
        )
        arms.append(
            ArmSpec(
                name=definition.name,
                global_batch_size=definition.global_batch_size,
                max_seq_len=definition.max_seq_len,
                total_tokens=batch.total_tokens,
                active_tokens=batch.active_tokens,
                dss_microbatches=definition.microbatches(
                    correctness_microbatch_tokens(cfg.max_tokens_per_mb), cfg.dp_size
                ),
                pad_fraction=batch.pad_fraction,
            )
        )

    implementations = attn_implementations or [cfg.attention_implementation]
    notes = []
    if not any(arm.dss_microbatches > 1 for arm in arms):
        notes.append(
            "No arm splits into more than one microbatch, so gradient accumulation is untested; "
            "raise a global batch size or lower mb_spec.max_tokens_per_mb."
        )
    return TestSpec(
        config_id=cfg.config_id,
        config_path=str(cfg.path),
        model=model,
        arms=arms,
        attn_implementations=implementations,
        applicable_tests=["single-step-grads"],
        config_checksum=config_checksum(cfg),
        inapplicable={},
        param_name_map={},
        notes=notes,
    )


def _write_initial_spec(args: argparse.Namespace, selected: dict, first_oom: dict | None) -> None:
    from arctic_platform.correctness.harness.arms import arms_for
    from arctic_platform.correctness.harness.config import load_config

    config = load_config(args.config)
    try:
        config.path = args.config.relative_to(args.checkout_root)
    except ValueError:
        pass
    spec = build_spec(
        config,
        str(args.source_checkpoint),
        str(args.cache_root),
        arms_for(args.tokens, config.n_gpus),
        num_layers=int(selected["layers"]),
        workdir=args.output_dir / "onboard",
    )
    spec.model.measured_reference_peak_gib = float(selected["peak_gib"])
    sizing_note = (
        f"Single-GPU reference uses {config.attention_implementation} with {selected['layers']} "
        f"complete-cycle layers and peaks at {selected['peak_gib']:.3f} GiB."
    )
    if first_oom is not None:
        sizing_note += f" The next complete cycle, {first_oom['layers']} layers, reaches CUDA OOM."
    spec.notes.append(sizing_note)
    spec.write(args.test_spec)


def _render_and_run_calibration(args: argparse.Namespace, selected: dict) -> Path:
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.runner import materialize_dss_peft_adapter

    from .calibration import render

    model_path = args.cache_root / f"{Path(args.source_checkpoint).name}-{selected['layers']}L"
    config = load_config(args.config)
    peft_adapter_path = materialize_dss_peft_adapter(
        config,
        str(model_path),
        args.output_dir / "calibration-adapter-init",
        attn_implementation=config.attention_implementation,
    )
    calibration_json = args.output_dir / "reference-repeatability.json"
    calibration_script = args.output_dir / "reference_repeatability.py"
    render_args = argparse.Namespace(
        config=args.config,
        source_checkpoint=args.source_checkpoint,
        layers=int(selected["layers"]),
        model_path=model_path,
        output_json=calibration_json,
        output_script=calibration_script,
        test_spec=args.test_spec,
        test_id=args.test_id,
        tokens=args.tokens,
        reference_token_budget=args.reference_token_budget,
        peft_adapter_path=peft_adapter_path,
    )
    calibration_script.write_text(render(render_args))
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run([sys.executable, str(calibration_script)], cwd=args.checkout_root, env=env, check=True)
    return calibration_json


def _run_optimizer_calibration(args: argparse.Namespace) -> Path:
    """Calibrate full-tensor optimizer deltas across a set of nondeterministic reference runs."""
    from decimal import ROUND_CEILING
    from decimal import Decimal

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoConfig

    from arctic_platform.correctness.checks.optimizer_state import max_pairwise_optimizer_delta
    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.batches import save
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.runner import materialize_dss_peft_adapter
    from arctic_platform.correctness.harness.runner import run_reference
    from arctic_platform.correctness.harness.seeds import SEED
    from arctic_platform.correctness.harness.spec import TestSpec
    from arctic_platform.correctness.harness.spec import TestTolerance

    config = load_config(args.config)
    spec = TestSpec.read(args.test_spec)
    if spec.config_checksum != config_checksum(config):
        raise ValueError("the existing spec does not match the config; onboard Test 1 before Test 2")
    gas1 = next((arm for arm in spec.arms if arm.name == "gas1"), None)
    if gas1 is None:
        raise ValueError("the existing spec has no gas1 case")
    model_path = Path(args.optimizer_model_path)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = getattr(model_config, "text_config", model_config).vocab_size
    batch = build_batch(gas1.name, gas1.global_batch_size, gas1.max_seq_len, vocab_size, seed=SEED)
    batch_path = args.output_dir / "optimizer-calibration-batch.pt"
    save(batch, batch_path)
    training = config.effective_training
    optimizer_config = config.training.get("optimizer")
    if not isinstance(optimizer_config, dict):
        raise ValueError("optimizer calibration requires an explicit optimizer config")
    adapter_path = materialize_dss_peft_adapter(
        config,
        str(model_path),
        args.output_dir / "optimizer-calibration-adapter",
        attn_implementation=config.attention_implementation,
    )

    lanes = max(1, min(CALIBRATION_RUNS, torch.cuda.device_count()))
    fast_root = Path("/data-fast/dss-correctness-calibration")
    calibration_root = (fast_root if fast_root.parent.exists() else args.output_dir) / config.config_id
    shutil.rmtree(calibration_root, ignore_errors=True)
    calibration_root.mkdir(parents=True)
    manifests: list[Path] = []
    worst_delta = 0.0
    worst_name = None
    losses = []
    peaks = []
    durations = []

    def one_run(run_index: int) -> tuple[int, float, dict]:
        run_dir = calibration_root / f"run-{run_index + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        result = run_reference(
            str(model_path),
            batch_path,
            run_dir / "result.json",
            token_budget=args.reference_token_budget,
            ce_chunk=2048,
            attn=config.attention_implementation,
            fp32_lm_head=bool(training.get("fp32_lm_head", False)),
            fused_cross_entropy=training.get("fused_cross_entropy", False),
            matmul_precision=training.get("matmul_precision", "highest"),
            peft_config=training.get("peft_config"),
            peft_adapter_path=adapter_path,
            lm_head_token_chunk_size=config.lm_head_token_chunk_size,
            lm_head_vocab_chunk_size=training.get("fused_lm_head_vocab_chunk_size", 8192),
            optimizer_config=optimizer_config,
            learning_rate=1e-2,
            gradient_clipping=config.training.get("gradient_clipping"),
            optimizer_dtype=config.optimizer_dtype,
            optimizer_output_dir=run_dir / "states",
            cuda_device=run_index % lanes,
            deterministic=False,
        )
        return run_index, time.monotonic() - started, result

    try:
        completed: dict[int, tuple[float, dict]] = {}
        with tqdm(
            total=CALIBRATION_RUNS, desc="Finding optimizer tolerance", unit="run", dynamic_ncols=True
        ) as progress:
            with ThreadPoolExecutor(max_workers=lanes) as pool:
                futures = [pool.submit(one_run, index) for index in range(CALIBRATION_RUNS)]
                for future in as_completed(futures):
                    index, elapsed, result = future.result()
                    completed[index] = (elapsed, result)
                    progress.set_postfix_str(f"last={elapsed:.1f}s, {lanes} per wave")
                    progress.update()
        for index in range(CALIBRATION_RUNS):
            elapsed, result = completed[index]
            durations.append(elapsed)
            losses.append(float(result["loss"]))
            peaks.append(float(result["peak_gib"]))
            manifests.append(Path(result["optimizer_state_manifest"]))
        parameter_count = len(json.loads(manifests[0].read_text())["parameters"])
        pair_count = CALIBRATION_RUNS * (CALIBRATION_RUNS - 1) // 2
        with tqdm(
            total=parameter_count, desc=f"Comparing {pair_count} reference pairs", unit="tensor", dynamic_ncols=True
        ) as progress:
            worst = max_pairwise_optimizer_delta(
                manifests,
                optimizer_config=optimizer_config,
                learning_rate=1e-2,
                progress=progress,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if worst is not None:
            worst_delta = worst.delta_norm
            worst_name = f"{worst.name}::{worst.state}"
    finally:
        shutil.rmtree(calibration_root, ignore_errors=True)

    quantum = Decimal("0.001")
    computed_gate = GATE_SAFETY_FACTOR * worst_delta
    rounded_gate = (Decimal(str(computed_gate)) / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
    selected_gate = max(float(quantum), float(rounded_gate))
    tolerance = TestTolerance(
        absolute=selected_gate,
        calibration_runs=CALIBRATION_RUNS,
        raw_max_same_tensor_range=worst_delta,
        multiplier=GATE_SAFETY_FACTOR,
        computed_gate=computed_gate,
        worst_tensor=worst_name,
        status=(
            f"calibrated from {CALIBRATION_RUNS} nondeterministic optimizer-reference executions; 4x the maximum "
            "pairwise moment or update-residual delta, rounded upward in 1e-3 units"
        ),
    )
    spec.test_tolerances["single-step-optimizer"] = tolerance
    spec.test_settings["single-step-optimizer"] = {
        "model_cache_path": str(model_path),
        "num_hidden_layers": int(args.optimizer_layers),
        "reference_token_budget": args.reference_token_budget,
        "ce_chunk": 2048,
        "learning_rate": 1e-2,
        "optimizer_dtype": config.optimizer_dtype,
        "calibration_metric": (
            f"maximum pairwise full-tensor moment or update-residual L2 delta across {CALIBRATION_RUNS} runs"
        ),
    }
    if "single-step-optimizer" not in spec.applicable_tests:
        spec.applicable_tests.append("single-step-optimizer")
    spec.write(args.test_spec)

    output = args.output_dir / "optimizer-reference-repeatability.json"
    output.write_text(
        json.dumps(
            {
                "configuration": {
                    "test_id": "single-step-optimizer",
                    "case": gas1.name,
                    "global_batch_size": gas1.global_batch_size,
                    "max_seq_len": gas1.max_seq_len,
                    "model_cache_path": str(model_path),
                    "num_hidden_layers": int(args.optimizer_layers),
                    "reference_token_budget": args.reference_token_budget,
                    "runs": CALIBRATION_RUNS,
                    "learning_rate": 1e-2,
                    "optimizer_dtype": config.optimizer_dtype,
                    "deterministic_algorithms": False,
                    "flash_attention_deterministic": False,
                },
                "losses": losses,
                "peak_gib": peaks,
                "durations_seconds": durations,
                "tolerance": {
                    "raw_max_baseline_delta_norm": worst_delta,
                    "worst_tensor_and_state": worst_name,
                    "multiplier": GATE_SAFETY_FACTOR,
                    "computed_gate": computed_gate,
                    "rounding_quantum": float(quantum),
                    "minimum_gate": float(quantum),
                    "selected_gate": selected_gate,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"optimizer raw variation: {worst_delta:.9e}", flush=True)
    print(f"optimizer selected gate: {selected_gate:.9e}", flush=True)
    return output


def _validation_outcome(returncode: int, output: str, report: dict | None) -> str:
    """Classify the frozen regression without mistaking process transport for a test result."""
    if _is_cuda_oom(RuntimeError(output)):
        return "oom"
    if returncode != 0:
        return "failed"
    if report is None:
        raise RuntimeError("final regression exited zero without writing report.json")
    totals = report.get("totals") or {}
    if int(totals.get("pass", 0)) <= 0:
        raise RuntimeError("final regression exited zero without a passing test result")
    if int(totals.get("fail", 0)) != 0:
        raise RuntimeError("final regression exited zero with failing test results")
    return "pass"


def _run_final_regression(args: argparse.Namespace, selected: dict, *, phase: str = "phase 3/3") -> dict:
    """Run the frozen config exactly as future regression runs will execute it."""
    layers = int(selected["layers"])
    out_dir = args.output_dir / "validation" / f"{layers}-layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    command = [
        sys.executable,
        "-m",
        "arctic_platform.correctness",
        "run",
        "--config",
        str(args.config),
        "--test",
        str(args.test_id),
        "--out",
        str(out_dir),
    ]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    print(f"{phase}: validating frozen {layers}-layer regression", flush=True)
    lines: list[str] = []
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=args.checkout_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        returncode = process.wait()
    output = "".join(lines)
    report_path = out_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    outcome = _validation_outcome(returncode, output, report)
    result = {
        "layers": layers,
        "status": outcome,
        "returncode": returncode,
        "log": str(log_path),
        "report": str(report_path) if report_path.exists() else None,
    }
    (out_dir / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


FIXED_TOLERANCE = "fixed-tolerance"
# The checks whose gate is drawn from repeated reference executions rather than stated as a constant.
CALIBRATED_TEST_IDS = ("single-step-grads", "single-step-optimizer")


def _onboarding_kind(test_id: str) -> str:
    """Which onboarding a test id takes, or a refusal for an id that has none.

    The gradient and optimizer checks each measure a reference spread. Checkpoint-resume measures a Arctic Platform
    loss range with determinism off and writes that gate. Every other check holds its gate as a constant
    in its own module, so there is nothing to size beyond the reduced model the spec already names.
    """
    from arctic_platform.correctness.harness.registry import registered_tests

    if test_id == "checkpoint-resume-loss" or test_id in CALIBRATED_TEST_IDS:
        return test_id
    if registered_tests().get(test_id) is not None:
        return FIXED_TOLERANCE
    raise ValueError(f"onboarding does not support test id {test_id!r}")


def _onboard_checkpoint_resume(args: argparse.Namespace, config) -> int:
    """Measure three undeterministic 10-step runs, write twice that range, then run the check.

    The measurement turns determinism off. The validation run is the ordinary regression, which keeps
    best-effort determinism and reads the gate just written. A failed validation restores the spec.
    """
    from transformers import AutoConfig

    from arctic_platform.correctness.checks.checkpoint_resume import TEST_ID
    from arctic_platform.correctness.checks.checkpoint_resume import loss_range_gate
    from arctic_platform.correctness.checks.checkpoint_resume import measure_undeterministic_loss_range
    from arctic_platform.correctness.harness.arms import arms_for
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.spec import TestSpec

    test_id = args.test_id
    if test_id != TEST_ID:
        raise ValueError(f"{test_id} is not the checkpoint-resume check")
    if not args.test_spec.exists():
        raise ValueError(
            f"{test_id} reads the reduced model an onboarded config already has, so it requires an "
            f"existing reviewed spec at {args.test_spec}; onboard single-step-grads for "
            f"{config.config_id} first"
        )
    previous_spec = args.test_spec.read_bytes()
    spec = TestSpec.read(args.test_spec)
    if spec.config_id != config.config_id:
        raise ValueError(f"spec {spec.config_id!r} does not describe config {config.config_id!r}")
    current_checksum = config_checksum(config)
    if spec.config_checksum != current_checksum:
        raise ValueError(
            f"{config.config_id}: the reviewed spec records training config checksum "
            f"{spec.config_checksum[:12] or 'missing'} and the config hashes to "
            f"{current_checksum[:12]}. This path writes one tolerance and changes nothing else, so it "
            "neither re-stamps the checksum nor runs a check against a configuration nobody reviewed; "
            f"onboard single-step-grads again for {args.config} first"
        )
    arms = list(spec.arms) if config.is_rl else arms_for(config.max_seq_len, config.n_gpus)
    model_cfg = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab_size = int(getattr(model_cfg, "text_config", model_cfg).vocab_size)
    print("phase 1/2: three 10-step runs with determinism off, per case", flush=True)
    raw_range, where = measure_undeterministic_loss_range(
        config.training,
        spec.model.cache_path,
        slots=config.n_gpus,
        vocab_size=vocab_size,
        arms=arms,
        workdir=args.output_dir / "checkpoint-resume-calibration",
        attn_implementation=config.attention_implementation,
    )
    gate = loss_range_gate(raw_range, where)
    print(
        f"phase 1/2 complete: raw_range={raw_range} worst={where} "
        f"computed={gate.computed_gate} absolute={gate.absolute}",
        flush=True,
    )
    layers = int(spec.model.num_hidden_layers)
    spec.test_tolerances[test_id] = gate
    spec.applicable_tests = sorted({*spec.applicable_tests, test_id})
    spec.write(args.test_spec)
    try:
        validation = _run_final_regression(args, {"layers": layers}, phase="phase 2/2")
        if validation["status"] != "pass":
            raise RuntimeError(f"{test_id} {validation['status']} for {config.config_id}; see {validation['log']}")
    except BaseException:
        args.test_spec.write_bytes(previous_spec)
        raise
    manifest = {
        "config_id": config.config_id,
        "config_path": str(args.config),
        "test_id": test_id,
        "raw_loss_range": raw_range,
        "worst": where,
        "multiplier": gate.multiplier,
        "computed_gate": gate.computed_gate,
        "absolute": gate.absolute,
        "model_cache_path": spec.model.cache_path,
        "num_hidden_layers": layers,
        "applicable_tests": spec.applicable_tests,
        "test_spec": str(args.test_spec),
        "validation_attempts": [validation],
        "validated_report": validation["report"],
    }
    manifest_path = args.output_dir / f"onboarding-{test_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"onboarded {config.config_id} {test_id}: absolute={gate.absolute} validation=pass",
        flush=True,
    )
    print(f"spec: {args.test_spec}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"validated report: {validation['report']}", flush=True)
    return 0


def _onboard_fixed_tolerance_test(args: argparse.Namespace, config) -> int:
    """Add a fixed-gate check to a config that already has a reviewed spec.

    Nothing is sized and nothing is calibrated. The gate is a constant in the check's own module, so no
    tolerance is measured and none is written; the check reads the reduced model the spec already names,
    so no model is materialized and nothing is written to ``test_settings``. What remains is to run the
    check the way regression will run it and, on a pass, list it in ``applicable_tests``.
    """
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.registry import registered_tests
    from arctic_platform.correctness.harness.spec import TestSpec

    test_id = args.test_id
    entry = registered_tests()[test_id]
    if entry.requires_hosted_control_plane:
        raise ValueError(
            f"{test_id} creates jobs only the hosted control plane can create, so onboarding cannot run "
            "it here and must not publish a verdict it did not measure. The hosted command runs the "
            "checks named on its command line whether or not a spec lists them as applicable, so this "
            "check needs no entry in applicable_tests:\n"
            f"  python -m arctic_platform.correctness hosted --config {args.config} --test {test_id} "
            "--host <account-host> --database <database> --schema <schema> --pat-file <file>"
        )
    if not args.test_spec.exists():
        raise ValueError(
            f"{test_id} reads the reduced model an onboarded config already has, so it requires an "
            f"existing reviewed spec at {args.test_spec}; onboard single-step-grads for "
            f"{config.config_id} first"
        )
    previous_spec = args.test_spec.read_bytes()
    spec = TestSpec.read(args.test_spec)
    if spec.config_id != config.config_id:
        raise ValueError(f"spec {spec.config_id!r} does not describe config {config.config_id!r}")
    current_checksum = config_checksum(config)
    if spec.config_checksum != current_checksum:
        raise ValueError(
            f"{config.config_id}: the reviewed spec records training config checksum "
            f"{spec.config_checksum[:12] or 'missing'} and the config hashes to "
            f"{current_checksum[:12]}. This path publishes one test id and changes nothing else, so it "
            "neither re-stamps the checksum nor runs a check against a configuration nobody reviewed; "
            f"onboard single-step-grads again for {args.config} first"
        )

    layers = int(spec.model.num_hidden_layers)
    already_listed = test_id in spec.applicable_tests
    print(
        f"phase 1/1: {test_id} is gated by a constant in {entry.fn.__module__}, so there is nothing to "
        f"size and nothing to calibrate; running it against the frozen {layers}-layer "
        f"{spec.model.cache_path}",
        flush=True,
    )
    spec.applicable_tests = sorted({*spec.applicable_tests, test_id})
    spec.write(args.test_spec)
    try:
        validation = _run_final_regression(args, {"layers": layers}, phase="phase 1/1")
        if validation["status"] != "pass":
            raise RuntimeError(f"{test_id} {validation['status']} for {config.config_id}; see {validation['log']}")
    except BaseException:
        args.test_spec.write_bytes(previous_spec)
        raise

    manifest = {
        "config_id": config.config_id,
        "config_path": str(args.config),
        "test_id": test_id,
        "tolerance": f"fixed constant in {entry.fn.__module__}; not calibrated and not written to the spec",
        "model_cache_path": spec.model.cache_path,
        "num_hidden_layers": layers,
        "applicable_tests": spec.applicable_tests,
        "already_listed_before_this_run": already_listed,
        "test_spec": str(args.test_spec),
        "validation_attempts": [validation],
        "validated_report": validation["report"],
    }
    manifest_path = args.output_dir / f"onboarding-{test_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"onboarded {config.config_id} {test_id}: validation=pass, "
        f"applicable_tests={', '.join(spec.applicable_tests)}",
        flush=True,
    )
    print(f"spec: {args.test_spec}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"validated report: {validation['report']}", flush=True)
    return 0


def _onboard_optimizer_test(args: argparse.Namespace, config) -> int:
    """Add Test 2 with its own reduced reference while preserving Test 1 and its materialized cases."""
    from arctic_platform.correctness.harness.spec import TestSpec
    from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained

    if not args.test_spec.exists():
        raise ValueError("Test 2 requires an existing Test 1 spec so both tests reuse the same inputs")
    previous_spec = args.test_spec.read_bytes()
    spec = TestSpec.read(args.test_spec)
    if spec.config_id != config.config_id:
        raise ValueError(f"spec {spec.config_id!r} does not describe config {config.config_id!r}")
    optimizer_model_path, optimizer_layers = _optimizer_reference_model(spec, args.cache_root)
    args.optimizer_model_path = optimizer_model_path
    args.optimizer_layers = optimizer_layers
    print(
        "phase 1/3: finding the HF reference configuration; "
        f"using the minimum representative {optimizer_layers}-layer prefix and the frozen Test 1 cases",
        flush=True,
    )
    materialize_pretrained(
        spec.model.source_checkpoint,
        str(optimizer_model_path),
        optimizer_layers,
        vision_depth=spec.model.vision_depth,
    )
    print(f"phase 1/3 complete: reference={optimizer_model_path}", flush=True)
    print(
        f"phase 2/3: finding optimizer tolerance from {CALIBRATION_RUNS} nondeterministic HF reference steps; "
        "the progress bar will report ETA after each run",
        flush=True,
    )
    try:
        calibration_json = _run_optimizer_calibration(args)
        selected = {"layers": optimizer_layers}
        validation = _run_final_regression(args, selected)
        if validation["status"] != "pass":
            raise RuntimeError(
                f"final {optimizer_layers}-layer Test 2 regression {validation['status']}; see {validation['log']}"
            )
    except BaseException:
        args.test_spec.write_bytes(previous_spec)
        raise

    calibration = json.loads(calibration_json.read_text())
    manifest = {
        "config_id": config.config_id,
        "config_path": str(args.config),
        "test_id": "single-step-optimizer",
        "reference_layers": optimizer_layers,
        "reference_model_path": str(optimizer_model_path),
        "reference_token_budget": args.reference_token_budget,
        "test_spec": str(args.test_spec),
        "calibration": str(calibration_json),
        "selected_gate": calibration["tolerance"]["selected_gate"],
        "validation_attempts": [validation],
        "validated_report": validation["report"],
    }
    manifest_path = args.output_dir / "onboarding-t02.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"onboarded {config.config_id} Test 2: layers={optimizer_layers} "
        f"gate={manifest['selected_gate']:.9e} validation=pass",
        flush=True,
    )
    print(f"spec: {args.test_spec}", flush=True)
    print(f"validated report: {validation['report']}", flush=True)
    return 0


def _parent(args: argparse.Namespace) -> int:
    args.checkout_root = _checkout_root(args.config)
    sys.path.insert(0, str(args.checkout_root))
    args.cache_root = args.cache_root.resolve()

    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.config import validate_against_host
    from arctic_platform.correctness.harness.dss_driver import available_gpus
    from arctic_platform.correctness.harness.preflight import ensure_no_interrupted_builds
    from arctic_platform.correctness.harness.workdir import correctness_workdir

    config = load_config(args.config)
    if config.is_rl:
        from arctic_platform.correctness.harness.registry import RL_INAPPLICABLE_REASON
        from arctic_platform.correctness.harness.registry import registered_tests

        entry = registered_tests().get(args.test_id)
        if entry is None or entry.compares_to_reference:
            own = sorted(t.test_id for t in registered_tests().values() if not t.compares_to_reference)
            raise ValueError(
                f"{config.config_id}: {RL_INAPPLICABLE_REASON}. Onboarding {args.test_id} freezes a tolerance "
                "against that reference, so there is nothing to onboard for it; the checks that do not use "
                f"the reference are onboarded with --test-id: {', '.join(own)}"
            )
    args.source_checkpoint = _select_source_checkpoint(args.source_checkpoint, config)
    validate_against_host(config, available_gpus())
    ensure_no_interrupted_builds()
    from arctic_platform.correctness.harness.arms import GAS1_TOKEN_CAP
    from arctic_platform.correctness.harness.arms import correctness_microbatch_tokens
    from arctic_platform.correctness.harness.arms import max_sequence_tokens

    requested_tokens = config.max_seq_len if args.tokens is None else min(args.tokens, config.max_seq_len)
    args.tokens = max_sequence_tokens(requested_tokens, config.n_gpus)
    if args.output_dir is None:
        # Onboarding diagnostics are large and recreatable, so they belong on the node's fast disk rather than
        # in the shared checkout. Losing them costs nothing: the next onboarding writes them again.
        args.output_dir = correctness_workdir("config-onboarding-") / config.config_id
    else:
        args.output_dir = args.output_dir.resolve()
    if args.test_spec is None:
        args.test_spec = Path(__file__).resolve().parent.parent / "specs" / f"{config.config_id}.json"
    else:
        args.test_spec = args.test_spec.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_reference_budget = (
        args.reference_token_budget
        if args.reference_token_budget is not None
        else correctness_microbatch_tokens(config.max_tokens_per_mb)
    )
    args.reference_token_budget = min(requested_reference_budget, GAS1_TOKEN_CAP)
    kind = _onboarding_kind(args.test_id)
    if kind == "single-step-optimizer":
        return _onboard_optimizer_test(args, config)
    if kind == "checkpoint-resume-loss":
        return _onboard_checkpoint_resume(args, config)
    if kind == FIXED_TOLERANCE:
        return _onboard_fixed_tolerance_test(args, config)

    full_layers, period, representative_layers = _model_shape(str(args.source_checkpoint))
    sizing_started = time.monotonic()
    print(
        f"phase 1/3: validating the minimum representative HF configuration at {args.tokens:,} tokens; "
        f"selected {representative_layers} of {full_layers} source layers, attention period {period}",
        flush=True,
    )
    selected = _run_probe(args, representative_layers)
    observations = [selected]
    if selected["status"] == "oom":
        raise RuntimeError(f"the minimum representative configuration ({representative_layers} layers) OOMs")
    if selected["status"] != "fit":
        raise RuntimeError(f"unexpected sizing status {selected['status']!r} for {representative_layers} layers")
    print(
        f"phase 1/3 complete: selected={selected['layers']} layers "
        f"peak={selected['peak_gib']:.3f} GiB elapsed={time.monotonic() - sizing_started:.1f}s",
        flush=True,
    )

    previous_spec = args.test_spec.read_bytes() if args.test_spec.exists() else None
    try:
        _write_initial_spec(args, selected, None)
        print(f"phase 2/3: finding tolerance from {CALIBRATION_RUNS} nondeterministic HF reference runs", flush=True)
        calibration_json = _render_and_run_calibration(args, selected)
        calibration = json.loads(calibration_json.read_text())
        validation = _run_final_regression(args, selected)
        if validation["status"] != "pass":
            raise RuntimeError(
                f"final {selected['layers']}-layer regression {validation['status']}; see {validation['log']}"
            )
    except BaseException:
        if previous_spec is None:
            args.test_spec.unlink(missing_ok=True)
        else:
            args.test_spec.write_bytes(previous_spec)
        raise

    manifest = {
        "config_id": config.config_id,
        "config_path": str(args.config),
        "architecture_period": period,
        "source_layers": full_layers,
        "sizing_policy": "minimum prefix containing every attention-layer type, with a four-layer floor",
        "sizing_observations": observations,
        "selected_layers": selected["layers"],
        "sizing_reference_attention": config.attention_implementation,
        "calibration_reference_attention": config.attention_implementation,
        "first_oom_layers": None,
        "test_spec": str(args.test_spec),
        "calibration": str(calibration_json),
        "selected_gate": calibration["tolerance"]["selected_gate"],
        "validation_attempts": [validation],
        "validated_report": validation["report"],
    }
    manifest_path = args.output_dir / "onboarding.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"onboarded {config.config_id}: layers={selected['layers']} "
        f"peak={selected['peak_gib']:.3f} GiB gate={manifest['selected_gate']:.9e} "
        "validation=pass",
        flush=True,
    )
    print(f"spec: {args.test_spec}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"validated report: {validation['report']}", flush=True)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint",
        help="override the training config model_name with a local path or another Hub model",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("/data-fast/base-models/synthetic"))
    parser.add_argument("--test-spec", type=Path)
    parser.add_argument("--test-id", default="single-step-grads")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--reference-token-budget", type=int)
    parser.add_argument("--probe-layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)


def run(args: argparse.Namespace) -> int:
    args.config = args.config.resolve()
    if args.probe_layers is not None:
        if args.probe_output is None:
            raise ValueError("sizing worker requires --probe-output")
        args.checkout_root = _checkout_root(args.config)
        sys.path.insert(0, str(args.checkout_root))
        args.cache_root = args.cache_root.resolve()
        args.probe_output = args.probe_output.resolve()
        return _probe_worker(args)
    return _parent(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Size, calibrate, and freeze one config-specific correctness test",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
