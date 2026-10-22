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
import ast
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from arctic_platform.correctness.harness.config import DEFAULT_REFERENCE_CALIBRATION_RUNS
from arctic_platform.correctness.harness.storage import CORRECTNESS_LOGDIR_ENV
from arctic_platform.correctness.harness.storage import correctness_logdir
from arctic_platform.correctness.harness.storage import correctness_workdir
from arctic_platform.correctness.harness.storage import require_data_fast_path

CALIBRATION_RUNS = DEFAULT_REFERENCE_CALIBRATION_RUNS
# How far above the run-to-run variation the calibrated gate sits. Sixteen calibration runs measure the
# tail imperfectly: on the CPU-offloaded Qwen3.8-27B config a factor of two produced a 2e-03 gate that the
# validation run then exceeded at 2.929e-03, so a factor that only just covers the observed spread does not
# survive the next run. Four keeps that measurement inside the gate with room, and stays far below the
# disagreement a real defect produces -- a dropped microbatch gradient is a fraction of the correct norm,
# not a per-mille difference.
GATE_SAFETY_FACTOR = 4.0

OOM_EXIT = 42


def _stage(phase: str, name: str, detail: str = "") -> None:
    marker = "###" if phase.split()[0].casefold() == "onboarding" else "***"
    print(f"\n{marker} {phase.title()}: {name.title()} {marker}", flush=True)
    if detail:
        print(detail, flush=True)
    print(flush=True)


class _ConsoleLog:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def _onboarding_failure(reason: str) -> None:
    print(f"\n\n!!!!! ONBOARDING FAILED: {reason.upper()} !!!!!\n\n", file=sys.stderr, flush=True)


def logged_run(args: argparse.Namespace) -> int:
    """Retain diagnostics even when prerequisites or specification loading fail."""
    from arctic_platform.correctness.harness.report import total_duration

    if getattr(args, "probe_layers", None) is not None:
        return run(args)
    started = time.monotonic()
    previous_root = os.environ.get(CORRECTNESS_LOGDIR_ENV)
    args._resume_root = Path(previous_root or Path.cwd() / "work" / "logs") / "onboarding-state"
    args._diagnostics_dir = correctness_logdir("onboarding-")
    log_path = args._diagnostics_dir / "onboarding.log"
    os.environ[CORRECTNESS_LOGDIR_ENV] = str(args._diagnostics_dir)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            stdout = _ConsoleLog(sys.stdout, log)
            stderr = _ConsoleLog(sys.stderr, log)
            handler = logging.StreamHandler(stderr)
            logger = logging.getLogger()
            previous_handlers = logger.handlers[:]
            logger.handlers = [handler]
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    print(f"Onboarding log: {log_path}", flush=True)
                    try:
                        return run(args)
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        traceback.print_exc()
                        (args._diagnostics_dir / "report.md").write_text(
                            f"# Onboarding Failed\n\nConfig: `{args.config}`\n\n"
                            f"## Reason\n\n{reason}\n\nComplete traceback: [onboarding.log](onboarding.log)\n",
                            encoding="utf-8",
                        )
                        _onboarding_failure(reason)
                        print(f"Report: {args._diagnostics_dir / 'report.md'}", file=sys.stderr, flush=True)
                        return 1
                    finally:
                        print(f"Total run time: {total_duration(time.monotonic() - started)}", flush=True)
            finally:
                logger.handlers = previous_handlers
                handler.close()
    finally:
        if previous_root is None:
            os.environ.pop(CORRECTNESS_LOGDIR_ENV, None)
        else:
            os.environ[CORRECTNESS_LOGDIR_ENV] = previous_root


def _stream_subprocess(command, *, cwd, env) -> int:
    """Stream child output through the console/log tee, including native-library output."""
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()


def _configure_fixture_target(args: argparse.Namespace, config, selection: list[str]) -> None:
    if getattr(args, "backend", None) == "cortex-training":
        args.backend = "cortex"
    if (
        getattr(args, "prepare_references", False)
        or getattr(args, "probe_layers", None) is not None
        or getattr(args, "backend", "onprem") not in {"cortex", "cortex-training"}
        or config.is_rl
        or not set(selection).intersection(CALIBRATED_TEST_IDS)
    ):
        return
    from arctic_platform.correctness.harness.execution import _client_backend
    from arctic_platform.correctness.harness.hosted import require_local_cortex_gateway

    configured = any(
        getattr(args, name, None) or os.environ.get(f"ARCTIC_CORTEX_{name.upper()}") for name in ("base_url", "host")
    )
    if not configured:
        args._automatic_cortex_gateway = True
        # This URL is only a configuration placeholder; validation supplies the owned server's port.
        args.base_url = "http://127.0.0.1:1"
    try:
        require_local_cortex_gateway(_client_backend(args))
    except ValueError as exc:
        raise ValueError(
            "Reduced-model onboarding requires an unauthenticated HTTP loopback Cortex-training gateway. "
            "Remove hosted ARCTIC_CORTEX_HOST/PAT settings to use the automatically managed local gateway, "
            "or pass --base-url for an existing local gateway. Hosted reduced-model onboarding is unsupported."
        ) from exc
    if args.test_id in CALIBRATED_TEST_IDS:
        args.local_cortex_fixture = True


@contextmanager
def _validation_gateway(args: argparse.Namespace):
    """Use the installed DSS gateway, and shut down only the server this command started."""
    if not getattr(args, "_automatic_cortex_gateway", False):
        yield None
        return
    from arctic_platform.correctness.harness.dss_driver import local_cortex_gateway

    directory = getattr(args, "_diagnostics_dir", args.output_dir)
    with local_cortex_gateway(directory) as url:
        yield url


def _checkpoint_file_identity(path: Path | str) -> dict:
    """Rehash small resume artifacts even on filesystems with coarse modification timestamps."""
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = os.fstat(stream.fileno()).st_size
    return {"sha256": digest, "size": size}


def _resume_preparation_request(request: dict, root: Path) -> dict:
    """Exclude target transport and reviewed measurement-preserving bookkeeping changes."""
    preparation = {**request, "code": dict(request["code"])}
    preparation["code"].pop("harness/cortex.py", None)
    for relative in ("onboarding/onboard.py", "harness/runner.py"):
        if relative not in preparation["code"]:
            continue
        path = root / relative
        if _checkpoint_file_identity(path) != request["code"][relative]:
            raise ValueError(f"Onboarding source changed while recording retry state: {path}")
        tree = ast.parse(path.read_text())
        if relative == "onboarding/onboard.py":
            footer_nodes = {
                ast.dump(ast.parse(source).body[0])
                for source in (
                    "from arctic_platform.correctness.harness.report import duration",
                    "from arctic_platform.correctness.harness.report import total_duration",
                    "started = time.monotonic()",
                    'print(f"Total run time: {duration(time.monotonic() - started)}", flush=True)',
                    'print(f"Total run time: {total_duration(time.monotonic() - started)}", flush=True)',
                )
            }
            for function in tree.body:
                if isinstance(function, ast.FunctionDef) and function.name == "logged_run":
                    for node in ast.walk(function):
                        for field, value in ast.iter_fields(node):
                            if isinstance(value, list) and field in {"body", "orelse", "finalbody"}:
                                setattr(node, field, [item for item in value if ast.dump(item) not in footer_nodes])
            tree.body = [
                node
                for node in tree.body
                if not (isinstance(node, ast.FunctionDef) and node.name.startswith("_resume_"))
                and not (isinstance(node, ast.Import) and [alias.name for alias in node.names] == ["ast"])
            ]
        else:
            # These exact changes affect reuse/logging, not the measured HF recipe or gate.
            progress_context = ast.dump(
                ast.parse(
                    """with tqdm(
                        total=reference_calibration_runs,
                        desc=f"Optimizer calibration {arm.name}",
                        unit="run",
                        file=sys.stdout,
                        dynamic_ncols=True,
                        mininterval=0,
                        miniters=1,
                    ) as progress:
                        pass"""
                )
                .body[0]
                .items[0]
            )
            progress_update = ast.dump(ast.parse("progress.update(1)").body[0])
            progress_import = ast.dump(ast.parse("from tqdm.auto import tqdm").body[0])

            class CalibrationDisplay(ast.NodeTransformer):
                def visit_ImportFrom(self, node):
                    return None if ast.dump(node) == progress_import else node

                def visit_With(self, node):
                    node = self.generic_visit(node)
                    if len(node.items) == 1 and ast.dump(node.items[0]) == progress_context:
                        for child in ast.walk(node):
                            for field, value in ast.iter_fields(child):
                                if isinstance(value, list) and field in {"body", "orelse", "finalbody"}:
                                    setattr(
                                        child, field, [item for item in value if ast.dump(item) != progress_update]
                                    )
                        return node.body
                    return node

            tree = CalibrationDisplay().visit(tree)
            summaries = [
                """print(
                    f"\\n*** Optimizer Norm Summary: {arm.name} ***\\n"
                    f"Reading saved optimizer states from {len(manifests)} completed HF runs.",
                    flush=True,
                )""",
                "summary_started = time.monotonic()",
                """print(
                    f"Optimizer norm summary complete: arm={arm.name}, "
                    f"elapsed={time.monotonic() - summary_started:.1f}s, "
                    f"maximum_pairwise_norm_delta={spread:.9e}\\n",
                    flush=True,
                )""",
            ]
            ignored = {ast.dump(ast.parse(source).body[0]) for source in summaries}
            reuse = ast.dump(ast.parse('stepping and "single-step-optimizer" not in records', mode="eval").body)
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and ast.dump(node.test) == reuse:
                    node.test = ast.Name(id="stepping", ctx=ast.Load())
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list) and field in {"body", "orelse", "finalbody"}:
                        setattr(node, field, [item for item in value if ast.dump(item) not in ignored])
        preparation["code"][relative] = {
            "schema": "onboarding-preparation-ast-v1",
            "sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        }
    return preparation


def _resume_setup(args: argparse.Namespace) -> None:
    """Bind retained phase results to this config, options, sources and Python environment."""
    if not hasattr(args, "_resume_root"):
        return
    from arctic_platform.correctness.harness.reference_records import model_identity

    root = Path(__file__).resolve().parents[1]
    options = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
        if not name.startswith("_")
        and name
        not in {
            "config",
            "checkout_root",
            "output_dir",
            "test_spec",
            "base_url",
            "func",
        }
    }
    if not getattr(args, "_automatic_cortex_gateway", False):
        options["base_url"] = getattr(args, "base_url", None) or os.environ.get("ARCTIC_CORTEX_BASE_URL")
    versions = {}
    for name in ("torch", "transformers", "huggingface-hub", "peft", "deepspeed", "ray", "dss", "dss-client"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source = Path(args.source_checkpoint)
    request = {
        "config": _checkpoint_file_identity(args.config),
        "options": options,
        "python": sys.executable,
        "dependencies": versions,
        "source": model_identity(source, refresh=True) if source.is_dir() else args.source_checkpoint,
        "code": {
            str(path.relative_to(root)): _checkpoint_file_identity(path)
            for folder in ("onboarding", "harness", "reference", "checks")
            for path in sorted((root / folder).rglob("*.py"))
        },
    }
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    args._resume_dir = args._resume_root / args.test_spec.stem / args.test_id / digest
    args._resume_dir.mkdir(parents=True, exist_ok=True)
    preparation = _resume_preparation_request(request, root)
    manifest = {"schema": "onboarding-resume-request-v1", "request": request, "preparation_request": preparation}
    manifest_path = args._resume_dir / "request.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Conflicting onboarding retry request: {manifest_path}")
    else:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
    args._resume_candidates = []
    for candidate in sorted(args._resume_dir.parent.glob("*/request.json")):
        if candidate == manifest_path:
            continue
        try:
            saved = json.loads(candidate.read_text())
            original_digest = hashlib.sha256(json.dumps(saved["request"], sort_keys=True).encode()).hexdigest()
            if (
                saved["schema"] == manifest["schema"]
                and candidate.parent.name == original_digest
                and saved["preparation_request"] == preparation
            ):
                args._resume_candidates.append(candidate.parent)
        except (OSError, ValueError, KeyError, TypeError):
            continue


def _resume_phase(args: argparse.Namespace, phase: str, operation):
    """Reuse only completed phases whose input specification and retained artifacts still match."""
    if not hasattr(args, "_resume_dir"):
        return operation()
    from arctic_platform.correctness.harness.reference_records import model_identity

    checkpoint = args._resume_dir / f"{phase}.json"
    input_spec = _checkpoint_file_identity(args.test_spec) if args.test_spec.exists() else None
    candidates = [checkpoint]
    if phase != "validation":
        candidates.extend(path / f"{phase}.json" for path in getattr(args, "_resume_candidates", []))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            saved = json.loads(candidate.read_text())
            valid = phase in {"sizing", "fixture"} or saved["input_spec"] == input_spec
            valid = valid and all(
                _checkpoint_file_identity(path) == identity for path, identity in saved["files"].items()
            )
            valid = valid and all(
                model_identity(path, refresh=True) == identity for path, identity in saved["models"].items()
            )
            if valid:
                if saved["spec"] is not None:
                    args.test_spec.write_text(saved["spec"])
                _stage("Resume", phase.replace("-", " "), "Reusing the verified completed phase.")
                return saved["result"]
        except (OSError, ValueError, KeyError):
            pass
        print(f"Retained {phase} work is missing or changed; rebuilding this phase.", flush=True)
    # A rebuilt phase invalidates downstream results even if it happens to reproduce identical bytes.
    args._resume_candidates = []
    phases = ("sizing", "fixture", "calibration", "references", "validation")
    for name in phases[phases.index(phase) :]:
        (args._resume_dir / f"{name}.json").unlink(missing_ok=True)
    result = operation()
    if isinstance(result, int) and result != 0:
        return result
    if isinstance(result, dict) and result.get("status") not in {None, "fit", "pass"}:
        return result
    files = {}
    models = {}
    if isinstance(result, str) and Path(result).is_file():
        files[result] = _checkpoint_file_identity(result)
    if isinstance(result, dict) and result.get("model_path"):
        models[result["model_path"]] = model_identity(result["model_path"])
    if phase in {"references", "calibration", "validation"}:
        from arctic_platform.correctness.harness.reference_records import DEFAULT_RECORDS_DIR

        records = Path(args.reference_records_dir or DEFAULT_RECORDS_DIR) / args.test_spec.stem
        files.update({str(path): _checkpoint_file_identity(path) for path in records.rglob("*.json")})
        if args.test_spec.exists():
            spec = json.loads(args.test_spec.read_text())
            model_path = spec.get("model", {}).get("cache_path")
            if model_path:
                models[model_path] = model_identity(model_path)
    if phase == "validation" and isinstance(result, dict):
        files.update(
            {
                str(result[name]): _checkpoint_file_identity(result[name])
                for name in ("report", "log")
                if result.get(name)
            }
        )
    saved = {
        "input_spec": input_spec,
        "result": result,
        "files": files,
        "models": models,
        "spec": args.test_spec.read_text() if args.test_spec.exists() else None,
    }
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    temporary.replace(checkpoint)
    return result


def _validate_storage(args: argparse.Namespace) -> None:
    args.cache_root = require_data_fast_path(args.cache_root, label="onboarding model cache")
    if getattr(args, "output_dir", None) is not None:
        args.output_dir = require_data_fast_path(args.output_dir, label="onboarding work directory")


def _checkout_root(config_path: Path) -> Path:
    """Resolve the executing library's Git checkout, not the operator config's directory."""
    source_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source_root,
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


def _onboarding_preflight(args: argparse.Namespace, config, selection: list[str] | None = None) -> None:
    """Fail before fixture creation, sizing, calibration or adapter generation."""
    from arctic_platform.correctness.harness import storage
    from arctic_platform.correctness.harness.dss_driver import _allocation_hostfile
    from arctic_platform.correctness.harness.dss_driver import recover_local_cortex_gateway
    from arctic_platform.correctness.harness.registry import onboarding_inapplicable_reason
    from arctic_platform.correctness.harness.registry import registered_tests

    tests = registered_tests()
    if selection is None:
        requested = list(tests) if args.test_id == "all" else [args.test_id]
        selection = [
            test_id for test_id in requested if onboarding_inapplicable_reason(tests[test_id], config) is None
        ]
    if not selection:
        return
    _configure_fixture_target(args, config, selection)
    if getattr(args, "_prerequisites_checked", False):
        return
    if config.n_gpus < 1 or config.sp_size < 1 or config.n_gpus % config.sp_size:
        raise ValueError(f"invalid topology: sp_size={config.sp_size} must divide n_gpus={config.n_gpus}")
    reference_only = getattr(args, "prepare_references", False)
    target_local = not reference_only and (
        getattr(args, "backend", "onprem") == "onprem" or getattr(args, "_automatic_cortex_gateway", False)
    )
    reference = reference_only or any(tests[test_id].compares_to_reference for test_id in selection)
    # A first SFT spec requires the gradient fixture even when another check was requested.
    spec = args.test_spec or Path(__file__).resolve().parents[1] / "specs" / f"{config.config_id}.json"
    reference = reference or not config.is_rl and not Path(spec).exists()
    common = ["torch", "transformers", "huggingface_hub", "safetensors"]
    if config.training.get("peft_config"):
        common.append("peft")
    reference_modules = ["liger_kernel.transformers.fused_linear_cross_entropy"] if reference else []
    target_modules = ["deepspeed", "ray"] if target_local else []
    if getattr(args, "_automatic_cortex_gateway", False):
        target_modules.append("dss.ray_dss.gateway_server")
    if target_local and (config.is_rl or "inference-checkpoint-loss" in selection):
        target_modules += ["arctic_inference", "vllm"]
    attention = []
    if reference:
        attention.append(config.reference_attention_implementation)
    if target_local:
        attention.append(config.attention_implementation)
    if reference_only:
        from arctic_platform.correctness.harness.config import TestSpec

        source = TestSpec.read(spec).model.cache_path
    else:
        source = _select_source_checkpoint(getattr(args, "source_checkpoint", None), config)
    writable = {
        "onboarding model cache": str(args.cache_root),
        "onboarding scratch": str(
            args.output_dir or os.environ.get(storage.CORRECTNESS_TMPDIR_ENV) or "/data-fast/tmp"
        ),
    }
    request = {
        "modules": common + reference_modules + target_modules,
        "attention": sorted(set(attention)),
        "hf_attention": [config.reference_attention_implementation] if reference else [],
        "min_gpus": max(1, getattr(args, "reference_calibration_workers", 1)) if reference else int(target_local),
        "gpu_type": config.gpu_type if target_local else None,
        "source": source if Path(source).is_absolute() else None,
        "model_id": source,
        "writable": {
            **writable,
            "specification output": str(Path(spec).parent),
            "reference records": str(
                getattr(args, "reference_records_dir", None) or Path(__file__).resolve().parents[1] / "reference_data"
            ),
        },
    }
    _stage(
        "Prerequisites",
        "Environment And Target",
        f"Config: {config.config_id}; checks: {', '.join(selection)}; Python: {sys.executable}\n"
        f"Config file: {args.config.resolve()}\nModel source: {source}\n"
        f"Configured sequence length: {config.max_seq_len:,} tokens",
    )
    if getattr(args, "_automatic_cortex_gateway", False):
        recover_local_cortex_gateway()
    local = storage.onboarding_node_prerequisites(request)
    failures = [f"{local['host']} ({local['python']}): {error}" for error in local["errors"]]
    if failures:
        raise ValueError("onboarding prerequisites failed before model work:\n  " + "\n  ".join(failures))
    required_gpus = config.n_gpus
    if target_local and config.is_rl:
        from arctic_platform.correctness.harness.dss_driver import rl_gpu_count

        required_gpus = rl_gpu_count(config)
    capacity = local["gpu_count"]
    if target_local and required_gpus > local["gpu_count"]:
        hostfile = _allocation_hostfile()
        if hostfile is None and Path("/job/hostfile").is_file():
            hostfile = Path("/job/hostfile")
        if hostfile is None:
            failures.append(
                f"config requires {required_gpus} GPUs, only {local['gpu_count']} are local and no allocation "
                "hostfile is available; set ARCTIC_HOSTFILE or DSS_GATEWAY_HOSTFILE to the same-job allocation"
            )
        else:
            peer_request = {
                **request,
                "modules": common + target_modules,
                "hf_attention": [],
                "attention": [config.attention_implementation],
                "min_gpus": 1,
                "source": None,
                "model_id": None,
                "writable": writable,
            }
            nodes = storage.allocation_onboarding_prerequisites(peer_request, hostfile)
            capacity = sum(entry.slots for entry in storage.parse_hostfile(hostfile))
            if capacity < required_gpus:
                failures.append(f"allocation hostfile supplies {capacity} GPUs, but config requires {required_gpus}")
            for host, node in nodes.items():
                failures += [f"{host} ({node['python']}): {error}" for error in node["errors"]]
                for name, version in local["versions"].items():
                    if name in peer_request["modules"] and node["versions"].get(name) != version:
                        failures.append(f"{host}: {name} version differs from the onboarding interpreter ({version})")
    if failures:
        raise ValueError("onboarding prerequisites failed before model work:\n  " + "\n  ".join(failures))
    if target_local:
        args._preflight_gpu_capacity = capacity
    args._prerequisites_checked = True
    scope = (
        "reference and target" if target_local else "local reference/fixture only; Cortex workers are not inspected"
    )
    print(f"preflight passed ({scope}); no attention backend was changed", flush=True)


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
    output = getattr(args, "_diagnostics_dir", args.output_dir) / "sizing" / f"{layers}-layers.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    _stage("Phase 1/3", "Sizing Probe", f"Layers: {layers}")
    returncode = _stream_subprocess(_probe_command(args, layers, output), cwd=args.checkout_root, env=env)
    if not output.exists():
        raise RuntimeError(f"{layers}-layer sizing worker exited {returncode} without writing {output}")
    result = json.loads(output.read_text())
    if returncode == 0 and result.get("status") == "fit":
        return result
    if returncode == OOM_EXIT and result.get("status") == "oom":
        return result
    raise RuntimeError(f"{layers}-layer sizing worker exited {returncode}: {result.get('error', 'no error recorded')}")


def _sizing_request(args: argparse.Namespace, layers: int) -> dict:
    """Describe the HF sizing inputs, excluding unrelated native startup code."""
    import torch
    import transformers

    from arctic_platform.correctness.harness.batches import SEED
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.reference_records import file_identity

    config = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((root / "reference").glob("*.py")),
        root / "onboarding" / "synth_model.py",
        *(root / "harness" / name for name in ("config.py", "batches.py", "sft.py", "names.py")),
    ]
    return {
        "config_sha256": file_identity(args.config)["sha256"],
        "source_checkpoint": str(args.source_checkpoint),
        "source_config_sha256": file_identity(Path(args.source_checkpoint) / "config.json")["sha256"],
        "layers": layers,
        "seed": SEED,
        "tokens": args.tokens,
        "reference_token_budget": args.reference_token_budget,
        "reference_attention_implementation": config.reference_attention_implementation,
        "interpreter": sys.executable,
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "probe_source_sha256": hashlib.sha256(inspect.getsource(_probe_worker).encode()).hexdigest(),
        "reference_sources": {str(path.relative_to(root)): file_identity(path)["sha256"] for path in sources},
    }


def _read_sizing_json(path: Path, sha256: str) -> dict:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("sizing reuse requires a full SHA256")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024:
        raise ValueError("sizing reuse JSON must be nonempty and at most 1 MiB")
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("sizing reuse JSON SHA256 mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("sizing reuse JSON must be an object")
    return value


def _reuse_sizing(args: argparse.Namespace, layers: int) -> dict:
    """Admit a reviewed, immutable sizing result before any calibration or model execution."""
    from transformers import AutoConfig

    from arctic_platform.correctness.harness.config import hash_directory
    from arctic_platform.correctness.harness.reference_records import model_identity
    from arctic_platform.correctness.onboarding.synth_model import count_parameters

    receipt = _read_sizing_json(args.reuse_sizing, args.reuse_sizing_sha256)
    if receipt.get("schema") != "correctness-sizing-reuse-v1":
        raise ValueError("unsupported sizing reuse schema")
    if receipt.get("request") != _sizing_request(args, layers):
        raise ValueError("sizing reuse workload/source/runtime mismatch")
    original = receipt["sizing_receipt"]
    selected = _read_sizing_json(Path(original["path"]), original["sha256"])
    if selected.get("status") != "fit" or type(selected.get("layers")) is not int or selected["layers"] != layers:
        raise ValueError("sizing reuse requires a successful fit at the original depth")
    peak = selected.get("peak_gib")
    if type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 0:
        raise ValueError("sizing reuse requires a finite positive peak")
    model_path = require_data_fast_path(selected["model_path"], label="reused sizing model")
    expected_path = args.cache_root / f"{Path(args.source_checkpoint).name}-{layers}L"
    if model_path != expected_path.resolve() or not (model_path / "config.json").is_file():
        raise ValueError("sizing reuse model path is missing or differs from the original cache")
    if not list(model_path.glob("*.safetensors")):
        raise ValueError("sizing reuse requires the retained model weights")
    if selected.get("model_hash") != hash_directory(model_path):
        raise ValueError("sizing reuse model identity mismatch")
    if receipt.get("model_identity") != model_identity(model_path, refresh=True):
        raise ValueError("sizing reuse model contents changed after admission")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(model_config, "text_config", model_config)
    if text_config.num_hidden_layers != layers:
        raise ValueError("sizing reuse model depth mismatch")
    if type(selected.get("param_count")) is not int or selected["param_count"] != count_parameters(model_config):
        raise ValueError("sizing reuse parameter count mismatch")
    return selected


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
    _validate_storage(args)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"

    import torch
    from transformers import AutoConfig

    from arctic_platform.correctness.harness.batches import SEED
    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.config import load_config
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
            attn_implementation=config.reference_attention_implementation,
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
    cache_root = require_data_fast_path(cache_root, label="onboarding model cache")
    if workdir is not None:
        workdir = require_data_fast_path(workdir, label="onboarding work directory")

    from transformers import AutoConfig

    from arctic_platform.correctness.harness.batches import SEED
    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.config import ArmSpec
    from arctic_platform.correctness.harness.config import TestSpec
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.config import correctness_microbatch_tokens
    from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained
    from arctic_platform.correctness.onboarding.synth_model import size_for_single_gpu

    temporary = workdir if workdir is not None else correctness_workdir("dss-onboard-")
    temporary.mkdir(parents=True, exist_ok=True)
    if num_layers is None:
        num_layers, _ = size_for_single_gpu(source_checkpoint)
    destination = str(cache_root / f"{Path(source_checkpoint).name}-{num_layers}L")
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
    from arctic_platform.correctness.harness.config import arms_for
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
        f"Single-GPU reference uses {config.reference_attention_implementation} with {selected['layers']} "
        f"complete-cycle layers and peaks at {selected['peak_gib']:.3f} GiB."
    )
    if first_oom is not None:
        sizing_note += f" The next complete cycle, {first_oom['layers']} layers, reaches CUDA OOM."
    spec.notes.append(sizing_note)
    spec.write(args.test_spec)


def _shared_adapter(args: argparse.Namespace, config, model_path: Path | None = None) -> Path | None:
    from arctic_platform.correctness.harness.shared_lora_adapter import adapter_identity

    path = getattr(args, "peft_adapter", None)
    if not config.training.get("peft_config"):
        if path is not None:
            raise ValueError("--peft-adapter requires a LoRA training config")
        return None
    if path is None:
        from arctic_platform.correctness.harness.shared_lora_adapter import generate_adapter

        if model_path is None:
            return None
        generated = getattr(args, "_generated_shared_adapters", {})
        key = str(model_path.resolve())
        if key not in generated:
            generated[key] = generate_adapter(
                model_path, config.training["peft_config"], correctness_workdir("shared-peft-") / "adapter"
            )
        args._generated_shared_adapters = generated
        identity = adapter_identity(generated[key], config.training["peft_config"])
        identities = getattr(args, "_generated_shared_adapter_identities", {})
        if key in identities and identities[key] != identity:
            raise ValueError("shared adapter changed during onboarding")
        identities[key] = identity
        args._generated_shared_adapter_identities = identities
        return generated[key]
    path = require_data_fast_path(path, label="shared PEFT adapter").resolve(strict=True)
    identity = adapter_identity(path, config.training["peft_config"])
    expected = getattr(args, "_admitted_shared_adapter_identity", identity)
    if identity != expected:
        raise ValueError("shared adapter changed during onboarding")
    args._admitted_shared_adapter_identity = identity
    return path


def _render_and_run_calibration(args: argparse.Namespace, selected: dict) -> Path:
    from arctic_platform.correctness.harness.config import load_config

    from .calibration import render

    model_path = args.cache_root / f"{Path(args.source_checkpoint).name}-{selected['layers']}L"
    config = load_config(args.config)
    peft_adapter_path = _shared_adapter(args, config, model_path)
    calibration_json = getattr(args, "_diagnostics_dir", args.output_dir) / "reference-repeatability.json"
    calibration_script = args.output_dir / "reference_repeatability.py"
    render_args = argparse.Namespace(
        config=args.config,
        source_checkpoint=args.source_checkpoint,
        layers=int(selected["layers"]),
        model_path=model_path,
        output_json=calibration_json,
        output_script=calibration_script,
        work_dir=args.output_dir / "calibration-runs",
        test_spec=args.test_spec,
        test_id=args.test_id,
        tokens=args.tokens,
        reference_token_budget=args.reference_token_budget,
        peft_adapter_path=peft_adapter_path,
    )
    calibration_script.write_text(render(render_args))
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    command = [sys.executable, str(calibration_script)]
    returncode = _stream_subprocess(command, cwd=args.checkout_root, env=env)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    _shared_adapter(args, config, model_path)
    return calibration_json


def _run_optimizer_calibration(args: argparse.Namespace) -> Path:
    """Calibrate full-tensor optimizer deltas across a set of nondeterministic reference runs."""
    args.output_dir = require_data_fast_path(args.output_dir, label="optimizer calibration work directory")
    model_path = require_data_fast_path(args.optimizer_model_path, label="optimizer calibration model cache")

    from decimal import ROUND_CEILING
    from decimal import Decimal

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoConfig

    from arctic_platform.correctness.checks.optimizer_state import max_pairwise_optimizer_delta
    from arctic_platform.correctness.harness.batches import SEED
    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.batches import save
    from arctic_platform.correctness.harness.config import TestSpec
    from arctic_platform.correctness.harness.config import TestTolerance
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.runner import run_reference

    config = load_config(args.config)
    spec = TestSpec.read(args.test_spec)
    if spec.config_checksum != config_checksum(config):
        raise ValueError("the existing spec does not match the config; onboard Test 1 before Test 2")
    gas1 = next((arm for arm in spec.arms if arm.name == "gas1"), None)
    if gas1 is None:
        raise ValueError("the existing spec has no gas1 case")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = getattr(model_config, "text_config", model_config).vocab_size
    batch = build_batch(gas1.name, gas1.global_batch_size, gas1.max_seq_len, vocab_size, seed=SEED)
    batch_path = args.output_dir / "optimizer-calibration-batch.pt"
    save(batch, batch_path)
    training = config.effective_training
    optimizer_config = config.training.get("optimizer")
    if not isinstance(optimizer_config, dict):
        raise ValueError("optimizer calibration requires an explicit optimizer config")
    adapter_path = _shared_adapter(args, config, model_path)

    lanes = max(1, min(CALIBRATION_RUNS, torch.cuda.device_count()))
    calibration_root = correctness_workdir("optimizer-calibration-")
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
            attn=config.reference_attention_implementation,
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

    output = getattr(args, "_diagnostics_dir", args.output_dir) / "optimizer-reference-repeatability.json"
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
    if any(int(count) != 0 for name, count in totals.items() if name not in {"pass", "inapplicable"}) or any(
        result.get("outcome") not in {"pass", "inapplicable"} for result in report.get("results", [])
    ):
        raise RuntimeError("final regression exited zero with failing test results")
    return "pass"


def _require_validation_pass(validation: dict, description: str, *, config_id: str | None = None) -> None:
    if validation["status"] == "pass":
        return
    details = validation.get("reason") or "The target did not complete a passing comparison."
    context = f" for {config_id}" if config_id is not None else ""
    raise RuntimeError(
        f"{description} {validation['status']}{context}\n{details}\n"
        f"Report: {validation.get('report') or 'not written'}\nComplete log: {validation['log']}"
    )


def _optimizer_validation_bootstrap(args: argparse.Namespace) -> dict | None:
    """Admit the caller-owned child bootstrap without importing its native scope."""
    names = (
        "optimizer_validation_bootstrap",
        "optimizer_validation_bootstrap_sha256",
        "optimizer_validation_config",
        "optimizer_validation_config_sha256",
    )
    values = [getattr(args, name, None) for name in names]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("optimizer validation requires both bootstrap/config paths and their full SHA256 pins")
    if (
        getattr(args, "backend", "onprem") != "cortex"
        or not getattr(args, "local_cortex_fixture", False)
        or args.test_id != "single-step-optimizer"
        or getattr(args, "prepare_references", False)
        or getattr(args, "probe_layers", None) is not None
        or getattr(args, "local_gateway_full_model", False)
        or getattr(args, "reference_tests", None)
    ):
        raise ValueError("optimizer validation bootstrap requires only local Cortex Test 2 onboarding")
    admitted = {}
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    for role, value, digest in (
        ("bootstrap", values[0], values[1]),
        ("config", values[2], values[3]),
    ):
        path = Path(value)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"optimizer validation {role} requires a canonical absolute path and full SHA256")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 1 << 20:
            raise ValueError(f"optimizer validation {role} must be a bounded single-link regular file")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read((1 << 20) + 1)
            after = os.fstat(stream.fileno())
        identity = [getattr(before, field) for field in fields]
        if (
            any(
                [getattr(snapshot, field) for field in fields] != identity
                for snapshot in (opened, after, path.lstat())
            )
            or hashlib.sha256(raw).hexdigest() != digest
        ):
            raise ValueError(f"optimizer validation {role} changed or SHA256 differs")
        admitted[role] = {"path": str(path), "sha256": digest, "file_identity": identity}
    return admitted


def _run_final_regression(args: argparse.Namespace, selected: dict, *, phase: str = "phase 3/3") -> dict:
    """Run the frozen config exactly as future regression runs will execute it."""
    bootstrap = _optimizer_validation_bootstrap(args)
    if bootstrap != getattr(args, "_admitted_optimizer_validation_bootstrap", bootstrap):
        raise ValueError("optimizer validation bootstrap/config changed after pre-calibration admission")
    layers = int(selected["layers"])
    out_dir = getattr(args, "_diagnostics_dir", args.output_dir) / "validation" / f"{layers}-layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    command = [
        sys.executable,
        "-m",
        "arctic_platform.correctness",
        "run",
        "--backend",
        getattr(args, "backend", "onprem"),
        "--config",
        str(args.config),
        "--test",
        str(args.test_id),
        "--out",
        str(out_dir),
    ]
    if getattr(args, "test_spec", None) is not None:
        command += ["--test-spec", str(args.test_spec)]
    for name in ("host", "database", "schema", "endpoint", "base_url", "pat_file", "gpu_pool"):
        value = getattr(args, name, None)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if getattr(args, "local_gateway_full_model", False):
        command.append("--local-gateway-full-model")
    if getattr(args, "reference_records_dir", None) is not None:
        command += ["--reference-records-dir", str(args.reference_records_dir)]
    if getattr(args, "local_cortex_fixture", False):
        command.append("--local-cortex-fixture")
    if getattr(args, "inference_sampling_topology", None) is not None:
        command += ["--inference-sampling-topology", *map(str, args.inference_sampling_topology)]
    if getattr(args, "peft_adapter", None) is not None:
        command += ["--peft-adapter", str(args.peft_adapter)]
    if getattr(args, "inference_export_allocation", None) is not None:
        command += [
            "--inference-export-allocation",
            str(args.inference_export_allocation),
            "--inference-export-allocation-sha256",
            str(args.inference_export_allocation_sha256),
        ]
    if getattr(args, "inference_initial_model_dir", None) is not None:
        command += ["--inference-initial-model-dir", str(args.inference_initial_model_dir)]
    if bootstrap is not None:
        command = [
            sys.executable,
            "-B",
            "-u",
            bootstrap["bootstrap"]["path"],
            "--bridge-config",
            bootstrap["config"]["path"],
            "--config-sha",
            bootstrap["config"]["sha256"],
            "--",
            *command[3:],
        ]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    description = selected.get("model_description", f"frozen {layers}-layer regression")
    _stage(phase, "Target Validation", description)
    lines: list[str] = []
    with _validation_gateway(args) as gateway_url, log_path.open("w") as log:
        if gateway_url is not None:
            command += ["--base-url", gateway_url]
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
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                lines.append(line)
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if bootstrap is not None and _optimizer_validation_bootstrap(args) != bootstrap:
        raise ValueError("optimizer validation bootstrap/config changed during target validation")
    output = "".join(lines)
    report_path = out_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    failures = [
        f"{row.get('test_id', args.test_id)} / {row.get('arm') or '-'}: "
        f"{row.get('reason') or row.get('summary') or row.get('outcome', 'no verdict')}"
        for row in (report or {}).get("results", [])
        if row.get("outcome") not in {"pass", "inapplicable"}
    ]
    reason = "\n".join(failures)
    try:
        outcome = _validation_outcome(returncode, output, report)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\n{reason or 'No valid passing target report was produced.'}\n"
            f"Report: {report_path if report is not None else 'not written'}\nComplete log: {log_path}"
        ) from exc
    if outcome != "pass" and not reason:
        tail = "\n".join(line for line in output.splitlines()[-20:] if line.strip())
        reason = f"Target process exited {returncode} without a reported failure reason."
        if tail:
            reason += f"\nLast target output:\n{tail}"
    result = {
        "layers": layers,
        "status": outcome,
        "returncode": returncode,
        "log": str(log_path),
        "report": str(report_path) if report_path.exists() else None,
        "reason": reason,
    }
    if bootstrap is not None:
        result["optimizer_validation_bootstrap"] = bootstrap
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
    from arctic_platform.correctness.harness.config import TestSpec
    from arctic_platform.correctness.harness.config import arms_for
    from arctic_platform.correctness.harness.config import config_checksum

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
        _require_validation_pass(validation, test_id, config_id=config.config_id)
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


def _build_reference_free_rl_spec(args: argparse.Namespace, config):
    """Create an RL candidate without a supervised reference run or calibrated gate."""
    from arctic_platform.correctness.harness.config import arms_for
    from arctic_platform.correctness.harness.registry import RL_INAPPLICABLE_REASON
    from arctic_platform.correctness.harness.registry import registered_tests

    entry = registered_tests()[args.test_id]
    if not config.is_rl or entry.compares_to_reference or entry.requires_hosted_control_plane:
        raise ValueError("initial RL onboarding requires a reference-free, locally runnable check")
    canonical_spec = Path(__file__).resolve().parents[1] / "specs" / f"{config.config_id}.json"
    if args.test_spec.resolve() != canonical_spec:
        raise ValueError(
            f"initial RL onboarding must publish to {canonical_spec}; regression reads the canonical "
            "spec, not a custom --test-spec destination"
        )
    arm_defs = arms_for(config.max_seq_len, config.n_gpus)
    if args.tokens != arm_defs[0].max_seq_len:
        raise ValueError(
            f"initial RL onboarding requires the config-derived {arm_defs[0].max_seq_len}-token "
            "case width; a smaller --tokens value would not match the regression cases"
        )
    full_layers, period, representative_layers = _model_shape(str(args.source_checkpoint))
    spec = build_spec(
        config,
        str(args.source_checkpoint),
        str(args.cache_root),
        arm_defs,
        num_layers=representative_layers,
        workdir=args.output_dir / "onboard",
    )
    spec.applicable_tests = [args.test_id]
    spec.inapplicable = {
        test_id: RL_INAPPLICABLE_REASON for test_id, test in registered_tests().items() if test.compares_to_reference
    }
    spec.notes.append(
        f"Reference-free RL bootstrap materialized {representative_layers} of {full_layers} source "
        f"layers (attention period {period}) and generated the cases without HF training, "
        "single-GPU sizing or tolerance calibration. RL checks use the configured full model; "
        "this reduced fixture supplies the spec metadata, not a supervised rollout baseline."
    )
    return spec


def _onboard_fixed_tolerance_test(args: argparse.Namespace, config) -> int:
    """Validate a fixed-gate check, bootstrapping an RL spec when none exists.

    An existing reviewed model and its settings remain frozen. Initial RL onboarding materializes an
    explicit representative prefix and cases, with no reference sizing or calibration. Both paths use
    the check's own fixed criterion and publish applicability only after the ordinary regression passes.
    """
    from arctic_platform.correctness.harness.config import TestSpec
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.registry import registered_tests

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
    bootstrap = config.is_rl and not args.test_spec.exists()
    if bootstrap:
        previous_spec = None
        spec = _build_reference_free_rl_spec(args, config)
    elif not args.test_spec.exists():
        raise ValueError(
            f"{test_id} reads the reduced model an onboarded config already has, so it requires an "
            f"existing reviewed spec at {args.test_spec}; onboard single-step-grads for "
            f"{config.config_id} first"
        )
    else:
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
    already_listed = not bootstrap and test_id in spec.applicable_tests
    full_model = config.is_rl or test_id == "inference-checkpoint-loss"
    model_description = (
        f"configured full model {config.model_name}"
        if full_model
        else f"frozen {layers}-layer {spec.model.cache_path}"
    )
    print(
        f"phase 1/1: {test_id} is gated by a constant in {entry.fn.__module__}, so there is nothing to "
        f"size and nothing to calibrate; running it against the {model_description}",
        flush=True,
    )
    spec.applicable_tests = sorted({*spec.applicable_tests, test_id})
    try:
        spec.write(args.test_spec)
        from arctic_platform.correctness.harness.reference_records import CRITERIA

        if test_id in CRITERIA:
            _prepare_references(args)
        validation = _run_final_regression(
            args, {"layers": layers, "model_description": model_description}, phase="phase 1/1"
        )
        _require_validation_pass(validation, test_id, config_id=config.config_id)
    except BaseException:
        if previous_spec is None:
            if args.test_spec.exists():
                failed_spec = args.output_dir / f"unvalidated-rl-spec-{time.time_ns()}.json"
                shutil.move(str(args.test_spec), str(failed_spec))
        else:
            args.test_spec.write_bytes(previous_spec)
        raise

    manifest = {
        "config_id": config.config_id,
        "config_path": str(args.config),
        "test_id": test_id,
        "tolerance": f"fixed constant in {entry.fn.__module__}; not calibrated and not written to the spec",
        "model_cache_path": spec.model.cache_path,
        "num_hidden_layers": layers,
        "validated_model_scope": "configured-full-model" if full_model else "frozen-reference",
        "validated_model_name": config.model_name if full_model else spec.model.cache_path,
        "frozen_reference_num_hidden_layers": layers,
        "applicable_tests": spec.applicable_tests,
        "already_listed_before_this_run": already_listed,
        "bootstrapped_reference_free_rl": bootstrap,
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


def _run_local_optimizer_calibration(args: argparse.Namespace, config, spec) -> Path:
    """Publish fresh per-arm scalar-norm references, never a legacy full-tensor gate."""
    from transformers import AutoConfig

    from arctic_platform.correctness.harness.batches import SEED
    from arctic_platform.correctness.harness.batches import load as load_batch
    from arctic_platform.correctness.harness.config import ArmDefinition
    from arctic_platform.correctness.harness.config import TestTolerance
    from arctic_platform.correctness.harness.execution import _client_backend
    from arctic_platform.correctness.harness.reference_records import CRITERIA
    from arctic_platform.correctness.harness.reference_records import batch_identity
    from arctic_platform.correctness.harness.reference_records import file_identity
    from arctic_platform.correctness.harness.reference_records import load_record
    from arctic_platform.correctness.harness.reference_records import model_identity
    from arctic_platform.correctness.harness.reference_records import publish_record
    from arctic_platform.correctness.harness.reference_records import require_git_records_dir
    from arctic_platform.correctness.harness.reference_records import workload_identity
    from arctic_platform.correctness.harness.runner import OPTIMIZER_LEARNING_RATE
    from arctic_platform.correctness.harness.runner import execute
    from arctic_platform.correctness.reference.model_features import uses_mixer_packing

    test_id = "single-step-optimizer"
    criterion = CRITERIA[test_id]
    runs = args.reference_calibration_runs
    workers = args.reference_calibration_workers
    if runs != CALIBRATION_RUNS or not 1 <= workers <= 8:
        raise ValueError(
            f"local optimizer norm onboarding requires {CALIBRATION_RUNS} repeats per arm and 1 to 8 HF workers"
        )
    records_root = require_git_records_dir(args.reference_records_dir)
    records_root.mkdir(parents=True, exist_ok=True)
    model_config = AutoConfig.from_pretrained(args.optimizer_model_path, trust_remote_code=True)
    vocab_size = int(getattr(model_config, "text_config", model_config).vocab_size)
    arms = [ArmDefinition(arm.name, arm.global_batch_size, arm.max_seq_len) for arm in spec.arms]
    if len(arms) != 2 or {arm.name for arm in arms} != {"gas1", "gas4"}:
        raise ValueError("local optimizer norm onboarding requires frozen GAS1 and GAS4 input arms")
    spec.test_settings[test_id] = {
        "model_cache_path": str(args.optimizer_model_path),
        "num_hidden_layers": args.optimizer_layers,
        "reference_token_budget": args.reference_token_budget,
        "ce_chunk": 2048,
        "learning_rate": OPTIMIZER_LEARNING_RATE,
        "optimizer_dtype": config.optimizer_dtype,
        "calibration_metric": criterion,
    }
    spec.applicable_tests = list(dict.fromkeys([*spec.applicable_tests, test_id]))
    ctx = SimpleNamespace(config_id=config.config_id, cfg=config, spec=spec)
    arm_receipts = {}
    tolerances = {}
    # A new compact-record directory forces fresh norm calibration without replacing retained records.
    with tempfile.TemporaryDirectory(prefix=".optimizer-onboarding-", dir=records_root) as temporary:
        ctx.reference_records_dir = Path(temporary)
        results = execute(
            config,
            spec,
            arms,
            vocab_size=vocab_size,
            select=[test_id],
            reference_token_budget=args.reference_token_budget,
            workdir=args.output_dir,
            prepare_references=True,
            reference_records_dir=ctx.reference_records_dir,
            reference_calibration_runs=runs,
            reference_calibration_workers=workers,
            client_backend=_client_backend(args),
            peft_adapter=_shared_adapter(args, config),
        )
        if results:
            raise ValueError(
                "optimizer reference preparation returned target results instead of reference-only success"
            )
        paths = sorted((ctx.reference_records_dir / config.config_id / test_id).glob("*.json"))
        if len(paths) != len(arms):
            raise ValueError("optimizer norm publication requires exactly one compact reference for every frozen arm")
        expected_model = model_identity(args.optimizer_model_path)
        initialization = "external-model"
        if config.training.get("peft_config"):
            from arctic_platform.correctness.harness.shared_lora_adapter import adapter_identity

            initialization = adapter_identity(_shared_adapter(args, config, args.optimizer_model_path))
        training = config.effective_training
        expected_training = {
            "fp32_lm_head": bool(training.get("fp32_lm_head", False)),
            "fused_cross_entropy": config.fused_cross_entropy,
            "mixer_packing": uses_mixer_packing(str(args.optimizer_model_path)),
            "matmul_precision": training.get("matmul_precision", "highest"),
            "peft_config": training.get("peft_config"),
            "lm_head_token_chunk_size": config.lm_head_token_chunk_size,
            "lm_head_vocab_chunk_size": training.get("fused_lm_head_vocab_chunk_size", 8192),
            "deterministic": True,
        }
        validated = []
        for path in paths:
            identity = json.loads(path.read_text())["identity"]
            recipe = identity["recipe"]
            inputs = recipe["synthetic_inputs"]
            arm = next((arm for arm in arms if arm.name == inputs["batch"]["name"]), None)
            if arm is None or arm.name in arm_receipts:
                raise ValueError("optimizer reference has an unexpected or duplicate frozen input arm")
            expected_inputs = {
                "version": "private-cpu-generator-v1",
                "seed": SEED,
                "rows": arm.global_batch_size,
                "max_seq_len": arm.max_seq_len,
                "vocab_size": vocab_size,
                "batch": batch_identity(load_batch(args.output_dir / f"batch-{arm.name}.pt")),
            }
            expected_optimizer = {
                "config": config.training["optimizer"],
                "learning_rate": OPTIMIZER_LEARNING_RATE,
                "gradient_clipping": config.training.get("gradient_clipping"),
                "dtype": config.optimizer_dtype,
            }
            if (
                recipe.get("model") != expected_model
                or recipe.get("initialization") != initialization
                or inputs != expected_inputs
                or recipe.get("optimizer") != expected_optimizer
                or recipe.get("attention") != config.reference_attention_implementation
                or recipe.get("token_budget") != args.reference_token_budget
                or recipe.get("ce_chunk") != 2048
                or recipe.get("training") != expected_training
                or identity != workload_identity(ctx, test_id, recipe)
            ):
                raise ValueError("optimizer norm reference does not bind the current fixture, inputs and sources")
            measurements = load_record(ctx, identity)
            gate = measurements["optimizer_norm_gate"]
            provenance = gate["provenance"]
            if not isinstance(provenance, dict) or (
                provenance.get("policy") != "onboarding-norm-repeatability-4x-upward-1e-3-v1"
                or provenance.get("calibration_runs") != runs
                or provenance.get("multiplier") != GATE_SAFETY_FACTOR
                or provenance.get("minimum_gate") != 0.001
            ):
                raise ValueError(f"optimizer norm gate requires fresh criterion-specific {runs}-repeat provenance")
            spread = provenance.get("maximum_pairwise_norm_delta")
            if type(spread) not in {float, int} or not math.isfinite(spread) or spread < 0:
                raise ValueError("optimizer norm calibration spread must be finite and nonnegative")
            tolerance = TestTolerance(
                absolute=gate["absolute"],
                calibration_runs=runs,
                raw_max_same_tensor_range=spread,
                multiplier=GATE_SAFETY_FACTOR,
                computed_gate=GATE_SAFETY_FACTOR * spread,
                worst_tensor=provenance.get("worst_quantity"),
                status=f"reviewed {criterion}; per-arm scalar norm spread, not full-tensor equivalence",
            )
            tolerances[arm.name] = tolerance
            arm_receipts[arm.name] = {"gate": gate, "identity": identity}
            validated.append((arm.name, identity, measurements))
        # Check both arms before immutable publication; regression reads each record's own gate.
        ctx.reference_records_dir = records_root
        for name, identity, measurements in validated:
            path = publish_record(ctx, identity, measurements)
            arm_receipts[name]["record"] = str(path)
            arm_receipts[name]["record_identity"] = file_identity(path)
    selected = max(tolerances.values(), key=lambda tolerance: tolerance.absolute)
    spec.test_tolerances[test_id] = selected
    spec.test_settings[test_id]["norm_gates_by_arm"] = {
        name: receipt["gate"]["absolute"] for name, receipt in arm_receipts.items()
    }
    output = getattr(args, "_diagnostics_dir", args.output_dir) / "optimizer-reference-repeatability.json"
    output.write_text(
        json.dumps(
            {
                "criterion": criterion,
                "calibration_runs_per_arm": runs,
                "calibration_workers": workers,
                "arms": arm_receipts,
                "tolerance": {
                    "selected_gate": selected.absolute,
                    "scope": "maximum per-arm gate for spec metadata; regression uses each immutable record's gate",
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    spec.write(args.test_spec)
    return output


def _onboard_optimizer_test(args: argparse.Namespace, config) -> int:
    """Add Test 2 with its own reduced reference while preserving Test 1 and its materialized cases."""
    from arctic_platform.correctness.harness.config import TestSpec
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained

    if not args.test_spec.exists():
        raise ValueError("Test 2 requires an existing Test 1 spec so both tests reuse the same inputs")
    previous_spec = args.test_spec.read_bytes()
    spec = TestSpec.read(args.test_spec)
    if spec.config_id != config.config_id:
        raise ValueError(f"spec {spec.config_id!r} does not describe config {config.config_id!r}")
    if spec.config_checksum != config_checksum(config):
        raise ValueError("the existing spec does not match the config; onboard Test 1 before Test 2")
    args._admitted_optimizer_validation_bootstrap = _optimizer_validation_bootstrap(args)
    optimizer_model_path, optimizer_layers = _optimizer_reference_model(spec, args.cache_root)
    local_fixture = getattr(args, "local_cortex_fixture", False)
    if local_fixture and "single-step-grads" not in spec.applicable_tests:
        raise ValueError("local optimizer onboarding requires a reviewed Test 1 fixture and inputs")
    if local_fixture and (
        optimizer_model_path.resolve() != Path(spec.model.cache_path).resolve()
        or optimizer_layers != spec.model.num_hidden_layers
    ):
        raise ValueError(
            "local optimizer fixture must match the frozen Test 1 model; a separate fixture is unsupported"
        )
    args.optimizer_model_path = optimizer_model_path
    args.optimizer_layers = optimizer_layers
    _stage(
        "Phase 1/3",
        "Optimizer Reference Fixture",
        f"using the minimum representative {optimizer_layers}-layer prefix and the frozen Test 1 cases",
    )
    if not local_fixture:
        materialize_pretrained(
            spec.model.source_checkpoint,
            str(optimizer_model_path),
            optimizer_layers,
            vision_depth=spec.model.vision_depth,
        )
    print(f"phase 1/3 complete: reference={optimizer_model_path}", flush=True)
    _stage(
        "Phase 2/3",
        "Optimizer Calibration",
        f"Finding tolerance from {CALIBRATION_RUNS} nondeterministic HF reference steps; "
        "the progress bar will report ETA after each run",
    )
    try:
        calibration_json = Path(
            _resume_phase(args, "calibration", lambda: str(_run_local_optimizer_calibration(args, config, spec)))
        )
        selected = {"layers": optimizer_layers}
        validation = _resume_phase(args, "validation", lambda: _run_final_regression(args, selected))
        _require_validation_pass(validation, f"final {optimizer_layers}-layer Test 2 regression")
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
    if local_fixture:
        manifest["criterion"] = calibration["criterion"]
        manifest["reference_records_by_arm"] = calibration["arms"]
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
    _validate_storage(args)
    args.checkout_root = _checkout_root(args.config)
    sys.path.insert(0, str(args.checkout_root))

    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.config import validate_against_host
    from arctic_platform.correctness.harness.dss_driver import available_gpus
    from arctic_platform.correctness.harness.storage import ensure_no_interrupted_builds

    config = load_config(args.config)
    _configure_fixture_target(args, config, [args.test_id])
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
    local_fixture = getattr(args, "local_cortex_fixture", False)
    if local_fixture and (
        getattr(args, "backend", "onprem") != "cortex"
        or config.is_rl
        or getattr(args, "local_gateway_full_model", False)
        or args.test_id not in CALIBRATED_TEST_IDS
    ):
        raise ValueError("local Cortex fixture onboarding requires supervised Cortex gradients or optimizer")
    if getattr(args, "backend", "onprem") == "cortex":
        from arctic_platform.correctness.harness.dss_driver import rl_gpu_count
        from arctic_platform.correctness.harness.execution import _client_backend
        from arctic_platform.correctness.harness.hosted import require_local_cortex_gateway

        full_model = getattr(args, "local_gateway_full_model", False)
        rl_inference = config.is_rl and args.test_id == "inference-checkpoint-loss" and not full_model
        supervised_full_model = (
            not config.is_rl and full_model and args.test_id in {"inference-checkpoint-loss", "e2e-sft-train-validate"}
        )
        if not (rl_inference or supervised_full_model or local_fixture):
            raise ValueError(
                "Cortex onboarding requires local reference-free RL inference or "
                "a local full-model supervised inference/e2e check or an explicit local Cortex fixture"
            )
        require_local_cortex_gateway(_client_backend(args))
        required_gpus = rl_gpu_count(config) if config.is_rl else config.n_gpus
        pool = getattr(args, "gpu_pool", None) or required_gpus
        if pool < required_gpus:
            raise ValueError(f"GPU pool is smaller than the required {required_gpus} GPUs")
    else:
        _onboarding_preflight(args, config)
        validate_against_host(
            config, args._preflight_gpu_capacity if hasattr(args, "_preflight_gpu_capacity") else available_gpus()
        )
        ensure_no_interrupted_builds()
    _onboarding_preflight(args, config)
    from arctic_platform.correctness.harness.config import GAS1_TOKEN_CAP
    from arctic_platform.correctness.harness.config import correctness_microbatch_tokens
    from arctic_platform.correctness.harness.config import max_sequence_tokens

    requested_tokens = config.max_seq_len if args.tokens is None else min(args.tokens, config.max_seq_len)
    args.tokens = max_sequence_tokens(requested_tokens, config.n_gpus)
    if args.output_dir is None:
        args.output_dir = correctness_workdir("config-onboarding-") / config.config_id
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
    _resume_setup(args)
    kind = _onboarding_kind(args.test_id)
    if kind == "single-step-optimizer":
        return _onboard_optimizer_test(args, config)
    if kind == "checkpoint-resume-loss":
        return _onboard_checkpoint_resume(args, config)
    if kind == FIXED_TOLERANCE:
        return _onboard_fixed_tolerance_test(args, config)

    full_layers, period, representative_layers = _model_shape(str(args.source_checkpoint))
    sizing_started = time.monotonic()
    if getattr(args, "reuse_sizing", None) is not None:
        _stage("Phase 1/3", "Retained Sizing", "Admitting the source-bound result without HF sizing replay.")
        selected = _resume_phase(args, "sizing", lambda: _reuse_sizing(args, representative_layers))
    else:
        _stage(
            "Phase 1/3",
            "Representative Fixture Sizing",
            f"Validating the minimum representative HF configuration at {args.tokens:,} tokens; "
            f"selected {representative_layers} of {full_layers} source layers, attention period {period}",
        )
        selected = _resume_phase(args, "sizing", lambda: _run_probe(args, representative_layers))
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
        _resume_phase(args, "fixture", lambda: _write_initial_spec(args, selected, None))
        _stage(
            "Phase 2/3",
            "Gradient Calibration",
            f"Finding tolerance from {CALIBRATION_RUNS} nondeterministic HF reference runs per GAS arm.",
        )
        calibration_json = Path(
            _resume_phase(args, "calibration", lambda: str(_render_and_run_calibration(args, selected)))
        )
        calibration = json.loads(calibration_json.read_text())
        if local_fixture:
            _resume_phase(args, "references", lambda: _prepare_references(args))
        validation = _resume_phase(args, "validation", lambda: _run_final_regression(args, selected))
        _require_validation_pass(validation, f"final {selected['layers']}-layer regression")
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
        "sizing_reference_attention": config.reference_attention_implementation,
        "calibration_reference_attention": config.reference_attention_implementation,
        "first_oom_layers": None,
        "test_spec": str(args.test_spec),
        "calibration": str(calibration_json),
        "selected_gate": calibration["tolerance"]["selected_gate"],
        "validation_attempts": [validation],
        "validated_report": validation["report"],
    }
    if getattr(args, "reuse_sizing", None) is not None:
        manifest["reused_sizing"] = {
            "path": str(args.reuse_sizing),
            "sha256": args.reuse_sizing_sha256,
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
    from arctic_platform.correctness.harness.execution import add_connection_arguments

    add_connection_arguments(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint",
        help="override the training config model_name with a local path or another Hub model",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("/data-fast/base-models/synthetic"))
    parser.add_argument("--test-spec", type=Path)
    parser.add_argument(
        "--inference-sampling-topology",
        type=int,
        nargs=3,
        metavar=("GPUS", "TP", "PP"),
        help="explicit original Flash-Next16 sampling allocation, TP and PP for final inference validation",
    )
    parser.add_argument("--inference-export-allocation", type=Path, help="same-job local inference allocation JSON")
    parser.add_argument("--inference-export-allocation-sha256", help="full SHA256 of the allocation JSON bytes")
    parser.add_argument(
        "--inference-initial-model-dir",
        type=Path,
        help="already admitted absolute FAST model directory for initial local Cortex inference training only",
    )
    parser.add_argument(
        "--test-id",
        "--test",
        default="all",
        help="onboard one check, or all applicable checks (default); prerequisites are automatic",
    )
    parser.add_argument(
        "--reference-tests",
        nargs="+",
        help="prepare these HF checks together using the same workload; requires --prepare-references",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--reference-token-budget", type=int)
    parser.add_argument("--probe-layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--reuse-sizing",
        type=Path,
        help="reuse a reviewed source-bound sizing receipt, then continue calibration and validation",
    )
    parser.add_argument("--reuse-sizing-sha256", help="full SHA256 of the immutable sizing reuse receipt")
    parser.add_argument(
        "--prepare-references",
        action="store_true",
        help="prepare or reuse compact Git HF references from an existing frozen spec; run no target",
    )
    parser.add_argument(
        "--reference-records-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference_data",
    )
    parser.add_argument("--reference-calibration-runs", type=int, default=CALIBRATION_RUNS)
    parser.add_argument(
        "--peft-adapter",
        type=Path,
        help="optional standard PEFT adapter under /data-fast; otherwise generate upstream-default LoRA factors",
    )
    parser.add_argument("--reference-calibration-workers", type=int, default=1)
    parser.add_argument(
        "--local-gateway-full-model",
        action="store_true",
        help="use the admitted offline full model for E2E preparation or inference/e2e validation",
    )
    parser.add_argument(
        "--local-cortex-fixture",
        action="store_true",
        help="validate the reduced supervised gradient or optimizer fixture through a local DSS gateway",
    )
    parser.add_argument(
        "--optimizer-validation-bootstrap",
        type=Path,
        help="source-pinned caller bootstrap for final local Cortex optimizer validation only",
    )
    parser.add_argument("--optimizer-validation-bootstrap-sha256", help="full SHA256 of the caller bootstrap")
    parser.add_argument("--optimizer-validation-config", type=Path, help="private caller bootstrap configuration")
    parser.add_argument("--optimizer-validation-config-sha256", help="full SHA256 of the private caller configuration")


def _prepare_references(args: argparse.Namespace) -> int:
    """Prepare an existing workload without resizing models or rewriting historical gates."""
    from transformers import AutoConfig

    from arctic_platform.client import CortexConfig
    from arctic_platform.correctness.harness.config import ArmDefinition
    from arctic_platform.correctness.harness.config import TestSpec
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.execution import _client_backend
    from arctic_platform.correctness.harness.reference_records import CRITERIA
    from arctic_platform.correctness.harness.reference_records import require_git_records_dir
    from arctic_platform.correctness.harness.runner import execute

    selection = args.reference_tests or [args.test_id]
    if any(test_id not in CRITERIA for test_id in selection):
        raise ValueError("--prepare-references supports only the three HF single-GPU checks")
    args.reference_records_dir = require_git_records_dir(args.reference_records_dir)
    config = load_config(args.config)
    spec_path = args.test_spec or Path(__file__).resolve().parents[1] / "specs" / f"{config.config_id}.json"
    spec = TestSpec.read(spec_path)
    if config.is_rl or spec.config_checksum != config_checksum(config) or spec.config_id != config.config_id:
        raise ValueError("reference preparation requires the matching reviewed SFT config and frozen spec")
    for test_id in selection:
        if test_id not in spec.applicable_tests:
            raise ValueError(f"{test_id} is not applicable to {config.config_id}")
    if args.local_gateway_full_model and selection != ["e2e-sft-train-validate"]:
        raise ValueError("full-model reference mode is only for E2E")
    _onboarding_preflight(args, config, selection)
    model = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab_size = int(getattr(model, "text_config", model).vocab_size)
    configured = any(
        getattr(args, name, None) or os.environ.get(f"ARCTIC_CORTEX_{name.upper()}") for name in ("base_url", "host")
    )
    # Reference-only execution needs the Cortex record format, not a reachable target service.
    backend = _client_backend(args) if configured else CortexConfig(base_url="http://127.0.0.1:1")
    arms = [ArmDefinition(arm.name, arm.global_batch_size, arm.max_seq_len) for arm in spec.arms]
    execute(
        config,
        spec,
        arms,
        vocab_size=vocab_size,
        select=selection,
        reference_token_budget=args.reference_token_budget,
        workdir=args.output_dir,
        prepare_references=True,
        reference_records_dir=args.reference_records_dir,
        reference_calibration_runs=args.reference_calibration_runs,
        reference_calibration_workers=args.reference_calibration_workers,
        client_backend=backend,
        local_gateway_full_model=args.local_gateway_full_model,
        peft_adapter=_shared_adapter(args, config),
    )
    return 0


def run(args: argparse.Namespace) -> int:
    from arctic_platform.correctness.harness.config import config_id
    from arctic_platform.correctness.harness.config import local_config_path
    from arctic_platform.correctness.harness.execution import _load_inference_export_allocation
    from arctic_platform.correctness.harness.execution import _load_inference_initial_model_dir

    if args.test_id == "all" and (getattr(args, "prepare_references", False) or args.probe_layers is not None):
        args.test_id = "single-step-grads"
    if args.test_id != "all" and not getattr(args, "prepare_references", False):
        _onboarding_kind(args.test_id)
    if getattr(args, "inference_export_allocation", None) is not None:
        args.inference_export_allocation = Path(args.inference_export_allocation).resolve()
    _load_inference_export_allocation(SimpleNamespace(**{**vars(args), "test": [args.test_id], "all_configs": False}))
    _load_inference_initial_model_dir(SimpleNamespace(**{**vars(args), "test": [args.test_id], "all_configs": False}))
    _optimizer_validation_bootstrap(args)
    if getattr(args, "local_cortex_fixture", False) and (
        getattr(args, "prepare_references", False) or args.probe_layers is not None
    ):
        raise ValueError("local Cortex fixture selection is only for onboarding with final target validation")
    topology = getattr(args, "inference_sampling_topology", None)
    if topology is not None and (
        getattr(args, "prepare_references", False)
        or args.probe_layers is not None
        or args.test_id != "inference-checkpoint-loss"
    ):
        raise ValueError("explicit inference sampling topology is only for final inference onboarding")
    reuse = getattr(args, "reuse_sizing", None)
    reuse_sha256 = getattr(args, "reuse_sizing_sha256", None)
    if (reuse is None) != (reuse_sha256 is None):
        raise ValueError("--reuse-sizing and --reuse-sizing-sha256 must be supplied together")
    if reuse is not None and (
        getattr(args, "prepare_references", False)
        or args.probe_layers is not None
        or args.test_id != "single-step-grads"
    ):
        raise ValueError("sizing reuse is only supported for gradient onboarding, not probes or reference prep")
    _validate_storage(args)
    args.config = local_config_path(args.config)
    if args.reference_records_dir is not None:
        args.reference_records_dir = args.reference_records_dir.resolve()
    from arctic_platform.correctness.harness.config import load_config

    if getattr(args, "peft_adapter", None) is not None:
        if (
            args.test_id not in {"single-step-grads", "single-step-optimizer"}
            or getattr(args, "reference_tests", None)
            and not set(args.reference_tests) <= {"single-step-grads", "single-step-optimizer"}
            or not getattr(args, "prepare_references", False)
            and getattr(args, "backend", "onprem") != "onprem"
        ):
            raise ValueError("--peft-adapter requires on-prem first-two SFT onboarding or reference preparation")
        config = load_config(args.config)
        _onboarding_preflight(args, config)
        args.peft_adapter = _shared_adapter(args, config)
    if getattr(args, "prepare_references", False):
        if args.probe_layers is not None:
            raise ValueError("reference preparation cannot be combined with model sizing")
        return _prepare_references(args)
    if getattr(args, "reference_tests", None):
        raise ValueError("--reference-tests requires --prepare-references")
    if args.probe_layers is not None:
        if args.probe_output is None:
            raise ValueError("sizing worker requires --probe-output")
        args.checkout_root = _checkout_root(args.config)
        sys.path.insert(0, str(args.checkout_root))
        args.probe_output = args.probe_output.resolve()
        return _probe_worker(args)
    if (
        args.test_id == "all"
        or not (
            args.test_spec or Path(__file__).resolve().parents[1] / "specs" / f"{config_id(args.config)}.json"
        ).exists()
    ):
        return _onboard_configuration(args)
    return _parent(args)


def _onboard_configuration(args: argparse.Namespace) -> int:
    """Resolve a config-only onboarding request and retain each check's actual result."""
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.dss_driver import UnsafeCleanupError
    from arctic_platform.correctness.harness.reference_records import CRITERIA
    from arctic_platform.correctness.harness.registry import onboarding_inapplicable_reason
    from arctic_platform.correctness.harness.registry import registered_tests

    config = load_config(args.config)
    tests = registered_tests()
    requested = list(tests) if args.test_id == "all" else [args.test_id]
    selected = []
    results = {}
    for test_id in requested:
        reason = onboarding_inapplicable_reason(tests[test_id], config)
        if reason:
            results[test_id] = {"status": "inapplicable", "reason": reason}
            print(f"skip {test_id}: {reason}", flush=True)
        else:
            selected.append(test_id)
    selected.sort(
        key=lambda test_id: (
            (
                0
                if test_id == "single-step-grads"
                else 1 if test_id == "single-step-optimizer" else 3 if test_id == "checkpoint-resume-loss" else 2
            ),
            test_id,
        )
    )
    spec_path = (
        Path(args.test_spec).resolve()
        if args.test_spec is not None
        else Path(__file__).resolve().parents[1] / "specs" / f"{config.config_id}.json"
    )
    _onboarding_preflight(args, config, selected)
    output = (
        (Path(args.output_dir) if args.output_dir is not None else correctness_workdir("config-onboarding-"))
        if selected
        else None
    )
    queue = list(selected)
    if selected and not config.is_rl and not spec_path.exists() and "single-step-grads" not in queue:
        queue.insert(0, "single-step-grads")
        print("preparing the gradient fixture required by the selected check", flush=True)
    for index, test_id in enumerate(queue, start=1):
        if test_id != "single-step-grads" and not config.is_rl and not spec_path.exists():
            results[test_id] = {"status": "failed", "reason": "gradient fixture onboarding failed"}
            continue
        child = argparse.Namespace(
            **{
                **vars(args),
                "test_id": test_id,
                "test_spec": spec_path,
                "output_dir": output / test_id,
            }
        )
        if hasattr(args, "_diagnostics_dir"):
            child._diagnostics_dir = args._diagnostics_dir / test_id
            child._diagnostics_dir.mkdir(parents=True, exist_ok=True)
        _stage(f"Onboarding {index}/{len(queue)}", test_id.replace("-", " "), f"Config: {config.config_id}")
        try:
            code = _parent(child)
            results[test_id] = {
                "status": "onboarded" if code == 0 else "failed",
                "exit_code": code,
                "output_dir": str(child.output_dir),
            }
        except UnsafeCleanupError:
            raise
        except Exception as exc:
            results[test_id] = {"status": "failed", "reason": str(exc)}
            traceback.print_exc()
            _onboarding_failure(f"{test_id}: {exc}")
    references = [test_id for test_id in selected if test_id in CRITERIA and results[test_id]["status"] == "onboarded"]
    preparation_error = None
    if len(references) > 1:
        preparation = argparse.Namespace(
            **{
                **vars(args),
                "test_id": references[0],
                "test_spec": spec_path,
                "reference_tests": references,
                "output_dir": output / "combined-references",
            }
        )
        try:
            _prepare_references(preparation)
        except UnsafeCleanupError:
            raise
        except Exception as exc:
            preparation_error = str(exc)
            traceback.print_exc()
            _onboarding_failure(f"combined reference preparation: {exc}")
    summary = {
        "config_id": config.config_id,
        "config": str(args.config),
        "test_spec": str(spec_path),
        "results": results,
        "combined_reference_error": preparation_error,
    }
    destination = spec_path.with_name(f"{spec_path.stem}.onboarding.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if hasattr(args, "_diagnostics_dir"):
        (args._diagnostics_dir / "onboarding.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        lines = ["# Onboarding Report", "", f"Config: `{config.config_id}`", ""]
        for test_id, result in results.items():
            lines.extend(
                [
                    f"## {test_id}",
                    "",
                    f"Status: {result['status']}",
                    "",
                    result.get("reason", f"Exit code: {result.get('exit_code', 'not run')}"),
                    "",
                ]
            )
        if preparation_error:
            lines.extend(["## Combined Reference Failure", "", preparation_error, ""])
        lines.extend(
            [
                "## Retry",
                "",
                (
                    "Resolve the reported prerequisite or execution failure, then repeat the same onboarding command. "
                    "Completed phases are reused only while their inputs, environment and retained artifacts match. "
                    "Missing or changed artifacts are rebuilt; failed validation is never counted as completed."
                ),
                "",
                "Complete console output and tracebacks: [onboarding.log](onboarding.log)",
                "",
            ]
        )
        report_path = args._diagnostics_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report: {report_path}", flush=True)
    print(f"onboarding summary: {destination}", flush=True)
    print(f"spec: {spec_path}" if spec_path.exists() else "no specification was published", flush=True)
    return int(preparation_error is not None or any(result["status"] == "failed" for result in results.values()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Size, calibrate, and freeze one config-specific correctness test",
    )
    add_arguments(parser)
    return logged_run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
