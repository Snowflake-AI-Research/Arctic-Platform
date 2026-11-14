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

"""Supported Python interface shared by operator and diagnostic drivers.

Regression consumes reviewed specifications and already-prepared references.
The API never allocates nodes, installs software or changes numerical gates.
"""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentTypeError
from argparse import Namespace
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from decimal import DecimalException
from pathlib import Path
from typing import Callable
from typing import Sequence

from .harness.config import TestSpec
from .harness.config import arms_for
from .harness.config import config_checksum
from .harness.config import config_id
from .harness.config import config_import_subpath
from .harness.config import config_paths
from .harness.config import load_config
from .harness.config import local_config_path
from .harness.config import resolve_config_path
from .harness.registry import TestOutcome
from .harness.registry import TestResult
from .harness.registry import registered_tests
from .harness.registry import workload_inapplicable_reason

PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunRequest:
    config: str | Path | None = None
    tests: tuple[str, ...] = ()
    all_configs: bool = False
    config_dir: Path = PACKAGE_DIR / "configs"


@dataclass(frozen=True)
class RunOptions:
    backend: str = "cortex-training"
    host: str | None = None
    database: str | None = None
    schema: str | None = None
    endpoint: str | None = None
    base_url: str | None = None
    pat_file: Path | None = None
    gpu_pool: int | None = None
    order: str = "batched"
    fail_fast: bool = False
    attn: tuple[str, ...] = ()
    any_gpu: bool = False
    case: tuple[str, ...] = ()
    tokens: int | None = None
    out: Path | None = None
    reference_records_dir: Path | None = None
    peft_adapter: Path | None = None
    cortex_fixture: Path | None = None
    local_cortex_fixture: bool = False
    local_gateway_full_model: bool = False
    test_spec: Path | None = None
    checkpoint_training_gpus: int | None = None
    inference_training_gpus: int | None = None
    inference_sampling_topology: tuple[int, int, int] | None = None
    inference_export_allocation: Path | None = None
    inference_export_allocation_sha256: str | None = None
    inference_initial_model_dir: Path | None = None
    dense_checkpoint_allocation: Path | None = None
    dense_checkpoint_allocation_sha256: str | None = None


@dataclass(frozen=True)
class SelectedCase:
    config_id: str
    config_path: Path
    test_id: str


@dataclass(frozen=True)
class ArtifactStatus:
    state: str
    reason: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordedOnboarding:
    state: str
    outcome: str | None = None
    reason: str = ""
    path: str | None = None
    runtime_identity: str | None = None


@dataclass(frozen=True)
class OnboardingStatus:
    config_id: str
    config_path: str
    test_id: str
    status: str
    reason: str
    spec: ArtifactStatus
    references: ArtifactStatus
    historical: RecordedOnboarding


@dataclass
class BatchResult:
    results: list[TestResult]
    report_dir: Path
    exit_code: int
    elapsed_seconds: float
    stopped_unsafe: bool = False
    cases: list[dict] = field(default_factory=list)


def catalog(config_dir: Path | None = None) -> dict[str, Path]:
    """Active reviewed configurations, ordered by exact ID; archived files are excluded."""
    root = Path(config_dir) if config_dir is not None else PACKAGE_DIR / "configs"
    found = {}
    paths = config_paths(root)
    for path in paths:
        identifier = config_id(path)
        if {"archive", "specs"} & set(path.relative_to(root).parts):
            continue
        if not (PACKAGE_DIR / "specs" / f"{identifier}.json").is_file():
            continue
        if identifier in found:
            raise ValueError(f"duplicate configuration ID: {identifier}")
        found[identifier] = path
    return dict(sorted(found.items()))


def _status_json(path: Path) -> dict:
    from .harness.storage import MAX_RECORD_BYTES
    from .harness.storage import _strict_record_json

    if path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError(f"oversized metadata record: {path}")
    value = _strict_record_json(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"metadata record must be an object: {path}")
    return value


def _status_spec(cfg, path: Path, *, source_path: Path) -> tuple[TestSpec | None, ArtifactStatus]:
    paths = (str(path),)
    try:
        _status_json(path)
        spec = TestSpec.read(path)
        if not isinstance(spec.applicable_tests, list) or not all(isinstance(t, str) for t in spec.applicable_tests):
            raise ValueError("applicable_tests must be a list of test IDs")
        if len(set(spec.applicable_tests)) != len(spec.applicable_tests):
            raise ValueError("duplicate applicable test IDs")
        if not isinstance(spec.attn_implementations, list) or not all(
            isinstance(a, str) for a in spec.attn_implementations
        ):
            raise ValueError("attn_implementations must be a list of names")
        if not isinstance(spec.inapplicable, dict) or not isinstance(spec.test_settings, dict):
            raise ValueError("inapplicable and test_settings must be objects")
        unknown = set(spec.applicable_tests) - registered_tests().keys()
        if unknown:
            raise ValueError(f"spec names unknown check(s): {', '.join(sorted(unknown))}")
        if spec.config_id != cfg.config_id or spec.config_checksum != config_checksum(cfg):
            return spec, ArtifactStatus("stale", "spec config identity/checksum differs; rerun onboarding", paths)
        current = [(a.name, a.global_batch_size, a.max_seq_len) for a in arms_for(cfg.max_seq_len, cfg.n_gpus)]
        reviewed = [(a.name, a.global_batch_size, a.max_seq_len) for a in spec.arms]
        if current != reviewed or cfg.attention_implementation not in spec.attn_implementations:
            return spec, ArtifactStatus("stale", "spec cases/attention differ; rerun onboarding", paths)
        return spec, ArtifactStatus("valid", "spec matches config checksum, cases and attention", paths)
    except FileNotFoundError:
        return None, ArtifactStatus("missing", f"no spec; run onboarding for {source_path}", paths)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, DecimalException) as exc:
        return None, ArtifactStatus("malformed", f"cannot validate spec: {exc}", paths)


def _status_references(cfg, spec, test_id: str, root: Path, *, combined: bool) -> ArtifactStatus:
    from types import SimpleNamespace

    from .harness import storage

    if test_id not in storage.CRITERIA:
        return ArtifactStatus("not_required", "this check does not use a prepared HF record")
    if spec is None:
        return ArtifactStatus("unverified", "a valid spec is required to locate reference workloads")
    ctx = SimpleNamespace(cfg=cfg, spec=spec, reference_records_dir=root)
    arms = [None] if test_id == "e2e-sft-train-validate" else spec.arms
    paths = []
    try:
        identities = []
        for arm in arms:
            identity = {"check_id": test_id, "config_id": cfg.config_id}
            if arm is not None:
                identity["recipe"] = {
                    "arm": arm.name,
                    "optimizer": {} if combined else None,
                    "synthetic_inputs": {"rows": arm.global_batch_size, "max_seq_len": arm.max_seq_len},
                }
            identities.append(identity)
        record_paths = [storage._record_path(ctx, identity) for identity in identities]
        # A partially published current set must not borrow arms from an older set.
        if not any(path.parent.exists() for path in record_paths):
            record_paths = [
                storage._record_path(
                    ctx,
                    {
                        **identity,
                        "recipe": {
                            key: value
                            for key, value in identity.get("recipe", {}).items()
                            if key != "synthetic_inputs"
                        },
                    },
                )
                for identity in identities
            ]
        for arm, path in zip(arms, record_paths):
            paths.append(str(path))
            record = _status_json(path)
            saved = record["identity"]
            if record.get("schema") != storage.SCHEMA_VERSION or not isinstance(saved, dict):
                raise ValueError("unsupported reference schema/identity")
            if record.get("payload_sha256") != storage._digest(record.get("measurements")):
                raise ValueError("reference payload checksum mismatch")
            if any(
                saved.get(key) != value
                for key, value in {
                    "schema": storage.SCHEMA_VERSION,
                    "config_id": cfg.config_id,
                    "config_checksum": spec.config_checksum,
                    "check_id": test_id,
                    "criterion": storage.CRITERIA[test_id],
                    "parameter_name_map": spec.param_name_map,
                }.items()
            ):
                return ArtifactStatus("stale", f"reference config/check identity differs: {path}", tuple(paths))
            if test_id == "single-step-grads" and saved.get("gate") != asdict(spec.test_tolerances[test_id]):
                return ArtifactStatus("stale", f"reference frozen gate differs: {path}", tuple(paths))
            if arm is not None:
                inputs = saved.get("recipe", {}).get("synthetic_inputs", {})
                if inputs.get("rows") != arm.global_batch_size or inputs.get("max_seq_len") != arm.max_seq_len:
                    return ArtifactStatus("stale", f"reference case shape differs: {path}", tuple(paths))
            storage.validate_measurements(saved, record["measurements"])
        return ArtifactStatus(
            "metadata_valid",
            "reference identity fields and scalar payload valid; full workload/runtime identity not checked",
            tuple(paths),
        )
    except FileNotFoundError as exc:
        return ArtifactStatus("missing", f"prepared reference missing: {exc.filename}; rerun onboarding", tuple(paths))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        return ArtifactStatus("malformed", f"cannot validate reference metadata: {exc}", tuple(paths))


def _status_history(path: Path, spec: TestSpec | None, config_identifier: str, test_id: str) -> RecordedOnboarding:
    try:
        summary = _status_json(path)
        if summary.get("config_id") != config_identifier:
            return RecordedOnboarding("stale", reason="summary config identity differs", path=str(path))
        if not isinstance(summary.get("results"), dict):
            raise ValueError("summary results must be an object")
        saved = summary["results"].get(test_id)
        if saved is None:
            return RecordedOnboarding("missing", reason="no recorded outcome for this check", path=str(path))
        if not isinstance(saved, dict) or saved.get("status") not in {
            "onboarded",
            "failed",
            "inapplicable",
            "not_run",
        }:
            raise ValueError("invalid recorded onboarding outcome")
        if saved["status"] == "onboarded" and (type(saved.get("exit_code")) is not int or saved["exit_code"] != 0):
            raise ValueError("recorded success requires exit_code=0")
        # The producer hashes the normalized TestSpec projection, not specification file bytes.
        identity = None
        if spec is not None and test_id in spec.applicable_tests and spec.config_checksum:
            raw = asdict(spec)
            fields = ("config_id", "config_checksum", "model", "arms", "attn_implementations", "param_name_map")
            projection = {name: raw[name] for name in fields}
            projection["test_tolerance"] = raw["test_tolerances"].get(test_id)
            projection["test_settings"] = raw["test_settings"].get(test_id)
            identity = hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest()
        state = (
            "unbound"
            if saved.get("spec_identity") is None
            else "current_spec" if saved["spec_identity"] == identity else "stale"
        )
        reason = str(saved.get("reason") or "recorded outcome only; current runtime/options are not checked")
        if state == "stale":
            reason = f"recorded spec identity differs; {reason}"
        return RecordedOnboarding(state, saved["status"], reason, str(path), saved.get("runtime_identity"))
    except FileNotFoundError:
        return RecordedOnboarding(
            "missing", reason="no onboarding summary; this does not invalidate an imported spec", path=str(path)
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return RecordedOnboarding("malformed", reason=f"cannot validate onboarding summary: {exc}", path=str(path))


def onboarding_status(
    request: RunRequest | None = None, *, reference_records_dir: Path | None = None
) -> tuple[OnboardingStatus, ...]:
    """Read config/test metadata only; no imports, publication, backend or readiness claim.

    Missing summaries do not make reviewed imported specifications unusable. Recorded
    successes are historical, even when their per-check specification identity matches.
    """
    request = request or RunRequest(config_dir=PACKAGE_DIR / "configs")
    if request.config is not None and request.all_configs:
        raise ValueError("choose --config or --all-configs, not both")
    tests = registered_tests()
    unknown = set(request.tests) - tests.keys()
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(sorted(unknown))}; use the list command")
    configs = {}
    for path in config_paths(request.config_dir):
        identifier = config_id(path)
        if identifier in configs:
            raise ValueError(f"duplicate configuration ID: {identifier}")
        configs[identifier] = path
    paths = list(configs.values())
    if request.config is not None:
        selector = str(request.config)
        path = configs.get(selector, resolve_config_path(selector))
        if not path.is_file() or path.suffix not in {".json", ".config"} or path.name == "model.json":
            raise ValueError(f"unknown configuration: {selector}; give an exact ID or existing config path")
        paths = [path]
    results = []
    test_ids = sorted(set(request.tests) or tests.keys())
    for path in sorted(paths, key=config_id):
        identifier = config_id(path)
        try:
            cfg = load_config(path)
            # Derive import identity without importing or probing local hardware.
            if not path.resolve().is_relative_to(request.config_dir.resolve()) and cfg.gpu_type is not None:
                subpath = config_import_subpath(cfg)
                cfg.path = request.config_dir / subpath
                identifier = config_id(cfg.path)
                cfg.config_id = identifier
            spec_path = PACKAGE_DIR / "specs" / f"{identifier}.json"
            spec, spec_status = _status_spec(cfg, spec_path, source_path=path)
            historical_spec = spec if spec_status.state == "valid" else None
            for test_id in test_ids:
                history = _status_history(
                    spec_path.with_suffix(".onboarding.json"), historical_spec, identifier, test_id
                )
                reason = workload_inapplicable_reason(tests[test_id], cfg)
                pair_spec = spec_status
                if reason is not None:
                    pair_spec = ArtifactStatus("inapplicable", reason, spec_status.paths)
                elif spec_status.state == "valid" and test_id not in spec.applicable_tests:
                    pair_spec = ArtifactStatus(
                        "not_reviewed", spec.inapplicable.get(test_id, "check not covered by spec"), spec_status.paths
                    )
                elif spec_status.state == "valid" and test_id in {
                    "single-step-grads",
                    "checkpoint-resume-loss",
                }:
                    try:
                        spec.tolerance_for(test_id)
                    except ValueError as exc:
                        pair_spec = ArtifactStatus("not_reviewed", str(exc), spec_status.paths)
                if test_id == "single-step-optimizer" and pair_spec.state == "valid":
                    settings = spec.test_settings.get(test_id)
                    if settings is None or isinstance(settings, dict) and "reference_token_budget" not in settings:
                        pair_spec = ArtifactStatus(
                            "not_reviewed",
                            "optimizer reference_token_budget not reviewed; rerun onboarding",
                            spec_status.paths,
                        )
                    elif not isinstance(settings, dict):
                        pair_spec = ArtifactStatus(
                            "malformed", "optimizer test_settings must be an object", spec_status.paths
                        )
                    elif (
                        type(settings["reference_token_budget"]) is not int or settings["reference_token_budget"] <= 0
                    ):
                        pair_spec = ArtifactStatus(
                            "malformed",
                            "optimizer reference_token_budget must be a positive integer",
                            spec_status.paths,
                        )
                references = _status_references(
                    cfg,
                    spec if pair_spec.state == "valid" else None,
                    test_id,
                    (
                        Path(reference_records_dir)
                        if reference_records_dir is not None
                        else PACKAGE_DIR / "reference_data"
                    ),
                    combined="single-step-optimizer" in test_ids
                    and spec is not None
                    and "single-step-optimizer" in spec.applicable_tests,
                )
                if pair_spec.state == "inapplicable":
                    references = ArtifactStatus("not_required", "check is inapplicable to this workload")
                status, explanation = pair_spec.state, pair_spec.reason
                if status == "valid":
                    status, explanation = references.state, references.reason
                    if status in {"metadata_valid", "not_required"}:
                        status = "metadata_valid"
                        explanation = (
                            "reviewed metadata valid; not proof of current runtime readiness or validation success"
                        )
                results.append(
                    OnboardingStatus(
                        identifier, str(path), test_id, status, explanation, pair_spec, references, history
                    )
                )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            failure = ArtifactStatus("malformed", f"cannot inspect configuration/artifacts: {exc}", (str(path),))
            results.extend(
                OnboardingStatus(
                    identifier,
                    str(path),
                    test_id,
                    "malformed",
                    failure.reason,
                    failure,
                    ArtifactStatus("unverified", "configuration invalid"),
                    RecordedOnboarding("unverified"),
                )
                for test_id in test_ids
            )
    return tuple(results)


def select(request: RunRequest) -> tuple[SelectedCase, ...]:
    """Validate selectors before any backend or model work and expand each pair once."""
    if request.config is not None and request.all_configs:
        raise ValueError("choose --config or --all-configs, not both")
    if request.config is None and not request.tests and not request.all_configs:
        raise ValueError("give --config <ID or path> or --test <ID>")
    tests = registered_tests()
    unknown = sorted(set(request.tests) - tests.keys())
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(unknown)}; use the list command")
    configs = catalog(request.config_dir)
    if request.config is not None:
        selector = str(request.config)
        if selector in configs:
            paths = [configs[selector]]
        elif resolve_config_path(selector).is_file() and Path(selector).suffix in {".config", ".json"}:
            paths = [local_config_path(selector, request.config_dir)]
        else:
            raise ValueError(f"unknown configuration: {selector}; use the list command or an existing config path")
    else:
        paths = list(configs.values())
    if not paths:
        raise ValueError(f"no active reviewed configurations in {request.config_dir}")
    selected = []
    for path in sorted(paths, key=config_id):
        if request.tests:
            test_ids = sorted(set(request.tests))
        else:
            config = load_config(path)
            test_ids = sorted(
                test_id for test_id, test in tests.items() if workload_inapplicable_reason(test, config) is None
            )
        selected.extend(SelectedCase(config_id(path), path, test_id) for test_id in test_ids)
    return tuple(selected)


def _namespace(request: RunRequest, options: RunOptions) -> Namespace:
    from .harness.execution import operator_backend

    values = asdict(options)
    try:
        values["backend"] = operator_backend(options.backend)
    except ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
    values.update(
        config=str(request.config) if request.config is not None else None,
        config_dir=str(request.config_dir),
        all_configs=request.all_configs,
        test=list(request.tests) or None,
        attn=list(options.attn) or None,
        case=list(options.case) or None,
    )
    return Namespace(**values)


def parse_run_args(argv: Sequence[str]) -> tuple[RunRequest, RunOptions]:
    """Parse the same run flags as the operator command without starting a workload."""
    from .harness.execution import parse_args

    args = parse_args(["run", *argv])
    return request_from_args(args), options_from_args(args)


def request_from_args(args: Namespace) -> RunRequest:
    return RunRequest(
        config=args.config,
        tests=tuple(args.test or ()),
        all_configs=args.all_configs,
        config_dir=Path(args.config_dir),
    )


def options_from_args(args: Namespace) -> RunOptions:
    values = {
        name: getattr(args, name)
        for name in RunOptions.__dataclass_fields__
        if hasattr(args, name) and getattr(args, name) is not None
    }
    for name in ("attn", "case", "inference_sampling_topology"):
        if name in values:
            values[name] = tuple(values[name])
    if "out" in values:
        values["out"] = Path(values["out"])
    if values.get("backend") == "cortex":
        values["backend"] = "cortex-training"
    return RunOptions(**values)


def run(request: RunRequest, options: RunOptions | None = None, *, executor: Callable | None = None) -> BatchResult:
    """Run selected cases sequentially with short output and complete saved logs."""
    from .harness.execution import _load_dense_checkpoint_allocation
    from .harness.execution import _load_inference_export_allocation
    from .harness.execution import _load_inference_initial_model_dir
    from .harness.execution import _validate_regression_tokens
    from .harness.execution import run_batch

    options = options or RunOptions()
    if options.test_spec is not None and (request.config is None or request.all_configs):
        raise ValueError("--test-spec requires one explicit --config")
    if options.checkpoint_training_gpus is not None:
        if request.config is None or request.all_configs or request.tests != ("checkpoint-resume-loss",):
            raise ValueError("--checkpoint-training-gpus requires one config and only checkpoint-resume-loss")
        if type(options.checkpoint_training_gpus) is not int or options.checkpoint_training_gpus <= 0:
            raise ValueError("--checkpoint-training-gpus must be a positive integer")
    if options.inference_sampling_topology is not None and (
        request.config is None or request.all_configs or request.tests != ("inference-checkpoint-loss",)
    ):
        raise ValueError("--inference-sampling-topology requires one config and only inference-checkpoint-loss")
    if options.inference_training_gpus is not None:
        if request.config is None or request.all_configs or request.tests != ("inference-checkpoint-loss",):
            raise ValueError("--inference-training-gpus requires one config and only inference-checkpoint-loss")
        if type(options.inference_training_gpus) is not int or options.inference_training_gpus <= 0:
            raise ValueError("--inference-training-gpus must be a positive integer")
    args = _namespace(request, options)
    _validate_regression_tokens(args)
    _load_dense_checkpoint_allocation(args)
    _load_inference_export_allocation(args)
    _load_inference_initial_model_dir(args)
    selected = select(request)
    return run_batch(args, selected=selected, executor=executor)


def aggregate_exit_code(results: Sequence[TestResult]) -> int:
    """Failures and blocked comparisons are nonzero; an empty/inapplicable batch is not a pass."""
    return (
        0
        if any(r.outcome is TestOutcome.PASS for r in results)
        and not any(r.outcome is TestOutcome.FAIL for r in results)
        else 1
    )


def main(argv: Sequence[str] | None = None) -> int:
    from .harness.execution import main as command_main

    return command_main(argv)
