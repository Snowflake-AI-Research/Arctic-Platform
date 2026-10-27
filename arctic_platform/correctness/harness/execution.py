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

"""Shared operator selection, execution and command argument handling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import SecretStr

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig

from . import report as console
from .config import ArmDefinition
from .config import TestSpec
from .config import arms_for
from .config import config_checksum
from .config import config_id
from .config import load_config
from .config import local_config_path
from .config import local_gpu_type
from .config import placement_widths
from .config import scaled_for_placement
from .config import validate_against_host
from .dss_driver import available_gpus
from .dss_driver import local_cortex_gateway
from .hosted import is_local_cortex_gateway
from .hosted import read_pat
from .hosted import require_local_cortex_gateway
from .registry import RL_INAPPLICABLE_REASON
from .registry import TestOutcome
from .registry import TestResult
from .registry import reference_tests
from .registry import registered_tests
from .report import write as write_report
from .runner import assert_applicable_tests_registered
from .runner import cortex_unavailable_reason
from .runner import execute
from .runner import execute_rl
from .runner import validate_local_full_model
from .storage import correctness_logdir
from .storage import correctness_workdir
from .storage import ensure_no_interrupted_builds
from .storage import require_data_fast_path

CORRECTNESS_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = CORRECTNESS_DIR / "configs"
SPEC_DIR = CORRECTNESS_DIR / "specs"


def _resolve_configs(args, *, filter_by_local_gpu: bool = True) -> List[Path]:
    """The configs this invocation runs. Local-GPU filtering applies only where the jobs use local GPUs."""
    if getattr(args, "resolved_paths", None) is not None:
        return list(args.resolved_paths)
    if args.all_configs:
        paths = sorted(Path(args.config_dir).rglob("*.config"))
        if not filter_by_local_gpu or args.any_gpu:
            return paths
        gpu_type = local_gpu_type()
        if gpu_type is None:
            return paths
        selected = [path for path in paths if load_config(path).gpu_type in {None, gpu_type}]
        omitted = len(paths) - len(selected)
        if omitted:
            print(f"selected {len(selected)} {gpu_type} config(s); omitted {omitted} for other GPU types", flush=True)
        return selected
    if not args.config:
        raise SystemExit("give --config <path> or --all-configs")
    from ..api import catalog

    path = catalog(Path(args.config_dir)).get(str(args.config), Path(args.config))
    return [local_config_path(path, Path(args.config_dir))]


def _describe(shapes) -> str:
    """``gas1 (1 x 2,048), gas4 (32 x 2,048)`` -- readable enough to see what changed."""
    return ", ".join(f"{name} ({rows:,} x {length:,})" for name, rows, length in shapes)


def _frozen_spec(
    cfg,
    path: Path,
    arm_defs: List[ArmDefinition],
    *,
    attention: str,
    cases: Optional[List[str]],
    results: List[TestResult],
    specs: Dict[str, TestSpec],
    test_spec: Path | None = None,
) -> Optional[Tuple[TestSpec, List[ArmDefinition]]]:
    """This config's reviewed spec and the cases to run, or ``None`` once the reason it cannot run is recorded.

    The spec is what makes a result a regression rather than a measurement: it names the configuration that
    was reviewed, the cases derived from it, and the attention backend onboarding covered. A config whose
    content, cases, or backend no longer match it is refused here, because a verdict about a configuration
    nobody reviewed says nothing.
    """
    spec_path = Path(test_spec) if test_spec is not None else SPEC_DIR / f"{cfg.config_id}.json"
    if not spec_path.exists():
        results.append(
            TestResult(
                test_id="-",
                config_id=cfg.config_id,
                outcome=TestOutcome.INAPPLICABLE,
                summary="no spec",
                reason=f"run onboarding for {path} first",
            )
        )
        print(f"skip {path.name}: no spec at {spec_path}", flush=True)
        return None
    spec = TestSpec.read(spec_path)
    if spec.config_id != cfg.config_id:
        results.append(
            TestResult(
                test_id="-",
                config_id=cfg.config_id,
                outcome=TestOutcome.FAIL,
                summary="spec config identity mismatch",
                reason=f"spec {spec.config_id!r} does not describe config {cfg.config_id!r}",
            )
        )
        return None
    current_config_checksum = config_checksum(cfg)
    if not spec.config_checksum or spec.config_checksum != current_config_checksum:
        expected = spec.config_checksum[:12] if spec.config_checksum else "missing"
        actual = current_config_checksum[:12]
        results.append(
            TestResult(
                test_id="-",
                config_id=cfg.config_id,
                outcome=TestOutcome.FAIL,
                summary="config checksum mismatch",
                reason=(
                    f"the reviewed spec records training config checksum {expected}, but the current "
                    f"config hashes to {actual}; run onboarding again for {path}"
                ),
            )
        )
        print(
            f"skip {path.name}: training config checksum is {actual}, reviewed spec has {expected}; "
            "run onboarding again",
            flush=True,
        )
        return None

    try:
        assert_applicable_tests_registered(spec, cfg.config_id)
    except ValueError as exc:
        results.append(
            TestResult(
                test_id="-",
                config_id=cfg.config_id,
                outcome=TestOutcome.FAIL,
                summary="spec names an unregistered check",
                reason=str(exc),
            )
        )
        print(f"skip {path.name}: {exc}", flush=True)
        return None

    def case_checksum(collection):
        return tuple((a.name, a.global_batch_size, a.max_seq_len) for a in collection)

    selected_arms = arm_defs
    reviewed_arms = spec.arms
    if cases:
        wanted = set(cases)
        current_names = [arm.name for arm in arm_defs]
        reviewed_names = [arm.name for arm in spec.arms]
        selection_error = None
        if len(wanted) != len(cases):
            selection_error = "selected case names must be unique"
        elif len(set(current_names)) != len(current_names) or len(set(reviewed_names)) != len(reviewed_names):
            selection_error = "current and reviewed case catalogs must contain unique names"
        elif unknown := wanted - set(current_names):
            selection_error = (
                f"unknown case(s) {', '.join(sorted(unknown))}; this config has {', '.join(current_names)}"
            )
        elif missing := wanted - set(reviewed_names):
            selection_error = f"reviewed spec does not cover selected case(s) {', '.join(sorted(missing))}"
        if selection_error is not None:
            results.append(
                TestResult(
                    test_id="-",
                    config_id=cfg.config_id,
                    outcome=TestOutcome.FAIL,
                    summary="case selection mismatch",
                    reason=selection_error,
                )
            )
            print(f"skip {path.name}: {selection_error}", flush=True)
            return None
        selected_arms = [arm for arm in arm_defs if arm.name in wanted]
        reviewed_by_name = {arm.name: arm for arm in spec.arms}
        reviewed_arms = [reviewed_by_name[arm.name] for arm in selected_arms]

    spec_checksum = case_checksum(reviewed_arms)
    current_checksum = case_checksum(selected_arms)
    if spec_checksum != current_checksum:
        results.append(
            TestResult(
                test_id="-",
                config_id=cfg.config_id,
                outcome=TestOutcome.FAIL,
                summary="case checksum mismatch",
                reason=(
                    f"the spec's case checksum is {_describe(spec_checksum)} but this config now "
                    f"derives {_describe(current_checksum)}; run onboarding again for {path}"
                ),
            )
        )
        print(
            f"skip {path.name}: case checksum is {_describe(spec_checksum)}, expected "
            f"{_describe(current_checksum)}; run onboarding again",
            flush=True,
        )
        return None
    specs[cfg.config_id] = spec
    if attention not in spec.attn_implementations:
        results.append(
            TestResult(
                test_id="-",
                config_id=cfg.config_id,
                outcome=TestOutcome.FAIL,
                summary="attention implementation mismatch",
                reason=(
                    f"the spec covers {spec.attn_implementations}, but this {cfg.gpu_type or 'untyped'} "
                    f"config selects {attention!r}; run onboarding for this GPU type"
                ),
            )
        )
        print(f"skip {path.name}: spec does not cover {attention}; run onboarding", flush=True)
        return None

    return spec, selected_arms


def _finish(
    results: List[TestResult],
    specs: Dict[str, TestSpec],
    out: Path,
    *,
    configs: int,
    started: float,
    execution: Dict[str, dict] | None = None,
) -> int:
    """Write the report and print what the run concluded. Zero only when something passed and none failed."""
    write_report(results, specs, out, execution=execution)
    failed = [r for r in results if r.outcome is TestOutcome.FAIL]
    passed = [r for r in results if r.outcome is TestOutcome.PASS]
    skipped = [r for r in results if r.outcome is TestOutcome.INAPPLICABLE]
    print(f"Overall: {len(passed)} passed, {len(failed)} failed, {len(skipped)} inapplicable")
    if not passed and not failed:
        print("  nothing ran: every config was skipped, so this result says nothing about correctness")
    print(f"report: {out / 'report.md'}")
    for r in failed:
        print(f"  FAIL {r.config_id} {r.test_id} {r.arm or ''}: {r.summary}")
    print(
        f"total run time across {configs} config(s): {console.total_duration(time.monotonic() - started)}", flush=True
    )
    return 0 if passed and not failed else 1


def _catalog_vocab_size(model_name: str) -> int:
    """The vocabulary of the model a hosted job will serve, read from that model's own config."""
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return int(getattr(model_cfg, "text_config", model_cfg).vocab_size)


def _client_backend(args) -> OnPremConfig | CortexConfig:
    if getattr(args, "backend", "cortex") == "onprem":
        return OnPremConfig()
    options = {
        key: getattr(args, key)
        for key in ("host", "database", "schema", "endpoint", "base_url")
        if getattr(args, key, None) is not None
    }
    if getattr(args, "pat_file", None) is not None:
        options["pat"] = SecretStr(read_pat(args.pat_file))
    backend = CortexConfig(**options)
    if is_local_cortex_gateway(backend):
        # The local SnowAPI router requires both segments even though it ignores their values.
        backend.database = backend.database or "dev"
        backend.schema_ = backend.schema_ or "dev"
    return backend


def _load_inference_export_allocation(args) -> dict | None:
    """Read pinned allocation metadata only for one explicit local inference check."""
    path = getattr(args, "inference_export_allocation", None)
    expected = getattr(args, "inference_export_allocation_sha256", None)
    if (path is None) != (expected is None):
        raise ValueError(
            "--inference-export-allocation and --inference-export-allocation-sha256 are required together"
        )
    if path is None:
        return None
    if (
        not getattr(args, "config", None)
        or getattr(args, "all_configs", False)
        or getattr(args, "test", None) != ["inference-checkpoint-loss"]
        or getattr(args, "backend", "cortex") != "cortex"
        or getattr(args, "prepare_references", False)
        or getattr(args, "probe_layers", None) is not None
        or getattr(args, "local_cortex_fixture", False)
        or getattr(args, "cortex_fixture", None) is not None
    ):
        raise ValueError("export allocation requires one explicit local Cortex inference-checkpoint-loss config")
    require_local_cortex_gateway(_client_backend(args))
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("export allocation requires a full lowercase SHA256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("inference export allocation SHA256 mismatch")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate inference export allocation key: {key}")
            result[key] = value
        return result

    admission = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(admission, dict):
        raise ValueError("inference export allocation must be a JSON object")
    json.dumps(admission, allow_nan=False)
    return admission


def _load_dense_checkpoint_allocation(args, *, allow_automatic_gateway: bool = False) -> dict | None:
    """Read a pinned same-job allocation before optimizer selection can be grouped."""
    path = getattr(args, "dense_checkpoint_allocation", None)
    expected = getattr(args, "dense_checkpoint_allocation_sha256", None)
    if (path is None) != (expected is None):
        raise ValueError(
            "--dense-checkpoint-allocation and --dense-checkpoint-allocation-sha256 are required together"
        )
    if path is None:
        return None
    selected = getattr(args, "test", None)
    if (
        not getattr(args, "config", None)
        or getattr(args, "all_configs", False)
        or not selected
        or "single-step-optimizer" not in selected
        or not set(selected) <= {"single-step-grads", "single-step-optimizer"}
        or getattr(args, "backend", "cortex") != "cortex"
        or getattr(args, "prepare_references", False)
        or getattr(args, "probe_layers", None) is not None
        or not getattr(args, "local_cortex_fixture", False)
        or getattr(args, "cortex_fixture", None) is not None
        or getattr(args, "local_gateway_full_model", False)
        or getattr(args, "peft_adapter", None) is not None
    ):
        raise ValueError("dense checkpoint allocation requires one explicit local Cortex first-two optimizer config")
    backend_args = args
    if allow_automatic_gateway and not any(
        getattr(args, name, None) or os.environ.get(f"ARCTIC_CORTEX_{name.upper()}") for name in ("base_url", "host")
    ):
        # Metadata admission precedes onboarding's owned gateway; never supply this URL to the target.
        backend_args = copy.copy(args)
        backend_args.base_url = "http://127.0.0.1:1"
    require_local_cortex_gateway(_client_backend(backend_args))
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("dense checkpoint allocation requires a full lowercase SHA256")
    with Path(path).open("rb") as handle:
        raw = handle.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("dense checkpoint allocation exceeds metadata bound or SHA256 mismatch")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate dense checkpoint allocation key: {key}")
            result[key] = value
        return result

    admission = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(admission, dict):
        raise ValueError("dense checkpoint allocation must be a JSON object")
    json.dumps(admission, allow_nan=False)
    return admission


def _load_inference_initial_model_dir(args) -> Path | None:
    """Forward an explicit initial training root without changing the configured model identity."""
    value = getattr(args, "inference_initial_model_dir", None)
    if value is None:
        return None
    if _load_inference_export_allocation(args) is None:
        raise ValueError("inference initial model directory requires a pinned local inference export allocation")
    from .inference_export import validate_initial_training_model_dir

    return validate_initial_training_model_dir(value)


def _load_historical_gradient_references(args) -> dict | None:
    """Load explicit raw-pinned bindings before batch grouping can narrow a selection."""
    path = getattr(args, "historical_gradient_references", None)
    expected = getattr(args, "historical_gradient_references_sha256", None)
    if (path is None) != (expected is None):
        raise ValueError(
            "--historical-gradient-references and --historical-gradient-references-sha256 are required together"
        )
    if path is None:
        return None
    if (
        not getattr(args, "config", None)
        or getattr(args, "all_configs", False)
        or getattr(args, "test", None) != ["single-step-grads"]
        or getattr(args, "prepare_references", False)
        or getattr(args, "probe_layers", None) is not None
    ):
        raise ValueError("historical gradient references require one explicit config and only single-step-grads")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("historical gradient references require a full lowercase SHA256")
    descriptor = Path(path).resolve()
    with descriptor.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("historical gradient references descriptor exceeds 1 MiB")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("historical gradient references descriptor SHA256 mismatch")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate historical gradient references key: {key}")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique_object)
    json.dumps(value, allow_nan=False)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "records"}
        or value["schema"] != "reviewed-gradient-references-v1"
        or not isinstance(value["records"], dict)
        or not value["records"]
    ):
        raise ValueError("invalid historical gradient references descriptor schema")
    bindings = {}
    for arm, binding in value["records"].items():
        if (
            not arm
            or not isinstance(binding, dict)
            or set(binding) != {"record_path", "record_sha256", "review"}
            or not isinstance(binding["record_path"], str)
            or not binding["record_path"]
            or not isinstance(binding["record_sha256"], str)
            or len(binding["record_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in binding["record_sha256"])
            or not isinstance(binding["review"], str)
            or not binding["review"].strip()
        ):
            raise ValueError(f"invalid historical gradient reference binding for arm {arm}")
        record_path = Path(binding["record_path"])
        if not record_path.is_absolute():
            record_path = descriptor.parent / record_path
        bindings[arm] = {**binding, "record_path": str(record_path.resolve())}
    return bindings


def _run_vocab_size(
    cfg, spec, select, backend, published_fixture: dict | None = None, *, local_full_model_path: Path | None = None
) -> int:
    from transformers import AutoConfig

    if local_full_model_path is not None:
        model_cfg = AutoConfig.from_pretrained(local_full_model_path, local_files_only=True, trust_remote_code=True)
        return int(getattr(model_cfg, "text_config", model_cfg).vocab_size)
    if (
        select
        and set(select) == {"inference-checkpoint-loss"}
        and "inference-checkpoint-loss" in spec.applicable_tests
    ):
        with console.activity("Reading configured model vocabulary"):
            return _catalog_vocab_size(cfg.model_name)

    if isinstance(backend, CortexConfig):
        runnable = any(
            test.test_id in spec.applicable_tests
            and (not select or test.test_id in select)
            and cortex_unavailable_reason(test, published_fixture) is None
            for test in registered_tests().values()
        )
        if not runnable:
            return 0
        if published_fixture is None:
            with console.activity("Reading catalog model vocabulary"):
                return _catalog_vocab_size(cfg.model_name)
    with console.activity("Loading model config"):
        model_cfg = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
        return int(getattr(model_cfg, "text_config", model_cfg).vocab_size)


def _configure_regression_target(args):
    """Supported config regression owns a local service only when no endpoint is configured."""
    if (
        getattr(args, "backend", "cortex") == "onprem"
        or not args.config
        or args.all_configs
        or not args.test
        or not (
            set(args.test) <= {"single-step-grads", "single-step-optimizer"}
            or set(args.test) == {"checkpoint-resume-loss"}
        )
        or getattr(args, "cortex_fixture", None) is not None
        or getattr(args, "local_gateway_full_model", False)
        or any(
            getattr(args, name, None) or os.environ.get(f"ARCTIC_CORTEX_{name.upper()}")
            for name in ("base_url", "host")
        )
    ):
        return args
    if load_config(_resolve_configs(args, filter_by_local_gpu=False)[0]).is_rl and args.test != [
        "checkpoint-resume-loss"
    ]:
        return args
    args = copy.copy(args)
    args._automatic_cortex_gateway = True
    args._automatic_checkpoint_bridge = "checkpoint-resume-loss" in args.test
    args.local_cortex_fixture = True
    args.base_url = "http://127.0.0.1:1"
    require_local_cortex_gateway(_client_backend(args))
    return args


@contextmanager
def _regression_gateway(args):
    if not getattr(args, "_automatic_cortex_gateway", False):
        yield None
        return
    if not getattr(args, "_automatic_checkpoint_bridge", False):
        with local_cortex_gateway(
            Path(args.out), gateway_module="arctic_platform.debug.correctness_participation"
        ) as url:
            yield url
        return
    from .cortex import LOCAL_CHECKPOINT_BRIDGE_ENV
    from .cortex import LOCAL_NATIVE_FORWARD_ENV

    names = (LOCAL_NATIVE_FORWARD_ENV, LOCAL_CHECKPOINT_BRIDGE_ENV)
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        with local_cortex_gateway(Path(args.out), gateway_module="arctic_platform.correctness.harness.gateway") as url:
            yield url
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_configs(args, *, executor=None, snapshot=None) -> int:
    started = time.monotonic()
    args = _configure_regression_target(args)
    dense_checkpoint_allocation = _load_dense_checkpoint_allocation(args)
    inference_export_allocation = _load_inference_export_allocation(args)
    inference_initial_model_dir = _load_inference_initial_model_dir(args)
    historical_gradient_references = _load_historical_gradient_references(args)
    test_spec = getattr(args, "test_spec", None)
    topology = getattr(args, "inference_sampling_topology", None)
    training_gpus = getattr(args, "inference_training_gpus", None)
    checkpoint_gpus = getattr(args, "checkpoint_training_gpus", None)
    if test_spec is not None and (not args.config or args.all_configs):
        raise SystemExit("--test-spec requires one explicit --config")
    if checkpoint_gpus is not None:
        if not args.config or args.all_configs or args.test != ["checkpoint-resume-loss"]:
            raise SystemExit("--checkpoint-training-gpus requires one config and only checkpoint-resume-loss")
        if type(checkpoint_gpus) is not int or checkpoint_gpus <= 0:
            raise SystemExit("--checkpoint-training-gpus must be a positive integer")
    if topology is not None and (not args.config or args.all_configs or args.test != ["inference-checkpoint-loss"]):
        raise SystemExit("--inference-sampling-topology requires one config and only inference-checkpoint-loss")
    if training_gpus is not None:
        if not args.config or args.all_configs or args.test != ["inference-checkpoint-loss"]:
            raise SystemExit("--inference-training-gpus requires one config and only inference-checkpoint-loss")
        if type(training_gpus) is not int or training_gpus <= 0:
            raise SystemExit("--inference-training-gpus must be a positive integer")
    if getattr(args, "out", None) is None:
        args.out = correctness_logdir("ap-correctness-run-")
    backend = _client_backend(args)
    if getattr(args, "reference_records_dir", None) is not None:
        from .reference_records import require_git_records_dir

        args.reference_records_dir = require_git_records_dir(args.reference_records_dir)
    remote = isinstance(backend, CortexConfig)
    fixture_path = getattr(args, "cortex_fixture", None)
    local_fixture = getattr(args, "local_cortex_fixture", False)
    local_full_model = getattr(args, "local_gateway_full_model", False)
    if fixture_path is not None and (not remote or not args.config or args.all_configs):
        raise SystemExit("--cortex-fixture requires one --config and --backend cortex")
    if local_fixture and (
        not remote
        or not args.config
        or args.all_configs
        or fixture_path is not None
        or not args.test
        or not (
            set(args.test) <= {"single-step-grads", "single-step-optimizer", "e2e-sft-train-validate"}
            or set(args.test) == {"checkpoint-resume-loss"}
        )
    ):
        raise SystemExit(
            "--local-cortex-fixture requires one Cortex config and only first-two, E2E or checkpoint checks"
        )
    if local_full_model and (
        not remote
        or not args.config
        or args.all_configs
        or local_fixture
        or fixture_path is not None
        or not args.test
        or not set(args.test) <= {"e2e-sft-train-validate", "inference-checkpoint-loss"}
    ):
        raise SystemExit("--local-gateway-full-model requires one Cortex config and only e2e/inference")
    if local_fixture or local_full_model:
        require_local_cortex_gateway(backend)
    config_paths = _resolve_configs(args, filter_by_local_gpu=not remote)
    if test_spec is not None and len(config_paths) != 1:
        raise SystemExit("--test-spec requires exactly one resolved config")
    if inference_export_allocation is not None and len(config_paths) != 1:
        raise ValueError("export allocation requires exactly one resolved config")
    if dense_checkpoint_allocation is not None and len(config_paths) != 1:
        raise ValueError("dense checkpoint allocation requires exactly one resolved config")
    results: List[TestResult] = []
    specs: Dict[str, TestSpec] = {}
    execution: Dict[str, dict] = {}
    if not remote:
        ensure_no_interrupted_builds()
    tests = registered_tests()
    run_executor = executor if executor is not None else execute
    unknown = sorted(set(args.test or []) - set(tests))
    if unknown:
        raise SystemExit(f"unknown check(s): {', '.join(unknown)}")

    for path in config_paths:
        try:
            cfg = load_config(path)
            if historical_gradient_references is not None and cfg.is_rl:
                raise ValueError("historical gradient references require a supervised config")
            if topology is not None:
                from types import SimpleNamespace

                from ..checks.inference_checkpoint_loss import _sampling_topology

                topology = tuple(topology)
                _sampling_topology(SimpleNamespace(cfg=cfg, inference_sampling_topology=topology))
            execution[cfg.config_id] = {
                "backend": "cortex" if remote else "onprem",
                **({"requested_model_name": cfg.model_name} if remote else {}),
            }
            pool = (args.gpu_pool or cfg.n_gpus) if remote else available_gpus()
            if remote:
                if pool < cfg.n_gpus:
                    raise ValueError(f"GPU pool {pool} is smaller than declared training width {cfg.n_gpus}")
            else:
                validate_against_host(cfg, pool, check_gpu_type=not args.any_gpu)
            if training_gpus is not None:
                if cfg.is_rl:
                    raise ValueError("--inference-training-gpus applies only to supervised training configs")
                if training_gpus > pool:
                    raise ValueError(f"requested training width {training_gpus} exceeds GPU pool {pool}")
                cfg.at_gpu_width(training_gpus)
            if checkpoint_gpus is not None:
                if cfg.is_rl:
                    raise ValueError("--checkpoint-training-gpus applies only to supervised training configs")
                if checkpoint_gpus > pool:
                    raise ValueError(f"requested checkpoint training width {checkpoint_gpus} exceeds GPU pool {pool}")
                cfg.at_gpu_width(checkpoint_gpus)
            # Both are read from the config as written. The cases belong to the reviewed spec and do not
            # move with the placement, so every width measures the same work on a different topology.
            arm_defs = arms_for(cfg.max_seq_len, cfg.n_gpus)
            selected_gpus = training_gpus if training_gpus is not None else checkpoint_gpus
            widths = (
                [selected_gpus]
                if selected_gpus is not None
                else (
                    [cfg.n_gpus]
                    if (
                        local_fixture
                        or local_full_model
                        or topology is not None
                        or inference_export_allocation is not None
                        or historical_gradient_references is not None
                    )
                    else placement_widths(cfg.n_gpus, pool)
                )
            )
        except (ValueError, KeyError) as exc:
            results.append(
                TestResult(
                    test_id="-",
                    config_id=config_id(path),
                    outcome=TestOutcome.FAIL,
                    summary="config not runnable on this host",
                    reason=str(exc),
                )
            )
            print(f"skip {path.name}: {exc}", flush=True)
            continue
        if cfg.is_rl:
            if dense_checkpoint_allocation is not None:
                raise ValueError("dense checkpoint allocation does not support reinforcement-learning configs")
            if (
                fixture_path is not None
                or local_full_model
                or (local_fixture and args.test != ["checkpoint-resume-loss"])
            ):
                raise SystemExit("RL local fixtures support only checkpoint-resume-loss on a local Cortex gateway")
            for test in reference_tests().values():
                if args.test and test.test_id not in args.test:
                    continue
                results.append(
                    TestResult(
                        test_id=test.test_id,
                        config_id=cfg.config_id,
                        outcome=TestOutcome.INAPPLICABLE,
                        summary="reinforcement-learning config",
                        reason=RL_INAPPLICABLE_REASON,
                    )
                )
            prepared = _frozen_spec(
                cfg,
                path,
                arm_defs,
                attention=args.attn[0] if args.attn else cfg.attention_implementation,
                cases=getattr(args, "case", None),
                results=results,
                specs=specs,
                test_spec=test_spec,
            )
            if prepared is None:
                continue
            spec, _ = prepared
            if (
                remote
                and "inference-checkpoint-loss" in spec.applicable_tests
                and (not args.test or "inference-checkpoint-loss" in args.test)
            ):
                execution[cfg.config_id]["inference_model_name"] = cfg.model_name
            print(f"=== {cfg.config_id} (reinforcement learning) ===", flush=True)
            with _regression_gateway(args) as gateway_url:
                if gateway_url is not None:
                    backend.base_url = gateway_url
                    execution[cfg.config_id]["gateway_url"] = gateway_url
                published_fixture = None
                if local_fixture:
                    from .cortex import load_local_fixture

                    try:
                        published_fixture = load_local_fixture(backend, spec)
                    except ValueError as exc:
                        results.append(
                            TestResult(
                                test_id="checkpoint-resume-loss",
                                config_id=cfg.config_id,
                                outcome=TestOutcome.FAIL,
                                summary="local Cortex fixture identity unavailable",
                                reason=str(exc),
                            )
                        )
                        continue
                    execution[cfg.config_id].update(
                        requested_model_name=published_fixture["model_name"],
                        fixture_sha256=published_fixture["fixture_sha256"],
                        fixture_distribution="local-gateway",
                        gateway_url=backend.base_url,
                    )
                vocab_size = _run_vocab_size(
                    cfg,
                    spec,
                    args.test,
                    backend,
                    published_fixture,
                    **(
                        {"local_full_model_path": inference_initial_model_dir}
                        if inference_initial_model_dir is not None
                        else {}
                    ),
                )
                results.extend(
                    execute_rl(
                        cfg,
                        spec,
                        vocab_size=vocab_size,
                        select=args.test or None,
                        cases=getattr(args, "case", None),
                        attn=args.attn[0] if args.attn else None,
                        client_backend=backend,
                        **({"local_cortex_fixture": True} if local_fixture else {}),
                        **({"inference_sampling_topology": topology} if topology is not None else {}),
                        **(
                            {"inference_export_allocation": inference_export_allocation}
                            if inference_export_allocation is not None
                            else {}
                        ),
                        **(
                            {"inference_initial_model_dir": inference_initial_model_dir}
                            if inference_initial_model_dir is not None
                            else {}
                        ),
                    )
                )
            continue
        prepared = _frozen_spec(
            cfg,
            path,
            arm_defs,
            attention=args.attn[0] if args.attn else cfg.attention_implementation,
            cases=getattr(args, "case", None),
            results=results,
            specs=specs,
            test_spec=test_spec,
        )
        if prepared is None:
            continue
        spec, selected_cases = prepared
        if (
            remote
            and "inference-checkpoint-loss" in spec.applicable_tests
            and (not args.test or "inference-checkpoint-loss" in args.test)
        ):
            execution[cfg.config_id]["inference_model_name"] = cfg.model_name
        published_fixture = None
        local_full_model_path = None
        if local_full_model:
            local_full_model_path = validate_local_full_model(backend, cfg, spec)
            execution[cfg.config_id].update(
                requested_model_name=cfg.model_name,
                local_full_model_path=str(local_full_model_path),
                full_model_distribution="local-gateway",
                gateway_url=backend.base_url,
            )
        if fixture_path is not None:
            from .cortex import load_fixture

            try:
                published_fixture = load_fixture(fixture_path, spec)
            except (OSError, ValueError) as exc:
                results.append(
                    TestResult(
                        test_id="-",
                        config_id=cfg.config_id,
                        outcome=TestOutcome.FAIL,
                        summary="published fixture identity unavailable",
                        reason=str(exc),
                    )
                )
                print(f"skip {path.name}: {exc}", flush=True)
                continue
            execution[cfg.config_id].update(
                requested_model_name=published_fixture["model_name"],
                fixture_sha256=published_fixture["fixture_sha256"],
                fixture_stage_path=published_fixture["publication"]["stage_path"],
            )
        with _regression_gateway(args) as gateway_url:
            if gateway_url is not None:
                backend.base_url = gateway_url
                execution[cfg.config_id]["gateway_url"] = gateway_url
            if local_fixture:
                from .cortex import load_local_fixture

                try:
                    published_fixture = load_local_fixture(backend, spec)
                except ValueError as exc:
                    results.extend(
                        [
                            TestResult(
                                test_id=test_id,
                                config_id=cfg.config_id,
                                outcome=TestOutcome.FAIL,
                                summary="local Cortex fixture identity unavailable",
                                reason=str(exc),
                            )
                            for test_id in args.test
                        ]
                    )
                    print(f"skip {path.name}: {exc}", flush=True)
                    continue
                execution[cfg.config_id].update(
                    requested_model_name=published_fixture["model_name"],
                    fixture_sha256=published_fixture["fixture_sha256"],
                    fixture_distribution="local-gateway",
                    gateway_url=backend.base_url,
                )

            print(f"=== {cfg.config_id} ===", flush=True)
            vocab_size = _run_vocab_size(
                cfg,
                spec,
                args.test,
                backend,
                published_fixture,
                **(
                    {"local_full_model_path": inference_initial_model_dir}
                    if inference_initial_model_dir is not None
                    else ({"local_full_model_path": local_full_model_path} if local_full_model else {})
                ),
            )

            for width in widths:
                multiple = width // cfg.n_gpus
                run_cfg = cfg.at_gpu_width(width)
                run_cases = scaled_for_placement(selected_cases, multiple)
                if len(widths) > 1:
                    print(f"--- {cfg.config_id} on {width} GPU(s) ---", flush=True)
                results.extend(
                    run_executor(
                        run_cfg,
                        spec,
                        run_cases,
                        order=args.order,
                        fail_fast=args.fail_fast,
                        vocab_size=vocab_size,
                        select=args.test or None,
                        attn=args.attn[0] if args.attn else None,
                        client_backend=backend,
                        **({"cortex_fixture": fixture_path} if fixture_path is not None else {}),
                        **({"local_cortex_fixture": True} if local_fixture else {}),
                        **({"local_gateway_full_model": True} if local_full_model else {}),
                        **({"inference_sampling_topology": topology} if topology is not None else {}),
                        **(
                            {"inference_export_allocation": inference_export_allocation}
                            if inference_export_allocation is not None
                            else {}
                        ),
                        **(
                            {"inference_initial_model_dir": inference_initial_model_dir}
                            if inference_initial_model_dir is not None
                            else {}
                        ),
                        **(
                            {"reference_records_dir": Path(args.reference_records_dir)}
                            if getattr(args, "reference_records_dir", None) is not None
                            else {}
                        ),
                        **(
                            {"peft_adapter": Path(args.peft_adapter)}
                            if getattr(args, "peft_adapter", None) is not None
                            else {}
                        ),
                        **(
                            {"historical_gradient_references": historical_gradient_references}
                            if historical_gradient_references is not None
                            else {}
                        ),
                        **(
                            {"dense_checkpoint_allocation": copy.deepcopy(dense_checkpoint_allocation)}
                            if dense_checkpoint_allocation is not None
                            else {}
                        ),
                    )
                )

    if snapshot is not None:
        snapshot(results, specs, execution)
    return _finish(results, specs, Path(args.out), configs=len(config_paths), started=started, execution=execution)


def _case_summary(case, results, *, status=None, reason="") -> dict:
    matching = [result for result in results if result.test_id == case.test_id]
    if status is None:
        failures = [result for result in matching if result.outcome is TestOutcome.FAIL]
        if failures:
            blocked_summaries = {
                "published Cortex fixture required",
                "check not supported on Cortex",
                "local Cortex fixture identity unavailable",
            }
            if all(
                result.summary in blocked_summaries
                or result.error_type == "ReferencePrerequisiteError"
                or result.summary.startswith("ReferencePrerequisiteError:")
                for result in failures
            ):
                status = "blocked"
            elif any(result.error_type or result.summary.startswith("execution error:") for result in failures):
                status = "error"
            else:
                status = "fail"
        elif any(result.outcome is TestOutcome.PASS for result in matching):
            status = "pass"
        elif matching:
            status = "inapplicable"
        else:
            status = "blocked"
            reason = reason or "no comparison was produced; see full.log"
    return {
        "config_id": case.config_id,
        "test_id": case.test_id,
        "status": status,
        "reason": (
            reason
            or "; ".join(result.reason or result.summary for result in matching if result.reason or result.summary)
        ),
    }


@contextmanager
def _saved_output(log):
    """Keep logger tracebacks with redirected prints instead of flooding the terminal."""
    logger = logging.getLogger()
    previous = logger.handlers[:]
    handler = logging.StreamHandler(log)
    logger.handlers = [handler]
    try:
        with redirect_stdout(log), redirect_stderr(log):
            yield
    finally:
        logger.handlers = previous
        handler.flush()
        handler.close()


def run_batch(args, *, selected, executor=None):
    """Orchestrate independent groups without replacing the numerical runner."""
    from ..api import BatchResult
    from ..api import aggregate_exit_code
    from .reference_records import ReferencePrerequisiteError

    started = time.monotonic()
    _load_dense_checkpoint_allocation(args)
    _load_historical_gradient_references(args)
    out = Path(args.out) if args.out is not None else correctness_logdir("ap-correctness-run-")
    out.mkdir(parents=True, exist_ok=True)
    specs = {}
    execution = {}
    results = []
    cases = []
    stopped = False
    # Per-arm checks share one prepared workload identity, including optimizer options.
    # Whole-run checks remain independent so a missing step record cannot hide them.
    groups = []
    for path in dict.fromkeys(case.config_path for case in selected):
        config_cases = [case for case in selected if case.config_path == path]
        step_cases = [case for case in config_cases if registered_tests()[case.test_id].per_arm is not None]
        steps_added = False
        for case in config_cases:
            if case in step_cases:
                if not steps_added:
                    groups.append(step_cases)
                    steps_added = True
            else:
                groups.append([case])
    print(f"Selected {len(selected)} configuration/test pair(s); running sequentially.", flush=True)
    print("Selected tests: " + ", ".join(dict.fromkeys(case.test_id for case in selected)), flush=True)
    with (out / "full.log").open("w", encoding="utf-8") as log:
        for index, group in enumerate(groups):
            group_results = []
            failure = None
            if stopped:
                failure = ("blocked", "not started: the earlier operation was interrupted or cleanup was not verified")
            else:
                run_args = copy.copy(args)
                run_args.config = str(group[0].config_path)
                run_args.all_configs = False
                run_args.resolved_paths = [group[0].config_path]
                run_args.test = [case.test_id for case in group]
                run_args.out = out / "cases" / group[0].config_id / str(index)

                def collect(produced, reviewed, metadata):
                    group_results.extend(produced)
                    specs.update(reviewed)
                    execution.update(metadata)

                with _saved_output(log):
                    try:
                        _run_configs(run_args, executor=executor, snapshot=collect)
                    except KeyboardInterrupt:
                        traceback.print_exc()
                        failure = ("error", "interrupted by operator; owned context cleanup was attempted")
                        stopped = True
                    except ReferencePrerequisiteError as exc:
                        traceback.print_exc()
                        failure = ("blocked", str(exc))
                    except (FileNotFoundError, ValueError, KeyError, SystemExit) as exc:
                        traceback.print_exc()
                        failure = ("blocked", str(exc))
                    except Exception as exc:
                        traceback.print_exc()
                        failure = ("error", f"{type(exc).__name__}: {exc}")
                        # An exception escaping the runner may be a failed context-manager
                        # exit. Conservatively stop rather than reuse possibly owned resources.
                        stopped = True
                    log.flush()
                generic = [result for result in group_results if result.test_id == "-"]
                if generic:
                    failure = ("blocked", "; ".join(result.reason or result.summary for result in generic))
                    group_results = [result for result in group_results if result.test_id != "-"]
            for case in group:
                if failure is not None:
                    group_results.append(
                        TestResult(
                            test_id=case.test_id,
                            config_id=case.config_id,
                            outcome=TestOutcome.FAIL,
                            summary=failure[0],
                            reason=failure[1],
                        )
                    )
                summary = _case_summary(
                    case,
                    group_results,
                    **({"status": failure[0], "reason": failure[1]} if failure is not None else {}),
                )
                if summary["status"] == "blocked" and not any(r.test_id == case.test_id for r in group_results):
                    group_results.append(
                        TestResult(
                            case.test_id, case.config_id, TestOutcome.FAIL, summary="blocked", reason=summary["reason"]
                        )
                    )
                cases.append(summary)
                log.write(f"{summary['status'].upper()} {case.config_id} {case.test_id}: {summary['reason']}\n")
                print(f"{summary['status'].upper():12s} {case.config_id}  {case.test_id}", flush=True)
                if summary["status"] in {"blocked", "error", "fail"} and summary["reason"]:
                    print(f"  Reason: {summary['reason']}\n", flush=True)
            results.extend(group_results)
    write_report(results, specs, out, execution=execution)
    elapsed = time.monotonic() - started
    counts = {
        name: sum(case["status"] == name for case in cases)
        for name in ("pass", "fail", "blocked", "error", "inapplicable")
    }
    (out / "batch.json").write_text(
        json.dumps(
            {"cases": cases, "counts": counts, "elapsed_seconds": elapsed, "stopped_unsafe": stopped},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        ", ".join(f"{count} {name}" for name, count in counts.items()) + f"; {console.duration(elapsed)}", flush=True
    )
    print(f"Report: {out / 'report.md'}; complete logs: {out / 'full.log'}", flush=True)
    if any(case["status"] in {"blocked", "error"} for case in cases):
        print("Action required: see the reason in batch.json and the full traceback in full.log.", flush=True)
    print(f"Total run time: {console.total_duration(time.monotonic() - started)}", flush=True)
    return BatchResult(results, out, aggregate_exit_code(results), elapsed, stopped, cases)


def cmd_run(args) -> int:
    """Compatibility for library callers; operator commands use the supported API."""
    if not getattr(args, "compact_output", False):
        return _run_configs(args)
    from ..api import options_from_args
    from ..api import request_from_args
    from ..api import run

    try:
        return run(request_from_args(args), options_from_args(args)).exit_code
    except ValueError as exc:
        print(f"Cannot run: {exc}", file=sys.stderr)
        return 2


def cmd_hosted(args) -> int:
    """Backward-compatible spelling of the Cortex-default run command."""
    args.backend = "cortex"
    return cmd_run(args)


def cmd_list(args) -> int:
    from ..api import catalog

    print("Configurations:")
    for identifier in catalog(Path(args.config_dir)):
        print(f"  {identifier}")
    print("Tests:")
    for test in sorted(registered_tests().values(), key=lambda test: test.test_id):
        print(f"{test.test_id:24s} {test.title}")
        print(f"{'':24s} criterion: {test.criterion}")
    return 0


def cmd_onboard(args) -> int:
    from ..onboarding.onboard import logged_run

    return logged_run(args)


def cmd_selftest(args) -> int:
    """Run the harness's own checks.

    They live in the package rather than in the repository's test tree, and are named ``check_*.py``, so a
    unit-test run cannot collect a correctness check. Supplying the collection pattern is therefore this
    command's job.
    """
    import pytest

    selftest_dir = CORRECTNESS_DIR / "selftest"
    runner_args = list(args.runner_args)
    has_basetemp = False
    for index, argument in enumerate(runner_args):
        if argument == "--basetemp":
            if index + 1 == len(runner_args):
                raise ValueError("--basetemp requires a path under /data-fast")
            require_data_fast_path(runner_args[index + 1], label="selftest temporary root")
            has_basetemp = True
        elif argument.startswith("--basetemp="):
            require_data_fast_path(argument.split("=", 1)[1], label="selftest temporary root")
            has_basetemp = True
    if not has_basetemp:
        runner_args.extend(["--basetemp", str(correctness_workdir("ap-correctness-selftest-") / "pytest")])
    return pytest.main(["-sv", "--instafail", "-o", "python_files=check_*.py", str(selftest_dir), *runner_args])


def cmd_publish_fixture(args) -> int:
    """Use the managed dev-model uploader on a disposable copy of reviewed HF weights."""
    from ..onboarding.cortex_fixture_upload import publish_fixture

    cfg = load_config(Path(args.config))
    spec = TestSpec.read(SPEC_DIR / f"{cfg.config_id}.json")
    if cfg.is_rl or cfg.training.get("peft_config"):
        raise SystemExit("fixture publication currently supports full supervised models only")
    if spec.config_checksum != config_checksum(cfg):
        raise SystemExit("config checksum does not match the reviewed spec")
    staging_root = Path(args.staging_dir).resolve()
    if Path("/data-fast") not in staging_root.parents:
        raise SystemExit("--staging-dir must be under /data-fast; do not put model weights in the checkout")
    try:
        binding = publish_fixture(
            spec, staging_root, Path(args.uploader), args.uploader_python, Path(args.out), stage=args.stage
        )
    except (OSError, ValueError) as exc:
        print(f"fixture publication failed: {exc}", flush=True)
        return 1
    print(f"published reduced model: {binding['model_name']}", flush=True)
    print(f"binding: {args.out}", flush=True)
    print("Verify the dev model-cache rollout before running the Cortex target.", flush=True)
    return 0


def operator_backend(value: str) -> str:
    """Keep the public backend name separate from the internal transport identifier."""
    if value in {"cortex-training", "cortex"}:
        return "cortex"
    raise argparse.ArgumentTypeError("use cortex-training; the onprem correctness backend is no longer supported")


def add_connection_arguments(parser, *, default_backend="cortex-training"):
    parser.add_argument(
        "--backend",
        type=operator_backend,
        default=default_backend,
        metavar="cortex-training",
        help="Cortex-training backend (default); cortex remains a compatibility alias",
    )
    parser.add_argument("--host", help="Snowflake account host; otherwise ARCTIC_CORTEX_HOST")
    parser.add_argument("--database", help="Snowflake database; otherwise ARCTIC_CORTEX_DATABASE")
    parser.add_argument("--schema", help="Snowflake schema; otherwise ARCTIC_CORTEX_SCHEMA")
    parser.add_argument("--endpoint", help="SnowAPI endpoint; default cortex-training")
    parser.add_argument("--base-url", help="direct Cortex development URL without PAT authentication")
    parser.add_argument("--pat-file", help="PAT file; otherwise ARCTIC_CORTEX_PAT, never logged")
    parser.add_argument("--gpu-pool", type=int, help="remote training GPU placement pool; default declared width")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="arctic_platform.correctness")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--config", help="exact catalog ID or path to one Arctic Platform job config")
        p.add_argument("--all-configs", action="store_true", help="every config in --config-dir")
        p.add_argument(
            "--config-dir",
            default=str(CONFIG_DIR),
            help="root of the model/gpu-type/workload config tree, searched recursively",
        )
        p.add_argument(
            "--order",
            choices=["batched", "per-arm"],
            default="batched",
            help=(
                "batched pays one gateway startup for all arms; per-arm runs "
                "reference then Arctic Platform for one arm before the next begins"
            ),
        )
        p.add_argument("--fail-fast", action="store_true", help="stop after the first arm whose verdict is not a pass")
        p.add_argument("--attn", action="append", help="override attn_implementation (repeatable)")
        p.add_argument(
            "--any-gpu",
            action="store_true",
            help=(
                "run a config on a GPU type its path does not name; the budgets and offload "
                "choices in it were sized for another accelerator"
            ),
        )

    r = sub.add_parser("run", help="run tests over the config-by-test matrix")
    shared(r)
    add_connection_arguments(r)
    r.add_argument(
        "--test-spec", type=Path, help="reviewed private spec for one explicit config; integrity checks apply"
    )
    r.add_argument(
        "--peft-adapter",
        type=Path,
        help="optional standard PEFT adapter under /data-fast; otherwise generate upstream-default LoRA factors",
    )
    r.add_argument(
        "--checkpoint-training-gpus",
        type=int,
        help="explicit supervised training placement for checkpoint-resume-loss; requires a sufficient GPU pool",
    )
    r.add_argument(
        "--inference-training-gpus",
        type=int,
        help="explicit supervised training placement for inference-checkpoint-loss; requires a sufficient GPU pool",
    )
    r.add_argument(
        "--inference-sampling-topology",
        type=int,
        nargs=3,
        metavar=("GPUS", "TP", "PP"),
        help="explicit inference sampling allocation, tensor parallelism and pipeline parallelism",
    )
    r.add_argument("--inference-export-allocation", type=Path, help="same-job local inference allocation JSON")
    r.add_argument("--inference-export-allocation-sha256", help="full SHA256 of the allocation JSON bytes")
    r.add_argument("--dense-checkpoint-allocation", type=Path, help="same-job local dense optimizer allocation JSON")
    r.add_argument(
        "--dense-checkpoint-allocation-sha256", help="full SHA256 of the dense optimizer allocation JSON bytes"
    )
    r.add_argument(
        "--inference-initial-model-dir",
        type=Path,
        help="already admitted absolute FAST model directory for initial local Cortex inference training only",
    )
    r.add_argument(
        "--historical-gradient-references", type=Path, help="explicit reviewed historical gradient bindings JSON"
    )
    r.add_argument("--historical-gradient-references-sha256", help="full SHA256 of the historical bindings JSON bytes")
    r.add_argument("--test", action="append", help="test id (repeatable); default every applicable test")
    r.add_argument(
        "--case",
        action="append",
        help=(
            "case name to run, e.g. gas1 (repeatable); default every case in the spec. "
            "Use this to see the output quickly: gas1 is one sequence and no accumulation"
        ),
    )
    r.add_argument(
        "--out",
        type=Path,
        help="report directory; default a unique directory under ./work/logs (or DSS_CORRECTNESS_LOGDIR)",
    )
    r.add_argument(
        "--reference-records-dir",
        type=Path,
        help="prepared HF records inside the checkout; missing records require explicit onboarding",
    )
    r.add_argument(
        "--cortex-fixture",
        type=Path,
        help="verified publication binding from a supported publisher; keeps HF local and selects the Cortex model",
    )
    r.add_argument(
        "--local-cortex-fixture",
        action="store_true",
        help="bind reviewed local HF weights to the existing loopback gateway for single-step comparisons",
    )
    r.add_argument(
        "--local-gateway-full-model",
        action="store_true",
        help="use the pinned original offline model on the existing loopback gateway for e2e/inference",
    )
    r.set_defaults(func=cmd_run, compact_output=True)

    publish = sub.add_parser("publish-fixture", help="publish reviewed reduced weights as a separate dev model")
    publish.add_argument("--config", required=True)
    publish.add_argument("--staging-dir", default="/data-fast/tmp/ap-cortex-fixtures")
    publish.add_argument("--uploader", required=True, help="Neutrino scripts/upload_model.py path")
    publish.add_argument("--uploader-python", default=sys.executable, help="interpreter with uploader dependencies")
    publish.add_argument(
        "--stage", required=True, help="explicit destination stage; authentication belongs to the uploader"
    )
    publish.add_argument("--out", required=True, help="new publication receipt JSON path")
    publish.set_defaults(func=cmd_publish_fixture)

    list_parser = sub.add_parser("list", help="show registered tests")
    list_parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    list_parser.set_defaults(func=cmd_list)

    h = sub.add_parser(
        "hosted",
        help="run Arctic Platform-against-Arctic Platform checks on the hosted server, which supplies the GPUs",
    )
    shared(h)
    add_connection_arguments(h)
    h.add_argument(
        "--test",
        action="append",
        required=True,
        help="check id to run (repeatable); requires reviewed applicability",
    )
    h.add_argument(
        "--case", action="append", help="case name to run, e.g. gas1 (repeatable); default every case in the spec"
    )
    h.add_argument("--out", default="correctness-report-hosted")
    h.set_defaults(func=cmd_hosted, compact_output=True)

    s = sub.add_parser("selftest", help="check the harness itself; no GPU and no config needed")
    s.add_argument("runner_args", nargs="*", help="extra arguments forwarded to the runner")
    s.set_defaults(func=cmd_selftest)

    from ..onboarding import add_arguments

    o = sub.add_parser("onboard", help="size, calibrate, and freeze one config")
    add_arguments(o)
    o.set_defaults(func=cmd_onboard)

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)
