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

from argparse import Namespace
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable
from typing import Sequence

from .harness.config import config_id
from .harness.registry import TestOutcome
from .harness.registry import TestResult
from .harness.registry import registered_tests

PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunRequest:
    config: str | Path | None = None
    tests: tuple[str, ...] = ()
    all_configs: bool = False
    config_dir: Path = PACKAGE_DIR / "configs"


@dataclass(frozen=True)
class RunOptions:
    backend: str = "cortex"
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
    out: Path | None = None
    reference_records_dir: Path | None = None
    cortex_fixture: Path | None = None
    local_cortex_fixture: bool = False
    local_gateway_full_model: bool = False
    test_spec: Path | None = None
    inference_sampling_topology: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class SelectedCase:
    config_id: str
    config_path: Path
    test_id: str


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
    for path in sorted(root.rglob("*.config")):
        identifier = config_id(path)
        if "archive" in path.relative_to(root).parts:
            continue
        if not (PACKAGE_DIR / "specs" / f"{identifier}.json").is_file():
            continue
        if identifier in found:
            raise ValueError(f"duplicate configuration ID: {identifier}")
        found[identifier] = path
    return dict(sorted(found.items()))


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
        elif Path(selector).is_file() and Path(selector).suffix == ".config":
            paths = [Path(selector)]
        else:
            raise ValueError(f"unknown configuration: {selector}; use the list command or an existing .config path")
    else:
        paths = list(configs.values())
    if not paths:
        raise ValueError(f"no active reviewed configurations in {request.config_dir}")
    test_ids = sorted(set(request.tests)) if request.tests else sorted(tests)
    return tuple(
        SelectedCase(config_id(path), path, test_id) for path in sorted(paths, key=config_id) for test_id in test_ids
    )


def _namespace(request: RunRequest, options: RunOptions) -> Namespace:
    values = asdict(options)
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
    return RunOptions(**values)


def run(request: RunRequest, options: RunOptions | None = None, *, executor: Callable | None = None) -> BatchResult:
    """Run selected cases sequentially with short output and complete saved logs."""
    from .harness.execution import run_batch

    options = options or RunOptions()
    if options.test_spec is not None and (request.config is None or request.all_configs):
        raise ValueError("--test-spec requires one explicit --config")
    if options.inference_sampling_topology is not None and (
        request.config is None or request.all_configs or request.tests != ("inference-checkpoint-loss",)
    ):
        raise ValueError("--inference-sampling-topology requires one config and only inference-checkpoint-loss")
    selected = select(request)
    return run_batch(_namespace(request, options), selected=selected, executor=executor)


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
