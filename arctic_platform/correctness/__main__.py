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

"""Onboard configs and run the config-by-check correctness matrix."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import SecretStr

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig

from .harness import console
from .harness.arms import ArmDefinition
from .harness.arms import arms_for
from .harness.arms import scaled_for_placement
from .harness.config import config_checksum
from .harness.config import load_config
from .harness.config import local_gpu_type
from .harness.config import placement_widths
from .harness.config import validate_against_host
from .harness.dss_driver import available_gpus
from .harness.hosted import read_pat
from .harness.hosted import require_local_cortex_gateway
from .harness.preflight import ensure_no_interrupted_builds
from .harness.registry import RL_INAPPLICABLE_REASON
from .harness.registry import TestOutcome
from .harness.registry import TestResult
from .harness.registry import reference_tests
from .harness.registry import registered_tests
from .harness.report import write as write_report
from .harness.runner import assert_applicable_tests_registered
from .harness.runner import cortex_unavailable_reason
from .harness.runner import execute
from .harness.runner import execute_rl
from .harness.runner import validate_local_full_model
from .harness.spec import TestSpec
from .harness.workdir import correctness_workdir
from .harness.workdir import require_data_fast_path

CORRECTNESS_DIR = Path(__file__).resolve().parent
CONFIG_DIR = CORRECTNESS_DIR / "configs"
SPEC_DIR = CORRECTNESS_DIR / "specs"


def _specs_for(config_paths: List[Path]) -> Dict[str, Path]:
    return {p.stem: SPEC_DIR / f"{p.stem}.json" for p in config_paths}


def _resolve_configs(args, *, filter_by_local_gpu: bool = True) -> List[Path]:
    """The configs this invocation runs. Local-GPU filtering applies only where the jobs use local GPUs."""
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
    return [Path(args.config)]


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
) -> Optional[Tuple[TestSpec, List[ArmDefinition]]]:
    """This config's reviewed spec and the cases to run, or ``None`` once the reason it cannot run is recorded.

    The spec is what makes a result a regression rather than a measurement: it names the configuration that
    was reviewed, the cases derived from it, and the attention backend onboarding covered. A config whose
    content, cases, or backend no longer match it is refused here, because a verdict about a configuration
    nobody reviewed says nothing.
    """
    spec_path = SPEC_DIR / f"{cfg.config_id}.json"
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

    spec_checksum = case_checksum(spec.arms)
    current_checksum = case_checksum(arm_defs)
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

    if not cases:
        return spec, arm_defs
    wanted = set(cases)
    unknown = wanted - {a.name for a in arm_defs}
    if unknown:
        print(
            f"skip {path.name}: unknown case(s) {', '.join(sorted(unknown))}; "
            f"this config has {', '.join(a.name for a in arm_defs)}",
            flush=True,
        )
        return None
    return spec, [a for a in arm_defs if a.name in wanted]


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
    print(f"total run time across {configs} config(s): {console.duration(time.monotonic() - started)}", flush=True)
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
    return CortexConfig(**options)


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


def cmd_run(args) -> int:
    started = time.monotonic()
    backend = _client_backend(args)
    if getattr(args, "reference_records_dir", None) is not None:
        from .harness.reference_records import require_git_records_dir

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
        or not set(args.test) <= {"single-step-grads", "single-step-optimizer"}
    ):
        raise SystemExit("--local-cortex-fixture requires one Cortex config and only single-step-grads/optimizer")
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
    results: List[TestResult] = []
    specs: Dict[str, TestSpec] = {}
    execution: Dict[str, dict] = {}
    if not remote:
        ensure_no_interrupted_builds()
    tests = registered_tests()
    unknown = sorted(set(args.test or []) - set(tests))
    if unknown:
        raise SystemExit(f"unknown check(s): {', '.join(unknown)}")

    for path in config_paths:
        try:
            cfg = load_config(path)
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
            # Both are read from the config as written. The cases belong to the reviewed spec and do not
            # move with the placement, so every width measures the same work on a different topology.
            arm_defs = arms_for(cfg.max_seq_len, cfg.n_gpus)
            widths = [cfg.n_gpus] if local_fixture or local_full_model else placement_widths(cfg.n_gpus, pool)
        except (ValueError, KeyError) as exc:
            results.append(
                TestResult(
                    test_id="-",
                    config_id=path.stem,
                    outcome=TestOutcome.INAPPLICABLE,
                    summary="config not runnable on this host",
                    reason=str(exc),
                )
            )
            print(f"skip {path.name}: {exc}", flush=True)
            continue
        if cfg.is_rl:
            if fixture_path is not None or local_fixture or local_full_model:
                raise SystemExit("--cortex-fixture currently supports supervised single-step-grads only")
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
                cases=None,
                results=results,
                specs=specs,
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
            vocab_size = _run_vocab_size(cfg, spec, args.test, backend)
            results.extend(
                execute_rl(
                    cfg,
                    spec,
                    vocab_size=vocab_size,
                    select=args.test or None,
                    attn=args.attn[0] if args.attn else None,
                    client_backend=backend,
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
            from .harness.cortex_fixture import load_fixture

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
        if local_fixture:
            from .harness.cortex_fixture import load_local_fixture

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
            **({"local_full_model_path": local_full_model_path} if local_full_model else {}),
        )

        for width in widths:
            multiple = width // cfg.n_gpus
            run_cfg = cfg.at_gpu_width(width)
            run_cases = scaled_for_placement(selected_cases, multiple)
            if len(widths) > 1:
                print(f"--- {cfg.config_id} on {width} GPU(s) ---", flush=True)
            results.extend(
                execute(
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
                    **(
                        {"reference_records_dir": Path(args.reference_records_dir)}
                        if getattr(args, "reference_records_dir", None) is not None
                        else {}
                    ),
                )
            )

    return _finish(results, specs, Path(args.out), configs=len(config_paths), started=started, execution=execution)


def cmd_hosted(args) -> int:
    """Backward-compatible spelling of the Cortex-default run command."""
    args.backend = "cortex"
    return cmd_run(args)


def cmd_list(args) -> int:
    for test in registered_tests().values():
        print(f"{test.test_id:24s} {test.title}")
        print(f"{'':24s} criterion: {test.criterion}")
    return 0


def cmd_onboard(args) -> int:
    from .onboarding import run

    return run(args)


def cmd_selftest(args) -> int:
    """Run the harness's own checks.

    They live in the package rather than in the repository's test tree, and are named ``check_*.py``, so a
    unit-test run cannot collect a correctness check. Supplying the collection pattern is therefore this
    command's job.
    """
    import pytest

    selftest_dir = Path(__file__).resolve().parent / "selftest"
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
    from .onboarding.cortex_fixture_upload import publish_fixture

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="arctic_platform.correctness")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--config", help="path to one Arctic Platform job config")
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

    def connection(p):
        p.add_argument("--backend", choices=["cortex", "onprem"], default="cortex")
        p.add_argument("--host", help="Snowflake account host; otherwise ARCTIC_CORTEX_HOST")
        p.add_argument("--database", help="Snowflake database; otherwise ARCTIC_CORTEX_DATABASE")
        p.add_argument("--schema", help="Snowflake schema; otherwise ARCTIC_CORTEX_SCHEMA")
        p.add_argument("--endpoint", help="SnowAPI endpoint; default cortex-training")
        p.add_argument("--base-url", help="direct Cortex development URL without PAT authentication")
        p.add_argument("--pat-file", help="PAT file; otherwise ARCTIC_CORTEX_PAT, never logged")
        p.add_argument("--gpu-pool", type=int, help="remote training GPU placement pool; default declared width")

    r = sub.add_parser("run", help="run tests over the config-by-test matrix")
    shared(r)
    connection(r)
    r.add_argument("--test", action="append", help="test id (repeatable); default every applicable test")
    r.add_argument(
        "--case",
        action="append",
        help=(
            "case name to run, e.g. gas1 (repeatable); default every case in the spec. "
            "Use this to see the output quickly: gas1 is one sequence and no accumulation"
        ),
    )
    r.add_argument("--out", default="correctness-report")
    r.add_argument(
        "--reference-records-dir",
        type=Path,
        help="Git-tracked prepared HF records; missing records require explicit onboarding",
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
    r.set_defaults(func=cmd_run)

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
    list_parser.set_defaults(func=cmd_list)

    h = sub.add_parser(
        "hosted",
        help="run Arctic Platform-against-Arctic Platform checks on the hosted server, which supplies the GPUs",
    )
    shared(h)
    connection(h)
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
    h.set_defaults(func=cmd_hosted)

    s = sub.add_parser("selftest", help="check the harness itself; no GPU and no config needed")
    s.add_argument("runner_args", nargs="*", help="extra arguments forwarded to the runner")
    s.set_defaults(func=cmd_selftest)

    from .onboarding import add_arguments

    o = sub.add_parser("onboard", help="size, calibrate, and freeze one config")
    add_arguments(o)
    o.set_defaults(func=cmd_onboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
