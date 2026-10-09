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
import os
import sys
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

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
from .harness.hosted import HostedConnection
from .harness.hosted import HostedTransport
from .harness.hosted import read_pat
from .harness.preflight import ensure_no_interrupted_builds
from .harness.registry import RL_INAPPLICABLE_REASON
from .harness.registry import TestOutcome
from .harness.registry import TestResult
from .harness.registry import reference_tests
from .harness.registry import registered_tests
from .harness.report import write as write_report
from .harness.runner import RunContext
from .harness.runner import assert_applicable_tests_registered
from .harness.runner import execute
from .harness.runner import execute_rl
from .harness.spec import TestSpec
from .harness.workdir import correctness_workdir

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


def _finish(results: List[TestResult], specs: Dict[str, TestSpec], out: Path, *, configs: int, started: float) -> int:
    """Write the report and print what the run concluded. Zero only when something passed and none failed."""
    print(f"total run time across {configs} config(s): {console.duration(time.monotonic() - started)}", flush=True)
    write_report(results, specs, out)
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


def cmd_run(args) -> int:
    from transformers import AutoConfig

    started = time.monotonic()
    config_paths = _resolve_configs(args)
    results: List[TestResult] = []
    specs: Dict[str, TestSpec] = {}
    ensure_no_interrupted_builds()

    for path in config_paths:
        try:
            cfg = load_config(path)
            pool = available_gpus()
            validate_against_host(cfg, pool, check_gpu_type=not args.any_gpu)
            # Both are read from the config as written. The cases belong to the reviewed spec and do not
            # move with the placement, so every width measures the same work on a different topology.
            arm_defs = arms_for(cfg.max_seq_len, cfg.n_gpus)
            widths = placement_widths(cfg.n_gpus, pool)
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
            print(f"=== {cfg.config_id} (reinforcement learning) ===", flush=True)
            with console.activity("Loading model config"):
                model_cfg = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
                vocab_size = getattr(model_cfg, "text_config", model_cfg).vocab_size
            results.extend(
                execute_rl(
                    cfg,
                    spec,
                    vocab_size=vocab_size,
                    select=args.test or None,
                    attn=args.attn[0] if args.attn else None,
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

        print(f"=== {cfg.config_id} ===", flush=True)
        with console.activity("Loading model config"):
            model_cfg = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
            vocab_size = getattr(model_cfg, "text_config", model_cfg).vocab_size

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
                )
            )

    return _finish(results, specs, Path(args.out), configs=len(config_paths), started=started)


def _hosted_placement_pool() -> int:
    """The GPU pool a hosted placement widens into.

    A local run reads the allocation from the gateway hostfile once ``DSS_GATEWAY_URL`` is set. The hosted
    command starts no gateway, so the same file is read directly when ``DSS_GATEWAY_HOSTFILE`` names it.
    Without that variable the pool is this process's own devices, and a config already as wide as those
    devices has no second slot.
    """
    hostfile = os.environ.get("DSS_GATEWAY_HOSTFILE")
    if hostfile and Path(hostfile).is_file():
        from arctic_platform.correctness.harness.hostfile import parse_hostfile

        slots = sum(entry.slots for entry in parse_hostfile(hostfile))
        if slots:
            return slots
    return available_gpus()


def cmd_hosted(args) -> int:
    """Run named checks against the hosted Arctic Platform server, which allocates the GPUs each job declares.

    The hosted server owns the checkpoint stage, which is why a job can be initialized from another job's
    checkpoint here and not through a gateway a client drives directly: the server mints the stage and
    stamps the scoped credentials onto the sub-job config before a zone reads it.

    The checks named on the command line run whether or not a config's spec lists them as applicable. That
    list records what onboarding measured against the single-GPU reference, and a check comparing Arctic Platform
    against Arctic Platform has no reference arm to onboard. Everything else the spec fixes still gates: a config whose
    content, cases, or attention backend has drifted from its reviewed spec is refused rather than run.

    A wider multiple runs when ``DSS_GATEWAY_HOSTFILE`` names a pool that fits one, with the batch identity
    and the case row counts scaled for that width, and that run does not repeat the declared width. A pool
    with no wider multiple runs the declared width, which is the only placement that fits. The file is not
    written.
    """
    started = time.monotonic()
    tests = registered_tests()
    unknown = sorted(set(args.test) - set(tests))
    if unknown:
        raise SystemExit(f"unknown check(s): {', '.join(unknown)}. Registered: {', '.join(sorted(tests))}")
    chosen = [tests[test_id] for test_id in args.test]
    against_reference = [t.test_id for t in chosen if t.compares_to_reference]
    if against_reference:
        raise SystemExit(
            f"{', '.join(against_reference)} compares Arctic Platform against the single-GPU Hugging Face reference, "
            "which runs on this node and not on the hosted server. Run those with the run command."
        )
    connection = HostedConnection(
        host=args.host, database=args.database, schema=args.schema, pat=read_pat(args.pat_file), endpoint=args.endpoint
    )

    config_paths = _resolve_configs(args, filter_by_local_gpu=False)
    results: List[TestResult] = []
    specs: Dict[str, TestSpec] = {}
    for path in config_paths:
        try:
            cfg = load_config(path)
            arm_defs = arms_for(cfg.max_seq_len, cfg.n_gpus)
        except (ValueError, KeyError) as exc:
            results.append(
                TestResult(
                    test_id="-",
                    config_id=path.stem,
                    outcome=TestOutcome.INAPPLICABLE,
                    summary="config not runnable",
                    reason=str(exc),
                )
            )
            print(f"skip {path.name}: {exc}", flush=True)
            continue
        prepared = _frozen_spec(
            cfg,
            path,
            arm_defs,
            attention=cfg.attention_implementation,
            cases=getattr(args, "case", None),
            results=results,
            specs=specs,
        )
        if prepared is None:
            continue
        spec, selected_cases = prepared
        selected = {case.name for case in selected_cases}
        # The spec's cases rather than freshly derived ones: the case checksum above proves the two agree,
        # and the spec carries the token counts and padding fraction each verdict prints.
        arms = [arm for arm in spec.arms if arm.name in selected]

        pool = _hosted_placement_pool()
        widths = [width for width in placement_widths(cfg.n_gpus, pool) if width != cfg.n_gpus]
        if not widths:
            # The declared width is the measurement when nothing wider fits. Skipping it would leave a
            # hosted-only check with no placement on a node the config was written for.
            widths = [cfg.n_gpus]
        print(f"=== {cfg.config_id} on the hosted server, pool {pool} ===", flush=True)
        with console.activity(f"Reading the vocabulary of {cfg.model_name}"):
            vocab_size = _catalog_vocab_size(cfg.model_name)
        for width in widths:
            multiple = width // cfg.n_gpus
            run_cfg = cfg.at_gpu_width(width)
            run_arms = scaled_for_placement(arms, multiple)
            print(f"--- {cfg.config_id} on {width} GPU(s) ---", flush=True)
            context = RunContext(
                config_id=cfg.config_id,
                cfg=run_cfg,
                spec=spec,
                arms=run_arms,
                attn_implementation=run_cfg.attention_implementation,
                # A hosted run materializes no batch files. Nothing on this node reads a batch: the request
                # bytes are built here and posted to the server.
                workdir=correctness_workdir("dss-correctness-hosted-"),
                batch_paths={},
                vocab_size=vocab_size,
                transport_factory=lambda _workdir, run_cfg=run_cfg: HostedTransport(connection, run_cfg),
            )
            by_name = {arm.name: arm for arm in run_arms}
            for test in chosen:
                try:
                    produced = list(test.fn(context))
                except Exception as exc:  # noqa: BLE001 - one failing check must not hide the rest
                    produced = [
                        TestResult(
                            test_id=test.test_id,
                            config_id=cfg.config_id,
                            outcome=TestOutcome.FAIL,
                            summary=f"{type(exc).__name__}: {exc}",
                        )
                    ]
                for result in produced:
                    if result.gpus is None:
                        result.gpus = width
                    print(console.arm_verdict(result, by_name.get(result.arm)), flush=True)
                results.extend(produced)

    return _finish(results, specs, Path(args.out), configs=len(config_paths), started=started)


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
    return pytest.main(["-q", "--instafail", "-o", "python_files=check_*.py", str(selftest_dir), *args.runner_args])


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

    r = sub.add_parser("run", help="run tests over the config-by-test matrix")
    shared(r)
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
    r.set_defaults(func=cmd_run)

    list_parser = sub.add_parser("list", help="show registered tests")
    list_parser.set_defaults(func=cmd_list)

    h = sub.add_parser(
        "hosted",
        help="run Arctic Platform-against-Arctic Platform checks on the hosted server, which supplies the GPUs",
    )
    h.add_argument("--config", help="path to one Arctic Platform job config")
    h.add_argument("--all-configs", action="store_true", help="every config in --config-dir")
    h.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="root of the model/gpu-type/workload config tree, searched recursively",
    )
    h.add_argument(
        "--test",
        action="append",
        required=True,
        help="check id to run (repeatable); runs whether or not the spec lists it as applicable",
    )
    h.add_argument(
        "--case", action="append", help="case name to run, e.g. gas1 (repeatable); default every case in the spec"
    )
    h.add_argument("--host", required=True, help="account host serving the Arctic Platform API")
    h.add_argument("--database", required=True, help="database the API path is scoped to")
    h.add_argument("--schema", required=True, help="schema the API path is scoped to")
    h.add_argument("--endpoint", help="API endpoint name; default is the client's own")
    h.add_argument(
        "--pat-file",
        required=True,
        help="file holding the programmatic access token; read at startup and never written out",
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


def _restart_with_ray_token_auth() -> None:
    """Start the CLI with Ray token mode set before any dependency imports Ray."""
    if os.environ.get("RAY_AUTH_MODE") == "token":
        return
    os.environ["RAY_AUTH_MODE"] = "token"
    os.execv(sys.executable, [sys.executable, "-m", "arctic_platform.correctness", *sys.argv[1:]])


if __name__ == "__main__":
    _restart_with_ray_token_auth()
    sys.exit(main())
