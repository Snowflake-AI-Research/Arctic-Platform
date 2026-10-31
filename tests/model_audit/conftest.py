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
"""Model audit: model-agnostic correctness checks for every stage of RL training, parametrized by model profile.

Opt-in (the root conftest skips this tree unless it is named)::

    pytest -p no:xdist -sv tests/model_audit --audit-model glm-5.3-flash
    AP_AUDIT_MODELS=glm-5.3-flash,qwen3-0.6b pytest -p no:xdist tests/model_audit -m "not vllm"

Every test carries ``@pytest.mark.audit_check("<id>")``. A profile's ``known_issues`` turns that check into a strict
xfail linked to its issue (so a fix that lands shows up as XPASS and fails until the entry is removed), and
``not_applicable`` skips it with the profile's reason. Probes run once per session and are cached on disk under
``$AP_MODEL_AUDIT_DIR`` keyed by the probe code, the tiny checkpoint and the arguments. Each check's outcome is
appended to ``$AP_MODEL_AUDIT_DIR/results/<session>.jsonl`` for the audit report. ``$AP_AUDIT_SAMPLER_PYTHON``
runs the sampler probe under another interpreter when the test env's vLLM is not the pinned one.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_lib  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS: list[dict] = []
_SESSION = time.strftime("%Y%m%d-%H%M%S")


def pytest_addoption(parser):
    group = parser.getgroup("model audit")
    group.addoption("--audit-model", action="append", default=[], help="profile name or path; repeatable")
    group.addoption("--audit-refresh", action="store_true", help="ignore cached probe results")


def pytest_configure(config):
    config.addinivalue_line("markers", "audit_check(id): the model-audit check this test implements")
    config.addinivalue_line("markers", "audit_gpus(n): GPUs the check's probes need")


def _selected_profiles(config) -> list[audit_lib.Profile]:
    names = list(config.getoption("--audit-model"))
    names += [n for n in os.environ.get("AP_AUDIT_MODELS", "").split(",") if n]
    if not names:
        return audit_lib.all_profiles()
    return [audit_lib.load_profile(n) for n in names]


def pytest_generate_tests(metafunc):
    if "profile" in metafunc.fixturenames:
        profiles = _selected_profiles(metafunc.config)
        metafunc.parametrize("profile", profiles, ids=[p.name for p in profiles], scope="session")


def _check_id(item) -> str | None:
    marker = item.get_closest_marker("audit_check")
    return marker.args[0] if marker else None


def _variant(item, profile) -> str | None:
    """The parametrization of ``item`` other than the profile, e.g. ``ep1-sp2`` or ``hf``."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    return callspec.id.replace(profile.name, "").strip("-").replace("--", "-") or None


def _profile_entry(getter, check: str, variant: str | None):
    """A profile entry for ``check[variant]`` wins over one for the whole check."""
    return (getter(f"{check}[{variant}]") if variant else None) or getter(check)


def pytest_collection_modifyitems(config, items):
    for item in items:
        check = _check_id(item)
        profile = getattr(getattr(item, "callspec", None), "params", {}).get("profile")
        if check is None or profile is None:
            continue
        # Session fixtures launch the probes inside the first test's setup; each probe bounds its own runtime.
        item.add_marker(pytest.mark.timeout(0))
        variant = _variant(item, profile)
        reason = _profile_entry(profile.not_applicable, check, variant)
        if reason:
            item.add_marker(pytest.mark.skip(reason=f"not applicable to {profile.name}: {reason}"))
            continue
        issue = _profile_entry(profile.known_issue, check, variant)
        if issue:
            item.add_marker(pytest.mark.xfail(strict=True, reason=f"{issue.get('issue')} {issue.get('summary', '')}"))
        gpus = item.get_closest_marker("audit_gpus")
        if gpus and torch.cuda.device_count() < gpus.args[0]:
            item.add_marker(pytest.mark.skip(reason=f"needs {gpus.args[0]} GPU(s); have {torch.cuda.device_count()}"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    check = _check_id(item)
    if check is None or report.when == "teardown" or (report.when == "setup" and report.passed):
        return
    profile = getattr(getattr(item, "callspec", None), "params", {}).get("profile")
    status = report.outcome
    if hasattr(report, "wasxfail"):
        status = "xfail" if report.skipped else "xpass"
    elif report.when == "setup" and report.failed:
        status = "error"
    _RESULTS.append(
        {
            "session": _SESSION,
            "model": profile.name if profile else None,
            "check": check,
            "claim": audit_lib.CHECK_IDS.get(check, ""),
            "test": item.nodeid,
            "phase": report.when,
            "outcome": status,
            "seconds": round(report.duration, 2),
            "issue": (
                (_profile_entry(profile.known_issue, check, _variant(item, profile)) or {}).get("issue")
                if profile
                else None
            ),
            "message": str(report.longrepr)[-4000:] if report.failed or report.skipped else "",
            "details": getattr(item, "_audit_details", None),
        }
    )


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    out = audit_lib.audit_root() / "results" / f"{_SESSION}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        for row in _RESULTS:
            handle.write(json.dumps(row, default=str) + "\n")
    print(f"\nmodel audit results: {out}")


@pytest.fixture
def record(request):
    """Attach measured numbers to the check's result row, so the report shows evidence for passes too."""

    def _record(**details):
        current = getattr(request.node, "_audit_details", None) or {}
        current.update(details)
        request.node._audit_details = current

    return _record


# ---------------------------------------------------------------------------------------------------------------------
# Tiny checkpoints and probes
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_ckpt(profile) -> Path:
    return audit_lib.build_tiny_checkpoint(profile, seed=0)


@pytest.fixture(scope="session")
def tiny_ckpt_b(profile) -> Path:
    """A second tiny checkpoint of the same config with different weights: the "trained" weights to sync."""
    return audit_lib.build_tiny_checkpoint(profile, seed=1)


@functools.cache
def _code_hash() -> str:
    digest = hashlib.sha256()
    for name in ("audit_lib.py", "trainer_probe.py", "sampler_probe.py"):
        digest.update((audit_lib.AUDIT_DIR / name).read_bytes())
    # The code under audit: a platform or library change must not replay a stale probe.
    for path in sorted((_REPO_ROOT / "arctic_platform").rglob("*.py")):
        stat = path.stat()
        digest.update(f"{path.relative_to(_REPO_ROOT)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for package in ("torch", "transformers", "deepspeed", "vllm", "flashinfer-python", "liger-kernel"):
        try:
            digest.update(f"{package}={importlib.metadata.version(package)}".encode())
        except importlib.metadata.PackageNotFoundError:
            pass
    return digest.hexdigest()[:12]


# Profile fields that only gate checks; editing them must not invalidate cached probe outputs.
_GATING_KEYS = frozenset({"expect", "known_issues", "not_applicable", "tolerances"})


class ProbeRunner:
    """Runs probe subprocesses on the first N visible GPUs and caches their outputs for the session."""

    def __init__(self, refresh: bool):
        self.refresh = refresh
        self.memo: dict[str, dict] = {}

    def _visible(self) -> list[str]:
        preset = os.environ.get("CUDA_VISIBLE_DEVICES")
        if preset:
            return [g for g in preset.split(",") if g]
        return [str(i) for i in range(torch.cuda.device_count())]

    def _launch(self, kind, profile, ckpt, gpus, timeout, args, out, log) -> None:
        out.unlink(missing_ok=True)
        visible = self._visible()
        if len(visible) < gpus:
            pytest.skip(f"probe needs {gpus} GPU(s); {len(visible)} visible")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(visible[:gpus])}
        env.pop("PYTEST_XDIST_WORKER", None)
        # Another interpreter may have a different arctic_platform checkout installed.
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
        script = audit_lib.AUDIT_DIR / f"{kind}_probe.py"
        cli = ["--profile", str(profile.path), "--ckpt", str(ckpt), "--out", str(out)]
        for name, value in args.items():
            values = value if isinstance(value, list) else [value]
            for v in values:
                cli += [f"--{name.replace('_', '-')}", v if isinstance(v, str) else json.dumps(v)]
        if kind == "trainer":
            from arctic_platform.testing_utils import get_unique_port_number
            from arctic_platform.testing_utils import reserve_free_port

            port = reserve_free_port(get_unique_port_number() + 1, span=7)
            cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={gpus}"]
            cmd += [f"--master_port={port}", str(script), *cli]
        else:
            cmd = [os.environ.get("AP_AUDIT_SAMPLER_PYTHON", sys.executable), str(script), *cli]
        with log.open("w") as handle:
            handle.write(" ".join(cmd) + "\n")
            handle.flush()
            proc = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        if not out.exists():
            pytest.fail(f"{kind} probe exited {proc.returncode} without output; log: {log}")

    def run(self, kind: str, profile, ckpt: Path, *, gpus: int = 1, timeout: int = 3600, **args) -> dict:
        inputs = {k: v for k, v in profile.raw.items() if k not in _GATING_KEYS}
        key_src = json.dumps([kind, inputs, str(ckpt), gpus, args, _code_hash()], sort_keys=True, default=str)
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        if key in self.memo:
            return self.memo[key]
        run_dir = audit_lib.audit_root() / "runs" / profile.name
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / f"{kind}-{args.get('mode', '')}-{key}.pt"
        log = out.with_suffix(".log")
        # A failed probe (e.g. a transient OOM) is not reused from disk.
        result = None if self.refresh or not out.exists() else torch.load(out, weights_only=False)
        if result is None or not result.get("status", {}).get("ok"):
            self._launch(kind, profile, ckpt, gpus, timeout, args, out, log)
            result = torch.load(out, weights_only=False)
        result["_log"] = str(log)
        self.memo[key] = result
        return result


@pytest.fixture(scope="session")
def probes(request) -> ProbeRunner:
    return ProbeRunner(request.config.getoption("--audit-refresh"))


def require_ok(result: dict, what: str) -> dict:
    status = result.get("status", {})
    if not status.get("ok"):
        pytest.fail(f"{what} failed before any comparison: {status.get('error')}\nlog: {result.get('_log')}")
    return result


@pytest.fixture(scope="session")
def reference(probes, profile, tiny_ckpt):
    return probes.run("trainer", profile, tiny_ckpt, mode="reference")


@pytest.fixture(scope="session")
def trainer_forward(probes, profile, tiny_ckpt):
    return probes.run("trainer", profile, tiny_ckpt, gpus=profile.baseline_gpus, mode="forward")


@pytest.fixture(scope="session")
def baseline_step(probes, profile, tiny_ckpt):
    """One production optimizer step at the audit lr on the baseline layout, with the engine-reduced gradients."""
    return probes.run(
        "trainer", profile, tiny_ckpt, gpus=profile.baseline_gpus, mode="step", parallelism=profile.tiny_parallelism()
    )


@pytest.fixture(scope="session")
def sampler_run(probes, profile, tiny_ckpt):
    return probes.run("sampler", profile, tiny_ckpt, gpus=profile.sampler_gpus)


@pytest.fixture(scope="session")
def export_b(probes, profile, tiny_ckpt_b):
    """Checkpoint B's production weight streams, in every format the production model exports."""
    export_dir = audit_lib.audit_root() / "exports" / f"{profile.name}-{tiny_ckpt_b.name}"
    export_dir.mkdir(parents=True, exist_ok=True)
    return probes.run(
        "trainer", profile, tiny_ckpt_b, gpus=profile.baseline_gpus, mode="export", export_dir=str(export_dir)
    )


@pytest.fixture(scope="session")
def fresh_b_ckpt(tiny_ckpt, tiny_ckpt_b, trainer_forward) -> Path:
    """B, except that tensors the production model's full HF export lacks keep A's values.

    The trainer holds no state for those, so training never changes them and a sampler synced from A keeps A's.
    ``trainer.export_round_trip`` reports them.
    """
    held = set(require_ok(trainer_forward, "production forward")["exported"])
    untrained = sorted(set(audit_lib.read_checkpoint_tensors(tiny_ckpt_b)) - held)
    return audit_lib.with_tensors_from(tiny_ckpt_b, tiny_ckpt, untrained)


@pytest.fixture(scope="session")
def sampler_fresh_b(probes, profile, fresh_b_ckpt):
    return probes.run("sampler", profile, fresh_b_ckpt, gpus=profile.sampler_gpus)


@pytest.fixture(scope="session")
def sampler_fresh_b_replica(probes, profile, fresh_b_ckpt):
    """A second engine started on the same weights: a synced sampler is another engine, so it differs by this much."""
    return probes.run("sampler", profile, fresh_b_ckpt, gpus=profile.sampler_gpus, replica=1)
