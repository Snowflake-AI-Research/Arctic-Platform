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

    pytest -p no:xdist -sv tests/model_audit --audit-model qwen3-0.6b
    AP_AUDIT_MODELS=qwen3-0.6b pytest -p no:xdist tests/model_audit -m "not vllm"

The code under audit is a Cortex Training deployment, reached only through the ``cortex_training`` client: each
probe of the trainer, optimizer, sampler or weight sync is a job on the service, started from the model-catalog
configs the profile names (``AP_AUDIT_CATALOG`` is the ``model-catalog/models`` directory). The truth is a local
fp32 reference on the same checkpoint, one GPU of this machine; checks whose reference does not fit skip. Checks
that need state the client does not expose skip as coverage gaps.

Every test carries ``@pytest.mark.audit_check("<id>")``. A profile's ``known_issues`` turns that check into a strict
xfail linked to its issue (so a fix that lands shows up as XPASS and fails until the entry is removed), and
``not_applicable`` skips it with the profile's reason. Probes run once per session and are cached on disk under
``$AP_MODEL_AUDIT_DIR`` keyed by the probe code, the job body and the arguments; ``--audit-refresh`` reruns them
(the service changes under a cached result). Each check's outcome is appended to
``$AP_MODEL_AUDIT_DIR/results/<session>.jsonl`` for the audit report.
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
            # Only a failed comparison is the known issue; a probe that fails first (pytest.fail) still fails.
            reason = f"{issue.get('issue')} {issue.get('summary', '')}"
            item.add_marker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))
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
# Probes: the local reference and the service jobs
# ---------------------------------------------------------------------------------------------------------------------


_PROBE_CODE = {
    "reference": (("audit_lib.py", "reference_probe.py", "deepseek_inference.py"), ("torch", "transformers"), ()),
    "client": (
        ("audit_lib.py", "client_probe.py"),
        ("torch", "cortex-training"),
        ("AP_AUDIT_CLIENT_CONNECTION", "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_DATABASE"),
    ),
}


@functools.cache
def _code_hash(kind: str) -> str:
    """The probe's own code, its libraries and, for the service, which deployment it talks to."""
    files, packages, env_vars = _PROBE_CODE[kind]
    digest = hashlib.sha256()
    for name in files:
        digest.update((audit_lib.AUDIT_DIR / name).read_bytes())
    for package in packages:
        try:
            digest.update(f"{package}={importlib.metadata.version(package)}".encode())
        except importlib.metadata.PackageNotFoundError:
            pass
    for var in env_vars:
        digest.update(f"{var}={os.environ.get(var, '')}".encode())
    return digest.hexdigest()[:12]


def _probe_inputs(kind: str, profile) -> list:
    """The profile fields a probe of ``kind`` reads besides its arguments."""
    inputs = [profile.probe, profile.reference]
    if kind == "reference":
        try:
            inputs.append(profile.launch_training())
        except (LookupError, FileNotFoundError):
            inputs.append(None)
    return inputs


def client_unavailable() -> str | None:
    """Why no service job can run from here, or None."""
    try:
        import cortex_training  # noqa: F401
    except ImportError:
        return "cortex_training is not importable (install cortex-training or put its src on PYTHONPATH)"
    if not os.environ.get("AP_AUDIT_CATALOG"):
        return "AP_AUDIT_CATALOG (the model-catalog/models directory) is not set"
    if os.environ.get("AP_AUDIT_CLIENT_CONNECTION"):
        return None
    missing = [
        v for v in ("CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT", "CORTEX_TRAINING_DATABASE") if not os.environ.get(v)
    ]
    return f"no service connection: set AP_AUDIT_CLIENT_CONNECTION or {', '.join(missing)}" if missing else None


class ProbeRunner:
    """Runs probe subprocesses, several at once when started ahead, and caches their outputs for the session."""

    def __init__(self, refresh: bool):
        self.refresh = refresh
        self.memo: dict[str, dict] = {}
        self.pending: dict[str, tuple[subprocess.Popen, Path, Path, int]] = {}

    def _paths(self, kind: str, profile, ckpt: Path, name: str, args: dict) -> tuple[str, Path, Path]:
        inputs = _probe_inputs(kind, profile)
        key_src = json.dumps([kind, inputs, str(ckpt), args, _code_hash(kind)], sort_keys=True, default=str)
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        run_dir = audit_lib.audit_root() / "runs" / profile.name
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / f"{kind}-{name}-{key}.pt"
        return key, out, out.with_suffix(".log")

    def start(self, kind: str, profile, ckpt: Path, *, name: str, timeout: int, **args) -> str:
        """Launch the probe unless its output is cached or it is already running; returns its key."""
        key, out, log = self._paths(kind, profile, ckpt, name, args)
        if key in self.memo or key in self.pending:
            return key
        # A failed probe (e.g. a transient service error) is not reused from disk.
        if not self.refresh and out.exists() and torch.load(out, weights_only=False).get("status", {}).get("ok"):
            return key
        out.unlink(missing_ok=True)
        env = dict(os.environ)
        env.pop("PYTEST_XDIST_WORKER", None)
        cli = ["--profile", str(profile.path), "--ckpt", str(ckpt), "--out", str(out)]
        for arg, value in args.items():
            for v in value if isinstance(value, list) else [value]:
                cli += [f"--{arg.replace('_', '-')}", v if isinstance(v, str) else json.dumps(v)]
        if kind == "reference":
            visible = [g for g in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if g] or ["0"]
            env["CUDA_VISIBLE_DEVICES"] = visible[0]
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [sys.executable, str(audit_lib.AUDIT_DIR / f"{kind}_probe.py"), *cli]
        handle = log.open("w")
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        self.pending[key] = (proc, out, log, timeout)
        return key

    def run(self, kind: str, profile, ckpt: Path, *, name: str, timeout: int = 3600, **args) -> dict:
        key = self.start(kind, profile, ckpt, name=name, timeout=timeout, **args)
        if key in self.memo:
            return self.memo[key]
        _, out, log = self._paths(kind, profile, ckpt, name, args)
        if key in self.pending:
            proc, out, log, timeout = self.pending.pop(key)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail(f"{kind} probe {name} exceeded {timeout}s; log: {log}")
            if not out.exists():
                pytest.fail(f"{kind} probe {name} exited {proc.returncode} without output; log: {log}")
        result = torch.load(out, weights_only=False)
        result["_log"] = str(log)
        self.memo[key] = result
        return result

    def reference(self, profile, ckpt: Path) -> dict:
        reason = reference_unavailable(profile, ckpt)
        if reason:
            pytest.skip(reason)
        return self.run("reference", profile, ckpt, name=Path(ckpt).name)


def reference_unavailable(profile, ckpt: Path) -> str | None:
    if not torch.cuda.is_available():
        return "the local reference needs a GPU"
    # fp32 weights, gradients and two gradient accumulators, from a checkpoint stored in the launch dtype.
    params = audit_lib.checkpoint_bytes(ckpt) / torch.finfo(getattr(torch, profile.dtype)).bits * 8
    need, have = params * 16, torch.cuda.get_device_properties(0).total_memory
    if need > 0.8 * have:
        return f"the fp32 reference needs ~{need / 2**30:.0f} GiB, more than one local GPU has"
    return None


@pytest.fixture(scope="session")
def probes(request) -> ProbeRunner:
    return ProbeRunner(request.config.getoption("--audit-refresh"))


def require_ok(result: dict, what: str) -> dict:
    status = result.get("status", {})
    if not status.get("ok"):
        pytest.fail(f"{what} failed before any comparison: {status.get('error')}\nlog: {result.get('_log')}")
    return result


@pytest.fixture(scope="session")
def model_ckpt(profile) -> Path:
    return audit_lib.local_checkpoint(profile)


@pytest.fixture(scope="session")
def reference(probes, profile, model_ckpt):
    return probes.reference(profile, model_ckpt)


_CLIENT_TIMEOUT = 4 * 3600
ISOLATION_GAP = (
    "coverage gap: a data-parallel job rejects requests with fewer rows than ranks, so rows never run alone"
)
FORCED_CLIP = 1e-3


def client_job(profile, name: str) -> dict:
    """Body, step lr and sync formats of the service job ``name``."""
    lr = profile.audit_lr
    if name == "baseline":
        return {"body": profile.job_body(training={}, sampling={}), "lr": lr, "sync_format": profile.sync_formats}
    if name == "launch-lr":
        return {"body": profile.job_body(training={}), "lr": profile.optimizer()["lr"], "forwards": "batch"}
    if name == "forced-clip":
        overrides = {"training_config": {"gradient_clipping": FORCED_CLIP}}
        return {"body": profile.job_body(training=overrides), "lr": lr, "forwards": "batch"}
    if name.startswith("layout-"):
        layout = next(lay for lay in profile.layouts if lay["id"] == name[len("layout-") :])
        overrides = {"training_config": layout["training_config"]}
        return {"body": profile.job_body(training=overrides), "lr": lr, "forwards": "batch"}
    if name.startswith("sampler-tp"):
        tp = int(name[len("sampler-tp") :])
        overrides = {"inference_config": {"n_gpus": tp, "vllm_config": {"tensor_parallel_size": tp}}}
        return {"body": profile.job_body(sampling=overrides)}
    raise KeyError(name)


def launch_tp(profile) -> int:
    return int(
        profile.launch_sampling().get("inference_config", {}).get("vllm_config", {}).get("tensor_parallel_size", 1)
    )


def _wanted_jobs(profile, items) -> list[str]:
    """The service jobs the selected, runnable tests of ``profile`` read."""
    wanted = []
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        if getattr(params.get("profile"), "name", None) != profile.name or item.get_closest_marker("skip"):
            continue
        fixtures = set(item.fixturenames)
        names = []
        if fixtures & {"client_baseline", "trainer_forward", "baseline_step", "sampler_run"}:
            names.append("baseline")
        if "step_launch_lr" in fixtures:
            names.append("launch-lr")
        if "parallel_step" in fixtures and any(lay["id"] == params.get("layout") for lay in profile.layouts):
            names.append(f"layout-{params['layout']}")
        if "tp" in params and params["tp"] < launch_tp(profile):
            names.append(f"sampler-tp{params['tp']}")
        wanted += [n for n in names if n not in wanted]
    return wanted


class ClientJobs:
    def __init__(self, probes: ProbeRunner, profile, ckpt: Path):
        self.probes, self.profile, self.ckpt = probes, profile, ckpt
        self.steps: dict[str, dict] = {}

    def _spec(self, name: str) -> dict:
        reason = client_unavailable()
        if reason:
            pytest.skip(reason)
        try:
            return client_job(self.profile, name)
        except (LookupError, FileNotFoundError) as err:
            pytest.skip(f"no service job for {self.profile.name}: {err}")

    def start(self, name: str) -> None:
        self.probes.start("client", self.profile, self.ckpt, name=name, timeout=_CLIENT_TIMEOUT, **self._spec(name))

    def get(self, name: str) -> dict:
        return self.probes.run(
            "client", self.profile, self.ckpt, name=name, timeout=_CLIENT_TIMEOUT, **self._spec(name)
        )

    def step(self, name: str) -> dict:
        """The optimizer step of job ``name`` as ``audit_lib.client_step`` reads it from the exports."""
        if name not in self.steps:
            run = self.get(name)
            self.steps[name] = run
            if run.get("status", {}).get("ok"):
                keep = reference_unavailable(self.profile, self.ckpt) is None
                self.steps[name] = {
                    **audit_lib.client_step(self.profile, run, keep_tensors=keep),
                    "status": run["status"],
                    "_log": run["_log"],
                }
        return self.steps[name]


@pytest.fixture(scope="session")
def client(request, probes, profile, model_ckpt) -> ClientJobs:
    jobs = ClientJobs(probes, profile, model_ckpt)
    if client_unavailable() is None:
        for name in _wanted_jobs(profile, request.session.items):
            try:
                jobs.start(name)
            except pytest.skip.Exception:
                pass
    return jobs


@pytest.fixture(scope="session")
def client_baseline(client):
    return client.get("baseline")


def _checkpoint_buffers(resumable: Path) -> set[str]:
    """The buffers a DeepSpeed checkpoint records: exported with the weights, but never trained."""
    states = next(resumable.glob("*mp_rank_00_model_states.pt"))
    return set(torch.load(states, map_location="cpu", weights_only=False, mmap=True).get("buffer_names", []))


@pytest.fixture(scope="session")
def trainer_forward(client_baseline):
    run = client_baseline
    if not run.get("status", {}).get("ok"):
        return run
    norms = run["step"]["reply"].get("gradient_norms_per_param") or {}
    exported = audit_lib.read_checkpoint_tensors(Path(run["exports"]["initial"]))
    buffers = _checkpoint_buffers(Path(run["exports"]["resumable"]))
    return {
        **run["forward"],
        "loader": run["job"]["training"]["training_config"].get("model_provider"),
        "spec": run["job"]["training"],
        "exported": exported,
        "buffers": sorted(buffers),
        "no_grad_params": sorted(n for n in exported if n not in buffers and not norms.get(n)),
        "nonfinite_grad_params": sorted(
            n for n, v in norms.items() if v is None or not torch.isfinite(torch.tensor(v))
        ),
        "status": run["status"],
        "_log": run["_log"],
    }


@pytest.fixture(scope="session")
def baseline_step(client):
    return client.step("baseline")


@pytest.fixture(scope="session")
def step_launch_lr(client):
    return client.step("launch-lr")


@pytest.fixture(scope="session")
def forced_clip_step(client):
    return client.step("forced-clip")


@pytest.fixture(scope="session")
def sampler_run(client_baseline):
    run = client_baseline
    if not run.get("status", {}).get("ok"):
        return run
    return {**run["sampler"], "exports": run.get("exports", {}), "status": run["status"], "_log": run["_log"]}
