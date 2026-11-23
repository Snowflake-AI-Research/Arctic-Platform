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

"""Publish semi_p images for a list of job configs on a cluster, then optionally test their restores.

    python -m arctic_platform.inference.semi_persistence.onboarding.semip_job \\
        cluster=qa6|prod image_tag=TAG configs=URL max_gpus=N \\
        [tests=functional,correctness,performance] [dry_run=false] [out_dir=semip_job] \\
        [connection=CONN.json] [kube_context=CTX] [env_hash=ENV12]

The driver behind the semip-publish Jenkins job; it runs the same anywhere.

  cluster      qa6 or prod: the kubeconfig context the node cache is checked on,
               the model bucket images are published to, and which connection
               file is used by default
  image_tag    the dss-backend image (release_<date>_<time>_<11 hex>-h|a|b, or on
               qa6 also dev_<date>_<time>_<11 hex>[-h|a|b]): publish's image_tag
               and onboard's debug_image_tag
  configs      a github.com blob link to a .jsonl under model-catalog/semip_models/
               in an <owner>/cortex-training repo, at a commit or a branch
               (resolved to its commit): one semi_p job body per line, exactly
               as a user submits it (see jobs.load_models)
  tests        subset of functional,correctness,performance (default none:
               publish only); correctness and performance imply functional
  max_gpus     GPU budget for publish and for onboard (reserve_gpus=0 for both)
  dry_run      validate, fetch and parse the configs, write and print the plan,
               and exit without submitting anything (default false)
  out_dir      where everything is written (default semip_job)
  connection   Cortex Training connection file (default $SEMIP_CONNECTION_QA6 /
               $SEMIP_CONNECTION_PROD for the cluster, else $SEMIP_CONNECTION)
  kube_context default: the cluster's
  env_hash     dry run only: the env12 that image_tag's skeleton keys carry, to
               tell a skeleton of this image from one of another

``configs`` is fetched from raw.githubusercontent.com at the resolved commit
(GitHub's API resolves a branch; ``GITHUB_TOKEN``, if set, authenticates both)
and recorded in ``<out_dir>/configs.json`` with the file in ``configs.jsonl``.
Each line's job is named by its label (``qwen3-0.6b-32k-full``: model,
max_seq_len, full or lora) in the plan, the reports and publish's summary.

Publish always runs first, on every config. The tests then run onboard on the
configs that published and whose test jobs fit within ``max_gpus``:

  functional              variants=on_hit checks=none: the restore from the
                          verified image must succeed; start-up is reported
  correctness             variants=on_hit,off_default checks=correctness
  performance             variants=on_hit,off_default checks=perf
  correctness,performance variants=on_hit,off_default checks=correctness,perf

with require_restore=true and onboard waiting for the node cache on the
cluster's context. ``<out_dir>/plan.json`` / ``plan.txt`` hold the jobs and
the exact commands, ``result.json`` / ``result.md`` each config's publish
status and test verdicts. Exits non-zero if a config did not publish, a
requested test failed, or a config's tests could not run (they need more than
``max_gpus``). SIGINT, SIGTERM and SIGHUP are passed on to publish or onboard,
which cancel their jobs.

A dry run also computes each job's cfg12 offline (``keys``) and, when kubectl
reaches the cluster's context, reads the node labels once to report which of
those skeletons the model-cache nodes have verified: per env12, on how many of
them. ``existing`` is ``none`` when publish would dump; with ``env_hash`` a
skeleton is marked as this image's or another image's, and without it the env
is unconfirmed. Without kubectl the lookup is skipped with a note.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any, Callable

from arctic_platform.inference.semi_persistence.onboarding import keys
from arctic_platform.inference.semi_persistence.onboarding import report
from arctic_platform.inference.semi_persistence.onboarding.client import load_connection
from arctic_platform.inference.semi_persistence.onboarding.common import as_bool
from arctic_platform.inference.semi_persistence.onboarding.common import parse_kv_args
from arctic_platform.inference.semi_persistence.onboarding.jobs import GPUS_PER_NODE
from arctic_platform.inference.semi_persistence.onboarding.jobs import ModelSpec
from arctic_platform.inference.semi_persistence.onboarding.jobs import load_models
from arctic_platform.inference.semi_persistence.onboarding.onboard import footprint

logger = logging.getLogger(__name__)

PACKAGE = "arctic_platform.inference.semi_persistence.onboarding"
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Cluster:
    kube_context: str
    model_bucket: str


CLUSTERS = {
    "qa6": Cluster("sfc-or-dev-meta-k8s-2", "neutrino-sfc-or-dev-meta-k8s-2"),
    "prod": Cluster("sfc-or-prod-meta-k8s-2", "neutrino-sfc-or-prod-meta-k8s-2"),
}
CONNECTION_ENV = "SEMIP_CONNECTION"
IMAGE_TAG = re.compile(r"^release_\d{8}_\d{6}_[0-9a-f]{11}-[hab]$")
# dev_* images are pushed to the dev registries only, so prod cannot pull them.
DEV_IMAGE_TAG = re.compile(r"^dev_\d{8}_\d{6}_[0-9a-f]{11}(-[hab])?$")
TESTS = ("functional", "correctness", "performance")
CONFIGS_REPO = "cortex-training"
CONFIGS_DIR = "model-catalog/semip_models/"
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
ENV_HASH = re.compile(r"^[0-9a-f]{12}$")
PUBLISHED = ("published", "already")
# An env12 shown only this far is taken as env_hash's when no weight hash can confirm it.
ENV_PREFIX_CHARS = 6


@dataclass(frozen=True)
class ConfigsRef:
    """A configs file on GitHub: ``<owner>/cortex-training`` at ``ref``, ``path`` in it."""

    url: str
    owner: str
    ref: str
    path: str

    @property
    def repo(self) -> str:
        return f"{self.owner}/{CONFIGS_REPO}"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def parse_cluster(value: str | None) -> str:
    if value not in CLUSTERS:
        raise SystemExit(f"cluster= must be one of {', '.join(CLUSTERS)}, got {value!r}")
    return value


def parse_image_tag(value: str | None, cluster: str = "prod") -> str:
    if IMAGE_TAG.match(value or ""):
        return value
    if DEV_IMAGE_TAG.match(value or ""):
        if cluster == "qa6":
            return value
        raise SystemExit(f"image_tag={value} is a dev_* image, which only qa6 can pull; use a release_* tag on {cluster}")
    raise SystemExit("image_tag= must look like release_20261119_163502_6ed8c6d2361-h "
                     f"(or dev_20261115_234921_a2c56b581ba on qa6), got {value!r}")


def parse_max_gpus(value: str | None) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", (value or "").strip()):
        raise SystemExit(f"max_gpus= must be a positive integer, got {value!r}")
    return int(value)


def parse_env_hash(value: str | None) -> str | None:
    if not value:
        return None
    if not ENV_HASH.match(value):
        raise SystemExit(f"env_hash= must be the 12 lowercase hex digits of a skeleton key's env12, got {value!r}")
    return value


def parse_tests(value: str | None) -> list[str]:
    """The tests ``tests=`` asks for, in order, functional added when implied."""
    wanted = {t.strip() for t in (value or "").split(",") if t.strip()}
    unknown = wanted - set(TESTS)
    if unknown:
        raise SystemExit(f"tests= takes a subset of {','.join(TESTS)}, got {', '.join(sorted(unknown))}")
    if wanted:
        wanted.add("functional")
    return [t for t in TESTS if t in wanted]


def onboard_options(tests: list[str]) -> dict[str, str] | None:
    """onboard's ``variants`` and ``checks`` for the tests, or None (publish only)."""
    if not tests:
        return None
    checks = [check for test, check in (("correctness", "correctness"), ("performance", "perf")) if test in tests]
    if not checks:
        return {"variants": "on_hit", "checks": "none"}
    return {"variants": "on_hit,off_default", "checks": ",".join(checks)}


def parse_configs_url(url: str | None) -> ConfigsRef:
    """``https://github.com/<owner>/cortex-training/blob/<ref>/model-catalog/semip_models/<file>.jsonl``;
    ``ref`` may hold slashes (a branch), so the link is split at the configs directory."""
    parsed = urllib.parse.urlsplit((url or "").strip())
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise SystemExit(f"configs= must be an https://github.com/ link, got {url!r}")
    if parsed.query:
        raise SystemExit(f"configs= must be a plain blob link, without a query: {url!r}")
    parts = urllib.parse.unquote(parsed.path).lstrip("/").split("/", 3)
    if len(parts) < 4 or parts[2] != "blob":
        raise SystemExit(f"configs= must be a blob link (github.com/<owner>/{CONFIGS_REPO}/blob/<ref>/...), got {url!r}")
    owner, repo, _, rest = parts
    if repo != CONFIGS_REPO:
        raise SystemExit(f"configs= must point into an <owner>/{CONFIGS_REPO} repository, got {owner}/{repo}")
    ref, sep, name = rest.partition(f"/{CONFIGS_DIR}")
    if not sep or not ref or not name:
        raise SystemExit(f"configs= must be a file under {CONFIGS_DIR}, got {url!r}")
    if not name.endswith(".jsonl") or any(p in ("", ".", "..") for p in name.split("/")):
        raise SystemExit(f"configs= must name a .jsonl file under {CONFIGS_DIR}, got {name!r}")
    return ConfigsRef(url=url.strip(), owner=owner, ref=ref, path=CONFIGS_DIR + name)


def http_get(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "arctic-platform-semip-job"}
    if accept:
        headers["Accept"] = accept
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise ValueError(f"GET {url}: HTTP {exc.code} {exc.reason}") from None
    except urllib.error.URLError as exc:
        raise ValueError(f"GET {url}: {exc.reason}") from None


def resolve_ref(configs: ConfigsRef, get: Callable = http_get) -> str:
    """The commit ``configs.ref`` names: itself if a full SHA, else GitHub's answer."""
    if COMMIT_SHA.match(configs.ref):
        return configs.ref
    url = f"https://api.github.com/repos/{configs.repo}/commits/{urllib.parse.quote(configs.ref, safe='')}"
    sha = get(url, "application/vnd.github.sha").decode().strip()
    if not COMMIT_SHA.match(sha):
        raise ValueError(f"GET {url}: expected a commit SHA, got {sha[:80]!r}")
    return sha


def raw_url(configs: ConfigsRef, sha: str) -> str:
    return f"https://raw.githubusercontent.com/{configs.repo}/{sha}/{urllib.parse.quote(configs.path)}"


def connection_path(args: dict[str, str], cluster: str) -> str | None:
    return (args.get("connection") or os.environ.get(f"{CONNECTION_ENV}_{cluster.upper()}")
            or os.environ.get(CONNECTION_ENV) or None)


def check_connection(path: str | None) -> str | None:
    """Why the connection file cannot be used, or None. Reads only its keys."""
    if not path:
        return f"no connection file: pass connection= or set {CONNECTION_ENV}"
    try:
        config = load_connection(path)
    except (OSError, ValueError) as exc:
        return f"connection file {path} unreadable ({type(exc).__name__})"
    missing = [k for k in ("host", "pat") if not config.get(k)]
    return f"connection file {path} lacks {', '.join(missing)}" if missing else None


def check_budget(specs: list[ModelSpec], max_gpus: int) -> None:
    """Refuse a budget a config's publish job cannot fit in (publish would wait for it forever)."""
    big = [f"{s.name} ({s.n_gpus})" for s in specs if s.n_gpus > max_gpus]
    if big:
        raise SystemExit(f"max_gpus={max_gpus} is below the n_gpus of {len(big)} config(s): {', '.join(big)}")


def fit_onboard(specs: list[ModelSpec], variants: list[str], max_gpus: int,
                gpus_per_node: int = GPUS_PER_NODE) -> tuple[list[ModelSpec], dict[str, str]]:
    """The configs whose test jobs fit the budget onboard derives from
    ``max_gpus``, and why each other one does not (onboard refuses the whole
    list if any does not fit)."""
    sizes = {s.name: footprint(s, variants, None, gpus_per_node) for s in specs}
    fits = [s for s in specs if sizes[s.name][0] <= max_gpus]
    while True:
        max_nodes = max_gpus // gpus_per_node + len(fits)
        kept = [s for s in fits if sizes[s.name][1] <= max_nodes]
        if len(kept) == len(fits):
            break
        fits = kept
    names = {s.name for s in fits}
    too_big = {s.name: f"needs {sizes[s.name][0]} GPUs on {sizes[s.name][1]} nodes; max_gpus={max_gpus}"
               for s in specs if s.name not in names}
    return fits, too_big


def line_number(spec: ModelSpec) -> int:
    """A config's line in its .jsonl (``load_models`` sets ``path`` to ``FILE:LINE``)."""
    return int(spec.path.rsplit(":", 1)[1])


def write_jsonl(specs: list[ModelSpec], path: Path, lines: dict[str, str]) -> Path:
    path.write_text("".join(lines[s.name] + "\n" for s in specs), encoding="utf-8")
    return path


class NodesUnavailable(Exception):
    """The node labels could not be read; the message says why."""


def cluster_nodes(context: str, run: Callable = subprocess.run) -> list[dict]:
    """Every node of kubeconfig context ``context``, from one ``kubectl get nodes``."""
    if not shutil.which("kubectl"):
        raise NodesUnavailable("kubectl is not on PATH")
    argv = ["kubectl", f"--context={context}", "--request-timeout=30s", "get", "nodes", "-o", "json"]
    try:
        proc = run(argv, capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NodesUnavailable(f"kubectl failed: {exc}") from None
    if proc.returncode:
        why = (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
        raise NodesUnavailable(f"kubectl --context={context} get nodes failed: {why[:200]}")
    try:
        return json.loads(proc.stdout)["items"]
    except (ValueError, KeyError, TypeError) as exc:
        raise NodesUnavailable(f"kubectl get nodes returned no node list ({exc!r})") from None


def env_text(found: keys.Found) -> str:
    if found.env_exact:
        return found.env[:10]
    return f"{found.env}.." if found.env else "env ?"


def match_of(found: keys.Found, env_hash: str | None, weights_known: bool) -> str:
    """Whether a skeleton is image ``env_hash``'s: ``exists``, ``exists for another image``,
    ``unconfirmed`` (its label shows too little of env12), or ``env unconfirmed`` (no env_hash)."""
    if not env_hash:
        return "env unconfirmed"
    if found.env_exact or not env_hash.startswith(found.env):
        return "exists" if found.env == env_hash else "exists for another image"
    if weights_known:
        # The label was checked against env_hash with every weight hash shown on the nodes.
        return "exists for another image"
    return "exists" if len(found.env) >= ENV_PREFIX_CHARS else "unconfirmed"


def skeleton_status(found: list[keys.Found], matches: list[str], nodes: int, env_hash: str | None) -> str:
    """``restore`` (this image's skeleton is on every node), ``partial``, ``dump`` or
    ``unknown``; without ``env_hash`` any env counts as this image's."""
    targets = [f for f, m in zip(found, matches) if m in ("exists", "env unconfirmed")]
    best = max((f.nodes for f in targets), default=0)
    if nodes and best == nodes:
        return "restore"
    if best:
        return "partial"
    return "unknown" if "unconfirmed" in matches else "dump"


def existing_text(found: list[keys.Found], nodes: int, env_hash: str | None, matches: list[str]) -> str:
    if not found:
        return "none"
    if not env_hash:
        parts = [f"{env_text(f)}: {f.nodes}/{nodes} nodes" for f in found]
        return ", ".join(parts) + (" (env unconfirmed)" if len(found) == 1 else "")
    mine = [f"{env_text(f)}: {f.nodes}/{nodes} nodes ({m})" for f, m in zip(found, matches) if m != "exists for another image"]
    other = [f"{env_text(f)} {f.nodes}/{nodes}" for f, m in zip(found, matches) if m == "exists for another image"]
    parts = [", ".join(mine)] if mine else []
    if other:
        parts.append(f"exists for another image: {', '.join(other)}")
    return "; ".join(parts)


class SemipJob:
    def __init__(self, args: dict[str, str], get: Callable = http_get, nodes: Callable | None = None):
        self.cluster = parse_cluster(args.get("cluster"))
        self.image_tag = parse_image_tag(args.get("image_tag"), self.cluster)
        self.max_gpus = parse_max_gpus(args.get("max_gpus"))
        self.tests = parse_tests(args.get("tests"))
        self.dry_run = as_bool(args.get("dry_run") or "false")
        self.configs = parse_configs_url(args.get("configs"))
        self.env_hash = parse_env_hash(args.get("env_hash"))
        self.nodes = nodes or cluster_nodes
        self.kube_context = args.get("kube_context") or CLUSTERS[self.cluster].kube_context
        self.model_bucket = CLUSTERS[self.cluster].model_bucket
        self.connection = connection_path(args, self.cluster)
        if self.connection:
            self.connection = str(Path(self.connection).expanduser().resolve())
        self.out = Path(args.get("out_dir", "semip_job")).expanduser().resolve()
        if "," in str(self.out):
            raise SystemExit(f"out_dir= must not contain a comma (it is passed in models=): {self.out}")
        self.get = get
        self.options = onboard_options(self.tests)
        self.warnings: list[str] = []

    def fetch(self) -> list[ModelSpec]:
        """Resolve and fetch the configs, record where they came from, parse every line."""
        self.out.mkdir(parents=True, exist_ok=True)
        self.sha = resolve_ref(self.configs, self.get)
        url = raw_url(self.configs, self.sha)
        text = self.get(url).decode("utf-8")
        self.configs_file = self.out / "configs.jsonl"
        self.configs_file.write_text(text, encoding="utf-8")
        specs = load_models(str(self.configs_file))
        lines = text.splitlines()
        self.lines = {s.name: lines[line_number(s) - 1] for s in specs}
        self.record = {"url": self.configs.url, "repo": self.configs.repo, "ref": self.configs.ref,
                       "sha": self.sha, "path": self.configs.path, "raw_url": url,
                       "sha256": hashlib.sha256(text.encode()).hexdigest(), "jobs": len(specs)}
        (self.out / "configs.json").write_text(json.dumps(self.record, indent=1) + "\n")
        logger.info("configs: %s@%s %s (ref %s), %d job(s)", self.configs.repo, self.sha, self.configs.path,
                    self.configs.ref, len(specs))
        return specs

    def publish_argv(self) -> list[str]:
        return [sys.executable, "-m", f"{PACKAGE}.publish", f"config={self.connection or '<connection>'}",
                f"image_tag={self.image_tag}", f"models={self.configs_file}", f"max_gpus={self.max_gpus}",
                "reserve_gpus=0", f"model_bucket={self.model_bucket}", f"out_dir={self.out / 'publish'}"]

    def onboard_argv(self, models: Path) -> list[str]:
        return [sys.executable, "-m", f"{PACKAGE}.onboard", f"config={self.connection or '<connection>'}",
                f"debug_image_tag={self.image_tag}", f"models={models}",
                f"variants={self.options['variants']}", f"checks={self.options['checks']}",
                "require_restore=true", f"max_gpus={self.max_gpus}", "reserve_gpus=0",
                f"model_bucket={self.model_bucket}", f"kube_context={self.kube_context}",
                f"publish_dir={self.out / 'publish'}", f"out_dir={self.out / 'onboard'}"]

    def plan(self, specs: list[ModelSpec]) -> dict:
        """What a run would do; written to ``plan.json`` and ``plan.txt``."""
        problem = check_connection(self.connection)
        if problem:
            self.warnings.append(problem)
        plan: dict[str, Any] = {
            "cluster": self.cluster, "kube_context": self.kube_context, "model_bucket": self.model_bucket,
            "image_tag": self.image_tag, "configs": self.record, "tests": self.tests, "max_gpus": self.max_gpus,
            "dry_run": self.dry_run, "connection": self.connection, "onboard": self.options, "jobs": [],
            "commands": {"publish": shlex.join(self.publish_argv())},
        }
        too_big: dict[str, str] = {}
        if self.options:
            fits, too_big = fit_onboard(specs, self.options["variants"].split(","), self.max_gpus)
            self.onboard_file = write_jsonl(fits, self.out / "onboard.jsonl", self.lines)
            plan["commands"]["onboard"] = shlex.join(self.onboard_argv(self.onboard_file))
            if too_big:
                self.warnings.append(f"{len(too_big)} of {len(specs)} config(s) need more than max_gpus="
                                     f"{self.max_gpus} for their tests; they are published but not tested, "
                                     "and the run fails")
        skeletons = self.skeletons(specs, plan) if self.dry_run else {}
        for s in specs:
            entry = {"name": s.name, "line": line_number(s), "model_name": s.sub_job.get("model_name"),
                     "max_seq_len": s.max_seq_len, "n_gpus": s.n_gpus, "tensor_parallel_size": s.tensor_parallel_size}
            if self.options:
                gpus, nodes = footprint(s, self.options["variants"].split(","))
                entry.update(test_gpus=gpus, test_nodes=nodes, onboard=too_big.get(s.name, "yes"))
            entry.update(skeletons.get(s.name, {}))
            plan["jobs"].append(entry)
        plan["warnings"] = self.warnings
        (self.out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
        text = plan_text(plan)
        (self.out / "plan.txt").write_text(text + "\n")
        for line in text.splitlines():
            logger.info("%s", line)
        return plan

    def skeletons(self, specs: list[ModelSpec], plan: dict) -> dict[str, dict]:
        """Each job's cfg12 and the skeletons the node cache verified for it, by job;
        sets ``plan["node_cache"]``. Reads the node labels once, if kubectl can."""
        keyings = {s.name: keys.keying(s.sub_job, json.loads(self.lines[s.name]).get("hardware")) for s in specs}
        try:
            cache = keys.NodeCache(self.nodes(self.kube_context), self.env_hash)
            note = None
        except NodesUnavailable as exc:
            cache, note = None, f"not checked: {exc}"
            logger.info("node cache %s", note)
        lookups: dict[str, dict] = {}
        chosen: dict[str, set] = {}
        for s in specs:
            k = keyings[s.name]
            if k.cfg12:
                lookups[s.name] = {k.reasoning_parser: cache.lookup(k.slug, k.cfg12)} if cache else {}
            elif k.candidates and cache:
                lookups[s.name] = {r: cache.lookup(k.slug, c) for r, c in k.candidates.items()}
                hit = [r for r, lk in lookups[s.name].items() if lk.found]
                if len(hit) == 1:
                    chosen.setdefault(s.sub_job.get("model_name"), set()).add(hit[0])
        # One checkpoint, one reasoner: what the labels show for a model holds for all its jobs.
        reasoner = {m: next(iter(rs)) for m, rs in chosen.items() if len(rs) == 1}
        out: dict[str, dict] = {}
        counts = {"restore": 0, "partial": 0, "dump": 0, "unknown": 0, "other_image": 0}
        for s in specs:
            k = keyings[s.name]
            entry: dict[str, Any] = {"slug": k.slug, "cfg12": k.cfg12, "reasoning_parser": k.reasoning_parser,
                                     "reasoning_parser_from": "job" if k.cfg12 else None}
            if k.reason:
                entry["cfg12_note"] = k.reason
            lk = lookups.get(s.name, {})
            if not k.cfg12 and k.candidates:
                entry["cfg12_candidates"] = {r or "none": c for r, c in k.candidates.items()}
                r = reasoner.get(s.sub_job.get("model_name"), "?")
                if r != "?":
                    entry.update(cfg12=k.candidates[r], reasoning_parser=r, reasoning_parser_from="node labels")
                    lk = {r: lk[r]}
                else:
                    entry["cfg12_note"] = ("the job names no reasoning_parser, so cfg12 depends on the one chat "
                                           "pairs with the checkpoint's architecture; "
                                           + ("no candidate is on the nodes" if cache else "the node labels tell which"))
            if cache is None:
                entry.update(skeleton="not checked", existing=[], existing_text="not checked")
                out[s.name] = entry
                continue
            found = [f for result in lk.values() for f in result.found]
            unreadable = max((result.unreadable for result in lk.values()), default=0)
            weights_known = bool(cache.weights(k.slug))
            pairs = sorted(((f, match_of(f, self.env_hash, weights_known)) for f in found),
                           key=lambda p: (p[1] != "exists", -p[0].nodes, p[0].env))
            found, matches = [f for f, _ in pairs], [m for _, m in pairs]
            status = skeleton_status(found, matches, cache.nodes, self.env_hash)
            if k.reason:
                status, text = "unknown", f"? ({k.reason})"
            elif not found and unreadable:
                status, text = "unknown", f"? ({unreadable} skeleton label(s) of {k.slug} cut before cfg12)"
            else:
                text = existing_text(found, cache.nodes, self.env_hash, matches)
            counts[status] += 1
            counts["other_image"] += status == "dump" and "exists for another image" in matches
            entry.update(skeleton=status, existing_text=text,
                         existing=[{"env": f.env, "env_exact": f.env_exact, "nodes": f.nodes, "of": cache.nodes,
                                    "match": m} for f, m in zip(found, matches)])
            out[s.name] = entry
        plan["node_cache"] = {"checked": cache is not None, "note": note, "kube_context": self.kube_context,
                              "env_hash": self.env_hash, "nodes": cache.nodes if cache else None,
                              "env_candidates": sorted(cache.envs) if cache else [],
                              "summary": counts if cache else None}
        return out

    def run(self, specs: list[ModelSpec], run_child: Callable | None = None) -> int:
        """Publish, then onboard what published and fits; write and log the result."""
        run_child = run_child or child_process
        problem = check_connection(self.connection)
        if problem:
            raise SystemExit(problem)
        if self.options and not shutil.which("kubectl"):
            logger.warning("kubectl is not on PATH: onboard cannot wait for the node cache to verify the images")
        code = run_child(self.publish_argv())
        logger.info("publish exited %d", code)
        summary_path = self.out / "publish" / self.image_tag / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        status = {s.name: (summary.get(s.name) or {}).get("status") or f"missing (publish exited {code})"
                  for s in specs}
        reports: dict[str, dict] = {}
        tests: dict[str, dict[str, str]] = {s.name: {} for s in specs}
        if self.options:
            published = [s for s in specs if status[s.name].startswith(PUBLISHED)]
            fits, too_big = fit_onboard(published, self.options["variants"].split(","), self.max_gpus)
            for s in specs:
                why = ("not run: not published" if not status[s.name].startswith(PUBLISHED)
                       else f"not run: {too_big[s.name]}" if s.name in too_big else None)
                if why:
                    tests[s.name] = {t: why for t in self.tests}
            if fits:
                models = write_jsonl(fits, self.out / "onboard.jsonl", self.lines)
                started = utc()
                code = run_child(self.onboard_argv(models))
                logger.info("onboard exited %d", code)
                reports = onboard_reports(self.out / "onboard", started)
                for s in fits:
                    tests[s.name] = verdicts(reports.get(s.name), self.tests, code)
        rows = [{"job": s.name, "publish": status[s.name], "tests": tests[s.name],
                 "report": (reports.get(s.name) or {}).get("report")} for s in specs]
        failed = [r["job"] for r in rows if not r["publish"].startswith(PUBLISHED)
                  or any(not v.startswith(("PASS", "WARN")) for v in r["tests"].values())]
        result = {"cluster": self.cluster, "image_tag": self.image_tag, "configs": self.record,
                  "tests": self.tests, "jobs": rows, "failed": failed}
        (self.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        text = result_table(rows, self.tests)
        (self.out / "result.md").write_text(text + "\n")
        for line in text.splitlines():
            logger.info("%s", line)
        logger.info("%d of %d job(s) failed%s", len(failed), len(rows), f": {', '.join(failed)}" if failed else "")
        return 1 if failed else 0


def plan_text(plan: dict) -> str:
    c = plan["configs"]
    onboard = plan["onboard"]
    lines = [
        f"cluster     {plan['cluster']} (kube context {plan['kube_context']}, model bucket {plan['model_bucket']})",
        f"image       {plan['image_tag']}",
        f"configs     {c['repo']}@{c['sha']} {c['path']} (requested ref {c['ref']})",
        f"jobs        {len(plan['jobs'])}",
        f"tests       {','.join(plan['tests']) or 'none (publish only)'}"
        + (f" -> onboard variants={onboard['variants']} checks={onboard['checks']}" if onboard else ""),
        f"max_gpus    {plan['max_gpus']} (reserve_gpus=0)",
        f"connection  {plan['connection'] or '-'}",
    ]
    cache = plan.get("node_cache")
    if cache:
        lines.append("node cache  " + (cache["note"] or f"{cache['nodes']} model-cache node(s) on {cache['kube_context']}, "
                                       + (f"env_hash {cache['env_hash']}" if cache["env_hash"] else "no env_hash")))
    lines += [
        "",
        "| line | job | model_name | max_seq_len | n_gpus | TP |" + (" cfg12 | existing |" if cache else "")
        + (" test GPUs | onboard |" if onboard else ""),
        "|---|---|---|---|---|---|" + ("---|---|" if cache else "") + ("---|---|" if onboard else ""),
    ]
    for j in plan["jobs"]:
        lines.append(f"| {j['line']} | {j['name']} | {j['model_name']} | {j['max_seq_len']} | {j['n_gpus']} "
                     f"| {j['tensor_parallel_size']} |"
                     + (f" {j['cfg12'] or '?'} | {j['existing_text'].replace('|', '/')} |" if cache else "")
                     + (f" {j['test_gpus']} | {j['onboard']} |" if onboard else ""))
    if cache:
        lines += ["", skeleton_summary(cache)]
        if any(not j["cfg12"] and "cfg12_candidates" in j for j in plan["jobs"]):
            lines.append("cfg12 ?: the job names no reasoning_parser and the node labels did not show which one chat "
                         "adds for the checkpoint (plan.json lists the candidates)")
    lines += ["", f"publish: {plan['commands']['publish']}"]
    if "onboard" in plan["commands"]:
        lines.append(f"onboard: {plan['commands']['onboard']}")
    lines += [f"WARNING: {w}" for w in plan["warnings"]]
    return "\n".join(lines)


def skeleton_summary(cache: dict) -> str:
    counts = cache["summary"]
    if counts is None:
        return f"skeletons: not checked ({cache['note'].removeprefix('not checked: ')})"
    text = (f"skeletons: {counts['restore']} will restore (all nodes), {counts['partial']} exist but not on "
            f"all nodes, {counts['dump']} will dump")
    if counts["other_image"]:
        text += f" ({counts['other_image']} of them exist for another image)"
    if counts["unknown"]:
        text += f", {counts['unknown']} unknown"
    return text + ("" if cache["env_hash"] else " (env unconfirmed: pass env_hash= to match this image)")


def onboard_reports(out_dir: Path, since: str) -> dict[str, dict]:
    """Each config's onboard report from the run started at ``since``, by name."""
    found: dict[str, dict] = {}
    for path in sorted(out_dir.glob("*.json")):
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or "model" not in r or (r.get("started_at") or "") < since:
            continue
        r["report"] = str(path.with_suffix(".md"))
        found[r["model"]["name"]] = r
    return found


def verdicts(result: dict | None, tests: list[str], code: int) -> dict[str, str]:
    """``{test: "PASS" | "WARN: ..." | "FAIL: ..."}`` from one onboard report."""
    if result is None:
        return {t: f"FAIL: no onboard report (onboard exited {code})" for t in tests}
    out = {"functional": functional_verdict(result)}
    missing = result.get("not_measured") or {}
    for test, check in (("correctness", "correctness"), ("performance", "perf")):
        if test not in tests:
            continue
        unmeasured = [f"{v}: {why}" for v, why in missing.items() if check in why.split(":")[0]]
        if check == "correctness":
            checks = [c for c in result.get("correctness") or [] if c["required"]]
            failed = [c["name"] for c in checks if not c["passed"]]
        else:
            checks = report.gated_perf_checks(result)
            failed = [c["name"] for c in checks if not c["passed"]]
        if failed:
            out[test] = f"FAIL: {', '.join(failed)}"
        elif unmeasured or not checks:
            out[test] = "FAIL: not measured" + (f" ({'; '.join(unmeasured)[:300]})" if unmeasured
                                                else f" ({(result.get('error') or result.get('perf_note') or '')[:300]})")
        else:
            out[test] = "PASS"
    return {t: out[t] for t in tests}


def functional_verdict(result: dict) -> str:
    """Whether on_hit restored from the published image."""
    st = result.get("startup") or {}
    restore = st.get("semi_p") or {}
    if restore.get("cache_miss"):
        return "FAIL: on_hit cold-started (cache miss)"
    if restore.get("restored"):
        return "PASS"
    if restore.get("markers_visible"):
        return "FAIL: not every replica restored from the published image"
    if "submit_to_running_s" in (st.get("on_timeline") or {}) and not result.get("error"):
        return "WARN: restore markers not in the logs; restore not confirmed"
    return f"FAIL: {(result.get('error') or 'on_hit did not come up')[:300]}"


def result_table(rows: list[dict], tests: list[str]) -> str:
    lines = ["| job | publish | " + " | ".join(tests) + (" |" if tests else ""),
             "|---|---|" + "---|" * len(tests)]
    for r in rows:
        cells = [r["job"], r["publish"][:120]] + [r["tests"].get(t, "-")[:120] for t in tests]
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(lines)


def child_process(argv: list[str]) -> int:
    """Run ``argv`` from the repository root, passing SIGINT, SIGTERM and SIGHUP on
    so it cancels its jobs; exit with 128 + the signal once it has."""
    logger.info("running: %s", shlex.join(argv))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
    proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env)
    received: list[int] = []

    def forward(signum: int, _frame: Any) -> None:
        received.append(signum)
        proc.send_signal(signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        code = proc.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if received:
        raise SystemExit(128 + received[0])
    return code


def main(argv: list[str], get: Callable = http_get, run_child: Callable | None = None,
         nodes: Callable | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    job = SemipJob(parse_kv_args(argv), get, nodes)
    try:
        specs = job.fetch()
    except ValueError as exc:
        raise SystemExit(f"configs: {exc}") from None
    check_budget(specs, job.max_gpus)
    job.plan(specs)
    if job.dry_run:
        logger.info("dry run: nothing submitted; plan in %s", job.out / "plan.txt")
        return 0
    return job.run(specs, run_child)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
