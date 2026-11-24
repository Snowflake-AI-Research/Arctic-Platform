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
        [connection=CONN.json] [kube_context=CTX] [show_all=false]

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
  show_all     dry run only, for admins: report the full skeleton lookup for any
               list (default false); it changes what is printed, not access

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

A dry run also computes each job's cfg12 offline (``keys``). A job that names no
reasoning_parser gets the one chat pairs with its model's architecture: the one
vLLM resolved in a publish (``keys.MODEL_ARCHITECTURES``), else the first its
Hugging Face config.json names (``architectures`` in the plan); when neither is
known, only the node labels can tell, and cfg12 is ``?`` without them. When kubectl
reaches the cluster's context, it reads the nodes once (``kubectl get nodes``) to
report which of those skeletons the model-cache nodes have verified: per env12,
on how many of them. ``existing`` is ``none`` when publish would dump.

The same read gives this image's env12, ``sha256(<digest hex> + <driver>)[:12]``
as the engine computes it. The digest is the one the nodes list for
``image_tag`` (any registry); a tag listed with several digests (tags can move)
gives one env12 candidate each, and the plan says so. A tag no node lists means
no skeleton of the image can be on the cluster: every job will dump. The driver
comes from the model-cache nodes' GPU Feature Discovery labels when present;
neither cluster has them today, nor any other read-only source of it (node
annotations and ``nodeInfo``, the device plugin's, DCGM exporter's and model
cache's logs), so ``DEFAULT_DRIVER_VERSION`` is assumed with a warning. Either
way a skeleton label carrying the env12 confirms it. ``env`` in the plan holds
the digests, the driver, its source and the env12s.

That full lookup is for the official catalog only: configs in
snowflakedb/cortex-training at ``main`` or at a commit on main (GitHub's compare
API says main is ``identical`` or ``ahead``; if it cannot be asked, the list is
not official). There each skeleton is marked as this image's or another
image's. Any other list is restricted, so it cannot show other users' dumps:
each job gets only its own skeleton under this image's env12 (``exists: N/M
nodes`` or ``none (will dump)``), and nothing about other images or labels.
Without kubectl either lookup is skipped with a note. ``visibility`` in the
plan says which, and ``show_all=true`` forces the full lookup.

After the jobs, every list's plan holds the backend inventory: every skeleton
the nodes verified under this image's env12, for every model (cfg12, wt12,
nodes, and ``in this list`` when it is one of these jobs'), and the weight sets
they use, then ``backend inventory: S skeletons across K models (F on all
nodes), W weight sets``. It is the same for official, restricted and show_all
lists and holds nothing of another env12. Cut labels are read by recomputing
their digest (``keys.NodeCache.inventory``); those whose env12 still cannot be
told are only counted, per model. It is ``none`` when the image is not pulled
and ``not checked`` without kubectl; ``backend_inventory`` in the plan.
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
OFFICIAL_OWNER = "snowflakedb"
OFFICIAL_BRANCH = "main"
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
# GITHUB_TOKEN is sent to these hosts only.
GITHUB_HOSTS = ("api.github.com", "raw.githubusercontent.com")
HF_CONFIG_URL = "https://huggingface.co/{model}/resolve/main/config.json"
HF_MODEL_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
PUBLISHED = ("published", "already")
# An env12 shown only this far is taken as this image's when no weight hash can confirm it.
ENV_PREFIX_CHARS = 6
# The H200 driver both clusters run (publish logs); assumed when no cluster source names it.
DEFAULT_DRIVER_VERSION = "580.159.03"
NOT_PULLED = "none (will dump; image not pulled on any node yet)"
INVENTORY_NOT_PULLED = "none (image not pulled on any node yet)"


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
    if os.environ.get("GITHUB_TOKEN") and urllib.parse.urlsplit(url).netloc in GITHUB_HOSTS:
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


def visibility(configs: ConfigsRef, sha: str, get: Callable = http_get) -> tuple[bool, str]:
    """``(official, why)``: official when the configs are in snowflakedb/cortex-training at a
    commit on main (the ref is main, or GitHub's compare puts ``sha`` at or behind main's head)."""
    official = f"official catalog ({OFFICIAL_OWNER}/{CONFIGS_REPO} {OFFICIAL_BRANCH})"
    restricted = f"restricted: {configs.repo}@{configs.ref} is not the official catalog on {OFFICIAL_BRANCH}"
    if configs.owner.lower() != OFFICIAL_OWNER:
        return False, restricted
    if configs.ref == OFFICIAL_BRANCH:
        return True, official
    url = f"https://api.github.com/repos/{OFFICIAL_OWNER}/{CONFIGS_REPO}/compare/{sha}...{OFFICIAL_BRANCH}"
    try:
        status = json.loads(get(url, "application/vnd.github+json"))["status"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"{restricted} (could not check: {exc})"
    if status in ("identical", "ahead"):
        return True, f"{official[:-1]}; {sha[:12]} is on {OFFICIAL_BRANCH})"
    return False, f"{restricted} (compare {sha[:12]}...{OFFICIAL_BRANCH}: {status})"


class ArchitectureUnknown(Exception):
    """A model's architecture could not be found; the message says why."""


def model_architecture(model_name: str, get: Callable = http_get) -> tuple[str, str]:
    """``(architecture, source)``: the one vLLM resolved for the baked checkpoint when a publish
    log recorded it (``keys.MODEL_ARCHITECTURES``), else the first one the model's config.json
    on Hugging Face names."""
    if model_name in keys.MODEL_ARCHITECTURES:
        return keys.MODEL_ARCHITECTURES[model_name], "publish logs"
    if not HF_MODEL_ID.fullmatch(model_name or ""):
        raise ArchitectureUnknown(f"{model_name!r} is not a Hugging Face model id")
    url = HF_CONFIG_URL.format(model=model_name)
    try:
        names = json.loads(get(url)).get("architectures")
    except (ValueError, AttributeError) as exc:
        raise ArchitectureUnknown(str(exc)) from None
    if not isinstance(names, list) or not names or not isinstance(names[0], str):
        raise ArchitectureUnknown(f"{url} names no architectures")
    return names[0], "Hugging Face config.json"


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


def derive_env(items: list[dict], image_tag: str, cache: keys.NodeCache) -> dict:
    """This image's env12 candidates from one ``get nodes`` read: the digests the nodes list
    for ``image_tag``, the driver (node labels, else ``DEFAULT_DRIVER_VERSION``), and whether
    the skeleton labels in ``cache`` show each env12. Recorded as ``plan["env"]``."""
    digests = keys.image_digests(items, image_tag)
    drivers, label = keys.node_drivers(items)
    if drivers:
        source, assumed = f"node label {label}", False
    else:
        drivers, source, assumed = {DEFAULT_DRIVER_VERSION: 0}, "DEFAULT_DRIVER_VERSION", True
    candidates = {keys.env_hash(d, v): {"digest": d, "driver": v} for d in digests for v in drivers}
    for env, inputs in candidates.items():
        inputs["confirmed"] = cache.shows_env(env, ENV_PREFIX_CHARS)
    return {"derived": True, "image_tag": image_tag, "digests": digests, "drivers": drivers,
            "driver_source": source, "driver_assumed": assumed, "env12": list(candidates),
            "candidates": candidates, "ambiguous": len(candidates) > 1, "pulled": bool(digests)}


def env_warnings(env: dict) -> list[str]:
    if not env.get("pulled"):
        return []
    out = []
    if len(env["digests"]) > 1:
        out.append(f"{env['image_tag']} is listed with {len(env['digests'])} digests across the nodes (tags can "
                   "move): " + ", ".join(f"{d[:12]} on {n} node(s)" for d, n in env["digests"].items())
                   + f"; a skeleton under any of their env12s ({', '.join(env['env12'])}) counts as this image's")
    if len(env["drivers"]) > 1:
        out.append(f"the model-cache nodes' {env['driver_source']} names {len(env['drivers'])} drivers: "
                   + ", ".join(f"{v} on {n} node(s)" for v, n in env["drivers"].items())
                   + "; each gives an env12 candidate")
    if env["driver_assumed"]:
        whole = [e for e, c in env["candidates"].items() if c["confirmed"] == "whole"]
        start = [e for e, c in env["candidates"].items() if c["confirmed"] == "start"]
        why = (f"a skeleton label on the nodes carries the env12 it gives ({', '.join(whole)})" if whole
               else f"a cut skeleton label on the nodes shows the start of the env12 it gives ({', '.join(start)})"
               if start else f"no skeleton label confirms env12 {', '.join(env['env12'])}, so if the nodes run "
                             "another driver this image's skeletons are reported missing")
        out.append(f"GPU driver assumed to be {DEFAULT_DRIVER_VERSION} (DEFAULT_DRIVER_VERSION): no read-only "
                   f"cluster source names it; {why}")
    return out


def match_of(found: keys.Found, envs: set[str], weights_known: bool) -> str:
    """Whether a skeleton is this image's (env12 in ``envs``): ``exists``, ``exists for
    another image`` or ``unconfirmed`` (its label shows too little of env12)."""
    if found.env in envs:
        return "exists"
    if found.env_exact or not any(e.startswith(found.env) for e in envs):
        return "exists for another image"
    if weights_known:
        # The label was checked against envs with every weight hash shown on the nodes.
        return "exists for another image"
    return "exists" if len(found.env) >= ENV_PREFIX_CHARS else "unconfirmed"


def own_nodes(cache: keys.NodeCache, slug: str, cfg12: str, envs: set[str]) -> tuple[int, str | None, bool]:
    """On how many nodes this image's skeleton of ``slug`` with ``cfg12`` is verified, its env12,
    and whether a label of that cfg12 is cut too short to tell if it is this image's."""
    weights_known = bool(cache.weights(slug))
    result = cache.lookup(slug, cfg12)
    matches = [(f, match_of(f, envs, weights_known)) for f in result.found]
    best = max(((f.nodes, f.env) for f, m in matches if m == "exists"), default=(0, None))
    env = next((e for e in sorted(envs) if best[1] is not None and e.startswith(best[1])), None)
    return best[0], env, bool(result.unreadable) or any(m == "unconfirmed" for _, m in matches)


def skeleton_status(found: list[keys.Found], matches: list[str], nodes: int) -> str:
    """``restore`` (this image's skeleton is on every node), ``partial``, ``dump`` or ``unknown``."""
    best = max((f.nodes for f, m in zip(found, matches) if m == "exists"), default=0)
    if nodes and best == nodes:
        return "restore"
    if best:
        return "partial"
    return "unknown" if "unconfirmed" in matches else "dump"


def existing_text(found: list[keys.Found], nodes: int, matches: list[str], pulled: bool = True) -> str:
    if not found:
        return "none" if pulled else NOT_PULLED
    mine = [f"{env_text(f)}: {f.nodes}/{nodes} nodes ({m})" for f, m in zip(found, matches) if m != "exists for another image"]
    other = [f"{env_text(f)} {f.nodes}/{nodes}" for f, m in zip(found, matches) if m == "exists for another image"]
    parts = [", ".join(mine)] if mine else ([] if pulled else [NOT_PULLED])
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
        if "env_hash" in args:
            raise SystemExit("env_hash= is no longer taken: the dry run derives this image's env12 from the digest "
                             "the nodes list for image_tag and the GPU driver; drop the argument")
        self.show_all = as_bool(args.get("show_all") or "false")
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
        self.envs: set[str] = set()
        self.other_envs: set[str] = set()
        self._keyings: dict[str, keys.Keying] = {}

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
        skeletons: dict[str, dict] = {}
        if self.dry_run:
            official, why = visibility(self.configs, self.sha, self.get)
            plan["visibility"] = {"official": official, "reason": why, "show_all": self.show_all}
            logger.info("visibility: %s", why)
            if self.show_all:
                logger.warning("show_all=true (admin override): reporting every skeleton the node cache has "
                               "for these jobs, of any image")
            full = official or self.show_all
            plan["architectures"] = self.architectures(specs)
            cache, note = self.node_cache(plan)
            lookup = self.skeletons if full else self.own_skeletons
            skeletons = lookup(specs, plan, cache, note)
            plan["backend_inventory"] = self.inventory(specs, plan, cache, note)
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

    def node_cache(self, plan: dict) -> tuple[keys.NodeCache | None, str | None]:
        """Read the nodes once, if kubectl can; derive this image's env12s from them into
        ``plan["env"]`` and ``self.envs``, and return the node cache keyed with them (or the
        note why there is none)."""
        try:
            items = self.nodes(self.kube_context)
        except NodesUnavailable as exc:
            note = f"not checked: {exc}"
            logger.info("node cache %s", note)
            plan["env"] = {"derived": False, "image_tag": self.image_tag, "note": note}
            return None, note
        env = derive_env(items, self.image_tag, keys.NodeCache(items))
        plan["env"] = env
        self.warnings.extend(env_warnings(env))
        self.envs = set(env["env12"])
        self.other_envs = {keys.env_hash(d, v) for d in keys.repository_digests(items, self.image_tag)
                           for v in env["drivers"]} - self.envs
        logger.info("env %s", env_text_line(env))
        return keys.NodeCache(items, env["env12"]), None

    def keyings(self, specs: list[ModelSpec]) -> dict[str, keys.Keying]:
        """Each job's skeleton key as far as its body determines it, by job (computed once)."""
        for s in specs:
            if s.name not in self._keyings:
                self._keyings[s.name] = keys.keying(s.sub_job, json.loads(self.lines[s.name]).get("hardware"))
        return {s.name: self._keyings[s.name] for s in specs}

    def architectures(self, specs: list[ModelSpec]) -> dict[str, dict]:
        """For each model with a job that names no reasoner: its architecture, where it came
        from and the reasoner chat pairs with it, or why the architecture is not known."""
        out: dict[str, dict] = {}
        for s, k in zip(specs, self.keyings(specs).values()):
            model = s.sub_job.get("model_name")
            if model in out or not k.candidates:
                continue
            try:
                arch, source = model_architecture(model, self.get)
            except ArchitectureUnknown as exc:
                out[model] = {"architecture": None, "note": str(exc)}
                continue
            out[model] = {"architecture": arch, "source": source, "reasoning_parser": keys.chat_reasoner(arch)}
        return out

    def reasoners(self, chosen: dict[str, set], archs: dict[str, dict]) -> dict[str, tuple[str | None, str]]:
        """Each model's reasoner for its jobs that name none, and what says so: the node labels
        when they show exactly one, else the model's architecture. When the two disagree the
        labels' is used, with a warning."""
        out = {m: (next(iter(rs)), "node labels") for m, rs in chosen.items() if len(rs) == 1}
        for model, a in archs.items():
            if a["architecture"] is None:
                continue
            via = f"architecture {a['architecture']} ({a['source']})"
            if model not in out:
                out[model] = (a["reasoning_parser"], via)
            elif out[model][0] != a["reasoning_parser"]:
                self.warnings.append(f"{model}: the node labels show its skeleton with reasoning_parser="
                                     f"{out[model][0]}, but {via} pairs with {a['reasoning_parser']}; "
                                     "the labels' is used")
        return out

    def inventory(self, specs: list[ModelSpec], plan: dict, cache: keys.NodeCache | None,
                  note: str | None) -> dict:
        """Every skeleton the nodes verified under this image's env12, for every model, with the
        weight sets they use, and whether it is one of these jobs'. The same for every list:
        nothing of another env12 is in it. Recorded as ``plan["backend_inventory"]``."""
        if cache is None:
            return {"checked": False, "note": note}
        if not plan["env"].get("pulled"):
            return {"checked": True, "pulled": False, "note": INVENTORY_NOT_PULLED}
        jobs: dict[tuple[str, str], list[str]] = {}
        for name, k in self.keyings(specs).items():
            for cfg in [k.cfg12] if k.cfg12 else k.candidates.values():
                jobs.setdefault((k.slug, cfg), []).append(name)
        cfgs: dict[str, set[str]] = {}
        for slug, cfg in jobs:
            cfgs.setdefault(slug, set()).add(cfg)
        inv = cache.inventory(self.envs, cfgs, self.other_envs)
        models: dict[str, dict] = {}
        for sk in inv.skeletons:
            model = models.setdefault(sk.slug, {"model": sk.slug, "skeletons": [], "weights": []})
            mine = jobs.get((sk.slug, sk.cfg12), [])
            model["skeletons"].append({"cfg12": sk.cfg12, "env12": sk.env12, "wt12": sk.wt12, "wt12_exact": sk.wt_exact,
                                       "nodes": sk.nodes, "of": cache.nodes, "in_list": bool(mine), "jobs": mine})
        for (slug, wt), nodes in sorted(inv.weights.items()):
            models[slug]["weights"].append({"wt12": wt, "wt12_exact": nodes is not None, "nodes": nodes,
                                            "of": cache.nodes})
        rows = [models[m] for m in sorted(models, key=lambda m: (m.lower(), m))]
        totals = {"skeletons": len(inv.skeletons), "models": len(rows),
                  "on_all_nodes": sum(sk.nodes == cache.nodes for sk in inv.skeletons), "weights": len(inv.weights)}
        unattributed = {"labels": sum(inv.unattributed.values()), "models": list(inv.unattributed),
                        "kind_unknown": inv.either}
        return {"checked": True, "pulled": True, "image_tag": self.image_tag, "env12": sorted(self.envs),
                "nodes": cache.nodes, "models": rows, "totals": totals, "unattributed": unattributed}

    def skeletons(self, specs: list[ModelSpec], plan: dict, cache: keys.NodeCache | None,
                  note: str | None) -> dict[str, dict]:
        """Each job's cfg12 and the skeletons the node cache verified for it, by job;
        sets ``plan["node_cache"]``."""
        keyings = self.keyings(specs)
        pulled = bool(plan["env"].get("pulled"))
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
        reasoner = self.reasoners(chosen, plan["architectures"])
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
                model = s.sub_job.get("model_name")
                entry["cfg12_candidates"] = {r or "none": c for r, c in k.candidates.items()}
                r, via = reasoner.get(model, ("?", None))
                if r != "?":
                    entry.update(cfg12=k.candidates[r], reasoning_parser=r, reasoning_parser_from=via)
                    lk = {r: lk[r]} if lk else {}
                else:
                    entry["cfg12_note"] = ("the job names no reasoning_parser, so cfg12 depends on the one chat "
                                           "pairs with the checkpoint's architecture, which is not known ("
                                           + plan["architectures"][model]["note"] + "); "
                                           + ("no candidate is on the nodes" if cache else "the node labels tell which"))
            if cache is None:
                entry.update(skeleton="not checked", existing=[], existing_text="not checked")
                out[s.name] = entry
                continue
            found = [f for result in lk.values() for f in result.found]
            unreadable = max((result.unreadable for result in lk.values()), default=0)
            weights_known = bool(cache.weights(k.slug))
            pairs = sorted(((f, match_of(f, self.envs, weights_known)) for f in found),
                           key=lambda p: (p[1] != "exists", -p[0].nodes, p[0].env))
            found, matches = [f for f, _ in pairs], [m for _, m in pairs]
            status = skeleton_status(found, matches, cache.nodes)
            if not pulled:
                text = existing_text(found, cache.nodes, matches, pulled=False)
            elif k.reason:
                status, text = "unknown", f"? ({k.reason})"
            elif not found and unreadable:
                status, text = "unknown", f"? ({unreadable} skeleton label(s) of {k.slug} cut before cfg12)"
            else:
                text = existing_text(found, cache.nodes, matches)
            counts[status] += 1
            counts["other_image"] += status == "dump" and "exists for another image" in matches
            entry.update(skeleton=status, existing_text=text,
                         existing=[{"env": f.env, "env_exact": f.env_exact, "nodes": f.nodes, "of": cache.nodes,
                                    "match": m} for f, m in zip(found, matches)])
            out[s.name] = entry
        plan["node_cache"] = {"checked": cache is not None, "note": note, "kube_context": self.kube_context,
                              "nodes": cache.nodes if cache else None, "pulled": pulled,
                              "env_candidates": sorted(cache.envs) if cache else [],
                              "summary": counts if cache else None}
        return out

    def own_skeletons(self, specs: list[ModelSpec], plan: dict, cache: keys.NodeCache | None,
                      note: str | None) -> dict[str, dict]:
        """``skeletons`` for a list that is not the official catalog: only each job's own
        skeleton under this image's env12. Nothing about other images, other cfg12
        candidates or other labels is reported."""
        keyings = self.keyings(specs)
        pulled = bool(plan["env"].get("pulled"))
        hits: dict[str, dict] = {}
        chosen: dict[str, set] = {}
        for s in specs:
            k = keyings[s.name]
            cfgs = {k.reasoning_parser: k.cfg12} if k.cfg12 else k.candidates
            hits[s.name] = ({r: own_nodes(cache, k.slug, c, self.envs) for r, c in cfgs.items()}
                            if cache and pulled else {})
            hit = [r for r, (n, _, _) in hits[s.name].items() if n]
            if not k.cfg12 and len(hit) == 1:
                chosen.setdefault(s.sub_job.get("model_name"), set()).add(hit[0])
        reasoner = self.reasoners(chosen, plan["architectures"])
        out: dict[str, dict] = {}
        counts = {"restore": 0, "partial": 0, "dump": 0, "unknown": 0}
        for s in specs:
            k = keyings[s.name]
            entry: dict[str, Any] = {"slug": k.slug, "cfg12": k.cfg12, "reasoning_parser": k.reasoning_parser,
                                     "reasoning_parser_from": "job" if k.cfg12 else None}
            if k.reason:
                entry["cfg12_note"] = k.reason
            mine = hits[s.name]
            if not k.cfg12 and k.candidates:
                model = s.sub_job.get("model_name")
                entry["cfg12_candidates"] = {r or "none": c for r, c in k.candidates.items()}
                r, via = reasoner.get(model, ("?", None))
                if r != "?":
                    entry.update(cfg12=k.candidates[r], reasoning_parser=r, reasoning_parser_from=via)
                    mine = {r: mine[r]} if mine else {}
                else:
                    entry["cfg12_note"] = ("the job names no reasoning_parser, so cfg12 depends on the one chat "
                                           "pairs with the checkpoint's architecture, which is not known ("
                                           + plan["architectures"][model]["note"] + "); "
                                           + ("no candidate's skeleton exists for this image" if cache
                                              else "the node labels tell which"))
            if cache is None:
                entry.update(skeleton="not checked", existing=[], existing_text=note)
                out[s.name] = entry
                continue
            nodes, env, _ = max(mine.values(), key=lambda hit: hit[0], default=(0, None, False))
            if not pulled:
                status, text = "dump", NOT_PULLED
            elif k.reason:
                status, text = "unknown", f"? ({k.reason})"
            elif nodes:
                status, text = "restore" if nodes == cache.nodes else "partial", f"exists: {nodes}/{cache.nodes} nodes"
            elif any(doubtful for _, _, doubtful in mine.values()):
                status, text = "unknown", f"? (labels of {k.slug} are cut too short to confirm this image's skeleton)"
            else:
                status, text = "dump", "none (will dump)"
            counts[status] += 1
            entry.update(skeleton=status, existing_text=text,
                         existing=[{"env": env, "nodes": nodes, "of": cache.nodes, "match": "exists"}]
                         if nodes else [])
            out[s.name] = entry
        plan["node_cache"] = {"checked": cache is not None, "restricted": True, "note": note,
                              "kube_context": self.kube_context, "nodes": cache.nodes if cache else None,
                              "pulled": pulled, "summary": counts if cache else None}
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
    vis = plan.get("visibility")
    if vis:
        lines.append(f"visibility  {vis['reason']}" + ("; show_all=true (admin override)" if vis["show_all"] else ""))
    cache = plan.get("node_cache")
    if cache:
        lines.append(f"env         {env_text_line(plan['env'])}")
        lines.append("node cache  " + (cache["note"]
                                       or f"{cache['nodes']} model-cache node(s) on {cache['kube_context']}"))
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
        archs = plan.get("architectures") or {}
        if archs:
            lines.append("reasoners of jobs that name no reasoning_parser, by the model's architecture:")
            lines += [f"  {m}: " + (f"{a['architecture']} ({a['source']}) -> {a['reasoning_parser'] or 'none'}"
                                    if a["architecture"] else f"not known ({a['note']})") for m, a in archs.items()]
        if any(not j["cfg12"] and "cfg12_candidates" in j for j in plan["jobs"]):
            why = ("the node labels did not show which one chat adds for the checkpoint" if not cache.get("restricted")
                   else "no candidate's skeleton was found for this image" if cache["checked"]
                   else "the node labels were not checked")
            lines.append(f"cfg12 ?: the job names no reasoning_parser, its model's architecture is not known and {why} "
                         "(plan.json lists the candidates)")
    if "backend_inventory" in plan:
        lines += ["", *inventory_lines(plan["backend_inventory"])]
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
    if counts.get("other_image"):
        text += f" ({counts['other_image']} of them exist for another image)"
    if counts["unknown"]:
        text += f", {counts['unknown']} unknown"
    if not cache["pulled"]:
        text += " (image not pulled on any node yet, so no skeleton of it is on the cluster)"
    return text


def inventory_lines(inv: dict) -> list[str]:
    """The plan's backend inventory: this image's skeletons and weight sets per model, the totals,
    and how many labels could not be told; or the one line saying why there is none."""
    if not inv["checked"] or not inv["pulled"]:
        return [f"backend inventory: {inv['note']}"]
    envs = len(inv["env12"]) > 1

    def wt(entry: dict) -> str:
        return entry["wt12"] if entry["wt12_exact"] else f"{entry['wt12']}.." if entry["wt12"] else "?"

    lines = [f"backend inventory (env {' or '.join(inv['env12'])}, image {inv['image_tag']})"]
    if inv["models"]:
        lines += ["| model | kind | cfg12 |" + (" env12 |" if envs else "") + " wt12 | nodes | this list |",
                  "|---|---|---|" + ("---|" if envs else "") + "---|---|---|"]
    for m in inv["models"]:
        for sk in m["skeletons"]:
            lines.append(f"| {m['model']} | skeleton | {sk['cfg12']} |" + (f" {sk['env12']} |" if envs else "")
                         + f" {wt(sk)} | {sk['nodes']}/{sk['of']} |" + (" in this list |" if sk["in_list"] else " |"))
        for w in m["weights"]:
            nodes = f"{w['nodes']}/{w['of']}" if w["nodes"] is not None else "?"
            lines.append(f"| {m['model']} | weights | - |" + (" - |" if envs else "") + f" {wt(w)} | {nodes} | |")
    t, u = inv["totals"], inv["unattributed"]
    lines.append(f"backend inventory: {t['skeletons']} skeletons across {t['models']} models "
                 f"({t['on_all_nodes']} on all nodes), {t['weights']} weight sets")
    if u["labels"]:
        lines.append(f"{u['labels']} skeleton label(s) of models {', '.join(u['models'])} could not be attributed to "
                     "an image (labels cut" + (f"; {u['kind_unknown']} of them are cut before saying skeleton or "
                                               "weight, so they may be weight labels)" if u["kind_unknown"] else ")"))
    return lines


def env_text_line(env: dict) -> str:
    """The plan header's ``env`` line: this image's env12(s) and where their inputs came from."""
    if not env.get("derived"):
        return f"not derived ({env['note'].removeprefix('not checked: ')})"
    if not env["pulled"]:
        return f"none: {env['image_tag']} is not pulled on any node yet, so no skeleton of it can be on the cluster"
    how = {"whole": "a skeleton label carries it", "start": "a cut skeleton label shows its start",
           None: "no skeleton label shows it"}
    source = "DEFAULT_DRIVER_VERSION (assumed)" if env["driver_assumed"] else env["driver_source"]
    parts = [f"{e} (digest {c['digest'][:12]} from node images, driver {c['driver']} from {source}; "
             f"{how[c['confirmed']]})" for e, c in env["candidates"].items()]
    return parts[0] if len(parts) == 1 else f"ambiguous, any of {len(parts)}: " + "; ".join(parts)


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
