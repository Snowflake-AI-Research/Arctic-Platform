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

"""Wait until a published skeleton and its weights are verified on every node.

Publishing ends with the upload; the neutrino-model-cache DaemonSet then copies
each directory to every node and stamps ``model.neutrino.snowflake.com/<dir>=ready``
on a node once it has digest-checked every file there. A restore refuses a
directory its node has not verified, so a test waits for the labels first.
The label rules and the node pools are semip_publish.py's; the labels are read
with ``kubectl``, so this needs a kubeconfig for the cluster.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PUBLISHER = Path(__file__).resolve().parent.parent / "scripts" / "semip_publish.py"


def _publisher() -> Any:
    spec = importlib.util.spec_from_file_location("semip_publish", _PUBLISHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def published_dirs(summary_path: Path, model: str) -> list[str]:
    """The skeleton and weight directories (under the bucket root) that publish
    recorded for ``model``, or ``[]`` when it recorded none."""
    try:
        entry = json.loads(Path(summary_path).read_text()).get(model) or {}
    except (OSError, ValueError):
        return []
    if not (entry.get("slug") and entry.get("skeleton") and entry.get("weight_hash")):
        return []
    pub = _publisher()
    root = f"{pub.DEFAULT_PREFIX}/{entry['slug']}"
    return [f"{root}/{pub.SKELETON_DIR}/{entry['skeleton']}", f"{root}/{pub.WEIGHT_DIR}/{entry['weight_hash']}"]


def node_counts(dirs: list[str], run: Callable = subprocess.run) -> tuple[dict[str, int], int]:
    """``({dir: nodes that verified it}, cache nodes)``."""
    pub = _publisher()
    pools = f"workergroup in ({','.join(pub._CACHE_NODE_POOLS)})"

    def count(selector: str) -> int:
        out = run(["kubectl", "--request-timeout=30s", "get", "nodes", "-l", selector, "-o", "name"],
                  capture_output=True, text=True, check=True).stdout
        return len(out.split())

    total = count(pools)
    ready = {d: count(f"{pools},{pub._NODE_LABEL_PREFIX}{pub.sanitize_label_segment(d)}=ready") for d in dirs}
    return ready, total


def wait_verified(dirs: list[str], timeout_s: float, name: str, poll_s: float = 20.0,
                  run: Callable = subprocess.run, sleep: Callable = time.sleep,
                  clock: Callable = time.monotonic) -> str:
    """Block until every directory is verified on every cache node; return a note
    for the report. Raises when ``timeout_s`` passes first."""
    if not dirs:
        return "not checked: the publish summary records no skeleton for this config and image"
    if run is subprocess.run and not shutil.which("kubectl"):
        return "not checked: kubectl is not available"
    deadline = clock() + timeout_s
    last = None
    while True:
        try:
            ready, total = node_counts(dirs, run)
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"not checked: kubectl failed ({exc})"
        if total and all(n == total for n in ready.values()):
            return f"skeleton and weights verified on {total}/{total} nodes"
        counts = ", ".join(f"{d.rsplit('/', 2)[-2]} {n}/{total}" for d, n in ready.items())
        if clock() > deadline:
            raise RuntimeError(f"published image not verified on every node after {timeout_s:.0f}s ({counts})")
        if counts != last:
            logger.info("%s: waiting for the node cache to verify the published image: %s", name, counts)
            last = counts
        sleep(poll_s)
