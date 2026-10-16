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

"""Node state a multi-node allocation must be in before a job is placed on it.

DeepSpeed JIT-compiles ops such as ``fused_adam`` on first use, inside the torch extensions cache, under a file
lock that every other process on the node waits on. A build killed after compiling and before linking leaves
that lock and no shared object; from then on every DeepSpeed worker on the node waits forever, and the workers
on the other nodes sit in their first collective until the training-zone init timeout. This module finds such a
directory on each node of the allocation, so the run stops with its path instead of hanging.
"""

from __future__ import annotations

import base64
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

MARKER = "INTERRUPTED_EXTENSION_BUILD"


def extensions_root() -> str:
    """The torch extensions cache the DeepSpeed workers build into, resolved the way torch resolves it."""
    from torch.utils.cpp_extension import get_default_build_root

    return os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()


def node_script(root: str) -> str:
    """Shell that prints one marker line per build directory holding a lock and no shared object.

    A lock beside a compiler that is still running belongs to a build in progress, not to an interrupted one,
    so nothing is reported while ``ninja``, ``nvcc`` or ``cicc`` runs on the node.
    """
    return (
        f"root={shlex.quote(root)}\n"
        "pgrep -x 'ninja|nvcc|cicc' >/dev/null && exit 0\n"
        'for lock in "$root"/*/*/lock; do\n'
        '  [ -e "$lock" ] || continue\n'
        '  d=$(dirname "$lock")\n'
        '  ls "$d"/*.so >/dev/null 2>&1 && continue\n'
        f'  echo "{MARKER} $(hostname) $d"\n'
        "done\n"
    )


def parse(output: str) -> List[Tuple[str, str]]:
    """``(host, directory)`` for every marker line, whatever prefix the remote shell put in front of it."""
    found = []
    for line in output.splitlines():
        _, sep, rest = line.partition(MARKER + " ")
        if sep:
            host, _, directory = rest.strip().partition(" ")
            found.append((host, directory))
    return found


def _ds_ssh() -> str:
    beside = Path(sys.executable).with_name("ds_ssh")
    return str(beside) if beside.exists() else (shutil.which("ds_ssh") or "ds_ssh")


def ensure_no_interrupted_builds(hostfile: Optional[str] = None) -> None:
    """Stop the run when a node of the named multi-node allocation holds an interrupted extension build.

    Applies only when ``DSS_GATEWAY_URL`` and ``DSS_GATEWAY_HOSTFILE`` name a gateway over more than one node;
    a gateway the harness starts itself serves this node alone. The script is sent base64-encoded because
    ``ds_ssh`` passes its arguments to ``pdsh`` unquoted, which would expand the globs on this node.
    """
    if hostfile is None:
        if not os.environ.get("DSS_GATEWAY_URL"):
            return
        hostfile = os.environ.get("DSS_GATEWAY_HOSTFILE")
    if not hostfile or not Path(hostfile).exists():
        return
    from arctic_platform.correctness.harness.hostfile import parse_hostfile

    if len(parse_hostfile(hostfile)) < 2:
        return
    encoded = base64.b64encode(node_script(extensions_root()).encode()).decode()
    done = subprocess.run(
        [_ds_ssh(), "-f", hostfile, f"echo {encoded} | base64 -d | bash"], capture_output=True, text=True, timeout=120
    )
    if done.returncode != 0:
        raise SystemExit(
            f"could not inspect the torch extensions cache on the nodes of {hostfile} "
            f"(ds_ssh exit {done.returncode}): {(done.stdout + done.stderr).strip()[-2000:]}"
        )
    found = parse(done.stdout)
    if found:
        listed = "\n".join(f"  {host}: {directory}" for host, directory in found)
        raise SystemExit(
            "interrupted torch extension build(s): each directory below holds a lock and no shared object, and "
            "every DeepSpeed worker on that node waits on the lock indefinitely. Remove the directory, or rerun "
            "the provisioning script on that node, which rebuilds fused_adam:\n"
            + listed
        )
