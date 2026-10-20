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


"Fast-storage paths, allocation hostfiles, and interrupted-build checks."

from __future__ import annotations

import base64
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

CORRECTNESS_TMPDIR_ENV = "DSS_CORRECTNESS_TMPDIR"
_DATA_FAST_ROOT = Path("/data-fast")
_DATA_FAST_TMPDIR = Path("/data-fast/tmp")


def require_data_fast_path(path: str | Path, *, label: str = "correctness storage path") -> Path:
    """Return a canonical fast-storage descendant without creating any directories."""
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        raise ValueError(f"{label} must resolve under /data-fast") from None
    if resolved == _DATA_FAST_ROOT or not resolved.is_relative_to(_DATA_FAST_ROOT):
        raise ValueError(f"{label} must be a resolved descendant of /data-fast")
    if not _DATA_FAST_ROOT.is_dir():
        raise ValueError(f"{label} requires an existing /data-fast directory")
    return resolved


def correctness_workdir(prefix: str) -> Path:
    """Create a heavyweight correctness workdir exclusively on fast storage."""
    if not isinstance(prefix, str) or os.sep in prefix or (os.altsep and os.altsep in prefix):
        raise ValueError("correctness workdir prefix must not contain a directory path")
    override = os.environ.get(CORRECTNESS_TMPDIR_ENV)
    root = require_data_fast_path(
        override if override else _DATA_FAST_TMPDIR,
        label=CORRECTNESS_TMPDIR_ENV if override else "correctness workdir",
    )
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


@dataclass(frozen=True)
class HostfileEntry:
    host: str
    slots: int


def parse_hostfile(path: str | Path) -> list[HostfileEntry]:
    entries: list[HostfileEntry] = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        host = fields[0]
        slots = 1
        for field in fields[1:]:
            if field.startswith("slots="):
                slots = int(field.partition("=")[2])
        if slots < 1:
            raise ValueError(f"{path}:{number}: slots must be positive, got {slots}")
        entries.append(HostfileEntry(host=host, slots=slots))
    return entries


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
