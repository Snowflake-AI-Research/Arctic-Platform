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

"""Machine and software identity of a run.

A gradient comparison is only meaningful against the stack that produced it: a kernel version, a CUDA
build, or a different GPU moves the last digits. This module answers "what produced these numbers", and
nothing here interprets or renders them.
"""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict
from typing import Optional

_PACKAGES = ("torch", "transformers", "deepspeed", "accelerate", "flash_attn", "flash_attn_3")


def _version(name: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _git(repo: Path) -> Dict[str, str]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = run("rev-parse", "--short", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    dirty = run("status", "--porcelain")
    info = {}
    if commit:
        info["commit"] = commit + (" (uncommitted changes present)" if dirty else "")
    if branch:
        info["branch"] = branch
    return info


def _gpus() -> Dict[str, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {"gpus": f"{count} x {name} ({total:.0f} GiB each)", "cuda": torch.version.cuda or "unknown"}
    except Exception:
        return {}


def collect(repo: Optional[Path] = None) -> Dict[str, str]:
    """Everything needed to reproduce a run, on a best-effort basis: a missing fact is omitted, never fatal."""
    repo = repo or Path(__file__).resolve().parents[2]
    info: Dict[str, str] = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }
    info.update(_gpus())
    info.update(_git(repo))
    versions = {name: v for name in _PACKAGES if (v := _version(name))}
    if versions:
        info["packages"] = ", ".join(f"{k} {v}" for k, v in versions.items())
    return info
