"""Scoped lifecycle helpers for vLLM routed-experts shared memory.

vLLM's ``RoutedExpertsCapturer`` stores per-token router decisions in POSIX
shared memory so the scheduler/output side can read them from a different
process. Normal capturer destruction unlinks the segment, but Ray actor death
or SIGKILL can bypass destructors. This module keeps a small per-host registry
of the exact shm objects an ArcticInference worker expects to own, so startup
and shutdown cleanup can be scoped to the current zone/model instead of
wildcard-deleting unrelated jobs.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any

_BUFFER_PREFIX = "vllm_routed_experts_buffer"
_LOCK_PREFIX = "vllm_routed_experts"
_DEFAULT_REGISTRY_ROOT = "/tmp/arctic-router-replay-shm"
_DEFAULT_SHM_DIR = "/dev/shm"
_DEFAULT_LOCK_DIR = "/tmp"


def current_scope(scope: str | None = None) -> str:
    return str(scope or os.environ.get("ARCTIC_ROUTER_REPLAY_SHM_SCOPE") or "default")


def _safe_component(value: str | None) -> str:
    text = str(value or "default")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("._")
    return text or "default"


def _registry_root() -> Path:
    return Path(os.environ.get("ARCTIC_ROUTER_REPLAY_SHM_REGISTRY_DIR", _DEFAULT_REGISTRY_ROOT))


def _shm_dir() -> Path:
    return Path(os.environ.get("ARCTIC_ROUTER_REPLAY_SHM_DIR", _DEFAULT_SHM_DIR))


def _lock_dir() -> Path:
    return Path(os.environ.get("ARCTIC_ROUTER_REPLAY_SHM_LOCK_DIR", _DEFAULT_LOCK_DIR))


def expected_names(instance_id: str | int, dp_rank: int = 0) -> tuple[str, str]:
    instance = str(instance_id)
    rank = int(dp_rank)
    return (
        f"{_BUFFER_PREFIX}_{instance}_{rank}",
        str(_lock_dir() / f"{_LOCK_PREFIX}_{instance}_{rank}.lock"),
    )


def _scope_dir(scope: str, model_id: str | None) -> Path:
    return _registry_root() / _safe_component(scope) / _safe_component(model_id)


def _entry_path(scope: str, model_id: str | None, pid: int, instance_id: str | int, dp_rank: int) -> Path:
    name = f"{int(pid)}-{_safe_component(str(instance_id))}-{int(dp_rank)}.json"
    return _scope_dir(scope, model_id) / name


def register_expected_buffer(
    *,
    scope: str | None,
    model_id: str | None,
    instance_id: str | int,
    dp_rank: int,
    pid: int | None = None,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    """Register the exact vLLM routed-experts shm names this worker may create."""
    resolved_scope = current_scope(scope)
    resolved_pid = int(pid or os.getpid())
    shm_name, lock_file = expected_names(instance_id, dp_rank)
    path = _entry_path(resolved_scope, model_id, resolved_pid, instance_id, dp_rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "scope": resolved_scope,
        "model_id": str(model_id or "default"),
        "pid": resolved_pid,
        "host": socket.gethostname(),
        "created_at": time.time(),
        "instance_id": str(instance_id),
        "dp_rank": int(dp_rank),
        "shm_name": shm_name,
        "shm_path": str(_shm_dir() / shm_name),
        "lock_file": lock_file,
        "size_bytes": size_bytes,
        "registry_path": str(path),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return entry


def _pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _unlink(path: str | None) -> bool:
    if not path:
        return False
    try:
        Path(path).unlink()
        return True
    except FileNotFoundError:
        return False
    except PermissionError:
        return False


def cleanup_entry(entry: dict[str, Any], *, stale_only: bool = False) -> dict[str, Any]:
    """Unlink one registered shm/lock pair and remove its registry entry."""
    pid = int(entry.get("pid") or -1)
    if stale_only and _pid_alive(pid):
        return {"status": "skipped_alive", "pid": pid, "removed": 0, "entry": entry}

    removed = 0
    removed += int(_unlink(entry.get("shm_path")))
    removed += int(_unlink(entry.get("lock_file")))
    removed += int(_unlink(entry.get("registry_path")))
    return {"status": "removed", "pid": pid, "removed": removed, "entry": entry}


def _iter_entries(scope: str | None, model_id: str | None = None):
    root = _registry_root()
    if not root.exists():
        return
    scope_part = _safe_component(current_scope(scope))
    scope_root = root / scope_part
    if model_id is None:
        model_dirs = [p for p in scope_root.iterdir() if p.is_dir()] if scope_root.exists() else []
    else:
        model_dirs = [scope_root / _safe_component(model_id)]
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        for path in model_dir.glob("*.json"):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entry.setdefault("registry_path", str(path))
            yield entry


def cleanup_scope(
    *,
    scope: str | None,
    model_id: str | None = None,
    stale_only: bool = False,
) -> dict[str, Any]:
    results = [
        cleanup_entry(entry, stale_only=stale_only)
        for entry in list(_iter_entries(scope, model_id) or [])
    ]
    return {
        "status": "ok",
        "scope": current_scope(scope),
        "model_id": str(model_id or "*"),
        "entries": len(results),
        "removed": sum(int(result.get("removed", 0)) for result in results),
        "results": results,
    }
