import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

import fcntl

logger = logging.getLogger(__name__)

WEIGHT_CONVERSION_CACHE_DIR_ENV = "DSS_WEIGHT_CONVERSION_CACHE_DIR"
WEIGHT_CONVERSION_CACHE_SCOPE_ENV = "DSS_WEIGHT_CONVERSION_CACHE_SCOPE"


def resolve_conversion_cache_path(config, source_path: Path, fmt: str) -> Path:
    cache_root = getattr(config, "weight_conversion_cache_dir", None) or os.environ.get(
        WEIGHT_CONVERSION_CACHE_DIR_ENV
    )
    if not cache_root:
        return source_path / fmt
    source_key = hashlib.sha1(str(source_path.resolve()).encode()).hexdigest()[:12]
    return Path(cache_root) / f"{source_path.name}-{source_key}" / fmt


def conversion_cache_ready(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    return all((path / str(shard)).is_file() for shard in set(weight_map.values()))


def conversion_cache_is_node_local(path: Path) -> bool:
    scope = os.environ.get(WEIGHT_CONVERSION_CACHE_SCOPE_ENV, "auto").lower()
    if scope in {"node", "node-local", "local"}:
        return True
    if scope in {"global", "shared"}:
        return False

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    node_local_roots = (Path("/data-fast"), Path("/tmp"))
    return any(resolved == root or root in resolved.parents for root in node_local_roots)


def _write_conversion_cache(
    source_path: Path,
    snapshot_path: Path,
    convert_fn: Callable,
    src_fmt: str,
    dst_fmt: str,
    load_state_dict_fn: Callable,
    save_state_dict_fn: Callable,
    *,
    rank: int,
    local_rank: int,
) -> None:
    tmp_path = snapshot_path.with_name(f"{snapshot_path.name}.tmp-rank{rank}-pid{os.getpid()}")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)

    logger.info(
        "Converting snapshot state dict from %s to %s and saving to %s on rank=%d local_rank=%d. "
        "This is a one-time operation.",
        src_fmt,
        dst_fmt,
        snapshot_path,
        rank,
        local_rank,
    )
    snapshot_state_dict = load_state_dict_fn(source_path)
    convert_fn(snapshot_state_dict)
    save_state_dict_fn(snapshot_state_dict, tmp_path)
    del snapshot_state_dict
    os.replace(tmp_path, snapshot_path)


def ensure_node_local_conversion_cache(
    source_path: Path,
    snapshot_path: Path,
    convert_fn: Callable,
    src_fmt: str,
    dst_fmt: str,
    load_state_dict_fn: Callable,
    save_state_dict_fn: Callable,
    *,
    rank: int,
    local_rank: int,
) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = snapshot_path.parent / f".{snapshot_path.name}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not conversion_cache_ready(snapshot_path):
                _write_conversion_cache(
                    source_path,
                    snapshot_path,
                    convert_fn,
                    src_fmt,
                    dst_fmt,
                    load_state_dict_fn,
                    save_state_dict_fn,
                    rank=rank,
                    local_rank=local_rank,
                )
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
