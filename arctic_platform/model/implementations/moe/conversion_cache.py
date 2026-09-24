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
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

from .logging_utils import get_logger

logger = get_logger()

WEIGHT_CONVERSION_CACHE_DIR_ENV = "DSS_WEIGHT_CONVERSION_CACHE_DIR"
WEIGHT_CONVERSION_CACHE_SCOPE_ENV = "DSS_WEIGHT_CONVERSION_CACHE_SCOPE"
DEFAULT_WEIGHT_CONVERSION_CACHE_DIR = "/data-fast/prime-rl-weight-cache"


def sibling_conversion_cache_path(source_path: Path, fmt: str) -> Path:
    """PrimeRL (or HF) shards written next to the source checkpoint: ``<source>/prime``."""
    return Path(source_path) / fmt


def hashed_conversion_cache_path(cache_root: str | Path, source_path: Path, fmt: str) -> Path:
    source_key = hashlib.sha1(str(Path(source_path).resolve()).encode()).hexdigest()[:12]
    return Path(cache_root) / f"{Path(source_path).name}-{source_key}" / fmt


def resolve_conversion_cache_root(override: str | Path | None = None) -> str | Path:
    """Resolve an explicit cache root before the environment and default."""
    if override is not None:
        return override
    return os.environ.get(WEIGHT_CONVERSION_CACHE_DIR_ENV) or DEFAULT_WEIGHT_CONVERSION_CACHE_DIR


def resolve_conversion_cache_path(config, source_path: Path, fmt: str) -> Path:
    sibling = sibling_conversion_cache_path(source_path, fmt)
    # A pre-baked sibling cache (offline convert_hf_to_primerl) wins over the
    # ephemeral /data-fast hashed dir so QA6/prod can ship converted weights
    # with the HF checkpoint.
    if conversion_cache_ready(sibling):
        return sibling

    cache_root = resolve_conversion_cache_root(getattr(config, "weight_conversion_cache_dir", None))
    if not cache_root:
        return sibling
    return hashed_conversion_cache_path(cache_root, source_path, fmt)


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
    committed = False
    t0 = time.perf_counter()
    try:
        snapshot_state_dict = load_state_dict_fn(source_path)
        t_load = time.perf_counter()
        convert_fn(snapshot_state_dict)
        t_convert = time.perf_counter()
        save_state_dict_fn(snapshot_state_dict, tmp_path)
        t_save = time.perf_counter()
        del snapshot_state_dict
        os.replace(tmp_path, snapshot_path)
        committed = True
    finally:
        if not committed and tmp_path.exists():
            shutil.rmtree(tmp_path)
    t_done = time.perf_counter()
    logger.info(
        "Converted %s→%s cache written to %s in %.1fs "
        "(load=%.1fs convert=%.1fs save=%.1fs rename=%.1fs) rank=%d local_rank=%d",
        src_fmt,
        dst_fmt,
        snapshot_path,
        t_done - t0,
        t_load - t0,
        t_convert - t_load,
        t_save - t_convert,
        t_done - t_save,
        rank,
        local_rank,
    )


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
    if conversion_cache_ready(snapshot_path):
        if local_rank == 0:
            logger.info(
                "Reusing existing %s→%s conversion cache at %s (rank=%d)",
                src_fmt,
                dst_fmt,
                snapshot_path,
                rank,
            )
        return

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
            elif local_rank == 0:
                logger.info(
                    "Reusing existing %s→%s conversion cache at %s (rank=%d)",
                    src_fmt,
                    dst_fmt,
                    snapshot_path,
                    rank,
                )
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
