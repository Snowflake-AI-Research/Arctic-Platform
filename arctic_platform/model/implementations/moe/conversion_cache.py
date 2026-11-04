import fcntl
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Callable

from .logging_utils import get_logger

logger = get_logger()

WEIGHT_CONVERSION_CACHE_DIR_ENV = "DSS_WEIGHT_CONVERSION_CACHE_DIR"
WEIGHT_CONVERSION_CACHE_SCOPE_ENV = "DSS_WEIGHT_CONVERSION_CACHE_SCOPE"
WEIGHT_CONVERSION_ATTEMPT_ID_ENV = "DSS_WEIGHT_CONVERSION_ATTEMPT_ID"
DEFAULT_WEIGHT_CONVERSION_CACHE_DIR = "/data-fast/prime-rl-weight-cache"
_CACHE_HEARTBEAT_INTERVAL = 1
_ABANDONED_CACHE_AGE = 24 * 3600


def _preparation_marker_path(attempt_id: str) -> Path:
    token = hashlib.sha256(attempt_id.encode()).hexdigest()
    return Path(resolve_conversion_cache_root()) / f".preparation-{token}"


def record_conversion_preparation_failure(function):
    """Report pre-cache setup failures using the launcher's common attempt identity."""
    parameters = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        attempt = os.environ.get(WEIGHT_CONVERSION_ATTEMPT_ID_ENV)
        arguments = parameters.bind_partial(*args, **kwargs).arguments
        rank = arguments.get("rank")
        if rank is None:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
        if not attempt or rank != 0:
            return function(*args, **kwargs)
        try:
            _preparation_marker_path(attempt).with_suffix(".failed").unlink(missing_ok=True)
            return function(*args, **kwargs)
        except BaseException as error:
            try:
                path = _preparation_marker_path(attempt)
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_cache_marker(path.with_suffix(".failed"), dict(attempt_id=attempt, error=str(error)))
            except Exception:
                logger.warning("Could not record weight preparation failure", exc_info=True)
            raise

    return wrapped


@contextmanager
def _conversion_writer_lock(path: Path, *, node_local: bool):
    # Shared scope has exactly one designated writer and must not require cross-node flock support.
    if not node_local:
        yield
        return
    with open(path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _cleanup_abandoned_conversion_workspaces(path: Path) -> None:
    now = time.time()
    for workspace in path.parent.glob(f"{path.name}.tmp-rank*-pid*-*"):
        try:
            if not workspace.is_dir():
                continue
            owner = workspace.with_name(f"{workspace.name}.owner")
            timestamp = owner.stat().st_mtime if owner.exists() else workspace.stat().st_mtime
            if now - timestamp > _ABANDONED_CACHE_AGE:
                shutil.rmtree(workspace)
                owner.unlink(missing_ok=True)
        except OSError:
            logger.info("Could not clean abandoned conversion workspace %s", workspace, exc_info=True)


def get_checkpoint_converter(model, fmt: str) -> Callable[[Path, Path], None] | None:
    converter = getattr(model, f"stream_convert_to_{fmt}", None)
    if converter is None:
        converter = getattr(model, f"convert_to_{fmt}_checkpoint", None)
    return converter


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


def _read_cache_marker(path: Path) -> dict:
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        return marker if isinstance(marker, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache_marker(path: Path, marker: dict) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(marker), encoding="utf-8", errors="replace")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.info("Could not remove temporary conversion marker %s", temporary, exc_info=True)


def conversion_cache_attempt_id(*, shared: bool = False) -> str | None:
    """Use the launcher's retry identity, or agree on one for a distributed load."""
    token = os.environ.get(WEIGHT_CONVERSION_ATTEMPT_ID_ENV)
    if token:
        return token
    import torch.distributed as dist

    if not dist.is_initialized():
        if shared:
            raise ValueError(
                f"Standalone shared-cache preparation requires a common {WEIGHT_CONVERSION_ATTEMPT_ID_ENV} "
                "on every process; choose a new value for each retry"
            )
        return None
    tokens = [uuid.uuid4().hex if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(tokens, src=0)
    return tokens[0]


def wait_for_conversion_cache(
    path: Path, *, timeout: float | None = None, poll_interval: float = 0.25,
    writer_idle_timeout: float = 120, attempt_id: str | None = None,
    writer_start_timeout: float | None = None,
) -> None:
    """Observe attempt-scoped progress without shared-lock coherence or synchronized clocks.

    Without an explicit attempt ID, an already failed attempt is ignored until a new writer starts. A launcher
    may supply a common attempt ID to recognize failures even when a waiter starts after the writer exits.
    Startup is separately bounded (300 seconds by default, configurable for slow independent launchers).
    Once observed, unchanged heartbeats bound killed-writer waits.
    """
    attempt_id = attempt_id or os.environ.get(WEIGHT_CONVERSION_ATTEMPT_ID_ENV)
    timeout = float(os.environ.get("DSS_WEIGHT_CONVERSION_TIMEOUT", "3600")) if timeout is None else timeout
    if writer_start_timeout is None:
        writer_start_timeout = float(os.environ.get("DSS_WEIGHT_CONVERSION_WRITER_START_TIMEOUT", "300"))
    if timeout < 0 or poll_interval <= 0 or writer_idle_timeout <= 0:
        raise ValueError("cache timeout must be nonnegative and polling/idle intervals must be positive")
    if writer_start_timeout <= 0:
        raise ValueError("writer startup timeout must be positive")
    started = time.monotonic()
    deadline = time.monotonic() + timeout
    failure_path = path.with_name(f"{path.name}.failed")
    progress_path = path.with_name(f"{path.name}.in-progress")
    old_failure = _read_cache_marker(failure_path).get("attempt_id") if attempt_id is None else None
    observed = None
    last_progress = time.monotonic()
    while not conversion_cache_ready(path):
        if attempt_id is not None:
            setup_failure = _read_cache_marker(_preparation_marker_path(attempt_id).with_suffix(".failed"))
            if setup_failure.get("attempt_id") == attempt_id:
                raise RuntimeError(f"Weight preparation writer failed: {setup_failure.get('error')}")
        failure = _read_cache_marker(failure_path)
        if attempt_id is not None and failure.get("attempt_id") == attempt_id:
            raise RuntimeError(f"Converted weight cache writer failed for {path}: {failure.get('error')}")
        progress = _read_cache_marker(progress_path)
        token = progress.get("attempt_id")
        if token is not None and token != old_failure and (attempt_id is None or token == attempt_id):
            current = (token, progress.get("heartbeat"))
            if current != observed:
                observed = current
                last_progress = time.monotonic()
            failure = _read_cache_marker(failure_path)
            if failure.get("attempt_id") == token:
                raise RuntimeError(f"Converted weight cache writer failed for {path}: {failure.get('error')}")
            if time.monotonic() - last_progress >= writer_idle_timeout:
                raise RuntimeError(f"Converted weight cache writer stopped heartbeating without publishing: {path}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Converted weight cache did not become ready within {timeout}s: {path}")
        if observed is None and time.monotonic() - started >= writer_start_timeout:
            raise TimeoutError(
                f"Converted weight cache writer did not publish startup progress within {writer_start_timeout}s: {path}"
            )
        time.sleep(min(poll_interval, remaining))


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
    stream_convert_fn: Callable[[Path, Path], None] | None = None,
) -> None:
    # A unique workspace avoids pid collisions across nodes and never reclaims another writer's live files.
    tmp_path = snapshot_path.with_name(f"{snapshot_path.name}.tmp-rank{rank}-pid{os.getpid()}-{uuid.uuid4().hex}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    owner_path = tmp_path.with_name(f"{tmp_path.name}.owner")
    owner_path.touch()
    stop = threading.Event()

    def refresh_owner():
        while not stop.wait(_CACHE_HEARTBEAT_INTERVAL):
            try:
                owner_path.touch()
            except OSError:
                logger.warning("Could not refresh conversion workspace ownership for %s", tmp_path, exc_info=True)

    owner_thread = threading.Thread(target=refresh_owner, daemon=True)
    owner_thread.start()

    def publish():
        if conversion_cache_ready(snapshot_path):
            shutil.rmtree(tmp_path)
            return
        # Do not delete an existing destination: another node may have published since the readiness check.
        os.replace(tmp_path, snapshot_path)

    def finish():
        stop.set()
        owner_thread.join()
        owner_path.unlink(missing_ok=True)
        if not committed and tmp_path.exists():
            shutil.rmtree(tmp_path)

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
    if stream_convert_fn is not None:
        try:
            stream_convert_fn(source_path, tmp_path)
            if not conversion_cache_ready(tmp_path):
                raise RuntimeError(f"Weight conversion did not publish a complete checkpoint: {tmp_path}")
            publish()
            committed = True
        finally:
            finish()
        logger.info(
            "Streamed %s→%s cache written to %s in %.1fs rank=%d local_rank=%d",
            src_fmt,
            dst_fmt,
            snapshot_path,
            time.perf_counter() - t0,
            rank,
            local_rank,
        )
        return
    try:
        snapshot_state_dict = load_state_dict_fn(source_path)
        t_load = time.perf_counter()
        # Bulk converters mutate the supplied state in place; preserve that existing contract.
        convert_fn(snapshot_state_dict)
        t_convert = time.perf_counter()
        save_state_dict_fn(snapshot_state_dict, tmp_path)
        t_save = time.perf_counter()
        del snapshot_state_dict
        if not conversion_cache_ready(tmp_path):
            raise RuntimeError(f"Weight conversion did not publish a complete checkpoint: {tmp_path}")
        publish()
        committed = True
    finally:
        finish()
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
    stream_convert_fn: Callable[[Path, Path], None] | None = None,
    attempt_id: str | None = None,
    node_local: bool | None = None,
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

    attempt_id = attempt_id or os.environ.get(WEIGHT_CONVERSION_ATTEMPT_ID_ENV)
    failure_path = snapshot_path.with_name(f"{snapshot_path.name}.failed")
    progress_path = snapshot_path.with_name(f"{snapshot_path.name}.in-progress")
    token = attempt_id or uuid.uuid4().hex
    writer_id = uuid.uuid4().hex
    node_local = conversion_cache_is_node_local(snapshot_path) if node_local is None else node_local
    failure_recorded = False

    def record_failure(error):
        nonlocal failure_recorded
        failure_recorded = True
        try:
            _write_cache_marker(
                failure_path, dict(
                    attempt_id=token, writer_id=writer_id, error=f"rank={rank}: {type(error).__name__}: {error}"
                ),
            )
        except Exception:
            logger.warning("Could not record conversion failure for %s", snapshot_path, exc_info=True)

    initial_failure = _read_cache_marker(failure_path)
    initial_writer = _read_cache_marker(progress_path).get("attempt_id")
    lock_path = snapshot_path.parent / f".{snapshot_path.name}.lock"
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with _conversion_writer_lock(lock_path, node_local=node_local):
            if conversion_cache_ready(snapshot_path):
                return
            failure = _read_cache_marker(failure_path)
            if failure.get("attempt_id") is not None and (
                failure != initial_failure
                or (not initial_failure and failure["attempt_id"] == initial_writer)
                or (attempt_id is not None and failure["attempt_id"] == attempt_id and local_rank != 0)
            ):
                failure_recorded = True
                raise RuntimeError(f"Converted weight cache writer failed for {snapshot_path}: {failure.get('error')}")
            failure_path.unlink(missing_ok=True)
            if attempt_id is not None:
                _preparation_marker_path(attempt_id).with_suffix(".failed").unlink(missing_ok=True)
            _cleanup_abandoned_conversion_workspaces(snapshot_path)
            stop = threading.Event()
            thread = None
            try:
                _write_cache_marker(progress_path, dict(attempt_id=token, writer_id=writer_id, heartbeat=0))

                def heartbeat():
                    count = 0
                    while not stop.wait(_CACHE_HEARTBEAT_INTERVAL):
                        count += 1
                        try:
                            _write_cache_marker(
                                progress_path, dict(attempt_id=token, writer_id=writer_id, heartbeat=count)
                            )
                        except Exception:
                            logger.warning(
                                "Could not update conversion heartbeat for %s; retrying", snapshot_path, exc_info=True,
                            )

                thread = threading.Thread(target=heartbeat, daemon=True)
                thread.start()
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
                    stream_convert_fn=stream_convert_fn,
                )
            except BaseException as error:
                # Publish failure before releasing the lock, so a waiting local rank cannot retry it.
                record_failure(error)
                raise
            finally:
                stop.set()
                if thread is not None and thread.ident is not None:
                    thread.join()
                try:
                    if _read_cache_marker(progress_path).get("writer_id") == writer_id:
                        progress_path.unlink(missing_ok=True)
                except OSError:
                    logger.info("Could not remove conversion progress marker for %s", snapshot_path, exc_info=True)
    except BaseException as error:
        if not failure_recorded:
            record_failure(error)
        raise
