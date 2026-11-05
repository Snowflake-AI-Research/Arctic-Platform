import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
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
_CACHE_HEARTBEAT_JOIN_TIMEOUT = 2
_ABANDONED_CACHE_AGE = 60
_ACTIVE_ATTEMPT = ContextVar("weight_conversion_attempt", default=None)
_PREPARATION_START = ContextVar("weight_conversion_preparation_start", default=None)
_monotonic = time.monotonic
_sleep = time.sleep
_replace = os.replace
_rename = os.rename
_flock = fcntl.flock


def _remove_cache_marker(path: Path) -> None:
    path.unlink(missing_ok=True)


def _preparation_marker_path(attempt_id: str, cache_root=None) -> Path:
    token = hashlib.sha256(attempt_id.encode()).hexdigest()
    root = resolve_conversion_cache_root(cache_root) or DEFAULT_WEIGHT_CONVERSION_CACHE_DIR
    return Path(root) / f".preparation-{token}"


def conversion_preparation_root(config, source_path: Path | None = None, *, model_name: str | None = None) -> Path:
    """Keep sibling-cache setup markers on the checkpoint's storage, never in the caller's cwd."""
    root = resolve_conversion_cache_root(getattr(config, "weight_conversion_cache_dir", None))
    if root:
        return Path(root)
    name = getattr(config, "name", model_name)
    if name is not None and Path(name).is_dir():
        return Path(name)
    if name is None and source_path is not None:
        return Path(source_path)
    # A hub source is not resolved until download finishes. Its cache root is stable before and after download.
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def begin_conversion_preparation() -> None:
    """Start setup liveness only after the caller has selected a needed conversion."""
    start = _PREPARATION_START.get()
    if start is not None:
        start()


def record_conversion_preparation_failure(function=None, *, distributed: bool = False, heartbeat: bool = True):
    """Report setup failures without collectives unless distributed=True is explicitly selected.

    Collective loaders must opt in and require every process-group rank to call. They establish the
    attempt before discovery so a rank-zero setup failure can reach waiting peers.
    """
    if function is None:
        return lambda wrapped: record_conversion_preparation_failure(
            wrapped, distributed=distributed, heartbeat=heartbeat
        )
    parameters = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _ACTIVE_ATTEMPT.get() is not None:
            return function(*args, **kwargs)
        arguments = parameters.bind_partial(*args, **kwargs).arguments
        config = arguments.get("config", arguments.get("options"))
        if getattr(getattr(config, "debug", None), "random_init", False):
            return function(*args, **kwargs)
        # Collective loaders agree before checkpoint inspection, model setup or source download can fail.
        attempt, fresh_attempt = conversion_cache_attempt_id(distributed=distributed)
        context = _ACTIVE_ATTEMPT.set((attempt, fresh_attempt))
        marker = None
        stop = threading.Event()
        thread = None
        writer_id = uuid.uuid4().hex
        start_context = None
        try:
            rank = arguments.get("rank")
            if rank is None:
                import torch.distributed as dist

                rank = dist.get_rank() if dist.is_initialized() else 0
            if attempt and rank == 0:
                root = conversion_preparation_root(
                    config, arguments.get("source_path"), model_name=arguments.get("model_name")
                )
                marker = _preparation_marker_path(attempt, root)
            def start_heartbeat():
                nonlocal thread
                if marker is None or thread is not None:
                    return
                try:
                    _remove_cache_marker(marker.with_suffix(".failed"))
                except OSError:
                    logger.info("Could not clear previous weight preparation failure", exc_info=True)

                def refresh():
                    count = 0
                    warned = False
                    while not stop.is_set():
                        try:
                            marker.parent.mkdir(parents=True, exist_ok=True)
                            _write_cache_marker(
                                marker.with_suffix(".in-progress"),
                                dict(attempt_id=attempt, writer_id=writer_id, heartbeat=count),
                            )
                        except Exception:
                            if not warned:
                                logger.warning("Could not record weight preparation liveness; retrying", exc_info=True)
                                warned = True
                        count += 1
                        if stop.wait(_CACHE_HEARTBEAT_INTERVAL):
                            break

                thread = threading.Thread(target=refresh, daemon=True)
                thread.start()
            start_context = _PREPARATION_START.set(start_heartbeat)
            if heartbeat:
                start_heartbeat()
            result = function(*args, **kwargs)
            if marker is not None and thread is not None:
                try:
                    _remove_cache_marker(marker.with_suffix(".failed"))
                except OSError:
                    logger.info("Could not clear weight preparation failure after success", exc_info=True)
            return result
        except BaseException as error:
            if marker is not None:
                try:
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    _write_cache_marker(marker.with_suffix(".failed"), dict(attempt_id=attempt, error=str(error)))
                except Exception:
                    logger.warning("Could not record weight preparation failure", exc_info=True)
            raise
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=_CACHE_HEARTBEAT_JOIN_TIMEOUT)
            if marker is not None and thread is not None and not thread.is_alive():
                try:
                    _remove_cache_marker(marker.with_suffix(".in-progress"))
                except OSError:
                    logger.info("Could not remove weight preparation progress marker", exc_info=True)
            if start_context is not None:
                _PREPARATION_START.reset(start_context)
            _ACTIVE_ATTEMPT.reset(context)

    return wrapped


@contextmanager
def _conversion_writer_lock(path: Path, *, node_local: bool):
    # Shared scope has exactly one designated writer and must not require cross-node flock support.
    if not node_local:
        yield
        return
    with open(path, "w", encoding="utf-8") as lock_file:
        _flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            _flock(lock_file, fcntl.LOCK_UN)


def _workspace_owner() -> dict:
    active = _ACTIVE_ATTEMPT.get()
    return dict(host=socket.gethostname(), pid=os.getpid(), attempt_id=active[0] if active else None)


def _workspace_is_abandoned(artifact: Path, owner: Path, *, node_local: bool) -> bool:
    if node_local:
        # The caller holds the node-local writer lock, so no workspace from another writer can be active.
        return True
    marker = _read_cache_marker(owner)
    if marker.get("host") == socket.gethostname() and type(marker.get("pid")) is int:
        try:
            os.kill(marker["pid"], 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        else:
            return False
    # A remote writer can be alive while filesystem I/O or its heartbeat is stalled.
    # Only durable failure evidence authorizes cross-host deletion; silence and mtimes do not.
    return marker.get("state") == "failed"


def _cleanup_abandoned_conversion_workspaces(path: Path, *, node_local: bool = False) -> None:
    for workspace in path.parent.glob(f"{path.name}.tmp-rank*-pid*"):
        try:
            if not workspace.is_dir() or re.fullmatch(
                re.escape(path.name) + r"\.tmp-rank\d+-pid\d+(?:-[\w-]+)?", workspace.name
            ) is None:
                continue
            owner = workspace.with_name(f"{workspace.name}.owner")
            if _workspace_is_abandoned(workspace, owner, node_local=node_local):
                shutil.rmtree(workspace)
                owner.unlink(missing_ok=True)
        except OSError:
            logger.info("Could not clean abandoned conversion workspace %s", workspace, exc_info=True)
    for artifact in (
        *path.parent.glob(f"{path.name}.tmp-rank*-pid*.owner"),
        *path.parent.glob(f"{path.name}.stale-*"),
    ):
        try:
            if artifact.is_dir() and conversion_cache_ready(artifact):
                continue
            if _workspace_is_abandoned(artifact, artifact, node_local=node_local):
                if artifact.is_dir():
                    shutil.rmtree(artifact)
                elif not artifact.with_suffix("").exists():
                    artifact.unlink(missing_ok=True)
        except OSError:
            logger.info("Could not clean abandoned conversion artifact %s", artifact, exc_info=True)


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
        temporary.write_text(json.dumps(dict(marker, generation=uuid.uuid4().hex)), encoding="utf-8", errors="replace")
        _replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.info("Could not remove temporary conversion marker %s", temporary, exc_info=True)


def conversion_cache_attempt_id(*, shared: bool = False, distributed: bool = True) -> tuple[str | None, bool]:
    """Use the launcher's retry identity, or agree on one for a distributed load."""
    active = _ACTIVE_ATTEMPT.get()
    if active is not None and (active[0] is not None or not shared):
        return active
    token = os.environ.get(WEIGHT_CONVERSION_ATTEMPT_ID_ENV)
    if token:
        return token, False
    import torch.distributed as dist

    if not distributed or active is not None or not dist.is_initialized():
        if shared:
            raise ValueError(
                f"Standalone shared-cache preparation requires a common {WEIGHT_CONVERSION_ATTEMPT_ID_ENV} "
                "on every process; choose a new value for each retry"
            )
        return None, False
    tokens = [uuid.uuid4().hex if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(tokens, src=0)
    return tokens[0], True


def wait_for_conversion_cache(
    path: Path,
    *,
    timeout: float | None = None,
    poll_interval: float = 1,
    writer_idle_timeout: float = 120,
    attempt_id: str | None = None,
    writer_start_timeout: float | None = None,
    fresh_attempt: bool = False,
    preparation_root: str | Path | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Observe attempt-scoped progress without shared-lock coherence or synchronized clocks.

    Existing markers are baseline generations, not evidence of this waiter's launch, even with a reused ID.
    Waiters must start before the writer or observe a subsequent heartbeat/failure generation. A waiter arriving
    after a writer has already exited is bounded by the startup timeout; use fresh IDs for each launcher retry.
    Callers that guarantee a newly generated, never-reused attempt ID may set fresh_attempt to observe terminal
    failures published before the waiter starts.
    Before conversion selection, only the overall timeout applies: independent model setup has no
    filesystem liveness signal. A launcher may explicitly configure a shorter writer-start deadline.
    Preparation heartbeats start after conversion selection, not for native or randomly initialized loads.
    Once observed, unchanged heartbeats bound killed-writer waits.
    """
    attempt_id = attempt_id or os.environ.get(WEIGHT_CONVERSION_ATTEMPT_ID_ENV)
    timeout = float(os.environ.get("DSS_WEIGHT_CONVERSION_TIMEOUT", "3600")) if timeout is None else timeout
    if writer_start_timeout is None:
        configured_start = os.environ.get("DSS_WEIGHT_CONVERSION_WRITER_START_TIMEOUT")
        writer_start_timeout = float(configured_start) if configured_start else None
    if timeout < 0 or poll_interval <= 0 or writer_idle_timeout <= 0:
        raise ValueError("cache timeout must be nonnegative and polling/idle intervals must be positive")
    if writer_start_timeout is not None and writer_start_timeout <= 0:
        raise ValueError("writer startup timeout must be positive")
    clock = _monotonic if clock is None else clock
    sleep = _sleep if sleep is None else sleep
    started = clock()
    deadline = clock() + timeout
    failure_path = path.with_name(f"{path.name}.failed")
    progress_path = path.with_name(f"{path.name}.in-progress")
    initial_failure = {} if fresh_attempt else _read_cache_marker(failure_path)
    initial_progress = {} if fresh_attempt else _read_cache_marker(progress_path)
    preparation_marker = _preparation_marker_path(attempt_id, preparation_root) if attempt_id else None
    preparation_path = preparation_marker.with_suffix(".failed") if preparation_marker else None
    preparation_progress = preparation_marker.with_suffix(".in-progress") if preparation_marker else None
    initial_preparation = _read_cache_marker(preparation_path) if preparation_path and not fresh_attempt else {}
    initial_preparation_progress = (
        _read_cache_marker(preparation_progress) if preparation_progress and not fresh_attempt else {}
    )
    baseline_errors = [
        str(marker["error"])
        for marker in (initial_failure, initial_preparation)
        if attempt_id is not None and marker.get("attempt_id") == attempt_id and marker.get("error")
    ]
    baseline_context = f"; existing same-attempt failure: {'; '.join(baseline_errors)}" if baseline_errors else ""
    observed: dict[str, dict] = {}
    last_progress = clock()
    while not conversion_cache_ready(path):
        if preparation_path is not None and preparation_progress is not None:
            setup_failure = _read_cache_marker(preparation_path)
            if setup_failure != initial_preparation and setup_failure.get("attempt_id") == attempt_id:
                raise RuntimeError(f"Weight preparation writer failed: {setup_failure.get('error')}")
            setup_progress = _read_cache_marker(preparation_progress)
            if (
                setup_progress != initial_preparation_progress
                and setup_progress.get("attempt_id") == attempt_id
                and setup_progress != observed.get("preparation")
            ):
                observed["preparation"] = setup_progress
                last_progress = clock()
        failure = _read_cache_marker(failure_path)
        if (
            failure != initial_failure
            and failure.get("attempt_id") is not None
            and (attempt_id is None or failure.get("attempt_id") == attempt_id)
        ):
            raise RuntimeError(f"Converted weight cache writer failed for {path}: {failure.get('error')}")
        progress = _read_cache_marker(progress_path)
        token = progress.get("attempt_id")
        if token is not None and progress != initial_progress and (attempt_id is None or token == attempt_id):
            if progress != observed.get("conversion"):
                observed["conversion"] = progress
                last_progress = clock()
            failure = _read_cache_marker(failure_path)
            if failure != initial_failure and failure.get("attempt_id") == token:
                raise RuntimeError(f"Converted weight cache writer failed for {path}: {failure.get('error')}")
        if observed and clock() - last_progress >= writer_idle_timeout:
            raise RuntimeError(f"Converted weight cache writer stopped heartbeating without publishing: {path}")
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(f"Converted weight cache did not become ready within {timeout}s: {path}{baseline_context}")
        if writer_start_timeout is not None and not observed and clock() - started >= writer_start_timeout:
            raise TimeoutError(
                f"Converted weight cache writer did not publish startup progress within {writer_start_timeout}s:"
                f" {path}{baseline_context}"
            )
        sleep(min(poll_interval, remaining))


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
    _write_cache_marker(owner_path, _workspace_owner())
    stop = threading.Event()

    def refresh_owner():
        while not stop.wait(_CACHE_HEARTBEAT_INTERVAL):
            try:
                _write_cache_marker(owner_path, _workspace_owner())
            except OSError:
                logger.warning("Could not refresh conversion workspace ownership for %s", tmp_path, exc_info=True)

    owner_thread = threading.Thread(target=refresh_owner, daemon=True)
    owner_thread.start()

    def publish():
        if conversion_cache_ready(snapshot_path):
            shutil.rmtree(tmp_path)
            return
        try:
            _replace(tmp_path, snapshot_path)
        except OSError:
            if conversion_cache_ready(snapshot_path):
                shutil.rmtree(tmp_path)
                return
            if not snapshot_path.exists():
                raise
            if conversion_cache_ready(snapshot_path):
                shutil.rmtree(tmp_path)
                return
            stale = snapshot_path.with_name(f"{snapshot_path.name}.stale-{uuid.uuid4().hex}")
            try:
                _rename(snapshot_path, stale)
            except FileNotFoundError:
                pass
            if conversion_cache_ready(stale):
                # It became ready after our pre-rename check. Restore that winner,
                # and never delete its shards if a third publisher has taken the destination.
                try:
                    _rename(stale, snapshot_path)
                except OSError:
                    if not conversion_cache_ready(snapshot_path):
                        raise
                shutil.rmtree(tmp_path)
                return
            try:
                # Recheck after moving the incomplete generation: a competing publisher may have won.
                if conversion_cache_ready(snapshot_path):
                    shutil.rmtree(tmp_path)
                    return
                try:
                    _replace(tmp_path, snapshot_path)
                except OSError:
                    if not conversion_cache_ready(snapshot_path):
                        raise
                    shutil.rmtree(tmp_path)
            finally:
                if stale.exists() and not conversion_cache_ready(stale):
                    shutil.rmtree(stale)

    def finish():
        stop.set()
        owner_thread.join(timeout=_CACHE_HEARTBEAT_JOIN_TIMEOUT)
        if not committed and tmp_path.exists():
            try:
                _write_cache_marker(owner_path, dict(_workspace_owner(), state="failed"))
                shutil.rmtree(tmp_path)
            except OSError:
                logger.info("Could not clean failed conversion workspace %s", tmp_path, exc_info=True)
                return
        owner_path.unlink(missing_ok=True)

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
    fresh_attempt: bool = False,
    preparation_root: str | Path | None = None,
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
                failure_path,
                dict(attempt_id=token, writer_id=writer_id, error=f"rank={rank}: {type(error).__name__}: {error}"),
            )
        except Exception:
            logger.warning("Could not record conversion failure for %s", snapshot_path, exc_info=True)

    initial_failure = _read_cache_marker(failure_path)
    initial_writer = _read_cache_marker(progress_path).get("writer_id")
    lock_path = snapshot_path.parent / f".{snapshot_path.name}.lock"
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with _conversion_writer_lock(lock_path, node_local=node_local):
            if conversion_cache_ready(snapshot_path):
                return
            if fresh_attempt and attempt_id is not None:
                marker = _preparation_marker_path(attempt_id, preparation_root).with_suffix(".failed")
                setup_failure = _read_cache_marker(marker)
                if setup_failure.get("attempt_id") == attempt_id:
                    failure_recorded = True
                    raise RuntimeError(f"Weight preparation writer failed: {setup_failure.get('error')}")
            failure = _read_cache_marker(failure_path)
            if failure.get("attempt_id") is not None and (
                failure != initial_failure
                or (not initial_failure and initial_writer is not None and failure.get("writer_id") == initial_writer)
                or (fresh_attempt and attempt_id is not None and failure["attempt_id"] == attempt_id)
            ):
                failure_recorded = True
                raise RuntimeError(f"Converted weight cache writer failed for {snapshot_path}: {failure.get('error')}")
            failure_path.unlink(missing_ok=True)
            _cleanup_abandoned_conversion_workspaces(snapshot_path, node_local=node_local)
            stop = threading.Event()
            thread = None
            try:
                _write_cache_marker(progress_path, dict(attempt_id=token, writer_id=writer_id, heartbeat=0))

                def heartbeat():
                    count = 0
                    next_sweep = _monotonic() + _ABANDONED_CACHE_AGE
                    while not stop.wait(_CACHE_HEARTBEAT_INTERVAL):
                        count += 1
                        try:
                            _write_cache_marker(
                                progress_path, dict(attempt_id=token, writer_id=writer_id, heartbeat=count)
                            )
                        except Exception:
                            logger.warning(
                                "Could not update conversion heartbeat for %s; retrying",
                                snapshot_path,
                                exc_info=True,
                            )
                        if not node_local and _monotonic() >= next_sweep:
                            _cleanup_abandoned_conversion_workspaces(snapshot_path)
                            next_sweep = _monotonic() + _ABANDONED_CACHE_AGE

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
                    thread.join(timeout=_CACHE_HEARTBEAT_JOIN_TIMEOUT)
                try:
                    if _read_cache_marker(progress_path).get("writer_id") == writer_id:
                        progress_path.unlink(missing_ok=True)
                except OSError:
                    logger.info("Could not remove conversion progress marker for %s", snapshot_path, exc_info=True)
    except BaseException as error:
        if not failure_recorded:
            record_failure(error)
        raise
