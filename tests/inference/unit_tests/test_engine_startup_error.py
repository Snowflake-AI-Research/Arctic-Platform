import hashlib
import json
import logging
import os
from types import SimpleNamespace

import pytest

from arctic_platform.debug import inference_startup_error as durable_startup_error
from arctic_platform.inference.vllm import startup_error


def _record(msg, exc=None, **attrs):
    exc_info = (type(exc), exc, exc.__traceback__) if exc is not None else None
    record = logging.LogRecord("vllm.v1.engine.core", logging.ERROR, __file__, 1, msg, None, exc_info)
    record.__dict__.update(attrs)
    return record


def _write(directory, pid, **entry):
    (directory / f"{pid}.json").write_text(json.dumps(entry))


@pytest.mark.parametrize("msg", ["EngineCore failed to start.", "WorkerProc failed to start."])
def test_handler_records_a_startup_failure(tmp_path, msg):
    handler = startup_error.StartupErrorHandler(str(tmp_path))
    record = _record(msg, ValueError("Field 'x' not found in VllmConfig."), vllm_process_name="Worker_TP0")

    handler.handle(record)

    assert os.listdir(tmp_path) == [f"{os.getpid()}.json"]
    assert json.loads((tmp_path / f"{os.getpid()}.json").read_text()) == {
        "process": "Worker_TP0",
        "type": "ValueError",
        "message": "Field 'x' not found in VllmConfig.",
        "time": record.created,
    }


@pytest.mark.parametrize(
    "record",
    [
        # A worker that fails during an RPC: the EngineCore records the failure it raises.
        _record("WorkerProc hit an exception.", RuntimeError("block_kv == 64")),
        _record("EngineCore encountered a fatal error.", RuntimeError("after startup")),
        _record("EngineCore failed to start."),
        _record({"msg": "not a string"}, RuntimeError("x")),
    ],
)
def test_handler_ignores_other_records(tmp_path, record):
    startup_error.StartupErrorHandler(str(tmp_path)).handle(record)

    assert os.listdir(tmp_path) == []


def test_handler_names_the_type_as_tracebacks_do(tmp_path):
    class OutOfMemoryError(RuntimeError):
        pass

    OutOfMemoryError.__module__ = "torch"
    OutOfMemoryError.__qualname__ = "OutOfMemoryError"
    startup_error.StartupErrorHandler(str(tmp_path)).handle(
        _record("EngineCore failed to start.", OutOfMemoryError("CUDA out of memory."))
    )

    assert startup_error.earliest_startup_error(str(tmp_path)) == (
        "Root cause in MainProcess: torch.OutOfMemoryError: CUDA out of memory."
    )


def test_handler_write_failure_goes_to_handle_error(tmp_path):
    handler = startup_error.StartupErrorHandler(str(tmp_path / "removed"))
    reported = []
    handler.handleError = reported.append
    record = _record("EngineCore failed to start.", RuntimeError("x"))

    handler.handle(record)

    assert reported == [record]


def test_message_and_line_are_bounded(tmp_path):
    startup_error.StartupErrorHandler(str(tmp_path)).handle(
        _record("WorkerProc failed to start.", ValueError("x" * 10_000), vllm_process_name="Worker_TP0")
    )

    entry = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
    assert len(entry["message"]) == startup_error.MAX_CHARS
    line = startup_error.earliest_startup_error(str(tmp_path))
    assert line.startswith("Root cause in Worker_TP0: ValueError: xxx")
    assert len(line) == startup_error.MAX_CHARS


def test_earliest_record_is_the_root(tmp_path):
    _write(
        tmp_path,
        101,
        process="EngineCore",
        type="Exception",
        message="WorkerProc initialization failed due to an exception in a background process.",
        time=20.0,
    )
    _write(tmp_path, 102, process="Worker_TP1", type="RuntimeError", message="peer died", time=12.5)
    _write(
        tmp_path,
        103,
        process="Worker_TP0",
        type="ValueError",
        message="Field '_shift_cudagraph_capture_sizes' not found in VllmConfig.",
        time=11.0,
    )

    assert startup_error.earliest_startup_error(str(tmp_path)) == (
        "Root cause in Worker_TP0: ValueError: Field '_shift_cudagraph_capture_sizes' not found in VllmConfig."
    )


def test_unreadable_records_are_skipped(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=startup_error.__name__)
    (tmp_path / "1.json").write_text("{not json")
    (tmp_path / "2.json").write_text(json.dumps({"time": 1.0}))
    (tmp_path / "3.json").write_text(json.dumps(["not", "a", "record"]))
    (tmp_path / "4.json.tmp").write_text(
        json.dumps({"process": "EngineCore", "type": "RuntimeError", "message": "mid-write", "time": 0.5})
    )

    assert startup_error.earliest_startup_error(str(tmp_path)) is None
    skipped = sorted(record.args[0] for record in caplog.records if record.msg.startswith("Skipping unreadable"))
    assert skipped == ["1.json", "2.json", "3.json"]

    _write(tmp_path, 5, process="EngineCore", type="RuntimeError", message="readable", time=9.0)
    assert startup_error.earliest_startup_error(str(tmp_path)) == "Root cause in EngineCore: RuntimeError: readable"


@pytest.fixture
def vllm_logger():
    logger = logging.getLogger("vllm")
    before = list(logger.handlers)
    yield logger
    for handler in logger.handlers[:]:
        if handler not in before:
            logger.removeHandler(handler)


def test_install_without_directory_adds_no_handler(monkeypatch, vllm_logger):
    monkeypatch.delenv(startup_error.ENV_VAR, raising=False)
    before = list(vllm_logger.handlers)

    startup_error.install_startup_error_handler()

    assert vllm_logger.handlers == before


def test_install_records_through_the_vllm_logger_once(monkeypatch, tmp_path, vllm_logger):
    monkeypatch.setenv(startup_error.ENV_VAR, str(tmp_path))
    before = list(vllm_logger.handlers)

    startup_error.install_startup_error_handler()
    startup_error.install_startup_error_handler()

    assert len(vllm_logger.handlers) == len(before) + 1
    engine_logger = logging.getLogger("vllm.v1.engine.core")
    engine_logger.warning("EngineCore failed to start.", exc_info=RuntimeError("below ERROR"))
    assert os.listdir(tmp_path) == []
    try:
        raise RuntimeError("VLLM batch_invariant mode is not supported for GDN_ATTN.")
    except RuntimeError:
        engine_logger.exception("EngineCore failed to start.")
    assert startup_error.earliest_startup_error(str(tmp_path)) == (
        "Root cause in MainProcess: RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN."
    )


@pytest.mark.vllm
def test_engine_startup_root_reaches_the_raised_error():
    """A real engine whose KV cache cannot fit: the EngineCore's own error is in the message.

    Fails when a vLLM bump renames the startup-failure records or moves plugin loading.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import ray

    from arctic_platform.inference.server.worker import InferenceWorker

    ray.init(include_dashboard=False)
    try:
        worker = InferenceWorker.options(num_gpus=1).remote()
        engine_kwargs = {
            "model": "Qwen/Qwen3-0.6B",
            "gpu_memory_utilization": 0.01,
            "max_model_len": 512,
            "enforce_eager": True,
        }
        with pytest.raises(Exception) as info:
            ray.get(worker.initialize.remote(engine_kwargs), timeout=600)
    finally:
        ray.shutdown()

    message = str(info.value.cause)
    assert message.startswith("Engine core initialization failed.")
    assert "\nRoot cause in EngineCore: ValueError: " in message
    assert "gpu_memory_utilization" in message


def _durable_startup_error_new_directory_identity(path):
    metadata = path.stat()
    return [
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    ]


@pytest.fixture
def durable_startup_error_new_case(tmp_path, monkeypatch):
    durable_startup_error._reset_inference_startup_error_for_tests()
    monkeypatch.delenv(durable_startup_error.REQUEST_ENV, raising=False)
    monkeypatch.delenv(durable_startup_error.ATTEMPT_ENV, raising=False)
    root = tmp_path / "durable-startup-errors"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    request = {
        "schema": "inference-startup-error-v1",
        "request_id": "future-e03-attempt",
        "config_id": "qwen35-e03",
        "check_id": "inference-checkpoint-loss",
        "generation": "generation-1",
        "root": {
            "path": str(root),
            "identity": _durable_startup_error_new_directory_identity(root),
            "parent_identity": _durable_startup_error_new_directory_identity(
                root.parent
            ),
        },
        "sources": {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in durable_startup_error._SOURCE_PATHS.items()
        },
    }
    case = SimpleNamespace(root=root, request=request)
    try:
        yield case
    finally:
        durable_startup_error._reset_inference_startup_error_for_tests()


def _durable_startup_error_new_enable(monkeypatch, request):
    monkeypatch.setenv(
        durable_startup_error.REQUEST_ENV,
        json.dumps(request, sort_keys=True, separators=(",", ":")),
    )


def _durable_startup_error_new_records(root):
    return [
        json.loads(path.read_bytes())
        for path in sorted(root.iterdir())
        if path.suffix == ".json"
    ]


def test_durable_startup_error_new_disabled_paths_are_exact_noops(
    monkeypatch,
    vllm_logger,
):
    durable_startup_error._reset_inference_startup_error_for_tests()
    monkeypatch.delenv(durable_startup_error.REQUEST_ENV, raising=False)
    before = list(vllm_logger.handlers)

    assert (
        durable_startup_error.maybe_install_inference_startup_error_producer()
        is None
    )
    assert (
        durable_startup_error.maybe_begin_inference_startup_attempt(1) is None
    )
    assert (
        durable_startup_error.maybe_finish_inference_startup_attempt(
            None,
            "ready",
            None,
        )
        is None
    )
    assert vllm_logger.handlers == before
    assert durable_startup_error._ACTIVE is None


def test_durable_startup_error_new_records_full_nonoverwriting_tracebacks(
    durable_startup_error_new_case,
    monkeypatch,
    vllm_logger,
):
    case = durable_startup_error_new_case
    _durable_startup_error_new_enable(monkeypatch, case.request)
    durable_startup_error.maybe_install_inference_startup_error_producer()
    durable_startup_error.maybe_install_inference_startup_error_producer()
    handlers = [
        handler
        for handler in vllm_logger.handlers
        if isinstance(
            handler,
            durable_startup_error.DurableStartupErrorHandler,
        )
    ]
    assert len(handlers) == 1

    marker_one = "complete-trace-one"
    marker_two = "complete-trace-two"
    engine_logger = logging.getLogger("vllm.v1.engine.core")
    try:
        raise ValueError(marker_one)
    except ValueError:
        engine_logger.exception(
            "WorkerProc failed to start.",
            extra={"vllm_process_name": "Worker_TP0"},
        )
    try:
        raise RuntimeError(marker_two)
    except RuntimeError:
        engine_logger.exception("EngineCore failed to start.")

    records = _durable_startup_error_new_records(case.root)
    assert [record["sequence"] for record in records] == [0, 1]
    assert [record["phase"] for record in records] == [
        "startup_error",
        "startup_error",
    ]
    assert records[0]["process_name"] == "Worker_TP0"
    assert records[0]["exception_message"] == marker_one
    assert records[1]["exception_message"] == marker_two
    assert marker_one in records[0]["traceback"]
    assert marker_two in records[1]["traceback"]
    assert all(record["traceback_complete"] for record in records)
    assert all(record["rank_state"] == "unknown" for record in records)
    before = {path.name: path.read_bytes() for path in case.root.iterdir()}
    durable_startup_error._reset_inference_startup_error_for_tests()
    assert {path.name: path.read_bytes() for path in case.root.iterdir()} == before


def test_durable_startup_error_new_create_only_collision_preserves_foreign_record(
    durable_startup_error_new_case,
    monkeypatch,
):
    case = durable_startup_error_new_case
    _durable_startup_error_new_enable(monkeypatch, case.request)
    producer = durable_startup_error._active_producer()
    name = (
        f"{producer.host_id}-{producer.pid}-{producer.start_ticks}-"
        "00000000.json"
    )
    path = case.root / name
    path.write_bytes(b"foreign")

    with pytest.raises(FileExistsError):
        producer.write("attempt_begin", {"coordinator": True})

    assert path.read_bytes() == b"foreign"


@pytest.mark.parametrize("damage", ["digest", "root-link", "root-generation"])
def test_durable_startup_error_new_rejects_unauthenticated_sources_and_roots(
    durable_startup_error_new_case,
    monkeypatch,
    tmp_path,
    damage,
):
    case = durable_startup_error_new_case
    request = json.loads(json.dumps(case.request))
    if damage == "digest":
        request["sources"]["producer"] = "0" * 64
    elif damage == "root-link":
        alias = tmp_path / "root-alias"
        alias.symlink_to(case.root, target_is_directory=True)
        request["root"]["path"] = str(alias)
    else:
        request["root"]["identity"][-1] += 1
    _durable_startup_error_new_enable(monkeypatch, request)

    with pytest.raises((RuntimeError, ValueError)):
        durable_startup_error.maybe_install_inference_startup_error_producer()

    assert durable_startup_error._ACTIVE is None
    assert list(case.root.iterdir()) == []


def test_durable_startup_error_new_rank_identity_changes_only_after_init(
    monkeypatch,
):
    distributed = pytest.importorskip("torch.distributed")
    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: False)
    assert durable_startup_error._rank_identity() == {
        "rank_state": "unknown",
        "global_rank": None,
        "local_rank": None,
        "world_size": None,
    }

    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_rank", lambda: 7)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 16)
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert durable_startup_error._rank_identity() == {
        "rank_state": "initialized",
        "global_rank": 7,
        "local_rank": 3,
        "world_size": 16,
    }


def test_durable_startup_error_new_attempt_restores_env_and_preserves_exception(
    durable_startup_error_new_case,
    monkeypatch,
):
    case = durable_startup_error_new_case
    _durable_startup_error_new_enable(monkeypatch, case.request)
    attempt = durable_startup_error.maybe_begin_inference_startup_attempt(3)
    attempt_value = json.loads(
        os.environ[durable_startup_error.ATTEMPT_ENV]
    )
    assert attempt_value["attempt_ordinal"] == 3
    assert attempt_value["attempt_id"] == attempt.attempt_id
    original = KeyboardInterrupt("engine interrupted")
    monkeypatch.setattr(
        attempt.producer,
        "write",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("diagnostic write failed")
        ),
    )

    durable_startup_error.maybe_finish_inference_startup_attempt(
        attempt,
        "error",
        original,
    )

    assert durable_startup_error.ATTEMPT_ENV not in os.environ
    assert str(original) == "engine interrupted"
    assert any(
        "diagnostic write failed" in note
        for note in getattr(original, "__notes__", [])
    )


def test_durable_startup_error_new_constructor_failure_closes_all_held_state(
    durable_startup_error_new_case,
    monkeypatch,
):
    case = durable_startup_error_new_case
    closed = []

    class Held:
        def __init__(self, label):
            self.label = label

        def close(self):
            closed.append(self.label)

    monkeypatch.setattr(
        durable_startup_error._HeldRoot,
        "open",
        lambda descriptor: Held("root"),
    )
    monkeypatch.setattr(
        durable_startup_error._HeldSource,
        "open",
        lambda label, path, digest: Held(label),
    )
    monkeypatch.setattr(
        durable_startup_error,
        "_process_identity",
        lambda: (_ for _ in ()).throw(
            RuntimeError("process identity unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="process identity unavailable"):
        durable_startup_error._Producer(case.request)

    assert closed == [
        *reversed(list(durable_startup_error._SOURCE_PATHS)),
        "root",
    ]
