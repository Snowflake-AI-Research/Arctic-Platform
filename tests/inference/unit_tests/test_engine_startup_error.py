import json
import logging
import os

import pytest

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
