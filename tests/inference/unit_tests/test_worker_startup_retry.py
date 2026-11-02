import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import pytest

if "vllm" not in sys.modules:
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_module.__version__ = "0.32.0"
    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")

    def check_stop(*args, **kwargs):
        return False

    check_stop._arctic_router_replay_patch = True
    scheduler_module.check_stop = check_stop
    xgrammar_mod = types.ModuleType("vllm.v1.structured_output.backend_xgrammar")

    class XgrammarGrammar:
        def fill_bitmask(self, bitmask, idx):
            return None

    XgrammarGrammar.fill_bitmask._arctic_stop_mask_fix = True
    xgrammar_mod.XgrammarGrammar = XgrammarGrammar
    for name, module in (
        ("vllm", vllm_module),
        ("vllm.v1", types.ModuleType("vllm.v1")),
        ("vllm.v1.core", types.ModuleType("vllm.v1.core")),
        ("vllm.v1.core.sched", types.ModuleType("vllm.v1.core.sched")),
        ("vllm.v1.core.sched.scheduler", scheduler_module),
        ("vllm.v1.structured_output", types.ModuleType("vllm.v1.structured_output")),
        ("vllm.v1.structured_output.backend_xgrammar", xgrammar_mod),
    ):
        module.__path__ = []
        sys.modules[name] = module

from arctic_platform.inference.engine_startup_error import EngineStartupError
from arctic_platform.inference.server import worker as worker_mod
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState


def _install_fake_vllm(monkeypatch, async_llm_cls):
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    vllm_mod.__version__ = "0.32.0"
    plugins_mod = types.ModuleType("vllm.plugins")
    plugins_mod.load_general_plugins = lambda: None
    v1_mod = types.ModuleType("vllm.v1")
    v1_mod.__path__ = []
    engine_mod = types.ModuleType("vllm.v1.engine")
    engine_mod.__path__ = []
    async_llm_mod = types.ModuleType("vllm.v1.engine.async_llm")
    async_llm_mod.AsyncLLM = async_llm_cls

    vllm_mod.plugins = plugins_mod
    v1_mod.engine = engine_mod
    engine_mod.async_llm = async_llm_mod

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.plugins", plugins_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", async_llm_mod)


def test_is_address_in_use_error_walks_exception_chain():
    bind_error = RuntimeError(
        "torch.distributed.DistNetworkError: port: 43041, code: -98, "
        "name: EADDRINUSE, message: address already in use"
    )
    startup_error = RuntimeError("Engine core initialization failed")
    startup_error.__cause__ = bind_error

    assert worker_mod._is_address_in_use_error(startup_error)
    assert not worker_mod._is_address_in_use_error(RuntimeError("CUDA out of memory"))


@pytest.mark.parametrize("synced, checkpoint", [(["kv_b_proj"], None), (["q_b_proj"], ["q_b_proj", "kv_b_proj"])])
def test_initialize_rejects_kv_b_lora_on_mla_before_engine_start(monkeypatch, tmp_path, synced, checkpoint):
    import json

    class FakeEngineArgs:
        def create_engine_config(self):
            return types.SimpleNamespace(
                model_config=types.SimpleNamespace(use_mla=True, architecture="Glm5NextForConditionalGeneration"),
                structured_outputs_config=types.SimpleNamespace(enable_in_reasoning=False, reasoning_parser="deepseek_r1"),
            )

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, *args, **kwargs):
            pytest.fail("engine started before rejecting an unservable LoRA target")

    _install_fake_vllm(monkeypatch, FakeAsyncLLM)
    for name in ("_ensure_arctic_vllm_patches", "_ensure_router_replay_vllm_patches",
                 "ensure_xgrammar_stop_mask_fix", "ensure_spec_decode_grammar_fix"):
        monkeypatch.setattr(worker_mod, name, lambda: None)
    monkeypatch.setattr(worker_mod, "_create_async_engine_args", lambda *args, **kwargs: FakeEngineArgs())
    kwargs = {"model": "test-model", "lora_peft_config": {"target_modules": synced}}
    if checkpoint:
        (tmp_path / "adapter_config.json").write_text(json.dumps({"target_modules": checkpoint}))
        kwargs["lora_adapter_path"] = str(tmp_path)
    worker = InferenceWorker.__ray_metadata__.modified_class()
    with pytest.raises(ValueError, match="kv_b_proj.*absorbed"):
        asyncio.run(worker.initialize(kwargs))


@pytest.mark.parametrize("use_mla, targets", [(True, ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj"]), (False, ["kv_b_proj"])])
def test_mla_lora_guard_allows_servable_targets(use_mla, targets):
    config = types.SimpleNamespace(use_mla=use_mla)
    worker_mod._validate_mla_lora_targets(config, {"target_modules": targets})
    worker_mod._validate_mla_lora_targets(config, None)


def test_arctic_patch_fallback_stays_flag_independent(monkeypatch):
    from arctic_platform.inference.vllm import patches, required_patches

    arg_utils_mod = types.ModuleType("vllm.engine.arg_utils")
    arg_utils_mod.AsyncEngineArgs = type("AsyncEngineArgs", (), {})
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_mod)
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    monkeypatch.setenv("ARCTIC_INFERENCE_SKIP_VERSION_CHECK", "1")

    calls = []
    monkeypatch.setattr(
        required_patches,
        "apply_required_vllm_patches",
        lambda: calls.append("required"),
    )
    monkeypatch.setattr(
        patches,
        "apply_arctic_patches",
        lambda: calls.append("arctic"),
    )

    worker_mod._ensure_arctic_vllm_patches()

    assert calls == ["required", "arctic"]


def test_initialize_retries_vllm_engine_startup_on_address_in_use(monkeypatch):
    calls = []

    class FakeEngineArgs:
        def create_engine_config(self):
            return types.SimpleNamespace(
                structured_outputs_config=types.SimpleNamespace(
                    enable_in_reasoning=False
                ),
                model_config=types.SimpleNamespace(
                    skip_tokenizer_init=True, architecture="LlamaForCausalLM"
                ),
            )

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, vllm_config, stat_loggers):
            calls.append((vllm_config, stat_loggers))
            if len(calls) == 1:
                bind_error = RuntimeError("DistNetworkError: EADDRINUSE address already in use")
                startup_error = RuntimeError("Engine core initialization failed")
                startup_error.__cause__ = bind_error
                raise startup_error
            return cls()

    _install_fake_vllm(monkeypatch, FakeAsyncLLM)
    monkeypatch.setattr(worker_mod, "_ensure_arctic_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "_ensure_router_replay_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_xgrammar_stop_mask_fix", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_spec_decode_grammar_fix", lambda: None)
    monkeypatch.setattr(
        worker_mod,
        "_create_async_engine_args",
        lambda kwargs, **_ignored: FakeEngineArgs(),
    )
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_STARTUP_ATTEMPTS", "2")
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_STARTUP_RETRY_BASE_S", "0")

    worker_cls = InferenceWorker.__ray_metadata__.modified_class
    worker = worker_cls()

    asyncio.run(worker.initialize({"model": "test-model"}, model_id="job-1"))

    assert len(calls) == 2
    assert worker.state == WorkerLifecycleState.READY
    assert isinstance(worker.llm, FakeAsyncLLM)


def test_initialize_drops_reasoning_parser_when_tokenizer_lacks_think_tokens(
    monkeypatch,
):
    configs = []

    class FakeEngineArgs:
        def __init__(self, kwargs):
            self.kwargs = dict(kwargs)

        def create_engine_config(self):
            if self.kwargs.get("reasoning_parser"):
                raise RuntimeError(
                    "deepseek_r1 reasoning parser requires think start/end tokens"
                )
            return types.SimpleNamespace(
                structured_outputs_config=types.SimpleNamespace(
                    enable_in_reasoning=False
                ),
                model_config=types.SimpleNamespace(
                    skip_tokenizer_init=True, architecture="LlamaForCausalLM"
                ),
            )

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, vllm_config, stat_loggers):
            configs.append(vllm_config)
            return cls()

    _install_fake_vllm(monkeypatch, FakeAsyncLLM)
    monkeypatch.setattr(worker_mod, "_ensure_arctic_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "_ensure_router_replay_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_xgrammar_stop_mask_fix", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_spec_decode_grammar_fix", lambda: None)
    monkeypatch.setattr(
        worker_mod,
        "_create_async_engine_args",
        lambda kwargs, **_ignored: FakeEngineArgs(kwargs),
    )

    worker_cls = InferenceWorker.__ray_metadata__.modified_class
    worker = worker_cls()
    asyncio.run(
        worker.initialize(
            {"model": "test-model", "reasoning_parser": "deepseek_r1"},
            model_id="job-1",
        )
    )

    assert len(configs) == 1
    assert worker.state == WorkerLifecycleState.READY
    assert worker._reasoning_parser is None


def _initialize_with_failing_engine_core(monkeypatch, tmp_path):
    class FakeEngineArgs:
        def create_engine_config(self):
            return types.SimpleNamespace(
                structured_outputs_config=types.SimpleNamespace(enable_in_reasoning=False),
                model_config=types.SimpleNamespace(
                    skip_tokenizer_init=True, architecture="LlamaForCausalLM"
                ),
            )

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, vllm_config, stat_loggers):
            # Like vLLM's engine-core subprocess: the root cause goes to inherited stdio only.
            script = "import sys; print('engine core starting'); sys.stderr.write('ImportError: deep_gemm\\n')"
            subprocess.run([sys.executable, "-c", script], check=True)
            raise RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")

    _install_fake_vllm(monkeypatch, FakeAsyncLLM)
    monkeypatch.setattr(worker_mod, "_ensure_arctic_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "_ensure_router_replay_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_xgrammar_stop_mask_fix", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_spec_decode_grammar_fix", lambda: None)
    monkeypatch.setattr(worker_mod, "_create_async_engine_args", lambda kwargs, **_ignored: FakeEngineArgs())
    worker = InferenceWorker.__ray_metadata__.modified_class()

    # A Ray worker's stdout/stderr are files under the session's logs/ directory.
    saved = os.dup(1), os.dup(2)
    with open(tmp_path / "worker.out", "wb") as out, open(tmp_path / "worker.err", "wb") as err:
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        try:
            with pytest.raises(RuntimeError) as info:
                asyncio.run(worker.initialize({"model": "test-model"}, model_id="job-1"))
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
    return info.value


def test_initialize_failure_carries_engine_console_tail(monkeypatch, tmp_path):
    exc = _initialize_with_failing_engine_core(monkeypatch, tmp_path)
    rendered = "".join(traceback.format_exception(exc))
    assert "engine core starting" in rendered
    assert "ImportError: deep_gemm" in rendered


def test_unreadable_console_keeps_original_startup_error(monkeypatch, tmp_path):
    def deny_open(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(worker_mod, "open", deny_open, raising=False)
    exc = _initialize_with_failing_engine_core(monkeypatch, tmp_path)
    assert type(exc) is RuntimeError
    assert "See root cause above" in str(exc)
    assert not getattr(exc, "__notes__", None)


_GENERIC_STARTUP_ERROR = "Engine core initialization failed. See root cause above. Failed core proc(s): {}"
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
# An engine process as the Arctic plugin leaves it: the handler installed from the
# inherited ARCTIC_ENGINE_STARTUP_ERROR_DIR, then vLLM's startup-failure record.
_ENGINE_PROCESS = """
import logging, sys
from arctic_platform.inference.vllm.startup_error import install_startup_error_handler
install_startup_error_handler()
try:
    raise ValueError(sys.argv[2])
except ValueError:
    logging.getLogger("vllm.v1.executor.multiproc_executor").exception(sys.argv[1])
"""


def _run_engine_process(record, message):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, "-c", _ENGINE_PROCESS, record, message], check=True, env=env)


def _initialize_with_engine_attempts(monkeypatch, tmp_path, attempts):
    """Run initialize against a fake engine; ``attempts[i]()`` is what attempt i+1 does."""
    seen_dirs = []

    class FakeEngineArgs:
        def create_engine_config(self):
            return types.SimpleNamespace(
                structured_outputs_config=types.SimpleNamespace(enable_in_reasoning=False),
                model_config=types.SimpleNamespace(skip_tokenizer_init=True, architecture="LlamaForCausalLM"),
            )

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, vllm_config, stat_loggers):
            seen_dirs.append(os.environ.get(worker_mod._STARTUP_ERROR_DIR_ENV))
            attempts[len(seen_dirs) - 1]()

    _install_fake_vllm(monkeypatch, FakeAsyncLLM)
    monkeypatch.setattr(worker_mod, "_ensure_arctic_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "_ensure_router_replay_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_xgrammar_stop_mask_fix", lambda: None)
    monkeypatch.setattr(worker_mod, "ensure_spec_decode_grammar_fix", lambda: None)
    monkeypatch.setattr(worker_mod, "_create_async_engine_args", lambda kwargs, **_ignored: FakeEngineArgs())
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_STARTUP_ATTEMPTS", str(len(attempts)))
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_STARTUP_RETRY_BASE_S", "0")
    worker = InferenceWorker.__ray_metadata__.modified_class()

    saved = os.dup(1), os.dup(2)
    with open(tmp_path / "worker.out", "wb") as out, open(tmp_path / "worker.err", "wb") as err:
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        try:
            with pytest.raises(BaseException) as info:
                asyncio.run(worker.initialize({"model": "test-model"}, model_id="job-1"))
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
    return info.value, seen_dirs


def _raise(exc):
    def attempt():
        raise exc

    return attempt


def _engine_fails(record, message, exc):
    def attempt():
        _run_engine_process(record, message)
        raise exc

    return attempt


def test_initialize_failure_carries_the_engine_root_cause(monkeypatch, tmp_path):
    original = RuntimeError(_GENERIC_STARTUP_ERROR)
    root = "Following weights were not initialized from checkpoint: {'model.norm.weight'}"

    exc, seen_dirs = _initialize_with_engine_attempts(
        monkeypatch, tmp_path, [_engine_fails("WorkerProc failed to start.", root, original)]
    )

    assert type(exc) is EngineStartupError
    assert str(exc) == f"{_GENERIC_STARTUP_ERROR}\nRoot cause in MainProcess: ValueError: {root}"
    assert (exc.root_type, exc.root_message, exc.process, exc.origin) == (
        "ValueError",
        root,
        "MainProcess",
        "__main__.<module>",
    )
    assert exc.__cause__ is original
    assert not getattr(original, "__notes__", None)
    # #196's console tail goes on the raised error, so it is rendered once, after the root.
    rendered = "".join(traceback.format_exception(exc))
    assert rendered.count("Engine console tail") == 1
    assert rendered.index("Root cause in") < rendered.index("Engine console tail")
    assert not os.path.exists(seen_dirs[0])
    assert worker_mod._STARTUP_ERROR_DIR_ENV not in os.environ


def test_initialize_failure_without_a_record_is_raised_unchanged(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=worker_mod.logger.name)
    original = RuntimeError(_GENERIC_STARTUP_ERROR)

    exc, _ = _initialize_with_engine_attempts(monkeypatch, tmp_path, [_raise(original)])

    assert exc is original
    assert str(exc) == _GENERIC_STARTUP_ERROR
    assert any("no engine process recorded a failure" in record.getMessage() for record in caplog.records)


def test_initialize_failure_takes_the_worker_type_from_a_failed_rpc(monkeypatch, tmp_path):
    """A warmup fault: the worker's RPC failure, not the EngineCore's wrapper, is the root."""
    original = RuntimeError(_GENERIC_STARTUP_ERROR)
    root = "Triton Error [CUDA]: an illegal memory access was encountered"

    def attempt():
        _run_engine_process("WorkerProc hit an exception.", root)
        _run_engine_process(
            "EngineCore failed to start.",
            f"Worker failed with error '{root}', please check the stack trace above for the root cause",
        )
        raise original

    exc, _ = _initialize_with_engine_attempts(monkeypatch, tmp_path, [attempt])

    assert type(exc) is EngineStartupError
    assert str(exc) == f"{_GENERIC_STARTUP_ERROR}\nRoot cause in MainProcess: ValueError: {root}"
    assert (exc.root_type, exc.root_message) == ("ValueError", root)


def test_unreadable_record_keeps_the_original_error(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=worker_mod.logger.name)

    def unreadable(directory):
        raise PermissionError(13, "Permission denied", directory)

    monkeypatch.setattr(worker_mod, "earliest_startup_record", unreadable)
    original = RuntimeError(_GENERIC_STARTUP_ERROR)

    exc, _ = _initialize_with_engine_attempts(
        monkeypatch, tmp_path, [_engine_fails("EngineCore failed to start.", "CUDA out of memory.", original)]
    )

    assert exc is original
    assert str(exc) == _GENERIC_STARTUP_ERROR
    failed = [record for record in caplog.records if "reading" in record.getMessage()]
    assert len(failed) == 1
    assert isinstance(failed[0].exc_info[1], PermissionError)


def test_capture_directory_failure_is_logged(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=worker_mod.logger.name)
    monkeypatch.delenv(worker_mod._STARTUP_ERROR_DIR_ENV, raising=False)

    def no_space(prefix):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(worker_mod.tempfile, "mkdtemp", no_space)

    assert worker_mod._begin_engine_startup_capture() is None
    assert worker_mod._STARTUP_ERROR_DIR_ENV not in os.environ
    assert any("not recorded for this attempt" in record.getMessage() for record in caplog.records)


def test_record_from_a_previous_attempt_is_ignored(monkeypatch, tmp_path):
    bind_error = RuntimeError("DistNetworkError: EADDRINUSE address already in use")
    original = RuntimeError(_GENERIC_STARTUP_ERROR)

    exc, seen_dirs = _initialize_with_engine_attempts(
        monkeypatch,
        tmp_path,
        [_engine_fails("EngineCore failed to start.", "stale root from attempt 1", bind_error), _raise(original)],
    )

    assert exc is original
    assert "stale root" not in str(exc)
    assert len(set(seen_dirs)) == 2 and None not in seen_dirs
    assert not any(os.path.exists(directory) for directory in seen_dirs)
    assert worker_mod._STARTUP_ERROR_DIR_ENV not in os.environ


class _OutOfMemoryError(RuntimeError):
    pass


@pytest.mark.parametrize(
    "original", [ValueError("max_model_len is too large"), _OutOfMemoryError("CUDA out of memory")]
)
def test_initialize_failure_of_another_type_keeps_its_type(monkeypatch, tmp_path, original):
    exc, _ = _initialize_with_engine_attempts(
        monkeypatch, tmp_path, [_engine_fails("EngineCore failed to start.", "CUDA out of memory.", original)]
    )

    assert exc is original
    assert "Root cause in" not in str(exc)
    assert any(note.startswith("Engine console tail") for note in exc.__notes__)


@pytest.mark.parametrize(
    "exc", [asyncio.CancelledError(), KeyboardInterrupt(), ValueError("x"), _OutOfMemoryError("x")]
)
def test_engine_startup_error_converts_only_a_plain_runtime_error(tmp_path, exc):
    (tmp_path / "1.json").write_text('{"process": "EngineCore", "type": "RuntimeError", "message": "x", "time": 1.0}')

    assert worker_mod._engine_startup_error(exc, str(tmp_path)) is None
    converted = worker_mod._engine_startup_error(RuntimeError("generic"), str(tmp_path))
    assert str(converted) == "generic\nRoot cause in EngineCore: RuntimeError: x"
    # A record written before ``origin`` existed still converts.
    assert (converted.root_type, converted.root_message, converted.process, converted.origin) == (
        "RuntimeError",
        "x",
        "EngineCore",
        None,
    )


# An EngineCore as vLLM leaves it after a CUDA illegal memory access in warmup:
# it records "EngineCore failed to start." and then hangs on the way out
# instead of exiting, so the front end's wait for it never ends.
_HUNG_ENGINE_PROCESS = _ENGINE_PROCESS + "import time\ntime.sleep(600)\n"
_SLEEP = "import time\ntime.sleep(600)\n"


def _titled(title, script):
    """``script`` run under a process title, set the way vLLM sets it."""
    return f"import setproctitle\nsetproctitle.setproctitle({title!r})\n" + script


# Every process a test here starts, so the fixture below can reap it.
_SPAWNED: list = []


def _spawn(argv, **kwargs):
    proc = subprocess.Popen(argv, **kwargs)
    _SPAWNED.append(proc)
    return proc


@pytest.fixture(autouse=True)
def _reap_spawned_processes():
    """Kill and wait on every process a test started, and their descendants,
    even when the test fails: a leftover ``VLLM::`` process would be seen,
    and killed, by a later test's kill pass."""
    import psutil

    yield
    procs = []
    for popen in _SPAWNED:
        try:
            root = psutil.Process(popen.pid)
            procs.extend(root.children(recursive=True))
            procs.append(root)
        except psutil.Error:
            pass
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)
    for popen in _SPAWNED:
        popen.kill()
        popen.wait()
    _SPAWNED.clear()
    # When pytest is a subreaper (e.g. PID 1 in a container), a grandchild
    # orphaned by an engine's kill is reparented here; reap it.
    for child in psutil.Process().children():
        try:
            if child.status() == psutil.STATUS_ZOMBIE:
                os.waitpid(child.pid, os.WNOHANG)
        except (psutil.Error, ChildProcessError):
            pass


def _spawn_engine_core(script, *args):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH")]))}
    return _spawn([sys.executable, "-c", _titled("VLLM::EngineCore", script), *args], env=env)


def _spawn_child(title, script, *args):
    """Source for the engine process to start a titled child of its own; it writes the child's pid to argv[-1]."""
    return (
        "import subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_titled(title, script)!r}, *{list(args)!r}])\n"
        "open(sys.argv[-1], 'w').write(str(child.pid))\n"
    )


def _wait_for_engine_core(proc):
    """What vLLM's wait_for_engine_startup does without a READY: wait for the EngineCore to exit."""
    proc.wait()
    raise RuntimeError(
        f"Engine core initialization failed. See root cause above. Failed core proc(s): {{'EngineCore': {proc.returncode}}}"
    )


def _gone(pid):
    import psutil

    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_gone(pid, timeout_s=10):
    deadline = time.monotonic() + timeout_s
    while not _gone(pid):
        assert time.monotonic() < deadline, f"pid {pid} is still running"
        time.sleep(0.05)


def _watchdog_threads():
    import threading

    return [thread for thread in threading.enumerate() if thread.name == "ArcticEngineStartupWatchdog"]


def _run_attempt(monkeypatch, tmp_path, spawn):
    procs = []

    def attempt():
        procs.append(spawn())
        _wait_for_engine_core(procs[-1])

    start = time.monotonic()
    try:
        exc, _ = _initialize_with_engine_attempts(monkeypatch, tmp_path, [attempt])
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()
    return exc, procs[0], time.monotonic() - start


def test_engine_that_records_its_failure_and_hangs_is_killed(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_FAILED_STARTUP_GRACE_S", "0.5")
    root = "Worker failed with error 'Triton Error [CUDA]: an illegal memory access was encountered'"

    exc, engine, elapsed = _run_attempt(
        monkeypatch, tmp_path, lambda: _spawn_engine_core(_HUNG_ENGINE_PROCESS, "EngineCore failed to start.", root)
    )

    assert elapsed < 10
    assert engine.returncode == -signal.SIGKILL
    assert type(exc) is EngineStartupError
    assert f"Root cause in MainProcess: ValueError: {root}" in str(exc)
    assert any(f"watchdog killed engine processes [{engine.pid}]" in note for note in exc.__notes__)
    assert _watchdog_threads() == []


def test_hung_engine_kill_takes_its_tp_workers_too(monkeypatch, tmp_path):
    """A TP worker wedged on the GPU records nothing; killing only the
    EngineCore would orphan it with the GPU memory it holds."""
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_FAILED_STARTUP_GRACE_S", "0.5")
    pid_file = tmp_path / "tp_worker.pid"
    script = _spawn_child("VLLM::Worker_TP1", _SLEEP) + _HUNG_ENGINE_PROCESS

    exc, engine, _ = _run_attempt(
        monkeypatch,
        tmp_path,
        lambda: _spawn_engine_core(script, "EngineCore failed to start.", "illegal memory access", str(pid_file)),
    )

    assert engine.returncode == -signal.SIGKILL
    assert "illegal memory access" in str(exc)
    _wait_gone(int(pid_file.read_text()))


def test_unrecorded_engine_core_is_killed_by_its_title(monkeypatch, tmp_path):
    """Only a TP worker recorded a failure; the EngineCore the front end waits on
    recorded nothing and hangs. It is found by vLLM's title."""
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_FAILED_STARTUP_GRACE_S", "0.5")
    pid_file = tmp_path / "tp_worker.pid"
    worker = _ENGINE_PROCESS.replace("sys.argv[1]", "'WorkerProc failed to start.'").replace(
        "sys.argv[2]", "'Triton Error [CUDA]: an illegal memory access was encountered'"
    )
    script = _spawn_child("VLLM::Worker_TP0", worker) + "child.wait()\n" + _SLEEP

    exc, engine, elapsed = _run_attempt(
        monkeypatch, tmp_path, lambda: _spawn_engine_core(script, str(pid_file))
    )

    assert elapsed < 10
    assert engine.returncode == -signal.SIGKILL
    assert "Root cause in MainProcess: ValueError: Triton Error [CUDA]: an illegal memory access" in str(exc)


def test_slow_engine_startup_without_a_record_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_FAILED_STARTUP_GRACE_S", "0")
    procs = []
    done = ValueError("engine came up")

    def attempt():
        procs.append(_spawn_engine_core("import time; time.sleep(2.5)"))
        procs[-1].wait()
        raise done

    exc, _ = _initialize_with_engine_attempts(monkeypatch, tmp_path, [attempt])

    assert exc is done
    assert procs[0].returncode == 0
    assert _watchdog_threads() == []


def test_engine_that_exits_after_recording_its_failure_is_not_killed(monkeypatch, tmp_path):
    # Shorter than the default so a kill would have time to land, longer than the exit takes.
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_FAILED_STARTUP_GRACE_S", "2")

    exc, engine, _ = _run_attempt(
        monkeypatch,
        tmp_path,
        lambda: _spawn_engine_core(
            _ENGINE_PROCESS + "import time\ntime.sleep(1)\n", "EngineCore failed to start.", "CUDA out of memory."
        ),
    )

    assert engine.returncode == 0
    assert "Root cause in MainProcess: ValueError: CUDA out of memory." in str(exc)
    assert not any("watchdog killed" in note for note in getattr(exc, "__notes__", []))


def test_kill_failed_engine_processes_spares_other_children():
    import psutil

    engine = _spawn_engine_core(_SLEEP)
    bystander = _spawn([sys.executable, "-c", _SLEEP])
    try:
        deadline = time.monotonic() + 10
        while not psutil.Process(engine.pid).name().startswith("VLLM::"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        # A recorded pid that is not a descendant (here pid 1) is never signalled.
        # Asserted only about this test's own processes: anything else under
        # vLLM's title in this process tree would be killed too, by design.
        killed = worker_mod._kill_failed_engine_processes({1})
        assert engine.pid in killed
        assert bystander.pid not in killed and 1 not in killed
        assert engine.wait(timeout=10) == -signal.SIGKILL
        assert bystander.poll() is None
    finally:
        engine.kill()
        bystander.kill()
        engine.wait()
        bystander.wait()


def test_kill_does_not_read_cmdline(monkeypatch, tmp_path):
    """Reading /proc/<pid>/cmdline can block on a wedged process, and psutil's
    name() falls back to it for vLLM's 15-char titles; the kill must not touch it."""
    import psutil

    def no_cmdline(self):
        raise AssertionError("cmdline() read")

    monkeypatch.setattr(psutil.Process, "cmdline", no_cmdline)
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_FAILED_STARTUP_GRACE_S", "0.5")
    pid_file = tmp_path / "tp_worker.pid"
    script = _spawn_child("VLLM::Worker_TP1", _SLEEP) + _HUNG_ENGINE_PROCESS

    exc, engine, elapsed = _run_attempt(
        monkeypatch,
        tmp_path,
        lambda: _spawn_engine_core(script, "EngineCore failed to start.", "illegal memory access", str(pid_file)),
    )

    assert elapsed < 10
    assert engine.returncode == -signal.SIGKILL
    _wait_gone(int(pid_file.read_text()))


def test_watchdog_retries_the_kill_while_vllm_has_not_raised(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_FAILED_STARTUP_GRACE_S", "0.5")
    real_kill = worker_mod._kill_failed_engine_processes
    calls = []

    def kill_on_second_try(failed, stop=None):
        calls.append(sorted(failed))
        return [] if len(calls) == 1 else real_kill(failed, stop)

    monkeypatch.setattr(worker_mod, "_kill_failed_engine_processes", kill_on_second_try)

    exc, engine, elapsed = _run_attempt(
        monkeypatch, tmp_path, lambda: _spawn_engine_core(_HUNG_ENGINE_PROCESS, "EngineCore failed to start.", "x")
    )

    assert len(calls) == 2
    assert engine.returncode == -signal.SIGKILL
    assert elapsed < 15


def test_stuck_watchdog_does_not_hold_startup(monkeypatch, tmp_path, caplog):
    import threading

    caplog.set_level(logging.WARNING, logger=worker_mod.logger.name)
    release = threading.Event()

    def stuck(directory):
        release.wait(30)  # like a read of a wedged process's /proc entry
        return set()

    monkeypatch.setattr(worker_mod, "recorded_pids", stuck)
    monkeypatch.setattr(worker_mod, "_WATCHDOG_JOIN_SLACK_S", 0.2)
    try:
        start = time.monotonic()
        with worker_mod._FailedStartupWatchdog(str(tmp_path), grace_s=0.0, poll_s=0.05):
            time.sleep(0.2)
        assert time.monotonic() - start < 2
        assert "watchdog did not stop" in caplog.text
    finally:
        release.set()


def test_kill_pass_signals_nothing_once_the_watchdog_is_stopped():
    """A watchdog that outlived its attempt must not kill a later attempt's engine."""
    import threading

    import psutil

    engine = _spawn_engine_core(_SLEEP)
    try:
        deadline = time.monotonic() + 10
        while not psutil.Process(engine.pid).name().startswith("VLLM::"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        stop = threading.Event()
        stop.set()
        assert worker_mod._kill_failed_engine_processes({engine.pid}, stop) == []
        time.sleep(0.2)
        assert engine.poll() is None
    finally:
        engine.kill()
        engine.wait()


def test_kill_pass_skips_zombie_descendants():
    """Under a subreaper, an exited engine process can sit here as a zombie
    descendant; it holds nothing and is not reported as killed."""
    import ctypes

    import psutil

    pr_set_child_subreaper = 36
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
        pytest.skip("needs prctl(PR_SET_CHILD_SUBREAPER)")
    try:
        # The parent exits at once, leaving its titled child, which exits too,
        # reparented to this (subreaper) process as an unreaped zombie.
        parent = _spawn([sys.executable, "-c", _spawn_child("VLLM::Worker_TP0", "import time; time.sleep(0.5)"), "/dev/null"])
        parent.wait()
        deadline = time.monotonic() + 10
        while not [c for c in psutil.Process().children() if c.status() == psutil.STATUS_ZOMBIE
                   and c.pid != parent.pid]:
            assert time.monotonic() < deadline, "no zombie descendant appeared"
            time.sleep(0.05)

        assert worker_mod._kill_failed_engine_processes(set()) == []
    finally:
        libc.prctl(pr_set_child_subreaper, 0, 0, 0, 0)
