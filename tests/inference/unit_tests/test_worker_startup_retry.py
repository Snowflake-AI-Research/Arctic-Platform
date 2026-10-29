import asyncio
import logging
import os
import subprocess
import sys
import traceback
import types
from pathlib import Path

import pytest

if "vllm" not in sys.modules:
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_module.__version__ = "0.31.0"
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

from arctic_platform.inference.server import worker as worker_mod
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState


def _install_fake_vllm(monkeypatch, async_llm_cls):
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    vllm_mod.__version__ = "0.31.0"
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


@pytest.mark.parametrize("checkpoint", [False, "unsafe", "safe"])
@pytest.mark.parametrize("targets", [["kv_b_proj"], "all-linear", r".*\.kv_b_proj"])
def test_initialize_rejects_absorbed_lora_before_engine_start(monkeypatch, tmp_path, checkpoint, targets):
    import json

    class FakeEngineArgs:
        def create_engine_config(self):
            return types.SimpleNamespace(
                structured_outputs_config=types.SimpleNamespace(enable_in_reasoning=False, reasoning_parser="deepseek_r1"),
                model_config=types.SimpleNamespace(
                    architecture="Glm5NextForConditionalGeneration", use_mla=True,
                    hf_text_config=types.SimpleNamespace(num_hidden_layers=2, model_type="glm5_next_text",
                                                        layer_types=["linear_attention", "deepseek_sparse_attention"])))

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, *args, **kwargs):
            pytest.fail("engine started before rejecting unservable LoRA")

    _install_fake_vllm(monkeypatch, FakeAsyncLLM)
    for name in ("_ensure_arctic_vllm_patches", "_ensure_router_replay_vllm_patches",
                 "ensure_xgrammar_stop_mask_fix", "ensure_spec_decode_grammar_fix"):
        monkeypatch.setattr(worker_mod, name, lambda: None)
    monkeypatch.setattr(worker_mod, "_create_async_engine_args", lambda *args, **kwargs: FakeEngineArgs())
    kwargs = {"model": "test-model", "lora_peft_config": {"target_modules": targets}}
    if checkpoint:
        (tmp_path / "adapter_config.json").write_text(json.dumps({
            "target_modules": targets if checkpoint == "unsafe" else ["q_b_proj"]}))
        kwargs["lora_adapter_path"] = str(tmp_path)
        if checkpoint == "unsafe":
            kwargs["lora_peft_config"] = {"target_modules": ["q_b_proj"]}
    worker = InferenceWorker.__ray_metadata__.modified_class()
    with pytest.raises(ValueError, match="kv_b_proj.*absorbed"):
        asyncio.run(worker.initialize(kwargs))


@pytest.mark.parametrize("use_mla,targets", [
    (True, ["q_b_proj"]), (True, r".*\.q_b_proj"), (False, ["kv_b_proj"]),
])
def test_mla_lora_guard_allows_servable_targets(use_mla, targets):
    config = types.SimpleNamespace(
        architecture="DeepseekV2ForCausalLM", use_mla=use_mla,
        hf_text_config=types.SimpleNamespace(num_hidden_layers=2))
    worker_mod._validate_mla_lora_targets(config, {"target_modules": targets})
    worker_mod._validate_mla_lora_targets(config, None)


def test_semip_rejects_synced_lora_before_restore():
    worker = InferenceWorker.__ray_metadata__.modified_class()
    with pytest.raises(ValueError, match="semi_p cannot serve LoRA"):
        asyncio.run(worker.initialize({"model": "test-model", "semi_p": True,
                                      "lora_peft_config": {"target_modules": ["kv_b_proj"]}}))


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

    assert type(exc) is RuntimeError
    assert str(exc) == f"{_GENERIC_STARTUP_ERROR}\nRoot cause in MainProcess: ValueError: {root}"
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


def test_unreadable_record_keeps_the_original_error(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=worker_mod.logger.name)

    def unreadable(directory):
        raise PermissionError(13, "Permission denied", directory)

    monkeypatch.setattr(worker_mod, "earliest_startup_error", unreadable)
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
    assert str(worker_mod._engine_startup_error(RuntimeError("generic"), str(tmp_path))) == (
        "generic\nRoot cause in EngineCore: RuntimeError: x"
    )
