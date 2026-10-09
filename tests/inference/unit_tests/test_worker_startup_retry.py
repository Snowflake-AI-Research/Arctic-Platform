import asyncio
import os
import subprocess
import sys
import traceback
import types

import pytest

if "vllm" not in sys.modules:
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_module.__version__ = "0.30.0"
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
    vllm_mod.__version__ = "0.30.0"
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
                model_config=types.SimpleNamespace(skip_tokenizer_init=True),
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
                model_config=types.SimpleNamespace(skip_tokenizer_init=True),
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


def test_initialize_failure_carries_engine_console_tail(monkeypatch, tmp_path):
    class FakeEngineArgs:
        def create_engine_config(self):
            return types.SimpleNamespace(
                structured_outputs_config=types.SimpleNamespace(enable_in_reasoning=False),
                model_config=types.SimpleNamespace(skip_tokenizer_init=True),
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
    rendered = "".join(traceback.format_exception(info.value))
    assert "engine core starting" in rendered
    assert "ImportError: deep_gemm" in rendered
