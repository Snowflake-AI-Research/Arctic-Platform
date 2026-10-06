import asyncio
import sys
import types

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
