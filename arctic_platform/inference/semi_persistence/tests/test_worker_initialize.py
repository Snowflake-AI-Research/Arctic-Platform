"""``InferenceWorker.initialize``: the plain engine and the semi_p restore.

``server/worker.py`` imports ray, vLLM and half the server package at module
scope, none of which is installed where these tests run, so it is loaded from
its file with each of those imports stubbed. ``tests/inference/`` holds the
fuller worker tests, which need the real packages.

semi_p builds its config on the plain path and hands it, with the engine
kwargs, to ``semip_asyncllm.restore_and_attach`` where the plain path builds
its ``AsyncLLM``; these tests hold the plain path to what it did before.

Run from the package directory::

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_worker_initialize.py -v
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_INFERENCE = os.path.dirname(os.path.dirname(_HERE))
_WORKER = os.path.join(_INFERENCE, "server", "worker.py")

_RING = object()
_ADDRESS_IN_USE = RuntimeError("DistNetworkError: EADDRINUSE address already in use")


def _remote(cls):
    cls.__ray_metadata__ = types.SimpleNamespace(modified_class=cls)
    return cls


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _vllm_config(kwargs):
    ns = types.SimpleNamespace
    return ns(
        structured_outputs_config=ns(enable_in_reasoning=False),
        model_config=ns(skip_tokenizer_init=True, architecture="QwenForCausalLM"),
        kwargs=dict(kwargs),
    )


class _Env:
    """Everything initialize reaches, recording each call as (name, args)."""

    def __init__(self):
        self.calls = []
        self.no_think_tokens = False
        self.async_llm_errors = []
        self.attached = types.SimpleNamespace(get_tokenizer=lambda: None)

    def record(self, name, *args):
        self.calls.append((name, args))

    def called(self, name):
        return [args for n, args in self.calls if n == name]

    def engine_args(self, kwargs, **_):
        env = self

        class EngineArgs:
            def create_engine_config(self):
                env.record("create_engine_config", dict(kwargs))
                if kwargs.get("reasoning_parser") and env.no_think_tokens:
                    raise RuntimeError("parser requires think start/end tokens")
                return _vllm_config(kwargs)

        return EngineArgs()

    def from_vllm_config(self, vllm_config, **kwargs):
        self.record("AsyncLLM.from_vllm_config", vllm_config, kwargs)
        if self.async_llm_errors:
            raise self.async_llm_errors.pop(0)
        return types.SimpleNamespace(get_tokenizer=lambda: None)

    async def restore_and_attach(self, engine_kwargs, agents, vllm_config):
        self.record("restore_and_attach", dict(engine_kwargs), agents, vllm_config)
        return self.attached

    def matched_baseline(self):
        return _module(
            "matched_baseline", MATCH_ENV="SEMIP_MATCHED_BASELINE",
            requested=lambda: self.record("matched_baseline.requested") or True,
            apply=lambda engine_kwargs: self.record("matched_baseline.apply") or [])


@pytest.fixture
def env(monkeypatch):
    env = _Env()
    noop = lambda *a, **k: None  # noqa: E731
    stubs = {
        "ray": _module("ray", remote=_remote),
        "arctic_platform.inference.envs": _module(
            "envs", arctic_inference_effective_enabled=lambda *a: False),
        "arctic_platform.inference.server.chat": _module(
            "chat", ChatModel=object, resolve_chat_model=noop),
        "arctic_platform.inference.server.metrics": _module(
            "metrics", RingStatLogger=_RING, get_collector=noop),
        "arctic_platform.inference.server.streaming": _module(
            "streaming", StreamingWorkerMixin=object,
            stream_lifecycle_change=lambda method: method),
        "arctic_platform.inference.server.token_prompts": _module(
            "token_prompts", check_token_tensor=noop,
            is_token_tensor=lambda value: False),
        "arctic_platform.inference.vllm.dense_prompt_logprobs": _module(
            "dense", RESULT_KEY="dense", stage_sampling_params=noop,
            take_dense=noop),
        "arctic_platform.inference.vllm.spec_decode_grammar": _module(
            "spec", ensure_spec_decode_grammar_fix=noop),
        "arctic_platform.inference.vllm.startup_error": _module(
            "startup_error", ENV_VAR="ARCTIC_ENGINE_STARTUP_ERROR_DIR",
            earliest_startup_error=noop),
        "arctic_platform.inference.vllm.xgrammar_stop_mask": _module(
            "xgrammar", ensure_xgrammar_stop_mask_fix=noop),
        "arctic_platform.inference.utils": _module(
            "utils", require_supported_vllm_version=noop),
        "arctic_platform.inference.semi_persistence": _module(
            "semi_persistence", matched_baseline=env.matched_baseline()),
        "arctic_platform.inference.server.semip_asyncllm": _module(
            "semip_asyncllm", restore_and_attach=env.restore_and_attach),
        "vllm": _module("vllm"),
        "vllm.plugins": _module("vllm.plugins", load_general_plugins=noop),
        "vllm.v1": _module("vllm.v1"),
        "vllm.v1.engine": _module("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": _module(
            "vllm.v1.engine.async_llm",
            AsyncLLM=types.SimpleNamespace(from_vllm_config=env.from_vllm_config)),
    }
    stubs["vllm"].plugins = stubs["vllm.plugins"]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("worker_under_test", _WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "_create_async_engine_args", env.engine_args)
    monkeypatch.setattr(worker, "_resolve_chat_mode", lambda *a: (None, False))
    monkeypatch.setattr(worker, "_ensure_router_replay_vllm_patches", noop)
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_STARTUP_RETRY_BASE_S", "0")
    env.worker = worker
    return env


def _initialize(env, **engine_kwargs):
    worker = env.worker.InferenceWorker.__ray_metadata__.modified_class()
    asyncio.run(worker.initialize({"model": "/models/qwen", **engine_kwargs},
                                  model_id="job-1"))
    return worker


# --------------------------------------------------------------------------
# Plain
# --------------------------------------------------------------------------

def test_plain_builds_asyncllm_with_the_ring_logger(env, caplog):
    with caplog.at_level(logging.INFO):
        worker = _initialize(env)
    [(_, kwargs)] = env.called("AsyncLLM.from_vllm_config")
    assert kwargs == {"stat_loggers": [_RING]}
    assert worker.state == env.worker.WorkerLifecycleState.READY
    assert env.called("matched_baseline.apply") == [()]
    assert env.called("restore_and_attach") == []
    assert "initialized: model=/models/qwen" in caplog.text


def test_plain_retries_engine_startup_on_address_in_use(env, monkeypatch):
    monkeypatch.setenv("ARCTIC_VLLM_ENGINE_STARTUP_ATTEMPTS", "2")
    env.async_llm_errors = [_ADDRESS_IN_USE]
    worker = _initialize(env)
    assert len(env.called("AsyncLLM.from_vllm_config")) == 2
    assert len(env.called("create_engine_config")) == 2
    assert worker.state == env.worker.WorkerLifecycleState.READY


def test_plain_drops_a_reasoning_parser_the_tokenizer_cannot_run(env):
    env.no_think_tokens = True
    worker = _initialize(env, reasoning_parser="deepseek_r1")
    assert [kwargs.get("reasoning_parser") for (kwargs,) in env.called(
        "create_engine_config")] == ["deepseek_r1", None]
    assert worker._reasoning_parser is None


# --------------------------------------------------------------------------
# semi_p
# --------------------------------------------------------------------------

def test_semi_p_restores_where_the_plain_path_builds(env):
    worker = _initialize(env, semi_p=True, semi_p_agents=["n1"],
                         semi_p_model_dir="/old")
    [(kwargs, agents, vllm_config)] = env.called("restore_and_attach")
    assert kwargs == {
        "model": "/models/qwen",
        "worker_extension_cls":
            "arctic_platform.inference.server.weight_sync.WeightSyncExtension",
    }
    assert agents == ["n1"]
    assert vllm_config.kwargs["model"] == "/models/qwen"
    assert worker.llm is env.attached
    assert worker.state == env.worker.WorkerLifecycleState.READY
    assert env.called("AsyncLLM.from_vllm_config") == []


def test_semi_p_skips_the_plain_only_matched_baseline(env):
    _initialize(env, semi_p=True)
    assert not [n for n, _ in env.calls if n.startswith("matched_baseline")]


def test_semi_p_dumps_without_a_dropped_reasoning_parser(env):
    env.no_think_tokens = True
    _initialize(env, semi_p=True, reasoning_parser="deepseek_r1")
    [(kwargs, _, _)] = env.called("restore_and_attach")
    assert "reasoning_parser" not in kwargs


@pytest.mark.parametrize("kwargs,match", [
    ({"lora_adapter_path": "/lora"}, "LoRA adapter"),
    ({"enable_return_routed_experts": True}, "routed experts"),
])
def test_semi_p_refuses_what_a_restore_cannot_serve(env, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _initialize(env, semi_p=True, **kwargs)
    assert env.called("restore_and_attach") == []
