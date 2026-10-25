"""Unit tests for the semi-p ``AsyncLLM`` front end and its attach.

``server/semip_frontend.py`` imports no vLLM, so it is loaded from its file and
driven against fakes: ``_SemiPAsyncLLMMixin`` over a fake ``AsyncLLM`` base, a
fake ``Instance`` that journals the commands it is sent, and a real
subprocess for the death watch. ``server/semip_asyncllm.py`` does import
vLLM; its ``restore_and_attach`` and ``attach`` are loaded with fake vLLM
modules in ``sys.modules``.

What none of this covers is the real ZMQ handshake between ``AsyncLLM`` and
the child's core, which needs vLLM and a GPU pod.

Run from the package directory::

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_semip_frontend.py -v
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import os
import subprocess
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_image_cache_key import _PKG, se  # noqa: E402

_SERVER = os.path.join(os.path.dirname(_PKG), "server")
_CHILD = os.path.join(_PKG, "vllm_child.py")
_FRONTEND_NAME = "arctic_platform.inference.server.semip_frontend"
_ENGINE_NAME = "arctic_platform.inference.server.semip_engine"


def _load_semip_frontend():
    """semip_frontend.py from its file; it imports only the standard library."""
    spec = importlib.util.spec_from_file_location(
        "semip_frontend_under_test", os.path.join(_SERVER, "semip_frontend.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fe = _load_semip_frontend()


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------

class _Inst:
    """The slice of ``Instance`` a front end drives, journaling every call."""

    def __init__(self, journal=None, pid=None, summary=None, fail=None):
        self.journal = [] if journal is None else journal
        self.pid = pid
        self.last_info = {}
        self.sent = []
        self._summary = summary or {}
        self._fail = fail
        self._failed = None

    def _do(self, cmd):
        self.journal.append(cmd)
        if self._fail == cmd:
            self._failed = cmd

    def _send(self, cmd, **kwargs):
        self.sent.append((cmd, kwargs))
        self._do(cmd)
        if cmd == "serve_core" and self._failed is None:
            self.last_info["serve_core"] = dict(self._summary)
        return self

    def wait(self):
        self.journal.append("wait")
        failed, self._failed = self._failed, None
        if failed is not None:
            raise RuntimeError(f"command {failed!r} failed: boom")
        return self

    def __getattr__(self, cmd):
        if cmd in ("stage", "sleep", "wake_up_weights", "restore_weights",
                   "wake_up_kv_cache", "teardown"):
            return lambda: self._do(cmd) or self
        raise AttributeError(cmd)


class _FakeAsyncLLM:
    """The ``AsyncLLM`` methods the mixin overrides or calls."""

    def __init__(self, journal=None, rpc_error=None):
        self.journal = [] if journal is None else journal
        self.rpcs = []
        self.rpc_error = rpc_error
        self.engine_core = types.SimpleNamespace(
            resources=types.SimpleNamespace(engine_dead=False),
            shutdown=lambda: self.journal.append("core.shutdown"))

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpcs.append((method, args, kwargs))
        if self.rpc_error is not None:
            raise self.rpc_error
        return [f"{method}-ok"]

    def shutdown(self):
        self.journal.append("shutdown")


class _Front(fe._SemiPAsyncLLMMixin, _FakeAsyncLLM):
    pass


def _front(inst=None, agents=(), **kwargs):
    inst = inst if inst is not None else _Inst()
    front = _Front(journal=inst.journal, **kwargs)
    front._init_semi_p(inst, list(agents))
    return front, inst


_WAKE = ["wake_up_weights", "restore_weights", "wake_up_kv_cache", "wait"]


# --------------------------------------------------------------------------
# Sleep, wake and the other RPCs
# --------------------------------------------------------------------------

def test_sleep_and_wake_rpcs_are_semi_ps():
    front, inst = _front()
    assert asyncio.run(front.collective_rpc("sleep", kwargs={"level": 2})) == [None]
    assert inst.journal == ["sleep", "wait"]
    inst.journal.clear()
    assert asyncio.run(front.collective_rpc(
        "wake_up", kwargs={"tags": ["weights"]})) == [None]
    assert inst.journal == _WAKE
    assert front.rpcs == []


def test_asyncllm_sleep_and_wake_methods_are_semi_ps_too():
    front, inst = _front()
    asyncio.run(front.sleep(level=1))
    asyncio.run(front.wake_up(tags=["kv_cache"]))
    assert inst.journal == ["sleep", "wait"] + _WAKE
    assert front.rpcs == []


@pytest.mark.parametrize("method", ["_arl_cuda_sync", "close_weight_sync",
                                    "sync_weights"])
def test_other_rpcs_go_to_vllm(method):
    front, inst = _front()
    out = asyncio.run(front.collective_rpc(method, args=(1,), kwargs={"k": 2}))
    assert out == [f"{method}-ok"]
    assert front.rpcs == [(method, (1,), {"k": 2})]
    assert inst.journal == []


def test_a_weight_sync_restages_the_host_copy_before_the_next_sleep():
    front, inst = _front()
    asyncio.run(front.collective_rpc("sync_weights_broadcast"))
    asyncio.run(front.collective_rpc("sleep"))
    assert inst.journal == ["stage", "wait", "sleep", "wait"]
    inst.journal.clear()
    asyncio.run(front.collective_rpc("wake_up"))
    asyncio.run(front.collective_rpc("sleep"))
    assert inst.journal == _WAKE + ["sleep", "wait"]


def test_rpcs_that_write_no_weights_leave_the_host_copy_alone():
    front, inst = _front()
    asyncio.run(front.collective_rpc("_arl_cuda_sync"))
    asyncio.run(front.collective_rpc("close_weight_sync"))
    asyncio.run(front.collective_rpc("sleep"))
    assert inst.journal == ["sleep", "wait"]


def test_a_failed_weight_sync_still_restages():
    front, inst = _front(rpc_error=RuntimeError("nccl"))
    with pytest.raises(RuntimeError, match="nccl"):
        asyncio.run(front.collective_rpc("sync_weights"))
    asyncio.run(front.collective_rpc("sleep"))
    assert inst.journal[:2] == ["stage", "wait"]


def test_a_failed_stage_does_not_sleep_on_a_stale_copy():
    front, inst = _front(inst=_Inst(fail="stage"))
    front._host_copy_stale = True
    with pytest.raises(RuntimeError, match="stage"):
        asyncio.run(front.collective_rpc("sleep"))
    assert "sleep" not in inst.journal
    assert front._host_copy_stale is True


# --------------------------------------------------------------------------
# Teardown
# --------------------------------------------------------------------------

@pytest.fixture
def fake_ray(monkeypatch):
    journal = []
    ray = types.ModuleType("ray")
    ray.get = lambda ref, timeout=None: journal.append(("ray.get", ref, timeout))
    monkeypatch.setitem(sys.modules, "ray", ray)
    return journal


class _Agent:
    def __init__(self, name):
        self.teardown = types.SimpleNamespace(remote=lambda: f"teardown-{name}")


def test_close_shuts_the_front_end_down_before_the_child(fake_ray):
    front, inst = _front(agents=[_Agent("n1"), _Agent("n2")])
    front.close()
    assert inst.journal == ["shutdown", "teardown", "wait"]
    assert fake_ray == [("ray.get", "teardown-n1", 120),
                        ("ray.get", "teardown-n2", 120)]
    assert front._inst is None and front._agents == []
    front.close()
    assert inst.journal == ["shutdown", "teardown", "wait", "shutdown"]


def test_close_tears_the_child_down_even_if_shutdown_raises():
    front, inst = _front()

    def broken():
        raise RuntimeError("zmq")

    front.shutdown = broken
    with pytest.raises(RuntimeError, match="zmq"):
        front.close()
    assert inst.journal == ["teardown", "wait"]


# --------------------------------------------------------------------------
# The death watch
# --------------------------------------------------------------------------

def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.01)


def test_pid_alive():
    assert fe._pid_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not fe._pid_alive(proc.pid)


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="needs /proc")
def test_a_zombie_counts_as_dead():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    _until(lambda: open(f"/proc/{proc.pid}/stat").read().rsplit(")", 1)[1]
           .split()[0] == "Z")
    assert not fe._pid_alive(proc.pid)
    proc.wait()


def test_death_watch_fires_once_when_the_child_dies(caplog):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    front, _ = _front(inst=_Inst(pid=proc.pid))
    deaths = []
    with caplog.at_level(logging.WARNING, logger=fe.logger.name):
        front._start_death_watch(lambda: deaths.append(1), interval_s=0.01)
        proc.kill()
        proc.wait()
        _until(lambda: deaths)
        time.sleep(0.05)
    assert deaths == [1]
    assert f"pid {proc.pid}" in caplog.text


def test_death_watch_is_quiet_after_close():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    front, _ = _front(inst=_Inst(pid=proc.pid))
    deaths = []
    front._start_death_watch(lambda: deaths.append(1), interval_s=0.01)
    front.close()
    proc.kill()
    proc.wait()
    time.sleep(0.1)
    assert deaths == []


def test_no_pid_no_watch(caplog):
    front, _ = _front()
    with caplog.at_level(logging.WARNING, logger=fe.logger.name):
        front._start_death_watch(lambda: None)
    assert front._death_watch_stop is None
    assert "no child pid" in caplog.text


def test_a_dead_child_fails_the_engine_core_as_vllm_would():
    front, inst = _front()
    front._on_child_death()
    assert front.engine_core.resources.engine_dead is True
    assert inst.journal == ["core.shutdown"]


# --------------------------------------------------------------------------
# Config comparison
# --------------------------------------------------------------------------

def _load_child_summary():
    with open(_CHILD) as handle:
        tree = ast.parse(handle.read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_semip_core_summary")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _CHILD, "exec"), ns)
    return ns["_semip_core_summary"]


def _vllm_config(**over):
    ns = types.SimpleNamespace
    cfg = ns(
        model_config=ns(model="/models/qwen", tokenizer="/models/qwen",
                        dtype="torch.bfloat16", max_model_len=32768),
        parallel_config=ns(tensor_parallel_size=8,
                           worker_extension_cls="arl.WeightSync"),
        scheduler_config=ns(max_num_seqs=256, max_num_batched_tokens=8192),
        cache_config=ns(block_size=16, num_gpu_blocks=None,
                        enable_prefix_caching=True),
        structured_outputs_config=ns(reasoning_parser="", backend="auto"),
        speculative_config=None,
        lora_config=None,
    )
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.parametrize("over", [
    {},
    {"speculative_config": types.SimpleNamespace(method="mtp",
                                                 num_speculative_tokens=2),
     "lora_config": object()},
    {"structured_outputs_config": None},
])
def test_front_end_reads_the_config_exactly_as_the_child_does(over):
    child = _load_child_summary()
    cfg = _vllm_config(**over)
    assert fe.core_summary(cfg) == child(cfg)


def test_identity_fields_are_all_in_the_summary():
    summary = fe.core_summary(_vllm_config())
    assert set(fe._IDENTITY_FIELDS) <= set(summary)


def test_matching_configs_compare_clean():
    summary = fe.core_summary(_vllm_config())
    assert fe.compare_core_config(summary, dict(summary)) == []


def test_synced_fields_are_not_compared():
    core = fe.core_summary(_vllm_config())
    front = dict(core, max_model_len=4096, num_gpu_blocks=0, block_size=32,
                 enable_prefix_caching=False)
    assert fe.compare_core_config(core, front) == []


def test_each_differing_identity_field_is_named_with_both_values():
    core = fe.core_summary(_vllm_config())
    front = dict(core, max_num_batched_tokens=16384, dtype="float16")
    assert fe.compare_core_config(core, front) == [
        "dtype: core='bfloat16' front end='float16'",
        "max_num_batched_tokens: core=8192 front end=16384",
    ]


def test_a_field_the_core_did_not_report_is_a_mismatch():
    front = fe.core_summary(_vllm_config())
    core = {k: v for k, v in front.items() if k != "worker_extension_cls"}
    assert fe.compare_core_config(core, front) == [
        "worker_extension_cls: core='<not reported>' "
        "front end='arl.WeightSync'"]


def test_serve_core_sends_the_addresses_and_returns_the_summary():
    inst = _Inst(summary={"model": "/m"})
    assert fe.serve_core(inst, "ipc://in", "ipc://out") == {"model": "/m"}
    assert inst.sent == [("serve_core", {"input_address": "ipc://in",
                                         "output_address": "ipc://out"})]
    assert inst.journal == ["serve_core", "wait"]


def test_serve_core_raises_what_the_child_refused():
    inst = _Inst(fail="serve_core")
    with pytest.raises(RuntimeError, match="serve_core"):
        fe.serve_core(inst, "ipc://in", "ipc://out")


# --------------------------------------------------------------------------
# attach, with fake vLLM modules
# --------------------------------------------------------------------------

class _FakeVllmAsyncLLM(_FakeAsyncLLM):
    built = []
    fail = None

    @classmethod
    def from_vllm_config(cls, vllm_config, **kwargs):
        if cls.fail is not None:
            raise cls.fail
        cls.built.append((vllm_config, kwargs))
        return cls()


_RING = object()


@pytest.fixture
def asyncllm(monkeypatch):
    """semip_asyncllm.py, loaded against fake vLLM and metrics modules."""
    paths = iter(f"ipc:///tmp/semip-test-{i}" for i in range(100))
    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.utils": types.ModuleType("vllm.utils"),
        "vllm.utils.network_utils": types.ModuleType("vllm.utils.network_utils"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.async_llm": types.ModuleType("vllm.v1.engine.async_llm"),
        "arctic_platform.inference.server.metrics":
            types.ModuleType("arctic_platform.inference.server.metrics"),
    }
    mods["vllm.utils.network_utils"].get_open_zmq_ipc_path = lambda: next(paths)
    mods["vllm.v1.engine.async_llm"].AsyncLLM = _FakeVllmAsyncLLM
    mods["arctic_platform.inference.server.metrics"].RingStatLogger = _RING
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, _FRONTEND_NAME, fe)
    _FakeVllmAsyncLLM.built = []
    _FakeVllmAsyncLLM.fail = None
    spec = importlib.util.spec_from_file_location(
        "semip_asyncllm_under_test", os.path.join(_SERVER, "semip_asyncllm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _handle(inst, agents=()):
    return se.RestoredEngine(inst, agents=tuple(agents))


def test_attach_serves_the_core_then_builds_asyncllm_on_its_addresses(
        asyncllm, caplog):
    cfg = _vllm_config()
    inst = _Inst(pid=os.getpid(), summary=fe.core_summary(cfg))
    with caplog.at_level(logging.INFO, logger=fe.logger.name):
        llm = asyncio.run(asyncllm.attach(_handle(inst), cfg))
    try:
        [(cmd, addresses)] = inst.sent
        assert cmd == "serve_core"
        [(built_cfg, kwargs)] = _FakeVllmAsyncLLM.built
        assert built_cfg is cfg
        assert kwargs == {"stat_loggers": [_RING],
                          "client_addresses": addresses}
        assert addresses["input_address"] != addresses["output_address"]
        assert isinstance(llm, asyncllm.SemiPAsyncLLM)
        assert llm._inst is inst
        assert llm._death_watch_stop is not None
        assert "front end attached in" in caplog.text
        assert "serve_core" in caplog.text and "AsyncLLM" in caplog.text
    finally:
        llm._stop_death_watch()


def test_attach_refuses_a_core_that_disagrees_and_tears_it_down(
        asyncllm, caplog):
    cfg = _vllm_config()
    summary = dict(fe.core_summary(cfg), model="/models/other")
    inst = _Inst(summary=summary)
    with caplog.at_level(logging.ERROR, logger=fe.logger.name):
        with pytest.raises(RuntimeError, match="model: core='/models/other'"):
            asyncio.run(asyncllm.attach(_handle(inst), cfg))
    assert _FakeVllmAsyncLLM.built == []
    assert inst.journal[-2:] == ["teardown", "wait"]
    assert "/models/other" in caplog.text


def test_attach_tears_down_when_asyncllm_fails(asyncllm, fake_ray):
    cfg = _vllm_config()
    inst = _Inst(summary=fe.core_summary(cfg))
    _FakeVllmAsyncLLM.fail = TimeoutError("no ready message")
    with pytest.raises(TimeoutError):
        asyncio.run(asyncllm.attach(_handle(inst, [_Agent("n1")]), cfg))
    assert inst.journal[-2:] == ["teardown", "wait"]
    assert fake_ray == [("ray.get", "teardown-n1", 120)]


def test_attach_tears_down_when_the_child_refuses_serve_core(asyncllm):
    inst = _Inst(fail="serve_core")
    with pytest.raises(RuntimeError, match="serve_core"):
        asyncio.run(asyncllm.attach(_handle(inst), _vllm_config()))
    assert inst.journal[-2:] == ["teardown", "wait"]
    assert _FakeVllmAsyncLLM.built == []


@pytest.fixture
def restored(monkeypatch):
    """A fake ``semip_engine.restore_engine``, recording what it was asked for."""
    calls = []
    inst = _Inst()

    def restore_engine(engine_kwargs, agents):
        calls.append((engine_kwargs, agents))
        return _handle(inst, agents or ())

    monkeypatch.setitem(sys.modules, _ENGINE_NAME, types.SimpleNamespace(
        restore_engine=restore_engine))
    return calls


def _restore_and_attach(asyncllm, engine_kwargs, cfg, agents=None):
    async def attach(handle, vllm_config):
        return handle, vllm_config

    asyncllm.attach = attach
    return asyncio.run(asyncllm.restore_and_attach(engine_kwargs, agents, cfg))


def test_restore_and_attach_pins_the_front_ends_limits_into_the_dump(
        asyncllm, restored):
    cfg = _vllm_config()
    engine_kwargs = {"model": "/models/qwen", "max_num_seqs": None}
    handle, attached_cfg = _restore_and_attach(asyncllm, engine_kwargs, cfg,
                                               agents=["n1"])
    [(kwargs, agents)] = restored
    assert kwargs == {"model": "/models/qwen", "max_num_seqs": 256,
                      "max_num_batched_tokens": 8192,
                      "disable_log_stats": False}
    assert agents == ["n1"] and handle.agents == ("n1",)
    assert attached_cfg is cfg
    assert engine_kwargs == {"model": "/models/qwen", "max_num_seqs": None}


def test_restore_and_attach_keeps_what_the_job_set(asyncllm, restored):
    _restore_and_attach(asyncllm, {"model": "/m", "max_num_seqs": 64,
                                   "disable_log_stats": True}, _vllm_config())
    [(kwargs, _)] = restored
    assert kwargs["max_num_seqs"] == 64
    assert kwargs["max_num_batched_tokens"] == 8192
    assert kwargs["disable_log_stats"] is True


def test_restore_and_attach_builds_the_core_with_the_front_ends_reasoner(
        asyncllm, restored):
    cfg = _vllm_config(structured_outputs_config=types.SimpleNamespace(
        reasoning_parser="qwen3", backend="auto"))
    _restore_and_attach(asyncllm, {"model": "/m"}, cfg)
    [(kwargs, _)] = restored
    assert kwargs["reasoning_parser"] == "qwen3"


@pytest.mark.parametrize("over", [{}, {"structured_outputs_config": None}])
def test_restore_and_attach_adds_no_reasoner_the_front_end_lacks(
        asyncllm, restored, over):
    _restore_and_attach(asyncllm, {"model": "/m"}, _vllm_config(**over))
    [(kwargs, _)] = restored
    assert "reasoning_parser" not in kwargs


def _imported(path, anywhere=False):
    """Module names ``path`` imports, at module scope or (``anywhere``) at all."""
    with open(path) as handle:
        tree = ast.parse(handle.read())
    nodes = list(ast.walk(tree)) if anywhere else tree.body
    names = [a.name for n in nodes if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in nodes if isinstance(n, ast.ImportFrom)]
    return names


def test_semip_frontend_imports_no_vllm_at_module_scope():
    names = _imported(os.path.join(_SERVER, "semip_frontend.py"))
    assert not [n for n in names if n.split(".")[0] in ("vllm", "torch", "ray")]


def test_semip_engine_never_imports_the_asyncllm_front_end():
    names = _imported(os.path.join(_SERVER, "semip_engine.py"), anywhere=True)
    assert not [n for n in names if "semip_asyncllm" in n]
