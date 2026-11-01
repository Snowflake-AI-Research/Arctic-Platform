"""An eager engine at TP>1 restores without a graph rebind, and nothing else does.

GLM-5.3's standard catalog config sets ``enforce_eager: true`` at TP=8. vLLM
turns that into ``cudagraph_mode = NONE`` and captures nothing, but every TP>1
restore still ends in ``rebind_graphs``, whose CustomAllreduce rewrite has no
graph to patch: discovery reports "no captured CUDA graphs found by any source",
``refresh_copy_path_slots`` returns ``skipped`` without ``ok``, and the
``rebind_graphs`` command raises ``semip rebind_graphs failed``.

Nothing in an eager process holds a pre-restore CustomAllreduce address -- its
all-reduces go through the CA object ``reinit_nccl`` just built -- so there is
nothing to rebind. The skip has to be narrow, though. A rebind that reports
success over graphs it did not patch is how GLM-5.3 died on its first replay
under 0.30, so the skip needs both the config saying eager *and* a census that
completed and found zero graphs; anything else goes through the rebind and is
judged by it.

``vllm_child`` imports torch and CUDA at module scope, so its functions are
lifted out of the source (as ``test_lora_host_copy`` does). ``ca_graph_rebind``
is loaded for real with ``torch`` and the vLLM wrapper registries stubbed on
``sys.modules`` (as ``test_graph_discovery`` does), so the census the skip
relies on is the production one.

Run from the package directory::

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_eager_rebind.py -v
"""
from __future__ import annotations

import ast
import enum
import gc
import importlib.util
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_CHILD = os.path.join(_PKG, "vllm_child.py")
_REBIND = os.path.join(_PKG, "ca_graph_rebind.py")
_NAMES = ("_semip_config_value", "_semip_tp_size", "_semip_cudagraphs_disabled",
          "_semip_confirmed_graphless", "_semip_rebind_graphs")


def _load_rebind():
    spec = importlib.util.spec_from_file_location("ca_graph_rebind_eager_test",
                                                  _REBIND)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cgr = _load_rebind()


# vLLM 0.32's enum, as far as the gate reads it: only ``.name``.
CUDAGraphMode = enum.Enum("CUDAGraphMode", {
    "NONE": 0, "PIECEWISE": 1, "FULL": 2, "FULL_DECODE_ONLY": (2, 0),
    "FULL_AND_PIECEWISE": (2, 1)})


class _Graph:
    """Stands in for torch.cuda.CUDAGraph. Per-test subclasses keep the heap
    scan honest: a graph another test left alive is not one of ours."""

    def raw_cuda_graph(self):
        return id(self)

    def raw_cuda_graph_exec(self):
        return id(self) | 1


class _Entry:
    def __init__(self, g):
        self.cudagraph = g


class _Wrapper:
    def __init__(self, graphs):
        self.concrete_cudagraph_entries = {i: _Entry(g)
                                           for i, g in enumerate(graphs)}


@pytest.fixture
def stubs(monkeypatch):
    """`torch` and the two wrapper registries, scoped to one test.

    Yields ``(graph_cls, wrappers)``; append to ``wrappers`` to register a
    CUDAGraphWrapper, instantiate ``graph_cls`` to make a graph exist."""
    graph_cls = type("CUDAGraph", (_Graph,), {})
    wrappers: list = []
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(CUDAGraph=graph_cls,
                                           synchronize=lambda *a: None)
    cg = types.ModuleType("vllm.compilation.cuda_graph")
    cg.CUDAGraphWrapper = type("CUDAGraphWrapper", (),
                               {"_all_instances": wrappers})
    bc = types.ModuleType("vllm.compilation.breakable_cudagraph")
    bc.BreakableCUDAGraphWrapper = type("BreakableCUDAGraphWrapper", (),
                                        {"_all_instances": []})
    comp = types.ModuleType("vllm.compilation")
    comp.cuda_graph, comp.breakable_cudagraph = cg, bc
    root = types.ModuleType("vllm")
    root.compilation = comp
    for name, mod in (("torch", torch_mod), ("vllm", root),
                      ("vllm.compilation", comp),
                      ("vllm.compilation.cuda_graph", cg),
                      ("vllm.compilation.breakable_cudagraph", bc)):
        monkeypatch.setitem(sys.modules, name, mod)
    try:
        yield graph_cls, wrappers
    finally:
        gc.unfreeze()   # the heap scan re-freezes; undo it for pytest


class _FakeRebind:
    """``ca_graph_rebind`` as the child sees it: the real census, and a
    ``rebind_after_reinit`` that records the call and returns a set verdict.

    The real one needs libcuda and a live CustomAllreduce; what is under test
    here is whether it is reached and whether its verdict is passed through."""

    def __init__(self, verdict, census=None):
        self.verdict = verdict
        self.calls = 0
        self._census = census

    def graph_exec_census(self, worker):
        if self._census is not None:
            return self._census
        return cgr.graph_exec_census(worker)

    def rebind_after_reinit(self, worker):
        self.calls += 1
        return dict(self.verdict)


# What `rebind_after_reinit` returned on an eager GLM-5.3 TP=8 restore, as read
# from the code path: nothing discovered, nothing refreshed, no `ok`.
_EAGER_VERDICT = {"enabled": True, "step": "rebind_after_reinit", "ok": False,
                  "rewrite": {"ok": False, "error": "incomplete graph discovery: "
                              "no captured CUDA graphs found by any source"},
                  "refresh_copy": {"skipped": "no CA _dp slots"}}


def _child(rebind):
    with open(_CHILD) as handle:
        tree = ast.parse(handle.read(), filename=_CHILD)
    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    assert {f.name for f in funcs} == set(_NAMES)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda *a: None))
    ns = {"torch": fake_torch, "_CA_REBIND_AVAILABLE": True, "ca_graph_rebind": rebind}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), _CHILD, "exec"), ns)
    return types.SimpleNamespace(**{n: ns[n] for n in _NAMES})


def _worker(tp=8, enforce_eager=False, mode=CUDAGraphMode.FULL_AND_PIECEWISE):
    cfg = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(tensor_parallel_size=tp),
        model_config=types.SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=types.SimpleNamespace(cudagraph_mode=mode))
    return types.SimpleNamespace(rank=0, vllm_config=cfg,
                                 model_runner=types.SimpleNamespace())


# --------------------------------------------------------------------------
# The skip: eager config and a census that found nothing.
# --------------------------------------------------------------------------

def test_an_eager_tp8_engine_rebinds_ok_without_touching_the_graphs(stubs):
    """The GLM-5.3 standard config. This raised `semip rebind_graphs failed`."""
    rebind = _FakeRebind(_EAGER_VERDICT)
    out = _child(rebind)._semip_rebind_graphs(
        _worker(enforce_eager=True, mode=CUDAGraphMode.NONE))
    assert out["ok"] is True
    assert out["skipped"] == "no_cuda_graphs"
    assert out["n_graphs"] == 0
    assert rebind.calls == 0, "the CA rewrite ran on a process with no graphs"


def test_cudagraph_mode_none_without_enforce_eager_is_also_eager(stubs):
    """`-cc.cudagraph_mode=none` is the other spelling of the same engine."""
    rebind = _FakeRebind(_EAGER_VERDICT)
    out = _child(rebind)._semip_rebind_graphs(
        _worker(enforce_eager=False, mode=CUDAGraphMode.NONE))
    assert out["ok"] is True
    assert out["skipped"] == "no_cuda_graphs"
    assert rebind.calls == 0


@pytest.mark.parametrize("eager,mode,expected", [
    (True, CUDAGraphMode.NONE, True),
    (True, CUDAGraphMode.FULL_AND_PIECEWISE, True),   # enforce_eager wins
    (False, CUDAGraphMode.NONE, True),
    (False, "NONE", True),
    (False, CUDAGraphMode.PIECEWISE, False),
    (False, CUDAGraphMode.FULL_DECODE_ONLY, False),
    (False, CUDAGraphMode.FULL_AND_PIECEWISE, False),
    (False, None, False),
])
def test_the_config_gate(eager, mode, expected):
    child = _child(_FakeRebind(_EAGER_VERDICT))
    assert child._semip_cudagraphs_disabled(
        _worker(enforce_eager=eager, mode=mode)) is expected


def test_a_worker_with_no_config_is_not_eager():
    child = _child(_FakeRebind(_EAGER_VERDICT))
    assert child._semip_cudagraphs_disabled(types.SimpleNamespace()) is False


# --------------------------------------------------------------------------
# What the skip must not swallow.
# --------------------------------------------------------------------------

def test_an_eager_config_with_a_live_graph_still_goes_through_the_rebind(stubs):
    """The config is what was asked for, not what the process holds."""
    graph_cls, wrappers = stubs
    wrappers.append(_Wrapper([graph_cls()]))
    rebind = _FakeRebind(_EAGER_VERDICT)
    out = _child(rebind)._semip_rebind_graphs(
        _worker(enforce_eager=True, mode=CUDAGraphMode.NONE))
    assert rebind.calls == 1
    assert out["ok"] is False
    assert "skipped" not in out


@pytest.mark.parametrize("verdict", [
    # Graphs present, CA rewrite failed.
    {"enabled": True, "ok": False, "error": "RuntimeError: cuGraphExec..."},
    # Graphs present, CustomAllreduce disabled after reinit.
    {"enabled": True, "skipped": "no ca_comm post-reinit"},
    # Graphs present, no snapshot recorded at the dump.
    {"enabled": True, "skipped": "no snapshot (snapshot step did not run?)"},
], ids=["rewrite_failed", "ca_disabled", "no_snapshot"])
def test_a_graph_engine_keeps_every_rebind_failure(stubs, verdict):
    graph_cls, wrappers = stubs
    wrappers.append(_Wrapper([graph_cls(), graph_cls()]))
    rebind = _FakeRebind(verdict)
    out = _child(rebind)._semip_rebind_graphs(_worker())
    assert rebind.calls == 1
    assert out["ok"] is False, f"a failed rebind was reported ok: {out}"
    assert out["ca_rebind"] == verdict


def test_a_graph_engine_whose_graphs_vanished_still_fails(stubs):
    """Graphs on in the config and none found is the broken-discovery case.
    It must keep failing rather than read as eager."""
    rebind = _FakeRebind(_EAGER_VERDICT)
    out = _child(rebind)._semip_rebind_graphs(_worker())
    assert rebind.calls == 1
    assert out["ok"] is False


def test_a_graph_engine_that_rebinds_cleanly_still_passes(stubs):
    graph_cls, wrappers = stubs
    wrappers.append(_Wrapper([graph_cls()]))
    rebind = _FakeRebind({"enabled": True, "ok": True})
    out = _child(rebind)._semip_rebind_graphs(_worker())
    assert rebind.calls == 1
    assert out["ok"] is True
    assert "skipped" not in out


@pytest.mark.parametrize("census", [
    {"ok": True, "n_graphs": 0, "discovery": {"gc_scanned": False}},
    {"ok": True, "n_graphs": 0, "discovery": {}},
    {"ok": False, "error": "RuntimeError: boom"},
    {"ok": True, "n_graphs": 3, "discovery": {"gc_scanned": True}},
], ids=["heap_scan_failed", "no_discovery", "census_failed", "graphs_found"])
def test_an_unproven_zero_does_not_skip(census):
    """Zero from a census that did not finish is not evidence of anything."""
    rebind = _FakeRebind(_EAGER_VERDICT, census=census)
    out = _child(rebind)._semip_rebind_graphs(
        _worker(enforce_eager=True, mode=CUDAGraphMode.NONE))
    assert rebind.calls == 1
    assert out["ok"] is False


def test_tp1_is_unchanged():
    rebind = _FakeRebind(_EAGER_VERDICT)
    out = _child(rebind)._semip_rebind_graphs(
        _worker(tp=1, enforce_eager=True, mode=CUDAGraphMode.NONE))
    assert out == {"ok": True, "recaptured": False, "skipped": "tp1_no_rebind",
                   "rank": 0}
    assert rebind.calls == 0


# --------------------------------------------------------------------------
# The census field the skip reads.
# --------------------------------------------------------------------------

def test_the_census_says_when_its_heap_scan_ran(stubs):
    out = cgr.graph_exec_census(_worker())
    assert out["ok"] is True
    assert out["n_graphs"] == 0
    assert out["discovery"]["gc_scanned"] is True
    assert out["discovery"]["complete"] is False


def test_the_census_says_when_its_heap_scan_failed(stubs, monkeypatch):
    """`import gc` inside the function resolves through sys.modules."""
    def _boom():
        raise RuntimeError("heap scan unavailable")

    monkeypatch.setitem(sys.modules, "gc", types.SimpleNamespace(
        unfreeze=lambda: None, freeze=lambda: None, get_objects=_boom))
    out = cgr.graph_exec_census(_worker())
    assert out["n_graphs"] == 0
    assert out["discovery"]["gc_scanned"] is False
    assert "gc fallback failed" in out["discovery"]["incomplete_why"]


def test_a_complete_registry_census_does_not_claim_a_heap_scan(stubs):
    graph_cls, wrappers = stubs
    wrappers.append(_Wrapper([graph_cls()]))
    out = cgr.graph_exec_census(_worker())
    assert out["n_graphs"] == 1
    assert out["discovery"]["complete"] is True
    assert out["discovery"]["gc_scanned"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
