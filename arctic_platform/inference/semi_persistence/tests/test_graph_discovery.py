"""Graph discovery must find every captured graph, or say that it did not.

After a restore, `ca_graph_rebind` rewrites the CustomAllreduce addresses baked
into every captured CUDA graph. Discovery has two sources -- the vLLM wrapper
registries and the runner's `cudagraph_manager` -- and if either silently
contributes nothing, the rebind patches a subset, reports success, and the
process faults on its first replay with `cudaErrorIllegalAddress`.

Measured on GLM-5.3 TP=8 under vLLM 0.30 (`~/r25-glmfix/inst1.log`):

    n_wrappers: 1, n_entries: 51, n_manager_graphs: 51, n_cudagraph_objs: 51

`n_cudagraph_objs` is the total and `n_manager_graphs` counts only what the
manager added that source 1 had not, so both being 51 means 51 wrapper entries
yielded ZERO graphs -- 51 of 4080. The entry scan was `vars(entry).values()`,
but the graph was never on `__dict__` to begin with: 0.30 keeps piecewise
segments behind bound methods (`segments.append(graph.replay)`), which no
isinstance walk of any depth reaches. A third source, the gc-unfreeze heap
scan, is what recovers them. Every rank reported `rebind graphs ok=True` with
`topo_readback ok: True` and `stale_ca_audit bad: 0` -- true statements about
the 1.2% of the graphs that were found.

The contrast case is 35B-A3B TP=4, which restores cleanly:

    n_wrappers: 41, n_entries: 2091, n_manager_graphs: 51, n_cudagraph_objs: 2142

2091 + 51: both sources contributing, 41 wrappers x 51 capture sizes.

So there are two properties here. Extraction must find the graph wherever the
entry keeps it, and -- because that will break again the next time vLLM moves
it -- discovery must notice a shortfall and refuse to call the rebind good.

On stubbing: `ca_graph_rebind` imports only `logging`, `os`, `sys` and `ctypes`
at module scope and does `import torch` inside the functions, so the module is
loaded for real and the stubs go on `sys.modules` around each call. Standing in
for an imported third-party dependency is a test double. Seeding a name the
module under test is itself supposed to define is the trap that let the
`_run_communicator_checkpoint_hook` NameError ship green, and is not what this
does: nothing below supplies a `ca_graph_rebind` global.

Run from the package directory::

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_graph_discovery.py -v
"""
from __future__ import annotations

import ast
import contextlib
import gc
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_REBIND = os.path.join(_PKG, "ca_graph_rebind.py")

# vLLM captures one graph per entry in `cudagraph_capture_sizes`, which is 51
# on both models below. The wrapper counts are what actually differ.
_N_SIZES = 51
_N_GLM_WRAPPERS = 1
_N_35B_WRAPPERS = 41


def _load_rebind():
    spec = importlib.util.spec_from_file_location("ca_graph_rebind_under_test",
                                                  _REBIND)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cgr = _load_rebind()


# --------------------------------------------------------------------------
# Doubles.
# --------------------------------------------------------------------------

class _FakeCUDAGraph:
    """Stands in for torch.cuda.CUDAGraph: a handle and an exec handle."""

    _next = 0x1000

    def __init__(self, handle=None):
        if handle is None:
            type(self)._next += 0x10
            handle = type(self)._next
        self._handle = handle

    def raw_cuda_graph(self):
        return self._handle

    def raw_cuda_graph_exec(self):
        return self._handle | 1


class _DictEntry:
    """The shape the old scan handled: graph straight on __dict__."""

    def __init__(self, g):
        self.cudagraph = g


class _SlotsEntry:
    """__slots__, so `vars()` raises and the old scan saw nothing. This is the
    GLM-5.3-under-0.30 shape."""

    __slots__ = ("cudagraph",)

    def __init__(self, g):
        self.cudagraph = g


class _ListEntry:
    """Segment list -- the old scan special-cased exactly this one nesting."""

    def __init__(self, g):
        self.segments = [None, g]


class _Holder:
    def __init__(self, g):
        self.cudagraph = g


class _NestedEntry:
    """Graph one object deeper; the old scan stopped at the first level."""

    def __init__(self, g):
        self.inner = _Holder(g)


_SIDE_TABLE: dict = {}


class _PropertyEntry:
    """Graph reachable only through a descriptor -- nothing on __dict__ but a
    key into a side table."""

    def __init__(self, g):
        self.key = len(_SIDE_TABLE)
        _SIDE_TABLE[self.key] = g

    @property
    def cudagraph(self):
        return _SIDE_TABLE[self.key]


class _AngryPropertyEntry:
    """A descriptor that raises must be stepped over, not propagated."""

    __slots__ = ("cudagraph",)

    def __init__(self, g):
        self.cudagraph = g

    @property
    def boom(self):
        raise RuntimeError("reading this property has side effects")


class _EmptyEntry:
    """An entry that genuinely holds no graph."""

    def __init__(self):
        self.shape = 128


class _BoundMethodEntry:
    """The vLLM 0.30 breakable shape, and the one no attribute walk can crack.

    ``BreakableCUDAGraphCapture._end_segment`` does
    ``self.segments.append(self._current_graph.replay)`` and then drops its last
    direct reference, so the only path to the graph is ``segments[i].__self__``.
    A bound method is not an instance of ``CUDAGraph``, so an isinstance-based
    scan of any depth walks straight past it -- which is why the heap fallback,
    not a deeper walk, is what recovers these.
    """

    def __init__(self, g):
        self.segments = [g.raw_cuda_graph]


class _FakeWrapper:
    def __init__(self, entries, attr="concrete_cudagraph_entries"):
        setattr(self, attr, dict(enumerate(entries)))


class _FakeManager:
    def __init__(self, graphs):
        self.graphs = dict(enumerate(graphs))


class _FakeRunner:
    def __init__(self, mgr=None):
        if mgr is not None:
            self.cudagraph_manager = mgr


class _FakeWorker:
    def __init__(self, runner):
        self.model_runner = runner


@contextlib.contextmanager
def _vllm(wrappers=(), breakable=(), graph_cls=None):
    """Stub `torch` and the two vLLM wrapper registries for one call.

    Withdrawn afterwards: `test_slots.py` and friends import real modules and
    which of us runs first is only collection order.

    ``graph_cls`` exists for the one case the heap fallback makes hard to stage:
    "no captured graph exists anywhere". Every `_FakeCUDAGraph` any other test
    left alive is a real hit for the scan, so emptiness has to be expressed as a
    class with no instances rather than as an empty registry.
    """
    names = ("torch", "torch.cuda", "vllm", "vllm.compilation",
             "vllm.compilation.cuda_graph",
             "vllm.compilation.breakable_cudagraph")
    saved = {n: sys.modules.get(n) for n in names}

    torch_mod = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.CUDAGraph = graph_cls or _FakeCUDAGraph
    torch_mod.cuda = cuda_mod

    cg = types.ModuleType("vllm.compilation.cuda_graph")
    cg.CUDAGraphWrapper = type("CUDAGraphWrapper", (),
                               {"_all_instances": list(wrappers)})
    bc = types.ModuleType("vllm.compilation.breakable_cudagraph")
    bc.BreakableCUDAGraphWrapper = type("BreakableCUDAGraphWrapper", (),
                                        {"_all_instances": list(breakable)})
    comp = types.ModuleType("vllm.compilation")
    comp.cuda_graph = cg
    comp.breakable_cudagraph = bc
    root = types.ModuleType("vllm")
    root.compilation = comp

    sys.modules.update({"torch": torch_mod, "torch.cuda": cuda_mod,
                        "vllm": root, "vllm.compilation": comp,
                        "vllm.compilation.cuda_graph": cg,
                        "vllm.compilation.breakable_cudagraph": bc})
    try:
        yield
    finally:
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


def _discover(wrappers=(), manager_graphs=None, breakable=()):
    mgr = None if manager_graphs is None else _FakeManager(manager_graphs)
    worker = _FakeWorker(_FakeRunner(mgr))
    with _vllm(wrappers, breakable):
        return cgr._find_captured_graphs(worker)


# --------------------------------------------------------------------------
# Extraction: find the graph wherever the entry keeps it.
# --------------------------------------------------------------------------

def test_the_glm_shape_no_longer_loses_the_wrapper_family():
    """1 wrapper x 51 slots entries + 51 manager graphs -> 102, not 51."""
    wrapper_graphs = [_FakeCUDAGraph() for _ in range(_N_SIZES)]
    mgr_graphs = [_FakeCUDAGraph() for _ in range(_N_SIZES)]
    wrappers = [_FakeWrapper([_SlotsEntry(g) for g in wrapper_graphs])]
    assert len(wrappers) == _N_GLM_WRAPPERS

    pairs, diag = _discover(wrappers, mgr_graphs)

    assert diag["n_entries"] == _N_SIZES
    assert diag["n_wrapper_graphs"] == _N_SIZES, (
        "the wrapper family was dropped again; this is the GLM-5.3 failure "
        f"(51 entries -> 0 graphs), diag={diag}")
    assert diag["n_manager_graphs"] == _N_SIZES
    assert diag["n_cudagraph_objs"] == 2 * _N_SIZES
    assert len(pairs) == 2 * _N_SIZES
    assert diag["complete"] is True
    assert diag["incomplete_why"] is None
    # Found honestly, not rescued by the heap scan.
    assert diag["n_gc_fallback_graphs"] == 0
    assert "gc_unfreeze_fallback" not in (diag["src"] or "")


def test_the_35b_shape_still_works():
    """41 wrappers x 51 dict entries + 51 manager graphs -> 2142."""
    wrappers = [_FakeWrapper([_DictEntry(_FakeCUDAGraph())
                              for _ in range(_N_SIZES)])
                for _ in range(_N_35B_WRAPPERS)]
    mgr_graphs = [_FakeCUDAGraph() for _ in range(_N_SIZES)]

    pairs, diag = _discover(wrappers, mgr_graphs)

    assert diag["n_wrappers"] == _N_35B_WRAPPERS
    assert diag["n_entries"] == _N_35B_WRAPPERS * _N_SIZES
    assert diag["n_manager_graphs"] == _N_SIZES
    assert diag["n_cudagraph_objs"] == _N_35B_WRAPPERS * _N_SIZES + _N_SIZES
    assert len(pairs) == 2142
    assert diag["complete"] is True


def test_every_entry_layout_yields_its_graph():
    """Each shape alone, so a regression names the layout that broke."""
    for factory in (_DictEntry, _SlotsEntry, _ListEntry, _NestedEntry,
                    _PropertyEntry, _AngryPropertyEntry):
        g = _FakeCUDAGraph()
        pairs, diag = _discover([_FakeWrapper([factory(g)])])
        assert diag["n_wrapper_graphs"] == 1, f"{factory.__name__} lost its graph"
        assert len(pairs) == 1, f"{factory.__name__} lost its graph"
        assert diag["complete"] is True, f"{factory.__name__}: {diag}"


def test_a_raising_property_does_not_abort_the_scan():
    g = _FakeCUDAGraph()
    pairs, diag = _discover([_FakeWrapper([_AngryPropertyEntry(g)])])
    assert len(pairs) == 1
    assert diag["err"] is None or "RuntimeError" not in diag["err"]


def test_breakable_wrappers_are_walked_too():
    g1, g2 = _FakeCUDAGraph(), _FakeCUDAGraph()
    pairs, diag = _discover(wrappers=[_FakeWrapper([_SlotsEntry(g1)])],
                            breakable=[_FakeWrapper([_SlotsEntry(g2)],
                                                    attr="entries")])
    assert diag["n_wrappers"] == 2
    assert diag["n_wrapper_graphs"] == 2
    assert len(pairs) == 2


def test_a_graph_in_both_sources_is_patched_once():
    """Applying an old->new map twice can misfire, so the dedupe is load-bearing."""
    shared = _FakeCUDAGraph()
    pairs, diag = _discover([_FakeWrapper([_SlotsEntry(shared)])], [shared])
    assert diag["n_wrapper_graphs"] == 1
    assert diag["n_manager_graphs"] == 0, "manager re-added a graph source 1 had"
    assert len(pairs) == 1


def test_two_objects_sharing_a_handle_are_patched_once():
    a, b = _FakeCUDAGraph(handle=0x4242), _FakeCUDAGraph(handle=0x4242)
    pairs, diag = _discover([_FakeWrapper([_SlotsEntry(a), _SlotsEntry(b)])])
    assert diag["n_wrapper_graphs"] == 2
    assert diag["n_dup_handles"] == 1
    assert len(pairs) == 1


def test_manager_only_discovery_is_complete():
    """No wrappers at all is legitimate; it must not read as a shortfall."""
    pairs, diag = _discover([], [_FakeCUDAGraph() for _ in range(_N_SIZES)])
    assert diag["n_entries"] == 0
    assert diag["complete"] is True
    assert len(pairs) == _N_SIZES


# --------------------------------------------------------------------------
# The census. Same two sources plus the same fallback, because it feeds the
# dump-side COLD IMAGE check: a census over 51 of 4080 graphs that finds all 51
# instantiated reports a warm image on 1.2% of the evidence.
# --------------------------------------------------------------------------

def _collect(wrappers=(), manager_graphs=None, breakable=(), graph_cls=None):
    mgr = None if manager_graphs is None else _FakeManager(manager_graphs)
    worker = _FakeWorker(_FakeRunner(mgr))
    with _vllm(wrappers, breakable, graph_cls):
        try:
            return cgr._collect_graph_entries(worker)
        finally:
            gc.unfreeze()   # production re-freezes; undo it for pytest


def test_the_census_recovers_the_breakable_family_it_used_to_miss():
    """The GLM-5.3 shape end to end: 51 entries yield nothing to the walk, the
    manager supplies 51, and the census used to stop there and call it a day."""
    hidden = [_FakeCUDAGraph() for _ in range(_N_SIZES)]
    mgr_graphs = [_FakeCUDAGraph() for _ in range(_N_SIZES)]
    wrappers = [_FakeWrapper([_BoundMethodEntry(g) for g in hidden])]

    pairs, diag = _collect(wrappers, mgr_graphs)

    assert diag["n_entries"] == _N_SIZES
    assert diag["n_wrapper_graphs"] == 0, (
        "a bound method is not a CUDAGraph; the walk cannot have found these")
    assert diag["n_entries_without_graph"] == _N_SIZES
    assert diag["n_manager_graphs"] == _N_SIZES
    assert diag["n_gc_fallback_graphs"] >= _N_SIZES, (
        f"the heap scan did not rescue the hidden family, diag={diag}")
    assert diag["complete"] is True
    found = {id(g) for _, g in pairs}
    assert found >= {id(g) for g in hidden}, (
        "the census is still blind to the graphs behind bound methods")


def test_gc_recovered_graphs_are_bucketed_as_such():
    """They arrive with no entry and so no shape. Folding them into a real shape
    would claim thousands of captures at a batch size that never ran."""
    hidden = [_FakeCUDAGraph() for _ in range(3)]
    wrappers = [_FakeWrapper([_BoundMethodEntry(g) for g in hidden])]

    pairs, _diag = _collect(wrappers, [_FakeCUDAGraph()])

    keys = {k for k, g in pairs if id(g) in {id(h) for h in hidden}}
    assert keys == {cgr._GC_SHAPE_KEY}, f"unexpected shape keys: {keys}"
    assert cgr._shape_label(cgr._GC_SHAPE_KEY) == cgr._GC_SHAPE_KEY


def test_a_complete_census_does_not_pay_for_the_heap_scan():
    """The fallback unfreezes the whole heap, so it must stay a fallback."""
    wrappers = [_FakeWrapper([_DictEntry(_FakeCUDAGraph())
                              for _ in range(_N_SIZES)])]
    _pairs, diag = _collect(wrappers, [_FakeCUDAGraph() for _ in range(3)])
    assert diag["complete"] is True
    assert diag["n_gc_fallback_graphs"] == 0


def test_the_census_reports_incompleteness_when_nothing_is_found():
    """Post-restore there is always a captured graph, so an empty result is a
    broken scan rather than an empty process, and must not read as complete."""

    class _NeverCaptured:
        pass

    _pairs, diag = _collect([], None, graph_cls=_NeverCaptured)
    assert diag["complete"] is False
    assert "including the gc fallback" in (diag["incomplete_why"] or "")


def test_graph_exec_census_counts_the_recovered_graphs():
    """The number the COLD IMAGE check reads. It has to be the whole set."""
    hidden = [_FakeCUDAGraph() for _ in range(_N_SIZES)]
    wrappers = [_FakeWrapper([_BoundMethodEntry(g) for g in hidden])]
    worker = _FakeWorker(_FakeRunner(_FakeManager([_FakeCUDAGraph()])))
    with _vllm(wrappers):
        try:
            out = cgr.graph_exec_census(worker)
        finally:
            gc.unfreeze()
    assert out["ok"] is True
    assert out["n_graphs"] >= _N_SIZES + 1
    assert out["discovery"]["complete"] is True
    assert cgr._GC_SHAPE_KEY in out["shapes_sample"]


# --------------------------------------------------------------------------
# The verdict: notice a shortfall instead of auditing the wrong set clean.
# --------------------------------------------------------------------------

def _diag(**kw):
    base = {"n_entries": 0, "n_wrapper_graphs": 0, "n_entries_without_graph": 0,
            "n_manager_graphs": 0}
    base.update(kw)
    return base


def test_entries_yielding_nothing_is_the_glm_signature_and_is_incomplete():
    ok, why = cgr._discovery_verdict(
        _diag(n_entries=51, n_wrapper_graphs=0, n_entries_without_graph=51,
              n_manager_graphs=51), 51)
    assert ok is False
    assert "51 wrapper entries yielded no CUDAGraph" in why


def test_a_partial_shortfall_is_also_incomplete():
    ok, why = cgr._discovery_verdict(
        _diag(n_entries=51, n_wrapper_graphs=40, n_entries_without_graph=11), 91)
    assert ok is False
    assert "11 of 51" in why


def test_finding_nothing_at_all_is_incomplete():
    ok, why = cgr._discovery_verdict(_diag(), 0)
    assert ok is False
    assert "no captured CUDA graphs" in why


def test_a_full_house_is_complete():
    ok, why = cgr._discovery_verdict(
        _diag(n_entries=2091, n_wrapper_graphs=2091, n_manager_graphs=51), 2142)
    assert ok is True
    assert why is None


def test_the_gc_fallback_now_fires_on_a_shortfall_not_only_on_a_barren_scan():
    """The old guard was `if not cgs`, so a manager-only result never reached
    it -- which is precisely why GLM's half-empty discovery sailed through."""
    mgr_graphs = [_FakeCUDAGraph() for _ in range(3)]
    wrappers = [_FakeWrapper([_EmptyEntry() for _ in range(3)])]
    try:
        _pairs, diag = _discover(wrappers, mgr_graphs)
    finally:
        gc.unfreeze()   # the production path re-freezes; undo it for pytest
    assert diag["n_entries"] == 3
    assert diag["n_entries_without_graph"] == 3
    assert "gc_unfreeze_fallback" in (diag["src"] or ""), (
        f"the shortfall did not trigger the fallback, src={diag['src']!r}")


# --------------------------------------------------------------------------
# The FlashInfer workspace probe. Diagnostic, but it monkey-patches a vLLM
# private inside the capture window, so it has to withdraw cleanly on every
# path -- a probe left installed rides into the CRIU image.
# --------------------------------------------------------------------------

_WORKER = os.path.join(_PKG, "_semip_worker.py")


@contextlib.contextmanager
def _fake_fiar(with_creator=True):
    path = "vllm.distributed.device_communicators"
    names = ["vllm", "vllm.distributed", path, f"{path}.flashinfer_all_reduce"]
    saved = {n: sys.modules.get(n) for n in names}
    mod = types.ModuleType(f"{path}.flashinfer_all_reduce")
    calls = []
    if with_creator:
        def _create_workspace(backend, *args, **kwargs):
            calls.append(backend)
            return f"ws:{backend}"

        mod._create_workspace = _create_workspace
    # `fired` is process-global so the probe speaks once per worker; reset it
    # here or the second test in this file silently asserts nothing.
    cgr._fiar_probe["fired"] = False
    try:
        for n in names[:-1]:
            sys.modules.setdefault(n, types.ModuleType(n))
        sys.modules[f"{path}.flashinfer_all_reduce"] = mod
        setattr(sys.modules[path], "flashinfer_all_reduce", mod)
        yield mod, calls
    finally:
        cgr.restore_fi_ar_workspace_probe()
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def test_the_probe_reports_the_first_allocation_only(capsys):
    with _fake_fiar() as (mod, calls):
        assert cgr.install_fi_ar_workspace_probe()["patched"] == [
            "_create_workspace"]
        assert mod._create_workspace("mnnvl", 8, 0) == "ws:mnnvl"
        assert mod._create_workspace("trtllm", 8, 0) == "ws:trtllm"
    out = capsys.readouterr().out
    assert out.count("[fi-ar-probe]") == 1, (
        "the probe must speak once, not once per allocation")
    assert "mnnvl" in out
    assert "test_the_probe_reports_the_first_allocation_only" in out, (
        "the stack is the entire point; without a caller frame it says nothing")
    assert calls == ["mnnvl", "trtllm"], "the probe swallowed an allocation"


def test_the_probe_withdraws_itself():
    with _fake_fiar() as (mod, _calls):
        original = mod._create_workspace
        cgr.install_fi_ar_workspace_probe()
        assert mod._create_workspace is not original
        cgr.restore_fi_ar_workspace_probe()
        assert mod._create_workspace is original


def test_installing_the_probe_twice_does_not_nest_it():
    """A second install must not wrap the wrapper; restoring once has to be
    enough to get back to vLLM's own function."""
    with _fake_fiar() as (mod, _calls):
        original = mod._create_workspace
        cgr.install_fi_ar_workspace_probe()
        wrapped = mod._create_workspace
        assert cgr.install_fi_ar_workspace_probe()["patched"] == [
            "_create_workspace"]
        assert mod._create_workspace is wrapped
        cgr.restore_fi_ar_workspace_probe()
        assert mod._create_workspace is original


def test_the_probe_is_a_no_op_on_pre_030():
    """No module, or a module without the private: say so, do not raise."""
    with _fake_fiar(with_creator=False):
        out = cgr.install_fi_ar_workspace_probe()
        assert out["patched"] == []
        assert out["missing"], "a failed install must name what it could not find"
        cgr.restore_fi_ar_workspace_probe()   # must not raise


def test_a_failed_install_is_distinguishable_from_a_quiet_capture():
    """The whole reason this returns a dict. Job 6e386668 printed nothing and
    there was no way to tell whether the allocator was never called or the
    patch was never applied -- on a run where the workspace was demonstrably
    built inside the probe's window."""
    with _fake_fiar(with_creator=False):
        assert cgr.install_fi_ar_workspace_probe()["patched"] == []
    with _fake_fiar(with_creator=True):
        assert cgr.install_fi_ar_workspace_probe()["patched"] != []


def test_the_suppress_switch_refuses_the_allocation(monkeypatch, capsys):
    """None is what vLLM already gets when a backend is unavailable, and every
    caller branches on it, so refusing is a supported outcome rather than a
    hack. If nothing builds the workspace there is nothing to detach, and the
    restore-side multicast re-map that cannot work never runs."""
    monkeypatch.setenv("SEMIP_SUPPRESS_FI_AR_WORKSPACE", "1")
    with _fake_fiar() as (mod, calls):
        cgr.install_fi_ar_workspace_probe()
        assert mod._create_workspace("mnnvl", 8, 0) is None
    assert calls == [], "the real allocator ran despite the kill switch"
    assert "suppress=True" in capsys.readouterr().out


def test_a_second_install_reports_the_live_suppress_state(monkeypatch):
    """The probe is installed from `init_device` and re-stated at warmup, so
    the second call's report is the one that shows up next to the capture. It
    has to say the kill switch is on when it is on; defaulting to False there
    would put a lie in the log this probe exists to keep honest."""
    monkeypatch.setenv("SEMIP_SUPPRESS_FI_AR_WORKSPACE", "1")
    with _fake_fiar() as (mod, _calls):
        first = cgr.install_fi_ar_workspace_probe()
        second = cgr.install_fi_ar_workspace_probe()
    assert first["suppress"] is True
    assert second["suppress"] is True, (
        "the re-install disowned the kill switch")
    assert second["patched"] == first["patched"]


def test_the_switch_is_off_unless_the_env_says_otherwise():
    with _fake_fiar() as (mod, calls):
        cgr.install_fi_ar_workspace_probe()
        assert mod._create_workspace("mnnvl", 8, 0) == "ws:mnnvl"
    assert calls == ["mnnvl"]


def _worker_method_src(name):
    tree = ast.parse(open(_WORKER).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in _semip_worker.py")


def test_the_probe_is_installed_before_the_first_capture():
    """GLM-5.3 runs two PIECEWISE capture passes and builds the workspace in
    the first, which is over before `compile_or_warm_up_model` is entered.
    Job 07fd79b4 proved it: probe installed (`patched=[all three]`,
    `suppress=True`), no `first ... call via` line, and `Initialized FlashInfer
    Allreduce` still 1. Scoping the install to the warmup hook cannot see that
    allocation, so `init_device` has to carry it."""
    fn = _worker_method_src("init_device")
    calls = [s for s in ast.walk(fn)
             if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
             and s.func.attr == "_semip_install_fi_ar_probe"]
    assert calls, (
        "init_device does not install the probe; the first capture pass runs "
        "unobserved and the kill switch cannot refuse it")


def test_the_installer_reports_what_stuck():
    """A diagnostic that returns a status nobody prints is not a diagnostic."""
    fn = _worker_method_src("_semip_install_fi_ar_probe")
    installs = [s for s in ast.walk(fn)
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                and s.func.attr == "install_fi_ar_workspace_probe"]
    assert installs, "the installer never installs the probe"
    prints = [s for s in ast.walk(fn)
              if isinstance(s, ast.Call) and isinstance(s.func, ast.Name)
              and s.func.id == "print"]
    assert prints, "the install result is never printed"


def test_the_worker_installs_the_probe_and_withdraws_it_in_finally():
    fn = _worker_method_src("compile_or_warm_up_model")
    installs = [s for s in ast.walk(fn)
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                and s.func.attr == "_semip_install_fi_ar_probe"]
    assert installs, "the worker never installs the probe"
    tries = [s for s in ast.walk(fn) if isinstance(s, ast.Try)]
    restored_in_finally = any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        and c.func.attr == "restore_fi_ar_workspace_probe"
        for t in tries for stmt in t.finalbody for c in ast.walk(stmt))
    assert restored_in_finally, (
        "the probe is withdrawn outside a finally; a capture that raises would "
        "leave a monkey-patched vLLM private in the CRIU image")


# --------------------------------------------------------------------------
# The gate. These paths need a CUDA context, so assert them on the source.
# --------------------------------------------------------------------------

def _function(name):
    tree = ast.parse(open(_REBIND).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in ca_graph_rebind.py")


def _assigns_ok_literal_true(fn_name):
    """True if the function contains a bare ``out["ok"] = True``."""
    for sub in ast.walk(_function(fn_name)):
        if not isinstance(sub, ast.Assign):
            continue
        for tgt in sub.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name) and tgt.value.id == "out"
                    and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value == "ok"
                    and isinstance(sub.value, ast.Constant)
                    and sub.value.value is True):
                return True
    return False


def test_the_rewrite_path_does_not_hardcode_success():
    assert not _assigns_ok_literal_true("rewrite_addrs_in_graphs"), (
        'rewrite_addrs_in_graphs sets out["ok"] = True unconditionally; a '
        "half-patched graph set would report a clean rebind and fault on the "
        "first replay")


def test_the_disk_wake_path_does_not_hardcode_success():
    """`addr_map == {}` returns through _force_recommit_ca_nodes, so gating only
    the rewrite branch would leave this one reporting success regardless."""
    assert not _assigns_ok_literal_true("_force_recommit_ca_nodes"), (
        '_force_recommit_ca_nodes sets out["ok"] = True unconditionally')


def test_both_exits_gate_on_the_discovery_verdict():
    for fn in ("rewrite_addrs_in_graphs", "_force_recommit_ca_nodes"):
        src = ast.dump(_function(fn))
        assert "'complete'" in src or '"complete"' in src, (
            f"{fn} never reads the discovery completeness verdict")


# --------------------------------------------------------------------------
# The CustomAllreduce copy path, and the window it has to cover.
#
# `CustomAllreduce.capture()` exports an IPC handle for every pointer recorded
# during capture (`cudaIpcGetMemHandle`, custom_all_reduce.cuh:164), and that
# legacy API refuses the VMM-backed memory `enable_sleep_mode` puts in play.
# The copy-path patch is what keeps the recorded pointer down to the one
# pre-registered staging buffer. 0.30 captures inside
# `determine_available_memory` as well as at warmup, so an install scoped to
# the warmup hook leaves the first capture exposed: job 37414847 died there on
# all eight ranks, from inside CUDACHECK, with no Python exception raised.
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _fake_ca(capturing):
    """Stub `torch` and vLLM's CustomAllreduce for one install/restore cycle.

    Yields the fake class and the list its `all_reduce` records into, so a
    test can read which dispatch the patch chose."""
    path = "vllm.distributed.device_communicators"
    names = ("torch", "torch.cuda", "vllm", "vllm.distributed", path,
             f"{path}.custom_all_reduce")
    saved = {n: sys.modules.get(n) for n in names}
    calls = []

    class _FakeCA:
        disabled = False

        def __init__(self):
            self._IS_CAPTURING = False

        def should_custom_ar(self, inp):
            return True

        def all_reduce(self, inp, *, out=None, registered=False):
            calls.append((inp, registered))
            return f"reduced:{inp}"

        def custom_all_reduce(self, inp):
            raise AssertionError("the unpatched dispatch should not run")

    torch_mod = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_current_stream_capturing = lambda: capturing
    torch_mod.cuda = cuda_mod
    torch_mod.empty_like = lambda inp: f"empty:{inp}"

    car = types.ModuleType(f"{path}.custom_all_reduce")
    car.CustomAllreduce = _FakeCA
    comm = types.ModuleType(path)
    comm.custom_all_reduce = car
    dist = types.ModuleType("vllm.distributed")
    dist.device_communicators = comm
    root = types.ModuleType("vllm")
    root.distributed = dist

    sys.modules.update({"torch": torch_mod, "torch.cuda": cuda_mod,
                        "vllm": root, "vllm.distributed": dist,
                        path: comm, f"{path}.custom_all_reduce": car})
    try:
        yield _FakeCA, calls
    finally:
        cgr.restore_force_copy_patch()
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


def test_the_forced_dispatch_copies_instead_of_registering():
    """Unpatched, `custom_all_reduce.py:448` passes `registered=True` while
    capturing, so the raw activation pointer is what lands in
    `graph_unreg_buffers_` and what the IPC export is asked to handle."""
    with _fake_ca(capturing=True) as (CA, calls):
        assert cgr.install_force_copy_patch()
        ca = CA()
        ca._IS_CAPTURING = True
        ca.custom_all_reduce("activation")
        # Both layers are patched -- `custom_all_reduce` chooses the copy
        # path and `all_reduce` forces it again -- so `calls` alone cannot
        # tell them apart. Observe the choice directly by standing a recorder
        # where the second layer sits.
        chosen = []
        CA.all_reduce = lambda self, inp, *, out=None, registered=False: (
            chosen.append(registered))
        ca.custom_all_reduce("activation")
    assert calls == [("activation", False)], (
        "the capture-time dispatch does not force the copy path; the graph "
        "records raw activations and the IPC export faults on any that are "
        "cumem-backed")
    assert chosen == [False], (
        "custom_all_reduce asks for the registered path during capture; it "
        "is only the all_reduce patch holding the copy path up")


def test_a_second_install_keeps_the_count_from_the_first_capture():
    """The warmup install is a re-statement, not a fresh start. If it reset
    the counters, `car_calls` at warmup would read 0 whether the profiling
    capture had been protected or had never run through the patch at all --
    and telling those two apart is the entire point of printing it."""
    with _fake_ca(capturing=True) as (CA, calls):
        cgr.install_force_copy_patch()
        ca = CA()
        ca._IS_CAPTURING = True
        ca.custom_all_reduce("first")
        ca.custom_all_reduce("second")
        assert cgr.force_copy_state()["car_calls"] == 2
        cgr.install_force_copy_patch()          # what warmup does
        state = cgr.force_copy_state()
    assert state["active"] is True
    assert state["car_calls"] == 2, (
        "re-installing reset the counters, so the warmup line can no longer "
        "say whether the profiling capture went through the copy path")


def test_force_copy_is_installed_before_the_first_capture():
    """0.30 runs a capture inside `determine_available_memory`
    (profile_cudagraph_memory -> capture_model(profile_only=True)) that is
    over before the warmup hook is entered. Job 37414847: eight ranks, eight
    `custom_all_reduce.cuh:164 'invalid argument'`, init FAILED at 310 s."""
    fn = _worker_method_src("init_device")
    calls = [s for s in ast.walk(fn)
             if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
             and s.func.attr == "_semip_install_force_copy"]
    assert calls, (
        "init_device does not force the copy path; the profiling capture "
        "records raw activation pointers and exports IPC handles for them")


def test_the_force_copy_installer_reports_what_stuck():
    """A diagnostic that returns a status nobody prints is not a diagnostic --
    and `install_force_copy_patch` returns a bool the old call site dropped."""
    fn = _worker_method_src("_semip_install_force_copy")
    installs = [s for s in ast.walk(fn)
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                and s.func.attr == "install_force_copy_patch"]
    assert installs, "the installer never installs the patch"
    reads = [s for s in ast.walk(fn)
             if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
             and s.func.attr == "force_copy_state"]
    assert reads, "the installer never reads the live state back"
    prints = [s for s in ast.walk(fn)
              if isinstance(s, ast.Call) and isinstance(s.func, ast.Name)
              and s.func.id == "print"]
    assert prints, "the install result is never printed"


def test_force_copy_is_withdrawn_in_a_finally():
    fn = _worker_method_src("compile_or_warm_up_model")
    tries = [s for s in ast.walk(fn) if isinstance(s, ast.Try)]
    restored_in_finally = any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        and c.func.attr == "restore_force_copy_patch"
        for t in tries for stmt in t.finalbody for c in ast.walk(stmt))
    assert restored_in_finally, (
        "the copy-path patch is withdrawn outside a finally; installing it "
        "from init_device widened the window, so a capture that raises would "
        "now leave a monkey-patched vLLM private in the CRIU image")


def test_keepgraph_stays_scoped_to_the_warmup_capture():
    """Only the copy-path patch moves earlier. `install_keepgraph_patch`
    suppresses instantiation as a side effect (torch's `capture_end`
    instantiates only when `keep_graph_` is false), and the profiling capture's
    graphs are discarded -- pulling it up changes what
    `instantiate_captured_graphs` is accounting for, for no benefit."""
    fn = _worker_method_src("init_device")
    calls = [s for s in ast.walk(fn)
             if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
             and s.func.attr == "install_keepgraph_patch"]
    assert not calls, (
        "init_device installs the keep-graph patch; the profiling capture's "
        "graphs would go uninstantiated and the census would count them")


# --------------------------------------------------------------------------
# The pre-armed fallback: skip registration for the profiling capture alone.
#
# `install_force_copy_patch` wraps `all_reduce` and `custom_all_reduce`. 0.30's
# CA also has `custom_all_gather` and `custom_reduce_scatter`, which reach the
# same `graph_unreg_buffers_`. If Flash-Next still faults at `cuh:164` with
# `car_calls > 0`, the copy path is not covering whatever recorded that
# pointer, and the answer for a capture whose graphs are discarded is to not
# register at all. Shipped dark so that branch costs a resubmit, not a build.
# --------------------------------------------------------------------------

def test_init_device_arms_the_profile_register_guard():
    fn = _worker_method_src("init_device")
    calls = [s for s in ast.walk(fn)
             if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
             and s.func.attr == "_semip_arm_profile_register_guard"]
    assert calls, (
        "init_device never arms the register guard, so the fallback cannot "
        "be reached from extra_env and testing it costs another image")


def test_the_profile_register_guard_is_off_by_default():
    """It changes what a capture context does on exit. It ships dark, so the
    build that tests the copy-path fix tests only the copy-path fix."""
    fn = _worker_method_src("_semip_arm_profile_register_guard")
    names = [n.value for n in ast.walk(fn)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "SEMIP_SUPPRESS_PROFILE_CA_REGISTER" in names, (
        "the guard reads no env var; it is either always on or always off")
    guard = next((s for s in fn.body if isinstance(s, ast.If)), None)
    assert guard is not None and any(
        isinstance(s, ast.Return) for s in guard.body), (
        "the env check does not return early, so the guard installs "
        "regardless of what the flag says")
    installs = [s for s in ast.walk(fn)
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                and s.func.attr == "install_suppress_register_patch"]
    assert installs, "the guard never installs anything"


def test_the_guard_is_released_before_the_warmup_capture():
    """Scope is the whole point. The warmup capture's graphs are the ones that
    survive into the image; only the profiling capture, whose graphs are
    discarded, may skip registration. Releasing in the `finally` alongside the
    other patches would silently widen it to both."""
    fn = _worker_method_src("compile_or_warm_up_model")
    released = [s for s in ast.walk(fn)
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                and s.func.attr == "_semip_release_profile_register_guard"]
    assert released, "the warmup hook never hands registration back"
    in_finally = [c for t in ast.walk(fn) if isinstance(t, ast.Try)
                  for stmt in t.finalbody for c in ast.walk(stmt)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                  and c.func.attr == "_semip_release_profile_register_guard"]
    assert not in_finally, (
        "the guard is released in the finally, so the warmup capture runs "
        "unregistered too and the image carries graphs whose peer table was "
        "never built")
    # The capture is the `super()` call inside the try/finally, not the one
    # in the import fallback, which returns before any of this is set up.
    captures = [c.lineno for t in ast.walk(fn)
                if isinstance(t, ast.Try) and t.finalbody
                for stmt in t.body for c in ast.walk(stmt)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "compile_or_warm_up_model"]
    assert captures and released[0].lineno < min(captures), (
        "registration is handed back after the warmup capture has run")


def test_the_skipped_count_is_readable_before_the_restore():
    """`restore_suppress_register_patch` clears `active`, and the count is the
    only evidence that the armed window is the window the capture landed in.
    A reader that restores first can report neither."""
    with _fake_ca(capturing=True) as (CA, _calls):
        CA.register_graph_buffers = lambda self: None
        try:
            assert cgr.install_suppress_register_patch()
            ca = CA()
            ca.register_graph_buffers()
            ca.register_graph_buffers()
            state = cgr.suppress_register_state()
            assert state["active"] is True
            assert state["skipped"] == 2, (
                "the guard did not count the registrations it swallowed")
            cgr.restore_suppress_register_patch()
            assert cgr.suppress_register_state()["active"] is False
        finally:
            cgr.restore_suppress_register_patch()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
