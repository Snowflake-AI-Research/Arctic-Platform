"""Graphs holding NCCL kernels are dropped before ``destroy_nccl``, on any node count.

NCCL counts every collective a CUDA graph captures against its communicator
(``localPersistentRefs``), and ``ncclCommAbort`` waits for that count to reach
zero, which only destroying the graph does. A kept graph holding one NCCL
kernel therefore runs ``destroy_nccl`` into its 60 s deadline, leaves the
comm's peer memory mapped, and produces an image whose ``cuda_restore`` fails
with CUresult=801. Measured on DeepSeek-V4.1-Flash TP=8 (Engram's PyNccl TP
all-gathers: 816 NCCL nodes per rank) and Qwen3.8-Flash-Next TP=8 with
``disable_custom_all_reduce`` (1157 per rank); every passing model held 0.

So the single-node dump counts NCCL kernel nodes in the kept graphs on every
rank and drops them when any rank holds one or cannot say; the image records
the decision, and the restore recaptures instead of rebinding. A model whose
graphs hold none must keep today's preserve-and-rebind path untouched.

``vllm_child`` and ``instance`` import torch, CUDA and NVML at module scope,
so their functions are lifted out of the source; ``ca_graph_rebind`` is loaded
for real with torch, the vLLM wrapper registry and libcuda stubbed.

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_nccl_graph_gate.py -v
"""
from __future__ import annotations

import ast
import contextlib
import gc
import importlib.util
import json
import logging
import os
import queue
import sys
import threading
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_INFERENCE = os.path.dirname(_PKG)
_CHILD = os.path.join(_PKG, "vllm_child.py")
_INSTANCE = os.path.join(_PKG, "instance.py")
_REBIND = os.path.join(_PKG, "ca_graph_rebind.py")
_WORKER = os.path.join(_PKG, "worker.py")
_ENGINE = os.path.join(_INFERENCE, "server", "semip_engine.py")


def _tree(path):
    with open(path) as handle:
        return ast.parse(handle.read(), filename=path)


def _function(path, name, cls=None):
    scope = _tree(path)
    if cls is not None:
        scope = next(n for n in ast.walk(scope)
                     if isinstance(n, ast.ClassDef) and n.name == cls)
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _lift(path, names, namespace=None, cls=None):
    """Compile the named functions out of *path* into one namespace."""
    ns = dict(namespace or {})
    body = [_function(path, n, cls=cls) for n in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# The decision, from every rank's census
# ---------------------------------------------------------------------------

_DECIDE = _lift(_CHILD, ["_semip_graph_drop_decision",
                         "_semip_graph_drop_message"])
decide = _DECIDE["_semip_graph_drop_decision"]
message = _DECIDE["_semip_graph_drop_message"]


def _census(rank, n, complete=True, ok=True, **extra):
    return {"rank": rank, "ok": ok, "complete": complete, "n_nccl_nodes": n,
            **extra}


def test_graphs_without_nccl_are_kept():
    """Every passing model: CustomAllreduce only, so preserve and rebind."""
    out = decide([_census(r, 0) for r in range(8)], 8)
    assert out["graphs_dropped"] is False
    assert out["graphs_dropped_reason"] is None
    assert out["graph_nccl_nodes"] == {str(r): 0 for r in range(8)}
    assert "keeping them for rebind_graphs" in message(out)


def test_one_nccl_node_on_one_rank_drops_on_all():
    counts = [0] * 8
    counts[3] = 816
    out = decide([_census(r, n) for r, n in enumerate(counts)], 8)
    assert out["graphs_dropped"] is True
    assert out["graphs_dropped_reason"] == "nccl_in_graphs"
    assert out["graph_nccl_nodes"]["3"] == 816
    line = message(out)
    assert ("kept graphs hold 816 NCCL kernel node(s) on rank 3 (max 816); "
            "dropping graphs before destroy_nccl, restore will recapture"
            in line)


@pytest.mark.parametrize("censuses", [
    [_census(0, 0), _census(1, 0, complete=False,
                            incomplete_why="2 graph(s) without a retained "
                                           "topology")],
    [_census(0, 0), _census(1, 0, ok=False)],
    [_census(0, 0), _census(1, None)],
    [_census(0, 0), "boom"],
    [_census(0, 0)],                       # a rank never answered
    [],                                    # the census RPC itself failed
], ids=["incomplete", "not_ok", "no_count", "not_a_dict", "missing_rank",
        "rpc_failed"])
def test_an_unproven_zero_drops(censuses):
    """A partial walk proves nothing; dropping costs a recapture, keeping a
    graph that held NCCL costs the image."""
    out = decide(censuses, 2)
    assert out["graphs_dropped"] is True
    assert out["graphs_dropped_reason"] == "census_incomplete"
    assert out["census_unknown"]
    assert "treating it as NCCL present" in message(out, censuses)


def test_a_partial_walk_that_found_nccl_is_reported_as_nccl():
    out = decide([_census(0, 12, complete=False), _census(1, 0)], 2)
    assert out["graphs_dropped_reason"] == "nccl_in_graphs"
    assert out["graph_nccl_nodes"] == {"0": 12, "1": 0}


def test_graphless_ranks_keep_their_graphs():
    """An eager engine's ranks answer without a walk; that is a full reading."""
    out = decide([_census(r, 0, skipped="no_cuda_graphs") for r in range(4)], 4)
    assert out["graphs_dropped"] is False


def test_the_recorded_fields_are_json():
    out = decide([_census(0, 1157), _census(1, 1157)], 2)
    assert json.loads(json.dumps(out)) == out


# ---------------------------------------------------------------------------
# The per-rank census RPC
# ---------------------------------------------------------------------------

class _FakeRebind:
    def __init__(self, n_graphs=0, nccl=0):
        self.census_calls = 0
        self.n_graphs = n_graphs
        self.nccl = nccl

    def graph_exec_census(self, worker):
        return {"ok": True, "n_graphs": self.n_graphs,
                "discovery": {"gc_scanned": True}}

    def nccl_node_census(self, worker):
        self.census_calls += 1
        return {"ok": True, "complete": True, "n_nccl_nodes": self.nccl}


def _rank_census(rebind):
    return _lift(_CHILD, ["_semip_config_value", "_semip_tp_size",
                          "_semip_cudagraphs_disabled",
                          "_semip_confirmed_graphless",
                          "_semip_nccl_graph_census"],
                 {"_CA_REBIND_AVAILABLE": True, "ca_graph_rebind": rebind}
                 )["_semip_nccl_graph_census"]


def _worker(tp=8, eager=False):
    cfg = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(tensor_parallel_size=tp),
        model_config=types.SimpleNamespace(enforce_eager=eager),
        compilation_config=types.SimpleNamespace(cudagraph_mode="FULL"))
    return types.SimpleNamespace(rank=2, vllm_config=cfg)


def test_tp1_runs_no_census():
    rebind = _FakeRebind(nccl=5)
    out = _rank_census(rebind)(_worker(tp=1))
    assert out["skipped"] == "tp1" and out["n_nccl_nodes"] == 0
    assert rebind.census_calls == 0


def test_a_proven_graphless_eager_rank_runs_no_walk():
    rebind = _FakeRebind(n_graphs=0)
    out = _rank_census(rebind)(_worker(eager=True))
    assert out["skipped"] == "no_cuda_graphs" and out["complete"] is True
    assert rebind.census_calls == 0


def test_an_eager_config_holding_graphs_is_walked():
    rebind = _FakeRebind(n_graphs=3, nccl=7)
    out = _rank_census(rebind)(_worker(eager=True))
    assert rebind.census_calls == 1
    assert out["n_nccl_nodes"] == 7 and out["rank"] == 2


# ---------------------------------------------------------------------------
# The walk itself: ca_graph_rebind.nccl_node_census over a fake libcuda
# ---------------------------------------------------------------------------

def _load_rebind():
    spec = importlib.util.spec_from_file_location("ca_graph_rebind_gate_test",
                                                  _REBIND)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cgr = _load_rebind()

_KERNEL, _CHILD_GRAPH, _CONDITIONAL, _MEMCPY = 0, 4, 13, 1


class _FakeCu:
    """libcuda as the census reads it. ``graphs`` maps a graph handle to its
    nodes; a node is ``(type, payload)``, payload being a kernel name or a
    child graph handle (``None`` for one the driver will not open)."""

    def __init__(self, graphs):
        self.graphs = graphs
        self.nodes = {}
        self.funcs = {}
        for handle, nodes in graphs.items():
            self.graphs[handle] = []
            for kind, payload in nodes:
                node = 1000 + len(self.nodes)
                self.nodes[node] = (kind, payload)
                self.graphs[handle].append(node)

    def cuGraphGetNodes(self, graph, arr, num):
        nodes = self.graphs[graph]
        num._obj.value = len(nodes)
        if arr is not None:
            for i, node in enumerate(nodes):
                arr[i] = node
        return 0

    def cuGraphNodeGetType(self, node, t):
        t._obj.value = self.nodes[node][0]
        return 0

    def cuGraphKernelNodeGetParams_v2(self, node, params):
        name = self.nodes[node][1]
        func = self.funcs.setdefault(name, 50 + len(self.funcs))
        params._obj.func = func
        return 0

    def cuFuncGetName(self, namep, func):
        name = next(n for n, f in self.funcs.items() if f == func)
        namep._obj.value = name.encode()
        return 0

    def cuGraphChildGraphNodeGetGraph(self, node, sub):
        child = self.nodes[node][1]
        if child is None:
            return 1
        sub._obj.value = child
        return 0


class _Graph:
    def __init__(self, handle):
        self.handle = handle

    def raw_cuda_graph(self):
        if self.handle is None:
            raise RuntimeError("keep_graph was False")
        return self.handle

    def raw_cuda_graph_exec(self):
        return 1


class _Entry:
    def __init__(self, g):
        self.cudagraph = g


class _Wrapper:
    def __init__(self, graphs):
        self.concrete_cudagraph_entries = {i: _Entry(g)
                                           for i, g in enumerate(graphs)}


@pytest.fixture
def libcuda(monkeypatch):
    """Registers graphs and their nodes; returns ``add(graph_nodes)``."""
    graph_cls = type("CUDAGraph", (_Graph,), {})
    wrappers: list = []
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(CUDAGraph=graph_cls)
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
    state = {}

    def add(topology, roots):
        cu = _FakeCu(dict(topology))
        monkeypatch.setattr(cgr, "_cu", lambda: cu)
        wrappers.append(_Wrapper([graph_cls(h) for h in roots]))
        state["cu"] = cu

    try:
        yield add
    finally:
        gc.unfreeze()


_CA = (_KERNEL, "cross_device_reduce_1stage<bf16>")
_GEMM = (_KERNEL, "sm90_xmma_gemm_bf16")
_AG = (_KERNEL, "ncclDevKernel_AllGather_RING_LL")
_AR = (_KERNEL, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL")


def test_custom_allreduce_graphs_count_zero(libcuda):
    libcuda({1: [_CA, _GEMM, (_MEMCPY, None)], 2: [_CA, _GEMM]}, [1, 2])
    out = cgr.nccl_node_census(None)
    assert out["complete"] is True, out["incomplete_why"]
    assert out["n_nccl_nodes"] == 0
    assert out["n_kernel_nodes"] == 4


def test_nccl_kernels_are_counted_by_name(libcuda):
    libcuda({1: [_AG, _GEMM, _AG], 2: [_AR, _CA]}, [1, 2])
    out = cgr.nccl_node_census(None)
    assert out["complete"] is True
    assert out["n_nccl_nodes"] == 3
    assert out["nccl_kernels"] == {_AG[1]: 2, _AR[1]: 1}


def test_child_graphs_are_walked(libcuda):
    """A collective one level down is still a persistent ref on the comm."""
    libcuda({1: [_GEMM, (_CHILD_GRAPH, 7)], 7: [(_CHILD_GRAPH, 8)],
             8: [_AG]}, [1])
    out = cgr.nccl_node_census(None)
    assert out["complete"] is True
    assert out["n_child_graphs"] == 2
    assert out["n_nccl_nodes"] == 1


@pytest.mark.parametrize("topology,roots,why", [
    ({1: [_CA, (_CHILD_GRAPH, None)]}, [1], "unreadable"),
    ({1: [_CA, (_CONDITIONAL, None)]}, [1], "unreadable"),
    ({1: [_CA]}, [1, None], "without a retained topology"),
], ids=["child_unreadable", "conditional", "no_topology"])
def test_what_cannot_be_read_makes_the_census_incomplete(libcuda, topology,
                                                         roots, why):
    libcuda(topology, roots)
    out = cgr.nccl_node_census(None)
    assert out["n_nccl_nodes"] == 0
    assert out["complete"] is False
    assert why in out["incomplete_why"]


def test_no_graphs_after_a_heap_scan_is_a_full_reading(libcuda):
    libcuda({}, [])
    out = cgr.nccl_node_census(None)
    assert out["complete"] is True and out["n_graphs"] == 0


# ---------------------------------------------------------------------------
# Instance: what cuda_checkpoint sends, what criu_dump records
# ---------------------------------------------------------------------------

def _instance(n_gpus=8, nnodes=1, node_rank=0):
    sent = []
    inst = types.SimpleNamespace(
        n_gpus=n_gpus, nnodes=nnodes, node_rank=node_rank,
        _log=lambda *a: None,
        _send=lambda cmd, **kw: sent.append((cmd, kw)) or inst,
        _resolve_image_dir=lambda f=None: f or "/m/image",
        vllm_config={"model": "m"}, model_dir="/m", total_gpu_bytes=1,
        pinned_cpu_bytes=0, max_pinned_bytes_per_worker=0)
    return inst, sent


_INST = _lift(_INSTANCE, ["cuda_checkpoint", "criu_dump"], cls="Instance")


@pytest.mark.parametrize("n_gpus,nnodes,node_rank,expected", [
    (8, 1, 0, [("drop_graphs_if_nccl", {}), ("destroy_nccl", {}),
               ("cuda_checkpoint", {})]),
    (8, 2, 0, [("drop_graphs", {"reason": "multinode"}),
               ("destroy_nccl", {}), ("cuda_checkpoint", {})]),
    (8, 2, 1, [("cuda_checkpoint", {})]),     # the leader's RPC reaches us
    (1, 1, 0, [("cuda_checkpoint", {})]),     # TP=1: no census, no teardown
], ids=["single_node", "multinode_leader", "multinode_follower", "tp1"])
def test_cuda_checkpoint_gates_the_drop(n_gpus, nnodes, node_rank, expected):
    inst, sent = _instance(n_gpus, nnodes, node_rank)
    _INST["cuda_checkpoint"](inst)
    assert sent == expected


def _dump_meta(nnodes, meta_extra=None):
    inst, sent = _instance(nnodes=nnodes)
    _INST["criu_dump"](inst, meta_extra=meta_extra)
    (cmd, kwargs), = sent
    assert cmd == "criu_dump"
    return kwargs["meta_extra"]


def test_a_multinode_image_records_its_drop_on_every_node():
    """A follower's child never sees drop_graphs, so its meta has to say so."""
    meta = _dump_meta(2)
    assert meta["graphs_dropped"] is True
    assert meta["graphs_dropped_reason"] == "multinode"


def test_a_single_node_dump_leaves_the_field_to_the_child():
    assert "graphs_dropped" not in _dump_meta(1)


# ---------------------------------------------------------------------------
# The worker: the child's decision goes into meta.json, a failure refuses
# ---------------------------------------------------------------------------

def _load_worker():
    torch = types.ModuleType("torch")
    torch.multiprocessing = types.ModuleType("torch.multiprocessing")
    stubs = {"pynvml": types.ModuleType("pynvml"), "torch": torch,
             "torch.multiprocessing": torch.multiprocessing,
             "semip_logging": types.ModuleType("semip_logging")}
    added = [name for name in stubs if name not in sys.modules]
    for name in added:
        sys.modules[name] = stubs[name]
    sys.path.insert(0, _PKG)
    try:
        spec = importlib.util.spec_from_file_location("worker_nccl_gate", _WORKER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(_PKG)
        for name in added:
            sys.modules.pop(name, None)
    return module


w = _load_worker()

_DROP = {"graphs_dropped": True, "graphs_dropped_reason": "nccl_in_graphs",
         "graph_nccl_nodes": {"0": 816, "1": 816}}


class _Pipe:
    """The child's end: each command answers with its scripted (error, info)."""

    def __init__(self, replies):
        self.replies = dict(replies)
        self.sent = []
        self._queue = []

    def send(self, message):
        cmd, _ = message
        self.sent.append(cmd)
        error, info = self.replies.get(cmd, (None, {}))
        self._queue.append((cmd, 0.0, error, info))

    def recv(self):
        return self._queue.pop(0)

    def poll(self, timeout=0):
        return False

    def close(self):
        pass


def _run(commands, replies, monkeypatch):
    saved = []
    checkpoints = []
    monkeypatch.setattr(w, "semip_logging", types.SimpleNamespace(
        worker=lambda *a: logging.getLogger("nccl_gate")), raising=False)
    monkeypatch.setattr(w, "_worker_checkpoint",
                        lambda pid, unlock=False: checkpoints.append(pid) or [pid])
    monkeypatch.setattr(w, "_kill_process_tree", lambda pid: None)
    monkeypatch.setattr(w, "_resolve_fd_resource", lambda pid, fd: "pipe:[1]")
    monkeypatch.setattr(w, "_inet_census", lambda pids: [])
    monkeypatch.setattr(w, "_get_descendant_pids", lambda pid: [])
    monkeypatch.setattr(w, "_reap_dumped_child", lambda *a: None)
    monkeypatch.setattr(w, "_reap_orphaned_descendants", lambda *a: None)

    def _save(child_pid, image_dir, fd, resource, gpus, meta_extra=None):
        saved.append(dict(meta_extra or {}))
        return dict(meta_extra or {})

    monkeypatch.setattr(w, "_worker_criu_save", _save)
    commands_q, results_q = queue.Queue(), queue.Queue()
    for cmd in [*commands, ("teardown", {})]:
        commands_q.put(cmd if isinstance(cmd, tuple) else (cmd, {}))
    counter = types.SimpleNamespace(value=0, get_lock=contextlib.nullcontext)
    pipe = _Pipe({"get_pipe_fd": (None, {"pipe_fd": 7}), **replies})
    w._child_thread("t", [0], 4242, pipe, commands_q, results_q, counter)
    results = []
    while not results_q.empty():
        cmd, _, error, _ = results_q.get()
        results.append((cmd, error))
    return results, saved, checkpoints, pipe.sent


_DUMP = ("criu_dump", {"filename": "/m/image",
                       "meta_extra": {"vllm_config": {"model": "m"},
                                      "image_ref": "img"}})


def test_the_childs_drop_is_recorded_into_meta(monkeypatch):
    results, saved, checkpoints, _ = _run(
        ["drop_graphs_if_nccl", "destroy_nccl", "cuda_checkpoint", _DUMP],
        {"drop_graphs_if_nccl": (None, {"graph_drop": _DROP})}, monkeypatch)
    assert dict(results)["criu_dump"] is None
    (meta,) = saved
    assert meta["image_ref"] == "img"
    assert {k: meta[k] for k in _DROP} == _DROP


def test_a_kept_decision_is_recorded_too(monkeypatch):
    kept = dict(_DROP, graphs_dropped=False, graphs_dropped_reason=None,
                graph_nccl_nodes={"0": 0, "1": 0})
    _, saved, _, _ = _run(
        ["drop_graphs_if_nccl", "destroy_nccl", "cuda_checkpoint", _DUMP],
        {"drop_graphs_if_nccl": (None, {"graph_drop": kept})}, monkeypatch)
    assert saved[0]["graphs_dropped"] is False


def test_a_multinode_leader_records_the_childs_drop(monkeypatch):
    drop = {"graphs_dropped": True, "graphs_dropped_reason": "multinode",
            "graph_nccl_nodes": None}
    dump = ("criu_dump", {"filename": "/m/image",
                          "meta_extra": {"graphs_dropped": True,
                                         "graphs_dropped_reason": "multinode"}})
    _, saved, _, _ = _run(["drop_graphs", "destroy_nccl", "cuda_checkpoint",
                           dump],
                          {"drop_graphs": (None, {"graph_drop": drop})},
                          monkeypatch)
    assert saved[0]["graphs_dropped_reason"] == "multinode"


def test_a_tp1_dump_records_no_drop(monkeypatch):
    _, saved, _, _ = _run(["cuda_checkpoint", _DUMP], {}, monkeypatch)
    assert "graphs_dropped" not in saved[0]


@pytest.mark.parametrize("failing", ["destroy_nccl", "drop_graphs_if_nccl"])
def test_a_failed_teardown_refuses_the_checkpoint_and_the_dump(monkeypatch,
                                                               failing):
    results, saved, checkpoints, sent = _run(
        ["drop_graphs_if_nccl", "destroy_nccl", "cuda_checkpoint", _DUMP],
        {failing: ("RuntimeError: ncclCommAbort did not return for tp:0",
                   {})}, monkeypatch)
    assert checkpoints == [] and saved == []
    assert "get_pipe_fd" not in sent
    by_cmd = dict(results)
    assert "NCCL teardown failed" in by_cmd["cuda_checkpoint"]
    assert "tp:0" in by_cmd["criu_dump"]


# ---------------------------------------------------------------------------
# The restore: recapture when the image says so, rebind otherwise
# ---------------------------------------------------------------------------

def _load_semip_engine():
    pkg = types.ModuleType("arctic_platform.inference")
    pkg.__path__ = []
    semip = types.ModuleType("arctic_platform.inference.semi_persistence")
    semip.Instance = object
    sys.modules.setdefault("arctic_platform.inference", pkg)
    sys.modules.setdefault("arctic_platform.inference.semi_persistence", semip)
    spec = importlib.util.spec_from_file_location("semip_engine_nccl_gate",
                                                  _ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


se = _load_semip_engine()


@pytest.mark.parametrize("meta,expected", [
    ({}, False),                                        # every image so far
    ({"graphs_dropped": False}, False),
    ({"graphs_dropped": True,
      "graphs_dropped_reason": "nccl_in_graphs"}, True),
    ({"graphs_dropped": True,
      "graphs_dropped_reason": "census_incomplete"}, True),
])
def test_the_meta_field_and_its_old_image_default(meta, expected):
    assert se._graphs_dropped(meta) is expected


def _restore_calls(tmp_path, monkeypatch, meta_extra):
    image = tmp_path / "image"
    image.mkdir()
    meta = {"vllm_config": {"model": "m", "tensor_parallel_size": 8},
            "gpus": list(range(8)), **meta_extra}
    (image / "meta.json").write_text(json.dumps(meta))
    calls = []

    class _Inst:
        def __init__(self, baked, model_dir):
            self.last_info = {}

        def __getattr__(self, name):
            def _call(*a, **k):
                if name != "wait":
                    calls.append(name)
                return self
            return _call

    monkeypatch.setattr(se, "Instance", _Inst)
    for name in ("_log_config_divergence", "_check_device_visibility",
                 "_check_gpu_placement", "_check_staging_budget"):
        monkeypatch.setattr(se, name, lambda *a, **k: None)
    monkeypatch.setattr(se, "_unprivileged_mismatch", lambda meta: None)
    monkeypatch.setattr(se, "_dump_placement", lambda meta, gpus, d: gpus)
    se._restore(str(tmp_path), {}, list(range(8)))
    return calls


def test_an_image_that_kept_its_graphs_rebinds(tmp_path, monkeypatch):
    calls = _restore_calls(tmp_path, monkeypatch, {})
    assert calls[-2:] == ["wake_up_kv_cache", "rebind_graphs"]
    assert "recapture_graphs" not in calls


def test_an_image_that_dropped_its_graphs_recaptures(tmp_path, monkeypatch):
    """Same order as the multi-node restore: NCCL first, the capture last."""
    calls = _restore_calls(tmp_path, monkeypatch, {
        "graphs_dropped": True, "graphs_dropped_reason": "nccl_in_graphs",
        "graph_nccl_nodes": {"0": 816}})
    assert "rebind_graphs" not in calls
    assert calls == ["criu_restore", "cuda_restore", "reinit_nccl", "attach",
                     "load_weights", "wake_up_weights",
                     "plan_restore_weights", "restore_weights",
                     "wake_up_kv_cache", "recapture_graphs"]


# ---------------------------------------------------------------------------
# The backstop: an abort that does not return fails the teardown
# ---------------------------------------------------------------------------

def test_the_concurrent_abort_names_the_comms_still_running():
    release = threading.Event()

    def _abort(ptr):
        if ptr == 2:
            release.wait(10)
        return 0

    fn = _lift(_CHILD, ["_nccl_abort_comms_concurrent"],
               {"_nccl_abort_comm": _abort})["_nccl_abort_comms_concurrent"]
    try:
        results, stuck = fn([("world:0", 1), ("tp:0", 2)], timeout=0.3)
    finally:
        release.set()
    assert stuck == ["tp:0"]
    assert results == {"world:0": ("ok", 0)}


def test_the_concurrent_abort_reports_nothing_stuck_when_all_return():
    fn = _lift(_CHILD, ["_nccl_abort_comms_concurrent"],
               {"_nccl_abort_comm": lambda ptr: 0}
               )["_nccl_abort_comms_concurrent"]
    results, stuck = fn([("tp:0", 1), ("ep:0", 2)], timeout=5)
    assert stuck == []
    assert results == {"tp:0": ("ok", 0), "ep:0": ("ok", 0)}


@pytest.fixture
def destroy(monkeypatch):
    """``_destroy_nccl`` over one tp group, with the abort's verdict scripted."""
    pynccl = types.SimpleNamespace(comm=0x1234)
    group = types.SimpleNamespace(device_communicator=types.SimpleNamespace(
        pynccl_comm=pynccl, ca_comm=None))
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps._groups = {"tp:0": lambda: group}
    ps.destroy_model_parallel = lambda: None
    ps.destroy_distributed_environment = lambda: None
    dist_pkg = types.ModuleType("vllm.distributed")
    dist_pkg.parallel_state = ps
    vllm = types.ModuleType("vllm")
    vllm.distributed = dist_pkg
    tdist = types.ModuleType("torch.distributed")
    tdist.is_initialized = lambda: True
    torch_mod = types.ModuleType("torch")
    torch_mod.distributed = tdist
    for name, mod in (("torch", torch_mod), ("torch.distributed", tdist),
                      ("vllm", vllm), ("vllm.distributed", dist_pkg),
                      ("vllm.distributed.parallel_state", ps)):
        monkeypatch.setitem(sys.modules, name, mod)

    def _make(verdict):
        ns = _lift(_CHILD, ["_semip_config_value", "_semip_tp_size",
                            "_destroy_nccl"], {
            "ctypes": __import__("ctypes"),
            "_mark_inet_sockets_rst": lambda where: None,
            "_run_communicator_checkpoint_hook": lambda *a: None,
            "_is_multinode": lambda: False,
            "_nccl_abort_comms_concurrent": lambda targets: verdict,
            "_abort_torch_process_groups": lambda: None,
            "_clear_fd_backed_nccl_env": lambda: None,
            "_PG_ABORTED": False})
        return ns, pynccl

    return _make


def _tp8_worker():
    cfg = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(tensor_parallel_size=8))
    return types.SimpleNamespace(rank=0, vllm_config=cfg)


def test_destroy_nccl_raises_on_a_stuck_comm(destroy):
    ns, pynccl = destroy(({}, ["tp:0"]))
    with pytest.raises(RuntimeError) as err:
        ns["_destroy_nccl"](_tp8_worker())
    text = str(err.value)
    assert "tp:0" in text
    assert "captured NCCL collectives" in text
    assert "801" in text
    assert ns["_PG_ABORTED"] is False, "a failed teardown must not claim the abort"


def test_destroy_nccl_passes_when_every_abort_returns(destroy):
    ns, pynccl = destroy(({"tp:0": ("ok", 0)}, []))
    assert ns["_destroy_nccl"](_tp8_worker()) == {}
    assert pynccl.comm is None
    assert ns["_PG_ABORTED"] is True


def test_destroy_nccl_is_a_tp1_noop(destroy):
    ns, _ = destroy(({}, ["tp:0"]))
    worker = types.SimpleNamespace(rank=0, vllm_config=types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1)))
    assert ns["_destroy_nccl"](worker) is None


# ---------------------------------------------------------------------------
# The child: a dropped image refuses to rebind
# ---------------------------------------------------------------------------

def test_the_child_refuses_to_rebind_graphs_it_dropped():
    src = open(_CHILD).read()
    seg = src[src.index('elif cmd == "rebind_graphs":'):]
    seg = seg[:seg.index("elif cmd ==", 10)]
    assert 'if _GRAPH_DROP.get("graphs_dropped"):' in seg
    assert seg.index("_GRAPH_DROP") < seg.index("collective_rpc"), (
        "the refusal must precede the rebind")


def test_a_recapture_clears_the_drop_record():
    src = open(_CHILD).read()
    seg = src[src.index('elif cmd in ("drop_graphs", "recapture_graphs"):'):]
    seg = seg[:seg.index('elif cmd == "drop_graphs_if_nccl":')]
    assert 'if cmd == "recapture_graphs":\n                    _GRAPH_DROP.clear()' in seg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
