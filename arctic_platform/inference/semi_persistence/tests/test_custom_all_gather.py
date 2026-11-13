"""TP all-gathers go through CustomAllreduce, and the rebind rewrites them.

DeepSeek-V4.1-Flash TP=8 graphs held 816 ``ncclDevKernel_AllGather_RING_LL``
nodes per rank -- Engram's ``embed`` (768 KiB per rank at 512 tokens) and
``wkv`` (3.125 MiB) gathers -- so every dump dropped 4692 graphs and every
restore recaptured them, ~85 s. ``ca_all_gather`` sends those gathers to
``CustomAllreduce.custom_all_gather`` (kernel ``cross_device_all_gather``),
and ``ca_graph_rebind`` rewrites that kernel's baked signal pads like an
all-reduce's: both take ``(RankData* _dp, RankSignals sg, Signal* self_sg,
T* result, int rank, int size)`` (csrc/custom_all_reduce.cuh,
custom_all_gather_reduce_scatter.cuh at v0.32.0).

The routing is tested on CPU tensors against a CustomAllreduce double that
concatenates the peers' shards; the rebind against a libcuda double that keeps
each kernel node's arguments in real ctypes storage, so ``_patch_node`` reads
and writes them exactly as it does on the driver's.

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_custom_all_gather.py -v
"""
from __future__ import annotations

import ast
import ctypes
import importlib.util
import os
import sys
import types

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_ROUTER = os.path.join(_PKG, "ca_all_gather.py")
_REBIND = os.path.join(_PKG, "ca_graph_rebind.py")
_WORKER = os.path.join(_PKG, "_semip_worker.py")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import matched_baseline as mb  # noqa: E402 - siblings load by bare name

_MiB = 1 << 20


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cgr = _load(_REBIND, "ca_graph_rebind_custom_ag_test")


@pytest.fixture
def cag():
    """A fresh router per test: its counters are module state."""
    mod = _load(_ROUTER, "ca_all_gather_under_test")
    yield mod
    mod.restore()


# --------------------------------------------------------------------------
# Doubles.
# --------------------------------------------------------------------------

class _FakeCA:
    """CustomAllreduce as the router sees it. ``custom_all_gather`` returns
    the rank-major concatenation of every rank's shard, this rank's being the
    input; it refuses what vLLM's ``should_custom_all_gather`` refuses."""

    def __init__(self, world=8, rank=3, disabled=False, multicast=0,
                 max_size=8 * _MiB, max_all_gather_size=2 * _MiB,
                 max_reduce_scatter_size=16 * _MiB):
        self.world_size = world
        self.rank = rank
        self.disabled = disabled
        self.mnnvl_multicast_ptr = multicast
        self.max_size = max_size
        self.max_all_gather_size = max_all_gather_size
        self.max_reduce_scatter_size = max_reduce_scatter_size
        self.peers = None
        self.calls = 0

    def custom_all_gather(self, inp):
        self.calls += 1
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return None
        nbytes = inp.numel() * inp.element_size()
        if not 0 < nbytes <= self.max_all_gather_size or nbytes % 16:
            return None
        peers = list(self.peers or [inp] * self.world_size)
        peers[self.rank] = inp
        return torch.cat(peers, dim=0)


def _comm(ca, name="tp:0", world=8):
    return types.SimpleNamespace(unique_name=name, ca_comm=ca, world_size=world)


def _shards(shape, world=8, dtype=torch.float32):
    """Rank r's shard: distinct per element and per rank."""
    n = 1
    for s in shape:
        n *= s
    base = torch.arange(n, dtype=torch.float32).reshape(shape)
    return [(base + r * n).to(dtype) for r in range(world)]


# --------------------------------------------------------------------------
# The routing decision.
# --------------------------------------------------------------------------

def test_an_eligible_tp_gather_takes_the_custom_path(cag):
    ca = _FakeCA()
    x = torch.zeros(4, 16)
    out = cag.route(_comm(ca), x, 0)
    assert out is not None and ca.calls == 1
    assert out.shape == (32, 16)
    assert cag.state()["custom"] == 1 and cag.state()["fallback"] == {}


@pytest.mark.parametrize("comm_kw,ca_kw,tensor,why", [
    ({"name": "ep:0"}, {}, torch.zeros(4, 16), "not_tp"),
    ({}, None, torch.zeros(4, 16), "no_ca"),
    ({}, {"disabled": True}, torch.zeros(4, 16), "no_ca"),
    ({}, {"multicast": 0xABC000}, torch.zeros(4, 16), "multicast"),
    ({"world": 16}, {"world": 16}, torch.zeros(4, 16), "world_size"),
    ({}, {}, torch.zeros(16, 4).t(), "non_contiguous"),
    ({}, {}, torch.zeros(2 * _MiB // 4 + 4), "size"),
    ({}, {}, torch.zeros(4, 16, dtype=torch.int32), "ca_refused"),
    ({}, {}, torch.zeros(3), "ca_refused"),
], ids=["not_tp", "no_ca", "disabled", "multicast", "world16",
        "non_contiguous", "over_limit", "int_dtype", "unaligned"])
def test_what_custom_cannot_take_falls_back(cag, comm_kw, ca_kw, tensor, why):
    ca = None if ca_kw is None else _FakeCA(**ca_kw)
    assert cag.route(_comm(ca, **comm_kw), tensor, 0) is None
    assert cag.state()["fallback"] == {why: 1}
    assert cag.state()["custom"] == 0


def test_the_multicast_branch_is_never_entered(cag):
    """custom_all_gather would take the MNNVL Lamport kernel, whose multicast
    mapping cuCheckpointProcess* cannot carry; the router must not call it."""
    ca = _FakeCA(multicast=0x1000)
    cag.route(_comm(ca), torch.zeros(4, 16), 0)
    assert ca.calls == 0


def test_a_fallback_inside_a_capture_is_reported(cag, monkeypatch):
    """That fallback is an NCCL kernel in a kept graph; it has to show up
    before the dump-time census finds it."""
    monkeypatch.setattr(cag, "_capturing", lambda: True)
    cag.route(_comm(_FakeCA()), torch.zeros(2 * _MiB // 4 + 4), 0)
    assert cag.state()["capture_fallback"] == {"size": 1}


@pytest.mark.parametrize("shape,dim", [
    ((5, 3, 16), 0), ((5, 3, 16), 1), ((5, 3, 16), 2), ((5, 3, 16), -1),
    ((5, 3, 16), -2), ((7, 32), -1),
])
def test_dim_gathers_match_a_concatenation(cag, shape, dim):
    """The PyNccl branch gathers along dim 0 and then reshapes; the custom
    output is rank-major the same way, so the same reshape must produce
    torch.cat(shards, dim)."""
    ca = _FakeCA()
    ca.peers = _shards(shape)
    out = cag.route(_comm(ca), ca.peers[ca.rank], dim)
    assert out is not None
    assert torch.equal(out, torch.cat(ca.peers, dim=dim))


def test_the_engram_gathers_fit_only_once_the_limit_is_widened(cag):
    """DeepSeek-V4.1-Flash at the largest capture size (512 tokens/rank):
    ``embed`` rows [512, 3, 256] gathered on dim 1, ``wkv`` [512, 3200] on
    dim -1, both bf16. The 2 MiB default turns ``wkv`` away -- an NCCL node
    in every graph that size -- and the 16 MiB staging buffer takes both."""
    embed = _shards((512, 3, 256), dtype=torch.bfloat16)
    wkv = _shards((512, 3200), dtype=torch.bfloat16)
    ca = _FakeCA()
    ca.peers = embed
    assert cag.route(_comm(ca), embed[ca.rank], 1) is not None
    ca.peers = wkv
    assert cag.route(_comm(ca), wkv[ca.rank], -1) is None
    assert cag.state()["fallback"] == {"size": 1}
    assert cag.widen_limit(ca) == 16 * _MiB
    out = cag.route(_comm(ca), wkv[ca.rank], -1)
    assert torch.equal(out, torch.cat(wkv, dim=-1))
    assert cag.state()["max_custom_bytes"] == 512 * 3200 * 2


@pytest.mark.parametrize("ca_kw,limit", [
    ({}, 16 * _MiB),
    ({"max_size": 4 * _MiB}, 16 * _MiB),
    ({"max_reduce_scatter_size": 4 * _MiB}, 8 * _MiB),
    ({"max_all_gather_size": 32 * _MiB}, 32 * _MiB),
])
def test_the_limit_never_outgrows_the_staging_buffer(cag, ca_kw, limit):
    """The constructor allocates max(max_size, max_all_gather_size,
    max_reduce_scatter_size); widening to that size allocates nothing."""
    ca = _FakeCA(**ca_kw)
    assert cag.widen_limit(ca) == limit
    assert ca.max_all_gather_size == limit


def test_a_disabled_ca_is_not_widened(cag):
    ca = _FakeCA(disabled=True)
    assert cag.widen_limit(ca) == 0
    assert ca.max_all_gather_size == 2 * _MiB


# --------------------------------------------------------------------------
# install / restore against a stubbed vLLM.
# --------------------------------------------------------------------------

@pytest.fixture
def vllm_stub(monkeypatch):
    class CudaCommunicator:
        def __init__(self, ca, unique_name="tp:0", world_size=8):
            self.ca_comm, self.unique_name = ca, unique_name
            self.world_size = world_size
            self.nccl_calls = []

        def all_gather(self, input_, dim=-1):
            self.nccl_calls.append(dim)
            return "nccl"

    class CustomAllreduce(_FakeCA):
        pass

    live_ca = CustomAllreduce()
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps._TP = types.SimpleNamespace(
        device_communicator=CudaCommunicator(live_ca))
    cc = types.ModuleType("vllm.distributed.device_communicators."
                          "cuda_communicator")
    cc.CudaCommunicator = CudaCommunicator
    car = types.ModuleType("vllm.distributed.device_communicators."
                           "custom_all_reduce")
    car.CustomAllreduce = CustomAllreduce
    dc = types.ModuleType("vllm.distributed.device_communicators")
    dc.cuda_communicator, dc.custom_all_reduce = cc, car
    dist = types.ModuleType("vllm.distributed")
    dist.device_communicators, dist.parallel_state = dc, ps
    root = types.ModuleType("vllm")
    root.distributed = dist
    for name, mod in (("vllm", root), ("vllm.distributed", dist),
                      ("vllm.distributed.parallel_state", ps),
                      ("vllm.distributed.device_communicators", dc),
                      (cc.__name__, cc), (car.__name__, car)):
        monkeypatch.setitem(sys.modules, name, mod)
    return types.SimpleNamespace(CudaCommunicator=CudaCommunicator,
                                 CustomAllreduce=CustomAllreduce,
                                 live_ca=live_ca,
                                 orig_ag=CudaCommunicator.all_gather,
                                 orig_init=CustomAllreduce.__init__)


@pytest.fixture
def compiled(cag, monkeypatch):
    monkeypatch.setattr(cag, "_op_compiled", lambda torch: True)
    return cag


def test_install_routes_and_falls_back(compiled, vllm_stub):
    info = compiled.install(environ={})
    assert info["active"] is True and info["err"] is None
    comm = vllm_stub.CudaCommunicator(vllm_stub.CustomAllreduce())
    assert isinstance(comm.all_gather(torch.zeros(4, 16), 0), torch.Tensor)
    assert comm.nccl_calls == []
    assert comm.all_gather(torch.zeros(4, 16, dtype=torch.int32), 1) == "nccl"
    assert comm.nccl_calls == [1], "the fallback did not reach the original"


def test_install_widens_the_ca_already_built_and_every_later_one(
        compiled, vllm_stub):
    """``Worker.init_device`` builds the TP communicator before the patch
    exists, and the restore's ``reinit_nccl`` builds another after it."""
    compiled.install(environ={})
    assert vllm_stub.live_ca.max_all_gather_size == 16 * _MiB
    rebuilt = vllm_stub.CustomAllreduce()
    assert rebuilt.max_all_gather_size == 16 * _MiB
    assert compiled.state()["widened"] == 2


def test_install_is_idempotent_and_restore_undoes_it(compiled, vllm_stub):
    compiled.install(environ={})
    patched = vllm_stub.CudaCommunicator.all_gather
    compiled.install(environ={})
    assert vllm_stub.CudaCommunicator.all_gather is patched
    compiled.restore()
    assert vllm_stub.CudaCommunicator.all_gather is vllm_stub.orig_ag
    assert vllm_stub.CustomAllreduce.__init__ is vllm_stub.orig_init
    assert compiled.state()["active"] is False


def test_the_switch_turns_it_off(compiled, vllm_stub):
    info = compiled.install(environ={compiled.ENV: "0"})
    assert info["active"] is False and "SEMIP_CUSTOM_ALL_GATHER" in info["err"]
    assert vllm_stub.CudaCommunicator.all_gather is vllm_stub.orig_ag


def test_a_build_without_the_op_installs_nothing(cag, vllm_stub):
    """CPU torch has no ``_C_custom_ar``; neither would an image whose vLLM
    was built without custom_all_gather_reduce_scatter_ops.cpp. Routing
    there would fail inside the capture, so the gate stays the fallback."""
    info = cag.install(environ={})
    assert info["active"] is False and "not in this build" in info["err"]
    assert vllm_stub.CudaCommunicator.all_gather is vllm_stub.orig_ag


# --------------------------------------------------------------------------
# The worker installs it for good; the matched baseline runs that worker.
# --------------------------------------------------------------------------

def _worker_method(name):
    tree = ast.parse(open(_WORKER).read())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "SemipGPUWorker")
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _attr_calls(fn, attr):
    return [c for c in ast.walk(fn) if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute) and c.func.attr == attr]


def test_init_device_installs_the_routing():
    """``determine_available_memory`` captures before the warmup hook; the
    routing must already be live there, as the copy-path patch is."""
    assert _attr_calls(_worker_method("init_device"),
                       "_semip_install_custom_all_gather")
    assert _attr_calls(_worker_method("_semip_install_custom_all_gather"),
                       "install")


def test_the_routing_is_never_withdrawn():
    """The restore's reinit and a fallback recapture both run in the image
    after the warmup hook is long gone."""
    src = open(_WORKER).read()
    assert "ca_all_gather.restore" not in src


def test_the_warmup_reports_capture_fallbacks_after_the_capture():
    fn = _worker_method("compile_or_warm_up_model")
    reports = [c.lineno for c in _attr_calls(
        fn, "_semip_install_custom_all_gather")]
    captures = [c.lineno for t in ast.walk(fn)
                if isinstance(t, ast.Try) and t.finalbody
                for stmt in t.body for c in ast.walk(stmt)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "compile_or_warm_up_model"]
    assert reports and captures and max(reports) > max(captures)


def test_the_matched_baseline_gets_the_same_routing():
    """off_matched has to run the kernels semi_p runs, which it does by
    running the worker class that installs the routing."""
    kwargs = {"tensor_parallel_size": 8}
    mb.apply(kwargs, environ={})
    assert kwargs["worker_cls"] == "_semip_worker.SemipGPUWorker"
    assert _attr_calls(_worker_method("init_device"),
                       "_semip_install_custom_all_gather")


# --------------------------------------------------------------------------
# The rebind: kernel names, and the arguments _patch_node rewrites.
# --------------------------------------------------------------------------

_AG = ("_ZN4vllm23cross_device_all_gatherILi8EEEvPNS_8RankDataENS_11"
       "RankSignalsEPNS_6SignalEPNS_7array_tImLi2EEEii")
_AR = ("_ZN4vllm26cross_device_reduce_1stageI13__nv_bfloat16Li8EEEvPNS_8"
       "RankDataENS_11RankSignalsEPNS_6SignalEPT_ii")
_RS = ("_ZN4vllm27cross_device_reduce_scatterI13__nv_bfloat16Li8EEEvPNS_8"
       "RankDataENS_11RankSignalsEPNS_6SignalEPT_ii")
_MNNVL_AG = ("_ZN4vllm24mnnvl_lamport_all_gatherI13__nv_bfloat16Li8EEEvPNS_8"
             "RankDataEPKT_PS4_S7_Pjiii")
_NCCL_AG = "ncclDevKernel_AllGather_RING_LL"
_GEMM = "sm90_xmma_gemm_bf16"


@pytest.mark.parametrize("name,kind", [
    (_AG, "all_gather"), (_AR, "all_reduce"), (_RS, "reduce_scatter"),
    ("cross_device_reduce_2stage<bf16, 8>", "all_reduce"),
    (_MNNVL_AG, None), (_NCCL_AG, None), (_GEMM, None), ("", None),
])
def test_ca_kernels_are_classified_by_name(name, kind):
    assert cgr._ca_kernel_kind(name) == kind


class _Args:
    """One kernel node's argument storage: _dp, sg (8 ptrs), self_sg,
    result, rank, size -- the CA kernels' 6-argument ABI."""

    def __init__(self, dp, sg, self_sg, result, rank, size):
        self.bufs = [ctypes.c_uint64(dp), (ctypes.c_uint64 * 8)(*sg),
                     ctypes.c_uint64(self_sg), ctypes.c_uint64(result),
                     ctypes.c_int(rank), ctypes.c_int(size)]
        self.kp = (ctypes.c_void_p * 6)(
            *[ctypes.addressof(b) for b in self.bufs])

    @classmethod
    def copy_of(cls, kp):
        """What the driver keeps after SetParams: a copy, by value."""
        u64 = ctypes.c_uint64
        return cls(u64.from_address(kp[0]).value,
                   list((u64 * 8).from_address(kp[1])),
                   u64.from_address(kp[2]).value,
                   u64.from_address(kp[3]).value,
                   ctypes.c_int.from_address(kp[4]).value,
                   ctypes.c_int.from_address(kp[5]).value)

    def values(self):
        dp, sg, self_sg, result, rank, size = self.bufs
        return {"dp": dp.value, "sg": list(sg), "self": self_sg.value,
                "result": result.value, "rank": rank.value, "size": size.value}


class _RebindCu:
    """libcuda as the rebind reads and writes it."""

    def __init__(self):
        self.graphs: dict = {}
        self.nodes: dict = {}
        self.funcs: dict = {}
        self.topo_sets: list = []
        self.exec_sets: list = []
        self.retired: list = []

    def _add(self, graph, node):
        nid = 0x2000 + len(self.nodes)
        self.nodes[nid] = node
        self.graphs.setdefault(graph, []).append(nid)
        return nid

    def kernel(self, graph, name, args=None):
        return self._add(graph, {"type": 0, "name": name, "args": args})

    def memcpy(self, graph, src, dst):
        p = cgr._Memcpy3D()
        p.srcDevice, p.dstDevice = src, dst
        return self._add(graph, {"type": 1, "p": p})

    def cuGraphGetNodes(self, graph, arr, num):
        nodes = self.graphs.get(graph, [])
        num._obj.value = len(nodes)
        if arr is not None:
            for i, n in enumerate(nodes):
                arr[i] = n
        return 0

    def cuGraphNodeGetType(self, node, t):
        t._obj.value = self.nodes[node]["type"]
        return 0

    def cuGraphKernelNodeGetParams_v2(self, node, params):
        n = self.nodes[node]
        name = n["name"].encode()
        params._obj.func = self.funcs.setdefault(name, 0x50 + len(self.funcs))
        args = n["args"] or _Args(0, [0] * 8, 0, 0, 0, 0)
        n["args"] = args
        params._obj.kernelParams = ctypes.cast(
            args.kp, ctypes.POINTER(ctypes.c_void_p))
        return 0

    def cuFuncGetName(self, namep, func):
        namep._obj.value = next(n for n, f in self.funcs.items() if f == func)
        return 0

    def cuGraphKernelNodeSetParams_v2(self, node, params):
        # _patch_node hands the exec set the same kp[0]/kp[3..5] pointers
        # into the storage GetParams returned, so that storage outlives this.
        self.retired.append(self.nodes[node]["args"])
        self.nodes[node]["args"] = _Args.copy_of(params._obj.kernelParams)
        self.topo_sets.append(node)
        return 0

    def cuGraphExecKernelNodeSetParams_v2(self, exec_, node, params):
        self.exec_sets.append(
            (exec_, node, _Args.copy_of(params._obj.kernelParams).values()))
        return 0

    def cuGraphMemcpyNodeGetParams(self, node, p):
        src = self.nodes[node]["p"]
        ctypes.memmove(ctypes.addressof(p._obj), ctypes.addressof(src),
                       ctypes.sizeof(src))
        return 0

    def cuGraphMemcpyNodeSetParams(self, node, p):
        dst = self.nodes[node]["p"]
        ctypes.memmove(ctypes.addressof(dst), ctypes.addressof(p._obj),
                       ctypes.sizeof(dst))
        return 0

    def cuGraphExecMemcpyNodeSetParams(self, exec_, node, p, ctx):
        self.exec_sets.append((exec_, node, int(p._obj.dstDevice)))
        return 0

    def cuCtxGetCurrent(self, ctx):
        return 0


_RANK, _WORLD = 3, 8
_OLD_META = [0x7F0000000000 + i * 0x100000 for i in range(_WORLD)]
_NEW_META = [0x7E0000000000 + i * 0x100000 for i in range(_WORLD)]
_OLD_BUF = [0x7D0000000000 + i * 0x1000000 for i in range(_WORLD)]
_NEW_BUF = [0x7C0000000000 + i * 0x1000000 for i in range(_WORLD)]
_RANK_DATA = 0x7B0000000000
_GRAPH, _EXEC = 0x10, 0xE10


def _ca_args(result, size):
    """What a copy-path capture bakes: slot 0 of rank_data, the pre-restore
    signal pads, and this rank's staging-buffer-independent output."""
    return _Args(_RANK_DATA, _OLD_META, _OLD_META[_RANK], result, _RANK, size)


def _addr_map():
    m = dict(zip(_OLD_META, _NEW_META))
    m.update(zip(_OLD_BUF, _NEW_BUF))
    return m


def test_an_all_gather_node_is_patched_like_an_all_reduce():
    cu = _RebindCu()
    node = cu.kernel(_GRAPH, _AG, _ca_args(0x7A0000001000, 4800))
    vfy = {"k_topo_ok": 0, "k_topo_bad": 0, "topo_readback_err": 0}
    kinds: dict = {}
    n = cgr._patch_node(cu, _EXEC, node, _addr_map(), vfy, kinds)
    assert n == 9, "8 sg slots + self_sg"
    after = cu.nodes[node]["args"].values()
    assert after["sg"] == _NEW_META
    assert after["self"] == _NEW_META[_RANK]
    assert (after["dp"], after["result"], after["rank"], after["size"]) == (
        _RANK_DATA, 0x7A0000001000, _RANK, 4800), (
        "the rebind touched an argument it has no new value for")
    assert cu.exec_sets and cu.exec_sets[0][2] == after
    assert vfy == {"k_topo_ok": 9, "k_topo_bad": 0, "topo_readback_err": 0}
    assert kinds == {"all_gather": 1}


@pytest.mark.parametrize("name", [_MNNVL_AG, _NCCL_AG, _GEMM])
def test_other_gathers_are_left_alone(name):
    """The rebind dereferences kp[1] as 8 pointers; on a kernel with another
    argument list that reads past an argument's storage."""
    cu = _RebindCu()
    args = _ca_args(0x7A0000001000, 4800)
    node = cu.kernel(_GRAPH, name, args)
    assert cgr._patch_node(cu, _EXEC, node, _addr_map()) == 0
    assert cu.topo_sets == [] and cu.exec_sets == []
    assert cu.nodes[node]["args"].values()["sg"] == _OLD_META


def _deepseek_graph(cu):
    """One Engram-layer stretch after routing: staging memcpy + all-gather,
    staging memcpy + all-reduce, a GEMM."""
    cu.memcpy(_GRAPH, 0x7A0000009000, _OLD_BUF[_RANK])
    ag = cu.kernel(_GRAPH, _AG, _ca_args(0x7A0000001000, 4800))
    cu.memcpy(_GRAPH, 0x7A000000A000, _OLD_BUF[_RANK])
    ar = cu.kernel(_GRAPH, _AR, _ca_args(0x7A0000002000, 640))
    cu.kernel(_GRAPH, _GEMM)
    return ag, ar


@pytest.fixture
def rebind_env(monkeypatch):
    cu = _RebindCu()
    ca = types.SimpleNamespace(meta_ptrs=_NEW_META, buffer_ptrs=_NEW_BUF,
                               rank=_RANK, world_size=_WORLD)
    monkeypatch.setattr(cgr, "_cu", lambda: cu)
    monkeypatch.setattr(cgr, "_find_ca", lambda worker: ca)
    monkeypatch.setattr(cgr, "_find_captured_graphs",
                        lambda worker=None: ([(_GRAPH, _EXEC)],
                                             {"complete": True}))
    return cu


def test_the_rewrite_covers_all_gather_nodes_and_audits_clean(rebind_env):
    cu = rebind_env
    ag, ar = _deepseek_graph(cu)
    out = cgr.rewrite_addrs_in_graphs(None, _addr_map())
    assert out["ok"] is True, out.get("error")
    assert out["kernel_nodes_patched"] == 2
    assert out["kernel_nodes_patched_by_kind"] == {"all_gather": 1,
                                                   "all_reduce": 1}
    assert out["memcpy_nodes_patched"] == 2
    assert out["topo_readback"]["ok"] is True
    audit = out["stale_ca_audit"]
    assert audit["pos_checked"] is True and audit["bad"] == 0
    assert audit["n_ca_kernel_by_kind"] == {"all_gather": 1, "all_reduce": 1}
    for node in (ag, ar):
        assert cu.nodes[node]["args"].values()["sg"] == _NEW_META
    staged = [n["p"].dstDevice for n in cu.nodes.values() if n["type"] == 1]
    assert staged == [_NEW_BUF[_RANK]] * 2


def test_a_missed_all_gather_node_fails_the_audit(rebind_env, monkeypatch):
    """What the old ``cross_device_reduce`` filter would have shipped: the
    all-gather keeps the pre-restore pads and the audit has to say so."""
    cu = rebind_env
    _deepseek_graph(cu)
    real = cgr._patch_node
    monkeypatch.setattr(
        cgr, "_patch_node",
        lambda c, e, node, m, vfy=None, kinds=None: (
            0 if cu.nodes[node].get("name") == _AG
            else real(c, e, node, m, vfy, kinds)))
    out = cgr.rewrite_addrs_in_graphs(None, _addr_map())
    assert out["stale_ca_audit"]["bad"] > 0


def test_the_disk_wake_recommit_includes_all_gather_nodes(rebind_env):
    cu = rebind_env
    _deepseek_graph(cu)
    out = cgr.rewrite_addrs_in_graphs(None, {})
    assert out["forced_recommit"] is True and out["ok"] is True
    assert out["kernel_nodes_recommitted"] == 2
    assert out["kernel_nodes_recommitted_by_kind"] == {"all_gather": 1,
                                                       "all_reduce": 1}


def test_an_all_gather_only_graph_still_yields_its_dp_slot(rebind_env):
    """refresh_copy_path_slots skips a rank whose graphs show no CA `_dp`."""
    cu = rebind_env
    cu.kernel(_GRAPH, _AG, _ca_args(0x7A0000001000, 4800))
    assert cgr._collect_dp_slots(cu, _RANK_DATA, 8 * _MiB) == [0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
