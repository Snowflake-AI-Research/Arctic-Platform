"""Multi-node semi-p: node identity stays out of the key, and the graph path
switches on ``nnodes``.

Two families of property, and they fail in opposite ways.

**Key isolation (L1).** ``vllm_config`` is hashed into the image cache key,
recorded in ``meta.json`` and compared at ``criu_restore``. The experiment
driver put ``node_rank``, ``master_addr`` and ``master_port`` in it, which
means the two halves of one job hash differently and no restored pair can ever
match its own image. That failure is silent at dump time and only shows up as
a permanent cache miss, so it is asserted here rather than discovered later.

**The graph path (G1/G3/G4).** Across nodes vLLM turns custom all-reduce off,
the captured graphs hold NCCL kernels, and NCCL will not release a
communicator a live graph captured -- ``commDestroySync`` spins on ``while
(comm->localPersistentRefs != 0)``. So a multi-node dump must destroy the
graphs *before* ``destroy_nccl``, and a single-node dump must keep doing
exactly what it does today. Both directions are asserted: a gate that is
always on is as wrong as one that never fires, and only the single-node half
is covered by the existing TP<=8 deployments.

``vllm_child.py`` and ``_semip_worker.py`` import torch and vLLM at module
scope, so those are asserted against the source text by AST rather than by
import.

Run from the package directory::

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_multinode.py -v
"""
from __future__ import annotations

import ast
import os
import re
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_CHILD = os.path.join(_PKG, "vllm_child.py")
_INSTANCE = os.path.join(_PKG, "instance.py")
_WORKER = os.path.join(_PKG, "_semip_worker.py")
_CRIU_WORKER = os.path.join(_PKG, "worker.py")
_SERVER = os.path.join(os.path.dirname(_PKG), "server")
_AGENT = os.path.join(_SERVER, "semip_agent.py")
_ENGINE = os.path.join(_SERVER, "semip_engine.py")

if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from multinode import (  # noqa: E402
    MULTINODE_COLD_START_ENV, PINNED_OFI_ENV, MultiNode)


def _tree(path):
    with open(path) as handle:
        return ast.parse(handle.read())


def _function(tree, name, cls=None):
    """The named function, optionally inside the named class."""
    scope = tree
    if cls is not None:
        scope = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == cls)
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    raise AssertionError(f"{name} not found in {cls or 'module'}")


def _sends(node):
    """Every ``self._send("cmd", ...)`` under *node*, in source order."""
    out = []
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "_send"
                and sub.args
                and isinstance(sub.args[0], ast.Constant)):
            out.append((sub.lineno, sub.args[0].value))
    return sorted(out)


# ---------------------------------------------------------------------------
# L1: node identity never reaches the hashed config
# ---------------------------------------------------------------------------

def test_multinode_carries_node_identity_and_not_nnodes():
    """``nnodes`` is topology and belongs to the hashed config; the rest is
    node identity and must not be hashed. Carrying ``nnodes`` here too would
    let the two disagree."""
    mn = MultiNode(node_rank=1, master_addr="10.0.0.1", master_port=29500)
    assert set(mn.as_init_kwargs()) == {
        "node_rank", "master_addr", "master_port", "ifname"}
    assert not hasattr(mn, "nnodes")


def test_init_sends_node_identity_beside_the_config_not_inside_it():
    """The ``init`` command carries ``multinode=`` as its own kwarg. If it were
    merged into ``vllm_config`` the two halves would hash differently."""
    src = open(_INSTANCE).read()
    init = _function(_tree(_INSTANCE), "init", cls="Instance")
    seg = ast.get_source_segment(src, init)
    assert "multinode=self.multinode.as_init_kwargs()" in seg, (
        "init must pass node identity as a command kwarg")
    for forbidden in ('vllm_config["node_rank"]', 'vllm_config["master_addr"]',
                      'vllm_config["master_port"]'):
        assert forbidden not in seg, (
            f"{forbidden} in Instance.init would hash node identity into the "
            "image cache key")


def test_the_child_merges_node_identity_into_its_private_copy_only():
    """The child is where node identity is allowed to reach the engine,
    because by then the dict the caller hashes has already been taken."""
    src = open(_CHILD).read()
    assert 'vllm_config["node_rank"] = _MULTINODE["node_rank"]' in src
    assert 'vllm_config["master_addr"] = _MULTINODE["master_addr"]' in src
    assert 'vllm_config["master_port"] = _MULTINODE["master_port"]' in src


def _lift(path, name, namespace=None):
    """Compile one module-level function out of *path* and return it.

    ``instance.py`` imports pynvml, torch and the sibling chain, none of which
    exist off the cluster, so the pure helpers are executed on their own rather
    than restated in the test.
    """
    fn = _function(_tree(path), name)
    ns = dict(namespace or {})
    ns.setdefault("os", os)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns[name]


@pytest.mark.parametrize("tp,nnodes,local", [
    (8, None, 8),    # single-node TP=8: the shipped path
    (8, 1, 8),
    (16, 2, 8),      # the TP=16 job: 8 GPUs per node
    (16, 4, 4),
])
def test_local_gpu_count_splits_the_group_over_nnodes(tp, nnodes, local):
    cfg = {"tensor_parallel_size": tp}
    if nnodes is not None:
        cfg["nnodes"] = nnodes
    assert _lift(_INSTANCE, "_local_gpu_count")(cfg) == local


def test_local_gpu_count_rejects_an_indivisible_split():
    """A TP group that does not divide evenly would give the halves different
    rank counts and deadlock in their first collective."""
    with pytest.raises(ValueError, match="divisible"):
        _lift(_INSTANCE, "_local_gpu_count")(
            {"tensor_parallel_size": 12, "nnodes": 8})


def test_instance_validates_multinode_against_nnodes():
    """``multinode=`` and ``nnodes`` have to agree, or the TP group is split
    one way while the rendezvous expects another."""
    src = open(_INSTANCE).read()
    seg = ast.get_source_segment(
        src, _function(_tree(_INSTANCE), "__init__", cls="Instance"))
    assert "multinode= requires nnodes > 1" in seg, (
        "a MultiNode with nnodes<2 must be rejected")
    assert "node_rank" in seg and "out of range" in seg, (
        "a node_rank outside nnodes must be rejected")


def test_single_node_construction_is_unchanged():
    """Every TP<=8 deployment constructs an Instance with two arguments, so
    the new parameter has to default."""
    init = _function(_tree(_INSTANCE), "__init__", cls="Instance")
    args = [a.arg for a in init.args.args]
    assert args[:3] == ["self", "vllm_config", "model_dir"], (
        "the existing positional signature must not move")
    assert args[3] == "multinode"
    # one default per optional arg, and multinode's is None
    assert len(init.args.defaults) == 2
    assert all(isinstance(d, ast.Constant) and d.value is None
               for d in init.args.defaults)


# ---------------------------------------------------------------------------
# L2: the pinned aws-ofi-nccl values
# ---------------------------------------------------------------------------

def test_the_pinned_ofi_values_are_the_measured_ones():
    """These are what aws-ofi-nccl 1.21.1 exports on EFA. They are pinned at
    the socket cold start because NCCL caches its parameters at the first init
    in a process, so the plugin's values would otherwise arrive at the
    restore's re-init -- too late to take effect. Pinning them is what makes a
    restored engine bit-identical to an EFA cold start, which is measured."""
    assert PINNED_OFI_ENV == {
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_P2P_NET_CHUNKSIZE": "524288",
        "NCCL_NVLS_CHUNKSIZE": "524288",
        "NCCL_NVLSTREE_MAX_CHUNKSIZE": "524288",
        "NCCL_NET_FORCE_FLUSH": "0",
        "NCCL_NETDEVS_POLICY": "max:1",
    }
    # fd-backed: the plugin supplies it at every init, and a pinned value would
    # point at an fd the dump closed.
    assert "NCCL_TOPO_FILE" not in PINNED_OFI_ENV


def test_the_cold_start_keeps_efa_out_of_the_image():
    """EFA state does not survive CRIU; the restore's re-init is the first EFA
    bring-up. GIN and RAS hold sockets the dump would have to account for."""
    assert MULTINODE_COLD_START_ENV["NCCL_NET"] == "Socket"
    assert MULTINODE_COLD_START_ENV["NCCL_GIN_ENABLE"] == "0"
    assert MULTINODE_COLD_START_ENV["NCCL_RAS_ENABLE"] == "0"


def test_the_child_applies_both_sets_before_vllm_is_imported():
    src = open(_CHILD).read()
    assert "MULTINODE_COLD_START_ENV, PINNED_OFI_ENV" in src, (
        "the child must apply both sets at a multi-node cold start")


# ---------------------------------------------------------------------------
# G1: the drop runs before destroy_nccl, and only across nodes
# ---------------------------------------------------------------------------

def test_the_drop_precedes_destroy_nccl():
    """The whole point of the ordering. ncclCommAbort does not return while a
    graph that captured the communicator is alive, so a drop after the
    teardown is the hang it exists to prevent."""
    sends = _sends(_function(_tree(_INSTANCE), "cuda_checkpoint",
                             cls="Instance"))
    names = [name for _line, name in sends]
    assert "drop_graphs" in names, "cuda_checkpoint must insert the drop"
    assert names.index("drop_graphs") < names.index("destroy_nccl"), (
        "drop_graphs must precede destroy_nccl, or the abort hangs")
    assert names.index("destroy_nccl") < names.index("cuda_checkpoint")


def test_the_drop_is_gated_on_nnodes_not_on_tp():
    """A single-node TP=8 dump keeps its graphs and rebinds them, which is
    faster (2 s against 19 s at TP=4) and is what every current deployment
    does. The gate is the node boundary, not the GPU count: two TP=8 replicas
    across two nodes are still two single-node engines."""
    src = open(_INSTANCE).read()
    seg = ast.get_source_segment(
        src, _function(_tree(_INSTANCE), "cuda_checkpoint", cls="Instance"))
    assert "if self.nnodes > 1:" in seg, (
        "the drop must be gated on nnodes, not on n_gpus or "
        "tensor_parallel_size")


def test_rebind_refuses_to_run_across_nodes():
    """There is nothing to rebind once the graphs were dropped, and a quiet
    no-op would report success and then wedge on the first replay of a graph
    that no longer exists."""
    src = open(_INSTANCE).read()
    seg = ast.get_source_segment(
        src, _function(_tree(_INSTANCE), "rebind_graphs", cls="Instance"))
    assert "if self.nnodes > 1:" in seg
    assert "raise RuntimeError" in seg, (
        "rebind_graphs must raise across nodes, not quietly do nothing")
    assert seg.index("raise RuntimeError") < seg.index('self._send'), (
        "the guard must precede the send")


# ---------------------------------------------------------------------------
# G3: the keep-graph machinery is skipped across nodes -- both directions
# ---------------------------------------------------------------------------

def test_the_keep_graph_machinery_is_gated_in_the_worker():
    """It serves the rebind path only. Across nodes the retained topology is
    never read and the instantiate pass builds execs for graphs the dump is
    about to destroy."""
    src = open(_WORKER).read()
    seg = ast.get_source_segment(
        src, _function(_tree(_WORKER), "compile_or_warm_up_model",
                       cls="SemipGPUWorker"))
    assert "keep_graphs = not _multinode()" in seg
    assert "if keep_graphs:" in seg, (
        "install_keepgraph_patch must be gated")
    assert "if not keep_graphs:" in seg, (
        "the instantiate pass and census must be skipped across nodes")


def test_the_worker_gate_reads_the_environment():
    """The worker processes are spawned by vLLM and import this module fresh;
    they never see the child's globals. SEMIP_NNODES is the only channel."""
    src = open(_WORKER).read()
    assert "SEMIP_NNODES" in src, (
        "the worker gate must read SEMIP_NNODES, not a child-side global")


def test_the_child_exports_the_gate_before_vllm_spawns_the_workers():
    src = open(_CHILD).read()
    assert 'os.environ["SEMIP_NNODES"]' in src
    assert 'os.environ["SEMIP_MULTINODE_IFNAME"]' in src


def _worker_multinode_fn():
    """``_semip_worker._multinode``, compiled on its own.

    That module imports vLLM at its top and vLLM is absent off the cluster, so
    the function is lifted out of the parsed source and executed by itself.
    This keeps the test on the real code rather than on a restatement of it.
    """
    fn = _function(_tree(_WORKER), "_multinode")
    namespace = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _WORKER, "exec"),
         namespace)
    return namespace["_multinode"]


@pytest.mark.parametrize("nnodes,multinode", [
    (None, False),   # unset: every TP<=8 deployment today
    ("1", False),    # two TP=8 replicas on two nodes are still single-node
    ("2", True),
    ("4", True),
])
def test_the_worker_gate_fires_in_both_directions(monkeypatch, nnodes,
                                                  multinode):
    """A gate that never fires and a gate that is always on fail the same
    review. Only the single-node half is exercised by existing deployments, so
    the multi-node half is asserted here."""
    monkeypatch.delenv("SEMIP_NNODES", raising=False)
    if nnodes is not None:
        monkeypatch.setenv("SEMIP_NNODES", nnodes)
    assert _worker_multinode_fn()() is multinode


# ---------------------------------------------------------------------------
# G4/G5: the recapture
# ---------------------------------------------------------------------------

def _call_lineno(fn, name):
    """Line of the first call to *name* under *fn*, ignoring the docstring.

    Comparing text offsets would be fooled by a docstring that names the call
    it is describing, which is exactly what this function's does.
    """
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call):
            func = sub.func
            attr = getattr(func, "attr", None) or getattr(func, "id", None)
            if attr == name:
                return sub.lineno
    return None


def test_the_recapture_unlocks_the_workspace_first():
    """capture_model() ends with lock_workspace(), which makes the workspace
    refuse any allocation larger than the current one. A recapture that does
    not unlock first runs in a state the cold-start capture never saw."""
    fn = _function(_tree(_CHILD), "_semip_recapture_graphs")
    unlock = _call_lineno(fn, "unlock_workspace")
    capture = _call_lineno(fn, "capture_model")
    assert unlock is not None, "the recapture must unlock the workspace"
    assert capture is not None
    assert unlock < capture, "the unlock must precede the capture"


def test_the_recapture_requires_every_graph_back():
    """`n_exec_ok > 0` would pass a recapture that built one graph of four
    thousand -- and capture_model() returns 0 without capturing when the
    manager thinks it has nothing to do."""
    src = open(_CHILD).read()
    seg = ast.get_source_segment(
        src, _function(_tree(_CHILD), "_semip_recapture_graphs"))
    assert "n_exec_ok == n_graphs" in seg, (
        "the recapture must verify the full count, not merely a nonzero one")


# ---------------------------------------------------------------------------
# E: the agent's half of the dump (first production job, 2026-10-06)
# ---------------------------------------------------------------------------

def _unpark_dir_fn():
    """``Instance._unpark_dir``, compiled on its own with a stub ``mq_plane``."""
    fn = _function(_tree(_INSTANCE), "_unpark_dir", cls="Instance")
    mq_plane = type("mq_plane", (), {"unpark_dir_for": staticmethod(
        lambda key: os.path.join("/dev/shm", f"semip-unpark-{key}"))})
    namespace = {"os": os, "re": re, "mq_plane": mq_plane}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _INSTANCE, "exec"),
         namespace)
    return namespace["_unpark_dir"]


def _unpark_dir(model_dir, nnodes):
    return _unpark_dir_fn()(types.SimpleNamespace(model_dir=model_dir,
                                                  nnodes=nnodes))


def test_both_halves_rendezvous_in_one_unpark_directory():
    """The leader's park hands its directory to every rank, so the parked
    readers on node 1 poll the leader's path. Named after each half's own
    ``node<k>``, the follower waited in, and would have unparked into, a
    directory its ranks never look at."""
    key = "/data-fast/image-cache_neutrino/79c172ce4f58_42387c11e280"
    leader = _unpark_dir(f"{key}/node0", 2)
    assert leader == "/dev/shm/semip-unpark-79c172ce4f58_42387c11e280"
    assert _unpark_dir(f"{key}/node1/", 2) == leader


@pytest.mark.parametrize("model_dir,nnodes,name", [
    ("/cache/abc_def", 1, "abc_def"),              # every TP<=8 image
    ("/cache/abc_def/replica1", 1, "replica1"),    # per-replica layout
    ("/cache/node0", 1, "node0"),                  # not a half: nnodes is 1
    ("/data-fast/exp2", 2, "exp2"),                # the experiment driver
])
def test_the_unpark_directory_is_unchanged_off_the_node_layout(
        model_dir, nnodes, name):
    assert _unpark_dir(model_dir, nnodes) == f"/dev/shm/semip-unpark-{name}"


def test_the_agent_defaults_unprivileged_mode_like_the_leader():
    """The leader's default is set in its own process. Without the same one
    the agent's CRIU ran without --unprivileged and refused at check_caps."""
    src = open(_AGENT).read()
    seg = ast.get_source_segment(
        src, _function(_tree(_AGENT), "__init__", cls="SemipNodeAgent"))
    assert "os.environ.setdefault(_UNPRIVILEGED_ENV, \"1\")" in seg


def test_the_agent_dump_waits_for_its_ranks_to_park():
    """The actor runs calls concurrently, so a separate ``wait_parked`` call
    from the leader did not hold the dump back: node 1 dumped 10 ms before
    its ranks parked."""
    dump = _function(_tree(_AGENT), "criu_dump", cls="SemipNodeAgent")
    wait = _call_lineno(dump, "wait_parked")
    send = _call_lineno(dump, "criu_dump")
    assert wait is not None and send is not None
    assert wait < send, "wait_parked must precede the Instance's criu_dump"
    engine = _function(_tree(_ENGINE), "_dump_multinode")
    assert _call_lineno(engine, "wait_parked") is None, (
        "the leader must not issue a fire-and-forget wait_parked")


def test_the_agent_init_clears_stale_park_markers():
    """The leader's park clears its own pod's directory only; a marker left
    on the follower's pod would satisfy wait_parked immediately."""
    init = _function(_tree(_AGENT), "init", cls="SemipNodeAgent")
    clear = _call_lineno(init, "unlink")
    start = next(sub.lineno for sub in ast.walk(init)
                 if isinstance(sub, ast.Call)
                 and getattr(sub.func, "attr", None) == "init")
    assert clear is not None and clear < start


def test_a_failed_dump_reports_both_criu_streams():
    """check_caps refuses on stdout while stderr holds the run id, so
    preferring one stream dropped the only line that named the problem."""
    src = open(_CRIU_WORKER).read()
    seg = ast.get_source_segment(
        src, _function(_tree(_CRIU_WORKER), "_worker_criu_save"))
    assert "result.stderr or result.stdout" not in seg
    assert "(result.stderr, result.stdout)" in seg

