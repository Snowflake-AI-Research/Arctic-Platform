"""The vLLM 0.30 communicator checkpoint hooks must be driven, on both paths.

vLLM 0.30 backs the FlashInfer all-reduce workspace -- and, under expert
parallelism, the MoE all2all buffers -- with MNNVL fabric handles. The CUDA
checkpoint API cannot carry those; NVIDIA's unsupported list is "IPC, UVM,
RDMA, or fabric handle" (cuda-checkpoint#14). vLLM 0.26 never allocated them,
so semi-p never had to care, and the omission only surfaced on the upgrade.

Measured on GLM-5.3 TP=8 with one payload held fixed at util 0.9 /
msl 327680, against the 0.26 control in ``~/r11-glm-warm``:

    0.26                cuda_checkpoint OK 26.8s   cuda_restore OK 15.9s
    0.30 (no hooks)     cuda_checkpoint OK 43.9s   cuda_restore FAILED 801
    0.30 (no hooks)     cuda_checkpoint hung       -- never reached

CUresult 801 is CUDA_ERROR_NOT_SUPPORTED. The two shapes are one bug: the
driver "does not attempt to keep the process in a good state if an error is
encountered during checkpoint or restore", so an unsupported allocation stalls
the checkpoint on one run and fails the restore on the next.

The property under test is that semi-p *drives* the hooks. It is deliberately
not "fabric memory is absent" -- that is vLLM's business and changes release to
release, which is exactly how this was missed the first time.

The second property is collective safety. ``checkpoint_prepare`` ends in a
cross-rank barrier, so the set of communicators visited and the order of the
visit must be identical on every rank, and one rank must not abandon the loop
on an error while the others continue into the barrier. Sorting and
exception-swallowing are therefore load-bearing, not style.

``vllm_child.py`` imports torch at module scope, so the helpers are extracted
and compiled in isolation rather than imported, and the call sites are asserted
by AST.

Run from the package directory::

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_communicator_checkpoint_hooks.py -v
"""
from __future__ import annotations

import ast
import contextlib
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_CHILD = os.path.join(_PKG, "vllm_child.py")

_HELPERS = ("_communicator_checkpoint_targets", "_run_communicator_checkpoint_hook",
            "_fi_ar_workspaces_for", "_checkpoint_target_inventory",
            "_current_role_communicator", "_restore_prepared_comm_state",
            "_assert_checkpoint_state_restored", "_prepared_device_indices",
            "_bind_cuda_context_for_restore", "_probe_restore_devices")

# Module-level names the helpers close over. They have to be compiled in too, or
# the helpers raise NameError in exactly the way `a63ab39` did on the cluster --
# which is the failure this harness exists to catch, so it must not be the
# failure the harness itself introduces.
_CONSTS = ("_CKPT_PREPARED_ATTR", "_CKPT_ROLE_GETTERS")


# --------------------------------------------------------------------------
# Loading the helpers without importing the module.
# --------------------------------------------------------------------------

def _load_helpers():
    """Compile just the helpers, seeding nothing they do not already have.

    The namespace is bare on purpose. Seeding it with a convenience binding --
    a stub logger, say -- hands the helpers a global that production does not
    have, and every assertion below then passes against a name that raises
    NameError on the cluster. That is not hypothetical: it cost a GLM-5.3 TP=8
    run at ``destroy_nccl`` after 1057 s of init.

    The only things added are the module constants in ``_CONSTS``, copied
    verbatim from the source rather than restated here, so a rename in
    ``vllm_child.py`` surfaces as a missing-constant assertion instead of a
    test that quietly keeps checking a stale value.
    """
    tree = ast.parse(open(_CHILD).read())
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _HELPERS]
    missing = set(_HELPERS) - {n.name for n in nodes}
    assert not missing, f"missing helper(s) in vllm_child.py: {sorted(missing)}"
    consts = [n for n in tree.body
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in _CONSTS
                      for t in n.targets)]
    found = {t.id for n in consts for t in n.targets if isinstance(t, ast.Name)}
    missing_consts = set(_CONSTS) - found
    assert not missing_consts, (
        f"missing constant(s) in vllm_child.py: {sorted(missing_consts)}")
    ns = {}
    exec(compile(ast.Module(body=consts + nodes, type_ignores=[]), _CHILD,
                 "exec"), ns)
    return ns


# --------------------------------------------------------------------------
# Doubles.
# --------------------------------------------------------------------------

class _FakeComm:
    """A device communicator that records which hooks ran."""

    def __init__(self, tag, journal, raises=None, hooks=("checkpoint_prepare",
                                                         "checkpoint_restore"),
                 cpu_group=None, all2all_manager=None):
        self.tag = tag
        self._journal = journal
        self._raises = raises
        self.cpu_group = cpu_group
        self.all2all_manager = all2all_manager
        for name in hooks:
            setattr(self, name, self._make(name))

    def _make(self, name):
        def _hook():
            self._journal.append((name, self.tag))
            if self._raises is not None:
                raise self._raises
        return _hook


class _CommlessComm:
    """An older communicator with no checkpoint protocol at all."""


class _FakeGroup:
    def __init__(self, comm):
        self.device_communicator = comm


class _FakePs:
    def __init__(self, groups, roles=None):
        # Stored as plain objects, not weakrefs, to exercise the non-callable
        # branch; a dead-weakref case is covered separately.
        self._groups = groups
        for role, group in (roles or {}).items():
            setattr(self, f"get_{role}_group", lambda _g=group: _g)


def _ps(pairs, roles=None):
    return _FakePs({name: _FakeGroup(comm) for name, comm in pairs}, roles)


# --------------------------------------------------------------------------
# Doubles for the restore side.
# --------------------------------------------------------------------------

class _FakeGroupHandle:
    """Stands in for a ProcessGroup. Compared by identity, never by value --
    which is the entire bug: vLLM matches a workspace to its creating group with
    ``workspace_group is group``, and a rebuild replaces the object."""

    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return f"<group {self.tag}>"


class _FakeAll2All:
    """A manager whose buffers live on the object, not in a global registry."""

    def __init__(self, journal, initialized=True, raises=None):
        self.cpu_group = None
        self.initialized = initialized
        self._journal = journal
        self._raises = raises

    def checkpoint_prepare(self):
        self._journal.append(("a2a_prepare", id(self)))

    def checkpoint_restore(self):
        if self._raises is not None:
            raise self._raises
        self._journal.append(("a2a_restore", id(self), self.cpu_group))


class _FakeFiar:
    """vLLM's ``flashinfer_all_reduce`` module, with the identity lookup intact.

    Mirrors the real one closely enough to matter: the registry is keyed by
    ``id(workspace)`` and the lookup raises rather than returning ``None`` when a
    workspace has lost its group, so a test that drops an entry instead of
    reassigning it fails the way production would.
    """

    def __init__(self):
        self.workspaces = []
        self._fi_ar_workspace_groups = {}

    def register(self, workspace, group):
        self.workspaces.append(workspace)
        self._fi_ar_workspace_groups[id(workspace)] = group

    def _fi_ar_workspaces_for_group(self, group):
        out = []
        for workspace in self.workspaces:
            owner = self._fi_ar_workspace_groups.get(id(workspace))
            if owner is None:
                raise RuntimeError("FlashInfer all-reduce workspace process "
                                   "group was not retained")
            if owner is group:
                out.append(workspace)
        return out


@contextlib.contextmanager
def _installed(fiar):
    """Put ``fiar`` where ``from vllm.distributed... import`` will find it."""
    path = "vllm.distributed.device_communicators"
    names = ["vllm", "vllm.distributed", path, f"{path}.flashinfer_all_reduce"]
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in names[:-1]:
            sys.modules.setdefault(n, types.ModuleType(n))
        sys.modules[f"{path}.flashinfer_all_reduce"] = fiar
        setattr(sys.modules[path], "flashinfer_all_reduce", fiar)
        yield fiar
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


@contextlib.contextmanager
def _no_flashinfer():
    """A vLLM with no fabric workspaces, i.e. 0.26. ``None`` in ``sys.modules``
    makes the import raise, which is what the helper has to tolerate; leaving it
    to whether vLLM happens to be installed on the machine running the tests
    would make this pass or fail for the wrong reason."""
    path = "vllm.distributed.device_communicators"
    saved = {n: sys.modules.get(n) for n in (path, f"{path}.flashinfer_all_reduce")}
    try:
        sys.modules[path] = None
        sys.modules[f"{path}.flashinfer_all_reduce"] = None
        yield
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


class _FakeSymmHandle:
    """A SymmDeviceMemory: it remembers the device it was built on, and the
    re-map feeds that straight back to the driver."""

    def __init__(self, device_idx):
        self.device_idx = device_idx


class _FakeHandle:
    """``MNNVLAllReduceFusionWorkspace.handle``. FlashInfer's own
    ``checkpoint_prepare``/``checkpoint_restore`` reach the SymmDeviceMemory
    through ``handle.mcast_device_memory``, so this is the shape production
    actually gets."""

    def __init__(self, device_idx):
        self.mcast_device_memory = _FakeSymmHandle(device_idx)


class _FakeWorkspace:
    def __init__(self, device_idx=3):
        self.handle = _FakeHandle(device_idx)


class _LegacyFakeWorkspace:
    """The MnnvlMemory-record shape, which carries ``mem_handles``. Kept
    because FlashInfer has both and neither should be assumed."""

    def __init__(self, device_idx=3):
        self.mem_handles = [_FakeSymmHandle(device_idx)]


class _OpaqueWorkspace:
    """Neither shape. The point is that this is reported, not guessed at."""


class _FakeWorker:
    local_rank = 3


@contextlib.contextmanager
def _cuda_stub(current=None, fail_on=None, attr_values=None):
    """Stub the cuda driver bindings and torch for the context-binding helper.

    ``current`` is the device the thread starts on, or None for "no context",
    which is what ``cuCtxGetDevice`` reports post-restore when nothing has bound
    one. ``journal`` records the driver calls so a test can assert the order.
    """
    journal = []
    state = {"device": current}

    class _CUresult:
        CUDA_SUCCESS = "CUDA_SUCCESS"
        CUDA_ERROR_INVALID_CONTEXT = "CUDA_ERROR_INVALID_CONTEXT"

    drv = types.ModuleType("cuda.bindings.driver")
    drv.CUresult = _CUresult

    def _get_device():
        if state["device"] is None:
            return (_CUresult.CUDA_ERROR_INVALID_CONTEXT,)
        return (_CUresult.CUDA_SUCCESS, state["device"])

    def _device_get(idx):
        journal.append(("cuDeviceGet", idx))
        if fail_on == "cuDeviceGet":
            return (_CUresult.CUDA_ERROR_INVALID_CONTEXT,)
        return (_CUresult.CUDA_SUCCESS, f"dev{idx}")

    def _retain(dev):
        journal.append(("cuDevicePrimaryCtxRetain", dev))
        return (_CUresult.CUDA_SUCCESS, f"ctx-{dev}")

    def _set_current(ctx):
        journal.append(("cuCtxSetCurrent", ctx))
        state["device"] = int(str(ctx).rsplit("dev", 1)[-1])
        return (_CUresult.CUDA_SUCCESS,)

    class _Attr:
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = "fabric"
        CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = "multicast"

    def _get_attribute(attr, idx):
        journal.append(("cuDeviceGetAttribute", attr, idx))
        if fail_on == f"attr:{attr}":
            return (_CUresult.CUDA_ERROR_INVALID_DEVICE,)
        # A successful query returning 0 is the case that matters: the driver
        # answers, and the answer is "not supported". `attr_values` is how a
        # test reaches it.
        return (_CUresult.CUDA_SUCCESS, (attr_values or {}).get(attr, 1))

    _CUresult.CUDA_ERROR_INVALID_DEVICE = "CUDA_ERROR_INVALID_DEVICE"
    drv.CUdevice_attribute = _Attr
    drv.cuDeviceGetAttribute = _get_attribute
    drv.cuCtxGetDevice = _get_device
    drv.cuDeviceGet = _device_get
    drv.cuDevicePrimaryCtxRetain = _retain
    drv.cuCtxSetCurrent = _set_current

    torch_mod = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.set_device = lambda i: journal.append(("torch.set_device", i))
    cuda_mod.device_count = lambda: 8
    torch_mod.cuda = cuda_mod

    names = {"cuda": types.ModuleType("cuda"),
             "cuda.bindings": types.ModuleType("cuda.bindings"),
             "cuda.bindings.driver": drv,
             "torch": torch_mod, "torch.cuda": cuda_mod}
    names["cuda.bindings"].driver = drv
    names["cuda"].bindings = names["cuda.bindings"]
    saved = {n: sys.modules.get(n) for n in names}
    try:
        sys.modules.update(names)
        yield journal
    finally:
        for n, mod in saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


# --------------------------------------------------------------------------
# Target selection: the collective-safety properties.
# --------------------------------------------------------------------------

def test_targets_are_ordered_by_group_name_not_insertion_order():
    """Ranks barrier inside the hook, so they must agree on the order.

    ``_groups`` is insertion-ordered and nothing guarantees every rank inserts
    identically. Built here in deliberately reversed order.
    """
    ns = _load_helpers()
    journal = []
    ps = _ps([("world:0", _FakeComm("w", journal)),
              ("tp:0", _FakeComm("t", journal)),
              ("ep:0", _FakeComm("e", journal))])
    names = [name for name, _ in ns["_communicator_checkpoint_targets"](ps)]
    assert names == ["ep:0", "tp:0", "world:0"], (
        f"targets must be sorted by group name, got {names}")


def test_targets_dedupe_a_shared_communicator():
    ns = _load_helpers()
    journal = []
    shared = _FakeComm("shared", journal)
    ps = _ps([("tp:0", shared), ("world:0", shared),
              ("ep:0", _FakeComm("ep", journal))])
    got = ns["_communicator_checkpoint_targets"](ps)
    assert len(got) == 2, f"shared communicator not deduped: {got}"
    assert [n for n, _ in got] == ["ep:0", "tp:0"]


def test_targets_skip_dead_weakrefs_and_groups_without_a_communicator():
    ns = _load_helpers()
    journal = []
    live = _FakeComm("live", journal)
    ps = _FakePs({
        "a:0": lambda: None,                       # dead weakref
        "b:0": _FakeGroup(None),                   # no communicator
        "c:0": lambda _g=_FakeGroup(live): _g,     # live weakref
    })
    got = ns["_communicator_checkpoint_targets"](ps)
    assert [n for n, _ in got] == ["c:0"], got


def test_targets_are_empty_when_parallel_state_has_no_groups():
    ns = _load_helpers()

    class _Bare:
        pass

    assert ns["_communicator_checkpoint_targets"](_Bare()) == []


# --------------------------------------------------------------------------
# Driving the hooks.
# --------------------------------------------------------------------------

def test_prepare_runs_on_every_communicator_in_sorted_order():
    ns = _load_helpers()
    journal = []
    ps = _ps([("world:0", _FakeComm("w", journal)),
              ("tp:0", _FakeComm("t", journal))])
    out = ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare")
    assert journal == [("checkpoint_prepare", "t"), ("checkpoint_prepare", "w")]
    assert out["ok"] == ["tp:0", "world:0"]
    assert out["failed"] == []


def test_restore_drives_the_other_half_of_the_protocol():
    ns = _load_helpers()
    journal = []
    ps = _ps([("tp:0", _FakeComm("t", journal))])
    out = ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_restore")
    assert journal == [("checkpoint_restore", "t")]
    assert out["ok"] == ["tp:0"]


def test_a_raising_hook_does_not_stop_the_remaining_ranks_work():
    """The barrier is why this must not propagate.

    Every rank runs identical code over an identical group set, so a raise is
    uniform and nobody reached the barrier. Abandoning the loop on one rank
    while the others continue would turn a clean, logged failure into a wedge
    with no diagnostic -- the exact failure mode this whole area keeps hitting.
    """
    ns = _load_helpers()
    journal = []
    ps = _ps([("a:0", _FakeComm("a", journal, raises=NotImplementedError(
                  "Stable-VA checkpointing is unavailable"))),
              ("b:0", _FakeComm("b", journal))])
    out = ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare")
    assert ("checkpoint_prepare", "b") in journal, (
        "a raising communicator stopped the loop; later groups were skipped")
    assert out["ok"] == ["b:0"]
    assert len(out["failed"]) == 1
    assert "NotImplementedError" in out["failed"][0]
    assert "a:0" in out["failed"][0]


def test_a_communicator_without_the_protocol_is_skipped_silently():
    """vLLM 0.26 has no such hooks; the image must still dump."""
    ns = _load_helpers()
    journal = []
    ps = _ps([("a:0", _CommlessComm()), ("b:0", _FakeComm("b", journal))])
    out = ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare")
    assert out["ok"] == ["b:0"]
    assert out["failed"] == []


def test_the_summary_goes_to_stdout(capsys):
    """These helpers run in the worker, which has no ``log`` -- that binding is
    a child-process local. stdout is the worker-side channel, and vLLM prefixes
    it with the rank, which is what makes a per-rank failure readable."""
    ns = _load_helpers()
    journal = []
    ps = _ps([("a:0", _FakeComm("a", journal, raises=RuntimeError("nope"))),
              ("b:0", _FakeComm("b", journal))])
    ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare")
    line = capsys.readouterr().out
    assert "checkpoint_prepare" in line and "ok=b:0" in line
    assert "a:0" in line and "RuntimeError" in line


def test_nothing_is_printed_when_there_are_no_targets():
    ns = _load_helpers()
    out = ns["_run_communicator_checkpoint_hook"](_ps([]), "checkpoint_prepare")
    assert out["ok"] == [] and out["failed"] == []
    assert out["n_workspaces"] == 0 and out["n_all2all"] == 0


# --------------------------------------------------------------------------
# Surviving the rebuild: prepare and restore must move the same memory.
#
# The ordering contract above keeps ranks agreeing with each other. It says
# nothing about the dump side agreeing with the restore side, and that is the
# axis this bug came down: `_reinit_nccl` replaces every ProcessGroup, vLLM
# matches a workspace to its creator with `workspace_group is group`, so the
# restore half looked at a new object, found nothing, and said so as success.
# --------------------------------------------------------------------------

def _prepared_world(journal, with_all2all=False, device_idx=3):
    """A dump-side parallel_state holding one workspace, owned by ``tp:0``."""
    fiar = _FakeFiar()
    old_tp = _FakeGroupHandle("tp-old")
    old_ep = _FakeGroupHandle("ep-old")
    workspace = _FakeWorkspace(device_idx)
    fiar.register(workspace, old_tp)
    a2a = _FakeAll2All(journal) if with_all2all else None
    ps = _ps([("tp:0", _FakeComm("tp", journal, cpu_group=old_tp)),
              ("ep:0", _FakeComm("ep", journal, cpu_group=old_ep,
                                 all2all_manager=a2a))])
    return fiar, ps, workspace, a2a


def _rebuilt(journal, roles=("tp",), groups=None):
    """The restore side: all-new communicators over all-new groups."""
    groups = groups or {r: _FakeGroupHandle(f"{r}-new") for r in roles}
    comms = {r: _FakeComm(f"{r}2", journal, cpu_group=g)
             for r, g in groups.items()}
    ps = _ps([(f"{r}:0", c) for r, c in comms.items()],
             roles={r: _FakeGroup(c) for r, c in comms.items()})
    return ps, groups, comms


def test_prepare_records_the_inventory_on_the_worker():
    ns = _load_helpers()
    journal = []
    fiar, ps, workspace, _ = _prepared_world(journal)
    worker = _FakeWorker()
    with _installed(fiar):
        out = ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare",
                                                      worker)
    stash = getattr(worker, ns["_CKPT_PREPARED_ATTR"])
    assert out["n_workspaces"] == 1 and stash["n_workspaces"] == 1
    assert [r["role"] for r in stash["targets"]] == ["ep", "tp"]
    # A strong reference, not a weakref: `_reinit_nccl` orphans the workspace by
    # replacing the group that owns it, and it has to survive that.
    assert any(workspace in r["workspaces"] for r in stash["targets"])


def test_restore_rekeys_a_workspace_whose_group_was_replaced():
    """The fix. The workspace is re-associated with the group that exists now,
    so the hook's identity lookup finds it and hands FlashInfer a live backend."""
    ns = _load_helpers()
    journal = []
    fiar, ps, workspace, _ = _prepared_world(journal)
    worker = _FakeWorker()
    with _installed(fiar), _cuda_stub():
        ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare", worker)
        new_ps, groups, _ = _rebuilt(journal)
        rekey = ns["_restore_prepared_comm_state"](new_ps, worker)
        restored = ns["_run_communicator_checkpoint_hook"](
            new_ps, "checkpoint_restore", worker)
        ns["_assert_checkpoint_state_restored"](worker, rekey, restored)
    assert fiar._fi_ar_workspace_groups[id(workspace)] is groups["tp"]
    assert rekey["n_workspaces"] == 1 and restored["n_workspaces"] == 1


def test_a_restore_that_finds_nothing_is_caught():
    """Without the re-key this is the observed failure: the hook completes, says
    ``failed=none``, and moves no memory at all. Nothing raised, so only the
    count can tell the difference -- which is why the count exists."""
    ns = _load_helpers()
    journal = []
    fiar, ps, _, _ = _prepared_world(journal)
    worker = _FakeWorker()
    with _installed(fiar):
        ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare", worker)
        new_ps, _, _ = _rebuilt(journal)
        restored = ns["_run_communicator_checkpoint_hook"](
            new_ps, "checkpoint_restore", worker)
        assert restored["failed"] == [], "the no-op still reports success"
        assert restored["n_workspaces"] == 0
        with pytest.raises(RuntimeError) as err:
            ns["_assert_checkpoint_state_restored"](
                worker, {"n_workspaces": 0, "n_all2all": 0, "problems": []},
                restored)
    assert "detached 1 FlashInfer workspace" in str(err.value)


def test_the_prepared_all2all_manager_is_the_one_restored():
    """The manager hangs off the communicator, so the rebuilt one is a different
    object that was never prepared. Only the stashed one has buffers to restore,
    and it needs a live group to do it over."""
    ns = _load_helpers()
    journal = []
    fiar, ps, _, prepared_a2a = _prepared_world(journal, with_all2all=True)
    worker = _FakeWorker()
    with _installed(fiar), _cuda_stub():
        ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare", worker)
        new_ps, groups, comms = _rebuilt(journal, roles=("tp", "ep"))
        comms["ep"].all2all_manager = _FakeAll2All(journal)   # never prepared
        rekey = ns["_restore_prepared_comm_state"](new_ps, worker)
    restored = [e for e in journal if e[0] == "a2a_restore"]
    assert restored == [("a2a_restore", id(prepared_a2a), groups["ep"])], (
        "the rebuilt manager was restored instead of the prepared one")
    assert rekey["n_all2all"] == 1


def test_an_uninitialized_all2all_manager_is_not_counted():
    """vLLM's own hooks no-op on it, so counting it would invent an obligation
    the restore side could never discharge."""
    ns = _load_helpers()
    journal = []
    fiar, _, _, _ = _prepared_world(journal)
    comm = _FakeComm("ep", journal, cpu_group=_FakeGroupHandle("ep-old"),
                     all2all_manager=_FakeAll2All(journal, initialized=False))
    worker = _FakeWorker()
    with _installed(fiar):
        out = ns["_run_communicator_checkpoint_hook"](
            _ps([("ep:0", comm)]), "checkpoint_prepare", worker)
    assert out["n_all2all"] == 0


def test_a_role_with_no_live_group_is_reported_not_skipped():
    """A detached workspace with nowhere to go is the state that faults later,
    so it has to be louder than a `continue`."""
    ns = _load_helpers()
    journal = []
    fiar, ps, _, _ = _prepared_world(journal)
    worker = _FakeWorker()
    with _installed(fiar), _cuda_stub():
        ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare", worker)
        new_ps, _, _ = _rebuilt(journal, roles=("ep",))   # no tp group at all
        rekey = ns["_restore_prepared_comm_state"](new_ps, worker)
    assert any("no live 'tp' group" in p for p in rekey["problems"])
    with pytest.raises(RuntimeError):
        ns["_assert_checkpoint_state_restored"](
            worker, rekey, {"n_workspaces": 0, "n_all2all": 0,
                            "failed_material": []})


def test_nothing_prepared_is_not_an_error():
    """bf16 models never build a workspace and 0.26 has no hooks. The invariant
    is that prepare and restore agree, not that either one found something."""
    ns = _load_helpers()
    journal = []
    worker = _FakeWorker()
    with _installed(_FakeFiar()):
        ns["_run_communicator_checkpoint_hook"](
            _ps([("tp:0", _FakeComm("t", journal))]), "checkpoint_prepare",
            worker)
        new_ps, _, _ = _rebuilt(journal)
        rekey = ns["_restore_prepared_comm_state"](new_ps, worker)
        restored = ns["_run_communicator_checkpoint_hook"](
            new_ps, "checkpoint_restore", worker)
    ns["_assert_checkpoint_state_restored"](worker, rekey, restored)   # no raise


def test_a_failure_on_a_target_holding_nothing_is_not_fatal():
    """A stale group left behind by the rebuild can raise here while owning no
    workspace. Failing the whole restore over that would break restores that
    work today."""
    ns = _load_helpers()
    journal = []
    worker = _FakeWorker()
    bare = _FakeComm("bare", journal, raises=RuntimeError("stale group"),
                     cpu_group=_FakeGroupHandle("stale"))
    with _installed(_FakeFiar()):
        ns["_run_communicator_checkpoint_hook"](
            _ps([("tp:0", _FakeComm("t", journal))]), "checkpoint_prepare",
            worker)
        restored = ns["_run_communicator_checkpoint_hook"](
            _ps([("pp:0", bare)]), "checkpoint_restore", worker)
    assert len(restored["failed"]) == 1 and restored["failed_material"] == []
    ns["_assert_checkpoint_state_restored"](
        worker, {"n_workspaces": 0, "n_all2all": 0, "problems": []}, restored)


def test_a_failure_on_a_target_holding_a_workspace_is_fatal():
    """The counts cannot see this one: the inventory is taken before the call,
    so a raise leaves the number right and the memory unmapped."""
    ns = _load_helpers()
    journal = []
    fiar, ps, workspace, _ = _prepared_world(journal)
    worker = _FakeWorker()
    with _installed(fiar):
        ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare", worker)
        group = _FakeGroupHandle("tp-new")
        fiar._fi_ar_workspace_groups[id(workspace)] = group
        broken = _FakeComm("tp2", journal, raises=RuntimeError("remap failed"),
                           cpu_group=group)
        restored = ns["_run_communicator_checkpoint_hook"](
            _ps([("tp:0", broken)]), "checkpoint_restore", worker)
    assert restored["n_workspaces"] == 1, "the count alone would pass this"
    assert len(restored["failed_material"]) == 1
    with pytest.raises(RuntimeError) as err:
        ns["_assert_checkpoint_state_restored"](
            worker, {"n_workspaces": 1, "n_all2all": 0, "problems": []},
            restored)
    assert "remap failed" in str(err.value)


def test_a_workspace_that_lost_its_group_entry_is_an_error():
    """The re-key reassigns and never deletes, because vLLM's lookup refuses a
    workspace whose creating group was dropped rather than ignoring it."""
    ns = _load_helpers()
    journal = []
    fiar = _FakeFiar()
    group = _FakeGroupHandle("tp-old")
    fiar.register(object(), group)
    fiar._fi_ar_workspace_groups.clear()          # entry dropped, not reassigned
    with _installed(fiar):
        rec = ns["_checkpoint_target_inventory"](
            "tp:0", _FakeComm("t", journal, cpu_group=group))
    assert rec["err"] and "was not retained" in rec["err"]


def test_pre_030_has_no_workspaces_and_that_is_not_an_error():
    """The module simply does not exist there; zero is the honest answer."""
    ns = _load_helpers()
    journal = []
    with _no_flashinfer():
        rec = ns["_checkpoint_target_inventory"](
            "tp:0", _FakeComm("t", journal, cpu_group=_FakeGroupHandle("g")))
    assert rec["workspaces"] == [] and rec["err"] is None


# --------------------------------------------------------------------------
# The CUDA context. FlashInfer's re-map reads the device out of the handle and
# trusts whatever context the caller has current; SymmDeviceMemory.__init__
# establishes one before mapping, and the restore path reaches
# _create_and_map_handles without going through __init__. Observed on
# job 2534c601 as CUDA error 101 (CUDA_ERROR_INVALID_DEVICE) on all 8 ranks.
# --------------------------------------------------------------------------

def test_the_device_comes_from_the_handle_not_from_the_caller():
    """SymmDeviceMemory feeds its own device_idx back into the allocation
    properties and the multicast bind, so that is the device to be on."""
    ns = _load_helpers()
    journal = []
    _fiar, _ps_, workspace, _ = _prepared_world(journal, device_idx=5)
    prepared = {"targets": [{"workspaces": [workspace], "all2all": None}]}
    assert ns["_prepared_device_indices"](prepared) == [5]


def test_the_device_is_read_off_the_path_flashinfer_itself_uses():
    """Job 6e386668 printed `probe={}` on all eight ranks because this looked
    for `mem_handles`, which belongs to MnnvlMemory's allocation record and not
    to the fusion workspace. FlashInfer's own checkpoint_restore reads
    `handle.mcast_device_memory`, so that is the path that has to resolve --
    otherwise `_probe_restore_devices` queries nothing and reports success."""
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_FakeWorkspace(6)],
                             "all2all": None}]}
    assert ns["_prepared_device_indices"](prepared) == [6]


def test_the_mnnvlmemory_record_shape_still_resolves():
    """Both shapes exist in FlashInfer; fixing one must not drop the other."""
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_LegacyFakeWorkspace(2)],
                             "all2all": None}]}
    assert ns["_prepared_device_indices"](prepared) == [2]


def test_a_workspace_carrying_no_device_yields_nothing():
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_OpaqueWorkspace()],
                             "all2all": None}]}
    assert ns["_prepared_device_indices"](prepared) == []


def test_the_local_rank_fallback_says_that_it_fired():
    """The whole reason `want=7 before=7` was over-read on job 6e386668. When
    no workspace carries a device the binding still proceeds, but it must not
    look like a measurement of the workspace's own device_idx."""
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_OpaqueWorkspace()],
                             "all2all": None}],
                "n_workspaces": 1, "n_all2all": 0}
    with _cuda_stub(current=3):
        out = ns["_bind_cuda_context_for_restore"](_FakeWorker(), prepared)
    assert out["want"] == 3, "the fallback still has to bind something"
    assert out["want_src"] == "local_rank_fallback"


def test_a_device_read_off_the_workspace_is_labelled_as_such():
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_FakeWorkspace(5)],
                             "all2all": None}],
                "n_workspaces": 1, "n_all2all": 0}
    with _cuda_stub(current=5):
        out = ns["_bind_cuda_context_for_restore"](_FakeWorker(), prepared)
    assert out["want"] == 5 and out["want_src"] == "workspace"


def test_the_context_is_bound_before_anything_is_remapped():
    ns = _load_helpers()
    journal = []
    fiar, ps, _, _ = _prepared_world(journal, device_idx=5)
    worker = _FakeWorker()
    with _installed(fiar), _cuda_stub(current=None) as calls:
        ns["_run_communicator_checkpoint_hook"](ps, "checkpoint_prepare", worker)
        new_ps, _, _ = _rebuilt(journal)
        rekey = ns["_restore_prepared_comm_state"](new_ps, worker)
    assert ("cuDeviceGet", 5) in calls
    assert ("cuCtxSetCurrent", "ctx-dev5") in calls
    assert ("torch.set_device", 5) in calls
    assert rekey["problems"] == [], rekey["problems"]


def test_a_thread_with_no_context_is_reported_as_such():
    """The discriminator. If `before` is already the right device then the 101
    is coming from a stale device_idx instead, which is a different fix."""
    ns = _load_helpers()
    journal = []
    _fiar, _ps_, workspace, _ = _prepared_world(journal, device_idx=5)
    prepared = {"targets": [{"workspaces": [workspace], "all2all": None}],
                "n_workspaces": 1, "n_all2all": 0}
    with _cuda_stub(current=None):
        out = ns["_bind_cuda_context_for_restore"](_FakeWorker(), prepared)
    assert out["want"] == 5
    assert isinstance(out["before"], str) and "none" in out["before"]
    assert out["after"] == 5
    assert out["problems"] == []


def test_a_context_already_on_the_right_device_is_left_alone():
    ns = _load_helpers()
    journal = []
    _fiar, _ps_, workspace, _ = _prepared_world(journal, device_idx=5)
    prepared = {"targets": [{"workspaces": [workspace], "all2all": None}],
                "n_workspaces": 1, "n_all2all": 0}
    with _cuda_stub(current=5):
        out = ns["_bind_cuda_context_for_restore"](_FakeWorker(), prepared)
    assert out["before"] == 5 and out["after"] == 5 and out["problems"] == []


def test_a_failed_binding_is_a_problem_not_a_warning():
    """FlashInfer's own _verify_cuda_context only logs; that is how the mismatch
    reached the driver in the first place."""
    ns = _load_helpers()
    journal = []
    _fiar, _ps_, workspace, _ = _prepared_world(journal, device_idx=5)
    prepared = {"targets": [{"workspaces": [workspace], "all2all": None}],
                "n_workspaces": 1, "n_all2all": 0}
    with _cuda_stub(current=None, fail_on="cuDeviceGet"):
        out = ns["_bind_cuda_context_for_restore"](_FakeWorker(), prepared)
    assert any("binding device 5" in p for p in out["problems"])


def test_workspaces_spanning_two_devices_is_a_problem():
    """One rank owns one device; two means the inventory crossed ranks."""
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_FakeWorkspace(4),
                                            _FakeWorkspace(5)],
                             "all2all": None}],
                "n_workspaces": 2, "n_all2all": 0}
    with _cuda_stub(current=None):
        out = ns["_bind_cuda_context_for_restore"](_FakeWorker(), prepared)
    assert any("span devices" in p for p in out["problems"])


def test_the_device_probe_asks_the_call_that_actually_failed():
    """`_create_and_map_handles` starts with make_handle_exchanger ->
    is_mnnvl_fabric_supported -> cuDeviceGetAttribute(FABRIC_SUPPORTED), before
    FlashInfer's own context check. Probing it here turns a RuntimeError six
    frames inside a vendor package into a logged line."""
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_FakeWorkspace(5)],
                             "all2all": None}]}
    with _cuda_stub(current=None):
        out = ns["_probe_restore_devices"](prepared)
    assert out["count"] == 8
    assert out["devices"][5] == {"get": "ok", "fabric": 1, "multicast": 1}


def test_a_supported_attribute_is_distinguishable_from_an_answered_query():
    """Job 07fd79b4 printed `multicast: 'ok'` on all eight ranks and that was
    read as "multicast still works after cuda_restore". It meant only that the
    driver answered. A query returning 0 returned CUDA_SUCCESS too, so
    supported and unsupported were the same observation -- in the one probe
    whose entire job is telling them apart."""
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_FakeWorkspace(5)],
                             "all2all": None}]}
    with _cuda_stub(current=None, attr_values={"multicast": 0}):
        out = ns["_probe_restore_devices"](prepared)
    assert out["devices"][5]["multicast"] == 0, (
        "an unsupported device must not read the same as a supported one")
    assert out["devices"][5]["fabric"] == 1


def test_the_device_probe_names_a_bad_ordinal_instead_of_raising():
    ns = _load_helpers()
    prepared = {"targets": [{"workspaces": [_FakeWorkspace(5)],
                             "all2all": None}]}
    with _cuda_stub(current=None, fail_on="attr:fabric"):
        out = ns["_probe_restore_devices"](prepared)
    assert out["devices"][5]["fabric"] == "CUDA_ERROR_INVALID_DEVICE"
    assert out["devices"][5]["get"] == "ok", (
        "cuDeviceGet succeeding while the attribute query fails is the "
        "distinction worth keeping visible")


def test_nothing_prepared_does_not_touch_the_context():
    """Every currently-working config goes down this path; it must stay inert."""
    ns = _load_helpers()
    journal = []
    worker = _FakeWorker()
    with _installed(_FakeFiar()), _cuda_stub() as calls:
        ns["_run_communicator_checkpoint_hook"](
            _ps([("tp:0", _FakeComm("t", journal))]), "checkpoint_prepare",
            worker)
        new_ps, _, _ = _rebuilt(journal)
        ns["_restore_prepared_comm_state"](new_ps, worker)
    assert calls == [], f"the context was touched with nothing to restore: {calls}"


# --------------------------------------------------------------------------
# Call sites.
# --------------------------------------------------------------------------

def _function(name):
    tree = ast.parse(open(_CHILD).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in vllm_child.py")


def _hook_calls(fn_name):
    """``{hook_name: lineno}`` for each hook driven inside the named function."""
    out = {}
    for sub in ast.walk(_function(fn_name)):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "_run_communicator_checkpoint_hook"):
            for arg in sub.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out[arg.value] = sub.lineno
    return out


def _call_lineno(fn_name, callee):
    for sub in ast.walk(_function(fn_name)):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == callee):
            return sub.lineno
    return None


def test_destroy_nccl_prepares_the_communicators():
    got = _hook_calls("_destroy_nccl")
    assert "checkpoint_prepare" in got, (
        "_destroy_nccl must drive checkpoint_prepare, or the fabric-backed "
        "workspaces ride into cuda_checkpoint and it stalls or the restore "
        "returns CUresult=801")


def test_prepare_runs_before_the_nccl_abort():
    """An aborted communicator cannot carry the hook's barrier."""
    prepare = _hook_calls("_destroy_nccl").get("checkpoint_prepare")
    abort = _call_lineno("_destroy_nccl", "_nccl_abort_comms_concurrent")
    assert prepare is not None and abort is not None
    assert prepare < abort, (
        f"checkpoint_prepare (line {prepare}) must precede the NCCL abort "
        f"(line {abort}); after the abort the cross-rank barrier cannot land")


def test_reinit_nccl_restores_the_communicators():
    got = _hook_calls("_reinit_nccl")
    assert "checkpoint_restore" in got, (
        "_reinit_nccl must drive checkpoint_restore, or the workspaces stay "
        "detached and the first collective after the restore faults")


def test_restore_runs_after_the_distributed_environment_is_rebuilt():
    """The hook barriers over the rebuilt cpu_group, so it cannot precede it."""
    restore = _hook_calls("_reinit_nccl").get("checkpoint_restore")
    init = _call_lineno("_reinit_nccl", "init_worker_distributed_environment")
    assert restore is not None and init is not None
    assert init < restore, (
        f"checkpoint_restore (line {restore}) must follow "
        f"init_worker_distributed_environment (line {init})")


def test_both_halves_of_the_protocol_are_wired():
    """Asymmetry is the failure mode: prepare without restore leaves the
    workspaces detached, restore without prepare never fixes the dump."""
    assert "checkpoint_prepare" in _hook_calls("_destroy_nccl")
    assert "checkpoint_restore" in _hook_calls("_reinit_nccl")


def test_prepare_hands_the_hook_the_worker_to_record_on():
    """Without it the inventory is never taken, and the restore side has nothing
    to compare against -- which is the state every run before this one was in."""
    for sub in ast.walk(_function("_destroy_nccl")):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                and sub.func.id == "_run_communicator_checkpoint_hook"):
            assert len(sub.args) >= 3, (
                "checkpoint_prepare is driven without a worker, so nothing "
                "records what it detached")
            return
    raise AssertionError("no hook call found in _destroy_nccl")


def test_the_rekey_runs_before_the_restore_hook():
    """The hook is what actually re-maps the memory, and it can only see a
    workspace whose group entry already points at a live group."""
    rekey = _call_lineno("_reinit_nccl", "_restore_prepared_comm_state")
    restore = _hook_calls("_reinit_nccl").get("checkpoint_restore")
    assert rekey is not None and restore is not None
    assert rekey < restore, (
        f"the re-key (line {rekey}) must precede checkpoint_restore "
        f"(line {restore}); after it the lookup has already missed")


def test_the_count_check_runs_after_the_restore_hook():
    """Not merely last for tidiness: the hook barriers across ranks inside its
    loop, so a rank that raises before it returns strands every other rank."""
    check = _call_lineno("_reinit_nccl", "_assert_checkpoint_state_restored")
    restore = _hook_calls("_reinit_nccl").get("checkpoint_restore")
    assert check is not None and restore is not None
    assert check > restore, (
        f"the count check (line {check}) must follow checkpoint_restore "
        f"(line {restore}); failing inside the loop turns a clean failure "
        f"into a wedge")


def test_the_ep_slot_is_not_gated_on_the_expert_parallel_flag():
    """vLLM builds ep/dp groups for any MoE model; the flag only decides whether
    experts are sharded. Gating on it left GLM-5.3 -- MoE, flag off -- with a
    dead weakref at ``ep:0`` and the rebuilt group registered as ``ep:1``, which
    is what made the two hook lines name different sets."""
    assert _call_lineno("_reinit_nccl", "_semip_ep_enabled") is None, (
        "_reinit_nccl still gates the ep:0/dp:0 rebinding on "
        "enable_expert_parallel")
    src = open(_CHILD).read()
    assert "def _semip_ep_enabled" not in src, (
        "_semip_ep_enabled has no callers left; leaving it invites the gate back")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
