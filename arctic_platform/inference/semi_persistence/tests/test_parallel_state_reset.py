"""Unit tests for the parallel_state reset that precedes every NCCL reinit.

``_force_dist_uninitialized_for_restore`` exists because an aborted teardown
leaves ``parallel_state`` populated in the CRIU image, and vLLM's
``initialize_model_parallel`` asserts each of its groups is ``None`` before
rebuilding. A group left behind does not degrade the restore -- it ends it,
with ``AssertionError: <name> group is already initialized`` 47 ms in.

The vLLM 0.30 upgrade is what these tests are made of. ``parallel_state`` there
holds two groups that did not exist in 0.26, ``_ETP`` and ``_ENGRAM_DP``, and
the reset was a hand-kept list that named neither, so every TP=1 restore died
on the first of them. A list that has to be edited for each new vLLM will miss
the next one the same way, which is why discovery -- clearing every
module-level ``GroupCoordinator`` -- carries the weight now and the names are
only a floor.

Both halves are tested, because neither is sufficient alone. Discovery cannot
see a group that is already ``None`` or a non-coordinator like ``_NODE_COUNT``;
the names cannot see a group that does not exist yet.

``vllm_child.py`` imports torch at module scope and is not importable here, so
the function is lifted out of the source by AST and run against a stub
``parallel_state``. That keeps the test on the shipped text rather than a copy.

Run from the package directory::

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_parallel_state_reset.py -v
"""
from __future__ import annotations

import ast
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_CHILD = os.path.join(_PKG, "vllm_child.py")

# Read off vLLM v0.30.0's `initialize_model_parallel`, which asserts each of
# these is None before it will rebuild. Pinned as a fixture rather than derived
# so that a vLLM bump which adds an eleventh fails this test loudly instead of
# failing a restore quietly -- the same reasoning as the live label fixtures in
# test_publish_layout.py.
_ASSERTED_IN_VLLM_030 = (
    "_DCP", "_DP", "_ENGRAM_DP", "_EP", "_EPLB",
    "_ETP", "_PCP", "_PP", "_TP", "_WORLD",
)


def _lift(path, name):
    """Exec one top-level def out of *path* and return it."""
    tree = ast.parse(open(path).read())
    ns = {"os": os, "sys": sys}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in {path}")


def _named_tuple_from_source():
    """The literal floor list the source carries, as a tuple of strings."""
    tree = ast.parse(open(_CHILD).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_force_dist_uninitialized_for_restore"):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign)
                        and getattr(sub.targets[0], "id", "") == "_named"):
                    return tuple(ast.literal_eval(sub.value))
    raise AssertionError("_named not found")


class _StubCoordinator:
    """Stands in for vllm.distributed.parallel_state.GroupCoordinator."""


def _install_stub_parallel_state(groups, with_coordinator=True):
    """Put a fake ``vllm.distributed.parallel_state`` on sys.modules.

    *groups* maps attribute name -> value. Returns the module so a test can
    read the attributes back after the reset runs.
    """
    ps = types.ModuleType("vllm.distributed.parallel_state")
    if with_coordinator:
        ps.GroupCoordinator = _StubCoordinator
    for k, v in groups.items():
        setattr(ps, k, v)
    dist_pkg = types.ModuleType("vllm.distributed")
    dist_pkg.parallel_state = ps
    root = types.ModuleType("vllm")
    root.distributed = dist_pkg
    sys.modules["vllm"] = root
    sys.modules["vllm.distributed"] = dist_pkg
    sys.modules["vllm.distributed.parallel_state"] = ps
    return ps


def _run_reset(groups, with_coordinator=True):
    """Run the lifted reset against a stub parallel_state; return the stub."""
    keys = ("vllm", "vllm.distributed", "vllm.distributed.parallel_state")
    saved = {k: sys.modules.get(k) for k in keys}
    try:
        ps = _install_stub_parallel_state(groups, with_coordinator)
        _lift(_CHILD, "_force_dist_uninitialized_for_restore")()
        return ps
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_every_group_vllm_030_asserts_on_is_in_the_floor():
    """The named list must cover every group 0.30 refuses to rebuild over.

    This is the regression itself: `_ETP` and `_ENGRAM_DP` were both absent,
    and fixing only the one that appeared in the traceback would have moved the
    failure to the other.
    """
    named = _named_tuple_from_source()
    missing = [g for g in _ASSERTED_IN_VLLM_030 if g not in named]
    assert not missing, (
        f"groups vLLM 0.30 asserts on but the reset omits: {missing}")


def test_named_groups_are_cleared():
    ps = _run_reset({
        "_WORLD": _StubCoordinator(),
        "_TP": _StubCoordinator(),
        "_ETP": _StubCoordinator(),
        "_ENGRAM_DP": _StubCoordinator(),
        "_NODE_COUNT": 8,
    })
    for name in ("_WORLD", "_TP", "_ETP", "_ENGRAM_DP", "_NODE_COUNT"):
        assert getattr(ps, name) is None, f"{name} survived the reset"


def test_a_group_no_list_knows_about_is_discovered():
    """The self-defending half: an unnamed coordinator is still cleared.

    A future vLLM adding `_QUUX_TP` must not reintroduce the 0.30 failure, so
    membership of the hand-kept list cannot be what decides this.
    """
    ps = _run_reset({
        "_TP": _StubCoordinator(),
        "_QUUX_TP": _StubCoordinator(),
    })
    assert ps._QUUX_TP is None, "an unnamed GroupCoordinator was left behind"


def test_non_group_attributes_are_left_alone():
    """Discovery must not scribble over unrelated module state."""
    ps = _run_reset({
        "_TP": _StubCoordinator(),
        "_SOME_CONSTANT": 1234,
    })
    assert ps._SOME_CONSTANT == 1234


def test_reset_survives_a_parallel_state_without_the_coordinator_class():
    """A stubbed or older vLLM missing GroupCoordinator must not raise.

    The reset runs on the restore path where nothing can be retried, so a
    discovery failure has to degrade to the named floor rather than propagate.
    """
    ps = _run_reset({"_TP": _StubCoordinator()}, with_coordinator=False)
    assert ps._TP is None, "the named floor must still apply"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
