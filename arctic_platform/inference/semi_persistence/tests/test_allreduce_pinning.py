"""The collective-path env vars must be pinned, and pinned in both places.

``ca_graph_rebind`` rewrites the addresses baked into captured CUDA graphs by
matching ``cross_device_reduce`` kernel nodes. That only works if the graphs
contain such nodes, which only holds if vLLM chose the custom all-reduce over
its alternatives. Three env vars decide that, and semi-p pins all three to 0.

Until vLLM 0.30 one of them, ``VLLM_ALLREDUCE_USE_FLASHINFER``, was merely
*assumed* off -- 0.26 defaulted it to False and a docstring in
``ca_graph_rebind`` stated it as fact. 0.30 flipped the default to True, the
captured graphs held FlashInfer all-reduce nodes instead, and the rebind
patched 0 nodes across 2142 graphs while reporting every graph discovered and
topologically fine. Restores then died on
``CUDA error: an illegal memory access was encountered``, up to three minutes
later, because the graphs still referenced pre-restore workspace addresses.

So the property under test is not "FlashInfer is off" -- it is that semi-p
*pins* it rather than inheriting whatever the vLLM of the day defaults to. An
inherited default is invisible until it changes, and then it is invisible
again until something faults.

The second property is symmetry. Capture and reinit must pin the same set: the
graphs are captured under the first and rebound under the second, and a
disagreement means the rebuilt communicator does not match the graphs it is
being rebound into.

``vllm_child.py`` imports torch at module scope, so these are asserted against
the source text by AST rather than by import.

Run from the package directory::

    cd arctic_inference/semi_persistence
    python -m pytest tests/test_allreduce_pinning.py -v
"""
from __future__ import annotations

import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_CHILD = os.path.join(_PKG, "vllm_child.py")
_REBIND = os.path.join(_PKG, "ca_graph_rebind.py")

# Every env var that steers vLLM off the custom all-reduce path. NVLS and
# symmetric memory exchange fds CRIU cannot serialize; FlashInfer produces
# all-reduce nodes the rebind cannot recognise. Different reasons, same
# requirement.
_COLLECTIVE_PINS = (
    "NCCL_NVLS_ENABLE",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_ALLREDUCE_USE_FLASHINFER",
)


def _env_assignments(node):
    """Every ``os.environ["K"] = "V"`` under *node*, as {K: V}."""
    found = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        for tgt in sub.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == "environ"
                    and isinstance(tgt.slice, ast.Constant)
                    and isinstance(sub.value, ast.Constant)):
                found[tgt.slice.value] = sub.value.value
    return found


def _find_function(tree, name):
    """The named FunctionDef *within the given tree*.

    Taking the tree as an argument rather than re-parsing is load-bearing: the
    capture-side scan excludes the reinit function by node identity, and two
    separate parses produce two sets of nodes that can never compare equal. A
    version of this that re-parsed silently excluded nothing, so the capture
    set also contained reinit's pins and the mutation that deletes the capture
    pin still passed.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _function(path, name):
    return _find_function(ast.parse(open(path).read()), name)


def _capture_side_assignments():
    """Env pins on the capture path: the TP>=2 block that precedes LLM().

    Identified as the assignments that are not inside `_reinit_nccl`, since the
    reinit function is the only other place these three are set.
    """
    tree = ast.parse(open(_CHILD).read())
    reinit_nodes = set(map(id, ast.walk(_find_function(tree, "_reinit_nccl"))))
    found = {}
    for sub in ast.walk(tree):
        if id(sub) in reinit_nodes or not isinstance(sub, ast.Assign):
            continue
        for tgt in sub.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == "environ"
                    and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value in _COLLECTIVE_PINS
                    and isinstance(sub.value, ast.Constant)):
                found[tgt.slice.value] = sub.value.value
    return found


def test_reinit_pins_every_collective_env_var():
    got = _env_assignments(_function(_CHILD, "_reinit_nccl"))
    for key in _COLLECTIVE_PINS:
        assert got.get(key) == "0", (
            f"_reinit_nccl must pin {key}=0, got {got.get(key)!r}")


def test_capture_pins_every_collective_env_var():
    got = _capture_side_assignments()
    for key in _COLLECTIVE_PINS:
        assert got.get(key) == "0", (
            f"the capture path must pin {key}=0, got {got.get(key)!r}")


def test_capture_and_reinit_pin_the_same_set():
    """Asymmetry is the failure mode, not absence.

    Graphs captured under one set and rebound under another give a communicator
    that disagrees with the graphs it is rebound into.
    """
    reinit = set(_env_assignments(_function(_CHILD, "_reinit_nccl")))
    capture = set(_capture_side_assignments())
    for key in _COLLECTIVE_PINS:
        assert (key in reinit) == (key in capture), (
            f"{key} is pinned on only one of the two paths: "
            f"capture={key in capture} reinit={key in reinit}")


def _environ_setdefaults(tree):
    """Every ``os.environ.setdefault("K", "V")`` in *tree*, as {K: V}."""
    found = {}
    for sub in ast.walk(tree):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "setdefault"
                and isinstance(sub.func.value, ast.Attribute)
                and sub.func.value.attr == "environ"
                and len(sub.args) == 2
                and all(isinstance(a, ast.Constant) for a in sub.args)):
            found[sub.args[0].value] = sub.args[1].value
    return found


def test_the_workspace_kill_switch_defaults_on_for_tp2_plus():
    """Pinning VLLM_ALLREDUCE_USE_FLASHINFER=0 is necessary and not sufficient.

    That flag gates `cuda_communicator` constructing a FlashInferAllReduce. It
    does not gate `get_fi_ar_workspace`, and job 14f13e5a caught fp8 models
    reaching the getter directly through the DeepSeek-V3.2 layer path during
    `determine_available_memory`. The resulting MNNVL multicast workspace is
    the one allocation `cuCheckpointProcess*` cannot carry, and it cannot be
    rebuilt on the restore side either -- `cuMulticastAddDevice` returns
    CUDA_ERROR_INVALID_DEVICE. Refusing the allocation is the fix, so it has to
    be the default rather than something a payload remembers to set.
    """
    got = _environ_setdefaults(ast.parse(open(_CHILD).read()))
    assert got.get("SEMIP_SUPPRESS_FI_AR_WORKSPACE") == "1", (
        "the workspace kill switch is not defaulted on; a payload that forgets "
        "the env var gets the multicast workspace and a failing reinit_nccl")


def test_the_kill_switch_keeps_an_escape_hatch():
    """`setdefault`, not assignment: the other three pins are unconditional
    because the rebind cannot cope either way, but a future driver or
    FlashInfer able to rebuild multicast would want the workspace back, and
    that has to be reachable from `extra_env` without an image rebuild."""
    tree = ast.parse(open(_CHILD).read())
    assert "SEMIP_SUPPRESS_FI_AR_WORKSPACE" not in _env_assignments(tree), (
        "the kill switch is hard-assigned; extra_env can no longer turn it off")


def test_the_rebind_docstring_no_longer_claims_an_unpinned_default():
    """The 0.30 bug survived review because a docstring asserted the default.

    `install_rank_data_reuse_patch` justified its one-shot torch.empty intercept
    with 'FlashInfer AR off', which was true only by inheritance. If that text
    ever again names the variable without naming where it is pinned, the same
    class of rot is back.
    """
    src = open(_REBIND).read()
    assert "FlashInfer AR off, so" not in src, (
        "the docstring still states FlashInfer AR as an inherited fact")
    assert "VLLM_ALLREDUCE_USE_FLASHINFER=0" in src, (
        "the docstring should name the pinned variable it depends on")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
