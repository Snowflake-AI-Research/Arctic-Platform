"""``matched_baseline`` must give a plain engine exactly what a semi_p engine runs with.

The point of the matched baseline is that a semi_p job and a plain job built
from it agree token for token, so any difference left is semi_p's own. That
only holds while ``matched_baseline`` sets what ``vllm_child`` and ``Instance``
set, so these tests read the TP>=2 block out of ``vllm_child.py`` and the
``worker_cls`` default out of ``instance.py`` by AST (both import torch at
module scope) and compare. A pin added to the child and not here fails them.

Run from the repository root::

    python -m pytest arctic_platform/inference/semi_persistence/tests/test_matched_baseline.py -v
"""
from __future__ import annotations

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import matched_baseline as mb  # noqa: E402 - siblings load by bare name, like the package's own modules
_CHILD = os.path.join(_PKG, "vllm_child.py")
_INSTANCE = os.path.join(_PKG, "instance.py")
_WORKER = os.path.join(os.path.dirname(_PKG), "server", "worker.py")


def _tp2_block():
    """The ``if _tp >= 2:`` block that steers the collectives before LLM()."""
    for node in ast.walk(ast.parse(open(_CHILD).read())):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "_tp"
                and isinstance(node.test.ops[0], ast.GtE)
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == 2):
            return node
    raise AssertionError("vllm_child.py has no `if _tp >= 2:` block")


def _is_multinode_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name) and node.test.func.id == "_is_multinode")


def _walk_single_pod(tree):
    """``ast.walk`` minus ``if _is_multinode():`` bodies, which matched_baseline does not cover."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node) if not _is_multinode_guard(c))


def _environ_assignments(tree, walk=_walk_single_pod):
    found = {}
    for sub in walk(tree):
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute)
                        and tgt.value.attr == "environ" and isinstance(tgt.slice, ast.Constant)
                        and isinstance(sub.value, ast.Constant)):
                    found[tgt.slice.value] = sub.value.value
    return found


def _environ_setdefaults(tree):
    found = {}
    for sub in _walk_single_pod(tree):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "setdefault" and isinstance(sub.func.value, ast.Attribute)
                and sub.func.value.attr == "environ" and len(sub.args) == 2
                and all(isinstance(a, ast.Constant) for a in sub.args)):
            found[sub.args[0].value] = sub.args[1].value
    return found


def _subscript_constant_assignments(tree):
    """Every ``x["k"] = <constant>`` under *tree*, as {k: value}."""
    found = {}
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Constant):
            for tgt in sub.targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant):
                    found[tgt.slice.value] = sub.value.value
    return found


def test_env_matches_the_childs_tp2_block():
    block = _tp2_block()
    assert _environ_assignments(block) == mb.COLLECTIVE_ENV
    assert _environ_setdefaults(block) == mb.COLLECTIVE_ENV_DEFAULTS


def test_shared_experts_stream_is_off_only_across_pods():
    guards = [n for n in ast.walk(_tp2_block()) if _is_multinode_guard(n)]
    across_pods = {}
    for guard in guards:
        across_pods.update(_environ_assignments(guard, walk=ast.walk))
    assert across_pods.get("VLLM_DISABLE_SHARED_EXPERTS_STREAM") == "1"
    assert "VLLM_DISABLE_SHARED_EXPERTS_STREAM" not in mb.COLLECTIVE_ENV


def test_pass_config_matches_the_childs_tp2_block():
    assigned = _subscript_constant_assignments(_tp2_block())
    for key, value in mb.PASS_CONFIG.items():
        assert assigned.get(key) == value, f"the child no longer sets {key}={value!r}"


def test_sleep_mode_matches_the_child():
    assigned = _subscript_constant_assignments(ast.parse(open(_CHILD).read()))
    assert assigned.get("enable_sleep_mode") is True


def test_worker_class_matches_instance():
    tree = ast.parse(open(_INSTANCE).read())
    defaults = [sub.args[1].value for sub in ast.walk(tree)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "setdefault" and len(sub.args) == 2
                and isinstance(sub.args[0], ast.Constant) and sub.args[0].value == "worker_cls"]
    assert defaults == [mb.WORKER_CLS]


def test_plain_worker_applies_it_before_building_engine_args():
    src = open(_WORKER).read()
    applied = src.find("matched_baseline.apply(engine_kwargs)")
    built = src.find("_create_async_engine_args(", applied)
    assert applied != -1 and built > applied
    assert src.find("restore_and_wrap", 0, applied) != -1, "must come after the semi_p branch returns"


def test_below_tp2_nothing_changes():
    kwargs, env = {"tensor_parallel_size": 1}, {}
    assert mb.apply(kwargs, env) == []
    assert kwargs == {"tensor_parallel_size": 1} and env == {}


def test_tp2_matches_and_keeps_what_the_job_set():
    kwargs = {"tensor_parallel_size": 4,
              "compilation_config": {"pass_config": {"enable_noop": True}, "level": 3}}
    env = {"SEMIP_SUPPRESS_FI_AR_WORKSPACE": "0", "PYTHONPATH": "/opt/x"}
    changed = mb.apply(kwargs, env)
    assert changed
    for key, value in mb.COLLECTIVE_ENV.items():
        assert env[key] == value
    assert env["SEMIP_SUPPRESS_FI_AR_WORKSPACE"] == "0", "an explicit extra_env value must win"
    assert kwargs["compilation_config"] == {"pass_config": {"enable_noop": True, "fuse_allreduce_rms": False},
                                            "level": 3}
    assert kwargs["enable_sleep_mode"] is True and kwargs["worker_cls"] == mb.WORKER_CLS
    assert env["PYTHONPATH"].split(os.pathsep) == [mb.SEMIP_DIR, "/opt/x"]
    assert mb.SEMIP_DIR in sys.path


def test_an_explicit_worker_class_is_kept_and_repeat_is_idempotent():
    kwargs, env = {"tensor_parallel_size": 2, "worker_cls": "my.Worker"}, {}
    mb.apply(kwargs, env)
    mb.apply(kwargs, env)
    assert kwargs["worker_cls"] == "my.Worker"
    assert env["PYTHONPATH"] == mb.SEMIP_DIR


def test_requested_reads_the_flag():
    assert mb.requested({mb.MATCH_ENV: "1"})
    assert not mb.requested({mb.MATCH_ENV: "0"}) and not mb.requested({})
