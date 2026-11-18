"""Inkling2's fused all-reduce workspace is refused before the model is built.

The ``snowflake-eng/inkling-2`` vLLM fork (``vllm-0.33.0+inkling2``) builds
``Inkling2AllReduceWorkspace`` in ``Inkling2Model.__init__``: FlashInfer
multicast memory owned by the model, which ``cuCheckpointProcess*`` cannot
carry and a restored process cannot re-map. ``create`` returning None is the
fork's own fallback to ``tensor_model_parallel_all_reduce``.

The fork is not installed here, so its module is a double holding the one
class the refusal touches; the child and the worker are read by AST.

    python -m pytest arctic_platform/inference/semi_persistence/tests/test_inkling2_ar_refusal.py -v
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_REBIND = os.path.join(_PKG, "ca_graph_rebind.py")
_CHILD = os.path.join(_PKG, "vllm_child.py")
_WORKER = os.path.join(_PKG, "_semip_worker.py")
_ENV = "SEMIP_SUPPRESS_INKLING2_AR_WORKSPACE"
_ALLREDUCE = "vllm.models.inkling2.ops.allreduce"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Workspace:
    built = 0

    @classmethod
    def create(cls, group, hidden_size):
        cls.built += 1
        return cls()


@pytest.fixture
def cgr():
    return _load(_REBIND, "ca_graph_rebind_inkling2_test")


@pytest.fixture
def fork(monkeypatch):
    """The fork's module tree, with a fresh workspace class per test."""
    workspace = type("Inkling2AllReduceWorkspace", (_Workspace,), {"built": 0})
    names = ("vllm", "vllm.models", "vllm.models.inkling2", "vllm.models.inkling2.ops", _ALLREDUCE)
    mods = {name: types.ModuleType(name) for name in names}
    for parent, child in zip(names, names[1:]):
        setattr(mods[parent], child.rsplit(".", 1)[1], mods[child])
    mods[_ALLREDUCE].Inkling2AllReduceWorkspace = workspace
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return workspace


def test_the_refusal_makes_create_return_none(cgr, fork, monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    # model.py imports the class by name before the worker builds the model.
    from vllm.models.inkling2.ops.allreduce import Inkling2AllReduceWorkspace
    info = cgr.install_inkling2_ar_refusal()
    assert info == {"active": True, "suppress": True, "refused": 0, "err": None}
    assert Inkling2AllReduceWorkspace.create(object(), 4608) is None
    assert fork.built == 0
    assert cgr.inkling2_ar_refusal_state() == {"active": True, "refused": 1}
    cgr.restore_inkling2_ar_refusal()
    assert isinstance(fork.create(object(), 4608), fork)


def test_a_second_install_reports_the_live_state(cgr, fork, monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    cgr.install_inkling2_ar_refusal()
    fork.create(object(), 4608)
    assert cgr.install_inkling2_ar_refusal() == {
        "active": True, "suppress": True, "refused": 1, "err": None}
    assert fork.create(object(), 4608) is None
    cgr.restore_inkling2_ar_refusal()


@pytest.mark.parametrize("value", [None, "0"])
def test_unset_or_zero_leaves_the_workspace_alone(cgr, fork, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, value)
    assert cgr.install_inkling2_ar_refusal()["active"] is False
    assert isinstance(fork.create(object(), 4608), fork)


def test_stock_vllm_is_a_reported_no_op(cgr, monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setitem(sys.modules, "vllm.models.inkling2", None)
    info = cgr.install_inkling2_ar_refusal()
    assert info["active"] is False and info["suppress"] is True
    assert info["err"].startswith(("ModuleNotFoundError", "ImportError"))


def _tp2_setdefaults():
    for node in ast.walk(ast.parse(open(_CHILD).read())):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "_tp"
                and isinstance(node.test.ops[0], ast.GtE)
                and getattr(node.test.comparators[0], "value", None) == 2):
            return {
                sub.args[0].value: sub.args[1].value for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "setdefault" and len(sub.args) == 2
                and all(isinstance(a, ast.Constant) for a in sub.args)
            }
    raise AssertionError("vllm_child.py has no `if _tp >= 2:` block")


def test_the_child_defaults_it_on_and_keeps_an_escape_hatch():
    assert _tp2_setdefaults().get(_ENV) == "1"
    src = open(_CHILD).read()
    assert f'os.environ["{_ENV}"] =' not in src, "hard-assigned; extra_env can no longer turn it off"


def test_the_worker_refuses_before_the_model_is_built():
    tree = ast.parse(open(_WORKER).read())
    load = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_model")
    calls = [n for n in ast.walk(load) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    refuse = [c.lineno for c in calls if c.func.attr == "_semip_refuse_inkling2_ar"]
    build = [c.lineno for c in calls if c.func.attr == "load_model"]
    assert refuse and build and refuse[0] < build[0]
