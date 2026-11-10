"""Where the vLLM child points every JIT cache, and that it does so in time.

A cache left at its node-local default restores on the dump node and nowhere
else. The DeepSeek-V4.1-Flash on_hit restore failed on another node with
``Can't open file home/neutrino/.humming/cache/<hash>/kernel.cubin``: humming
cuModuleLoad()s its cubins from ``HUMMING_CACHE_DIR`` (default
``~/.humming/cache/``), CRIU recorded the mapping, and only ``compilation/``
travels with the image.

``vllm_child`` imports torch and CUDA at module scope, so the table and the
helper are lifted out of the source and run against a temporary model_dir.
"""
from __future__ import annotations

import ast
import os

import pytest

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHILD = os.path.join(_PKG, "vllm_child.py")
_CONSTS = ("_COMPILE_CACHE_DIRS", "_COMPILE_CACHE_ENVS")


def _tree():
    with open(_CHILD) as handle:
        return ast.parse(handle.read(), filename=_CHILD)


def _child_loop():
    return next(n for n in _tree().body
                if isinstance(n, ast.FunctionDef) and n.name == "vllm_child_loop")


@pytest.fixture
def child(monkeypatch):
    body = [n for n in _tree().body
            if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in _CONSTS)
            or (isinstance(n, ast.FunctionDef)
                and n.name == "_pin_compile_caches")]
    assert len(body) == len(_CONSTS) + 1
    ns = {"os": os}
    exec(compile(ast.Module(body=body, type_ignores=[]), _CHILD, "exec"), ns)
    for env in ns["_COMPILE_CACHE_ENVS"]:
        # setenv first, so monkeypatch puts back whatever the helper sets.
        monkeypatch.setenv(env, "")
        monkeypatch.delenv(env)
    return ns


def test_every_cache_lands_under_the_model_dirs_compilation(child, tmp_path):
    root = child["_pin_compile_caches"](str(tmp_path))
    assert root == os.path.join(str(tmp_path), "compilation")
    for env in child["_COMPILE_CACHE_ENVS"]:
        cache_dir = os.environ[env]
        assert cache_dir.startswith(root + os.sep), (env, cache_dir)
        assert os.path.isdir(cache_dir), env


def test_humming_and_the_other_load_by_path_caches_are_pinned(child):
    assert {"TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR",
            "FLASHINFER_WORKSPACE_BASE", "HUMMING_CACHE_DIR",
            "TILELANG_CACHE_DIR", "TVM_FFI_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
            "QUACK_CACHE_DIR", "DG_JIT_CACHE_DIR"} <= set(
                child["_COMPILE_CACHE_ENVS"])


def test_humming_cache_is_its_own_directory(child, tmp_path):
    child["_pin_compile_caches"](str(tmp_path))
    assert os.environ["HUMMING_CACHE_DIR"] == os.path.join(
        str(tmp_path), "compilation", "humming")


def test_deep_gemm_goes_where_vllm_would_have_put_it(child, tmp_path):
    """vLLM sets DG_JIT_CACHE_DIR to VLLM_CACHE_ROOT/deep_gemm when unset."""
    child["_pin_compile_caches"](str(tmp_path))
    assert os.environ["DG_JIT_CACHE_DIR"] == os.path.join(
        os.environ["VLLM_CACHE_ROOT"], "deep_gemm")


def test_no_two_caches_share_a_directory(child):
    subs = [sub for _, sub in child["_COMPILE_CACHE_DIRS"]]
    assert len(subs) == len(set(subs)), subs
    envs = child["_COMPILE_CACHE_ENVS"]
    assert len(envs) == len(set(envs)), envs


def test_the_caches_are_pinned_before_vllm_is_imported():
    loop = _child_loop()
    pins = [n.lineno for n in ast.walk(loop)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_pin_compile_caches"]
    assert len(pins) == 1, pins
    vllm_imports = [n.lineno for n in ast.walk(loop)
                    if (isinstance(n, ast.ImportFrom) and n.module
                        and n.module.split(".")[0] == "vllm")
                    or (isinstance(n, ast.Import)
                        and any(a.name.split(".")[0] == "vllm"
                                for a in n.names))]
    assert vllm_imports, "vllm_child_loop no longer imports vLLM"
    assert pins[0] < min(vllm_imports), (pins, min(vllm_imports))


def test_payload_env_cannot_move_a_pinned_cache():
    """A per-model ``_env`` override would bake an unrestorable path."""
    reserved = None
    for node in ast.walk(_child_loop()):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_RESERVED_ENV"):
            reserved = node.value
    assert isinstance(reserved, ast.Set), "vllm_child_loop lost _RESERVED_ENV"
    assert any(isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)
               and e.value.id == "_COMPILE_CACHE_ENVS"
               for e in reserved.elts)
