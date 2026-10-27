"""The pinned double buffer plan_restore_weights forms and restore_weights stages through.

``vllm_child`` imports torch and CUDA at module scope, so the double-buffer
functions are lifted out of the source -- the treatment ``test_pod_log`` gives
its helpers -- and run on CPU tensors with the CUDA pieces stubbed: registering
is recorded instead of done, and the side stream and events do nothing. That
checks the bookkeeping (which plans form a double buffer, that every byte
arrives across half boundaries, that nothing stays registered), not the
overlap, which only a GPU shows.
"""
from __future__ import annotations

import ast
import contextlib
import os
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAMES = ("_semip_form_double_buffer", "_semip_free_double_buffer",
          "_semip_double_buffer_h2d", "_semip_plan_load_weights")


class _Nop:
    def __init__(self, *args, **kwargs):
        pass

    def synchronize(self):
        pass

    def record(self, stream=None):
        pass

    def wait_stream(self, other):
        pass


@contextlib.contextmanager
def _stream(_):
    yield


@pytest.fixture
def child():
    """The lifted functions, plus the set of currently registered pointers."""
    path = os.path.join(_PKG, "vllm_child.py")
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    assert {f.name for f in funcs} == set(_NAMES)
    registered = set()
    cuda = types.SimpleNamespace(
        Event=_Nop, Stream=_Nop, stream=_stream,
        current_stream=lambda: _Nop(),
        mem_get_info=lambda device: (1 << 40, 1 << 40))
    ns = {
        "torch": types.SimpleNamespace(empty=torch.empty, uint8=torch.uint8,
                                       cuda=cuda),
        "_repin_buffer": lambda buf: registered.add(buf.data_ptr()),
        "_unpin_buffer": lambda buf: registered.remove(buf.data_ptr()),
        "_STAGING_FREE_FRACTION": 0.9,
    }
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    ns["registered"] = registered
    return types.SimpleNamespace(**ns)


def _worker(total=10_000, pinned=False):
    buf = (torch.arange(total) % 251).to(torch.uint8)
    half = total // 2
    index = {"main:p:a": (0, half, torch.uint8, (half,)),
             "main:p:b": (half, total - half, torch.uint8, (total - half,))}
    return types.SimpleNamespace(_semip_buf=buf, _semip_index=index,
                                 _semip_pinned=pinned, device="cpu")


def test_a_pageable_buffer_gets_the_double_buffer_it_asks_for(child):
    worker = _worker()
    out = child._semip_plan_load_weights(worker, None, 4096, 2)
    db = worker._semip_double_buffer
    assert len(db["halves"]) == 2 and db["half_bytes"] == 2048
    assert db["threads"] == 2
    assert len(child.registered) == 2
    assert (out["double_buffer_bytes"], out["pinned"]) == (4096, False)


@pytest.mark.parametrize("pinned,buffer_bytes", [
    (True, 4096),    # a repinned host copy is copied directly
    (False, 0),      # SEMIP_PINNED_BUFFER_BYTES=0
    (False, None),   # a caller that sends no settings
])
def test_no_double_buffer_when_pinned_or_asked_for_none(child, pinned, buffer_bytes):
    worker = _worker(pinned=pinned)
    out = child._semip_plan_load_weights(worker, None, buffer_bytes, 2)
    assert getattr(worker, "_semip_double_buffer", None) is None
    assert out["double_buffer_bytes"] == 0 and not child.registered


def test_planning_again_replaces_the_double_buffer_instead_of_leaking_it(child):
    worker = _worker()
    child._semip_plan_load_weights(worker, None, 4096, 2)
    child._semip_plan_load_weights(worker, None, 2048, 2)
    assert len(child.registered) == 2   # 4 if the first double buffer had leaked
    assert child._semip_free_double_buffer(worker) == 2048
    assert not child.registered and worker._semip_double_buffer is None


@pytest.mark.parametrize("n,half_bytes,threads", [
    (10_000, 1024, 3),    # alternates halves ten times, ragged last piece
    (1000, 1024, 4),      # smaller than one half
    (2048, 1024, 1),      # exactly fills both halves
])
def test_every_byte_arrives_half_by_half(child, n, half_bytes, threads):
    worker = _worker(total=n)
    child._semip_form_double_buffer(worker, 2 * half_bytes, threads)
    src = worker._semip_buf
    dst = torch.zeros(n + 7, dtype=torch.uint8)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        child._semip_double_buffer_h2d(worker._semip_double_buffer, src, dst[:n], pool)
    assert torch.equal(dst[:n], src)
    assert not dst[n:].any()
