"""The LoRA state a serving child stages before sleep and puts back on wake.

``vllm_child`` imports torch and CUDA at module scope, so the LoRA functions
are lifted out of the source, as ``test_double_buffer`` lifts its helpers, and
run on CPU tensors posing as CUDA ones. vLLM's LoRA layer base class is a
stand-in: what is checked is which storages are found -- every LoRA tensor
once, nothing the model owns, nothing on the host -- and that their bytes
survive a sleep, not the copies' speed, which only a GPU shows.
"""
from __future__ import annotations

import ast
import dataclasses
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAMES = ("_semip_lora_manager", "_semip_lora_views", "_semip_stage_lora", "_semip_restore_lora")


class _Cuda(torch.Tensor):
    """A CPU tensor that reports itself as a CUDA one."""

    is_cuda = True


def _cuda(*shape):
    return torch.rand(*shape).as_subclass(_Cuda)


class _LoRALayer(torch.nn.Module):
    pass


@dataclasses.dataclass
class _KernelMeta:
    """Stands in for vLLM's LoRAKernelMeta: device tensors, and host ones."""

    token_lora_mapping: torch.Tensor
    no_lora_flag_cpu: torch.Tensor


class _Torch:
    """torch with a no-op ``cuda.synchronize``: these tensors are on the host."""

    cuda = types.SimpleNamespace(synchronize=lambda *a: None)

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def child(monkeypatch):
    path = os.path.join(_PKG, "vllm_child.py")
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    assert {f.name for f in funcs} == set(_NAMES)
    namespace = {"torch": _Torch()}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), namespace)
    layers = types.ModuleType("vllm.lora.layers")
    layers.BaseLayerWithLoRA = _LoRALayer
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.layers", layers)
    return types.SimpleNamespace(**{name: namespace[name] for name in _NAMES})


def _worker(lora=True):
    """A model with one LoRA layer and a punica wrapper, the way vLLM lays them out."""
    model = torch.nn.Module()
    model.base = torch.nn.Linear(4, 4)
    layer = _LoRALayer()
    w13 = _cuda(2, 3, 4)
    layer.lora_a_stacked = (_cuda(2, 1, 8, 4),)
    layer.lora_b_stacked = [_cuda(2, 1, 4, 8)]
    layer.w13_lora_a_stacked = (w13,)
    layer.flat_views = [w13[0], w13[1]]                      # views of w13: one storage
    layer.adapter_enabled = _cuda(3)
    layer.base_weight = model.base.weight.data.as_subclass(_Cuda)  # owned by the model
    layer.host_copy = torch.rand(5)                         # not on the device
    layer.slices = (4, 4)
    model.layer = layer
    punica = types.SimpleNamespace(
        _token_lora_indices=_cuda(16), device="cuda",
        token_mapping_meta=_KernelMeta(_cuda(16), torch.ones(1)))
    manager = types.SimpleNamespace(punica_wrapper_mapping={"language": punica},
                                    _last_mapping="mapping", _last_slot_layout=(1,))
    runner = types.SimpleNamespace(
        model=model,
        lora_manager=types.SimpleNamespace(_adapter_manager=manager) if lora else None)
    lora_tensors = [layer.lora_a_stacked[0], layer.lora_b_stacked[0], w13,
                    layer.adapter_enabled, punica._token_lora_indices,
                    punica.token_mapping_meta.token_lora_mapping]
    return types.SimpleNamespace(model_runner=runner), lora_tensors


def _nbytes(tensors):
    return sum(t.untyped_storage().nbytes() for t in tensors)


def test_finds_each_lora_storage_once_and_nothing_the_model_owns(child):
    worker, lora = _worker()
    views = child._semip_lora_views(worker)
    assert len(views) == len(lora)
    assert sum(v.numel() for v in views) == _nbytes(lora)
    owned = {p.untyped_storage().data_ptr() for p in worker.model_runner.model.parameters()}
    assert not owned & {v.untyped_storage().data_ptr() for v in views}


def test_wake_puts_back_what_sleep_staged(child):
    worker, lora = _worker()
    before = [t.clone() for t in lora]
    base = worker.model_runner.model.base.weight.detach().clone()
    assert child._semip_stage_lora(worker) == _nbytes(lora)
    for t in lora:
        t.fill_(7)                                          # what a level-2 sleep leaves
    assert child._semip_restore_lora(worker) == _nbytes(lora)
    for t, b in zip(lora, before):
        assert torch.equal(t, b)
    assert torch.equal(worker.model_runner.model.base.weight, base)


def test_the_first_restore_has_nothing_staged_and_zeroes_the_state(child):
    worker, lora = _worker()
    child._semip_restore_lora(worker)
    assert all(not t.any() for t in lora)
    assert worker.model_runner.model.layer.host_copy.any()


def test_a_restore_makes_vllm_rebuild_the_punica_metadata(child):
    worker, _ = _worker()
    child._semip_restore_lora(worker)
    manager = worker.model_runner.lora_manager._adapter_manager
    assert manager._last_mapping is None and manager._last_slot_layout is None


def test_a_restage_reuses_the_host_buffers(child):
    worker, _ = _worker()
    child._semip_stage_lora(worker)
    bufs = worker._semip_lora_bufs
    child._semip_stage_lora(worker)
    assert worker._semip_lora_bufs is bufs


def test_an_engine_without_lora_has_no_lora_state(child):
    worker, _ = _worker(lora=False)
    assert child._semip_lora_views(worker) == []
    assert child._semip_stage_lora(worker) == 0
    assert child._semip_restore_lora(worker) == 0


def test_a_layout_that_changed_since_staging_is_refused(child):
    worker, _ = _worker()
    child._semip_stage_lora(worker)
    worker.model_runner.model.layer.extra = _cuda(9)
    with pytest.raises(RuntimeError, match="layout changed"):
        child._semip_restore_lora(worker)
