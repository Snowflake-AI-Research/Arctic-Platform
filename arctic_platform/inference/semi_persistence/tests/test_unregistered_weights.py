"""Weights vLLM keeps outside ``named_parameters()`` travel in the host copy.

vLLM's triton_kernels MXFP4 MoE backends (gpt-oss-120b on H200) delete the
expert Parameters after loading and hold the swizzled experts as plain layer
attributes wrapped in triton_kernels dataclasses, with their scales in the quant
method's PrecisionConfig. A host copy of the parameters alone was 4.64 GiB of a
66 GiB model, the level-2 sleep discarded the rest, and the restored engine ran
on whatever memory wake_up mapped back.

``vllm_child`` imports torch and CUDA at module scope, so the staging functions
are lifted out of the source, as ``test_lora_host_copy`` lifts its own, and run
on CPU tensors posing as CUDA ones. vLLM's allocator is a stand-in recording
which storages sit in the "weights" pool.
"""
from __future__ import annotations

import ast
import re
import dataclasses
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAMES = ("_semip_weight_pool_ranges", "_semip_unregistered_storages",
          "_semip_param_tensors", "_semip_layout", "_semip_attach", "_semip_stage",
          "_semip_restore_weights", "_semip_lora_manager", "_semip_lora_views",
          "_semip_weight_like", "_semip_state_views", "_semip_stage_state",
          "_semip_restore_state", "_semip_digest", "_semip_weight_like_digests",
          "_semip_record_weight_digests", "_semip_check_weight_digests",
          "_semip_mark_volatile")
_CONSTANTS = ("_WEIGHT_LIKE_PATH", "_DIGEST_CHUNK")
GiB = 2**30


class _Cuda(torch.Tensor):
    """A CPU tensor that reports itself as a CUDA one."""

    is_cuda = True


class _Torch:
    """torch with no-op CUDA synchronization: these tensors are on the host."""

    cuda = types.SimpleNamespace(synchronize=lambda *a: None, empty_cache=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)


class _Allocator:
    """Stands in for vLLM's CuMemAllocator: one allocation per storage."""

    instance = None

    def __init__(self):
        self.pointer_to_data = {}

    def alloc(self, tag, *shape, dtype=torch.float32):
        t = (torch.randint(1, 256, shape, dtype=dtype) if dtype == torch.uint8
             else torch.rand(*shape, dtype=dtype)).as_subclass(_Cuda)
        storage = t.untyped_storage()
        self.pointer_to_data[storage.data_ptr()] = types.SimpleNamespace(
            handle=(0, storage.nbytes(), storage.data_ptr(), 0), tag=tag)
        return t


class _LoRALayer(torch.nn.Module):
    pass


# triton_kernels' wrappers, as the MXFP4 TRITON backend leaves them.
@dataclasses.dataclass
class _Storage:
    data: torch.Tensor
    layout: object


@dataclasses.dataclass
class _TritonTensor:
    storage: _Storage
    dtype: str
    shape: list


@dataclasses.dataclass
class _FlexCtx:
    rhs_data: object


@dataclasses.dataclass
class _PrecisionConfig:
    weight_scale: _TritonTensor
    flex_ctx: _FlexCtx


class _QuantMethod:
    """GptOssMxfp4MoEMethod after process_weights_after_loading: a plain object."""

    def apply(self):
        pass


@pytest.fixture
def child(monkeypatch):
    path = os.path.join(_PKG, "vllm_child.py")
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    assert {f.name for f in funcs} == set(_NAMES)
    ns = {"torch": _Torch(), "ThreadPoolExecutor": ThreadPoolExecutor, "time": time,
          "re": re}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in _CONSTANTS
                                                for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)

    allocator = _Allocator()
    cumem = types.ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = type("CuMemAllocator", (), {"instance": allocator})
    layers = types.ModuleType("vllm.lora.layers")
    layers.BaseLayerWithLoRA = _LoRALayer
    for name, mod in {"vllm": types.ModuleType("vllm"),
                      "vllm.device_allocator": types.ModuleType("vllm.device_allocator"),
                      "vllm.device_allocator.cumem": cumem,
                      "vllm.lora": types.ModuleType("vllm.lora"),
                      "vllm.lora.layers": layers}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    ns["allocator"] = allocator
    ns["cumem"] = cumem
    return types.SimpleNamespace(**ns)


def _param(allocator, *shape, dtype=torch.float32):
    return torch.nn.Parameter(allocator.alloc("weights", *shape, dtype=dtype),
                              requires_grad=False)


def _gpt_oss(allocator, n_layers=2, lora=False):
    """A gpt-oss-shaped model after the TRITON MXFP4 backend has processed it.

    Returns the worker and the storages a correct host copy must hold beyond
    the parameters.
    """
    model = torch.nn.Module()
    model.embed = torch.nn.Module()
    model.embed.weight = _param(allocator, 8, 4)
    model.register_buffer("cos_sin_cache", allocator.alloc("weights", 6))
    model.layers = torch.nn.ModuleList()
    model.host_table = torch.rand(3)                         # not on the device
    unregistered = []
    for _ in range(n_layers):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv = _param(allocator, 4, 4)
        block.attn.kv_cache = [allocator.alloc("kv_cache", 64)]   # another pool
        experts = torch.nn.Module()
        experts.w13_bias = _param(allocator, 2, 6)
        experts.w2_bias = _param(allocator, 2, 4)
        w13 = allocator.alloc("weights", 2, 6, 2, dtype=torch.uint8)
        w2 = allocator.alloc("weights", 2, 4, 3, dtype=torch.uint8)
        w13_scale = allocator.alloc("weights", 2, 6, dtype=torch.uint8)
        w2_scale = allocator.alloc("weights", 2, 4, dtype=torch.uint8)
        experts.w13_weight = _TritonTensor(_Storage(w13.transpose(1, 2), "swizzled"),
                                           "fp4", [2, 6, 4])
        experts.w2_weight = _TritonTensor(_Storage(w2, "swizzled"), "fp4", [2, 4, 6])
        qm = _QuantMethod()
        qm.w13_precision_config = _PrecisionConfig(
            _TritonTensor(_Storage(w13_scale, "scale"), "e8m0", [2, 6]), _FlexCtx(None))
        qm.w2_precision_config = _PrecisionConfig(
            _TritonTensor(_Storage(w2_scale, "scale"), "e8m0", [2, 4]), _FlexCtx(None))
        # The kernel's quant config holds the same objects again, and a bias
        # parameter: neither may be taken twice.
        qm.moe_quant_config = types.SimpleNamespace(
            w1_scale=qm.w13_precision_config, w1_bias=experts.w13_bias,
            w2_scale=qm.w2_precision_config)
        experts.quant_method = qm
        block.experts = experts
        model.layers.append(block)
        unregistered += [w13, w2, w13_scale, w2_scale]
    manager = None
    if lora:
        holder = _LoRALayer()
        holder.lora_a_stacked = (allocator.alloc("weights", 2, 4),)
        model.lora = holder
        manager = types.SimpleNamespace(punica_wrapper_mapping={})
    runner = types.SimpleNamespace(
        model=model,
        lora_manager=types.SimpleNamespace(_adapter_manager=manager) if manager else None)
    return types.SimpleNamespace(model_runner=runner, device="cpu"), unregistered


def _dense(allocator):
    model = torch.nn.Module()
    model.embed = torch.nn.Linear(4, 8)
    model.proj = torch.nn.Module()
    model.proj.weight = _param(allocator, 4, 4)
    model.register_buffer("cos_sin_cache", allocator.alloc("weights", 6))
    return types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model),
                                 device="cpu")


def _storage_bytes(tensors):
    return sum(t.untyped_storage().nbytes() for t in tensors)


def test_the_swizzled_experts_and_their_scales_are_in_the_layout(child):
    worker, unregistered = _gpt_oss(child.allocator)
    layout = child._semip_layout(worker)
    names = [name for name, *_ in layout]
    params = [f"main:p:{n}" for n, _ in worker.model_runner.model.named_parameters()]
    assert names[:len(params)] == params
    extra = layout[len(params):]
    assert all(name.startswith("main:s:") for name, *_ in extra)
    assert len(extra) == len(unregistered)
    assert sum(nb for _, nb, _, _ in extra) == _storage_bytes(unregistered)
    assert "main:s:layers.0.experts.w13_weight.storage.data" in names
    assert ("main:s:layers.1.experts.quant_method.w2_precision_config"
            ".weight_scale.storage.data") in names


def test_kv_cache_buffers_host_tensors_and_lora_slots_stay_out(child):
    worker, unregistered = _gpt_oss(child.allocator, lora=True)
    tensors = child._semip_param_tensors(worker)
    extra = {v.untyped_storage().data_ptr()
             for name, v in tensors.items() if ":s:" in name}
    assert extra == {t.untyped_storage().data_ptr() for t in unregistered}


def test_a_model_with_only_parameters_keeps_the_layout_it_had(child):
    worker = _dense(child.allocator)
    before = [(f"main:p:{n}", p.data.nbytes, p.data.dtype, tuple(p.data.shape))
              for n, p in worker.model_runner.model.named_parameters()]
    assert child._semip_layout(worker) == before


def test_without_vllms_allocator_the_layout_is_the_parameters(child):
    child.cumem.CuMemAllocator.instance = None
    worker, _ = _gpt_oss(child.allocator)
    assert all(":p:" in name for name, *_ in child._semip_layout(worker))


def test_a_sleep_that_discards_the_experts_is_undone_by_restore(child):
    worker, unregistered = _gpt_oss(child.allocator)
    params = [p for p in worker.model_runner.model.parameters()]
    total, n_entries, n_storages, storage_bytes = child._semip_attach(worker)
    assert n_storages == len(unregistered)
    assert storage_bytes == _storage_bytes(unregistered)
    assert total == storage_bytes + sum(p.data.nbytes for p in params)
    child._semip_stage(worker)
    before = [t.clone() for t in [*unregistered, *params]]

    for t in [*unregistered, *params]:          # level-2 sleep, then wake_up
        t.data.zero_()
    out = child._semip_restore_weights(worker)

    assert out["loaded"] == n_entries
    assert all(torch.equal(a, b) for a, b in zip(before, [*unregistered, *params]))


def _with_scratch(allocator, tokens):
    """A dense model plus the kind of config-sized scratch a sparse-attention
    model keeps in the weight pool (GLM-5.3's topk_indices_buffer)."""
    worker = _dense(allocator)
    worker.model_runner.model.topk_indices_buffer = allocator.alloc(
        "weights", tokens, 8)
    return worker


def test_config_sized_scratch_stays_out_of_the_weight_layout(child):
    small = child._semip_layout(_with_scratch(child.allocator, 4096))
    large = child._semip_layout(_with_scratch(child.allocator, 8192))
    assert [r for r in small if r[0].startswith("main:p:")] == \
        [r for r in large if r[0].startswith("main:p:")]
    assert not any(":s:" in r[0] for r in small + large)
    state = child._semip_state_views(_with_scratch(child.allocator, 4096))
    assert [name for name, _ in state] == ["main:s:topk_indices_buffer"]


def test_state_is_staged_and_put_back(child):
    worker = _with_scratch(child.allocator, 16)
    buf = worker.model_runner.model.topk_indices_buffer
    expected = buf.clone()
    assert child._semip_stage_state(worker) == buf.untyped_storage().nbytes()
    buf.zero_()
    assert child._semip_restore_state(worker) == buf.untyped_storage().nbytes()
    assert torch.equal(buf, expected)


def test_nothing_staged_leaves_state_alone(child):
    worker = _with_scratch(child.allocator, 16)
    buf = worker.model_runner.model.topk_indices_buffer
    before = buf.clone()
    assert child._semip_restore_state(worker) == 0
    assert torch.equal(buf, before)


def _with_weight_like(allocator):
    """A dense model plus two weight-pool storages named like weights: one the
    model reads, one a kernel writes while it serves."""
    worker = _dense(allocator)
    worker.model_runner.model.proj.packed_weight = allocator.alloc("weights", 4, 4)
    worker.model_runner.model.proj.staged_weight = allocator.alloc("weights", 4, 4)
    return worker


_PACKED = "main:s:proj.packed_weight"
_STAGED = "main:s:proj.staged_weight"


def _serve(worker):
    worker.model_runner.model.proj.staged_weight.add_(1.0)


def test_a_weight_like_storage_written_while_serving_becomes_state(child):
    worker = _with_weight_like(child.allocator)
    assert child._semip_record_weight_digests(worker) == 2
    _serve(worker)
    assert child._semip_check_weight_digests(worker) == [_STAGED]
    names = [name for name, *_ in child._semip_layout(worker)]
    assert _PACKED in names and _STAGED not in names
    assert [name for name, _ in child._semip_state_views(worker)] == [_STAGED]


def test_weight_like_storages_left_alone_stay_weights(child):
    worker = _with_weight_like(child.allocator)
    child._semip_record_weight_digests(worker)
    assert child._semip_check_weight_digests(worker) == []
    names = [name for name, *_ in child._semip_layout(worker)]
    assert _PACKED in names and _STAGED in names
    assert child._semip_state_views(worker) == []


def test_the_check_runs_once(child):
    worker = _with_weight_like(child.allocator)
    child._semip_record_weight_digests(worker)
    assert child._semip_check_weight_digests(worker) == []
    _serve(worker)
    assert child._semip_check_weight_digests(worker) == []
    assert _STAGED in [name for name, *_ in child._semip_layout(worker)]


def test_without_digests_the_check_reads_nothing(child, monkeypatch):
    worker = _with_weight_like(child.allocator)
    _serve(worker)

    def _asleep(*a, **k):
        raise AssertionError("storages read")

    monkeypatch.setitem(child._semip_check_weight_digests.__globals__,
                        "_semip_weight_like_digests", _asleep)
    assert child._semip_check_weight_digests(worker) == []
    assert not getattr(worker, "_semip_volatile", None)
    worker._semip_weight_digests = {_STAGED: (0, 0)}
    worker._semip_digests_checked = True          # restored: checked at the dump
    assert child._semip_check_weight_digests(worker) == []


def test_every_rank_takes_what_any_rank_saw_change(child):
    worker = _with_weight_like(child.allocator)
    child._semip_record_weight_digests(worker)
    assert child._semip_check_weight_digests(worker) == []
    child._semip_mark_volatile(worker, [_STAGED])
    assert _STAGED not in [name for name, *_ in child._semip_layout(worker)]


def test_the_checksum_sees_a_permutation_and_ignores_the_chunking(child, monkeypatch):
    base = torch.arange(40, dtype=torch.int32)
    view = base.view(torch.uint8)
    swapped = base.clone()
    swapped[[3, 17]] = swapped[[17, 3]]
    whole = child._semip_digest(view)
    assert child._semip_digest(swapped.view(torch.uint8)) != whole
    assert child._semip_digest(swapped.view(torch.uint8))[0] == whole[0]
    monkeypatch.setitem(child._semip_digest.__globals__, "_DIGEST_CHUNK", 7)
    assert child._semip_digest(view) == whole
    odd = torch.arange(13, dtype=torch.uint8)
    monkeypatch.setitem(child._semip_digest.__globals__, "_DIGEST_CHUNK", 1 << 22)
    expected = child._semip_digest(odd)
    monkeypatch.setitem(child._semip_digest.__globals__, "_DIGEST_CHUNK", 5)
    assert child._semip_digest(odd) == expected
