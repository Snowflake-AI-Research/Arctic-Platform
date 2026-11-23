"""A LoRA config shares the full config's weight set.

With LoRA enabled vLLM wraps every layer it can adapt in a
``BaseLayerWithLoRA`` that holds the original as ``base_layer``, so the
parameter names gain a segment (``main:p:language_model.model.embed_tokens.
base_layer.weight``) while the shards stay byte-identical.  The names went into
``weights_meta.json`` as they were, and ``weights_hash`` hashes that file with
the shards, so the LoRA configs forked the hash: Qwen3.8 full 64e76626e58c
against LoRA 970d92031a24, GLM-5.3-Flash long f858d7580d72 against LoRA
dda747885ca6.

``vllm_child``'s attach/save/load are lifted out of its source, as
``test_unregistered_weights`` lifts its staging functions, and ``vllm.lora``
is a stand-in class.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHILD = os.path.join(_PKG, "vllm_child.py")
_PUBLISH = os.path.join(_PKG, "scripts", "semip_publish.py")
_NAMES = (
    "_semip_weight_pool_ranges", "_semip_unregistered_storages", "_semip_param_tensors",
    "_semip_lora_manager", "_semip_lora_views", "_semip_weight_like", "_semip_layout",
    "_semip_attach", "_semip_stage", "_semip_segments", "_semip_read_into",
    "_semip_save_weights", "_semip_load_weights", "_semip_config_value", "_semip_tp_size",
    "_semip_lora_wrappers", "_semip_canonical_name", "_semip_canonical_names")
_CONSTANTS = ("_WEIGHT_LIKE_PATH", "_WEIGHTS_SHARD_BYTES", "_WEIGHTS_IO_WORKERS")


class BaseLayerWithLoRA(torch.nn.Module):
    """Stands in for ``vllm.lora.layers.BaseLayerWithLoRA``."""

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.lora_a_stacked = (torch.zeros(2, 4),)


class _Torch:
    cuda = types.SimpleNamespace(synchronize=lambda *a: None, empty_cache=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def child(monkeypatch):
    with open(_CHILD) as handle:
        tree = ast.parse(handle.read(), filename=_CHILD)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    assert {f.name for f in funcs} == set(_NAMES)
    ns = {"torch": _Torch(), "ThreadPoolExecutor": ThreadPoolExecutor, "time": time,
          "re": re, "os": os, "json": json, "shutil": shutil}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in _CONSTANTS
                                                for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), _CHILD, "exec"), ns)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), _CHILD, "exec"), ns)
    cumem = types.ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = type("CuMemAllocator", (), {"instance": None})
    layers = types.ModuleType("vllm.lora.layers")
    layers.BaseLayerWithLoRA = BaseLayerWithLoRA
    for name, mod in {"vllm": types.ModuleType("vllm"),
                      "vllm.device_allocator": types.ModuleType("vllm.device_allocator"),
                      "vllm.device_allocator.cumem": cumem,
                      "vllm.lora": types.ModuleType("vllm.lora"),
                      "vllm.lora.layers": layers}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return types.SimpleNamespace(**ns)


def _param(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.rand(*shape, generator=g), requires_grad=False)


def _layer(seed, bias=False):
    layer = torch.nn.Module()
    layer.weight = _param(4, 6, seed=seed)
    if bias:
        layer.bias = _param(4, seed=seed + 100)
    return layer


def _worker(lora, tp=2):
    """A small Qwen-shaped model; with ``lora`` its adaptable layers are wrapped."""
    def wrap(layer):
        return BaseLayerWithLoRA(layer) if lora else layer

    lm = torch.nn.Module()
    lm.model = torch.nn.Module()
    lm.model.embed_tokens = wrap(_layer(1))
    lm.model.layers = torch.nn.ModuleList()
    for i in range(2):
        block = torch.nn.Module()
        block.qkv_proj = wrap(_layer(10 + i, bias=True))
        block.norm = _layer(20 + i)
        lm.model.layers.append(block)
    root = torch.nn.Module()
    root.language_model = lm
    manager = (types.SimpleNamespace(punica_wrapper_mapping={}) if lora else None)
    runner = types.SimpleNamespace(
        model=root,
        lora_manager=types.SimpleNamespace(_adapter_manager=manager) if lora else None)
    return types.SimpleNamespace(model_runner=runner, rank=0, device="cpu",
                                 vllm_config={"tensor_parallel_size": tp})


def _save(child, worker, weights_dir, shard_bytes=256):
    child._semip_attach(worker)
    child._semip_stage(worker)
    child._semip_save_weights(worker, str(weights_dir), shard_bytes=shard_bytes)
    return weights_dir / "rank0"


def _files(rank_dir):
    return {p.name: p.read_bytes() for p in sorted(rank_dir.iterdir())}


def _weights_hash(rank_dir):
    spec = importlib.util.spec_from_file_location("semip_publish_t", _PUBLISH)
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    return pub.weights_hash([
        {"path": f"{pub.WEIGHT_DIR}/rank0/{p.name}",
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in rank_dir.iterdir()])


def test_the_lora_engine_writes_the_full_engines_weight_set(child, tmp_path):
    lora = _worker(lora=True)
    names = [n for n, _ in lora.model_runner.model.named_parameters()]
    assert "language_model.model.embed_tokens.base_layer.weight" in names
    full_dir = _save(child, _worker(lora=False), tmp_path / "full")
    lora_dir = _save(child, lora, tmp_path / "lora")

    assert _files(lora_dir) == _files(full_dir)
    layout = json.loads((lora_dir / "weights_meta.json").read_text())["layout"]
    assert not any("base_layer" in row[0] for row in layout)
    assert "main:p:language_model.model.layers.1.qkv_proj.bias" in [r[0] for r in layout]
    assert _weights_hash(lora_dir) == _weights_hash(full_dir)


def test_a_full_engines_manifest_is_the_format_it_always_was(child, tmp_path):
    worker = _worker(lora=False)
    rank_dir = _save(child, worker, tmp_path, shard_bytes=300)
    params = list(worker.model_runner.model.named_parameters())
    layout, off = [], 0
    for n, p in params:
        layout.append([f"main:p:{n}", off, p.nbytes, str(p.dtype), list(p.shape)])
        off += p.nbytes
    shards = [{"name": f"shard_{i:04d}.bin", "offset": lo, "nbytes": min(300, off - lo)}
              for i, lo in enumerate(range(0, off, 300))]
    expected = {"total_bytes": off, "n_params": len(params), "shard_bytes": 300,
                "shards": shards, "layout": layout}
    assert (rank_dir / "weights_meta.json").read_text() == json.dumps(expected)


def test_two_names_that_canonicalize_alike_fail_the_save_before_it_writes(child, tmp_path):
    worker = _worker(lora=True)
    wrapper = worker.model_runner.model.language_model.model.embed_tokens
    wrapper.weight = _param(4, 6, seed=99)
    rank_dir = tmp_path / "rank0"
    rank_dir.mkdir()
    (rank_dir / "shard_0000.bin").write_bytes(b"published")
    child._semip_attach(worker)
    with pytest.raises(RuntimeError, match=r"embed_tokens\.(base_layer\.)?weight and .* would "
                                           r"both be main:p:language_model\.model\.embed_tokens\.weight"):
        child._semip_save_weights(worker, str(tmp_path))
    assert (rank_dir / "shard_0000.bin").read_bytes() == b"published"


def test_only_a_wrappers_own_base_layer_is_dropped(child):
    wrappers = {"main": {"a.lora", "a.lora.base_layer"}, "drafter": {"d"}}
    canon = child._semip_canonical_name
    assert canon("main:p:a.lora.base_layer.base_layer.weight", wrappers) == "main:p:a.lora.weight"
    assert canon("main:p:a.plain.base_layer.weight", wrappers) == "main:p:a.plain.base_layer.weight"
    assert canon("main:s:a.lora.base_layer.quant_method.w13", wrappers) == "main:s:a.lora.quant_method.w13"
    assert canon("drafter:p:d.base_layer.weight", wrappers) == "drafter:p:d.weight"
    assert canon("drafter:p:a.lora.base_layer.weight", wrappers) == "drafter:p:a.lora.base_layer.weight"
    assert canon("main:p:a.lora.base_layer.weight", {}) == "main:p:a.lora.base_layer.weight"


def test_a_module_called_base_layer_that_is_no_wrapper_keeps_its_name(child, tmp_path):
    worker = _worker(lora=True)
    worker.model_runner.model.language_model.model.layers[0].norm.base_layer = _layer(7)
    layout = json.loads((_save(child, worker, tmp_path) / "weights_meta.json").read_text())
    assert "main:p:language_model.model.layers.0.norm.base_layer.weight" in [r[0] for r in layout["layout"]]


def _load(child, worker, weights_dir):
    child._semip_attach(worker)
    child._semip_load_weights(worker, str(weights_dir))
    return worker._semip_buf.clone()


@pytest.mark.parametrize("writer,reader", [(False, True), (True, False), (True, True)])
def test_either_engine_loads_the_others_weight_set(child, tmp_path, writer, reader):
    source = _worker(lora=writer)
    _save(child, source, tmp_path)
    expected = source._semip_buf.clone()
    assert torch.equal(_load(child, _worker(lora=reader), tmp_path), expected)


def _rewrite_names(rank_dir, fn):
    path = rank_dir / "weights_meta.json"
    meta = json.loads(path.read_text())
    for row in meta["layout"]:
        row[0] = fn(row[0])
    path.write_text(json.dumps(meta))


def test_a_lora_engine_loads_a_weight_set_written_with_its_wrapped_names(child, tmp_path):
    rank_dir = _save(child, _worker(lora=True), tmp_path)
    wrapped = {"embed_tokens", "qkv_proj"}
    _rewrite_names(rank_dir, lambda n: re.sub(
        r"\.(%s)\.(weight|bias)$" % "|".join(wrapped), r".\1.base_layer.\2", n))
    layout = json.loads((rank_dir / "weights_meta.json").read_text())["layout"]
    assert sum("base_layer" in r[0] for r in layout) == 5
    _load(child, _worker(lora=True), tmp_path)
    with pytest.raises(RuntimeError, match="layout mismatch"):
        _load(child, _worker(lora=False), tmp_path)


def test_the_load_check_still_compares_offsets_sizes_and_dtypes(child, tmp_path):
    rank_dir = _save(child, _worker(lora=False), tmp_path)
    path = rank_dir / "weights_meta.json"
    good = path.read_text()
    for field, value in [(1, 1), (2, 1), (3, "torch.bfloat16")]:
        meta = json.loads(good)
        meta["layout"][1][field] = (meta["layout"][1][field] + value
                                    if isinstance(value, int) else value)
        path.write_text(json.dumps(meta))
        with pytest.raises(RuntimeError, match="layout mismatch"):
            _load(child, _worker(lora=True), tmp_path)
