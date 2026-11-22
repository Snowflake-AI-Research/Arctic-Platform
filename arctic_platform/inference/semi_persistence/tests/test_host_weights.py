"""Large pinned host parameters leave the CRIU image (``host_weights``).

DeepSeek-V4.1-Flash's Engram tables are pinned CPU parameters, 23.6 GiB per
rank, that were in the weight set and in the CRIU image both.  ``host_weights``
moves them into mappings of its own at load, empties those at the dump, and the
restore reads them back from the shards.

Here registration is a fake that records which addresses are registered (and
``is_pinned`` reads it), a UVA view is the CPU tensor itself, and
/proc/self/maps is a stand-in.  The mappings, their PROT_NONE and the shard I/O
are real.  ``vllm_child``'s attach/save/load/refill are lifted out of its source,
as ``test_unregistered_weights`` lifts its staging functions.
"""
from __future__ import annotations

import ast
import ctypes
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHILD = os.path.join(_PKG, "vllm_child.py")
_WORKER = os.path.join(_PKG, "_semip_worker.py")
_ENGINE = os.path.join(os.path.dirname(_PKG), "server", "semip_engine.py")
_RECEIVER = os.path.join(os.path.dirname(_PKG), "server", "weight_sync", "receiver.py")
_PUBLISH = os.path.join(_PKG, "scripts", "semip_publish.py")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import host_weights  # noqa: E402

MIN = 4096
_FP8 = getattr(torch, "float8_e4m3fn", torch.uint8)
_WEIGHT = "main:p:layers.1.engram.embed_tokens.weight"
_SCALE = "main:p:layers.1.engram.embed_tokens.weight_scale_inv"
_CHILD_NAMES = (
    "_semip_weight_pool_ranges", "_semip_unregistered_storages", "_semip_param_tensors",
    "_semip_lora_manager", "_semip_lora_views", "_semip_weight_like", "_semip_layout",
    "_semip_attach", "_semip_stage", "_semip_detach", "_semip_free_double_buffer",
    "_semip_segments", "_semip_read_into", "_semip_save_weights", "_semip_load_weights",
    "_semip_config_value", "_semip_tp_size", "_semip_host_census", "_semip_unpin_host",
    "_semip_repin_host", "_semip_release_host_weights", "_semip_refill_host_weights",
    "_semip_lora_wrappers", "_semip_canonical_name", "_semip_canonical_names")
_CHILD_CONSTANTS = ("_WEIGHT_LIKE_PATH", "_WEIGHTS_SHARD_BYTES", "_WEIGHTS_IO_WORKERS")


class _Registry:
    """What the driver knows: registered ranges, torch's pinned blocks, file mappings."""

    def __init__(self):
        self.registered = {}
        self.pinned_ptrs = set()
        self.files = {}
        self.device_offset = 0

    def register(self, ptr, nbytes):
        if ptr in self.registered:
            return 712          # cudaErrorHostMemoryAlreadyRegistered
        self.registered[ptr] = nbytes
        return 0

    def unregister(self, ptr):
        if ptr not in self.registered:
            raise RuntimeError(f"cudaHostUnregister(0x{ptr:x}) failed with cudaError=713")
        del self.registered[ptr]

    def device_pointer(self, ptr):
        return ptr + self.device_offset if ptr in self.registered else None

    def pinned(self, tensor):
        ptr = tensor.untyped_storage().data_ptr()
        return ptr in self.pinned_ptrs or any(
            lo <= ptr < lo + n for lo, n in self.registered.items())

    def maps(self):
        return ([(ptr, ptr + 1, path) for ptr, path in self.files.items()]
                + [(0, 2**64, "")])


@pytest.fixture
def hw(monkeypatch):
    reg = _Registry()
    drops = []
    real_drop = host_weights._drop_pages
    monkeypatch.setattr(host_weights, "_register", reg.register)
    monkeypatch.setattr(host_weights, "_unregister", reg.unregister)
    monkeypatch.setattr(host_weights, "_device_pointer", reg.device_pointer)
    monkeypatch.setattr(host_weights, "_pinned", reg.pinned)
    monkeypatch.setattr(host_weights, "_uva_view", lambda tensor: tensor)
    monkeypatch.setattr(host_weights, "_mappings", reg.maps)
    monkeypatch.setattr(host_weights, "_host_empty_cache", lambda: None)
    monkeypatch.setattr(host_weights, "_drop_pages",
                        lambda mapping: (drops.append(mapping), real_drop(mapping)))
    return types.SimpleNamespace(reg=reg, drops=drops)


def _pinned_param(reg, rows, cols, dtype=torch.uint8):
    t = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    if dtype is not torch.uint8:
        t = t.view(dtype)
    reg.pinned_ptrs.add(t.untyped_storage().data_ptr())
    return torch.nn.Parameter(t, requires_grad=False)


def _engram_worker(reg):
    """A model with an Engram-shaped layer between two small (device) parameters."""
    model = torch.nn.Module()
    model.embed = torch.nn.Module()
    model.embed.weight = torch.nn.Parameter(torch.rand(10, 25), requires_grad=False)
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    engram = torch.nn.Module()
    engram.embed_tokens = torch.nn.Module()
    engram.embed_tokens.weight = _pinned_param(reg, 300, 64, _FP8)
    engram.embed_tokens.weight_scale_inv = _pinned_param(reg, 300, 16)
    engram.embed_tokens._views = None
    model.layers[1].engram = engram
    model.norm = torch.nn.Module()
    model.norm.weight = torch.nn.Parameter(torch.rand(7, 25), requires_grad=False)
    return types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model),
                                 rank=0, device="cpu",
                                 vllm_config={"tensor_parallel_size": 2})


def _rehomed(reg):
    worker = _engram_worker(reg)
    worker._semip_host = host_weights.rehome(worker, host_weights.HostWeights(),
                                             min_bytes=MIN)
    return worker


def _bytes(tensor):
    return tensor.detach().reshape(-1).view(torch.uint8).clone()


def _faults(ptr):
    """Whether reading ``ptr`` kills a forked copy of this process."""
    pid = os.fork()
    if pid == 0:
        try:
            ctypes.string_at(ptr, 1)
        finally:
            os._exit(0)
    _, status = os.waitpid(pid, 0)
    return os.WIFSIGNALED(status) and os.WTERMSIG(status) in (signal.SIGSEGV, signal.SIGBUS)


# --------------------------------------------------------------------------
# Which parameters move
# --------------------------------------------------------------------------

def test_large_pinned_whole_storage_parameters_qualify_and_the_rest_say_why(hw):
    worker = _engram_worker(hw.reg)
    model = worker.model_runner.model
    model.small = _pinned_param(hw.reg, 4, 16)
    model.pageable = torch.nn.Parameter(torch.zeros(64, 128, dtype=torch.uint8),
                                        requires_grad=False)
    whole = _pinned_param(hw.reg, 64, 128)
    model.view = torch.nn.Parameter(whole.data[1:], requires_grad=False)
    model.shm = _pinned_param(hw.reg, 64, 128)
    hw.reg.files[model.shm.data_ptr()] = "/dev/shm/vllm_engram_0"
    model.ple = torch.nn.Module()
    model.ple.weight = _pinned_param(hw.reg, 64, 128)
    model.ple._uva_weight = model.ple.weight.data.view(torch.int32)
    host = host_weights.rehome(worker, host_weights.HostWeights(), min_bytes=MIN)

    assert sorted(host) == sorted([_WEIGHT, _SCALE])
    assert dict(host.skipped) == {
        "main:p:pageable": "not pinned",
        "main:p:view": "a view of a larger storage",
        "main:p:shm": "file-backed (/dev/shm/vllm_engram_0)",
        "main:p:ple.weight": "aliased by main:ple._uva_weight",
    }
    assert "uva_identity=ok" in host_weights.describe(host)


def test_a_driver_memfd_is_not_a_named_file(hw):
    worker = _engram_worker(hw.reg)
    weight = worker.model_runner.model.layers[1].engram.embed_tokens.weight
    hw.reg.files[weight.data_ptr()] = "/memfd:cuda-pinned (deleted)"
    host = host_weights.rehome(worker, host_weights.HostWeights(), min_bytes=MIN)
    assert _WEIGHT in host


def test_a_uva_view_cached_by_the_module_keeps_its_parameter(hw):
    worker = _engram_worker(hw.reg)
    emb = worker.model_runner.model.layers[1].engram.embed_tokens
    emb._views = (emb.weight.data.view(torch.uint8), emb.weight_scale_inv.data)
    host = host_weights.rehome(worker, host_weights.HostWeights(), min_bytes=MIN)
    assert not host
    assert dict(host.skipped) == {
        _WEIGHT: "aliased by main:layers.1.engram.embed_tokens._views.0",
        _SCALE: "aliased by main:layers.1.engram.embed_tokens._views.1"}


def test_a_model_on_the_device_has_nothing_to_move(hw):
    model = torch.nn.Module()
    model.embed = torch.nn.Module()
    model.embed.weight = torch.nn.Parameter(torch.empty(4096, 4096, device="meta"),
                                            requires_grad=False)
    worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model))
    host = host_weights.rehome(worker, host_weights.HostWeights(), min_bytes=MIN)
    assert not host and not host.skipped
    assert host_weights.describe(host).startswith("rehomed 0 param(s)")


def test_old_pinned_blocks_that_stay_referenced_are_reported(hw):
    host = host_weights.rehome(_engram_worker(hw.reg), host_weights.HostWeights(),
                               min_bytes=MIN)
    moved = host.nbytes()
    host.torch_reserved = (10 * moved, 9 * moved)
    assert host_weights.unfreed_bytes(host) == 0
    assert "WARNING" not in host_weights.describe(host)
    host.torch_reserved = (10 * moved, 10 * moved)
    assert host_weights.unfreed_bytes(host) == moved
    assert "still referenced" in host_weights.describe(host)
    host.torch_reserved = (None, None)
    assert host_weights.unfreed_bytes(host) is None
    assert "WARNING" not in host_weights.describe(host)


def test_without_proc_maps_nothing_moves(hw, monkeypatch):
    monkeypatch.setattr(host_weights, "_mappings", lambda: None)
    host = host_weights.rehome(_engram_worker(hw.reg), host_weights.HostWeights(),
                               min_bytes=MIN)
    assert not host
    assert {why for _, why in host.skipped} == {"/proc/self/maps unreadable"}


# --------------------------------------------------------------------------
# Rehome, release, refill
# --------------------------------------------------------------------------

def test_rehome_moves_the_bytes_into_a_registered_mapping_at_its_own_address(hw):
    worker = _engram_worker(hw.reg)
    emb = worker.model_runner.model.layers[1].engram.embed_tokens
    before = {n: (_bytes(p), p.dtype, p.shape, p.data_ptr())
              for n, p in [("w", emb.weight), ("s", emb.weight_scale_inv)]}
    host = host_weights.rehome(worker, host_weights.HostWeights(), min_bytes=MIN)

    for key, param, name in [("w", emb.weight, _WEIGHT), ("s", emb.weight_scale_inv, _SCALE)]:
        data, dtype, shape, old_ptr = before[key]
        entry = host[name]
        assert param.data_ptr() == entry.ptr != old_ptr
        assert (param.dtype, param.shape) == (dtype, shape)
        assert torch.equal(_bytes(param), data)
        assert hw.reg.registered[entry.ptr] == entry.nbytes == data.numel()
        assert entry.expected == host_weights.digest(param.data)
        assert param._semip_verify_unchanged == entry.verify_unchanged
    assert not host.down()


def test_a_device_pointer_other_than_the_host_address_leaves_the_parameter(hw):
    hw.reg.device_offset = 1 << 20
    worker = _engram_worker(hw.reg)
    weight = worker.model_runner.model.layers[1].engram.embed_tokens.weight
    ptr = weight.data_ptr()
    host = host_weights.rehome(worker, host_weights.HostWeights(), min_bytes=MIN)
    assert not host and not hw.reg.registered
    assert weight.data_ptr() == ptr
    assert all("device pointer" in why for _, why in host.skipped)


def test_release_unregisters_drops_the_pages_and_makes_the_mapping_inaccessible(hw):
    worker = _rehomed(hw.reg)
    host = worker._semip_host
    ptrs = [e.ptr for e in host.values()]
    assert not any(_faults(p) for p in ptrs)
    out = host.release()

    assert sorted(n for n, _ in out["released"]) == sorted([_WEIGHT, _SCALE])
    assert out["kept"] == [] and out["down"] and host.down()
    assert not hw.reg.registered
    assert len(hw.drops) == 2
    assert all(e.dropped and not e.registered for e in host.values())
    assert all(_faults(p) for p in ptrs)
    if os.path.exists("/proc/self/smaps"):
        assert _rss_kb(ptrs[0]) == 0
    assert host.release()["released"] == []


def _rss_kb(ptr):
    with open("/proc/self/smaps") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        lo = line.split("-", 1)[0]
        if re.fullmatch(r"[0-9a-f]+", lo) and int(lo, 16) == ptr:
            return next(int(x.split()[1]) for x in lines[i + 1:] if x.startswith("Rss:"))
    raise AssertionError("mapping not in smaps")


def test_a_table_that_changed_since_load_stays_in_the_image(hw):
    worker = _rehomed(hw.reg)
    host = worker._semip_host
    emb = worker.model_runner.model.layers[1].engram.embed_tokens
    weight = emb.weight
    weight.data.view(torch.uint8)[0, 0] ^= 0xFF
    changed, scale = _bytes(weight), _bytes(emb.weight_scale_inv)
    out = host.release()

    assert [n for n, _ in out["kept"]] == [_WEIGHT]
    assert [n for n, _ in out["released"]] == [_SCALE]
    assert len(hw.drops) == 1
    entry = host[_WEIGHT]
    assert not entry.dropped and not entry.registered
    assert not _faults(entry.ptr)
    host[_SCALE].open()
    host[_SCALE].view()[:] = scale.numpy().tobytes()
    done = host.finish_refill([_SCALE])
    assert done["registered"] == 2 and not done["down"]
    assert torch.equal(_bytes(weight), changed)


def test_unpin_and_repin_cover_the_mappings_but_never_register_a_released_one(hw):
    host = _rehomed(hw.reg)._semip_host
    assert host.unpin()["down"] and not hw.reg.registered
    assert not host.repin()["down"] and len(hw.reg.registered) == 2
    host.release()
    assert host.repin()["down"] and not hw.reg.registered


def test_a_released_table_that_is_not_refilled_cannot_be_registered(hw):
    host = _rehomed(hw.reg)._semip_host
    host.release()
    with pytest.raises(RuntimeError, match="not refilled"):
        host.finish_refill([])
    assert not hw.reg.registered


def test_weight_sync_into_a_rehomed_table_fails_loudly(hw):
    worker = _rehomed(hw.reg)
    weight = worker.model_runner.model.layers[1].engram.embed_tokens.weight
    refuse = _lift(_RECEIVER, ["_refuse_synced_host_resident_params"])
    refuse(worker.model_runner.model)
    weight.data.view(torch.uint8)[3, 3] ^= 1
    with pytest.raises(RuntimeError, match="weight sync wrote .*embed_tokens.weight"):
        refuse(worker.model_runner.model)
    weight.data = weight.data.clone()
    with pytest.raises(RuntimeError, match="replaced the storage"):
        weight._semip_verify_unchanged()


# --------------------------------------------------------------------------
# The weight set: unchanged bytes, refilled in place
# --------------------------------------------------------------------------

class _Torch:
    cuda = types.SimpleNamespace(synchronize=lambda *a: None, empty_cache=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)


def _lift(path, names, ns=None):
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in funcs} == set(names)
    ns = dict(ns or {})
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), ns)
    return ns[names[0]] if len(names) == 1 else types.SimpleNamespace(**ns)


@pytest.fixture
def child(monkeypatch):
    with open(_CHILD) as handle:
        tree = ast.parse(handle.read(), filename=_CHILD)
    ns = {"torch": _Torch(), "ThreadPoolExecutor": ThreadPoolExecutor, "time": time,
          "re": re, "os": os, "json": json, "shutil": shutil,
          "_unpin_buffer": lambda buf: None}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in _CHILD_CONSTANTS
                                                for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), _CHILD, "exec"), ns)
    cumem = types.ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = type("CuMemAllocator", (), {"instance": None})
    for name, mod in {"vllm": types.ModuleType("vllm"),
                      "vllm.device_allocator": types.ModuleType("vllm.device_allocator"),
                      "vllm.device_allocator.cumem": cumem}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return _lift(_CHILD, _CHILD_NAMES, ns)


def _save(child, worker, weights_dir, shard_bytes=4096):
    child._semip_attach(worker)
    child._semip_stage(worker)
    child._semip_save_weights(worker, str(weights_dir), shard_bytes=shard_bytes)
    return weights_dir / "rank0"


def _files(rank_dir):
    return {p.name: p.read_bytes() for p in sorted(rank_dir.iterdir())}


def test_the_weight_set_is_byte_identical_with_and_without_thinning(hw, child, tmp_path):
    worker = _rehomed(hw.reg)
    thin = _files(_save(child, worker, tmp_path / "thin"))
    host = worker._semip_host
    worker._semip_host = None
    flat = _files(_save(child, worker, tmp_path / "flat"))
    worker._semip_host = host

    assert thin == flat
    meta = json.loads(thin["weights_meta.json"])
    by_name = {row[0]: row for row in meta["layout"]}
    assert by_name[_WEIGHT][2] == 300 * 64
    assert sum(row[2] for row in meta["layout"]) == meta["total_bytes"]
    assert len([s for s in meta["shards"]
                if s["offset"] < by_name[_WEIGHT][1] + by_name[_WEIGHT][2]
                and by_name[_WEIGHT][1] < s["offset"] + s["nbytes"]]) >= 5
    spec = importlib.util.spec_from_file_location("semip_publish_t", _PUBLISH)
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)

    def rows(rank_dir):
        import hashlib
        return [{"path": f"{pub.WEIGHT_DIR}/rank0/{p.name}",
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in rank_dir.iterdir()]
    assert pub.weights_hash(rows(tmp_path / "thin" / "rank0")) == \
        pub.weights_hash(rows(tmp_path / "flat" / "rank0"))


def test_host_resident_tables_take_no_room_in_the_host_copy(hw, child):
    worker = _rehomed(hw.reg)
    total, n_entries, _, _ = child._semip_attach(worker)
    model = worker.model_runner.model
    assert n_entries == 4
    assert set(worker._semip_index) == {"main:p:embed.weight", "main:p:norm.weight"}
    assert total == worker._semip_buf.numel() == (
        model.embed.weight.nbytes + model.norm.weight.nbytes)
    assert [r[0] for r in worker._semip_rows] == [
        "main:p:embed.weight", _WEIGHT, _SCALE, "main:p:norm.weight"]


def _released_and_reattached(hw, child, tmp_path):
    """Dump (save + release), then the restore's attach + load_weights."""
    worker = _rehomed(hw.reg)
    model = worker.model_runner.model
    expected = {n: _bytes(p) for n, p in model.named_parameters()}
    rank_dir = _save(child, worker, tmp_path)
    out = child._semip_release_host_weights(worker)
    assert len(out["released"]) == 2 and out["down"]
    child._semip_detach(worker)
    child._semip_attach(worker)
    child._semip_load_weights(worker, str(tmp_path))
    return worker, expected, rank_dir


def test_refill_reads_the_shards_into_the_mappings_and_registers_them(hw, child, tmp_path):
    worker, expected, _ = _released_and_reattached(hw, child, tmp_path)
    model = worker.model_runner.model
    buf = worker._semip_buf
    assert torch.equal(buf[:model.embed.weight.nbytes], expected["embed.weight"])
    assert torch.equal(buf[model.embed.weight.nbytes:], expected["norm.weight"])

    out = child._semip_refill_host_weights(worker, str(tmp_path), 3)
    assert (out["refilled"], out["registered"], out["down"]) == (2, 2, False)
    assert out["refilled_bytes"] == 300 * 64 + 300 * 16
    emb = model.layers[1].engram.embed_tokens
    assert torch.equal(_bytes(emb.weight), expected["layers.1.engram.embed_tokens.weight"])
    assert torch.equal(_bytes(emb.weight_scale_inv),
                       expected["layers.1.engram.embed_tokens.weight_scale_inv"])
    assert len(hw.reg.registered) == 2
    assert child._semip_refill_host_weights(worker, str(tmp_path))["refilled"] == 0


def test_a_short_shard_fails_the_refill(hw, child, tmp_path):
    worker, _, rank_dir = _released_and_reattached(hw, child, tmp_path)
    shard = rank_dir / "shard_0002.bin"
    shard.write_bytes(shard.read_bytes()[:100])
    with pytest.raises(RuntimeError, match="shard_0002.bin read 100 != 4096"):
        child._semip_refill_host_weights(worker, str(tmp_path))
    assert not hw.reg.registered


def test_a_corrupt_shard_fails_the_checksum(hw, child, tmp_path):
    worker, _, rank_dir = _released_and_reattached(hw, child, tmp_path)
    shard = rank_dir / "shard_0002.bin"
    data = bytearray(shard.read_bytes())
    data[7] ^= 0x40
    shard.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="embed_tokens.weight: checksum after refill"):
        child._semip_refill_host_weights(worker, str(tmp_path))


def test_a_weight_set_without_the_table_fails_the_refill(hw, child, tmp_path):
    worker, _, rank_dir = _released_and_reattached(hw, child, tmp_path)
    meta_path = rank_dir / "weights_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["layout"] = [row for row in meta["layout"] if row[0] != _SCALE]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="weight_scale_inv: 4800 B in the image but absent"):
        child._semip_refill_host_weights(worker, str(tmp_path))


def test_a_missing_manifest_fails_the_refill(hw, child, tmp_path):
    worker, _, rank_dir = _released_and_reattached(hw, child, tmp_path)
    (rank_dir / "weights_meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        child._semip_refill_host_weights(worker, str(tmp_path))


def test_without_host_resident_parameters_the_commands_do_nothing(child):
    worker = types.SimpleNamespace(rank=0)
    assert child._semip_release_host_weights(worker)["down"] is False
    assert child._semip_refill_host_weights(worker, "/nonexistent")["refilled"] == 0
    assert child._semip_unpin_host(worker) == child._semip_repin_host(worker) == {
        "n": 0, "bytes": 0, "down": False}


# --------------------------------------------------------------------------
# The switch, the worker hook and the wiring
# --------------------------------------------------------------------------

def _worker_class(monkeypatch):
    loaded = []

    class Worker:
        def load_model(self, *args, **kwargs):
            loaded.append((args, kwargs))
            return "loaded"

    gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = Worker
    for name, mod in {"vllm": types.ModuleType("vllm"), "vllm.v1": types.ModuleType("vllm.v1"),
                      "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
                      "vllm.v1.worker.gpu_worker": gpu_worker}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("_semip_worker_t", _WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SemipGPUWorker, loaded


@pytest.mark.parametrize("flag,moved", [(None, 2), ("1", 2), ("0", 0)])
def test_load_model_rehomes_unless_switched_off(hw, monkeypatch, capsys, flag, moved):
    if flag is None:
        monkeypatch.delenv(host_weights.FLAG_ENV, raising=False)
    else:
        monkeypatch.setenv(host_weights.FLAG_ENV, flag)
    monkeypatch.setattr(host_weights, "MIN_BYTES", MIN)
    monkeypatch.setattr(host_weights.rehome, "__defaults__", (MIN,))
    cls, loaded = _worker_class(monkeypatch)
    worker = cls.__new__(cls)
    fake = _engram_worker(hw.reg)
    worker.model_runner, worker.rank = fake.model_runner, 3
    weight = fake.model_runner.model.layers[1].engram.embed_tokens.weight
    ptr = weight.data_ptr()

    assert worker.load_model() == "loaded" and len(loaded) == 1
    out = capsys.readouterr().out
    if moved:
        assert len(worker._semip_host) == moved and weight.data_ptr() != ptr
        assert "[semip-host] rank=3 rehomed 2 param(s)" in out
    else:
        assert getattr(worker, "_semip_host", None) is None and weight.data_ptr() == ptr
        assert "[semip-host] rank=3 off (SEMIP_THIN_PINNED_PARAMS=0)" in out


def _source(path, name):
    tree = ast.parse(open(path).read())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(open(path).read(), node)


@pytest.mark.parametrize("fn", ["_dump", "_dump_multinode"])
def test_the_dump_releases_after_save_weights_and_before_the_checkpoint(fn):
    src = _source(_ENGINE, fn)
    order = [src.index(s) for s in ("inst.save_weights(", "inst.release_host_weights()",
                                    "inst.detach()", "inst.cuda_checkpoint(")]
    assert order == sorted(order)


@pytest.mark.parametrize("fn", ["_restore", "_restore_multinode"])
def test_the_restore_refills_after_load_weights_and_before_the_kv_cache(fn):
    src = _source(_ENGINE, fn)
    order = [src.index(s) for s in (".load_weights(", "inst.refill_host_weights(",
                                    "inst.wake_up_kv_cache()")]
    assert order == sorted(order)


def test_nothing_runs_the_model_while_a_table_is_unregistered():
    tree = ast.parse(open(_CHILD).read())
    refused = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", "") == "_HOST_DOWN_REFUSED_CMDS")
    assert set(ast.literal_eval(refused.args[0])) == {
        "wake_up_kv_cache", "rebind_graphs", "recapture_graphs", "serve_core", "resume"}
    handle = _source(_CHILD, "_handle_command")
    assert handle.index("if _host_down and cmd in _HOST_DOWN_REFUSED_CMDS") < \
        handle.index('if cmd == "init"')
    assert "if _host_down and not _paused" in _source(_CHILD, "_submit_generate")
