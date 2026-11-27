# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the expert-parallel checkpoint format.

The expert files are read back on resume from storage that someone else may control, so they must load without a full
unpickle. These tests drive the real saver and loader on CPU: anything ``save_expert_shards`` writes, and any plain
``{name: tensor}`` dict ``torch.save`` writes, must load, and a file that names an arbitrary callable must be refused
without running it. The DeepSpeed engine is stubbed for the save and load entry points.
"""

from __future__ import annotations

import contextlib
import os
import struct
import tempfile
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.common.utils import expert_parallel_checkpoint as epc
from arctic_platform.testing_utils import require_deepspeed

EP_RANK = 2
TAG = "global_step9"
# Archive framing (zip headers, pickle, record alignment) of a shard beyond its tensor bytes.
_ARCHIVE_OVERHEAD = 64 * 1024
_SMALL_SHAPES = {"layers.0.mlp.experts.gate_up_proj": (2, 8, 4), "layers.0.mlp.experts.down_proj": (2, 4, 4)}
# 256 MiB of bf16 experts in eight 32 MiB tensors.
_GPU_SHAPES = {f"layers.{i}.mlp.experts.down_proj": (8, 1024, 2048) for i in range(8)}


def _expert_params(dtype, *, flat_views=False, seed=0, shapes=_SMALL_SHAPES, tail_bytes=1024 * 1024, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    if flat_views:
        # ZeRO points bf16 param ``.data`` at views of one flat buffer per optimizer group. The buffer also holds
        # ``tail_bytes`` that are not expert parameters, so a shard that writes the backing storage grows by them.
        numel = sum(torch.Size(s).numel() for s in shapes.values())
        tail = tail_bytes // torch.empty(0, dtype=dtype).element_size()
        flat = torch.randn(numel + tail, generator=gen).to(dtype).to(device)
        tensors, offset = {}, 0
        for name, shape in shapes.items():
            n = torch.Size(shape).numel()
            tensors[name] = flat[offset : offset + n].view(shape)
            offset += n
    else:
        tensors = {name: torch.randn(shape, generator=gen).to(dtype).to(device) for name, shape in shapes.items()}
    items = []
    for name, t in tensors.items():
        p = nn.Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        p.data = t
        p.allreduce = False
        p.group_name = "ep_size_4"
        items.append((name, p))
    return items


def _with_experts(monkeypatch, items):
    monkeypatch.setattr(epc, "expert_params_and_ranks", lambda engine: (items, EP_RANK, 0))


def _save(monkeypatch, items, tag_dir):
    _with_experts(monkeypatch, items)
    epc.save_expert_shards(SimpleNamespace(), str(tag_dir))


def _load(monkeypatch, items, tag_dir):
    _with_experts(monkeypatch, items)
    epc.load_expert_shards(SimpleNamespace(), str(tag_dir))


def _shard_path(tag_dir: Path) -> Path:
    return tag_dir / epc.expert_shard_name(EP_RANK)


_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
if hasattr(torch, "float8_e4m3fn"):
    _DTYPES.append(torch.float8_e4m3fn)


@pytest.mark.parametrize("flat_views", [False, True], ids=["standalone", "flat_views"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_saved_shard_round_trips_through_weights_only_load(tmp_path, monkeypatch, dtype, flat_views):
    saved = _expert_params(dtype, flat_views=flat_views, seed=0)
    _save(monkeypatch, saved, tmp_path)
    path = _shard_path(tmp_path)
    assert path.is_file()
    assert not (tmp_path / (path.name + ".partial")).exists()
    expert_bytes = sum(p.data.numel() * p.data.element_size() for _, p in saved)
    assert path.stat().st_size <= expert_bytes + _ARCHIVE_OVERHEAD

    # The shard format itself must be weights-only clean, independent of the loader.
    torch.load(path, map_location="cpu", weights_only=True)

    restored = _expert_params(dtype, seed=1)
    _load(monkeypatch, restored, tmp_path)
    for (name, want), (_, got) in zip(saved, restored):
        assert got.data.dtype == dtype
        assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8)), name


@pytest.mark.parametrize("flat_views", [False, True], ids=["standalone", "flat_views"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_plain_torch_save_shard_loads(tmp_path, monkeypatch, dtype, flat_views):
    # A direct ``torch.save`` of the tensors. On CPU ``.cpu()`` returns the tensor itself, so the flat-view shard holds
    # the whole flat buffer, non-expert bytes included, as one storage that every expert shares.
    items = _expert_params(dtype, flat_views=flat_views, seed=0)
    path = _shard_path(tmp_path)
    torch.save({n: p.data.detach().cpu() for n, p in items}, path)
    shard = torch.load(path, mmap=True, weights_only=True)
    assert len({t.untyped_storage().data_ptr() for t in shard.values()}) == (1 if flat_views else len(items))
    del shard

    restored = _expert_params(dtype, seed=1)
    _load(monkeypatch, restored, tmp_path)
    for (name, want), (_, got) in zip(items, restored):
        assert got.data.dtype == dtype
        assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8)), name


def _spy_load_syscalls(monkeypatch):
    """Record the loader's reads as ``("read", offset, n)`` and its page-cache drops as ``("drop", ...)``."""
    events = []
    real = {name: getattr(os, name) for name in ("pread", "preadv", "posix_fadvise")}

    def pread(fd, n, offset):
        data = real["pread"](fd, n, offset)
        events.append(("read", offset, len(data)))
        return data

    def preadv(fd, buffers, offset):
        n = real["preadv"](fd, buffers, offset)
        events.append(("read", offset, n))
        return n

    def posix_fadvise(fd, offset, length, advice):
        events.append(("drop", offset, length, advice))
        return real["posix_fadvise"](fd, offset, length, advice)

    for name, fn in (("pread", pread), ("preadv", preadv), ("posix_fadvise", posix_fadvise)):
        monkeypatch.setattr(os, name, fn)
    return events


def _read_ranges(events):
    # A read that continues the previous one extends its range.
    ranges = []
    for event in events:
        if event[0] == "read" and ranges and ranges[-1][0] == "read" and ranges[-1][1] + ranges[-1][2] == event[1]:
            ranges[-1] = ("read", ranges[-1][1], ranges[-1][2] + event[2])
        else:
            ranges.append(event)
    return ranges


def test_shared_storage_record_is_read_once(tmp_path, monkeypatch):
    # Views of one flat buffer that also holds 1 MiB of non-expert bytes, saved directly, share one storage record.
    saved = _expert_params(torch.bfloat16, flat_views=True)
    path = _shard_path(tmp_path)
    torch.save({n: p.data for n, p in saved}, path)
    records = set(epc._shard_slot_offsets(str(path)).values())
    assert len(records) == 1
    ((offset, size),) = records
    assert size == saved[0][1].data.untyped_storage().nbytes()

    events = _spy_load_syscalls(monkeypatch)
    restored = _expert_params(torch.bfloat16, seed=1)
    _load(monkeypatch, restored, tmp_path)

    assert _read_ranges(events) == [("read", offset, size), ("drop", offset, size, os.POSIX_FADV_DONTNEED)]
    for (name, want), (_, got) in zip(saved, restored):
        assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8)), name


def test_non_contiguous_tensor_loads(tmp_path, monkeypatch):
    name = "layers.0.mlp.experts.down_proj"
    want = torch.randn(8, 4, generator=torch.Generator().manual_seed(0)).t()
    assert not want.is_contiguous()
    torch.save({name: want}, _shard_path(tmp_path))
    restored = _expert_params(torch.float32, seed=1, shapes={name: (4, 8)})
    _load(monkeypatch, restored, tmp_path)
    assert torch.equal(restored[0][1].data, want)


def test_tensor_in_a_larger_storage_loads(tmp_path, monkeypatch):
    name = "layers.0.mlp.experts.down_proj"
    big = torch.randn(100, generator=torch.Generator().manual_seed(0))
    want = big[:32].view(2, 4, 4)
    torch.save({name: want}, _shard_path(tmp_path))
    restored = _expert_params(torch.float32, seed=1, shapes={name: (2, 4, 4)})
    _load(monkeypatch, restored, tmp_path)
    assert torch.equal(restored[0][1].data, want)


def test_rank_without_experts_round_trips(tmp_path, monkeypatch):
    _save(monkeypatch, [], tmp_path)
    _load(monkeypatch, [], tmp_path)


_PWNED = []


def _pwn(marker):
    _PWNED.append(marker)
    return {}


class _Evil:
    def __reduce__(self):
        return (_pwn, ("executed",))


def test_tampered_shard_is_refused_without_executing(tmp_path, monkeypatch):
    items = _expert_params(torch.bfloat16)
    torch.save({"payload": _Evil(), **{n: p.data for n, p in items}}, _shard_path(tmp_path))
    before = [p.data.clone() for _, p in items]
    reads = []
    for name in ("pread", "preadv"):
        monkeypatch.setattr(os, name, lambda *args, name=name: reads.append(name))

    with pytest.raises(RuntimeError, match="corrupt or not a plain tensor dict"):
        _load(monkeypatch, items, tmp_path)

    assert _PWNED == []
    assert reads == []
    assert all(torch.equal(p.data, b) for (_, p), b in zip(items, before))


def test_non_dict_shard_is_rejected(tmp_path, monkeypatch):
    items = _expert_params(torch.float32)
    torch.save([p.data for _, p in items], _shard_path(tmp_path))
    with pytest.raises(TypeError, match="expected a dict"):
        _load(monkeypatch, items, tmp_path)


def test_garbage_shard_is_reported_as_corrupt(tmp_path, monkeypatch):
    items = _expert_params(torch.float32)
    _shard_path(tmp_path).write_bytes(b"not a torch checkpoint")
    with pytest.raises(RuntimeError, match="corrupt or not a plain tensor dict"):
        _load(monkeypatch, items, tmp_path)


def test_truncated_shard_fails_loudly(tmp_path, monkeypatch):
    # A cut-off zip has no central directory, so it is refused as not a zip archive before torch opens it.
    items = _expert_params(torch.float32)
    _save(monkeypatch, items, tmp_path)
    path = _shard_path(tmp_path)
    path.write_bytes(path.read_bytes()[:64])
    with pytest.raises(RuntimeError, match="corrupt"):
        _load(monkeypatch, items, tmp_path)


@pytest.mark.parametrize(
    "tamper, reason",
    [
        ("truncated_directory", "not a zip archive"),
        ("unreferenced_record", "its zip directory does not locate"),
        ("record_past_end", r"its zip directory does not locate .*\): .* past the end of the \d+-byte file"),
    ],
)
def test_tampered_zip_directory_is_reported_as_corrupt(tmp_path, monkeypatch, tamper, reason):
    items = _expert_params(torch.float32)
    _save(monkeypatch, items, tmp_path)
    path = _shard_path(tmp_path)
    if tamper == "truncated_directory":
        # Cutting into the end-of-central-directory record leaves no zip directory.
        path.write_bytes(path.read_bytes()[:-8])
    elif tamper == "unreferenced_record":
        # The directory gains a data record that no tensor refers to, which the unpickler never sees.
        with zipfile.ZipFile(path, "a") as archive:
            prefix = archive.namelist()[0].split("/")[0]
            archive.writestr(f"{prefix}/data/99", b"\0" * 64)
    else:
        # The directory declares a record one byte longer than the file holds after the record's start. torch's zip
        # reader bounds a record from its local header, before the header's name and padding, so it still opens this
        # archive; a record that reaches further past the end is refused by torch before the loader sees it.
        raw = bytearray(path.read_bytes())
        with zipfile.ZipFile(path) as archive:
            info = next(i for i in archive.infolist() if i.filename.endswith("/data/0"))
        name_len, extra_len = struct.unpack_from("<HH", raw, info.header_offset + 26)
        declared = len(raw) - (info.header_offset + 30 + name_len + extra_len) + 1
        entry = raw.rindex(info.filename.encode()) - 46
        assert raw[entry : entry + 4] == b"PK\x01\x02"
        struct.pack_into("<II", raw, entry + 20, declared, declared)
        path.write_bytes(raw)
    before = [p.data.clone() for _, p in items]
    # The loader's host buffers for storage records are uint8; none is allocated before the shard is refused.
    allocations = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.get("dtype") is torch.uint8:
            allocations.append(args)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(RuntimeError, match=f"corrupt or not a plain tensor dict \\({reason}"):
        _load(monkeypatch, items, tmp_path)
    assert allocations == []
    assert all(torch.equal(p.data, b) for (_, p), b in zip(items, before))


def test_non_tensor_value_is_rejected(tmp_path, monkeypatch):
    items = _expert_params(torch.float32)
    name = items[0][0]
    torch.save({n: p.data for n, p in items} | {name: [1.0, 2.0]}, _shard_path(tmp_path))
    with pytest.raises(TypeError, match=f"{name}.*expected a tensor"):
        _load(monkeypatch, items, tmp_path)


def test_loader_never_requests_unsafe_load(tmp_path, monkeypatch):
    items = _expert_params(torch.float32)
    _save(monkeypatch, items, tmp_path)
    calls = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    _load(monkeypatch, items, tmp_path)
    assert calls and all(weights_only is True for weights_only in calls)


def test_failed_save_leaves_no_partial_or_shard(tmp_path, monkeypatch):
    items = _expert_params(torch.float32)
    calls = []
    real_pwrite = os.pwrite

    def pwrite_then_fail(fd, data, offset):
        calls.append(offset)
        if len(calls) == 2:
            raise OSError("write interrupted")
        return real_pwrite(fd, data, offset)

    monkeypatch.setattr(os, "pwrite", pwrite_then_fail)
    with pytest.raises(OSError, match="write interrupted"):
        _save(monkeypatch, items, tmp_path)
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_each_tensor_is_flushed_and_dropped_from_the_cache_before_the_next(tmp_path, monkeypatch):
    # Every written range is on disk and out of the page cache before the next tensor is staged, so a rank never holds
    # more than one tensor of dirty pages.
    items = _expert_params(torch.bfloat16, flat_views=True)
    events = []
    real = {name: getattr(os, name) for name in ("pwrite", "fdatasync", "posix_fadvise")}

    def pwrite(fd, data, offset):
        n = real["pwrite"](fd, data, offset)
        events.append(("write", offset, n))
        return n

    def fdatasync(fd):
        events.append(("sync",))
        return real["fdatasync"](fd)

    def posix_fadvise(fd, offset, length, advice):
        events.append(("drop", offset, length, advice))
        return real["posix_fadvise"](fd, offset, length, advice)

    for name, fn in (("pwrite", pwrite), ("fdatasync", fdatasync), ("posix_fadvise", posix_fadvise)):
        monkeypatch.setattr(os, name, fn)
    _save(monkeypatch, items, tmp_path)

    expected = []
    for _, p in items:
        offset = next(e[1] for e in events[len(expected) :] if e[0] == "write")
        n = p.data.numel() * p.data.element_size()
        expected += [("write", offset, n), ("sync",), ("drop", offset, n, os.POSIX_FADV_DONTNEED)]
    assert events == expected


def test_each_record_is_read_alone_and_dropped_from_the_cache_before_the_next(tmp_path, monkeypatch):
    # Every storage record is read on its own and dropped from the page cache before the next record is read, in file
    # order, and the archive is only ever opened mapped. The saver gives every tensor its own storage, so a rank never
    # holds more than one tensor of the shard in host memory.
    saved = _expert_params(torch.bfloat16, flat_views=True)
    _save(monkeypatch, saved, tmp_path)
    slots = epc._shard_slot_offsets(str(_shard_path(tmp_path)))
    loads = []
    real_load = torch.load

    def load(*args, **kwargs):
        loads.append(kwargs)
        return real_load(*args, **kwargs)

    events = _spy_load_syscalls(monkeypatch)
    monkeypatch.setattr(torch, "load", load)
    restored = _expert_params(torch.bfloat16, seed=1)
    _load(monkeypatch, restored, tmp_path)

    assert len(set(slots.values())) == len(restored)
    expected = []
    for offset, size in sorted(slots[name] for name, _ in restored):
        expected += [("read", offset, size), ("drop", offset, size, os.POSIX_FADV_DONTNEED)]
    assert _read_ranges(events) == expected
    assert loads and all(kwargs.get("mmap") is True and kwargs.get("weights_only") is True for kwargs in loads)
    for (name, want), (_, got) in zip(saved, restored):
        assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8)), name


def test_slot_offsets_on_a_zip64_layout(tmp_path):
    # Production shards are tens of GiB, so torch.save writes zip64 records: here one record is larger than 4 GiB and
    # the records after it start beyond 4 GiB. Placeholders under ``skip_data`` carry no bytes, so the file is sparse;
    # its directory is removed when the test ends, whatever the outcome.
    gib = 1 << 30
    layout = {
        "layers.0.mlp.experts.gate_up_proj": torch.empty(5 * gib // 2, dtype=torch.bfloat16),
        "layers.0.mlp.experts.down_proj": torch.empty(1024, dtype=torch.bfloat16),
        "layers.1.mlp.experts.gate_up_proj": torch.empty(gib, dtype=torch.uint8),
    }
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        partial = Path(scratch) / "layout.pt.partial"
        with torch.serialization.skip_data():
            torch.save(layout, partial)

        slots = epc._shard_slot_offsets(str(partial))
        assert set(slots) == set(layout)
        for name, t in layout.items():
            assert slots[name][1] == t.numel() * t.element_size(), name
        spans = sorted(slots.values())
        assert spans[-1][0] > 4 * gib
        for (offset, size), (next_offset, _) in zip(spans, spans[1:]):
            assert offset + size <= next_offset
        assert spans[-1][0] + spans[-1][1] <= partial.stat().st_size

        # A marker at both ends of every slot lands at both ends of that tensor when the archive is loaded.
        fd = os.open(partial, os.O_WRONLY)
        try:
            for i, (offset, size) in enumerate(slots[name] for name in layout):
                os.pwrite(fd, bytes([i + 1]) * 8, offset)
                os.pwrite(fd, bytes([i + 101]) * 8, offset + size - 8)
        finally:
            os.close(fd)
        loaded = torch.load(partial, mmap=True, weights_only=True)
        for i, name in enumerate(layout):
            data = loaded[name].reshape(-1).view(torch.uint8)
            assert data[:8].tolist() == [i + 1] * 8, name
            assert data[-8:].tolist() == [i + 101] * 8, name
        del data, loaded
    assert list(tmp_path.iterdir()) == []


def _rss_anon_bytes():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("RssAnon missing from /proc/self/status")


def _peak_rss_anon_growth(fn):
    """Peak growth of this process's anonymous memory while ``fn`` runs, sampled every 1 ms."""
    torch.cuda.synchronize()
    baseline = _rss_anon_bytes()
    peak = [baseline]
    done = threading.Event()

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], _rss_anon_bytes())
            done.wait(0.001)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        fn()
    finally:
        done.set()
        sampler.join()
    return peak[0] - baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_flat_view_shard_holds_only_expert_bytes_without_a_host_copy(tmp_path, monkeypatch):
    # The experts are views of a flat CUDA buffer that carries another 256 MiB of non-expert parameters.
    saved = _expert_params(torch.bfloat16, flat_views=True, shapes=_GPU_SHAPES, tail_bytes=256 << 20, device="cuda")
    expert_bytes = sum(p.data.numel() * p.data.element_size() for _, p in saved)
    assert len({p.data.untyped_storage().data_ptr() for _, p in saved}) == 1

    growth = _peak_rss_anon_growth(lambda: _save(monkeypatch, saved, tmp_path))

    path = _shard_path(tmp_path)
    assert path.stat().st_size <= expert_bytes + _ARCHIVE_OVERHEAD
    # A saver that stages the experts on the host grows anonymous memory by at least ``expert_bytes`` (by the whole
    # 512 MiB buffer if it writes the backing storage); this writer stages one 32 MiB tensor at a time.
    assert growth < expert_bytes // 2, (growth, expert_bytes)

    shard = torch.load(path, map_location="cpu", weights_only=True)
    assert len({t.untyped_storage().data_ptr() for t in shard.values()}) == len(saved)
    for name, p in saved:
        assert torch.equal(shard[name].view(torch.uint8), p.data.cpu().view(torch.uint8)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_load_reads_one_tensor_at_a_time_without_a_host_copy(tmp_path, monkeypatch):
    # The shard is restored into views of a second flat CUDA buffer with the same non-expert tail.
    saved = _expert_params(torch.bfloat16, flat_views=True, shapes=_GPU_SHAPES, tail_bytes=256 << 20, device="cuda")
    _save(monkeypatch, saved, tmp_path)
    restored = _expert_params(
        torch.bfloat16, flat_views=True, seed=1, shapes=_GPU_SHAPES, tail_bytes=256 << 20, device="cuda"
    )
    expert_bytes = sum(p.data.numel() * p.data.element_size() for _, p in restored)

    growth = _peak_rss_anon_growth(lambda: _load(monkeypatch, restored, tmp_path))

    # A loader that reads the shard into host memory grows anonymous memory by at least ``expert_bytes``; this reader
    # holds one 32 MiB tensor at a time.
    assert growth < expert_bytes // 2, (growth, expert_bytes)
    for (name, want), (_, got) in zip(saved, restored):
        assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8)), name


class _Engine:
    """Records ``save_checkpoint``/``load_checkpoint`` with the MoE and universal flags each saw."""

    def __init__(self, *, expert_parallel: bool = True):
        self.global_steps = 9
        self.has_moe_layers = True
        self._config = SimpleNamespace(load_universal_checkpoint=False)
        self.calls: list[tuple] = []
        self.master_refreshes: list[bool] = []
        self.optimizer = SimpleNamespace(refresh_fp32_params=lambda: self.master_refreshes.append(True))
        param = nn.Parameter(torch.zeros(1))
        if expert_parallel:
            param.allreduce = False
            param.group_name = "ep_size_2"
        self.module = SimpleNamespace(named_parameters=lambda: iter([("layers.0.mlp.experts.w1", param)]))

    def save_checkpoint(self, save_dir, tag=None, **kwargs):
        self.calls.append(("save", save_dir, tag, self.has_moe_layers, kwargs))
        os.makedirs(os.path.join(save_dir, tag), exist_ok=True)

    def load_checkpoint(self, load_dir, tag=None, **kwargs):
        self.calls.append(("load", load_dir, tag, self.has_moe_layers, self._config.load_universal_checkpoint, kwargs))
        return f"{load_dir}/{tag}", {}


@pytest.fixture
def stub_experts(monkeypatch):
    """Expert-file reads and writes, and the dense-only model-states load, recorded instead of run."""
    events = []
    monkeypatch.setattr(epc, "save_expert_shards", lambda engine, tag_dir: events.append(("save_shards", tag_dir)))
    monkeypatch.setattr(epc, "load_expert_shards", lambda engine, tag_dir: events.append(("load_shards", tag_dir)))
    monkeypatch.setattr(epc, "keep_local_expert_params", lambda engine: contextlib.nullcontext())
    return events


def test_save_writes_the_dense_state_without_moe_then_the_expert_files(tmp_path, stub_experts):
    engine = _Engine()
    assert epc.save_expert_parallel_checkpoint(engine, str(tmp_path)) == TAG
    assert engine.calls == [("save", str(tmp_path), TAG, False, {})]
    assert engine.has_moe_layers is True
    assert stub_experts == [("save_shards", str(tmp_path / TAG))]


@pytest.mark.parametrize(
    "module_only, load_optimizer_states, expected_lr_scheduler",
    [
        (False, True, True),  # full resume: shards, optimizer, and scheduler
        (False, False, False),  # keep fp32 masters, fresh moments and scheduler
        (True, False, False),  # module-only: weights only, masters rebuilt from bf16
    ],
)
def test_load_forwards_flags_and_refreshes_masters_only_after_a_module_only_load(
    stub_experts, module_only, load_optimizer_states, expected_lr_scheduler
):
    # The expert files load after DeepSpeed, so module-only masters must be refreshed from the restored experts. A
    # refresh on the other paths would copy the module over the restored fp32 masters.
    engine = _Engine()
    epc.load_expert_parallel_checkpoint(
        engine, "/dir", TAG, load_module_only=module_only, load_optimizer_states=load_optimizer_states
    )
    flags = {
        "load_module_only": module_only,
        "load_optimizer_states": load_optimizer_states,
        "load_lr_scheduler_states": expected_lr_scheduler,
    }
    assert engine.calls == [("load", "/dir", TAG, False, False, flags)]
    assert stub_experts == [("load_shards", f"/dir/{TAG}")]
    assert engine.master_refreshes == ([True] if module_only else [])


def test_universal_load_reads_no_expert_file(stub_experts):
    engine = _Engine()
    epc.load_expert_parallel_checkpoint(engine, "/universal", "copy", universal=True, load_optimizer_states=False)
    flags = {"load_optimizer_states": False, "load_lr_scheduler_states": False}
    assert engine.calls == [("load", "/universal", "copy", False, True, flags)]
    assert (stub_experts, engine._config.load_universal_checkpoint) == ([], False)


def test_failed_dense_load_reads_no_expert_file(stub_experts):
    engine = _Engine()
    engine.load_checkpoint = lambda *a, **k: (None, None)
    assert epc.load_expert_parallel_checkpoint(engine, "/dir", TAG) == (None, None)
    assert stub_experts == []


def test_load_resolves_the_latest_tag(tmp_path, stub_experts):
    (tmp_path / "latest").write_text(TAG)
    engine = _Engine()
    epc.load_expert_parallel_checkpoint(engine, str(tmp_path))
    assert engine.calls[0][2] == TAG
    with pytest.raises(RuntimeError, match="no 'latest' marker"):
        epc.load_expert_parallel_checkpoint(engine, str(tmp_path / "empty"))


def test_load_deepspeed_checkpoint_follows_the_format_on_disk(tmp_path, stub_experts):
    native = tmp_path / "native"
    (native / TAG).mkdir(parents=True)
    (native / "latest").write_text(TAG)
    ours = tmp_path / "ours"
    (ours / TAG).mkdir(parents=True)
    (ours / TAG / epc.expert_shard_name(0)).write_bytes(b"")
    (ours / "latest").write_text(TAG)

    engine = _Engine()
    epc.load_deepspeed_checkpoint(engine, str(native))
    epc.load_deepspeed_checkpoint(engine, str(ours))
    # DeepSpeed's own format keeps its MoE path; this format loads the dense state without it, then the experts.
    assert [(c[1], c[2], c[3]) for c in engine.calls] == [(str(native), None, True), (str(ours), TAG, False)]
    assert stub_experts == [("load_shards", str(ours / TAG))]


@pytest.mark.parametrize("expert_parallel", [True, False])
def test_load_deepspeed_checkpoint_loads_a_universal_copy_by_the_engine(stub_experts, expert_parallel):
    engine = _Engine(expert_parallel=expert_parallel)
    epc.load_deepspeed_checkpoint(engine, "/universal", "copy", universal=True)
    ((_, _, _, has_moe, universal, _),) = engine.calls
    assert (has_moe, universal) == (not expert_parallel, True)
    assert stub_experts == []


@require_deepspeed
class TestExpertShardPaths:
    @staticmethod
    def _engine(monkeypatch, ep_size: int):
        import deepspeed.utils.groups as ds_groups
        import torch.distributed as dist

        expert = SimpleNamespace(group_name=f"ep_size_{ep_size}")
        monkeypatch.setattr(epc, "expert_params_and_ranks", lambda engine: ([("e.w1", expert)], 0, 0))
        monkeypatch.setattr(ds_groups, "_get_expert_parallel_group", lambda name: name)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: int(group.rsplit("_", 1)[1]))
        return SimpleNamespace()

    @staticmethod
    def _write(tag_dir: Path, ranks) -> None:
        tag_dir.mkdir(parents=True, exist_ok=True)
        for rank in ranks:
            (tag_dir / epc.expert_shard_name(rank)).write_bytes(b"")

    def test_lists_each_ep_ranks_file_in_order(self, tmp_path, monkeypatch):
        self._write(tmp_path, [1, 0])
        (tmp_path / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"")
        paths = epc.expert_shard_paths(self._engine(monkeypatch, 2), str(tmp_path))
        assert paths == [str(tmp_path / epc.expert_shard_name(r)) for r in (0, 1)]
        assert epc.has_expert_shards(str(tmp_path)) and not epc.has_expert_shards(str(tmp_path / "missing"))

    def test_another_ep_size_is_refused(self, tmp_path, monkeypatch):
        self._write(tmp_path, [0, 1])
        with pytest.raises(RuntimeError, match="saved at ep_size 2 but this job runs at ep_size 4"):
            epc.expert_shard_paths(self._engine(monkeypatch, 4), str(tmp_path))

    def test_a_missing_ep_rank_is_refused(self, tmp_path, monkeypatch):
        self._write(tmp_path, [0, 2])
        with pytest.raises(RuntimeError, match="expected"):
            epc.expert_shard_paths(self._engine(monkeypatch, 2), str(tmp_path))
