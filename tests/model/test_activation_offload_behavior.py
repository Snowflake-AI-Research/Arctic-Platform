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

"""Tests for ``activation_offload``.

CPU-only tests cover the install/toggle plumbing. Single-GPU tests cover the CUDA offload path, including
numerical transparency, peak-memory savings, small-tensor passthrough, and pinned staging-cache behavior.
No gateway, SP, DeepEP, or 35B model is required.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from arctic_platform.model.implementations.gpu.activation_offload import ActivationOffloadManager
from arctic_platform.model.implementations.gpu.activation_offload import PoolKey
from arctic_platform.model.implementations.gpu.activation_offload import activation_offload_stats
from arctic_platform.model.implementations.gpu.activation_offload import host_pin_reserved_bytes
from arctic_platform.model.implementations.gpu.activation_offload import install_activation_offload

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="activation CPU-offload requires a GPU")

# Sized above the offload crossover: offloading adds a fixed overhead (pinned staging + a resident reload
# buffer during backward), so it only lowers the peak once the streamed boundaries dominate that overhead.
# Each boundary is 16384 x 2048 x 4B (fp32) = 128 MiB (>> the 1 MiB threshold); with 16 blocks the boundaries
# dominate and offload wins by ~20%. Smaller stacks (e.g. 4096 x 1024 x 6) sit below the crossover and offload
# can raise the peak -- see scripts/exercise_activation_offload.py --sweep for the curve.
TOKENS = 16384
HIDDEN = 2048
NUM_BLOCKS = 16


class _Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + self.lin2(torch.relu(self.lin1(x)))


class _Net(nn.Module):
    """Stack of residual blocks, each wrapped in non-reentrant activation checkpointing; the per-block saved
    inputs are the boundaries the offload manager streams to CPU."""

    def __init__(self, num_blocks: int, hidden: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            checkpoint_wrapper(_Block(hidden), preserve_rng_state=False) for _ in range(num_blocks)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _fresh_net() -> nn.Module:
    torch.manual_seed(0)  # identical init on every build so baseline vs offload are comparable
    return _Net(NUM_BLOCKS, HIDDEN).cuda()


def _run_step(model: nn.Module, x: torch.Tensor) -> tuple[float, list[torch.Tensor]]:
    for p in model.parameters():
        p.grad = None
    loss = model(x).float().pow(2).mean()
    loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    return loss.item(), grads


def _tiny_checkpointed_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(checkpoint_wrapper(nn.Linear(8, 8), preserve_rng_state=False))


def _count_pack_calls(model: nn.Module, manager: ActivationOffloadManager) -> int:
    """Run one fwd/bwd and count how many times the manager's pack hook fires."""
    calls = 0
    original_pack = manager.pack

    def counting_pack(tensor):
        nonlocal calls
        calls += 1
        return original_pack(tensor)

    manager.pack = counting_pack
    try:
        x = torch.randn(4, 8, requires_grad=True)
        model(x).pow(2).sum().backward()
    finally:
        manager.pack = original_pack
    return calls


def test_install_applies_all_knobs_to_manager():
    manager = install_activation_offload(
        _tiny_checkpointed_model(),
        keep_last_n=3,
        use_streams=False,
        tensor_size_threshold=4096,
        pin_memory_enabled=False,
        pin_memory_max_size_gib=0.125,
        pin_memory_bucket_size_mib=2,
    )
    assert manager.enabled is True
    assert manager.keep_last_n == 3
    assert manager.use_streams is False
    assert manager.tensor_size_threshold == 4096
    assert manager.pin_memory_enabled is False
    assert manager._pin_memory_hard_max_size_bytes == 128 << 20
    assert manager.pin_memory_bucket_size_bytes == 2 << 20


def test_install_validates_pin_memory_sizes():
    with pytest.raises(ValueError, match="pin_memory_max_size_gib"):
        install_activation_offload(_tiny_checkpointed_model(), pin_memory_max_size_gib=-1)
    with pytest.raises(ValueError, match="pin_memory_bucket_size_mib"):
        install_activation_offload(_tiny_checkpointed_model(), pin_memory_bucket_size_mib=0)


def test_repeat_install_reconfigures_the_same_manager():
    model = _tiny_checkpointed_model()
    first = install_activation_offload(model, keep_last_n=1, use_streams=True)
    second = install_activation_offload(model, keep_last_n=5, use_streams=False)

    assert first is second, "install must be idempotent per module (one manager, reconfigured)"
    assert second.keep_last_n == 5
    assert second.use_streams is False


def test_reconfigure_none_tensor_size_threshold_resets_to_default():
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=1,
        use_streams=False,
        tensor_size_threshold=4096,
        pin_memory_enabled=True,
    )
    assert manager.tensor_size_threshold == 4096

    manager.configure(keep_last_n=1, use_streams=False, tensor_size_threshold=None)
    assert manager.tensor_size_threshold == 1 << 20


def test_offload_toggle_gates_the_saved_tensor_hooks():
    model = _tiny_checkpointed_model()
    manager = install_activation_offload(model, keep_last_n=1, use_streams=False)

    # ON: the forward runs under saved_tensors_hooks, so pack intercepts the checkpoint boundary.
    manager.enabled = True
    assert _count_pack_calls(model, manager) > 0, "enabled offload must intercept saved activations"

    # OFF: the wrapped forward short-circuits to the original, so pack never fires.
    manager.enabled = False
    assert _count_pack_calls(model, manager) == 0, "disabled offload must be a pure passthrough"


@pytest.mark.integration
@requires_cuda
def test_offload_is_numerically_transparent_and_saves_memory():
    x = torch.randn(TOKENS, HIDDEN, device="cuda", requires_grad=True)

    # -- baseline: identical model, no offload --
    baseline = _fresh_net()
    torch.cuda.reset_peak_memory_stats()
    base_loss, base_grads = _run_step(baseline, x)
    base_peak = torch.cuda.max_memory_allocated()
    # Free the baseline model+grads off the GPU before measuring the offloaded run, otherwise their ~GiB
    # footprint would be counted in off_peak and mask the saving; keep the grads on CPU for comparison.
    base_grads = [g.cpu() for g in base_grads]
    del baseline
    torch.cuda.empty_cache()

    # -- offloaded: same init, saved activations streamed to CPU --
    offloaded = _fresh_net()
    manager = install_activation_offload(offloaded, keep_last_n=1, use_streams=True)
    assert manager.enabled
    torch.cuda.reset_peak_memory_stats()
    off_loss, off_grads = _run_step(offloaded, x)
    torch.cuda.synchronize()
    off_peak = torch.cuda.max_memory_allocated()

    # correctness: loss and all grads match (same init/input, offload only relocates the saved boundary).
    assert off_loss == pytest.approx(base_loss, rel=1e-5, abs=1e-5)
    for base_grad, off_grad in zip(base_grads, off_grads):
        torch.testing.assert_close(off_grad.cpu(), base_grad, rtol=1e-4, atol=1e-4)

    # memory: offloading the block boundaries lowers the peak CUDA allocation.
    assert off_peak < base_peak, f"expected offload to reduce peak: baseline={base_peak}, offload={off_peak}"

    # stats: boundaries were actually offloaded and restored.
    assert manager.stats.offloaded_tensors > 0
    assert manager.stats.restored_tensors > 0
    assert manager.stats.offloaded_bytes > 0
    assert activation_offload_stats(offloaded) is not None


@pytest.mark.integration
@requires_cuda
def test_small_tensors_are_passed_through_not_offloaded():
    """Tensors below ``tensor_size_threshold`` stay resident (offloading tiny tensors is pure latency)."""
    manager = ActivationOffloadManager()
    manager.configure(keep_last_n=0, use_streams=False, tensor_size_threshold=1 << 20)
    tiny = torch.ones(16, device="cuda", requires_grad=True)  # 64 B << 1 MiB
    with manager.hooks_ctx():
        y = tiny.pow(2).sum()  # pow saves its input -> the pack hook fires on a sub-threshold tensor
    y.backward()
    assert manager.stats.offloaded_tensors == 0
    assert manager.stats.passed_small > 0


class _FakeSlot:
    def __init__(self, seq_len: int, hidden: int = 256, dtype: torch.dtype = torch.float16):
        self.shape = (seq_len, hidden)
        self.stride = (hidden, 1)
        self.dtype = dtype
        self.nbytes = seq_len * hidden * torch.empty((), dtype=dtype).element_size()
        self.is_contiguous = True
        self.pinned_entry = None
        self.cpu = None
        self.d2h_event = None
        self.offloaded = True


class _FakeStridedSlot:
    def __init__(self, shape: tuple[int, ...], stride: tuple[int, ...], dtype: torch.dtype = torch.float16):
        self.shape = shape
        self.stride = stride
        self.dtype = dtype
        numel = 1
        for dim in shape:
            numel *= dim
        self.nbytes = numel * torch.empty((), dtype=dtype).element_size()
        self.is_contiguous = False
        self.pinned_entry = None
        self.cpu = None
        self.d2h_event = None
        self.offloaded = True


class _PrunedSavedTensorModule(nn.Module):
    def forward(self, x):
        _ = (x * x).sum()  # packed by saved_tensors_hooks, then pruned from the returned loss graph
        return x.sum()


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _host_pin_reserved_bytes() -> int:
    """Pinned host bytes retained by PyTorch's host caching allocator (not manager counters)."""
    return host_pin_reserved_bytes()


def test_host_pin_reserved_bytes_reads_the_key_torch_publishes(monkeypatch):
    """The pinned pool is reported from ``host_memory_stats``, whose key names differ between torch versions.

    A key that no longer exists reads as zero, and a zero here is indistinguishable from an offload that never
    pinned anything -- the report looks healthy while measuring nothing. The stats below are the ones torch 2.11
    publishes for the caching host allocator.
    """
    stats = {
        "active_bytes.current": 67_108_864,
        "allocated_bytes.current": 67_108_864,
        "allocated_bytes.peak": 67_108_864,
        "allocations.current": 1,
        "num_host_alloc": 1,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda.memory, "host_memory_stats", lambda: stats)

    assert host_pin_reserved_bytes() == 67_108_864


def _pinned_cache_acquire(manager: ActivationOffloadManager, slot: _FakeSlot | _FakeStridedSlot) -> torch.Tensor:
    buffer = manager.pinned_cache.acquire(
        slot,
        offload_stream=manager._offload_stream,
        use_streams=manager.use_streams,
    )
    manager.sync_pinned_stats()
    return buffer


def _pinned_cache_release(manager: ActivationOffloadManager, slot: _FakeSlot | _FakeStridedSlot, event=None) -> None:
    manager.pinned_cache.release(slot, event)
    manager.sync_pinned_stats()


def _finalize_pinned_cache_step(manager: ActivationOffloadManager) -> None:
    manager.pinned_cache.finalize_step()
    manager.sync_pinned_stats()


def _run_fake_step(
    manager: ActivationOffloadManager,
    slots: list[_FakeSlot] | list[_FakeStridedSlot],
) -> None:
    buffers = [_pinned_cache_acquire(manager, slot) for slot in slots]
    for slot, buffer in zip(slots, buffers):
        assert buffer.is_pinned()
        _pinned_cache_release(manager, slot, None)
    _finalize_pinned_cache_step(manager)


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_reuses_capacity_buffers_for_variable_sequence_lengths():
    """Five different sequence lengths should reuse the first step's larger contiguous pinned buffers."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=1,
        pin_memory_bucket_size_mib=1,
    )

    for seq_len in [512, 480, 448, 416, 384]:
        slots = [_FakeSlot(seq_len) for _ in range(4)]
        buffers = [_pinned_cache_acquire(manager, slot) for slot in slots]
        for slot, buffer in zip(slots, buffers):
            assert buffer.shape == slot.shape
            assert buffer.stride() == slot.stride
            assert buffer.is_pinned()
            _pinned_cache_release(manager, slot, None)

    cache_tensors, _ = manager.pinned_cache.cache_size()
    assert manager.stats.pinned_allocations == 4
    assert manager.stats.pinned_reuses == 16
    assert cache_tensors == 4


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_replaces_dominated_buffers_for_low_high_alternating_sequence_lengths():
    """After seeing the high seqlen, lower-seqlen pinned buffers should not remain cached beside it."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=1,
        pin_memory_bucket_size_mib=1,
    )

    hidden = 4096
    dtype = torch.float16
    seq_lens = [4096, 8192, 4096, 8192, 4096, 8192]
    bucket_nbytes = 1 << 20
    boundary_nbytes = max(seq_lens) * hidden * torch.empty((), dtype=dtype).element_size()
    buffers_per_step = 2
    expected_cache_bytes = buffers_per_step * _round_up(boundary_nbytes, bucket_nbytes)
    observed_rss = []

    for seq_len in seq_lens:
        slots = [_FakeSlot(seq_len, hidden=hidden, dtype=dtype) for _ in range(buffers_per_step)]
        _run_fake_step(manager, slots)
        observed_rss.append(_rss_bytes())

    cache_tensors, cache_bytes = manager.pinned_cache.cache_size()
    assert cache_tensors == buffers_per_step
    assert cache_bytes == expected_cache_bytes
    assert manager.stats.pinned_evictions == buffers_per_step
    assert manager.stats.pinned_reuses >= buffers_per_step * (len(seq_lens) - 2)

    cache_cap_mibs = sorted(entry.capacity_bytes // (1 << 20) for entry in manager.pinned_cache.fifo_entries)
    assert (
        cache_cap_mibs == [boundary_nbytes // bucket_nbytes] * buffers_per_step
    ), f"expected only the high-seqlen bucket capacities in cache, got {cache_cap_mibs} MiB"

    baseline_rss = observed_rss[1]  # step 1 can include one-time allocator and paging noise.
    tolerance_bytes = boundary_nbytes
    for step_idx, rss in enumerate(observed_rss[2:], start=3):
        assert rss <= baseline_rss + tolerance_bytes, (
            f"CPU RSS increased too much at step {step_idx}: baseline_step_2={baseline_rss}, "
            f"rss={rss}, tolerance={tolerance_bytes}, one_boundary={boundary_nbytes}"
        )


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_auto_cap_grows_from_completed_step_footprints():
    """The default auto cap follows the largest completed-step pinned footprint without retaining low+high."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )

    hidden = 1024
    buffers_per_step = 2

    for seq_len in [512, 1024, 512]:
        slots = [_FakeSlot(seq_len, hidden=hidden) for _ in range(buffers_per_step)]
        buffers = [_pinned_cache_acquire(manager, slot) for slot in slots]
        for slot, buffer in zip(slots, buffers):
            _pinned_cache_release(manager, slot, None)
        _finalize_pinned_cache_step(manager)

    high_boundary_nbytes = 1024 * hidden * torch.empty((), dtype=torch.float16).element_size()
    expected_cache_bytes = buffers_per_step * _round_up(high_boundary_nbytes, 1 << 20)
    cache_tensors, cache_bytes = manager.pinned_cache.cache_size()

    assert manager.pinned_cache.auto_max_size_bytes == expected_cache_bytes
    assert cache_tensors == buffers_per_step
    assert cache_bytes == expected_cache_bytes
    assert manager.stats.pinned_evictions == buffers_per_step


@pytest.mark.integration
@requires_cuda
def test_pin_memory_eviction_releases_host_allocator_reserved_bytes():
    """Eviction must drop PyTorch host pinned reservations, not only manager cache counters."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=4 / 1024,
        pin_memory_bucket_size_mib=1,
    )

    torch.cuda.memory.reset_peak_host_memory_stats()
    torch.cuda.memory.reset_accumulated_host_memory_stats()
    baseline_host = _host_pin_reserved_bytes()
    observed_host = []

    for seq_len in [512, 1024, 2048, 4096, 8192]:
        slots = [_FakeSlot(seq_len, hidden=1024) for _ in range(4)]
        _run_fake_step(manager, slots)
        observed_host.append(_host_pin_reserved_bytes())

    _, manager_cache_bytes = manager.pinned_cache.cache_size()
    assert manager_cache_bytes <= 4 << 20
    assert manager.stats.pinned_evictions > 0

    growth = observed_host[-1] - observed_host[0]
    last_step_bytes = 4 * _round_up(8192 * 1024 * 2, 1 << 20)
    assert growth <= last_step_bytes, (
        f"host pinned reservations grew by {growth} bytes across increasing bucket sizes; "
        f"expected bounded near the final capped footprint ({last_step_bytes}), "
        f"manager_cache={manager_cache_bytes}, observed={observed_host}"
    )
    assert observed_host[-1] <= baseline_host + (4 << 20) + last_step_bytes


@pytest.mark.integration
@requires_cuda
def test_pin_memory_mixed_capacity_step_preserves_small_buffer_multiset():
    """One large buffer must not evict every smaller buffer required concurrently in the same step."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )

    one_mib_seq = 512
    eight_mib_seq = 4096
    hidden = 1024
    expected_step_bytes = (8 << 20) + 4 * (1 << 20)

    allocation_counts = []
    for _ in range(4):
        slots = [_FakeSlot(eight_mib_seq, hidden=hidden)] + [_FakeSlot(one_mib_seq, hidden=hidden) for _ in range(4)]
        _run_fake_step(manager, slots)
        allocation_counts.append(manager.stats.pinned_allocations)

    assert allocation_counts == [
        5,
        5,
        5,
        5,
    ], f"expected pinned allocations to stabilize after the first step, got {allocation_counts}"
    assert manager.pinned_cache.auto_max_size_bytes == expected_step_bytes
    cache_tensors, cache_bytes = manager.pinned_cache.cache_size()
    assert cache_tensors == 5
    assert cache_bytes == expected_step_bytes


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_cap_evicts_cached_buffers():
    """The pinned cache is bounded even when more buffers are returned than the configured cap allows."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=2 / 1024,
        pin_memory_bucket_size_mib=1,
    )

    slots = [_FakeSlot(512) for _ in range(4)]
    buffers = [_pinned_cache_acquire(manager, slot) for slot in slots]
    for slot, buffer in zip(slots, buffers):
        _pinned_cache_release(manager, slot, None)

    cache_tensors, cache_bytes = manager.pinned_cache.cache_size()
    assert cache_tensors == 2
    assert cache_bytes <= 2 << 20
    assert manager.stats.pinned_evictions == 2


@pytest.mark.integration
@requires_cuda
def test_reset_pending_recycles_stale_pinned_entries():
    """Saved tensors pruned by autograd should still return their pinned entries to the bounded cache."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=1,
        pin_memory_bucket_size_mib=1,
    )

    slots = [_FakeSlot(512) for _ in range(2)]
    for slot_id, slot in enumerate(slots):
        slot.cpu = _pinned_cache_acquire(manager, slot)
        manager._slots[slot_id] = slot
        manager._order.append(slot_id)

    manager.reset_pending()

    cache_tensors, _ = manager.pinned_cache.cache_size()
    assert manager.stats.stale_slots_dropped == 2
    assert manager.stats.stale_pinned_recycled == 2
    assert cache_tensors == 2

    reused = _pinned_cache_acquire(manager, _FakeSlot(480))
    assert reused.shape == (480, 256)
    assert manager.stats.pinned_reuses == 1


@pytest.mark.integration
@requires_cuda
def test_noncontiguous_saved_views_round_trip_and_reuse_exact_pinned_layout():
    """Real saved hooks should preserve transposed activations and reuse only the exact strided pool."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=True,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=1,
        pin_memory_bucket_size_mib=1,
    )
    data = torch.randn(512, 1024, device="cuda")

    def grad_with_optional_offload(use_offload: bool) -> torch.Tensor:
        x = data.detach().clone().requires_grad_(True)
        ctx = manager.hooks_ctx() if use_offload else nullcontext()
        with ctx:
            view = x.t()
            assert not view.is_contiguous()
            loss = (view * view).sum()
        loss.backward()
        return x.grad.detach()

    for _ in range(2):
        expected_grad = grad_with_optional_offload(use_offload=False)
        actual_grad = grad_with_optional_offload(use_offload=True)
        torch.testing.assert_close(actual_grad, expected_grad)

    torch.cuda.synchronize()
    assert manager.stats.offloaded_tensors > 0
    assert manager.stats.restored_tensors == manager.stats.offloaded_tensors
    assert manager.stats.pinned_reuses > 0
    cache_tensors, _ = manager.pinned_cache.cache_size()
    assert cache_tensors > 0


@pytest.mark.integration
@requires_cuda
def test_installed_offload_recycles_pruned_saved_tensors_into_auto_pinned_cache():
    """A later forward should reclaim pinned buffers from saved tensors that backward never unpacks."""
    model = _PrunedSavedTensorModule().cuda()
    manager = install_activation_offload(
        model,
        keep_last_n=0,
        use_streams=True,
        tensor_size_threshold=1,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )

    first = torch.randn(512, 1024, device="cuda", requires_grad=True)
    model(first).backward()
    stale_tensors, stale_bytes = manager._active_slot_size()
    assert stale_tensors > 0
    assert stale_bytes > 0

    second = torch.randn_like(first, requires_grad=True)
    model(second).backward()  # wrapper reset_pending() recycles the first step's stale slots before forward
    torch.cuda.synchronize()

    assert manager.stats.stale_slots_dropped == stale_tensors
    assert manager.stats.stale_pinned_recycled == stale_tensors
    assert manager.stats.stale_pinned_recycled_bytes == stale_bytes
    assert manager.pinned_cache.auto_max_size_bytes is not None
    assert manager.pinned_cache.auto_max_size_bytes >= stale_bytes
    assert manager.stats.pinned_reuses > 0


@pytest.mark.integration
@requires_cuda
def test_pin_memory_auto_cap_shrinks_after_sustained_smaller_steps():
    """Auto cap should drop after consecutive steps finish below the current high-water mark."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )

    _run_fake_step(manager, [_FakeSlot(4096, hidden=1024)])
    large_footprint = 8 << 20
    assert manager.pinned_cache.auto_max_size_bytes == large_footprint

    for _ in range(2):
        _run_fake_step(manager, [_FakeSlot(512, hidden=1024)])

    small_footprint = 1 << 20
    assert manager.pinned_cache.auto_max_size_bytes == small_footprint
    _, cache_bytes = manager.pinned_cache.cache_size()
    assert cache_bytes <= small_footprint


@pytest.mark.integration
@requires_cuda
def test_pin_memory_historical_multiset_preserves_buffers_across_alternating_steps():
    """Large-only steps must not evict small buffers still required by earlier mixed steps."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )

    def mixed_step() -> None:
        slots = [_FakeSlot(4096, hidden=1024)] + [_FakeSlot(512, hidden=1024) for _ in range(4)]
        _run_fake_step(manager, slots)

    def large_only_step() -> None:
        _run_fake_step(manager, [_FakeSlot(4096, hidden=1024), _FakeSlot(4096, hidden=1024)])

    allocation_counts = []
    for pattern in (mixed_step, large_only_step, mixed_step, large_only_step):
        pattern()
        allocation_counts.append(manager.stats.pinned_allocations)

    assert allocation_counts == [
        5,
        6,
        6,
        6,
    ], f"expected allocations to stabilize after the first large-only step, got {allocation_counts}"
    assert (
        manager.pinned_cache.retained_capacity_counts[PoolKey(is_contiguous=True, dtype=torch.float16)][1 << 20] == 4
    )


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cap_eviction_prefers_excess_capacity_tiers():
    """Cap pressure should evict above-retained tiers before needed capacity classes."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=4 / 1024,
        pin_memory_bucket_size_mib=1,
    )

    _run_fake_step(
        manager,
        [_FakeSlot(512, hidden=1024), _FakeSlot(512, hidden=1024), _FakeSlot(1024, hidden=1024)],
    )
    retained = manager.pinned_cache.retained_capacity_counts[PoolKey(is_contiguous=True, dtype=torch.float16)]
    assert retained[1 << 20] == 2
    assert retained[2 << 20] == 1

    extra_slot = _FakeSlot(2048, hidden=1024)
    _pinned_cache_acquire(manager, extra_slot)
    _pinned_cache_release(manager, extra_slot, None)

    cache_caps = sorted(entry.capacity_bytes for entry in manager.pinned_cache.fifo_entries)
    assert cache_caps.count(4 << 20) == 0
    assert cache_caps.count(1 << 20) == 2
    assert cache_caps.count(2 << 20) == 1


@pytest.mark.integration
@requires_cuda
def test_reconfigure_keep_last_n_preserves_learned_auto_cap():
    """Reconfiguring unrelated knobs must not reset a learned auto pinned-cache cap."""
    model = _tiny_checkpointed_model()
    manager = install_activation_offload(
        model,
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )
    slot = _FakeSlot(1024, hidden=1024)
    slot.cpu = _pinned_cache_acquire(manager, slot)
    _pinned_cache_release(manager, slot, None)
    _finalize_pinned_cache_step(manager)
    learned_cap = manager.pinned_cache.auto_max_size_bytes
    assert learned_cap is not None

    second = install_activation_offload(
        model,
        keep_last_n=3,
        use_streams=True,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )
    assert second is manager
    assert manager.pinned_cache.auto_max_size_bytes == learned_cap


@pytest.mark.integration
@requires_cuda
def test_format_stats_reports_host_pin_reserved_bytes():
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=1,
        pin_memory_bucket_size_mib=1,
    )
    _run_fake_step(manager, [_FakeSlot(512)])
    stats = manager.format_stats()
    assert "host-pin-reserved" in stats
    assert host_pin_reserved_bytes() > 0


@pytest.mark.integration
@requires_cuda
def test_exact_layout_cache_prunes_to_retained_multiset():
    """Exact-layout pinned pools should not grow without bound across distinct strides."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib="auto",
        pin_memory_bucket_size_mib=1,
    )

    layouts = [
        _FakeStridedSlot(shape=(8, 8), stride=(1, 8)),
        _FakeStridedSlot(shape=(8, 8), stride=(16, 1)),
        _FakeStridedSlot(shape=(8, 8), stride=(8, 1)),
    ]
    _run_fake_step(manager, layouts)
    assert manager.stats.pinned_allocations == 3

    for _ in range(3):
        _run_fake_step(manager, layouts)

    assert manager.stats.pinned_allocations <= 4
    assert manager.stats.pinned_reuses >= 6
    assert len(manager.pinned_cache.fifo_entries) == 3


@pytest.mark.integration
@requires_cuda
def test_broadcast_stride_zero_views_are_not_offloaded():
    """Broadcast (stride-0) saved views must stay resident; overlap detection uses PyTorch's internal check."""
    manager = ActivationOffloadManager()
    manager.configure(keep_last_n=0, use_streams=False, tensor_size_threshold=1)
    base = torch.arange(4, device="cuda", dtype=torch.float32)
    broadcast = base.as_strided((4, 4), (0, 1))

    offloaded, _ = manager.pack(broadcast)

    assert offloaded is False
    assert manager.stats.skipped_overlap == 1


@pytest.mark.integration
@requires_cuda
def test_nonzero_stride_overlapping_views_are_not_offloaded():
    """Sliding-window/as_strided views can overlap without stride 0 and should be left resident."""
    manager = ActivationOffloadManager()
    manager.configure(keep_last_n=0, use_streams=False, tensor_size_threshold=1)
    base = torch.arange(16, device="cuda", dtype=torch.float32)
    window = base.as_strided((4, 4), (1, 1))

    offloaded, _ = manager.pack(window)

    assert offloaded is False
    assert manager.stats.skipped_overlap == 1


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_does_not_reuse_different_noncontiguous_strides():
    """Same-shaped strided tensors need exact layout pools so a transpose buffer cannot satisfy a padded view."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=1,
        pin_memory_bucket_size_mib=1,
    )

    transposed = _FakeStridedSlot(shape=(8, 8), stride=(1, 8))
    padded = _FakeStridedSlot(shape=(8, 8), stride=(16, 1))

    transposed_buffer = _pinned_cache_acquire(manager, transposed)
    assert transposed_buffer.stride() == transposed.stride
    transposed_entry = transposed.pinned_entry
    _pinned_cache_release(manager, transposed, None)

    padded_buffer = _pinned_cache_acquire(manager, padded)
    assert padded_buffer.stride() == padded.stride
    assert padded.pinned_entry is not transposed_entry
    assert manager.stats.pinned_allocations == 2
    assert manager.stats.pinned_reuses == 0
    _pinned_cache_release(manager, padded, None)

    another_transposed = _FakeStridedSlot(shape=(8, 8), stride=(1, 8))
    reused = _pinned_cache_acquire(manager, another_transposed)
    assert reused.stride() == transposed.stride
    assert another_transposed.pinned_entry is transposed_entry
    assert manager.stats.pinned_reuses == 1


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_long_horizon_host_reserved_does_not_creep():
    """Pinned host reservations stay bounded across alternating activation sizes."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=4 / 1024,
        pin_memory_bucket_size_mib=1,
    )

    hidden = 1024
    pattern = [4096, 8192, 4096, 8192]
    observed_host = []
    for _ in range(80):
        for seq_len in pattern:
            _run_fake_step(manager, [_FakeSlot(seq_len, hidden=hidden) for _ in range(4)])
            observed_host.append(_host_pin_reserved_bytes())

    post_warmup = observed_host[40:]
    baseline = post_warmup[0]
    high_water = max(post_warmup)
    boundary_nbytes = max(pattern) * hidden * torch.empty((), dtype=torch.float16).element_size()
    tolerance_bytes = 3 * boundary_nbytes

    assert high_water - baseline <= tolerance_bytes, (
        f"host pinned reservations drifted by {high_water - baseline} bytes over {len(observed_host)} steps; "
        f"baseline={baseline}, high_water={high_water}, tolerance={tolerance_bytes}"
    )


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_long_horizon_scaled_pressure_does_not_creep():
    """Pinned host reservations stay bounded when activation boundaries approximate a larger model."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=16 / 1024,
        pin_memory_bucket_size_mib=1,
    )

    hidden = 4096
    pattern = [4096, 8192, 4096, 8192]
    observed_host = []
    for _ in range(60):
        for seq_len in pattern:
            _run_fake_step(manager, [_FakeSlot(seq_len, hidden=hidden) for _ in range(2)])
            observed_host.append(_host_pin_reserved_bytes())

    post_warmup = observed_host[32:]
    baseline = post_warmup[0]
    high_water = max(post_warmup)
    boundary_nbytes = max(pattern) * hidden * torch.empty((), dtype=torch.float16).element_size()
    tolerance_bytes = 2 * boundary_nbytes

    assert high_water - baseline <= tolerance_bytes, (
        f"scaled-pressure host reservations drifted by {high_water - baseline} bytes over "
        f"{len(observed_host)} steps; baseline={baseline}, high_water={high_water}, tolerance={tolerance_bytes}"
    )


@pytest.mark.integration
@requires_cuda
def test_pin_memory_cache_long_horizon_plateaus_at_fixed_capacity():
    """Pinned host reservations plateau when every step uses one fixed capacity class."""
    manager = ActivationOffloadManager()
    manager.configure(
        keep_last_n=0,
        use_streams=False,
        tensor_size_threshold=1,
        pin_memory_max_size_gib=4 / 1024,
        pin_memory_bucket_size_mib=1,
    )

    hidden = 1024
    seq_len = 8192
    observed_host = []
    for _ in range(200):
        _run_fake_step(manager, [_FakeSlot(seq_len, hidden=hidden) for _ in range(4)])
        observed_host.append(_host_pin_reserved_bytes())

    post_warmup = observed_host[20:]
    baseline = post_warmup[0]
    high_water = max(post_warmup)
    boundary_nbytes = seq_len * hidden * torch.empty((), dtype=torch.float16).element_size()
    tolerance_bytes = 2 * boundary_nbytes

    assert high_water <= baseline + tolerance_bytes, (
        f"host pinned reservations did not plateau over a fixed-size run; baseline={baseline}, "
        f"high_water={high_water}, tolerance={tolerance_bytes}"
    )
