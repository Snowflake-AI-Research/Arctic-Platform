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
"""Activation CPU offload (model-agnostic).

Streams the activations autograd saves for backward -- the per-layer checkpoint boundaries under non-reentrant
activation checkpointing -- to pinned CPU during forward and back to GPU during backward. At long sequence
lengths those boundaries (num_layers * seq_len * hidden) dominate GPU memory, so moving them off device is what
lets longer sequences fit.

Implemented with PyTorch saved-tensor pack/unpack hooks, not a reentrant CheckpointFunction: under reentrant
checkpointing the block outputs stay retained by the autograd graph, so offloading the inputs frees nothing
(measured: identical peak); the pack/unpack hooks intercept exactly the tensors kept for backward. Copies run
on dedicated offload/reload streams to overlap compute, the last N boundaries stay resident (needed first in
backward), and each pull prefetches the previous slot (backward consumes LIFO).

Precondition: the model must use non-reentrant activation checkpointing (checkpoint_wrapper defaults to it).
Use ``install_activation_offload(model, ...)``; for ad-hoc use wrap a forward with ``manager.hooks_ctx()``.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from dataclasses import dataclass
from dataclasses import fields
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch

from arctic_platform.model.config import ActivationOffloadConfig
from arctic_platform.model.config import PinMemoryMaxSize
from arctic_platform.model.implementations.debug.activation_offload import maybe_log_activation_offload_slot
from arctic_platform.model.implementations.gpu.pinned_staging_cache import PoolKey
from arctic_platform.model.implementations.gpu.pinned_staging_cache import _PinnedCacheStats
from arctic_platform.model.implementations.gpu.pinned_staging_cache import _PinnedEntry
from arctic_platform.model.implementations.gpu.pinned_staging_cache import _PinnedStagingCache

logger = logging.getLogger(__name__)

_DEFAULT_TENSOR_SIZE_THRESHOLD = 1 << 20  # 1 MiB
_DEFAULT_PIN_MEMORY_BUCKET_SIZE_MIB = 64
_GIB = 1 << 30
_MIB = 1 << 20


# The pinned pool the caching host allocator holds, in torch's own naming. ``allocated_bytes`` counts what the
# allocator has taken from the OS, which is the pool; ``active_bytes`` counts only what tensors are using right
# now, and would under-report a pool that is holding freed blocks for reuse. The second name is what older torch
# published for the same quantity.
_HOST_PIN_POOL_KEYS = ("allocated_bytes.current", "reserved_bytes.current")


def host_pin_reserved_bytes() -> int:
    """Pinned host bytes retained by PyTorch's host caching allocator (0 when unavailable)."""
    if not torch.cuda.is_available():
        return 0
    try:
        torch.cuda.synchronize()
        stats = torch.cuda.memory.host_memory_stats()
    except AttributeError as exc:
        logger.debug("host_memory_stats unavailable (%s); reporting host-pin-reserved as 0", exc)
        return 0
    except RuntimeError as exc:
        logger.warning("host_pin_reserved_bytes failed (%s); reporting host-pin-reserved as 0", exc)
        return 0
    for key in _HOST_PIN_POOL_KEYS:
        if key in stats:
            return int(stats[key])
    logger.warning(
        "host_memory_stats has none of %s (got %s); reporting host-pin-reserved as 0",
        _HOST_PIN_POOL_KEYS,
        sorted(stats),
    )
    return 0


def _fmt_gib(num_bytes: int) -> str:
    return f"{num_bytes / _GIB:.2f} GiB"


def _tensor_has_storage_overlap(tensor: torch.Tensor) -> bool:
    """True when non-contiguous elements alias the same storage (unsafe to copy into a strided CPU buffer)."""
    if tensor.is_contiguous():
        return False
    debug_overlap = getattr(torch, "_debug_has_internal_overlap", None)
    if debug_overlap is not None:
        # 0 = no overlap; 1 = yes (e.g. broadcast stride-0); 2 = too hard (conservative skip).
        return debug_overlap(tensor) != 0
    if 0 in tensor.stride():
        return True
    size = tensor.size()
    stride = tensor.stride()
    min_offset = tensor.storage_offset()
    max_offset = min_offset
    for dim in range(len(size)):
        if size[dim] > 0:
            max_offset += (size[dim] - 1) * stride[dim]
    return (max_offset - min_offset + 1) < tensor.numel()


@dataclass
class _OffloadStats:
    offloaded_tensors: int = 0
    offloaded_bytes: int = 0
    restored_tensors: int = 0
    restored_bytes: int = 0
    pinned_allocations: int = 0
    pinned_allocated_bytes: int = 0
    pinned_reuses: int = 0
    pinned_reused_bytes: int = 0
    pinned_evictions: int = 0
    pinned_evicted_bytes: int = 0
    pageable_allocations: int = 0
    pageable_allocated_bytes: int = 0
    stale_slots_dropped: int = 0
    stale_slot_bytes: int = 0
    stale_pinned_recycled: int = 0
    stale_pinned_recycled_bytes: int = 0
    skipped_overlap: int = 0  # overlapping/broadcast saved views left resident
    passed_small: int = 0  # eligible device tensor below tensor_size_threshold

    def reset(self) -> None:
        for field in fields(self):
            setattr(self, field.name, 0)


class _Slot:
    __slots__ = (
        "slot_id",
        "device",
        "shape",
        "stride",
        "dtype",
        "nbytes",
        "gpu",
        "cpu",
        "d2h_event",
        "h2d_event",
        "offloaded",
        "is_contiguous",
        "pinned_entry",
    )

    def __init__(self, slot_id: int, tensor: torch.Tensor):
        self.slot_id = slot_id
        self.device = tensor.device
        self.shape = tensor.shape
        self.stride = tensor.stride()
        self.dtype = tensor.dtype
        self.nbytes = tensor.numel() * tensor.element_size()
        self.gpu: Optional[torch.Tensor] = tensor
        self.cpu: Optional[torch.Tensor] = None
        self.d2h_event: Optional[torch.cuda.Event] = None
        self.h2d_event: Optional[torch.cuda.Event] = None
        self.offloaded = False
        self.is_contiguous = tensor.is_contiguous()
        self.pinned_entry: Optional[_PinnedEntry] = None


class ActivationOffloadManager:
    """Streams saved activations to/from CPU via pack/unpack hooks. One instance per model.

    Slots are packed in forward order and unpacked LIFO during backward; state self-cleans as slots are
    unpacked. ``reset`` clears leftovers after an error.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.use_streams = True
        self.keep_last_n = 1
        self.tensor_size_threshold = _DEFAULT_TENSOR_SIZE_THRESHOLD
        self.pin_memory_enabled = True
        self.pin_memory_bucket_size_bytes = _DEFAULT_PIN_MEMORY_BUCKET_SIZE_MIB * _MIB
        self.pin_memory_max_size_gib: PinMemoryMaxSize = "auto"
        self._pin_memory_hard_max_size_bytes: Optional[int] = None
        self._offload_stream: Optional[torch.cuda.Stream] = None
        self._reload_stream: Optional[torch.cuda.Stream] = None
        self._slots: Dict[int, _Slot] = {}
        self._order: List[int] = []
        self._order_index: Dict[int, int] = {}
        self._next_id = 0
        self._active_depth = 0
        self._pinned_cache_stats = _PinnedCacheStats()
        self.pinned_cache = _PinnedStagingCache(self._pinned_cache_stats)
        self.stats = _OffloadStats()

    def configure(
        self,
        keep_last_n: int = 1,
        use_streams: bool = True,
        tensor_size_threshold: Optional[int] = None,
        pin_memory_enabled: bool = True,
        pin_memory_max_size_gib: PinMemoryMaxSize = "auto",
        pin_memory_bucket_size_mib: int = _DEFAULT_PIN_MEMORY_BUCKET_SIZE_MIB,
        config: Optional[ActivationOffloadConfig] = None,
    ) -> None:
        resolved = config or ActivationOffloadConfig(
            keep_last_n=keep_last_n,
            use_streams=use_streams,
            tensor_size_threshold=tensor_size_threshold,
            pin_memory_enabled=pin_memory_enabled,
            pin_memory_max_size_gib=pin_memory_max_size_gib,
            pin_memory_bucket_size_mib=pin_memory_bucket_size_mib,
        )
        self._apply_config(resolved)

    def _apply_config(self, config: ActivationOffloadConfig) -> None:
        self.enabled = True
        self.keep_last_n = config.keep_last_n
        self.use_streams = config.use_streams
        self.pin_memory_enabled = config.pin_memory_enabled
        self.pin_memory_bucket_size_bytes = config.pin_memory_bucket_size_bytes
        self.pin_memory_max_size_gib = config.pin_memory_max_size_gib
        self._pin_memory_hard_max_size_bytes = config.pin_memory_hard_max_size_bytes
        self.tensor_size_threshold = (
            int(config.tensor_size_threshold)
            if config.tensor_size_threshold is not None
            else _DEFAULT_TENSOR_SIZE_THRESHOLD
        )
        self.pinned_cache.configure(
            enabled=config.pin_memory_enabled,
            bucket_size_bytes=config.pin_memory_bucket_size_bytes,
            hard_max_size_bytes=config.pin_memory_hard_max_size_bytes,
        )
        self.sync_pinned_stats()

    def reset(self) -> None:
        self._slots.clear()
        self._order.clear()
        self._order_index.clear()
        self.pinned_cache.clear()
        self.pinned_cache.reset_learned_state()
        self.sync_pinned_stats()

    def sync_pinned_stats(self) -> None:
        """Copy pinned-cache counters into ``manager.stats`` for logging/tests."""
        self._sync_pinned_stats_from_cache()

    def reset_pending(self) -> None:
        """Drop slots not drained by the previous backward (keep the pinned cache).

        A saved tensor can be packed but pruned from the backward graph (never unpacked), leaving stale entries
        that would corrupt the next forward's LIFO order. Called at the start of every forward.
        """
        for slot in self._slots.values():
            self.stats.stale_slots_dropped += 1
            self.stats.stale_slot_bytes += slot.nbytes
            if slot.cpu is not None and slot.pinned_entry is not None:
                self.stats.stale_pinned_recycled += 1
                self.stats.stale_pinned_recycled_bytes += slot.nbytes
                self._cache_pinned(slot, slot.d2h_event)
                slot.cpu = None
        self._slots.clear()
        self._order.clear()
        self._order_index.clear()
        self.pinned_cache.finalize_step()
        self.sync_pinned_stats()

    def hooks_ctx(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    @contextlib.contextmanager
    def step_hooks(self):
        """Enter the pack/unpack hooks for one forward, clearing stale slots only on the outermost entry.

        More than one module in a single stack may be wrapped -- a causal-LM and its own backbone -- so this can
        be entered twice per forward. ``reset_pending`` drops every live slot, which would discard the outer
        scope's packs, so only the outermost entry may call it. The nested contexts share this manager, hence one
        slot book and one packing order, which is what backward's LIFO unpacking requires.
        """
        if self._active_depth == 0:
            self.reset_pending()
        self._active_depth += 1
        try:
            with self.hooks_ctx():
                yield
        finally:
            self._active_depth -= 1

    def _ensure_streams(self, device: torch.device) -> None:
        if not self.use_streams or self._offload_stream is not None:
            return
        self._offload_stream = torch.cuda.Stream(device=device)
        self._reload_stream = torch.cuda.Stream(device=device)

    def _append_order(self, slot_id: int) -> None:
        self._order_index[slot_id] = len(self._order)
        self._order.append(slot_id)

    def _remove_order(self, slot_id: int) -> None:
        index = self._order_index.pop(slot_id, None)
        if index is None:
            return
        last_id = self._order[-1]
        self._order[index] = last_id
        self._order_index[last_id] = index
        self._order.pop()

    def _get_pinned(self, slot: _Slot) -> torch.Tensor:
        if not self.pin_memory_enabled:
            self.stats.pageable_allocations += 1
            self.stats.pageable_allocated_bytes += slot.nbytes
            return torch.empty_strided(slot.shape, slot.stride, dtype=slot.dtype, device="cpu")
        cpu = self.pinned_cache.acquire(
            slot,
            offload_stream=self._offload_stream,
            use_streams=self.use_streams,
        )
        self.sync_pinned_stats()
        return cpu

    def _cache_pinned(self, slot: _Slot, event: Optional[torch.cuda.Event]) -> None:
        if not self.pin_memory_enabled:
            return
        self.pinned_cache.release(slot, event)
        self.sync_pinned_stats()

    def _sync_pinned_stats_from_cache(self) -> None:
        cache_stats = self._pinned_cache_stats
        self.stats.pinned_allocations = cache_stats.pinned_allocations
        self.stats.pinned_allocated_bytes = cache_stats.pinned_allocated_bytes
        self.stats.pinned_reuses = cache_stats.pinned_reuses
        self.stats.pinned_reused_bytes = cache_stats.pinned_reused_bytes
        self.stats.pinned_evictions = cache_stats.pinned_evictions
        self.stats.pinned_evicted_bytes = cache_stats.pinned_evicted_bytes

    def _eligible(self, tensor: torch.Tensor) -> bool:
        if not (self.enabled and tensor.device.type == "cuda") or isinstance(tensor, torch.nn.Parameter):
            return False
        if tensor.numel() * tensor.element_size() < self.tensor_size_threshold:
            self.stats.passed_small += 1
            return False
        if not tensor.is_contiguous() and _tensor_has_storage_overlap(tensor):
            self.stats.skipped_overlap += 1
            return False
        return True

    def pack(self, tensor: torch.Tensor):
        if not self._eligible(tensor):
            maybe_log_activation_offload_slot(tensor, staged=False)
            return (False, tensor)
        self._ensure_streams(tensor.device)
        slot_id = self._next_id
        self._next_id += 1
        maybe_log_activation_offload_slot(tensor, staged=True, slot_id=slot_id)
        self._slots[slot_id] = _Slot(slot_id, tensor)
        self._append_order(slot_id)
        offload_index = len(self._order) - 1 - self.keep_last_n
        if offload_index >= 0:
            self._start_offload(self._slots[self._order[offload_index]])
        return (True, slot_id)

    def unpack(self, payload):
        offloaded, value = payload
        if not offloaded:
            return value
        return self._pull(value)

    def _start_offload(self, slot: _Slot) -> None:
        if slot.offloaded or slot.gpu is None:
            return
        gpu = slot.gpu
        cpu = self._get_pinned(slot)
        if self.use_streams:
            assert self._offload_stream is not None
            self._offload_stream.wait_stream(torch.cuda.current_stream(gpu.device))
            with torch.cuda.stream(self._offload_stream):
                cpu.copy_(gpu, non_blocking=True)
                slot.d2h_event = torch.cuda.Event()
                slot.d2h_event.record(self._offload_stream)
            gpu.record_stream(self._offload_stream)
        else:
            cpu.copy_(gpu)
            slot.d2h_event = None
        slot.cpu = cpu
        slot.gpu = None
        slot.offloaded = True
        self.stats.offloaded_tensors += 1
        self.stats.offloaded_bytes += slot.nbytes

    def _start_reload(self, slot: _Slot) -> None:
        if not slot.offloaded or slot.gpu is not None:
            return
        assert slot.cpu is not None
        if self.use_streams:
            assert self._reload_stream is not None
            gpu = torch.empty_strided(slot.shape, slot.stride, dtype=slot.dtype, device=slot.device)
            self._reload_stream.wait_stream(torch.cuda.current_stream(slot.device))
            if slot.d2h_event is not None:
                self._reload_stream.wait_event(slot.d2h_event)
            with torch.cuda.stream(self._reload_stream):
                gpu.record_stream(self._reload_stream)
                gpu.copy_(slot.cpu, non_blocking=True)
                slot.h2d_event = torch.cuda.Event()
                slot.h2d_event.record(self._reload_stream)
            slot.gpu = gpu
        else:
            slot.gpu = slot.cpu.to(slot.device)
            slot.h2d_event = None

    def _prefetch_prev(self, slot_id: int) -> None:
        index = self._order_index.get(slot_id)
        if index is None or index <= 0:
            return
        self._start_reload(self._slots[self._order[index - 1]])

    def _pull(self, slot_id: int) -> torch.Tensor:
        slot = self._slots[slot_id]
        if slot.offloaded:
            self._start_reload(slot)
        self._prefetch_prev(slot_id)
        if slot.offloaded:
            if self.use_streams and slot.h2d_event is not None:
                torch.cuda.current_stream(slot.device).wait_event(slot.h2d_event)
            self._cache_pinned(slot, slot.h2d_event)
            slot.cpu = None
            gpu = slot.gpu
            if self.use_streams and gpu is not None:
                gpu.record_stream(torch.cuda.current_stream(slot.device))
            self.stats.restored_tensors += 1
            self.stats.restored_bytes += slot.nbytes
        else:
            gpu = slot.gpu
        slot.gpu = None
        self._slots.pop(slot_id, None)
        self._remove_order(slot_id)
        if not self._order:
            self.pinned_cache.finalize_step()
            self.sync_pinned_stats()
        return gpu

    def reset_stats(self) -> None:
        self.stats.reset()
        self._pinned_cache_stats.reset()
        self.sync_pinned_stats()

    def _active_slot_size(self) -> Tuple[int, int]:
        tensors = len(self._slots)
        bytes_total = sum(slot.nbytes for slot in self._slots.values())
        return tensors, bytes_total

    def format_stats(self) -> str:
        stats = self.stats
        cache_tensors, cache_bytes = self.pinned_cache.cache_size()
        active_tensors, active_bytes = self._active_slot_size()
        auto_max = self.pinned_cache.auto_max_size_bytes
        parts = {
            "offloaded": f"{stats.offloaded_tensors} tensors ({_fmt_gib(stats.offloaded_bytes)})",
            "restored": f"{stats.restored_tensors} ({_fmt_gib(stats.restored_bytes)})",
            "skipped-overlap": str(stats.skipped_overlap),
            "passed-small": str(stats.passed_small),
            "pinned-alloc": f"{stats.pinned_allocations} ({_fmt_gib(stats.pinned_allocated_bytes)})",
            "pinned-reuse": f"{stats.pinned_reuses} ({_fmt_gib(stats.pinned_reused_bytes)})",
            "pinned-cache": f"{cache_tensors} ({_fmt_gib(cache_bytes)})",
            "host-pin-reserved": _fmt_gib(host_pin_reserved_bytes()),
            "active-slots": f"{active_tensors} ({_fmt_gib(active_bytes)})",
            "pinned-evict": f"{stats.pinned_evictions} ({_fmt_gib(stats.pinned_evicted_bytes)})",
            "pageable-alloc": f"{stats.pageable_allocations} ({_fmt_gib(stats.pageable_allocated_bytes)})",
            "pin-memory-enabled": str(self.pin_memory_enabled),
            "pin-memory-max": (
                _fmt_gib(self._pin_memory_hard_max_size_bytes)
                if self._pin_memory_hard_max_size_bytes is not None
                else "auto"
            ),
            "pin-memory-auto-max": _fmt_gib(auto_max) if auto_max is not None else "unset",
            "pin-memory-bucket": f"{self.pin_memory_bucket_size_bytes / _MIB:.0f} MiB",
            "stale-dropped": f"{stats.stale_slots_dropped} ({_fmt_gib(stats.stale_slot_bytes)})",
            "stale-pinned-recycled": f"{stats.stale_pinned_recycled} ({_fmt_gib(stats.stale_pinned_recycled_bytes)})",
        }
        body = ", ".join(f"{key} {value}" for key, value in parts.items())
        return f"act-offload stats: {body}"


_INSTALLED_ATTR = "_activation_offload_installed"
_MANAGER_ATTR = "_activation_offload_manager"


def install_activation_offload(
    model: torch.nn.Module,
    keep_last_n: int = 1,
    use_streams: bool = True,
    tensor_size_threshold: Optional[int] = None,
    pin_memory_enabled: bool = True,
    pin_memory_max_size_gib: PinMemoryMaxSize = "auto",
    pin_memory_bucket_size_mib: int = _DEFAULT_PIN_MEMORY_BUCKET_SIZE_MIB,
    config: Optional[ActivationOffloadConfig] = None,
    manager: Optional[ActivationOffloadManager] = None,
) -> ActivationOffloadManager:
    """Wrap ``model.forward`` so saved activations stream to CPU, and return the model's offload manager.

    Idempotent per module: the manager is created once and attached as ``model._activation_offload_manager``;
    repeat calls only re-configure it. The model must use non-reentrant activation checkpointing.

    Pass ``manager`` to wrap a second module with an existing manager. Two modules in one stack must share a
    manager: separate managers keep separate slot books, which interleave and break backward's LIFO unpacking.
    """
    resolved = config or ActivationOffloadConfig(
        keep_last_n=keep_last_n,
        use_streams=use_streams,
        tensor_size_threshold=tensor_size_threshold,
        pin_memory_enabled=pin_memory_enabled,
        pin_memory_max_size_gib=pin_memory_max_size_gib,
        pin_memory_bucket_size_mib=pin_memory_bucket_size_mib,
    )
    attached_manager: Optional[ActivationOffloadManager] = getattr(model, _MANAGER_ATTR, None)
    if manager is not None and attached_manager is not None and manager is not attached_manager:
        raise ValueError(
            "cannot reinstall activation offload with a different activation-offload manager; "
            "the forward wrapper still uses the manager installed on its first call"
        )
    if manager is None:
        manager = attached_manager or ActivationOffloadManager()
    setattr(model, _MANAGER_ATTR, manager)
    manager.configure(config=resolved)
    if getattr(model, _INSTALLED_ATTR, False):
        return manager
    original_forward = model.forward
    installed_manager = manager

    @functools.wraps(original_forward)
    def forward_with_offload(*args, **kwargs):
        if not installed_manager.enabled or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        with installed_manager.step_hooks():
            return original_forward(*args, **kwargs)

    model.forward = forward_with_offload
    setattr(model, _INSTALLED_ATTR, True)
    return manager


def activation_offload_stats(model: torch.nn.Module) -> Optional[str]:
    manager: Optional[ActivationOffloadManager] = getattr(model, _MANAGER_ATTR, None)
    return manager.format_stats() if manager is not None else None


__all__ = [
    "ActivationOffloadConfig",
    "ActivationOffloadManager",
    "PoolKey",
    "PinMemoryMaxSize",
    "activation_offload_stats",
    "host_pin_reserved_bytes",
    "install_activation_offload",
]
