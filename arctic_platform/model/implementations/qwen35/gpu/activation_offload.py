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

import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

_DEFAULT_TENSOR_SIZE_THRESHOLD = 1 << 20  # 1 MiB


@dataclass
class _OffloadStats:
    offloaded_tensors: int = 0
    offloaded_bytes: int = 0
    restored_tensors: int = 0
    restored_bytes: int = 0
    skipped_overlap: int = 0  # non-contiguous but overlapping/broadcast (stride 0)
    passed_small: int = 0  # eligible device tensor below tensor_size_threshold

    def reset(self) -> None:
        self.offloaded_tensors = 0
        self.offloaded_bytes = 0
        self.restored_tensors = 0
        self.restored_bytes = 0
        self.skipped_overlap = 0
        self.passed_small = 0


class _Slot:
    __slots__ = ("slot_id", "device", "shape", "stride", "dtype", "nbytes", "gpu", "cpu",
                 "d2h_event", "h2d_event", "offloaded")

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
        self._offload_stream: Optional[torch.cuda.Stream] = None
        self._reload_stream: Optional[torch.cuda.Stream] = None
        self._slots: Dict[int, _Slot] = {}
        self._order: List[int] = []
        self._next_id = 0
        # Pool of reusable pinned staging buffers, keyed by exact layout so strided tensors round-trip safely.
        self._pinned_free: Dict[
            Tuple[tuple, tuple, torch.dtype], List[Tuple[torch.Tensor, Optional[torch.cuda.Event]]]
        ] = {}
        self.stats = _OffloadStats()

    def configure(self, keep_last_n: int, use_streams: bool, tensor_size_threshold: Optional[int] = None) -> None:
        self.enabled = True
        self.keep_last_n = max(0, int(keep_last_n))
        self.use_streams = bool(use_streams)
        if tensor_size_threshold is not None:
            self.tensor_size_threshold = int(tensor_size_threshold)

    def reset(self) -> None:
        self._slots.clear()
        self._order.clear()
        self._pinned_free.clear()

    def reset_pending(self) -> None:
        """Drop slots not drained by the previous backward (keep the pinned pool).

        A saved tensor can be packed but pruned from the backward graph (never unpacked), leaving stale entries
        that would corrupt the next forward's LIFO order. Called at the start of every forward.
        """
        self._slots.clear()
        self._order.clear()

    def hooks_ctx(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def _ensure_streams(self, device: torch.device) -> None:
        if not self.use_streams or self._offload_stream is not None:
            return
        self._offload_stream = torch.cuda.Stream(device=device)
        self._reload_stream = torch.cuda.Stream(device=device)

    def _get_pinned(self, slot: _Slot) -> torch.Tensor:
        key = (tuple(slot.shape), slot.stride, slot.dtype)
        pool = self._pinned_free.get(key)
        if pool:
            cpu, event = pool.pop()
            if event is not None and self.use_streams:
                self._offload_stream.wait_event(event)  # prior H2D read of this buffer must finish
            return cpu
        return torch.empty_strided(slot.shape, slot.stride, dtype=slot.dtype, device="cpu", pin_memory=True)

    def _put_pinned(self, slot: _Slot, cpu: torch.Tensor, h2d_event: Optional[torch.cuda.Event]) -> None:
        key = (tuple(slot.shape), slot.stride, slot.dtype)
        self._pinned_free.setdefault(key, []).append((cpu, h2d_event))

    def _eligible(self, tensor: torch.Tensor) -> bool:
        if not (self.enabled and tensor.device.type == "cuda") or isinstance(tensor, torch.nn.Parameter):
            return False
        if tensor.numel() * tensor.element_size() < self.tensor_size_threshold:
            self.stats.passed_small += 1
            return False
        # Overlapping/broadcast views (a 0 in stride) alias storage; copying them into a strided buffer would
        # double-write, so leave them resident.
        if not tensor.is_contiguous() and 0 in tensor.stride():
            self.stats.skipped_overlap += 1
            return False
        return True

    def pack(self, tensor: torch.Tensor):
        if not self._eligible(tensor):
            return (False, tensor)
        self._ensure_streams(tensor.device)
        slot_id = self._next_id
        self._next_id += 1
        self._slots[slot_id] = _Slot(slot_id, tensor)
        self._order.append(slot_id)
        # Offload the slot that just fell out of the keep-last-N window; the most-recent N stay resident.
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
            self._offload_stream.wait_stream(torch.cuda.current_stream(gpu.device))
            with torch.cuda.stream(self._offload_stream):
                cpu.copy_(gpu, non_blocking=True)
                slot.d2h_event = torch.cuda.Event()
                slot.d2h_event.record(self._offload_stream)
            gpu.record_stream(self._offload_stream)  # keep the allocator from recycling gpu mid-copy
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
        if self.use_streams:
            # Allocate from the default-stream pool, then copy on the reload stream. The block may be recycled
            # memory with a pending default-stream op, so the reload stream must wait for it (and for this
            # slot's own D2H) before writing.
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
        try:
            position = self._order.index(slot_id)
        except ValueError:
            return
        if position > 0:
            self._start_reload(self._slots[self._order[position - 1]])

    def _pull(self, slot_id: int) -> torch.Tensor:
        slot = self._slots[slot_id]
        if slot.offloaded:
            self._start_reload(slot)  # idempotent if already prefetched
        self._prefetch_prev(slot_id)  # overlap the next layer's reload
        if slot.offloaded:
            if self.use_streams and slot.h2d_event is not None:
                torch.cuda.current_stream(slot.device).wait_event(slot.h2d_event)
            self._put_pinned(slot, slot.cpu, slot.h2d_event)
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
        try:
            self._order.remove(slot_id)
        except ValueError:
            pass
        return gpu

    def reset_stats(self) -> None:
        self.stats.reset()

    def format_stats(self) -> str:
        stats = self.stats
        gib = 1 << 30
        return (
            f"act-offload stats: offloaded {stats.offloaded_tensors} tensors "
            f"({stats.offloaded_bytes / gib:.2f} GiB), restored {stats.restored_tensors} "
            f"({stats.restored_bytes / gib:.2f} GiB), skipped-overlap {stats.skipped_overlap}, "
            f"passed-small {stats.passed_small}"
        )


_INSTALLED_ATTR = "_activation_offload_installed"
_MANAGER_ATTR = "_activation_offload_manager"


def install_activation_offload(
    model: torch.nn.Module,
    keep_last_n: int = 1,
    use_streams: bool = True,
    tensor_size_threshold: Optional[int] = None,
) -> ActivationOffloadManager:
    """Wrap ``model.forward`` so saved activations stream to CPU, and return the model's offload manager.

    Idempotent per module: the manager is created once and attached as ``model._activation_offload_manager``;
    repeat calls only re-configure it. The model must use non-reentrant activation checkpointing.
    """
    manager: Optional[ActivationOffloadManager] = getattr(model, _MANAGER_ATTR, None)
    if manager is None:
        manager = ActivationOffloadManager()
        setattr(model, _MANAGER_ATTR, manager)
    manager.configure(keep_last_n=keep_last_n, use_streams=use_streams, tensor_size_threshold=tensor_size_threshold)
    if getattr(model, _INSTALLED_ATTR, False):
        return manager
    original_forward = model.forward

    @functools.wraps(original_forward)
    def forward_with_offload(*args, **kwargs):
        if not manager.enabled or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        manager.reset_pending()
        with manager.hooks_ctx():
            return original_forward(*args, **kwargs)

    model.forward = forward_with_offload
    setattr(model, _INSTALLED_ATTR, True)
    return manager


def activation_offload_stats(model: torch.nn.Module) -> Optional[str]:
    manager: Optional[ActivationOffloadManager] = getattr(model, _MANAGER_ATTR, None)
    return manager.format_stats() if manager is not None else None
