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
"""Keep the frozen routed-expert weights in pinned host memory; copy each layer's in around every call of its experts.

GLM-5.3's block-FP8 routed experts are 91 GiB per GPU at EP8, which leaves too little of an H200 for a 256K-token
sequence. With LoRA they are frozen and every layer reads them twice per step: in its forward and in its
activation-checkpoint recompute (the backward reuses the recompute's copy). So each rank keeps its experts' codes in
one host buffer, registered with CUDA, and a layer's codes (1.2 GiB per GPU for GLM-5.3) are copied in on a side
stream for each call, into one of two persistent GPU buffers by the layer's parity: a layer's codes always sit at the
same addresses (which UltraEP's spare slots, ``moe/ultraep_frozen.py``, read in place), and no call allocates. The
next layer in call order (forward: the next one; backward: the previous one) is prefetched while the current one runs,
so the copy overlaps compute.

The offload happens at the first call, after PEFT and DeepSpeed have wrapped the model; the expert parameters keep
their objects with empty data, and during a call each is replaced by a temporary parameter viewing the copied bytes.
The temporaries are tagged ``_ap_transient``: the expert GEMMs then save them with ``save_for_backward``
(``fp8.stash_frozen``), so the checkpointed forward drops them after the layer and only the recompute's copy lives
until that layer's backward. Only the code stacks ``w1`` / ``w2`` / ``w3`` move; their FP32 scales (one per 128 x
128 block: a 4,096th of the codes' bytes) stay on the GPU. A model with offloaded experts cannot export them.

A layer whose experts are not recomputed (``ac_config.layers`` leaves its block out of checkpointing) saves its
expert weights for backward, so a copy of its codes would live until then and hold a buffer the next layers need: such
layers keep their codes on the GPU and are not managed.
"""

from __future__ import annotations

import os
import weakref

import torch
from torch import nn

OFFLOAD_FROZEN_EXPERTS_ENV = "ARL_OFFLOAD_FROZEN_EXPERTS"
_ALIGN = 256
_NAMES = ("w1", "w2", "w3")


_configured: bool | None = None  # loader option moe.frozen_expert_offload (moe/runtime_options.py); None: the env


def configure_from_options(value: bool | None) -> None:
    global _configured
    _configured = value


def frozen_expert_offload_requested() -> bool:
    if _configured is not None:
        return _configured
    return os.environ.get(OFFLOAD_FROZEN_EXPERTS_ENV, "0") not in ("", "0", "false", "False")


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _inner_layer(layer: nn.Module) -> nn.Module:
    return getattr(layer, "_checkpoint_wrapped_module", layer)


def _slot(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Where an expert stack ``name`` lives right now: the experts' own parameter, or the ``original`` of its PEFT
    parametrization (PEFT's ParamWrapper registers one around each call on a BF16 base, so look it up per call)."""
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is not None and name in parametrizations:
        return parametrizations[name], "original"
    return module, name


def _expert_code_params(layer: nn.Module):
    """``(experts module, name, param)`` of every frozen routed-expert code stack in ``layer``."""
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    for module in layer.modules():
        if not isinstance(module, GroupedExperts):
            continue
        for name in _NAMES:
            owner, key = _slot(module, name)
            param = owner._parameters.get(key)
            if param is None or param.requires_grad or param.numel() == 0:
                continue
            yield module, name, param


class FrozenExpertOffload:
    """The host copies of every managed layer's expert codes, and the per-call copies in."""

    def __init__(self, layers: list[nn.Module]) -> None:
        self.layers = layers
        # Per layer: [(experts module, name, param, shape, dtype, offset in the layer's bytes, nbytes)], its offset
        # in the host buffer, and its size.
        self.entries: list[list[tuple]] = []
        self.host_offsets: list[int] = []
        self.layer_bytes: list[int] = []
        self.host: torch.Tensor | None = None
        self.registered = False
        self.stream: torch.cuda.Stream | None = None
        # Which layer's codes each buffer holds (or is being copied), and the copy's event.
        self.resident: list[int] = [-1, -1]
        self.events: list[torch.cuda.Event | None] = [None, None]
        # Two persistent GPU buffers: layer i always copies into buffers[i % 2], so its codes keep their addresses
        # from call to call (UltraEP's pointer pools read them in place) and no call allocates. Each buffer
        # remembers the temporaries of the layer that last used it, to refuse an overwrite while they live.
        self.buffers: list[torch.Tensor] = []
        self.holders: list[tuple[int, list]] = [(-1, []), (-1, [])]
        self.installed: set[int] = set()
        self.last: tuple[int, bool] | None = None
        self.copies = 0
        self.prefetched = 0

    @property
    def offloaded(self) -> bool:
        return self.host is not None

    def _offload_all(self) -> None:
        total = 0
        for layer in self.layers:
            entries, offset = [], 0
            for module, name, param in _expert_code_params(layer):
                nbytes = param.numel() * param.element_size()
                entries.append((module, name, param, param.shape, param.dtype, offset, nbytes))
                offset = _align(offset + nbytes)
            self.entries.append(entries)
            self.host_offsets.append(total)
            self.layer_bytes.append(offset)
            total += offset
        # One pageable buffer registered with CUDA: the caching pinned allocator rounds every block up to a power of
        # two, which for 91 GiB per rank in 1.2 GiB layers would take 1.6x the host memory.
        raw = torch.empty(max(total, 1) + 4096, dtype=torch.uint8)
        start = -raw.data_ptr() % 4096
        host = raw[start : start + max(total, 1)]
        if total:
            err = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), host.numel(), 0)
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister of {total / 2**30:.1f} GiB of expert codes failed: {err}")
            self.registered = True
            # Unregister before the buffer's memory goes back to the host allocator, or a later registration (or
            # pinning) of reused memory fails with cudaErrorAlreadyMapped.
            weakref.finalize(raw, torch.cuda.cudart().cudaHostUnregister, host.data_ptr())
        self.host = host
        for i, entries in enumerate(self.entries):
            base = self.host_offsets[i]
            for _, _, param, _, dtype, start, nbytes in entries:
                self.host[base + start : base + start + nbytes].copy_(
                    param.data.contiguous().view(torch.uint8).view(-1)
                )
                param.data = torch.empty(0, dtype=dtype, device=param.device)
        self.stream = torch.cuda.Stream()
        # The codes' GPU blocks went back to PyTorch's cache; hand them to the driver too, or NCCL and DeepEP, which
        # allocate outside the cache, still see the memory as taken (out of memory in an NCCL call).
        torch.cuda.empty_cache()
        size = max(self.layer_bytes, default=0)
        device = torch.device("cuda", torch.cuda.current_device())
        self.buffers = [torch.empty(size, dtype=torch.uint8, device=device) for _ in range(2)]

    def _check_free(self, i: int) -> None:
        owner, refs = self.holders[i % 2]
        if owner != i and any(ref() is not None for ref in refs):
            raise RuntimeError(
                f"frozen expert offload: layer {i} would overwrite buffer {i % 2} while layer {owner}'s copy is still "
                "in use (the layers were called out of the forward / reverse-recompute order)"
            )

    def _fetch(self, i: int) -> None:
        """Start copying layer ``i``'s codes into its buffer, unless the buffer holds them already."""
        b = i % 2
        if self.resident[b] == i or not self.layer_bytes[i]:
            return
        assert self.host is not None and self.stream is not None
        current = torch.cuda.current_stream()
        self._check_free(i)
        # The side stream starts after everything queued on the compute stream, the last reads of this buffer included;
        # the compute stream waits for the copy's event before reading it.
        buffer = self.buffers[b][: self.layer_bytes[i]]
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            base = self.host_offsets[i]
            buffer.copy_(self.host[base : base + self.layer_bytes[i]], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.resident[b], self.events[b] = i, event
        self.copies += 1

    def before(self, i: int) -> None:
        """Install layer ``i``'s codes for one call of its experts (one dispatch chunk, or its recompute)."""
        if not self.offloaded:
            self._offload_all()
        self.after(i)
        if not self.layer_bytes[i]:
            return
        from arctic_platform.model.implementations.gpu.router_replay_recompute import _in_recompute
        from arctic_platform.model.implementations.gpu.router_replay_recompute import _install_recompute_detector

        _install_recompute_detector()
        b = i % 2
        recompute = _in_recompute()
        # The first call of this layer in this pass (later calls are the same layer's other dispatch chunks).
        first_call = self.last != (i, recompute)
        if first_call and self.resident[b] == i:
            self.prefetched += 1
        self._fetch(i)
        event = self.events[b]
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        buffer = self.buffers[b]
        refs = self.holders[b][1] if self.holders[b][0] == i else []
        for module, name, param, shape, dtype, start, nbytes in self.entries[i]:
            temporary = nn.Parameter(buffer[start : start + nbytes].view(dtype).view(shape), requires_grad=False)
            for attr in ("allreduce", "_dss_keep_fp32", "ds_shape"):
                if hasattr(param, attr):
                    setattr(temporary, attr, getattr(param, attr))
            temporary._ap_transient = True
            owner, key = _slot(module, name)
            owner._parameters[key] = temporary
            refs = [*refs, weakref.ref(temporary)]
        self.holders[b] = (i, refs)
        self.installed.add(i)
        if first_call:
            # The forward runs the layers up, the recompute (in backward) down; prefetch the next in that order.
            following = i + (-1 if recompute else 1)
            if 0 <= following < len(self.layers):
                self._fetch(following)
        self.last = (i, recompute)

    def after(self, i: int) -> None:
        if i not in self.installed:
            return
        for module, name, param, *_ in self.entries[i]:
            owner, key = _slot(module, name)
            owner._parameters[key] = param
        self.installed.discard(i)

    def host_bytes(self) -> int:
        return 0 if self.host is None else self.host.numel()


def experts_recomputed(layer: nn.Module) -> bool:
    """Whether a decoder layer's routed experts run again in its backward: the whole block checkpointed, or a selective
    checkpoint around them. Otherwise their saved tensors keep the codes alive until backward."""
    if hasattr(layer, "_checkpoint_wrapped_module"):
        return True
    from arctic_platform.model.implementations.moe.layers.checkpointing import _PATCHED_METHODS_ATTR

    mlp = getattr(layer, "mlp", None)
    patched = getattr(mlp, _PATCHED_METHODS_ATTR, ()) if mlp is not None else ()
    return bool({"_run_routed_experts", "_run_local_routed_experts"} & set(patched))


def _expert_call_sites(layer: nn.Module):
    """The modules the MoE calls for its routed experts: ``<moe>.experts`` as installed, which is PEFT's outermost
    ParamWrapper when expert LoRA wraps the experts, else the experts themselves."""
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    seen = set()
    for module in layer.modules():
        site = getattr(module, "experts", None)
        if not isinstance(site, nn.Module) or id(site) in seen:
            continue
        base = site
        while hasattr(base, "base_layer"):
            base = base.base_layer
        if isinstance(base, GroupedExperts):
            seen.add(id(site))
            yield site


def install_frozen_expert_offload(layers: nn.ModuleList) -> FrozenExpertOffload:
    """Offload the frozen routed-expert codes of every decoder layer whose experts are recomputed; copied in per call.

    The hooks sit on the module the MoE calls for its experts (PEFT's outermost wrapper with expert LoRA: the copy is
    installed before PEFT registers its parametrization over the weights), so they run for every call of the experts:
    each dispatch chunk, the forward and the recompute, a selective recompute of the experts alone included. PEFT may
    wrap the experts after this returns (the trainer applies LoRA to the loaded model), so each layer binds its hooks
    to the call sites at its first call.

    Layers whose experts are not recomputed (``ac_config.layers`` leaves them out) keep their codes on the GPU: their
    saved tensors hold them until backward.
    """
    managed = []
    for layer in layers:
        if not experts_recomputed(layer):
            continue
        inner = _inner_layer(layer)
        if any(True for _ in _expert_call_sites(inner)):
            managed.append(inner)
    offload = FrozenExpertOffload(managed)
    for i, inner in enumerate(managed):
        _bind_at_first_call(offload, i, inner)
    return offload


def _bind_at_first_call(offload: FrozenExpertOffload, i: int, layer: nn.Module) -> None:
    """Hook layer ``i``'s experts call sites as they are when the layer first runs, then drop this hook."""
    handle = None

    def bind(module, args):
        for site in _expert_call_sites(layer):
            # First, so any other pre-hook on the experts already sees the installed copy.
            site.register_forward_pre_hook(lambda module, args: offload.before(i), prepend=True)
            site.register_forward_hook(lambda module, args, output: offload.after(i))
            site._ap_frozen_expert_offload = offload
        handle.remove()

    handle = layer.register_forward_pre_hook(bind)
