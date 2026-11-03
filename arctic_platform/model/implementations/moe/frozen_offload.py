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
"""Keep the frozen routed-expert weights in pinned host memory; copy each layer's in around every call of the layer.

GLM-5.3's block-FP8 routed experts are 91 GiB per GPU at EP8, which leaves too little of an H200 for a 256K-token
sequence. With LoRA they are frozen and every layer reads them twice per step: in its forward and in its
activation-checkpoint recompute (the backward reuses the recompute's copy). So each rank keeps its experts' codes in
one host buffer, registered with CUDA, and a layer's codes (1.2 GiB per GPU for GLM-5.3) are copied in on a side
stream for each call. The next layer in call order (forward: the next one; backward: the previous one) is prefetched
while the current one runs, so the copy overlaps compute.

As in ``frozen_shard``, the offload happens at the first call, after PEFT and DeepSpeed have wrapped the model; the
expert parameters keep their objects with empty data, and during a call each is replaced by a temporary parameter
viewing the copied bytes. The temporaries are tagged ``_ap_transient``: the expert GEMMs then save them with
``save_for_backward`` (``fp8.stash_frozen``), so the checkpointed forward drops them after the layer and only the
recompute's copy lives until that layer's backward. Only the code stacks ``w1`` / ``w2`` / ``w3`` move; their scales
stay on the GPU (W4A8 rewrites INT4 scales in place on first use). A model with offloaded experts cannot export them.
"""

from __future__ import annotations

import os

import torch
from torch import nn

OFFLOAD_FROZEN_EXPERTS_ENV = "ARL_OFFLOAD_FROZEN_EXPERTS"
_ALIGN = 256
_NAMES = ("w1", "w2", "w3")


def frozen_expert_offload_requested() -> bool:
    return os.environ.get(OFFLOAD_FROZEN_EXPERTS_ENV, "0") not in ("", "0", "false", "False")


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _inner_layer(layer: nn.Module) -> nn.Module:
    return getattr(layer, "_checkpoint_wrapped_module", layer)


def _expert_code_params(layer: nn.Module):
    """``(owner, name, param)`` of every frozen routed-expert code stack in ``layer``: the experts' own parameter, or
    the ``original`` of its PEFT parametrization."""
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    for module in layer.modules():
        if not isinstance(module, GroupedExperts):
            continue
        parametrizations = getattr(module, "parametrizations", None)
        for name in _NAMES:
            if parametrizations is not None and name in parametrizations:
                owner, key = parametrizations[name], "original"
            else:
                owner, key = module, name
            param = owner._parameters.get(key)
            if param is None or param.requires_grad or param.numel() == 0:
                continue
            yield owner, key, param


class FrozenExpertOffload:
    """The host copies of every managed layer's expert codes, and the per-call copies in."""

    def __init__(self, layers: list[nn.Module]) -> None:
        self.layers = layers
        # Per layer: [(owner, name, param, shape, dtype, offset in the layer's bytes, nbytes)], its host offset, size.
        self.entries: list[list[tuple]] = []
        self.host_offsets: list[int] = []
        self.layer_bytes: list[int] = []
        self.host: torch.Tensor | None = None
        self.registered = False
        self.stream: torch.cuda.Stream | None = None
        self.pending: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self.installed: set[int] = set()
        self.last: int | None = None
        self.direction = 1
        self.copies = 0
        self.prefetched = 0

    @property
    def offloaded(self) -> bool:
        return self.host is not None

    def _offload_all(self) -> None:
        total = 0
        for layer in self.layers:
            entries, offset = [], 0
            for owner, name, param in _expert_code_params(layer):
                nbytes = param.numel() * param.element_size()
                entries.append((owner, name, param, param.shape, param.dtype, offset, nbytes))
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
        self.host = host
        for i, entries in enumerate(self.entries):
            base = self.host_offsets[i]
            for _, _, param, _, dtype, start, nbytes in entries:
                self.host[base + start : base + start + nbytes].copy_(param.data.contiguous().view(torch.uint8).view(-1))
                param.data = torch.empty(0, dtype=dtype, device=param.device)
        self.stream = torch.cuda.Stream()

    def _fetch(self, i: int) -> None:
        if i in self.pending or not self.layer_bytes[i]:
            return
        assert self.host is not None and self.stream is not None
        current = torch.cuda.current_stream()
        # Allocated on the compute stream, which waits for the copy's event before any use or free of the buffer.
        buffer = torch.empty(self.layer_bytes[i], dtype=torch.uint8, device=current.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            base = self.host_offsets[i]
            buffer.copy_(self.host[base : base + self.layer_bytes[i]], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.pending[i] = (buffer, event)
        self.copies += 1

    def before(self, i: int) -> None:
        if not self.offloaded:
            self._offload_all()
        # A recompute that stopped early never reached the post-hook.
        self.after(i)
        if i in self.pending:
            self.prefetched += 1
        self._fetch(i)
        if self.layer_bytes[i]:
            buffer, event = self.pending.pop(i)
            torch.cuda.current_stream().wait_event(event)
            for owner, name, param, shape, dtype, start, nbytes in self.entries[i]:
                temporary = nn.Parameter(buffer[start : start + nbytes].view(dtype).view(shape), requires_grad=False)
                for attr in ("allreduce", "_dss_keep_fp32", "ds_shape"):
                    if hasattr(param, attr):
                        setattr(temporary, attr, getattr(param, attr))
                temporary._ap_transient = True
                owner._parameters[name] = temporary
            self.installed.add(i)
        # Call order: forward 0..n-1, then the recomputes n-1..0; the same layer twice in a row turns around.
        if self.last is not None:
            self.direction = -self.direction if i == self.last else (1 if i > self.last else -1)
        self.last = i
        following = i + self.direction
        if 0 <= following < len(self.layers):
            self._fetch(following)

    def after(self, i: int) -> None:
        if i not in self.installed:
            return
        for owner, name, param, *_ in self.entries[i]:
            owner._parameters[name] = param
        self.installed.discard(i)

    def host_bytes(self) -> int:
        return 0 if self.host is None else self.host.numel()


def install_frozen_expert_offload(layers: nn.ModuleList) -> FrozenExpertOffload:
    """Offload the frozen routed-expert codes of every decoder layer that has routed experts; copied in per call.

    The hooks sit on the layer inside its activation-checkpoint wrapper, so they also run for the recompute, which
    must run the whole layer (``early_stop=False``, as the MoE builders wrap their blocks).
    """
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    inners = [_inner_layer(layer) for layer in layers]
    managed = [inner for inner in inners if any(isinstance(m, GroupedExperts) for m in inner.modules())]
    offload = FrozenExpertOffload(managed)
    for i, inner in enumerate(managed):
        inner.register_forward_pre_hook(lambda module, args, i=i: offload.before(i))
        inner.register_forward_hook(lambda module, args, output, i=i: offload.after(i))
        inner._ap_frozen_expert_offload = offload
    return offload
