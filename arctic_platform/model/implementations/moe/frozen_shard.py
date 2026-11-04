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
"""Shard each decoder layer's frozen replicated weights across ranks; gather them around every call of the layer.

With LoRA every non-expert weight (attention, shared and dense MLPs, router, norms) is frozen and replicated on all
ranks: about 16 GiB per GPU for GLM-5.3 in FP8, twice that in BF16. Frozen weights need no gradient reduction, so a
rank can keep 1/world of their bytes and all-gather the layer's weights when the layer runs, in its forward and in
its activation-checkpoint recompute, as ZeRO-3 does for its parameters. Expert parameters (EP-sharded, tagged
``allreduce=False``) and trainable parameters are left alone.

Sharding happens at each layer's first call, so it sees the model after PEFT and DeepSpeed have wrapped it. The
layer's frozen parameters keep their objects (and DeepSpeed's references to them) with empty data; during a call
each is replaced in its owner's ``_parameters`` by a temporary parameter viewing the gathered bytes, which autograd
keeps alive for as long as a saved tensor needs it. A model whose frozen weights are sharded cannot export them.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

SHARD_FROZEN_DENSE_ENV = "ARL_SHARD_FROZEN_DENSE"
# Byte alignment of every tensor inside the gathered buffer (TMA and vectorized loads need 16 B).
_ALIGN = 256


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class FrozenLayerShard:
    """The frozen replicated parameters of one layer, held as a ``1 / world`` byte shard per rank."""

    def __init__(self, layer: nn.Module, group: dist.ProcessGroup) -> None:
        self.layer = layer
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.entries: list[tuple[nn.Module, str, nn.Parameter, torch.Size, torch.dtype, int, int]] | None = None
        self.shard: torch.Tensor | None = None
        self.shard_bytes = 0
        self.full_bytes = 0
        self.gathered = False

    def _targets(self):
        for module in self.layer.modules():
            for name, param in module._parameters.items():
                if param is None or param.requires_grad or param.numel() == 0:
                    continue
                if getattr(param, "allreduce", True) is False:
                    continue
                yield module, name, param

    def _shard(self) -> None:
        entries = []
        offset = 0
        for owner, name, param in self._targets():
            nbytes = param.numel() * param.element_size()
            entries.append((owner, name, param, param.shape, param.dtype, offset, nbytes))
            offset = _align(offset + nbytes)
        self.full_bytes = offset
        self.shard_bytes = _align((offset + self.world - 1) // self.world)
        device = entries[0][2].device if entries else torch.device("cuda", torch.cuda.current_device())
        full = torch.zeros(self.shard_bytes * self.world, dtype=torch.uint8, device=device)
        for _, _, param, _, _, start, nbytes in entries:
            full[start : start + nbytes].copy_(param.data.contiguous().view(torch.uint8).view(-1))
        self.shard = full[self.rank * self.shard_bytes : (self.rank + 1) * self.shard_bytes].clone()
        del full
        for _, _, param, _, dtype, _, _ in entries:
            param.data = torch.empty(0, dtype=dtype, device=device)
        self.entries = entries

    def gather(self) -> None:
        # A recompute that stopped early never reached the post-hook; drop what it gathered first.
        self.release()
        if self.entries is None:
            self._shard()
        assert self.entries is not None and self.shard is not None
        full = torch.empty(self.shard_bytes * self.world, dtype=torch.uint8, device=self.shard.device)
        dist.all_gather_into_tensor(full, self.shard, group=self.group)
        for owner, name, param, shape, dtype, start, nbytes in self.entries:
            view = full[start : start + nbytes].view(dtype).view(shape)
            temporary = nn.Parameter(view, requires_grad=False)
            for attr in ("_dss_keep_fp32",):
                if hasattr(param, attr):
                    setattr(temporary, attr, getattr(param, attr))
            owner._parameters[name] = temporary
        self.gathered = True

    def release(self) -> None:
        if not self.gathered or self.entries is None:
            return
        for owner, name, param, *_ in self.entries:
            owner._parameters[name] = param
        self.gathered = False


def _inner_layer(layer: nn.Module) -> nn.Module:
    return getattr(layer, "_checkpoint_wrapped_module", layer)


def install_frozen_layer_shards(layers: nn.ModuleList, group: dist.ProcessGroup | None = None) -> list[FrozenLayerShard]:
    """Shard every layer's frozen replicated weights over ``group`` (default: all ranks), gathered per call.

    The hooks sit on the layer inside its activation-checkpoint wrapper, so they also run for the recompute, which
    must run the whole layer (``early_stop=False``, as the MoE builders wrap their blocks) for the post-hook to
    release the gathered weights. Only such whole-block-checkpointed layers are sharded: a layer that is not
    recomputed keeps its gathered weights saved for backward anyway, and selective checkpointing recomputes submodule
    methods in backward, outside the layer call, after the gathered weights were released.
    """
    if group is None:
        group = dist.new_group(list(range(dist.get_world_size())))
    shards = []
    for layer in layers:
        if not hasattr(layer, "_checkpoint_wrapped_module"):
            continue
        inner = _inner_layer(layer)
        shard = FrozenLayerShard(inner, group)
        inner.register_forward_pre_hook(lambda module, args, shard=shard: shard.gather())
        inner.register_forward_hook(lambda module, args, output, shard=shard: shard.release())
        inner._ap_frozen_shard = shard
        shards.append(shard)
    return shards


_configured: bool | None = None  # loader option moe.shard_frozen_dense (moe/runtime_options.py); None: the env


def configure_from_options(value: bool | None) -> None:
    global _configured
    _configured = value


def frozen_dense_sharding_requested() -> bool:
    if _configured is not None:
        return _configured
    return os.environ.get(SHARD_FROZEN_DENSE_ENV, "0") not in ("", "0", "false", "False")
