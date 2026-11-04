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
"""Shard the trainable replicated weights of each decoder layer over the GPUs of a node, FSDP-style, for full FT.

Full fine-tuning keeps every weight that is not expert-parallel (attention and its DSA indexer, shared experts, the
dense MLPs, router, norms, embedding, LM head) whole on every GPU: GLM-5.3-BF16 has ~19B of them, 35.8 GiB per GPU,
beside the 21 GiB of each GPU's own routed experts at EP64, and its first backward runs out of memory. Here each GPU of
a node keeps ``1 / g`` of them (``g`` GPUs per node), as ZeRO-3 / FSDP do inside the node only.

At load, before DeepSpeed sees the model, every target module (each decoder layer, the embedding, the LM head) has its
trainable parameters that are not expert-parallel flattened into one buffer per dtype and weight-decay class, each at
a 256-byte aligned offset, padded to a multiple of ``g``. Rank ``i`` of the node keeps slice ``i`` as one parameter
registered on the module (``dense_shard.shard_<k>``), and the originals leave their owners' ``_parameters`` (their
shapes and offsets are kept). Parameters kept in FP32 (``_dss_keep_fp32``), FP8 or frozen ones, and parameters shared
with a module outside the target stay as they are.

A forward pre-hook on the module (inside its activation-checkpoint wrapper, so the recompute gathers again) calls one
autograd function that all-gathers the shards over the node and returns each parameter as its own output, a view of
the gathered buffer; the hook puts those into the owners' ``_parameters`` for the call and the post-hook removes them.
The function's backward writes the outputs' gradients into the flat layout (zeros where a parameter got none and in the
padding), in FP32 staging buffers of at most 256 MiB, reduce-scatters them with SUM over the node, and returns the
shard's gradient. Under full activation checkpointing a layer's gathered weights live only through its forward and its
recompute plus backward.

The node group is DeepSpeed's expert-parallel group ``ep_size_<g>`` (consecutive blocks of ``g`` ranks, checked to be
the ranks of one host; the node's sequence-parallel group when it has one, so no new communicator), and each shard is
tagged as an expert-parallel parameter of it (``allreduce=False``, ``group_name``). ZeRO therefore reduces and
partitions a shard over that group's expert-data-parallel group, the ranks with the same local rank on every node, and
divides it by the same ``world / sp`` as every bucket: the in-node sum followed by the cross-node sum is the sum over
all ranks it takes for a replicated parameter. The gradients equal the replicated path's up to one BF16 rounding of
the in-node FP32 sum (the shard's gradient has the shard's dtype). ZeRO's gradient norm counts each partition of an
expert-parallel group once over the whole world and its expert average is then a no-op, so the norm, and the
clipping, are the replicated run's.

A sharded model cannot be saved or exported: ``state_dict``, ``load_state_dict`` and the HF / vLLM weight export raise.
"""

from __future__ import annotations

import os
import socket
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn

SHARD_DENSE_ENV = "ARL_SHARD_DENSE"
SHARD_DENSE_GROUP_SIZE_ENV = "ARL_SHARD_DENSE_GROUP_SIZE"
# Set on every shard parameter (DeepSpeed's MoE detection and the optimizer's group split look for it).
DENSE_SHARD_ATTR = "_ap_dense_shard"
# Byte alignment of every parameter inside the gathered buffer (TMA and vectorized loads need 16 B).
_ALIGN_BYTES = 256
# FP32 elements of one reduce-scatter's staging buffer (256 MiB): no backward holds a module's whole gradient in FP32.
_REDUCE_CHUNK_ELEMENTS = 64 << 20

_configured: bool | None = None  # loader option moe.shard_dense (moe/runtime_options.py); None: the env
_configured_group_size: int | None = None  # loader option moe.shard_dense_group_size; None: the env, else a host


def configure_from_options(value: bool | None) -> None:
    global _configured
    _configured = value


def configure_group_size(value: int | None) -> None:
    global _configured_group_size
    _configured_group_size = value


def resolve_dense_sharding(value: bool | None) -> bool:
    """The switch for an option ``value``: the value when set, else ``ARL_SHARD_DENSE``."""
    if value is not None:
        return value
    return os.environ.get(SHARD_DENSE_ENV, "0") not in ("", "0", "false", "False")


def dense_sharding_requested() -> bool:
    return resolve_dense_sharding(_configured)


def configured_group_size() -> int | None:
    """GPUs per shard group: the option, else ``ARL_SHARD_DENSE_GROUP_SIZE``, else None (the ranks of one host)."""
    if _configured_group_size is not None:
        return _configured_group_size
    value = os.environ.get(SHARD_DENSE_GROUP_SIZE_ENV, "")
    return int(value) if value else None


def is_dense_shard(param: torch.Tensor) -> bool:
    return bool(getattr(param, DENSE_SHARD_ATTR, False))


@dataclass(frozen=True)
class _Member:
    """One original parameter: where it sits in the flat layout, and the owners' slots it fills during a call."""

    name: str
    slots: tuple[tuple[nn.Module, str], ...]
    shape: torch.Size
    offset: int
    numel: int


@dataclass
class _Bucket:
    """The members of one dtype and weight-decay class, and this rank's shard of their flat layout."""

    members: list[_Member]
    shard: nn.Parameter
    total: int  # padded elements of the flat layout: the group size times the shard's


def _reduce_scatter(bucket: _Bucket, grads: tuple, group: dist.ProcessGroup, world: int) -> torch.Tensor | None:
    """This rank's slice of the bucket's flat gradient, summed over ``group`` in FP32, in the shard's dtype."""
    flats = [None if grad is None else grad.reshape(-1) for grad in grads]
    present = [flat for flat in flats if flat is not None]
    if not present:
        return None
    size = bucket.total // world
    out = torch.empty(size, dtype=torch.float32, device=present[0].device)
    step = max(1, min(size, _REDUCE_CHUNK_ELEMENTS // world))
    for start in range(0, size, step):
        end = min(size, start + step)
        staging = torch.zeros(world, end - start, dtype=torch.float32, device=out.device)
        for rank in range(world):
            low, high = rank * size + start, rank * size + end
            for member, flat in zip(bucket.members, flats):
                if flat is None:
                    continue
                first, last = max(low, member.offset), min(high, member.offset + member.numel)
                if first < last:
                    staging[rank, first - low : last - low].copy_(flat[first - member.offset : last - member.offset])
        dist.reduce_scatter_tensor(out[start:end], staging.view(-1), op=dist.ReduceOp.SUM, group=group)
    return out.to(bucket.shard.dtype)


class _GatherShards(torch.autograd.Function):
    """Shards in, every member parameter out (each its own output); backward: their gradients reduce-scattered."""

    @staticmethod
    def forward(ctx, sharded: "ShardedModule", *shards: torch.Tensor):
        ctx.sharded = sharded
        ctx.set_materialize_grads(False)  # an unused parameter's gradient stays None, not a full-size zero
        outputs = []
        for bucket, shard in zip(sharded.buckets, shards):
            full = torch.empty(bucket.total, dtype=shard.dtype, device=shard.device)
            dist.all_gather_into_tensor(full, shard, group=sharded.group)
            outputs.extend(full.narrow(0, m.offset, m.numel).view(m.shape) for m in bucket.members)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        sharded = ctx.sharded
        # A recompute that stopped early never reached the post-hook; its gathered weights are not needed any more.
        sharded.release()
        out = []
        index = 0
        for bucket in sharded.buckets:
            count = len(bucket.members)
            out.append(_reduce_scatter(bucket, grads[index : index + count], sharded.group, sharded.world))
            index += count
        return (None, *out)


class _ShardParams(nn.Module):
    """Holds a module's shard parameters (``shard_<k>``); never called."""


def _refuse_state_dict(module, *args, **kwargs):
    raise RuntimeError(
        f"{type(module).__name__}'s trainable replicated weights are sharded over the node (moe.shard_dense / "
        f"{SHARD_DENSE_ENV}): its state dict would hold shards instead of its parameters. Saving, loading and "
        "exporting a sharded model are not supported; run without shard_dense to save checkpoints."
    )


class ShardedModule:
    """A module whose trainable replicated parameters are held as ``1 / world`` shards per rank of ``group``."""

    def __init__(self, module: nn.Module, buckets: list[_Bucket], group: dist.ProcessGroup) -> None:
        self.module = module
        self.buckets = buckets
        self.group = group
        self.world = dist.get_world_size(group)
        self.gathered = False

    @property
    def members(self) -> Iterable[_Member]:
        for bucket in self.buckets:
            yield from bucket.members

    def gather(self) -> None:
        self.release()
        tensors = _GatherShards.apply(self, *(bucket.shard for bucket in self.buckets))
        for member, tensor in zip(self.members, tensors):
            for owner, name in member.slots:
                owner._parameters[name] = tensor
        self.gathered = True

    def release(self) -> None:
        if not self.gathered:
            return
        for member in self.members:
            for owner, name in member.slots:
                owner._parameters.pop(name, None)
        self.gathered = False

    def full_parameters(self) -> dict[str, torch.Tensor]:
        """Each member's whole tensor, by its name in the model (collective over the shard group)."""
        out = {}
        for bucket in self.buckets:
            shard = bucket.shard.detach().contiguous()
            full = torch.empty(bucket.total, dtype=shard.dtype, device=shard.device)
            dist.all_gather_into_tensor(full, shard, group=self.group)
            for member in bucket.members:
                out[member.name] = full.narrow(0, member.offset, member.numel).view(member.shape)
        return out

    def full_bytes(self) -> int:
        return sum(m.numel * b.shard.element_size() for b in self.buckets for m in b.members)

    def shard_bytes(self) -> int:
        return sum(b.shard.numel() * b.shard.element_size() for b in self.buckets)


def _parameter_slots(root: nn.Module) -> tuple[dict[int, list[tuple[nn.Module, str]]], dict[int, str]]:
    """Every (owner, name) slot of each parameter under ``root``, and each parameter's first qualified name."""
    slots: dict[int, list[tuple[nn.Module, str]]] = {}
    names: dict[int, str] = {}
    for prefix, module in root.named_modules(remove_duplicate=False):
        for name, param in module._parameters.items():
            if param is None:
                continue
            slots.setdefault(id(param), []).append((module, name))
            names.setdefault(id(param), f"{prefix}.{name}" if prefix else name)
    return slots, names


def _shardable(module: nn.Module, name: str, param: nn.Parameter | None) -> bool:
    from arctic_platform.model.implementations.fp8 import keeps_fp32

    if param is None or not param.requires_grad or param.numel() == 0 or not param.is_floating_point():
        return False
    if getattr(param, "allreduce", True) is False or getattr(module, "_dss_shard_on_ep", False):
        return False  # expert-parallel: already one copy per EP rank
    return not keeps_fp32(param)


def shard_module(
    module: nn.Module,
    group: dist.ProcessGroup,
    group_name: str,
    slots: dict[int, list[tuple[nn.Module, str]]],
    names: dict[int, str],
) -> ShardedModule | None:
    """Replace ``module``'s shardable parameters by shards over ``group`` and gather them around its calls; None when
    it has none. ``slots`` / ``names``: ``_parameter_slots`` of the whole model (a parameter with a slot outside
    ``module`` stays whole)."""
    from arctic_platform.common.deepspeed_engine import exempt_from_weight_decay

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    inside = {id(m) for m in module.modules()}
    by_class: dict[tuple[torch.dtype, bool], list[tuple[nn.Parameter, list]]] = {}
    seen: set[int] = set()
    for owner in module.modules():
        for name, param in owner._parameters.items():
            if id(param) in seen or not _shardable(owner, name, param):
                continue
            seen.add(id(param))
            param_slots = slots[id(param)]
            if any(id(slot_owner) not in inside for slot_owner, _ in param_slots):
                continue
            decay_class = (param.dtype, exempt_from_weight_decay(names[id(param)], param))
            by_class.setdefault(decay_class, []).append((param, param_slots))
    if not by_class:
        return None

    holder = _ShardParams()
    buckets = []
    for (dtype, no_decay), params in by_class.items():
        align = max(1, _ALIGN_BYTES // params[0][0].element_size())
        members = []
        offset = 0
        for param, param_slots in params:
            members.append(_Member(names[id(param)], tuple(param_slots), param.shape, offset, param.numel()))
            offset = (offset + param.numel() + align - 1) // align * align
        total = (offset + world * align - 1) // (world * align) * (world * align)
        size = total // world
        flat = torch.zeros(total, dtype=dtype, device=params[0][0].device)
        for member, (param, _) in zip(members, params):
            flat.narrow(0, member.offset, member.numel).copy_(param.data.reshape(-1))
        shard = nn.Parameter(flat.narrow(0, rank * size, size).clone())
        del flat
        shard.allreduce = False
        shard.group_name = group_name
        setattr(shard, DENSE_SHARD_ATTR, True)
        if no_decay:
            from arctic_platform.common.deepspeed_engine import NO_WEIGHT_DECAY_ATTR

            setattr(shard, NO_WEIGHT_DECAY_ATTR, True)
        holder.register_parameter(f"shard_{len(buckets)}", shard)
        buckets.append(_Bucket(members, shard, total))
        for member in members:
            for owner, name in member.slots:
                del owner._parameters[name]
    module.add_module("dense_shard", holder)
    sharded = ShardedModule(module, buckets, group)
    module.register_forward_pre_hook(lambda _module, _args: sharded.gather(), prepend=True)
    module.register_forward_hook(lambda _module, _args, _output: sharded.release())
    module.register_state_dict_pre_hook(_refuse_state_dict)
    module.register_load_state_dict_pre_hook(_refuse_state_dict)
    module._ap_dense_shards = sharded
    return sharded


def node_shard_group(group_size: int | None = None) -> tuple[dist.ProcessGroup | None, str, int]:
    """The process group of this rank's node and its DeepSpeed expert-group name, ``(group, "ep_size_<g>", g)``;
    ``group`` is None when a node has one GPU.

    ``group_size`` None: the ranks of one host (every host must hold as many). Each group is a block of consecutive
    ranks, all on one host (checked from the hostnames), registered as DeepSpeed's expert-parallel group of that size
    with expert-data-parallel groups of the ranks at the same position in every block (``groups.py``).
    """
    from arctic_platform.model.parallel.groups import ep_group_name
    from arctic_platform.model.parallel.groups import setup_consecutive_expert_groups

    world = dist.get_world_size()
    hosts: list = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    if group_size is None:
        counts = set(Counter(hosts).values())
        if len(counts) != 1:
            raise RuntimeError(
                f"shard_dense: the hosts hold different numbers of ranks ({dict(Counter(hosts))}); set "
                f"moe.shard_dense_group_size ({SHARD_DENSE_GROUP_SIZE_ENV})"
            )
        group_size = counts.pop()
    if world % group_size:
        raise ValueError(f"shard_dense: group size {group_size} does not divide the world size {world}")
    for start in range(0, world, group_size):
        block = set(hosts[start : start + group_size])
        if len(block) != 1:
            raise RuntimeError(
                f"shard_dense: ranks {start}-{start + group_size - 1} run on {sorted(block)}; a dense shard group "
                "must be consecutive ranks of one node (rank // gpus_per_node names the node)"
            )
    name = ep_group_name(group_size)
    if group_size == 1:
        return None, name, 1
    return setup_consecutive_expert_groups(group_size), name, group_size


def install_dense_shards(
    root: nn.Module, modules: Iterable[nn.Module], group_size: int | None = None
) -> list[ShardedModule]:
    """Shard each of ``modules`` (submodules of ``root``) over this rank's node; see the module docstring.

    Collective over the default group (the shard groups are created here). Run after the weights are loaded and the
    experts tagged, before the optimizer and DeepSpeed are built.
    """
    group, name, _ = node_shard_group(group_size)
    if group is None:
        return []
    slots, names = _parameter_slots(root)
    sharded = [s for s in (shard_module(m, group, name, slots, names) for m in modules) if s is not None]

    from arctic_platform.model.weight_export import WeightExportContract
    from arctic_platform.model.weight_export import register_weight_export

    def refuse_export(model):
        _refuse_state_dict(model)

    register_weight_export(root, WeightExportContract(hf=refuse_export, vllm=refuse_export))
    root.register_state_dict_pre_hook(_refuse_state_dict)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # the originals' blocks
    return sharded


def _inner(module: nn.Module) -> nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


def install_model_dense_shards(model: nn.Module, group_size: int | None = None) -> list[ShardedModule]:
    """``install_dense_shards`` for a Prime-style MoE model: each decoder layer (inside its activation-checkpoint
    wrapper), the input embedding and the LM head (an LM head tied to the embedding stays whole)."""
    from .logging_utils import get_logger
    from .vlm import get_language_model

    language_model = get_language_model(model)
    layers = list(language_model.layers)
    modules = [_inner(layer) for layer in layers]
    for extra in (getattr(language_model, "embed_tokens", None), getattr(model, "lm_head", None)):
        if isinstance(extra, nn.Module):
            modules.append(extra)
    sharded = install_dense_shards(model, modules, group_size)
    if sharded:
        group = sharded[0].group
        full = sum(s.full_bytes() for s in sharded)
        kept = sum(s.shard_bytes() for s in sharded)
        unchecked = sum(1 for layer in layers if _inner(layer) is layer)
        get_logger().info(
            f"Dense weights of {len(sharded)} modules sharded over {dist.get_world_size(group)} GPUs of the node: "
            f"{full / 2**30:.2f} GiB -> {kept / 2**30:.2f} GiB per GPU"
            + (
                f" ({unchecked} decoder layers are not recomputed: they keep their gathered weights for the backward)"
                if unchecked
                else ""
            )
        )
    return sharded
