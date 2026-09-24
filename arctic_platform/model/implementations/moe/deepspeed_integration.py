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
"""Shared DeepSpeed lifecycle for Prime-style MoE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Mapping

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module

from arctic_platform.model.implementations.debug.token_combine import maybe_use_fixed_order_row_sum
from arctic_platform.model.implementations.fp8 import carry_keep_fp32
from arctic_platform.model.implementations.gpu.tiled_mlp import enable_tiled_mlp

from .config_validation import validate_lm_head_fused_ce_config
from .distributed.ep_backend import uses_dispatch_ep
from .distributed.expert_parallel import DeepEPExpertParallel
from .layers.moe import LatentMoE
from .layers.moe import MoE
from .logging_utils import get_logger
from .parallel_dims import ParallelDims
from .vlm import get_language_model


@dataclass(frozen=True)
class MoEDeepSpeedAdapter:
    dtype_map: Mapping[str, torch.dtype]
    get_model: Callable[..., nn.Module]
    configure_moe_ep_backend: Callable[[nn.Module, Any], None]
    configure_family_backend: Callable[[Any], None]
    inject_lm_head: Callable[..., None]
    apply_sequence_parallelism: Callable[[nn.Module, int, Any], None]
    apply_ac: Callable[[nn.Module, Any], None]
    load_dcp_from_hf: Callable[[nn.Module, Any, ParallelDims], None]
    reset_runtime_moe_buffers: Callable[[nn.Module], None]
    shared_expert_type: type[nn.Module]
    shared_expert_forward: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    build_model_config: Callable[[str, int, int, dict], Any]
    normalize_config: Callable[[dict], dict]
    extra_weight_iterators: tuple[tuple[str, Callable[[nn.Module], Callable]], ...] = ()


def patch_deepspeed_moe_detection() -> None:
    import deepspeed.runtime.engine as ds_engine

    if getattr(ds_engine.DeepSpeedEngine, "_ap_moe_patched", False):
        return
    original = ds_engine.DeepSpeedEngine._configure_distributed_model

    def patched(self, model):
        original(self, model)
        if self.has_moe_layers:
            return
        for module in self.module.modules():
            if not isinstance(module, (MoE, LatentMoE)):
                continue
            experts = getattr(module, "experts", None)
            if experts is not None and not any(parameter.requires_grad for parameter in experts.parameters()):
                continue
            self.has_moe_layers = True
            num_experts = getattr(module, "num_experts", None)
            if num_experts is None and experts is not None:
                num_experts = getattr(experts, "num_experts", None)
            if num_experts is not None:
                self.num_experts.append(num_experts)

    ds_engine.DeepSpeedEngine._configure_distributed_model = patched
    ds_engine.DeepSpeedEngine._ap_moe_patched = True


def apply_ep_with_mesh(model: nn.Module, config: Any, ep_mesh: DeviceMesh) -> None:
    if not uses_dispatch_ep(config.ep_comm_backend):
        raise NotImplementedError(
            f"EP comm backend must be one of ('deepep', 'uccl'), got {config.ep_comm_backend!r}."
        )
    for transformer_block in get_language_model(model).layers:
        block_mlp = getattr(transformer_block, "mlp", None)
        if isinstance(block_mlp, (MoE, LatentMoE)):
            parallelize_module(
                block_mlp.experts,
                device_mesh=ep_mesh,
                parallelize_plan=DeepEPExpertParallel(),
            )


def convert_dtensors_to_local(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if not isinstance(parameter.data, DTensor) and not hasattr(parameter, "to_local"):
                continue
            local = parameter.to_local().detach()
            replacement = carry_keep_fp32(
                parameter,
                nn.Parameter(local, requires_grad=parameter.requires_grad),
            )
            module.register_parameter(name, replacement)
            count += 1
    return count


def tag_expert_params_for_deepspeed(model: nn.Module, ep_group_name: str) -> int:
    count = 0
    for module in model.modules():
        if not isinstance(module, (MoE, LatentMoE)):
            continue
        for parameter in module.experts.parameters(recurse=False):
            parameter.allreduce = False
            parameter.group_name = ep_group_name
            count += 1
    return count


def tag_expert_lora_adapters_for_deepspeed(model: nn.Module, ep_group_name: str) -> int:
    count = 0
    for module in model.modules():
        if not isinstance(module, (MoE, LatentMoE)):
            continue
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        for parameter in experts.parameters(recurse=True):
            if parameter.requires_grad and getattr(parameter, "group_name", None) is None:
                parameter.allreduce = False
                parameter.group_name = ep_group_name
                count += 1
    return count


def build_iter_full_hf_weights(model: nn.Module):
    model_type = type(model)
    convert_layer_to_hf = getattr(model_type, "convert_layer_to_hf", None)
    if convert_layer_to_hf is None:
        raise RuntimeError(f"build_iter_full_hf_weights requires {model_type.__name__}.convert_layer_to_hf")

    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    from arctic_platform.model.implementations.moe.weights import hf_export_param_name

    def layer_index(name: str) -> int:
        parts = name.split(".")
        if len(parts) >= 4 and parts[:3] == ["model", "language_model", "layers"]:
            index = parts[3]
        elif len(parts) >= 3 and parts[:2] == ["model", "layers"]:
            index = parts[2]
        else:
            return -1
        return int(index) if index.isdigit() else -1

    def iterator():
        is_master = dist.get_rank() == 0
        by_layer: dict[int, list[tuple[str, nn.Parameter]]] = {}
        for name, parameter in model.named_parameters():
            hf_name = hf_export_param_name(name)
            if hf_name is not None:
                by_layer.setdefault(layer_index(hf_name), []).append((hf_name, parameter))

        ordered_layers = ([-1] if -1 in by_layer else []) + sorted(index for index in by_layer if index >= 0)
        for index in ordered_layers:
            layer_state: dict[str, torch.Tensor] = {}
            for name, parameter in by_layer[index]:
                if hasattr(parameter, "group_name") and getattr(parameter, "allreduce", True) is False:
                    ep_group = ds_groups._get_expert_parallel_group(parameter.group_name)
                    local = parameter.data.contiguous()
                    shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=ep_group))]
                    dist.all_gather(shards, local, group=ep_group)
                    if is_master:
                        layer_state[name] = torch.cat(shards, dim=0)
                elif is_master:
                    layer_state[name] = parameter.data

            if not is_master:
                continue
            layer_state = {
                name.replace("._checkpoint_wrapped_module", ""): tensor for name, tensor in layer_state.items()
            }
            convert_layer_to_hf(layer_state, index)
            yield from layer_state.items()

    return iterator


def setup_model_local_no_train(
    adapter: MoEDeepSpeedAdapter,
    config: Any,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    *,
    fused_cross_entropy: bool | str = False,
    tiled_mlp_token_chunk_size: int | None = None,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    logger = get_logger()
    model = adapter.get_model(
        config,
        device=torch.device("meta"),
        dtype=adapter.dtype_map[config.optimization_dtype],
    )
    adapter.configure_moe_ep_backend(model, config)
    adapter.configure_family_backend(config)

    chunk_size = (
        config.fused_lm_head_token_chunk_size if isinstance(config.fused_lm_head_token_chunk_size, int) else None
    )
    adapter.inject_lm_head(
        model,
        chunk_size=chunk_size,
        fused_cross_entropy=fused_cross_entropy,
        fp32_lm_head=config.fp32_lm_head,
    )
    if parallel_dims.ep_enabled:
        apply_ep_with_mesh(model, config, ep_mesh)
    adapter.apply_sequence_parallelism(model, sp_size, sp_group)
    if config.ac is not None:
        adapter.apply_ac(model, config.ac)

    adapter.load_dcp_from_hf(model, config, parallel_dims)
    adapter.reset_runtime_moe_buffers(model)
    count = convert_dtensors_to_local(model)
    logger.info(f"Unwrapped {count} DTensor parameters into local tensors")
    enable_tiled_mlp(
        model,
        is_target=lambda module: isinstance(module, adapter.shared_expert_type),
        mlp_forward=adapter.shared_expert_forward,
        token_chunk_size=tiled_mlp_token_chunk_size,
        logger=logger,
    )
    return model


def load_moe_model_for_deepspeed(
    adapter: MoEDeepSpeedAdapter,
    model_config: Any,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    ep_group_name: str,
    *,
    fused_cross_entropy: bool | str = False,
    tiled_mlp_token_chunk_size: int | None = None,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    model = setup_model_local_no_train(
        adapter,
        model_config,
        parallel_dims,
        ep_mesh,
        fused_cross_entropy=fused_cross_entropy,
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
        sp_size=sp_size,
        sp_group=sp_group,
    )
    count = tag_expert_params_for_deepspeed(model, ep_group_name)
    get_logger().info(f"Tagged {count} expert parameters with group_name='{ep_group_name}', allreduce=False")
    model._iter_full_hf_weights = build_iter_full_hf_weights(model)
    for attribute, builder in adapter.extra_weight_iterators:
        setattr(model, attribute, builder(model))
    return model


def load_moe_model_for_dss(
    adapter: MoEDeepSpeedAdapter,
    family_loader: Callable,
    *,
    model_name: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    prl_config: dict | None = None,
    tiled_mlp_token_chunk_size: int | None = None,
    patch_moe_detection: Callable[[], None] = patch_deepspeed_moe_detection,
    device_mesh_type=DeviceMesh,
) -> nn.Module:
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    config = adapter.normalize_config(dict(prl_config or {}))
    validate_lm_head_fused_ce_config(config)
    world_size = dist.get_world_size()
    if world_size % ep_size:
        raise ValueError(f"world_size={world_size} must be divisible by ep_size={ep_size}")
    patch_moe_detection()

    ep_group_name = f"ep_size_{ep_size}"
    if ep_group is None:
        ep_group = ds_groups._get_expert_parallel_group(ep_group_name)
    if dist.get_world_size(ep_group) != ep_size:
        raise ValueError("ep_group size must match ep_size")
    if sp_size > 1 and (sp_group is None or dist.get_world_size(sp_group) != sp_size):
        raise ValueError("sp_group size must match sp_size")
    ep_mesh = device_mesh_type.from_group(
        group=ep_group,
        device_type="cuda",
        mesh_dim_names=("ep",),
    )
    ep_backend = config.get("ep_comm_backend", "deepep")
    if not uses_dispatch_ep(ep_backend):
        raise NotImplementedError(f"EP comm backend must be one of ('deepep', 'uccl'), got {ep_backend!r}.")

    dp_replicate = world_size // ep_size
    model_config = adapter.build_model_config(model_name, ep_size, dp_replicate, config)
    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=-1,
        cp=1,
        pp=1,
        ep=ep_size,
        world_size=world_size,
    )
    model = family_loader(
        model_config,
        parallel_dims,
        ep_mesh,
        ep_group_name,
        fused_cross_entropy=config.get("fused_cross_entropy", "liger"),
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
        sp_size=sp_size,
        sp_group=sp_group,
    )
    maybe_use_fixed_order_row_sum(model, config)
    return model
