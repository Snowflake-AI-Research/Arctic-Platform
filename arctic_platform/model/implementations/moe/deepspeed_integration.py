"""Shared DeepSpeed lifecycle for Prime-style MoE models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
from urllib.parse import quote

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module

from arctic_platform.debug.lora import maybe_observe_model_materialization
from arctic_platform.model.implementations.debug.token_combine import maybe_use_fixed_order_row_sum
from arctic_platform.model.implementations.fp8 import carry_keep_fp32
from arctic_platform.model.implementations.gpu.tiled_mlp import enable_tiled_mlp
from arctic_platform.model.loader import canonical_parameter_name

from .distributed.ep_backend import uses_dispatch_ep
from .distributed.expert_parallel import DeepEPExpertParallel
from .distributed.expert_parallel import DeepEPShardParallel
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
    build_model_config: Callable[[str, int, int, str, str, Any], Any]
    vllm_weight_export: Callable[[nn.Module], Callable] | None = None
    hf_weight_export: Callable[[nn.Module], Callable] | None = None
    # ``(config) -> source dir`` and ``(model, config, source, *, rank, local_rank) -> cache dir | None``: the
    # conversion half of ``load_dcp_from_hf``, callable without a process group.
    resolve_weight_source: Callable[[Any], Path] | None = None
    prepare_weight_conversion: Callable[..., Path | None] | None = None


def _ep_checkpoint_parameters(model: nn.Module) -> dict[str, dict[str, nn.Parameter]]:
    if not any(isinstance(module, (MoE, LatentMoE)) for module in model.modules()):
        return {}
    groups: dict[str, dict[str, nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "allreduce", True) is False:
            group_name = parameter.group_name
            groups.setdefault(group_name, {})[name] = parameter
    return groups


def _ep_checkpoint_identity(group_name: str) -> dict:
    import deepspeed.utils.groups as ds_groups

    return dict(
        group_name=group_name,
        ep_size=ds_groups._get_expert_parallel_world_size(group_name),
        ep_rank=ds_groups._get_expert_parallel_rank(group_name),
    )


def _ep_checkpoint_path(directory, tag, identity: dict) -> Path:
    group_name = quote(identity["group_name"], safe="")
    return Path(directory) / tag / f"ap_ep_{group_name}_rank_{identity['ep_rank']:02d}_model_states.pt"


def _patch_deepspeed_ep_checkpoints(engine_type) -> None:
    import deepspeed.utils.groups as ds_groups
    from deepspeed.checkpoint.constants import FROZEN_PARAM_FRAGMENTS

    original_load = engine_type._load_checkpoint
    original_module_load = engine_type.load_module_state_dict
    load_signature = signature(original_load)
    module_load_signature = signature(original_module_load)

    def wrap_save(original_save):
        save_signature = signature(original_save)

        @wraps(original_save)
        def save(self, *args, **kwargs):
            bound = save_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            parameters = _ep_checkpoint_parameters(self.module)
            if len(parameters) > 0 and self.zero_optimization_partition_weights():
                raise ValueError("AP expert-parallel checkpoint shards do not support ZeRO stage 3")
            result = original_save(self, *args, **kwargs)
            for group_name, named_parameters in parameters.items():
                data_rank = ds_groups._get_expert_data_parallel_rank(group_name)
                if not self.checkpoint_engine.is_data_parallel_writer(data_rank):
                    continue
                identity = _ep_checkpoint_identity(group_name)
                state = {
                    name: parameter.detach()
                    for name, parameter in named_parameters.items()
                    if parameter.requires_grad or not arguments["exclude_frozen_parameters"]
                }
                self.checkpoint_engine.save(
                    dict(
                        version=1,
                        **identity,
                        exclude_frozen_parameters=arguments["exclude_frozen_parameters"],
                        module=state,
                    ),
                    str(_ep_checkpoint_path(arguments["save_dir"], arguments["tag"], identity)),
                )
            return result

        return save

    @wraps(original_load)
    def load(self, *args, **kwargs):
        bound = load_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        parameters = _ep_checkpoint_parameters(self.module)
        if len(parameters) == 0:
            return original_load(self, *args, **kwargs)
        if self.zero_optimization_partition_weights():
            raise ValueError("AP expert-parallel checkpoint shards do not support ZeRO stage 3")
        local_state = {}
        for group_name, named_parameters in parameters.items():
            identity = _ep_checkpoint_identity(group_name)
            path = _ep_checkpoint_path(arguments["load_dir"], arguments["tag"], identity)
            if not path.is_file():
                raise FileNotFoundError(f"AP expert-parallel checkpoint is missing its rank-local shard: {path}")
            shard = self.checkpoint_engine.load(str(path), map_location=torch.device("cpu"))
            if (
                not isinstance(shard, dict)
                or type(shard.get("version")) is not int
                or shard["version"] != 1
                or any(shard.get(key) != value for key, value in identity.items())
                or type(shard.get("exclude_frozen_parameters")) is not bool
                or not isinstance(shard.get("module"), dict)
            ):
                raise ValueError(f"AP expert-parallel checkpoint shard metadata does not match: {path}")
            expected_names = {
                name
                for name, parameter in named_parameters.items()
                if parameter.requires_grad or not shard["exclude_frozen_parameters"]
            }
            if set(shard["module"]) != expected_names:
                raise ValueError(f"AP expert-parallel checkpoint parameter coverage does not match: {path}")
            for name, value in shard["module"].items():
                if not isinstance(value, torch.Tensor) or value.shape != named_parameters[name].shape:
                    raise ValueError(f"AP expert-parallel checkpoint tensor shape does not match for {name}: {path}")
            local_state.update(shard["module"])
        self._ap_ep_checkpoint_state = local_state
        try:
            return original_load(*bound.args, **bound.kwargs)
        finally:
            del self._ap_ep_checkpoint_state

    @wraps(original_module_load)
    def load_module(self, *args, **kwargs):
        if not hasattr(self, "_ap_ep_checkpoint_state"):
            return original_module_load(self, *args, **kwargs)
        bound = module_load_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        checkpoint = dict(bound.arguments["checkpoint"])
        checkpoint["module"] = dict(checkpoint["module"])
        checkpoint["module"].update(
            (canonical_parameter_name(name), value) for name, value in self._ap_ep_checkpoint_state.items()
        )
        # DeepSpeed restores frozen fragments after the module, so both must use the local shard.
        if checkpoint.get(FROZEN_PARAM_FRAGMENTS) is not None:
            fragments = dict(checkpoint[FROZEN_PARAM_FRAGMENTS])
            fragments.update(
                (name, value) for name, value in self._ap_ep_checkpoint_state.items() if name in fragments
            )
            checkpoint[FROZEN_PARAM_FRAGMENTS] = fragments
        bound.arguments["checkpoint"] = checkpoint
        return original_module_load(*bound.args, **bound.kwargs)

    engine_type._save_moe_checkpoint = wrap_save(engine_type._save_moe_checkpoint)
    # Frozen expert-only models follow DeepSpeed's ordinary checkpoint path.
    engine_type._save_checkpoint = wrap_save(engine_type._save_checkpoint)
    engine_type._load_checkpoint = load
    engine_type.load_module_state_dict = load_module


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

    _patch_deepspeed_ep_checkpoints(ds_engine.DeepSpeedEngine)
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
    for module in model.modules():
        if getattr(module, "_dss_shard_on_ep", False):
            parallelize_module(
                module,
                device_mesh=ep_mesh,
                parallelize_plan=DeepEPShardParallel(),
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
            if getattr(parameter, "_dss_skip_weight_sync", False):
                replacement._dss_skip_weight_sync = True
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
    for module in model.modules():
        if not getattr(module, "_dss_ep_sharded", False):
            continue
        for parameter in module.parameters(recurse=False):
            if parameter.numel() == 0:
                continue
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


def _gather_ep_weight_to_cpu(
    tensor: torch.Tensor,
    ep_group,
    *,
    is_master: bool,
    max_gather_bytes: int = 64 * 1024 * 1024,
) -> torch.Tensor | None:
    import torch.distributed as dist

    local = tensor.detach().contiguous()
    ep_size = dist.get_world_size(group=ep_group)
    if local.ndim == 0:
        raise ValueError("expert-parallel export requires a tensor sharded along dimension zero")
    if max_gather_bytes < ep_size * local.element_size():
        raise ValueError("max_gather_bytes must hold at least one expert-parallel element per rank")

    full = None
    if is_master:
        full = torch.empty(
            (ep_size * local.shape[0], *local.shape[1:]),
            dtype=local.dtype,
            device="cpu",
        )
    if local.numel() == 0:
        return full
    if ep_size == 1:
        if full is not None:
            full.copy_(local)
        return full

    # Stage bounded, group-ordered chunks on CPU instead of retaining full GPU shards.
    flat = local.view(-1)
    chunk_elements = min(flat.numel(), max_gather_bytes // (ep_size * local.element_size()))
    gathered = torch.empty(ep_size * chunk_elements, dtype=local.dtype, device=local.device)
    full_shards = full.view(ep_size, flat.numel()) if full is not None else None
    use_direct_buffer = local.is_cuda and dist.get_backend(group=ep_group) == "nccl"
    for offset in range(0, flat.numel(), chunk_elements):
        count = min(chunk_elements, flat.numel() - offset)
        outputs = gathered[: ep_size * count]
        inputs = flat[offset : offset + count]
        if use_direct_buffer:
            # NCCL's list gather allocates another packed output; write into the bounded buffer directly.
            dist.all_gather_into_tensor(outputs, inputs, group=ep_group)
        else:
            dist.all_gather(list(outputs.view(ep_size, count).unbind(0)), inputs, group=ep_group)
        if full_shards is not None:
            for rank, shard in enumerate(outputs.view(ep_size, count)):
                full_shards[rank, offset : offset + count].copy_(shard)
    return full


def _gather_ep_shards_to_rank0_host(local: torch.Tensor, ep_group) -> torch.Tensor | None:
    """Gather an EP-sharded tensor to rank 0's host memory with one GPU piece buffer per rank."""
    import torch.distributed as dist

    ranks = dist.get_process_group_ranks(ep_group)
    if 0 not in ranks:
        return None
    is_master = dist.get_rank() == 0
    world, rows = dist.get_world_size(group=ep_group), local.shape[0]
    pieces = local.tensor_split(world)
    full = torch.empty((world * rows, *local.shape[1:]), dtype=local.dtype, device="cpu") if is_master else None
    received = torch.empty_like(pieces[0])
    start = 0
    for piece in pieces:
        for group_rank, rank in enumerate(ranks):
            if rank == 0:
                if full is not None:
                    full[group_rank * rows + start : group_rank * rows + start + len(piece)].copy_(piece)
                continue
            buffer = piece if dist.get_rank() == rank else received[: len(piece)]
            dist.broadcast(buffer, src=rank, group=ep_group)
            if full is not None:
                full[group_rank * rows + start : group_rank * rows + start + len(piece)].copy_(received[: len(piece)])
        start += len(piece)
    return full


def build_iter_full_hf_weights(model: nn.Module):
    model_type = type(model)
    if getattr(model_type, "convert_layer_to_hf", None) is None:
        raise RuntimeError(f"build_iter_full_hf_weights requires {model_type.__name__}.convert_layer_to_hf")
    # Bound from the instance so a family whose layer mapping depends on its config can read it.
    convert_layer_to_hf = model.convert_layer_to_hf

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

    def iterator(*, include_skipped: bool = False):
        is_master = dist.get_rank() == 0
        from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors

        skipped_parameter_ids = {
            id(parameter)
            for module in model.modules()
            if getattr(module, "_dss_skip_weight_sync", False)
            for parameter in module.parameters(recurse=False)
        }
        by_layer: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in named_weight_sync_tensors(model, include_export_buffers=include_skipped):
            if not include_skipped and (
                getattr(tensor, "_dss_skip_weight_sync", False) or id(tensor) in skipped_parameter_ids
            ):
                continue
            hf_name = hf_export_param_name(name)
            if hf_name is not None:
                by_layer.setdefault(layer_index(hf_name), []).append((hf_name, tensor))

        ordered_layers = ([-1] if -1 in by_layer else []) + sorted(index for index in by_layer if index >= 0)
        for index in ordered_layers:
            layer_state: dict[str, torch.Tensor] = {}
            for name, tensor in by_layer[index]:
                if hasattr(tensor, "group_name") and getattr(tensor, "allreduce", True) is False:
                    ep_group = ds_groups._get_expert_parallel_group(tensor.group_name)
                    if getattr(tensor, "_dss_skip_weight_sync", False) or id(tensor) in skipped_parameter_ids:
                        # Only checkpoints export these (Qwen3.8's 95 GiB n-gram table); gathered on a GPU, it OOMs.
                        local = tensor.data.contiguous()
                        full = _gather_ep_shards_to_rank0_host(local, ep_group)
                        if is_master:
                            layer_state[name] = full
                        del full
                        continue
                    full = _gather_ep_weight_to_cpu(tensor, ep_group, is_master=is_master)
                    if is_master:
                        layer_state[name] = full
                elif is_master:
                    layer_state[name] = tensor.data

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

    maybe_observe_model_materialization(model, adapter, config, parallel_dims, ep_mesh)
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
    from arctic_platform.model.weight_export import WeightExportContract
    from arctic_platform.model.weight_export import register_weight_export

    register_weight_export(
        model,
        WeightExportContract(
            hf=adapter.hf_weight_export or build_iter_full_hf_weights,
            vllm=adapter.vllm_weight_export,
        ),
    )
    return model


def prepare_moe_weights(
    adapter: MoEDeepSpeedAdapter,
    *,
    model_name: str,
    optimization_dtype: str,
    attn_implementation: str,
    ep_size: int,
    world_size: int,
    options: Any,
    rank: int,
    local_rank: int,
) -> Path | None:
    """Fill the converted-weight cache that ``load_moe_model`` with the same arguments loads from.

    Runs without a process group, on a meta-device model, so one process per node can convert before the training
    job starts and the distributed load then finds a ready cache. Returns the cache path, or None when the family
    has no conversion step, the checkpoint already matches the model's format, or the model is randomly initialized.
    """
    if adapter.resolve_weight_source is None or adapter.prepare_weight_conversion is None:
        return None
    if world_size % ep_size:
        raise ValueError(f"world_size={world_size} must be divisible by ep_size={ep_size}")
    model_config = adapter.build_model_config(
        model_name,
        ep_size,
        world_size // ep_size,
        optimization_dtype,
        attn_implementation,
        options,
    )
    if model_config.debug.random_init:
        return None
    model = adapter.get_model(
        model_config,
        device=torch.device("meta"),
        dtype=adapter.dtype_map[model_config.optimization_dtype],
    )
    source_path = adapter.resolve_weight_source(model_config)
    return adapter.prepare_weight_conversion(model, model_config, source_path, rank=rank, local_rank=local_rank)


def load_moe_model(
    adapter: MoEDeepSpeedAdapter,
    family_loader: Callable,
    *,
    model_name: str,
    optimization_dtype: str,
    attn_implementation: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    options: Any,
    patch_moe_detection: Callable[[], None] = patch_deepspeed_moe_detection,
    device_mesh_type=DeviceMesh,
) -> nn.Module:
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

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
    ep_backend = options.ep_comm_backend
    if not uses_dispatch_ep(ep_backend):
        raise NotImplementedError(f"EP comm backend must be one of ('deepep', 'uccl'), got {ep_backend!r}.")

    dp_replicate = world_size // ep_size
    model_config = adapter.build_model_config(
        model_name,
        ep_size,
        dp_replicate,
        optimization_dtype,
        attn_implementation,
        options,
    )
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
        fused_cross_entropy=options.fused_cross_entropy,
        tiled_mlp_token_chunk_size=options.tiled_mlp_token_chunk_size,
        sp_size=sp_size,
        sp_group=sp_group,
    )
    maybe_use_fixed_order_row_sum(model, options.model_dump())
    return model
