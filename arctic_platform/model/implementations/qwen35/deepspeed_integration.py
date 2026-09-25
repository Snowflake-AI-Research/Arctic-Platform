"""DeepSpeed integration for the carved-out Qwen3.5 MoE model.

Builds a Prime-RL-style MoE model with expert parallelism but without FSDP,
with local-tensor parameters that DeepSpeed ZeRO can manage.

Public API:
  ``patch_deepspeed_moe_detection``  -- make DeepSpeed recognize the MoE layers.
  ``load_moe_model_for_deepspeed``   -- build a model ready for ``deepspeed.initialize``.
  ``load_moe_model_for_dss``         -- dss-platform entry (model_name + ep_size).

Caller owns ``deepspeed.init_distributed``, ``torch.cuda.set_device``,
``deepspeed.utils.groups._create_expert_and_data_parallel(ep_size)``, optimizer
construction, and ``deepspeed.initialize``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module

from .config import ActivationCheckpointConfig, DebugModelConfig, ModelConfig
from .config_validation import validate_lm_head_fused_ce_config
from .distributed.expert_parallel import DeepEPExpertParallel
from .logging_utils import get_logger
from .gpu.tiled_mlp import apply_tiled_mlp

from .sequence_parallel import apply_sequence_parallelism
from .model_builder import (
    DTYPE_MAP,
    _reset_runtime_moe_buffers,
    apply_ac,
    configure_moe_ep_backend,
    get_model,
    load_dcp_from_hf,
)
from .models.layers.lm_head import inject_prime_lm_head
from .models.layers.moe import FeedForward, LatentMoE, MoE
from .parallel_dims import ParallelDims
from .hf_vllm_weight_sync import pack_qwen35_gdn_layer
from .vlm import get_language_model


def shared_expert_mlp_forward(feed_forward: FeedForward, hidden_states: torch.Tensor) -> torch.Tensor:
    """Un-tiled dense shared-expert FFN compute (``w2(silu(w1 x) * w3 x)``), run per token shard by TiledMLP."""
    return feed_forward.w2(F.silu(feed_forward.w1(hidden_states)) * feed_forward.w3(hidden_states))


def shared_expert_compute_params(feed_forward: FeedForward) -> list[torch.Tensor]:
    """Weights whose ZeRO grad reduction TiledMLP defers to the last token shard."""
    return [feed_forward.w1.weight, feed_forward.w2.weight, feed_forward.w3.weight]


def patch_deepspeed_moe_detection() -> None:
    """Make DeepSpeed recognize the carved-out MoE wrappers as MoE layers.

    DeepSpeed's `_configure_distributed_model` only sets `has_moe_layers=True`
    when the model contains a `deepspeed.moe.layer.MoE` instance. Without
    this, `expert_data_parallel_group` is passed as `None` to ZeRO/BF16
    optimizers, and gradient reduction crashes when it tries to look up
    the expert-DP group for tagged params (stage_1_and_2.py:1258).

    Class-level monkey-patch — process-wide and idempotent.
    """
    import deepspeed.runtime.engine as ds_engine

    if getattr(ds_engine.DeepSpeedEngine, "_primerl_moe_patched", False):
        return
    original = ds_engine.DeepSpeedEngine._configure_distributed_model

    def patched(self, model):
        original(self, model)
        if not self.has_moe_layers:
            for _, m in self.module.named_modules():
                if isinstance(m, (MoE, LatentMoE)):
                    self.has_moe_layers = True
                    n = getattr(m, "num_experts", None)
                    if n is None and hasattr(m, "experts"):
                        n = getattr(m.experts, "num_experts", None)
                    if n is not None:
                        self.num_experts.append(n)

    ds_engine.DeepSpeedEngine._configure_distributed_model = patched
    ds_engine.DeepSpeedEngine._primerl_moe_patched = True


def _apply_ep_with_mesh(model: nn.Module, config: ModelConfig, ep_mesh: DeviceMesh) -> None:
    """Apply expert parallelism using the provided EP mesh directly.

    Threads DeepSpeed's EP `ProcessGroup` (wrapped via `DeviceMesh.from_group`)
    into the EP `parallelize_module` call.
    """
    if config.ep_comm_backend != "deepep":
        raise NotImplementedError(
            f"Only the 'deepep' EP comm backend is supported, got "
            f"{config.ep_comm_backend!r}."
        )
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        block_mlp = getattr(transformer_block, "mlp", None)
        if block_mlp is not None and isinstance(block_mlp, (MoE, LatentMoE)):
            parallelize_module(
                block_mlp.experts,
                device_mesh=ep_mesh,
                parallelize_plan=DeepEPExpertParallel(),
            )


def _convert_dtensors_to_local(model: nn.Module) -> int:
    """Replace every DTensor parameter with an nn.Parameter wrapping its local shard."""
    num_unwrapped = 0
    for module in model.modules():
        for pname, p in list(module.named_parameters(recurse=False)):
            if not isinstance(p.data, DTensor) and not hasattr(p, "to_local"):
                continue
            local = p.to_local().detach()
            new_param = nn.Parameter(local, requires_grad=p.requires_grad)
            module.register_parameter(pname, new_param)
            num_unwrapped += 1
    return num_unwrapped


def _tag_expert_params_for_deepspeed(model: nn.Module, ep_group_name: str) -> int:
    """Tag every expert parameter so DeepSpeed ZeRO routes its grad on the expert-DP group."""
    count = 0
    for module in model.modules():
        if isinstance(module, (MoE, LatentMoE)):
            for _, p in module.experts.named_parameters(recurse=False):
                p.allreduce = False
                p.group_name = ep_group_name
                count += 1
    return count


def _convert_qwen3_5_moe_layer_to_vllm(
    layer_sd: dict,
    layer_idx: int,
    *,
    layer_prefix: str | None = None,
) -> dict:
    """Rename/pack one Qwen3.5 MoE layer from Prime-RL to vLLM bf16 Triton storage."""
    import torch

    if layer_idx < 0:
        return layer_sd

    prefix = layer_prefix if layer_prefix is not None else f"model.layers.{layer_idx}"

    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key in layer_sd:
        layer_sd[f"{prefix}.mlp.gate.weight"] = layer_sd.pop(router_key)

    w1_key = f"{prefix}.mlp.experts.w1"
    w3_key = f"{prefix}.mlp.experts.w3"
    w2_key = f"{prefix}.mlp.experts.w2"
    if w1_key in layer_sd and w3_key in layer_sd:
        w1 = layer_sd.pop(w1_key)
        w3 = layer_sd.pop(w3_key)
        layer_sd[f"{prefix}.mlp.experts.w13_weight"] = torch.cat([w1, w3], dim=1)
    if w2_key in layer_sd:
        layer_sd[f"{prefix}.mlp.experts.w2_weight"] = layer_sd.pop(w2_key)

    shared_renames = {
        f"{prefix}.shared_expert.w1.weight": f"{prefix}.mlp.shared_expert.gate_proj.weight",
        f"{prefix}.shared_expert.w3.weight": f"{prefix}.mlp.shared_expert.up_proj.weight",
        f"{prefix}.shared_expert.w2.weight": f"{prefix}.mlp.shared_expert.down_proj.weight",
        f"{prefix}.shared_expert_gate.weight": f"{prefix}.mlp.shared_expert_gate.weight",
    }
    for old, new in shared_renames.items():
        if old in layer_sd:
            layer_sd[new] = layer_sd.pop(old)

    pack_qwen35_gdn_layer(layer_sd, prefix)

    for buf_key in (f"{prefix}.mlp.expert_bias", f"{prefix}.mlp.tokens_per_expert"):
        layer_sd.pop(buf_key, None)

    return layer_sd


def _to_vllm_vlm_name(name: str) -> str:
    """Rename a Prime-RL VLM parameter key to vLLM's VLM internal layout."""
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual."):]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("lm_head."):
        return "language_model.lm_head." + name[len("lm_head."):]
    raise RuntimeError(
        f"_to_vllm_vlm_name: no Prime-RL -> vLLM VLM rule for parameter "
        f"name {name!r}. Either Prime-RL grew a new top-level submodule "
        f"or this model is not the Qwen3.5-MoE VLM shape this helper "
        f"was written for. Extend _to_vllm_vlm_name with a new rule."
    )


def _build_iter_full_vllm_weights(model: nn.Module):
    """Iterator yielding ``(vllm_name, full_tensor)`` on rank 0 for weight sync."""
    cls_name = type(model).__name__
    assert cls_name.startswith("Qwen3_5Moe"), (
        f"_build_iter_full_vllm_weights only supports Qwen3.5 MoE today "
        f"(got {cls_name}); add a per-family layer converter and dispatch."
    )

    import deepspeed.utils.groups as ds_groups
    import torch
    import torch.distributed as dist

    is_vlm = bool(getattr(model, "_is_vlm", False))
    layer_prefix_pattern = "model.language_model.layers.{i}" if is_vlm else "model.layers.{i}"

    def _strip_ac_wrapper(name: str) -> str:
        return name.replace("._checkpoint_wrapped_module", "")

    def _layer_idx(name: str) -> int:
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            try:
                return int(parts[2])
            except ValueError:
                pass
        if (
            len(parts) >= 4
            and parts[0] == "model"
            and parts[1] == "language_model"
            and parts[2] == "layers"
        ):
            try:
                return int(parts[3])
            except ValueError:
                pass
        return -1

    def _iter():
        is_master = dist.get_rank() == 0
        by_layer: dict[int, list[tuple[str, nn.Parameter]]] = {}
        for name, param in model.named_parameters():
            by_layer.setdefault(_layer_idx(name), []).append((name, param))

        ordered_layers: list[int] = []
        if -1 in by_layer:
            ordered_layers.append(-1)
        ordered_layers.extend(sorted(k for k in by_layer if k >= 0))

        for layer_idx in ordered_layers:
            layer_sd: dict[str, torch.Tensor] = {}
            for name, param in by_layer[layer_idx]:
                if hasattr(param, "group_name") and getattr(param, "allreduce", True) is False:
                    ep_pg = ds_groups._get_expert_parallel_group(param.group_name)
                    local = param.data.contiguous()
                    shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=ep_pg))]
                    dist.all_gather(shards, local, group=ep_pg)
                    if is_master:
                        layer_sd[name] = torch.cat(shards, dim=0)
                elif is_master:
                    layer_sd[name] = param.data

            if not is_master:
                continue

            layer_sd = {_strip_ac_wrapper(k): v for k, v in layer_sd.items()}

            _convert_qwen3_5_moe_layer_to_vllm(
                layer_sd,
                layer_idx,
                layer_prefix=layer_prefix_pattern.format(i=layer_idx) if layer_idx >= 0 else None,
            )

            for name, tensor in layer_sd.items():
                out_name = _to_vllm_vlm_name(name) if is_vlm else name
                yield out_name, tensor

    return _iter


def _build_iter_full_hf_weights(model: nn.Module):
    """Iterator yielding ``(hf_name, full_tensor)`` on rank 0 for HF-format weight sync."""
    cls = type(model)
    cls_name = cls.__name__
    assert cls_name.startswith("Qwen3_5Moe"), (
        f"_build_iter_full_hf_weights only supports Qwen3.5 MoE today "
        f"(got {cls_name}); add a per-family layer converter and dispatch."
    )
    convert_layer_to_hf = getattr(cls, "convert_layer_to_hf", None)
    if convert_layer_to_hf is None:
        raise RuntimeError(
            f"_build_iter_full_hf_weights requires {cls_name}.convert_layer_to_hf "
            "(Prime-RL classmethod). Ensure the model inherits from "
            "PreTrainedModelPrimeRL."
        )

    import deepspeed.utils.groups as ds_groups
    import torch
    import torch.distributed as dist

    def _strip_ac_wrapper(name: str) -> str:
        return name.replace("._checkpoint_wrapped_module", "")

    def _layer_idx(name: str) -> int:
        parts = name.split(".")
        if (
            len(parts) >= 4
            and parts[0] == "model"
            and parts[1] == "language_model"
            and parts[2] == "layers"
        ):
            try:
                return int(parts[3])
            except ValueError:
                pass
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            try:
                return int(parts[2])
            except ValueError:
                pass
        return -1

    def _iter():
        is_master = dist.get_rank() == 0
        by_layer: dict[int, list[tuple[str, nn.Parameter]]] = {}
        for name, param in model.named_parameters():
            by_layer.setdefault(_layer_idx(name), []).append((name, param))

        ordered_layers: list[int] = []
        if -1 in by_layer:
            ordered_layers.append(-1)
        ordered_layers.extend(sorted(k for k in by_layer if k >= 0))

        for layer_idx in ordered_layers:
            layer_sd: dict[str, torch.Tensor] = {}
            for name, param in by_layer[layer_idx]:
                if hasattr(param, "group_name") and getattr(param, "allreduce", True) is False:
                    ep_pg = ds_groups._get_expert_parallel_group(param.group_name)
                    local = param.data.contiguous()
                    shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=ep_pg))]
                    dist.all_gather(shards, local, group=ep_pg)
                    if is_master:
                        layer_sd[name] = torch.cat(shards, dim=0)
                elif is_master:
                    layer_sd[name] = param.data

            if not is_master:
                continue

            layer_sd = {_strip_ac_wrapper(k): v for k, v in layer_sd.items()}

            convert_layer_to_hf(layer_sd, layer_idx)

            for name, tensor in layer_sd.items():
                yield name, tensor

    return _iter


_validate_lm_head_fused_ce_config = validate_lm_head_fused_ce_config


def _setup_model_local_no_train(
    config: ModelConfig,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    *,
    fused_cross_entropy: bool | str = False,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    """Build a Prime-RL model with EP applied on `ep_mesh` and FSDP skipped.

    Result is ready to be wrapped by an external runtime such as DeepSpeed.
    """
    logger = get_logger()

    model = get_model(config, device=torch.device("meta"), dtype=DTYPE_MAP[config.optimization_dtype])
    configure_moe_ep_backend(model, config)

    lm_head_chunk_size: int | None = None
    if isinstance(config.fused_lm_head_token_chunk_size, int):
        lm_head_chunk_size = config.fused_lm_head_token_chunk_size

    inject_prime_lm_head(
        model,
        chunk_size=lm_head_chunk_size,
        fused_cross_entropy=fused_cross_entropy,
        fp32_lm_head=config.fp32_lm_head,
    )

    if parallel_dims.ep_enabled:
        _apply_ep_with_mesh(model, config, ep_mesh)

    # Install SP before AC wrapping (attention submodules must still be directly addressable).
    if sp_size > 1:
        apply_sequence_parallelism(model, sp_size, sp_group)

    if config.ac is not None:
        apply_ac(model, config.ac)

    load_dcp_from_hf(model, config, parallel_dims)
    _reset_runtime_moe_buffers(model)

    num_unwrapped = _convert_dtensors_to_local(model)
    logger.info(f"Unwrapped {num_unwrapped} DTensor parameters into local tensors")

    return model


def load_moe_model_for_deepspeed(
    model_config: ModelConfig,
    parallel_dims: ParallelDims,
    ep_mesh: DeviceMesh,
    ep_group_name: str,
    *,
    fused_cross_entropy: bool | str = False,
    sp_size: int = 1,
    sp_group=None,
) -> nn.Module:
    """Build a Prime-RL MoE model ready for `deepspeed.initialize`.

    Caller owns DS group init, ``patch_deepspeed_moe_detection``, ``ep_mesh``
    construction, optimizer setup, and the ``deepspeed.initialize`` call.
    """
    logger = get_logger()
    model = _setup_model_local_no_train(
        model_config,
        parallel_dims,
        ep_mesh,
        fused_cross_entropy=fused_cross_entropy,
        sp_size=sp_size,
        sp_group=sp_group,
    )
    num_tagged = _tag_expert_params_for_deepspeed(model, ep_group_name)
    logger.info(f"Tagged {num_tagged} expert parameters with group_name='{ep_group_name}', allreduce=False")
    model._iter_full_vllm_weights = _build_iter_full_vllm_weights(model)
    model._iter_full_hf_weights = _build_iter_full_hf_weights(model)
    return model


def load_moe_model_for_dss(
    *,
    model_name: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    prl_config: dict | None = None,
) -> nn.Module:
    """dss-platform entry: build a Prime-RL MoE model from `model_name` + `ep_size` + a config dict.

    `world_size` is read from `torch.distributed`, the EP `ProcessGroup`
    is fetched from `deepspeed.utils.groups` (so the caller must have
    already run ``ds_groups._create_expert_and_data_parallel(ep_size)``),
    and `patch_deepspeed_moe_detection` is called internally.

    Caller still owns: ``deepspeed.init_distributed()``,
    ``torch.cuda.set_device(local_rank)``,
    ``deepspeed.utils.groups._create_expert_and_data_parallel(ep_size)``,
    optimizer construction, and ``deepspeed.initialize(...)``.

    `prl_config` keys (all optional):
        seq_len, attn, ep_comm_backend, optimization_dtype, reduce_dtype,
        moe_use_grouped_mm, ac_config (dict for ActivationCheckpointConfig),
        fused_lm_head_token_chunk_size, fp32_lm_head, fused_cross_entropy,
        weight_conversion_cache_dir (where to cache the one-time HF<->Prime
        weight conversion; defaults next to the source weights, which fails for
        read-only model mounts).
    """
    import deepspeed.utils.groups as ds_groups
    import torch.distributed as dist

    cfg = prl_config or {}
    _validate_lm_head_fused_ce_config(cfg)
    fused_cross_entropy = cfg.get("fused_cross_entropy", "liger")
    world_size = dist.get_world_size()
    if world_size % ep_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by ep_size={ep_size}")

    patch_deepspeed_moe_detection()

    ep_group_name = f"ep_size_{ep_size}"
    ep_pg = ds_groups._get_expert_parallel_group(ep_group_name)
    ep_mesh = DeviceMesh.from_group(
        group=ep_pg,
        device_type="cuda",
        mesh_dim_names=("ep",),
    )

    assert cfg.get("ep_comm_backend", "deepep") == "deepep", "Only deepep is supported for now"

    dp_replicate = world_size // ep_size
    ac_cfg = cfg.get("ac_config")
    # `debug` is a test-only block (tiny random model: skip the HF weight load, truncate depth) that lets a
    # tiny model exercise the SP/EP/AC paths without the full checkpoint. Production configs never carry it, so
    # the kwarg is omitted and ModelConfig's default no-op DebugModelConfig stands.
    debug_kwargs = {"debug": DebugModelConfig(**cfg["debug"])} if cfg.get("debug") else {}
    model_config = ModelConfig(
        name=model_name,
        # Fall back to the ModelConfig default when prl_config doesn't set it,
        # rather than clobbering the default with None.
        weight_conversion_cache_dir=(
            cfg.get("weight_conversion_cache_dir") or ModelConfig.weight_conversion_cache_dir
        ),
        seq_len=cfg.get("seq_len", 4096),
        attn=cfg.get("attn", "flash_attention_3"),
        ep=ep_size,
        ep_comm_backend="deepep",
        deepep_token_chunk_size=cfg.get("deepep_token_chunk_size"),
        dp_replicate=dp_replicate,
        cp=1,
        impl="custom",
        optimization_dtype=cfg.get("optimization_dtype", "bfloat16"),
        reduce_dtype=cfg.get("reduce_dtype", "float32"),
        moe_use_grouped_mm=cfg.get("moe_use_grouped_mm", True),
        ac=ActivationCheckpointConfig(**ac_cfg) if ac_cfg else None,
        fused_lm_head_token_chunk_size=cfg.get("fused_lm_head_token_chunk_size", "disabled"),
        fp32_lm_head=cfg.get("fp32_lm_head", False),
        **debug_kwargs,
    )
    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=-1,
        cp=1,
        pp=1,
        ep=ep_size,
        world_size=world_size,
    )

    model = load_moe_model_for_deepspeed(
        model_config,
        parallel_dims,
        ep_mesh,
        ep_group_name,
        fused_cross_entropy=fused_cross_entropy,
        sp_size=sp_size,
        sp_group=sp_group,
    )

    # ALST Tiled MLP for the dense shared-expert FFN (opt-in via prime_rl.tiled_mlp_token_chunk_size): shards
    # the token dim to cut the shared-expert activation peak at long sequence lengths. See gpu/tiled_mlp.py.
    tiled_mlp_token_chunk_size = cfg.get("tiled_mlp_token_chunk_size")
    if tiled_mlp_token_chunk_size is not None:
        apply_tiled_mlp(
            model,
            is_target=lambda module: isinstance(module, FeedForward),
            mlp_forward=shared_expert_mlp_forward,
            compute_params=shared_expert_compute_params,
            token_chunk_size=int(tiled_mlp_token_chunk_size),
            logger=get_logger(),
        )

    return model
