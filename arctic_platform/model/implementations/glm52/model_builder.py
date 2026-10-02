"""Slimmed model build / weight-load helpers for the carved-out GLM-5.2 path."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PretrainedConfig

from arctic_platform.model.implementations.moe.conversion_cache import (
    WEIGHT_CONVERSION_CACHE_SCOPE_ENV,
    conversion_cache_is_node_local,
    conversion_cache_ready,
    ensure_node_local_conversion_cache,
    resolve_conversion_cache_path,
    _write_conversion_cache,
)
from arctic_platform.model.implementations.moe.logging_utils import get_logger
from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.layers.checkpointing import (
    get_supported_targets,
    set_selective_activation_checkpointing,
    supports_selective_activation_checkpointing,
)
from arctic_platform.model.implementations.moe.layers.moe import LatentMoE, MoE
from arctic_platform.model.implementations.moe.parallel_dims import ParallelDims
from arctic_platform.model.implementations.moe.vlm import get_language_model, is_vlm_architecture
from arctic_platform.model.implementations.moe.weights import load_state_dict, load_state_dict_keys, save_state_dict
from arctic_platform.model.implementations.moe.world import get_world
from arctic_platform.model.implementations.debug.determinism import CHECKPOINT_PRESERVE_RNG_STATE

from .config import ActivationCheckpointConfig, ModelConfig
from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep
from .models import (
    AutoModelForCausalLMPrimeRL,
    get_custom_vlm_cls,
    supports_custom_impl,
)
from .models.glm_moe_dsa.debug_overrides import (
    apply_glm_moe_dsa_debug_overrides,
    estimate_glm_moe_dsa_params,
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def strip_lora_from_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v for k, v in state_dict.items() if "lora_" not in k}


def configure_moe_ep_backend(model: nn.Module, config: ModelConfig) -> None:
    backend = config.ep_comm_backend
    if uses_dispatch_ep(backend):
        from arctic_platform.model.implementations.moe.distributed.ep_backend import get_ep_comm_module

        get_ep_comm_module(backend).configure_num_sms(config.deepep_num_sms)
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        if not isinstance(transformer_block.mlp, (MoE, LatentMoE)):
            continue
        transformer_block.mlp.set_ep_comm_backend(backend)
        transformer_block.mlp.set_deepep_token_chunk_size(config.deepep_token_chunk_size)


def configure_sparse_mla_backend(config: ModelConfig) -> None:
    from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import (
        set_sparse_mla_backend,
    )

    resolved = set_sparse_mla_backend(config.sparse_mla_backend)
    get_logger().info(
        f"Sparse MLA backend: {resolved} (requested={config.sparse_mla_backend})"
    )


def get_model(
    config: ModelConfig, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    logger = get_logger()
    logger.info(
        f"Loading model config (name={config.name}, attn={config.attn}, trust_remote_code={config.trust_remote_code})"
    )

    model_config = cast(
        PretrainedConfig,
        AutoConfig.from_pretrained(
            config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
        ),
    )
    model_config.use_cache = False
    is_vlm_arch = is_vlm_architecture(model_config)

    for subconfig_key in getattr(model_config, "sub_configs", {}):
        subconfig = getattr(model_config, subconfig_key, None)
        if subconfig is not None and hasattr(subconfig, "use_cache"):
            subconfig.use_cache = False
    model_config.use_grouped_mm = config.moe_use_grouped_mm

    if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
        gen_config = GenerationConfig.from_model_config(model_config)
        pad_token_id = next(
            (
                v
                for v in [gen_config.pad_token_id, gen_config.eos_token_id, getattr(model_config, "eos_token_id", None)]
                if v is not None
            ),
            None,
        )
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]
        model_config.pad_token_id = pad_token_id

    if isinstance(getattr(model_config, "pad_token_id", None), list):
        model_config.pad_token_id = model_config.pad_token_id[0]

    logger.debug(f"Loaded model config ({model_config.to_dict()})")

    target_config = getattr(model_config, "text_config", model_config)
    apply_glm_moe_dsa_debug_overrides(target_config, config.debug, logger=logger)
    is_glm_moe_dsa = getattr(target_config, "model_type", None) == "glm_moe_dsa"
    if is_glm_moe_dsa and (
        config.debug.hidden_size is not None
        or config.debug.num_layers is not None
        or config.debug.random_init
    ):
        est_b = estimate_glm_moe_dsa_params(target_config) / 1e9
        logger.info(
            f"GLM MoE DSA debug model ~{est_b:.2f}B params "
            f"({target_config.num_hidden_layers} layers, hidden={target_config.hidden_size})"
        )

    if config.debug.num_layers is not None and not is_glm_moe_dsa:
        num_hidden_layers = min(config.debug.num_layers, target_config.num_hidden_layers)
        logger.warning(
            f"Setting the number of layers to {config.debug.num_layers} in the model config. "
            f"This means {target_config.num_hidden_layers - num_hidden_layers} layers will not be loaded."
        )
        target_config.num_hidden_layers = num_hidden_layers

    if config.debug.first_k_dense_replace is not None and not is_glm_moe_dsa:
        if hasattr(target_config, "first_k_dense_replace"):
            target_config.first_k_dense_replace = config.debug.first_k_dense_replace

    if config.debug.skip_attention or config.debug.skip_mlp:
        target_config = getattr(model_config, "text_config", model_config)
        if config.debug.skip_attention and config.debug.skip_mlp:
            raise ValueError("Cannot set both debug.skip_attention and debug.skip_mlp")
        if hasattr(target_config, "skip_attention"):
            target_config.skip_attention = config.debug.skip_attention
        if hasattr(target_config, "skip_mlp"):
            target_config.skip_mlp = config.debug.skip_mlp

    custom_vlm_cls = get_custom_vlm_cls(model_config) if is_vlm_arch else None
    if config.impl == "auto":
        if is_vlm_arch:
            impl_to_use = "custom" if custom_vlm_cls is not None else "hf"
        else:
            impl_to_use = "custom" if supports_custom_impl(model_config) else "hf"
        logger.info(f"Auto-selected implementation: {impl_to_use}")
    else:
        impl_to_use = config.impl

    with device:
        if impl_to_use == "custom" and custom_vlm_cls is not None:
            model_cls = custom_vlm_cls
        elif is_vlm_arch:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        else:
            match impl_to_use:
                case "hf":
                    model_cls = AutoModelForCausalLM
                case "custom":
                    model_cls = AutoModelForCausalLMPrimeRL

        load_model_start_time = time.perf_counter()
        use_torch_dtype = is_vlm_arch and model_cls is not custom_vlm_cls
        dtype_kwarg = {"torch_dtype": dtype} if use_torch_dtype else {"dtype": dtype}
        if device == torch.device("meta"):
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to meta device")
            model = model_cls.from_config(model_config, trust_remote_code=config.trust_remote_code, **dtype_kwarg)
        else:
            logger.info(f"Loading model {config.name} using {model_cls.__name__} to CPU")
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=config.name,
                config=model_config,
                trust_remote_code=config.trust_remote_code,
                **dtype_kwarg,
            )
        logger.debug(f"Loaded model {config.name} in {time.perf_counter() - load_model_start_time:.2f} seconds")

    assert model.lm_head.weight.dtype == dtype, (
        f"LM head dtype wasnt loaded correctly {model.lm_head.weight.dtype} != {dtype}"
    )
    return model


def fix_model_post_empty(model: nn.Module):
    buffer_names = [name for name, _ in model.named_buffers()]
    if "model.rotary_emb.inv_freq" in buffer_names:
        rotary_emb = model.model.rotary_emb
        if hasattr(rotary_emb, "rope_init_fn"):
            rope_init_fn = rotary_emb.rope_init_fn
        else:
            rope_init_fn = rotary_emb._init_rope
        inv_freq, _ = rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device if rotary_emb.inv_freq is not None else None)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


def _init_buffers_post_meta(model: nn.Module) -> None:
    if isinstance(model, PreTrainedModelPrimeRL):
        model.init_buffers_post_meta()
    else:
        fix_model_post_empty(model)


def _move_buffers_to_cuda(model: nn.Module, config: ModelConfig) -> None:
    if not config.fsdp_cpu_offload:
        model.to("cuda")


def _init_random_moe_weights(model: nn.Module, init_std: float) -> None:
    buffer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = 0
    for module in model.modules():
        if isinstance(module, (MoE, LatentMoE)):
            module.init_weights(init_std, buffer_device)
            count += 1
    if count:
        get_logger().info(f"Initialized {count} MoE/LatentMoE modules after random_init")


def load_dcp_from_hf(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    device = "cpu" if config.fsdp_cpu_offload else "cuda"
    model.to_empty(device=device)
    torch.distributed.barrier()

    logger = get_logger()
    if config.debug.random_init:
        init_seed = config.debug.init_seed
        logger.warning(
            f"Randomly initializing model with init_weights(seed={init_seed}). Skipping loading weights from HF."
        )
        _init_buffers_post_meta(model)
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
        model.init_weights()
        _init_random_moe_weights(model, init_std=0.02)
        torch.distributed.barrier()
        _move_buffers_to_cuda(model, config)
        return

    if not Path(config.name).exists():
        snapshot_path = Path(snapshot_download(repo_id=config.name, repo_type="model"))
    else:
        logger.info(
            f"Loading model weights from path {config.name}, skipping snapshot download. If this is not expected, "
            f"please remove the directory {config.name} and run again"
        )
        snapshot_path = Path(config.name)

    conversion = None
    if isinstance(model, PreTrainedModelPrimeRL):
        snapshot_keys = dict.fromkeys(load_state_dict_keys(snapshot_path))
        model_keys = dict.fromkeys(model.state_dict().keys())

        source_path = snapshot_path
        if model.is_hf_state_dict(snapshot_keys) and model.is_prime_state_dict(model_keys):
            conversion = ("prime", model.convert_to_prime, "HF", "PrimeRL")
        elif model.is_prime_state_dict(snapshot_keys) and model.is_hf_state_dict(model_keys):
            conversion = ("hf", model.convert_to_hf, "PrimeRL", "HF")

        if conversion is not None:
            fmt, convert_fn, src_fmt, dst_fmt = conversion
            logger.warning(
                f"Found {src_fmt} weight format in snapshot state dict and {dst_fmt} weight format in model "
                "state dict. Trying to auto-convert..."
            )
            snapshot_path = resolve_conversion_cache_path(config, source_path, fmt)
            node_local_cache = conversion_cache_is_node_local(snapshot_path)
            world = get_world()
            if node_local_cache:
                ensure_node_local_conversion_cache(
                    source_path,
                    snapshot_path,
                    convert_fn,
                    src_fmt,
                    dst_fmt,
                    load_state_dict,
                    save_state_dict,
                    rank=world.rank,
                    local_rank=world.local_rank,
                )
            elif not conversion_cache_ready(snapshot_path) and world.is_master:
                _write_conversion_cache(
                    source_path,
                    snapshot_path,
                    convert_fn,
                    src_fmt,
                    dst_fmt,
                    load_state_dict,
                    save_state_dict,
                    rank=world.rank,
                    local_rank=world.local_rank,
                )

    torch.distributed.barrier()
    if conversion is not None and not conversion_cache_ready(snapshot_path):
        world = get_world()
        raise FileNotFoundError(
            "Converted weight cache is missing or incomplete after conversion barrier: "
            f"path={snapshot_path}, rank={world.rank}, local_rank={world.local_rank}, "
            f"scope={os.environ.get(WEIGHT_CONVERSION_CACHE_SCOPE_ENV, 'auto')}"
        )

    logger.info(f"Loading weights using HF DCP from {snapshot_path}")
    load_dcp_start_time = time.perf_counter()
    state_dict = model.state_dict()
    state_dict = strip_lora_from_state_dict(state_dict)
    if model.config.tie_word_embeddings:
        del state_dict["lm_head.weight"]
    dcp_load(
        state_dict,
        storage_reader=HuggingFaceStorageReader(path=snapshot_path.as_posix()),
    )
    if not isinstance(model, PreTrainedModelPrimeRL) and model.config.tie_word_embeddings:
        model.tie_weights()
    _init_buffers_post_meta(model)

    _move_buffers_to_cuda(model, config)
    logger.debug(f"Loaded weights using HF DCP in {time.perf_counter() - load_dcp_start_time:.2f} seconds")


def _reset_runtime_moe_buffers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (MoE, LatentMoE)) and module.tokens_per_expert.device.type != "meta":
            module.tokens_per_expert.zero_()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.model.implementations.gpu.activation_offload import install_activation_offload
    from arctic_platform.model.implementations.gpu.router_replay_recompute import install_self_router_replay

    logger = get_logger()
    language_model = get_language_model(model)
    target_list = sorted(frozenset(ac_config.targets))
    selective_layers = 0
    full_layers = 0
    replay_wrapped_routers = 0
    fallback_layer_types: set[str] = set()
    model_supported_targets: set[str] = set()

    if ac_config.offload_config.enabled:
        if ac_config.mode == "selective":
            raise ValueError(
                "Activation-checkpoint CPU offload (ac_config.offload_config.enabled=True) requires "
                f"ac_config.mode='full', but the active mode is '{ac_config.mode}'."
            )
        install_activation_offload(model, config=ac_config.offload_config)
        logger.info(
            "Activation CPU offload enabled (saved-tensor hooks, "
            f"keep_last_n={ac_config.offload_config.keep_last_n}, "
            f"streams={ac_config.offload_config.use_streams}, "
            f"tensor_size_threshold={ac_config.offload_config.tensor_size_threshold})"
        )

    for layer_id, (layer_name, transformer_block) in enumerate(language_model.layers.named_children()):
        if layer_id % ac_config.freq != 0:
            continue

        if ac_config.mode == "selective" and supports_selective_activation_checkpointing(transformer_block):
            model_supported_targets.update(get_supported_targets(transformer_block))
            set_selective_activation_checkpointing(transformer_block, target_list)
            selective_layers += 1
        else:
            if ac_config.mode == "selective":
                fallback_layer_types.add(type(transformer_block).__name__)
            if ac_config.router_replay_recompute:
                replay_wrapped_routers += install_self_router_replay(transformer_block)
            transformer_block = checkpoint_wrapper(
                transformer_block, preserve_rng_state=CHECKPOINT_PRESERVE_RNG_STATE
            )
            full_layers += 1

        language_model.layers.register_module(layer_name, transformer_block)

    if ac_config.mode == "selective":
        unsupported_targets = frozenset(target_list) - model_supported_targets
        if unsupported_targets:
            raise ValueError(
                f"Selective activation checkpoint targets {sorted(unsupported_targets)} are not supported "
                f"by the selected model layers. Supported targets across the model: {sorted(model_supported_targets)}"
            )
        if fallback_layer_types:
            logger.warning(
                "Selective activation checkpointing is not supported for layer types "
                f"{sorted(fallback_layer_types)}; falling back to full checkpointing for those layers."
            )
        logger.info(
            "Applied selective activation checkpointing "
            f"(freq={ac_config.freq}, targets={target_list}, selective_layers={selective_layers}, "
            f"full_fallback_layers={full_layers})"
        )
        return

    logger.info(
        f"Applied activation checkpointing (freq={ac_config.freq}, "
        f"router_replay_recompute={ac_config.router_replay_recompute}, "
        f"replay_wrapped_routers={replay_wrapped_routers})"
    )
