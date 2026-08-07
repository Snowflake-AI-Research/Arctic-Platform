"""Slimmed model build / weight-load helpers for the carved-out Qwen3.5 path.

Derived from prime-rl's ``trainer/model.py`` but reduced to exactly what the
DSS (DeepSpeed + EP + DeepEP) load path needs:

- ``get_model`` (text-only / custom impl selection)
- ``load_dcp_from_hf`` (HF -> Prime weight conversion cache + DCP load)
- ``apply_ac`` (activation checkpointing)
- ``configure_moe_ep_backend`` (set the DeepEP backend on MoE layers)
- ``_reset_runtime_moe_buffers``
- ``DTYPE_MAP`` and the three Qwen3.5 transformers monkeypatches

All FSDP / torch.compile / LoRA / VLM-training / setup_model paths are dropped.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import cast

# Disable transformers hub kernel interception by default. The `kernels` package, when installed,
# causes transformers to auto-replace modules (e.g. mamba-ssm) with hub kernel versions that may
# have incompatible CUDA requirements.
os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
import torch._dynamo
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PretrainedConfig

from .gpu.activation_offload import install_activation_offload
from .gpu.router_replay_recompute import install_self_router_replay

from .config import ActivationCheckpointConfig, ModelConfig
from .conversion_cache import (
    WEIGHT_CONVERSION_CACHE_SCOPE_ENV,
    conversion_cache_is_node_local,
    conversion_cache_ready,
    ensure_node_local_conversion_cache,
    resolve_conversion_cache_path,
    _write_conversion_cache,
)
from .logging_utils import get_logger
from .models import (
    AutoModelForCausalLMPrimeRL,
    PreTrainedModelPrimeRL,
    get_custom_vlm_cls,
    supports_custom_impl,
)
from .models.layers.checkpointing import (
    get_supported_targets,
    set_selective_activation_checkpointing,
    supports_selective_activation_checkpointing,
)
from .models.layers.moe import LatentMoE, MoE
from .parallel_dims import ParallelDims
from .gpu.router_replay_recompute import install_self_router_replay
from .vlm import get_language_model, is_vlm_architecture
from .weights import load_state_dict, load_state_dict_keys, save_state_dict
from .world import get_world


def _patch_qwen3_5_moe_conversion_mapping():
    """Fix Qwen3.5 MoE conversion mapping incorrectly applying qwen2_moe expert weight splitting."""
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    qwen3_5_text_mapping = get_checkpoint_conversion_mapping("qwen3_5_text")
    if qwen3_5_text_mapping is not None:
        register_checkpoint_conversion_mapping("qwen3_5_moe_text", qwen3_5_text_mapping, overwrite=True)

    register_checkpoint_conversion_mapping("qwen3_5_moe", [], overwrite=True)


def _patch_qwen3_5_text_position_ids():
    """Fix Qwen3.5 passing 3D MRoPE position_ids to decoder layers instead of 2D text_position_ids."""
    import inspect

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextModel
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextModel

    for text_model_cls, decoder_layer_cls in [
        (Qwen3_5TextModel, Qwen3_5DecoderLayer),
        (Qwen3_5MoeTextModel, Qwen3_5MoeDecoderLayer),
    ]:
        source = inspect.getsource(text_model_cls.forward)
        if "decoder_layer" in source and "position_ids=text_position_ids" in source.split("decoder_layer")[-1]:
            continue  # already fixed upstream

        _original_forward = decoder_layer_cls.forward

        def _make_patched_forward(original):
            def _patched_forward(self, hidden_states, position_ids=None, **kwargs):
                if position_ids is not None and position_ids.ndim == 3:
                    position_ids = position_ids[0]
                return original(self, hidden_states, position_ids=position_ids, **kwargs)

            return _patched_forward

        decoder_layer_cls.forward = _make_patched_forward(_original_forward)


def _patch_qwen3_5_linear_attn_varlen():
    """Thread cu_seqlens through Qwen3.5 GatedDeltaNet so packed batches don't
    leak conv/SSM state across sequences."""
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5GatedDeltaNet,
        Qwen3_5TextModel,
        apply_mask_to_padding_states,
    )

    if getattr(Qwen3_5GatedDeltaNet.forward, "_prl_varlen_patched", False):
        return

    _gdn_orig = Qwen3_5GatedDeltaNet.forward

    def _gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, cu_seqlens=None):
        if cu_seqlens is None or cache_params is not None:
            return _gdn_orig(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if self.causal_conv1d_fn is not None:
            seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            seq_idx = torch.repeat_interleave(
                torch.arange(seg_lens.numel(), dtype=torch.int32, device=hidden_states.device),
                seg_lens,
            ).unsqueeze(0)
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        else:
            # Per-segment conv1d so the kernel-1 left pad only draws from within each sequence.
            cu = cu_seqlens.tolist()
            conv_outs = []
            for i in range(len(cu) - 1):
                s, e = cu[i], cu[i + 1]
                if s == e:
                    continue
                conv_outs.append(self.conv1d(mixed_qkv[:, :, s:e])[:, :, : e - s])
            mixed_qkv = F.silu(torch.cat(conv_outs, dim=-1))

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    _gdn_forward._prl_varlen_patched = True
    Qwen3_5GatedDeltaNet.forward = _gdn_forward

    _dec_orig = Qwen3_5DecoderLayer.forward

    def _dec_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cu_seqlens=None,
        **kwargs,
    ):
        if position_ids is not None and position_ids.ndim == 3:
            position_ids = position_ids[0]
        if self.layer_type != "linear_attention":
            return _dec_orig(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    Qwen3_5DecoderLayer.forward = _dec_forward

    _text_orig = Qwen3_5TextModel.forward

    def _text_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        attn_impl = getattr(self.config, "_attn_implementation", None)
        cu_seqlens = None
        if attn_impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4") and position_ids is not None:
            pids = position_ids
            if pids.ndim == 3:
                pids = pids[0]
            flat = pids.view(-1)
            seqlens = torch.cat([flat[0:1], flat[:-1][(flat == 0)[1:]] + 1, flat[-1:] + 1])
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
        kwargs["cu_seqlens"] = cu_seqlens
        return _text_orig(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    Qwen3_5TextModel.forward = _text_forward


# Suppress flash-attention dtype warnings from transformers.modeling_utils since
# mixed precision is handled by the optimizer/DeepSpeed integration.
_transformers_modeling_utils_logger = logging.getLogger("transformers.modeling_utils")
_transformers_modeling_utils_logger.addFilter(
    lambda record: "Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes" not in record.getMessage()
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Increase the torch.compile recompile limit and cache size (carried over from prime-rl).
torch._dynamo.config.recompile_limit = 16  # default: 8


def strip_lora_from_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Drop LoRA adapter keys and unwrap base-layer prefixes.

    On the DSS path no LoRA is applied, so this is a no-op for plain models.
    """
    if not any("lora_" in k or ".base_layer." in k for k in state_dict):
        return state_dict
    cleaned: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        cleaned[key.replace(".base_layer.", ".")] = value
    return cleaned


def is_moe_model(model: nn.Module) -> bool:
    return hasattr(model.config, "num_experts") or hasattr(model.config, "n_routed_experts")


def configure_moe_ep_backend(model: nn.Module, config: ModelConfig) -> None:
    backend = config.ep_comm_backend
    if backend == "deepep":
        from .distributed.deepep import configure_num_sms

        configure_num_sms(config.deepep_num_sms)
    language_model = get_language_model(model)
    for transformer_block in language_model.layers:
        if not isinstance(transformer_block.mlp, (MoE, LatentMoE)):
            continue
        transformer_block.mlp.set_ep_comm_backend(backend)
        transformer_block.mlp.set_deepep_token_chunk_size(config.deepep_token_chunk_size)


def get_model(
    config: ModelConfig, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    logger = get_logger()
    logger.info(
        f"Loading model config (name={config.name}, attn={config.attn}, trust_remote_code={config.trust_remote_code})"
    )

    if "Qwen3.5" in config.name or "qwen3_5" in config.name.lower():
        _patch_qwen3_5_text_position_ids()
        _patch_qwen3_5_moe_conversion_mapping()
        _patch_qwen3_5_linear_attn_varlen()

    model_config = cast(
        PretrainedConfig,
        AutoConfig.from_pretrained(
            config.name, attn_implementation=config.attn, trust_remote_code=config.trust_remote_code
        ),
    )
    model_config.use_cache = False
    is_vlm_arch = is_vlm_architecture(model_config)

    # Fallback Qwen3.5 patch detection from loaded config model_type
    if getattr(model_config, "model_type", "").startswith("qwen3_5_moe"):
        _patch_qwen3_5_text_position_ids()
        _patch_qwen3_5_moe_conversion_mapping()
        _patch_qwen3_5_linear_attn_varlen()
    for subconfig_key in getattr(model_config, "sub_configs", {}):
        subconfig = getattr(model_config, subconfig_key, None)
        if subconfig is not None and hasattr(subconfig, "use_cache"):
            subconfig.use_cache = False
    model_config.use_grouped_mm = config.moe_use_grouped_mm

    # Ensure pad_token_id is set (some models like Qwen3MoE don't have it).
    # In transformers v5, token IDs moved from PretrainedConfig to GenerationConfig.
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

    if config.debug.num_layers is not None:
        # VLM configs nest num_hidden_layers under text_config
        target_config = getattr(model_config, "text_config", model_config)
        num_hidden_layers = min(config.debug.num_layers, target_config.num_hidden_layers)
        logger.warning(
            f"Setting the number of layers to {config.debug.num_layers} in the model config. This means "
            f"{target_config.num_hidden_layers - num_hidden_layers} layers will not be loaded."
        )
        target_config.num_hidden_layers = num_hidden_layers
        # Truncate layer_types too: a strict validator requires num_hidden_layers == len(layer_types).
        if getattr(target_config, "layer_types", None) is not None:
            target_config.layer_types = target_config.layer_types[:num_hidden_layers]

    # Determine the implementation to use
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
        # HF VLM models require torch_dtype; custom PrimeRL models and text Auto models use dtype
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
    # HF standard transformer model
    if "model.rotary_emb.inv_freq" in buffer_names:
        rotary_emb = model.model.rotary_emb
        if hasattr(rotary_emb, "rope_init_fn"):
            rope_init_fn = rotary_emb.rope_init_fn
        else:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_init_fn = (
                ROPE_INIT_FUNCTIONS[rotary_emb.rope_type]
                if rotary_emb.rope_type != "default"
                else rotary_emb.compute_default_rope_parameters
            )
        inv_freq, rotary_emb.attention_scaling = rope_init_fn(rotary_emb.config, rotary_emb.inv_freq.device)
        rotary_emb.inv_freq.copy_(inv_freq)
        if "model.rotary_emb.original_inv_freq" in buffer_names:
            rotary_emb.original_inv_freq.copy_(inv_freq)
    # Gemma3 local rotary emb
    if "model.rotary_emb_local.inv_freq" in buffer_names:
        rotary_emb_local = model.model.rotary_emb_local
        inv_freq_local, rotary_emb_local.attention_scaling = rotary_emb_local.rope_init_fn(
            rotary_emb_local.config, rotary_emb_local.inv_freq.device
        )
        rotary_emb_local.inv_freq.copy_(inv_freq_local)
    # Gemma3 embed_scale (scalar computed from hidden_size)
    if "model.embed_tokens.embed_scale" in buffer_names:
        embed_scale = model.config.hidden_size**0.5
        model.model.embed_tokens.embed_scale.fill_(embed_scale)


def _move_buffers_to_cuda(model: nn.Module, config: ModelConfig) -> None:
    """FSDP CPU offloading only manages parameters, not buffers. Move buffers to CUDA."""
    if not config.fsdp_cpu_offload:
        return
    for _, buffer in model.named_buffers():
        if buffer.device.type == "cpu":
            buffer.data = buffer.data.to("cuda")


def load_dcp_from_hf(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims):
    device = "cpu" if config.fsdp_cpu_offload else "cuda"
    model.to_empty(device=device)
    torch.distributed.barrier()

    def _init_buffers_post_meta():
        if isinstance(model, PreTrainedModelPrimeRL):
            model.init_buffers_post_meta()
        else:
            fix_model_post_empty(model)

    logger = get_logger()
    if config.debug.random_init:
        logger.warning("Randomly initializing model. Skipping loading weights from HF.")
        _init_buffers_post_meta()
        # Fill the (uninitialized) to_empty storage with name-seeded normals. The seed is per-parameter-name,
        # so the model is identical across world size / SP / EP layout -- required for the SP-correctness test
        # to compare each topology against its sp1 reference (see tests/sp/test_sp_topologies.py).
        _random_init_deterministic(model, config)
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

    # Dynamically convert between different weight formats if needed.
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

    # All ranks wait for master rank to finish conversion
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
    # Restore weight tying broken by to_empty() for HF models
    if not isinstance(model, PreTrainedModelPrimeRL) and model.config.tie_word_embeddings:
        model.tie_weights()
    _init_buffers_post_meta()

    _move_buffers_to_cuda(model, config)
    logger.debug(f"Loaded weights using HF DCP in {time.perf_counter() - load_dcp_start_time:.2f} seconds")


_RANDOM_INIT_BASE_SEED = 1234567


def _name_seed(name: str) -> int:
    """Stable 63-bit seed from a parameter name.

    Uses ``hashlib`` (not the builtin ``hash``, which is per-process salted) so the seed is identical across
    ranks and across separate runs/configs.
    """
    hex_digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return (_RANDOM_INIT_BASE_SEED ^ int(hex_digest[:16], 16)) & ((1 << 63) - 1)


def _random_init_deterministic(model: nn.Module, config: ModelConfig) -> None:
    """Fill every float parameter with name-seeded normals, layout-independently. DTensor params fill a
    replicated full tensor and re-shard it, so the gathered weights are independent of the EP/SP degree."""
    init_std = float(getattr(getattr(model, "config", None), "initializer_range", 0.02) or 0.02)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not torch.is_floating_point(param):
                continue
            is_dtensor = isinstance(param, DTensor)
            local_tensor = param.to_local() if hasattr(param, "to_local") else param
            generator = torch.Generator(device=local_tensor.device)
            generator.manual_seed(_name_seed(name))
            if is_dtensor:
                # Same full tensor on every rank -> deterministic per-rank shard.
                full_tensor = torch.empty(
                    tuple(param.shape), dtype=local_tensor.dtype, device=local_tensor.device
                )
                full_tensor.normal_(mean=0.0, std=init_std, generator=generator)
                sharded_tensor = distribute_tensor(full_tensor, param.device_mesh, param.placements)
                local_tensor.copy_(sharded_tensor.to_local())
                del full_tensor, sharded_tensor
            else:
                local_tensor.normal_(mean=0.0, std=init_std, generator=generator)


def _reset_runtime_moe_buffers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (MoE, LatentMoE)) and module.tokens_per_expert.device.type != "meta":
            module.tokens_per_expert.zero_()


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
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
                f"Activation-checkpoint CPU offload (ac_config.offload_config.enabled=True) requires "
                f"ac_config.mode='full', but the active mode is '{ac_config.mode}'. "
                f"mode selects what the backward pass recomputes: 'full' recheckpoints each whole "
                f"transformer block, so the only saved activation is the block input -- a single boundary "
                f"that can be streamed to CPU; 'selective' keeps chosen intermediate activations on GPU and "
                f"has no such offloadable boundary. Both live in the job's training_config.ac_config -- set "
                f"mode='full', or set offload_config.enabled=False to keep mode='{ac_config.mode}'."
            )
        install_activation_offload(
            model,
            keep_last_n=ac_config.offload_config.keep_last_n,
            use_streams=ac_config.offload_config.use_streams,
            tensor_size_threshold=ac_config.offload_config.tensor_size_threshold,
        )
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
                # Install before the checkpoint wrap so only checkpointed blocks capture; see
                # router_replay_recompute.py.
                replay_wrapped_routers += install_self_router_replay(transformer_block)
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
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
        f"Applied activation checkpointing (freq={ac_config.freq}, full_layers={full_layers}, "
        f"router_replay_recompute={ac_config.router_replay_recompute}, "
        f"replay_wrapped_routers={replay_wrapped_routers})"
    )
