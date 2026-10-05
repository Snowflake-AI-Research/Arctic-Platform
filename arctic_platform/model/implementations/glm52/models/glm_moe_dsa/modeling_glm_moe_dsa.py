import warnings
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg

from arctic_platform.model.implementations.gpu.lm_head import inherit_lm_head_target_validation
from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
    _index_cache_skip_topk,
)
from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.converting_glm_moe_dsa import (
    convert_hf_layer_to_tt,
    convert_hf_to_tt_moe,
    convert_tt_layer_to_hf,
    convert_tt_layer_to_vllm_kernel,
    convert_tt_to_hf_moe,
)
from arctic_platform.model.implementations.glm52.models.fp8 import fp8_weight_block_size
from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention, SparseMlaAttentionArgs
from arctic_platform.model.implementations.moe.layers.lm_head import PrimeLmOutput
from arctic_platform.model.implementations.glm52.models.layers.mlp import MLP, MLPConfig
from arctic_platform.model.implementations.moe.layers.moe import MoE, MoEArgs
from arctic_platform.model.implementations.glm52.models.layers.norms import RMSNorm, RMSNormConfig
from arctic_platform.model.implementations.glm52.models.layers.rotary_emb import (
    RotaryEmbedding,
    RotaryEmbeddingConfig,
)
from arctic_platform.model.implementations.glm52.utils.cp import gather_for_cp, shard_for_cp


def _sparse_mla_attention_args(config: GlmMoeDsaConfig, layer_idx: int) -> SparseMlaAttentionArgs:
    if config.q_lora_rank is None:
        raise ValueError("Sparse MLA attention requires q_lora_rank to be set")
    return SparseMlaAttentionArgs(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=config.q_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_head_dim=config.qk_head_dim,
        v_head_dim=config.v_head_dim,
        attention_bias=config.attention_bias,
        rms_norm_eps=config.rms_norm_eps,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        use_index_cache=getattr(config, "use_index_cache", False),
        skip_topk=_index_cache_skip_topk(config, layer_idx),
        fp8_block_size=fp8_weight_block_size(config),
    )


class GlmMoeDsaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.skip_attention = config.skip_attention
        self.skip_mlp = config.skip_mlp
        self.self_attn = GlmMoeDsaAttention(_sparse_mla_attention_args(config, layer_idx))

        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            score_func="sigmoid",
            route_norm=config.norm_topk_prob,
            route_scale=config.routed_scaling_factor,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=1e-3,
            use_grouped_mm=config.use_grouped_mm,
            fp8_block_size=fp8_weight_block_size(config),
        )
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_act=config.hidden_act,
            bias=False,
            fp8_block_size=fp8_weight_block_size(config),
        )

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = MoE(moe_args, dim=config.hidden_size, hidden_dim=config.moe_intermediate_size)
        else:
            self.mlp = MLP(mlp_config)

        self.input_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    @property
    def cp_enabled(self) -> bool:
        return hasattr(self, "_cp_group") and hasattr(self, "_cp_rank") and hasattr(self, "_cp_world_size")

    def shard_to_cp(self, t: torch.Tensor) -> torch.Tensor:
        if not self.cp_enabled:
            return t

        return shard_for_cp(t, self._cp_rank, self._cp_world_size)

    def gather_for_cp(self, t: torch.Tensor) -> torch.Tensor:
        if not self.cp_enabled:
            return t

        return gather_for_cp(t, self._cp_group)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        ks: Optional[torch.Tensor] = None,
        ke: Optional[torch.Tensor] = None,
        cached_indices: Optional[torch.Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.skip_attention:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.gather_for_cp(hidden_states)
            hidden_states, cached_indices = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                ks=ks,
                ke=ke,
                cached_indices=cached_indices,
            )
            hidden_states = self.shard_to_cp(hidden_states)
            hidden_states = residual + hidden_states

        if not self.skip_mlp:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, routed_experts=routed_experts)
            hidden_states = residual + hidden_states
        return hidden_states, cached_indices


@auto_docstring
class GlmMoeDsaPreTrainedModel(PreTrainedModelPrimeRL):
    config: GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GlmMoeDsaDecoderLayer,
    }

    def _init_weights(self, module):
        if isinstance(module, RotaryEmbedding):
            inv_freq, module.attention_scaling = module.rope_init_fn(
                module.config, module.inv_freq.device if module.inv_freq is not None else None
            )
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            return
        super()._init_weights(module)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(
            "mlp.experts.1.up_proj" in name
            or "mlp.experts.0.gate_proj" in name
            or "mlp.experts.gate_up_proj" in name
            for name in state_dict.keys()
        )

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.w1" in module_name for module_name in state_dict.keys())

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_tt_to_hf_moe(state_dict)
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        convert_hf_to_tt_moe(state_dict)
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_tt_layer_to_hf(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        convert_hf_layer_to_tt(state_dict, layer_idx)
        return state_dict

    @classmethod
    def convert_layer_to_vllm_kernel(
        cls, state_dict: dict[str, Tensor], layer_idx: int, quantize_fp8: bool = False
    ) -> dict[str, Tensor]:
        return convert_tt_layer_to_vllm_kernel(state_dict, layer_idx, quantize_fp8=quantize_fp8)


@auto_docstring
class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmMoeDsaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(RMSNormConfig(hidden_size=config.hidden_size, eps=config.rms_norm_eps))

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default") if isinstance(rope_parameters, dict) else "default"
        rotary_config = RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_type,
            model_config=config,
        )
        self.rotary_emb = RotaryEmbedding(rotary_config)
        self.gradient_checkpointing = False
        self.pp_enabled = False
        self.pp_layer_offset = 0
        self.pp_layer_offsets: list[int] | None = None
        self.layer_chunks: nn.ModuleList | None = None
        self.pp_runtime = None

        self.post_init()

    def _context_parallel_state(self) -> tuple[dist.ProcessGroup | None, int, int]:
        if self.layer_chunks is not None and len(self.layer_chunks) > 0 and len(self.layer_chunks[0]) > 0:
            layer = self.layer_chunks[0][0]
        elif len(self.layers) > 0:
            layer = self.layers[0]
        else:
            return None, 0, 1

        return getattr(layer, "_cp_group", None), getattr(layer, "_cp_rank", 0), getattr(layer, "_cp_world_size", 1)

    def _pp_stage_bounds(self, pp) -> tuple[bool, bool, nn.ModuleList, int]:
        if pp is None:
            return True, True, self.layers, 0
        if pp.is_interleaved:
            chunk_id = pp.active_chunk_id
            return (
                pp.is_first_virtual(chunk_id),
                pp.is_last_virtual(chunk_id),
                self.layer_chunks[chunk_id],
                self.pp_layer_offsets[chunk_id],
            )
        return pp.is_first, pp.is_last, self.layers, self.pp_layer_offset

    def _gather_position_ids_for_cp(
        self,
        position_ids: torch.LongTensor,
        cp_group: dist.ProcessGroup,
        cp_world_size: int,
    ) -> torch.LongTensor:
        gathered_position_ids = [torch.empty_like(position_ids) for _ in range(cp_world_size)]
        dist.all_gather(gathered_position_ids, position_ids.contiguous(), group=cp_group)
        return torch.cat(gathered_position_ids, dim=1)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: object,
    ) -> BaseModelOutputWithPast:
        """
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        if use_cache:
            raise ValueError("use_cache is not supported for custom glm_moe_dsa for now")
        pp = self.pp_runtime
        is_first_stage, is_last_stage, active_layers, layer_offset = self._pp_stage_bounds(pp)

        if pp is not None and not is_first_stage:
            input_ids = None
            inputs_embeds = None

        if pp is not None and not is_first_stage:
            if input_ids is not None or inputs_embeds is not None:
                raise ValueError("Non-first pipeline stages must not pass input_ids or inputs_embeds")
        elif (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.embed_tokens is not None and is_first_stage:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None and is_first_stage:
            raise ValueError("inputs_embeds required when embed_tokens is not available on this pipeline stage")

        if pp is not None and inputs_embeds is not None:
            pp.activation_shape = inputs_embeds.shape
            if not pp.use_wavefront_comm:
                pp.activation_dtype = inputs_embeds.dtype

        cp_group, _, cp_world_size = self._context_parallel_state()
        if cp_group is not None and cp_world_size > 1:
            position_ids_for_attn = self._gather_position_ids_for_cp(position_ids, cp_group, cp_world_size)
        else:
            position_ids_for_attn = position_ids

        flat_position_ids = position_ids_for_attn.view(-1)
        S = flat_position_ids.shape[0]

        if pp is not None and not is_first_stage:
            if pp.activation_shape is None:
                batch, seq = position_ids.shape[:2]
                pp.activation_shape = torch.Size((batch, seq, self.config.hidden_size))
            if pp.activation_dtype is None:
                if self.embed_tokens is not None:
                    pp.activation_dtype = self.embed_tokens.weight.dtype
                else:
                    pp.activation_dtype = torch.bfloat16
            if pp.use_wavefront_comm:
                if pp.stage_input is None:
                    raise RuntimeError("1F1B runner must set pp.stage_input before forward on non-first stages.")
                hidden_states = pp.stage_input
            else:
                hidden_states = pp.recv_activation()
            pp.stage_input = hidden_states
        else:
            hidden_states = inputs_embeds
            if pp is not None:
                pp.stage_input = hidden_states
                if hidden_states is not None and not hidden_states.is_leaf:
                    hidden_states.retain_grad()

        if self.config.skip_attention:
            ks = ke = position_embeddings = None
        else:
            ks = torch.arange(S, dtype=torch.int32, device=flat_position_ids.device) - flat_position_ids.to(
                torch.int32
            )
            ke = torch.arange(1, S + 1, dtype=torch.int32, device=flat_position_ids.device)
            position_embeddings = self.rotary_emb(hidden_states, position_ids_for_attn)

        cached_indices = None
        use_index_cache = getattr(self.config, "use_index_cache", False)
        for local_idx, decoder_layer in enumerate(active_layers):
            global_layer_idx = layer_offset + local_idx
            routed_experts_layer = (
                routed_experts[:, :, global_layer_idx, :] if routed_experts is not None else None
            )
            hidden_states, next_cached_indices = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                ks=ks,
                ke=ke,
                cached_indices=cached_indices,
                routed_experts=routed_experts_layer,
            )
            cached_indices = next_cached_indices if use_index_cache else None

        if pp is not None:
            pp.stage_output = hidden_states
            if not is_last_stage:
                pp.activation_dtype = hidden_states.dtype
                if not pp.use_wavefront_comm:
                    pp.send_activation(hidden_states)
                return BaseModelOutputWithPast(last_hidden_state=hidden_states)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        warnings.warn("GlmMoeDsaForCausalLM is experimental, higher trainer<->inference KL mismatch may be observed.")
        warnings.warn("`model.attn` is ignored, GlmMoeDsa uses only sparse attention.")

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        routed_experts: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        routed_experts (`torch.LongTensor` of shape `(batch_size, sequence_length, num_hidden_layers, num_experts_per_tok)`, *optional*):
            Routed experts for each token in the sequence. Only used for router replay.
        """
        if use_cache:
            raise ValueError("use_cache is not supported for custom glm_moe_dsa for now")
        if past_key_values is not None:
            raise ValueError("past_key_values is not supported for custom glm_moe_dsa for now")

        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            elif input_ids is not None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            else:
                raise ValueError("position_ids cannot be inferred without input_ids or inputs_embeds")

        pp = getattr(self.model, "pp_runtime", None) or getattr(self, "pp_runtime", None)
        if pp is not None and pp.is_interleaved:
            is_first_stage = pp.is_first_virtual(pp.active_chunk_id)
            is_last_stage = pp.is_last_virtual(pp.active_chunk_id)
        else:
            is_first_stage = pp is None or pp.is_first
            is_last_stage = pp is None or pp.is_last

        model_input_ids = input_ids if is_first_stage else None
        model_inputs_embeds = inputs_embeds if is_first_stage else None
        if pp is not None and not is_first_stage:
            # Non-first stages only need position_ids (and optional routed_experts) for local layers.
            batch, seq = position_ids.shape
            if pp.activation_shape is None:
                pp.activation_shape = torch.Size((batch, seq, self.config.hidden_size))
                pp.activation_dtype = self.model.embed_tokens.weight.dtype if self.model.embed_tokens else torch.bfloat16

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=model_input_ids,
            position_ids=position_ids,
            inputs_embeds=model_inputs_embeds,
            routed_experts=routed_experts,
        )

        if pp is not None and not is_last_stage:
            return PrimeLmOutput(logits=None, logprobs=None, entropy=None)

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            inherit_lm_head_target_validation(labels, labels[:, slice_indices])
            if labels is not None
            else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
        )

    def init_buffers_post_meta(self):
        buffer_names = [name for name, _ in self.named_buffers()]
        if "model.rotary_emb.inv_freq" in buffer_names:
            rotary_emb = self.model.rotary_emb
            inv_freq, rotary_emb.attention_scaling = rotary_emb.rope_init_fn(
                rotary_emb.config, rotary_emb.inv_freq.device
            )
            rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)


__all__ = ["GlmMoeDsaConfig", "GlmMoeDsaPreTrainedModel", "GlmMoeDsaModel", "GlmMoeDsaForCausalLM"]
