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

"""
Qwen3-specific model patcher.

Patches Qwen3Model and Qwen3ForCausalLM forward methods to handle
deduplication at the model level.
"""

import inspect
import os
import sys

# 2nd half of code
import threading
import warnings
from functools import partial
from typing import List
from typing import Optional
from typing import Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast as _BaseModelOutputWithPast
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from arctic_platform.rl.utils.debug import pr0
from arctic_platform.rl.utils.debug import see_memory_usage
from arctic_platform.rl.zorro_train.module_patcher import ModuleReconstructionPatcher
from arctic_platform.rl.zorro_train.qwen_attention_patcher import QwenAttentionOncePatcher
from arctic_platform.rl.zorro_train.qwen_attention_patcher import QwenAttentionPatcher
from arctic_platform.rl.zorro_train.zorro_train import ReconstructionInfo
from arctic_platform.rl.zorro_train.zorro_train import ZoRRoTrain

# Global debug object for storing baseline/patched tensors
debug_object = {"baseline": None, "patched": None}

SUPPORTED_MODEL_TYPES = {
    "qwen3",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_next",
}

MODEL_TYPE_ALIASES = {
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5_moe",
}

MODEL_TYPE_TO_TEXT_BACKBONE_CLASS = {
    "qwen3": "Qwen3Model",
    "qwen3_moe": "Qwen3MoeModel",
    "qwen3_5": "Qwen3_5TextModel",
    "qwen3_5_moe": "Qwen3_5MoeTextModel",
    "qwen3_next": "Qwen3NextModel",
}


def _normalize_model_type(model_type: str) -> str:
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


def get_supported_model_type(model) -> str:
    """Return normalized ``config.model_type`` for ZoRRO-supported Qwen model families."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError(f"Model {type(model).__name__} does not expose config.model_type")

    normalized = _normalize_model_type(model_type)
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type}. Supported model types: {sorted(SUPPORTED_MODEL_TYPES)} "
            f"(aliases: {sorted(MODEL_TYPE_ALIASES.keys())})"
        )
    return normalized


def _init_dynamic_cache(dynamic_cache_cls, config):
    """Initialize DynamicCache across transformers versions with signature differences."""
    if dynamic_cache_cls is None:
        return None
    try:
        sig = inspect.signature(dynamic_cache_cls)
        if "config" in sig.parameters:
            return dynamic_cache_cls(config=config)
    except (TypeError, ValueError):
        pass
    return dynamic_cache_cls()


def _get_text_backbone(causal_lm_model):
    backbone = causal_lm_model.model
    return getattr(backbone, "language_model", backbone)


def _update_linear_attn_mask(module, attention_mask, past_key_values, cache_position):
    update_fn = getattr(module, "_update_linear_attn_mask", None)
    if update_fn is None:
        return None

    # transformers changed this model method's 2nd positional arg: recent versions take ``past_key_values``,
    # transformers <= 4.57 takes ``cache_position``.
    try:
        if "past_key_values" in inspect.signature(update_fn).parameters:
            return update_fn(attention_mask, past_key_values)
    except (TypeError, ValueError):
        pass
    return update_fn(attention_mask, cache_position)


def _build_mask_kwargs(create_mask_fn, config, inputs_embeds, attention_mask, cache_position, past_key_values, position_ids):
    """Build kwargs for ``create_causal_mask`` / ``create_sliding_window_causal_mask`` using the embeds keyword the
    ``inputs_embeds`` on recent versions, ``input_embeds`` on transformers <= 4.57."""
    mask_kwargs = {
        "config": config,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    try:
        params = inspect.signature(create_mask_fn).parameters
        embeds_key = "input_embeds" if ("input_embeds" in params and "inputs_embeds" not in params) else "inputs_embeds"
    except (TypeError, ValueError):
        embeds_key = "inputs_embeds"
    mask_kwargs[embeds_key] = inputs_embeds
    return mask_kwargs


try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


class DedupActorWrapper:
    """
    Wrapper that automatically handles deduplication patching for actor_module calls.

    This eliminates the need to manually check for deduplication and create patchers
    at each call site. Instead, the wrapper handles it transparently.
    """

    def __init__(self, actor_module, micro_batch, config):
        """
        Args:
            actor_module: The model to wrap
            micro_batch: The micro-batch dict that may contain reconstruction_info
            config: Config object with dedup settings
        """
        self.actor_module = actor_module
        self.micro_batch = micro_batch
        self.config = config
        self.patcher = None
        self.should_patch = False

        # Check if deduplication is needed
        use_dedup = "reconstruction_info" in micro_batch

        if use_dedup:
            # Determine if we need to create a patcher based on training mode
            # During inference (_forward_micro_batch for log_prob), create patcher
            # During training (update_policy), we always need the patcher
            self.should_patch = True

        # pr0(f"{self.should_patch=}")
        # pr0(micro_batch)

    def __enter__(self):
        """Enter the context and apply patching if needed."""

        # import ipdb; ipdb.set_trace()
        if self.should_patch:
            from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelPatcher

            self.patcher = Qwen3ModelPatcher(
                model=self.actor_module,
                reconstruction_info=self.micro_batch["reconstruction_info"],
                use_split_attention=True,
            )
            self.patcher.__enter__()

            assert self.patcher.is_model_patched, "Deduplication is not supported without patching the model"

            # avoid computing logits on full sequence, just extract the response part
            def lm_head_new_code_path_fn(hidden_states):
                hidden_states_extracted = ZoRRoTrain.extract_unpadded_responses_from_deduped_packed_ids(
                    hidden_states.squeeze(0), self.micro_batch["reconstruction_info"], offset=-1
                )
                return self.actor_module.lm_head_old_forward(hidden_states_extracted).unsqueeze(0)

            self.actor_module.lm_head_old_forward = self.actor_module.lm_head.forward
            self.actor_module.lm_head.forward = lm_head_new_code_path_fn

        return self.actor_module

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and clean up patcher if it was created."""
        if self.should_patch:
            if hasattr(self.actor_module, "lm_head_old_forward"):
                self.actor_module.lm_head.forward = self.actor_module.lm_head_old_forward
            self.patcher.__exit__(exc_type, exc_val, exc_tb)
        return False


class Qwen3ModelOncePatcher:
    """
    Qwen3-specific model patcher that is done once in a model lifetime (never unpatched), unlike Qwen3ModelPatcher that can restore the original

    Handles deduplication/reconstruction for Qwen3Model and Qwen3ForCausalLM.

    Flow:
    1. Input: deduplicated input_ids [1, total_tokens]
    2. After embedding: reconstruct to [batch_size, seq_len, hidden_dim]
    3. Before decoder layers: deduplicate to [1, total_tokens, hidden_dim]
    4. Inside decoder layers: stays deduplicated (attention handles reconstruction)
    5. After decoder layers: reconstruct to [batch_size, seq_len, hidden_dim] for logits
    """

    def __init__(
        self,
        model,
        response_len,
        max_token_len,
        rollout_n,
        temperature,
        logits_optimization,
        world_size,
        logits_optimization_peak_mem_size_in_gib=4,
        logits_compute_from_fp32_inputs=False,
        logits_compute_in_fp32=False,
        use_unpad=True,
        patch_with_local=False,
        use_split_attention=True,
    ):
        """
        Args:
            model: Qwen3Model or Qwen3ForCausalLM instance
            response_len:
            max_token_len:
            rollout_n:
            logits_optimization: one of "none" | "memory" | "compute" (see the logprob/entropy dispatch in the
                patched causal-lm forward).
            logits_optimization_peak_mem_size_in_gib: peak memory overhead budget (GiB) used to size the
                chunks/shards/tiles for the "memory" and "compute" modes. Ignored for "none".
            logits_compute_from_fp32_inputs: if True, upcast the LM-head input to fp32 so the logits projection
                (and the logprob/entropy math) runs in fp32.
            logits_compute_in_fp32: if True, upcast the produced logits to fp32 before the logprob/entropy math
                consumes them.
            use_unpad:
            world_size:
            patch_with_local: If True, use local (baseline) implementation
            use_split_attention: If True, use split attention (2 calls: prompt-to-prompt + response-to-full).
                                 If False, use standard approach (1 call: full attention on reconstructed Q/K/V).
        """
        self.model = model
        self.model_type = get_supported_model_type(model)
        self.patch_target_class_name = MODEL_TYPE_TO_TEXT_BACKBONE_CLASS[self.model_type]
        self.causal_lm_class_name = type(model).__name__

        self.response_len = response_len
        self.max_token_len = max_token_len
        self.rollout_n = rollout_n
        self.temperature = temperature
        self.logits_optimization = logits_optimization
        self.logits_optimization_peak_mem_size_in_gib = logits_optimization_peak_mem_size_in_gib
        self.logits_compute_from_fp32_inputs = logits_compute_from_fp32_inputs
        self.logits_compute_in_fp32 = logits_compute_in_fp32
        self.world_size = world_size

        self.use_unpad = use_unpad
        self.use_split_attention = use_split_attention
        self.patch_with_local = patch_with_local

        # This object needs to be updated (not overwritten) on every request in CausalLM module before calling the model's first forward.
        # The update has to happen via self.reconstruction_info.update(**reconstruction_info)
        self.reconstruction_info = ReconstructionInfo()

        # This should be known at model init and not change half-way through, since _create_patched_forward_split_attention depends on it
        self.reconstruction_info["is_unpadded"] = self.use_unpad

    def patch_forward(self):
        self.is_model_patched = False

        """Patch specific module forward methods."""
        for name, module in self.model.named_modules():
            module_class_name = type(module).__name__

            if module_class_name == self.patch_target_class_name:
                # 1. patch the main model
                self.is_model_patched = True
                module.forward = self._create_patched_main_model_forward(module, name)
            elif module_class_name == self.causal_lm_class_name:
                # 2. patch the causal_lm module
                module.forward = self._create_patched_causal_lm_forward(module, name)

        assert self.is_model_patched, (
            f"Deduplication is not supported without patching the model for {type(self.model).__name__}"
        )

        # 3. patch the attention layers
        self.attention_patcher = QwenAttentionOncePatcher(
            self.model,
            reconstruction_info=self.reconstruction_info,
            patch_with_local=self.patch_with_local,
            use_split_attention=self.use_split_attention,
        )

    def _create_patched_main_model_forward(self, module, module_name):
        """
        Returns patched main model forward
        """
        reconstruction_info = self.reconstruction_info

        def patched_forward(
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):

            # pr0(f"_create_patched_main_model_forward {reconstruction_info=}")

            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            DynamicCache = getattr(src_mod, "DynamicCache", None)
            create_causal_mask = getattr(src_mod, "create_causal_mask", None)
            create_sliding_window_causal_mask = getattr(src_mod, "create_sliding_window_causal_mask", None)
            BaseModelOutputWithPast = (
                getattr(src_mod, "BaseModelOutputWithPast", None)
                or getattr(src_mod, "MoeModelOutputWithPast", None)
                or _BaseModelOutputWithPast
            )

            # Validate input
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            # import pdb; pdb.set_trace()
            if inputs_embeds is None:
                inputs_embeds_dedup = module.embed_tokens(input_ids)
                inputs_embeds = ZoRRoTrain.reconstruct_sequences(inputs_embeds_dedup, reconstruction_info)

            if use_cache and past_key_values is None:
                past_key_values = _init_dynamic_cache(DynamicCache, module.config)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            # It may already have been prepared by e.g. `generate`
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                # Prepare mask arguments
                mask_kwargs = _build_mask_kwargs(
                    create_causal_mask,
                    module.config,
                    inputs_embeds,
                    attention_mask,
                    cache_position,
                    past_key_values,
                    position_ids,
                )
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                # The sliding window alternating layers are not always activated depending on the config
                if getattr(module, "has_sliding_layers", False):
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            linear_attn_mask = _update_linear_attn_mask(module, attention_mask, past_key_values, cache_position)

            hidden_states = inputs_embeds
            # create position embeddings to be shared across the decoder layers
            position_ids_dedup = ZoRRoTrain.deduplicate_sequences(position_ids, reconstruction_info)
            hidden_states_dedup = ZoRRoTrain.deduplicate_sequences(hidden_states, reconstruction_info)

            if reconstruction_info.get("is_unpadded", False):
                position_embeddings = module.rotary_emb(hidden_states_dedup, position_ids_dedup)
                layer_position_ids = position_ids_dedup
            else:
                position_embeddings = module.rotary_emb(hidden_states, position_ids)
                layer_position_ids = position_ids

            # pr0(f"{position_ids_dedup.shape=}")
            # pr0(f"{hidden_states_dedup.shape=}")
            # pr0(f"{position_embeddings[0].shape=}")

            for decoder_layer in module.layers[: module.config.num_hidden_layers]:
                layer_type = getattr(decoder_layer, "layer_type", None)
                if layer_type == "linear_attention":
                    layer_mask = linear_attn_mask
                else:
                    attention_type = getattr(decoder_layer, "attention_type", "full_attention")
                    layer_mask = causal_mask_mapping[attention_type]

                hidden_states_dedup = decoder_layer(
                    hidden_states_dedup,
                    attention_mask=layer_mask,
                    position_ids=layer_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states_dedup = module.norm(hidden_states_dedup)

            # hidden_states = ZoRRoTrain.reconstruct_sequences(hidden_states_dedup, reconstruction_info)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states_dedup,
                past_key_values=past_key_values if use_cache else None,
            )

        return patched_forward

    def _create_patched_causal_lm_forward(self, module, module_name):
        """
        Returns patched causal lm forward
        """
        reconstruction_info = self.reconstruction_info
        model = self.model
        response_len = self.response_len
        # max_token_len = self.max_token_len
        # rollout_n = self.rollout_n
        temperature = self.temperature
        use_unpad = self.use_unpad
        world_size = self.world_size
        logits_optimization = self.logits_optimization
        peak_mem_gib = self.logits_optimization_peak_mem_size_in_gib
        logits_compute_from_fp32_inputs = self.logits_compute_from_fp32_inputs
        logits_compute_in_fp32 = self.logits_compute_in_fp32

        def patched_forward(
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            calculate_entropy: bool = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs],
        ):

            device = input_ids.device

            tname = timers.start("zorro fwd")

            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            # MoE model modules (e.g. ``modeling_qwen3_moe``) export ``MoeModelOutputWithPast`` instead of
            # ``BaseModelOutputWithPast``; fall back to it (and finally to the canonical class) so the patched forward
            # works for both dense and MoE architectures. We only carry ``last_hidden_state`` through it.
            BaseModelOutputWithPast = (
                getattr(src_mod, "BaseModelOutputWithPast", None)
                or getattr(src_mod, "MoeModelOutputWithPast", None)
                or _BaseModelOutputWithPast
            )

            if attention_mask is None:
                raise ValueError("attention_mask is required for ZoRRO")

            # Off by default: this runs an extra full-batch attention-mask
            # analysis on every forward, which is expensive on long-context
            # (16K+) prompts. Enable with ARCTIC_ZORRO_DEBUG=1 when debugging.
            DEBUG = os.environ.get("ARCTIC_ZORRO_DEBUG", "0") == "1"

            if DEBUG:
                pr0(f"{input_ids.shape=} {input_ids=}")
                from .zorro_train import analyze_normal_batch_via_attention_mask

                analyze_normal_batch_via_attention_mask(input_ids, attention_mask, response_len)

            # pr0(f"{input_ids.shape=}")
            # pr0(f"{position_ids.shape=}")

            deduplicator = ZoRRoTrain()
            prompt_groups, unique_prompts = deduplicator.find_prompt_groups(
                input_ids=input_ids, response_length=response_len
            )
            dedup_input_ids, adapted_position_ids, reconstruction_info_this_batch = (
                deduplicator.create_deduplicated_batch(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    response_length=response_len,
                    prompt_groups=prompt_groups,
                    unique_prompts=unique_prompts,
                    attention_mask=attention_mask,
                    use_unpad=use_unpad,
                )
            )

            # this updates all the closures in the patched forwards
            reconstruction_info.update(**reconstruction_info_this_batch)
            # input_ids_rmpad = dedup_input_ids
            # position_ids_rmpad = adapted_position_ids

            if DEBUG:
                pr0(f"{reconstruction_info=}")
                pr0(f"{reconstruction_info['original_attention_mask'].sum()=}")

            attention_mask = (
                None  # we want attention to use pos ids, but we need it not None for zorro packing at the moment
            )

            pr0(f"{dedup_input_ids.shape=}")
            pr0(f"{adapted_position_ids.shape=}")
            pr0(f"{adapted_position_ids=}")

            outputs: BaseModelOutputWithPast = _get_text_backbone(model)(
                input_ids=dedup_input_ids,
                position_ids=adapted_position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state

            hidden_states_extracted = ZoRRoTrain.extract_unpadded_responses_from_deduped_packed_ids(
                hidden_states.squeeze(0), reconstruction_info, offset=-1
            )

            # Labels for the log_prob: the next-token target for each extracted
            # hidden state. ``hidden_states_extracted`` (offset=-1) yields, per
            # sample, ``[prompt_final, resp_pos_0..resp_pos_{R-2}]`` which predict
            # tokens ``[resp_0..resp_{R-1}]`` -- i.e. the labels are exactly each
            # sample's own response tokens.
            #
            # We must NOT derive them by rolling the *deduplicated* packed
            # sequence: ``torch.roll`` pairs the single shared prompt-final
            # position with the token physically packed after it, which is the
            # FIRST rollout's first response token. Because every rollout in a
            # prompt group extracts that same shared slot, all rollouts would get
            # the first rollout's first-token label (the prompt is deduplicated,
            # so the per-rollout boundary token is not adjacent in memory). That
            # makes the first response-token log-prob wrong for every rollout
            # except the first in each group.
            #
            # Extracting the responses directly with offset=0 gives each rollout
            # its own response tokens as labels, which aligns 1:1 with the
            # offset=-1 hidden states above.
            input_ids_extracted = ZoRRoTrain.extract_unpadded_responses_from_deduped_packed_ids(
                dedup_input_ids.squeeze(0), reconstruction_info, offset=0
            )

            timers.stop_and_print_elapsed(tname)
            tname = timers.start("zorro head+post-process")

            if logits_optimization == "memory":
                # `memory`: never manifest the full logits -- tiled compute under no_grad with an extra forward
                # replay in backward. Use this for long seqlen x large vocab, at the cost of a small additional
                # forward call.

                see_memory_usage(f"{torch.distributed.get_rank()}: before TiledLogProbEntropy", force=False)
                # Size shards so each shard's logits block stays within the configured peak-memory budget
                # (arctic_rl.train.logits.optimization_peak_mem_size_in_gib).
                chunk_rows = _logits_chunk_rows(model.config.vocab_size, peak_mem_gib)
                num_shards = max(1, ceildiv(hidden_states_extracted.shape[0], chunk_rows))
                # pr0(f"derived {num_shards=}")

                # sync num shards across gpus so that deepspeed won't hang if the values are different
                if world_size > 1:

                    local_num_shards = torch.tensor(num_shards, dtype=torch.long, device=device)
                    torch.distributed.all_reduce(local_num_shards, op=torch.distributed.ReduceOp.MAX)
                    # all_num_shards = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
                    # torch.distributed.all_gather(all_num_shards, local_num_shards)
                    # num_shards = max(t.item() for t in all_num_shards)
                    num_shards = local_num_shards.item()
                    # pr0(f"synced {num_shards=}")

                compute_params = [model.lm_head.weight]  # tied with self.model.embed_tokens.weight
                # bind the fp32 upcast flags so they flow through TiledLogProbEntropy's forward and (replayed)
                # backward without changing its signature.
                logprobs, entropy = TiledLogProbEntropy.apply(
                    partial(
                        tiled_entropy_and_logprobs_with_temperature_from_logits,
                        logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
                        logits_compute_in_fp32=logits_compute_in_fp32,
                    ),
                    model,
                    hidden_states_extracted,
                    input_ids_extracted,
                    temperature,
                    calculate_entropy,
                    num_shards,
                    compute_params,
                )
            elif logits_optimization == "compute":
                # `compute`: manifest the full logits once, but run the softmax/entropy follow-up in chunks so
                # the full-size intermediates are never materialized at once.
                logprobs, entropy = chunked_entropy_and_logprobs_with_temperature_from_logits(
                    model,
                    hidden_states_extracted,
                    input_ids_extracted,
                    temperature,
                    calculate_entropy,
                    peak_mem_gib=peak_mem_gib,
                    logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
                    logits_compute_in_fp32=logits_compute_in_fp32,
                )
            elif logits_optimization == "none":
                # `none`: manifest the full logits and compute in one shot.
                logprobs, entropy = tiled_entropy_and_logprobs_with_temperature_from_logits(
                    model,
                    hidden_states_extracted,
                    input_ids_extracted,
                    temperature,
                    calculate_entropy,
                    logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
                    logits_compute_in_fp32=logits_compute_in_fp32,
                )
            else:
                raise ValueError(
                    f"Unknown arctic_rl.train.logits.optimization={logits_optimization!r}; "
                    "expected one of: none, memory, compute"
                )

            # similar to ZoRRoTrain.extract_unpadded_responses_from_deduped_packed_ids, but leaving responses packed in
            # 1D w/o padding, but we are removing any permutations ZoRRoTrain may have applied originally
            logprobs = ZoRRoTrain.responses_in_orig_sample_order(logprobs, reconstruction_info)
            if calculate_entropy:
                entropy = ZoRRoTrain.responses_in_orig_sample_order(entropy, reconstruction_info)

            # DO NOT DELETE:
            # 1. if later for some reason we need to returned padded 2D responses (e.g. if the loss performs per-sample math and per token is wrong, then the following should be used instead
            # if calculate_entropy:
            #     #entropy = PromptDeduplicator.pad_responses(entropy, reconstruction_info)
            #     #pr(f"{entropy.shape=}")
            # else:
            #     entropy = None
            # logprobs = PromptDeduplicator.pad_responses(logprobs, reconstruction_info)
            # print(f"aft {logprobs.shape=}")
            #
            #
            # 2. this can be used for if we need to return 2D full tensors that are both padded and contain prompt-padding
            # if 0:
            #     # for now hack to prepad prompt width to make it work with ARL - will go away
            #     max_prompt_len = input_ids.shape[1] - response_len
            #     def prepand_max_prompt_len_zeros(tensor: torch.Tensor, max_prompt_len):
            #         prepand = torch.zeros([tensor.shape[0], max_prompt_len],  dtype=torch.int64, device=tensor.device)
            #         return torch.cat([prepand, tensor], dim=1)
            #     if calculate_entropy:
            #         entropy = prepand_max_prompt_len_zeros(entropy, max_prompt_len)
            #     logprobs = prepand_max_prompt_len_zeros(logprobs, max_prompt_len)

            timers.stop_and_print_elapsed(tname)

            model_outputs = ModelOutput(logprobs=logprobs, entropy=entropy)
            return model_outputs

        return patched_forward


class Qwen3ModelPatcher(ModuleReconstructionPatcher):
    """
    Qwen3-specific model patcher.

    .. warning::
        OUTDATED / DO NOT USE WITHOUT SYNCING FIRST. This context-manager patcher has drifted out of sync with
        current transformers: recent Qwen3 precomputes the rotary ``position_embeddings`` once in the main model
        forward (on the *deduplicated* sequence length) and passes them down, while this patcher's attention path
        reconstructs Q/K back to the *original* (full) length before applying rotary -- so ``apply_rotary_pos_emb``
        raises a sequence-length mismatch and no forward completes. ``DeduplicatedActor`` (which uses this class) is
        broken for the same reason.

        The maintained, production patcher is ``Qwen3ModelOncePatcher`` (installed by ``deepspeed_worker.py`` and
        exercised by ``tests/zorro_train/test_once_patcher.py``). Before using ``Qwen3ModelPatcher`` again, bring it
        back in line with ``Qwen3ModelOncePatcher`` -- specifically the rotary/position-embedding flow in the
        main-model and attention forwards (apply rotary on the deduplicated Q/K, or reconstruct ``cos``/``sin`` to
        the full length) -- and re-verify against a non-deduplicated reference.

    Handles deduplication/reconstruction for Qwen3Model and Qwen3ForCausalLM.

    Flow:
    1. Input: deduplicated input_ids [1, total_tokens]
    2. After embedding: reconstruct to [batch_size, seq_len, hidden_dim]
    3. Before decoder layers: deduplicate to [1, total_tokens, hidden_dim]
    4. Inside decoder layers: stays deduplicated (attention handles reconstruction)
    5. After decoder layers: reconstruct to [batch_size, seq_len, hidden_dim] for logits
    """

    def __init__(self, model, reconstruction_info, patch_with_local=False, use_split_attention=True):
        """
        Initialize Qwen3 model patcher.

        Args:
            model: Qwen3Model or Qwen3ForCausalLM instance
            reconstruction_info: Deduplication metadata
            patch_with_local: If True, use local (baseline) implementation
            use_split_attention: If True, use split attention (2 calls: prompt-to-prompt + response-to-full).
                                If False, use standard approach (1 call: full attention on reconstructed Q/K/V).
        """
        # OUTDATED: kept for reference only. See the class docstring -- this path is stale against current
        # transformers and must be synced with Qwen3ModelOncePatcher (the production patcher) before use.
        warnings.warn(
            "Qwen3ModelPatcher is outdated and currently broken against the installed transformers (rotary "
            "position-embedding length mismatch); it must be synced with Qwen3ModelOncePatcher before use. See the "
            "Qwen3ModelPatcher class docstring.",
            stacklevel=2,
        )
        super().__init__(model, reconstruction_info, patch_with_local=patch_with_local)
        self.attention_patcher = QwenAttentionPatcher(
            model, reconstruction_info, patch_with_local=patch_with_local, use_split_attention=use_split_attention
        )

        self.is_model_patched = None

    @staticmethod
    def _should_patch_module_forward(name, module):
        """Check if this is a Qwen3 model module we should patch."""
        # Check if this is the main Qwen3Model (not layers inside it)
        module_class_name = type(module).__name__

        # We want to patch Qwen3Model specifically
        if module_class_name == "Qwen3Model":
            return True

        return False

    def _patch_forward(self):
        """Patch all module forward methods."""
        for name, module in self.model.named_modules():
            if self._should_patch_module_forward(name, module):
                # this is to signal that the model was patched to DP Actor
                self.is_model_patched = True

                # Store original forward
                self.original_forwards[name] = module.forward
                if self.patch_with_local:
                    # assert False, "This code path is only for debugging purposes"
                    assert (
                        self._create_unpatched_forward_local is not None
                    ), "Subclass must implement _create_unpatched_forward_local"
                    module.forward = self._create_unpatched_forward_local(module, name)
                else:
                    # Create patched forward that optimizes QKV
                    module.forward = self._create_patched_forward(module, name)

        # Patch attention layers after model forward is patched
        if self.is_model_patched:
            self.attention_patcher.__enter__()

    def _unpatch_forward(self):
        """Restore original forward methods."""
        for name, module in self.model.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.original_forwards.clear()

        # Unpatch attention layers
        self.attention_patcher.__exit__(None, None, None)

    def _create_unpatched_forward_local(self, module, module_name):
        """Create unpatched forward for Qwen3Model."""

        def unpatched_forward(
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            DynamicCache = getattr(src_mod, "DynamicCache", None)
            create_causal_mask = getattr(src_mod, "create_causal_mask", None)
            create_sliding_window_causal_mask = getattr(src_mod, "create_sliding_window_causal_mask", None)
            # MoE model modules (e.g. ``modeling_qwen3_moe``) export ``MoeModelOutputWithPast`` instead of
            # ``BaseModelOutputWithPast``; fall back to it (and finally to the canonical class) so the patched forward
            # works for both dense and MoE architectures. We only carry ``last_hidden_state`` through it.
            BaseModelOutputWithPast = (
                getattr(src_mod, "BaseModelOutputWithPast", None)
                or getattr(src_mod, "MoeModelOutputWithPast", None)
                or _BaseModelOutputWithPast
            )

            # Validate input
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = module.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            # It may already have been prepared by e.g. `generate`
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                # Prepare mask arguments.
                mask_kwargs = _build_mask_kwargs(
                    create_causal_mask,
                    module.config,
                    inputs_embeds,
                    attention_mask,
                    cache_position,
                    past_key_values,
                    position_ids,
                )
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                # The sliding window alternating layers are not always activated depending on the config
                if module.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = module.rotary_emb(hidden_states, position_ids)

            for decoder_layer in module.layers[: module.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = module.norm(hidden_states)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        return unpatched_forward

    def _create_patched_forward(self, module, module_name):
        """ """
        reconstruction_info = self.reconstruction_info

        def patched_forward(
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            DynamicCache = getattr(src_mod, "DynamicCache", None)
            create_causal_mask = getattr(src_mod, "create_causal_mask", None)
            create_sliding_window_causal_mask = getattr(src_mod, "create_sliding_window_causal_mask", None)
            # MoE model modules (e.g. ``modeling_qwen3_moe``) export ``MoeModelOutputWithPast`` instead of
            # ``BaseModelOutputWithPast``; fall back to it (and finally to the canonical class) so the patched forward
            # works for both dense and MoE architectures. We only carry ``last_hidden_state`` through it.
            BaseModelOutputWithPast = (
                getattr(src_mod, "BaseModelOutputWithPast", None)
                or getattr(src_mod, "MoeModelOutputWithPast", None)
                or _BaseModelOutputWithPast
            )

            # Validate input
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            # import pdb; pdb.set_trace()
            if inputs_embeds is None:
                inputs_embeds_dedup = module.embed_tokens(input_ids)
                inputs_embeds = ZoRRoTrain.reconstruct_sequences(inputs_embeds_dedup, reconstruction_info)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            # It may already have been prepared by e.g. `generate`
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                # Prepare mask arguments.
                mask_kwargs = _build_mask_kwargs(
                    create_causal_mask,
                    module.config,
                    inputs_embeds,
                    attention_mask,
                    cache_position,
                    past_key_values,
                    position_ids,
                )
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                # The sliding window alternating layers are not always activated depending on the config
                if module.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            hidden_states = inputs_embeds
            # create position embeddings to be shared across the decoder layers
            position_ids_dedup = ZoRRoTrain.deduplicate_sequences(position_ids, reconstruction_info)
            hidden_states_dedup = ZoRRoTrain.deduplicate_sequences(hidden_states, reconstruction_info)

            position_embeddings = module.rotary_emb(hidden_states_dedup, position_ids_dedup)

            for decoder_layer in module.layers[: module.config.num_hidden_layers]:
                hidden_states_dedup = decoder_layer(
                    hidden_states_dedup,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states_dedup = module.norm(hidden_states_dedup)

            # hidden_states = ZoRRoTrain.reconstruct_sequences(hidden_states_dedup, reconstruction_info)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states_dedup,
                past_key_values=past_key_values if use_cache else None,
            )

        return patched_forward


def ceildiv(a, b):
    return -(a // -b)


def _logits_chunk_rows(vocab_size, peak_mem_gib, bytes_per_elem=4):
    """Max number of token rows whose ``[rows, vocab_size]`` logits block stays within ``peak_mem_gib`` GiB of
    peak memory overhead.

    Used to size the chunks/shards/tiles for the `memory` and `compute` logits-optimization modes from a single
    memory budget (``arctic_rl.train.logits.optimization_peak_mem_size_in_gib``). ``bytes_per_elem`` defaults to 4
    (fp32), the conservative accounting used by the logits follow-up math. Always returns at least 1.
    """
    budget_bytes = max(1, int(peak_mem_gib * 2**30))
    row_bytes = max(1, int(vocab_size) * bytes_per_elem)
    return max(1, budget_bytes // row_bytes)


def _lm_head_logits_with_temperature(
    model, hidden_states, temperature, logits_compute_from_fp32_inputs=False, logits_compute_in_fp32=False
):
    """Project hidden states to vocab logits via the LM head, applying optional
    temperature scaling. Returns the (possibly squeezed) logits tensor.

    Shared by the `none`/`compute` logits-optimization paths; the full logits are
    manifested here (the `memory` path avoids this by tiling inside the autograd
    function instead).

    When ``logits_compute_from_fp32_inputs`` is set, the LM-head input is upcast to fp32 so the projection
    (and hence the logits / logprob / entropy math) runs in fp32 (arctic_rl.train.logits.compute_from_fp32_inputs).

    When ``logits_compute_in_fp32`` is set, the produced logits are upcast to fp32 before they are consumed
    (temperature scaling + downstream logprob/entropy math) (arctic_rl.train.logits.compute_in_fp32). No-op if
    already fp32.
    """
    if logits_compute_from_fp32_inputs:
        hidden_states = hidden_states.float()
    logits = model.lm_head(hidden_states)
    if logits_compute_in_fp32:
        logits = logits.float()
    if temperature != 1.0:
        # logits = logits / temperature
        logits = logits.squeeze(0)  # (total_nnz, vocab_size)
        temperature = torch.tensor(temperature, device=logits.device)
        logits.div_(temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype))
    return logits


def _logprobs_and_entropy_from_flat_logits(flat_logits, flat_labels, calculate_entropy):
    """Per-token logprobs (and optionally entropy) from a 2D ``[N, V]`` logits
    block, returned as 1D ``[N]`` tensors (entropy is ``None`` when not
    requested).

    Uses the fused flash-attn cross-entropy kernel when available, otherwise a
    logsumexp/gather fallback. This is the common core for both the single-shot
    (`tiled_...`) and chunked (`chunked_...`) entrypoints; callers own any
    reshape back to the original batch dims.
    """
    entropy = None
    if FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        inplace_backward = flat_logits.requires_grad
        output = cross_entropy_loss(flat_logits, flat_labels, inplace_backward=inplace_backward)
        logprobs = -output[0]
        if calculate_entropy:
            gathered_logits = torch.gather(flat_logits, -1, flat_labels.unsqueeze(-1)).squeeze(-1)
            logsumexp = gathered_logits - logprobs
            probs = torch.exp(flat_logits - logsumexp.unsqueeze(-1))
            entropy = logsumexp - torch.sum(probs * flat_logits, dim=-1)
    else:
        # using 2 different implementation paths to optimize for whether
        # calculate_entropy is needed or not
        if calculate_entropy:
            logsumexp = torch.logsumexp(flat_logits, dim=-1)
            logprobs = torch.gather(flat_logits, -1, flat_labels.unsqueeze(-1)).squeeze(-1) - logsumexp
            probs = torch.exp(flat_logits - logsumexp.unsqueeze(-1))
            entropy = logsumexp - torch.sum(probs * flat_logits, dim=-1)
        else:
            # Fastest logprobs-only: log_softmax fused kernel (single pass) +
            # gather on the result.
            logprobs = torch.gather(
                torch.nn.functional.log_softmax(flat_logits, dim=-1), -1, flat_labels.unsqueeze(-1)
            ).squeeze(-1)
    return logprobs, entropy


def tiled_entropy_and_logprobs_with_temperature_from_logits(
    model,
    hidden_states,
    labels,
    temperature=1.0,
    calculate_entropy=True,
    logits_compute_from_fp32_inputs=False,
    logits_compute_in_fp32=False,
):
    # `none` mode: manifest the full logits, then compute logprobs/entropy in one shot. (Also reused per-shard
    # by TiledLogProbEntropy in `memory` mode, where it is called on tiled hidden_states so the full logits
    # aren't manifested.)
    tname_e2e = timers.start("logprob: tiled_entropy_and_logprobs_with_temperature_from_logits e2e")

    logits = _lm_head_logits_with_temperature(
        model,
        hidden_states,
        temperature,
        logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
        logits_compute_in_fp32=logits_compute_in_fp32,
    )

    batch_dim = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)

    logprobs, entropy = _logprobs_and_entropy_from_flat_logits(flat_logits, flat_labels, calculate_entropy)
    logprobs = logprobs.view(*batch_dim)
    if entropy is not None:
        entropy = entropy.view(*batch_dim)

    timers.stop_and_print_elapsed(tname_e2e)

    return logprobs, entropy


def chunked_entropy_and_logprobs_with_temperature_from_logits(
    model,
    hidden_states,
    labels,
    temperature=1.0,
    calculate_entropy=True,
    peak_mem_gib=4.0,
    logits_compute_from_fp32_inputs=False,
    logits_compute_in_fp32=False,
):
    """`compute` logits-optimization mode for ZoRRO.

    Manifests the full logits once (a single ``model.lm_head`` over all tokens)
    and then runs the softmax/entropy follow-up in chunks along the token
    dimension, so the full-size follow-up intermediates (``probs`` /
    ``log_softmax``, each as large as the logits) are never materialized at once.
    Each chunk's follow-up working set is bounded by ``peak_mem_gib`` GiB
    (arctic_rl.train.logits.optimization_peak_mem_size_in_gib).
    Memory cost: the full logits tensor, once. Compute cost: a Python loop over
    token chunks. Mirrors the chunking pattern used by
    ``processors.pipeline.chunked_logprobs_and_entropy_from_logits``.

    Contrast with the other modes:
      * ``none``   -> :func:`tiled_entropy_and_logprobs_with_temperature_from_logits`:
                      full logits + full-size follow-up intermediates.
      * ``memory`` -> :class:`TiledLogProbEntropy`: logits never fully manifested,
                      at the cost of an extra forward replay in backward.
    """
    tname_e2e = timers.start("logprob: chunked_entropy_and_logprobs_with_temperature_from_logits e2e")

    logits = _lm_head_logits_with_temperature(
        model,
        hidden_states,
        temperature,
        logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
        logits_compute_in_fp32=logits_compute_in_fp32,
    )

    batch_dim = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)

    chunk_size = _logits_chunk_rows(flat_logits.shape[-1], peak_mem_gib)

    logprobs_chunks = []
    entropy_chunks = [] if calculate_entropy else None
    for start in range(0, flat_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_logits.shape[0])
        logprobs_chunk, entropy_chunk = _logprobs_and_entropy_from_flat_logits(
            flat_logits[start:end], flat_labels[start:end], calculate_entropy
        )
        logprobs_chunks.append(logprobs_chunk)
        if calculate_entropy:
            entropy_chunks.append(entropy_chunk)

    logprobs = torch.cat(logprobs_chunks, dim=0).view(*batch_dim)
    entropy = torch.cat(entropy_chunks, dim=0).view(*batch_dim) if calculate_entropy else None

    timers.stop_and_print_elapsed(tname_e2e)

    return logprobs, entropy


class TiledLogProbEntropy(torch.autograd.Function):
    """
    TiledLogProbEntropy implementation using gradient hooks (the grad hooks were copied from Axolotl). This has been adapted from TiledMLP in Deepspeed.
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        model,
        hidden_states,
        labels,
        temperature,
        calculate_entropy,
        shards,
        compute_params,
    ) -> torch.Tensor:

        # don't store anything for bwd if this is a torch.no_grad forward
        if hidden_states.requires_grad:
            ctx.fn = fn
            ctx.model = model
            ctx.shards = shards
            ctx.compute_params = [p for p in compute_params if p.requires_grad]
            ctx.temperature = temperature
            ctx.calculate_entropy = calculate_entropy
            ctx.save_for_backward(hidden_states, labels)

        hidden_states_shards = list(torch.chunk(hidden_states, chunks=shards, dim=0))
        labels_shards = list(torch.chunk(labels, chunks=shards, dim=0))

        with torch.no_grad():
            logprobs_shards, entropy_shards = list(
                zip(
                    *[
                        fn(model, hidden_states_shards[idx], labels_shards[idx], temperature, calculate_entropy)
                        for idx in range(shards)
                    ]
                )
            )

        if calculate_entropy:
            entropy = torch.cat(entropy_shards, dim=0)
            # pr(f"{entropy.shape=}")
        else:
            entropy = None

        logprobs = torch.cat(logprobs_shards, dim=0)

        return logprobs, entropy

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (hidden_states, labels) = ctx.saved_tensors
        model = ctx.model
        shards = ctx.shards
        compute_params = ctx.compute_params

        temperature = ctx.temperature
        calculate_entropy = ctx.calculate_entropy

        hs = hidden_states

        hs_requires_grad = hs.requires_grad
        hs = hs.detach()
        hs.requires_grad_(hs_requires_grad)

        logprobs_grads, entropy_grads = grads

        hs_grad = torch.zeros_like(hs)

        # return (None, None, hs_grad, None, None, None, None, None)

        hs_shards = list(torch.chunk(hs, chunks=shards, dim=0))
        labels_shards = list(torch.chunk(labels, chunks=shards, dim=0))

        # not using GradientAccumulator since it's not needed under deepspeed (needed for ddp/fsdp, so leaving the code here, but commented out)
        # Create a gradient accumulator for parameters
        # grad_accumulator = GradientAccumulator(compute_params, shards, dtype=hs.dtype)

        # Tell deepspeed not to add a new grad to its ipg bucket during this backward
        # oddly because of self.lm_head.weight being tied with self.model.embed_tokens.weight we have to tell DS that the grad isn't ready and it'll be reduced when model.embed_tokens.weight grad is reduced
        # otherwise it asserts the parameter model.embed_tokens.weight has already been reduced.
        for param in compute_params:
            param.ds_grad_is_ready = False

        labels_step = labels_shards[0].shape[0]
        shard_step = hs_shards[0].numel()
        for i, hs_shard in enumerate(hs_shards):
            hs_shard.requires_grad_(hs_requires_grad)

            shard_offset = i * shard_step
            hs_shard.grad = hs_grad.view(-1).narrow(0, shard_offset, hs_shard.numel()).view_as(hs_shard)

            # Install hooks for this shard
            # is_last_shard = i + 1 == shards
            # grad_accumulator.install_hooks(is_last_shard)

            with torch.enable_grad():
                logprobs_shard, entropy_shard = fn(model, hs_shard, labels_shards[i], temperature, calculate_entropy)

            incoming_grad_shards = []
            tensors = []
            if entropy_shard is not None:
                tensors += [entropy_shard]
                incoming_grad_shards += [
                    (
                        entropy_grads.view(-1)
                        .narrow(0, i * labels_step, labels_shards[i].shape[0])
                        .view(labels_shards[i].shape[0])
                    )
                ]

            tensors += [logprobs_shard]
            incoming_grad_shards += [
                (
                    logprobs_grads.view(-1)
                    .narrow(0, i * labels_step, labels_shards[i].shape[0])
                    .view(labels_shards[i].shape[0])
                )
            ]

            torch.autograd.backward(tensors, incoming_grad_shards)

        # Clean up hooks
        # grad_accumulator.cleanup()
        # del grad_accumulator

        return (
            None,
            None,
            hs_grad,
            None,
            None,
            None,
            None,
            None,
        )


class GradientAccumulator:
    """
    Manual gradient accumulator for TiledLogProbEntropy with configurable precision
    Accumulates in specified dtype and rescales the gradient at the end
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        total_shards: int,
        dtype: torch.dtype | None = None,
    ):
        self.params = params
        self.total_shards = total_shards
        self.grad_accumulation_dtype = dtype or torch.float32
        self.accumulated_grads = {}
        self.hooks = []
        self.lock = threading.Lock()
        self.gradient_scale = 1.0 / total_shards

        # Initialize accumulated gradients in the specified dtype
        for param in self.params:
            if param.grad is not None:
                self.accumulated_grads[param] = param.grad.to(self.grad_accumulation_dtype)
                param.grad = None
            else:
                self.accumulated_grads[param] = torch.zeros_like(param, dtype=self.grad_accumulation_dtype)

    def install_hooks(self, is_last_shard: bool):
        """Install gradient hooks that accumulate gradients in higher precision"""

        def create_hook(param):
            def hook(grad):
                with self.lock:
                    grad_to_accum_dtype = grad.to(self.grad_accumulation_dtype)
                    scaled_grad = grad_to_accum_dtype * self.gradient_scale

                    if param in self.accumulated_grads:
                        self.accumulated_grads[param] += scaled_grad
                    else:
                        self.accumulated_grads[param] = scaled_grad.clone()

                    # Only assign the averaged gradient on the last shard
                    if is_last_shard:
                        param.grad = self.accumulated_grads[param].to(param.dtype)
                        return param.grad
                    return None

            return hook

        # Install hooks on all parameters
        for param in self.params:
            if param.requires_grad:
                hook = param.register_hook(create_hook(param))
                self.hooks.append(hook)

    def cleanup(self):
        """Remove all installed hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        del self.accumulated_grads
