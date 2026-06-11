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
Qwen-specific attention patcher with QKV optimization.

This module implements optimized attention patching for Qwen2/Qwen3 models.
"""

import sys

import torch
import torch.nn.functional as F

from arctic_platform.rl.utils.debug import pr0

from .module_patcher import ModuleReconstructionPatcher
from .zorro_train import ZoRRoTrain

# capture_at_invocation at 1e12 or some other high number disables the debug - which otherwise will allocate huge tensors which would persist through the whole step
debug_object = {
    "baseline": None,
    "patched": None,
    "baseline_counter": 0,
    "patched_counter": 0,
    "capture_at_invocation": int(1e12),  # Which invocation to capture (0 = first)
}

cos_dedup = None
sin_dedup = None
group_sizes = None


def reset_debug_object():
    debug_object["baseline"] = None
    debug_object["patched"] = None
    debug_object["baseline_counter"] = 0
    debug_object["patched_counter"] = 0


class QwenAttentionPatcher(ModuleReconstructionPatcher):
    """
    Qwen-specific attention patcher that optimizes QKV projections.

    Qwen models have separate q_proj, k_proj, v_proj, and o_proj layers.
    We compute Q, K, V on deduplicated batch, reconstruct them individually,
    then run the attention computation.
    """

    def __init__(self, model, reconstruction_info, patch_with_local=False, use_split_attention=True):
        """
        Initialize Qwen attention patcher.

        Args:
            model: The model to patch
            reconstruction_info: Metadata for deduplication
            patch_with_local: Whether to use local (baseline) implementation
            use_split_attention: If True, use split attention (2 calls: prompt-to-prompt + response-to-full).
                                If False, use standard approach (1 call: full attention on reconstructed Q/K/V).
        """
        self.use_split_attention = use_split_attention
        super().__init__(model, reconstruction_info, patch_with_local=patch_with_local)

    @staticmethod
    def _should_patch_module_forward(name, module):
        """Check if this is a Qwen attention module we should patch."""
        name_lower = name.lower()

        # Must contain 'attn' or 'attention'
        if "attn" not in name_lower and "attention" not in name_lower:
            return False

        # Must be a self_attn module (not its submodules)
        # Example: "model.layers.0.self_attn" YES
        # Example: "model.layers.0.self_attn.q_proj" NO
        name_parts = name.split(".")
        if not name_parts:
            return False

        # Check if it's a main attention module
        if name_parts[-1] in ["self_attn", "attention"]:
            # Verify it has q_proj, k_proj, v_proj, o_proj
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "k_proj")
                and hasattr(module, "v_proj")
                and hasattr(module, "o_proj")
            ):
                return True

        return False

    @staticmethod
    def _extract_response_to_full_mask_packed(
        attention_mask, cu_seqlens_response, cu_seqlens_packed, prompt_len, original_attention_mask
    ):
        """
        Extract response-to-(prompt+response) attention mask from packed attention mask.

        In packed format, sequences are concatenated without padding. This function extracts
        the rows corresponding to response tokens from the full attention mask, creating a
        mask that shows which tokens each response token can attend to.

        Args:
            attention_mask: Full attention mask in packed format.
                Shape: [1, 1, total_packed_tokens, total_packed_tokens] or
                       [total_packed_tokens, total_packed_tokens]
            cu_seqlens_response: Cumulative sequence lengths for response tokens.
                Shape: [batch_size + 1], where cu_seqlens_response[i] is the starting
                index of sequence i's response tokens in the packed tensor (response-only).
            cu_seqlens_packed: Cumulative sequence lengths for full packed sequences.
                Shape: [batch_size + 1], where cu_seqlens_packed[i] is the starting
                index of sequence i in the full packed tensor (prompt + response).
            prompt_len: Maximum prompt length.
            original_attention_mask: Original attention mask before packing.
                Shape: [batch_size, seq_len], used to get valid prompt token counts.

        Returns:
            mask: Attention mask of shape [total_response_tokens, total_packed_tokens]
                extracted from rows corresponding to response tokens.
        """
        batch_size = len(cu_seqlens_response) - 1

        # Handle different attention_mask shapes
        # Remove batch and head dimensions if present: [1, 1, seq, seq] -> [seq, seq]
        if attention_mask.dim() == 4:
            attention_mask = attention_mask.squeeze(0).squeeze(0)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(0)

        # Collect indices of response tokens in the FULL packed sequence
        response_token_indices = []
        for seq_idx in range(batch_size):
            # Get the range of this sequence in the full packed tensor
            seq_start_full = cu_seqlens_packed[seq_idx].item()
            seq_end_full = cu_seqlens_packed[seq_idx + 1].item()

            # Get number of valid prompt tokens from original attention mask
            sample_mask = original_attention_mask[seq_idx]
            prompt_mask = sample_mask[:prompt_len]
            num_valid_prompt = prompt_mask.sum().item()

            # Response tokens start after the prompt tokens in the full sequence
            response_start_full = seq_start_full + num_valid_prompt
            response_end_full = seq_end_full

            response_token_indices.extend(range(response_start_full, response_end_full))

        # Extract rows corresponding to response tokens
        response_to_full_mask = attention_mask[response_token_indices, :]

        return response_to_full_mask

    @staticmethod
    def _extract_prompt_to_prompt_mask_packed(
        attention_mask,
        cu_seqlens_unique_prompts,
        cu_seqlens_packed,
        prompt_len,
        original_attention_mask,
        prompt_groups,
    ):
        """
        Extract prompt-to-prompt attention mask from packed attention mask.

        Args:
            attention_mask: Full attention mask in packed format.
                Shape: [1, 1, total_packed_tokens, total_packed_tokens] or
                       [total_packed_tokens, total_packed_tokens]
            cu_seqlens_unique_prompts: Cumulative sequence lengths for unique prompt tokens.
                Shape: [num_unique_prompts + 1]
            cu_seqlens_packed: Cumulative sequence lengths for full packed sequences.
                Shape: [batch_size + 1]
            prompt_len: Maximum prompt length.
            original_attention_mask: Original attention mask before packing.
                Shape: [batch_size, seq_len]
            prompt_groups: List of lists, each inner list contains sample indices sharing a prompt.

        Returns:
            mask: Attention mask of shape [total_unique_prompt_tokens, total_unique_prompt_tokens]
                Block diagonal mask for unique prompts.
        """
        # Handle different attention_mask shapes
        if attention_mask.dim() == 4:
            attention_mask = attention_mask.squeeze(0).squeeze(0)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(0)

        # num_unique_prompts = len(prompt_groups)

        # For each unique prompt, extract the prompt tokens from the full packed sequence
        prompt_token_indices = []
        for group_idx, group in enumerate(prompt_groups):
            # Get first sequence from this group
            first_seq_idx = group[0]

            # Get the range of this sequence in the full packed tensor
            seq_start_full = cu_seqlens_packed[first_seq_idx].item()

            # Get number of valid prompt tokens
            sample_mask = original_attention_mask[first_seq_idx]
            prompt_mask = sample_mask[:prompt_len]
            num_valid_prompt = prompt_mask.sum().item()

            # Prompt tokens are at the beginning of the sequence
            prompt_start_full = seq_start_full
            prompt_end_full = seq_start_full + num_valid_prompt

            prompt_token_indices.extend(range(prompt_start_full, prompt_end_full))

        # Extract submatrix: rows and columns corresponding to prompt tokens
        # This creates the prompt-to-prompt attention mask
        prompt_mask_2d = attention_mask[prompt_token_indices, :][:, prompt_token_indices]

        return prompt_mask_2d

    @staticmethod
    def _precompute_attention_masks_packed(
        attention_mask, reconstruction_info, cu_seqlens_unique_prompts, cu_seqlens_response
    ):
        """
        Precompute and store attention masks for prompt and response attention.

        This should be called once (in the first layer) to avoid recomputing masks
        in every layer.

        Args:
            attention_mask: Full 2D attention mask in packed format
            reconstruction_info: Dict to store the precomputed masks
            cu_seqlens_unique_prompts: Cumulative sequence lengths for unique prompts
            cu_seqlens_response: Cumulative sequence lengths for responses
        """
        cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]
        prompt_len = reconstruction_info["prompt_len"]
        original_attention_mask = reconstruction_info["original_attention_mask"]
        prompt_groups = reconstruction_info["prompt_groups"]

        # Precompute prompt-to-prompt mask
        prompt_to_prompt_mask = QwenAttentionPatcher._extract_prompt_to_prompt_mask_packed(
            attention_mask,
            cu_seqlens_unique_prompts,
            cu_seqlens_packed,
            prompt_len,
            original_attention_mask,
            prompt_groups,
        )

        # Precompute response-to-full mask
        response_to_full_mask = QwenAttentionPatcher._extract_response_to_full_mask_packed(
            attention_mask, cu_seqlens_response, cu_seqlens_packed, prompt_len, original_attention_mask
        )

        # Store in reconstruction_info
        reconstruction_info["prompt_to_prompt_mask"] = prompt_to_prompt_mask
        reconstruction_info["response_to_full_mask"] = response_to_full_mask

    @staticmethod
    def _prepare_attention_kwargs_and_masks(
        use_cumsum_mask,
        reconstruction_info,
        cu_seqlens_unique_prompts,
        cu_seqlens_response,
        cu_seqlens_packed,
        attention_mask,
        device,
        kwargs,
    ):
        """
        Prepare attention kwargs and masks for prompt and response attention.

        Args:
            use_cumsum_mask: If True, use FA 2.1+ varlen API. If False, use explicit masks.
            reconstruction_info: Dict with reconstruction metadata
            cu_seqlens_unique_prompts: Cumulative sequence lengths for unique prompts
            cu_seqlens_response: Cumulative sequence lengths for responses
            cu_seqlens_packed: Cumulative sequence lengths for full packed sequences
            attention_mask: Full 2D attention mask in packed format
            kwargs: Existing kwargs to merge with

        Returns:
            tuple: (prompt_kwargs, response_kwargs, prompt_mask, response_mask)
        """

        if use_cumsum_mask:
            # FA 2.1+: use varlen API with bottom-right aligned causal mask

            flash_kwargs_prompt = {
                "cu_seq_lens_q": cu_seqlens_unique_prompts.to(device),
                "cu_seq_lens_k": cu_seqlens_unique_prompts.to(device),
                "max_length_q": reconstruction_info["max_prompt_valid_len"],
                "max_length_k": reconstruction_info["max_prompt_valid_len"],
                "causal": True,
            }
            flash_kwargs_response = {
                "cu_seq_lens_q": cu_seqlens_response.to(device),
                "cu_seq_lens_k": cu_seqlens_packed.to(device),
                "max_length_q": reconstruction_info["max_response_valid_len"],
                "max_length_k": reconstruction_info["max_seqlen_packed"],
                "causal": True,  # FA 2.1+ bottom-right aligned: responses see all prompts + causal responses
            }

            # Merge with existing kwargs
            prompt_kwargs = {**kwargs, **flash_kwargs_prompt}
            response_kwargs = {**kwargs, **flash_kwargs_response}
            prompt_mask = None
            response_mask = None
        else:

            assert attention_mask is not None, "Attention mask is required for standard approach"
            # Precompute masks once (common path for both branches)
            if "prompt_to_prompt_mask" not in reconstruction_info:
                QwenAttentionPatcher._precompute_attention_masks_packed(
                    attention_mask, reconstruction_info, cu_seqlens_unique_prompts, cu_seqlens_response
                )

            prompt_to_prompt_mask = reconstruction_info["prompt_to_prompt_mask"]
            response_to_full_mask = reconstruction_info["response_to_full_mask"]

            # Use explicit masks
            prompt_mask = prompt_to_prompt_mask
            response_mask = response_to_full_mask
            prompt_kwargs = kwargs
            response_kwargs = kwargs

        return prompt_kwargs, response_kwargs, prompt_mask, response_mask

    @staticmethod
    def _compute_response_attention_grouped_unpacked(
        q_packed,
        k_packed,
        v_packed,
        info,
        module,
        attention_interface,
        attention_mask,
        dropout,
        scaling,
        sliding_window,
        **kwargs,
    ):
        # Unpadded case only: q/k/v are [num_heads, total_tokens, head_dim]
        groups = info["prompt_groups"]
        cu_q = info["cu_seqlens_response"]
        cu_kv = info["cu_seqlens_packed"]

        outputs = []
        for group_seqs in groups:
            q_tensors, q_lens = [], []
            k_tensors = []
            v_tensors = []
            for seq_idx in group_seqs:
                q_start, q_end = cu_q[seq_idx].item(), cu_q[seq_idx + 1].item()
                kv_start, kv_end = cu_kv[seq_idx].item(), cu_kv[seq_idx + 1].item()
                q_tensors.append(q_packed[:, q_start:q_end, :])
                k_tensors.append(k_packed[:, kv_start:kv_end, :])
                v_tensors.append(v_packed[:, kv_start:kv_end, :])
                q_lens.append(q_end - q_start)

            max_q = max(q_lens)
            max_kv = max([k.shape[1] for k in k_tensors])
            q_batch = torch.stack([F.pad(t, (0, 0, 0, max_q - t.shape[1])) for t in q_tensors]).transpose(1, 2)
            k_batch = torch.stack([F.pad(t, (0, 0, 0, max_kv - t.shape[1])) for t in k_tensors]).transpose(1, 2)
            v_batch = torch.stack([F.pad(t, (0, 0, 0, max_kv - t.shape[1])) for t in v_tensors]).transpose(1, 2)

            out, _ = attention_interface(
                module,
                q_batch.contiguous(),
                k_batch.contiguous(),
                v_batch.contiguous(),
                attention_mask=None,
                dropout=dropout,
                scaling=scaling,
                sliding_window=sliding_window,
                **kwargs,
            )

            for i, seq_idx in enumerate(group_seqs):
                outputs.append((seq_idx, out[i].transpose(0, 1), q_lens[i]))

        outputs.sort(key=lambda x: x[0])
        return torch.cat([o[1][:, : o[2], :] for o in outputs], dim=1)

    def _create_unpatched_forward_local(self, module, module_name):

        def unpatched_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            global debug_object

            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            apply_rotary_pos_emb = getattr(src_mod, "apply_rotary_pos_emb")
            eager_attention_forward = getattr(src_mod, "eager_attention_forward")
            all_attention_functions = getattr(src_mod, "ALL_ATTENTION_FUNCTIONS", {})

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            # Compute projections
            q_proj_output = module.q_proj(hidden_states).view(hidden_shape)
            k_proj_output = module.k_proj(hidden_states).view(hidden_shape)
            v_proj_output = module.v_proj(hidden_states).view(hidden_shape)

            # Apply normalization
            query_states = module.q_norm(q_proj_output).transpose(1, 2)
            key_states = module.k_norm(k_proj_output).transpose(1, 2)
            value_states = v_proj_output.transpose(1, 2)

            cos, sin = position_embeddings

            # Capture debug data
            if debug_object["baseline_counter"] == debug_object["capture_at_invocation"]:
                debug_object["baseline"] = {}
                debug_object["baseline"]["hidden_states_input"] = hidden_states.clone()
                debug_object["baseline"]["position_embeddings"] = (cos, sin)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, module.layer_idx, cache_kwargs
                )

            attention_interface = eager_attention_forward
            if module.config._attn_implementation != "eager":
                attention_interface = all_attention_functions[module.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,  # diff with Llama
                **kwargs,
            )

            # Capture baseline data if this is the specified invocation
            if debug_object["baseline_counter"] == debug_object["capture_at_invocation"]:
                debug_object["baseline"]["attn_output"] = attn_output
                debug_object["baseline"]["attn_weights"] = attn_weights
                debug_object["baseline"]["query_states"] = query_states
                debug_object["baseline"]["key_states"] = key_states
                debug_object["baseline"]["value_states"] = value_states

                # attention mask
                debug_object["baseline"]["attention_mask"] = attention_mask

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()

            attn_output = module.o_proj(attn_output)
            if debug_object["baseline_counter"] == debug_object["capture_at_invocation"]:
                debug_object["baseline"]["o_proj_output"] = attn_output
            debug_object["baseline_counter"] += 1

            return attn_output, attn_weights

        return unpatched_forward

    def _create_patched_forward(self, module, module_name):
        """
        Create optimized forward for Qwen attention.
        Dispatches to either standard or split attention based on use_split_attention flag.
        """
        if self.use_split_attention:
            layer_id = int(module_name.split("layers.")[-1].split(".")[0])
            return self._create_patched_forward_split_attention(module, module_name, layer_id)
        else:
            return self._create_patched_forward_standard(module, module_name)

    def _create_patched_forward_standard(self, module, module_name):
        """
        Create optimized forward for Qwen attention (standard approach).

        Strategy:
        1. Compute Q, K, V on deduplicated batch
        2. Reconstruct Q, K, V individually
        3. Run attention with reconstructed Q, K, V
        4. Deduplicate output
        """
        reconstruction_info = self.reconstruction_info

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            apply_rotary_pos_emb = getattr(src_mod, "apply_rotary_pos_emb")
            eager_attention_forward = getattr(src_mod, "eager_attention_forward")
            all_attention_functions = getattr(src_mod, "ALL_ATTENTION_FUNCTIONS", {})

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            # 1. Compute Q, K, V on deduplicated batch
            # First project
            q_proj_output_dedup = module.q_proj(hidden_states).view(hidden_shape)
            k_proj_output_dedup = module.k_proj(hidden_states).view(hidden_shape)
            v_proj_output_dedup = module.v_proj(hidden_states).view(hidden_shape)

            # Then normalize
            query_dedup = module.q_norm(q_proj_output_dedup)
            key_dedup = module.k_norm(k_proj_output_dedup)
            value_dedup = v_proj_output_dedup

            # 2. Reconstruct Q, K, V individually to full batch
            query_states = ZoRRoTrain.reconstruct_sequences(query_dedup, reconstruction_info).transpose(1, 2)
            key_states = ZoRRoTrain.reconstruct_sequences(key_dedup, reconstruction_info).transpose(1, 2)
            value_states = ZoRRoTrain.reconstruct_sequences(value_dedup, reconstruction_info).transpose(1, 2)

            cos, sin = position_embeddings

            # Capture debug data
            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"] = {}
                debug_object["patched"]["hidden_states_input"] = ZoRRoTrain.reconstruct_sequences(
                    hidden_states.unsqueeze(0) if hidden_states.dim() == 2 else hidden_states, reconstruction_info
                )
                debug_object["patched"]["position_embeddings"] = (cos, sin)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, module.layer_idx, cache_kwargs
                )

            attention_interface = eager_attention_forward
            if module.config._attn_implementation != "eager":
                attention_interface = all_attention_functions[module.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,  # diff with Llama
                **kwargs,
            )

            # Capture patched data if this is the specified invocation
            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"]["attn_output"] = attn_output
                debug_object["patched"]["attn_weights"] = attn_weights
                debug_object["patched"]["query_states"] = query_states
                debug_object["patched"]["key_states"] = key_states
                debug_object["patched"]["value_states"] = value_states

            # 4. get the reconstructed shape with duplication
            input_shape = ZoRRoTrain.get_reconstructed_shape(input_shape, reconstruction_info)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()

            # 5. Deduplicate output
            attn_output_dedup = ZoRRoTrain.deduplicate_sequences(attn_output, reconstruction_info)

            o_proj_dedup = module.o_proj(attn_output_dedup)
            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"]["o_proj_output"] = ZoRRoTrain.reconstruct_sequences(
                    o_proj_dedup, reconstruction_info
                )
            debug_object["patched_counter"] += 1

            return o_proj_dedup, attn_weights

        return patched_forward

    def _create_patched_forward_split_attention(self, module, module_name, layer_id):
        """
        Create optimized forward with two attention calls:
        1. Prompt-to-prompt attention (deduplicated)
        2. Response-to-(prompt+response) attention

        Dispatches to unpadded or padded version based on reconstruction_info.
        """
        is_unpadded = self.reconstruction_info.get("is_unpadded", False)

        if is_unpadded:
            return self._create_patched_forward_split_attention_unpadded(module, module_name, layer_id)
        else:
            return self._create_patched_forward_split_attention_padded(module, module_name)

    def _create_patched_forward_split_attention_padded(self, module, module_name):
        """
        Split attention for padded sequences (original implementation).
        """
        reconstruction_info = self.reconstruction_info
        # response_length = reconstruction_info['response_length']
        # prompt_groups = reconstruction_info['prompt_groups']

        def patched_forward_split(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            apply_rotary_pos_emb = getattr(src_mod, "apply_rotary_pos_emb")
            eager_attention_forward = getattr(src_mod, "eager_attention_forward")
            all_attention_functions = getattr(src_mod, "ALL_ATTENTION_FUNCTIONS", {})

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            # Step 1: Compute Q, K, V on deduplicated batch
            # First project
            q_proj_output_dedup = module.q_proj(hidden_states).view(hidden_shape)
            k_proj_output_dedup = module.k_proj(hidden_states).view(hidden_shape)
            v_proj_output_dedup = module.v_proj(hidden_states).view(hidden_shape)

            # Then normalize
            query_dedup = module.q_norm(q_proj_output_dedup)
            key_dedup = module.k_norm(k_proj_output_dedup)
            value_dedup = v_proj_output_dedup

            # Step 2: Reconstruct Q, K, V to full batch
            query_states = ZoRRoTrain.reconstruct_sequences(query_dedup, reconstruction_info).transpose(1, 2)
            key_states = ZoRRoTrain.reconstruct_sequences(key_dedup, reconstruction_info).transpose(1, 2)
            value_states = ZoRRoTrain.reconstruct_sequences(value_dedup, reconstruction_info).transpose(1, 2)

            # Apply RoPE
            cos, sin = position_embeddings

            # Capture debug data
            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"] = {}
                debug_object["patched"]["hidden_states_input"] = ZoRRoTrain.reconstruct_sequences(
                    hidden_states.unsqueeze(0) if hidden_states.dim() == 2 else hidden_states, reconstruction_info
                )
                debug_object["patched"]["position_embeddings"] = (cos, sin)

            # pr0(f"{query_states.shape=}")
            # pr0(f"{key_states.shape=}")
            # pr0(f"{cos.shape=}")
            # pr0(f"{sin.shape=}")

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # query_states, key_states, value_states: [batch_size, num_heads, seq_len, head_dim]
            # batch_size = query_states.shape[0]
            # num_heads = query_states.shape[1]

            # Get attention interface
            attention_interface = eager_attention_forward
            if module.config._attn_implementation != "eager":
                attention_interface = all_attention_functions[module.config._attn_implementation]

            # Extract unique prompts and compute prompt-to-prompt attention
            prompt_q = ZoRRoTrain.extract_and_deduplicate_prompts(query_states, reconstruction_info)
            prompt_k = ZoRRoTrain.extract_and_deduplicate_prompts(key_states, reconstruction_info)
            prompt_v = ZoRRoTrain.extract_and_deduplicate_prompts(value_states, reconstruction_info)

            prompt_len = reconstruction_info["prompt_len"]
            prompt_groups = reconstruction_info["prompt_groups"]

            # Extract attention masks for unique prompts (first sample in each group)
            if attention_mask is not None:
                unique_prompt_indices = [group[0] for group in prompt_groups]
                prompt_mask = attention_mask[unique_prompt_indices, :, :prompt_len, :prompt_len]
            else:
                prompt_mask = None

            # Compute prompt-to-prompt attention (single batched call)
            prompt_outputs, _ = attention_interface(
                module,
                prompt_q,
                prompt_k,
                prompt_v,
                prompt_mask,  # Use prompt-only block diagonal mask
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                **kwargs,
            )
            # prompt_outputs: [num_unique_prompts, prompt_len, num_heads, head_dim]

            # Extract response queries for response-to-full attention
            response_q = ZoRRoTrain.extract_response_queries(query_states, reconstruction_info)
            response_mask = attention_mask[:, :, prompt_len:, :] if attention_mask is not None else None

            # Compute response-to-(prompt+response) attention (single batched call)
            response_outputs, _ = attention_interface(
                module,
                response_q,
                key_states,  # Already has full context (prompt + response)
                value_states,  # Already has full context (prompt + response)
                response_mask,  # Use response-to-full mask
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                **kwargs,
            )
            # response_outputs: [batch_size, response_len, num_heads, head_dim]

            # Step 5: Replicate prompt outputs and concatenate with response outputs
            attn_output = ZoRRoTrain.replicate_and_concat_prompt_response(
                prompt_outputs, response_outputs, reconstruction_info
            )

            # attn_output: [batch_size, seq_len, num_heads, head_dim]

            # Capture patched data if this is the specified invocation
            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"]["attn_output"] = attn_output  # [batch, seq, heads, dim]
                debug_object["patched"]["attn_weights"] = None
                debug_object["patched"]["query_states"] = query_states
                debug_object["patched"]["key_states"] = key_states
                debug_object["patched"]["value_states"] = value_states

                # compare_debug_tensors(debug_object)

            # Reshape to [batch_size, seq_len, num_heads * head_dim]
            original_batch_size = reconstruction_info["original_batch_size"]
            seq_len = reconstruction_info["original_seq_len"]

            # attn_output = attn_output.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
            attn_output = attn_output.reshape(original_batch_size, seq_len, -1).contiguous()
            # Step 6: Deduplicate
            attn_output_dedup = ZoRRoTrain.deduplicate_sequences(attn_output, reconstruction_info)

            # Apply output projection
            o_proj_dedup = module.o_proj(attn_output_dedup)

            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"]["o_proj_output"] = ZoRRoTrain.reconstruct_sequences(
                    o_proj_dedup, reconstruction_info
                )
            debug_object["patched_counter"] += 1

            return o_proj_dedup, None

        return patched_forward_split

    def _create_patched_forward_split_attention_unpadded(self, module, module_name, layer_id):
        """
        Split attention for unpadded sequences (packed format).
        Uses Flash Attention varlen API through attention_interface.

        With Flash Attention 2.1+:
        - Response-to-full attention uses bottom-right aligned causal mask
        - Responses can attend to ALL prompt tokens + causally to response tokens
        - No explicit attention mask needed, just cu_seqlens with causal=True

        This implementation mirrors the padded version as closely as possible,
        with the only difference being the flash_kwargs that use cumulative_seqlens
        for packed sequences instead of attention masks.
        """
        reconstruction_info = self.reconstruction_info
        # response_length = reconstruction_info['response_length']
        # prompt_groups = reconstruction_info['prompt_groups']
        # cu_seqlens_dedup = reconstruction_info['cu_seqlens_dedup']
        # cu_seqlens_packed = reconstruction_info['cu_seqlens_packed']
        # max_seqlen_dedup = reconstruction_info['max_seqlen_dedup']
        # max_seqlen_packed = reconstruction_info['max_seqlen_packed']

        def Dedup_Cosine_Sine_Coeff(cos, sin, cu_seqlens_dedup, cu_seqlens_response, reconstruction_info):

            RUN = False

            if RUN:
                splits = cu_seqlens_dedup[1:] - cu_seqlens_dedup[:-1]
                splits2 = cu_seqlens_response[1:] - cu_seqlens_response[:-1]
                assert splits.numel() == splits2.numel(), (
                    f"Mismatch in dedup cosine parts for the response and prompt parts: {splits.numel()} !="
                    f" {splits2.numel()}"
                )
                splits3 = [0] * splits.numel() * 2
                s1 = (splits - splits2).tolist()
                s2 = splits2.tolist()
                splits3[::2] = s1
                splits3[1::2] = s2
                cos_parts = torch.split(cos, splits3, dim=1)
                sin_parts = torch.split(sin, splits3, dim=1)
            group_sizes = torch.tensor(
                [0] + [len(group) for group in reconstruction_info["prompt_groups"]], device=cos.device
            ).cumsum_(0)
            if RUN:
                cos_part1 = [cos_parts[group_sizes[i] * 2] for i in range(len(group_sizes) - 1)]
                sin_part1 = [sin_parts[group_sizes[i] * 2] for i in range(len(group_sizes) - 1)]
                cos_part2 = cos_parts[1::2]
                sin_part2 = sin_parts[1::2]
                all_cos_parts = []
                all_sin_parts = []

                for i, (cp1, sp1) in enumerate(zip(cos_part1, sin_part1)):
                    all_cos_parts.extend([cp1] + [cos_part2[j] for j in range(group_sizes[i], group_sizes[i + 1])])
                    all_sin_parts.extend([sp1] + [sin_part2[j] for j in range(group_sizes[i], group_sizes[i + 1])])
                cos_dedup = torch.cat(all_cos_parts, dim=1)
                sin_dedup = torch.cat(all_sin_parts, dim=1)
            else:
                cos_dedup = cos
                sin_dedup = sin
                return cos_dedup, sin_dedup, group_sizes

            cos_part1 = [cos_parts[group_sizes[i] * 2] for i in range(len(group_sizes) - 1)]
            sin_part1 = [sin_parts[group_sizes[i] * 2] for i in range(len(group_sizes) - 1)]
            cos_part2 = cos_parts[1::2]
            sin_part2 = sin_parts[1::2]
            all_cos_parts = []
            all_sin_parts = []

            for i, (cp1, sp1) in enumerate(zip(cos_part1, sin_part1)):
                all_cos_parts.extend([cp1] + [cos_part2[j] for j in range(group_sizes[i], group_sizes[i + 1])])
                all_sin_parts.extend([sp1] + [sin_part2[j] for j in range(group_sizes[i], group_sizes[i + 1])])
            cos_dedup = torch.cat(all_cos_parts, dim=1)
            sin_dedup = torch.cat(all_sin_parts, dim=1)
            return cos_dedup, sin_dedup, group_sizes

        def patched_forward_split_unpadded(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            use_cumsum_mask=False,  # FA 2.1+: Use varlen API with bottom-right aligned causal mask
            **kwargs,
        ):

            cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]
            if attention_mask is None:
                use_cumsum_mask = True

            # Access helpers from the original defining module
            src_mod = sys.modules[type(module).__module__]
            apply_rotary_pos_emb = getattr(src_mod, "apply_rotary_pos_emb")
            eager_attention_forward = getattr(src_mod, "eager_attention_forward")
            all_attention_functions = getattr(src_mod, "ALL_ATTENTION_FUNCTIONS", {})

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            # Step 1: Compute Q, K, V on deduplicated batch
            query_dedup = module.q_norm(module.q_proj(hidden_states).view(hidden_shape))
            key_dedup = module.k_norm(module.k_proj(hidden_states).view(hidden_shape))
            value_dedup = module.v_proj(hidden_states).view(hidden_shape)

            cos, sin = position_embeddings
            cu_seqlens_response = ZoRRoTrain._extract_cu_seqlens(reconstruction_info, hidden_states.device)
            # if layer_id == 0:
            #     global cos_dedup, sin_dedup, group_sizes
            cos_dedup, sin_dedup, group_sizes = Dedup_Cosine_Sine_Coeff(
                cos, sin, cu_seqlens_packed.to(cu_seqlens_response.device), cu_seqlens_response, reconstruction_info
            )

            # Capture debug data
            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"] = {}
                debug_object["patched"]["hidden_states_input"] = ZoRRoTrain.reconstruct_sequences(
                    hidden_states.unsqueeze(0) if hidden_states.dim() == 2 else hidden_states, reconstruction_info
                )
                debug_object["patched"]["position_embeddings"] = (cos, sin)
            qs, ks = apply_rotary_pos_emb(
                query_dedup.transpose(1, 2), key_dedup.transpose(1, 2), cos_dedup, sin_dedup, unsqueeze_dim=1
            )

            # Get attention interface
            attention_interface = eager_attention_forward
            if module.config._attn_implementation != "eager":
                attention_interface = all_attention_functions[module.config._attn_implementation]

            # Extract unique prompts and compute prompt-to-prompt attention
            prompt_q = ZoRRoTrain._get_sequences_packed_from_dedup_tensor(qs, reconstruction_info, cu_seqlens_response)
            prompt_k = ZoRRoTrain._get_sequences_packed_from_dedup_tensor(ks, reconstruction_info, cu_seqlens_response)
            prompt_v = ZoRRoTrain._get_sequences_packed_from_dedup_tensor(
                value_dedup.transpose(1, 2), reconstruction_info, cu_seqlens_response
            )
            response_q = ZoRRoTrain._get_responses_packed_from_dedup_tensor(
                qs, reconstruction_info, cu_seqlens_response
            )

            # Get cu_seqlens for unique prompts and response
            cu_seqlens_unique_prompts = reconstruction_info["cu_seqlens_unique_prompts"]
            cu_seqlens_response = reconstruction_info["cu_seqlens_response"]

            # Prepare attention kwargs and masks
            prompt_kwargs, response_kwargs, prompt_mask, response_mask = self._prepare_attention_kwargs_and_masks(
                use_cumsum_mask,
                reconstruction_info,
                cu_seqlens_unique_prompts,
                cu_seqlens_response,
                cu_seqlens_packed,
                attention_mask,
                hidden_states.device,
                kwargs,
            )

            # Compute prompt-to-prompt attention
            prompt_outputs, _ = attention_interface(
                module,
                prompt_q.contiguous(),
                prompt_k.contiguous(),
                prompt_v.contiguous(),
                attention_mask=prompt_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                **prompt_kwargs,
            )

            # Compute response-to-full attention
            key_states = ZoRRoTrain._get_sequences_reconstructed_from_dedup_tensors(
                ks, prompt_k, reconstruction_info, cu_seqlens_response, group_sizes
            )
            value_states = ZoRRoTrain._get_sequences_reconstructed_from_dedup_tensors(
                value_dedup.transpose(1, 2), prompt_v, reconstruction_info, cu_seqlens_response, group_sizes
            )

            response_outputs, _ = attention_interface(
                module,
                response_q.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                attention_mask=response_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                sliding_window=module.sliding_window,
                **response_kwargs,
            )
            attn_output_dedup = ZoRRoTrain._replicate_and_concat_prompt_responses(
                prompt_outputs, response_outputs, reconstruction_info, cu_seqlens_response
            )

            # Capture patched data if this is the specified invocation
            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"]["attn_output"] = attn_output_dedup  # [batch, seq, heads, dim]
                debug_object["patched"]["attn_weights"] = None
                debug_object["patched"]["query_states"] = qs
                debug_object["patched"]["key_states"] = ks
                debug_object["patched"]["value_states"] = value_states
                debug_object["patched"]["prompt_mask"] = prompt_mask
                debug_object["patched"]["response_mask"] = response_mask

            # Reshape to [1, total_valid_tokens, num_heads * head_dim]
            # For packed format, attn_output is [1, total_valid_tokens, num_heads, head_dim]
            batch_size_packed = attn_output_dedup.shape[0]  # Should be 1
            total_valid_tokens = attn_output_dedup.shape[1]  # Actual number of valid tokens

            attn_output_dedup = attn_output_dedup.reshape(batch_size_packed, total_valid_tokens, -1).contiguous()
            # Apply output projection
            o_proj_dedup = module.o_proj(attn_output_dedup)

            if debug_object["patched_counter"] == debug_object["capture_at_invocation"]:
                debug_object["patched"]["o_proj_output"] = ZoRRoTrain.reconstruct_sequences(
                    o_proj_dedup, reconstruction_info
                )
            debug_object["patched_counter"] += 1

            return o_proj_dedup, None

        return patched_forward_split_unpadded


class QwenAttentionOncePatcher(QwenAttentionPatcher):
    def __init__(self, model, reconstruction_info, patch_with_local=False, use_split_attention=True):
        """
        Same as QwenAttentionPatcher, but not using dynamic patching enter/exit - patching the model once at init instead

        This class doesn't use the grand-parent ModuleReconstructionPatcher so just overriding _init__
        """
        self.model = model
        self.reconstruction_info = reconstruction_info
        self.use_split_attention = use_split_attention

        """Patch all module forward methods."""
        for name, module in self.model.named_modules():
            if self._should_patch_module_forward(name, module):
                # Create patched forward that optimizes QKV
                module.forward = self._create_patched_forward(module, name)

                # # Store original forward
                # self.original_forwards[name] = module.forward

                # if self.patch_with_local:
                #     assert self._create_unpatched_forward_local is not None, "Subclass must implement _create_unpatched_forward_local"
                #     module.forward = self._create_unpatched_forward_local(module, name)
                # else:
                #     # Create patched forward that optimizes QKV
                #     module.forward = self._create_patched_forward(module, name)


def compare_debug_tensors(debug_object, atol=1e-5, rtol=1e-5, verbose=True, num_samples=10):
    """Compare tensors in debug_object['baseline'] vs ['patched'].
    Prints a summary and returns a dict of results per field.

    Args:
        debug_object: Debug object to compare
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        verbose: Whether to print comparison results
        num_samples: Number of sample values to print from each tensor
    """
    fields = ["hidden_states_input", "query_states", "key_states", "value_states", "attn_output", "attn_weights"]

    def _compare(a, b, field_name):
        result = {
            "exists": (a is not None) and (b is not None),
            "shape_equal": False,
            "equal": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
        if not result["exists"]:
            return result
        if hasattr(a, "shape") and hasattr(b, "shape"):
            result["shape_equal"] = tuple(a.shape) == tuple(b.shape)
        if not result["shape_equal"]:
            return result
        a32 = a.detach().float()
        b32 = b.detach().float()
        diff = (a32 - b32).abs()
        result["max_abs_diff"] = diff.max().item() if diff.numel() > 0 else 0.0
        result["mean_abs_diff"] = diff.mean().item() if diff.numel() > 0 else 0.0
        result["equal"] = torch.allclose(a32, b32, rtol=rtol, atol=atol)

        # Find location of max difference
        if result["max_abs_diff"] and result["max_abs_diff"] > 0:
            max_idx = diff.argmax()
            max_idx_unraveled = torch.unravel_index(max_idx, diff.shape)
            result["max_diff_location"] = tuple(idx.item() for idx in max_idx_unraveled)
            result["max_diff_baseline_value"] = a32[max_idx_unraveled].item()
            result["max_diff_patched_value"] = b32[max_idx_unraveled].item()

            # Count how many values have large errors (> 0.1)
            large_errors = (diff > 0.1).sum().item()
            result["num_large_errors"] = large_errors
            if large_errors > 0:
                result["pct_large_errors"] = 100.0 * large_errors / diff.numel()

        # Store sample values for printing
        if verbose and result["shape_equal"]:
            # Flatten and take first num_samples values
            a_flat = a32.flatten()
            b_flat = b32.flatten()
            n_samples = min(num_samples, a_flat.numel())
            result["baseline_samples"] = a_flat[:n_samples].cpu().numpy()
            result["patched_samples"] = b_flat[:n_samples].cpu().numpy()

        return result

    if "baseline" not in debug_object or "patched" not in debug_object:
        if verbose:
            pr0("debug_object missing 'baseline' or 'patched'.")
        pr0(f"debug_object = {debug_object.keys()}")
        return None

    baseline = debug_object["baseline"]
    patched = debug_object["patched"]
    if baseline is None or patched is None:
        if verbose:
            pr0("debug_object entries are None. Capture baseline and patched before comparing.")
        return None

    report = {}
    for f in fields:
        a = baseline.get(f, None)
        b = patched.get(f, None)
        report[f] = _compare(a, b, f)

    if verbose:
        pr0("\n" + "=" * 80)
        pr0("TENSOR COMPARISON REPORT")
        pr0("=" * 80)

        for f in fields:
            r = report[f]
            pr0(f"\n{f}:")
            pr0(f"  Shape Equal: {r['shape_equal']}")
            pr0(f"  Values Equal (atol={atol}, rtol={rtol}): {r['equal']}")
            pr0(f"  Max Abs Diff: {r['max_abs_diff']}")
            pr0(f"  Mean Abs Diff: {r['mean_abs_diff']}")

            # Print location of max difference if it exists and is significant
            if r.get("max_abs_diff") and r["max_abs_diff"] > 1e-4:
                if "max_diff_location" in r:
                    pr0(f"  Max Diff Location: {r['max_diff_location']}")
                    pr0(f"    Baseline value at max: {r['max_diff_baseline_value']}")
                    pr0(f"    Patched value at max:  {r['max_diff_patched_value']}")
                if "num_large_errors" in r:
                    pr0(f"  Large errors (>0.1): {r['num_large_errors']} ({r.get('pct_large_errors', 0):.3f}%)")

            # Print sample values if available
            # if 'baseline_samples' in r and 'patched_samples' in r:
            #     pr0(f"  First {num_samples} values:")
            #     pr0(f"    Baseline: {r['baseline_samples']}")
            #     pr0(f"    Patched:  {r['patched_samples']}")
            #     # Print differences for each sample
            #     diffs = r['baseline_samples'] - r['patched_samples']
            #     pr0(f"    Diffs:    {diffs}")

        pr0("\n" + "=" * 80)

    return report
