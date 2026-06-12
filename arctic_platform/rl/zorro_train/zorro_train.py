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
Prompt deduplication utilities.

Identifies and deduplicates shared prompts across batch samples.
Supports both padded and unpadded sequences.
"""

import functools
import operator
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from arctic_platform.rl.utils.debug import pr
from arctic_platform.rl.utils.debug import pr0

# import numpy as np


def analyze_normal_batch_via_attention_mask(input_ids, attention_mask, padded_response_len):
    """
    When debugging shapes mismatch this helper util will dump shapes and count of pad vs non-pad tokens separately for prompts and responses. It's dealing with a normal non-zorro packed - batch of `[bs, left-pad-prompt+right-pad-response]` shape.

    Args:
      - input_ids: input_ids
      - attention_mask: 1D attention mask that can tell padded from non-padded tokens
      - padded_response_len: padded response len - think the second component of `[left-pad-prompt | right-pad-response]` - we derive prompt-len from total len

    Here is an example of an output:

    [0] *** Batch shape analysis: ***
    [0] full_attention_mask.shape=torch.Size([16, 20480])
    [0] padded_prompt_len=16384
    [0] padded_response_len=4096
    [0] sorted(prompts_pad_lens.tolist())=[133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133]
    [0] sorted(responses_pad_lens.tolist())=[3054, 3281, 3351, 3363, 3412, 3413, 3430, 3430, 3465, 3466, 3519, 3521, 3548, 3551, 3626, 3707]
    [0] prompts_pad_lens_total.item()=2128
    [0] responses_pad_lens_total.item()=55137
    [0] prompts_non_pad_lens_total.item()=260016
    [0] responses_non_pad_len_total.item()=10399
    [0] sorted(unique_prompts_lens)=[tensor(16251)]
    [0] len(unique_prompt_ids)=1
    [0] sorted(responses_non_pad_lens.tolist())=[389, 470, 545, 548, 575, 577, 630, 631, 666, 666, 683, 684, 733, 745, 815, 1042]
    [0] total_deduped_prompt_and_response_non_pad_tokens=tensor(26650)
    [0] non_pad_len_total=tensor(270415)
    [0] pad_len_total=tensor(57265)
    [0] non_pad_percent=82.52%
    [0] batch stats: prompt lens: 15.9Ki resp count=16/min=0.4Ki/mean=0.6Ki/max=1.0Ki non-pad-tokens=26.6K/82.5%

    """
    batch_size = input_ids.shape[0]
    padded_prompt_len = attention_mask.shape[1] - padded_response_len
    input_ids = input_ids.clone().detach().cpu()

    full_attention_mask = attention_mask.clone().detach().cpu()
    prompts_attention_mask = full_attention_mask[:, :padded_prompt_len]
    responses_attention_mask = full_attention_mask[:, padded_prompt_len:]

    prompts_non_pad_lens = prompts_attention_mask.sum(dim=1)
    prompts_pad_lens = padded_prompt_len - prompts_non_pad_lens
    prompts_pad_lens_total = prompts_pad_lens.sum()
    prompts_non_pad_lens_total = prompts_attention_mask.numel() - prompts_pad_lens_total

    responses_non_pad_lens = responses_attention_mask.sum(dim=1)
    responses_pad_lens = padded_response_len - responses_non_pad_lens
    responses_pad_lens_total = responses_pad_lens.sum()
    responses_non_pad_len_total = responses_attention_mask.numel() - responses_pad_lens_total

    non_pad_len_total = full_attention_mask.sum()
    pad_len_total = full_attention_mask.numel() - non_pad_len_total
    non_pad_percent = non_pad_len_total / full_attention_mask.numel() * 100

    prompt_ids = input_ids[:, :padded_prompt_len]
    unique_prompts = {tuple(prompt_ids[i].tolist()): prompts_attention_mask[i].sum() for i in range(batch_size)}
    unique_prompts_lens = list(unique_prompts.values())
    # unique_prompts = {}
    # for prompt_id in prompt_ids.tolist():
    #     unique_prompts[prompt_id] = prompts_attention_mask[prompt_id].sum()
    total_unique_prompt_non_pad_tokens = sum(v for v in unique_prompts.values())
    total_deduped_prompt_and_response_non_pad_tokens = total_unique_prompt_non_pad_tokens + responses_non_pad_len_total

    unique_prompt_ids = set(tuple(x) for x in prompt_ids.tolist())

    responses_non_pad_lens = responses_non_pad_lens.numpy()
    responses_non_pad_lens_min_in_k = responses_non_pad_lens.min()
    responses_non_pad_lens_mean_in_k = responses_non_pad_lens.mean()
    responses_non_pad_lens_max_in_k = responses_non_pad_lens.max()

    pr0("*** Batch shape analysis: ***")
    pr0(f"{full_attention_mask.shape=}")
    pr0(f"{padded_prompt_len=}")
    pr0(f"{padded_response_len=}")

    pr0(f"{sorted(prompts_pad_lens.tolist())=}")
    pr0(f"{sorted(responses_pad_lens.tolist())=}")

    pr0(f"{prompts_pad_lens_total.item()=}")
    pr0(f"{responses_pad_lens_total.item()=}")

    pr0(f"{prompts_non_pad_lens_total.item()=}")
    pr0(f"{responses_non_pad_len_total.item()=}")

    # pr0(f"{sorted(prompts_non_pad_lens)=}")
    pr0(f"{sorted(unique_prompts_lens)=}")
    pr0(f"{len(unique_prompt_ids)=}")
    pr0(f"{sorted(responses_non_pad_lens.tolist())=}")
    # pr0(f"{min_responses_non_pad_lens)=}")
    pr0(f"{total_deduped_prompt_and_response_non_pad_tokens=}")

    pr0(f"{non_pad_len_total=}")
    pr0(f"{pad_len_total=}")
    pr0(f"{non_pad_percent=:0.2f}%")

    unique_prompts_lens_in_k = [f"{x/1024:0.1f}Ki" for x in sorted(unique_prompts_lens)]
    if len(unique_prompts_lens_in_k) == 1:
        # smaller printout since usually for zorro it's one prompt group per mbs
        unique_prompts_lens_in_k = unique_prompts_lens_in_k[0]

    # this is the one line summary so that it's easy to grep for - and print it for all ranks
    pr(
        f"batch stats: prompt lens: {unique_prompts_lens_in_k} resp"
        f" count={len(responses_non_pad_lens)}/min={responses_non_pad_lens_min_in_k/1024:0.1f}Ki/mean={responses_non_pad_lens_mean_in_k/1024:0.1f}Ki/max={responses_non_pad_lens_max_in_k/1024:0.1f}Ki"
        f" non-pad-tokens={total_deduped_prompt_and_response_non_pad_tokens/1000:0.1f}K/{non_pad_percent:0.1f}%"
    )


class ReconstructionInfo(dict):
    """this class is used to ensure the object persists and can be used as a closure and doesn't get replaced with a new dict object. It forces the use of `update` to overwrite the contents of the object

    x = ReconstructionInfo(a=1, b=2)
    pr0(id(x))
    x.update(a=2, b=3)
    pr0(id(x)) # same object different contents

    """

    pass


@triton.jit
def _triton_reconstruct_seq(
    total_seq_tensor_ptr,
    prompt_ptr,
    response_ptr,
    cu_seqlens_prompt_ptr,
    cu_seqlens_response_ptr,
    cu_seqlens_packed_ptr,
    prompt_ids_ptr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
):

    pid = tl.program_id(0).to(tl.int64)
    sid = tl.program_id(1).to(tl.int64)
    group_id = tl.load(prompt_ids_ptr + pid)

    prompt_start = cu_seqlens_prompt_ptr[group_id]
    prompt_end = cu_seqlens_prompt_ptr[group_id + 1]
    response_start = cu_seqlens_response_ptr[pid]

    seq_offset = cu_seqlens_packed_ptr[group_id]

    hids = tl.arange(0, head_dim * num_heads)
    if sid < (prompt_end - prompt_start):
        # copy prompt tokens
        prompt_ptrs = prompt_ptr + (prompt_start + sid) * head_dim * num_heads + hids
        total_seq_tensor_ptrs = total_seq_tensor_ptr + (seq_offset + sid) * head_dim * num_heads + hids
        tl.store(total_seq_tensor_ptrs, tl.load(prompt_ptrs))

    else:
        # copy response tokens
        response_sid = sid - (prompt_end - prompt_start)
        response_ptrs = response_ptr + (response_start + response_sid) * head_dim * num_heads + hids
        total_seq_tensor_ptrs = total_seq_tensor_ptr + (seq_offset + sid) * head_dim * num_heads + hids
        tl.store(total_seq_tensor_ptrs, tl.load(response_ptrs))


def reconstruct_sequence(
    total_seq_tensor,
    prompt_tensor,
    response_tensor,
    cu_seqlens_prompt,
    cu_seqlens_response,
    cu_seqlens_packed,
    prompt_ids,
    head_dim,
):
    """
    Reconstruct full sequence from deduplicated prompt and response tensors using Triton.

    Args:
        total_seq_tensor: [1, total_valid_tokens, num_heads * head_dim] - output tensor to write reconstructed
          sequence into
        prompt_tensor: [1, total_unique_prompt_tokens, num_heads * head_dim] - deduplicated prompts
        response_tensor: [1, total_valid_response_tokens, num_heads * head_dim] - deduplicated responses
        cu_seqlens_prompt: [num_groups + 1] cumulative sequence lengths for prompts
        cu_seqlens_response: [total_valid_tokens + 1] cumulative sequence lengths for responses
        cu_seqlens_packed: [num_samples + 1] cumulative sequence lengths for packed format
        prompt_ids: [num_samples] mapping from each sample to its prompt group ID
    """
    num_heads = prompt_tensor.shape[2] // head_dim
    total_valid_tokens = total_seq_tensor.shape[1]
    grid = (prompt_ids.shape[0], total_valid_tokens)

    _triton_reconstruct_seq[grid](
        total_seq_tensor,
        prompt_tensor,
        response_tensor,
        cu_seqlens_prompt,
        cu_seqlens_response,
        cu_seqlens_packed,
        prompt_ids,
        head_dim=head_dim,
        num_heads=num_heads,
    )


class ZoRRoTrain:
    """Handles prompt deduplication logic."""

    @staticmethod
    def find_prompt_groups(
        input_ids: torch.Tensor,
        response_length: int,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """
        Find which samples share the same prompt.

        Args:
            input_ids: [batch_size, seq_len] - concatenated prompt + response
            response_length: length of response portion

        Returns:
            prompt_groups: List of lists, each inner list contains indices sharing a prompt
            unique_prompts: [num_unique, prompt_len] - unique prompt tensors
        """
        batch_size, seq_len = input_ids.shape
        prompt_len = seq_len - response_length

        # Extract prompts (simple slice, no padding)
        prompts = input_ids[:, :prompt_len]

        # Group by prompt equality
        prompt_groups = []
        unique_prompts = []

        for i in range(batch_size):
            # Check if this prompt matches any existing group
            found_group = False
            for group in prompt_groups:
                # Compare with first sample in group
                representative = group[0]
                if torch.equal(prompts[i], prompts[representative]):
                    group.append(i)
                    found_group = True
                    break

            if not found_group:
                # New unique prompt
                prompt_groups.append([i])
                unique_prompts.append(prompts[i])

        unique_prompts_tensor = torch.stack(unique_prompts)

        return prompt_groups, unique_prompts_tensor

    @staticmethod
    def _dedup_input_ids(
        input_ids: torch.Tensor,
        response_length: int,
        prompt_groups: List[List[int]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Deduplicate input_ids by removing duplicate prompts.

        Args:
            input_ids: [batch_size, seq_len]
            response_length: Length of response section
            prompt_groups: List of lists with sample indices sharing same prompt
            attention_mask: [batch_size, seq_len] - optional

        Returns:
            dedup_input_ids: [1, total_dedup_tokens]
            dedup_attention_mask: [1, total_dedup_tokens] or None (only if unpadding needed)
            reconstruction_info: Dict with reconstruction metadata
        """
        batch_size, seq_len = input_ids.shape
        prompt_len = seq_len - response_length

        # Build concatenated sequence: [prompt_1, response_1_1, response_1_2, ...]
        concatenated_tokens = []
        segment_info = []
        current_pos = 0

        for group_idx, group in enumerate(prompt_groups):
            # Add prompt once per group
            first_sample = group[0]
            prompt_tokens = input_ids[first_sample, :prompt_len]
            concatenated_tokens.append(prompt_tokens)

            # Record prompt segment for all samples in group
            prompt_start = current_pos
            prompt_end = current_pos + prompt_len

            for sample_idx in group:
                segment_info.append(
                    {
                        "start": prompt_start,
                        "end": prompt_end,
                        "original_idx": sample_idx,
                        "type": "prompt",
                        "group_idx": group_idx,
                    }
                )

            current_pos += prompt_len

            # Add each response in the group
            for sample_idx in group:
                response_tokens = input_ids[sample_idx, prompt_len:]
                concatenated_tokens.append(response_tokens)

                # Record response segment
                response_start = current_pos
                response_end = current_pos + response_length

                segment_info.append(
                    {
                        "start": response_start,
                        "end": response_end,
                        "original_idx": sample_idx,
                        "type": "response",
                        "group_idx": group_idx,
                    }
                )

                current_pos += response_length

        # Concatenate into single sequence with batch_size=1
        dedup_input_ids = torch.cat(concatenated_tokens).unsqueeze(0)  # [1, total_dedup_tokens]

        # Create attention mask for deduplicated batch (if provided)
        if attention_mask is not None:
            concatenated_mask = []
            for group_idx, group in enumerate(prompt_groups):
                first_sample = group[0]
                prompt_mask = attention_mask[first_sample, :prompt_len]
                concatenated_mask.append(prompt_mask)

                for sample_idx in group:
                    response_mask = attention_mask[sample_idx, prompt_len:]
                    concatenated_mask.append(response_mask)

            dedup_attention_mask = torch.cat(concatenated_mask).unsqueeze(0)  # [1, total_dedup_tokens]
        else:
            dedup_attention_mask = None

        reconstruction_info = {
            "segment_info": segment_info,
            "prompt_groups": prompt_groups,
            "prompt_len": prompt_len,
            "response_length": response_length,
            "original_batch_size": batch_size,
            "original_seq_len": seq_len,
            "is_unpadded": False,
            "original_attention_mask": attention_mask,
        }

        return dedup_input_ids, dedup_attention_mask, reconstruction_info

    @staticmethod
    def _unpad_dedup_input_ids(
        dedup_input_ids: torch.Tensor,
        dedup_attention_mask: torch.Tensor,
        reconstruction_info: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Remove padding from deduplicated input_ids.

        Args:
            dedup_input_ids: [1, total_dedup_tokens] - with padding
            dedup_attention_mask: [1, total_dedup_tokens]
            reconstruction_info: Dict

        Returns:
            dedup_input_ids_unpad: [1, total_dedup_valid_tokens] - without padding
            reconstruction_info_updated: Dict with unpadding metadata
        """
        # Reuse existing unpad_deduplicated_batch logic
        # We pass a dummy position_ids since we'll handle that separately
        dummy_position_ids = torch.zeros_like(dedup_input_ids)
        dedup_input_ids_unpad, _, reconstruction_info_updated = ZoRRoTrain.unpad_deduplicated_batch(
            dedup_input_ids, dummy_position_ids, dedup_attention_mask, reconstruction_info
        )
        return dedup_input_ids_unpad, reconstruction_info_updated

    @staticmethod
    def _unpad_replicated_ids(
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove padding from original (non-deduplicated) position_ids.

        Args:
            position_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            position_ids_unpad: [1, total_all_valid_tokens] - packed, without padding
        """
        batch_size = ids.shape[0]
        unpadded_position_ids = []

        for i in range(batch_size):
            sample_mask = attention_mask[i].bool()  # [seq_len]
            sample_pos = ids[i, sample_mask]  # [num_valid_tokens]
            unpadded_position_ids.append(sample_pos)

        # Concatenate all samples into packed format: [1, total_valid_tokens_all_samples]
        return torch.cat(unpadded_position_ids).unsqueeze(0)

    @staticmethod
    def attention_mask_to_flash_attn_params(
        attention_mask: torch.Tensor, device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert packed attention mask to Flash Attention varlen parameters.

        This method extracts sequence boundaries from a sparse attention mask
        that represents multiple packed sequences. The number of sequences is the same
        for Q and K, but each sequence can have different token counts (e.g., with KV caching).

        Args:
            attention_mask: [total_q_tokens, total_kv_tokens] - Sparse block-diagonal mask
                           where each block represents one packed sequence. 1 for valid attention,
                           0 for masked positions. Can be square or non-square.
            device: Device to create tensors on (defaults to attention_mask.device)

        Returns:
            dict containing:
                - cu_seqlens_q: [num_sequences + 1] cumulative sequence lengths for queries
                - cu_seqlens_k: [num_sequences + 1] cumulative sequence lengths for keys/values
                - max_seqlen_q: int, maximum query sequence length
                - max_seqlen_k: int, maximum key/value sequence length
                - total_q_tokens: int, total number of query tokens
                - total_kv_tokens: int, total number of key/value tokens
                - num_sequences: int, number of packed sequences
                - seq_lengths_q: list of query sequence lengths [len_seq0, len_seq1, ...]
                - seq_lengths_k: list of key sequence lengths [len_seq0, len_seq1, ...]

        Example 1 (self-attention, square mask):
            3 sequences, Q and K both [5, 3, 4]: mask is [12, 12]
            → cu_seqlens_q = [0, 5, 8, 12], cu_seqlens_k = [0, 5, 8, 12]

        Example 2 (with KV cache, non-square):
            3 sequences, Q: [2, 2, 2] new tokens, K: [5, 3, 4] full sequences
            mask is [6, 12]
            → cu_seqlens_q = [0, 2, 4, 6], cu_seqlens_k = [0, 5, 8, 12]
        """
        if device is None:
            device = attention_mask.device

        # Handle different attention_mask shapes
        if attention_mask.dim() == 4:
            attention_mask = attention_mask.squeeze(0).squeeze(0)  # [total_q, total_kv]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(0)  # [total_q, total_kv]

        total_q_tokens, total_kv_tokens = attention_mask.shape

        # Strategy: Find Q sequence boundaries first, then find corresponding K boundaries
        # Q sequence boundary: row i where no overlap with row i-1's attention pattern
        q_boundaries = [0]

        for i in range(1, total_q_tokens):
            # Check if query i can attend to anything that query i-1 could attend to
            prev_attended = attention_mask[i - 1].nonzero(as_tuple=False).flatten()
            curr_attended = attention_mask[i].nonzero(as_tuple=False).flatten()

            if len(prev_attended) > 0 and len(curr_attended) > 0:
                # Check if there's a gap: current min > previous max means new sequence
                prev_max = prev_attended.max().item()
                curr_min = curr_attended.min().item()

                if curr_min > prev_max:
                    q_boundaries.append(i)
            elif len(curr_attended) > 0 and i > 0:
                # Current has attention but previous didn't → new sequence
                q_boundaries.append(i)

        q_boundaries.append(total_q_tokens)
        num_sequences = len(q_boundaries) - 1

        # Find K boundaries: for each Q sequence, find the range of K positions it attends to
        k_boundaries = [0]

        for seq_idx in range(num_sequences):
            q_start = q_boundaries[seq_idx]
            q_end = q_boundaries[seq_idx + 1]

            # Get all K positions attended by this Q sequence
            seq_mask = attention_mask[q_start:q_end, :]  # [q_seq_len, total_kv]
            attended_k = seq_mask.sum(dim=0).nonzero(as_tuple=False).flatten()

            if len(attended_k) > 0:
                # K sequence ends at max attended position + 1
                k_end = attended_k.max().item() + 1
                k_boundaries.append(k_end)
            else:
                # No attention (shouldn't happen), assume no K tokens
                k_boundaries.append(k_boundaries[-1])

        # Build cu_seqlens
        cu_seqlens_q = torch.tensor(q_boundaries, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(k_boundaries, dtype=torch.int32, device=device)

        # Verify same number of sequences
        assert len(cu_seqlens_q) == len(
            cu_seqlens_k
        ), f"Q and K must have same number of sequences, got {len(cu_seqlens_q)-1} vs {len(cu_seqlens_k)-1}"

        # Calculate sequence lengths
        seq_lengths_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu().tolist()
        seq_lengths_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu().tolist()

        # Maximum sequence lengths
        max_seqlen_q = max(seq_lengths_q) if seq_lengths_q else 0
        max_seqlen_k = max(seq_lengths_k) if seq_lengths_k else 0

        return {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "total_q_tokens": total_q_tokens,
            "total_kv_tokens": total_kv_tokens,
            "num_sequences": num_sequences,
            "seq_lengths_q": seq_lengths_q,
            "seq_lengths_k": seq_lengths_k,
        }

    @staticmethod
    def create_deduplicated_batch(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
        prompt_groups: List[List[int]],
        unique_prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_unpad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Create deduplicated batch by concatenating tokens into a single sequence.
        Replicated prompts are removed - each unique prompt appears once followed by all its responses.

        Args:
            input_ids: [batch_size, seq_len] - input token IDs (may contain padding)
            position_ids: [batch_size, seq_len] - position IDs
            response_length: Length of response section (including any padding)
            prompt_groups: List of lists, each containing indices of samples with same prompt
            unique_prompts: Tensor of unique prompts
            attention_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding (optional)
            use_unpad: If True, remove padding from deduplicated batch to produce packed format

        Returns:
            dedup_input_ids: [1, total_tokens] - deduplicated (packed if use_unpad=True)
            unpadded_position_ids: [batch_size, seq_len] or [1, total_valid_tokens] - original structure, or unpadded and packed if use_unpad=True
            reconstruction_info: Dict with info to reconstruct original batch
        """
        # Step 1: Deduplicate input_ids
        dedup_input_ids, dedup_attention_mask, reconstruction_info = ZoRRoTrain._dedup_input_ids(
            input_ids, response_length, prompt_groups, attention_mask
        )

        # Step 2 & 3: Optionally unpad
        if use_unpad and attention_mask is not None:
            # Unpad deduplicated input_ids
            dedup_input_ids, reconstruction_info = ZoRRoTrain._unpad_dedup_input_ids(
                dedup_input_ids, dedup_attention_mask, reconstruction_info
            )

            # Unpad original (non-deduplicated) position_ids
            unpadded_position_ids = ZoRRoTrain._unpad_replicated_ids(position_ids, attention_mask)
        else:
            # No unpadding: keep original position_ids unchanged
            unpadded_position_ids = position_ids

        return dedup_input_ids, unpadded_position_ids, reconstruction_info

    @staticmethod
    def unpad_deduplicated_batch(
        dedup_input_ids: torch.Tensor,
        dedup_position_ids: torch.Tensor,
        dedup_attention_mask: torch.Tensor,
        reconstruction_info: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Remove padding from deduplicated batch and update reconstruction_info.

        Args:
            dedup_input_ids: [1, total_tokens] - deduplicated input IDs with padding
            dedup_position_ids: [1, total_tokens] - deduplicated position IDs with padding
            dedup_attention_mask: [1, total_tokens] - attention mask (1=valid, 0=padding)
            reconstruction_info: Dict with reconstruction metadata

        Returns:
            dedup_input_ids_unpad: [1, total_valid_tokens] - without padding
            dedup_position_ids_unpad: [1, total_valid_tokens] - without padding
            reconstruction_info_updated: Dict with unpadding metadata added
        """
        # Get valid token indices (where attention_mask == 1)
        valid_mask = dedup_attention_mask.squeeze(0).bool()  # [total_tokens]
        unpad_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)  # [total_valid_tokens]

        # Extract valid tokens
        dedup_input_ids_unpad = dedup_input_ids[:, unpad_indices]  # [1, total_valid_tokens]
        dedup_position_ids_unpad = dedup_position_ids[:, unpad_indices]  # [1, total_valid_tokens]

        # Build cu_seqlens for deduplicated batch
        # We need to track where each segment (prompt/response) starts in the packed format
        segment_info = reconstruction_info["segment_info"]
        prompt_groups = reconstruction_info["prompt_groups"]

        # Update segment_info with unpadded positions
        updated_segment_info = []
        cu_seqlens_list = [0]  # Start with 0
        current_packed_pos = 0

        for seg in segment_info:
            # Get valid tokens in this segment
            seg_start, seg_end = seg["start"], seg["end"]
            seg_mask = valid_mask[seg_start:seg_end]
            num_valid = seg_mask.sum().item()

            if num_valid > 0:
                updated_segment_info.append(
                    {
                        **seg,
                        "num_valid_tokens": num_valid,
                    }
                )

                current_packed_pos += num_valid
            else:
                # Segment has no valid tokens (all padding) - keep info but mark as empty
                updated_segment_info.append(
                    {
                        **seg,
                        "num_valid_tokens": 0,
                    }
                )

        # Build cu_seqlens for packed format
        # Each original sample becomes one sequence in the packed format
        batch_size = reconstruction_info["original_batch_size"]
        for sample_idx in range(batch_size):
            # Sum valid tokens for this sample (prompt + response)
            sample_segments = [s for s in updated_segment_info if s["original_idx"] == sample_idx]
            sample_valid_tokens = sum(s["num_valid_tokens"] for s in sample_segments)
            cu_seqlens_list.append(cu_seqlens_list[-1] + sample_valid_tokens)

        cu_seqlens_packed = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=dedup_input_ids.device)
        max_seqlen_packed = cu_seqlens_packed.diff().max().item()

        # Also build cu_seqlens for deduplicated format (by prompt groups + responses)
        cu_seqlens_dedup_list = [0]
        for group_idx, group in enumerate(prompt_groups):
            # Prompt segment
            prompt_seg = [s for s in updated_segment_info if s["group_idx"] == group_idx and s["type"] == "prompt"][0]
            prompt_valid = prompt_seg["num_valid_tokens"]

            # Each response in the group
            for sample_idx in group:
                response_seg = [
                    s
                    for s in updated_segment_info
                    if s["group_idx"] == group_idx and s["type"] == "response" and s["original_idx"] == sample_idx
                ][0]
                response_valid = response_seg["num_valid_tokens"]

                # One sequence = prompt + response
                seq_len = prompt_valid + response_valid
                cu_seqlens_dedup_list.append(cu_seqlens_dedup_list[-1] + seq_len)

        cu_seqlens_dedup = torch.tensor(cu_seqlens_dedup_list, dtype=torch.int32, device=dedup_input_ids.device)
        max_seqlen_dedup = cu_seqlens_dedup.diff().max().item() if len(cu_seqlens_dedup) > 1 else 0

        # Update reconstruction_info
        reconstruction_info.update(
            {
                "is_unpadded": True,
                "segment_info": updated_segment_info,
                "cu_seqlens_packed": cu_seqlens_packed,
                "max_seqlen_packed": max_seqlen_packed,
                "cu_seqlens_dedup": cu_seqlens_dedup,
                "max_seqlen_dedup": max_seqlen_dedup,
            }
        )

        return dedup_input_ids_unpad, dedup_position_ids_unpad, reconstruction_info

    @staticmethod
    def reconstruct_sequences(dedup_hidden: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Reconstruct full batch from deduplicated sequence.

        Args:
            dedup_hidden: [1, total_tokens, *extra_dims] - deduplicated hidden states
                         Can have arbitrary dimensions after the first two
                         E.g., [1, total_tokens, hidden_dim] or [1, total_tokens, num_heads, head_dim]

        Returns:
            If is_unpadded=True:
                full_hidden: [1, total_valid_tokens, *extra_dims] - packed format (no padding)
            Else:
                full_hidden: [original_batch, seq_len, *extra_dims] - padded format
        """
        is_unpadded = reconstruction_info.get("is_unpadded", False)

        if is_unpadded:
            return ZoRRoTrain._reconstruct_sequences_packed(dedup_hidden, reconstruction_info)
        else:
            return ZoRRoTrain._reconstruct_sequences_padded(dedup_hidden, reconstruction_info)

    @staticmethod
    def _reconstruct_sequences_padded(dedup_hidden: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Reconstruct to padded format [original_batch, seq_len, *extra_dims].
        """
        segment_info = reconstruction_info["segment_info"]
        original_batch_size = reconstruction_info["original_batch_size"]
        original_seq_len = reconstruction_info["original_seq_len"]
        prompt_len = reconstruction_info["prompt_len"]

        # Remove batch dimension
        dedup_hidden = dedup_hidden.squeeze(0)  # [total_tokens, *extra_dims]
        extra_dims = dedup_hidden.shape[1:]  # All dimensions after token dimension
        device = dedup_hidden.device
        dtype = dedup_hidden.dtype

        # Build mapping from original sample idx to segments
        sample_segments = {}
        for seg in segment_info:
            sample_idx = seg["original_idx"]
            if sample_idx not in sample_segments:
                sample_segments[sample_idx] = {"prompt": None, "response": None}

            if seg["type"] == "prompt":
                sample_segments[sample_idx]["prompt"] = seg
            else:  # response
                sample_segments[sample_idx]["response"] = seg

        # Reconstruct each sample
        full_sequences = []
        for sample_idx in range(original_batch_size):
            segs = sample_segments[sample_idx]

            # Create sequence with proper shape
            seq = torch.zeros((original_seq_len,) + extra_dims, dtype=dtype, device=device)

            # Place prompt
            prompt_seg = segs["prompt"]
            prompt_start = prompt_seg["start"]
            prompt_end = prompt_seg["end"]
            prompt_hidden = dedup_hidden[prompt_start:prompt_end]
            seq[:prompt_len] = prompt_hidden

            # Place response
            response_seg = segs["response"]
            response_start = response_seg["start"]
            response_end = response_seg["end"]
            response_hidden = dedup_hidden[response_start:response_end]
            seq[prompt_len:] = response_hidden

            full_sequences.append(seq)

        return torch.stack(full_sequences)  # [original_batch, seq_len, *extra_dims]

    @staticmethod
    def _get_responses_packed_from_dedup_tensor(
        dedup_tensor: torch.Tensor,
        reconstruction_info: Dict,
        cu_seqlens_response: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct to packed format [1, total_valid_tokens, *extra_dims] (no padding).
        Replicates deduplicated prompts for each sample.

        In the deduplicated space, prompts appear once per group followed by all responses.
        We need to compute deduplicated positions since segment_info contains replicated positions.
        """

        prompt_groups = reconstruction_info["prompt_groups"]
        # prompt_len = reconstruction_info["prompt_len"]
        cu_seqlens_packed = reconstruction_info["cu_seqlens_unique_prompts"]
        reordered_seq_idx = reconstruction_info["reordered_seq_idx"]

        # Remove batch dimension: [1, num_heads, total_valid_tokens, head_dim] -> [num_heads, total_valid_tokens, head_dim]

        dedup_tensor = dedup_tensor.squeeze(0)  # [total_dedup_valid_tokens, *extra_dims]
        # num_heads = dedup_tensor.shape[0]
        # head_dim = dedup_tensor.shape[2]

        # Collect all unique prompt tensors (without padding)
        response_tensors = []
        prev_packed_response_lengths = 0
        for gid, group in enumerate(prompt_groups):
            # Get first sequence from this group
            first_secondary_idx = reordered_seq_idx[group[-1]] + 1  # Last sample's response

            response_offset = cu_seqlens_packed[gid + 1].item()
            response_len = cu_seqlens_response[first_secondary_idx].item()

            response_qkv = dedup_tensor[
                :, prev_packed_response_lengths + response_offset : response_offset + response_len, :
            ]

            prev_packed_response_lengths = response_len

            response_tensors.append(response_qkv)
        # import pdb; pdb.set_trace()
        # Concatenate all unique prompts along token dimension (packed format)
        # [num_heads, total_unique_prompt_tokens, head_dim]
        qkv_responses_packed = torch.cat(response_tensors, dim=1)

        # Add batch dimension: [1, num_heads, total_unique_prompt_tokens, head_dim]
        qkv_responses_packed = qkv_responses_packed.unsqueeze(0)

        # Return packed tensor: [1, num_heads, total_unique_prompt_tokens, head_dim]
        reconstruction_info["cu_seqlens_response"] = cu_seqlens_response
        reconstruction_info["max_response_valid_len"] = max(cu_seqlens_response[1:] - cu_seqlens_response[:-1])

        return qkv_responses_packed

    @staticmethod
    def _get_sequences_reconstructed_from_dedup_tensors(
        dedup_tensor: torch.Tensor,
        prompt_packed_tensor: torch.Tensor,
        reconstruction_info: Dict,
        cu_seqlens_response: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct to packed format [1, total_valid_tokens, *extra_dims] (no padding).
        Replicates deduplicated prompts for each sample.

        In the deduplicated space, prompts appear once per group followed by all responses.
        We need to compute deduplicated positions since segment_info contains replicated positions.
        """

        prompt_groups = reconstruction_info["prompt_groups"]
        # prompt_len = reconstruction_info["prompt_len"]
        cu_seqlens_packed = reconstruction_info["cu_seqlens_unique_prompts"]
        reordered_seq_idx = reconstruction_info["reordered_seq_idx"]

        # Remove batch dimension: [1, num_heads, total_valid_tokens, head_dim] -> [num_heads, total_valid_tokens, head_dim]

        dedup_tensor = dedup_tensor.squeeze(0)  # [total_dedup_valid_tokens, *extra_dims]

        # Collect all unique prompt tensors (without padding)
        response_tensors = []
        prev_packed_response_lengths = 0
        all_tensors = []

        prompt_lengths = reconstruction_info["prompt_lengths"][1:]
        response_lengths = (cu_seqlens_response[1:] - cu_seqlens_response[:-1]).tolist()
        prompt_tensors = torch.split(prompt_packed_tensor.squeeze(0), prompt_lengths, dim=1)
        for gid, group in enumerate(prompt_groups):
            # Get first sequence from this group
            first_secondary_idx = reordered_seq_idx[group[-1]] + 1  # Last sample's response

            response_offset = cu_seqlens_packed[gid + 1].item()
            response_len = cu_seqlens_response[first_secondary_idx].item()

            response_qkv = dedup_tensor[
                :, prev_packed_response_lengths + response_offset : response_offset + response_len, :
            ]

            prev_packed_response_lengths = response_len
            response_tensors = torch.split(
                response_qkv, response_lengths[group_sizes[gid] : group_sizes[gid + 1]], dim=1
            )
            all_tensors.extend(
                [torch.cat([prompt_tensors[gid], response_tensor], dim=1) for response_tensor in response_tensors]
            )

        # Add batch dimension: [1, num_heads, total_unique_prompt_tokens, head_dim]
        reconstructed_seq = torch.cat(all_tensors, dim=1).unsqueeze(0)

        return reconstructed_seq

    @staticmethod
    def _get_sequences_packed_from_dedup_tensor(
        dedup_tensor: torch.Tensor,
        reconstruction_info: Dict,
        cu_seqlens_response: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct to packed format [1, total_valid_tokens, *extra_dims] (no padding).
        Replicates deduplicated prompts for each sample.

        In the deduplicated space, prompts appear once per group followed by all responses.
        We need to compute deduplicated positions since segment_info contains replicated positions.
        """

        prompt_groups = reconstruction_info["prompt_groups"]
        prompt_len = reconstruction_info["prompt_len"]
        # cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]
        # original_attention_mask = reconstruction_info["original_attention_mask"]

        # Remove batch dimension: [1, num_heads, total_valid_tokens, head_dim] -> [num_heads, total_valid_tokens, head_dim]

        dedup_tensor = dedup_tensor.squeeze(0)  # [total_dedup_valid_tokens, *extra_dims]
        num_heads = dedup_tensor.shape[0]
        head_dim = dedup_tensor.shape[2]

        # Recover prompt lengths from cu_seqlens
        cu_seqlens_unique_prompts = reconstruction_info["cu_seqlens_unique_prompts"]
        prompt_lengths = reconstruction_info["prompt_lengths"]

        packed_prmpt_lengths = [0]

        prompt_tensors = torch.empty(
            (num_heads, cu_seqlens_unique_prompts[-1], head_dim), dtype=dedup_tensor.dtype, device=dedup_tensor.device
        )
        for id, group in enumerate(prompt_groups):
            # Get first sequence from this group
            first_seq_idx = group[0]
            # first_secondary_idx = first_seq_idx + 1

            response_offset = cu_seqlens_response[first_seq_idx].item()
            # response_len = (cu_seqlens_response[first_secondary_idx] - response_offset).item()

            prompt_len = prompt_lengths[id + 1]

            seq_offset = packed_prmpt_lengths[-1] + response_offset
            prompt_qkv = dedup_tensor[
                :, seq_offset : seq_offset + prompt_len, :
            ]  # [seq_valid_tokens, num_heads, head_dim]
            prompt_tensors[:, cu_seqlens_unique_prompts[id] : cu_seqlens_unique_prompts[id + 1], :] = prompt_qkv

            packed_prmpt_lengths.append(packed_prmpt_lengths[-1] + prompt_len)

        # Concatenate all unique prompts along token dimension (packed format)
        # [num_heads, total_unique_prompt_tokens, head_dim]
        # unique_qkv_packed = prompt_tensors.transpose(0, 1)

        # Add batch dimension: [1, num_heads, total_unique_prompt_tokens, head_dim]
        unique_qkv_packed = prompt_tensors.unsqueeze(0)

        # Return packed tensor: [1, num_heads, total_unique_prompt_tokens, head_dim]
        return unique_qkv_packed

    @staticmethod
    def _get_sequences_unpacked_from_dedup_tensor(
        dedup_tensor: torch.Tensor,
        reconstruction_info: Dict,
        cu_seqlens_response: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct to packed format [1, total_valid_tokens, *extra_dims] (no padding).
        Replicates deduplicated prompts for each sample.

        In the deduplicated space, prompts appear once per group followed by all responses.
        We need to compute deduplicated positions since segment_info contains replicated positions.
        """

        prompt_groups = reconstruction_info["prompt_groups"]
        prompt_len = reconstruction_info["prompt_len"]
        cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]
        # original_attention_mask = reconstruction_info["original_attention_mask"]

        # Remove batch dimension: [1, num_heads, total_valid_tokens, head_dim] -> [num_heads, total_valid_tokens, head_dim]

        dedup_tensor = dedup_tensor.squeeze(0)  # [total_dedup_valid_tokens, *extra_dims]
        num_heads = dedup_tensor.shape[0]
        head_dim = dedup_tensor.shape[2]

        # Collect all unique prompt tensors (without padding)
        seq_lengths = [0]
        prompt_lengths = []
        for group in prompt_groups:
            # Extract this sequence using cu_seqlens

            first_seq_idx = group[0]
            first_secondary_idx = first_seq_idx + 1

            response_offset = cu_seqlens_response[first_seq_idx].item()
            response_len = (cu_seqlens_response[first_secondary_idx] - response_offset).item()

            response_len_total = (cu_seqlens_response[group[-1] + 1] - response_offset).item()

            seq_start = cu_seqlens_packed[first_seq_idx].item()
            seq_end = cu_seqlens_packed[first_seq_idx + 1].item()
            prompt_response_len = seq_end - seq_start
            prompt_len = prompt_response_len - response_len
            prompt_lengths.append(prompt_len)
            seq_lengths.append(prompt_len * len(group) + response_len_total)

        # Build cu_seqlens for unique prompts (cumulative token positions)
        cu_seqlens_unique_prompts = torch.tensor(seq_lengths, dtype=torch.int32, device=dedup_tensor.device).cumsum_(
            dim=0
        )

        packed_prmpt_lengths = [0]

        complete_tensor = torch.empty(
            (num_heads, cu_seqlens_unique_prompts[-1], head_dim), dtype=dedup_tensor.dtype, device=dedup_tensor.device
        )
        for id, group in enumerate(prompt_groups):
            # Get first sequence from this group

            first_seq_idx = group[0]
            first_secondary_idx = first_seq_idx + 1

            response_offset = cu_seqlens_response[first_seq_idx].item()
            response_len = (cu_seqlens_response[first_secondary_idx] - response_offset).item()
            response_len_total = (cu_seqlens_response[group[-1] + 1] - response_offset).item()

            prompt_len = prompt_lengths[id]

            seq_offset = packed_prmpt_lengths[-1] + response_offset
            prompt_qkv = dedup_tensor[
                :, seq_offset : seq_offset + prompt_len, :
            ]  # [seq_valid_tokens, num_heads, head_dim]
            responses_list = torch.split(
                dedup_tensor[:, seq_offset + prompt_len : seq_offset + prompt_len + response_len_total, :],
                (
                    cu_seqlens_response[first_secondary_idx : group[-1] + 2]
                    - cu_seqlens_response[first_seq_idx : group[-1] + 1]
                ).tolist(),
                dim=1,
            )
            all_prompt_response_pairs = [0] * len(group) * 2

            all_prompt_response_pairs[0::2] = [prompt_qkv] * len(group)
            all_prompt_response_pairs[1::2] = responses_list
            complete_tensor[:, cu_seqlens_unique_prompts[id] : cu_seqlens_unique_prompts[id + 1], :] = torch.cat(
                all_prompt_response_pairs, dim=1
            )

        # Add batch dimension: [1, num_heads, total_unique_prompt_tokens, head_dim]
        unique_qkv_packed = complete_tensor.unsqueeze(0)

        # Return packed tensor: [1, num_heads, total_unique_prompt_tokens, head_dim]
        return unique_qkv_packed

    @staticmethod
    def _reconstruct_sequences_packed(dedup_hidden: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Reconstruct to packed format [1, total_valid_tokens, *extra_dims] (no padding).
        Replicates deduplicated prompts for each sample.

        In the deduplicated space, prompts appear once per group followed by all responses.
        We need to compute deduplicated positions since segment_info contains replicated positions.
        """
        segment_info = reconstruction_info["segment_info"]
        original_batch_size = reconstruction_info["original_batch_size"]
        prompt_groups = reconstruction_info["prompt_groups"]

        # Remove batch dimension
        dedup_hidden = dedup_hidden.squeeze(0)  # [total_dedup_valid_tokens, *extra_dims]

        # Build mapping: sample_idx -> group_idx
        sample_to_group = {}
        for group_idx, group in enumerate(prompt_groups):
            for sample_idx in group:
                sample_to_group[sample_idx] = group_idx

        # Organize segments by group and type
        # segments_by_group[group_idx] = {'prompt': seg, 'responses': [seg1, seg2, ...]}
        segments_by_group = {}
        for seg in segment_info:
            group_idx = seg["group_idx"]
            if group_idx not in segments_by_group:
                segments_by_group[group_idx] = {"prompt": None, "responses": {}}

            if seg["type"] == "prompt":
                # Only store first occurrence (they're all identical in unpadded coords)
                if segments_by_group[group_idx]["prompt"] is None:
                    segments_by_group[group_idx]["prompt"] = seg
            else:  # response
                sample_idx = seg["original_idx"]
                segments_by_group[group_idx]["responses"][sample_idx] = seg

        # Compute deduplicated positions for each group
        # Layout: [group0_prompt, group0_responses..., group1_prompt, group1_responses..., ...]
        current_dedup_pos = 0
        group_dedup_positions = {}

        for group_idx in sorted(segments_by_group.keys()):
            group_data = segments_by_group[group_idx]
            prompt_seg = group_data["prompt"]

            # Prompt position in deduplicated space
            prompt_len = prompt_seg["num_valid_tokens"]
            prompt_dedup_start = current_dedup_pos
            prompt_dedup_end = current_dedup_pos + prompt_len
            current_dedup_pos += prompt_len

            # Response positions in deduplicated space
            response_dedup_positions = {}
            for sample_idx in sorted(group_data["responses"].keys()):
                response_seg = group_data["responses"][sample_idx]
                response_len = response_seg["num_valid_tokens"]
                response_dedup_start = current_dedup_pos
                response_dedup_end = current_dedup_pos + response_len
                current_dedup_pos += response_len
                response_dedup_positions[sample_idx] = (response_dedup_start, response_dedup_end)

            group_dedup_positions[group_idx] = {
                "prompt": (prompt_dedup_start, prompt_dedup_end),
                "responses": response_dedup_positions,
            }

        # Reconstruct each sample in packed format
        packed_sequences = []
        for sample_idx in range(original_batch_size):
            group_idx = sample_to_group[sample_idx]
            group_pos = group_dedup_positions[group_idx]

            # Extract prompt from deduplicated space
            prompt_start, prompt_end = group_pos["prompt"]
            prompt_hidden = dedup_hidden[prompt_start:prompt_end]

            # Extract response from deduplicated space
            response_start, response_end = group_pos["responses"][sample_idx]
            response_hidden = dedup_hidden[response_start:response_end]

            # Concatenate prompt + response for this sample
            sample_hidden = torch.cat([prompt_hidden, response_hidden], dim=0)
            packed_sequences.append(sample_hidden)

        # Concatenate all samples into packed format
        packed_hidden = torch.cat(packed_sequences, dim=0)

        return packed_hidden.unsqueeze(0)  # [1, total_all_valid_tokens, *extra_dims]

    @staticmethod
    def deduplicate_sequences(full_hidden: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Convert full batch to deduplicated sequence.

        Args:
            full_hidden: Input depends on is_unpadded flag:
                - If is_unpadded=False: [original_batch, seq_len, *extra_dims] - padded format
                - If is_unpadded=True: [1, total_valid_tokens, *extra_dims] - packed format

        Returns:
            dedup_hidden: [1, total_dedup_tokens, *extra_dims]
        """
        is_unpadded = reconstruction_info.get("is_unpadded", False)

        if is_unpadded:
            return ZoRRoTrain._deduplicate_sequences_packed(full_hidden, reconstruction_info)
        else:
            return ZoRRoTrain._deduplicate_sequences_padded(full_hidden, reconstruction_info)

    @staticmethod
    def _deduplicate_sequences_padded(full_hidden: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Deduplicate from padded format [original_batch, seq_len, *extra_dims].
        """
        segment_info = reconstruction_info["segment_info"]
        prompt_groups = reconstruction_info["prompt_groups"]
        prompt_len = reconstruction_info["prompt_len"]
        response_length = reconstruction_info["response_length"]

        # Calculate total tokens
        total_tokens = len(prompt_groups) * prompt_len + full_hidden.size(0) * response_length

        # Get extra dimensions (everything after batch and seq_len)
        extra_dims = full_hidden.shape[2:]

        # Create tensor for concatenated sequence
        dedup_hidden = torch.zeros((total_tokens,) + extra_dims, dtype=full_hidden.dtype, device=full_hidden.device)

        # Fill in the concatenated sequence
        for group_idx, group in enumerate(prompt_groups):
            # Find prompt segment for this group
            prompt_seg = None
            for seg in segment_info:
                if seg["group_idx"] == group_idx and seg["type"] == "prompt":
                    prompt_seg = seg
                    break

            # Extract prompt from first sample (they should all be identical)
            first_sample = group[0]
            prompt_hidden = full_hidden[first_sample, :prompt_len]

            start = prompt_seg["start"]
            end = prompt_seg["end"]
            dedup_hidden[start:end] = prompt_hidden

            # Extract each response
            for sample_idx in group:
                # Find response segment for this sample
                response_seg = None
                for seg in segment_info:
                    if seg["original_idx"] == sample_idx and seg["type"] == "response":
                        response_seg = seg
                        break

                response_hidden = full_hidden[sample_idx, prompt_len:]

                start = response_seg["start"]
                end = response_seg["end"]
                dedup_hidden[start:end] = response_hidden

        return dedup_hidden.unsqueeze(0)  # [1, total_tokens, *extra_dims]

    @staticmethod
    def _deduplicate_sequences_packed(full_hidden: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Deduplicate from packed format [1, total_valid_tokens, *extra_dims].
        """
        segment_info = reconstruction_info["segment_info"]
        prompt_groups = reconstruction_info["prompt_groups"]
        original_attention_mask = reconstruction_info["original_attention_mask"]
        prompt_len = reconstruction_info["prompt_len"]
        cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]

        # Remove batch dimension
        full_hidden = full_hidden.squeeze(0)  # [total_valid_tokens, *extra_dims]
        extra_dims = full_hidden.shape[1:]
        device = full_hidden.device
        dtype = full_hidden.dtype

        # Calculate total deduplicated tokens (unpadded)
        # Only count unique prompts (one per group) + all responses
        total_dedup_tokens = 0
        counted_prompts = set()
        for seg in segment_info:
            if seg["type"] == "prompt":
                group_idx = seg["group_idx"]
                if group_idx not in counted_prompts:
                    total_dedup_tokens += seg["num_valid_tokens"]
                    counted_prompts.add(group_idx)
            else:  # response
                total_dedup_tokens += seg["num_valid_tokens"]

        # pr0(f"[_deduplicate_sequences_packed] Input: {full_hidden.shape[0]} tokens")
        # pr0(f"[_deduplicate_sequences_packed] Expected output: {total_dedup_tokens} tokens")
        # pr0(f"[_deduplicate_sequences_packed] Num groups: {len(prompt_groups)}, Batch size: {original_batch_size}")

        # Create tensor for deduplicated sequence
        dedup_hidden = torch.zeros((total_dedup_tokens,) + extra_dims, dtype=dtype, device=device)

        # Organize segments by group and type
        segments_by_group = {}
        for seg in segment_info:
            group_idx = seg["group_idx"]
            if group_idx not in segments_by_group:
                segments_by_group[group_idx] = {"prompt": None, "responses": {}}

            if seg["type"] == "prompt":
                if segments_by_group[group_idx]["prompt"] is None:
                    segments_by_group[group_idx]["prompt"] = seg
            else:  # response
                sample_idx = seg["original_idx"]
                segments_by_group[group_idx]["responses"][sample_idx] = seg

        # Compute deduplicated positions for each group
        current_dedup_pos = 0
        group_dedup_positions = {}

        for group_idx in sorted(segments_by_group.keys()):
            group_data = segments_by_group[group_idx]
            prompt_seg = group_data["prompt"]

            # Prompt position in deduplicated space
            prompt_len_valid = prompt_seg["num_valid_tokens"]
            prompt_dedup_start = current_dedup_pos
            prompt_dedup_end = current_dedup_pos + prompt_len_valid
            current_dedup_pos += prompt_len_valid

            # Response positions in deduplicated space
            response_dedup_positions = {}
            for sample_idx in sorted(group_data["responses"].keys()):
                response_seg = group_data["responses"][sample_idx]
                response_len_valid = response_seg["num_valid_tokens"]
                response_dedup_start = current_dedup_pos
                response_dedup_end = current_dedup_pos + response_len_valid
                current_dedup_pos += response_len_valid
                response_dedup_positions[sample_idx] = (response_dedup_start, response_dedup_end)

            group_dedup_positions[group_idx] = {
                "prompt": (prompt_dedup_start, prompt_dedup_end),
                "responses": response_dedup_positions,
            }

        # Extract tokens from packed format and place into deduplicated tensor
        for group_idx, group in enumerate(prompt_groups):
            group_pos = group_dedup_positions[group_idx]

            # Extract prompt from first sample in the group
            first_sample = group[0]
            sample_start = cu_seqlens_packed[first_sample].item()
            sample_end = cu_seqlens_packed[first_sample + 1].item()
            sample_tokens = full_hidden[sample_start:sample_end]

            # Get number of valid prompt tokens
            sample_mask = original_attention_mask[first_sample]
            prompt_mask = sample_mask[:prompt_len]
            num_valid_prompt = prompt_mask.sum().item()

            # Extract prompt and place in deduplicated tensor
            prompt_hidden = sample_tokens[:num_valid_prompt]
            prompt_dedup_start, prompt_dedup_end = group_pos["prompt"]
            dedup_hidden[prompt_dedup_start:prompt_dedup_end] = prompt_hidden

            # Extract each response
            for sample_idx in group:
                # Extract tokens for this sample from packed format
                sample_start = cu_seqlens_packed[sample_idx].item()
                sample_end = cu_seqlens_packed[sample_idx + 1].item()
                sample_tokens = full_hidden[sample_start:sample_end]

                # Get number of valid prompt tokens to skip
                sample_mask = original_attention_mask[sample_idx]
                prompt_mask = sample_mask[:prompt_len]
                num_valid_prompt = prompt_mask.sum().item()

                # Extract response tokens (after prompt)
                response_hidden = sample_tokens[num_valid_prompt:]
                response_dedup_start, response_dedup_end = group_pos["responses"][sample_idx]
                dedup_hidden[response_dedup_start:response_dedup_end] = response_hidden

        return dedup_hidden.unsqueeze(0)  # [1, total_dedup_tokens, *extra_dims]

    @staticmethod
    def reconstruct_position_ids(dedup_pos: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """Reconstruct full position_ids. NO PADDING."""
        if dedup_pos is None:
            return None

        segment_info = reconstruction_info["segment_info"]
        original_batch_size = reconstruction_info["original_batch_size"]
        original_seq_len = reconstruction_info["original_seq_len"]
        prompt_len = reconstruction_info["prompt_len"]

        # Remove batch dimension
        dedup_pos = dedup_pos.squeeze(0)  # [total_tokens]
        device = dedup_pos.device
        dtype = dedup_pos.dtype

        # Build mapping
        sample_segments = {}
        for seg in segment_info:
            sample_idx = seg["original_idx"]
            if sample_idx not in sample_segments:
                sample_segments[sample_idx] = {"prompt": None, "response": None}

            if seg["type"] == "prompt":
                sample_segments[sample_idx]["prompt"] = seg
            else:
                sample_segments[sample_idx]["response"] = seg

        # Reconstruct
        full_positions = []
        for sample_idx in range(original_batch_size):
            segs = sample_segments[sample_idx]

            pos = torch.zeros(original_seq_len, dtype=dtype, device=device)

            # Place prompt positions
            prompt_seg = segs["prompt"]
            prompt_pos = dedup_pos[prompt_seg["start"] : prompt_seg["end"]]
            pos[:prompt_len] = prompt_pos

            # Place response positions
            response_seg = segs["response"]
            response_pos = dedup_pos[response_seg["start"] : response_seg["end"]]
            pos[prompt_len:] = response_pos

            full_positions.append(pos)

        return torch.stack(full_positions)  # [original_batch, seq_len]

    @staticmethod
    def deduplicate_position_ids(position_ids_full: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Create deduplicated position_ids from full position_ids.

        Args:
            position_ids_full: [batch_size, seq_len]
            reconstruction_info: Metadata for deduplication

        Returns:
            position_ids_dedup: [1, total_tokens]
        """
        segment_info = reconstruction_info["segment_info"]
        prompt_groups = reconstruction_info["prompt_groups"]
        prompt_len = reconstruction_info["prompt_len"]
        response_length = reconstruction_info["response_length"]

        # Calculate total tokens
        total_tokens = len(prompt_groups) * prompt_len + position_ids_full.size(0) * response_length

        device = position_ids_full.device
        dtype = position_ids_full.dtype

        # Create deduplicated position_ids
        position_ids_dedup = torch.zeros(total_tokens, dtype=dtype, device=device)

        # Fill in deduplicated position_ids
        for group_idx, group in enumerate(prompt_groups):
            # Find prompt segment
            prompt_seg = None
            for seg in segment_info:
                if seg["group_idx"] == group_idx and seg["type"] == "prompt":
                    prompt_seg = seg
                    break

            # Get position_ids for prompt (from first sample in group)
            first_sample = group[0]
            prompt_pos = position_ids_full[first_sample, :prompt_len]

            start = prompt_seg["start"]
            end = prompt_seg["end"]
            position_ids_dedup[start:end] = prompt_pos

            # Fill in response position_ids for each sample
            for sample_idx in group:
                # Find response segment
                response_seg = None
                for seg in segment_info:
                    if seg["original_idx"] == sample_idx and seg["type"] == "response":
                        response_seg = seg
                        break

                response_pos = position_ids_full[sample_idx, prompt_len:]

                start = response_seg["start"]
                end = response_seg["end"]
                position_ids_dedup[start:end] = response_pos

        return position_ids_dedup.unsqueeze(0)  # [1, total_tokens]

    @staticmethod
    def reconstruct_position_embeddings(dedup_emb: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Reconstruct full position_embeddings (cos/sin). NO PADDING.

        Args:
            dedup_emb: [1, total_tokens, head_dim] or [1, 1, total_tokens, head_dim]

        Returns:
            full_emb: [original_batch, seq_len, head_dim] or [original_batch, 1, seq_len, head_dim]
        """
        if dedup_emb is None:
            return None

        segment_info = reconstruction_info["segment_info"]
        original_batch_size = reconstruction_info["original_batch_size"]
        original_seq_len = reconstruction_info["original_seq_len"]
        prompt_len = reconstruction_info["prompt_len"]

        # Handle both 3D and 4D formats
        has_head_dim_for_broadcast = dedup_emb.dim() == 4

        if has_head_dim_for_broadcast:
            dedup_emb = dedup_emb.squeeze(1)  # [1, total_tokens, head_dim]

        # Remove batch dimension
        dedup_emb = dedup_emb.squeeze(0)  # [total_tokens, head_dim]
        head_dim = dedup_emb.size(-1)
        device = dedup_emb.device
        dtype = dedup_emb.dtype

        # Build mapping
        sample_segments = {}
        for seg in segment_info:
            sample_idx = seg["original_idx"]
            if sample_idx not in sample_segments:
                sample_segments[sample_idx] = {"prompt": None, "response": None}

            if seg["type"] == "prompt":
                sample_segments[sample_idx]["prompt"] = seg
            else:
                sample_segments[sample_idx]["response"] = seg

        # Reconstruct
        full_embeddings = []
        for sample_idx in range(original_batch_size):
            segs = sample_segments[sample_idx]

            emb = torch.zeros((original_seq_len, head_dim), dtype=dtype, device=device)

            # Place prompt embeddings
            prompt_seg = segs["prompt"]
            prompt_emb = dedup_emb[prompt_seg["start"] : prompt_seg["end"]]
            emb[:prompt_len] = prompt_emb

            # Place response embeddings
            response_seg = segs["response"]
            response_emb = dedup_emb[response_seg["start"] : response_seg["end"]]
            emb[prompt_len:] = response_emb

            full_embeddings.append(emb)

        result = torch.stack(full_embeddings)  # [original_batch, seq_len, head_dim]

        # Restore original dimensionality if needed
        if has_head_dim_for_broadcast:
            result = result.unsqueeze(1)  # [original_batch, 1, seq_len, head_dim]

        return result

    @staticmethod
    def reconstruct_input_ids(dedup_input_ids: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Reconstruct full batch input_ids from concatenated sequence.

        Args:
            dedup_input_ids: [1, total_tokens] - concatenated

        Returns:
            full_input_ids: [original_batch, seq_len]
        """
        segment_info = reconstruction_info["segment_info"]
        original_batch_size = reconstruction_info["original_batch_size"]
        original_seq_len = reconstruction_info["original_seq_len"]
        prompt_len = reconstruction_info["prompt_len"]

        # Remove batch dimension
        dedup_input_ids = dedup_input_ids.squeeze(0)  # [total_tokens]
        device = dedup_input_ids.device
        dtype = dedup_input_ids.dtype

        # Build mapping
        sample_segments = {}
        for seg in segment_info:
            sample_idx = seg["original_idx"]
            if sample_idx not in sample_segments:
                sample_segments[sample_idx] = {"prompt": None, "response": None}

            if seg["type"] == "prompt":
                sample_segments[sample_idx]["prompt"] = seg
            else:
                sample_segments[sample_idx]["response"] = seg

        # Reconstruct
        full_sequences = []
        for sample_idx in range(original_batch_size):
            segs = sample_segments[sample_idx]

            seq = torch.zeros(original_seq_len, dtype=dtype, device=device)

            # Place prompt
            prompt_seg = segs["prompt"]
            prompt_ids = dedup_input_ids[prompt_seg["start"] : prompt_seg["end"]]
            seq[:prompt_len] = prompt_ids

            # Place response
            response_seg = segs["response"]
            response_ids = dedup_input_ids[response_seg["start"] : response_seg["end"]]
            seq[prompt_len:] = response_ids

            full_sequences.append(seq)

        return torch.stack(full_sequences)  # [original_batch, seq_len]

    @staticmethod
    def get_reconstructed_shape(dedup_shape: tuple, reconstruction_info: Dict) -> tuple:
        """
        Get the reconstructed shape from deduplicated shape.

        Args:
            dedup_shape: Shape of deduplicated tensor (1, dedup_tokens, *extra_dims)
            reconstruction_info: Metadata for reconstruction

        Returns:
            reconstructed_shape: (batch_size, seq_len, *extra_dims)

        Example:
            dedup_shape = (1, 450)  -> returns (8, 64)
            dedup_shape = (1, 450, 4096) -> returns (8, 64, 4096)
            dedup_shape = (1, 450, 32, 128) -> returns (8, 64, 32, 128)
        """
        if len(dedup_shape) < 2:
            raise ValueError(f"dedup_shape must have at least 2 dimensions, got {dedup_shape}")

        # First dimension should be 1 (deduplicated batch)
        if dedup_shape[0] != 1:
            raise ValueError(f"First dimension of dedup_shape should be 1, got {dedup_shape[0]}")

        batch_size = reconstruction_info["original_batch_size"]
        seq_len = reconstruction_info["original_seq_len"]

        # Extra dimensions beyond [1, dedup_tokens]
        extra_dims = dedup_shape[2:]

        return (batch_size, seq_len) + extra_dims

    @staticmethod
    def extract_and_deduplicate_prompts(qkv_states: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Extract unique prompts from full batch Q/K/V states.

        Args:
            qkv_states: Depends on is_unpadded flag:
                - If is_unpadded=False: [batch_size, num_heads, seq_len, head_dim] (padded)
                - If is_unpadded=True: [1, num_heads, total_valid_tokens, head_dim] (packed)
            reconstruction_info: Metadata with prompt groups

        Returns:
            unique_qkv: [num_unique_prompts, num_heads, prompt_len, head_dim]
        """
        is_unpadded = reconstruction_info.get("is_unpadded", False)

        if is_unpadded:
            return ZoRRoTrain._extract_and_deduplicate_prompts_packed(qkv_states, reconstruction_info)
        else:
            return ZoRRoTrain._extract_and_deduplicate_prompts_padded(qkv_states, reconstruction_info)

    @staticmethod
    def _extract_and_deduplicate_prompts_padded(qkv_states: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Extract unique prompts from padded sequences.

        Args:
            qkv_states: [batch_size, num_heads, seq_len, head_dim]
            reconstruction_info: Metadata with prompt groups

        Returns:
            unique_qkv: [num_unique_prompts, num_heads, prompt_len, head_dim]
        """
        prompt_groups = reconstruction_info["prompt_groups"]
        prompt_len = reconstruction_info["prompt_len"]

        unique_qkv = []
        for group in prompt_groups:
            # Get first sequence from this group (group is a list of indices)
            first_seq_idx = group[0]
            # Extract prompt part: [1, num_heads, prompt_len, head_dim]
            prompt_qkv = qkv_states[first_seq_idx : first_seq_idx + 1, :, :prompt_len, :]
            unique_qkv.append(prompt_qkv)

        # Stack: [num_unique_prompts, num_heads, prompt_len, head_dim]
        return torch.cat(unique_qkv, dim=0)

    @staticmethod
    def _extract_and_deduplicate_prompts_packed(qkv_states: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Extract unique prompts from packed sequences (unpadded).

        Args:
            qkv_states: [1, num_heads, total_valid_tokens, head_dim] - packed format
            reconstruction_info: Metadata with prompt groups and cu_seqlens

        Returns:
            unique_qkv: [1, num_heads, total_unique_prompt_tokens, head_dim] - packed format
        """
        prompt_groups = reconstruction_info["prompt_groups"]
        prompt_len = reconstruction_info["prompt_len"]
        cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]
        original_attention_mask = reconstruction_info["original_attention_mask"]

        # Remove batch dimension: [1, num_heads, total_valid_tokens, head_dim] -> [num_heads, total_valid_tokens, head_dim]
        qkv_states = qkv_states.squeeze(0)

        # Collect all unique prompt tensors (without padding)
        prompt_tensors = []
        prompt_lengths = []

        for group in prompt_groups:
            # Get first sequence from this group
            first_seq_idx = group[0]

            # Extract this sequence using cu_seqlens
            seq_start = cu_seqlens_packed[first_seq_idx].item()
            seq_end = cu_seqlens_packed[first_seq_idx + 1].item()
            seq_qkv = qkv_states[:, seq_start:seq_end, :]  # [num_heads, seq_valid_tokens, head_dim]

            # Get number of valid prompt tokens from attention mask
            sample_mask = original_attention_mask[first_seq_idx]
            prompt_mask = sample_mask[:prompt_len]
            num_valid_prompt = prompt_mask.sum().item()

            # Extract prompt part (first num_valid_prompt tokens)
            prompt_qkv = seq_qkv[:, :num_valid_prompt, :]  # [num_heads, prompt_valid_tokens, head_dim]
            prompt_tensors.append(prompt_qkv)
            prompt_lengths.append(num_valid_prompt)

        # Concatenate all unique prompts along token dimension (packed format)
        # [num_heads, total_unique_prompt_tokens, head_dim]
        unique_qkv_packed = torch.cat(prompt_tensors, dim=1)

        # Add batch dimension: [1, num_heads, total_unique_prompt_tokens, head_dim]
        unique_qkv_packed = unique_qkv_packed.unsqueeze(0)

        # Build cu_seqlens for unique prompts (cumulative token positions)
        cu_seqlens_unique_prompts = torch.tensor(
            [0] + torch.cumsum(torch.tensor(prompt_lengths, dtype=torch.int32), dim=0).tolist(),
            dtype=torch.int32,
            device=qkv_states.device,
        )
        reconstruction_info["cu_seqlens_unique_prompts"] = cu_seqlens_unique_prompts
        reconstruction_info["max_prompt_valid_len"] = max(prompt_lengths)
        reconstruction_info["prompt_lengths"] = prompt_lengths  # Store individual lengths for reconstruction

        # Return packed tensor: [1, num_heads, total_unique_prompt_tokens, head_dim]
        return unique_qkv_packed

    @staticmethod
    def extract_padded_responses_from_deduped_packed_ids(
        packed_ids: torch.Tensor, reconstruction_info: Dict, offset: int = 0
    ) -> torch.Tensor:
        """
        Extract responses from deduplicated packed sequences and pad them.

        Args:
            packed_ids: [total_valid_tokens] - packed format consisting of prompt and response tokens with prompts deduplicated
                        Structure: [unique_prompt_0, response_0_0, response_0_1, ..., unique_prompt_1, response_1_0, ...]
            reconstruction_info: Metadata with prompt groups and segment_info
            offset: offset to add to the packed_position.

        Returns:
            responses: [original_batch, response_len]
        """
        response_length = reconstruction_info["response_length"]
        prompt_groups = reconstruction_info["prompt_groups"]
        segment_info = reconstruction_info["segment_info"]
        original_batch_size = reconstruction_info["original_batch_size"]

        # Initialize response list in original batch order
        responses = [None] * original_batch_size

        # Iterate through prompt_groups to track cumulative positions correctly
        # (avoiding double-counting prompts which appear once per group)
        packed_position = 0

        for group_idx, group in enumerate(prompt_groups):
            # Get prompt segment info (all samples in group share the same prompt entry)
            prompt_segs = [s for s in segment_info if s["group_idx"] == group_idx and s["type"] == "prompt"]
            prompt_ids = None
            if prompt_segs:
                prompt_valid = prompt_segs[0]["num_valid_tokens"]

                # prompt_ids from prompt before using response tokens
                prompt_ids = packed_ids[packed_position + prompt_valid + offset : packed_position + prompt_valid]

                # Skip past the prompt
                packed_position += prompt_valid

            # Now process response segments for each sample in the group
            for sample_idx in group:
                response_segs = [
                    s
                    for s in segment_info
                    if s["group_idx"] == group_idx and s["type"] == "response" and s["original_idx"] == sample_idx
                ]
                if response_segs:
                    response_valid = response_segs[0]["num_valid_tokens"]

                    # Extract response tokens from packed tensor
                    response_ids = packed_ids[
                        packed_position : packed_position + offset + response_valid
                    ]  # [response_valid]

                    # Pad to response_length
                    if response_valid <= response_length:
                        # Pad with zeros
                        pad_length = response_length - response_valid
                        response_ids = torch.cat(
                            [
                                prompt_ids,
                                response_ids,
                                torch.zeros(pad_length, dtype=response_ids.dtype, device=response_ids.device),
                            ],
                            dim=0,
                        )
                    elif response_valid > response_length:
                        # Truncate if somehow longer
                        response_ids = torch.cat(
                            [
                                prompt_ids,
                                response_ids[: response_length + offset],
                            ],
                            dim=0,
                        )

                    # Store in original batch position
                    responses[sample_idx] = response_ids

                    # Move to next response position
                    packed_position += response_valid

        # Stack into batch: [original_batch_size, response_length]
        responses = torch.stack(responses, dim=0)

        return responses

    @staticmethod
    def extract_unpadded_responses_from_deduped_packed_ids(
        packed_tensor: torch.Tensor, reconstruction_info: Dict, offset: int = 0
    ) -> torch.Tensor:
        """
        Extract responses from deduplicated packed sequences, keeping them packed (unpadded).

        Similar to extract_padded_responses_from_deduped_packed_ids but:
        1. Input can be multi-dimensional with first dimension being total_dedup_tokens
        2. Output remains packed (not padded)

        Args:
            packed_tensor: [total_dedup_tokens, *extra_dims] - packed format with prompts deduplicated
                          Structure: [unique_prompt_0, response_0_0, response_0_1, ..., unique_prompt_1, response_1_0, ...]
            reconstruction_info: Metadata with prompt groups and segment_info
            offset: offset to include last |offset| tokens from prompt and exclude last |offset| tokens from response.
                    Typically -1 for log probability computation.

        Returns:
            packed_responses: [total_response_tokens, *extra_dims] - packed responses only
        """
        prompt_groups = reconstruction_info["prompt_groups"]
        segment_info = reconstruction_info["segment_info"]

        # Collect all response segments
        response_segments = []

        # Track position in packed tensor
        packed_position = 0

        for group_idx, group in enumerate(prompt_groups):
            # Get prompt segment info (all samples in group share the same prompt entry)
            prompt_segs = [s for s in segment_info if s["group_idx"] == group_idx and s["type"] == "prompt"]
            prompt_start = packed_position
            prompt_valid = 0
            if prompt_segs:
                prompt_valid = prompt_segs[0]["num_valid_tokens"]
                # Skip past the prompt
                packed_position += prompt_valid

            # Now process response segments for each sample in the group
            for sample_idx in group:
                response_segs = [
                    s
                    for s in segment_info
                    if s["group_idx"] == group_idx and s["type"] == "response" and s["original_idx"] == sample_idx
                ]
                if response_segs:
                    response_valid = response_segs[0]["num_valid_tokens"]

                    # Extract prompt suffix (last |offset| tokens from prompt)
                    if offset < 0 and prompt_valid > 0:
                        prompt_suffix = packed_tensor[
                            prompt_start + prompt_valid + offset : prompt_start + prompt_valid
                        ]
                    else:
                        prompt_suffix = None

                    # Extract response prefix (all but last |offset| tokens)
                    response_end = (
                        packed_position + response_valid + offset if offset < 0 else packed_position + response_valid
                    )
                    response_tokens = packed_tensor[packed_position:response_end]

                    # Concatenate prompt suffix and response prefix
                    if prompt_suffix is not None and prompt_suffix.numel() > 0:
                        segment = torch.cat([prompt_suffix, response_tokens], dim=0)
                    else:
                        segment = response_tokens

                    response_segments.append(segment)

                    # Move to next response position
                    packed_position += response_valid

        # Concatenate all response segments
        packed_responses = torch.cat(response_segments, dim=0)

        return packed_responses

    @staticmethod
    def pad_responses(packed_responses: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Pad packed responses and arrange into batched format.

        Takes the output of extract_unpadded_responses_from_deduped_packed_ids and converts
        it to a padded batch tensor in original sample order.

        Args:
            packed_responses: [total_response_tokens, *extra_dims] - packed responses
                             (output from extract_unpadded_responses_from_deduped_packed_ids)
            reconstruction_info: Metadata with prompt groups and segment_info

        Returns:
            padded_responses: [original_batch_size, response_length, *extra_dims] - padded batch
        """
        response_length = reconstruction_info["response_length"]
        prompt_groups = reconstruction_info["prompt_groups"]
        segment_info = reconstruction_info["segment_info"]
        original_batch_size = reconstruction_info["original_batch_size"]

        # Get extra dimensions from input tensor
        extra_dims = packed_responses.shape[1:]

        # Initialize output tensor with zeros (for padding)
        output_shape = (original_batch_size, response_length) + extra_dims
        padded_responses = torch.zeros(output_shape, dtype=packed_responses.dtype, device=packed_responses.device)

        # Track position in packed tensor
        # Note: packed_responses contains responses in the same order as extract_unpadded_responses_from_deduped_packed_ids
        # which iterates through groups, then samples within each group
        packed_position = 0

        for group_idx, group in enumerate(prompt_groups):
            for sample_idx in group:
                response_segs = [
                    s
                    for s in segment_info
                    if s["group_idx"] == group_idx and s["type"] == "response" and s["original_idx"] == sample_idx
                ]
                if response_segs:
                    response_valid = response_segs[0]["num_valid_tokens"]

                    # The actual length in packed_responses after offset adjustment
                    # (extract_unpadded_responses_from_deduped_packed_ids maintains response_valid tokens per sample)
                    actual_len = min(response_valid, response_length)

                    # Extract from packed tensor
                    response_tokens = packed_responses[packed_position : packed_position + actual_len]

                    # Place in output (zeros already handle padding)
                    padded_responses[sample_idx, :actual_len] = response_tokens

                    # Move to next response position in packed tensor
                    packed_position += response_valid

        return padded_responses

    @staticmethod
    def responses_in_orig_sample_order(packed_responses: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Return the same 1D responses tensor, but in the original sample order (should it have changed to deal with a situation where incoming samples aren't already ordered by prompt groups)

        Args:
            packed_responses: [total_response_tokens, *extra_dims] - packed responses
                             (output from extract_unpadded_responses_from_deduped_packed_ids)
            reconstruction_info: Metadata with prompt groups and segment_info

        Returns:
            responses: [total_response_tokens, *extra_dims] - restored original sample order batch
        """
        device = packed_responses.device

        prompt_groups = reconstruction_info["prompt_groups"]
        cu_seqlens_response = reconstruction_info["cu_seqlens_response"]

        # flatten prompt groups into a flag permutation indices
        indices_order = functools.reduce(operator.iconcat, prompt_groups, [])
        num_samples = len(indices_order)

        # if samples haven't been permuted, i.e. flattened prompt_groups == [0,1,2,...,n] - then return immediately
        if all(indices_order[i] == i for i in range(num_samples)):
            return packed_responses

        # the rest of the logic is a cursor-generated fast vectorized version of the following python equivalent:
        # samples = [None] * num_samples
        # for cur_pos in range(num_samples):
        #     orig_pos = remap[cur_pos]
        #     samples[orig_pos] = inputs[cu_seqlens_response[cur_pos]:cu_seqlens_response[cur_pos + 1]]
        # return torch.cat(samples)

        indices_order = torch.tensor(indices_order, dtype=torch.long, device=device)

        # Length of each sample as it sits in the input
        lengths = cu_seqlens_response[1:] - cu_seqlens_response[:-1]

        # Invert the permutation: inv_indices_order[original_idx] = current_position_in_input
        inv_indices_order = torch.empty(num_samples, dtype=torch.long, device=device)
        inv_indices_order[indices_order] = torch.arange(num_samples, device=device)

        # Lengths reordered to the original sample order
        orig_lengths = lengths[inv_indices_order]

        # Cumulative offsets for the *output* tensor (original order)
        orig_offsets = torch.cat(
            [torch.zeros(1, dtype=cu_seqlens_response.dtype, device=device), orig_lengths.cumsum(dim=0)]
        )

        # For every element in the output, determine:
        #   1) which original sample it belongs to
        #   2) its position within that sample
        output_sample_id = torch.arange(num_samples, device=device).repeat_interleave(orig_lengths)
        pos_within_sample = torch.arange(packed_responses.shape[0], device=device) - orig_offsets[output_sample_id]

        # Map each original sample id back to its position in the input
        input_position = inv_indices_order[output_sample_id]

        # Absolute index into the input tensor
        gather_idx = cu_seqlens_response[input_position] + pos_within_sample

        return packed_responses[gather_idx]

    @staticmethod
    def extract_response_queries(query_states: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Extract response queries from full query states.

        Args:
            query_states: Depends on is_unpadded flag:
                - If is_unpadded=False: [batch_size, num_heads, seq_len, head_dim] (padded)
                - If is_unpadded=True: [1, num_heads, total_valid_tokens, head_dim] (packed)
            reconstruction_info: Metadata with prompt groups

        Returns:
            response_q: [batch_size, num_heads, response_len, head_dim]
        """
        is_unpadded = reconstruction_info.get("is_unpadded", False)

        if is_unpadded:
            return ZoRRoTrain._extract_response_queries_packed(query_states, reconstruction_info)
        else:
            return ZoRRoTrain._extract_response_queries_padded(query_states, reconstruction_info)

    @staticmethod
    def _extract_response_queries_padded(query_states: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Extract response queries from padded sequences.

        Args:
            query_states: [batch_size, num_heads, seq_len, head_dim]
            reconstruction_info: Metadata with prompt groups

        Returns:
            response_q: [batch_size, num_heads, response_len, head_dim]
        """
        prompt_len = reconstruction_info["prompt_len"]

        # Simply extract response part from all sequences
        # query_states[:, :, prompt_len:, :] extracts response queries
        return query_states[:, :, prompt_len:, :]

    @staticmethod
    def _extract_cu_seqlens(reconstruction_info: Dict, device) -> torch.Tensor:
        """
        Extract cu_seqlens from reconstruction_info.

        Args:
            reconstruction_info: Metadata with cu_seqlens

        Returns:
            cu_seqlens: [num_sequences + 1] tensor
        """

        original_attention_mask = reconstruction_info["original_attention_mask"]

        prompt_len = reconstruction_info["prompt_len"]
        cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]
        prompt_groups = reconstruction_info["prompt_groups"]
        original_batch_size = reconstruction_info["original_batch_size"]

        # ex: [[0, 1, 4, 7], [2, 3, 5, 6]] -> {0: 0, 1: 1, 4: 2, 7: 3, 2: 4, 3: 5, 5: 6, 6: 7}
        reordered_seq_idx = {}
        for group in prompt_groups:
            for sample_idx in group:
                reordered_seq_idx[sample_idx] = len(reordered_seq_idx)

        # Build cumulative sequence lengths for responses
        cu_seqlens_response = torch.empty(original_batch_size + 1, dtype=torch.int32, device=device)
        cu_seqlens_response[0] = 0
        for sample_idx in range(original_batch_size):
            # Extract this sequence using cu_seqlens
            seq_start = cu_seqlens_packed[sample_idx].item()
            seq_end = cu_seqlens_packed[sample_idx + 1].item()

            # Get number of valid prompt tokens from attention mask
            sample_mask = original_attention_mask[sample_idx]
            prompt_mask = sample_mask[:prompt_len]
            num_valid_prompt = prompt_mask.sum().item()

            # Update cu_seqlens for responses
            num_response_tokens = seq_end - num_valid_prompt - seq_start
            cu_seqlens_response[reordered_seq_idx[sample_idx] + 1] = num_response_tokens

        cu_seqlens_response.cumsum_(0)

        # Collect all unique prompt tensors (without padding)
        prompt_lengths = [0]

        for group in prompt_groups:
            # Extract this sequence using cu_seqlens
            first_seq_idx = group[0]  # Get the first sequence index for this group
            reordered_first_seq_idx = reordered_seq_idx[first_seq_idx]

            response_offset = cu_seqlens_response[reordered_first_seq_idx].item()
            response_len = (cu_seqlens_response[reordered_first_seq_idx + 1] - response_offset).item()

            seq_start = cu_seqlens_packed[first_seq_idx].item()
            seq_end = cu_seqlens_packed[first_seq_idx + 1].item()
            prompt_response_len = seq_end - seq_start
            prompt_len = prompt_response_len - response_len
            prompt_lengths.append(prompt_len)

        # Build cu_seqlens for unique prompts (cumulative token positions)
        cu_seqlens_unique_prompts = torch.tensor(prompt_lengths, dtype=torch.int32, device=device).cumsum_(dim=0)

        reconstruction_info["cu_seqlens_unique_prompts"] = cu_seqlens_unique_prompts
        reconstruction_info["max_prompt_valid_len"] = max(prompt_lengths)
        reconstruction_info["prompt_lengths"] = prompt_lengths  # Store individual lengths for reconstruction
        reconstruction_info["reordered_seq_idx"] = reordered_seq_idx

        return cu_seqlens_response

    # @staticmethod
    # def _extract_cu_seqlens(
    #     reconstruction_info: Dict
    # ) -> torch.Tensor:
    #     """
    #     Extract cu_seqlens from reconstruction_info.

    #     Args:
    #         reconstruction_info: Metadata with cu_seqlens

    #     Returns:
    #         cu_seqlens: [num_sequences + 1] tensor
    #     """

    #     cu_seqlens_response = [0]  # Build cumulative sequence lengths for responses
    #     original_attention_mask = reconstruction_info['original_attention_mask']

    #     prompt_len = reconstruction_info['prompt_len']
    #     cu_seqlens_packed = reconstruction_info['cu_seqlens_packed']
    #     original_batch_size = reconstruction_info['original_batch_size']
    #     for sample_idx in range(original_batch_size):
    #         # Extract this sequence using cu_seqlens
    #         seq_start = cu_seqlens_packed[sample_idx].item()
    #         seq_end = cu_seqlens_packed[sample_idx + 1].item()

    #         # Get number of valid prompt tokens from attention mask
    #         sample_mask = original_attention_mask[sample_idx]
    #         prompt_mask = sample_mask[:prompt_len]
    #         num_valid_prompt = prompt_mask.sum().item()

    #         # Update cu_seqlens for responses
    #         num_response_tokens = (seq_end - num_valid_prompt - seq_start)
    #         cu_seqlens_response.append(cu_seqlens_response[-1] + num_response_tokens)
    #     return torch.tensor(cu_seqlens_response, dtype=torch.int32, device=f'cuda:{torch.cuda.current_device()}')

    @staticmethod
    def _extract_response_queries_packed(query_states: torch.Tensor, reconstruction_info: Dict) -> torch.Tensor:
        """
        Extract response queries from packed sequences (unpadded).

        Args:
            query_states: [1, num_heads, total_valid_tokens, head_dim] - packed format
            reconstruction_info: Metadata with prompt groups and cu_seqlens

        Returns:
            response_q: [1, num_heads, total_response_tokens, head_dim] - packed format with only response tokens
        """
        prompt_len = reconstruction_info["prompt_len"]
        cu_seqlens_packed = reconstruction_info["cu_seqlens_packed"]
        original_attention_mask = reconstruction_info["original_attention_mask"]
        original_batch_size = reconstruction_info["original_batch_size"]

        # Remove batch dimension: [1, num_heads, total_valid_tokens, head_dim] -> [num_heads, total_valid_tokens, head_dim]
        query_states = query_states.squeeze(0)

        # Extract response portion for each sequence and concatenate them
        response_queries = []
        cu_seqlens_response = [0]  # Build cumulative sequence lengths for responses

        for sample_idx in range(original_batch_size):
            # Extract this sequence using cu_seqlens
            seq_start = cu_seqlens_packed[sample_idx].item()
            seq_end = cu_seqlens_packed[sample_idx + 1].item()
            seq_query = query_states[:, seq_start:seq_end, :]  # [num_heads, seq_valid_tokens, head_dim]

            # Get number of valid prompt tokens from attention mask
            sample_mask = original_attention_mask[sample_idx]
            prompt_mask = sample_mask[:prompt_len]
            num_valid_prompt = prompt_mask.sum().item()

            # Extract response part (after prompt tokens)
            response_query = seq_query[:, num_valid_prompt:, :]  # [num_heads, response_valid_tokens, head_dim]
            response_queries.append(response_query)

            # Update cu_seqlens for responses
            num_response_tokens = response_query.shape[1]
            cu_seqlens_response.append(cu_seqlens_response[-1] + num_response_tokens)

        # Concatenate along token dimension: [num_heads, total_response_tokens, head_dim]
        response_q_packed = torch.cat(response_queries, dim=1)

        # Add batch dimension: [1, num_heads, total_response_tokens, head_dim]
        response_q_packed = response_q_packed.unsqueeze(0)

        # Store cu_seqlens_response in reconstruction_info for use in flash attention
        cu_seqlens_response_tensor = torch.tensor(cu_seqlens_response, dtype=torch.int32, device=query_states.device)
        reconstruction_info["cu_seqlens_response"] = cu_seqlens_response_tensor
        reconstruction_info["max_response_valid_len"] = max(
            cu_seqlens_response_tensor[1:] - cu_seqlens_response_tensor[:-1]
        )

        return response_q_packed

    @staticmethod
    def replicate_and_concat_prompt_response(
        prompt_outputs: torch.Tensor, response_outputs: torch.Tensor, reconstruction_info: Dict
    ) -> torch.Tensor:
        """
        Replicate prompt outputs and concatenate with response outputs.

        Note: attention_forward returns tensors with shape [batch, seq_len, num_heads, head_dim]
        after internally transposing from [batch, num_heads, seq_len, head_dim].

        Args:
            prompt_outputs: Depends on is_unpadded flag:
                - If is_unpadded=False: [num_unique_prompts, prompt_len, num_heads, head_dim]
                - If is_unpadded=True: [1, total_unique_prompt_tokens, num_heads, head_dim] (packed, after attention)
            response_outputs: Depends on is_unpadded flag:
                - If is_unpadded=False: [batch_size, response_len, num_heads, head_dim]
                - If is_unpadded=True: [1, total_response_tokens, num_heads, head_dim] (packed, after attention)
            reconstruction_info: Metadata with prompt groups

        Returns:
            full_outputs: Depends on is_unpadded flag:
                - If is_unpadded=False: [batch_size, seq_len, num_heads, head_dim]
                - If is_unpadded=True: [1, total_valid_tokens, num_heads, head_dim] (packed)
        """
        is_unpadded = reconstruction_info.get("is_unpadded", False)

        if is_unpadded:
            return ZoRRoTrain._replicate_and_concat_prompt_response_packed(
                prompt_outputs, response_outputs, reconstruction_info
            )
        else:
            return ZoRRoTrain._replicate_and_concat_prompt_response_padded(
                prompt_outputs, response_outputs, reconstruction_info
            )

    @staticmethod
    def _replicate_and_concat_prompt_response_padded(
        prompt_outputs: torch.Tensor, response_outputs: torch.Tensor, reconstruction_info: Dict
    ) -> torch.Tensor:
        """
        Replicate and concatenate for padded sequences.

        Args:
            prompt_outputs: [num_unique_prompts, prompt_len, num_heads, head_dim]
            response_outputs: [batch_size, response_len, num_heads, head_dim]
            reconstruction_info: Metadata with prompt groups

        Returns:
            full_outputs: [batch_size, seq_len, num_heads, head_dim]
        """
        prompt_groups = reconstruction_info["prompt_groups"]
        batch_size = reconstruction_info["original_batch_size"]

        # Build mapping: sequence_idx -> group_idx
        seq_to_group = {}
        for group_idx, group in enumerate(prompt_groups):
            for seq_idx in group:  # group is a list of indices
                seq_to_group[seq_idx] = group_idx

        full_outputs = []
        for seq_idx in range(batch_size):
            group_idx = seq_to_group[seq_idx]

            # Get prompt output for this group
            prompt_out = prompt_outputs[group_idx : group_idx + 1]  # [1, prompt_len, num_heads, head_dim]

            # Get response output for this sequence
            response_out = response_outputs[seq_idx : seq_idx + 1]  # [1, response_len, num_heads, head_dim]

            # Concatenate along sequence dimension (dim=1)
            seq_out = torch.cat([prompt_out, response_out], dim=1)
            full_outputs.append(seq_out)

        # Stack: [batch_size, seq_len, num_heads, head_dim]
        return torch.cat(full_outputs, dim=0)

    @staticmethod
    def _replicate_and_concat_prompt_response_packed(
        prompt_outputs: torch.Tensor, response_outputs: torch.Tensor, reconstruction_info: Dict
    ) -> torch.Tensor:
        """
        Replicate and concatenate for packed sequences (unpadded).

        Note: Attention functions transpose output from [batch, heads, seq, dim] to [batch, seq, heads, dim]

        Args:
            prompt_outputs: [1, total_unique_prompt_tokens, num_heads, head_dim] - packed format after attention
            response_outputs: [1, total_response_tokens, num_heads, head_dim] - packed format after attention
            reconstruction_info: Metadata with prompt groups and cu_seqlens

        Returns:
            full_outputs: [1, total_valid_tokens, num_heads, head_dim] - packed format
        """
        prompt_groups = reconstruction_info["prompt_groups"]
        batch_size = reconstruction_info["original_batch_size"]
        cu_seqlens_response = reconstruction_info["cu_seqlens_response"]
        cu_seqlens_unique_prompts = reconstruction_info["cu_seqlens_unique_prompts"]

        # Build mapping: sequence_idx -> group_idx
        seq_to_group = {}
        for group_idx, group in enumerate(prompt_groups):
            for seq_idx in group:
                seq_to_group[seq_idx] = group_idx

        # Remove batch dimension
        # prompt_outputs: [1, total_unique_prompt_tokens, num_heads, head_dim] -> [total_unique_prompt_tokens, num_heads, head_dim]
        # response_outputs: [1, total_response_tokens, num_heads, head_dim] -> [total_response_tokens, num_heads, head_dim]
        prompt_outputs = prompt_outputs.squeeze(0)
        response_outputs = response_outputs.squeeze(0)

        # Build full packed sequence by concatenating prompt + response for each sample
        packed_sequences = []
        for seq_idx in range(batch_size):
            group_idx = seq_to_group[seq_idx]

            # Extract prompt output for this group from packed prompt_outputs
            # Use cu_seqlens_unique_prompts to find the boundaries
            prompt_start = cu_seqlens_unique_prompts[group_idx].item()
            prompt_end = cu_seqlens_unique_prompts[group_idx + 1].item()
            prompt_out = prompt_outputs[prompt_start:prompt_end, :, :]  # [prompt_len_valid, num_heads, head_dim]

            # Get response output for this sequence using cu_seqlens_response
            response_start = cu_seqlens_response[seq_idx].item()
            response_end = cu_seqlens_response[seq_idx + 1].item()
            response_out = response_outputs[
                response_start:response_end, :, :
            ]  # [response_len_valid, num_heads, head_dim]

            # Concatenate along token dimension (dim=0)
            seq_out = torch.cat([prompt_out, response_out], dim=0)  # [seq_len_valid, num_heads, head_dim]
            packed_sequences.append(seq_out)

        # Concatenate all sequences along token dimension: [total_valid_tokens, num_heads, head_dim]
        packed_output = torch.cat(packed_sequences, dim=0)

        # Add batch dimension: [1, total_valid_tokens, num_heads, head_dim]
        return packed_output.unsqueeze(0)

    @staticmethod
    def _replicate_and_concat_prompt_responses(
        prompt_outputs: torch.Tensor,
        response_outputs: torch.Tensor,
        reconstruction_info: Dict,
        cu_seqlens_response: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replicate and concatenate for packed sequences (unpadded).

        Note: Attention functions transpose output from [batch, heads, seq, dim] to [batch, seq, heads, dim]

        Args:
            prompt_outputs: [1, total_unique_prompt_tokens, num_heads, head_dim] - packed format after attention
            response_outputs: [1, total_response_tokens, num_heads, head_dim] - packed format after attention
            reconstruction_info: Metadata with prompt groups and cu_seqlens
        Returns:
            full_outputs: [1, total_valid_tokens, num_heads, head_dim] - packed format
        """

        prompt_groups = reconstruction_info["prompt_groups"]
        reordered_seq_idx = reconstruction_info["reordered_seq_idx"]

        # Remove batch dimension: [1, num_heads, total_valid_tokens, head_dim] -> [num_heads, total_valid_tokens, head_dim]

        response_outputs = response_outputs.squeeze(0)  # [total_dedup_valid_tokens, *extra_dims]

        # Collect all unique prompt tensors (without padding)
        prev_packed_response_lengths = 0
        all_tensors = []

        prompt_lengths = reconstruction_info["prompt_lengths"][1:]
        prompt_tensors = torch.split(prompt_outputs.squeeze(0), prompt_lengths, dim=0)
        for gid, group in enumerate(prompt_groups):
            # Get first sequence from this group
            first_secondary_idx = reordered_seq_idx[group[-1]] + 1  # Last sample's response

            response_len = cu_seqlens_response[first_secondary_idx].item()

            response_qkv = response_outputs[prev_packed_response_lengths:response_len, :, :]

            prev_packed_response_lengths = response_len
            all_tensors.append(torch.cat([prompt_tensors[gid], response_qkv], dim=0))
        # Add batch dimension: [1, num_heads, total_unique_prompt_tokens, head_dim]
        reconstructed_seq = torch.cat(all_tensors, dim=0).unsqueeze(0)

        return reconstructed_seq
