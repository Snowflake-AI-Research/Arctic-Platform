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

"""CPU unit tests for the ZoRRO prompt-deduplication algorithm (``rl/zorro_train/zorro_train.py``).

ZoRRO packs rollouts that share a prompt as ``[prompt][resp_0][resp_1]...`` so the shared prompt is encoded once.
The three static methods that implement this are pure tensor logic and need no model, so they are exercised here on
CPU::

    pytest tests/zorro_train/test_dedup.py

Batches use the same layout as the GPU RL tests under ``tests/rl/``: prompts are LEFT-padded and responses
RIGHT-padded to a fixed ``[left_pad][prompt][response][right_pad]`` row, with variable real prompt/response lengths.

Coverage:
    * ``find_prompt_groups`` groups rows by prompt identity and returns the unique prompts.
    * ``create_deduplicated_batch`` -> ``reconstruct_sequences`` round-trips per-token tensors exactly, for both the
      padded and unpadded/packed layouts. The model forward only ever sees per-token hidden states, so embedding the
      deduplicated token ids and reconstructing must reproduce the embeddings of the original batch bit-for-bit --
      this is the correctness guarantee the whole optimization rests on.
    * ``deduplicate_sequences`` is the exact inverse of ``reconstruct_sequences``.
    * deduplication actually drops the duplicated prompt tokens.

The GPU forward/backward equivalence of the attention patcher (``qwen_attention_patcher`` / ``qwen_model_patcher``)
is intentionally not covered here: it requires a real Qwen checkpoint (the patcher recomputes rotary embeddings on
the deduplicated positions and assumes Qwen3 head geometry, which a tiny random config does not satisfy).
"""

from __future__ import annotations

import torch
from parameterized import parameterized

from arctic_platform.rl.zorro_train import ZoRRoTrain
from arctic_platform.rl.zorro_train.tests import create_dummy_batch
from arctic_platform.testing_utils import TestCasePlus

vocab_size = 100
hidden_size = 8
prompt_len = 8
response_len = 8


def _embedding_table() -> torch.Tensor:
    return torch.randn(vocab_size, hidden_size, generator=torch.Generator().manual_seed(1234))


def _make_batch(batch_size: int, num_unique_prompts: int) -> dict:
    """Left-padded prompt / right-padded response batch with variable real lengths (verl / ``tests/rl`` convention)."""
    torch.manual_seed(0)
    return create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=vocab_size,
        device="cpu",
        include_training_fields=False,
        add_padding=True,
    )


class TestFindPromptGroups(TestCasePlus):
    def test_groups_shared_prompts(self):
        batch = _make_batch(batch_size=6, num_unique_prompts=2)
        groups, unique_prompts = ZoRRoTrain.find_prompt_groups(
            input_ids=batch["input_ids"], response_length=response_len
        )
        self.assertEqual(groups, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(unique_prompts.shape, (2, prompt_len))
        # Each group's rows must carry an identical prompt prefix.
        prompts = batch["input_ids"][:, :prompt_len]
        for group in groups:
            representative = prompts[group[0]]
            for row in group:
                self.assertTrue(torch.equal(prompts[row], representative))

    def test_all_prompts_unique(self):
        batch = _make_batch(batch_size=4, num_unique_prompts=4)
        groups, unique_prompts = ZoRRoTrain.find_prompt_groups(
            input_ids=batch["input_ids"], response_length=response_len
        )
        self.assertEqual(groups, [[0], [1], [2], [3]])
        self.assertEqual(unique_prompts.shape, (4, prompt_len))

    def test_single_shared_prompt(self):
        batch = _make_batch(batch_size=5, num_unique_prompts=1)
        groups, unique_prompts = ZoRRoTrain.find_prompt_groups(
            input_ids=batch["input_ids"], response_length=response_len
        )
        self.assertEqual(groups, [[0, 1, 2, 3, 4]])
        self.assertEqual(unique_prompts.shape, (1, prompt_len))


class TestRoundTrip(TestCasePlus):
    @parameterized.expand([("padded", False), ("unpadded", True)])
    def test_reconstruct_matches_original(self, _name, use_unpad):
        batch = _make_batch(batch_size=6, num_unique_prompts=2)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        groups, unique_prompts = ZoRRoTrain.find_prompt_groups(input_ids=input_ids, response_length=response_len)

        dedup_input_ids, _, info = ZoRRoTrain.create_deduplicated_batch(
            input_ids=input_ids,
            position_ids=batch["position_ids"],
            response_length=response_len,
            prompt_groups=groups,
            unique_prompts=unique_prompts,
            attention_mask=attention_mask,
            use_unpad=use_unpad,
        )

        embedding = _embedding_table()
        reconstructed = ZoRRoTrain.reconstruct_sequences(embedding[dedup_input_ids], info)

        if use_unpad:
            # Packed reconstruction: a single row holding each sequence's valid tokens concatenated.
            valid_rows = [embedding[input_ids[row, attention_mask[row].bool()]] for row in range(input_ids.shape[0])]
            expected = torch.cat(valid_rows).unsqueeze(0)
        else:
            expected = embedding[input_ids]

        self.assertEqual(reconstructed.shape, expected.shape)
        self.assertTrue(torch.equal(reconstructed, expected))

    def test_deduplicate_recovers_deduplicated_hidden(self):
        # ``reconstruct`` copies the representative prompt to every group member, so it is a left inverse of
        # ``deduplicate``: deduplicate(reconstruct(dedup_hidden)) == dedup_hidden (the reverse order is lossy).
        batch = _make_batch(batch_size=6, num_unique_prompts=2)
        input_ids = batch["input_ids"]
        groups, unique_prompts = ZoRRoTrain.find_prompt_groups(input_ids=input_ids, response_length=response_len)
        dedup_input_ids, _, info = ZoRRoTrain.create_deduplicated_batch(
            input_ids=input_ids,
            position_ids=batch["position_ids"],
            response_length=response_len,
            prompt_groups=groups,
            unique_prompts=unique_prompts,
            attention_mask=batch["attention_mask"],
        )

        dedup_hidden = torch.randn(1, dedup_input_ids.shape[1], hidden_size)
        round_tripped = ZoRRoTrain.deduplicate_sequences(ZoRRoTrain.reconstruct_sequences(dedup_hidden, info), info)
        self.assertTrue(torch.equal(round_tripped, dedup_hidden))


class TestTokenSavings(TestCasePlus):
    def test_dedup_drops_duplicate_prompts(self):
        batch_size, num_unique_prompts = 6, 2
        batch = _make_batch(batch_size=batch_size, num_unique_prompts=num_unique_prompts)
        input_ids = batch["input_ids"]
        groups, unique_prompts = ZoRRoTrain.find_prompt_groups(input_ids=input_ids, response_length=response_len)
        dedup_input_ids, _, _ = ZoRRoTrain.create_deduplicated_batch(
            input_ids=input_ids,
            position_ids=batch["position_ids"],
            response_length=response_len,
            prompt_groups=groups,
            unique_prompts=unique_prompts,
            attention_mask=batch["attention_mask"],
        )

        # Padded (non-unpacked) dedup keeps one full prompt block per group + each row's full response block.
        expected_tokens = num_unique_prompts * prompt_len + batch_size * response_len
        self.assertEqual(dedup_input_ids.shape, (1, expected_tokens))
        self.assertLess(dedup_input_ids.numel(), input_ids.numel())


class TestResponseLabelExtraction(TestCasePlus):
    def test_offset0_yields_each_rollouts_own_response(self):
        # The log-prob labels are extracted from the packed deduplicated ids with ``offset=0`` so each rollout gets
        # its OWN response tokens. This guards the first-response-token correctness: a naive roll of the packed
        # sequence would hand every rollout in a group the first rollout's first token (the shared prompt-final slot
        # is adjacent only to the first rollout's response), which is exactly the bug the offset-aware extraction
        # avoids. Pure tensor logic, so verified on CPU without a model.
        batch_size, num_unique_prompts = 6, 2
        batch = _make_batch(batch_size=batch_size, num_unique_prompts=num_unique_prompts)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        groups, unique_prompts = ZoRRoTrain.find_prompt_groups(input_ids=input_ids, response_length=response_len)
        dedup_input_ids, _, info = ZoRRoTrain.create_deduplicated_batch(
            input_ids=input_ids,
            position_ids=batch["position_ids"],
            response_length=response_len,
            prompt_groups=groups,
            unique_prompts=unique_prompts,
            attention_mask=attention_mask,
            use_unpad=True,
        )

        labels = ZoRRoTrain.extract_unpadded_responses_from_deduped_packed_ids(
            dedup_input_ids.squeeze(0), info, offset=0
        )

        # Each row's real response tokens (mask == 1 beyond the fixed prompt boundary), concatenated in row order.
        expected_rows = []
        for row in range(batch_size):
            real_prompt_len = int(attention_mask[row, :prompt_len].sum())
            expected_rows.append(input_ids[row, attention_mask[row].bool()][real_prompt_len:])
        expected = torch.cat(expected_rows)

        self.assertEqual(labels.shape, expected.shape)
        self.assertTrue(torch.equal(labels, expected))


class TestSampleOrderReordering(TestCasePlus):
    """``responses_in_orig_sample_order`` undoes the prompt-group permutation.

    Deduplication packs responses grouped by prompt, so when the incoming samples are not already ordered by prompt
    the packed responses come out permuted. This restores the original sample order. The method reads only
    ``prompt_groups`` and ``cu_seqlens_response`` from the reconstruction info, so it is exercised here directly with
    a hand-built layout and per-token values tagged by original sample id (variable response lengths to stress the
    vectorized offset math).
    """

    def test_reorders_non_contiguous_groups(self):
        # Samples 0,2 share one prompt and 1,3 another -> packed order is [0, 2, 1, 3].
        prompt_groups = [[0, 2], [1, 3]]
        # Response lengths of samples 0, 2, 1, 3 in that packed order.
        cu_seqlens_response = torch.tensor([0, 2, 3, 6, 10], dtype=torch.int32)
        packed = torch.tensor([0, 0, 2, 1, 1, 1, 3, 3, 3, 3], dtype=torch.float)
        info = {"prompt_groups": prompt_groups, "cu_seqlens_response": cu_seqlens_response}

        reordered = ZoRRoTrain.responses_in_orig_sample_order(packed, info)

        expected = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 3, 3], dtype=torch.float)
        self.assertTrue(torch.equal(reordered, expected))

    def test_already_ordered_is_noop(self):
        # Flattened prompt groups == [0, 1, 2, 3]; the early-return path must hand back the input untouched.
        cu_seqlens_response = torch.tensor([0, 2, 3, 6, 10], dtype=torch.int32)
        packed = torch.arange(10, dtype=torch.float)
        info = {"prompt_groups": [[0, 1], [2, 3]], "cu_seqlens_response": cu_seqlens_response}

        reordered = ZoRRoTrain.responses_in_orig_sample_order(packed, info)

        self.assertTrue(torch.equal(reordered, packed))
