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

"""CPU unit tests for dedup-aware micro-batch balancing (``rl/zorro_train/seqlen_balancing.py``).

``rearrange_micro_batches_with_dedup`` is the production load balancer the DeepSpeed worker uses to split a global
batch into micro-batches: it groups samples by shared prompt, greedily bin-packs whole groups by their deduplicated
token cost, and attaches the per-micro-batch deduplication metadata. The logic is pure tensor / Python bookkeeping
(the only distributed work is guarded behind ``torch.distributed.is_initialized()``), so it runs single-process on
CPU::

    pytest tests/zorro_train/test_seqlen_balancing.py

Batches use the same layout as the GPU RL tests under ``tests/rl/``: prompts LEFT-padded and responses RIGHT-padded,
with variable real lengths.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from arctic_platform.rl.zorro_train import ZoRRoTrain
from arctic_platform.rl.zorro_train.seqlen_balancing import rearrange_micro_batches_with_dedup
from arctic_platform.rl.zorro_train.tests import create_dummy_batch
from arctic_platform.testing_utils import TestCasePlus

vocab_size = 100
prompt_len = 16
response_len = 8


def _make_tensordict(batch_size: int, num_unique_prompts: int) -> TensorDict:
    torch.manual_seed(0)
    batch = create_dummy_batch(
        batch_size=batch_size,
        num_unique_prompts=num_unique_prompts,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=vocab_size,
        device="cpu",
        include_training_fields=False,
        add_padding=True,
    )
    return TensorDict(
        {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "position_ids": batch["position_ids"],
        },
        batch_size=[batch_size],
    )


class TestRearrangeMicroBatchesWithDedup(TestCasePlus):
    def test_partition_covers_every_sample_once(self):
        batch_size, num_unique_prompts = 8, 4
        batch = _make_tensordict(batch_size, num_unique_prompts)
        # A budget around a single group's cost (prompt + a couple of responses) forces more than one micro-batch.
        micro_batches, micro_indices = rearrange_micro_batches_with_dedup(
            batch, response_length=response_len, max_token_len=prompt_len + 2 * response_len
        )

        self.assertGreater(len(micro_batches), 1)
        self.assertEqual(len(micro_batches), len(micro_indices))
        flattened = sorted(idx for indices in micro_indices for idx in indices)
        self.assertEqual(flattened, list(range(batch_size)))

    def test_prompt_groups_stay_within_one_micro_batch(self):
        # Whole prompt groups are bin-packed as a unit, so every sample sharing a prompt must land together.
        batch_size, num_unique_prompts = 8, 2
        batch = _make_tensordict(batch_size, num_unique_prompts)
        groups, _ = ZoRRoTrain.find_prompt_groups(input_ids=batch["input_ids"], response_length=response_len)

        _, micro_indices = rearrange_micro_batches_with_dedup(
            batch, response_length=response_len, max_token_len=prompt_len + batch_size * response_len
        )

        owning_micro_batch = {sample: mb for mb, indices in enumerate(micro_indices) for sample in indices}
        for group in groups:
            assigned = {owning_micro_batch[sample] for sample in group}
            self.assertEqual(len(assigned), 1)

    def test_each_micro_batch_carries_dedup_metadata(self):
        batch_size, num_unique_prompts = 6, 2
        batch = _make_tensordict(batch_size, num_unique_prompts)
        micro_batches, _ = rearrange_micro_batches_with_dedup(
            batch, response_length=response_len, max_token_len=prompt_len + 2 * response_len
        )

        for micro_batch in micro_batches:
            self.assertIn("dedup_input_ids", micro_batch)
            self.assertIn("adapted_position_ids", micro_batch)
            self.assertIn("reconstruction_info", micro_batch)
            metrics = micro_batch["dedup_metrics"]
            # Any micro-batch with a shared prompt drops tokens; with a single sample the counts are simply equal.
            self.assertLessEqual(metrics["dedup_tokens"], metrics["orig_tokens"])
            self.assertEqual(metrics["tokens_saved"], metrics["orig_tokens"] - metrics["dedup_tokens"])

    def test_single_micro_batch_reconstructs_full_batch(self):
        # A budget larger than the whole batch keeps everything in one micro-batch; embedding the deduplicated ids and
        # reconstructing must reproduce the embeddings of the micro-batch's valid (unpadded, packed) tokens exactly.
        batch_size, num_unique_prompts = 6, 2
        batch = _make_tensordict(batch_size, num_unique_prompts)
        micro_batches, _ = rearrange_micro_batches_with_dedup(
            batch, response_length=response_len, max_token_len=10_000
        )
        self.assertEqual(len(micro_batches), 1)

        micro_batch = micro_batches[0]
        info = micro_batch["reconstruction_info"]
        dedup_input_ids = micro_batch["dedup_input_ids"]
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]

        embedding = torch.randn(vocab_size, 4, generator=torch.Generator().manual_seed(7))
        reconstructed = ZoRRoTrain.reconstruct_sequences(embedding[dedup_input_ids], info)

        valid_rows = [embedding[input_ids[row, attention_mask[row].bool()]] for row in range(input_ids.shape[0])]
        expected = torch.cat(valid_rows).unsqueeze(0)
        self.assertEqual(reconstructed.shape, expected.shape)
        self.assertTrue(torch.equal(reconstructed, expected))
