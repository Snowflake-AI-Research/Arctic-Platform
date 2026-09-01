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

"""CPU unit test for the flash-attention sequence packer (``processors/packing.py``).

``pack_sequences`` flattens a padded ``[B, S]`` tensor dict into a single ``[1, T]`` row with ``cu_seqlens`` /
``position_ids``; ``unpack_sequences`` reverses it; ``pad_packed_for_model`` pads the packed row to a 256-token page
boundary for the model forward. The live DeepSpeed worker runs with ``pack=False``, so these are exercised directly
on CPU via a pack -> unpack round-trip (no GPU, no dist)::

    pytest tests/rl/test_packing.py
"""

from __future__ import annotations

import torch

from arctic_platform.rl.processors.packing import N_TOKENS_PER_PAGE
from arctic_platform.rl.processors.packing import pack_sequences
from arctic_platform.rl.processors.packing import pad_packed_for_model
from arctic_platform.rl.processors.packing import unpack_sequences
from arctic_platform.testing_utils import TestCasePlus

seq_len = 6
real_token_counts = [6, 4, 2]  # per-row real tokens; total T = 12


def _make_padded_dict() -> dict:
    gen = torch.Generator().manual_seed(0)
    batch_size = len(real_token_counts)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for row, count in enumerate(real_token_counts):
        attention_mask[row, :count] = 1
    # Zero the padded region so an unpack (which pad-fills with 0) reconstructs the input exactly.
    input_ids = torch.randint(1, 100, (batch_size, seq_len), generator=gen) * attention_mask
    extra = torch.randn(batch_size, seq_len, 2, generator=gen) * attention_mask.unsqueeze(-1)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "extra": extra}


class TestPackSequences(TestCasePlus):
    def test_pack_shapes_and_metadata(self):
        data = _make_padded_dict()
        total_tokens = sum(real_token_counts)
        packed = pack_sequences(data)

        self.assertEqual(packed["input_ids"].shape, (1, total_tokens))
        self.assertEqual(packed["extra"].shape, (1, total_tokens, 2))
        self.assertEqual(packed["cu_seqlens"].tolist(), [0, 6, 10, 12])
        # position_ids restart at 0 within each packed sequence.
        expected_positions = torch.cat([torch.arange(n) for n in real_token_counts])
        self.assertTrue(torch.equal(packed["position_ids"][0], expected_positions))

    def test_round_trip_reconstructs_padded_tensors(self):
        data = _make_padded_dict()
        packed = pack_sequences(data)
        meta = packed["_pack_meta"]
        for key in ("input_ids", "extra"):
            unpacked = unpack_sequences(packed[key], meta)
            self.assertTrue(torch.equal(unpacked, data[key]), f"{key} did not round-trip through pack/unpack")

    def test_pad_packed_for_model_aligns_to_page(self):
        data = _make_padded_dict()
        packed = pack_sequences(data)
        model_kwargs, pad_length = pad_packed_for_model(packed)

        total_tokens = sum(real_token_counts)
        padded_len = model_kwargs["input_ids"].shape[1]
        self.assertEqual(padded_len % N_TOKENS_PER_PAGE, 0, "packed row not aligned to a page boundary")
        self.assertEqual(pad_length, padded_len - total_tokens)
        self.assertEqual(model_kwargs["max_length_q"], max(real_token_counts))
        self.assertTrue(torch.equal(model_kwargs["cu_seq_lens_q"], packed["cu_seqlens"]))
        self.assertEqual(model_kwargs["seq_idx"].shape, (1, padded_len))
        self.assertEqual(model_kwargs["seq_idx"][0, :total_tokens].tolist(), [0] * 6 + [1] * 4 + [2] * 2)
        # Right-padded tail gets its own segment id so conv1d cannot leak into pad.
        if pad_length:
            self.assertTrue(torch.all(model_kwargs["seq_idx"][0, total_tokens:] == 3))


class TestVarlenKwargs(TestCasePlus):
    def test_model_reads_varlen_from_layer_types(self):
        from types import SimpleNamespace

        from arctic_platform.rl.processors.packing import model_reads_varlen_kwargs

        hybrid = SimpleNamespace(config=SimpleNamespace(layer_types=["full_attention", "linear_attention"]))
        dense = SimpleNamespace(config=SimpleNamespace(layer_types=["full_attention"]))
        text_cfg = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(layer_types=["linear_attention"]), layer_types=None)
        )
        self.assertTrue(model_reads_varlen_kwargs(hybrid))
        self.assertFalse(model_reads_varlen_kwargs(dense))
        self.assertTrue(model_reads_varlen_kwargs(text_cfg))

    def test_ulysses_sp_returns_empty_kwargs(self):
        from arctic_platform.rl.processors.packing import derive_varlen_model_kwargs

        packed = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
        }
        self.assertEqual(derive_varlen_model_kwargs(packed), {})

    def test_packing_boundaries_reset_position_ids(self):
        from arctic_platform.rl.processors.packing import packing_boundaries_from_attention_mask

        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        cu, pos = packing_boundaries_from_attention_mask(mask)
        self.assertEqual(cu.tolist(), [0, 3, 5])
        self.assertEqual(pos.tolist(), [[0, 1, 2, 0, 1]])


class TestEngineForwardKwargs(TestCasePlus):
    def test_strips_cu_seqlens_from_batch_and_keeps_varlen(self):
        from arctic_platform.rl.processors.pipeline import _engine_forward_kwargs

        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        seq_idx = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
        batch = {
            "input_ids": torch.ones(1, 8, dtype=torch.long),
            "cu_seqlens": cu,
            "cu_seq_lens_q": cu,
            "seq_idx": seq_idx,
        }
        fwd = _engine_forward_kwargs(batch, {})
        self.assertNotIn("cu_seqlens", fwd)
        self.assertTrue(torch.equal(fwd["cu_seq_lens_q"], cu))
        self.assertTrue(torch.equal(fwd["seq_idx"], seq_idx))
        self.assertTrue(torch.equal(fwd["input_ids"], batch["input_ids"]))

    def test_strips_cu_seqlens_from_meta(self):
        from arctic_platform.rl.processors.pipeline import _engine_forward_kwargs

        cu = torch.tensor([0, 3], dtype=torch.int32)
        batch = {"input_ids": torch.ones(1, 3, dtype=torch.long)}
        meta = {"cu_seqlens": cu, "pad_token_id": 0}
        fwd = _engine_forward_kwargs(batch, meta)
        self.assertNotIn("cu_seqlens", fwd)
        self.assertEqual(fwd["pad_token_id"], 0)
        self.assertTrue(torch.equal(fwd["input_ids"], batch["input_ids"]))

    def test_collect_model_outputs_accepts_prime_dict(self):
        from arctic_platform.rl.processors.pipeline import collect_model_outputs

        logprobs = torch.zeros(1, 4)
        out = collect_model_outputs({"logprobs": logprobs, "entropy": torch.ones(1, 4), "logits": None})
        self.assertTrue(torch.equal(out["logprobs"], logprobs))
        self.assertTrue(torch.equal(out["entropy"], torch.ones(1, 4)))
        self.assertNotIn("logits", out)

    def test_chunked_lm_head_kwargs_stay_off_for_vanilla_linear(self):
        from arctic_platform.rl.processors.pipeline import _maybe_add_chunked_lm_head_kwargs

        class Engine:
            module = torch.nn.Linear(4, 4)

        batch = {"input_ids": torch.arange(6)}
        fwd = _maybe_add_chunked_lm_head_kwargs(Engine(), batch)
        self.assertNotIn("labels", fwd)
        self.assertNotIn("temperature", fwd)


class TestPackedMetricAggregation(TestCasePlus):
    """The ``--max-tokens-per-mb`` path must combine per-microbatch metrics with
    the same paired-key convention as the GAS worker (``combine_metric_microbatches``):
    ``{name}.sum`` / ``{name}.tokens`` are SUMMED so the server folds
    ``Σsum / Σtokens``; other keys are averaged; ``avg_loss`` is summed. A
    token-weighted mean (the previous behaviour) biased ``kl/per_token`` to
    ``Σ(S·c) / Σc²`` and mean-averaged the parity ``*_max`` gate inputs.
    """

    def test_paired_keys_summed_not_weight_averaged(self):
        from unittest import mock

        from arctic_platform.rl.processors import pipeline

        # 3 rollouts of real length [6, 4, 2]; capacity 6 -> exactly 2 microbatches.
        attn = torch.zeros(3, seq_len, dtype=torch.long)
        for row, count in enumerate(real_token_counts):
            attn[row, :count] = 1
        input_ids = torch.randint(1, 100, (3, seq_len)) * attn
        batch = {"input_ids": input_ids, "attention_mask": attn, "loss_mask": attn.clone()}

        per_mb = {
            "avg_loss": 2.0,
            "metrics": {"distill_kl.sum": 4.0, "distill_kl.tokens": 2.0, "distill_kl_max": 3.0},
            "batch": {},
        }

        with mock.patch.object(pipeline, "run_pipeline", return_value=per_mb) as m:
            out = pipeline._run_pipeline_with_packing(
                object(),
                (),
                batch,
                {},
                {},
                "cpu",
                backward=False,
                max_tokens_per_mb=6,
            )

        n = m.call_count
        self.assertGreaterEqual(n, 2, "capacity 6 over lengths [6,4,2] should split into >=2 microbatches")
        # Paired accumulators grow with microbatch count (summed), not collapsed to one mb's value.
        self.assertAlmostEqual(out["metrics"]["distill_kl.sum"], 4.0 * n)
        self.assertAlmostEqual(out["metrics"]["distill_kl.tokens"], 2.0 * n)
        # Non-paired keys (e.g. the parity ``*_max`` gate inputs) are averaged.
        self.assertAlmostEqual(out["metrics"]["distill_kl_max"], 3.0)
        # Globally-normalized per-mb losses sum to the whole-batch loss.
        self.assertAlmostEqual(out["avg_loss"], 2.0 * n)
