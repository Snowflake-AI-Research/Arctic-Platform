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
"""Readable JSON specs -> the batch dict `ArcticClient.fwd_bwd` / `generate` take.

Only the pre-tokenized paths are covered here: the tokenizer path needs a real
model download, so it is exercised against a live job instead.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.client.cortex import batch


class TestFwdBwdBatch:
    def test_wraps_kwargs_in_the_batch_envelope(self):
        """The client takes {"args": [...], "kwargs": {...}}, not serialized bytes."""
        out = batch.build_fwd_bwd_batch({"payload": {"kwargs": {"input_ids": [[1, 2, 3]]}}})
        assert set(out) == {"args", "kwargs"}
        assert torch.equal(out["kwargs"]["input_ids"], torch.tensor([[1, 2, 3]]))

    def test_accepts_a_bare_payload(self):
        """A spec may be the payload itself, without the wrapper key."""
        out = batch.build_fwd_bwd_batch({"kwargs": {"input_ids": [[1]]}})
        assert out["kwargs"]["input_ids"].tolist() == [[1]]

    def test_passes_args_through(self):
        out = batch.build_fwd_bwd_batch({"payload": {"args": [1, "x"], "kwargs": {}}})
        assert out["args"] == [1, "x"]

    def test_rejects_non_list_args(self):
        with pytest.raises(ValueError, match="args must be a JSON list"):
            batch.build_fwd_bwd_batch({"payload": {"args": {"a": 1}, "kwargs": {}}})

    def test_honors_an_explicit_dtype(self):
        out = batch.build_fwd_bwd_batch(
            {"payload": {"kwargs": {"weights": {"data": [[1.0, 2.0]], "dtype": "float32"}}}}
        )
        assert out["kwargs"]["weights"].dtype == torch.float32

    def test_rejects_an_unknown_dtype(self):
        with pytest.raises(ValueError, match="unsupported tensor dtype"):
            batch.build_fwd_bwd_batch({"payload": {"kwargs": {"x": {"data": [1], "dtype": "quad"}}}})

    def test_leaves_scalars_alone(self):
        out = batch.build_fwd_bwd_batch({"payload": {"kwargs": {"input_ids": [[1]], "use_cache": False}}})
        assert out["kwargs"]["use_cache"] is False


class TestDerivedTensors:
    def test_builds_next_token_labels_by_default(self):
        out = batch.build_fwd_bwd_batch({"payload": {"input_ids": [[10, 11, 12]]}})
        assert out["kwargs"]["labels"].tolist() == [[11, 12, -100]]

    def test_self_labels_copy_the_input(self):
        out = batch.build_fwd_bwd_batch({"payload": {"input_ids": [[10, 11]], "labels": "input_ids"}})
        assert out["kwargs"]["labels"].tolist() == [[10, 11]]

    def test_labels_none_omits_them(self):
        out = batch.build_fwd_bwd_batch({"payload": {"input_ids": [[1, 2]], "labels": "none"}})
        assert "labels" not in out["kwargs"]

    def test_masks_padding_in_next_token_labels(self):
        spec = {"payload": {"input_ids": [[10, 11, 0]], "attention_mask": [[1, 1, 0]], "labels": "next_token"}}
        assert batch.build_fwd_bwd_batch(spec)["kwargs"]["labels"].tolist() == [[11, -100, -100]]

    def test_honors_a_custom_ignore_index(self):
        spec = {"payload": {"input_ids": [[10, 11]], "ignore_index": -1}}
        assert batch.build_fwd_bwd_batch(spec)["kwargs"]["labels"].tolist() == [[11, -1]]

    def test_include_attention_mask_synthesizes_ones(self):
        spec = {"payload": {"input_ids": [[1, 2]], "include_attention_mask": True}}
        assert batch.build_fwd_bwd_batch(spec)["kwargs"]["attention_mask"].tolist() == [[1, 1]]

    def test_arange_position_ids_broadcast_over_the_batch(self):
        spec = {"payload": {"input_ids": [[1, 2, 3], [4, 5, 6]], "position_ids": "arange"}}
        assert batch.build_fwd_bwd_batch(spec)["kwargs"]["position_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]

    def test_rejects_a_bad_position_spec(self):
        with pytest.raises(ValueError, match="position_ids must be"):
            batch.build_fwd_bwd_batch({"payload": {"input_ids": [[1]], "position_ids": "sequential"}})

    def test_rejects_a_rank1_input(self):
        with pytest.raises(ValueError, match=r"input_ids must have shape \[batch, seq_len\]"):
            batch.build_fwd_bwd_batch({"payload": {"input_ids": [1, 2, 3]}})

    def test_rejects_a_mismatched_label_shape(self):
        spec = {"payload": {"input_ids": [[1, 2]], "labels": {"data": [[1, 2, 3]]}}}
        with pytest.raises(ValueError, match="same shape as input_ids"):
            batch.build_fwd_bwd_batch(spec)


class TestGenerateSpec:
    def test_reads_prompts_and_params(self):
        spec = {"payload": {"prompts": ["hi"], "sampling_params": {"max_tokens": 8}}}
        assert batch.read_generate_spec(spec) == {
            "prompts": ["hi"],
            "sampling_params": {"max_tokens": 8},
            "routing_key": None,
            "strict": False,
        }

    def test_requires_prompts(self):
        with pytest.raises(ValueError, match="non-empty prompts list"):
            batch.read_generate_spec({"payload": {"prompts": []}})

    def test_accepts_per_prompt_params(self):
        spec = {"payload": {"prompts": ["a", "b"], "sampling_params": [{"max_tokens": 1}, None]}}
        assert batch.read_generate_spec(spec)["sampling_params"] == [{"max_tokens": 1}, None]

    def test_rejects_a_param_list_of_the_wrong_length(self):
        spec = {"payload": {"prompts": ["a", "b"], "sampling_params": [{"max_tokens": 1}]}}
        with pytest.raises(ValueError, match="1 items but there are 2 prompts"):
            batch.read_generate_spec(spec)

    def test_rejects_a_non_boolean_strict(self):
        with pytest.raises(ValueError, match="strict must be a boolean"):
            batch.read_generate_spec({"payload": {"prompts": ["a"], "strict": "yes"}})

    def test_poll_defaults_to_true(self):
        assert batch.should_poll({"payload": {}}) is True
        assert batch.should_poll({"poll": False}) is False
