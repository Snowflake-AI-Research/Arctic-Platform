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

"""The two sides\' extraction and the one reduction they share, without a GPU.

Both engines are replaced by an array whose entry at logit position ``s`` is ``s`` itself. Any
misalignment then shows up as the wrong integers rather than as a small numerical difference, which is the
failure mode the check exists to catch and the one a real run cannot distinguish from a model difference.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import inference_checkpoint_loss as check
from arctic_platform.correctness.harness import gsm8k
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome

# Two rows of different length so padding is present and the shorter row's tail is never scored.
ROWS = [
    gsm8k.Example(
        question="q0", prompt="Question: q0\nAnswer:", completion=" a0", input_ids=(11, 12, 13, 14, 15, 16), n_prompt=3
    ),
    gsm8k.Example(
        question="q1", prompt="Question: q1\nAnswer:", completion=" a1", input_ids=(21, 22, 23, 24), n_prompt=2
    ),
]


def positional_logprobs_tensor(rows, width: int) -> torch.Tensor:
    """What the forward-only route returns if the log-probability at logit position ``s`` were ``s``."""
    return torch.arange(width, dtype=torch.float32).repeat(len(rows), 1)


def positional_log_prob_results(rows):
    """What the sampling zone returns for the same model: position 0 absent, entry ``k`` scoring ``k + 1``."""
    return [
        {
            "token_ids": list(row.input_ids[1:]),
            "logprobs": [float(k) for k in range(row.length - 1)],
            "seq_len": row.length - 1,
        }
        for row in rows
    ]


def test_exported_checkpoint_reuses_the_validated_tokenizer(tmp_path) -> None:
    class Tokenizer:
        def __init__(self):
            self.saved_to = None

        def save_pretrained(self, path):
            self.saved_to = path

    tokenizer = Tokenizer()
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()

    check.materialize_checkpoint_tokenizer(tokenizer, checkpoint)

    assert tokenizer.saved_to == checkpoint


def test_correctness_sampling_disables_cuda_graphs() -> None:
    from arctic_platform.correctness.harness.dss_driver import sampling_payload

    payload = sampling_payload("/checkpoint", 42, dtype="bfloat16", n_gpus=1, max_seq_len=4096)

    extra = payload["inference_config"]["vllm_config"]
    assert extra["enforce_eager"] is True
    assert extra["disable_custom_all_reduce"] is True


def test_forward_request_asks_for_per_token_logprobs() -> None:
    batch = gsm8k.build_batch("validation", ROWS, pad_id=0)

    body = check.pack_logit_aligned(batch)

    assert body["processing"]["return_logprobs"] is True
    assert body["meta"]["labels_are_shifted"] is True


def test_forward_request_attends_to_prompts_and_masks_only_padding() -> None:
    batch = gsm8k.build_batch("validation", ROWS, pad_id=0)

    attention_mask = check.pack_logit_aligned(batch)["batch"]["attention_mask"]

    for index, row in enumerate(ROWS):
        assert attention_mask[index, : row.length].tolist() == [1] * row.length
        assert attention_mask[index, row.length :].tolist() == [0] * (attention_mask.shape[1] - row.length)


def test_the_reduction_is_the_negative_mean_over_every_answer_token() -> None:
    value, count = check.cross_entropy([[-1.0, -2.0], [-3.0]])

    assert count == 3
    assert value == pytest.approx(2.0)


def test_the_reduction_pools_across_rows_rather_than_averaging_row_means() -> None:
    """A per-row mean would weight a one-token answer as heavily as a hundred-token one."""
    pooled, _ = check.cross_entropy([[-1.0, -1.0, -1.0, -1.0], [-5.0]])

    assert pooled == pytest.approx(1.8)


def test_the_reduction_refuses_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="not a measurement"):
        check.cross_entropy([[], []])


def test_the_training_side_reads_the_logits_that_predict_the_answer_tokens() -> None:
    """Answer tokens of a row of length L beginning at p are predicted by logits p-1 through L-2."""
    width = max(row.length for row in ROWS)

    spans = check.reference_answer_logprobs(positional_logprobs_tensor(ROWS, width), ROWS)

    assert spans == [[2.0, 3.0, 4.0], [1.0, 2.0]]
    assert [len(span) for span in spans] == [row.answer_tokens for row in ROWS]


def test_the_sampling_side_reads_the_same_logit_positions() -> None:
    """The arrays differ in length by the position vLLM does not report, and the slice is the same."""
    spans = check.target_answer_logprobs(positional_log_prob_results(ROWS), ROWS)

    assert spans == [[2.0, 3.0, 4.0], [1.0, 2.0]]


def test_the_two_sides_agree_when_the_model_does() -> None:
    """The whole point: one set of per-position log-probabilities reduces to one number on both paths."""
    width = max(row.length for row in ROWS)

    reference, _ = check.cross_entropy(check.reference_answer_logprobs(positional_logprobs_tensor(ROWS, width), ROWS))
    target, _ = check.cross_entropy(check.target_answer_logprobs(positional_log_prob_results(ROWS), ROWS))

    assert reference == target


def test_a_row_carrying_a_logprob_for_position_zero_is_refused() -> None:
    """An entry per position, rather than one per position after the first, shifts every answer token."""
    results = positional_log_prob_results(ROWS)
    results[0]["logprobs"] = [0.0] + results[0]["logprobs"]

    with pytest.raises(RuntimeError, match="one per position after the first"):
        check.target_answer_logprobs(results, ROWS)


def test_a_row_scored_against_other_tokens_is_refused() -> None:
    """What a ranked ``top_k`` or a mismatched tokenizer in the served tree produces."""
    results = positional_log_prob_results(ROWS)
    results[1]["token_ids"] = [99, 99, 99]

    with pytest.raises(RuntimeError, match="token stream"):
        check.target_answer_logprobs(results, ROWS)


def test_unshifted_labels_would_score_the_wrong_span() -> None:
    """Why the forward-only request is packed with shifted labels.

    ``pack`` sends HuggingFace-convention labels for a ``huggingface`` provider because ``/fwd-bwd``
    shifts them at dispatch. The forward-only route does not, so those labels would make the route report
    the log-probability of the token AT each position instead of the one after it -- the span below,
    shifted by one, which is a different measurement rather than a noisier one.
    """
    batch = gsm8k.build_batch("validation", ROWS, pad_id=0)

    shifted = batch.shifted_labels()
    row, example = 0, ROWS[0]
    span = slice(example.n_prompt - 1, example.length - 1)
    assert shifted[row, span].tolist() == list(example.input_ids[example.n_prompt :])
    assert batch.labels[row, span].tolist() != list(example.input_ids[example.n_prompt :])


def test_the_steps_consume_the_training_slice_once_each() -> None:
    rows_per_step = 2
    examples = [
        gsm8k.Example(
            question=f"q{index}", prompt="p", completion=" a", input_ids=(index + 1, index + 2, index + 3), n_prompt=1
        )
        for index in range(check.TRAIN_STEPS * rows_per_step)
    ]

    batches = check.training_batches(examples, rows_per_step, pad_id=0)

    assert len(batches) == check.TRAIN_STEPS
    assert all(batch.rows == rows_per_step for batch in batches)
    seen = [tuple(row) for batch in batches for row in batch.input_ids.tolist()]
    assert len(set(seen)) == len(examples)


def test_agreement_inside_the_gate_passes_and_outside_it_fails() -> None:
    context = SimpleNamespace(config_id="example", cfg=SimpleNamespace(gpu_type="h200"))

    inside = check.verdict(context, 1.750000, 1.750500)
    outside = check.verdict(context, 1.750000, 1.753000)

    assert inside.outcome is Outcome.PASS
    assert inside.mismatches == []
    assert inside.metrics["stated_criterion_abs"] == 1e-3
    assert outside.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in outside.mismatches] == [check.QUANTITY]
    assert outside.metrics["reference_cross_entropy"] == 1.750000
    assert outside.metrics["target_cross_entropy"] == 1.753000
    assert outside.metrics["max_abs_diff"] == pytest.approx(3e-3)


def test_b200_uses_three_milliabsolute_gate() -> None:
    context = SimpleNamespace(config_id="b200-example", cfg=SimpleNamespace(gpu_type="b200"))

    result = check.verdict(context, 4.299332, 4.297708)

    assert result.outcome is Outcome.PASS
    assert result.metrics["stated_criterion_abs"] == 3e-3
    assert result.metrics["max_abs_diff"] == pytest.approx(1.624e-3)


def test_forward_loss_reads_the_ap_response_contract() -> None:
    from arctic_platform.correctness.harness.dss_driver import forward_loss

    client = SimpleNamespace(fwd_no_grad=lambda body: {"avg_loss": [1.25]})
    job = SimpleNamespace(client=client)

    assert forward_loss(None, job, {"batch": {}}) == pytest.approx(1.25)
