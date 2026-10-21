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

"""Static correctness selftests: inference rl."""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.utils import hub

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.common.utils.batch import unpack_batch
from arctic_platform.correctness.checks import inference_checkpoint_loss as check
from arctic_platform.correctness.checks import rl_router_replay
from arctic_platform.correctness.checks import rl_weight_sync
from arctic_platform.correctness.harness import dss_driver as rl_driver
from arctic_platform.correctness.harness import inference_export as export
from arctic_platform.correctness.harness import inference_export as projection
from arctic_platform.correctness.harness import router_replay_trace
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness import sft as gsm8k
from arctic_platform.correctness.harness.batches import SEED
from arctic_platform.correctness.harness.config import TestSpec as Spec
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.dss_driver import _inference_client_config
from arctic_platform.correctness.harness.dss_driver import rl_batch
from arctic_platform.correctness.harness.dss_driver import rl_gpu_count
from arctic_platform.correctness.harness.dss_driver import rl_sampling_payload
from arctic_platform.correctness.harness.dss_driver import sampling_payload
from arctic_platform.correctness.harness.inference_export import FAMILIES
from arctic_platform.correctness.harness.registry import TestOutcome
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import reference_tests
from arctic_platform.correctness.harness.registry import registered_tests

from .helpers import _config_factory_native_config

# Inference checkpoint loss.


_inference_checkpoint_loss_ROWS = [
    gsm8k.Example(
        question="q0", prompt="Question: q0\nAnswer:", completion=" a0", input_ids=(11, 12, 13, 14, 15, 16), n_prompt=3
    ),
    gsm8k.Example(
        question="q1", prompt="Question: q1\nAnswer:", completion=" a1", input_ids=(21, 22, 23, 24), n_prompt=2
    ),
]


def _inference_checkpoint_loss_positional_logprobs_tensor(rows, width: int) -> torch.Tensor:
    """What the forward-only route returns if the log-probability at logit position ``s`` were ``s``."""
    return torch.arange(width, dtype=torch.float32).repeat(len(rows), 1)


def _inference_checkpoint_loss_positional_log_prob_results(rows):
    """What the sampling zone returns for the same model: position 0 absent, entry ``k`` scoring ``k + 1``."""
    return [
        {
            "token_ids": list(row.input_ids[1:]),
            "logprobs": [float(k) for k in range(row.length - 1)],
            "seq_len": row.length - 1,
        }
        for row in rows
    ]


def test_inference_checkpoint_loss_exported_checkpoint_reuses_the_validated_tokenizer(tmp_path) -> None:

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


def test_inference_checkpoint_loss_correctness_sampling_disables_cuda_graphs() -> None:
    from arctic_platform.correctness.harness.dss_driver import sampling_payload

    payload = sampling_payload("/checkpoint", 42, dtype="bfloat16", n_gpus=1, max_seq_len=4096)
    extra = payload["inference_config"]["vllm_config"]
    assert extra["enforce_eager"] is True
    assert extra["disable_custom_all_reduce"] is True


def test_inference_checkpoint_loss_forward_request_asks_for_per_token_logprobs() -> None:
    batch = gsm8k.build_batch("validation", _inference_checkpoint_loss_ROWS, pad_id=0)
    body = check.pack_logit_aligned(batch)
    assert body["processing"]["return_logprobs"] is True
    assert body["meta"]["labels_are_shifted"] is True


def test_inference_checkpoint_loss_forward_request_attends_to_prompts_and_masks_only_padding() -> None:
    batch = gsm8k.build_batch("validation", _inference_checkpoint_loss_ROWS, pad_id=0)
    attention_mask = check.pack_logit_aligned(batch)["batch"]["attention_mask"]
    for index, row in enumerate(_inference_checkpoint_loss_ROWS):
        assert attention_mask[index, : row.length].tolist() == [1] * row.length
        assert attention_mask[index, row.length :].tolist() == [0] * (attention_mask.shape[1] - row.length)


def test_inference_checkpoint_loss_the_reduction_is_the_negative_mean_over_every_answer_token() -> None:
    value, count = check.cross_entropy([[-1.0, -2.0], [-3.0]])
    assert count == 3
    assert value == pytest.approx(2.0)


def test_inference_checkpoint_loss_the_reduction_pools_across_rows_rather_than_averaging_row_means() -> None:
    """A per-row mean would weight a one-token answer as heavily as a hundred-token one."""
    pooled, _ = check.cross_entropy([[-1.0, -1.0, -1.0, -1.0], [-5.0]])
    assert pooled == pytest.approx(1.8)


def test_inference_checkpoint_loss_the_reduction_refuses_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="not a measurement"):
        check.cross_entropy([[], []])


def test_inference_checkpoint_loss_the_training_side_reads_the_logits_that_predict_the_answer_tokens() -> None:
    """Answer tokens of a row of length L beginning at p are predicted by logits p-1 through L-2."""
    width = max((row.length for row in _inference_checkpoint_loss_ROWS))
    spans = check.reference_answer_logprobs(
        _inference_checkpoint_loss_positional_logprobs_tensor(_inference_checkpoint_loss_ROWS, width),
        _inference_checkpoint_loss_ROWS,
    )
    assert spans == [[2.0, 3.0, 4.0], [1.0, 2.0]]
    assert [len(span) for span in spans] == [row.answer_tokens for row in _inference_checkpoint_loss_ROWS]


def test_inference_checkpoint_loss_the_sampling_side_reads_the_same_logit_positions() -> None:
    """The arrays differ in length by the position vLLM does not report, and the slice is the same."""
    spans = check.target_answer_logprobs(
        _inference_checkpoint_loss_positional_log_prob_results(_inference_checkpoint_loss_ROWS),
        _inference_checkpoint_loss_ROWS,
    )
    assert spans == [[2.0, 3.0, 4.0], [1.0, 2.0]]


def test_inference_checkpoint_loss_the_two_sides_agree_when_the_model_does() -> None:
    """The whole point: one set of per-position log-probabilities reduces to one number on both paths."""
    width = max((row.length for row in _inference_checkpoint_loss_ROWS))
    reference, _ = check.cross_entropy(
        check.reference_answer_logprobs(
            _inference_checkpoint_loss_positional_logprobs_tensor(_inference_checkpoint_loss_ROWS, width),
            _inference_checkpoint_loss_ROWS,
        )
    )
    target, _ = check.cross_entropy(
        check.target_answer_logprobs(
            _inference_checkpoint_loss_positional_log_prob_results(_inference_checkpoint_loss_ROWS),
            _inference_checkpoint_loss_ROWS,
        )
    )
    assert reference == target


def test_inference_checkpoint_loss_a_row_carrying_a_logprob_for_position_zero_is_refused() -> None:
    """An entry per position, rather than one per position after the first, shifts every answer token."""
    results = _inference_checkpoint_loss_positional_log_prob_results(_inference_checkpoint_loss_ROWS)
    results[0]["logprobs"] = [0.0] + results[0]["logprobs"]
    with pytest.raises(RuntimeError, match="one per position after the first"):
        check.target_answer_logprobs(results, _inference_checkpoint_loss_ROWS)


def test_inference_checkpoint_loss_a_row_scored_against_other_tokens_is_refused() -> None:
    """What a ranked ``top_k`` or a mismatched tokenizer in the served tree produces."""
    results = _inference_checkpoint_loss_positional_log_prob_results(_inference_checkpoint_loss_ROWS)
    results[1]["token_ids"] = [99, 99, 99]
    with pytest.raises(RuntimeError, match="token stream"):
        check.target_answer_logprobs(results, _inference_checkpoint_loss_ROWS)


def test_inference_checkpoint_loss_unshifted_labels_would_score_the_wrong_span() -> None:
    """Why the forward-only request is packed with shifted labels.

    ``pack`` sends HuggingFace-convention labels for a ``huggingface`` provider because ``/fwd-bwd``
    shifts them at dispatch. The forward-only route does not, so those labels would make the route report
    the log-probability of the token AT each position instead of the one after it -- the span below,
    shifted by one, which is a different measurement rather than a noisier one.
    """
    batch = gsm8k.build_batch("validation", _inference_checkpoint_loss_ROWS, pad_id=0)
    shifted = batch.shifted_labels()
    row, example = (0, _inference_checkpoint_loss_ROWS[0])
    span = slice(example.n_prompt - 1, example.length - 1)
    assert shifted[row, span].tolist() == list(example.input_ids[example.n_prompt :])
    assert batch.labels[row, span].tolist() != list(example.input_ids[example.n_prompt :])


def test_inference_checkpoint_loss_the_steps_consume_the_training_slice_once_each() -> None:
    rows_per_step = 2
    examples = [
        gsm8k.Example(
            question=f"q{index}", prompt="p", completion=" a", input_ids=(index + 1, index + 2, index + 3), n_prompt=1
        )
        for index in range(check.TRAIN_STEPS * rows_per_step)
    ]
    batches = check.training_batches(examples, rows_per_step, pad_id=0)
    assert len(batches) == check.TRAIN_STEPS
    assert all((batch.rows == rows_per_step for batch in batches))
    seen = [tuple(row) for batch in batches for row in batch.input_ids.tolist()]
    assert len(set(seen)) == len(examples)


@pytest.mark.parametrize("composite,ngram,kernel,expected", [(True, 3, 4, 144), (False, 2, 7, 192), (True, 5, 1, 64)])
def test_inference_ple_padding_floor_uses_effective_offline_config(monkeypatch, composite, ngram, kernel, expected):
    text = SimpleNamespace(
        model_type="qwen4_exp_text", num_hidden_layers=48, ngram_size=ngram, ple_conv_kernel_size=kernel
    )
    source = SimpleNamespace(model_type="qwen4_exp", text_config=text) if composite else text
    cfg = SimpleNamespace(config_id=check._FLASHNEXT_CONFIG_ID, sp_size=16, training={"model_provider": "prime_rl"})
    calls = []

    def offline(path, **kwargs):
        calls.append((path, kwargs))
        return source

    monkeypatch.setattr(AutoConfig, "from_pretrained", offline)
    assert check._minimum_context_parallel_row_tokens(SimpleNamespace(cfg=cfg), "/actual/local/model") == expected
    assert calls == [("/actual/local/model", {"local_files_only": True})]


@pytest.mark.parametrize(
    "field,value",
    [
        ("ngram_size", True),
        ("ngram_size", 0),
        ("ngram_size", None),
        ("ple_conv_kernel_size", -1),
        ("ple_conv_kernel_size", 4.0),
        ("num_hidden_layers", 4),
        ("model_type", "qwen3"),
    ],
)
def test_inference_ple_padding_refuses_missing_invalid_or_reduced_config(monkeypatch, field, value):
    text = SimpleNamespace(model_type="qwen4_exp_text", num_hidden_layers=48, ngram_size=3, ple_conv_kernel_size=4)
    setattr(text, field, value)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: text)
    cfg = SimpleNamespace(config_id=check._FLASHNEXT_CONFIG_ID, sp_size=16, training={"model_provider": "prime_rl"})
    with pytest.raises(ValueError, match="context-parallel inference requires"):
        check._minimum_context_parallel_row_tokens(SimpleNamespace(cfg=cfg), "/actual/local/model")


@pytest.mark.parametrize(
    "config_id,provider,sp",
    [
        ("other", "prime_rl", 16),
        (check._FLASHNEXT_CONFIG_ID, "huggingface", 16),
        (check._FLASHNEXT_CONFIG_ID, "prime_rl", 1),
    ],
)
def test_inference_ple_padding_leaves_unaffected_routes_without_config_reads(monkeypatch, config_id, provider, sp):
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *args, **kwargs: pytest.fail("unaffected route config read")
    )
    cfg = SimpleNamespace(config_id=config_id, sp_size=sp, training={"model_provider": provider})
    assert check._minimum_context_parallel_row_tokens(SimpleNamespace(cfg=cfg), "unused") == 0
    sentinel = object()
    assert check._extend_short_causal_rows(sentinel, 0, 0) is sentinel


@pytest.mark.parametrize("long_row", [False, True])
@pytest.mark.parametrize("shifted", [False, True])
def test_inference_ple_padding_survives_real_pack_and_all_sequence_shards(long_row, shifted):
    from arctic_platform.common.packing.pack import IGNORE_INDEX
    from arctic_platform.common.packing.pack import pack_microbatch
    from arctic_platform.common.packing.pack import pad_packed_microbatch
    from arctic_platform.common.packing.windows import cu_seqlens_from_position_ids
    from arctic_platform.common.packing.windows import token_shard

    rows = list(_inference_checkpoint_loss_ROWS)
    if long_row:
        rows[0] = replace(rows[0], input_ids=tuple(range(1, 182)))
    batch = gsm8k.build_batch("validation", rows, pad_id=0)
    original = copy.deepcopy(batch)
    padded = check._extend_short_causal_rows(batch, 144, pad_id=0)
    assert padded.name == batch.name and padded.rows == batch.rows
    assert padded.active_tokens == batch.active_tokens
    for field in ("input_ids", "position_ids", "labels"):
        assert torch.equal(getattr(batch, field), getattr(original, field))
    for index, row in enumerate(rows):
        for field in ("input_ids", "position_ids", "labels"):
            assert torch.equal(getattr(padded, field)[index, : row.length], getattr(batch, field)[index, : row.length])
        assert padded.labels[index, row.length :].eq(IGNORE_INDEX).all()
        assert padded.input_ids[index, row.length :].eq(0).all()
        assert padded.position_ids[index, : max(row.length, 144)].tolist() == list(range(max(row.length, 144)))

    body = check.pack_logit_aligned(padded) if shifted else check.pack(padded, model_provider="huggingface")
    original_body = check.pack_logit_aligned(batch) if shifted else check.pack(batch, model_provider="huggingface")
    # A token-budget group may contain just the shortest row, even in a wider ragged batch.
    for indices in ([0], [1], [0, 1]):
        group = {key: value[indices] for key, value in body["batch"].items()}
        packed, metadata = pack_microbatch(group)
        assert metadata.valid_lengths.tolist() == [max(rows[index].length, 144) for index in indices]
        expected_labels = original_body["batch"]["labels"][indices]
        assert torch.equal(
            packed["labels"][packed["labels"] != IGNORE_INDEX], expected_labels[expected_labels != IGNORE_INDEX]
        )
        assert cu_seqlens_from_position_ids(packed["position_ids"]).tolist() == metadata.cu_seqlens.tolist()
        packed, metadata = pad_packed_microbatch(packed, metadata, 16)
        shards = [token_shard(packed, metadata, 16, rank) for rank in range(16)]
        assert all(shard["input_ids"].shape[1] >= 9 for shard in shards)
        for field in ("input_ids", "position_ids", "labels"):
            assert torch.equal(torch.cat([shard[field] for shard in shards], dim=1), packed[field])


def test_inference_ple_padding_already_long_rows_are_identity_noop():
    rows = [replace(row, input_ids=tuple(range(1, 162))) for row in _inference_checkpoint_loss_ROWS]
    batch = gsm8k.build_batch("train", rows, pad_id=0)
    assert check._extend_short_causal_rows(batch, 144, 0) is batch


@pytest.mark.parametrize("token_budget", [128, 256])
def test_inference_ple_padding_real_run_packs_training_and_validation_without_reselecting(
    tmp_path, monkeypatch, token_budget
):
    train = [replace(_inference_checkpoint_loss_ROWS[0], question=f"train{index}") for index in range(10)]
    validation = [_inference_checkpoint_loss_ROWS[1]]
    cfg = SimpleNamespace(
        config_id=check._FLASHNEXT_CONFIG_ID,
        model_name="/actual/local/model",
        n_gpus=16,
        sp_size=16,
        dp_size=1,
        max_tokens_per_mb=token_budget,
        sub_job={},
        training={"model_provider": "prime_rl", "train_batch_size": 1, "optimizer": {"lr": 2e-5}},
    )
    ctx = SimpleNamespace(cfg=cfg, workdir=tmp_path, attn_implementation="qsa_flex")
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: SimpleNamespace(pad_token_id=0, eos_token_id=0))
    monkeypatch.setattr(
        check.gsm8k, "read_rows", lambda split, **kwargs: train if split == check.TRAIN_SPLIT else validation
    )
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda tokenizer, rows: rows)
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: (validation, 4))
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="qwen4_exp_text", num_hidden_layers=48, ngram_size=3, ple_conv_kernel_size=4
        ),
    )
    monkeypatch.setattr(check, "build_payload", lambda *args, **kwargs: {})
    real_pack, real_validation_pack = check.pack, check.pack_logit_aligned
    packed = []

    def training_body(batch, **kwargs):
        packed.append(batch)
        return real_pack(batch, **kwargs)

    def validation_body(batch):
        packed.append(batch)
        return real_validation_pack(batch)

    monkeypatch.setattr(check, "pack", training_body)
    monkeypatch.setattr(check, "pack_logit_aligned", validation_body)

    def no_gateway(*args, **kwargs):
        raise RuntimeError("stop before network")

    monkeypatch.setattr(check, "gateway", no_gateway)
    if token_budget == 128:
        with pytest.raises(ValueError, match="unchanged token budget; rows cannot be dropped"):
            check._run(ctx)
        assert packed == []
    else:
        with pytest.raises(RuntimeError, match="stop before network"):
            check._run(ctx)
        assert len(packed) == check.TRAIN_STEPS + 1
        for batch, row in zip(packed, validation + train):
            assert batch.rows == 1 and batch.seq_len == 144
            assert batch.input_ids[0, : row.length].tolist() == list(row.input_ids)
            assert batch.active_tokens == row.answer_tokens


def test_inference_checkpoint_loss_agreement_inside_the_gate_passes_and_outside_it_fails() -> None:
    context = SimpleNamespace(config_id="example", cfg=SimpleNamespace(gpu_type="h200"))
    inside = check.verdict(context, 1.75, 1.7505)
    outside = check.verdict(context, 1.75, 1.753)
    assert inside.outcome is Outcome.PASS
    assert inside.mismatches == []
    assert inside.metrics["stated_criterion_abs"] == 0.001
    assert outside.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in outside.mismatches] == [check.QUANTITY]
    assert outside.metrics["reference_cross_entropy"] == 1.75
    assert outside.metrics["target_cross_entropy"] == 1.753
    assert outside.metrics["max_abs_diff"] == pytest.approx(0.003)


def test_inference_checkpoint_loss_b200_uses_three_milliabsolute_gate() -> None:
    context = SimpleNamespace(config_id="b200-example", cfg=SimpleNamespace(gpu_type="b200"))
    result = check.verdict(context, 4.299332, 4.297708)
    assert result.outcome is Outcome.PASS
    assert result.metrics["stated_criterion_abs"] == 0.003
    assert result.metrics["max_abs_diff"] == pytest.approx(0.001624)


def test_inference_checkpoint_loss_forward_loss_reads_the_ap_response_contract() -> None:
    from arctic_platform.correctness.harness.dss_driver import forward_loss

    client = SimpleNamespace(fwd_no_grad=lambda body: {"avg_loss": [1.25]})
    job = SimpleNamespace(client=client)
    assert forward_loss(None, job, {"batch": {}}) == pytest.approx(1.25)


def _inference_checkpoint_loss_generate_results():
    return [
        {
            "prompt_len": row.length,
            "prompt_logprobs": [
                None,
                *[
                    {999: {"logprob": -99.0}, str(token): {"logprob": -float(position)}}
                    for position, token in enumerate(row.input_ids[1:])
                ],
            ],
            "token_ids": [999],
        }
        for row in _inference_checkpoint_loss_ROWS
    ]


def test_inference_checkpoint_loss_cortex_selects_expected_prompt_tokens_not_first_dictionary_entries():
    results = _inference_checkpoint_loss_generate_results()
    results[0]["prompt_logprobs"][1] = {999: {"logprob": -99.0}, _inference_checkpoint_loss_ROWS[0].input_ids[1]: -0.0}
    normalized = check.cortex_prompt_logprobs(results, _inference_checkpoint_loss_ROWS)
    assert [row["token_ids"] for row in normalized] == [
        list(row.input_ids[1:]) for row in _inference_checkpoint_loss_ROWS
    ]
    assert check.target_answer_logprobs(normalized, _inference_checkpoint_loss_ROWS) == [
        [-2.0, -3.0, -4.0],
        [-1.0, -2.0],
    ]


@pytest.mark.parametrize(
    "results",
    [
        None,
        {},
        [],
        _inference_checkpoint_loss_generate_results()[:1],
        _inference_checkpoint_loss_generate_results() + [{}],
    ],
)
def test_inference_checkpoint_loss_cortex_rejects_wrong_generate_row_counts(results):
    with pytest.raises(RuntimeError, match="one row per validation example"):
        check.cortex_prompt_logprobs(results, _inference_checkpoint_loss_ROWS)


@pytest.mark.parametrize(
    "change",
    [
        "non_dict_row",
        "missing_positions",
        "short_positions",
        "long_positions",
        "initial_dict",
        "interior_none",
        "empty_position",
        "missing_token",
        "duplicate_token",
        "wrong_prompt_length",
        "missing_logprob",
        "nan",
        "inf",
        "bool",
        "text",
    ],
)
def test_inference_checkpoint_loss_cortex_rejects_malformed_prompt_logprobs(change):
    results = _inference_checkpoint_loss_generate_results()
    row = results[0]
    token = _inference_checkpoint_loss_ROWS[0].input_ids[1]
    if change == "non_dict_row":
        results[0] = None
    elif change == "missing_positions":
        del row["prompt_logprobs"]
    elif change == "short_positions":
        row["prompt_logprobs"].pop()
    elif change == "long_positions":
        row["prompt_logprobs"].append({token: -1.0})
    elif change == "initial_dict":
        row["prompt_logprobs"][0] = {}
    elif change == "wrong_prompt_length":
        row["prompt_len"] += 1
    else:
        values = {
            "interior_none": None,
            "empty_position": {},
            "missing_token": {999: {"logprob": -1.0}},
            "duplicate_token": {token: -1.0, str(token): -1.0},
            "missing_logprob": {token: {"rank": 1}},
            "nan": {token: {"logprob": float("nan")}},
            "inf": {token: {"logprob": float("-inf")}},
            "bool": {token: {"logprob": True}},
            "text": {token: {"logprob": "-1.0"}},
        }
        row["prompt_logprobs"][1] = values[change]
    with pytest.raises(RuntimeError):
        check.cortex_prompt_logprobs(results, _inference_checkpoint_loss_ROWS)


@pytest.mark.parametrize(
    "boundary", [0, -1, _inference_checkpoint_loss_ROWS[0].length, _inference_checkpoint_loss_ROWS[0].length + 1]
)
def test_inference_checkpoint_loss_cortex_rejects_invalid_answer_boundaries(boundary):
    examples = [replace(_inference_checkpoint_loss_ROWS[0], n_prompt=boundary), _inference_checkpoint_loss_ROWS[1]]
    with pytest.raises(RuntimeError, match="answer span"):
        check.cortex_prompt_logprobs(_inference_checkpoint_loss_generate_results(), examples)


@pytest.mark.parametrize(
    "results",
    [
        [],
        _inference_checkpoint_loss_positional_log_prob_results(_inference_checkpoint_loss_ROWS)[:1],
        _inference_checkpoint_loss_positional_log_prob_results(_inference_checkpoint_loss_ROWS) + [{}],
    ],
)
def test_inference_checkpoint_loss_target_extraction_does_not_silently_truncate_rows(results):
    with pytest.raises(RuntimeError, match="validation examples"):
        check.target_answer_logprobs(results, _inference_checkpoint_loss_ROWS)


@pytest.mark.parametrize("shape", [(1, 6), (3, 6), (2, 6, 1)])
def test_inference_checkpoint_loss_reference_extraction_requires_exact_rows_and_rank(shape):
    with pytest.raises(RuntimeError, match="two-dimensional row"):
        check.reference_answer_logprobs(torch.zeros(shape), _inference_checkpoint_loss_ROWS)


def test_inference_checkpoint_loss_nonfinite_answer_scores_fail_instead_of_passing_the_gate():
    values = _inference_checkpoint_loss_positional_logprobs_tensor(_inference_checkpoint_loss_ROWS, 6)
    values[0, _inference_checkpoint_loss_ROWS[0].n_prompt] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        check.reference_answer_logprobs(values, _inference_checkpoint_loss_ROWS)
    results = _inference_checkpoint_loss_positional_log_prob_results(_inference_checkpoint_loss_ROWS)
    results[0]["logprobs"][0] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        check.target_answer_logprobs(results, _inference_checkpoint_loss_ROWS)


@pytest.mark.parametrize("model_name", [None, "", "   ", "/node/local/model"])
def test_inference_checkpoint_loss_cortex_requires_full_configured_model_before_loading_tokenizer(
    model_name, tmp_path, monkeypatch
):
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda *args: pytest.fail("must reject before tokenizer load"))
    ctx = SimpleNamespace(
        client_backend=CortexConfig(base_url="http://unused"),
        cfg=SimpleNamespace(model_name=model_name),
        target_model_name="published-reduced-fixture",
        spec=SimpleNamespace(model=SimpleNamespace(cache_path="/local/reference")),
    )
    with pytest.raises(ValueError, match="full configured model"):
        check._run(ctx)


# Inference export.


@pytest.fixture(params=["qwen3_5", "qwen3_5_moe"])
def _inference_export_case(request):
    family = request.param
    original_class, text_type, exported_class = projection.FAMILIES[family]
    text = {
        "model_type": text_type,
        "num_hidden_layers": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "use_cache": True,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1,
    }
    if family.endswith("_moe"):
        text["num_experts"] = 4
        text["num_experts_per_tok"] = 2
    source = {
        "model_type": family,
        "architectures": [original_class],
        "text_config": text,
        "vision_config": {"depth": 1},
    }
    exported = {
        **copy.deepcopy(text),
        "architectures": [exported_class],
        "transformers_version": "synthetic-only",
        "use_cache": False,
    }
    schema = {
        "model.language_model.layers.0.norm.weight": ["F32", [4]],
        "model.language_model.layers.1.norm.weight": ["BF16", [4]],
        "model.language_model.embed_tokens.weight": ["BF16", [8, 4]],
        "lm_head.weight": ["BF16", [8, 4]],
        "model.visual.block.weight": ["BF16", [4, 4]],
        "mtp.fc.weight": ["BF16", [4, 8]],
    }
    if family.endswith("_moe"):
        schema["model.language_model.layers.0.mlp.experts.gate_up_proj"] = ["BF16", [4, 16, 4]]
    output = {
        name: ["BF16", list(row[1])]
        for name, row in schema.items()
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    return (source, exported, schema, output)


def test_inference_export_named_projection_uses_original_not_job12_dimensions(_inference_export_case):
    original = copy.deepcopy(_inference_export_case)
    receipt = projection.validate_text_projection(*_inference_export_case)
    assert receipt["text_tensor_count"] == len(_inference_export_case[3])
    assert receipt["source_tensor_count"] == len(_inference_export_case[2])
    assert receipt["excluded_vision_tensor_count"] == 1
    assert receipt["excluded_mtp_tensor_count"] == 1
    assert receipt["text_layer_count"] == 2
    assert receipt["runtime_admitted"] is False
    assert receipt["tensor_payloads_read"] is False
    assert receipt["numerical_pass"] is None
    assert _inference_export_case == original


@pytest.mark.parametrize("family", ["qwen3", "qwen3_moe", "qwen3_5_text", "qwen3_5_moe_text"])
def test_inference_export_ordinary_family_delegates_unchanged(family):
    assert projection.validate_text_projection({"model_type": family}, {}, None, None) is None


def test_inference_export_unchanged_composite_delegates_without_mutation(_inference_export_case):
    source = _inference_export_case[0]
    exported = copy.deepcopy(source)
    original = copy.deepcopy((source, exported))
    assert projection.validate_text_projection(source, exported, None, None) is None
    assert (source, exported) == original


def test_inference_export_scalar_text_tensor_retains_exact_shape_and_bytes(_inference_export_case):
    source, exported, schema, output = _inference_export_case
    name = "model.language_model.layers.0.scale"
    schema[name] = ["F32", []]
    output[name] = ["BF16", []]
    receipt = projection.validate_text_projection(source, exported, schema, output)
    assert receipt["export_payload_bytes"] == projection.schema_bytes(output)
    assert projection.schema_bytes({name: ["BF16", []]}) == 2
    output[name][1] = [1]
    with pytest.raises(ValueError, match="shapes"):
        projection.validate_text_projection(source, exported, schema, output)


def test_inference_export_unknown_projection_rejected(_inference_export_case):
    source, exported, schema, output = _inference_export_case
    source["model_type"] = "unknown_composite"
    with pytest.raises(ValueError, match="unknown"):
        projection.validate_text_projection(source, exported, schema, output)


@pytest.mark.parametrize(
    "failure",
    [
        "source-class",
        "export-class",
        "text-family",
        "reduced-layers",
        "reduced-width",
        "remote-code",
        "rope-drift",
        "added-field",
        "missing-field",
        "bool-dimension",
        "cache-type",
        "nonfinite-config",
    ],
)
def test_inference_export_configuration_fail_closed(_inference_export_case, failure):
    source, exported, _, _ = _inference_export_case
    if failure == "source-class":
        source["architectures"] = ["UnrelatedForConditionalGeneration"]
    elif failure == "export-class":
        exported["architectures"] = source["architectures"]
    elif failure == "text-family":
        exported["model_type"] = "qwen3"
    elif failure == "reduced-layers":
        exported["num_hidden_layers"] = 1
    elif failure == "reduced-width":
        exported["hidden_size"] = 2
    elif failure == "remote-code":
        exported["auto_map"] = {"AutoModel": "untrusted.Model"}
    elif failure == "rope-drift":
        exported["rope_parameters"]["rope_theta"] = 10000
    elif failure == "added-field":
        exported["unreviewed_switch"] = True
    elif failure == "missing-field":
        exported.pop("linear_conv_kernel_dim")
    elif failure == "bool-dimension":
        exported["num_key_value_heads"] = True
    elif failure == "cache-type":
        exported["use_cache"] = 1
    else:
        exported["rope_parameters"]["rope_theta"] = float("nan")
    with pytest.raises(ValueError):
        projection.validate_text_projection(*_inference_export_case)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-text",
        "extra-text",
        "vision-export",
        "mtp-export",
        "rename",
        "same-byte-shape",
        "fp32",
        "bool-shape",
        "unknown-source",
        "source-layer-missing",
    ],
)
def test_inference_export_header_schema_coverage_fail_closed(_inference_export_case, failure):
    _, _, source, exported = _inference_export_case
    name = "model.language_model.layers.1.norm.weight"
    if failure == "missing-text":
        exported.pop(name)
    elif failure == "extra-text":
        exported["model.language_model.extra.weight"] = ["BF16", [4]]
    elif failure == "vision-export":
        exported["model.visual.block.weight"] = ["BF16", [4, 4]]
    elif failure == "mtp-export":
        exported["mtp.fc.weight"] = ["BF16", [4, 8]]
    elif failure == "rename":
        exported["model.layers.1.norm.weight"] = exported.pop(name)
    elif failure == "same-byte-shape":
        exported[name][1] = [2, 2]
    elif failure == "fp32":
        exported[name] = ["F32", [4]]
    elif failure == "bool-shape":
        exported[name][1] = [True, 4]
    elif failure == "unknown-source":
        source["unreviewed.weight"] = ["BF16", [4]]
    else:
        source.pop(name)
        exported.pop(name)
    with pytest.raises(ValueError):
        projection.validate_text_projection(*_inference_export_case)


def test_inference_export_additional_model_fields_are_not_exempt(_inference_export_case):
    source, exported, schema, output = _inference_export_case
    field = "num_experts" if source["model_type"].endswith("_moe") else "linear_conv_kernel_dim"
    exported[field] = 2
    with pytest.raises(ValueError, match=field):
        projection.validate_text_projection(source, exported, schema, output)


@pytest.fixture
def _inference_export_headers(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "weights"
    root.mkdir()
    path = root / "weights.safetensors"
    path.write_bytes(b"synthetic-header-only")
    schema = {"model.norm.weight": ["BF16", [4]]}
    keys = ["model.norm.weight"]

    def safe_open(filename, **kwargs):
        assert Path(filename) == path
        assert kwargs == {"framework": "pt", "device": "cpu"}

        class Headers:

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def keys(self):
                return list(keys)

            def get_slice(self, name):
                dtype, shape = schema[name]
                return SimpleNamespace(get_dtype=lambda: dtype, get_shape=lambda: list(shape))

            def get_tensor(self, name):
                pytest.fail("header admission must not materialize tensors")

        return Headers()

    monkeypatch.setattr(safetensors, "safe_open", safe_open)
    index = {"weight_map": {"model.norm.weight": path.name}, "metadata": {"total_size": 8}}
    return SimpleNamespace(root=root, path=path, schema=schema, keys=keys, index=index)


@pytest.mark.parametrize("count", [8, 8.0])
def test_inference_export_header_inventory_without_values(_inference_export_headers, count):
    _inference_export_headers.index["metadata"]["total_size"] = count
    assert (
        projection.read_inventory(_inference_export_headers.root, _inference_export_headers.index)
        == _inference_export_headers.schema
    )


def test_inference_export_scalar_header_inventory_without_values(_inference_export_headers):
    _inference_export_headers.schema["model.norm.weight"][1] = []
    _inference_export_headers.index["metadata"]["total_size"] = 2
    assert (
        projection.read_inventory(_inference_export_headers.root, _inference_export_headers.index)
        == _inference_export_headers.schema
    )


@pytest.mark.parametrize("count", [True, float("nan"), float("inf"), 8.5])
def test_inference_export_nonintegral_or_nonfinite_index_bytes_rejected(_inference_export_headers, count):
    _inference_export_headers.index["metadata"]["total_size"] = count
    with pytest.raises(ValueError, match="byte inventory"):
        projection.read_inventory(_inference_export_headers.root, _inference_export_headers.index)


@pytest.mark.parametrize("failure", ["unsafe-shard", "missing-key", "duplicate-key", "wrong-bytes", "symlink"])
def test_inference_export_index_header_fail_closed(_inference_export_headers, failure):
    if failure == "unsafe-shard":
        _inference_export_headers.index["weight_map"]["model.norm.weight"] = "../escape.safetensors"
    elif failure == "missing-key":
        _inference_export_headers.keys.clear()
    elif failure == "duplicate-key":
        _inference_export_headers.keys.append("model.norm.weight")
    elif failure == "wrong-bytes":
        _inference_export_headers.index["metadata"]["total_size"] = 9
    else:
        stored = _inference_export_headers.path.with_suffix(".stored")
        _inference_export_headers.path.rename(stored)
        _inference_export_headers.path.symlink_to(stored)
    with pytest.raises(ValueError):
        projection.read_inventory(_inference_export_headers.root, _inference_export_headers.index)


def test_inference_export_duplicate_json_rejected(tmp_path):
    path = tmp_path / "index.json"
    path.write_text('{"weight_map":{},"weight_map":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        projection.read_export_json(path)


# Inference rendezvous.


@pytest.fixture
def _inference_rendezvous_inference_run(tmp_path, monkeypatch):
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    distribution = {"examples": 1, "minimum_tokens": 4, "median_tokens": 4, "maximum_tokens": 4}
    cfg = SimpleNamespace(
        model_name="Qwen/Qwen3-8B",
        training={"model_provider": "huggingface", "dtype": "bfloat16"},
        sub_job={},
        n_gpus=16,
        dp_size=16,
        max_tokens_per_mb=8192,
    )
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=SimpleNamespace(model=SimpleNamespace(cache_path=str(tmp_path / "model"))),
        workdir=tmp_path,
        attn_implementation="flash_attention_3",
    )
    monkeypatch.setattr(check, "configured_learning_rate", lambda cfg: 0.01)
    monkeypatch.setattr(check, "configured_rows_per_step", lambda cfg: 1)
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda value: 0)
    monkeypatch.setattr(check.gsm8k, "read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda *args: [])
    monkeypatch.setattr(check.gsm8k, "length_distribution", lambda rows: distribution)
    example = SimpleNamespace(prompt="question", completion=" answer")
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: ([example], 4))
    monkeypatch.setattr(check.gsm8k, "assert_disjoint", lambda *args: None)
    monkeypatch.setattr(check.gsm8k, "build_batch", lambda *args: object())
    monkeypatch.setattr(check, "training_batches", lambda *args: [object()] * check.TRAIN_STEPS)
    monkeypatch.setattr(check, "pack_logit_aligned", lambda batch: {})
    monkeypatch.setattr(check, "pack", lambda *args, **kwargs: {})
    monkeypatch.setattr(check, "build_payload", lambda *args, **kwargs: {"job_type": "training"})
    monkeypatch.setattr(check, "sampling_payload", lambda *args, **kwargs: {"job_type": "sampling"})
    monkeypatch.setattr(check, "fwd_bwd_step", lambda *args, **kwargs: SimpleNamespace(avg_loss=2.0))
    monkeypatch.setattr(check, "forward_logprobs", lambda *args: None)
    monkeypatch.setattr(check, "reference_answer_logprobs", lambda *args: [[-2.0]])
    monkeypatch.setattr(check, "target_answer_logprobs", lambda *args: [[-2.0]])
    monkeypatch.setattr(check, "log_probs", lambda *args, **kwargs: None)
    monkeypatch.setattr(check, "save_weights_only_checkpoint", lambda *args: str(tmp_path / "hf"))
    monkeypatch.setattr(check, "verdict", lambda *args: "verified")
    ports, jobs, allocations = ([], [], [])

    @contextmanager
    def gateway(path, slots, *, backend):
        assert isinstance(backend, OnPremConfig)
        yield object()

    @contextmanager
    def running_job(url, payload):
        ports.append(os.environ.get("MASTER_PORT"))
        jobs.append(payload["job_type"])
        if len(jobs) == 3 and failure[0]:
            raise RuntimeError("reload failed")
        yield object()

    def allocate_port(*args, **kwargs):
        allocations.append((args, kwargs))
        return 30001

    failure = [False]
    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", running_job)
    monkeypatch.setattr(check, "get_unique_port_number", lambda: 30000, raising=False)
    monkeypatch.setattr(check, "reserve_free_port", allocate_port, raising=False)
    return SimpleNamespace(context=ctx, ports=ports, jobs=jobs, allocations=allocations, failure=failure)


@pytest.mark.parametrize("configured_port", [None, "29500"])
def test_inference_rendezvous_reload_leaves_standard_runtime_rendezvous_and_caller_environment_unchanged(
    _inference_rendezvous_inference_run, monkeypatch, configured_port
):
    if configured_port is None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
    else:
        monkeypatch.setenv("MASTER_PORT", configured_port)
    assert check._run(_inference_rendezvous_inference_run.context) == "verified"
    assert _inference_rendezvous_inference_run.jobs == ["training", "sampling", "training"]
    assert not _inference_rendezvous_inference_run.allocations
    assert _inference_rendezvous_inference_run.ports == [configured_port] * 3
    assert os.environ.get("MASTER_PORT") == configured_port


def test_inference_rendezvous_reload_error_does_not_allocate_an_alternate_port(
    _inference_rendezvous_inference_run, monkeypatch
):
    monkeypatch.setenv("MASTER_PORT", "29500")
    _inference_rendezvous_inference_run.failure[0] = True
    with pytest.raises(RuntimeError, match="reload failed"):
        check._run(_inference_rendezvous_inference_run.context)
    assert not _inference_rendezvous_inference_run.allocations
    assert _inference_rendezvous_inference_run.ports == ["29500"] * 3
    assert os.environ["MASTER_PORT"] == "29500"


# Reviewed training weight-format callers.


def _reviewed_weight_format_callers_config(tmp_path, weight_format, *, lora=False):
    root = Path(__file__).resolve().parents[1] / "configs"
    relative = (
        "qwen3-8b/h200/train-sft-lora-4gpus-64k.config"
        if lora
        else "qwen3.8-flash-next/h200/train-sft-full-8gpus-sp8-256k.config"
    )
    source = root / relative
    recipe = json.loads(source.read_text())
    recipe["training"].pop("weight_format", None)
    if weight_format is not None:
        recipe["training"]["weight_format"] = weight_format
    path = tmp_path / source.name
    path.write_text(json.dumps(recipe))
    return load_config(path)


def _reviewed_weight_format_callers_assert_payload(payload, cfg, model_path, *, gradient_norms_per_param=True):
    expected = rl_driver.build_payload(
        cfg.training,
        model_path,
        SEED,
        attn_implementation="sdpa",
        gradient_norms_per_param=gradient_norms_per_param,
    )
    observed = deepcopy(payload)
    assert ("weight_format" in observed) == (cfg.sub_job.get("weight_format") is not None)
    assert observed.pop("weight_format", None) == cfg.sub_job.get("weight_format")
    assert observed == expected
    assert "weight_format" not in observed["training_config"]


@pytest.fixture
def _reviewed_weight_format_callers_e2e_run(tmp_path, monkeypatch):
    from arctic_platform.correctness.checks import e2e_sft_train_validate as e2e

    cfg = _reviewed_weight_format_callers_config(tmp_path, "vllm")
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=SimpleNamespace(model=SimpleNamespace(cache_path="/data-fast/tmp/reviewed-caller-model")),
        attn_implementation="sdpa",
        client_backend=OnPremConfig(),
    )
    captured = []

    @contextmanager
    def gateway(directory, slots, *, backend):
        assert slots == ctx.cfg.n_gpus
        assert backend is ctx.client_backend
        yield SimpleNamespace(backend=backend, workdir=tmp_path)

    @contextmanager
    def running_job(session, payload):
        captured.append(payload)
        yield "job"

    monkeypatch.setattr(e2e, "gateway", gateway)
    monkeypatch.setattr(e2e, "running_job", running_job)
    monkeypatch.setattr(
        e2e,
        "request_bodies",
        lambda *args, **kwargs: ([{"training": 1}], {"processing": {"return_logprobs": True}}),
    )
    monkeypatch.setattr(e2e, "fwd_bwd_step", lambda *args, **kwargs: SimpleNamespace(avg_loss=2.0))
    monkeypatch.setattr(e2e, "forward_logprobs", lambda *args: torch.tensor([[-1.0]]))
    return SimpleNamespace(module=e2e, context=ctx, captured=captured)


@pytest.mark.parametrize("weight_format", [None, "vllm", "hf", "lora"])
def test_reviewed_weight_format_callers_e2e_trajectory_preserves_top_level_format_and_recipe(
    _reviewed_weight_format_callers_e2e_run, tmp_path, weight_format
):
    run = _reviewed_weight_format_callers_e2e_run
    run.context.cfg = _reviewed_weight_format_callers_config(tmp_path, weight_format)
    before = deepcopy(run.context.cfg)
    losses, logprobs = run.module.run_dss_trajectory(run.context, object(), tmp_path)
    assert losses == [2.0]
    assert torch.equal(logprobs, torch.tensor([[-1.0]]))
    assert len(run.captured) == 1
    _reviewed_weight_format_callers_assert_payload(run.captured[0], run.context.cfg, run.context.spec.model.cache_path)
    assert run.context.cfg == before


@pytest.mark.parametrize("weight_format", [None, "vllm", "hf", "lora"])
def test_reviewed_weight_format_callers_adapter_preparation_preserves_format_and_recipe(
    tmp_path, monkeypatch, weight_format
):
    from arctic_platform.correctness.checks import e2e_sft_train_validate as e2e
    from arctic_platform.correctness.harness import lora

    cfg = _reviewed_weight_format_callers_config(tmp_path, weight_format, lora=True)
    before = deepcopy(cfg)
    ctx = SimpleNamespace(cfg=cfg, attn_implementation="sdpa", prepare_references=False)
    adapter = tmp_path / "adapter-init"
    captured = []
    monkeypatch.setattr(lora, "materialize_adapter", lambda *args: adapter)
    monkeypatch.setattr(lora, "identity_metadata", lambda path: {"mock-adapter": str(path)})

    def bind(payload, path):
        assert path == adapter
        captured.append(payload)
        return payload

    monkeypatch.setattr(lora, "bind_payload", bind)
    actual, identity = e2e._prepare_adapter(ctx, cfg.model_name, tmp_path)
    assert actual == adapter
    assert identity == {"kind": "deterministic-lora", "metadata": {"mock-adapter": str(adapter)}}
    assert len(captured) == 1
    _reviewed_weight_format_callers_assert_payload(captured[0], cfg, cfg.model_name)
    assert cfg == before


@pytest.mark.parametrize("weight_format", [None, "vllm", "hf", "lora"])
def test_reviewed_weight_format_callers_inference_training_and_reload_preserve_format_and_recipe(
    _inference_rendezvous_inference_run, tmp_path, monkeypatch, weight_format
):
    run = _inference_rendezvous_inference_run
    run.context.cfg = _reviewed_weight_format_callers_config(tmp_path, weight_format)
    run.context.attn_implementation = "sdpa"
    before = deepcopy(run.context.cfg)
    captured = []

    @contextmanager
    def running_job(session, payload):
        captured.append(payload)
        yield "job"

    monkeypatch.setattr(check, "running_job", running_job)
    monkeypatch.setattr(check, "build_payload", rl_driver.build_payload)
    monkeypatch.setattr(check, "materialize_checkpoint_tokenizer", lambda *args: None)
    assert check._run(run.context) == "verified"
    assert [payload["job_type"] for payload in captured] == ["training", "sampling", "training"]
    _reviewed_weight_format_callers_assert_payload(
        captured[0], run.context.cfg, run.context.cfg.model_name, gradient_norms_per_param=False
    )
    _reviewed_weight_format_callers_assert_payload(
        captured[2], run.context.cfg, str(tmp_path / "hf"), gradient_norms_per_param=False
    )
    assert run.context.cfg == before


def test_reviewed_weight_format_callers_explicit_onprem_format_reaches_prestart_route_rejection(
    _reviewed_weight_format_callers_e2e_run, monkeypatch, tmp_path
):
    from arctic_platform.correctness.harness import cortex

    run = _reviewed_weight_format_callers_e2e_run
    captured = []

    def must_not_start(*args, **kwargs):
        pytest.fail("unsupported reviewed format must fail before starting the public client")

    @contextmanager
    def running_job(session, payload):
        captured.append(payload)
        config = rl_driver._training_client_config(session, payload)
        cortex.CorrectnessClient(config)
        pytest.fail("unsupported reviewed format must not create a job")
        yield

    monkeypatch.setattr(cortex.ArcticClient, "__init__", must_not_start)
    monkeypatch.setattr(run.module, "running_job", running_job)
    with pytest.raises(ValueError, match="require the Cortex correctness route"):
        run.module.run_dss_trajectory(run.context, object(), tmp_path)
    assert len(captured) == 1
    assert captured[0]["weight_format"] == "vllm"


def test_reviewed_weight_format_callers_onprem_rejects_before_gateway_runtime_or_client_start(
    _reviewed_weight_format_callers_e2e_run, monkeypatch, tmp_path
):
    run = _reviewed_weight_format_callers_e2e_run

    @contextmanager
    def gateway(directory, slots, *, backend):
        yield SimpleNamespace(backend=backend, workdir=tmp_path, slots=slots)

    def must_not_start(*args, **kwargs):
        pytest.fail("unsupported reviewed format must fail before entering the gateway runtime or client")

    monkeypatch.setattr(run.module, "gateway", gateway)
    monkeypatch.setattr(run.module, "running_job", rl_driver.running_job)
    monkeypatch.setattr(rl_driver, "_gateway_runtime", must_not_start)
    monkeypatch.setattr(rl_driver, "ArcticClient", must_not_start)
    with pytest.raises(ValueError, match="require the Cortex correctness route"):
        run.module.run_dss_trajectory(run.context, object(), tmp_path)


# Inference sampling topology.


@pytest.fixture
def _inference_export_locality_saved(tmp_path):
    root = tmp_path / "trained-step10"
    root.mkdir()
    save_file({"model.weight": torch.zeros((2, 2), dtype=torch.bfloat16)}, root / "model.safetensors")
    for name, value in {
        "config.json": {"model_type": "qwen3", "num_hidden_layers": 4},
        "model.safetensors.index.json": {
            "weight_map": {"model.weight": "model.safetensors"},
            "metadata": {"total_size": 8},
        },
        "tokenizer.json": {"version": "1.0", "model": {"vocab": {"x": 0}}},
        "tokenizer_config.json": {"eos_token": "x"},
    }.items():
        (root / name).write_text(json.dumps(value))
    return root


def _inference_export_locality_node(host="127.0.0.1", slots=8):
    return {
        "host": host,
        "slots": slots,
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "python": sys.executable,
        "python_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
    }


@pytest.fixture
def _inference_export_locality_admission(monkeypatch, tmp_path):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("127.0.0.1 slots=8\n")
    monkeypatch.setattr(export, "_LOCAL_HOSTFILE", hostfile)
    return {
        "allocation_id": "test-allocation",
        "hostfile_content": hostfile.read_text(),
        "hostfile_sha256": hashlib.sha256(hostfile.read_bytes()).hexdigest(),
        "nodes": [_inference_export_locality_node()],
    }


@pytest.fixture
def _inference_export_prescribed_hostfile(_inference_export_locality_admission, monkeypatch, tmp_path):
    hostfile = export._LOCAL_HOSTFILE
    target = tmp_path / "data-fast-hostfile"
    hostfile.rename(target)
    hostfile.symlink_to(target)
    monkeypatch.setattr(export, "_LOCAL_HOSTFILE_TARGET", target)
    return _inference_export_locality_admission


def test_inference_export_hostfile_symlink_prescribed_target_binds_bytes_and_order(
    _inference_export_prescribed_hostfile,
):
    admission = _inference_export_prescribed_hostfile
    target = export._LOCAL_HOSTFILE_TARGET
    target.write_text(target.read_text() + "127.0.0.2 slots=4\n")
    admission["hostfile_content"] = target.read_text()
    admission["hostfile_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    worker = _inference_export_locality_node("127.0.0.2", 4)
    worker["hostname"] = "fixture-worker"
    admission["nodes"].append(worker)
    before = export._local_hostfile_identity()
    assert export.validate_local_inference_allocation(admission) == admission["nodes"]
    assert export._local_hostfile_identity() == before
    assert before["link"] == str(target)
    assert before["target_path"] == str(target)
    assert before["original"] != before["target"]


@pytest.mark.parametrize("change", ["link_target", "same_bytes_target_replacement"])
def test_inference_export_hostfile_symlink_changed_during_validation_rejected(
    _inference_export_prescribed_hostfile, monkeypatch, tmp_path, change
):
    original_parse = export.parse_hostfile

    def parse_then_change(path):
        rows = original_parse(path)
        replacement = tmp_path / "replacement-hostfile"
        replacement.write_bytes(path.read_bytes())
        if change == "link_target":
            export._LOCAL_HOSTFILE.unlink()
            export._LOCAL_HOSTFILE.symlink_to(replacement)
        else:
            replacement.replace(path)
        return rows

    monkeypatch.setattr(export, "parse_hostfile", parse_then_change)
    with pytest.raises(ValueError, match="hostfile"):
        export.validate_local_inference_allocation(_inference_export_prescribed_hostfile)


@pytest.mark.parametrize("change", ["same_target_link_replacement", "same_bytes_target_replacement"])
def test_inference_export_hostfile_symlink_identity_change_blocks_published_sampling(
    _inference_export_prescribed_hostfile, _inference_export_locality_saved, monkeypatch, tmp_path, change
):
    admission = _inference_export_prescribed_hostfile
    root = _inference_export_locality_saved
    monkeypatch.setattr(export, "require_data_fast_path", lambda path: Path(path))
    proof = export.distribute_local_inference_export(root, admission, timeout_s=30)
    ctx = SimpleNamespace(inference_export_allocation=admission, _inference_export_locality=proof)
    export.require_local_inference_export_locality(ctx, root)
    replacement = tmp_path / "replacement"
    if change == "same_target_link_replacement":
        replacement.symlink_to(export._LOCAL_HOSTFILE_TARGET)
        replacement.replace(export._LOCAL_HOSTFILE)
    else:
        replacement.write_bytes(export._LOCAL_HOSTFILE_TARGET.read_bytes())
        replacement.replace(export._LOCAL_HOSTFILE_TARGET)
    with pytest.raises(ValueError, match="hostfile link or target identity changed"):
        export.require_local_inference_export_locality(ctx, root)


def test_inference_export_integration_preserves_reviewed_glm_materializer_bytes():
    # The independent GLM release must survive the locality-only runner merge.
    source = inspect.getsource(runner.materialize_dss_peft_adapter).encode()
    assert hashlib.sha256(source).hexdigest() == "d517f8fa302b1ad110eb4ea0caca73305dc42faec10aa8b0bca559277eb379cc"


def test_inference_export_cleanup_exited_leader_stubborn_descendant_is_killed_and_group_empty():
    import ctypes
    import signal
    import threading

    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    leader = None
    sentinel = None
    reaper = None
    try:
        sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os,signal,time\n"
                    "reader,writer=os.pipe()\n"
                    "child=os.fork()\n"
                    "if child==0:\n"
                    " os.close(reader)\n"
                    " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                    " os.write(writer,b'ready');os.close(writer)\n"
                    " while True: time.sleep(1)\n"
                    "os.close(writer);os.read(reader,5);os.close(reader)\n"
                    "print(child,flush=True)\n"
                    "os._exit(0)\n"
                ),
            ],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        descendant = int(leader.stdout.readline())
        assert leader.wait(timeout=5) == 0
        reaped = []
        reaper = threading.Thread(target=lambda: reaped.append(os.waitpid(descendant, 0)), daemon=True)
        reaper.start()
        started = time.monotonic()
        export._stop_local_children([leader])
        reaper.join(timeout=1)
        assert time.monotonic() - started < 7
        assert len(reaped) == 1 and reaped[0][0] == descendant
        assert os.WIFSIGNALED(reaped[0][1]) and os.WTERMSIG(reaped[0][1]) == signal.SIGKILL
        with pytest.raises(ProcessLookupError):
            os.killpg(leader.pid, 0)
        assert sentinel.poll() is None
    finally:
        if leader is not None:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            leader.wait(timeout=5)
            leader.stdout.close()
        if reaper is not None:
            reaper.join(timeout=5)
        if sentinel is not None:
            sentinel.terminate()
            sentinel.wait(timeout=5)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.fixture
def _inference_export_integration_runner(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config")
    spec = SimpleNamespace(
        applicable_tests=["inference-checkpoint-loss"],
        test_settings={},
        model=SimpleNamespace(cache_path=str(tmp_path / "fixture")),
    )
    return cfg, spec


def test_inference_export_integration_real_runner_admits_and_copies_allocation(
    _inference_export_integration_runner, _inference_export_prescribed_hostfile, monkeypatch, tmp_path
):
    cfg, spec = _inference_export_integration_runner
    admission = _inference_export_prescribed_hostfile
    before = deepcopy(admission)
    observed = []
    backend = CortexConfig(base_url="http://localhost:8003")
    monkeypatch.setattr(runner, "require_data_fast_path", lambda path, **kwargs: Path(path))

    def check_context(ctx):
        observed.append(ctx)
        assert ctx.inference_export_allocation == admission
        assert ctx.inference_export_allocation is not admission
        assert ctx.inference_export_allocation["nodes"] is not admission["nodes"]
        assert ctx._inference_export_locality is None
        assert ctx.client_backend is backend
        return []

    monkeypatch.setattr(registered_tests()["inference-checkpoint-loss"], "fn", check_context)
    assert (
        runner.execute(
            cfg,
            spec,
            [],
            vocab_size=128,
            workdir=tmp_path,
            select=["inference-checkpoint-loss"],
            client_backend=backend,
            inference_export_allocation=admission,
        )
        == []
    )
    assert len(observed) == 1
    assert admission == before


@pytest.mark.parametrize("invalid", ["nonlocal_cortex", "onprem", "other_check"])
def test_inference_export_integration_real_runner_rejects_unadmitted_route_before_work(
    _inference_export_integration_runner, _inference_export_prescribed_hostfile, monkeypatch, invalid
):
    cfg, spec = _inference_export_integration_runner
    backend = (
        OnPremConfig()
        if invalid == "onprem"
        else CortexConfig(
            base_url="http://other-job.invalid" if invalid == "nonlocal_cortex" else "http://localhost:8003"
        )
    )
    selected = ["single-step-grads"] if invalid == "other_check" else ["inference-checkpoint-loss"]
    monkeypatch.setattr(
        runner, "registered_tests", lambda: pytest.fail("invalid admission must stop before test or model setup")
    )
    with pytest.raises(ValueError, match="only local Cortex inference-checkpoint-loss"):
        runner.execute(
            cfg,
            spec,
            [],
            vocab_size=128,
            select=selected,
            client_backend=backend,
            inference_export_allocation=_inference_export_prescribed_hostfile,
        )


def _inference_export_locality_manifest(root):
    return export._local_manifest(root, time.monotonic() + 30)


def _inference_export_locality_wire(root, manifest):
    return (
        json.dumps(manifest, sort_keys=True).encode()
        + b"\n"
        + b"".join((root / row["path"]).read_bytes() for row in manifest["files"])
    )


def test_inference_export_locality_manifest_hashes_all_trained_files_and_preserves_header_identity(
    _inference_export_locality_saved,
):
    root = _inference_export_locality_saved
    manifest = _inference_export_locality_manifest(root)
    assert {row["path"] for row in manifest["files"]} == {p.name for p in root.iterdir()}
    for row in manifest["files"]:
        assert row["sha256"] == hashlib.sha256((root / row["path"]).read_bytes()).hexdigest()
        assert row["size_bytes"] == (root / row["path"]).stat().st_size
    assert manifest["tensor_schema_sha256"] == export.schema_sha({"model.weight": ["BF16", [2, 2]]})
    assert manifest["tensor_payload_bytes"] == 8
    assert manifest["config_sha256"] == export.schema_sha(json.loads((root / "config.json").read_text()))
    assert set(manifest["tokenizer_files"]) == {"tokenizer.json", "tokenizer_config.json"}


@pytest.mark.parametrize("damage", ["missing_tokenizer", "symlink", "directory", "extra_shard", "wrong_index"])
def test_inference_export_locality_manifest_rejects_incomplete_or_nonregular_exports(
    _inference_export_locality_saved,
    damage,
):
    root = _inference_export_locality_saved
    if damage == "missing_tokenizer":
        (root / "tokenizer.json").unlink()
    elif damage == "symlink":
        (root / "aliased.json").symlink_to(root / "config.json")
    elif damage == "directory":
        (root / "nested").mkdir()
    elif damage == "extra_shard":
        (root / "unindexed.safetensors").write_bytes(b"unused")
    else:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"other": "model.safetensors"}, "metadata": {"total_size": 8}})
        )
    with pytest.raises((ValueError, OSError)):
        _inference_export_locality_manifest(root)


@pytest.mark.parametrize("changed", ["hostfile_hash", "hostfile_content", "ordered_hosts", "boot", "python"])
def test_inference_export_locality_admission_rejects_changed_job_or_node(
    _inference_export_locality_admission,
    changed,
):
    admission = deepcopy(_inference_export_locality_admission)
    if changed == "hostfile_hash":
        admission["hostfile_sha256"] = "0" * 64
    elif changed == "hostfile_content":
        admission["hostfile_content"] += "#changed\n"
    elif changed == "ordered_hosts":
        admission["nodes"][0]["host"] = "external-job.invalid"
    elif changed == "boot":
        admission["nodes"][0]["boot_id"] = "0" * 36
    else:
        admission["nodes"][0]["python_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        export.validate_local_inference_allocation(admission)


def test_inference_export_locality_receiver_publishes_verified_files_exclusively(
    monkeypatch,
    tmp_path,
    _inference_export_locality_saved,
):
    root = _inference_export_locality_saved
    manifest = _inference_export_locality_manifest(root)
    destination = tmp_path / "worker-trained-step10"
    request = {
        "node": _inference_export_locality_node(),
        "root": str(destination),
        "allocation_id": "test-allocation",
        "manifest_sha256": export.schema_sha(manifest),
    }
    wire = _inference_export_locality_wire(root, manifest)
    monkeypatch.setattr(export.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(wire)))
    receipt = export._receive_local_export(request, time.monotonic() + 30)
    assert receipt["semantic_exit"] == 0
    assert receipt["files_verified"] == 5
    assert json.loads((destination / export._LOCAL_RECEIPT).read_text()) == receipt
    for row in manifest["files"]:
        assert (destination / row["path"]).read_bytes() == (root / row["path"]).read_bytes()
    monkeypatch.setattr(export.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(wire)))
    with pytest.raises(FileExistsError):
        export._receive_local_export(request, time.monotonic() + 30)


@pytest.mark.parametrize("damage", ["truncated", "corrupted", "extra_bytes", "manifest_identity", "node_identity"])
def test_inference_export_locality_receiver_never_publishes_partial_or_wrong_bytes(
    monkeypatch,
    tmp_path,
    _inference_export_locality_saved,
    damage,
):
    root = _inference_export_locality_saved
    manifest = _inference_export_locality_manifest(root)
    destination = tmp_path / "failed-worker-save"
    request = {
        "node": _inference_export_locality_node(),
        "root": str(destination),
        "allocation_id": "test-allocation",
        "manifest_sha256": export.schema_sha(manifest),
    }
    wire = _inference_export_locality_wire(root, manifest)
    if damage == "truncated":
        wire = wire[:-1]
    elif damage == "corrupted":
        wire = wire[:-1] + bytes([wire[-1] ^ 1])
    elif damage == "extra_bytes":
        wire += b"unexpected"
    elif damage == "manifest_identity":
        request["manifest_sha256"] = "0" * 64
    else:
        request["node"]["boot_id"] = "0" * 36
    monkeypatch.setattr(export.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(wire)))
    with pytest.raises(ValueError):
        export._receive_local_export(request, time.monotonic() + 30)
    assert not (destination / export._LOCAL_RECEIPT).exists()


def test_inference_export_locality_strict_transport_uses_only_admitted_host_and_keyfile(
    tmp_path,
    _inference_export_locality_admission,
):
    admission = _inference_export_locality_admission
    known = tmp_path / "known_hosts"
    known.write_text("admitted-public-key-placeholder\n")
    admission["ssh"] = {
        "port": 7878,
        "known_hosts": str(known),
        "known_hosts_sha256": hashlib.sha256(known.read_bytes()).hexdigest(),
    }
    node = admission["nodes"][0]
    argv = export._local_ssh_argv(admission, node, {"timeout_s": 5}, time.monotonic() + 5)
    assert argv[-2] == node["host"]
    assert "StrictHostKeyChecking=yes" in argv
    assert "BatchMode=yes" in argv
    assert "UserKnownHostsFile=" + str(known) in argv
    known.write_text("changed-keys\n")
    with pytest.raises(ValueError, match="strict SSH"):
        export._local_ssh_argv(admission, node, {}, time.monotonic() + 5)


def test_inference_export_locality_bounded_child_timeout_reaps_only_created_session(monkeypatch):
    children = []
    original = subprocess.Popen

    def observe(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(export.subprocess, "Popen", observe)
    before = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        export._bounded_local_json([sys.executable, "-c", "import time;time.sleep(60)"], time.monotonic() + 0.2)
    assert time.monotonic() - before < 5
    assert len(children) == 1 and children[0].poll() is not None


def test_inference_export_locality_single_head_proof_is_bound_and_mutation_blocks_sampling(
    monkeypatch,
    _inference_export_locality_saved,
    _inference_export_locality_admission,
):
    root = _inference_export_locality_saved
    admission = _inference_export_locality_admission

    def manifest_worker(argv, deadline):
        fingerprints = {}
        manifest = export._local_manifest(root, deadline, fingerprints)
        return {"manifest": manifest, "head_files": fingerprints}

    monkeypatch.setattr(export, "_bounded_local_json", manifest_worker)
    proof = export.distribute_local_inference_export(root, admission, timeout_s=30)
    ctx = SimpleNamespace(inference_export_allocation=admission, _inference_export_locality=proof)
    export.require_local_inference_export_locality(ctx, root)
    (root / "tokenizer_config.json").write_text('{"eos_token":"changed"}')
    with pytest.raises(ValueError, match="changed after locality"):
        export.require_local_inference_export_locality(ctx, root)


def test_inference_export_locality_missing_publication_blocks_sampling_before_payload(monkeypatch, tmp_path):
    ctx = SimpleNamespace()
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    monkeypatch.setattr(check, "validate_local_inference_export", lambda ctx, path: path)

    def forbidden(*args):
        pytest.fail("sampling payload created without trained export locality")

    monkeypatch.setattr(check, "_sampling_payload", forbidden)
    with pytest.raises(ValueError, match="receipt required before sampling"):
        check.local_cortex_sampling_config(ctx, backend, tmp_path / "unpublished", 10240)


def test_inference_export_locality_local_run_rejects_absent_admission_before_tokenizer_or_model(monkeypatch):
    ctx = SimpleNamespace(
        cfg=load_config(_inference_sampling_topology_ORDINARY_CONFIG),
        client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
    )

    def forbidden(*args):
        pytest.fail("model preparation started without same-allocation admission")

    monkeypatch.setattr(check.gsm8k, "load_tokenizer", forbidden)
    with pytest.raises(ValueError, match="current allocation admission required"):
        check._run(ctx)


def test_inference_export_locality_save_publishes_only_after_real_save_and_tokenizer_validation(
    monkeypatch,
    tmp_path,
):
    events = []
    ctx = SimpleNamespace(inference_export_allocation={"admitted": True})
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    result = {"path": str(tmp_path), "tag": "global_step10"}
    tokenizer = SimpleNamespace(get_vocab=lambda: {"x": 0}, special_tokens_map={"eos_token": "x"})

    def save(**kwargs):
        assert kwargs == {"path": str(tmp_path), "checkpoint_type": "weights-only", "export_hf": False}
        events.append("public_save")
        return result

    def metadata(actual, root, operation):
        assert actual == result and root == tmp_path and operation == "save"
        return "global_step10"

    monkeypatch.setattr(check, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(check, "_local_checkpoint_metadata", metadata)
    monkeypatch.setattr(check, "validate_local_inference_export", lambda ctx, path: path)
    monkeypatch.setattr(
        check, "materialize_checkpoint_tokenizer", lambda tokenizer, path: events.append("tokenizer_save")
    )
    monkeypatch.setattr(check, "_export_json", lambda path: {})
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: tokenizer)
    proof = {"actual_receipts": True}

    def distribute(root, admission):
        assert root == tmp_path / "global_step10" and admission == {"admitted": True}
        events.append("locality")
        return proof

    monkeypatch.setattr(export, "distribute_local_inference_export", distribute)
    weights = check.save_local_inference_checkpoint(ctx, backend, SimpleNamespace(save_checkpoint=save), tokenizer)
    assert weights == tmp_path / "global_step10"
    assert events == ["public_save", "tokenizer_save", "locality"]
    assert ctx._inference_export_locality is proof


def test_inference_export_locality_changed_tokenizer_never_reaches_transfer(monkeypatch, tmp_path):
    ctx = SimpleNamespace()
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    tokenizer = SimpleNamespace(get_vocab=lambda: {"x": 0}, special_tokens_map={})
    changed = SimpleNamespace(get_vocab=lambda: {"y": 0}, special_tokens_map={})
    monkeypatch.setattr(check, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(check, "_local_checkpoint_metadata", lambda *args: "global_step10")
    monkeypatch.setattr(check, "validate_local_inference_export", lambda ctx, path: path)
    monkeypatch.setattr(check, "materialize_checkpoint_tokenizer", lambda *args: None)
    monkeypatch.setattr(check, "_export_json", lambda path: {})
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: changed)

    def forbidden(*args):
        pytest.fail("transfer started before tokenizer identity was verified")

    monkeypatch.setattr(export, "distribute_local_inference_export", forbidden)
    with pytest.raises(RuntimeError, match="tokenizer differs"):
        check.save_local_inference_checkpoint(
            ctx, backend, SimpleNamespace(save_checkpoint=lambda **kwargs: {}), tokenizer
        )


def test_inference_export_locality_process_stream_cpu_roundtrip_checks_complete_saved_bytes(
    tmp_path,
    _inference_export_locality_saved,
):
    source = _inference_export_locality_saved
    manifest = _inference_export_locality_manifest(source)
    destination = tmp_path / "cpu-receiver"
    request = {
        "root": str(destination),
        "node": _inference_export_locality_node(),
        "allocation_id": "cpu-control",
        "manifest_sha256": export.schema_sha(manifest),
        "timeout_s": 15,
    }
    children = []
    try:
        sender = subprocess.Popen(
            export._local_worker_argv(
                sys.executable, "send", {"root": str(source), "manifest": manifest, "timeout_s": 15}
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        children.append(sender)
        receiver = subprocess.Popen(
            export._local_worker_argv(sys.executable, "supervise", request),
            stdin=sender.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        children.append(receiver)
        sender.stdout.close()
        output, errors = receiver.communicate(timeout=15)
        sender.wait(timeout=15)
        sender_errors = sender.stderr.read()
        assert receiver.returncode == 0, errors.decode()
        assert sender.returncode == 0, sender_errors.decode()
        receipt = json.loads(output)
        assert receipt == json.loads((destination / export._LOCAL_RECEIPT).read_text())
        for row in manifest["files"]:
            assert (destination / row["path"]).read_bytes() == (source / row["path"]).read_bytes()
    finally:
        export._stop_local_children(children)


def test_inference_export_locality_process_backpressure_obeys_global_deadline_and_reaps_both_children(
    monkeypatch,
    _inference_export_locality_saved,
    _inference_export_locality_admission,
):
    root = _inference_export_locality_saved
    # Larger than a pipe buffer: the sender blocks if its recipient never reads.
    (root / "tokenizer.json").write_text(json.dumps({"padding": "x" * 1024**2}))
    fingerprints = {}
    manifest = export._local_manifest(root, time.monotonic() + 30, fingerprints)
    admission = _inference_export_locality_admission
    node = admission["nodes"][0]
    monkeypatch.setattr(export, "_validate_local_allocation", lambda admission, deadline: [node, node])
    monkeypatch.setattr(
        export, "_bounded_local_json", lambda argv, deadline: {"manifest": manifest, "head_files": fingerprints}
    )
    monkeypatch.setattr(
        export,
        "_local_ssh_argv",
        lambda *args: [sys.executable, "-c", "import time;time.sleep(60)"],
    )
    children = []
    original = subprocess.Popen

    def observe(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(export.subprocess, "Popen", observe)
    before = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        export.distribute_local_inference_export(root, admission, timeout_s=0.3)
    assert time.monotonic() - before < 5
    assert len(children) == 2 and all(child.poll() is not None for child in children)


def test_inference_export_locality_source_binding_worker_ignores_other_package_import_root(
    monkeypatch,
    _inference_export_locality_saved,
):
    root = _inference_export_locality_saved
    other_paths = [
        path for path in os.environ.get("PYTHONPATH", "").split(":") if "inference-export-locality" not in path
    ]
    monkeypatch.setenv("PYTHONPATH", ":".join(other_paths))
    result = export._bounded_local_json(
        export._local_worker_argv(sys.executable, "manifest", {"root": str(root), "timeout_s": 15}),
        time.monotonic() + 15,
    )
    assert result["manifest"] == _inference_export_locality_manifest(root)
    assert result["head_files"] == {
        row["path"]: list(export._fingerprint(root / row["path"])) for row in result["manifest"]["files"]
    }


def test_inference_export_locality_source_binding_expired_global_deadline_stops_before_receive(
    monkeypatch,
):
    monkeypatch.setattr(export, "_receive_local_export", lambda *args: pytest.fail("expired receiver started"))
    with pytest.raises(TimeoutError, match="locality deadline"):
        export._local_worker("receive", {"timeout_s": 60, "deadline_epoch": time.time() - 1})


_inference_sampling_topology_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

_inference_sampling_topology_FLASH_CONFIG = (
    _inference_sampling_topology_CONFIG_ROOT / "qwen3.8-flash-next/h200/train-sft-full-16gpus-sp16-256k.config"
)

_inference_sampling_topology_ORDINARY_CONFIG = (
    _inference_sampling_topology_CONFIG_ROOT / "qwen3-8b/h200/train-sft-full-4gpus-64k.config"
)

_inference_sampling_topology_FLASH_ID = "qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k"

_inference_sampling_topology_TOPOLOGIES = [(8, 8, 1), (16, 8, 2)]

_inference_sampling_topology_TOKEN_BUDGET = 10240


def _inference_sampling_topology_context(config=_inference_sampling_topology_FLASH_CONFIG, topology=None):
    cfg = load_config(config)
    return SimpleNamespace(
        cfg=cfg,
        config_id=cfg.config_id,
        inference_sampling_topology=topology,
        client_backend=OnPremConfig(),
        attn_implementation="qsa_flex",
    )


def _inference_sampling_topology_assert_topology(payload, topology):
    allocation, tp, pp = topology
    inf = payload["inference_config"]
    engine = inf["vllm_config"]
    assert inf["n_gpus"] == allocation == tp * pp
    assert engine["tensor_parallel_size"] == tp
    assert engine["pipeline_parallel_size"] == pp
    assert engine["enable_expert_parallel"] is True
    extra = engine["extra_engine_kwargs"]
    assert extra["distributed_executor_backend"] == "ray"
    assert extra["data_parallel_size"] == 1
    assert extra["prefill_context_parallel_size"] == 1
    assert extra["decode_context_parallel_size"] == 1
    assert extra["enable_eplb"] is False
    assert extra["enable_elastic_ep"] is False
    assert extra.get("nnodes", 1) == 1
    assert extra["engram_config"] == {
        "cpu_offload": False,
        "embedding_across_dp": False,
        "dp_shared_memory": False,
        "use_thp": False,
    }


@pytest.mark.parametrize("topology", _inference_sampling_topology_TOPOLOGIES)
def test_inference_sampling_topology_original_flashnext_request_separates_allocation_from_tp_and_pp(topology):
    ctx = _inference_sampling_topology_context(topology=topology)
    assert ctx.cfg.config_id == _inference_sampling_topology_FLASH_ID
    before = deepcopy(ctx.cfg)
    payload = check._sampling_payload(ctx, "/saved/full-checkpoint", _inference_sampling_topology_TOKEN_BUDGET)
    assert check.sampling_gpu_count(ctx) == topology[0]
    _inference_sampling_topology_assert_topology(payload, topology)
    assert payload["model_name"] == "/saved/full-checkpoint"
    assert payload["job_type"] == "log_prob"
    assert payload["seed"] == SEED
    assert payload["dtype"] == "bfloat16"
    inf = payload["inference_config"]
    assert inf["max_seq_len"] == _inference_sampling_topology_TOKEN_BUDGET + 1
    engine = inf["vllm_config"]
    assert engine["max_model_len"] == _inference_sampling_topology_TOKEN_BUDGET + 1
    assert engine["dtype"] == "bfloat16"
    assert engine["enforce_eager"] is True
    assert engine["disable_custom_all_reduce"] is True
    assert engine["gpu_memory_utilization"] == 0.85
    assert engine["trust_remote_code"] is True
    assert "model" not in engine
    assert "speculative_config" not in engine
    assert "speculative_config" not in engine["extra_engine_kwargs"]
    assert ctx.cfg == before


@pytest.mark.parametrize(
    "config", [_inference_sampling_topology_FLASH_CONFIG, _inference_sampling_topology_ORDINARY_CONFIG]
)
@pytest.mark.parametrize("missing_attribute", [False, True])
def test_inference_sampling_topology_without_opt_in_the_exact_one_gpu_payload_is_unchanged(config, missing_attribute):
    ctx = _inference_sampling_topology_context(config)
    if missing_attribute:
        del ctx.inference_sampling_topology
    assert check.sampling_gpu_count(ctx) == 1
    assert check._sampling_payload(
        ctx, "/saved/full-checkpoint", _inference_sampling_topology_TOKEN_BUDGET
    ) == sampling_payload(
        "/saved/full-checkpoint",
        SEED,
        dtype="bfloat16",
        n_gpus=1,
        max_seq_len=_inference_sampling_topology_TOKEN_BUDGET + 1,
    )


@pytest.mark.parametrize(
    "topology",
    [
        (),
        (16, 8),
        (16, 8, 2, 1),
        [16, 8, 2],
        {"allocation": 16, "tp": 8, "pp": 2},
        "16,8,2",
        (True, 8, 2),
        (16, True, 2),
        (16, 8, True),
        (0, 8, 2),
        (16, -8, 2),
        (16, 8.0, 2),
        (16, "8", 2),
        (8, 8, 2),
        (16, 16, 1),
        (8, 4, 2),
        (1, 1, 1),
    ],
)
def test_inference_sampling_topology_malformed_or_unreviewed_topology_is_rejected(topology):
    ctx = _inference_sampling_topology_context(topology=topology)
    with pytest.raises(ValueError):
        check.sampling_gpu_count(ctx)
    with pytest.raises(ValueError):
        check._sampling_payload(ctx, "/saved/full-checkpoint", _inference_sampling_topology_TOKEN_BUDGET)


@pytest.mark.parametrize("different_identity", ["config_id", "model_name", "training_width"])
def test_inference_sampling_topology_flashnext_opt_in_requires_exact_original_training_identity(different_identity):
    ctx = _inference_sampling_topology_context(topology=(16, 8, 2))
    if different_identity == "config_id":
        ctx.cfg.config_id += "-alias"
        ctx.config_id = ctx.cfg.config_id
    elif different_identity == "model_name":
        ctx.cfg.sub_job["model_name"] = "Qwen/Qwen3-8B"
        ctx.cfg.native.model_name = "Qwen/Qwen3-8B"
    else:
        ctx.cfg.native.training_gpus = 8
        ctx.cfg.training["n_gpus"] = 8
    with pytest.raises(ValueError):
        check.sampling_gpu_count(ctx)


@pytest.mark.parametrize("topology", _inference_sampling_topology_TOPOLOGIES)
def test_inference_sampling_topology_managed_sampling_uses_helper_and_preserves_topology_on_cortex_wire(
    monkeypatch, topology
):
    ctx = _inference_sampling_topology_context(topology=topology)
    backend = CortexConfig(base_url="https://unused.invalid")
    calls = []
    original = check._sampling_payload

    def observe(actual_ctx, model_path, token_budget):
        calls.append((actual_ctx, model_path, token_budget))
        return original(actual_ctx, model_path, token_budget)

    monkeypatch.setattr(check, "_sampling_payload", observe)
    config = check.cortex_sampling_config(
        ctx, backend, "cp_full_saved", "owned-training", _inference_sampling_topology_TOKEN_BUDGET
    )
    assert calls == [(ctx, ctx.cfg.model_name, _inference_sampling_topology_TOKEN_BUDGET)]
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    _inference_sampling_topology_assert_topology(sub, topology)
    assert sub["model_name"] == ctx.cfg.model_name
    assert sub["source_checkpoint_info"] == {"checkpoint_id": "cp_full_saved", "source_job_id": "owned-training"}
    assert sub["inference_config"]["max_seq_len"] == _inference_sampling_topology_TOKEN_BUDGET + 1
    assert "max_model_len" not in sub["inference_config"]["vllm_config"]
    assert config.backend == backend
    assert config.hardware == "H200"
    sub["inference_config"]["vllm_config"]["extra_engine_kwargs"]["engram_config"]["cpu_offload"] = True
    assert config.model_dump() == before
    _inference_sampling_topology_assert_topology(config.to_cortex()[0], topology)


@pytest.mark.parametrize("topology", _inference_sampling_topology_TOPOLOGIES)
def test_inference_sampling_topology_local_sampling_validates_export_then_uses_same_helper(
    monkeypatch, tmp_path, topology
):
    ctx = _inference_sampling_topology_context(topology=topology)
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    weights = tmp_path / "saved-full-export"
    events = []
    original = check._sampling_payload

    def validate(actual_ctx, actual_weights):
        assert actual_ctx is ctx
        assert actual_weights == weights
        events.append("validate")
        return actual_weights

    def observe(actual_ctx, model_path, token_budget):
        assert actual_ctx is ctx
        assert str(model_path) == str(weights)
        assert token_budget == _inference_sampling_topology_TOKEN_BUDGET
        events.append("helper")
        return original(actual_ctx, model_path, token_budget)

    monkeypatch.setattr(check, "validate_local_inference_export", validate)
    monkeypatch.setattr(export, "require_local_inference_export_locality", lambda ctx, path: events.append("locality"))
    monkeypatch.setattr(check, "_sampling_payload", observe)
    config = check.local_cortex_sampling_config(ctx, backend, weights, _inference_sampling_topology_TOKEN_BUDGET)
    assert events == ["validate", "locality", "helper"]
    sub = config.to_cortex()[0]
    _inference_sampling_topology_assert_topology(sub, topology)
    assert sub["model_name"] == str(weights)
    assert "source_checkpoint_info" not in sub
    assert config.source_checkpoint_info is None
    assert config.backend == backend


def test_inference_sampling_topology_local_topology_never_bypasses_full_export_rejection(monkeypatch, tmp_path):
    ctx = _inference_sampling_topology_context(topology=(16, 8, 2))
    backend = CortexConfig(base_url="http://127.0.0.1:8003")

    def invalid_export(*args):
        raise RuntimeError("full configured tensor inventory missing")

    def forbidden_helper(*args):
        pytest.fail("sampling helper called after rejected full export")

    monkeypatch.setattr(check, "validate_local_inference_export", invalid_export)
    monkeypatch.setattr(check, "_sampling_payload", forbidden_helper)
    with pytest.raises(RuntimeError, match="full configured tensor inventory"):
        check.local_cortex_sampling_config(
            ctx, backend, tmp_path / "incomplete", _inference_sampling_topology_TOKEN_BUDGET
        )


@pytest.mark.parametrize("topology", _inference_sampling_topology_TOPOLOGIES)
def test_inference_sampling_topology_onprem_sampling_uses_same_helper_before_engine_creation(
    monkeypatch, tmp_path, topology
):
    ctx = _inference_sampling_topology_context(topology=topology)
    session = object()
    weights = tmp_path / "saved-full-export"
    calls = []
    original = check._sampling_payload

    class StopBeforeEngine(Exception):
        pass

    def observe(actual_ctx, model_path, token_budget):
        assert actual_ctx is ctx
        assert model_path == weights
        assert token_budget == _inference_sampling_topology_TOKEN_BUDGET
        calls.append("helper")
        return original(actual_ctx, model_path, token_budget)

    @contextmanager
    def engine(actual_session, payload):
        assert actual_session is session
        assert calls == ["helper"]
        _inference_sampling_topology_assert_topology(payload, topology)
        raise StopBeforeEngine
        yield

    monkeypatch.setattr(check, "_sampling_payload", observe)
    monkeypatch.setattr(check, "running_job", engine)
    with pytest.raises(StopBeforeEngine):
        check._onprem_sampling_and_reload(ctx, session, weights, _inference_sampling_topology_TOKEN_BUDGET, [], {})


def test_inference_sampling_topology_production_client_cortex_serializer_remains_outside_correctness_override():
    payload = sampling_payload("Qwen/Qwen3-8B", SEED, dtype="bfloat16", n_gpus=4, max_seq_len=128)
    payload["job_type"] = "sampling"
    config = _inference_client_config(payload, backend=CortexConfig(base_url="https://unused.invalid"))
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    assert sub["inference_config"]["n_gpus"] == 4
    assert "tensor_parallel_size" not in sub["inference_config"]["vllm_config"]
    assert config.model_dump() == before


@pytest.mark.parametrize(
    "config_path", [_inference_sampling_topology_FLASH_CONFIG, _inference_sampling_topology_ORDINARY_CONFIG]
)
def test_inference_sampling_topology_correctness_one_gpu_wire_does_not_opt_in_implicitly(config_path):
    ctx = _inference_sampling_topology_context(config_path)
    backend = CortexConfig(base_url="https://unused.invalid")
    config = check.cortex_sampling_config(
        ctx, backend, "cp_full_saved", "owned-training", _inference_sampling_topology_TOKEN_BUDGET
    )
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    assert config.inference_tensor_parallel_size is None
    assert config.sampling_gpus == sub["inference_config"]["n_gpus"] == 1
    assert "tensor_parallel_size" not in sub["inference_config"]["vllm_config"]
    assert "pipeline_parallel_size" not in sub["inference_config"]["vllm_config"]
    assert "enable_expert_parallel" not in sub["inference_config"]["vllm_config"]
    assert config.model_dump() == before


@pytest.mark.parametrize(
    "invalid",
    [
        "tensor_bool",
        "tensor_float",
        "tensor_zero",
        "pipeline_bool",
        "allocation_mismatch",
        "engine_tensor_mismatch",
        "engine_tensor_bool",
        "no_inference_allocation",
        "non_cortex_backend",
    ],
)
def test_inference_sampling_topology_scoped_cortex_field_rejects_inconsistent_or_non_cortex_topology(invalid):
    ctx = _inference_sampling_topology_context(topology=(16, 8, 2))
    backend = CortexConfig(base_url="https://unused.invalid")
    config = check.cortex_sampling_config(
        ctx, backend, "cp_full_saved", "owned-training", _inference_sampling_topology_TOKEN_BUDGET
    )
    values = deepcopy(config.model_dump())
    if invalid == "tensor_bool":
        values["inference_tensor_parallel_size"] = True
    elif invalid == "tensor_float":
        values["inference_tensor_parallel_size"] = 8.0
    elif invalid == "tensor_zero":
        values["inference_tensor_parallel_size"] = 0
    elif invalid == "pipeline_bool":
        values["sampling"]["vllm"]["pipeline_parallel_size"] = True
    elif invalid == "allocation_mismatch":
        values["sampling_gpus"] = 8
    elif invalid == "engine_tensor_mismatch":
        values["sampling"]["vllm"]["tensor_parallel_size"] = 4
    elif invalid == "engine_tensor_bool":
        values["inference_tensor_parallel_size"] = 1
        values["sampling"]["vllm"]["tensor_parallel_size"] = True
        values["sampling"]["vllm"]["pipeline_parallel_size"] = 16
    elif invalid == "no_inference_allocation":
        values["sampling_gpus"] = 0
    else:
        values["backend"] = OnPremConfig().model_dump()
    with pytest.raises(ValueError):
        CortexTrainingConfig.model_validate(values)


@pytest.mark.parametrize("invalid_identity", [False, True])
def test_inference_sampling_topology_invalid_opt_in_fails_before_training_or_gateway_creation(
    monkeypatch, tmp_path, invalid_identity
):
    ctx = _inference_sampling_topology_context(
        (
            _inference_sampling_topology_ORDINARY_CONFIG
            if invalid_identity
            else _inference_sampling_topology_FLASH_CONFIG
        ),
        topology=(16, 8, 2),
    )
    if not invalid_identity:
        ctx.inference_sampling_topology = (16, 16, 1)
    ctx.workdir = tmp_path

    def forbidden(*args, **kwargs):
        pytest.fail("invalid sampling topology reached preparation, gateway or training")

    monkeypatch.setattr(check, "_full_model_name", forbidden)
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", forbidden)
    monkeypatch.setattr(check, "gateway", forbidden)
    monkeypatch.setattr(check, "running_job", forbidden)
    with pytest.raises(ValueError):
        check._run(ctx)


@pytest.mark.parametrize(
    "config_path,topology,expected_slots",
    [
        (_inference_sampling_topology_FLASH_CONFIG, (8, 8, 1), 16),
        (_inference_sampling_topology_FLASH_CONFIG, (16, 8, 2), 16),
        (_inference_sampling_topology_ORDINARY_CONFIG, None, 5),
    ],
)
def test_inference_sampling_topology_opt_in_gateway_reuses_training_capacity_only_after_destruction(
    monkeypatch, tmp_path, config_path, topology, expected_slots
):
    ctx = _inference_sampling_topology_context(config_path, topology=topology)
    ctx.workdir = tmp_path
    events = []
    steps = []
    rows = [[-2.0]]
    distribution = {"examples": 1, "minimum_tokens": 4, "median_tokens": 4, "maximum_tokens": 4}
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: object())
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda tokenizer: 0)
    monkeypatch.setattr(check.gsm8k, "read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda *args: [])
    monkeypatch.setattr(check.gsm8k, "length_distribution", lambda rows: distribution)
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: ([object()], 4))
    monkeypatch.setattr(check.gsm8k, "assert_disjoint", lambda *args: None)
    monkeypatch.setattr(check.gsm8k, "build_batch", lambda *args: object())
    monkeypatch.setattr(check, "training_batches", lambda *args: [object()] * check.TRAIN_STEPS)
    monkeypatch.setattr(check, "pack_logit_aligned", lambda batch: {})
    monkeypatch.setattr(check, "pack", lambda *args, **kwargs: {})
    monkeypatch.setattr(check, "build_payload", lambda *args, **kwargs: {"job_type": "training"})
    monkeypatch.setattr(check, "forward_logprobs", lambda *args: None)
    monkeypatch.setattr(check, "reference_answer_logprobs", lambda *args: rows)
    monkeypatch.setattr(check, "save_weights_only_checkpoint", lambda *args: tmp_path / "full-export")
    monkeypatch.setattr(check, "materialize_checkpoint_tokenizer", lambda *args: None)
    monkeypatch.setattr(check, "verdict", lambda *args: "synthetic-routing-only")

    def step(*args, **kwargs):
        steps.append(kwargs["learning_rate"])
        return SimpleNamespace(avg_loss=2.0)

    @contextmanager
    def gateway(path, slots, *, backend):
        assert slots == expected_slots
        assert isinstance(backend, OnPremConfig)
        events.append("gateway")
        yield object()

    @contextmanager
    def training(session, payload):
        assert payload["job_type"] == "training"
        events.append("training_start")
        try:
            yield object()
        finally:
            events.append("training_destroyed")

    def sampling(actual_ctx, session, weights_dir, token_budget, examples, validation_body):
        assert actual_ctx is ctx
        assert events == ["gateway", "training_start", "training_destroyed"]
        events.append("sampling")
        return (rows, rows)

    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", training)
    monkeypatch.setattr(check, "fwd_bwd_step", step)
    monkeypatch.setattr(check, "_onprem_sampling_and_reload", sampling)
    assert check._run(ctx) == "synthetic-routing-only"
    assert events == ["gateway", "training_start", "training_destroyed", "sampling"]
    assert steps == [check.configured_learning_rate(ctx.cfg)] * check.TRAIN_STEPS


# Inference text projection.


@pytest.fixture(params=["qwen3_5", "qwen3_5_moe"])
def _inference_text_projection_projection_case(tmp_path, monkeypatch, request):
    family = request.param
    composite_class, text_type, text_class = FAMILIES[family]
    text = {
        "model_type": text_type,
        "num_hidden_layers": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "use_cache": True,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {"rope_theta": 10000000},
    }
    if family.endswith("_moe"):
        text.update(num_experts=4, num_experts_per_tok=2)
    source_config = {
        "model_type": family,
        "architectures": [composite_class],
        "text_config": text,
        "vision_config": {"depth": 1},
    }
    exported_config = {
        **copy.deepcopy(text),
        "architectures": [text_class],
        "use_cache": False,
        "transformers_version": "synthetic-test",
    }
    tensors = {
        "model.language_model.embed_tokens.weight": torch.ones(8, 4, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(8, 4, dtype=torch.bfloat16),
        "model.language_model.layers.0.norm.weight": torch.ones(4, dtype=torch.float32),
        "model.language_model.layers.1.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "model.visual.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "mtp.fc.weight": torch.ones(4, 8, dtype=torch.bfloat16),
    }
    if family.endswith("_moe"):
        tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"] = torch.ones(4, 16, 4, dtype=torch.bfloat16)
    exported_tensors = {
        name: tensor.to(torch.bfloat16)
        for name, tensor in tensors.items()
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    source, exported = (tmp_path / "source", tmp_path / "export")
    source.mkdir()
    exported.mkdir()

    def write(root, config, values, *, original=False):
        (root / "config.json").write_text(json.dumps(config))
        save_file(values, str(root / "model.safetensors"))
        total = sum((tensor.numel() * tensor.element_size() for tensor in values.values()))
        index = {
            "metadata": {"total_size": float(total) if original else total},
            "weight_map": {name: "model.safetensors" for name in values},
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))

    write(source, source_config, tensors, original=True)
    write(exported, exported_config, exported_tensors)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(**source_config))

    def cached_file(model_name, filename, **kwargs):
        assert model_name == "Qwen/synthetic-composite"
        assert kwargs == {"local_files_only": True}
        return str(source / filename)

    monkeypatch.setattr(hub, "cached_file", cached_file)
    ctx = SimpleNamespace(cfg=SimpleNamespace(model_name="Qwen/synthetic-composite"))
    return SimpleNamespace(
        ctx=ctx,
        source=source,
        exported=exported,
        source_config=source_config,
        config=exported_config,
        tensors=exported_tensors,
        write=write,
    )


def test_inference_text_projection_local_full_text_export_is_validated_against_original_headers(
    _inference_text_projection_projection_case,
):
    case = _inference_text_projection_projection_case
    original = (case.source / "config.json").read_bytes()
    assert check.validate_local_inference_export(case.ctx, case.exported) == case.exported
    assert (case.source / "config.json").read_bytes() == original


@pytest.mark.parametrize("field", ["num_hidden_layers", "hidden_size", "rope_parameters", "dtype", "unknown"])
def test_inference_text_projection_local_text_export_rejects_model_configuration_drift(
    _inference_text_projection_projection_case, field
):
    case = _inference_text_projection_projection_case
    config = copy.deepcopy(case.config)
    if field == "rope_parameters":
        config[field]["rope_theta"] = 10000
    elif field == "dtype":
        config[field] = "float32"
    else:
        config[field] = 1
    case.write(case.exported, config, case.tensors)
    with pytest.raises(ValueError):
        check.validate_local_inference_export(case.ctx, case.exported)


@pytest.mark.parametrize("failure", ["missing", "extra", "vision", "mtp", "shape", "rename", "dtype"])
def test_inference_text_projection_local_text_export_rejects_incomplete_or_altered_weights(
    _inference_text_projection_projection_case, failure
):
    case = _inference_text_projection_projection_case
    tensors = dict(case.tensors)
    name = "model.language_model.layers.1.norm.weight"
    if failure == "missing":
        tensors.pop(name)
    elif failure in {"extra", "vision", "mtp"}:
        key = {
            "extra": "model.language_model.unexpected.weight",
            "vision": "model.visual.norm.weight",
            "mtp": "mtp.fc.weight",
        }[failure]
        tensors[key] = torch.ones(4, dtype=torch.bfloat16)
    elif failure == "shape":
        tensors[name] = tensors[name].reshape(2, 2)
    elif failure == "rename":
        tensors["model.layers.1.norm.weight"] = tensors.pop(name)
    else:
        tensors[name] = tensors[name].float()
    case.write(case.exported, case.config, tensors)
    with pytest.raises(ValueError):
        check.validate_local_inference_export(case.ctx, case.exported)


def test_inference_text_projection_local_text_export_requires_one_original_snapshot(
    _inference_text_projection_projection_case, tmp_path, monkeypatch
):
    case = _inference_text_projection_projection_case
    other = tmp_path / "other-snapshot"
    other.mkdir()
    index = other / "model.safetensors.index.json"
    index.write_bytes((case.source / index.name).read_bytes())
    monkeypatch.setattr(
        hub,
        "cached_file",
        lambda model, filename, **kwargs: str(index if filename == index.name else case.source / filename),
    )
    with pytest.raises(RuntimeError, match="one snapshot"):
        check.validate_local_inference_export(case.ctx, case.exported)


# Local cortex inference.

_local_cortex_inference_CONFIG = "arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"

_local_cortex_inference_FULL_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "tie_word_embeddings": False,
}

_local_cortex_inference_ROWS = [
    gsm8k.Example(question="validation0", prompt="p", completion=" a", input_ids=(1, 2, 3), n_prompt=1),
    gsm8k.Example(question="validation1", prompt="p", completion=" b", input_ids=(1, 4, 5), n_prompt=1),
]


class _local_cortex_inference_Tokenizer:
    special_tokens_map = {"eos_token": "eos"}

    def get_vocab(self):
        return {"p": 1, "a": 2, "b": 4, "eos": 5}

    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": self.get_vocab()}}))
        (path / "tokenizer_config.json").write_text(json.dumps(self.special_tokens_map))


@pytest.fixture
def _local_cortex_inference_export_factory(tmp_path, monkeypatch):
    source = tmp_path / "source-index.json"
    index = {"metadata": {"total_size": 4}, "weight_map": {"model.weight": "model.safetensors"}}
    source.write_text(json.dumps(index))
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(**_local_cortex_inference_FULL_CONFIG)
    )
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(source))
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda *args: _local_cortex_inference_Tokenizer())

    def create(path):
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps(_local_cortex_inference_FULL_CONFIG))
        (path / "model.safetensors.index.json").write_text(json.dumps(index))
        save_file({"model.weight": torch.ones(2, dtype=torch.bfloat16)}, str(path / "model.safetensors"))
        return path

    return create


def _local_cortex_inference_context(tmp_path, backend=None):
    cfg = load_config(_local_cortex_inference_CONFIG)
    return SimpleNamespace(
        cfg=cfg,
        config_id=cfg.config_id,
        client_backend=backend or CortexConfig(base_url="http://127.0.0.1:8003"),
        attn_implementation="sdpa",
        workdir=tmp_path,
        target_model_name="must-not-use-reduced",
        spec=SimpleNamespace(model=SimpleNamespace(cache_path="/unavailable-reduced")),
    )


def test_local_cortex_inference_local_sampling_uses_exact_export_same_backend_one_gpu_no_source(
    tmp_path, _local_cortex_inference_export_factory
):
    ctx = _local_cortex_inference_context(tmp_path)
    weights = _local_cortex_inference_export_factory(tmp_path / "global_step10")
    config = check.local_cortex_sampling_config(ctx, ctx.client_backend, weights, 10240)
    payload = config.to_cortex()[0]
    assert config.backend == ctx.client_backend
    assert config.hardware == "H200"
    assert payload["model_name"] == str(weights)
    assert payload["inference_config"]["n_gpus"] == 1
    assert config.source_checkpoint_info is None
    assert "source_checkpoint_info" not in payload


@pytest.mark.parametrize(
    "backend",
    [
        CortexConfig(base_url="https://remote"),
        CortexConfig(base_url="http://127.0.0.1:8003", pat="test-not-a-credential"),
        CortexConfig(base_url="http://127.0.0.1:8003/invalid"),
    ],
)
def test_local_cortex_inference_local_sampling_refuses_nonlocal_or_authenticated_backend(tmp_path, backend):
    with pytest.raises(ValueError, match="local Cortex"):
        check.local_cortex_sampling_config(
            _local_cortex_inference_context(tmp_path, backend), backend, tmp_path, 10240
        )


@pytest.mark.parametrize(
    "broken",
    [
        "reduced_config",
        "missing_config",
        "remote_code",
        "missing_index",
        "missing_tensor",
        "unsafe_shard",
        "missing_shard",
        "extra_shard",
        "symlink_shard",
        "wrong_dtype",
        "wrong_byte_count",
        "invalid_metadata",
        "truncated_weights",
    ],
)
def test_local_cortex_inference_local_export_rejects_incomplete_or_mismatched_weights(
    tmp_path, _local_cortex_inference_export_factory, broken
):
    weights = _local_cortex_inference_export_factory(tmp_path / "weights")
    if broken in {"reduced_config", "remote_code"}:
        config = dict(_local_cortex_inference_FULL_CONFIG)
        config.update({"num_hidden_layers": 4} if broken == "reduced_config" else {"auto_map": {"x": "y"}})
        (weights / "config.json").write_text(json.dumps(config))
    elif broken == "missing_config":
        (weights / "config.json").rename(weights / "missing-config")
    elif broken == "missing_index":
        (weights / "model.safetensors.index.json").rename(weights / "missing-index")
    elif broken in {"missing_tensor", "unsafe_shard", "wrong_byte_count"}:
        index = json.loads((weights / "model.safetensors.index.json").read_text())
        if broken == "missing_tensor":
            index["weight_map"] = {"different": "model.safetensors"}
        elif broken == "unsafe_shard":
            index["weight_map"]["model.weight"] = "../model.safetensors"
        else:
            index["metadata"]["total_size"] = 999
        (weights / "model.safetensors.index.json").write_text(json.dumps(index))
    elif broken == "missing_shard":
        (weights / "model.safetensors").rename(weights / "missing-shard")
    elif broken == "extra_shard":
        save_file({"x": torch.zeros(1)}, str(weights / "extra.safetensors"))
    elif broken == "symlink_shard":
        (weights / "model.safetensors").rename(tmp_path / "outside.safetensors")
        (weights / "model.safetensors").symlink_to(tmp_path / "outside.safetensors")
    elif broken == "wrong_dtype":
        save_file({"model.weight": torch.ones(2)}, str(weights / "model.safetensors"))
    elif broken == "invalid_metadata":
        (weights / "config.json").write_text("[]")
    elif broken == "truncated_weights":
        (weights / "model.safetensors").write_bytes(b"bad")
    with pytest.raises(Exception):
        check.validate_local_inference_export(_local_cortex_inference_context(tmp_path), weights)


@pytest.mark.parametrize("broken", ["path", "tag", "version", "wrong_step", "tokenizer"])
def test_local_cortex_inference_save_receipt_and_tokenizer_fail_closed(
    tmp_path, _local_cortex_inference_export_factory, monkeypatch, broken
):
    ctx = _local_cortex_inference_context(tmp_path)
    root = tmp_path / "save"
    root.mkdir()
    _local_cortex_inference_export_factory(root / "global_step10")
    monkeypatch.setattr(check, "correctness_workdir", lambda prefix: root)
    result = {"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10}
    if broken == "path":
        result["checkpoint_path"] = str(tmp_path)
    elif broken == "tag":
        result["checkpoint_tag"] = "cp_unsupported"
    elif broken == "version":
        result["version"] = 9
    elif broken == "wrong_step":
        result.update(checkpoint_tag="global_step9", version=9)
    elif broken == "tokenizer":

        class WrongTokenizer(_local_cortex_inference_Tokenizer):

            def get_vocab(self):
                return {"wrong": 999}

        monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: WrongTokenizer())
    client = SimpleNamespace(save_checkpoint=lambda **kwargs: result)
    with pytest.raises(RuntimeError):
        check.save_local_inference_checkpoint(ctx, ctx.client_backend, client, _local_cortex_inference_Tokenizer())
    assert root.exists()
    assert (root / "global_step10/model.safetensors").exists()


@pytest.mark.parametrize("malformed", [False, True])
def test_local_cortex_inference_local_execution_preserves_recipe_and_closes_owned_clients(
    tmp_path, _local_cortex_inference_export_factory, monkeypatch, malformed
):
    ctx = _local_cortex_inference_context(tmp_path)
    events = []
    saved = []

    def save(**kwargs):
        saved.append(kwargs)
        root = check.Path(kwargs["path"])
        _local_cortex_inference_export_factory(root / "global_step10")
        return {"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10}

    @contextmanager
    def gateway(directory, slots, **kwargs):
        assert kwargs["backend"] is ctx.client_backend
        assert slots == 5
        events.append("gateway-enter")
        try:
            yield "session"
        finally:
            events.append("gateway-exit")

    @contextmanager
    def training(session, payload):
        assert payload["model_name"] == "Qwen/Qwen3-8B"
        events.append("training-enter")
        try:
            yield SimpleNamespace(client=SimpleNamespace(save_checkpoint=save))
        finally:
            events.append("training-exit")

    @contextmanager
    def sampling(config):
        assert events[-1] == "training-exit"
        assert config.source_checkpoint_info is None
        assert config.to_cortex()[0]["model_name"].endswith("/global_step10")
        events.append("sampling-enter")
        results = [
            {
                "prompt_len": row.length,
                "prompt_logprobs": [None, *[{str(token): {"logprob": -1.0}} for token in row.input_ids[1:]]],
            }
            for row in _local_cortex_inference_ROWS
        ]
        if malformed:
            results[0]["prompt_logprobs"][1] = {999: -1.0}
        try:
            yield SimpleNamespace(generate=lambda *args, **kwargs: results)
        finally:
            events.append("sampling-exit")

    def read_rows(path, count=None):
        return (
            [
                gsm8k.Example(question=f"train{i}", prompt="p", completion=" a", input_ids=(1, 2, 3), n_prompt=1)
                for i in range(count)
            ]
            if count is not None
            else _local_cortex_inference_ROWS
        )

    learning_rates = []
    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", training)
    monkeypatch.setattr(check, "CorrectnessClient", sampling)
    monkeypatch.setattr(check.gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda tokenizer, rows: rows)
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda tokenizer: 0)
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: (_local_cortex_inference_ROWS, 3))
    monkeypatch.setattr(
        check,
        "fwd_bwd_step",
        lambda *args, **kwargs: learning_rates.append(kwargs["learning_rate"]) or SimpleNamespace(avg_loss=2.0),
    )
    monkeypatch.setattr(check, "forward_logprobs", lambda *args: torch.full((2, 3), -1.0))
    monkeypatch.setattr(check, "save_and_resolve_checkpoint_id", lambda *args: pytest.fail("no hosted catalog"))
    monkeypatch.setattr(check, "_onprem_sampling_and_reload", lambda *args: pytest.fail("no reload diagnostic"))
    result = check.run(ctx)[0]
    assert result.outcome is (TestOutcome.FAIL if malformed else TestOutcome.PASS)
    assert learning_rates == [check.configured_learning_rate(ctx.cfg)] * 10
    assert len(saved) == 1
    assert saved[0]["checkpoint_type"] == "weights-only"
    assert saved[0]["export_hf"] is False
    assert saved[0]["path"].startswith("/data-fast/")
    assert events == [
        "gateway-enter",
        "training-enter",
        "training-exit",
        "sampling-enter",
        "sampling-exit",
        "gateway-exit",
    ]
    if not malformed:
        assert result.metrics["train_steps"] == 10
        assert result.metrics["stated_criterion_abs"] == 0.001
        assert result.metrics["learning_rate"] == check.configured_learning_rate(ctx.cfg)


# Qwen4 inference schema.


@pytest.mark.parametrize("as_string", [False, True])
def test_inference_initial_model_dir_validates_without_model_materialization(tmp_path, as_string):
    root = tmp_path / "full-model"
    root.mkdir()
    value = str(root) if as_string else root
    assert export.validate_initial_training_model_dir(value) == root
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "invalid", ["missing", "file", "relative", "outside_fast", "symlink", "ancestor_symlink", "dotdot", "boolean"]
)
def test_inference_initial_model_dir_storage_refuses_unvalidated_paths(tmp_path, invalid):
    root = tmp_path / "full-model"
    root.mkdir()
    value = root
    if invalid == "missing":
        value = tmp_path / "missing"
    elif invalid == "file":
        value = tmp_path / "not-a-directory"
        value.write_bytes(b"synthetic")
    elif invalid == "relative":
        value = Path(root.name)
    elif invalid == "outside_fast":
        value = Path("/tmp")
    elif invalid == "symlink":
        value = tmp_path / "alias"
        value.symlink_to(root, target_is_directory=True)
    elif invalid == "ancestor_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        value = alias / root.name
    elif invalid == "dotdot":
        value = root / ".." / root.name
    elif invalid == "boolean":
        value = True
    with pytest.raises(ValueError):
        export.validate_initial_training_model_dir(value)


@pytest.mark.parametrize("enabled", [False, True])
def test_inference_initial_model_dir_public_training_subjob_preserves_canonical_model(tmp_path, enabled):
    root = tmp_path / "full-model"
    root.mkdir()
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    payload = rl_driver.build_payload({"n_gpus": 4, "max_seq_len": 128}, "Qwen/canonical-full", SEED)
    if enabled:
        payload["local_model_dir"] = str(root)
    before = deepcopy(payload)
    config = rl_driver._training_client_config(SimpleNamespace(backend=backend), payload)
    baseline = rl_driver._training_client_config(
        SimpleNamespace(backend=backend), {key: value for key, value in payload.items() if key != "local_model_dir"}
    ).to_cortex()[0]
    public = config.to_cortex()[0]
    assert payload == before
    assert config.model_name == "Qwen/canonical-full"
    assert public["model_name"] == "Qwen/canonical-full"
    assert public["training_config"]["n_gpus"] == 4
    assert config.initial_training_model_dir == (str(root) if enabled else None)
    if enabled:
        assert public.pop("local_model_dir") == str(root)
    else:
        assert "local_model_dir" not in public
    assert public == baseline
    assert "local_model_dir" not in public["training_config"]


@pytest.mark.parametrize(
    "invalid", ["onprem", "nonlocal", "authenticated", "no_training", "sampling", "log_prob", "checkpoint_source"]
)
def test_inference_initial_model_dir_typed_config_refuses_other_roles(tmp_path, invalid):
    root = tmp_path / "full-model"
    root.mkdir()
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    base = rl_driver._training_client_config(
        SimpleNamespace(backend=backend), rl_driver.build_payload({"n_gpus": 4}, "Qwen/canonical-full", SEED)
    ).model_dump()
    base["initial_training_model_dir"] = str(root)
    if invalid == "onprem":
        base["backend"] = OnPremConfig()
    elif invalid == "nonlocal":
        base["backend"] = CortexConfig(base_url="https://example.invalid")
    elif invalid == "authenticated":
        base["backend"] = CortexConfig(base_url="http://127.0.0.1:8003", pat="synthetic-not-a-credential")
    elif invalid == "no_training":
        base["training_gpus"] = 0
    elif invalid == "sampling":
        base["sampling_gpus"] = 1
    elif invalid == "log_prob":
        base["log_prob_gpus"] = 1
    elif invalid == "checkpoint_source":
        base["source_checkpoint_info"] = {"checkpoint_id": "cp_saved", "source_job_id": "owned-parent"}
    with pytest.raises(ValueError):
        CortexTrainingConfig.model_validate(base)


def test_inference_initial_model_dir_onprem_payload_refuses_silent_drop(tmp_path):
    root = tmp_path / "full-model"
    root.mkdir()
    payload = rl_driver.build_payload({"n_gpus": 4}, "Qwen/canonical-full", SEED)
    payload["local_model_dir"] = str(root)
    with pytest.raises(ValueError):
        rl_driver._training_client_config(SimpleNamespace(backend=OnPremConfig(), workdir=tmp_path), payload)


def test_inference_initial_model_dir_rl_context_keeps_config_identity(
    _inference_export_prescribed_hostfile, tmp_path, monkeypatch
):
    cfg = _rl_config_write(
        tmp_path,
        {"n_gpus": 16, "max_seq_len": 256, "attn_implementation": "sdpa"},
        {"n_gpus": 16, "max_seq_len": 256},
    )
    root = tmp_path / "full-model"
    root.mkdir()
    spec = SimpleNamespace(applicable_tests=["inference-checkpoint-loss"], arms=[], test_settings={})
    admission = _inference_export_prescribed_hostfile
    before = deepcopy(admission)
    original = cfg.native.model_dump()
    captured = []

    def inspect_context(ctx):
        captured.append(ctx)
        assert ctx.inference_initial_model_dir == root
        assert ctx.cfg.model_name == cfg.model_name
        assert ctx.cfg.native.model_dump() == original
        assert ctx.cfg.n_gpus == ctx.cfg.native.sampling_gpus == 16
        assert ctx.inference_export_allocation == before
        assert ctx.inference_export_allocation is not admission
        assert ctx.spec is spec
        return []

    monkeypatch.setattr(registered_tests()["inference-checkpoint-loss"], "fn", inspect_context)
    monkeypatch.setattr(runner, "gateway", lambda *a, **kw: pytest.fail("no native allocation"))
    assert (
        runner.execute_rl(
            cfg,
            spec,
            vocab_size=128,
            select=["inference-checkpoint-loss"],
            workdir=tmp_path,
            client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
            inference_export_allocation=admission,
            inference_initial_model_dir=root,
        )
        == []
    )
    assert len(captured) == 1
    assert admission == before
    assert cfg.native.model_dump() == original


@pytest.mark.parametrize("rl", [False, True])
def test_inference_initial_model_dir_runner_refuses_missing_allocation_before_work(tmp_path, monkeypatch, rl):
    cfg = _rl_config_write(tmp_path, {"n_gpus": 4}, {"n_gpus": 4} if rl else None)
    root = tmp_path / "full-model"
    root.mkdir()
    monkeypatch.setattr(runner, "correctness_workdir", lambda *a: pytest.fail("invalid root allocated work"))
    monkeypatch.setattr(runner, "registered_tests", lambda: pytest.fail("invalid root reached checks"))
    kwargs = {
        "vocab_size": 128,
        "select": ["inference-checkpoint-loss"],
        "client_backend": CortexConfig(base_url="http://127.0.0.1:8003"),
        "inference_initial_model_dir": root,
    }
    with pytest.raises(ValueError, match="allocation"):
        if rl:
            runner.execute_rl(cfg, SimpleNamespace(), **kwargs)
        else:
            runner.execute(cfg, SimpleNamespace(), [], **kwargs)


@pytest.mark.parametrize("invalid", ["onprem", "nonlocal", "authenticated", "missing_directory"])
def test_inference_initial_model_dir_check_rejects_before_tokenizer(
    _inference_export_prescribed_hostfile, tmp_path, monkeypatch, invalid
):
    root = tmp_path / "full-model"
    root.mkdir()
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    if invalid == "onprem":
        backend = OnPremConfig()
    elif invalid == "nonlocal":
        backend = CortexConfig(base_url="https://example.invalid")
    elif invalid == "authenticated":
        backend = CortexConfig(base_url="http://127.0.0.1:8003", pat="synthetic-not-a-credential")
    ctx = _local_cortex_inference_context(tmp_path, backend)
    ctx.inference_initial_model_dir = tmp_path / "missing" if invalid == "missing_directory" else root
    ctx.inference_export_allocation = _inference_export_prescribed_hostfile
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda *a: pytest.fail("invalid root loaded tokenizer"))
    monkeypatch.setattr(check, "gateway", lambda *a, **kw: pytest.fail("invalid root allocated gateway"))
    with pytest.raises(ValueError):
        check._run(ctx)


def test_inference_initial_model_dir_actual_check_binds_training_only_and_samples_export(
    _inference_export_prescribed_hostfile, _local_cortex_inference_export_factory, tmp_path, monkeypatch
):
    root = tmp_path / "initial-full-model"
    root.mkdir()
    ctx = _local_cortex_inference_context(tmp_path)
    ctx.inference_initial_model_dir = root
    ctx.inference_export_allocation = _inference_export_prescribed_hostfile
    weights = _local_cortex_inference_export_factory(tmp_path / "global_step10")
    original = ctx.cfg.native.model_dump()
    events = []
    steps = []
    tokenizers = []

    def load_tokenizer(model):
        tokenizers.append(model)
        return object()

    def read_rows(path, count=None):
        return (
            [
                gsm8k.Example(question=f"train{i}", prompt="p", completion=" a", input_ids=(1, 2, 3), n_prompt=1)
                for i in range(count)
            ]
            if count is not None
            else _local_cortex_inference_ROWS
        )

    @contextmanager
    def gateway(directory, slots, **kwargs):
        assert slots == 5
        assert kwargs["backend"] is ctx.client_backend
        yield SimpleNamespace(backend=ctx.client_backend)

    @contextmanager
    def training(session, payload):
        assert payload["model_name"] == ctx.cfg.model_name
        assert payload["local_model_dir"] == str(root)
        public = rl_driver._training_client_config(session, payload).to_cortex()[0]
        assert public["model_name"] == ctx.cfg.model_name
        assert public["local_model_dir"] == str(root)
        assert public["training_config"]["n_gpus"] == 4
        events.append("training-enter")
        try:
            yield SimpleNamespace(client=object())
        finally:
            events.append("training-exit")

    @contextmanager
    def sampling(config):
        assert events[-1] == "training-exit"
        assert config.model_name == str(weights)
        assert config.initial_training_model_dir is None
        assert config.source_checkpoint_info is None
        assert "local_model_dir" not in config.to_cortex()[0]
        events.append("sampling-enter")
        results = [
            {
                "prompt_len": row.length,
                "prompt_logprobs": [None, *[{str(token): {"logprob": -1.0}} for token in row.input_ids[1:]]],
            }
            for row in _local_cortex_inference_ROWS
        ]
        try:
            yield SimpleNamespace(generate=lambda *a, **kw: results)
        finally:
            events.append("sampling-exit")

    monkeypatch.setattr(check.gsm8k, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(check.gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda tokenizer, rows: rows)
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda tokenizer: 0)
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *a: (_local_cortex_inference_ROWS, 3))
    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", training)
    monkeypatch.setattr(check, "CorrectnessClient", sampling)
    monkeypatch.setattr(
        check,
        "fwd_bwd_step",
        lambda *a, **kw: steps.append(kw["learning_rate"]) or SimpleNamespace(avg_loss=2.0),
    )
    monkeypatch.setattr(check, "forward_logprobs", lambda *a: torch.full((2, 3), -1.0))
    monkeypatch.setattr(check, "save_local_inference_checkpoint", lambda *a: weights)
    monkeypatch.setattr(export, "require_local_inference_export_locality", lambda actual_ctx, path: None)
    result = check._run(ctx)
    assert result.outcome is TestOutcome.PASS
    assert result.metrics["train_steps"] == 10
    assert result.metrics["stated_criterion_abs"] == 0.001
    assert tokenizers == [ctx.cfg.model_name]
    assert steps == [check.configured_learning_rate(ctx.cfg)] * 10
    assert ctx.cfg.native.model_dump() == original
    assert events == ["training-enter", "training-exit", "sampling-enter", "sampling-exit"]


_qwen4_inference_schema_PREFIX = "model.language_model.layers.0.ple.ple_embedding."


@pytest.fixture
def _qwen4_inference_schema_config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "dtype": "bfloat16",
        "transformers_version": "5.17.0",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "use_cache": True,
            "rope_parameters": {"rope_theta": 1000000.0},
            "ple_config": {"ngram_orders": [2, 3], "ngram_vocab_sizes": [8, 16]},
        },
        "vision_config": {"model_type": "qwen4_exp_vision", "depth": 2},
    }


def test_qwen4_inference_schema_full_nested_config_preserved_without_mutating_source(_qwen4_inference_schema_config):
    original = copy.deepcopy(_qwen4_inference_schema_config)
    target = copy.deepcopy(_qwen4_inference_schema_config)
    target["transformers_version"] = "5.18.0"
    target["text_config"]["use_cache"] = False
    assert export.validate_qwen4_full_config(_qwen4_inference_schema_config, target) is None
    assert _qwen4_inference_schema_config == original


@pytest.mark.parametrize(
    "path",
    [
        ("model_type",),
        ("architectures",),
        ("dtype",),
        ("text_config", "model_type"),
        ("text_config", "hidden_size"),
        ("text_config", "num_hidden_layers"),
        ("text_config", "rope_parameters", "rope_theta"),
        ("text_config", "ple_config", "ngram_orders"),
        ("text_config", "ple_config", "ngram_vocab_sizes"),
        ("vision_config", "model_type"),
        ("vision_config", "depth"),
    ],
)
@pytest.mark.parametrize("failure", ["changed", "missing"])
def test_qwen4_inference_schema_full_config_rejects_changed_or_missing_original_semantics(
    _qwen4_inference_schema_config, path, failure
):
    target = copy.deepcopy(_qwen4_inference_schema_config)
    container = target
    for name in path[:-1]:
        container = container[name]
    if failure == "missing":
        container.pop(path[-1])
    else:
        container[path[-1]] = None
    with pytest.raises(ValueError, match="configuration (differs|field missing)"):
        export.validate_qwen4_full_config(_qwen4_inference_schema_config, target)


@pytest.mark.parametrize("path", [(), ("text_config",), ("vision_config",), ("text_config", "ple_config")])
def test_qwen4_inference_schema_full_config_rejects_remote_code_at_each_nested_level(
    _qwen4_inference_schema_config, path
):
    target = copy.deepcopy(_qwen4_inference_schema_config)
    container = target
    for name in path:
        container = container[name]
    container["auto_map"] = {"AutoConfig": "remote.Configuration"}
    with pytest.raises(ValueError, match="without remote code"):
        export.validate_qwen4_full_config(_qwen4_inference_schema_config, target)


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_qwen4_inference_schema_full_config_cache_setting_must_remain_an_explicit_boolean(
    _qwen4_inference_schema_config, value
):
    target = copy.deepcopy(_qwen4_inference_schema_config)
    target["text_config"]["use_cache"] = value
    with pytest.raises(ValueError, match="explicit boolean"):
        export.validate_qwen4_full_config(_qwen4_inference_schema_config, target)


def test_qwen4_inference_schema_full_config_rejects_json_scalar_type_substitution(_qwen4_inference_schema_config):
    target = copy.deepcopy(_qwen4_inference_schema_config)
    target["text_config"]["num_hidden_layers"] = True
    with pytest.raises(ValueError, match="configuration differs"):
        export.validate_qwen4_full_config(_qwen4_inference_schema_config, target)


@pytest.mark.parametrize("path", [(), ("text_config",), ("vision_config",), ("text_config", "ple_config")])
@pytest.mark.parametrize("name", ["quantization_config", "unknown_model_option", "extra_remote_config"])
def test_qwen4_inference_schema_full_config_rejects_export_only_semantic_fields(
    _qwen4_inference_schema_config, path, name
):
    target = copy.deepcopy(_qwen4_inference_schema_config)
    container = target
    for key in path:
        container = container[key]
    container[name] = {"auto_map": {"AutoConfig": "remote.Configuration"}}
    with pytest.raises(ValueError, match="unexpected Qwen4 configuration fields"):
        export.validate_qwen4_full_config(_qwen4_inference_schema_config, target)


def test_qwen4_inference_schema_full_config_accepts_only_optional_serializer_and_boolean_cache_fields(
    _qwen4_inference_schema_config,
):
    target = copy.deepcopy(_qwen4_inference_schema_config)
    target.pop("transformers_version")
    target["use_cache"] = False
    target["vision_config"]["transformers_version"] = "5.18.0"
    target["vision_config"]["use_cache"] = True
    assert export.validate_qwen4_full_config(_qwen4_inference_schema_config, target) is None


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_qwen4_inference_schema_full_config_new_cache_setting_must_be_boolean(_qwen4_inference_schema_config, value):
    target = copy.deepcopy(_qwen4_inference_schema_config)
    target["use_cache"] = value
    with pytest.raises(ValueError, match="explicit boolean"):
        export.validate_qwen4_full_config(_qwen4_inference_schema_config, target)


@pytest.fixture
def _qwen4_inference_schema_schema():
    return {
        "model.language_model.layers.0.norm.weight": ["BF16", [4]],
        "model.language_model.embed_tokens.weight": ["BF16", [8, 4]],
        **{_qwen4_inference_schema_PREFIX + name: ["I64", [2]] for name in export.QWEN4_INTEGER_BUFFERS},
    }


def test_qwen4_inference_schema_original_integer_buffer_headers_preserve_dtype_shape_and_bytes(
    _qwen4_inference_schema_schema,
):
    original = copy.deepcopy(_qwen4_inference_schema_schema)
    assert (
        export.validate_qwen4_full_schema(
            _qwen4_inference_schema_schema, copy.deepcopy(_qwen4_inference_schema_schema)
        )
        == 120
    )
    assert export.schema_bytes({_qwen4_inference_schema_PREFIX + "layer_multipliers": ["I64", []]}) == 8
    assert _qwen4_inference_schema_schema == original


@pytest.mark.parametrize("name", sorted(export.QWEN4_INTEGER_BUFFERS))
@pytest.mark.parametrize("failure", ["missing", "float", "shape"])
def test_qwen4_inference_schema_each_integer_buffer_is_required_without_cast_or_reshape(
    _qwen4_inference_schema_schema, name, failure
):
    target = copy.deepcopy(_qwen4_inference_schema_schema)
    key = _qwen4_inference_schema_PREFIX + name
    if failure == "missing":
        target.pop(key)
    elif failure == "float":
        target[key][0] = "BF16"
    else:
        target[key][1] = [1, 2]
    with pytest.raises(ValueError, match="exact original"):
        export.validate_qwen4_full_schema(_qwen4_inference_schema_schema, target)


@pytest.mark.parametrize("failure", ["extra", "rename", "floating-weight", "unrecognized-integer", "buffer-source"])
def test_qwen4_inference_schema_qwen4_full_inventory_remains_strict(_qwen4_inference_schema_schema, failure):
    target = copy.deepcopy(_qwen4_inference_schema_schema)
    if failure == "extra":
        target["unexpected.weight"] = ["BF16", [1]]
    elif failure == "rename":
        target[_qwen4_inference_schema_PREFIX + "renamed"] = target.pop(
            _qwen4_inference_schema_PREFIX + "layer_multipliers"
        )
    elif failure == "floating-weight":
        target["model.language_model.layers.0.norm.weight"][0] = "F32"
    elif failure == "unrecognized-integer":
        _qwen4_inference_schema_schema["unrecognized.integer"] = ["I64", [2]]
        target["unrecognized.integer"] = ["I64", [2]]
    else:
        _qwen4_inference_schema_schema[_qwen4_inference_schema_PREFIX + "layer_multipliers"][0] = "F32"
        target[_qwen4_inference_schema_PREFIX + "layer_multipliers"][0] = "BF16"
    with pytest.raises(ValueError):
        export.validate_qwen4_full_schema(_qwen4_inference_schema_schema, target)


def test_qwen4_inference_schema_floating_source_projection_preserves_integer_buffers(_qwen4_inference_schema_schema):
    target = copy.deepcopy(_qwen4_inference_schema_schema)
    _qwen4_inference_schema_schema["model.language_model.layers.0.norm.weight"][0] = "F32"
    assert export.validate_qwen4_full_schema(_qwen4_inference_schema_schema, target) == 120


@pytest.mark.parametrize("failure", [None, "text", "vision", "ple", "remote-code"])
def test_qwen4_inference_schema_local_qwen4_export_validates_real_mixed_headers(
    tmp_path, monkeypatch, _qwen4_inference_schema_config, failure
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    values = {
        "model.language_model.layers.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        **{
            _qwen4_inference_schema_PREFIX + name: torch.tensor([2**53 + 1, 2**53 + 3], dtype=torch.int64)
            for name in export.QWEN4_INTEGER_BUFFERS
        },
    }
    for root in (source, target):
        root.mkdir()
        (root / "config.json").write_text(json.dumps(_qwen4_inference_schema_config))
        save_file(values, str(root / "weights.safetensors"))
        index = {
            "metadata": {"total_size": sum((value.numel() * value.element_size() for value in values.values()))},
            "weight_map": {name: "weights.safetensors" for name in values},
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(**_qwen4_inference_schema_config)
    )
    monkeypatch.setattr(hub, "cached_file", lambda model, name, **kwargs: str(source / name))
    ctx = SimpleNamespace(cfg=SimpleNamespace(model_name="Qwen/synthetic-qwen4"))
    before = {name: value.clone() for name, value in values.items()}
    if failure is None:
        assert check.validate_local_inference_export(ctx, target) == target
    else:
        exported = copy.deepcopy(_qwen4_inference_schema_config)
        if failure == "text":
            exported["text_config"]["rope_parameters"]["rope_theta"] = 1.0
        elif failure == "vision":
            exported["vision_config"]["depth"] = 1
        elif failure == "ple":
            exported["text_config"]["ple_config"]["ngram_vocab_sizes"] = [16, 8]
        else:
            exported["text_config"]["auto_map"] = {"AutoConfig": "remote.Configuration"}
        (target / "config.json").write_text(json.dumps(exported))
        with pytest.raises(ValueError):
            check.validate_local_inference_export(ctx, target)
    for name, value in values.items():
        assert torch.equal(value, before[name])


# Rl config.


def test_inference_export_allocation_rl_context_preserves_declared_width_and_deepcopy(
    _inference_export_prescribed_hostfile, monkeypatch, tmp_path
):
    cfg = _rl_config_write(
        tmp_path,
        {"n_gpus": 16, "max_seq_len": 256, "attn_implementation": "sdpa"},
        {"n_gpus": 16, "max_seq_len": 256},
    )
    spec = SimpleNamespace(applicable_tests=["inference-checkpoint-loss"], arms=[], test_settings={})
    admission = _inference_export_prescribed_hostfile
    before = deepcopy(admission)
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    captured = []
    monkeypatch.setattr(runner, "require_data_fast_path", lambda path, **kwargs: Path(path))
    monkeypatch.setattr(runner, "gateway", lambda *a, **kw: pytest.fail("no gateway allocation"))

    def inspect_context(ctx):
        captured.append(ctx)
        assert ctx.cfg.n_gpus == 16
        assert ctx.cfg.native.sampling_gpus == 16
        assert ctx.cfg.model_name == cfg.model_name
        assert ctx.client_backend is backend
        assert ctx.inference_export_allocation == before
        assert ctx.inference_export_allocation is not admission
        assert ctx.inference_export_allocation["nodes"] is not admission["nodes"]
        assert ctx.inference_export_allocation["nodes"][0] is not admission["nodes"][0]
        ctx.inference_export_allocation["nodes"][0]["slots"] = -1
        return []

    monkeypatch.setattr(registered_tests()["inference-checkpoint-loss"], "fn", inspect_context)
    assert (
        runner.execute_rl(
            cfg,
            spec,
            vocab_size=128,
            select=["inference-checkpoint-loss"],
            workdir=tmp_path,
            client_backend=backend,
            inference_export_allocation=admission,
        )
        == []
    )
    assert len(captured) == 1
    assert admission == before


@pytest.mark.parametrize(
    "invalid", ["onprem", "nonlocal", "authenticated", "other_check", "all_checks", "mixed_checks"]
)
def test_inference_export_allocation_rl_refuses_unowned_route_before_work(
    _inference_export_prescribed_hostfile, monkeypatch, invalid
):
    backend = CortexConfig(base_url="http://localhost:8003")
    selected = ["inference-checkpoint-loss"]
    if invalid == "onprem":
        backend = OnPremConfig()
    elif invalid == "nonlocal":
        backend = CortexConfig(base_url="https://unowned.invalid")
    elif invalid == "authenticated":
        backend = CortexConfig(base_url="http://localhost:8003", pat="synthetic-test-token")
    elif invalid == "other_check":
        selected = ["single-step-grads"]
    elif invalid == "all_checks":
        selected = None
    elif invalid == "mixed_checks":
        selected = ["inference-checkpoint-loss", "rl-weight-sync"]
    monkeypatch.setattr(runner, "correctness_workdir", lambda *a: pytest.fail("invalid route allocated a workdir"))
    monkeypatch.setattr(runner, "registered_tests", lambda: pytest.fail("invalid route reached model/check setup"))
    with pytest.raises(ValueError):
        runner.execute_rl(
            None,
            None,
            vocab_size=128,
            select=selected,
            client_backend=backend,
            inference_export_allocation=_inference_export_prescribed_hostfile,
        )


def test_inference_export_allocation_rl_checks_live_hostfile_before_work(
    _inference_export_prescribed_hostfile, monkeypatch
):
    admission = _inference_export_prescribed_hostfile
    before = deepcopy(admission)
    export._LOCAL_HOSTFILE_TARGET.write_text("127.0.0.1 slots=4\n")
    monkeypatch.setattr(runner, "correctness_workdir", lambda *a: pytest.fail("drifted allocation created a workdir"))
    monkeypatch.setattr(runner, "registered_tests", lambda: pytest.fail("drifted allocation reached checks"))
    with pytest.raises(ValueError, match="hostfile"):
        runner.execute_rl(
            None,
            None,
            vocab_size=128,
            select=["inference-checkpoint-loss"],
            client_backend=CortexConfig(base_url="http://localhost:8003"),
            inference_export_allocation=admission,
        )
    assert admission == before


_rl_config_ROOT = Path(__file__).resolve().parents[1]

_rl_config_CONFIG = _rl_config_ROOT / "configs/qwen3.6-35b-a3b/h200/train-rl-16gpus-64k.config"

_rl_config_SPEC = _rl_config_ROOT / "specs/qwen3.6-35b-a3b-h200-train-rl-16gpus-64k.json"

_rl_config_TRAINING_CONFIG = {"max_seq_len": 4096, "n_gpus": 8, "attn_implementation": "flash_attention_3"}

_rl_config_SAMPLING_CONFIG = {"max_seq_len": 4096, "n_gpus": 8}


def _rl_config_write(tmp_path, training: dict, sampling: dict | None = None):
    path = tmp_path / "job.config"
    path.write_text(json.dumps(_config_factory_native_config(training, sampling=sampling)))
    return load_config(path)


def test_rl_config_training_alone_is_not_rl(tmp_path):
    assert _rl_config_write(tmp_path, _rl_config_TRAINING_CONFIG).is_rl is False


def test_rl_config_training_with_sampling_is_rl(tmp_path):
    assert _rl_config_write(tmp_path, _rl_config_TRAINING_CONFIG, _rl_config_SAMPLING_CONFIG).is_rl is True


def test_rl_config_prime_rl_provider_alone_is_not_rl(tmp_path):
    """The training sub-job's own options never decide this; a supervised job may select PrimeRL."""
    training = json.loads(json.dumps(_rl_config_TRAINING_CONFIG))
    training["model_provider"] = "prime_rl"
    training["prime_rl"] = {"fused_cross_entropy": False}
    assert _rl_config_write(tmp_path, training).is_rl is False


def test_rl_config_every_reference_check_is_registered():
    """The checks an RL config is refused for are exactly the registered ones that need the reference.

    Not an equality: a check that compares Arctic Platform against Arctic Platform declares ``compares_to_reference=False`` and is
    registered without belonging to this set.
    """
    assert set(reference_tests()) <= set(registered_tests())


@pytest.mark.parametrize("check", ["rl-weight-sync", "rl-router-replay"])
def test_rl_config_frozen_rl_checks_do_not_require_the_reduced_reference(check):
    cfg = load_config(_rl_config_CONFIG)
    spec = Spec.read(_rl_config_SPEC)
    assert cfg.is_rl
    assert cfg.model_name == spec.model.source_checkpoint == "Qwen/Qwen3.6-35B-A3B"
    assert spec.model.cache_path == "/data-fast/base-models/synthetic/Qwen3.6-35B-A3B-4L"
    assert spec.model.num_hidden_layers == 4
    assert check in spec.applicable_tests
    assert registered_tests()[check].compares_to_reference is False
    assert check not in reference_tests()


@pytest.mark.parametrize("check", ["rl-weight-sync", "rl-router-replay"])
def test_rl_config_frozen_rl_cortex_capability_guard_precedes_check_setup(tmp_path, monkeypatch, check):
    cfg = load_config(_rl_config_CONFIG)
    spec = Spec.read(_rl_config_SPEC)
    entry = registered_tests()[check]
    monkeypatch.setattr(entry, "fn", lambda ctx: pytest.fail("unsupported check must not load a model or client"))
    results = runner.execute_rl(
        cfg,
        spec,
        vocab_size=128,
        select=[check],
        workdir=tmp_path,
        client_backend=CortexConfig(base_url="http://unused"),
    )
    assert len(results) == 1
    assert results[0].outcome is Outcome.FAIL
    assert results[0].summary == "check not supported on Cortex"
    assert results[0].reason == runner.cortex_unavailable_reason(entry)
    assert "worker-local" in results[0].reason


# Rl checks.


def test_rl_checks_rl_shutdown_failure_is_unsafe(monkeypatch, tmp_path):
    from arctic_platform.correctness.harness.dss_driver import UnsafeCleanupError

    failure = RuntimeError("RL cancel failed")
    cfg = load_config(_rl_config_CONFIG)

    class Client:

        def __init__(self, config):
            pass

        def shutdown(self):
            raise failure

    monkeypatch.setattr(rl_driver, "RLClient", Client)
    payload = {"training_config": cfg.training, "model_name": cfg.model_name}
    with pytest.raises(UnsafeCleanupError) as caught:
        with rl_driver.running_rl_client(cfg, payload, cfg.model_name, tmp_path):
            pass
    assert caught.value.__cause__ is failure


def test_rl_checks_rl_execution_retains_traceback_for_unwrapped_check_failure(monkeypatch, tmp_path, capsys):
    from arctic_platform.correctness.harness import runner
    from arctic_platform.correctness.harness.registry import registered_tests

    cfg = load_config(_rl_config_CONFIG)
    spec = Spec.read(_rl_config_SPEC)

    def fail(ctx):
        raise RuntimeError("synthetic check failed before returning")

    monkeypatch.setattr(registered_tests()["rl-weight-sync"], "fn", fail)
    results = runner.execute_rl(cfg, spec, vocab_size=128, select=["rl-weight-sync"], workdir=tmp_path)
    assert len(results) == 1 and results[0].outcome is Outcome.FAIL
    assert results[0].error_type == "RuntimeError"
    assert "RuntimeError: synthetic check failed before returning" in capsys.readouterr().err


def _rl_checks_entry(numel, digest):
    return {"shape": [numel], "dtype": "bfloat16", "numel": numel, "sha256": digest}


def _rl_checks_sync(ranks=2):
    sent = {"a.weight": _rl_checks_entry(4, "aa"), "b.weight": _rl_checks_entry(6, "bb")}
    record = {
        "status": "ok",
        "received": {name: {**entry, "destinations": [name]} for name, entry in sent.items()},
        "destinations": {
            "a.weight": {"numel": 2, "uncovered": 0, "mismatched": 0, "equal": True},
            "b.weight": {"numel": 3, "uncovered": 0, "mismatched": 0, "equal": True},
        },
        "untouched_parameters": {},
        "non_parameter_destinations": [],
        "unverifiable_destinations": [],
    }
    return {
        "weight_format": "vllm",
        "targets": [
            {
                "send": {
                    "results": [
                        {"sent_manifest": sent, "trainer_parameters": {"count": 2, "numel": 10}},
                        {"sent_manifest": None},
                    ]
                },
                "recv": {
                    "recv": {
                        "workers": [
                            {"all_rank_weight_sync_verification": [copy.deepcopy(record) for _ in range(ranks)]}
                        ]
                    }
                },
            }
        ],
    }


def _rl_checks_rank(sync, index):
    return sync["targets"][0]["recv"]["recv"]["workers"][0]["all_rank_weight_sync_verification"][index]


def test_rl_checks_weight_sync_agreement_passes():
    failed, metrics = rl_weight_sync.judge(_rl_checks_sync())
    assert failed == []
    assert metrics["sent_elements"] == metrics["trainer_elements"] == 10
    assert metrics["sampler_ranks"] == 2


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda s: s["targets"][0]["send"]["results"][0]["trainer_parameters"].update(numel=12),
            "trainer elements sent",
        ),
        (lambda s: _rl_checks_rank(s, 1)["received"].pop("b.weight"), "sampler rank 1: b.weight not received"),
        (lambda s: _rl_checks_rank(s, 0)["received"]["a.weight"].update(sha256="xx"), "different bytes"),
        (lambda s: _rl_checks_rank(s, 0)["received"]["a.weight"].update(destinations=[]), "written to no parameter"),
        (lambda s: _rl_checks_rank(s, 0)["destinations"]["a.weight"].update(uncovered=1), "no tensor wrote"),
        (
            lambda s: _rl_checks_rank(s, 0)["destinations"]["b.weight"].update(mismatched=2),
            "differing from what was written",
        ),
        (lambda s: _rl_checks_rank(s, 0)["untouched_parameters"].update({"visual.w": 5}), "written by no tensor"),
    ],
)
def test_rl_checks_weight_sync_disagreement_fails(mutate, expected):
    sync = _rl_checks_sync()
    mutate(sync)
    failed, _ = rl_weight_sync.judge(sync)
    assert any((expected in m.name for m in failed)), [m.name for m in failed]


def test_rl_checks_weight_sync_without_sampler_verification_raises():
    sync = _rl_checks_sync()
    sync["targets"][0]["recv"] = {"recv": {"workers": [{}]}}
    with pytest.raises(RuntimeError, match=rl_weight_sync.VERIFY_ENV):
        rl_weight_sync.judge(sync)


def _rl_checks_samples():
    rows = [([5, 6, 7], 2, [[[1], [2]], [[3], [4]]]), ([8, 9, 4], 1, [[[5], [6]], [[7], [0]]])]
    samples = []
    for index, (tokens, prompt_len, routing) in enumerate(rows):
        marker = {
            "replay_id": f"rr1:{index}",
            "prompt_len": prompt_len,
            "generation_len": len(tokens) - prompt_len,
            "capture_len": len(routing),
            "routed_experts": routing,
        }
        samples.append({"tokens": tokens, "prompt_len": prompt_len, "marker": marker, "routing": routing})
    return samples


def _rl_checks_trace(samples):
    tokens = samples[0]["tokens"] + samples[1]["tokens"]
    positions = [0, 1, 2, 0, 1, 2]
    routed = copy.deepcopy(samples[0]["routing"] + [[[0], [0]]] + samples[1]["routing"] + [[[0], [0]]])
    router = [
        {"layer": layer, "phase": phase, "replayed": True, "indices": [row[layer] for row in routed]}
        for phase in ("forward", "recompute")
        for layer in (0, 1)
    ]
    return {
        "rank": 0,
        "received": {s["marker"]["replay_id"]: s["routing"] for s in samples},
        "calls": [
            {
                "padding": False,
                "input_ids": tokens,
                "position_ids": positions,
                "routed_experts": routed,
                "router": router,
            }
        ],
        "unattributed_router_calls": [],
    }


def test_rl_checks_router_replay_agreement_passes():
    samples = _rl_checks_samples()
    failed, metrics = rl_router_replay.judge(samples, [_rl_checks_trace(samples)], full_recompute=True)
    assert failed == []
    assert metrics["compared_positions"] == 4
    assert metrics["forward_invocations"] == metrics["recompute_invocations"] == 2


def test_rl_checks_router_trace_hooks_external_router_implementations():
    import torch

    class TokenChoiceTopKRouter(torch.nn.Module):

        def forward(self, hidden, *, routed_experts=None):
            indices = routed_experts if routed_experts is not None else torch.zeros_like(hidden, dtype=torch.long)
            return (hidden, indices, torch.ones(1))

    class Layer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.router = TokenChoiceTopKRouter()

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    model = Model()
    trace = router_replay_trace.start(model)
    trace.begin_call(
        {
            "input_ids": torch.tensor([[1]]),
            "position_ids": torch.tensor([[0]]),
            "routed_experts": torch.tensor([[[0]]]),
        },
        padding=False,
    )
    model.layers[0].router(torch.zeros(1, 1), routed_experts=torch.tensor([[3]]))
    exported = router_replay_trace.finish(0)
    assert exported["calls"][0]["router"] == [{"layer": 0, "phase": "forward", "replayed": True, "indices": [[3]]}]
    model.layers[0].router(torch.zeros(1, 1), routed_experts=torch.tensor([[2]]))
    assert len(exported["calls"][0]["router"]) == 1


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda t: t["received"].update({"rr1:1": [[[5], [6]], [[7], [1]]]}), "routing received differs"),
        (lambda t: t["calls"][0]["routed_experts"][1].__setitem__(0, [9]), "routed_experts differ"),
        (lambda t: t["calls"][0]["router"][3].update(indices=[[9]] * 6), "layer 1 recompute tokens"),
        (lambda t: t["calls"][0]["router"][0].update(replayed=False), "computed its own routing"),
        (lambda t: t["calls"][0]["router"].pop(2), "layer 0 has no recompute"),
        (lambda t: t["unattributed_router_calls"].append({"layer": 0}), "outside a traced call"),
        (lambda t: t["received"].pop("rr1:0"), "sample 0: no training rank received"),
    ],
)
def test_rl_checks_router_replay_disagreement_fails(mutate, expected):
    samples = _rl_checks_samples()
    trace = _rl_checks_trace(samples)
    mutate(trace)
    failed, _ = rl_router_replay.judge(samples, [trace], full_recompute=True)
    assert any((expected in m.name for m in failed)), [m.name for m in failed]


def test_rl_checks_router_replay_recompute_not_required_without_full_checkpointing():
    samples = _rl_checks_samples()
    trace = _rl_checks_trace(samples)
    trace["calls"][0]["router"] = [r for r in trace["calls"][0]["router"] if r["phase"] == "forward"]
    failed, _ = rl_router_replay.judge(samples, [trace], full_recompute=False)
    assert failed == []
    assert rl_router_replay.expects_full_recompute({"ac_config": {"mode": "full", "freq": 1}})
    assert not rl_router_replay.expects_full_recompute({})


def test_rl_checks_rl_batch_scores_only_answer_predictions():
    batch = rl_batch([[5, 6, 7, 8], [9, 4]], [2, 1], pad_id=0)
    assert batch["kwargs"]["input_ids"].tolist() == [[5, 6, 7, 8], [9, 4, 0, 0]]
    assert batch["kwargs"]["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["context"]["loss_mask"].tolist() == [[0, 1, 1, 0], [1, 0, 0, 0]]


def test_rl_checks_rl_batch_uses_native_scalar_temperature_boundary():
    request = rl_batch([[5, 6, 7, 8], [9, 4]], [2, 1], pad_id=0)
    _, batch, meta, processing = unpack_batch(request)
    assert "temperature" not in request["kwargs"]
    assert request["context"]["temperature"] == meta["temperature"] == 1.0
    assert batch["input_ids"].tolist() == [[5, 6, 7, 8], [9, 4, 0, 0]]
    assert batch["loss_mask"].tolist() == [[0, 1, 1, 0], [1, 0, 0, 0]]
    assert batch["advantages"].eq(1).all()
    assert processing == {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": {"loss_agg_mode": "token-mean"}}


@pytest.mark.parametrize("check", [rl_weight_sync, rl_router_replay], ids=["weight-sync", "router-replay"])
@pytest.mark.parametrize("local_model", [False, True], ids=["frozen-catalog-model", "configured-local-model"])
def test_rl_checks_rl_checks_use_configured_model_for_tokenizer_and_both_zones(
    tmp_path, monkeypatch, check, local_model
):
    cfg = load_config(_rl_config_CONFIG)
    if local_model:
        model_name = "/data-fast/base-models/full-configured-model"
        sub_job = {**cfg.sub_job, "model_name": model_name}
        cfg = replace(
            cfg,
            sub_job=sub_job,
            sub_jobs=[{**job, "model_name": model_name} for job in cfg.sub_jobs],
            native=cfg.native.model_copy(update={"model_name": model_name}, deep=True),
        )
    spec = Spec.read(_rl_config_SPEC)
    original_native = cfg.native.model_dump()
    original_training = copy.deepcopy(cfg.training)
    original_sampling = copy.deepcopy(cfg.sub_jobs[1])
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        config_id=cfg.config_id,
        workdir=tmp_path,
        attn_implementation=cfg.attention_implementation,
        target_model_name="published-reduced-reference",
    )
    assert cfg.model_name != spec.model.cache_path
    tokenizer_paths, configs, batches, generations, rates, closed = ([], [], [], [], [], [])
    rows_per_step = check.configured_rows_per_step(cfg)
    examples = [
        check.gsm8k.Example(
            question=str(i),
            prompt="prompt",
            completion="answer",
            input_ids=(3 * i + 5, 3 * i + 6, 3 * i + 7),
            n_prompt=2,
        )
        for i in range(rows_per_step)
    ]
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    generated_samples = []

    def load_tokenizer(path):
        tokenizer_paths.append(path)
        return tokenizer

    def read_rows(path, *, count):
        assert path == check.TRAIN_SPLIT
        assert count == rows_per_step
        return [("question", "answer")] * count

    def tokenize_examples(actual_tokenizer, rows):
        assert actual_tokenizer is tokenizer
        assert len(rows) == rows_per_step
        return examples

    class Client:
        jobs = SimpleNamespace(sampling="native-sampling-job")

        def __init__(self, config):
            configs.append(config)

        def generate(self, prompts, params):
            generations.append((prompts, params))
            results = []
            for example, prompt, parameters in zip(examples, prompts, params, strict=True):
                assert prompt == list(example.input_ids[: example.n_prompt])
                assert parameters["max_tokens"] == example.answer_tokens
                assert parameters["dss_return_back_router_info"] is True
                marker = {
                    "sample_id": parameters["dss_sample_id"],
                    "replay_id": parameters["dss_router_replay_id"],
                    "prompt_len": len(prompt),
                    "generation_len": 1,
                    "capture_len": 2,
                    "routed_experts": [[[1], [2]], [[3], [4]]],
                }
                result = {"token_ids": list(example.input_ids[example.n_prompt :]), "router_replay": marker}
                results.append(result)
                generated_samples.append(rl_router_replay.sample_from_result(prompt, result))
            return results

        def fwd_bwd(self, batch, *, router_replay=None):
            batches.append((batch, router_replay))
            traces = [_rl_checks_trace(generated_samples[i : i + 2]) for i in range(0, len(generated_samples), 2)]
            return {"avg_loss": 1.0, "router_replay_trace": traces}

        def step(self, learning_rate):
            rates.append(learning_rate)
            return {}

        def sync_weights(self):
            assert check is rl_weight_sync
            return _rl_checks_sync()

        def shutdown(self):
            closed.append(True)

    monkeypatch.setattr(check.gsm8k, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(check.gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", tokenize_examples)
    monkeypatch.setattr(rl_driver, "RLClient", Client)
    monkeypatch.setenv(rl_weight_sync.VERIFY_ENV, "test-original")
    result = check._run(ctx)
    assert result.outcome is Outcome.PASS
    assert tokenizer_paths == [cfg.model_name]
    assert len(configs) == len(batches) == len(closed) == 1
    native = configs[0]
    assert native.model_name == cfg.model_name
    assert native.to_onprem("training")["model_name"] == native.to_onprem("sampling")["model_name"] == cfg.model_name
    assert native.sampling.model_dump() == cfg.native.sampling.model_dump()
    assert native.training_gpus == cfg.native.training_gpus == 8
    assert native.sampling_gpus == cfg.native.sampling_gpus == 8
    assert native.max_seq_len == cfg.native.max_seq_len == 32768
    assert native.training.ds_config == cfg.native.training.ds_config
    assert native.training.ds_worker_config["model_provider"] == "prime_rl"
    assert native.training.ds_worker_config["attn_implementation"] == cfg.attention_implementation
    assert native.training.checkpoint_path == str(tmp_path / check.TEST_ID / "gateway" / "checkpoints")
    batch, replay = batches[0]
    assert batch["kwargs"]["input_ids"].tolist() == [list(example.input_ids) for example in examples]
    if check is rl_weight_sync:
        assert native.training.ds_worker_config["debug"]["weight_sync_manifest"] is True
        assert replay is None
        assert rates == [check.configured_learning_rate(cfg)]
        assert generations == []
    else:
        assert native.training.ds_worker_config["debug"]["router_replay_trace"] is True
        assert replay == {"sampling_job_id": "native-sampling-job", "identity_version": 1}
        assert len(generations) == 1
        assert rates == []
        assert batch["kwargs"]["rr_sample_ids"] == [sample["marker"]["replay_id"] for sample in generated_samples]
        assert batch["kwargs"]["rr_discard"] is True
    assert cfg.native.model_dump() == original_native
    assert cfg.training == original_training
    assert cfg.sub_jobs[1] == original_sampling


def test_rl_checks_sampling_payload_keeps_the_config_and_swaps_only_the_model(tmp_path):
    path = tmp_path / "job.config"
    path.write_text(
        json.dumps(_config_factory_native_config(_rl_config_TRAINING_CONFIG, sampling=_rl_config_SAMPLING_CONFIG))
    )
    cfg = load_config(path)
    payload = rl_sampling_payload(cfg, "/models/reduced")
    assert payload == {
        "job_type": "sampling",
        "model_name": "/models/reduced",
        "dtype": "bfloat16",
        "seed": 42,
        "inference_config": {"n_gpus": 8, "max_seq_len": 4096, "vllm_config": {}},
    }
    assert rl_gpu_count(cfg) == 16
