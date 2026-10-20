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
import json
import os
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


# Inference sampling topology.


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
    monkeypatch.setattr(check, "_sampling_payload", observe)
    config = check.local_cortex_sampling_config(ctx, backend, weights, _inference_sampling_topology_TOKEN_BUDGET)
    assert events == ["validate", "helper"]
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
