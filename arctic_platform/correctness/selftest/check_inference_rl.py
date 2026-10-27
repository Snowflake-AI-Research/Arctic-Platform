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
import errno
import hashlib
import importlib.util
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
from types import ModuleType
from types import SimpleNamespace

import pytest
import safetensors
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.utils import hub

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client.config import ArcticClientConfig
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
from arctic_platform.correctness.harness import storage
from arctic_platform.correctness.harness.batches import SEED
from arctic_platform.correctness.harness.config import TestSpec as Spec
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.cortex import CortexTrainingConfig
from arctic_platform.correctness.harness.dss_driver import rl_batch
from arctic_platform.correctness.harness.dss_driver import rl_gpu_count
from arctic_platform.correctness.harness.dss_driver import rl_sampling_payload
from arctic_platform.correctness.harness.dss_driver import sampling_payload
from arctic_platform.correctness.harness.inference_export import FAMILIES
from arctic_platform.correctness.harness.registry import TestOutcome
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import reference_tests
from arctic_platform.correctness.harness.registry import registered_tests
from arctic_platform.debug import broadcast_weight_sync_evidence as broadcast_evidence
from arctic_platform.debug import inference_startup_error as startup_error_observer
from arctic_platform.debug import inference_warmup_observer as warmup_observer
from arctic_platform.debug import router_replay_trace as replay_evidence

from .helpers import _config_factory_native_config

# Inference checkpoint loss.


_inference_checkpoint_loss_ROWS = [
    gsm8k.Example(
        question="q0",
        prompt="Question: q0\nAnswer:",
        completion=" a0",
        input_ids=(11, 12, 13, 14, 15, 16),
        n_prompt=3,
    ),
    gsm8k.Example(
        question="q1",
        prompt="Question: q1\nAnswer:",
        completion=" a1",
        input_ids=(21, 22, 23, 24),
        n_prompt=2,
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


def test_inference_checkpoint_loss_exported_checkpoint_reuses_the_validated_tokenizer(
    tmp_path,
) -> None:

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


def _processor_metadata_case(tmp_path, monkeypatch, *, model_type="qwen3_5_moe", video=True):
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    config = (json.dumps({"model_type": model_type}, indent=2) + "\n").encode()
    index = b'{"metadata":{"total_size":2},"weight_map":{"model.weight":"model.safetensors"}}\n'
    image = b'{\n  "do_resize": true,\n  "size": {"shortest_edge": 504}\n}\n'
    (source / "config.json").write_bytes(config)
    (source / "model.safetensors.index.json").write_bytes(index)
    (source / "preprocessor_config.json").write_bytes(image)
    if video:
        (source / "video_preprocessor_config.json").write_bytes(b'{"fps":2,"do_resize":false}\n')
    monkeypatch.setattr(export, "validate_initial_training_model_dir", lambda value: Path(value))
    return SimpleNamespace(inference_initial_model_dir=source), source, checkpoint


def test_processor_metadata_noncomposite_and_absent_source_are_unchanged(tmp_path, monkeypatch):
    absent = SimpleNamespace()
    checkpoint = tmp_path / "absent"
    checkpoint.mkdir()
    check.materialize_checkpoint_processor_metadata(absent, checkpoint)
    assert list(checkpoint.iterdir()) == []

    ctx, source, checkpoint = _processor_metadata_case(tmp_path / "ordinary", monkeypatch, model_type="qwen3")
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    check.materialize_checkpoint_processor_metadata(ctx, checkpoint)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert list(checkpoint.iterdir()) == []


def test_processor_metadata_copies_exact_mandatory_and_optional_bytes_without_touching_tokenizer(
    tmp_path, monkeypatch
):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch)
    tokenizer = b'{"model":{"vocab":{"exact":0}}}\n'
    checkpoint.joinpath("tokenizer.json").write_bytes(tokenizer)
    expected = {
        name: source.joinpath(name).read_bytes()
        for name in ("preprocessor_config.json", "video_preprocessor_config.json")
    }

    check.materialize_checkpoint_processor_metadata(ctx, checkpoint)

    assert {name: checkpoint.joinpath(name).read_bytes() for name in expected} == expected
    assert checkpoint.joinpath("tokenizer.json").read_bytes() == tokenizer
    assert all(checkpoint.joinpath(name).stat().st_mode & 0o777 == 0o600 for name in expected)


def test_processor_metadata_accepts_absent_optional_video_metadata(tmp_path, monkeypatch):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch, video=False)
    check.materialize_checkpoint_processor_metadata(ctx, checkpoint)
    assert (
        checkpoint.joinpath("preprocessor_config.json").read_bytes()
        == source.joinpath("preprocessor_config.json").read_bytes()
    )
    assert not checkpoint.joinpath("video_preprocessor_config.json").exists()


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing", "preprocessor_config"),
        ("duplicate", "duplicate"),
        ("nonobject", "object"),
        ("oversized", "bound"),
        ("source-symlink", "regular"),
        ("config-symlink", "regular"),
        ("index-symlink", "regular"),
        ("existing", "already exists"),
        ("destination-symlink", "already exists"),
    ],
)
def test_processor_metadata_rejects_unavailable_or_unsafe_snapshot_bytes(tmp_path, monkeypatch, damage, match):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch)
    image = source / "preprocessor_config.json"
    if damage == "missing":
        image.unlink()
    elif damage == "duplicate":
        image.write_text('{"size":1,"size":2}')
    elif damage == "nonobject":
        image.write_text("[]")
    elif damage == "oversized":
        monkeypatch.setattr(check, "_PROCESSOR_JSON_LIMIT_BYTES", 16)
        image.write_text('{"long":"' + "x" * 32 + '"}')
    elif damage == "source-symlink":
        stored = source / "stored.json"
        image.rename(stored)
        image.symlink_to(stored.name)
    elif damage in {"config-symlink", "index-symlink"}:
        name = "config.json" if damage == "config-symlink" else "model.safetensors.index.json"
        path = source / name
        stored = source / ("stored-" + name)
        path.rename(stored)
        path.symlink_to(stored.name)
    elif damage == "existing":
        (checkpoint / "preprocessor_config.json").write_text('{"existing":true}')
    else:
        target = tmp_path / "outside.json"
        target.write_text('{"outside":true}')
        (checkpoint / "preprocessor_config.json").symlink_to(target)

    with pytest.raises((RuntimeError, ValueError), match=match):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)


def test_processor_metadata_rejects_source_file_generation_change(tmp_path, monkeypatch):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch, video=False)
    original = check._read_processor_metadata

    def mutate(fd, name):
        value = original(fd, name)
        if name == "preprocessor_config.json":
            (source / name).write_bytes(value + b" ")
        return value

    monkeypatch.setattr(check, "_read_processor_metadata", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)


def test_processor_metadata_rejects_source_parent_generation_change(tmp_path, monkeypatch):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch, video=False)
    original = check._copy_processor_metadata_file

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        source.rename(source.with_name("source-held"))
        source.mkdir()
        return result

    monkeypatch.setattr(check, "_copy_processor_metadata_file", mutate)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)


def test_processor_metadata_rejects_destination_parent_replacement(tmp_path, monkeypatch):
    ctx, _, checkpoint = _processor_metadata_case(tmp_path, monkeypatch, video=False)
    original = check._copy_processor_metadata_file

    def replace(*args, **kwargs):
        result = original(*args, **kwargs)
        checkpoint.rename(checkpoint.with_name("checkpoint-held"))
        checkpoint.mkdir()
        return result

    monkeypatch.setattr(check, "_copy_processor_metadata_file", replace)
    with pytest.raises(RuntimeError, match="destination changed"):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)


def test_processor_metadata_rejects_in_place_destination_and_noncomposite_config_changes(tmp_path, monkeypatch):
    ctx, _, checkpoint = _processor_metadata_case(tmp_path / "destination", monkeypatch, video=False)
    original_copy = check._copy_processor_metadata_file

    def rewrite_destination(*args, **kwargs):
        result = original_copy(*args, **kwargs)
        destination_fd = result[0]
        os.ftruncate(destination_fd, 0)
        os.write(destination_fd, b'{"corrupted":true}\n')
        os.fsync(destination_fd)
        return result

    monkeypatch.setattr(check, "_copy_processor_metadata_file", rewrite_destination)
    with pytest.raises(RuntimeError, match="destination changed"):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)

    ctx, source, checkpoint = _processor_metadata_case(
        tmp_path / "noncomposite", monkeypatch, model_type="qwen3", video=False
    )
    original_read = check._read_processor_metadata
    changed = False

    def rewrite_config(fd, name):
        nonlocal changed
        raw = original_read(fd, name)
        if name == "config.json" and not changed:
            changed = True
            source.joinpath(name).write_bytes(raw.replace(b"qwen3", b"qwen4"))
        return raw

    monkeypatch.setattr(check, "_read_processor_metadata", rewrite_config)
    with pytest.raises(RuntimeError, match="source changed"):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)


def test_processor_metadata_existing_destination_accepts_equal_mandatory_and_optional_files(tmp_path, monkeypatch):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch)
    names = ("preprocessor_config.json", "video_preprocessor_config.json")
    before = {}
    for name in names:
        destination = checkpoint / name
        destination.write_bytes((source / name).read_bytes())
        destination.chmod(0o640)
        before[name] = (
            destination.stat().st_ino,
            destination.stat().st_mode,
            destination.read_bytes(),
        )

    check.materialize_checkpoint_processor_metadata(ctx, checkpoint)

    assert {
        name: (
            checkpoint.joinpath(name).stat().st_ino,
            checkpoint.joinpath(name).stat().st_mode,
            checkpoint.joinpath(name).read_bytes(),
        )
        for name in names
    } == before


@pytest.mark.parametrize("damage", ["different", "duplicate", "symlink"])
def test_processor_metadata_existing_destination_rejects_unsafe_or_different_file(tmp_path, monkeypatch, damage):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch, video=False)
    destination = checkpoint / "preprocessor_config.json"
    if damage == "different":
        destination.write_bytes(b'{"different":true}\n')
    elif damage == "duplicate":
        destination.write_bytes(b'{"size":1,"size":2}\n')
    else:
        target = tmp_path / "outside.json"
        target.write_bytes((source / destination.name).read_bytes())
        destination.symlink_to(target)

    with pytest.raises((RuntimeError, ValueError)):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)


@pytest.mark.parametrize("mutation", ["replacement", "in-place"])
def test_processor_metadata_existing_destination_rejects_change_after_acceptance(tmp_path, monkeypatch, mutation):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch, video=False)
    name = "preprocessor_config.json"
    destination = checkpoint / name
    destination.write_bytes((source / name).read_bytes())
    original = check._copy_processor_metadata_file

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if mutation == "replacement":
            destination.rename(checkpoint / "held-preprocessor.json")
            destination.write_bytes(result[2])
        else:
            with destination.open("r+b") as stream:
                stream.seek(0)
                stream.write(b'{"changed":true}\n')
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
        return result

    monkeypatch.setattr(check, "_copy_processor_metadata_file", mutate)
    with pytest.raises(RuntimeError, match="destination changed"):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)


def test_processor_metadata_existing_destination_survives_rollback_of_later_created_file(tmp_path, monkeypatch):
    ctx, source, checkpoint = _processor_metadata_case(tmp_path, monkeypatch)
    mandatory = checkpoint / "preprocessor_config.json"
    mandatory.write_bytes((source / mandatory.name).read_bytes())
    mandatory.chmod(0o640)
    before = (mandatory.stat().st_ino, mandatory.stat().st_mode, mandatory.read_bytes())
    original = check._copy_processor_metadata_file

    def mutate(source_fd, destination_parent_fd, name):
        result = original(source_fd, destination_parent_fd, name)
        if name == "video_preprocessor_config.json":
            source.joinpath("config.json").write_text('{"model_type":"qwen3_5_moe","changed":true}\n')
        return result

    monkeypatch.setattr(check, "_copy_processor_metadata_file", mutate)
    with pytest.raises(RuntimeError, match="source changed"):
        check.materialize_checkpoint_processor_metadata(ctx, checkpoint)

    assert (
        mandatory.stat().st_ino,
        mandatory.stat().st_mode,
        mandatory.read_bytes(),
    ) == before
    assert not checkpoint.joinpath("video_preprocessor_config.json").exists()


def test_processor_metadata_save_completes_after_tokenizer_identity_and_before_publication(monkeypatch, tmp_path):
    events = []
    source = tmp_path / "source"
    root = tmp_path / "save"
    weights = root / "global_step10"
    source.mkdir()
    weights.mkdir(parents=True)
    ctx = SimpleNamespace(
        inference_initial_model_dir=source,
        inference_export_allocation={"admitted": True},
    )
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    tokenizer = SimpleNamespace(get_vocab=lambda: {"exact": 0}, special_tokens_map={"eos_token": "exact"})
    monkeypatch.setattr(check, "correctness_workdir", lambda prefix: root)
    monkeypatch.setattr(check, "_local_checkpoint_metadata", lambda *args: "global_step10")
    monkeypatch.setattr(check, "validate_local_inference_export", lambda ctx, path: path)
    monkeypatch.setattr(
        check,
        "materialize_checkpoint_tokenizer",
        lambda tokenizer, path: events.append("tokenizer"),
    )
    monkeypatch.setattr(check, "_export_json", lambda path: {})
    monkeypatch.setattr(
        check.gsm8k,
        "load_tokenizer",
        lambda path: events.append("identity") or tokenizer,
    )
    monkeypatch.setattr(
        check,
        "materialize_checkpoint_processor_metadata",
        lambda actual, path: events.append("processor"),
    )
    monkeypatch.setattr(
        export,
        "distribute_local_inference_export",
        lambda path, allocation: events.append("publication") or {"complete": True},
    )
    client = SimpleNamespace(save_checkpoint=lambda **kwargs: {"checkpoint_tag": "global_step10"})

    assert check.save_local_inference_checkpoint(ctx, backend, client, tokenizer) == weights
    assert events == ["tokenizer", "identity", "processor", "publication"]


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
            question=f"q{index}",
            prompt="p",
            completion=" a",
            input_ids=(index + 1, index + 2, index + 3),
            n_prompt=1,
        )
        for index in range(check.TRAIN_STEPS * rows_per_step)
    ]
    batches = check.training_batches(examples, rows_per_step, pad_id=0)
    assert len(batches) == check.TRAIN_STEPS
    assert all((batch.rows == rows_per_step for batch in batches))
    seen = [tuple(row) for batch in batches for row in batch.input_ids.tolist()]
    assert len(set(seen)) == len(examples)


@pytest.mark.parametrize(
    "composite,ngram,kernel,expected",
    [(True, 3, 4, 144), (False, 2, 7, 192), (True, 5, 1, 64)],
)
def test_inference_ple_padding_floor_uses_effective_offline_config(monkeypatch, composite, ngram, kernel, expected):
    text = SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=48,
        ngram_size=ngram,
        ple_conv_kernel_size=kernel,
    )
    source = SimpleNamespace(model_type="qwen4_exp", text_config=text) if composite else text
    cfg = SimpleNamespace(
        config_id=check._FLASHNEXT_CONFIG_ID,
        sp_size=16,
        training={"model_provider": "prime_rl"},
    )
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
    text = SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=48,
        ngram_size=3,
        ple_conv_kernel_size=4,
    )
    setattr(text, field, value)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: text)
    cfg = SimpleNamespace(
        config_id=check._FLASHNEXT_CONFIG_ID,
        sp_size=16,
        training={"model_provider": "prime_rl"},
    )
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
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("unaffected route config read"),
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
            assert torch.equal(
                getattr(padded, field)[index, : row.length],
                getattr(batch, field)[index, : row.length],
            )
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
            packed["labels"][packed["labels"] != IGNORE_INDEX],
            expected_labels[expected_labels != IGNORE_INDEX],
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
        training={
            "model_provider": "prime_rl",
            "train_batch_size": 1,
            "optimizer": {"lr": 2e-5},
        },
    )
    ctx = SimpleNamespace(cfg=cfg, workdir=tmp_path, attn_implementation="qsa_flex")
    monkeypatch.setattr(
        check.gsm8k,
        "load_tokenizer",
        lambda path: SimpleNamespace(pad_token_id=0, eos_token_id=0),
    )
    monkeypatch.setattr(
        check.gsm8k,
        "read_rows",
        lambda split, **kwargs: train if split == check.TRAIN_SPLIT else validation,
    )
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda tokenizer, rows: rows)
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: (validation, 4))
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="qwen4_exp_text",
            num_hidden_layers=48,
            ngram_size=3,
            ple_conv_kernel_size=4,
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
    results[0]["prompt_logprobs"][1] = {
        999: {"logprob": -99.0},
        _inference_checkpoint_loss_ROWS[0].input_ids[1]: -0.0,
    }
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
    "boundary",
    [
        0,
        -1,
        _inference_checkpoint_loss_ROWS[0].length,
        _inference_checkpoint_loss_ROWS[0].length + 1,
    ],
)
def test_inference_checkpoint_loss_cortex_rejects_invalid_answer_boundaries(boundary):
    examples = [
        replace(_inference_checkpoint_loss_ROWS[0], n_prompt=boundary),
        _inference_checkpoint_loss_ROWS[1],
    ]
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
def test_inference_checkpoint_loss_target_extraction_does_not_silently_truncate_rows(
    results,
):
    with pytest.raises(RuntimeError, match="validation examples"):
        check.target_answer_logprobs(results, _inference_checkpoint_loss_ROWS)


@pytest.mark.parametrize("shape", [(1, 6), (3, 6), (2, 6, 1)])
def test_inference_checkpoint_loss_reference_extraction_requires_exact_rows_and_rank(
    shape,
):
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
    monkeypatch.setattr(
        check.gsm8k,
        "load_tokenizer",
        lambda *args: pytest.fail("must reject before tokenizer load"),
    )
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
        schema["model.language_model.layers.0.mlp.experts.gate_up_proj"] = [
            "BF16",
            [4, 16, 4],
        ]
    output = {
        name: ["BF16", list(row[1])]
        for name, row in schema.items()
        if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    return (source, exported, schema, output)


def test_inference_export_named_projection_uses_original_not_job12_dimensions(
    _inference_export_case,
):
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


def test_inference_export_unchanged_composite_delegates_without_mutation(
    _inference_export_case,
):
    source = _inference_export_case[0]
    exported = copy.deepcopy(source)
    original = copy.deepcopy((source, exported))
    assert projection.validate_text_projection(source, exported, None, None) is None
    assert (source, exported) == original


def test_inference_export_scalar_text_tensor_retains_exact_shape_and_bytes(
    _inference_export_case,
):
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


def test_inference_export_additional_model_fields_are_not_exempt(
    _inference_export_case,
):
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
    index = {
        "weight_map": {"model.norm.weight": path.name},
        "metadata": {"total_size": 8},
    }
    return SimpleNamespace(root=root, path=path, schema=schema, keys=keys, index=index)


@pytest.mark.parametrize("count", [8, 8.0])
def test_inference_export_header_inventory_without_values(_inference_export_headers, count):
    _inference_export_headers.index["metadata"]["total_size"] = count
    assert (
        projection.read_inventory(_inference_export_headers.root, _inference_export_headers.index)
        == _inference_export_headers.schema
    )


def test_inference_export_scalar_header_inventory_without_values(
    _inference_export_headers,
):
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


@pytest.mark.parametrize(
    "failure",
    ["unsafe-shard", "missing-key", "duplicate-key", "wrong-bytes", "symlink"],
)
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
    distribution = {
        "examples": 1,
        "minimum_tokens": 4,
        "median_tokens": 4,
        "maximum_tokens": 4,
    }
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
    assert _inference_rendezvous_inference_run.jobs == [
        "training",
        "sampling",
        "training",
    ]
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
        lambda *args, **kwargs: (
            [{"training": 1}],
            {"processing": {"return_logprobs": True}},
        ),
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

    def bind(payload, path, *, native_backend=None):
        assert path == adapter
        assert native_backend is None
        captured.append(payload)
        return payload

    monkeypatch.setattr(lora, "bind_payload", bind)
    actual, identity = e2e._prepare_adapter(ctx, cfg.model_name, tmp_path)
    assert actual == adapter
    assert identity == {
        "kind": "deterministic-lora",
        "metadata": {"mock-adapter": str(adapter)},
    }
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
    assert [payload["job_type"] for payload in captured] == [
        "training",
        "sampling",
        "training",
    ]
    _reviewed_weight_format_callers_assert_payload(
        captured[0],
        run.context.cfg,
        run.context.cfg.model_name,
        gradient_norms_per_param=False,
    )
    _reviewed_weight_format_callers_assert_payload(
        captured[2],
        run.context.cfg,
        str(tmp_path / "hf"),
        gradient_norms_per_param=False,
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
    save_file(
        {"model.weight": torch.zeros((2, 2), dtype=torch.bfloat16)},
        root / "model.safetensors",
    )
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


@pytest.fixture
def _inference_export_locality_adapter(_inference_export_locality_saved):
    root = _inference_export_locality_saved / "default"
    root.mkdir()
    (root / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 2, "target_modules": ["proj"]})
    )
    save_file(
        {
            "base_model.model.proj.lora_A.weight": torch.full((2, 4), 0.25),
            "base_model.model.proj.lora_B.weight": torch.full((4, 2), 0.5),
        },
        root / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    (root / "README.md").write_text("Trained adapter\n")
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
    _inference_export_prescribed_hostfile,
    _inference_export_locality_saved,
    monkeypatch,
    tmp_path,
    change,
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
    _inference_export_integration_runner,
    _inference_export_prescribed_hostfile,
    monkeypatch,
    tmp_path,
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
    _inference_export_integration_runner,
    _inference_export_prescribed_hostfile,
    monkeypatch,
    invalid,
):
    cfg, spec = _inference_export_integration_runner
    backend = (
        OnPremConfig()
        if invalid == "onprem"
        else CortexConfig(
            base_url=("http://other-job.invalid" if invalid == "nonlocal_cortex" else "http://localhost:8003")
        )
    )
    selected = ["single-step-grads"] if invalid == "other_check" else ["inference-checkpoint-loss"]
    monkeypatch.setattr(
        runner,
        "registered_tests",
        lambda: pytest.fail("invalid admission must stop before test or model setup"),
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
    assert set(manifest["tokenizer_files"]) == {
        "tokenizer.json",
        "tokenizer_config.json",
    }


def test_inference_export_locality_adapter_manifest_preserves_all_bytes_and_tensor_headers(
    _inference_export_locality_adapter,
):
    adapter = _inference_export_locality_adapter
    fingerprints = {}
    manifest = export._local_manifest(adapter.parent, time.monotonic() + 30, fingerprints)
    assert {row["path"] for row in manifest["files"]} == {
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "default/adapter_config.json",
        "default/adapter_model.safetensors",
        "default/README.md",
    }
    for row in manifest["files"]:
        assert row["sha256"] == hashlib.sha256((adapter.parent / row["path"]).read_bytes()).hexdigest()
    assert manifest["adapter"]["tensor_payload_bytes"] == 64
    assert manifest["adapter"]["tensor_schema_sha256"] == export.schema_sha(
        {
            "base_model.model.proj.lora_A.weight": ["F32", [2, 4]],
            "base_model.model.proj.lora_B.weight": ["F32", [4, 2]],
        }
    )
    assert fingerprints["default"] == list(export._fingerprint(adapter))


@pytest.mark.parametrize(
    "damage",
    [
        "missing_config",
        "empty_weights",
        "wrong_type",
        "symlink",
        "nested",
        "extra_file",
        "wrapped_keys",
    ],
)
def test_inference_export_locality_rejects_incomplete_or_unportable_adapter(
    _inference_export_locality_adapter,
    damage,
):
    adapter = _inference_export_locality_adapter
    if damage == "missing_config":
        (adapter / "adapter_config.json").unlink()
    elif damage == "empty_weights":
        save_file({}, adapter / "adapter_model.safetensors")
    elif damage == "wrong_type":
        (adapter / "adapter_config.json").write_text('{"peft_type":"OTHER"}')
    elif damage == "symlink":
        (adapter / "adapter_config.json").unlink()
        (adapter / "adapter_config.json").symlink_to(adapter.parent / "config.json")
    elif damage == "nested":
        (adapter / "another").mkdir()
    elif damage == "extra_file":
        (adapter / "unexpected.bin").write_bytes(b"unexpected")
    else:
        save_file(
            {"base_model.model.proj._checkpoint_wrapped_module.lora_A.weight": torch.ones((2, 4))},
            adapter / "adapter_model.safetensors",
        )
    with pytest.raises((ValueError, OSError)):
        _inference_export_locality_manifest(adapter.parent)


@pytest.mark.parametrize(
    "damage",
    ["missing_tokenizer", "symlink", "directory", "extra_shard", "wrong_index"],
)
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
            json.dumps(
                {
                    "weight_map": {"other": "model.safetensors"},
                    "metadata": {"total_size": 8},
                }
            )
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


@pytest.mark.parametrize("with_adapter", [False, True])
def test_inference_export_locality_receiver_publishes_verified_files_exclusively(
    monkeypatch,
    tmp_path,
    _inference_export_locality_saved,
    request,
    with_adapter,
):
    if with_adapter:
        request.getfixturevalue("_inference_export_locality_adapter")
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
    assert receipt["files_verified"] == (8 if with_adapter else 5)
    assert json.loads((destination / export._LOCAL_RECEIPT).read_text()) == receipt
    for row in manifest["files"]:
        assert (destination / row["path"]).read_bytes() == (root / row["path"]).read_bytes()
    monkeypatch.setattr(export.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(wire)))
    with pytest.raises(FileExistsError):
        export._receive_local_export(request, time.monotonic() + 30)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "default/../escape",
        "other/weights",
        "default/nested/weights",
        "default/unknown",
        "default/..",
        "./config.json",
    ],
)
def test_inference_export_locality_receiver_rejects_paths_outside_flat_export_and_default_adapter(
    monkeypatch,
    tmp_path,
    _inference_export_locality_saved,
    name,
):
    source = _inference_export_locality_saved
    manifest = _inference_export_locality_manifest(source)
    manifest["files"][0]["path"] = name
    destination = tmp_path / "invalid-worker-save"
    request = {
        "node": _inference_export_locality_node(),
        "root": str(destination),
        "allocation_id": "test-allocation",
        "manifest_sha256": export.schema_sha(manifest),
    }
    monkeypatch.setattr(
        export.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(json.dumps(manifest).encode() + b"\n")),
    )
    with pytest.raises(ValueError, match="invalid export file manifest"):
        export._receive_local_export(request, time.monotonic() + 30)
    assert not destination.exists()


@pytest.mark.parametrize(
    "damage",
    ["truncated", "corrupted", "extra_bytes", "manifest_identity", "node_identity"],
)
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
    with export._local_ssh_argv(admission, node, {"timeout_s": 5}, time.monotonic() + 5) as argv:
        assert argv[-2] == node["host"]
        assert "StrictHostKeyChecking=yes" in argv
        assert "BatchMode=yes" in argv
        held_path = next(value.split("=", 1)[1] for value in argv if value.startswith("UserKnownHostsFile="))
        assert Path(held_path).read_bytes() == known.read_bytes()
    known.write_text("changed-keys\n")
    with pytest.raises(ValueError, match="strict SSH"):
        with export._local_ssh_argv(admission, node, {}, time.monotonic() + 5):
            pytest.fail("changed trust reached consumer")


def test_inference_export_locality_bounded_child_timeout_reaps_only_created_session(
    monkeypatch,
):
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


@pytest.mark.parametrize("change", ["weights", "config", "extra_file", "replacement"])
def test_inference_export_locality_adapter_mutation_blocks_sampling(
    monkeypatch,
    _inference_export_locality_adapter,
    _inference_export_locality_admission,
    change,
):
    adapter = _inference_export_locality_adapter
    root = adapter.parent
    proof = export.distribute_local_inference_export(root, _inference_export_locality_admission, timeout_s=30)
    ctx = SimpleNamespace(
        inference_export_allocation=_inference_export_locality_admission,
        _inference_export_locality=proof,
    )
    export.require_local_inference_export_locality(ctx, root)
    if change == "weights":
        save_file({"changed": torch.ones((2, 4))}, adapter / "adapter_model.safetensors")
    elif change == "config":
        (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":4}')
    elif change == "extra_file":
        (adapter / "unexpected.bin").write_bytes(b"unexpected")
    else:
        adapter.rename(root / "previous")
        adapter.mkdir()
    with pytest.raises((ValueError, OSError)):
        export.require_local_inference_export_locality(ctx, root)


def test_inference_export_locality_lora_sampling_requires_trained_adapter(
    _inference_export_locality_saved,
    _inference_export_locality_admission,
):
    root = _inference_export_locality_saved
    proof = export.distribute_local_inference_export(root, _inference_export_locality_admission, timeout_s=30)
    ctx = SimpleNamespace(
        inference_export_allocation=_inference_export_locality_admission,
        _inference_export_locality=proof,
        cfg=SimpleNamespace(training={"peft_config": {"peft_type": "LORA", "r": 2}}),
    )
    with pytest.raises(ValueError, match="trained PEFT adapter"):
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


def test_inference_export_locality_local_run_rejects_absent_admission_before_tokenizer_or_model(
    monkeypatch,
):
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
        assert kwargs == {
            "path": str(tmp_path),
            "checkpoint_type": "weights-only",
            "export_hf": False,
        }
        events.append("public_save")
        return result

    def metadata(actual, root, operation):
        assert actual == result and root == tmp_path and operation == "save"
        return "global_step10"

    monkeypatch.setattr(check, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(check, "_local_checkpoint_metadata", metadata)
    monkeypatch.setattr(check, "validate_local_inference_export", lambda ctx, path: path)
    monkeypatch.setattr(
        check,
        "materialize_checkpoint_tokenizer",
        lambda tokenizer, path: events.append("tokenizer_save"),
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
            ctx,
            backend,
            SimpleNamespace(save_checkpoint=lambda **kwargs: {}),
            tokenizer,
        )


@pytest.mark.parametrize("with_adapter", [False, True])
def test_inference_export_locality_process_stream_cpu_roundtrip_checks_complete_saved_bytes(
    tmp_path,
    _inference_export_locality_saved,
    request,
    with_adapter,
):
    if with_adapter:
        request.getfixturevalue("_inference_export_locality_adapter")
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
                sys.executable,
                "send",
                {"root": str(source), "manifest": manifest, "timeout_s": 15},
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
    monkeypatch.setattr(
        export,
        "_validate_local_allocation",
        lambda admission, deadline, hostfile_identity: [node, node],
    )
    monkeypatch.setattr(
        export,
        "_bounded_local_json",
        lambda argv, deadline: {"manifest": manifest, "head_files": fingerprints},
    )

    @contextmanager
    def stalled_transport(*args):
        yield [sys.executable, "-c", "import time;time.sleep(60)"]

    monkeypatch.setattr(export, "_local_ssh_argv", stalled_transport)
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
    monkeypatch.setattr(
        export,
        "_receive_local_export",
        lambda *args: pytest.fail("expired receiver started"),
    )
    with pytest.raises(TimeoutError, match="locality deadline"):
        export._local_worker("receive", {"timeout_s": 60, "deadline_epoch": time.time() - 1})


_inference_sampling_topology_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

_inference_sampling_topology_FLASH_CONFIG = (
    _inference_sampling_topology_CONFIG_ROOT / "qwen3.8-flash-next/h200/train-sft-full-16gpus-sp16-256k.config"
)

_inference_sampling_topology_ORDINARY_CONFIG = (
    _inference_sampling_topology_CONFIG_ROOT / "qwen3-8b/h200/train-sft-full-4gpus-64k.config"
)

_inference_sampling_topology_GLM_CONFIG = (
    _inference_sampling_topology_CONFIG_ROOT / "glm5.3-flash-bf16/h200/train-rl-32gpus-sp16-256k.config"
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


def test_inference_sampling_topology_glm_reviewed_sampler_survives_cortex_wire():
    ctx = _inference_sampling_topology_context(_inference_sampling_topology_GLM_CONFIG, topology=(16, 16, 1))
    before = deepcopy(ctx.cfg)
    config = check.cortex_sampling_config(
        ctx,
        CortexConfig(base_url="https://unused.invalid"),
        "cp_glm_full",
        "owned-training",
        10240,
    )
    sub = config.to_cortex()[0]
    assert check.sampling_gpu_count(ctx) == 16
    assert sub["model_name"] == "zai-org/GLM-5.3-Flash-BF16"
    assert sub["dtype"] == "bfloat16"
    assert sub["source_checkpoint_info"] == {
        "checkpoint_id": "cp_glm_full",
        "source_job_id": "owned-training",
    }
    inference = sub["inference_config"]
    engine = inference["vllm_config"]
    assert inference["n_gpus"] == engine["tensor_parallel_size"] == 16
    assert engine["pipeline_parallel_size"] == 1
    assert engine["enable_expert_parallel"] is True
    assert "engram_config" not in engine["extra_engine_kwargs"]
    assert engine["extra_engine_kwargs"]["distributed_executor_backend"] == "ray"
    assert ctx.cfg.n_gpus == ctx.cfg.sp_size == ctx.cfg.training["ep_size"] == 16
    assert ctx.cfg == before


def test_inference_sampling_topology_lora_recipe_survives_payload_and_cortex_wire():
    path = _inference_sampling_topology_CONFIG_ROOT / "glm5.3-flash/h200/train-rl-lora-16gpus-sp8-256k.config"
    ctx = _inference_sampling_topology_context(path, topology=(8, 8, 1))
    before = deepcopy(ctx.cfg)
    payload = check._sampling_payload(ctx, "/saved/trained-glm", 10240)
    expected = deepcopy(ctx.cfg.training["peft_config"])
    assert payload["inference_config"]["peft_config"] == expected
    payload["job_type"] = "sampling"
    config = rl_driver._inference_client_config(payload, backend=CortexConfig(base_url="http://127.0.0.1:8003"))
    sub = config.to_cortex()[0]
    assert sub["inference_config"]["peft_config"] == expected
    assert sub["model_name"] == "/saved/trained-glm"
    payload["inference_config"]["peft_config"]["target_modules"].append("changed")
    assert ctx.cfg == before


@pytest.mark.parametrize("topology", [(8, 8, 1), (16, 8, 2)])
def test_inference_sampling_topology_glm_rejects_changed_reviewed_placement(topology):
    ctx = _inference_sampling_topology_context(_inference_sampling_topology_GLM_CONFIG, topology=topology)
    with pytest.raises(ValueError, match="configured sampling allocation/TP/PP"):
        check.sampling_gpu_count(ctx)


@pytest.mark.parametrize("field", ["n_gpus", "tensor_parallel_size", "pipeline_parallel_size"])
@pytest.mark.parametrize("value", [True, 16.0, 0, -1])
def test_inference_sampling_topology_glm_rejects_invalid_reviewed_engine_sizes(field, value):
    ctx = _inference_sampling_topology_context(_inference_sampling_topology_GLM_CONFIG, topology=(16, 16, 1))
    inference = next(sub["inference_config"] for sub in ctx.cfg.sub_jobs if sub["job_type"] == "sampling")
    if field == "n_gpus":
        inference[field] = value
    else:
        inference["vllm_config"][field] = value
    with pytest.raises(ValueError, match="configured sampling allocation/TP/PP"):
        check.sampling_gpu_count(ctx)


def test_inference_sampling_topology_glm_local_sampling_keeps_full_model_and_reviewed_topology(monkeypatch, tmp_path):
    ctx = _inference_sampling_topology_context(_inference_sampling_topology_GLM_CONFIG, topology=(16, 16, 1))
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    weights = tmp_path / "saved-full-glm"
    validations = []
    monkeypatch.setattr(
        check,
        "validate_local_inference_export",
        lambda actual, path: validations.append(path) or path,
    )
    monkeypatch.setattr(
        export,
        "require_local_inference_export_locality",
        lambda actual, path: validations.append(path),
    )
    config = check.local_cortex_sampling_config(ctx, backend, weights, 10240)
    sub = config.to_cortex()[0]
    assert validations == [weights, weights]
    assert sub["model_name"] == str(weights)
    assert sub["dtype"] == "bfloat16"
    engine = sub["inference_config"]["vllm_config"]
    assert sub["inference_config"]["n_gpus"] == engine["tensor_parallel_size"] == 16
    assert engine["pipeline_parallel_size"] == 1
    assert "engram_config" not in engine["extra_engine_kwargs"]
    assert check._full_model_name(ctx) == ctx.cfg.model_name
    assert check.DEFAULT_TOLERANCE == 0.001
    assert check.TRAIN_STEPS == 10


def test_inference_sampling_topology_glm_without_opt_in_keeps_default_one_gpu():
    ctx = _inference_sampling_topology_context(_inference_sampling_topology_GLM_CONFIG)
    assert check.sampling_gpu_count(ctx) == 1
    assert check._sampling_payload(ctx, "/full-glm", 10240) == sampling_payload(
        "/full-glm", SEED, dtype="bfloat16", n_gpus=1, max_seq_len=10241
    )


def test_inference_sampling_topology_glm_service_recipe_keeps_declared_placement(
    tmp_path,
):
    original = _inference_sampling_topology_context(_inference_sampling_topology_GLM_CONFIG, topology=(16, 16, 1))
    recipe = tmp_path / "glm-service.config"
    recipe.write_text(json.dumps({"sub_job_configs": original.cfg.sub_jobs, "hardware": "H200"}))
    ctx = _inference_sampling_topology_context(recipe, topology=(16, 16, 1))
    assert check.sampling_gpu_count(ctx) == 16
    ctx.inference_sampling_topology = (16, 8, 2)
    with pytest.raises(ValueError, match="configured sampling allocation/TP/PP"):
        check.sampling_gpu_count(ctx)


@pytest.mark.parametrize("topology", _inference_sampling_topology_TOPOLOGIES)
def test_inference_sampling_topology_original_flashnext_request_separates_allocation_from_tp_and_pp(
    topology,
):
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
    "config",
    [
        _inference_sampling_topology_FLASH_CONFIG,
        _inference_sampling_topology_ORDINARY_CONFIG,
    ],
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
    ],
)
def test_inference_sampling_topology_malformed_topology_is_rejected(topology):
    ctx = _inference_sampling_topology_context(topology=topology)
    with pytest.raises(ValueError):
        check.sampling_gpu_count(ctx)
    with pytest.raises(ValueError):
        check._sampling_payload(ctx, "/saved/full-checkpoint", _inference_sampling_topology_TOKEN_BUDGET)


@pytest.mark.parametrize("different_identity", ["config_id", "model_name", "training_width"])
def test_inference_sampling_topology_explicit_placement_is_not_restricted_by_training_identity(
    different_identity,
):
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
    assert check.sampling_gpu_count(ctx) == 16


@pytest.mark.parametrize("topology", [(16, 16, 1), (8, 4, 2), (1, 1, 1)])
def test_inference_sampling_topology_explicit_training_only_placement_accepts_valid_sizes(
    topology,
):
    ctx = _inference_sampling_topology_context(_inference_sampling_topology_ORDINARY_CONFIG, topology=topology)
    assert check.sampling_gpu_count(ctx) == topology[0]
    engine = check._sampling_payload(ctx, "/full-model", 10240)["inference_config"]["vllm_config"]
    assert engine["tensor_parallel_size"] == topology[1]
    assert engine["pipeline_parallel_size"] == topology[2]
    assert "engram_config" not in engine["extra_engine_kwargs"]


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
        ctx,
        backend,
        "cp_full_saved",
        "owned-training",
        _inference_sampling_topology_TOKEN_BUDGET,
    )
    assert calls == [(ctx, ctx.cfg.model_name, _inference_sampling_topology_TOKEN_BUDGET)]
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    _inference_sampling_topology_assert_topology(sub, topology)
    assert sub["model_name"] == ctx.cfg.model_name
    assert sub["source_checkpoint_info"] == {
        "checkpoint_id": "cp_full_saved",
        "source_job_id": "owned-training",
    }
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
    monkeypatch.setattr(
        export,
        "require_local_inference_export_locality",
        lambda ctx, path: events.append("locality"),
    )
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
            ctx,
            backend,
            tmp_path / "incomplete",
            _inference_sampling_topology_TOKEN_BUDGET,
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
    config = ArcticClientConfig(
        model_name=payload["model_name"],
        seed=payload["seed"],
        dtype=payload["dtype"],
        sampling_gpus=payload["inference_config"]["n_gpus"],
        sampling={"vllm": payload["inference_config"]["vllm_config"]},
        backend=CortexConfig(base_url="https://unused.invalid"),
    )
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    assert sub["inference_config"]["n_gpus"] == 4
    assert "tensor_parallel_size" not in sub["inference_config"]["vllm_config"]
    assert config.model_dump() == before


@pytest.mark.parametrize(
    "config_path",
    [
        _inference_sampling_topology_FLASH_CONFIG,
        _inference_sampling_topology_ORDINARY_CONFIG,
    ],
)
def test_inference_sampling_topology_correctness_one_gpu_wire_keeps_default_placement(
    config_path,
):
    ctx = _inference_sampling_topology_context(config_path)
    backend = CortexConfig(base_url="https://unused.invalid")
    config = check.cortex_sampling_config(
        ctx,
        backend,
        "cp_full_saved",
        "owned-training",
        _inference_sampling_topology_TOKEN_BUDGET,
    )
    before = deepcopy(config.model_dump())
    sub = config.to_cortex()[0]
    assert config.inference_tensor_parallel_size == 1
    assert config.sampling_gpus == sub["inference_config"]["n_gpus"] == 1
    assert sub["inference_config"]["vllm_config"]["tensor_parallel_size"] == 1
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
def test_inference_sampling_topology_scoped_cortex_field_rejects_inconsistent_or_non_cortex_topology(
    invalid,
):
    ctx = _inference_sampling_topology_context(topology=(16, 8, 2))
    backend = CortexConfig(base_url="https://unused.invalid")
    config = check.cortex_sampling_config(
        ctx,
        backend,
        "cp_full_saved",
        "owned-training",
        _inference_sampling_topology_TOKEN_BUDGET,
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


@pytest.mark.parametrize("reviewed_layout", [False, True])
def test_inference_sampling_topology_invalid_opt_in_fails_before_training_or_gateway_creation(
    monkeypatch, tmp_path, reviewed_layout
):
    ctx = _inference_sampling_topology_context(
        (_inference_sampling_topology_GLM_CONFIG if reviewed_layout else _inference_sampling_topology_FLASH_CONFIG),
        topology=(16, 8, 2) if reviewed_layout else (16, 8, 1),
    )
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
    distribution = {
        "examples": 1,
        "minimum_tokens": 4,
        "median_tokens": 4,
        "maximum_tokens": 4,
    }
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: object())
    monkeypatch.setattr(check, "_minimum_context_parallel_row_tokens", lambda *args: 0)
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
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(**source_config),
    )

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


@pytest.mark.parametrize("as_string", [False, True])
def test_inference_text_projection_export_uses_admitted_initial_model_directory(
    _inference_text_projection_projection_case, monkeypatch, as_string
):
    case = _inference_text_projection_projection_case
    case.ctx.inference_initial_model_dir = str(case.source) if as_string else case.source
    config_before = copy.deepcopy(vars(case.ctx.cfg))
    loaded = []

    def from_pretrained(model_name, **kwargs):
        assert model_name == str(case.source)
        assert kwargs == {"local_files_only": True}
        loaded.append("configuration")
        return SimpleNamespace(**case.source_config)

    def cached_file(model_name, filename, **kwargs):
        assert model_name == str(case.source)
        assert kwargs == {"local_files_only": True}
        loaded.append(filename)
        return str(case.source / filename)

    monkeypatch.setattr(AutoConfig, "from_pretrained", from_pretrained)
    monkeypatch.setattr(hub, "cached_file", cached_file)
    assert check.validate_local_inference_export(case.ctx, case.exported) == case.exported
    assert loaded == ["configuration", "config.json", "model.safetensors.index.json"]
    assert vars(case.ctx.cfg) == config_before

    altered = copy.deepcopy(case.config)
    altered["hidden_size"] += 1
    case.write(case.exported, altered, case.tensors)
    with pytest.raises(ValueError):
        check.validate_local_inference_export(case.ctx, case.exported)


@pytest.mark.parametrize("invalid", ["missing", "symlink", "outside_fast", "boolean"])
def test_inference_text_projection_export_rejects_unvalidated_initial_model_directory(
    _inference_text_projection_projection_case, monkeypatch, tmp_path, invalid
):
    case = _inference_text_projection_projection_case
    if invalid == "missing":
        value = tmp_path / "missing"
    elif invalid == "symlink":
        value = tmp_path / "source-alias"
        value.symlink_to(case.source, target_is_directory=True)
    elif invalid == "outside_fast":
        value = Path("/tmp")
    else:
        value = True
    case.ctx.inference_initial_model_dir = value

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid source must be rejected before HF metadata lookup")

    monkeypatch.setattr(AutoConfig, "from_pretrained", unexpected_load)
    monkeypatch.setattr(hub, "cached_file", unexpected_load)
    with pytest.raises(ValueError):
        check.validate_local_inference_export(case.ctx, case.exported)


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
    gsm8k.Example(
        question="validation0",
        prompt="p",
        completion=" a",
        input_ids=(1, 2, 3),
        n_prompt=1,
    ),
    gsm8k.Example(
        question="validation1",
        prompt="p",
        completion=" b",
        input_ids=(1, 4, 5),
        n_prompt=1,
    ),
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
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {"model.weight": "model.safetensors"},
    }
    source.write_text(json.dumps(index))
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(**_local_cortex_inference_FULL_CONFIG),
    )
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(source))
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda *args: _local_cortex_inference_Tokenizer())

    def create(path):
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps(_local_cortex_inference_FULL_CONFIG))
        (path / "model.safetensors.index.json").write_text(json.dumps(index))
        save_file(
            {"model.weight": torch.ones(2, dtype=torch.bfloat16)},
            str(path / "model.safetensors"),
        )
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


@pytest.mark.parametrize(
    "missing,unexpected",
    [
        (["mtp.fc.weight"], []),
        ([], ["model.unexpected.weight"]),
        (["model.weight", "mtp.fc.weight"], ["model.replacement.weight"]),
        ([f"mtp.layers.{index}.weight" for index in reversed(range(19))], []),
    ],
)
def test_local_inference_export_inventory_diagnostic_preserves_strict_names(
    tmp_path, _local_cortex_inference_export_factory, monkeypatch, missing, unexpected
):
    weights = _local_cortex_inference_export_factory(tmp_path / "weights")
    source_index = tmp_path / "source-index.json"
    original = json.loads(source_index.read_text())
    original["weight_map"].update({name: "model.safetensors" for name in missing})
    source_index.write_text(json.dumps(original))
    exported_index = weights / "model.safetensors.index.json"
    exported = json.loads(exported_index.read_text())
    for name in missing:
        exported["weight_map"].pop(name, None)
    exported["weight_map"].update({name: "model.safetensors" for name in unexpected})
    exported_index.write_text(json.dumps(exported))
    before = (source_index.read_bytes(), exported_index.read_bytes())
    monkeypatch.setattr(check, "require_data_fast_path", lambda path, **kwargs: path)
    with pytest.raises(RuntimeError, match="full configured tensor inventory") as failure:
        check.validate_local_inference_export(_local_cortex_inference_context(tmp_path), weights)
    message = str(failure.value)
    assert f"missing={len(missing)} {sorted(missing)[:8]!r}" in message
    assert f"unexpected={len(unexpected)} {sorted(unexpected)[:8]!r}" in message
    if len(missing) > 8:
        assert sorted(missing)[8] not in message
    assert (source_index.read_bytes(), exported_index.read_bytes()) == before


@pytest.mark.parametrize("side", ["original", "exported"])
@pytest.mark.parametrize("mapping", [None, [], {}])
def test_local_inference_export_inventory_diagnostic_rejects_invalid_mapping(
    tmp_path, _local_cortex_inference_export_factory, monkeypatch, side, mapping
):
    weights = _local_cortex_inference_export_factory(tmp_path / "weights")
    path = tmp_path / "source-index.json" if side == "original" else weights / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"] = mapping
    path.write_text(json.dumps(index))
    monkeypatch.setattr(check, "require_data_fast_path", lambda path, **kwargs: path)
    with pytest.raises(
        RuntimeError,
        match="full configured tensor inventory.*invalid or empty weight_map",
    ):
        check.validate_local_inference_export(_local_cortex_inference_context(tmp_path), weights)


@pytest.mark.parametrize("original_count,exported_count", [(4.0, 4), (4, 4.0), (4.0, 4.0)])
def test_local_inference_export_integral_metadata_accepts_exact_float(
    tmp_path,
    _local_cortex_inference_export_factory,
    monkeypatch,
    original_count,
    exported_count,
):
    weights = _local_cortex_inference_export_factory(tmp_path / "weights")
    source_index = tmp_path / "source-index.json"
    exported_index = weights / "model.safetensors.index.json"
    for path, count in (
        (source_index, original_count),
        (exported_index, exported_count),
    ):
        index = json.loads(path.read_text())
        index["metadata"]["total_size"] = count
        path.write_text(json.dumps(index))
    before = (source_index.read_bytes(), exported_index.read_bytes())
    monkeypatch.setattr(check, "require_data_fast_path", lambda path, **kwargs: path)
    assert check.validate_local_inference_export(_local_cortex_inference_context(tmp_path), weights) == weights
    assert (source_index.read_bytes(), exported_index.read_bytes()) == before


@pytest.mark.parametrize("side", ["original", "exported"])
@pytest.mark.parametrize("count", [True, False, 0, -4, 4.5, 6.0, "4", None, float("nan"), float("inf")])
def test_local_inference_export_integral_metadata_rejects_invalid_count(
    tmp_path, _local_cortex_inference_export_factory, monkeypatch, side, count
):
    weights = _local_cortex_inference_export_factory(tmp_path / "weights")
    path = tmp_path / "source-index.json" if side == "original" else weights / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["metadata"]["total_size"] = count
    path.write_text(json.dumps(index))
    monkeypatch.setattr(check, "require_data_fast_path", lambda path, **kwargs: path)
    with pytest.raises(RuntimeError, match="tensor byte inventory"):
        check.validate_local_inference_export(_local_cortex_inference_context(tmp_path), weights)


@pytest.mark.parametrize("side", ["original", "exported"])
@pytest.mark.parametrize("metadata", [None, [], "invalid", 4])
def test_local_inference_export_integral_metadata_rejects_malformed_metadata(
    tmp_path, _local_cortex_inference_export_factory, monkeypatch, side, metadata
):
    weights = _local_cortex_inference_export_factory(tmp_path / "weights")
    path = tmp_path / "source-index.json" if side == "original" else weights / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["metadata"] = metadata
    path.write_text(json.dumps(index))
    monkeypatch.setattr(check, "require_data_fast_path", lambda path, **kwargs: path)
    with pytest.raises(RuntimeError, match="tensor byte inventory"):
        check.validate_local_inference_export(_local_cortex_inference_context(tmp_path), weights)


def test_local_cortex_inference_local_sampling_uses_exact_export_same_backend_one_gpu_no_source(
    tmp_path,
    _local_cortex_inference_export_factory,
    _inference_export_prescribed_hostfile,
):
    ctx = _local_cortex_inference_context(tmp_path)
    weights = _local_cortex_inference_export_factory(tmp_path / "global_step10")
    ctx.inference_export_allocation = _inference_export_prescribed_hostfile
    _local_cortex_inference_Tokenizer().save_pretrained(weights)
    ctx._inference_export_locality = export.distribute_local_inference_export(
        weights, ctx.inference_export_allocation, timeout_s=30
    )
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
    result = {
        "checkpoint_path": str(root),
        "checkpoint_tag": "global_step10",
        "version": 10,
    }
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
    tmp_path,
    _local_cortex_inference_export_factory,
    _inference_export_prescribed_hostfile,
    monkeypatch,
    malformed,
):
    ctx = _local_cortex_inference_context(tmp_path)
    ctx.inference_export_allocation = _inference_export_prescribed_hostfile
    events = []
    saved = []

    def save(**kwargs):
        saved.append(kwargs)
        root = check.Path(kwargs["path"])
        _local_cortex_inference_export_factory(root / "global_step10")
        return {
            "checkpoint_path": str(root),
            "checkpoint_tag": "global_step10",
            "version": 10,
        }

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
                "prompt_logprobs": [
                    None,
                    *[{str(token): {"logprob": -1.0}} for token in row.input_ids[1:]],
                ],
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
                gsm8k.Example(
                    question=f"train{i}",
                    prompt="p",
                    completion=" a",
                    input_ids=(1, 2, 3),
                    n_prompt=1,
                )
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
    monkeypatch.setattr(
        check.gsm8k,
        "select_by_token_budget",
        lambda *args: (_local_cortex_inference_ROWS, 3),
    )
    monkeypatch.setattr(
        check,
        "fwd_bwd_step",
        lambda *args, **kwargs: learning_rates.append(kwargs["learning_rate"]) or SimpleNamespace(avg_loss=2.0),
    )
    monkeypatch.setattr(check, "forward_logprobs", lambda *args: torch.full((2, 3), -1.0))
    monkeypatch.setattr(
        check,
        "save_and_resolve_checkpoint_id",
        lambda *args: pytest.fail("no hosted catalog"),
    )
    monkeypatch.setattr(
        check,
        "_onprem_sampling_and_reload",
        lambda *args: pytest.fail("no reload diagnostic"),
    )
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
    "invalid",
    [
        "missing",
        "file",
        "relative",
        "outside_fast",
        "symlink",
        "ancestor_symlink",
        "dotdot",
        "boolean",
    ],
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
        SimpleNamespace(backend=backend),
        {key: value for key, value in payload.items() if key != "local_model_dir"},
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


@pytest.fixture
def glm53_inference_export_case():
    config = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "use_cache": True,
        },
        "vision_config": {"model_type": "glm5_next_vision", "depth": 1},
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"],
        },
    }
    prefix = "model.language_model.layers."
    source = {
        prefix + "0.self_attn.A_log": ["F32", [2]],
        prefix + "0.self_attn.dt_bias": ["F32", [2]],
        prefix + "1.hc_attn_base": ["F32", [2]],
        prefix + "1.hc_attn_scale": ["F32", [2]],
        prefix + "1.hc_ffn_base": ["F32", [2]],
        prefix + "1.hc_ffn_scale": ["F32", [2]],
        prefix + "1.mlp.gate.weight": ["F32", [2, 2]],
        prefix + "1.mlp.experts.0.down_proj.weight": ["F8_E4M3", [2, 2]],
        prefix + "1.mlp.experts.0.down_proj.weight_scale_inv": ["F32", [1, 1]],
        prefix + "2.eh_proj.weight": ["F8_E4M3", [2, 4]],
        prefix + "2.eh_proj.weight_scale_inv": ["F32", [1, 1]],
        prefix + "2.enorm.weight": ["BF16", [2]],
        prefix + "2.hnorm.weight": ["BF16", [2]],
        prefix + "2.hc_attn_base": ["F32", [2]],
        "model.visual.norm.weight": ["BF16", [2]],
    }
    target = copy.deepcopy(source)
    for name in source:
        if name.endswith(
            (
                "A_log",
                "dt_bias",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_scale",
            )
        ):
            if not name.startswith(prefix + "2."):
                target[name][0] = "BF16"
    return config, source, target


def test_glm53_inference_schema_preserves_quantized_and_inactive_headers(
    glm53_inference_export_case,
):
    config, source, target = glm53_inference_export_case
    before = copy.deepcopy((config, source, target))
    payload_bytes = export.validate_glm53_full_schema(config, source, target)
    assert payload_bytes == export.original_schema_bytes(target, config["quantization_config"])
    assert payload_bytes == export.original_schema_bytes(source, config["quantization_config"]) - 24
    assert (config, source, target) == before


@pytest.mark.parametrize(
    "damage",
    ["missing", "unexpected", "shape", "scale", "fp8", "gate", "auxiliary", "norm"],
)
def test_glm53_inference_schema_rejects_any_unapproved_header_change(glm53_inference_export_case, damage):
    config, source, target = glm53_inference_export_case
    prefix = "model.language_model.layers."
    if damage == "missing":
        target.pop(prefix + "2.enorm.weight")
    elif damage == "unexpected":
        target["unexpected.weight"] = ["BF16", [2]]
    elif damage == "shape":
        target[prefix + "1.mlp.gate.weight"][1] = [4]
    elif damage == "scale":
        target[prefix + "1.mlp.experts.0.down_proj.weight_scale_inv"][1] = [2, 1]
    elif damage == "fp8":
        target[prefix + "1.mlp.experts.0.down_proj.weight"][0] = "BF16"
    elif damage == "gate":
        target[prefix + "1.mlp.gate.weight"][0] = "BF16"
    elif damage == "auxiliary":
        target[prefix + "2.hc_attn_base"][0] = "BF16"
    else:
        target[prefix + "0.self_attn.A_log"][0] = "F32"
    with pytest.raises(ValueError):
        export.validate_glm53_full_schema(config, source, target)


@pytest.mark.parametrize(
    "path",
    [
        ("text_config", "num_hidden_layers"),
        ("text_config", "num_nextn_predict_layers"),
        ("vision_config", "depth"),
        ("quantization_config", "weight_block_size"),
        ("quantization_config", "modules_to_not_convert"),
    ],
)
def test_glm53_inference_config_preserves_nested_model_and_quantization_settings(glm53_inference_export_case, path):
    config, _, _ = glm53_inference_export_case
    target = copy.deepcopy(config)
    assert export.validate_glm53_full_config(config, target) is None
    target[path[0]][path[1]] = None
    with pytest.raises(ValueError):
        export.validate_glm53_full_config(config, target)


@pytest.mark.parametrize("damage", ["unsupported-quantization", "missing-layer", "missing-auxiliary"])
def test_glm53_inference_schema_rejects_incomplete_or_unsupported_original(glm53_inference_export_case, damage):
    config, source, target = glm53_inference_export_case
    if damage == "unsupported-quantization":
        config["quantization_config"]["fmt"] = "e5m2"
    elif damage == "missing-layer":
        for schema in (source, target):
            for name in list(schema):
                if name.startswith("model.language_model.layers.0."):
                    schema.pop(name)
    else:
        for schema in (source, target):
            schema.pop("model.language_model.layers.2.enorm.weight")
    with pytest.raises(ValueError):
        export.validate_glm53_full_schema(config, source, target)


@pytest.fixture
def glm53_local_inference_export(tmp_path, monkeypatch, glm53_inference_export_case):
    config, original_schema, exported_schema = glm53_inference_export_case
    source, target = tmp_path / "source", tmp_path / "target"
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
    }
    for root, schema in ((source, original_schema), (target, exported_schema)):
        root.mkdir()
        values = {name: torch.ones(shape).to(dtypes[dtype]) for name, (dtype, shape) in schema.items()}
        save_file(values, str(root / "weights.safetensors"))
        (root / "config.json").write_text(json.dumps(config))
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": sum(value.numel() * value.element_size() for value in values.values())},
                    "weight_map": {name: "weights.safetensors" for name in values},
                }
            )
        )
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (root / name).write_text("{}")

    def load_config(model, **kwargs):
        assert model == str(source)
        assert kwargs == {"local_files_only": True}
        return SimpleNamespace(**config)

    def cached_file(model, name, **kwargs):
        assert model == str(source)
        assert kwargs == {"local_files_only": True}
        return str(source / name)

    monkeypatch.setattr(AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(hub, "cached_file", cached_file)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(model_name="synthetic/glm5-next"),
        inference_initial_model_dir=source,
    )
    return ctx, source, target, config, original_schema, exported_schema


def test_glm53_inference_schema_local_export_admits_real_mixed_headers_without_modifying_files(
    glm53_local_inference_export,
):
    ctx, source, target, config, original_schema, exported_schema = glm53_local_inference_export
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for root in (source, target) for path in root.iterdir()
    }
    assert check.validate_local_inference_export(ctx, target) == target
    assert (
        export.read_inventory(
            target,
            export.read_export_json(target / "model.safetensors.index.json"),
            original_quantization_config=config["quantization_config"],
        )
        == exported_schema
    )
    assert original_schema != exported_schema
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before


@pytest.mark.parametrize("damage", ["config", "missing", "dtype", "scale", "bytes", "index", "symlink"])
def test_glm53_inference_schema_local_export_rejects_real_file_damage(glm53_local_inference_export, damage):
    ctx, _, target, config, _, schema = glm53_local_inference_export
    index_path = target / "model.safetensors.index.json"
    index = export.read_export_json(index_path)
    if damage == "config":
        config["quantization_config"]["weight_block_size"] = [64, 64]
        (target / "config.json").write_text(json.dumps(config))
    elif damage in ("missing", "dtype", "scale"):
        values = safetensors.torch.load_file(str(target / "weights.safetensors"))
        if damage == "missing":
            values.pop("model.language_model.layers.2.hnorm.weight")
        elif damage == "dtype":
            name = "model.language_model.layers.2.hc_attn_base"
            values[name] = values[name].to(torch.bfloat16)
        else:
            values["model.language_model.layers.1.mlp.experts.0.down_proj.weight_scale_inv"] = torch.ones(2, 1)
        save_file(values, str(target / "weights.safetensors"))
        index["weight_map"] = {name: "weights.safetensors" for name in values}
        index["metadata"]["total_size"] = sum(value.numel() * value.element_size() for value in values.values())
        index_path.write_text(json.dumps(index))
    elif damage == "bytes":
        index["metadata"]["total_size"] += 1
        index_path.write_text(json.dumps(index))
    elif damage == "index":
        index["weight_map"].pop(next(iter(schema)))
        index_path.write_text(json.dumps(index))
    else:
        original = target / "retained.safetensors"
        (target / "weights.safetensors").rename(original)
        (target / "weights.safetensors").symlink_to(original)
    with pytest.raises(ValueError):
        check.validate_local_inference_export(ctx, target)


def test_glm53_inference_schema_same_allocation_manifest_accounts_for_quantized_bytes(
    glm53_local_inference_export,
):
    _, _, target, config, _, schema = glm53_local_inference_export
    manifest = export._local_manifest(target, time.monotonic() + 30)
    assert manifest["tensor_schema_sha256"] == export.schema_sha(schema)
    assert manifest["tensor_payload_bytes"] == export.original_schema_bytes(schema, config["quantization_config"])
    assert manifest["config_sha256"] == export.schema_sha(config)
    assert {entry["path"] for entry in manifest["files"]} == {path.name for path in target.iterdir()}


def test_glm53_inference_schema_same_allocation_manifest_rejects_quantization_recipe_change(
    glm53_local_inference_export,
):
    _, _, target, config, _, _ = glm53_local_inference_export
    config["quantization_config"]["fmt"] = "e5m2"
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="supported dynamic block-FP8"):
        export._local_manifest(target, time.monotonic() + 30)


@pytest.mark.parametrize(
    "invalid",
    [
        "onprem",
        "nonlocal",
        "authenticated",
        "no_training",
        "sampling",
        "log_prob",
        "checkpoint_source",
    ],
)
def test_inference_initial_model_dir_typed_config_refuses_other_roles(tmp_path, invalid):
    root = tmp_path / "full-model"
    root.mkdir()
    backend = CortexConfig(base_url="http://127.0.0.1:8003")
    base = rl_driver._training_client_config(
        SimpleNamespace(backend=backend),
        rl_driver.build_payload({"n_gpus": 4}, "Qwen/canonical-full", SEED),
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
        base["source_checkpoint_info"] = {
            "checkpoint_id": "cp_saved",
            "source_job_id": "owned-parent",
        }
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
    monkeypatch.setattr(
        runner,
        "correctness_workdir",
        lambda *a: pytest.fail("invalid root allocated work"),
    )
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
    monkeypatch.setattr(
        check.gsm8k,
        "load_tokenizer",
        lambda *a: pytest.fail("invalid root loaded tokenizer"),
    )
    monkeypatch.setattr(check, "gateway", lambda *a, **kw: pytest.fail("invalid root allocated gateway"))
    with pytest.raises(ValueError):
        check._run(ctx)


@pytest.mark.parametrize("source_kind", ["initial_model", "admitted_full_model", "catalog"])
def test_inference_initial_model_dir_actual_check_binds_training_only_and_samples_export(
    _inference_export_prescribed_hostfile,
    _local_cortex_inference_export_factory,
    tmp_path,
    monkeypatch,
    source_kind,
):
    root = tmp_path / "initial-full-model"
    root.mkdir()
    ctx = _local_cortex_inference_context(tmp_path)
    if source_kind == "initial_model":
        ctx.inference_initial_model_dir = root
    elif source_kind == "admitted_full_model":
        ctx.local_gateway_full_model = True
        ctx.local_full_model_path = str(root)
    ctx.inference_export_allocation = _inference_export_prescribed_hostfile
    weights = _local_cortex_inference_export_factory(tmp_path / "global_step10")
    original = ctx.cfg.native.model_dump()
    events = []
    steps = []
    tokenizers = []
    context_sources = []

    def load_tokenizer(model):
        tokenizers.append(model)
        return object()

    def read_rows(path, count=None):
        return (
            [
                gsm8k.Example(
                    question=f"train{i}",
                    prompt="p",
                    completion=" a",
                    input_ids=(1, 2, 3),
                    n_prompt=1,
                )
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
        if source_kind == "initial_model":
            assert payload["local_model_dir"] == str(root)
        else:
            assert "local_model_dir" not in payload
        public = rl_driver._training_client_config(session, payload).to_cortex()[0]
        assert public["model_name"] == ctx.cfg.model_name
        if source_kind == "initial_model":
            assert public["local_model_dir"] == str(root)
        else:
            assert "local_model_dir" not in public
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
                "prompt_logprobs": [
                    None,
                    *[{str(token): {"logprob": -1.0}} for token in row.input_ids[1:]],
                ],
            }
            for row in _local_cortex_inference_ROWS
        ]
        try:
            yield SimpleNamespace(generate=lambda *a, **kw: results)
        finally:
            events.append("sampling-exit")

    monkeypatch.setattr(check.gsm8k, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(
        check,
        "_minimum_context_parallel_row_tokens",
        lambda actual_ctx, path: context_sources.append(path) or 0,
    )
    monkeypatch.setattr(check.gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda tokenizer, rows: rows)
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda tokenizer: 0)
    monkeypatch.setattr(
        check.gsm8k,
        "select_by_token_budget",
        lambda *a: (_local_cortex_inference_ROWS, 3),
    )
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
    expected_source = ctx.cfg.model_name if source_kind == "catalog" else str(root)
    assert tokenizers == context_sources == [expected_source]
    assert steps == [check.configured_learning_rate(ctx.cfg)] * 10
    assert ctx.cfg.native.model_dump() == original
    assert events == [
        "training-enter",
        "training-exit",
        "sampling-enter",
        "sampling-exit",
    ]


@pytest.mark.parametrize("invalid", ["missing", "symlink", "outside_fast", "boolean"])
def test_inference_admitted_full_model_input_rejects_invalid_path_before_tokenizer(
    _inference_export_prescribed_hostfile, tmp_path, monkeypatch, invalid
):
    root = tmp_path / "initial-full-model"
    root.mkdir()
    if invalid == "missing":
        value = tmp_path / "missing"
    elif invalid == "symlink":
        value = tmp_path / "alias"
        value.symlink_to(root, target_is_directory=True)
    elif invalid == "outside_fast":
        value = Path("/tmp")
    else:
        value = True
    ctx = _local_cortex_inference_context(tmp_path)
    ctx.inference_export_allocation = _inference_export_prescribed_hostfile
    ctx.local_gateway_full_model = True
    ctx.local_full_model_path = value
    monkeypatch.setattr(
        check.gsm8k,
        "load_tokenizer",
        lambda *a: pytest.fail("invalid source loaded tokenizer"),
    )
    monkeypatch.setattr(
        check,
        "gateway",
        lambda *a, **kw: pytest.fail("invalid source allocated gateway"),
    )
    with pytest.raises(ValueError):
        check._run(ctx)


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


def test_qwen4_inference_schema_full_nested_config_preserved_without_mutating_source(
    _qwen4_inference_schema_config,
):
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


def test_qwen4_inference_schema_full_config_rejects_json_scalar_type_substitution(
    _qwen4_inference_schema_config,
):
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
            _qwen4_inference_schema_schema,
            copy.deepcopy(_qwen4_inference_schema_schema),
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


@pytest.mark.parametrize(
    "failure",
    ["extra", "rename", "floating-weight", "unrecognized-integer", "buffer-source"],
)
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


def test_qwen4_inference_schema_floating_source_projection_preserves_integer_buffers(
    _qwen4_inference_schema_schema,
):
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
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(**_qwen4_inference_schema_config),
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


@pytest.mark.parametrize("change", ["missing", "unexpected", "shape", "float", "integer", "combined"])
def test_qwen4_schema_diagnostic_reports_exact_mismatch_without_changing_gate(_qwen4_inference_schema_schema, change):
    original = copy.deepcopy(_qwen4_inference_schema_schema)
    target = copy.deepcopy(original)
    weight = "model.language_model.layers.0.norm.weight"
    integer = _qwen4_inference_schema_PREFIX + "layer_multipliers"
    if change in ("missing", "combined"):
        target.pop("model.language_model.embed_tokens.weight")
    if change in ("unexpected", "combined"):
        target["extra.weight"] = ["BF16", [2]]
    if change in ("shape", "combined"):
        target[weight][1] = [2, 2]
    if change in ("float", "combined"):
        target[weight][0] = "F32"
    if change in ("integer", "combined"):
        target[integer][0] = "BF16"
    before = copy.deepcopy(target)
    with pytest.raises(ValueError, match="exact original Qwen4") as failure:
        export.validate_qwen4_full_schema(original, target)
    receipt = failure.value.diagnostic
    assert receipt["missing"] == sorted(set(original) - set(target))
    assert receipt["unexpected"] == sorted(set(target) - set(original))
    expected_differences = {
        name: {"expected": original[name], "actual": target[name]}
        for name in sorted(set(original) & set(target))
        if original[name] != target[name]
    }
    assert receipt["differences"] == expected_differences
    for name, schema in (
        ("source", original),
        ("projected_source", original),
        ("export", target),
    ):
        assert receipt[name + "_schema"] == schema
        assert receipt[name + "_schema_sha256"] == export.schema_sha(schema)
        assert receipt[name + "_tensor_count"] == len(schema)
        assert receipt[name + "_payload_bytes"] == export.schema_bytes(schema)
    assert receipt["tensor_payloads_read"] is False
    assert receipt["numerical_pass"] is None
    assert original == _qwen4_inference_schema_schema
    assert target == before
    target[weight][1].append(7)
    assert receipt["export_schema"] == before


def test_qwen4_schema_diagnostic_projects_float_source_but_preserves_integer_expectations(
    _qwen4_inference_schema_schema,
):
    source = copy.deepcopy(_qwen4_inference_schema_schema)
    target = copy.deepcopy(source)
    source["model.language_model.layers.0.norm.weight"][0] = "F32"
    target[_qwen4_inference_schema_PREFIX + "layer_multipliers"][0] = "F32"
    with pytest.raises(ValueError) as failure:
        export.validate_qwen4_full_schema(source, target)
    receipt = failure.value.diagnostic
    assert receipt["source_schema"] == source
    assert receipt["projected_source_schema"] == _qwen4_inference_schema_schema
    assert receipt["differences"] == {
        _qwen4_inference_schema_PREFIX
        + "layer_multipliers": {
            "expected": ["I64", [2]],
            "actual": ["F32", [2]],
        }
    }


@pytest.fixture
def _qwen4_schema_diagnostic_local(tmp_path, monkeypatch, _qwen4_inference_schema_config):
    source = tmp_path / "source"
    target = tmp_path / "target"
    values = {
        "model.language_model.layers.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        _qwen4_inference_schema_PREFIX + "layer_multipliers": torch.tensor([2**53 + 1], dtype=torch.int64),
    }
    for root in (source, target):
        root.mkdir()
        (root / "config.json").write_text(json.dumps(_qwen4_inference_schema_config))
        save_file(values, str(root / "weights.safetensors"))
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 16},
                    "weight_map": {name: "weights.safetensors" for name in values},
                }
            )
        )
    logs = tmp_path / "logs"
    monkeypatch.setenv("DSS_CORRECTNESS_LOGDIR", str(logs))
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(**_qwen4_inference_schema_config),
    )
    monkeypatch.setattr(hub, "cached_file", lambda model, name, **kwargs: str(source / name))
    return SimpleNamespace(
        source=source,
        target=target,
        logs=logs,
        values=values,
        ctx=SimpleNamespace(cfg=SimpleNamespace(model_name="Qwen/synthetic-qwen4")),
    )


def _qwen4_schema_diagnostic_change_export(case):
    values = dict(case.values)
    values["model.language_model.layers.0.norm.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
    save_file(values, str(case.target / "weights.safetensors"))


def test_qwen4_schema_diagnostic_local_retains_full_header_bound_receipt(
    _qwen4_schema_diagnostic_local,
):
    case = _qwen4_schema_diagnostic_local
    _qwen4_schema_diagnostic_change_export(case)
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (case.source, case.target)
        for path in root.iterdir()
    }
    with pytest.raises(ValueError, match="exact original Qwen4.*diagnostic=") as failure:
        check.validate_local_inference_export(case.ctx, case.target)
    files = list(case.logs.glob("inference-schema-*/schema-mismatch.json"))
    assert len(files) == 1
    path = files[0]
    receipt = json.loads(path.read_text())
    assert str(path) in str(failure.value)
    assert hashlib.sha256(path.read_bytes()).hexdigest() in str(failure.value)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert receipt["differences"] == {
        "model.language_model.layers.0.norm.weight": {
            "expected": ["BF16", [4]],
            "actual": ["BF16", [2, 2]],
        }
    }
    for side, root in (("source", case.source), ("export", case.target)):
        provenance = receipt[side + "_provenance"]
        assert provenance["root"] == str(root)
        assert len(provenance["root_identity"]) == 5
        assert set(provenance["files"]) == {
            "config.json",
            "model.safetensors.index.json",
            "weights.safetensors",
        }
        for name in ("config.json", "model.safetensors.index.json"):
            entry = provenance["files"][name]
            assert entry["sha256"] == hashlib.sha256((root / name).read_bytes()).hexdigest()
            assert len(entry["identity"]) == 5
        shard = provenance["files"]["weights.safetensors"]
        data = (root / "weights.safetensors").read_bytes()
        header = data[8 : 8 + int.from_bytes(data[:8], "little")]
        assert shard["header_sha256"] == hashlib.sha256(header).hexdigest()
        assert shard["header_size_bytes"] == len(header)
    assert receipt["tensor_payloads_read"] is False
    assert receipt["runtime_admitted"] is False
    assert receipt["numerical_pass"] is None
    for filename, digest in before.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest


def test_qwen4_schema_diagnostic_local_success_creates_no_diagnostic(
    _qwen4_schema_diagnostic_local,
):
    case = _qwen4_schema_diagnostic_local
    assert check.validate_local_inference_export(case.ctx, case.target) == case.target
    assert not case.logs.exists()


def test_qwen4_schema_diagnostic_local_write_failure_preserves_schema_error(
    _qwen4_schema_diagnostic_local, monkeypatch
):
    from arctic_platform.correctness.harness import storage

    case = _qwen4_schema_diagnostic_local
    _qwen4_schema_diagnostic_change_export(case)

    def unavailable(prefix):
        raise OSError("diagnostic storage unavailable")

    monkeypatch.setattr(storage, "correctness_logdir", unavailable)
    with pytest.raises(ValueError, match="exact original Qwen4") as failure:
        check.validate_local_inference_export(case.ctx, case.target)
    assert any("diagnostic storage unavailable" in note for note in failure.value.__notes__)
    assert not case.logs.exists()


@pytest.mark.parametrize("side", ["source", "export"])
def test_qwen4_schema_diagnostic_local_rejects_generation_change_before_comparison(
    _qwen4_schema_diagnostic_local, monkeypatch, side
):
    case = _qwen4_schema_diagnostic_local
    _qwen4_schema_diagnostic_change_export(case)
    original = export.read_inventory
    changed = case.source if side == "source" else case.target

    def read_then_change(root, index, **kwargs):
        schema = original(root, index, **kwargs)
        if Path(root) == case.target:
            config = changed / "config.json"
            config.write_text(config.read_text() + "\n")
        return schema

    monkeypatch.setattr(export, "read_inventory", read_then_change)
    with pytest.raises(ValueError, match="changed during Qwen4 schema validation"):
        check.validate_local_inference_export(case.ctx, case.target)
    assert not case.logs.exists()


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
    "invalid",
    [
        "onprem",
        "nonlocal",
        "authenticated",
        "other_check",
        "all_checks",
        "mixed_checks",
    ],
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
    monkeypatch.setattr(
        runner,
        "correctness_workdir",
        lambda *a: pytest.fail("invalid route allocated a workdir"),
    )
    monkeypatch.setattr(
        runner,
        "registered_tests",
        lambda: pytest.fail("invalid route reached model/check setup"),
    )
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
    monkeypatch.setattr(
        runner,
        "correctness_workdir",
        lambda *a: pytest.fail("drifted allocation created a workdir"),
    )
    monkeypatch.setattr(
        runner,
        "registered_tests",
        lambda: pytest.fail("drifted allocation reached checks"),
    )
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

_rl_config_TRAINING_CONFIG = {
    "max_seq_len": 4096,
    "n_gpus": 8,
    "attn_implementation": "flash_attention_3",
}

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
    monkeypatch.setattr(
        entry,
        "fn",
        lambda ctx: pytest.fail("unsupported check must not load a model or client"),
    )
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


def test_rl_checks_native_broadcast_configuration_reaches_both_cortex_workers(tmp_path, monkeypatch):
    cfg = load_config(_rl_config_CONFIG)
    before = cfg.native.model_dump()
    payload = rl_driver.build_payload(cfg.training, cfg.model_name, SEED)
    payload["training_config"].setdefault("ds_config", {})[broadcast_evidence.TRAINING_OPTION] = True
    configurations, closed = [], []

    class Client:
        def __init__(self, config):
            configurations.append(config)

        def shutdown(self):
            closed.append(True)

    monkeypatch.setattr(rl_driver, "RLClient", Client)
    with rl_driver.running_rl_client(
        cfg,
        payload,
        cfg.model_name,
        tmp_path,
        backend=CortexConfig(base_url="http://127.0.0.1:8003"),
        weight_sync_verification=True,
    ):
        pass
    native = configurations[0]
    roundtrip = CortexTrainingConfig.model_validate_json(native.model_dump_json())
    jobs = {job["job_type"]: job for job in roundtrip.to_cortex()}
    training = jobs["training"]["training_config"]
    sampling = jobs["sampling"]["inference_config"]
    assert training["ds_config"][broadcast_evidence.TRAINING_OPTION] is True
    assert sampling["vllm_config"]["extra_env"][rl_weight_sync.VERIFY_ENV] == "1"
    assert roundtrip.training_gpus == cfg.native.training_gpus
    assert roundtrip.sampling_gpus == cfg.native.sampling_gpus
    assert cfg.native.model_dump() == before
    assert closed == [True]


@pytest.mark.parametrize(
    "url,host,pat",
    [
        ("http://127.0.0.1:8003/private", None, None),
        ("http://127.0.0.1:8003?secret=x", None, None),
        ("http://127.0.0.1:8003", "hosted-account", None),
        ("http://127.0.0.1:8003", None, "secret"),
    ],
)
@pytest.mark.parametrize("check_id", ["rl-weight-sync", "rl-router-replay"])
def test_rl_checks_broadcast_rejects_decorated_or_hosted_local_routes(tmp_path, monkeypatch, url, host, pat, check_id):
    cfg = load_config(_rl_config_CONFIG)
    spec = Spec.read(_rl_config_SPEC)
    monkeypatch.setattr(
        registered_tests()[check_id],
        "fn",
        lambda ctx: pytest.fail("must reject before model work"),
    )
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback"):
        runner.execute_rl(
            cfg,
            spec,
            vocab_size=128,
            select=[check_id],
            workdir=tmp_path,
            client_backend=CortexConfig(base_url=url, host=host, pat=pat),
        )


def test_rl_checks_local_gateway_admits_both_observed_rl_checks(tmp_path, monkeypatch):
    cfg = load_config(_rl_config_CONFIG)
    before = cfg.native.model_dump()
    spec = Spec.read(_rl_config_SPEC)
    seen = []

    def weight_sync(ctx):
        seen.append(ctx.cfg.native.model_dump())
        return [
            rl_weight_sync.verdict(
                ctx,
                [],
                {
                    "sent_tensors": 1,
                    "sent_elements": 2,
                    "trainer_elements": 2,
                    "sampler_ranks": 1,
                },
                weight_format="vllm",
            )
        ]

    monkeypatch.setattr(registered_tests()["rl-weight-sync"], "fn", weight_sync)

    def router_replay(ctx):
        seen.append(ctx.cfg.native.model_dump())
        samples = _rl_checks_samples()
        failed, metrics = rl_router_replay.judge(samples, [_rl_checks_trace(samples)], full_recompute=True)
        return [rl_router_replay.verdict(ctx, failed, metrics)]

    monkeypatch.setattr(registered_tests()["rl-router-replay"], "fn", router_replay)
    results = runner.execute_rl(
        cfg,
        spec,
        vocab_size=128,
        select=["rl-weight-sync", "rl-router-replay"],
        workdir=tmp_path,
        client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
    )
    assert seen == [cfg.at_gpu_width(cfg.n_gpus).native.model_dump()] * 2
    assert cfg.native.model_dump() == before
    assert {result.test_id: result.outcome for result in results} == {
        "rl-weight-sync": Outcome.PASS,
        "rl-router-replay": Outcome.PASS,
    }


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
    sent = {
        "a.weight": _rl_checks_entry(4, "aa"),
        "b.weight": _rl_checks_entry(6, "bb"),
    }
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
                        {
                            "sent_manifest": sent,
                            "trainer_parameters": {"count": 2, "numel": 10},
                        },
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
        (
            lambda s: _rl_checks_rank(s, 1)["received"].pop("b.weight"),
            "sampler rank 1: b.weight not received",
        ),
        (
            lambda s: _rl_checks_rank(s, 0)["received"]["a.weight"].update(sha256="xx"),
            "different bytes",
        ),
        (
            lambda s: _rl_checks_rank(s, 0)["received"]["a.weight"].update(destinations=[]),
            "written to no parameter",
        ),
        (
            lambda s: _rl_checks_rank(s, 0)["destinations"]["a.weight"].update(uncovered=1),
            "no tensor wrote",
        ),
        (
            lambda s: _rl_checks_rank(s, 0)["destinations"]["b.weight"].update(mismatched=2),
            "differing from what was written",
        ),
        (
            lambda s: _rl_checks_rank(s, 0)["untouched_parameters"].update({"visual.w": 5}),
            "written by no tensor",
        ),
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
    rows = [
        ([5, 6, 7], 2, [[[1], [2]], [[3], [4]]]),
        ([8, 9, 4], 1, [[[5], [6]], [[7], [0]]]),
    ]
    samples = []
    for index, (tokens, prompt_len, routing) in enumerate(rows):
        marker = {
            "replay_id": f"rr1:{index}",
            "prompt_len": prompt_len,
            "generation_len": len(tokens) - prompt_len,
            "capture_len": len(routing),
            "routed_experts": routing,
        }
        samples.append(
            {
                "tokens": tokens,
                "prompt_len": prompt_len,
                "marker": marker,
                "routing": routing,
            }
        )
    return samples


def _rl_checks_trace(samples):
    tokens = samples[0]["tokens"] + samples[1]["tokens"]
    positions = [0, 1, 2, 0, 1, 2]
    routed = copy.deepcopy(samples[0]["routing"] + [[[0], [0]]] + samples[1]["routing"] + [[[0], [0]]])
    router = [
        {
            "layer": layer,
            "phase": phase,
            "replayed": True,
            "indices": [row[layer] for row in routed],
        }
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


@pytest.fixture
def _router_replay_evidence_case(tmp_path):
    replay_evidence.finish(0)
    model = torch.nn.Module()
    option = {
        "request_id": "request-1",
        "output_dir": "/data-fast/tmp/router-replay-selftest",
    }
    result = replay_evidence.maybe_configure_router_replay_trace(
        {replay_evidence.TRAINING_OPTION: option, "bf16": {"enabled": True}}, model, 0
    )
    assert result == {"bf16": {"enabled": True}}
    replay_evidence._publication["output_dir"] = tmp_path
    engine = SimpleNamespace(global_rank=0)
    kwargs = {
        "input_ids": torch.tensor([[1, 2]]),
        "position_ids": torch.tensor([[0, 1]]),
        "routed_experts": torch.tensor([[[[2]], [[3]]]]),
    }
    try:
        yield engine, kwargs, tmp_path
    finally:
        replay_evidence.finish(0)


def test_router_replay_evidence_disabled_helpers_leave_inputs_unchanged():
    replay_evidence.finish(0)
    config = {"bf16": {"enabled": True}}
    assert replay_evidence.maybe_configure_router_replay_trace(config, None, 0) is config
    replay_evidence.maybe_record_router_replay_received("sample", None)
    replay_evidence.maybe_begin_router_replay_call(None, None, backward=True)
    replay_evidence.maybe_complete_router_replay_call(None)
    assert replay_evidence.active() is None


@pytest.mark.parametrize(
    "option",
    [
        True,
        {},
        {"request_id": "../escape", "output_dir": "/data-fast/tmp"},
        {"request_id": "valid", "output_dir": "/code/tmp"},
    ],
)
def test_router_replay_evidence_rejects_invalid_option_before_installing_hooks(option):
    with pytest.raises(ValueError):
        replay_evidence.maybe_configure_router_replay_trace({replay_evidence.TRAINING_OPTION: option}, None, 0)
    assert replay_evidence.active() is None


def test_router_replay_evidence_publishes_only_after_successful_backward(
    _router_replay_evidence_case,
):
    engine, kwargs, output = _router_replay_evidence_case
    replay_evidence.maybe_record_router_replay_received("rr1:sample", torch.tensor([[[2]]]))
    replay_evidence.maybe_begin_router_replay_call(engine, kwargs, backward="loss_only")
    assert list(output.iterdir()) == []
    replay_evidence.maybe_complete_router_replay_call(engine)
    record = json.loads((output / "rank-0-call-0.json").read_text())
    assert record["schema"] == replay_evidence.TRACE_SCHEMA
    assert record["request_id"] == "request-1" and record["rank"] == record["call_index"] == 0
    assert record["received"] == {"rr1:sample": [[[2]]]}
    assert record["calls"][0]["input_ids"] == [1, 2]
    assert record["calls"][0]["routed_experts"] == [[[2]], [[3]]]
    assert record["calls"][0]["padding"] is False
    with pytest.raises(RuntimeError, match="unique preceding"):
        replay_evidence.maybe_complete_router_replay_call(engine)


def test_router_replay_evidence_rejects_incomplete_call_and_duplicate_rx(
    _router_replay_evidence_case,
):
    engine, kwargs, _ = _router_replay_evidence_case
    replay_evidence.maybe_record_router_replay_received("rr1:sample", torch.tensor([[[2]]]))
    with pytest.raises(ValueError, match="duplicate"):
        replay_evidence.maybe_record_router_replay_received("rr1:sample", torch.tensor([[[3]]]))
    replay_evidence.maybe_begin_router_replay_call(engine, kwargs, backward="loss_only")
    with pytest.raises(RuntimeError, match="no successful backward"):
        replay_evidence.maybe_begin_router_replay_call(engine, kwargs, backward="loss_only")


def test_router_replay_evidence_checks_rank_and_does_not_trace_forward_only(
    _router_replay_evidence_case,
):
    engine, kwargs, output = _router_replay_evidence_case
    replay_evidence.maybe_begin_router_replay_call(engine, kwargs, backward=False)
    assert replay_evidence.active().calls == [] and list(output.iterdir()) == []
    with pytest.raises(RuntimeError, match="rank differs"):
        replay_evidence.maybe_begin_router_replay_call(SimpleNamespace(global_rank=1), kwargs, backward=True)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.bfloat16)],
    ids=["cpu-float32", "cuda-float32", "cuda-bfloat16"],
)
@pytest.mark.parametrize("backward_path", ["tensor", "packed"])
def test_router_replay_evidence_hooks_actual_nonreentrant_router_recompute(
    monkeypatch, tmp_path, device, dtype, backward_path
):
    from torch.utils.checkpoint import checkpoint

    from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter
    from arctic_platform.rl.processors.packed_reduction import apply_packed_loss_reduction

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime required")
    if device == "cpu":
        histc = torch.histc
        monkeypatch.setattr(torch, "histc", lambda value, **kwargs: histc(value.float(), **kwargs))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = TokenChoiceTopKRouter(
                dim=8,
                num_experts=6,
                top_k=2,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
            )
            self.projection = torch.nn.Linear(8, 8, bias=False)

        def forward(self, hidden, routed):
            scores, _, _ = self.router(hidden, routed_experts=routed)
            return self.projection(hidden) * scores.sum(-1, keepdim=True)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Block()])
    model.to(device=device, dtype=dtype)
    routed = torch.tensor([[1, 2], [3, 4]], device=device)
    option = {
        "request_id": "actual-router-backward",
        "output_dir": "/data-fast/tmp/router-replay-selftest",
    }
    replay_evidence.maybe_configure_router_replay_trace({replay_evidence.TRAINING_OPTION: option}, model, 0)
    replay_evidence._publication["output_dir"] = tmp_path
    trace = replay_evidence.active()

    def backward(loss, *, scale_wrt_gas):
        assert scale_wrt_gas is False
        assert list(tmp_path.iterdir()) == []
        loss.backward()

    engine = SimpleNamespace(global_rank=0, backward=backward)
    try:
        replay_evidence.maybe_begin_router_replay_call(
            engine,
            {
                "input_ids": torch.tensor([[1, 2]], device=device),
                "position_ids": torch.tensor([[0, 1]], device=device),
                "routed_experts": routed[:, None, :],
            },
            backward="loss_only",
        )
        hidden = torch.randn(2, 8, device=device, dtype=dtype)
        loss = checkpoint(model.layers[0], hidden, routed, use_reentrant=False).sum()
        assert list(tmp_path.iterdir()) == []
        if backward_path == "packed":
            assert apply_packed_loss_reduction(engine, loss, 0.25, backward=True) is None
        else:
            loss.backward()
            replay_evidence.maybe_complete_router_replay_call(engine)
        observations = trace.calls[0]["router"]
        assert [entry["phase"] for entry in observations] == ["forward", "recompute"]
        assert all(entry["layer"] == 0 and entry["replayed"] for entry in observations)
        assert all(entry["indices"] == routed.tolist() for entry in observations)
        assert model.layers[0].projection.weight.grad is not None
        assert torch.isfinite(model.layers[0].projection.weight.grad).all()
        records = list(tmp_path.iterdir())
        assert len(records) == 1
        record = json.loads(records[0].read_text())
        assert record["request_id"] == option["request_id"]
        assert record["rank"] == record["call_index"] == 0
        assert record["calls"][0]["router"] == observations
    finally:
        replay_evidence.finish(0)


def _router_replay_collector_receipt(rank=0, calls=2, host="node-0"):
    return {
        "host": host,
        "request_id": "request-1",
        "records": [
            {
                "filename": f"rank-{rank}-call-{index}.json",
                "sha256": "retained-raw-artifact-digest",
                "record": {
                    "schema": replay_evidence.TRACE_SCHEMA,
                    "request_id": "request-1",
                    "rank": rank,
                    "call_index": index,
                    "received": {"sample": [[[1]]]},
                    "calls": [{"padding": False, "router": [], "input_ids": [index]}],
                    "unattributed_router_calls": [],
                },
            }
            for index in range(calls)
        ],
    }


def test_router_replay_evidence_collector_requires_every_rank_and_call():
    receipts = [_router_replay_collector_receipt(rank) for rank in range(2)]
    traces = router_replay_trace._merge_trace_receipts(receipts, request_id="request-1", ranks=2, calls=2)
    assert [trace["rank"] for trace in traces] == [0, 1]
    assert all([call["input_ids"] for call in trace["calls"]] == [[0], [1]] for trace in traces)
    with pytest.raises(ValueError, match="rank 1 lacks"):
        router_replay_trace._merge_trace_receipts(receipts[:1], request_id="request-1", ranks=2, calls=2)
    receipts[1]["records"].pop()
    with pytest.raises(ValueError, match="exact 2"):
        router_replay_trace._merge_trace_receipts(receipts, request_id="request-1", ranks=2, calls=2)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate"),
        (lambda rows: rows[0]["records"][0]["record"].update(rank=2), "identity"),
        (
            lambda rows: rows[0]["records"][0]["record"].update(call_index=True),
            "identity",
        ),
        (
            lambda rows: rows[0]["records"][0]["record"].update(request_id="old"),
            "identity",
        ),
        (lambda rows: rows[0]["records"][0]["record"].update(calls=[]), "exactly one"),
        (
            lambda rows: rows[0]["records"][1]["record"].update(received={}),
            "RX observations changed",
        ),
    ],
)
def test_router_replay_evidence_collector_rejects_incomplete_or_mixed_identity(mutation, expected):
    receipts = [_router_replay_collector_receipt()]
    mutation(receipts)
    with pytest.raises(ValueError, match=expected):
        router_replay_trace._merge_trace_receipts(receipts, request_id="request-1", ranks=1, calls=2)


@pytest.mark.parametrize("count", [None, 0, -1, True, 1.5])
def test_router_replay_evidence_collector_requires_actual_model_call_count(count):
    with pytest.raises(ValueError, match="actual rl_model_calls"):
        router_replay_trace.collect_traces({}, {"metrics": {"rl_model_calls": count}}, ranks=1)


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "token", "hostfile", "exit"])
def test_router_replay_evidence_collector_checks_actual_fanout(tmp_path, monkeypatch, fault):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("10.0.0.1 slots=1\n10.0.0.2 slots=1\n")
    monkeypatch.setattr(rl_driver, "_allocation_hostfile", lambda: hostfile)
    monkeypatch.setattr(storage, "require_data_fast_path", lambda path, **kwargs: tmp_path / "traces")
    monkeypatch.setattr(storage, "_ds_ssh", lambda: "/allocation/ds_ssh")
    commands = []

    def fanout(command, **kwargs):
        commands.append(command)
        assert command[:3] == ["/allocation/ds_ssh", "-f", str(hostfile)]
        encoded = command[3].split()[1]
        import base64

        script = base64.b64decode(encoded).decode()
        import ast

        call = next(line for line in script.splitlines() if line.startswith("receipt = "))
        arguments = ast.parse(call).body[0].value.args
        token = ast.literal_eval(arguments[-1])
        lines = []
        for rank in range(2):
            receipt = _router_replay_collector_receipt(rank, host=f"node-{rank}")
            receipt["token"] = "wrong" if fault == "token" else token
            lines.extend(
                f"10.0.0.{rank + 1}: {router_replay_trace._RECEIPT}{json.dumps(chunk)}"
                for chunk in router_replay_trace._receipt_chunks(receipt)
            )
        if fault == "missing":
            lines.pop()
        elif fault == "duplicate":
            lines.append(lines[0])
        elif fault == "hostfile":
            hostfile.write_text("10.0.0.3 slots=2\n")
        return SimpleNamespace(returncode=int(fault == "exit"), stdout="\n".join(lines), stderr="")

    monkeypatch.setattr(router_replay_trace.subprocess, "run", fanout)
    option = {"request_id": "request-1", "output_dir": "/data-fast/tmp/traces"}
    if fault:
        with pytest.raises((ValueError, RuntimeError), match="collector|collection"):
            router_replay_trace.collect_traces(option, {"metrics": {"rl_model_calls": 2}}, ranks=2)
        assert not (tmp_path / "traces" / "request-1-receipts.json").exists()
    else:
        traces = router_replay_trace.collect_traces(option, {"metrics": {"rl_model_calls": 2}}, ranks=2)
        assert [trace["rank"] for trace in traces] == [0, 1]
        assert len(json.loads((tmp_path / "traces" / "request-1-receipts.json").read_text())) == 2
    assert len(commands) == 1


def test_router_replay_evidence_collector_transfers_large_interleaved_receipts():
    receipts, rows = [], []
    for rank in range(2):
        receipt = _router_replay_collector_receipt(rank, host=f"node-{rank}")
        receipt["token"] = "transfer-token"
        receipt["records"][0]["record"]["received"]["sample"] = [[[index % 128] for index in range(180000)]]
        receipts.append(receipt)
        rows.extend((f"10.0.0.{rank + 1}", chunk) for chunk in router_replay_trace._receipt_chunks(receipt))
    assert len(json.dumps(receipts).encode()) > 2 * 1024 * 1024
    lines = [f"{host}: {router_replay_trace._RECEIPT}{json.dumps(chunk)}" for host, chunk in reversed(rows)]
    assert all(len(line.encode()) < 8192 for line in lines)
    actual = router_replay_trace._decode_trace_chunks(
        "transport banner\n" + "\n".join(lines),
        hosts=["10.0.0.1", "10.0.0.2"],
        request_id="request-1",
        token="transfer-token",
    )
    assert actual == receipts


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "data",
        "base64",
        "digest",
        "token",
        "request",
        "node",
        "host",
        "count",
        "index",
        "size",
    ],
)
def test_router_replay_evidence_collector_rejects_bad_transfer_chunks(fault):
    receipt = _router_replay_collector_receipt()
    receipt["token"] = "transfer-token"
    receipt["records"][0]["record"]["calls"][0]["input_ids"] = list(range(5000))
    chunks = list(router_replay_trace._receipt_chunks(receipt))
    assert len(chunks) > 2
    host = "10.0.0.1"
    if fault == "missing":
        chunks.pop(1)
    elif fault == "duplicate":
        chunks.append(copy.deepcopy(chunks[0]))
    elif fault == "data":
        chunks[0]["data"] = "A" + chunks[0]["data"][1:]
    elif fault == "base64":
        chunks[0]["data"] = "!" + chunks[0]["data"][1:]
    elif fault == "digest":
        for chunk in chunks:
            chunk["sha256"] = "0" * 64
    elif fault == "token":
        chunks[0]["token"] = "other-token"
    elif fault == "request":
        chunks[0]["request_id"] = "other-request"
    elif fault == "node":
        chunks[0]["host"] = "other-node"
    elif fault == "host":
        host = "10.0.0.2"
    elif fault == "count":
        chunks[0]["count"] += 1
    elif fault == "index":
        chunks[0]["index"] = True
    elif fault == "size":
        chunks[0]["data"] = chunks[0]["data"][:-4]
    output = "\n".join(f"{host}: {router_replay_trace._RECEIPT}{json.dumps(chunk)}" for chunk in chunks)
    with pytest.raises((ValueError, RuntimeError), match="collector|collection"):
        router_replay_trace._decode_trace_chunks(
            output,
            hosts=["10.0.0.1"],
            request_id="request-1",
            token="transfer-token",
        )


def test_router_replay_evidence_observes_rx_only_and_successful_packed_backward(
    _router_replay_evidence_case, monkeypatch
):
    from arctic_platform.rl.processors.packed_reduction import apply_packed_loss_reduction

    path = Path(__file__).resolve().parents[2] / "inference/server/router_replay/cache.py"
    spec = importlib.util.spec_from_file_location("_router_replay_cache_control", path)
    cache = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, cache)
    spec.loader.exec_module(cache)
    engine, kwargs, output = _router_replay_evidence_case
    tx, rx = cache.RouterReplayCacheTX("cpu", 1024), cache.RouterReplayCacheRX("cpu", 1024)
    routed = torch.tensor([[[2]]], dtype=torch.uint8)
    tx.put("rr1:sample", routed)
    tx.pop("rr1:sample")
    assert replay_evidence.active().received == {}
    rx.put("rr1:sample", routed)
    assert torch.equal(rx.pop("rr1:sample"), routed)
    assert replay_evidence.active().received == {"rr1:sample": [[[2]]]}
    replay_evidence.maybe_begin_router_replay_call(engine, kwargs, backward="loss_only")

    def failed_backward(*args, **kwargs):
        raise RuntimeError("actual backward failed")

    engine.backward = failed_backward
    with pytest.raises(RuntimeError, match="actual backward"):
        apply_packed_loss_reduction(engine, torch.tensor(1.0), 0.0, backward=True)
    assert list(output.iterdir()) == []
    engine.backward = lambda loss, **kwargs: None
    apply_packed_loss_reduction(engine, torch.tensor(1.0), 0.0, backward=True)
    assert len(list(output.iterdir())) == 1
    assert json.loads(next(output.iterdir()).read_text())["calls"][0]["padding"] is False


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda t: t["received"].update({"rr1:1": [[[5], [6]], [[7], [1]]]}),
            "routing received differs",
        ),
        (
            lambda t: t["calls"][0]["routed_experts"][1].__setitem__(0, [9]),
            "routed_experts differ",
        ),
        (
            lambda t: t["calls"][0]["router"][3].update(indices=[[9]] * 6),
            "layer 1 recompute tokens",
        ),
        (
            lambda t: t["calls"][0]["router"][0].update(replayed=False),
            "computed its own routing",
        ),
        (lambda t: t["calls"][0]["router"].pop(2), "layer 0 has no recompute"),
        (
            lambda t: t["unattributed_router_calls"].append({"layer": 0}),
            "outside a traced call",
        ),
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
    assert processing == {
        "post": ["compute_logprobs"],
        "loss_fn": "grpo",
        "config": {"loss_agg_mode": "token-mean"},
    }


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
    tokenizer_paths, configs, batches, generations, rates, closed, collected = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
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
                result = {
                    "token_ids": list(example.input_ids[example.n_prompt :]),
                    "router_replay": marker,
                }
                results.append(result)
                generated_samples.append(rl_router_replay.sample_from_result(prompt, result))
            return results

        def fwd_bwd(self, batch, *, router_replay=None):
            batches.append((batch, router_replay))
            return {"avg_loss": 1.0, "metrics": {"rl_model_calls": 1}}

        def step(self, learning_rate):
            rates.append(learning_rate)
            return {}

        def sync_weights(self):
            assert check is rl_weight_sync
            return _rl_checks_sync()

        def bootstrap_router_replay(self, *, max_cache_bytes=None):
            assert check is rl_router_replay
            assert max_cache_bytes is None

        def shutdown(self):
            closed.append(True)

    monkeypatch.setattr(check.gsm8k, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(check.gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", tokenize_examples)
    monkeypatch.setattr(rl_driver, "RLClient", Client)
    if check is rl_router_replay:

        def collect(option, response, *, ranks):
            assert not closed
            assert ranks == cfg.native.training_gpus
            assert response["metrics"]["rl_model_calls"] == 1
            assert option["output_dir"] == str(tmp_path / check.TEST_ID / "traces")
            collected.append(option)
            return [_rl_checks_trace(generated_samples[i : i + 2]) for i in range(0, len(generated_samples), 2)]

        monkeypatch.setattr(check, "collect_traces", collect)
    monkeypatch.setenv(rl_weight_sync.VERIFY_ENV, "test-original")
    result = check._run(ctx)
    assert result.outcome is Outcome.PASS
    assert tokenizer_paths == [cfg.model_name]
    assert len(configs) == len(batches) == len(closed) == 1
    native = configs[0]
    assert native.model_name == cfg.model_name
    assert native.to_onprem("training")["model_name"] == native.to_onprem("sampling")["model_name"] == cfg.model_name
    expected_sampling = copy.deepcopy(cfg.native.sampling.model_dump())
    if check is rl_weight_sync:
        expected_sampling["vllm"].setdefault("extra_env", {})[check.VERIFY_ENV] = "1"
    else:
        expected_sampling.setdefault("arctic_inference_config", {}).setdefault("router_replay", {})["enabled"] = True
    assert native.sampling.model_dump() == expected_sampling
    assert native.training_gpus == cfg.native.training_gpus == 8
    assert native.sampling_gpus == cfg.native.sampling_gpus == 8
    assert native.max_seq_len == cfg.native.max_seq_len == 32768
    expected_ds_config = copy.deepcopy(cfg.native.training.ds_config)
    if check is rl_weight_sync:
        expected_ds_config[broadcast_evidence.TRAINING_OPTION] = True
    else:
        assert len(collected) == 1
        expected_ds_config[replay_evidence.TRAINING_OPTION] = collected[0]
    assert native.training.ds_config == expected_ds_config
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
        assert native.training.ds_worker_config["router_replay"]["enabled"] is True
        assert replay == {
            "sampling_job_id": "native-sampling-job",
            "identity_version": 1,
        }
        assert len(generations) == 1
        assert rates == []
        assert batch["kwargs"]["rr_sample_ids"] == [sample["marker"]["replay_id"] for sample in generated_samples]
        assert batch["kwargs"]["rr_discard"] is True
    assert cfg.native.model_dump() == original_native
    assert cfg.training == original_training
    assert cfg.sub_jobs[1] == original_sampling
    assert os.environ[rl_weight_sync.VERIFY_ENV] == "test-original"


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


# Opt-in inference warmup observation.


class _InferenceWarmupObserverNewProcessGroup:
    def __init__(self, backend):
        self.backend = backend

    def _get_backend_name(self):
        return self.backend


@pytest.fixture
def _inference_warmup_observer_new_case(tmp_path, monkeypatch):
    warmup_observer._reset_inference_warmup_observer_for_tests()
    monkeypatch.delenv(warmup_observer.REQUEST_ENV, raising=False)
    calls = []

    modules = {}
    sources = {}
    for module_name in warmup_observer._SOURCE_MODULES:
        path = tmp_path / (module_name.replace(".", "-") + ".py")
        path.write_text(f"# authenticated fake source for {module_name}\n")
        module = ModuleType(module_name)
        module.__spec__ = SimpleNamespace(origin=str(path), loader=SimpleNamespace())
        module.__cached__ = None
        modules[module_name] = module
        sources[module_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setitem(sys.modules, module_name, module)

    def kernel_warmup(*args, **kwargs):
        calls.append(("kernel_warmup", args, kwargs))
        return ("kernel-result", args, kwargs)

    def flashinfer_autotune(*args, **kwargs):
        calls.append(("flashinfer_autotune", args, kwargs))
        return ("flashinfer-result", args, kwargs)

    class GroupCoordinator:
        def __init__(self):
            self.ranks = [0, 1]
            self.rank_in_group = 0
            self.unique_name = "world"
            self.cpu_group = _InferenceWarmupObserverNewProcessGroup("gloo")
            self.device_group = _InferenceWarmupObserverNewProcessGroup("nccl")

        def broadcast_object(self, obj=None, src=0):
            calls.append(("broadcast_object", obj, src))
            return ("object-result", obj, src)

        def broadcast_object_list(self, obj_list, src=0, group=None):
            calls.append(("broadcast_object_list", obj_list, src, group))
            return ("list-result", obj_list, src, group)

    warmup = modules["vllm.model_executor.warmup.kernel_warmup"]
    gpu_worker = modules["vllm.v1.worker.gpu_worker"]
    parallel = modules["vllm.distributed.parallel_state"]
    warmup.kernel_warmup = kernel_warmup
    warmup.flashinfer_autotune = flashinfer_autotune
    gpu_worker.kernel_warmup = kernel_warmup
    parallel.GroupCoordinator = GroupCoordinator

    distributed = ModuleType("torch.distributed")
    rank_state = {"rank": 0, "world_size": 2, "initialized": True}
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: rank_state["initialized"]
    distributed.get_rank = lambda: rank_state["rank"]
    distributed.get_world_size = lambda: rank_state["world_size"]
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    request = {
        "schema": "inference-warmup-observer-v1",
        "request_id": "e03-warmup-observer-new",
        "config_id": "qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k",
        "check_id": "inference-checkpoint-loss",
        "world_size": 2,
        "ranks": [0, 1],
        "sources": sources,
    }
    case = SimpleNamespace(
        calls=calls,
        modules=modules,
        paths={name: Path(module.__spec__.origin) for name, module in modules.items()},
        request=request,
        rank_state=rank_state,
        GroupCoordinator=GroupCoordinator,
    )
    try:
        yield case
    finally:
        warmup_observer._reset_inference_warmup_observer_for_tests()


def _inference_warmup_observer_new_enable(monkeypatch, request):
    monkeypatch.setenv(
        warmup_observer.REQUEST_ENV,
        json.dumps(request, sort_keys=True, separators=(",", ":")),
    )


def _inference_warmup_observer_new_records(capfd):
    return [
        json.loads(line)
        for line in capfd.readouterr().err.splitlines()
        if '"schema":"inference-warmup-observer-v1"' in line
    ]


def test_inference_warmup_observer_new_disabled_paths_are_exact_noops(monkeypatch):
    warmup_observer._reset_inference_warmup_observer_for_tests()
    monkeypatch.delenv(warmup_observer.REQUEST_ENV, raising=False)
    payload = {
        "inference_config": {
            "n_gpus": 2,
            "vllm_config": {"tensor_parallel_size": 2, "extra_env": {"EXISTING": "1"}},
        }
    }
    before = deepcopy(payload)
    warmup_observer.maybe_bind_inference_warmup_observer(
        payload,
        SimpleNamespace(config_id="ordinary", spec=SimpleNamespace(test_settings={})),
    )
    assert payload == before
    assert warmup_observer.maybe_install_inference_warmup_observer() is None
    assert warmup_observer._ACTIVE is None


def test_inference_warmup_observer_new_sampling_payload_binds_descriptor_before_ray_import(
    tmp_path,
):
    ctx = _inference_sampling_topology_context(topology=(16, 8, 2))
    sources = {}
    for module_name in warmup_observer._SOURCE_MODULES:
        source = tmp_path / (module_name.replace(".", "-") + ".py")
        source.write_text(module_name)
        sources[module_name] = hashlib.sha256(source.read_bytes()).hexdigest()
    request = {
        "schema": "inference-warmup-observer-v1",
        "request_id": "e03-future-useful-retry",
        "config_id": ctx.config_id,
        "check_id": "inference-checkpoint-loss",
        "world_size": 16,
        "ranks": list(range(16)),
        "sources": sources,
    }
    ctx.spec = SimpleNamespace(test_settings={"inference-checkpoint-loss": {"inference_warmup_observer": request}})

    payload = check._sampling_payload(ctx, "/saved/full-checkpoint", _inference_sampling_topology_TOKEN_BUDGET)

    extra_env = payload["inference_config"]["vllm_config"]["extra_env"]
    assert extra_env["RAY_DEDUP_LOGS"] == "0"
    assert json.loads(extra_env[warmup_observer.REQUEST_ENV]) == request
    assert payload["model_name"] == "/saved/full-checkpoint"
    _inference_sampling_topology_assert_topology(payload, (16, 8, 2))


@pytest.mark.parametrize(
    "damage",
    ["fields", "rank-gap", "world-size", "config", "dedup", "prebound"],
)
def test_inference_warmup_observer_new_binding_rejects_malformed_or_conflicting_requests(
    _inference_warmup_observer_new_case,
    damage,
):
    case = _inference_warmup_observer_new_case
    request = deepcopy(case.request)
    payload = {
        "inference_config": {
            "n_gpus": 2,
            "vllm_config": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
        }
    }
    ctx = SimpleNamespace(
        config_id=request["config_id"],
        spec=SimpleNamespace(test_settings={"inference-checkpoint-loss": {"inference_warmup_observer": request}}),
    )
    if damage == "fields":
        request["extra"] = True
    elif damage == "rank-gap":
        request["ranks"] = [0]
    elif damage == "world-size":
        request["world_size"] = 3
        request["ranks"] = [0, 1, 2]
    elif damage == "config":
        request["config_id"] = "different"
    elif damage == "dedup":
        payload["inference_config"]["vllm_config"]["extra_env"] = {"RAY_DEDUP_LOGS": "1"}
    else:
        payload["inference_config"]["vllm_config"]["extra_env"] = {warmup_observer.REQUEST_ENV: "{}"}

    with pytest.raises(ValueError):
        warmup_observer.maybe_bind_inference_warmup_observer(payload, ctx)


def test_inference_warmup_observer_new_install_rejects_source_digest_mismatch(
    _inference_warmup_observer_new_case,
    monkeypatch,
):
    case = _inference_warmup_observer_new_case
    request = deepcopy(case.request)
    request["sources"]["vllm.v1.worker.gpu_worker"] = "0" * 64
    original = case.modules["vllm.v1.worker.gpu_worker"].kernel_warmup
    _inference_warmup_observer_new_enable(monkeypatch, request)

    with pytest.raises(RuntimeError, match="digest differs"):
        warmup_observer.maybe_install_inference_warmup_observer()

    assert case.modules["vllm.v1.worker.gpu_worker"].kernel_warmup is original
    assert warmup_observer._ACTIVE is None


def test_inference_warmup_observer_new_preserves_calls_returns_groups_and_gap_free_sequence(
    _inference_warmup_observer_new_case,
    monkeypatch,
    capfd,
):
    case = _inference_warmup_observer_new_case
    _inference_warmup_observer_new_enable(monkeypatch, case.request)
    warmup_observer.maybe_install_inference_warmup_observer()
    gpu_worker = case.modules["vllm.v1.worker.gpu_worker"]
    warmup = case.modules["vllm.model_executor.warmup.kernel_warmup"]
    group = case.GroupCoordinator()
    payload = object()
    marker = object()

    assert gpu_worker.kernel_warmup(1, named=2) == (
        "kernel-result",
        (1,),
        {"named": 2},
    )
    assert warmup.flashinfer_autotune("shape", mode="exact") == (
        "flashinfer-result",
        ("shape",),
        {"mode": "exact"},
    )
    assert group.broadcast_object(payload, src=1) == (
        "object-result",
        payload,
        1,
    )
    values = [marker]
    assert group.broadcast_object_list(values, 1, marker) == (
        "list-result",
        values,
        1,
        marker,
    )

    assert case.calls == [
        ("kernel_warmup", (1,), {"named": 2}),
        ("flashinfer_autotune", ("shape",), {"mode": "exact"}),
        ("broadcast_object", payload, 1),
        ("broadcast_object_list", values, 1, marker),
    ]
    records = _inference_warmup_observer_new_records(capfd)
    assert [record["sequence"] for record in records] == list(range(len(records)))
    assert [(record["operation"], record["phase"]) for record in records] == [
        ("observer", "installed"),
        ("kernel_warmup", "enter"),
        ("kernel_warmup", "exit"),
        ("flashinfer_autotune", "enter"),
        ("flashinfer_autotune", "exit"),
        ("broadcast_object", "enter"),
        ("broadcast_object", "exit"),
        ("broadcast_object_list", "enter"),
        ("broadcast_object_list", "exit"),
    ]
    for record in records:
        assert record["rank"] == record["local_rank"] == 0
        assert record["world_size"] == 2
        assert "args" not in record and "result" not in record
    object_records = [record for record in records if record["operation"] == "broadcast_object"]
    assert all(
        record["group"]
        == {
            "backend": "gloo",
            "name": "world",
            "rank_in_group": 0,
            "ranks": [0, 1],
            "source_global_rank": 1,
            "source_rank_in_group": 1,
        }
        for record in object_records
    )
    list_records = [record for record in records if record["operation"] == "broadcast_object_list"]
    assert all(record["group"]["backend"] == "nccl" for record in list_records)


def test_inference_warmup_observer_new_preserves_baseexception_identity_and_full_traceback(
    _inference_warmup_observer_new_case,
    monkeypatch,
    capfd,
):
    case = _inference_warmup_observer_new_case

    class WarmupFailure(BaseException):
        pass

    failure = WarmupFailure("rank-local marker")

    def failing_kernel(*args, **kwargs):
        raise failure

    warmup = case.modules["vllm.model_executor.warmup.kernel_warmup"]
    gpu_worker = case.modules["vllm.v1.worker.gpu_worker"]
    warmup.kernel_warmup = failing_kernel
    gpu_worker.kernel_warmup = failing_kernel
    _inference_warmup_observer_new_enable(monkeypatch, case.request)
    warmup_observer.maybe_install_inference_warmup_observer()

    secret_argument = "-".join(("secret", "argument", "is", "not", "recorded"))
    with pytest.raises(WarmupFailure) as caught:
        gpu_worker.kernel_warmup(secret_argument)

    assert caught.value is failure
    records = _inference_warmup_observer_new_records(capfd)
    error = records[-1]
    assert error["operation"] == "kernel_warmup"
    assert error["phase"] == "error"
    assert error["source_generation_verified"] is True
    assert "failing_kernel" in error["traceback"]
    assert "WarmupFailure: rank-local marker" in error["traceback"]
    assert secret_argument not in json.dumps(records)


@pytest.mark.parametrize("mutation", ["replacement", "in-place"])
def test_inference_warmup_observer_new_rejects_source_generation_mutation_before_call(
    _inference_warmup_observer_new_case,
    monkeypatch,
    mutation,
):
    case = _inference_warmup_observer_new_case
    _inference_warmup_observer_new_enable(monkeypatch, case.request)
    warmup_observer.maybe_install_inference_warmup_observer()
    path = case.paths["vllm.v1.worker.gpu_worker"]
    if mutation == "replacement":
        path.rename(path.with_name("held-gpu-worker.py"))
        path.write_text("# replacement with authenticated-looking path\n")
    else:
        with path.open("r+b") as stream:
            stream.seek(0)
            stream.write(b"# changed in place\n")
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())

    with pytest.raises(RuntimeError, match="source changed"):
        case.modules["vllm.v1.worker.gpu_worker"].kernel_warmup()

    assert case.calls == []


def test_inference_warmup_observer_new_install_is_idempotent_and_wraps_direct_alias(
    _inference_warmup_observer_new_case,
    monkeypatch,
    capfd,
):
    case = _inference_warmup_observer_new_case
    _inference_warmup_observer_new_enable(monkeypatch, case.request)
    warmup_observer.maybe_install_inference_warmup_observer()
    warmup = case.modules["vllm.model_executor.warmup.kernel_warmup"]
    gpu_worker = case.modules["vllm.v1.worker.gpu_worker"]
    installed = gpu_worker.kernel_warmup

    warmup_observer.maybe_install_inference_warmup_observer()

    assert gpu_worker.kernel_warmup is installed
    assert warmup.kernel_warmup is installed
    assert installed() == ("kernel-result", (), {})
    records = _inference_warmup_observer_new_records(capfd)
    assert sum(record["operation"] == "observer" and record["phase"] == "installed" for record in records) == 1


def test_inference_warmup_observer_new_rejects_runtime_rank_mismatch_before_warmup(
    _inference_warmup_observer_new_case,
    monkeypatch,
):
    case = _inference_warmup_observer_new_case
    _inference_warmup_observer_new_enable(monkeypatch, case.request)
    warmup_observer.maybe_install_inference_warmup_observer()
    case.rank_state["world_size"] = 3

    with pytest.raises(RuntimeError, match="rank identity differs"):
        case.modules["vllm.v1.worker.gpu_worker"].kernel_warmup()

    assert case.calls == []


def test_inference_warmup_observer_new_install_record_reports_loader_cache_and_retention_limits(
    _inference_warmup_observer_new_case,
    monkeypatch,
    capfd,
):
    case = _inference_warmup_observer_new_case
    _inference_warmup_observer_new_enable(monkeypatch, case.request)
    warmup_observer.maybe_install_inference_warmup_observer()

    record = _inference_warmup_observer_new_records(capfd)[0]
    assert record["operation"] == "observer"
    assert record["phase"] == "installed"
    assert set(record["source_authentication"]) == set(warmup_observer._SOURCE_MODULES)
    for module_name, source in record["source_authentication"].items():
        assert source["sha256"] == case.request["sources"][module_name]
        assert source["loader"].endswith("SimpleNamespace")
        assert source["cache"] == {"status": "not_reported"}
        assert source["loaded_bytecode_proven"] is False
    assert "do not prove already-loaded bytecode" in record["limitation"]
    assert "complete logs before signals" in record["limitation"]


@pytest.fixture
def _strict_ssh_consumer_new_case(tmp_path):
    trust = tmp_path / "trust"
    raw = b"[10.0.0.2]:7878 ssh-ed25519 ORIGINAL\n"
    trust.write_bytes(raw)
    trust.chmod(0o400)
    node = {"host": "10.0.0.2", "python": sys.executable}
    admission = {
        "nodes": [node],
        "ssh": {
            "port": 7878,
            "known_hosts": str(trust),
            "known_hosts_sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    return admission, node, trust, raw


def _strict_ssh_consumer_new_path(argv):
    return next(item.split("=", 1)[1] for item in argv if item.startswith("UserKnownHostsFile="))


def test_strict_ssh_consumer_new_config_independent_route_and_exclusive_trust(
    _strict_ssh_consumer_new_case,
):
    admission, node, _, _ = _strict_ssh_consumer_new_case
    with export._strict_ssh_argv(admission, node, "owned command", time.monotonic() + 15) as argv:
        assert argv[:6] == ["ssh", "-T", "-p", "7878", "-F", "/dev/null"]
        for option in [
            "HostName=10.0.0.2",
            "ProxyCommand=none",
            "ProxyJump=none",
            "CanonicalizeHostname=no",
            "GlobalKnownHostsFile=/dev/null",
            "KnownHostsCommand=none",
            "UpdateHostKeys=no",
            "VerifyHostKeyDNS=no",
            "BatchMode=yes",
            "StrictHostKeyChecking=yes",
        ]:
            assert argv[argv.index(option) - 1] == "-o"
        assert not any(value.startswith("HostKeyAlias=") for value in argv)
        assert argv[-3:] == ["--", "10.0.0.2", "owned command"]


def test_strict_ssh_consumer_new_cpu_child_reads_parent_generation_without_inherited_fd(
    _strict_ssh_consumer_new_case,
):
    admission, node, _, raw = _strict_ssh_consumer_new_case
    with export._strict_ssh_argv(admission, node, "owned", time.monotonic() + 15) as argv:
        path = _strict_ssh_consumer_new_path(argv)
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
                path,
            ],
            close_fds=True,
            check=True,
            capture_output=True,
            timeout=5,
        )
        assert child.stdout == raw
    assert not Path(path).exists()


@pytest.mark.parametrize("operation", ["write", "grow", "shrink", "unseal"])
def test_strict_ssh_consumer_new_sealed_generation_refuses_same_uid_mutation(
    _strict_ssh_consumer_new_case,
    operation,
):
    admission, node, _, raw = _strict_ssh_consumer_new_case
    with export._strict_ssh_argv(admission, node, "owned", time.monotonic() + 15) as argv:
        path = _strict_ssh_consumer_new_path(argv)
        os.chmod(path, 0o600)
        fd = os.open(path, os.O_RDWR)
        try:
            with pytest.raises(OSError) as failure:
                if operation == "write":
                    os.write(fd, b"replacement")
                elif operation == "grow":
                    os.ftruncate(fd, len(raw) + 1)
                elif operation == "shrink":
                    os.ftruncate(fd, 0)
                else:
                    export.fcntl.fcntl(fd, export._SSH_ADD_SEALS, export._SSH_SEAL_WRITE)
            assert failure.value.errno == errno.EPERM
        finally:
            os.close(fd)
        assert Path(path).read_bytes() == raw


@pytest.mark.parametrize("mutation", ["replace", "content", "chmod", "hardlink", "parent"])
def test_strict_ssh_consumer_new_replacement_cannot_change_consumed_generation(
    _strict_ssh_consumer_new_case,
    mutation,
):
    admission, node, trust, raw = _strict_ssh_consumer_new_case
    with pytest.raises(ValueError, match="strict SSH trust.*changed"):
        with export._strict_ssh_argv(admission, node, "owned", time.monotonic() + 15) as argv:
            path = _strict_ssh_consumer_new_path(argv)
            if mutation == "replace":
                replacement = trust.with_name("replacement")
                replacement.write_bytes(b"OTHER")
                replacement.replace(trust)
            elif mutation == "content":
                trust.chmod(0o600)
                trust.write_bytes(b"OTHER")
            elif mutation == "chmod":
                trust.chmod(0o600)
            elif mutation == "hardlink":
                os.link(trust, trust.with_name("alias"))
            else:
                moved = trust.parent.with_name(trust.parent.name + "-moved")
                trust.parent.rename(moved)
                trust.parent.mkdir()
            assert Path(path).read_bytes() == raw
    assert not Path(path).exists()


def test_strict_ssh_consumer_new_trust_drift_preserves_primary_child_failure(
    _strict_ssh_consumer_new_case,
):
    admission, node, trust, _ = _strict_ssh_consumer_new_case
    with pytest.raises(RuntimeError, match="primary child") as failure:
        with export._strict_ssh_argv(admission, node, "owned", time.monotonic() + 15) as argv:
            path = _strict_ssh_consumer_new_path(argv)
            trust.chmod(0o600)
            trust.write_bytes(b"changed")
            raise RuntimeError("primary child failure")
    assert "Additional strict SSH trust failure" in failure.value.__notes__[0]
    assert not Path(path).exists()


@pytest.mark.parametrize(
    "invalid",
    ["wrong_hash", "symlink", "parent_symlink", "writable", "empty", "hardlink"],
)
def test_strict_ssh_consumer_new_unpinned_generation_fails_before_consumer(
    _strict_ssh_consumer_new_case,
    invalid,
):
    admission, node, trust, _ = _strict_ssh_consumer_new_case
    if invalid == "wrong_hash":
        admission["ssh"]["known_hosts_sha256"] = "0" * 64
    elif invalid == "symlink":
        target = trust.with_name("target")
        trust.rename(target)
        trust.symlink_to(target)
    elif invalid == "parent_symlink":
        alias = trust.parent.with_name(trust.parent.name + "-alias")
        alias.symlink_to(trust.parent, target_is_directory=True)
        admission["ssh"]["known_hosts"] = str(alias / trust.name)
    elif invalid == "writable":
        trust.chmod(0o622)
    elif invalid == "empty":
        trust.chmod(0o600)
        trust.write_bytes(b"")
    else:
        os.link(trust, trust.with_name("alias"))
    with pytest.raises((ValueError, OSError)):
        with export._strict_ssh_argv(admission, node, "owned", time.monotonic() + 15):
            pytest.fail("invalid generation reached consumer")


@pytest.mark.parametrize("invalid", ["foreign_node", "option_host", "boolean_port", "expired"])
def test_strict_ssh_consumer_new_unadmitted_endpoint_or_expired_budget(
    _strict_ssh_consumer_new_case,
    invalid,
):
    admission, node, _, _ = _strict_ssh_consumer_new_case
    budget = time.monotonic() + 15
    if invalid == "foreign_node":
        node = {"host": "other-job"}
    elif invalid == "option_host":
        node["host"] = "-proxy"
    elif invalid == "boolean_port":
        admission["ssh"]["port"] = True
    else:
        budget = time.monotonic() - 1
    with pytest.raises((ValueError, TimeoutError)):
        with export._strict_ssh_argv(admission, node, "owned", budget):
            pytest.fail("invalid endpoint reached consumer")


def test_strict_ssh_consumer_new_closes_every_descriptor_after_one_close_failure(
    _strict_ssh_consumer_new_case,
    monkeypatch,
):
    admission, node, _, _ = _strict_ssh_consumer_new_case
    opened, closed = [], []
    original_open, original_memfd, original_close = os.open, export._ssh_memfd, os.close

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def tracked_memfd(*args, **kwargs):
        fd = original_memfd(*args, **kwargs)
        opened.append(fd)
        return fd

    def tracked_close(fd):
        original_close(fd)
        closed.append(fd)
        if len(closed) == 1:
            raise OSError("injected close failure")

    monkeypatch.setattr(export.os, "open", tracked_open)
    monkeypatch.setattr(export, "_ssh_memfd", tracked_memfd)
    monkeypatch.setattr(export.os, "close", tracked_close)
    with pytest.raises(OSError, match="injected close failure"):
        with export._strict_ssh_argv(admission, node, "owned", time.monotonic() + 15):
            pass
    assert closed == opened[::-1]


def test_strict_ssh_consumer_new_inference_worker_context_keeps_trust_alive(
    _strict_ssh_consumer_new_case,
):
    admission, node, _, raw = _strict_ssh_consumer_new_case
    with export._local_ssh_argv(admission, node, {"timeout_s": 5}, time.monotonic() + 15) as argv:
        path = _strict_ssh_consumer_new_path(argv)
        assert Path(path).read_bytes() == raw
        assert "supervise" in argv[-1]
    assert not Path(path).exists()


def test_strict_ssh_publisher_new_exact_created_generation_consumed_by_final_context(
    tmp_path,
):
    path = tmp_path / "created-trust"
    raw = b"[peer]:7878 ssh-ed25519 ALREADY_AUTHENTICATED\n"
    result = export._publish_ssh_trust(path, raw, time.monotonic() + 15)
    assert result == {
        "known_hosts": str(path),
        "known_hosts_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert path.stat().st_mode & 0o777 == 0o400
    node = {"host": "peer"}
    with export._strict_ssh_argv(
        {"nodes": [node], "ssh": {"port": 7878, **result}},
        node,
        "owned",
        time.monotonic() + 15,
    ) as argv:
        assert Path(_strict_ssh_consumer_new_path(argv)).read_bytes() == raw


@pytest.mark.parametrize("existing", ["file", "symlink", "directory"])
def test_strict_ssh_publisher_new_never_replaces_existing_slot(tmp_path, existing):
    path = tmp_path / "created-trust"
    if existing == "file":
        path.write_bytes(b"original")
    elif existing == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"original")
        path.symlink_to(target)
    else:
        path.mkdir()
    before = path.lstat()
    with pytest.raises(FileExistsError):
        export._publish_ssh_trust(path, b"authorized", time.monotonic() + 15)
    assert path.lstat() == before
    if existing != "directory":
        assert path.read_bytes() == b"original"


def test_strict_ssh_publisher_new_descriptor_chmod_cannot_mutate_replacement(tmp_path, monkeypatch):
    path = tmp_path / "created-trust"
    moved = tmp_path / "created-original"
    original_chmod = os.fchmod

    def substitute(fd, mode):
        path.rename(moved)
        path.write_bytes(b"replacement")
        path.chmod(0o600)
        original_chmod(fd, mode)

    monkeypatch.setattr(export.os, "fchmod", substitute)
    with pytest.raises(ValueError, match="publication slot or file identity changed"):
        export._publish_ssh_trust(path, b"authorized", time.monotonic() + 15)
    assert moved.stat().st_mode & 0o777 == 0o400
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_bytes() == b"replacement"


@pytest.mark.parametrize("mutation", ["content", "hardlink", "parent"])
def test_strict_ssh_publisher_new_rejects_drift_before_returning_any_binding(tmp_path, monkeypatch, mutation):
    path = tmp_path / "created-trust"
    original_sync = os.fsync

    def mutate(fd):
        original_sync(fd)
        if mutation == "content":
            path.chmod(0o600)
            path.write_bytes(b"changed-trust")
        elif mutation == "hardlink":
            os.link(path, tmp_path / "alias")
        else:
            tmp_path.rename(tmp_path.with_name(tmp_path.name + "-moved"))
            tmp_path.mkdir()

    monkeypatch.setattr(export.os, "fsync", mutate)
    with pytest.raises(ValueError, match="strict SSH trust.*changed"):
        export._publish_ssh_trust(path, b"authorized", time.monotonic() + 15)


@pytest.mark.parametrize("invalid", ["empty", "text", "oversized", "outside_fast", "parent_symlink"])
def test_strict_ssh_publisher_new_invalid_inputs_fail_before_slot_creation(tmp_path, invalid):
    path = tmp_path / "created-trust"
    raw = b"authorized"
    if invalid == "empty":
        raw = b""
    elif invalid == "text":
        raw = "not original bytes"
    elif invalid == "oversized":
        raw = b"x" * (4 * 1024**2 + 1)
    elif invalid == "outside_fast":
        path = Path("/code") / "not-created-trust"
    else:
        alias = tmp_path.with_name(tmp_path.name + "-alias")
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / path.name
    with pytest.raises((ValueError, OSError)):
        export._publish_ssh_trust(path, raw, time.monotonic() + 15)
    assert not path.exists()


def test_strict_ssh_publisher_new_interrupted_write_retains_partial_evidence_and_closes_fd(tmp_path, monkeypatch):
    path = tmp_path / "created-trust"
    seen = []

    def interrupted(fd, raw):
        seen.append(fd)
        raise OSError("write interrupted")

    monkeypatch.setattr(export.os, "write", interrupted)
    with pytest.raises(OSError, match="write interrupted"):
        export._publish_ssh_trust(path, b"authorized", time.monotonic() + 15)
    assert path.is_file() and path.read_bytes() == b""
    with pytest.raises(OSError):
        os.fstat(seen[0])


def test_strict_ssh_publisher_new_libc_creation_failure_preserves_errno(monkeypatch):
    import ctypes

    class Create:
        def __call__(self, name, flags):
            assert name == b"admitted-ssh-trust" and flags == 3
            return -1

    class Libc:
        memfd_create = Create()

    monkeypatch.delattr(export.os, "memfd_create", raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: Libc())
    monkeypatch.setattr(ctypes, "get_errno", lambda: errno.EPERM)
    with pytest.raises(OSError) as rejected:
        export._ssh_memfd()
    assert rejected.value.errno == errno.EPERM


@pytest.mark.parametrize("worker_failure", [False, True])
def test_strict_ssh_lifetime_new_export_pipe_keeps_generation_until_owned_cleanup(
    _strict_ssh_consumer_new_case,
    tmp_path,
    monkeypatch,
    worker_failure,
):
    admission, node, _, raw = _strict_ssh_consumer_new_case
    admission["nodes"] = [{"host": "head", "python": sys.executable}, node]
    root = tmp_path / "saved-export"
    root.mkdir()
    manifest = {"files": []}
    receipt = {
        "semantic_exit": 0,
        "allocation_id": "same-job",
        "node": node,
        "root": str(root),
        "manifest_sha256": export.schema_sha(manifest),
        "files_verified": 0,
    }
    admission["allocation_id"] = "same-job"
    paths, stopped = [], []

    class Child:
        returncode = 0

        def __init__(self, argv, **kwargs):
            self.stdout = io.BytesIO()
            if argv[0] == "ssh":
                paths.append(_strict_ssh_consumer_new_path(argv))
                assert Path(paths[0]).read_bytes() == raw

        def communicate(self, timeout):
            assert Path(paths[0]).read_bytes() == raw
            if worker_failure:
                raise RuntimeError("new export initiating worker failure")
            return json.dumps(receipt).encode(), None

        def wait(self, timeout):
            assert Path(paths[0]).read_bytes() == raw

    def stop(children):
        assert len(children) == 2 and Path(paths[0]).read_bytes() == raw
        stopped.extend(children)

    monkeypatch.setattr(export, "_validate_local_allocation", lambda *args: admission["nodes"])
    monkeypatch.setattr(
        export,
        "_bounded_local_json",
        lambda *args: {"manifest": manifest, "head_files": {}},
    )
    monkeypatch.setattr(export.subprocess, "Popen", Child)
    monkeypatch.setattr(export, "_stop_local_children", stop)
    if worker_failure:
        with pytest.raises(RuntimeError, match="new export initiating worker failure"):
            export.distribute_local_inference_export(root, admission, timeout_s=15)
    else:
        result = export.distribute_local_inference_export(root, admission, timeout_s=15)
        assert result["remote_receipts"] == [receipt]
    assert len(stopped) == 2 and not Path(paths[0]).exists()


def _durable_startup_error_new_directory_identity(path):
    metadata = path.stat()
    return [
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    ]


def _durable_startup_error_new_request(root, config_id):
    return {
        "schema": "inference-startup-error-v1",
        "request_id": "future-e03-useful-retry",
        "config_id": config_id,
        "check_id": "inference-checkpoint-loss",
        "generation": "generation-1",
        "root": {
            "path": str(root),
            "identity": _durable_startup_error_new_directory_identity(root),
            "parent_identity": _durable_startup_error_new_directory_identity(root.parent),
        },
        "sources": {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in startup_error_observer._SOURCE_PATHS.items()
        },
    }


def test_durable_startup_error_new_sampling_payload_binds_static_request(
    tmp_path,
):
    root = tmp_path / "startup-error-records"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    ctx = _inference_sampling_topology_context(topology=(16, 8, 2))
    request = _durable_startup_error_new_request(root, ctx.config_id)
    ctx.spec = SimpleNamespace(
        test_settings={
            "inference-checkpoint-loss": {
                "inference_startup_error": request,
            }
        }
    )

    payload = check._sampling_payload(
        ctx,
        "/saved/full-checkpoint",
        _inference_sampling_topology_TOKEN_BUDGET,
    )

    extra_env = payload["inference_config"]["vllm_config"]["extra_env"]
    assert json.loads(extra_env[startup_error_observer.REQUEST_ENV]) == request
    assert startup_error_observer.ATTEMPT_ENV not in extra_env
    assert payload["model_name"] == "/saved/full-checkpoint"
    _inference_sampling_topology_assert_topology(payload, (16, 8, 2))


def test_durable_startup_error_new_disabled_binding_keeps_payload_exact(
    tmp_path,
):
    payload = {
        "inference_config": {
            "vllm_config": {
                "extra_env": {"EXISTING": "unchanged"},
            }
        }
    }
    before = deepcopy(payload)
    ctx = SimpleNamespace(
        config_id="ordinary",
        spec=SimpleNamespace(test_settings={}),
    )

    startup_error_observer.maybe_bind_inference_startup_error(payload, ctx)

    assert payload == before


@pytest.mark.parametrize(
    "damage",
    [
        "config",
        "check",
        "generation",
        "source",
        "root-link",
        "request-env",
        "attempt-env",
    ],
)
def test_durable_startup_error_new_binding_rejects_invalid_identity_or_collision(
    tmp_path,
    damage,
):
    root = tmp_path / "startup-error-records"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    config_id = "qwen35-e03"
    request = _durable_startup_error_new_request(root, config_id)
    if damage == "config":
        request["config_id"] = "other-config"
    elif damage == "check":
        request["check_id"] = "other-check"
    elif damage == "generation":
        request["generation"] = "invalid/generation"
    elif damage == "source":
        request["sources"]["worker"] = "0" * 64
    elif damage == "root-link":
        alias = tmp_path / "startup-error-alias"
        alias.symlink_to(root, target_is_directory=True)
        request["root"]["path"] = str(alias)
    payload = {
        "inference_config": {
            "vllm_config": {
                "extra_env": {},
            }
        }
    }
    if damage == "request-env":
        payload["inference_config"]["vllm_config"]["extra_env"][startup_error_observer.REQUEST_ENV] = "{}"
    elif damage == "attempt-env":
        payload["inference_config"]["vllm_config"]["extra_env"][startup_error_observer.ATTEMPT_ENV] = "{}"
    before = deepcopy(payload)
    ctx = SimpleNamespace(
        config_id=config_id,
        spec=SimpleNamespace(
            test_settings={
                "inference-checkpoint-loss": {
                    "inference_startup_error": request,
                }
            }
        ),
    )

    with pytest.raises((RuntimeError, ValueError)):
        startup_error_observer.maybe_bind_inference_startup_error(
            payload,
            ctx,
        )

    assert payload == before
