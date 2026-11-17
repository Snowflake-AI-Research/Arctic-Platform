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
"""CPU tests of DFlash2 online training: mask alignment, packed rows, the loader, and the draft export."""

from __future__ import annotations

import copy
import json
import logging
import shutil

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM

from arctic_platform.model import PEFT_ADAPTER_DIRNAME
from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import Patches
from arctic_platform.model import build_model
from arctic_platform.model.implementations.dflash2.online import DFlash2OnlineModel
from arctic_platform.model.implementations.dflash2.online import build_online_model
from arctic_platform.model.implementations.dflash2.online import trainable_row_mask
from arctic_platform.model.implementations.dflash2.specforge.algorithms.common import dflash_family_model
from arctic_platform.model.loaders import dflash2 as dflash2_loader
from arctic_platform.rl.processors.dflash2 import _supervised_token_weight
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

VOCAB = 64
HIDDEN = 32

DRAFT_CONFIG = dict(
    architectures=["DFlash2DraftModel"],
    model_type="qwen3",
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    num_target_layers=4,
    dflash_config=dict(
        block_size=4,
        target_layer_ids=[1, 2],
        mask_token_id=3,
        conv_kernel_size=3,
        conv_group_size=4,
        selector_rank=4,
        selector_top_k=4,
    ),
)

# (token ids, platform loss_mask). The mask is prediction aligned: entry i weights the logits at i that predict token
# i + 1, so a response of tokens [A, B] after two prompt tokens is supervised at [0, 1, 1, 0].
PROMPT_THEN_TWO_TOKEN_RESPONSE = ([10, 11, 12, 13], [0, 1, 1, 0])
PROMPT_THEN_THREE_TOKEN_RESPONSE = ([20, 21, 22, 23], [1, 1, 1, 0])
LONGER_ROW_WITH_TWO_TOKEN_RESPONSE = ([40, 41, 42, 43, 44], [0, 0, 1, 1, 0])
# One response token has no successor to serve as a block target, so the row cannot be trained.
PROMPT_THEN_ONE_TOKEN_RESPONSE = ([30, 31, 32], [0, 1, 0])


def packed_window(*rows):
    """One ``[1, T]`` window with ``position_ids`` restarting at every row, as the data plane delivers it."""
    ids = [token for row_ids, _ in rows for token in row_ids]
    mask = [weight for _, row_mask in rows for weight in row_mask]
    positions = [position for row_ids, _ in rows for position in range(len(row_ids))]
    return dict(
        input_ids=torch.tensor([ids]),
        position_ids=torch.tensor([positions]),
        loss_mask=torch.tensor([mask], dtype=torch.float32),
    )


def tiny_target_config():
    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )


@pytest.fixture
def target():
    torch.manual_seed(0)
    return Qwen3ForCausalLM(tiny_target_config()).eval()


def make_online(target, **kwargs):
    options = dict(dtype=torch.float32, attention_backend="sdpa", num_anchors=4, seed=0)
    options.update(kwargs)
    return build_online_model(target, copy.deepcopy(DRAFT_CONFIG), **options)


@pytest.fixture
def online(target):
    return make_online(target).train()


def spy_on_specforge(model, monkeypatch):
    """Record the batch that SpecForge's ``OnlineDFlashModel`` is called with.

    The record holds ``input_ids``, ``loss_mask`` and ``target_last_hidden_states`` (``None`` when not passed), and stays
    empty until the model is called.
    """
    seen = {}
    original = model.online_model.forward

    def spy(input_ids, hidden_states, loss_mask, **kwargs):
        seen["input_ids"] = input_ids.clone()
        seen["loss_mask"] = loss_mask.clone()
        seen["target_last_hidden_states"] = kwargs.get("target_last_hidden_states")
        return original(input_ids, hidden_states, loss_mask, **kwargs)

    monkeypatch.setattr(model.online_model, "forward", spy)
    return seen


@pytest.fixture
def seen_by_specforge(online, monkeypatch):
    return spy_on_specforge(online, monkeypatch)


@pytest.mark.parametrize(
    "prediction_mask, token_mask",
    [
        ([0, 1, 1, 0], [0, 0, 1, 1]),
        ([1, 1, 1, 0], [0, 1, 1, 1]),
        ([0, 0, 1, 1, 0], [0, 0, 0, 1, 1]),
        ([0, 1, 0.75, 0], [0, 0, 1, 0.75]),  # fractional weights are preserved
    ],
)
def test_trainable_row_mask_moves_prediction_alignment_to_token_alignment(prediction_mask, token_mask):
    result = trainable_row_mask(torch.tensor([prediction_mask], dtype=torch.float32))
    assert result.tolist() == [token_mask]


@pytest.mark.parametrize(
    "prediction_mask",
    [
        [0, 1, 0],  # one response token: an anchor with no supervised successor
        [0, 0, 0, 0],
        [1, 0, 1, 0],  # supervised tokens that are never adjacent
        [0, 0.2, 0.2, 0],  # below the 0.5 threshold SpecForge uses to mark a supervised token
        [1, 1],  # the first token of a row is never supervised, so two tokens cannot hold an anchor and a target
    ],
)
def test_trainable_row_mask_is_none_when_no_anchor_has_a_supervised_successor(prediction_mask):
    assert trainable_row_mask(torch.tensor([prediction_mask], dtype=torch.float32)) is None


def test_forward_gives_specforge_the_token_aligned_mask_of_each_packed_row(online, seen_by_specforge):
    window = packed_window(
        PROMPT_THEN_TWO_TOKEN_RESPONSE,
        PROMPT_THEN_ONE_TOKEN_RESPONSE,
        LONGER_ROW_WITH_TWO_TOKEN_RESPONSE,
    )

    online(**window)

    # The one-token row is dropped and the others are right-padded to the longest kept row.
    assert seen_by_specforge["input_ids"].tolist() == [[10, 11, 12, 13, 0], [40, 41, 42, 43, 44]]
    assert seen_by_specforge["loss_mask"].tolist() == [[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]


def test_forward_supervises_the_final_response_token(online, seen_by_specforge):
    """With the platform mask [0, 1, 1, 0] for response [A, B], B is supervised and the last prompt token is not."""
    online(**packed_window(PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_THREE_TOKEN_RESPONSE))

    first_row = seen_by_specforge["loss_mask"][0].tolist()
    assert first_row[-1] == 1
    assert first_row[1] == 0


def test_forward_trains_only_the_draft(online):
    out = online(**packed_window(PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_THREE_TOKEN_RESPONSE))

    assert torch.isfinite(out["loss"])
    assert out["dflash2_terms"]
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in online.draft.parameters())
    assert all(p.grad is None for p in online.target.parameters())
    assert not any(p.requires_grad for p in online.target.parameters())


def teacher_terms(terms):
    return {name: pair for name, pair in terms.items() if name.startswith("dflash/teacher/")}


def test_forward_reports_teacher_terms_for_a_window_with_a_skipped_row(target, monkeypatch):
    online = make_online(target, teacher_metrics=True).train()
    seen = spy_on_specforge(online, monkeypatch)
    rows = (PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_ONE_TOKEN_RESPONSE, LONGER_ROW_WITH_TWO_TOKEN_RESPONSE)

    out = online(**packed_window(*rows))

    top_k = DRAFT_CONFIG["dflash_config"]["selector_top_k"]
    terms = teacher_terms(out["dflash2_terms"])
    assert terms.keys() == {
        "dflash/teacher/unary_overlap_chain_length",
        "dflash/teacher/unary_distribution_overlap",
        "dflash/teacher/unary_top1_agreement",
        f"dflash/teacher/unary_top{top_k}_mass",
    }
    for name, (numerator, denominator) in terms.items():
        assert torch.isfinite(numerator) and torch.isfinite(denominator), name
        assert denominator > 0, name  # computed over supervised positions, not an empty reduction

    # The target's final hidden states reach SpecForge in the order of the rows it trains on: the skipped row, which
    # the target is never run on for the teacher, does not shift the rows after it.
    kept_rows = (rows[0], rows[2])
    given = seen["target_last_hidden_states"]
    assert given.shape[0] == len(kept_rows)
    assert given.shape[1] == seen["input_ids"].shape[1]
    for index, (row_ids, _) in enumerate(kept_rows):
        _, last_hidden = online.target_features(torch.tensor([row_ids]))
        torch_assert_equal(given[index, : len(row_ids)], last_hidden[0])
        assert given[index, len(row_ids) :].abs().sum() == 0  # right padding, as for the ids


@pytest.mark.parametrize(
    "teacher_metrics, collect_detailed_metrics",
    [
        (False, True),  # the default: the teacher metrics are off
        (True, False),  # they are a detailed diagnostic
        (False, False),
    ],
)
def test_forward_skips_the_teacher_terms_unless_they_are_both_enabled_and_collected(
    target, monkeypatch, teacher_metrics, collect_detailed_metrics
):
    online = make_online(target, teacher_metrics=teacher_metrics).train()
    seen = spy_on_specforge(online, monkeypatch)

    out = online(
        **packed_window(PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_THREE_TOKEN_RESPONSE),
        collect_detailed_metrics=collect_detailed_metrics,
    )

    assert torch.isfinite(out["loss"])
    assert seen["target_last_hidden_states"] is None
    assert teacher_terms(out["dflash2_terms"]) == {}


def test_forward_returns_a_connected_zero_when_no_row_can_form_a_block(online):
    out = online(**packed_window(PROMPT_THEN_ONE_TOKEN_RESPONSE, ([50, 51, 52], [0, 0, 0])))

    assert out["loss"].item() == 0.0
    assert out["loss"].requires_grad
    assert out["dflash2_terms"] == {}
    out["loss"].backward()


@pytest.mark.parametrize(
    "rows",
    [
        [PROMPT_THEN_TWO_TOKEN_RESPONSE],
        [PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_ONE_TOKEN_RESPONSE],
        [PROMPT_THEN_ONE_TOKEN_RESPONSE, PROMPT_THEN_THREE_TOKEN_RESPONSE, LONGER_ROW_WITH_TWO_TOKEN_RESPONSE],
        [PROMPT_THEN_ONE_TOKEN_RESPONSE],
    ],
)
def test_reduction_weight_is_the_mass_the_model_trains_on(online, seen_by_specforge, rows):
    """The packed-loss weight and the model must agree on which rows count and what they weigh."""
    window = packed_window(*rows)

    online(**window)

    trained_mass = seen_by_specforge["loss_mask"].sum().item() if seen_by_specforge else 0.0
    assert _supervised_token_weight(window) == pytest.approx(trained_mass)


def test_save_pretrained_writes_the_draft_that_init_from_reloads(target, tmp_path):
    trained = make_online(target, seed=0)
    export_dir = tmp_path / "draft"

    trained.save_pretrained(export_dir)

    assert sorted(path.name for path in export_dir.iterdir()) == ["config.json", "model.safetensors"]
    assert json.loads((export_dir / "config.json").read_text())["architectures"] == ["DFlash2DraftModel"]
    assert not any(key.startswith("target") for key in load_file(export_dir / "model.safetensors"))

    different_init = make_online(target, seed=1)
    assert not all(
        torch.equal(a, b)
        for a, b in zip(trained.draft.state_dict().values(), different_init.draft.state_dict().values())
    )
    reloaded = make_online(target, seed=1, init_from=export_dir)
    for name, value in trained.draft.state_dict().items():
        assert torch.equal(value, reloaded.draft.state_dict()[name]), name


def consolidated_state(online_model):
    """What the DeepSpeed worker hands ``save_pretrained``: full-precision weights keyed by the wrapper module."""
    return {name: value.detach().clone().float() for name, value in online_model.state_dict().items()}


def test_save_pretrained_exports_the_state_dict_the_worker_gathers_not_the_live_parameters(target, tmp_path):
    online = make_online(target, seed=0)
    consolidated = consolidated_state(online)
    with torch.no_grad():  # under ZeRO-3 the live parameters are empty shards
        for parameter in online.draft.parameters():
            parameter.zero_()

    online.save_pretrained(tmp_path, state_dict=consolidated, safe_serialization=True)

    exported = load_file(tmp_path / "model.safetensors")
    prefix = "online_model.draft_model."
    expected = {name.removeprefix(prefix): value for name, value in consolidated.items() if name.startswith(prefix)}
    assert exported.keys() == {name for name in expected if "rotary_emb" not in name}
    for name, value in exported.items():
        assert torch.equal(value, expected[name]), name
    assert any(value.abs().sum() > 0 for value in exported.values())


def test_save_pretrained_writes_the_state_dict_in_the_draft_dtype(target, tmp_path):
    online = make_online(target, dtype=torch.bfloat16)

    online.save_pretrained(tmp_path, state_dict=consolidated_state(online))

    floating = [value for value in load_file(tmp_path / "model.safetensors").values() if value.is_floating_point()]
    assert floating and all(value.dtype == torch.bfloat16 for value in floating)


def test_save_pretrained_rejects_a_state_dict_that_lacks_draft_entries(target, tmp_path):
    online = make_online(target)
    truncated = {
        name: value for name, value in consolidated_state(online).items() if not name.startswith("online_model")
    }

    with pytest.raises(ValueError, match="lacks draft entries"):
        online.save_pretrained(tmp_path, state_dict=truncated)


def test_init_from_rejects_a_checkpoint_that_does_not_match_the_draft(target, tmp_path):
    wider = copy.deepcopy(DRAFT_CONFIG)
    wider["intermediate_size"] = 128
    other = build_online_model(target, wider, dtype=torch.float32, attention_backend="sdpa", num_anchors=4)
    other.save_pretrained(tmp_path)

    with pytest.raises(Exception):
        make_online(target, init_from=tmp_path)


def test_build_rejects_a_draft_vocabulary_that_differs_from_the_target(target):
    config = copy.deepcopy(DRAFT_CONFIG)
    config["vocab_size"] = VOCAB + 8

    with pytest.raises(ValueError, match="does not match the target LM head"):
        build_online_model(target, config, dtype=torch.float32, attention_backend="sdpa")


def test_build_rejects_a_draft_width_that_differs_from_the_target(target):
    config = copy.deepcopy(DRAFT_CONFIG)
    config["hidden_size"] = HIDDEN * 2

    with pytest.raises(ValueError, match="hidden_size=64 does not match the target's 32"):
        build_online_model(target, config, dtype=torch.float32, attention_backend="sdpa")


def test_build_rejects_a_draft_trained_for_a_different_target_depth(target):
    config = copy.deepcopy(DRAFT_CONFIG)
    config["num_target_layers"] = 6

    with pytest.raises(ValueError, match="num_target_layers=6 does not match the target's 4 layers"):
        build_online_model(target, config, dtype=torch.float32, attention_backend="sdpa")


def test_build_rejects_a_captured_layer_the_target_does_not_have_below_its_last(target):
    # Without ``num_target_layers`` the draft config cannot catch this itself; the target can.
    config = copy.deepcopy(DRAFT_CONFIG)
    del config["num_target_layers"]
    config["dflash_config"]["target_layer_ids"] = [1, 3]

    with pytest.raises(ValueError, match="must stay below the target's last layer"):
        build_online_model(target, config, dtype=torch.float32, attention_backend="sdpa")


@pytest.fixture
def target_dir(target, tmp_path):
    path = tmp_path / "target"
    target.save_pretrained(path)
    return path


def loader_spec(target_dir, **overrides):
    options = dict(draft_config=copy.deepcopy(DRAFT_CONFIG), attention_backend="sdpa", num_anchors=4)
    options.update(overrides.pop("loader_options", {}))
    return ModelSpec(
        model_path_or_name=str(target_dir),
        dtype="float32",
        attn_implementation="eager",
        loader="dflash2",
        loader_options=options,
        **overrides,
    )


def test_loader_builds_a_frozen_target_with_a_trainable_draft(target_dir):
    model = build_model(loader_spec(target_dir)).model

    assert isinstance(model, DFlash2OnlineModel)
    assert not any(p.requires_grad for p in model.target.parameters())
    assert all(p.requires_grad for p in model.draft.parameters())
    model.train()
    assert not model.target.training
    assert model.draft.training


def test_loader_trains_a_packed_window_through_the_loaded_model(target_dir):
    model = build_model(loader_spec(target_dir)).model.train()

    out = model(**packed_window(PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_THREE_TOKEN_RESPONSE))

    assert torch.isfinite(out["loss"])


def test_loader_rejects_an_option_it_does_not_define(target_dir):
    with pytest.raises((ValidationError, ValueError), match="checkpoint_draft"):
        build_model(loader_spec(target_dir, loader_options={"checkpoint_draft": True}))


def test_loader_rejects_a_draft_config_without_required_settings(target_dir):
    broken = copy.deepcopy(DRAFT_CONFIG)
    del broken["dflash_config"]["selector_rank"]

    with pytest.raises((ValidationError, ValueError), match="selector_rank"):
        build_model(loader_spec(target_dir, loader_options={"draft_config": broken}))


def test_loader_rejects_a_draft_that_reads_the_targets_last_layer(target_dir):
    last_layer = copy.deepcopy(DRAFT_CONFIG)
    last_layer["dflash_config"]["target_layer_ids"] = [1, 3]

    with pytest.raises((ValidationError, ValueError), match="last layer"):
        build_model(loader_spec(target_dir, loader_options={"draft_config": last_layer}))


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), -1.0])
def test_loader_rejects_a_selector_weight_that_is_not_finite_and_non_negative(target_dir, alpha):
    with pytest.raises((ValidationError, ValueError), match="selector_loss_alpha"):
        build_model(loader_spec(target_dir, loader_options={"selector_loss_alpha": alpha}))


@pytest.mark.parametrize("patch", ["gradient_checkpointing", "liger"])
def test_loader_rejects_generic_model_patches(target_dir, patch):
    with pytest.raises((ValidationError, ValueError), match="generic model patches"):
        build_model(loader_spec(target_dir, patches=Patches(**{patch: True})))


def train_step(target_dir, **loader_options):
    """``(loss, draft grad norm, terms)`` of one packed-window forward and backward of the loader's model."""
    model = build_model(loader_spec(target_dir, loader_options=loader_options)).model.train()

    out = model(**packed_window(PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_THREE_TOKEN_RESPONSE))
    out["loss"].backward()

    grad_norm = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.draft.parameters() if p.grad is not None))
    return out["loss"].detach(), grad_norm, out["dflash2_terms"]


@pytest.mark.parametrize("lk_loss_type", [None, "alpha", "lambda", "tv"])
def test_loader_trains_under_each_lk_objective(target_dir, lk_loss_type):
    loss, grad_norm, terms = train_step(target_dir, lk_loss_type=lk_loss_type)

    assert torch.isfinite(loss)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    # The objective that reached SpecForge names its loss term: cross entropy by default, else an LK objective.
    assert ("lk_loss" in terms) == (lk_loss_type is not None)
    assert ("ce_loss" in terms) == (lk_loss_type is None)
    assert ("objective/lk_kl_weight" in terms) == (lk_loss_type == "lambda")


def test_lk_alpha_is_the_cross_entropy_objective(target_dir):
    cross_entropy_loss, cross_entropy_grad_norm, _ = train_step(target_dir)

    alpha_loss, alpha_grad_norm, _ = train_step(target_dir, lk_loss_type="alpha")

    torch_assert_equal(alpha_loss, cross_entropy_loss)
    torch_assert_equal(alpha_grad_norm, cross_entropy_grad_norm)


def test_lk_lambda_weights_cross_entropy_by_kl_scale_and_kl_decay(target_dir):
    """The cross entropy weight is ``kl_scale * exp(-kl_decay * acceptance)``, so 0 leaves TV and 1 leaves CE."""
    cross_entropy_loss, cross_entropy_grad_norm, _ = train_step(target_dir)
    tv_loss, tv_grad_norm, _ = train_step(target_dir, lk_loss_type="tv")
    assert tv_loss != cross_entropy_loss  # the two ends differ, so the checks below tell them apart

    no_weight_loss, no_weight_grad_norm, _ = train_step(target_dir, lk_loss_type="lambda", kl_scale=0.0)
    full_weight_loss, full_weight_grad_norm, _ = train_step(target_dir, lk_loss_type="lambda", kl_decay=0.0)

    torch_assert_equal(no_weight_loss, tv_loss)
    torch_assert_equal(no_weight_grad_norm, tv_grad_norm)
    torch_assert_equal(full_weight_loss, cross_entropy_loss)
    torch_assert_equal(full_weight_grad_norm, cross_entropy_grad_norm)


def test_loader_rejects_an_unknown_lk_loss_type(target_dir):
    with pytest.raises((ValidationError, ValueError), match="lk_loss_type"):
        build_model(loader_spec(target_dir, loader_options={"lk_loss_type": "kl"}))


@pytest.mark.parametrize("option", ["kl_scale", "kl_decay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_loader_rejects_an_lk_weight_that_is_not_finite(target_dir, option, value):
    with pytest.raises((ValidationError, ValueError), match=option):
        build_model(loader_spec(target_dir, loader_options={option: value}))


@pytest.fixture
def reductions(monkeypatch):
    """``(chunk_size, number of anchor blocks)`` of every chunked reduction SpecForge runs, in order."""
    calls = []
    reduce_in_chunks = dflash_family_model.checkpointed_chunk_reduce

    def record(function, *tensors, chunk_size, dim=0, **kwargs):
        blocks = next(tensor for tensor in tensors if tensor is not None).shape[dim]
        calls.append((chunk_size, blocks))
        return reduce_in_chunks(function, *tensors, chunk_size=chunk_size, dim=dim, **kwargs)

    monkeypatch.setattr(dflash_family_model, "checkpointed_chunk_reduce", record)
    return calls


@pytest.mark.parametrize(
    "chunk_blocks, multi_chunk, atol",
    [
        (128, False, 0.0),  # every block in one chunk: the same sums as the unchunked path
        # One block per chunk: a real multi-chunk reduction. It sums in another order, so fp32 rounding differs. An
        # ulp of the largest term (about 16) is 2e-6, while a dropped or doubled chunk moves the loss by O(1).
        (1, True, 1e-5),
    ],
)
def test_loader_trains_an_unchunked_objective_like_a_chunked_one(
    target_dir, reductions, chunk_blocks, multi_chunk, atol
):
    """``objective_chunk_blocks=0`` projects every anchor block at once, and must agree with a chunked objective."""
    loss, grad_norm, terms = train_step(target_dir, teacher_metrics=True, objective_chunk_blocks=0)
    assert teacher_terms(terms)  # the teacher terms are part of what is compared
    assert {chunk_size for chunk_size, _ in reductions} == {0}
    reductions.clear()

    chunked_loss, chunked_grad_norm, chunked_terms = train_step(
        target_dir, teacher_metrics=True, objective_chunk_blocks=chunk_blocks
    )

    assert {chunk_size for chunk_size, _ in reductions} == {chunk_blocks}
    # More blocks than the chunk size means more than one chunk; the window must keep this case multi-chunk.
    assert all((blocks > chunk_blocks) == multi_chunk for _, blocks in reductions)
    torch_assert_close(chunked_loss, loss, rtol=0.0, atol=atol)
    torch_assert_close(chunked_grad_norm, grad_norm, rtol=0.0, atol=atol)
    assert chunked_terms.keys() == terms.keys()
    for name, (numerator, denominator) in terms.items():
        torch_assert_close(chunked_terms[name][0], numerator, rtol=0.0, atol=atol, msg=name)
        torch_assert_close(chunked_terms[name][1], denominator, rtol=0.0, atol=atol, msg=name)


def test_loader_rejects_a_negative_objective_chunk(target_dir):
    with pytest.raises((ValidationError, ValueError), match="objective_chunk_blocks"):
        build_model(loader_spec(target_dir, loader_options={"objective_chunk_blocks": -1}))


@pytest.fixture
def lora_fine_tune(target, tmp_path):
    """``(adapter dir, the fine-tune as base + unmerged adapter)`` of a LoRA fine-tune of ``target``."""
    from peft import LoraConfig
    from peft import get_peft_model

    torch.manual_seed(1)
    # init_lora_weights=False draws both factors at random, so the adapter changes the target's outputs.
    tuned = get_peft_model(
        copy.deepcopy(target),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False),
    ).eval()
    path = tmp_path / "adapter"
    tuned.save_pretrained(path)
    return path, tuned


def test_loader_trains_a_packed_window_through_a_lora_fine_tune(target_dir, lora_fine_tune):
    adapter_dir, tuned = lora_fine_tune
    shutil.copytree(adapter_dir, target_dir / PEFT_ADAPTER_DIRNAME)

    model = build_model(loader_spec(target_dir)).model.train()

    out = model(**packed_window(PROMPT_THEN_TWO_TOKEN_RESPONSE, PROMPT_THEN_THREE_TOKEN_RESPONSE))
    assert torch.isfinite(out["loss"])
    # The draft reads the fine-tune (base plus merged adapter), not the stock target.
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    with torch.no_grad():
        hidden_states = tuned(input_ids=input_ids, output_hidden_states=True).hidden_states
    expected = torch.cat([hidden_states[i + 1] for i in model.draft.target_layer_ids], dim=-1)
    torch.testing.assert_close(model.target_context(input_ids), expected)


def test_loader_logs_the_adapter_pack_it_merges(target_dir, lora_fine_tune, caplog):
    adapter_dir, _ = lora_fine_tune
    shutil.copytree(adapter_dir, target_dir / PEFT_ADAPTER_DIRNAME)

    with caplog.at_level(logging.INFO, logger=dflash2_loader.__name__):
        build_model(loader_spec(target_dir))

    assert str(target_dir / PEFT_ADAPTER_DIRNAME) in caplog.text


@pytest.mark.parametrize("parallelism", [dict(sequence_parallel=2), dict(expert_parallel=2)])
def test_loader_supports_data_parallelism_only(target_dir, parallelism):
    with pytest.raises((ValidationError, ValueError), match="data parallelism only"):
        build_model(loader_spec(target_dir, parallelism=ParallelismConfig(**parallelism)))
