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

"""Per-sequence image inputs through ``run_pipeline``: microbatching, packing and the model forward."""

from types import SimpleNamespace

import pytest
import torch

import arctic_platform.rl.processors  # noqa: F401  # populate registries
from arctic_platform.rl.processors.multimodal import collate_multi_modal_inputs
from arctic_platform.rl.processors.pipeline import fast_logprobs_and_entropy_from_logits
from arctic_platform.rl.processors.pipeline import run_pipeline
from tests.model.test_qwen35_moe_vlm_images import IMAGE
from tests.model.test_qwen35_moe_vlm_images import MERGE
from tests.model.test_qwen35_moe_vlm_images import SEQUENCES
from tests.model.test_qwen35_moe_vlm_images import _images
from tests.model.test_qwen35_moe_vlm_images import _sequence
from tests.model.test_qwen35_moe_vlm_images import _tiny_vlm
from tests.model.test_qwen35_moe_vlm_images import _without_optional_gated_delta_net_kernels  # noqa: F401


def _batch(sequences, marker: int | None = None):
    """A right-padded batch whose rows carry their own images; ``marker`` prepends a distinct first token per row."""
    rows = [([marker + row] if marker is not None else []) + ids for row, (ids, _) in enumerate(sequences)]
    width = max(len(ids) for ids in rows)
    input_ids = torch.zeros(len(rows), width, dtype=torch.long)
    attention_mask = torch.zeros(len(rows), width, dtype=torch.long)
    for row, ids in enumerate(rows):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention_mask[row, : len(ids)] = 1
    images = [
        {"pixel_values": _images(grids, seed=row), "image_grid_thw": torch.tensor(grids)} if grids else None
        for row, (_, grids) in enumerate(sequences)
    ]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "multi_modal_inputs": images}, rows


class _RecordingEngine:
    global_rank = 0

    def __init__(self):
        self.calls = []

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        ids = ids[None] if ids.ndim == 1 else ids
        return SimpleNamespace(logits=torch.zeros(*ids.shape, 64))


def test_collate_concatenates_in_sequence_order_and_skips_text_rows():
    first = {"pixel_values": torch.ones(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    second = {"pixel_values": torch.zeros(8, 3), "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]])}
    out = collate_multi_modal_inputs([first, None, second])
    assert torch.equal(out["pixel_values"], torch.cat([first["pixel_values"], second["pixel_values"]]))
    assert out["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
    assert collate_multi_modal_inputs([None, None]) == {}


def test_collate_rejects_inconsistent_or_unsupported_entries():
    with pytest.raises(ValueError, match="do not hold the 4 patches"):
        collate_multi_modal_inputs([{"pixel_values": torch.ones(3, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}])
    with pytest.raises(ValueError, match="unsupported keys"):
        collate_multi_modal_inputs([{"pixel_values_videos": torch.ones(4, 3), "video_grid_thw": [[1, 2, 2]]}])
    with pytest.raises(ValueError, match="missing"):
        collate_multi_modal_inputs([{"pixel_values": torch.ones(4, 3)}])


@pytest.mark.parametrize("max_tokens_per_mb", [24, 40, 1000])
def test_each_microbatch_gets_exactly_its_sequences_images_in_packed_order(max_tokens_per_mb):
    sequences = SEQUENCES + [_sequence(1, (1, 2, 4), 2)]
    batch, rows = _batch(sequences, marker=40)
    engine = _RecordingEngine()
    run_pipeline(
        engine,
        (),
        batch,
        {},
        {"loss_fn": None, "post": []},
        "cpu",
        backward=False,
        max_tokens_per_mb=max_tokens_per_mb,
    )

    seen = []
    for call in engine.calls:
        ids, positions = call["input_ids"][0], call["position_ids"][0]
        starts = (positions == 0).nonzero().flatten().tolist()
        packed_rows = [int(ids[start]) - 40 for start in starts]
        seen += packed_rows
        expected = [batch["multi_modal_inputs"][row] for row in packed_rows]
        expected = collate_multi_modal_inputs(expected)
        if not expected:
            assert "pixel_values" not in call and "image_grid_thw" not in call
            continue
        assert torch.equal(call["pixel_values"], expected["pixel_values"])
        assert torch.equal(call["image_grid_thw"], expected["image_grid_thw"])
        image_tokens = int((ids == IMAGE).sum())
        assert image_tokens == int(expected["image_grid_thw"].prod(-1).sum()) // MERGE**2
    assert sorted(seen) == list(range(len(rows)))
    if max_tokens_per_mb == 24:
        assert len(engine.calls) > 2, "the token budget must split the batch for this test to mean anything"


def test_unpacked_forward_collates_the_whole_batch():
    batch, _ = _batch(SEQUENCES)
    engine = _RecordingEngine()
    run_pipeline(
        engine, (), batch, {"pad_token_id": 0}, {"loss_fn": None, "post": []}, "cpu", backward=False, pack=False
    )
    expected = collate_multi_modal_inputs(batch["multi_modal_inputs"])
    assert torch.equal(engine.calls[0]["pixel_values"], expected["pixel_values"])
    assert torch.equal(engine.calls[0]["image_grid_thw"], expected["image_grid_thw"])


def test_batch_level_pixel_values_are_refused_when_microbatching():
    batch, _ = _batch(SEQUENCES)
    images = collate_multi_modal_inputs(batch.pop("multi_modal_inputs"))
    with pytest.raises(ValueError, match="cannot split batch-level"):
        run_pipeline(
            _RecordingEngine(), (), {**batch, **images}, {}, {"loss_fn": None, "post": []}, "cpu", backward=False
        )


class _ModelEngine:
    global_rank = 0

    def __init__(self, model):
        self.model = model

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward(self, loss, scale_wrt_gas=True):
        loss.backward()


def _reference_logprobs(model, ids, grids, seed):
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            position_ids=torch.arange(len(ids))[None],
            pixel_values=_images(grids, seed=seed) if grids else None,
            image_grid_thw=torch.tensor(grids) if grids else None,
        )["logits"]
    logprobs, _ = fast_logprobs_and_entropy_from_logits(logits, torch.roll(input_ids, shifts=-1, dims=-1), False)
    return logprobs[0]


def test_microbatched_model_logprobs_match_each_sequence_run_alone():
    """With a budget no two sequences fit in, each packed microbatch is one sequence, which SDPA can run unmasked."""
    sequences = [SEQUENCES[0], SEQUENCES[1], SEQUENCES[3], _sequence(10)]
    budget = max(len(ids) for ids, _ in sequences)
    assert min(len(a) + len(b) for a, _ in sequences for b, _ in sequences if a is not b) > budget
    model = _tiny_vlm()
    batch, rows = _batch(sequences)

    out = run_pipeline(
        _ModelEngine(model),
        (),
        batch,
        {},
        {"loss_fn": None, "post": ["ap_compute_logprobs"]},
        "cpu",
        backward=False,
        max_tokens_per_mb=budget,
    )

    logprobs = torch.tensor(out["batch"]["logprobs"])
    for row, (ids, grids) in enumerate(sequences):
        expected = _reference_logprobs(model, ids, grids, seed=row)
        torch.testing.assert_close(logprobs[row, : len(ids)], expected, rtol=1e-5, atol=1e-5)
