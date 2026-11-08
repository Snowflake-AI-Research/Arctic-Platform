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
"""``run_pipeline`` gives Qwen3.8 (``qwen4_exp``) its images and video, packed and unpacked.

The reference is a direct model forward on each row alone, with Hugging Face computing that row's M-RoPE
positions from ``mm_token_type_ids``. The pipeline must reproduce its per-token logprobs:

* unpacked (``pack=False``): the row is unpadded and its media and token types reach the model;
* packed: each microbatch takes only the images and videos its rows own (``image_row_ids`` /
  ``video_row_ids``), and positions are the packed ``[4, 1, T]`` M-RoPE ids.

CPU only, on the tiny random-weight config of ``tests/model/test_qwen4_exp.py`` with the MoE swapped for a dense
projection, as ``tests/model/test_qwen4_exp_multimodal.py`` builds it. This tests what the pipeline hands the model,
not GPU kernels. Hugging Face's CPU reference GatedDeltaNet functions (``causal_conv1d_fn``,
``torch_chunk_gated_delta_rule``) ignore row boundaries, where the fla / causal-conv1d varlen kernels take them; the
``varlen_linear_attention`` fixture runs them one packed row at a time, with the rows read off ``position_ids``
channel 0 as ``cu_seqlens_from_position_ids`` does. Tests that put several rows in one call use it.
"""

from __future__ import annotations

import pytest
import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as hf_qwen4_exp

from arctic_platform.common.packing import cu_seqlens_from_position_ids
from arctic_platform.common.packing import mrope_position_ids
from arctic_platform.rl.processors.pipeline import run_pipeline
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_qwen4_exp_multimodal import IMAGE_TOKEN_ID
from tests.model.test_qwen4_exp_multimodal import _image_tokens
from tests.model.test_qwen4_exp_multimodal import _model

VIDEO_TOKEN_ID = 30
PAD_TOKEN_ID = 0
IMAGE_GRIDS = torch.tensor([[1, 2, 2], [1, 2, 4]])
# Two frames of 2x2 patches: 8 placeholder tokens at spatial merge 1.
VIDEO_GRIDS = torch.tensor([[2, 2, 2]])
PROCESSING = {"post": ["compute_logprobs"], "loss_fn": None}


@pytest.fixture(scope="module")
def model():
    return _model().eval()


@pytest.fixture
def varlen_linear_attention(model, monkeypatch):
    """Run the CPU GatedDeltaNet reference functions per packed row, as the varlen GPU kernels do."""
    boundaries = {}

    def read_rows(_module, _args, kwargs):
        position_ids = kwargs.get("position_ids")
        boundaries["cu"] = None if position_ids is None else cu_seqlens_from_position_ids(position_ids).tolist()

    def rows(length):
        cu = boundaries.get("cu") or [0, length]
        return list(zip(cu[:-1], cu[1:])) + ([(cu[-1], length)] if cu[-1] < length else [])

    conv, chunk_rule = hf_qwen4_exp.causal_conv1d_fn, hf_qwen4_exp.torch_chunk_gated_delta_rule

    def varlen_conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        spans = rows(hidden_states.shape[-1])
        if len(spans) == 1:
            return conv(hidden_states, weight, bias, activation=activation, **kwargs)
        return torch.cat([conv(hidden_states[..., s:e], weight, bias, activation=activation) for s, e in spans], -1)

    def varlen_chunk_rule(query, key, value, g, beta, **kwargs):
        spans = rows(query.shape[1])
        if len(spans) == 1:
            return chunk_rule(query, key, value, g=g, beta=beta, **kwargs)
        kwargs.pop("cu_seqlens", None)
        assert not kwargs.get("output_final_state"), "a packed forward keeps no cache"
        pieces = [
            chunk_rule(query[:, s:e], key[:, s:e], value[:, s:e], g=g[:, s:e], beta=beta[:, s:e], **kwargs)[0]
            for s, e in spans
        ]
        return torch.cat(pieces, dim=1), None

    monkeypatch.setattr(hf_qwen4_exp, "causal_conv1d_fn", varlen_conv)
    monkeypatch.setattr(hf_qwen4_exp, "torch_chunk_gated_delta_rule", varlen_chunk_rule)
    handle = model.register_forward_pre_hook(read_rows, with_kwargs=True)
    yield
    handle.remove()


def _patches(model, grids: torch.Tensor) -> torch.Tensor:
    vision = model.config.vision_config
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return torch.randn(int(grids.prod(-1).sum()), patch_dim)


def _video_tokens(grid_thw: torch.Tensor) -> list[int]:
    """The processor's video layout: every frame wrapped in vision start/end after its timestamp text."""
    frame_tokens = int(grid_thw[1:].prod())
    tokens = []
    for frame in range(int(grid_thw[0])):
        tokens += [20 + frame, 27] + [VIDEO_TOKEN_ID] * frame_tokens + [28]
    return tokens


def _token_types(input_ids: torch.Tensor) -> torch.Tensor:
    return (input_ids == IMAGE_TOKEN_ID).long() + 2 * (input_ids == VIDEO_TOKEN_ID).long()


def _rows():
    """Image row, video row, text row, image row: one item each, so every row owns different media."""
    set_seed(1)
    return [
        dict(tokens=[2, 3] + _image_tokens(IMAGE_GRIDS[0]) + [4, 5, 6], image_grid=IMAGE_GRIDS[:1]),
        dict(tokens=[7] + _video_tokens(VIDEO_GRIDS[0]) + [8, 9], video_grid=VIDEO_GRIDS),
        # Long enough that no two rows fit the one-row-per-call budget below.
        dict(tokens=list(range(10, 20)) + [22, 23, 24]),
        dict(tokens=[15] + _image_tokens(IMAGE_GRIDS[1]) + [16, 17, 18, 19], image_grid=IMAGE_GRIDS[1:]),
    ]


def _with_pixels(model, rows):
    for row in rows:
        if "image_grid" in row:
            row["pixel_values"] = _patches(model, row["image_grid"])
        if "video_grid" in row:
            row["pixel_values_videos"] = _patches(model, row["video_grid"])
    return rows


def _padded_batch(rows) -> dict:
    """Right-padded ``[B, S]`` batch with each media kind concatenated in row order and its row ids."""
    seqlen = max(len(row["tokens"]) for row in rows)
    input_ids = torch.full((len(rows), seqlen), PAD_TOKEN_ID)
    attention_mask = torch.zeros(len(rows), seqlen, dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, : len(row["tokens"])] = torch.tensor(row["tokens"])
        attention_mask[index, : len(row["tokens"])] = 1
    batch = dict(input_ids=input_ids, attention_mask=attention_mask, mm_token_type_ids=_token_types(input_ids))
    for values_key, grid_key, row_ids_key, row_key in (
        ("pixel_values", "image_grid_thw", "image_row_ids", "image_grid"),
        ("pixel_values_videos", "video_grid_thw", "video_row_ids", "video_grid"),
    ):
        owners = [index for index, row in enumerate(rows) if row_key in row]
        if owners:
            batch[values_key] = torch.cat([rows[index][values_key] for index in owners])
            batch[grid_key] = torch.cat([rows[index][row_key] for index in owners])
            batch[row_ids_key] = torch.cat([torch.full((rows[index][row_key].shape[0],), index) for index in owners])
    return batch


def _direct_logprobs(model, row) -> torch.Tensor:
    """Next-token logprobs of one row forwarded alone; Hugging Face builds its positions."""
    input_ids = torch.tensor([row["tokens"]])
    kwargs = dict(input_ids=input_ids, mm_token_type_ids=_token_types(input_ids))
    if "image_grid" in row:
        kwargs.update(pixel_values=row["pixel_values"], image_grid_thw=row["image_grid"])
    if "video_grid" in row:
        kwargs.update(pixel_values_videos=row["pixel_values_videos"], video_grid_thw=row["video_grid"])
    with torch.no_grad():
        logits = model(**kwargs).logits.float()
    return logits[0, :-1].log_softmax(-1).gather(-1, input_ids[0, 1:, None]).squeeze(-1)


def _record_model_calls(model):
    calls = []

    def hook(_module, _args, kwargs):
        calls.append(dict(kwargs))

    return calls, model.register_forward_pre_hook(hook, with_kwargs=True)


def _run(model, batch, *, pack: bool, max_tokens_per_mb: int = 10240) -> torch.Tensor:
    result = run_pipeline(
        model,
        (),
        batch,
        {"pad_token_id": PAD_TOKEN_ID},
        PROCESSING,
        "cpu",
        backward=False,
        pack=pack,
        max_tokens_per_mb=max_tokens_per_mb,
        return_tensors=True,
    )
    # The packed path returns lists (``detensorize``) whatever ``return_tensors`` says.
    return torch.as_tensor(result["batch"]["logprobs"])


def _row_logprobs(logprobs: torch.Tensor, index: int, length: int) -> torch.Tensor:
    """Row ``index``'s first ``length - 1`` logprobs from a padded ``[B, S]`` or unpadded ``[1, N]`` output."""
    return logprobs.reshape(logprobs.shape[0], -1)[index, : length - 1].float()


@pytest.mark.parametrize("row_index", [0, 1, 3], ids=["image", "video", "second-image"])
def test_unpacked_media_row_logprobs_match_a_direct_forward(model, row_index):
    rows = _with_pixels(model, _rows())
    row = rows[row_index]
    batch = _padded_batch([row])
    # One padded column, so the row really is unpadded on the way in.
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        batch[key] = torch.nn.functional.pad(batch[key], (0, 1))
    calls, handle = _record_model_calls(model)
    try:
        logprobs = _run(model, batch, pack=False)
    finally:
        handle.remove()

    torch_assert_close(_row_logprobs(logprobs, 0, len(row["tokens"])), _direct_logprobs(model, row), rtol=0, atol=1e-5)
    (call,) = calls
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in batch:
            torch_assert_equal(call[key], batch[key])
        else:
            assert key not in call
    assert "image_row_ids" not in call and "video_row_ids" not in call
    assert call["position_ids"].shape == (4, 1, len(row["tokens"]))


def test_packed_rows_one_per_call_match_a_direct_forward(model):
    rows = _with_pixels(model, _rows())
    batch = _padded_batch(rows)
    calls, handle = _record_model_calls(model)
    try:
        # Budget for one row per microbatch: every row is its own packed call.
        logprobs = _run(model, batch, pack=True, max_tokens_per_mb=max(len(row["tokens"]) for row in rows))
    finally:
        handle.remove()

    assert len(calls) == len(rows)
    for index, row in enumerate(rows):
        torch_assert_close(
            _row_logprobs(logprobs, index, len(row["tokens"])), _direct_logprobs(model, row), rtol=0, atol=1e-5
        )
    # The text-only row's call runs no vision forward.
    text_calls = [call for call in calls if call["input_ids"].shape[-1] == len(rows[2]["tokens"])]
    assert len(text_calls) == 1
    assert not {"pixel_values", "pixel_values_videos"} & set(text_calls[0])


def test_packed_call_gets_its_rows_media_and_packed_mrope_positions(model):
    rows = _with_pixels(model, _rows())
    batch = _padded_batch(rows)
    calls, handle = _record_model_calls(model)
    try:
        _run(model, batch, pack=True, max_tokens_per_mb=10240)
    finally:
        handle.remove()

    (call,) = calls
    # The splitter reorders rows; read the packed order back off the call (every row's tokens differ).
    packed_ids = call["input_ids"][0].tolist()
    order = []
    while sum(len(rows[index]["tokens"]) for index in order) < len(packed_ids):
        offset = sum(len(rows[index]["tokens"]) for index in order)
        (index,) = [i for i, row in enumerate(rows) if packed_ids[offset:][: len(row["tokens"])] == row["tokens"]]
        order.append(index)
    assert sorted(order) == list(range(len(rows)))
    tokens = [token for index in order for token in rows[index]["tokens"]]
    torch_assert_equal(call["input_ids"], torch.tensor([tokens]))
    image_owners = [index for index in order if "image_grid" in rows[index]]
    torch_assert_equal(call["pixel_values"], torch.cat([rows[index]["pixel_values"] for index in image_owners]))
    torch_assert_equal(call["image_grid_thw"], torch.cat([rows[index]["image_grid"] for index in image_owners]))
    torch_assert_equal(call["pixel_values_videos"], rows[1]["pixel_values_videos"])
    torch_assert_equal(call["video_grid_thw"], VIDEO_GRIDS)
    torch_assert_equal(call["mm_token_type_ids"], _token_types(torch.tensor([tokens])))

    lengths = torch.tensor([len(rows[index]["tokens"]) for index in order])
    cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
    expected = mrope_position_ids(
        _token_types(torch.tensor([tokens])),
        cu_seqlens,
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        image_grid_thw=call["image_grid_thw"],
        video_grid_thw=VIDEO_GRIDS,
    )
    torch_assert_equal(call["position_ids"], expected)
    # Each row's rotary positions are the ones Hugging Face gives that row alone.
    for start, end, index in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), order):
        torch_assert_equal(call["position_ids"][0, 0, start:end], torch.arange(end - start))
        row_alone = mrope_position_ids(
            _token_types(torch.tensor([rows[index]["tokens"]])),
            torch.tensor([0, end - start]),
            spatial_merge_size=1,
            image_grid_thw=rows[index].get("image_grid"),
            video_grid_thw=rows[index].get("video_grid"),
        )
        torch_assert_equal(call["position_ids"][..., start:end], row_alone)


def test_text_only_batch_forwards_as_before(model):
    set_seed(2)
    rows = [dict(tokens=[2, 3, 4, 5, 6, 7]), dict(tokens=[8, 9, 10, 11])]
    batch = _padded_batch(rows)
    del batch["mm_token_type_ids"]
    calls, handle = _record_model_calls(model)
    try:
        logprobs = _run(model, batch, pack=True, max_tokens_per_mb=6)
    finally:
        handle.remove()

    for index, row in enumerate(rows):
        torch_assert_close(
            _row_logprobs(logprobs, index, len(row["tokens"])), _direct_logprobs(model, row), rtol=0, atol=1e-5
        )
    for call in calls:
        assert set(call) == {"input_ids", "position_ids", "use_cache"}
        torch_assert_equal(call["position_ids"], torch.arange(call["input_ids"].shape[-1])[None])


def test_media_of_a_many_row_batch_needs_row_ids(model):
    rows = _with_pixels(model, _rows())
    batch = _padded_batch(rows)
    del batch["image_row_ids"]

    with pytest.raises(ValueError, match="needs image_row_ids"):
        _run(model, batch, pack=True)


def test_image_tokens_in_another_row_are_refused(model):
    rows = _with_pixels(model, _rows())
    batch = _padded_batch(rows)
    # Claim the first image for the text row.
    batch["image_row_ids"] = torch.tensor([2, 3])

    with pytest.raises(ValueError, match="rows hold"):
        _run(model, batch, pack=False)


def test_packed_rows_sharing_a_call_match_a_direct_forward(model, varlen_linear_attention):
    rows = _with_pixels(model, _rows())
    batch = _padded_batch(rows)
    calls, handle = _record_model_calls(model)
    try:
        logprobs = _run(model, batch, pack=True, max_tokens_per_mb=10240)
    finally:
        handle.remove()

    assert len(calls) == 1
    for index, row in enumerate(rows):
        torch_assert_close(
            _row_logprobs(logprobs, index, len(row["tokens"])), _direct_logprobs(model, row), rtol=0, atol=1e-5
        )


def test_unpacked_media_batch_rows_match_a_direct_forward(model, varlen_linear_attention):
    rows = _with_pixels(model, _rows())
    batch = _padded_batch(rows)
    logprobs = _run(model, batch, pack=False)

    # Every row runs in one unpadded call; the output comes back padded to [B, S].
    assert logprobs.shape == batch["input_ids"].shape
    for index, row in enumerate(rows):
        torch_assert_close(
            _row_logprobs(logprobs, index, len(row["tokens"])), _direct_logprobs(model, row), rtol=0, atol=1e-5
        )


def test_one_row_batch_may_omit_row_ids(model):
    rows = _with_pixels(model, _rows())
    batch = _padded_batch([rows[3]])
    del batch["image_row_ids"]
    logprobs = _run(model, batch, pack=True)

    expected = _direct_logprobs(model, rows[3])
    torch_assert_close(_row_logprobs(logprobs, 0, len(rows[3]["tokens"])), expected, rtol=0, atol=1e-5)


def test_media_only_on_meta_does_not_reach_the_model(model):
    """Meta is replicated across ranks, so a model input found only there must not be forwarded."""
    set_seed(2)
    rows = _with_pixels(model, _rows())
    batch = _padded_batch([dict(tokens=[2, 3, 4, 5, 6, 7])])
    meta_media = {key: value for key, value in _padded_batch([rows[0]]).items() if key.startswith(("pixel", "image"))}
    calls, handle = _record_model_calls(model)
    try:
        run_pipeline(
            model,
            (),
            batch,
            {"pad_token_id": PAD_TOKEN_ID, **meta_media},
            PROCESSING,
            "cpu",
            backward=False,
            pack=True,
            return_tensors=True,
        )
    finally:
        handle.remove()

    (call,) = calls
    assert not {"pixel_values", "image_grid_thw", "image_row_ids"} & set(call)
