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
"""Qwen3.8 context parallelism on media calls: windowed (the default) and sharded after the vision merge.

Windowed, what the trainers dispatch today: each rank gets ``token_shard``'s window of the packed call with the
whole media items in it and their ``*_window_token_ranges``. The call is already sharded and passes through the
CP wrapper unchanged, as does any call not marked whole, text-only calls included.

After the vision merge (prime-rl #3916), opt-in: the caller passes ``cp_shard_after_vision_merge=True`` and the
whole call, images and video included, to every rank. Each rank runs the vision tower and the merge, then hands
the language model only its ``token_window`` of the merged ``inputs_embeds``, the raw ``ple_input_ids``, the
positions and ``routed_experts``.

CPU only, on the tiny config and dense-MLP model of ``test_qwen4_exp_multimodal.py``. The CP group is faked the
way that file's spec does: ``dist.get_world_size`` / ``dist.get_rank`` are patched, and the GatedDeltaNet adapter
is stubbed because it imports fla, a CUDA package. The backbone's collectives (position gather, PLE and QSA
context exchange) and the fla kernels need a real multi-rank group, so these tests stop at the language model's
inputs, or replace the language model with an identity to run a backward through the vision tower.
"""

from __future__ import annotations

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from arctic_platform.common.packing import mrope_position_ids
from arctic_platform.common.packing import pack_microbatch
from arctic_platform.common.packing import pad_packed_microbatch
from arctic_platform.common.packing import token_shard
from arctic_platform.common.packing import token_window
from arctic_platform.model import get_language_model
from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_qwen4_exp_multimodal import IMAGE_GRID_THW
from tests.model.test_qwen4_exp_multimodal import IMAGE_TOKEN_ID
from tests.model.test_qwen4_exp_multimodal import TEXT_ONLY_INPUT_IDS
from tests.model.test_qwen4_exp_multimodal import VISION_END_TOKEN_ID
from tests.model.test_qwen4_exp_multimodal import VISION_START_TOKEN_ID
from tests.model.test_qwen4_exp_multimodal import _count_vision_forwards
from tests.model.test_qwen4_exp_multimodal import _image_row
from tests.model.test_qwen4_exp_multimodal import _lora
from tests.model.test_qwen4_exp_multimodal import _model
from tests.model.test_qwen4_exp_multimodal import _StopForward

# One three-frame video of 2x2 patches (merge 1): each frame's 4 tokens follow its timestamp text.
VIDEO_GRID_THW = torch.tensor([[3, 2, 2]])
TIMESTAMP_TOKEN_IDS = (10, 11, 12)

GROUPS = [(2, rank) for rank in range(2)] + [(4, rank) for rank in range(4)]


class _FakeGroup:
    """A CP group of ``world_size`` ranks whose current rank the test sets."""

    def __init__(self, monkeypatch, world_size: int):
        from arctic_platform.model.implementations.qwen38 import context_parallel

        self.rank = 0
        monkeypatch.setattr(context_parallel.dist, "get_world_size", lambda group=None: world_size)
        monkeypatch.setattr(context_parallel.dist, "get_rank", lambda group=None: self.rank)
        monkeypatch.setattr(context_parallel, "_adapt_linear_attention", lambda module, process_group: None)


def _cp_model(monkeypatch, world_size: int):
    from arctic_platform.model.implementations.qwen38 import context_parallel

    model = _model()
    group = _FakeGroup(monkeypatch, world_size)
    context_parallel.apply_context_parallelism(model, world_size, object())
    return model, group


def _patch_dim(model) -> int:
    vision = model.config.vision_config
    return vision.in_channels * vision.temporal_patch_size * vision.patch_size**2


def _image_call(model):
    """Two packed rows: the spec's two-image row (24 tokens), then a text row (8 tokens)."""
    image_ids, image_types = _image_row()
    input_ids = torch.cat([image_ids, torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])], dim=1)
    mm_token_type_ids = torch.cat([image_types, torch.zeros(1, 8, dtype=image_types.dtype)], dim=1)
    set_seed(1)
    pixel_values = torch.randn(int(IMAGE_GRID_THW.prod(-1).sum()), _patch_dim(model))
    position_ids = mrope_position_ids(
        mm_token_type_ids,
        torch.tensor([0, image_ids.shape[-1], input_ids.shape[-1]]),
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        image_grid_thw=IMAGE_GRID_THW,
    )
    return dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=IMAGE_GRID_THW,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
    )


def _video_call(model):
    """One row holding a video whose frames are separated by their timestamp text, as the processor lays it out."""
    video_token_id = model.config.video_token_id
    frame_tokens = int(VIDEO_GRID_THW[0, 1:].prod())
    tokens = [2]
    for timestamp in TIMESTAMP_TOKEN_IDS:
        tokens += [timestamp, VISION_START_TOKEN_ID] + [video_token_id] * frame_tokens + [VISION_END_TOKEN_ID]
    tokens += [4, 5]
    input_ids = torch.tensor([tokens])
    mm_token_type_ids = (input_ids == video_token_id).int() * 2
    set_seed(2)
    pixel_values_videos = torch.randn(int(VIDEO_GRID_THW.prod(-1).sum()), _patch_dim(model))
    position_ids = mrope_position_ids(
        mm_token_type_ids,
        torch.tensor([0, input_ids.shape[-1]]),
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        video_grid_thw=VIDEO_GRID_THW,
    )
    return dict(
        input_ids=input_ids,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=VIDEO_GRID_THW,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
    )


CALLS = {"image": _image_call, "video": _video_call}


def _whole(inputs):
    """``inputs`` marked as the whole packed call, to be sharded after the vision merge."""
    return dict(inputs, cp_shard_after_vision_merge=True)


def _language_model_inputs(model, inputs):
    """The keyword inputs the language model gets, after the vision merge (and the CP shard); resolved through PEFT."""
    captured = {}

    def capture(_module, _args, kwargs):
        captured.update(kwargs)
        raise _StopForward

    handle = get_language_model(model).register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with pytest.raises(_StopForward), torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured


@pytest.mark.parametrize("world_size, rank", GROUPS)
@pytest.mark.parametrize("kind", sorted(CALLS))
def test_each_rank_gets_its_window_of_the_merged_media_call(monkeypatch, kind, world_size, rank):
    reference = _model().eval()
    inputs = CALLS[kind](reference)
    unsharded = _language_model_inputs(reference, inputs)
    model, group = _cp_model(monkeypatch, world_size)
    model.eval()
    group.rank = rank

    sharded = _language_model_inputs(model, _whole(inputs))

    start, end = token_window(inputs["input_ids"].shape[-1], world_size, rank)
    assert "cp_shard_after_vision_merge" not in sharded
    torch_assert_equal(sharded["inputs_embeds"], unsharded["inputs_embeds"][:, start:end])
    torch_assert_equal(sharded["ple_input_ids"], inputs["input_ids"][:, start:end])
    torch_assert_equal(sharded["position_ids"], inputs["position_ids"][..., start:end])
    torch_assert_equal(sharded["ple_position_ids"], inputs["position_ids"][0, :, start:end])


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("kind", sorted(CALLS))
def test_ranks_windows_concatenate_to_the_unsharded_merged_call(monkeypatch, kind, world_size):
    reference = _model().eval()
    inputs = CALLS[kind](reference)
    unsharded = _language_model_inputs(reference, inputs)
    model, group = _cp_model(monkeypatch, world_size)
    model.eval()

    windows = []
    for rank in range(world_size):
        group.rank = rank
        windows.append(_language_model_inputs(model, _whole(inputs)))

    for key in ("inputs_embeds", "ple_input_ids", "ple_position_ids"):
        torch_assert_equal(torch.cat([window[key] for window in windows], dim=1), unsharded[key], msg=key)
    torch_assert_equal(torch.cat([window["position_ids"] for window in windows], dim=-1), unsharded["position_ids"])


@pytest.mark.parametrize("world_size, rank", GROUPS)
def test_text_only_call_reaches_the_language_model_as_without_cp(monkeypatch, world_size, rank):
    # A text-only call is sharded before the model, so a rank's window is what it passes in.
    input_ids = torch.arange(2, 2 + 4 * world_size).reshape(1, -1)
    start, end = token_window(input_ids.shape[-1], world_size, rank)
    inputs = dict(input_ids=input_ids[:, start:end], position_ids=torch.arange(start, end).reshape(1, -1))
    reference = _model().eval()
    expected = _language_model_inputs(reference, inputs)
    model, group = _cp_model(monkeypatch, world_size)
    model.eval()
    group.rank = rank

    actual = _language_model_inputs(model, inputs)

    assert actual.keys() == expected.keys()
    for key in ("inputs_embeds", "ple_input_ids", "ple_position_ids", "position_ids"):
        torch_assert_equal(actual[key], expected[key], msg=key)


def _identity_language_model(model):
    """Stand in for the backbone, whose CP collectives need a real group: return the merged embeddings as is."""

    def forward(*_args, inputs_embeds, **_kwargs):
        return BaseModelOutputWithPast(last_hidden_state=inputs_embeds)

    model.model.language_model.forward = forward


@pytest.mark.parametrize("world_size, rank", GROUPS)
def test_dummy_vision_forward_gives_zero_vision_grads_under_cp(monkeypatch, world_size, rank):
    model, group = _cp_model(monkeypatch, world_size)
    model.train()
    group.rank = rank
    _identity_language_model(model)
    calls, _ = _count_vision_forwards(model)

    model(input_ids=TEXT_ONLY_INPUT_IDS).logits.float().square().mean().backward()

    assert len(calls) == 1
    for name, parameter in model.model.visual.named_parameters():
        assert parameter.grad is not None, name
        torch_assert_equal(parameter.grad, torch.zeros_like(parameter), msg=name)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("kind", sorted(CALLS))
def test_ranks_vision_grads_sum_to_the_unsharded_vision_grads(monkeypatch, kind, world_size):
    reference = _model().train()
    _identity_language_model(reference)
    inputs = CALLS[kind](reference)
    # A per-token sum, so the unsharded loss is the sum of the ranks' window losses.
    reference(**inputs).logits.float().square().sum().backward()
    model, group = _cp_model(monkeypatch, world_size)
    model.train()
    _identity_language_model(model)

    for rank in range(world_size):
        group.rank = rank
        model(**_whole(inputs)).logits.float().square().sum().backward()

    vision_grads = dict(model.model.visual.named_parameters())
    for name, parameter in reference.model.visual.named_parameters():
        # The ranks' grads accumulate in a different order than one backward over every token: fp32 rounding only.
        torch_assert_close(vision_grads[name].grad, parameter.grad, rtol=0, atol=1e-5, msg=name)


def test_media_call_without_packed_positions_is_refused(monkeypatch):
    # Without position_ids the forward builds [3, B, T] M-RoPE ids: no row channel for the backbone's boundaries.
    model, group = _cp_model(monkeypatch, 2)
    model.eval()
    group.rank = 1
    inputs = _image_call(model)
    del inputs["position_ids"]

    with pytest.raises(ValueError, match=r"packed \[4, batch, sequence\] M-RoPE position_ids"), torch.no_grad():
        model(**_whole(inputs))


def test_lora_wrapped_model_shards_its_media_calls(monkeypatch):
    from arctic_platform.model.implementations.qwen38 import context_parallel

    reference = _lora(_model(), ["q_proj", "k_proj", "v_proj", "o_proj"]).eval()
    inputs = _image_call(reference)
    unsharded = _language_model_inputs(reference, inputs)
    model = _lora(_model(), ["q_proj", "k_proj", "v_proj", "o_proj"]).eval()
    group = _FakeGroup(monkeypatch, 2)
    context_parallel.apply_context_parallelism(model, 2, object())
    group.rank = 1

    sharded = _language_model_inputs(model, _whole(inputs))

    start, end = token_window(inputs["input_ids"].shape[-1], 2, 1)
    torch_assert_equal(sharded["inputs_embeds"], unsharded["inputs_embeds"][:, start:end])
    torch_assert_equal(sharded["ple_input_ids"], inputs["input_ids"][:, start:end])


def test_media_call_that_does_not_split_evenly_is_refused(monkeypatch):
    # The video call has 24 tokens: four ranks split it, five cannot.
    model, _group = _cp_model(monkeypatch, 5)
    model.eval()
    inputs = _video_call(model)

    with pytest.raises(ValueError, match="must divide by shard_count"), torch.no_grad():
        model(**_whole(inputs))


def test_media_call_with_windowed_positions_is_refused(monkeypatch):
    # position_ids that are already one rank's window would be cut a second time.
    model, group = _cp_model(monkeypatch, 2)
    model.eval()
    group.rank = 1
    inputs = _image_call(model)
    inputs["position_ids"] = inputs["position_ids"][..., 16:]

    with pytest.raises(ValueError, match="media call"), torch.no_grad():
        model(**_whole(inputs))


def test_media_call_already_cut_to_a_mid_row_window_is_refused(monkeypatch):
    # Rank 1's 16-token window of the image call, every input cut alike: its width divides by the CP size, but it
    # opens mid-row (position 16), so it cannot be a whole packed call and must not be cut a second time.
    from arctic_platform.model.implementations.qwen38 import context_parallel

    _FakeGroup(monkeypatch, 2)
    reference = _model().eval()
    inputs = _image_call(reference)
    merged = _language_model_inputs(reference, inputs)
    window = {key: merged[key][:, 16:] for key in ("inputs_embeds", "ple_input_ids", "ple_position_ids")}
    window["position_ids"] = merged["position_ids"][..., 16:]

    with pytest.raises(ValueError, match=r"first positions \[16\]"):
        context_parallel._shard_merged_call(None, (), window, process_group=object())


def test_routed_experts_and_attention_mask_are_cut_on_the_token_axis(monkeypatch):
    # Router replay's [B, T, layers, top_k] expert ids must cover the rank's tokens only, like the hidden states,
    # and so must a [B, T] attention mask.
    model, group = _cp_model(monkeypatch, 2)
    model.eval()
    group.rank = 1
    inputs = _image_call(model)
    tokens = inputs["input_ids"].shape[-1]
    layers = model.config.text_config.num_hidden_layers
    routed_experts = torch.arange(tokens * layers * 2).reshape(1, tokens, layers, 2)

    attention_mask = torch.ones(1, tokens, dtype=torch.long)
    attention_mask[:, -3:] = 0

    sharded = _language_model_inputs(
        model, _whole(dict(inputs, routed_experts=routed_experts, attention_mask=attention_mask))
    )

    start, end = token_window(tokens, 2, 1)
    torch_assert_equal(sharded["routed_experts"], routed_experts[:, start:end])
    torch_assert_equal(sharded["attention_mask"], attention_mask[:, start:end])


@pytest.mark.parametrize("rank", [0, 1])
def test_media_call_not_marked_whole_passes_through_unchanged(monkeypatch, rank):
    # Without the flag a call is taken to be the rank's window already, whatever its media and positions.
    reference = _model().eval()
    inputs = _image_call(reference)
    expected = _language_model_inputs(reference, inputs)
    model, group = _cp_model(monkeypatch, 2)
    model.eval()
    group.rank = rank

    actual = _language_model_inputs(model, inputs)

    assert actual.keys() == expected.keys()
    for key in ("inputs_embeds", "ple_input_ids", "ple_position_ids", "position_ids"):
        torch_assert_equal(actual[key], expected[key], msg=key)


def _two_row_windows(model, shard_count=2):
    """The review's sp=2 example through the trainers' own sharding: two 16-token rows, one image in row 0.

    Returns the packed whole call, its metadata, and ``token_shard``'s windows (with ``window_media``'s whole
    images and ``image_window_token_ranges``). Rank 0's window opens a row at position 0, like a whole call.
    """
    image_tokens = int(IMAGE_GRID_THW[0].prod())
    row_0 = [2, VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * image_tokens + [VISION_END_TOKEN_ID]
    row_0 += [3 + i % 7 for i in range(16 - len(row_0))]
    row_1 = [2 + i % 8 for i in range(16)]
    input_ids = torch.tensor([row_0, row_1])
    set_seed(3)
    batch = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(16).repeat(2, 1),
        mm_token_type_ids=(input_ids == IMAGE_TOKEN_ID).long(),
        pixel_values=torch.randn(int(IMAGE_GRID_THW[0].prod()), _patch_dim(model)),
        image_grid_thw=IMAGE_GRID_THW[:1],
        image_row_ids=torch.tensor([0]),
    )
    merge = model.config.vision_config.spatial_merge_size
    packed, metadata = pack_microbatch(batch)
    packed, metadata = pad_packed_microbatch(packed, metadata, shard_count)
    packed["position_ids"] = mrope_position_ids(
        packed["mm_token_type_ids"], metadata.cu_seqlens, spatial_merge_size=merge, image_grid_thw=IMAGE_GRID_THW[:1]
    )
    windows = [
        token_shard(packed, metadata, shard_count, rank, spatial_merge_size=merge) for rank in range(shard_count)
    ]
    return packed, metadata, windows


def _without_empty_media(window):
    # window_media keeps a kind with zero items in the window; the HF vision tower cannot encode zero images,
    # so a window without media reaches the model as a text call.
    if window["image_grid_thw"].shape[0]:
        return window
    return {key: value for key, value in window.items() if "image" not in key and key != "pixel_values"}


@pytest.mark.parametrize("world_size", [2, 4])
def test_token_shard_windows_pass_through_and_each_rank_gets_its_own_window(monkeypatch, world_size):
    reference = _model().eval()
    packed, metadata, windows = _two_row_windows(reference, world_size)
    assert windows[0]["image_window_token_ranges"].tolist() == [[0, int(IMAGE_GRID_THW[0].prod())]]
    assert all(window["image_grid_thw"].shape[0] == 0 for window in windows[1:])
    unsharded = _language_model_inputs(reference, {k: v for k, v in packed.items() if k != "image_row_ids"})
    model, group = _cp_model(monkeypatch, world_size)
    model.eval()

    received = []
    for rank, window in enumerate(windows):
        group.rank = rank
        received.append(_language_model_inputs(model, _without_empty_media(window)))

    assert [inputs["inputs_embeds"].shape[1] for inputs in received] == [32 // world_size] * world_size
    for rank, inputs in enumerate(received):
        start, end = token_window(metadata.padded_tokens, world_size, rank)
        torch_assert_equal(inputs["inputs_embeds"], unsharded["inputs_embeds"][:, start:end], msg=str(rank))
        torch_assert_equal(inputs["ple_input_ids"], packed["input_ids"][:, start:end], msg=str(rank))
        torch_assert_equal(inputs["position_ids"], packed["position_ids"][..., start:end], msg=str(rank))
    # The backbone all-gathers the ranks' row positions on the token axis and reads the global rows from them.
    from arctic_platform.model.implementations.qwen38 import context_parallel

    gathered = torch.cat([context_parallel._text_position_ids(inputs["position_ids"]) for inputs in received], dim=-1)
    torch_assert_equal(cu_seqlens_from_position_ids(gathered).to(metadata.cu_seqlens.dtype), metadata.cu_seqlens)


def test_token_shard_window_marked_whole_is_refused(monkeypatch):
    # Rank 0's window opens at position 0 and divides evenly: only its window ranges say it is not the whole call.
    model, group = _cp_model(monkeypatch, 2)
    model.eval()
    _, _, windows = _two_row_windows(model)

    with pytest.raises(ValueError, match="image_window_token_ranges"), torch.no_grad():
        model(**_whole(windows[0]))


def test_text_only_call_marked_whole_is_refused(monkeypatch):
    model, _group = _cp_model(monkeypatch, 2)
    model.eval()

    with pytest.raises(ValueError, match="a text-only call is sharded before the model"), torch.no_grad():
        model(**_whole(dict(input_ids=TEXT_ONLY_INPUT_IDS)))
