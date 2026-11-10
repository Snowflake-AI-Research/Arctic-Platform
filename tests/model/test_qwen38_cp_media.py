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
"""Qwen3.8 context parallelism on windowed media calls.

What the trainers dispatch: each rank gets ``token_shard``'s window of the packed call with the whole media items
in it and their ``*_window_token_ranges``. The call is already sharded and reaches the language model as it would
without CP, text-only calls included.

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
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_qwen4_exp_multimodal import IMAGE_GRID_THW
from tests.model.test_qwen4_exp_multimodal import IMAGE_TOKEN_ID
from tests.model.test_qwen4_exp_multimodal import TEXT_ONLY_INPUT_IDS
from tests.model.test_qwen4_exp_multimodal import VISION_END_TOKEN_ID
from tests.model.test_qwen4_exp_multimodal import VISION_START_TOKEN_ID
from tests.model.test_qwen4_exp_multimodal import _count_vision_forwards
from tests.model.test_qwen4_exp_multimodal import _image_row
from tests.model.test_qwen4_exp_multimodal import _model
from tests.model.test_qwen4_exp_multimodal import _StopForward

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


def _language_model_inputs(model, inputs):
    """The keyword inputs the language model gets, after the vision merge; resolved through PEFT."""
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


@pytest.mark.parametrize("rank", [0, 1])
def test_media_call_passes_through_unchanged(monkeypatch, rank):
    # A media call is taken to be the rank's window already, whatever its media and positions.
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
    images and ``image_window_token_ranges``).
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

