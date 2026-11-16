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

"""CPU checks for image inputs to the custom Qwen3.5-MoE vision-language model.

The M-RoPE positions and rotary angles are compared with the transformers implementation, which the sampler's
vLLM model reproduces. The tiny model swaps each routed-expert MoE for a dense projection and runs GatedDeltaNet
through its pure-PyTorch fallback, as ``test_qwen35_moe_numerics`` does, because the MoE router cannot run on CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3_5MoeConfig as HFQwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel as HFQwen3_5MoeModel
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
from arctic_platform.model.implementations.moe.mrope import mrope_cos_sin
from arctic_platform.model.implementations.moe.mrope import mrope_position_ids
from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

IMAGE, VIDEO, START, END = 60, 61, 62, 63
MERGE = 2
PATCH_FEATURES = 3 * 2 * 4 * 4  # channels * temporal_patch_size * patch_size**2

_TEXT_CONFIG = dict(
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_conv_kernel_dim=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    num_experts=4,
    num_experts_per_tok=2,
    layer_types=["linear_attention", "full_attention"],
    pad_token_id=0,
    rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.5, mrope_section=[2, 1, 1]),
)
_VISION_CONFIG = dict(depth=1, hidden_size=16, num_heads=2, intermediate_size=16, out_hidden_size=16, patch_size=4)


def _hf_config() -> HFQwen3_5MoeConfig:
    return HFQwen3_5MoeConfig(
        text_config=_TEXT_CONFIG,
        vision_config=_VISION_CONFIG,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=START,
        vision_end_token_id=END,
    )


def _sequence(*parts) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Token ids and image grids for parts that are text lengths or ``(t, h, w)`` image grids."""
    ids, grids, token = [], [], 1
    for part in parts:
        if isinstance(part, int):
            ids += list(range(token, token + part))
            token += part
        else:
            ids += [START] + [IMAGE] * (part[0] * part[1] * part[2] // MERGE**2) + [END]
            grids.append(part)
    return ids, grids


SEQUENCES = [
    _sequence(3, (1, 4, 4), 2),
    _sequence(2, (1, 4, 8), 1, (1, 2, 2), 3),
    _sequence(5),
    _sequence((1, 6, 4), 4),
]


def _hf_positions(ids: list[int], grids: list[tuple[int, int, int]], attention_mask=None) -> torch.Tensor:
    """``[3, 1, S]`` positions from transformers' ``get_rope_index`` for one sequence."""
    host = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=MERGE)))
    host.get_vision_position_ids = lambda *args, **kwargs: HFQwen3_5MoeModel.get_vision_position_ids(
        host, *args, **kwargs
    )
    input_ids = torch.tensor([ids])
    positions, _ = HFQwen3_5MoeModel.get_rope_index(
        host,
        input_ids,
        (input_ids == IMAGE).int(),
        image_grid_thw=torch.tensor(grids) if grids else None,
        attention_mask=attention_mask,
    )
    return positions


@pytest.mark.parametrize("index", range(len(SEQUENCES)))
def test_positions_match_transformers_for_one_sequence(index):
    ids, grids = SEQUENCES[index]
    input_ids = torch.tensor([ids])
    ours = mrope_position_ids(
        input_ids,
        torch.arange(len(ids))[None],
        torch.tensor(grids, dtype=torch.long).reshape(-1, 3),
        image_token_id=IMAGE,
        spatial_merge_size=MERGE,
    )
    torch_assert_equal(ours, _hf_positions(ids, grids))


def test_packed_sequences_are_numbered_independently():
    ids = [token for seq_ids, _ in SEQUENCES for token in seq_ids]
    position_ids = torch.cat([torch.arange(len(seq_ids)) for seq_ids, _ in SEQUENCES])[None]
    grids = torch.tensor([grid for _, seq_grids in SEQUENCES for grid in seq_grids])
    ours = mrope_position_ids(torch.tensor([ids]), position_ids, grids, image_token_id=IMAGE, spatial_merge_size=MERGE)
    expected = torch.cat([_hf_positions(seq_ids, seq_grids) for seq_ids, seq_grids in SEQUENCES], dim=-1)
    torch_assert_equal(ours, expected)


def test_padded_rows_match_transformers_on_real_tokens():
    rows = SEQUENCES[:2]
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.zeros(len(rows), width, dtype=torch.long)
    for row, (ids, _) in enumerate(rows):
        input_ids[row, : len(ids)] = torch.tensor(ids)
    grids = torch.tensor([grid for _, seq_grids in rows for grid in seq_grids])
    ours = mrope_position_ids(
        input_ids, torch.arange(width)[None], grids, image_token_id=IMAGE, spatial_merge_size=MERGE
    )
    for row, (ids, seq_grids) in enumerate(rows):
        torch_assert_equal(ours[:, row, : len(ids)], _hf_positions(ids, seq_grids)[:, 0])


def test_image_tokens_and_grids_must_agree():
    ids, grids = SEQUENCES[0]
    input_ids, positions = torch.tensor([ids]), torch.arange(len(ids))[None]
    with pytest.raises(ValueError, match="describes 2 images but input_ids hold 1"):
        mrope_position_ids(input_ids, positions, torch.tensor(grids * 2), image_token_id=IMAGE, spatial_merge_size=2)
    with pytest.raises(ValueError, match="consecutive image tokens"):
        mrope_position_ids(input_ids, positions, torch.tensor([[1, 8, 8]]), image_token_id=IMAGE, spatial_merge_size=2)
    with pytest.raises(ValueError, match="more image tokens"):
        mrope_position_ids(
            input_ids, positions, torch.zeros(0, 3, dtype=torch.long), image_token_id=IMAGE, spatial_merge_size=2
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_angles_match_transformers(dtype):
    config = _hf_config().text_config
    reference = Qwen3_5MoeTextRotaryEmbedding(config)
    positions = torch.randint(0, 50, (3, 2, 11), generator=torch.Generator().manual_seed(0))
    x = torch.zeros(2, 11, 16, dtype=dtype)
    expected_cos, expected_sin = reference(x, positions)
    cos, sin = mrope_cos_sin(reference.inv_freq, reference.attention_scaling, positions, (2, 1, 1), dtype)
    torch_assert_equal(cos, expected_cos)
    torch_assert_equal(sin, expected_sin)


def test_text_positions_rotate_exactly_as_one_dimensional_rope():
    model = _tiny_vlm()
    rotary = model.model.language_model.rotary_emb
    positions = torch.arange(9)[None]
    x = torch.zeros(1, 9, 16, dtype=torch.bfloat16)
    expected = rotary(x, positions)
    actual = mrope_cos_sin(rotary.inv_freq, rotary.attention_scaling, positions.expand(3, 1, -1), (2, 1, 1), x.dtype)
    torch_assert_equal(actual[0], expected[0])
    torch_assert_equal(actual[1], expected[1])


class _DenseExperts(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(hidden_states)


def _torch_gated_delta_rule(query, key, value, *, cu_seqlens=None, **kwargs):
    assert cu_seqlens is None
    return qwen.torch_chunk_gated_delta_rule(query, key, value, **kwargs)


@pytest.fixture(autouse=True)
def _without_optional_gated_delta_net_kernels(monkeypatch):
    monkeypatch.setattr(qwen, "causal_conv1d_fn", None)
    monkeypatch.setattr(qwen, "chunk_gated_delta_rule", None)
    monkeypatch.setattr(qwen, "FusedRMSNormGated", None)


def _tiny_vlm() -> qwen.Qwen3_5MoeForCausalLM:
    set_seed(0)
    config = _hf_config()
    config.text_config.use_grouped_mm = False
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "eager"
    model = qwen.Qwen3_5MoeForCausalLM(config)
    for layer in model.model.language_model.layers:
        layer.mlp = _DenseExperts(16)
        if layer.layer_type == "linear_attention":
            layer.linear_attn._chunk_gated_delta_rule = _torch_gated_delta_rule
    inject_prime_lm_head(model)
    return model.float().train()


def _images(grids, seed=0) -> torch.Tensor:
    patches = sum(t * h * w for t, h, w in grids)
    return torch.randn(patches, PATCH_FEATURES, generator=torch.Generator().manual_seed(seed))


def _inputs(index: int):
    ids, grids = SEQUENCES[index]
    return torch.tensor([ids]), torch.arange(len(ids))[None], _images(grids, seed=index), torch.tensor(grids)


def test_forward_scatters_image_embeddings_and_rotates_by_transformers_positions():
    model = _tiny_vlm()
    input_ids, position_ids, pixel_values, grids = _inputs(1)
    with torch.no_grad():
        actual = model.model(
            input_ids=input_ids, position_ids=position_ids, pixel_values=pixel_values, image_grid_thw=grids
        ).last_hidden_state

        embeds = model.model.language_model.embed_tokens(input_ids)
        image_embeds = model.model.visual(pixel_values, grid_thw=grids, return_dict=True).pooler_output
        embeds = embeds.masked_scatter((input_ids == IMAGE)[..., None].expand_as(embeds), image_embeds)
        language_model = model.model.language_model
        hf_positions = _hf_positions(*SEQUENCES[1])
        expected = language_model(
            inputs_embeds=embeds, position_ids=position_ids, mrope_position_ids=hf_positions
        ).last_hidden_state
        one_dimensional = language_model(inputs_embeds=embeds, position_ids=position_ids).last_hidden_state

    torch_assert_equal(actual, expected)
    assert not torch.equal(actual, one_dimensional), "image tokens must not rotate by 1-D positions"


def test_images_change_only_the_tokens_from_the_first_image_on():
    model = _tiny_vlm()
    input_ids, position_ids, pixel_values, grids = _inputs(0)
    with torch.no_grad():
        first = model(input_ids=input_ids, position_ids=position_ids, pixel_values=pixel_values, image_grid_thw=grids)
        second = model(
            input_ids=input_ids, position_ids=position_ids, pixel_values=pixel_values + 1.0, image_grid_thw=grids
        )
    first_image = int((input_ids[0] == IMAGE).nonzero()[0])
    torch_assert_equal(first["logits"][:, :first_image], second["logits"][:, :first_image])
    assert not torch.allclose(first["logits"][:, first_image:], second["logits"][:, first_image:])


def test_backward_reaches_the_vision_tower():
    model = _tiny_vlm()
    input_ids, position_ids, pixel_values, grids = _inputs(3)
    out = model(input_ids=input_ids, position_ids=position_ids, pixel_values=pixel_values, image_grid_thw=grids)
    out["logits"].float().logsumexp(-1).sum().backward()
    vision_grads = [p.grad for p in model.model.visual.parameters() if p.requires_grad]
    assert vision_grads and all(g is not None for g in vision_grads)
    assert any(g.abs().sum() > 0 for g in vision_grads)


def test_mismatched_and_unsupported_inputs_raise():
    model = _tiny_vlm()
    input_ids, position_ids, pixel_values, grids = _inputs(0)
    with pytest.raises(ValueError, match="image tokens but the images encode"):
        model(
            input_ids=input_ids[:, :-3],
            position_ids=position_ids[:, :-3],
            pixel_values=torch.cat([pixel_values, pixel_values]),
            image_grid_thw=torch.cat([grids, grids]),
        )
    with pytest.raises(ValueError, match="passed together"):
        model.model(input_ids=input_ids, position_ids=position_ids, pixel_values=pixel_values)
    with pytest.raises(NotImplementedError, match="video"):
        model.model(input_ids=input_ids, position_ids=position_ids, pixel_values_videos=pixel_values)
