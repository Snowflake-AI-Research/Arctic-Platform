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
"""Image training cases for Qwen3.8 (``qwen4_exp``), the spec for porting prime-rl #3916 into Arctic.

PrimeIntellect-ai/prime-rl PR #3916 ("Add Qwen 3-8 flash multimodal support", Apache-2.0) trains Qwen3.8 on
images: a checkpoint-compatible vision encoder, image features scattered into the placeholder tokens, packed
M-RoPE positions from ``image_grid_thw``, context-parallel sharding after the vision merge, raw placeholder ids
kept for the PLE n-gram hashing, vision weights kept through checkpoint conversion, and a dummy vision forward
for frozen-vision and text-only microbatches. Its LoRA recipe keeps the vision encoder frozen and puts adapters on
the language model only. The PR's own test files could not be read from the host this was
written on, so the cases follow the PR's description and the media research plan, re-expressed against
``Qwen4ExpForConditionalGenerationPrimeRL``; no prime-rl code is copied.

Cases Arctic already satisfies pass. Cases that need trainer code not yet ported are
``xfail(strict=True)`` with the missing piece in the reason, so they flip to failures once it lands.

CPU only, on the tiny random-weight config of ``test_qwen4_exp.py``. The DSS MoE router counts tokens with
``torch.histc`` on integer ids, which has no CPU kernel, so every decoder layer's MoE is swapped for a dense
projection; the vision tower, PLE, hyper-connections, GatedDeltaNet and QSA run as built.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_qwen4_exp import _tiny_config

IMAGE_TOKEN_ID = 29
VISION_START_TOKEN_ID = 27
VISION_END_TOKEN_ID = 28
# Patches before the spatial merge (merge 1 in the tiny config): 4 and 8 placeholder tokens.
IMAGE_GRID_THW = torch.tensor([[1, 2, 2], [1, 2, 4]])


class _DenseMLP(torch.nn.Module):
    """CPU stand-in for the decoder's MoE: one projection, so gradients still reach the layers below."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, routed_experts=None):
        return self.proj(hidden_states)


class _StopForward(Exception):
    pass


def _model(*, dense_mlp: bool = True):
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL

    config = _tiny_config()
    for sub_config in (config, config.text_config, config.vision_config):
        sub_config._attn_implementation = "sdpa"
    set_seed(0)
    model = Qwen4ExpForConditionalGenerationPrimeRL(config)
    if dense_mlp:
        for layer in model.model.language_model.layers:
            layer.mlp = _DenseMLP(config.text_config.hidden_size)
    return model


def _image_row():
    """One row with two images, laid out as the processor does: text on both sides of each image."""
    tokens = [2, 3]
    for grid_thw in IMAGE_GRID_THW:
        tokens += _image_tokens(grid_thw) + [4]
    tokens += [5, 6, 7, 8]
    input_ids = torch.tensor([tokens])
    mm_token_type_ids = (input_ids == IMAGE_TOKEN_ID).int()
    return input_ids, mm_token_type_ids


def _image_tokens(grid_thw: torch.Tensor) -> list[int]:
    return [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * int(grid_thw.prod()) + [VISION_END_TOKEN_ID]


def _pixel_values(model):
    vision = model.config.vision_config
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return torch.randn(int(IMAGE_GRID_THW.prod(-1).sum()), patch_dim)


def _capture_language_model_inputs(model, *, stop: bool = False):
    """Record the keyword inputs the language model is called with, after the vision merge."""
    captured = {}

    def hook(_module, _args, kwargs):
        captured.update(kwargs)
        if stop:
            raise _StopForward

    handle = model.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
    return captured, handle


def _packed_positions(model, mm_token_type_ids):
    from arctic_platform.common.packing import mrope_position_ids

    return mrope_position_ids(
        mm_token_type_ids,
        torch.tensor([0, mm_token_type_ids.shape[-1]]),
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        image_grid_thw=IMAGE_GRID_THW,
    )


def test_image_features_fill_placeholders_in_grid_order():
    model = _model().eval()
    input_ids, mm_token_type_ids = _image_row()
    pixel_values = _pixel_values(model)
    captured, _ = _capture_language_model_inputs(model)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=IMAGE_GRID_THW,
            mm_token_type_ids=mm_token_type_ids,
        )
        image_features = model.model.get_image_features(pixel_values, IMAGE_GRID_THW).pooler_output
        token_embeddings = model.model.get_input_embeddings()(input_ids)

    inputs_embeds = captured["inputs_embeds"]
    image_positions = input_ids[0] == IMAGE_TOKEN_ID
    assert [feature.shape[0] for feature in image_features] == IMAGE_GRID_THW.prod(-1).tolist()
    torch_assert_equal(inputs_embeds[0, image_positions], torch.cat(image_features))
    torch_assert_equal(inputs_embeds[0, ~image_positions], token_embeddings[0, ~image_positions])


def test_image_placeholder_count_must_match_the_grids():
    model = _model().eval()
    input_ids, mm_token_type_ids = _image_row()
    first_image_token = int((input_ids[0] == IMAGE_TOKEN_ID).nonzero()[0])
    input_ids[0, first_image_token] = 4
    mm_token_type_ids[0, first_image_token] = 0

    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        model(
            input_ids=input_ids,
            pixel_values=_pixel_values(model),
            image_grid_thw=IMAGE_GRID_THW,
            mm_token_type_ids=mm_token_type_ids,
        )


def test_ple_hashes_raw_placeholder_ids_not_image_features():
    model = _model().eval()
    input_ids, mm_token_type_ids = _image_row()
    captured, _ = _capture_language_model_inputs(model)
    ple_inputs = []

    def record_ple_ids(_module, args, kwargs):
        ple_inputs.append(kwargs.get("input_ids", args[1] if len(args) > 1 else None))

    for layer in model.model.language_model.layers:
        if layer.ple is not None:
            layer.ple.register_forward_pre_hook(record_ple_ids, with_kwargs=True)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            pixel_values=_pixel_values(model),
            image_grid_thw=IMAGE_GRID_THW,
            mm_token_type_ids=mm_token_type_ids,
        )

    # The image rows of inputs_embeds no longer decode to the placeholder, so PLE must not recover ids from them.
    torch_assert_equal(captured["ple_input_ids"], input_ids)
    assert len(ple_inputs) == 1
    torch_assert_equal(ple_inputs[0], input_ids)


def test_packed_mrope_positions_match_hugging_face_positions_in_the_forward():
    model = _model().eval()
    input_ids, mm_token_type_ids = _image_row()
    pixel_values = _pixel_values(model)
    inputs = dict(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=IMAGE_GRID_THW)
    captured, _ = _capture_language_model_inputs(model)
    packed_positions = _packed_positions(model, mm_token_type_ids)

    with torch.no_grad():
        reference = model(**inputs, mm_token_type_ids=mm_token_type_ids).logits
        reference_positions = captured["position_ids"]
        packed = model(**inputs, position_ids=packed_positions).logits

    # Hugging Face's get_rope_index returns the three rotary channels; Arctic's packer adds the text channel first.
    torch_assert_equal(packed_positions[1:], reference_positions)
    torch_assert_equal(packed_positions[0, 0], torch.arange(input_ids.shape[-1]))
    torch_assert_equal(captured["position_ids"], packed_positions)
    torch_assert_equal(packed, reference)


def test_packed_rows_own_their_images_and_restart_positions():
    from arctic_platform.common.packing import mrope_position_ids

    model = _model().eval()
    rows = [
        [2, 3] + _image_tokens(IMAGE_GRID_THW[0]) + [4, 5],
        [6] + _image_tokens(IMAGE_GRID_THW[1]) + [7, 8, 9],
    ]
    input_ids = torch.tensor([rows[0] + rows[1]])
    mm_token_type_ids = (input_ids == IMAGE_TOKEN_ID).int()
    cu_seqlens = torch.tensor([0, len(rows[0]), len(rows[0]) + len(rows[1])])
    position_ids = mrope_position_ids(
        mm_token_type_ids,
        cu_seqlens,
        spatial_merge_size=model.config.vision_config.spatial_merge_size,
        image_grid_thw=IMAGE_GRID_THW,
    )
    pixel_values = _pixel_values(model)
    captured, _ = _capture_language_model_inputs(model)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=IMAGE_GRID_THW,
            position_ids=position_ids,
        )
        image_features = model.model.get_image_features(pixel_values, IMAGE_GRID_THW).pooler_output

    second_row = slice(len(rows[0]), None)
    for row, features in zip((slice(0, len(rows[0])), second_row), image_features):
        image_positions = input_ids[0, row] == IMAGE_TOKEN_ID
        torch_assert_equal(captured["inputs_embeds"][0, row][image_positions], features)
    torch_assert_equal(captured["position_ids"], position_ids)
    torch_assert_equal(position_ids[0, 0, second_row], torch.arange(len(rows[1])))
    torch_assert_equal(captured["ple_input_ids"], input_ids)


def test_frozen_vision_trains_the_language_model_only():
    model = _model().train()
    model.model.visual.requires_grad_(False)
    input_ids, mm_token_type_ids = _image_row()

    logits = model(
        input_ids=input_ids,
        pixel_values=_pixel_values(model),
        image_grid_thw=IMAGE_GRID_THW,
        mm_token_type_ids=mm_token_type_ids,
    ).logits
    logits.float().square().mean().backward()

    assert all(parameter.grad is None for parameter in model.model.visual.parameters())
    language_model = model.model.language_model
    assert language_model.embed_tokens.weight.grad is not None
    assert language_model.layers[1].self_attn.q_proj.weight.grad is not None


def test_vision_weights_survive_checkpoint_conversion():
    from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_hf_to_prime
    from arctic_platform.model.implementations.qwen38.converting_qwen4_exp import convert_prime_to_hf

    model = _model(dense_mlp=False)
    prime_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    vision_state = {name: tensor for name, tensor in prime_state.items() if name.startswith("model.visual.")}
    assert any(".merger." in name for name in vision_state)
    assert any(".pos_embed." in name for name in vision_state)

    state = dict(prime_state)
    convert_prime_to_hf(state)
    assert type(model).is_hf_state_dict(state)
    for name, tensor in vision_state.items():
        torch_assert_equal(state[name], tensor)

    convert_hf_to_prime(state)
    assert {name for name in state if name.startswith("model.visual.")} == set(vision_state)
    for name, tensor in vision_state.items():
        torch_assert_equal(state[name], tensor)


def test_meta_init_restores_the_vision_rotary_frequencies():
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL

    expected = _model(dense_mlp=False).model.visual.rotary_pos_emb.inv_freq
    with torch.device("meta"):
        model = Qwen4ExpForConditionalGenerationPrimeRL(_tiny_config())
    model.to_empty(device="cpu")
    model.init_buffers_post_meta()

    torch_assert_close(model.model.visual.rotary_pos_emb.inv_freq, expected, rtol=0, atol=0)


def test_vlm_registry_resolves_the_vision_tower_and_language_model():
    from arctic_platform.model import get_vision_encoder
    from arctic_platform.model.implementations.moe.vlm import get_language_model

    model = _model()

    assert get_vision_encoder(model) is model.model.visual
    assert get_language_model(model) is model.model.language_model


def _lora(model, target_modules):
    pytest.importorskip("peft")
    from arctic_platform.model.patches.peft import apply_peft

    return apply_peft(model, {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": target_modules})


def _vision_adapters(model):
    return [name for name, _ in model.named_parameters() if "lora_" in name and ".visual." in name]


def test_lora_on_language_projections_leaves_the_vision_tower_frozen():
    from arctic_platform.model import get_vision_encoder

    model = _lora(_model(), ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Resolved through the PEFT wrapper.
    vision = get_vision_encoder(model)

    assert any("lora_" in name for name, _ in model.named_parameters())
    assert _vision_adapters(model) == []
    assert not any(parameter.requires_grad for parameter in vision.parameters())


def test_all_linear_lora_skips_the_vision_tower():
    model = _lora(_model(), "all-linear")

    assert _vision_adapters(model) == []


TEXT_ONLY_INPUT_IDS = torch.tensor([[2, 3, 4, 5, 6, 7]])


def test_text_only_microbatch_still_touches_a_trainable_vision_tower():
    model = _model().train()
    input_ids = TEXT_ONLY_INPUT_IDS

    model(input_ids=input_ids).logits.float().square().mean().backward()

    for name, parameter in model.model.visual.named_parameters():
        assert parameter.grad is not None, name
        torch_assert_equal(parameter.grad, torch.zeros_like(parameter))


def _count_vision_forwards(model):
    calls = []
    handle = model.model.visual.register_forward_hook(lambda *_: calls.append(None))
    return calls, handle


def test_dummy_vision_forward_leaves_text_only_logits_and_language_grads_bitwise_unchanged():
    model = _model().train()
    reference = _model().train()
    reference.model.visual.requires_grad_(False)
    calls, _ = _count_vision_forwards(model)

    logits = model(input_ids=TEXT_ONLY_INPUT_IDS).logits
    logits.float().square().mean().backward()
    reference_logits = reference(input_ids=TEXT_ONLY_INPUT_IDS).logits
    reference_logits.float().square().mean().backward()

    assert len(calls) == 1
    torch_assert_equal(logits, reference_logits)
    language_grads = dict(model.model.language_model.named_parameters())
    for name, parameter in reference.model.language_model.named_parameters():
        if parameter.grad is None:
            assert language_grads[name].grad is None, name
        else:
            torch_assert_equal(language_grads[name].grad, parameter.grad, msg=name)


def test_frozen_vision_tower_skips_the_dummy_forward():
    model = _model().train()
    model.model.visual.requires_grad_(False)
    calls, _ = _count_vision_forwards(model)

    model(input_ids=TEXT_ONLY_INPUT_IDS).logits.float().square().mean().backward()

    assert calls == []
    assert all(parameter.grad is None for parameter in model.model.visual.parameters())


def test_no_grad_and_eval_skip_the_dummy_forward():
    model = _model().train()
    calls, _ = _count_vision_forwards(model)

    with torch.no_grad():
        model(input_ids=TEXT_ONLY_INPUT_IDS)
    model.eval()
    model(input_ids=TEXT_ONLY_INPUT_IDS)

    assert calls == []


def test_image_batch_runs_the_vision_tower_once_and_is_unchanged():
    model = _model().train()
    reference = _model().train()
    input_ids, mm_token_type_ids = _image_row()
    pixel_values = _pixel_values(model)
    inputs = dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=IMAGE_GRID_THW,
        mm_token_type_ids=mm_token_type_ids,
    )
    calls, _ = _count_vision_forwards(model)

    logits = model(**inputs).logits
    logits.float().square().mean().backward()
    # The stock Hugging Face forward, without the Arctic override, is the baseline.
    reference_logits = super(type(reference.model), reference.model).forward(**inputs).last_hidden_state
    reference_logits = reference.lm_head(reference_logits)
    reference_logits.float().square().mean().backward()

    assert len(calls) == 1
    torch_assert_equal(logits, reference_logits)
    vision_grads = dict(model.model.visual.named_parameters())
    for name, parameter in reference.model.visual.named_parameters():
        torch_assert_equal(vision_grads[name].grad, parameter.grad, msg=name)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "needs #3916 port: context-parallel sharding after the vision merge; each rank takes the full packed row "
        "with its images, merges image features, then keeps its own slice of embeddings, raw PLE ids and positions"
    ),
)
def test_context_parallel_shards_after_the_vision_merge(monkeypatch):
    from arctic_platform.model.implementations.qwen38 import context_parallel

    model = _model().eval()
    input_ids, mm_token_type_ids = _image_row()
    pixel_values = _pixel_values(model)
    packed_positions = _packed_positions(model, mm_token_type_ids)
    sequence_length = input_ids.shape[-1]
    assert sequence_length % 2 == 0
    local = slice(sequence_length // 2, sequence_length)
    inputs = dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=IMAGE_GRID_THW,
        position_ids=packed_positions,
    )

    unsharded, handle = _capture_language_model_inputs(model, stop=True)
    with pytest.raises(_StopForward), torch.no_grad():
        model(**inputs)
    handle.remove()

    # Rank 1 of a two-rank context-parallel group; the collectives run only inside the backbone forward.
    monkeypatch.setattr(context_parallel.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(context_parallel.dist, "get_rank", lambda group=None: 1)
    # The GatedDeltaNet adapter imports fla, a CUDA package; the shard is taken before any layer runs.
    monkeypatch.setattr(context_parallel, "_adapt_linear_attention", lambda module, process_group: None)
    context_parallel.apply_context_parallelism(model, 2, object())
    sharded, _ = _capture_language_model_inputs(model, stop=True)
    with pytest.raises(_StopForward), torch.no_grad():
        model(**inputs)

    torch_assert_equal(sharded["inputs_embeds"], unsharded["inputs_embeds"][:, local])
    torch_assert_equal(sharded["ple_input_ids"], input_ids[:, local])
    torch_assert_equal(sharded["position_ids"], packed_positions[..., local])


def test_video_batch_runs_the_vision_tower_once():
    model = _model().train()
    video_grid_thw = IMAGE_GRID_THW[:1]
    video_token_id = model.config.video_token_id
    input_ids = torch.tensor([[2, 3, VISION_START_TOKEN_ID] + [video_token_id] * 4 + [VISION_END_TOKEN_ID, 4, 5]])
    mm_token_type_ids = (input_ids == video_token_id).int() * 2
    pixel_values_videos = torch.randn(4, _pixel_values(model).shape[-1])
    calls, _ = _count_vision_forwards(model)

    model(
        input_ids=input_ids,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    ).logits.float().square().mean().backward()

    assert len(calls) == 1
    assert all(parameter.grad is not None for parameter in model.model.visual.parameters())


def _vision_attention_backends(model):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionAttention

    modules = model.modules()
    return {module.config._attn_implementation for module in modules if isinstance(module, Qwen4ExpVisionAttention)}


def test_qsa_flex_moves_the_vision_tower_to_sdpa():
    from arctic_platform.model.implementations.qwen38.qsa_flex import apply_qsa_flex
    from arctic_platform.model.implementations.qwen38.qsa_flex import register_qsa_flex_backend

    model = _model()
    register_qsa_flex_backend()
    # Training sets the backend on the top config; the setter copies it into the vision config.
    model.config._attn_implementation = "qsa_flex"
    assert _vision_attention_backends(model) == {"qsa_flex"}

    apply_qsa_flex(model)

    assert _vision_attention_backends(model) == {"sdpa"}
    with torch.no_grad():
        model.model.visual(_pixel_values(model), grid_thw=IMAGE_GRID_THW)


def test_qsa_flex_leaves_another_vision_backend_alone():
    from arctic_platform.model.implementations.qwen38.qsa_flex import apply_qsa_flex

    model = _model()
    model.config.vision_config._attn_implementation = "eager"
    assert _vision_attention_backends(model) == {"eager"}

    apply_qsa_flex(model)

    assert _vision_attention_backends(model) == {"eager"}
