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
"""Periodic and listed gradient checkpointing on a multimodal model: its decoder layers are not ``model.model.layers``.

Gemma 4 (``Gemma4ForConditionalGeneration``) keeps its text decoder at ``model.model.language_model``; the patch
wraps that decoder's listed layers, and a text step (CPU, FP32) gives the loss and gradients of the unwrapped model.
"""

from __future__ import annotations

import copy
import types

import pytest
import torch

from arctic_platform.model import LoaderContext
from arctic_platform.model.patches.gradient_checkpointing import apply_gradient_checkpointing
from arctic_platform.model.patches.utils import transformer_layers

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

VOCAB = 512
LAYERS = 3


def _ctx(gradient_checkpointing) -> LoaderContext:
    patches = types.SimpleNamespace(gradient_checkpointing=gradient_checkpointing)
    return LoaderContext(spec=types.SimpleNamespace(patches=patches))


def _gemma4():
    # gemma-4-31B-it's layout (sliding layers of head size 256, a full layer of 512 with K = V) at toy sizes.
    config = transformers.Gemma4Config(
        text_config={
            "attention_k_eq_v": True,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 64,
            "head_dim": 32,
            "hidden_size": 128,
            "hidden_size_per_layer_input": 0,
            "intermediate_size": 256,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
            "max_position_embeddings": 256,
            "num_attention_heads": 2,
            "num_global_key_value_heads": 1,
            "num_hidden_layers": LAYERS,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "sliding_window": 16,
            "vocab_size": VOCAB,
            "vocab_size_per_layer_input": VOCAB,
        },
        vision_config={
            "global_head_dim": 16,
            "head_dim": 16,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
        },
        audio_config=None,
        image_token_id=VOCAB - 2,
        video_token_id=VOCAB - 1,
    )
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return transformers.Gemma4ForConditionalGeneration(config).float().train()


def _step(model, input_ids):
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return loss.detach(), grads


def test_layers_are_the_language_models():
    model = _gemma4()
    assert getattr(model.model, "layers", None) is None
    assert transformer_layers(model, patch_name="test") is model.model.language_model.layers


@pytest.mark.parametrize("value", [2, [0, 2]], ids=["every-2nd-layer", "layer-list"])
def test_wraps_the_language_models_layers_and_keeps_the_step(value):
    plain = _gemma4()
    wrapped = copy.deepcopy(plain)
    apply_gradient_checkpointing(wrapped, _ctx(value))

    kinds = [type(layer).__name__ for layer in wrapped.model.language_model.layers]
    assert kinds == ["CheckpointWrapper", type(plain.model.language_model.layers[1]).__name__, "CheckpointWrapper"]

    input_ids = torch.randint(0, VOCAB - 8, (2, 48), generator=torch.Generator().manual_seed(1))
    plain_loss, plain_grads = _step(plain, input_ids)
    wrapped_loss, wrapped_grads = _step(wrapped, input_ids)
    torch.testing.assert_close(wrapped_loss, plain_loss, rtol=0, atol=0)
    assert wrapped_grads.keys() == plain_grads.keys()
    for name, grad in plain_grads.items():
        torch.testing.assert_close(wrapped_grads[name], grad, rtol=1e-6, atol=1e-7, msg=name)


def test_rejects_a_model_without_one_text_decoder():
    model = torch.nn.Module()
    with pytest.raises(ValueError, match="requires model.model.layers or one text decoder"):
        apply_gradient_checkpointing(model, _ctx(2))
