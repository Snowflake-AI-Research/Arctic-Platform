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

from types import SimpleNamespace

from torch import nn

from arctic_platform.model.implementations.moe.vlm import get_language_model
from arctic_platform.model.implementations.moe.vlm import get_vision_encoder


class _PeftWrapper(nn.Module):
    """The model nesting exposed by PeftModel without importing PEFT."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.model = base_model
        self.config = base_model.config
        self.peft_config = {"default": object()}

    def get_base_model(self):
        return self.base_model


def test_dense_peft_wrapper_resolves_transformer_backbone():
    backbone = nn.Module()
    backbone.layers = nn.ModuleList()
    base_model = nn.Module()
    base_model.config = SimpleNamespace(model_type="qwen3_5_text")
    base_model.model = backbone

    assert get_language_model(_PeftWrapper(base_model)) is backbone


def test_vlm_peft_wrapper_resolves_language_and_vision_modules():
    language_model = nn.Module()
    language_model.layers = nn.ModuleList()
    vision_encoder = nn.Module()
    base_model = nn.Module()
    base_model.config = SimpleNamespace(model_type="qwen3_5")
    base_model.model = nn.Module()
    base_model.model.language_model = language_model
    base_model.model.visual = vision_encoder
    wrapped = _PeftWrapper(base_model)

    assert get_language_model(wrapped) is language_model
    assert get_vision_encoder(wrapped) is vision_encoder


def test_plain_text_model_keeps_the_text_fallback():
    backbone = nn.Module()
    backbone.layers = nn.ModuleList()
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_5_text")
    model.model = backbone

    assert get_language_model(model) is backbone
    assert get_vision_encoder(model) is None


def test_plain_vlm_resolves_language_and_vision_modules():
    language_model = nn.Module()
    language_model.layers = nn.ModuleList()
    vision_encoder = nn.Module()
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_5")
    model.model = nn.Module()
    model.model.language_model = language_model
    model.model.visual = vision_encoder

    assert get_language_model(model) is language_model
    assert get_vision_encoder(model) is vision_encoder


def test_empty_peft_config_does_not_unwrap():
    backbone = nn.Module()
    backbone.layers = nn.ModuleList()
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_5_text")
    model.model = backbone
    model.peft_config = {}
    decoy = nn.Module()
    decoy.model = nn.Module()
    model.get_base_model = lambda: decoy

    assert get_language_model(model) is backbone
    assert get_vision_encoder(model) is None


def test_override_resolves_on_the_pretrained_model():
    language_model = nn.Module()
    language_model.layers = nn.ModuleList()
    vision_encoder = nn.Module()
    base_model = nn.Module()
    base_model.config = SimpleNamespace(model_type="qwen3_5")
    base_model.model = nn.Module()
    base_model.model.language_model = language_model
    base_model.model.visual = vision_encoder
    wrapped = _PeftWrapper(base_model)

    assert get_language_model(wrapped, override="model.language_model") is language_model
    assert get_vision_encoder(wrapped, override="model.visual") is vision_encoder
