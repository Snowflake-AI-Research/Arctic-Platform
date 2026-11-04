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

"""``numerics.near_certain_logprobs`` is on by default and reaches every log-prob path from ``loader_options``."""

import types

import pytest
import torch

from arctic_platform.common.near_certain_logprobs import META_KEY
from arctic_platform.common.near_certain_logprobs import mark_model
from arctic_platform.common.near_certain_logprobs import model_enabled
from arctic_platform.model.implementations.moe.layers.lm_head import FusedOutputLinear
from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
from arctic_platform.model.loaders.generic_moe import GenericMoeOptions
from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
from arctic_platform.model.loaders.huggingface import HuggingfaceOptions
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions
from arctic_platform.rl.processors import run_pipeline

VOCAB = 8
TARGET = 3


@pytest.mark.parametrize("options", [HuggingfaceOptions, GenericMoeOptions, GlmMoeDsaOptions, Qwen3_5MoeOptions])
def test_loader_option_defaults_on(options):
    assert options.model_validate({}).near_certain_logprobs is True
    assert options.model_validate({"near_certain_logprobs": False}).near_certain_logprobs is False


class _Causal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(VOCAB, VOCAB, bias=False)
        self.config = types.SimpleNamespace()


@pytest.mark.parametrize("enabled", [False, True])
def test_inject_prime_lm_head_marks_the_model_and_its_chunked_head(enabled):
    model = _Causal()
    inject_prime_lm_head(model, chunk_size=4, near_certain_logprobs=enabled)

    assert isinstance(model.lm_head, FusedOutputLinear)
    assert model.lm_head.near_certain_logprobs is enabled
    assert model_enabled(model) is enabled
    assert model_enabled(types.SimpleNamespace(module=model)) is enabled


def test_an_unmarked_model_counts_as_on():
    assert model_enabled(torch.nn.Linear(2, 2)) is True
    assert model_enabled(types.SimpleNamespace(module=torch.nn.Linear(2, 2))) is True


class _Engine(torch.nn.Module):
    """Returns logits whose label at every position is 20 nats ahead of the rest."""

    global_rank = 0

    def forward(self, input_ids=None, **kwargs):
        logits = torch.zeros(*input_ids.shape, VOCAB)
        logits[..., TARGET] = 20.0 + torch.log(torch.tensor(VOCAB - 1.0))
        return {"logits": logits}


@pytest.mark.parametrize("enabled", [False, True, None])
def test_pipeline_passes_the_engine_mark_to_compute_logprobs(enabled):
    """``None`` leaves the engine unmarked, which counts as on."""
    engine = _Engine()
    if enabled is not None:
        mark_model(engine, enabled)
    input_ids = torch.ones(1, 4, dtype=torch.long)
    labels = torch.full((1, 4), TARGET)
    result = run_pipeline(
        engine,
        (),
        {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        {"labels": labels, "pad_token_id": 0},
        {"post": ["compute_logprobs"]},
        "cpu",
        backward=False,
        pack=False,
        return_tensors=True,
    )
    logprobs = result["batch"]["logprobs"]

    expected = -torch.nn.functional.softplus(torch.tensor(-20.0, dtype=torch.float64))
    if enabled is not False:
        torch.testing.assert_close(logprobs.double(), expected.expand_as(logprobs), rtol=1e-5, atol=0.0)
    else:
        assert (logprobs == 0).all()


def test_an_explicit_meta_value_wins_over_the_engine_mark():
    engine = _Engine()
    mark_model(engine, True)
    input_ids = torch.ones(1, 4, dtype=torch.long)
    result = run_pipeline(
        engine,
        (),
        {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        {"labels": torch.full((1, 4), TARGET), "pad_token_id": 0, META_KEY: False},
        {"post": ["compute_logprobs"]},
        "cpu",
        backward=False,
        pack=False,
        return_tensors=True,
    )
    assert (result["batch"]["logprobs"] == 0).all()
