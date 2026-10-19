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

"""Small CPU controls for independent QSA loading and frozen-table placement."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.correctness.reference.hf_single_gpu import build_engine
from arctic_platform.correctness.reference.qwen4_reference import load_qwen4_reference


class TinyQwen4(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.ple = nn.Module()
        layer.ple.ple_embedding = nn.Module()
        layer.ple.ple_embedding.ngram_embedding = nn.Embedding(5, 2)
        layer.ple.key_proj = nn.Linear(2, 2, bias=False)
        layer.self_attn = nn.Module()
        layer.self_attn.indexer = nn.Linear(2, 2, bias=False)
        layer.self_attn.q_proj = nn.Linear(2, 2, bias=False)
        layer.register_buffer("positions", torch.arange(2), persistent=False)
        self.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(2, 5, bias=False)
        self.config = SimpleNamespace(hidden_size=2)
        self.checkpointing = False

    def get_decoder(self):
        return self

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self):
        self.checkpointing = True


def loader(model, info=None):
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return model, info or {}

    return SimpleNamespace(from_pretrained=from_pretrained), calls


@pytest.mark.parametrize("attention,expected", [("qsa_flex", "sdpa"), ("sdpa", "sdpa"), ("eager", "eager")])
def test_upstream_backend_and_cpu_checkpoint_load(attention, expected):
    model = TinyQwen4()
    cls, calls = loader(model)
    result = load_qwen4_reference(cls, "fixture", dtype=torch.bfloat16, attn_implementation=attention, device="cpu")
    assert result is model
    assert calls == [
        ("fixture", {"dtype": torch.bfloat16, "attn_implementation": expected, "output_loading_info": True})
    ]
    layer = model.layers[0]
    assert not layer.ple.ple_embedding.ngram_embedding.weight.requires_grad
    assert not layer.self_attn.indexer.weight.requires_grad
    assert layer.ple.key_proj.weight.requires_grad
    assert layer.self_attn.q_proj.weight.requires_grad


def test_table_never_reaches_accelerator_placement(monkeypatch):
    model = TinyQwen4()
    table = model.layers[0].ple.ple_embedding.ngram_embedding.weight
    original = table.detach().clone()
    parameter_ids = {id(parameter) for parameter in model.parameters()} - {id(table)}
    moved = []
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            assert tensor is not table
            moved.append(id(tensor))
            return original_to(tensor, device="cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    cls, _ = loader(model)
    load_qwen4_reference(cls, "fixture", dtype=torch.bfloat16, attn_implementation="qsa_flex", device="cuda")
    assert parameter_ids <= set(moved)
    assert id(model.layers[0].positions) in moved
    assert model.layers[0].ple.ple_embedding.ngram_embedding.weight is table
    assert table.device.type == "cpu"
    assert torch.equal(table, original)


@pytest.mark.parametrize("attention", ["flash_attention_3", "flash_attention_4", "flex_attention", "unknown"])
def test_unsupported_attention_rejected_before_loading(attention):
    cls, calls = loader(TinyQwen4())
    with pytest.raises(ValueError, match="upstream QSA"):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation=attention, device="cpu")
    assert calls == []


@pytest.mark.parametrize("key", ["missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs"])
def test_incomplete_checkpoint_rejected_before_placement(key, monkeypatch):
    model = TinyQwen4()
    cls, _ = loader(model, {key: ["ngram_embedding.weight"]})
    monkeypatch.setattr(model, "_apply", lambda *args, **kwargs: pytest.fail("placement reached"))
    with pytest.raises(ValueError, match=key):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation="qsa_flex", device="cpu")


def test_meta_checkpoint_rejected_before_placement():
    model = TinyQwen4().to("meta")
    cls, _ = loader(model)
    with pytest.raises(ValueError, match="load on CPU"):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation="qsa_flex", device="cpu")


@pytest.mark.parametrize("missing", ["ple", "self_attn"])
def test_missing_layer_period_rejected(missing):
    model = TinyQwen4()
    delattr(model.layers[0], missing)
    cls, _ = loader(model)
    with pytest.raises(ValueError, match="complete PLE/QSA"):
        load_qwen4_reference(cls, "fixture", dtype=None, attn_implementation="qsa_flex", device="cpu")


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text", "qwen3"])
def test_actual_build_engine_family_dispatch(monkeypatch, model_type):
    model = TinyQwen4()
    calls = []
    cfg = SimpleNamespace(model_type=model_type, architectures=["Tiny"], hidden_size=2)

    def from_pretrained(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("output_loading_info"):
            return model, {}
        return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: cfg),
            Tiny=SimpleNamespace(from_pretrained=from_pretrained),
        ),
    )
    engine = build_engine("fixture", attn_implementation="qsa_flex", device="cpu")
    assert engine.model is model
    assert model.checkpointing and model.training
    if model_type.startswith("qwen4"):
        assert calls[0]["attn_implementation"] == "sdpa"
        assert calls[0]["output_loading_info"] is True
        assert not model.layers[0].ple.ple_embedding.ngram_embedding.weight.requires_grad
    else:
        assert calls[0] == {"dtype": torch.bfloat16, "attn_implementation": "qsa_flex"}
        assert model.layers[0].ple.ple_embedding.ngram_embedding.weight.requires_grad
