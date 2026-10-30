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
"""``save_exported_weights`` writes the dict export's files without holding every exported tensor at once."""

import functools
import hashlib
import weakref

import pytest
import torch
from huggingface_hub import split_torch_state_dict_into_shards

from arctic_platform.model import save_exported_state_dict
from arctic_platform.model import save_exported_weights
from arctic_platform.model.implementations.glm53.converting_glm5_next import _layer_prefix
from arctic_platform.model.implementations.glm53.converting_glm5_next import convert_prime_layer_to_hf
from arctic_platform.model.implementations.moe import weights as moe_weights

EXPERTS_PER_LAYER = (4, 4, 8, 4, 6, 4, 4, 8, 4, 4, 6, 4)


class _LiveBytes:
    """Bytes of producer-allocated storages that are still alive, and the most ever alive at once."""

    def __init__(self):
        self.allocated = 0
        self.live = 0
        self.peak = 0

    def track(self, tensor):
        storage = tensor.untyped_storage()
        self.allocated += storage.nbytes()
        self.live += storage.nbytes()
        self.peak = max(self.peak, self.live)
        weakref.finalize(storage, self._release, storage.nbytes())
        return tensor

    def _release(self, nbytes):
        self.live -= nbytes


def _glm_export(counter, layer_bytes=None, *, edge_cases=False):
    """GLM-5.3's ``build_iter_full_hf_weights`` stream: per layer, EP-gathered stacked expert buffers on the host,
    split by the production converter into per-expert views, beside replicated tensors copied off the device."""
    generator = torch.Generator().manual_seed(0)

    def rand(*shape):
        return counter.track(torch.randn(*shape, generator=generator).to(torch.bfloat16))

    yield "lm_head.weight", rand(32, 8)
    yield "model.language_model.embed_tokens.weight", rand(32, 8)
    for index, experts in enumerate(EXPERTS_PER_LAYER):
        before = counter.allocated
        prefix = _layer_prefix(index)
        layer_state = {}
        layer_state[f"{prefix}.input_layernorm.weight"] = rand(8)
        layer_state[f"{prefix}.self_attn.o_proj.weight"] = rand(8, 8)
        layer_state[f"{prefix}.mlp.router.gate.weight"] = rand(experts, 8)
        layer_state[f"{prefix}.mlp.experts.w1"] = rand(experts, 12, 8)
        layer_state[f"{prefix}.mlp.experts.w2"] = rand(experts, 8, 12)
        layer_state[f"{prefix}.mlp.experts.w3"] = rand(experts, 12, 8)
        if layer_bytes is not None:
            layer_bytes.append(counter.allocated - before)
        convert_prime_layer_to_hf(layer_state, index)
        yield from layer_state.items()
    if edge_cases:
        yield "edge.partial_view", rand(4, 8)[:2]
        yield "edge.transposed", rand(8, 4).t()
        yield "edge.empty_a", torch.empty(0, dtype=torch.bfloat16)
        yield "edge.empty_b", torch.empty(0, dtype=torch.bfloat16)
    yield "model.language_model.norm.weight", rand(8)


def _files(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir())}


def _limit_shard_size(monkeypatch, max_shard_size):
    if max_shard_size is not None:
        monkeypatch.setattr(
            moe_weights,
            "split_torch_state_dict_into_shards",
            functools.partial(split_torch_state_dict_into_shards, max_shard_size=max_shard_size),
        )


@pytest.mark.parametrize("edge_cases", [False, True])
@pytest.mark.parametrize("max_shard_size", [1000, 2500, None])
def test_streamed_export_writes_the_dict_export_files(tmp_path, monkeypatch, max_shard_size, edge_cases):
    _limit_shard_size(monkeypatch, max_shard_size)
    # The weights-only checkpointer collects the stream this way before ``save_exported_state_dict``.
    state_dict = {name: tensor.detach().cpu() for name, tensor in _glm_export(_LiveBytes(), edge_cases=edge_cases)}
    save_exported_state_dict(state_dict, str(tmp_path / "dict"))
    save_exported_weights(_glm_export(_LiveBytes(), edge_cases=edge_cases), str(tmp_path / "stream"))

    expected = _files(tmp_path / "dict")
    assert _files(tmp_path / "stream") == expected
    shards = [name for name in expected if name.endswith(".safetensors")]
    if max_shard_size is None:
        assert shards == ["model.safetensors"]
    else:
        assert len(shards) > 1


def test_streamed_export_of_nothing_writes_the_dict_export_files(tmp_path):
    save_exported_state_dict({}, str(tmp_path / "dict"))
    save_exported_weights(iter(()), str(tmp_path / "stream"))
    assert _files(tmp_path / "stream") == _files(tmp_path / "dict")


def test_streamed_export_holds_one_layer_not_the_model(tmp_path, monkeypatch):
    max_shard_size = 1000
    _limit_shard_size(monkeypatch, max_shard_size)

    collected = _LiveBytes()
    state_dict = {name: tensor.detach().cpu() for name, tensor in _glm_export(collected)}
    save_exported_state_dict(state_dict, str(tmp_path / "dict"))
    assert collected.peak == collected.allocated

    streamed, layer_bytes = _LiveBytes(), []
    save_exported_weights(_glm_export(streamed, layer_bytes), str(tmp_path / "stream"))
    assert streamed.allocated == collected.allocated
    # The open shard (at most ``max_shard_size``) carries across layers; every closed shard is written when its
    # layer's views tile their storages, before the producer allocates the next layer.
    assert streamed.peak <= max(layer_bytes) + max_shard_size < collected.allocated // 3
    assert streamed.live == 0
