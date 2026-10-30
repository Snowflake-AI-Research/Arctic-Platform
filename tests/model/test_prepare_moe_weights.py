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
"""A node's HF->Prime conversion runs before its ranks start, and the ranks' load then reuses it."""

import pytest
import torch


def test_prepare_moe_weights_converts_without_a_collective_and_the_rank_load_reuses_it(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from arctic_platform.model import conversion_cache_ready
    from arctic_platform.model import moe_model_spec
    from arctic_platform.model import prepare_moe_weights
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen_ds
    from arctic_platform.model.implementations.qwen35 import model_builder
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    source = tmp_path / "Qwen3-MoE-tiny"
    config = transformers.Qwen3MoeConfig(vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                         num_experts=4, num_experts_per_tok=2)  # fmt: skip
    transformers.Qwen3MoeForCausalLM(config).save_pretrained(source)
    options = {"attn": "sdpa", "weight_conversion_cache_dir": str(tmp_path / "cache")}
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node")

    def no_collective(*_args, **_kwargs):
        raise AssertionError("the per-node preparation ran a collective")

    monkeypatch.setattr(torch.distributed, "barrier", no_collective)
    prepare_moe_weights(model_name=str(source), ep_size=2, loader_options=options)

    [cache] = (tmp_path / "cache").glob("*/prime")
    assert conversion_cache_ready(cache)

    # What load_dcp_from_hf does on each rank: it reads the same directory and converts nothing.
    spec = moe_model_spec(model_name=str(source), ep_size=2, loader_options=options)
    model_config = qwen_ds._build_model_config(str(source), 2, 1, spec.dtype, spec.attn_implementation,
                                               Qwen3_5MoeOptions.model_validate(spec.loader_options))  # fmt: skip
    model = qwen_ds._generic_adapter().get_model(model_config, device=torch.device("meta"))

    def no_conversion(*_args, **_kwargs):
        raise AssertionError("the rank converted again")

    monkeypatch.setattr(model_builder, "_write_conversion_cache", no_conversion)
    assert model_builder._convert_snapshot(model, model_config) == (cache, True)
