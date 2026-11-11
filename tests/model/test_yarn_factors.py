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

import copy

import pytest
import torch
from transformers import Qwen3_5ForCausalLM
from transformers import Qwen3_5TextConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal


def config():
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=256,
        layer_types=["full_attention"],
        max_position_embeddings=262144,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=1e7,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
        ),
    )


def test_shared_tables_match_transformers():
    from arctic_platform.common.yarn_factors import build_yarn_factors

    cfg = config()
    with pytest.raises(ValueError, match="yarn"):
        build_yarn_factors(cfg, [1.0, 2.0])
    cfg.max_position_embeddings = 524288
    cfg.rope_parameters.update(rope_type="yarn", factor=1.0, original_max_position_embeddings=262144)
    missing_original = copy.deepcopy(cfg)
    del missing_original.rope_parameters["original_max_position_embeddings"]
    with pytest.raises(ValueError, match="original_max_position_embeddings"):
        build_yarn_factors(missing_original, [1.0])
    table = build_yarn_factors(cfg, [2.0, 1.0, 1.25, 1.5, 1.75])
    assert table.factors == (1.0, 2.0, 1.25, 1.5, 1.75)
    native, _ = Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters(cfg)
    torch_assert_equal(table.inv_freq[0], native)
    assert table.attention_scaling[0] == 1
    for f in table.factors[1:]:
        static = copy.deepcopy(cfg)
        static.rope_parameters.update(rope_type="yarn", factor=f, original_max_position_embeddings=262144)
        inv, scale = ROPE_INIT_FUNCTIONS["yarn"](static)
        torch_assert_equal(table.inv_freq[table.slot(f)], inv)
        assert table.attention_scaling[table.slot(f)] == torch.tensor(scale, dtype=torch.float32)
    with pytest.raises(ValueError, match="default"):
        build_yarn_factors(cfg, [2.0])
    for key in ("attention_factor", "mscale", "mscale_all_dim"):
        bad = copy.deepcopy(cfg)
        bad.rope_parameters[key] = 1.0
        with pytest.raises(ValueError, match=key):
            build_yarn_factors(bad, [1.0, 2.0])


def test_loader_packed_factors_and_checkpoint_recompute(tmp_path):
    from arctic_platform.model import ModelSpec
    from arctic_platform.model import build_model

    set_seed(42)
    base = Qwen3_5ForCausalLM(config()).eval()
    base.save_pretrained(tmp_path)

    def load(factors=(), override=None):
        return build_model(
            ModelSpec(
                model_path_or_name=str(tmp_path),
                loader="huggingface",
                dtype="float32",
                attn_implementation="eager",
                yarn_factors=list(factors),
                text_config_overrides=override or {},
            )
        ).model

    native_override = dict(
        max_position_embeddings=524288,
        rope_parameters=dict(rope_type="yarn", factor=1.0, original_max_position_embeddings=262144),
    )
    model = load([1.0, 2.0], native_override).eval()
    ids = torch.tensor([[3, 5, 7, 9, 11, 13]])
    positions = torch.tensor([[250000, 250001, 250002, 250000, 250001, 250002]])
    mask = torch.full((1, 1, 6, 6), float("-inf"))
    mask[:, :, :3, :3] = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
    mask[:, :, 3:, 3:] = mask[:, :, :3, :3]
    factors = torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    kwargs = dict(input_ids=ids, position_ids=positions, attention_mask=mask, use_cache=False)
    native = load().eval()
    static_native = load(override=native_override).eval()
    torch_assert_equal(static_native(**kwargs).logits, native(**kwargs).logits)
    torch_assert_equal(model(**kwargs).logits, native(**kwargs).logits)
    from arctic_platform.rl.processors.pipeline import _engine_forward_kwargs

    routed = _engine_forward_kwargs(dict(**kwargs, yarn_factor=factors), {})
    logits = model(**routed).logits
    for start, f in ((0, 1.0), (3, 2.0)):
        override = (
            {}
            if f == 1
            else {"rope_parameters": dict(rope_type="yarn", factor=f, original_max_position_embeddings=262144)}
        )
        static = load(override=override).eval()
        expected = static(
            input_ids=ids[:, start : start + 3],
            position_ids=positions[:, start : start + 3],
            use_cache=False,
        ).logits
        # Packed and isolated CPU GEMMs differ only by fp32 rounding.
        torch_assert_close(logits[:, start : start + 3], expected, atol=2e-6, rtol=0)
    assert torch.equal(
        model(**kwargs, yarn_factor=torch.ones_like(factors)).logits,
        model(**kwargs).logits,
    )
    # The direct backbone is also the chunked lm-head entry point.
    direct = model.model(**kwargs, yarn_factor=factors).last_hidden_state
    assert torch.equal(model.lm_head(direct), logits)
    model.train()
    model(**kwargs, yarn_factor=factors).logits.square().sum().backward()
    grad = model.model.layers[0].self_attn.q_proj.weight.grad.clone()
    model.zero_grad()
    model.gradient_checkpointing_enable()
    model(**kwargs, yarn_factor=factors).logits.square().sum().backward()
    assert torch.equal(grad, model.model.layers[0].self_attn.q_proj.weight.grad)

    scaled = load(
        [1.0, 1.5, 2.0],
        {"rope_parameters": dict(rope_type="yarn", factor=1.5, original_max_position_embeddings=262144)},
    ).eval()
    assert torch.equal(
        scaled(**kwargs, yarn_factor=torch.full_like(factors, 1.5)).logits,
        scaled(**kwargs).logits,
    )
