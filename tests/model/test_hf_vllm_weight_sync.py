# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for HF Qwen3.5 → vLLM weight-sync conversion."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.qwen35.hf_vllm_weight_sync import (
    expected_hf_names_for_text_sync,
)
from arctic_platform.model.implementations.qwen35.hf_vllm_weight_sync import (
    pack_qwen35_gdn_layer,
)
from arctic_platform.model.implementations.qwen35.hf_vllm_weight_sync import (
    to_vllm_sync_weights,
)


def _qwen35_2b_like_layer(layer: int) -> list[tuple[str, torch.Tensor]]:
    prefix = f"model.layers.{layer}"
    return [
        (f"{prefix}.input_layernorm.weight", torch.ones(2048)),
        (f"{prefix}.linear_attn.in_proj_qkv.weight", torch.arange(6144 * 2048).reshape(6144, 2048).float()),
        (f"{prefix}.linear_attn.in_proj_z.weight", torch.arange(2048 * 2048).reshape(2048, 2048).float() + 1),
        (f"{prefix}.linear_attn.in_proj_b.weight", torch.arange(16 * 2048).reshape(16, 2048).float() + 2),
        (f"{prefix}.linear_attn.in_proj_a.weight", torch.arange(16 * 2048).reshape(16, 2048).float() + 3),
        (f"{prefix}.linear_attn.conv1d.weight", torch.arange(2048 * 4).reshape(2048, 4).float()),
        (f"{prefix}.linear_attn.out_proj.weight", torch.ones(2048, 2048)),
        (f"{prefix}.mlp.down_proj.weight", torch.ones(2048, 2048)),
    ]


def test_qwen35_packs_gdn_prefix_and_conv1d():
    embed = torch.ones(100, 2048)
    layer_weights = _qwen35_2b_like_layer(0)
    by_hf = dict(layer_weights)
    weights = [
        ("model.embed_tokens.weight", embed),
        *layer_weights,
        ("model.visual.patch_embed.weight", torch.ones(8)),
        ("mtp.layers.0.weight", torch.ones(4)),
    ]
    converted = dict(to_vllm_sync_weights(weights))

    qkv = by_hf["model.layers.0.linear_attn.in_proj_qkv.weight"]
    z = by_hf["model.layers.0.linear_attn.in_proj_z.weight"]
    b = by_hf["model.layers.0.linear_attn.in_proj_b.weight"]
    a = by_hf["model.layers.0.linear_attn.in_proj_a.weight"]
    conv = by_hf["model.layers.0.linear_attn.conv1d.weight"]

    qkvz = converted["language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"]
    ba = converted["language_model.model.layers.0.linear_attn.in_proj_ba.weight"]
    conv_out = converted["language_model.model.layers.0.linear_attn.conv1d.weight"]

    assert qkvz.shape == (8192, 2048)
    assert torch.equal(qkvz[:6144], qkv)
    assert torch.equal(qkvz[6144:], z)
    assert ba.shape == (32, 2048)
    assert torch.equal(ba[:16], b)
    assert torch.equal(ba[16:], a)
    assert conv_out.shape == (2048, 1, 4)
    assert torch.equal(conv_out.squeeze(1), conv)

    assert "language_model.model.embed_tokens.weight" in converted
    assert "language_model.lm_head.weight" not in converted
    assert "visual.patch_embed.weight" not in converted
    assert "mtp.layers.0.weight" not in converted
    assert "model.layers.0.linear_attn.in_proj_qkv.weight" not in converted


def test_qwen35_vlm_language_model_prefix():
    weights = [
        ("model.language_model.embed_tokens.weight", torch.ones(4, 8)),
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            torch.ones(6, 8),
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            torch.ones(2, 8),
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            torch.ones(1, 8),
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            torch.ones(1, 8),
        ),
        ("lm_head.weight", torch.ones(4, 8)),
    ]
    converted_list = to_vllm_sync_weights(weights)
    converted = dict(converted_list)
    assert "language_model.model.embed_tokens.weight" in converted
    assert "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight" in converted
    assert converted["language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"].shape == (8, 8)
    assert "language_model.lm_head.weight" not in converted


def test_qwen3_passthrough_without_unpacked_gdn():
    weights = [
        ("model.embed_tokens.weight", torch.ones(4, 8)),
        ("model.layers.0.self_attn.q_proj.weight", torch.ones(8, 8)),
        ("model.layers.0.mlp.down_proj.weight", torch.ones(8, 8)),
        ("lm_head.weight", torch.ones(4, 8)),
    ]
    converted = to_vllm_sync_weights(weights)
    assert [name for name, _ in converted] == [name for name, _ in weights]
    for (_, src), (_, dst) in zip(weights, converted):
        assert src is dst


def test_pack_qwen35_gdn_layer_matches_vllm_cat_order():
    prefix = "model.layers.1"
    qkv = torch.arange(12).reshape(6, 2).float()
    z = torch.arange(4).reshape(2, 2).float() + 100
    b = torch.arange(2).reshape(1, 2).float() + 200
    a = torch.arange(2).reshape(1, 2).float() + 300
    layer_sd = {
        f"{prefix}.linear_attn.in_proj_qkv.weight": qkv,
        f"{prefix}.linear_attn.in_proj_z.weight": z,
        f"{prefix}.linear_attn.in_proj_b.weight": b,
        f"{prefix}.linear_attn.in_proj_a.weight": a,
        f"{prefix}.linear_attn.out_proj.weight": torch.ones(2, 2),
    }
    pack_qwen35_gdn_layer(layer_sd, prefix)
    assert f"{prefix}.linear_attn.in_proj_qkv.weight" not in layer_sd
    assert torch.equal(
        layer_sd[f"{prefix}.linear_attn.in_proj_qkvz.weight"],
        torch.cat([qkv, z], dim=0),
    )
    assert torch.equal(
        layer_sd[f"{prefix}.linear_attn.in_proj_ba.weight"],
        torch.cat([b, a], dim=0),
    )


def test_expected_hf_names_drop_missing_visual_keep_shipped():
    expected = {
        "language_model.model.embed_tokens.weight",
        "visual.patch_embed.weight",
        "mtp.layers.0.weight",
    }
    sender = {"language_model.model.embed_tokens.weight"}
    filtered = expected_hf_names_for_text_sync(expected, sender)
    assert filtered == {"language_model.model.embed_tokens.weight"}

    sender_with_vision = sender | {"visual.patch_embed.weight"}
    filtered_vlm = expected_hf_names_for_text_sync(expected, sender_with_vision)
    assert "visual.patch_embed.weight" in filtered_vlm
    assert "mtp.layers.0.weight" not in filtered_vlm


def test_text_only_extension_allows_missing_visual():
    from arctic_platform.common.weight_sync_extension import TextOnlyWeightSyncExtension

    ext = TextOnlyWeightSyncExtension()
    expected = {
        "language_model.model.embed_tokens.weight",
        "visual.patch_embed.weight",
        "mtp.layers.0.weight",
    }

    import arctic_inference.server.weight_sync.utils as ws_utils

    orig = ws_utils.compute_expected_hf_param_names
    ws_utils.compute_expected_hf_param_names = lambda model: set(expected)
    try:
        ext._validate_weight_sync_names(
            object(),
            ["language_model.model.embed_tokens.weight"],
            context="test",
        )
    finally:
        ws_utils.compute_expected_hf_param_names = orig


def test_text_only_extension_still_rejects_unexpected_lm_name():
    from arctic_platform.common.weight_sync_extension import TextOnlyWeightSyncExtension

    ext = TextOnlyWeightSyncExtension()
    import arctic_inference.server.weight_sync.utils as ws_utils

    orig = ws_utils.compute_expected_hf_param_names
    ws_utils.compute_expected_hf_param_names = lambda model: {
        "language_model.model.embed_tokens.weight",
        "visual.patch_embed.weight",
    }
    try:
        with pytest.raises(RuntimeError, match="does NOT expect"):
            ext._validate_weight_sync_names(
                object(),
                [
                    "language_model.model.embed_tokens.weight",
                    "language_model.lm_head.weight",
                ],
                context="test",
            )
    finally:
        ws_utils.compute_expected_hf_param_names = orig


def test_build_model_config_registers_enginecore_extension():
    from arctic_platform.common.utils.server_models import build_model_config
    from arctic_platform.common.weight_sync_extension import WORKER_EXTENSION_CLS

    cfg = build_model_config("Qwen/Qwen3.5-2B", {})
    assert cfg.extra_engine_kwargs["worker_extension_cls"] == WORKER_EXTENSION_CLS
    kwargs = cfg.to_engine_kwargs()
    assert kwargs["worker_extension_cls"] == WORKER_EXTENSION_CLS
