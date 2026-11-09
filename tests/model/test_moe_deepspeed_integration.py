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

import torch
from torch import nn

from arctic_platform.model.implementations.moe import deepspeed_integration as common


def test_shared_setup_runs_family_hooks_in_order(monkeypatch):
    events = []
    model = nn.Module()

    def record(name, result=None):
        def call(*_args, **_kwargs):
            events.append(name)
            return result

        return call

    monkeypatch.setattr(common, "apply_ep_with_mesh", record("ep"))
    monkeypatch.setattr(common, "convert_dtensors_to_local", record("convert", 0))
    monkeypatch.setattr(common, "enable_tiled_mlp", record("tiled"))
    adapter = common.MoEDeepSpeedAdapter(
        dtype_map={"bfloat16": torch.bfloat16},
        get_model=record("model", model),
        configure_moe_ep_backend=record("moe_backend"),
        configure_family_backend=record("family_backend"),
        inject_lm_head=record("lm_head"),
        apply_sequence_parallelism=record("sp"),
        apply_ac=record("ac"),
        load_dcp_from_hf=record("weights"),
        reset_runtime_moe_buffers=record("reset"),
        shared_expert_type=nn.Linear,
        shared_expert_forward=lambda module, value: module(value),
        build_model_config=lambda *_args: None,
    )
    config = SimpleNamespace(
        optimization_dtype="bfloat16",
        fused_lm_head_token_chunk_size="disabled",
        fp32_lm_head=False,
        ac=object(),
    )

    common.setup_model_local_no_train(
        adapter,
        config,
        SimpleNamespace(ep_enabled=True),
        object(),
        sp_size=2,
        sp_group=object(),
    )

    assert events == [
        "model",
        "moe_backend",
        "family_backend",
        "lm_head",
        "ep",
        "sp",
        "ac",
        "weights",
        "reset",
        "convert",
        "tiled",
    ]


def test_shared_expert_tagging(monkeypatch):
    class DummyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.Linear(2, 2)

    monkeypatch.setattr(common, "MoE", DummyMoE)
    monkeypatch.setattr(common, "LatentMoE", type("DummyLatentMoE", (nn.Module,), {}))
    model = nn.Sequential(DummyMoE())

    count = common.tag_expert_params_for_deepspeed(model, "ep_size_2")

    assert count == 2
    assert all(
        parameter.group_name == "ep_size_2" and parameter.allreduce is False
        for parameter in model[0].experts.parameters()
    )


def test_family_defaults_remain_distinct():
    from arctic_platform.model.implementations.glm52 import deepspeed_integration as glm
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    qwen_config = qwen._build_model_config("qwen", 2, 4, "bfloat16", "flash_attention_3", Qwen3_5MoeOptions())
    glm_config = glm._build_model_config("glm", 2, 4, "bfloat16", "flash_attention_2", GlmMoeDsaOptions())

    assert (qwen_config.attn, qwen_config.moe_use_grouped_mm) == (
        "flash_attention_3",
        True,
    )
    assert (
        glm_config.attn,
        glm_config.moe_use_grouped_mm,
        glm_config.sparse_mla_backend,
    ) == ("flash_attention_2", False, "ref")


def test_qwen_adapter_keeps_sp_and_vllm_export(monkeypatch):
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    calls = []
    monkeypatch.setattr(
        qwen,
        "apply_sequence_parallelism",
        lambda model, size, group: calls.append((model, size, group)),
    )
    model, group = nn.Module(), object()
    qwen._apply_sequence_parallelism(model, 2, group)

    assert calls == [(model, 2, group)]
    assert qwen._adapter().vllm_weight_export is qwen._build_iter_full_vllm_weights


def test_selective_checkpointing_falls_back_for_glm():
    from arctic_platform.model.implementations.moe.layers.checkpointing import (
        supports_selective_activation_checkpointing,
    )

    layer_type = type(
        "DecoderLayer",
        (nn.Module,),
        {"__module__": "arctic_platform.model.implementations.glm52.models.decoder"},
    )

    assert supports_selective_activation_checkpointing(layer_type()) is False
