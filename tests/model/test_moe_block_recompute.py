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
"""Full activation checkpointing must recompute the whole block.

The EP backends finish a MoE forward with ``sync_combine``, after the block's last saved tensor. An early-stopped
recompute never reaches it, so the pending combines and dispatch handles of every recomputed layer stay alive.
"""

import pytest
import torch
from torch import nn

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.implementations.glm52 import model_builder as glm52_builder
from arctic_platform.model.implementations.moe.distributed.combine_replay import CombineReplay
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs
from arctic_platform.model.implementations.qwen35 import model_builder as qwen35_builder


class _Block(nn.Module):
    def __init__(self, tail_runs: list[str]):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.tail_runs = tail_runs

    def forward(self, hidden_states):
        out = self.proj(hidden_states)  # the block's last saved tensor
        self.tail_runs.append("tail")  # stands in for sync_combine
        return out + 1


@pytest.mark.parametrize("apply_ac", [glm52_builder.apply_ac, qwen35_builder.apply_ac], ids=["glm52", "qwen35"])
def test_full_checkpointing_recomputes_past_the_last_saved_tensor(apply_ac):
    tail_runs = []
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([_Block(tail_runs)])
    apply_ac(model, ActivationCheckpointConfig(mode="full", freq=1, router_replay_recompute=False))

    hidden_states = torch.randn(2, 4, requires_grad=True)
    model.model.layers[0](hidden_states).sum().backward()

    assert tail_runs == ["tail", "tail"]


@pytest.mark.parametrize("apply_ac", [glm52_builder.apply_ac, qwen35_builder.apply_ac], ids=["glm52", "qwen35"])
def test_listed_layers_are_the_only_checkpointed_blocks(apply_ac):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([_Block([]) for _ in range(4)])
    apply_ac(model, ActivationCheckpointConfig(mode="full", layers=[1, 3], router_replay_recompute=False))

    wrapped = [type(block).__name__ == "CheckpointWrapper" for block in model.model.layers]
    assert wrapped == [False, True, False, True]
    with pytest.raises(ValueError, match="past the model"):
        apply_ac(model, ActivationCheckpointConfig(mode="full", layers=[4], router_replay_recompute=False))


def _language_model_with_moe() -> tuple[nn.Module, MoE]:
    moe = MoE(
        MoEArgs(num_experts=2, num_shared_experts=0, top_k=1, use_grouped_mm=False, load_balance_coeff=None),
        dim=4,
        hidden_dim=8,
    )
    block = nn.Module()
    block.mlp = moe
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([block])
    return model, moe


@pytest.mark.parametrize(
    ("builder", "config_cls"),
    [
        (qwen35_builder, qwen35_builder.ModelConfig),
        (glm52_builder, glm52_builder.ModelConfig),
    ],
    ids=["qwen35", "glm52"],
)
def test_sonicmoe_option_reaches_each_moe(monkeypatch, builder, config_cls):
    from types import SimpleNamespace

    import arctic_platform.model.implementations.moe.distributed.ep_backend as ep_backend

    def fake_ep_module(_backend):
        return SimpleNamespace(configure_num_sms=lambda _num_sms: None)

    monkeypatch.setattr(ep_backend, "get_ep_comm_module", fake_ep_module)
    monkeypatch.setattr(builder, "get_ep_comm_module", fake_ep_module, raising=False)
    model, moe = _language_model_with_moe()
    assert moe.use_sonicmoe is True

    builder.configure_moe_ep_backend(model, config_cls(moe_use_sonicmoe=False))

    assert moe.use_sonicmoe is False


@pytest.mark.parametrize("apply_ac", [glm52_builder.apply_ac, qwen35_builder.apply_ac], ids=["glm52", "qwen35"])
def test_full_checkpointing_installs_combine_replay_unless_disabled(apply_ac):
    model, moe = _language_model_with_moe()
    apply_ac(model, ActivationCheckpointConfig(mode="full", router_replay_recompute=False))
    assert isinstance(moe.combine_replay, CombineReplay)

    model, moe = _language_model_with_moe()
    apply_ac(model, ActivationCheckpointConfig(mode="full", combine_replay=False, router_replay_recompute=False))
    assert moe.combine_replay is None
