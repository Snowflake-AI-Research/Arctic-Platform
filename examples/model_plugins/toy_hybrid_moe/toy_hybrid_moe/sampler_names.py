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

"""Where each Hugging Face tensor lands in the vLLM model, which fuses projections and stacks experts.

``vllm_model.ToyHybridMoEForCausalLM.load_weights`` uses this both for the checkpoint at startup and for every
weight sync of an RL step. It imports nothing from vLLM, so CPU tests can check that every tensor the trainer
streams has exactly one destination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FUSED: tuple[tuple[str, str, str | int], ...] = (
    (".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", "q"),
    (".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight", "k"),
    (".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight", "v"),
    (".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.gate_up_proj.weight", 0),
    (".mlp.shared_expert.up_proj.weight", ".mlp.shared_expert.gate_up_proj.weight", 1),
    (".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", 0),
    (".mlp.up_proj.weight", ".mlp.gate_up_proj.weight", 1),
    (".recurrence.in_proj.weight", ".recurrence.in_gate_proj.weight", 0),
    (".recurrence.gate_proj.weight", ".recurrence.in_gate_proj.weight", 1),
    (".conv.in_proj.weight", ".conv.in_gate_proj.weight", 0),
    (".conv.gate_proj.weight", ".conv.in_gate_proj.weight", 1),
)
_EXPERT = re.compile(r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
# vLLM 0.33's fused MoE keeps its stacked expert weights in a ``routed_experts`` submodule.
_EXPERT_SHARD = {
    "gate_proj": ("routed_experts.w13_weight", "w1"),
    "up_proj": ("routed_experts.w13_weight", "w3"),
    "down_proj": ("routed_experts.w2_weight", "w2"),
}
TRAINER_ONLY = re.compile(r"\.mlp\.stats\.")


@dataclass(frozen=True)
class SamplerTarget:
    param: str
    shard: str | int | None = None
    expert: int | None = None


def sampler_target(name: str) -> SamplerTarget:
    """The vLLM parameter, fused shard and expert index a Hugging Face tensor loads into."""
    if TRAINER_ONLY.search(name):
        raise ValueError(f"{name} is trainer-only state and must not reach the sampler")
    match = _EXPERT.match(name)
    if match:
        param, shard = _EXPERT_SHARD[match["proj"]]
        return SamplerTarget(f"{match['prefix']}.{param}", shard, int(match["expert"]))
    for suffix, fused, shard_id in _FUSED:
        if name.endswith(suffix):
            return SamplerTarget(name[: -len(suffix)] + fused, shard_id)
    return SamplerTarget(name)
