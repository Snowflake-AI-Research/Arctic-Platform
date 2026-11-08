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

"""Native Inkling attention and routing with AP's expert-parallel compute."""

from __future__ import annotations

import torch
from transformers.models.inkling.modeling_inkling import InklingForCausalLM
from transformers.models.inkling.modeling_inkling import InklingSharedExperts
from transformers.models.inkling.modeling_inkling import InklingTopkRouter

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs


class InklingMoE(MoE):
    def __init__(self, config):
        super().__init__(
            MoEArgs(
                num_experts=config.n_routed_experts,
                num_shared_experts=0,
                top_k=config.num_experts_per_tok,
                score_before_experts=False,
                load_balance_coeff=None,
                use_grouped_mm=getattr(config, "use_grouped_mm", True),
            ),
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
        )
        del self.router
        self.gate = InklingTopkRouter(config)
        self.shared_experts = InklingSharedExperts(config)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        _, weights, indices, gammas = self.gate(hidden_states)
        flat = hidden_states.reshape(-1, shape[-1])
        if uses_dispatch_ep(self.ep_comm_backend):
            routed = self._run_deepep_routed_experts(flat, indices, weights)
        else:
            order = torch.argsort(indices.reshape(-1), stable=True)
            token_indices = order // indices.shape[-1]
            scores = weights.reshape(-1)[order]
            counts = torch.bincount(indices.reshape(-1), minlength=self.gate.num_experts)
            outputs = self.experts(flat[token_indices], counts, scores)
            routed = torch.zeros_like(flat).index_add(0, token_indices, outputs)
        return routed.view(shape) + self.shared_experts(hidden_states, gammas)


class InklingForCausalLMPrimeRL(InklingForCausalLM, PreTrainedModelPrimeRL):
    def __init__(self, config):
        super().__init__(config)
        for index, layer in enumerate(self.model.layers):
            if config.mlp_layer_types[index] == "sparse":
                layer.mlp = InklingMoE(config)
        for name, module in self.named_modules():
            if name.endswith(("attn_sconv", "mlp_sconv", "k_sconv", "v_sconv")):
                module.float()

    @staticmethod
    def keeps_fp32_parameter(name):
        return any(f".{part}." in name for part in ("attn_sconv", "mlp_sconv", "k_sconv", "v_sconv"))

    @staticmethod
    def vllm_lora_name(name):
        for source, target in (
            ("q_proj", "wq_du"),
            ("k_proj", "wk_dv"),
            ("v_proj", "wv_dv"),
            ("r_proj", "wr_du"),
            ("o_proj", "wo_ud"),
        ):
            name = name.replace(f".self_attn.{source}.", f".attn.{target}.")
        return name

    def init_buffers_post_meta(self):
        for module in self.modules():
            if isinstance(module, InklingMoE):
                module.tokens_per_expert.zero_()

    @classmethod
    def convert_layer_to_hf(cls, state_dict, layer_idx):
        prefix = f"model.layers.{layer_idx}.mlp.experts."
        if prefix + "w1" in state_dict:
            state_dict[prefix + "gate_up_proj"] = torch.cat(
                [state_dict.pop(prefix + "w1"), state_dict.pop(prefix + "w3")], dim=1
            )
            state_dict[prefix + "down_proj"] = state_dict.pop(prefix + "w2")
        return state_dict
