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

"""Inkling text layers with AP's expert-parallel compute."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.inkling.modeling_inkling import InklingForCausalLM
from transformers.models.inkling.modeling_inkling import InklingMLP as NativeInklingMLP
from transformers.models.inkling.modeling_inkling import InklingSharedExperts as NativeInklingSharedExperts
from transformers.models.inkling.modeling_inkling import InklingTopkRouter as NativeInklingTopkRouter

from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs


class InklingTopkRouter(NativeInklingTopkRouter):
    def __init__(self, config):
        super().__init__(config)
        # DeepSpeed's parameter cast guard preserves the constant selection bias too.
        bias = self.e_score_correction_bias
        del self.e_score_correction_bias
        self.e_score_correction_bias = torch.nn.Parameter(bias.float(), requires_grad=False)
        self.global_scale.data = self.global_scale.data.float()

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        routed, shared = logits.split([self.num_experts, self.n_shared_experts], dim=-1)
        indices = torch.topk(routed.sigmoid() + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False)[1]
        selected = torch.cat([routed.gather(-1, indices), shared], dim=-1)
        log_scores = F.logsigmoid(selected)
        weights = (log_scores - torch.logsumexp(log_scores, dim=-1, keepdim=True)).exp()
        weights = weights * self.route_scale * self.global_scale
        return routed, weights[:, : self.top_k].contiguous(), indices, weights[:, self.top_k :].contiguous()


class InklingSharedExperts(NativeInklingSharedExperts):
    def forward(self, hidden_states, gammas):
        shape = hidden_states.shape
        inputs = hidden_states.reshape(1, -1, shape[-1]).expand(self.n_shared_experts, -1, -1)
        gammas = gammas.reshape(-1, self.n_shared_experts, 1).transpose(0, 1)
        gate = torch.bmm(inputs, self.gate_proj.transpose(1, 2))
        up = torch.bmm(inputs, self.up_proj.transpose(1, 2))
        activated = (self.act_fn(gate.float()) * up.float() * gammas.float()).to(inputs.dtype)
        down = torch.bmm(activated, self.down_proj.transpose(1, 2))
        return down.float().sum(dim=0).to(hidden_states.dtype).view(shape)


class InklingMLP(NativeInklingMLP):
    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        activated = (self.act_fn(gate.float()) * up.float()).to(hidden_states.dtype)
        outputs = self.down_proj(activated)
        return (outputs.float() * self.global_scale).to(hidden_states.dtype)


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
                fp32_routed_weighting=True,
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
            else:
                layer.mlp = InklingMLP(config)
        for name, module in self.named_modules():
            if name.endswith(("attn_sconv", "mlp_sconv", "k_sconv", "v_sconv")):
                module.float()
        for name, parameter in self.named_parameters():
            if self.keeps_fp32_parameter(name):
                parameter.data = parameter.data.float()

    @staticmethod
    def keeps_fp32_parameter(name):
        return name.endswith((".gate.global_scale", ".gate.e_score_correction_bias")) or any(
            f".{part}." in name for part in ("attn_sconv", "mlp_sconv", "k_sconv", "v_sconv")
        )

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
