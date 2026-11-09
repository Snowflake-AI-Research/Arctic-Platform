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
from torch.utils.checkpoint import checkpoint
from transformers.models.inkling.modeling_inkling import InklingCausalLMOutputWithPast
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
        self.token_chunk_size = getattr(config, "inkling_mlp_token_chunk_size", None) or 2048
        # DeepSpeed's parameter cast guard preserves the constant selection bias too.
        bias = self.e_score_correction_bias
        del self.e_score_correction_bias
        self.e_score_correction_bias = torch.nn.Parameter(bias.float(), requires_grad=False)
        self.global_scale.data = self.global_scale.data.float()

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        chunks = []
        for shard in flat.split(self.token_chunk_size):
            tokens = shard.shape[0]
            # Fix GEMM geometry so SP does not change FP32 routing weights.
            padded = F.pad(shard, (0, 0, 0, self.token_chunk_size - tokens))
            chunks.append(tuple(value[:tokens] for value in self._forward(padded)))
        return tuple(torch.cat(parts, dim=0) for parts in zip(*chunks, strict=True))

    def _forward(self, flat):
        logits = F.linear(flat.float(), self.weight.float())
        routed, shared = logits.split([self.num_experts, self.n_shared_experts], dim=-1)
        indices = torch.topk(routed.sigmoid() + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False)[1]
        selected = torch.cat([routed.gather(-1, indices), shared], dim=-1)
        log_scores = F.logsigmoid(selected)
        weights = (log_scores - torch.logsumexp(log_scores, dim=-1, keepdim=True)).exp()
        weights = weights * self.route_scale * self.global_scale
        return (routed, weights[:, : self.top_k].contiguous(), indices, weights[:, self.top_k :].contiguous())


class InklingSharedExperts(NativeInklingSharedExperts):
    def __init__(self, config):
        super().__init__(config)
        self.token_chunk_size = getattr(config, "inkling_mlp_token_chunk_size", None)

    def forward(self, hidden_states, gammas):
        shape = hidden_states.shape
        if self.token_chunk_size is not None and hidden_states.numel() // shape[-1] > self.token_chunk_size:
            flat = hidden_states.reshape(-1, shape[-1])
            gammas = gammas.reshape(-1, self.n_shared_experts)
            chunks = []
            for start in range(0, flat.shape[0], self.token_chunk_size):
                args = (flat[start : start + self.token_chunk_size], gammas[start : start + self.token_chunk_size])
                output = (
                    checkpoint(self._forward, *args, use_reentrant=False)
                    if torch.is_grad_enabled()
                    else self._forward(*args)
                )
                chunks.append(output)
            return torch.cat(chunks).view(shape)
        return self._forward(hidden_states, gammas)

    def _forward(self, hidden_states, gammas):
        shape = hidden_states.shape
        inputs = hidden_states.reshape(1, -1, shape[-1]).expand(self.n_shared_experts, -1, -1)
        gammas = gammas.reshape(-1, self.n_shared_experts, 1).transpose(0, 1)
        gate = torch.bmm(inputs, self.gate_proj.transpose(1, 2))
        up = torch.bmm(inputs, self.up_proj.transpose(1, 2))
        activated = (self.act_fn(gate.float()) * up.float() * gammas.float()).to(inputs.dtype)
        down = torch.bmm(activated, self.down_proj.transpose(1, 2))
        return down.float().sum(dim=0).to(hidden_states.dtype).view(shape)


class InklingMLP(NativeInklingMLP):
    def __init__(self, config):
        super().__init__(config)
        self.token_chunk_size = getattr(config, "inkling_mlp_token_chunk_size", None)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        if self.token_chunk_size is not None and hidden_states.numel() // shape[-1] > self.token_chunk_size:
            flat = hidden_states.reshape(-1, shape[-1])
            chunks = []
            for shard in flat.split(self.token_chunk_size):
                output = (
                    checkpoint(self._forward, shard, use_reentrant=False)
                    if torch.is_grad_enabled()
                    else self._forward(shard)
                )
                chunks.append(output)
            return torch.cat(chunks).view(shape)
        return self._forward(hidden_states)

    def _forward(self, hidden_states):
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
        # UCCL combine events retain their receive buffers until the stream wait.
        self.sync_combine_per_chunk = True
        self.checkpoint_dispatch_chunks = True

    def _run_local_routed_experts(self, hidden_states, counts, scores=None):
        if self.deepep_token_chunk_size is not None and torch.is_grad_enabled():
            return checkpoint(self.experts, hidden_states, counts, scores, use_reentrant=False)
        return self.experts(hidden_states, counts, scores)

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        dss_compute_logprobs=False,
        temperature=None,
        action_masks=None,
        group_token_ids=None,
        **kwargs,
    ):
        chunk_size = getattr(self, "_inkling_lm_head_token_chunk_size", None)
        if chunk_size is None or (labels is None and not dss_compute_logprobs):
            if dss_compute_logprobs:
                raise ValueError("Inkling chunked logprobs require an enabled chunked head and explicit labels")
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        if labels is None:
            raise ValueError("Inkling chunked logprobs require explicit next-token labels")
        from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
        from arctic_platform.model.implementations.gpu.lm_head import slice_action_masks_for_logits_to_keep
        from arctic_platform.model.implementations.gpu.lm_head import slice_temperature_for_logits_to_keep

        num_items = kwargs.pop("num_items_in_batch", None)
        shifted = kwargs.pop("shift_labels", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden = outputs.last_hidden_state / self.config.logits_mup_width_multiplier
        if not dss_compute_logprobs:
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("Inkling chunked SFT requires logits_to_keep=0")
            labels = shifted if shifted is not None else F.pad(labels, (0, 1), value=-100)[:, 1:]
        else:
            action_masks = slice_action_masks_for_logits_to_keep(
                action_masks,
                batch_size=hidden.shape[0],
                original_seq_len=hidden.shape[1],
                logits_to_keep=logits_to_keep,
            )
            columns = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int) and logits_to_keep > 0
                else (slice(None) if isinstance(logits_to_keep, int) else logits_to_keep)
            )
            hidden, labels = (hidden[:, columns], labels.reshape(hidden.shape[:2])[:, columns])
            temperature = slice_temperature_for_logits_to_keep(temperature, columns)
            if group_token_ids is not None:
                group_token_ids = group_token_ids[:, columns]
        vocab = self.config.unpadded_vocab_size or self.vocab_size
        logprobs = chunked_lm_head_logprobs(
            hidden,
            self.lm_head.weight[:vocab],
            labels,
            token_chunk_size=chunk_size,
            vocab_chunk_size=8192,
            fp32_lm_head=self._inkling_fp32_lm_head,
            temperature=temperature if dss_compute_logprobs else None,
            action_masks=action_masks if dss_compute_logprobs else None,
            group_token_ids=group_token_ids if dss_compute_logprobs else None,
        )
        if dss_compute_logprobs:
            if group_token_ids is not None:
                return {"logprobs": logprobs[0], "group_log_probs": logprobs[1]}
            return {"logprobs": logprobs}
        valid = labels != -100
        denominator = valid.sum() if num_items is None else num_items
        loss = -logprobs.masked_fill(~valid, 0).sum() / denominator
        return InklingCausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
