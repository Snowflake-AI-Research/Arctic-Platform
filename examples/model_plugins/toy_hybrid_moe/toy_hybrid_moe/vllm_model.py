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

"""ToyHybridMoE for vLLM: the sampler half of the plugin.

Written against vLLM 0.33's model interfaces and loaded through the ``vllm.general_plugins`` entry point. It needs
a GPU and vLLM, so the CPU conformance tiers do not run it; the service's sampler tier compares its log-probs with
the trainer's after a weight sync.

Tensor parallelism shards attention and recurrence heads, convolution channels and the MLP intermediate dimension.
The recurrence and the short convolution are vLLM stateful ("mamba-like") layers. Both declare one state layout, a
convolution history and a recurrent state per request, so vLLM keeps them in one cache group; the convolution leaves
the recurrent state unused. Each layer reads and updates its state in place, one request at a time, from the
metadata of vLLM's short-convolution backend. They are not registered as custom ops, so the manifest sets
``enforce_eager``. Routed experts run through vLLM's fused MoE layer, which also records the experts each token was
routed to when the engine runs with ``enable_return_routed_experts``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .config import FULL_ATTENTION
from .config import MOE
from .config import SHORT_CONV
from .config import ToyHybridMoEConfig
from .sampler_names import TRAINER_ONLY
from .sampler_names import sampler_target

# Recurrent and convolution state stay fp32 whatever the model dtype, as the trainer computes them in fp32.
_STATE_DTYPES = (torch.float32, torch.float32)


def register() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry

    # The sampler process never calls PLUGIN.register, so the config type must be registered here too.
    AutoConfig.register(ToyHybridMoEConfig.model_type, ToyHybridMoEConfig, exist_ok=True)
    ModelRegistry.register_model("ToyHybridMoEForCausalLM", "toy_hybrid_moe.vllm_model:ToyHybridMoEForCausalLM")


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """The trainer's rotary embedding on ``[tokens, heads, dim]``."""
    dim = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim)
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[:, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[:, None, :]
    first, second = x.float().chunk(2, dim=-1)
    return (x.float() * cos + torch.cat((-second, first), dim=-1) * sin).to(x.dtype)


class ToyAttentionVLLM(nn.Module):
    def __init__(self, config, cache_config, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.heads = config.num_attention_heads // tp
        self.kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False, prefix=f"{prefix}.o_proj"
        )
        self.attn = Attention(
            self.heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.kv_heads,
            cache_config=cache_config,
            per_layer_sliding_window=config.sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(
            [self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim], -1
        )
        q = _rope(q.view(-1, self.heads, self.head_dim), positions, self.theta).flatten(1)
        k = _rope(k.view(-1, self.kv_heads, self.head_dim), positions, self.theta).flatten(1)
        out, _ = self.o_proj(self.attn(q, k, v))
        return out


def _state_shapes(config, tp: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The layout every stateful layer declares: a convolution history in vLLM's orientation, and a recurrent
    state of ``[heads, head_dim]``."""
    width = config.recurrence_heads * config.recurrence_head_dim // tp
    conv = MambaStateShapeCalculator.short_conv_state_shape(
        tp_world_size=1, intermediate_size=width, conv_kernel=config.conv_kernel
    )[0]
    return conv, (config.recurrence_heads // tp, config.recurrence_head_dim)


class _StatefulLayer(MambaBase, nn.Module):
    """vLLM bookkeeping shared by the stateful layers: registration, state layout and per-request iteration."""

    def __init__(self, config, prefix: str):
        super().__init__()
        self.tp = get_tensor_model_parallel_world_size()
        self.config = config
        self.prefix = prefix
        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"duplicate layer name {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return _STATE_DTYPES

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return _state_shapes(self.config, self.tp)

    def _states(self) -> tuple[object, torch.Tensor, torch.Tensor] | None:
        """This step's metadata, the convolution history as ``[slots, kernel - 1, width]`` and the recurrent
        state; ``None`` in the profiling run before caches exist."""
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        metadata = metadata.get(self.prefix) if isinstance(metadata, dict) else metadata
        if metadata is None:
            return None
        conv, recurrent = self.kv_cache
        return metadata, conv.transpose(-1, -2) if is_conv_state_dim_first() else conv, recurrent

    @staticmethod
    def _requests(metadata):
        """``(begin, end, slot, resumes)`` per request: its tokens, its state slot, and whether it has state.

        vLLM orders a step's tokens decodes first, one token each, then prefills.
        """
        decodes = metadata.num_decode_tokens
        if decodes > 0:
            for index, slot in enumerate(metadata.state_indices_tensor_d.reshape(decodes, -1)[:, 0].tolist()):
                yield index, index + 1, slot, True
        if metadata.num_prefills > 0:
            starts = metadata.query_start_loc_p.tolist()
            slots = metadata.state_indices_tensor_p.reshape(metadata.num_prefills, -1)[:, 0].tolist()
            for index, (slot, resumes) in enumerate(zip(slots, metadata.has_initial_states_p.tolist())):
                yield decodes + starts[index], decodes + starts[index + 1], slot, resumes


class ToyShortConvVLLM(_StatefulLayer):
    """The gated short convolution; its state is the last ``kernel - 1`` inputs."""

    def __init__(self, config, prefix: str):
        super().__init__(config, prefix)
        if config.conv_channels != config.recurrence_heads * config.recurrence_head_dim:
            raise ValueError(
                "this vLLM model keeps one state layout, so conv_channels must equal the recurrence width"
            )
        self.in_gate_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.conv_channels] * 2, bias=False, prefix=f"{prefix}.in_gate_proj"
        )
        self.out_proj = RowParallelLinear(
            config.conv_channels, config.hidden_size, bias=False, prefix=f"{prefix}.out_proj"
        )
        self.conv1d = _conv_weight(config.conv_channels, config.conv_kernel, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, _ = self.in_gate_proj(x)
        u, gate = projected.float().chunk(2, dim=-1)
        conv = torch.zeros_like(u)
        states = self._states()
        if states is not None:
            metadata, history_cache, _ = states
            weight = self.conv1d.weight[:, 0, :].float()
            for begin, end, slot, resumes in self._requests(metadata):
                history = history_cache[slot].float() if resumes else torch.zeros_like(history_cache[slot]).float()
                window = torch.cat((history, u[begin:end]), dim=0)
                conv[begin:end] = _causal_conv(window, weight)
                history_cache[slot] = window[-history.shape[0] :].to(history_cache.dtype)
        out, _ = self.out_proj((conv * F.silu(gate)).to(x.dtype))
        return out


class ToyRecurrenceVLLM(_StatefulLayer):
    """The gated recurrence: a short convolution, then ``h_t = a_t * h_{t-1} + (1 - a_t) * c_t``."""

    def __init__(self, config, prefix: str):
        super().__init__(config, prefix)
        self.heads = config.recurrence_heads // self.tp
        self.head_dim = config.recurrence_head_dim
        width = config.recurrence_heads * config.recurrence_head_dim
        self.in_gate_proj = MergedColumnParallelLinear(
            config.hidden_size, [width, width], bias=False, prefix=f"{prefix}.in_gate_proj"
        )
        self.z_proj = ColumnParallelLinear(config.hidden_size, width, bias=False, prefix=f"{prefix}.z_proj")
        self.out_proj = RowParallelLinear(width, config.hidden_size, bias=False, prefix=f"{prefix}.out_proj")
        self.conv1d = _conv_weight(width, config.conv_kernel, prefix)
        self.decay_bias = nn.Parameter(torch.empty(self.heads))
        set_weight_attrs(self.decay_bias, {"weight_loader": sharded_weight_loader(0)})
        self.norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, _ = self.in_gate_proj(x)
        u, g = projected.float().chunk(2, dim=-1)
        z, _ = self.z_proj(x)
        h = torch.zeros_like(u)
        states = self._states()
        if states is not None:
            metadata, history_cache, state_cache = states
            weight = self.conv1d.weight[:, 0, :].float()
            decay = torch.sigmoid(g.view(-1, self.heads, self.head_dim) + self.decay_bias.float()[None, :, None])
            for begin, end, slot, resumes in self._requests(metadata):
                history = history_cache[slot].float() if resumes else torch.zeros_like(history_cache[slot]).float()
                state = state_cache[slot].float() if resumes else torch.zeros_like(state_cache[slot]).float()
                window = torch.cat((history, u[begin:end]), dim=0)
                c = F.silu(_causal_conv(window, weight)).view(-1, self.heads, self.head_dim)
                for t in range(end - begin):
                    state = decay[begin + t] * state + (1 - decay[begin + t]) * c[t]
                    h[begin + t] = state.flatten()
                history_cache[slot] = window[-history.shape[0] :].to(history_cache.dtype)
                state_cache[slot] = state.to(state_cache.dtype)
        h = self.norm(h.view(-1, self.heads, self.head_dim).to(x.dtype))
        out, _ = self.out_proj((h * F.silu(z.view(-1, self.heads, self.head_dim))).flatten(1))
        return out


def _conv_weight(width: int, kernel: int, prefix: str) -> ColumnParallelLinear:
    # A ColumnParallelLinear shards the Conv1d weight [channels, 1, kernel] by channel, as vLLM's mamba layers do.
    conv1d = ColumnParallelLinear(kernel, width, bias=False, prefix=f"{prefix}.conv1d")
    conv1d.weight.data = conv1d.weight.data.unsqueeze(1)
    return conv1d


def _causal_conv(window: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution of ``window`` ``[history + T, C]`` with ``weight`` ``[C, K]``, where the
    history is the ``K - 1`` inputs before the first token; returns ``[T, C]``."""
    kernel = weight.shape[1]
    tokens = window.shape[0] - (kernel - 1)
    return sum(window[lag : lag + tokens] * weight[:, lag] for lag in range(kernel))


class ToyMoEVLLM(nn.Module):
    """Softmax top-k routing with renormalized weights through vLLM's fused MoE, plus the shared expert."""

    def __init__(self, config, prefix: str):
        super().__init__()
        self.gate = GateLinear(config.hidden_size, config.num_experts, bias=False, prefix=f"{prefix}.gate")
        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            scoring_func="softmax",
            routed_scaling_factor=config.routed_scaling_factor,
            prefix=f"{prefix}.experts",
        )
        shared = config.moe_intermediate_size * config.num_shared_experts
        self.shared_expert = _SwiGLU(config.hidden_size, shared, f"{prefix}.shared_expert")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gate(x)
        return self.experts(hidden_states=x, router_logits=logits) + self.shared_expert(x)


class _SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False, prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, prefix=f"{prefix}.down_proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x)[0].chunk(2, dim=-1)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


class _DecoderLayer(nn.Module):
    def __init__(self, config, cache_config, index: int, prefix: str):
        super().__init__()
        self.kind = config.layer_types[index]
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.kind == FULL_ATTENTION:
            self.self_attn = ToyAttentionVLLM(config, cache_config, f"{prefix}.self_attn")
        elif self.kind == SHORT_CONV:
            self.conv = ToyShortConvVLLM(config, f"{prefix}.conv")
        else:
            self.recurrence = ToyRecurrenceVLLM(config, f"{prefix}.recurrence")
        if config.mlp_types[index] == MOE:
            self.mlp = ToyMoEVLLM(config, f"{prefix}.mlp")
        else:
            self.mlp = _SwiGLU(config.hidden_size, config.intermediate_size, f"{prefix}.mlp")

    def forward(self, positions: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(h)
        if self.kind == FULL_ATTENTION:
            h = h + self.self_attn(positions, x)
        else:
            h = h + (self.conv(x) if self.kind == SHORT_CONV else self.recurrence(x))
        return h + self.mlp(self.post_norm(h))


class _Model(nn.Module):
    def __init__(self, config, cache_config, prefix: str):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            _DecoderLayer(config, cache_config, i, f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class ToyHybridMoEForCausalLM(nn.Module, HasInnerState, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = _Model(config, vllm_config.cache_config, "model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        return _STATE_DTYPES

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return _state_shapes(vllm_config.model_config.hf_config, vllm_config.parallel_config.tensor_parallel_size)

    @classmethod
    def get_mamba_state_copy_func(cls):
        # A convolution history and a recurrent state, the same pair of states as a Mamba-1 layer.
        return MambaStateCopyFuncCalculator.mamba1_state_copy_func()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        for layer in self.model.layers:
            h = layer(positions, h)
        return self.model.norm(h)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Hugging Face-named tensors, from the checkpoint or a weight sync. Returns the parameters written."""
        params = dict(self.named_parameters())
        modules = dict(self.named_modules())
        loaded: set[str] = set()
        for name, tensor in weights:
            if TRAINER_ONLY.search(name):
                continue
            target = sampler_target(name)
            param = params[target.param]
            if target.expert is not None:
                experts = modules[target.param.rsplit(".", 1)[0]]
                experts.weight_loader(param, tensor, target.param, target.shard, target.expert, return_success=True)
            elif target.shard is not None:
                param.weight_loader(param, tensor, target.shard)
            else:
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
            loaded.add(target.param)
        return loaded
