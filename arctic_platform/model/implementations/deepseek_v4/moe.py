"""DeepSeek V4 mixture of experts on Arctic Platform's MoE runtime.

Hash layers replace the learned top-k with a frozen token-id table. Learned
layers keep the shared router, including the aux-loss-free expert bias.
"""

import torch
from torch import nn

from arctic_platform.model.implementations.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs
from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter


class DeepseekV4HashRouter(TokenChoiceTopKRouter):
    """Selection is ``tid2eid[token_id]``; the gate still scores those experts."""

    def __init__(self, *, vocab_size: int, **router_kwargs) -> None:
        super().__init__(**router_kwargs)
        self.register_buffer("tid2eid", torch.zeros(vocab_size, self.top_k, dtype=torch.long), persistent=True)
        self.force_balanced = False


class DeepseekV4MoE(MoE):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        if config.hidden_act != "silu":
            raise ValueError(f"DeepSeek V4 experts implement SiLU, got hidden_act={config.hidden_act!r}")
        if config.mlp_bias:
            raise ValueError("DeepSeek V4 does not support mlp_bias")
        is_hash = layer_idx < config.num_hash_layers
        super().__init__(
            MoEArgs(
                num_experts=config.n_routed_experts,
                num_shared_experts=1 if config.n_shared_experts > 0 else 0,
                score_func=config.scoring_func,
                route_norm=True,
                route_scale=config.routed_scaling_factor,
                score_before_experts=False,
                top_k=config.num_experts_per_tok,
                load_balance_coeff=None if is_hash else 1e-3,
                swiglu_limit=config.swiglu_limit,
            ),
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
        )
        self.layer_idx = layer_idx
        self.is_hash = is_hash
        if is_hash:
            self.router = DeepseekV4HashRouter(
                vocab_size=config.vocab_size,
                dim=config.hidden_size,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                score_func=config.scoring_func,
                route_norm=True,
                route_scale=config.routed_scaling_factor,
            )

    def init_weights(self, init_std: float, buffer_device: torch.device) -> None:
        super().init_weights(init_std, buffer_device)
        nn.init.trunc_normal_(self.experts.w3, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_hash and routed_experts is None and not self.router.force_balanced:
            if input_ids is None:
                raise ValueError(f"layer {self.layer_idx} is hash-routed and needs input_ids")
            routed_experts = self.router.tid2eid[input_ids]
        return super().forward(x, routed_experts=routed_experts)
