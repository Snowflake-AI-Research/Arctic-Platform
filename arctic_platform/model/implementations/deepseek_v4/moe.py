"""DeepSeek V4 mixture of experts on Arctic Platform's MoE runtime.

Hash layers replace the learned top-k with a frozen token-id table. Learned
layers keep the shared router, including the aux-loss-free expert bias.
"""

import torch
from torch import Tensor
from torch import nn

from arctic_platform.model.implementations.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from arctic_platform.model.implementations.deepseek_v4.dequantize import dequantize_mxfp4
from arctic_platform.model.implementations.fp8 import fp8_weight_block_size
from arctic_platform.model.implementations.fp8 import mark_keep_fp32
from arctic_platform.model.implementations.moe.config import DISPATCH_EP_BACKENDS
from arctic_platform.model.implementations.moe.config import EPCommBackend
from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep
from arctic_platform.model.implementations.moe.layers.moe import MoE
from arctic_platform.model.implementations.moe.layers.moe import MoEArgs
from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter
from arctic_platform.model.implementations.moe.layers.moe import _maybe_to_local
from arctic_platform.model.implementations.moe.layers.moe import _run_experts_for_loop_impl
from arctic_platform.model.implementations.moe.layers.moe import _run_experts_grouped_mm_impl

_MXFP4_BLOCK = 32


class _Mxfp4ExpertsFn(torch.autograd.Function):
    """MXFP4 grouped expert forward. Backward recomputes the BF16 weights from the packed values."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        w1: Tensor,
        s1: Tensor,
        w2: Tensor,
        s2: Tensor,
        w3: Tensor,
        s3: Tensor,
        num_tokens_per_expert: Tensor,
        swiglu_limit: float | None,
        use_grouped_mm: bool,
    ) -> Tensor:
        ctx.save_for_backward(x, w1, s1, w2, s2, w3, s3, num_tokens_per_expert)
        ctx.swiglu_limit = swiglu_limit
        ctx.use_grouped_mm = use_grouped_mm
        return _mxfp4_experts(x, w1, s1, w2, s2, w3, s3, num_tokens_per_expert, swiglu_limit, use_grouped_mm)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, w1, s1, w2, s2, w3, s3, num_tokens_per_expert = ctx.saved_tensors
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            output = _mxfp4_experts(
                x_in,
                w1,
                s1,
                w2,
                s2,
                w3,
                s3,
                num_tokens_per_expert,
                ctx.swiglu_limit,
                False,
            )
            (grad_x,) = torch.autograd.grad(output, x_in, grad_output)
        return grad_x, None, None, None, None, None, None, None, None, None


def _mxfp4_experts(
    x: Tensor,
    w1: Tensor,
    s1: Tensor,
    w2: Tensor,
    s2: Tensor,
    w3: Tensor,
    s3: Tensor,
    num_tokens_per_expert: Tensor,
    swiglu_limit: float | None,
    use_grouped_mm: bool,
) -> Tensor:
    out_dtype = x.dtype
    if x.dtype != torch.bfloat16:
        x = x.bfloat16()
    restored = (
        dequantize_mxfp4(w1, s1),
        dequantize_mxfp4(w2, s2),
        dequantize_mxfp4(w3, s3),
    )
    if x.is_cuda and use_grouped_mm:
        output = _run_experts_grouped_mm_impl(*restored, x, num_tokens_per_expert, swiglu_limit)
    else:
        output = _run_experts_for_loop_impl(*restored, x, num_tokens_per_expert, swiglu_limit)
    return output if output.dtype == out_dtype else output.to(dtype=out_dtype)


class Mxfp4GroupedExperts(nn.Module):
    """Routed experts stored as packed MXFP4, the official Flash checkpoint layout.

    The forward dequantizes into a temporary BF16 tensor for the grouped GEMM
    and drops it. LoRA does not wrap these parameters.
    """

    mxfp4 = True

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, swiglu_limit: float | None):
        super().__init__()
        if dim % _MXFP4_BLOCK or hidden_dim % _MXFP4_BLOCK:
            raise ValueError(
                f"MXFP4 experts need hidden sizes divisible by {_MXFP4_BLOCK}, got dim={dim} hidden_dim={hidden_dim}"
            )
        self.num_experts = num_experts
        self.swiglu_limit = swiglu_limit
        self.use_grouped_mm = True
        self.ep_comm_backend: EPCommBackend = "deepep"
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim // 2, dtype=torch.int8), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim // 2, dtype=torch.int8), requires_grad=False)
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim // 2, dtype=torch.int8), requires_grad=False)
        self.w1_scale_inv = _mxfp4_scale(num_experts, hidden_dim, dim)
        self.w2_scale_inv = _mxfp4_scale(num_experts, dim, hidden_dim)
        self.w3_scale_inv = _mxfp4_scale(num_experts, hidden_dim, dim)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def init_weights(self, init_std: float) -> None:
        del init_std

    def forward(self, x: Tensor, num_tokens_per_expert: Tensor) -> Tensor:
        if not uses_dispatch_ep(self.ep_comm_backend):
            raise NotImplementedError(
                f"EP comm backend must be one of {DISPATCH_EP_BACKENDS}, got {self.ep_comm_backend!r}."
            )
        return _Mxfp4ExpertsFn.apply(
            x,
            _maybe_to_local(self.w1),
            _maybe_to_local(self.w1_scale_inv),
            _maybe_to_local(self.w2),
            _maybe_to_local(self.w2_scale_inv),
            _maybe_to_local(self.w3),
            _maybe_to_local(self.w3_scale_inv),
            num_tokens_per_expert,
            self.swiglu_limit,
            self.use_grouped_mm,
        )


def _mxfp4_scale(num_experts: int, rows: int, cols: int) -> nn.Parameter:
    return mark_keep_fp32(
        nn.Parameter(
            torch.empty(num_experts, rows, cols // _MXFP4_BLOCK, dtype=torch.float32),
            requires_grad=False,
        )
    )


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
        block = fp8_weight_block_size(config)
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
                fp8_block_size=block,
                swiglu_limit=config.swiglu_limit,
            ),
            dim=config.hidden_size,
            hidden_dim=config.moe_intermediate_size,
        )
        if block is not None and getattr(config, "expert_dtype", "fp4") == "fp4":
            self.experts = Mxfp4GroupedExperts(
                dim=config.hidden_size,
                hidden_dim=config.moe_intermediate_size,
                num_experts=config.n_routed_experts,
                swiglu_limit=config.swiglu_limit,
            )
            self.experts.set_ep_comm_backend(self.ep_comm_backend)
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
        if getattr(self.experts, "mxfp4", False):
            return
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
