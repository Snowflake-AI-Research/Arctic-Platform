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

"""CPU checks for the rounding behaviour of the custom Qwen3.5-MoE trainer forward.

With ``fp32_parity`` (the default) the custom model rounds like vLLM in three places: the residual add happens in fp32
inside the next RMSNorm, partial RoPE is evaluated in fp32, and the GatedDeltaNet beta gate is a sigmoid of fp32 ``b``.
These tests check each of them against an independently written reference, check that the residual threading between
decoder layers survives activation checkpointing and activation offload in forward and backward, and check that the
state dict keys and shapes are the expected ones. One test pins the fp32 beta gate that the generic transformers
sequence-parallel wrapper gets from the same shared head-parallel function.

With ``fp32_parity`` off each decoder layer adds its residuals in the input dtype at the end of the layer, and the
rotary embedding and the beta gate run in the input dtype. Those tests compare the model bitwise, in forward and
backward, with a reference kept here that computes the same operations in the same order (``_input_dtype_*``), and
check that each decoder layer passes on one tensor.

Every test here runs with the optional ``causal_conv1d`` and ``fla`` kernels hidden from the model module, so no
GatedDeltaNet layer built here reaches those CUDA-only packages even where they are installed: a layer whose
kernels a test does not stub uses the model's pure-PyTorch convolution and gated-norm fallbacks. The tiny model
also runs the GatedDeltaNet recurrence through the model's pure-PyTorch fallback, and replaces each decoder layer's
routed-expert MoE with a dense projection, because the MoE router's token histogram is not implemented for integer
tensors on CPU. None of these substitutions touches the norms, the residual threading or the attention projections
that these tests are about.

The tests also clear ``FLASH_ATTENTION_DETERMINISTIC``, which a fully deterministic worker environment exports, so
the flash-attention variant can be built with a stand-in kernel. The other variables that environment exports
configure CUDA, NCCL and hashing, none of which these CPU tests depend on.
"""

import contextlib
import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.implementations.debug.determinism import FLASH_ATTENTION_DETERMINISTIC_ENV
from arctic_platform.model.implementations.gpu import activation_offload
from arctic_platform.model.implementations.gpu.activation_offload import ActivationOffloadManager
from arctic_platform.model.implementations.gpu.sp import gated_delta_net as sp_gated_delta_net
from arctic_platform.model.implementations.moe.layers.checkpointing import checkpoint_method
from arctic_platform.model.implementations.qwen35.model_builder import apply_ac
from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen
from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

EPS = 1e-6


@pytest.fixture(autouse=True)
def _without_optional_gated_delta_net_kernels(monkeypatch):
    """Hide the CUDA-only GatedDeltaNet kernels from the model module, which reads them when a module is built."""
    monkeypatch.setattr(qwen, "causal_conv1d_fn", None)
    monkeypatch.setattr(qwen, "chunk_gated_delta_rule", None)
    monkeypatch.setattr(qwen, "FusedRMSNormGated", None)


@pytest.fixture(autouse=True)
def _without_flash_attention_determinism_request(monkeypatch):
    """Clear the environment variable that asks flash attention for a deterministic backward.

    The flash-attention variant reads it when it is built and refuses a kernel that cannot honour it, which the
    stand-in kernel used here cannot.
    """
    monkeypatch.delenv(FLASH_ATTENTION_DETERMINISTIC_ENV, raising=False)


@pytest.fixture
def single_rank_group():
    """The default process group, with one rank.

    The root conftest normally provides it, but tests that tear the session group down leave none behind, so a
    single-rank gloo group is created here when none exists and destroyed again only if it was created here.
    """
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="gloo", world_size=1, rank=0, store=dist.HashStore())
    assert dist.get_world_size() == 1
    yield dist.group.WORLD
    if created and dist.is_initialized():
        dist.destroy_process_group()


def _input_dtype_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Qwen3_5MoeRMSNorm.forward`` without a residual, written out: the same operations in the same order."""
    x_fp32 = x.float()
    output = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.type_as(x)


def _reference_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add in fp32, normalize the fp32 sum, and round both outputs to the input dtype."""
    summed = x.float() + residual.float()
    normalized = summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())
    return normalized.to(x.dtype), summed.to(x.dtype)


def _norm(hidden_size: int, dtype: torch.dtype) -> qwen.Qwen3_5MoeRMSNorm:
    norm = qwen.Qwen3_5MoeRMSNorm(hidden_size, eps=EPS)
    with torch.no_grad():
        norm.weight.normal_(std=0.1)
    return norm.to(dtype)


def _norm_inputs(dtype: torch.dtype, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.randn(2, 5, 64) * 3).to(dtype).requires_grad_(requires_grad)
    residual = (torch.randn(2, 5, 64) * 3).to(dtype).requires_grad_(requires_grad)
    return x, residual


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_norm_without_residual_matches_the_input_dtype_reference(dtype):
    set_seed(0)
    norm = _norm(64, dtype)
    x, _ = _norm_inputs(dtype)

    output = norm(x)

    assert isinstance(output, torch.Tensor)
    torch_assert_equal(output, _input_dtype_rms_norm(x, norm.weight, EPS))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_norm_forward_matches_fp32_reference(dtype):
    set_seed(0)
    norm = _norm(64, dtype)
    x, residual = _norm_inputs(dtype)

    normalized, residual_out = norm(x, residual)
    expected_normalized, _ = _reference_add_rms_norm(x, residual, norm.weight, EPS)

    assert normalized.dtype == dtype
    assert residual_out.dtype == dtype
    torch_assert_equal(normalized, expected_normalized)
    torch_assert_equal(residual_out, (x.float() + residual.float()).to(x.dtype))

    # A second reference with a different order of operations: torch's own RMSNorm of the fp32 sum. The two
    # differ by fp32 rounding before the final cast. For these unit-scale outputs that is well below 1e-5 in fp32;
    # in bf16 it can flip the final rounding by one bf16 ulp (2**-7 relative), so the bound is that ulp there.
    summed = x.float() + residual.float()
    torch_reference = F.rms_norm(summed, (64,), weight=1.0 + norm.weight.float(), eps=EPS).to(dtype)
    if dtype == torch.float32:
        torch_assert_close(normalized, torch_reference, rtol=0, atol=1e-5)
    else:
        torch_assert_close(normalized.float(), torch_reference.float(), rtol=2**-7, atol=0)


def test_fused_norm_gradcheck_float64():
    set_seed(0)
    x, residual = _norm_inputs(torch.float64, requires_grad=True)
    weight = (torch.randn(64, dtype=torch.float64) * 0.1).requires_grad_(True)

    def fused(x, residual, weight):
        return qwen._FusedAddRMSNorm.apply(x, residual, weight, EPS)

    assert torch.autograd.gradcheck(fused, (x, residual, weight))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_norm_backward_matches_eager_autograd(dtype):
    set_seed(0)
    norm = _norm(64, dtype)
    x, residual = _norm_inputs(dtype, requires_grad=True)
    grad_normalized = torch.randn(2, 5, 64, dtype=dtype)
    grad_residual = torch.randn(2, 5, 64, dtype=dtype)

    normalized, residual_out = norm(x, residual)
    torch.autograd.backward((normalized, residual_out), (grad_normalized, grad_residual))
    fused_grads = [x.grad, residual.grad, norm.weight.grad]

    eager_x = x.detach().clone().requires_grad_(True)
    eager_residual = residual.detach().clone().requires_grad_(True)
    eager_weight = norm.weight.detach().clone().requires_grad_(True)
    summed = eager_x + eager_residual
    eager_normalized = summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + eager_weight)
    torch.autograd.backward((eager_normalized, summed), (grad_normalized, grad_residual))
    eager_grads = [eager_x.grad, eager_residual.grad, eager_weight.grad]

    # The fused backward evaluates the analytic RMSNorm gradient, autograd chains the elementwise derivatives;
    # the two orders of operations differ only by rounding. For these inputs the gradients are O(1) and the weight
    # gradient sums 10 rows, so the absolute rounding difference is well below 1e-5 in fp32 and 1e-12 in fp64.
    atol = 1e-5 if dtype == torch.float32 else 1e-12
    for name, fused_grad, eager_grad in zip(["x", "residual", "weight"], fused_grads, eager_grads, strict=True):
        torch_assert_close(fused_grad, eager_grad, rtol=0, atol=atol, msg=f"gradient of {name}")


def test_fused_norm_bf16_backward_matches_fp32_autograd():
    set_seed(0)
    norm = _norm(64, torch.bfloat16)
    x = (torch.randn(8, 64, 64) * 3).to(torch.bfloat16).requires_grad_(True)
    residual = (torch.randn(8, 64, 64) * 3).to(torch.bfloat16).requires_grad_(True)
    grad_normalized = torch.randn(8, 64, 64).to(torch.bfloat16)
    grad_residual = torch.randn(8, 64, 64).to(torch.bfloat16)

    normalized, residual_out = norm(x, residual)
    torch.autograd.backward((normalized, residual_out), (grad_normalized, grad_residual))
    fused_grads = [x.grad, residual.grad, norm.weight.grad]

    # Reference: eager autograd on fp32 copies of the same bf16 values, so the sum is never rounded to bf16.
    eager_x, eager_residual, eager_weight = [
        tensor.detach().float().requires_grad_(True) for tensor in (x, residual, norm.weight)
    ]
    summed = eager_x + eager_residual
    eager_normalized = summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + eager_weight)
    torch.autograd.backward((eager_normalized, summed), (grad_normalized.float(), grad_residual.float()))
    eager_grads = [eager_x.grad, eager_residual.grad, eager_weight.grad]

    # Both gradients are fp32 values that differ only by fp32 rounding before the final cast to bf16. For these
    # pinned inputs that flips the cast by at most one bf16 ulp (2**-7 relative). This is not a general bound:
    # an element that is near zero after cancellation can differ by more than one ulp from fp32 noise alone, so
    # a changed shape or seed may need a small atol justified by that noise. Recomputing the sum in bf16 instead
    # moves many elements here by several ulps.
    for name, fused_grad, eager_grad in zip(["x", "residual", "weight"], fused_grads, eager_grads, strict=True):
        assert fused_grad.dtype == torch.bfloat16, name
        torch_assert_close(
            fused_grad.float(), eager_grad.to(torch.bfloat16).float(), rtol=2**-7, atol=0, msg=f"gradient of {name}"
        )


def test_fused_norm_gradients_identical_under_checkpointing():
    set_seed(0)
    norm = _norm(64, torch.bfloat16)
    x, residual = _norm_inputs(torch.bfloat16)
    grad_normalized = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    grad_residual = torch.randn(2, 5, 64, dtype=torch.bfloat16)

    def run(forward):
        module = copy.deepcopy(norm)
        x_leaf = x.clone().requires_grad_(True)
        residual_leaf = residual.clone().requires_grad_(True)
        outputs = forward(module, x_leaf, residual_leaf)
        torch.autograd.backward(outputs, (grad_normalized, grad_residual))
        return [*outputs, x_leaf.grad, residual_leaf.grad, module.weight.grad]

    def eager(module, x, residual):
        return module(x, residual)

    def torch_checkpoint(module, x, residual):
        return checkpoint(module, x, residual, use_reentrant=False)

    def model_checkpoint(module, x, residual):
        checkpoint_method(module, "forward")
        return module(x, residual)

    expected = run(eager)
    for forward in (torch_checkpoint, model_checkpoint):
        for actual, reference in zip(run(forward), expected, strict=True):
            torch_assert_equal(actual, reference, msg=forward.__name__)


def test_fused_norm_fp32_matches_add_then_norm():
    set_seed(0)
    norm = _norm(64, torch.float32)
    x, residual = _norm_inputs(torch.float32)

    normalized, residual_out = norm(x, residual)

    # In fp32 the fused path performs the same add as an add followed by the norm; the bound is the 1e-6 relative
    # fp32 rounding allowance.
    torch_assert_close(normalized, _input_dtype_rms_norm(residual + x, norm.weight, EPS), rtol=1e-6, atol=0)
    torch_assert_equal(residual_out, residual + x)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def _rope_inputs(dtype: torch.dtype, head_dim: int = 256, rotary_dim: int = 64):
    batch, heads, seq = 2, 3, 7
    x = torch.randn(batch, heads, seq, head_dim).to(dtype)
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    positions = torch.arange(seq, dtype=torch.float32)[None, :].expand(batch, -1)
    freqs = positions[..., None] * inv_freq
    angles = torch.cat((freqs, freqs), dim=-1)
    return x, angles.cos().to(dtype), angles.sin().to(dtype)


def _reference_partial_rope_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Upcast all of ``x``, rotate the first ``rotary_dim`` channels pairwise in fp32, and cast back once.

    Channel ``i`` of the first half pairs with channel ``i + half`` of the rotary slice:
    ``(a, b) -> (a * cos - b * sin, b * cos + a * sin)``. The remaining channels pass through the fp32 round trip.
    """
    x_fp32 = x.float()
    cos_fp32 = cos.float().unsqueeze(1)
    sin_fp32 = sin.float().unsqueeze(1)
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    first, second = x_fp32[..., :half], x_fp32[..., half:rotary_dim]
    rotated_first = first * cos_fp32[..., :half] - second * sin_fp32[..., :half]
    rotated_second = second * cos_fp32[..., half:] + first * sin_fp32[..., half:]
    return torch.cat((rotated_first, rotated_second, x_fp32[..., rotary_dim:]), dim=-1).to(x.dtype)


def test_rope_rotary_slice_matches_full_fp32_rotation():
    set_seed(0)
    x, cos, sin = _rope_inputs(torch.bfloat16)

    rotated = qwen._rope_rotary_slice_fp32(x, cos, sin)

    expected = _reference_partial_rope_fp32(x, cos, sin)
    assert rotated.dtype == torch.bfloat16
    assert rotated.shape == x.shape
    torch_assert_equal(rotated, expected)
    torch_assert_equal(rotated[..., 64:], x[..., 64:])


def _flash_attention(config: qwen.Qwen3_5MoeGatedAttentionConfig, monkeypatch) -> nn.Module:
    """Build the flash-attention variant without a flash-attention install; only its projections are used."""

    def unavailable_kernel(*args, **kwargs):
        raise AssertionError("the attention kernel is not part of these checks")

    monkeypatch.setattr(qwen.Qwen3_5MoeGatedFlashAttention, "_funcs", {3: unavailable_kernel})
    return qwen.Qwen3_5MoeGatedFlashAttention(config, flash_attn_version=3)


@pytest.mark.parametrize("variant", ["sdpa", "flash"])
def test_attention_variants_rotate_in_fp32(variant, monkeypatch):
    set_seed(0)
    config = qwen.Qwen3_5MoeGatedAttentionConfig(
        hidden_size=32, head_dim=256, num_attention_heads=2, num_key_value_heads=1, rms_norm_eps=EPS
    )
    if variant == "sdpa":
        attention = qwen.Qwen3_5MoeGatedSDPAAttention(config)
    else:
        attention = _flash_attention(config, monkeypatch)
    attention = attention.to(torch.bfloat16)
    with torch.no_grad():
        attention.q_norm.weight.normal_(std=0.1)
        attention.k_norm.weight.normal_(std=0.1)

    calls = []
    original = qwen._rope_rotary_slice_fp32

    def recording_rope(x, cos, sin):
        calls.append(x.shape)
        return original(x, cos, sin)

    monkeypatch.setattr(qwen, "_rope_rotary_slice_fp32", recording_rope)
    hidden_states = torch.randn(1, 7, 32).to(torch.bfloat16)
    _, cos, sin = _rope_inputs(torch.bfloat16)
    cos, sin = cos[:1], sin[:1]

    query, key, _, _ = attention.attn_projections(hidden_states, (cos, sin))

    assert len(calls) == 2
    expected_query = attention.q_norm(
        torch.chunk(attention.q_proj(hidden_states).view(1, 7, -1, 512), 2, dim=-1)[0].reshape(1, 7, -1, 256)
    ).transpose(1, 2)
    expected_key = attention.k_norm(attention.k_proj(hidden_states).view(1, 7, -1, 256)).transpose(1, 2)
    expected_query = original(expected_query, cos, sin)
    expected_key = original(expected_key, cos, sin)
    if variant == "flash":
        expected_query = expected_query.transpose(1, 2)
        expected_key = expected_key.transpose(1, 2)
    torch_assert_equal(query, expected_query)
    torch_assert_equal(key, expected_key)


# ---------------------------------------------------------------------------
# GatedDeltaNet beta gate
# ---------------------------------------------------------------------------


def _tiny_config(**overrides) -> Qwen3_5MoeConfig:
    config_args = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        pad_token_id=0,
    )
    config_args.update(overrides)
    config = Qwen3_5MoeConfig(**config_args)
    config.use_grouped_mm = False
    config._attn_implementation = "sdpa"
    return config


def _gate_projections(module: qwen.Qwen3_5MoeGatedDeltaNet, hidden_states: torch.Tensor):
    b = module.in_proj_b(hidden_states)
    a = module.in_proj_a(hidden_states)
    expected_g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    return b, expected_g


def test_gated_delta_net_beta_is_fp32():
    set_seed(0)
    module = qwen.Qwen3_5MoeGatedDeltaNet(_tiny_config()).to(torch.bfloat16)
    captured = {}

    def convolution(*, x, weight, bias, activation, seq_idx):
        captured["convolution_input"] = x
        return F.silu(x)

    def gated_delta_rule(query, key, value, *, g, beta, **kwargs):
        captured.update(g=g, beta=beta)
        return torch.zeros_like(value), None

    module._causal_conv1d_fn = convolution
    module._chunk_gated_delta_rule = gated_delta_rule
    hidden_states = torch.randn(1, 6, 16).to(torch.bfloat16)

    module(hidden_states)

    b, expected_g = _gate_projections(module, hidden_states)
    assert b.dtype == torch.bfloat16
    assert captured["beta"].dtype == torch.float32
    torch_assert_equal(captured["beta"], b.float().sigmoid())
    torch_assert_equal(captured["g"], expected_g)


def test_head_parallel_gated_delta_net_beta_is_fp32(single_rank_group, monkeypatch):
    # With one rank the all-to-all is the identity.
    monkeypatch.setattr(sp_gated_delta_net, "sequence_head_all_to_all", lambda group, tensor, **kwargs: tensor)
    set_seed(0)
    module = qwen.Qwen3_5MoeGatedDeltaNet(_tiny_config()).to(torch.bfloat16)
    hidden_states = torch.randn(1, 6, 16).to(torch.bfloat16)
    query, key, value = torch.split(module.in_proj_qkv(hidden_states), [8, 8, 8], dim=-1)
    b, expected_g = _gate_projections(module, hidden_states)
    a = module.in_proj_a(hidden_states)
    captured = {}

    def convolution(*, x, weight, bias, activation, seq_idx):
        return F.silu(x)

    def gated_delta_rule(query, key, value, g, beta, **kwargs):
        captured.update(g=g, beta=beta)
        return torch.zeros_like(value), None

    sp_gated_delta_net.head_parallel_gated_delta_net(
        convolution,
        gated_delta_rule,
        query.reshape(1, 6, 2, 4),
        key.reshape(1, 6, 2, 4),
        value.reshape(1, 6, 2, 4),
        b,
        a,
        convolution_weight=module.conv1d.weight.squeeze(1),
        convolution_bias=module.conv1d.bias,
        convolution_activation=module.activation,
        A_log=module.A_log,
        dt_bias=module.dt_bias,
        process_group=single_rank_group,
        global_cu_seqlens=torch.tensor([0, 6], dtype=torch.int32),
        num_key_heads=2,
        num_value_heads=2,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert captured["beta"].dtype == torch.float32
    torch_assert_equal(captured["beta"], b.float().sigmoid())
    torch_assert_equal(captured["g"], expected_g)


class _TransformersStyleGatedDeltaNet(nn.Module):
    """A module shaped like the GatedDeltaNet layers the generic transformers sequence-parallel wrapper adapts.

    The wrapper recognizes such a layer by its ``causal_conv1d_fn``, ``chunk_gated_delta_rule`` and
    ``conv_kernel_size`` attributes; this one records the gates its recurrence receives.
    """

    def __init__(self, hidden_size: int = 16, num_heads: int = 2, head_dim: int = 4):
        super().__init__()
        self.num_k_heads = num_heads
        self.num_v_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim
        self.activation = "silu"
        self.conv_kernel_size = 2
        conv_dim = 3 * num_heads * head_dim
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=2, groups=conv_dim, bias=False, padding=1)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, num_heads + 1, dtype=torch.float32)))
        self.norm = qwen.Qwen3_5MoeRMSNormGated(head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_heads, bias=False)
        self.captured: dict[str, torch.Tensor] = {}
        self.causal_conv1d_fn = self._convolution
        self.chunk_gated_delta_rule = self._gated_delta_rule

    @staticmethod
    def _convolution(*, x, weight, bias, activation, seq_idx):
        return F.silu(x)

    def _gated_delta_rule(self, query, key, value, g, beta, **kwargs):
        self.captured.update(g=g, beta=beta)
        return torch.zeros_like(value), None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the sequence-parallel wrapper replaces this forward")


def test_transformers_sequence_parallel_wrapper_beta_is_fp32(single_rank_group, monkeypatch):
    # head_parallel_gated_delta_net is shared by the custom model and the generic transformers wrapper, so the
    # wrapper's adapted layers get the fp32 beta gate too. This pins that, so narrowing the fp32 gate to the custom
    # model is a deliberate change. With one rank the all-to-all is the identity.
    monkeypatch.setattr(sp_gated_delta_net, "sequence_head_all_to_all", lambda group, tensor, **kwargs: tensor)
    set_seed(0)
    module = _TransformersStyleGatedDeltaNet().to(torch.bfloat16)
    assert sp_gated_delta_net._is_gated_delta_net_module(module)
    sp_gated_delta_net._adapt_gated_delta_net_module(module, single_rank_group)
    setattr(module, sp_gated_delta_net._GLOBAL_CU_SEQLENS_ATTRIBUTE, torch.tensor([0, 6], dtype=torch.int32))
    hidden_states = torch.randn(1, 6, 16).to(torch.bfloat16)

    module(hidden_states)

    b = module.in_proj_b(hidden_states)
    assert b.dtype == torch.bfloat16
    assert module.captured["beta"].dtype == torch.float32
    torch_assert_equal(module.captured["beta"], b.float().sigmoid())


# ---------------------------------------------------------------------------
# Tiny model: residual threading and activation checkpointing
# ---------------------------------------------------------------------------


class _DenseExperts(nn.Module):
    """Dense stand-in for the routed-expert MoE, which cannot run on CPU (see the module docstring)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        return self.proj(hidden_states)


def _torch_gated_delta_rule(query, key, value, *, cu_seqlens=None, **kwargs):
    """The model's pure-PyTorch recurrence, for a single unpacked sequence (``cu_seqlens`` is None under SDPA)."""
    assert cu_seqlens is None
    return qwen.torch_chunk_gated_delta_rule(query, key, value, **kwargs)


def _tiny_model(dtype: torch.dtype = torch.bfloat16, fp32_parity: bool = True) -> qwen.Qwen3_5MoeForCausalLM:
    set_seed(0)
    model = qwen.Qwen3_5MoeForCausalLM(_tiny_config(fp32_parity=fp32_parity))
    for layer in model.model.layers:
        layer.mlp = _DenseExperts(16)
        if layer.layer_type == "linear_attention":
            layer.linear_attn._chunk_gated_delta_rule = _torch_gated_delta_rule
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, qwen.Qwen3_5MoeRMSNorm):
                module.weight.normal_(std=0.1)
    return model.to(dtype).train()


def _tiny_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    position_ids = torch.arange(8)[None, :]
    return input_ids, position_ids


def _reference_backbone(model: qwen.Qwen3_5MoeForCausalLM, input_ids, position_ids) -> torch.Tensor:
    """The backbone written out layer by layer with every residual add done in fp32 inside the next norm."""
    backbone = model.model
    hidden_states = backbone.embed_tokens(input_ids)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
    residual = None
    for layer in backbone.layers:
        if residual is None:
            residual = hidden_states
            normed = _input_dtype_rms_norm(hidden_states, layer.input_layernorm.weight, layer.input_layernorm.eps)
        else:
            normed, residual = _reference_add_rms_norm(
                hidden_states, residual, layer.input_layernorm.weight, layer.input_layernorm.eps
            )
        if layer.layer_type == "linear_attention":
            mixed = layer.linear_attn(normed)
        else:
            mixed, _ = layer.self_attn(hidden_states=normed, position_embeddings=position_embeddings)
        normed, residual = _reference_add_rms_norm(
            mixed, residual, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
        )
        flat = normed.view(-1, normed.shape[-1])
        shared = qwen._shared_expert_gate(flat, layer.shared_expert_gate) * layer.shared_expert(flat)
        hidden_states = layer.mlp(normed) + shared.view_as(normed)
    normed, _ = _reference_add_rms_norm(hidden_states, residual, backbone.norm.weight, backbone.norm.eps)
    return normed


def test_decoder_layers_thread_residual(monkeypatch):
    model = _tiny_model()
    input_ids, position_ids = _tiny_inputs()
    calls = []

    for index, layer in enumerate(model.model.layers):
        original_forward = layer.forward

        def recording_forward(hidden_states, residual=None, *args, _index=index, _forward=original_forward, **kwargs):
            outputs = _forward(hidden_states, residual, *args, **kwargs)
            calls.append(dict(index=_index, residual_in=residual, outputs=outputs))
            return outputs

        monkeypatch.setattr(layer, "forward", recording_forward)

    with torch.no_grad():
        output = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state
        expected = _reference_backbone(model, input_ids, position_ids)

    assert [call["index"] for call in calls] == [0, 1, 2]
    assert calls[0]["residual_in"] is None
    for previous, current in zip(calls[:-1], calls[1:]):
        assert current["residual_in"] is previous["outputs"][1]
    for call in calls:
        assert isinstance(call["outputs"], tuple) and len(call["outputs"]) == 2
        hidden_states, residual = call["outputs"]
        assert hidden_states.dtype == torch.bfloat16
        assert residual.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    torch_assert_equal(output, expected)


def _forward_backward(
    model: qwen.Qwen3_5MoeForCausalLM, forward_context=contextlib.nullcontext
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    input_ids, position_ids = _tiny_inputs()
    with forward_context():
        output = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state
    projection = torch.linspace(-1.0, 1.0, output.numel()).view_as(output)
    loss = (output.float() * projection).sum()
    loss.backward()
    # Full checkpointing wraps each decoder layer, which inserts this segment into its parameter names.
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): parameter.grad
        for name, parameter in model.model.named_parameters()
    }
    return output.detach(), grads


def _assert_same_run(output, grads, expected_output, expected_grads) -> None:
    torch_assert_equal(output, expected_output)
    assert expected_grads.keys() == grads.keys()
    for name, expected_grad in expected_grads.items():
        assert expected_grad is not None, name
        torch_assert_equal(grads[name], expected_grad, msg=name)


@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "input-dtype"])
@pytest.mark.parametrize(
    "ac_config",
    [dict(mode="full"), dict(mode="selective", targets=["norm"])],
    ids=["full", "selective-norm"],
)
def test_activation_checkpointing_forward_backward_matches_eager(ac_config, fp32_parity):
    reference_model = _tiny_model(fp32_parity=fp32_parity)
    checkpointed_model = copy.deepcopy(reference_model)
    apply_ac(checkpointed_model, ActivationCheckpointConfig(**ac_config))

    expected_output, expected_grads = _forward_backward(reference_model)
    output, grads = _forward_backward(checkpointed_model)

    _assert_same_run(output, grads, expected_output, expected_grads)


def _eligible_on_any_device(self: ActivationOffloadManager, tensor: torch.Tensor) -> bool:
    """``ActivationOffloadManager._eligible`` without its CUDA-only device check and its skip counters."""
    if not self.enabled or isinstance(tensor, nn.Parameter):
        return False
    if tensor.numel() * tensor.element_size() < self.tensor_size_threshold:
        return False
    if not tensor.is_contiguous() and activation_offload._tensor_has_storage_overlap(tensor):
        return False
    return True


@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "input-dtype"])
def test_activation_offload_forward_backward_matches_eager(monkeypatch, fp32_parity):
    # The manager only moves CUDA tensors. On CPU, treating CPU tensors as eligible, with a zero size threshold
    # and nothing kept resident, sends the tensors saved at the checkpoint boundaries through the manager's
    # copy-out and copy-back path, which is the part that has to handle the (branch output, residual) pair each
    # decoder layer passes on.
    monkeypatch.setattr(ActivationOffloadManager, "_eligible", _eligible_on_any_device)
    reference_model = _tiny_model(fp32_parity=fp32_parity)
    offloaded_model = copy.deepcopy(reference_model)
    offload_config = dict(
        enabled=True, keep_last_n=0, use_streams=False, tensor_size_threshold=0, pin_memory_enabled=False
    )
    apply_ac(offloaded_model, ActivationCheckpointConfig(mode="full", offload_config=offload_config))
    manager = offloaded_model._activation_offload_manager

    expected_output, expected_grads = _forward_backward(reference_model)
    # The installed wrapper sits on the causal-LM forward, which these tests bypass to read the backbone output,
    # so the backbone forward runs inside the same hooks the wrapper would enter.
    output, grads = _forward_backward(offloaded_model, forward_context=manager.step_hooks)

    assert manager.stats.offloaded_tensors > 0
    assert manager.stats.restored_tensors == manager.stats.offloaded_tensors
    _assert_same_run(output, grads, expected_output, expected_grads)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

_EXPECTED_STATE_DICT = [
    ("model.embed_tokens.weight", (32, 16)),
    ("model.layers.0.linear_attn.dt_bias", (2,)),
    ("model.layers.0.linear_attn.A_log", (2,)),
    ("model.layers.0.linear_attn.conv1d.weight", (24, 1, 2)),
    ("model.layers.0.linear_attn.norm.weight", (4,)),
    ("model.layers.0.linear_attn.out_proj.weight", (16, 8)),
    ("model.layers.0.linear_attn.in_proj_qkv.weight", (24, 16)),
    ("model.layers.0.linear_attn.in_proj_z.weight", (8, 16)),
    ("model.layers.0.linear_attn.in_proj_b.weight", (2, 16)),
    ("model.layers.0.linear_attn.in_proj_a.weight", (2, 16)),
    ("model.layers.0.mlp.experts.w1", (4, 8, 16)),
    ("model.layers.0.mlp.experts.w2", (4, 16, 8)),
    ("model.layers.0.mlp.experts.w3", (4, 8, 16)),
    ("model.layers.0.mlp.router.gate.weight", (4, 16)),
    ("model.layers.0.shared_expert.w1.weight", (8, 16)),
    ("model.layers.0.shared_expert.w2.weight", (16, 8)),
    ("model.layers.0.shared_expert.w3.weight", (8, 16)),
    ("model.layers.0.shared_expert_gate.weight", (1, 16)),
    ("model.layers.0.input_layernorm.weight", (16,)),
    ("model.layers.0.post_attention_layernorm.weight", (16,)),
    ("model.layers.1.self_attn.q_proj.weight", (64, 16)),
    ("model.layers.1.self_attn.k_proj.weight", (16, 16)),
    ("model.layers.1.self_attn.v_proj.weight", (16, 16)),
    ("model.layers.1.self_attn.o_proj.weight", (16, 32)),
    ("model.layers.1.self_attn.q_norm.weight", (16,)),
    ("model.layers.1.self_attn.k_norm.weight", (16,)),
    ("model.layers.1.mlp.experts.w1", (4, 8, 16)),
    ("model.layers.1.mlp.experts.w2", (4, 16, 8)),
    ("model.layers.1.mlp.experts.w3", (4, 8, 16)),
    ("model.layers.1.mlp.router.gate.weight", (4, 16)),
    ("model.layers.1.shared_expert.w1.weight", (8, 16)),
    ("model.layers.1.shared_expert.w2.weight", (16, 8)),
    ("model.layers.1.shared_expert.w3.weight", (8, 16)),
    ("model.layers.1.shared_expert_gate.weight", (1, 16)),
    ("model.layers.1.input_layernorm.weight", (16,)),
    ("model.layers.1.post_attention_layernorm.weight", (16,)),
    ("model.norm.weight", (16,)),
    ("lm_head.weight", (32, 16)),
]

_EXPECTED_BUFFERS = [
    ("model.layers.0.mlp.tokens_per_expert", (4,)),
    ("model.layers.1.mlp.tokens_per_expert", (4,)),
    ("model.rotary_emb.inv_freq", (2,)),
]


def test_state_dict_and_buffers_match_the_expected_structure():
    config = _tiny_config(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"])
    model = qwen.Qwen3_5MoeForCausalLM(config).to(torch.bfloat16)

    state_dict = [(name, tuple(tensor.shape)) for name, tensor in model.state_dict().items()]
    buffers = [(name, tuple(tensor.shape)) for name, tensor in model.named_buffers()]

    assert state_dict == _EXPECTED_STATE_DICT
    assert buffers == _EXPECTED_BUFFERS
    assert {tensor.dtype for tensor in model.state_dict().values()} == {torch.bfloat16}
    for module in model.modules():
        if isinstance(module, qwen.Qwen3_5MoeRMSNorm):
            assert [name for name, _ in module.named_parameters()] == ["weight"]
            assert list(module.named_buffers()) == []


# ---------------------------------------------------------------------------
# fp32_parity off: residual adds, rotary embedding and beta gate in the input dtype
# ---------------------------------------------------------------------------


def _input_dtype_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _input_dtype_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """The partial RoPE in the input dtype, written out as the reference for ``fp32_parity`` off."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_input_dtype_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_input_dtype_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _input_dtype_decoder_layer(layer, hidden_states, position_embeddings):
    """The decoder layer forward with both residual adds in the input dtype, inside the layer."""
    residual = hidden_states
    hidden_states = _input_dtype_rms_norm(hidden_states, layer.input_layernorm.weight, layer.input_layernorm.eps)
    if layer.layer_type == "linear_attention":
        hidden_states = layer.linear_attn(hidden_states)
    else:
        hidden_states, _ = layer.self_attn(hidden_states=hidden_states, position_embeddings=position_embeddings)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = _input_dtype_rms_norm(
        hidden_states, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
    )
    routed_output = layer.mlp(hidden_states)
    bs, slen, dim = hidden_states.shape
    hidden_flat = hidden_states.view(-1, dim)
    shared_output = layer.shared_expert(hidden_flat)
    shared_output = qwen._shared_expert_gate(hidden_flat, layer.shared_expert_gate) * shared_output
    shared_output = shared_output.view(bs, slen, dim)
    return residual + routed_output + shared_output


def _input_dtype_backbone(model: qwen.Qwen3_5MoeForCausalLM, input_ids, position_ids) -> torch.Tensor:
    backbone = model.model
    hidden_states = backbone.embed_tokens(input_ids)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
    for layer in backbone.layers:
        hidden_states = _input_dtype_decoder_layer(layer, hidden_states, position_embeddings)
    return _input_dtype_rms_norm(hidden_states, backbone.norm.weight, backbone.norm.eps)


def _input_dtype_rope_attention(module: nn.Module, monkeypatch) -> None:
    """Make ``module`` rotate with the reference RoPE, so the reference shares none of the switch's code."""
    monkeypatch.setattr(
        module, "_rotate", lambda query, key, cos, sin: _input_dtype_apply_rotary_pos_emb(query, key, cos, sin)
    )


def test_input_dtype_mode_rope_matches_the_reference_forward_and_backward():
    set_seed(0)
    attention = qwen.Qwen3_5MoeGatedSDPAAttention(
        qwen.Qwen3_5MoeGatedAttentionConfig(
            hidden_size=16,
            head_dim=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=EPS,
            fp32_parity=False,
        )
    )
    query = torch.randn(1, 2, 6, 16, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, 1, 6, 16, dtype=torch.bfloat16, requires_grad=True)
    cos = torch.randn(1, 6, 4, dtype=torch.bfloat16)
    sin = torch.randn(1, 6, 4, dtype=torch.bfloat16)
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()

    rotated = attention._rotate(query, key, cos, sin)
    expected = _input_dtype_apply_rotary_pos_emb(query_ref, key_ref, cos, sin)
    for actual, reference in zip(rotated, expected):
        assert actual.dtype == torch.bfloat16
        torch_assert_equal(actual, reference)
    (rotated[0].float().sum() + rotated[1].float().pow(2).sum()).backward()
    (expected[0].float().sum() + expected[1].float().pow(2).sum()).backward()
    torch_assert_equal(query.grad, query_ref.grad)
    torch_assert_equal(key.grad, key_ref.grad)


def test_input_dtype_mode_gated_delta_net_beta_is_in_the_input_dtype():
    set_seed(0)
    module = qwen.Qwen3_5MoeGatedDeltaNet(_tiny_config(fp32_parity=False)).to(torch.bfloat16)
    captured = {}

    def convolution(*, x, weight, bias, activation, seq_idx):
        return F.silu(x)

    def gated_delta_rule(query, key, value, *, g, beta, **kwargs):
        captured.update(beta=beta)
        return torch.zeros_like(value), None

    module._causal_conv1d_fn = convolution
    module._chunk_gated_delta_rule = gated_delta_rule
    hidden_states = torch.randn(1, 6, 16).to(torch.bfloat16)

    module(hidden_states)

    b = module.in_proj_b(hidden_states)
    assert captured["beta"].dtype == torch.bfloat16
    torch_assert_equal(captured["beta"], b.sigmoid())


@pytest.mark.parametrize("fp32_beta", [True, False])
def test_head_parallel_gated_delta_net_beta_follows_its_argument(single_rank_group, monkeypatch, fp32_beta):
    monkeypatch.setattr(sp_gated_delta_net, "sequence_head_all_to_all", lambda group, tensor, **kwargs: tensor)
    set_seed(0)
    module = qwen.Qwen3_5MoeGatedDeltaNet(_tiny_config()).to(torch.bfloat16)
    hidden_states = torch.randn(1, 6, 16).to(torch.bfloat16)
    query, key, value = torch.split(module.in_proj_qkv(hidden_states), [8, 8, 8], dim=-1)
    b = module.in_proj_b(hidden_states)
    a = module.in_proj_a(hidden_states)
    captured = {}

    def gated_delta_rule(query, key, value, g, beta, **kwargs):
        captured.update(beta=beta)
        return torch.zeros_like(value), None

    sp_gated_delta_net.head_parallel_gated_delta_net(
        lambda *, x, weight, bias, activation, seq_idx: F.silu(x),
        gated_delta_rule,
        query.reshape(1, 6, 2, 4),
        key.reshape(1, 6, 2, 4),
        value.reshape(1, 6, 2, 4),
        b,
        a,
        convolution_weight=module.conv1d.weight.squeeze(1),
        convolution_bias=module.conv1d.bias,
        convolution_activation=module.activation,
        A_log=module.A_log,
        dt_bias=module.dt_bias,
        process_group=single_rank_group,
        global_cu_seqlens=torch.tensor([0, 6], dtype=torch.int32),
        num_key_heads=2,
        num_value_heads=2,
        fp32_beta=fp32_beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    expected = b.float().sigmoid() if fp32_beta else b.sigmoid()
    assert captured["beta"].dtype == expected.dtype
    torch_assert_equal(captured["beta"], expected)


@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "input-dtype"])
def test_sequence_parallel_gated_delta_net_beta_follows_the_switch(single_rank_group, monkeypatch, fp32_parity):
    # With one rank the all-to-all is the identity; the layer reaches head_parallel_gated_delta_net through cp_group.
    monkeypatch.setattr(sp_gated_delta_net, "sequence_head_all_to_all", lambda group, tensor, **kwargs: tensor)
    set_seed(0)
    module = qwen.Qwen3_5MoeGatedDeltaNet(_tiny_config(fp32_parity=fp32_parity)).to(torch.bfloat16)
    captured = {}

    def gated_delta_rule(query, key, value, g, beta, **kwargs):
        captured.update(beta=beta)
        return torch.zeros_like(value), None

    module._causal_conv1d_fn = lambda *, x, weight, bias, activation, seq_idx: F.silu(x)
    module._chunk_gated_delta_rule = gated_delta_rule
    module.cp_group = single_rank_group
    module._dss_sp_global_cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
    hidden_states = torch.randn(1, 6, 16).to(torch.bfloat16)

    module(hidden_states)

    b = module.in_proj_b(hidden_states)
    expected = b.float().sigmoid() if fp32_parity else b.sigmoid()
    assert captured["beta"].dtype == expected.dtype
    torch_assert_equal(captured["beta"], expected)


def test_input_dtype_mode_matches_the_reference_forward_and_backward(monkeypatch):
    model = _tiny_model(fp32_parity=False)
    reference = copy.deepcopy(model)
    for layer in reference.model.layers:
        if layer.layer_type == "full_attention":
            _input_dtype_rope_attention(layer.self_attn, monkeypatch)
    input_ids, position_ids = _tiny_inputs()

    output = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state
    expected = _input_dtype_backbone(reference, input_ids, position_ids)
    torch_assert_equal(output, expected)

    projection = torch.linspace(-1.0, 1.0, output.numel()).view_as(output)
    (output.float() * projection).sum().backward()
    (expected.float() * projection).sum().backward()
    for (name, parameter), (_, reference_parameter) in zip(
        model.model.named_parameters(), reference.model.named_parameters()
    ):
        assert parameter.grad is not None, name
        torch_assert_equal(parameter.grad, reference_parameter.grad, msg=name)


def _layer_boundaries(model: qwen.Qwen3_5MoeForCausalLM, monkeypatch) -> list[tuple[int, list | None]]:
    """For each decoder layer call, the number of tensors it receives and what it returns as the residual."""
    calls = []
    for layer in model.model.layers:
        original_forward = layer.forward

        def recording_forward(hidden_states, residual=None, *args, _forward=original_forward, **kwargs):
            outputs = _forward(hidden_states, residual, *args, **kwargs)
            received = sum(torch.is_tensor(tensor) for tensor in (hidden_states, residual))
            calls.append((received, outputs))
            return outputs

        monkeypatch.setattr(layer, "forward", recording_forward)
    return calls


# fp32 weights under bf16 autocast: the residual leaves each norm in the dtype of the branch output.
_AUTOCAST_DTYPES = {
    True: ([(torch.bfloat16, torch.bfloat16)] * 3, torch.bfloat16),
    False: ([(torch.float32, None)] * 3, torch.float32),
}


@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "input-dtype"])
def test_residual_dtype_with_fp32_weights_under_bf16_autocast(monkeypatch, fp32_parity):
    model = _tiny_model(dtype=torch.float32, fp32_parity=fp32_parity)
    calls = _layer_boundaries(model, monkeypatch)
    input_ids, position_ids = _tiny_inputs()

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state

    layer_dtypes, final_dtype = _AUTOCAST_DTYPES[fp32_parity]
    assert [
        (hidden_states.dtype, None if residual is None else residual.dtype) for _, (hidden_states, residual) in calls
    ] == layer_dtypes
    assert output.dtype == final_dtype


@pytest.mark.parametrize(("fp32_parity", "tensors"), [(True, [1, 2, 2]), (False, [1, 1, 1])])
def test_tensors_passed_between_decoder_layers(monkeypatch, fp32_parity, tensors):
    model = _tiny_model(fp32_parity=fp32_parity)
    calls = _layer_boundaries(model, monkeypatch)
    input_ids, position_ids = _tiny_inputs()

    with torch.no_grad():
        output = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state

    assert [received for received, _ in calls] == tensors
    for _, (hidden_states, residual) in calls:
        assert hidden_states.dtype == torch.bfloat16
        assert (residual is None) is not fp32_parity
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize(("fp32_parity", "tensors"), [(True, [1, 2, 2]), (False, [1, 1, 1])])
def test_full_checkpointing_boundary_holds_these_hidden_tensors_per_layer(fp32_parity, tensors):
    model = _tiny_model(fp32_parity=fp32_parity)
    apply_ac(model, ActivationCheckpointConfig(mode="full"))
    hidden_size = model.config.hidden_size
    held = []
    for layer in model.model.layers:
        # A non-reentrant checkpoint keeps the arguments it is handed and recomputes the layer from them.
        checkpoint_fn = layer.checkpoint_fn

        def recording_checkpoint_fn(module, *args, _checkpoint_fn=checkpoint_fn, **kwargs):
            held.append(sum(torch.is_tensor(arg) and arg.shape == (1, 8, hidden_size) for arg in args))
            return _checkpoint_fn(module, *args, **kwargs)

        layer.checkpoint_fn = recording_checkpoint_fn
    input_ids, position_ids = _tiny_inputs()

    model.model(input_ids=input_ids, position_ids=position_ids)

    # Each layer after the first is also handed the residual when fp32_parity is on.
    assert held == tensors


def test_state_dict_is_the_same_in_both_modes():
    models = {}
    for fp32_parity in (True, False):
        config = _tiny_config(
            num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], fp32_parity=fp32_parity
        )
        models[fp32_parity] = qwen.Qwen3_5MoeForCausalLM(config).to(torch.bfloat16)
        state_dict = models[fp32_parity].state_dict()
        assert [(name, tuple(tensor.shape)) for name, tensor in state_dict.items()] == _EXPECTED_STATE_DICT
        assert {tensor.dtype for tensor in state_dict.values()} == {torch.bfloat16}
    # A checkpoint written in one mode loads strictly in the other.
    models[False].load_state_dict(models[True].state_dict(), strict=True)
    models[True].load_state_dict(models[False].state_dict(), strict=True)


def test_fp32_parity_option_reaches_the_model_config():
    from arctic_platform.model.implementations.qwen35.deepspeed_integration import _build_model_config
    from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

    assert Qwen3_5MoeOptions().fp32_parity is True
    for fp32_parity in (True, False):
        options = Qwen3_5MoeOptions.model_validate(dict(fp32_parity=fp32_parity))
        assert _build_model_config("tiny", 2, 1, "bfloat16", "sdpa", options).fp32_parity is fp32_parity


_HF_TEXT_CONFIG = dict(
    vocab_size=32,
    hidden_size=16,
    num_hidden_layers=3,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_conv_kernel_dim=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    num_experts=4,
    num_experts_per_tok=2,
    layer_types=["linear_attention", "full_attention", "linear_attention"],
    pad_token_id=0,
)


@pytest.mark.parametrize("checkpoint", ["composite", "text-only"])
@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "input-dtype"])
def test_get_model_hands_fp32_parity_to_every_layer(tmp_path, checkpoint, fp32_parity):
    from transformers import Qwen3_5MoeConfig as HFQwen3_5MoeConfig
    from transformers import Qwen3_5MoeTextConfig as HFQwen3_5MoeTextConfig

    from arctic_platform.model.implementations.qwen35.config import ModelConfig
    from arctic_platform.model.implementations.qwen35.model_builder import get_model

    if checkpoint == "composite":
        vision = dict(depth=1, hidden_size=16, num_heads=2, intermediate_size=16, out_hidden_size=16)
        hf_config = HFQwen3_5MoeConfig(text_config=_HF_TEXT_CONFIG, vision_config=vision)
    else:
        hf_config = HFQwen3_5MoeTextConfig(**_HF_TEXT_CONFIG)
    hf_config.save_pretrained(tmp_path)
    model_config = ModelConfig(
        name=str(tmp_path), impl="custom", attn="sdpa", moe_use_grouped_mm=False, fp32_parity=fp32_parity
    )

    model = get_model(model_config, device=torch.device("meta"), dtype=torch.bfloat16)

    layers = [module for module in model.modules() if isinstance(module, qwen.Qwen3_5MoeDecoderLayer)]
    assert [layer.layer_type for layer in layers] == _HF_TEXT_CONFIG["layer_types"]
    for layer in layers:
        assert layer.fp32_parity is fp32_parity
        mixer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        assert mixer.fp32_parity is fp32_parity
