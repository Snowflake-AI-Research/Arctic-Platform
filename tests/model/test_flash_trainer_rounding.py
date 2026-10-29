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
"""GLM-5.3-Flash norms, Qwen3.8-Flash-Next QSA attention and both models' routed experts round to bf16 once, from
fp32, as vLLM's kernels do.

Each test runs the trainer's own modules with bf16 parameters (DeepSpeed's cast) and checks one site against the
fp64 value of the same formula from the same bf16 inputs, rounded once. With ``fp32_parity`` off, the sites keep
the transformers order.
"""

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.model import loader as loader_mod
from tests.model.test_glm5_next import _tiny_config as _glm_tiny_config


def _assert_rounded_once(actual: torch.Tensor, reference: torch.Tensor) -> None:
    """``actual`` is ``reference`` (fp64) rounded once, except where fp32 noise could flip that rounding."""
    expected = reference.to(actual.dtype)
    ambiguous = (reference * (1 - 1e-5)).to(actual.dtype) != (reference * (1 + 1e-5)).to(actual.dtype)
    assert ambiguous.double().mean() < 0.01
    mismatched = int((actual != expected)[~ambiguous].sum())
    assert mismatched == 0, f"{mismatched} of {int((~ambiguous).sum())} values were rounded more than once"


@pytest.mark.parametrize("name", ["glm5_next", "qwen4_exp"])
def test_flash_loaders_take_fp32_parity(name):
    options = loader_mod._LOADERS[name].options
    assert options().fp32_parity is True
    assert options.model_validate({"fp32_parity": False}).fp32_parity is False


def _glm_norms(fp32_parity: bool) -> list:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm

    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    torch.manual_seed(0)
    config = _glm_tiny_config()
    config.use_grouped_mm = False
    config.fp32_parity = fp32_parity
    model = Glm5NextForConditionalGenerationPrimeRL(config).to(torch.bfloat16)
    norms = {name: module for name, module in model.named_modules() if type(module) is Glm5NextTextRMSNorm}
    kinds = {name.rsplit(".", 1)[-1] for name in norms}
    assert kinds >= {"input_layernorm", "post_attention_layernorm", "q_a_layernorm", "kv_a_layernorm", "norm"}
    with torch.no_grad():
        for norm in norms.values():
            norm.weight.normal_(1.0, 0.5)
    return list(norms.values())


def test_glm53_rms_norms_multiply_the_weight_before_rounding():
    for norm in _glm_norms(fp32_parity=True):
        with torch.no_grad():
            hidden_states = (3 * torch.randn(256, norm.weight.numel())).to(torch.bfloat16)
            x = hidden_states.double()
            reference = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.double()
            _assert_rounded_once(norm(hidden_states), reference)


def test_glm53_rms_norms_keep_the_reference_order_without_fp32_parity():
    for norm in _glm_norms(fp32_parity=False):
        with torch.no_grad():
            hidden_states = (3 * torch.randn(256, norm.weight.numel())).to(torch.bfloat16)
            expected = type(norm).forward(norm, hidden_states)
            torch.testing.assert_close(norm(hidden_states), expected, rtol=0, atol=0)


def _qsa_sites(monkeypatch, fp32_parity: bool = True) -> dict[str, torch.Tensor]:
    """One QSA layer's forward through ``apply_qsa_flex``, with the tensors at each rounding site recorded."""
    pytest.importorskip("transformers.models.qwen4_exp")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    from arctic_platform.model.implementations.qwen38 import qsa_flex as qsa

    torch.manual_seed(7)
    config = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.5,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        layer_types=["qwen_sparse_attention"],
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "mrope_section": [2, 1, 1]},
    )
    qsa.register_qsa_flex_backend()
    config._attn_implementation = "qsa_flex"
    model = Qwen4ExpForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.5)
    model = model.to(torch.bfloat16).eval()
    qsa.apply_qsa_flex(model, fp32_parity=fp32_parity)
    attention = model.model.layers[0].self_attn

    sites: dict[str, torch.Tensor] = {}

    def attend(query, key, value, ids, softmax_scale, fp32_parity):
        output = qsa._SparseGQAAttention.apply(query, key, value, ids, softmax_scale, 128, fp32_parity)
        sites.update(query=query, key=key, value=value, ids=ids, scale=torch.tensor(softmax_scale), attention=output)
        return output

    # The real attention runs on CPU; only its CUDA/bf16 entry guard is replaced.
    monkeypatch.setattr(qsa, "flex_sparse_gqa_attention", attend)
    attention.register_forward_pre_hook(lambda _module, args: sites.update(zip(("cos", "sin"), args[1])))
    attention.q_norm.register_forward_hook(lambda _m, _a, output: sites.update(normed_query=output))
    attention.k_norm.register_forward_hook(lambda _m, _a, output: sites.update(normed_key=output))
    attention.q_proj.register_forward_hook(lambda _m, _a, output: sites.update(query_and_gate=output))
    attention.o_proj.register_forward_pre_hook(lambda _m, args: sites.update(gated=args[0]))

    ids = torch.randint(1, config.vocab_size, (2, 32))
    with torch.no_grad():
        model(input_ids=ids, position_ids=torch.arange(32)[None] + 5, use_cache=False)
    return sites


def _rotated(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    x = x.double()
    cos, sin = cos.double()[:, :, None], sin.double()[:, :, None]
    rotary, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    half = torch.cat((-rotary[..., rotary_dim // 2 :], rotary[..., : rotary_dim // 2]), dim=-1)
    return torch.cat((rotary * cos + half * sin, rest), dim=-1)


def test_qwen38_qsa_rotary_rounds_once(monkeypatch):
    sites = _qsa_sites(monkeypatch)
    _assert_rounded_once(sites["query"], _rotated(sites["normed_query"], sites["cos"], sites["sin"]))
    _assert_rounded_once(sites["key"], _rotated(sites["normed_key"], sites["cos"], sites["sin"]))


def _attention_reference(sites: dict[str, torch.Tensor]) -> torch.Tensor:
    query, key, value, ids = sites["query"].double(), sites["key"].double(), sites["value"].double(), sites["ids"]
    groups = query.shape[2] // key.shape[2]
    key, value = key.repeat_interleave(groups, dim=2), value.repeat_interleave(groups, dim=2)
    valid = (ids >= 0) & (ids < key.shape[1])
    hits = torch.zeros(ids.shape[0], ids.shape[1], key.shape[1], dtype=torch.int32)
    selected = hits.scatter_add_(-1, ids.clamp(min=0, max=key.shape[1] - 1).long(), valid.int()) > 0
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * sites["scale"].double()
    probabilities = torch.softmax(scores.masked_fill(~selected[:, None], -torch.inf), dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, value)


def _gate(sites: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, heads, head_dim = sites["attention"].shape
    gate = sites["query_and_gate"].view(batch, length, heads, 2 * head_dim)[..., head_dim:].reshape(batch, length, -1)
    return sites["attention"].reshape(batch, length, -1), gate


def test_qwen38_qsa_attention_rounds_once(monkeypatch):
    sites = _qsa_sites(monkeypatch)
    _assert_rounded_once(sites["attention"], _attention_reference(sites))


def test_qwen38_qsa_output_gate_rounds_once(monkeypatch):
    sites = _qsa_sites(monkeypatch)
    attention, gate = _gate(sites)
    _assert_rounded_once(sites["gated"], attention.double() * torch.sigmoid(gate.double()))


def test_qwen38_qsa_without_fp32_parity_rounds_like_transformers(monkeypatch):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    sites = _qsa_sites(monkeypatch, fp32_parity=False)
    query, key = apply_rotary_pos_emb(
        sites["normed_query"].transpose(1, 2), sites["normed_key"].transpose(1, 2), sites["cos"], sites["sin"]
    )
    torch.testing.assert_close(sites["query"], query.transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(sites["key"], key.transpose(1, 2), rtol=0, atol=0)
    with pytest.raises(AssertionError, match="rounded more than once"):
        _assert_rounded_once(sites["attention"], _attention_reference(sites))
    attention, gate = _gate(sites)
    torch.testing.assert_close(sites["gated"], attention * torch.sigmoid(gate), rtol=0, atol=0)


def test_qwen38_qsa_fp32_math_survives_autocast():
    from transformers import set_seed

    from arctic_platform.model.implementations.qwen38.qsa_flex import _SparseGQAAttention

    set_seed(13)
    inputs = [torch.randn(1, 8, heads, 16).bfloat16() for heads in (4, 2, 2)]
    ids = torch.arange(8, dtype=torch.int32)[None, None].expand(1, 8, 8)
    grad = torch.randn_like(inputs[0])
    results = []
    for enabled in (False, True):
        leaves = [x.detach().requires_grad_(True) for x in inputs]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
            output = _SparseGQAAttention.apply(*leaves, ids, 0.25, 4, True)
        results.append((output, *torch.autograd.grad(output, leaves, grad)))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _OneRankDispatch:
    """DeepEP/UCCL dispatch and combine with every expert on this rank; records what each combine receives."""

    def __init__(self):
        self.received: list[SimpleNamespace] = []

    @staticmethod
    def dispatch_tokens_async(hidden, indices, scores, num_experts, group, *, score_before_experts):
        assert not score_before_experts
        return hidden, indices.masked_fill(scores == 0, -1), scores.float(), num_experts

    @staticmethod
    def finalize_dispatch_tokens(pending):
        from arctic_platform.model.implementations.moe.distributed.token_permute import permute_tokens

        hidden, indices, scores, num_experts = pending
        rows, row_scores, row_tokens, counts = permute_tokens(hidden, indices, scores, num_experts)
        state = SimpleNamespace(permuted_scores=row_scores, permuted_indices=row_tokens, num_recv_tokens=len(hidden))
        state.dispatched_scores, state.counts = row_scores, counts
        return rows, counts, state

    def combine_tokens(self, rows, state):
        from arctic_platform.model.implementations.moe.distributed.token_permute import unpermute_tokens
        from arctic_platform.model.implementations.moe.routing_weights import apply_routing_weights

        if state.permuted_scores is not None:  # as deepep.combine_tokens and ucclep.combine_tokens
            rows = apply_routing_weights(rows, state.permuted_scores)
        self.received.append(SimpleNamespace(rows=rows, scores=state.dispatched_scores, counts=state.counts))
        return unpermute_tokens(rows, state.permuted_indices, state.num_recv_tokens)

    @staticmethod
    def sync_combine():
        pass


def _flash_moe(model: str, fp32_parity: bool, use_grouped_mm: bool):
    from arctic_platform.model.implementations.moe.layers.moe import MoE

    if model == "glm5_next":
        from arctic_platform.model.implementations.glm53.modeling_glm5_next import (
            Glm5NextForConditionalGenerationPrimeRL as model_cls,
        )

        config = _glm_tiny_config()
    else:
        from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import (
            Qwen4ExpForConditionalGenerationPrimeRL as model_cls,
        )
        from tests.model.test_qwen4_exp import _tiny_config as _qwen_tiny_config

        config = _qwen_tiny_config()
    config.use_grouped_mm = use_grouped_mm
    config.fp32_parity = fp32_parity
    torch.manual_seed(0)
    layer = next(module for module in model_cls(config).modules() if isinstance(module, MoE)).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.5)
    return layer


@pytest.mark.parametrize("fp32_parity", [True, False])
@pytest.mark.parametrize("use_grouped_mm", [True, False])
@pytest.mark.parametrize("model", ["glm5_next", "qwen4_exp"])
def test_flash_moe_weights_the_down_projection_accumulator_before_rounding(
    monkeypatch, model, use_grouped_mm, fp32_parity
):
    """The routed rows reaching the expert-parallel combine are ``score * (h @ w2.T)`` rounded once, as vLLM's
    fused MoE kernel stores them, through the production dispatch path with only the cross-rank exchange replaced."""
    from arctic_platform.model.implementations.moe.distributed import expert_parallel
    from arctic_platform.model.implementations.moe.layers import moe

    layer = _flash_moe(model, fp32_parity, use_grouped_mm)
    dispatch = _OneRankDispatch()
    hidden = []
    swiglu = moe._apply_swiglu

    def routed_hidden(gate, up, limit):  # the down projection's input; shared experts run after the combine
        output = swiglu(gate, up, limit)
        if not dispatch.received:
            hidden.append(output)
        return output

    histc = torch.histc
    monkeypatch.setattr(torch, "histc", lambda values, *args, **kwargs: histc(values.float(), *args, **kwargs))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(expert_parallel, "get_ep_group", lambda experts: None)
    monkeypatch.setattr(moe, "get_ep_comm_module", lambda backend: dispatch)
    monkeypatch.setattr(moe, "_apply_swiglu", routed_hidden)

    output = layer(torch.randn(1, 512, layer.experts.w2.shape[1]).to(torch.bfloat16))
    (received,) = dispatch.received
    h, w2 = torch.cat(hidden).double(), layer.experts.w2.detach().double()
    expert = torch.repeat_interleave(torch.arange(len(received.counts)), received.counts)
    reference = torch.einsum("ri,rhi->rh", h, w2[expert]) * received.scores.detach().double()[:, None]
    if fp32_parity:
        _assert_rounded_once(received.rows, reference)
    else:
        with pytest.raises(AssertionError, match="rounded more than once"):
            _assert_rounded_once(received.rows, reference)
    output.float().square().sum().backward()
    assert layer.router.gate.weight.grad.abs().sum() > 0
    assert layer.experts.w2.grad.abs().sum() > 0


def test_weighted_down_projection_backward_is_that_of_the_bf16_gemm_then_weighting():
    from arctic_platform.model.implementations.moe.routing_weights import apply_routing_weights
    from arctic_platform.model.implementations.moe.routing_weights import weighted_down_projection

    torch.manual_seed(0)
    offsets = torch.tensor([37, 37, 96], dtype=torch.int32)  # the second expert has no rows
    inputs = [torch.randn(96, 32).bfloat16(), torch.randn(3, 48, 32).bfloat16(), 2.5 * torch.rand(96)]
    grad = torch.randn(96, 48).bfloat16()
    gradients = []
    for weighted_first in (True, False):
        hidden, weight, scores = (tensor.clone().requires_grad_(True) for tensor in inputs)
        if weighted_first:
            output = weighted_down_projection(hidden, weight, offsets, scores)
        else:
            output = apply_routing_weights(torch._grouped_mm(hidden, weight.transpose(-2, -1), offs=offsets), scores)
        gradients.append(torch.autograd.grad(output, (hidden, weight, scores), grad))
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weighted_down_projection_keeps_the_fp32_accumulator_under_autocast():
    from arctic_platform.model.implementations.moe.routing_weights import weighted_down_projection

    torch.manual_seed(0)
    offsets = torch.tensor([40, 100], dtype=torch.int32)
    inputs = (torch.randn(100, 64).bfloat16(), torch.randn(2, 96, 64).bfloat16(), offsets, 2.5 * torch.rand(100))
    expected = weighted_down_projection(*inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = weighted_down_projection(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
