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
"""GLM-5.3-Flash norms and Qwen3.8-Flash-Next QSA attention round to bf16 once, from fp32, as vLLM's kernels do.

Each test runs the trainer's own modules with bf16 parameters (DeepSpeed's cast) and checks one site against the
fp64 value of the same formula from the same bf16 inputs, rounded once. With ``fp32_parity`` off, the sites keep
the transformers order.
"""

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
