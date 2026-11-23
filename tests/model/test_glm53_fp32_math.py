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
"""GLM-5.3-Flash computes in fp32 where the vLLM sampler does, with bf16 (DeepSpeed-cast) parameters.

Router logits (``moe_router_dtype: float32``; transformers' reference also uses fp32), the mHC remix (vLLM ``mhc_post``:
fp32, one rounding), the KDA beta gate and Qwen3.8-Flash-Next's GatedDeltaNet beta gate at both SP paths (vLLM sigmoids
the bf16 logit in fp32), and the router-score gradient through the EP dispatch backward. The forward sites also run
under bf16 autocast, as DeepSpeed's ``torch_autocast`` does, and keep the transformers rounding with ``fp32_parity`` off.
"""

import importlib
import sys
import types
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_glm5_next import _tiny_config


def _bf16_model(hc_mult=2, fp32_parity=True):
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    config = _tiny_config()
    config.text_config.hc_mult = hc_mult
    config.use_grouped_mm = False
    config.fp32_parity = fp32_parity
    set_seed(0)
    model = Glm5NextForConditionalGenerationPrimeRL(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.5)
    return model.to(torch.bfloat16)  # DeepSpeed's bf16 parameter cast


def _autocast(enabled):
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("fp32_parity", [True, False])
def test_glm53_router_gating_weights_use_fp32_logits(monkeypatch, fp32_parity, autocast):
    histc = torch.histc  # CPU has no integer histc kernel; the counts are not under test
    monkeypatch.setattr(torch, "histc", lambda values, *args, **kwargs: histc(values.float(), *args, **kwargs))
    router = _bf16_model(fp32_parity=fp32_parity).model.language_model.layers[3].mlp.router
    x = torch.randn(512, 16).to(torch.bfloat16)
    with torch.no_grad():
        logits = F.linear(x.float(), router.gate.weight.float()) if fp32_parity else router.gate(x).float()
    routed = logits.topk(router.top_k, dim=-1).indices

    with torch.no_grad(), _autocast(autocast):
        scores, _, _ = router(x, routed_experts=routed)

    expected = torch.sigmoid(logits).gather(1, routed)
    expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-20) * router.route_scale
    torch_assert_close(scores, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
def test_glm53_hyper_connection_remix_rounds_once_from_fp32(autocast):
    model = _bf16_model(hc_mult=4)
    layer = model.model.language_model.layers[0]
    attention_output = torch.randn(1, 64, 16).to(torch.bfloat16)
    mlp_output = torch.randn(1, 64, 16).to(torch.bfloat16)
    layer.self_attn.forward = lambda hidden_states, **_kwargs: attention_output
    layer.mlp.forward = lambda hidden_states, **_kwargs: mlp_output
    streams = torch.randn(1, 64, 4, 16).to(torch.bfloat16)

    def remix(hyper_connection, output, residual):
        post, comb, _ = hyper_connection(residual)
        mixed = post.double().unsqueeze(-1) * output.double().unsqueeze(-2)
        return (mixed + comb.double().transpose(-1, -2) @ residual.double()).to(torch.bfloat16)

    # Under autocast ``post`` comes out of the hyper-connection in bf16; the remix must still accumulate in fp32.
    with torch.no_grad(), _autocast(autocast):
        out, _ = layer(streams)
        expected = remix(layer.ffn_hc, mlp_output, remix(layer.attn_hc, attention_output, streams))

    # fp32 accumulation may differ from fp64 by one bf16 rounding near a tie, never more.
    ulp = torch.finfo(torch.bfloat16).eps * expected.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    ulps = (out.float() - expected.float()).abs() / ulp
    differ = (out != expected).float().mean()
    assert ulps.max() <= 1 and differ < 0.01, f"max {ulps.max():.1f} bf16 ulps off fp64; {differ:.1%} of values differ"


def test_glm53_hyper_connection_remix_keeps_the_reference_rounding_without_fp32_parity():
    model = _bf16_model(hc_mult=4, fp32_parity=False)
    layer = model.model.language_model.layers[0]
    attention_output = torch.randn(1, 64, 16).to(torch.bfloat16)
    mlp_output = torch.randn(1, 64, 16).to(torch.bfloat16)
    layer.self_attn.forward = lambda hidden_states, **_kwargs: attention_output
    layer.mlp.forward = lambda hidden_states, **_kwargs: mlp_output
    streams = torch.randn(1, 64, 4, 16).to(torch.bfloat16)

    def remix(hyper_connection, output, residual):
        post, comb, _ = hyper_connection(residual)
        mixed = post.to(torch.bfloat16).unsqueeze(-1) * output.unsqueeze(-2)
        return mixed + torch.matmul(comb.to(torch.bfloat16).transpose(-1, -2), residual)

    with torch.no_grad():
        out, _ = layer(streams)
        expected = remix(layer.ffn_hc, mlp_output, remix(layer.attn_hc, attention_output, streams))

    torch_assert_equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the fused mix kernel")
def test_glm53_remix_with_trainable_hyper_connections_uses_the_fused_kernel_on_cuda():
    from arctic_platform.model.implementations.glm53.hyper_connections import mix_streams
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import _remix

    set_seed(0)
    streams = torch.randn(2, 256, 4, 512, device="cuda").to(torch.bfloat16)
    output = torch.randn(2, 256, 512, device="cuda").to(torch.bfloat16)
    post = (2 * torch.rand(2, 256, 4, device="cuda")).requires_grad_(True)
    comb = torch.softmax(torch.randn(2, 256, 4, 4, device="cuda"), dim=-1).requires_grad_(True)
    grad = torch.randn_like(streams)

    with patch(
        "arctic_platform.model.implementations.glm53.modeling_glm5_next.mix_streams", wraps=mix_streams
    ) as fused:
        out = _remix(False, True, post, comb, output, streams)
    assert fused.call_count == 1
    leaves = (post, comb)
    grads = torch.autograd.grad(out, leaves, grad)

    reference_leaves = [tensor.detach().double().requires_grad_(True) for tensor in leaves]
    reference = reference_leaves[0].unsqueeze(-1) * output.double().unsqueeze(-2) + torch.matmul(
        reference_leaves[1].transpose(-1, -2), streams.double()
    )
    reference_grads = torch.autograd.grad(reference, reference_leaves, grad.double())
    assert (out != reference.to(torch.bfloat16)).float().mean() < 0.01
    for actual, expected in zip(grads, reference_grads):
        torch_assert_close(actual.double(), expected, rtol=1e-5, atol=1e-5)


def _record_kda_beta(captured):
    def chunk_kda(query, key, value, g, beta, **_kwargs):
        captured.append(beta)
        return torch.zeros_like(value), None

    fla, ops, kda = (types.ModuleType(name) for name in ("fla", "fla.ops", "fla.ops.kda"))
    kda.chunk_kda = chunk_kda
    fla.ops, ops.kda = ops, kda
    return {"fla": fla, "fla.ops": ops, "fla.ops.kda": kda}


@pytest.mark.parametrize("fp32_parity", [True, False])
def test_glm53_kda_beta_gate_is_an_fp32_sigmoid(tiny_glm_sparse_model, single_rank_group, fp32_parity):
    from arctic_platform.model.implementations.glm53 import context_parallel as cp
    from tests.model.test_glm53_packed_rows import _unfused_causal_conv1d_module

    set_seed(0)
    attention = tiny_glm_sparse_model(with_linear_attention=True).model.layers[1].self_attn
    hidden_states = torch.randn(1, 8, attention.b_proj.in_features).to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    setattr(attention, cp._GLOBAL_CU_SEQLENS, cu_seqlens)
    attention._cp_group, attention._cp_world_size = single_rank_group, 1
    attention._fp32_parity = fp32_parity
    captured = []
    modules = {**_record_kda_beta(captured), "causal_conv1d": _unfused_causal_conv1d_module(cu_seqlens)}

    with patch.dict(sys.modules, modules), torch.no_grad():
        cp._local_linear_attention_forward(attention, hidden_states)
        cp._linear_attention_forward(attention, hidden_states)
        logits = attention.b_proj(hidden_states)
        expected = torch.sigmoid(logits.float() if fp32_parity else logits)

    assert len(captured) == 2
    for beta in captured:
        torch_assert_equal(beta.reshape(expected.shape), expected)


def test_glm53_kda_keeps_normalized_qk_in_fp32_like_the_sampler_decode(tiny_glm_sparse_model, single_rank_group):
    for enabled in (False, True):
        _check_kda_operands(tiny_glm_sparse_model, single_rank_group, enabled)


def _check_kda_operands(tiny_glm_sparse_model, single_rank_group, fp32_kda):
    """Check the operands and normalization flag FLA actually receives, the result at o_norm, and the gradient
    through the post-convolution q/k/v cast on both GLM forwards."""
    from arctic_platform.model.implementations.glm53 import context_parallel as cp
    from tests.model.test_glm53_packed_rows import _unfused_causal_conv1d_module

    set_seed(0)
    attention = tiny_glm_sparse_model(with_linear_attention=True).model.layers[1].self_attn
    hidden_states = torch.randn(1, 6, attention.b_proj.in_features).to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    setattr(attention, cp._GLOBAL_CU_SEQLENS, cu_seqlens)
    attention._cp_group, attention._cp_world_size, attention._fp32_kda = single_rank_group, 1, fp32_kda
    calls, convolved, outputs = [], [], []
    convolution = _unfused_causal_conv1d_module(cu_seqlens)
    convolve = convolution.causal_conv1d_fn

    def record_conv(*args, **kwargs):
        result = convolve(*args, **kwargs)
        convolved.append(result)
        return result

    def chunk_kda(q, k, v, *, cu_seqlens, use_qk_l2norm_in_kernel, **kwargs):
        calls.append(((q, k, v), cu_seqlens, use_qk_l2norm_in_kernel))
        return q + k * v, None

    convolution.causal_conv1d_fn = record_conv
    attention.o_norm.register_forward_pre_hook(lambda module, args: outputs.append(args[0]))
    modules = {"fla.ops.kda": types.SimpleNamespace(chunk_kda=chunk_kda), "causal_conv1d": convolution}
    with patch.dict(sys.modules, modules):
        cp._local_linear_attention_forward(attention, hidden_states)
        cp._linear_attention_forward(attention, hidden_states)

    assert len(outputs) == 2
    for mixed, (operands, boundaries, normalize_in_kernel), output in zip(convolved, calls, outputs, strict=True):
        reference = mixed.detach().to(torch.float32 if fp32_kda else mixed.dtype).requires_grad_(fp32_kda)
        qkv = reference.transpose(1, 2).unflatten(-1, (3, attention.num_heads, attention.head_dim))
        q, k, v = qkv.unbind(2)
        if fp32_kda:
            q, k = (x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k))
        for actual, expected_operand in zip(operands, (q, k, v), strict=True):
            torch_assert_equal(actual, expected_operand)
        assert normalize_in_kernel is (not fp32_kda)
        torch_assert_equal(boundaries, cu_seqlens)
        expected = (q + k * v).to(mixed.dtype)
        torch_assert_equal(output, expected)
        if fp32_kda:
            grad = torch.randn_like(output)
            (actual_grad,) = torch.autograd.grad(output, mixed, grad)
            (reference_grad,) = torch.autograd.grad(expected, reference, grad)
            torch_assert_equal(actual_grad, reference_grad.to(mixed.dtype))


def test_glm53_loader_passes_fp32_kda_to_attention(tmp_path):
    from arctic_platform.model import ModelSpec
    from arctic_platform.model.implementations.glm53.deepspeed_integration import _adapter
    from arctic_platform.model.loaders.flash_moe import _FlashLoadOptions

    _tiny_config().save_pretrained(tmp_path)
    adapter = _adapter()
    for loader_options in ({"fp32_kda": True}, {}):
        spec = ModelSpec(model_path_or_name=str(tmp_path), loader="glm5_next", loader_options=loader_options)
        options = _FlashLoadOptions.model_validate(spec.loader_options)
        config = adapter.build_model_config(str(tmp_path), 1, 1, "bfloat16", spec.attn_implementation, options)
        model = adapter.get_model(config, torch.device("meta"), torch.bfloat16)
        adapter.apply_sequence_parallelism(model, 1, None)
        assert model.model.language_model.layers[0].self_attn._fp32_kda is bool(loader_options)
    with pytest.raises(ValueError, match="fp32_kda"):
        ModelSpec(model_path_or_name=str(tmp_path), loader="qwen4_exp", loader_options={"fp32_kda": True})


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("fp32_parity", [True, False])
def test_qwen38_gated_delta_net_beta_is_the_same_at_every_sequence_parallel_size(monkeypatch, fp32_parity, autocast):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as stock

    from arctic_platform.model.implementations.qwen38 import context_parallel as cp
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL
    from tests.model.test_qwen4_exp import _tiny_config

    set_seed(0)
    config = _tiny_config()
    config.fp32_parity = fp32_parity
    module = Qwen4ExpForConditionalGenerationPrimeRL(config).to(torch.bfloat16).model.language_model.layers[0]
    module = module.linear_attn
    captured = []

    def chunk_gated_delta_rule(query, key, value, g, beta, **_kwargs):
        captured.append(beta)
        return torch.zeros_like(value), None

    monkeypatch.setattr(stock, "torch_chunk_gated_delta_rule", chunk_gated_delta_rule)
    fla, modules, conv, ops, rule = (
        types.ModuleType(name)
        for name in ("fla", "fla.modules", "fla.modules.conv", "fla.ops", "fla.ops.gated_delta_rule")
    )
    conv.causal_conv1d = lambda x, **_kwargs: (x, None)
    rule.chunk_gated_delta_rule = chunk_gated_delta_rule
    fla.modules, modules.conv, fla.ops, ops.gated_delta_rule = modules, conv, ops, rule
    fakes = {"fla": fla, "fla.modules": modules, "fla.modules.conv": conv, "fla.ops": ops}
    hidden_states = torch.randn(1, 8, config.text_config.hidden_size).to(torch.bfloat16)
    with patch.dict(sys.modules, {**fakes, "fla.ops.gated_delta_rule": rule}), torch.no_grad(), _autocast(autocast):
        module(hidden_states)  # sp 1: transformers' stock forward
        cp._adapt_linear_attention(module, None)
        setattr(module, cp._FLA_CP_CONTEXT, object())
        module(hidden_states)  # sp > 1: the context-parallel forward
    logits = F.linear(hidden_states, module.in_proj_b.weight)
    expected = torch.sigmoid(logits.float() if fp32_parity else logits)

    assert len(captured) == 2
    for beta in captured:
        torch_assert_equal(beta, expected)


def _fake_ep_runtime():
    class Stub:
        def __init__(self, *args, **kwargs):
            pass

    uccl, uccl_ep, deep_ep, deep_ep_utils = (
        types.ModuleType(name) for name in ("uccl", "uccl.ep", "deep_ep", "deep_ep.utils")
    )
    uccl_ep.EventHandle = uccl_ep.Config = deep_ep.Buffer = Stub
    deep_ep_utils.EventHandle = deep_ep_utils.EventOverlap = Stub
    uccl.ep, deep_ep.utils = uccl_ep, deep_ep_utils
    return {"uccl": uccl, "uccl.ep": uccl_ep, "deep_ep": deep_ep, "deep_ep.utils": deep_ep_utils}


def _import_ep_backend(backend):
    name = f"arctic_platform.model.implementations.moe.distributed.{backend}"
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    with patch.dict(sys.modules, _fake_ep_runtime()):
        module = importlib.import_module(name)
    # Importing registers process-wide custom ops, so a second import of the module would fail.
    sys.modules[name] = module
    return module


@pytest.mark.parametrize("backend", ["ucclep", "deepep"])
def test_ep_dispatch_backward_keeps_the_router_score_gradient_fp32(backend):
    class Event:
        def current_stream_wait(self):
            pass

    class Buffer:
        def combine(self, x, handle, topk_weights, **_kwargs):
            return x, topk_weights, Event()

    grad_scores = torch.tensor([[0.1234567, -2.3456789], [3.1415927, 1e-3]])
    module = _import_ep_backend(backend)
    with patch.object(module, "_buffer", Buffer()), patch.object(module, "_new_event_overlap", lambda: None):
        context = types.SimpleNamespace(input_dtype=torch.bfloat16, saved_handle=object())
        gradients = module._dispatch_backward(
            context, torch.ones(2, 4, dtype=torch.bfloat16), None, grad_scores, None, None
        )

    torch_assert_equal(gradients[2], grad_scores)
