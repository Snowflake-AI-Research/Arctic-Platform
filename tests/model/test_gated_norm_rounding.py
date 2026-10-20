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
"""CPU checks for the GatedDeltaNet output norm that rounds once.

The stock transformers norms round the normalized value to the input dtype before the weight multiply and again at
the output. The replacement is checked bitwise against vLLM's ``RMSNormGated`` formula, written here, for every stock
class, through the huggingface loader with and without ``fp32_parity``, and behind the sequence-parallel wrapper.
On CPU the torch formula runs and the dispatch to fla's kernel is checked with a stand-in. With CUDA and fla, fla's
kernel is checked against the fp32 formula, forward and backward, and the compiled norm is checked to be one graph.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model.implementations.gpu import vllm_rounding
from arctic_platform.model.loader import LoaderContext
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

_SEQ = 12
_HEAD_DIM = 128  # Qwen3.8-27B's GatedDeltaNet value head width


def _vllm_rms_norm_gated(x, gate, weight, eps, activation):
    """vLLM ``RMSNormGated.forward_static`` with ``norm_before_gate=True`` and no group size."""
    orig_dtype = x.dtype
    x, weight, gate = x.float(), weight.float(), gate.float()
    act_fn = F.sigmoid if activation == "sigmoid" else F.silu
    x_normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_normed * weight
    out = out * act_fn(gate)
    return out.to(orig_dtype)


_QWEN3_5 = "transformers.models.qwen3_5.modeling_qwen3_5"
_QWEN4_EXP = "transformers.models.qwen4_exp.modeling_qwen4_exp"
_STOCK_NORM_CASES = [
    pytest.param(_QWEN3_5, "Qwen3_5RMSNormGated", {}, "silu", id="Qwen3_5RMSNormGated"),
    pytest.param(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeRMSNormGated",
        {},
        "silu",
        id="Qwen3_5MoeRMSNormGated",
    ),
    pytest.param(
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "Qwen3NextRMSNormGated",
        {},
        "silu",
        id="Qwen3NextRMSNormGated",
    ),
    pytest.param(_QWEN4_EXP, "Qwen4ExpTextRMSNormGated", {}, "silu", id="Qwen4ExpTextRMSNormGated-silu"),
    pytest.param(
        _QWEN4_EXP,
        "Qwen4ExpTextRMSNormGated",
        {"activation": "sigmoid"},
        "sigmoid",
        id="Qwen4ExpTextRMSNormGated-sigmoid",
    ),
]


def _stock_norm(module_name, class_name, kwargs) -> nn.Module:
    module = pytest.importorskip(module_name)
    norm = getattr(module, class_name)(_HEAD_DIM, eps=1e-6, **kwargs).to(torch.bfloat16)
    with torch.no_grad():  # Qwen3.8-27B's output norm weights: mean 0.95, standard deviation 0.12
        norm.weight.copy_(torch.normal(0.95, 0.12, (_HEAD_DIM,), generator=torch.Generator().manual_seed(0)))
    return norm


def _norm_inputs(rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    x, gate = torch.randn(2, rows, _HEAD_DIM, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).unbind(0)
    return x, gate


@pytest.mark.parametrize(("module_name", "class_name", "kwargs", "activation"), _STOCK_NORM_CASES)
def test_output_norm_rounds_once_as_vllm_does(module_name, class_name, kwargs, activation):
    norm = _stock_norm(module_name, class_name, kwargs)
    x, gate = _norm_inputs()
    expected = _vllm_rms_norm_gated(x, gate, norm.weight, norm.variance_epsilon, activation)
    # The stock norm rounds twice, and a third or so of its outputs then differ from vLLM's.
    assert (norm(x, gate) != expected).float().mean() > 0.1

    vllm_rounding.round_gated_norm_once(norm)

    torch_assert_equal(norm(x, gate), expected)


def test_rounded_norm_keeps_its_class_parameters_and_state_dict():
    norm = _stock_norm(_QWEN3_5, "Qwen3_5RMSNormGated", {})
    state_dict_keys = list(norm.state_dict())
    vllm_rounding.round_gated_norm_once(norm)

    assert type(norm) is modeling_qwen3_5.Qwen3_5RMSNormGated
    assert list(norm.state_dict()) == state_dict_keys
    copied = copy.deepcopy(norm)
    x, gate = _norm_inputs(rows=8)
    torch_assert_equal(copied(x, gate), _vllm_rms_norm_gated(x, gate, norm.weight, norm.variance_epsilon, "silu"))


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_backward_matches_the_formula_in_fp64(activation):
    x, gate = torch.randn(2, 6, 16, dtype=torch.float64).unbind(0)
    weight = torch.randn(16, dtype=torch.float64)
    inputs = tuple(tensor.requires_grad_() for tensor in (x, gate, weight))
    assert torch.autograd.gradcheck(lambda *a: vllm_rounding.rms_norm_gated_fp32(*a, 1e-6, activation), inputs)


def _bf16_inputs(rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    x, gate = torch.randn(2, rows, _HEAD_DIM, generator=generator).to(torch.bfloat16).unbind(0)
    weight = torch.normal(0.95, 0.12, (_HEAD_DIM,), generator=generator).to(torch.bfloat16)
    return tuple(tensor.requires_grad_() for tensor in (x, gate, weight))


def test_backward_saves_only_the_inputs_and_rstd():
    x, gate, weight = _bf16_inputs()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append((str(t.dtype), tuple(t.shape))) or t, lambda t: t
    ):
        vllm_rounding.rms_norm_gated_fp32(x, gate, weight, 1e-6, "silu")
    rows = x.shape[0]
    assert sorted(saved) == sorted(
        [
            ("torch.bfloat16", (rows, _HEAD_DIM)),
            ("torch.bfloat16", (rows, _HEAD_DIM)),
            ("torch.bfloat16", (_HEAD_DIM,)),
            ("torch.float32", (rows, 1)),
        ]
    )


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_bf16_gradients_are_as_close_to_fp64_as_autograd_over_the_formula(activation):
    inputs = _bf16_inputs()
    output = vllm_rounding.rms_norm_gated_fp32(*inputs, 1e-6, activation)
    torch_assert_equal(output, _vllm_rms_norm_gated(*inputs, 1e-6, activation))

    upstream = torch.randn(output.shape, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    gradients = torch.autograd.grad(output, inputs, upstream)
    autograd_gradients = torch.autograd.grad(_vllm_rms_norm_gated(*inputs, 1e-6, activation), inputs, upstream)
    inputs_fp64 = [tensor.detach().double().requires_grad_() for tensor in inputs]
    reference = torch.autograd.grad(
        _vllm_rms_norm_gated(*inputs_fp64, 1e-6, activation), inputs_fp64, upstream.double()
    )
    for name, gradient, autograd_gradient, exact in zip(
        ("x", "gate", "weight"), gradients, autograd_gradients, reference
    ):
        assert gradient.dtype == torch.bfloat16, name
        error = (gradient.double() - exact).abs().max()
        autograd_error = (autograd_gradient.double() - exact).abs().max()
        assert error <= 2 * autograd_error + 1e-6, (name, error.item(), autograd_error.item())


def test_compiled_norm_matches_eager_in_one_graph():
    x, gate, weight = _bf16_inputs(rows=256)
    upstream = torch.randn(x.shape, generator=torch.Generator().manual_seed(6)).to(torch.bfloat16)
    eager = vllm_rounding.rms_norm_gated_fp32(x, gate, weight, 1e-6, "silu")
    eager_gradients = torch.autograd.grad(eager, (x, gate, weight), upstream)
    compiled = torch.compile(vllm_rounding.rms_norm_gated_fp32, fullgraph=True)
    try:
        output = compiled(x, gate, weight, 1e-6, "silu")
        gradients = torch.autograd.grad(output, (x, gate, weight), upstream)
    finally:
        torch._dynamo.reset()
    torch.testing.assert_close(output, eager, rtol=2**-7, atol=2**-14)
    for gradient, eager_gradient in zip(gradients, eager_gradients):
        torch.testing.assert_close(gradient, eager_gradient, rtol=1.6e-2, atol=1e-2)


def test_an_unsupported_gate_activation_is_refused():
    norm = _stock_norm(_QWEN4_EXP, "Qwen4ExpTextRMSNormGated", {"activation": "gelu"})
    with pytest.raises(RuntimeError, match="unsupported GatedDeltaNet output-norm gate activation 'gelu'"):
        vllm_rounding.round_gated_norm_once(norm)


def test_cuda_inputs_run_fla_kernel(monkeypatch):
    calls = []

    def fla_kernel(x, gate, weight, bias, activation, *, eps):
        calls.append((x, gate, weight, bias, activation, eps))
        return x

    monkeypatch.setattr(vllm_rounding, "fla_rms_norm_gated", fla_kernel)
    x, gate = _norm_inputs(rows=8)
    weight = torch.ones(_HEAD_DIM, dtype=torch.bfloat16)
    # CPU inputs run the torch formula even when fla is importable.
    assert not vllm_rounding._uses_fla_kernel(x)
    vllm_rounding.rms_norm_gated_fp32(x, gate, weight, 1e-6, "silu")
    assert calls == []

    monkeypatch.setattr(vllm_rounding, "_uses_fla_kernel", lambda tensor: True)
    vllm_rounding.rms_norm_gated_fp32(x, gate, weight, 1e-6, "sigmoid")
    assert calls == [(x, gate, weight, None, "sigmoid", 1e-6)]


_needs_fla_on_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or vllm_rounding.fla_rms_norm_gated is None,
    reason="fla's rms_norm_gated kernel needs CUDA and fla",
)


def _cuda_inputs(rows: int = 32768) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    x, gate = torch.randn(2, rows, _HEAD_DIM, generator=generator).to(torch.bfloat16).unbind(0)
    weight = torch.normal(0.95, 0.12, (_HEAD_DIM,), generator=generator).to(torch.bfloat16)
    return tuple(tensor.cuda().requires_grad_() for tensor in (x, gate, weight))


def _gradients(output: torch.Tensor, inputs, upstream: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.autograd.grad(output, inputs, upstream))


@_needs_fla_on_cuda
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_fla_kernel_rounds_once_like_the_fp32_formula(activation):
    inputs = _cuda_inputs()
    x, gate, weight = inputs
    assert vllm_rounding._uses_fla_kernel(x)

    output = vllm_rounding.rms_norm_gated_fp32(x, gate, weight, 1e-6, activation)
    expected = _vllm_rms_norm_gated(x, gate, weight, 1e-6, activation)
    twice = (weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)).float()
    twice = (twice * (F.sigmoid if activation == "sigmoid" else F.silu)(gate.float())).to(x.dtype)

    # Both evaluate in fp32 and round once; only the fp32 reduction order differs, so a rare output lands on the
    # neighbouring bf16 value. Rounding twice moves about a third of them.
    assert output.dtype == torch.bfloat16
    assert (output != expected).float().mean() < 1e-3
    assert (twice != expected).float().mean() > 0.1
    torch.testing.assert_close(output, expected, rtol=2**-7, atol=2**-14)

    # Gradients against an fp64 evaluation of the same formula: fla's fused backward is as close as autograd over
    # the torch formula on the same bf16 inputs.
    upstream = torch.randn_like(output)
    fla_gradients = _gradients(output, inputs, upstream)
    torch_gradients = _gradients(expected, inputs, upstream)
    inputs_fp64 = [tensor.detach().double().requires_grad_() for tensor in inputs]
    reference = _vllm_rms_norm_gated(*inputs_fp64[:2], inputs_fp64[2], 1e-6, activation)
    reference_gradients = _gradients(reference, inputs_fp64, upstream.double())
    for name, fla_gradient, torch_gradient, reference_gradient in zip(
        ("x", "gate", "weight"), fla_gradients, torch_gradients, reference_gradients
    ):
        assert fla_gradient.dtype == torch.bfloat16, name
        fla_error = (fla_gradient.double() - reference_gradient).abs().max()
        torch_error = (torch_gradient.double() - reference_gradient).abs().max()
        assert fla_error <= 2 * torch_error + 1e-6, (name, fla_error.item(), torch_error.item())


@_needs_fla_on_cuda
def test_compiled_norm_runs_the_torch_formula_in_one_graph():
    x, gate, weight = _cuda_inputs(rows=4096)
    compiled = torch.compile(vllm_rounding.rms_norm_gated_fp32, fullgraph=True)
    try:
        output = compiled(x, gate, weight, 1e-6, "silu")
        output.float().sum().backward()
    finally:
        torch._dynamo.reset()
    expected = _vllm_rms_norm_gated(x.detach(), gate.detach(), weight.detach(), 1e-6, "silu")
    torch.testing.assert_close(output, expected, rtol=2**-7, atol=2**-14)
    assert all(tensor.grad is not None for tensor in (x, gate, weight))


def test_an_unknown_output_norm_is_refused():
    class OtherNorm(nn.Module):
        def forward(self, hidden_states, gate):
            return hidden_states

    model = _tiny_model()
    model.model.layers[0].linear_attn.norm = OtherNorm()
    with pytest.raises(RuntimeError, match="OtherNorm; its rounding is unknown"):
        vllm_rounding.round_gated_norms_once(model)


def test_other_models_are_left_alone():
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM

    qwen3 = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            intermediate_size=64,
            vocab_size=64,
        )
    )
    assert vllm_rounding.round_gated_norms_once(qwen3) == 0
    assert [name for name, module in qwen3.named_modules() if "forward" in vars(module)] == []


def test_the_stock_classes_resolve_in_the_installed_transformers():
    # A transformers release that renames one of these classes drops it from the lists instead of failing on import.
    norms = {cls.__name__ for cls in vllm_rounding._resolve(vllm_rounding._STOCK_GATED_NORMS)}
    gated_delta_nets = {cls.__name__ for cls in vllm_rounding._resolve(vllm_rounding._STOCK_GATED_DELTA_NETS)}
    assert {"Qwen3_5RMSNormGated", "Qwen3_5MoeRMSNormGated", "Qwen3NextRMSNormGated"} <= norms
    assert gated_delta_nets == {"Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet", "Qwen3NextGatedDeltaNet"}


# ---------------------------------------------------------------------------
# The huggingface loader
# ---------------------------------------------------------------------------


def _tiny_model() -> modeling_qwen3_5.Qwen3_5ForCausalLM:
    set_seed(0)
    config = Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=_HEAD_DIM,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    model = modeling_qwen3_5.Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
        for module in model.modules():
            if isinstance(module, modeling_qwen3_5.Qwen3_5RMSNormGated):
                module.weight.normal_(0.95, 0.12)
    return model.train()


def _recorded_norm_calls(model: nn.Module) -> list[tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]]:
    calls = []
    for module in model.modules():
        if isinstance(module, modeling_qwen3_5.Qwen3_5RMSNormGated):
            module.register_forward_hook(
                lambda norm, args, output: calls.append((norm, args[0].detach(), args[1].detach(), output.detach()))
            )
    return calls


def _assert_every_norm_call_rounds_once(calls) -> None:
    assert len(calls) == 3  # one per linear-attention layer
    for norm, x, gate, output in calls:
        torch_assert_equal(output, _vllm_rms_norm_gated(x, gate, norm.weight, norm.variance_epsilon, "silu"))


def _input_ids() -> torch.Tensor:
    return torch.randint(0, 64, (1, _SEQ), generator=torch.Generator().manual_seed(2))


@pytest.mark.parametrize("loader_options", [{}, {"fp32_parity": False}], ids=["fp32-parity", "no-fp32-parity"])
def test_loader_rounds_every_output_norm_once(monkeypatch, loader_options):
    from arctic_platform.model.loaders.huggingface import load_huggingface

    model = _tiny_model()
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)
    spec = ModelSpec(model_path_or_name="tiny", loader="huggingface", loader_options=loader_options)

    load_huggingface(LoaderContext(spec=spec))

    calls = _recorded_norm_calls(model)
    model.model(input_ids=_input_ids())
    _assert_every_norm_call_rounds_once(calls)


@_needs_fla_on_cuda
def test_loader_rounds_every_output_norm_once_on_cuda(monkeypatch):
    """On CUDA the stock GatedDeltaNet runs fla's kernels, and each loaded output norm runs fla's rms_norm_gated."""
    from arctic_platform.model.loaders.huggingface import load_huggingface

    model = _tiny_model().cuda()
    stock = copy.deepcopy(model)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)

    load_huggingface(LoaderContext(spec=ModelSpec(model_path_or_name="tiny", loader="huggingface")))

    stock_calls, calls = _recorded_norm_calls(stock), _recorded_norm_calls(model)
    with torch.no_grad():
        stock.model(input_ids=_input_ids().cuda())
        model.model(input_ids=_input_ids().cuda())
    assert len(stock_calls) == len(calls) == 3
    for (norm, stock_x, stock_gate, stock_output), (_, x, gate, output) in zip(stock_calls, calls):
        assert vllm_rounding._uses_fla_kernel(x)
        expected = _vllm_rms_norm_gated(x, gate, norm.weight, norm.variance_epsilon, "silu")
        assert (output != expected).float().mean() < 1e-3
        torch.testing.assert_close(output, expected, rtol=2**-7, atol=2**-14)
        stock_expected = _vllm_rms_norm_gated(stock_x, stock_gate, norm.weight, norm.variance_epsilon, "silu")
        assert (stock_output != stock_expected).float().mean() > 0.1


def test_loader_rounds_output_norms_once_behind_the_sequence_parallel_wrapper(monkeypatch, single_rank_group):
    """The wrapper replaces the GatedDeltaNet forward and still calls the module's norm, which rounds once.

    With one rank the all-to-all is the identity; the causal convolution and the gated delta rule need kernels this CPU
    does not have, so they are replaced by stand-ins.
    """
    from arctic_platform.model.implementations.gpu.sp import gated_delta_net as sp_gated_delta_net
    from arctic_platform.model.loaders.huggingface import load_huggingface

    def convolution(*, x, weight, bias, activation, seq_idx):
        return F.silu(x + weight.mean())

    def gated_delta_rule(query, key, value, g, beta, **kwargs):
        return value * 0.5, None

    monkeypatch.setattr(sp_gated_delta_net, "sequence_head_all_to_all", lambda group, tensor, **kwargs: tensor)
    monkeypatch.setattr(
        sp_gated_delta_net,
        "_resolve_gated_delta_net_kernel",
        lambda module, attribute: gated_delta_rule if attribute == "chunk_gated_delta_rule" else convolution,
    )
    model = _tiny_model()
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)
    spec = ModelSpec(
        model_path_or_name="tiny", loader="huggingface", parallelism=ParallelismConfig(sequence_parallel=2)
    )

    load_huggingface(LoaderContext(spec=spec, parallel_groups={"sp_group": single_rank_group}))

    linear_attention = [layer.linear_attn for layer in model.model.layers[:3]]
    assert all(getattr(module, "_dss_gated_delta_net_sequence_parallel", False) for module in linear_attention)
    calls = _recorded_norm_calls(model)
    model.model(input_ids=_input_ids(), position_ids=torch.arange(_SEQ).unsqueeze(0))
    _assert_every_norm_call_rounds_once(calls)
