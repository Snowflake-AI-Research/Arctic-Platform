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
"""CPU checks for the fp32 parity patch of the stock transformers dense Qwen3.5 text model.

With the patch, the three roundings are checked at the stock classes against references written here: the residual
add in fp32 inside the next RMSNorm, the partial rotary embedding in fp32, and the GatedDeltaNet beta gate as a sigmoid
of fp32 ``b``. Forward and backward are checked under the generic gradient-checkpointing patch (full layers, every
layer and every other layer, reentrant and not), under activation offload, with each decoder layer compiled (traced by
dynamo, as one graph and with graph breaks, and run eagerly; also with a graph break forced inside or before the
attention of models with two full-attention layers, and with stock and patched models compiled in one process), and
with fp32 weights under bf16 autocast. The
experimental ``fp32_residual`` setting, which carries the residual stream in fp32 between decoder layers, is checked
against the same reference loop and under the same checkpointing, offload and compile paths.
The huggingface loader's ``fp32_parity`` set to false installs nothing but the GatedDeltaNet output norms that round
once, which the loader installs whatever the option is (``test_gated_norm_rounding.py``). The copied stock
forward is pinned against the stock source of the verified transformers version, and the patch is applied only to the
dense Qwen3.5 text model, once, and can be removed.

The optional ``causal_conv1d`` and ``fla`` kernels are not installed on CPU, so the stock GatedDeltaNet uses its torch
fallbacks.
"""

from __future__ import annotations

import ast
import copy
import difflib
import inspect
import textwrap

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from arctic_platform.model import ModelSpec
from arctic_platform.model import Patches
from arctic_platform.model.implementations.gpu import activation_offload
from arctic_platform.model.implementations.gpu import qwen3_5_dense_parity as hf_dense_parity
from arctic_platform.model.implementations.gpu.activation_offload import ActivationOffloadManager
from arctic_platform.model.implementations.gpu.vllm_rounding import round_gated_norms_once
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patches.gradient_checkpointing import apply_gradient_checkpointing
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

_SEQ = 12
_EPS = 1e-6

# The stock class attributes and module names some patches rewrite process-wide, as they are when this module is
# collected, before any test has run. ``_stock_classes`` restores them for every test, so no test depends on which
# other tests ran first in the process.
_STOCK_ATTRIBUTES = [
    (modeling_qwen3_5.Qwen3_5TextModel, "forward", vars(modeling_qwen3_5.Qwen3_5TextModel)["forward"]),
    (modeling_qwen3_5.Qwen3_5DecoderLayer, "forward", vars(modeling_qwen3_5.Qwen3_5DecoderLayer)["forward"]),
    (modeling_qwen3_5.Qwen3_5Attention, "forward", vars(modeling_qwen3_5.Qwen3_5Attention)["forward"]),
    (modeling_qwen3_5.Qwen3_5GatedDeltaNet, "forward", vars(modeling_qwen3_5.Qwen3_5GatedDeltaNet)["forward"]),
    (modeling_qwen3_5, "apply_rotary_pos_emb", modeling_qwen3_5.apply_rotary_pos_emb),
    (modeling_qwen3_5, "Qwen3_5RMSNorm", modeling_qwen3_5.Qwen3_5RMSNorm),
    (modeling_qwen3_5, "Qwen3_5MLP", modeling_qwen3_5.Qwen3_5MLP),
]


@pytest.fixture(autouse=True)
def _stock_classes(monkeypatch):
    for owner, name, value in _STOCK_ATTRIBUTES:
        monkeypatch.setattr(owner, name, value)


def _config(layer_types: list[str] | None = None) -> Qwen3_5TextConfig:
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
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rms_norm_eps=_EPS,
        tie_word_embeddings=False,
        layer_types=layer_types,
    )
    config._attn_implementation = "sdpa"
    return config


def _tiny_model(dtype: torch.dtype = torch.bfloat16, seed: int = 0) -> modeling_qwen3_5.Qwen3_5ForCausalLM:
    set_seed(seed)
    model = modeling_qwen3_5.Qwen3_5ForCausalLM(_config()).to(dtype)
    # The zero-initialized (1 + weight) norm weights would hide a dropped weight; give every parameter a value.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    return model.train()


def _input_ids(seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 64, (1, _SEQ), generator=torch.Generator().manual_seed(seed + 1))


def _forward_backward(model: nn.Module, autocast: bool = False):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        hidden = model.model(input_ids=_input_ids()).last_hidden_state
    hidden.float().pow(2).mean().backward()
    # A checkpoint wrapper renames the parameters of the layers it wraps.
    grads = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return hidden.detach(), grads


def _assert_same_run(output, grads, expected_output, expected_grads) -> None:
    torch_assert_equal(output, expected_output)
    assert grads.keys() == expected_grads.keys()
    for name in grads:
        torch_assert_equal(grads[name], expected_grads[name], msg=name)


def _patched(model: nn.Module, *, sequence_parallel: bool = False, fp32_residual: bool = False) -> nn.Module:
    assert (
        hf_dense_parity.apply_fp32_parity(model, sequence_parallel=sequence_parallel, fp32_residual=fp32_residual) == 1
    )
    return model


def _rounded_norms(model: nn.Module) -> nn.Module:
    """``model`` with the GatedDeltaNet output norms the huggingface loader installs, as a loaded model has them."""
    assert round_gated_norms_once(model) == 3
    return model


# The forwards the checkpointing, offload and compile checks run: the patch, the patch with an fp32 residual stream,
# and the stock model.
_MODES = ["fp32-parity", "fp32-residual", "stock"]


def _model_in_mode(mode: str, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    model = _tiny_model(dtype)
    if mode != "stock":
        _patched(model, fp32_residual=mode == "fp32-residual")
    return model


# ---------------------------------------------------------------------------
# The stock code the patch copies or relies on
# ---------------------------------------------------------------------------


def test_stock_source_is_the_verified_source():
    # The source hashes are the contract; any transformers release whose source matches passes.
    hf_dense_parity.check_stock_classes()
    hf_dense_parity._check_text_model(_tiny_model().model)


def _body_lines(function) -> list[str]:
    """Source lines of ``function``'s body after any docstring, stripped, without blank lines and comments."""
    source = textwrap.dedent(inspect.getsource(function))
    statements = ast.parse(source).body[0].body
    if isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        statements = statements[1:]
    body = source.splitlines()[statements[0].lineno - 1 :]
    return [line.strip() for line in body if line.strip() and not line.strip().startswith("#")]


def test_copied_text_model_forward_differs_from_stock_only_in_the_layer_loop():
    first = "if (input_ids is None) ^ (inputs_embeds is not None):"
    stock = _body_lines(modeling_qwen3_5.Qwen3_5TextModel.forward)
    stock_body = stock[stock.index(first) :]
    # The copy reaches the stock module's names through ``modeling.``.
    copied = [line.replace("modeling.", "") for line in _body_lines(hf_dense_parity._text_model_forward_body)]
    copied_body = copied[copied.index(first) :]
    changes = [
        (tag, stock_body[i1:i2], copied_body[j1:j2])
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=stock_body, b=copied_body, autojunk=False).get_opcodes()
        if tag != "equal"
    ]
    assert changes == [
        ("insert", [], ["residual = None"]),
        ("replace", ["hidden_states = decoder_layer("], ["hidden_states, residual = decoder_layer("]),
        ("insert", [], ["residual,"]),
        ("insert", [], ["if residual is None:"]),
        ("insert", [], ["else:", "hidden_states, _ = _fused_add_rms_norm(self.norm, hidden_states, residual)"]),
    ]


def test_replaced_decoder_layer_forward_keeps_the_stock_token_mixer():
    stock = _body_lines(modeling_qwen3_5.Qwen3_5DecoderLayer.forward)
    copied = _body_lines(hf_dense_parity._decoder_layer_forward)
    start = 'if self.block_type == "linear_attention":'
    stock_mixer = stock[stock.index(start) : stock.index("hidden_states = residual + hidden_states")]
    copied_mixer = copied[copied.index(start) : copied.index(start) + len(stock_mixer)]
    assert copied_mixer == stock_mixer


def test_mismatching_stock_source_raises_and_patches_nothing(monkeypatch):
    monkeypatch.setitem(hf_dense_parity._STOCK_SOURCE_SHA256, "Qwen3_5TextModel.forward", "0" * 64)
    model = _tiny_model()
    with pytest.raises(RuntimeError, match="fp32_parity to false"):
        hf_dense_parity.apply_fp32_parity(model, sequence_parallel=False)
    assert "forward" not in vars(model.model)
    assert all("forward" not in vars(layer) for layer in model.model.layers)


# ---------------------------------------------------------------------------
# Where the patch applies
# ---------------------------------------------------------------------------


def _instance_forwards(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if "forward" in vars(module)]


def _beta_hooks(model: nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules() if isinstance(module, nn.Linear))


def _resolved_epsilons(model: nn.Module) -> list[float]:
    name = hf_dense_parity._NORM_EPSILON_ATTRIBUTE
    return [vars(module)[name] for module in model.modules() if name in vars(module)]


def test_patch_is_applied_once_and_can_be_removed():
    model = _tiny_model()
    assert hf_dense_parity.apply_fp32_parity(model, sequence_parallel=False) == 1
    patched = sorted(_instance_forwards(model))
    hooks = _beta_hooks(model)
    assert hf_dense_parity.apply_fp32_parity(model, sequence_parallel=False) == 0
    assert sorted(_instance_forwards(model)) == patched
    assert _beta_hooks(model) == hooks == 3  # one per linear-attention layer
    # The text model, every decoder layer and every full-attention module.
    assert patched == sorted(["model"] + [f"model.layers.{i}" for i in range(4)] + ["model.layers.3.self_attn"])
    # The module-level rotary function stays the stock one.
    assert modeling_qwen3_5.apply_rotary_pos_emb is not hf_dense_parity._apply_rotary_pos_emb_fp32

    # Every fused norm holds the epsilon resolved at patch time: two per decoder layer and the final norm.
    assert len(_resolved_epsilons(model)) == 9
    assert hf_dense_parity.remove_fp32_parity(model) == 1
    assert _instance_forwards(model) == []
    assert _beta_hooks(model) == 0
    assert _resolved_epsilons(model) == []
    expected_output, expected_grads = _forward_backward(_tiny_model())
    _assert_same_run(*_forward_backward(model), expected_output, expected_grads)


def test_sequence_parallel_patch_installs_no_beta_hook():
    # With sequence parallelism the wrapper replaces the GatedDeltaNet forward and takes fp32_beta itself.
    model = _patched(_tiny_model(), sequence_parallel=True)
    assert _beta_hooks(model) == 0


def test_other_models_are_unaffected():
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
    assert hf_dense_parity.apply_fp32_parity(qwen3, sequence_parallel=False) == 0
    assert _instance_forwards(qwen3) == []
    assert _beta_hooks(qwen3) == 0


# ---------------------------------------------------------------------------
# The three roundings, against references written here
# ---------------------------------------------------------------------------


def _reference_fused_add_rms_norm(x, residual, weight, eps, fp32_residual=False):
    # fp32 for the 16-bit dtypes; fp64 stays fp64 so the backward check can use it.
    compute = torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype
    summed = x.to(compute) + residual.to(compute)
    normed = summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.to(compute))
    return normed.to(x.dtype), summed if fp32_residual else summed.to(x.dtype)


def _reference_text_model(text_model, input_ids, fp32_residual=False):
    """The patched model's loop, with every norm written out: the residual add in fp32 inside the next norm.

    With ``fp32_residual`` the residual is the fp32 embeddings for the first layer and every sum stays unrounded.
    """
    hidden = text_model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(4, 1, -1)
    position_embeddings = text_model.rotary_emb(hidden, position_ids[1:])
    masks = {
        "full_attention": modeling_qwen3_5.create_causal_mask(
            config=text_model.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids[0],
        ),
        "linear_attention": None,
    }
    residual = None
    for layer, layer_type in zip(text_model.layers, text_model.config.layer_types):
        if residual is None:
            residual = hidden.float() if fp32_residual else hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = _reference_fused_add_rms_norm(
                hidden, residual, layer.input_layernorm.weight, layer.input_layernorm.eps, fp32_residual
            )
        if layer_type == "linear_attention":
            hidden = layer.linear_attn(hidden_states=hidden, attention_mask=None)
        else:
            hidden, _ = layer.self_attn(
                hidden_states=hidden,
                attention_mask=masks["full_attention"],
                position_ids=position_ids[0],
                position_embeddings=position_embeddings,
            )
        hidden, residual = _reference_fused_add_rms_norm(
            hidden, residual, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps, fp32_residual
        )
        hidden = layer.mlp(hidden)
    hidden, _ = _reference_fused_add_rms_norm(hidden, residual, text_model.norm.weight, text_model.norm.eps)
    return hidden


def test_residual_add_is_evaluated_in_fp32_inside_the_next_norm():
    model = _patched(_tiny_model())
    input_ids = _input_ids()
    # The token mixers are the patched modules on both sides, so this isolates the norms and the residual stream.
    with torch.no_grad():
        expected = _reference_text_model(model.model, input_ids)
        actual = model.model(input_ids=input_ids).last_hidden_state
    torch_assert_equal(actual, expected)


def test_patched_backward_matches_the_reference_loop():
    model = _patched(_tiny_model(torch.float64).to(torch.float64))
    input_ids = _input_ids()
    reference = copy.deepcopy(model)
    hf_dense_parity.remove_fp32_parity(reference)
    hf_dense_parity.apply_fp32_parity(reference, sequence_parallel=False)
    model.model(input_ids=input_ids).last_hidden_state.pow(2).mean().backward()
    _reference_text_model(reference.model, input_ids).pow(2).mean().backward()
    for (name, parameter), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        if parameter.grad is not None:
            torch_assert_close(parameter.grad, expected.grad, msg=name)


def test_rotary_embedding_is_evaluated_in_fp32():
    set_seed(0)
    q = torch.randn(1, 2, _SEQ, 16).to(torch.bfloat16)
    k = torch.randn(1, 1, _SEQ, 16).to(torch.bfloat16)
    cos = torch.randn(1, _SEQ, 4).to(torch.bfloat16)
    sin = torch.randn(1, _SEQ, 4).to(torch.bfloat16)

    def reference(x):
        x32, cos32, sin32 = x.float(), cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
        rotary = x32[..., :4]
        rotated = torch.cat((-rotary[..., 2:], rotary[..., :2]), dim=-1)
        return torch.cat(((rotary * cos32 + rotated * sin32).to(x.dtype), x[..., 4:]), dim=-1)

    q_embed, k_embed = hf_dense_parity._apply_rotary_pos_emb_fp32(q, k, cos, sin)
    torch_assert_equal(q_embed, reference(q))
    torch_assert_equal(k_embed, reference(k))
    stock_q, _ = modeling_qwen3_5.apply_rotary_pos_emb(q, k, cos, sin)
    assert not torch.equal(q_embed, stock_q)  # the stock rotation rounds each product in bf16


def test_patched_attention_uses_the_fp32_rotation(monkeypatch):
    model = _patched(_tiny_model())
    attention = model.model.layers[3].self_attn
    assert attention.forward.__func__.__globals__["apply_rotary_pos_emb"] is hf_dense_parity._apply_rotary_pos_emb_fp32
    set_seed(0)
    hidden = torch.randn(1, _SEQ, 32).to(torch.bfloat16)
    position_ids = torch.arange(_SEQ).view(1, 1, -1).expand(3, 1, -1)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    with torch.no_grad():
        actual, _ = attention(hidden_states=hidden, attention_mask=None, position_embeddings=position_embeddings)
        # The stock forward with the module-level function swapped for the fp32 rotation gives the same output.
        monkeypatch.setattr(modeling_qwen3_5, "apply_rotary_pos_emb", hf_dense_parity._apply_rotary_pos_emb_fp32)
        expected, _ = modeling_qwen3_5.Qwen3_5Attention.forward(
            attention, hidden_states=hidden, attention_mask=None, position_embeddings=position_embeddings
        )
    torch_assert_equal(actual, expected)


def test_gated_delta_net_beta_is_a_sigmoid_of_fp32_b(monkeypatch):
    model = _patched(_tiny_model())
    gated_delta_net = model.model.layers[0].linear_attn
    captured = {}
    stock_rule = modeling_qwen3_5.torch_chunk_gated_delta_rule

    def recording_rule(query, key, value, g, beta, **kwargs):
        captured["beta"] = beta
        return stock_rule(query, key, value, g=g, beta=beta, **kwargs)

    monkeypatch.setattr(modeling_qwen3_5, "torch_chunk_gated_delta_rule", recording_rule)
    set_seed(0)
    hidden = torch.randn(1, _SEQ, 32).to(torch.bfloat16)
    with torch.no_grad():
        gated_delta_net(hidden_states=hidden)
        b = F.linear(hidden, gated_delta_net.in_proj_b.weight)
    assert captured["beta"].dtype == torch.float32
    torch_assert_equal(captured["beta"], b.float().sigmoid())


# ---------------------------------------------------------------------------
# Activation checkpointing, offload and autocast
# ---------------------------------------------------------------------------


def _checkpointed(model: nn.Module, gradient_checkpointing) -> nn.Module:
    spec = ModelSpec(
        model_path_or_name="tiny",
        loader="huggingface",
        patches=Patches(gradient_checkpointing=gradient_checkpointing),
    )
    replacement = apply_gradient_checkpointing(model, LoaderContext(spec=spec))
    return replacement if replacement is not None else model


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("gradient_checkpointing", [True, 2], ids=["every-layer", "every-other-layer"])
def test_gradient_checkpointing_patch_forward_backward_matches_eager(mode, gradient_checkpointing):
    reference_model = _model_in_mode(mode)
    checkpointed_model = _checkpointed(copy.deepcopy(reference_model), gradient_checkpointing)

    expected_output, expected_grads = _forward_backward(reference_model)
    _assert_same_run(*_forward_backward(checkpointed_model), expected_output, expected_grads)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_transformers_gradient_checkpointing_forward_backward_matches_eager(use_reentrant, mode):
    reference_model = _model_in_mode(mode)
    checkpointed_model = copy.deepcopy(reference_model)
    checkpointed_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})

    expected_output, expected_grads = _forward_backward(reference_model)
    _assert_same_run(*_forward_backward(checkpointed_model), expected_output, expected_grads)


@pytest.mark.parametrize("mode", _MODES)
def test_tensors_each_checkpointed_layer_receives(mode):
    model = _model_in_mode(mode)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    received = []
    for layer in model.model.layers:
        checkpoint = layer._gradient_checkpointing_func

        def recording(function, *args, checkpoint=checkpoint, **kwargs):
            received.append([arg.dtype for arg in args if torch.is_tensor(arg) and arg.shape[-1] == 32])
            return checkpoint(function, *args, **kwargs)

        layer._gradient_checkpointing_func = recording
    _forward_backward(model)
    # With the patch every layer after the first holds the branch output and the residual at its boundary; with the
    # fp32 residual stream that residual is fp32.
    expected = {
        "fp32-parity": [[torch.bfloat16]] + [[torch.bfloat16, torch.bfloat16]] * 3,
        "fp32-residual": [[torch.bfloat16]] + [[torch.bfloat16, torch.float32]] * 3,
        "stock": [[torch.bfloat16]] * 4,
    }
    assert received == expected[mode]


def _eligible_on_any_device(self: ActivationOffloadManager, tensor: torch.Tensor) -> bool:
    """``ActivationOffloadManager._eligible`` without its CUDA-only device check and its skip counters."""
    if not self.enabled or isinstance(tensor, nn.Parameter):
        return False
    if tensor.numel() * tensor.element_size() < self.tensor_size_threshold:
        return False
    if not tensor.is_contiguous() and activation_offload._tensor_has_storage_overlap(tensor):
        return False
    return True


@pytest.mark.parametrize("mode", _MODES)
def test_activation_offload_forward_backward_matches_eager(monkeypatch, mode):
    monkeypatch.setattr(ActivationOffloadManager, "_eligible", _eligible_on_any_device)
    reference_model = _model_in_mode(mode)
    offloaded_model = _checkpointed(copy.deepcopy(reference_model), True)
    manager = activation_offload.install_activation_offload(
        offloaded_model.model, keep_last_n=0, use_streams=False, tensor_size_threshold=0, pin_memory_enabled=False
    )

    expected_output, expected_grads = _forward_backward(reference_model)
    output, grads = _forward_backward(offloaded_model)

    assert manager.stats.offloaded_tensors > 0
    assert manager.stats.restored_tensors == manager.stats.offloaded_tensors
    _assert_same_run(output, grads, expected_output, expected_grads)


def _compiled_layers(model: nn.Module, *, fullgraph: bool) -> nn.Module:
    torch._dynamo.reset()
    # The compile patch's per-layer compilation, with a backend that traces the graph but runs it eagerly.
    for layer in model.model.layers:
        layer.compile(fullgraph=fullgraph, backend="aot_eager")
    return model


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("fullgraph", [True, False], ids=["fullgraph", "graph-breaks"])
def test_compiled_layers_forward_backward_match_eager(mode, fullgraph):
    reference_model = _model_in_mode(mode)
    compiled_model = _compiled_layers(copy.deepcopy(reference_model), fullgraph=fullgraph)

    expected_output, expected_grads = _forward_backward(reference_model)
    try:
        output, grads = _forward_backward(compiled_model)
    finally:
        torch._dynamo.reset()
    _assert_same_run(output, grads, expected_output, expected_grads)


_GRAPH_BREAK_ATTENTION = "sdpa_after_a_graph_break"


def _sdpa_after_a_graph_break(*args, **kwargs):
    """SDPA behind a graph break, as a compiled layer meets one at an attention kernel dynamo cannot trace."""
    torch._dynamo.graph_break()
    return sdpa_attention_forward(*args, **kwargs)


def _graph_break_model(monkeypatch, mode: str, where: str, *, sequence_parallel: bool = False) -> nn.Module:
    """A model with two full-attention layers whose compiled decoder layers break their graph at ``where``."""
    set_seed(0)
    config = _config(layer_types=["linear_attention", "full_attention"] * 2)
    model = modeling_qwen3_5.Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    model.train()
    if mode != "stock":
        _patched(model, sequence_parallel=sequence_parallel, fp32_residual=mode == "fp32-residual")
    if where == "in-attention":
        monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._local_mapping, _GRAPH_BREAK_ATTENTION, _sdpa_after_a_graph_break)
        model.config._attn_implementation = _GRAPH_BREAK_ATTENTION
    else:
        # A break in the decoder layer before the attention call, so that the attention forward is compiled as a
        # frame of its own.
        for layer in model.model.layers:
            if layer.block_type == "full_attention":
                layer.self_attn.register_forward_pre_hook(lambda module, args: torch._dynamo.graph_break())
    return model


# The patched modes everywhere a graph break used to reuse another layer's compiled attention code, and one stock
# control. A break before the attention of layers that are not checkpointed is left out: it never reused that code.
_GRAPH_BREAK_CASES = [
    pytest.param(mode, checkpointing, where, id=f"{where}-{'checkpointed' if checkpointing else 'eager'}-{mode}")
    for mode in ("fp32-parity", "fp32-residual")
    for checkpointing, where in ((False, "in-attention"), (True, "in-attention"), (True, "before-attention"))
] + [pytest.param("stock", True, "in-attention", id="in-attention-checkpointed-stock")]


@pytest.mark.parametrize(("mode", "checkpointing", "where"), _GRAPH_BREAK_CASES)
def test_compiled_layers_with_a_graph_break_match_eager(monkeypatch, mode, checkpointing, where):
    reference_model = _graph_break_model(monkeypatch, mode, where)
    compiled_model = copy.deepcopy(reference_model)
    if checkpointing:
        compiled_model = _checkpointed(compiled_model, True)
    compiled_model = _compiled_layers(compiled_model, fullgraph=False)

    expected_output, expected_grads = _forward_backward(reference_model)
    try:
        output, grads = _forward_backward(compiled_model)
    finally:
        torch._dynamo.reset()
    _assert_same_run(output, grads, expected_output, expected_grads)


def _sequence_parallel(monkeypatch, model: nn.Module, group, *, fp32_beta: bool) -> nn.Module:
    """``model`` behind the sequence-parallel GatedDeltaNet wrapper, on one rank, with stand-ins for its kernels.

    With one rank the all-to-all is the identity. The causal convolution and the gated delta rule need kernels this
    CPU does not have, so they are replaced by differentiable stand-ins; the attention layers are the ones checked.
    """
    from arctic_platform.model.implementations.gpu.sp import gated_delta_net as sp_gated_delta_net
    from arctic_platform.model.implementations.gpu.sp.transformers import (
        configure_transformers_sequence_parallel_model,
    )

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
    assert configure_transformers_sequence_parallel_model(model, group, fp32_beta=fp32_beta) == 2
    return model


def _sequence_parallel_forward_backward(model: nn.Module):
    model.zero_grad(set_to_none=True)
    # The sequence-parallel wrapper locates the packed sequences from the position ids.
    hidden = model.model(input_ids=_input_ids(), position_ids=torch.arange(_SEQ).unsqueeze(0)).last_hidden_state
    hidden.float().pow(2).mean().backward()
    grads = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return hidden.detach(), grads


@pytest.mark.parametrize("mode", ["fp32-parity", "fp32-residual"])
def test_compiled_checkpointed_layers_with_sequence_parallelism_match_eager(monkeypatch, single_rank_group, mode):
    # The wrapper binds the forwards it replaces to their model, so each model is built and wrapped on its own.
    reference_model, compiled_model = (
        _sequence_parallel(
            monkeypatch,
            _graph_break_model(monkeypatch, mode, "in-attention", sequence_parallel=True),
            single_rank_group,
            fp32_beta=True,
        )
        for _ in range(2)
    )
    compiled_model = _compiled_layers(_checkpointed(compiled_model, True), fullgraph=False)

    expected_output, expected_grads = _sequence_parallel_forward_backward(reference_model)
    try:
        output, grads = _sequence_parallel_forward_backward(compiled_model)
    finally:
        torch._dynamo.reset()
    _assert_same_run(output, grads, expected_output, expected_grads)


def test_stock_and_patched_models_compiled_in_one_process_match_eager(monkeypatch):
    # The stock attention forward and the patched one run the same bytecode; each must keep its own compiled code.
    models = [_graph_break_model(monkeypatch, mode, "in-attention") for mode in ("stock", "fp32-parity", "stock")]
    expected = [_forward_backward(model) for model in models]
    compiled = [_compiled_layers(copy.deepcopy(model), fullgraph=False) for model in models]
    try:
        for model, (expected_output, expected_grads) in zip(compiled, expected):
            _assert_same_run(*_forward_backward(model), expected_output, expected_grads)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("mode", _MODES)
def test_fullgraph_compile_refuses_a_graph_break_in_attention(monkeypatch, mode):
    model = _compiled_layers(_graph_break_model(monkeypatch, mode, "in-attention"), fullgraph=True)
    try:
        with pytest.raises(torch._dynamo.exc.Unsupported, match="graph_break"):
            _forward_backward(model)
    finally:
        torch._dynamo.reset()


def test_fp32_weights_under_bf16_autocast_keep_a_bf16_residual_stream():
    model = _patched(_tiny_model(torch.float32))
    residual_dtypes = []
    for layer in model.model.layers:
        layer.register_forward_hook(lambda module, args, output: residual_dtypes.append(output[1].dtype))
    output, grads = _forward_backward(model, autocast=True)
    # Each norm returns the residual in the dtype of the branch output, so the stream is bf16 from the first layer on.
    assert output.dtype == torch.bfloat16
    assert residual_dtypes == [torch.bfloat16] * 4
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    checkpointed = _checkpointed(copy.deepcopy(model), True)
    _assert_same_run(*_forward_backward(checkpointed, autocast=True), output, grads)


def test_fp32_model_matches_the_stock_model():
    # In fp32 every rounding of the patch is already the stock one, up to the order of operations in the norm.
    stock = _tiny_model(torch.float32)
    patched = _patched(copy.deepcopy(stock))
    stock_output, stock_grads = _forward_backward(stock)
    output, grads = _forward_backward(patched)
    torch_assert_close(output, stock_output)
    for name in grads:
        torch_assert_close(grads[name], stock_grads[name], msg=name)


def test_output_hidden_states_is_refused():
    model = _patched(_tiny_model())
    with pytest.raises(NotImplementedError, match="fp32_parity to false"):
        model.model(input_ids=_input_ids(), output_hidden_states=True)


# ---------------------------------------------------------------------------
# The experimental fp32 residual stream
# ---------------------------------------------------------------------------


def _residual_dtypes(model: nn.Module) -> list[torch.dtype]:
    dtypes = []
    for layer in model.model.layers:
        layer.register_forward_hook(lambda module, args, output: dtypes.append(output[1].dtype))
    return dtypes


@pytest.mark.parametrize(("dtype", "autocast"), [(torch.bfloat16, False), (torch.float32, True)])
def test_fp32_residual_carries_the_residual_stream_in_fp32(dtype, autocast):
    model = _patched(_tiny_model(dtype), fp32_residual=True)
    residual_dtypes = _residual_dtypes(model)
    output, grads = _forward_backward(model, autocast=autocast)
    # From the first layer on, including the seed the embeddings give it; the final norm returns the branch dtype.
    assert residual_dtypes == [torch.float32] * 4
    assert output.dtype == torch.bfloat16
    assert all(torch.isfinite(grad).all() for grad in grads.values())


def test_fp32_residual_forward_matches_the_reference_loop():
    model = _patched(_tiny_model(), fp32_residual=True)
    input_ids = _input_ids()
    with torch.no_grad():
        expected = _reference_text_model(model.model, input_ids, fp32_residual=True)
        actual = model.model(input_ids=input_ids).last_hidden_state
    torch_assert_equal(actual, expected)
    # The rounded residual stream gives a different result, so the setting is in effect.
    with torch.no_grad():
        rounded = _reference_text_model(model.model, input_ids)
    assert not torch.equal(actual, rounded)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def test_fp32_residual_fused_norm_backward_matches_autograd_through_the_reference():
    from arctic_platform.model.implementations.gpu.vllm_rounding import FusedAddRMSNormFp32Residual

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, _SEQ, 32, generator=generator).bfloat16().requires_grad_()
    residual = torch.randn(2, _SEQ, 32, generator=generator).requires_grad_()
    weight = (0.2 * torch.randn(32, generator=generator)).bfloat16().requires_grad_()
    grad_out = torch.randn(2, _SEQ, 32, generator=generator).bfloat16()
    grad_sum = torch.randn(2, _SEQ, 32, generator=generator)

    out, summed = FusedAddRMSNormFp32Residual.apply(x, residual, weight, _EPS)
    expected_out, expected_sum = _reference_fused_add_rms_norm(x, residual, weight, _EPS, fp32_residual=True)
    torch_assert_equal(out, expected_out)
    torch_assert_equal(summed, expected_sum)
    assert summed.dtype == torch.float32
    grads = torch.autograd.grad(
        (out.float() * grad_out.float()).sum() + (summed * grad_sum).sum(), (x, residual, weight)
    )
    expected_grads = torch.autograd.grad(
        (expected_out.float() * grad_out.float()).sum() + (expected_sum * grad_sum).sum(), (x, residual, weight)
    )
    # One operation: the two backwards differ only in the order of their fp32 reductions (about 1e-7 here).
    for name, grad, expected in zip(("x", "residual", "weight"), grads, expected_grads):
        assert grad.dtype == expected.dtype, name
        assert _relative_error(grad, expected) < 1e-5, name


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fp32_residual_backward_matches_the_reference_loop(seed):
    model = _patched(_tiny_model(seed=seed), fp32_residual=True)
    reference = copy.deepcopy(model)
    input_ids = _input_ids(seed)
    model.model(input_ids=input_ids).last_hidden_state.float().pow(2).mean().backward()
    _reference_text_model(reference.model, input_ids, fp32_residual=True).float().pow(2).mean().backward()
    grads = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    expected = {name: parameter.grad for name, parameter in reference.named_parameters() if parameter.grad is not None}
    assert grads.keys() == expected.keys()
    # The fused backward orders its fp32 reductions differently from autograd through the reference, and through
    # four bf16 layers that moves single elements by a bf16 step or two, by a different amount on each CPU. Each
    # parameter's gradient is compared as a whole instead: over seeds and thread counts its relative error stays
    # near 3e-2 at most, and the whole gradient's near 2.5e-2.
    for name in grads:
        assert _relative_error(grads[name], expected[name]) < 5e-2, name
    whole = torch.cat([grads[name].float().flatten() for name in grads])
    expected_whole = torch.cat([expected[name].float().flatten() for name in grads])
    assert _relative_error(whole, expected_whole) < 4e-2


def test_fp32_residual_is_cleared_when_the_patch_is_removed():
    model = _patched(_tiny_model(), fp32_residual=True)
    assert all(getattr(layer, "_arctic_fp32_residual") for layer in model.model.layers)
    assert hf_dense_parity.remove_fp32_parity(model) == 1
    assert not any("_arctic_fp32_residual" in vars(layer) for layer in model.model.layers)
    _patched(model)
    residual_dtypes = _residual_dtypes(model)
    with torch.no_grad():
        model.model(input_ids=_input_ids())
    assert residual_dtypes == [torch.bfloat16] * 4


# ---------------------------------------------------------------------------
# The huggingface loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("loader_options", "fp32_residual"), [({}, False), ({"fp32_residual": True}, True)])
def test_loader_passes_fp32_residual_to_the_patch(monkeypatch, loader_options, fp32_residual):
    from arctic_platform.model.loaders.huggingface import load_huggingface

    model = _tiny_model()
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)
    spec = ModelSpec(model_path_or_name="tiny", loader="huggingface", loader_options=loader_options)

    load_huggingface(LoaderContext(spec=spec))

    assert [getattr(layer, "_arctic_fp32_residual") for layer in model.model.layers] == [fp32_residual] * 4


def test_fp32_residual_without_fp32_parity_is_rejected():
    from arctic_platform.model.loaders.huggingface import HuggingfaceOptions

    with pytest.raises(ValueError, match="fp32_residual requires fp32_parity"):
        HuggingfaceOptions.model_validate({"fp32_parity": False, "fp32_residual": True})


@pytest.mark.parametrize(("loader_options", "patched"), [({}, True), ({"fp32_parity": False}, False)])
def test_loader_applies_the_patch_only_when_the_option_is_on(monkeypatch, loader_options, patched):
    from arctic_platform.model.loaders.huggingface import load_huggingface

    model = _tiny_model()
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)
    spec = ModelSpec(model_path_or_name="tiny", loader="huggingface", loader_options=loader_options)

    loaded = load_huggingface(LoaderContext(spec=spec))

    assert loaded.model is model
    assert getattr(model.model, "_arctic_fp32_parity", False) is patched
    if not patched:
        # Off installs nothing but the output norms that round once: the model is otherwise the stock one.
        assert _instance_forwards(model) == [f"model.layers.{i}.linear_attn.norm" for i in range(3)]
        assert _beta_hooks(model) == 0
        expected_output, expected_grads = _forward_backward(_rounded_norms(_tiny_model()))
        _assert_same_run(*_forward_backward(model), expected_output, expected_grads)


def test_loader_checks_the_source_before_loading_weights(monkeypatch, tmp_path):
    import json

    from arctic_platform.model.loaders.huggingface import load_huggingface

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {}}))
    monkeypatch.setitem(hf_dense_parity._STOCK_SOURCE_SHA256, "Qwen3_5DecoderLayer.forward", "0" * 64)

    def no_weights(*args, **kwargs):
        raise AssertionError("the weights must not load when the check fails")

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", no_weights)
    spec = ModelSpec(model_path_or_name=str(tmp_path), loader="huggingface")
    with pytest.raises(RuntimeError, match="Qwen3_5DecoderLayer.forward"):
        load_huggingface(LoaderContext(spec=spec))


def test_a_different_installed_source_is_named_as_such(monkeypatch):
    monkeypatch.setitem(hf_dense_parity._STOCK_SOURCE_SHA256, "Qwen3_5RMSNorm.forward", "0" * 64)
    with pytest.raises(RuntimeError) as excinfo:
        hf_dense_parity.check_stock_classes()
    message = str(excinfo.value)
    assert "has different source for Qwen3_5RMSNorm.forward" in message
    assert "fp32_parity to false" in message


def _replacement_forward(self, *args, **kwargs):
    raise AssertionError("never called")


def test_a_class_level_replacement_is_named_with_its_source(monkeypatch):
    monkeypatch.setattr(modeling_qwen3_5.Qwen3_5DecoderLayer, "forward", _replacement_forward)
    model = _tiny_model()
    with pytest.raises(RuntimeError) as excinfo:
        hf_dense_parity.apply_fp32_parity(model, sequence_parallel=False)
    message = str(excinfo.value)
    assert f"Qwen3_5DecoderLayer.forward by {__name__}._replacement_forward has replaced the stock code" in message
    assert "has different source" not in message
    assert _instance_forwards(model) == []


def test_the_custom_moe_builders_class_patch_is_named():
    # Building a custom Qwen3.5-MoE model patches the stock dense classes for packed sequences, process-wide.
    from arctic_platform.model.implementations.qwen35 import model_builder

    model_builder._patch_qwen3_5_linear_attn_varlen()
    with pytest.raises(RuntimeError, match="_patch_qwen3_5_linear_attn_varlen"):
        hf_dense_parity.check_stock_classes()
    with pytest.raises(RuntimeError, match="_patch_qwen3_5_linear_attn_varlen"):
        hf_dense_parity.apply_fp32_parity(_tiny_model(), sequence_parallel=False)


# ---------------------------------------------------------------------------
# The liger patch
# ---------------------------------------------------------------------------


def _cpu_liger(monkeypatch) -> list[int]:
    """Let Liger's patched modules run on CPU, count its RMSNorm calls, and restore what its patch rebinds.

    Liger's kernels need a GPU. Its RMSNorm forward is replaced by the stock formula and its SwiGLU kernel by the stock
    MLP arithmetic, so a Liger-patched model computes what the stock modules compute.
    """
    from liger_kernel.transformers import rms_norm as liger_rms_norm
    from liger_kernel.transformers import swiglu as liger_swiglu

    calls = []

    def rms_norm_forward(self, hidden_states):
        calls.append(1)
        output = hidden_states.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (output * (self.offset + self.weight.float())).type_as(hidden_states)

    monkeypatch.setattr(liger_rms_norm.LigerRMSNorm, "forward", rms_norm_forward)
    monkeypatch.setattr(liger_swiglu, "_swiglu_dispatch", lambda gate, up: F.silu(gate) * up)
    # The liger patch rebinds these process-wide; monkeypatch restores them after the test.
    for name in ("Qwen3_5RMSNorm", "Qwen3_5MLP"):
        monkeypatch.setattr(modeling_qwen3_5, name, getattr(modeling_qwen3_5, name))
    for cls in (modeling_qwen3_5.Qwen3_5ForCausalLM, modeling_qwen3_5.Qwen3_5ForConditionalGeneration):
        monkeypatch.setattr(cls, "forward", cls.forward)
    return calls


def _build(monkeypatch, model: nn.Module, **patches):
    from arctic_platform.model import build_model

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)
    return build_model(ModelSpec(model_path_or_name="tiny", loader="huggingface", patches=Patches(**patches))).model


def test_liger_patch_with_the_default_runs_liger_rmsnorm_only_in_the_first_input_norm(monkeypatch):
    liger_calls = _cpu_liger(monkeypatch)
    reference = _rounded_norms(_patched(_tiny_model()))
    model = _build(monkeypatch, copy.deepcopy(_tiny_model()), liger=True)

    assert getattr(model.model, "_arctic_fp32_parity", False)
    assert modeling_qwen3_5.Qwen3_5RMSNorm is not reference.model.norm.__class__  # the liger patch rebound it
    with torch.no_grad():
        output = model.model(input_ids=_input_ids()).last_hidden_state
        expected = reference.model(input_ids=_input_ids()).last_hidden_state
    # Every residual add and the norm after it are one fused fp32 operation, which reads the norm's weight and
    # epsilon; only the first layer's input norm is called on one tensor.
    assert len(liger_calls) == 1
    torch_assert_equal(output, expected)


def test_a_dense_model_built_after_a_liger_patch_loads_with_the_default(monkeypatch):
    from arctic_platform.model.loaders.huggingface import load_huggingface

    liger_calls = _cpu_liger(monkeypatch)
    stock = _rounded_norms(_patched(_tiny_model()))
    _build(monkeypatch, _tiny_model(), liger=True)
    # The liger patch replaced the module's norm class, so this second model is built with Liger's norm.
    second = _tiny_model()
    assert type(second.model.norm) is not type(stock.model.norm)
    second.load_state_dict(stock.state_dict())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: second)

    load_huggingface(LoaderContext(spec=ModelSpec(model_path_or_name="tiny", loader="huggingface")))

    assert getattr(second.model, "_arctic_fp32_parity", False)
    liger_calls.clear()
    with torch.no_grad():
        torch_assert_equal(
            second.model(input_ids=_input_ids()).last_hidden_state,
            stock.model(input_ids=_input_ids()).last_hidden_state,
        )
    # Built with Liger's class, the attention's q and k norms run it too: the first input norm, q_norm and k_norm.
    assert len(liger_calls) == 3


def test_a_norm_that_is_no_1_plus_weight_rmsnorm_is_refused():
    model = _tiny_model()
    model.model.layers[1].post_attention_layernorm = nn.LayerNorm(32)
    with pytest.raises(RuntimeError, match="norms of type LayerNorm"):
        hf_dense_parity.apply_fp32_parity(model, sequence_parallel=False)
    assert _instance_forwards(model) == []


def test_fp32_rotation_rejects_another_unsqueeze_dim():
    q = torch.zeros(1, _SEQ, 2, 16)
    cos = torch.zeros(1, _SEQ, 4)
    with pytest.raises(ValueError, match="unsqueeze_dim=1"):
        hf_dense_parity._apply_rotary_pos_emb_fp32(q, q, cos, cos, unsqueeze_dim=2)


@pytest.mark.parametrize(("sequence_parallel", "beta_hooks"), [(1, 3), (2, 0)])
def test_loader_installs_the_beta_hook_only_without_sequence_parallelism(monkeypatch, sequence_parallel, beta_hooks):
    from arctic_platform.model import ParallelismConfig
    from arctic_platform.model.loaders.huggingface import load_huggingface

    model = _tiny_model()
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)
    configured = []
    monkeypatch.setattr(
        "arctic_platform.model.implementations.gpu.sp.transformers.configure_transformers_sequence_parallel_model",
        lambda configured_model, group, *, fp32_beta: configured.append(fp32_beta),
    )
    spec = ModelSpec(
        model_path_or_name="tiny",
        loader="huggingface",
        parallelism=ParallelismConfig(sequence_parallel=sequence_parallel),
    )

    load_huggingface(LoaderContext(spec=spec, parallel_groups={"sp_group": object()}))

    assert getattr(model.model, "_arctic_fp32_parity", False)
    # Without sequence parallelism the hook makes the stock forward's gate fp32; with it the wrapper does.
    assert _beta_hooks(model) == beta_hooks
    assert configured == ([True] if sequence_parallel > 1 else [])
