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
layer and every other layer, reentrant and not), under activation offload, and with fp32 weights under bf16 autocast.
The huggingface loader's ``fp32_parity`` set to false installs nothing, so the model is the stock one. The copied stock
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
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from arctic_platform.model import ModelSpec
from arctic_platform.model import Patches
from arctic_platform.model.implementations.gpu import activation_offload
from arctic_platform.model.implementations.gpu import qwen3_5_dense_parity as hf_dense_parity
from arctic_platform.model.implementations.gpu.activation_offload import ActivationOffloadManager
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


def _config() -> Qwen3_5TextConfig:
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
    )
    config._attn_implementation = "sdpa"
    return config


def _tiny_model(dtype: torch.dtype = torch.bfloat16) -> modeling_qwen3_5.Qwen3_5ForCausalLM:
    set_seed(0)
    model = modeling_qwen3_5.Qwen3_5ForCausalLM(_config()).to(dtype)
    # The zero-initialized (1 + weight) norm weights would hide a dropped weight; give every parameter a value.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    return model.train()


def _input_ids() -> torch.Tensor:
    return torch.randint(0, 64, (1, _SEQ), generator=torch.Generator().manual_seed(1))


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


def _patched(model: nn.Module, *, sequence_parallel: bool = False) -> nn.Module:
    assert hf_dense_parity.apply_fp32_parity(model, sequence_parallel=sequence_parallel) == 1
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

    assert hf_dense_parity.remove_fp32_parity(model) == 1
    assert _instance_forwards(model) == []
    assert _beta_hooks(model) == 0
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


def _reference_fused_add_rms_norm(x, residual, weight, eps):
    # fp32 for the 16-bit dtypes; fp64 stays fp64 so the backward check can use it.
    compute = torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype
    summed = x.to(compute) + residual.to(compute)
    normed = summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.to(compute))
    return normed.to(x.dtype), summed.to(x.dtype)


def _reference_text_model(text_model, input_ids):
    """The patched model's loop, with every norm written out: the residual add in fp32 inside the next norm."""
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
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = _reference_fused_add_rms_norm(
                hidden, residual, layer.input_layernorm.weight, layer.input_layernorm.eps
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
            hidden, residual, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
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


@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "stock"])
@pytest.mark.parametrize("gradient_checkpointing", [True, 2], ids=["every-layer", "every-other-layer"])
def test_gradient_checkpointing_patch_forward_backward_matches_eager(fp32_parity, gradient_checkpointing):
    reference_model = _tiny_model()
    if fp32_parity:
        _patched(reference_model)
    checkpointed_model = _checkpointed(copy.deepcopy(reference_model), gradient_checkpointing)

    expected_output, expected_grads = _forward_backward(reference_model)
    _assert_same_run(*_forward_backward(checkpointed_model), expected_output, expected_grads)


@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "stock"])
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_transformers_gradient_checkpointing_forward_backward_matches_eager(use_reentrant, fp32_parity):
    reference_model = _tiny_model()
    if fp32_parity:
        _patched(reference_model)
    checkpointed_model = copy.deepcopy(reference_model)
    checkpointed_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})

    expected_output, expected_grads = _forward_backward(reference_model)
    _assert_same_run(*_forward_backward(checkpointed_model), expected_output, expected_grads)


@pytest.mark.parametrize("fp32_parity", [True, False], ids=["fp32-parity", "stock"])
def test_tensors_each_checkpointed_layer_receives(fp32_parity):
    model = _tiny_model()
    if fp32_parity:
        _patched(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    received = []
    for layer in model.model.layers:
        checkpoint = layer._gradient_checkpointing_func

        def recording(function, *args, checkpoint=checkpoint, **kwargs):
            received.append(sum(1 for arg in args if torch.is_tensor(arg) and arg.shape[-1] == 32))
            return checkpoint(function, *args, **kwargs)

        layer._gradient_checkpointing_func = recording
    _forward_backward(model)
    # With the patch every layer after the first holds the branch output and the residual at its boundary.
    assert received == ([1, 2, 2, 2] if fp32_parity else [1, 1, 1, 1])


def _eligible_on_any_device(self: ActivationOffloadManager, tensor: torch.Tensor) -> bool:
    """``ActivationOffloadManager._eligible`` without its CUDA-only device check and its skip counters."""
    if not self.enabled or isinstance(tensor, nn.Parameter):
        return False
    if tensor.numel() * tensor.element_size() < self.tensor_size_threshold:
        return False
    if not tensor.is_contiguous() and activation_offload._tensor_has_storage_overlap(tensor):
        return False
    return True


def test_activation_offload_forward_backward_matches_eager(monkeypatch):
    monkeypatch.setattr(ActivationOffloadManager, "_eligible", _eligible_on_any_device)
    reference_model = _patched(_tiny_model())
    offloaded_model = _checkpointed(copy.deepcopy(reference_model), True)
    manager = activation_offload.install_activation_offload(
        offloaded_model.model, keep_last_n=0, use_streams=False, tensor_size_threshold=0, pin_memory_enabled=False
    )

    expected_output, expected_grads = _forward_backward(reference_model)
    output, grads = _forward_backward(offloaded_model)

    assert manager.stats.offloaded_tensors > 0
    assert manager.stats.restored_tensors == manager.stats.offloaded_tensors
    _assert_same_run(output, grads, expected_output, expected_grads)


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
# The huggingface loader
# ---------------------------------------------------------------------------


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
        # Off installs nothing: the model is the stock one.
        assert _instance_forwards(model) == []
        assert _beta_hooks(model) == 0
        expected_output, expected_grads = _forward_backward(_tiny_model())
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
    reference = _patched(_tiny_model())
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
    stock = _patched(_tiny_model())
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
