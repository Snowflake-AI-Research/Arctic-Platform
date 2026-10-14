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
#
# ``_text_model_forward_body`` and ``_decoder_layer_forward`` are adapted from
# transformers' models/qwen3_5/modeling_qwen3_5.py (transformers 5.17.0),
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team, licensed under the
# Apache License, Version 2.0.
"""vLLM-style rounding for the stock transformers dense Qwen3.5 text model, applied after loading.

``apply_fp32_parity`` changes three operations of every ``Qwen3_5TextModel`` in a model so that they round as vLLM
does, in the same way as the custom Qwen3.5-MoE implementation:

1. Each residual add is evaluated in fp32 inside the following RMSNorm, which normalizes the fp32 sum (vLLM's
   ``fused_add_rms_norm``). The residual stream stays in the dtype of the branch output. Decoder layers take and
   return ``(hidden_states, residual)``, so the text model's forward is replaced by a copy of the stock forward whose
   layer loop carries the pair and whose final norm consumes it. The fused norm uses each norm module's weight and
   epsilon, not its forward, so a kernel that replaces the norm's forward (the liger patch's) runs, of the
   decoder-layer and final norms, only where a norm is called on one tensor: the first layer's input norm.
2. The partial rotary embedding of q and k is evaluated in fp32 and cast back once. Each patched attention module
   gets its class's forward with ``apply_rotary_pos_emb`` bound to the fp32 rotation; the module-level function and
   every unpatched model keep the stock rotation.
3. The GatedDeltaNet beta gate is a sigmoid of fp32 ``b``: a forward hook makes ``in_proj_b`` return fp32, and the
   stock forward uses ``b`` only for ``beta = b.sigmoid()``. With sequence parallelism the sequence-parallel wrapper
   replaces that forward and takes the setting itself (its ``fp32_beta`` argument), so no hook is installed.

The code each patched model runs, read from the classes of its own modules, is checked against the source of the
verified transformers version before anything is patched; a mismatch raises instead of patching part of the model or
silently keeping the stock rounding. ``check_stock_classes`` runs the same check on the classes a model would be built
from, so a loader can fail before the weights load. Every other model is left alone, and calling
``apply_fp32_parity`` again is a no-op. ``remove_fp32_parity`` restores the stock forwards.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import types
from typing import Any
from typing import Callable

import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
from transformers.utils.output_capturing import _active_collector

from arctic_platform.model.implementations.gpu.vllm_rounding import FusedAddRMSNorm
from arctic_platform.model.implementations.gpu.vllm_rounding import rope_rotary_slice_fp32

VERIFIED_TRANSFORMERS_VERSION = "5.17.0"

# The Hugging Face model types of the dense Qwen3.5 family, for a loader that checks before loading weights.
QWEN3_5_DENSE_MODEL_TYPES = ("qwen3_5", "qwen3_5_text")

# SHA-256 of the source of the stock code this module copies, rebinds or relies on, in the verified transformers
# version: the text ``inspect.getsource`` gives for it, decorators included. ``Qwen3_5RMSNorm.forward`` is listed
# because the fused norm reproduces its (1 + weight) scaling and reads its ``eps``; ``Qwen3_5GatedDeltaNet.forward``
# because the beta hook relies on ``b`` being used only for the beta gate.
_STOCK_SOURCE_SHA256: dict[str, str] = {
    "Qwen3_5TextModel.forward": "8059c2ff169e914a0fe184fbb24a45129a5d75f91594165130532be9d337d8da",
    "Qwen3_5DecoderLayer.forward": "07a700417d4e0eaec79fb27d09a86a053928f62197d228ab4c64b903eb344d5a",
    "Qwen3_5Attention.forward": "282e26183eb923131affae687e924eab5cda69c6128b4956a76ecc2230bec424",
    "apply_rotary_pos_emb": "ab8feb64f19555713efc4c559ce306b35205b1197f8431784585999d9a5b6662",
    "Qwen3_5GatedDeltaNet.forward": "bd25761e1031fff0f6b24115dff8175ee9c8fb585287a4026b73a12bb0b5d816",
    "Qwen3_5RMSNorm.forward": "350041feaa40063352594579e212741fdff004594990c36162458d778dae1a9e",
}

_PATCHED_ATTRIBUTE = "_arctic_fp32_parity"
_BETA_HOOK_ATTRIBUTE = "_arctic_fp32_parity_beta_hook"


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


@functools.lru_cache(maxsize=1)
def _file_sources() -> dict[str, str]:
    """The source of each stock object, read from the installed transformers file.

    A patch that replaces a class attribute or rebinds a module-level name at run time cannot change this, so it is
    the reference both for the version check and for recognizing a replaced function.
    """
    path = inspect.getsourcefile(modeling)
    assert path is not None
    with open(path) as source_file:
        text = source_file.read()
    lines = text.splitlines(keepends=True)
    sources = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "apply_rotary_pos_emb":
            functions = {node.name: node}
        elif isinstance(node, ast.ClassDef) and f"{node.name}.forward" in _STOCK_SOURCE_SHA256:
            functions = {
                f"{node.name}.forward": item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "forward"
            }
        else:
            continue
        for name, function in functions.items():
            start = min([decorator.lineno for decorator in function.decorator_list] + [function.lineno])
            assert function.end_lineno is not None
            sources[name] = "".join(lines[start - 1 : function.end_lineno])
    return sources


def _fail(cause: str) -> None:
    raise RuntimeError(
        "fp32_parity patches the transformers Qwen3.5 dense text model and is verified against transformers "
        f"{VERIFIED_TRANSFORMERS_VERSION}, but {cause}. Set the huggingface loader option fp32_parity to false "
        "to load the stock model."
    )


def _check_functions(functions: dict[str, Callable[..., Any]]) -> None:
    """Fail unless the installed source is the verified one and each live function in ``functions`` is that source."""
    import transformers

    sources = _file_sources()
    changed = [name for name in _STOCK_SOURCE_SHA256 if _digest(sources.get(name, "")) != _STOCK_SOURCE_SHA256[name]]
    if len(changed) > 0:
        _fail(f"the installed transformers {transformers.__version__} has different source for {', '.join(changed)}")
    replaced = []
    for name, function in sorted(functions.items()):
        if _digest(inspect.getsource(function)) != _STOCK_SOURCE_SHA256[name]:
            original = inspect.unwrap(function)
            replaced.append(f"{name} by {original.__module__}.{original.__qualname__}")
    if len(replaced) > 0:
        _fail(f"in this process {'; '.join(replaced)} has replaced the stock code")


def _class_functions(
    text_model_class, decoder_layer_class, attention_class, gated_delta_net_class
) -> dict[str, Callable[..., Any]]:
    return {
        "Qwen3_5TextModel.forward": text_model_class.forward,
        "Qwen3_5DecoderLayer.forward": decoder_layer_class.forward,
        "Qwen3_5Attention.forward": attention_class.forward,
        # The rotary function the attention forward resolves at call time, which the patch rebinds.
        "apply_rotary_pos_emb": attention_class.forward.__globals__["apply_rotary_pos_emb"],
        "Qwen3_5GatedDeltaNet.forward": gated_delta_net_class.forward,
    }


def check_stock_classes() -> None:
    """Check the classes a dense Qwen3.5 text model would be built from now, before it is built.

    The norm class is not checked here: the liger patch replaces the module's ``Qwen3_5RMSNorm`` with a compatible
    one, and ``apply_fp32_parity`` checks the norms a model actually has.
    """
    _check_functions(
        _class_functions(
            modeling.Qwen3_5TextModel,
            modeling.Qwen3_5DecoderLayer,
            modeling.Qwen3_5Attention,
            modeling.Qwen3_5GatedDeltaNet,
        )
    )


def _norm_epsilon(norm: nn.Module) -> float | None:
    """The epsilon of a (1 + weight) RMSNorm module, or ``None`` when the module is not one.

    The stock class is recognized by its forward's source. The liger patch's Qwen3.5 norm keeps the (1 + weight)
    scaling as ``offset == 1.0``, computes in fp32 (``casting_mode == "gemma"``) and stores the epsilon as
    ``variance_epsilon``.
    """
    if _digest(inspect.getsource(type(norm).forward)) == _STOCK_SOURCE_SHA256["Qwen3_5RMSNorm.forward"]:
        return norm.eps
    if getattr(norm, "offset", None) == 1.0 and getattr(norm, "casting_mode", None) == "gemma":
        return getattr(norm, "variance_epsilon", None)
    return None


def _check_text_model(text_model: nn.Module) -> None:
    """Check the code ``text_model``'s own modules run, and that every norm is a (1 + weight) RMSNorm."""
    layers = list(text_model.layers)
    layer_classes = {type(layer) for layer in layers}
    attention_classes = {type(layer.self_attn) for layer in layers if layer.block_type == "full_attention"}
    gated_delta_net_classes = {type(layer.linear_attn) for layer in layers if layer.block_type == "linear_attention"}
    # A model without one kind of layer is checked against the module's class for it.
    if len(layer_classes) == 0:
        layer_classes = {modeling.Qwen3_5DecoderLayer}
    if len(attention_classes) == 0:
        attention_classes = {modeling.Qwen3_5Attention}
    if len(gated_delta_net_classes) == 0:
        gated_delta_net_classes = {modeling.Qwen3_5GatedDeltaNet}
    for layer_class in layer_classes:
        for attention_class in attention_classes:
            for gated_delta_net_class in gated_delta_net_classes:
                _check_functions(
                    _class_functions(type(text_model), layer_class, attention_class, gated_delta_net_class)
                )
    norms = [text_model.norm]
    for layer in layers:
        norms.extend([layer.input_layernorm, layer.post_attention_layernorm])
    unrecognized = sorted({type(norm).__qualname__ for norm in norms if _norm_epsilon(norm) is None})
    if len(unrecognized) > 0:
        _fail(f"its norms of type {', '.join(unrecognized)} are not a (1 + weight) RMSNorm it recognizes")


def _fused_add_rms_norm(norm: nn.Module, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Add ``residual`` to ``x`` in fp32 and apply ``norm``'s (1 + weight) RMSNorm to the fp32 sum.

    Returns the normalized value and the sum, both in ``x``'s dtype; the sum is the next residual.
    """
    return FusedAddRMSNorm.apply(x, residual, norm.weight, _norm_epsilon(norm))


def _apply_rotary_pos_emb_fp32(q, k, cos, sin, unsqueeze_dim=1):
    """``apply_rotary_pos_emb`` with the rotation evaluated in fp32, for ``[batch, heads, seq, head_dim]`` q and k."""
    if unsqueeze_dim != 1:
        raise ValueError(f"the fp32 rotary embedding expects unsqueeze_dim=1, got {unsqueeze_dim}")
    return rope_rotary_slice_fp32(q, cos, sin), rope_rotary_slice_fp32(k, cos, sin)


def _attention_forward_with_fp32_rope(attention_class) -> types.FunctionType:
    """``attention_class``'s forward, with its global ``apply_rotary_pos_emb`` resolved to the fp32 rotation.

    The code object is the class's own; only the name lookup differs. The globals are a copy of the forward's own,
    taken when the patch is applied.
    """
    stock = attention_class.forward
    module_globals = dict(stock.__globals__)
    module_globals["apply_rotary_pos_emb"] = _apply_rotary_pos_emb_fp32
    return types.FunctionType(stock.__code__, module_globals, stock.__name__, stock.__defaults__, stock.__closure__)


def _fp32_output(module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
    """Return a 16-bit ``in_proj_b`` output in fp32, so that ``b.sigmoid()`` is a sigmoid of fp32 ``b``."""
    if output.dtype in (torch.bfloat16, torch.float16):
        return output.float()
    return output


def _decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(branch_output, residual)``.

    Each residual add is deferred into the next norm, which performs it in fp32. ``hidden_states`` is the previous
    layer's MLP output, or the embeddings for the first layer, which receives ``residual=None``. The returned
    ``branch_output`` is this layer's MLP output, not yet added to the returned ``residual``; the next layer's input
    norm, or the model's final norm, adds them. The token mixer is the stock one.
    """
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = _fused_add_rms_norm(self.input_layernorm, hidden_states, residual)

    # Token Mixer
    if self.block_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
    elif self.block_type == "full_attention":
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    # Fully Connected
    hidden_states, residual = _fused_add_rms_norm(self.post_attention_layernorm, hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


def _text_model_forward_body(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """The stock ``Qwen3_5TextModel.forward`` with a layer loop that carries ``(hidden_states, residual)``."""
    collected = _active_collector.get()
    if collected is not None and "hidden_states" in collected:
        raise NotImplementedError(
            "output_hidden_states is not available with fp32_parity: the decoder layers pass the branch output and "
            "the residual separately, not their sum. Set the huggingface loader option fp32_parity to false to get "
            "the per-layer hidden states."
        )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = modeling.DynamicCache(config=self.config)

    # the hard coded `4` is for text, temporal, height and width.
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": modeling.create_causal_mask(**mask_kwargs),
            "linear_attention": modeling.create_recurrent_attention_mask(**mask_kwargs),
        }

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    residual = None
    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        hidden_states, residual = decoder_layer(
            hidden_states,
            residual,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask_mapping[self.config.layer_types[i]],
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    if residual is None:
        hidden_states = self.norm(hidden_states)
    else:
        hidden_states, _ = _fused_add_rms_norm(self.norm, hidden_states, residual)

    return modeling.Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _text_model_forward():
    """``_text_model_forward_body`` under the stock forward's decorators."""
    return modeling.merge_with_config_defaults(modeling.capture_outputs(_text_model_forward_body))


def _text_models(model: nn.Module) -> list[nn.Module]:
    return [module for module in model.modules() if type(module) is modeling.Qwen3_5TextModel]


def apply_fp32_parity(model: nn.Module, *, sequence_parallel: bool) -> int:
    """Make every stock dense Qwen3.5 text model in ``model`` round like vLLM; return how many were patched.

    ``sequence_parallel`` says whether the sequence-parallel wrapper will replace the GatedDeltaNet forwards; then the
    wrapper sets the beta gate's precision and no beta hook is installed. A model without a ``Qwen3_5TextModel`` is
    left alone. A text model patched before is skipped.
    """
    text_models = [module for module in _text_models(model) if not getattr(module, _PATCHED_ATTRIBUTE, False)]
    for text_model in text_models:
        _check_text_model(text_model)
    text_model_forward = _text_model_forward()
    for text_model in text_models:
        for layer in text_model.layers:
            layer.forward = types.MethodType(_decoder_layer_forward, layer)
            if layer.block_type == "full_attention":
                attention_forward = _attention_forward_with_fp32_rope(type(layer.self_attn))
                layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
            elif layer.block_type == "linear_attention" and not sequence_parallel:
                in_proj_b = layer.linear_attn.in_proj_b
                setattr(in_proj_b, _BETA_HOOK_ATTRIBUTE, in_proj_b.register_forward_hook(_fp32_output))
        text_model.forward = types.MethodType(text_model_forward, text_model)
        setattr(text_model, _PATCHED_ATTRIBUTE, True)
    return len(text_models)


def remove_fp32_parity(model: nn.Module) -> int:
    """Restore the stock forwards of every text model ``apply_fp32_parity`` patched in ``model``; return the count."""
    restored = 0
    for text_model in _text_models(model):
        if not getattr(text_model, _PATCHED_ATTRIBUTE, False):
            continue
        for layer in text_model.layers:
            layer.__dict__.pop("forward", None)
            if layer.block_type == "full_attention":
                layer.self_attn.__dict__.pop("forward", None)
            elif layer.block_type == "linear_attention":
                handle = layer.linear_attn.in_proj_b.__dict__.pop(_BETA_HOOK_ATTRIBUTE, None)
                if handle is not None:
                    handle.remove()
        text_model.__dict__.pop("forward", None)
        delattr(text_model, _PATCHED_ATTRIBUTE)
        restored += 1
    return restored
