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

"""Weight conversion between a Nemotron-H Omni checkpoint and Arctic's composite model.

The checkpoint (``nvidia/NVIDIA-Nemotron-3.5-Super-VL-*``) keeps the layout of NVIDIA's remote code:

  - language model: ``language_model.backbone.*`` and ``language_model.lm_head.weight`` (Hugging Face Nemotron-H
    names, converted layer by layer as in ``converting_nemotron_h``); ``language_model.mtp.*`` is dropped, as for
    Nemotron-H.
  - vision tower: ``vision_model.radio_model.model.*`` (timm names), plus ``vision_model.summary_idxs`` and the
    ``vision_model.radio_model.input_conditioner.*`` normalization, which the processor applies, so it is dropped.
  - projector: ``mlp1.{0,1,3}`` and ``vision_projector.vision_final_layernorm.*``.

Arctic's model holds transformers' native RADIO tower and projector under ``model.visual.``, with transformers'
names, and Arctic's Nemotron-H under ``model.language_model.``. The vision renames are transformers' own ``radio``
and ``NemotronH_Omni_Reasoning_V3`` checkpoint conversion rules. One addition: transformers 5.20 has no rule for
``vision_projector.vision_final_layernorm``, so its native model keeps that norm at its identity initialization;
the checkpoint's values (near, but not exactly, identity) are loaded here, as NVIDIA's remote code and the trained
model use them.

Both directions take any subset of the weights (a whole model, one decoder layer's, or everything outside the
layers) and convert it in place.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .converting_nemotron_h import _rename_keys
from .converting_nemotron_h import convert_hf_layer_to_prime
from .converting_nemotron_h import convert_prime_layer_to_hf

CHECKPOINT_BACKBONE = "language_model.backbone."
PRIME_LANGUAGE_MODEL = "model.language_model."
LM_HEAD = ("language_model.lm_head.weight", "lm_head.weight")
# Hugging Face Nemotron-H's global names and Arctic's, within the language model.
TEXT_GLOBAL_RENAMES = (
    ("model.embeddings.weight", "model.embed_tokens.weight"),
    ("model.norm_f.weight", "model.norm.weight"),
)

# Checkpoint prefixes with no counterpart in the model.
DROPPED_PREFIXES = ("language_model.mtp.", "vision_model.radio_model.input_conditioner.")

# (checkpoint prefix, model prefix), most specific first.
VISION_RENAMES = (
    (
        "vision_model.radio_model.model.patch_generator.video_embedder.",
        "model.visual.vision_model.embeddings.video_patch_projection.",
    ),
    (
        "vision_model.radio_model.model.patch_generator.embedder.",
        "model.visual.vision_model.embeddings.patch_projection.",
    ),
    (
        "vision_model.radio_model.model.patch_generator.pos_embed",
        "model.visual.vision_model.embeddings.position_embedding",
    ),
    (
        "vision_model.radio_model.model.patch_generator.cls_token.token",
        "model.visual.vision_model.embeddings.cls_register_token",
    ),
    ("vision_model.radio_model.model.blocks.", "model.visual.vision_model.encoder.layer."),
    ("vision_model.summary_idxs", "model.visual.vision_model.summary_idxs"),
    ("vision_projector.vision_final_layernorm.", "model.visual.vision_final_layernorm."),
    ("mlp1.0.", "model.visual.multi_modal_projector.layer_norm."),
    ("mlp1.1.", "model.visual.multi_modal_projector.linear_1."),
    ("mlp1.3.", "model.visual.multi_modal_projector.linear_2."),
)
VISION_BLOCK = "model.visual.vision_model.encoder.layer."


def is_checkpoint_state_dict(state_dict) -> bool:
    return any(name.startswith(CHECKPOINT_BACKBONE) for name in state_dict)


def is_prime_state_dict(state_dict) -> bool:
    return any(name.startswith(PRIME_LANGUAGE_MODEL) for name in state_dict)


def _layer_indices(state_dict) -> list[int]:
    return sorted({int(name.split(".")[2]) for name in state_dict if name.startswith("model.layers.")})


def _split_qkv(state_dict: dict[str, Tensor]) -> None:
    """timm's fused ``attn.qkv`` into transformers' ``attention.{q,k,v}_proj``; ``attn.proj`` into ``o_proj``."""
    for key in [key for key in state_dict if key.startswith(VISION_BLOCK) and ".attn." in key]:
        value = state_dict.pop(key)
        block, rest = key.split(".attn.", 1)
        if rest.startswith("qkv."):
            suffix = rest[len("qkv.") :]
            for name, part in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                state_dict[f"{block}.attention.{name}.{suffix}"] = part.contiguous()
        elif rest.startswith("proj."):
            state_dict[f"{block}.attention.o_proj.{rest[len('proj.'):]}"] = value
        else:
            raise KeyError(f"unexpected RADIO attention weight {key!r}")


def _fuse_qkv(state_dict: dict[str, Tensor]) -> None:
    for key in [key for key in state_dict if key.startswith(VISION_BLOCK) and ".attention.q_proj." in key]:
        block, suffix = key.split(".attention.q_proj.", 1)
        parts = [state_dict.pop(f"{block}.attention.{name}.{suffix}") for name in ("q_proj", "k_proj", "v_proj")]
        state_dict[f"{block}.attn.qkv.{suffix}"] = torch.cat(parts, dim=0)
    for key in [key for key in state_dict if key.startswith(VISION_BLOCK) and ".attention.o_proj." in key]:
        block, suffix = key.split(".attention.o_proj.", 1)
        state_dict[f"{block}.attn.proj.{suffix}"] = state_dict.pop(key)


def convert_checkpoint_to_prime(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Checkpoint weights in Arctic's composite layout, in place."""
    from .modeling_nemotron_h import _infer_layer_type_hf

    for key in [key for key in state_dict if key.startswith(DROPPED_PREFIXES)]:
        del state_dict[key]
    checkpoint_head, prime_head = LM_HEAD
    if checkpoint_head in state_dict:
        state_dict[prime_head] = state_dict.pop(checkpoint_head)
    stray = [
        key for key in state_dict if key.startswith("language_model.") and not key.startswith(CHECKPOINT_BACKBONE)
    ]
    if stray:
        raise KeyError(f"unexpected Nemotron-H weights {stray[:5]}")

    text = {
        "model." + key[len(CHECKPOINT_BACKBONE) :]: state_dict.pop(key)
        for key in list(state_dict)
        if key.startswith(CHECKPOINT_BACKBONE)
    }
    for hf_name, prime_name in TEXT_GLOBAL_RENAMES:
        if hf_name in text:
            text[prime_name] = text.pop(hf_name)
    for index in _layer_indices(text):
        convert_hf_layer_to_prime(text, index, _infer_layer_type_hf(text, index))
    _rename_keys(text, "model.", PRIME_LANGUAGE_MODEL)
    state_dict.update(text)

    for old, new in VISION_RENAMES:
        _rename_keys(state_dict, old, new)
    _split_qkv(state_dict)
    return state_dict


def convert_prime_to_checkpoint(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Arctic's composite weights in the checkpoint layout (without the dropped MTP and conditioner), in place."""
    from .modeling_nemotron_h import _infer_layer_type_prime

    checkpoint_head, prime_head = LM_HEAD
    if prime_head in state_dict:
        state_dict[checkpoint_head] = state_dict.pop(prime_head)

    text = {
        "model." + key[len(PRIME_LANGUAGE_MODEL) :]: state_dict.pop(key)
        for key in list(state_dict)
        if key.startswith(PRIME_LANGUAGE_MODEL)
    }
    for hf_name, prime_name in TEXT_GLOBAL_RENAMES:
        if prime_name in text:
            text[hf_name] = text.pop(prime_name)
    for index in _layer_indices(text):
        convert_prime_layer_to_hf(text, index, _infer_layer_type_prime(text, index))
    _rename_keys(text, "model.", CHECKPOINT_BACKBONE)
    state_dict.update(text)

    _fuse_qkv(state_dict)
    for old, new in VISION_RENAMES:
        _rename_keys(state_dict, new, old)
    return state_dict
