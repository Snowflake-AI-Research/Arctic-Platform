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
"""Packed-sequence tiled rematerialization of attention and linear-attention mixers.

The recompute runtime is ``deepspeed.runtime.activation_checkpointing.tiled_rematerialization`` from the
DeepSpeed build that contains that module. This patch is the caller: it derives packed boundaries from
``position_ids`` when the mixer is not given ``cu_seq_lens_q``, and it runs each mixer forward one tile of
whole samples at a time.
"""

from __future__ import annotations

import inspect
import logging
import sys

import torch
import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch

logger = logging.getLogger(__name__)

_FLASH_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")
_HF_PACKED_MAX_LEN_NAMES = ("max_length_q", "max_length_k", "max_seqlen_q", "max_seqlen_k")
_GDN_BOUNDARY_RESET_KERNELS = (
    ("torch_chunk_gated_delta_rule", "cu_seqlens", "recurrent state", "fla (flash-linear-attention)"),
    ("causal_conv1d_fn", "seq_idx", "short-conv state", "causal-conv1d"),
)

_logged_position_ids_fallback = False
_logged_captured_position_ids = False
_captured_position_ids = None


def _hf_kernel_applicable_params(fn):
    """Kwarg names a transformers kernel-fallback wrapper lets through, or None when ``fn`` is not one."""
    while callable(fn):
        try:
            nonlocals = inspect.getclosurevars(fn).nonlocals
        except TypeError:
            return None
        if "applicable_params" in nonlocals:
            return nonlocals["applicable_params"]
        fn = getattr(fn, "__wrapped__", None)
    return None


def _gdn_boundary_reset_gaps(mixer_cls):
    """Reasons ``mixer_cls`` would not reset linear-attention state at every packed boundary."""
    module = sys.modules.get(mixer_cls.__module__)
    gaps = []
    for func_name, reset_kwarg, state, package in _GDN_BOUNDARY_RESET_KERNELS:
        params = _hf_kernel_applicable_params(getattr(module, func_name, None))
        if params is None:
            gaps.append(
                f"{state}: no transformers kernel-fallback wrapper named {func_name} in "
                f"{mixer_cls.__module__}, so boundary resets cannot be verified"
            )
        elif reset_kwarg not in params:
            gaps.append(
                f"{state}: {func_name} resolved to an implementation that drops '{reset_kwarg}' (install {package})"
            )
    return gaps


def _seq_idx_from_cu_seqlens(cu_seqlens, dtype):
    lengths = cu_seqlens.diff().long()
    sample_ids = torch.arange(lengths.numel(), device=cu_seqlens.device, dtype=dtype)
    return torch.repeat_interleave(sample_ids, lengths)


def _is_text_decoder_layer(mod):
    cls_name = type(mod).__name__
    if "Vision" in cls_name:
        return False
    if "DecoderLayer" in cls_name:
        return True
    has_mixer = hasattr(mod, "self_attn") or hasattr(mod, "linear_attn")
    return has_mixer and hasattr(mod, "mlp")


def _attn_implementation(model: nn.Module):
    config = getattr(model, "config", None)
    impl = getattr(config, "_attn_implementation", None)
    get_text_config = getattr(config, "get_text_config", None)
    if impl is None and callable(get_text_config):
        impl = getattr(get_text_config(), "_attn_implementation", None)
    return impl


def _install_top_level_position_ids_capture(model: nn.Module) -> None:
    """Record ``position_ids`` from the top-level forward for mixers the decoder does not pass them to."""
    if getattr(model, "_ap_position_ids_capture_installed", False):
        return

    def _capture(_module, _args, kwargs):
        global _captured_position_ids
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            _captured_position_ids = position_ids
        return None

    model.register_forward_pre_hook(_capture, with_kwargs=True)
    model._ap_position_ids_capture_installed = True


def _squeeze_batch(value):
    if torch.is_tensor(value):
        if value.dim() >= 2 and value.shape[0] == 1:
            return value.squeeze(0), True
        return value, False
    if isinstance(value, tuple):
        items = [_squeeze_batch(item) for item in value]
        if items and all(squeezed for _, squeezed in items):
            return tuple(item for item, _ in items), True
        return value, False
    return value, False


def _require_tiled_remat_runtime():
    try:
        from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import classify_packed_kwarg
        from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import (
            packed_seq_tiled_rematerialize_compute,
        )
        from deepspeed.utils import z3_leaf_parameter
    except ImportError as exc:
        raise ImportError(
            "tiled_remat requires deepspeed.runtime.activation_checkpointing.tiled_rematerialization. "
            f"Original import error: {exc}"
        ) from exc
    return classify_packed_kwarg, packed_seq_tiled_rematerialize_compute, z3_leaf_parameter


def _bind_call_kwargs(fwd_sig, mixer, hidden_states, args, kwargs):
    bound = fwd_sig.bind_partial(mixer, hidden_states, *args, **kwargs)
    bound.apply_defaults()
    call_kw = {}
    for name, value in bound.arguments.items():
        if name in ("self", "hidden_states"):
            continue
        param = fwd_sig.parameters.get(name)
        if param is not None and param.kind == param.VAR_KEYWORD:
            call_kw.update(value or {})
        else:
            call_kw[name] = value
    return call_kw


def _take_packed_cu_seqlens(call_kw):
    cu_seqlens = call_kw.pop("cu_seq_lens_q", None)
    if cu_seqlens is None:
        return call_kw.pop("cu_seq_lens_k", None)
    call_kw.pop("cu_seq_lens_k", None)
    return cu_seqlens


def _drop_hf_max_len(call_kw):
    for meta_name in _HF_PACKED_MAX_LEN_NAMES:
        call_kw.pop(meta_name, None)


def _position_ids_for_mixer(hidden_states, call_kw):
    """Use ``position_ids`` on the call, or the top-level capture when the decoder did not pass them."""
    global _logged_captured_position_ids
    position_ids = call_kw.get("position_ids")
    if position_ids is not None or _captured_position_ids is None:
        return position_ids
    candidate = _captured_position_ids
    n_tokens = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
    described = int(candidate.reshape(-1).shape[0])
    if described != int(n_tokens):
        raise RuntimeError(
            "tiled_remat: captured position_ids describe "
            f"{described} tokens but the mixer forward was handed {int(n_tokens)}. "
            "The capture is from a different forward pass; this is only safe with one forward "
            "per backward (micro batch 1)."
        )
    if not _logged_captured_position_ids:
        logger.info("tiled_remat: using position_ids captured at the top-level forward")
        _logged_captured_position_ids = True
    return candidate


def _resolve_cu_seqlens(mixer_name, cu_seqlens, position_ids, prepare_fa_kwargs_from_position_ids):
    global _logged_position_ids_fallback
    if cu_seqlens is None and position_ids is not None:
        (cu_seq_lens_q, _), _ = prepare_fa_kwargs_from_position_ids(position_ids)
        cu_seqlens = cu_seq_lens_q
        if not _logged_position_ids_fallback:
            logger.info("tiled_remat: derived cu_seqlens from position_ids")
            _logged_position_ids_fallback = True
    if cu_seqlens is None:
        raise RuntimeError(
            f"tiled_remat: {mixer_name}.forward has no cu_seq_lens_q/cu_seq_lens_k and no "
            "position_ids. Packed batches are required."
        )
    return cu_seqlens


def _resolve_seq_idx(mixer_name, call_kw, cu_seqlens, position_ids, needs_seq_idx):
    seq_idx = call_kw.pop("seq_idx", None)
    if seq_idx is None and needs_seq_idx and position_ids is not None:
        seq_idx = _seq_idx_from_cu_seqlens(cu_seqlens, torch.int32).reshape(position_ids.shape)
    if seq_idx is None and needs_seq_idx:
        raise RuntimeError(
            f"tiled_remat: {mixer_name}.forward needs seq_idx, or position_ids so it can be "
            "derived, otherwise the short conv slides across packed samples"
        )
    if seq_idx is None:
        return None
    expected_seq_idx = _seq_idx_from_cu_seqlens(cu_seqlens, seq_idx.dtype)
    if not torch.equal(seq_idx.reshape(-1), expected_seq_idx):
        raise RuntimeError("tiled_remat: seq_idx disagrees with cu_seq_lens_q")
    return seq_idx


def _hidden_without_batch(hidden_states):
    if hidden_states.dim() != 3:
        return False, hidden_states
    if hidden_states.shape[0] != 1:
        raise RuntimeError(f"tiled_remat requires micro batch 1, got hidden_states.shape={tuple(hidden_states.shape)}")
    return True, hidden_states.squeeze(0)


def _trainable_params(mixer, z3_leaf_parameter):
    params = [parameter for parameter in mixer.parameters() if parameter.requires_grad]
    zero3 = [parameter for parameter in params if hasattr(parameter, "ds_id")]
    if zero3 and not any(z3_leaf_parameter(parameter) for parameter in zero3):
        raise RuntimeError(
            "tiled_remat requires ZeRO-2. ZeRO-3 parameters on this mixer are not leaf modules, "
            "and tiling calls the mixer once per tile."
        )
    return params


def _is_cache_kwarg(name):
    if name == "cache_position":
        return False
    lowered = name.lower()
    cache_names = ("past_key_value", "past_key_values", "cache_params", "cache")
    return lowered in cache_names or "cache" in lowered


def _classify_call_kwarg(name, value, num_samples, classify_packed_kwarg):
    if _is_cache_kwarg(name):
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"tiled_remat: {name} must be None in training, got {type(value).__name__}")
        return "pass", None, False
    squeezed_value, was_squeezed = _squeeze_batch(value)
    try:
        kind = classify_packed_kwarg(name, squeezed_value, num_samples)
    except TypeError:
        if value is None:
            return "pass", None, False
        raise
    return kind, squeezed_value, was_squeezed


def _split_call_kwargs(call_kw, num_samples, classify_packed_kwarg):
    squeezed_names = set()
    to_shard = {}
    to_pass = {}
    for name, value in call_kw.items():
        kind, squeezed_value, was_squeezed = _classify_call_kwarg(name, value, num_samples, classify_packed_kwarg)
        if was_squeezed:
            squeezed_names.add(name)
        if kind == "token":
            to_shard[name] = squeezed_value
        else:
            to_pass[name] = squeezed_value
    return squeezed_names, to_shard, to_pass


def _restore_batch_dim(value):
    if torch.is_tensor(value):
        return value.unsqueeze(0)
    return tuple(item.unsqueeze(0) for item in value)


def _mixer_tensor(mixer_name, out, orig_tuple_len):
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
        extra = [item for item in out[1:] if item is not None]
        if extra:
            raise TypeError(f"tiled_remat: {mixer_name} returned extra outputs")
        orig_tuple_len[:] = [len(out)]
        return out[0]
    raise TypeError(f"tiled_remat: {mixer_name} returned {type(out)}")


def _run_one_tile(orig, mixer, hidden, kw, had_batch_dim, squeezed_names, seq_idx, orig_tuple_len):
    tile_cu_seqlens = kw.pop("cu_seqlens")
    local_max = int(tile_cu_seqlens.diff().max())
    for name in squeezed_names:
        if name not in kw:
            continue
        kw[name] = _restore_batch_dim(kw[name])
    kw["cu_seq_lens_q"] = tile_cu_seqlens
    kw["cu_seq_lens_k"] = tile_cu_seqlens
    kw["max_length_q"] = local_max
    kw["max_length_k"] = local_max
    if seq_idx is not None:
        tile_seq_idx = _seq_idx_from_cu_seqlens(tile_cu_seqlens, seq_idx.dtype)
        kw["seq_idx"] = tile_seq_idx.reshape(*seq_idx.shape[:-1], -1)
    hidden_in = hidden.unsqueeze(0) if had_batch_dim else hidden
    result = _mixer_tensor(type(mixer).__name__, orig(mixer, hidden_in, **kw), orig_tuple_len)
    return result.squeeze(0) if had_batch_dim else result


def _make_tiled_forward(orig, fwd_sig, cls_budget, needs_seq_idx, prepare_fa, classify, compute, z3_leaf_parameter):
    def tiled_forward(self, hidden_states, *args, **kwargs):
        call_kw = _bind_call_kwargs(fwd_sig, self, hidden_states, args, kwargs)
        cu_seqlens = _take_packed_cu_seqlens(call_kw)
        _drop_hf_max_len(call_kw)
        position_ids = _position_ids_for_mixer(hidden_states, call_kw)
        cu_seqlens = _resolve_cu_seqlens(type(self).__name__, cu_seqlens, position_ids, prepare_fa)
        num_samples = int(cu_seqlens.shape[0]) - 1
        seq_idx = _resolve_seq_idx(type(self).__name__, call_kw, cu_seqlens, position_ids, needs_seq_idx)
        had_batch_dim, hidden_states_2d = _hidden_without_batch(hidden_states)
        params = _trainable_params(self, z3_leaf_parameter)
        squeezed_names, to_shard, to_pass = _split_call_kwargs(call_kw, num_samples, classify)
        orig_tuple_len = []

        def _fn(hidden, **kw):
            return _run_one_tile(orig, self, hidden, kw, had_batch_dim, squeezed_names, seq_idx, orig_tuple_len)

        out2d = compute(
            _fn,
            hidden_states_2d,
            cu_seqlens,
            shards=None,
            token_budget=cls_budget,
            compute_params=params,
            kwargs_to_token_shard=to_shard,
            kwargs_to_pass=to_pass,
        )
        out = out2d.unsqueeze(0) if had_batch_dim else out2d
        if not orig_tuple_len:
            return out
        return (out,) + (None,) * (orig_tuple_len[0] - 1)

    return tiled_forward


def _zero_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        dropout = getattr(module, "attention_dropout", None)
        if isinstance(dropout, float):
            module.attention_dropout = 0.0


def _text_mixers(model: nn.Module):
    mixers = []
    linear_attn_classes = set()
    for layer in model.modules():
        if not _is_text_decoder_layer(layer):
            continue
        mixer = getattr(layer, "self_attn", None)
        if mixer is None:
            mixer = getattr(layer, "linear_attn", None)
            if mixer is not None:
                linear_attn_classes.add(type(mixer))
        if mixer is not None:
            mixers.append(mixer)
    return mixers, linear_attn_classes


def _require_boundary_resets(linear_attn_classes) -> None:
    for cls in linear_attn_classes:
        gaps = _gdn_boundary_reset_gaps(cls)
        if gaps:
            raise RuntimeError(
                f"tiled_remat: {cls.__name__} does not reset at packed sample boundaries: " + "; ".join(gaps)
            )


def install_packed_seq_tiled_remat(model: nn.Module, token_budget: int) -> list[type]:
    """Tile every text decoder's ``self_attn`` or ``linear_attn`` forward along the packed token axis."""
    impl = _attn_implementation(model)
    if impl not in _FLASH_IMPLEMENTATIONS:
        raise ValueError(
            "tiled_remat requires flash attention "
            f"({', '.join(_FLASH_IMPLEMENTATIONS)}); model attention implementation is {impl!r}"
        )
    classify_packed_kwarg, packed_seq_tiled_rematerialize_compute, z3_leaf_parameter = _require_tiled_remat_runtime()
    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    _zero_dropout(model)
    _install_top_level_position_ids_capture(model)
    mixers, linear_attn_classes = _text_mixers(model)
    _require_boundary_resets(linear_attn_classes)

    patched: list[type] = []
    for mixer in mixers:
        cls = type(mixer)
        if cls in patched:
            continue
        orig_fwd = cls.forward
        sig = inspect.signature(orig_fwd)

        cls.forward = _make_tiled_forward(
            orig_fwd,
            sig,
            token_budget,
            cls in linear_attn_classes,
            prepare_fa_kwargs_from_position_ids,
            classify_packed_kwarg,
            packed_seq_tiled_rematerialize_compute,
            z3_leaf_parameter,
        )
        patched.append(cls)

    n_modules = sum(1 for module in model.modules() if type(module) in patched)
    logger.info(
        "tiled_remat token_budget=%s classes=%s modules=%s",
        token_budget,
        [cls.__name__ for cls in patched],
        n_modules,
    )
    if n_modules == 0:
        raise RuntimeError("tiled_remat requested but no self_attn/linear_attn mixer modules found")
    return patched


@register_patch("tiled_remat")
def apply_tiled_remat(model: nn.Module, ctx: LoaderContext) -> None:
    settings = ctx.spec.patches.tiled_remat
    if settings is None:
        raise ValueError("tiled_remat patch requires patches.tiled_remat configuration")
    install_packed_seq_tiled_remat(model, settings.token_budget)
