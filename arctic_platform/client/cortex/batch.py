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
"""Readable JSON -> the batch `ArcticClient.fwd_bwd` / `generate` already take.

A smoke test shouldn't require writing tensors by hand, so a spec file may say
"tokenize these texts" and this turns that into real tensors. Pre-tokenized data
works too (``payload.kwargs``, or ``input_ids`` / ``labels`` as JSON lists), which
keeps the path usable with no tokenizer installed.

This stops at the batch dict: serialization is the transport's job, so nothing
here touches the wire.

``torch`` and ``transformers`` are imported lazily -- control-plane commands share
this module's package and shouldn't pay for them.
"""

from __future__ import annotations

from typing import Any


def build_fwd_bwd_batch(spec: dict[str, Any]) -> dict[str, Any]:
    """A fwd-bwd spec -> ``{"args": [...], "kwargs": {...}}``."""
    payload = _payload(spec, "fwd-bwd")
    return {"args": list(_args(payload)), "kwargs": build_fwd_bwd_kwargs(payload)}


def read_generate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """A generate spec -> validated `ArcticClient.generate` kwargs."""
    payload = _payload(spec, "generate")

    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("generate spec needs a non-empty prompts list")

    sampling_params = payload.get("sampling_params")
    if isinstance(sampling_params, list):
        if len(sampling_params) != len(prompts):
            raise ValueError(
                f"generate sampling_params list has {len(sampling_params)} items but there are {len(prompts)} prompts"
            )
        if any(item is not None and not isinstance(item, dict) for item in sampling_params):
            raise ValueError("generate sampling_params list items must be objects or null")
    elif sampling_params is not None and not isinstance(sampling_params, dict):
        raise ValueError("generate sampling_params must be an object or list")

    strict = payload.get("strict")
    return {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "routing_key": payload.get("routing_key"),
        "strict": False if strict is None else _boolean(strict, "generate strict"),
    }


def should_poll(spec: dict[str, Any]) -> bool:
    """``"poll": false`` submits without waiting for the result."""
    return _boolean(spec.get("poll", True), "poll")


def build_fwd_bwd_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Tensor kwargs for one training batch, from texts or from tensor data."""
    if "kwargs" in payload:
        explicit = payload["kwargs"]
        if not isinstance(explicit, dict):
            raise ValueError("fwd-bwd payload kwargs must be an object")
        return _tensorize(explicit)

    import torch

    if "input_ids" in payload:
        input_ids = _tensor(payload["input_ids"])
        raw_mask = payload.get("attention_mask")
        attention_mask = _tensor(raw_mask) if isinstance(raw_mask, (dict, list)) else None
    else:
        encoded = _tokenize(payload)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    _check_shape(input_ids, input_ids, "input_ids")
    if attention_mask is not None:
        _check_shape(attention_mask, input_ids, "attention_mask")

    kwargs: dict[str, Any] = {"input_ids": input_ids.contiguous()}

    include_attention = payload.get("include_attention_mask", False)
    if isinstance(payload.get("attention_mask"), bool):
        include_attention = payload["attention_mask"]
    if _boolean(include_attention, "fwd-bwd payload include_attention_mask"):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        kwargs["attention_mask"] = attention_mask.contiguous()

    position_spec = payload.get("position_ids")
    if position_spec is None and payload.get("include_position_ids", False):
        position_spec = "arange"
    if (position_ids := _position_ids(input_ids, position_spec)) is not None:
        _check_shape(position_ids, input_ids, "position_ids")
        kwargs["position_ids"] = position_ids.contiguous()

    if (labels := _labels(input_ids, attention_mask, payload)) is not None:
        _check_shape(labels, input_ids, "labels")
        kwargs["labels"] = labels.contiguous()

    return kwargs


# ── spec plumbing ────────────────────────────────────────────────────────────
def _payload(spec: dict[str, Any], what: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"{what} spec must be an object")
    payload = spec.get("payload", spec)
    if not isinstance(payload, dict):
        raise ValueError(f"{what} payload must be an object")
    return payload


def _args(payload: dict[str, Any]) -> tuple:
    args = payload.get("args", [])
    if args is None:
        return ()
    if not isinstance(args, list):
        raise ValueError("fwd-bwd payload args must be a JSON list")
    return tuple(args)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _check_shape(tensor, like, name: str) -> None:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} must have shape [batch, seq_len]")
    if tensor.shape != like.shape:
        raise ValueError(f"{name} must have the same shape as input_ids")


# ── tensors ──────────────────────────────────────────────────────────────────
def _dtype(name: str | None):
    import torch

    if name is None:
        return torch.long
    aliases = {
        "bool": torch.bool,
        "boolean": torch.bool,
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "int": torch.int32,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.long,
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype: {name}") from exc


def _tensor(value: Any, *, default_dtype: str | None = "long"):
    import torch

    dtype_name = default_dtype
    if isinstance(value, dict):
        if "data" not in value:
            raise ValueError("tensor object must contain a data field")
        dtype_name = value.get("dtype", default_dtype)
        value = value["data"]
    return torch.tensor(value, dtype=_dtype(dtype_name))


def _tensorize(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _tensor(value) if isinstance(value, list) or (isinstance(value, dict) and "data" in value) else value
        for key, value in kwargs.items()
    }


def _position_ids(input_ids, spec: Any):
    import torch

    if spec is None or spec is False:
        return None
    if spec is True or spec == "arange":
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
    if isinstance(spec, (dict, list)):
        return _tensor(spec)
    raise ValueError("fwd-bwd payload position_ids must be true, false, arange, or tensor data")


def _label_config(payload: dict[str, Any]) -> tuple[Any, int, bool]:
    labels = payload.get("labels", payload.get("label_strategy", "next_token"))
    if isinstance(labels, dict) and "data" in labels:
        return labels, int(payload.get("ignore_index", -100)), True
    if isinstance(labels, dict):
        return (
            labels.get("strategy", "next_token"),
            int(labels.get("ignore_index", payload.get("ignore_index", -100))),
            _boolean(labels.get("mask_padding", payload.get("mask_padding", True)), "labels.mask_padding"),
        )
    return (
        labels,
        int(payload.get("ignore_index", -100)),
        _boolean(payload.get("mask_padding", True), "fwd-bwd payload mask_padding"),
    )


def _labels(input_ids, attention_mask, payload: dict[str, Any]):
    import torch

    labels, ignore_index, mask_padding = _label_config(payload)
    if labels is None or labels == "none":
        return None
    if isinstance(labels, (dict, list)):
        return _tensor(labels)
    if labels in ("next_token", "shifted_input_ids"):
        out = torch.roll(input_ids, shifts=-1, dims=1)
        out[:, -1] = ignore_index
        if mask_padding and attention_mask is not None:
            target_mask = torch.roll(attention_mask, shifts=-1, dims=1)
            target_mask[:, -1] = 0
            out = out.masked_fill(target_mask == 0, ignore_index)
        return out
    if labels in ("input_ids", "self"):
        out = input_ids.clone()
        if mask_padding and attention_mask is not None:
            out = out.masked_fill(attention_mask == 0, ignore_index)
        return out
    raise ValueError("fwd-bwd labels strategy must be next_token, input_ids, none, or tensor data")


# ── tokenizer ────────────────────────────────────────────────────────────────
def _repeat_texts(texts: Any, batch_size: Any) -> list[str]:
    if isinstance(texts, str):
        text_list = [texts]
    elif isinstance(texts, list) and texts and all(isinstance(x, str) for x in texts):
        text_list = texts
    else:
        raise ValueError("fwd-bwd payload texts must be a non-empty string list")

    if batch_size is None:
        return text_list
    batch = _positive_int(batch_size, "fwd-bwd payload batch_size")
    return [text_list[i % len(text_list)] for i in range(batch)]


def _load_tokenizer(spec: Any):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tokenizing a fwd-bwd spec needs transformers; install it, or pass pre-tokenized "
            "input_ids / payload.kwargs instead",
            name="transformers",
        ) from exc

    kwargs: dict[str, Any] = {}
    model_name: Any
    if isinstance(spec, str):
        model_name = spec
    elif isinstance(spec, dict):
        model_name = spec.get("model_name") or spec.get("name") or spec.get("path")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("tokenizer.model_name is required")
        kwargs = {key: spec[key] for key in ("trust_remote_code", "use_fast", "revision") if key in spec}
    else:
        raise ValueError("fwd-bwd payload tokenizer must be a string or object")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload.get("tokenizer", payload.get("model_name"))
    if spec is None:
        raise ValueError("fwd-bwd payload requires tokenizer or input_ids")

    tokenizer = _load_tokenizer(spec)
    texts = _repeat_texts(payload.get("texts"), payload.get("batch_size"))
    padding = payload.get("padding", "max_length")
    max_length = payload.get("max_length")
    if padding == "max_length" or max_length is not None:
        max_length = _positive_int(max_length, "fwd-bwd payload max_length")

    encode_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": padding,
        "truncation": _boolean(payload.get("truncation", True), "fwd-bwd payload truncation"),
        "add_special_tokens": _boolean(payload.get("add_special_tokens", True), "fwd-bwd payload add_special_tokens"),
    }
    if max_length is not None:
        encode_kwargs["max_length"] = max_length

    encoded = tokenizer(texts, **encode_kwargs)
    if "input_ids" not in encoded:
        raise ValueError("tokenizer output did not include input_ids")
    return encoded
