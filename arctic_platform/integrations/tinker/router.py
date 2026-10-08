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

"""Convert Tinker datums, losses, and sampling params into Cortex batch fields.

The in-process client in :mod:`arctic_platform.tinker` calls these helpers.
This module does not open a server and does not import Cortex.
"""

from __future__ import annotations

import math
from typing import Any
from typing import Literal
from typing import Sequence
from typing import Union

import numpy as np
from pydantic import BaseModel
from pydantic import Field


class TensorData(BaseModel):
    dtype: Literal["float32", "int64"]
    data: list[float] | list[int] = Field(default_factory=list)
    shape: list[int] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None


class EncodedTextChunk(BaseModel):
    type: Literal["encoded_text"] = "encoded_text"
    tokens: list[int]


class ModelInput(BaseModel):
    # Text tokens only. Image and audio chunks are refused.
    chunks: list[EncodedTextChunk]


class Datum(BaseModel):
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData] = Field(default_factory=dict)


class AdamParams(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class SamplingParams(BaseModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: Union[str, Sequence[str], Sequence[int], None] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


# =============================================================================
# Adapters — Tinker wire types → Arctic native shapes
# =============================================================================

# Tinker loss name -> the intermediate Arctic loss name. The Cortex binder
# maps ratio losses to ``grpo`` and custom cross-entropy to its gradient
# surrogate.
_BACKEND_LOSS_FNS = {
    "ppo": "verl_grpo",
    "importance_sampling": "verl_grpo",
    "cross_entropy": "weighted_logprob_sum",
}

# The intermediate name is the same for both ratio losses, so the bounds Tinker
# puts on the ratio p/q travel separately as ``processing["ratio_clip"]``.
_PPO_CLIP_DEFAULTS = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}


def _ratio_clip(loss_fn: str, loss_fn_config: dict[str, float] | None) -> tuple[float, float] | None:
    """Tinker's ``(low, high)`` bounds on p/q, or ``None`` for a loss without a ratio.

    ``importance_sampling`` is unclipped. ``ppo`` takes its thresholds from
    ``loss_fn_config``. A key this adapter does not implement is refused, since
    ignoring it would train with a different loss than the caller asked for.
    """
    config = dict(loss_fn_config or {})
    if loss_fn == "ppo":
        unknown = sorted(set(config) - set(_PPO_CLIP_DEFAULTS))
        if unknown:
            raise ValueError(f"loss_fn='ppo' supports loss_fn_config keys {sorted(_PPO_CLIP_DEFAULTS)}; got {unknown}")
        bounds = {**_PPO_CLIP_DEFAULTS, **config}
        low, high = float(bounds["clip_low_threshold"]), float(bounds["clip_high_threshold"])
        if not 0.0 <= low <= 1.0 <= high:
            raise ValueError(f"ppo needs 0 <= clip_low_threshold <= 1 <= clip_high_threshold; got {low}, {high}")
        return low, high
    if config:
        raise ValueError(f"loss_fn={loss_fn!r} takes no loss_fn_config; got {sorted(config)}")
    if loss_fn == "importance_sampling":
        return 0.0, math.inf
    return None


def _model_input_to_tokens(model_input: ModelInput) -> list[int]:
    """Flatten a ``ModelInput`` into a token list."""
    out: list[int] = []
    for chunk in model_input.chunks:
        if not isinstance(chunk, EncodedTextChunk):  # pragma: no cover — Pydantic guard
            raise ValueError(f"only text chunks are supported, got {type(chunk).__name__}")
        out.extend(chunk.tokens)
    return out


def _tensor_data_to_numpy(td: TensorData) -> np.ndarray:
    """Materialise a ``TensorData`` back to a numpy array.

    Sparse-CSR weights and target tokens are reconstructed as dense arrays.
    """
    dtype = np.float32 if td.dtype == "float32" else np.int64
    if td.sparse_crow_indices is not None:
        assert td.shape is not None, "sparse TensorData requires shape"
        assert td.sparse_col_indices is not None
        rows, cols = td.shape
        dense = np.zeros((rows, cols), dtype=dtype)
        crow = td.sparse_crow_indices
        col = td.sparse_col_indices
        values = np.asarray(td.data, dtype=dtype)
        for r in range(rows):
            for j in range(crow[r], crow[r + 1]):
                dense[r, col[j]] = values[j]
        return dense
    arr = np.asarray(td.data, dtype=dtype)
    if td.shape is not None and list(arr.shape) != list(td.shape):
        arr = arr.reshape(td.shape)
    return arr


def _split_prompt_response(tokens: list[int], candidates: list[np.ndarray | None]) -> int:
    """Return the prompt / response boundary index for one ``Datum``.

    Convention (matches ``tinker-cookbook`` SFT + RL examples): prompt
    tokens are zero-masked and response tokens carry the training signal
    (``weights=1`` for cross-entropy, non-zero ``advantages`` for RL). We
    scan the provided masks in priority order and return the first
    non-zero index. Falls back to ``len(tokens)`` (whole prompt, no
    response) when nothing is masked.
    """
    for arr in candidates:
        if arr is None or len(arr) == 0:
            continue
        nz = np.flatnonzero(arr[: len(tokens)])
        if nz.size:
            return int(nz[0])
    return len(tokens)


def _per_token(inputs: dict[str, TensorData], key: str, index: int, n_tokens: int) -> np.ndarray:
    """``loss_fn_inputs[key]`` as float32, one entry per ``model_input`` token."""
    arr = _tensor_data_to_numpy(inputs[key]).astype(np.float32).reshape(-1)
    if arr.shape[0] != n_tokens:
        raise ValueError(
            f"datum {index}: loss_fn_inputs[{key!r}] has {arr.shape[0]} entries for {n_tokens} model_input tokens"
        )
    return arr


def datum_list_to_arctic_batch(
    data: list[Datum],
    loss_fn: str,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int,
    forward_only: bool = False,
    loss_fn_config: dict[str, float] | None = None,
) -> tuple[dict, list[tuple[int, int, int]]]:
    """Pack a list of Tinker ``Datum`` into an Arctic ``fwd_bwd`` batch dict.

    Rows are padded to ``max_prompt_length + max_response_length`` rather than
    to the batch's own longest row -- ZoRRo requires the config-max width. The
    prompt/response boundary, inferred per-datum by
    :func:`_split_prompt_response`, sits at column ``max_prompt_length`` when
    the row allows it; a longer prompt or response shifts the whole row
    instead of being cut, because a cut prompt changes what the trainer
    conditions on and a cut response drops scored tokens. A row longer than
    the full width is refused.

    Per-token inputs stay on the column of the ``model_input`` token they are
    indexed by. Tinker indexes them by target, ``target_tokens[j] ==
    model_input[j + 1]``, which is the frame Cortex scores in, hence the
    ``_shifted`` names.

    Returns ``(batch_dict, row_slices)``. ``row_slices[i]`` is ``(start, end,
    tinker_len)``: Tinker's contract is that returned log-probs line up with
    the datum's own tokens, so the slices reverse this padded layout on the way
    back out.
    """
    ratio_clip = None if forward_only else _ratio_clip(loss_fn, loss_fn_config)
    mpl = int(max_prompt_length)
    mrl = int(max_response_length)
    total_len = mpl + mrl
    batch_size = len(data)

    input_ids = np.full((batch_size, total_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, total_len), dtype=np.int64)
    # Full sequence width, not response width, so these flatten alongside
    # ``attention_mask`` and stay 1:1 with the returned log-probs. The prompt
    # columns are inert: ``response_mask`` is 0 there.
    response_mask = np.zeros((batch_size, total_len), dtype=np.int64)
    advantages = np.zeros((batch_size, total_len), dtype=np.float32)
    old_log_probs = np.zeros((batch_size, total_len), dtype=np.float32)
    logprob_weights = np.zeros((batch_size, total_len), dtype=np.float32)
    row_slices: list[tuple[int, int, int]] = []

    for i, datum in enumerate(data):
        toks = _model_input_to_tokens(datum.model_input)
        inputs = datum.loss_fn_inputs
        if ratio_clip is not None and "logprobs" not in inputs:
            # Zeros in their place would make the ratio exp(logp) instead of p/q.
            raise ValueError(f"loss_fn={loss_fn!r} needs the sampler's 'logprobs' in datum {i}")

        # SFT datums carry ``weights``, RL datums ``advantages`` +
        # ``target_tokens``. Prompt tokens are zero-masked in all of them, so
        # an explicit ``weights`` or ``mask`` locates the boundary, and failing
        # that the first marker that has one does. An RL datum from a group
        # with equal rewards has all-zero advantages, and the cookbook strips
        # ``mask`` before sending, so the sampler's ``logprobs`` -- zero on
        # observation tokens -- come next.
        explicit = [key for key in ("weights", "mask") if key in inputs]
        markers = explicit[:1] or [key for key in ("advantages", "logprobs", "target_tokens") if key in inputs]
        candidates = [_tensor_data_to_numpy(inputs[key]).astype(np.float32) for key in markers]

        # Append the final target as a scoring token. It stays out of
        # ``response_mask`` and ``advantages``.
        target_tokens = inputs.get("target_tokens")
        scoring_tok = None
        if target_tokens is not None:
            target_arr = _tensor_data_to_numpy(target_tokens)
            if len(target_arr):
                scoring_tok = int(np.asarray(target_arr).reshape(-1)[-1])

        n = len(toks)
        width = n + int(scoring_tok is not None)
        if width > total_len:
            raise ValueError(
                f"datum {i} needs {width} positions (model_input plus the final target) but "
                f"max_prompt_length + max_response_length = {total_len}"
            )
        p_end = _split_prompt_response(toks, candidates) if not forward_only else n
        start = min(max(mpl - p_end, 0), total_len - width)
        resp = slice(start + p_end, start + n)

        input_ids[i, start : start + n] = np.asarray(toks, dtype=np.int64)
        attention_mask[i, start : start + width] = 1
        if scoring_tok is not None:
            input_ids[i, start + n] = scoring_tok
        response_mask[i, resp] = 1
        row_slices.append((start, start + n, n))

        if "advantages" in inputs:
            advantages[i, resp] = _per_token(inputs, "advantages", i, n)[p_end:]
        if "logprobs" in inputs:
            old_log_probs[i, resp] = _per_token(inputs, "logprobs", i, n)[p_end:]
        if "weights" in inputs:
            # Tinker's cross-entropy is ``L = sum(-logprobs * weights)`` while
            # ``weighted_logprob_sum`` computes ``sum(logprobs * w)``, so the
            # sign flips here. Positions before ``p_end`` are zero by
            # construction -- that is how _split_prompt_response found p_end.
            logprob_weights[i, resp] = -_per_token(inputs, "weights", i, n)[p_end:]

    processing: dict[str, Any] = {
        "post": ["compute_entropy_and_logprobs"],
        "loss_fn": _BACKEND_LOSS_FNS[loss_fn] if not forward_only else None,
    }
    if ratio_clip is not None:
        processing["ratio_clip"] = ratio_clip
    batch_dict = {
        "batch": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Cortex rebuilds position ids from the attention mask.
            "response_mask": response_mask,
            "advantages": advantages,
            "old_log_probs_shifted": old_log_probs,
            "logprob_weights_shifted": logprob_weights,
        },
        "meta": {
            "batch_num_tokens": int(response_mask.sum()),
            "global_batch_size": batch_size,
        },
        "processing": processing,
    }
    return batch_dict, row_slices


def _unpad_logprobs_to_loss_fn_outputs(
    logprobs_batch: Any, row_slices: list[tuple[int, int, int]]
) -> list[dict[str, Any]]:
    """Un-pad Arctic's logprob tensor into per-Datum ``LossFnOutput`` dicts
    of exactly ``model_input_len`` each.

    Tinker's contract: ``loss_fn_outputs[i]["logprobs"]`` is 1-D with length
    ``len(data[i].model_input.tokens)`` -- the upstream cookbook indexes it
    with a per-Datum mask of that length. Arctic returns logprobs in the
    padded compute layout (``[B, mpl+mrl]`` for the 2-D case, packed 1-D
    otherwise); slice via ``row_slices`` and pad/truncate the tail so the
    shape invariant always holds. Masked-out positions carry no signal.
    """
    arr = np.asarray(logprobs_batch, dtype=np.float32)
    outputs: list[dict[str, Any]] = []
    flat_offset = 0
    for i, (start, end, expected_len) in enumerate(row_slices):
        if arr.ndim == 2 and i < arr.shape[0]:
            row = arr[i, start:end]
        else:
            take = end - start
            row = arr.reshape(-1)[flat_offset : flat_offset + take]
            flat_offset += take
        if row.shape[0] < expected_len:
            row = np.pad(row, (0, expected_len - row.shape[0]))
        elif row.shape[0] > expected_len:
            row = row[:expected_len]
        outputs.append(
            {"logprobs": TensorData(dtype="float32", data=row.tolist(), shape=[int(expected_len)]).model_dump()}
        )
    return outputs


def _tinker_metric_name(name: str, default_reduction: str = "mean") -> str:
    """Annotate an Arctic metric with a Tinker-style ``:reduction`` suffix.

    Tinker's ``combine_fwd_bwd_output_results`` requires every metric to
    encode its cross-actor reduction as ``name:reduction`` (e.g.
    ``loss:mean``). Arctic handlers do not follow that convention, so we
    coerce plain names to ``:mean`` (a safe default that weights by
    per-actor sample count). Names that already include a valid suffix
    pass through untouched.
    """
    if ":" in name:
        return name
    return f"{name}:{default_reduction}"


def arctic_metrics_to_tinker(metrics: dict[str, Any] | None) -> dict[str, float]:
    """Filter Arctic metrics down to numeric values and annotate them for Tinker."""
    if not metrics:
        return {}
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        out[_tinker_metric_name(str(k))] = float(v)
    return out


def adam_params_to_optim_overrides(p: AdamParams) -> dict[str, Any]:
    """Translate ``AdamParams`` -> Arctic ``DeepSpeedWorker.step(optim_overrides=...)``."""
    return {
        "lr": float(p.learning_rate),
        "betas": (float(p.beta1), float(p.beta2)),
        "eps": float(p.eps),
        "weight_decay": float(p.weight_decay),
    }


_ADAM_FIELDS = ("beta1", "beta2", "eps", "weight_decay", "grad_clip_norm")


def check_fixed_adam(served: dict[str, float], requested: AdamParams) -> None:
    """Refuse Adam settings the job cannot change after it is created.

    Cortex applies betas, eps, weight decay, and grad clip at provisioning.
    Only the learning rate is sent on each step.
    """
    mismatched = {
        field: (getattr(requested, field), served[field])
        for field in _ADAM_FIELDS
        if not math.isclose(float(getattr(requested, field)), float(served[field]), rel_tol=1e-9, abs_tol=0.0)
    }
    if mismatched:
        detail = ", ".join(f"{field}={got} (provisioned {want})" for field, (got, want) in mismatched.items())
        raise ValueError(
            f"Adam hyperparameters other than the learning rate are fixed when the job is created; got {detail}"
        )


def sampling_params_tinker_to_vllm(p: SamplingParams, num_samples: int) -> dict[str, Any]:
    """Translate Tinker ``SamplingParams`` -> vLLM ``SamplingParams(...)`` kwargs.

    ``logprobs=1`` is forced so downstream RL loops receive per-token
    ``old_log_probs`` for their PPO / IS ratio computation. vLLM stops on
    integer ``stop_token_ids`` or string ``stop``; Tinker packs both into
    a single ``stop`` union which we splat.
    """
    out: dict[str, Any] = {
        "n": int(num_samples),
        "temperature": float(p.temperature),
        "top_p": float(p.top_p),
        "top_k": int(p.top_k),
        "logprobs": 1,
    }
    if p.max_tokens is not None:
        out["max_tokens"] = int(p.max_tokens)
    if p.seed is not None:
        out["seed"] = int(p.seed)
    if p.stop is not None:
        stop = p.stop
        if isinstance(stop, (list, tuple)) and stop and isinstance(stop[0], int):
            out["stop_token_ids"] = list(stop)
        else:
            out["stop"] = stop if isinstance(stop, str) else list(stop)
    return out
