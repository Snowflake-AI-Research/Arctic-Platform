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

"""The upstream `tinker <https://github.com/thinking-machines-lab/tinker>`_
HTTP protocol, as a backend-agnostic FastAPI router.

:func:`init_tinker_state` injects the handlers (forward-backward, optimizer step,
weight sync, generate, and optionally forward) and this module knows nothing
else about the backend. :mod:`arctic_platform.integrations.tinker.cortex` builds them from
the unified client; an on-prem handler set would drop in the same way.

Scope (v1): one global training run, no auth. The run is full fine-tuning or a
LoRA adapter, fixed when the backend is provisioned; ``create_model`` refuses a
``LoraConfig`` that differs from it, since the trained adapter could not change.

Every long-running Tinker verb is future-based on the wire. Each request runs as
a task; one that finishes within :data:`_INLINE_BUDGET_S` is answered after it
completes, and a slower one is answered with its future, so the SDK never
resends work that is still running.

Wire schemas are Pydantic models mirroring ``tinker.types.*`` so that serving
the JSON verbs needs no ``tinker`` install. The proto verbs do need it, and
take their schema from the SDK directly -- see
:mod:`arctic_platform.integrations.tinker.proto_wire`.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import time
import uuid
from enum import Enum
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import Union

import numpy as np
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from arctic_platform.integrations.tinker.proto_wire import PROTO_CONTENT_TYPE
from arctic_platform.integrations.tinker.proto_wire import decode_forward_backward_request
from arctic_platform.integrations.tinker.proto_wire import encode_forward_backward_output
from arctic_platform.integrations.tinker.proto_wire import encode_sample_response
from arctic_platform.integrations.tinker.proto_wire import wants_proto

logger = logging.getLogger(__name__)

# =============================================================================
# Wire schemas — Pydantic mirrors of ``tinker.types.*``
# =============================================================================
#
# ``test_wire_schema.py`` feeds upstream ``model_dump()`` payloads through these
# classes to catch drift.


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
    # v1 supports only ``EncodedTextChunk``. Images / DMEL / asset-pointer
    # chunks return HTTP 400.
    chunks: list[EncodedTextChunk]


class Datum(BaseModel):
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData] = Field(default_factory=dict)


LossFnType = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
    "dro",
]


class ForwardBackwardInput(BaseModel):
    data: list[Datum]
    loss_fn: LossFnType
    loss_fn_config: dict[str, float] | None = None


class ForwardBackwardRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None


class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str = "TorchLossReturn"
    loss_fn_outputs: list[dict[str, TensorData]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class AdamParams(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class OptimStepRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    adam_params: AdamParams
    model_id: str
    seq_id: int | None = None


class OptimStepResponse(BaseModel):
    metrics: dict[str, float] | None = None


class LoraConfig(BaseModel):
    rank: int
    seed: int | None = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True


class CreateSessionRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] | None = None
    sdk_version: str | None = None
    project_id: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    type: Literal["create_session"] = "create_session"


class SessionHeartbeatRequest(BaseModel):
    session_id: str


class ClientConfigRequest(BaseModel):
    sdk_version: str


class ClientConfigResponse(BaseModel):
    # ``proto_compress_fwdbwd`` stays off: there is no zstd path here. Current
    # SDKs no longer consult ``proto_write_fwdbwd`` and post proto regardless,
    # which ``proto_wire`` handles.
    pjwt_auth_enabled: bool = False
    credential_default_source: str = "api_key"
    sample_dispatch_bytes_semaphore_size: int = 10 * 1024 * 1024
    inflight_response_bytes_semaphore_size: int = 50 * 1024 * 1024
    parallel_fwdbwd_chunks: bool = True
    proto_write_fwdbwd: bool = False
    proto_compress_fwdbwd: bool = False
    fwd_via_fwdbwd: bool = False
    billing_exception_max_pause_duration_sec: int = 60 * 60
    sample_no_retries: bool = False
    sample_enable_stuck_detection: bool = True
    sample_max_concurrent_requests: int = 2000
    use_pyqwest_transport: bool = False
    # The SDK splits a forward_backward into requests of at most this many
    # datums and estimated bytes (its defaults).
    fwdbwd_max_chunk_len: int = 1024
    fwdbwd_max_chunk_bytes_count: int = 5_000_000


class AuthTokenResponse(BaseModel):
    jwt: str


class TelemetryResponse(BaseModel):
    status: Literal["accepted"] = "accepted"


class SupportedModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


class CreateModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    model_seq_id: int
    base_model: str
    user_metadata: dict[str, Any] | None = None
    lora_config: LoraConfig | None = None


class CreateModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    base_model: str
    lora_config: LoraConfig | None = None
    status: str = "created"
    request_id: str | None = None
    type: Literal["create_model"] = "create_model"


class GetInfoRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    type: str | None = None


class ModelData(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    base_model: str
    lora_config: LoraConfig | None = None
    model_name: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    status: str
    model_data: ModelData


class CreateSamplingSessionRequest(BaseModel):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None


class CreateSamplingSessionResponse(BaseModel):
    sampling_session_id: str
    type: Literal["create_sampling_session"] = "create_sampling_session"


class SaveWeightsForSamplerRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None


class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    sampling_session_id: str | None = None
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"


class SaveWeightsRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    path: str | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None
    overwrite: bool = False


class SaveWeightsResponse(BaseModel):
    path: str
    type: Literal["save_weights"] = "save_weights"


class SamplingParams(BaseModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: Union[str, Sequence[str], Sequence[int], None] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


class SampleRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prompt: ModelInput
    sampling_params: SamplingParams
    num_samples: int = 1
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0


class StopReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"


class SampledSequence(BaseModel):
    tokens: list[int]
    logprobs: list[float] | None = None
    stop_reason: StopReason


class SampleResponse(BaseModel):
    sequences: list[SampledSequence]
    prompt_logprobs: list[float | None] | None = None
    type: Literal["sample"] = "sample"


class UntypedAPIFuture(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_id: str | None = None
    type: Literal["future"] = "future"


class RequestFailedResponse(BaseModel):
    error: str
    category: Literal["unknown", "server", "user"]


class TryAgainResponse(BaseModel):
    type: Literal["try_again"] = "try_again"


class FutureRetrieveRequest(BaseModel):
    request_id: str
    allow_metadata_only: bool = False


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
            raise HTTPException(
                400, f"loss_fn='ppo' supports loss_fn_config keys {sorted(_PPO_CLIP_DEFAULTS)}; got {unknown}"
            )
        bounds = {**_PPO_CLIP_DEFAULTS, **config}
        low, high = float(bounds["clip_low_threshold"]), float(bounds["clip_high_threshold"])
        if not 0.0 <= low <= 1.0 <= high:
            raise HTTPException(
                400, f"ppo needs 0 <= clip_low_threshold <= 1 <= clip_high_threshold; got {low}, {high}"
            )
        return low, high
    if config:
        raise HTTPException(400, f"loss_fn={loss_fn!r} takes no loss_fn_config; got {sorted(config)}")
    if loss_fn == "importance_sampling":
        return 0.0, math.inf
    return None


def _model_input_to_tokens(model_input: ModelInput) -> list[int]:
    """Flatten a ``ModelInput`` (v1: text-only) into a token list. Raises 400
    for any non-``EncodedTextChunk`` chunk."""
    out: list[int] = []
    for chunk in model_input.chunks:
        if not isinstance(chunk, EncodedTextChunk):  # pragma: no cover — Pydantic guard
            raise HTTPException(400, f"v1 supports text chunks only, got {type(chunk).__name__}")
        out.extend(chunk.tokens)
    return out


def _tensor_data_to_numpy(td: TensorData) -> np.ndarray:
    """Materialise a wire ``TensorData`` back to a numpy array.

    Supports the dense path only in v1; sparse-CSR encoded weights/target
    tokens fall back to dense reconstruction.
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
        raise HTTPException(
            400, f"datum {index}: loss_fn_inputs[{key!r}] has {arr.shape[0]} entries for {n_tokens} model_input tokens"
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
            raise HTTPException(400, f"loss_fn={loss_fn!r} needs the sampler's 'logprobs' in datum {i}")

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
            raise HTTPException(
                400,
                f"datum {i} needs {width} positions (model_input plus the final target) but the "
                f"server was started with max_prompt_length + max_response_length = {total_len}; "
                "restart it with larger limits",
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


# =============================================================================
# In-memory future store
# =============================================================================


class TinkerFutureStore:
    """One ``asyncio.Task`` per future; ``retrieve_future`` answers
    ``try_again`` until it finishes.

    Work on the trained model (forward-backward, optimizer step, weight saves)
    runs one request at a time in arrival order, as Tinker orders a model's
    requests; sampling runs concurrently. ``asyncio.Lock`` wakes waiters in
    the order they queued, and tasks start in the order they were created.
    """

    def __init__(self) -> None:
        self._counter = itertools.count()
        self._tasks: dict[str, tuple[asyncio.Task, str | None]] = {}
        self._serial = asyncio.Lock()

    def new_request_id(self) -> str:
        return str(next(self._counter))

    def submit(
        self, runner: Callable[[], Awaitable[dict[str, Any]]], kind: str | None = None, *, serial: bool = False
    ) -> tuple[str, asyncio.Task]:
        """Start ``runner``. ``kind`` names the proto encoder that
        ``retrieve_future`` must use, since current SDKs reject a JSON reply for
        ``ForwardBackwardOutput`` and ``SampleResponse``."""

        async def run() -> dict[str, Any]:
            if not serial:
                return await runner()
            async with self._serial:
                return await runner()

        request_id = self.new_request_id()
        task = asyncio.create_task(run())
        self._tasks[request_id] = (task, kind)
        return request_id, task

    async def pop(self, request_id: str, wait: float) -> tuple[asyncio.Task, str | None] | None:
        """The finished task, waiting up to ``wait`` seconds; ``None`` while it runs or for an unknown id."""
        entry = self._tasks.get(request_id)
        if entry is None:
            return None
        task, kind = entry
        await asyncio.wait({task}, timeout=wait)
        if not task.done():
            return None
        del self._tasks[request_id]
        return task, kind


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/v1")

_KIND_FWD_BWD = "forward_backward_output"
_KIND_SAMPLE = "sample_response"
_PROTO_ENCODERS: dict[str, Callable[[dict[str, Any]], bytes]] = {
    _KIND_FWD_BWD: encode_forward_backward_output,
    _KIND_SAMPLE: encode_sample_response,
}

_V1_SUPPORTED_LOSSES = frozenset({"ppo", "importance_sampling", "cross_entropy"})
_V1_UNSUPPORTED_LOSSES = frozenset({"cispo", "dro"})


def _require_state(app_state: Any, name: str) -> Any:
    if not hasattr(app_state, name):
        raise HTTPException(
            500,
            f"Tinker layer misconfigured: app.state.{name} is unset. "
            "Call init_tinker_state() before starting "
            "arctic_platform.integrations.tinker.serve.",
        )
    return getattr(app_state, name)


# The SDK times a request out after 60 s and sends it again, which would run a
# forward-backward or an optimizer step twice. Work that outlasts this budget is
# answered with its future and finished in the background.
_INLINE_BUDGET_S = 30.0
_RETRIEVE_WAIT_S = 30.0


async def _submit(
    request: Request,
    runner: Callable[[], Awaitable[dict[str, Any]]],
    *,
    model_id: str | None = None,
    kind: str | None = None,
    serial: bool = False,
) -> UntypedAPIFuture:
    """Run ``runner`` as a future, inline while it fits the budget.

    A runner that fails within the budget fails the request itself, as a 400
    for an ``HTTPException``; a later failure reaches ``retrieve_future``.
    """
    store: TinkerFutureStore = _require_state(request.app.state, "tinker_futures")
    request_id, task = store.submit(runner, kind, serial=serial)
    await asyncio.wait({task}, timeout=_INLINE_BUDGET_S)
    if task.done() and task.exception() is not None:
        await store.pop(request_id, 0)
        raise task.exception()
    return UntypedAPIFuture(request_id=request_id, model_id=model_id)


# ---- session / bootstrap verbs ----------------------------------------------


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    """Liveness plus bind check: ``bound`` is True once
    :func:`init_tinker_state` has supplied the handlers."""
    return {"status": "ok", "bound": getattr(request.app.state, "tinker_base_model", None) is not None}


@router.post("/create_session", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest, request: Request) -> CreateSessionResponse:
    sessions = _require_state(request.app.state, "tinker_sessions")
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    sessions[session_id] = {"created_at": time.time(), "tags": list(req.tags)}
    return CreateSessionResponse(session_id=session_id)


@router.post("/session_heartbeat")
async def session_heartbeat(req: SessionHeartbeatRequest) -> dict[str, Any]:
    return {}


@router.post("/client/config", response_model=ClientConfigResponse)
async def client_config(req: ClientConfigRequest, request: Request) -> ClientConfigResponse:
    if getattr(request.app.state, "tinker_accumulates_gradients", True):
        return ClientConfigResponse()
    # A backend that steps on the last forward_backward alone must receive a
    # cookbook batch (512+ datums) as one request, not the SDK's 5 MB chunks.
    unbounded = 2**62
    return ClientConfigResponse(fwdbwd_max_chunk_len=unbounded, fwdbwd_max_chunk_bytes_count=unbounded)


@router.post("/auth/token", response_model=AuthTokenResponse)
async def auth_token() -> AuthTokenResponse:
    return AuthTokenResponse(jwt="tml-dummy")


@router.post("/telemetry", response_model=TelemetryResponse)
async def telemetry(req: dict) -> TelemetryResponse:
    return TelemetryResponse()


@router.get("/get_server_capabilities", response_model=GetServerCapabilitiesResponse)
async def get_server_capabilities(request: Request) -> GetServerCapabilitiesResponse:
    base_model = _require_state(request.app.state, "tinker_base_model")
    teachers = getattr(request.app.state, "tinker_teacher_generate", None) or {}
    return GetServerCapabilitiesResponse(
        supported_models=[SupportedModel(model_name=name) for name in [base_model, *teachers]]
    )


# ---- model lifecycle --------------------------------------------------------


_LORA_TARGETS = ("rank", "train_mlp", "train_attn", "train_unembed")


def _check_lora(served: LoraConfig | None, requested: LoraConfig | None) -> None:
    """Refuse a model whose adapter differs from the one the backend trains.

    ``None`` or rank 0 is full fine-tuning on both sides. ``seed`` is not
    compared: it only picks the adapter's random initialization.
    """
    served_rank = served.rank if served is not None else 0
    requested_rank = requested.rank if requested is not None else 0
    if served_rank == 0 and requested_rank == 0:
        return
    if served_rank != 0 and requested_rank != 0:
        mismatched = [f for f in _LORA_TARGETS if getattr(served, f) != getattr(requested, f)]
        if not mismatched:
            return
    describe = "full fine-tuning" if served_rank == 0 else f"LoRA {served.model_dump(include=set(_LORA_TARGETS))}"
    raise HTTPException(
        400,
        f"this server trains {describe}, got lora_config={requested.model_dump() if requested else None}. "
        "The adapter is fixed when the job is provisioned; restart the server with matching "
        "--lora-rank / --lora-modules, or pass the served configuration.",
    )


_ADAM_FIELDS = ("beta1", "beta2", "eps", "weight_decay", "grad_clip_norm")


def _check_adam(served: Mapping[str, float] | None, requested: AdamParams) -> None:
    """Refuse Adam hyperparameters the backend cannot apply at step time.

    ``served`` is ``None`` when the backend applies every field per step;
    otherwise only the learning rate may vary.
    """
    if served is None:
        return
    mismatched = {
        f: (getattr(requested, f), served[f])
        for f in _ADAM_FIELDS
        if not math.isclose(getattr(requested, f), served[f], rel_tol=1e-9, abs_tol=0.0)
    }
    if mismatched:
        detail = ", ".join(f"{f}={got} (served {want})" for f, (got, want) in mismatched.items())
        raise HTTPException(
            400,
            "this backend fixes Adam's hyperparameters when the job is provisioned and "
            f"changes only the learning rate per step; got {detail}. Restart the server "
            "with matching --adam-beta1 / --adam-beta2 / --adam-eps / --weight-decay / --grad-clip-norm.",
        )


@router.post("/create_model", response_model=UntypedAPIFuture)
async def create_model(req: CreateModelRequest, request: Request) -> UntypedAPIFuture:
    base_model = _require_state(request.app.state, "tinker_base_model")
    if req.base_model != base_model:
        raise HTTPException(
            400,
            f"server was started with base_model={base_model!r}, got base_model={req.base_model!r}",
        )
    _check_lora(getattr(request.app.state, "tinker_lora", None), req.lora_config)
    models = _require_state(request.app.state, "tinker_models")
    model_id = "main"  # single-tenant in v1
    models[model_id] = {"base_model": req.base_model, "lora_config": req.lora_config}

    async def runner() -> dict[str, Any]:
        return CreateModelResponse(
            model_id=model_id,
            base_model=req.base_model,
            lora_config=req.lora_config,
        ).model_dump(mode="json")

    return await _submit(request, runner, model_id=model_id)


@router.post("/get_info", response_model=ModelInfoResponse)
async def get_info(req: GetInfoRequest, request: Request) -> ModelInfoResponse:
    models = _require_state(request.app.state, "tinker_models")
    m = models.get(req.model_id)
    if m is None:
        raise HTTPException(404, f"model_id={req.model_id!r} not found")
    return ModelInfoResponse(
        model_id=req.model_id,
        status="created",
        model_data=ModelData(
            base_model=m["base_model"],
            lora_config=m.get("lora_config"),
            model_name=m["base_model"],
        ),
    )


# ---- training verbs ---------------------------------------------------------


def _gate_loss_fn(loss_fn: str) -> None:
    if loss_fn in _V1_UNSUPPORTED_LOSSES:
        raise HTTPException(
            400,
            f"loss_fn={loss_fn!r} not supported in v1; supported: {sorted(_V1_SUPPORTED_LOSSES)}",
        )
    if loss_fn not in _V1_SUPPORTED_LOSSES:
        raise HTTPException(400, f"unknown loss_fn={loss_fn!r}")


@router.post("/forward_backward", response_model=UntypedAPIFuture)
async def forward_backward(request: Request) -> UntypedAPIFuture:
    """Accepts either encoding, and carries ``forward`` as well.

    The body is read by hand rather than declared as a pydantic parameter
    because current SDKs post protobuf here, and because upstream folded
    ``forward`` into this endpoint behind a ``forward_only`` flag -- a JSON-only
    signature would reject both.
    """
    body = await request.body()
    if wants_proto(None, request.headers.get("content-type")):
        req, forward_only = decode_forward_backward_request(body)
    else:
        req, forward_only = ForwardBackwardRequest.model_validate_json(body), False
    if forward_only:
        return await _run_forward(req, request)
    return await _run_forward_backward(req, request)


async def _run_forward_backward(req: ForwardBackwardRequest, request: Request) -> UntypedAPIFuture:
    fbi = req.forward_backward_input
    _gate_loss_fn(fbi.loss_fn)
    handler = _require_state(request.app.state, "tinker_fwd_bwd")
    max_prompt = _require_state(request.app.state, "tinker_max_prompt_length")
    max_resp = _require_state(request.app.state, "tinker_max_response_length")
    pad_id = _require_state(request.app.state, "tinker_pad_token_id")
    batch, row_slices = datum_list_to_arctic_batch(
        fbi.data,
        fbi.loss_fn,
        max_prompt,
        max_resp,
        pad_id,
        forward_only=False,
        loss_fn_config=fbi.loss_fn_config,
    )

    n_data = len(fbi.data)
    _claim_gradient(request.app.state, req.model_id)

    async def runner() -> dict[str, Any]:
        try:
            r = await handler(batch)
        except BaseException:
            request.app.state.tinker_pending_gradients.discard(req.model_id)
            raise
        # Empty dicts rather than a short list when the backend returns no
        # log-probs: the cookbook weights its metric reduction by datum count.
        logprobs_batch = r.get("batch", {}).get("logprobs") if r.get("batch") else None
        if logprobs_batch is not None:
            outputs = _unpad_logprobs_to_loss_fn_outputs(logprobs_batch, row_slices)
        else:
            outputs = [{} for _ in range(n_data)]
        return ForwardBackwardOutput(
            loss_fn_outputs=outputs,
            metrics=arctic_metrics_to_tinker(r.get("metrics")),
        ).model_dump(mode="json")

    return await _submit(request, runner, model_id=req.model_id, kind=_KIND_FWD_BWD, serial=True)


def _claim_gradient(app_state: Any, model_id: str | None) -> None:
    """Refuse a second ``forward_backward`` before ``optim_step`` on a backend that keeps only the last one's gradient.

    Tinker sums the gradients of every ``forward_backward`` since the last
    ``optim_step``. Cortex steps on the last call's gradient alone, so the
    earlier calls' data would go untrained without any error.
    """
    if getattr(app_state, "tinker_accumulates_gradients", True):
        return
    pending = app_state.tinker_pending_gradients
    if model_id in pending:
        raise HTTPException(
            400,
            "this backend steps on the gradient of the last forward_backward only and cannot accumulate "
            "several: a second forward_backward before optim_step would leave the first one's data untrained. "
            "Send each step's batch in one forward_backward call (in tinker-cookbook, leave "
            "stream_minibatch_config unset or set num_minibatches=1).",
        )
    pending.add(model_id)


async def _run_forward(req: ForwardBackwardRequest, request: Request) -> UntypedAPIFuture:
    fbi = req.forward_backward_input
    _gate_loss_fn(fbi.loss_fn)
    handler = getattr(request.app.state, "tinker_fwd_no_grad", None)
    if handler is None:
        raise HTTPException(
            400,
            "forward (log-probs without a gradient) is not supported by this backend; "
            "forward_backward returns the same log-probs alongside its gradient.",
        )
    max_prompt = _require_state(request.app.state, "tinker_max_prompt_length")
    max_resp = _require_state(request.app.state, "tinker_max_response_length")
    pad_id = _require_state(request.app.state, "tinker_pad_token_id")
    batch, row_slices = datum_list_to_arctic_batch(
        fbi.data,
        fbi.loss_fn,
        max_prompt,
        max_resp,
        pad_id,
        forward_only=True,
    )

    async def runner() -> dict[str, Any]:
        r = await handler(batch)
        logprobs_batch = r.get("batch", {}).get("logprobs")
        outputs = _unpad_logprobs_to_loss_fn_outputs(logprobs_batch, row_slices) if logprobs_batch is not None else []
        return ForwardBackwardOutput(
            loss_fn_output_type="ArrayRecord",
            loss_fn_outputs=outputs,
            metrics=arctic_metrics_to_tinker(r.get("metrics")),
        ).model_dump(mode="json")

    return await _submit(request, runner, model_id=req.model_id, kind=_KIND_FWD_BWD, serial=True)


@router.post("/optim_step", response_model=UntypedAPIFuture)
async def optim_step(req: OptimStepRequest, request: Request) -> UntypedAPIFuture:
    handler = _require_state(request.app.state, "tinker_step")
    _check_adam(getattr(request.app.state, "tinker_fixed_adam", None), req.adam_params)
    overrides = adam_params_to_optim_overrides(req.adam_params)
    request.app.state.tinker_pending_gradients.discard(req.model_id)

    async def runner() -> dict[str, Any]:
        r = await handler(overrides)
        return OptimStepResponse(
            metrics=arctic_metrics_to_tinker(r.get("metrics")),
        ).model_dump(mode="json")

    return await _submit(request, runner, model_id=req.model_id, serial=True)


# ---- weight sync / sampling -------------------------------------------------


@router.post("/save_weights", response_model=UntypedAPIFuture)
async def save_weights(req: SaveWeightsRequest, request: Request) -> UntypedAPIFuture:
    """Ack-only state save so cookbook recipes ending in ``save_state`` don't
    crash. On-disk persistence is extension E2."""

    async def runner() -> dict[str, Any]:
        gen = getattr(request.app.state, "tinker_state_gen", 0) + 1
        request.app.state.tinker_state_gen = gen
        path = req.path or f"tinker://main/state/{gen}"
        return SaveWeightsResponse(path=path).model_dump(mode="json")

    return await _submit(request, runner, model_id=req.model_id, serial=True)


@router.post("/save_weights_for_sampler", response_model=UntypedAPIFuture)
async def save_weights_for_sampler(req: SaveWeightsForSamplerRequest, request: Request) -> UntypedAPIFuture:
    handler = _require_state(request.app.state, "tinker_sync_weights")

    async def runner() -> dict[str, Any]:
        request.app.state.tinker_weight_gen = getattr(request.app.state, "tinker_weight_gen", 0) + 1
        gen = request.app.state.tinker_weight_gen
        await handler()
        return SaveWeightsForSamplerResponse(
            path=f"tinker://main/sampler_weights/{gen}",
            sampling_session_id=f"ss@{gen}",
        ).model_dump(mode="json")

    return await _submit(request, runner, model_id=req.model_id, serial=True)


def _gate_temperature(app_state: Any, temperature: float) -> None:
    """Refuse a sampling temperature the trainer cannot reproduce.

    A backend without a temperature post-processor scores every log-prob at
    1.0. Sampling at anything else makes the sampler and the trainer two
    different distributions: the importance ratio is then wrong by a factor
    that no metric on this path reports, and
    ``forward_backward_custom`` differentiates those same log-probs. Refusing
    costs a recipe one config change; accepting costs a silently worse model.
    """
    if getattr(app_state, "tinker_supports_temperature_scaling", True):
        return
    if abs(float(temperature) - 1.0) > 1e-9:
        raise HTTPException(
            400,
            f"sampling temperature={temperature!r} is not supported by this backend: "
            "it scores training log-probs at temperature 1.0 and has no "
            "temperature post-processor, so sampling at any other temperature "
            "would silently mismatch the sampler and the trainer. "
            "Set temperature=1.0 in your sampling params.",
        )


_TEACHER_SESSION_PREFIX = "teacher@"


def _teacher_for(app_state: Any, base_model: str | None, model_path: str | None) -> str | None:
    """The teacher a sampler for ``base_model`` reads from, or None for the trained model.

    The server samples only the model it trains and the teachers it was started
    with. Any other model would otherwise be answered by the trained model's
    sampler -- a distillation run would then score the student against itself.
    """
    served = _require_state(app_state, "tinker_base_model")
    teachers = getattr(app_state, "tinker_teacher_generate", None) or {}
    if base_model in teachers and model_path is None:
        return base_model
    if base_model is None or base_model == served:
        return None
    if base_model not in teachers:
        raise HTTPException(
            400,
            f"no sampler for base_model={base_model!r}: this server trains {served!r} and serves "
            f"teachers {sorted(teachers)}. Start it with --teacher-model {base_model} to sample it.",
        )
    if model_path is not None:
        raise HTTPException(
            400,
            f"teacher {base_model!r} is served from its base weights; loading model_path={model_path!r} "
            "into it is not supported",
        )
    return base_model


@router.post("/create_sampling_session", response_model=CreateSamplingSessionResponse)
async def create_sampling_session(
    req: CreateSamplingSessionRequest, request: Request
) -> CreateSamplingSessionResponse:
    teacher = _teacher_for(request.app.state, req.base_model, req.model_path)
    if teacher is not None:
        return CreateSamplingSessionResponse(sampling_session_id=f"{_TEACHER_SESSION_PREFIX}{teacher}")
    # A base-model session means the untrained weights, which the sampler holds
    # only until the first sync; pinning it to generation 0 makes later use 409.
    gen = 0 if req.model_path is None else getattr(request.app.state, "tinker_weight_gen", 0)
    return CreateSamplingSessionResponse(sampling_session_id=f"ss@{gen}")


def _sampler_for(
    app_state: Any, req: SampleRequest
) -> tuple[Callable[[list[int], dict], Awaitable[dict]], int | None]:
    """The generate handler a sample request reads from, and the weight generation
    it was issued for (None when it is not tied to one)."""
    session_id = req.sampling_session_id
    if session_id and session_id.startswith(_TEACHER_SESSION_PREFIX):
        teachers = getattr(app_state, "tinker_teacher_generate", None) or {}
        teacher = session_id[len(_TEACHER_SESSION_PREFIX) :]
        if teacher not in teachers:
            raise HTTPException(400, f"unknown sampling_session_id={session_id!r}")
        return teachers[teacher], None
    if not session_id:
        teacher = _teacher_for(app_state, req.base_model, req.model_path)
        if teacher is not None:
            return app_state.tinker_teacher_generate[teacher], None

    handler = _require_state(app_state, "tinker_generate")
    if session_id and session_id.startswith("ss@"):
        try:
            return handler, int(session_id.split("@", 1)[1])
        except ValueError:
            raise HTTPException(400, f"malformed sampling_session_id={session_id!r}")
    if not session_id and req.base_model is not None and req.model_path is None:
        return handler, 0  # see create_sampling_session
    return handler, None


@router.post("/asample", response_model=UntypedAPIFuture)
async def asample(req: SampleRequest, request: Request) -> UntypedAPIFuture:
    handler, gen = _sampler_for(request.app.state, req)
    _gate_temperature(request.app.state, req.sampling_params.temperature)
    if req.topk_prompt_logprobs:
        raise HTTPException(
            400,
            f"topk_prompt_logprobs={req.topk_prompt_logprobs} is not supported; "
            "prompt_logprobs returns the log-prob of each prompt token only",
        )

    current_gen = getattr(request.app.state, "tinker_weight_gen", 0)

    async def runner() -> dict[str, Any]:
        if gen is not None and gen < current_gen:
            raise HTTPException(
                409,
                f"stale sampling_session_id={req.sampling_session_id!r}; "
                f"server is at weight_gen={current_gen}, v1 requires strict-monotonic "
                "usage (multi-snapshot async-RL is extension E1).",
            )
        vllm_params = sampling_params_tinker_to_vllm(req.sampling_params, req.num_samples)
        if req.prompt_logprobs:
            # vLLM's 0 means "only the prompt token itself, no top-k extras".
            vllm_params["prompt_logprobs"] = 0
        prompt_tokens = _model_input_to_tokens(req.prompt)
        r = await handler(prompt_tokens, vllm_params)
        sequences = [
            SampledSequence(
                tokens=list(o.get("token_ids", [])),
                logprobs=list(o["logprobs"]) if o.get("logprobs") is not None else None,
                stop_reason=(StopReason.STOP if o.get("finish_reason") == "stop" else StopReason.LENGTH),
            )
            for o in (r.get("outputs") or [])
        ]
        prompt_logprobs = None
        if req.prompt_logprobs:
            prompt_logprobs = r.get("prompt_logprobs")
            if prompt_logprobs is None or len(prompt_logprobs) != len(prompt_tokens):
                raise HTTPException(
                    502,
                    f"asked for prompt log-probs of {len(prompt_tokens)} tokens and the sampler returned "
                    f"{None if prompt_logprobs is None else len(prompt_logprobs)}",
                )
        return SampleResponse(sequences=sequences, prompt_logprobs=prompt_logprobs).model_dump(mode="json")

    return await _submit(request, runner, kind=_KIND_SAMPLE)


# ---- futures ----------------------------------------------------------------


@router.post("/retrieve_future")
async def retrieve_future(req: FutureRetrieveRequest, request: Request):
    store: TinkerFutureStore = _require_state(request.app.state, "tinker_futures")
    entry = await store.pop(req.request_id, _RETRIEVE_WAIT_S)
    if entry is None:
        return TryAgainResponse().model_dump()
    task, kind = entry
    err = task.exception()
    if err is not None:
        if isinstance(err, HTTPException) and err.status_code < 500:
            return RequestFailedResponse(error=str(err.detail), category="user").model_dump()
        logger.error("tinker request %s failed", req.request_id, exc_info=err)
        return RequestFailedResponse(error=f"{type(err).__name__}: {err}", category="server").model_dump()
    payload = task.result()
    encoder = _PROTO_ENCODERS.get(kind or "")
    # A proto-only result type is signalled by the Accept header; a JSON reply
    # to such a caller raises on its side rather than degrading.
    if encoder is not None and wants_proto(request.headers.get("accept")):
        return Response(content=encoder(payload), media_type=PROTO_CONTENT_TYPE)
    return payload


# =============================================================================
# App wiring helper
# =============================================================================


def init_tinker_state(
    app,
    *,
    base_model: str,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int,
    fwd_bwd_handler: Callable[[dict], Awaitable[dict]],
    step_handler: Callable[[dict | None], Awaitable[dict]],
    sync_weights_handler: Callable[[], Awaitable[Any]],
    generate_handler: Callable[[list[int], dict], Awaitable[dict]],
    fwd_no_grad_handler: Callable[[dict], Awaitable[dict]] | None = None,
    supports_temperature_scaling: bool = True,
    teacher_generate_handlers: Mapping[str, Callable[[list[int], dict], Awaitable[dict]]] | None = None,
    lora: LoraConfig | None = None,
    fixed_adam: Mapping[str, float] | None = None,
    accumulates_gradients: bool = True,
) -> None:
    """Wire the Tinker verbs onto ``app.state`` as async closures. Callers
    (real Arctic http_server, in-process tests with a mocked backend) inject
    per-verb handlers so the router never reaches into ``app.state.jobs``.

    ``supports_temperature_scaling=False`` declares that the backend scores
    log-probs at temperature 1.0 regardless of what the sampler was asked for,
    which makes any other sampling temperature a silent train/sample mismatch;
    ``sample`` then refuses it. See :func:`asample`.

    Without ``fwd_no_grad_handler``, ``forward`` returns 400.

    ``teacher_generate_handlers`` maps a model name to the sampler serving its
    base weights, for ``create_sampling_client(base_model=...)`` on a model
    other than the one being trained (on-policy distillation's teacher).

    ``lora`` is the adapter the backend trains, ``None`` for full fine-tuning.
    ``fixed_adam`` holds the ``beta1`` / ``beta2`` / ``eps`` / ``weight_decay`` /
    ``grad_clip_norm`` a backend provisioned and cannot change per step;
    ``optim_step`` refuses any other values. ``None`` means the step handler
    applies them all.

    ``accumulates_gradients=False`` declares that ``step`` applies only a
    single ``forward_backward``'s gradient; a second call before ``optim_step``
    then returns 400. See :func:`_claim_gradient`."""
    app.state.tinker_base_model = base_model
    app.state.tinker_max_prompt_length = int(max_prompt_length)
    app.state.tinker_max_response_length = int(max_response_length)
    app.state.tinker_pad_token_id = int(pad_token_id)
    app.state.tinker_futures = TinkerFutureStore()
    app.state.tinker_sessions = {}
    app.state.tinker_models = {}
    app.state.tinker_weight_gen = 0
    app.state.tinker_fwd_bwd = fwd_bwd_handler
    app.state.tinker_fwd_no_grad = fwd_no_grad_handler
    app.state.tinker_step = step_handler
    app.state.tinker_sync_weights = sync_weights_handler
    app.state.tinker_generate = generate_handler
    app.state.tinker_teacher_generate = dict(teacher_generate_handlers or {})
    app.state.tinker_lora = lora
    app.state.tinker_fixed_adam = None if fixed_adam is None else {f: float(fixed_adam[f]) for f in _ADAM_FIELDS}
    app.state.tinker_supports_temperature_scaling = bool(supports_temperature_scaling)
    app.state.tinker_accumulates_gradients = bool(accumulates_gradients)
    app.state.tinker_pending_gradients = set()
