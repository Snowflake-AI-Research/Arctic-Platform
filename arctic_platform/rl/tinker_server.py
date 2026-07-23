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

"""Tinker HTTP layer for Arctic-Platform (v1).

Exposes Arctic's colocated RL server over the upstream
`tinker <https://github.com/thinking-machines-lab/tinker>`_ HTTP protocol.

Scope (v1): RL only, colocated (``colocate=True``, CUDA-IPC weight sync),
single global training run, no auth. Full-weight DeepSpeed training via the
SkyRL-tx ``LoraConfig(rank=0)`` = FFT convention; ``rank>0`` returns 400.

Design:
    - Every long-running Tinker verb (``forward``, ``forward_backward``,
      ``optim_step``, ``save_weights_for_sampler``, ``asample``,
      ``create_model``) is future-based on the wire. v1 runs the work
      synchronously in the request handler and caches the terminal
      response in an in-memory ``dict[request_id] -> response``; the
      first ``retrieve_future`` poll returns the completed result.
    - Wire schemas are Pydantic models pinned to
      ``tinker.types.*`` (SDK at HEAD of ``main`` when this was
      landed). Round-trip tests in ``tests/tinker_layer/`` guard against
      upstream drift.
    - The router lowers into existing Arctic HTTP handlers via in-process
      calls, not a second HTTP hop. Adapters live in this file.

See ``arctic_platform/rl/TINKER_COMPAT.md`` for the full design.
"""

from __future__ import annotations

import itertools
import time
import uuid
from enum import Enum
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Literal
from typing import Sequence
from typing import Union

import numpy as np
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# =============================================================================
# Wire schemas — Pydantic mirrors of ``tinker.types.*``
# =============================================================================
#
# We redefine the wire shape locally so the server has no runtime dependency on
# the ``tinker`` SDK. ``tests/tinker_layer/test_wire_schema.py`` runs upstream
# ``model_dump()`` payloads through these classes to guard against drift.


class TensorData(BaseModel):
    """Mirror of ``tinker.types.TensorData``. The upstream dataclass
    projects to this shape via ``model_dump(mode='json')``."""

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


class ForwardInput(BaseModel):
    data: list[Datum]
    loss_fn: LossFnType


class ForwardRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    forward_input: ForwardInput
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
    # Force JSON over proto in v1 — the server has no zstd/proto path.
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
    prompt_logprobs: list[float] | None = None
    type: Literal["sample"] = "sample"


class UntypedAPIFuture(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_id: str | None = None
    type: Literal["future"] = "future"


class TryAgainResponse(BaseModel):
    type: Literal["try_again"] = "try_again"


class FutureRetrieveRequest(BaseModel):
    request_id: str
    allow_metadata_only: bool = False


# =============================================================================
# Adapters — Tinker wire types → Arctic native shapes
# =============================================================================


# Tinker cookbook (tutorials/202_loss_functions.py) documents the PPO keys:
# ``clip_low_threshold`` (default 0.8) and ``clip_high_threshold`` (default 1.2).
# Arctic's GRPO loss (arctic_platform/rl/processors/grpo.py) reads
# ``eps_clip`` (symmetric) and ``eps_clip_higher`` (asymmetric upper bound).
# ``importance_sampling`` = PPO with clipping disabled (unbounded IS ratio).
_TINKER_PPO_CLIP_LOW_DEFAULT = 0.8
_TINKER_PPO_CLIP_HIGH_DEFAULT = 1.2
_UNBOUNDED_CLIP = 1e9


def _loss_fn_config_to_actor_config(
    loss_fn: str,
    loss_fn_config: dict[str, float] | None,
) -> dict[str, Any]:
    """Translate a Tinker ``loss_fn`` + ``loss_fn_config`` into an Arctic
    ``actor_config`` dict consumed by ``processors/grpo.py``.

    PPO: ``clip_low_threshold`` -> ``eps_clip`` (as ``1 - x``, since Arctic's
        ``eps_clip`` is an epsilon around 1.0 in the standard PPO ratio
        formulation while Tinker exposes the actual ratio bounds).
    importance_sampling: both bounds effectively disabled.

    Only ``ppo`` and ``importance_sampling`` are supported in v1; the wire
    gate in ``forward_backward`` rejects the others upstream.
    """
    cfg = dict(loss_fn_config or {})
    if loss_fn == "ppo":
        low = cfg.get("clip_low_threshold", _TINKER_PPO_CLIP_LOW_DEFAULT)
        high = cfg.get("clip_high_threshold", _TINKER_PPO_CLIP_HIGH_DEFAULT)
        # Arctic ``eps_clip`` is a symmetric epsilon around ratio=1: ratio is
        # clamped to [1 - eps_clip, 1 + eps_clip_higher]. Convert to that shape.
        eps_clip = max(0.0, 1.0 - float(low))
        eps_clip_higher = max(0.0, float(high) - 1.0)
        actor_cfg: dict[str, Any] = {
            "eps_clip": eps_clip,
            "eps_clip_higher": eps_clip_higher,
        }
    elif loss_fn == "importance_sampling":
        # Disable clipping — worst case falls back to unclipped IS.
        actor_cfg = {
            "eps_clip": _UNBOUNDED_CLIP,
            "eps_clip_higher": _UNBOUNDED_CLIP,
        }
    else:  # pragma: no cover — gated by route
        raise HTTPException(400, f"unsupported loss_fn={loss_fn!r}")

    if "kl_coef" in cfg:
        actor_cfg["kl_loss_coef"] = float(cfg["kl_coef"])
        actor_cfg["use_kl_loss"] = float(cfg["kl_coef"]) > 0.0
    if "entropy_coef" in cfg:
        actor_cfg["entropy_coeff"] = float(cfg["entropy_coef"])
    return actor_cfg


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


def _pad_to(arr: np.ndarray, target_len: int, pad_value: float | int) -> np.ndarray:
    """Right-pad a 1-D array to ``target_len`` with ``pad_value``. Truncates
    if the input is longer — the caller must have config-max padding
    invariant (ZoRRo compatibility).
    """
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]
    pad = np.full(target_len - arr.shape[0], pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad])


def datum_list_to_arctic_batch(
    data: list[Datum],
    loss_fn: str,
    loss_fn_config: dict[str, float] | None,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int,
    forward_only: bool = False,
) -> dict:
    """Pack a list of Tinker ``Datum`` into an Arctic ``fwd_bwd`` batch dict.

    Layout produced (mirrors ``arctic_platform/rl/utils/batch.py::unpack_batch``):
        - ``batch``: {input_ids, attention_mask, position_ids, advantages,
                      old_log_probs, loss_mask, response_mask}
        - ``meta``:  {actor_config, max_response_len, forward_only}
        - ``processing``: {loss_fn: <mapped>}

    ZoRRo invariant: pad each row to ``max_prompt_length +
    max_response_length`` (config-max), never batch-local.
    """
    total_len = int(max_prompt_length) + int(max_response_length)
    batch_size = len(data)

    input_ids = np.full((batch_size, total_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((batch_size, total_len), dtype=np.int64)
    advantages = np.zeros((batch_size, total_len), dtype=np.float32)
    old_log_probs = np.zeros((batch_size, total_len), dtype=np.float32)
    loss_mask = np.zeros((batch_size, total_len), dtype=np.float32)

    for i, datum in enumerate(data):
        toks = _model_input_to_tokens(datum.model_input)
        n = min(len(toks), total_len)
        input_ids[i, :n] = np.asarray(toks[:n], dtype=np.int64)
        attention_mask[i, :n] = 1

        inputs = datum.loss_fn_inputs
        if "advantages" in inputs:
            arr = _tensor_data_to_numpy(inputs["advantages"]).astype(np.float32)
            advantages[i] = _pad_to(arr, total_len, 0.0)
        if "logprobs" in inputs:
            arr = _tensor_data_to_numpy(inputs["logprobs"]).astype(np.float32)
            old_log_probs[i] = _pad_to(arr, total_len, 0.0)
        # Both ``mask`` and ``weights`` map to Arctic's loss_mask; cookbook
        # writes ``mask`` (data_processing.py L159), the wire spec calls out
        # ``weights`` in Datum._KEY_TO_TYPE. Accept either.
        mask_td = inputs.get("weights") or inputs.get("mask")
        if mask_td is not None:
            arr = _tensor_data_to_numpy(mask_td).astype(np.float32)
            loss_mask[i] = _pad_to(arr, total_len, 0.0)

    # position_ids reflect real (unpadded) positions.
    position_ids = np.tile(np.arange(total_len, dtype=np.int64), (batch_size, 1))

    actor_config: dict[str, Any] = {}
    if not forward_only:
        actor_config = _loss_fn_config_to_actor_config(loss_fn, loss_fn_config)

    return {
        "batch": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
            "loss_mask": loss_mask,
            "response_mask": loss_mask,
        },
        "meta": {
            "actor_config": actor_config,
            "max_response_len": int(max_response_length),
            "forward_only": bool(forward_only),
        },
        "processing": {"loss_fn": loss_fn},
    }


_UNIQUE_REDUCTIONS = {"unique", "hash_unordered"}


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
    """Trivial in-memory future backing store.

    v1 executes work inline in the request handler and stashes the terminal
    response keyed by ``request_id``. The client's first
    ``retrieve_future`` call pops it and moves on. Retention beyond one
    read is not required by the SDK's ``_APIFuture`` implementation.

    E-async replaces this with an ``asyncio.Task``-per-future model that
    returns ``TryAgainResponse`` until the task completes; the wire and
    Pydantic surface do not change.
    """

    def __init__(self) -> None:
        self._counter = itertools.count()
        self._store: dict[str, dict[str, Any]] = {}

    def new_request_id(self) -> str:
        return str(next(self._counter))

    def put(self, request_id: str, payload: dict[str, Any]) -> None:
        self._store[request_id] = payload

    def pop(self, request_id: str) -> dict[str, Any] | None:
        return self._store.pop(request_id, None)


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/v1")

_V1_SUPPORTED_LOSSES = frozenset({"ppo", "importance_sampling"})
_V1_UNSUPPORTED_LOSSES = frozenset({"cispo", "dro", "cross_entropy"})


def _require_state(app_state: Any, name: str) -> Any:
    """Fetch app.state.<name> or 500 if the layer wasn't initialized. Kept
    small and explicit so mock backends in tests can override per-attribute."""
    if not hasattr(app_state, name):
        raise HTTPException(
            500,
            f"Tinker layer misconfigured: app.state.{name} is unset. "
            "Call arctic_platform.rl.tinker_server.init_tinker_state() at startup.",
        )
    return getattr(app_state, name)


async def _submit_inline(
    request: Request,
    runner: Callable[[], Awaitable[dict[str, Any]]],
    *,
    model_id: str | None = None,
) -> UntypedAPIFuture:
    """Execute ``runner`` inline, stash the response in the future store, and
    return the SDK's ``UntypedAPIFuture`` shape."""
    store: TinkerFutureStore = _require_state(request.app.state, "tinker_futures")
    request_id = store.new_request_id()
    result = await runner()
    store.put(request_id, result)
    return UntypedAPIFuture(request_id=request_id, model_id=model_id)


# ---- session / bootstrap verbs ----------------------------------------------


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
async def client_config(req: ClientConfigRequest) -> ClientConfigResponse:
    return ClientConfigResponse()


@router.post("/auth/token", response_model=AuthTokenResponse)
async def auth_token() -> AuthTokenResponse:
    return AuthTokenResponse(jwt="tml-dummy")


@router.post("/telemetry", response_model=TelemetryResponse)
async def telemetry(req: dict) -> TelemetryResponse:
    # Drop events on the floor; the SDK sends batched telemetry as
    # best-effort. Explicit ack keeps the client from retrying.
    return TelemetryResponse()


@router.get("/get_server_capabilities", response_model=GetServerCapabilitiesResponse)
async def get_server_capabilities(request: Request) -> GetServerCapabilitiesResponse:
    base_model = _require_state(request.app.state, "tinker_base_model")
    return GetServerCapabilitiesResponse(supported_models=[SupportedModel(model_name=base_model)])


# ---- model lifecycle --------------------------------------------------------


@router.post("/create_model", response_model=UntypedAPIFuture)
async def create_model(req: CreateModelRequest, request: Request) -> UntypedAPIFuture:
    base_model = _require_state(request.app.state, "tinker_base_model")
    if req.base_model != base_model:
        raise HTTPException(
            400,
            f"server was started with base_model={base_model!r}, "
            f"got base_model={req.base_model!r}",
        )
    if req.lora_config is not None and req.lora_config.rank != 0:
        raise HTTPException(
            400,
            "Arctic v1 supports full-weight training only; pass "
            "LoraConfig(rank=0) to opt into the SkyRL-tx FFT convention. "
            "LoRA (rank>0) is captured as extension E1.",
        )
    models = _require_state(request.app.state, "tinker_models")
    model_id = "main"  # single-tenant in v1
    models[model_id] = {"base_model": req.base_model, "lora_config": req.lora_config}

    async def runner() -> dict[str, Any]:
        return CreateModelResponse(
            model_id=model_id,
            base_model=req.base_model,
            lora_config=req.lora_config,
        ).model_dump(mode="json")

    return await _submit_inline(request, runner, model_id=model_id)


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
            f"loss_fn={loss_fn!r} not supported in v1; "
            f"supported: {sorted(_V1_SUPPORTED_LOSSES)}",
        )
    if loss_fn not in _V1_SUPPORTED_LOSSES:
        raise HTTPException(400, f"unknown loss_fn={loss_fn!r}")


@router.post("/forward_backward", response_model=UntypedAPIFuture)
async def forward_backward(
    req: ForwardBackwardRequest, request: Request
) -> UntypedAPIFuture:
    fbi = req.forward_backward_input
    _gate_loss_fn(fbi.loss_fn)
    handler = _require_state(request.app.state, "tinker_fwd_bwd")
    max_prompt = _require_state(request.app.state, "tinker_max_prompt_length")
    max_resp = _require_state(request.app.state, "tinker_max_response_length")
    pad_id = _require_state(request.app.state, "tinker_pad_token_id")
    batch = datum_list_to_arctic_batch(
        fbi.data,
        fbi.loss_fn,
        fbi.loss_fn_config,
        max_prompt,
        max_resp,
        pad_id,
        forward_only=False,
    )

    n_data = len(fbi.data)

    async def runner() -> dict[str, Any]:
        r = await handler(batch)
        # Tinker expects ``len(loss_fn_outputs)`` to equal the per-actor
        # sample count; it is used as the reduction weight when combining
        # metrics across actors. Arctic returns a single aggregated batch,
        # so we emit one empty ``LossFnOutput`` per Datum which keeps the
        # weight correct without fabricating per-sample tensors.
        return ForwardBackwardOutput(
            loss_fn_outputs=[{} for _ in range(n_data)],
            metrics=arctic_metrics_to_tinker(r.get("metrics")),
        ).model_dump(mode="json")

    return await _submit_inline(request, runner, model_id=req.model_id)


@router.post("/forward", response_model=UntypedAPIFuture)
async def forward(req: ForwardRequest, request: Request) -> UntypedAPIFuture:
    _gate_loss_fn(req.forward_input.loss_fn)
    handler = _require_state(request.app.state, "tinker_fwd_no_grad")
    max_prompt = _require_state(request.app.state, "tinker_max_prompt_length")
    max_resp = _require_state(request.app.state, "tinker_max_response_length")
    pad_id = _require_state(request.app.state, "tinker_pad_token_id")
    batch = datum_list_to_arctic_batch(
        req.forward_input.data,
        req.forward_input.loss_fn,
        None,
        max_prompt,
        max_resp,
        pad_id,
        forward_only=True,
    )

    async def runner() -> dict[str, Any]:
        r = await handler(batch)
        # ``fwd-no-grad`` returns per-token logprobs in ``batch['logprobs']``.
        # Repack per Datum so the SDK's LossFnOutput mapping matches.
        logprobs_batch = r.get("batch", {}).get("logprobs")
        outputs: list[dict[str, TensorData]] = []
        if logprobs_batch is not None:
            arr = np.asarray(logprobs_batch, dtype=np.float32)
            for row in arr:
                outputs.append({"logprobs": TensorData(dtype="float32",
                                                       data=row.tolist(),
                                                       shape=list(row.shape))})
        return ForwardBackwardOutput(
            loss_fn_output_type="ArrayRecord",
            loss_fn_outputs=[{k: v.model_dump() for k, v in out.items()}
                             for out in outputs],
            metrics=arctic_metrics_to_tinker(r.get("metrics")),
        ).model_dump(mode="json")

    return await _submit_inline(request, runner, model_id=req.model_id)


@router.post("/optim_step", response_model=UntypedAPIFuture)
async def optim_step(req: OptimStepRequest, request: Request) -> UntypedAPIFuture:
    handler = _require_state(request.app.state, "tinker_step")
    overrides = adam_params_to_optim_overrides(req.adam_params)

    async def runner() -> dict[str, Any]:
        r = await handler(overrides)
        return OptimStepResponse(
            metrics=arctic_metrics_to_tinker(r.get("metrics")),
        ).model_dump(mode="json")

    return await _submit_inline(request, runner, model_id=req.model_id)


# ---- weight sync / sampling -------------------------------------------------


@router.post("/save_weights_for_sampler", response_model=UntypedAPIFuture)
async def save_weights_for_sampler(
    req: SaveWeightsForSamplerRequest, request: Request
) -> UntypedAPIFuture:
    handler = _require_state(request.app.state, "tinker_sync_weights")

    async def runner() -> dict[str, Any]:
        request.app.state.tinker_weight_gen = getattr(
            request.app.state, "tinker_weight_gen", 0
        ) + 1
        gen = request.app.state.tinker_weight_gen
        await handler()
        return SaveWeightsForSamplerResponse(
            path=f"tinker://main/sampler_weights/{gen}",
            sampling_session_id=f"ss@{gen}",
        ).model_dump(mode="json")

    return await _submit_inline(request, runner, model_id=req.model_id)


@router.post("/create_sampling_session", response_model=CreateSamplingSessionResponse)
async def create_sampling_session(
    req: CreateSamplingSessionRequest, request: Request
) -> CreateSamplingSessionResponse:
    gen = getattr(request.app.state, "tinker_weight_gen", 0)
    return CreateSamplingSessionResponse(sampling_session_id=f"ss@{gen}")


@router.post("/asample", response_model=UntypedAPIFuture)
async def asample(req: SampleRequest, request: Request) -> UntypedAPIFuture:
    handler = _require_state(request.app.state, "tinker_generate")
    gen = None
    if req.sampling_session_id and req.sampling_session_id.startswith("ss@"):
        try:
            gen = int(req.sampling_session_id.split("@", 1)[1])
        except ValueError:
            raise HTTPException(
                400, f"malformed sampling_session_id={req.sampling_session_id!r}"
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
        prompt_tokens = _model_input_to_tokens(req.prompt)
        r = await handler(prompt_tokens, vllm_params)
        return _pack_sample_response(r).model_dump(mode="json")

    return await _submit_inline(request, runner)


def _pack_sample_response(r: dict[str, Any]) -> SampleResponse:
    """Wrap Arctic ``/generate`` result into Tinker ``SampleResponse``.

    Arctic returns ``{"outputs": [{"token_ids": [...], "logprobs": [...],
    "finish_reason": "stop"|"length", ...}, ...]}`` per sample. Missing
    ``logprobs`` fields fall through as ``None`` on the wire.
    """
    outputs = r.get("outputs") or []
    sequences: list[SampledSequence] = []
    for out in outputs:
        reason = out.get("finish_reason", "length")
        stop = StopReason.STOP if reason == "stop" else StopReason.LENGTH
        sequences.append(
            SampledSequence(
                tokens=list(out.get("token_ids", [])),
                logprobs=(list(out["logprobs"]) if out.get("logprobs") is not None else None),
                stop_reason=stop,
            )
        )
    return SampleResponse(sequences=sequences)


# ---- futures ----------------------------------------------------------------


@router.post("/retrieve_future")
async def retrieve_future(req: FutureRetrieveRequest, request: Request):
    store: TinkerFutureStore = _require_state(request.app.state, "tinker_futures")
    payload = store.pop(req.request_id)
    if payload is None:
        return TryAgainResponse().model_dump()
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
    fwd_no_grad_handler: Callable[[dict], Awaitable[dict]],
    step_handler: Callable[[dict | None], Awaitable[dict]],
    sync_weights_handler: Callable[[], Awaitable[Any]],
    generate_handler: Callable[[list[int], dict], Awaitable[dict]],
) -> None:
    """Bind the Tinker layer onto a FastAPI ``app`` and register per-verb
    Arctic handlers.

    The handlers are thin async closures so callers (real Arctic HTTP
    server, test harness with a mocked backend) can plug them in without
    reaching into ``app.state.jobs``. The lambda in Arctic's real
    ``http_server`` binds the training/sampling job_ids captured at
    startup.
    """
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
