"""HTTP client for the Neutrino GS API.

Single-purpose, production-shaped client:

- One auth path: Programmatic Access Token (PAT) — see ``NeutrinoClient.from_pat``.
- One CreateJob shape: a list of typed :class:`SubJobConfig` (each carries either
  a :class:`TrainingConfig` or an :class:`InferenceConfig`). Validation mirrors
  the Control Plane validators in
  ``cortex/neutrino/pkg/controlplane/server.go`` (``protoToSubJobConfig`` /
  ``trainingConfig.validate`` / ``inferenceConfig.validate``).
- One wire format: the GS REST shape from
  ``GlobalServices/.../neutrino.yaml``: ``{job_id?, sub_jobs:[SubJobConfig]}``
  with per-sub-job ``training`` / ``sampling`` blocks.

The Control Plane proto (``cortex/neutrino/protos/control_plane.proto``) and the
GS yaml keep ``additionalProperties: true`` on the config blocks so new fields
land in the Control Plane only. The strict shape lives here in the client.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from tenacity import Retrying
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential_jitter
from urllib3.exceptions import NewConnectionError

from dss_client import wire

logger = logging.getLogger(__name__)

# Env var that unlocks create-job debug options. Debug options (e.g. the
# image_tag override that selects an arbitrary dss-backend build for a job's
# zone) are an internal-only capability: the client refuses to forward a
# create-job request carrying a `debug` block unless this is set to a truthy
# value, keeping the feature hidden from normal external use.
DEBUG_OPTIONS_ENV = "DSS_NEUTRINO_ENABLE_DEBUG_OPTIONS"


def _debug_options_enabled() -> bool:
    """True when create-job debug options are explicitly enabled via env var."""
    return os.environ.get(DEBUG_OPTIONS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# HTTP statuses worth retrying: the request was well-formed, so the same call
# may succeed once transient load/infra conditions clear. 4xx (except 429) are
# excluded because they signal a client/config error that won't fix itself.
#   429 Too Many Requests  - rate limited; back off and retry
#   500 Internal Server Err - transient server-side failure
#   502 Bad Gateway         - upstream/proxy hiccup between GS and the backend
#   503 Service Unavailable - server temporarily overloaded or restarting
#   504 Gateway Timeout      - upstream didn't respond in time
#   409                      - ZMD restarting
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504, 404, 409}


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = exc.response
        return resp is not None and resp.status_code in _TRANSIENT_STATUSES
    return False


def _is_connect_error(exc: BaseException) -> bool:
    """Only failures proving the request never reached the server (safe for create_job)."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        cause = exc.args[0] if exc.args else None
        reason = getattr(cause, "reason", None)
        return isinstance(cause, NewConnectionError) or isinstance(reason, NewConnectionError)
    return False


def _truthy_fused_cross_entropy(value: object) -> bool:
    return value is not False


def _has_int_chunk_size(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_primerl_lm_head_config(config: dict, *, location: str) -> None:
    fused_cross_entropy = config.get("fused_cross_entropy", "liger")
    if _truthy_fused_cross_entropy(fused_cross_entropy) and (
        config.get("fp32_lm_head", False) is True
        or _has_int_chunk_size(config.get("fused_lm_head_token_chunk_size"))
    ):
        raise ValueError(
            f"{location} cannot combine fused_cross_entropy with "
            "fp32_lm_head or fused_lm_head_token_chunk_size. "
            "Set fused_cross_entropy=false to use fp32/chunked LM head, "
            "or remove fp32/chunked LM-head knobs to use fused CE."
        )


# ─── Canonical types ─────────────────────────────────────────────────────


class JobType(str, Enum):
    """Sub-job types supported by the Neutrino platform.

    Mirrors ``subjob.Type`` in ``cortex/neutrino/pkg/subjob/types.go`` and the
    ``job_type`` enum in the GS REST yaml.
    """

    TRAINING = "training"
    SAMPLING = "sampling"
    LOG_PROBABILITY = "log_probability"


def _effective_primerl_config(extra: dict) -> dict:
    prime_rl = extra.get("prime_rl")
    if isinstance(prime_rl, dict):
        # DSS runtime expands the nested prime_rl block over the training
        # config, so validate the same effective values here.
        return {**extra, **prime_rl}
    return extra


def _validate_primerl_lm_head_config(extra: dict, *, location: str) -> None:
    cfg = _effective_primerl_config(extra)
    fused_cross_entropy = cfg.get("fused_cross_entropy", "liger")
    if fused_cross_entropy is False:
        return

    token_chunk_size = cfg.get("fused_lm_head_token_chunk_size")
    token_chunked_lm_head = isinstance(token_chunk_size, int) and not isinstance(
        token_chunk_size, bool
    )
    if cfg.get("fp32_lm_head", False) or token_chunked_lm_head:
        raise ValueError(
            f"{location} cannot combine fused_cross_entropy with "
            "fp32_lm_head or fused_lm_head_token_chunk_size. "
            "Set fused_cross_entropy=false to use fp32/chunked LM head, "
            "or remove fp32/chunked LM-head knobs to use fused CE."
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters for a training sub-job.

    Required fields mirror the Control Plane validator in
    ``cortex/neutrino/pkg/controlplane/server.go`` (``trainingConfig``):
    ``max_seq_len > 0`` and ``train_batch_size > 0``.

    ``extra`` carries any additional fields the dss-platform training worker
    consumes. The proto Struct is intentionally open (additionalProperties),
    so unknown keys flow through unchanged.
    """

    optimizer: dict
    max_seq_len: int
    train_batch_size: int
    n_gpus: int
    gradient_clipping: float | None = None
    multiplex_job_id: str | None = None
    # When False, resuming from a checkpoint loads weights only and starts the
    # optimizer fresh. Required to change DP size (the DP-sharded optimizer
    # cannot be resized); None leaves the server default (True).
    load_optimizer_states: bool | None = None
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.optimizer, dict) or not self.optimizer:
            raise ValueError("training.optimizer is required and must be a non-empty dict")
        if self.max_seq_len <= 0:
            raise ValueError("training.max_seq_len must be > 0")
        if self.train_batch_size <= 0:
            raise ValueError("training.train_batch_size must be > 0")
        if self.n_gpus <= 0:
            raise ValueError("training.n_gpus must be > 0")
        _validate_primerl_lm_head_config(self.extra, location="training.extra")
        prime_rl = self.extra.get("prime_rl")
        if isinstance(prime_rl, dict):
            _validate_primerl_lm_head_config(prime_rl, location="training.extra.prime_rl")

    def to_wire(self) -> dict:
        out: dict = {
            "optimizer": self.optimizer,
            "max_seq_len": self.max_seq_len,
            "train_batch_size": self.train_batch_size,
            "n_gpus": self.n_gpus,
        }
        if self.gradient_clipping is not None:
            out["gradient_clipping"] = self.gradient_clipping
        if self.multiplex_job_id is not None:
            out["multiplex_job_id"] = self.multiplex_job_id
        if self.load_optimizer_states is not None:
            out["load_optimizer_states"] = self.load_optimizer_states
        for k, v in self.extra.items():
            out.setdefault(k, v)
        return out


@dataclass
class InferenceConfig:
    """Sampling/log-probability config for an inference sub-job.

    Required fields mirror the Control Plane validator
    (``inferenceConfig`` in ``controlplane/server.go``): ``max_seq_len > 0``.
    ``extra`` carries vLLM-style passthrough keys (e.g. gpu_memory_utilization).
    """

    max_seq_len: int
    n_gpus: int
    multiplex_job_id: str | None = None
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.max_seq_len <= 0:
            raise ValueError("sampling.max_seq_len must be > 0")
        if self.n_gpus <= 0:
            raise ValueError("sampling.n_gpus must be > 0")

    def to_wire(self) -> dict:
        out: dict = {"max_seq_len": self.max_seq_len, "n_gpus": self.n_gpus}
        if self.multiplex_job_id is not None:
            out["multiplex_job_id"] = self.multiplex_job_id
        for k, v in self.extra.items():
            out.setdefault(k, v)
        return out


@dataclass
class SubJobConfig:
    """One sub-job within a CreateJob request.

    Mirrors ``subjob.Config`` in ``cortex/neutrino/pkg/subjob/types.go`` and the
    ``SubJobConfig`` schema in the GS yaml.

    Exactly one of ``training`` or ``sampling`` must be set, matching the
    sub-job's ``job_type`` (see :meth:`validate`).
    """

    job_type: JobType
    model_name: str
    training: TrainingConfig | None = None
    sampling: InferenceConfig | None = None
    global_batch_size: int | None = None
    dtype: str | None = None
    seed: int | None = None
    model_post_init: list[str] | None = None
    source_checkpoint_info: dict | None = None

    @classmethod
    def training_job(
        cls,
        model_name: str,
        *,
        optimizer: dict,
        max_seq_len: int,
        train_batch_size: int,
        n_gpus: int,
        gradient_clipping: float | None = None,
        multiplex_job_id: str | None = None,
        load_optimizer_states: bool | None = None,
        extra_training: dict | None = None,
        global_batch_size: int | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        model_post_init: list[str] | None = None,
        source_checkpoint_info: dict | None = None,
    ) -> "SubJobConfig":
        """Build a training :class:`SubJobConfig`. All training fields are explicit."""
        return cls(
            job_type=JobType.TRAINING,
            model_name=model_name,
            training=TrainingConfig(
                optimizer=optimizer,
                max_seq_len=max_seq_len,
                train_batch_size=train_batch_size,
                n_gpus=n_gpus,
                gradient_clipping=gradient_clipping,
                multiplex_job_id=multiplex_job_id,
                load_optimizer_states=load_optimizer_states,
                extra=dict(extra_training) if extra_training else {},
            ),
            global_batch_size=global_batch_size,
            dtype=dtype,
            seed=seed,
            model_post_init=model_post_init,
            source_checkpoint_info=source_checkpoint_info,
        )

    @classmethod
    def sampling_job(
        cls,
        model_name: str,
        *,
        max_seq_len: int,
        n_gpus: int,
        multiplex_job_id: str | None = None,
        extra_sampling: dict | None = None,
        job_type: JobType = JobType.SAMPLING,
        global_batch_size: int | None = None,
        dtype: str | None = None,
        seed: int | None = None,
        model_post_init: list[str] | None = None,
        source_checkpoint_info: dict | None = None,
    ) -> "SubJobConfig":
        """Build a sampling/log-probability :class:`SubJobConfig`."""
        if job_type not in (JobType.SAMPLING, JobType.LOG_PROBABILITY):
            raise ValueError(
                f"sampling_job() only accepts SAMPLING or LOG_PROBABILITY, got {job_type!r}"
            )
        return cls(
            job_type=job_type,
            model_name=model_name,
            sampling=InferenceConfig(
                max_seq_len=max_seq_len,
                n_gpus=n_gpus,
                multiplex_job_id=multiplex_job_id,
                extra=dict(extra_sampling) if extra_sampling else {},
            ),
            global_batch_size=global_batch_size,
            dtype=dtype,
            seed=seed,
            model_post_init=model_post_init,
            source_checkpoint_info=source_checkpoint_info,
        )

    def validate(self) -> None:
        # Mirrors protoToSubJobConfig in controlplane/server.go (lines 772-823).
        if not self.model_name:
            raise ValueError("sub_job.model_name is required")
        if self.training is not None and self.sampling is not None:
            raise ValueError("sub_job.training and sub_job.sampling are mutually exclusive")
        if self.job_type == JobType.TRAINING:
            if self.training is None:
                raise ValueError("training sub-job requires a `training` block")
            self.training.validate()
        elif self.job_type in (JobType.SAMPLING, JobType.LOG_PROBABILITY):
            if self.sampling is None:
                raise ValueError(
                    f"{self.job_type.value} sub-job requires a `sampling` block"
                )
            self.sampling.validate()
        else:  # pragma: no cover - JobType enum is closed
            raise ValueError(f"unknown job_type: {self.job_type!r}")

    def to_wire(self) -> dict:
        wire: dict = {
            "job_type": self.job_type.value,
            "model_name": self.model_name,
        }
        if self.global_batch_size is not None:
            wire["global_batch_size"] = self.global_batch_size
        if self.dtype is not None:
            wire["dtype"] = self.dtype
        if self.seed is not None:
            wire["seed"] = self.seed
        if self.model_post_init is not None:
            wire["model_post_init"] = list(self.model_post_init)
        if self.training is not None:
            wire["training_config"] = self.training.to_wire()
        if self.sampling is not None:
            wire["inference_config"] = self.sampling.to_wire()
        if self.source_checkpoint_info is not None:
            wire["source_checkpoint_info"] = self.source_checkpoint_info
        return wire


# ─── Forward-backward payload helpers ────────────────────────────────────


def _load_torch():
    import torch

    return torch


def _tensor_dtype(torch, dtype_name: str | None):
    if dtype_name is None:
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
        return aliases[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype: {dtype_name}") from exc


def _json_tensor(value: Any, *, default_dtype: str | None = "long"):
    torch = _load_torch()
    if isinstance(value, dict):
        if "data" not in value:
            raise ValueError("tensor object must contain a data field")
        dtype_name = value.get("dtype", default_dtype)
        value = value["data"]
    else:
        dtype_name = default_dtype
    return torch.tensor(value, dtype=_tensor_dtype(torch, dtype_name))


def _read_fwd_bwd_args(payload: dict[str, Any]) -> tuple:
    args = payload.get("args", [])
    if args is None:
        return ()
    if not isinstance(args, list):
        raise ValueError("fwd-bwd payload args must be a JSON list")
    return tuple(args)


def _tensorize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and "data" in value:
            out[key] = _json_tensor(value)
        elif isinstance(value, list):
            out[key] = _json_tensor(value)
        else:
            out[key] = value
    return out


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


def _load_tokenizer(tokenizer_spec: Any):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No module named 'transformers'",
            name="transformers",
        ) from exc

    if isinstance(tokenizer_spec, str):
        model_name = tokenizer_spec
        kwargs: dict[str, Any] = {}
    elif isinstance(tokenizer_spec, dict):
        model_name = (
            tokenizer_spec.get("model_name")
            or tokenizer_spec.get("name")
            or tokenizer_spec.get("path")
        )
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("tokenizer.model_name is required")
        kwargs = {}
        for key in ("trust_remote_code", "use_fast", "revision"):
            if key in tokenizer_spec:
                kwargs[key] = tokenizer_spec[key]
    else:
        raise ValueError("fwd-bwd payload tokenizer must be a string or object")

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
    return tokenizer


def _tokenize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tokenizer_spec = payload.get("tokenizer", payload.get("model_name"))
    if tokenizer_spec is None:
        raise ValueError("fwd-bwd payload requires tokenizer or input_ids")

    tokenizer = _load_tokenizer(tokenizer_spec)
    texts = _repeat_texts(payload.get("texts"), payload.get("batch_size"))
    padding = payload.get("padding", "max_length")
    truncation = payload.get("truncation", True)
    add_special_tokens = payload.get("add_special_tokens", True)
    max_length = payload.get("max_length")
    if padding == "max_length":
        max_length = _positive_int(max_length, "fwd-bwd payload max_length")
    elif max_length is not None:
        max_length = _positive_int(max_length, "fwd-bwd payload max_length")
    _require_bool(truncation, "fwd-bwd payload truncation")
    _require_bool(add_special_tokens, "fwd-bwd payload add_special_tokens")

    encode_kwargs = {
        "return_tensors": "pt",
        "padding": padding,
        "truncation": truncation,
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        encode_kwargs["max_length"] = max_length
    encoded = tokenizer(texts, **encode_kwargs)
    if "input_ids" not in encoded:
        raise ValueError("tokenizer output did not include input_ids")
    return encoded


def _ensure_rank2(tensor, name: str) -> None:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} must have shape [batch, seq_len]")


def _build_position_ids(input_ids, position_spec: Any):
    torch = _load_torch()
    if position_spec is None or position_spec is False:
        return None
    if position_spec is True or position_spec == "arange":
        batch_size, seq_len = input_ids.shape
        return (
            torch.arange(seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
    if isinstance(position_spec, (dict, list)):
        return _json_tensor(position_spec)
    raise ValueError("fwd-bwd payload position_ids must be true, false, arange, or tensor data")


def _label_config(payload: dict[str, Any]) -> tuple[Any, int, bool]:
    labels = payload.get("labels", payload.get("label_strategy", "next_token"))
    if isinstance(labels, dict) and "data" in labels:
        return labels, int(payload.get("ignore_index", -100)), True
    if isinstance(labels, dict):
        strategy = labels.get("strategy", "next_token")
        ignore_index = int(labels.get("ignore_index", payload.get("ignore_index", -100)))
        mask_padding = labels.get("mask_padding", payload.get("mask_padding", True))
        return strategy, ignore_index, _require_bool(mask_padding, "labels.mask_padding")
    return labels, int(payload.get("ignore_index", -100)), _require_bool(
        payload.get("mask_padding", True),
        "fwd-bwd payload mask_padding",
    )


def _build_labels(input_ids, attention_mask, payload: dict[str, Any]):
    torch = _load_torch()
    labels, ignore_index, mask_padding = _label_config(payload)
    if labels is None or labels == "none":
        return None
    if isinstance(labels, (dict, list)):
        return _json_tensor(labels)
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
    raise ValueError(
        "fwd-bwd labels strategy must be next_token, input_ids, none, or tensor data"
    )


def build_forward_backward_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Build tensor kwargs for a forward-backward training batch.

    ``payload`` is intentionally JSON-shaped so CLI examples can stay readable:
    callers may provide either tokenizable ``texts`` plus a ``tokenizer`` or
    direct tensor data under ``input_ids`` / ``labels`` / ``position_ids``.
    """
    if not isinstance(payload, dict):
        raise ValueError("fwd-bwd payload must be a JSON object")

    if "kwargs" in payload:
        kwargs = payload["kwargs"]
        if not isinstance(kwargs, dict):
            raise ValueError("fwd-bwd payload kwargs must be an object")
        return _tensorize_kwargs(kwargs)

    torch = _load_torch()
    encoded = None
    if "input_ids" in payload:
        input_ids = _json_tensor(payload["input_ids"])
        attention_mask = (
            _json_tensor(payload["attention_mask"])
            if isinstance(payload.get("attention_mask"), (dict, list))
            else None
        )
    else:
        encoded = _tokenize_payload(payload)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    _ensure_rank2(input_ids, "input_ids")
    if attention_mask is not None:
        _ensure_rank2(attention_mask, "attention_mask")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

    kwargs: dict[str, Any] = {"input_ids": input_ids.contiguous()}

    include_attention = payload.get("include_attention_mask", False)
    if "attention_mask" in payload and isinstance(payload["attention_mask"], bool):
        include_attention = payload["attention_mask"]
    include_attention = _require_bool(
        include_attention,
        "fwd-bwd payload include_attention_mask",
    )
    if include_attention:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        kwargs["attention_mask"] = attention_mask.contiguous()

    position_spec = payload.get("position_ids")
    if position_spec is None and payload.get("include_position_ids", False):
        position_spec = "arange"
    position_ids = _build_position_ids(input_ids, position_spec)
    if position_ids is not None:
        _ensure_rank2(position_ids, "position_ids")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")
        kwargs["position_ids"] = position_ids.contiguous()

    labels = _build_labels(input_ids, attention_mask, payload)
    if labels is not None:
        _ensure_rank2(labels, "labels")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        kwargs["labels"] = labels.contiguous()

    return kwargs


def serialize_forward_backward_args(args: tuple = (), kwargs: dict[str, Any] | None = None) -> bytes:
    """Serialize ``{"args": args, "kwargs": kwargs}`` for Neutrino fwd-bwd."""
    if not isinstance(args, tuple):
        raise ValueError("forward-backward args must be a tuple")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("forward-backward kwargs must be a dict")

    return wire.dumps(
        {"args": args, "kwargs": kwargs},
        metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
    )


def build_forward_backward_payload(spec: dict[str, Any]) -> bytes:
    """Build a serialized fwd-bwd byte payload from readable JSON."""
    if not isinstance(spec, dict):
        raise ValueError("fwd-bwd JSON must be an object")
    payload = spec.get("payload", spec)
    if not isinstance(payload, dict):
        raise ValueError("fwd-bwd payload must be an object")
    args = _read_fwd_bwd_args(payload)
    kwargs = build_forward_backward_kwargs(payload)
    return serialize_forward_backward_args(args, kwargs)


def _parse_s3_stage_credentials(raw_value: Any) -> dict[str, str]:
    stage = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    if not isinstance(stage, dict):
        raise ValueError("stage credentials response is not a JSON object")
    if (stage.get("locationType") or "").upper() != "S3":
        raise NotImplementedError(
            f"execution log download only supports S3 stages; got "
            f"{stage.get('locationType')!r}"
        )
    location = (stage.get("location") or "").removeprefix("s3://").strip("/")
    bucket, _, prefix = location.partition("/")
    if not bucket:
        raise ValueError(f"stage credentials missing bucket: {stage.get('location')!r}")
    creds = stage.get("creds")
    if not isinstance(creds, dict):
        raise ValueError("stage credentials missing AWS creds")
    try:
        return {
            "bucket": bucket,
            "prefix": prefix,
            "region": stage.get("region") or "",
            "access_key_id": creds["AWS_KEY_ID"],
            "secret_access_key": creds["AWS_SECRET_KEY"],
            "session_token": creds.get("AWS_TOKEN"),
        }
    except KeyError as exc:
        raise ValueError(f"stage credentials missing AWS field: {exc.args[0]}") from exc


# ─── HTTP client ─────────────────────────────────────────────────────────


class NeutrinoClient:
    """HTTP client for the Neutrino GS API (``cortex-training``).

    Construct with :meth:`from_pat`:

        client = NeutrinoClient.from_pat(
            host="dsa_test.qa6.us-west-2.aws.snowflakecomputing.com",
            pat="...",
            database="DSA_TEST_DB",
            schema="PUBLIC",
        )

    For local development, point ``base_url`` at the mock SnowAPI server::

        cd cortex/neutrino && make mock-snowapi   # starts on :8084
        client = NeutrinoClient(
            base_url="http://localhost:8084",
            database="mydb",
            schema="public",
        )

    Then build one or more sub-jobs and submit::

        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw", "lr": 1e-5},
            max_seq_len=2048,
            train_batch_size=1,
        )
        job_id = client.create_job(sub_jobs=[sub])
    """

    def __init__(
        self,
        base_url: str,
        database: str,
        schema: str,
        endpoint: str = "cortex-training",
        poll_interval: float = 0.5,
        poll_timeout: float = 1800.0,
        poll_backoff_multiplier: float = 1.25,
        poll_max_interval: float = 6.0,
        pool_maxsize: int = 1024,
        max_retries: int = 10,
    ):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if poll_timeout <= 0:
            raise ValueError("poll_timeout must be > 0")
        if poll_backoff_multiplier < 1.0:
            raise ValueError("poll_backoff_multiplier must be >= 1.0")
        if poll_max_interval < poll_interval:
            raise ValueError("poll_max_interval must be >= poll_interval")
        if pool_maxsize <= 0:
            raise ValueError("pool_maxsize must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.base_url = base_url.rstrip("/")
        self.database = database
        self.schema = schema
        self.endpoint = endpoint
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.poll_backoff_multiplier = poll_backoff_multiplier
        self.poll_max_interval = poll_max_interval
        self.max_retries = max_retries
        self._fwd_bwd_send_count = 0
        self._fwd_bwd_request_debug: dict[str, dict[str, Any]] = {}
        self._generate_request_ids: set[str] = set()
        # Per-job cache of the sampling sub-job's ``inference_config.max_seq_len``
        # (the value the backend launches vLLM with as ``max_model_len``). Used
        # to validate generate prompt lengths client-side. A cached ``None``
        # means the job was fetched but exposes no sampling max_seq_len (skip
        # validation); a missing key means it has not been resolved yet (e.g. a
        # transient get_job failure), so a later call will retry.
        self._sampling_max_seq_len: dict[str, int | None] = {}
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @classmethod
    def from_pat(
        cls,
        host: str,
        pat: str,
        database: str,
        schema: str,
        endpoint: str = "cortex-training",
        verify_ssl: bool = True,
        **kwargs: Any,
    ) -> "NeutrinoClient":
        """Authenticate with a Programmatic Access Token (PAT).

        Sends ``Authorization: Bearer <pat>`` plus the
        ``X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN`` header
        on every request. No login round-trip.
        """
        client = cls(
            base_url=f"https://{host}",
            database=database,
            schema=schema,
            endpoint=endpoint,
            **kwargs,
        )
        client._session.headers["Authorization"] = f"Bearer {pat}"
        client._session.headers["X-Snowflake-Authorization-Token-Type"] = "PROGRAMMATIC_ACCESS_TOKEN"
        client._session.verify = verify_ssl
        return client

    @property
    def _prefix(self) -> str:
        return (
            f"{self.base_url}/api/v2/databases/{self.database}"
            f"/schemas/{self.schema}/{self.endpoint}"
        )

    @staticmethod
    def _debug_json_summary(value: Any, *, limit: int = 2000) -> str:
        try:
            text = json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            text = repr(value)
        if len(text) > limit:
            return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
        return text

    @classmethod
    def _debug_response_summary(cls, resp: requests.Response, *, limit: int = 2000) -> str:
        try:
            return cls._debug_json_summary(resp.json(), limit=limit)
        except ValueError:
            text = getattr(resp, "text", "")
        except Exception as exc:
            text = f"<failed to read response body: {type(exc).__name__}: {exc}>"
        text = str(text).replace("\n", "\\n")
        if len(text) > limit:
            return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
        return text

    @staticmethod
    def _debug_context_label(debug_context: dict[str, Any]) -> str:
        parts = ["[neutrino fwd-bwd]"]
        for key in ("phase", "call", "job_id", "request_id", "payload_bytes"):
            value = debug_context.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def _send(self, method: str, url: str, *, retry_on=_is_transient, **kwargs) -> requests.Response:
        fn = getattr(self._session, method.lower())
        debug_context = kwargs.pop("debug_context", None)
        debug_label = (
            self._debug_context_label(debug_context) if debug_context is not None else None
        )
        attempt_no = 0
        max_attempts = 1 + self.max_retries

        def attempt() -> requests.Response:
            nonlocal attempt_no
            attempt_no += 1
            if debug_label is not None:
                print(
                    f"{debug_label} sending {method.upper()} {url} "
                    f"attempt={attempt_no}/{max_attempts}",
                    flush=True,
                )
            try:
                resp = fn(url, **kwargs)
            except Exception as exc:
                if debug_label is not None:
                    print(
                        f"{debug_label} request exception {method.upper()} {url} "
                        f"attempt={attempt_no}/{max_attempts}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                raise
            status_code = getattr(resp, "status_code", None)
            headers = getattr(resp, "headers", {}) or {}
            sf_request_id = headers.get("x-snowflake-request-id")
            if not isinstance(sf_request_id, str) or not sf_request_id:
                sf_request_id = None
            if sf_request_id:
                req = getattr(resp, "request", None)
                logger.info(
                    "snowflake request_id=%s  %s %s  status=%d",
                    sf_request_id,
                    getattr(req, "method", method.upper()),
                    getattr(req, "path_url", url),
                    status_code,
                )
            if debug_label is not None:
                try:
                    status_int = int(status_code)
                except (TypeError, ValueError):
                    status_int = None
                outcome = (
                    "successful response"
                    if status_int is not None and 200 <= status_int < 400
                    else "failed response"
                )
                snowflake = f" snowflake_request_id={sf_request_id}" if sf_request_id else ""
                print(
                    f"{debug_label} {outcome} {method.upper()} {url} "
                    f"attempt={attempt_no}/{max_attempts} status={status_code}"
                    f"{snowflake} body={self._debug_response_summary(resp)}",
                    flush=True,
                )
            resp.raise_for_status()
            return resp

        retryer = Retrying(
            retry=retry_if_exception(retry_on),
            stop=stop_after_attempt(1 + self.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=10.0),
            reraise=True,
        )
        return retryer(attempt)

    # ─── Job management ──────────────────────────────────────────────────

    def create_job(
        self,
        sub_jobs: list[SubJobConfig],
        job_id: str | None = None,
        experiment_name: str | None = None,
    ) -> str:
        """Create a job from a list of sub-jobs and return its server job_id.

        Each :class:`SubJobConfig` is validated client-side before the request
        is sent (see :meth:`SubJobConfig.validate`). ``job_id`` is optional;
        when omitted the server generates one. ``experiment_name`` is optional;
        when omitted the server auto-creates an experiment for the job.
        """
        if not sub_jobs:
            raise ValueError("create_job requires a non-empty sub_jobs list")
        for sj in sub_jobs:
            sj.validate()
        body: dict = {"sub_job_configs": [sj.to_wire() for sj in sub_jobs]}
        if job_id is not None:
            body["job_id"] = job_id
        if experiment_name is not None:
            body["experiment_name"] = experiment_name
        return self.create_job_from_body(body)["job_id"]

    def create_job_from_body(self, body: dict) -> dict:
        """Create a job from a raw REST CreateJob request body.

        This is useful for tooling that already has the SNOWAPI JSON payload,
        while :meth:`create_job` remains the typed path for Python callers.
        """
        if not isinstance(body, dict):
            raise ValueError("create_job_from_body requires a JSON object")
        sub_job_configs = body.get("sub_job_configs")
        if not isinstance(sub_job_configs, list) or not sub_job_configs:
            raise ValueError(
                "create_job_from_body requires a non-empty sub_job_configs list"
            )
        if body.get("debug") and not _debug_options_enabled():
            raise ValueError(
                "create-job debug options are an internal-only capability; set "
                f"{DEBUG_OPTIONS_ENV}=1 to send a request with a `debug` block"
            )
        resp = self._send(
            "POST", self._prefix, json=body, retry_on=_is_connect_error
        )
        return resp.json()

    @staticmethod
    def _normalize_job_status(status) -> str:
        """Normalize full enum names (``JOB_STATE_RUNNING``) to short form (``running``).

        Coerces to ``str`` first: a Control Plane running ahead of Global
        Services can return a JobState enum value GS doesn't yet know (e.g. a
        newly added state), which GS renders as a raw integer rather than a
        name. Tolerating it keeps :meth:`wait_for_job` polling — an unknown
        status is neither ``running`` nor terminal, so it stays in-progress —
        instead of crashing on ``int.lower()``.
        """
        return str(status or "").lower().removeprefix("job_state_")

    def _sleep_with_backoff(self, delay: float, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return delay
        time.sleep(min(delay, remaining))
        return min(delay * self.poll_backoff_multiplier, self.poll_max_interval)

    def wait_for_job(self, job_id: str) -> dict:
        """Poll until the job reaches ``running``. Raises on terminal states or timeout."""
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            status = self._normalize_job_status(job.get("status", ""))
            if status == "running":
                return job
            if status in ("failed", "done", "cancelled", "canceled"):
                reason = job.get("reason", "")
                raise RuntimeError(
                    f"Job {job_id} reached terminal state '{status}': {reason}"
                )
            delay = self._sleep_with_backoff(delay, deadline)
        raise TimeoutError(
            f"Job {job_id} did not become running within {self.poll_timeout}s"
        )

    def get_job(self, job_id: str) -> dict:
        resp = self._send("GET", f"{self._prefix}/{job_id}")
        return resp.json()

    def list_jobs(self, status: str | None = None) -> list:
        params = {}
        if status is not None:
            params["status"] = status
        resp = self._send("GET", self._prefix, params=params)
        return resp.json().get("jobs", [])

    def cancel_job(self, job_id: str) -> None:
        # GS uses colon-action syntax: /{jobId}:cancel
        self._send("POST", f"{self._prefix}/{job_id}:cancel")

    def get_capacity(self) -> dict:
        """Return the calling account's reserved GPU capacity and current usage.

        Backed by the account-scoped GS endpoint ``/cortex-training/capacity``
        (not under ``/{job_id}``). The account is resolved server-side from the
        caller's session — never from a request field — so a caller can only
        ever read its own account's capacity.

        The returned dict always carries all four fields. The server emits
        proto3 JSON, which omits zero/false fields (an unreserved account's
        response is literally ``{}``), so we fill in the documented defaults:

        - ``has_reservation`` (bool): whether the account has a configured GPU
          reservation. When ``False`` the account uses shared/on-demand
          placement and the ``*_gpus`` fields are all 0.
        - ``reserved_gpus`` (int): total GPUs reserved for the account.
        - ``in_use_gpus`` (int): GPUs consumed by the account's RUNNING +
          PLACING jobs.
        - ``available_gpus`` (int): ``reserved_gpus - in_use_gpus``, floored at 0.
        """
        resp = self._send("GET", f"{self._prefix}/capacity")
        body = resp.json()
        return {
            "has_reservation": bool(body.get("has_reservation", False)),
            "reserved_gpus": int(body.get("reserved_gpus", 0)),
            "in_use_gpus": int(body.get("in_use_gpus", 0)),
            "available_gpus": int(body.get("available_gpus", 0)),
        }

    def get_experiment_run(self, job_id: str) -> dict[str, str]:
        """Return ``{experiment_name, experiment_run_name}`` for ``job_id``.

        Backed by the GS endpoint ``/cortex-training/{job_id}/experiment-run``.
        """
        resp = self._send("GET", f"{self._prefix}/{job_id}/experiment-run")
        return resp.json()

    def _query_sql_scalar(self, statement: str) -> Any:
        """Execute a synchronous SQL statement and return ``data[0][0]``.

        Used for ``SYSTEM$`` scalar functions (single JSON-encoded value).
        """
        resp = self._send(
            "POST",
            f"{self.base_url}/api/v2/statements",
            json={"statement": statement, "database": self.database, "schema": self.schema},
        )
        rows = resp.json().get("data")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], list) or not rows[0]:
            raise ValueError(f"SQL query returned no rows: {statement}")
        return rows[0][0]

    def fetch_execution_logs(self, job_id: str) -> list[dict[str, str]]:
        """Download every log file under the job's experiment run stage.

        Returns a list of ``{"sub_job_id", "filename", "s3_uri", "content"}``
        entries, one per object found under a ``_logs/<sub_job_id>/`` subtree
        in the run's stage. Filenames such as ``execution.jsonl`` and
        ``server.log`` are both included; bodies are decoded as UTF-8.

        Steps: ``experiment-run`` → ``SYSTEM$GET_VSTAGE_WRITE_CREDS`` →
        ``ListObjectsV2`` (scoped by stage prefix) → ``GetObject`` per match.
        """
        run = self.get_experiment_run(job_id)
        try:
            experiment_name = run["experiment_name"]
            run_name = run["experiment_run_name"]
        except KeyError as exc:
            raise ValueError(
                f"experiment-run response missing field: {exc.args[0]}"
            ) from exc

        run_uri = f"snow://experiment/{experiment_name}/versions/{run_name}/"
        creds = _parse_s3_stage_credentials(
            self._query_sql_scalar(f"SELECT SYSTEM$GET_VSTAGE_WRITE_CREDS('{run_uri}')")
        )
        bucket = creds["bucket"]
        stage_prefix = creds["prefix"]
        if stage_prefix:
            stage_prefix += "/"

        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=creds["access_key_id"],
            aws_secret_access_key=creds["secret_access_key"],
            aws_session_token=creds["session_token"],
            region_name=creds["region"] or None,
        )

        results: list[dict[str, str]] = []
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=stage_prefix
        ):
            for item in page.get("Contents") or []:
                key = item.get("Key")
                if not isinstance(key, str):
                    continue
                _, sep, after_logs = key.partition("/_logs/")
                if not sep:
                    continue
                sub_job_id, _, filename = after_logs.partition("/")
                if not sub_job_id or not filename:
                    continue
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                results.append(
                    {
                        "sub_job_id": sub_job_id,
                        "filename": filename,
                        "s3_uri": f"s3://{bucket}/{key}",
                        "content": body.decode("utf-8"),
                    }
                )
        return results

    # ─── Data-plane async operations ─────────────────────────────────────

    # Maximum payload size for forward-backward requests.
    # Payloads beyond this limit will fail fast with a message.
    _MAX_FWD_BWD_BYTES = 60 * 1024 * 1024  # 60 MB

    # Maximum payload size for generate / generate_stream request bodies.
    # The encoded JSON body must stay under this cap; oversized requests fail
    # fast client-side with a clear message.
    _MAX_GENERATE_BYTES = 60 * 1024 * 1024  # 60 MB

    def _check_generate_payload(self, op: str, body: bytes) -> None:
        if len(body) > self._MAX_GENERATE_BYTES:
            size_mb = len(body) / (1024 * 1024)
            raise ValueError(
                f"{op} payload is {size_mb:.1f} MB, which exceeds the "
                f"maximum allowed size of "
                f"{self._MAX_GENERATE_BYTES / (1024 * 1024):.0f} MB. "
                f"Reduce the number of prompts or shorten each prompt to "
                f"bring the serialized request under the limit."
            )

    def _resolve_sampling_max_seq_len(self, job_id: str) -> int | None:
        """Return the sampling sub-job's ``max_seq_len``, or ``None`` if unknown.

        The value is read from the job's sampling ``inference_config`` (the same
        number the backend launches vLLM with as ``max_model_len``) and cached
        per ``job_id``. A successful lookup — including a definitive "no sampling
        sub-job" miss — is cached; a transient ``get_job`` failure is not, so a
        later call retries rather than disabling validation permanently.
        """
        if job_id in self._sampling_max_seq_len:
            return self._sampling_max_seq_len[job_id]
        try:
            job = self.get_job(job_id)
        except Exception:
            # Don't let a transient lookup failure break generate, and don't
            # cache the miss — the real request will surface any hard error.
            return None
        max_seq_len: int | None = None
        for sub in job.get("sub_jobs", []) or []:
            if not isinstance(sub, dict):
                continue
            cfg = sub.get("inference_config")
            if not isinstance(cfg, dict):
                continue
            raw = cfg.get("max_seq_len")
            # Proto numbers can arrive as floats (e.g. 2048.0); coerce to int.
            if isinstance(raw, (int, float)) and raw > 0:
                max_seq_len = int(raw)
                break
        self._sampling_max_seq_len[job_id] = max_seq_len
        return max_seq_len

    def _check_prompt_lengths(
        self, op: str, job_id: str, prompts: list[str | list[int]]
    ) -> None:
        """Fail fast when a pre-tokenized prompt cannot fit the sampling window.

        Mirrors vLLM's decoder-prompt check (``vllm/v1/engine/input_processor``):
        a generation prompt must be strictly shorter than ``max_model_len`` so
        at least one output token fits, i.e. it is rejected when
        ``len(prompt_token_ids) >= max_seq_len``. Only pre-tokenized prompts
        (``list[int]``) are validated here, since their token count is exact;
        string prompts would require replicating the server tokenizer and are
        left to fail server-side.
        """
        if not any(isinstance(p, list) for p in prompts):
            return
        max_seq_len = self._resolve_sampling_max_seq_len(job_id)
        if max_seq_len is None:
            return
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, list):
                continue
            n = len(prompt)
            if n >= max_seq_len:
                raise ValueError(
                    f"{op} prompt at index {i} has {n} tokens, which does not "
                    f"fit the sampling job's max_seq_len of {max_seq_len}: a "
                    f"prompt must be shorter than max_seq_len to leave room for "
                    f"at least one generated token (vLLM rejects "
                    f"len(prompt) >= max_model_len). Shorten the prompt to at "
                    f"most {max_seq_len - 1} tokens, or recreate the job with a "
                    f"larger max_seq_len."
                )

    def _post_octet_request_chunks(
        self,
        *,
        job_id: str,
        path_suffix: str,
        operation: str,
        frame: bytes,
        max_bytes: int,
        debug_context: dict[str, Any] | None = None,
    ) -> dict:
        chunks = wire.encode_byte_chunks(
            frame,
            kind="request",
            operation=operation,
            max_bytes=max_bytes,
        )
        final_body: dict | None = None
        for idx, chunk in enumerate(chunks):
            resp = self._send(
                "POST",
                f"{self._prefix}/{job_id}/{path_suffix}",
                data=chunk,
                headers={"Content-Type": "application/octet-stream"},
                debug_context=debug_context,
            )
            body = resp.json()
            if idx < len(chunks) - 1:
                if isinstance(body, dict) and body.get("request_id"):
                    raise RuntimeError(
                        f"{operation} chunk {idx} unexpectedly returned request_id"
                    )
                continue
            final_body = body
        if final_body is None:
            raise RuntimeError(f"{operation} produced no request body")
        return final_body

    def forward_backward(self, job_id: str, data: bytes) -> str:
        """Submit a forward+backward pass. Returns request_id."""
        self._fwd_bwd_send_count += 1
        call = self._fwd_bwd_send_count
        debug_context = {
            "phase": "submit",
            "call": call,
            "job_id": job_id,
            "payload_bytes": len(data),
        }
        debug_label = self._debug_context_label(debug_context)
        print(f"{debug_label} payload total bytes: {len(data)}", flush=True)
        body = self._post_octet_request_chunks(
            job_id=job_id,
            path_suffix="forward-backward",
            operation="fwd-bwd",
            frame=data,
            max_bytes=self._MAX_FWD_BWD_BYTES,
            debug_context=debug_context,
        )
        request_id = body["request_id"]
        request_debug_context = {
            "phase": "poll",
            "call": call,
            "job_id": job_id,
            "request_id": request_id,
            "payload_bytes": len(data),
        }
        self._fwd_bwd_request_debug[request_id] = request_debug_context
        print(
            f"{debug_label} submitted request_id={request_id} "
            f"body={self._debug_json_summary(body)}",
            flush=True,
        )
        return request_id

    def generate(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> str:
        """Submit a generate request to a sampling sub-job. Returns request_id.

        The typed fields are serialized as a DSSST1 safetensors frame and sent
        as ``application/octet-stream``. Oversized frames are byte-chunked
        across multiple POSTs; the final POST returns the pollable request id.
        Poll with :meth:`poll_request`.

        Prompt-length validation is asymmetric by input type:

        * **Pre-tokenized prompts** (``list[int]``) are checked client-side
          against the sampling sub-job's ``max_seq_len`` and fail fast with a
          ``ValueError`` when ``len(prompt) >= max_seq_len`` — the token count
          is exact, so this never rejects a prompt vLLM would accept.
        * **String prompts** are *not* validated here. Counting their tokens
          would require replicating the server tokenizer (including special
          tokens), so an over-long string instead fails server-side in vLLM
          with a ``maximum model length`` error rather than client-side.
        """
        payload: dict = {"prompts": prompts}
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params
        if routing_key is not None:
            payload["routing_key"] = routing_key
        if strict is not None:
            payload["strict"] = strict
        self._check_prompt_lengths("generate", job_id, prompts)
        body = wire.dumps(
            payload,
            metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
        )
        response = self._post_octet_request_chunks(
            job_id=job_id,
            path_suffix="generate",
            operation="generate",
            frame=body,
            max_bytes=self._MAX_GENERATE_BYTES,
        )
        request_id = response["request_id"]
        self._generate_request_ids.add(request_id)
        return request_id

    def generate_stream(
        self,
        job_id: str,
        prompts: list[str | list[int]],
        sampling_params: dict | list[dict | None] | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool | None = None,
    ) -> dict:
        """Start a streaming generate request. Returns the response body.

        Progress is read with :meth:`get_request_status` using the returned
        ``request_id``. Cancellation uses :meth:`cancel_request`.

        Prompt-length validation matches :meth:`generate`: pre-tokenized
        (``list[int]``) prompts are checked client-side against the sampling
        sub-job's ``max_seq_len`` and fail fast, while over-long string
        prompts are left to fail server-side in vLLM (see :meth:`generate`).
        """
        payload: dict = {"prompts": prompts}
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params
        if routing_key is not None:
            payload["routing_key"] = routing_key
        if strict is not None:
            payload["strict"] = strict
        self._check_prompt_lengths("generate_stream", job_id, prompts)
        body = json.dumps(payload).encode("utf-8")
        self._check_generate_payload("generate_stream", body)
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/generate-stream",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        return resp.json()

    def step(self, job_id: str, learning_rate: float | None = None) -> str:
        """Submit an optimizer step. Returns request_id."""
        body: dict = {}
        if learning_rate is not None:
            body["learning_rate"] = learning_rate
        resp = self._send("POST", f"{self._prefix}/{job_id}/step", json=body)
        return resp.json()["request_id"]

    def save(
        self,
        job_id: str,
        checkpoint_id: str | None = None,
        checkpoint_type: str | None = None,
    ) -> str:
        """Submit a checkpoint save. Returns request_id."""
        body: dict = {}
        if checkpoint_id is not None:
            body["checkpoint_id"] = checkpoint_id
        if checkpoint_type is not None:
            normalized_type = checkpoint_type.lower()
            if normalized_type not in ("resumable", "weights-only"):
                raise ValueError(
                    "checkpoint_type must be 'resumable' or 'weights-only'"
                )
            body["checkpoint_type"] = normalized_type
        resp = self._send("POST", f"{self._prefix}/{job_id}/save", json=body)
        return resp.json()["request_id"]

    def load(
        self,
        job_id: str,
        checkpoint_id: str,
        source_job_id: str | None = None,
    ) -> str:
        """Submit a checkpoint load into an existing job. Returns request_id."""
        body: dict = {"checkpoint_id": checkpoint_id}
        if source_job_id is not None:
            body["source_job_id"] = source_job_id
        resp = self._send("POST", f"{self._prefix}/{job_id}/load", json=body)
        return resp.json()["request_id"]

    def _operation(
        self,
        job_id: str,
        operation_type: str,
        *,
        payload: dict | bytes | bytearray | memoryview | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Submit a generic data-plane operation through GS/Control Plane."""
        body: dict = {"operation_type": operation_type}
        if sub_job_id is not None:
            body["sub_job_id"] = sub_job_id
        if sub_job_type is not None:
            body["sub_job_type"] = sub_job_type
        if isinstance(payload, (bytes, bytearray, memoryview)):
            payload = {
                "payload_b64": base64.b64encode(bytes(payload)).decode("ascii"),
                "content_type": "application/octet-stream",
            }
        if payload is not None:
            body["payload"] = payload
        print(f"***** POST operation {self._prefix}/{job_id}/operation {body=}")
        resp = self._send("POST", f"{self._prefix}/{job_id}/operation", json=body)
        return resp.json()

    def forward(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Submit a generic forward operation. Returns the response body."""
        if isinstance(payload, (bytes, bytearray, memoryview)) and len(payload) > self._MAX_FWD_BWD_BYTES:
            size_mb = len(payload) / (1024 * 1024)
            raise ValueError(
                f"forward payload is {size_mb:.1f} MB, which exceeds the "
                f"maximum allowed size of "
                f"{self._MAX_FWD_BWD_BYTES / (1024 * 1024):.0f} MB. "
                f"Reduce the batch size or sequence length in your training "
                f"configuration to bring the serialized input batch under "
                f"the limit."
            )
        return self._operation(
            job_id,
            "forward",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def fwd(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(
            job_id,
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def fwd_no_grad(
        self,
        job_id: str,
        payload: dict | bytes | bytearray | memoryview | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(
            job_id,
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def weight_sync(
        self,
        job_id: str,
        source_sub_job_id: str,
        target_sub_job_ids: list[str],
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> str:
        """Sync model weights from a training sub-job into one or more sampling
        sub-jobs (used in RL training loops). Returns request_id.

        Multi-sub-job sessions require the operation envelope to include a
        routing hint. By default, route through the source training sub-job.

        ***** POST operation 
        https://bbb39214.snowflakecomputing.com/api/v2/databases/neutrino_db/schemas/neutrino_schema/cortex-training/b1fcb345-de6f-40b8-a6b8-c4b4fd02dbec/operation 

        body={'operation_type': 'weight-sync',
              'sub_job_id': 'b1fcb345-de6f-40b8-a6b8-c4b4fd02dbec:training:0', 
              'payload': {
                  'source_sub_job_id': 'b1fcb345-de6f-40b8-a6b8-c4b4fd02dbec:training:0', 
                  'target_sub_job_ids': ['b1fcb345-de6f-40b8-a6b8-c4b4fd02dbec:sampling:0']
              }
             }

            request_id = client.weight_sync(
                self.training_job_id,
                source_sub_job_id=self._training_sub_job_id,
                target_sub_job_ids=[self._sampling_sub_job_id],
            )
        """
        body = {
            "source_sub_job_id": source_sub_job_id, # training
            "target_sub_job_ids": list(target_sub_job_ids), # sampling
        }
        return self._operation(
            job_id,
            "weight-sync",
            payload=body,
            sub_job_id=sub_job_id or source_sub_job_id, # training
            sub_job_type=(
                sub_job_type if sub_job_type is not None else "training"
            ),
        )["request_id"]

    def bootstrap_router_replay(
        self,
        job_id: str,
        source_sub_job_id: str,
        target_sub_job_id: str,
        *,
        max_cache_bytes: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Bootstrap router replay from one sub-job to another."""
        payload: dict = {
            "source_sub_job_id": source_sub_job_id,
            "target_sub_job_id": target_sub_job_id,
        }
        if max_cache_bytes is not None:
            payload["max_cache_bytes"] = max_cache_bytes
        return self._operation(
            job_id,
            "bootstrap-router-replay",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def router_replay_discard(
        self,
        job_id: str,
        sample_ids: list[str] | None = None,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Discard router replay samples on a routed sub-job."""
        payload = dict(extra or {})
        payload.setdefault("sample_ids", list(sample_ids or []))
        return self._operation(
            job_id,
            "router-replay-discard",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def reset_prefix_cache(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = "sampling",
        drain: bool = True,
        timeout_s: float = 60.0,
        retry_interval_s: float = 0.1,
        extra: dict | None = None,
    ) -> dict:
        """Reset the sampling prefix cache for a Neutrino job.

        Operation-specific fields are carried inside the opaque operation
        payload. ``extra`` allows new reset options to flow through without a
        dss-client release.
        """
        payload = dict(extra or {})
        payload.setdefault("drain", drain)
        payload.setdefault("timeout_s", timeout_s)
        payload.setdefault("retry_interval_s", retry_interval_s)
        return self._operation(
            job_id,
            "reset-prefix-cache",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    # ─── Async request polling ────────────────────────────────────────────

    @staticmethod
    def _normalize_request_status(status: str) -> str:
        """Normalize full enum names (``REQUEST_STATE_DONE``) to short form (``done``)."""
        return status.lower().removeprefix("request_state_")

    @staticmethod
    def _decode_result_payload(result: dict) -> dict | None:
        if not isinstance(result, dict):
            return None
        if result.get("wire_format") != wire.WIRE_FORMAT_VERSION:
            return None
        if result.get("encoding") != "base64":
            raise RuntimeError("DSSST1 result payload must use base64 encoding")
        raw = result.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            raise RuntimeError("DSSST1 result payload missing payload_b64")
        payload = base64.b64decode(raw)
        decoded = wire.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError("DSSST1 result payload did not decode to a dict")
        return decoded

    @staticmethod
    def _decode_result_chunk_event(event: dict) -> bytes | None:
        if not isinstance(event, dict) or event.get("type") != "result_chunk":
            return None
        raw = event.get("payload_b64")
        if not isinstance(raw, str) or not raw:
            raise RuntimeError("result_chunk event missing payload_b64")
        payload = base64.b64decode(raw)
        expected = event.get("payload_sha256")
        if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError("result_chunk payload_sha256 mismatch")
        return payload

    @staticmethod
    def _restore_generate_result_lists(value: Any, torch_module: Any | None = None) -> Any:
        torch_module = _load_torch() if torch_module is None else torch_module
        if torch_module.is_tensor(value):
            return value.cpu().tolist()
        if isinstance(value, dict):
            return {
                key: NeutrinoClient._restore_generate_result_lists(item, torch_module)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                NeutrinoClient._restore_generate_result_lists(item, torch_module)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                NeutrinoClient._restore_generate_result_lists(item, torch_module)
                for item in value
            )
        return value

    def _normalize_generate_result_if_needed(self, request_id: str, result: dict) -> dict:
        if request_id not in self._generate_request_ids:
            return result
        self._generate_request_ids.discard(request_id)
        if isinstance(result, dict) and "results" in result:
            result = dict(result)
            result["results"] = self._restore_generate_result_lists(result["results"])
        return result

    @staticmethod
    def _decode_stream_result_event(event: Any) -> Any:
        if not isinstance(event, dict) or event.get("type") != "result":
            return event
        result = event.get("result")
        if not isinstance(result, dict):
            return event
        decoded = NeutrinoClient._decode_result_payload(result)
        if decoded is None:
            return event
        event = dict(event)
        event["result"] = NeutrinoClient._restore_generate_result_lists(decoded)
        return event

    @staticmethod
    def _decode_stream_result_events(status: dict) -> dict:
        events = status.get("events")
        if not isinstance(events, list):
            return status
        decoded = [NeutrinoClient._decode_stream_result_event(event) for event in events]
        if decoded == events:
            return status
        status = dict(status)
        status["events"] = decoded
        return status

    def poll_request(self, job_id: str, request_id: str) -> dict:
        """Poll until the async request completes. Returns the result dict."""
        deadline = time.monotonic() + self.poll_timeout
        delay = self.poll_interval
        debug_context = self._fwd_bwd_request_debug.get(request_id)
        debug_label = (
            self._debug_context_label(debug_context) if debug_context is not None else None
        )
        result_chunks: list[bytes] = []
        cursor: str | None = None
        while time.monotonic() < deadline:
            status = self.get_request_status(job_id, request_id, cursor=cursor)
            state = self._normalize_request_status(status.get("status", ""))
            events = status.get("events") or []
            received_chunk = False
            for event in events:
                chunk = self._decode_result_chunk_event(event)
                if chunk is not None:
                    result_chunks.append(chunk)
                    received_chunk = True
            next_cursor = status.get("next_cursor")
            if isinstance(next_cursor, str) and next_cursor:
                cursor = next_cursor
                # Drain available result chunks without backing off.
                continue
            if state in ("completed", "done", "succeeded"):
                if result_chunks:
                    result = wire.decode_result_chunks(result_chunks)
                else:
                    result = status.get("result") or {}
                    decoded = self._decode_result_payload(result)
                    if decoded is not None:
                        result = decoded
                result = self._normalize_generate_result_if_needed(request_id, result)
                if debug_label is not None:
                    print(
                        f"{debug_label} completed state={state} "
                        f"result={self._debug_json_summary(result)}",
                        flush=True,
                    )
                    self._fwd_bwd_request_debug.pop(request_id, None)
                return result
            if state in ("failed", "cancelled", "canceled"):
                error = status.get("error", "")
                if debug_label is not None:
                    print(
                        f"{debug_label} failed state={state} error={error} "
                        f"body={self._debug_json_summary(status)}",
                        flush=True,
                    )
                    self._fwd_bwd_request_debug.pop(request_id, None)
                self._generate_request_ids.discard(request_id)
                raise RuntimeError(
                    f"Request {request_id} ended with state '{state}': {error}"
                )
            if received_chunk:
                # Defensive: if the server returns a chunk without next_cursor
                # before terminal status, poll again immediately rather than
                # sleeping.
                continue
            delay = self._sleep_with_backoff(delay, deadline)
        if debug_label is not None:
            print(
                f"{debug_label} timed out after {self.poll_timeout}s",
                flush=True,
            )
            self._fwd_bwd_request_debug.pop(request_id, None)
        self._generate_request_ids.discard(request_id)
        raise TimeoutError(
            f"Request {request_id} did not complete within {self.poll_timeout}s"
        )

    def get_request_status(
        self,
        job_id: str,
        request_id: str,
        *,
        max_events: int | None = None,
        cursor: str | None = None,
    ) -> dict:
        params = {}
        if max_events is not None:
            params["max_events"] = max_events
        if cursor:
            params["cursor"] = cursor
        debug_context = self._fwd_bwd_request_debug.get(request_id)
        resp = self._send(
            "GET",
            f"{self._prefix}/{job_id}/requests/{request_id}",
            params=params or None,
            debug_context=debug_context,
        )
        return self._decode_stream_result_events(resp.json())

    def cancel_request(
        self,
        job_id: str,
        request_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        return self._operation(
            job_id,
            "cancel-request",
            payload={"request_id": request_id},
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    # ─── Logs (read-only) ────────────────────────────────────────────────

    def tail_logs(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_lines: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Fetch one page of a sub-job's logs from ``cursor`` (read-only).

        The target is the sub-job's zone-manager (Ray head) pod, whose stdout
        already includes worker output (Ray ``log_to_driver``); identify it via
        ``sub_job_id`` (or ``sub_job_type`` when unambiguous). Returns
        ``{"entries": [...], "next_cursor": str, "eof": bool}``. Re-call with
        ``next_cursor`` to continue; an empty ``cursor`` reads from the start.
        """
        payload: dict = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_lines is not None:
            payload["max_lines"] = max_lines
        return self._operation(
            job_id,
            "tail-logs",
            payload=payload,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def _stream_pages(
        self,
        fetch,
        entries_key: str,
        *,
        follow: bool,
        cursor: str | None = None,
        poll_interval: float | None = None,
    ) -> Iterator[dict]:
        """Drive a cursor-paged tail: call ``fetch(cursor)`` for a page, yield
        its entries (under ``entries_key``), advance the cursor, and poll with
        backoff. With ``follow`` False, stop once a page reports ``eof``.
        """
        base_delay = poll_interval or self.poll_interval
        delay = base_delay
        while True:
            page = fetch(cursor)
            cursor = page.get("next_cursor", cursor)
            entries = page.get(entries_key) or []
            for entry in entries:
                yield entry
            if entries:
                delay = base_delay  # made progress; poll again promptly
                continue
            if page.get("eof", False) and not follow:
                return
            time.sleep(delay)
            delay = min(delay * self.poll_backoff_multiplier, self.poll_max_interval)

    def stream_logs(
        self,
        job_id: str,
        *,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        cursor: str | None = None,
        max_lines: int | None = None,
        poll_interval: float | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        """Yield a sub-job's log entries, polling with backoff.

        Drains the log then, with ``follow=True`` (the default), keeps polling
        for new entries until the caller stops iterating — suitable for a live
        tail. With ``follow=False`` it stops once the log reports ``eof``.
        """
        return self._stream_pages(
            lambda cur: self.tail_logs(
                job_id,
                cursor=cur,
                max_lines=max_lines,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            ),
            "entries",
            follow=follow,
            cursor=cursor,
            poll_interval=poll_interval,
        )

    # ─── ZMD scheduling/zone events (read-only) ───────────────────────────

    def tail_events(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_events: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
    ) -> dict:
        """Fetch one page of ZMD scheduling/zone events for the session.

        Returns ``{"events": [...], "next_cursor": str, "eof": bool}``. Served
        by the ZMD itself (not routed to a Zone Manager).
        """
        payload: dict = {}
        if cursor is not None:
            payload["cursor"] = cursor
        if max_events is not None:
            payload["max_events"] = max_events
        return self._operation(
            job_id,
            "zmd-events",
            payload=payload or None,
            sub_job_id=sub_job_id,
            sub_job_type=sub_job_type,
        )

    def stream_events(
        self,
        job_id: str,
        *,
        cursor: str | None = None,
        max_events: int | None = None,
        sub_job_id: str | None = None,
        sub_job_type: str | None = None,
        poll_interval: float | None = None,
        follow: bool = True,
    ) -> Iterator[dict]:
        """Yield ZMD scheduling/zone events for the session, polling with backoff."""
        return self._stream_pages(
            lambda cur: self.tail_events(
                job_id,
                cursor=cur,
                max_events=max_events,
                sub_job_id=sub_job_id,
                sub_job_type=sub_job_type,
            ),
            "events",
            follow=follow,
            cursor=cursor,
            poll_interval=poll_interval,
        )

    # ─── Checkpoints ─────────────────────────────────────────────────────

    def list_checkpoints(self, job_id: str) -> list:
        resp = self._send("GET", f"{self._prefix}/{job_id}/checkpoints")
        return resp.json().get("checkpoints", [])

    def export_checkpoint(self, job_id: str, checkpoint_id: str) -> dict:
        # GS uses colon-action syntax: /{jobId}/checkpoints/{cpId}:export
        resp = self._send(
            "POST",
            f"{self._prefix}/{job_id}/checkpoints/{checkpoint_id}:export",
        )
        return resp.json()

    def delete_checkpoint(self, job_id: str, checkpoint_id: str) -> None:
        """Delete a checkpoint. Returns None on success."""
        self._send("DELETE", f"{self._prefix}/{job_id}/checkpoints/{checkpoint_id}")
