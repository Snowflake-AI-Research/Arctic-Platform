from __future__ import annotations

import asyncio
import builtins
import logging
import os
import random
import re
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any
from uuid import uuid4

import ray

from arctic_platform.inference.envs import arctic_inference_effective_enabled
from arctic_platform.inference.server.metrics import RingStatLogger, get_collector
from arctic_platform.inference.server.streaming import StreamingWorkerMixin, stream_lifecycle_change
from arctic_platform.inference.vllm.dense_prompt_logprobs import (
    RESULT_KEY as _DENSE_PROMPT_LOGPROBS_KEY,
    stage_sampling_params as _stage_dense_prompt_logprobs,
    take_dense as _take_dense_prompt_logprobs,
)
from arctic_platform.inference.vllm.xgrammar_stop_mask import (
    ensure_xgrammar_stop_mask_fix,
)
from arctic_platform.inference.utils import require_supported_vllm_version

logger = logging.getLogger("arctic_platform.inference.server")

_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument '([^']+)'")
_SAMPLE_ID_PARAM_KEY = "dss_sample_id"
_REPLAY_ID_PARAM_KEY = "dss_router_replay_id"
_ROUTER_REPLAY_MARKER_KEY = "router_replay"
_ROUTER_REPLAY_RETURN_INFO_PARAM_KEY = "dss_return_back_router_info"
_ROUTER_REPLAY_STOP_TOKEN_SEQUENCES_PARAM_KEY = "dss_stop_token_sequences"
_ROUTER_REPLAY_MAX_CACHE_BYTES_ENGINE_KEY = "router_replay_max_cache_bytes"
_ENABLE_THINKING_PARAM_KEY = "enable_thinking"
_RETURN_ACTION_MASKS_PARAM_KEY = "return_action_masks"
_ADDRESS_IN_USE_MARKERS = (
    "eaddrinuse",
    "address already in use",
    "code: -98",
    "errno 98",
)
_BASE_EXCEPTION_GROUP = getattr(builtins, "BaseExceptionGroup", None)


def _grammar_stop_token_ids(vllm_config: Any, tokenizer: Any) -> tuple[int, ...]:
    """The sampler's stop set, as the live xgrammar backend masks it.

    The generation config's ``eos_token_id`` list only reaches the per-request
    SamplingParams clone inside the engine, so replay has to read it from the
    config. The tokenizer eos is included because ``override_stop_tokens``
    replaces xgrammar's stop set rather than extending it.
    """
    eos = vllm_config.model_config.try_get_generation_config().get("eos_token_id")
    stop = set()
    if isinstance(eos, int):
        stop.add(eos)
    elif eos is not None:
        stop.update(int(token_id) for token_id in eos)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        stop.add(int(tokenizer_eos))
    return tuple(sorted(stop))


def _coerce_structured_outputs_params(sampling_params: dict[str, Any]) -> None:
    """Normalize OpenAI-compatible structured-output kwargs for vLLM core."""
    from vllm.sampling_params import StructuredOutputsParams

    guided_json = sampling_params.pop("guided_json", None)
    structured_outputs = sampling_params.get("structured_outputs")

    if isinstance(structured_outputs, Mapping):
        sampling_params["structured_outputs"] = StructuredOutputsParams(**dict(structured_outputs))
        return

    if structured_outputs is not None:
        return

    if isinstance(guided_json, Mapping):
        sampling_params["structured_outputs"] = StructuredOutputsParams(json=dict(guided_json))


def _coerce_structured_outputs_config(engine_kwargs: dict[str, Any]) -> None:
    """Normalize programmatic structured-output engine config for vLLM."""
    from vllm.config import StructuredOutputsConfig

    raw_config = engine_kwargs.get("structured_outputs_config")
    if isinstance(raw_config, Mapping):
        engine_kwargs["structured_outputs_config"] = StructuredOutputsConfig(**dict(raw_config))


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool when provided, got {type(value).__name__}")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%r is below %d; using %d", name, raw, minimum, minimum)
        return minimum
    return value


def _ray_blanked_cuda_visible_devices() -> bool:
    """True when Ray set CUDA_VISIBLE_DEVICES to '' (num_gpus=0)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    return cvd is not None and cvd.strip() == ""


def _clear_ray_blanked_cuda_visible_devices() -> bool:
    """Drop Ray's empty CVD before any CUDA init (empty CVD sticks at 0 devices)."""
    if not _ray_blanked_cuda_visible_devices():
        return False
    del os.environ["CUDA_VISIBLE_DEVICES"]
    return True


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a float; using %.3f", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%r is below %.3f; using %.3f", name, raw, minimum, minimum)
        return minimum
    return value


def _is_address_in_use_error(exc: BaseException) -> bool:
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        message = repr(current).lower()
        if any(marker in message for marker in _ADDRESS_IN_USE_MARKERS):
            return True

        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)
        if isinstance(context, BaseException):
            stack.append(context)
        if _BASE_EXCEPTION_GROUP is not None and isinstance(current, _BASE_EXCEPTION_GROUP):
            stack.extend(current.exceptions)
    return False


def _ensure_arctic_vllm_patches() -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs

    if "__new__" in getattr(AsyncEngineArgs, "_arctic_patches", {}):
        return

    from arctic_platform.inference.vllm.plugin import arctic_inference_plugin

    try:
        # Use the same entrypoint as installed-package discovery so required
        # correctness fixes and optional Arctic patches cannot drift apart.
        arctic_inference_plugin()
    except ValueError as exc:
        if "is already patched by" not in str(exc):
            raise


def _ensure_router_replay_vllm_patches() -> None:
    from arctic_platform.inference.vllm.router_replay import (
        ensure_router_replay_vllm_patches,
    )

    ensure_router_replay_vllm_patches()


def _create_async_engine_args(
    engine_kwargs: dict[str, Any],
    *,
    enable_arctic_patches: bool = True,
):
    if os.getenv("ARCTIC_INFERENCE_SKIP_VERSION_CHECK", "0") != "1":
        require_supported_vllm_version("InferenceWorker")

    if enable_arctic_patches:
        _ensure_arctic_vllm_patches()
        from vllm.engine.arg_utils import AsyncEngineArgs
        engine_args_cls = AsyncEngineArgs
    else:
        from arctic_platform.inference.vllm.fp32_lm_head import (
            Fp32LmHeadAsyncEngineArgs,
            ensure_fp32_lm_head_vllm_patches,
        )
        engine_args_cls = Fp32LmHeadAsyncEngineArgs

    _coerce_structured_outputs_config(engine_kwargs)

    try:
        engine_args = engine_args_cls(**engine_kwargs)
        if not enable_arctic_patches:
            ensure_fp32_lm_head_vllm_patches(
                enabled=(
                    bool(getattr(engine_args, "fp32_lm_head", False))
                    or os.getenv("ARCTIC_FP32_LM_HEAD", "0") == "1"
                )
            )
        return engine_args
    except TypeError as exc:
        match = _UNEXPECTED_KWARG_RE.search(str(exc))
        if match is None:
            raise

        unsupported = match.group(1)
        if unsupported not in engine_kwargs:
            raise

        if (
            enable_arctic_patches
            and unsupported == "fp32_lm_head"
            and engine_kwargs.get(unsupported)
        ):
            raise TypeError(
                "fp32_lm_head was requested but ArcticInference could not "
                "install the vLLM ArcticArgs patch before AsyncEngineArgs "
                "validation."
            ) from exc

        filtered_kwargs = dict(engine_kwargs)
        filtered_kwargs.pop(unsupported, None)
        logger.warning("Dropping unsupported vLLM engine kwargs: %s", [unsupported])
        return _create_async_engine_args(
            filtered_kwargs,
            enable_arctic_patches=enable_arctic_patches,
        )


def _pop_router_replay_engine_kwargs(engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if _ROUTER_REPLAY_MAX_CACHE_BYTES_ENGINE_KEY in engine_kwargs:
        out[_ROUTER_REPLAY_MAX_CACHE_BYTES_ENGINE_KEY] = engine_kwargs.pop(
            _ROUTER_REPLAY_MAX_CACHE_BYTES_ENGINE_KEY
        )
    return out


def _has_string_stop(sampling_params: dict[str, Any]) -> bool:
    stop = sampling_params.get("stop")
    if isinstance(stop, str):
        return bool(stop)
    if isinstance(stop, list):
        return any(isinstance(item, str) and item for item in stop)
    return False


def _normalize_stop_token_sequences(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("dss_stop_token_sequences must be a list")
    out = []
    for item in value:
        if isinstance(item, dict):
            token_ids = item.get("token_ids")
            include = bool(item.get("include_in_output", False))
        else:
            token_ids = item
            include = False
        if not isinstance(token_ids, list) or not all(isinstance(tok, int) for tok in token_ids):
            raise ValueError(f"Invalid dss_stop_token_sequences entry: {item!r}")
        out.append({"token_ids": list(token_ids), "include_in_output": include})
    return out


def _extract_reasoning_content(parser: Any, text: str, *, reasoning_ended: bool | None = None) -> tuple[str | None, str]:
    if parser is None or not text:
        return None, text
    reasoning, content = parser.extract_reasoning(text, None)
    if not content and reasoning_ended is True:
        return None, text
    if content is None and reasoning is None:
        return None, text
    return reasoning, content or ""


def _find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    needle_len = len(needle)
    for start in range(len(haystack) - needle_len, -1, -1):
        if list(haystack[start : start + needle_len]) == list(needle):
            return start
    return None


def _split_reasoning_content_token_ids(
    parser: Any,
    token_ids: Sequence[int],
    *,
    reasoning_ended: bool | None = None,
) -> tuple[list[int] | None, list[int]]:
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        return None, []
    if parser is None:
        return None, ids
    if reasoning_ended is True:
        return None, ids

    extract_content_ids = getattr(parser, "extract_content_ids", None)
    if callable(extract_content_ids):
        content_ids = [int(token_id) for token_id in extract_content_ids(ids)]
        if content_ids:
            start = _find_last_subsequence(ids, content_ids)
            if start is not None:
                return ids[:start] or None, ids[start : start + len(content_ids)]

    return None, ids


def _serialize_logprobs_position(pos_data: dict | None) -> dict[int, dict] | None:
    """Convert a single position's {token_id: Logprob} to {token_id: dict}.

    Works for both prompt_logprobs and sample_logprobs positions.
    vLLM's Logprob is a dataclass with .logprob and .rank attributes.
    """
    if pos_data is None:
        return None
    out = {}
    for tok_id, lp in pos_data.items():
        if hasattr(lp, "logprob"):
            out[tok_id] = {"logprob": lp.logprob, "rank": lp.rank}
        else:
            out[tok_id] = {"logprob": float(lp), "rank": None}
    return out


def _sampled_logprob(pos_data: dict | None, token_id: int) -> float:
    if not pos_data:
        return 0.0
    entry = pos_data.get(token_id)
    if entry is None:
        entry = pos_data.get(str(token_id))
    if entry is None and pos_data:
        entry = next(iter(pos_data.values()))
    if isinstance(entry, dict):
        return float(entry.get("logprob", 0.0))
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    return float(entry or 0.0)


def _sampled_logprobs_for_output(choice: Any) -> list[float]:
    token_ids = list(choice.token_ids)
    if choice.logprobs is None:
        return [0.0] * len(token_ids)
    logprobs = [
        _sampled_logprob(pos, int(token_id))
        for pos, token_id in zip(choice.logprobs, token_ids)
    ]
    if len(logprobs) < len(token_ids):
        logprobs.extend([0.0] * (len(token_ids) - len(logprobs)))
    return logprobs


def _result_from_output(
    final_output: Any,
    *,
    return_sampled_logprobs_only: bool,
    cache_tx: Any = None,
    sample_id: str | None = None,
    replay_id: str | None = None,
    request_id: str | None = None,
    replica_label: str | None = None,
    return_back_router_info: bool = False,
) -> dict[str, Any]:
    if not final_output or not final_output.outputs:
        return {
            "text": "",
            "token_ids": [],
            "finish_reason": "abort",
            "prompt_len": 0,
            "generation_len": 0,
            "prefix_cache_len": 0,
        }

    choice = final_output.outputs[0]
    prompt_token_ids = getattr(final_output, "prompt_token_ids", None) or []
    num_cached = getattr(final_output, "num_cached_tokens", None)
    finish_reason = choice.finish_reason or "abort"
    result: dict[str, Any] = {
        "text": choice.text,
        "token_ids": list(choice.token_ids),
        "finish_reason": finish_reason,
        # Per-request metric fields. These are read by the scheduler to
        # populate `RequestRecord` and are silently ignored by callers
        # that don't care about metrics.
        "prompt_len": len(prompt_token_ids),
        "generation_len": len(choice.token_ids),
        "prefix_cache_len": int(num_cached) if num_cached is not None else 0,
    }

    if final_output.prompt_logprobs is not None:
        # The dense patch leaves tensors here instead of one dict per position;
        # everything else still gets the per-position dicts, unchanged.
        dense = _take_dense_prompt_logprobs(final_output.prompt_logprobs)
        if dense is not None:
            result[_DENSE_PROMPT_LOGPROBS_KEY] = dense
        else:
            result["prompt_logprobs"] = [
                _serialize_logprobs_position(pos)
                for pos in final_output.prompt_logprobs
            ]

    if choice.logprobs is not None:
        if return_sampled_logprobs_only:
            result["logprobs"] = _sampled_logprobs_for_output(choice)
        else:
            result["logprobs"] = [
                _serialize_logprobs_position(pos) for pos in choice.logprobs
            ]

    _maybe_cache_routed_experts(
        result,
        final_output=final_output,
        cache_tx=cache_tx,
        sample_id=sample_id,
        replay_id=replay_id,
        request_id=request_id,
        replica_label=replica_label,
        return_back_router_info=return_back_router_info,
    )
    return result


def _maybe_cache_routed_experts(
    result: dict[str, Any],
    *,
    final_output: Any,
    cache_tx: Any,
    sample_id: str | None,
    replay_id: str | None,
    request_id: str | None,
    replica_label: str | None,
    return_back_router_info: bool,
) -> None:
    if cache_tx is None or sample_id is None:
        return
    routed = getattr(final_output, "routed_experts", None)
    if routed is None and getattr(final_output, "outputs", None):
        routed = getattr(final_output.outputs[0], "routed_experts", None)
    if routed is None:
        logger.warning("router-replay: routed_experts missing for sample_id=%s", sample_id)
        return
    capture_len = int(routed.shape[0])
    prompt_len = int(result["prompt_len"])
    generation_len = int(result["generation_len"])
    expected_capture_len = prompt_len + generation_len - 1
    if capture_len != expected_capture_len:
        raise ValueError(
            "router-replay capture length mismatch "
            f"trajectory_id={sample_id!r} replay_id={replay_id!r} request_id={request_id!r} "
            f"prompt_len={prompt_len} generation_len={generation_len} "
            f"capture_len={capture_len} expected_capture_len={expected_capture_len} "
            f"capture_shape={list(routed.shape)}"
        )
    cache_key = replay_id or sample_id
    if replay_id is not None and replay_id.startswith("rr1:"):
        cache_tx.put_new(cache_key, routed)
    else:
        cache_tx.put(cache_key, routed)
    marker = {
        "sample_id": cache_key,
        "replay_id": cache_key,
        "trajectory_id": sample_id,
        "request_id": request_id,
        "replica": replica_label,
        "prompt_len": prompt_len,
        "generation_len": generation_len,
        "capture_len": capture_len,
    }
    if return_back_router_info:
        tensor = routed.detach().cpu() if hasattr(routed, "detach") else routed
        marker.update(
            {
                "routed_experts": tensor.tolist(),
                "routed_experts_shape": list(tensor.shape),
                "routed_experts_dtype": str(tensor.dtype).removeprefix("torch."),
            }
        )
    result[_ROUTER_REPLAY_MARKER_KEY] = marker



class WorkerLifecycleState(str, Enum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    SLEEPING = "sleeping"


@ray.remote
class InferenceWorker(StreamingWorkerMixin):
    """Ray actor that hosts an in-process vLLM AsyncLLM engine."""

    def __init__(self) -> None:
        self.llm = None
        self.state = WorkerLifecycleState.UNINITIALIZED
        self._reasoning_parser: Any = None
        self._router_replay_tx: Any = None
        self._router_replay_group: Any = None
        self._replica_label: str | None = None
        self._router_replay_shm_entry: dict[str, Any] | None = None
        self._router_replay_shm_scope: str | None = None
        self._return_reasoning_content = False
        self._structured_outputs_enabled_in_reasoning = False
        self._grammar_stop_token_ids: tuple[int, ...] = ()
        self._active_lora_int_id: int | None = None
        self._active_lora_name: str | None = None
        self._active_lora_is_3d: bool = False

    def _active_lora_request(self) -> Any | None:
        """LoRARequest for the synced adapter, or None (base serving).

        Weight sync already replaced the resident stable id through the adapter
        manager. ``load_inplace`` stays false so vLLM does not try to load the
        placeholder ``lora_path`` from disk.
        """
        if self._active_lora_int_id is None:
            return None
        from vllm.lora.request import LoRARequest

        name = self._active_lora_name or f"policy-{self._active_lora_int_id}"
        return LoRARequest(
            lora_name=name,
            lora_int_id=self._active_lora_int_id,
            lora_path=f"arctic-nccl://{name}",
            load_inplace=False,
            is_3d_lora_weight=self._active_lora_is_3d,
        )

    async def _generate_once(
        self,
        prompt_input: Any,
        params: Any,
        request_id: str,
        reasoning_ended: bool | None = None,
        lora_request: Any | None = None,
    ) -> Any:
        final_output = None
        generate_kwargs: dict[str, Any] = {
            "request_id": request_id,
            "reasoning_ended": reasoning_ended,
        }
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request
        try:
            async for output in self.llm.generate(
                prompt_input,
                params,
                **generate_kwargs,
            ):
                final_output = output
        except GeneratorExit:
            raise

        return final_output

    async def initialize(
        self,
        engine_kwargs: dict[str, Any],
        extra_env: dict[str, str] | None = None,
        model_id: str | None = None,
    ) -> None:
        if extra_env:
            os.environ.update(extra_env)

        # Multi-node coordinators use num_gpus=0 (CVD=''). Clear before any CUDA
        # touch so router-replay NCCL can use a local GPU.
        if _clear_ray_blanked_cuda_visible_devices():
            logger.info(
                "Cleared empty CUDA_VISIBLE_DEVICES before vLLM engine init"
            )

        # `reasoning_parser` is used by vLLM structured outputs to avoid
        # constraining reasoning tokens.
        reasoning_parser_name = engine_kwargs.get("reasoning_parser")
        self._return_reasoning_content = bool(engine_kwargs.pop("return_reasoning_content", False))
        lora_adapter_path = engine_kwargs.pop("lora_adapter_path", None)

        # Force-load vLLM general_plugins before constructing AsyncEngineArgs.
        # vLLM normally calls load_general_plugins() in
        # AsyncEngineArgs.__post_init__, which runs *after* __init__ has
        # already validated kwargs against the un-patched field set.
        import vllm.plugins
        vllm.plugins.load_general_plugins()

        arctic_enabled = arctic_inference_effective_enabled()
        if arctic_enabled:
            # Some PrimeRL uv envs import arctic_platform.inference from PYTHONPATH
            # without installed entrypoint metadata, so load_general_plugins()
            # cannot discover the Arctic plugin. Apply the same patches
            # directly when ArcticInference is explicitly enabled.
            _ensure_arctic_vllm_patches()
        else:
            engine_kwargs.pop("forest_cascade_attn_configs", None)

        _ensure_router_replay_vllm_patches()
        ensure_xgrammar_stop_mask_fix()

        from vllm.v1.engine.async_llm import AsyncLLM

        router_replay_engine_kwargs = _pop_router_replay_engine_kwargs(engine_kwargs)
        engine_kwargs.setdefault(
            "worker_extension_cls",
            "arctic_platform.inference.server.weight_sync.WeightSyncExtension",
        )
        max_attempts = _env_int("ARCTIC_VLLM_ENGINE_STARTUP_ATTEMPTS", 5)
        retry_base_s = _env_float("ARCTIC_VLLM_ENGINE_STARTUP_RETRY_BASE_S", 1.0)
        for attempt in range(1, max_attempts + 1):
            try:
                attempt_kwargs = dict(engine_kwargs)
                engine_args = _create_async_engine_args(
                    attempt_kwargs,
                    enable_arctic_patches=arctic_enabled,
                )
                vllm_config = engine_args.create_engine_config()
                self._structured_outputs_enabled_in_reasoning = bool(
                    vllm_config.structured_outputs_config.enable_in_reasoning
                )
                self._register_router_replay_shm(attempt_kwargs, vllm_config, model_id)
                self.llm = AsyncLLM.from_vllm_config(
                    vllm_config,
                    stat_loggers=[RingStatLogger],
                )
                break
            except BaseException as exc:
                self._cleanup_registered_router_replay_shm()
                if attempt >= max_attempts or not _is_address_in_use_error(exc):
                    raise
                delay_s = retry_base_s * attempt + random.uniform(0.0, retry_base_s)
                logger.warning(
                    "vLLM engine startup hit address-in-use on attempt %d/%d; "
                    "retrying in %.2fs",
                    attempt,
                    max_attempts,
                    delay_s,
                    exc_info=True,
                )
                await asyncio.sleep(delay_s)
        self._maybe_init_router_replay_tx(engine_kwargs, router_replay_engine_kwargs)
        # The live grammar now always masks the sampler's stop set, so replay
        # mirrors it unconditionally. get_tokenizer() raises under
        # skip_tokenizer_init, and such an engine cannot compile a grammar to
        # replay in the first place.
        if not vllm_config.model_config.skip_tokenizer_init:
            self._grammar_stop_token_ids = _grammar_stop_token_ids(
                vllm_config, self.llm.get_tokenizer())
        self._maybe_init_reasoning_parser(reasoning_parser_name)
        if lora_adapter_path:
            await self._load_checkpoint_lora_adapter(lora_adapter_path)
        self.state = WorkerLifecycleState.READY
        logger.info("Worker %d initialized: model=%s", os.getpid(), engine_kwargs.get("model"))

    async def _load_checkpoint_lora_adapter(self, lora_path: str) -> None:
        import json

        from vllm.lora.request import LoRARequest

        cfg_path = os.path.join(lora_path, "adapter_config.json")
        is_3d = False
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                adapter_cfg = json.load(f)
            is_3d = bool(isinstance(adapter_cfg, dict) and adapter_cfg.get("target_parameters"))
        request = LoRARequest(
            lora_name="default",
            lora_int_id=1,
            lora_path=lora_path,
            load_inplace=True,
            is_3d_lora_weight=is_3d,
        )
        added = await self.llm.add_lora(request)
        if not added:
            raise RuntimeError(f"vLLM failed to load LoRA adapter from {lora_path}")
        self._active_lora_int_id = 1
        self._active_lora_name = "default"
        self._active_lora_is_3d = is_3d
        logger.info("Loaded checkpoint LoRA adapter from %s (3d=%s)", lora_path, is_3d)

    def _maybe_init_reasoning_parser(self, reasoning_parser_name: str | None) -> None:
        if not reasoning_parser_name:
            return
        from vllm.reasoning import ReasoningParserManager

        parser_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
        self._reasoning_parser = parser_cls(self.llm.get_tokenizer())
        logger.info("Worker %d reasoning parser enabled: %s", os.getpid(), reasoning_parser_name)

    def _is_reasoning_open(self, prompt_ids: Sequence[int]) -> bool:
        parser = self._reasoning_parser
        start_token_id = getattr(parser, "start_token_id", None)
        end_token_id = getattr(parser, "end_token_id", None)
        if start_token_id is None or end_token_id is None:
            return False
        last_start = -1
        last_end = -1
        for index, token_id in enumerate(prompt_ids):
            if token_id == start_token_id:
                last_start = index
            elif token_id == end_token_id:
                last_end = index
        return last_start > last_end

    def _prefill_think_for_generation(
        self,
        prompt: str | list[int],
        *,
        enable_thinking: bool | None,
    ) -> tuple[str | list[int], list[int], bool | None]:
        if isinstance(prompt, list):
            prompt_ids = [int(token_id) for token_id in prompt]
        else:
            prompt_ids = []

        if enable_thinking is None:
            reasoning_ended = self._reasoning_ended_for_prompt(prompt_ids) if prompt_ids else None
            return prompt, prompt_ids, reasoning_ended

        if not enable_thinking:
            return prompt, prompt_ids, True

        parser = self._reasoning_parser
        start_token_id = getattr(parser, "start_token_id", None)
        if start_token_id is None:
            return prompt, prompt_ids, None

        if isinstance(prompt, list):
            if not self._is_reasoning_open(prompt_ids):
                prompt_ids = [*prompt_ids, int(start_token_id)]
            return prompt_ids, prompt_ids, False

        start_token = getattr(parser, "start_token", "<think>")
        prompt_text = str(prompt)
        if start_token not in prompt_text.rsplit("</think>", 1)[-1]:
            prompt_text = f"{prompt_text}{start_token}"
        return prompt_text, [], False

    def _reasoning_ended_for_prompt(self, prompt_ids: list[int]) -> bool | None:
        parser = self._reasoning_parser
        if parser is None or not hasattr(parser, "is_reasoning_end"):
            return None
        return bool(parser.is_reasoning_end(prompt_ids))

    def _maybe_init_router_replay_tx(
        self,
        engine_kwargs: dict[str, Any],
        router_replay_engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not engine_kwargs.get("enable_return_routed_experts"):
            return
        from arctic_platform.inference.server.router_replay import RouterReplayCacheTX
        import torch

        # Avoid torch.cuda.* while CVD is still ''; keep TX on CPU until NCCL init.
        if _ray_blanked_cuda_visible_devices():
            tx_device = torch.device("cpu")
            logger.warning(
                "router-replay TX cache on CPU (CUDA_VISIBLE_DEVICES=%r)",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        else:
            tx_device = torch.device("cuda:0")
        max_bytes = int(
            (router_replay_engine_kwargs or {}).get(
                _ROUTER_REPLAY_MAX_CACHE_BYTES_ENGINE_KEY,
                16 * 1024**3,
            )
        )
        self._router_replay_tx = RouterReplayCacheTX(
            device=tx_device,
            max_bytes=max_bytes,
        )
        self._replica_label = f"pid={os.getpid()}"
        logger.info("router-replay TX cache initialized max_bytes=%d", max_bytes)

    def _register_router_replay_shm(
        self,
        engine_kwargs: dict[str, Any],
        vllm_config: Any,
        model_id: str | None,
    ) -> None:
        if not engine_kwargs.get("enable_return_routed_experts"):
            return
        from arctic_platform.inference.server.router_replay.shm import (
            cleanup_scope,
            current_scope,
            register_expected_buffer,
        )

        scope = current_scope(None)
        cleanup_scope(scope=scope, model_id=model_id, stale_only=True)
        parallel_config = vllm_config.parallel_config
        instance_id = getattr(parallel_config, "data_parallel_rank", 0)
        dp_rank = getattr(parallel_config, "data_parallel_rank", 0)
        self._router_replay_shm_entry = register_expected_buffer(
            scope=scope,
            model_id=model_id,
            instance_id=instance_id,
            dp_rank=dp_rank,
            pid=os.getpid(),
        )
        self._router_replay_shm_scope = scope

    def _cleanup_registered_router_replay_shm(self) -> dict[str, Any] | None:
        if self._router_replay_shm_entry is None:
            return None
        from arctic_platform.inference.server.router_replay.shm import cleanup_entry

        result = cleanup_entry(self._router_replay_shm_entry, stale_only=False)
        logger.info("router-replay shm: cleanup result=%s", result)
        self._router_replay_shm_entry = None
        self._router_replay_shm_scope = None
        return result

    def _replay_action_masks(
        self,
        *,
        final_output: Any,
        result: dict[str, Any],
        sampling_params: Any,
        fallback_prompt_token_ids: Sequence[int],
        reasoning_ended: bool | None,
    ) -> Any | None:
        from arctic_platform.inference.server.action_mask_replay import build_action_masks_for_output

        prompt_token_ids = getattr(final_output, "prompt_token_ids", None)
        if prompt_token_ids is None:
            prompt_token_ids = fallback_prompt_token_ids
        completion_token_ids: Sequence[int] = result.get("token_ids") or []
        if getattr(final_output, "outputs", None):
            completion_token_ids = list(getattr(final_output.outputs[0], "token_ids", None) or completion_token_ids)
        grammar_stop_token_ids = tuple(
            set(self._grammar_stop_token_ids) | sampling_params.all_stop_token_ids)
        return build_action_masks_for_output(
            prompt_token_ids=list(prompt_token_ids or []),
            completion_token_ids=list(completion_token_ids),
            text=str(result.get("text") or ""),
            sampling_params=sampling_params,
            tokenizer=self.llm.get_tokenizer(),
            reasoning_parser=self._reasoning_parser,
            reasoning_ended=reasoning_ended,
            structured_outputs_enabled_in_reasoning=self._structured_outputs_enabled_in_reasoning,
            grammar_stop_token_ids=grammar_stop_token_ids,
        )

    async def generate(self, prompt: str | list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        if self.state != WorkerLifecycleState.READY:
            raise RuntimeError(f"Worker not ready: state={self.state.value}")
        if getattr(self, "_stream_cleanup_failed", False):
            raise RuntimeError("Worker streaming cleanup unconfirmed")

        from vllm import SamplingParams

        sampling_params = dict(sampling_params)
        enable_thinking = _optional_bool(
            sampling_params.pop(_ENABLE_THINKING_PARAM_KEY, None),
            name=_ENABLE_THINKING_PARAM_KEY,
        )
        return_sampled_logprobs_only = bool(sampling_params.pop("return_sampled_logprobs_only", False))
        return_action_masks = bool(sampling_params.pop(_RETURN_ACTION_MASKS_PARAM_KEY, False))
        extra_args = dict(sampling_params.get("extra_args", None) or {})
        extra_args.pop(_RETURN_ACTION_MASKS_PARAM_KEY, None)
        if extra_args:
            sampling_params["extra_args"] = extra_args
        else:
            sampling_params.pop("extra_args", None)
        sample_id = sampling_params.pop(_SAMPLE_ID_PARAM_KEY, None)
        if sample_id is not None and not isinstance(sample_id, str):
            sample_id = str(sample_id)
        replay_id = sampling_params.pop(_REPLAY_ID_PARAM_KEY, None)
        if replay_id is not None and not isinstance(replay_id, str):
            replay_id = str(replay_id)
        if replay_id is not None and sample_id is None:
            raise ValueError(f"{_REPLAY_ID_PARAM_KEY} requires {_SAMPLE_ID_PARAM_KEY}")
        return_back_router_info = bool(
            sampling_params.pop(_ROUTER_REPLAY_RETURN_INFO_PARAM_KEY, False)
        )
        stop_token_sequences = _normalize_stop_token_sequences(
            sampling_params.pop(_ROUTER_REPLAY_STOP_TOKEN_SEQUENCES_PARAM_KEY, None)
        )
        router_replay_required = sample_id is not None and self._router_replay_tx is not None
        if router_replay_required and _has_string_stop(sampling_params) and not stop_token_sequences:
            raise ValueError(
                "Router replay generation with string stop requires dss_stop_token_sequences."
            )
        if stop_token_sequences:
            sampling_params.pop("stop", None)
            extra_args = dict(sampling_params.pop("extra_args", None) or {})
            extra_args[_ROUTER_REPLAY_STOP_TOKEN_SEQUENCES_PARAM_KEY] = stop_token_sequences
            sampling_params["extra_args"] = extra_args
        # Park the dense-prompt-logprobs opt-in in extra_args so SamplingParams
        # accepts it and the engine-side patch can read it back off the request.
        _stage_dense_prompt_logprobs(sampling_params)
        _coerce_structured_outputs_params(sampling_params)
        params = SamplingParams(**sampling_params)
        request_id = str(uuid4())
        effective_prompt, prompt_token_ids, reasoning_ended = self._prefill_think_for_generation(
            prompt,
            enable_thinking=enable_thinking,
        )

        if isinstance(effective_prompt, list):
            prompt_input: Any = {"prompt_token_ids": effective_prompt}
        else:
            prompt_input = effective_prompt

        final_output = await self._generate_once(
            prompt_input,
            params,
            request_id,
            reasoning_ended=reasoning_ended,
            lora_request=self._active_lora_request(),
        )
        result = _result_from_output(
            final_output,
            return_sampled_logprobs_only=return_sampled_logprobs_only,
            cache_tx=self._router_replay_tx,
            sample_id=sample_id,
            replay_id=replay_id,
            request_id=request_id,
            replica_label=self._replica_label,
            return_back_router_info=return_back_router_info,
        )
        if self._return_reasoning_content:
            self._apply_reasoning_parser(result, reasoning_ended=reasoning_ended)
        if return_action_masks:
            result["action_masks"] = self._replay_action_masks(
                final_output=final_output,
                result=result,
                sampling_params=params,
                fallback_prompt_token_ids=prompt_token_ids,
                reasoning_ended=reasoning_ended,
            )
        return result

    def _apply_reasoning_parser(self, result: dict[str, Any], *, reasoning_ended: bool | None = None) -> None:
        if self._reasoning_parser is None or not result.get("text"):
            return
        # `request` is only used by chat/responses-specific parsers; the
        # think-token parsers (qwen3, deepseek_r1, ...) ignore it, and we have
        # no request object on this raw generate path, so pass None.
        reasoning, content = _extract_reasoning_content(
            self._reasoning_parser,
            result["text"],
            reasoning_ended=reasoning_ended,
        )
        reasoning_token_ids, content_token_ids = _split_reasoning_content_token_ids(
            self._reasoning_parser,
            result.get("token_ids") or [],
            reasoning_ended=reasoning_ended,
        )
        result["reasoning"] = reasoning
        result["content"] = content
        result["reasoning_token_ids"] = reasoning_token_ids
        result["content_token_ids"] = content_token_ids

    def is_healthy(self) -> bool:
        return self.llm is not None and self.state in (
            WorkerLifecycleState.READY, WorkerLifecycleState.SLEEPING,
        )

    def get_state(self) -> str:
        return self.state.value

    def get_stats(self) -> dict[str, Any]:
        """Return the latest scheduler stats, plus liveness metadata.

        Used by the per-replica concurrency-adjust loop in
        :class:`arctic_platform.inference.server.scheduler.Scheduler`. The keys
        ``gpu_cache_usage`` / ``num_requests_running`` /
        ``num_requests_waiting`` are read by the scheduler to drive the
        ``utilization_based_concurrency`` heuristic.
        """
        latest = get_collector().latest()
        return {
            "state": self.state.value,
            "pid": os.getpid(),
            **latest,
        }

    def drain_metrics(self) -> dict[str, Any]:
        """Return and clear the buffered per-step replica snapshots."""
        from arctic_platform.inference.server.action_mask_replay import (
            action_mask_replay_cache_stats,
        )

        return {
            "pid": os.getpid(),
            "snapshots": get_collector().drain_snapshots(),
            "action_mask_replay_cache": action_mask_replay_cache_stats(),
        }

    def set_replica_id(self, replica_id: int) -> None:
        """Tell this worker its position in the replica pool so that the
        snapshots it emits carry a stable replica identifier."""
        get_collector().set_replica_id(int(replica_id))

    def pid(self) -> int:
        return os.getpid()

    async def init_router_replay(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        is_server: bool = False,
    ) -> dict[str, Any]:
        if self._router_replay_tx is None:
            raise RuntimeError("router-replay TX cache is not initialized")
        from arctic_platform.inference.server.router_replay import (
            RouterReplayCacheTX,
            init_router_replay_group,
        )
        import torch

        # Re-clear if CVD was blanked again. Prefer runtime device count over
        # is_available() (NVML can lie after an empty-CVD probe).
        if _clear_ray_blanked_cuda_visible_devices():
            logger.info(
                "Cleared empty CUDA_VISIBLE_DEVICES before router-replay NCCL"
            )
        raw_count = int(torch._C._cuda_getDeviceCount())
        if raw_count <= 0:
            raise RuntimeError(
                "router-replay NCCL requires CUDA on InferenceWorker after "
                f"clearing empty CVD (runtime_device_count={raw_count}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
                f"is_available={torch.cuda.is_available()})"
            )
        tx_device = torch.device("cuda:0")
        if self._router_replay_tx.device != tx_device:
            max_bytes = self._router_replay_tx.max_bytes
            self._router_replay_tx = RouterReplayCacheTX(
                device=tx_device,
                max_bytes=max_bytes,
            )
            logger.info("Moved router-replay TX cache to %s for NCCL", tx_device)

        if self._router_replay_group is not None:
            self._router_replay_group.close()
        self._router_replay_group = init_router_replay_group(
            role="sender",
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            device=tx_device,
            is_server=is_server,
        )
        return {"status": "ok", "rank": rank, "world_size": world_size}

    async def send_router_replay(self) -> dict[str, Any]:
        if self._router_replay_group is None:
            raise RuntimeError("router-replay sender group is not initialized")
        if self._router_replay_tx is None:
            raise RuntimeError("router-replay TX cache is not initialized")
        result = self._router_replay_group.send(self._router_replay_tx)
        result["tx_stats"] = self._router_replay_tx.stats()
        return result

    async def discard_router_replay(self, sample_ids: list[str]) -> dict[str, Any]:
        if self._router_replay_tx is None:
            return {"status": "no_cache", "removed": 0}
        removed = self._router_replay_tx.discard(sample_ids)
        return {"status": "ok", "removed": removed, "tx_stats": self._router_replay_tx.stats()}

    async def close_router_replay(self) -> dict[str, Any]:
        if self._router_replay_group is not None:
            self._router_replay_group.close()
            self._router_replay_group = None
        if self._router_replay_tx is not None:
            self._router_replay_tx.clear()
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    @stream_lifecycle_change
    async def sleep(self, level: int = 1) -> dict[str, Any]:
        """Free GPU memory by offloading weights and/or KV cache.

        Args:
            level: 1 = free KV cache only, 2 = free KV cache + weights.
        """
        if self.state != WorkerLifecycleState.READY:
            raise RuntimeError(f"Cannot sleep: worker state is {self.state.value}")
        await self.llm.collective_rpc("sleep", kwargs={"level": level})
        self.state = WorkerLifecycleState.SLEEPING
        logger.info("Worker %d sleeping (level=%d)", os.getpid(), level)
        return {"status": "sleeping", "level": level}

    async def wake_up(self, tags: list[str] | None = None) -> dict[str, Any]:
        """Restore GPU memory (reverse of :meth:`sleep`).

        Args:
            tags: What to restore, e.g. ``["weights"]`` or
                  ``["weights", "kv_cache"]``.  ``None`` restores everything.
        """
        if self.state != WorkerLifecycleState.SLEEPING:
            if self.state == WorkerLifecycleState.READY:
                return {"status": "already_ready"}
            raise RuntimeError(f"Cannot wake up: worker state is {self.state.value}")
        kwargs: dict[str, Any] = {}
        if tags is not None:
            kwargs["tags"] = tags
        await self.llm.collective_rpc("wake_up", kwargs=kwargs)
        self.state = WorkerLifecycleState.READY
        logger.info("Worker %d awake", os.getpid())
        return {"status": "ready"}

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    @stream_lifecycle_change
    async def pause_generation(
        self,
        mode: str = "keep",
        clear_cache: bool = False,
    ) -> dict[str, Any]:
        if mode not in {"keep", "abort"}:
            raise ValueError(f"Unknown pause_generation mode: {mode!r}. Use: keep, abort")
        if self.state != WorkerLifecycleState.READY or self.llm is None:
            return {"status": "skipped", "reason": f"state={self.state.value}"}
        await self.llm.pause_generation(mode=mode, clear_cache=clear_cache)
        self._stream_engine_paused = True
        logger.info(
            "Worker %d pause_generation(mode=%s, clear_cache=%s)",
            os.getpid(),
            mode,
            clear_cache,
        )
        return {"status": "paused", "mode": mode, "clear_cache": clear_cache}

    async def resume_generation(self) -> dict[str, Any]:
        if self.state != WorkerLifecycleState.READY or self.llm is None:
            return {"status": "skipped", "reason": f"state={self.state.value}"}
        await self.llm.collective_rpc("_arl_cuda_sync")
        await self.llm.resume_generation()
        self._stream_engine_paused = False
        logger.info("Worker %d cuda barrier + resume_generation", os.getpid())
        return {"status": "resumed"}

    async def reset_prefix_cache(
        self,
        timeout_s: float = 0.0,
        retry_interval_s: float = 0.1,
    ) -> dict[str, Any]:
        if self.state != WorkerLifecycleState.READY or self.llm is None:
            return {
                "status": "skipped",
                "reset_ok": False,
                "reason": f"state={self.state.value}",
                "attempts": 0,
            }
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        attempts = 0
        while True:
            attempts += 1
            reset_ok = bool(await self.llm.reset_prefix_cache())
            logger.info(
                "Worker %d reset prefix cache: ok=%s attempts=%d",
                os.getpid(),
                reset_ok,
                attempts,
            )
            if reset_ok or time.monotonic() >= deadline:
                return {
                    "status": "ok" if reset_ok else "failed",
                    "reset_ok": reset_ok,
                    "attempts": attempts,
                }
            await asyncio.sleep(max(0.01, float(retry_interval_s)))

    @stream_lifecycle_change
    async def sync_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        direct_mode: bool = False,
        reverse: bool = False,
    ) -> dict[str, Any]:
        """Receive + load weights on all TP workers via a single call."""
        results = await self.llm.collective_rpc(
            "sync_weights",
            args=(master_addr, master_port, rank_offset, world_size,
                  bucket_size, engine_only, direct_mode, reverse),
        )
        return results[0] if results else {}

    @stream_lifecycle_change
    async def sync_weights_broadcast(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        weight_format: str = "vllm",
    ) -> dict[str, Any]:
        """Receive + load weights on all TP workers via NCCL broadcast (rank 0 = trainer).

        When ``envs.ARCTIC_INFERENCE_DUMP_PARAM_L2`` is set (opt-in, default off),
        each TP rank's full-model and loaded-destination L2 maps are gathered
        here (only rank 0's dict escapes ``collective_rpc``). Mapping and skip
        artifacts remain on the rank-0 result for TP1 trace validation.
        """
        results = await self.llm.collective_rpc(
            "sync_weights_broadcast",
            args=(master_addr, master_port, rank_offset, world_size,
                  bucket_size, engine_only, weight_format),
        )
        if not results:
            return {}
        from arctic_platform.inference import envs
        if not envs.ARCTIC_INFERENCE_DUMP_PARAM_L2:
            return results[0]
        # Verification path only: preserve the original full-model trace and
        # gather the loaded-only trace under a distinct field.
        head = dict(results[0])
        head["all_rank_param_l2"] = [
            (r.get("param_l2") if isinstance(r, dict) else None) for r in results
        ]
        head.pop("param_l2", None)
        if head.get("loaded_destination_trace_collected"):
            head["all_rank_loaded_param_l2"] = [
                (
                    r.get("loaded_param_l2")
                    if isinstance(r, dict)
                    else None
                )
                for r in results
            ]
            head.pop("loaded_param_l2", None)
        return head

    @stream_lifecycle_change
    async def sync_lora_weights_broadcast(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        lora_int_id: int,
        lora_name: str,
        lora_config: dict[str, Any],
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        staging: str = "cpu",
        evict_first: bool = True,
    ) -> dict[str, Any]:
        """Install a LoRA adapter on all TP workers via NCCL broadcast, then make
        it this replica's active adapter. ``engine_only`` does just the NCCL
        rendezvous.
        """
        results = await self.llm.collective_rpc(
            "sync_lora_weights_broadcast",
            args=(master_addr, master_port, rank_offset, world_size,
                  lora_int_id, lora_name, lora_config, bucket_size,
                  engine_only, staging, evict_first),
        )
        if not engine_only:
            self._active_lora_int_id = lora_int_id
            self._active_lora_name = lora_name
            self._active_lora_is_3d = bool(
                results[0].get("is_3d_lora_weight", False) if results else False
            )
        return results[0] if results else {}

    @stream_lifecycle_change
    async def sync_spec_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        reverse: bool = False,
    ) -> dict[str, Any]:
        """Receive + load spec (drafter) model weights on all TP workers."""
        results = await self.llm.collective_rpc(
            "sync_spec_weights",
            args=(master_addr, master_port, rank_offset, world_size,
                  bucket_size, engine_only, reverse),
        )
        return results[0] if results else {}

    async def close_weight_sync(self) -> dict[str, Any]:
        """Destroy persistent NCCLEngine on all TP workers."""
        results = await self.llm.collective_rpc("close_weight_sync")
        return results[0] if results else {}

    def shutdown(self) -> None:
        self._cleanup_registered_router_replay_shm()
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.state = WorkerLifecycleState.UNINITIALIZED
