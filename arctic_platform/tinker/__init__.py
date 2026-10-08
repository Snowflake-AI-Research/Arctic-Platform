# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Drop-in ``tinker`` module that trains on Cortex in-process.

Importing this module registers it as ``sys.modules["tinker"]``, so a
tinker-cookbook recipe's later ``import tinker`` uses Cortex. There is no
local HTTP server. GPU counts are constructor arguments, the same ones the
Cortex client CLI takes::

    from arctic_platform import tinker
    service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)

A recipe that constructs ``ServiceClient(base_url=...)`` itself still works
when launched through ``python -m arctic_platform.tinker.run``, which sets
those counts first. ``base_url`` is ignored. Connection settings stay in
``ARCTIC_CORTEX_*``.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import logging
import sys
import threading
from typing import Any
from typing import NoReturn

import numpy as np

_sdk = importlib.import_module("tinker")

for _name in getattr(_sdk, "__all__", []):
    if _name == "ServiceClient":
        continue
    globals()[_name] = getattr(_sdk, _name)

types = _sdk.types
lib = _sdk.lib

logger = logging.getLogger(__name__)

_settings: dict[str, Any] = {
    "training_gpus": None,
    "sampling_gpus": None,
    "teacher_sampling_gpus": 1,
    "max_prompt_length": 2048,
    "max_response_length": 512,
}
_sessions: list[Any] = []


def configure(
    *,
    training_gpus: int,
    sampling_gpus: int,
    teacher_sampling_gpus: int = 1,
    max_prompt_length: int = 2048,
    max_response_length: int = 512,
) -> None:
    """Record the GPU counts a recipe's ``ServiceClient(base_url=...)`` should use."""
    if training_gpus < 1 or sampling_gpus < 1 or teacher_sampling_gpus < 1:
        raise ValueError("training_gpus, sampling_gpus, and teacher_sampling_gpus must all be at least 1")
    _settings["training_gpus"] = int(training_gpus)
    _settings["sampling_gpus"] = int(sampling_gpus)
    _settings["teacher_sampling_gpus"] = int(teacher_sampling_gpus)
    _settings["max_prompt_length"] = int(max_prompt_length)
    _settings["max_response_length"] = int(max_response_length)


def _release_sessions() -> None:
    for client in _sessions:
        transport = getattr(client, "transport", None)
        shutdown = getattr(transport, "shutdown", None)
        if shutdown is None:
            continue
        try:
            shutdown()
        except Exception as exc:  # noqa: BLE001 - a failed release must not hide the original error
            logger.warning("Cortex session release failed: %s", exc)


atexit.register(_release_sessions)


class _Future(_sdk.APIFuture[Any]):
    def __init__(self, task: asyncio.Task) -> None:
        self._task = task

    async def result_async(self, timeout: float | None = None) -> Any:
        if timeout is None:
            return await self._task
        return await asyncio.wait_for(asyncio.shield(self._task), timeout)

    def result(self, timeout: float | None = None) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = self._task.get_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(self.result_async(timeout), loop).result(timeout)
            return loop.run_until_complete(self.result_async(timeout))
        raise RuntimeError("call result_async() from a running event loop")


class _Path:
    def __init__(self, path: str) -> None:
        self.path = path


def _refuse_checkpoint(path: str) -> NoReturn:
    raise RuntimeError(f"cannot load {path}: this process does not store checkpoints, so a new process cannot resume")


def _pick(explicit: int | None, key: str) -> int:
    value = explicit if explicit is not None else _settings[key]
    if value is None:
        raise ValueError(
            "pass training_gpus and sampling_gpus to ServiceClient, or launch with "
            "python -m arctic_platform.tinker.run --training-gpus N --sampling-gpus N"
        )
    return int(value)


def _router_tensor(value: Any) -> Any:
    from arctic_platform.integrations.tinker.router import TensorData

    array = np.asarray(value.to_numpy())
    dtype = "float32" if array.dtype.kind == "f" else "int64"
    return TensorData(dtype=dtype, data=array.reshape(-1).tolist(), shape=[int(n) for n in array.shape])


def _router_datum(datum: Any) -> Any:
    from arctic_platform.integrations.tinker.router import Datum
    from arctic_platform.integrations.tinker.router import ModelInput

    return Datum(
        model_input=ModelInput.model_validate(datum.model_input.model_dump()),
        loss_fn_inputs={key: _router_tensor(value) for key, value in datum.loss_fn_inputs.items()},
    )


def _sdk_logprobs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for output in outputs:
        payload = output["logprobs"]
        converted.append(
            {
                "logprobs": _sdk.TensorData(
                    data=payload["data"],
                    dtype=payload.get("dtype", "float32"),
                    shape=payload.get("shape"),
                )
            }
        )
    return converted


class SamplingClient:
    def __init__(self, session: TrainingClient) -> None:
        self._session = session

    async def sample_async(
        self,
        prompt: Any,
        num_samples: int = 1,
        sampling_params: Any | None = None,
        **_: Any,
    ) -> Any:
        from arctic_platform.integrations.tinker.router import ModelInput
        from arctic_platform.integrations.tinker.router import SamplingParams
        from arctic_platform.integrations.tinker.router import _model_input_to_tokens
        from arctic_platform.integrations.tinker.router import sampling_params_tinker_to_vllm

        if sampling_params is None:
            sampling_params = _sdk.SamplingParams()
        if abs(float(sampling_params.temperature) - 1.0) > 1e-9:
            raise RuntimeError("sampling temperature must be 1.0; Cortex scores training log-probs at temperature 1.0")
        router_prompt = ModelInput.model_validate(prompt.model_dump())
        tokens = _model_input_to_tokens(router_prompt)
        params = sampling_params_tinker_to_vllm(
            SamplingParams.model_validate(sampling_params.model_dump()),
            num_samples,
        )
        result = await self._session.generate(tokens, params)
        return self._sample_response(result)

    async def compute_logprobs_async(self, prompt: Any) -> list[float | None]:
        """Log-prob of each prompt token. The first entry is ``None``, matching Tinker."""
        from arctic_platform.integrations.tinker.router import ModelInput
        from arctic_platform.integrations.tinker.router import SamplingParams
        from arctic_platform.integrations.tinker.router import _model_input_to_tokens
        from arctic_platform.integrations.tinker.router import sampling_params_tinker_to_vllm

        router_prompt = ModelInput.model_validate(prompt.model_dump())
        tokens = _model_input_to_tokens(router_prompt)
        params = sampling_params_tinker_to_vllm(SamplingParams(max_tokens=1, temperature=1.0), 1)
        params["prompt_logprobs"] = 0  # vLLM: 0 is the prompt token's own log-prob
        result = await self._session.generate(tokens, params)
        prompt_logprobs = result.get("prompt_logprobs")
        if prompt_logprobs is None or len(prompt_logprobs) != len(tokens):
            raise RuntimeError(
                f"asked for prompt log-probs of {len(tokens)} tokens and the sampler returned "
                f"{None if prompt_logprobs is None else len(prompt_logprobs)}"
            )
        return list(prompt_logprobs)

    def _sample_response(self, result: dict) -> Any:
        sequences = []
        for output in result.get("outputs") or []:
            logprobs = output.get("logprobs")
            if logprobs is None:
                raise RuntimeError("sampler returned no logprobs")
            reason = "stop" if output.get("finish_reason") == "stop" else "length"
            sequences.append(
                _sdk.SampledSequence(
                    stop_reason=reason,
                    tokens_np=np.asarray(output.get("token_ids") or [], dtype=np.int64),
                    logprobs_np=np.asarray(logprobs, dtype=np.float32),
                )
            )
        return _sdk.SampleResponse(sequences=sequences)


class TrainingClient:
    """In-process training client. Work on one model is serialized, sampling is not."""

    def __init__(
        self,
        client: Any,
        handlers: dict[str, Any],
        pad_token_id: int,
        max_prompt: int,
        max_response: int,
        fixed_adam: dict[str, float] | None = None,
    ) -> None:
        self._client = client
        self._handlers = handlers
        self._pad_token_id = pad_token_id
        self._max_prompt = max_prompt
        self._max_response = max_response
        self._fixed_adam = fixed_adam
        self._lock = asyncio.Lock()
        self._have_grad = False
        self._sampler_path: str | None = None
        self._announced_checkpoint = False

    async def generate(self, tokens: list[int], params: dict) -> dict:
        return await self._handlers["generate_handler"](tokens, params)

    async def forward_backward_async(
        self, data: list[Any], loss_fn: str = "cross_entropy", loss_fn_config: dict | None = None
    ) -> _Future:
        task = asyncio.create_task(self._forward_backward(data, loss_fn, loss_fn_config))
        return _Future(task)

    async def _forward_backward(self, data: list[Any], loss_fn: str, loss_fn_config: dict | None) -> Any:
        from arctic_platform.integrations.tinker.router import _unpad_logprobs_to_loss_fn_outputs
        from arctic_platform.integrations.tinker.router import arctic_metrics_to_tinker
        from arctic_platform.integrations.tinker.router import datum_list_to_arctic_batch

        async with self._lock:
            if self._have_grad:
                raise RuntimeError(
                    "a second forward_backward before optim_step would drop the first gradient; "
                    "send the whole step in one call"
                )
            batch, row_slices = datum_list_to_arctic_batch(
                [_router_datum(datum) for datum in data],
                loss_fn,
                self._max_prompt,
                self._max_response,
                self._pad_token_id,
                forward_only=False,
                loss_fn_config=loss_fn_config,
            )
            try:
                result = await self._handlers["fwd_bwd_handler"](batch)
            except BaseException:
                self._have_grad = False
                raise
            self._have_grad = True
            logprobs = result.get("batch", {}).get("logprobs") if result.get("batch") else None
            if logprobs is not None:
                outputs = _sdk_logprobs(_unpad_logprobs_to_loss_fn_outputs(logprobs, row_slices))
            else:
                outputs = [{} for _ in data]
            return _sdk.ForwardBackwardOutput(
                loss_fn_output_type="ArrayRecord",
                loss_fn_outputs=outputs,
                metrics=arctic_metrics_to_tinker(result.get("metrics")),
            )

    async def optim_step_async(self, adam_params: Any) -> _Future:
        task = asyncio.create_task(self._optim_step(adam_params))
        return _Future(task)

    async def _optim_step(self, adam_params: Any) -> Any:
        from arctic_platform.integrations.tinker.router import AdamParams
        from arctic_platform.integrations.tinker.router import adam_params_to_optim_overrides
        from arctic_platform.integrations.tinker.router import arctic_metrics_to_tinker
        from arctic_platform.integrations.tinker.router import check_fixed_adam

        async with self._lock:
            if not self._have_grad:
                raise RuntimeError("optim_step without a successful forward_backward")
            requested = AdamParams.model_validate(adam_params.model_dump())
            if self._fixed_adam is not None:
                check_fixed_adam(self._fixed_adam, requested)
            self._have_grad = False
            result = await self._handlers["step_handler"](adam_params_to_optim_overrides(requested))
            return _sdk.OptimStepResponse(metrics=arctic_metrics_to_tinker(result.get("metrics")))

    async def save_weights_and_get_sampling_client_async(self, **_: Any) -> SamplingClient:
        async with self._lock:
            await self._handlers["sync_weights_handler"]()
        return SamplingClient(self)

    def create_sampling_client(self, model_path: str, retry_config: Any = None) -> SamplingClient:
        del retry_config
        if model_path != self._sampler_path:
            raise RuntimeError("a sampling client can only be opened from the sampler path just saved in this process")
        return SamplingClient(self)

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> _Future:
        del ttl_seconds
        task = asyncio.create_task(self._path(f"cortex://session/{name}/state"))
        return _Future(task)

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None) -> _Future:
        del ttl_seconds
        task = asyncio.create_task(self._sync_and_path(f"cortex://session/{name}/sampler"))
        return _Future(task)

    async def _path(self, path: str) -> _Path:
        if not self._announced_checkpoint:
            self._announced_checkpoint = True
            logger.warning("checkpoint %s cannot be loaded in another process", path)
        return _Path(path)

    async def _sync_and_path(self, path: str) -> _Path:
        async with self._lock:
            await self._handlers["sync_weights_handler"]()
            self._sampler_path = path
        return await self._path(path)


class ServiceClient:
    """Builds one Cortex training and sampling session. ``base_url`` is ignored."""

    def __init__(
        self,
        base_url: str | None = None,
        user_metadata: dict | None = None,
        training_gpus: int | None = None,
        sampling_gpus: int | None = None,
        teacher_sampling_gpus: int | None = None,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
        **_: Any,
    ) -> None:
        del base_url, user_metadata
        self.training_gpus = _pick(training_gpus, "training_gpus")
        self.sampling_gpus = _pick(sampling_gpus, "sampling_gpus")
        self.teacher_sampling_gpus = int(
            teacher_sampling_gpus if teacher_sampling_gpus is not None else _settings["teacher_sampling_gpus"]
        )
        if self.teacher_sampling_gpus < 1:
            raise ValueError("teacher_sampling_gpus must be at least 1")
        self.max_prompt_length = int(max_prompt_length or _settings["max_prompt_length"])
        self.max_response_length = int(max_response_length or _settings["max_response_length"])
        self._session: TrainingClient | None = None
        self._student_spec: tuple[Any, ...] | None = None
        self._teachers: dict[str, TrainingClient] = {}
        self._teacher_lock = threading.Lock()

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict | None = None,
    ) -> TrainingClient:
        del user_metadata
        spec = (base_model, rank, train_mlp, train_attn, train_unembed)
        if self._session is not None:
            if spec != self._student_spec:
                raise RuntimeError(
                    "this service already opened a training client; a second model or LoRA rank needs its own process"
                )
            return self._session
        groups = [
            name for name, enabled in (("mlp", train_mlp), ("attn", train_attn), ("unembed", train_unembed)) if enabled
        ]
        if not groups:
            raise ValueError("at least one of train_mlp, train_attn, train_unembed must be set")
        self._session = await asyncio.to_thread(self._open, base_model, rank, seed, ",".join(groups))
        self._student_spec = spec
        return self._session

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: Any = None,
    ) -> SamplingClient:
        del retry_config
        if model_path is not None:
            raise RuntimeError("loading a sampler checkpoint is not supported")
        if base_model is None:
            raise ValueError("pass base_model; checkpoint paths are not supported")
        with self._teacher_lock:
            if base_model not in self._teachers:
                self._teachers[base_model] = self._open_teacher_from_caller(base_model)
        return SamplingClient(self._teachers[base_model])

    def _open_teacher_from_caller(self, model: str) -> TrainingClient:
        # Sync call from the cookbook's running loop. Open off-thread or the loop deadlocks.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._open_teacher(model)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(self._open_teacher, model).result()

    def create_rest_client(self, *_: Any, **__: Any) -> Any:
        raise ValueError("checkpoint metadata is not available")

    def create_training_client_from_state(self, path: str, *_: Any, **__: Any) -> TrainingClient:
        _refuse_checkpoint(path)

    def create_training_client_from_state_with_optimizer(self, path: str, *_: Any, **__: Any) -> TrainingClient:
        _refuse_checkpoint(path)

    async def create_training_client_from_state_async(self, path: str, *_: Any, **__: Any) -> TrainingClient:
        _refuse_checkpoint(path)

    async def create_training_client_from_state_with_optimizer_async(
        self, path: str, *_: Any, **__: Any
    ) -> TrainingClient:
        _refuse_checkpoint(path)

    def _open_teacher(self, model: str) -> TrainingClient:
        from transformers import AutoConfig

        from arctic_platform.client import AsyncArcticRLClient
        from arctic_platform.integrations.tinker.cortex import build_handlers
        from arctic_platform.integrations.tinker.serve import TinkerServeConfig
        from arctic_platform.integrations.tinker.serve import _client_config
        from arctic_platform.integrations.tinker.serve import _isolation

        # Prompt cap fits a full student sequence. Response cap matches the student.
        cfg = TinkerServeConfig(
            model=model,
            training_gpus=0,
            sampling_gpus=self.teacher_sampling_gpus,
            max_prompt_length=self.max_prompt_length + self.max_response_length,
            max_response_length=self.max_response_length,
            lora_rank=0,
            seed=7,
        )
        model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        cfg, isolate = _isolation(cfg, model_config)
        client = AsyncArcticRLClient(_client_config(cfg))
        _sessions.append(client)
        return TrainingClient(
            client,
            build_handlers(client, isolate),
            0,
            cfg.max_prompt_length,
            cfg.max_response_length,
            fixed_adam=cfg.fixed_adam,
        )

    def _open(self, base_model: str, rank: int, seed: int | None, lora_modules: str) -> TrainingClient:
        from transformers import AutoConfig
        from transformers import AutoTokenizer

        from arctic_platform.client import AsyncArcticRLClient
        from arctic_platform.integrations.tinker.cortex import build_handlers
        from arctic_platform.integrations.tinker.serve import TinkerServeConfig
        from arctic_platform.integrations.tinker.serve import _client_config
        from arctic_platform.integrations.tinker.serve import _isolation

        cfg = TinkerServeConfig(
            model=base_model,
            training_gpus=self.training_gpus,
            sampling_gpus=self.sampling_gpus,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
            lora_rank=rank,
            lora_modules=lora_modules,
            seed=7 if seed is None else seed,
        )
        model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        cfg, isolate = _isolation(cfg, model_config)
        client = AsyncArcticRLClient(_client_config(cfg))
        _sessions.append(client)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if pad is None:
            raise RuntimeError(f"{base_model} has no pad or eos token id")
        return TrainingClient(
            client,
            build_handlers(client, isolate),
            int(pad),
            cfg.max_prompt_length,
            cfg.max_response_length,
            fixed_adam=cfg.fixed_adam,
        )


def install() -> None:
    module = sys.modules[__name__]
    sys.modules["tinker"] = module
    module.types = _sdk.types
    module.lib = _sdk.lib


install()
