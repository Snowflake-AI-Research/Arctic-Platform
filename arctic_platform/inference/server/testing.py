"""Dummy worker for CPU-only testing. Drop-in replacement for InferenceWorker."""
from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from types import SimpleNamespace

import ray
import torch

from arctic_platform.inference.server.worker import WorkerLifecycleState
from arctic_platform.inference.vllm.dense_prompt_logprobs import (
    RESULT_KEY as _DENSE_PROMPT_LOGPROBS_KEY,
    densify as _densify_prompt_logprobs,
    stage_sampling_params as _stage_dense_prompt_logprobs,
)


@ray.remote
class DummyWorker:
    """Mimics InferenceWorker without vLLM. Generates deterministic dummy outputs."""

    def __init__(self) -> None:
        self.state = WorkerLifecycleState.UNINITIALIZED

    async def initialize(
        self,
        engine_kwargs: dict[str, Any],
        extra_env: dict[str, str] | None = None,
        model_id: str | None = None,
    ) -> None:
        self.state = WorkerLifecycleState.READY
        self._model = engine_kwargs.get("model", "dummy")

    async def generate(self, prompt: str | list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        if self.state != WorkerLifecycleState.READY:
            raise RuntimeError(f"Worker not ready: state={self.state.value}")

        await asyncio.sleep(0.01)
        sampling_params = dict(sampling_params)
        return_sampled_logprobs_only = bool(sampling_params.pop("return_sampled_logprobs_only", False))
        # Mirror InferenceWorker: the opt-in moves into extra_args, and the
        # result carries tensors instead of one dict per position.
        dense_prompt_logprobs = _stage_dense_prompt_logprobs(sampling_params)

        if isinstance(prompt, list):
            text = f"dummy({len(prompt)} tokens)"
            token_ids = list(range(len(prompt), len(prompt) + 5))
            num_prompt_tokens = len(prompt)
        else:
            text = f"dummy({prompt})"
            token_ids = list(range(5))
            num_prompt_tokens = max(1, len(prompt.split()))

        result: dict[str, Any] = {
            "text": text,
            "token_ids": token_ids,
            "finish_reason": "stop",
            "prompt_len": num_prompt_tokens,
            "generation_len": len(token_ids),
            "prefix_cache_len": 0,
        }

        top_k = sampling_params.get("prompt_logprobs")
        if top_k is not None:
            if dense_prompt_logprobs:
                result[_DENSE_PROMPT_LOGPROBS_KEY] = _dummy_dense_prompt_logprobs(
                    num_prompt_tokens, int(top_k),
                )
            else:
                result["prompt_logprobs"] = _dummy_prompt_logprobs(
                    num_prompt_tokens, int(top_k),
                )

        top_k_sample = sampling_params.get("logprobs")
        if top_k_sample is not None:
            if return_sampled_logprobs_only:
                result["logprobs"] = [-0.2 for _ in token_ids]
            else:
                result["logprobs"] = _dummy_sample_logprobs(
                    len(token_ids), int(top_k_sample),
                )

        return result

    def is_healthy(self) -> bool:
        return self.state in (WorkerLifecycleState.READY, WorkerLifecycleState.SLEEPING)

    def get_state(self) -> str:
        return self.state.value

    def get_stats(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "pid": os.getpid(),
            "gpu_cache_usage": 0.0,
            "num_requests_running": 0,
            "num_requests_waiting": 0,
        }

    def drain_metrics(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "snapshots": [], "engine_totals": {}}

    def set_replica_id(self, replica_id: int) -> None:
        return None

    def pid(self) -> int:
        return os.getpid()

    async def sleep(self, level: int = 1) -> dict[str, Any]:
        self.state = WorkerLifecycleState.SLEEPING
        return {"status": "sleeping", "level": level}

    async def wake_up(self, tags: list[str] | None = None) -> dict[str, Any]:
        self.state = WorkerLifecycleState.READY
        return {"status": "ready"}

    async def sync_weights(
        self, master_addr: str, master_port: int, rank_offset: int,
        world_size: int, bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False, direct_mode: bool = False,
        reverse: bool = False,
    ) -> dict[str, Any]:
        return {"status": "done", "params_loaded": 0, "elapsed": 0.0}

    async def sync_weights_broadcast(
        self, master_addr: str, master_port: int, rank_offset: int,
        world_size: int, bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False, weight_format: str = "vllm",
    ) -> dict[str, Any]:
        return {"status": "done", "params_loaded": 0, "elapsed": 0.0}

    async def sync_lora_weights_broadcast(
        self, master_addr: str, master_port: int, rank_offset: int,
        world_size: int, lora_int_id: int, lora_name: str,
        lora_config: dict[str, Any], bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        staging: str = "cpu",
        evict_first: bool = True,
    ) -> dict[str, Any]:
        self._active_lora_int_id = lora_int_id
        self._active_lora_name = lora_name
        return {
            "status": "done", "params_loaded": 0, "elapsed": 0.0,
            "weight_format": "lora", "lora_int_id": lora_int_id,
            "lora_name": lora_name, "lora_sync_staging": staging,
            "lora_evict_first": evict_first,
        }

    async def pause_generation(
        self, mode: str = "keep", clear_cache: bool = False,
    ) -> dict[str, Any]:
        return {"status": "paused", "mode": mode, "clear_cache": clear_cache}

    async def resume_generation(self) -> dict[str, Any]:
        return {"status": "resumed"}

    async def reset_prefix_cache(
        self, timeout_s: float = 0.0, retry_interval_s: float = 0.1,
    ) -> dict[str, Any]:
        return {"status": "ok", "reset_ok": True, "attempts": 1}

    async def close_weight_sync(self) -> dict[str, Any]:
        return {"status": "ok"}

    def shutdown(self) -> None:
        self.state = WorkerLifecycleState.UNINITIALIZED


def _dummy_prompt_logprobs(
    num_tokens: int, top_k: int,
) -> list[dict[int, dict] | None]:
    """Produce deterministic dummy prompt_logprobs matching vLLM's format."""
    out: list[dict[int, dict] | None] = [None]  # first position is always None
    for pos in range(1, num_tokens):
        out.append({
            pos + k: {"logprob": -0.1 * (k + 1), "rank": k + 1}
            for k in range(top_k)
        })
    return out


def _dummy_dense_prompt_logprobs(
    num_tokens: int, top_k: int,
) -> dict[str, torch.Tensor]:
    """The dense counterpart of :func:`_dummy_prompt_logprobs`.

    Builds the ``[n-1, k+1]`` tensors vLLM would hand the patch and runs them
    through the real :func:`densify`, so the fake engine cannot drift from
    production: padding, dtypes and the position-0 markers all come from the one
    implementation. Only the numbers are invented.
    """
    scored = num_tokens - 1
    # Column 0 is the observed token, columns 1.. are the top-k -- same layout
    # LogprobsTensors uses. Values mirror _dummy_prompt_logprobs.
    token_ids = torch.zeros((scored, top_k + 1), dtype=torch.int32)
    values = torch.zeros((scored, top_k + 1), dtype=torch.float32)
    for row in range(scored):
        pos = row + 1
        token_ids[row, 0] = pos
        values[row, 0] = -0.1
        for k in range(top_k):
            token_ids[row, k + 1] = pos + k
            values[row, k + 1] = -0.1 * (k + 1)

    return _densify_prompt_logprobs(
        SimpleNamespace(logprob_token_ids=token_ids, logprobs=values),
        top_k,
    )


def _dummy_sample_logprobs(
    num_tokens: int, top_k: int,
) -> list[dict[int, dict]]:
    """Produce deterministic dummy sample logprobs."""
    return [
        {
            pos + k: {"logprob": -0.2 * (k + 1), "rank": k + 1}
            for k in range(top_k)
        }
        for pos in range(num_tokens)
    ]
