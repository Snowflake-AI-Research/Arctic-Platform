# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex dispatch shim for the legacy ``arctic_platform.rl`` entry point.

SkyRL / verl adapters build their client via::

    from arctic_platform.rl import create_arctic_rl_client
    client = create_arctic_rl_client(config, server_state)

and expect an ``async`` surface (``fwd_bwd``, ``generate``, ...). The unified
client at ``arctic_platform.client.ArcticRLClient`` is currently synchronous;
this module wraps it in async coroutines so the adapters need no patches.

Once ``AsyncArcticRLClient`` from PR #58 merges, the ``async def`` wrappers
here collapse to ``return await self._client.foo(...)`` and this file drops
back to ~100 loc of payload translation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from arctic_platform.rl.config import ArcticRLClientConfig as _LegacyConfig

logger = logging.getLogger(__name__)


# ── legacy → nested unified config ──────────────────────────────────────────
_OPT_KEEP = {"lr", "weight_decay", "betas", "beta1", "beta2", "eps", "name", "type"}


def _to_unified_config(legacy: _LegacyConfig):
    from arctic_platform.client.config import (
        ArcticRLClientConfig as U,
        CortexConfig,
        SamplingConfig,
        TrainingConfig,
    )

    tc_in = dict(legacy.training_config or {})
    opt_in = tc_in.pop("optimizer", None) or {}
    opt = {k: v for k, v in opt_in.items() if k in _OPT_KEEP}
    if opt:
        tc_in["optimizer"] = opt
    training_cfg = TrainingConfig(**tc_in) if tc_in else TrainingConfig()

    cx = CortexConfig(
        base_url=legacy.cortex_base_url or os.environ.get("ARCTIC_CORTEX_BASE_URL"),
        host=legacy.cortex_host or os.environ.get("ARCTIC_CORTEX_HOST"),
        pat_env_var=legacy.cortex_pat_env_var or os.environ.get("ARCTIC_CORTEX_PAT_ENV_VAR") or "CORTEX_PAT",
        database=legacy.cortex_database or os.environ.get("ARCTIC_CORTEX_DATABASE") or "",
        endpoint=legacy.cortex_endpoint or os.environ.get("ARCTIC_CORTEX_ENDPOINT") or "cortex-training",
        max_retries=legacy.cortex_max_retries,
        **{"schema": legacy.cortex_schema or os.environ.get("ARCTIC_CORTEX_SCHEMA") or ""},
    )

    fields: dict[str, Any] = {
        "model_name": legacy.model_name,
        "seed": legacy.seed,
        "max_seq_len": legacy.max_seq_len or 8192,
        "training_gpus": legacy.training_gpus,
        "sampling_gpus": legacy.sampling_gpus,
        "log_prob_gpus": legacy.log_prob_gpus,
        "job_ready_timeout": legacy.job_ready_timeout,
        "backend_config": cx,
        "training": training_cfg,
        "sampling": SamplingConfig(vllm=dict(legacy.vllm_config or {})),
    }
    for k in ("training_job_id", "sampling_job_id", "log_prob_job_id"):
        v = getattr(legacy, k, None)
        if v is not None:
            fields[k] = v
    return U(**fields)


# ── response normalization ──────────────────────────────────────────────────
# Cortex returns scalars flat: ``{"avg_loss", "grad_norm", "last_lr",
# "global_steps", "update_successful", "approx_kl", ...}``. verl reads
# ``response["metrics"]["loss"]`` per microbatch; SkyRL reads ``metrics.*``
# for the training-progress bar. Promote known keys into ``metrics`` and
# alias ``avg_loss → loss`` so both see the on-prem shape.
_ALIASES = {"avg_loss": "loss"}
_KEYS = ("avg_loss", "loss", "approx_kl", "importance_weight", "clip_ratio", "entropy",
         "grad_norm", "last_lr", "global_steps", "update_successful")


def _normalize(response: Any) -> Any:
    if not isinstance(response, dict):
        return response
    m = response.get("metrics")
    if not isinstance(m, dict):
        m = {}
        response["metrics"] = m
    for k in _KEYS:
        if k in response:
            m.setdefault(_ALIASES.get(k, k), response[k])
    return response


# ── shim ────────────────────────────────────────────────────────────────────
class _CortexClientShim:
    """Async facade over the sync unified client (``async`` methods that never
    ``await`` are still valid coroutines returning the sync result).
    """

    def __init__(self, legacy_config: _LegacyConfig) -> None:
        from arctic_platform.client import ArcticRLClient

        self._legacy_config = legacy_config
        self._unified_config = _to_unified_config(legacy_config)
        self._client = ArcticRLClient(self._unified_config)
        self._colocate = False

    @property
    def config(self) -> _LegacyConfig:
        return self._legacy_config

    @property
    def training_job_id(self) -> Any: return self._client.jobs.training

    @property
    def sampling_job_id(self) -> Any: return self._client.jobs.sampling

    @property
    def log_prob_job_id(self) -> Any: return self._client.jobs.log_prob

    @property
    def server_state(self) -> Any: return None

    def get_server_state(self) -> Any: return None

    def reconnect_config(self) -> _LegacyConfig:
        """Serializable config a reconnecting worker hands back to
        ``create_arctic_rl_client`` to reattach without creating new sub-jobs.
        """
        return self._legacy_config.model_copy(
            update={
                "training_job_id": self._client.jobs.training,
                "sampling_job_id": self._client.jobs.sampling,
                "log_prob_job_id": self._client.jobs.log_prob,
            }
        )

    def shutdown(self):
        # SkyRL calls this sync; verl awaits it. Do the work eagerly and
        # return an already-resolved coroutine so both call sites are happy.
        self._client.shutdown()

        async def _done() -> None:
            return None

        return _done()

    async def fwd_bwd(self, batch: dict, **legacy_kwargs: Any) -> dict:
        # Normalize SkyRL / verl fwd_bwd payloads onto Cortex's canonical
        # ``{args, kwargs, context, processing}`` shape. We deliberately do NOT
        # populate ``context.old_log_probs_shifted``: Cortex's server-side grpo
        # loss defaults ``old_log_probs = logprobs.detach()`` when absent,
        # which is the correct π_old for the single-epoch on-policy regime
        # SkyRL / verl / the ``rl_loop.py`` reference recipe all run.
        import torch

        payload = dict(batch)
        processing_in = legacy_kwargs.pop("processing", None) or payload.pop("processing", None)
        legacy_kwargs.pop("router_replay", None)
        payload.pop("router_replay", None)

        if "batch" in payload and isinstance(payload["batch"], dict):
            tensors, meta = dict(payload["batch"]), dict(payload.get("meta") or {})
        else:
            tensors = dict(payload)
            meta = dict(tensors.pop("context", None) or {})

        input_ids = tensors.get("input_ids")
        attention_mask = tensors.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise ValueError("cortex fwd_bwd requires 'input_ids' and 'attention_mask'")

        loss_mask = tensors.pop("loss_mask", None) or tensors.pop("response_mask", None) or attention_mask
        if torch.is_tensor(loss_mask):
            loss_mask = loss_mask.to(torch.bool)
        advantages = tensors.pop("advantages", None)
        if advantages is None:
            raise ValueError("cortex fwd_bwd requires 'advantages' [B, S]")
        tensors.pop("old_log_probs", None)

        kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        for k in ("position_ids", "labels"):
            if k in tensors:
                kwargs_out[k] = tensors[k]

        proc_config = dict((processing_in or {}).get("config") or {})
        proc_config.setdefault("eps_clip", 0.2)
        proc_config.setdefault("prox_logp_method", "recompute")
        proc_config.setdefault("dp_size", int(self._unified_config.training_gpus or 1))
        for k in ("batch_num_tokens", "global_batch_size"):
            if k not in proc_config and k in meta:
                proc_config[k] = int(meta[k])

        return _normalize(self._client.fwd_bwd({
            "args": (),
            "kwargs": kwargs_out,
            "context": {"input_ids": input_ids, "advantages": advantages, "loss_mask": loss_mask},
            "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
        }))

    async def fwd_no_grad(self, batch: dict, **legacy_kwargs: Any) -> dict:
        # Cortex has no ``/forward`` endpoint. ``fwd_bwd`` above drops
        # ``old_log_probs_shifted`` so the server grpo loss defaults to
        # ``logprobs.detach()`` — so π_old ≡ π_new and zero placeholders are
        # correct. Shape ``[B, T_full]`` matches the on-prem ``/forward``
        # layout, so both SkyRL's response-only rebuild and verl's
        # ``make_njt`` slice into a valid tensor.
        import torch

        b_data = batch.get("batch") if isinstance(batch, dict) else None
        if not isinstance(b_data, dict):
            b_data = batch if isinstance(batch, dict) else {}
        ids = b_data.get("input_ids")
        b, t = (int(ids.shape[0]), int(ids.shape[-1])) if torch.is_tensor(ids) else (1, 1)
        z = torch.zeros((b, max(t, 1)), dtype=torch.float32)
        return {"batch": {"logprobs": z, "log_probs": z, "entropy": z, "entropies": z}}

    async def step(self, learning_rate: float | None = None) -> dict:
        return _normalize(self._client.step(learning_rate=learning_rate))

    async def save_checkpoint(self, stage_info: dict | None = None, path: str | None = None) -> dict:
        cid = None
        if isinstance(stage_info, dict):
            cid = stage_info.get("checkpoint_id") or stage_info.get("id")
        return self._client.save_checkpoint(checkpoint_id=cid)

    async def save_weights(self, path: str) -> dict:
        logger.warning("save_weights is a no-op on Cortex (path=%s ignored)", path)
        return {}

    async def generate(self, prompts, sampling_params=None, **kwargs) -> list:
        return self._client.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            routing_key=kwargs.pop("routing_key", None),
            strict=kwargs.pop("strict", False),
        )

    async def sync_weights(self, cuda_ipc: bool = False, low_memory: bool = False) -> dict:
        return self._client.sync_weights()

    async def reset_prefix_cache(self, drain: bool = True, timeout_s: float = 60.0) -> dict:
        return self._client.reset_prefix_cache(drain=drain, timeout_s=timeout_s)

    # colocation lifecycle — Cortex sub-jobs live in separate placements, so no-ops.
    async def wake_inference(self, **_): return {}
    async def sleep_inference(self, **_): return {}
    async def wake_training(self, **_): return {}
    async def sleep_training(self, **_): return {}
    async def wake_log_prob(self, **_): return {}
    async def sleep_log_prob(self, **_): return {}
    async def empty_training_cache(self, **_): return {}
    async def weight_norm(self, **_): return {}

    async def log_probs(self, batch: dict, **_) -> dict:
        raise NotImplementedError("Cortex has no log-probs endpoint; disable KL/ref-model recipes.")


def create_cortex_client(config: _LegacyConfig) -> _CortexClientShim:
    return _CortexClientShim(config)
