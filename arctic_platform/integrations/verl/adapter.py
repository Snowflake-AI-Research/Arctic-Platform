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

# Originally lived at ``verl/workers/remote_client/arctic_rl.py`` in
# Snowflake-AI-Research/verl (fork of verl-project/verl, Apache-2.0).
# Moved here so the Arctic adapter can be plugged into upstream verl via
# ``VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register``
# without living in the verl source tree.
"""Arctic-RL adapter: implements verl's :class:`RemoteBackend` on top of
:mod:`arctic_platform.rl`.

Registered under the name ``"arctic"`` by
``@RemoteBackendRegistry.register("arctic")`` at import time; the plugin
entry point (:mod:`arctic_platform.integrations.verl.register`) imports
this module for that side effect, and pairs it with the matching per-
backend worker and rollout replica.
"""

import math
import os
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.remote_backend.base import RemoteBackend
from verl.remote_backend.base import RemoteBackendRegistry

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import AsyncArcticRLClient
from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.client import SamplingConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.client.config import cortex_job_packs
from arctic_platform.common.balance_metadata import prompt_rows_for_balance
from arctic_platform.common.balance_metadata import stamp_client_balance
from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload
from arctic_platform.integrations._cortex_shared import to_cortex_fwd_payload
from arctic_platform.rl.ray_server import ArcticRLRayServerState

_ARCTIC_METRIC_REDUCTION_FN = {
    "actor/pg_clipfrac_lower": np.mean,
    "actor/pg_clipfrac": np.mean,
    "actor/pg_loss": np.mean,
    "actor/ppo_kl": np.mean,
    "grad_norm": np.mean,
    "kl_coef": np.mean,
    "kl_loss": np.mean,
}

# Cortex step answers flat; these keys are bookkeeping, not metrics to average.
_CORTEX_STEP_NON_METRICS = ("job_id", "global_steps", "update_successful", "bandwidth", "metrics")


def _uses_cortex(comms_cfg) -> bool:
    """``comms.protocol``: ``ray`` (or unset) runs on-prem; ``cortex`` or ``http`` runs on Cortex."""
    protocol = str(comms_cfg.get("protocol", None) or "ray").lower()
    if protocol == "ray":
        return False
    if protocol in ("cortex", "http"):
        return True
    raise ValueError(
        f"remote_backend.comms.protocol must be 'ray' (on-prem) or 'cortex' / 'http' (Cortex), got {protocol!r}"
    )


def _scalar(value):
    """A replicated meta value as a plain Python scalar. The Cortex wire drops tensor-valued meta."""
    value = getattr(value, "data", value)
    if torch.is_tensor(value):
        return value.item()
    return value


def _positive_temperature(value) -> float:
    """Training temperature on the Cortex wire: finite and greater than zero."""
    value = _scalar(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"temperature must be a finite number greater than 0, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"temperature must be a finite number greater than 0, got {number!r}")
    return number


def _rollout_correction_enabled(algorithm) -> bool:
    """True when VeRL will build rollout-importance weights or take the bypass path.

    The default block is present with ``rollout_is`` and ``rollout_rs`` null and ``bypass_mode`` false.
    That block is off. A set aggregation or bypass mode is what puts ``rollout_is_weights`` on the update.
    """
    correction = algorithm.get("rollout_correction", None) if algorithm is not None else None
    if not correction:
        return False
    if correction.get("bypass_mode", False):
        return True
    return any(correction.get(key, None) not in (None, False, "") for key in ("rollout_is", "rollout_rs"))


def _require_cortex_step_applied(step_response: dict) -> None:
    """Refuse a Cortex step whose optimizer update did not apply.

    DSS reports ``update_successful`` as 1 when the global grad norm is finite and 0 otherwise.
    Dropping the flag would let a failed step look like a normal metric merge.
    """
    nested = step_response.get("metrics") if isinstance(step_response.get("metrics"), dict) else {}
    if "update_successful" in step_response:
        value = step_response["update_successful"]
    elif "update_successful" in nested:
        value = nested["update_successful"]
    else:
        raise RuntimeError(
            "Cortex step reply has no update_successful; refusing to report the optimizer step as applied. "
            f"got keys={sorted(step_response)}"
        )
    applied = value is True or (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 1)
    if not applied:
        raise RuntimeError(
            f"Cortex optimizer step was not applied (update_successful={value!r}); "
            "the step reply is not merged into training metrics"
        )


def _cortex_logprobs(result) -> torch.Tensor:
    logprobs = result.get("logprobs") if isinstance(result, dict) else None
    if logprobs is None and isinstance(result, dict) and isinstance(result.get("batch"), dict):
        logprobs = result["batch"].get("logprobs", result["batch"].get("log_probs"))
    if logprobs is None:
        keys = list(result) if isinstance(result, dict) else type(result).__name__
        raise RuntimeError(f"Cortex forward returned no logprobs; got keys={keys}")
    return torch.as_tensor(logprobs, dtype=torch.float32)


def _left_aligned_to_client_layout(values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Move each row's leading span back onto its attended columns.

    A packed job answers on the left-aligned rows it was sent. ``make_njt`` reads row ``b`` from its first
    attended column, and the client layout keeps each row's tokens contiguous.
    """
    out = torch.zeros(attention_mask.shape, dtype=values.dtype)
    lengths = attention_mask.long().sum(dim=1)
    starts = attention_mask.long().argmax(dim=1)
    for b in range(attention_mask.shape[0]):
        start, length = int(starts[b]), int(lengths[b])
        out[b, start : start + length] = values[b, :length]
    return out


def _merge_cortex_update_metrics(fwd_bwd_response: dict, step_response: dict) -> dict:
    """On-prem's merged ``step.metrics`` + ``fwd_bwd.metrics``, rebuilt from Cortex replies.

    Cortex fwd_bwd answers ``{avg_loss, metrics}``; step answers flat ``{last_lr, grad_norm, ...}``.
    Step can also carry opt-in diagnostics (``peak_memory`` dicts, gradient samples, parameter names)
    that the per-key ``np.mean`` reduction cannot average, so only numeric step values are kept.
    """
    _require_cortex_step_applied(step_response)
    metrics: dict = {}
    if isinstance(step_response.get("metrics"), dict):
        metrics.update({key: value for key, value in step_response["metrics"].items() if key != "update_successful"})
    for key, value in step_response.items():
        if key in _CORTEX_STEP_NON_METRICS or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[key] = value
    metrics.update(fwd_bwd_response.get("metrics") or {})
    if "loss" not in metrics:
        loss = fwd_bwd_response.get("loss", fwd_bwd_response.get("avg_loss"))
        if loss is None:
            raise RuntimeError(f"Cortex fwd_bwd reply carries no loss; got keys={sorted(fwd_bwd_response)}")
        metrics["loss"] = loss
    return metrics


def _no_padding_2_padding_prompt_response(tensor: torch.Tensor, data, pad_token_id):
    """Convert a jagged tensor into a left-padded prompt + right-padded
    response of shape ``[bsz, max_prompt_len + max_response_len]`` —
    the wire format Arctic expects. Arctic-specific; kept alongside the
    Arctic adapter in this plugin package so generic verl code stays
    untouched.
    """
    import torch.nn.functional as F
    from verl.utils import tensordict_utils as tu

    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]
    attention_mask = data["attention_mask"]

    max_prompt_len = tu.get_non_tensor_data(data=data, key="max_prompt_len", default=-1)
    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_prompt_len < 0:
            max_prompt_len = prompt_lens.max().item()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_prompt_len = prompt_ids.shape[1]
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0], f"{sequence_offsets[-1].item()} != {values.shape[0]}"

    rows = []
    for prompt_len, resp_len, seq_offset in zip(prompt_lens, response_lens, sequence_offsets, strict=True):
        prompt_pad = max_prompt_len - prompt_len
        response_pad = max_response_len - resp_len
        prompt = values[seq_offset - prompt_len - resp_len : seq_offset - resp_len]
        response = values[seq_offset - resp_len : seq_offset]
        prompt_padded_left = F.pad(prompt, (prompt_pad, 0), value=pad_token_id)
        response_padded_right = F.pad(response, (0, response_pad), value=pad_token_id)
        rows.append(torch.cat((prompt_padded_left, response_padded_right)))

    return torch.stack(rows, dim=0), max_prompt_len, max_response_len


def _prepare_padded_arctic_batch_dict(data, pad_token_id, *, drop_position_ids: bool = False):
    """Convert verl's nested-jagged ``input_ids`` / ``position_ids`` into the dense padded shape Arctic expects on the
    wire, and report the max prompt / response lengths. With ``drop_position_ids=True`` the server reconstructs
    position_ids from attention_mask (set False for mrope / 3D rope)."""
    input_ids = data["input_ids"]
    input_ids, max_prompt_len, max_response_len = _no_padding_2_padding_prompt_response(
        tensor=input_ids, data=data, pad_token_id=pad_token_id
    )

    batch_dict = dict(
        input_ids=input_ids,
        attention_mask=data["attention_mask"],
        prompts=prompt_rows_for_balance(data.get("prompts"), input_ids, max_prompt_len),
    )
    if not drop_position_ids:
        position_ids, _, _ = _no_padding_2_padding_prompt_response(
            tensor=data["position_ids"], data=data, pad_token_id=0
        )
        batch_dict["position_ids"] = position_ids
    return batch_dict, max_prompt_len, max_response_len


@RemoteBackendRegistry.register("arctic")
class ArcticRLClientWrapper(RemoteBackend):
    """Arctic backend: a thin wrapper around ArcticRL's ``AsyncArcticRLClient``.

    Implements the generic :class:`verl.remote_backend.RemoteBackend`
    interface. Registered under the name ``"arctic"`` so
    :class:`verl.remote_backend.trainer.RemoteBackendTrainer` can resolve
    it via ``config.trainer.remote_backend = "arctic"``.
    """

    def __init__(
        self,
        config,
        reconnect_job_config: dict = None,
        rl_server_state: ArcticRLRayServerState = None,
    ):
        self.config = config
        # Per-backend yaml is loaded flat at `config.remote_backend`; the file
        # name (arctic.yaml) names the backend, so no extra nesting is needed.
        self._backend_config = config.remote_backend
        self._client = None
        _train_cfg = self._backend_config.train
        _logits_cfg = _train_cfg.logits
        self.zorro_train_enable = _train_cfg.zorro_train.enable
        self.zorro_train_max_rollouts = _train_cfg.zorro_train.max_rollouts
        # Set False for non-arange position_ids (mrope / 3D rope).
        self.drop_position_ids = self._backend_config.comms.get("drop_position_ids", True)
        self.logits_optimization = _logits_cfg.get("optimization", "none")
        self.logits_optimization_peak_mem_size_in_gib = _logits_cfg.get("optimization_peak_mem_size_in_gib", 4)
        self.logits_compute_in_fp32 = _logits_cfg.get("compute_in_fp32", False)
        self.logits_compute_from_fp32_inputs = _logits_cfg.get("compute_from_fp32_inputs", False)
        # False keeps arrival order. True balances sample lengths, or whole ZoRRo prompt groups.
        if _train_cfg.get("balance_batch", None) is not None:
            self.balance_batch = bool(_train_cfg.get("balance_batch"))
        else:
            self.balance_batch = True
        # Sent as ds_worker_config.pack. A Cortex training job reads it; on-prem always runs unpacked.
        self.pack = bool(_train_cfg.get("pack", True))
        self.use_cortex = _uses_cortex(self._backend_config.comms)
        # On-prem rebuilds dropped position_ids in the Ray splitter. Cortex
        # never does, and ZoRRo dereferences the tensor, so keep it on the wire.
        if self.use_cortex:
            self.drop_position_ids = False
        self._check_train_options()
        if self.use_cortex:
            self._check_cortex_options()
        # CUDA-IPC bypasses NCCL for weight sync; only valid when colocate=True.
        self.cuda_ipc_weight_sync = self._backend_config.weight_sync.cuda_ipc
        self.low_memory_weight_sync = self._backend_config.weight_sync.low_memory
        self.use_liger = self.config.actor_rollout_ref.model.use_liger
        self._max_token_len_per_gpu = self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.actor_rollout_ref.model.path)
        self._client = self._initialize_client(reconnect_job_config, rl_server_state)

    # ------------------------------------------------------------------ #
    # RemoteBackend interface
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        main_config,
        *,
        handle: dict | None = None,
    ) -> "ArcticRLClientWrapper":
        """Sole public constructor.

        * ``handle=None`` (driver path): build a fresh ``AsyncArcticRLClient``
          and the underlying Ray actors.
        * ``handle=<reconnect_handle()>``: re-attach to the existing
          driver-side instance (used by forwarder workers and
          ``ArcticReplica``).
        """
        if handle is None:
            return cls(config=main_config)
        return cls(
            config=main_config,
            reconnect_job_config=handle["rl_client_reconnect_config"],
            rl_server_state=handle["rl_server_state"],
        )

    def reconnect_handle(self) -> dict:
        """Serializable bundle the forwarder workers / rollout replicas
        pass back to :meth:`from_config` (as ``handle=...``) to re-attach
        to this same backend instance.

        Ray transport shares an in-process ``server_state`` actor handle.
        Cortex reattaches by the job ids in the config alone.
        """
        return {
            "rl_client_reconnect_config": self.reconnect_config(),
            "rl_server_state": None if self.use_cortex else self._client.get_server_state(),
        }

    # Parallelism contract --------------------------------------------- #

    def requires_single_forwarder(self) -> bool:
        # Arctic owns its own training/sampling parallelism; the verl-side
        # worker group must be a single forwarder.
        return True

    # ------------------------------------------------------------------ #
    # Core RL ops — called by `ArcticRLActorRolloutRefWorker`
    # ------------------------------------------------------------------ #
    # Build Arctic's payload shape (dense padded batch, ``meta`` dict,
    # ``processing`` pipeline) and dispatch through the private
    # `_send_*` wire helpers below.

    async def compute_log_prob(
        self,
        data,
        *,
        ref: bool,
        calculate_entropy: bool,
        rollout_n: int,
        temperature,
        pad_token_id: int,
    ) -> dict:
        batch, max_prompt_len, max_response_len = _prepare_padded_arctic_batch_dict(
            data, pad_token_id, drop_position_ids=self.drop_position_ids
        )
        meta = dict(
            zorro_train_enable=self.zorro_train_enable,
            zorro_train_max_rollouts=self.zorro_train_max_rollouts,
            rollout_n=rollout_n,
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            max_token_len_per_gpu=self._max_token_len_per_gpu,
            temperature=data["temperature"],
            calculate_entropy=calculate_entropy,
            pad_token_id=pad_token_id,
            drop_position_ids=self.drop_position_ids,
            logits_optimization=self.logits_optimization,
            logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
            logits_compute_in_fp32=self.logits_compute_in_fp32,
        )
        # Log-probs are read by position, so this call keeps arrival order.
        stamp_client_balance(
            meta,
            balance_batch=False,
            zorro_train_enable=self.zorro_train_enable,
            rollout_n=rollout_n,
        )
        payload = dict(batch=batch, meta=meta)

        if self.use_cortex:
            if ref:
                raise NotImplementedError(
                    "Cortex cannot serve reference-model log-probs: it has no reference forward, and zeros would "
                    "silently corrupt the KL term. Set actor_rollout_ref.actor.use_kl_loss=False and "
                    "algorithm.use_kl_in_reward=False."
                )
            response = await self._send_cortex_compute_log_prob(payload)
        else:
            response = await (
                self._send_compute_ref_log_prob(payload) if ref else self._send_compute_log_prob(payload)
            )

        model_output = {"log_probs": response["batch"]["log_probs"]}
        if calculate_entropy and "entropy" in response["batch"]:
            model_output["entropy"] = response["batch"]["entropy"]

        metrics = dict(response.get("metrics", {}))
        return {"model_output": model_output, "metrics": metrics}

    async def update_actor(
        self,
        data,
        *,
        actor_config,
        pad_token_id: int,
        rollout_n: int,
        temperature,
    ) -> dict:
        import json

        global_token_num = data["input_ids"].offsets().diff().tolist()

        input_ids, max_prompt_len, max_response_len = _no_padding_2_padding_prompt_response(
            tensor=data["input_ids"], data=data, pad_token_id=pad_token_id
        )

        prompts = prompt_rows_for_balance(data.get("prompts"), input_ids, max_prompt_len)
        responses = data["responses"]
        if not torch.is_tensor(responses) or responses.is_nested or responses.ndim != 2:
            responses = input_ids[:, max_prompt_len:]
        batch = dict(
            input_ids=input_ids,
            attention_mask=data["attention_mask"],
            prompts=prompts,
            responses=responses,
            response_mask=data["response_mask"],
            old_log_probs=data["old_log_probs"],
            advantages=data["advantages"],
        )
        if not self.drop_position_ids:
            position_ids, _, _ = _no_padding_2_padding_prompt_response(
                tensor=data["position_ids"], data=data, pad_token_id=0
            )
            batch["position_ids"] = position_ids
        if actor_config.use_kl_loss:
            batch["ref_log_prob"] = data["ref_log_prob"]
        # Batch-dim: must live in ``batch`` so ``_split_batch`` shards it with
        # advantages / logprobs. In ``meta`` it is replicated to every DP rank.
        if data.get("rollout_is_weights") is not None:
            batch["rollout_is_weights"] = data["rollout_is_weights"]

        per_step_global_bsz = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * rollout_n
        meta = dict(
            zorro_train_enable=self.zorro_train_enable,
            zorro_train_max_rollouts=self.zorro_train_max_rollouts,
            rollout_n=rollout_n,
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            max_token_len_per_gpu=self._max_token_len_per_gpu,
            temperature=data["temperature"],
            calculate_entropy=actor_config.calculate_entropy,
            pad_token_id=pad_token_id,
            drop_position_ids=self.drop_position_ids,
            logits_optimization=self.logits_optimization,
            logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
            logits_compute_in_fp32=self.logits_compute_in_fp32,
        )

        def _safe_serialize(obj):
            return json.loads(json.dumps(obj, default=lambda o: None))

        meta["actor_config"] = _safe_serialize(vars(actor_config))
        # `policy_loss` may be None on configs that never wired up a custom
        # loss (e.g. early test/dev configs); send an empty dict instead of
        # crashing in `vars(None)`.
        policy_loss = getattr(actor_config, "policy_loss", None)
        meta["policy_loss_config"] = _safe_serialize(vars(policy_loss)) if policy_loss is not None else {}

        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        assert ppo_epochs >= 1, f"ppo_epochs must be >= 1, got {ppo_epochs}"
        num_minibatches = self.config.data.train_batch_size // self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        assert num_minibatches >= 1, (
            f"data.train_batch_size ({self.config.data.train_batch_size}) must be >= "
            f"actor.ppo_mini_batch_size ({self.config.actor_rollout_ref.actor.ppo_mini_batch_size})"
        )
        total_rows = input_ids.shape[0]
        assert total_rows % num_minibatches == 0, (
            f"DP-shard rows {total_rows} not divisible by num_minibatches {num_minibatches} "
            f"(data.train_batch_size={self.config.data.train_batch_size}, "
            f"actor.ppo_mini_batch_size={self.config.actor_rollout_ref.actor.ppo_mini_batch_size})"
        )
        chunk_rows = total_rows // num_minibatches

        agg_metrics: dict = {}
        loss_list: list = []

        for _epoch_idx in range(ppo_epochs):
            for mb_idx in range(num_minibatches):
                lo = mb_idx * chunk_rows
                hi = lo + chunk_rows

                batch_chunk = {k: v[lo:hi] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                meta_chunk = dict(meta)
                meta_chunk["global_batch_size"] = per_step_global_bsz
                meta_chunk["batch_num_tokens"] = batch_chunk["response_mask"].sum()
                stamp_client_balance(
                    meta_chunk,
                    balance_batch=self.balance_batch,
                    zorro_train_enable=self.zorro_train_enable,
                    rollout_n=rollout_n,
                    prompts=batch_chunk.get("prompts"),
                    input_ids=batch_chunk["input_ids"],
                    max_prompt_len=max_prompt_len,
                )

                payload = dict(batch=batch_chunk, meta=meta_chunk)
                response = await self._send_update_actor(payload)

                mb_metrics = dict(response["metrics"])
                loss_list.append(mb_metrics.pop("loss"))
                for k, v in mb_metrics.items():
                    agg_metrics.setdefault(k, []).append(v)

        last_lr = agg_metrics.pop("last_lr", None)

        metrics: dict = {}
        for k, v in agg_metrics.items():
            if k.startswith("mtp_losses"):
                flat = [sub[0] for sub in v]
                metrics[k] = sum(flat) / len(flat)
                continue
            reduction_fn = _ARCTIC_METRIC_REDUCTION_FN.get(k, np.mean)
            reduced = reduction_fn(v)
            metrics[k] = float(reduced) if np.ndim(reduced) == 0 else reduced

        if last_lr is not None:
            metrics["lr"] = last_lr[-1]

        loss = float(np.mean(loss_list))

        return {
            "loss": loss,
            "metrics": metrics,
            "global_token_num": global_token_num,
        }

    def _check_train_options(self) -> None:
        # verl's own trainer.balance_batch runs on the driver with dp_size=1 (single forwarder), where it only
        # sorts rows by length. Arrival order is then already gone before Arctic sees the batch.
        if not self.balance_batch and self.config.trainer.get("balance_batch", False):
            raise ValueError(
                "remote_backend.train.balance_batch=False asks Arctic for arrival order, but "
                "trainer.balance_batch=True has already reordered the batch on the verl driver. "
                "Set trainer.balance_batch=False."
            )
        if self.logits_optimization == "memory" and not self.zorro_train_enable:
            raise ValueError(
                "remote_backend.train.logits.optimization=memory requires remote_backend.train.zorro_train.enable=True; "
                "the server rejects it on the first non-ZoRRo step. Use none or compute."
            )

    def _check_cortex_options(self) -> None:
        backend = self._backend_config
        actor = self.config.actor_rollout_ref.actor
        problems = []
        if backend.get("colocate", True):
            problems.append("remote_backend.colocate=False (Cortex runs training and sampling as separate sub-jobs)")
        if backend.weight_sync.cuda_ipc:
            problems.append("remote_backend.weight_sync.cuda_ipc=False (sub-jobs sync weights over NCCL)")
        if backend.get("log_prob_gpus", 0):
            problems.append(
                "remote_backend.log_prob_gpus=0 (Cortex builds a vLLM log-prob sub-job, not this adapter's "
                "DeepSpeed one)"
            )
        if actor.get("use_kl_loss", False) or self.config.algorithm.get("use_kl_in_reward", False):
            problems.append(
                "actor_rollout_ref.actor.use_kl_loss=False and algorithm.use_kl_in_reward=False (no reference model)"
            )
        entropy_coeff = actor.get("entropy_coeff", 0)
        if isinstance(entropy_coeff, bool) or not isinstance(entropy_coeff, (int, float)) or float(entropy_coeff) != 0:
            problems.append(
                "actor_rollout_ref.actor.entropy_coeff=0 (the Cortex forward returns log-probs and no entropy)"
            )
        if _rollout_correction_enabled(self.config.algorithm):
            problems.append(
                "algorithm.rollout_correction.rollout_is=null, rollout_rs=null, and bypass_mode=False "
                "(the Cortex verl_grpo wire carries no rollout_is_weights)"
            )
        rollout = backend.get("rollout", None) or {}
        if rollout.get("zorro_inference", {}).get("enable", False) or rollout.get("speculative_decoding", {}).get(
            "model", ""
        ):
            problems.append(
                "remote_backend.rollout.zorro_inference.enable=False and speculative_decoding.model='' "
                "(not wired for Cortex sampling)"
            )
        if problems:
            raise ValueError("remote_backend.comms.protocol selects Cortex, which needs: " + "; ".join(problems))

    def _create_ds_config(self, n_gpus: int, training: bool = True) -> dict[str, Any]:
        # Pass the entire `remote_backend.train.deepspeed` YAML block through to the DS engine (torch_autocast/communication_data_type/data_types/zero_optimization/log_level/...) and merge in per-step batch sizing. Matches recipe/rl-correctness arctic_rl_client._create_ds_config.
        actor_cfg = self.config.actor_rollout_ref.actor
        data_cfg = self.config.data
        rollout_n = self.config.actor_rollout_ref.rollout.n
        deepspeed_config = OmegaConf.to_container(self._backend_config.train.deepspeed, resolve=True)

        micro_batch_size = actor_cfg.ppo_micro_batch_size_per_gpu or 1
        # DS train_batch_size = samples consumed per optimizer step.
        # We do one optimizer step per PPO mini-batch (matching verl baseline's
        # dp_actor.update_policy loop), not per global batch, so the DS engine's
        # gradient-accumulation gating must reflect one mini-batch, not BSZ.
        global_batch_size = data_cfg.train_batch_size * rollout_n
        train_batch_size = actor_cfg.ppo_mini_batch_size * rollout_n
        assert global_batch_size % train_batch_size == 0, (
            f"data.train_batch_size ({data_cfg.train_batch_size}) must be divisible by "
            f"actor.ppo_mini_batch_size ({actor_cfg.ppo_mini_batch_size}); "
            f"got global={global_batch_size}, per-step={train_batch_size}"
        )
        # The server cuts each rank's rows into micro-batches in order. A ZoRRo micro-batch that holds part of a
        # prompt group fails in the response unpad with a shape mismatch.
        if self.zorro_train_enable and micro_batch_size % rollout_n != 0:
            raise ValueError(
                f"ZoRRo micro-batch (actor.ppo_micro_batch_size_per_gpu={micro_batch_size}) must be a multiple of "
                f"actor_rollout_ref.rollout.n ({rollout_n}) so each micro-batch holds whole prompt groups"
            )
        if train_batch_size % (micro_batch_size * n_gpus) != 0:
            raise ValueError(
                f"per-step batch {train_batch_size} (actor.ppo_mini_batch_size x rollout.n) must be divisible by "
                f"actor.ppo_micro_batch_size_per_gpu ({micro_batch_size}) x {n_gpus} GPUs"
            )
        grad_accum_steps = max(1, train_batch_size // (micro_batch_size * n_gpus))
        train_seq_parallel_size = actor_cfg.fsdp_config.get("ulysses_sequence_parallel_size", 1)

        ds_config = deepcopy(deepspeed_config)
        ds_config.update(
            {
                "train_micro_batch_size_per_gpu": micro_batch_size,
                "train_batch_size": train_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "sequence_parallel_size": train_seq_parallel_size,
            }
        )
        if not training:
            ds_config.pop("data_types", None)

        return ds_config

    def reconnect_config(self):
        return self._client.reconnect_config()

    def get_server_state(self):
        return self._client.get_server_state()

    def _create_ds_worker_config(self):
        attn_implementation = self.config.actor_rollout_ref.model.override_config.get(
            "attn_implementation", "flash_attention_2"
        )
        ds_worker_config = dict(
            use_liger=self.use_liger,
            enable_gradient_checkpointing=self.config.actor_rollout_ref.model.enable_gradient_checkpointing,
            attn_implementation=attn_implementation,
        )

        if self.zorro_train_enable:
            # XXX: can't find where it's configured
            use_unpad = True

            ds_worker_config.update(
                zorro_train_enable=True,
                response_len=self.config.data.max_response_length,
                # Matches recipe/rl-correctness: read per-GPU token budget
                # from actor.ppo_max_token_len_per_gpu (the budget the train
                # engine sees), not rollout.max_num_batched_tokens.
                max_token_len=self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
                rollout_n=self.config.actor_rollout_ref.rollout.n,
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                use_unpad=use_unpad,
                # Server reads these from ds_worker_config (arctic_platform/rl/deepspeed_worker.py), not per-call meta.
                logits_optimization=self.logits_optimization,
                logits_optimization_peak_mem_size_in_gib=self.logits_optimization_peak_mem_size_in_gib,
                logits_compute_from_fp32_inputs=self.logits_compute_from_fp32_inputs,
                logits_compute_in_fp32=self.logits_compute_in_fp32,
            )
        if not self.pack:
            ds_worker_config["pack"] = False

        return ds_worker_config

    def _initialize_client(
        self,
        reconnect_job_config: dict = None,
        rl_server_state: ArcticRLRayServerState = None,
    ):
        if rl_server_state is not None:
            # Reattach to the driver's already-running in-process Ray server.
            return AsyncArcticRLClient(reconnect_job_config, server_state=rl_server_state)

        if reconnect_job_config is not None:
            return AsyncArcticRLClient(reconnect_job_config)

        model_name = self.config.actor_rollout_ref.model.path
        n_training_gpus = self._backend_config.get("training_gpus", self.config.trainer.n_gpus_per_node)
        n_sampling_gpus = self._backend_config.get("sampling_gpus", self.config.trainer.n_gpus_per_node)
        n_log_prob_gpus = self._backend_config.get("log_prob_gpus", self.config.trainer.n_gpus_per_node)
        colocate = self._backend_config.get("colocate", True)
        attn_implementation = self.config.actor_rollout_ref.model.override_config.get("attn_implementation", "eager")

        actor_cfg = self.config.actor_rollout_ref.actor
        optim_cfg = actor_cfg.optim
        data_cfg = self.config.data

        max_length = data_cfg.max_prompt_length + data_cfg.max_response_length

        # LR-scheduler horizon = global_steps * ppo_epochs * num_minibatches
        # (matches recipe/rl-correctness arctic_rl_client.py). verl injects
        # actor.optim.total_training_steps after dataloader build; fall back
        # to trainer.total_training_steps for early-construction paths.
        global_training_steps = optim_cfg.get("total_training_steps", None)
        if not global_training_steps or int(global_training_steps) <= 0:
            global_training_steps = self.config.trainer.get("total_training_steps", None)
        assert global_training_steps and int(global_training_steps) > 0, (
            "total_training_steps not populated before Arctic client build; expected "
            "ray_trainer._create_dataloader to inject actor.optim.total_training_steps"
        )
        num_minibatches = data_cfg.train_batch_size // actor_cfg.ppo_mini_batch_size
        opt_steps_per_global_step = actor_cfg.ppo_epochs * num_minibatches
        training_horizon = int(global_training_steps) * opt_steps_per_global_step
        lr_scheduler_type = optim_cfg.get("lr_scheduler_type", "constant")
        print(
            f"[ArcticRLClientWrapper] lr_scheduler: type={lr_scheduler_type} "
            f"training_horizon={training_horizon} (global_steps={int(global_training_steps)} "
            f"x ppo_epochs={actor_cfg.ppo_epochs} x num_minibatches={num_minibatches}); "
            f"warmup_steps={round(optim_cfg.lr_warmup_steps_ratio * training_horizon)}",
            flush=True,
        )

        rollout_cfg = self.config.actor_rollout_ref.rollout
        vllm_config = {
            "tensor_parallel_size": self._backend_config.sampling_tp_size,
            "gpu_memory_utilization": rollout_cfg.gpu_memory_utilization,
            "max_model_len": rollout_cfg.get("max_model_len") or max_length,
            "max_num_seqs": rollout_cfg.max_num_seqs,
            "enforce_eager": rollout_cfg.enforce_eager,
            "enable_chunked_prefill": rollout_cfg.enable_chunked_prefill,
        }
        if rollout_cfg.get("quantization"):
            vllm_config["quantization"] = rollout_cfg.quantization

        # Arctic inference signals (FCA / speculative decoding) are NOT vLLM
        # engine args: the server (arctic_platform.rl) expands this block via
        # `parse_arctic_inference_rollout` and treats None/empty as "Arctic
        # inference off". Pass the whole `remote_backend.rollout` sub-config
        # through; the server keys on `zorro_inference.enable` /
        # `speculative_decoding.model`.
        rollout_inference_cfg = self._backend_config.get("rollout", None)
        arctic_inference_config = (
            OmegaConf.to_container(rollout_inference_cfg, resolve=True) if rollout_inference_cfg is not None else None
        )

        # Forward grad_clip to the DeepSpeed engine so it clips global grad-norm
        # to the same threshold verl applies in the FSDP path (otherwise DS
        # defaults to no clipping -> trajectory divergence vs verl baseline).
        # Fold optimizer + LR scheduler into ds_config (canonical nested shape
        # from #59: DeepSpeed owns these knobs inside ds_config, not a separate
        # training_config wire field).
        ds_config = self._create_ds_config(n_training_gpus)
        opt_params = {
            "lr": optim_cfg.lr,
            "weight_decay": optim_cfg.weight_decay,
            "betas": list(optim_cfg.betas),
            "eps": 1e-8,
        }
        ds_config["optimizer"] = {"type": "AdamW", "params": opt_params}
        grad_clip = optim_cfg.get("clip_grad", None)
        if grad_clip is None:
            grad_clip = actor_cfg.get("grad_clip", None)
        if grad_clip is not None and float(grad_clip) > 0:
            ds_config["gradient_clipping"] = float(grad_clip)

        warmup_steps = round(optim_cfg.lr_warmup_steps_ratio * training_horizon)
        if lr_scheduler_type == "cosine":
            ds_config["scheduler"] = {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": training_horizon,
                    "warmup_num_steps": warmup_steps,
                    "warmup_min_ratio": 0.0,
                    "cos_min_ratio": optim_cfg.get("min_lr_ratio") or 0.0,
                    "warmup_type": "linear",
                },
            }
        elif warmup_steps > 0:
            ds_config["scheduler"] = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": optim_cfg.lr,
                    "warmup_num_steps": warmup_steps,
                    "warmup_type": "linear",
                },
            }

        log_prob_ds_config = None if n_log_prob_gpus == 0 else self._create_ds_config(n_log_prob_gpus)

        # attn_implementation also lives on ds_worker_config; keep it there for
        # the DeepSpeed worker (matches recipe/rl-correctness).
        ds_worker_config = self._create_ds_worker_config()
        ds_worker_config.setdefault("attn_implementation", attn_implementation)

        rl_config = ArcticClientConfig(
            model_name=model_name,
            seed=self._backend_config.train.determinism.get("seed", 42),
            max_seq_len=max_length,
            training_gpus=n_training_gpus,
            sampling_gpus=n_sampling_gpus,
            log_prob_gpus=n_log_prob_gpus,
            # CortexConfig() reads ARCTIC_CORTEX_HOST / _PAT / _DATABASE / _SCHEMA / ... from the environment.
            backend=CortexConfig() if self.use_cortex else OnPremConfig(colocate=colocate),
            training=TrainingConfig(
                full_determinism=self._backend_config.train.determinism.get("full", False),
                checkpoint_path=self.config.trainer.default_local_dir,
                ds_config=ds_config,
                ds_worker_config=ds_worker_config,
                cuda_ipc=self.cuda_ipc_weight_sync,
                low_memory=self.low_memory_weight_sync,
            ),
            sampling=SamplingConfig(
                vllm=vllm_config,
                # Cortex spreads this block onto inference_config; _check_cortex_options has it disabled there.
                arctic_inference_config=None if self.use_cortex else (arctic_inference_config or None),
                log_prob_engine="deepspeed",
                log_prob_ds_config=log_prob_ds_config,
            ),
        )

        if self.use_cortex:
            return AsyncArcticRLClient(rl_config)

        # AsyncArcticRLClient is constructed as a ray remote actor with num_gpus=0,
        # which causes CUDA_VISIBLE_DEVICES to be empty.
        if colocate:
            num_visible = n_training_gpus + n_sampling_gpus + n_log_prob_gpus
        else:
            num_visible = rl_config.training_gpus + rl_config.sampling_gpus + rl_config.log_prob_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_visible))

        return AsyncArcticRLClient(rl_config)

    _default_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 1024,
    }

    async def generate(self, prompt_ids, sampling_params, routing_key=None) -> list:
        # `routing_key` kept for caller-API compat; arctic_platform.rl handles routing internally.
        merged_params = {**self._default_sampling_params, **sampling_params}
        if self.use_cortex:
            return await self._cortex_generate(prompt_ids, merged_params)
        prompts = [self.tokenizer.decode(prompt_ids)]
        return await self._client.generate(
            prompts=prompts,
            sampling_params=merged_params,
        )

    async def _cortex_generate(self, prompt_ids, sampling_params: dict) -> list:
        # Cortex takes token-id prompts, so the sampler continues the exact ids training concatenates.
        params = {key: value for key, value in sampling_params.items() if value is not None}
        results = await self._client.generate(prompts=[[int(t) for t in prompt_ids]], sampling_params=params)
        if len(results) != 1:
            raise RuntimeError(f"asked Cortex for one completion and got {len(results)}")
        outputs = []
        for result in results:
            token_ids = [int(t) for t in result.get("token_ids") or []]
            text = result.get("text")
            if text is None:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            outputs.append({"text": text, "token_ids": token_ids, "finish_reason": result.get("finish_reason")})
        return outputs

    # ------------------------------------------------------------------ #
    # Arctic wire helpers (private; not on `RemoteBackend`)
    # ------------------------------------------------------------------ #
    # Invoked from `compute_log_prob` / `update_actor` above. Future
    # backends are free to pick their own wire format; verl never calls
    # these directly.

    async def _send_compute_ref_log_prob(self, payload: dict):
        payload["processing"] = {
            "post": ["compute_entropy_and_logprobs"],
            "loss_fn": None,
        }
        response = await self._client.fwd_no_grad(payload, reference_model=True)
        response["batch"]["log_probs"] = response["batch"].pop("logprobs")
        return response

    async def _send_compute_log_prob(self, payload: dict):
        payload["processing"] = {
            "post": ["compute_entropy_and_logprobs"],
            "loss_fn": None,
        }
        response = await self._client.fwd_no_grad(payload, reference_model=False)
        response["batch"]["log_probs"] = response["batch"].pop("logprobs")
        return response

    async def _send_update_actor(self, payload: dict):
        payload["processing"] = {
            "post": ["apply_temperature", "compute_entropy_and_logprobs"],
            "loss_fn": "verl_grpo",
        }

        def _left_pad(t: torch.Tensor, seq_len: int) -> torch.Tensor:
            """Left-pad a response-only tensor to full sequence length with zeros."""
            pad_len = seq_len - t.shape[-1]
            if pad_len <= 0:
                return t
            pad = torch.zeros(*t.shape[:-1], pad_len, dtype=t.dtype, device=t.device)
            return torch.cat([pad, t], dim=-1)

        seq_len = payload["batch"]["input_ids"].shape[-1]
        for name in ["old_log_probs", "advantages", "response_mask", "ref_log_prob"]:
            if name in payload["batch"]:
                payload["batch"][name] = _left_pad(payload["batch"][name], seq_len)
        payload["batch"]["loss_mask"] = payload["batch"]["response_mask"]

        if self.use_cortex:
            return await self._send_cortex_update_actor(payload)

        fwd_bwd_response = await self._client.fwd_bwd(payload)
        step_response = await self._client.step()
        step_response["metrics"].update(**fwd_bwd_response["metrics"])
        return step_response

    def _cortex_job_packs(self) -> bool:
        return cortex_job_packs(self._client.config.training.ds_worker_config)

    async def _send_cortex_compute_log_prob(self, payload: dict):
        """On-prem's ``{batch: {log_probs}}`` from a Cortex forward.

        The training job's forward returns log-probs and no entropy. The key is omitted so a later
        reduction cannot record NaN as ``actor/entropy``.
        """
        payload["meta"]["temperature"] = _positive_temperature(payload["meta"]["temperature"])
        packed = self._cortex_job_packs()
        result = await self._client.fwd_no_grad(to_cortex_fwd_payload(payload, packed=packed))
        attention_mask = payload["batch"]["attention_mask"]
        log_probs = _cortex_logprobs(result)
        if tuple(log_probs.shape) != tuple(attention_mask.shape):
            raise RuntimeError(
                f"Cortex forward log-probs have shape {tuple(log_probs.shape)}; the client layout is "
                f"{tuple(attention_mask.shape)}"
            )
        if packed:
            log_probs = _left_aligned_to_client_layout(log_probs, attention_mask)
        return {"batch": {"log_probs": log_probs}}

    async def _send_cortex_update_actor(self, payload: dict):
        if "rollout_is_weights" in payload["batch"] or "ref_log_prob" in payload["batch"]:
            raise NotImplementedError(
                "the Cortex verl_grpo wire carries no rollout_is_weights or ref_log_prob; disable rollout "
                "correction and the KL loss"
            )
        meta = payload["meta"]
        meta["temperature"] = _positive_temperature(meta["temperature"])
        meta["batch_num_tokens"] = int(_scalar(meta["batch_num_tokens"]))
        processing = payload.pop("processing")
        # verl places response_mask / advantages / old_log_probs on the token they score.
        cortex_payload = to_cortex_fwd_bwd_payload(
            payload, processing=processing, response_aligned=True, packed=self._cortex_job_packs()
        )
        fwd_bwd_response = await self._client.fwd_bwd(cortex_payload)
        step_response = await self._client.step()
        return {"metrics": _merge_cortex_update_metrics(fwd_bwd_response, step_response)}

    async def save_checkpoint(self):
        return await self._client.save_checkpoint()

    async def update_weights(self):
        return await self._client.sync_weights()

    async def wake_up_inference(self, tags: list[str] = None):
        return await self._client.wake_inference(tags=tags)

    async def sleep_inference(self, level: int = 1):
        return await self._client.sleep_inference(level=level)

    async def reset_prefix_cache(self):
        return await self._client.reset_prefix_cache()

    async def destroy(self) -> None:
        # Idempotent: `RemoteBackendTrainer` may call `destroy()` on
        # several exception paths, and callers rely on a second call
        # being a no-op instead of double-shutting the client.
        if self._client is not None:
            client, self._client = self._client, None
            await client.shutdown()
