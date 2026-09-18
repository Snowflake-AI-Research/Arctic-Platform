# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Data contract for the Harbor -> Arctic post-training backend.

Mirrors the Harbor post-training RFC and Harbor's own ``LLMResponse``
rollout fields (prompt_token_ids / completion_token_ids / logprobs) so
a Harbor job's ``rollout_details`` map onto ``Rollout`` 1:1.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class Rollout(BaseModel):
    """One scored trajectory. Field names match Harbor's rollout_details.

    The adapter flattens multi-turn per-turn tokens into ``prompt_token_ids``
    (context into the final turn) and ``completion_token_ids`` (final turn's
    response). ``loss_mask``, when set, has length
    ``len(prompt_token_ids) + len(completion_token_ids)`` and is 1.0 for
    model-produced positions across all turns and 0.0 elsewhere.
    """

    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    reward: float
    logprobs: list[float] | None = None
    loss_mask: list[float] | None = None
    prompt_text: str | None = None
    completion_text: str | None = None
    group_id: str | None = None  # rollouts sharing a prompt form a GRPO group
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutDataset(BaseModel):
    schema_version: str = "harbor-rollout-v1"
    rollouts: list[Rollout]
    dataset_id: str
    model_name: str
    tokenizer_name: str

    def mean_reward(self) -> float:
        return sum(r.reward for r in self.rollouts) / max(len(self.rollouts), 1)


class PostTrainingConfig(BaseModel):
    backend: str = "arctic-cortex"
    algorithm: Literal["grpo"] = "grpo"
    base_model: str
    learning_rate: float = 1e-6
    # Remaining AdamW settings. Defaults are torch's, not the recipe's: SWE-RL
    # recipes commonly run beta2 well below 0.999 and eps far below 1e-8,
    # because rewards are sparse and the second-moment estimate otherwise
    # adapts too slowly to be useful over a few hundred steps.
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    n_samples_per_prompt: int = Field(4, ge=2)  # GRPO needs a group
    train_gpus: int = 1
    sample_gpus: int = 1
    max_seq_len: int = 1024
    eps_clip: float = 0.2
    # Divide group-relative advantages by the group's reward std. On a binary
    # pass/fail reward this rescales by 1/std, which is largest exactly for the
    # groups with the least signal (one success in eight), so it amplifies the
    # noisiest gradients. Recipes tuned on SWE tasks commonly turn it off and
    # mean-center only.
    std_normalization: bool = True
    # Sequences per ``fwd_bwd`` call. A step's worth of long agent rollouts is
    # far too large for one request; gradients accumulate across micro-batches
    # and the optimizer steps once at the end, so this changes only memory and
    # payload size, not the update.
    micro_batch_size: int = 8
    # Cortex/SnowAPI target.
    cortex_host: str | None = None
    cortex_database: str = "NEUTRINO_DB"
    cortex_schema: str = "PUBLIC"
    cortex_pat_env_var: str = "CORTEX_PAT"
    job_ready_timeout: float = 1800.0


class TrainingProgress(BaseModel):
    step: int
    eval_reward: float
    loss: float | None = None
    grad_norm: float | None = None
    backend_extras: dict[str, Any] = Field(default_factory=dict)


class TrainingRun(BaseModel):
    run_id: str
    backend: str
    training_job_id: str | None = None
    sampling_job_id: str | None = None
    history: list[TrainingProgress] = Field(default_factory=list)


class InferenceEndpoint(BaseModel):
    """A handle to the live, updated model. On Cortex the sampling sub-job
    serves the just-synced weights, so re-eval uses it directly."""

    model: str
    sampling_job_id: str | None = None
    note: str = ""
