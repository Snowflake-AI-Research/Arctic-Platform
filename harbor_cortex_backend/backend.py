# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""ArcticCortexBackend: reference PostTrainingBackend over Cortex Training.

``connect`` creates a Cortex job (training sub-job + sampling sub-job on the
same model). The Harbor agent samples via ``generate``; Harbor scores;
``train`` turns the scored rollouts into one GRPO step (fwd_bwd -> step ->
sync_weights). The sync pushes new weights to the sampling sub-job, so the
next eval reads the improved model from the same endpoint.
"""

from __future__ import annotations

import asyncio
import uuid

import torch

from arctic_platform.integrations.harbor.models import InferenceEndpoint
from arctic_platform.integrations.harbor.models import PostTrainingConfig
from arctic_platform.integrations.harbor.models import RolloutDataset
from arctic_platform.integrations.harbor.models import TrainingRun


def _grpo_advantages(rewards: list[float], group_ids: list[str]) -> list[float]:
    """Group-relative advantage: z-score rewards within each shared-prompt group.

    This is the whole of GRPO's credit assignment — no learned critic. A group
    where every sample scored the same yields zero advantage (nothing to learn).
    """
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)

    adv = [0.0] * len(rewards)
    for idxs in groups.values():
        vals = [rewards[i] for i in idxs]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var**0.5
        for i in idxs:
            adv[i] = (rewards[i] - mean) / (std + 1e-6)
    return adv


class ArcticCortexBackend:
    """Reference ``PostTrainingBackend`` implementation over Cortex Training."""

    def __init__(self, config: PostTrainingConfig) -> None:
        self.config = config
        self._client = None
        self.run = TrainingRun(run_id=f"run_{uuid.uuid4().hex[:8]}", backend=config.backend)

    @staticmethod
    def name() -> str:
        return "arctic-cortex"

    # ── lifecycle ────────────────────────────────────────────────────────
    def connect(self) -> TrainingRun:
        """Create the Cortex job (training + sampling sub-jobs). Blocks on
        cold-start (weights + vLLM warmup, typically a few minutes)."""
        from arctic_platform.rl import ArcticRLClientConfig
        from arctic_platform.rl import create_arctic_rl_client

        c = self.config
        cfg = ArcticRLClientConfig(
            backend="cortex",
            model_name=c.base_model,
            training_gpus=c.train_gpus,
            sampling_gpus=c.sample_gpus,
            log_prob_gpus=0,
            max_seq_len=c.max_seq_len,
            cortex_host=c.cortex_host,
            cortex_database=c.cortex_database,
            cortex_schema=c.cortex_schema,
            cortex_pat_env_var=c.cortex_pat_env_var,
            job_ready_timeout=c.job_ready_timeout,
            training_config={
                "train_batch_size": 1,
                "optimizer": {"lr": c.learning_rate},
            },
            vllm_config={"gpu_memory_utilization": 0.6, "enable_prefix_caching": True},
        )
        self._client = create_arctic_rl_client(cfg)
        self.run.training_job_id = str(self._client.training_job_id)
        self.run.sampling_job_id = str(self._client.sampling_job_id)
        return self.run

    async def generate(self, prompts: list[str], sampling_params: dict) -> list[dict]:
        """Sample from the live Cortex-hosted model (what the BYO agent calls)."""
        assert self._client is not None, "call connect() first"
        return await self._client.generate(prompts=prompts, sampling_params=sampling_params)

    # ── the RFC's train() — one GRPO step on the collected rollouts ────────
    async def train(self, rollouts: RolloutDataset, step: int = 0) -> dict:
        assert self._client is not None, "call connect() first"
        batch = self._build_grpo_batch(rollouts)
        fb = await self._client.fwd_bwd(batch)
        st = await self._client.step()
        await self._client.sync_weights()  # push trainer -> sampler
        metrics = {**(st.get("metrics") or {}), **(fb.get("metrics") or {})}
        return metrics

    def deploy_inference(self) -> InferenceEndpoint:
        """The sampling sub-job already serves the synced weights."""
        assert self._client is not None
        return InferenceEndpoint(
            model=self.config.base_model,
            sampling_job_id=str(self._client.sampling_job_id),
            note="Cortex sampling sub-job serves the latest synced weights.",
        )

    def cancel(self) -> None:
        if self._client is not None:
            res = self._client.shutdown()  # shim does the work eagerly, returns a coroutine
            if asyncio.iscoroutine(res):
                res.close()
            self._client = None

    # ── batch construction ───────────────────────────────────────────────
    def _build_grpo_batch(self, ds: RolloutDataset) -> dict:
        """Pack scored rollouts into the {input_ids, attention_mask, advantages,
        loss_mask} tensors the Cortex shim's fwd_bwd expects."""
        rewards = [r.reward for r in ds.rollouts]
        groups = [r.group_id or "g0" for r in ds.rollouts]
        advs = _grpo_advantages(rewards, groups)

        seqs, prompt_lens = [], []
        for r in ds.rollouts:
            seqs.append(r.prompt_token_ids + r.completion_token_ids)
            prompt_lens.append(len(r.prompt_token_ids))
        max_len = min(max(len(s) for s in seqs), self.config.max_seq_len)

        B = len(seqs)
        input_ids = torch.zeros((B, max_len), dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        loss_mask = torch.zeros((B, max_len), dtype=torch.long)
        advantages = torch.zeros((B, max_len), dtype=torch.float32)

        for i, (seq, plen) in enumerate(zip(seqs, prompt_lens)):
            seq = seq[:max_len]
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :n] = 1
            # response tokens only (mask out the prompt) get gradient + advantage
            resp_start = min(plen, n)
            loss_mask[i, resp_start:n] = 1
            advantages[i, resp_start:n] = advs[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }
