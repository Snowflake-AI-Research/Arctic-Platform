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
import logging
import time
import uuid

import torch

from arctic_platform.integrations.harbor.models import (
    InferenceEndpoint,
    PostTrainingConfig,
    RolloutDataset,
    TrainingRun,
)

_log = logging.getLogger(__name__)

# Trainable-token count reported per fwd_bwd; doubles as the averaging weight.
_WEIGHT_KEY = "trainable_logprob_count_all"


def _merge_fwd_bwd_metrics(per_micro: list[dict]) -> dict:
    """Collapse one metrics dict per micro-batch into one dict for the step.

    Keeping only the last call's metrics would describe a single slice, and
    because slices are ordered longest-first that slice is the shortest
    rollouts of the step — the least representative sample available. Counts
    sum, extremes take the extreme, and everything else is averaged weighted
    by trainable tokens so a 2-token slice cannot outvote a 30k-token one.
    """
    dicts = [m for m in per_micro if m]
    if not dicts:
        return {}

    weights = [float(m.get(_WEIGHT_KEY) or 0.0) for m in dicts]
    if sum(weights) <= 0:  # nothing reported a count; fall back to a plain mean
        weights = [1.0] * len(dicts)

    out: dict = {}
    for key in {k for m in dicts for k in m}:
        pairs = [
            (m[key], w)
            for m, w in zip(dicts, weights)
            if isinstance(m.get(key), (int, float)) and not isinstance(m.get(key), bool)
        ]
        if not pairs:
            out[key] = next(m[key] for m in dicts if key in m)
            continue
        vals = [v for v, _ in pairs]
        if key.endswith("_count_all") or key == "rl_model_calls":
            out[key] = sum(vals)
        elif key.endswith("/max") or "_max_" in key or key.endswith("_max"):
            out[key] = max(vals)
        elif key.endswith("/min") or key.endswith("_min"):
            out[key] = min(vals)
        elif key == "rank":
            out[key] = vals[0]
        else:
            total = sum(w for _, w in pairs)
            out[key] = (
                sum(v * w for v, w in pairs) / total
                if total > 0
                else sum(vals) / len(vals)
            )
    return out


def _grpo_advantages(
    rewards: list[float],
    group_ids: list[str],
    std_normalization: bool = True,
) -> list[float]:
    """Group-relative advantage: centre rewards within each shared-prompt group.

    This is the whole of GRPO's credit assignment — no learned critic. A group
    where every sample scored the same yields zero advantage (nothing to learn).

    ``std_normalization`` additionally divides by the group's standard
    deviation. With a binary pass/fail reward that factor is 1/std, which peaks
    on the groups carrying the weakest evidence — one success in eight — so it
    scales up the noisiest gradients. Turning it off keeps advantages
    proportional to how far a rollout beat its group.
    """
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)

    adv = [0.0] * len(rewards)
    for idxs in groups.values():
        vals = [rewards[i] for i in idxs]
        mean = sum(vals) / len(vals)
        scale = 1.0
        if std_normalization:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            scale = var**0.5 + 1e-6
        for i in idxs:
            adv[i] = (rewards[i] - mean) / scale
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
                "optimizer": {
                    "lr": c.learning_rate,
                    # Cortex's optimizer schema takes beta1/beta2 as scalars; a
                    # ``betas`` pair is accepted by the client filter but ignored
                    # server-side, which silently leaves beta2 at its 0.999
                    # default. Send both spellings so the value actually lands.
                    "betas": list(c.adam_betas),
                    "beta1": c.adam_betas[0],
                    "beta2": c.adam_betas[1],
                    "eps": c.adam_eps,
                    "weight_decay": c.weight_decay,
                },
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

        # Advantages are computed once over the whole step, before splitting:
        # they are group-relative, so a group must be scored against its own
        # members. Slicing first and normalising per slice would compare a
        # rollout against whatever else happened to land beside it.
        batch = self._build_grpo_batch(rollouts)
        micro = max(1, self.config.micro_batch_size)
        n = batch["input_ids"].shape[0]

        # Group length-similar sequences together. Padding is per micro-batch,
        # so mixing a 200-token turn with a 100k one pays the 100k width on
        # both; sorting makes each micro-batch about as wide as its own
        # longest member. Advantages are already fixed, so reordering is safe.
        import torch

        lengths = batch["attention_mask"].sum(dim=1)
        order = torch.argsort(lengths, descending=True)

        per_micro: list[dict] = []
        n_micro = (n + micro - 1) // micro
        t0 = time.monotonic()
        _log.info(
            "step %d: %d rollouts -> %d fwd_bwd calls (micro=%d, longest=%d tokens)",
            step,
            n,
            n_micro,
            micro,
            int(lengths.max().item()),
        )
        for i, lo in enumerate(range(0, n, micro), start=1):
            idx = order[lo : lo + micro]
            width = int(lengths[idx].max().item())
            # Trim the columns that are padding for every row in this slice.
            fb = await self._client.fwd_bwd(
                {k: v[idx][:, :width] for k, v in batch.items()}
            )
            per_micro.append(fb.get("metrics") or {})
            # A step is many minutes of round trips; without this the caller
            # cannot tell a slow step from a hung one.
            elapsed = time.monotonic() - t0
            _log.info(
                "step %d: fwd_bwd %d/%d width=%d %.1fs elapsed, ~%.0fs left",
                step,
                i,
                n_micro,
                width,
                elapsed,
                elapsed / i * (n_micro - i),
            )
        # One optimizer step per training step, after the gradients from every
        # micro-batch have accumulated.
        _log.info("step %d: all fwd_bwd done in %.1fs, optimizer step", step, time.monotonic() - t0)
        st = await self._client.step()
        await self._client.sync_weights()  # push trainer -> sampler
        metrics = {**(st.get("metrics") or {}), **_merge_fwd_bwd_metrics(per_micro)}
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
        advs = _grpo_advantages(
            rewards, groups, std_normalization=self.config.std_normalization
        )

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
        # Sampler log-probs, aligned to input_ids. Only meaningful once every
        # rollout carries them: a partially-filled tensor would read as π_old=1
        # (log-prob 0) on the missing rows, silently inflating their importance
        # ratio rather than falling back to the on-policy default.
        old_log_probs = torch.zeros((B, max_len), dtype=torch.float32)
        have_logprobs = all(r.logprobs is not None for r in ds.rollouts)

        for i, (seq, plen) in enumerate(zip(seqs, prompt_lens)):
            seq = seq[:max_len]
            n = len(seq)
            input_ids[i, :n] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :n] = 1
            # Prefer the rollout's own mask: for a multi-turn agent the flattened
            # prompt interleaves earlier assistant turns (trainable) with tool
            # output (not), so prompt-vs-completion alone would drop gradient on
            # every turn but the last.
            turn_mask = ds.rollouts[i].loss_mask
            if turn_mask is not None:
                mask = torch.tensor(turn_mask[:n], dtype=torch.long)
                loss_mask[i, : len(mask)] = mask
                advantages[i, : len(mask)] = mask.to(torch.float32) * advs[i]
            else:
                # response tokens only (mask out the prompt) get gradient + advantage
                resp_start = min(plen, n)
                loss_mask[i, resp_start:n] = 1
                advantages[i, resp_start:n] = advs[i]

            if have_logprobs:
                # The sampler only scores what it generated, so the rollout's
                # log-probs cover the completion and are laid down starting at
                # the prompt boundary.
                lp = ds.rollouts[i].logprobs or []
                start = min(plen, n)
                room = max(n - start, 0)
                if room and lp:
                    take = min(room, len(lp))
                    old_log_probs[i, start:start + take] = torch.tensor(
                        lp[:take], dtype=torch.float32
                    )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }
        if have_logprobs:
            batch["old_log_probs"] = old_log_probs
        return batch
