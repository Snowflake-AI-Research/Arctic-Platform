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

    Keeping only the last call's metrics would describe one slice, and since
    slices run longest-first that slice holds the step's shortest rollouts.
    Counts sum, extremes take the extreme, and the rest is averaged weighted
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
    traj_ids: list[str] | None = None,
) -> list[float]:
    """Group-relative advantage: centre rewards within each shared-prompt group.

    This is the whole of GRPO's credit assignment — no learned critic. A group
    where every sample scored the same yields zero advantage (nothing to learn).

    ``traj_ids`` matters when one trajectory contributes several rollouts, the
    normal case for a multi-turn agent trained one turn per sequence. Centring
    over turns weights a trajectory by how long it ran, letting a 40-turn
    failure drag the group mean forty times harder than a 1-turn success. With
    ``traj_ids`` the statistics are over equally-weighted trajectories and
    every turn inherits its trajectory's advantage.
    """
    from collections import defaultdict

    if traj_ids is None:
        traj_ids = [str(i) for i in range(len(rewards))]

    # group -> ordered unique trajectories, and trajectory -> its row indices.
    group_trajs: dict[str, list[str]] = defaultdict(list)
    traj_rows: dict[str, list[int]] = defaultdict(list)
    traj_reward: dict[str, float] = {}
    for i, (g, t) in enumerate(zip(group_ids, traj_ids)):
        if t not in traj_rows:
            group_trajs[g].append(t)
            traj_reward[t] = rewards[i]
        traj_rows[t].append(i)

    adv = [0.0] * len(rewards)
    for trajs in group_trajs.values():
        vals = [traj_reward[t] for t in trajs]
        mean = sum(vals) / len(vals)
        scale = 1.0
        if std_normalization:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            scale = max(var**0.5, 1e-6)
        for t, v in zip(trajs, vals):
            for i in traj_rows[t]:
                adv[i] = (v - mean) / scale
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
            vllm_config={
                "gpu_memory_utilization": c.gpu_memory_utilization,
                "enable_prefix_caching": True,
                "max_num_seqs": c.max_num_seqs,
                "tensor_parallel_size": c.tensor_parallel_size,
            },
        )
        self._client = create_arctic_rl_client(cfg)
        self.run.training_job_id = str(self._client.training_job_id)
        self.run.sampling_job_id = str(self._client.sampling_job_id)
        self._preflight_weight_sync()
        return self.run

    def _preflight_weight_sync(self) -> None:
        """Push weights once before any rollouts are collected.

        Sync stages weights into a buffer on every sampler GPU, so its memory
        demand is invisible while the engine is only serving requests. Too
        little headroom then costs a full step of collection before failing at
        the first sync; a no-op sync at connect time surfaces it in seconds.
        """
        if self.config.sample_gpus <= 0:
            return
        try:
            asyncio.run(self._client.sync_weights())
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance
            raise RuntimeError(
                "weight sync failed before collection started, which usually "
                "means the sampler has no room for the broadcast buffer; lower "
                "gpu_memory_utilization and retry. Original error: " f"{exc}"
            ) from exc

    async def generate(
        self,
        prompts: list[str],
        sampling_params: dict,
        routing_key: str | None = None,
    ) -> list[dict]:
        """Sample from the live Cortex-hosted model (what the BYO agent calls).

        ``routing_key`` asks the sampler to send a caller's requests to the
        same data-parallel replica. For a single-turn task it is irrelevant;
        for an agent that resends a growing conversation every turn it decides
        whether the prefix cache is warm or whether each turn pays to prefill
        the whole transcript again.
        """
        assert self._client is not None, "call connect() first"
        return await self._client.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            routing_key=routing_key,
        )

    # ── the RFC's train() — one GRPO step on the collected rollouts ────────
    async def train(self, rollouts: RolloutDataset, step: int = 0) -> dict:
        assert self._client is not None, "call connect() first"

        # Advantages are computed once over the whole step, before splitting:
        # they are group-relative, so a group must be scored against its own
        # members. Slicing first and normalising per slice would compare a
        # rollout against whatever else happened to land beside it.
        batch = self._build_grpo_batch(rollouts)
        return await self._train_batch(batch, step=step)

    async def _train_batch(self, batch: dict, step: int = 0) -> dict:
        """Micro-batch, accumulate, step -- optionally several times over.

        Split out from ``train`` so the loop can be exercised without building
        a rollout dataset: the thing worth testing here is the call sequence,
        not the tensors.
        """
        # Every data-parallel training shard must receive at least one row, so
        # the backend rejects a micro-batch narrower than the training world.
        # Raising the floor costs a little memory and keeps the step alive.
        micro = max(1, self.config.micro_batch_size, self.config.train_gpus)
        if micro != self.config.micro_batch_size:
            _log.info(
                "micro_batch_size %d raised to %d to cover %d training GPUs",
                self.config.micro_batch_size, micro, self.config.train_gpus,
            )
        n = batch["input_ids"].shape[0]

        # Group length-similar sequences together. Padding is per micro-batch,
        # so mixing a 200-token turn with a 100k one pays the 100k width on
        # both; sorting makes each micro-batch about as wide as its own
        # longest member. Advantages are already fixed, so reordering is safe.
        import torch

        lengths = batch["attention_mask"].sum(dim=1)
        order = torch.argsort(lengths, descending=True)

        # Slice boundaries, with a short tail folded into the slice before it.
        # An odd row count would otherwise end in a one-row request, which the
        # backend refuses for the same data-parallel reason as above.
        bounds = list(range(0, n, micro))
        if len(bounds) > 1 and n - bounds[-1] < micro:
            bounds.pop()

        n_micro = len(bounds)
        # Replaying a collection is only sound if the batch carries the
        # sampling policy's log-probs: without them the loss treats pi_old as
        # pi_new, so every replayed step would look on-policy to the clip and
        # the drift would go uncorrected.
        inner = max(int(self.config.max_off_policy_steps), 1)
        if inner > 1 and not ({"old_log_probs", "old_log_probs_shifted"} & batch.keys()):
            _log.warning(
                "step %d: max_off_policy_steps=%d ignored, batch has no sampler "
                "log-probs to measure drift against",
                step,
                inner,
            )
            inner = 1

        _log.info(
            "step %d: %d rollouts -> %d fwd_bwd calls (micro=%d, longest=%d tokens)"
            " x %d optimizer step(s)",
            step,
            n,
            n_micro,
            micro,
            int(lengths.max().item()),
            inner,
        )

        per_micro: list[dict] = []
        last_step_metrics: dict = {}
        t0 = time.monotonic()
        for inner_i in range(inner):
            micro_start = len(per_micro)
            for i, lo in enumerate(bounds, start=1):
                # The last slice runs to the end, absorbing any short tail.
                hi = bounds[i] if i < len(bounds) else n
                idx = order[lo:hi]
                width = int(lengths[idx].max().item())
                # Trim the columns that are padding for every row in this slice.
                fb = await self._client.fwd_bwd(
                    {k: v[idx][:, :width] for k, v in batch.items()},
                )
                per_micro.append(fb.get("metrics") or {})
                # A step is many minutes of round trips; without this the caller
                # cannot tell a slow step from a hung one.
                elapsed = time.monotonic() - t0
                done = inner_i * n_micro + i
                total = inner * n_micro
                _log.info(
                    "step %d.%d: fwd_bwd %d/%d width=%d %.1fs elapsed, ~%.0fs left",
                    step,
                    inner_i,
                    i,
                    n_micro,
                    width,
                    elapsed,
                    elapsed / done * (total - done),
                )
            st = await self._client.step()
            last_step_metrics = st.get("metrics") or {}
            # The importance ratio is what makes replay safe, so log it per
            # inner step. It arrives on the fwd_bwd metrics rather than the
            # optimizer's, hence the merge over this step's micro-batches.
            fb = _merge_fwd_bwd_metrics(per_micro[micro_start:])
            _log.info(
                "step %d.%d: optimizer step, importance_weight=%s clip_ratio=%s "
                "approx_kl=%s entropy=%s grad_norm=%s",
                step,
                inner_i,
                fb.get("importance_weight"),
                fb.get("clip_ratio"),
                fb.get("approx_kl"),
                fb.get("entropy"),
                last_step_metrics.get("grad_norm"),
            )

        # One sync at the end: the sampler only needs the policy the *next*
        # collection runs against. A training-only job has nothing to push to.
        if self.config.sample_gpus > 0:
            await self._client.sync_weights()  # push trainer -> sampler
        metrics = {
            **last_step_metrics,
            **_merge_fwd_bwd_metrics(per_micro),
            "off_policy_steps": float(inner),
        }
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
        # A multi-turn agent emits one rollout per assistant turn, all tagged
        # with the trajectory they came from, so that the group statistics are
        # taken over trajectories rather than over turns.
        traj_ids = [r.metadata.get("traj_id") or str(i)
                    for i, r in enumerate(ds.rollouts)]
        advs = _grpo_advantages(
            rewards,
            groups,
            std_normalization=self.config.std_normalization,
            traj_ids=traj_ids,
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
            # Sent unshifted on purpose: the dispatch shim owns the roll(-1)
            # onto the server's ``old_log_probs_shifted`` contract, including
            # zeroing the position that wraps. Rolling here as well would
            # misalign every ratio by one token.
            batch["old_log_probs"] = old_log_probs
        return batch
