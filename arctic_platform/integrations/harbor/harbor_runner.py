# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: real ``harbor run`` -> Arctic GRPO on Cortex -> real ``harbor run``.

Every LLM call happens inside a Harbor trial (real ``BaseAgent``, real
``BaseEnvironment``, real ``BaseVerifier``, real ``RolloutDetail`` written to
``result.json``). The middle step reads Harbor's on-disk output, hands it to
``ArcticCortexBackend.train`` on real Cortex QA6, and ``sync_weights`` makes
the next ``harbor run`` sample from the updated model at the same endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from arctic_platform.integrations.harbor.adapter import load_job_dir, pass_at_1
from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
from arctic_platform.integrations.harbor.models import PostTrainingConfig
from arctic_platform.integrations.harbor.task_gen import write_dataset


AGENT_PATH = "arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent"
ENV_PATH = "arctic_platform.integrations.harbor.host_environment:HostEnvironment"
# Verifier: Harbor's default ``harbor.verifier.verifier:Verifier``. It uploads
# each task's ``tests/`` dir into the environment, execs ``test.sh``, and reads
# ``/logs/verifier/reward.txt``. We ship a real test.sh (see task_gen._TEST_SH)
# — no custom BaseVerifier subclass, no ``--verifier`` override on the CLI.


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] harbor_runner: {msg}", flush=True)


def _harbor_bin() -> str:
    # Same interpreter as our runner -> same conda env -> Harbor CLI present.
    return str(Path(sys.executable).parent / "harbor")


def _run_harbor(
    dataset_dir: Path,
    jobs_dir: Path,
    job_name: str,
    reconnect_config_path: Path,
    model_name: str,
    n_concurrent: int,
    n_attempts: int,
    temperature: float,
    max_tokens: int,
) -> Path:
    """Invoke ``harbor run`` on our generated dataset. Returns the job dir."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _harbor_bin(), "run",
        "-p", str(dataset_dir),
        "-a", AGENT_PATH,
        "-m", model_name,
        "-e", ENV_PATH,
        "-o", str(jobs_dir),
        "--job-name", job_name,
        "-n", str(n_concurrent),
        "-k", str(n_attempts),
        "--yes",
        "--no-force-build",
        "--ak", f"reconnect_config_path={reconnect_config_path}",
        "--ak", f"temperature={temperature}",
        "--ak", f"max_tokens={max_tokens}",
    ]
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return jobs_dir / job_name


def _mean_reward(job_dir: Path) -> float:
    ds = load_job_dir(job_dir, dataset_id="tmp", model_name="tmp")
    if not ds.rollouts:
        return 0.0
    return sum(r.reward for r in ds.rollouts) / len(ds.rollouts)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument("--n-attempts", type=int, default=4)
    ap.add_argument("--heldout", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--n-concurrent", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--task", choices=["add", "mul"], default="mul")
    ap.add_argument("--low", type=int, default=20)
    ap.add_argument("--high", type=int, default=99)
    ap.add_argument("--a-low", type=int, default=None,
                    help="Override low bound of operand a (default: --low)")
    ap.add_argument("--a-high", type=int, default=None,
                    help="Override high bound of operand a (default: --high)")
    ap.add_argument("--b-low", type=int, default=None,
                    help="Override low bound of operand b (default: --low)")
    ap.add_argument("--b-high", type=int, default=None,
                    help="Override high bound of operand b (default: --high)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for train + held-out problem sampling. "
                         "Different seeds give independent runs; report the "
                         "distribution across seeds instead of one number.")
    ap.add_argument("--reuse-training-job-id", default=None,
                    help="Skip Cortex cold-start; attach to an existing training sub-job token.")
    ap.add_argument("--reuse-sampling-job-id", default=None,
                    help="Skip Cortex cold-start; attach to an existing sampling sub-job token.")
    args = ap.parse_args()

    work_dir = Path(args.work_dir or tempfile.mkdtemp(prefix="harbor_e2e_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    _log(f"work_dir = {work_dir}")

    # Warm the HF tokenizer cache once, up front. Otherwise every Harbor trial
    # subprocess (dozens per GRPO step at n_concurrent > 1) hits HF Hub and
    # eventually gets rate-limited to ``HfHubHTTPError`` — trials silently
    # drop, GRPO batches lose rollouts, gradient collapses to zero.
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.model)
        _log(f"tokenizer cache warm for {args.model}")
    except Exception as e:  # noqa: BLE001 — pre-flight failure shouldn't kill the run
        _log(f"tokenizer warmup skipped ({type(e).__name__}: {e}); trials may hit HF Hub")

    # ── 1. Stand up a real Cortex Training job ─────────────────────────────
    cfg = PostTrainingConfig(
        base_model=args.model,
        learning_rate=args.lr,
        n_samples_per_prompt=args.n_attempts,
        max_seq_len=512,
        cortex_host=os.environ["ARCTIC_CORTEX_HOST"],
        cortex_database=os.environ.get("ARCTIC_CORTEX_DATABASE", "NEUTRINO_DB"),
        cortex_schema=os.environ.get("ARCTIC_CORTEX_SCHEMA", "PUBLIC"),
        cortex_pat_env_var=os.environ.get("ARCTIC_CORTEX_PAT_ENV_VAR", "CORTEX_PAT"),
    )
    backend = ArcticCortexBackend(cfg)
    if args.reuse_training_job_id and args.reuse_sampling_job_id:
        _log(f"REUSING existing Cortex job: train={args.reuse_training_job_id} sample={args.reuse_sampling_job_id}")
        from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client

        legacy_cfg = ArcticRLClientConfig(
            backend="cortex",
            model_name=cfg.base_model,
            training_gpus=cfg.train_gpus,
            sampling_gpus=cfg.sample_gpus,
            log_prob_gpus=0,
            max_seq_len=cfg.max_seq_len,
            cortex_host=cfg.cortex_host,
            cortex_database=cfg.cortex_database,
            cortex_schema=cfg.cortex_schema,
            cortex_pat_env_var=cfg.cortex_pat_env_var,
            training_job_id=args.reuse_training_job_id,
            sampling_job_id=args.reuse_sampling_job_id,
            training_config={"train_batch_size": 1, "optimizer": {"lr": cfg.learning_rate}},
            vllm_config={"gpu_memory_utilization": 0.6, "enable_prefix_caching": True},
        )
        backend._client = create_arctic_rl_client(legacy_cfg)
        backend.run.training_job_id = str(backend._client.training_job_id)
        backend.run.sampling_job_id = str(backend._client.sampling_job_id)
        run = backend.run
    else:
        _log("creating Cortex job (training + sampling sub-jobs); cold-start can take a few minutes ...")
        run = backend.connect()
    _log(f"connected: run={run.run_id} train_job={run.training_job_id} sample_job={run.sampling_job_id}")

    # Dump reconnect config so each Harbor trial (spawned by harbor run) can
    # reattach to the same running Cortex sub-jobs. The job-id fields are
    # ``Field(exclude=True)`` in the legacy config so ``model_dump_json``
    # drops them — add them back explicitly.
    reconnect_cfg = backend._client.reconnect_config()
    cfg_dict = json.loads(reconnect_cfg.model_dump_json())
    cfg_dict["training_job_id"] = reconnect_cfg.training_job_id
    cfg_dict["sampling_job_id"] = reconnect_cfg.sampling_job_id
    cfg_dict["log_prob_job_id"] = reconnect_cfg.log_prob_job_id
    reconnect_path = work_dir / "reconnect_config.json"
    reconnect_path.write_text(json.dumps(cfg_dict))
    _log(f"reconnect config -> {reconnect_path}  (train_job_id={cfg_dict['training_job_id']!r})")

    try:
        a_lo = args.a_low if args.a_low is not None else args.low
        a_hi = args.a_high if args.a_high is not None else args.high
        b_lo = args.b_low if args.b_low is not None else args.low
        b_hi = args.b_high if args.b_high is not None else args.high
        _log(f"operand ranges: a in [{a_lo},{a_hi}], b in [{b_lo},{b_hi}], op={args.task}")

        rng = random.Random(args.seed)
        heldout_rng = random.Random(args.seed + 999)
        heldout_probs = [
            (heldout_rng.randint(a_lo, a_hi), heldout_rng.randint(b_lo, b_hi))
            for _ in range(args.heldout)
        ]
        heldout_ds_dir = work_dir / "dataset_heldout"
        write_dataset(heldout_ds_dir, heldout_probs, op=args.task, prefix="heldout")

        # ── 2. Baseline: real harbor run on the held-out set ────────────────
        _log("BASELINE harbor run (greedy, k=1) ...")
        baseline_job_dir = _run_harbor(
            dataset_dir=heldout_ds_dir,
            jobs_dir=work_dir / "harbor_jobs",
            job_name="baseline",
            reconnect_config_path=reconnect_path,
            model_name=args.model,
            n_concurrent=args.n_concurrent,
            n_attempts=1,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        base_ds = load_job_dir(baseline_job_dir, "baseline", args.model)
        base_pass = pass_at_1(base_ds)
        _log(f"BASELINE pass@1 = {base_pass:.3f}  ({sum(1 for r in base_ds.rollouts if r.reward >= 1.0)}/{len(base_ds.rollouts)})")

        # ── 3. Training loop: N iterations of harbor rollout -> GRPO step ──
        curve: list[float] = []
        for step in range(args.iters):
            probs = [
                (rng.randint(a_lo, a_hi), rng.randint(b_lo, b_hi))
                for _ in range(args.prompts_per_step)
            ]
            step_ds_dir = work_dir / f"dataset_step_{step:02d}"
            write_dataset(step_ds_dir, probs, op=args.task, prefix=f"step{step}")

            _log(f"STEP {step:02d} harbor run (k={args.n_attempts}, temp={args.temperature}) ...")
            step_job_dir = _run_harbor(
                dataset_dir=step_ds_dir,
                jobs_dir=work_dir / "harbor_jobs",
                job_name=f"step_{step:02d}",
                reconnect_config_path=reconnect_path,
                model_name=args.model,
                n_concurrent=args.n_concurrent,
                n_attempts=args.n_attempts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            step_ds = load_job_dir(step_job_dir, f"step{step}", args.model)
            reward = _mean_reward(step_job_dir)
            correct = sum(1 for r in step_ds.rollouts if r.reward >= 1.0)
            _log(f"  rollouts={len(step_ds.rollouts)} reward_mean={reward:.3f} correct={correct}")

            metrics = await backend.train(step_ds, step=step)
            curve.append(reward)
            _log(f"  step {step:02d}: loss={metrics.get('loss')} grad_norm={metrics.get('grad_norm')}")

        # ── 4. Post-training re-eval on the SAME held-out set ──────────────
        _log("FINAL harbor run (greedy, k=1) ...")
        final_job_dir = _run_harbor(
            dataset_dir=heldout_ds_dir,
            jobs_dir=work_dir / "harbor_jobs",
            job_name="final",
            reconnect_config_path=reconnect_path,
            model_name=args.model,
            n_concurrent=args.n_concurrent,
            n_attempts=1,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        final_ds = load_job_dir(final_job_dir, "final", args.model)
        final_pass = pass_at_1(final_ds)
        # Mean held-out reward — captures the improvements in output quality
        # (e.g. "escaping loops") that a binary pass@1 metric hides. When
        # GRPO can't fully move pass@1 on a small model in a few dozen steps,
        # the mean reward often still moves substantially.
        base_mean_reward = sum(r.reward for r in base_ds.rollouts) / max(1, len(base_ds.rollouts))
        final_mean_reward = sum(r.reward for r in final_ds.rollouts) / max(1, len(final_ds.rollouts))
        _log(f"FINAL pass@1 = {final_pass:.3f}  ({sum(1 for r in final_ds.rollouts if r.reward >= 1.0)}/{len(final_ds.rollouts)})")
        _log(f"reward curve: {[round(x, 3) for x in curve]}")
        _log(
            f"RESULT  pass@1: {base_pass:.3f} -> {final_pass:.3f}  ({final_pass - base_pass:+.3f})  "
            f"|  mean held-out reward: {base_mean_reward:.3f} -> {final_mean_reward:.3f}  "
            f"({final_mean_reward - base_mean_reward:+.3f})"
        )
        (work_dir / "summary.json").write_text(json.dumps({
            "baseline_pass_at_1": base_pass,
            "final_pass_at_1": final_pass,
            "delta": final_pass - base_pass,
            "baseline_mean_reward": base_mean_reward,
            "final_mean_reward": final_mean_reward,
            "delta_mean_reward": final_mean_reward - base_mean_reward,
            "reward_curve": curve,
            "run_id": run.run_id,
            "training_job_id": run.training_job_id,
            "sampling_job_id": run.sampling_job_id,
            "task": args.task,
            "a_range": [a_lo, a_hi],
            "b_range": [b_lo, b_hi],
            "seed": args.seed,
            "iters": args.iters,
            "prompts_per_step": args.prompts_per_step,
            "n_attempts": args.n_attempts,
            "heldout": args.heldout,
            "lr": args.lr,
            "temperature": args.temperature,
        }, indent=2))
        _log(f"summary -> {work_dir / 'summary.json'}")
    finally:
        _log("shutting down Cortex job")
        backend.cancel()


if __name__ == "__main__":
    asyncio.run(main())
