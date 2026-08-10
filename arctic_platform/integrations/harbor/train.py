# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""``harbor-cortex-train``: alternate ``harbor run`` -> GRPO -> ``sync_weights``.

Each LLM call happens inside a Harbor trial (``BaseAgent`` under
``BaseEnvironment``, ``RolloutDetail`` written to ``result.json``). The
middle step reads Harbor's on-disk output, hands it to
``ArcticCortexBackend.train``, and ``sync_weights`` makes the next
``harbor run`` sample from the updated model at the same endpoint.
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


# Import paths passed to ``harbor run --agent/--env``. Users can also refer
# to these by the short names registered in the ``harbor.plugins``
# entry-point group in Arctic-Platform's top-level ``pyproject.toml``:
#   ``arctic-cortex-agent`` -> arctic_platform.integrations.harbor.agent:CortexRLAgent
#   ``arctic-cortex-env``   -> arctic_platform.integrations.harbor.env:HostEnvironment
DEFAULT_AGENT_PATH = "arctic_platform.integrations.harbor.agent:CortexRLAgent"
DEFAULT_ENV_PATH = "arctic_platform.integrations.harbor.env:HostEnvironment"
# Verifier: Harbor's default ``harbor.verifier.verifier:Verifier``. It uploads
# each task's ``tests/`` dir into the environment, execs ``test.sh``, and reads
# ``/logs/verifier/reward.txt``. No custom BaseVerifier subclass, no
# ``--verifier`` override on the CLI. Users provide their own ``tests/test.sh``
# inside each task dir — the built-in arithmetic generator ships one, the
# ``--tasks-dir`` path expects one to already exist.


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
    agent_path: str = DEFAULT_AGENT_PATH,
    env_path: str = DEFAULT_ENV_PATH,
    extra_instruction_paths: list[Path] | None = None,
    skill_paths: list[Path] | None = None,
) -> Path:
    """Invoke ``harbor run`` on a Harbor dataset. Returns the job dir.

    ``extra_instruction_paths`` maps to Harbor's ``--extra-instruction-path``
    (each file is appended to every task's instruction). ``skill_paths`` maps
    to ``--skill`` (skill directories the agent can consult). Both are
    Harbor's own extension surfaces — no custom flag translation.
    """
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _harbor_bin(), "run",
        "-p", str(dataset_dir),
        "-a", agent_path,
        "-m", model_name,
        "-e", env_path,
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
    for p in extra_instruction_paths or []:
        cmd += ["--extra-instruction-path", str(p)]
    for p in skill_paths or []:
        cmd += ["--skill", str(p)]
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return jobs_dir / job_name


def _enumerate_task_dirs(pool_dir: Path) -> list[Path]:
    """Return every Harbor task subdir under a pool directory.

    Two layouts are supported:

    * A Harbor dataset directory containing ``dataset.toml``: the ``tasks``
      list is used as-is (path resolution relative to the dataset dir).
    * A plain directory of task subdirs (each with a ``task.toml``): every
      such subdir is treated as a task.

    The runner samples from this pool per training step.
    """
    pool_dir = Path(pool_dir).resolve()
    ds_toml = pool_dir / "dataset.toml"
    if ds_toml.exists():
        import tomllib
        data = tomllib.loads(ds_toml.read_text())
        return [(pool_dir / t["path"]).resolve() for t in data.get("tasks", [])]
    return sorted(
        p.resolve() for p in pool_dir.iterdir()
        if p.is_dir() and (p / "task.toml").exists()
    )


def _write_step_manifest(dataset_dir: Path, task_dirs: list[Path]) -> None:
    """Write a Harbor ``dataset.toml`` at ``dataset_dir`` pointing at ``task_dirs``.

    Uses absolute paths so we don't have to move the user's task directories.
    Called each GRPO step to spin up a sampled sub-dataset of training tasks.
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    lines = ['version = "1.0"', "", "tasks = ["]
    for td in task_dirs:
        lines.append(f'    {{ path = "{td}" }},')
    lines.append("]")
    (dataset_dir / "dataset.toml").write_text("\n".join(lines) + "\n")


def _mean_reward(job_dir: Path) -> float:
    ds = load_job_dir(job_dir, dataset_id="tmp", model_name="tmp")
    if not ds.rollouts:
        return 0.0
    return sum(r.reward for r in ds.rollouts) / len(ds.rollouts)


async def main() -> None:
    ap = argparse.ArgumentParser(description=(
        "Drive Harbor-native RL training on Cortex. Bring your own Harbor "
        "task dir (with tests/test.sh) via --tasks-dir + --heldout-dir, "
        "optionally --skill-md and --agent, and this drives the full loop."
    ))
    # Model + training hyperparams
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument("--n-attempts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--n-concurrent", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)

    # "Bring your own tasks" — the recommended flow for a Harbor user.
    ap.add_argument("--tasks-dir", default=None,
                    help="Directory of Harbor tasks to train on. Either a "
                         "dataset dir (contains dataset.toml) or a directory "
                         "of task subdirs (each with task.toml + tests/test.sh). "
                         "If omitted, an arithmetic MUL/ADD dataset is generated.")
    ap.add_argument("--heldout-dir", default=None,
                    help="Held-out dataset dir for baseline + post-training "
                         "re-eval. Same layout as --tasks-dir. If omitted, a "
                         "held-out arithmetic set is generated.")
    ap.add_argument("--skill-md", action="append", default=[],
                    help="Path to a SKILL.md-style file appended to every "
                         "task's instruction (Harbor's --extra-instruction-path). "
                         "Repeat to layer multiple.")
    ap.add_argument("--skill-dir", action="append", default=[],
                    help="Path to a Harbor skills directory (Harbor's --skill). "
                         "Repeat to layer multiple.")
    ap.add_argument("--agent", default=DEFAULT_AGENT_PATH,
                    help="Agent import path (module:Class). Default is our "
                         "CortexRLAgent that samples from the Cortex sub-job. "
                         "Users with a custom BaseAgent that samples from a "
                         "given endpoint can plug it in here.")
    ap.add_argument("--env", default=DEFAULT_ENV_PATH,
                    help="Environment import path. Default is HostEnvironment "
                         "(no container). Swap for DockerEnvironment / Modal / "
                         "Daytona for real sandboxing.")

    # Arithmetic-generator fallback (only used if --tasks-dir/--heldout-dir omitted).
    ap.add_argument("--task", choices=["add", "mul"], default="mul",
                    help="Arithmetic generator op (ignored if --tasks-dir set).")
    ap.add_argument("--heldout", type=int, default=12,
                    help="Generated arithmetic held-out size (ignored if "
                         "--heldout-dir set).")
    ap.add_argument("--low", type=int, default=20)
    ap.add_argument("--high", type=int, default=99)
    ap.add_argument("--a-low", type=int, default=None)
    ap.add_argument("--a-high", type=int, default=None)
    ap.add_argument("--b-low", type=int, default=None)
    ap.add_argument("--b-high", type=int, default=None)

    # Bookkeeping.
    ap.add_argument("--out", "--work-dir", dest="work_dir", default=None,
                    help="Output directory for summary.json, harbor_jobs/, "
                         "reconnect_config.json. If omitted, a tmpdir is used.")
    ap.add_argument("--seed", type=int, default=0)
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
        rng = random.Random(args.seed)

        # ── Assemble training and held-out task pools ──────────────────────
        # Two modes: (a) user brings their own Harbor task dirs — the real
        # user flow. (b) fall back to the arithmetic generator for the demo.
        skill_paths = [Path(p) for p in args.skill_md]
        skill_dirs = [Path(p) for p in args.skill_dir]

        # Arithmetic bounds are always defined — used only when the caller
        # hasn't provided --tasks-dir or --heldout-dir.
        a_lo = args.a_low if args.a_low is not None else args.low
        a_hi = args.a_high if args.a_high is not None else args.high
        b_lo = args.b_low if args.b_low is not None else args.low
        b_hi = args.b_high if args.b_high is not None else args.high

        if args.tasks_dir:
            train_pool = _enumerate_task_dirs(Path(args.tasks_dir))
            if not train_pool:
                raise SystemExit(f"--tasks-dir {args.tasks_dir} has no Harbor tasks")
            _log(f"training pool: {len(train_pool)} tasks under {args.tasks_dir}")
        else:
            train_pool = None
            _log(f"arithmetic generator: a in [{a_lo},{a_hi}], b in [{b_lo},{b_hi}], op={args.task}")

        if args.heldout_dir:
            heldout_ds_dir = Path(args.heldout_dir).resolve()
            _log(f"held-out dataset: {heldout_ds_dir}")
        else:
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
            agent_path=args.agent,
            env_path=args.env,
            extra_instruction_paths=skill_paths,
            skill_paths=skill_dirs,
        )
        base_ds = load_job_dir(baseline_job_dir, "baseline", args.model)
        base_pass = pass_at_1(base_ds)
        _log(f"BASELINE pass@1 = {base_pass:.3f}  ({sum(1 for r in base_ds.rollouts if r.reward >= 1.0)}/{len(base_ds.rollouts)})")

        # ── 3. Training loop: N iterations of harbor rollout -> GRPO step ──
        curve: list[float] = []
        for step in range(args.iters):
            step_ds_dir = work_dir / f"dataset_step_{step:02d}"
            if train_pool is not None:
                # Sample without replacement if we have enough tasks, else with.
                k = args.prompts_per_step
                if k <= len(train_pool):
                    chosen = rng.sample(train_pool, k)
                else:
                    chosen = [rng.choice(train_pool) for _ in range(k)]
                _write_step_manifest(step_ds_dir, chosen)
            else:
                probs = [
                    (rng.randint(a_lo, a_hi), rng.randint(b_lo, b_hi))
                    for _ in range(args.prompts_per_step)
                ]
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
                agent_path=args.agent,
                env_path=args.env,
                extra_instruction_paths=skill_paths,
                skill_paths=skill_dirs,
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
            agent_path=args.agent,
            env_path=args.env,
            extra_instruction_paths=skill_paths,
            skill_paths=skill_dirs,
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
        summary = {
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
            "reconnect_config_path": str(reconnect_path),
            "model": args.model,
            "agent": args.agent,
            "env": args.env,
            "skill_md": [str(p) for p in skill_paths],
            "skill_dir": [str(p) for p in skill_dirs],
            "tasks_dir": str(Path(args.tasks_dir).resolve()) if args.tasks_dir else None,
            "heldout_dir": str(heldout_ds_dir),
            "seed": args.seed,
            "iters": args.iters,
            "prompts_per_step": args.prompts_per_step,
            "n_attempts": args.n_attempts,
            "n_heldout": len(base_ds.rollouts),
            "lr": args.lr,
            "temperature": args.temperature,
        }
        if train_pool is None:
            summary.update({
                "task": args.task,
                "a_range": [a_lo, a_hi],
                "b_range": [b_lo, b_hi],
                "heldout": args.heldout,
            })
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        _log(f"summary -> {work_dir / 'summary.json'}")
    finally:
        _log("shutting down Cortex job")
        backend.cancel()


def cli() -> None:
    """Synchronous entry point for the ``harbor-cortex-train`` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
