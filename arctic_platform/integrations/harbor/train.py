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


def _derive_openai_base_url(client: object, model_name: str) -> tuple[str, object | None]:
    """Best-effort ``/v1`` URL from a connected ``ArcticRLClient``.

    Two cases:

    * On-prem: the client's transport carries ``base_url`` (e.g.
      ``http://localhost:7000``); the OpenAI-compat surface is served
      at ``/v1`` on the same host by
      ``arctic_platform.rl.http_server``. Return that URL and no
      gateway (nothing to clean up).
    * Cortex: sub-jobs are only reachable via SnowAPI's op-name
      dispatch, so no ``/v1/*`` route exists externally. Boot a
      driver-local ``DriverOpenAIGateway`` bound to ``127.0.0.1`` that
      re-exposes ``/v1/*`` and forwards each call to
      ``client.generate`` (which already speaks Cortex's ``operation``
      envelope). Return that local URL plus a handle the caller must
      ``stop()`` before exit.

    Returning the gateway (or ``None`` on-prem) lets ``main`` shut it
    down in the same ``finally`` block that tears down the Cortex job,
    so there's no leaked uvicorn thread after a crash.
    """
    transport = getattr(client, "transport", None)
    base_url = getattr(transport, "base_url", None)
    if base_url:
        return f"{str(base_url).rstrip('/')}/v1", None

    # Cortex path — no external HTTP endpoint. Spin up the driver-side
    # gateway so any OpenAI-compat harness (LiteLLM, openai SDK, ...)
    # can hit it locally without any Cortex control-plane change.
    from transformers import AutoTokenizer

    from arctic_platform.integrations.harbor.openai_gateway import DriverOpenAIGateway

    # Same tokenizer-cache warm-up rationale as ``CortexRLAgent``: hit
    # HF Hub once, then let every subsequent trial hit the local cache.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    gateway = DriverOpenAIGateway(client=client, tokenizer=tokenizer, model_name=model_name)
    gateway.start()
    return gateway.base_url, gateway


def _run_harbor(
    dataset_dir: Path,
    jobs_dir: Path,
    job_name: str,
    model_name: str,
    n_concurrent: int,
    n_attempts: int,
    temperature: float,
    max_tokens: int,
    agent_path: str = DEFAULT_AGENT_PATH,
    env_path: str = DEFAULT_ENV_PATH,
    reconnect_config_path: Path | None = None,
    sampling_api_base: str | None = None,
    extra_instruction_paths: list[Path] | None = None,
    skill_paths: list[Path] | None = None,
    extra_agent_kwargs: dict[str, str] | None = None,
) -> Path:
    """Invoke ``harbor run`` on a Harbor dataset. Returns the job dir.

    Two sampling modes, controlled by which of ``reconnect_config_path`` or
    ``sampling_api_base`` is set:

    * Native: ``reconnect_config_path`` — the agent (``CortexRLAgent`` by
      default) reattaches via ``ArcticRLClient.reconnect_config`` and
      calls the sub-job's RL-shaped ``/generate`` route.
    * OpenAI-compat: ``sampling_api_base`` — the runner passes
      ``--model-base-url`` and ``--ak api_base=`` to ``harbor run``, so
      any ``BaseAgent`` that speaks OpenAI-chat (Terminus 2 etc.) works
      unchanged. Requires Cortex to expose ``/v1/chat/completions`` on
      the sampling sub-job (see ``PLAN.md``).

    ``extra_instruction_paths`` maps to Harbor's ``--extra-instruction-path``;
    ``skill_paths`` maps to ``--skill``.
    """
    if (reconnect_config_path is None) == (sampling_api_base is None):
        raise ValueError(
            "exactly one of reconnect_config_path (native mode) or "
            "sampling_api_base (OpenAI-compat mode) must be set"
        )
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
    ]
    if reconnect_config_path is not None:
        cmd += [
            "--ak", f"reconnect_config_path={reconnect_config_path}",
            "--ak", f"temperature={temperature}",
            "--ak", f"max_tokens={max_tokens}",
        ]
    else:
        # OpenAI-compat mode. Harbor exposes agent constructor kwargs
        # through ``--ak key=value`` (``--agent-kwarg``); we pass
        # ``api_base`` and the sampling knobs the agent needs. Any
        # Harbor agent whose ``__init__`` accepts ``api_base``
        # (Terminus 2, LiteLLMChatAgent, community agents that follow
        # the LLMBackend convention) picks this up unchanged.
        cmd += [
            "--ak", f"api_base={sampling_api_base}",
            "--ak", f"temperature={temperature}",
            "--ak", f"max_tokens={max_tokens}",
        ]
    for k, v in (extra_agent_kwargs or {}).items():
        cmd += ["--ak", f"{k}={v}"]
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
    """Symlink ``task_dirs`` into ``dataset_dir`` for Harbor's ``-p`` flag.

    Harbor's ``DatasetConfig._get_local_task_configs`` walks ``path.iterdir()``
    (it does *not* honor a ``dataset.toml`` manifest for the ``harbor run -p``
    directory form), so the sampled step dataset has to look like a real
    dataset dir with per-task subdirectories. Symlinks keep this cheap and
    avoid copying task assets each step.
    """
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        for entry in dataset_dir.iterdir():
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for td in task_dirs:
        target = Path(td).resolve()
        link = dataset_dir / target.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)


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
                    help="Agent import path (module:Class) or a short name "
                         "registered under harbor.plugins. Defaults to "
                         "CortexRLAgent (native reconnect-config mode). "
                         "In OpenAI-compat mode, pass e.g. "
                         "harbor.agents.terminus_2:Terminus2.")
    ap.add_argument("--env", default=DEFAULT_ENV_PATH,
                    help="Environment import path. Defaults to HostEnvironment "
                         "(no container). Swap for DockerEnvironment / Modal / "
                         "Daytona in production.")
    ap.add_argument("--sampling-api-base", default=None,
                    help="OpenAI-compat mode: sampling endpoint base URL "
                         "(e.g. http://localhost:7000/v1 for on-prem, or "
                         "https://.../sub-jobs/<id>/v1 for Cortex). Pass "
                         "'auto' to derive it from the connected client's "
                         "transport (on-prem only today). When set, harbor "
                         "run gets --model-base-url and --ak api_base= "
                         "instead of the reconnect-config plumbing.")
    ap.add_argument("--llm-backend", default=None,
                    help="Optional --ak llm_backend=<value> forwarded to the "
                         "agent (e.g. 'litellm'). Only used in OpenAI-compat "
                         "mode.")

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

    # Sizing knobs — defaults fit the PR #66 Qwen3-0.6B/arithmetic reference;
    # bump for bigger models or longer completions (reasoning-gym etc.).
    ap.add_argument("--train-gpus", type=int, default=1,
                    help="GPUs for the Cortex training sub-job (default: 1).")
    ap.add_argument("--sample-gpus", type=int, default=1,
                    help="GPUs for the Cortex sampling sub-job (default: 1).")
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="Max sequence length (prompt + completion) fed to the "
                         "trainer / vLLM. Bump for chain-of-thought benchmarks "
                         "(default: 512).")
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
        max_seq_len=args.max_seq_len,
        train_gpus=args.train_gpus,
        sample_gpus=args.sample_gpus,
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

    # Mode selection: native (reconnect_config) vs OpenAI-compat (api_base).
    openai_compat_mode = args.sampling_api_base is not None
    gateway: object | None = None
    if openai_compat_mode:
        if args.sampling_api_base == "auto":
            args.sampling_api_base, gateway = _derive_openai_base_url(backend._client, args.model)
            if gateway is not None:
                _log(f"driver-side OpenAI-compat gateway up at {args.sampling_api_base}")
            else:
                _log(f"auto-derived sampling api base: {args.sampling_api_base}")
        else:
            _log(f"OpenAI-compat mode: sampling via {args.sampling_api_base}")
        reconnect_path = None
        extra_agent_kwargs: dict[str, str] = {}
        if args.llm_backend:
            extra_agent_kwargs["llm_backend"] = args.llm_backend
    else:
        # Native mode: dump reconnect config so each Harbor trial (spawned by
        # harbor run) can reattach to the same running Cortex sub-jobs. The
        # job-id fields are ``Field(exclude=True)`` in the legacy config so
        # ``model_dump_json`` drops them — add them back explicitly.
        reconnect_cfg = backend._client.reconnect_config()
        cfg_dict = json.loads(reconnect_cfg.model_dump_json())
        cfg_dict["training_job_id"] = reconnect_cfg.training_job_id
        cfg_dict["sampling_job_id"] = reconnect_cfg.sampling_job_id
        cfg_dict["log_prob_job_id"] = reconnect_cfg.log_prob_job_id
        reconnect_path = work_dir / "reconnect_config.json"
        reconnect_path.write_text(json.dumps(cfg_dict))
        _log(f"reconnect config -> {reconnect_path}  (train_job_id={cfg_dict['training_job_id']!r})")
        extra_agent_kwargs = {}

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
            sampling_api_base=args.sampling_api_base,
            model_name=args.model,
            n_concurrent=args.n_concurrent,
            n_attempts=1,
            temperature=0.0,
            max_tokens=args.max_tokens,
            agent_path=args.agent,
            env_path=args.env,
            extra_instruction_paths=skill_paths,
            skill_paths=skill_dirs,
            extra_agent_kwargs=extra_agent_kwargs,
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
                sampling_api_base=args.sampling_api_base,
                model_name=args.model,
                n_concurrent=args.n_concurrent,
                n_attempts=args.n_attempts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                agent_path=args.agent,
                env_path=args.env,
                extra_instruction_paths=skill_paths,
                skill_paths=skill_dirs,
                extra_agent_kwargs=extra_agent_kwargs,
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
            sampling_api_base=args.sampling_api_base,
            model_name=args.model,
            n_concurrent=args.n_concurrent,
            n_attempts=1,
            temperature=0.0,
            max_tokens=args.max_tokens,
            agent_path=args.agent,
            env_path=args.env,
            extra_instruction_paths=skill_paths,
            skill_paths=skill_dirs,
            extra_agent_kwargs=extra_agent_kwargs,
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
            "reconnect_config_path": str(reconnect_path) if reconnect_path else None,
            "sampling_api_base": args.sampling_api_base,
            "mode": "openai-compat" if openai_compat_mode else "native",
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
        if gateway is not None:
            _log("stopping OpenAI-compat gateway")
            try:
                gateway.stop()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — cleanup best-effort
                _log(f"gateway.stop raised {type(exc).__name__}: {exc}")
        _log("shutting down Cortex job")
        backend.cancel()


def cli() -> None:
    """Synchronous entry point for the ``harbor-cortex-train`` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
