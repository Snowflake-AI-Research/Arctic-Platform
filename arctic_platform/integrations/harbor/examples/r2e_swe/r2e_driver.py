"""The reference recipe on Cortex Training: R2E-Gym tasks, its harness, its model.

Mirrors the rollout side of
``20260908coco-abl1-...-qwen35-4b-r2ebase-cispo-...-official``:

  * tasks       R2E-Gym-Subset_validgold_unique_baseline (the exact reference parquet)
  * sandbox     one agent-sandbox CR per rollout, from the instance's own image
  * agent       mini-swe-agent-plus, staged verbatim from the reference verifiers tree
  * interception  the only route out of a sandbox is our gateway on cni0
  * reward      run_tests.sh parsed against expected_output_json, exact match
  * training    GRPO advantages -> Cortex fwd_bwd, with sampler logprobs so a
                batch can be replayed off-policy

Deviations from the reference run, deliberate and logged: Cortex's server-side loss is
PPO-clip GRPO rather than CISPO (a Cortex-side registration is needed for the
exact objective), and the agent is mini-swe-agent-plus rather than coco, whose
binary is node-local to the reference training pod.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mini_swe_plus  # noqa: E402
import reference_paths  # noqa: E402
from curriculum import Curriculum  # noqa: E402
from r2e_grade import reward as grade_reward  # noqa: E402
from r2e_grade import run_tests  # noqa: E402
from sandbox import BRIDGE_HOST  # noqa: E402
from sandbox import Sandbox  # noqa: E402


# The reference instance prompt. The agent is told the tests exist but not shown them;
# `hide_tests_from_agent` is emulated by leaving /r2e_tests out of the repo
# until scoring.
# The reference R2EGymTaskSet.get_instruction returns the bare problem statement, and the
# harness's own INSTANCE_TEMPLATE supplies every piece of scaffolding around it:
# the working directory, "do not modify tests", one-native-tool-call-per-turn, and
# the exact submission command. Wrapping the statement ourselves nested a second
# prompt inside the harness's {{task}} slot, and the word "submit" in it competed with the
# harness's `echo MINI_SWE_AGENT_FINAL_OUTPUT` — there is no submit tool, so every
# model that took our wording at face value ended the rollout on unknown_tool.
def task_prompt(rec: dict) -> str:
    return rec["problem_statement"]


def _fixed_set(instances: list[dict], args) -> list[dict] | None:
    """The instance list for a fixed-batch overfit run, or None for normal mode.

    Overfitting is the cheapest end-to-end correctness proof we have: on a batch
    that never changes, a working loop has to drive reward up, because nothing
    stops it from memorising. A flat curve indicts the plumbing -- advantages,
    loss mask, token alignment, or the weight sync -- rather than the task mix.
    """
    if args.task_ids:
        by_id = {r["instance_id"]: r for r in instances}
        want = [t.strip() for t in args.task_ids.split(",") if t.strip()]
        missing = [t for t in want if t not in by_id]
        if missing:
            raise SystemExit(f"--task-ids not in dataset: {', '.join(missing)}")
        return [by_id[t] for t in want]
    if args.overfit:
        if args.overfit > len(instances):
            raise SystemExit(f"--overfit {args.overfit} exceeds {len(instances)} instances")
        # load_instances already shuffled under --seed, so a prefix is a
        # reproducible random draw rather than the dataset's own ordering.
        return instances[: args.overfit]
    return None


def load_instances(limit: int | None, seed: int) -> list[dict]:
    rows = []
    with open(reference_paths.r2e_dataset()) as fh:
        for line in fh:
            rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def run_one_rollout(
    rec: dict,
    *,
    rollout_id: str,
    base_url: str,
    model: str,
    max_turns: int,
    command_timeout: int,
    rollout_timeout: int,
    setup_timeout: int,
    transcript_dir: Path,
) -> dict:
    """One sandboxed attempt at one R2E instance; returns reward + metadata."""
    name = f"r2e-{uuid.uuid4().hex[:10]}"
    sb = Sandbox(image=rec["docker_image"], name=name)
    started = time.time()
    info: dict = {
        "instance_id": rec["instance_id"],
        "rollout_id": rollout_id,
        "sandbox": name,
        "reward": 0.0,
        "earned_reward": 0.0,
        "reward_override": None,
        "stop": None,
        "error": None,
    }
    try:
        sb.create(ready_timeout_s=setup_timeout)
        info["setup_s"] = round(time.time() - started, 1)

        # Park the graded tests outside the repo for the duration of the
        # rollout; an agent that can read them can pass without fixing
        # anything, which is a reward-hacking path the reference setup closes too.
        sb.exec("mv /r2e_tests /tmp/r2e_tests_hidden 2>/dev/null || true", timeout=120)

        mini_swe_plus.stage(sb, python_command="/testbed/.venv/bin/python")
        res = mini_swe_plus.run(
            sb,
            base_url=base_url,
            api_key=rollout_id,  # doubles as the capture key
            model=model,
            task=task_prompt(rec),
            working_dir="/testbed",
            command_timeout_seconds=command_timeout,
            timeout=rollout_timeout,
        )
        info["stop"] = res["stop"]
        (transcript_dir / f"{rollout_id}.log").write_text(res["stdout"][-200_000:])

        sb.exec("mv /tmp/r2e_tests_hidden /testbed/r2e_tests 2>/dev/null || true", timeout=120)
        out = run_tests(sb)
        info["reward"], got, exp = grade_reward(out, rec["expected_output_json"])
        info["tests_parsed"] = len(got)
        info["tests_expected"] = len(exp)
        info["earned_reward"] = info["reward"]

        # A rollout that broke protocol scores zero even if its patch passed the
        # tests. prime-rl enforces this with an auto-injected swe_terminal_invalid
        # filter (retain=True, reward_override=0.0), so the trace is still trained
        # on, with its loss mask intact — it just trains as a failure. Keeping the
        # earned reward here would teach the policy that an invalid trajectory is
        # worth as much as a valid one, which is the opposite of what the format
        # gates exist for.
        cond = (res["stop"] or {}).get("stop_condition")
        if cond in mini_swe_plus.terminal_stop_conditions():
            info["reward"] = 0.0
            info["reward_override"] = cond
    except Exception as exc:  # noqa: BLE001 — one bad sandbox must not kill the step
        info["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        info["total_s"] = round(time.time() - started, 1)
        try:
            sb.delete()
        except Exception:  # noqa: BLE001
            pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    # Defaults follow the reference config, read off
    # configs/train/main/20260904main_qwen35_4b_r2ebase_cispo_mops8_.../rl.toml.
    # Only the loss differs: Cortex's server-side objective is PPO-clip GRPO,
    # the reference is cispo_ref_kl_loss. The reference *advantages* are already GRPO
    # ([orchestrator.algo] type = "grpo"), so that is the sole delta.
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--prompts-per-step", type=int, default=16,
                    help="reference batch_size 128 / group_size 8")
    ap.add_argument("--group", type=int, default=8, help="reference group_size")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="reference max_inflight_rollouts is 252; ours is bounded by "
                         "sampler throughput on 2 GPUs, not by the node")
    ap.add_argument("--max-turns", type=int, default=100, help="reference agent.max_turns")
    ap.add_argument("--command-timeout", type=int, default=60)
    ap.add_argument("--rollout-timeout", type=int, default=5400, help="reference timeout.rollout")
    ap.add_argument("--setup-timeout", type=int, default=1800, help="reference timeout.setup")
    ap.add_argument("--max-seq-len", type=int, default=131072, help="reference seq_len")
    ap.add_argument("--max-tokens-per-turn", type=int, default=32768,
                    help="reference max_completion_tokens")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--adam-eps", type=float, default=1e-15, help="reference optim.eps")
    ap.add_argument("--adam-beta2", type=float, default=0.95, help="reference betas2")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="reference weight_decay")
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="sequences per fwd_bwd; a whole step will not fit in one")
    ap.add_argument("--std-norm", action="store_true",
                    help="reference std_normalization is false, so off by default")
    ap.add_argument("--port", type=int, default=19914)
    ap.add_argument("--train-gpus", type=int, default=2, help="reference num_train_gpus")
    ap.add_argument("--sample-gpus", type=int, default=2,
                    help="he uses 6; we have fewer to spend")
    ap.add_argument("--out", default="./run")
    ap.add_argument(
        "--log-mirror",
        default=None,
        help="copy of the run log on shared storage, readable while the run "
             "holds the pod's serial command channel",
    )
    ap.add_argument("--seed", type=int, default=42, help="reference buffer seed")
    ap.add_argument(
        "--chat-template",
        default=reference_paths.default_chat_template(),
        help="reference [tokenizer] chat_template, resolved under "
             f"${reference_paths.ENV_VAR}; pass empty to use the stock one",
    )
    ap.add_argument("--curriculum", default=None,
                    help="path to the persistent difficulty tally "
                         "(defaults to <out>/curriculum.json)")
    ap.add_argument("--overfit", type=int, default=0, metavar="N",
                    help="correctness test: draw N instances once and train on "
                         "that same fixed set every step, with no curriculum and "
                         "no eviction. Reward has to climb on a fixed batch, so "
                         "a flat curve here means the loop itself is broken")
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated instance_ids to use as the fixed "
                         "overfit set, instead of drawing them by seed")
    ap.add_argument("--dry-run", action="store_true",
                    help="rollouts only, no Cortex job and no training step")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    sinks = [(out_dir / "run.log").open("a")]
    if args.log_mirror:
        # ``out`` may live on node-local disk, invisible from
        # outside the pod, and the pod's command channel is serial — it is
        # blocked by the very run we want to watch. A copy on shared storage is
        # the only way to follow a step while it is in progress.
        mirror = Path(args.log_mirror)
        mirror.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(mirror.open("a"))

    def log(msg: str) -> None:
        print(msg, flush=True)
        for sink in sinks:
            sink.write(msg + "\n")
            sink.flush()

    # The backend reports fwd_bwd progress through the logging module; send it
    # to the same sinks so a long step is visible in the run log.
    class _LogSink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log(f"[{record.name.rsplit('.', 1)[-1]}] {record.getMessage()}")

    backend_log = logging.getLogger("arctic_platform.integrations.harbor")
    backend_log.setLevel(logging.INFO)
    backend_log.addHandler(_LogSink())

    instances = load_instances(None, args.seed)
    curriculum = Curriculum.load(
        args.curriculum or (out_dir / "curriculum.json"),
        # The reference hard_window is 3: three failed groups before an instance is
        # written off, so one unlucky group does not evict it.
        min_attempts=3 * args.group,
    )
    rng = random.Random(args.seed)

    fixed = _fixed_set(instances, args)
    if fixed is not None:
        args.prompts_per_step = len(fixed)
        log(f"[r2e] OVERFIT: {len(fixed)} fixed instances x group {args.group} "
            f"= {len(fixed) * args.group} rollouts/step, curriculum disabled")
        for rec in fixed:
            log(f"[r2e]   {rec['instance_id']}")
    else:
        log(f"[r2e] dataset: {len(instances)} instances | {curriculum.summary()}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template:
        # The reference [tokenizer] chat_template, with thinking_retention = "all". The
        # stock Qwen template drops prior turns' <think> blocks, so the policy
        # would be conditioned on a transcript it never produced — and every
        # turn after the first would be sampled from a different distribution
        # than the one the reference run trains on.
        tokenizer.chat_template = Path(args.chat_template).read_text()
        log(f"[r2e] chat template: {args.chat_template}")

    backend = None
    base_url = f"http://{BRIDGE_HOST}:{args.port}/v1"
    gateway = None

    if not args.dry_run:
        # Imported here, not at module scope: a dry run exercises sandboxes,
        # the harness and the grader on a CPU-only pod, and must not require
        # arctic_platform (which pulls torch and friends) to be installed.
        from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
        from arctic_platform.integrations.harbor.models import PostTrainingConfig

        from capture import CapturingGateway

        cfg = PostTrainingConfig(
            base_model=args.model,
            train_gpus=args.train_gpus,
            sample_gpus=args.sample_gpus,
            max_seq_len=args.max_seq_len,
            learning_rate=args.lr,
            n_samples_per_prompt=args.group,
            std_normalization=args.std_norm,
            micro_batch_size=args.micro_batch,
            adam_betas=(0.9, args.adam_beta2),
            adam_eps=args.adam_eps,
            weight_decay=args.weight_decay,
            cortex_host=os.environ["ARCTIC_CORTEX_HOST"],
            cortex_database=os.environ["ARCTIC_CORTEX_DATABASE"],
            cortex_schema=os.environ["ARCTIC_CORTEX_SCHEMA"],
            cortex_pat_env_var="CORTEX_PAT",
        )
        log(f"[r2e] connecting to Cortex: {args.model} {args.train_gpus}T+{args.sample_gpus}S")
        t0 = time.time()
        backend = ArcticCortexBackend(cfg)
        run = backend.connect()
        log(f"[r2e] job ready in {time.time() - t0:.0f}s training={run.training_job_id}")

        gateway = CapturingGateway(
            client=backend._client,
            tokenizer=tokenizer,
            model_name=args.model,
            host=BRIDGE_HOST,
            port=args.port,
            max_turns=args.max_turns,
            default_max_tokens=args.max_tokens_per_turn,
            raw_log=str(out_dir / "raw_completions.jsonl"),
        )
        base_url = gateway.start()
    log(f"[r2e] gateway {base_url}")

    history: list[dict] = []
    tally: dict[str, list[float]] = {}
    try:
        for step in range(args.steps):
            log(f"\n[r2e] ===== step {step} =====")
            picked = fixed if fixed is not None else curriculum.pick(
                instances, args.prompts_per_step, rng
            )
            jobs = []
            for rec in picked:
                for k in range(args.group):
                    jobs.append((rec, f"s{step}-{rec['instance_id']}-{k}"))
            if gateway is not None:
                for _, rid in jobs:
                    gateway.reset(rid)

            t0 = time.time()
            results = []
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(
                        run_one_rollout,
                        rec,
                        rollout_id=rid,
                        base_url=base_url,
                        model=args.model,
                        max_turns=args.max_turns,
                        command_timeout=args.command_timeout,
                        rollout_timeout=args.rollout_timeout,
                        setup_timeout=args.setup_timeout,
                        transcript_dir=out_dir / "transcripts",
                    ): rid
                    for rec, rid in jobs
                }
                # Log each rollout as it lands rather than after the whole
                # step. A step of 100-turn agent rollouts runs for tens of
                # minutes, and with only an end-of-step summary there is no way
                # to tell a slow step from a wedged one while it is happening.
                for done in as_completed(futures):
                    info = done.result()
                    results.append(info)
                    stop = info["stop"] or {}
                    log(f"    [{len(results)}/{len(futures)}] "
                        f"{info['instance_id']:<28} reward={info['reward']:.1f} "
                        f"stop={stop.get('stop_condition')} "
                        f"t={info.get('total_s')}s {info['error'] or ''}")
            wall = time.time() - t0
            if gateway is not None:
                for r in results:
                    r["turns"] = len(gateway.turns(r["rollout_id"]))

            rewards = [r["reward"] for r in results]
            earned = [r["earned_reward"] for r in results]
            errors = [r for r in results if r["error"]]
            n = max(len(results), 1)
            # Two different quantities: pass rate is how often the agent fixed
            # the bug, train reward is what GRPO sees after protocol violations
            # are zeroed. The gap between them is the cost of bad formatting.
            log(f"[r2e] {len(results)} rollouts in {wall:.0f}s "
                f"train_reward={sum(rewards)/n:.3f} "
                f"pass_rate={sum(earned)/n:.3f} "
                f"solved={sum(1 for x in rewards if x > 0)}/{len(rewards)} "
                f"zeroed={sum(1 for r in results if r['reward_override'])} "
                f"errors={len(errors)}")
            for r in results:
                stop = r["stop"] or {}
                zeroed = r["reward_override"]
                log(f"    {r['instance_id']:<28} reward={r['reward']:.1f} "
                    f"{'(earned 1.0, zeroed) ' if zeroed and r['earned_reward'] else ''}"
                    f"turns={r.get('turns', '?')} "
                    f"stop={stop.get('stop_condition')}"
                    f"{'/' + stop['detail'] if stop.get('detail') else ''} "
                    f"t={r.get('total_s')}s {r['error'] or ''}")

            by_instance: dict[str, list[float]] = {}
            for r in results:
                by_instance.setdefault(r["instance_id"], []).append(r["reward"])

            if fixed is None:
                # Eviction observes the post-override reward, not the earned one:
                # the reference swe_terminal_invalid filter sits in pre_batch_filters, so the
                # difficulty signal the reference buffer evicts on is already zeroed. A task
                # the agent can solve but keeps fumbling the protocol on is, to this
                # loop, genuinely unlearnable until the formatting improves.
                for inst, rs in by_instance.items():
                    curriculum.observe(inst, rs)
                curriculum.save()
                log(f"[r2e] {curriculum.summary()}")
            else:
                # On a fixed batch the per-instance trace is the whole experiment:
                # memorisation shows up as a specific instance climbing, which a
                # batch mean can hide.
                for inst in sorted(by_instance):
                    tally.setdefault(inst, []).append(
                        sum(by_instance[inst]) / len(by_instance[inst])
                    )
                log("[r2e] per-instance solve rate by step:")
                for inst in sorted(tally):
                    hist = " ".join(f"{v:.2f}" for v in tally[inst])
                    log(f"    {inst:<28} {hist}")

            history.append({"step": step, "mean_reward": sum(rewards) / max(len(rewards), 1),
                            "pass_rate": sum(earned) / n,
                            "solved": sum(1 for x in rewards if x > 0),
                            "zeroed": sum(1 for r in results if r["reward_override"]),
                            "n": len(rewards),
                            "rewards": rewards,
                            "earned_rewards": earned})
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

            if args.dry_run or gateway is None or backend is None:
                continue

            ds = _build_dataset(results, jobs, gateway, args, step)
            if ds is None:
                log("[r2e] no trainable rollouts this step; skipping update")
                continue
            log(f"[r2e] training on {len(ds.rollouts)} turn-rollouts")
            import asyncio
            metrics = asyncio.run(backend.train(ds, step=step))
            log(f"[r2e] train metrics: {metrics}")
    finally:
        if gateway is not None:
            gateway.stop()
        if backend is not None:
            # Cortex jobs bill until cancelled and only idle-time out after
            # ~35 minutes, so release them on the way out even if the step
            # above raised.
            backend.cancel()
    return 0


def _build_dataset(results, jobs, gateway, args, step):
    """One Rollout per captured assistant turn, carrying the trajectory reward.

    Per-turn rather than one flattened sequence: the chat template re-renders
    earlier turns when the agent replays them, so a concatenated transcript no
    longer tokenizes to the ids that were actually sampled, and the loss mask
    drifts. Each turn's own prompt/completion ids are exact by construction.
    """
    from arctic_platform.integrations.harbor.models import Rollout, RolloutDataset

    by_id = {r["rollout_id"]: r for r in results}
    rollouts = []
    for _, rollout_id in jobs:
        res = by_id.get(rollout_id)
        if res is None or res["error"]:
            continue
        for turn in gateway.turns(rollout_id):
            rollouts.append(Rollout(
                prompt_token_ids=turn.prompt_token_ids,
                completion_token_ids=turn.completion_token_ids,
                reward=res["reward"],
                logprobs=turn.logprobs,
                group_id=f"s{step}-{res['instance_id']}",
            ))
        gateway.reset(rollout_id)

    if not rollouts:
        return None
    # GRPO needs spread inside a group; a group where every rollout scored the
    # same contributes exactly zero gradient, so drop it rather than pay to
    # tokenize and ship it.
    keep_groups = {
        g for g in {r.group_id for r in rollouts}
        if len({r.reward for r in rollouts if r.group_id == g}) > 1
    }
    rollouts = [r for r in rollouts if r.group_id in keep_groups]
    if not rollouts:
        return None

    return RolloutDataset(
        rollouts=rollouts,
        dataset_id=f"r2e-step-{step}",
        model_name=args.model,
        tokenizer_name=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
