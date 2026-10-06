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
import asyncio
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mini_swe_plus  # noqa: E402
import reference_paths  # noqa: E402
from curriculum import Curriculum  # noqa: E402
from pack import pack_trajectory_exact  # noqa: E402
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
    grading_policy: dict | None = None,
    zero_on_violation: bool = True,
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
            grading_policy=grading_policy,
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
        # Derived from the same policy the harness was run under, so the set
        # of conditions we zero on cannot drift from the set it enforces.
        if cond in mini_swe_plus.terminal_stop_conditions(grading_policy):
            info["reward_override"] = cond
            # The reference's own overfit config detects these conditions but
            # does not zero on them (zero_reward_on_{format_error,truncation,
            # repetition} = false, format_error_penalty = 0.0). Zeroing cost us
            # a third of the batch -- pass_rate 0.438 against train_reward 0.266
            # -- so when the goal is to show the reward curve move, keep the
            # earned reward and let the gates stay diagnostic.
            if zero_on_violation:
                info["reward"] = 0.0
    except Exception as exc:  # noqa: BLE001 — one bad sandbox must not kill the step
        info["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        info["total_s"] = round(time.time() - started, 1)
        try:
            sb.delete()
        except Exception:  # noqa: BLE001
            pass
    return info


class _TrainingKeepalive:
    """Keep the training sub-job's session alive while collection runs.

    Cortex applies an idle timeout per sub-job but terminates the whole *job*
    when one trips: a 4h16m collection left the training sub-job untouched, it
    was reclaimed after 2h42m, and that killed the actively-serving sampler with
    it -- the batch was lost at the moment we tried to train on it. Collection
    only ever talks to the sampling sub-job, so the training side has to be
    touched on its own schedule.

    ``save`` is the only training op Cortex exposes that neither accumulates
    gradient nor applies an optimizer step, so it is the one safe thing to call
    mid-collection. Failures are logged and swallowed: a keepalive that takes the
    run down with it is worse than the idle timeout it guards against.
    """

    def __init__(self, backend: Any, log: Any, interval: float = 1200.0) -> None:
        self._log = log
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client = self._resolve(backend)

    @staticmethod
    def _resolve(backend: Any) -> Any:
        """Walk down to whichever wrapper exposes a synchronous ``save_checkpoint``."""
        node = getattr(backend, "_client", None)
        for _ in range(3):
            if node is None:
                return None
            fn = getattr(node, "save_checkpoint", None)
            if callable(fn) and not asyncio.iscoroutinefunction(fn):
                return node
            node = getattr(node, "_client", None)
        return None

    def __enter__(self) -> _TrainingKeepalive:
        if self._client is None:
            self._log("[keepalive] no sync save_checkpoint found; training may idle out")
            return self
        self._thread = threading.Thread(
            target=self._loop, name="training-keepalive", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            # Collection is over and training is next, so never leave a save
            # in flight against the sub-job we are about to drive.
            self._thread.join(timeout=300.0)

    def _loop(self) -> None:
        n = 0
        while not self._stop.wait(self._interval):
            n += 1
            try:
                t0 = time.time()
                self._client.save_checkpoint(checkpoint_id=f"keepalive-{int(t0)}")
                self._log(f"[keepalive] training touched ({n}) in {time.time() - t0:.1f}s")
            except Exception as exc:  # noqa: BLE001 - never kill the run
                self._log(f"[keepalive] touch {n} failed: {type(exc).__name__}: {exc}")


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
    # Measured ceiling, not a guess: four OOMs across 2 and 4 training GPUs fit
    # total GiB = 87.6 + 1.083 per 1k tokens against 139.80 GiB per H200, so
    # ~41k tokens is where a step stops fitting. 0 disables the guard.
    ap.add_argument("--max-pack-tokens", type=int, default=41000,
                    help="drop packed trajectories longer than this (0=off)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--adam-eps", type=float, default=1e-15, help="reference optim.eps")
    ap.add_argument("--adam-beta2", type=float, default=0.95, help="reference betas2")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="reference weight_decay")
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="sequences per fwd_bwd; a whole step will not fit in one")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                    help="how much room the sampler's prefix cache gets; must "
                         "leave a few GiB free for the weight-sync broadcast "
                         "buffer, which only allocates after the first step")
    ap.add_argument("--max-num-seqs", type=int, default=56,
                    help="reference max_num_seqs")
    ap.add_argument("--tensor-parallel", type=int, default=1,
                    help="shard width per sampling engine; 1 turns the sampler's "
                         "GPUs into that many independent replicas, which is "
                         "what makes concurrent rollouts concurrent")
    # The coco run sets this false; the mini-swe-agent-plus run we actually
    # reproduce sets it true, and that is the one to match.
    ap.add_argument("--no-std-norm", dest="std_norm", action="store_false",
                    help="mean-centre advantages only, without dividing by the "
                         "group standard deviation")
    ap.set_defaults(std_norm=True)
    # One masked sequence per trajectory is the reference's batch unit and the
    # difference between ~100 and ~2000 sequences in a step. Per-turn remains
    # available for comparison, and is used automatically for any trajectory
    # that fails the append-only check.
    ap.add_argument("--no-pack", action="store_true",
                    help="train each assistant turn as its own sequence")
    ap.add_argument("--pack-fallback-per-turn", action="store_true",
                    help="on a failed append-only check, train the trajectory "
                         "as per-turn rows instead of dropping it; this "
                         "weights it by its turn count, which destabilised "
                         "the lr 1e-5 run")
    ap.add_argument("--send-logprobs", action="store_true",
                    help="ship sampler log-probs as old_log_probs_shifted, so "
                         "the loss computes a real importance ratio; required "
                         "by --mops above 1")
    ap.add_argument("--mops", type=int, default=1,
                    help="optimizer steps per collection (reference "
                         "max_off_policy_steps = 8). Above 1 the later steps "
                         "are off-policy, which is what makes the clip active "
                         "and amortises a two-hour collection over more than "
                         "one update. Needs --send-logprobs")
    ap.add_argument("--port", type=int, default=19914)
    ap.add_argument("--train-gpus", type=int, default=2, help="reference num_train_gpus")
    ap.add_argument("--sample-gpus", type=int, default=2,
                    help="he uses 6; we have fewer to spend")
    # A large GPU request can sit in awaiting_fleet_capacity for longer than the
    # client's default wait. The client then gives up, but the job it created
    # keeps its place in the queue and starts anyway -- burning GPUs with no
    # driver attached. Size this to how long we are willing to queue.
    ap.add_argument("--job-ready-timeout", type=float, default=1800.0,
                    help="seconds to wait for the Cortex job to reach RUNNING")
    # The observed idle timeout was 2h42m and it takes the whole job down, not
    # just the sub-job that idled, so stay well inside it.
    ap.add_argument("--keepalive-interval", type=float, default=1200.0,
                    help="seconds between touches of the training sub-job during collection")
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
    ap.add_argument("--no-zero-on-violation", action="store_true",
                    help="keep the earned reward when a format/truncation/repetition "
                         "gate trips (the reference overfit config's behaviour)")
    ap.add_argument("--allow-content", action="store_true",
                    help="permit prose between the reasoning block and the "
                         "tool call, as the R2E-160 ablation config does. The "
                         "mini-swe reference run forbids it, and it is our "
                         "single largest source of terminal-invalid rollouts")
    ap.add_argument("--explore-frac", type=float, default=0.25,
                    help="share of each step spent on instances with no known "
                         "spread. The rest goes to instances already shown to "
                         "be solved sometimes, which are the only ones that "
                         "produce a gradient")
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

    # Silently replaying a batch the loss cannot measure drift against would
    # take eight uncorrected on-policy steps at once, which is the fastest way
    # known to this project to destroy a policy.
    if args.mops > 1 and not args.send_logprobs:
        raise SystemExit("--mops > 1 requires --send-logprobs")

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

    grading_policy = mini_swe_plus.policy_with(args.allow_content)
    if args.allow_content:
        log("[r2e] grading: content between reasoning and tool call ALLOWED "
            "(R2E-160 ablation setting)")

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
            max_off_policy_steps=args.mops,
            cortex_host=os.environ["ARCTIC_CORTEX_HOST"],
            cortex_database=os.environ["ARCTIC_CORTEX_DATABASE"],
            cortex_schema=os.environ["ARCTIC_CORTEX_SCHEMA"],
            cortex_pat_env_var="CORTEX_PAT",
            gpu_memory_utilization=args.gpu_mem_util,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel,
            job_ready_timeout=args.job_ready_timeout,
        )
        log(f"[r2e] connecting to Cortex: {args.model} {args.train_gpus}T+{args.sample_gpus}S")
        t0 = time.time()
        backend = ArcticCortexBackend(cfg)
        run = backend.connect()
        log(f"[r2e] job ready in {time.time() - t0:.0f}s training={run.training_job_id}")
        # Killing the driver does not release the GPUs: the allocation belongs
        # to the Cortex job, and only an explicit cancel returns it to the
        # pool. Record the ids where a later cancel can find them, since a
        # driver that has been SIGKILLed cannot report them itself.
        (Path(args.out) / "cortex_job_ids.json").write_text(json.dumps({
            "training": str(run.training_job_id),
            "sampling": str(run.sampling_job_id),
        }, indent=2))

        gateway = CapturingGateway(
            client=backend._client,
            tokenizer=tokenizer,
            model_name=args.model,
            host=BRIDGE_HOST,
            port=args.port,
            # Every agent samples through this gateway at the same time, and a
            # pool smaller than the rollout fan-out throttles the step without
            # reporting anything. Headroom above the fan-out costs only stacks.
            max_inflight=max(64, args.concurrency * 2),
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
                instances, args.prompts_per_step, rng,
                explore_frac=args.explore_frac,
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
            with _TrainingKeepalive(
                backend, log, interval=args.keepalive_interval
            ), ThreadPoolExecutor(max_workers=args.concurrency) as pool:
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
                        grading_policy=grading_policy,
                        zero_on_violation=not args.no_zero_on_violation,
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

            # Turn count is recorded because the terminal-invalid rate on its
            # own is not interpretable: it confounds how often the policy
            # breaks the protocol on a given turn with how many turns it takes.
            # A policy whose per-turn behaviour is unchanged still looks like
            # it is regressing if its trajectories grow, and that is exactly
            # what the early steps of this run do.
            turn_counts = [r.get("turns") for r in results if isinstance(r.get("turns"), int)]
            history.append({"step": step, "mean_reward": sum(rewards) / max(len(rewards), 1),
                            "pass_rate": sum(earned) / n,
                            "solved": sum(1 for x in rewards if x > 0),
                            "zeroed": sum(1 for r in results if r["reward_override"]),
                            "n": len(rewards),
                            "mean_turns": (sum(turn_counts) / len(turn_counts)) if turn_counts else None,
                            "rewards": rewards,
                            "earned_rewards": earned})
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

            if args.dry_run or gateway is None or backend is None:
                continue

            ds = _build_dataset(results, jobs, gateway, args, step, log, tokenizer)
            if ds is None:
                log("[r2e] no trainable rollouts this step; skipping update")
                continue
            log(f"[r2e] training on {len(ds.rollouts)} turn-rollouts")
            import asyncio
            try:
                metrics = asyncio.run(backend.train(ds, step=step))
                log(f"[r2e] train metrics: {metrics}")
            except Exception as exc:  # noqa: BLE001
                # A step costs ~2h of collection, so one unusable batch must
                # not end the run: skip it and keep the rollouts we can still
                # train on. A dead session is the exception -- every later
                # request would fail too, so stop rather than spin.
                if "cancelled" in str(exc).lower() or "terminal state" in str(exc):
                    log(f"[r2e] session gone, stopping: {exc}")
                    raise
                log(f"[r2e] TRAIN FAILED on step {step}, continuing: {exc}")
    finally:
        if gateway is not None:
            gateway.stop()
        if backend is not None:
            # Cortex jobs bill until cancelled and only idle-time out after
            # ~35 minutes, so release them on the way out even if the step
            # above raised.
            backend.cancel()
    return 0


def _dump_turn_ids(args, step, rollout_id, turns):
    """Record the exact ids each turn was sampled with.

    Two things need this and neither can be reconstructed afterwards from text,
    because re-rendering the conversation through the chat template does not
    have to reproduce the original tokenisation: auditing that a turn's prompt
    really is its predecessor's prompt plus completion, and packing a whole
    trajectory into one masked sequence on the strength of that.
    """
    out = Path(args.out) / "turn_ids"
    out.mkdir(parents=True, exist_ok=True)
    payload = [
        {"prompt": t.prompt_token_ids, "completion": t.completion_token_ids}
        for t in turns
    ]
    (out / f"s{step}-{rollout_id}.json").write_text(json.dumps(payload))


def _build_dataset(results, jobs, gateway, args, step, log, tokenizer):
    """One Rollout per trajectory where possible, else one per assistant turn.

    Packing a trajectory needs each turn to be an append-only extension of the
    previous one. Compared in token ids that almost never holds, because
    tokenising a long context merges across boundaries that incremental
    sampling left split. Compared as text it holds essentially always. So the
    sequence is assembled in text space with each completion's sampled ids
    spliced in unchanged, which keeps the trained spans exact while letting the
    masked context re-segment freely. Trajectories that really do diverge fall
    back to per-turn, which costs tokens but stays correct.
    """
    from arctic_platform.integrations.harbor.models import Rollout, RolloutDataset

    by_id = {r["rollout_id"]: r for r in results}
    rollouts = []
    n_packed = 0
    pack_failures: list[str] = []
    over_length: list[tuple[str, int]] = []
    for _, rollout_id in jobs:
        res = by_id.get(rollout_id)
        if res is None or res["error"]:
            continue
        turns = list(gateway.turns(rollout_id))
        if not turns:
            continue
        # Trajectory-level counts, defined as the reference defines them: every
        # generated token across all turns, but context counted once via the
        # final turn, since each turn's prompt already contains the previous
        # ones. These drive both the group statistics and the length penalty.
        stats = {
            "num_turns": float(len(turns)),
            "num_output_tokens": float(
                sum(len(t.completion_token_ids) for t in turns)
            ),
            "num_total_tokens": float(
                len(turns[-1].prompt_token_ids)
                + len(turns[-1].completion_token_ids)
            ),
        }
        _dump_turn_ids(args, step, rollout_id, turns)

        if not args.no_pack:
            pairs = [(t.prompt_token_ids, t.completion_token_ids) for t in turns]
            packed, err = pack_trajectory_exact(
                pairs,
                tokenizer,
                [t.logprobs for t in turns] if args.send_logprobs else None,
            )
            if packed is not None and args.max_pack_tokens > 0 and (
                len(packed.input_ids) > args.max_pack_tokens
            ):
                # The turn cap bounds turns, not tokens: a single turn can
                # return a huge file or test log. Step 2 of conv15 packed one
                # 52,543-token trajectory and OOM'd the trainer, discarding a
                # 2.2h collection whose other 101 trajectories fit. Dropping
                # the one over-length trajectory costs ~1% of the batch; not
                # dropping it costs the step.
                over_length.append((rollout_id, len(packed.input_ids)))
                gateway.reset(rollout_id)
                continue
            if packed is not None:
                n_packed += 1
                head = len(turns[0].prompt_token_ids)
                rollouts.append(Rollout(
                    metadata={"traj_id": rollout_id, "traj_stats": stats},
                    # The whole conversation is one sequence; the mask, not the
                    # prompt/completion split, is what selects trained tokens.
                    prompt_token_ids=packed.input_ids[:head],
                    completion_token_ids=packed.input_ids[head:],
                    loss_mask=packed.loss_mask,
                    # The backend lays log-probs down from the prompt boundary,
                    # so hand it the slice that starts there rather than the
                    # whole-sequence array.
                    logprobs=(
                        packed.logprobs[head:] if packed.logprobs is not None else None
                    ),
                    reward=res["reward"],
                    group_id=f"s{step}-{res['instance_id']}",
                ))
                continue
            # Falling back to per-turn keeps the step trainable, but the
            # divergence is worth knowing about: it means the context the
            # trainer would see is not the one the sampler saw.
            pack_failures.append(f"{rollout_id}: {err}")

            # Dropped rather than expanded, because there is no per-rollout
            # loss weight to compensate with. A packed trajectory is one
            # sequence; the same trajectory as per-turn rows is one sequence
            # per turn, each carrying the full trajectory reward, so it enters
            # the gradient with its turn count as a multiplier. In the lr 1e-5
            # run a single 100-turn fallback supplied 100 of step 0's 134
            # sequences -- three quarters of the update came from one rollout,
            # grad_norm went from ~1 to 15, and the policy degenerated to
            # one-turn outputs within six steps. Losing one trajectory in
            # sixty is cheap; weighting one sixty-fold is not.
            if not args.pack_fallback_per_turn:
                gateway.reset(rollout_id)
                continue

        for turn in turns:
            rollouts.append(Rollout(
                metadata={"traj_id": rollout_id, "traj_stats": stats},
                prompt_token_ids=turn.prompt_token_ids,
                completion_token_ids=turn.completion_token_ids,
                reward=res["reward"],
                # Sending sampler log-probs makes Cortex compute a real
                # importance ratio. On this single-epoch GRPO path π_old is
                # π_new, so the ratio is 1 by construction and the only thing
                # the log-probs buy is an approx_kl that is sensitive to any
                # drift between what the sampler saw and what the trainer
                # re-tokenizes. Off by default: opt in when diagnosing that.
                logprobs=turn.logprobs if args.send_logprobs else None,
                group_id=f"s{step}-{res['instance_id']}",
            ))
        gateway.reset(rollout_id)

    if not args.no_pack:
        fate = "fell back to per-turn" if args.pack_fallback_per_turn else "dropped"
        log(f"[r2e] packed {n_packed} trajectories into one sequence each"
                   f"; {len(pack_failures)} {fate}")
        for line in pack_failures[:5]:
            log(f"[r2e]   append-only check failed: {line}")
        if over_length:
            worst = max(n for _, n in over_length)
            log(f"[r2e] dropped {len(over_length)} trajectories over "
                f"{args.max_pack_tokens:,} tokens (longest {worst:,})")

    if not rollouts:
        return None
    # GRPO needs spread inside a group; a group where every rollout scored the
    # same contributes exactly zero gradient, so drop it rather than pay to
    # tokenize and ship it.
    all_groups = {r.group_id for r in rollouts}
    keep_groups = {
        g for g in all_groups
        if len({r.reward for r in rollouts if r.group_id == g}) > 1
    }
    log(f"[r2e] groups with spread: {len(keep_groups)}/{len(all_groups)}"
               f" (a group with none contributes no gradient)")
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
