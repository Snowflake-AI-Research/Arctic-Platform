#!/usr/bin/env python3
"""E2E POC: black-box agent in a sandbox, trained with Cortex Training.

Flow per step, mirroring prime-rl's SWE agent loop with Cortex swapped in for the
prime-rl trainer:

  1. sample tasks; for each, run `group` rollouts (GRPO needs within-group spread)
  2. each rollout gets a fresh k3s sandbox; the agent runs *inside* it and can
     only reach the model through the gateway on the cni0 bridge
  3. the gateway records exact prompt/completion token ids per rollout while the
     agent remains oblivious
  4. the task's test command decides the reward
  5. token ids + loss mask + reward go to ArcticCortexBackend, which computes
     GRPO advantages, runs fwd_bwd on Cortex, and syncs weights back to the
     sampler so the next step samples from the updated policy

Convergence is explicitly not the goal; closing this loop is.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture import CapturingGateway  # noqa: E402
from sandbox import BRIDGE_HOST, Sandbox  # noqa: E402
from tasks import TASKS, Task  # noqa: E402

AGENT_PATH_IN_SANDBOX = "/tmp/sandbox_agent.py"
INSTRUCTION_PATH_IN_SANDBOX = "/tmp/instruction.txt"
TRANSCRIPT_DIR = "./transcripts"


def run_rollout(
    task: Task,
    rollout_id: str,
    gateway: CapturingGateway,
    agent_source: str,
    model: str,
    port: int,
    max_turns: int,
    max_tokens: int,
    image: str,
    log,
) -> tuple[float, list, str]:
    """Run one trajectory in a fresh sandbox; return (reward, turns, note)."""
    gateway.reset(rollout_id)
    base_url = f"http://{BRIDGE_HOST}:{port}/v1"

    sandbox = Sandbox(image=image)
    try:
        sandbox.create()
        sandbox.write_file(AGENT_PATH_IN_SANDBOX, agent_source)
        sandbox.write_file(INSTRUCTION_PATH_IN_SANDBOX, task.prompt())

        code, out = sandbox.exec(task.setup)
        if code != 0:
            return 0.0, [], f"setup failed: {out[:200]}"

        command = (
            f"python {AGENT_PATH_IN_SANDBOX}"
            f" --base-url {base_url}"
            f" --api-key {rollout_id}"
            f" --model {model}"
            f" --instruction-file {INSTRUCTION_PATH_IN_SANDBOX}"
            f" --working-dir /workspace"
            f" --max-turns {max_turns}"
            f" --max-tokens {max_tokens}"
        )
        code, out = sandbox.exec(command)
        log(f"      agent exit={code}")
        transcript_dir = Path(TRANSCRIPT_DIR)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        (transcript_dir / f"{rollout_id}.txt").write_text(out)
        for line in out.splitlines()[-4:]:
            log(f"      | {line[:150]}")

        test_code, test_out = sandbox.exec(task.test)
        reward = 1.0 if test_code == 0 else 0.0
        note = "pass" if reward else f"fail: {test_out.strip()[:120]}"
        return reward, gateway.turns(rollout_id), note
    finally:
        sandbox.delete()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--group", type=int, default=4, help="rollouts per task (GRPO group)")
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--port", type=int, default=19914)
    ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--train-gpus", type=int, default=1)
    ap.add_argument("--sample-gpus", type=int, default=1)
    ap.add_argument("--out", default="./run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "run.log").open("a")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
    from arctic_platform.integrations.harbor.models import (
        PostTrainingConfig,
        Rollout,
        RolloutDataset,
    )

    agent_source = (Path(__file__).resolve().parent / "sandbox_agent.py").read_text()

    cfg = PostTrainingConfig(
        base_model=args.model,
        train_gpus=args.train_gpus,
        sample_gpus=args.sample_gpus,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        n_samples_per_prompt=args.group,
        cortex_host=os.environ["ARCTIC_CORTEX_HOST"],
        cortex_database=os.environ["ARCTIC_CORTEX_DATABASE"],
        cortex_schema=os.environ["ARCTIC_CORTEX_SCHEMA"],
        cortex_pat_env_var="CORTEX_PAT",
    )

    log(f"[poc] connecting to Cortex: {args.model}, {args.train_gpus}T+{args.sample_gpus}S")
    backend = ArcticCortexBackend(cfg)
    t0 = time.time()
    run = backend.connect()
    log(f"[poc] job ready in {time.time() - t0:.0f}s  training={run.training_job_id}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    gateway = CapturingGateway(
        client=backend._client,
        tokenizer=tokenizer,
        model_name=args.model,
        host=BRIDGE_HOST,
        port=args.port,
    )
    base_url = gateway.start()
    log(f"[poc] gateway on {base_url} (reachable from sandboxes)")

    history = []
    try:
        for step in range(args.steps):
            log(f"\n[poc] ===== step {step} =====")
            chosen = random.sample(TASKS, min(args.prompts_per_step, len(TASKS)))
            rollouts: list[Rollout] = []
            rewards: list[float] = []

            for task in chosen:
                log(f"  task {task.name}")
                for g in range(args.group):
                    rollout_id = f"s{step}-{task.name}-g{g}"
                    reward, turns, note = run_rollout(
                        task,
                        rollout_id,
                        gateway,
                        agent_source,
                        args.model,
                        args.port,
                        args.max_turns,
                        args.max_tokens,
                        args.image,
                        log,
                    )
                    log(
                        f"    g{g} reward={reward} turns={len(turns)} "
                        f"tokens={[len(t.completion_token_ids) for t in turns]} {note}"
                    )
                    if not turns:
                        log(f"    g{g}: no tokens captured, dropping")
                        continue
                    rewards.append(reward)
                    # One training sequence per assistant turn, each carrying the
                    # trajectory's reward. Flattening the whole trajectory instead
                    # would need Harbor's prompt[i+1] == prompt[i]+completion[i]
                    # invariant, which the chat template breaks when it re-renders
                    # between turns -- that fallback trains the final turn only,
                    # and for this agent the final turn is just "TASK_DONE".
                    # Per-turn keeps every action trainable and puts tool output
                    # in the prompt, where it is masked out for free.
                    for turn in turns:
                        rollouts.append(
                            Rollout(
                                prompt_token_ids=turn.prompt_token_ids,
                                completion_token_ids=turn.completion_token_ids,
                                reward=reward,
                                # Rollouts sharing a task form the GRPO group
                                # whose reward spread becomes the advantage.
                                group_id=f"s{step}-{task.name}",
                                metadata={"task": task.name, "rollout_id": rollout_id},
                            )
                        )

            if len(rollouts) < 2:
                log(f"[poc] step {step}: only {len(rollouts)} usable rollouts, skipping train")
                continue

            mean_reward = sum(rewards) / len(rewards)
            spread = max(rewards) - min(rewards)
            log(
                f"[poc] step {step}: {len(rollouts)} rollouts, mean reward {mean_reward:.3f}, "
                f"spread {spread:.3f}"
            )
            if spread == 0.0:
                log("[poc] all rewards identical -> GRPO advantages will be zero this step")

            dataset = RolloutDataset(
                rollouts=rollouts,
                dataset_id=f"poc-step-{step}",
                model_name=args.model,
                tokenizer_name=args.model,
            )
            t1 = time.time()
            # backend.train is async (fwd_bwd -> step -> sync_weights); the
            # gateway's uvicorn loop lives on its own thread, so driving a
            # private loop here doesn't contend with it.
            metrics = asyncio.run(backend.train(dataset, step=step))
            log(f"[poc] step {step}: trained in {time.time() - t1:.0f}s metrics={metrics}")
            history.append(
                {"step": step, "mean_reward": mean_reward, "n": len(rollouts), "metrics": str(metrics)}
            )

        (out_dir / "summary.json").write_text(json.dumps({"history": history}, indent=2) + "\n")
        log(f"\n[poc] done; wrote {out_dir / 'summary.json'}")
    finally:
        gateway.stop()
        backend.cancel()
        log("[poc] gateway stopped, Cortex job cancelled")

    return 0


if __name__ == "__main__":
    sys.exit(main())
