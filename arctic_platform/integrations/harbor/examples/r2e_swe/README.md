# R2E-Gym SWE agent → Cortex Training

The driver behind [R2E_SWE_RUN.md](../../R2E_SWE_RUN.md). It runs a
black-box SWE agent against R2E-Gym tasks in per-rollout containers and
trains the policy with GRPO on Cortex, with no GPU on the driver side.

This is **experiment code, not a supported entry point.** It is checked
in because it is what produced the numbers, and because the shape of it
is the argument for what a `PostTrainingBackend` protocol would need to
cover. It will not run outside our cluster as-is — see the
prerequisites.

## Shape of the loop

```
r2e_driver.py
  ├── sandbox.py        one k3s Sandbox CR per rollout
  ├── mini_swe_plus.py  stages the harness into the sandbox and runs it
  ├── capture.py        DriverOpenAIGateway + per-rollout token capture
  ├── r2e_grade.py      R2E's own test command, before/after, for reward
  └── curriculum.py     difficulty filtering (evict too-easy / too-hard)
        │
        └── ArcticCortexBackend.train()  →  Cortex fwd_bwd + weight sync
```

The agent never knows it is being trained. It speaks OpenAI HTTP to a
gateway on the cluster bridge; the gateway records the exact
`prompt_token_ids` / `completion_token_ids` / log-probs per turn,
attributed to a rollout id carried in the bearer token, and forwards the
call to Cortex. That capture is what makes a black-box agent trainable
without patching it.

## Prerequisites (why this won't run elsewhere)

* **An in-pod k3s sandbox host** with the agent-sandbox controller and a
  privileged pod (full capabilities, seccomp Unconfined). The bridge
  address the sandboxes reach the gateway on is `10.42.0.1`.
* **A prime-rl checkout**, from which `mini_swe_plus.py` stages the
  `mini-swe-agent-plus` harness verbatim rather than reimplementing it,
  and which also supplies the chat template.
* **The R2E-Gym instance table**, a site-local export of the public
  dataset.
* **Cortex credentials** in the environment (`ARCTIC_CORTEX_HOST`,
  `CORTEX_PAT`, `ARCTIC_CORTEX_DATABASE`, `ARCTIC_CORTEX_SCHEMA`).

Everything site-specific is resolved from the environment rather than
hard-coded, so nothing here assumes our filesystem layout:

| Variable | Purpose | Default |
|---|---|---|
| `PRIME_RL_ROOT` | prime-rl checkout: harness + chat template | required |
| `R2E_DATASET` | R2E-Gym instance table (`train.jsonl`) | required |
| `K3S_DIR` | k3s binary and kubeconfig | `/data-fast/k3s` |

The chat template matters more than it looks: the stock Qwen template
drops prior turns' `<think>` blocks, which would condition the policy on
a transcript it never produced.

## Running

```bash
export PRIME_RL_ROOT=/path/to/prime-rl
export R2E_DATASET=/path/to/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl

python r2e_driver.py --steps 3 --prompts-per-step 4 --group 8 \
  --concurrency 32 --micro-batch 2 \
  --curriculum ./curriculum.json --out ./run
```

Write `--out` to shared storage, not node-local disk: the transcripts
and `raw_completions.jsonl` from the first 3-step run were lost with the
pod that produced them.

Useful flags: `--dry-run` collects rollouts with no Cortex job at all,
and `--overfit N` pins a fixed batch so a correctness change can be
judged without curriculum drift.

## Checking it without burning GPUs

`smoke.py` stands a fake OpenAI server on the bridge and runs the real
agent in a real sandbox against it, exercising sandbox lifecycle,
routing, the agent loop, token capture and reward scoring — everything
except Cortex. `harness_smoke.py` does the same for the harness's
protocol validation, `gateway_check.py` proves the gateway's sampling
defaults against a stub client, and `diag_stops.py` tabulates rollout
outcomes by detector from a run log.
