# Debugging sampling-zone overhead

How to reproduce and measure the sampler-side overhead in this recipe. Written
for someone with their own pod and their own Cortex credentials.

## The one thing to know before you start

**A measurement taken while rollouts are in flight is mostly queue wait, not
sampler overhead.** With 24 live rollouts a single request's latency swings
between 6 and 33 seconds, which is larger than every effect you are trying to
measure — it is why a tiny prompt and a 40k-token prompt cost the same during
collection. The only window where the sampler is up but idle is the
forward/backward phase, after collection ends and the trainer takes over. It
lasts a few minutes, arrives unannounced, and is hours apart.

So: either probe during that window (the two wrapper scripts below wait for it
for you), or stand up a sampling job that nothing else is using.

## Before you launch anything: check the lock

```bash
python3 poc/run_lock.py show
```

If that prints a run, **someone's training run is live — do not launch.** Every
`poc/launch_*.sh` force-deletes all `r2e-` pods and reaps every sandbox before
starting. That is indiscriminate: launching your own run kills the in-flight
rollouts of a run you never targeted. Two complete runs have been destroyed
this way, each costing several hours of collection.

## Prerequisites

- Your **own** Cortex PAT. Credentials live at `/data-fast/cortex.env`, which is
  pod-local scratch (mode 600) and is deliberately not on shared storage. Do
  not copy anyone else's, and do not copy yours onto `/modeling-code`.
- The pod-local venv, `/data-fast/ap-venv/bin/python`. The repo lives on shared
  Lustre; the venv and credentials do not.
- Set `CORTEX_JOB_COMMENT` before creating a job. Every job in the shared schema
  reports `submitted_by=ADMIN`, so the comment is the only signal of who owns
  it, and it is what stops someone else cancelling your run.

```bash
set -a; . /data-fast/cortex.env; set +a
export CORTEX_JOB_COMMENT="<your name> sampling overhead probe -- ask before cancelling"
```

## Step 0: the no-GPU check, first

Before paying for provisioning, confirm the gateway actually forwards what you
think it does. This runs against a stub client in about a second:

```bash
/data-fast/ap-venv/bin/python poc/gateway_check.py
```

This exists because a live run once burned five minutes of provisioning to
discover the middleware's edits never reached the router. If a parameter you
care about is being dropped, it shows up here rather than as a wall of zero
rewards.

## Step 1: get a sampler and a gateway URL

The driver stands up the Cortex job and the OpenAI-compatible gateway together,
then logs the URL:

```
[r2e] gateway http://10.42.0.1:19914/v1
```

That host is the `cni0` bridge address, reachable from the sandboxes and from
the pod. The port is `--port`, default 19914. Take the URL from the log rather
than assuming it — if another driver holds 19914 you will get
`[Errno 98] address already in use`, which means someone else's run is alive.

For overhead work you want sampling GPUs and as little else as possible:

```bash
export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl     # read-only reference
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/\
R2E-Gym-Subset_validgold_unique_baseline/train.jsonl

/data-fast/ap-venv/bin/python poc/r2e_driver.py \
  --steps 1 --group 8 --concurrency 24 \
  --max-turns 45 --max-pack-tokens 41000 \
  --train-gpus 2 --sample-gpus 6 \
  --tensor-parallel 1 --gpu-mem-util 0.85 --max-num-seqs 56 \
  --port 19914 --out ./runs/sampling-probe 2>&1 | tee ./runs/sampling-probe/run.log
```

`--tensor-parallel 1` is load-bearing. `--sample-gpus` is the sub-job's GPU
count, not the shard width: at `tensor_parallel_size > 1` the server builds one
wide engine with a single scheduler, and aggregate throughput then does not move
when you add concurrency. At 1 you get that many independent replicas, which is
the thing whose scaling you are trying to measure.

Expect provisioning to take ~300s. Note that an 8-GPU request needs a nearly
empty H200 node and can sit in `PLACING` for ~45 minutes before failing with
`reason: placement_timeout`; `poc/launch_until_placed.sh` retries that for you.

## Step 2: run the probes in the idle window

Two wrappers watch the run log for the training phase and fire automatically.
Start them right after the driver, and they will wait:

```bash
# prefix-cache hit rate, with and without replica affinity
bash poc/probe_when_idle.sh ./runs/sampling-probe http://10.42.0.1:19914/v1
# -> writes ./runs/sampling-probe/prefix_probe.txt

# per-request fixed cost, swept over concurrency
bash poc/probe_on_idle.sh ./runs/sampling-probe http://10.42.0.1:19914/v1 \
     /tmp/idle_probe.txt
```

To probe by hand instead, against an idle sampler:

```bash
PY=/data-fast/ap-venv/bin/python
URL=http://10.42.0.1:19914/v1

# Is a request slow because of its tokens, or because of itself?
$PY poc/probe_concurrency.py       "$URL" 16

# Do the data-parallel replicas actually share the load?
$PY poc/probe_sampler_scaling.py   "$URL" 24 40

# Is the sampler re-prefilling context it already holds?
$PY poc/probe_prefix_cache.py      "$URL" 40 3           # affinity on
$PY poc/probe_prefix_cache.py      "$URL" 40 3 none      # affinity off
```

## How to read them

Each probe is built to isolate one thing, so run all three before concluding.

**`probe_concurrency`** sends a deliberately tiny prompt, which strips out
prefill and leaves the fixed cost of getting one request through the gateway and
sampler. If a short request is already slow, conversation length was never the
problem. If wall time grows in step with concurrency, something is serving
requests one at a time.

**`probe_sampler_scaling`** sends prompts big enough that prefill dominates,
which is exactly what `probe_concurrency` cannot see. Aggregate tokens/sec that
stays flat as concurrency rises means one replica is doing all the work.
Scaling with concurrency means the fleet is healthy and the real limit is how
many tokens you ask it to process. For calibration, one idle 11k-token request
returning in ~0.5s is ~22k tok/s, about right for a 4B model on one H200.

**`probe_prefix_cache`** sends three requests capped at one output token, so the
timing is prefill: `cold` with a unique prefix, `warm` byte-identical to cold,
and `grow` with one more turn appended — the shape a real agent produces. If
`warm` and `grow` are not markedly faster than `cold`, turns are re-prefilling
from scratch, and the fix is cache capacity or request routing rather than
sampling. Compare the `none` run against the default: that difference is what
the `routing_key` affinity hint buys. Without it, each turn of a conversation
lands on an arbitrary replica and the prefix cache is enabled but useless.

A multi-turn agent re-sends its whole conversation every turn, so a step
processes roughly nineteen times more tokens than it has distinct context. That
is only wasted work if the cache misses, which is why this probe matters more
than its size suggests.

## Step 3: release the GPUs

**Killing the driver does not release anything.** The allocation belongs to the
Cortex job, and only an explicit cancel returns it to the pool — a SIGKILLed
driver leaves sub-jobs running and billing.

```bash
cat ./runs/sampling-probe/cortex_job_ids.json      # ids recorded at startup
/data-fast/ap-venv/bin/python poc/job_cancel.py <job_id>
```

If the driver died before writing that file — which is what happens when
placement fails — find the job by listing the schema:

```bash
/data-fast/ap-venv/bin/python poc/job_list.py --all | \
  grep -vE 'CANCELLED|FAILED|TERMINATED|COMPLETED|SUCCEEDED'
```

Use `--all`. Plain `job_list.py` filters to a fixed set of alive states that
does **not** include `PLACING`, so it can report a clean schema while your job
is holding a placement request. Check the `comment` column and never cancel a
job you did not start.

Also note the `gpus=` column reads 0 for every job, including running ones: the
list endpoint does not return `sub_job_configs`. Use the job state, not that
number, to decide whether something is live.

## Known traps

- **Any job-creating command run through `rexec.sh` needs a timeout longer than
  its own wait.** The relay's default kill will take out the process before its
  cleanup can cancel the job, leaking the allocation. This has happened.
- **Rollout lines are logged as they finish**, so early rewards are biased
  toward the fast rollouts, which are disproportionately successes. Do not read
  a step's reward before the step closes.
- **`reason=response_length` on a stopped rollout is usually a mislabel.** In
  one sample, 272 of 381 rollouts stopped at exactly the turn cap with a largest
  single response of 6,355 tokens against a 32,768 limit. The wall was the turn
  cap, not the token limit.
