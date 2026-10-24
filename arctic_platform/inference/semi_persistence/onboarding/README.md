# Semi-persistence model onboarding

Answers three questions about a model served with semi-persistence (`semi_p`),
by running it next to plain vLLM on the same backend image:

1. **Correctness:** does semi_p return exactly what an engine with the same
   numerics returns, and where does it differ from stock vLLM?
2. **Throughput and latency:** are TTFT, TPOT, latency under load and aggregate
   throughput unchanged against stock vLLM, and does load spread across replicas?
3. **Start-up:** how long does each phase take, and how much does semi_p save?

One command per model (or a list of models) submits the jobs, waits for them,
runs the checks, cancels them and writes a Markdown and a JSON report.
## The four jobs

| Job | What it is | Size | Used for |
|---|---|---|---|
| `on_hit` | semi_p restored from the published image | perf layout | the reference; perf |
| `off_default` | plain vLLM as shipped | perf layout | perf baseline; correctness, reported |
| `on_miss` | semi_p with published images turned off (`SEMIP_IMAGE_SOURCE=""`) and `SEMIP_DUMP_PUBLISH=1`: it cold-starts, dumps and restores its own fresh image, and never publishes it (the engine does not publish with the mirror off) | published | correctness, gated |
| `off_matched` | plain vLLM given semi_p's collective setup (below) | perf layout | correctness, gated; perf breakdown |

`on_miss` and `off_matched` run the same numerics as `on_hit`, so all three must
agree token for token; a difference among them is a defect. Which pair differs
says where: `on_miss` against `on_hit` points at the published image (publish or
restore), `off_matched` against both points at a setup difference between the
semi_p engine and plain vLLM. `off_default` runs stock collectives and is
expected to differ at TP >= 2; the report gives where it first differs.

`off_matched` is not run at TP = 1 (semi_p changes nothing there, so it equals
`off_default`) or for a pod-spanning engine (its NCCL network pinning cannot be
matched on plain workers). The perf layout is twice the published `n_gpus` for a
whole-node config, so two pods restore the same image, and the published size
otherwise (`copies=` overrides).

### semi_p's collective setup at TP >= 2

The semi_p engine moves vLLM's collectives onto paths a checkpoint can carry: no
fused allreduce + RMSNorm, NVLS, symmetric-memory or FlashInfer allreduce, no MoE
shared-experts side stream, sleep mode on, and a worker class that refuses the
FlashInfer allreduce workspace and forces CustomAllreduce's copy path while CUDA
graphs are captured. Each changes a reduction order, enough to flip near-tie
greedy tokens. `off_matched` asks for all of it: the environment variables and
`pass_config` directly, and the rest with `SEMIP_MATCH_COLLECTIVES=1`, which the
backend applies when it supports it. The report says whether it did; if not,
`off_matched` is reported instead of gated.

## Prerequisites

- The `cortex-training` client in the Python that runs the suite
  (`pip install cortex-training`, or `uv pip install -e .` in a cortex-training
  checkout), a connection file (cortex-training's
  `examples/config/connection.json.template`), and the commands run from the
  arctic-platform repository root.
- **The model's semi_p image must already be published** for the exact config in
  the model file and for the backend image under test (see
  [Publishing images](#publishing-images)). The config fields that key the image
  (model, `max_seq_len`, `n_gpus`, every `vllm_config` field) must match the job
  that dumped it. A semi_p job that finds no image fails, telling you to set
  `semi_p` to false or add `SEMIP_DUMP_PUBLISH=1`, so `on_hit` fails the run.
- Free GPUs for all of a model's jobs at once: `copies x n_gpus` for each perf job
  (`on_hit`, `off_default`, `off_matched`) plus `n_gpus` for `on_miss`.
- `on_miss` leaves its fresh image on the node's local disk; point
  `miss_image_cache` at a scratch root you clean up.

## Commands

```bash
# One model
python -m arctic_platform.inference.semi_persistence.onboarding.onboard config=connection.json \
    models=qwen36-35b-a3b-tp1 miss_image_cache=SCRATCH_ROOT

# Several models: run concurrently within max_gpus, largest first
python -m arctic_platform.inference.semi_persistence.onboarding.onboard config=connection.json \
    models=glm53-tp16,glm53-tp8,qwen38-flash-next-tp8 max_gpus=96

# Re-run the checks against jobs that are already running (kept afterwards)
python -m arctic_platform.inference.semi_persistence.onboarding.onboard config=connection.json models=qwen38-27b-tp1x8 \
    job_on_hit=JOB job_off_default=JOB [job_on_miss=JOB job_off_matched=JOB]
```

Each model's jobs run on the backend image listed for it in
`models/backends.json`, the image its semi_p images are published on; a
model with no entry stops the run before anything is submitted.
`debug_image_tag` overrides the file for every model. Either way the suite sets
`CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS=1` for you.

| Argument | Default | Meaning |
|---|---|---|
| `models` | required | Comma-separated names under `models/`, or paths to model files |
| `debug_image_tag` | the model's entry in `backends.json` | Backend image for every job |
| `variants` | all four | Subset of `on_hit,off_default,on_miss,off_matched`; the first two are required |
| `copies` | 2 for whole-node configs, else 1 | Perf pair size, as a multiple of the published `n_gpus` |
| `miss_image_cache` | none | Local image-cache root for `on_miss`; a per-run subdirectory is used, so a copy left by an earlier job cannot turn the miss into a hit |
| `checks` | `correctness,perf` | Run a subset |
| `require_restore` | `true` | Stop if `on_hit` cold-starts |
| `max_gpus` | free GPUs at start, minus `reserve_gpus` | GPU budget for concurrent models |
| `max_nodes` | none | Node budget; each job counts as whole nodes, so a 1-GPU job costs one |
| `reserve_gpus` | `0` | Submit only while at least this many GPUs stay free for others |
| `gpus_per_node` | `8` | For the node count |
| `out_dir` | `onboarding_reports` | Where reports go |
| `keep_jobs` | `false` | Leave the jobs running |
| `job_<variant>` | none | Attach to running jobs instead of submitting (`on_job_id` / `off_job_id` still work) |
| `log_files_on`, `log_files_off` | none | Saved GPU pod logs of `on_hit` / `off_default` (per-step restore breakdown) |
| `timeout_s` | `3600` | How long to wait for the jobs to reach `running` |

### Running unattended on a shared cluster

The command needs no interaction once started; run it under `nohup` (and
`caffeinate -i` on a Mac, so the machine does not sleep). Before every model it
waits until that model's jobs fit within `max_gpus` / `max_nodes` with
`reserve_gpus` still free, so it never takes the last GPUs on the cluster, and
it cancels each model's jobs as soon as that model is done.

Every job is appended to `<out_dir>/jobs-<time>.jsonl` when submitted and when
cancelled. SIGINT, SIGTERM and SIGHUP cancel every job still held before the
process exits. If it dies harder than that (a crash, a machine that slept or lost
its connection), cancel what it left behind:

```bash
python -m arctic_platform.inference.semi_persistence.onboarding.cleanup config=connection.json \
    ledger=onboarding_reports/jobs-TIME.jsonl [dry_run=true]
```

## Publishing images

A semi_p job that misses (no image on the node, none published) fails unless
its `vllm_config.extra_env` sets `SEMIP_DUMP_PUBLISH=1`. With it, the engine
cold-starts, dumps, restores and serves, then publishes the image to the
cluster's model bucket in the background; the `neutrino-model-cache` DaemonSet
puts it on every node within minutes. The engine does not publish when the
mirror is off (`SEMIP_IMAGE_SOURCE=""`) or when the key is published already,
and at TP>1 it publishes only a whole-node pod (8 GPUs). At TP=1 every pod size
has its own key (`n_gpus=1` the flat `-tp1` image, 8 the `-tp1x8` one, each
replica restoring its own `replica<K>/`), because the restore's GPU map covers
every GPU in the pod; a TP=1 pod of 2-7 GPUs therefore needs a dump at that
size, and publishes it.

To publish every config for a new backend image:

```bash
python -m arctic_platform.inference.semi_persistence.onboarding.publish config=connection.json \
    image_tag=dev_... models=glm53-tp16,qwen38-27b-tp1x8 [max_gpus=40] [reserve_gpus=8]
```

It submits one job per model with the flag, waits for the engine's
`semi_p: published` line in the sub-job log (or a restore from an image that is
published already), saves the log and cancels the job. Results are in
`onboarding_reports/publish/<image_tag>/summary.json`. Then point the models'
entries in `models/backends.json` at the image.

## Model files

`models/<name>.json` holds the sampling sub-job exactly as published, without
`semi_p` (the suite sets it per job), plus optional overrides:

```json
{
  "name": "qwen38-27b-tp1x8",
  "sub_job": {"job_type": "sampling", "model_name": "Qwen/Qwen3.8-27B", "dtype": "bfloat16",
              "seed": 42, "inference_config": {"max_seq_len": 128, "n_gpus": 8,
              "vllm_config": {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}}},
  "copies": 2,
  "thresholds": {"ttft_ratio_max": 1.10, "tpot_ratio_max": 1.10,
                 "latency_ratio_max": 1.15, "throughput_ratio_min": 0.90},
  "perf": {"shapes": [[16, 16], [16, 96], [64, 48]], "concurrency": [1, 8, 32]},
  "correctness": {"greedy_prompts": 8, "max_tokens": 48, "logprob_atol": 0.001}
}
```

Replicas are `n_gpus / (tensor_parallel_size x pipeline_parallel_size)`.

## What is measured

### Correctness (gated jobs must match; stock plain is reported)

Every check sends the same requests to every job and compares each with
`on_hit`; none assumes how the model answers, so a check fails only when a gated
job disagrees. Requests go one prompt at a time unless noted.

| Check | Request | A job matches when |
|---|---|---|
| Greedy, one prompt at a time | 8 different prompts, `temperature 0`, up to 48 tokens | `token_ids`, `text`, `finish_reason`, `prompt_len`, `generation_len` identical for all 8 |
| `ignore_eos` runs to `max_tokens` | one prompt, `ignore_eos`, 32 tokens | identical, and the reference generates exactly 32 tokens |
| Same answer from every replica | one prompt under `max(4, 2 x replicas)` routing keys | the job gives a single answer across all keys, the reference's |
| Sampled logprobs | `logprobs 1`, sampled token only | max abs difference <= `logprob_atol` |
| Top-k logprobs shape | `logprobs 2` | same number of candidates at every position |
| Prompt logprobs | `prompt_logprobs 1` | same number of positions |
| Guided JSON | JSON-schema constrained decoding | identical text, and the reference's parses as JSON or `off_matched` gives the same text (a reasoning model's schema applies only after its reasoning) |
| Stop strings | `stop: [".", "\n\n"]` | identical text and finish reason |
| Seeded sampling | `temperature 1, seed 1234` | identical, and a repeat on the reference is identical too |
| Token ids are real | every semi_p result | non-empty, not all zero, one id per generated token |
| Greedy, batched (info only) | 16 prompts in one request | reported as a match rate; batch shape changes numerics |

### Throughput and latency (thresholds gate the verdict)

Closed-loop threads send single-prompt greedy requests with `ignore_eos`, token-id
prompts varied per request so prefix caching and prefix-affinity routing cannot
favour either job. The grid is concurrency `1, replicas, 4 x replicas` (or
`1, 8, 32` for one replica) by three prompt/output shapes, alternating jobs cell
by cell. Server latency is the request's `updated_at - created_at`; client
latency adds submission and polling.

| Check | Measured as | Default limit |
|---|---|---|
| TTFT ratio | server p50 of 16 `max_tokens 1` requests, short and long prompt; worst of the two | semi_p / plain <= 1.10, or within 12 ms (a restored TP=1 engine sits about 1 or about 11 ms above plain) |
| TPOT ratio | (server p50 at N tokens - TTFT) / (N - 1), N up to 128 | <= 1.10, or within 0.5 ms |
| Worst server p50 latency ratio | across every grid cell | <= 1.15 |
| Worst throughput ratio | generated tokens / cell wall time, across cells with concurrency > 1 | >= 0.90 |
| Request errors and zero-id responses | every perf request | 0 |
| Replica scaling (multi-replica) | tok/s at concurrency = replicas over replicas x tok/s at concurrency 1 | >= 0.90 x the plain job's scaling |

Where `off_matched` runs (TP >= 2, one pod), it runs the same grid too, and the
report splits the gap: **settings** = `off_matched` / `off_default`, the cost of
semi_p's collective setup on a plain engine, and **semi_p** = `on_hit` /
`off_matched`, the cost of semi_p itself. The gates stay on `on_hit` against
`off_default`, but a TTFT, TPOT, latency or throughput check that fails there
passes when semi_p is within the same limit against `off_matched`: that part of
the gap is the settings', which the report states under the verdict.

A semi_p generate returns its result in one piece, so streaming TTFT is not
observable; both jobs are measured the same non-streaming way. The absolute
slack keeps millisecond jitter from failing a ratio at tens of milliseconds, and
concurrency-1 throughput (just 1 / latency) is left to the latency gate because
it is the most exposed to client-side jitter. The report also gives each job's
client overhead (client p50 minus server p50) for information.

Whole-node configs are tested at `copies=2` (twice the published `n_gpus`, so
two pods restore the same image); single-GPU and multi-node configs at their
published size. Override with `copies=` or a model file's `"copies"`.

### Start-up (reported; a cold start fails the run)

- Job timeline for both jobs: submit to `running`, and `initializing` to `running`.
- semi_p, per replica, from the job log: whether it restored from the published
  image (`materialized`, `reading weights in place`) or cold-started (`cache
  miss`), the image copy time, restore time and key-to-ready time.
- The restore's primitive breakdown per replica (`criu_restore`, `cuda_restore`,
  `reinit_nccl`, `load_weights`, `wake_up_weights`, `repin`, `restore_weights`,
  `wake_up_kv_cache`, `rebind_graphs` / `recapture_graphs`, ...), when the GPU
  pods' own logs are supplied with `log_files_on`: the sub-job log carries the
  markers but not the per-step lines. Each pod's file is parsed separately and
  its replicas are labelled by pod; a pod that holds only a later node-partition
  of a multi-node engine shows its own steps as a "node-partition agent" row.
- Plain vLLM cold start: weight load, model loading, torch.compile, profiling,
  graph capture and the engine-init total (slowest replica).
- Generate requests served per semi_p replica, when the log shows them.

If neither the live log nor the archived execution logs carry the semi_p
markers, the verdict is `WARN` (restore not confirmed); pass saved logs with
`log_files_on` / `log_files_off`.

## Reports and verdict

`<out_dir>/<name>-<UTC time>.md` and `.json` per model, plus `summary-<UTC time>.md`
for a multi-model run. Every log the run read is saved in
`<out_dir>/<name>-<UTC time>-logs/` (`semi_p-*` and `plain-*`), so a report can be
rebuilt offline after the jobs are gone, for example once GPU pod logs are at hand:

```bash
python -m arctic_platform.inference.semi_persistence.onboarding.rereport report=onboarding_reports/NAME-TIME.json \
    log_files_on=POD_LOG_DIR
```

With several reports, `POD_LOG_DIR` may hold one subdirectory per semi_p job,
named by the first eight characters of its job id.

### The onboarding book

`<out_dir>/ONBOARDING.md` collects every onboarded model config into one set of
tables, keeping the latest run per model config, backend image and test copies,
so it grows as models are onboarded and a re-run replaces its earlier row:
overview and verdicts, start-up (image copy, restore, key to ready, plain cold
start, speedup), restore primitives, plain vLLM cold-start phases, latency (TTFT,
TPOT, client overhead), throughput (tok/s per concurrency, worst ratios, replica
scaling) and a correctness matrix. `onboard` and `rereport` regenerate it; to
rebuild it by hand:

```bash
python -m arctic_platform.inference.semi_persistence.onboarding.book reports=onboarding_reports [out=ONBOARDING.md]
```

It contains no job ids, so it can be committed alongside the model files.

The verdict is:

- **FAIL**: a required correctness check failed, a perf threshold was missed,
  the semi_p job cold-started, a replica did not restore, or a job failed.
- **WARN**: everything passed but the restore could not be confirmed from logs.
- **PASS**: otherwise.

The command exits non-zero if any model failed.

## Troubleshooting

- **"cache miss"**: no published image matches this config and backend image,
  or the nodes have not finished syncing it. Publish it, wait, and re-run.
- **Only `off_default` differs**: expected at TP >= 2; stock vLLM runs other
  collectives. The report shows where it first differs.
- **`off_matched` is reported, not gated**: the backend predates
  `SEMIP_MATCH_COLLECTIVES`, so only the environment and `pass_config` were
  matched. Use a backend image that supports it.
- **`on_miss` "did not cold-start"**: it found a local image on its node; pass
  `miss_image_cache` so it uses a fresh root.
- **`403 Forbidden`**: the connection's PAT expired or the network cannot reach
  the account; refresh the PAT or reconnect.
- **`max_gpus cannot fit`**: a model's jobs need their GPUs at once; raise
  `max_gpus`, run that model alone, or drop variants.
