# Simple — GRPO on GSM8K, Cortex backend

Same recipe as [`simple_gsm8k/`](../simple_gsm8k/), with training + sampling
dispatched to Cortex-training sub-jobs instead of a local Ray + vLLM stack.
The SkyRL driver runs on a CPU-only laptop / VM; Cortex owns the GPUs.

"Same recipe" is meant literally: every hyperparameter is that recipe's value
unchanged, down to `policy_mini_batch_size` and the one-epoch schedule, so the
two differ only in where the GPUs are. The single exception is
`micro_train_batch_size_per_gpu`, which has to differ — SkyRL defaults it to 1,
which cannot satisfy DeepSpeed's `micro × accum × n_gpus == train_batch` across
4 training GPUs.

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B` |
| Reward | SkyRL built-in `gsm8k` env (exact match on `#### <number>`) |
| Trainer | Cortex training sub-job (server-side GRPO loss) |
| Sampling | Cortex sampling sub-job (vLLM inside Cortex) |
| Sequence lengths | prompt 512, response 1024 |
| Batch | 32 prompts × 4 samples, mini-batch 4 — the sibling's values (§6) |
| Schedule | 1 epoch = 233 steps, eval every 10 |
| GPUs | 4 training + 4 sampling, allocated by Cortex; driver is CPU-only |

## 1. Install

`pip install arctic-platform[cortex]` on the driver — no local GPU deps.
SkyRL still needs to be cloned at the pinned commit (see
[`../README.md`](../README.md)) with `SKYRL_HOME` exported. The Arctic RL
× SkyRL integration code lives under `$SKYRL_HOME/integrations/arctic_rl/`.

## 2. Set Cortex env

```bash
export ARCTIC_BACKEND=cortex
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export CORTEX_PAT=<your PAT>
```

`ArcticRLClientConfig`'s `_backend_from_env` validator promotes SkyRL's
baked-in `backend="local"` to `"cortex"` when `ARCTIC_BACKEND=cortex`;
`CortexConfig.from_env()` reads the rest.

## 3. Prepare the dataset

```bash
python ../simple_gsm8k/download_data.py --output_dir ${HOME}/data/gsm8k-skyrl
```

The parquet must have SkyRL's schema (`reward_spec`, `env_class`), **not**
verl's (`reward_model`). The launcher does a pre-flight check and refuses
to start on a verl-shaped parquet — otherwise `GSM8kEnv` reads `reward_spec`
under a name that isn't there and silently scores every rollout 0.0.

The default `DATA_DIR` is `~/data/gsm8k-skyrl` (deliberately distinct from
verl's `~/data/gsm8k`, which uses a different schema). Override with
`DATA_DIR=...` if your parquets live elsewhere.

## 4. Run

```bash
./run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

The launcher uses `python -m arctic_platform.integrations.skyrl` instead
of `python -m skyrl.train.entrypoints.main_base` so the driver-side
`peer_access_supported` shim is installed before SkyRL's Ray probe runs.

### Stopping it, and getting the GPUs back

Ctrl-C, or `kill` on the launcher, is enough. Both stop the run and release
the Cortex job, and the launcher prints which job it released.

Worth knowing why that needs saying. Nothing releases the GPUs on its own:
SkyRL never calls the client's shutdown when training ends, so reaching the
last step holds all 8 GPUs exactly as firmly as being killed does. The driver
also ignores SIGINT and SIGTERM outright — Ray installs handlers — so a
graceful stop cannot be asked for, and the launcher escalates to SIGKILL on
the driver's whole process tree. It kills the descendants first, because the
dataloader workers survive a bare kill of their parent, reparent to init, and
hold ~0.5 GB each until something unrelated OOMs.

`kill -9` on the launcher itself is the one case it cannot cover, since no
trap runs. That is what [`cortex_jobs.py`](cortex_jobs.py) is for:

```bash
python cortex_jobs.py            # what is holding GPUs
python cortex_jobs.py --cancel   # release it
```

## 5. What a healthy run looks like

The whole schedule, start to finish: one epoch of GSM8K at the shipped
defaults, all 233 steps, no overrides and no early stop. A bare
`./run_qwen3_0.6b_gsm8k_grpo_cortex.sh` is what produced it. Held-out
`eval/all/pass_at_1` over the full 1319-example GSM8K test set:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass@1 | 0.2949 | 0.3912 | 0.4989 | 0.5891 | 0.6588 | 0.6876 | 0.6907 | 0.7096 | 0.7165 | 0.7316 | 0.7172 | 0.7316 |

| step | 120 | 130 | 140 | 150 | 160 | 170 | 180 | 190 | 200 | 210 | 220 | 230 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pass@1 | 0.7225 | 0.7400 | 0.7384 | 0.7422 | 0.7233 | 0.7369 | 0.7415 | 0.7346 | 0.7460 | 0.7528 | 0.7566 | 0.7582 |

Step 0 is the untrained baseline: **0.2949 → 0.7582**, 2.6x, in 2h03m
wall-clock (~4 min of Cortex provisioning, then ~20s per training step and
~2 min per eval). Zero errors, zero retries.

The shape is the point. It climbs steeply to ~0.69 by step 50, then flattens:
everything from step 90 on sits in 0.72–0.76, drifting up by about a point
over the last 100 steps. That is a run that has converged for this
configuration, not one that was stopped while still climbing — which is what
the earlier 50-step number in this file was.

Because the epoch is the whole schedule, the run ends on its own and there is
nothing to decide about when to stop it.

**Two metrics are worth watching, because the failure mode here is a run that
looks fine and learns nothing:**

* `entropy` should sit near **0.5**. It is the model's negative log-probability
  of its own sampled tokens, so ~0.5 means it rates its own output at roughly
  60% probability. A value near 3.0, especially one that climbs, means the loss
  is being applied to the wrong token positions — the shapes still match and
  `grad_norm` is still nonzero, so nothing errors.
* `pass_at_1` should clear the baseline within ~20 steps. Flat-at-baseline
  across tens of steps while reward looks plausible is the signature of the
  same problem.

`approx_kl` reads exactly **0.0** and that is expected, not a bug — see the
clipping note below.

## 6. Scaling up, and the ceiling we hit

The defaults are the sibling recipe's, which are smaller than SkyRL's published
multi-GPU GSM8K scale (`train_batch_size=1024`,
`policy_mini_batch_size=256`). Keeping them is not only about matching the
sibling: **that larger scale does not currently work on this backend.**
Measured twice:

```bash
TRAIN_BSZ=1024 MINI_BSZ=256 MICRO_BSZ=64 ./run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

Step 1 completes normally (228s, 4096 sequences, packer and loss all fine).
Step 2 then dies fetching its result:

```
WireError: not a valid DSSST1 safetensors payload: invalid DSSST1 safetensors header length
  preceded by: ClientConnectionError('Connection lost: SSL shutdown timed out')
```

Both attempts failed at exactly step 2 with that signature. Step 1's response
is the same ~108 MiB, so this is not the size of a single transfer but the
second large transfer on a reused connection: the stream drops, the chunk set
is left incomplete, and the decoder is handed a truncated payload. Tracked as
[#99](https://github.com/Snowflake-AI-Research/Arctic-Platform/issues/99) — it
is in the shared transport's chunk handling, not in this recipe.

Two consequences for scaling up:

* `N_SAMPLES` cannot reach the canonical 5 regardless. 1024 × 5 sequences of
  1536 tokens is ~135 MiB of per-token logprobs and entropies against a 128 MiB
  response cap, so the launcher refuses it up front. At 4 it is at 84%.
* Intermediate scales between 32 and 1024 are untested. If you raise
  `TRAIN_BSZ`, expect the failure above once responses get large, and get a
  clean run past step 2 — where both canonical attempts died — before trusting
  a new size.

## Troubleshooting

| what you see | what it means |
| --- | --- |
| `429 ... per-account GPU cap reached: 8 GPUs in use` | A previous run's Cortex job still holds GPUs. This recipe needs all 8, so nothing else can be running. The launcher releases what it started on any exit including Ctrl-C, so this means either a `kill -9` (no trap runs) or a teammate on the same account. `python cortex_jobs.py` shows what is holding them, `--cancel` releases. Releasing is not instant — a launch within ~a minute of a cancel can still hit the cap. |
| `429 ... gRPC message exceeds maximum size 134217728` | The step's response exceeded Cortex's 128 MiB cap. The launcher preflights this, so you should only reach it by overriding `TRAIN_BSZ`, `N_SAMPLES` or `RESPONSE_LEN` past the printed ceiling. |
| `packing requires left-aligned rows` | The batch reached Cortex with padding at the head of a row. The shim left-aligns before sending, so this indicates a payload path that bypassed `to_cortex_fwd_bwd_payload`. |
| `WireError: ... invalid DSSST1 safetensors header length`, usually after `Connection lost: SSL shutdown timed out` | A large result came back truncated. Seen reproducibly from step 2 onward at `TRAIN_BSZ=1024` (~108 MiB per response); see section 6. Reduce `TRAIN_BSZ` / `RESPONSE_LEN`. |
| `num_engines should be equal to the number of remote_urls` | `NUM_ENGINES` was changed without the matching placeholder URL list. The launcher derives them together; setting `generator.inference_engine.*` by hand breaks that. |
| `cortex: set base_url (direct URL) or host (PAT auth)` | The `ARCTIC_CORTEX_*` environment isn't set in this shell. See step 2. |

## Notes

* Only single-epoch on-policy GRPO is supported (`use_kl_loss=false`,
  `use_kl_in_reward=false`, `ppo_epochs=1`); see
  [`docs/cortex-integration.md`](../../../../docs/cortex-integration.md)
  for why. Cortex has no `/forward` sub-job, so there are no reference-model
  log-probs to build a KL term from. Asking for them raises rather than
  silently substituting zeros, which would make the KL a function of the
  current policy alone.
* **`approx_kl` reads exactly 0.0, and clipping is inert by construction.**
  Cortex re-derives π_old from the live forward rather than taking a
  rollout-time snapshot. On this path that is exact rather than approximate:
  SkyRL's Arctic trainer issues one `fwd_bwd` and one optimizer step per
  collected batch, and `MINI_BSZ` only sets the server's gradient-accumulation
  chunk, so the policy cannot move within a step and π_old ≡ π_new holds. The
  ratio is 1, `eps_clip` never binds, and this is single-update on-policy GRPO.
  The consequence to know is that a recipe relying on clipping to take several
  updates per batch cannot be reproduced here.
* Every batch knob is env-overridable (`TRAIN_BSZ`, `MINI_BSZ`, `N_SAMPLES`,
  `MICRO_BSZ`, `PROMPT_LEN`, `RESPONSE_LEN`, `LR`, `TOTAL_EPOCHS`,
  `EVAL_INTERVAL`), but the defaults are the sibling's and the curve in section
  5 is what they produce; section 6 covers what happens when you raise them.
* Two runs cannot share an account: 4 training + 4 sampling is exactly the
  8-GPU cap. The launcher releases its own job on the way out, so back-to-back
  runs work; leave a moment between them, as the release is not instant.
* Weight sync is NCCL between Cortex sub-jobs. The driver never touches
  weights.
