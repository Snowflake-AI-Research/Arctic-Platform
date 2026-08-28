# Simple — GRPO on GSM8K, Cortex backend

Same recipe as [`simple_gsm8k/`](../simple_gsm8k/), with training + sampling
dispatched to Cortex-training sub-jobs instead of a local Ray + vLLM stack.
The SkyRL driver runs on a CPU-only laptop / VM; Cortex owns the GPUs.

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B` |
| Reward | SkyRL built-in `gsm8k` env (exact match on `#### <number>`) |
| Trainer | Cortex training sub-job (server-side GRPO loss) |
| Sampling | Cortex sampling sub-job (vLLM inside Cortex) |
| Sequence lengths | prompt 512, response 1024 |
| GPUs | Cortex allocates; driver is CPU-only |

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

## 5. What a healthy run looks like

Qwen3-0.6B on GSM8K at 4 training + 4 sampling GPUs, `TRAIN_BSZ=32
MINI_BSZ=4 N_SAMPLES=4`. Held-out `eval/all/pass_at_1` on the full 1319-example
test set, against an untrained baseline of ~0.29–0.31:

| step | 0 | 10 | 20 | 30 | 40 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| pass@1 | 0.3055 | 0.3146 | 0.3632 | 0.3897 | 0.4003 | 0.4321 |

About 20s per training step plus ~2min per evaluation, so roughly 35 minutes to
step 50. Provisioning the Cortex sub-jobs takes ~4 minutes before step 0.

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

## Troubleshooting

| what you see | what it means |
| --- | --- |
| `429 ... per-account GPU cap reached: 8 GPUs in use` | A previous run's Cortex job still holds GPUs. This recipe needs all 8, so nothing else can be running. Run `python cortex_jobs.py` to see what is holding them and `--cancel` to release. A driver that exits cleanly releases its own; one that was SIGKILLed does not. |
| `429 ... gRPC message exceeds maximum size 134217728` | The step's response exceeded Cortex's 128 MiB cap. The launcher preflights this, so you should only reach it by overriding `TRAIN_BSZ`, `N_SAMPLES` or `RESPONSE_LEN` past the printed ceiling. |
| `packing requires left-aligned rows` | The batch reached Cortex with padding at the head of a row. The shim left-aligns before sending, so this indicates a payload path that bypassed `to_cortex_fwd_bwd_payload`. |
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
* **PPO clipping does not engage on this path.** π_old is re-derived from the
  live forward on every `fwd_bwd` rather than snapshotted, so the importance
  ratio is exactly 1 and `eps_clip` never binds — which is why `approx_kl`
  reads 0.0. With `MINI_BSZ < TRAIN_BSZ` the recipe is therefore running
  unclipped updates, not clipped GRPO. It trains, but it is not the same
  objective, and that matters if you are comparing against published numbers.
* `N_SAMPLES` defaults to 4 rather than SkyRL's canonical 5, because 1024 × 5
  overruns the 128 MiB response cap by 4.2%. This and `use_kl_loss=false` are
  the only two departures from the canonical recipe.
* Two runs cannot share an account: 4 training + 4 sampling is exactly the
  8-GPU cap.
* Weight sync is NCCL between Cortex sub-jobs. The driver never touches
  weights.
