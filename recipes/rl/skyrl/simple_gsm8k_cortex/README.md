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

## 3. Run

```bash
python download_data.py
./run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

The launcher uses `python -m arctic_platform.integrations.skyrl` instead
of `python -m skyrl.train.entrypoints.main_base` so the driver-side
`peer_access_supported` shim is installed before SkyRL's Ray probe runs.

## Notes

* Only single-epoch on-policy GRPO is supported (`use_kl_loss=false`,
  `use_kl_in_reward=false`, `ppo_epochs=1`); see
  [`docs/cortex-integration.md`](../../../../docs/cortex-integration.md)
  for why.
* Weight sync is NCCL between Cortex sub-jobs. The driver never touches
  weights.
