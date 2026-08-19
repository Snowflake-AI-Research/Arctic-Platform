# verl × Arctic-Platform × Cortex-training

Companion to [`run_gsm8k_grpo_arl.sh`](run_gsm8k_grpo_arl.sh): same verl
`RemoteBackend` adapter and same GRPO recipe on Qwen3-0.6B / GSM8K.
`ARCTIC_BACKEND=cortex` (read in
[`adapter.py::_create_rl_client_config`](../adapter.py)) swaps the default
`OnPremConfig` for `CortexConfig.from_env()`. The verl YAML stays backend-agnostic.

## Install + env

```bash
pip install arctic-platform[cortex]
export ARCTIC_BACKEND=cortex
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export CORTEX_PAT=<pat>
```

## Run

```bash
./run_gsm8k_grpo_cortex.sh
```

## Adapter changes for the Cortex path

Two shape mismatches versus the on-prem wire format live in
`arctic_platform/integrations/verl/adapter.py`:

* `_send_compute_{ref_,}log_prob`: return zero-shaped `log_probs` / `entropy`;
  Cortex has no `/forward` sub-job. Only correct for single-epoch on-policy
  GRPO without KL — `use_kl_loss`, `use_kl_in_reward`, or `ppo_epochs > 1`
  raise `NotImplementedError`.
* `_send_update_actor`: reshape `{batch, meta, processing}` →
  `{args, kwargs, context, processing}` via
  [`to_cortex_fwd_bwd_payload`](../../_cortex_shared.py) (shared with the
  SkyRL shim).

Everything else (`generate`, `sync_weights`, `save_checkpoint`, wake/sleep)
goes through `ArcticRLClient` unchanged.
