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

Both of the Cortex shape differences now live in the Cortex transport rather
than in this adapter, so they apply to any caller:

* `forward` (`fwd_no_grad`) returns zero-shaped `log_probs` / `entropy`, because
  Cortex has no `/forward` sub-job. Correct only for single-epoch on-policy GRPO
  without KL, which is why `_reject_cortex_incompatible_knobs` refuses
  `use_kl_loss`, `use_kl_in_reward`, `ppo_epochs > 1`, a non-GRPO advantage
  estimator, a custom `policy_loss_fn` or multi-turn rollout before the client
  is ever built.
* `forward-backward` is lowered from verl's `{batch, meta, processing}` to
  Cortex's `{args, kwargs, context, processing}` by
  [`lower_fwd_bwd_batch`](../../../client/cortex_batch.py).

Cortex also reports `avg_loss` and `last_lr` as top-level fields where on-prem
puts them under `metrics`; the transport mirrors them in, so `update_actor`
reads responses from either backend unchanged.

Everything else (`generate`, `sync_weights`, `save_checkpoint`, wake/sleep) goes
through `AsyncArcticRLClient` unchanged.
