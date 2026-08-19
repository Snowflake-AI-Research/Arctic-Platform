# Cortex-training as an Arctic-Platform backend

Run SkyRL or verl RL recipes against Cortex-training GPUs without changing
either framework. The driver is CPU-only; every GPU op (training
fwd/bwd/step, sampling generate, weight-sync) is dispatched to Cortex
sub-jobs over SnowAPI.

## Supported training regime

Cortex sub-jobs expose training + sampling + weight-sync but no `/forward`
sub-job, no disk-shared weight reload, and no colocation lifecycle. That
constrains what recipes work correctly on this backend:

* **Single-epoch on-policy GRPO only.** `π_old ≡ π_new` is restored by the
  server-side GRPO loss defaulting `old_log_probs = logprobs.detach()` when
  `context.old_log_probs_shifted` is absent. Multi-epoch PPO
  (`ppo_epochs > 1`) needs the rollout-time snapshot and will raise from the
  verl adapter.
* **No KL-to-reference.** `use_kl_loss=True` or `use_kl_in_reward=True`
  needs real ref log-probs; the adapter raises. Disable both, or run on the
  on-prem backend for KL-anchored recipes.
* **Weight sync via NCCL only.** `save_weights` (SkyRL's disk-based reload)
  raises `NotImplementedError`; use `sync_weights()` or `save_checkpoint()`.
* **`wake_*` / `sleep_*` are no-ops.** Cortex sub-jobs are always awake;
  the transport short-circuits these ops.

## What's provided

* **Unified client** —
  [`arctic_platform.client.ArcticRLClient`](../arctic_platform/client/client.py)
  routes to `CortexTransport` whenever `backend` is a `CortexConfig`.
  `CortexConfig.from_env()` hydrates from `ARCTIC_CORTEX_*` env vars —
  explicit constructor args always win.
* **Legacy shim** —
  [`arctic_platform.rl._cortex_dispatch`](../arctic_platform/rl/_cortex_dispatch.py)
  wraps the unified client behind the legacy `arctic_platform.rl` surface
  so SkyRL, which still builds `arctic_platform.rl.ArcticRLClientConfig`,
  routes to Cortex. That legacy config has a `_backend_from_env` validator
  that flips `backend="local"` → `"cortex"` when `ARCTIC_BACKEND=cortex`.
* **verl adapter** —
  [`arctic_platform.integrations.verl.adapter`](../arctic_platform/integrations/verl/adapter.py)
  reads `ARCTIC_BACKEND=cortex` at one call-site and swaps its default
  `OnPremConfig` for `CortexConfig.from_env()`. verl's YAML has no backend
  discriminator field, so this env knob is the only way to route without
  patching verl itself.
* **SkyRL driver shim** —
  [`arctic_platform.integrations.skyrl`](../arctic_platform/integrations/skyrl/__init__.py)
  provides `install_cortex_driver_shims()` to patch out SkyRL's
  `peer_access_supported` probe (hangs a CPU-only driver). Invoked
  automatically by `python -m arctic_platform.integrations.skyrl`.
* **Recipes** —
  [`recipes/rl/skyrl/simple_gsm8k_cortex/`](../recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
  (SkyRL GRPO on GSM8K) and
  [`arctic_platform/integrations/verl/examples/README-cortex.md`](../arctic_platform/integrations/verl/examples/README-cortex.md)
  (verl GRPO on GSM8K).

## Env vars

| Env var | Default | Description |
|---|---|---|
| `ARCTIC_BACKEND` | *(unset)* | Set to `cortex` to route through Cortex. Read by the legacy `ArcticRLClientConfig` validator (SkyRL path) and by the verl adapter's `_create_rl_client_config` (verl path). |
| `ARCTIC_CORTEX_HOST` | *(required for PAT auth)* | Snowflake host, e.g. `<account>.<region>.snowflakecomputing.com`. |
| `ARCTIC_CORTEX_BASE_URL` | *(optional)* | Direct/mock GS URL for dev; bypasses PAT auth (mutually exclusive with `_HOST`). |
| `CORTEX_PAT` | *(required for PAT auth)* | Snowflake Programmatic Access Token. Env var name overridable via `ARCTIC_CORTEX_PAT_ENV_VAR`. |
| `ARCTIC_CORTEX_DATABASE` | *(required for PAT auth)* | Snowflake database. |
| `ARCTIC_CORTEX_SCHEMA` | *(required for PAT auth)* | Snowflake schema. |
| `ARCTIC_CORTEX_ENDPOINT` | `cortex-training` | SnowAPI endpoint name. |
