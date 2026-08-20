# Cortex-training as an Arctic-Platform backend

Run SkyRL or verl RL recipes against Cortex-training GPUs without changing
either framework. The driver is CPU-only; every GPU op (training
fwd/bwd/step, sampling generate, weight-sync) is dispatched to Cortex
sub-jobs over SnowAPI.

## Install

```bash
pip install "arctic-platform[cortex]"
```

The `[cortex]` extra is the client-only install from
[#75](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/75): it
skips DeepSpeed, `transformers`, Ray and vLLM. Enough to drive both the
SkyRL shim and the verl adapter against Cortex — the verl user still
supplies verl at their pinned version.

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

### Supported recipes (verl adapter)

The verl adapter (`arctic_platform.integrations.verl.adapter`) runs a
compat validator at init time and refuses to construct a client if any of
the knobs below are set. This is deliberate — every one of them would
silently train on zero-filled tensors on Cortex, producing wrong gradients
with no runtime error.

| Knob | Cortex support | Reason |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` only | GAE / RLOO / REMAX / REINFORCE++ consume real `old_log_probs` in the loss |
| `actor_rollout_ref.actor.use_kl_loss` | must be `False` | needs `/forward` for ref log-probs |
| `algorithm.use_kl_in_reward` | must be `False` | same |
| `algorithm.kl_penalty` | `none` / unset | typed KL reads `ref_log_probs` from the batch |
| `algorithm.kl_ctrl.kl_coef` (with KL on) | must be `0` | adaptive KL reads `ref_log_probs` |
| `actor_rollout_ref.actor.ppo_epochs` | `1` | off-policy PPO needs the rollout-time snapshot |
| `actor_rollout_ref.actor.policy_loss_fn` | must be unset | Cortex server has no custom-loss hook |
| `actor_rollout_ref.rollout.multi_turn.enable` | must be `False` | multi-turn recomputes log-probs per turn |

For any of these, use the on-prem backend (`ARCTIC_BACKEND=local` or
unset). The validator raises `NotImplementedError` at
`ArcticRLClientWrapper.__init__`, so misconfigured recipes fail at launch,
not after N minutes of bogus optimizer steps.

## What's provided

* **Unified client** —
  [`arctic_platform.client.ArcticRLClient`](../arctic_platform/client/client.py)
  routes to `CortexTransport` whenever `backend` is a `CortexConfig`.
  `CortexConfig.from_env()` hydrates from `ARCTIC_CORTEX_*` env vars —
  explicit constructor args always win.
* **Legacy shim** —
  [`arctic_platform.integrations._cortex_dispatch`](../arctic_platform/integrations/_cortex_dispatch.py)
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
