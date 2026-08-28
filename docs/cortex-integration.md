# Cortex-training as an Arctic-Platform backend

Run SkyRL RL recipes against Cortex-training GPUs without changing SkyRL. The
driver is CPU-only; every GPU op (training fwd/bwd/step, sampling generate,
weight-sync) is dispatched to Cortex sub-jobs over SnowAPI.

## Install

```bash
pip install "arctic-platform[cortex]"
```

The `[cortex]` extra is the client-only install from
[#75](https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/75): it
skips DeepSpeed, `transformers`, Ray and vLLM.

## Supported training regime

Cortex sub-jobs expose training + sampling + weight-sync but no `/forward`
sub-job, no disk-shared weight reload, and no colocation lifecycle. That
constrains what runs correctly here:

* **Single-epoch on-policy GRPO only.** With `context.old_log_probs_shifted`
  absent, the server-side GRPO loss defaults `old_log_probs =
  logprobs.detach()`, restoring π_old ≡ π_new. Multi-epoch PPO needs the
  rollout-time snapshot, which this backend cannot provide.
* **No KL-to-reference.** There are no reference log-probs to build the term
  from, so `use_kl_loss` / `use_kl_in_reward` must be off. Asking for them
  raises from `_CortexClientShim.fwd_no_grad` rather than substituting zeros,
  which would reduce the KL to a function of the current policy alone.
* **PPO clipping does not engage.** Because π_old is re-derived per `fwd_bwd`
  rather than snapshotted, the importance ratio is exactly 1 and `eps_clip`
  never binds — so `approx_kl` reads 0.0, and a step split into minibatches is
  running unclipped updates. It trains, but it is not clipped GRPO.
* **Weight sync via NCCL only.** `save_weights` (SkyRL's disk-based reload)
  raises; use `sync_weights()` or `save_checkpoint()`.
* **`wake_*` / `sleep_*` are no-ops.** Cortex sub-jobs are always awake; the
  transport short-circuits these ops.

## Operational limits

* **8 GPUs per account.** The GSM8K recipe asks for 4 training + 4 sampling,
  which is exactly the cap, so two runs cannot coexist.
* **A response is capped at 128 MiB.** `post=["compute_logprobs"]` returns a
  logprob and an entropy per token, ~18 B/token, so a step is limited to
  roughly 4900 sequences. The recipe preflights this.
* **A driver that dies without exiting cleanly keeps its GPUs.** `shutdown()`
  cancels the job; SIGKILL skips it, and the next launch then fails the GPU
  cap. `recipes/rl/skyrl/simple_gsm8k_cortex/cortex_jobs.py` lists and
  releases them.

## What's provided

* **Unified client** —
  [`arctic_platform.client.ArcticRLClient`](../arctic_platform/client/rl.py)
  routes to `CortexTransport` whenever `backend` is a `CortexConfig`.
  `CortexConfig.from_env()` hydrates from `ARCTIC_CORTEX_*` env vars —
  explicit constructor args always win.
* **Legacy shim** —
  [`arctic_platform.integrations._cortex_dispatch`](../arctic_platform/integrations/_cortex_dispatch.py)
  wraps the unified client behind the legacy `arctic_platform.rl` surface so
  SkyRL, which still builds `arctic_platform.rl.ArcticRLClientConfig`, routes
  to Cortex. That legacy config has a `_backend_from_env` validator that flips
  `backend="local"` → `"cortex"` when `ARCTIC_BACKEND=cortex`.
* **Payload lowering** —
  [`arctic_platform.integrations._cortex_shared`](../arctic_platform/integrations/_cortex_shared.py)
  reshapes SkyRL's batch into Cortex's wire format and left-aligns each row,
  which the server's microbatch packer requires.
* **SkyRL driver shim** —
  [`arctic_platform.integrations.skyrl`](../arctic_platform/integrations/skyrl/__init__.py)
  provides `install_cortex_driver_shims()` to patch out SkyRL's
  `peer_access_supported` probe (hangs a CPU-only driver). Invoked
  automatically by `python -m arctic_platform.integrations.skyrl`.
* **Recipe** —
  [`recipes/rl/skyrl/simple_gsm8k_cortex/`](../recipes/rl/skyrl/simple_gsm8k_cortex/README.md),
  SkyRL GRPO on GSM8K, with a measured reference curve.

## Env vars

| Env var | Default | Description |
|---|---|---|
| `ARCTIC_BACKEND` | *(unset)* | Set to `cortex` to route through Cortex. Read by the legacy `ArcticRLClientConfig` validator. |
| `ARCTIC_CORTEX_HOST` | *(required for PAT auth)* | Snowflake host, e.g. `<account>.<region>.snowflakecomputing.com`. |
| `ARCTIC_CORTEX_BASE_URL` | *(optional)* | Direct/mock GS URL for dev; bypasses PAT auth (mutually exclusive with `_HOST`). |
| `CORTEX_PAT` | *(required for PAT auth)* | Snowflake Programmatic Access Token. Env var name overridable via `ARCTIC_CORTEX_PAT_ENV_VAR`. |
| `ARCTIC_CORTEX_DATABASE` | *(required for PAT auth)* | Snowflake database. |
| `ARCTIC_CORTEX_SCHEMA` | *(required for PAT auth)* | Snowflake schema. |
| `ARCTIC_CORTEX_ENDPOINT` | `cortex-training` | SnowAPI endpoint name. |
