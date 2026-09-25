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
* **Clipping is inert, and `approx_kl` reads 0.0.** π_old is re-derived per
  `fwd_bwd` rather than snapshotted. With one optimizer step per collected
  batch — which is what SkyRL's Arctic trainer does, `policy_mini_batch_size`
  being only the gradient-accumulation chunk — the policy cannot move within a
  step, so π_old ≡ π_new holds exactly and the ratio is 1. That makes this
  single-update on-policy GRPO rather than a broken approximation, but a recipe
  that needs clipping to take several updates per batch cannot run here.
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
* **Large responses are not reliably delivered well below that cap.** At
  ~108 MiB per response the first transfer succeeds and the second fails with
  `Connection lost: SSL shutdown timed out`, surfacing as a `WireError` about a
  truncated DSSST1 payload: the chunk set is left incomplete and decoded
  anyway. Reproduced twice at `TRAIN_BSZ=1024`, which is why the GSM8K recipe
  ships a smaller default. Tracked as
  [#99](https://github.com/Snowflake-AI-Research/Arctic-Platform/issues/99).
* **A job holds its GPUs until something explicitly cancels it, and nothing
  does.** `shutdown()` cancels, but SkyRL never calls it when training ends, so
  a run that completes its final step keeps all 8 GPUs just as a killed one
  does; the next launch then fails the per-account cap. The driver additionally
  ignores SIGINT and SIGTERM, Ray having installed handlers, so there is no
  graceful stop to trigger it either. Callers must cancel explicitly — the
  GSM8K recipe's launcher does this on every exit path, and
  `CortexTransport.list_jobs()` / `cancel_job()` are there for anything it
  cannot cover, such as a `kill -9` of the launcher itself.

## What's provided

* **Unified client** —
  [`arctic_platform.client.ArcticRLClient`](../arctic_platform/client/rl.py)
  routes to `CortexTransport` whenever `backend` is a `CortexConfig`.
  `CortexConfig` is a `pydantic-settings` model, so it hydrates from
  `ARCTIC_CORTEX_*` env vars — explicit constructor and YAML values always win.
* **SkyRL entrypoint** —
  [`arctic_platform.integrations.skyrl.entrypoint`](../arctic_platform/integrations/skyrl/entrypoint.py),
  named by the recipe as `trainer.override_entrypoint`. SkyRL's own Arctic
  entrypoint is pinned to the legacy `arctic_platform.rl` client factory, so
  this module swaps that one name for a Cortex client built on
  `arctic_platform.client` and leaves the legacy package untouched. Choosing
  the entrypoint is what chooses Cortex; no environment variable decides it.
* **Client shim** —
  [`arctic_platform.integrations._cortex_dispatch`](../arctic_platform/integrations/_cortex_dispatch.py)
  presents the client surface SkyRL expects over `AsyncArcticRLClient`.
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

Every field of `CortexConfig` reads from `ARCTIC_CORTEX_<FIELD>` via
`pydantic-settings`; the table is that mapping, not a separate contract. No
variable selects the backend — the caller does, and for SkyRL that is which
entrypoint the recipe names.

| Env var | Default | Description |
|---|---|---|
| `ARCTIC_CORTEX_HOST` | *(required for PAT auth)* | Snowflake host, e.g. `<account>.<region>.snowflakecomputing.com`. |
| `ARCTIC_CORTEX_BASE_URL` | *(optional)* | Direct/mock GS URL for dev; bypasses PAT auth (mutually exclusive with `_HOST`). |
| `ARCTIC_CORTEX_PAT` | *(required for PAT auth)* | Snowflake Programmatic Access Token. |
| `ARCTIC_CORTEX_DATABASE` | *(required for PAT auth)* | Snowflake database. |
| `ARCTIC_CORTEX_SCHEMA` | *(required for PAT auth)* | Snowflake schema. |
| `ARCTIC_CORTEX_ENDPOINT` | `cortex-training` | SnowAPI endpoint name. |
