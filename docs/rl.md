# Arctic Platform RL

Reinforcement-learning backend: a thin client drives three GPU engines on a
remote (or colocated) Arctic server. The RL framework keeps the training loop,
rewards, and advantage estimation; Arctic owns the heavy compute.

```
┌─────────────────────────────────────────────────────────────┐
│  RL framework (verl / SkyRL / custom loop)                  │
│  rollouts → rewards → advantages → ArcticRL client          │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP or Ray
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Arctic Platform server  (see common.md)                    │
│  • Training   — DeepSpeed + optimizer (GRPO / custom loss)  │
│  • Sampling   — vLLM + ArcticInference                      │
│  • Log-prob   — DeepSpeed forward-only or vLLM              │
└─────────────────────────────────────────────────────────────┘
```

Shared server details (CLI, endpoints, metrics, colocation):
[`common.md`](common.md). Training-only SFT (no sampling) is planned in a
forthcoming SFT PR.

## vs planned SFT

| | SFT (planned) | RL |
|---|---|---|
| Jobs | training only | training + sampling (+ optional log_prob) |
| Client | forthcoming `arctic_platform.sft` | `arctic_platform.rl` factory (async HTTP/Ray) |
| Loss path | planned `sft` / `sft_ce` | `run_pipeline` (e.g. GRPO) |
| Extra ops | — | `generate`, `log_probs`, `sync_weights`, sleep/wake |

## Entry points

**Primary (production — tests, verl adapter, recipes):**

```python
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client
```

- Config: `arctic_platform.rl.config.ArcticRLClientConfig`
- Factory: `arctic_platform.rl.client.create_arctic_rl_client` → async
  `ArcticRLRayClient`

**Unified client (migration target; subset of ops).** The unqualified name blocks;
the `Async` prefix awaits:

```python
from arctic_platform.client import ArcticClientConfig, ArcticRLClient, AsyncArcticRLClient
```

Prefer `arctic_platform.rl` for full RL (sleep/wake, `weight_norm`, …). See
`arctic_platform/client/UNIFICATION_NOTES.md` for what the unified client still
lacks.

## Quick start

```python
import asyncio
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client

config = ArcticRLClientConfig(
    model_name="Qwen/Qwen3-4B",
    training_gpus=8,
    sampling_gpus=8,
    log_prob_gpus=0,              # 0 = disabled
    colocate=True,
    checkpoint_path="/data-fast/my-rl-run/ckpt",  # required when training_gpus > 0
)
client = create_arctic_rl_client(config)

results = asyncio.run(client.generate(
    prompts=["Hello"],
    sampling_params={"max_tokens": 64, "temperature": 0.7},
))
client.shutdown()
```

Reconnect after a Ray handoff:

```python
rc = client.reconnect_config()  # fills training/sampling/log_prob job ids
client2 = create_arctic_rl_client(rc)
```

## Config (`ArcticRLClientConfig`)

| Field | Default | Notes |
|-------|---------|-------|
| `model_name` | **required** | HF id |
| `training_gpus` / `sampling_gpus` / `log_prob_gpus` | `0` | Job created iff > 0 |
| `log_prob_engine` | `"vllm"` | `"deepspeed"` or `"vllm"` |
| `colocate` | `False` | Fractional GPU sharing |
| `cuda_ipc` | `False` | Colocated weight-sync strategy, baked onto the training job at init: CUDA-IPC vs CPU-file |
| `low_memory` | `False` | With `cuda_ipc`, stream one param at a time (bounds peak GPU mem); sent at init |
| `ds_config` | `{}` | Training DeepSpeed config |
| `training_config` | `None` | Optimizer / scheduler / `training_horizon` |
| `log_prob_ds_config` | `None` | Log-prob DeepSpeed engine |
| `ds_worker_config` | `None` | Worker knobs, including **`zorro_train_enable`** |
| `vllm_config` | `None` | Sampling / vLLM log-prob |
| `arctic_inference_config` | `None` | **`zorro_inference`**, speculative decoding |
| `checkpoint_path` | `None` | Required for new training jobs |
| `full_determinism` / `seed` | `False` / `42` | Reproducibility |
| `job_ready_timeout` | `600` | Seconds |
| `training_job_id` / `sampling_job_id` / `log_prob_job_id` | `None` | Reconnect mode |

## Client methods

Async methods on the Ray client from `create_arctic_rl_client`:

| Method | Job | Notes |
|--------|-----|-------|
| `fwd_bwd(batch, processing=None)` | training | GRPO via `processing["loss_fn"]` |
| `fwd_no_grad(batch, processing=None, reference_model=False)` | training or log_prob | Old / ref log-probs |
| `step()` | training | Optimizer step |
| `save_checkpoint()` | training | |
| `generate(prompts, sampling_params, …)` | sampling | Rollouts |
| `log_probs(prompts, completions, top_k=1)` | log_prob | |
| `sync_weights(cuda_ipc=None, low_memory=None)` | cross-job | NCCL or CUDA-IPC; `None` uses the strategy set on the training job at init (config `cuda_ipc` / `low_memory`), pass a value to override one call |
| `weight_norm()` | cross-job | Debug after sync |
| `reset_prefix_cache()` | sampling | |
| `sleep_inference` / `wake_inference` | sampling | Colocation VRAM |
| `sleep_training` / `wake_training` | training | Offload / reload |
| `sleep_log_prob` / `wake_log_prob` | log_prob | |
| `empty_training_cache()` | training | |
| `reconnect_config()` | — | Serializable reconnect |
| `shutdown()` | all | Destroy jobs |

`save_weights(path)` is a stub on-prem (not implemented for disk reload yet).

## GRPO wire batch (sketch)

Unlike SFT's flat labels batch, RL `fwd_bwd` typically carries:

```python
{
    "batch": {...},
    "meta": {
        "rollout_n": int,
        "max_prompt_len": int,
        "max_response_len": int,
        "zorro_train_enable": bool,   # optional per-call override
        ...
    },
    "processing": {
        "loss_fn": "ap_grpo",          # or "grpo" / dotted path
        "post": ["ap_compute_logprobs", ...],  # Cortex post is "compute_logprobs"
        "config": {"eps_clip": 0.2, ...},
    },
}
```

Log-prob tensors often use a `_shifted` suffix convention (see
`arctic_platform.rl.ray_client`). Batch shapes still differ slightly across
backends — treat unification as WIP.

`fwd_bwd` omits the per-token result `batch` by default (VeRL only reads
`metrics`). Set `meta.return_fwd_batch` (or Cortex `context.return_fwd_batch`)
to include the merged `batch` in the response (TRL server-side-loss).

Metrics use the shared `{name}.sum` / `{name}.tokens` pairing; see
[`common.md`](common.md#metric-aggregation).

On packed input (`cu_seqlens` with a 1-D or single-row loss), `agg_loss` now
uses the packed sequence boundaries in `seq-mean-token-mean`,
`seq-mean-token-sum`, and `seq-mean-token-sum-norm`; earlier releases treated
the packed row as one sequence. `seq-mean-token-mean` therefore averages each
sequence's tokens and then the sequences. `seq-mean-token-sum-norm` divides
by the context's `packed_loss_scale_factor` when GRPO receives one, which is
the padded row width `S`: DSS sets it from `pack_meta.sequence_length`, and
this package's packing pipeline from `pack_meta["S"]`. When none is supplied,
it divides by the longest packed segment, after summing each segment's length
over sequence-parallel ranks. Earlier releases divided packed input by the
packed row's token count. Without `global_batch_size`, all three divide by the
number of sequences with policy tokens instead of by one. Unpacked `[B, S]`
input is unchanged.

With `sequence_loss_weights`, `loss_agg_mode="prompt-mean"` computes
`dp_size * sum(sequence_weight * sequence_loss_sum / sequence_token_count)`,
offsetting DeepSpeed's DP gradient averaging as the other modes do. Earlier
releases omitted `dp_size` on this weighted path, so with `dp_size > 1` its
gradient scale is now `dp_size` times larger; the unweighted
`prompt_group_ids` path already applied it. In this repository's integrations
(verl, SkyRL, the Cortex adapter) and recipes, nothing sends
`sequence_loss_weights`. In the PrimeRL Arctic adapter, the context builders
(`prime_rl/arctic/context.py`) attach them when every rollout's source
microbatch carries a weight or, failing that, when every rollout carries an
example ID, and the trainer step (`ArcticTrainerAdapter.run` via `_prepare_grpo_context`)
strips them before sending unless `loss_agg_mode` is `prompt-mean`.

The registered `ap_grpo` loss accepts `teacher_tau` with a positive `teacher_clip`
and prediction-aligned `teacher_log_probs_shifted` for the GRPO teacher term.
The term clamps the teacher-minus-policy log ratio to
`[-teacher_clip_negative, teacher_clip]`; `teacher_clip_negative` is an
optional finite non-negative lower-clip magnitude that defaults to
`teacher_clip`. `ap_grpo` accepts only the baseline GRPO config keys
(`_GRPO_CONFIG_DEFAULTS` in `grpo.py`) and the ratio-control keys below; the
unprefixed `grpo` additionally accepts its grouped-distillation `kd_coef`,
`kd_divergence`, `kd_beta`, and `kd_batch_num_tokens` keys. Any other key is
rejected by name in the batching and validation callbacks and again in the
loss, so a misspelled key such as `teacher_tao` fails instead of training
without its term.
`ap_grpo_echo_v1` requires `aux_ce_weight` and `echo_global_num_sequences`;
its `echo_observation_token_counts` batch column supplies the full observation
denominator when sequence parallelism splits observation tokens, and the
response then carries `echo_full_observation_denominator=1`. Both losses
support CISPO-only ratio gates (`ratio_mask_bounds_pos` / `_neg`,
`prob_diff_mask_max_pos` / `_neg`, `seq_mask_stat`, `seq_mask_bounds_pos` /
`_neg`, `ratio_m2_threshold`) and the independent `log_ratio_sq_coef` penalty;
`ratio_stats=True` enables additive per-bin telemetry without changing the
objective. Ratio-control keys without `use_cispo_loss=True` pass the batching
and validation callbacks; they are rejected when the packed loss reduction is
resolved (DSS and the native worker do this before any forward), and otherwise
by the loss. An explicit `null` is rejected for every ratio-control key (omit
the key instead), unlike most baseline keys where `null` turns the feature off;
a `null` end inside a bounds pair still means unbounded. The `_pos` / `_neg` gates split tokens by the sign of the
token-level advantage (before any sequence-level averaging; it includes the
teacher term when enabled): `advantage >= 0` counts as positive, so a
zero-advantage token uses the `_pos` settings. The `seq_stat_bin_*` histogram
always bins the sequence mean log ratio, even when `seq_mask_stat="mean_k3"`
selects the gating statistic. `ratio_m2_threshold` requires one packed model call per worker and
must be positive; it does not support sequence parallelism because M2PO
ranking needs one complete token set. `ap_grpo_mixed_v1` requires CISPO, a
finite positive `is_weight_clip_max`, token-level `importance_sampling_level`, and the
prediction-aligned `nll_mask` column; it intentionally rejects ratio-mask
options to avoid applying policy-only penalties to NLL tokens. It accepts only
the baseline GRPO config keys (`_GRPO_CONFIG_DEFAULTS` in `grpo.py`), so ECHO
keys and any unrecognized key are rejected. It also rejects `use_sapo_loss`,
`use_decoupled_loss`, `use_kl_loss`, an `entropy_coeff` other than `0.0`
(including `null`; omit it instead), any non-null `m2_threshold`, `c_clip`,
`behav_imp_weight_cap` or `current_version`, `prox_logp_method` other than
`recompute`, and the `prox_logp_shifted` and `rollout_is_weights` context
columns. The config-value rejections and the CISPO and importance-sampling
requirements name the key and the received value; a rejected context column is
named with its shape when it is a tensor, otherwise with its value. The ratio-mask and unknown-key rejections list the
offending keys without values, and a missing `nll_mask` is reported by name.
The unprefixed `grpo_mixed_v1` applies the same rules.

## ZoRRo Train

**What:** Prompt deduplication during RL forward/backward. Shared prompts are
packed once; per-response logprobs/gradients are reconstructed. Mathematically
equivalent to the naive path for supported models, with large wins on
long/shared prompts.

**Code:** `arctic_platform/rl/zorro_train/` (design + supported models:
[zorro_train/README.md](../arctic_platform/rl/zorro_train/README.md)).

**Enable (direct client / server)** — flat key on the worker config:

```python
ArcticRLClientConfig(
    ...,
    ds_worker_config={
        "zorro_train_enable": True,
        "response_len": 512,
        "max_token_len": 8192,
        "rollout_n": 8,
        "temperature": 1.0,
        "use_unpad": True,
        "logits_optimization": "none",  # "none" | "memory" | "compute"
    },
)
```

**Enable (verl Hydra)** — nested yaml (not the same string as the flat key):

```yaml
remote_backend:
  train:
    zorro_train:
      enable: True
      max_rollouts: ${actor_rollout_ref.rollout.n}
```

Shell override: `remote_backend.train.zorro_train.enable=True`.

> README shorthand `zorro_train.enable` refers to the **verl yaml** path. The
> server/client flat key is `ds_worker_config.zorro_train_enable`.

`ArcticRLClient` and `AsyncArcticRLClient` accept the flat key.
`ArcticClient`, `AsyncArcticClient`, and `ArcticSFTClient` raise `ValueError`
at construction when `zorro_train_enable` is truthy, and when a nested
`zorro_train.enable` dict is sitting on `ds_worker_config`. A later per-call
`meta["zorro_train_enable"]` is not checked at the client.

A Cortex `fwd_bwd` that sets `zorro_train_enable` must also send
`max_prompt_len`. One that sets `load_balancer` must also send
`max_response_len`, `max_token_len_per_gpu`, and either `rollout_n` or
`zorro_train_max_rollouts`. The payload builder raises `ValueError` naming
whatever of those keys is missing. The worker indexes them directly.
`run_pipeline` then raises `ValueError` unless that training job was created
with `ds_worker_config["zorro_train_enable"]`, which is what patches the model.
A request that only sets the per-call flag against an unpatched model is rejected.
That covers on-prem `fwd_bwd` and `fwd_no_grad`, because both call
`run_pipeline`. Cortex `fwd_no_grad` raises the same way: it has no forward
sub-job to patch.

## ZoRRo Inference (Forest Cascade Attention)

**What:** During decode, groups requests that share a KV-cache prefix and runs
grouped + per-suffix attention so each shared prefix block is read once per
*group* instead of once per *request*. Equivalent to standard attention;
larger wins with longer / more-shared prefixes.

**Code:** Implemented in ArcticInference / vLLM attention — not in this
package. Activated when `arctic_inference_config` contains:

```python
arctic_inference_config={
    "zorro_inference": {"enable": True},
}
```

That maps to `ModelConfig.use_fca = True` in
`arctic_platform.common.utils.server_models`.

**verl yaml:**

```yaml
remote_backend:
  rollout:
    zorro_inference:
      enable: True
```

Requires a matching vLLM + ArcticInference build. Design/tuning:
[Forest Cascade Attention](https://github.com/snowflakedb/ArcticInference/tree/main/arctic_inference/vllm/attention).

## Framework integrations

| Framework | In-repo path | Recipes |
|-----------|--------------|---------|
| **verl** | `arctic_platform/integrations/verl/` (`ArcticRLClientWrapper`, `arctic.yaml`) | [`recipes/rl/verl/`](../recipes/rl/verl/) |
| **SkyRL** | Driven from SkyRL's `integrations/arctic_rl/` | [`recipes/rl/skyrl/`](../recipes/rl/skyrl/) |

verl bootstrap sketch:

```bash
export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register
# hydra.searchpath += integrations/verl/config
trainer.remote_backend=arctic
```

verl integration lives in-tree; upstream merge may still be pending (see
project README).

## Examples and tests

Prefer the README snippet, `tests/rl/rl_harness.py`, and `tests/rl/test_e2e.py`
as reference. Some files under `arctic_platform/rl/examples/` are stale
(imports / sync-vs-async mismatches) — do not treat them as canonical until
cleaned up.

## Status / WIP

- Dual client stacks (`rl` async full vs `client` sync partial).
- Batch/response schema unification across backends.
- On-prem `save_weights` disk path unimplemented.
- ZoRRo Train model coverage is limited — see the supported-models list in
  `zorro_train/README.md`.
