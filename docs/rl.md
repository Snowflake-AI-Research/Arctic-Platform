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
`_neg`, `ratio_m2_threshold`), the independent `log_ratio_sq_coef` penalty and
`ratio_mask_rebalance`, which rescales the kept advantages (below);
`ratio_stats=True` enables additive per-bin telemetry without changing the
objective. An explicit `false` of `ratio_stats` or of `ratio_mask_rebalance`
with no other ratio control is inert: no echo, no metrics and no
`use_cispo_loss` requirement.

Whenever ratio controls are active (any ratio-control key, or
`ratio_stats=True`, or `ratio_mask_rebalance=True`), the loss also reports four
additive signed-mass metrics,
`ratio_mask_{pos,neg}_{pre,kept}_mass_sum`. `pre` is the sum of `|A|` over the
policy tokens the loss trains; `kept` is the same sum over those that survive
every active ratio-control drop (ratio band, probability gap, sequence gate and
`ratio_m2_threshold`). The legacy `m2_threshold` shrinks the loss mask before
the ratio controls act, so a token it removes is not a trained token: it is in
neither `pre` nor `kept`, whereas a `ratio_m2_threshold` drop stays in `pre` and
leaves `kept`. `pos` covers the tokens whose advantage `A` is `>= 0` and `neg`
those with `A < 0`. `A` is the advantage the loss uses: with
`importance_sampling_level="token"` the token-level advantage (including the
teacher term when enabled), and with `"sequence"` the sequence-averaged
advantage, which gives every token of a sequence that sequence's mean, so the
whole sequence counts as `pos` or `neg` by the sign of its mean (the `_pos` /
`_neg` gates keep splitting by the token-level sign, below). The sums run
through the policy term's own aggregation, so they carry its denominators
(`batch_num_tokens`, `global_batch_size`, `loss_scale_factor`, row weights)
without the DP gradient compensation (`dp_size`), and no ratio, clip,
`rollout_is_weights` or behavior weight (`behav_imp_weight_cap`) is applied.
`token-mean` reports `sum(|A|) / batch_num_tokens`; weighted `prompt-mean`
reports `sum(sequence_weight * sum(|A|) / sequence_token_count)`. These
measurements precede the optional advantage scaling of `ratio_mask_rebalance`.

The sums add up across microbatches, DP workers and SP ranks under the
conditions in which the loss itself is split-invariant, that is when its
denominators are step-global:

- `token-mean`: `batch_num_tokens`.
- `seq-mean-token-sum` and `seq-mean-token-mean`: `global_batch_size`.
- `seq-mean-token-sum-norm`: `global_batch_size` and an explicit
  `loss_scale_factor` (context `packed_loss_scale_factor`); without it every
  call divides by its own longest segment or row width.
- `prompt-mean` with `sequence_loss_weights`: the weights.
- `prompt-mean` with `prompt_group_ids`: `global_batch_size`, and
  `prompt_token_counts` (the total number of loss-mask tokens over all rollouts
  of the row's prompt in the whole step, the same value on every row of a
  prompt; DSS derives it for grouped requests) whenever the rollouts of a prompt
  are spread over several calls or workers. Without it only prompts that sit whole in one
  call add up. One exception lets `global_batch_size` be omitted: with
  `torch.distributed` initialised and `dp_size > 1`, the loss all-reduces each
  call's prompt count, so prompts that sit whole on one worker, in one call per
  worker, add up across DP workers (not across the microbatches of a worker,
  where the count is each call's own).

Apart from that exception, the sums are not additive outside these conditions: a
call that lacks a denominator normalizes by its own counts, as the loss does, so
adding calls does not recover the unsplit value (a cut prompt without
`prompt_token_counts`, for example, is counted once per call). SP shards are
summed once: only the SP leader reports the group's total and the other ranks
report 0, as does a worker with no policy tokens. The M2PO ranking is per model
call, so both M2 mechanisms depend on how rows are placed. `ratio_m2_threshold`
ranks the tokens of each worker's single model call (see below), so its `kept`,
its dropped-token count and the k3 sum below are the sums of what each worker's
own ranking kept or dropped: they can differ from a single-process run, while
`pre` and `ratio_trainable_token_count` do not.
The legacy `m2_threshold` shrinks the loss mask per call before the ratio
controls act, so with it `pre`, `kept`, the k3 sum and
`ratio_trainable_token_count` all depend on how rows are placed in calls.

`ratio_mask_kept_k3_sum` is meant to be read next to dss-platform's
`trainable_logprob_lowvar_all/mean`, the k3 estimate averaged over all trainable
tokens before masking. For a token, k3 = exp(Δ) − 1 − Δ, where Δ is the trainer
log-probability minus the sampler log-probability (`old_log_probs_shifted`),
clamped to [−20, 20] before k3 is taken, as `trainable_logprob_lowvar_all/mean`
does. The sum is therefore finite for any finite log-probabilities (at most
exp(20) − 21 ≈ 4.85e8 per token). The clamp applies to this reported sum only:
`seq_mask_stat="mean_k3"` gates on the unclamped k3, so which tokens are dropped
does not depend on it. The sum runs over the kept tokens, the loss-mask tokens
that survive every active drop. It is a raw sum, not divided by the loss
denominators, so the denominator conditions above do not apply to it: it adds up
across microbatches, DP workers and SP ranks (counted once, like the masses) in
every aggregation mode, except for the placement dependence under
`ratio_m2_threshold` and the legacy `m2_threshold` described above. Two counts
come with it: `ratio_trainable_token_count`, the number of tokens the ratio
controls see (the loss-mask tokens, minus those the legacy `m2_threshold`
removes), and `ratio_mask_dropped_token_count`, the number of those that some
active drop removes, each counted once. The kept-token mean k3 is
`ratio_mask_kept_k3_sum / (ratio_trainable_token_count −
ratio_mask_dropped_token_count)`; with no drop, no legacy `m2_threshold` and
finite log-probabilities, `ratio_mask_kept_k3_sum / ratio_trainable_token_count`
is `trainable_logprob_lowvar_all/mean` up to rounding. That rounding is
dss-platform's: it evaluates `exp(Δ) − 1 − Δ` in fp32, which loses precision for
|Δ| of about 1e-3 or less (k3 is then about Δ²/2, far below the fp32 resolution
of `exp(Δ) − 1`), while this code uses `expm1`. A gap in that regime comes from
the dss-platform expression, not from a different definition. A trainer
log-probability that is not finite is replaced by 0 before the loss, and its
token still counts in this population (its Δ is then minus the sampler
log-probability), whereas `trainable_logprob_lowvar_all/mean` leaves such tokens
out of both its sum and its count. A sampler log-probability that is not finite
is not sanitised here: its token counts too, and adds k3 at the clamped Δ (4.85e8
for a −inf sampler log-probability, 19 for +inf) or makes the sum NaN for a NaN
one, whereas `trainable_logprob_lowvar_all/mean` leaves it out.

Ratio-control keys without `use_cispo_loss=True` pass the batching
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
ranking needs one complete token set.

`ratio_mask_rebalance=True` (boolean, default false; `true` activates the ratio
controls like `ratio_stats=True`, while `false` with no other control is inert)
undoes the loss of signed advantage mass that ratio masking causes.
Once the masks have decided, the kept tokens of positive advantage are scaled by
`P0/Pk` and those of negative advantage by `N0/Nk`, where `P0`, `N0` are the
`pre` and `Pk`, `Nk` the `kept` masses of the sign
(`ratio_mask_{pos,neg}_{pre,kept}_mass_sum`: the loss's own normalization, DP
compensation removed) summed over all DP and SP ranks, so every rank applies the
same scale. The sign is that of the advantage the loss uses, the sequence mean
under `importance_sampling_level="sequence"`. Only advantage magnitudes change:
which tokens are dropped, the counts and histograms and `ratio_mask_kept_k3_sum`
do not depend on the scaling. Each sign returns to its own pre-mask mass, not to
the other sign's: masses that were unbalanced before masking stay so. A sign with
no surviving mass has nothing to scale and gets scale 1, so its mass stays lost.
The scale is computed as `(Pk + Pd)/Pk`, where `Pd` is the dropped mass summed
over the ranks (each rank contributes `pre - kept`, which is an exact 0 when it
dropped nothing of that sign), provided the loss aggregation returns bit-equal
sums for bit-equal inputs. So when no token of a sign is dropped on any rank its
scale is exactly 1, and with nothing dropped the loss and gradients are
bit-identical to `ratio_mask_rebalance=False`. This was tested on CPU for every
aggregation mode (float32 advantages, one process) and for float64 advantages on
4 and 7 ranks (`token-mean`). For float32, bfloat16 and float16 advantages the
cast of the scale to the advantage dtype also absorbs a difference of about
1e-16, on any device. For float64 advantages with a scatter-based aggregation on
CUDA (`scatter_add_`, which the packed per-sequence sums and the grouped
`prompt-mean` use, is nondeterministic there) it is untested, and the scale could
differ from 1 by about 1e-16. The CISPO cap still clips ratios, independently of
advantage scaling; this does not equalize ratio-weighted gradients.

The scale of a sign is `pre/kept` of the world-summed masses (computed as
above). It is not capped and is not reported as a metric, but it can be derived from the reported
`ratio_mask_{pos,neg}_pre_mass_sum` and `ratio_mask_{pos,neg}_kept_mass_sum`
(each summed over the workers). It is large when little of a sign's mass
survives: with one of 100 equal-magnitude positive tokens kept it is 100, and
with a survivor whose advantage is a thousandth of the dropped token's it is
1001; the survivors then carry the sign's whole pre-mask mass.

This version requires exactly one synchronized model call per worker: a request
with more than one model call on any worker is rejected before any forward, with
a message that names `ratio_mask_rebalance` and the per-worker counts. DSS checks
it once, when it prepares the request, on the counts of all workers, and gives
every worker the same request-wide count. The native worker and
`run_pipeline(pack=True)` check it on each worker, before their first forward,
and do not exchange the verdict: the worker passes its number of
gradient-accumulation microbatches, the pipeline its number of packed
microbatches, which the splitter first raises to the maximum over the ranks when
`torch.distributed` is initialised (a rank with fewer rows than that maximum
fails in the splitter instead). Under data parallelism every worker must
therefore see a count of one, that is one gradient-accumulation step and, with
`pack=True`, every worker's tokens within `max_tokens_per_mb`: a worker that
rejects the request while a peer has passed the check leaves that peer waiting in
the restoration's world collective until the process-group timeout, instead of
the request failing. Exact multi-microbatch support requires a request-wide
statistics prepass. Balancing each microbatch independently is a different
objective: it restores the request's per-sign mass only when no microbatch loses
a whole sign, and it concentrates each microbatch's removed mass on that
microbatch's survivors.

Whenever ratio controls are active the response carries `ratio_mask_rebalance`
(1.0 or 0.0), the echo of the flag. With the flag true it adds
`ratio_rebalance_{pos,neg}_{post,unrestored}_mass_sum`, built on the existing
`ratio_mask_{pos,neg}_{pre,kept}_mass_sum`: `post` is the kept mass times the
scale, in float64, so it is 0 for a sign whose kept mass is 0 and otherwise equals `pre` to rounding (about 1e-16 relative), while
the advantages the loss uses are scaled in their own dtype, so the mass they
carry matches `pre` to that dtype's rounding (about 6e-8 relative in fp32);
`unrestored` is the pre-mask mass of a sign with no surviving mass and 0 otherwise.
Like the other mass sums, these four are additive over DP workers (there is one
model call per worker), and the SP leader alone reports its group's total.
With `ratio_m2_threshold` the masses, and so the scale, come from what each
worker's own ranking kept.

`ap_grpo_mixed_v1` requires CISPO, a
finite positive `is_weight_clip_max`, token-level `importance_sampling_level`, and the
prediction-aligned `nll_mask` column; it intentionally rejects ratio-mask
options, by key presence (an explicit `ratio_stats=false` or
`ratio_mask_rebalance=false` as well), to avoid applying policy-only penalties
to NLL tokens. It accepts only
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
