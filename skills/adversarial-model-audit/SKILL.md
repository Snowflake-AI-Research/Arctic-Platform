---
name: adversarial-model-audit
description: >-
  Adversarial bug hunt for a model enabled (or about to be enabled) on Arctic Platform: try to disprove that RL
  training works correctly in every layer, attention kind, trainer and sampler, optimizer, weight sync and every
  other pipeline stage, at the configuration we will launch first. Studies the latest code and the latest release in
  parallel, launches the smallest jobs that prove each bug, keeps an HTML plan and findings report current, files
  every confirmed bug in DSS-Issue-Tracker with ISSUE.md, an SVG explainer, a reproducer, a fix PR and a Codex
  cross-review, and grows the model-agnostic regression suite in tests/model_audit. Use when asked to find bugs in a
  model implementation, audit a newly onboarded model, or harden a model before an RL launch.
disable-model-invocation: true
---

# Adversarial model audit

**The claim to disprove**, with `<MODEL>` and the launch configuration filled in:

> RL training for `<MODEL>` works correctly everywhere: every layer, every attention kind, in the trainer AND the
> sampler, the optimizer, weight sync, checkpointing, every stage of the pipeline, at `<LAUNCH CONFIG>`.

The job is to break that claim, exhaustively, and to leave behind a regression check for everything tried. A
passing check is a result too: it stays in the suite as a guard.

## Inputs

Ask only for what you cannot find:

| Input | Default |
| --- | --- |
| Model (HF id or local checkpoint) | required |
| Launch configuration | the longest context the user names, `bf16`, the release's default parallelism for this model |
| Code under test | the Cortex Training deployment the user names (host, database), reached only through the `cortex_training` client; record the DSS and AP commits it runs |
| Code to read | latest `origin/main` of Arctic-Platform AND the AP commit the latest DSS release pins (`dss-monorepo`) |
| Issue tracker | `Snowflake-AI-Research/DSS-Issue-Tracker` |
| GPUs | whatever this pod has; never take down a shared machine |

Follow the repo's `AGENTS.md` and `tests/AGENTS.md`. Checkpoints, downloaded exports and run artifacts live on fast
local storage (`AP_MODEL_AUDIT_DIR`, default `$TMPDIR/ap-model-audit`); the report and issue drafts go to durable
results storage.

## 1. Set up (first 15 minutes)

1. Two worktrees: latest `origin/main`, and the release pin. Record both SHAs in the report. A bug that is only in
   one of them is still a bug; say which.
2. Create the report from [report-template.html](report-template.html) **before any experiment**, with the plan
   tab filled in. Update it after every experiment, not at the end. Each update rewrites the file in place.
3. Write the model profile `tests/model_audit/profiles/<model>.yaml` (schema below) and run the CPU contracts:

   ```bash
   pytest -p no:xdist -sv tests/model_audit/test_contracts.py --audit-model <model>
   ```

## 2. Run the regression suite first

The suite is the cheapest broad sweep, and it already encodes every bug class found on earlier models. Run it
before reading code, then triage every non-pass.

Every trainer, optimizer, sampler and weight-sync probe is a job on the service, created from the model-catalog
configs the profile names and driven only through the client (`forward`, `forward_backward`, `step`, `generate`,
`weight_sync`, `save`/`export_checkpoint`). Nothing under audit runs in-process. The suite starts every job the
selected tests need at once, so a session costs about one job's scheduling and run time.

```bash
export AP_AUDIT_CATALOG=<cortex-training checkout>/model-catalog/models
export CORTEX_TRAINING_HOST=<host> CORTEX_TRAINING_DATABASE=<db> CORTEX_TRAINING_PAT=<from the environment, never inline>
pytest -p no:xdist -sv --session-timeout=36000 tests/model_audit --audit-model <model>                 # everything
pytest -p no:xdist -sv --session-timeout=36000 tests/model_audit --audit-model <model> -m "not vllm"   # trainer side
```

`AP_AUDIT_CLIENT_CONNECTION=<profile>` uses a Snowflake connection profile instead of the host and PAT; the explicit
host is the default so a machine's default profile cannot redirect the audit. `cortex_training` must be importable
(install it, or put a checkout's `src` on `PYTHONPATH`). `--session-timeout` matters: 0 means zero seconds, not
unlimited. `AP_MODEL_AUDIT_DIR` sets the artifact root (fast local disk); each training job downloads about 20 bytes per
parameter of exports (12 GB for Qwen3-0.6B).

Results land in `$AP_MODEL_AUDIT_DIR/results/<session>.jsonl` (one row per check, with the measured numbers). Copy
them into the report's suite tab. A failure is a lead, not a finding, until you have read the code path and
reproduced it outside the suite.

| Tier | File | Needs | What it compares |
| --- | --- | --- | --- |
| Contracts | `test_contracts.py` | CPU | launch length vs the config, the reference's reading of the config |
| Trainer | `test_trainer.py` | baseline job | the job runs the config it was sent and the expected loader; `forward` log-probs vs the fp32 reference per position bucket, determinism, rows in one request vs alone and reversed, requests ending in 1..8-token rows; per-parameter gradient norms for coverage; the reduced gradient (from the resumable export's Adam moment) vs the reference; a step-0 weights-only export vs the checkpoint |
| Optimizer | `test_optimizer.py` | baseline job, launch-lr job | the weights-only update vs AdamW on the gradient the optimizer consumed; the step's global norm vs the per-parameter norms; the applied clip coefficient (moment norm over reduced norm) vs the exact one; fp32 master-weight update loss at the launch lr; frozen params |
| Parallel | `test_parallel.py` | one job per layout | `client.layouts` (EP/SP/DP overrides of the launch config) vs the baseline job: log-probs, gradients, the update and the clip norm |
| Sampler | `test_sampler.py` | baseline job | `generate` prompt log-probs vs the trainer and vs fp32, repeatability, batch invariance; after `weight_sync` in each format, vs the reference on the trainer's export of the synced weights; TP degrees below launch (one sampling job each) vs fp32 |

Checks that need state the client does not expose skip as coverage gaps (the LM head's input gradient, the config
class the service's loader resolves, the optimizer implementation). The service also rejects requests with fewer
rows than data-parallel ranks, so layout jobs send only the full batch.

The reference runs locally on the real checkpoint, one GPU: the model's transformers class, or for a model
transformers does not implement the checkpoint's own inference code (`reference: deepseek_inference` runs the
`inference/model.py` DeepSeek ships, with its tilelang kernels replaced by torch and activation quantization off).
Check any new kernel stand-in against the real kernel before trusting the reference. Checks whose fp32 reference
(weights, gradients and accumulators, about 16 bytes per parameter) does not fit one local GPU skip with that
reason; say so in the report rather than scaling the reference down.

The baseline job is the catalog's launch config as the service runs it. A layout the service cannot build is a
finding about the service or the catalog, not a harness artifact, but check the catalog's conventions first (e.g.
`train_batch_size` must cover the data-parallel ranks).

Gates are relative to the reference's own launch-dtype error against fp32 (the noise floor), per position bucket
split at the profile's thresholds and per row (each row is its own request, so a short-request fault is not diluted
by long rows). Gradient floors also account for cancellation: a tensor whose per-row gradients nearly cancel gets
the typical per-row error rate scaled by its gradient mass, so a 3-element scale is not flagged for bf16 rounding.
The floor's low-precision run uses the launch's attention implementation (FlashAttention or SDPA move bf16
gradients several times further from fp32 than eager attention does); the fp32 truth stays eager.
Routed experts (per-expert tensors, or fused ones split along the expert dimension) are gated as a group: bf16
noise flips near-tied top-k choices, and each flip moves a whole token's gradient between
experts. Each expert gets its layer's worst sibling floor, and the layer's
lower-quartile expert error must stay within the lower-quartile floor, so a math error that touches every expert
fails while a flip-sized error in one expert does not (a known blind spot).
A check that a floor can hide needs its own observable; when the client has none (the LM head's input gradient),
it is a coverage gap.

Probe results are cached by their job body and inputs plus a hash of the probe code, the client version and the
deployment's host and database. The service changes under a cached result: pass `--audit-refresh` (or delete
`$AP_MODEL_AUDIT_DIR/runs/<model>`) for every audit session. Each job log names its job id; record them in the
report. Never widen a tolerance to make a check pass; a check that needs a wide tolerance
is reporting a real difference.

## 3. Plan: minimize time to the first critical bug

Order work by `P(bug) x severity / cost`, and put the launch configuration first. Severity, highest first: silent
wrong gradients or log-probs at the launch config; trainer/sampler mismatch (breaks the RL ratio); weight sync that
leaves the sampler stale; optimizer updates lost or misapplied; crashes at launch scale; everything else.

Default order, adjusted to what the model is:

1. The suite against the service (above). One job's scheduling and run time.
2. Code paths that switch with scale, at or just past each switch: sparse top-k thresholds, sliding windows, chunk
   and tile sizes (LM head, indexer, MLP, attention backward), kernel shortcuts for small inputs, RoPE/YaRN original
   length, int32 index arithmetic (`tokens x heads x dim` past 2^31 at long context), SP slice boundaries with
   uneven row lengths. Prove at the smallest length that crosses the switch, not at the launch length.
3. Trainer vs sampler on real weights: a few real layers on one GPU (layer slice), then the full model at modest
   length, k3 bucketed by position.
4. Weight sync on the real model: every derived tensor the sampler caches (merged or fp32 copies, dequantized
   weights, fused projections), FP8 packing, TP sharding, EP placement, buffers.
5. Parallel layouts at launch degrees on 8 GPUs, then multi-node only if the bug needs it.
6. Launch-length jobs last, to confirm what smaller runs predicted.

## 4. Investigate in parallel

Run three streams concurrently (subagents for the first two):

- **Code study, latest main.** Read every file the model touches: config, loader and matcher, conversion,
  modeling, attention and linear-attention kernels, MoE routing and token dispatch, LM head, sequence and expert
  parallel wrappers, optimizer policy, weight export (HF and vLLM), the vLLM model and AP's vLLM patches, the weight
  sync receiver. For each, write down the invariant it must keep and how it could break.
- **Release study.** The same for the release pin, plus the diff to main: bugs fixed on main but shipped in the
  release, and regressions on main.
- **Experiments.** The suite, then hypothesis-driven runs from the plan.

Hypothesis checklist (model-agnostic; tick each in the report with its evidence):

- Dispatch: loader resolution with and without an explicit loader, `model_type` vs text `model_type`, VLM wrappers,
  family predicates that pick kernels or optimizer policy.
- Checkpoint: the real tensor layout (packed or fused experts, FP8 scales, MTP layers, tied weights, vision
  towers), conversion caches keyed by path rather than content.
- Numerics per layer kind and attention kind against fp32; the indexer or router in the trainer vs the sampler.
- Packing: row boundaries in every attention, convolution and recurrent state; position resets; `cu_seqlens` under SP.
- Determinism: atomics in top-k or scatter, batch-composition effects, prefix caching.
- Trainer/sampler parity: precision casts, fused vs eager rounding, LM-head precision, temperature, router replay.
- Weight sync: name coverage both ways, values after sync, cached derived tensors, every sync path (CUDA IPC, CPU
  file, NCCL broadcast, LoRA). The suite syncs each `client.sync_formats` entry after a step (each format after the
  first one step further on, so a sync that does nothing cannot pass on the previous one's weights) and compares the
  sampler with the fp32 reference on the trainer's weights-only export of that state. It also requires the synced
  state to score unlike the sampler's previous weights. A broken format is a finding even when this launch takes
  another, since the client picks the format per job. The transport is the service's choice; a path it does not
  take for this job is untested.
- Optimizer: policy per family, frozen and FP8 parameters, bf16 master-weight update loss at the launch lr,
  parameters the model keeps in fp32 under bf16 master weights, weight decay groups, clipping across EP/SP,
  expert-data-parallel reduction.
- Parallelism: EP/SP parity, reduction groups, expert DP with SP, uneven sequence splits.
- Checkpoint save, resume and HF export; LoRA if it will be launched.
- Platform: config keys silently dropped, client/server default mismatches, retries that change workload identity.

## 5. Prove each bug with the smallest job

Escalate only when the smaller step cannot show it: CPU unit test, then a real-weights layer slice on one local GPU,
then the real model in a small service job (the suite's `client_probe.py` drives one), then launch scale. Use the platform's own job launcher for real-model runs, and record the job
id in the report.

CPU references: estimate peak memory before running (`params x bytes + activations`), keep it under half of free
RAM, slice layers rather than loading the whole model, and never run two big CPU references at once.

A finding needs a measured difference from a stated expectation, on named code at a named SHA. Execution errors
before any comparison are findings too, but say so. "Untested on GPU" must be stated when true.

## 6. File every confirmed bug

One folder per issue in `Snowflake-AI-Research/DSS-Issue-Tracker`:

```text
issues/<YYYY-MM-DD>-<slug>-<uuid4 hex>/
  ISSUE.md          from issue-template.md
  diagram.svg       explainer: the data flow, where it diverges, and what the fix changes
  repro/
    repro.py        standalone; prints the measured difference and exits non-zero on the bug
    README.md       one command to run, hardware, expected output before and after the fix
```

The fix is a PR to the repo that owns the code (usually Arctic-Platform), with a red/green test: the test fails on
the base and passes with the fix, and the PR body shows both. Link the PR from `ISSUE.md` and the issue from the PR.
Branch from latest main; if the bug is in the release, say whether the fix cherry-picks.

Cross-review every issue with Codex before calling it filed:

```bash
codex exec -s read-only -C <issue-folder> -o <issue-folder>/codex-review.md \
  "Review this bug report adversarially: is the claim true, is the reproducer minimal and correct, does the \
  evidence support the severity, and could the measured difference come from the harness instead of the code?"
codex exec review -C <fix-worktree> --base origin/main -o <fix-worktree>/codex-pr-review.md \
  "Review this fix for correctness, missed call sites, and whether its test fails without the fix."
```

Address every point or rebut it in `ISSUE.md`'s cross-review section. Record the verdict in the report.

## 7. Grow the regression suite

For every hypothesis tested, passing or failing, leave a check in `tests/model_audit` that would catch it on any
model:

- Prefer extending an existing check (a new position threshold, a new probe output) over a new file.
- A new check is a test with `@pytest.mark.audit_check("<tier>.<name>")` and a `register_check` claim. Compute
  model-specific inputs from the config or the profile, never by naming a model in the test.
- Keep it model-agnostic in what it touches, not only in what it names:
  - Observe service state only through what the client returns: the job's echoed config, the step reply, exports.
    Never re-create it by importing one family's module locally.
  - Read whole-model quantities from the service's own per-parameter reports or exports. Checkpoint names can tie,
    fuse or split tensors.
  - Handle both checkpoint layouts of a structure, e.g. per-expert tensors and fused `[experts, ...]` ones.
  - Force the condition a check needs (clipping engaged, a threshold crossed) instead of relying on the inputs
    reaching it.
  - Report a path the probe cannot drive (an FP8 receiver, a TP1-only route) as a coverage-gap skip, not a failure.
- Before keeping a check, run it on every profile. A failure on a model it was not written for is a finding or a
  false positive; resolve which before moving on.
- If it fails on this model, add `known_issues: {<check id>: {issue: <ISSUE.md URL>, summary: ...}}` to the profile.
  That turns it into a strict xfail, so the check starts failing (XPASS) the day the fix lands, until the entry is
  removed. Only xfail a deterministic failure; a crash that depends on free memory or timing would flap, so let it
  fail and record it. Key it `<check id>[<variant>]` (e.g. `parallel.gradients_match_reference[ep1-sp2]`) when only one
  parametrization fails.
- `not_applicable` needs a reason that is about the model, not about the harness.
- Hypotheses the suite's request sizes cannot reach (launch-length memory, multi-node) are recorded in the report
  with their job evidence, and get the smallest check that guards their mechanism.

## Profile schema

```yaml
name: <file stem>
checkpoint: <hub id or local dir>            # the local reference and the export checks read its weights
reference: transformers                      # or deepseek_inference: the checkpoint's own inference/model.py
client:
  catalog: <model-catalog/models directory name>
  variant: standard                          # or long
  training: h200-training-full               # catalog config names; leave one out to skip that sub-job
  sampling: h200-inference-full
  overrides: {training: {...}, sampling: {...}}   # deep-merged into the catalog sub-job configs
  layouts: [{id: sp2, training_config: {n_gpus: 2, sp_size: 2}}]
  sync_formats: [vllm, hf]
probe:
  audit_lr: 1.0e-3
  row_lengths: [7, 61, 130, 1030]            # or max_row_len plus thresholds
  thresholds: [{name: ..., position: ..., reason: ...}]   # token positions where code paths switch
tolerances: {noise_factor: 4.0, noise_slack: 2.0e-3, adamw_mismatch_fraction: 1.0e-3, max_lost_update_fraction: 0.5, grad_norm_rel: 1.0e-4}
expect:
  loader: <the training sub-job's model_provider>
  no_grad_params: [{pattern: <regex>, reason: ...}]
  export_up_to_shift: [{pattern: <regex>, reason: ...}]   # exported equal to the checkpoint up to a constant
known_issues: {<check id> or <check id>[<variant>]: {issue: <ISSUE.md or PR URL, or a label such as "upstream (vLLM)">, summary: ...}}
not_applicable: {<check id>: <reason>}
```

## Report

Keep one HTML file per model audit, from [report-template.html](report-template.html). Tabs: **Plan** (ordered
hypotheses with status), **Findings** (one row per issue: severity, affected configs, SHAs, links to ISSUE.md,
diagram, repro, fix PR, Codex verdict, regression check id), **Suite** (latest results JSONL rendered, with the
measured numbers), **Log** (timestamped experiments, job ids, dead ends). Link each finding to its `ISSUE.md` on
GitHub (`https://github.com/Snowflake-AI-Research/DSS-Issue-Tracker/blob/main/issues/<folder>/ISSUE.md`).

Final message to the user: the strongest finding first, then the count by severity, what was ruled out with
evidence, what remains untested and why, and the suite command that now guards all of it.
