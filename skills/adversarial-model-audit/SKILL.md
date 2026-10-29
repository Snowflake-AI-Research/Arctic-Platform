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
| Code under test | latest `origin/main` of Arctic-Platform AND the AP commit the latest DSS release pins (`dss-monorepo`) |
| Issue tracker | `Snowflake-AI-Research/DSS-Issue-Tracker` |
| GPUs | whatever this pod has; never take down a shared machine |

Follow the repo's `AGENTS.md` and `tests/AGENTS.md`. Checkpoints, tiny models and run artifacts live on fast
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

```bash
pytest -p no:xdist -sv tests/model_audit --audit-model <model>                 # everything the pod can run
pytest -p no:xdist -sv tests/model_audit --audit-model <model> -m "not vllm"   # trainer side only
```

`AP_MODEL_AUDIT_DIR` sets the artifact root (fast local disk). `AP_AUDIT_SAMPLER_PYTHON` points the sampler probes
at an interpreter with the release's pinned vLLM when the training env has a different one.

Results land in `$AP_MODEL_AUDIT_DIR/results/<session>.jsonl` (one row per check, with the measured numbers). Copy
them into the report's suite tab. A failure is a lead, not a finding, until you have read the code path and
reproduced it outside the suite.

| Tier | File | Needs | What it compares |
| --- | --- | --- | --- |
| Contracts | `test_contracts.py` | CPU | launch spec resolution and validation, length support, tiny-checkpoint coverage and layout |
| Trainer | `test_trainer.py` | baseline layout | production model vs transformers fp32 on the same tiny checkpoint: log-probs per position bucket, determinism, packed-row isolation, row order, 1..8-token samples, gradient coverage and values, LM-head gradient vs an unfused head, export round trip |
| Optimizer | `test_optimizer.py` | baseline layout | one step through `initialize_training_engine` vs reference AdamW; master-weight update loss at the launch lr; frozen params; implementation policy |
| Parallel | `test_parallel.py` | 2-8 GPUs | EP/SP/DP layouts vs the baseline layout: log-probs and the optimizer update (real gradient reductions) |
| Sampler | `test_sampler.py` | 1-8 GPUs, vLLM | sampler vs trainer and vs fp32, repeatability, batch invariance, production weight sync vs a fresh load, TP degrees below launch vs fp32 |

The baseline layout is the profile's `tiny.trainer.parallelism`: the smallest layout the production loader actually
builds (one GPU for dense models, EP2 for MoE loaders that require EP > 1). Never test a layout production cannot
build; its failures are harness artifacts.

Gates are relative to the reference's own launch-dtype error against fp32 (the noise floor), per position bucket
split at the profile's thresholds and per row (each row is its own request, so a short-request fault is not diluted
by long rows). Gradient floors also account for cancellation: a tensor whose per-row gradients nearly cancel gets
the typical per-row error rate scaled by its gradient mass, so a 3-element scale is not flagged for bf16 rounding.
A check that a floor can hide gets its own probe: the LM-head check compares the per-channel head-input contraction
against an unfused head, because a 25% error there is under the cancellation floor of the norm weight it feeds.

Probe results are cached by inputs plus a hash of the audit code, the platform sources and the torch, transformers,
DeepSpeed, vLLM, FlashInfer and Liger versions. Delete `$AP_MODEL_AUDIT_DIR/runs/<model>` to force a rerun. Other
jobs on the node change free GPU memory and with it vLLM's block counts; check a sampler failure's log for startup
errors before reading its numbers. Never widen a tolerance to make a check pass; a check that needs a wide tolerance
is reporting a real difference.

## 3. Plan: minimize time to the first critical bug

Order work by `P(bug) x severity / cost`, and put the launch configuration first. Severity, highest first: silent
wrong gradients or log-probs at the launch config; trainer/sampler mismatch (breaks the RL ratio); weight sync that
leaves the sampler stale; optimizer updates lost or misapplied; crashes at launch scale; everything else.

Default order, adjusted to what the model is:

1. The suite at the tiny scale (above). Minutes.
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
- Weight sync: name coverage both ways, values after sync vs a fresh load, cached derived tensors, every sync path
  (CUDA IPC, CPU file, NCCL broadcast, LoRA). Find the format and transport production uses for this model first
  (the orchestrator may switch formats for composite checkpoints) and set `probe.sync_formats` to it. The suite's
  probe syncs through the CPU file path only; if production uses another transport, leave `sync_formats` empty and
  record the gap instead of testing a path production never takes.
- Optimizer: policy per family, frozen and FP8 parameters, bf16 master-weight update loss at the launch lr, weight
  decay groups, clipping across EP/SP, expert-data-parallel reduction.
- Parallelism: EP/SP parity, reduction groups, expert DP with SP, uneven sequence splits.
- Checkpoint save, resume and HF export; LoRA if it will be launched.
- Platform: config keys silently dropped, client/server default mismatches, retries that change workload identity.

## 5. Prove each bug with the smallest job

Escalate only when the smaller step cannot show it: CPU unit test, then one GPU on a tiny checkpoint (the suite's
`audit_lib.build_tiny_checkpoint`), then 8 GPUs tiny, then a real-weights layer slice on one GPU, then the real
model in a small job, then launch scale. Use the platform's own job launcher for real-model runs, and record the job
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
- If it fails on this model, add `known_issues: {<check id>: {issue: <ISSUE.md URL>, summary: ...}}` to the profile.
  That turns it into a strict xfail, so the check starts failing (XPASS) the day the fix lands, until the entry is
  removed. Only xfail a deterministic failure; a crash that depends on free memory or timing would flap, so let it
  fail and record it. Key it `<check id>[<variant>]` (e.g. `parallel.gradients_match_reference[ep1-sp2]`) when only one
  parametrization fails.
- `not_applicable` needs a reason that is about the model, not about the harness.
- Hypotheses the tiny scale cannot reach (launch-length memory, multi-node) are recorded in the report with their
  job evidence, and get the smallest check that guards their mechanism.

## Profile schema

```yaml
name: <file stem>
checkpoint: <hub id or local dir>            # config.json and the safetensors index are read; weights are not
launch:
  dtype: bfloat16
  max_seq_len: 262144
  trainer:                                   # ModelSpec fields exactly as the platform sends them
    loader: <name>
    attn_implementation: <backend>
    parallelism: {expert_parallel: 8, sequence_parallel: 8}
    loader_options: {...}
  engine: {gradient_clipping: 1.0, fp32_masters: false}    # TrainingEngineSpec fields
  sampler: {tensor_parallel_size: 8, ...}    # vLLM engine args
  optimizer: {lr: 1.0e-6, betas: [0.9, 0.999], eps: 1.0e-8, weight_decay: 0.0}
probe: {pass_labels: true, audit_lr: 1.0e-3, pad_to_multiple: 8, sync_formats: [vllm]}
tiny:
  layers: [..]               # optional; default keeps the first layer of every distinct kind
  experts: 16
  text_overrides: {...}      # keep kernel-constrained dims (head dims, block sizes) and anything that sets the
                             # sampler's KV page layout (e.g. linear-attention heads) real
  config_overrides: {...}    # top-level, e.g. a vision tower
  thresholds: [{name: ..., position: ..., reason: ...}]   # token positions where code paths switch
  max_row_len: 4224
  trainer: {parallelism: {expert_parallel: 2, sequence_parallel: 1}}   # baseline layout: smallest one production builds
  parallel: [{expert_parallel: 2, sequence_parallel: 2, world_size: 4}]
  sampler: {tensor_parallel_size: 8, max_model_len: 8192}   # the launch TP; smaller degrees are swept
tolerances: {noise_factor: 4.0, noise_slack: 2.0e-3, adamw_mismatch_fraction: 1.0e-3, max_lost_update_fraction: 0.5}
expect:
  loader: <name>
  optimizer_implementation: <fused_adam|torch_adamw|...>
  no_grad_params: [{pattern: <regex>, reason: ...}]
known_issues: {<check id> or <check id>[<variant>]: {issue: <ISSUE.md or PR URL>, summary: ...}}
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
