---
name: correctness
description: Implement, debug, onboard, and validate the Arctic Platform correctness harness, its HF references, numerical checks, and operator commands. Use for correctness-suite changes and model-config validation, not unrelated training or inference work.
---

# Arctic Platform Correctness

## Maintained Guides

**Use the shared library and its maintained commands, not a new model-specific launcher.** Start with the [operator guide](../../arctic_platform/correctness/docs/running-tests.doc) for prerequisites, selection and reports. The [code inventory](../../arctic_platform/correctness/docs/code-inventory.md) identifies implementation boundaries and retired experiments.

**Read the [HF reference onboarding guide](../../arctic_platform/correctness/docs/hf-reference-onboarding.md) before changing preparation, compact records, calibration or cached-reference regression.** It defines reference identity, immutable publication, retry compatibility and explicit historical reuse.

**Keep maintained filenames descriptive and stable.** Do not include agent IDs, thread IDs or hashes in code filenames. Content-addressed reference records and immutable run receipts are data, not code, and retain their integrity keys. Extend existing responsibility-based selftest modules; `check_lora_training.py` is the maintained LoRA driver.

## Onboarding And Regression

**Inspect the registered checks and supplied config before selecting work.** `python -m arctic_platform.correctness list` shows the current branch's checks. P1 registers gradients and optimizer only; the full-suite branch also registers checkpoint, inference, E2E and RL checks. Applicability comes from the config and reviewed specification, not from a successful check on another model.

**Onboarding prepares references; regression only consumes compatible prepared records.** Preserve initial weights or adapter factors, materialized inputs, declared GPU placement, parameter coverage and reviewed gates. A missing or incompatible record is a prerequisite error, never permission to generate a replacement reference during regression or widen tolerance.

**The first-two supervised checks support automatic local gateway ownership.** When no Cortex host or base URL is configured, onboarding and regression start the installed upstream DSS service, use the reduced fixture and close their owned gateway. Operators use one command in one terminal, without manually starting another gateway. Explicit service settings remain external and must not be replaced or shut down. This automation does not establish support for later full-model or RL workflows.

**Verify a changed operator workflow through its final report, not just startup.** Exercise the documented command on eligible hardware when available, inspect every selected arm, exit status, retained full log and owned-resource cleanup, then verify the documented regression against prepared references. Report unavailable runtime verification explicitly. Mocked lifecycle tests alone do not establish numerical correctness or complete distributed teardown.

**Reuse only compatible completed onboarding phases after a failure.** Consume the saved retry manifest and artifact checks. A target transport repair may preserve preparation but still requires target validation; changed reference code, dependencies, weights, inputs or gates cannot be treated as equivalent preparation.

## Numerical And Evidence Boundaries

**Gradient comparison requires every aligned trainable parameter.** Compare named gradient L2 norms after target reduction and before clipping. Missing reference gradients, unmatched target names and nonfinite values are coverage failures or prerequisites; do not silently omit tensors.

**The current optimizer norm criterion is weaker than full-tensor equality.** `optimizer-state-l2-norm-v1` compares absolute L2-norm differences for first moments, second moments and moment-independent update residuals after one AdamW step. Equal magnitudes can hide sign, permutation and directional errors. Preserve the criterion and calibrated absolute gate in records and reports.

**Checkpoint and RL checks have different contracts from supervised HF checks.** Follow the registered implementation and reviewed applicability. Do not manufacture supervised references for an RL configuration or infer checkpoint restore support from checkpoint export.

**Keep preparation, CPU controls, target execution and numerical verdicts separate.** A CPU pass, source pin, accepted dispatch, calibration or successful file staging is not a GPU comparison. A reduced fixture is not full-model or maximum-sequence-length validation. A retained supplied-DSS result is not automatically local AP production validation.

**Classify results from actual evidence.** A numerical pass requires completed comparisons. A numerical failure means a measured mismatch; an execution error precedes comparison. `B` is reserved for a verified missing production API, not missing test code, references, integration or capacity. `H` denotes harness/setup work and `?` denotes pending work in progress tables.

**Owned gateway shutdown and complete worker cleanup are separate claims.** A direct child being reaped does not prove detached Ray workers are gone. Treat unsafe cleanup as a reason to stop subsequent workloads; preserve the primary failure and retained cleanup diagnostics.

## Changes And Verification

**Keep repairs at the responsible library boundary.** Reproduce the concrete failure with focused coverage in the existing domain module, repair it without changing workload identity or numerical acceptance criteria, and verify the affected cases. Broaden testing when shared behavior changes; retain earlier failures and avoid calling focused reruns a clean full-suite result.

**Use `pytest -sv` for direct selftest invocations.** `python -m arctic_platform.correctness selftest` is the supported suite command. Keep large model, checkpoint and tensor scratch on the harness's required fast storage; durable logs and compact records have separate locations.

**Document the actual operator experience.** Keep setup, onboarding and Running instructions in lifecycle order, including exact report/log locations, external-service behavior and failure handling. Preserve operator-authored PR text when updating instructions.

**Use root `correctness-progress.md` for local fleet outcomes when that file exists.** Static guides and this skill contain reusable instructions, not live status. Do not duplicate counts or promote numerical, runtime, maintenance or cleanup claims without their corresponding evidence. Fleet authorization and personal reporting rules come from the active operator instructions, not this skill.
