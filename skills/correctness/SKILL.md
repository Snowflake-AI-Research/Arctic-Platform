---
name: correctness
description: Implement, debug, onboard, and validate the Arctic Platform correctness harness, its HF references, numerical checks, and operator commands. Use for correctness-suite changes and model-config validation, not unrelated training or inference work.
---

# Arctic Platform Correctness

## Maintained Guides

**Use the shared library and its maintained commands, not a new model-specific launcher.** Start with the [operator guide](../../arctic_platform/correctness/docs/running-tests.md) for prerequisites, selection and reports. The [code inventory](../../arctic_platform/correctness/docs/code-inventory.md) identifies implementation boundaries and retired experiments.

**Read the [HF reference onboarding guide](../../arctic_platform/correctness/docs/hf-reference-onboarding.md) before changing preparation, compact records, calibration or cached-reference regression.** It defines reference identity, immutable publication, retry compatibility and explicit historical reuse.

**Keep maintained filenames descriptive and stable.** Do not include agent IDs, thread IDs or hashes in code filenames. Content-addressed reference records and immutable run receipts are data, not code, and retain their integrity keys. Extend existing responsibility-based selftest modules; `check_lora_training.py` is the maintained LoRA driver.

## Onboarding And Regression

### Autonomous Failure Recovery

**The operator supplies only the config path and intended operation or check IDs.** Discover the remaining context locally; do not ask the operator to paste commands, collect logs, name report directories, choose internal backend flags, locate reference records or provide environment inventories. If the operation is omitted, use the matching failed attempt to determine whether onboarding or regression was intended; if no attempt exists, inspect the specification and references before choosing onboarding for a new config or regression for prepared inputs. An omitted test selection means the applicable registered checks, not always two.

**Resolve the executing checkout and configuration before reading run evidence.** Check the current Git root, active interpreter and installed `arctic_platform.correctness` location; follow the local config import in `harness/config.py`. Read the supplied config and its imported copy, deriving the config ID through that helper rather than guessing from a directory name. Inspect model source, precision, sequence length, GPU placement, attention implementation, optimizer and LoRA/RL settings. Run `python -m arctic_platform.correctness list` and the relevant command's `--help` for the actual branch's supported checks and options.

**Find reports by configuration identity, not merely by the newest directory.** Inspect `work/reports/`, `work/logs/`, and any `DSS_CORRECTNESS_LOGDIR` override. Read matching `batch.json`, `report.json`, `report.md`, `full.log`, `onboarding.log` and the config's `specs/<config-id>.onboarding.json`; follow their diagnostic, gateway and work-directory paths. Match config ID/path and check IDs, then compare attempt timestamps to identify the relevant failure without borrowing another run's pass. Inspect `work/logs/onboarding-state/<config-id>/` or the override's `onboarding-state/` request/phase manifests to find reusable preparation and original artifact locations. `harness/execution.py` and `onboarding/onboard.py` define these locations if the installed version differs.

**Discover prerequisites and prepared inputs from their recorded identities.** Read `specs/<config-id>.json` and its model/cache settings, plus matching records under `reference_data/<config-id>/<check-id>/` or a recorded `--reference-records-dir`. Inspect `DSS_CORRECTNESS_TMPDIR`, allocation-local hostfile settings, visible CUDA devices and installed dependencies using the command's existing preflight; do not manufacture GPU placement or switch kernels to bypass an error. Inspect Cortex connection variables without printing credentials. Follow the owned-gateway lifecycle and ownership records in `harness/dss_driver.py`; never kill an unrelated or explicitly external service.

**Repair the concrete failure and verify the requested operation through completion.** Use compatible retry state, preserve reviewed gates, run focused controls for code changes and rerun the requested onboarding or regression on eligible authorized hardware. After successful onboarding, run the requested applicable regression checks against its prepared records and inspect every selected arm, exit code, report and owned cleanup. If artifacts were deleted, discover that absence and rebuild only through explicit onboarding, not regression. Missing hardware, unavailable external files or unsupported production APIs must be reported precisely; do not repeatedly request evidence the agent can read or imply access that is unavailable.

**Inspect the registered checks and supplied config before selecting work.** `python -m arctic_platform.correctness list` shows the current branch's checks. P1 registers gradients and optimizer only; the full-suite branch also registers checkpoint, inference, E2E and RL checks. Applicability comes from the config and reviewed specification, not from a successful check on another model.

**Onboarding prepares references; regression only consumes compatible prepared records.** Preserve initial weights or adapter factors, materialized inputs, declared GPU placement, parameter coverage and reviewed gates. A missing or incompatible record is a prerequisite error, never permission to generate a replacement reference during regression or widen tolerance.

**Compact reference records are generated local artifacts, not files to commit.** Keep `reference_data/` Git-ignored and retain its compatible records for subsequent regression. Staging instructions include config and reviewed specifications only; cleanup must not delete references still needed for testing. Demo cleanup is restricted to local checkout subdirectories; allocation scratch is left for job reset.

**The first-two supervised checks support automatic local gateway ownership.** When no Cortex host or base URL is configured, onboarding and regression start the installed upstream DSS service, use the reduced fixture and close their owned gateway. Operators use one command in one terminal, without manually starting another gateway. Explicit service settings remain external and must not be replaced or shut down. This automation does not establish support for later full-model or RL workflows.

**Verify a changed operator workflow through its final report, not just startup.** Exercise the documented command on eligible hardware when available, inspect every selected arm, exit status, retained full log and owned-resource cleanup, then verify the documented regression against prepared references. Report unavailable runtime verification explicitly. Mocked lifecycle tests alone do not establish numerical correctness or complete distributed teardown.

**Reuse only compatible completed onboarding phases after a failure.** Consume the saved retry manifest and artifact checks. A target transport repair may preserve preparation but still requires target validation; changed reference code, dependencies, weights, inputs or gates cannot be treated as equivalent preparation.

## Numerical And Evidence Boundaries

**Gradient comparison requires every aligned trainable parameter.** Compare named gradient L2 norms after target reduction and before clipping. Missing reference gradients, unmatched target names and nonfinite values are coverage failures or prerequisites; do not silently omit tensors.

**The current optimizer norm criterion is weaker than full-tensor equality.** `optimizer-state-l2-norm-v1` compares absolute L2-norm differences for first moments, second moments and moment-independent update residuals after one AdamW step. Equal magnitudes can hide sign, permutation and directional errors. Preserve the criterion and calibrated absolute gate in records and reports.

**Checkpoint and RL checks have different contracts from supervised HF checks.** Follow the registered implementation and reviewed applicability. Do not manufacture supervised references for an RL configuration or infer checkpoint restore support from checkpoint export.

**Keep preparation, CPU controls, target execution and numerical verdicts separate.** A CPU pass, source pin, accepted dispatch, calibration or successful file staging is not a GPU comparison. A reduced fixture is not full-model or maximum-sequence-length validation. A retained supplied-DSS result is not automatically local AP production validation.

**Classify results from actual evidence.** A numerical pass requires completed comparisons. A numerical failure means a measured mismatch; an execution error precedes comparison. `B` is reserved for a verified missing production API, not missing test code, references, integration or capacity. `H` denotes harness/setup work and `?` denotes pending work in progress tables.

**Count distinct blockages, not affected configurations or test arms.** A verified shared cause affecting three configurations is one issue with three affected configurations. Keep issue totals separate from configuration/check/placement coverage, start human-facing issue IDs at 1, and do not infer a shared cause from matching error text alone. Unknown distinct-cause totals stay `UNKNOWN`; harness defects do not enter core `B`/`E` counts.

**Owned gateway shutdown and complete worker cleanup are separate claims.** A direct child being reaped does not prove detached Ray workers are gone. Treat unsafe cleanup as a reason to stop subsequent workloads; preserve the primary failure and retained cleanup diagnostics.

## Changes And Verification

**Keep repairs at the responsible library boundary.** Reproduce the concrete failure with focused coverage in the existing domain module, repair it without changing workload identity or numerical acceptance criteria, and verify the affected cases. Broaden testing when shared behavior changes; retain earlier failures and avoid calling focused reruns a clean full-suite result.

**Use `pytest -sv` for direct selftest invocations.** `python -m arctic_platform.correctness selftest` is the supported suite command. Keep large model, checkpoint and tensor scratch on the harness's required fast storage; durable logs and compact records have separate locations.

**Document the actual operator experience.** Keep setup, onboarding and Running instructions in lifecycle order, including exact report/log locations, external-service behavior and failure handling. Preserve operator-authored PR text when updating instructions.

**Keep root `correctness-progress.md` as a concise, human-facing results page.** Put the AP outcome-by-config/check tables first; update actual result cells and only necessary scope notes. Keep operational logs, cadence reports, reviews, source bindings and detailed history in `correctness-progress-notes.md`, not the results page. Static guides and this skill contain reusable instructions, not live status. Do not promote numerical, runtime, maintenance or cleanup claims without their corresponding evidence. Fleet authorization and personal reporting rules come from the active operator instructions, not this skill.
