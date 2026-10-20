# Correctness Code Layout Plan

**Status: implemented and CPU-verified on 2026-10-20.** Code reorganization covers only `arctic_platform/correctness/`. The only separately requested outside documentation change is a link from `docs/correctness.md` to the static operator guide. No production code, repository-level packaging, DSS, or existing run scripts were reorganized. GPU numerical testing remains a separate activity.

## Implementation

**The operator command now calls a shared library.** [`../api.py`](../api.py) provides `RunRequest`, `RunOptions`, selection, and structured batch results; [`../harness/execution.py`](../harness/execution.py) handles command parsing, sequential execution, and saved reports. [`../__main__.py`](../__main__.py) is a thin driver. Existing `run`, `hosted`, `onboard`, `publish-fixture`, `list`, and `selftest` commands remain available.

**Common behavior has one implementation.** [`../harness/sft.py`](../harness/sft.py) holds configured training settings and answer-token scoring, replacing check-to-check helper imports. [`../models/original.py`](../models/original.py) centralizes pinned original-model metadata and validation. Diagnostics use [`../diagnostics/__main__.py`](../diagnostics/__main__.py); the LM-head experiment uses the library's executor argument rather than changing command-module globals.

**Batch results are short on screen and detailed on disk.** Each selected pair receives a terminal result row and a saved reason. Reports retain per-arm comparisons and full tracebacks. Missing prerequisites do not hide independent cases; interruption or failed owned-resource cleanup stops subsequent work.

**Cleanup removed duplicated implementation, not required history.** Selection and execution no longer live in the main driver, shared scoring no longer lives in an individual check, and original-model validation no longer lives in the numerical runner. The [code inventory](code-inventory.md) records retained files and their actual dependencies. No diagnostic met the evidence requirements for safe archival or removal, so no archive directory was invented and required paths remain available.

**The static operator guide and integrated CPU checks are verified.** [`running-tests.doc`](running-tests.doc) documents all three selection modes and prepared-node operation. The documented `selftest` command collected 2,500 tests: 2,499 passed and one Ray-dependent test was skipped, with no failures or errors. Operator list/help commands and invalid-selector rejection were also exercised. This is implementation validation, not a new numerical verdict or GPU-capacity admission.

## Intended Result

**An operator can run one configuration/test pair, all tests for one configuration, or one test across all configurations with one documented command.** The command uses the shared correctness library, not a newly written model-specific launcher. Adding a model normally adds configuration data and, only where required, a model-family adapter.

**The library owns reusable behavior; drivers only select and invoke it.** Numerical checks, reference computation, model validation, execution, and reporting have separate responsibilities. These boundaries do not require one file per helper or a new framework.

## Original Foundations

**The migration extends the existing package rather than building a competing library.** Before the migration, the package already had a test registry, reviewed specifications, reusable checks, target transports, compact reference records, and command-line entry points.

- `__main__.py` previously combined operator commands with substantial selection and execution logic; the current [driver](../__main__.py) delegates those responsibilities.
- [`../harness/runner.py`](../harness/runner.py) combines execution, reference invocation, model-specific admission, and result handling.
- [`../harness/transport.py`](../harness/transport.py) already defines a shared training/checkpoint interface.
- [`../harness/registry.py`](../harness/registry.py) already supplies stable check IDs.
- `checks/e2e_sft_train_validate.py` previously imported shared dataset and scoring helpers from `inference_checkpoint_loss.py`; both now import the shared library.
- [`../harness/reference_records.py`](../harness/reference_records.py) binds records to source contents and paths; relocation is therefore not automatically reference-compatible.
- `diagnostics/` currently contains 37 Python files, many with independent command-line entry points. Lack of an import does not prove these directly invoked experiments are obsolete.

## Organization

**Retain familiar directories and give each a clear role.** Avoid a new top-level package, a duplicate `lib/` hierarchy, or a large module containing every responsibility.

```text
correctness/
  __main__.py       Thin, stable operator command
  api.py            Small supported Python interface used by the drivers
  harness/          Shared configuration, execution, inputs, records, and reporting
  checks/           Test-specific numerical comparisons
  reference/        Independent reference math and a stable subprocess driver
  onboarding/       Explicit preparation and calibration, separate from regression
  models/           Small number of model-family adapters and reviewed metadata
  diagnostics/      Only maintained experiments, with one shared command interface
  configs/          Existing reviewed job configurations
  specs/            Existing reviewed specifications and tolerance gates
  reference_data/   Existing compact records, preserved
  selftest/         Library and command regression tests, grouped by behavior
  docs/
    running-tests.doc  Static human operator guide
    ...                Package-local maintainer instructions and migration plan
  archive/          Only created when a historical file is proven safe to retire
```

**Use a few stable drivers, not one launcher per job or model.** The operator command covers regression and explicit preparation; the existing reference subprocess command preserves process isolation; maintained diagnostics share one dispatch path. Existing supported commands remain compatible during migration.

**Expose a small library interface, not command-parser internals.** Drivers convert arguments into a validated run request and invoke preparation or regression through `api.py`. Library callers receive structured results. Reuse existing configuration, specification, result, and transport types where possible. Keep low-level helpers private; diagnostics must not replace functions on the command module to customize execution.

**Consolidate meaningful duplication, not merely short files.** Move common dataset preparation, answer-token scoring, and training settings out of individual check modules into shared library code. Group tightly related state capture, fixture validation, and execution helpers where dependencies justify it. Keep distinct numerical algorithms and regression tests readable.

**Model adapters contain model differences, not copies of the runner.** A family adapter can provide original-model validation, reference loading/placement, canonical parameter mapping, and explicitly supported capabilities. Common code still validates runtime availability, reviewed configuration, and actual capacity. Registering metadata never makes an unsupported model runnable.

**Keep the reference implementation independent of target math.** Shared data preparation and identities are appropriate; reusing the target optimizer or forward implementation as the reference is not.

## Operator Workflow

**The operator launches on an already-prepared compute node.** The operator has secured sufficient CPU memory, GPUs, GPU memory, and storage for the selected cases and installed all required software. Examples run from the repository root using that prepared Python environment. The runner does not allocate nodes, submit allocation requests, install dependencies, or provision a remote machine. Existing target connection options remain available where needed; any multi-node configuration must fit within one already-secured allocation.

**The ordinary regression command supports three selection modes once the execution environment is configured.** Accept an existing configuration path or an exact catalog ID; do not require operators to find generated scripts or paste hashes into routine commands. Configuration and test selection are independent: omitting the test selects all maintained tests for that configuration, while omitting the configuration selects all active reviewed configurations for that test.

Available interface:

```bash
python -m arctic_platform.correctness list

# One configuration and one test.
python -m arctic_platform.correctness run --config <config-id> --test <test-id>

# All tests for one configuration.
python -m arctic_platform.correctness run --config <config-id>

# One test across all configurations.
python -m arctic_platform.correctness run --test <test-id>

python -m arctic_platform.correctness selftest
```

**Expand selections through the existing configuration catalog and test registry.** Use a documented deterministic order and include each selected configuration/test pair once. Archived experiments are not candidates. Reject unknown selectors before starting work. Require at least one selector; an argument-free `run` does not accidentally start every test on every configuration.

**A batch reports every selected pair, not just the pairs that can run.** Execute applicable cases with satisfied prerequisites; record an explicit reason for each inapplicable or blocked case. Continue independent cases after a numerical failure or prerequisite error, but stop when failed cleanup or shared execution-environment damage makes further execution unsafe. Never silently enlarge placement, combine allocations, change tolerances, or regenerate a reference to make a case runnable.

**Document the human workflow once in a static guide.** `docs/correctness.md` points to `arctic_platform/correctness/docs/running-tests.doc`, using the requested filename. The guide starts with a checklist of required resources, installed software, and environment settings rather than an allocation or installation workflow. It explains configuration/test selection, ordinary regression, explicit reference preparation, saved results, interruption, and common failures with worked command examples. Onboarding and test runs never write or append to this guide. Documentation changes are deliberate maintenance, not run artifacts; add regression coverage for this non-writing requirement.

**Keep instructions separate from changing results.** The existing `docs/correctness.md` remains the reader's entry point and retains its recorded outcomes; add a prominent relative link to the operator guide without moving or rewriting historical evidence. The guide contains no current-job status, onboarding receipt, run log, or automatically updated outcome table. Runtime logs and reports use separate run directories, with large artifacts under `/data-fast`; compact reference records retain their existing publication rules.

**Keep environment settings separate from test selection.** Store supported connection and storage defaults in one documented environment configuration. Preserve existing command-line overrides and never write secrets into configurations or reports. Check required settings and resources before starting the selected workload, without installing or allocating anything automatically.

**Report missing prerequisites before expensive work where possible.** Errors name the missing model files, reference record, checkpoint, runtime, or node count and give the supported next action. Use only nodes from the same allocation. Capacity observations remain observations, not a lasting reservation.

**Regression must not silently prepare a new baseline.** Reference generation, calibration, model acquisition, or specification changes remain explicit preparation actions. A shorter command must preserve the reviewed model, input order, tolerances, and comparison rules.

**Every run provides a short terminal summary and complete saved detail.** Display the selected configuration/test pairs, progress, outcomes, elapsed time, report location, and any required operator action. Batch output uses one compact result row per pair and final counts, with complete per-case results, full tracebacks, and cleanup status saved separately. Numerical failures, execution errors, or blocked applicable cases produce a nonzero aggregate exit status; inapplicable cases are reported separately and never counted as passes.

## Cleanup Rules

**Classify each candidate before moving or removing it.** Check imports, direct module commands, registry entries, tests, documentation links, saved source manifests, and current run-owner dependencies. Do not treat a timestamp, version suffix, or missing import as proof that code is dead.

**Remove proven unused duplication; archive historical experiments with continuing evidentiary value.** Archive entries record the original path, original checksum, replacement if any, reason for retirement, and relevant historical result. Archived code is not registered as an active check or collected by ordinary selftests.

**Do not break references outside the allowed scope.** A historical file still required at its existing path by an external report, frozen manifest, or active run stays in place until that dependency can be retired safely. Do not move outside files to make the cleanup possible.

**Preserve valuable regression coverage.** Combine duplicated tests where equivalent coverage is demonstrable; keep tests for distinct failures. Passing once is not a reason to delete a regression test.

**Preserve immutable evidence and current work.** Compact measurements, reviewed specifications, recorded failures, submitted scripts, and existing source snapshots are not disposable clutter. `.codex-run-state/`, `blocked.md`, and files elsewhere remain untouched by this reorganization.

## Migration Order

1. **Inventory and classify.** Produce a package-local keep/consolidate/archive/remove list, including direct invocations and current owners. Establish current supported operator workflows and select one already-supported configuration for the first migration.
2. **Define the operator interface and static guide.** Use the agreed prepared-node launch assumption and document environment defaults. Specify all three selection modes, command examples, short output, batch outcomes and failure behavior, and the small library API that commands call. Write `correctness/docs/running-tests.doc` and add the requested link from `docs/correctness.md`; no generated operation updates this guide.
3. **Separate drivers from library behavior.** Extract selection and orchestration from the main command without changing numerical behavior. Reuse the existing registry and transports. Provide only compatibility wrappers that have real callers.
4. **Consolidate shared inputs and model support.** Remove check-to-check helper dependencies, centralize validated model differences, and combine duplicated helpers. Preserve independent reference math.
5. **Resolve source-bound reference compatibility explicitly.** Preserve historical identities and measurements. Validate any proposed old-to-new compatibility mapping from exact source changes and focused parity tests; never silently rename record keys or claim a new source created an old measurement. If equivalence cannot be established, keep the incompatible path unavailable pending explicit preparation.
6. **Migrate future runs, then retire old code.** Let current runs finish with their pinned sources. Switch new runs to the stable drivers and shared library. Historical scripts outside this folder remain unchanged; new routine execution should not require a new Python program.
7. **Validate and clean up.** Run focused tests after each phase, then the maintained CPU selftest suite for the integrated library change. Exercise operator success, missing prerequisites, numerical failure, interruption, full traceback retention, and owned-process cleanup. A limited GPU integration run is separate and should reuse existing references, not replay passed numerical cases just for a directory change.

## Acceptance Criteria

**The change is complete only when structure and operation both improve.**

- All code reorganization stays inside `arctic_platform/correctness/`; no production file is moved or modified. The only outside documentation change in this plan is the requested operator-guide link in `docs/correctness.md`.
- A human can list configurations/tests and run one pair, all tests for one configuration, or one test across all configurations using the documented command without locating a job script.
- Commands run on an operator-provided, sufficiently resourced node with prerequisites already installed; the runner validates requirements but does not allocate machines, submit jobs, or install software.
- Focused selection tests cover both selectors, either selector alone, unknown selectors, an empty selection, deterministic ordering, duplicate prevention, mixed batch outcomes, and aggregate exit status. Blocked and inapplicable cases remain visible.
- Adding support for a model does not require editing stable driver files.
- Routine runs use the library directly and do not depend on chains of versioned scripts in `.codex-run-state/`.
- Existing supported commands continue to work or have an explicitly agreed migration.
- Reviewed configurations, tolerance gates, baseline provenance, and numerical behavior remain unchanged unless separately authorized.
- Inactive code is removed from active discovery; historical material remains recoverable.
- Console output is short, saved reports contain complete error details, and cancellation cleans up owned processes.
- `docs/correctness.md` links to the requested `arctic_platform/correctness/docs/running-tests.doc`; the guide has usable worked examples and remains byte-identical through onboarding and regression execution. No runtime workflow writes status or results into the guide.
- Focused and integrated regression checks pass; any untested GPU behavior is identified explicitly.

## Agreed Launch Boundary

**No launch-location question remains.** For this iteration, operators provide the prepared compute environment and issue commands there. Resource allocation and prerequisite installation are outside the operator-interface redesign. Batch cases run sequentially by default so the resource requirement is the largest selected case, not the sum of all selected cases; concurrent scheduling is not required for this refactor.
