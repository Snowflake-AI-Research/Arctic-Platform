---
name: correctness
description: Implement, debug, onboard, and validate the Arctic Platform correctness harness, its HF references, numerical checks, operator commands, and onboarding-group reports. Use for correctness-suite changes, model-config validation, and correctness reporting, not unrelated training or inference work.
---

# Arctic Platform Correctness

## Maintained Guides

**Use the shared library and its maintained commands, not a new model-specific launcher.** Start with the [operator guide](../../arctic_platform/correctness/docs/running-tests.md) for prerequisites, selection and reports. The [code inventory](../../arctic_platform/correctness/docs/code-inventory.md) identifies implementation boundaries and retired experiments.

**Read the [HF reference onboarding guide](../../arctic_platform/correctness/docs/hf-reference-onboarding.md) before changing preparation, compact records, calibration or cached-reference regression.** It defines reference identity, latest-set publication and retry compatibility. Historical records remain evidence; current regression requires compatible current identities.

**Keep maintained filenames descriptive and stable.** Do not include agent IDs, thread IDs or hashes in code filenames. Content-addressed reference records and immutable run receipts are data, not code, and retain their integrity keys. Extend the responsibility-based sections in `selftest/library.py`; `selftest/check_suite.py` is the single driver for all correctness selftests.

## Onboarding And Regression

**An onboarding request needs only the configs and tests to onboard.** Treat context discovery, independent per-test execution, compatible retry reuse, combined-reference preparation for the selected tests, retained outcome evidence and progress reporting as the skill's responsibilities, not extra instructions the operator must include. Do not require the operator to repeat the workflow or report format in the prompt. Preserve the existing authorization boundaries; onboarding does not authorize commits or pushes.

**Create and maintain root `correctness-progress.md` automatically for every onboarding request.** Read [the exact report templates](references/group-report-format.md), create the report if absent or update its existing matching group, and add each requested config immediately with `?` for pending selected tests before preparation or execution. Follow the template, status-change updates, runtime evidence, tallies and conditional failure proposals below without waiting for a separate reporting request. If selected tests already have compatible successful onboarding, preserve their recorded outcomes and report the skips rather than replacing evidence with `?`.

### GPU Execution Location

**Read [GPU job operation](references/gpu-job-operation.md) before using autorunner or launching on more than one GPU.** It gives the submit/watch protocol, all-node probe and setup procedure, single-node and multi-node correctness commands, and distributed diagnostic launch patterns. Apply it to the operator's selected allocation; do not ask the operator to supply launcher commands that the runbook already defines.

**Launch correctness once on the allocation head, not once per GPU or node.** Autorunner executes the submitted script on one host. The correctness CLI starts its owned DSS gateway with the verified same-job hostfile; that gateway starts the target workers. Use `--gpu-pool` for the selected target widths and keep single-node HF calibration separate. Do not wrap `onboard` or `run` in `torchrun`, `deepspeed`, or all-host `ds_ssh`.

**Keep coordination proportional to the task.** Continue as the execution owner unless a genuinely active command already owns the same source or allocation. An old handoff file or unrelated runner job is not evidence of conflicting ownership. Do not replace requested onboarding with agent-role negotiation or an open-ended audit; use the concrete probe, setup, execution and result steps in the runbook.

**Honor an execution target supplied in the initial prompt.** "Use the node you're running on" selects the current node; `Use job <job-name> via <connection-method>` selects that job. Verify the selected target's GPU and runtime eligibility, reuse supplied access details and ask only for missing information. Do not silently choose another target when the selected one is unavailable or unsuitable; explain the prerequisite and ask how to proceed.

**Do not assume the agent's checkout is running on a GPU job.** When the operator has not selected a target, verify an eligible local allocation or use an already identified GPU job with authorized connection details. If neither is available, ask: "Which GPU job should I use for onboarding, and how should I connect to it?" Ask only for missing job/access details; reuse known details rather than making the operator repeat them. If no eligible job is available, explain the required hardware and keep numerical work pending until the operator supplies it. Do not invent a job, acquire resources without authorization or substitute CPU controls for GPU onboarding.

**A remotely controlling agent runs the GPU workflow on the chosen job.** Use its supported command-execution tool or authorized connection method to perform setup, reference preparation, calibration, target validation and requested regression there. Discover and verify the job's checkout, active interpreter, visible GPUs and scratch locations; ask for the remote checkout path or environment only when they cannot be discovered. Shared source storage and a remote gateway do not supply local HF reference GPUs. For multi-node execution, verify the same-job hostfile and transport; never combine separate allocations. Prefer existing authentication or a secure credential source. Operator-supplied credentials may be used only for the authorized connection; never echo them or copy them into code, configs, documentation, logs, reports or durable work state.

**QA6 is a planned execution target, not a supported route yet.** A future request may say "Use QA6 with the supplied connection details and credentials." Until QA6 support is implemented, report that limitation rather than attempting an invented connection workflow or switching targets. Do not implement QA6 support as part of an ordinary onboarding request.

### Multi-Config Failure Strategy

**Ask which failure strategy to use before onboarding multiple configs.** Unless the operator has already chosen for this batch, ask: "Should I attempt onboarding all configs first regardless of outcome, then fix the failures, or fix each config until all applicable tests pass before moving to the next config?" Wait for the answer before starting or dispatching onboarding for the new batch. Record the answer in durable work state and follow it across retries, fleet assignments and session recovery. Do not assume a strategy, carry a previous batch's choice into a new batch without instruction, or ask again when an explicit instruction already selects one.

**Attempt-all mode records failures and continues across configs.** Attempt every selected applicable test for each config, report each outcome immediately, and continue to the next eligible config after ordinary failures. After the initial pass, repair the recorded failures in the operator's chosen priority order.

**Repair-first mode does not advance past a failing config.** Run its selected applicable tests independently, then repair and rerun failures until every applicable selected test and requested placement is `P` before starting the next config. `?`, `H`, `E`, `B` and `F` are not passes; `-` is inapplicable and does not need a pass. Never widen a gate, change workload identity or label a blocker as passed to advance. If a repair needs unavailable access, hardware or an out-of-scope change, explain the concrete blocker and ask for the required help or an explicit exception to advancing. Already-running work may finish. Unsafe cleanup stops further execution in either mode.

**The strategy governs progression between configs, not dependencies between tests.** A failed gradient test must not suppress optimizer or another selected test's independent onboarding on the same config. Preserve per-test success and retry only failed or missing tests.

### Configuration And Reference Layout

**Preserve catalog-relative recipe paths.** For a config inside `cortex-training/model-catalog/models/`, replace that prefix with `arctic_platform/correctness/configs/` and keep every remaining directory and the filename exactly. For example, `qwen3.5-2b/standard/h200-inference-lora.json` stays that relative path. Import recipe JSON, not the catalog's `model.json` metadata; do not rewrite recipe fields or bytes.

**Derive placement only for configs outside the catalog.** Use `<model-name>/<standard|long>/<gpu>-<inference|training>-<full|lora>.json`. The model name is the lowercase model basename, without its repository owner (for HF cache snapshots, use the model component rather than the snapshot hash). `long` means an explicitly configured `max_tokens_per_mb >= 262144`, including the training `mb_spec` or native `training.ds_worker_config.mb_spec`; missing budgets are `standard`, regardless of `max_seq_len` or runtime defaults. Training takes precedence when both training and sampling jobs exist. `lora` comes from the selected job's nonempty PEFT settings; otherwise use `full`. Resolve the GPU type from explicit recipe hardware, existing accelerator path/filename, or the executing GPU; do not guess a hardware identity when none is available.

**Reference folders mirror config folders, excluding the extension.** A config at `configs/qwen3.5-2b/standard/h200-training-full.json` uses `reference_data/qwen3.5-2b/standard/h200-training-full/<check-id>/...`. An explicit reference root changes only the prefix. Moving placement does not remeasure references, relabel numerical identities, or rewrite historical evidence.

**Distinguish migrated frozen IDs from new path-derived IDs.** Existing migrated configs retain their frozen IDs through `config-layout.json`. Otherwise, `config_id` joins the config-relative path components without the extension using `-`. Specifications and onboarding summaries stay under `specs/<config-id>.json` and `specs/<config-id>.onboarding.json`; reference folders follow placement, not the frozen ID. Resolve the ID through the helper or command output rather than guessing it from the source filename.

**Aliases are local, and recipe collisions need explicit variants.** Manifest aliases resolve deleted correctness-config paths only within their original config root; they do not replace a missing external Cortex recipe. Different existing recipes that share the derived filename retain descriptive `--<variant>` suffixes. New imports reuse identical bytes and refuse conflicting bytes; for a noncatalog variant append a lowercase descriptive suffix such as `--rl-16gpus` to the source filename stem before importing. Catalog recipes retain their exact catalog-relative filename.

**Keep canonical catalog paths free of incompatible historical recipes.** If a migrated frozen recipe occupies a catalog destination with different bytes, relocate that historical recipe and its mirrored references to a descriptive `--legacy-...` path before importing the catalog recipe. Re-key the manifest placement while preserving its frozen `config_id` and `previous_path`, verify all reference payload bytes, and leave frozen specifications unchanged. Update maintained examples and selftest expectations to the historical variant instead of treating the new catalog recipe as equivalent.

**A precision variant is a new workload, not a placement migration.** When an example needs explicit FP32 accumulation, use a noncatalog filename variant such as `--fp32` and prepare compatible new specifications and references. Do not overwrite an existing imported recipe, change its frozen manifest ID, or relabel its historical evidence. Catalog imports remain byte-identical to their source recipes.

**Inference-only catalog recipes are imports, not training jobs.** Preserve their sampling settings and placement. A training-based correctness check still requires a training sub-job and must reject an inference-only recipe before model preparation.

**Catalog variants are independent workloads.** When onboarding a catalog model's available recipes, inventory both `standard` and `long` directories and each `full` and `lora` training recipe; a pass on one does not validate another. Preserve absent variants as absent, and admit each existing recipe against the actual allocation and registered checks before execution. Keep inference-only imports and recipes exceeding the allocation's GPU width separate from numerical results.

**Report imported recipes separately from tested recipes.** Byte-preserving import and CPU layout controls establish placement, not a numerical pass. For each numerical onboarding or regression result, record the selected checks and GAS arms, actual GPU width, reduced-fixture scope, report location and cleanup evidence. Put detailed receipts in the results notes and only measured outcomes in the summary; leave untested inference or LoRA imports explicitly untested.

### Autonomous Failure Recovery

**The operator supplies only the config path and intended operation or check IDs once GPU-job access is established.** Discover the remaining context in the executing checkout and job; do not ask the operator to paste commands, collect logs, name report directories, choose internal backend flags, locate reference records or provide environment inventories. Missing GPU-job access is the exception: follow the GPU Execution Location section and ask where to run and how to connect. If the operation is omitted, use the matching failed attempt to determine whether onboarding or regression was intended; if no attempt exists, inspect the specification and references before choosing onboarding for a new config or regression for prepared inputs. An omitted test selection means the applicable registered checks, not always two.

**Resolve the executing checkout and configuration before reading run evidence.** Check the current Git root, active interpreter and installed `arctic_platform.correctness` location; follow the local config import in `harness/config.py`. Read the supplied config and its imported copy, deriving the config ID through that helper rather than guessing from a directory name. Inspect model source, precision, sequence length, GPU placement, attention implementation, optimizer and LoRA/RL settings. Run `python -m arctic_platform.correctness list` and the relevant command's `--help` for the actual branch's supported checks and options.

**Find reports by configuration identity, not merely by the newest directory.** Inspect `work/reports/`, `work/logs/`, and any `DSS_CORRECTNESS_LOGDIR` override. Read matching `batch.json`, `report.json`, `report.md`, `full.log`, `onboarding.log` and the config's `specs/<config-id>.onboarding.json`; follow their diagnostic, gateway and work-directory paths. Match config ID/path and check IDs, then compare attempt timestamps to identify the relevant failure without borrowing another run's pass. Inspect `work/logs/onboarding-state/<config-id>/` or the override's `onboarding-state/` request/phase manifests to find reusable preparation and original artifact locations. `harness/execution.py` and `onboarding/onboard.py` define these locations if the installed version differs.

**Discover prerequisites and prepared inputs from their recorded identities.** Read `specs/<config-id>.json` and its model/cache settings, plus matching records under `reference_data/<model>/<sequence-choice>/<recipe-stem>/<check-id>/` or a recorded `--reference-records-dir`. Inspect `DSS_CORRECTNESS_TMPDIR`, allocation-local hostfile settings, visible CUDA devices and installed dependencies using the command's existing preflight; do not manufacture GPU placement or switch kernels to bypass an error. Inspect Cortex connection variables without printing credentials. Follow the owned-gateway lifecycle and ownership records in `harness/gateway.py`; never kill an unrelated or explicitly external service.

**Repair the concrete failure and verify the requested operation through completion.** Use compatible retry state, preserve reviewed gates, run focused controls for code changes and rerun the requested onboarding or regression on eligible authorized hardware. After successful onboarding, run the requested applicable regression checks against its prepared records and inspect every selected arm, exit code, report and owned cleanup. If artifacts were deleted, discover that absence and rebuild only through explicit onboarding, not regression. Missing hardware, unavailable external files or unsupported production APIs must be reported precisely; do not repeatedly request evidence the agent can read or imply access that is unavailable.

**Inspect the registered checks and supplied config before selecting work.** `python -m arctic_platform.correctness list` shows the current branch's checks. P1 registers gradients and optimizer only; the full-suite branch also registers checkpoint, inference, E2E and RL checks. Applicability comes from the config and reviewed specification, not from a successful check on another model.

**Onboarding prepares references; regression only consumes compatible prepared records.** Preserve initial weights or adapter factors, materialized inputs, declared GPU placement, parameter coverage and reviewed gates. A missing or incompatible record is a prerequisite error, never permission to generate a replacement reference during regression or widen tolerance.

**Repeated onboarding is incremental per check.** Give the same config to `onboard` again, with or without an explicit check selector. The harness skips checks with a recorded successful validation matching the config, that check's frozen model/input/gate/settings and workload options; it runs only missing, failed or newly registered applicable checks. Adding another check does not invalidate earlier successes. Keep `specs/<config-id>.onboarding.json` alongside the specification: the summary merges individual-check successes instead of replacing them with the latest selection, and reports skipped checks separately from work executed in the current attempt. Applicability alone is not evidence of successful onboarding; older runs can be recovered from matching passing phase records under `work/logs/onboarding-state/`.

**Each check onboards independently, including its prerequisite checks and numerical validation.** Shared model/input preparation is not gradient onboarding. Optimizer-only onboarding prepares that fixture directly; a gradient mismatch must not stop optimizer or other applicable checks. After an ordinary prerequisite, calibration or validation failure, record the failing check and continue with the others. Preserve shared fixture inputs while rolling back only the unsuccessful check's specification changes. Interruptions propagate, and verified unsafe worker cleanup stops further execution; mark remaining checks `not_run` with the cleanup reason rather than inventing a dependency failure.

**Finish config-wide onboarding with every selected check's outcome.** Print and persist successes, failures, already-onboarded skips, inapplicable checks and checks not run for safety, including each failure's cause. Return nonzero when any selected check or combined-reference preparation fails without discarding independent successes. Retry only failed or missing checks once their concrete cause is repaired.

**Prepare references in the onboarding invocation, not as a separate operator step.** The default manual example is one command: `python -m arctic_platform.correctness onboard --config <config-path>`. It selects all registered applicable tests and prepares individual and combined-selection references automatically. Repeating that command prepares or reuses combined records even when the selected tests are already onboarded. Retry failed combined preparation with the same onboarding command without recalibrating successful tests. An onboarding skip is not a new regression pass; run regression and report its actual comparisons separately.

**Configs and compact reference records remain local until explicitly selected for publication.** Keep both `configs/` and `reference_data/` at their existing paths, untracked and not Git-ignored; the operator will choose individual files to commit later. Never blanket-stage either tree. Before removing tracking, back up and verify both complete trees, then change only the Git index, preserving every working file. Specifications are a separate tree and are not part of this untracking operation. Retain compatible references for regression; cleanup must not delete reusable inputs.

**The first-two supervised checks support automatic local gateway ownership.** When no Cortex host or base URL is configured, onboarding and regression start the installed upstream DSS service, use the reduced fixture and close their owned gateway. Operators use one command in one terminal, without manually starting another gateway. Explicit service settings remain external and must not be replaced or shut down. This automation does not establish support for later full-model or RL workflows.

**Automatic LoRA target support is scoped to local first-two supervised checks.** The local reduced-fixture Cortex route binds generated adapters after checking gateway capability and exact factor identity. Omit `--peft-adapter` for automatic generation. Use config-only onboarding by default; it invokes each applicable test independently and prepares individual and combined references for successful HF checks. A `--test` selector is optional for an intentionally narrower onboarding request; select both first-two tests explicitly during regression when only that adapter route is intended. Do not describe all Cortex LoRA execution as unsupported or extend this admission to hosted endpoints, RL, later checks or explicit supplied-adapter onboarding.

**Rebuilding a missing frozen fixture requires complete identity verification.** Use the maintained `onboarding.synth_model.materialize_pretrained` API with the specification's original source, layer count and vision depth; random materialization is not equivalent. Compare the rebuilt model's complete `harness.storage.model_identity` with every retained reference arm before reusing measurements. The short `content_hash` checks names and sizes only and cannot establish identical weight bytes. A mismatch remains a prerequisite failure; do not rewrite historical records or gates to accept it.

**Verify a changed operator workflow through its final report, not just startup.** Exercise the documented command on eligible hardware when available, inspect every selected arm, exit status, retained full log and owned-resource cleanup, then verify the documented regression against prepared references. Report unavailable runtime verification explicitly. Mocked lifecycle tests alone do not establish numerical correctness or complete distributed teardown.

**Reuse only compatible completed onboarding phases after a failure.** Consume the saved retry manifest and artifact checks. A target transport repair may preserve preparation but still requires target validation; changed reference code, dependencies, weights, inputs or gates cannot be treated as equivalent preparation.

**Generate one adapter identity across calibration and regression.** Onboarding and the runner must use the same explicit harness seed, PEFT recipe and base fixture. `generate_adapter`'s standalone default seed is not the runner's seed. Compare the prepared adapter identity before target work; repairing a seed mismatch requires fresh compatible preparation, not accepting the old records.

**A config edit invalidates every frozen spec of that config.** `config_checksum` in `harness/config.py` covers the training config; `run` reports a config checksum mismatch instead of comparing (`harness/execution.py`), and checkpoint-resume onboarding refuses the spec and asks for single-step-grads onboarding again. Before proposing a config change, trace what it does to each frozen spec of that config and which onboarding preflights then refuse it. The supplied recipe is what is measured; never change a knob so that a comparison agrees.

**A retained fixture lives on node-local fast storage, and its spec hash binds file names and sizes, not weights.** Reduced fixtures, staged models and harness work directories sit under `/data-fast` on the nodes that wrote them (`harness/storage.py`). `describe_fixture` (`harness/cortex.py`) accepts a directory whose `hash_directory` digest (`harness/config.py`), with or without the tokenizer JSON sidecars, equals the spec's `model.content_hash`, and that digest covers only the names and sizes of the safetensors and JSON files. Rerunning on another allocation therefore needs the original fixture copied to every node, or fresh onboarding: a regenerated random model with the same shapes can pass the hash check with different weights.

**One correctness command owns a host's local gateway at a time.** A second command started by the same user fails on the owner lock with "another correctness command still owns the local gateway" (`harness/gateway.py`). Select several checks in one command, or run the commands one after another.

## Numerical And Evidence Boundaries

**Gradient comparison requires every aligned trainable parameter.** Compare named gradient L2 norms after target reduction and before clipping. Missing reference gradients, unmatched target names and nonfinite values are coverage failures or prerequisites; do not silently omit tensors.

**The current optimizer norm criterion is weaker than full-tensor equality.** `optimizer-state-l2-norm-v1` compares absolute L2-norm differences for first moments, second moments and moment-independent update residuals after one AdamW step. Equal magnitudes can hide sign, permutation and directional errors. Preserve the criterion and calibrated absolute gate in records and reports.

**Checkpoint and RL checks have different contracts from supervised HF checks.** Follow the registered implementation and reviewed applicability. Do not manufacture supervised references for an RL configuration or infer checkpoint restore support from checkpoint export.

**Keep preparation, CPU controls, target execution and numerical verdicts separate.** A CPU pass, source pin, accepted dispatch, calibration or successful file staging is not a GPU comparison. A reduced fixture is not full-model or maximum-sequence-length validation. A retained supplied-DSS result is not automatically local AP production validation.

**Classify results from actual evidence.** A numerical pass requires completed comparisons. A numerical failure means a measured mismatch; an execution error precedes comparison. `B` is reserved for a verified missing production API, not missing test code, references, integration or capacity. `H` denotes harness/setup work and `?` denotes pending work in progress tables.

### Durable Outcome Evidence

**Save outcome evidence under Git-ignored `.agent-work/correctness/outcomes/`, never in tracked repository files.** Keep `/.agent-work/` in the checkout's `.gitignore` and verify each destination with `git check-ignore` before writing. Use a persistent checkout, not `/tmp` or node-local scratch that disappears on job reset. Keep the skill and report templates reusable: they describe evidence storage but do not contain live measurements. Reviewed specifications and reference records retain their existing locations; this directory holds the additional outcome evidence.

**Retain one immutable, compact record per config, test, placement and attempt before cleanup or updating the report.** Use `<config-relative-path-without-extension>/<test-id>/<unique-attempt-id>/` and include the config ID and checksum, executed source revision or dirty-source digest, fixture/reference identities, invocation, allocation/node/GPU counts, GAS arms when applicable, operation type and status. Preserve exact start/end timestamps and elapsed seconds with the timer's scope, startup/teardown and exit/cleanup results. Store all available scalar scores, named target/reference norms, differences, tolerance values and calibration provenance, not just extrema or rounded report cells. Keep compact JSON reports and relevant diagnostic text; exclude credentials, checkpoints, model weights, tensors and serialized tensor payloads. Preserve failures, skips and superseded attempts rather than overwriting them.

**Recover report numbers from matching saved records, and rerun an already executed test if its required timer is missing.** Match the config, test, fixture/reference identity, placement and operation before using historical measurements. Never split or duplicate a multi-test batch timer into per-test durations, substitute onboarding preparation time, or estimate missing numbers. When an independently timed matching test cannot be recovered, run that test separately through the maintained CLI on authorized hardware and retain its timing and outcome before replacing the report cell. Unrun tests stay pending; a blocked rerun needs its concrete prerequisite and next action recorded. Recompute the displayed-minute node-hour tally after each recovered or newly measured result.

### Required Group Report Format

**Follow the operator's specification exactly; do not add anything outside it.** Do not invent requirements, columns, tests, sections or scope extensions. Preserve specified column names, count and order in both report tables. P1 and explicitly first-two groups have exactly `config`, `single-step-grads` and `single-step-optimizer`; never add `H200 GPUs tested` or any separate GPU-count column. Record GPU placements only in the specified cells. An apparent improvement is not permission to change the specification.

**The report template constrains the report, not the agent's own work files.** The agent may write diagnostics, explanations, calculations and plans in its Git-ignored work files. Do not copy that prose into `correctness-progress.md`: every report element must come from the literal template or an explicitly requested operator change. Even verified, useful explanations are not permission to add paragraphs, footnotes or sections absent from the template. Before publishing a report update, remove any non-template prose introduced by that update.

**Always retain the report title and all three operator-instruction paragraphs.** Begin `correctness-progress.md` with the exact `# Correctness Onboarding Progress` title and three paragraphs in the template below, once per file. They explain creating or updating the tracker, keeping it outside Git and deleting or moving it after onboarding and PR publication. They are required operator instructions, not optional narrative. Preserve them on every update, even when all tests pass. Follow them with the exact `## Onboarding status tables and runtimes` heading; do not replace it with a catalog, date or group-specific heading. Keep the progress file untracked; do not stage it with configs or reference data.

**Apply rendering corrections to the templates as well as the live report.** Each operator correction updates the literal templates below and any corresponding example in [the report-format reference](references/group-report-format.md), then the affected rendering in `correctness-progress.md`. Do not leave a one-off report edit that the next group will undo. Use "test", not "check", in report headings and explanatory prose; preserve registered identifiers such as `single-step-grads`.

**Add requested onboarding work to the report immediately.** As soon as the operator gives a model or config to onboard, add its row to both the outcome and runtime tables, before discovery, preparation or testing; initialize the group if needed. Mark unrun selected tests `?` (`??` for the first 2 tests' GAS1/GAS4 outcomes) and runtimes `?`, then recompute the tallies. Include GPU counts only when known, and use `-` only after inapplicability is established. If a model request does not yet identify its recipe paths, use the supplied model name as a temporary row label and replace it with the actual config rows when discovered. Preserve existing measured outcomes on repeat onboarding; only missing or new tests start pending. Do not wait for onboarding success or a completed test to make requested work visible.

**Select only one larger placement per config.** Show the configured GPU count first and at most one additional result after `\|`: use 4x the configured count when it fits one authorized allocation, otherwise 2x when it fits. Never request or display both 2x and 4x for the same config/test. If neither fits, omit the separator and larger placement; do not combine separate allocations. Apply the same selection to outcome and runtime cells and count only displayed outcomes. For a sixteen-GPU allocation, configured counts of 1, 2, 4 and 8 select larger counts of 4, 8, 16 and 16 respectively; a configured count of 16 has no larger placement. Retain any measurements from an unselected placement in ignored outcome evidence or linked historical notes, not as a second larger entry.

**Copy the complete report header, outcome and runtime templates when starting a report.** Initialize the status section and its `Runtimes` subsection as soon as onboarding is requested, even when all outcomes and times are unknown. For an existing report, retain its mandatory title and introduction and add the requested rows to both tables. The first-two example has one unrun training config at eight GPUs and four pending GAS outcomes; replace only the config/link, selected test columns, applicability, placements, evidence, failure notes and recomputed counts, never the fixed headings or operator instructions. The seven-test full-suite example and counting rules are in [the report-format reference](references/group-report-format.md); use only the selected test columns in their existing order. Do not invent a catalog-specific format or add paragraphs absent from the templates. Keep catalog inventory, progress summaries and execution-scope details in linked results notes, not in an added preamble before the tally.

**Keep the outcome section free of narrative summaries.** When needed, `Possible fixes` follows the legend and GAS/placement explanation directly. Do not insert duplicated tally or inventory counts, summaries of passing runs, tensor counts or general scope paragraphs between them. Keep detailed scope and successful-run evidence in the linked results notes; merge only information needed to explain a current failure or its next action into that config/test's `Possible fixes` entry.

**Keep the runtime section strictly within the template too.** Do not append paragraphs listing measured seconds, explaining missing or combined timers, describing GPU/node accounting or summarizing attempts. Those details belong in Git-ignored outcome evidence and linked results notes, not the rendered group. The runtime tally contains only node-count rows with measured nonzero displayed time, followed by `Total`; omit unused buckets and `0.00` node-count rows. A pending group has only the `Total` row until measured time exists.

**Keep fixed wording and placement literal.** Use the three introductory paragraphs and legend below exactly, without paraphrasing. Runtime wording about both GAS arms applies only to the first 2 tests, `single-step-grads` and `single-step-optimizer`; do not generalize it to all tests. Use `### Possible fixes` only for current failures, then `### Runtimes` under `## Onboarding status tables and runtimes`. Do not add group names or units to these headings. Put the complete sentence `Reported test times are rounded to the nearest whole minute.` immediately above each runtime table, separated only by a blank line; do not use a bare unit label or put units in the section name. Use the capitalized label `Tally` immediately before the node-hour table.

#### Report Header And Outcome Section Template

```markdown
# Correctness Onboarding Progress

Create this file if it does not exist, or update it if it already exists. Use it to track progress when onboarding new config/test combinations.

Keep this file outside Git. It is a temporary progress tracker that each developer can use to see the current onboarding status.

Once onboarding is complete and the PR containing the new configs and `reference_data` files is published, you can safely delete this file or move it elsewhere.

## Onboarding status tables and runtimes

P+**F**+**E**+**B**+**H**+?+- 4(0+**0**+**0**+**0**+**0**+4+0)

| config | single-step-grads | single-step-optimizer |
| ------ | ----------------- | --------------------- |
| [<config-relative-path>](arctic_platform/correctness/configs/<config-relative-path>) | ??(8) | ??(8) |

Legend:

- `P` measured pass; `F` numerical mismatch; `E` execution error before comparison; `B` verified missing production API; `H` harness/setup work needed; `?` pending; `-` inapplicable.
- 2-letter results, like `PP` or `PF`, report 2 outcomes: one for GAS1 and the other for GAS4.
- Parenthesized numbers identify GPU placements, e.g., `P(1)` means 1 GPU, and `P(8)` means 8 GPUs.
- `|` separates the result at the config's specified GPU count from one additional result: 4x that count when it fits the allocation, otherwise 2x. Never show both. For example, `P(8) | P(16)` means the config specifies eight GPUs, and the test passed on eight GPUs and in an additional run on sixteen GPUs (2x the configured count).
```

**Revisit `Possible fixes` every time any outcome status changes.** Reconcile its entries with the current config/test/arm/placement outcomes, not only after test completion. Include proposals only for current `F`, `E`, `B` or `H` outcomes in this group's table. Remove a proposal when its corresponding outcome becomes `P`, `?` or `-`; if another arm or placement still fails, retain only that unresolved scope. When all outcomes are `P`, `?` or `-`, omit the entire subsection, including its heading; do not substitute "Nothing to fix" or a no-failures sentence. Keep historical failures in the linked notes, not as resolved fix proposals.

**Make each current fix proposal actionable.** Name every affected config/test pair, link its evidence, state the measured difference and current gate or concrete error, distinguish speculative causes, and give the next action and required GAS/placement rerun. A tolerance change requires explicit approval and verification; it does not turn a retained failure into a pass.

#### Conditional Failure Subsection Template

Include this subsection between the legend and `Runtimes` only when current `F`, `E`, `B` or `H` outcomes remain:

```markdown
### Possible fixes

- `<config-relative-path>` / `<test-id>`: [observed failure](<report-or-notes-link>) with the measured difference and current gate, or the concrete admission/runtime error. Possible cause (hypothesis): <unproved explanation>. Next action: <implementation investigation or repair>. Verification needed: <affected test, GAS arms and placements to rerun>.
```

#### Runtime Subsection Template

```markdown
### Runtimes

Test time excludes onboarding preparation, fixture preparation and tolerance calibration. Each verified single-test timer includes startup and recorded teardown, and both GAS arms for the first 2 tests (`single-step-grads` and `single-step-optimizer`).

Reported test times are rounded to the nearest whole minute.

| config | single-step-grads | single-step-optimizer |
| ------ | ----------------- | --------------------- |
| [<config-relative-path>](arctic_platform/correctness/configs/<config-relative-path>) | ?(8) | ?(8) |

Tally

| nodes | time in hours |
| ----- | ------------- |
| Total | 0.00 node-hours |
```

#### Finished Work Section Template

**End the report with exactly `## Add finished work`, once after all groups and runtime tallies.** Include a config only when at least one selected test is successfully onboarded with compatible current config, source, fixture, reference and gate identities. Include that test's reference directory only when every required gradient accumulation step (GAS) arm and every selected placement is `P`; `PF`, `PP(8) | ??(16)`, failed, pending, inapplicable or stale evidence is not eligible. Another selected test's failure does not exclude an independently successful test's config or references.

**List explicit successful paths, never whole trees.** Replace every placeholder below with the actual config path and mirrored `reference_data/<model>/<sequence>/<recipe>/<test-id>/` directory. Include each config once and only its successful selected tests' reference directories; remove unselected or unsuccessful test lines. Never include `correctness-progress.md`, an entire `configs/`, `reference_data/` or recipe directory, unrelated files, logs or scratch. When nothing qualifies, retain the final heading without a command block. The following block illustrates the selection when both first-two tests qualify; it does not claim they passed.

**The commands are instructions for manual publication, not actions to execute from the template.** Rendering or updating the report does not authorize staging, committing or pushing. Before separately authorized execution, review the Git index so the commit contains only the intended successful artifacts.

````markdown
## Add finished work

```bash
git add -- \
  arctic_platform/correctness/configs/<model>/<sequence>/<recipe>.json \
  arctic_platform/correctness/reference_data/<model>/<sequence>/<recipe>/single-step-grads/ \
  arctic_platform/correctness/reference_data/<model>/<sequence>/<recipe>/single-step-optimizer/
git commit -m "Add successfully onboarded correctness config and references"
git push
```
````

**Populate the tally from measured placements, not fixed node-count rows.** Each timing entry covers one test attempt at one placement, with both GAS arms only for the first 2 tests. Failed attempts include elapsed time until failure and recorded teardown. Round each duration once from exact seconds; sum those displayed minutes per verified node count and divide by 60. Insert only nonzero buckets in ascending node-count order. `Total` is the node-count-weighted sum, not the unweighted sum of bucket hours; unknown timings and unallocated prerequisite rejections add no time. Keep these calculation instructions out of the rendered report.

**Use the same config rows, selected test columns and placement order in both tables.** Keep the existing test names and order. Unselected tests are outside the table's scope, not `-`. Cells contain only linked result flags or durations and placement notation; use plain `-` for an inapplicable test, never `- (inference-only recipe)` or prose explaining a pending test. Put actionable failure explanations in `Possible fixes`, and detailed scope in the linked results notes.

**Update the whole report group after every status change and completed test, not after the whole batch.** Whenever a status changes, immediately recompute the outcome tally and reconcile `Possible fixes`, removing resolved or no-longer-applicable proposals and the entire subsection when no `F`, `E`, `B` or `H` remains. Once the actual report and cleanup evidence are available, update its report link, whole-minute runtime and runtime tally too. Run the reference's completion checklist before reporting progress. Preserve historical failures in the notes and keep independent unrun tests pending; a failed gradient test does not block optimizer onboarding. Imported recipes and CPU controls are not numerical passes.

**Count distinct blockages, not affected configurations or test arms.** A verified shared cause affecting three configurations is one issue with three affected configurations. Keep issue totals separate from configuration/check/placement coverage, start human-facing issue IDs at 1, and do not infer a shared cause from matching error text alone. Unknown distinct-cause totals stay `UNKNOWN`; harness defects do not enter core `B`/`E` counts.

**Owned gateway shutdown and complete worker cleanup are separate claims.** A direct child being reaped does not prove detached Ray workers are gone. Treat unsafe cleanup as a reason to stop subsequent workloads; preserve the primary failure and retained cleanup diagnostics.

**BF16 comparisons accumulate gradients in FP32 on both sides.** The HF reference accumulates in float32 (`reference/hf_single_gpu.py`). `apply_default_grad_accum_dtype` (`arctic_platform/common/deepspeed_engine.py`) sets `data_types.grad_accum_dtype` to fp32 when the config leaves it unset, except in the two ZeRO configurations where DeepSpeed refuses FP32 accumulation, and onboarding refuses a non-RL BF16 config whose resolved accumulation is not FP32 when a selected check compares to the reference. The FP32 accumulator costs GPU memory, so a full model that fits with BF16 accumulation can run out of memory in its first backward. Fix that memory use in Arctic Platform and keep the config and the default; a run with the default disabled only locates the cause and is never a result.

**An RL judge needs positive evidence for every rank and every sampler parameter.** Under sequence parallelism, compare what each rank received with that rank's own window, never with the whole capture. A parameter the trainer deliberately does not send, such as a frozen table it reports as skipped, still needs proof that the sampler's copy equals the trainer's, for example matching content digests. Exempting it by name proves only the trainer's side: a sampler that lost or overwrote its copy would still pass.

## Changes And Verification

**Synchronize P1 selectively, not by copying the full suite.** P1 registers only `single-step-grads` and `single-step-optimizer`. Carry shared config, reference, onboarding and capture fixes with their imports and tests, while preserving P1-only artifact APIs and first-step participation rules. Do not import checkpoint, inference or RL implementations merely because the full-suite file contains them. Before migrating local data, inventory config, specification and reference bytes; use a target-specific manifest and verify the payloads after moving them. A legacy recipe without hardware evidence cannot be assigned another recipe's accelerator.

**Trace synchronized behavior through the operator entry point.** A copied helper or mocked route test does not prove that the CLI invokes the helper. Check selection, per-width executor keywords, gateway and zone route installation, worker capture, decoding and cleanup together. Keep native optimizer routing limited to applicable local expert-parallel optimizer checks; dense and gradient-only runs must retain their ordinary capture path.

**Native optimizer capture needs its own receipts and node-local checkpoint inventory.** Keep ordinary participation receipts out of the native optimizer decoder; the native collector supplies the capture metadata. An AP expert-resume sidefile is not a dense model-state or optimizer shard. Select the model-state writer per node and restore the engine's writer predicate even on failure. The gateway and zone-head capture routes must bind the installed producer source, not assume that the invoking checkout is installed everywhere.

**Validate sampling payloads against the installed server schema.** Engine settings such as `dtype`, `enforce_eager` and `disable_custom_all_reduce` belong in `vllm_config.extra_engine_kwargs` for the supported DSS schema. Merge topology additions into that mapping rather than replacing existing settings. Exercise request parsing and engine preparation as well as dictionary assertions; accepted dispatch alone does not establish that the sampling engine can initialize.

**Run CPU validation without advertising GPU correctness.** Disable visible GPUs for `make test` when matching GitHub's CPU-only CI. Run focused harness controls and the configured formatting hooks too. Attribute residual failures to main only after reproducing the relevant cases against the baseline; keep the full-suite verdict separate from focused passes and previously completed numerical runs.

**Keep repairs at the responsible library boundary.** Reproduce the concrete failure with focused coverage in the existing domain module, repair it without changing workload identity or numerical acceptance criteria, and verify the affected cases. Broaden testing when shared behavior changes; retain earlier failures and avoid calling focused reruns a clean full-suite result.

**Diagnose a failing cell on a tiny random-weight model at the cell's real topology before running the full model.** Keep the architecture: the same layer types, the dense prefix, at least two MoE layers, MTP when the config has it, and an expert count divisible by the expert-parallel width; stage the model on every node. Run the full model only for a failure that depends on size, such as host memory during a save, a rank-0 gather or an export-copy deadline, and only after the tiny-model work is done.

**Prove which model a reduced-model diagnosis loaded.** When the config's canonical spec exists, checkpoint-resume onboarding binds the training jobs to that spec's `model.cache_path` (`onboarding/onboard.py`, `load_local_fixture` in `harness/cortex.py`), so substituting a reduced model through a hub redirect or a different `--source-checkpoint` still trains the model the spec names. For an RL config, run the diagnosis from a copy of the checkout without the canonical spec: onboarding then bootstraps a reference-free spec from `--source-checkpoint` and publishes it to the canonical path, which must not happen in the shared checkout. Confirm from the gateway log that every created job named the reduced model before calling the run tiny.

**A fix is accepted only when every previously passing cell its code can affect still passes.** Run the narrow reproducer first, then replay each affected passing run with its own invocation: the same spec, the same work-directory placement and exactly the harness switches that run set, no more and no fewer. Against a configured gateway, as on multi-node allocations, the invoker sets those switches: a checkpoint-resume run without `ARCTIC_CORRECTNESS_LOCAL_CHECKPOINT_BRIDGE=1` (`harness/cortex.py`) stops at its first save with "local resumable save returned only a FAST path/tag", and one with `ARCTIC_CORRECTNESS_PARTICIPATION_CAPTURE` (`arctic_platform/debug/training.py`) switched on fails its resumed job with "participation must be armed before the first backward". Revert a candidate that turns a passing cell into a failure or an unmeasured cell before trying another fix.

**Judge a silent phase by its own deadline and by process activity, not by log silence.** Some phases print nothing until they finish: `distribute_local_inference_export` (`harness/inference.py`) copies the step-save bytes to every admitted sampling node under a 14,400 s default deadline, and its docstring estimates about 2,000 s per node for a 642.7 GB export. Call a phase hung only when two samples of its processes' I/O counters and CPU time show no change.

**Use `pytest -sv` for direct selftest invocations.** `python -m arctic_platform.correctness selftest` is the supported suite command. Keep large model, checkpoint and tensor scratch on the harness's required fast storage; durable logs and compact records have separate locations.

**Document the actual operator experience.** Keep setup, onboarding and Running instructions in lifecycle order, including exact report/log locations, external-service behavior and failure handling. Preserve operator-authored PR text when updating instructions.

**Runnable examples must satisfy the executing branch's preflight.** Check the imported config and admission path, not only the helper or filename. P1 requires explicitly configured FP32 accumulation for BF16 reference comparisons; the full suite resolves the shared engine default. Do not copy one branch's accumulation guidance into the other without tracing that branch's onboarding implementation, or edit frozen recipes and references to make an example work.

**Catalog import and onboarding admission are separate checks.** Read the current recipe's declared hardware, GPU width, precision and accumulation settings rather than borrowing assumptions from an older filename or guide. `local_config_path` imports exact recipe bytes without preparing models or references; successful import does not satisfy the executing branch's numerical preflight. Label an incompatible catalog recipe as import-only and explain the rejected prerequisite, instead of rewriting catalog bytes or historical identities. A newly authored precision workload requires its own explicit noncatalog variant and fresh compatible preparation.

**The default manual example onboards all applicable tests in one command.** Use `python -m arctic_platform.correctness onboard --config <config-path>` without test selectors or a separate reference-preparation command. P1 and no-debug register only gradients and optimizer state; the full-suite checkout also selects compatible later tests, which have their own runtime prerequisites. State the executing branch's actual scope rather than describing full-suite default onboarding as a first-two-only run. Individual `--test` selectors are optional for intentionally narrower requests, not additional steps in the default workflow.

**Keep root `correctness-progress.md` strictly within the approved report templates.** Begin with the mandatory title and three operator-instruction paragraphs, then the status and runtime section in the template's exact order; finish with `## Add finished work`. Update result cells, tallies and current failure entries only as the template permits. Keep operational logs, cadence reports, reviews, source bindings, scope explanations and detailed history in the agent's Git-ignored work files or existing linked notes, not as extra report prose. Static guides and this skill contain reusable instructions, not live status. Do not promote numerical, runtime, maintenance or cleanup claims without their corresponding evidence. Fleet authorization and personal reporting rules come from the active operator instructions, not this skill.
