# Required Onboarding Group Report Format

**Every onboarding report uses the operator-edited structure in root `correctness-progress.md`.** Read this reference before creating or updating a report. Always include the exact `# Correctness Onboarding Progress` title and all three introductory operator-instruction paragraphs from the template below, once per file. Keep them on every update, including when all tests pass. Follow them with `## Onboarding status tables and runtimes`, the outcome table, complete legend, optional `### Possible fixes`, matching `### Runtimes` and both tallies. Finish the file with exactly `## Add finished work`, after all groups and runtime tallies. Initialize both tables immediately when onboarding is requested, before discovery, preparation or testing. Add `Possible fixes` only when the current table contains `F`, `E`, `B` or `H`. Revisit it after every status change, not only after completed tests. Keep the progress file outside Git; detailed receipts and history belong in `correctness-progress-notes.md`.

**Use the selected seven-test registry scope, not a fixed first-two report.** The registered order is `single-step-grads`, `single-step-optimizer`, `checkpoint-resume-loss`, `inference-checkpoint-loss`, `e2e-sft-train-validate`, `rl-weight-sync`, `rl-router-replay`. The RL tests compare live training/sampling evidence, not HF reference norms. Include only selected test columns in this order in both tables. The skill's first-two template is a deliberately narrower example; the full-suite example below retains all seven columns. The first two tests require paired GAS1/GAS4 outcomes. Later-test cells follow their actual returned arms and placements; do not invent GAS pairs or collapse measured arms into one result.

**Requested onboarding work appears immediately with pending status.** Add the supplied model or config to both tables as soon as the operator gives it for onboarding. Unrun selected tests start as `?`, or `??` for the first 2 tests' GAS1/GAS4 outcomes; their runtime cells start as `?`. Recompute the tallies immediately, include placement counts only when known, and reserve `-` for established inapplicability. For a model request without identified recipes, temporarily use the supplied model name as the row label, then replace it with the discovered config rows. Repeat onboarding preserves existing measured results; initialize only missing or newly selected tests as pending. Never wait for onboarding success or test completion to add the row.

**The operator's specification is exact, not a starting point for additions.** Do not add unrequested columns, tests, sections, requirements or scope, or paragraphs absent from the templates. The three introductory operator-instruction paragraphs are mandatory; do not remove them as narrative or add other preambles. Keep catalog inventory, progress summaries and execution-scope details in linked results notes. Use the specified column names, count and order in both tables; do not add `H200 GPUs tested` or any separate GPU-count column. GPU placements belong only in the specified result and runtime cells. If the operator specifies a different scope from the example below, follow that scope exactly.

**Work files are unrestricted by the report template; the report is not.** The agent may retain diagnostics, explanations, calculations and plans in its own Git-ignored work files. Copy only the specified template elements and verified cell values into the report. Do not add narrative, footnotes or sections merely because their contents are true or useful; additions require an explicit operator change to the template.

**Play each rendering correction back into the skill's literal templates, this example and the live report.** The complete first-two templates live directly in [SKILL.md](../SKILL.md#required-group-report-format). Use "test", not "check", in report prose and headings, without changing registered identifiers. Copy the title, three introductory paragraphs and legend verbatim. Use the exact headings `## Onboarding status tables and runtimes`, optional `### Possible fixes` and `### Runtimes`; do not replace them with catalog, date or group-specific headings. Runtime wording about both GAS arms applies only to the first 2 tests, `single-step-grads` and `single-step-optimizer`; do not generalize it to all tests. Put the complete sentence `Reported test times are rounded to the nearest whole minute.` immediately above each runtime table, separated only by a blank line, never a bare unit label or units in the heading. Use the capitalized label `Tally` immediately before the node-hour table.

**Do not add narrative summaries after the legend.** When needed, `Possible fixes` follows the GAS/placement explanation directly; otherwise omit the entire subsection. Do not repeat tally or catalog-inventory counts, summarize passing runs, list tensor counts or insert general scope paragraphs. Keep detailed scope and successful-run evidence in the linked results notes. Merge only current failure-relevant facts and next actions into the affected config/test's `Possible fixes` entry.

**Do not append narrative after the runtime table or tally.** Follow the literal runtime template, then the final `Add finished work` section, without paragraphs listing exact seconds, explaining combined or missing timers, summarizing attempts or discussing node accounting. Retain the supporting scalar evidence under Git-ignored `.agent-work/correctness/outcomes/` as required by [the skill](../SKILL.md#durable-outcome-evidence); detailed history belongs in linked results notes. Recover missing timers from matching records or rerun the individual test, never divide a combined timer.

## Copyable Template

**Copy the structure, not the example's counts or results.** The example below is an unrun supervised report with one eight-GPU placement and all seven full-suite columns. Its eight pending outcomes are two GAS arms for each of the first three tests, plus one inference and one end-to-end outcome; the two RL tests are inapplicable. Replace the config path, selected test columns, placements, applicability and counts with the actual selected workload; preserve the exact headings and three introductory paragraphs. No values in this example establish a real test result.

**Keep the selected scope identical in both tables.** Use the seven columns below for a full-suite group. For P1 or an explicitly first-two group, retain exactly `config`, `single-step-grads` and `single-step-optimizer` in both tables and recalculate the tally; the same unrun example then has `4(0+**0**+**0**+**0**+**0**+4+0)`. For other explicit selections, retain the selected columns in their existing order. Do not turn omitted tests into `-` cells. Do not add a separate GPU-count column; placements already appear in the cells.

**Choose one larger placement, not both 2x and 4x.** In each outcome and runtime cell, show the config's specified GPU count first, followed by at most one `\|` and one larger placement. Prefer 4x when it fits one authorized allocation; otherwise choose 2x when it fits. Omit the separator when neither fits, and never combine separate allocations. Use the same choice for both tables and both tests. With sixteen available GPUs, 1 selects 4, 2 selects 8, 4 selects 16, 8 selects 16, and 16 has no larger placement. Keep unselected measured placements in ignored outcome evidence or linked historical notes instead of adding another result to the cell.

```markdown
# Correctness Onboarding Progress

Create this file if it does not exist, or update it if it already exists. Use it to track progress when onboarding new config/test combinations.

Keep this file outside Git. It is a temporary progress tracker that each developer can use to see the current onboarding status.

Once onboarding is complete and the PR containing the new configs and `reference_data` files is published, you can safely delete this file or move it elsewhere.

## Onboarding status tables and runtimes

P+**F**+**E**+**B**+**H**+?+- 10(0+**0**+**0**+**0**+**0**+8+2)

| config | single-step-grads | single-step-optimizer | checkpoint-resume-loss | inference-checkpoint-loss | e2e-sft-train-validate | rl-weight-sync | rl-router-replay |
| ------ | ----------------- | --------------------- | ---------------------- | ------------------------- | --------------------- | -------------- | ---------------- |
| [<config-relative-path>](arctic_platform/correctness/configs/<config-relative-path>) | ??(8) | ??(8) | ??(8) | ?(8) | ?(8) | - | - |

Legend:

- `P` measured pass; `F` numerical mismatch; `E` execution error before comparison; `B` verified missing production API; `H` harness/setup work needed; `?` pending; `-` inapplicable.
- 2-letter results, like `PP` or `PF`, report 2 outcomes: one for GAS1 and the other for GAS4.
- Parenthesized numbers identify GPU placements, e.g., `P(1)` means 1 GPU, and `P(8)` means 8 GPUs.
- `|` separates the result at the config's specified GPU count from one additional result: 4x that count when it fits the allocation, otherwise 2x. Never show both. For example, `P(8) | P(16)` means the config specifies eight GPUs, and the test passed on eight GPUs and in an additional run on sixteen GPUs (2x the configured count).
- A suffix after a passing placement gives its verified passing gate in scientific notation, e.g., `PP(8) 3e2`. Different gates use `<GAS1-gate>/<GAS4-gate>` in that order; mixed results name only passing arms. No passing-gate suffix is added to pending, failure-only or inapplicable results.

### Runtimes

Test time excludes onboarding preparation, fixture preparation and tolerance calibration. Each verified single-test timer includes startup and recorded teardown, and both GAS arms for the first 2 tests (`single-step-grads` and `single-step-optimizer`).

Reported test times are rounded to the nearest whole minute.

| config | single-step-grads | single-step-optimizer | checkpoint-resume-loss | inference-checkpoint-loss | e2e-sft-train-validate | rl-weight-sync | rl-router-replay |
| ------ | ----------------- | --------------------- | ---------------------- | ------------------------- | --------------------- | -------------- | ---------------- |
| [<config-relative-path>](arctic_platform/correctness/configs/<config-relative-path>) | ?(8) | ?(8) | ?(8) | ?(8) | ?(8) | - | - |

Tally

| nodes | time in hours |
| ----- | ------------- |
| Total | 0.00 node-hours |

## Add finished work
```

## Finished Work Selection

**Include only successfully onboarded config/test combinations with compatible current identities.** A config qualifies when at least one selected test has compatible current config, source, fixture, reference and gate identities and `P` for every required gradient accumulation step (GAS) arm at every selected placement. Include only those successful tests' reference directories. `PF`, `PP(8) | ??(16)`, failures, pending or inapplicable tests and stale evidence do not qualify. Another selected test's failure does not exclude an independently successful test's config or references.

**The final section lists exact paths for manual publication only.** Replace the placeholders with actual config paths and individual `reference_data/<model>/<sequence>/<recipe>/<test-id>/` directories, including each config once. Remove unselected or unsuccessful test lines; extend the list only for other qualifying tests/configs. Never stage whole config, reference or recipe trees, the progress report, unrelated files, logs or scratch. The unrun example above has no eligible paths, so its final heading has no command block. Use the following literal section only when paths qualify; both example tests must qualify to retain both reference lines.

**Do not execute the commands while creating or updating a report.** The template does not authorize staging, committing or pushing. Before separately authorized manual execution, review the Git index so the commit contains only the intended successful artifacts.

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

## Outcomes And Tally

**Count displayed states, not imported configs or executed preparation phases.** The exact tally syntax is `P+**F**+**E**+**B**+**H**+?+- N(p+**f**+**e**+**b**+**h**+q+a)`, with the total equal to the seven category counts; `a` counts inapplicable entries. Include `-`, pending outcomes, errors and harness/setup work. Count each displayed `-` once without inventing GAS arms or placements for an inapplicable test. Bold the `F`, `E`, `B` and `H` labels and their counts, including zero counts. A completed paired test contributes two outcomes, one per GAS arm. Each placement contributes separately, so `PP(8) \| ??(16)` contributes two passes and two pending outcomes. Retries replace the current outcome at that same identity; preserve previous attempts in the notes rather than counting them again.

**Keep the cells compact and evidence-linked.** Use the illustrative forms `PP(8) 3e2`, `P**F**(8) GAS1=3e2`, `??(8)` and the established `\|` placement separator. Bold numerical failures and link measured/error cells, including their verified passing-gate suffixes, to their actual report or a results-note section that links the receipt. Link config labels to the canonical imported config using paths relative to the root report. Keep explanations outside cells, including inference-only reasons and pending repair descriptions.

**Append exact passing gates separately for each measured placement.** Use lowercase scientific `e` notation with no exponent plus sign or leading zeros; preserve every significant digit, so `0.029` is `2.9e-2` and `31.778` is `3.1778e1`, not `3.178e1`. If both GAS arms pass with the same verified gate, show it once. If their gates differ, use `PP(8) 3e-3/1e-3` in GAS1/GAS4 order, never a fabricated maximum or minimum gate. For mixed results, identify only the passing arm, as in `**F**P(8) GAS4=4e2`. Preserve named gates for tests with multiple acceptance criteria. These forms are formatting examples, not measured results. Gate suffixes do not affect tallies or runtime cells.

**Use the gate actually consumed by the linked comparison.** Match config, test, arm, placement, attempt and fixture/reference identities in the report, retained outcome or executed specification; exact per-arm comparison fields take precedence over rounded Markdown summaries. A matched specification supplies a gate only when retained comparison evidence confirms it was used. Never substitute a measured difference, calibration spread, default, newer spec or another arm's gate. Pending, failure-only, error, blocked, setup and inapplicable outcomes have no passing-gate suffix. If the gate of a recorded pass cannot be recovered, preserve the outcome without a guessed suffix and report the evidence gap. `report-runnable` readiness and reviewed gate metadata are not measured onboarding or regression passes.

**Do not invent arm outcomes from a command failure.** A startup error with no GAS-specific comparison does not prove both arms failed. Report the known test-level error and explain the arm coverage in its `Possible fixes` entry; leave independently unrun work pending. A successful onboarding validation is distinguishable from a later regression and must be identified in the linked evidence. Import, sizing, calibration, reference preparation and CPU controls are not measured target passes.

**Repeat the complete legend for each group.** Never rely on another group's legend. Keep the meanings of `P`, `F`, `E`, `B`, `H`, `?` and `-`, the exact GAS sentence, GPU parentheses and placement separators exactly as in the template. Unselected tests are outside the table, whereas `-` means inapplicable within its selected scope.

## Possible Fixes

**Include `Possible fixes` only when this group's current table contains `F`, `E`, `B` or `H`.** Cover every affected config/test pair directly after the legend and GAS/placement explanation. If several pairs share a verified cause, name every affected pair in that one entry; matching messages alone do not prove a shared cause. When no such outcomes remain, remove the entire subsection, including its heading; do not write "Nothing to fix" or a no-failures sentence.

**Revisit the subsection every time any outcome status changes.** Add or update proposals for current `F`, `E`, `B` or `H` outcomes; remove a proposal when its corresponding config/test/arm/placement outcome becomes `P`, `?` or `-`. Retain only unresolved arms or placements if other outcomes for that config/test still fail. If all outcomes are `P`, `?` or `-`, remove the entire subsection. Do not keep resolved proposals or findings outside the group's current table in this subsection. Historical failures, gate changes and successful rerun evidence remain in the linked notes.

Use this entry shape:

```markdown
### Possible fixes

- `<config-relative-path>` / `<test-id>`: [observed failure](<report-or-notes-link>) with the measured difference and current gate, or the concrete admission/runtime error. Possible cause (hypothesis): <unproved explanation>. Next action: <implementation investigation or repair>. Verification needed: <affected test, GAS arms and placements to rerun>.
```

**For a tolerance issue, give the operator actionable choices without silently changing the gate.** Explain implementation investigation and a justified, explicitly approved tolerance review as separate options. Name the exact config/test gate, the observed difference and the rerun needed after approval. A changed gate is not a pass until verified; retain the original failure. A failing gradient test must not be described as the reason an independent optimizer test cannot onboard.

## Runtime And Node-Hour Tally

**Use one verified duration per test and placement.** Report whole minutes in the same placement order as the outcome table; each of the first 2 tests, `single-step-grads` and `single-step-optimizer`, has one duration covering both GAS arms, not duplicated arm durations. Do not assume later tests have GAS1/GAS4 arms. Include startup and recorded teardown, including failed attempts. Exclude sizing, calibration, fixture construction and reference preparation. If only target validation within onboarding was timed, identify that scope explicitly rather than claiming a separate regression time.

**Unknown time stays unknown.** Use `?` or `?(8)` for an unrun or unmeasured test and plain `-` for inapplicable tests. A combined gradient-plus-optimizer batch duration is not the duration of each test; use per-test timer receipts or keep both times `?`. Do not substitute a controller's overall duration for a test timer.

**Compute hours from the displayed whole minutes once and omit unused buckets.** For each verified node count, sum its displayed test-placement minutes and divide by 60. Display only buckets with nonzero measured time, in ascending node-count order and with two decimal places; never emit unused or `0.00` node-count rows. Keep `Total` even when all tests are pending, so the unrun template has only `Total = 0.00 node-hours`. The `Total` is `sum(node_count * bucket_minutes) / 60`, rounded only for display and suffixed `node-hours`; it is not the unweighted sum of the bucket hours. Obtain node counts from the test's verified allocation/placement, not the config's original width or unrelated held fleet nodes. At eight GPUs per node, placements using up to eight GPUs on one node count as one node. Exclude unknown timings and unallocated admission rejections. These totals describe the current displayed test attempts, not cumulative fleet expenditure. Keep these calculation instructions out of the rendered report.

For example, a two-test group with gradient `P**F**(8)` and optimizer `??(8)` has tally `P+**F**+**E**+**B**+**H**+?+- 4(1+**1**+**0**+**0**+**0**+2+0)`. A verified four-minute gradient attempt on one node and an unknown optimizer time give row `1` = `0.07` and `Total` = `0.07 node-hours`; the unknown test adds no measured time. A four-minute attempt on two nodes instead gives row `2` = `0.07` and `Total` = `0.13 node-hours`, computed from unrounded minutes.

## Completion Checklist

**Run this check when creating the group, after every status change and after every completed test.**

- Both tables exist, with identical config rows, selected test columns and placement order; column names, count and order match the operator's specification exactly, with no unrequested additions or separate GPU-count column.
- The report starts with the exact `# Correctness Onboarding Progress` title and all three mandatory operator-instruction paragraphs, once per file, followed by `## Onboarding status tables and runtimes`. They remain present on every update, even when all tests pass, and the progress file remains untracked.
- The final section is exactly `## Add finished work`, once after all groups and runtime tallies. Its documentation-only Bash block lists each qualifying config explicitly and only successful selected tests' reference directories; every required GAS arm and selected placement is `P` with compatible current identities. No whole trees, report, failed/pending/inapplicable tests or unrelated artifacts are included. With no qualifying tests, the heading remains without commands; rendering never executes Git actions.
- Each test cell has at most one larger placement after `\|`, choosing 4x when it fits one authorized allocation, otherwise 2x; when neither fits, no larger placement is shown. Outcome/runtime selections match, and the tally excludes unselected placements.
- Every model or config requested for onboarding appears immediately in both tables, before discovery, preparation or tests; unrun selected outcomes and runtimes are pending, unknown placements are not invented, and repeat requests preserve measured results.
- The exact outcome-tally expression above the table recomputes from all seven displayed states, including `?` and `-`; imports and calibration do not enter it. The `F`, `E`, `B` and `H` labels and counts are bold.
- The complete legend, verbatim GAS sentence and placement explanation appear immediately below this group's outcome table; no cell contains explanatory prose.
- Every recoverable passing placement has its exact verified gate in compact scientific notation inside its evidence link; different GAS gates retain GAS1/GAS4 order and mixed results name only passing arms. No maximum/minimum substitutes, rounded gates or guessed suffixes appear, and gate suffixes do not change counts or runtime cells. Missing passing-gate evidence is reported explicitly.
- `Possible fixes` is revisited after every status change, exists only for current `F`, `E`, `B` or `H` outcomes and follows the legend and GAS/placement explanation directly, without intervening narrative summaries. It names every affected config/test pair, links evidence and states next action and required verification; proposals for outcomes now `P`, `?` or `-` are removed, and the whole subsection is absent when all outcomes are `P`, `?` or `-`. Independent tests continue.
- The optional failure subsection is named exactly `### Possible fixes`, and the runtime subsection is named exactly `### Runtimes` under `## Onboarding status tables and runtimes`; no group name or units are added. The complete sentence `Reported test times are rounded to the nearest whole minute.` appears immediately above the runtime table. The table uses whole-minute per-test timings or explicit `?`, followed by literal `Tally`, only nonzero measured node-count rows in ascending order and a correctly weighted `Total` row. Unused or `0.00` node-count rows are absent.
- Every operator rendering correction is reflected in the literal skill templates, this example and the live report.
- Every report element matches the literal template or an explicit operator change. The three introductory operator-instruction paragraphs are required, not optional narrative. No other unrequested paragraphs, footnotes or sections have been copied from agent work files, including inventory or execution-scope preambles before the tally and exact-seconds lists, missing-timer explanations or accounting summaries after the runtime tally.
- Compact outcome records, exact timings and named numerical measurements are saved before cleanup under Git-ignored `.agent-work/correctness/outcomes/`, not tracked repo files. Missing per-test timers for completed tests are recovered from matching evidence or independently remeasured; combined timers are never split.
- Updated results link actual reports and identify numerical scope and cleanup evidence in the notes; historical failures are retained, and no missing measurement is presented as a pass.
