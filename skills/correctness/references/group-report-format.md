# Required Onboarding Group Report Format

**Use this report structure for the selected seven-test scope in root `correctness-progress.md`.** The registered order is `single-step-grads`, `single-step-optimizer`, `checkpoint-resume-loss`, `inference-checkpoint-loss`, `e2e-sft-train-validate`, `rl-weight-sync`, `rl-router-replay`. The RL tests compare live training/sampling evidence, not HF reference norms. Include only selected check columns in this order in both tables. The worked example below intentionally selects the first two checks; it is not the config-only default scope. Preserve the exact title, three introductory paragraphs, headings, legend, current-failure proposals and tallies. End with exactly `## Add finished work`; retain evidence in linked notes and keep the report outside Git.

**Initialize requested work immediately, then replace pending cells with verified outcomes.** Add configs to both tables before execution. Only first-two cells use paired GAS1/GAS4 flags such as `??`; later-check cells use their actual single-result or per-arm evidence, never invented GAS pairs. Unknown timings remain `?`; use `-` only for established inapplicability. Preserve compatible measured outcomes. Imports, calibration and CPU controls are not numerical passes.

**Select one larger placement, not both 2x and 4x.** Show the configured GPU count first; add 4x when it fits one authorized allocation, otherwise 2x when it fits. If neither fits, omit the separator and additional placement. Use identical config rows, selected test columns and placement order in both tables; do not add a GPU-count column or combine allocations.

## Copyable Template

**Replace the example's paths, counts and pending outcomes with actual selected work.** The example has one eight-GPU placement and four pending GAS outcomes, not a numerical result. Keep only selected test columns; omit the entire `Possible fixes` subsection unless current `F`, `E`, `B` or `H` outcomes remain.

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
- A suffix after a passing placement gives its verified passing gate in scientific notation, e.g., `PP(8) 3e2`. Different gates use `<GAS1-gate>/<GAS4-gate>` in that order; mixed results name only passing arms. No passing-gate suffix is added to pending, failure-only or inapplicable results.

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

## Add finished work
```

## Outcomes, Failures And Runtimes

**Count every displayed outcome once per actual arm and placement.** Only first-two tests have GAS1/GAS4 outcome pairs. Use the existing tally format with its total equal to category counts; an inapplicable `-` counts once. For example, `PP(8) \| ??(16)` contributes two passes and two pending first-two outcomes. Link measured/error cells to matching reports and configs to imported paths. Startup errors without arm-specific evidence remain test-level errors, not fabricated per-arm outcomes. Keep historical failures in linked notes.

**Append exact passing gates separately for each measured placement.** Keep each gate inside the result's evidence link, immediately after its GPU parentheses, as in the illustrative cell `[PP(8) 3e2](<report-link>) \| ??(16)`. Use lowercase scientific `e` notation with no exponent plus sign or leading zeros; preserve every significant digit, so `0.029` is `2.9e-2` and `31.778` is `3.1778e1`, not `3.178e1`. If both GAS arms pass with the same verified gate, show it once. If their gates differ, use `PP(8) 3e-3/1e-3` in GAS1/GAS4 order, never a fabricated maximum or minimum gate. For mixed results, identify only the passing arm, as in `**F**P(8) GAS4=4e2`. Preserve named gates for tests with multiple acceptance criteria. These forms are formatting examples, not measured results. Gate suffixes do not affect tallies or runtime cells.

**Use the gate actually consumed by the linked comparison.** Match config, test, arm, placement, attempt and fixture/reference identities in the report, retained outcome or executed specification; exact per-arm comparison fields take precedence over rounded Markdown summaries. A matched specification supplies a gate only when retained comparison evidence confirms it was used. Never substitute a measured difference, calibration spread, default, newer spec or another arm's gate. Pending, failure-only, error, blocked, setup and inapplicable outcomes have no passing-gate suffix. If the gate of a recorded pass cannot be recovered, preserve the outcome without a guessed suffix and report the evidence gap. `report-runnable` readiness and reviewed gate metadata are not measured onboarding or regression passes.

**Put only current actionable failure proposals directly after the legend.** Name each affected config/test, link the observed error or measured difference and gate, distinguish hypotheses from evidence, and identify the repair and required verification. Remove resolved entries and omit the entire subsection when no `F`, `E`, `B` or `H` remains. A failing gradient test does not block independent optimizer onboarding; do not widen gates to turn failure into success.

```markdown
### Possible fixes

- `<config-relative-path>` / `<test-id>`: [observed failure](<report-or-notes-link>) with the measured difference and current gate, or the concrete admission/runtime error. Possible cause (hypothesis): <unproved explanation>. Next action: <implementation investigation or repair>. Verification needed: <affected test, GAS arms and placements to rerun>.
```

**Use a verified per-test timer, not a divided combined duration.** Each first-two timer covers both GAS arms, startup and recorded teardown, excluding preparation and calibration. Round exact seconds once to whole minutes; unknown timings remain `?`. Sum displayed minutes per verified node count and divide by 60, displaying only nonzero buckets in ascending order. `Total` is `sum(node_count * bucket_minutes) / 60`, displayed to two decimal places with `node-hours`; keep `Total = 0.00 node-hours` while all times are unknown. Do not append timing or execution narrative after the tally.

## Finished Work Selection

**Include only successfully onboarded config/test combinations with compatible current identities.** A config qualifies when at least one selected test has compatible current config, source, fixture, reference and gate identities and `P` for every required gradient accumulation step (GAS) arm at every selected placement. Include only those successful tests' reference directories. `PF`, `PP(8) | ??(16)`, failures, pending or inapplicable tests and stale evidence do not qualify. Another selected test's failure does not exclude an independently successful test's config or references.

**The final section lists exact paths for manual publication only.** Replace the placeholders with actual config paths and individual `reference_data/<model>/<sequence>/<recipe>/<test-id>/` directories, including each config once. Remove unselected or unsuccessful test lines. Never stage whole config, reference or recipe trees, the progress report, unrelated files, logs or scratch. The unrun example above has no eligible paths, so its final heading has no command block. Use the following literal section only when paths qualify; both example tests must qualify to retain both reference lines.

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

## Completion Checklist

- The exact title and three introductory paragraphs remain, both tables have identical selected scope and placement order, the complete legend is present, and both tallies match verified cells.
- Every recoverable passing placement has its exact verified gate in compact scientific notation inside its evidence link; different GAS gates retain GAS1/GAS4 order and mixed results name only passing arms. No maximum/minimum substitutes, rounded gates or guessed suffixes appear, and gate suffixes do not change counts or runtime cells. Missing passing-gate evidence is reported explicitly.
- Current failure proposals follow the legend directly and cover only remaining `F`, `E`, `B` or `H`; the subsection is absent when none remain.
- Runtime cells use verified whole-minute per-test durations or `?`, not split batch timers; the node-hour tally contains only measured nonzero buckets and a correctly weighted `Total`.
- The final section is exactly `## Add finished work`. Each listed config has at least one successful selected test; only its successful tests' reference directories are listed, with compatible current identities and every required GAS arm and selected placement `P`. No whole trees, report, failed/pending/inapplicable tests or unrelated artifacts are included. With no qualifying tests, keep only the heading.
- Commands are documentation only: creating or updating the report never executes Git actions or grants publication permission.
