# Stage 1 — Probe

**Break this code in as many ways as possible.** Do not write tests or change product code in this stage.

**Gates:** (1) operator has stated code scope and timeline — see SKILL.md Scope; do not guess. (2) write an **attack plan** into the dated findings file *before* you conclude. (3) finish the file (Failed first among results, Held last, Residual required) before leaving this stage. Then [repro.md](repro.md) reads the Failed product rows. Multi-model: write only your `PROBE_FINDINGS.<stamp>.<slug>.md`; the parent merges separately ([multi-model.md](multi-model.md)).

Walk [checklist.md](checklist.md) as **attack recipes**, only for surfaces you actually chose. Do not dump the list. Do not use this file’s sibling examples as the claim list.

## This is not a probe

Stop and restart the pass if you are doing any of these:

- Re-reading last month’s fixes (`sft_ce` `#69`, `colocate` / `cuda_ipc` fallback, `sub_job_id: int | str`, GAS length check, ZeRO gather) and marking them Held.
- “Tried: read the function / grep / looked at the annotation / `inspect.getsource`.” That is Residual, not Held **and not Failed-executed**.
- Walking the sibling list in this skill as if it were the plan.
- Whole-tree scope → eight shallow claims from memory, no constructed input, or a long plan with one executed check.
- Leading with green. Confirming `#69` is not a finding.

Known-bug categories exist so that **after a new product finding** you search siblings. They are not the menu.

## Attack plan (write this first)

Decompose the change (or, for whole-tree, the subsystems you picked) into **claims** — what the code asserts. Then, for each claim, write the **attack** before you trace:

- A concrete counterexample: input, second call, empty/all-masked batch, `world_size>1`, resume, retry, omitted field, wrong type, `pack=True`, rank ≠ 0, path the tests never enter.
- The oracle that would prove the break: **exact** (`assertEqual`, accept/reject) or **differential** (two modes/transports/kernels, same seed/batch/config). “The line ran,” “no crash,” and `grad_norm > 0` are not oracles for a same-math claim.

A generic edge-case dump is not a plan. “I will read `processor.py`” is not an attack.

### Whole-tree / no time bound

Do not skim the famous contracts. Pick **2–3 high-risk subsystems** by blast radius. **One must be a live contract** (DeepSpeed worker, HTTP or Ray server, or the real CUDA kernel) unless the operator carves that out. “RL tensor losses + SFT labels + Pydantic” with the worker in Residual is a failed pick.

Inside those, be **exhaustive**: every checklist recipe that can hit the surface, every path tests skip, every sibling after a landing. Depth is not “forty CPU calls and stop.”

### Branch / named set

Attacks come from the diff and its callers/callees, not from this skill’s examples.

## How to attack

Prefer the path that can lie:

1. **What the tests do not cover.** Read the test. Name the input or call it never makes. That is the first attack.
2. **Second time / empty / error / resume / rank ≠ 0.** Happy-path once on rank 0 is the tour.
3. **Lie in the comment.** If a docstring, Cortex-parity note, or `# gathered` disagrees with the next ten lines, believe the lines and construct the call that hits them.
4. **Two implementations of one claim.** Different kernel, transport, client, or fallback. Same seed/batch/config; compare the thing the claim is about (loss **and** backward, or the exact payload).
5. **Execute every attack.** CPU pytest here (cwd = resolved checkout). Kernel / HTTP / Ray / DeepSpeed via autorun on the GPU box — resolve the autorun root and host at runtime ([SKILL.md](SKILL.md) Paths). Local `import` / `pip list` / mtimes are the CPU node. `inspect.getsource`, comment grep, and “the docstring says unused” are **reads**, not runs. A GPU/HTTP/Ray claim run only as a CPU helper is incomplete — run the live path or mark Residual-impossible **after** a runtime probe (`import ray`, `torch.cuda.is_available()` on the worker), not after one ImportError.

Construct the counterexample in the findings row: the request body, the batch shape, the kwargs, the branch that would `TypeError`.

## Exhaustive (not a quota)

The job is **as many breaks as you can find**, not “enough attacks to look serious.”

- Write the full attack plan first (stable ids `A01`…). Then **run every row** that can run. Isolate rows: one failure must not abort the plan. The probe is incomplete until the log has a result line per id.
- Residual is only for attacks that are **impossible this pass** (autorun down, operator deferred, runtime probe showed the stack missing). “I skipped it” is not Residual.
- After a landing, grep the category and **run** each sibling. “Would same-path” is not a run. Table: site → ran → saw.
- Walk every [checklist.md](checklist.md) recipe that the chosen surfaces can hit. Put a **coverage table** in the findings (section → row ids or impossible-why). Empty / second call / error / resume / rank ≠ 0 / path tests skip are not optional samples.
- For a GPU/kernel/HTTP/Ray claim: the real contract (autorun), not a CPU-client torch fallback.
- Do not re-file a hole already in a prior iteration’s Failed (or operator-deferred) as a new landing. Put it under **Still broken (already filed)**. `New landings` in the report counts only new rows.
- Keep inventing attacks until a pass adds none that you can still construct and run.

Invent attacks, not bugs. Empty Failed + every planned attack run and missed is a valid hard pass. Empty Failed + unrun rows is invalid.

## Classify each landing

A story without a concrete scenario is a hypothesis — keep it in Residual until you attack it.

- **Product** — implementation violates a **supported** contract (doc, test, or advertised API). Goes to Repro and Fix. If you invented the promise (“NaN must raise”), that is spec-gap.
- **Test** — wrong setup, subject, or expected result. Fix the test, not the product.
- **Env / harness** — xdist isolation, stale GPU worker, port collision. Not a product bug until it reproduces serially / on the real runtime.
- **Spec gap** — no authority defines the expected outcome. Surface the counterexample; do not guess.
- **Flake** — same code and config both pass and fail. Control time/seed/order; do not add sleeps or retries.

“I did not attack X” is honest Residual. “There is no X” requires saying where you attacked.

## After a landing — search siblings

Name the **category** and grep for other instances. Do not stop at the first site. Do not start the pass here.

- Same-math, different kernel — RL `compute_entropy_and_logprobs` and ZoRRo `logits_optimization` (those already share the tiled core; a finite/nonzero grad check is not a cross-mode oracle).
- Remainder that invents a short last shard (GAS flush).
- Type coercion on the wire (`int`≠`str`).
- Advertised no-op; launch state sent per call (`colocate`).
- Rank-local treated as global (ZeRO shard).

## Self-gate (before you call probe done)

If any box is unchecked, do not write Held for those claims — move them to Residual and keep attacking.

- [ ] Every Held **Tried** is an attack that **ran**. Not read / grep / `inspect.getsource`.
- [ ] Same-math Held used a real oracle (loss **and** backward, large vocab if that is the claim) — not `maxabs=0` on a toy forward.
- [ ] The claim list is not this skill’s example bugs unless the operator named those surfaces.
- [ ] Every plan id has a result line. Residual has only impossible rows, each with a runtime probe or operator deferral.
- [ ] Each Failed row has a sibling table (site → ran → saw), not “would.”
- [ ] Coverage table present. Whole-tree includes a live-contract subsystem (or operator carve-out).
- [ ] New landings are deduped against prior Failed / deferred rows.
- [ ] A skeptical reviewer would not call this a quota tour, a source-grep, or a code-reading tour.

If you cannot check a box, the probe is **incomplete**. Do not write `Self-gate: pass`.

## Dated findings file

Write `PROBE_FINDINGS.<stamp>.md` (parent-only) or `PROBE_FINDINGS.<stamp>.<slug>.md` (multi-model runner). Stamp is UTC `YYYYMMDDTHHMMSSZ`. **Do not overwrite** an existing findings file. Add `Recorded: <ISO-8601 UTC>` immediately under the title so later merges can point at a moment in time.

Be specific: files and lines, the counterexample, what you ran, what you expected, what you saw. “Looks fine” is not a row. Include the coverage table and, for each Failed product/spec row, the sibling-ran table.

```markdown
# Probe findings

Recorded: <ISO-8601 UTC>
Scope (operator): <whole tree on this checkout (main + branch if not on main) | this branch’s diff vs main | named set>
Checkout: <branch; absolute path resolved this pass; do not switch>
Timeline (operator): <date/commit | no time bound>
Subsystems (if whole-tree): <2–3 and why high-risk>
Claims in scope: <list>
Oracle named per claim: <exact | differential>

## Attack plan

### <claim>
- Attack: <concrete counterexample — input / call / config / path tests skip>
- Oracle: <exact | differential — what would prove the break>
- Execute: <pytest node | autorun | trace-only and why execute is impossible>

## Failed

What broke, or what the claim does not hold. Product rows go to `PROBE_REPRO_TESTS.md`.

**Number every Failed row** `#1`, `#2`, … after the list is final (High → Medium → Low). That `#N` is the unique id for grouping and Repro+Fix. Do not reuse a number. Multi-model runners may use local headings; the parent merge (or this file, if parent-only) assigns the canonical `#N`.

### #N <short title>
- Class: product | test | env | spec-gap | flake
- Severity: High | Medium | Low
- Claim: <the assertion that failed>
- Where: <`file:line` of the start of each area; multiple if the hole spans files or distant blocks>
- Attack: <the counterexample you used>
- Expected: <oracle>
- Saw: <actual>
- Category / siblings: <pattern → other sites or none>

## Held

Only claims you **attacked** that survived. Last on purpose — do not lead with green.

### <short title>
- Claim: <the assertion>
- Attack: <the counterexample>
- Where: <files / paths>
- Execute: <what ran>
- Why it held: <the attack missed — evidence, not “matches the claim”>

## Residual

Only attacks that could not run this pass, each with why (impossible, not “skipped”). Surfaces outside the chosen subsystems go here as out-of-scope, not as unrun plan rows.
```
