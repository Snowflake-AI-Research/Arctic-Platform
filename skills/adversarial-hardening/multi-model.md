# Multi-model probe

A skill **cannot** switch the parent chat’s model (the picker is outside the agent). It **can** launch parallel Task subagents, each with an operator-chosen `model` slug, then write a **separate** dated merge report. Per-agent files are left alone.

Use this when the operator says “probe with N models”, “best-of-N probe”, or asks to pick models. They may pick **one or more** (any count ≥ 1). Do not invent a set.

## Paths and names (resolve at runtime)

Resolve the checkout once ([SKILL.md](SKILL.md) Paths). **Never hardcode** a machine path, clone folder, autorun root, or GPU hostname.

Stamp = UTC `YYYYMMDDTHHMMSSZ` at the moment that file is created (same stamp for every runner in one wave; merge uses a new stamp when it is written).

| Who | File (checkout root) | Rule |
| --- | --- | --- |
| Each runner | `PROBE_FINDINGS.<stamp>.<slug>.md` | Sanitize `/` in the slug to `-`. Write once. **Never overwrite** this file later. First line after the title: `Recorded: <ISO-8601 UTC>`. |
| Parent merge | `PROBE_FINDINGS.merge.<stamp>.md` | New file. Do not write into a per-agent file. Do not overwrite `PROBE_FINDINGS.md` or an older merge. |
| Failed / partial runner | leave whatever it wrote | Retry uses a **new** stamp (or the wave stamp + `-retryN` suffix). Do not edit the abandoned file. |

A runner does **not** write `PROBE_FINDINGS.md` or the merge file.

## Pick models by number (required)

Do not ask them to type slugs. **Enumerate** the Task tool’s **current** allowed `model` values (the `model` enum you have this turn — do not use a list cached in this file). **Drop `inherit`.** If this chat’s model has a slug on the list, show that slug like any other — do not call it inherit or pin it at the top.

**Sort — tiers, not a flat version dump:**

1. Family = vendor + model stem, stripping the version and trailing size/quality tokens (`thinking-high`, `high-fast`, `medium`, `pro`, `codex`, `fast`). Examples: `claude-opus`, `claude-sonnet`, `cursor-grok`, `gpt`, `gemini`, `composer`. `gpt-5.6-sol` and `gpt-5.6-terra` are the same family (`gpt`), same version (`5.6`).
2. Version = first `X.Y` or hyphenated `X-Y` in the slug (`4-8` → `4.8`, bare `5` → `5.0`).
3. **Tier 1** = highest version of *each* family (all ties at that version). **Tier 2** = next-highest remaining version of each family that still has one. Repeat until none left.
4. Inside a tier, sort by family name, then slug.

Number from 1 across tiers (do not restart per tier). A blank line between tiers is fine.

```
Pick models by number (space- or comma-separated). At least one. As many as they want.

1. <tier-1 slug>
…

<tier-2 slugs>
…
```

Then **stop and wait**. Accept a single number (`4`) or several (`2 5 8` / `2,5,8`). Map numbers back to slugs. If a number is out of range or the pick is empty, say so and wait. Do not substitute a nearby model. Do not require a second pick.

**Primary UI:** AskQuestion with the same numbered slugs and `allow_multiple: true` (clickable multi-select). Still print the numbered list in chat so they can reply `2 5 8` if they dismiss the form.

Scope + timeline still required ([SKILL.md](SKILL.md)). If they already named slugs, use those (no list) unless a name is not on this turn’s enum — then show the numbered list and wait.

**Time budget (optional, ask once with the pick):** default is **no absolute cap** — use the timeout rules below. Accept `45m` / `2h` / `no cap`. Do not invent a cap from chat history.

## Launch (one turn, parallel)

One message, N `Task` calls, `subagent_type: generalPurpose`. **Background them when N > 1** (or when a live-contract probe can run for hours) so the parent can watch heartbeats and enforce timeout without tying the chat to one waiter — a blocked parent turn is how the first Opus slot came back `Aborted`. Use `run_in_background: false` only for a single short parent-only-style runner, or if they asked to wait.

Each call:

- `model:` the slug for each chosen number.
- `description:` `probe <slug>`
- Isolated findings path: `<checkout>/PROBE_FINDINGS.<stamp>.<slug>.md`
- Prompt must be self-contained (subagents do not see this chat). **Resolve the checkout at runtime** and substitute that absolute path — do not copy a path from an old chat or this file.

```
Checkout (absolute, resolved this turn): <git rev-parse --show-toplevel>
Branch: stay on current; never git checkout.
Scope (operator): <whole tree | branch diff | named set>
Timeline (operator): <or no time bound>
Wave stamp (UTC): <YYYYMMDDTHHMMSSZ>
You are one of N independent probes. Do not read or edit other PROBE_FINDINGS*.md files.
Read <checkout>/skills/adversarial-hardening/SKILL.md and probe.md. Follow them. Resolve further paths from that checkout; do not hardcode machine or clone-folder paths.
Probe only. Do not write tests or change product code. Do not start Repro or Fix. Do not commit.
Write your full findings to <checkout>/PROBE_FINDINGS.<stamp>.<slug>.md only. First line after the title: Recorded: <ISO-8601 UTC matching the stamp>. Do not write PROBE_FINDINGS.md or a merge file. Do not overwrite any existing PROBE_FINDINGS*.md.
Return: path to that file, new-landing count, live-contract yes/no, self-gate pass/incomplete.
```

Parent records **`launched_at` (UTC)** for each Task in the same message as the launch. On each Task return or abort, record **`finished_at`**. Do not ask the runner to guess tokens or dollars.

Do not share an attack plan across runners. Independent invention is the point.

## Timeout (laggards and stuck runners)

Do **not** use one global “every probe dies at 15 minutes.” This wave: composer ~7 min, grok ~10 min, opus-5 Task handle ~20 min then findings ~3.7 h — and opus-5 found the most. A short fixed cap kills the useful one.

Two clocks, both required:

| Clock | Meaning | Not this |
| --- | --- | --- |
| **Heartbeat** | transcript or findings file grew in the last **15 min**, or an autorun job that runner enqueued is still running | UI spinner |
| **Grace** | still allowed to run after siblings have finished | GPU-job duration as if it were model time |

### Estimate (default, no operator cap)

1. Do nothing until **at least one** sibling has succeeded (findings file complete).
2. **Soft warn** when the first sibling succeeds: name who is still running, elapsed, last heartbeat (file mtime / last transcript line). Offer skip / wait / replace. Do not kill yet.
3. **Grace deadline** = `max(45 min after the first success, 3 × longest finished sibling wall)`. Example: first success at T+7 min, longest finished 10 min → grace ends at T+52 min (`max(7+45, 3×10)`).
4. **Hard timeout** at the grace deadline **or** after **15 min with no heartbeat**, whichever comes first. Treat that slot as failed (same retry / replace path as Aborted). Leave any partial findings file alone. Do **not** kill autorun jobs it started (they are Residual for that runner, not something to `rm`).
5. **One agent only:** no sibling grace. Heartbeat 15 min still applies. Absolute cap only if they set one.

### Operator cap

If they set `45m` / `2h`, that is a **hard** deadline from `launched_at` and wins over the sibling formula. Heartbeat 15 min still applies inside the cap. `no cap` = sibling formula + heartbeat only.

A thinking-high + live 4-GPU HTTP probe can honestly take hours. Prefer `no cap` + heartbeat unless they want a budget.

## Failed / aborted runners (retry, then replace)

A slot **succeeds** when its dated findings file exists and has Failed / Held / Residual sections. A UI spinner is not success.

A slot **fails** when the Task returns Aborted or an error, the parent turn is aborted before that Task writes a file, the return has no path, the path is missing after siblings have finished, or the **timeout** rules fire (grace deadline or 15 min no heartbeat).

**Why Aborted is common:** launching several long probes in one parent turn ties them to that turn. If the operator stops the parent (or asks a mid-wait question that aborts the waiter), unfinished Tasks come back `Aborted` and write nothing. That is a launch failure, not a slow probe. A `thinking-high` exhaustive probe can also be slow — only treat it as running if the findings file is growing or the Task is still open.

### Retry (same slug)

- Retry the **same** slug immediately. Do not replace it yet.
- **2 retries** (3 attempts total) per slug per wave.
- Each retry is a new Task with the same prompt, a **new** output path (`<stamp>-retry1`, `<stamp>-retry2`). Leave any partial file from a earlier attempt alone.
- Do not retry a slug that already succeeded.
- If the operator aborted the **parent** turn: on the next parent turn, count unfinished no-file slots as failed attempts already used, finish remaining retries for those slugs, and do **not** relaunch slugs that already wrote a complete file.

### Hard fail, then unused-model list

After 3 failed attempts on a slug: **hard-fail** that slot. Do not invent a substitute. Do not silently drop the slot from the wave.

1. Say which slug hard-failed and why (Aborted / no file / error), and how many attempts were used.
2. Build a **replacement list** from this turn’s Task `model` enum: drop `inherit`, drop every slug that already **succeeded** this wave, drop the hard-failed slug. Sort with the same tier rules as the first picker.
3. Print the numbered unused list and show AskQuestion (`allow_multiple: true`). Prompt: replace the hard-failed slot (one or more replacements allowed).
4. **Stop and wait.** Launch only the chosen replacement(s), same scope/timeline, new stamp on their files. Replacements get their own 3-attempt budget.
5. If the unused list is empty, say so and merge with the slots that succeeded. Do not reuse a hard-failed slug unless the operator types it anyway.

## Merge (parent, after successful slots finish)

**Leave every per-agent file alone.** Read them. Write a new `PROBE_FINDINGS.merge.<stamp>.md`.

| Source row | Merge |
| --- | --- |
| Failed, same `Where` / claim | **One item.** Tag `Models: a, b`. If the write-ups differ, keep both nuances under that item (`### Nuance — <slug>`), do not drop the extra detail and do not keep two top-level duplicates. |
| Failed on one, Held or Residual on others | Keep Failed. Tag `Models: a (others: held\|unattacked)`. Parent **re-runs that attack** before calling the merge done — do not drop a singleton. |
| Held on all that attacked it | Held. Tag models. |
| Held on one, unattacked on others | Residual or a parent re-run, not Held for the merge. |
| Residual | Union, with why. |

Classification: if models disagree on class (product vs spec-gap), keep the more conservative (spec-gap) unless a contract is cited. If they disagree on severity, keep the higher one and note the other.

Count **issues** as Failed rows (new landings). Dedup by the same `Where` / claim. After dedup, **number every Failed item `#1`, `#2`, …** (High → Medium → Low). That `#N` is the unique id for the rest of the pipeline — groups, chat, and Repro+Fix cite it, never a title alone. Do not reuse a number. Write **one** scoreboard at the top of the merge file **and** print it in chat (finish report — do not skip it). Hard-failed slots count as 0 issues.

### Scoreboard (issues + time, one table)

Always: Agent, Issues, High, Wall time, Live, Status.

**Tokens and USD columns only if at least one row has a known value.** If every row is unknown, omit those columns. Do not print `unknown`.

Wall time is the Task handle, not GPU-job time. `launched_at` / `finished_at` as before (transcript mtime if the parent died; report findings-file duration too if it is later). Parallel wave wall-clock = `max(finished) − min(launched)`; also note the sum of per-agent walls (they overlap). Do not derive rates (issues/hour, tokens/issue).

Live = `CUDA` / `HTTP` / `Ray` / `DeepSpeed` / `no` / `carved out`. Status = `ok` / `aborted` / `timeout` / `hard-fail`.

Tokens / USD, only when a source exists: Task usage object this turn; else `usage` / `tokenUsage` in the subagent transcript; else Admin API `POST /teams/filtered-usage-events` (`chargedCents`, may lag ~hourly); else an operator rate card **this wave** applied to **known** token counts. Never invent. Never scrape a guessed vendor price.

```
## Scoreboard
Recorded (merge): <ISO-8601 UTC>
Sources: <per-agent dated paths>
| Agent | Issues | High | Wall | Live | Status |
| <slug> | <n> | <n High> | <Xm Ys> | <CUDA|HTTP|…> | ok |
| <aborted> | 0 | 0 | <Xm Ys> | — | aborted |
| Total distinct | <identical + unique> |  |  |  |  |  |
```

Add Tokens / USD columns on the right only when known. Then:

```
Identical (same Where/claim, 2+ agents): <n>
Unique (exactly one agent): <n>
Wave wall-clock (parallel): <duration>
### Identical
- #N <title> — models: a, b
### Unique
- #N <title> — model: a
```

For one successful agent: identical = 0, unique = that agent’s count. Still print the table.

Do not add extra columns (file size, attack-plan length, guessed tokens, issues/hour). High and live are the useful extras besides wall time.

Then the parent writes the usual chat report ([SKILL.md](SKILL.md) Report) pointing at the **merge** file and the per-agent dated files, with this scoreboard above it. Repro/fix still wait for the operator.

## Do not

- Change the parent model, or tell the user to “switch the picker” as the automation.
- Launch probes without a completed number→slug pick (empty pick is not “at least one”).
- Require two or more models. One is enough.
- Overwrite or edit a per-agent findings file when merging or retrying.
- Write the merge into `PROBE_FINDINGS.md` or into a runner’s file.
- Let two runners write the same path.
- Treat “N models” as a substitute for a live-contract subsystem or the self-gate. Each runner still has to pass [probe.md](probe.md).
- Finish without the scoreboard (issues + wall, and tokens/USD only when known).
- Print a Tokens or USD column when every cell would be unknown.
- Invent token counts or dollar costs.
- After a hard-fail, skip the unused-model list or pick a replacement without the operator.
