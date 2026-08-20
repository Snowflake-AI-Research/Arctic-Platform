# Repro + Fix — groups, clones, branches

After Probe, do **not** write repro tests or fixes for the whole Failed list on the probe checkout. Split product findings into groups, give each group its own clone and branch from `main`, then Repro then Fix on that clone only.

Env / spec-gap / flake rows do not get a group.

## 1. Group (write this first)

Write `PROBE_GROUPS.md` at the **probe checkout** root before copying anything. One section per group. Show it in chat and **wait** if two issues would hit the same hunk/block or if “same contract” is ambiguous — do not invent a bundle to make fewer PRs. Sharing a file is not a reason to wait or bundle.

**Default: one product issue → one group.**

Bundle into the same group only when:

- The issues are the **same contract** (one fix closes all of them), or
- The planned edits hit the **same lines / same contiguous block**. That is what merge-conflicts. Two groups may edit the same file if they touch different functions or non-overlapping hunks.

Do **not** bundle just because two issues live in one module. After grouping, no two **in-flight** branches should plan to change the same hunk. If they would, merge those groups or defer the later one until the first PR lands.

Order groups high→low. A group’s slug is a short kebab phrase for the cluster (`lm-head-squeeze`, `split-dict-chunk`, `wire-dotted-keys`).

**Do not edit the merge** to add `#N` or to drop rows. Assign `#1`, `#2`, … here (High → Medium → Low) when a row enters a group. Cite those numbers in chat, `PROBE_GROUPS.md`, and `PROBE_REPRO_TESTS.md` only.

Related merge rows (same story, different agent) may share a group when they are the **same contract** or the **same hunk**. Do not drop one as a duplicate. At Repro, one correct-contract test that fails for every row in the group means they were one hole (one fix). If a row’s test stays green after the others fail, split that row into its own group. Same file is not a reason to group.

```markdown
# Probe groups

From: <dated merge or findings file>
User: <resolved prefix, see below>

## Issues (canonical)

| # | Sev | Title | Class |
| 1 | High | <short title> | product |
| 2 | … | … | … |

## G01 <slug>
- Issues: #1
- Files (expected): <paths this PR may touch>
- Why bundled: single issue | same contract | same hunk/block
- Branch: <user>/adversarial-<slug>
- Clone: sibling of the probe checkout (`<probe-basename>-<slug>`), never a hardcoded name or absolute path
- Status: planned
```

A bundled group lists every `#N` (`Issues: #4, #9`). Never refer to an issue by title alone once numbers exist.

## 2. User prefix

Do **not** use `$USER`.

1. If `$OWNER` is set (k8s), take the part before `@`, then the first dotted word: `foo.bar@snowflake.com` → `foo`.
2. Else ask. Do not guess from chat history.

## 3. Clone and branch (one per group)

Leave the probe checkout on its current branch. **Never** `git checkout main` there.

For each group, in order:

1. Resolve the probe checkout (`git rev-parse --show-toplevel`). Parent = that directory’s parent. Sibling = `<parent>/<probe-basename>-<slug>` (or another name next to that checkout, recorded in `PROBE_GROUPS.md`). Never hardcode a clone folder or an absolute tree. Origin URL is **`git config --get remote.origin.url`** — the stored value, which is HTTPS. Do **not** use `git remote get-url` (it applies `url.*.insteadof` and turns HTTPS into SSH). **`git clone --branch main <that-https-url> <sibling>`** — no `--single-branch` (that rewrites `fetch` to only `main`). Do not `git remote set-url` after clone. Do not `cp -a`. If GitHub is unreachable, `git clone --branch main <another-local-checkout-already-on-main> <sibling>`, then copy that checkout’s `[remote "origin"]` `url` and `fetch` lines into the new clone’s `.git/config` (HTTPS + `+refs/heads/*:refs/remotes/origin/*`). Do not invent an SSH URL. Uncommitted probe files stay on the probe checkout.
2. In the **copy only**: create and switch to `<user>/adversarial-<slug>` **from that clone’s `main` / `origin/main`** (not from the probe branch). Do not edit remotes.
3. Write `PR_DESCRIPTION.md` at that copy’s root. First line is the title: `[bug fix] <short description of the bug>` — after the prefix, **67 characters or fewer** if possible (GitHub truncates a longer commit/PR subject). Then `## Summary` / `## Testing`. Keep it current for **this** group’s work only.
4. Present this group in chat (section 4), then Repro then Fix, then run tests (below). **One group at a time.** Do not present or start the next group until this one’s repro tests exist and its fix work is done or handed back. Parallel clone *directories* are fine when planned hunks do not overlap (same file is OK); do not run two groups’ Repro+Fix in the same turn.

Do not commit or push unless the operator asks.

## 4. Present one group, then work it

When Repro+Fix starts, do **not** dump the merge. Show **this group only**, then write the repro tests and fix it.

Work-unit index `k` is 1 for the first group you start, 2 for the next, and so on. Cite the canonical `#N` on every card.

**One issue in the group** (`#N`). Repro plan **before** Fix plan — the repro tests and the suites are part of the card, not an afterthought:

```markdown
### k — #N <title>
- Problem: <when this happens — inputs, config, second call, empty/all-masked, rank ≠ 0, …>
- Where: `<file>:<line>` [, `<file>:<line>` …]
- Repro plan: <correct-contract asserts; test path + node; CPU here vs autorun GPU; must fail on main today; existing suite after fix>
- Fix plan: <what you will change>
```

`Where` is the **start** of each hole (function or block). Multiple `file:line` when the hole spans files or distant areas in one file — not a dump of every touched line.

**Repro plan** must name: (1) the **correct** contract the test asserts, (2) `tests/…/file.py::Class.test_name` (create or extend), (3) where it runs (CPU pytest in this clone vs autorun for kernel / HTTP / Ray / DeepSpeed), (4) why it **fails** on current `main`, (5) which **existing** suite you re-run after the fix. Follow [../test-writing/SKILL.md](../test-writing/SKILL.md). Copy this into `PROBE_REPRO_TESTS.md` on the group clone — do not invent a second plan.

**Related issues, same group** (same contract or same hunk). One Problem / Where / Repro plan per issue (`kA`, `kB`, …). A combined fix refers to those labels:

```markdown
### kA — #4 <title>
- Problem: <circumstances>
- Where: `<file>:<line>` [, …]
- Repro plan: <correct-contract test + run + must-fail-today + existing suite for kA>

### kB — #9 <title>
- Problem: <circumstances>
- Where: `<file>:<line>` [, …]
- Repro plan: <correct-contract test + run + must-fail-today + existing suite for kB>

### Fix plan (kA–kB)
- <one plan that closes kA and kB>
```

If only some of the bundle share a fix, say `Fix plan (kA–kB)` for that subset and give `kC` its own plan. Do not collapse `kA` and `kB` into one Problem paragraph. One repro test can cover `kA`–`kB` only when it is the same contract — say so on both Repro lines.

**Worked example** (first High, one `#N`, so labels are `1` and `#1` — not a dump of the merge):

```markdown
### 1 — #1 `lm_head_logits` squeeze drops a size-1 batch when temperature ≠ 1
- Problem: SFT/RL hidden CE with `temperature != 1` and a leading dim of 1 (`[1, H]` microbatch, or a one-row ZoRRo tile shard). `lm_head_logits` squeezes dim 0 before the divide, so `[1, V]` becomes `[V]`. The caller then `view(*batch_dim)` with an empty `batch_dim` and raises `TypeError: view() got ()`. `[1, S, H]` with S>1 does not crash but silently drops the batch axis (B=1 and B=2 disagree on rank). `temperature=1` skips the branch and is fine.
- Where: `arctic_platform/common/utils/tiled_logits.py:96` (`lm_head_logits` squeeze), `arctic_platform/common/utils/tiled_logits.py:171` (`tiled_logprobs_entropy_from_hidden` `view`)
- Repro plan: Tests in `tests/common/test_tiled_logits.py::TestLmHeadTemperatureLayout` assert logits/logprobs keep the incoming rank at any temperature. They must **fail** on current `main` ((a) `[1,H]` `temperature=0.7` → `TypeError: view() got ()`; (b) `[1,S,H]` vs `[2,S,H]` at `1.5` → B=1 drops the batch axis; (c) `TiledLogProbEntropy.apply` 1-row shards at `0.7` → crash). After the fix the **same** tests pass. Existing suite: `tests/common/test_tiled_logits.py` (CPU here; GPU class via autorun). Current GPU tests use `temperature=1.0` and would miss this.
- Fix plan: Scale by temperature without changing rank. Remove `squeeze(0)`; divide in place (or broadcast `/ temperature`) so `[N, V]` / `[B, S, V]` still match the hidden layout.
```

## 5. Tests after Fix

On that clone, after the product fix (same repro tests, no flip):

1. Run the **same** repro tests. They must pass now, and fail if the fix is reverted.
2. Run the **existing** suite that covers the touched area (CPU here; GPU / HTTP / Ray / DeepSpeed via autorun). A green repro test with a broken neighbor is not done.
3. Run **`make format`** on this clone before calling the group ready — `GIT_CONFIG_GLOBAL` temp file, see [fix.md](fix.md). Do not skip to autorun or ask the operator.

Follow [../test-writing/SKILL.md](../test-writing/SKILL.md).
