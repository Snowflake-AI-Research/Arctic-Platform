---
name: adversarial-hardening
description: >-
  Break freshly implemented or modified arctic-platform code in as many ways as
  possible: invent every constructible attack, run all of them, write repro tests
  for the ones that land, then fix and re-attack. Use after a feature or fix
  (before calling it done), when the user says "find holes / break the code /
  adversarial", or
  when hardening an SFT/RL/HTTP/Ray integration. Not confirm-it-works.
---

# Adversarial hardening

**Goal: break this code in as many ways as possible.** Assume it is wrong. Invent every constructible attack in scope and **run all of them**. A probe that re-reads last month’s contracts, plans ten attacks and executes one, or scores 46 cheap CPU calls while the live server is Residual, has failed.

Do not start until an implementation exists. Do not confirm it works. Do not use this skill’s example bugs as the claim list.

## Paths (resolve at runtime)

This clone can live anywhere and be named anything. **Never hardcode** a machine path, a clone folder name, an autorun root, or a GPU hostname in prompts, findings, or scripts.

Resolve once at the start of a pass and use the absolute paths you got:

- **Checkout root:** `git rev-parse --show-toplevel` from the workspace, or walk up from this `SKILL.md` (`…/skills/adversarial-hardening/SKILL.md` → two parents). Confirm the root contains `arctic_platform/` and `skills/`.
- **This skill:** `<checkout>/skills/adversarial-hardening/` — **commit-only** (`SKILL.md`, `probe.md`, `groups.md`, `repro.md`, `fix.md`, `checklist.md`, `multi-model.md`). Do not write findings, process audits, or other pass artifacts here.
- **Findings / grouping:** dated files at the **probe checkout root** — parent-only `PROBE_FINDINGS.<stamp>.md`, per-agent `PROBE_FINDINGS.<stamp>.<slug>.md`, merge `PROBE_FINDINGS.merge.<stamp>.md`, plus `PROBE_GROUPS.md`. Each Repro/Fix clone has its own `PROBE_REPRO_TESTS.md` and `PR_DESCRIPTION.md`. Stamp is UTC `YYYYMMDDTHHMMSSZ`. Never overwrite an existing findings file.
- **Autorun / GPU host:** `$AUTORUN_DIR` if set, else the directory that contains `enqueue.sh` (ask if missing). Host = basename of a fresh `status.<host>.txt` there — do not bake in a hostname.
- **Companion skills** (autorun, remote runtime, never-delete): follow them if they are loaded. Do not assume a home-directory path for them.

When you write a subagent prompt, substitute these resolved absolutes. Keep placeholders in this skill (`<checkout>`).

## Scope (operator, required)

Do not start probe until the operator states **code scope** and, if they care, **timeline**. Do not invent a window (“this month,” “recent work,” “the last PR”) from chat history or `git log`.

**Stay on the current checkout for Probe.** Never `git checkout main` (or any other branch) there to “get the right tree.” A feature branch already is `main` plus this branch’s commits. Repro and Fix use **copies** of this clone, each branched from `origin/main` — see [groups.md](groups.md).

Ask and wait if code scope is missing:

- **`main` / whole tree** — probe the **current working tree** (whatever HEAD is). That is `main` when you are on `main`; it is `main` + this branch when you are not. Do not switch branches. Whole-tree is **not** “skim eight famous claims” and **not** “only tensor functions.” Pick the highest-risk subsystems and go deep (see [probe.md](probe.md)). One of them must be a **live contract** (worker / HTTP / Ray / real CUDA kernel), unless the operator carves that out.
- **`branch` / this branch’s changes** — only files and hunks in `git diff main...HEAD` (merge-base with `main`). Not the rest of the tree.
- A tighter set they name (paths, a PR, a commit range).

**Timeline** is optional. A date, a commit, or “no time bound.” If they omit it, record `no time bound` and do not add one.

Write the stated scope at the top of the dated findings file. Stay inside it.

**Multi-model (optional):** a skill cannot switch this chat’s model. If the operator wants to pick models, show the numbered Task-slug list (latest of each family first, then the next tier; no `inherit`) and wait for numbers — **one or more**, any count. See [multi-model.md](multi-model.md). Runners write dated per-agent files and are left alone; the parent writes a dated merge that **unions** every Failed row (related pointers, never drop a “duplicate”). Finish with one scoreboard: Failed rows + High + wall time + live contract + status per agent. Tokens and USD columns only if a real count exists — omit them when unknown. A Task that returns Aborted / writes no file / **hits timeout** is retried **twice** (3 attempts), then hard-fails; present unused models so the operator can replace that slot. Timeout is sibling-relative (`max(45 min after first success, 3 × longest finished wall)`) plus a 15 min heartbeat — not a short global cap. Parent-only probe (no Task runners) is the default.

Three **stages**, in order. Do not skip ahead (no Fix without a Probe finding; no Fix without a Repro test that **fails** on the buggy code).

**Repro → Fix sequence:** write tests that assert the **correct** contract → they **fail** on current `main` → change the product → the **same** tests **pass**. Do not flip asserts. Do not start Fix while a repro test is green. Shipped writing follows [../good-code/SKILL.md](../good-code/SKILL.md) and [../test-writing/SKILL.md](../test-writing/SKILL.md).

| Stage | Name | Job | Read |
| --- | --- | --- | --- |
| 1 | **Probe** | Invent attacks; land or miss; classify | [probe.md](probe.md), [checklist.md](checklist.md) |
| 2 | **Repro** | Per group: tests that fail on buggy `main` | [groups.md](groups.md), [repro.md](repro.md), [../test-writing/SKILL.md](../test-writing/SKILL.md) |
| 3 | **Fix** | Same clone/branch: product fix; same tests pass; existing suite | [groups.md](groups.md), [fix.md](fix.md) |

Stop when a probe pass has **run every constructible attack in scope**, the self-gate passes, and **nothing new landed** — or after **3** full 1→2→3 iterations. Do not stop because a quota was met. Do not invent fake bugs. Invent more attacks. A static tour, or a plan with unrun attacks sitting in Residual, is incomplete — go back and run them.

```
- [ ] 0. SCOPE: operator states code scope and timeline (or no time bound). Do not switch branches.
- [ ] 1. PROBE: attack plan first; run every row; classify; write dated PROBE_FINDINGS*.md; self-gate
- [ ] 2. GROUP: assign `#N` here (not in the merge); write PROBE_GROUPS.md (one `#N` unless same contract or the same hunk); related write-ups stay until Repro; wait if a bundle is ambiguous
- [ ] 3. REPRO+FIX one group at a time: present Problem / Where / Repro plan / Fix plan (`k` or `kA`/`kB` + `#N`); clone `main`; branch `<user>/adversarial-<slug>`; write repro tests (must **fail**); then fix (same tests must **pass**); existing suite; `make format`; PR_DESCRIPTION.md on that clone
```

## Artifacts (required)

Write both at the **checkout root** of the repo being hardened. Do not commit them unless the user asks. Do not start the next stage until the file for this stage exists.

| When | File | What |
| --- | --- | --- |
| End of probe, before any test or fix | `PROBE_FINDINGS.<stamp>.md` (parent-only) or per-agent `PROBE_FINDINGS.<stamp>.<slug>.md` plus `PROBE_FINDINGS.merge.<stamp>.md` | Attack plan, then Failed, then Held, then Residual. `Recorded:` UTC on every file. Template in [probe.md](probe.md). Multi-model merge rules in [multi-model.md](multi-model.md). |
| After probe, before any clone | `PROBE_GROUPS.md` | Canonical `#N` table + tight groups (one `#N` unless same contract or the same hunk/block). Template in [groups.md](groups.md). |
| Start of Repro on a group clone | `PROBE_REPRO_TESTS.md` | Repro-test plan for **that group only**. Template in [repro.md](repro.md). |
| Each group clone | `PR_DESCRIPTION.md` | First line: `[bug fix] <description ≤67 chars if possible>` (GitHub truncates a longer subject). Then `## Summary` / `## Testing`. This branch only. |

The chat report below is a short pointer. The findings files are the detailed record. Once Repro+Fix starts, also show **one group’s** Problem / Where / Repro plan / Fix plan card in chat ([groups.md](groups.md) §4) — that card is not a dump of the merge.

Severity — stop the line on High:

- **High** — wrong results, data loss, crash, silent corruption.
- **Medium** — wrong under a specific config (multi-GPU, GAS, resume, colocate), misleading API.
- **Low** — dead code, stale docs/examples, unexported symbols.

## Report

Point at the dated findings / merge file, `PROBE_GROUPS.md`, and each group’s `PROBE_REPRO_TESTS.md`. Do not paste those files into chat. The Repro+Fix **card** (Problem / Where / Repro plan / Fix plan) is the exception — show that in chat, one group at a time.

```
## Hardening findings (iteration N)
See: PROBE_FINDINGS.merge.<stamp>.md (or PROBE_FINDINGS.<stamp>.md), PROBE_GROUPS.md
### Probe
Attacks planned: <n>  Run: <n>  Impossible: <n>  New landings: <n>  Re-filed: <n>
Checklist coverage: <each section → row ids or impossible-why>
Live contract run: <worker|HTTP|Ray|CUDA kernel|operator carved out>
- 🔴 High:   <what breaks> — <where> — <class: product|test|env|spec|flake>
- 🟡 Medium: ...
- 🟢 Low:    ...
Self-gate: <pass|incomplete — if incomplete, do not call the probe done>
### Repro / Fix (one group at a time — card in chat, then work)
- k / kA–kB — #N … — G<id> <slug> — <branch> — repro tests: <yes/no> — fix: <yes/no> — new+existing tests: <pass/fail>
Residual: <what this pass did not attack>
Result: <M fixed, K deferred>; tests: <suite> <pass/fail>; real-contract: <yes/no>
```
