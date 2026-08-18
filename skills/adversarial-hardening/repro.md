# Stage 2 — Repro

Operate **per group** ([groups.md](groups.md)). Do not write repro tests for the whole Failed list on the probe checkout.

Turn this group’s **product** findings into tests that assert the **correct** contract. Do not fix the product in this stage.

**Sequence (do not invert):** write repro tests → **they must fail** on current `main` → only then [fix.md](fix.md) → **the same tests must pass**. Do not flip assertions. Do not start Fix while a repro test is green.

**Gates:** (1) `PROBE_GROUPS.md` exists on the probe checkout and this group is listed with canonical `#N` ids. (2) This group’s Problem / Where / Repro plan / Fix plan card is already in chat ([groups.md](groups.md) §4) — do not start the next group. The chat Repro plan is the one you write into `PROBE_REPRO_TESTS.md`. (3) This work is on the group clone, branch `<user>/adversarial-<slug>` from `origin/main`. (4) Write `PROBE_REPRO_TESTS.md` at **this clone’s** root **before** adding or editing tests. Product rows come from the dated merge (or parent-only findings) — copy the relevant `#N` Failed text in; do not edit per-agent probe files.

Follow [../test-writing/SKILL.md](../test-writing/SKILL.md) for harness, skips, seeds, ports, `gpu_serial`, and numeric parity (`rtol=0`, loss **and** `grad_norm`). This file only adds the repro-test contract and the plan format.

## Repro test

For each product finding **in this group** (cite `#N`; for a bundle, one repro test per `kA` / `kB`):

1. Write the smallest test that asserts the **CORRECT** behavior (shape, value, status code — not “today’s crash”).
2. Run it on today’s `main` code. It must **fail** (AssertionError, TypeError, wrong status, …) **because of the hole**. If it passes, the probe finding is wrong or the test does not catch it — go back to Probe; do not start Fix.
3. If it fails for a setup bug (ImportError, wrong device, bad fixture), fix the test and re-run until the failure is the product hole.
4. Env / spec-gap / flake findings do **not** get a product repro test.

Never widen `atol`/`rtol` so a same-math test looks green. Never assert “finite / positive / decreasing” when the claim is equivalence.

## Where to run

Resolve checkout and autorun from [SKILL.md](SKILL.md) Paths (this clone’s root). Do not hardcode a clone path or GPU hostname.

- CPU routing / wire / client tests here (cwd = this clone).
- Kernel / HTTP / Ray e2e / DeepSpeed: autorun on the GPU box. Local `import` / `pip list` / mtimes are the CPU node.
- Client often has `CUDA_VISIBLE_DEVICES=` empty; the server has the GPUs. A CPU-client torch fallback does not prove the CUDA kernel.

Hand the **failing** repro tests to [fix.md](fix.md) **on this same clone**. Update `PROBE_REPRO_TESTS.md` with the real test path and “repro test failed on current code” once each test has run.

## `PROBE_REPRO_TESTS.md`

One section per product row **in this group**. Do not start coding until this plan is written.

```markdown
# Probe repro plan

Group: <slug>
Branch: <user>/adversarial-<slug>
From: <dated merge> Failed (this group only: #N …)

## #N <title matching the findings row>
- Card: <k or kA>
- Where: `<file>:<line>` [, …]
- Oracle: exact | differential
- Repro test asserts (CORRECT contract): <what must be true after the fix>
- Why it fails on main today: <crash / wrong shape / wrong value>
- Test file: <tests/…/test_….py :: TestClass.test_name> (create or extend)
- Harness: TestCasePlus / pytest; require_*; gpu_serial yes/no; CPU vs autorun
- Seed / batch / config: <what must match the baseline>
- Compare: <loss and grad_norm | wire payload | …>
- Run: <pytest node id, or autorun job>
- Repro test failed on current code: <pending | yes | no — if no, back to Probe>
```
