# Stage 3 — Fix

Operate **per group** on the same clone and branch as Repro ([groups.md](groups.md)). Do not start here without that group’s chat card (`k` / `kA`–`kB` + `#N`), `PROBE_REPRO_TESTS.md`, classified product findings, and repro tests that **fail** on the buggy code. Do not start another group’s Fix in the same turn.

Keep `PR_DESCRIPTION.md` at this clone’s root current for **this** group only. First line stays `[bug fix] <description>` — after the prefix, 67 characters or fewer if possible.

## Repair

- Only this group’s product defects. Do not pull in a neighboring group to “save a PR.”
- Comments and docstrings: [../good-code/SKILL.md](../good-code/SKILL.md). Tests: [../test-writing/SKILL.md](../test-writing/SKILL.md).
- Do **not** change the repro tests to match the bug, and do **not** flip their asserts. Change the product. After the fix, the **same** tests must **pass**.
- If a repro test still fails, the fix is incomplete. If it would still pass after you revert the product change, the test does not catch the hole — that is a test defect, not a done fix.
- Stay on this group’s issues. If the fix would rewrite a hunk another in-flight group already claimed, update `PROBE_GROUPS.md` (merge or defer) before editing. Same file, different block is fine.

## Tests (required)

1. Re-run the **same** repro tests. They must pass now, and fail if the fix is reverted.
2. Run the **existing** suite for the touched tree (CPU here; GPU / HTTP / Ray / DeepSpeed via autorun). Follow [../test-writing/SKILL.md](../test-writing/SKILL.md).

A green repro test with a red neighbor is not done.

## Format (required before ready)

On this clone, run **`make format`**. Do not call the group ready until it has been run and any formatter edits are in the tree. Re-run the repro tests if the formatter touched them.

`make format` does not call GitHub. `pre-commit` only does if hook repos are missing from `~/.cache/pre-commit`, or if `url.*.insteadOf` rewrites `https://github.com/` to SSH so the cache key misses. Check with `git config --show-origin --get-regexp 'url\..*insteadof'`.

**Do not edit the operator’s `~/.gitconfig`.**

If `insteadOf` is set on **this** agent HOME, override it for the format command only — do not rewrite SSH remotes in a clone’s `.git/config`:

```bash
# temp global with user.name / user.email only (no url.*.insteadOf)
GIT_CONFIG_GLOBAL=<temp-file> GIT_CONFIG_SYSTEM=/dev/null make format
```

`pre-commit`’s child `git` inherits that and uses the existing HTTPS hook cache. Commenting out `insteadOf` in `$HOME/.gitconfig` is a last resort, and only when `--show-origin` points at this agent HOME (leftover laptop `templateDir` under `/Users/` is a tell). Never comment out a config that has the operator’s `safe.directory` / `credential` store / `lfs`.

If format still hangs on a GitHub fetch, ask the operator for a GPU host and run `make format` there via autorun.

## Re-probe the fix

A fix is new code. Run [probe.md](probe.md) on **this group’s** fix diff — invent new attacks (callers/callees, same category, siblings), do not re-confirm the repro test is green. New product findings stay on this branch if they need the same hunk; otherwise they become a new group from `main`.

- GAS: a “flush the trailing partial accumulation group” fix **crashed the server** — the engine requires a fixed microbatch count. Correct fix was drop + warn. The unit with a fake client was green; the real-contract run caught it.
- `sft_ce`: the fix was one CE core for all three modes, not a looser `assertEqual`.

## Real-contract gate

Before calling this group done: one run against a real server or the real CUDA kernel when the finding is a kernel / HTTP / Ray / DeepSpeed hole (autorun — resolve root and host from [SKILL.md](SKILL.md) Paths). A green mock/unit is not enough.

Do not commit or push unless the operator asks.
