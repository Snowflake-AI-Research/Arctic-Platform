# <One sentence: what is wrong, in which model, at which stage>

| | |
| --- | --- |
| Severity | critical / high / medium / low, and why |
| Model and config | `<model>`, `<dtype>`, `<context>`, `<parallelism>`, sampler `<engine args>` |
| Code | Arctic-Platform `<sha>` (main) and/or `<sha>` (release `<name>`); vLLM `<version>` |
| Stage | trainer forward / backward / optimizer / weight sync / sampler / checkpoint / platform |
| Fix | <PR link> |
| Regression check | `tests/model_audit` `<check id>` |

![explainer](diagram.svg)

## What happens

Concrete failure with numbers: what was measured, against what expectation, at which positions, layers or ranks.

## Why

The code path, with file and line references at the SHA above, and the invariant it breaks.

## Reproduce

```bash
<one command from repro/README.md>
```

Expected output with the bug, and with the fix. Hardware used. State "untested on GPU" if true.

## Fix

What the PR changes and why this is the right boundary; what it does not fix.

## Cross-review

Codex verdict (`codex-review.md`, `codex-pr-review.md`) and how each point was addressed or rebutted.
