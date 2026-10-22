# Correctness

**This branch supports two canonical checks: `single-step-grads` and `single-step-optimizer`.** Run from a prepared allocated head. Supply a native model config to onboarding; the harness selects both applicable checks, creates a reduced-fixture specification, calibrates gates, prepares compact Hugging Face records and validates the target. The default backend is `cortex-training`; the removed `onprem` correctness backend is rejected. The harness does not allocate machines or install the GPU runtime.

- [Operator guide](../arctic_platform/correctness/docs/running-tests.doc): prerequisites, config/test selection, backend options, reports, and diagnostics.
- [HF reference onboarding](../arctic_platform/correctness/docs/hf-reference-onboarding.md): sizing, calibration, compact records, and immutable workload identities.
- [Code map](../arctic_platform/correctness/docs/code-inventory.md): retained implementation boundaries and Test2 export dependencies.
- [Correctness skill](../skills/correctness/SKILL.md): agent workflow for harness changes, reference integrity, validation and evidence boundaries; replaces the correctness-specific `AGENTS.md`.

**Regression consumes compatible prepared references and never measures a missing baseline or relaxes a reviewed gate.** Model fixtures and tensor scratch stay on fast storage; small logs, reports and retry checkpoints use `./work/logs`, and regression reports can use `./work/reports`. Hugging Face reuses the source-model cache at `/checkpoint/huggingface`. Compact reference records are generated locally and are not bundled with the package. See the operator guide's separate [Onboarding](../arctic_platform/correctness/docs/running-tests.doc#onboarding) and [Running](../arctic_platform/correctness/docs/running-tests.doc#running) sections for complete commands.
