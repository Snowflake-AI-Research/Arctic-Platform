# Correctness

**This library supports two canonical checks: `single-step-grads` and `single-step-optimizer`.** Run from an already prepared allocated head, with externally supplied configurations, reviewed specifications, and explicitly prepared Hugging Face records. The default backend is `onprem`; no fleet allocation or runtime installation is performed.

- [Operator guide](../arctic_platform/correctness/docs/running-tests.doc): prerequisites, config/test selection, backend options, reports, and diagnostics.
- [HF reference onboarding](../arctic_platform/correctness/docs/hf-reference-onboarding.md): sizing, calibration, compact records, and immutable workload identities.
- [Code map](../arctic_platform/correctness/docs/code-inventory.md): retained implementation boundaries and Test2 export dependencies.

**Regression never measures a missing reference automatically or relaxes a reviewed gate.** Model assets and tensor scratch stay under `/data-fast`; compact reference records are generated locally and are not bundled with the package.
