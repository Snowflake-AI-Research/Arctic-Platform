# Correctness Tests

**Read [Getting started](../arctic_platform/correctness/docs/getting-started.md) first.** Follow the complete demo from setup and onboarding through regression, troubleshooting, staging and cleanup.

**For config import and reference locations, follow the [configuration and reference layout](../arctic_platform/correctness/docs/hf-reference-onboarding.md#configuration-and-reference-layout).** Catalog imports preserve paths below `cortex-training/model-catalog/models/`; noncatalog configs derive a model/sequence/recipe hierarchy. Reference folders mirror config-relative paths without the extension. The `config-layout.json` manifest retains existing IDs, and layout migration leaves historical measurement bytes unchanged. Import acceptance, including for sampling-only recipes, is not a numerical onboarding pass.

| Guide | Contents |
| --- | --- |
| [Getting started](../arctic_platform/correctness/docs/getting-started.md) | Complete onboarding and regression demo, troubleshooting, staging and cleanup. |
| [Running correctness tests](../arctic_platform/correctness/docs/running-tests.md) | Operator instructions for prepared compute nodes, imported config locations, running tests, and reading saved results. |
| [HF reference onboarding](../arctic_platform/correctness/docs/hf-reference-onboarding.md) | Config/reference hierarchy, frozen ID preservation, reference preparation, and review of measurement identities and tolerances. |
| [Maintained correctness code](../arctic_platform/correctness/docs/code-inventory.md) | Library layout, stable commands, and retired code. |
| [Correctness skill](../skills/correctness/SKILL.md) | Agent workflow for harness changes, reference integrity, validation and evidence boundaries. |
