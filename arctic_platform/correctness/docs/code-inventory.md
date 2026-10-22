# Maintained Correctness Code

**Operators use a few stable commands backed by the shared library.** Start with [running-tests.md](running-tests.md). Commands run on an already-prepared allocation; the library does not allocate machines or install prerequisites.

## Active Layout

Paths in this table are relative to `arctic_platform/correctness/`.

| Area | Responsibility |
| --- | --- |
| `__main__.py`, `api.py` | Stable operator command and supported Python selection/run interface. |
| `harness/` | Shared execution, input preparation, transport, configuration, reporting, and record validation. |
| `harness/config.py` | Reviewed specifications, model settings, and placement arms. |
| `harness/storage.py` | Fast-storage paths, allocation hostfiles, and interrupted-build checks. |
| `harness/sft.py` | Shared dataset locations, configured training settings, and answer-token scoring. |
| `harness/transport.py` | Training transport contracts and explicitly requested microbatch packing. Ordinary requests remain unsplit. |
| `harness/shared_lora_adapter.py` | Upstream PEFT reference adapter generation and content verification; retained internal Ray worker loading is not an admitted Cortex-training target route. |
| `harness/lora.py` | Shared tensor-name validation and legacy E2E reference adapter recipes. Retired SAVE0 execution and its verifier are removed. |
| `harness/cortex.py`, `harness/cortex_optimizer.py` | Local gateway client, fixture admission, checkpoint transport, and optimizer capture/decoding. |
| `models/` | Shared original-model metadata and exact validation; metadata alone does not establish runtime or capacity admission. |
| `checks/` | Registered numerical comparisons; adding a model does not require a new launcher. |
| `reference/` | Independent reference algorithms and their existing subprocess driver. |
| `onboarding/` | Explicit model/reference preparation and calibration, separate from regression. |
| `configs/`, `specs/`, `reference_data/` | Reviewed workload definitions, unchanged gates, and compact reference measurements. |
| `selftest/` | Responsibility-based CPU tests and shared helpers. Extend the existing domain module; `check_lora_training.py` is the single LoRA driver. |
| `diagnostics/` | One maintained diagnostic, `lm_head_chunking`, and the shared list/run command. |
| `docs/` | Static operator guide and maintainer instructions; no generated run updates. |

## Retired Experiments

**Retired experiments are not part of the branch's supported workflow.** Optional local archives are not bundled, linked as prerequisites or included in operator selections, ordinary selftests or package builds.

**Standalone probes are no longer supported active commands.** This retirement includes the B200 gradient probes, Qwen3.6 width/grouping/telemetry/scaling experiments, DeepEP tracing, checkpoint-state investigation, engine ablations, and numerical-instability experiments. Their tests accompany the archived code; deleting those tests does not remove maintained-library coverage. The dispatcher now rejects retired names.

**Reusable behavior remains in the library with its tests.** `pack_microbatches` lives in `harness/transport.py`; packing controls in `selftest/check_runtime_clients.py` exercise row splitting, shifted labels, and copied processing settings. The removed `with_accumulation` configuration override is not part of the supported interface.

**Consolidation removes replaced modules rather than leaving forwarding files.** Maintained coverage lives in `selftest/`, not in archived drivers. Historical suite counts and receipts are run results, not guarantees about the current checkout.

**The documented LM-head command remains supported.** Both forms invoke the same maintained diagnostic:

```bash
python -m arctic_platform.correctness.diagnostics list
python -m arctic_platform.correctness.diagnostics run lm_head_chunking --help
python -m arctic_platform.correctness.diagnostics.lm_head_chunking --help
```

**Historical evidence is not rewritten.** Saved results, compact reference records, submitted scripts, and frozen source snapshots keep their existing paths and identities. Frozen copies of retired files remain history, not active dependencies. Archiving an investigation does not resolve its numerical failure or turn an old result into a new pass. New source-bound runs must bind their actual current file set.
