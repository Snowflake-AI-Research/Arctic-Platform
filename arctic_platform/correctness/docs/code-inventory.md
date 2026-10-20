# Maintained Correctness Code

**Operators use a few stable commands backed by the shared library.** Start with [running-tests.doc](running-tests.doc). Commands run on an already-prepared allocation; the library does not allocate machines or install prerequisites.

## Active Layout

Paths in this table are relative to `arctic_platform/correctness/`.

| Area | Responsibility |
| --- | --- |
| `__main__.py`, `api.py` | Stable operator command and supported Python selection/run interface. |
| `harness/` | Shared execution, input preparation, transport, configuration, reporting, and record validation. |
| `harness/sft.py` | Shared dataset locations, configured training settings, and answer-token scoring. |
| `harness/request_packing.py` | Explicit microbatch request packing, extracted unchanged from the retired experiment helper. Ordinary requests remain unsplit. |
| `models/` | Shared original-model metadata and exact validation; metadata alone does not establish runtime or capacity admission. |
| `checks/` | Registered numerical comparisons; adding a model does not require a new launcher. |
| `reference/` | Independent reference algorithms and their existing subprocess driver. |
| `onboarding/` | Explicit model/reference preparation and calibration, separate from regression. |
| `configs/`, `specs/`, `reference_data/` | Reviewed workload definitions, unchanged gates, and compact reference measurements. |
| `selftest/` | Maintained library and command regression tests. |
| `diagnostics/` | One maintained diagnostic, `lm_head_chunking`, and the shared list/run command. |
| `docs/` | Static operator guide and maintainer instructions; no generated run updates. |

## Retired Experiments

**The backup is at the repository root, outside the package.** [`correctness-backup/`](../../../correctness-backup/README.md) contains 36 retired diagnostic modules, their 11 experiment-specific test files, and two superseded layout documents. Original paths and SHA256 checksums are preserved. These files are not included in operator selections, ordinary selftests, or package builds.

**Standalone probes are no longer supported active commands.** This retirement includes the B200 gradient probes, Qwen3.6 width/grouping/telemetry/scaling experiments, DeepEP tracing, checkpoint-state investigation, engine ablations, and numerical-instability experiments. Their tests accompany the archived code; deleting those tests does not remove maintained-library coverage. The dispatcher now rejects retired names.

**Reusable behavior remains in the library with its tests.** `pack_microbatches` moved to `harness/request_packing.py`; the three existing packing controls in `selftest/check_dss_driver.py` still exercise row splitting, shifted labels, and copied processing settings. The unused `with_accumulation` configuration override is retained only in the backup.

**The documented LM-head command remains supported.** Both forms invoke the same maintained diagnostic:

```bash
python -m arctic_platform.correctness.diagnostics list
python -m arctic_platform.correctness.diagnostics run lm_head_chunking --help
python -m arctic_platform.correctness.diagnostics.lm_head_chunking --help
```

**Historical evidence is not rewritten.** Saved results, compact reference records, submitted scripts, and frozen source snapshots keep their existing paths and identities. Frozen copies of retired files remain history, not active dependencies. Archiving an investigation does not resolve its numerical failure or turn an old result into a new pass. New source-bound runs must bind their actual current file set.
