# Retired Correctness Experiments

**These files are historical source backups, not active test programs.** The `2026-10-20/` directory preserves the original repository-relative paths and bytes of 36 diagnostic modules, 11 tests specific to those experiments, and two superseded layout documents. `2026-10-20/SHA256SUMS` records their original paths and checksums. The old documents describe the state before actual retirement; use the [current inventory](../arctic_platform/correctness/docs/code-inventory.md) for the maintained layout.

**Use the maintained library and commands instead.** Operators run `python -m arctic_platform.correctness run`; the static guide is [running-tests.doc](../arctic_platform/correctness/docs/running-tests.doc). The documented `lm_head_chunking` diagnostic remains active. The reusable `pack_microbatches` function moved from the retired `execution_controls` module to `arctic_platform.correctness.harness.request_packing`; its regression tests remain active. The unused `with_accumulation` override is retired.

**The other experiments have no replacement numerical result.** Their custom probes, ablations, runtime monkeypatches, and associated tests are intentionally retired from the maintained package. Archiving does not resolve the failures they investigated or change recorded outcomes. Historical reports, compact reference records, submitted scripts, and frozen source snapshots remain in their original locations and are not rewritten.

**This backup is excluded from active discovery and packaging.** The ordinary selftest command collects only `arctic_platform/correctness/selftest`; the package build includes only `arctic_platform`. The local pytest exclusion also prevents collection when a broader test search includes this directory. Do not add the backup to `PYTHONPATH` or use archived commands for new runs. Restore selected original paths and their dependencies deliberately before reviving an experiment.

To check the archived bytes from the repository root:

```bash
cd correctness-backup/2026-10-20
sha256sum -c SHA256SUMS
```
