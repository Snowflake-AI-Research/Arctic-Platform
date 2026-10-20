# Correctness Code Inventory

**This inventory records the approved layout migration, not numerical run results.** Paths below are relative to `arctic_platform/correctness/`. The [layout plan](code-layout-plan.md) defines the migration boundaries. This inventory was inspected on 2026-10-20; absence of an import is not evidence that an experiment is obsolete.

## Classification

**Keep preserves a supported interface or an existing dependency.** Consolidate means moving reusable behavior behind a shared interface while preserving callers and numerical rules. Archive requires retirement evidence and an index with original path, checksum, replacement and historical result. Remove requires proof of redundant behavior and no surviving import, direct command, documentation, manifest or run-owner dependency.

| Area | Decision | Dependency or migration boundary |
| --- | --- | --- |
| `__main__.py`, `api.py`, `harness/execution.py` | Consolidated | The command delegates to the supported API; shared selection and execution live in `harness/execution.py`. Existing operator commands stay available. |
| `harness/` | Keep | Shared configuration, transport, execution, reporting and source-bound record validation remain library responsibilities. |
| `checks/`, `harness/sft.py` | Keep / shared inputs consolidated | Registered numerical checks stay distinct. Shared dataset/scoring helpers now live in `harness/sft.py`, without reusing target math as reference math. |
| `reference/` | Keep | Independent reference algorithms and the existing subprocess entry remain separate from target execution. |
| `onboarding/` | Keep | Preparation and calibration stay explicit actions, not side effects of regression. |
| `models/` | Model support consolidated | `models/original.py` holds the immutable original-model metadata registry and exact offline validation; registering metadata does not grant runtime admission. |
| `configs/`, `specs/`, `reference_data/` | Keep | Reviewed workload definitions, gates and compact measurements retain their paths and identities. |
| `selftest/` | Keep | Imports and direct CPU selftest collection preserve distinct regression coverage, including dated files with current callers. |
| `diagnostics/` | Dispatch consolidated / experiments retained | The shared command lists 34 existing module entries without importing experiments. Three helper-only modules stay importable but are not executable catalog entries. |
| `docs/` | Keep | Maintainer plans and static operator instructions are deliberate documentation, not generated run receipts. |
| `archive/` | No new entries | No inspected diagnostic has sufficient dependency-retirement evidence to move safely. |

## Diagnostic Commands

**The shared dispatcher preserves each selected module's original argument handling.** Names are exact module stems. Listing does not import diagnostic runtimes; running loads only the selected entry and its own dependencies. Unknown names are rejected before execution. Exceptions and exit status remain visible. Model execution still needs the diagnostic's existing prepared environment; listing does not establish GPU capacity or supported model admission.

```bash
python -m arctic_platform.correctness.diagnostics list
python -m arctic_platform.correctness.diagnostics run lm_head_chunking \
  --config arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config \
  --test single-step-grads
```

**Direct module commands remain supported.** For example, `python -m arctic_platform.correctness.diagnostics.lm_head_chunking ...` remains available. Diagnostics with an existing DeepSpeed launcher, positional configuration, distributed environment or capture mode retain those requirements; the dispatcher does not allocate nodes or replace launchers.

**The LM-head diagnostic now customizes a run through the supported API.** `lm_head_chunking.main` uses `api.parse_run_args` and `api.run(..., executor=...)`, rather than replacing `__main__.execute`. The injected executor removes only `fused_lm_head_token_chunk_size` from copied training configurations after the regression checksum gate. The input configuration, options, reference identities and numerical gates are not rewritten by the diagnostic.

## Diagnostic Inventory

**All 37 original modules stay at their existing paths.** Each command below has an existing `if __name__ == "__main__"` entry. Similar investigation subjects do not establish duplicate algorithms or equivalent coverage. The new dispatch catalog is in [`diagnostics/__main__.py`](../diagnostics/__main__.py); the catalog's entry coverage is checked in [`check_diagnostics_dispatch.py`](../selftest/check_diagnostics_dispatch.py).

| Module in `diagnostics/` | Decision | Concrete usage or distinct purpose |
| --- | --- | --- |
| `attention_ablation.py` | Keep | Direct command; full-attention layer ablation. |
| `b200_gradient_boundary.py` | Keep | Direct capture/analysis command; imported by `b200_gradient_control` and `selftest/check_gradient_boundary.py`. |
| `b200_gradient_control.py` | Keep | Direct synchronization/capture modes; imported by `selftest/check_gradient_boundary.py`. |
| `b200_post_metric_capture.py` | Keep helper | Imported by `b200_gradient_control` and `selftest/check_gradient_boundary.py`; no standalone entry. |
| `b200_reduction_boundary.py` | Keep helper | Imported by `b200_gradient_control` and `selftest/check_gradient_boundary.py`; no standalone entry. |
| `b200_reference_capture.py` | Keep | Direct reference capture entry and imports from gradient controls/selftests. |
| `bias_source.py` | Keep | Direct command; gradient-norm bias sources. |
| `bias_vs_sp.py` | Keep | Direct command; bias versus sequence parallelism. |
| `checkpoint_resume_state.py` | Keep | Direct command; fresh trajectories and shared-prefix resume state. |
| `clip_norm_nan.py` | Keep | Direct command; non-finite global clipping norm. |
| `conv1d_kernels.py` | Keep | Direct command; short-convolution kernel comparisons. |
| `deepep_trace.py` | Keep | Direct command; imports in `selftest/check_deepep_trace.py` and `check_permutation_error_trace.py`. |
| `deepspeed_vs_plain.py` | Keep | Direct/DeepSpeed command; `loss_settings` imported by `selftest/check_deepspeed_vs_plain.py`. |
| `delta_rule_cu_seqlens.py` | Keep | Direct command; delta-rule sequence-boundary comparison. |
| `depth_threshold.py` | Keep | Direct command; depth threshold for step corruption. |
| `dp_gradient_sources.py` | Keep | Direct command; observer/accumulation helpers imported by `selftest/check_dp_gradient_sources.py`. |
| `engine_knobs.py` | Keep | Direct command; engine-setting ablations. |
| `execution_controls.py` | Keep helper | Imported by `b200_gradient_boundary` and `selftest/check_dss_driver.py`; no standalone entry. |
| `floor_decomposition.py` | Keep | Direct command; gradient spread decomposition. |
| `floor_variance.py` | Keep | Direct command; run-to-run spread and tensor size. |
| `floor_vs_depth.py` | Keep | Direct command; spread versus model depth. |
| `floor_vs_row_tokens.py` | Keep | Direct command; spread versus row length. |
| `gdn_invocation_full_model.py` | Keep | Direct command; packed gated delta net invocation within a model. |
| `gdn_varlen_vs_stock.py` | Keep | Direct command; packed versus stock gated delta net forward. |
| `ladder.py` | Keep | Direct/DeepSpeed command; incremental engine components in backward. |
| `lm_head_chunking.py` | Consolidate interface / keep path | Direct command documented in repository `docs/correctness.md`; supported injected executor replaces the command-module patch. |
| `nan_isolate.py` | Keep | Direct command; non-finite loss versus depth and row length. |
| `pretrained_vs_random.py` | Keep | Direct command; pretrained versus random-weight bias. |
| `qwen36_dp_width.py` | Keep | Direct command; gradient disagreement versus data-parallel width. |
| `qwen36_expert_grad_scaling.py` | Keep | Direct command; imports telemetry helpers; `selftest/check_qwen36_expert_grad_scaling.py` uses that module boundary. |
| `qwen36_expert_grad_telemetry.py` | Keep | Direct command and helper provider to expert-gradient scaling. |
| `qwen36_grouping_control.py` | Keep | Direct command; payload/optimizer helpers imported by `selftest/check_qwen36_grouping_control.py`. |
| `reduce_dtype_norm.py` | Keep | Direct command; precision-setting ablations. |
| `reference_mixer_packing.py` | Keep | Direct command; reference sequence-boundary packing. |
| `second_step_nan.py` | Keep | Direct command; first-step effect on the next step's weights. |
| `seqlen_ab.py` | Keep | Direct command; sequence-length comparison. |
| `step_corrupts_weights.py` | Keep | Direct command; backward versus optimizer corruption. |

## Source And Owner Dependencies

**Source manifests prevent declaring these paths disposable.** The retained Job2 plan `.codex-run-state/job-scripts/job2-normal36-retained-reload-01a141c7-20261019-v10-plan.json` includes correctness diagnostic source paths. The Job2 pending-optimizer release `job2-qwen36-normal16-pending-optimizer-controls-01a141c7-20261019-v1-release.json` binds all 37 diagnostic files in its frozen `setup-v1/sources/arctic-platform/` tree. The B200 owner snapshot `.codex-run-state/b20012-current-source-01a14270-20261019-v1/ap-source.json` also records diagnostic paths. These are retained evidence references, not proof that a current live run imports every experiment.

**Frozen copies and owner operations are outside this migration.** The inspected Job2 source/ops owner is Nietzsche (`01a141c7-569b`); the B200 snapshot owner is Peirce (`01a14270`). No snapshot, release envelope, owner script, source checksum or recorded outcome is modified. Changes to live library paths do not imply compatibility with old source-bound records. `harness/reference_records.py` explicitly records selected source contents and paths; any future relocation needs its own compatibility proof or explicit preparation.

**No archive or removal is performed in this increment.** Direct entries, selftest imports, documented usage and retained source references remain. A future retirement requires a specific replacement, equivalent retained coverage and retirement of affected path dependencies. An archive entry must record the original checksum before relocation; this inventory does not invent such a retirement or claim a historical numerical result.
