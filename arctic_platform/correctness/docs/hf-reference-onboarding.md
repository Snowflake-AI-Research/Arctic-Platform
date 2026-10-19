# HF Reference Onboarding (Agent Guide)

Internal preparation, regression, and provenance notes for agents maintaining the correctness harness. User-facing results remain in [the correctness report](../../../docs/correctness.md).

**All three dense-Qwen3-8B HF-backed checks pass using cached references at the original 4-GPU placement.** Both gradient arms pass all 47 norms at absolute `0.005`, and both optimizer arms pass all 141 norms under the weaker `optimizer-state-l2-norm-v1` absolute `0.001` criterion. These one-step checks use the reduced four-layer fixture, not the full configured model.

**The full-model 100-step E2E target also passes.** Final training-loss absolute difference `0.015883851796388626` and validation difference `0.0016909044929511174` are both within the frozen `0.03` gate. Corrected runner8 [handle `20261015T210855Z-3293480-000-510927488`](/code/users/stas/ai/autorun/logs/job-stas-dev-8-0-20261015T210855Z-3293480-000-510927488.log) loaded unchanged E2E `6366a1f5` and allocated zone `3086b243` on GPUs `0-3` at the earlier `21:09:50Z` snapshot.

**The E2E pass is terminal, with bounded cleanup evidence.** Its [controller](../../../.codex-run-state/dense-qwen8-fleet-stas-dev-8-0-target-e2e-sft-train-validate-1792098536308642448/controller.json) records semantic exit `0`, numerical pass, unchanged sources, no error traceback or cleanup errors, and no HF remeasurement. The owner confirms termination at `21:14:36Z`, shell exit `0`, and exact absence at `21:14:44Z`. Owned gateway/child groups are empty and reaped, but strict zone/Ray and detached verification remain `false`, not full-resource release proof; keep-alive `211452` restores the archived payload.

**The final integrated CPU suite passes with both cancellation and Torch-identity fixes.** The [JUnit receipt](/data-fast/tmp/hf-onboard-final-01a13767-20261015T2117.xml) covers 1,844 collected tests: 1,843 passed, one Ray-dependent skip, and 275 warnings in `21.98s`, with terminal exit `0`. The test environment used `DS_ACCELERATOR=cuda` and empty `CUDA_VISIBLE_DEVICES`; this is implementation coverage, not another numerical target pass. The earlier 1,804-pass suite receipt remains retained in [Progress Status](../../../docs/correctness.md#progress-status).

**The final-build runner10 gradients and optimizer both pass from cached references.** Handle `211340` ran gradients before optimizer, both GAS arms at the original 4-GPU placement. The [published gradient controller](../../../.codex-run-state/dense-qwen8-fleet-stas-dev-10-0-target-single-step-grads-1792098823201745393/controller.json) confirms both arms pass `47/47` at absolute `0.005`, with exact maximum absolute norm differences `0.0007724761962890625` and `0.0019488334655761719`; HF time is `0.0/0.0`. Semantic exit is `0`, numerical pass is `true`, sources are unchanged, and cleanup errors are empty. Owned gateway `1135005` and child `1134658` groups are empty and reaped without escalation; strict zone/Ray and detached verification remain `false`. Gradient CLI wall time remains `4m48.4s`, without a runtime-tally update.

**The optimizer controller confirms both final-build GAS passes.** After gradient completion, `211340` advanced at `21:20:11Z` to its [optimizer controller](../../../.codex-run-state/dense-qwen8-fleet-stas-dev-10-0-target-single-step-optimizer-1792099211232810332/controller.json). Both arms pass `141/141`, with zero comparisons over the weaker `optimizer-state-l2-norm-v1` absolute `0.001` gate. Exact maximum absolute norm differences are `0.000006370246410369873` and `0.000015653669834136963`, not zero; loss deltas are `6.556510925292969e-7` and `3.110617399215698e-7` for GAS1 and GAS4. Semantic exit is `0`, numerical pass is `true`, sources are unchanged, error is `null`, and cleanup errors are empty. CLI HF time is `0.0/0.0`, target time `43.5/48.0s`, and wall time `7m03.1s`; controller elapsed is `488.73929968383163s`. These distinct durations do not change the runtime tally.

**Runner10 is terminal with bounded owned-process closeout and restored keep-alive.** The keyed log reports `SEQUENTIAL_COMPLETE` and shell exit `0`; Noether confirms termination at `21:28:26Z` and fresh exact absence at `21:29:53Z`. Owned gateway `1170851` returned `-15`; gateway and child `1170518` groups are empty and reaped without escalation. Strict zone/Ray and detached verification remain `false`, not full-resource release proof. [KA10 handle `20261015T212932Z-3312979-000-344120093`](/code/users/stas/ai/autorun/logs/job-stas-dev-10-0-20261015T212932Z-3312979-000-344120093.log) restores byte-identical archive `516eaa55`, active at PID `1210403` with public heartbeat `21:29:34Z` and `gpu_touch=1.0`. All three cached-reference checks now pass, including full-model E2E; historical evidence, table slots, counts, and runtime remain unchanged. Other configs and the unsupported LoRA target path have no new validation claim. Supplied DSS gateway and DSS GPU-worker execution is not proof of the local Arctic Platform production path.

## Reference Storage

**Five active scalar JSON records total `101,972` bytes, with no binary tensor payload.** All are prepared, including the unchanged full-model 100-step E2E reference: gradients use `13,581` bytes, optimizer `30,804`, and E2E `57,587`. Four historical originals remain `44,361` bytes; all nine JSONs total `146,333` bytes. The earlier `101,948`-byte preparation total is a historical snapshot, not a new measurement.

**Runtime metadata was corrected from pinned preparation provenance, without changing measurements.** The [normalization tool](../../../.codex-run-state/analysis/normalize-hf-torch-build-records-20261015.py) completed with pinned original preparation/runtime/source/asset/config provenance, unchanged measurement checksums, and original file hashes preserved; the [root-cause note](../../../.codex-run-state/analysis/hf-torch-build-identity-20261015.md) retains the migration mapping. The parent confirms all five active records loaded through the real `load_record` with HF imports blocked and local exit `0`. Normalization and record loading establish compatibility, not a new measurement, pass, runtime, or gate change.

## Prior Attempts

**The earlier runner1 E2E attempt failed before engine initialization or training.** Its [retained startup traceback](/code/users/stas/ai/autorun/logs/job-stas-dev-1-0-20261015T204253Z-3273673-000-2569316110.log) reported `server_worker:127 -> cluster:123/322/478 -> Ray node:192/1033 -> AuthenticationError: InvalidAuthToken` despite `RAY_AUTH_MODE='disabled'`, followed by the head's GCS `ConnectionError`.

**Runner8 successor `205356` failed the reference-identity prerequisite before target initialization or comparison.** Its [full chained traceback](/code/users/stas/ai/autorun/logs/job-stas-dev-8-0-20261015T205356Z-3283394-000-165620943.log) records `e2e_sft_train_validate.py:565 run -> :800 _run -> reference_records.py:244/248 load_record -> FileNotFoundError -> ReferencePrerequisiteError: prepared HF reference missing`, requesting `3ec1443d` rather than retained E2E `6366a1f5`.

**The parent reproduced the failed key with one Torch metadata-field change.** Changing only `identity.dependencies.torch` from `2.13.0+cu130` to distribution metadata `2.13.0` gives `3ec1443d`. The current [identity implementation](../harness/reference_records.py) binds the loaded Torch build through `torch.__version__`, not varying distribution metadata. The [durable controls note](../../../.codex-run-state/analysis/hf-torch-build-identity-20261015.md) retains 126 reference-record and 61 affected CPU passes, including no-HF reuse and drift rejection, plus the earlier collection traceback and test-environment-only correction.

**Earlier failures and historical optimizer criteria remain evidence, not erased outcomes.** [Progress Status](../../../docs/correctness.md#progress-status) retains both failures, terminal cancellation and bounded cleanup receipts, and historical full-tensor optimizer passes without adding pass slots or runtime.

## Prepare Once

**Hugging Face (HF) reference measurements are prepared explicitly during onboarding, not regenerated during regression.** Use `onboard --prepare-references` with a matching reviewed SFT (supervised fine-tuning) config and frozen test specification. Preparation does not resize the fixture, rewrite historical gates, or run the target backend. Matching records are reused; missing records require HF computation, while incompatible or corrupt records must be resolved explicitly.

**The three supported check IDs select different compact measurements.**

- `single-step-grads`: named gradient L2 norms and loss for each configured arm.
- `single-step-optimizer`: step-one Adam first-moment, second-moment, and update-residual L2 norms, with canonical parameter names and shapes.
- `e2e-sft-train-validate`: 100 training losses and gradient norms, optimizer-step count, held-out validation cross-entropy, and scored-answer-token count.

**Records default to the Git-tracked location `arctic_platform/correctness/reference_data`.** Override the location with `--reference-records-dir` on both onboarding and regression. [reference_records.py](../harness/reference_records.py) stores bounded JSON at `<config-id>/<check-id>/<workload-identity-sha256>.json`, with a measurement checksum. The workload identity binds the config, reference sources and dependency versions, model asset contents, initialization, input tensors, and check-specific options rather than machine-specific cache roots. Publication is immutable: conflicting measurements cannot silently replace an existing record.

**Prepare each required check with its exact CLI ID.** These examples use the reviewed dense Qwen3-8B config; gradient and optimizer preparation use the fixtures selected by its frozen spec.

```bash
CONFIG=arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config

python -m arctic_platform.correctness onboard --config "$CONFIG" \
  --prepare-references --test-id single-step-grads

python -m arctic_platform.correctness onboard --config "$CONFIG" \
  --prepare-references --test-id single-step-optimizer \
  --reference-calibration-runs 16 --reference-calibration-workers 8

python -m arctic_platform.correctness onboard --config "$CONFIG" \
  --prepare-references --test-id e2e-sft-train-validate \
  --local-gateway-full-model
```

**Prepare gradients and optimizer together when regression selects both checks.** The prep-only `--reference-tests` option replaces the scalar `--test-id` selection and prepares both records for the combined workload; use the full check IDs, not `grads` or `optimizer` aliases. Selecting optimizer enables an optimizer step and its frozen model/token-budget settings, which also enter the gradient record's workload identity. Separate scalar preparations are therefore not interchangeable with combined preparation; regression must find records matching its actual selection.

```bash
python -m arctic_platform.correctness onboard --config "$CONFIG" \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer \
  --reference-calibration-runs 16 --reference-calibration-workers 8
```

Full-model mode admits only explicitly pinned original snapshots. [validate_local_full_model](../harness/runner.py) accepts `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218` and the original `qwen3.8-27b-b200-train-sft-8gpus-2k` source at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The B200 source must retain all 64 text layers, 27 vision layers, 18 shards and 1,199 tensors, with pinned configuration, index, and complete tensor-name/dtype/shape header hashes. Header validation does not read or hash the tensor payloads and does not establish sampling-runtime compatibility. Both sources require `HF_HUB_OFFLINE=1`, `DSS_ALLOW_REMOTE_MODELS=1`, and `HF_HUB_CACHE` pointing to the populated cache under `/data-fast`.

`--local-gateway-full-model` is restricted to E2E and inference regression; reference preparation remains E2E-only. The reference-free inference check compares training against sampling and does not need an HF record. Source admission alone does not supply a missing E2E record or compatible sampling runtime. The supplied gateway is used for target regression; these examples do not authorize gateway startup, a custom adapter, or model downloads.

## Calibrate Optimizer Norms

**`optimizer-state-l2-norm-v1` compares scalar magnitudes, not full tensors.** For each named parameter and comparison state, the criterion is `abs(target_norm - reference_norm)`. The update residual subtracts the Adam update explained by each backend's own moments before taking the norm. Parameter coverage, shapes, and finite nonnegative norms must agree; zero and near-zero norms have no exemption. See [optimizer_state.py](../checks/optimizer_state.py) and the [optimizer check](../checks/fwd_bwd_step.py).

**Equal norms can hide sign, permutation, or directional errors.** `abs(||target||2 - ||reference||2)` is weaker than the full-tensor difference `||target - reference||2`; a norm pass does not establish elementwise optimizer-state equivalence.

**A new optimizer norm record needs fresh, criterion-specific onboarding calibration.** The requested workflow uses 16 nondeterministic single-GPU HF repeats per arm, spread over 8 workers, in addition to the deterministic baseline capture. The current CLI defaults are 16 repeats and 1 worker, so request 8 workers explicitly and provide 8 available GPUs. The [runner](../harness/runner.py) measures the largest absolute scalar-norm difference over all repeat pairs, multiplies that spread by 4, rounds upward to a `1e-3` increment, and applies a minimum absolute gate of `1e-3`. The record retains the gate and calibration provenance. No legacy full-tensor tolerance is inherited, and regression neither recalibrates nor widens this gate.

## Run Regression

**Regression consumes a compatible prepared record or stops with a prerequisite error; it never falls back to a fresh HF reference run.** Regenerate a baseline only through explicit onboarding and review, not to hide a target mismatch. The record loader may hash assets and reconstruct input identities; that validation is not a reference-model training run. The [E2E check](../checks/e2e_sft_train_validate.py) loads the stored trajectory and validation cross-entropy before running the target.

```bash
python -m arctic_platform.correctness run --config "$CONFIG" \
  --test e2e-sft-train-validate --backend cortex \
  --base-url http://127.0.0.1:8003 --database dev --schema dev \
  --local-gateway-full-model --out /data-fast/tmp/qwen3-8b-e2e-report
```

**Only compact records belong in the tracked reference directory.** Model weights, adapter artifacts, optimizer captures, replay tensors, and calibration intermediates stay under `/data-fast`; use onboarding `--output-dir /data-fast/tmp/<unique-run>` when an explicit work directory is needed. Otherwise [workdir.py](../harness/workdir.py) selects `/data-fast/tmp`, with `DSS_CORRECTNESS_TMPDIR` as a fast-storage override. Reference preparation, CPU controls, and gateway admission alone do not justify new pass cells in the historical report.
