# HF Reference Onboarding

**Prepare independent Hugging Face (HF) references explicitly before regression.** The two supported checks are `single-step-grads` and `single-step-optimizer`. Onboarding runs on an already prepared allocation and publishes compact JSON measurements; regression consumes those measurements without falling back to fresh HF computation.

## Create A Reviewed Specification

**A new configuration needs sizing and calibration before reference preparation.** Supply an external native configuration and a readable source checkpoint. Onboarding stores heavyweight assets and temporary calculations under `/data-fast`; review the resulting fixture, specification, arms, and tolerance before regression.

```bash
CONFIG=/data-fast/operator-inputs/model/h200/train.config
SOURCE=/data-fast/base-models/source-checkpoint
SPEC=/data-fast/operator-inputs/specs/model-h200-train.json
RECORDS="$PWD/local-correctness-records"
mkdir -p /data-fast/operator-inputs/specs

python -m arctic_platform.correctness onboard \
  --config "$CONFIG" --source-checkpoint "$SOURCE" \
  --test-id single-step-grads \
  --reference-records-dir "$RECORDS" \
  --test-spec "$SPEC" --output-dir /data-fast/tmp/model-onboarding

python -m arctic_platform.correctness onboard \
  --config "$CONFIG" --source-checkpoint "$SOURCE" \
  --test-id single-step-optimizer \
  --reference-records-dir "$RECORDS" \
  --test-spec "$SPEC" --output-dir /data-fast/tmp/model-optimizer-onboarding
```

**Add the optimizer check to the gradient specification before preparing both checks.** The second command retains the first check's inputs and adds the optimizer criterion and gate; inspect both onboarding reports and the resulting specification.

**Each onboarding command prepares its own single-check records before validating the target.** The two-check regression below needs a separate combined selection, so prepare that selection explicitly after both checks are onboarded.

**Use an existing reviewed spec for preparation-only runs.** `--prepare-references` does not resize the fixture or rewrite an existing numerical gate.

```bash
# Prepare the identity used by a regression selecting both checks.
python -m arctic_platform.correctness onboard \
  --config "$CONFIG" --test-spec "$SPEC" --prepare-references \
  --reference-tests single-step-grads single-step-optimizer \
  --reference-records-dir "$RECORDS" \
  --reference-calibration-runs 16 --reference-calibration-workers 1 \
  --output-dir /data-fast/tmp/reference-preparation
```

**Single-check and combined-check workloads have different identities.** Use `--test-id single-step-grads` or `--test-id single-step-optimizer` to prepare a single-check selection. Use both IDs with `--reference-tests` when regression selects both. Optimizer selection changes the measured step and frozen token/model settings, so separately prepared single-check records are not interchangeable with a combined selection.

## Record Identity

**Records bind measurements to the exact workload.** Identity includes configuration checksum, check and arm, model asset contents, reference source and dependency identities, adapter initialization where applicable, input tensor semantics, and check-specific settings. Records are compact JSON at `<records>/<config-id>/<check-id>/<identity-sha256>.json`; they do not contain model or optimizer tensors.

**Keep compact records in an explicitly chosen local checkout directory.** The record publisher validates the reference directory's containment; the example uses `local-correctness-records` under the repository root. No measurement directory is bundled. Pass the same `--reference-records-dir` to preparation and regression.

**Publication is immutable.** Identical records can be reused. Corrupt records, conflicting measurements, changed sources, changed assets, or changed workloads require explicit resolution and reviewed preparation; regression cannot replace or recalibrate them.

## Optimizer Criterion

**`optimizer-state-l2-norm-v1` compares scalar magnitudes.** For each canonical parameter, it compares the L2 norms of the update residual, first moment, and second moment. The residual subtracts the Adam update explained by each backend's own moments. Shape and coverage checks remain mandatory, including zero and near-zero norms. See [optimizer_state.py](../checks/optimizer_state.py).

**Calibration measures nondeterministic HF norm variation.** The prepared gate uses the largest pairwise spread across repeat captures, multiplies it by four, rounds upward to a `1e-3` increment, and applies a minimum of `1e-3`. The record retains the reviewed criterion, gate, and provenance. `--reference-calibration-workers` requires that many available reference GPUs; one worker is the conservative default.

**A norm pass is weaker than full-tensor equivalence.** Equal norms can hide directional errors. The report must not describe the scalar criterion as elementwise equality.

## LoRA Prerequisites

**Low-rank adaptation (LoRA) requires identical initial adapter values on both backends.** The deterministic helper generates a versioned CPU recipe and verifies named adapter tensor identities. Reference preparation alone does not prove that the target can load or initialize that adapter. Target binding fails explicitly when the selected public backend has no supported initialization route.

**The explicit local SAVE0 path is narrowly guarded.** Where admitted by the runner, it verifies the public step-zero adapter export before any measurement. Do not bypass its model, topology, asset, reference-record, or initialization checks to run another configuration.

## Run Regression

```bash
python -m arctic_platform.correctness run \
  --config "$CONFIG" --test-spec "$SPEC" \
  --reference-records-dir "$RECORDS" \
  --out /data-fast/tmp/prepared-regression
```

**CPU tests and preparation receipts are not numerical target passes.** Only an actual target comparison under the reviewed workload establishes a verdict. See [the operator guide](running-tests.doc) for backend prerequisites and result files.
