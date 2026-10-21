# HF Reference Onboarding

**Prepare independent Hugging Face (HF) references explicitly before regression.** The two supported checks are `single-step-grads` and `single-step-optimizer`. Onboarding runs on an already prepared allocation and publishes compact JSON measurements; regression consumes those measurements without falling back to fresh HF computation.

## Create A Reviewed Specification

**Start with the complete example in the [operator guide](running-tests.doc).** That example checks out this branch, downloads Qwen3-8B's weights and tokenizer, creates a complete four-H200 training config, and defines every directory used below. It requires an exclusively allocated, prepared head node; on-premises target execution stops Ray services on the allocation.

**A new configuration needs sizing and calibration before reference preparation.** The specification is generated JSON describing the reduced fixture, input batches, applicable checks, and calibrated tolerances. Compact reference records contain independent Hugging Face measurements for that specification. Model weights and temporary calculations stay under `/data-fast`; records must also be under the executing library's Git checkout. The example puts the checkout itself under `/data-fast` to satisfy both requirements.

```bash
python -m arctic_platform.correctness onboard \
  --config /data-fast/correctness-p1/configs/qwen3-8b-four-h200.json \
  --source-checkpoint /data-fast/correctness-p1/models/Qwen3-8B \
  --cache-root /data-fast/correctness-p1/fixtures \
  --test-id single-step-grads \
  --test-spec /data-fast/correctness-p1/specs/qwen3-8b-four-h200.json \
  --reference-records-dir /data-fast/correctness-p1/Arctic-Platform/local-correctness-records \
  --reference-calibration-runs 16 --reference-calibration-workers 1 \
  --output-dir /data-fast/correctness-p1/reports/onboard-grads

python -m arctic_platform.correctness onboard \
  --config /data-fast/correctness-p1/configs/qwen3-8b-four-h200.json \
  --source-checkpoint /data-fast/correctness-p1/models/Qwen3-8B \
  --cache-root /data-fast/correctness-p1/fixtures \
  --test-id single-step-optimizer \
  --test-spec /data-fast/correctness-p1/specs/qwen3-8b-four-h200.json \
  --reference-records-dir /data-fast/correctness-p1/Arctic-Platform/local-correctness-records \
  --reference-calibration-runs 16 --reference-calibration-workers 1 \
  --output-dir /data-fast/correctness-p1/reports/onboard-optimizer
```

**Add the optimizer check to the gradient specification before preparing both checks.** The second command retains the first check's inputs and adds the optimizer criterion and gate; inspect both onboarding reports and the resulting specification.

**Each onboarding command prepares its own single-check records before validating the target.** The two-check regression below needs a separate combined selection, so prepare that selection explicitly after both checks are onboarded.

**Use an existing reviewed spec for preparation-only runs.** `--prepare-references` does not resize the fixture or rewrite an existing numerical gate.

```bash
# Prepare the identity used by a regression selecting both checks.
python -m arctic_platform.correctness onboard \
  --config /data-fast/correctness-p1/configs/qwen3-8b-four-h200.json \
  --test-spec /data-fast/correctness-p1/specs/qwen3-8b-four-h200.json \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer \
  --reference-records-dir /data-fast/correctness-p1/Arctic-Platform/local-correctness-records \
  --reference-calibration-runs 16 --reference-calibration-workers 1 \
  --output-dir /data-fast/correctness-p1/reports/prepare-both
```

**Single-check and combined-check workloads have different identities.** Use `--test-id single-step-grads` or `--test-id single-step-optimizer` to prepare a single-check selection. Use both IDs with `--reference-tests` when regression selects both. Optimizer selection changes the measured step and frozen token/model settings, so separately prepared single-check records are not interchangeable with a combined selection.

## Record Identity

**Records bind measurements to the exact workload.** Identity includes configuration checksum, check and arm, model asset contents, reference source and dependency identities, adapter initialization where applicable, input tensor semantics, and check-specific settings. Records are compact JSON at `<records>/<config-id>/<check-id>/<identity-sha256>.json`; they do not contain model or optimizer tensors.

**Keep compact records in an explicitly chosen local checkout directory.** The record publisher validates containment under the executing library's Git root, not merely the shell's current directory. The example uses `/data-fast/correctness-p1/Arctic-Platform/local-correctness-records`; an unrelated `/data-fast/records` directory is rejected. No measurement directory is bundled. Pass the same `--reference-records-dir` to preparation and regression.

**Publication is immutable.** Identical records can be reused. Corrupt records, conflicting measurements, changed sources, changed assets, or changed workloads require explicit resolution and reviewed preparation; regression cannot replace or recalibrate them.

### Reviewed Historical Gradients

**New-model onboarding does not need historical-reference options.** The optional `run --historical-gradient-references <descriptor.json> --historical-gradient-references-sha256 <full-sha256>` pair is for maintainers who already hold reviewed, original gradient records. An identity mismatch alone does not authorize reuse. This branch includes no historical measurements or ready-made descriptor.

**Only the reviewed DSS shared-expert alias change is accepted.** The [loader](../harness/reference_records.py) permits `harness/names.py` to change from SHA-256 `a5c941eb9592fabe042c8f48a28bfe7e650f5b65d3b8b7a6fd3824397c63ba09` (9,905 bytes) to `4d6cd215ef4639ab2d85f5928ca2df83954012878c39366c74e29207cc92725c` (10,294 bytes). Every other workload field must match exactly, including HF sources, dependencies, model files, input tensors, initialization, config, parameter-name map and gradient tolerance. The installed naming source must match the current pin. Reuse does not exclude tensors, widen a tolerance or rerun HF measurements.

**The descriptor binds every selected arm to its original record bytes.** The JSON object has exactly two fields: `schema`, whose value is `"reviewed-gradient-references-v1"`, and `records`. Each key in `records` names a selected arm, such as `gas1` or `gas4`. Each arm has exactly `record_path` (the original JSON filename), `record_sha256` (64 lowercase hexadecimal characters hashing the entire original file) and `review` (a nonempty string identifying the source review). Relative record paths resolve against the descriptor's directory. The record filename must remain its historical workload-identity digest; renaming the record to the current lookup key is rejected. The second CLI option separately pins the descriptor's complete bytes.

**Reuse supports one explicit supervised-fine-tuning config and only `single-step-grads`.** Optimizer or mixed selections, all-config runs, reference preparation, reinforcement-learning configs, and unknown or missing arm bindings are rejected. The [runner](../harness/runner.py) writes `reference-record-provenance.json` in the run's work directory before target work, retaining both complete identities, the original record path and raw hash, and the review string. A rejected binding never falls back to another record or fresh HF computation. Historical gradient reuse does not admit optimizer references or establish a model pass.

## Optimizer Criterion

**Each HF worker reports its arm, run name, device, start and completion or failure, with elapsed time.** Calibration consumes completed workers immediately but retains the original run order for gate calculation. After each subprocess exits, the [runner](../harness/runner.py) saves complete stdout, stderr and the exit code beside its `ref-<run>.json` as `ref-<run>.stdout.log`, `ref-<run>.stderr.log` and `ref-<run>.returncode`, on success as well as failure. The console prints the actual paths; logs are under the FAST work directory, not the compact-record directory. If the requested output directory is outside FAST storage, only these logs use a separate `/data-fast` work directory. These messages cover future executions and do not establish progress for an already-running worker whose source lacks this logging.

**`optimizer-state-l2-norm-v1` compares scalar magnitudes.** For each canonical parameter, it compares the L2 norms of the update residual, first moment, and second moment. The residual subtracts the Adam update explained by each backend's own moments. Shape and coverage checks remain mandatory, including zero and near-zero norms. See [optimizer_state.py](../checks/optimizer_state.py).

**Calibration measures nondeterministic HF norm variation.** The prepared gate uses the largest pairwise spread across repeat captures, multiplies it by four, rounds upward to a `1e-3` increment, and applies a minimum of `1e-3`. The record retains the reviewed criterion, gate, and provenance. `--reference-calibration-workers` requires that many available reference GPUs; one worker is the conservative default.

**A norm pass is weaker than full-tensor equivalence.** Equal norms can hide directional errors. The report must not describe the scalar criterion as elementwise equality.

## LoRA Prerequisites

**First-two on-premises low-rank adaptation (LoRA) checks generate their adapter automatically without an agent or Git tensor inputs.** The suite uses upstream PEFT's default initializer and seed `42` with a metadata-only base architecture and CPU adapter factors. The HF reference and every target worker load identical values before optimizer construction. Production PEFT code does not call a correctness initializer. [The operator guide](running-tests.doc#lora-without-adapter-inputs) contains a copyable config addition.

**Compact records contain the generation recipe and content hashes, not adapter tensors.** Identity binds the dependency versions, topology, tensor shapes/dtypes and exact values. Regression reconstructs adapter factors and requires matching prepared references; it never runs HF again to accommodate a mismatch. Only explicit onboarding measures references.

**Node-local storage does not require a shared `/data-fast` mount.** The runner sends verified adapter bytes through the allocation's existing Ray object store. Each worker writes, verifies, loads and removes a private local copy during initialization. Base-model files and installed code still require supported per-node distribution. The same optional `--peft-adapter` override must be supplied during onboarding and regression. Dense `target_modules` with default initialization are supported; expert `target_parameters` and advanced initializers fail explicitly. The old deterministic SAVE0 execution path is retired.

## Run Regression

```bash
python -m arctic_platform.correctness run \
  --config /data-fast/correctness-p1/configs/qwen3-8b-four-h200.json \
  --test-spec /data-fast/correctness-p1/specs/qwen3-8b-four-h200.json \
  --reference-records-dir /data-fast/correctness-p1/Arctic-Platform/local-correctness-records \
  --out /data-fast/correctness-p1/reports/both
```

**CPU tests and preparation receipts are not numerical target passes.** Only an actual target comparison under the reviewed workload establishes a verdict. See [the operator guide](running-tests.doc) for backend prerequisites and result files.
