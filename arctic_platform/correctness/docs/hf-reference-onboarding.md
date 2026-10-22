# HF Reference Onboarding

**Prepare independent Hugging Face (HF) references explicitly before regression.** The two supported checks are `single-step-grads` and `single-step-optimizer`. Onboarding runs on an already prepared allocation and publishes compact JSON measurements; regression consumes those measurements without falling back to fresh HF computation.

## Create A Reviewed Specification

**Start with the complete example in the [operator guide](running-tests.doc).** That example checks out this branch, resolves Qwen3-8B's weights and tokenizer through Hugging Face's existing `/checkpoint/huggingface` cache, creates a complete four-H200 training config, and defines every directory used below. Hugging Face reuses cached files and automatically downloads missing files. The example requires an exclusively allocated, prepared head node with the upstream DSS gateway installed. Correctness uses Cortex-training; onboarding manages its own local gateway when no service is configured.

**A new configuration needs only its native config as onboarding input.** The harness derives the source checkpoint from `model_name`, prepares the reduced fixture, calibrates gates, publishes individual and combined references, and validates applicable checks. The generated specification describes the fixture, input batches, applicable checks and calibrated tolerances. Compact records contain independent Hugging Face measurements, not tensor payloads. Source weights use `/checkpoint/huggingface/hub`; the demo config and final reports use the checkout's Git-ignored `./work/` directory. Generated fixtures and calibration calculations use temporary fast storage. Records must be inside the executing library's checkout, which need not be on temporary storage.

```bash
python -m arctic_platform.correctness onboard \
  --config ./work/configs/qwen3-8b-four-h200.json
```

**Default-all onboarding prepares gradients followed by optimizer state, then their combined reference selection.** Inspect `arctic_platform/correctness/specs/qwen3-8b-four-h200.json` and the adjacent `qwen3-8b-four-h200.onboarding.json` summary before accepting the gates. To select only one check, add `--test single-step-grads` or `--test single-step-optimizer`; missing fixture prerequisites are automatic. Both checks are inapplicable to RL configs.

**Repeating a failed command resumes verified completed preparation.** Recovery records under `./work/logs/onboarding-state/` bind configuration, options, source, code and dependency versions to retained artifacts. Missing or changed artifacts invalidate the affected phase and its successors; failed phases are not completion checkpoints. An optimizer validation failure can therefore retry without recalibrating its compatible references. See [diagnostics and retry](running-tests.doc#onboarding) for retained reports.

**Use an existing reviewed spec for preparation-only runs.** `--prepare-references` does not resize the fixture or rewrite an existing numerical gate.

```bash
# Prepare the identity used by a regression selecting both checks.
python -m arctic_platform.correctness onboard \
  --config ./work/configs/qwen3-8b-four-h200.json \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer
```

**Single-check and combined-check workloads have different identities.** Use `--test-id single-step-grads` or `--test-id single-step-optimizer` to prepare a single-check selection. Use both IDs with `--reference-tests` when regression selects both. Optimizer selection changes the measured step and frozen token/model settings, so separately prepared single-check records are not interchangeable with a combined selection.

## Record Identity

**Records bind measurements to the exact workload.** Identity includes configuration checksum, check and arm, model asset contents, reference source and dependency identities, adapter initialization where applicable, input tensor semantics, and check-specific settings. Records are compact JSON at `<records>/<config-id>/<check-id>/<identity-sha256>.json`; they do not contain model or optimizer tensors.

**Compact records default to the Git-ignored `arctic_platform/correctness/reference_data` directory.** The publisher validates containment under the executing library's Git root, not merely the shell's current directory. An unrelated sibling records directory is rejected. No measurements are bundled. To override the default, pass the same `--reference-records-dir` inside the checkout to preparation and regression. Model weights and adapter tensors do not belong in this directory or in Git.

**Publication is immutable.** Identical records can be reused. Corrupt records, conflicting measurements, changed sources, changed assets, or changed workloads require explicit resolution and reviewed preparation; regression cannot replace or recalibrate them.

### Reviewed Historical Gradients

**New-model onboarding does not need historical-reference options.** The optional `run --historical-gradient-references <descriptor.json> --historical-gradient-references-sha256 <full-sha256>` pair is for maintainers who already hold reviewed, original gradient records. An identity mismatch alone does not authorize reuse. This branch includes no historical measurements or ready-made descriptor.

**Only the reviewed DSS shared-expert alias change is accepted.** The [loader](../harness/reference_records.py) permits `harness/names.py` to change from SHA-256 `a5c941eb9592fabe042c8f48a28bfe7e650f5b65d3b8b7a6fd3824397c63ba09` (9,905 bytes) to `4d6cd215ef4639ab2d85f5928ca2df83954012878c39366c74e29207cc92725c` (10,294 bytes). Every other workload field must match exactly, including HF sources, dependencies, model files, input tensors, initialization, config, parameter-name map and gradient tolerance. The installed naming source must match the current pin. Reuse does not exclude tensors, widen a tolerance or rerun HF measurements.

**The descriptor binds every selected arm to its original record bytes.** The JSON object has exactly two fields: `schema`, whose value is `"reviewed-gradient-references-v1"`, and `records`. Each key in `records` names a selected arm, such as `gas1` or `gas4`. Each arm has exactly `record_path` (the original JSON filename), `record_sha256` (64 lowercase hexadecimal characters hashing the entire original file) and `review` (a nonempty string identifying the source review). Relative record paths resolve against the descriptor's directory. The record filename must remain its historical workload-identity digest; renaming the record to the current lookup key is rejected. The second CLI option separately pins the descriptor's complete bytes.

**Reuse supports one explicit supervised-fine-tuning config and only `single-step-grads`.** Optimizer or mixed selections, all-config runs, reference preparation, reinforcement-learning configs, and unknown or missing arm bindings are rejected. The [runner](../harness/runner.py) writes `reference-record-provenance.json` in the run's work directory before target work, retaining both complete identities, the original record path and raw hash, and the review string. A rejected binding never falls back to another record or fresh HF computation. Historical gradient reuse does not admit optimizer references or establish a model pass.

## Optimizer Criterion

**Each HF worker reports its arm, run name, device, completion or failure, and retained logs.** Calibration consumes completed workers immediately but retains the original run order for gate calculation. During onboarding, the [runner](../harness/runner.py) writes complete stdout, stderr and exit-code files under that attempt's `./work/logs/onboarding-*/hf-reference-*/` directory, including successful workers. The console prints their exact locations. The top-level `onboarding.log` retains console output and chained failures; `report.md` states which check failed and why. `DSS_CORRECTNESS_LOGDIR` overrides the log root. Tensor scratch remains on fast storage, separate from logs and compact records.

**`optimizer-state-l2-norm-v1` compares scalar magnitudes.** For each canonical parameter, it compares the L2 norms of the update residual, first moment, and second moment. The residual subtracts the Adam update explained by each backend's own moments. Shape and coverage checks remain mandatory, including zero and near-zero norms. See [optimizer_state.py](../checks/optimizer_state.py).

**Calibration measures nondeterministic HF norm variation.** The prepared gate uses the largest pairwise spread across repeat captures, multiplies it by four, rounds upward to a `1e-3` increment, and applies a minimum of `1e-3`. The record retains the reviewed criterion, gate, and provenance. `--reference-calibration-workers` requires that many available reference GPUs; one worker is the conservative default.

**A norm pass is weaker than full-tensor equivalence.** Equal norms can hide directional errors. The report must not describe the scalar criterion as elementwise equality.

## LoRA Prerequisites

**First-two reference preparation detects low-rank adaptation (LoRA) settings directly from the supplied config.** Native `training.peft` or `training.ds_worker_config.peft_config` enables automatic generation; a config without LoRA settings remains full fine-tuning. Generation uses upstream PEFT's default initializer and seed `42`, with a metadata-only base architecture and CPU adapter factors. Production PEFT code does not call a correctness initializer. Cortex-training currently has no admitted shared-adapter target-loading route, so LoRA target onboarding and regression remain unsupported. [The operator guide](running-tests.doc#lora-without-adapter-inputs) distinguishes reference generation from target support.

**Compact records contain the generation recipe and content hashes, not adapter tensors.** Identity binds the dependency versions, topology, tensor shapes/dtypes and exact values. Regression reconstructs adapter factors and requires matching prepared references; it never runs HF again to accommodate a mismatch. Only explicit onboarding measures references.

**The retained internal Ray adapter transport is not a Cortex-training loading route.** That integration transfers verified adapter bytes through the allocation's object store, rather than assuming `/data-fast` is shared. Base-model files and installed code still require supported per-node distribution. Optional `--peft-adapter` overrides automatic generation for reference preparation. Dense `target_modules` with default initialization are supported; expert `target_parameters` and advanced initializers fail explicitly. The old deterministic SAVE0 execution path is retired.

## Run Regression

```bash
python -m arctic_platform.correctness run \
  --backend cortex-training --base-url http://127.0.0.1:8003 --local-cortex-fixture \
  --config ./work/configs/qwen3-8b-four-h200.json \
  --out ./work/reports/both
```

**CPU tests and preparation receipts are not numerical target passes.** Only an actual target comparison under the reviewed workload establishes a verdict. See [the operator guide](running-tests.doc) for backend prerequisites and result files.
