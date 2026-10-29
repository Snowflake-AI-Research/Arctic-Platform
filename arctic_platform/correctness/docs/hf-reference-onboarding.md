# HF Reference Onboarding

**Prepare independent Hugging Face (HF) references explicitly before regression.** The two supported checks are `single-step-grads` and `single-step-optimizer`. Onboarding runs on an already prepared allocation and publishes compact JSON measurements; regression consumes those measurements without falling back to fresh HF computation.

## Create A Reviewed Specification

**Start with the complete example in the [operator guide](running-tests.md).** That example checks out this branch, resolves `axolotl-ai-co/tiny-qwen3-129m`'s weights and tokenizer through Hugging Face's existing `/checkpoint/huggingface` cache, creates a complete one-H200, 128-token training config, and defines every directory used below. Hugging Face reuses cached files and automatically downloads missing files. The example requires an exclusively allocated, prepared head node with the upstream DSS gateway installed. Correctness uses Cortex-training; onboarding manages its own local gateway when no service is configured.

**A new configuration needs only its config file as onboarding input.** Both native client configs and existing Cortex-training recipe JSON with `sub_job_configs` are accepted; pass either file directly with `--config`, without converting or flattening the recipe. The harness derives the source checkpoint from `model_name`, prepares the reduced fixture, calibrates gates, publishes individual and combined references, and validates applicable checks. The generated specification describes the fixture, input batches, applicable checks and calibrated tolerances. Compact records contain independent Hugging Face measurements, not tensor payloads. Source weights use `/checkpoint/huggingface/hub`; the demo config and final reports use the checkout's Git-ignored `./work/` directory. Generated fixtures and calibration calculations use temporary fast storage. Records must be inside the executing library's checkout, which need not be on temporary storage.

```bash
python -m arctic_platform.correctness onboard \
  --config ./work/configs/tiny-qwen3-129m-one-h200.json
```

**Default-all onboarding prepares gradients followed by optimizer state, then their combined reference selection.** Inspect `arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.json` and the adjacent `tiny-qwen3-129m-one-h200.onboarding.json` summary before accepting the gates. To select only one check, add `--test single-step-grads` or `--test single-step-optimizer`; missing fixture prerequisites are automatic. Both checks are inapplicable to RL configs.

**Repeating a failed command resumes verified completed preparation.** Recovery records under `./work/logs/onboarding-state/` bind configuration, options, source, code and dependency versions to retained artifacts. Missing or changed artifacts invalidate the affected phase and its successors; failed phases are not completion checkpoints. An optimizer validation failure can therefore retry without recalibrating its compatible references. See [diagnostics and retry](running-tests.md#onboarding) for retained reports.

**Use an existing reviewed spec for preparation-only runs.** `--prepare-references` does not resize the fixture or rewrite an existing numerical gate.

```bash
# Prepare the identity used by a regression selecting both checks.
python -m arctic_platform.correctness onboard \
  --config ./work/configs/tiny-qwen3-129m-one-h200.json \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer
```

**Single-check and combined-check workloads have different identities.** Use `--test-id single-step-grads` or `--test-id single-step-optimizer` to prepare a single-check selection. Use both IDs with `--reference-tests` when regression selects both. Optimizer selection changes the measured step and frozen token/model settings, so separately prepared single-check records are not interchangeable with a combined selection.

## Record Identity

**Records bind measurements to the exact workload.** Identity includes configuration checksum, check and arm, model asset contents, reference source and dependency identities, adapter initialization where applicable, input tensor semantics, and check-specific settings. Records use readable arm filenames; integrity hashes remain inside each compact JSON record. They do not contain model or optimizer tensors.

**Each workload folder retains only its latest complete reference set.** Publication stages and validates the complete replacement, renames the previous set aside, installs the new set, and removes the previous directory. Failed publication restores the previous complete set; regression never replaces references. The maintained layout is:

```text
<config-id>/single-step-grads/gradients-only/gas1.json
<config-id>/single-step-grads/gradients-only/gas4.json
<config-id>/single-step-grads/with-optimizer/gas1.json
<config-id>/single-step-grads/with-optimizer/gas4.json
<config-id>/single-step-optimizer/gas1.json
<config-id>/single-step-optimizer/gas4.json
```

**Compact records are generated local artifacts, not files to commit.** They default to the Git-ignored `arctic_platform/correctness/reference_data` directory. Retain them for later regression; regression cannot regenerate missing HF measurements. The publisher validates containment under the executing library's Git root, not Git tracking or merely the shell's current directory. An unrelated sibling records directory is rejected. No measurements are bundled. To override the default, pass the same `--reference-records-dir` inside the checkout to preparation and regression. Model weights and adapter tensors do not belong in this directory or in Git.

**A repeated identity cannot acquire different measurements.** Identical records can be reused. Corrupt records or conflicting measurements require explicit resolution. Changed sources, assets, or workloads require explicit onboarding and reviewed preparation; regression cannot replace or recalibrate them.

**New single-step measurements also record the implementation and configuration used by HF.** The [reference engine](../reference/hf_single_gpu.py) captures the loaded model and decoder class identities, Python `forward` code digests, source-file content hashes, and effective configurations, including the attention backend that HF configuration serialization can omit. The engine compares these observations before and after execution and rejects a changed implementation, source file or configuration. The [reference CLI](../reference/__main__.py) retains these observations in `reference_execution`; compact-record publication and loading preserve that field without rerunning HF. This limited observation does not capture every transitive kernel or prove native target execution. Historical records without this field remain historical: neither regression nor this change reconstructs their original execution evidence.

### Historical Records

**The old naming-source exception is retired.** The removed `--historical-gradient-references` options admitted only one reviewed change to the former `harness/names.py`. Consolidation into `harness/optimizer.py` changes the bound source and does not satisfy those original pins. Current regression requires an exact workload identity; use explicit onboarding to prepare compatible records. Frozen historical records and their original reports remain evidence, not compatible current references.

## Optimizer Criterion


**Each HF worker reports its arm, run name, device, completion or failure, and retained logs.** Calibration consumes completed workers immediately but retains the original run order for gate calculation. During onboarding, the [runner](../harness/runner.py) writes complete stdout, stderr and exit-code files under that attempt's `./work/logs/onboarding-*/hf-reference-*/` directory, including successful workers. The console prints their exact locations. `onboarding.log` retains console output and chained failures; `report.md` states which check failed and why. `DSS_CORRECTNESS_LOGDIR` overrides the log root. Tensor scratch remains on fast storage, separate from logs and compact records.

**`optimizer-state-l2-norm-v1` compares scalar magnitudes, not full tensors.** For each named parameter and comparison state, the criterion is `abs(target_norm - reference_norm)`. The update residual subtracts the Adam update explained by each backend's own moments before taking the norm. Parameter coverage, shapes, and finite nonnegative norms must agree; zero and near-zero norms have no exemption. See [optimizer_state.py](../checks/optimizer_state.py) and the [optimizer check](../checks/fwd_bwd_step.py).

**Equal norms can hide sign, permutation, or directional errors.** `abs(||target||2 - ||reference||2)` is weaker than the full-tensor difference `||target - reference||2`; a norm pass does not establish elementwise optimizer-state equivalence.

**A new optimizer norm record needs fresh, criterion-specific onboarding calibration.** Config-only onboarding uses eight nondeterministic single-GPU HF repeats per arm, in addition to the deterministic baseline capture, for tiny and real models alike. The [runner](../harness/runner.py) measures the largest absolute scalar-norm difference over all repeat pairs, multiplies that spread by 4, rounds upward to a `1e-3` increment, and applies a minimum absolute gate of `1e-3`. The record retains the gate and actual calibration run count. Existing records retain their original provenance and reviewed gates. No legacy full-tensor tolerance is inherited, and regression neither recalibrates nor widens this gate.

**Calibration automatically uses the visible GPUs on one node, capped at eight workers.** Each worker runs an independent single-GPU HF process. Eight samples run in one wave of eight per GAS arm when eight or more local GPUs are visible, including when sixteen GPUs are visible. Fewer local GPUs require more waves. `--reference-calibration-workers` preserves an explicit override from one to eight; the operator must make those GPUs available. Gradient and optimizer calibration use this policy independently of the target's GPU count.

**Independent process startup and optimizer capture can dominate tiny-model calibration.** Every HF repeat loads the fixture in a new process and writes optimizer-state artifacts before exiting. Eight repeats plus a deterministic baseline for each of GAS1 and GAS4 means eighteen processes for the optimizer check; concurrent workers overlap repeats, not the two GAS arms or their baselines. Progress counts successful worker completions and reports unfinished run IDs and wave elapsed time, with a waiting update every 30 seconds and no sequential ETA.

**TODO: Use every available GPU within one allocation for calibration.** Multi-node support must handle remote worker launches, node-local model and input staging, unique GPU assignments, and result and failure collection before removing the one-node limit.

## LoRA Prerequisites

**First-two reference preparation detects low-rank adaptation (LoRA) settings directly from the supplied config.** Native `training.peft` or `training.ds_worker_config.peft_config` enables automatic generation; a config without LoRA settings remains full fine-tuning. Generation uses upstream PEFT's default initializer and seed `42`, with a metadata-only base architecture and CPU adapter factors. Production PEFT code does not call a correctness initializer. Cortex-training currently has no admitted shared-adapter target-loading route, so LoRA target onboarding and regression remain unsupported. [The operator guide](running-tests.md#lora-without-adapter-inputs) distinguishes reference generation from target support.

**Compact records contain the generation recipe and content hashes, not adapter tensors.** Identity binds the dependency versions, topology, tensor shapes/dtypes and exact values. Regression reconstructs adapter factors and requires matching prepared references; it never runs HF again to accommodate a mismatch. Only explicit onboarding measures references.

**The retained internal Ray adapter transport is not a Cortex-training loading route.** That integration transfers verified adapter bytes through the allocation's object store, rather than assuming `/data-fast` is shared. Base-model files and installed code still require supported per-node distribution. Optional `--peft-adapter` overrides automatic generation for reference preparation. Dense `target_modules` with default initialization are supported; expert `target_parameters` and advanced initializers fail explicitly. The old deterministic SAVE0 execution path is retired.

## Run Regression

### Native Multi-Rank Optimizer Capture

**A local optimizer checkpoint must include every actual training rank, not just the rank-zero response from the save API.** Expert parallelism (EP) partitions experts across workers; sequence parallelism (SP) partitions a sequence across workers. The optimizer checkpoint can therefore span all configured GPUs even when the config's data-parallel width is one. The optional [runner integration](../harness/runner.py) uses `cfg.n_gpus` for native collection; ordinary capture continues to use `cfg.dp_size`.

**This integration is an internal Python callback interface, not a new CLI flag or automatic runtime discovery.** Pass `native_optimizer_scope`, `native_optimizer_scope_source`, and optionally `native_optimizer_supplier_source` to `runner.execute(...)`. A source descriptor has exactly `{"path": "/absolute/loaded/source.py", "sha256": "<full-file-sha256>"}` and must describe the callback's actual loaded source. If scope and supplier come from different files, provide both descriptors; otherwise the supplier uses the scope descriptor.

**The scope owns collection and cleanup for one target arm.** `native_optimizer_scope(session, job)` returns a context manager whose entered value is a callable `supplier(checkpoint_root, *, step)`. The [capture helper](../harness/optimizer.py) invokes that supplier after each existing save, at step zero and step one. The supplier gathers the already-saved rank files and returns the validated native layout with the actual parent, training child and worker bindings. It must not initiate another optimizer step or substitute expert sidefiles for optimizer tensors. Context exit must restore owned hooks and finish owned cleanup before the training child is released.

**Admission remains explicit and narrow.** The runner requires target execution with a loopback `CortexConfig`, `local_cortex_fixture=True`, and an explicit selection containing `single-step-optimizer` and no checks beyond the first two supervised-reference checks. Reference preparation and reinforcement-learning configs cannot use this callback. Existing model identity, parameter coverage, dtype, partition ordering, snapshot integrity and numerical gates still apply. Installing these hooks or passing CPU controls does not establish a live worker route, a complete checkpoint, successful resource cleanup or a numerical pass.


**Run regression with one command in one terminal, just like onboarding.** With no Cortex host or base URL configured, regression starts and stops its owned installed DSS gateway and selects the reduced fixture automatically. An explicitly configured endpoint is external and must already be running; the command does not stop it. Loopback database/schema segments default to `dev`. See the [Running section](running-tests.md#running) for prerequisites, diagnostics and cleanup limits.

```bash
python -m arctic_platform.correctness run \
  --config ./work/configs/tiny-qwen3-129m-one-h200.json \
  --out ./work/reports/both
```

**CPU tests and preparation receipts are not numerical target passes.** Only an actual target comparison under the reviewed workload establishes a verdict. See [the operator guide](running-tests.md) for backend prerequisites and result files.
