# HF Reference Onboarding

**Config-only onboarding selects all applicable checks from this checkout's seven-test registry.** `single-step-grads` and `single-step-optimizer` use the HF scalar/norm records described below; `e2e-sft-train-validate` uses its own ordered-trajectory HF record. `checkpoint-resume-loss` compares AP uninterrupted and resumed training, and `inference-checkpoint-loss` compares training and sampling losses for the saved weights. Neither is admitted by a first-two HF record. The RL tests compare live training/sampling evidence, not HF reference norms. Onboarding prepares compatible selected references automatically; regression never recomputes missing references. See [check-specific prerequisites](running-tests.md#check-specific-prerequisites).

## Configuration And Reference Layout

**Catalog imports preserve the entire recipe path below `cortex-training/model-catalog/models/`.** Replace that prefix with `arctic_platform/correctness/configs/`, keeping every remaining directory and the filename exactly. For example, `cortex-training/model-catalog/models/qwen3.5-2b/standard/h200-inference-lora.json` imports as `arctic_platform/correctness/configs/qwen3.5-2b/standard/h200-inference-lora.json`. Import recipe JSON, not the catalog's `model.json` metadata; copying a recipe must not rewrite fields, reformat JSON or otherwise change its bytes.

**Only noncatalog configs use derived placement: `<model-name>/<standard|long>/<gpu>-<training|inference>-<full|lora>.json`.** Derive each component from the supplied recipe:

- `<model-name>` is the lowercase model basename, without its repository owner: `Qwen/Qwen3-8B` becomes `qwen3-8b`. For an HF cache snapshot, use the model component, not the snapshot hash.
- `long` requires an explicitly configured `max_tokens_per_mb >= 262144`, including a catalog-style training sub-job's `mb_spec.max_tokens_per_mb` or native `training.ds_worker_config.mb_spec.max_tokens_per_mb`. Missing or smaller budgets mean `standard`; `max_seq_len`, an old filename containing `256k`, and runtime defaults do not select `long`.
- `training` takes precedence when training and sampling jobs both exist; a sampling-only job uses `inference`. Placement does not change whether a training-plus-sampling recipe is reinforcement learning (RL).
- `lora` means the selected job has nonempty parameter-efficient fine-tuning (PEFT) settings for low-rank adaptation (LoRA); otherwise use `full`. Inspect the selected training or sampling job, not an unrelated sub-job.
- `<gpu>` comes from explicit recipe hardware, an existing accelerator directory or filename, or the executing GPU. If none identifies the hardware, report the missing prerequisite rather than guess a GPU type.

**Reference folders mirror the config-relative path with the extension removed.** A config at `configs/qwen3-8b/standard/h200-training-full.json` uses `reference_data/qwen3-8b/standard/h200-training-full/<check-id>/...`. An explicit `--reference-records-dir` changes only the root prefix; the model, sequence-choice and recipe-stem directories remain. Check-specific subdirectories and measurement filenames are unchanged.

**The `arctic_platform/correctness/config-layout.json` manifest preserves migrated frozen config IDs independently of placement.** Moving an existing config or its reference folder does not rename its specification, relabel recorded numerical identities, remeasure references or rewrite historical measurement bytes. A placement-only move is not reference preparation and does not make a historically incompatible identity compatible.

**Configs without a manifest entry derive new IDs from their config-relative paths.** The [config identity helper](../harness/config.py) joins the relative path components, without the extension, with `-`. The new demo variant `tiny-qwen3-129m/standard/h200-training-full--fp32.json` derives `tiny-qwen3-129m-standard-h200-training-full--fp32`. The historical unsuffixed path retains the manifest-frozen `tiny-qwen3-129m-one-h200` ID and its unchanged recipe, specifications and reference bytes. Specifications remain `specs/<config-id>.json` and summaries remain `specs/<config-id>.onboarding.json`. Use the resolved ID printed by the command or `list`, not the source filename or a guess based on the new folder.

**Old correctness-config paths remain aliases only within their original config root.** Frozen specifications can still locate migrated configs through the manifest. A deleted source recipe under `cortex-training/model-catalog/models/` is not such an alias; importing that source still requires the recipe file to exist.

**Distinct existing recipes retain explicit descriptive `--<variant>` filename suffixes when their derived placements collide.** For example, different training recipes may use `h200-training-full--sft-16gpus-sp16-256k.json` and `h200-training-full--rl-32gpus-sp16-256k.json`. Their reference folders include those suffixes, but their frozen IDs remain unchanged. New imports reuse an identical existing copy and error on conflicting bytes. For a noncatalog variant, append a lowercase descriptive suffix such as `--rl-16gpus` to the source filename stem before importing; the importer retains that suffix in the derived recipe filename. Catalog imports still retain their exact catalog-relative filenames.

**The P1 demo's explicit FP32 accumulation makes it a new recipe variant.** Its external filename is `tiny-qwen3-129m-one-h200--fp32.json`, and its imported path is `configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json`. Explicit `training.ds_config.data_types.grad_accum_dtype: "fp32"` avoids dependence on the installed runtime's default. An omitted or null accumulation value is admitted only when the automatically owned local DSS service's submitted-payload builder and DeepSpeed parsers resolve FP32 accumulation and select local workers. Supplied endpoints, nonlocal workers and unverified defaults remain unsupported. The demo's explicit setting differs from the historical tiny recipe; onboard fresh compatible specifications and references rather than editing the old recipe or reusing its frozen identity.

**The first-two comparisons support BF16 model weights only.** Onboarding and regression reject other declared model dtypes before model or target work rather than silently measure them as BF16. FP32 gradient accumulation is a separate requirement, not FP32 model precision; the demo's `--fp32` variant still declares `"dtype": "bfloat16"`. Do not rewrite an imported recipe to bypass this admission rule.

**Sampling-only recipes can be imported, but P1's two correctness checks require training.** Preserve an inference-only catalog recipe's sampling settings and catalog placement. Import acceptance is not numerical onboarding or runtime validation: a training-based check must reject a recipe without a training sub-job before preparing model assets.

## Create A Reviewed Specification

**Start with the complete example in the [operator guide](running-tests.md).** That example checks out this branch, resolves `axolotl-ai-co/tiny-qwen3-129m`'s weights and tokenizer through Hugging Face's existing `/checkpoint/huggingface` cache, creates a complete one-H200, 128-token training config, and defines every directory used below. Hugging Face reuses cached files and automatically downloads missing files. The example requires an exclusively allocated, prepared head node with the upstream DSS gateway installed. Correctness uses Cortex-training; onboarding manages its own local gateway when no service is configured.

**A new configuration needs only its config file as onboarding input.** Both native client configs and existing Cortex-training recipe JSON with `sub_job_configs` are accepted; pass either file directly with `--config`, without converting or flattening the recipe. The harness derives the source checkpoint from `model_name`, prepares the reduced fixture, calibrates gates, publishes individual and combined references, and validates applicable checks. The generated specification describes the fixture, input batches, applicable checks and calibrated tolerances. Compact records contain independent Hugging Face measurements, not tensor payloads. Source weights use `/checkpoint/huggingface/hub`; the demo config and final reports use the checkout's Git-ignored `./work/` directory. Generated fixtures and calibration calculations use temporary fast storage. Records must be inside the executing library's checkout, which need not be on temporary storage.

```bash
python -m arctic_platform.correctness onboard \
  --config ./work/configs/tiny-qwen3-129m-one-h200--fp32.json
```

**Default-all onboarding attempts applicable checks independently and retains each outcome.** The two first-step checks prepare individual and compatible combined HF selections; E2E has a separate trajectory record, while resume and inference retain their own calibration/execution contracts. Inspect the generated spec and `.onboarding.json` summary using the resolved config ID before accepting gates. Explicit check selectors are optional, not extra default steps. Successful first-two preparation does not establish the other checks' runtime prerequisites.

**Repeating a failed command resumes verified completed preparation.** Recovery records under `./work/logs/onboarding-state/` bind configuration, options, source, code and dependency versions to retained artifacts. Missing or changed artifacts invalidate the affected phase and its successors; failed phases are not completion checkpoints. An optimizer validation failure can therefore retry without recalibrating its compatible references. See [diagnostics and retry](running-tests.md#onboarding) for retained reports.

**Fresh calibration can retain separately reviewed absolute acceptance gates.** Supply `--reviewed-gate single-step-grads=0.008 --reviewed-gate single-step-optimizer=0.001` only when those limits have already been reviewed for the requested workload. With a fresh specification, fixture/cache and log directory, onboarding measures calibration again and records both the newly computed gate and the explicit acceptance gate; no prior calibration provenance is copied. Optimizer records retain the explicit gate at every selected placement and for combined selections. These inputs do not themselves force fresh preparation: preserve prior artifacts separately before a requested restart, and do not use them to widen an existing frozen gradient gate.

**Preparation-only mode is optional, not a prerequisite for config-only onboarding.** Use `--prepare-references` only when intentionally preparing or repairing records from an existing reviewed spec without target execution. This mode does not resize the fixture or rewrite an existing numerical gate.

```bash
# Prepare the identity used by a regression selecting both checks.
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer
```

**Single-check and combined-check workloads have different identities.** Default-all onboarding handles both identities automatically. In preparation-only mode, use `--test-id single-step-grads` or `--test-id single-step-optimizer` for a single-check selection, or both IDs with `--reference-tests` when regression selects both. Optimizer selection changes the measured step and frozen token/model settings, so separately prepared single-check records are not interchangeable with a combined selection.

## Record Identity

**Records bind measurements to the exact workload.** Identity includes configuration checksum, check and arm, model asset contents, reference source and dependency identities, adapter initialization where applicable, input tensor semantics, and check-specific settings. Records use readable arm filenames; integrity hashes remain inside each compact JSON record. They do not contain model or optimizer tensors.

**Each workload folder retains only its latest complete reference set.** Publication stages and validates the complete replacement, renames the previous set aside, installs the new set, and removes the previous directory. Failed publication restores the previous complete set; regression never replaces references. For the new demo config `configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json`, the maintained filenames below are relative to the reference root and separate from the historical unsuffixed recipe's records:

```text
tiny-qwen3-129m/standard/h200-training-full--fp32/single-step-grads/gradients-only/gas1.json
tiny-qwen3-129m/standard/h200-training-full--fp32/single-step-grads/gradients-only/gas4.json
tiny-qwen3-129m/standard/h200-training-full--fp32/single-step-grads/with-optimizer/gas1.json
tiny-qwen3-129m/standard/h200-training-full--fp32/single-step-grads/with-optimizer/gas4.json
tiny-qwen3-129m/standard/h200-training-full--fp32/single-step-optimizer/gas1.json
tiny-qwen3-129m/standard/h200-training-full--fp32/single-step-optimizer/gas4.json
```

**Compact records are generated local artifacts, not files to commit automatically.** They default to `arctic_platform/correctness/reference_data` and remain untracked, not ignored, until explicitly selected for publication. Retain them for later regression; regression cannot regenerate missing HF measurements. The publisher validates containment under the executing library's Git root, not Git tracking or merely the shell's current directory. An unrelated sibling records directory is rejected. No measurements are bundled. To override the default, pass the same `--reference-records-dir` inside the checkout to preparation and regression. Model weights and adapter tensors do not belong in this directory or in Git.

**A repeated identity cannot acquire different measurements.** Identical records can be reused. Corrupt records or conflicting measurements require explicit resolution. Changed sources, assets, or workloads require explicit onboarding and reviewed preparation; regression cannot replace or recalibrate them.

**New single-step measurements also record the implementation and configuration used by HF.** The [reference engine](../reference/hf_single_gpu.py) captures the loaded model and decoder class identities, Python `forward` code digests, source-file content hashes, and effective configurations, including the attention backend that HF configuration serialization can omit. The engine compares these observations before and after execution and rejects a changed implementation, source file or configuration. The [reference CLI](../reference/__main__.py) retains these observations in `reference_execution`; compact-record publication and loading preserve that field without rerunning HF. This limited observation does not capture every transitive kernel or prove native target execution. Historical records without this field remain historical: neither regression nor this change reconstructs their original execution evidence.

### Historical Records

**The old naming-source exception is retired.** The removed `--historical-gradient-references` options admitted only one reviewed change to the former `harness/names.py`. Consolidation into `harness/optimizer.py` changes the bound source and does not satisfy those original pins. Current regression requires an exact workload identity; use explicit onboarding to prepare compatible records. Frozen historical records and their original reports remain evidence, not compatible current references.

## Optimizer Criterion


**Each HF worker reports its arm, run name, device, completion or failure, and retained logs.** Calibration consumes completed workers immediately but retains the original run order for gate calculation. During onboarding, the [runner](../harness/runner.py) writes complete stdout, stderr and exit-code files under that attempt's `./work/logs/onboarding-*/hf-reference-*/` directory, including successful workers. The console prints their exact locations. `onboarding.log` retains console output and chained failures; `report.md` states which check failed and why. `AP_CORRECTNESS_LOGDIR` overrides the log root. Tensor scratch remains on fast storage, separate from logs and compact records.

**`optimizer-state-l2-norm-v1` compares scalar magnitudes, not full tensors.** For each named parameter and comparison state, the criterion is `abs(target_norm - reference_norm)`. The update residual subtracts the Adam update explained by each backend's own moments before taking the norm. Parameter coverage, shapes, and finite nonnegative norms must agree; zero and near-zero norms have no exemption. See [optimizer_state.py](../checks/optimizer_state.py) and the [optimizer check](../checks/fwd_bwd_step.py).

**Equal norms can hide sign, permutation, or directional errors.** `abs(||target||2 - ||reference||2)` is weaker than the full-tensor difference `||target - reference||2`; a norm pass does not establish elementwise optimizer-state equivalence.

**A new optimizer norm record needs fresh, criterion-specific onboarding calibration.** Config-only onboarding uses eight nondeterministic single-GPU HF repeats per arm, in addition to the deterministic baseline capture, for tiny and real models alike. The [runner](../harness/runner.py) measures the largest absolute scalar-norm difference over all repeat pairs, multiplies that spread by 4, rounds upward to a `1e-3` increment, and applies a minimum absolute gate of `1e-3`. The record retains the gate and actual calibration run count. Existing records retain their original provenance and reviewed gates. No legacy full-tensor tolerance is inherited, and regression neither recalibrates nor widens this gate.

**Calibration automatically uses the visible GPUs on one node, capped at eight workers.** Each worker runs an independent single-GPU HF process. Eight samples run in one wave of eight per GAS arm when eight or more local GPUs are visible, including when sixteen GPUs are visible. Fewer local GPUs require more waves. `--reference-calibration-workers` preserves an explicit override from one to eight; the operator must make those GPUs available. Gradient and optimizer calibration use this policy independently of the target's GPU count.

**Independent process startup and optimizer capture can dominate tiny-model calibration.** Every HF repeat loads the fixture in a new process and writes optimizer-state artifacts before exiting. Eight repeats plus a deterministic baseline for each of GAS1 and GAS4 means eighteen processes for the optimizer check; concurrent workers overlap repeats, not the two GAS arms or their baselines. Progress counts successful worker completions and reports unfinished run IDs and wave elapsed time, with a waiting update every 30 seconds and no sequential ETA.

**TODO: Use every available GPU within one allocation for calibration.** Multi-node support must handle remote worker launches, node-local model and input staging, unique GPU assignments, and result and failure collection before removing the one-node limit.

## LoRA Prerequisites

**First-two reference preparation detects low-rank adaptation (LoRA) settings directly from the supplied config.** Native `training.peft` or `training.ds_worker_config.peft_config` enables automatic generation; a config without LoRA settings remains full fine-tuning. Onboarding and regression use upstream PEFT's default initializer with the explicit harness seed `1234`, a metadata-only base architecture and CPU adapter factors. Production PEFT code does not call a correctness initializer. The correctness-owned local gateway loads the shared adapter for target onboarding and regression. An external gateway requires compatible correctness routes; hosted Cortex does not support this reduced-fixture workflow. See [the operator guide](running-tests.md#lora-without-adapter-inputs).

**Compact records contain the generation recipe and content hashes, not adapter tensors.** Identity binds the dependency versions, topology, tensor shapes/dtypes and exact values. Regression reconstructs adapter factors and requires matching prepared references; it never runs HF again to accommodate a mismatch. Only explicit onboarding measures references.

**The correctness-owned local gateway transfers verified adapter bytes through Ray's allocation-local object store.** Workers do not open an adapter path on the head node or assume `/data-fast` is shared. Base-model files and installed code still require supported per-node distribution. Optional `--peft-adapter` overrides automatic generation for reference preparation. Dense `target_modules` with default initialization are supported; expert `target_parameters` and advanced initializers fail explicitly. The old deterministic SAVE0 execution path is retired.

## Run Regression

### Native Multi-Rank Optimizer Capture

**A local optimizer checkpoint must include every actual training rank, not just the rank-zero response from the save API.** Expert parallelism (EP) partitions experts across workers; sequence parallelism (SP) partitions a sequence across workers. The optimizer checkpoint can therefore span all configured GPUs even when the config's data-parallel width is one. The optional [runner integration](../harness/runner.py) uses `cfg.n_gpus` for native collection; ordinary capture continues to use `cfg.dp_size`.

**This integration is an internal Python callback interface, not a new CLI flag or automatic runtime discovery.** Pass `native_optimizer_scope`, `native_optimizer_scope_source`, and optionally `native_optimizer_supplier_source` to `runner.execute(...)`. A source descriptor has exactly `{"path": "/absolute/loaded/source.py", "sha256": "<full-file-sha256>"}` and must describe the callback's actual loaded source. If scope and supplier come from different files, provide both descriptors; otherwise the supplier uses the scope descriptor.

**The scope owns collection and cleanup for one target arm.** `native_optimizer_scope(session, job)` returns a context manager whose entered value is a callable `supplier(checkpoint_root, *, step)`. The [capture helper](../harness/optimizer.py) invokes that supplier after each existing save, at step zero and step one. The supplier gathers the already-saved rank files and returns the validated native layout with the actual parent, training child and worker bindings. It must not initiate another optimizer step or substitute expert sidefiles for optimizer tensors. Context exit must restore owned hooks and finish owned cleanup before the training child is released.

**Admission remains explicit and narrow.** The runner requires target execution with a loopback `CortexConfig`, `local_cortex_fixture=True`, and an explicit selection containing `single-step-optimizer` and no checks beyond the first two supervised-reference checks. Reference preparation and reinforcement-learning configs cannot use this callback. Existing model identity, parameter coverage, dtype, partition ordering, snapshot integrity and numerical gates still apply. Installing these hooks or passing CPU controls does not establish a live worker route, a complete checkpoint, successful resource cleanup or a numerical pass.


**Run regression with one command in one terminal, just like onboarding.** With no Cortex host or base URL configured, regression starts and stops its owned installed DSS gateway and selects the reduced fixture automatically. An explicitly configured endpoint is external and must already be running; the command does not stop it. Loopback database/schema segments default to `dev`. See the [Running section](running-tests.md#running) for prerequisites, diagnostics and cleanup limits.

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --out ./work/reports/both
```

**CPU tests and preparation receipts are not numerical target passes.** Only an actual target comparison under the reviewed workload establishes a verdict. See [the operator guide](running-tests.md) for backend prerequisites and result files.
