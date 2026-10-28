# HF Reference Onboarding (Agent Guide)

**This guide describes maintained reference preparation and regression, not a particular fleet run.** Start with the [operator guide](running-tests.md) for setup and copyable commands. The [documentation index](../../../docs/correctness.md) lists the maintained guides. Local run history and numerical results belong in the uncommitted `correctness-progress.md` at the repository root, not in installation instructions.

## Prepare Once

### Automatic Local Validation And Retry

**A first-two config-only onboarding command manages its local Cortex-training target.** With no Cortex host or base URL configured, the harness selects the reduced fixture, starts the installed upstream DSS gateway for validation and closes that owned process afterwards. It does not replace an explicitly configured service. See the [operator guide](running-tests.md#onboarding) for environment prerequisites.

**Repeating a failed command resumes verified completed phases.** Recovery records under `./work/logs/onboarding-state/` bind the configuration, options, source, code and dependency versions to retained artifacts. Missing or changed artifacts invalidate affected phases and successors; failed validation is retried without repeating compatible preparation. Logs and reports live in a unique `./work/logs/onboarding-*/` directory, while models and tensor scratch remain on fast storage.

**A target transport fix can reuse preparation, but must run target validation again.** Retry metadata distinguishes target-only transport changes and the explicitly reviewed cache-reuse/logging changes from changes to the reference computation, configuration, dependencies, inputs or numerical gates. Compatible preparation stays in its original directory and its artifacts are checked before reuse. An older retry directory without a saved request manifest cannot be assumed compatible after a source change.

### Automatic LoRA Initialization

**First-two reference preparation detects LoRA settings directly from the supplied config.** Native `training.peft` or `training.ds_worker_config.peft_config` enables automatic generation; a config without LoRA settings remains full fine-tuning. Generation uses upstream PEFT's default initializer and seed `42`, with a metadata-only base architecture and CPU adapter factors. Production PEFT code does not call a correctness initializer. Cortex-training currently has no admitted shared-adapter target-loading route, so LoRA target onboarding and regression remain unsupported. [The operator guide](running-tests.md#lora-without-adapter-inputs) distinguishes reference generation from target support.

**Compact records bind generated values, not filesystem paths.** The initialization identity contains the recipe, exact dependency versions, topology, tensor shapes/dtypes and content hashes. Regression reconstructs those factors and requires the original reviewed reference identity; it never measures HF again to accommodate changed factors. Stage the config and reviewed specification only; compact measurements remain local and Git-ignored, while adapter, model and optimizer tensors stay on fast storage.

**The retained internal Ray adapter transport is not a Cortex-training loading route.** That integration transfers verified adapter bytes through the allocation's object store, rather than assuming `/data-fast` is shared. Model weights and installed code still require the runtime's supported per-node distribution. Optional `--peft-adapter` overrides automatic generation for reference preparation. Generation supports dense `target_modules` LoRA with upstream default initialization; unsupported expert-parameter layouts and advanced initializers fail explicitly.

**For a new model, give onboarding only its config file.** Both native client configs and existing Cortex-training recipe JSON with `sub_job_configs` are accepted; pass either file directly with `--config`, without converting or flattening the recipe. The complete example in the [operator guide](running-tests.md#setup) resolves weights and tokenizer through Hugging Face's `/checkpoint/huggingface` cache, creates a native config and runs onboarding. The harness selects applicable registered checks, prepares reduced fixtures for the first two checks, calibrates gates, publishes individual and combined references, and validates the targets. P1 selects only its two registered checks; the full branch also selects compatible later checks, whose target prerequisites still apply.

```bash
python -m arctic_platform.correctness onboard \
  --config ./work/configs/tiny-qwen3-129m-one-h200.json
```

**Specifications and compact records serve different purposes.** A specification records model/input settings, applicability and reviewed tolerances; compact reference records contain independent HF measurements. The preparation-only commands below assume an existing reviewed specification and matching model assets.

**Hugging Face (HF) reference measurements are prepared explicitly during onboarding, not regenerated during regression.** Use `onboard --prepare-references` with a matching reviewed SFT (supervised fine-tuning) config and frozen test specification. Preparation does not resize the fixture, rewrite historical gates, or run the target backend. Matching records are reused; explicit preparation computes missing or incompatible records and replaces their complete workload sets. Corrupt records must be resolved before preparation; regression never replaces records.

**The three supported check IDs select different compact measurements.**

- `single-step-grads`: named gradient L2 norms and loss for each configured arm.
- `single-step-optimizer`: step-one Adam first-moment, second-moment, and update-residual L2 norms, with canonical parameter names and shapes.
- `e2e-sft-train-validate`: 100 training losses and gradient norms, optimizer-step count, held-out validation cross-entropy, and scored-answer-token count.

**Each workload folder retains only its latest complete reference set.** Compact records default to the local, Git-ignored `arctic_platform/correctness/reference_data` directory; they are generated artifacts, not files to commit. Override the directory with `--reference-records-dir` on both onboarding and regression. Publication requires a location inside the checkout, not Git tracking. [storage.py](../harness/storage.py) uses readable filenames:

```text
<config-id>/single-step-grads/gradients-only/gas1.json
<config-id>/single-step-grads/gradients-only/gas4.json
<config-id>/single-step-grads/with-optimizer/gas1.json
<config-id>/single-step-grads/with-optimizer/gas4.json
<config-id>/single-step-optimizer/gas1.json
<config-id>/single-step-optimizer/gas4.json
<config-id>/e2e-sft-train-validate/trajectory.json
```

**Replacement never deletes the previous set before preparation succeeds.** Preparation validates every selected arm and writes the complete replacement to a temporary directory. Under a per-configuration lock, publication renames the old configuration directory aside, renames the new directory into place, then deletes the old directory without automatic archival. The published layout uses ordinary directories, not symbolic links. Failed or incomplete staging leaves the previous set intact; a failed second rename restores the old directory. The next read or publication recovers an interrupted replacement and removes abandoned staging. Unselected workloads remain unchanged.

**Readable filenames do not weaken reference validation.** Every JSON record retains its complete workload identity and measurement checksum. The identity binds the config, reference sources and dependency versions, model contents, initialization, input tensors and check-specific options, not machine-specific cache roots. Changed identities require explicit preparation; regression cannot regenerate references. Different measurements for an identical identity remain a publication conflict. Existing SHA-named records can be consumed only by exact identity; successful preparation removes validated legacy generations for the workloads being replaced. Explicit historical bindings below retain their original filenames and review requirements.

**New single-step measurements also record the implementation and configuration used by HF.** The [reference engine](../reference/hf_single_gpu.py) captures the loaded model and decoder class identities, Python `forward` code digests, source-file content hashes, and effective configurations, including the attention backend that HF configuration serialization can omit. The engine compares these observations before and after execution and rejects a changed implementation, source file or configuration. The [reference CLI](../reference/__main__.py) retains these observations in `reference_execution`; compact-record publication and loading preserve that field without rerunning HF. This limited observation does not capture every transitive kernel or prove native target execution. Historical records without this field remain historical: neither regression nor this change reconstructs their original execution evidence.

**Prepare each required check with its exact CLI ID.** These examples use the reviewed dense Qwen3-8B config; gradient and optimizer preparation use the fixtures selected by its frozen spec.

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --prepare-references --test-id single-step-grads

python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --prepare-references --test-id single-step-optimizer

python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --prepare-references --test-id e2e-sft-train-validate \
  --local-gateway-full-model
```

**Prepare gradients and optimizer together when regression selects both checks.** The prep-only `--reference-tests` option replaces the scalar `--test-id` selection and prepares both records for the combined workload; use the full check IDs, not `grads` or `optimizer` aliases. Selecting optimizer enables an optimizer step and its frozen model/token-budget settings, which also enter the gradient record's workload identity. Separate scalar preparations are therefore not interchangeable with combined preparation; regression must find records matching its actual selection.

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer
```

**Advanced full-model mode is separate from config-only reduced-fixture onboarding.** [The original-model validator](../models/original.py) defines the accepted model/config pairs, source revisions, configuration hashes and indexed tensor layouts. It currently lists Qwen3-8B, the B200 Qwen3.8-27B config and the reviewed Flash-Next config; metadata inclusion does not establish a compatible sampling runtime or numerical pass. This mode requires `HF_HUB_OFFLINE=1`, `DSS_ALLOW_REMOTE_MODELS=1` and `HF_HUB_CACHE` pointing to the populated cache under `/data-fast`. Indexed models must match complete tensor-name/dtype/shape headers. Header validation does not read or hash tensor payloads.

**`--local-gateway-full-model` is restricted to E2E and inference regression; reference preparation remains E2E-only.** The reference-free inference check compares training against sampling and does not need an HF record. Source admission alone does not supply a missing E2E record or compatible sampling runtime. These advanced targets require an already running, externally managed gateway supplied through `--base-url`; the automatic gateway lifecycle in the [operator guide](running-tests.md#running) covers only the first two supervised-reference checks.

## Calibrate Optimizer Norms

**Each HF worker reports its arm, run name, device, completion or failure, and retained logs.** Calibration consumes completed workers immediately but retains the original run order for gate calculation. During onboarding, the [runner](../harness/runner.py) writes complete stdout, stderr and exit-code files under that attempt's `./work/logs/onboarding-*/hf-reference-*/` directory, including successful workers. The console prints their exact locations. `onboarding.log` retains console output and chained failures; `report.md` states which check failed and why. `DSS_CORRECTNESS_LOGDIR` overrides the log root. Tensor scratch remains on fast storage, separate from logs and compact records.

**`optimizer-state-l2-norm-v1` compares scalar magnitudes, not full tensors.** For each named parameter and comparison state, the criterion is `abs(target_norm - reference_norm)`. The update residual subtracts the Adam update explained by each backend's own moments before taking the norm. Parameter coverage, shapes, and finite nonnegative norms must agree; zero and near-zero norms have no exemption. See [optimizer_state.py](../checks/optimizer_state.py) and the [optimizer check](../checks/fwd_bwd_step.py).

**Equal norms can hide sign, permutation, or directional errors.** `abs(||target||2 - ||reference||2)` is weaker than the full-tensor difference `||target - reference||2`; a norm pass does not establish elementwise optimizer-state equivalence.

**A new optimizer norm record needs fresh, criterion-specific onboarding calibration.** Config-only onboarding uses eight nondeterministic single-GPU HF repeats per arm, in addition to the deterministic baseline capture, for tiny and real models alike. The [runner](../harness/runner.py) measures the largest absolute scalar-norm difference over all repeat pairs, multiplies that spread by 4, rounds upward to a `1e-3` increment, and applies a minimum absolute gate of `1e-3`. The record retains the gate and actual calibration run count. Existing records retain their original provenance and reviewed gates. No legacy full-tensor tolerance is inherited, and regression neither recalibrates nor widens this gate.

**Calibration automatically uses the visible GPUs on one node, capped at eight workers.** Each worker runs an independent single-GPU HF process. Eight samples run in one wave of eight per GAS arm when eight or more local GPUs are visible, including when sixteen GPUs are visible. Fewer local GPUs require more waves. `--reference-calibration-workers` preserves an explicit override from one to eight; the operator must make those GPUs available. Gradient and optimizer calibration use this policy independently of the target's GPU count.

**Independent process startup and optimizer capture can dominate tiny-model calibration.** Every HF repeat loads the fixture in a new process and writes optimizer-state artifacts before exiting. Eight repeats plus a deterministic baseline for each of GAS1 and GAS4 means eighteen processes for the optimizer check; concurrent workers overlap repeats, not the two GAS arms or their baselines. The progress bar measures complete reference runs, including this overhead; its ETA appears after a completion.

**TODO: Use every available GPU within one allocation for calibration.** Multi-node support must handle remote worker launches, node-local model and input staging, unique GPU assignments, and result and failure collection before removing the one-node limit.

## Run Regression

### Native Multi-Rank Optimizer Capture

**A local optimizer checkpoint must include every actual training rank, not just the rank-zero response from the save API.** Expert parallelism (EP) partitions experts across workers; sequence parallelism (SP) partitions a sequence across workers. The optimizer checkpoint can therefore span all configured GPUs even when the config's data-parallel width is one. The optional [runner integration](../harness/runner.py) uses `cfg.n_gpus` for native collection; ordinary capture continues to use `cfg.dp_size`.

**This integration is an internal Python callback interface, not a new CLI flag or automatic runtime discovery.** Pass `native_optimizer_scope`, `native_optimizer_scope_source`, and optionally `native_optimizer_supplier_source` to `runner.execute(...)`. A source descriptor has exactly `{"path": "/absolute/loaded/source.py", "sha256": "<full-file-sha256>"}` and must describe the callback's actual loaded source. If scope and supplier come from different files, provide both descriptors; otherwise the supplier uses the scope descriptor.

**The scope owns collection and cleanup for one target arm.** `native_optimizer_scope(session, job)` returns a context manager whose entered value is a callable `supplier(checkpoint_root, *, step)`. The [capture helper](../harness/optimizer.py) invokes that supplier after each existing save, at step zero and step one. The supplier gathers the already-saved rank files and returns the validated native layout with the actual parent, training child and worker bindings. It must not initiate another optimizer step or substitute expert sidefiles for optimizer tensors. Context exit must restore owned hooks and finish owned cleanup before the training child is released.

**Admission remains explicit and narrow.** The runner requires target execution with a loopback `CortexConfig`, `local_cortex_fixture=True`, and an explicit selection containing `single-step-optimizer` and no checks beyond the first two supervised-reference checks. Reference preparation and reinforcement-learning configs cannot use this callback. Existing model identity, parameter coverage, dtype, partition ordering, snapshot integrity and numerical gates still apply. Installing these hooks or passing CPU controls does not establish a live worker route, a complete checkpoint, successful resource cleanup or a numerical pass.

**Regression consumes a compatible prepared record or stops with a prerequisite error; it never falls back to a fresh HF reference run.** Regenerate a baseline only through explicit onboarding and review, not to hide a target mismatch. The record loader may hash assets and reconstruct input identities; that validation is not a reference-model training run. The [E2E check](../checks/e2e_sft_train_validate.py) loads the stored trajectory and validation cross-entropy before running the target.

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config \
  --test e2e-sft-train-validate --backend cortex-training \
  --base-url http://127.0.0.1:8003 --database dev --schema dev \
  --local-gateway-full-model --out ./work/reports/qwen3-8b-e2e
```

### Reviewed Historical Gradients

**New-model onboarding does not need historical-reference options.** The optional `run --historical-gradient-references <descriptor.json> --historical-gradient-references-sha256 <full-sha256>` pair is for maintainers who already hold reviewed, original gradient records. An identity mismatch alone does not authorize reuse.

**Only the reviewed DSS shared-expert alias change is accepted.** The [loader](../harness/storage.py) permits the historical `harness/names.py` source pin to change from SHA-256 `a5c941eb9592fabe042c8f48a28bfe7e650f5b65d3b8b7a6fd3824397c63ba09` (9,905 bytes) to `4d6cd215ef4639ab2d85f5928ca2df83954012878c39366c74e29207cc92725c` (10,294 bytes). Every other workload field must match exactly, including HF sources, dependencies, model files, input tensors, initialization, config, parameter-name map and gradient tolerance. The installed naming source must match the current pin. Reuse does not exclude tensors, widen a tolerance or rerun HF measurements.

**The descriptor binds every selected arm to its original record bytes.** The JSON object has exactly two fields: `schema`, whose value is `"reviewed-gradient-references-v1"`, and `records`. Each key in `records` names a selected arm, such as `gas1` or `gas4`. Each arm has exactly `record_path` (the original JSON filename), `record_sha256` (64 lowercase hexadecimal characters hashing the entire original file) and `review` (a nonempty string identifying the source review). Relative record paths resolve against the descriptor's directory. The record filename must remain its historical workload-identity digest; renaming the record to the current lookup key is rejected. The second CLI option separately pins the descriptor's complete bytes.

**Reuse supports one explicit supervised-fine-tuning config and only `single-step-grads`.** Optimizer or mixed selections, all-config runs, reference preparation, reinforcement-learning configs, and unknown or missing arm bindings are rejected. The [runner](../harness/runner.py) writes `reference-record-provenance.json` in the run's work directory before target work, retaining both complete identities, the original record path and raw hash, and the review string. A rejected binding never falls back to another record or fresh HF computation. Historical gradient reuse does not admit optimizer references or establish a model pass.

**Only compact JSON records belong in the reference directory.** Model fixtures, adapter artifacts, optimizer captures, replay tensors and calibration tensor scratch stay under `/data-fast`; `--output-dir` overrides that fast-storage work directory, not the durable log directory. [storage.py](../harness/storage.py) selects `/data-fast/tmp` by default, with `DSS_CORRECTNESS_TMPDIR` as its override. Small reports, complete logs and retry checkpoints use `./work/logs`, or `DSS_CORRECTNESS_LOGDIR`; regression `--out` should likewise select `./work/reports/`. Preparation, CPU controls and gateway admission alone do not establish numerical target passes.
