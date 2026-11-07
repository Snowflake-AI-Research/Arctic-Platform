# Running The First Two Correctness Checks

**Run these commands from an already prepared, allocated head node.** The library does not allocate machines, install a runtime, download a missing reference during regression, or modify reviewed numerical gates. Onboarding and regression can start and stop their owned installed local DSS gateway. The correctness backend is `cortex-training`; `cortex` remains a compatibility alias.

**This branch measures inside correctness without changing production code.** Seed-0 reduced pretrained `prime_rl` fixtures with positive prediction-layer counts require loading and authentication absent from main: `mtp_num_hidden_layers` for `qwen3_5_moe_text` and `qwen4_exp_text`, or `num_nextn_predict_layers` for `glm5_next_text`. Such fixtures fail explicitly. Absent or zero counts pass this admission check unchanged; do not remove prediction layers or alter reviewed gates to bypass it.

## Prerequisites

- Install the project and its compatible PyTorch, DeepSpeed, Transformers, and PEFT dependencies in the selected interpreter.
- Install with `python -m pip install -e '.[correctness]'`. The correctness extra supplies training and reference dependencies plus `pytest` and `pytest-instafail` for CPU selftests; the separate `testing` extra is not required for this workflow. Install CUDA and model-specific attention kernels separately.
- Make the allocation's GPU family and available devices match the supplied configuration. For multiple nodes, provide the allocated hostfile through `DSS_GATEWAY_HOSTFILE` and working allocation-local `ds_ssh`.
- Make the reviewed model and tokenizer assets readable at the paths recorded in the test specification.
- Keep generated fixtures, adapter packs, optimizer snapshots and calibration scratch on allocation-local fast storage. The harness uses `/data-fast` defaults; `DSS_CORRECTNESS_TMPDIR` can select another directory under that mount. Temporary files disappear with the allocation.
- Choose persistent locations for the checkout, downloaded source model, native config and final reports. Supply the native config (`.config` or `.json`) to onboarding; it creates the specification and references. Regression requires those reviewed inputs. No real configurations, specifications or measurements are bundled.

## Checks

**Only two numerical checks are registered.** Both exercise the synthetic input arms recorded in the specification; the arms include gradient accumulation settings.

| Check ID | Comparison |
| --- | --- |
| `single-step-grads` | Named per-parameter gradient L2 norms against the prepared single-GPU Hugging Face reference. |
| `single-step-optimizer` | Step-one parameter-update residual, Adam first-moment, and Adam second-moment L2 norms, with matching parameter names and shapes. |

**Optimizer norm agreement is not elementwise agreement.** Equal L2 norms can hide a sign change, permutation, or directional error. Regression uses the reviewed criterion and calibrated gate stored in the prepared record; regression never widens that gate.

## Setup

**This quick demo uses `axolotl-ai-co/tiny-qwen3-129m`, a 129M-parameter Qwen3 model, with 128-token sequences on one exclusively allocated H200 GPU.** Activate a Python 3.12+ environment with working CUDA, FlashAttention-3 kernels and the installed upstream DSS GPU gateway first. The `correctness` extra installs the Cortex client, training and Hugging Face reference dependencies, and CPU selftest tools; it does not install the separate DSS service, CUDA or model-specific attention kernels. Use the existing `/checkpoint/huggingface` cache; Hugging Face downloads missing model files there automatically, requiring network access and sufficient cache space. The gateway uses the allocation's Ray runtime, so do not run on a shared cluster. No fresh GPU verification is established by the controls for gateway, progress and timing changes.

**Both checks require BF16 model weights with verified FP32 gradient accumulation.** An explicit `data_types.grad_accum_dtype: "fp32"` is supported without runtime-default inspection. An omitted or null value requires the automatically owned local DSS service: admission passes the actual submitted payload through the installed engine builder and DeepSpeed's typed parsers, and verifies that the service selects only local workers. A BF16 result, nonlocal placement, supplied endpoint or unavailable builder is rejected without overriding the recipe. Other declared model dtypes are also rejected rather than converted for comparison. The demo's `--fp32` filename refers to accumulation, while its model dtype remains `bfloat16`.

**Start in the directory where you want to keep the checkout and results, then copy the blocks in order.** No pre-existing config, specification or measurement is needed. After cloning, remain in the Arctic Platform checkout: the demo config and final reports use its Git-ignored `./work/` directory. Model weights stay in `/checkpoint/huggingface`, not the checkout. A specification is generated JSON describing the reduced fixture, input batches, applicable checks and calibrated tolerances. Reference records contain compact measurements from independent single-GPU Hugging Face runs, not tensor payloads. Records must be inside the executing library's checkout, but that checkout need not be on temporary storage. Generated fixtures and calibration scratch use temporary allocation-local defaults; retain printed preparation logs separately if the allocation will disappear.

```bash
set -e
mkdir -p ./correctness-no-debug
cd ./correctness-no-debug
git clone --single-branch --branch stas/correctness-testing-no-debug \
  https://github.com/Snowflake-AI-Research/Arctic-Platform.git \
  ./Arctic-Platform
cd ./Arctic-Platform
python -m pip install -e '.[correctness]'
python -m arctic_platform.correctness list
```

## Model Configuration

**Resolve the model through the existing Hugging Face cache and write a complete native training config.** `snapshot_download()` reuses cached files and downloads missing files into `/checkpoint/huggingface/hub`, the Hub subdirectory of `/checkpoint/huggingface`. No commit hash is required: Hugging Face resolves the model ID's default revision. All subsequent commands run from the checkout root. The config uses the returned snapshot path and declares the GPU count, precision, attention implementation and optimizer.

```bash
mkdir -p ./work/configs ./work/reports ./work/logs
export DSS_CORRECTNESS_TMPDIR="$(mktemp -d /data-fast/tmp/correctness-demo.XXXXXXXX)"
export DSS_CORRECTNESS_LOGDIR="$PWD/work/logs/correctness-demo"
export HF_HOME=/checkpoint/huggingface
export HF_HUB_CACHE=/checkpoint/huggingface/hub
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="axolotl-ai-co/tiny-qwen3-129m",
    cache_dir="/checkpoint/huggingface/hub",
)
config = {
    "model_name": str(model_dir),
    "seed": 42,
    "dtype": "bfloat16",
    "max_seq_len": 128,
    "training_gpus": 1,
    "sampling_gpus": 0,
    "log_prob_gpus": 0,
    "training": {
        "ds_config": {
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
            "data_types": {"grad_accum_dtype": "fp32"},
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 2e-5,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
        },
        "ds_worker_config": {
            "model_provider": "huggingface",
            "attn_implementation": "flash_attention_3",
            "activation_checkpointing": False,
        },
    },
    "sampling": {},
    "backend": {
        "type": "remote",
        "protocol": "cortex",
        "base_url": "http://127.0.0.1:1",
        "database": "dev",
        "schema": "dev",
    },
}
Path("./work/configs/tiny-qwen3-129m-one-h200--fp32.json").write_text(
    json.dumps(config, indent=2) + "\n"
)
PY
```

**The config uses the remote Cortex protocol, matching the correctness commands' `cortex-training` backend.** The loopback URL on port 1 is a non-serving placeholder required by the client schema. Onboarding and regression replace that placeholder with their automatically started gateway's actual URL. Do not start a server on port 1 or export the placeholder as `ARCTIC_CORTEX_BASE_URL`; an explicit endpoint selects an externally managed service. `--backend onprem` is rejected.

**The example uses full fine-tuning, not low-rank adaptation (LoRA).** Native `training.peft` or `training.ds_worker_config.peft_config` is detected automatically. For local first-two testing, the correctness-owned gateway installs the same generated adapter used by the HF reference and checks its identity and parameter coverage. Hosted Cortex does not admit this reduced-fixture shared-adapter workflow. See [LoRA without adapter inputs](#lora-without-adapter-inputs).

### Imported Config And Reference Locations

**External configs are imported byte for byte into this checkout before onboarding or running.** The harness uses the local copy for configuration identity, specifications and execution; it does not rewrite recipe settings or automatically commit the import. The original external file is no longer needed after import. An identical existing copy is reused; conflicting bytes at the destination cause an error without overwriting the existing recipe.

**Catalog recipes keep their complete path below `cortex-training/model-catalog/models/`.** Replace that prefix with `arctic_platform/correctness/configs/`, preserving every remaining directory and filename. For example, `qwen3.5-2b/standard/h200-inference-lora.json` retains that relative path even though the recipe is inference-only. Import recipe JSON, not the catalog's `model.json` metadata.

**Noncatalog configs use `<model-name>/<standard|long>/<gpu>-<training|inference>-<full|lora>.json` with any explicit `--<variant>` suffix.** The model name is the lowercase model basename without the repository owner; an HF cache snapshot uses the model component, not the snapshot hash. Only an explicit `max_tokens_per_mb >= 262144` selects `long`, including a training sub-job's `mb_spec` or native `training.ds_worker_config.mb_spec`; missing or smaller budgets select `standard`, regardless of `max_seq_len` or runtime defaults. Training takes precedence over sampling when both exist, and `lora` comes from the selected job's nonempty parameter-efficient fine-tuning (PEFT) settings; otherwise use `full`. Resolve the GPU from explicit recipe hardware, an existing accelerator directory or filename, or the executing GPU; missing hardware identity is not permission to guess. The demo imports as `configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json`, retaining the source filename's explicit `--fp32` suffix.

**Reference folders mirror the imported config path without its extension.** The new demo's compact records use `arctic_platform/correctness/reference_data/tiny-qwen3-129m/standard/h200-training-full--fp32/<check-id>/...`. `--reference-records-dir` changes only the root prefix. Existing check-specific subdirectories and measurement filenames remain unchanged; see the [reference layout](hf-reference-onboarding.md#configuration-and-reference-layout) for examples.

**Migrated config IDs remain frozen through `arctic_platform/correctness/config-layout.json`; new IDs derive from relative paths when no manifest entry exists.** New IDs join the config-relative path components, without the extension, with `-`. The new demo derives `tiny-qwen3-129m-standard-h200-training-full--fp32`; the historical unsuffixed tiny config keeps the manifest-frozen `tiny-qwen3-129m-one-h200` ID. Specifications remain `specs/<config-id>.json` and summaries remain `specs/<config-id>.onboarding.json`; use the resolved ID printed by the command or `list`, not the original filename. Historical measurement bytes and recorded identities are not rewritten or remeasured.

**Recipe variants distinguish different bytes at the same derived placement.** Distinct migrated recipes retain descriptive `--<variant>` filename suffixes and their frozen IDs. For a new noncatalog variant, append a lowercase descriptive suffix such as `--sft-4gpus` to the source filename stem before importing; changing the source basename alone does not distinguish recipes. Catalog imports retain their exact relative filenames, and conflicting bytes at that destination are rejected.

**The P1 demo uses a new explicit FP32 variant rather than changing the historical tiny recipe.** The example supplies `training.ds_config.data_types.grad_accum_dtype: "fp32"` so it does not depend on runtime-default admission, and uses `tiny-qwen3-129m-one-h200--fp32.json`; prepare its specifications and references for the new derived ID. Keep the old unsuffixed config, manifest entry, specs and reference bytes unchanged.

## Onboarding

**Onboarding checks prerequisites before loading model weights or creating a reduced fixture.** The supplied config and selected applicable tests determine the required attention backend, LoRA dependencies and whether a sampling runtime is needed. The check reports the executing Python interpreter, verifies imports and CUDA/GPU compatibility, and checks access to the model source and writable cache, scratch, specification and reference-record locations. It does not install packages, download weights or substitute another attention backend. For example, an H200 config selecting `flash_attention_3` fails immediately when only FlashAttention2 is installed; install compatible FlashAttention3 kernels in the reported interpreter before retrying.

**Automatically managed local Cortex-training onboarding checks the target allocation too.** When the configured GPU count exceeds the local node's visible GPUs, onboarding checks every host using the existing `ds_ssh` transport and invoking Python executable. It requires every host to respond, checks GPU counts against the hostfile slots and rejects differing target dependency versions. Set `ARCTIC_HOSTFILE` or `DSS_GATEWAY_HOSTFILE` to the same-job hostfile, or use the allocation's `/job/hostfile`. Without a hostfile, a single-node gateway gets an automatically generated local hostfile. Reference dependencies and the original checkpoint are checked on the reference node; target dependencies and writable scratch/cache locations are checked on each host without assuming `/data-fast` is shared. An explicitly supplied gateway is not owned or inspected by this command. These checks do not predict model memory use or prove successful kernel execution; sizing and validation still follow.

**Give onboarding the config; it selects all applicable checks automatically.** On P1 that means gradients followed by optimizer state. The harness derives the checkpoint from `model_name`, prepares fixture prerequisites, calibrates gates, publishes references for individual checks and the combined selection, and validates the target. The quick demo uses a four-layer fixture, 128-token sequences and one GPU, with eight nondeterministic calibration samples per arm and both gradient accumulation step (GAS) arms, GAS1 and GAS4. Eight samples is also the default for real-model onboarding. It does not establish full-model, distributed-training or maximum-sequence-length correctness.

```bash
python -m arctic_platform.correctness onboard \
  --config ./work/configs/tiny-qwen3-129m-one-h200--fp32.json
```

**No manual gateway startup or fixture flag is needed for this onboarding command.** With no Cortex host or base URL configured, onboarding selects the reduced local fixture, starts the installed `dss.ray_dss.gateway_server` on loopback for target validation, and closes its owned gateway afterwards. An existing loopback gateway can be supplied with `--base-url`; onboarding does not start or stop that external service. Hosted Cortex account endpoints cannot use this reduced-fixture onboarding mode.

**Every onboarding attempt retains diagnostics in `./work/logs/`.** The printed attempt directory contains `onboarding.log`, `report.md`, the summary, sizing results, calibration summaries, target reports and complete reference-worker logs. `DSS_CORRECTNESS_LOGDIR` overrides this root. Stage headings use vertical spacing. Queued checks use `###` markers and automatic numbering, for example `### Onboarding 1/2: Single Step Grads ###` and `### Onboarding 2/2: Single Step Optimizer ###`; inner phases retain `***` markers. Prerequisite gradient onboarding counts toward the total when needed. Model fixtures, adapters and tensor scratch remain on fast storage, never Git.

**After fixing a reported failure, repeat exactly the same command.** Small recovery records live in `./work/logs/onboarding-state/`. The harness reuses completed sizing, fixture, calibration, reference preparation and successful validation only when their inputs, source, environment and retained artifacts match. It retries failed phases and rebuilds missing or changed artifacts. A failed optimizer validation does not force another gradient calibration. Removing temporary model fixtures after an allocation ends can require rebuilding those phases; logs alone are not reusable model data.

**Inspect the generated specification and summary before accepting the gates.** Use the resolved ID printed by onboarding to find `arctic_platform/correctness/specs/<config-id>.json` and `arctic_platform/correctness/specs/<config-id>.onboarding.json`; the [layout rules](#imported-config-and-reference-locations) distinguish migrated frozen IDs from newly derived IDs. The summary records each check's result and work directory, plus any combined-reference preparation error. Compact records are generated locally and untracked under the mirrored location in `arctic_platform/correctness/reference_data`; generated tensors never belong in Git. Importing the tiny model leaves existing 8B recipes and their records untouched. A preparation or validation failure is not a numerical pass.

**To onboard one check, add `--test single-step-grads` or `--test single-step-optimizer`.** Missing fixture prerequisites are automatic. RL configs are reported as inapplicable to both supervised-reference checks; onboarding does not manufacture SFT measurements for RL. No separate specification, record directory or prerequisite command is needed for default onboarding.

**Existing Cortex-training recipe JSON can be supplied without converting it to the native client schema.** Import accepts `sub_job_configs` and preserves the original bytes and engine settings, including sampling-only recipes. P1's two supervised checks still require exactly one training sub-job and reject an inference-only recipe before model preparation; import acceptance does not make a recipe eligible for numerical onboarding. Training plus sampling remains an RL configuration, not supervised training. The first-two checks still select a reduced fixture and apply their existing correctness token cap; acceptance by the loader does not establish full-model runtime support.

**Import this catalog recipe separately from onboarding.** The existing config helper copies only the recipe bytes and prints the local path; it does not prepare models, specifications or references, or run target validation. Identical existing copies are reused; conflicting bytes are rejected.

```bash
python - <<'PY'
from arctic_platform.correctness.harness.config import local_config_path

recipe = "../cortex-training/model-catalog/models/qwen3.8-flash-next/long/h200-training-full.json"
print(local_config_path(recipe))
PY
```

**This 32-GPU catalog example demonstrates import, not numerical onboarding.** Omitted accumulation requires an automatically owned local service whose installed builder resolves FP32 accumulation and selects only local workers. Nonlocal or supplied-service defaults cannot be verified by this check. Import acceptance neither applies a precision setting nor establishes numerical support; continue with the tiny explicit-FP32 demo and preserve catalog and historical reference bytes.

**Use the recipe path present in your Cortex-training checkout.** The current recipe declares 32 H200 GPUs and 256K context. The imported path is `arctic_platform/correctness/configs/qwen3.8-flash-next/long/h200-training-full.json`; another model or sequence directory can contain the same filename without conflict. Different bytes at that exact relative path require a distinct variant. Deleted Cortex source paths have no automatic replacement; old migrated correctness-config paths remain aliases only within their original config root through `config-layout.json`.

**Calibration automatically uses the visible GPUs on one node, capped at eight concurrent HF workers.** Eight samples require one wave of eight per GAS arm when eight or more local GPUs are visible, even if sixteen GPUs are visible. A node with fewer GPUs needs more waves; this demo runs serially if only one GPU is visible. `--reference-calibration-workers` accepts an explicit override from one to eight available local GPUs. The target's GPU count does not limit calibration concurrency.

**TODO: Use every available GPU within one allocation for calibration.** Multi-node support must handle remote worker launches, node-local model and input staging, unique GPU assignments, and result and failure collection before removing the one-node limit.

**Onboarding reports elapsed time per attempted test before the final total.** Each test's time includes its sizing, calibration, reference preparation, target validation and owned cleanup performed during this invocation, including failed attempts. Reused phases contribute only the time spent checking and reusing them. Config-wide `onboarding.json` records each attempted result's `elapsed_seconds`, and `report.md` shows its formatted duration; skipped tests say `not run`. Combined reference preparation is timed separately because it serves both tests. Shared prerequisites and report writing remain in the overall total, so test durations need not sum to that total.

## Running

**Run both checks with one command in one terminal, just like onboarding.** When no Cortex host or base URL is configured, regression selects the reduced fixture, starts the installed upstream DSS gateway on an available loopback port, supplies local database/schema segments, and closes only its owned gateway after execution or failure. No separate gateway command or fixture flag is required. Omitting `--test` selects both P1 checks. Regression never generates missing references or widens reviewed tolerances.

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --out ./work/reports/both
```

Read `./work/reports/both/report.md`, `report.json`, and `full.log`. Every run ends with its total wall-clock runtime, after the report paths and any error guidance. A missing prepared reference or failed execution is not a numerical pass. The [workload selection section](#select-a-workload) includes single-check and multi-config commands, plus how to adapt the example to another supported model.

**Regression names the selected checks before running them.** The unflagged tiny demo prints `Selected tests: single-step-grads, single-step-optimizer` beneath the selected-pair count.

**The gateway log is retained under `./work/reports/both/cases/<config-id>/<group>/gateway.log`; no second terminal or manual shutdown is needed.** `./work/reports/both/full.log` records the exact gateway log path and detailed execution progress. An explicitly configured `--base-url` or `ARCTIC_CORTEX_BASE_URL` connects to an existing service and leaves it running; add `--local-cortex-fixture` for reduced-fixture checks on that loopback service. Remove stale host, base URL or PAT settings when switching back to automatic local execution. A multi-node run needs the allocation's real same-job hostfile and working `ds_ssh`. Reference preparation prints per-worker progress and complete log paths under `./work/logs`; compact reference records and large tensor scratch are separate from these durable logs.

**A gateway shutdown failure stops subsequent workloads and makes the command fail, even when the numerical comparisons passed.** Inspect `full.log` and the recorded `gateway.log` before retrying; automatic gateway shutdown is not proof that every detached Ray worker has exited.

**Ctrl-C, SIGTERM and SIGHUP stop the automatically owned gateway during startup or execution.** The command reports shutdown, waits for its owned process, and restores the previous signal handlers; repeated stop signals do not interrupt that bounded cleanup. This applies to onboarding and regression, not an explicitly configured external service.

**An owned cleanup process stops the local gateway if onboarding or regression is killed, including with SIGKILL (`kill -9`).** The command holds a private lifetime pipe; unexpected closure triggers cleanup without waiting for another command. Cleanup verifies the launch token, process identity and ownership before stopping the gateway and its token-bound local descendants, including detached Ray agents. Complete distributed worker teardown is a separate check; failures remain in `gateway-cleanup.log` and the ownership record for inspection.

**Both commands also recover a verified orphan before starting a replacement; onboarding checks before HF calibration.** Older unrecorded gateways require an exact harness launch, an orphaned parent and a retained harness log. Active commands and explicitly configured external services are never stopped. Unproven ownership or unsuccessful bounded cleanup blocks startup rather than allowing another workload to overlap surviving resources.

**Confirm the resolved config file, model source and configured sequence length printed before model work.** The quick demo must show the tiny Qwen3 snapshot and 128 configured tokens. HF calibration uses those settings independently of the gateway; output naming an old 8B configuration is not the tiny demo.

**The automatically started gateway's loopback readiness check ignores environment proxies.** No proxy changes are required for that startup check; explicitly configured service connections keep their existing behavior.

## Explicit Reference Preparation

**Preparation-only runs use an existing reviewed specification.** Default onboarding handles preparation; this advanced command explicitly prepares or reuses records without target execution. Combined and single-check workloads have different identities. See [HF reference onboarding](hf-reference-onboarding.md) for immutable publication and identity requirements.

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer
```

## Select A Workload

**A config-only selection runs both checks; a test-only selection runs every config in the supplied directory.** Catalog discovery accepts `.config` and `.json`, excludes archive/spec directories, and rejects duplicate config IDs. Configs and specs can live outside the package.

**The following commands use the files created above.** Default-all onboarding prepares individual references as well as combined records, so no additional preparation is needed for these selections.

```bash
# One config and one check.
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --test single-step-grads \
  --out ./work/reports/grads

python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --test single-step-optimizer \
  --out ./work/reports/optimizer

# One config and both checks, using the combined records prepared above.
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --out ./work/reports/config
```

**Multi-config runs use the specs generated by onboarding under each config ID.** This example has one config; onboard every additional config separately before adding it to a batch. A custom `--test-spec` is restricted to one explicit `--config`.

```bash
# One check and every externally supplied config.
python -m arctic_platform.correctness run \
  --config-dir ./work/configs \
  --test single-step-optimizer \
  --out ./work/reports/all-optimizers

# Every config and both checks.
python -m arctic_platform.correctness run \
  --config-dir ./work/configs --all-configs \
  --out ./work/reports/all
```

## Onboard Another Model

**A new model starts with a supported native AP training config and its actual checkpoint.** In the complete example, replace `repo_id` in `snapshot_download()` with the intended model's Hugging Face ID; keep `model_name` set to the returned snapshot path. Hugging Face reuses or populates `/checkpoint/huggingface/hub` automatically. Keep the checkpoint's configuration, tokenizer sidecars, and weights together; downloading only `config.json` is insufficient.

**Change the training config to describe the intended job, not just the model name.** Set GPU counts, dtype, attention, optimizer and model-specific or LoRA settings supported by the runtime. This part supports the two supervised checks, not RL configs. Check the resolved imported path and config ID; different source filenames alone do not distinguish recipes with the same derived placement. Use an explicit distinct variant for conflicting recipes. Onboarding names the specification and summary from the resolved ID, preserving migrated IDs through the manifest.

**Supply the new config to the same config-only onboarding command.** Do not copy Qwen's measured gate or records onto another model. Failed onboarding is not an admitted regression input. Compact records default to this checkout's `arctic_platform/correctness/reference_data`. For multiple nodes, set `DSS_GATEWAY_HOSTFILE` to the real allocation hostfile and provide supported model/code distribution within that allocation; a head-node path is not shared storage.

**Regression defaults to `cortex-training` and manages its owned local gateway when no endpoint is configured.** Hosted execution requires the account connection and a verified fixture publication receipt supplied with `--cortex-fixture`. Existing unauthenticated loopback DSS gateways use `--base-url http://127.0.0.1:<port> --local-cortex-fixture`; that explicit endpoint remains externally owned. `cortex` is a compatibility alias; `onprem` is not a supported CLI backend.

## Results And Diagnostics

**Every regression ends with total wall-clock runtime after the report paths and any error guidance.** The final runtime is rounded upward to whole seconds in `HH:MM:SS` format and includes gateway startup, execution, owned shutdown and report writing. Reports and complete logs default to `./work/logs`, overridden by `DSS_CORRECTNESS_LOGDIR` or `--out`; tensor scratch remains on fast storage.

**Optimizer calibration displays a `tqdm` progress bar for each GAS arm.** The bar counts eight nondeterministic HF runs by default and updates elapsed time and ETA after each completed run. Onboarding ends with total wall-clock runtime rounded upward to whole seconds in `HH:MM:SS` format in the terminal and `onboarding.log`, including when onboarding fails.

**Reports retain comparison results and complete execution errors.** Read the report directory's `report.md`, `report.json`, `batch.json`, and `full.log`. A missing reference, an execution exception, an unsafe cleanup, or an empty selection is not a numerical pass. Unsafe cleanup stops the remaining batch.

**HF preparation also reports each worker's progress and complete subprocess logs.** Start and completion or failure messages identify the arm, run and device, with elapsed time. After a worker exits, the console prints its `ref-<run>.stdout.log`, `ref-<run>.stderr.log` and `ref-<run>.returncode` paths under `./work/logs`, including successful workers. These files are separate from compact reference records; `DSS_CORRECTNESS_LOGDIR` overrides their root. Output remains captured until that subprocess exits; a start message alone is not evidence of continuing computation.

**The LM-head diagnostic changes the workload and is not a canonical result.** It removes `fused_lm_head_token_chunk_size` from a copied in-memory config without editing the source config or spec. Compatible diagnostic reference records are still required; do not relabel canonical records.

```bash
python -m arctic_platform.correctness.diagnostics list
python -m arctic_platform.correctness.diagnostics run lm_head_chunking \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --test single-step-grads
```

**CPU selftests validate the library, not a model's numerical correctness.**

```bash
python -m arctic_platform.correctness selftest
```

## LoRA Without Adapter Inputs

**The supplied config determines whether LoRA is used.** Native `training.peft` and `training.ds_worker_config.peft_config` settings are recognized automatically; a config without LoRA settings remains full fine-tuning. Local onboarding generates one shared adapter for HF calibration and target validation, then regression reuses the prepared identity. No separate adapter-generation command is needed. This route belongs to the correctness-owned gateway, not production hooks; an external gateway needs compatible correctness route installation, and hosted Cortex remains unsupported for reduced-fixture LoRA testing.

**Reference preparation generates one adapter using upstream PEFT's default initializer.** Onboarding and regression explicitly use the harness seed `1234`, a metadata-only base architecture and CPU adapter factors. The compact reference identity records the recipe, dependency versions, topology, tensor shapes/dtypes and hashes. Keep the reference environment unchanged when reusing its records; regeneration is not permission to replace measured references.

**Adapter files remain temporary, node-local artifacts under `/data-fast`, never Git inputs.** The correctness-owned local gateway transfers verified adapter bytes through Ray's allocation-local object store rather than opening the head node's path. Base-model weights, tokenizer files and installed code require the target runtime's model-distribution mechanism on each participating node. `/data-fast` is not shared storage.

**Optional pre-existing adapters are reference-preparation inputs only for the supported CLI backend.** Use `--prepare-references --peft-adapter /data-fast/path/to/adapter` with standard `adapter_config.json` and `adapter_model.safetensors` files; their exact contents become part of the reference identity. Automatic generation supports dense `target_modules` LoRA with upstream default initialization, not expert `target_parameters`, DoRA, rank patterns or initializers that modify base weights. Loading validates complete tensor coverage and exact dtype; unsupported runtime layouts fail explicitly.

## When A Step Fails


**Give the coding agent only the config path and the operation you intended.** If any manual step fails, ask:

```text
Use skills/correctness/SKILL.md to fix onboarding for
arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json,
then run both single-step-grads and single-step-optimizer.
Discover the environment, reports, logs and prepared artifacts yourself.
```

For a regression-only failure, say "run both checks" or name the desired check IDs with the config path. The skill discovers the matching command, reports, logs, retry state, model assets and references without asking you to collect them. Keep compatible prepared references and reviewed tolerances intact; do not widen a gate to make a failing comparison pass.

## Add To Repo

**For a real config addition, review and stage only explicitly selected configs and reviewed specifications after onboarding and regression succeed.** The demo does not require a commit. The paths below select only this demo's new FP32 variant; for another recipe, substitute its resolved imported path and config ID. Configs and compact reference records are untracked, not ignored; only specs require `-f`. Review the selected files without staging other local recipes or historical evidence.

```bash
git add -f -- \
  arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  arctic_platform/correctness/specs/tiny-qwen3-129m-standard-h200-training-full--fp32.json \
  arctic_platform/correctness/specs/tiny-qwen3-129m-standard-h200-training-full--fp32.onboarding.json
git diff --cached --stat
```

**Do not stage `reference_data/`, `work/`, logs, model weights, adapter tensors, optimizer snapshots or fast-storage scratch.** Compact HF records in `reference_data/` are generated local artifacts, not files to commit. Retain those records alongside the Hugging Face source cache and reduced fixtures for later regression; deleting the records requires explicit reference preparation before regression can run again.

## Cleanup

**Clean up only local checkout files after both commands succeed and you have reviewed their results.** Remove the duplicate original config under `./work/configs/`, keeping the imported config, reviewed specification and local compact references for later regression. Allocation scratch is left untouched and will be removed automatically on job reset.

```bash
rm ./work/configs/tiny-qwen3-129m-one-h200--fp32.json
```

**Optionally remove these two explicit directories once you no longer need the diagnostic evidence.** `./work/logs/correctness-demo/` contains this demo's onboarding logs and retry state; `./work/reports/both/` contains its regression reports. Keep both while investigating a failure; deleting retry state may require preparation again on a later onboarding attempt. Run from the checkout root. These literal paths do not use environment variables or wildcards and do not select other runs' log directories or reusable regression inputs.

```bash
rm -r ./work/logs/correctness-demo/
rm -r ./work/reports/both/
```

**Use the retained imported config for future runs after cleanup.**

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --out ./work/reports/both
```
