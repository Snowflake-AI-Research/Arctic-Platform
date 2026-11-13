# Running The Seven Correctness Tests

**Run these commands from an already prepared, allocated head node.** The library does not allocate machines, install a runtime, download a missing reference during regression, or modify reviewed numerical gates. Onboarding and regression can start and stop their owned installed local DSS gateway. The correctness backend is `cortex-training`; `cortex` remains a compatibility alias.

**This checkout registers seven correctness tests, including two reinforcement-learning (RL) comparisons.** `single-step-grads`, `single-step-optimizer`, `checkpoint-resume-loss`, `inference-checkpoint-loss`, `e2e-sft-train-validate`, `rl-weight-sync` and `rl-router-replay` are the registered IDs. The first five tests retain their existing workload identities and prerequisites. The two RL tests require narrowly scoped, opt-in trainer/sampler evidence: actual tensor-send and receiver-write records for weight synchronization, and received routing plus forward/backward router invocations for replay. CLI discovery and help do not import these observers; ordinary training and sampling calls leave them disabled.

**Reduced first-two fixtures still need compatible native model loading.** A positive prediction-layer count that the installed loader cannot authenticate is an explicit prerequisite failure, not permission to drop layers or add production hooks. Later-only selection does not inherit this reduced-fixture workflow.

## Check-Specific Prerequisites

**Registration is not proof that a target runtime can execute a check.** Install the compatible training/reference environment and the separately installed DSS gateway on the selected allocation. Inference additionally needs a compatible sampling runtime; CUDA and the configured attention kernels are separate prerequisites. Each participating host must import the admitted sources and read identical model, tokenizer and adapter assets. Use supported same-allocation distribution and node-local fast scratch; a head-node path is not shared storage.

| Check | Required execution evidence |
| --- | --- |
| `single-step-grads`, `single-step-optimizer` | Compatible reduced SFT fixture, current HF records for the selected workload and GAS arms, BF16 model weights, verified FP32 accumulation and complete correctness-owned parameter coverage. Optimizer comparison additionally requires decodable first-step state. |
| `checkpoint-resume-loss` | The same ordered batches and training identity in uninterrupted and resumed jobs; complete resumable state, authenticated saved checkpoint identity and supported restore/create-time initialization. A weights-only export is not resumable optimizer/scheduler state. This check compares AP with AP, not HF. |
| `inference-checkpoint-loss` | Supported training-to-sampling checkpoint export, validated weights/config/tokenizer schema, identical saved weights available on each sampling node, supported sampling payload/runtime and matching answer-token coverage. Successful export alone does not establish sampler initialization or loss agreement. |
| `e2e-sft-train-validate` | SFT configuration, matching initial weights or adapters, the same ordered 100-step replay and held-out validation inputs, compatible HF trajectory records and the target training runtime. Full-model mode requires its separately admitted local snapshot; reduced first-two records cannot substitute for an E2E record. |
| `rl-weight-sync` | A native local training/sampling route, one completed RL optimizer step, collective trainer inventory, every sampling rank's received-tensor records and actual destination-write or LoRA-install proof. FP8 replay needs a supported independent updater. Sender checksums alone are insufficient. |
| `rl-router-replay` | A supported MoE model with identifiable routers, sampler routing capture, trainer RX-cache observations, immutable successful-backward records for every actual rank and model call, and model-bound router-layer inventory. Routing and captured-token coverage must match exactly. |

**Later-only selection retains its configured supported precision and ordinary training lifecycle.** The first-two BF16/FP32 admission and participation/optimizer capture do not become requirements for a selected later check. Do not change the recipe, widen a reviewed gate or add unrelated production changes to obtain a pass. The RL tests additionally require their enabled evidence observers and compatible native training/sampling runtimes.

**Missing capabilities remain separate non-passes.** Config-only onboarding attempts applicable checks independently; it does not infer later prerequisites from a successful first-two fixture. Unsafe cleanup stops subsequent workloads. CPU controls, command help and registration are not numerical or GPU evidence.

**Automatic regression startup is narrower than registration.** The current local startup helper recognizes an explicit first-two-only selection or a resume-only selection. A default mixed selection, inference or E2E still needs the admitted execution route; listing seven IDs does not install or start missing services. Keep the config-only default example and report any missing route as a non-pass rather than silently narrow the selected checks.

## Prerequisites

- Install the project and its compatible PyTorch, DeepSpeed, Transformers, and PEFT dependencies in the selected interpreter.
- Install with `python -m pip install -e '.[correctness]'`. The correctness extra supplies training and reference dependencies plus `pytest` and `pytest-instafail` for CPU selftests; the separate `testing` extra is not required for this workflow. Install CUDA and model-specific attention kernels separately.
- For inference or RL target execution, install `python -m pip install -e '.[correctness,rl]'` on the GPU allocation. The `rl` extra adds the in-tree Arctic Inference sampler and its pinned vLLM dependency; do not require that GPU sampling stack for CPU-only selftests. The separately installed DSS service must import compatible training and sampling sources on every participating node.
- Make the allocation's GPU family and available devices match the supplied configuration. For multiple nodes, provide the allocated hostfile through `DSS_GATEWAY_HOSTFILE` and working allocation-local `ds_ssh`.
- Make the reviewed model and tokenizer assets readable at the paths recorded in the test specification.
- Keep generated fixtures, adapter packs, optimizer snapshots and calibration scratch on allocation-local fast storage. The harness uses `/data-fast` defaults; `AP_CORRECTNESS_TMPDIR` can select another directory under that mount. Temporary files disappear with the allocation.
- Choose persistent locations for the checkout, downloaded source model, native config and final reports. Supply the native config (`.config` or `.json`) to onboarding; it creates the specification and references. Regression requires those reviewed inputs. No real configurations, specifications or measurements are bundled.

## Checks

**Seven correctness tests are registered.** Config-only selection includes every applicable check. An explicit check ID selects that workload without changing its inputs, precision or reviewed gate.

| Check ID | Comparison |
| --- | --- |
| `single-step-grads` | Named per-parameter gradient L2 norms against the prepared single-GPU Hugging Face reference. |
| `single-step-optimizer` | Step-one parameter-update residual, Adam first-moment, and Adam second-moment L2 norms, with matching parameter names and shapes. |
| `checkpoint-resume-loss` | Uninterrupted and checkpoint-resumed training losses, with identical batch order and verified restored state. |
| `inference-checkpoint-loss` | Answer-token cross entropy from the training zone and a sampling zone serving its saved checkpoint. |
| `e2e-sft-train-validate` | Final SFT training loss and held-out validation loss against the prepared HF reference; both trajectories remain diagnostic evidence. |
| `rl-weight-sync` | Trained tensor bytes and element inventory against every sampling rank's received tensors and actual installed destinations. |
| `rl-router-replay` | Sampler routing against trainer-received routing, packed-call routing and every expected MoE router's forward/recompute indices. |

**Optimizer norm agreement is not elementwise agreement.** Equal L2 norms can hide a sign change, permutation, or directional error. Regression uses the reviewed criterion and calibrated gate stored in the prepared record; regression never widens that gate.

## Query Onboarding Status

**`list` inspects configuration/test metadata without running workloads.** It reads existing recipes, specifications, compact Hugging Face (HF) reference records and onboarding summaries; it does not import external configs, publish records, start a gateway or inspect GPUs. Unreviewed recipes remain visible, including configs whose specification is missing.

```bash
python -m arctic_platform.correctness list
python -m arctic_platform.correctness list --config <ID-or-path> --test single-step-grads
python -m arctic_platform.correctness list --config <ID-or-path> --json
```

**The Python API returns one structured result per selected config/test pair.** `onboarding_status(RunRequest(...))` uses the same IDs and recipe parser as regression. Omit the config selector to inspect every discovered recipe, including unreviewed configs. Repeated test IDs are collapsed; unknown IDs and conflicting config selectors are errors. External paths are read in place; when explicit recipe/path hardware permits deriving the import identity, the query uses that identity without creating a local config or probing hardware. `--config-dir` changes recipe discovery and `--reference-records-dir` changes the inspected reference root; reviewed specifications and summaries remain under this package's `specs/`.

```python
from dataclasses import asdict
from arctic_platform.correctness.api import RunRequest, onboarding_status

rows = onboarding_status(RunRequest(config="<ID-or-path>", tests=("single-step-grads",)))
data = [asdict(row) for row in rows]
```

**Metadata validity is not runtime readiness or a passing validation.** Each result separates `spec`, `references` and `historical`. Specs are checked against the current config checksum, case shapes, attention and per-test coverage; missing, stale and malformed artifacts include a reason. Current reference filenames are checked for schema, payload integrity, config/check identity, frozen gradient gate, case shape and complete scalar measurements. `metadata_valid` does not authenticate the complete reference workload, installed dependencies, model/adapter bytes, wider placements or current target capabilities; regression performs its own admission checks. Legacy content-addressed records are not authenticated by this query and require regression's full identity check. Selecting gradients alone inspects `gradients-only` records; selecting gradients with an applicable optimizer check inspects `with-optimizer` records.

**Recorded onboarding outcomes remain historical.** `historical.outcome` preserves the recorded outcome, while `historical.state` distinguishes matching current per-test spec identity (`current_spec`), stale identity, older unbound evidence, missing evidence and malformed evidence. A matching identity does not authenticate today's runtime or options, and a recorded success cannot substitute for missing references. A missing summary does not invalidate an imported reviewed spec or change regression selection. JSON output uses schema `correctness-onboarding-status-v1`; querying missing artifacts succeeds and reports their status, while invalid selectors return exit code 2.

**Rerun onboarding when compatible references are missing.** The config-only `onboard --config <ID-or-path>` workflow prepares the selected references automatically, including the combined gradient/optimizer records; no separate operator preparation step is needed.

## Setup

**This quick demo uses `axolotl-ai-co/tiny-qwen3-129m`, a 129M-parameter Qwen3 model, with 128-token sequences on one exclusively allocated H200 GPU.** Activate a Python 3.12+ environment with working CUDA, FlashAttention-3 kernels and the installed upstream DSS GPU gateway first. The `correctness` extra installs the Cortex client, training and Hugging Face reference dependencies, and CPU selftest tools; it does not install the separate DSS service, CUDA or model-specific attention kernels. Use the existing `/checkpoint/huggingface` cache; Hugging Face downloads missing model files there automatically, requiring network access and sufficient cache space. The gateway uses the allocation's Ray runtime, so do not run on a shared cluster. No fresh GPU verification is established by the controls for gateway, progress and timing changes.

**Only the first two checks require BF16 model weights and verified FP32 gradient accumulation.** An explicit `data_types.grad_accum_dtype: "fp32"` is accepted without runtime-default inspection. An omitted value requires the automatically owned local service to verify the installed engine's resolved default. No recipe override is applied. Explicit later-only selection retains the config's supported dtype and does not arm first-two participation or optimizer-state capture. The demo's `--fp32` suffix denotes accumulation, not model dtype.

**Start in the directory where you want to keep the checkout and results, then copy the blocks in order.** No pre-existing config, specification or measurement is needed. After cloning, remain in the Arctic Platform checkout: the demo config and final reports use its Git-ignored `./work/` directory. Model weights stay in `/checkpoint/huggingface`, not the checkout. A specification is generated JSON describing the reduced fixture, input batches, applicable checks and calibrated tolerances. Reference records contain compact measurements from independent single-GPU Hugging Face runs, not tensor payloads. Records must be inside the executing library's checkout, but that checkout need not be on temporary storage. Generated fixtures and calibration scratch use temporary allocation-local defaults; retain printed preparation logs separately if the allocation will disappear.

```bash
set -e
mkdir -p ./correctness-no-debug
cd ./correctness-no-debug
git clone --single-branch --branch stas/correctness-testing-rl-minimal-debug \
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
export AP_CORRECTNESS_TMPDIR="$(mktemp -d /data-fast/tmp/correctness-demo.XXXXXXXX)"
export AP_CORRECTNESS_LOGDIR="$PWD/work/logs/correctness-demo"
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

**The demo uses a new explicit FP32 variant for the first-two comparisons.** The explicit accumulation setting does not change the BF16 model dtype. Keep the historical recipe, specifications and reference bytes unchanged; the new variant needs compatible preparation. Default-all selection also attempts applicable later checks, but this training-only config does not supply a sampling runtime or establish resume/export/full-SFT capabilities.

## Onboarding

**Onboarding checks prerequisites before loading model weights or creating a reduced fixture.** The supplied config and selected applicable tests determine the required attention backend, LoRA dependencies and whether a sampling runtime is needed. The check reports the executing Python interpreter, verifies imports and CUDA/GPU compatibility, and checks access to the model source and writable cache, scratch, specification and reference-record locations. It does not install packages, download weights or substitute another attention backend. For example, an H200 config selecting `flash_attention_3` fails immediately when only FlashAttention2 is installed; install compatible FlashAttention3 kernels in the reported interpreter before retrying.

**Automatically managed local Cortex-training onboarding checks the target allocation too.** When the configured GPU count exceeds the local node's visible GPUs, onboarding checks every host using the existing `ds_ssh` transport and invoking Python executable. It requires every host to respond, checks GPU counts against the hostfile slots and rejects differing target dependency versions. Set `ARCTIC_HOSTFILE` or `DSS_GATEWAY_HOSTFILE` to the same-job hostfile, or use the allocation's `/job/hostfile`. Without a hostfile, a single-node gateway gets an automatically generated local hostfile. Reference dependencies and the original checkpoint are checked on the reference node; target dependencies and writable scratch/cache locations are checked on each host without assuming `/data-fast` is shared. An explicitly supplied gateway is not owned or inspected by this command. These checks do not predict model memory use or prove successful kernel execution; sizing and validation still follow.

**Give onboarding only the config; all applicable checks are attempted independently.** No test selector or separate reference-preparation command is required for default-all onboarding. The first-two checks retain their reduced fixture workflow. Later checks need resumable state for checkpoint resume, checkpoint export plus a compatible sampler for inference, or full ordered SFT inputs and a prepared HF record for E2E. A missing capability remains that check's reported non-pass and does not suppress independent selected checks. The config-only demo is not proof that all seven tests or their numerical comparisons passed.

```bash
python -m arctic_platform.correctness onboard \
  --config ./work/configs/tiny-qwen3-129m-one-h200--fp32.json
```

**No separate gateway command is required for the supported automatically managed local route.** The harness owns startup and cleanup when no endpoint is supplied. That ownership does not supply missing resumable-state, sampler or model-loading capabilities. An explicitly supplied service remains externally owned and must satisfy the selected check's admission.

**Every onboarding attempt retains diagnostics in `./work/logs/`.** The printed attempt directory contains `onboarding.log`, reports, summaries and complete worker logs; `AP_CORRECTNESS_LOGDIR` overrides the root. Queue numbering follows the actual selected checks and fixture prerequisites, not a fixed `1/2` and `2/2`. Each result and elapsed time remains independent; unsafe cleanup stops later work. Fixtures, adapters and tensor scratch remain on fast storage.

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

**Run all applicable checks with one config-only command in one terminal.** Omitting `--test` selects the applicable set from the seven-test registry, not only the two reduced-fixture checks. No separate gateway command is needed for a supported automatically managed local route. Each check requires its own prepared identity and target capabilities; missing prerequisites remain separate non-passes. Regression never prepares missing references, changes precision or widens reviewed gates.

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full--fp32.json \
  --out ./work/reports/both
```

Read `./work/reports/both/report.md`, `report.json`, and `full.log`. Every run ends with its total wall-clock runtime, after the report paths and any error guidance. A missing prepared reference or failed execution is not a numerical pass. The [workload selection section](#select-a-workload) includes single-check and multi-config commands, plus how to adapt the example to another supported model.

**Regression names the selected checks before running them.** The config-only demo reports every applicable selected check, not a fixed two-check list.

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

**A config-only selection runs every applicable registered test; a test-only selection runs that test across discovered configs.** Use repeated `--test` flags for an intentional subset. The RL tests require their opt-in evidence and a supported native training/sampling route. Discovery accepts `.config` and `.json`, excludes archive/spec directories and rejects duplicate config IDs.

**Default-all onboarding prepares compatible references for each independently successful selected workload.** Retain every check's result separately, including failed or missing later-check prerequisites. The examples below are optional narrower regression selections, not additional steps required by config-only onboarding. The explicit reference-preparation command above remains an advanced preparation-only operation.

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

# Optional: both first-two checks with their compatible combined reference.
python -m arctic_platform.correctness run \
  --test single-step-grads --test single-step-optimizer \
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

# Every config and all applicable registered checks.
python -m arctic_platform.correctness run \
  --config-dir ./work/configs --all-configs \
  --out ./work/reports/all
```

## Onboard Another Model

**A new model starts with a supported native AP training config and its actual checkpoint.** In the complete example, replace `repo_id` in `snapshot_download()` with the intended model's Hugging Face ID; keep `model_name` set to the returned snapshot path. Hugging Face reuses or populates `/checkpoint/huggingface/hub` automatically. Keep the checkpoint's configuration, tokenizer sidecars, and weights together; downloading only `config.json` is insufficient.

**Change the training config to describe the intended job, not just the model name.** Set supported GPU counts, dtype, attention, optimizer and adapter settings for the selected test. The first-two and E2E reference comparisons require SFT configs; the two RL tests require RL training and sampling sub-jobs. Import preserves recipe bytes and identity, while each numerical test has separate execution admission. Use a distinct variant for a different workload and retain existing reviewed references.

**Supply the new config to the same config-only onboarding command.** Do not copy Qwen's measured gate or records onto another model. Failed onboarding is not an admitted regression input. Compact records default to this checkout's `arctic_platform/correctness/reference_data`. For multiple nodes, set `DSS_GATEWAY_HOSTFILE` to the real allocation hostfile and provide supported model/code distribution within that allocation; a head-node path is not shared storage.

**Regression defaults to `cortex-training` and manages its owned local gateway when no endpoint is configured.** Hosted execution requires the account connection and a verified fixture publication receipt supplied with `--cortex-fixture`. Existing unauthenticated loopback DSS gateways use `--base-url http://127.0.0.1:<port> --local-cortex-fixture`; that explicit endpoint remains externally owned. `cortex` is a compatibility alias; `onprem` is not a supported CLI backend.

## Results And Diagnostics

**Every regression ends with total wall-clock runtime after the report paths and any error guidance.** The final runtime is rounded upward to whole seconds in `HH:MM:SS` format and includes gateway startup, execution, owned shutdown and report writing. Reports and complete logs default to `./work/logs`, overridden by `AP_CORRECTNESS_LOGDIR` or `--out`; tensor scratch remains on fast storage.

**Optimizer calibration displays a `tqdm` progress bar for each GAS arm.** The bar counts eight nondeterministic HF runs by default and updates elapsed time and ETA after each completed run. Onboarding ends with total wall-clock runtime rounded upward to whole seconds in `HH:MM:SS` format in the terminal and `onboarding.log`, including when onboarding fails.

**Reports retain comparison results and complete execution errors.** Read the report directory's `report.md`, `report.json`, `batch.json`, and `full.log`. A missing reference, an execution exception, an unsafe cleanup, or an empty selection is not a numerical pass. Unsafe cleanup stops the remaining batch.

**HF preparation also reports each worker's progress and complete subprocess logs.** Start and completion or failure messages identify the arm, run and device, with elapsed time. After a worker exits, the console prints its `ref-<run>.stdout.log`, `ref-<run>.stderr.log` and `ref-<run>.returncode` paths under `./work/logs`, including successful workers. These files are separate from compact reference records; `AP_CORRECTNESS_LOGDIR` overrides their root. Output remains captured until that subprocess exits; a start message alone is not evidence of continuing computation.

**The LM-head diagnostic changes the workload and is not a canonical result.** It sets `model_options["lm_head.token_chunk_size"]` to `"disabled"` in a copied in-memory config, including an existing nested `model_options["lm_head"]["token_chunk_size"]`, without editing the source config or spec. The explicit option takes precedence over legacy settings and prevents the runner from restoring the spec's chunk size. Compatible diagnostic reference records are still required; do not relabel canonical records.

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
