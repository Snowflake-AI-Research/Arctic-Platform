# Running The Seven Correctness Tests

**Run these commands from the repository root on a prepared compute node.** Use the Python environment in which Arctic Platform and the required model runtimes are installed. The operator provides enough nodes, CPU memory, GPUs, GPU memory, and fast storage for the selected configuration. The harness does not allocate machines or install a GPU runtime. First-two onboarding and regression can manage their own installed local DSS gateway; later checks require their supported target runtime. Multi-node tests must use nodes within one allocation, never nodes from separate jobs.

**This checkout registers seven correctness tests, including two reinforcement-learning (RL) comparisons.** `single-step-grads`, `single-step-optimizer`, `checkpoint-resume-loss`, `inference-checkpoint-loss`, `e2e-sft-train-validate`, `rl-weight-sync` and `rl-router-replay` are the registered IDs. The first five tests retain their existing workload identities and prerequisites. The two RL tests require narrowly scoped, opt-in trainer/sampler evidence: actual tensor-send and receiver-write records for weight synchronization, and received routing plus forward/backward router invocations for replay. CLI discovery and help do not import these observers; ordinary training and sampling calls leave them disabled.

## Before Running

**A regression needs a reviewed configuration and its prerequisites.** Confirm:

- The selected configuration has a reviewed specification under `arctic_platform/correctness/specs/`, or supplied with `--test-spec` for one explicit config.
- The installed runtime supports the model, attention implementation, training topology, and checkpoint operations required by the test.
- Required model files and checkpoints are accessible from this allocation.
- Hugging Face reference measurements are already prepared for tests that need them. Regression never creates or replaces a missing baseline.
- `/data-fast/tmp` is writable for generated fixtures and calibration scratch. Keep model weights and checkpoints out of Git; the demo uses `/checkpoint/huggingface` for model weights and the Git-ignored `./work/` directory inside the checkout for its native config and final reports.
- The target runtime is installed. The default backend is `cortex-training`; supervised first-two regression starts its local gateway automatically when no Cortex host or base URL is configured. An explicitly configured endpoint must already be available.

**Connection defaults come from environment variables; command-line options override them.** For an existing direct development gateway, for example:

```bash
export ARCTIC_CORTEX_BASE_URL=http://127.0.0.1:8003
export ARCTIC_CORTEX_DATABASE=dev
export ARCTIC_CORTEX_SCHEMA=dev
```

For an authenticated Cortex endpoint use `ARCTIC_CORTEX_HOST`, `ARCTIC_CORTEX_DATABASE`, `ARCTIC_CORTEX_SCHEMA`, and `ARCTIC_CORTEX_PAT` instead of a direct URL. Supply the token through a protected environment or `--pat-file`, not through a configuration file or a pasted command. The operator-facing backend is `cortex-training`; `cortex` is a compatibility alias and `onprem` is rejected. For a local multi-node gateway, provide the same-job hostfile through `DSS_GATEWAY_HOSTFILE` and working allocation-local `ds_ssh`.

**Large artifacts and small diagnostics have separate defaults.** Temporary model fixtures and tensor scratch use `/data-fast/tmp`; `AP_CORRECTNESS_TMPDIR` can select another directory under `/data-fast`. Small reports and complete logs use `./work/logs`, overridden by `AP_CORRECTNESS_LOGDIR` or a regression `--out` directory. Compact reference JSON files default to the locally retained `arctic_platform/correctness/reference_data` and remain untracked, not Git-ignored, until explicitly selected for publication; `--reference-records-dir` must select a directory inside the checkout, not scratch storage.

## Check-Specific Prerequisites

**Registration is not proof that a target runtime can execute a test.** Each participating host must import compatible training and sampling sources and read identical model, tokenizer and adapter assets. Use supported same-allocation distribution and node-local fast scratch; a head-node path is not shared storage. Reduced first-two fixtures still require authenticated native prediction-weight loading when configured prediction layers are present. Missing loader capability is a prerequisite error, not permission to drop layers or add an RL observer workaround.

| Check | Required execution evidence |
| --- | --- |
| `single-step-grads`, `single-step-optimizer` | Compatible reduced SFT fixture, current HF records for the selected workload and GAS arms, BF16 model weights, verified FP32 accumulation and complete correctness-owned parameter coverage. Optimizer comparison additionally requires decodable first-step state. |
| `checkpoint-resume-loss` | The same ordered batches and training identity in uninterrupted and resumed jobs; complete resumable state, authenticated saved checkpoint identity and supported restore/create-time initialization. A weights-only export is not resumable optimizer/scheduler state. This check compares AP with AP, not HF. |
| `inference-checkpoint-loss` | Supported training-to-sampling checkpoint export, validated weights/config/tokenizer schema, identical saved weights available on each sampling node, supported sampling payload/runtime and matching answer-token coverage. Successful export alone does not establish sampler initialization or loss agreement. |
| `e2e-sft-train-validate` | SFT configuration, matching initial weights or adapters, the same ordered 100-step replay and held-out validation inputs, compatible HF trajectory records and the target training runtime. Full-model mode requires its separately admitted local snapshot; reduced first-two records cannot substitute for an E2E record. |
| `rl-weight-sync` | A native local training/sampling route, one completed RL optimizer step, collective trainer inventory, every sampling rank's received-tensor records and actual destination-write or LoRA-install proof. FP8 replay needs a supported independent updater. Sender checksums alone are insufficient. |
| `rl-router-replay` | A supported mixture-of-experts (MoE) model with identifiable routers, sampler routing capture, trainer received-routing-cache observations, immutable successful-backward records for every actual rank and model call, and model-bound router-layer inventory. Routing and captured-token coverage must match exactly. |

**Later-only selection retains its configured supported precision and ordinary training lifecycle.** The first-two BF16/FP32 admission and participation/optimizer capture do not become requirements for a selected later check. Do not change the recipe, widen a reviewed gate or add unrelated production changes to obtain a pass. The RL tests additionally require their enabled evidence observers and compatible native training/sampling runtimes.

**Missing capabilities remain separate non-passes.** Config-only onboarding attempts applicable checks independently; it does not infer later prerequisites from a successful first-two fixture. Unsafe cleanup stops subsequent workloads. CPU controls, command help and registration are not numerical or GPU evidence.

**Automatic regression startup is narrower than registration.** The [local startup helper](../harness/execution.py) recognizes an explicit first-two-only selection, a resume-only selection or a supervised E2E-only selection. Default mixed regression selections, inference and RL still need an admitted execution route; listing seven IDs does not install or start missing services. Config-only onboarding selects the execution mode independently for each test.

**Local supervised E2E automatically uses the original full model.** `onboard --test e2e-sft-train-validate` and `run --test e2e-sft-train-validate` select the admitted original checkpoint without requiring `--local-gateway-full-model`. With no configured endpoint, the harness owns gateway startup and cleanup; an explicit loopback `--base-url` remains externally owned. Gradient and optimizer children, including the automatic specification prerequisite, keep their reduced fixtures. An explicit `--local-cortex-fixture` or published `--cortex-fixture` retains fixture mode; hosted, on-prem and RL execution are unchanged.

**Automatic selection does not waive full-model prerequisites.** Stage the pinned snapshot and tokenizer under `/data-fast`, set `HF_HUB_CACHE` to that cache, and retain `HF_HUB_OFFLINE=1` and `DSS_ALLOW_REMOTE_MODELS=1`. Multi-node execution still needs the same-job hostfile described below. Qwen3-0.6B's original single-file safetensors checkpoint is admitted by its pinned weight digest and complete tensor schema. Missing admission, cache assets or compatible trajectory records fail explicitly; the harness never falls back to a reduced model.

```bash
python -m arctic_platform.correctness onboard \
  --config ../cortex-training/model-catalog/models/qwen3-0.6b/standard/h200-training-full.json \
  --test e2e-sft-train-validate
```

**RL observers record existing production operations without replacing them.** Tensor sends, receiver writes, optimizer steps and router execution stay on their ordinary paths. Descriptive one-line `maybe_...` calls delegate opt-in checks and evidence capture to `arctic_platform/debug/`; disabled calls preserve ordinary behavior. These observers do not authorize unrelated startup, allocation, model-loading or cleanup workarounds.

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

## Setup

**This quick demo uses `axolotl-ai-co/tiny-qwen3-129m` and 128-token sequences on one exclusively allocated H200 GPU in the seven-test checkout.** The first-two comparisons use a reduced supervised fixture; later tests retain their own prerequisites, and the training-only demo does not supply a sampling service or RL workload. Activate a Python 3.12+ environment with working CUDA, FlashAttention-3 kernels and the installed upstream DSS GPU gateway first. The `correctness` extra installs the Cortex client, training and Hugging Face (HF) reference dependencies, and CPU selftest tools; it does not install the separate DSS service, CUDA, model-specific kernels, or the sampling runtime needed by later tests. Hugging Face reuses `/checkpoint/huggingface` and downloads missing model files there automatically.

**Start in the directory where you want to keep the checkout and results, then copy the blocks in order.** After cloning, remain in the Arctic Platform checkout: the demo config and final reports use its Git-ignored `./work/` directory. Model weights stay in `/checkpoint/huggingface`, not the checkout. Specifications describe the reduced model fixture, input batches, applicable checks and calibrated tolerances; compact reference records contain independent single-GPU Hugging Face measurements. Onboarding creates both automatically. Compact records must be inside the executing library's checkout, but that checkout need not be on temporary storage. Generated fixtures and calibration tensor scratch use allocation-local fast storage and may disappear when the node is recycled. Onboarding automatically retains logs and recovery metadata under the checkout's `./work/logs/`.

```bash
set -e
mkdir -p ./correctness-demo
cd ./correctness-demo
git clone --single-branch --branch stas/correctness-testing \
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
Path("./work/configs/tiny-qwen3-129m-one-h200.json").write_text(
    json.dumps(config, indent=2) + "\n"
)
PY
```

**The config uses the remote Cortex protocol, matching the correctness commands' `cortex-training` backend.** The loopback URL on port 1 is a non-serving placeholder required by the client schema. Onboarding and regression replace that placeholder with their automatically started gateway's actual URL. Do not start a server on port 1 or export the placeholder as `ARCTIC_CORTEX_BASE_URL`; an explicit endpoint selects an externally managed service. `--backend onprem` is rejected.

**The example uses full fine-tuning, not low-rank adaptation (LoRA).** Native `training.peft` or `training.ds_worker_config.peft_config` enables automatic reference generation and the admitted local first-two shared-adapter route. That route does not establish hosted or later-test LoRA support. See [LoRA without adapter inputs](#lora-without-adapter-inputs).

### Imported Config And Reference Locations

**External configs are imported byte for byte into this checkout before onboarding or running.** The harness uses the local copy for configuration identity, specifications and execution; it does not rewrite recipe settings or automatically commit the import. The original external file is no longer needed after import. An identical existing copy is reused; conflicting bytes at the destination cause an error without overwriting the existing recipe.

**Catalog recipes keep their complete path below `cortex-training/model-catalog/models/`.** Replace that prefix with `arctic_platform/correctness/configs/`, preserving every remaining directory and filename. For example, `qwen3.5-2b/standard/h200-inference-lora.json` retains that relative path even though the recipe is inference-only. Import recipe JSON, not the catalog's `model.json` metadata.

**Noncatalog configs use `<model-name>/<standard|long>/<gpu>-<training|inference>-<full|lora>.json`.** The model name is the lowercase model basename without the repository owner; an HF cache snapshot uses the model component, not the snapshot hash. Only an explicit `max_tokens_per_mb >= 262144` selects `long`, including a training sub-job's `mb_spec` or native `training.ds_worker_config.mb_spec`; missing or smaller budgets select `standard`, regardless of `max_seq_len` or runtime defaults. Training takes precedence over sampling when both exist, and `lora` comes from the selected job's nonempty parameter-efficient fine-tuning (PEFT) settings; otherwise use `full`. Resolve the GPU from explicit recipe hardware, an existing accelerator directory or filename, or the executing GPU; missing hardware identity is not permission to guess. The demo therefore imports as `configs/tiny-qwen3-129m/standard/h200-training-full.json`, regardless of its source filename.

**Reference folders mirror the imported config path without its extension.** The demo's compact records use `arctic_platform/correctness/reference_data/tiny-qwen3-129m/standard/h200-training-full/<check-id>/...`. `--reference-records-dir` changes only the root prefix. Existing check-specific subdirectories and measurement filenames remain unchanged; see the [reference layout](hf-reference-onboarding.md#configuration-and-reference-layout) for examples.

**Existing config IDs remain frozen through `arctic_platform/correctness/config-layout.json`, even after their files move.** Specifications remain `specs/<config-id>.json` and summaries remain `specs/<config-id>.onboarding.json`; use the ID printed by the command or `list`, not a guess based on the new folder. Historical measurement bytes and recorded identities are not rewritten or remeasured. Distinct old recipes sharing a derived placement retain descriptive `--<variant>` filename suffixes. New imports with conflicting bytes require an explicit distinct variant, not an overwrite.

## Onboarding

**Onboarding checks prerequisites before loading model weights or creating a reduced fixture.** The supplied config and selected applicable tests determine the required attention backend, LoRA dependencies and whether a sampling runtime is needed. The check reports the executing Python interpreter, verifies imports and CUDA/GPU compatibility, and checks access to the model source and writable cache, scratch, specification and reference-record locations. It does not install packages, download weights or substitute another attention backend. For example, an H200 config selecting `flash_attention_3` fails immediately when only FlashAttention2 is installed; install compatible FlashAttention3 kernels in the reported interpreter before retrying.

**Automatically managed local Cortex-training onboarding checks the target allocation too.** When the configured GPU count exceeds the local node's visible GPUs, onboarding checks every host using the existing `ds_ssh` transport and invoking Python executable. It requires every host to respond, checks GPU counts against the hostfile slots and rejects differing target dependency versions. Set `ARCTIC_HOSTFILE` or `DSS_GATEWAY_HOSTFILE` to the same-job hostfile, or use the allocation's `/job/hostfile`. Without a hostfile, a single-node gateway gets an automatically generated local hostfile. Reference dependencies and the original checkpoint are checked on the reference node; target dependencies and writable scratch/cache locations are checked on each host without assuming `/data-fast` is shared. An explicitly supplied gateway is not owned or inspected by this command. These checks do not predict model memory use or prove successful kernel execution; sizing and validation still follow.

**Give onboarding only the config; all applicable tests are attempted independently.** No test selector or separate reference-preparation command is required for default-all onboarding. The harness derives the checkpoint from `model_name`, prepares fixture prerequisites, calibrates gates, publishes individual and combined-selection references, and validates each target. The first-two tests retain their reduced fixture workflow. Later tests need resumable state for checkpoint resume, checkpoint export plus a compatible sampler for inference, full ordered SFT inputs and a prepared HF record for E2E, or live training/sampling evidence for RL. A missing capability remains that test's reported non-pass and does not suppress independent selected tests. The config-only demo is not proof that all seven numerical comparisons passed.

**The first-two demo uses a four-layer fixture, 128-token sequences and one target GPU.** Calibration uses eight nondeterministic samples per gradient accumulation step (GAS) arm, GAS1 and GAS4; eight samples is also the default for real-model onboarding. Reduced-fixture results do not establish full-model, distributed-training or maximum-sequence-length correctness.

**HF calibration uses the visible GPUs on one node, with at most eight concurrent workers by default.** Eight samples run in one wave of eight per GAS arm when eight or more local GPUs are visible, even with sixteen visible GPUs. With only one local GPU, the eight samples run serially. Use `--reference-calibration-workers` to select one to eight available local GPUs explicitly; the target's GPU count does not select calibration concurrency. Multi-node HF calibration is future work; target execution can still use its configured same-allocation topology.

```bash
python -m arctic_platform.correctness onboard \
  --config ./work/configs/tiny-qwen3-129m-one-h200.json
```

**The first two reduced-fixture checks manage local validation automatically.** Without a configured Cortex host or base URL, onboarding selects the local fixture, starts the installed `dss.ray_dss.gateway_server` on loopback for validation and closes its owned gateway afterwards. No manual fixture flag or gateway command is needed. An explicitly supplied loopback gateway is left running; a hosted account endpoint cannot accept this reduced-fixture workflow. This automation does not add target capabilities for later checks.

**Every onboarding attempt retains diagnostics in `./work/logs/`.** The printed attempt directory contains `onboarding.log`, `report.md`, the summary, sizing results, calibration summaries, target reports and complete reference-worker logs. `AP_CORRECTNESS_LOGDIR` overrides this root. Stage headings use vertical spacing and `***` prefixes. Queued tests are numbered against the actual selected queue, including required fixture prerequisites; the denominator is not fixed at two. Model fixtures, adapters and tensor scratch remain on fast storage, never Git.

**Repeat the same command after resolving a reported failure.** Recovery records in `./work/logs/onboarding-state/` reuse completed phases only when configuration, options, source, code, dependencies and retained artifacts match. Missing or changed artifacts rebuild the affected phase and its successors. Failed validation is retried, not counted as complete; compatible sizing and calibration do not repeat. Recycling a node can remove temporary fixtures and require rebuilding them, even though the diagnostic logs remain.

**Inspect the generated specification and summary before accepting the gates.** Use the resolved ID printed by onboarding to find `arctic_platform/correctness/specs/<config-id>.json` and `arctic_platform/correctness/specs/<config-id>.onboarding.json`. A migrated existing demo retains its `tiny-qwen3-129m-one-h200` ID through the layout manifest. The summary records each check's result and work directory, plus any combined-reference preparation error. Compact records use the mirrored [reference location](#imported-config-and-reference-locations); generated tensors never belong in Git. Importing the tiny model leaves existing 8B recipes and their records untouched. Import or preparation alone, and any validation failure, are not numerical passes.

**To onboard one check, add `--test single-step-grads` or another listed test ID.** Missing gradient-fixture prerequisites are automatic. For RL configs, the harness skips supervised-reference checks and selects compatible RL checks. Unsupported hosted-only checks are reported as inapplicable to local onboarding. List available IDs with `python -m arctic_platform.correctness list`; no configuration-specific specification or record arguments are required for default onboarding.

**Existing Cortex-training recipe JSON can be supplied without converting it to the native client schema.** Import accepts `sub_job_configs` and preserves the original bytes and engine settings, including sampling-only recipes. Existing correctness checks still require exactly one training sub-job and reject an inference-only recipe before model preparation; import acceptance does not make a recipe eligible for numerical onboarding. Training plus sampling remains a reinforcement learning (RL) configuration, not supervised training. The first-two checks still select a reduced fixture and apply their existing correctness token cap; acceptance by the loader does not establish full-model runtime support.

```bash
python -m arctic_platform.correctness onboard \
  --config ../cortex-training/model-catalog/models/qwen3.8-flash-next/long/h200-training-full.json
```

**Use the recipe path present in your Cortex-training checkout.** The model catalog replaced `recipes/sft/configs/qwen38_flash_next.json`; the example above selects the Qwen3.8-Flash-Next full-training config for eight H200 GPUs and 256K context. Deleted Cortex recipe paths have no automatic replacement. Migrated correctness config paths remain resolvable through `config-layout.json`. The imported path is `arctic_platform/correctness/configs/qwen3.8-flash-next/long/h200-training-full.json`; another model or sequence directory can contain the same filename without conflict. Different bytes at that exact relative path require a distinct variant.

## Running

**Regression names the selected tests before running them.** The selected-pair count and test IDs reflect the applicable seven-test registry selection, not a fixed first-two list. Workload mode and the reviewed specification determine applicability.

**Run all applicable checks with one config-only command in one terminal.** Omitting `--test` selects the applicable set from the seven-test registry, not only the two reduced-fixture checks. No separate gateway command is needed for a supported automatically managed local route. Each check requires its own prepared identity and target capabilities; missing prerequisites remain separate non-passes. Regression never prepares missing references, changes precision or widens reviewed gates.

**An explicit first-two-only regression can manage its local gateway automatically.** With no Cortex host or base URL configured, regression selects the reduced fixture, starts the installed DSS service on an available loopback port, and closes its owned gateway after execution or failure. No separate gateway command, database/schema exports or fixture flag are needed for that admitted route. The [startup boundary](#check-specific-prerequisites) distinguishes this selection from default-all regression and later-test routes.

```bash
python -m arctic_platform.correctness run \
  --config ./work/configs/tiny-qwen3-129m-one-h200.json \
  --out ./work/reports/all
```

**Reports and the owned gateway log remain in the requested directory.** Read `./work/reports/all/report.md`, `report.json` and `full.log`. Gateway logs are under `./work/reports/all/cases/<config-id>/<group>/gateway.log`; `full.log` records the exact gateway log path and detailed execution progress. The harness supplies nonempty local database/schema segments automatically. It uses the same allocation hostfile selection as onboarding, or generates a local hostfile from visible GPUs. For multi-node execution, supply the real same-job hostfile and working `ds_ssh`; no machines are allocated by this command. Regression consumes prepared references and never regenerates them or changes their gates.

**Explicit connection settings remain supported and are never replaced.** `--base-url` or `ARCTIC_CORTEX_BASE_URL` connects to an existing gateway, which the command leaves running; add `--local-cortex-fixture` for first-two reduced-fixture checks on that loopback service. `--host` or `ARCTIC_CORTEX_HOST` selects the configured account endpoint instead. Remove stale host, base URL or PAT settings from the environment when switching back to automatically managed local execution. To select only one check, add `--test` with the desired ID. Default-all onboarding prepares both individual records and the combined applicable reference selection; a custom multi-check subset may need [explicit preparation](#explicit-preparation) for its exact reference identity.

**Ctrl-C, SIGTERM and SIGHUP stop the automatically owned gateway during startup or execution.** The command reports shutdown, waits for its owned process, and restores the previous signal handlers; repeated stop signals do not interrupt that bounded cleanup. This applies to onboarding and regression, not an explicitly configured external service.

**An owned cleanup process stops the local gateway if onboarding or regression is killed, including with SIGKILL (`kill -9`).** The command holds a private lifetime pipe; unexpected closure triggers cleanup without waiting for another command. Cleanup verifies the launch token, process identity and ownership before stopping the gateway and its token-bound local descendants, including detached Ray agents. Complete distributed worker teardown is a separate check; failures remain in `gateway-cleanup.log` and the ownership record for inspection.

**Both commands also recover a verified orphan before starting a replacement; onboarding checks before HF calibration.** Older unrecorded gateways require an exact harness launch, an orphaned parent and a retained harness log. Active commands and explicitly configured external services are never stopped. Unproven ownership or unsuccessful bounded cleanup blocks startup rather than allowing another workload to overlap surviving resources.

**Confirm the resolved config file, model source and configured sequence length printed before model work.** The quick demo must show the tiny Qwen3 snapshot and 128 configured tokens. HF calibration uses those settings independently of the gateway; output naming an old 8B configuration is not the tiny demo.

**To onboard another supported model, supply its native job config to the same command.** In this example replace `repo_id` in `snapshot_download()` with the intended model's Hugging Face ID; keep `model_name` set to the returned snapshot path. Set GPU counts, dtype, attention, optimizer and supported LoRA options for the intended job. Check the resolved imported path and config ID; different source filenames alone do not distinguish recipes with the same derived placement. Use an explicit distinct variant for conflicting recipes. Do not copy Qwen's records or calibrated gates. Multi-node jobs need the real hostfile, `ds_ssh`, and model/code distribution within that one allocation; a path on the head node is not shared storage.

### LoRA Without Adapter Inputs

**The supplied config determines whether LoRA is used.** Native `training.peft` and `training.ds_worker_config.peft_config` settings are recognized automatically; a config without LoRA settings remains full fine-tuning. Local first-two onboarding generates one shared adapter for HF calibration and target validation, then regression reuses the prepared identity. No separate adapter-generation command is needed. This route belongs to the correctness-owned gateway, not production hooks; an external loopback gateway needs compatible correctness routes, and hosted Cortex remains unsupported for reduced-fixture LoRA testing. Select both first-two tests explicitly during regression when only that adapter route is intended; later-test support is not implied.

**Reference preparation generates one adapter using upstream PEFT's default initializer.** Onboarding and regression explicitly use the harness seed `1234`, a metadata-only base architecture and CPU adapter factors. The compact reference identity records the recipe, dependency versions, topology, tensor shapes/dtypes and hashes. Keep the reference environment unchanged when reusing its records; regeneration is not permission to replace measured references.

**Adapter files remain temporary, node-local artifacts under `/data-fast`, never Git inputs.** The correctness-owned local gateway transfers verified adapter bytes through Ray's allocation-local object store rather than opening the head node's path. Base-model weights, tokenizer files and installed code require the target runtime's model-distribution mechanism on each participating node. `/data-fast` is not shared storage.

**Optional pre-existing adapters are reference-preparation inputs only for the supported CLI backend.** Use `--prepare-references --peft-adapter /data-fast/path/to/adapter` with standard `adapter_config.json` and `adapter_model.safetensors` files; their exact contents become part of the reference identity. Automatic generation supports dense `target_modules` LoRA with upstream default initialization, not expert `target_parameters`, DoRA, rank patterns or initializers that modify base weights. Loading validates complete tensor coverage and exact dtype; unsupported runtime layouts fail explicitly.

## Select And Run

**All seven test IDs are available in this full-suite checkout, but each needs its own prepared inputs and target runtime.** P1 remains a separate first-two-only checkout; replaying shared harness changes must not expand P1's registry.

**List available configuration IDs and test IDs first.**

```bash
python -m arctic_platform.correctness list
```

**Run one configuration/test pair.** `--config` accepts an exact listed ID or an existing `.config` or `.json` file path. Existing IDs such as the one below remain valid through the layout manifest; moving the config does not rename its frozen ID:

```bash
python -m arctic_platform.correctness run \
  --config qwen3-8b-h200-train-sft-full-4gpus-64k \
  --test checkpoint-resume-loss
```

**Omit `--test` to automatically select all checks appropriate for one configuration.**

```bash
python -m arctic_platform.correctness run \
  --config qwen3-8b-h200-train-sft-full-4gpus-64k
```

**Omit `--config` to run one test across every active reviewed configuration.**

```bash
python -m arctic_platform.correctness run --test inference-checkpoint-loss
```

**Selections run sequentially in configuration-ID order.** Tests within a configuration use test-ID order; single-step tests run together to preserve their combined reference identity. Repeating a test ID does not repeat the comparison. Tests that do not apply remain visible; insufficient resources and missing prerequisites are reported as blocked, not silently omitted. An argument-free `run` is rejected. Explicit `--all-configs` selects the full reviewed catalog, and repeatable `--test` flags select a subset.

**External directories support the same selection workflow.** Discovery accepts `.config` and `.json`, excludes archive/spec directories and rejects duplicate config IDs. A test-only selection runs that test across discovered configs; repeated `--test` flags select an intentional subset. RL tests require their opt-in evidence and supported native training/sampling route.

```bash
# One test and every externally supplied config.
python -m arctic_platform.correctness run \
  --config-dir ./work/configs \
  --test single-step-optimizer \
  --out ./work/reports/all-optimizers

# Every config and all applicable registered tests.
python -m arctic_platform.correctness run \
  --config-dir ./work/configs --all-configs \
  --out ./work/reports/all
```

**Use the local full-model mode only for supported, pinned original models.** It requires an existing loopback gateway and a populated offline model cache, and supports only `e2e-sft-train-validate` and `inference-checkpoint-loss`:

```bash
export HF_HUB_OFFLINE=1
export DSS_ALLOW_REMOTE_MODELS=1
# This advanced mode requires its reviewed snapshot in a populated fast-storage cache.
export HF_HUB_CACHE=/data-fast/path/to/populated/hub

python -m arctic_platform.correctness run \
  --config qwen3-8b-h200-train-sft-full-4gpus-64k \
  --test inference-checkpoint-loss \
  --local-gateway-full-model \
  --out ./work/reports/qwen3-8b-inference
```

For reviewed reduced-fixture single-step checks with an explicitly configured loopback gateway, `--local-cortex-fixture` binds that service to matching local weights. Without an endpoint, first-two regression selects this mode and starts its owned service automatically. A published reduced fixture instead uses `--cortex-fixture <publication.json>` with an existing endpoint. Fixture identity validation does not download model files. `--gpu-pool` describes available target placement capacity and does not reserve or acquire GPUs. Do not use `--any-gpu` to bypass a hardware mismatch without separately reviewing the configuration.

### Same-Allocation Inference Export

**A local multi-node inference check can use a hash-pinned allocation descriptor.** The descriptor is a JSON object supplied by the allocation owner after admitting the current nodes. The descriptor binds checkpoint export to the nodes in that one allocation; it does not establish model-file readiness, install a runtime, start a gateway, reserve GPUs or provide keepalive protection. This advanced route is separate from the complete first-two onboarding example above.

**The descriptor must describe the actual allocation hostfile and node identities.** [The live validator](../harness/inference.py) requires these fields:

| Field | Required value |
| ----- | -------------- |
| `allocation_id` | The current allocation identifier, using letters, digits, underscores or hyphens. |
| `hostfile_content` | Exact text of the allocation's prescribed `/job/hostfile`, including its original host order. |
| `hostfile_sha256` | Full lowercase SHA256 of that hostfile's bytes. |
| `nodes` | One object per hostfile row, in the identical order. |
| Each node's `host` and `slots` | The original hostfile address and positive integer slot count, not a hostname substitution. |
| Each node's `hostname` and `boot_id` | The actual node hostname and current boot identifier. |
| Each node's `python` and `python_sha256` | Absolute interpreter path and full lowercase SHA256 of that executable. |

**Supply the descriptor path and its byte hash together, with one explicit config and only `inference-checkpoint-loss`.** The following command assumes the owner has published the admitted descriptor at `./work/inference-allocation.json`, prepared the selected GLM model and sampling runtime, and provided an unauthenticated loopback gateway on port 8003. It computes the descriptor hash directly rather than relying on an undefined shell variable:

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/glm-5.3-flash-bf16/long/h200-training-full--rl-32gpus-sp16-256k.json \
  --test inference-checkpoint-loss \
  --backend cortex-training --base-url http://127.0.0.1:8003 \
  --database dev --schema dev \
  --inference-export-allocation ./work/inference-allocation.json \
  --inference-export-allocation-sha256 "$(sha256sum ./work/inference-allocation.json | cut -d ' ' -f 1)" \
  --out ./work/reports/glm-inference
```

**The paired options also apply to final inference onboarding.** Use `onboard --test-id inference-checkpoint-loss` with the same pair and connection options; onboarding resolves a relative descriptor path before changing the final child's working directory. The child verifies the original byte hash again. These options cannot be combined with all-config selection, another test, reference preparation, sizing probes or reduced-fixture mode. The file hash is checked before JSON parsing; duplicate keys, nonfinite numbers and a non-object root are rejected. The runner checks live allocation identity before training, and checkpoint export verifies worker receipts before sampling. The descriptor does not widen the config's declared placement or replace its reviewed numerical gate.

**Use `--inference-initial-model-dir` only when initial training needs an already admitted local full-model directory.** With the descriptor pair above, this optional argument supplies the native training subjob's top-level `local_model_dir` instead of letting the training worker resolve its default model cache. The path must be an absolute, canonical, existing directory under `/data-fast`, with no symlink components. For example, after admitting `/data-fast/models/GLM-5.3-Flash-BF16` on every node in the allocation, add `--inference-initial-model-dir /data-fast/models/GLM-5.3-Flash-BF16` to that command. The same option is forwarded to the final `onboard` validation child. Relative paths are rejected.

**Directory validation does not verify model contents or their presence on other nodes.** The allocation owner must separately establish that the directory contains the unchanged configured model and is accessible at the identical path on every training node. This option does not download weights, change the canonical config/model identity or tokenizer source, alter training steps or tolerances, or provide allocation maintenance. It is restricted to the single-config, unauthenticated loopback Cortex inference route and is rejected without the pinned allocation descriptor.

**The sampler still loads the learned checkpoint, never the initial training directory.** After training finishes, the check validates and distributes the exported learned weights within the allocation, shuts down the training job, and starts sampling from the export's absolute directory. The initial-directory option is absent from that sampling subjob. It does not bypass export, tokenizer or same-allocation validation.

## Explicit Preparation

**Onboarding is a deliberate preparation operation, not a regression fallback.** Use a configuration path for onboarding. Preparing references requires the existing reviewed specification, matching assets, and enough resources; preparation does not run the target test.

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/standard/h200-training-full.json \
  --prepare-references --test-id single-step-grads
```

**Prepare gradients and optimizer together when regression selects both.** A combined workload has a different identity from independently prepared scalar checks:

```bash
python -m arctic_platform.correctness onboard \
  --config arctic_platform/correctness/configs/qwen3-8b/standard/h200-training-full.json \
  --prepare-references \
  --reference-tests single-step-grads single-step-optimizer
```

**The calibration example automatically uses up to eight GPUs on the reference node.** Eight samples need one wave with eight available GPUs, or more waves with fewer GPUs. Calibration establishes an optimizer-norm gate; it is not a target pass. For detailed reference identity, original-model admission, and calibration requirements, see [HF reference onboarding](hf-reference-onboarding.md). Changing a specification or reference to hide a mismatch is not a valid remedy.

**Source moves can invalidate older references.** The current E2E identity also binds shared training helpers in `harness/sft.py`. Historical records remain unchanged; a record without the current identity cannot be used as if it measured the new source. Resolve missing or incompatible records through explicit, reviewed preparation, never an automatic migration during regression.

## Read Results

**The terminal prints one short result row per selected pair and the report directory, then ends with total wall-clock runtime rounded upward to whole seconds in `HH:MM:SS` format.** The final runtime line follows any error guidance and includes gateway startup, execution, owned shutdown and report writing. Detailed arm and placement results are saved in `report.md` and the report's JSON file. `batch.json` records each selected pair, its reason, counts, elapsed time, and whether the batch stopped for safety. `full.log` retains detailed execution output and full Python tracebacks; individual execution groups also have reports under `cases/`.

**HF preparation also reports each worker's progress and complete subprocess logs.** Start and completion or failure messages identify the arm, run and device, with elapsed time. During onboarding, the console prints each worker's stdout, stderr and exit-code files under the attempt's `./work/logs/onboarding-*/hf-reference-*/` directory, including successful workers. These files are separate from compact reference records and tensor scratch. Output remains captured until the subprocess exits; a start message alone is not evidence of continuing computation.

**Parallel HF calibration reports completed workers rather than a sequential progress bar or ETA.** Each GAS arm uses eight nondeterministic HF runs by default. Each wave reports the completed count, unfinished run IDs and elapsed time at startup, after successful completions and every 30 seconds while waiting. Starting a worker does not advance the completed count; a worker exception is reported as a failure.

**Each onboarding phase announces its work and reports elapsed time on exit, including failure.** The first-two checks each number sizing, calibration and target validation as phases 1/3, 2/3 and 3/3. Intermediate sizing probes are steps within phase 1, not additional phase headers. Onboarding ends with total wall-clock time rounded upward to whole seconds in `HH:MM:SS` format in the terminal and `onboarding.log`.

**Onboarding also reports elapsed time per attempted test before the final total.** Each test's time includes its sizing, calibration, reference preparation, target validation and owned cleanup performed during this invocation, including failed attempts. Reused phases contribute only the time spent checking and reusing them. Config-wide `onboarding.json` records each attempted result's `elapsed_seconds`, and `report.md` shows its formatted duration; skipped tests say `not run`. Combined reference preparation is timed separately because it serves multiple tests. Shared prerequisites and report writing remain in the overall total, so test durations need not sum to that total.

| Status | Meaning |
| --- | --- |
| `PASS` | The selected numerical comparison passed. |
| `FAIL` | At least one comparison failed. Read the detailed measurements. |
| `BLOCKED` | A required specification, asset, reference, resource, or connection setting is missing or incompatible. |
| `ERROR` | Execution raised an unexpected error, or the operator interrupted the run. |
| `INAPPLICABLE` | The test does not apply to this configuration. This is not a pass. |

**Exit code zero requires a pass and no failed, blocked, or errored comparison.** An entirely inapplicable batch is nonzero because it measured nothing. Invalid selectors produce exit code `2`. Missing prerequisites do not prevent independent cases from running; an unexpected escaping error or interruption stops the remaining cases conservatively.

**Interrupt with Ctrl-C and inspect cleanup before starting another run.** The harness attempts the existing backend's owned-context cleanup, saves the interruption traceback, and blocks the remaining batch. A stopped batch is not proof that all remote resources were released. Check retained cleanup detail and use the backend's supported cleanup procedure for resources owned by that run; do not stop another user's processes.

## Library Checks And Diagnostics

**Check the harness implementation without a model or GPU test.**

```bash
python -m pip install -e '.[correctness]'
CUDA_VISIBLE_DEVICES= python -m arctic_platform.correctness selftest
```

This collects the controls in `selftest/library.py` through the single `selftest/check_suite.py` driver using installed test dependencies. Implementation passes do not establish a full-model numerical pass.

**Maintained experiments use one diagnostic command.**

```bash
python -m arctic_platform.correctness.diagnostics list
python -m arctic_platform.correctness.diagnostics run <diagnostic-id> --help
```

Diagnostics have their own resource requirements and are not substitutes for the reviewed regression tests.

**The LM-head diagnostic changes the workload and is not a canonical result.** It sets `model_options["lm_head.token_chunk_size"]` to `"disabled"` in a copied in-memory config, including an existing nested `model_options["lm_head"]["token_chunk_size"]`, without editing the source config or spec. The explicit option takes precedence over legacy settings and prevents the runner from restoring the spec's chunk size. Compatible diagnostic reference records are still required; do not relabel canonical records.

**This guide is static.** Onboarding and test execution never append results, status, receipts, or logs here. Run artifacts belong in the selected report directory.

## When A Step Fails

**Give the coding agent only the config path and the operation you intended.** If any manual step fails, ask:

```text
Use skills/correctness/SKILL.md to fix onboarding for
./work/configs/tiny-qwen3-129m-one-h200.json, then run its applicable tests.
Discover the environment, reports, logs and prepared artifacts yourself.
```

For a regression-only failure, say "run both checks" or name the desired check IDs with the config path. The skill discovers the matching command, reports, logs, retry state, model assets and references without asking you to collect them. Keep compatible prepared references and reviewed tolerances intact; do not widen a gate to make a failing comparison pass.

## Add To Repo

**For a real config addition, review and stage only explicitly selected configs and reviewed specifications after onboarding and regression succeed.** The demo does not require a commit. The harness imports the config into the hierarchy under `configs/`; `specs/` contains the reviewed gates and onboarding summary keyed by config ID. Preserve existing ID bindings in `config-layout.json` when reviewing layout changes. Configs and compact references remain local and untracked until selected for publication; only `specs/` requires `-f`. The documentation-only commands below select this demo's exact paths, not whole directories or other local recipes.

```bash
git add -- \
  arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full.json
git add -f -- \
  arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.json \
  arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.onboarding.json
git diff --cached --stat
```

**Do not stage unselected `reference_data/` records, `work/`, logs, model weights, adapter tensors, optimizer snapshots or fast-storage scratch.** Compact HF records remain local unless explicitly selected for a later commit; the [finished-work template](../../../skills/correctness/references/group-report-format.md#finished-work-selection) lists only successful selected tests' exact reference directories. Retain compatible records alongside the Hugging Face source cache and reduced fixtures for later regression; deleting the records requires explicit reference preparation before regression can run again. These examples do not authorize Git actions.

## Cleanup

**Clean up only local checkout files after both commands succeed and you have reviewed their results.** Remove the duplicate original config under `./work/configs/`, keeping the imported config, reviewed specification and local compact references for later regression. Allocation scratch is left untouched and will be removed automatically on job reset.

```bash
rm ./work/configs/tiny-qwen3-129m-one-h200.json
```

**Optionally remove these two explicit directories once you no longer need the diagnostic evidence.** `./work/logs/correctness-demo/` contains this demo's onboarding logs and retry state; `./work/reports/all/` contains its regression reports. Keep both while investigating a failure; deleting retry state may require preparation again on a later onboarding attempt. Run from the checkout root. These literal paths do not use environment variables or wildcards and do not select other runs' log directories or reusable regression inputs.

```bash
rm -r ./work/logs/correctness-demo/
rm -r ./work/reports/all/
```

**Use the retained imported config for future runs after cleanup.**

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full.json \
  --out ./work/reports/all
```
