# Seven-Test Correctness

## About

This doc gives a simple way to figure out how to use the correctness system via a simple demo that onboards a new config and runs the tests.

Onboarding takes longer than regression because calibration repeats the same workload to measure the expected variation between nondeterministic runs.

**This checkout registers seven correctness tests, including two reinforcement-learning (RL) comparisons.** `single-step-grads`, `single-step-optimizer`, `checkpoint-resume-loss`, `inference-checkpoint-loss`, `e2e-sft-train-validate`, `rl-weight-sync` and `rl-router-replay` are the registered IDs. The first five tests retain their existing workload identities and prerequisites. The two RL tests require narrowly scoped, opt-in trainer/sampler evidence: actual tensor-send and receiver-write records for weight synchronization, and received routing plus forward/backward router invocations for replay. CLI discovery and help do not import these observers; ordinary training and sampling calls leave them disabled.

## Step 1. Setup

**This quick demo uses `axolotl-ai-co/tiny-qwen3-129m`, a 129M-parameter Qwen3 model, with 128-token sequences on one exclusively allocated H200 GPU.**

Activate a Python 3.12+ environment with working CUDA, FlashAttention-3 kernels, and any custom python packages that don't get installed via AP's requirements.

Supported local onboarding and regression selections start and stop their own gateway automatically; do not start a second gateway in another terminal. Default mixed regression selections still need an admitted route, as described in [check-specific prerequisites](running-tests.md#check-specific-prerequisites).

The assumption is that you are already inside an Arctic-Platform clone.

```
# assuming you're already in the AP clone
pip install -e '.[correctness]'
python -m arctic_platform.correctness list
```

**`list` shows only onboardable test IDs, titles and criteria, not configurations or onboarding status.** It does not inspect artifacts or start Ray, PyTorch workloads or GPU jobs. Use the separate `report-runnable` command below to inspect config-by-test metadata.

**Install the sampling stack on the GPU allocation when selecting inference or RL tests.** Use `python -m pip install -e '.[correctness,rl]'` in the executing environment; the `rl` extra includes the in-tree Arctic Inference sampler and its pinned vLLM dependency. The `correctness` extra alone remains sufficient for CPU selftests and training/reference dependencies. The separately installed DSS service must use a compatible runtime on every participating node.

## Step 2. Create a Job Config file

First let's create a simple config that will onboarded:

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
    json.dumps(config, indent=2, sort_keys=True) + "\n"
)
PY
```

**Use `AP_CORRECTNESS_*` for harness-owned environment settings.** Existing `DSS_CORRECTNESS_*` settings remain accepted as compatibility aliases; the AP setting takes precedence when both are present. Actual installed DSS gateway and worker variables retain their upstream names.

All subsequent commands run from the checkout root.

**The demo config uses the Cortex protocol, not the on-prem backend.** Its loopback URL on port 1 is a non-serving placeholder required by the client schema. The documented correctness commands replace that placeholder with their automatically started gateway's actual URL; do not start a server on port 1 or export the placeholder as `ARCTIC_CORTEX_BASE_URL`.

**LoRA uses one shared adapter for reference preparation and local first-two target execution.** Native `training.peft` or `training.ds_worker_config.peft_config` enables automatic HF adapter generation. The correctness-owned local gateway installs that adapter for the two supervised-reference checks after verifying runtime capability, factor identity and parameter coverage. An external loopback gateway needs compatible correctness routes; hosted Cortex and later-test LoRA routes are not admitted by this reduced-fixture support. The example config above has no LoRA settings and therefore uses full fine-tuning. See [LoRA without adapter inputs](running-tests.md#lora-without-adapter-inputs).

## Step 3. Config Onboard

### A. Use An Agent With The Correctness Skill

**Give the agent the config path to onboard all applicable checks.** The [correctness skill](../../../skills/correctness/SKILL.md) discovers prerequisites and retained attempts, updates the progress report and preserves compatible records. Optional check IDs narrow an intentional request.

```text
Use the correctness skill to onboard
./work/configs/tiny-qwen3-129m-one-h200.json
for all applicable registered checks.
```

**Review the generated gates and per-test onboarding summary before regression.** The agent follows the same preparation and numerical acceptance rules as the manual route below. A failed test is reported separately and does not become a pass because another test succeeded.

### B. Run The Onboarding Command Manually

**The supplied config is copied byte for byte into the checkout's hierarchical config tree.** Recipes below `cortex-training/model-catalog/models/` retain their complete relative path under `arctic_platform/correctness/configs/`. Other `.json` and `.config` files use `<model-name>/<standard|long>/<gpu>-<training|inference>-<full|lora>.json` with any explicit `--<variant>` suffix; the [layout contract](hf-reference-onboarding.md#configuration-and-reference-layout) defines those components. This demo imports as `configs/tiny-qwen3-129m/standard/h200-training-full.json`, and its compact reference folders mirror that relative path without the extension. Identical copies are reused; different contents at the same destination are rejected without overwriting. No automatic Git commit is made.

**Give onboarding only the config; all applicable checks are attempted independently.** The seven-test registry is the default selection; no test selector or separate preparation command is required. The harness derives the checkpoint from `model_name`, creates specifications and compatible references, calibrates reviewed gates, publishes single-check and combined-selection records, and validates the targets. A reduced first-two fixture does not establish resume, sampling or full-SFT prerequisites. A missing later-check capability is reported as that check's non-pass, without suppressing independent selected checks or changing the config, inputs, precision or gates. Test selectors remain optional for deliberately narrower work.

**The first-two demo uses a four-layer fixture and short batches.** Calibration uses eight nondeterministic samples per gradient accumulation step (GAS) arm, GAS1 and GAS4; eight samples is also the default for real-model onboarding. This is a workflow demo, not full-model, distributed-training or maximum-sequence-length validation. The training-only config does not supply a sampling service or RL workload; see [check-specific prerequisites](running-tests.md#check-specific-prerequisites).

**Calibration automatically uses the visible GPUs on one node, capped at eight concurrent HF workers.** Eight samples run in one wave of eight per GAS arm when eight or more local GPUs are visible, even if sixteen GPUs are visible. A node with fewer GPUs needs more waves; this demo runs serially if only one GPU is visible. `--reference-calibration-workers` accepts an explicit override from one to eight available local GPUs. The target's GPU count does not limit calibration concurrency.

```bash
python -m arctic_platform.correctness onboard --config ./work/configs/tiny-qwen3-129m-one-h200.json
```

**Inspect the generated specification and onboarding summary before accepting the gates.** In this checkout, the migrated manifest retains `tiny-qwen3-129m-one-h200`, so the files are `arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.json` and `arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.onboarding.json`. Without that manifest entry, use the path-derived ID printed by onboarding. The imported config is `configs/tiny-qwen3-129m/standard/h200-training-full.json`, and compact measurements use `reference_data/tiny-qwen3-129m/standard/h200-training-full/<check-id>/...`. No model or adapter tensors belong in Git; moving placement does not remeasure historical records.

**Onboarding numbers the actual queued checks.** Default-all onboarding includes each applicable check and any missing fixture prerequisites; the denominator is not fixed at two. An optional optimizer-only selection may queue its gradient prerequisite first.

**Gradient calibration uses eight nondeterministic HF runs per GAS arm with that arm's frozen inputs and model-specific packing.** GAS1 and GAS4 are measured separately; calibration uses the largest within-arm gradient-norm spread with the existing 4x rounding policy. After a calibration-code repair, repeat the same onboarding command: incompatible old calibration is rebuilt automatically, without deleting logs or manually changing a tolerance.

**Calibration reports completed HF workers, unfinished run IDs and elapsed time for each wave.** A run counts as complete only after its worker finishes successfully. While workers are unfinished, the console prints a waiting update every 30 seconds; it does not estimate a sequential ETA for parallel work.

**Each first-two check has three phases: sizing, tolerance calibration and target validation.** Each phase announces its start and ends with elapsed time, including on failure. Onboarding reports elapsed time per attempted test and then total wall-clock time in `HH:MM:SS` format, retained in `onboarding.log` as well as the terminal. Test times include work and owned cleanup during this invocation; checking reused phases still takes time, and shared preparation is reported separately.

**Each optimizer arm uses one deterministic baseline plus eight calibration repeats.** Each repeat starts an independent HF process, loads the fixture and saves optimizer state, so startup and capture time can dominate a tiny model. Both arms require eighteen processes in total; calibration repeats overlap according to the local worker count. Compatible completed preparation is reused after a failed validation; regression uses prepared records without recalibrating. Existing records retain their original run counts and reviewed gates.

**To onboard only one check, add `--test single-step-grads` or `--test single-step-optimizer`.** Missing fixture prerequisites are handled automatically. An RL config is reported as inapplicable for these two supervised-reference checks; onboarding does not manufacture SFT measurements for RL.

**Existing Cortex-training recipes with `sub_job_configs` are accepted directly, alongside native client configs.** Point the same onboarding command at the recipe JSON; do not flatten or rewrite it. Import preserves the original bytes, training topology, sequence limit, optimizer, offload settings and weight format. The first-two checks still use their reduced reference fixture and existing correctness token cap, not a full-model or maximum-sequence-length run. A recipe with training and sampling sub-jobs remains classified as RL and is inapplicable to these two supervised checks.

```bash
python -m arctic_platform.correctness onboard \
  --config ../cortex-training/model-catalog/models/qwen3.8-flash-next/long/h200-training-full.json
```

**Use the recipe path present in your Cortex-training checkout.** The model catalog replaced `recipes/sft/configs/qwen38_flash_next.json`; the example above selects the Qwen3.8-Flash-Next full-training config for eight H200 GPUs and 256K context. Onboarding requires an existing file and does not resolve deleted Cortex recipe paths automatically. Its imported path is `configs/qwen3.8-flash-next/long/h200-training-full.json`, so other models or sequence choices can retain the same recipe basename without colliding.

## Step 4. Run Config Tests

**Optionally inspect the demo's recorded metadata before regression.** This command writes a config-by-test Markdown table to `correctness-runnable.md` in the current directory. Artifact inspection is read-only: no configs are imported, no references or gates are changed, and no Ray, PyTorch workload or GPU job starts. The report can also be generated on a CPU host.

```bash
python -m arctic_platform.correctness report-runnable \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full.json
```

**An `R` cell means reviewed metadata is available, not a numerical pass or full runtime admission.** Historical onboarding outcomes appear separately in linked details; current hardware, backend availability and complete reference/runtime identity are not verified. Optional `--out`, `--config-dir`, `--config`, repeatable `--test` and `--reference-records-dir` control the report location and selection; see [the discovery options](running-tests.md#discover-tests-and-runnable-metadata). Without a config selector, the report covers every discovered config, including configs missing reviewed specifications.

**Run all applicable checks with the config-only command below.** Omitting `--test` selects the applicable set from the seven-test registry. Each check uses its own prepared identity and execution prerequisites; missing prerequisites remain separate non-passes. Automatic gateway startup applies only to the admitted selections in the [startup boundary](running-tests.md#check-specific-prerequisites), not every default mixed selection. Regression never prepares missing references, changes precision or widens gates.

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full.json \
  --out ./work/reports/output1
```

Read `./work/reports/output1/report.md`, `report.json`, and `full.log`. Every run ends with its total wall-clock runtime rounded upward to whole seconds in `HH:MM:SS` format, after the report paths and any error guidance. A missing prepared reference or failed execution is not a numerical pass. The [operator guide](running-tests.md) includes single-check and multi-config commands, plus how to adapt the example to another supported model.

**The command names the selected checks before execution.** The config-only demo reports the applicable set from the seven-test registry, not a fixed two-check list.

**The gateway log is retained under `./work/reports/output1/cases/<config-id>/<group>/gateway.log`; no second terminal or manual shutdown is needed.** `./work/reports/output1/full.log` records the exact gateway log path and detailed execution progress. An explicitly configured `--base-url` or `ARCTIC_CORTEX_BASE_URL` connects to an existing service and leaves it running; add `--local-cortex-fixture` for reduced-fixture checks on that loopback service. Remove stale host, base URL or PAT settings when switching back to automatic local execution. A multi-node run needs the allocation's real same-job hostfile and working `ds_ssh`. Reference preparation prints per-worker progress and complete log paths under `./work/logs`; compact reference records and large tensor scratch are separate from these durable logs.

**A gateway shutdown failure stops subsequent workloads and makes the command fail, even when the numerical comparisons passed.** Inspect `full.log` and the recorded `gateway.log` before retrying; automatic gateway shutdown is not proof that every detached Ray worker has exited.

**Ctrl-C, SIGTERM and SIGHUP stop the automatically owned gateway during startup or execution.** The command reports shutdown, waits for its owned process, and restores the previous signal handlers; repeated stop signals do not interrupt that bounded cleanup. This applies to onboarding and regression, not an explicitly configured external service.

**Both commands recover a stale harness-owned gateway before starting a replacement; onboarding also checks before HF calibration.** A local ownership record binds the gateway to its launch token, process start time and owning command. If that command was killed, startup stops the surviving gateway and waits for it to exit. Older, unrecorded gateways are recovered only when their exact launch, orphaned parent and retained onboarding or regression log identify them as harness-owned. An active command, unproven ownership or forced-shutdown failure stops the new command instead of killing an unrelated service.

**An independent cleanup process watches the owning command's lifetime pipe.** If the command exits unexpectedly, including through SIGKILL (`kill -9`), the cleanup process verifies ownership before stopping the gateway and its token-bound local descendants. Node loss can also terminate the cleanup process; complete distributed worker teardown remains a separate verification, and failed cleanup blocks subsequent workloads.

**Check the resolved config file, model source and configured sequence length printed before model work.** For this demo, the source must be the tiny Qwen3 snapshot and the configured sequence length must be 128 tokens. An old gateway cannot change those HF reference settings. A log naming an old 8B configuration is not this tiny demo.

**The automatically started gateway's loopback readiness check ignores environment proxies.** No proxy changes are required for that startup check; explicitly configured service connections keep their existing behavior.

## Use the Correctness Skill if manual onboarding or testing is failing

It shouldn't be needed for model archs that have already been onboarded in the past and the corresponding tooling is in the correctness code base, but a new model architecture may easily fail to onboard since the script may not know what to do about some of its new ingredients.

**Give the coding agent only the config path and the operation you intended.** If any manual step fails, ask:

```text
Use skills/correctness/SKILL.md to fix onboarding for
arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full.json,
then run its applicable tests.
Discover the environment, reports, logs and prepared artifacts yourself.
```

For a regression-only failure, say "run both checks" or name the desired check IDs with the config path. The skill discovers the matching command, reports, logs, retry state, model assets and references without asking you to collect them. Keep compatible prepared references and reviewed tolerances intact; do not widen a gate to make a failing comparison pass.


## Step 5. Add Results to the Repo

You don't need to do it for the demo, but for the real config addition

**For a real config addition, review and stage only explicitly selected configs and reviewed specifications after onboarding and regression succeed.** The demo does not require a commit. Configs and compact reference records remain local and untracked until selected for publication; they are not Git-ignored in this checkout. Only `specs/` requires `-f`. The documentation-only commands below name this demo's imported config and manifest-frozen specification ID; substitute another recipe's resolved paths without staging whole trees or unrelated evidence.

```bash
git add -- \
  arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full.json
git add -f -- \
  arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.json \
  arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.onboarding.json
git diff --cached --stat
```

**Do not stage unselected `reference_data/` records, `work/`, logs, model weights, adapter tensors, optimizer snapshots or fast-storage scratch.** Compact HF records remain local unless explicitly selected for a later commit; the [finished-work template](../../../skills/correctness/references/group-report-format.md#finished-work-selection) lists only successful selected tests' exact reference directories. Retain compatible records alongside the Hugging Face source cache and reduced fixtures for later regression; deleting the records requires explicit reference preparation before regression can run again. These examples do not authorize Git actions.

## Step 6. Optional Cleanup

You can ignore this step and only do it if you need to free up space on the shared partition.

**Clean up only local checkout files after both commands succeed and you have reviewed their results.** Remove the duplicate original config under `./work/configs/`, keeping the imported config, reviewed specification and local compact references for later regression. Allocation scratch is left untouched and will be removed automatically on job reset.

```bash
rm ./work/configs/tiny-qwen3-129m-one-h200.json
```

**Optionally remove these two explicit directories once you no longer need the diagnostic evidence.** `./work/logs/correctness-demo/` contains this demo's onboarding logs and retry state; `./work/reports/output1/` contains its regression reports. Keep both while investigating a failure; deleting retry state may require preparation again on a later onboarding attempt. Run from the checkout root. These literal paths do not use environment variables or wildcards and do not select other runs' log directories or reusable regression inputs.

```bash
rm -r ./work/logs/correctness-demo/
rm -r ./work/reports/output1/
```

**Use the retained imported config for future runs after cleanup.**

```bash
python -m arctic_platform.correctness run \
  --config arctic_platform/correctness/configs/tiny-qwen3-129m/standard/h200-training-full.json \
  --out ./work/reports/output1
```


## Validation And Limits

CPU selftests cover the extracted interfaces, training-only boundaries, external configurations, reference publication, optimizer decoding, shared LoRA adapter generation/loading/transport, and synthetic backend behavior. These are library tests, not new model or distributed-GPU results.

The `correctness` extra includes `pytest` and `pytest-instafail`, so the installation above also supports the selftest command:

```bash
python -m arctic_platform.correctness selftest
```

**The first two tests compare scalar norms, not elementwise tensor equality.** Reduced fixtures do not validate a full model at its maximum sequence length. Checkpoint resume tests state restoration; inference compares training and sampling answer-token loss; E2E compares final training and held-out validation losses while retaining both trajectories. E2E and the first two tests require SFT configs. The RL tests compare observed weight installation and router replay on a supported native training/sampling route. Compatible target runtimes, matching initial weights or adapters and reviewed references are still required.

**Some reduced pretrained fixtures need native prediction-weight support absent from unchanged main.** For seed-0 `prime_rl` fixtures, positive `mtp_num_hidden_layers` in `qwen3_5_moe_text` or `qwen4_exp_text`, and positive `num_nextn_predict_layers` in `glm5_next_text`, require authenticated native prediction-weight loading. Report an explicit prerequisite error when that capability is unavailable. Absent or zero counts pass this admission check without changing the model configuration. RL evidence observers do not authorize loader workarounds or silently omitted prediction weights.
