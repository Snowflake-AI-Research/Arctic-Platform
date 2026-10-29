# Part One: Gradient And Optimizer Correctness

## About

This doc gives a simple way to figure out how to use the correctness system via a simple demo that onboards a new config and runs the tests.

The onboarding takes much longer than running tests, since that's where calibration happens and the same worklaod is re-run multiple times to deal with non-determinism and to identify an expected variance of tensor values between different runs.

## Step 1. Setup

**This quick demo uses `axolotl-ai-co/tiny-qwen3-129m`, a 129M-parameter Qwen3 model, with 128-token sequences on one exclusively allocated H200 GPU.**

Activate a Python 3.12+ environment with working CUDA, FlashAttention-3 kernels, and any custom python packages that don't get installed via AP's requirements.

Onboarding and regression start and stop their own gateway automatically; do not start a gateway in another terminal.

The assumption is that you are already inside an Arctic-Platform clone.

```
# assuming you're already in the AP clone
git checkout -b stas/correctness-testing-p1
pip install -e '.[correctness]'
python -m arctic_platform.correctness list
```

## Step 2. Create a Job Config file

First let's create a simple config that will onboarded:

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

All subsequent commands run from the checkout root.

**The demo config uses the Cortex protocol, not the on-prem backend.** Its loopback URL on port 1 is a non-serving placeholder required by the client schema. The documented correctness commands replace that placeholder with their automatically started gateway's actual URL; do not start a server on port 1 or export the placeholder as `ARCTIC_CORTEX_BASE_URL`.

**LoRA is detected automatically for reference preparation, not admitted target execution.** Native `training.peft` or `training.ds_worker_config.peft_config` enables automatic HF adapter generation. Cortex-training target execution rejects shared adapters, including generated ones, so a LoRA config cannot complete the onboarding and regression commands below. The example config above has no LoRA settings and therefore uses full fine-tuning.

## Step 3. Config Onboard

**Give onboarding the config; it selects all applicable checks automatically.** On this P1 branch that means gradients followed by optimizer state. The harness derives the checkpoint from `model_name`, creates the fixture and specification, calibrates tolerances, publishes single-check and combined-selection references, and validates the target. You do not need to invent a specification, record location or prerequisite command. The quick demo uses a four-layer fixture and short batches, with eight nondeterministic calibration samples per gradient accumulation step (GAS) arm, GAS1 and GAS4. Eight samples is also the default for real-model onboarding. This is a workflow demo, not full-model, distributed-training or maximum-sequence-length validation.

**Calibration automatically uses the visible GPUs on one node, capped at eight concurrent HF workers.** Eight samples run in one wave of eight per GAS arm when eight or more local GPUs are visible, even if sixteen GPUs are visible. A node with fewer GPUs needs more waves; this demo runs serially if only one GPU is visible. `--reference-calibration-workers` accepts an explicit override from one to eight available local GPUs. The target's GPU count does not limit calibration concurrency.

```bash
python -m arctic_platform.correctness onboard --config ./work/configs/tiny-qwen3-129m-one-h200.json
```

**Inspect the generated specification and onboarding summary before accepting the gates.** They are `arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.json` and `arctic_platform/correctness/specs/tiny-qwen3-129m-one-h200.onboarding.json`. The summary records each check's result and diagnostic directory, plus any combined-reference preparation error. Compact measurements default to `arctic_platform/correctness/reference_data`; no model or adapter tensors belong in Git. The new config name keeps this demo separate from existing 8B configs and their reference records.

**Onboarding numbers the queued checks: `### Onboarding 1/2: Single Step Grads ###`, then `### Onboarding 2/2: Single Step Optimizer ###` for this demo.** Inner phases retain `***` markers. A single selected check uses `1/1` when its fixture already exists; prerequisite gradient onboarding is included in the total when needed.

**Gradient calibration uses eight nondeterministic HF runs per GAS arm with that arm's frozen inputs and model-specific packing.** GAS1 and GAS4 are measured separately; calibration uses the largest within-arm gradient-norm spread with the existing 4x rounding policy. After a calibration-code repair, repeat the same onboarding command: incompatible old calibration is rebuilt automatically, without deleting logs or manually changing a tolerance.

**Optimizer calibration shows a `tqdm` progress bar for each GAS arm, with elapsed time and ETA after every completed HF run.** Onboarding finishes with total wall-clock runtime rounded upward to whole seconds in `HH:MM:SS` format, retained in `onboarding.log` as well as the terminal.

**Each optimizer arm uses one deterministic baseline plus eight calibration repeats.** Each repeat starts an independent HF process, loads the fixture and saves optimizer state, so startup and capture time can dominate a tiny model. Both arms require eighteen processes in total; calibration repeats overlap according to the local worker count. Compatible completed preparation is reused after a failed validation; regression uses prepared records without recalibrating. Existing records retain their original run counts and reviewed gates.

**To onboard only one check, add `--test single-step-grads` or `--test single-step-optimizer`.** Missing fixture prerequisites are handled automatically. An RL config is reported as inapplicable for these two supervised-reference checks; onboarding does not manufacture SFT measurements for RL.

**The supplied config is copied into the checkout first, with its filename unchanged.** External `.json` and `.config` files are imported byte for byte into `arctic_platform/correctness/configs/`, then treated as local configs. Identical copies are reused; different contents under an existing local filename are rejected without overwriting it. No automatic Git commit is made.

**Existing Cortex-training recipes with `sub_job_configs` are accepted directly, alongside native client configs.** Point the same onboarding command at the recipe JSON; do not flatten or rewrite it. Import preserves the original bytes, training topology, sequence limit, optimizer, offload settings and weight format. The first-two checks still use their reduced reference fixture and existing correctness token cap, not a full-model or maximum-sequence-length run. A recipe with training and sampling sub-jobs remains classified as RL and is inapplicable to these two supervised checks.

```bash
python -m arctic_platform.correctness onboard \
  --config ../cortex-training/model-catalog/models/qwen3.8-flash-next/long/h200-training-full.json
```

**Use the recipe path present in your Cortex-training checkout.** The model catalog replaced `recipes/sft/configs/qwen38_flash_next.json`; the example above selects the Qwen3.8-Flash-Next full-training config for eight H200 GPUs and 256K context. Onboarding requires an existing file and does not resolve deleted recipe paths automatically. The imported filename is `h200-training-full.json`; use a distinct filename when importing a different recipe with that same basename.

## Step 4. Run Config Tests

**Run both checks with one command in one terminal, just like onboarding.** When no Cortex host or base URL is configured, regression selects the reduced fixture, starts the installed upstream DSS gateway on an available loopback port, supplies local database/schema segments, and closes only its owned gateway after execution or failure. No separate gateway command or fixture flag is required. Omitting `--test` selects both P1 checks. Regression never generates missing references or widens reviewed tolerances.

```bash
python -m arctic_platform.correctness run \
  --config ./work/configs/tiny-qwen3-129m-one-h200.json \
  --out ./work/reports/output1
```

Read `./work/reports/output1/report.md`, `report.json`, and `full.log`. Every run ends with its total wall-clock runtime rounded upward to whole seconds in `HH:MM:SS` format, after the report paths and any error guidance. A missing prepared reference or failed execution is not a numerical pass. The [operator guide](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/stas/correctness-testing-p1/arctic_platform/correctness/docs/running-tests.md) includes single-check and multi-config commands, plus how to adapt the example to another supported model.

**The command names the selected checks before execution.** For the unflagged demo above, it prints `Selected tests: single-step-grads, single-step-optimizer` beneath the configuration/test-pair count.

**The gateway log is retained under `./work/reports/output1/cases/<config-id>/<group>/gateway.log`; no second terminal or manual shutdown is needed.** `./work/reports/output1/full.log` records the exact gateway log path and detailed execution progress. An explicitly configured `--base-url` or `ARCTIC_CORTEX_BASE_URL` connects to an existing service and leaves it running; add `--local-cortex-fixture` for reduced-fixture checks on that loopback service. Remove stale host, base URL or PAT settings when switching back to automatic local execution. A multi-node run needs the allocation's real same-job hostfile and working `ds_ssh`. Reference preparation prints per-worker progress and complete log paths under `./work/logs`; compact reference records and large tensor scratch are separate from these durable logs.

**A gateway shutdown failure stops subsequent workloads and makes the command fail, even when the numerical comparisons passed.** Inspect `full.log` and the recorded `gateway.log` before retrying; automatic gateway shutdown is not proof that every detached Ray worker has exited.

**Ctrl-C, SIGTERM and SIGHUP stop the automatically owned gateway during startup or execution.** The command reports shutdown, waits for its owned process, and restores the previous signal handlers; repeated stop signals do not interrupt that bounded cleanup. This applies to onboarding and regression, not an explicitly configured external service.

**Both commands recover a stale harness-owned gateway before starting a replacement; onboarding also checks before HF calibration.** A local ownership record binds the gateway to its launch token, process start time and owning command. If that command was killed, startup stops the surviving gateway and waits for it to exit. Older, unrecorded gateways are recovered only when their exact launch, orphaned parent and retained onboarding or regression log identify them as harness-owned. An active command, unproven ownership or forced-shutdown failure stops the new command instead of killing an unrelated service. SIGKILL (`kill -9`), node failure and forced container termination cannot be trapped at the time they occur; detached Ray worker cleanup still requires separate verification.

**Check the resolved config file, model source and configured sequence length printed before model work.** For this demo, the source must be the tiny Qwen3 snapshot and the configured sequence length must be 128 tokens. An old gateway cannot change those HF reference settings. A log naming an old 8B configuration is not this tiny demo.

**The automatically started gateway's loopback readiness check ignores environment proxies.** No proxy changes are required for that startup check; explicitly configured service connections keep their existing behavior.

## Use the Correctness Skill if manual onboarding or testing is failing

It shouldn't be needed for model archs that have already been onboarded in the past and the corresponding tooling is in the correctness code base, but a new model architecture may easily fail to onboard since the script may not know what to do about some of its new ingredients.

**Give the coding agent only the config path and the operation you intended.** If any manual step fails, ask:

```text
Use skills/correctness/SKILL.md to fix onboarding for
./work/configs/tiny-qwen3-129m-one-h200.json, then run its applicable tests.
Discover the environment, reports, logs and prepared artifacts yourself.
```

For a regression-only failure, say "run both checks" or name the desired check IDs with the config path. The skill discovers the matching command, reports, logs, retry state, model assets and references without asking you to collect them. Keep compatible prepared references and reviewed tolerances intact; do not widen a gate to make a failing comparison pass.


## Step 5. Add Results to the Repo

You don't need to do it for the demo, but for the real config addition

**For a real config addition, review and stage the imported config and reviewed specifications after onboarding and regression succeed.** The demo does not require a commit. The harness imports the config into `configs/`; `specs/` contains the reviewed gates and onboarding summary. Those directories are ignored by default on P1, so staging them deliberately requires `-f`. Review all contents of `configs/` and `specs/` before adding those directories, especially in a checkout with other local recipes.

```bash
git add -f -- arctic_platform/correctness/configs/ arctic_platform/correctness/specs/
git diff --cached --stat
```

**Do not stage `reference_data/`, `work/`, logs, model weights, adapter tensors, optimizer snapshots or fast-storage scratch.** Compact HF records in `reference_data/` are generated local artifacts, not files to commit. Retain those records alongside the Hugging Face source cache and reduced fixtures for later regression; deleting the records requires explicit reference preparation before regression can run again.

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
  --config arctic_platform/correctness/configs/tiny-qwen3-129m-one-h200.json \
  --out ./work/reports/output1
```


## Validation And Limits

CPU selftests cover the extracted interfaces, training-only boundaries, external configurations, reference publication, optimizer decoding, shared LoRA adapter generation/loading/transport, and synthetic backend behavior. These are library tests, not new model or distributed-GPU results.

The first assembled CPU run passed 1,601 tests and failed 33. Focused reruns passed all 33 corrected cases; nine added controls also passed. This is retained coverage plus focused reruns, not a clean full-suite rerun after the final fixes. Those fixes include fresh on-premises reference publication before validation and detection of changed model bytes when filesystem timestamps do not change.

The `correctness` extra includes `pytest` and `pytest-instafail`, so the installation above also supports the selftest command:

```bash
python -m arctic_platform.correctness selftest
```

Both checks compare scalar norms, not elementwise tensor equality. Reduced reference fixtures do not establish correctness of a full model at its configured maximum sequence length. These supervised reference checks are inapplicable to reinforcement-learning configurations. A model still needs compatible runtime support, matching initial weights or adapters, and reviewed references; this branch does not promise arbitrary model support.
