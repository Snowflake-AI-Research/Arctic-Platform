# GPU Job Operation

## Execution Model

**Autorunner gets a command onto one node; the correctness gateway launches the distributed target.** The agent may run on a CPU client with shared source storage. Enqueue one script on the selected allocation's head runner, such as `stas-dev-1-0` for job `stas-dev-1`. The script runs the correctness CLI once. The CLI launches the installed DSS gateway with `--backend gpu --hostfile <same-job-hostfile>`, and the gateway creates workers on the allocation. Eight visible GPUs on the head plus eight verified GPUs on its worker mean sixteen target GPUs, not eight.

**Do not launch one onboarding process per rank.** `torchrun`, `deepspeed` and all-host `ds_ssh` are not wrappers for `python -m arctic_platform.correctness onboard` or `run`. Replicating that CLI would duplicate calibration, reference publication and gateway ownership. Use `ds_ssh` for node setup and probes; use the correctness CLI for numerical tests. Standalone distributed diagnostic scripts use the separate launch patterns at the end of this guide.

**Use one allocation only.** A two-node `stas-dev-1` and a separate two-node `stas-dev-2` are independent resources, not a four-node job. Never connect them or construct a combined hostfile. If the selected placement needs more nodes than one authorized job has, ask for a sufficiently large single allocation.

## 1. Select The Runner

**Discover the live head runner from the shared autorunner directory.** In this environment the directory is `/code/users/stas/ai/autorun`. Read its `.cursor/skills/use-autorun/SKILL.md` and every companion `SKILL.md` in that skills directory before authoring a job; these define remote-runtime, non-destructive script and GPU-stack constraints. If the directory is unavailable, use the operator's supplied autorunner location or ask for that missing location. Do not assume tools from an earlier session are installed.

```bash
AUTORUN=/code/users/stas/ai/autorun
ls "$AUTORUN"/status.*.txt
cat "$AUTORUN/status.stas-dev-1-0.txt"
```

**A status file alone does not prove the runner is alive.** Read the selected file twice a few seconds apart and check that its heartbeat advances and its state is not `stopped`. Use the exact advertised host key, not just `stas-dev-1` or the CPU client's hostname. If no live runner serves the chosen head, ask the operator to start `./start` in the autorunner directory on that head; do not silently enqueue work that cannot run.

**Check existing work once before submitting.** The runner starts every queued script immediately; it is not a serial queue. Inspect current `job:` lines and their scripts/logs for conflicting GPU use or shared-file writes. Preserve unrelated jobs. Do not cancel an unknown handle or start another owner for a live identical onboarding attempt. An old session note is not a live process. Where there is no conflict, proceed without a separate coordination exercise.

## 2. Probe And Prepare Every Node

**Run the runtime probe on the GPU allocation, never on the CPU client.** Author fresh scripts with the available file-editing tool under this checkout's Git-ignored `work/jobs/` directory, using unique filenames. The examples below use the no-debug checkout; change `AP` to the actual executing checkout. Confirm the interpreter from the GPU job's setup/history; `/home/yak/miniconda3/envs/dss/bin/python` is the established location here, not a universal default.

**Save this node probe as a fresh shared file, for example `work/jobs/<attempt>-node-probe.sh`.** Replace the attempt placeholder before use. Absolute script paths avoid autorunner's default working directory, which is the parent of its own directory, not the AP checkout.

```bash
#!/usr/bin/env bash
set -euo pipefail
AP=/code/users/stas/github/sf/stas-dss-repro/arctic-platform-correctness-testing-no-debug
PYTHON=/home/yak/miniconda3/envs/dss/bin/python
cd "$AP"
export PATH="$(dirname "$PYTHON"):$PATH"
printf 'node=%s boot=' "$(hostname)"
cat /proc/sys/kernel/random/boot_id
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
df -h /data-fast /checkpoint/huggingface
"$PYTHON" - "$AP" <<'PY'
import importlib
import importlib.metadata as metadata
import json
import os
import socket
import sys
from pathlib import Path

import torch

torch.cuda.init()
sources = {}
for name in ("arctic_platform.correctness", "dss", "deepspeed"):
    module = importlib.import_module(name)
    sources[name] = str(Path(module.__file__).resolve())
assert Path(sources["arctic_platform.correctness"]).is_relative_to(Path(sys.argv[1]).resolve()), sources
print(json.dumps({
    "host": socket.gethostname(),
    "address": socket.gethostbyname(socket.gethostname()),
    "python": str(Path(sys.executable).resolve()),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch_cuda": torch.version.cuda,
    "versions": {name: metadata.version(name) for name in ("torch", "deepspeed", "transformers", "ray")},
    "sources": sources,
}, sort_keys=True), flush=True)
PY
```

**Find the hostfile on the GPU head.** Resolve `DSS_GATEWAY_HOSTFILE`, then `ARCTIC_HOSTFILE`, then `/data-fast/hostfile`, then `/job/hostfile`, matching `harness/gateway.py::_allocation_hostfile`. Check explicit variables for stale values. Validate every row against the selected job's actual node identity, unique hostname/address and visible GPU count; `slots` must not exceed that node's usable GPUs. Retain the original order and contents. Do not build a new multi-node hostfile from guessed node names.

**Fixture distribution can require the prescribed `/job/hostfile`, not any capacity-readable file.** On branches using `harness/inference.py::_validate_local_allocation`, `/job/hostfile` must be a regular file or the exact symlink to `/data-fast/hostfile`, and its first row must identify the executing head. Select `/job/hostfile` explicitly for this path and verify that it has the same bytes as the discovered allocation file. Do not rewrite, reorder or create aliases for the allocation file to bypass admission. If the allocation does not satisfy this contract, repair the specific supported harness path or ask for the missing allocation prerequisite before numerical work.

**Submit one head probe that fans out within the allocation.** Replace the probe path with the fresh file authored above. If there is no hostfile on a genuinely single-node job, run the node probe locally; the correctness gateway can generate its single-node hostfile. Missing hostfile on a multi-node job is a prerequisite to fix, not permission to fall back to one node.

```bash
#!/usr/bin/env bash
set -euo pipefail
AP=/code/users/stas/github/sf/stas-dss-repro/arctic-platform-correctness-testing-no-debug
PYTHON=/home/yak/miniconda3/envs/dss/bin/python
PROBE="$AP/work/jobs/<attempt>-node-probe.sh"
cd "$AP"
export PATH="$(dirname "$PYTHON"):$PATH"
HOSTFILE="${DSS_GATEWAY_HOSTFILE:-${ARCTIC_HOSTFILE:-}}"
if [[ -z "$HOSTFILE" ]]; then
    for candidate in /data-fast/hostfile /job/hostfile; do
        if [[ -f "$candidate" ]]; then
            HOSTFILE="$candidate"
            break
        fi
    done
fi
if [[ -n "$HOSTFILE" ]]; then
    test -r "$HOSTFILE"
    printf 'allocation_hostfile=%s\n' "$HOSTFILE"
    cat "$HOSTFILE"
    ds_ssh -f "$HOSTFILE" -- "bash $PROBE"
else
    printf 'No allocation hostfile; verify this is a single-node job.\n'
    bash "$PROBE"
fi
```

**Do not assume identical paths mean identical runtimes or model bytes.** Compare one successful probe receipt per hostfile row. Verify the expected AP checkout, installed DSS source and required dependencies on every node. Check `/data-fast` free space and the required original checkpoint/cache; `/data-fast` is node-local. A source checkout on shared `/code` does not make a worker's package install, fixture, adapter or checkpoint available. The CLI's prerequisite admission and supported fixture distribution must also succeed; do not generate independent random fixtures on different nodes or bypass hash checks.

**Use the existing setup when a newly created job lacks prerequisites.** For this installation, the maintained GPU procedure is `bash /code/users/stas/github/sf/stas-dss-repro/setup.sh` with `ARCTIC_PLATFORM_DIR` selecting the executing AP checkout and `DSS_PLATFORM_DIR` selecting its sibling `dss-platform-main`. Read that script first; do not modify it or install from a different DSS fix clone. The correctness extra supplies harness/reference dependencies, not the entire GPU service.

```bash
export ARCTIC_PLATFORM_DIR=/code/users/stas/github/sf/stas-dss-repro/arctic-platform-correctness-testing-no-debug
export DSS_PLATFORM_DIR=/code/users/stas/github/sf/stas-dss-repro/dss-platform-main
bash /code/users/stas/github/sf/stas-dss-repro/setup.sh
/home/yak/miniconda3/envs/dss/bin/python -m pip install -e "$ARCTIC_PLATFORM_DIR[correctness]"
```

**Apply setup to every node that needs it.** Put the setup block in a fresh shared node script and invoke it through same-job `ds_ssh -f "$HOSTFILE" -- "bash <absolute-node-script>"` after verifying the transport. If the nodes share the Python environment, install once and verify all nodes; do not run competing installers into that shared environment. For node-local environments, provision each node and check its result. Re-run the all-node probe after setup, before numerical work. A successful head-node install does not finish multi-node setup.

**Verify the harness's strict SSH transport separately from `ds_ssh`.** Runtime admission and fixture transfer use `ssh -F /dev/null`, ignore ambient SSH configuration and require authenticated host keys. Export `AP_CORRECTNESS_SSH_KNOWN_HOSTS` as the existing trusted, owned regular known-hosts file and `AP_CORRECTNESS_SSH_PORT` as the allocation's actual SSH port; defaults are `~/.ssh/known_hosts` and `22`. A successful `ds_ssh` command does not establish that these defaults work. Obtain missing trust or port details through the allocation's authenticated setup or the operator; never disable host-key checking or trust an unauthenticated key scan.

**Run this read-only transport probe once on the head after setup.** Keep the verified interpreter, checkout and hostfile from the preceding probe. It uses the executing harness's exact trust handling and reports each authenticated node's hostname. A failure is a setup problem to resolve before dispatch, not permission to weaken admission.

```bash
export AP_CORRECTNESS_SSH_KNOWN_HOSTS="$HOME/.ssh/known_hosts"
export AP_CORRECTNESS_SSH_PORT=22  # Replace with the verified allocation port.
"$PYTHON" - "$HOSTFILE" <<'PY'
import subprocess
import sys
import time
from pathlib import Path

from arctic_platform.correctness.harness.engine_admission import _existing_allocation_transport
from arctic_platform.correctness.harness.inference import _strict_ssh_argv
from arctic_platform.correctness.harness.storage import parse_hostfile

nodes = [{"host": entry.host} for entry in parse_hostfile(Path(sys.argv[1]))]
admission = {"ssh": _existing_allocation_transport(), "nodes": nodes}
for node in nodes:
    deadline = time.monotonic() + 30
    with _strict_ssh_argv(admission, node, "hostname", deadline) as argv:
        result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30)
    print(f"strict_ssh_host={node['host']} hostname={result.stdout.strip()}", flush=True)
PY
```

**Respect existing allocation maintenance ownership.** When authorized to hold a job, follow the installed `/code/users/stas/ai/keep-alive/SKILL.md` and applicable operator instructions for every node. Reuse a verified active owner; do not create duplicate daemons or stop unknown maintenance. Recreated jobs require fresh node/boot identities. Setup and CPU phases do not release an allocation.

## 3. Launch Correctness

**Choose target widths from verified capacity and the unchanged recipe.** Read the recipe through `harness/config.py::load_config`; its `n_gpus` is the configured target width. Select only one larger width: 4x when it fits the same allocation, otherwise 2x when it fits, otherwise configured width only. Pass that selected maximum as `--gpu-pool` to both onboarding and regression. This avoids accidentally selecting 3x on an allocation between 2x and 4x. The pool does not reserve GPUs and does not alter recipe bytes.

| Configured GPUs | Verified allocation | `--gpu-pool` | Expected target widths |
| --- | --- | --- | --- |
| 1 | One available GPU | 1 | 1 |
| 8 | One eight-GPU node | 8 | 8 |
| 8 | Two eight-GPU nodes | 16 | 8 and 16 |
| 4 | Two eight-GPU nodes | 16 | 4 and 16 |
| 16 | Two eight-GPU nodes | 16 | 16 |

**HF calibration remains local to the head.** It uses at most eight concurrent visible HF workers on that node, independently of the target width. Do not distribute the onboarding parent or report sixteen reference workers merely because the target uses sixteen GPUs. Leave `--reference-calibration-workers` unset unless intentionally choosing fewer local workers.

**Save the following as a fresh onboarding job and fill in its verified values.** This eight-H200 Qwen3 example selects eight- and sixteen-GPU target runs on one two-node allocation. For a one-node eight-GPU job set `POOL=8`; for a one-GPU recipe set its actual config and `POOL=1` when only that width is requested. `HOSTFILE` is the exact path verified by the head probe. On a hostfile-free single-node job, omit the two hostfile exports and verify there is no stale explicit setting.

```bash
#!/usr/bin/env bash
set -euo pipefail
AP=/code/users/stas/github/sf/stas-dss-repro/arctic-platform-correctness-testing-no-debug
PYTHON=/home/yak/miniconda3/envs/dss/bin/python
CONFIG=../cortex-training/model-catalog/models/qwen3-1.7b/standard/h200-training-full.json
HOSTFILE=/job/hostfile
POOL=16
cd "$AP"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONUNBUFFERED=1
export DSS_GATEWAY_HOSTFILE="$HOSTFILE"
export ARCTIC_HOSTFILE="$HOSTFILE"
# Set these to the trust file and port verified by the strict SSH probe.
export AP_CORRECTNESS_SSH_KNOWN_HOSTS="$HOME/.ssh/known_hosts"
export AP_CORRECTNESS_SSH_PORT=22
export HF_HOME=/checkpoint/huggingface
export HF_HUB_CACHE=/checkpoint/huggingface/hub
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export AP_CORRECTNESS_TMPDIR
AP_CORRECTNESS_TMPDIR="$(mktemp -d /data-fast/tmp/correctness.XXXXXXXX)"
export AP_CORRECTNESS_LOGDIR="$AP/work/logs/onboarding-$ATTEMPT"
printf 'head=%s pool=%s hostfile=%s logs=%s scratch=%s\n' \
    "$(hostname)" "$POOL" "$HOSTFILE" "$AP_CORRECTNESS_LOGDIR" "$AP_CORRECTNESS_TMPDIR"
"$PYTHON" -m arctic_platform.correctness list
"$PYTHON" -m arctic_platform.correctness onboard --config "$CONFIG" --gpu-pool "$POOL"
```

**Use one config-only onboarding command to prepare all applicable tests and their references.** No-debug and P1 register only gradients and optimizer state; the full-suite checkout also registers later checks with additional prerequisites. For an intentionally narrower request append the requested `--test` selectors. Do not add a separate reference-preparation step to ordinary onboarding. Do not export a fake gateway URL or start a manual gateway: absent an explicit external endpoint, the CLI owns startup, admission and teardown. Inspect existing endpoint settings without printing secrets; an intentional external service must not be unset or shut down.

**An exit failure is not permission to suppress the other test.** Config-wide onboarding attempts tests independently and prints its per-test summary even when one fails. Inspect that summary before choosing retries or regression selections. Preserve successful compatible work; never rewrite the recipe, widen a gate or disable an admission check to get a pass. If a CLI rejects verified multi-node capacity, source, accumulation or fixture locality, reproduce and repair the specific correctness-library limitation within the authorized scope. Do not turn available GPUs into a permanent "unsupported recipe" claim or replace execution with a new trust/launcher framework.

**Check that reused onboarding covers the requested widths.** Some branches' `_onboarding_options` omit `gpu_pool`, so an eight-GPU success can be reused when sixteen GPUs are requested. Inspect the retained validation and compatible reference sets before accepting reuse. Missing wider coverage stays pending and needs the placement-aware correctness-library repair; preserve earlier successful artifacts rather than deleting them to force another calibration. Run the missing placement and inspect its actual report before claiming sixteen-GPU coverage.

**Use the executing admission code for omitted accumulation settings.** Older linked guides may still describe admission as local-workers-only. Where `engine_admission.py` implements authenticated all-node runtime receipts, verify those receipts for the selected workers; the old wording is not a reason to exclude the worker node. Do not assume that another checkout has the same implementation.

**Run regression against the imported path with the same pool and hostfile.** In the head script's prepared environment, each following command runs once, serially. Use a fresh output path per attempt; the example assumes both tests onboarded successfully. Run an independently successful test even if the other failed; run the combined selection only once both have compatible individual and combined references.

```bash
LOCAL_CONFIG=arctic_platform/correctness/configs/qwen3-1.7b/standard/h200-training-full.json
"$PYTHON" -m arctic_platform.correctness run --config "$LOCAL_CONFIG" \
    --test single-step-grads --gpu-pool "$POOL" \
    --out "$AP/work/reports/$ATTEMPT/single-step-grads"
"$PYTHON" -m arctic_platform.correctness run --config "$LOCAL_CONFIG" \
    --test single-step-optimizer --gpu-pool "$POOL" \
    --out "$AP/work/reports/$ATTEMPT/single-step-optimizer"
"$PYTHON" -m arctic_platform.correctness run --config "$LOCAL_CONFIG" \
    --test single-step-grads --test single-step-optimizer --gpu-pool "$POOL" \
    --out "$AP/work/reports/$ATTEMPT/combined"
```

## 4. Submit, Observe And Finish

**Submit a fresh file, retain its exact handle and watch that handle's log.** After reading the autorunner skills and syntax-checking the job with `bash -n`, enqueue the probe or execution file from the CPU client. Replace the placeholders with the actual authored file and returned job ID.

```bash
AUTORUN=/code/users/stas/ai/autorun
HOST=stas-dev-1-0
bash -n work/jobs/<attempt>-onboard.sh
"$AUTORUN/enqueue.sh" "$HOST" -- "$PWD/work/jobs/<attempt>-onboard.sh"
cat "$AUTORUN/status.$HOST.txt"
tail -n 80 "$AUTORUN/logs/job-$HOST-<job-id>.log"
```

**Wait for the submitted attempt, not whichever job finished last.** Completion requires its `job:` line to disappear and its keyed log to end with `--- EXIT <rc> ---`. `last_job` and `last_log` may belong to unrelated parallel work. If a wrapper intentionally continues after failures, retain each command's exit code and return a nonzero aggregate code; a final successful `printf` must not hide failed onboarding. Do not blindly resubmit a slow or silent attempt. Inspect its current phase and process activity against that phase's deadline.

**Cancel only your known handle when cancellation is needed.** Use `"$AUTORUN/cancel.sh" "$HOST" "<job-id>"`; do not stop the runner, which leaves its in-flight jobs running. Verify the correctness gateway's owned-worker teardown on every selected host before another workload. Never use a blanket Ray shutdown or broad process kill against unrelated work.

**Inspect actual widths, every required arm and cleanup before reporting `P`.** Read the onboarding summary and target `report.json`, `report.md`, `batch.json`, `full.log` and gateway log at the command's printed paths. Confirm that eight- and sixteen-GPU outcomes actually ran for `POOL=16` on an eight-GPU recipe, rather than infer coverage from hostfile discovery or the pool argument. Check participation and token-bound remote worker cleanup, not only head-process exit. Unsafe cleanup stops further execution.

**Observe worker teardown on every host, using the recorded attempt identities.** Before execution and after each gateway phase, retain the following same-job observations from the head. Compare exact worker PIDs and start times with the phase's launch receipts and owned gateway token, rather than interpreting any unrelated Ray process as yours. While a phase is running, retain its gateway ownership record from `harness/gateway.py::_gateway_state_directory()`; normal teardown can remove that record.

```bash
ds_ssh -f "$HOSTFILE" -- "ps -eo pid,lstart,args"
ds_ssh -f "$HOSTFILE" -- "nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv"
```

**An empty GPU inventory alone does not prove cleanup.** Check the recorded owned worker identities and token-bound processes, including CPU-side Ray agents, on each selected node. If the remote launch does not preserve the token, use its recorded remote process identities; if neither can establish ownership and absence, report cleanup as unverified and resolve it before another workload. These commands observe processes only: never use broad `pkill`, stop all Ray processes or remove another owner's resources.

**Update the progress template from retained evidence immediately.** Save compact runtimes, scores, gates, norms, command exits and source/placement identities in Git-ignored work files; keep tensors and checkpoints on FAST storage. Follow `group-report-format.md` exactly, including pending entries, tallies, failure proposals and `Add finished work`. A probe, setup, CPU test or calibration is not a numerical pass.

## Standalone Distributed Diagnostics

**Use these only for a diagnostic script that implements distributed ranks.** They are not alternative correctness-onboarding commands. Probe the allocation, interpreter, transport and script on every selected node first; use a free rendezvous port and the script's supported process count. Here `PYTHON` names the verified runtime, `DIAGNOSTIC` is an existing absolute shared script path, and no other workload owns the chosen GPUs.

**One node, multiple GPUs: launch once with local rank spawning.** For a torch-distributed script:

```bash
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 "$DIAGNOSTIC"
```

**A script requiring the DeepSpeed launcher has a single-node no-SSH route.** The installed `deepspeed` executable must come from `dirname "$PYTHON"`. This avoids the launcher's default SSH-to-self behavior:

```bash
deepspeed --no_ssh --node_rank=0 --num_gpus=8 --master_port="$PORT" "$DIAGNOSTIC"
```

**Multiple nodes: the DeepSpeed head launcher fans out through the same-job hostfile.** For a DeepSpeed-compatible diagnostic, run this once on the head; `NODES` is the selected node count, `GPUS_PER_NODE` the verified equal per-node GPU count, and `HEAD_ADDRESS` a reachable address of that head inside the allocation:

```bash
deepspeed --hostfile="$HOSTFILE" --num_nodes="$NODES" --num_gpus="$GPUS_PER_NODE" \
    --master_addr="$HEAD_ADDRESS" --master_port="$PORT" "$DIAGNOSTIC"
```

**A torchrun-only multi-node diagnostic needs one launcher per selected node.** Use one autorunner head script to fan out a shared node-launch helper through allocation-local `ds_ssh`. In that helper, derive `NODE_RANK` from the node's position in the verified hostfile's original order, matching its verified hostname/address; all nodes use the identical head address, free port, node count and per-node process count. Each helper then runs:

```bash
"$PYTHON" -m torch.distributed.run --nnodes="$NODES" --nproc-per-node="$GPUS_PER_NODE" \
    --node-rank="$NODE_RANK" --master-addr="$HEAD_ADDRESS" --master-port="$PORT" "$DIAGNOSTIC"
```

**Keep the parent fan-out alive until all node launchers finish.** Do not submit two uncoordinated head jobs, use the autorunner host suffix as a rank, or launch `--standalone` independently on every node: those form separate rendezvous groups. These raw diagnostic patterns never establish a correctness result without the maintained harness's numerical comparisons and evidence.
