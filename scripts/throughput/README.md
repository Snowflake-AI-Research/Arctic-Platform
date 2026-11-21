# H200 SFT throughput suite

Measures the training throughput of a cortex-training model catalog file on a local DSS gateway: Arctic Platform,
dss-platform and DeepSpeed from source trees, on one 8-GPU H200 node or several. The H200 stack's throughput was
checked with it ([results](#results)).

| File | What |
| --- | --- |
| `run_file.sh` | one catalog file: starts a gateway over the trees, runs the job, stops the gateway |
| `run_catalog.sh` | a plan, one line after another, each once its nodes are idle |
| `plan.py` | the plan: one line per catalog training file, or the MoE worst-case probes |
| `report.py` | the results as a Markdown table, optionally against a baseline |
| `run_sft.py` | the job driver (cortex-training client) |
| `env.sh` | the environment, sourced by the two shell scripts |
| `pod_env.example.sh` | an example of the optional pod settings file |

## Setup

| Variable | Default | What |
| --- | --- | --- |
| `AP`, `DSSP` | required | Arctic-Platform and dss-platform trees, first on `PYTHONPATH` |
| `DSP` | the venv's DeepSpeed | a DeepSpeed tree |
| `CORTEX` | required | a cortex-training checkout: its client (`src/`), `recipes/` and `model-catalog/` |
| `VENV` | the active venv | the DSS venv (torch, vLLM, Transformers, DeepSpeed) |
| `CORPUS` | required for real text | JSONL, one `{"text": ...}` object per line (below) |
| `SCRATCH` | `~/.cache/dss-throughput` | node-local caches (Triton, TileLang, FlashInfer, torch extensions) and DSS checkpoints |
| `BAKED_ROOT` | `$SCRATCH/baked` | checkpoints by Hugging Face id: `<BAKED_ROOT>/<org>/<name>/config.json` |
| `MODEL_ROOT` | `$SCRATCH/models` | downloads by name, `<MODEL_ROOT>/<name>`; a checkpoint missing in `BAKED_ROOT` is linked from here |
| `OUT_DIR` | `./throughput-out` | `results/<run>.json`, `logs/<run>.log`, `logs/<run>.gateway.log`, `summary.txt` |
| `RUN_HOSTFILE` | this node | multi-node files: one `<host> slots=8` line per node (not `HOSTFILE`, often the pod's) |
| `STEPS` | 12 | steps per run |
| `PORT` | 8003 | the gateway's port |
| `POD_ENV` | none | a file `env.sh` sources first: the pod's NCCL / EFA settings, `CUDA_HOME`, cache paths (`pod_env.example.sh`) |

- Every node of a job needs the trees, these scripts and the venv at the same paths, and its own copy of the
  checkpoint.
- A run needs its nodes to itself, since the gateway takes all 8 GPUs of every node in the job.
  - `run_file.sh` refuses a node with GPU processes or a DSS gateway or zone.
  - Afterwards it stops its gateway and kills, on every node, what that gateway started: the processes tagged with
    `DSS_THROUGHPUT_RUN`, and the rest of their sessions.

**Corpus.** Any text works if it holds steps x rows x `max_seq_len` tokens, or it is repeated.
- The numbers below used the Markdown files of a DSS monorepo checkout taken on 2026-11-06: 2,025 files, 24.8 MB,
  sha256 `e23a7a57...54ae`.
- To build one the same way:

  ```bash
  cd <monorepo> && find "$PWD" -name node_modules -prune -o -name '.*' -prune -o -name '*.md' -print | LC_ALL=C sort |
    python3 -c 'import json, sys; [print(json.dumps({"path": p, "text": open(p, errors="ignore").read()})) for p in sys.stdin.read().splitlines()]' > corpus.jsonl
  ```

- Dense throughput does not depend on the text. MoE throughput does, through the routing.

## Run one file

```bash
export AP=... DSSP=... DSP=... CORTEX=... VENV=... CORPUS=... OUT_DIR=...
CAT=$CORTEX/model-catalog/models
scripts/throughput/run_file.sh qwen3-8b-standard-full $CAT/qwen3-8b/standard/h200-training-full.json
```

- **Multi-node files** (more than 8 GPUs): start on the hostfile's first node, since the gateway runs where the script
  runs:

  ```bash
  RUN_HOSTFILE=hosts scripts/throughput/run_file.sh glm-5.3-flash-bf16-standard-full \
    $CAT/glm-5.3-flash-bf16/standard/h200-training-full.json
  ```

  The job takes the first `n_gpus / 8` lines of the hostfile.
- **Other arguments** go to `run_sft.py`, for example `--set sp_size=4` or
  `--set ds_config.zero_optimization.stage=1` to try a knob.

## Run a catalog

```bash
scripts/throughput/plan.py > plan.txt                                # one line per full training file
scripts/throughput/run_catalog.sh plan.txt                           # the one-node lines
RUN_HOSTFILE=hosts-4n scripts/throughput/run_catalog.sh plan.txt     # then, on its first node, the lines up to 4 nodes
scripts/throughput/report.py --markdown report.md
```

- `plan.py --mode lora|all` adds the LoRA files, and `--only 'qwen3*/*/full'` filters.
- `run_catalog.sh` skips lines that need more nodes than its hostfile has, and runs whose result exists.

## What is measured

- **tok/s:** tokens trained (`max_seq_len` x `train_batch_size`) per second of step wall time. A step is the client's
  `forward_backward` plus `optimizer_step`.
  - The mean is taken over steps 3..N, as total tokens over total seconds; the first steps compile kernels and grow
    the allocator.
  - A MoE file with 10 or more steps counts from step 5.
- **Rows:** real text, the corpus through the model's tokenizer cut into full-length rows, with every token supervised.
- **Config:** the file's own `training_config` (parallelism, ZeRO stage, recompute, and its optimizer: the full files'
  AdamW keeps FP32 master weights, moments and gradient accumulation). Only `step_peak_memory_log` is added.
- **Also recorded:** per step, the loss, the grad norm and the job-wide peak GPU memory, in `results/<run>.json`.

## How we compared

- **A gap is checked on the same nodes.** For any gap against an older number, the old trees and the new trees ran
  back to back on the same node or node set: the same file, the same steps, in both orders where it was close.
  - `report.py --baseline <old OUT_DIR>` sets runs of the same names side by side.
- **Noise:**
  - pods and node sets differ by 1-3% with the same trees;
  - MoE runs vary about 3% run to run.
- **Regression bar:** new / old below 0.99 for a dense file, or below 0.97 for a MoE file, in the same-node A/B.

## Worst-case probes (MoE)

`plan.py --mode all --repeat --prefix probe` and `plan.py --mode all --random --prefix probe-random` list every MoE
file (`ep_size > 1`), each with its own row kind:

- `--repeat`: rows of `max_seq_len - 1` copies of one token, as in the RL lifecycle suite's batch. Every token routes
  to the same experts, so one EP rank receives what the whole group sends.
- `--random`: uniform random tokens, as in the correctness suite's fixtures. Every token fans out to as many EP ranks
  as it can.

Run them with `STEPS=4`. A file passes when its run completes every step and writes its result (no OOM, no hang).
Its peak memory is in the result, and its tok/s is not a throughput number.

## 262,144-token MoE full files: the UCCL-EP fix

The DSS wheel's UCCL-EP (`a3d520e`) keeps its GPU-to-proxy handle table in managed memory.
- When a long MoE full fine-tune fills the GPU, the driver evicts those pages.
- The next UCCL kernel then faults (Xid 31 `FAULT_PDE`, "an illegal memory access"), and the other ranks time out.
- This hits GLM-5.3-Flash-BF16, GLM-5.3-BF16 and DeepSeek-V4.1-Flash long full.

Upstream `18b588a` (#1085) keeps the table in device memory. Build it as an overlay against the venv's torch, and put it
on `PYTHONPATH` before `run_file.sh` (`env.sh` keeps it after the trees):

```bash
git clone https://github.com/uccl-project/uccl.git && cd uccl && git checkout a3d520e && git cherry-pick 18b588a
$VENV/bin/python -m pip install --target /tmp/nb nanobind
cd ep && PYTHONPATH=/tmp/nb TORCH_CUDA_ARCH_LIST=9.0 USE_DMABUF=0 PER_EXPERT_BATCHING=0 $VENV/bin/python setup.py build
mkdir -p $OVERLAY/uccl   # OVERLAY: an absolute path
cp $(find build -name 'ep*.so') $($VENV/bin/python -c 'import importlib.util as u; print(u.find_spec("uccl").origin)') $OVERLAY/uccl/
export PYTHONPATH=$OVERLAY
```

## Results

**Trees.** The v3 stack:
- Arctic Platform `f6715a7e`, dss-platform `e13d4bd`, DeepSpeed `61c11b39`, catalog `bea6834`;
- or, for files run before the last commits, trees whose differences touch only the correctness suite's determinism
  paths.

**Rechecks on the stack's tip** (Arctic `4abc7717` or the same code, dss `6914816`, DeepSpeed `61c11b39`):
- GLM-5.3-Flash-BF16 standard full, 32 GPUs, 10 steps: the catalog file 673 tok/s/GPU, `--set sp_size=4` 628.
- Qwen3-0.6B standard full (1 GPU) and Qwen3-1.7B standard full (8 GPUs), 6 steps, run from this directory: 32,296
  and 21,046 tok/s/GPU, against the table's 32,379 and 20,868. The losses are the same.

**GLM-5.3-Flash-BF16 on 64 GPUs** (8 nodes; Arctic `31cc7bc8`, dss `6914816`, DeepSpeed `61c11b39`; 20 steps, mean of
steps 6-19). The catalog files stay on 32 GPUs, the fewest nodes they fit. These are the best layouts found at 64 GPUs,
each the catalog file with only the listed keys changed:

| File | Layout at 64 GPUs | tok/s/GPU | 32-GPU catalog file |
| --- | --- | ---: | ---: |
| standard full (65,536 tokens) | 16 rows, `sp_size` 8, `zero_optimization.overlap_comm` true, `optimizer_state_swap.fraction` 0.3 | 810 | 674 |
| long full (262,144 tokens) | 8 rows, `sp_size` 16, `overlap_comm` true, `optimizer_state_swap.fraction` 0.7 | 760 | 763 |

- Both keep each GPU's tokens per step (16,384 standard, 32,768 long). At 64 GPUs each expert's FP32 optimizer state
  is split over two expert-data-parallel ranks, so less of it has to stream through host memory each step.
- `overlap_comm` matters at 64 GPUs: each expert's gradients are now reduced over two ranks, and without overlap that
  reduce blocks the backward. The catalog body at 64 GPUs gives 616 on standard full.
- Standard full gains 20% per GPU. Long full doesn't: its larger backward already hides the optimizer stream.
- Memory limits the swap fraction. Standard full peaks at 118 GiB at 0.3; at 0.2 it fills the GPU (143 GiB) and its
  steps take 2-6x longer; with no swap it runs out of memory. Long full peaks at 116 GiB at 0.7.
- `sp_size` 4 at standard full was slower: 8% at 32 GPUs, and memory-bound at 64 GPUs.
- The 128-GPU runs of the same standard layout (`sp_size` 4, 32 rows, no swap) gave 828-848 tok/s/GPU on an older pod
  and stack.

**How to read the table.**
- The table's runs used 10-16 steps of real text, tok/s/GPU as above.
- "v2" is the same stack before its rebase onto current upstream; its bests came from another pod, which runs 1-2%
  faster with the same trees.
- "Same-node v3/v2" is the A/B on one node or node set, with the number of pairs.
- No file regressed in a same-node A/B. gemma-3-4b-it long full's three pairs spread from -4.4% to +1.7%, which is
  pod noise.

| File | GPUs | v3 tok/s/GPU | v2 best | v3 / v2 best | Same-node v3/v2 | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek-v4.1-flash-long-full | 64 | 622 [^ds] | 632 | 0.984 | 0.996 (1) | 90.9 |
| deepseek-v4.1-flash-standard-full | 64 | 726 | 681 | 1.065 | | 114.5 |
| gemma-3-4b-it-long-full | 8 | 12,122 | 12,277 | 0.987 | 0.983 (3) | 112.6 |
| gemma-3-4b-it-standard-full | 1 | 14,617 | 14,646 | 0.998 | | 108.1 |
| gemma-4-31b-it-long-full | 8 | 427 | 423 | 1.010 | | 102.6 |
| gemma-4-31b-it-standard-full | 8 | 1,045 | 1,046 | 0.999 | | 117.2 |
| glm-5.3-bf16-long-full | 64 | 282 | 270 | 1.044 | | 84.9 |
| glm-5.3-bf16-standard-full | 64 | 301 | 294 | 1.023 | | 91.3 |
| glm-5.3-flash-bf16-long-full | 32 | 763 | 669 | 1.141 | | 118.1 |
| glm-5.3-flash-bf16-standard-full | 32 | 671 | 660 | 1.016 | | 97.2 |
| nvidia-nemotron-3-nano-30b-a3b-bf16-long-full | 8 | 5,850 | 6,006 | 0.974 | | 89.1 |
| nvidia-nemotron-3-nano-30b-a3b-bf16-standard-full | 8 | 9,992 | 10,241 | 0.976 | 1.008 (1) | 103.2 |
| nvidia-nemotron-3.5-lightning-30b-a3b-bf16-long-full | 8 | 6,153 | 6,101 | 1.008 | | 102.3 |
| nvidia-nemotron-3.5-lightning-30b-a3b-bf16-standard-full | 8 | 9,915 | 9,764 | 1.015 | 0.994 (2) | 102.3 |
| qwen3-0.6b-long-full | 1 | 27,650 | 27,983 | 0.988 | 1.001 (2) | 110.3 |
| qwen3-0.6b-standard-full | 1 | 32,379 | 32,554 | 0.995 | 1.001 (2) | 90.9 |
| qwen3-1.7b-long-full | 8 | 18,825 | 19,098 | 0.986 | 0.997 (2) | 96.5 |
| qwen3-1.7b-standard-full | 8 | 20,868 | 21,205 | 0.984 | | 81.1 |
| qwen3-30b-a3b-standard-full | 8 | 5,537 | 5,609 | 0.987 | 0.989 (2) | 102.7 |
| qwen3-4b-instruct-2507-long-full | 8 | 1,960 | 1,983 | 0.988 | | 89.7 |
| qwen3-4b-instruct-2507-standard-full | 1 | 6,019 | 5,994 | 1.004 | | 102.0 |
| qwen3-8b-long-full | 4 | 6,023 | 6,121 | 0.984 | | 130.6 |
| qwen3-8b-standard-full | 4 | 6,502 | 6,479 | 1.004 | 0.993 (2) | 131.3 |
| qwen3.5-2b-long-full | 8 | 11,199 | 11,478 | 0.976 | 1.010 (2) | 103.8 |
| qwen3.5-2b-standard-full | 8 | 21,495 | 21,954 | 0.979 | 1.009 (2) | 101.5 |
| qwen3.5-4b-long-full | 8 | 4,808 | 4,860 | 0.989 | | 127.7 |
| qwen3.5-4b-standard-full | 8 | 9,961 | 10,131 | 0.983 | 1.003 (2) | 126.2 |
| qwen3.5-9b-long-full | 8 | 3,849 | 3,867 | 0.995 | | 83.3 |
| qwen3.5-9b-standard-full | 4 | 6,449 | 6,476 | 0.996 | | 100.7 |
| qwen3.6-35b-a3b-long-full | 8 | 3,245 | 3,299 | 0.984 | | 111.8 |
| qwen3.6-35b-a3b-standard-full | 8 | 7,437 | 7,021 | 1.059 | | 119.8 |
| qwen3.8-27b-long-full | 8 | 1,240 | 1,259 | 0.985 | | 104.4 |
| qwen3.8-27b-standard-full | 8 | 2,089 | 2,060 | 1.014 | | 86.6 |
| qwen3.8-flash-next-standard-full | 32 | 1,194 | 1,216 [^q38] | 0.982 | | 74.1 |
| qwen3.8-flash-next-long-full | 32 | 802 | - | - | | 80.0 |
| qwen3.8-flash-next-long-lora | 8 | 3,126 | 3,108 | 1.006 | | 118.4 |
| qwen3.8-flash-next-standard-lora | 4 | 3,003 | - | - | | 114.1 |

[^ds]: Measured on the same 8 nodes as the v2 trees' 625. The first v3 run, on other nodes, gave 599.
[^q38]: The v2 trees on the venv v3 ran with.

**Worst cases.**
- Repeat-token rows: the catalog's 24 MoE files of the time and Qwen3.8-Flash-Next's four survive. The tightest is
  DeepSeek-V4.1-Flash standard full at 124.9 GiB.
- Uniform random rows: the files with online expert balancing, the other one-node MoE files and Qwen3.8-Flash-Next's
  LoRA files survive.
