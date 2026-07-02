# Long-Context QA — multi-hop QA with Arctic RL × SkyRL

GRPO training for **Qwen3** on the 16K-context [LoongRL-Train-Data][loongrl] multi-hop
QA corpus (HotpotQA + MuSiQue + 2WikiMultiHopQA), driven by SkyRL's GRPO trainer with
either the Arctic RL + ZoRRo backend or SkyRL's native FSDP backend. Three launchers
ship in this directory:

| Launcher | Backend | Topology | Notes |
| --- | --- | --- | --- |
| `run_qwen3_8b_loongrl_grpo_arl.sh` | Arctic RL | 1 × 8 H200 | Iteration target — Qwen3-8B, fits on a standalone host. |
| `run_qwen3_32b_loongrl_grpo_arl_4node.sh` | Arctic RL | **4 × 8 H200** | Qwen3-32B, 16K prompts, 4K responses. |
| `run_qwen3_32b_loongrl_grpo_fsdp_4node.sh` | SkyRL FSDP-native | **4 × 8 H200** | Same hyperparams as the arctic sibling — the wall-clock A/B baseline. |

Behind the [Arctic RL launch blog][blog]'s long-context QA result (avg LongBench v1 QA
accuracy 69.6% → 72.3%, biggest deltas on the hardest benchmarks — +7.5 MuSiQue,
+4.5 HotpotQA, +3.5 2WikiMQA) is the 4-node Qwen3-32B + YaRN-128K training run —
the arctic 4-node launcher above is the SkyRL twin of that recipe (verl twin lives
at [`recipes/rl/verl/long_context_qa/run_qwen3_32b_longcontext_grpo_arl.sh`](../../verl/long_context_qa/run_qwen3_32b_longcontext_grpo_arl.sh)).

## What's in this folder

| File | Role |
| --- | --- |
| `download_data.py`                  | Pulls LoongRL from HF and writes SkyRL-format parquets |
| `run_qwen3_8b_loongrl_grpo_arl.sh`  | Single-node Arctic RL launcher (8 GPU, Qwen3-8B) |
| `run_qwen3_32b_loongrl_grpo_arl_4node.sh` | 4-node Arctic RL launcher (32 GPU, Qwen3-32B) |
| `run_qwen3_32b_loongrl_grpo_fsdp_4node.sh` | 4-node **FSDP-native** launcher (baseline sibling for the A/B) |
| `fsdp_loongrl_entry.py`             | FSDP-native entrypoint that side-effect-registers `long_context_qa` on driver + Ray workers (sibling of upstream's `fsdp_bird_entry.py`) |
| `requirements.txt`, `overrides.txt` | Pinned Python deps (`uv` install) |
| `arctic_rl/`                        | Recipe-local shim: registers the `long_context_qa` env with `skyrl_gym` and re-defines the Ray entrypoint so workers pick up the registration on deserialization |
| `sitecustomize.py`                  | Registers `long_context_qa` in `ProcessPoolExecutor` spawn children (used by the reward scorer) |

Everything else — config, trainer, generator — is imported directly from
`$SKYRL_HOME/integrations/arctic_rl/`. The sibling `simple/` and `txt2sql/`
recipes reuse envs already registered upstream, so they don't need any of the
above — this shim exists purely because `long_context_qa` is a new env.

## 1. Install

The launchers call bare `python`, matching upstream's
`integrations/arctic_rl/examples/run_bird_grpo_*` on the pinned SkyRL merge —
you activate a compatible env once, then launch. Same env as the sibling
`simple/` and `txt2sql/` recipes; if you've already done either of those you
can skip this step.

```bash
# 1. Clone SkyRL at the pinned merge commit on the ``arctic-rl-public`` branch.
#    ``arctic-rl-public`` is the branch that ships the verified BIRD Arctic-RL +
#    FSDP recipes; later commits on ``main`` / ``novasky-main`` rebased in
#    unreleased upstream changes that call
#    ``nn.Module.named_non_persistent_buffers`` (a method that isn't in any
#    released PyTorch as of 2026-06) and consequently break the FSDP path.
#    ``SKYRL_HOME`` is what the launcher dispatches from — the Arctic RL × SkyRL
#    integration code lives at ``integrations/arctic_rl/``, which is NOT inside
#    the pip-installed ``skyrl`` package.
git clone https://github.com/Snowflake-AI-Research/SkyRL
cd SkyRL && git checkout 7636101a71f1849b6127ee10232fb277d2f31174 && cd ..
export SKYRL_HOME=$PWD/SkyRL

# 2. Create the env (same closure as ``simple/`` and ``txt2sql/`` — if either
#    of those already worked for you, just ``conda activate skyrl_arl`` and
#    skip this block).
conda create -y -n skyrl_arl python=3.12.13
conda activate skyrl_arl
pip install -q uv
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r requirements.txt --override overrides.txt
```

FlashAttention 3 (Hopper-only) is pulled by `arctic-inference[vllm]`. On
A100/L40S set `ATTN_IMPL=flash_attention_2` when launching (see step 4).

## 2. Data

```bash
python download_data.py --output_dir ~/data/loongrl
```

Writes per-task and merged parquets:

```
~/data/loongrl/
├── hotpotqa/{train,test}.parquet
├── musique/{train,test}.parquet
├── 2wikimqa/{train,test}.parquet
└── merged/
    ├── train.parquet   # ~14k rows (all three tasks)
    └── test.parquet    # ~750 rows
```

Each row is in SkyRL format: `data_source`, `prompt` (chat list with the LoongRL
system prompt that instructs the model to emit `<think>…</think> \boxed{…}`),
`env_class="long_context_qa"`, `reward_spec={method:"rule", ground_truth:…}`,
`extra_info`.

The launcher consumes `merged/{train,test}.parquet` by default; override
`TRAIN_PARQUET` / `VAL_PARQUET` to focus on a single task.

## 3. Reward

The recipe-local env [`arctic_rl/envs/long_context_qa.py`](arctic_rl/envs/long_context_qa.py)
extracts the model's last `\boxed{…}` answer and matches against the ground truth
with SQuAD-style normalization. Pick the scorer via `REWARD_CALC_TYPE`:

| `REWARD_CALC_TYPE`     | Scoring                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `pure_exact_match`     | (default) substring match in `\boxed{}` — 0/1              |
| `format_exact_match`   | exact match with format + answer/EOT overflow penalties    |
| `format_f1_score`      | token F1 with the same format guardrails                   |

Same scorer code as the verl long_context_qa recipe — see
[`arctic_rl/envs/long_context_qa_reward.py`](arctic_rl/envs/long_context_qa_reward.py).

## 4. Train

```bash
bash run_qwen3_8b_loongrl_grpo_arl.sh
```

Useful overrides (set as env vars or Hydra args after the script):

| Knob             | Default          | Notes |
| ---------------- | ---------------- | --- |
| `MODEL`          | `Qwen/Qwen3-8B`  | Larger models work but expect to tune `TP_SIZE` + `OFFLOAD_OPTIMIZER` |
| `PROMPT_LEN`     | `16384`          | LoongRL's native context; drop to 8192 on A100-80G |
| `RESPONSE_LEN`   | `2048`           | |
| `TRAIN_BSZ`      | `16`             | Global GRPO batch |
| `MINI_BSZ`       | `8`              | Actor mini-batch |
| `N_SAMPLES`      | `4`              | GRPO group size |
| `TP_SIZE`        | `4`              | Sampling TP — 2 engines/node at TP=4 |
| `ATTN_IMPL`      | `flash_attention_3` | Set to `flash_attention_2` on A100/L40S |
| `REWARD_CALC_TYPE` | `pure_exact_match` | See table above |

The launcher passes the rest through unchanged — pure GRPO, `use_kl_loss=false`,
log-probs recomputed via the training engine under ZoRRo (no frozen reference
model and no separate log-prob GPUs).

## 5. 4-node Qwen3-32B (blog config) + FSDP A/B

The 4-node launcher matches the SkyRL twin of the verl long-context 32B recipe:
NUM_NODES=4, GPUS_PER_NODE=8, TP=2, 16 vLLM engine replicas, ZeRO-3 + optimizer
offload, prompt 16K / response 4K, 256 prompts × 8 samples = 2048 trajectories/step,
FA2. Re-run `download_data.py` to a shared-FS `DATA_DIR` first, then:

```bash
# On the head node + each of the 3 workers (matching python + deps everywhere):
ray start --head    --port=6379 --num-gpus=8            # head
ray start --address=<head_ip>:6379 --num-gpus=8         # x3 workers
ray status  # -> "4 active node(s)" and 32 GPUs total

# On the head node only:
DATA_DIR=/shared/data/loongrl \
CKPT_DIR=/shared/checkpoints/<run> \
LOGGER=wandb \
bash run_qwen3_32b_loongrl_grpo_arl_4node.sh
```

`DATA_DIR` and `CKPT_DIR` **must be on a shared filesystem** — the head writes
the CUDA-IPC weight-sync tensor to `CKPT_DIR/_arctic_rl/`, every worker
mmap-reads it, and Ray's data-loader tasks stream parquets from `DATA_DIR` on
any node.

### FSDP-native baseline (for the wall-clock A/B)

Run the FSDP sibling with the **same** `DATA_DIR` / hyperparams / hostfile so
the only variable is the training backend:

```bash
bash run_qwen3_32b_loongrl_grpo_fsdp_4node.sh
```

The launcher shares the ARL sibling's env, launch pattern, and hyperparameters
verbatim — the only differences are the trainer flags (`trainer.strategy=fsdp2`,
no `trainer.arctic_rl.*`) and the entrypoint (`fsdp_loongrl_entry.py`, which
side-effect-imports `arctic_rl.envs` to register `long_context_qa` on the driver
and monkey-patches SkyRL's `prepare_runtime_environment` to forward the recipe
dir onto Ray workers' `runtime_env`).

### Measured wall-clock (this cluster, 32 × H200)

3-step apples-to-apples A/B on this exact recipe (Qwen3-32B, 16K prompts, 4K
responses, 256 prompts × 8 samples = 2048 trajectories/step, TP=4, ZeRO-3 with
optimizer offload for ARL / FSDP2 with `offload_after_step=true` for FSDP,
identical wandb project so the runs sit side-by-side):

| Backend | Step 1 | Step 2 | **Step 3 (steady)** | `timing/generate` (steady) | `timing/train` (steady) | `avg_final_rewards` (steady) |
| --- | --- | --- | --- | --- | --- | --- |
| Arctic RL + ZoRRo | 632.2 s | 609.7 s | **591.1 s** | 144.0 s | 320.6 s | 0.842 |
| SkyRL native FSDP | 1316.9 s | 1311.2 s | **1279.8 s** | 213.4 s | 823.0 s | 0.846 |
| **Speedup** |  |  | **2.17×** | 1.48× (rollout) | 2.57× (train) | — |

Same wall-clock story as the [Arctic RL blog][blog]'s BIRD result (2.38×) —
Arctic RL saves roughly 3× on the train phase (ZoRRo + optimizer offload +
FCA) and ~1.5× on rollout (Arctic Inference + FCA vs plain vLLM), for an
overall ~2× end-to-end. Rewards match to within 0.5% across the first 3 steps,
confirming that the speedup isn't paid for by numerical drift.

[loongrl]: https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data
[blog]: https://www.snowflake.com/en/blog/engineering/arctic-rl-open-source-backend/
