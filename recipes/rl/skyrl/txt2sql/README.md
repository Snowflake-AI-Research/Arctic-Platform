# Txt2SQL — BIRD-SQL GRPO with Arctic RL × SkyRL

GRPO training for **Qwen3** on **BIRD-SQL**, driven by SkyRL's GRPO trainer with
either the Arctic RL + ZoRRo backend or SkyRL's native FSDP backend. Three
launchers ship in this directory:

| Launcher | Backend | Topology | Notes |
| --- | --- | --- | --- |
| `run_qwen3_8b_bird_grpo_arl.sh` | Arctic RL | 1 × 8 H200 | Iteration target — Qwen3-8B, fits on a standalone host. |
| `run_qwen3_32b_bird_grpo_arl_4node.sh` | Arctic RL | **4 × 8 H200** | The exact run behind the ~2× speedup in the [Arctic RL launch blog][blog]. |
| `run_qwen3_32b_bird_grpo_fsdp_4node.sh` | SkyRL FSDP-native | **4 × 8 H200** | Same hyperparams as the arctic sibling — the wall-clock A/B baseline. |

The 4-node BIRD launcher is the direct SkyRL twin of the blog's flagship BIRD
run (verl twin: [`recipes/rl/verl/txt2sql/run_qwen3_32b_bird_grpo_arl.sh`](../../verl/txt2sql/run_qwen3_32b_bird_grpo_arl.sh)).

## What's in this folder

| File | Role |
| --- | --- |
| `download_data.py`                       | Preprocesses raw BIRD into SkyRL-format parquets (wrapper around upstream's `integrations.arctic_rl.envs.preprocess_bird`) |
| `run_qwen3_8b_bird_grpo_arl.sh`          | Single-node Arctic RL launcher (8 GPU, Qwen3-8B) |
| `run_qwen3_32b_bird_grpo_arl_4node.sh`   | 4-node Arctic RL launcher (32 GPU, Qwen3-32B) |
| `run_qwen3_32b_bird_grpo_fsdp_4node.sh`  | 4-node **FSDP-native** launcher (baseline sibling for the A/B) |
| `fsdp_bird_entry.py`                     | FSDP-native entrypoint that side-effect-registers `bird` / `bird_sql` on driver + Ray workers (sibling of the `long_context_qa` shim) |
| `arctic_rl/`                             | Recipe-local shim: imports upstream `integrations.arctic_rl.envs` and re-defines the Ray `skyrl_entrypoint` so workers pick up the registration on deserialization |
| `sitecustomize.py`                       | Registers `bird` / `bird_sql` in `ProcessPoolExecutor` spawn children (used by the reward scorer) |
| `requirements.txt`, `overrides.txt`      | Pinned Python deps (`uv` install) |

Config, trainer, generator, and `BirdEnv` all live upstream at
`$SKYRL_HOME/integrations/arctic_rl/`. The recipe-local `arctic_rl/` shim
imports upstream's env module for the `register()` side-effect but doesn't
vendor the env itself. Same skeleton as the sibling `long_context_qa/`
recipe.

## 1. Install

Same env as the sibling `simple/` and `long_context_qa/` recipes — if you've
built either of those, `conda activate skyrl_arl` and skip step 2.

```bash
# 1. Clone SkyRL at the pinned merge commit on the ``arctic-rl-public`` branch.
#    ``arctic-rl-public`` ships the verified BIRD Arctic-RL + FSDP recipes;
#    later commits on ``main`` / ``novasky-main`` call
#    ``nn.Module.named_non_persistent_buffers`` (not in any released PyTorch
#    as of 2026-06) and break the FSDP path. The launchers dispatch from
#    ``$SKYRL_HOME/integrations/arctic_rl/`` — this directory is not shipped
#    in the pip-installed ``skyrl`` package, so a checkout is required.
git clone https://github.com/Snowflake-AI-Research/SkyRL
cd SkyRL && git checkout 7636101a71f1849b6127ee10232fb277d2f31174 && cd ..
export SKYRL_HOME=$PWD/SkyRL

# 2. Create the env.
conda create -y -n skyrl_arl python=3.12.13
conda activate skyrl_arl
pip install -q uv
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r requirements.txt --override overrides.txt
```

FlashAttention 3 (Hopper-only) is pulled by `arctic-inference[vllm]`. On
A100/L40S set `ATTN_IMPL=flash_attention_2` when launching (see step 4).

## 2. Data

BIRD-SQL is gated behind a sign-up form, so the raw download is **not**
automated. Stage it manually first:

1. Grab the BIRD-SQL train + dev releases from the
   [BIRD-bench site](https://bird-bench.github.io/) (registration required).
2. Unpack them so the layout matches what BIRD ships:

   ```
   ~/data/bird/raw/
   ├── train/
   │   ├── train.json
   │   └── train_databases/
   │       ├── california_schools/california_schools.sqlite
   │       └── ...
   └── dev/
       ├── dev.json
       └── dev_databases/
           ├── card_games/card_games.sqlite
           └── ...
   ```

Then preprocess:

```bash
python download_data.py \
    --bird_dir ~/data/bird/raw \
    --output_dir ~/data/bird \
    --max_tokens 8192 \
    --tokenizer Qwen/Qwen3-8B
```

Writes:

```
~/data/bird/
├── train.parquet      ~9k rows after the 8K-token filter
└── val.parquet        ~1.5k rows
```

The `--max_tokens 8192` cap matches the single-node launcher's
`PROMPT_LEN=8192` and drops BIRD's long-tail outlier DBs whose schema doesn't
fit. Re-run with `--max_tokens 32768 --tokenizer Qwen/Qwen3-32B` before the
4-node 32B run so the long-context examples survive the filter.

`download_data.py` is a thin wrapper around upstream's
`integrations.arctic_rl.envs.preprocess_bird` — it needs `SKYRL_HOME` set.

## 3. Reward

Upstream `integrations.arctic_rl.envs.bird:BirdEnv` — executes the model's SQL
against the per-sample SQLite and compares result sets to the gold query. Same
reward function as the [verl BIRD recipe](../../verl/txt2sql/) (verl PR #6).

## 4. Train

```bash
bash run_qwen3_8b_bird_grpo_arl.sh
```

Useful overrides (set as env vars or Hydra args after the script):

| Knob             | Default          | Notes |
| ---------------- | ---------------- | --- |
| `MODEL`          | `Qwen/Qwen3-8B`  | Larger models work but expect to tune `TP_SIZE` + `OFFLOAD_OPTIMIZER` |
| `PROMPT_LEN`     | `8192`           | Raise to 16K/32K only after re-running `download_data.py --max_tokens` |
| `RESPONSE_LEN`   | `2048`           | |
| `TRAIN_BSZ`      | `32`             | Global GRPO batch |
| `MINI_BSZ`       | `16`             | Actor mini-batch |
| `N_SAMPLES`      | `8`              | GRPO group size |
| `TP_SIZE`        | `2`              | Sampling TP — 4 engines at TP=2 |
| `ATTN_IMPL`      | `flash_attention_3` | Set to `flash_attention_2` on A100/L40S |
| `LOGGER`         | `console`        | Set to `wandb` and export `WANDB_API_KEY` to log to wandb |

The launcher passes the rest through unchanged — pure GRPO, `use_kl_loss=false`,
log-probs recomputed via the training engine under ZoRRo (no frozen reference
model and no separate log-prob GPUs).

### Enabling Arctic speculative decoding

Off by default: the 32B-trained 3-head spec checkpoint from the blog is tied
to Qwen3-32B's hidden size and won't load on 8B. To turn it on, supply an
8B-sized 3-head checkpoint:

```bash
SPEC_MODEL=/path/to/qwen3-8b-bird-3head \
    bash run_qwen3_8b_bird_grpo_arl.sh
```

## 5. 4-node Qwen3-32B + FSDP A/B

The 4-node launcher matches the SkyRL twin of the blog's flagship BIRD run:
NUM_NODES=4, GPUS_PER_NODE=8, TP=4, 8 vLLM engines, ZeRO-3 + optimizer offload,
32K prompt / 4K response, 128 prompts × 16 samples = 2048 trajectories/step,
FA3 on Hopper. Re-run `download_data.py` with `--max_tokens 32768
--tokenizer Qwen/Qwen3-32B` to a shared-FS `DATA_DIR`, then:

```bash
# On the head node + each of the 3 workers (matching python + deps everywhere):
ray start --head    --port=6379 --num-gpus=8            # head
ray start --address=<head_ip>:6379 --num-gpus=8         # x3 workers
ray status  # -> "4 active node(s)" and 32 GPUs total

# On the head node only:
DATA_DIR=/shared/data/bird \
CKPT_DIR=/shared/checkpoints/<run> \
LOGGER=wandb \
bash run_qwen3_32b_bird_grpo_arl_4node.sh
```

`DATA_DIR` and `CKPT_DIR` **must be on a shared filesystem** — the head writes
the CUDA-IPC weight-sync tensor to `CKPT_DIR/_arctic_rl/`, every worker
mmap-reads it, and Ray's data-loader tasks stream parquets + per-sample SQLite
files from `DATA_DIR` on any node.

### FSDP-native baseline (for the wall-clock A/B)

Run the FSDP sibling with the **same** `DATA_DIR` / hyperparams / hostfile so
the only variable is the training backend:

```bash
bash run_qwen3_32b_bird_grpo_fsdp_4node.sh
```

Same env, launch pattern, and hyperparameters as the ARL sibling — only the
trainer flags (`trainer.strategy=fsdp2`, no `trainer.arctic_rl.*`) and the
entrypoint (`fsdp_bird_entry.py`, which registers `bird` / `bird_sql` on the
driver and forwards the recipe dir onto Ray workers) differ.

### Speedup

The [Arctic RL launch blog][blog] reports **~2.38×** end-to-end on this exact
4-node BIRD configuration (Arctic RL + ZoRRo vs SkyRL FSDP-native). To
reproduce the A/B locally, run both launchers back-to-back against the same
`DATA_DIR` and compare `timing/generate` + `timing/train` at steady state (step
3+). The companion `long_context_qa/` recipe reproduces the same story on the
16K-context corpus at **2.17×** end-to-end (see [`long_context_qa/README.md §5`](../long_context_qa/README.md#5-4-node-qwen3-32b--fsdp-ab)).

[blog]: https://www.snowflake.com/en/blog/engineering/arctic-rl-open-source-backend/
