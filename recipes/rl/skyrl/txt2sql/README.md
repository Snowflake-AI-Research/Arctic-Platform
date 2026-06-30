# Txt2SQL — BIRD-SQL GRPO (Qwen3-8B single-node + Qwen3-32B 4-node)

GRPO training for **Qwen3** on **BIRD-SQL**, driven by SkyRL's PPO trainer with the
[Arctic RL](../../../arctic_platform/rl/) backend. Two launchers ship in this directory:

| Launcher | Topology | Notes |
| --- | --- | --- |
| `run_qwen3_8b_bird_grpo_arl.sh` | 1 node × 8 H200 | Iteration target — fits on a standalone host. |
| `run_qwen3_32b_bird_grpo_arl_4node.sh` | **4 nodes × 8 H200** | The exact run behind the **~2× speedup** vs SkyRL FSDP-native baseline reported in the [Arctic RL launch blog][blog]. |

Both launchers use the same Arctic RL stack: FCA + `fuse_allreduce_rms` workaround,
CUDA-IPC weight sync, ZoRRo, Liger, FA3 trainer / FLASH\_ATTN inference.

| Knob              | 8B single-node | 32B 4-node |
| ---               | --- | --- |
| Model             | `Qwen/Qwen3-8B` | `Qwen/Qwen3-32B` |
| Reward            | Upstream `integrations.arctic_rl.envs.bird:BirdEnv` (gold-SQL execution match; same reward fn used in verl PR #6) | same |
| Trainer           | DeepSpeed ZeRO-3, optimizer offload off | DeepSpeed ZeRO-3, optimizer offload **on** |
| Sampling          | vLLM 0.18.0 (TP=2, 4 engines)  | vLLM 0.18.0 (TP=4, 8 engines) |
| GPU layout        | Arctic RL colocates train + sample across 8 GPUs | Arctic RL colocates train + sample across 32 GPUs |
| Sequence lengths  | prompt 8192, response 2048 | prompt 32768, response 4096 |
| Global batch      | 32 prompts × 8 samples = 256 trajectories | 128 prompts × 16 samples = 2048 trajectories |

## 1. Install packages

```bash
conda create -y -n skyrl_txt2sql python=3.12
conda activate skyrl_txt2sql
pip install uv

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone https://github.com/NovaSky-AI/SkyRL
cd SkyRL && git checkout 76f5f467c6804e8acc6273cc677098b7679b0315 && cd ..
export SKYRL_HOME=$PWD/SkyRL          # required by the launcher + download_data.py

cd Arctic-Platform/recipes/rl/skyrl/txt2sql

# CUDA 12.8 — change the index URL if you're on a different CUDA version.
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r requirements.txt --override overrides.txt
uv pip install -U pip wheel packaging setuptools
# FA2 (matches torch 2.10 + cu12 + cp312 ABI)
uv pip install \
    "flash-attn@https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
# FA3 (Hopper only — H100/H200). Skip this line and set
# ATTN_IMPL=flash_attention_2 in the launcher if you're on A100/L40S.
uv pip install \
    "flash-attn-3@https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
```

The SkyRL clone gives you `integrations/arctic_rl/` (config/trainer/generator/BirdEnv/
preprocessor) — used directly via `$SKYRL_HOME`. The `requirements.txt` pull additionally
installs the `skyrl` Python *package* (Hydra entrypoint + dataset/utils) from the same
commit so both pieces stay in sync.

## 2. Data preparation

BIRD-SQL is gated behind a sign-up form, so the raw download is **not** automated.
Stage the files manually first:

1. Download the BIRD-SQL train + dev releases from the
   [BIRD-bench site](https://bird-bench.github.io/) (you'll need to register).
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

Then preprocess with `download_data.py`, a thin wrapper around upstream's
`integrations.arctic_rl.envs.preprocess_bird` (in your SkyRL clone) that materializes
per-sample SQLite paths and the `arctic_text_to_sql_r1` prompt format:

```bash
python download_data.py \
    --bird_dir ~/data/bird/raw \
    --output_dir ~/data/bird \
    --max_tokens 8192 \
    --tokenizer Qwen/Qwen3-8B
```

Result:

```
~/data/bird/
├── train.parquet      ~9k rows after the 8K-token filter
└── val.parquet        ~1.5k rows
```

The `--max_tokens 8192` cap matches the launcher's `PROMPT_LEN=8192` and drops BIRD's
long-tail outlier DBs (e.g. `works_cycles`, `movie_3`) whose schema doesn't fit. Raise it
(e.g. `--max_tokens 16384`) only if you also raise `PROMPT_LEN` in the launcher.

## 3. Train

### 3a. Single-node 8B

```bash
bash run_qwen3_8b_bird_grpo_arl.sh
```

Common overrides (env vars consumed by the script):

```bash
LOGGER=wandb                  # default: console
DATA_DIR=~/data/bird          # default
CKPT_DIR=~/checkpoints/<run>  # default: ~/checkpoints/<auto-named>
MODEL=Qwen/Qwen3-8B           # default
TRAIN_BSZ=64 MINI_BSZ=32      # scale up if you have headroom
PROMPT_LEN=16384              # raise to 16K once you re-run download_data with --max_tokens 16384
ATTN_IMPL=flash_attention_2   # for A100/L40S (default: flash_attention_3, Hopper)
OFFLOAD_OPTIMIZER=true        # if you run into OOM
```

You can also pass any SkyRL Hydra override straight through:

```bash
bash run_qwen3_8b_bird_grpo_arl.sh \
    trainer.train_batch_size=64 \
    generator.n_samples_per_prompt=16
```

### 3b. 4-node 32B (blog speedup run)

This is the launcher behind the 2× wall-clock speedup numbers in the
[Arctic RL launch blog][blog]. Re-stage the BIRD parquets with `--max_tokens 32768
--tokenizer Qwen/Qwen3-32B` first so the long-context examples survive the filter.

```bash
# On the head node + each of the 3 workers (matching python + deps everywhere):
ray start --head    --port=6379 --num-gpus=8            # head
ray start --address=<head_ip>:6379 --num-gpus=8         # x3 workers

# Sanity check:
ray status   # -> "4 active node(s)" and 32 GPUs total

# On the head node only:
DATA_DIR=/shared/data/bird \
CKPT_DIR=/shared/checkpoints/<run> \
LOGGER=wandb \
bash run_qwen3_32b_bird_grpo_arl_4node.sh
```

`DATA_DIR` and `CKPT_DIR` **must be on a shared filesystem** (NFS / Lustre / S3-FUSE)
that all 4 nodes can read — the head writes the CUDA-IPC weight-sync tensor to
`CKPT_DIR/_arctic_rl/`, and every worker mmap-reads it for weight refresh.

The companion **FSDP-native baseline** (same recipe, no Arctic RL) lives upstream at
[`run_bird_grpo_32b_32gpu_fsdp.sh`][skyrl-32b-fsdp] — use it to reproduce the speedup
A/B exactly. Both runs share train batch / sequence lengths / optimizer state so the
wall-clock difference is the Arctic stack's contribution.

### Enabling Arctic speculative decoding

By default speculative decoding is **off**: the 32B-trained 3-head spec checkpoint from
the blog is tied to Qwen3-32B's hidden size and won't load on 8B. To turn it on, drop in
an 8B-trained 3-head checkpoint:

```bash
SPEC_MODEL=/path/to/qwen3-8b-bird-3head \
    bash run_qwen3_8b_bird_grpo_arl.sh
```

## How this is wired

- `trainer.override_entrypoint=arctic_rl.entrypoint` dispatches to the recipe-side shim
  at [`../_lib/arctic_rl/entrypoint.py`](../_lib/arctic_rl/entrypoint.py), which reuses
  upstream's `ArcticRLExp` + `build_rl_config` (imported from
  `$SKYRL_HOME/integrations/arctic_rl/`) and re-defines the `@ray.remote skyrl_entrypoint`
  so Ray workers re-import the shim and re-register the recipe's env classes.
- The launcher composes `PYTHONPATH = $SKYRL_HOME : ../_lib/ : $PYTHONPATH`; the shim
  forwards both directories to Ray workers' `runtime_env`.
- `environment.env_class=bird` resolves to upstream's
  `integrations.arctic_rl.envs.bird:BirdEnv` — the recipe-side `envs/__init__.py` just
  re-binds the `bird` / `bird_sql` registration ids to it so the same launcher Hydra
  knobs work in either checkout.
- `trainer.arctic_rl.colocate=true` shares the 8 GPUs across Arctic RL's training and
  sampling jobs; `trainer.placement.colocate_all=false` keeps SkyRL from claiming a
  conflicting placement group on top.

[blog]: https://www.snowflake.com/en/blog/engineering/arctic-rl-open-source-backend/
[skyrl-32b]: https://github.com/NovaSky-AI/SkyRL/blob/main/integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu.sh
[skyrl-32b-fsdp]: https://github.com/NovaSky-AI/SkyRL/blob/main/integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu_fsdp.sh
