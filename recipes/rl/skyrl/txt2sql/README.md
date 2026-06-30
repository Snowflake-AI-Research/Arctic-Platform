# Txt2SQL — single-node BIRD-SQL GRPO (Qwen3-8B)

Single-node 8-GPU GRPO for **Qwen3-8B** on **BIRD-SQL**, driven by SkyRL's PPO trainer
with the [Arctic RL](../../../arctic_platform/rl/) backend. Same Arctic RL stack as the
4-node Qwen3-32B run that produced the 2× speedup in the
[Arctic RL launch blog][blog] — FCA, CUDA-IPC weight sync, ZoRRo, Liger, FA3 — just scaled
down to one node so it runs on a standalone host.

| Knob              | Value |
| ---               | --- |
| Model             | `Qwen/Qwen3-8B` |
| Reward            | Vendored `arctic_rl.envs.bird:BirdEnv` (gold-SQL execution match; same reward fn used in verl PR #6) |
| Trainer           | DeepSpeed ZeRO-3, optimizer offload off |
| Sampling          | vLLM 0.18.0 (TP=2, 4 engines) |
| GPU layout        | Arctic RL colocates train + sample across the 8 GPUs |
| Sequence lengths  | prompt 8192, response 2048 (drop the long-tail BIRD DBs at 16K+) |

To run the same recipe at 32B / 4 nodes, see [`run_bird_grpo_32b_32gpu.sh`][skyrl-32b] in
SkyRL — the launcher in this directory is the single-node equivalent of that script.

## 1. Install packages

```bash
conda create -y -n skyrl_txt2sql python=3.12
conda activate skyrl_txt2sql
pip install uv
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
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

No SkyRL clone needed — SkyRL is pulled from git via `requirements.txt`, pinned at the
PR #1837 merge commit. The matching `arctic_rl/` integration code is vendored at
[`../_lib/arctic_rl/`](../_lib/arctic_rl) and added to `PYTHONPATH` by the launcher.

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

Then preprocess with `download_data.py`, a thin wrapper around the vendored
[`arctic_rl.envs.preprocess_bird`](../_lib/arctic_rl/envs/preprocess_bird.py) that
materializes per-sample SQLite paths and the `arctic_text_to_sql_r1` prompt format:

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

### Enabling Arctic speculative decoding

By default speculative decoding is **off**: the 32B-trained 3-head spec checkpoint from
the blog is tied to Qwen3-32B's hidden size and won't load on 8B. To turn it on, drop in
an 8B-trained 3-head checkpoint:

```bash
SPEC_MODEL=/path/to/qwen3-8b-bird-3head \
    bash run_qwen3_8b_bird_grpo_arl.sh
```

## How this is wired

- `trainer.override_entrypoint=arctic_rl.entrypoint` dispatches to the vendored
  entrypoint at [`../_lib/arctic_rl/entrypoint.py`](../_lib/arctic_rl/entrypoint.py).
- The launcher puts `../_lib/` on `PYTHONPATH`; the entrypoint forwards the same path to
  Ray workers' `runtime_env` so worker tasks can import `arctic_rl.*` too.
- `environment.env_class=bird` resolves to the vendored
  [`arctic_rl.envs.bird:BirdEnv`](../_lib/arctic_rl/envs/bird.py), which runs the
  gold-SQL execution reward (same reward fn from verl PR #6, vendored next to it as
  `bird_reward.py`).
- `trainer.arctic_rl.colocate=true` shares the 8 GPUs across Arctic RL's training and
  sampling jobs; `trainer.placement.colocate_all=false` keeps SkyRL from claiming a
  conflicting placement group on top.

[blog]: https://www.snowflake.com/en/blog/engineering/arctic-rl-open-source-backend/
[skyrl-32b]: https://github.com/NovaSky-AI/SkyRL/blob/main/integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu.sh
