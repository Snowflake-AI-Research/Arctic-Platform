# Txt2SQL — BIRD-SQL GRPO with Arctic RL × TRL

GRPO training for **Qwen3-1.7B** on **BIRD-SQL**, driven by TRL's
`AsyncGRPOTrainer` with either the [Arctic RL](../../../../arctic_platform/rl/)
backend or native TRL. Three launchers ship in this directory:

| Launcher | Backend | Topology | Notes |
| --- | --- | --- | --- |
| `run_qwen3_1.7b_bird_grpo_baseline.sh` | Native TRL | 4 trainer + 4 `vllm serve` (DP=4, TP=1) | C1 A/B baseline. Do not edit stock TRL `AsyncRolloutWorker`. |
| `run_qwen3_1.7b_bird_grpo_arl.sh` | Arctic, no ZoRRo | 4 train + 4 sample | C2. Same generate path as C3. |
| `run_qwen3_1.7b_bird_grpo_arl_zorro.sh` | Arctic + ZoRRo + load balancer | 4 train + 4 sample | C3. Train GAS=16. |

Pure GRPO, no frozen reference model. Held-out greedy val (`n=1`, `temperature=0`)
logs the same keys as the [verl txt2sql recipe](../../verl/txt2sql/):
`val-core/bird/reward/mean@1` and
`val-aux/bird/{execution_success,format_correct}/mean@1`.
Val is **opt-in** (`VAL_EVERY=0` by default) so speed jobs stay clean.

## What's in this folder

| File | Role |
| --- | --- |
| `run_qwen3_1.7b_bird_grpo_arl.sh` | Arctic, no ZoRRo (C2) |
| `run_qwen3_1.7b_bird_grpo_arl_zorro.sh` | Arctic + ZoRRo + LB (C3) |
| `run_qwen3_1.7b_bird_grpo_arl.py` | Shared Arctic trainer (owns the Arctic server) |
| `run_qwen3_1.7b_bird_grpo_baseline.sh` | Native TRL + external `vllm serve` |
| `run_qwen3_1.7b_bird_grpo_baseline.py` | Native trainer under `accelerate launch` |
| `bird_task.py` | Flatten verl parquet → TRL columns; thread-pooled `sql_reward` |
| `bird_val.py` | Held-out greedy val + verl metric keys |
| `wandb_logging.py` | Shared W&B run-name / project defaults |
| `test_bird_task.py`, `test_bird_val.py` | CPU tests for the scorer + val keys |
| `requirements.txt`, `overrides.txt` | Pinned Python deps |

Reward execution is the sibling recipe's
[`bird_reward.py`](../../verl/txt2sql/bird_reward.py) (SQLite exec-match).
This folder does not vendor a second copy.

## 1. Install

Same env as the sibling [`gsm8k/`](../gsm8k) recipe — if you've built that,
`conda activate trl_arl` and skip to step 2.

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone -b api-training-client https://github.com/kashif/trl
cd Arctic-Platform/recipes/rl/trl/txt2sql

conda create -y -n trl_arl python=3.12
conda activate trl_arl
pip install -q uv
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 -U
uv pip install -r requirements.txt --override overrides.txt
uv pip install -e ../../../../[trl]
uv pip install -e ../../../../../trl
uv pip install flash-attn --no-build-isolation
```

Native C1 also needs `vllm serve` extras on the 0.26 stack (`openai>=2.5`,
`kernels~=0.16`). `requirements.txt` already lists them.

See the top-level [`README`](../README.md#install) for why FA2 is required.

## 2. Data

Reuse the verl txt2sql parquets (same schema: `prompt`, `reward_model.ground_truth`,
`extra_info.db_path`). Either run
[`recipes/rl/verl/txt2sql/preprocess_bird.py`](../../verl/txt2sql/README.md#4-preprocess-to-verl-parquets)
or point at an existing pair:

```
/data/snowflakesql/txt2sql/train.parquet    # ~8.6k augmented BIRD train rows
/data/snowflakesql/txt2sql/val.parquet      # 1534 clean BIRD dev rows
```

The SQLite files referenced by `extra_info.db_path` must still exist at those
absolute paths when training runs. Override with `BIRD_TRAIN_PARQUET` /
`BIRD_VAL_PARQUET`.

## 3. Train

```bash
# Native TRL baseline (C1)
bash run_qwen3_1.7b_bird_grpo_baseline.sh

# Arctic, no ZoRRo (C2)
bash run_qwen3_1.7b_bird_grpo_arl.sh

# Arctic + ZoRRo + load balancer (C3)
bash run_qwen3_1.7b_bird_grpo_arl_zorro.sh
```

Useful overrides:

| Knob | C1 default | C2 default | C3 default | Notes |
| --- | --- | --- | --- | --- |
| `NUM_GEN` | 16 | 16 | 16 | GRPO group size |
| `PER_DEVICE_BSZ` | 2 | 256 | 256 | C1 is per trainer rank; C2/C3 are one TRL forward |
| `GRAD_ACCUM` | 32 | 32 | 16 | C2/C3: DeepSpeed engine GAS (TRL GAS stays 1) |
| `MAX_SEQ_LEN` / `MAX_MODEL_LEN` | 36864 | 36864 | 36864 | |
| `MAX_COMPLETION_LEN` | 4096 | 4096 | 4096 | |
| `MAX_TOKEN_LEN_PER_GPU` | — | 40960 | 40960 | Arctic packed prompt+responses |
| `TOKEN_BUDGET` | 0 | 0 | 0 | `0` → FixedCountBatcher |
| `VAL_EVERY` | 0 | 0 | 0 | Set `10` for greedy val; `VAL_MAX_SAMPLES=0` = full 1534 |
| `NUM_PROMPTS` | 128 | 128 | 128 | Set `0` or `8629` for the full train parquet |
| `REPORT_TO` | wandb | wandb | wandb | `none` to disable |

Quality runs typically set `VAL_EVERY=10 VAL_MAX_SAMPLES=0 NUM_PROMPTS=0`.

C1 train `reward` is not comparable to C2/C3 (HTTP rollout reorders groups).
Compare `val-core/bird/reward/mean@1`.

## 4. Tests

```bash
python -m pytest test_bird_task.py test_bird_val.py -q
```
