# GSM8K GRPO with Arctic RL × TRL

The smallest end-to-end TRL recipe: GRPO for **Qwen3-1.7B** on **GSM8K**,
driven by TRL's `AsyncGRPOTrainer` with either the
[Arctic RL](../../../../arctic_platform/rl/) backend or native TRL
(`LocalTrainingClient` + stock `vllm serve`).

GSM8K is scored by TRL's built-in `accuracy_reward` (exact match on the
`#### <number>` final answer), so this recipe ships no custom reward.

| Knob | Arctic (`*_arl`) | Native baseline |
| --- | --- | --- |
| Model | `Qwen/Qwen3-1.7B` | same |
| Reward | TRL `accuracy_reward` | same |
| Data | HuggingFace `openai/gsm8k` at launch (no parquet step) | same |
| Layout | 8 GPU, `colocate=True` by default | 1 trainer GPU + 1 `vllm serve` GPU |
| Sequence lengths | prompt 1024, completion 256, `n=8` | same |

## 1. Install

Same env as the sibling [`txt2sql/`](../txt2sql) recipe — if you've built that,
`conda activate trl_arl` and skip to step 2.

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone -b api-training-client https://github.com/kashif/trl
cd Arctic-Platform/recipes/rl/trl/gsm8k

conda create -y -n trl_arl python=3.12
conda activate trl_arl
pip install -q uv
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 -U
uv pip install -r requirements.txt --override overrides.txt
uv pip install -e ../../../../[trl]
uv pip install -e ../../../../../trl
uv pip install flash-attn --no-build-isolation
```

See the top-level [`README`](../README.md#install) for why FA2 is required.

## 2. Data

No preprocess step. Both launchers call `datasets.load_dataset("openai/gsm8k", "main")`
and map each row to `{prompt, solution}`. Cap the train split with `NUM_PROMPTS`
(default: full 7473 for Arctic, 64 for the native baseline smoke).

## 3. Train

Arctic (colocated 8-GPU default):

```bash
bash run_qwen3_1.7b_gsm8k_grpo_arl.sh
```

Native TRL A/B baseline (trainer GPU 0, `vllm serve` on GPU 1):

```bash
bash run_qwen3_1.7b_gsm8k_grpo_baseline.sh
```

Common overrides (env vars consumed by the scripts):

```bash
MAX_STEPS=30
NUM_PROMPTS=128
NUM_GEN=8
ARCTIC_ZORRO=1                    # Arctic only; also set ARCTIC_ZORRO_LOAD_BALANCER=1
ARCTIC_COLOCATE=0 TRAINING_GPUS=4 SAMPLING_GPUS=4   # disaggregated, matches txt2sql
REPORT_TO=none                    # default wandb on the BIRD recipe; GSM8K follows each script
```

Once this works, the BIRD recipe ([txt2sql](../txt2sql)) is the same shape
with long shared-prefix SQL prompts and a SQLite exec-match reward.

## Files

| File | What it is |
| --- | --- |
| `run_qwen3_1.7b_gsm8k_grpo_arl.sh` | Arctic GRPO (optional ZoRRo via `ARCTIC_ZORRO`) |
| `run_qwen3_1.7b_gsm8k_grpo_arl.py` | Arctic trainer (owns the Arctic HTTP/Ray server) |
| `run_qwen3_1.7b_gsm8k_grpo_baseline.sh` | Native TRL A/B baseline |
| `run_qwen3_1.7b_gsm8k_grpo_baseline.py` | Native trainer + in-process `vllm serve` |
| `requirements.txt`, `overrides.txt` | Pinned Python deps |
