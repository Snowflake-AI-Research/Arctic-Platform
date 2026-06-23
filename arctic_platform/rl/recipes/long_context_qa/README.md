# Long-Context QA with Arctic RL

GRPO training for **Qwen3-32B** on long-context multi-hop QA, served by
[Arctic RL](../../) with the [Zorro](../../zorro) trainer. KL-anchored
against a frozen reference model (matches the long-context verl baseline
[uglrinrq](https://wandb.ai/snowflake/verl) topology).

The training data is [LoongRL-Train-Data](https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data),
a 16 K-context corpus that merges three QA sources:

| Source | Subsets used |
| --- | --- |
| HotpotQA | `hotpotqa_qwen_0_2500` + `hotpotqa_distractor_2500_5000` |
| MuSiQue | `musique_qwen_0_2500` + `musique_distractor_2500_5000` |
| 2WikiMultiHopQA | `2wikipedia_qwen_0_2500` + `2wikipedia_distractor_2500_5000` |

Topology: 4 nodes × 8 H200 GPUs (32 GPUs total), `colocate=True`,
ZeRO-3 with CPU optimizer offload, vLLM rollout (TP=2). With KL the
sampling and ref-log-prob pools each get half the GPUs.

## 1. Download and merge the data

`download_data.py` pulls the three subset pairs from HuggingFace,
prepends a system prompt that asks the model to think inside `<think>`
tags and answer inside `\boxed{}`, drops any non-verl columns, and
writes per-task and merged train/test parquets.

```bash
cd projects/long_context_qa

pip install datasets

# Defaults: --output_dir /data/snowflakesql/xyu/long-context, test_ratio=0.05, seed=42
python download_data.py --output_dir /data/snowflakesql/long-context
```

Output layout:

```
/data/snowflakesql/long-context/
├── hotpotqa/{train,test}.parquet
├── musique/{train,test}.parquet
├── 2wikimqa/{train,test}.parquet
└── merged/
    ├── train.parquet      # ~14k rows: all three tasks concatenated
    └── test.parquet       # ~750 rows
```

The training recipe consumes `merged/train.parquet` and
`merged/test.parquet` by default.

If you want to train on a single task, point the training command at
that task's `{train,test}.parquet` instead of `merged/`.

## 2. Train

The `_kl` recipe is standalone (no wrapper). It picks up `NNODES` from
`/data-fast/hostfile`.

```bash
bash run_qwen3_32b_longcontext_grpo_arl_zorro_yes_kl.sh \
    data.train_files=/data/snowflakesql/long-context/merged/train.parquet \
    data.val_files=/data/snowflakesql/long-context/merged/test.parquet
```

Or just edit `DATA_DIR` at the top of the script (default
`/data/snowflakesql/xyu/long-context`) and launch with no overrides.

Key recipe knobs (set inside the script):

| Knob | Default | Notes |
| --- | --- | --- |
| `PROMPT_LEN` | 16384 | LoongRL is a 16 K-context dataset |
| `RESPONSE_LEN` | 4096 | |
| `ROLL_N` | 8 | GRPO group size |
| `MAX_TOKENS_PER_GPU` | 49 152 | ≥ `PROMPT_LEN + ROLL_N * RESPONSE_LEN` so each GRPO group fits a Zorro tile |
| `BSZ` | 256 | Train batch size (data) |
| `PPO_MINI_BSZ` | 64 | Actor mini-batch |
| `LR` | 1e-6 | |
| `USE_KL_LOSS` | True | Low-variance KL vs. frozen ref, coef `0.001` |

## Files

| File | What it is |
| --- | --- |
| `run_qwen3_32b_longcontext_grpo_arl_zorro_yes_kl.sh` | KL-enabled GRPO + Arctic/Zorro recipe |
| `download_data.py` | LoongRL-Train-Data → verl parquets |
