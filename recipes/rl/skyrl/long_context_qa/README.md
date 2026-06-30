# Long-Context QA — multi-hop QA with Arctic RL × SkyRL

Single-node 8-GPU GRPO training for **Qwen3-8B** on the 16K-context [LoongRL-Train-Data][loongrl]
multi-hop QA corpus (HotpotQA + MuSiQue + 2WikiMultiHopQA), driven by SkyRL's GRPO
trainer with Arctic RL + ZoRRo as the backend.

This is the single-node iteration target for the multi-node Qwen3-32B + YaRN-128K run
behind the [Arctic RL launch blog][blog]'s long-context QA result (avg LongBench v1 QA
accuracy 69.6% → 72.3%, biggest deltas on the hardest benchmarks — +7.5 MuSiQue,
+4.5 HotpotQA, +3.5 2WikiMQA). Same Arctic RL stack here — FCA + CUDA-IPC weight sync +
ZoRRo + Liger + FA3 trainer / FLASH_ATTN inference — scaled to one node so it runs on a
standalone 8x H100/H200 host.

## What's in this folder

| File | Role |
| --- | --- |
| `download_data.py`                  | Pulls LoongRL from HF and writes SkyRL-format parquets |
| `run_qwen3_8b_loongrl_grpo_arl.sh`  | Launcher — 16K prompt, 8 GPUs, TP=4, ZeRO-3 |
| `requirements.txt`, `overrides.txt` | Pinned Python deps (`uv` install) |

The Arctic RL × SkyRL glue (env class registration, ZoRRo entrypoint, generator) is
vendored at [`../_lib/arctic_rl/`](../_lib/arctic_rl) — no SkyRL checkout required.

## 1. Install

The recipe ships its own pinned dependency closure. Same environment as the other
SkyRL recipes in this folder; you only need to install once if you've already
done [`../simple/`](../simple/) or [`../txt2sql/`](../txt2sql/).

```bash
conda create -y -n skyrl_arl python=3.12.13
conda activate skyrl_arl
pip install -q uv

# Clone SkyRL at the pinned commit — required by the launcher and the recipe-side
# shim (Arctic RL × SkyRL integration code lives at integrations/arctic_rl/, NOT
# inside the pip-installed `skyrl` package).
git clone https://github.com/NovaSky-AI/SkyRL
cd SkyRL && git checkout 76f5f467c6804e8acc6273cc677098b7679b0315 && cd ..
export SKYRL_HOME=$PWD/SkyRL

uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r requirements.txt --override overrides.txt
```

FlashAttention 3 (Hopper-only) is pulled by `arctic-inference[vllm]`. On A100/L40S
override `ATTN_IMPL=flash_attention_2` when launching (see step 4).

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

The vendored env [`long_context_qa`](../_lib/arctic_rl/envs/long_context_qa.py)
extracts the model's last `\boxed{…}` answer and matches against the ground truth
with SQuAD-style normalization. Pick the scorer via `REWARD_CALC_TYPE`:

| `REWARD_CALC_TYPE`     | Scoring                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `pure_exact_match`     | (default) substring match in `\boxed{}` — 0/1              |
| `format_exact_match`   | exact match with format + answer/EOT overflow penalties    |
| `format_f1_score`      | token F1 with the same format guardrails                   |

Same scorer code as the verl long_context_qa recipe — see
[`../_lib/arctic_rl/envs/long_context_qa_reward.py`](../_lib/arctic_rl/envs/long_context_qa_reward.py).

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

## 5. Scaling up

This recipe is the single-node iteration target. To match the blog's long-context
config, train Qwen3-32B across 4 nodes (32 GPUs) and serve inference with YaRN
extended to 128K. The verl twin
([`recipes/rl/verl/long_context_qa/`](../../verl/long_context_qa/)) ships that
multi-node launcher today; a SkyRL multi-node launcher will follow the same shape
as the txt2sql 32B multi-node recipe (driven by the same vendored `arctic_rl`
package, plus a hostfile-aware `restart_multi_ray.sh`).

[loongrl]: https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data
[blog]: https://www.snowflake.com/en/blog/engineering/arctic-rl-open-source-backend/
