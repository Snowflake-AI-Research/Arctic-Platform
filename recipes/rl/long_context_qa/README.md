# Long-Context QA with Arctic RL

GRPO training for **Qwen3-32B** on long-context multi-hop QA, served by
[Arctic RL](../../../arctic_platform/rl/) with the [ZoRRo](../../../arctic_platform/rl/zorro_train/) trainer. KL-anchored
against a frozen reference model.

The training data is [LoongRL-Train-Data](https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data),
a 16 K-context corpus that merges three QA sources:

| Source | Subsets used |
| --- | --- |
| HotpotQA | `hotpotqa_qwen_0_2500` + `hotpotqa_distractor_2500_5000` |
| MuSiQue | `musique_qwen_0_2500` + `musique_distractor_2500_5000` |
| 2WikiMultiHopQA | `2wikipedia_qwen_0_2500` + `2wikipedia_distractor_2500_5000` |

Topology: 4 nodes × 8 H200 GPUs (32 GPUs total), `colocate=True` with **3-way
colocation** (training + sampling + ref log-prob share each GPU bundle),
Deepspeed ZeRO stage-3 with CPU optimizer offload, vLLM rollout (TP=2). With KL the
ref-log-prob engine is colocated on the same GPUs (`log_prob_gpus` = full GPU count) —
no separate ref pool or 50/50 split.

## 1. Install packages

First create a new virtual environment of your preference or use the existing one.

We are going to use `uv` for much faster installations:
```bash
pip install uv
```

Install the Arctic packages:
```bash
uv pip install arctic-platform[rl] arctic-inference[server]
```

Install Verl and its dependencies:

Please note the assumption is cuda-12.9 - if you use a different version change the `torch` and `cuda-bindings` lines to the version you need.

Also the arctic-inference patches vllm-0.18.0, therefore we explicitly install that one.

```bash
git clone https://github.com/verl-project/verl
cd verl

grep -v flash-attn requirements.txt > requirements-no-fa.txt
uv pip install -r requirements-no-fa.txt
uv pip install -e .

uv pip install -U pip wheel packaging setuptools
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129 -U
uv pip install vllm==0.18.0
uv pip install flash-attn --no-build-isolation
uv pip install numpy==1.26.4
uv pip install transformers==4.57.6
uv pip install flashinfer-python==0.5.3
uv pip install cuda-bindings==12.9.0
```


## 2. Data preparation

`download_data.py` pulls the three subset pairs from HuggingFace,
prepends a system prompt that asks the model to think inside `<think>`
tags and answer inside `\boxed{}`, drops any non-verl columns,
writes per-task and merged train/test parquets.

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform
cd recipes/rl/long_context_qa

pip install datasets

# Defaults: --output_dir /data/snowflakesql/long-context, test_ratio=0.05, seed=42
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

## 3. Ray and multi-node hostfile

when using a multi-node training environment Ray and DeepSpeed need a special file called `hostfile` (comes from MPI) that they use to find the participating nodes and the number of gpus on each node.

Most likely your CSP already provides one for you, if that's the case please note its path on the filesystem.

If you don't have one you need to create it. It looks like this:

```
10.1.1.1 slots=8
10.1.1.2 slots=8
10.1.1.3 slots=8
10.1.1.4 slots=8
```
the first column is the IPs of the participating nodes, the second column is the number of gpus on each node.

now run:
```
export JOB_HOSTFILE=/path/of/hostfile
```
(or you can change the `HOSTFILE` setting on top of both scripts)

now launch the multi-node ray launcher:
```
bash ./restart_multi_ray.sh
```

## 4. Train

Next edit the environment variables in `run_qwen3_32b_longcontext_grpo_arl_zorro_yes_kl.sh` to match your setup. In particular:
- `HF_HOME` - where you HF hub cache is (you can unset it as well)
- `VLLM_CACHE_ROOT` - some path where vllm could cache its work

```bash
bash run_qwen3_32b_longcontext_grpo_arl_zorro_yes_kl.sh \
    data.train_files=/data/snowflakesql/long-context/merged/train.parquet \
    data.val_files=/data/snowflakesql/long-context/merged/test.parquet
```

Alternatively you can edit `DATA_DIR` in the script (defaults to
`/data/snowflakesql/long-context`) and launch with no overrides:

```bash
bash run_qwen3_32b_longcontext_grpo_arl_zorro_yes_kl.sh
```

Key recipe knobs (set inside the script):

| Knob | Default | Notes |
| --- | --- | --- |
| `PROMPT_LEN` | 16384 | LoongRL is a 16 K-context dataset |
| `RESPONSE_LEN` | 4096 | |
| `ROLL_N` | 8 | GRPO group size |
| `MAX_TOKENS_PER_GPU` | 49152 | ≥ `PROMPT_LEN + ROLL_N * RESPONSE_LEN` so each GRPO group fits a ZoRRo tile |
| `BSZ` | 256 | Train batch size (data) |
| `PPO_MINI_BSZ` | 64 | Actor mini-batch |
| `LR` | 1e-6 | |
| `USE_KL_LOSS` | True | Low-variance KL vs. frozen ref, coef `0.001` |
