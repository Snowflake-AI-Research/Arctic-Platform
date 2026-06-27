# Long-Context QA with Arctic RL

GRPO training for **Qwen3-32B** on long-context multi-hop QA, served by [Arctic RL](../../../../arctic_platform/rl/)
with the [ZoRRo](../../../../arctic_platform/rl/zorro_train/) trainer. Pure GRPO, without a frozen reference model (no
KL anchoring).

The training data is [LoongRL-Train-Data](https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data),
a 16 K-context corpus that merges three QA sources:

| Source | Subsets used |
| --- | --- |
| HotpotQA | `hotpotqa_qwen_0_2500` + `hotpotqa_distractor_2500_5000` |
| MuSiQue | `musique_qwen_0_2500` + `musique_distractor_2500_5000` |
| 2WikiMultiHopQA | `2wikipedia_qwen_0_2500` + `2wikipedia_distractor_2500_5000` |

Topology: 4 nodes × 8 H200 GPUs (32 GPUs total), `colocate=True` (training + sampling share each GPU bundle),
Deepspeed ZeRO stage-3 with CPU optimizer offload, vLLM rollout (TP=2). Without KL there is no frozen reference model,
so the ref log-prob pool is disabled (`log_prob_gpus=0`); under ZoRRo log-probs are recomputed through the training
engine itself.

## 1. Ray and multi-node hostfile

When using a multi-node training environment Ray and DeepSpeed need a special file called `hostfile` (comes from MPI)
that they use to find the participating nodes and the number of gpus on each node.

Most likely your CSP already provides one for you, if that's the case please note its path on the filesystem.

If you don't have one you need to create it. It looks like this:

```
10.1.1.1 slots=8
10.1.1.2 slots=8
10.1.1.3 slots=8
10.1.1.4 slots=8
```
the first column is the IPs of the participating nodes, the second column is the number of gpus on each node.

Export its path - the install step (and the launcher) use it to fan out across all nodes:
```
export JOB_HOSTFILE=/path/of/hostfile
```
(or you can change the `HOSTFILE` setting on top of both scripts)

The Ray cluster itself is started later, in the Train step, once the environment is installed on every node.

## 2. Install packages

The steps below install the environment on **every** node via `ds_ssh` (the DeepSpeed multi-node helper, which reads
`$JOB_HOSTFILE` from [step (1)](#1-ray-and-multi-node-hostfile)). They assume the environment lives on a shared
filesystem, or that you otherwise make it available on each node.

On the launching node, bootstrap `uv` (a much faster installer) and `ds_ssh`:
```bash
pip install uv             # bootstrap uv on the launching node
uv pip install deepspeed   # provides ds_ssh
ds_ssh -f $JOB_HOSTFILE pip install uv   # bootstrap uv on every other node
```

Clone this repo (it carries `requirements.txt` and the launcher scripts) and the verl fork:
```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone -b arctic_rl_share_v0.7.1 --single-branch https://github.com/Snowflake-AI-Research/verl
cd Arctic-Platform/recipes/rl/verl/long_context_qa
```

Install the pinned dependencies on all nodes. The assumption is cuda-12.9 - if you use a different version change the
`torch` index URL below and the `cuda-bindings` pin in `requirements.txt`. `arctic-inference` patches vllm-0.18.0, so
that exact version is pinned in `requirements.txt`.
```bash
# torch (CUDA 12.9) first, then the rest of the pinned packages.
# overrides.txt forces the few transitive deps (flashinfer/numpy/transformers)
# this recipe is validated against, which vLLM 0.18.0's metadata otherwise pins
# higher (without it the single resolve is unsatisfiable).
ds_ssh -f $JOB_HOSTFILE uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129 -U
ds_ssh -f $JOB_HOSTFILE uv pip install -r $PWD/requirements.txt --override $PWD/overrides.txt

# flash-attn builds against the freshly installed torch
ds_ssh -f $JOB_HOSTFILE uv pip install -U pip wheel packaging setuptools
```

To install flash attention, you can build it from source (may take a long time to build):
```bash
ds_ssh -f $JOB_HOSTFILE uv pip install flash-attn --no-build-isolation
```
or you can install directly from a wheel, find the automatic instructions
[here](https://windreamer.github.io/flash-attention3-wheels/) or download directly from
https://github.com/Dao-AILab/flash-attention/releases.

Install verl (Snowflake fork) editable on all nodes:
```bash
cd ../../../../../verl
grep -v flash-attn requirements.txt > requirements-no-fa.txt
ds_ssh -f $JOB_HOSTFILE "cd $PWD && uv pip install -r requirements-no-fa.txt && uv pip install -e ."
cd -
```


## 3. Data preparation

`download_data.py` pulls the three subset pairs from HuggingFace, prepends a system prompt that asks the model to
think inside `<think>` tags and answer inside `\boxed{}`, drops any non-verl columns, writes per-task and merged
train/test parquets.

From the recipe directory (`Arctic-Platform/recipes/rl/verl/long_context_qa`, cloned in step 2):

```bash
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

The training recipe consumes `merged/train.parquet` and `merged/test.parquet` by default.

If you want to train on a single task, point the training command at that task's `{train,test}.parquet` instead of
`merged/`.

## 4. Train

First start the Ray cluster across all nodes (now that the environment is installed everywhere, and `$JOB_HOSTFILE`
is exported from step 1):
```bash
bash ./restart_multi_ray.sh
```

Next edit the environment variables in `run_qwen3_32b_longcontext_grpo_arl.sh` to match your setup. In particular:
- `HF_HOME` - where you HF hub cache is (you can unset it as well)
- `VLLM_CACHE_ROOT` - some path where vllm could cache its work

```bash
bash run_qwen3_32b_longcontext_grpo_arl.sh \
    data.train_files=/data/snowflakesql/long-context/merged/train.parquet \
    data.val_files=/data/snowflakesql/long-context/merged/test.parquet
```

Alternatively you can edit `DATA_DIR` in the script (defaults to `/data/snowflakesql/long-context`) and launch with
no overrides:

```bash
bash run_qwen3_32b_longcontext_grpo_arl.sh
```

The answer reward is scored by `reward.py` (shipped with this recipe and auto-wired in the launcher via
`custom_reward_function`): it extracts the model's `\boxed{}` answer and matches it against the ground truth. Upstream
`verl` has no built-in scorer for this dataset's `data_source`, so the recipe supplies its own — no extra setup needed.

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
| `USE_KL_LOSS` | False | Pure GRPO; set `True` to add low-variance KL vs. a frozen ref |
