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

Conda environments are **node-local**: each node has its own `~/miniconda3` even when your code and data sit on
shared storage, so the environment must be created and populated on **every** node in `$JOB_HOSTFILE` (from
[step (1)](#1-ray-and-multi-node-hostfile)). Use a fresh, recipe-specific env name (don't reuse a shared/dev env)
so the install is actually exercised on all nodes.

We fan out with `ds_ssh` (the DeepSpeed multi-node helper) and install with `uv pip install --python
<env>/bin/python`, i.e. we address the env by **absolute path** rather than relying on `conda activate`. A
non-interactive `ds_ssh`/pdsh shell does not reliably keep `conda activate` in effect, so activation-based installs
can silently land in the base env; absolute paths avoid that (this is the same reason `restart_multi_ray.sh` starts
Ray via the env's absolute `ray` binary).

Pick the env name and resolve the env's `bin/` once (the path is identical on every node):
```bash
export CONDA_ENV=long_context_qa
CONDA_BASE=$(conda info --base)
ENV=$CONDA_BASE/envs/$CONDA_ENV/bin        # the env's python / uv / ds_ssh / ray live here, on each node
```

Bootstrap the env on the launching node first — this is what gives you `uv` and `ds_ssh`:
```bash
conda create -y -n $CONDA_ENV python=3.12
$ENV/python -m pip install -q uv
$ENV/uv pip install --python $ENV/python deepspeed       # provides $ENV/ds_ssh
```

Clone this repo (it carries `requirements.txt` and the launcher scripts) and the verl fork onto storage visible to
all nodes:
```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone -b arctic_rl_share_v0.7.1 --single-branch https://github.com/Snowflake-AI-Research/verl
cd Arctic-Platform/recipes/rl/verl/long_context_qa
```

Create the env on the remaining nodes (idempotent — the launching node already has it), then install the pinned
dependencies on all nodes. cuda-12.9 is assumed — if you use a different version change the `torch` index URL below
and the `cuda-bindings` pin in `requirements.txt`. `arctic-inference` patches vllm-0.18.0, so that exact version is
pinned in `requirements.txt`; `overrides.txt` forces the few transitive deps (flashinfer / numpy / transformers /
datasets) this recipe is validated against, which vLLM 0.18.0's metadata otherwise pins higher (without it the
single resolve is unsatisfiable).
```bash
# create the env on every node
# guarded: `conda create -y` on an existing env would wipe and recreate it, which would clobber the uv/ds_ssh you
# just bootstrapped on the launching node — the `[ -x ... ]` makes it a no-op there
$ENV/ds_ssh -f $JOB_HOSTFILE "[ -x $ENV/python ] || $CONDA_BASE/bin/conda create -y -n $CONDA_ENV python=3.12"
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/python -m pip install -q uv"

# torch (CUDA 12.9) first, then the rest of the pinned packages
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129 -U"
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python -r $PWD/requirements.txt --override $PWD/overrides.txt"
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python -U pip wheel packaging setuptools"
```

The launcher uses `flash_attention_2` by default, which the `flash-attn` package below
provides. You can build it from source (may take a long time to build):
```bash
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python flash-attn --no-build-isolation"
```
or download a prebuilt FA2 wheel directly from
https://github.com/Dao-AILab/flash-attention/releases.

To use `flash_attention_3` instead (faster on Hopper), install the matching `flash_attn_3`
wheel (find prebuilt FA3 wheels [here](https://windreamer.github.io/flash-attention3-wheels/)),
then enable it in the launcher: comment out the `flash_attention_v=flash_attention_2` line
and uncomment the GPU-type auto-selection block beneath it.

Install verl (Snowflake fork) editable on all nodes (the source is shared, but the editable install must register it
in each node's env):
```bash
cd ../../../../../verl
grep -v flash-attn requirements.txt > requirements-no-fa.txt
$ENV/ds_ssh -f $JOB_HOSTFILE "cd $PWD && $ENV/uv pip install --python $ENV/python -r requirements-no-fa.txt && $ENV/uv pip install --python $ENV/python -e ."
cd -
```

> For a **single-node** run you don't need `ds_ssh`: create the env, bootstrap `uv`, and run the same
> `$ENV/uv pip install --python $ENV/python ...` commands directly on the local node.


## 3. Data preparation

`download_data.py` pulls the three subset pairs from HuggingFace, prepends a system prompt that asks the model to
think inside `<think>` tags and answer inside `\boxed{}`, drops any non-verl columns, writes per-task and merged
train/test parquets.

From the recipe directory (`Arctic-Platform/recipes/rl/verl/long_context_qa`, cloned in step 2):

```bash
# Defaults: --output_dir /data/snowflakesql/long-context, test_ratio=0.05, seed=42
$ENV/python download_data.py --output_dir /data/snowflakesql/long-context
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
