# Simple single-GPU GRPO with Arctic RL (GSM8K)

The smallest end-to-end Arctic RL recipe: GRPO training for **Qwen3-1.7B** on **GSM8K**, on a **single GPU**, served
by [Arctic RL](../../../../arctic_platform/rl/) with the [ZoRRo](../../../../arctic_platform/rl/zorro_train/) trainer.
Pure GRPO, without a frozen reference model (no KL anchoring).

This recipe is meant as a quick way to get a full Arctic RL loop running on one GPU — there is **no Ray cluster setup
and no hostfile**. verl starts a local Ray instance automatically.

Topology: 1 node, 1 GPU, `colocate=True` (training + sampling share the GPU), Deepspeed ZeRO stage-2, vLLM rollout
(TP=1). Without KL there is no frozen reference model, so the ref log-prob pool is disabled (`log_prob_gpus=0`);
under ZoRRo log-probs are recomputed through the training engine itself.

GSM8K is scored by verl's **built-in** reward (`data_source="openai/gsm8k"`: exact match on the `#### <number>` final
answer), so this recipe ships no custom reward function.

## 1. Install packages

This is a single-node, single-GPU recipe, so there is no `ds_ssh` fan-out — everything installs into one conda env on
the local node. Create a fresh, recipe-specific env (don't reuse a shared/dev env, so the install is actually
exercised) and use `uv` for much faster installs:
```bash
conda create -y -n simple python=3.12
conda activate simple
pip install uv
```

Clone this repo (it carries `requirements.txt` and the launcher script) and the verl fork:
```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone -b arctic_rl_share_v0.7.1 --single-branch https://github.com/Snowflake-AI-Research/verl
cd Arctic-Platform/recipes/rl/verl/simple
```

Install the pinned dependencies. The assumption is cuda-12.9 - if you use a different version change the `torch` index
URL below and the `cuda-bindings` pin in `requirements.txt`. `arctic-inference` patches vllm-0.18.0, so that exact
version is pinned in `requirements.txt`.
```bash
# torch (CUDA 12.9) first, then the rest of the pinned packages.
# overrides.txt forces the few transitive deps (flashinfer/numpy/transformers)
# this recipe is validated against, which vLLM 0.18.0's metadata otherwise pins
# higher (without it the single resolve is unsatisfiable).
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129 -U
uv pip install -r requirements.txt --override overrides.txt

# flash-attn builds against the freshly installed torch
uv pip install -U pip wheel packaging setuptools
```

To install flash attention, you can build it from source (may take a long time to build):
```bash
uv pip install flash-attn --no-build-isolation
```
or you can install directly from a wheel, find the automatic instructions
[here](https://windreamer.github.io/flash-attention3-wheels/) or download directly from
https://github.com/Dao-AILab/flash-attention/releases.

Install verl (Snowflake fork) editable:
```bash
cd ../../../../../verl
grep -v flash-attn requirements.txt > requirements-no-fa.txt
uv pip install -r requirements-no-fa.txt
uv pip install -e .
cd -
```

## 2. Data preparation

`download_data.py` pulls GSM8K from HuggingFace and writes verl-compatible train/test parquets
(`data_source="openai/gsm8k"`, with the gold `#### <number>` answer as the reward ground truth).

From the recipe directory (`Arctic-Platform/recipes/rl/verl/simple`, cloned in step 1):

```bash
# Default: --output_dir ~/data/gsm8k
python download_data.py --output_dir ~/data/gsm8k
```

Output layout:
```
~/data/gsm8k/
├── train.parquet      # ~7.5k rows
└── test.parquet       # ~1.3k rows
```

## 3. Train

The launcher needs **no Ray cluster or hostfile** — just run it. It defaults to a single GPU and reads the parquets
produced in step 2.

You may want to edit the environment variables at the top of `run_qwen3_1.7b_gsm8k_grpo_arl.sh`:
- `HF_HOME` - where your HF hub cache is (left online by default so the model and dataset download on first run)
- `VLLM_CACHE_ROOT` - some path where vLLM could cache its work

```bash
bash run_qwen3_1.7b_gsm8k_grpo_arl.sh \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet
```

Alternatively, if you kept the default data path (`DATA_DIR` defaults to
`~/data/gsm8k`), launch with no overrides:

```bash
bash run_qwen3_1.7b_gsm8k_grpo_arl.sh
```

Key recipe knobs (set inside the script):

| Knob | Default | Notes |
| --- | --- | --- |
| `NGPU_PER_JOB` | 1 | Single GPU |
| `PROMPT_LEN` | 1024 | GSM8K prompts are short |
| `RESPONSE_LEN` | 1024 | |
| `ROLL_N` | 8 | GRPO group size |
| `MAX_TOKENS_PER_GPU` | 16384 | ≥ `PROMPT_LEN + ROLL_N * RESPONSE_LEN` so each GRPO group fits a ZoRRo tile |
| `BSZ` | 32 | Train batch size (data) |
| `PPO_MINI_BSZ` | 32 | Actor mini-batch |
| `LR` | 1e-6 | |
| `ARCTIC_ZERO_STAGE` | 2 | 1.7B fits on one GPU; no offload |
| `USE_KL_LOSS` | False | Pure GRPO; set `True` to add low-variance KL vs. a frozen ref |
