# Simple — single-GPU GRPO on GSM8K

Single-GPU GRPO for **Qwen3-0.6B** on **GSM8K**, driven by SkyRL's PPO trainer with the
[Arctic RL](../../../arctic_platform/rl/) backend and the
[ZoRRo](../../../arctic_platform/rl/zorro_train/) optimization layer. Pure GRPO; no frozen
reference model.

This is the entry-point recipe: 1 GPU, 1 node, **no Ray cluster setup and no hostfile**.
SkyRL boots a local Ray instance automatically.

| Knob              | Value |
| ---               | --- |
| Model             | `Qwen/Qwen3-0.6B` |
| Reward            | SkyRL built-in `gsm8k` env (exact match on `#### <number>`) |
| Trainer           | DeepSpeed ZeRO-2, no offload |
| Sampling          | vLLM 0.18.0 (TP=1, 1 engine) |
| GPU layout        | Arctic RL colocates train + sample on the single GPU |
| Sequence lengths  | prompt 512, response 1024 |

## 1. Install packages

Use a fresh conda env so the install is actually exercised (don't share with a dev env):

```bash
conda create -y -n skyrl_simple python=3.12
conda activate skyrl_simple
pip install uv
```

Clone Arctic-Platform (this recipe) and SkyRL (Arctic RL × SkyRL integration code).
The recipe uses SkyRL's `integrations/arctic_rl/` directory, which is *not* in the
pip-installed `skyrl` package — so a checkout at the pinned commit is required:

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform

git clone https://github.com/NovaSky-AI/SkyRL
cd SkyRL && git checkout 76f5f467c6804e8acc6273cc677098b7679b0315 && cd ..
export SKYRL_HOME=$PWD/SkyRL

cd Arctic-Platform/recipes/rl/skyrl/simple
```

Install pinned deps. Assumes **CUDA 12.8** — change the `torch` index URL below if you're
on a different CUDA version. `overrides.txt` forces `transformers==4.57.6`, which both
vLLM 0.18.0 and the Arctic RL trainer are validated against.

```bash
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r requirements.txt --override overrides.txt
uv pip install -U pip wheel packaging setuptools
uv pip install \
    "flash-attn@https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

On Hopper (H100/H200) you can also install FlashAttention 3 from PyTorch's cu128 index for
faster decode; the multi-node BIRD recipes do that by default. This recipe sticks with FA2
since it's broadly available.

## 2. Data preparation

`download_data.py` pulls GSM8K from Hugging Face and writes the train/validation parquets
SkyRL's `gsm8k` env expects (`env_class="gsm8k"`, gold `#### <n>` as `reward_spec.ground_truth`):

```bash
python download_data.py --output_dir ~/data/gsm8k
```

Result:

```
~/data/gsm8k/
├── train.parquet         ~7.5k rows
└── validation.parquet    ~1.3k rows
```

## 3. Train

```bash
bash run_qwen3_0.6b_gsm8k_grpo_arl.sh
```

Common overrides (env vars consumed by the script):

```bash
LOGGER=wandb                                    # default: console
DATA_DIR=~/data/gsm8k                           # default
CKPT_DIR=~/checkpoints/<run-name>               # default: ~/checkpoints/<exp>/
MODEL=Qwen/Qwen3-1.7B                           # default: Qwen/Qwen3-0.6B
```

You can also pass any SkyRL Hydra override straight through:

```bash
bash run_qwen3_0.6b_gsm8k_grpo_arl.sh \
    trainer.train_batch_size=64 \
    generator.n_samples_per_prompt=8
```

## How this is wired

- `trainer.override_entrypoint=integrations.arctic_rl.entrypoint` tells SkyRL's `main_base`
  to dispatch into the Arctic RL × SkyRL glue in your `$SKYRL_HOME` clone. This recipe
  ships zero Python — the launcher sets `PYTHONPATH=$SKYRL_HOME` and dispatches straight
  to upstream's Ray entrypoint. Workers pick up `$SKYRL_HOME` automatically because
  upstream's entrypoint forwards it onto their `runtime_env`.
- `environment.env_class=gsm8k` resolves to the GSM8K env registered by upstream `skyrl_gym`.
- `trainer.arctic_rl.colocate=true` puts Arctic RL's training and sampling jobs on the
  same GPU; `trainer.placement.colocate_all=false` keeps SkyRL from also trying to grab a
  placement group for the GPU Arctic RL already owns.

Once this works, the multi-GPU recipes ([txt2sql](../txt2sql),
[long_context_qa](../long_context_qa)) are the same shape scaled up.
