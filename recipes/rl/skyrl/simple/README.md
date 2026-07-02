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

## 1. Install

Same env as the sibling `txt2sql/` and `long_context_qa/` recipes — if you've
built either of those, `conda activate skyrl_arl` and skip step 2.

```bash
# 1. Clone SkyRL at the pinned merge commit on the ``arctic-rl-public`` branch.
git clone https://github.com/Snowflake-AI-Research/SkyRL
cd SkyRL && git checkout 7636101a71f1849b6127ee10232fb277d2f31174 && cd ..
export SKYRL_HOME=$PWD/SkyRL

# 2. Create the env.
conda create -y -n skyrl_arl python=3.12.13
conda activate skyrl_arl
pip install -q uv
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r requirements.txt --override overrides.txt
```

FlashAttention 3 (Hopper-only) is pulled by `arctic-inference[vllm]`. On
A100/L40S the recipe falls back to FA2 automatically.

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
