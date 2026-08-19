<!--
Copyright 2025 Snowflake Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Arctic verl integration — V1 setup (from scratch)

End-to-end recipe for running the Arctic RL backend against
`verl-project/verl` `main` (V1 trainer). Verified on an 8× H200 node with
Python 3.12, torch 2.10.0+cu129, vLLM 0.18.0, flashinfer 0.6.6.

Companion PRs:

- verl-core (V1 seam): <https://github.com/verl-project/verl/pull/7102>
- Arctic-Platform (this PR): <https://github.com/Snowflake-AI-Research/Arctic-Platform/pull/41>

---

## 1. Layout

Pick a fast-scratch root (below assumes `/data-fast/karthik/`) and a
sources root (below assumes `/modeling-code/karthik/abstract-remote-exps/`).
Environment goes on the fast scratch; source checkouts can live anywhere
you can `pip install -e`.

```
/data-fast/karthik/conda_envs/arctic_verl_v18/         # conda env
/modeling-code/karthik/abstract-remote-exps/
    ├── verl/                        # verl-project/verl @ main + this PR
    ├── Arctic-Platform/             # this repo, karthik/verl-v1-plugin
    ├── ArcticInference-internal/    # Arctic vLLM plugin
    ├── ArcticTraining-dss/          # arctic-training training library
    └── dss-client/                  # DSS client (dependency of arctic-training)
```

## 2. Clone the sources

```bash
SRC=/modeling-code/karthik/abstract-remote-exps
mkdir -p "$SRC" && cd "$SRC"

git clone https://github.com/verl-project/verl.git
git -C verl fetch origin karthik/v1-remote-backend:v1-remote-backend
git -C verl checkout v1-remote-backend

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform.git
git -C Arctic-Platform checkout karthik/verl-v1-plugin

# Internal repos (Snowflake org access required)
git clone https://github.com/Snowflake-AI-Research/ArcticInference-internal.git
git clone https://github.com/Snowflake-AI-Research/ArcticTraining-dss.git
git clone https://github.com/Snowflake-AI-Research/dss-client.git
```

## 3. Create the env

```bash
ENV=/data-fast/karthik/conda_envs/arctic_verl_v18
conda create -y -p "$ENV" python=3.12
conda activate "$ENV"
python -m pip install --upgrade pip
```

## 4. Install packages (order matters)

```bash
cd "$SRC"

# 4a. torch first, pinned to the cu129 wheels.
python -m pip install --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.10.0 torchvision

# 4b. dss-client BEFORE arctic-training (arctic-training imports it at install-time).
python -m pip install -e dss-client

# 4c. arctic-training (depends on dss-client).
python -m pip install -e ArcticTraining-dss

# 4d. Arctic inference vLLM plugin (installs vllm==0.18.0 as a dep).
python -m pip install -e ArcticInference-internal

# 4e. Arctic-Platform (this repo).
python -m pip install -e Arctic-Platform

# 4f. verl (V1 branch with the RemoteBackend seam).
python -m pip install -e verl

# 4g. flashinfer downgrades torch as a side effect; reinstall torch afterwards.
python -m pip install flashinfer-python
python -m pip install --index-url https://download.pytorch.org/whl/cu129 \
    --force-reinstall --no-deps torch==2.10.0 torchvision
```

Sanity:

```bash
python - <<'PY'
import torch, vllm, flashinfer, verl, arctic_platform, arctic_inference, arctic_training
print("torch      :", torch.__version__, "cuda", torch.version.cuda)
print("vllm       :", vllm.__version__)
print("flashinfer :", flashinfer.__version__)
print("verl       :", verl.__file__)
PY
```

Expected: `torch 2.10.0+cu129 cuda 12.9`, `vllm 0.18.0`, `flashinfer 0.6.6`.

## 5. Data + reward function

The BIRD text-to-SQL launcher expects:

- `/data/snowflakesql/txt2sql/train.parquet` (see the Snowflake-internal
  data mirror; earlier ablations also used
  `train_128_avg9k.parquet` if the full file is unavailable).
- validation parquet at `/code/users/truwase/data/open-source-text2sql/val.parquet`
  (override via `VAL_FILES`).
- Reward function at
  `Arctic-Platform/recipes/rl/verl/txt2sql/bird_reward.py` (already in
  this repo). Launcher wires it via
  `reward.custom_reward_function.path`.

GSM8K launcher expects `/code/shared/gsm8k/{train,test}.parquet`.

## 6. Run

Smoke test on 1× GPU (1.7B, GSM8K):

```bash
cd "$SRC/Arctic-Platform"
bash arctic_platform/integrations/verl/v1/examples/run_gsm8k_grpo_arl_v1.sh
```

8× H200, Qwen3-8B, BIRD (recipe-aligned):

```bash
bash /data-fast/karthik/run_bird_arctic_v1_recipe.sh
```

## 7. Env vars the plugin cares about

Set in the launcher; override on the command line if needed.

| Variable | Purpose |
| :--- | :--- |
| `VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register` | Loads the plugin at verl bootstrap. |
| `USE_ARCTIC_TRAINING_CLIENT=1` | Route training through the Arctic RL server. |
| `ARCTIC_INFERENCE_ENABLED=1` | Register the Arctic vLLM plugin (Forest Cascade Attention hooks). |
| `VLLM_BATCH_INVARIANT=0` | Off = throughput mode. Set to `1` only for bit-exact reproducibility (disables CUDA graphs / speculative decoding). |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | Required by the Arctic vLLM plugin. |
| `HF_HOME` | Model cache. Point at the shared cache if available (`/checkpoint/huggingface`). |
| `HF_HUB_OFFLINE=0` | Allow HF hub access on first pull; flip to `1` once the model is cached. |

## 8. Known follow-up

Under vLLM 0.18.0 the default `cudagraph_mode=FULL_AND_PIECEWISE`
silently disables Forest Cascade Attention when `enforce_eager=False`.
The pre-plugin recipe worked around this with
`actor_rollout_ref.rollout.enforce_eager=True` (forces
`cudagraph_mode=NONE`), which the current launcher also does. Tracking
a real fix in `parse_arctic_inference_rollout` (override
`compilation_config.cudagraph_mode=PIECEWISE` when
`zorro_inference.enable=True`) in a separate PR.
