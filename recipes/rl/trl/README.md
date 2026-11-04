# Arctic RL × TRL Recipes

End-to-end recipes for training models with [Arctic RL](../../../arctic_platform/rl/)
on top of [TRL](https://github.com/huggingface/trl) async-GRPO
(`AsyncGRPOTrainer` + the Arctic `TrainingClientProtocol` / rollout / weight-transfer hooks).

Available recipes:

* [GSM8K](gsm8k) — smallest end-to-end loop (HF GSM8K + built-in `accuracy_reward`)
* [Txt2SQL (BIRD)](txt2sql) — Qwen3-1.7B on BIRD, native-TRL baseline **+** Arctic (no ZoRRo) **+** Arctic+ZoRRo

These recipes drive TRL's async-GRPO trainer with either:

* **Arctic** (`ArcticTrainingClient` / `ArcticRolloutWorker` / `ArcticWeightTransfer`) — C2 (no ZoRRo) and C3 (ZoRRo train + load balancer)
* **Native TRL** (`LocalTrainingClient` + stock `vllm serve`) — C1, the A/B baseline

The GRPO algorithm and logged train metrics live in `AsyncGRPOTrainer.compute_loss` in both cases.

## Install

Each recipe is a standalone folder with its own `requirements.txt`, `overrides.txt`,
launchers, and README. Same env across both — build it once and either recipe
launches from that env.

1. **Clone this repo and TRL main.** The paired trainer is
   [huggingface/trl](https://github.com/huggingface/trl) `main` at `9141962`
   (instantiated models, no `training_client` argument). That commit includes
   the merged [huggingface/trl#6676](https://github.com/huggingface/trl/pull/6676).

   ```bash
   git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
   git clone https://github.com/huggingface/trl
   git -C trl checkout 914196293543f609d0a8cb68b16154934e6b0c51
   cd Arctic-Platform/recipes/rl/trl
   ```

2. **Install pinned Python deps** into a fresh conda env. The validated stack
   is Python 3.12, torch 2.11 (cu130), and vLLM 0.31 — in-tree `arctic_platform.inference`
   owns the vLLM pin; `overrides.txt` forces the few transitive deps that
   otherwise drift.

   ```bash
   conda create -y -n trl_arl python=3.12
   conda activate trl_arl
   pip install -q uv
   uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 -U
   uv pip install -r <recipe>/requirements.txt --override <recipe>/overrides.txt
   uv pip install -e ../../..[trl]      # this repo; ../../.. is Arctic-Platform/
   uv pip install -e ../../../../trl    # the sibling TRL checkout from step 1
   ```

3. **FlashAttention-2** is required for correctness, not just speed. The Arctic
   training server packs sequences varlen-style; `flash_attention_2` turns
   position-id resets into block-diagonal `cu_seqlens`. SDPA attends across
   packed boundaries and corrupts per-token logprobs.

   ```bash
   uv pip install flash-attn --no-build-isolation
   ```

   or a prebuilt wheel from https://github.com/Dao-AILab/flash-attention/releases.

4. **Run** — `cd <recipe>/`, follow its README for data, then `bash run_*.sh`.

TRL+Arctic is **disaggregated** on the BIRD recipe (trainer GPUs disjoint from
sampling GPUs). The GSM8K starter defaults to colocate; pass `--no-colocate`
to match the BIRD layout.
