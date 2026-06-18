[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/arctic-platform.svg)](https://pypi.org/project/arctic-platform/)

<h3 align="center">
  <img src="docs/images/arctic_platform_logo.svg" width=500px><br>
</h3>

# ArcticPlatform: Simplifying and Accelerating Post-Training for LLMs

ArcticPlatform is a framework designed to simplify and accelerate the post-training process for large language models (LLMs). It addresses challenges in current frameworks, such as limited support for rapid prototyping and the lack of native data generation tools, by offering modular trainer designs, simplified code structures, and integrated pipelines for creating and cleaning synthetic data. These features enable users to enhance LLM capabilities, like code generation and complex reasoning, with greater efficiency and flexibility. Read more about ArcticPlatform [in our blog](https://www.snowflake.com/en/engineering-blog/arcticplatform-llm-post-training-framework/).

This is a work in progress, starting with the RL components, later integrating more training and inference components.

# Project Scope

ArcticPlatform aims to cover the full post-training stack for LLMs behind a small, composable API. The codebase is being built out incrementally:

- **Reinforcement Learning (available today)** — a high-throughput RL training/inference backend that plugs into existing RL frameworks (see below).
- **ZoRRO Train (available today)** — a prompt-deduplication optimization that removes redundant prompt computation during RL training (see below).
- **Coming next** — additional trainers (SFT/distillation), synthetic data generation and cleaning pipelines, and tighter inference integration.

## Reinforcement Learning

Arctic RL is designed to **integrate into existing RL frameworks** rather than replace them. The RL framework keeps ownership of the training loop, rollouts, rewards, and advantage estimation; ArcticPlatform provides the heavy compute engines behind a thin client:

- **Training engine** — a DeepSpeed engine that runs forward/backward and the optimizer step.
- **Log-prob / reference engine** — a forward-only DeepSpeed engine for reference / old log-prob computation.
- **Sampling engine** — a vLLM engine for fast rollouts.

These engines are orchestrated over [Ray](https://www.ray.io/), can be **colocated** on shared GPUs (via fractional Ray resources) or split across separate GPUs, and keep the sampler in sync with the trainer through NCCL or CUDA-IPC weight transfer. The trainer ↔ engine communication runs over either Ray or HTTP.

A framework integrates by constructing a client and driving the standard operations (`generate`, forward/backward, optimizer `step`, `sync_weights`, and `wake`/`sleep` for memory management):

```python
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client

config = ArcticRLClientConfig(
    model_name="Qwen/Qwen3-4B",
    comm_protocol="ray",        # or "http"
    training_gpus=1,
    sampling_gpus=1,
    log_prob_gpus=0,
    colocate=False,
)
client = create_arctic_rl_client(config)
```

The reference integration is [verl](https://github.com/volcengine/verl) (https://github.com/verl-project/verl/pull/6422), which drives Arctic RL from its PPO/GRPO trainer. End-to-end recipes live under [`arctic_platform/rl`](arctic_platform/rl/README.md), including [Txt2SQL](arctic_platform/rl/projects/txt2sql) and [long-context QA](arctic_platform/rl/projects/long_context_qa).

Many more frameworks integrations are in works and will be added here once available.

### ZoRRO Train

In RL training (PPO/GRPO) the same prompt is sampled many times to explore different responses, so **80–95% of the tokens in a batch are redundant prompt tokens** — and with transformer attention's O(n²) cost, recomputing those shared prompts dominates the bill for long-context RL.

**ZoRRO Train** eliminates that waste with automatic **prompt deduplication** at the attention layer: it detects sequences that share a prompt, packs each unique prompt once, runs the model a single time over the deduplicated sequence, and transparently reconstructs per-response `logprobs`/`entropy` in the original sample order. The result is mathematically equivalent to the naive forward/backward (gradients match the baseline within numerical precision) while substantially cutting memory use and increasing throughput — the longer and more-shared the prompts, the larger the win.

It is installed transparently by the DeepSpeed training/log-prob engines and toggled per run via the RL config (`zorro_train.enable`).

Supported model families today:

- `qwen3`
- `qwen3-moe`
- `qwen3-next-moe`
- `qwen3.6`
- `qwen3.6-moe`

This spans dense, MoE, and hybrid (linear + full attention) architectures, and **more models will be added in the future**.

See [`arctic_platform/rl/zorro_train/README.md`](arctic_platform/rl/zorro_train/README.md) for the full design, the deduplication/attention internals, and benchmarks.

# Installation

## From PyPI

Install the latest released version and its dependencies from [PyPI](https://pypi.org/project/arctic-platform/):

```bash
pip install arctic-platform
```

## From source (git)

To get the latest development version (or to contribute), clone the repository and install it in editable mode:

```bash
git clone https://github.com/Snowflake-AI-Research/arctic-platform.git
cd arctic-platform
pip install -e .
```

# Quickstart

To get started training a model with ArcticPlatform, first [install the package](#installation), then follow the recipes under [`arctic_platform/rl`](arctic_platform/rl/README.md).
