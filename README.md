[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/arctic-platform.svg)](https://pypi.org/project/arctic-platform/)

<h3 align="center">
  <img src="docs/images/arctic_platform_logo.svg" width=500px><br>
</h3>

# Arctic Platform: Simplifying and Accelerating Post-Training for LLMs

Arctic Platform is a framework for addressing challenges in current post-training frameworks, such as limited support for rapid prototyping and the lack of native data generation tools, by offering modularity across training and inference components, simplified code structures, and integrated pipelines for creating and cleaning synthetic data. These features enable users to enhance LLM capabilities, like code generation and complex reasoning, with greater efficiency and flexibility.

This is a work in progress, starting with the RL components, later integrating more training and inference components.

# Project Scope

Arctic Platform aims to cover the full post-training stack for LLMs behind a small, composable API. The codebase is being built out incrementally:

* **Reinforcement Learning (available today)** — a high-throughput RL training/inference backend that plugs into existing RL frameworks (see below).
* **ZoRRo Train (available today)** — a prompt-deduplication optimization that removes redundant prompt computation during RL training (see below).
* **ZoRRo Inference (available today)** — forest cascade attention for efficient rollout step that eliminates redundant memory accesses via grouping (see below).
* **Coming next** — additional trainers (SFT/distillation), synthetic data generation and cleaning pipelines, and tighter inference integration.

## Arctic Reinforcement Learning

Arctic RL is designed to **integrate into existing RL frameworks** rather than replace them. The RL framework keeps ownership of the training loop, rollouts, rewards, and advantage estimation; Arctic Platform provides the heavy compute engines behind a thin client:

* **Training engine** — a DeepSpeed engine that runs forward/backward and the optimizer step.
* **Log-prob / reference engine** — a forward-only DeepSpeed engine for reference / old log-prob computation.
* **Sampling engine** — a vLLM engine with ArcticInference for fast rollouts.

These engines are orchestrated over [Ray](https://www.ray.io/), can be **colocated** on shared GPUs (via fractional Ray resources) or split across separate GPUs, and keep the sampler in sync with the trainer through NCCL or CUDA-IPC weight transfer. The front-end to back-end communication runs over either Ray or HTTP. ArcticInference extends the core optimizations with multi-replica scheduling, load-balancing, weight-sync, and router-replay.

A framework integrates by constructing a client and driving the standard operations (generate, forward/backward, optimizer step, sync_weights, and wake/sleep for memory management):

```py
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

python

Copy

The reference integration is [verl](https://github.com/volcengine/verl) ([https://github.com/verl-project/verl/pull/6422](https://github.com/verl-project/verl/pull/6422)), which drives Arctic RL from its PPO/GRPO trainer. End-to-end recipes live under [arctic_platform/rl](arctic_platform/rl/README.md), including [Txt2SQL](https://raw.githubusercontent.com/Snowflake-AI-Research/Arctic-Platform/912de39206cf5f53d995cef1661e59c1b17c3b8f/arctic_platform/rl/projects/txt2sql) and [long-context QA](https://raw.githubusercontent.com/Snowflake-AI-Research/Arctic-Platform/912de39206cf5f53d995cef1661e59c1b17c3b8f/arctic_platform/rl/projects/long_context_qa).

The other upcoming integration is into [SkyRL](https://github.com/Snowflake-AI-Research/SkyRL/pull/1)

Many more frameworks integrations are in the works and will be added here once available.

### ZoRRo Train

In RL training (PPO/GRPO) the same prompt is sampled many times to explore different responses, therefore for long sequence tasks **80–95% of the tokens in a batch are redundant prompt tokens** — and with transformer attention’s O(n²) cost, recomputing those shared prompts dominates the bill for long-context RL.

**ZoRRo Train** eliminates that waste with automatic **prompt deduplication** at the attention layer: it detects sequences that share a prompt, packs each unique prompt once, runs the model a single time over the deduplicated sequence, and transparently reconstructs per-response logprobs/entropy in the original sample order. The result is mathematically equivalent to the naive forward/backward (gradients match the baseline within numerical precision) while substantially cutting memory use and increasing throughput — the longer and more-shared the prompts, the larger the win.

It is installed transparently by the DeepSpeed training/log-prob engines and toggled per run via the RL config (zorro_train.enable).

Supported model families today:

* qwen3
* qwen3-moe
* qwen3-next-moe
* qwen3.6
* qwen3.6-moe

This spans dense, MoE, and hybrid (linear + full attention) architectures, and **more models will be added in the future**.

See [arctic_platform/rl/zorro_train/README.md](arctic_platform/rl/zorro_train/README.md) for the full design, the deduplication/attention internals, and benchmarks.

### ZoRRo Inference

During RL rollouts many sequences are generated from the same prompt. In the decode step, standard attention re-reads the KV cache of those shared prefixes **once per request**, so the sampler spends most of its memory bandwidth fetching identical keys and values over and over.

**ZoRRo Inference** removes that waste with **Forest Cascade Attention (FCA)**, which deduplicates shared KV reads at the attention layer of the sampling engine. For each decode batch, FCA discovers groups of requests that share a KV-cache prefix. Then it splits each attention call into a single grouped pass over the shared prefix blocks plus a per-request pass over the unique suffix blocks. FCA then reduces the two partial results with rigorous weighting. This reads each shared prefix block once per *group* instead of once per *request*, cutting redundant memory accesses while remaining mathematically equivalent to standard attention — the longer and more-shared the prefixes, the larger the win.

It is implemented in the vLLM sampling engine and activates transparently for decode-heavy batches with shared prefixes.

See the [Forest Cascade Attention README](https://github.com/snowflakedb/ArcticInference/tree/arctic_rl/integration/arctic_inference/vllm/attention) in Arctic Inference for the full design, the grouping/attention internals, and the tuning knobs.

# Installation

## From PyPI

Install the latest released version and its dependencies from [PyPI](https://pypi.org/project/arctic-platform/):

```shell
pip install arctic-platform
```

bash

Copy

## From source (git)

To get the latest development version (or to contribute), clone the repository and install it in editable mode:

```shell
git clone https://github.com/Snowflake-AI-Research/arctic-platform.git
cd arctic-platform
pip install -e .
```

bash

Copy

# Quickstart

To get started training a model with Arctic Platform, first [install the package](https://raw.githubusercontent.com/Snowflake-AI-Research/Arctic-Platform/912de39206cf5f53d995cef1661e59c1b17c3b8f/README.md#installation), then follow the recipes under [arctic_platform/rl](arctic_platform/rl/README.md).
