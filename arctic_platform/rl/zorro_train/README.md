# ZoRRo Train: Prompt Deduplication Optimization for RL Training

ZoRRO Train: Zero Redundancy Rollouts for Training

## Motivation

In reinforcement learning from human feedback (RLHF) and other RL-based language model training scenarios, a critical computational bottleneck emerges: **prompt redundancy**.

### The Problem

During RL training (e.g., PPO), the typical workflow involves:

1. **Sampling**: Generate multiple responses for the same prompt to explore different strategies
2. **Evaluation**: Score each response using a reward model
3. **Training**: Compute policy gradients by processing all prompt-response pairs

This creates significant computational waste:

```
Prompt A + Response 1  ──┐
Prompt A + Response 2  ──┤
Prompt A + Response 3  ──┼──→  Same prompt processed N times!
    ...                  │
Prompt A + Response N  ──┘
```

**Key Statistics:**
- In typical RL training, 80-95% of tokens are redundant prompt tokens
- For a 10K-token prompt with 10 responses of 1K tokens each:
  - Total: 110K tokens (10K × 10 + 1K × 10)
  - Unique: 20K tokens (10K prompt + 1K × 10 responses)
  - **Redundancy: 82%** of computation is wasted on duplicate prompts!

### Why This Matters

Transformer attention has **O(n²)** complexity. When the same prompt appears multiple times:
- **Memory**: N copies of the same prompt embeddings, attention keys, and values
- **Compute**: N identical attention computations over the same prompt tokens
- **Time**: Training throughput is severely limited

For long-context RL (prompts with 4K-32K tokens), this redundancy becomes the dominant cost.

## Solution

This implementation provides **automatic prompt deduplication** at the attention layer, transparently removing redundant computation while maintaining mathematical correctness.

### How It Works

The optimization operates in three phases:

#### 1. **Prompt Detection & Deduplication**
```
Input Batch (8 sequences):
┌─────────────────────────────────┐
│ [Prompt A][Response 1]          │
│ [Prompt A][Response 2]          │  ──→  Deduplicated Batch:
│ [Prompt B][Response 3]          │       [Prompt A][Response 1][Response 2]
│ [Prompt B][Response 4]          │       [Prompt B][Response 3][Response 4]
│     ...                         │
└─────────────────────────────────┘
```

- Identifies which sequences share identical prompts
- Creates a **single concatenated sequence** with each unique prompt appearing once
- Tracks reconstruction metadata for reversing the transformation

#### 2. **Optimized Attention**

Two attention strategies are available:

**Standard QKV Optimization** (`use_split_attention=False`):
- Compute Q, K, V projections on deduplicated batch
- Reconstruct to full batch shape
- Run standard causal attention

**Split Attention Optimization** (`use_split_attention=True`, default):
- Compute Q, K, V projections on deduplicated batch
- Split into prompt and response parts
- Run **two separate attention calls**:
  1. **Prompt-to-Prompt**: Deduplicated prompts attend to themselves
  2. **Response-to-Full**: Each response attends to its prompt + itself
- Replicate prompt results and concatenate with response results

The split attention approach saves additional computation by avoiding redundant prompt-to-prompt attention computations.

#### 3. **Transparent Reconstruction**

- Output logits are automatically reconstructed to match the original batch shape
- Gradients flow correctly through the deduplicated computation
- The rest of the model (embeddings, MLP layers, final projection) sees the expected batch shape

### Architecture

The optimization is delivered by **monkey-patching** a Hugging Face Qwen3 model.
There are two patchers:

- **`Qwen3ModelOncePatcher`** and **`QwenAttentionOncePatcher`** — patched once onto the model by the
  DeepSpeed worker (`arctic_platform/rl/deepspeed_worker.py`). Called with the full
  `[B, S]` batch, it deduplicates shared prompts internally, runs the packed
  forward/backward, and returns per-response-token `logprobs` / `entropy` in the
  original sample order (no padding). This is the path exercised by the GPU tests
  (`tests/zorro_train/test_once_patcher.py`).
- **`Qwen3ModelPatcher`** + **`QwenAttentionPatcher`** — a
  context-manager harness driven by `DeduplicatedActor`, used for the demo and
  gradient-correctness reference. And which could also be adapted to do the same work patching and unpatching the model before each batch.

The main functionality can be viewed as:

```
┌──────────────────────────────────────────────────────┐
│  ZoRRoTrain                                          |
│  ┌────────────────────────────────────────────────┐  │
│  │  1. Find prompt groups                         │  │
│  │  2. Create deduplicated batch                  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │  Qwen3ModelOncePatcher                   │  │  │
│  │  │  ┌────────────────────────────────────┐  │  │  │
│  │  │  │  QwenAttentionOncePatcher          │  │  │  │
│  │  │  │  - Intercepts attention forward()  │  │  │  │
│  │  │  │  - Applies deduplication logic     │  │  │  │
│  │  │  │  - Reconstructs outputs            │  │  │  │
│  │  │  └────────────────────────────────────┘  │  │  │
│  │  │  Model Forward Pass                      │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │  3. Reconstruct logits to original batch       │  │
│  └────────────────────────────────────────────────┘  │
│  Backward Pass (with patching active)                │
└──────────────────────────────────────────────────────┘
```

## Key Features

✅ **Mathematically Correct**: Gradients match baseline implementation (within numerical precision)
✅ **Transparent**: Works as a drop-in replacement for standard forward/backward
✅ **Flexible**: Supports multiple attention implementations (SDPA, Flash Attention 2/3)
✅ **Gradient Checkpointing**: Compatible with activation checkpointing
✅ **Mixed Precision**: Optimized for `bfloat16` training

## Getting Started

### Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers
```

For Flash Attention (optional, for best performance):
```bash
pip install flash-attn --no-build-isolation
```

### Basic Usage

```python
from arctic_platform.rl.zorro_train import DeduplicatedActor
from arctic_platform.rl.zorro_train.tests import create_dummy_batch

# Initialize actor with deduplication
actor = DeduplicatedActor(
    model_name_or_path="Qwen/Qwen3-4B",
    device="cuda",
    use_split_attention=True,  # Use optimized split attention (default)
    attn_implementation="flash_attention_3"  # or "sdpa", "flash_attention_2"
)

# Create a batch with shared prompts
batch = create_dummy_batch(
    batch_size=8,
    num_unique_prompts=2,  # 8 sequences, but only 2 unique prompts
    prompt_len=4096,
    response_len=512,
    device="cuda",
    include_training_fields=True
)

# Forward pass (automatically deduplicates)
entropy, log_probs = actor._forward_micro_batch(
    batch,
    temperature=1.0,
    calculate_entropy=True
)

# Training step with backward pass
actor.model.train()
metrics = actor.compute_policy_loss_and_backward(
    batch,
    temperature=1.0,
    gradient_accumulation=1
)

print(f"Policy loss: {metrics['actor/policy_loss']:.4f}")
```

### Running the Demo

```bash
python arctic_platform/rl/zorro_train/demo.py
```

This will run:
1. A simple forward pass demonstration
2. A gradient correctness test
3. An optional performance benchmark

### Supported models

Currently ZoRROTrain supports these model families:

* qwen3
* qwen3-moe
* qwen3-next-moe
* qwen3.6
* qwen3.6-moe

This spans dense, MoE, and hybrid (linear + full attention) architectures, and **more models will be added in the future**.

### Testing

The maintained correctness tests live in the top-level test suite (`tests/zorro_train/`):

```bash
pytest tests/zorro_train/                          # everything below

pytest tests/zorro_train/test_dedup.py             # CPU: dedup-algorithm round-trips (no model)
pytest tests/zorro_train/test_once_patcher.py      # GPU: Qwen3ModelOncePatcher forward/backward vs reference
pytest tests/zorro_train/test_seqlen_balancing.py  # CPU: sequence-length balancing
```

`test_once_patcher.py` sweeps the `tiny-random` Qwen3 checkpoints (dense, MoE, hybrid),
both `flash_attention_2` and `eager`, padded/unpadded batches, and all three
`logits_optimization` modes (`none` / `memory` / `compute`).

**Performance Benchmark:**
```bash
python arctic_platform/rl/zorro_train/test_perf.py
```

Measures forward and backward pass speedup on large batches with prompt deduplication.

## Performance Expectations

Expected speedups depend on the deduplication ratio:

| Scenario | Batch Size | Unique Prompts | Prompt Len | Response Len | Expected Speedup |
|----------|------------|----------------|------------|--------------|------------------|
| High dedup | 16 | 1 | 8K | 1K | ~2-4x |
| Medium dedup | 16 | 4 | 8K | 1K | ~1.5-2x |
| Low dedup | 16 | 16 | 8K | 1K | ~1x (no benefit) |

**Note**: Actual speedups depend on:
- Hardware (GPU compute vs memory bottleneck)
- Attention implementation (Flash Attention gives best results)
- Sequence lengths (longer prompts = more benefit)

## Project Structure

```
arctic_platform/rl/zorro_train/
├── README.md                  # This file
├── __init__.py                # Public exports (ZoRRoTrain, DeduplicatedActor, patchers)
├── zorro_train.py             # Core dedup algorithm: ZoRRoTrain + ReconstructionInfo
├── actor.py                   # DeduplicatedActor reference harness (demo / correctness)
├── qwen_model_patcher.py      # Qwen3ModelOncePatcher (production), Qwen3ModelPatcher, logprob/entropy kernels
├── qwen_attention_patcher.py  # Attention-level patching (reference path)
├── module_patcher.py          # ModuleReconstructionPatcher base class
├── seqlen_balancing.py        # Sequence-length balancing across micro-batches
├── demo.py                    # Interactive demonstration
├── test_perf.py               # Performance benchmark
└── tests.py                   # Batch builders + gradient/perf helpers (create_dummy_batch, ...)
```

Correctness tests live under `tests/zorro_train/`: `test_dedup.py` (CPU dedup-algorithm
round-trips), `test_once_patcher.py` (GPU `Qwen3ModelOncePatcher` forward/backward), and
`test_seqlen_balancing.py` (CPU sequence-length balancing).

## API Reference

All public symbols are re-exported from the package root:

```python
from arctic_platform.rl.zorro_train import (
    ZoRRoTrain,
    DeduplicatedActor,
    Qwen3ModelPatcher,
    QwenAttentionPatcher,
    ModuleReconstructionPatcher,
)
# Production patcher (imported from the submodule):
from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelOncePatcher
```

### `DeduplicatedActor`

Reference harness for running forward and backward passes with deduplication (used by
the demo and the gradient-correctness reference; production uses `Qwen3ModelOncePatcher`).

**Constructor:**
```python
DeduplicatedActor(
    model_name_or_path: str,
    device: str = "cuda",
    patcher_class = None,  # Auto-detected (Qwen -> Qwen3ModelPatcher)
    use_split_attention: bool = True,
    attn_implementation: str = "sdpa"
)
```

**Methods:**
- `_forward_micro_batch(micro_batch, temperature=1.0, calculate_entropy=False)`: Forward pass with deduplication
- `compute_policy_loss_and_backward(micro_batch, temperature=1.0, gradient_accumulation=1)`: PPO training step

### `ZoRRoTrain`

Static utility class implementing the core (model-free) deduplication tensor logic.

**Key Methods:**
- `find_prompt_groups(input_ids, response_length)`: Group rows by prompt identity; returns `(prompt_groups, unique_prompts)`
- `create_deduplicated_batch(input_ids, position_ids, response_length, prompt_groups, unique_prompts, attention_mask=None, use_unpad=False)`: Pack each unique prompt followed by its responses; returns `(dedup_input_ids, position_ids, reconstruction_info)`
- `reconstruct_sequences(dedup_hidden, reconstruction_info)`: Reconstruct the full batch from deduplicated per-token output
- `deduplicate_sequences(full_hidden, reconstruction_info)`: Inverse of `reconstruct_sequences`
- `extract_unpadded_responses_from_deduped_packed_ids(packed_ids, reconstruction_info, offset=0)`: Pull each rollout's own response tokens from the packed sequence
- `responses_in_orig_sample_order(packed_responses, reconstruction_info)`: Undo the prompt-group permutation back to original sample order

`reconstruction_info` is a `ReconstructionInfo` (dict subclass) returned by
`create_deduplicated_batch` and threaded through the reconstruct/deduplicate helpers.

## Limitations & Future Work

**Current Limitations:**
- Best speedups require identical prompts (partial overlap not exploited)

**Future Directions:**
- Support for other model architectures (Llama, Mistral, etc.)
- Prefix caching for partially overlapping prompts

## Citation

If you use this optimization in your research, please consider citing:

```bibtex
@misc{zorro_train_2025,
  title={ZoRRO Train: Zero Redundancy Rollouts for Efficient RL Training},
  author={Snowflake AI Research},
  year={2025},
  howpublished={\url{https://github.com/Snowflake-AI-Research/Arctic-Platform}}
}
```

## License

Apache License 2.0. See the repository
[LICENSE](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/LICENSE).

## Acknowledgments

This implementation builds on:
- Hugging Face Transformers for model implementations
- Flash Attention for efficient attention kernels
- VERL framework for RL training infrastructure
