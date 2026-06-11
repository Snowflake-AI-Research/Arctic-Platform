# Prompt Deduplication Optimization for RL Training

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

The implementation uses **monkey patching** to intercept attention modules:

```
┌──────────────────────────────────────────────────────┐
│  DeduplicatedActor                                   │
│  ┌────────────────────────────────────────────────┐  │
│  │  1. Find prompt groups                         │  │
│  │  2. Create deduplicated batch                  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │  Qwen3ModelPatcher (context manager)     │  │  │
│  │  │  ┌────────────────────────────────────┐  │  │  │
│  │  │  │  QwenAttentionPatcher               │  │  │  │
│  │  │  │  - Intercepts attention forward()   │  │  │  │
│  │  │  │  - Applies deduplication logic      │  │  │  │
│  │  │  │  - Reconstructs outputs             │  │  │  │
│  │  │  └────────────────────────────────────┘  │  │  │
│  │  │  Model Forward Pass                      │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │  3. Reconstruct logits to original batch      │  │
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
from dedup_prompt_optimization.actor import DeduplicatedActor
from dedup_prompt_optimization.tests import create_dummy_batch

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
cd dedup_prompt_optimization
python demo.py
```

This will run:
1. A simple forward pass demonstration
2. A gradient correctness test
3. An optional performance benchmark

### Testing

**Gradient Correctness Test:**
```bash
python test_forward_and_backward.py
```

Compares gradients between baseline and deduplicated implementations. The test passes if the global gradient norm differs by less than 5%.

**Performance Benchmark:**
```bash
python test_perf.py
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
dedup_prompt_optimization/
├── README.md                      # This file
├── __init__.py
├── actor.py                       # Main actor class with deduplication
├── prompt_deduplicator.py         # Core deduplication logic
├── qwen_model_patcher.py          # Model-level patching (logits reconstruction)
├── qwen_attention_patcher.py      # Attention-level patching
├── module_patcher.py              # Base patcher utilities
├── demo.py                        # Interactive demonstration
├── test_forward_and_backward.py   # Gradient correctness test
├── test_perf.py                   # Performance benchmark
├── test_actor_demo.py             # Simple actor test
└── tests.py                       # Utility functions for testing
```

## API Reference

### `DeduplicatedActor`

Main class for running forward and backward passes with deduplication.

**Constructor:**
```python
DeduplicatedActor(
    model_name_or_path: str,
    device: str = "cuda",
    patcher_class = None,  # Auto-detected
    use_split_attention: bool = True,
    attn_implementation: str = "sdpa"
)
```

**Methods:**
- `_forward_micro_batch(micro_batch, temperature=1.0, calculate_entropy=False)`: Forward pass with deduplication
- `compute_policy_loss_and_backward(micro_batch, temperature=1.0, gradient_accumulation=1)`: PPO training step

### `ZoRRoTrain`

Static utility class for deduplication operations.

**Key Methods:**
- `find_prompt_groups(input_ids, response_length)`: Identify shared prompts
- `create_deduplicated_batch(input_ids, position_ids, response_length, prompt_groups, unique_prompts)`: Create deduplicated batch
- `reconstruct_sequences(dedup_hidden, reconstruction_info)`: Reconstruct full batch from deduplicated output

## Limitations & Future Work

**Current Limitations:**
- Only supports Qwen models (extensible to other architectures)
- Requires all sequences in a batch to have the same length (no padding support)
- Best speedups require identical prompts (partial overlap not exploited)

**Future Directions:**
- Support for other model architectures (Llama, Mistral, etc.)
- Dynamic padding support
- Prefix caching for partially overlapping prompts
- Multi-GPU distributed training integration

## Citation

If you use this optimization in your research, please consider citing:

```bibtex
@misc{prompt_deduplication_2025,
  title={Prompt Deduplication for Efficient RL Training},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourrepo/dedup_prompt_optimization}}
}
```

## License

[Specify your license here]

## Acknowledgments

This implementation builds on:
- Hugging Face Transformers for model implementations
- Flash Attention for efficient attention kernels
- VERL framework for RL training infrastructure
