# Improve Onboarding for Cortex Training

This roadmap tracks the documentation and recipe work needed for a complete
Cortex Training onboarding experience.

## P-1

### End-to-End Training Guides

Provide getting-started recipes for:

- LoRA and QLoRA
- Full-parameter fine-tuning
- Continued pre-training
- Reinforcement learning

Each guide must include a README, hardware requirements, and configuration
examples.

### Open-Weights Model Coverage

Maintain a compatibility matrix containing:

- Supported models and architectures
- Supported training methods
- Links to training examples
- Precision and quantization options
- Model-cache status
- GPU requirements
- Context limits
- Last-validated date

### Inference and Sampling

Provide examples for:

- Loading a trained checkpoint and creating an inference endpoint
- Open-weight serving
- LoRA serving with and without adapter merging
- Multi-LoRA serving
- Full-weight serving
- Scaling inference GPU capacity

### Job Management and Observability

Provide a dedicated guide covering:

- Listing and filtering jobs
- Status, logs, and checkpoints
- GPU utilization, memory, throughput, tokens per second, and MFU
- Weights & Biases configuration for loss, evaluation, reward, and KL
- Canceling, resuming, and retrying runs
- Tool-specific instructions and redacted real output

Current gap: complete GPU utilization and memory metrics are not yet available.

### Developer Documentation

Provide:

- Product overview
- Getting-started guide
- API reference

## P-2

### Typed Training and Sampling Configuration

Provide typed YAML and Python configuration for:

- Batch size and gradient accumulation
- Sequence length and packing
- Loss, optimizer, and scheduler
- Precision and distributed strategy
- Checkpoint and evaluation cadence
- Weights & Biases settings
- Sensible defaults

### Open-Source Recipe Ports

Adapt selected workflows from:

- Tinker Cookbook: conversational SFT, math and code RL, preference
  optimization, knowledge distillation, tool use, multi-agent RL, audio, and
  vision-language training
- nanoGPT and nanochat: tokenization, pre-training, SFT, RL, evaluation,
  inference, scaling, and Weights & Biases
- Hugging Face TRL and Alignment Handbook: SFT, continued pre-training, reward
  modeling, preference optimization, and reinforcement learning

### Framework Recipes

Demonstrate Cortex Training integrations for:

- Axolotl
- Unsloth
- SkyRL
- veRL

Target examples include math, code execution, search, and tool use.

### Evaluation

Provide:

- A base-model versus fine-tuned-model evaluation workflow
- RL evaluation with reward, success rate, and pass@k
- Checkpoint-over-time evaluation with task score and train/eval loss

## P-3

Use torchtune's hackable recipe and configuration patterns as a reference for
LoRA, full fine-tuning, DPO, GRPO, distributed execution, and command-line
configuration overrides.
