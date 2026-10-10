# Cortex Training Recipes

Recipes are runnable, end-to-end workflows organized by training task. Each
recipe owns its code, documentation, metadata, optional notebooks, and future
typed configuration examples.

## Available Recipes

| Recipe | Method | Dataset | Status |
|---|---|---|---|
| [Conversational SFT](sft/conversational/README.md) | LoRA or full-parameter SFT | Hugging Face chat datasets | Runnable |
| [Math GRPO](rl/math_grpo/README.md) | Reinforcement learning | Hendrycks MATH and MATH-500 | Runnable |
| [Sampling walkthrough](inference/sampling/README.md) | Inference and sampling | User prompts | Notebook |

## Planned Recipes

- [Continued pre-training](continued_pretraining/README.md)
- [Preference optimization](alignment/README.md)
- [Knowledge distillation](distillation/README.md)
- [Tool-use training](tool_use/README.md)
- [Multimodal training](multimodal/README.md)
- [Framework integrations](integrations/README.md)

## Prerequisites

Install the client and recipe dependencies from the repository root:

```bash
uv pip install -e .
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Create a local connection config from
`examples/config/connection.json.template`. Do not commit credentials.

## Running Recipes

Recipes are Python modules so they can share code without path manipulation:

```bash
python -m recipes.sft.conversational.train config=/path/to/config.json
python -m recipes.rl.math_grpo.train config=/path/to/config.json lora_rank=32
```

Pass configuration overrides as `name=value` arguments. See each recipe README
for hardware, expected metrics, and common variations.

## Recipe Contract

Every runnable recipe should provide:

- A README with outcome, prerequisites, hardware, commands, expected results,
  configuration knobs, evaluation, and troubleshooting
- `recipe.yaml` metadata used by the compatibility catalog
- A runnable entry point
- A last-validated date and environment
- Tests for local validation logic where practical
- Attribution when adapted from an upstream cookbook or framework

See [the recipe template](../docs/contributing/recipe-template.md) before adding
a new workflow.
