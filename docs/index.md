# Arctic Platform documentation

Reference docs for the post-training stack. High-level overview and install
live in the [project README](../README.md).

## Architecture

```
┌──────────────────────────────────────────────┐
│  Framework / script (CPU or GPU driver)      │
│  SFT client  ·  RL client (verl / SkyRL)     │
└──────────────────────┬───────────────────────┘
                       │ HTTP or Ray
                       ▼
┌──────────────────────────────────────────────┐
│  arctic_platform.common                      │
│  DeepSpeed workers · vLLM/ArcticInference    │
│  HTTP / Ray servers                          │
└──────────────────────────────────────────────┘
```

## Docs

| Doc | Contents |
|-----|----------|
| [**SFT**](sft.md) | Supervised fine-tuning — CPU client, wire batch, `sft` vs `sft_ce`, config |
| [**RL**](rl.md) | Reinforcement learning — engines, client API, ZoRRo Train / Inference, integrations |
| [**Common**](common.md) | Shared server infra — HTTP CLI, jobs/endpoints, DeepSpeed worker, metrics, env |

## Quick paths

- **SFT smoke (colocated):** see [sft.md § Quick start](sft.md#quick-start)
- **RL starter recipe:** [`recipes/rl/verl/simple/`](../recipes/rl/verl/)
- **SkyRL recipes:** [`recipes/rl/skyrl/`](../recipes/rl/skyrl/)
- **verl plugin:** [`arctic_platform/integrations/verl/`](../arctic_platform/integrations/verl/)
- **ZoRRo Train design:** [`arctic_platform/rl/zorro_train/README.md`](../arctic_platform/rl/zorro_train/README.md)

## Install

The base install carries config models only; pick the extra for your backend:

```bash
pip install "arctic-platform[cortex]"   # drive Cortex training
pip install "arctic-platform[sft]"      # local training-only server
pip install "arctic-platform[rl]"       # ...plus the sampling stack
# or from a checkout:
pip install -e ".[rl]"
```

Details: [README § Installation](../README.md#installation).
