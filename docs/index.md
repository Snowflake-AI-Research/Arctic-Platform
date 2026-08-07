# Arctic Platform documentation

Reference docs for the post-training stack. High-level overview and install
live in the [project README](../README.md).

## Architecture

```
┌──────────────────────────────────────────────┐
│  Framework / script (CPU or GPU driver)      │
│  RL client (verl / SkyRL)  ·  SFT (planned)  │
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
| [**RL**](rl.md) | Reinforcement learning — engines, client API, ZoRRo Train / Inference, integrations |
| [**Common**](common.md) | Shared server infra — HTTP CLI, jobs/endpoints, DeepSpeed worker, metrics, env |
| **SFT** (forthcoming) | Supervised fine-tuning — CPU client + remote DeepSpeed server (separate PR) |

## Quick paths

- **RL starter recipe:** [`recipes/rl/verl/simple/`](../recipes/rl/verl/)
- **SkyRL recipes:** [`recipes/rl/skyrl/`](../recipes/rl/skyrl/)
- **verl plugin:** [`arctic_platform/integrations/verl/`](../arctic_platform/integrations/verl/)
- **ZoRRo Train design:** [`arctic_platform/rl/zorro_train/README.md`](../arctic_platform/rl/zorro_train/README.md)

## Install

```bash
pip install arctic-platform          # PyPI
# or from a checkout:
pip install -e .[rl]
```

Details: [README § Installation](../README.md#installation).
