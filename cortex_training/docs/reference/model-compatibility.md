# Model Compatibility

The compatibility catalog is being converted from prose into structured data
under [`docs/_data/models.yaml`](../_data/models.yaml).

The current entries were recovered from the repository API snapshot. They have
not been revalidated as part of this documentation move, so `last_validated`
is intentionally unset.

| Model | Training | Inference | Methods Documented |
|---|---:|---:|---|
| `Qwen/Qwen3-0.6B` | Yes | Yes | None yet |
| `Qwen/Qwen3-1.7B` | Yes | Yes | None yet |
| `Qwen/Qwen3-8B` | Yes | Yes | LoRA, full SFT, GRPO |
| `Qwen/Qwen3.5-4B` | Yes | Yes | None yet |
| `Qwen/Qwen3.6-35B-A3B` | Yes | Yes | Full SFT example |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | No | Yes | None yet |
| `openai/gpt-oss-120b` | No | Yes | None yet |
| `zai-org/GLM-5.2` | Planned | Yes | None yet |
| `zai-org/GLM-5.2-FP8` | Planned | Yes | None yet |

Before this becomes a support guarantee, each row needs architecture, method,
precision, quantization, cache status, GPU requirements, context limit, recipe
links, and a concrete validation date.
