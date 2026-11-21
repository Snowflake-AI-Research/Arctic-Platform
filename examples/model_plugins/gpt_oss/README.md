# gpt-oss: mixture of experts with its own attention core, on vLLM's built-in model

[gpt-oss](https://huggingface.co/openai/gpt-oss-20b) is a mixture-of-experts decoder whose attention has a learned
*sink* per head: an extra logit in each softmax that absorbs probability and is then dropped. The platform has no
implementation of it, and its attention core has no sinks; vLLM serves it with its built-in `GptOssForCausalLM`.
Against the [OLMoE example](../olmoe/README.md) it adds:

| File | What it adds |
|---|---|
| `gpt_oss_plugin/modeling.py` | `GptOssAttention.mix`: the plugin's own attention core (sinks, sliding window, biases), on the platform's head-parallel policy, slicing its per-head sinks by `HeadShard` |
| `gpt_oss_plugin/__init__.py` | Two layer types, `sliding_attention` and `full_attention`, sharing that core; `StackedExpertLayer` over four expert tensors, biases included |
| `gpt_oss_plugin/modeling.py` | Experts kept in the checkpoint's stacked layout and names, so no converters; top-k by router logit, then a softmax over the chosen experts; a clamped SwiGLU; YaRN rotary from transformers' own initialization |
| `gpt_oss_plugin/manifest.toml` | `expert_parallel` and `router_replay` on vLLM's built-in model, whose fused MoE layer records routed experts |

Released gpt-oss checkpoints store experts in MXFP4. This plugin loads bf16 checkpoints, so dequantize the released
weights first; a converter that dequantizes MXFP4 under `[checkpoint] dtype = "bfloat16"` (spec §7.9) would load them
directly.

```bash
PYTHONPATH=examples/model_plugins/gpt_oss \
  python -m arctic_platform.model_plugin.conformance \
  --plugin gpt_oss_plugin:PLUGIN --spec examples/model_plugins/gpt_oss/conformance.toml
```
