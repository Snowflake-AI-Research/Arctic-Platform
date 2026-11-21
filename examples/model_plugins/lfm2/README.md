# LFM2: a new layer type on vLLM's built-in model

[LFM2](https://huggingface.co/LiquidAI/LFM2-1.2B) is a dense hybrid: gated short-convolution layers and attention
layers, chosen per layer by `config.layer_types`. The platform has no implementation of it; vLLM serves it with its
built-in `Lfm2ForCausalLM`. Against the dense [OLMo-2 example](../olmo2/README.md) it adds:

| File | What it adds |
|---|---|
| `lfm2_plugin/modeling.py` | `Lfm2ShortConv`, a layer type the platform has never seen: its core `mix` is a depthwise causal convolution that restarts at every packed row and computes only the channels its `HeadShard` holds |
| `lfm2_plugin/__init__.py` | `mixers={"conv": HeadParallelMixer("conv", Lfm2ShortConv)}`: the platform's default sequence-parallel policy, with each channel as a "head"; the channel count goes in `ParallelismInfo.heads` |
| `lfm2_plugin/manifest.toml` | `conv = "plugin"` next to `full_attention = "platform:attention"`; prefix caching stays off for the stateful convolution layers |
| `conformance.toml` | Tied input and output embeddings, as LFM2's checkpoints have: stored once, as `model.embed_tokens.weight` |

The head-parallel policy moves every activation twice under sequence parallelism. For a short convolution, a halo
exchange with the previous rank moves only `kernel - 1` tokens; the [toy example](../toy_hybrid_moe/README.md)'s
`ShortConvMixer` implements it.

```bash
PYTHONPATH=examples/model_plugins/lfm2 \
  python -m arctic_platform.model_plugin.conformance \
  --plugin lfm2_plugin:PLUGIN --spec examples/model_plugins/lfm2/conformance.toml
```
