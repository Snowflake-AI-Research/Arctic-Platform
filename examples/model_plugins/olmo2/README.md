# OLMo-2: the minimal dense plugin

[OLMo-2](https://huggingface.co/allenai/OLMo-2-1124-7B) is a dense decoder the platform has no built-in
implementation for, and vLLM serves with its built-in `Olmo2ForCausalLM`. It is the shortest path through the
[model plugin contract](../../../docs/model-plugins/spec.md), and the template to copy for a dense model:

| File | What it holds |
|---|---|
| `olmo2_plugin/manifest.toml` | One layer type on the platform's attention core, and vLLM's built-in architecture |
| `olmo2_plugin/modeling.py` | The trainer model, named like the Hugging Face checkpoint, so no converters |
| `olmo2_plugin/reference.py` | The conformance oracle: Hugging Face's own `Olmo2ForCausalLM` with the same weights |
| `olmo2_plugin/__init__.py` | The `ModelPlugin`: config class, model, parallelism info, reference, admission checks |
| `conformance.toml` | The tiny fixture conformance runs |

Run its conformance tiers on CPU from the repository root:

```bash
PYTHONPATH=examples/model_plugins/olmo2 \
  python -m arctic_platform.model_plugin.conformance \
  --plugin olmo2_plugin:PLUGIN --spec examples/model_plugins/olmo2/conformance.toml
```

For new layer types, mixture of experts, checkpoint conversion or a custom vLLM model, see
[`toy_hybrid_moe`](../toy_hybrid_moe/README.md).
