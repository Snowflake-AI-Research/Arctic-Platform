# OLMoE: a mixture-of-experts plugin on vLLM's built-in model

[OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924) is a mixture-of-experts decoder the platform has no
built-in implementation for, and vLLM serves with its built-in `OlmoeForCausalLM`. It is the dense
[OLMo-2 example](../olmo2/README.md) plus what mixture of experts needs:

| File | What it adds to the dense path |
|---|---|
| `olmoe_plugin/modeling.py` | `OlmoeSparseMoeBlock`: routes through `route(tokens, routed=None)` with `choose_experts`, so the platform can replay routing (router replay, recompute); computes experts through `experts_core`, so the platform can shard them (expert parallelism); experts stacked on dim 0 |
| `olmoe_plugin/__init__.py` | `expert_layer=StackedExpertLayer(...)` naming the stacked expert tensors, and `moe_layers` in `ParallelismInfo` |
| `olmoe_plugin/weights.py` | Converters between stacked trainer experts and the checkpoint's per-expert tensors |
| `olmoe_plugin/manifest.toml` | `expert_parallel` and `router_replay`; `prefix_caching`, which router replay requires |
| `olmoe_plugin/reference.py` | Transformers' `OlmoeForCausalLM`, loaded through the converters, so it checks them too |

Run its conformance tiers on CPU from the repository root, including router replay and the SP, EP and SP+EP tiers:

```bash
PYTHONPATH=examples/model_plugins/olmoe \
  python -m arctic_platform.model_plugin.conformance \
  --plugin olmoe_plugin:PLUGIN --spec examples/model_plugins/olmoe/conformance.toml
```
