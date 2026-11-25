# ToyHybridMoE: an example model plugin

A tiny architecture that is new to the platform and touches every part of the
[model plugin contract](../../../docs/model-plugins/spec.md):

- attention layers on the platform's attention core;
- a gated linear recurrence as a plugin layer type on the default, head-parallel sequence-parallel policy;
- a gated short convolution as a plugin layer type with its own policy, a halo exchange with the previous rank;
- dense and mixture-of-experts MLPs, with expert parallelism and router replay; the MoE code never moves tokens;
- trainer-only state left out of the weight sync;
- Hugging Face checkpoint conversion that renames and reshapes;
- a vLLM model for sampling.

Run its CPU conformance tiers from the repository root:

```bash
PYTHONPATH=examples/model_plugins/toy_hybrid_moe \
  python -m arctic_platform.model_plugin.conformance \
  --plugin toy_hybrid_moe:PLUGIN --spec examples/model_plugins/toy_hybrid_moe/conformance.toml
```

`vllm_model.py` needs a GPU and the vLLM version of the runtime release, so it is not part of the CPU tiers.
`tests/model_plugin/test_toy_plugin_vllm.py` checks it against the trainer on GPUs: prompt and decode log-probs,
weight sync and routed experts, at tensor parallelism 1 and 2.
