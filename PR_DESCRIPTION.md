[bug fix] Require arctic-inference 0.3.0+ for the [rl] extra

## Summary

`[rl]` asks for `arctic-inference[server,vllm]>=0.3.0` and does not declare `vllm`. That extra pulls vLLM and applies that release's pin. `tensordict` lives on `[sft]`. `[testing]` is pytest plugins only. Unit Tests Setup environment is `uv pip install ".[sft,testing]"` after CPU torch, so the CPU runner does not pull arctic-inference.

## Testing

`tests/test_dependency_groups.py`. GitHub Unit Tests Setup environment is unrun on this SHA.
