# Arctic RL × SkyRL Recipes

End-to-end recipes for training models with [Arctic RL](../../../arctic_platform/rl/) on
top of [SkyRL](https://github.com/NovaSky-AI/SkyRL).

Available recipes:
  * [Simple single-GPU (GSM8K)](simple) — smallest end-to-end loop, one GPU, no Ray/hostfile
  * [Txt2SQL](txt2sql)
  * [Long-context QA](long_context_qa)

These recipes drive SkyRL's GRPO trainer with the Arctic RL backend (ZoRRo train + Forest
Cascade Attention + Arctic-Inference speculative decoding). They're **self-contained** —
no SkyRL repo checkout is required. SkyRL itself is pulled from git via each recipe's
`requirements.txt`, and the SkyRL-side `arctic_rl/` integration code is vendored at
[`_lib/arctic_rl/`](_lib/arctic_rl) (added to `PYTHONPATH` by the launchers).

Upstream source for the vendored integration: [`integrations/arctic_rl/`][skyrl-arctic-rl]
in SkyRL, pinned at [PR #1837][skyrl-pr]. See
[`_lib/arctic_rl/VENDOR.md`](_lib/arctic_rl/VENDOR.md) for the exact SHA and re-sync
instructions.

[skyrl-arctic-rl]: https://github.com/NovaSky-AI/SkyRL/tree/main/integrations/arctic_rl
[skyrl-pr]: https://github.com/NovaSky-AI/SkyRL/pull/1837
