# Long-Context QA — multi-hop QA with Arctic RL (planned)

> **Coming in a follow-up PR.** Mirrors
> [`recipes/rl/verl/long_context_qa/`](../../verl/long_context_qa/), but drives SkyRL's PPO
> trainer instead of verl's.

Planned scope: Qwen3-32B trained on [LoongRL-Train-Data][loongrl] (16K-context multi-hop QA
merging HotpotQA, MuSiQue, 2WikiMultiHopQA), with inference extended to 128K via YaRN.
Pure GRPO, no frozen reference model, ZoRRo train + Forest Cascade Attention enabled.

Reference numbers from the [Arctic RL launch blog][blog]: average LongBench v1 QA accuracy
69.6% → 72.3%, with the largest gains on the hardest benchmarks (+7.5 on MuSiQue,
+4.5 on HotpotQA, +3.5 on 2WikiMQA).

When this lands it will reuse the same vendored
[`arctic_rl/`](../_lib/arctic_rl) integration as the other recipes in this directory.

[loongrl]: https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data
[blog]: https://www.snowflake.com/en/blog/engineering/arctic-rl-open-source-backend/
