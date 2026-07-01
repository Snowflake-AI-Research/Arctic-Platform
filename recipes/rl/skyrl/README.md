# Arctic RL × SkyRL Recipes

End-to-end recipes for training models with [Arctic RL](../../../arctic_platform/rl/)
on top of [SkyRL](https://github.com/NovaSky-AI/SkyRL).

Available recipes:
  * [Simple single-GPU (GSM8K)](simple) — smallest end-to-end loop, one GPU, no Ray/hostfile
  * [Txt2SQL (BIRD)](txt2sql) — single-node 8-GPU Qwen3-8B **+** 4-node 32-GPU Qwen3-32B (the blog 2× speedup run)
  * [Long-context QA (LoongRL)](long_context_qa) — single-node 8-GPU, Qwen3-8B, 16K prompts

These recipes drive SkyRL's GRPO trainer with the Arctic RL backend (ZoRRo train + Forest
Cascade Attention + Arctic-Inference speculative decoding).

## Install model

Each recipe is a standalone folder with its own `requirements.txt`, `overrides.txt`,
`download_data.py`, launcher, and README. To run one:

1. **Install pinned Python deps** for the recipe (pulls SkyRL from git as a Python
   package — but *not* its `integrations/arctic_rl/` directory, which is what step 2
   is for):

   ```bash
   uv pip install -r <recipe>/requirements.txt --override <recipe>/overrides.txt
   ```

2. **Clone SkyRL at the pinned commit** (this gives you the Arctic RL × SkyRL
   integration code at `integrations/arctic_rl/`, which lives outside the
   pip-installed `skyrl` package and is required by the launchers):

   ```bash
   git clone https://github.com/NovaSky-AI/SkyRL
   cd SkyRL && git checkout 76f5f467c6804e8acc6273cc677098b7679b0315 && cd ..
   export SKYRL_HOME=$PWD/SkyRL
   ```

   The pin is the merge commit for SkyRL [PR #1837][skyrl-pr] (the dispatch hook
   the Arctic RL integration requires). Mirrors what the per-recipe
   `requirements.txt` pulls as a Python package, so the two stay in sync.

3. **Run the recipe** — `cd <recipe>/`, follow its README for data prep, then
   `bash run_*.sh`.

## Anatomy of a recipe

- **`simple/`, `txt2sql/`** are pure config: a launcher, `requirements.txt`,
  `overrides.txt`, `download_data.py`, and a README. Their launchers set
  `PYTHONPATH=$SKYRL_HOME` and dispatch to upstream's Ray entrypoint
  (`trainer.override_entrypoint=integrations.arctic_rl.entrypoint`) directly.
  Nothing to import beyond what's in the SkyRL clone.

- **`long_context_qa/`** is the exception: it adds a new `skyrl_gym` env
  (`long_context_qa`) that isn't registered upstream. To get that registration
  to fire on the driver, Ray workers, *and* the `ProcessPoolExecutor` spawn
  children the reward scorer uses, it ships a small recipe-local shim under
  [`long_context_qa/arctic_rl/`](long_context_qa/arctic_rl/) plus a
  [`sitecustomize.py`](long_context_qa/sitecustomize.py). The shim re-defines
  the Ray `skyrl_entrypoint` so workers re-import the recipe on deserialization;
  `sitecustomize.py` handles the spawn children. Everything else — config,
  trainer, generator — is reused verbatim from `$SKYRL_HOME`.

If you're writing a new recipe that only *reuses* an env registered upstream
(`gsm8k`, `bird`, `bird_sql`, …), copy the shape of `simple/` or `txt2sql/`.
If you're adding a new env, copy the shape of `long_context_qa/`.

[skyrl-pr]: https://github.com/NovaSky-AI/SkyRL/pull/1837
