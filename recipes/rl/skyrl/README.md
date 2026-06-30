# Arctic RL × SkyRL Recipes

End-to-end recipes for training models with [Arctic RL](../../../arctic_platform/rl/)
on top of [SkyRL](https://github.com/NovaSky-AI/SkyRL).

Available recipes:
  * [Simple single-GPU (GSM8K)](simple) — smallest end-to-end loop, one GPU, no Ray/hostfile
  * [Txt2SQL (BIRD)](txt2sql) — single-node 8-GPU, Qwen3-8B
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

## What lives in `_lib/arctic_rl/`

The launchers add this directory to `PYTHONPATH` so a small recipe-side shim is
importable alongside `$SKYRL_HOME/integrations/arctic_rl/`. The shim is tiny — its
only job is to register the recipe-private `long_context_qa` env with `skyrl_gym`
in both the SkyRL driver and Ray workers (workers re-import the shim when they
deserialize the Arctic RL `@ray.remote` task). Everything else — config, trainer,
generator, BIRD env, BIRD preprocessor — is used directly from `$SKYRL_HOME`. See
[`_lib/arctic_rl/__init__.py`](_lib/arctic_rl/__init__.py) for the design notes.

[skyrl-pr]: https://github.com/NovaSky-AI/SkyRL/pull/1837
