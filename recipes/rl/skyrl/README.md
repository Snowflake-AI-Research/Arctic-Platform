# Arctic RL × SkyRL Recipes

End-to-end recipes for training models with [Arctic RL](../../../arctic_platform/rl/)
on top of [SkyRL](https://github.com/NovaSky-AI/SkyRL).

Available recipes:
  * [Simple (GSM8K)](simple) — smallest end-to-end loop, one GPU, no Ray/hostfile
  * [Txt2SQL (BIRD)](txt2sql) — single-node 8B **+** 4-node 32B ARL, with an FSDP A/B baseline (the blog ~2.38× speedup run)
  * [Long-context QA (LoongRL)](long_context_qa) — single-node 8B **+** 4-node 32B ARL, with an FSDP A/B baseline (locally measured 2.17× speedup)

These recipes drive SkyRL's GRPO trainer with the Arctic RL backend (ZoRRo train + Forest
Cascade Attention + Arctic-Inference speculative decoding).

## Install

Each recipe is a standalone folder with its own `requirements.txt`, `overrides.txt`,
`download_data.py`, launchers, and README. Same env across all three — build it
once, `conda activate skyrl_arl`, and any recipe launches from bare `python`.

1. **Clone SkyRL at the pinned commit** on the ``arctic-rl-public`` branch. The
   launchers dispatch from `$SKYRL_HOME/integrations/arctic_rl/`, which is not
   shipped in the pip-installed `skyrl` package — a checkout is required.

   ```bash
   git clone https://github.com/Snowflake-AI-Research/SkyRL
   cd SkyRL && git checkout 7636101a71f1849b6127ee10232fb277d2f31174 && cd ..
   export SKYRL_HOME=$PWD/SkyRL
   ```

   ``arctic-rl-public`` ships the verified BIRD Arctic-RL + FSDP recipes; later
   commits on ``main`` / ``novasky-main`` call
   ``nn.Module.named_non_persistent_buffers`` (not in any released PyTorch as
   of 2026-06) and break the FSDP path.

2. **Install pinned Python deps** into a fresh conda env:

   ```bash
   conda create -y -n skyrl_arl python=3.12.13
   conda activate skyrl_arl
   pip install -q uv
   uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
   uv pip install -r <recipe>/requirements.txt --override <recipe>/overrides.txt
   ```

3. **(Hopper only) Install FlashAttention-3.** Both 4-node 32B launchers
   default to `ATTN_IMPL=flash_attention_3` for the 2× speedup vs FSDP. Grab
   PyTorch's [official FA3 wheel](https://dev-discuss.pytorch.org/t/flash-attention-3-wheels/3322)
   matching your CUDA build:

   ```bash
   uv pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128
   ```

   (`cu126`, `cu130` indices also available.) Skip this on A100/L40S and
   launch with `ATTN_IMPL=flash_attention_2` — the 8B iteration launchers
   already default to FA2.

4. **Run** — `cd <recipe>/`, follow its README for data prep, then `bash run_*.sh`.

## Anatomy of a recipe

All three recipes share the same skeleton — each is a standalone folder that
depends only on `$SKYRL_HOME` for the actual Arctic RL × SkyRL library
(config/trainer/generator/envs) and vendors everything the recipe itself
needs:

- **`arctic_rl/`** shim — recipe-local package whose `entrypoint.py` is what
  the ARL launchers dispatch to (`trainer.override_entrypoint=arctic_rl.entrypoint`).
  Wraps upstream's `ArcticRLExp` but re-defines the `@ray.remote skyrl_entrypoint`
  task so workers import *this* package on deserialization, triggering env
  registration in the worker's address space.
- **`fsdp_<name>_entry.py`** — FSDP-native sibling of `arctic_rl/entrypoint.py`.
  What the FSDP A/B launcher dispatches to. Same three-interpreter registration
  dance through `BasePPOExp` instead of `ArcticRLExp`.
- **`sitecustomize.py`** — auto-import hook picked up by Python's `site.py`.
  Registers envs in the `ProcessPoolExecutor` spawn children that the
  reward-scorer spins up.

The `simple/` recipe skips `fsdp_<name>_entry.py` (single-GPU, no A/B baseline)
and `sitecustomize.py` (no PPE reward scorer), but the ARL shim shape is
identical. `long_context_qa/` additionally vendors its own env package under
`arctic_rl/envs/` because `long_context_qa` isn't registered upstream; the
other two rely on upstream's `integrations.arctic_rl.envs`.

The actual Arctic RL × SkyRL machinery (config/trainer/generator) is *not*
vendored — it lives in `$SKYRL_HOME/integrations/arctic_rl/`. Launchers put
`$SKYRL_HOME` and the recipe dir on `PYTHONPATH` so both are importable
side-by-side.
