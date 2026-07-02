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

3. **Run** — `cd <recipe>/`, follow its README for data prep, then `bash run_*.sh`.

## Anatomy of a recipe

- **`simple/`, `txt2sql/`** are pure config: launchers, `requirements.txt`,
  `overrides.txt`, `download_data.py`, and a README. Launchers set
  `PYTHONPATH=$SKYRL_HOME` and dispatch to upstream's Ray entrypoint
  (`trainer.override_entrypoint=integrations.arctic_rl.entrypoint` for ARL,
  upstream's `integrations/arctic_rl/examples/fsdp_bird_entry.py` for the FSDP
  A/B baseline). Nothing to import beyond what's in the SkyRL clone.

- **`long_context_qa/`** is the exception: it adds a new `skyrl_gym` env
  (`long_context_qa`) that isn't registered upstream, so it ships a small
  recipe-local shim under [`long_context_qa/arctic_rl/`](long_context_qa/arctic_rl/),
  a [`sitecustomize.py`](long_context_qa/sitecustomize.py) for
  `ProcessPoolExecutor` spawn children, and a [`fsdp_loongrl_entry.py`](long_context_qa/fsdp_loongrl_entry.py)
  that mirrors upstream's `fsdp_bird_entry.py` for the FSDP A/B baseline.
  Everything else is reused verbatim from `$SKYRL_HOME`.

If you're writing a new recipe that reuses an env registered upstream
(`gsm8k`, `bird`, `bird_sql`, …), copy the shape of `simple/` or `txt2sql/`.
If you're adding a new env, copy the shape of `long_context_qa/`.
