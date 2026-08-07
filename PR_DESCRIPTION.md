## Summary

Intermediary plumbing PR after `#62` (rl→common relocate). No SFT package yet — that lands in a follow-up.

- Extract tiled logits / CE kernels from `qwen_model_patcher` into `common.utils.tiled_logits` (`none` / `compute` / `memory`); RL imports them back.
- Shared `common.registry` for loss / post-processor registration (used by RL pipeline; SFT will register later).
- Shared `common.utils.weight_sync` helpers; HTTP/Ray servers use them instead of inlined ArcticInference imports where possible.
- `finalize_fwd_bwd_metrics` for token-exact `avg_loss` on fwd-bwd responses (HTTP + Ray).
- `prune_checkpoint_dirs` util (+ unit test) — **not** wired to save/load API yet (checkpoint extras in SFT PR).
- Client: `server_cuda_visible_devices`, sleep/wake inference+training, staged `sync_weights` wake flow.
- HTTP log-prob workers honor `MASTER_PORT` (was hardcoded `29501`).
- Base deps: `ray` / `uvicorn` (servers import them unconditionally); `[rl]` keeps vLLM/sampling stack only.
- Docs: `docs/common.md`, `docs/rl.md`, `docs/index.md` (SFT marked forthcoming). `tests/AGENTS.md`.
- FA2 once-patcher test skips when real `flash_attn` is unavailable.

## Out of scope (next PR)

- `arctic_platform.sft` package, demos, `tests/sft/`, `docs/sft.md`
- Worker `run_sft_pipeline` / `sft_ce` dispatch
- `sft_profile`
- Checkpoint extras: `load-checkpoint`, `export_hf`, `save_total_limit`, step-tagged save paths

## Test plan

- [ ] `tests/common/test_tiled_logits.py`
- [ ] `tests/common/test_finalize_fwd_bwd_metrics.py`
- [ ] `tests/common/test_prune_checkpoint_dirs.py`
- [ ] `tests/client/test_client_ops.py` (sleep/wake/sync)
- [ ] `tests/zorro_train/test_once_patcher.py`
- [ ] Spot-check RL e2e still imports / runs against common servers
