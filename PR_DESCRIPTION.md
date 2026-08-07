## Summary

- Add a first-class **SFT** surface (`arctic_platform.sft`) alongside RL: `ArcticSFTClient` / `ArcticSFTClientConfig`, server-side SFT processor (`sft` / `sft_ce`), demos, docs, and CPU + GPU tests. Intended consumer: Axolotl’s remote-SFT plugin (and any other CPU client).
- Lift shared DeepSpeed / HTTP / Ray server code out of `rl/` into `arctic_platform.common` so SFT and RL share one training+sampling stack. RL modules become thin shims / re-exports where needed.
- SFT client is HTTP-first and CPU-safe (`CUDA_VISIBLE_DEVICES=` empty on the client). Training ops always available; sampling ops (`generate`, `sync_weights`, sleep/wake) require `sampling_gpus > 0` and reuse the RL sampling job topology (including colocate staging).
- Checkpoint / resume / HF export:
  - step-tagged remote dirs `checkpoint-{step}/` + `latest`
  - optional `export_hf` → `{ckpt}/hf/`
  - `save_total_limit` via shared `prune_checkpoint_dirs`
  - `load_checkpoint(step=…)` restore for resume / best-model
- Token-exact `avg_loss` via `finalize_fwd_bwd_metrics` on both `fwd_bwd` and `fwd_no_grad` (paired global token mean, not a broken shard merge).
- Shared weight-sync helpers extracted to `common/utils/weight_sync.py` (used by HTTP and Ray servers).
- Base deps include `ray` / `uvicorn` (unconditionally imported by the servers); `[rl]` still brings vLLM for sampling.
- Docs: `docs/sft.md`, `docs/common.md`, `docs/rl.md`, `docs/index.md`.

## Testing

**CPU (no GPUs).** FakeTransport client suites pin the op surface after the SnowAPI rename: `tests/client/test_client_ops.py` (unified sync/async RL client — forward-backward / save / operation envelope / sleep-wake) and `tests/sft/test_sft_client_ops.py` + `test_sft_config.py` (SFT client defaults, save/load body, staged weight-sync). Shared helpers are covered without Ray: `tests/common/test_finalize_fwd_bwd_metrics.py` (token-exact `avg_loss`), `test_prune_checkpoint_dirs.py`, `test_tiled_logits.py`. Loss math and edge cases: `tests/sft/test_sft_losses.py`, `test_sft_edge_cases.py`.

**GPU e2e / demos.** Colocated HTTP smokes under `arctic_platform.sft.examples`: `run_sft_http_demo` (train steps), `run_sft_ckpt_resume_demo` (save → eval → train → load → eval + HF export), `run_sft_generate_demo` (train → sleep/sync → generate → wake; needs `[rl]` + ArcticInference). Pytest mirrors: `tests/sft/test_sft_ckpt_resume_gpu.py`, `test_sft_ce_gpu.py`.

**Regression against the `common/` move.** Existing RL client/HTTP paths stay importable via `rl/` shims; RL e2e (`tests/rl/`) and the unified-client ops suite above are the guard that SnowAPI wire + SFT extras didn’t break the RL surface.
