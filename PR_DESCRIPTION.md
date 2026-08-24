## Summary

- SFT returns a single merged `metrics` dict after `fwd_bwd` + `step`, matching RL `update_actor` (step metrics as the base, fwd_bwd keys win).
- `ArcticSFTClient.train_step` and `merge_sft_step_metrics` live on the unified client (`arctic_platform/client/sft.py`). The worker already emitted `loss` / `grad_norm` (and any extra loss-fn keys); this wires them through as one dict instead of making every caller cherry-pick.
- Demos and `docs/sft.md` use `train_step`.
- Merges `origin/main` client unification (`ArcticSFTClient` subclasses `ArcticClient`; old `arctic_platform/sft/client.py` is gone).

## Testing

- CPU on `stas-dev-2-0` (`CUDA_VISIBLE_DEVICES=`, conda `dev`): `tests/sft/test_sft_client_ops.py` + `tests/sft/test_sft_config.py` + `tests/client/test_client_ops.py` — 84 passed after the merge (job `20260824T163728Z-1073948-000-2668422822`).
- No GPU / live-server run for this change (client surface only).
