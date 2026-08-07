## Summary

- Relocate the protocol-agnostic training infrastructure out of `arctic_platform.rl` into a new `arctic_platform.common` package so a future SFT surface can share one DeepSpeed / HTTP / Ray stack with RL.
- Modules moved (`rl/` → `common/`): `deepspeed_worker`, `http_server`, `ray_server`, `ray_cluster`, `server`, and `utils/{batch,cuda_ipc,debug,ray_pg,record_replay,server_models}` (plus `utils/__init__.py`).
- Leave thin back-compat shims at the old `arctic_platform.rl.*` import paths (`from arctic_platform.common… import *`), so existing RL importers and `python -m arctic_platform.rl.http_server` keep working unchanged.
- RL-specific modules stay in `rl/` and are still imported from there: `processors`, `zorro_train`, `client`, `config`, `weight_sync`.
- Behavior-preserving move only — no new features, no SFT package, no docs rewrite, no `pyproject.toml` change. Each `common/` module is byte-identical to its former `rl/` counterpart except for rewriting imports of the relocated siblings (`arctic_platform.rl.{deepspeed_worker,http_server,ray_server,ray_cluster,server,utils}` → `arctic_platform.common.*`).

## Testing

**Import / structural.** `py_compile` over the moved modules + shims. Structural equality check: every `common/X` matches `origin/main`'s `rl/X` after only the sibling-import rewrite. Residual `arctic_platform.rl.*` refs inside `common/` are limited to non-moved siblings (`processors`, `zorro_train`).

**RL regression.** Existing RL client / HTTP / Ray paths remain importable via the `rl/` shims. Run the existing RL suites (`tests/rl/`, `tests/client/`) — they exercise the same code under the new locations without needing new SFT tests in this PR.
