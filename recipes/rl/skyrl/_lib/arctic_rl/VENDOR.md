# `arctic_rl/` — vendored SkyRL integration

A verbatim copy of SkyRL's `integrations/arctic_rl/` tree, vendored so the recipes under
`recipes/rl/skyrl/` are self-contained — users do not need to clone SkyRL.

## Upstream source

- Repo: <https://github.com/NovaSky-AI/SkyRL>
- Path: `integrations/arctic_rl/`
- Pinned commit: `76f5f467c6804e8acc6273cc677098b7679b0315`
  ([PR #1837](https://github.com/NovaSky-AI/SkyRL/pull/1837) merge — Arctic RL training
  backend integration)

The matching SkyRL version is pinned in each recipe's `requirements.txt`
(`skyrl @ git+...@76f5f467...`). Re-syncs MUST bump both the SHA above and the SHA in
every `requirements.txt`.

## Local modifications

Vendor source is taken verbatim except for:

1. **`entrypoint.py`** — `_pkg_parent` uses two `dirname()` calls instead of three, so the
   PYTHONPATH that gets forwarded to Ray workers points at `_lib/` (the parent of the
   `arctic_rl/` package) rather than `_lib/arctic_rl/`. The intent is unchanged: workers
   need to `import arctic_rl.*` when deserializing the entrypoint task.
2. **`entrypoint.py`** and **`__init__.py`** — docstrings refer to the vendored module
   path (`arctic_rl.entrypoint`) instead of SkyRL's (`integrations.arctic_rl.entrypoint`).
3. **`envs/bird.py`** — docstring points at the sibling `bird_reward.py`.

`envs/__init__.py` uses `__name__` for the `entry_point`, so env registration with
`skyrl_gym` works under any import path; no change needed.

## Re-syncing from upstream

```bash
# At the new pinned commit in a SkyRL checkout:
SKYRL_HOME=/path/to/SkyRL
AP_VENDORED=/path/to/Arctic-Platform/recipes/rl/skyrl/_lib/arctic_rl

cp -r "${SKYRL_HOME}/integrations/arctic_rl/." "${AP_VENDORED}/"

# Re-apply the three local modifications listed above, then bump the SHA in
# this file AND in every recipes/rl/skyrl/*/requirements.txt.
```
