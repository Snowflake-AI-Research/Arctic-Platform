# Manage Jobs

Common lifecycle commands:

```bash
dss-neutrino list
dss-neutrino list --status running
dss-neutrino get JOB_ID
dss-neutrino checkpoints JOB_ID
dss-neutrino wait JOB_ID
dss-neutrino cancel JOB_ID
```

Use `dss-neutrino capacity` before starting a recipe. Resume and retry guidance
is tracked separately because support depends on checkpoint type and failure
state.
