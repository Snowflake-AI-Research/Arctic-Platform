# Manage Jobs

Common lifecycle commands:

```bash
cortex-training list
cortex-training list --status running
cortex-training get JOB_ID
cortex-training checkpoints JOB_ID
cortex-training wait JOB_ID
cortex-training cancel JOB_ID
```

Use `cortex-training capacity` before starting a recipe. Resume and retry guidance
is tracked separately because support depends on checkpoint type and failure
state.
