# Cancel, Resume, and Retry

Cancel a running job with:

```bash
dss-neutrino cancel JOB_ID
```

Runtime checkpoint loading and create-time checkpoint initialization are
documented in the [CLI reference](../../reference/cli.md). A complete recovery
matrix covering retryable failures, optimizer-state compatibility, and
automatic resume remains planned.
