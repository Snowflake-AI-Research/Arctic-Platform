# Jobs and Sub-Jobs

A job is the top-level lifecycle resource. It contains one or more sub-jobs:

- `training` handles forward/backward requests, optimizer steps, and training
  checkpoints.
- `sampling` handles generation.
- `log_probability` configures log-probability workers.

Internal sub-job identifiers use the form:

```text
{job_id}:{sub_job_type}:{index}
```

For example, a colocated RL job commonly has `{job_id}:training:0` and
`{job_id}:sampling:0`.

For complete request and routing details, see the
[REST API reference](../reference/rest-api.md).
