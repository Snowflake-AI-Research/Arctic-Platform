# Job Management and Observability

After you create a job, use the `cortex-training` CLI or `CortexTrainingClient`
to inspect status, download logs, list checkpoints, and cancel or resume work.

This page is the in-repo copy of the Private Preview job-management guide
([docs.snowflake.com](https://docs.snowflake.com/en/LIMITEDACCESS/snowflake-cortex/cortex-training-job-management)).

---

## Quick start

Replace `JOB_ID` or `job_id` with the identifier returned when you created the
job. Before running CLI commands, authenticate once with
`cortex-training login --config config.json`. Python examples reuse the
`CortexTrainingClient` from the getting-started guide.

```bash
cortex-training capacity
cortex-training list --status running
cortex-training get JOB_ID
cortex-training wait JOB_ID
cortex-training download-log JOB_ID --output-dir ./logs
cortex-training cancel JOB_ID
```

For a live tail while the job is running:

```bash
cortex-training tui JOB_ID
```

### SDK equivalents

The rest of this guide shows the CLI. The same operations are available on
`CortexTrainingClient` after you create a client once:

```python
from cortex_training import CortexTrainingClient

client = CortexTrainingClient.from_pat(
    host="YOUR_ACCOUNT.snowflakecomputing.com",
    pat="YOUR_PAT",
    database="YOUR_DB",
    schema="YOUR_SCHEMA",
)
```

| Task | CLI | SDK |
| --- | --- | --- |
| Capacity | `cortex-training capacity` | `client.get_capacity()` |
| List jobs | `cortex-training list` / `list --status running` | `client.list_jobs()` / `client.list_jobs(status="running")` |
| Get job | `cortex-training get JOB_ID` | `client.get_job(job_id)` |
| Wait until running | `cortex-training wait JOB_ID` | `client.wait_for_job(job_id)` |
| List checkpoints | `cortex-training checkpoints JOB_ID` | `client.list_checkpoints(job_id)` |
| Download execution logs | `cortex-training download-log JOB_ID --output-dir DIR` | `client.fetch_execution_logs(job_id)` |
| Cancel | `cortex-training cancel JOB_ID` | `client.cancel_job(job_id)` |
| Load checkpoint | `cortex-training --job-id JOB_ID load CHECKPOINT_ID` | `client.load(job_id, checkpoint_id)` |
| Live tail | `cortex-training tui JOB_ID` | *(CLI / TUI only)* |

---

## GPU capacity

```bash
cortex-training capacity
```

Example response for an account with a GPU reservation:

```json
{
  "has_reservation": true,
  "reserved_gpus": 64,
  "in_use_gpus": 40,
  "available_gpus": 24
}
```

| Field | Meaning |
| --- | --- |
| `has_reservation` | Whether the account has a configured GPU reservation. When `false`, the account uses shared or on-demand placement and the `*_gpus` fields are `0`. |
| `reserved_gpus` | Total GPUs reserved for the account. |
| `in_use_gpus` | GPUs consumed by the account’s `RUNNING` and `PLACING` jobs. |
| `available_gpus` | `reserved_gpus - in_use_gpus`, floored at `0`. |

If a job stays in `PENDING`, check capacity together with the job details.

---

## List jobs

```bash
cortex-training list
cortex-training list --status running
cortex-training list --status failed
```

Each entry includes `job_id`, `status`, timestamps, and `sub_jobs`. The Python
method returns the following list; the CLI wraps the same list in a `jobs`
field. Example (identifiers redacted):

```json
[
  {
    "job_id": "11111111-2222-3333-4444-555555555555",
    "status": "FAILED",
    "reason": "sub_job_readiness",
    "created_at": "2026-05-28T20:46:03Z",
    "updated_at": "2026-05-28T20:47:28Z",
    "sub_jobs": [
      {
        "sub_job_id": "11111111-2222-3333-4444-555555555555:training:0",
        "status": "FAILED",
        "job_type": "TRAINING",
        "model_name": "Qwen/Qwen3-0.6B"
      },
      {
        "sub_job_id": "11111111-2222-3333-4444-555555555555:sampling:0",
        "status": "FAILED",
        "job_type": "SAMPLING",
        "model_name": "Qwen/Qwen3-0.6B"
      }
    ]
  }
]
```

---

## Job details

```bash
cortex-training get JOB_ID
```

The object has the same shape as one element from the job list.

| Field | Meaning |
| --- | --- |
| `status` | Lifecycle such as `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, or `CANCELLED`. |
| `reason` | Terminal or transition reason when set (for example, `cancelled_by_client`). |
| `sub_jobs` | Per worker group: `sub_job_id`, `job_type`, `model_name`, and configuration. |
| `created_at` / `updated_at` | UTC timestamps. |

Sub-job IDs use the pattern `job_id:type:index`. For example,
`JOB_ID:training:0` and `JOB_ID:sampling:0` on a colocated RL job.

---

## Wait for a job

```bash
cortex-training wait JOB_ID
```

The command or method blocks until the job reaches `RUNNING`. It raises an
error if the job reaches a terminal state first or the poll timeout expires.

---

## Checkpoints

```bash
cortex-training checkpoints JOB_ID
```

The CLI wraps checkpoints in a `checkpoints` field, as shown here. The Python
method returns the list inside that field.

```json
{
  "checkpoints": [
    {
      "checkpoint_id": "cp_33158699-c227-4d5a-b7ba-cf1da92bfa8c",
      "created_at": "2026-06-17T20:51:40.783112791Z"
    }
  ]
}
```

Short runs, cancellations, or failures before the first save often return an
empty `checkpoints` array.

---

## Logs and metrics

Every job gets an experiment run in Snowflake ML Experiments, where training
metrics such as loss and learning rate are logged.

### Download execution logs

```bash
cortex-training download-log JOB_ID --output-dir ./logs
```

The command writes one directory per sub-job under `--output-dir`.

Use `execution.jsonl` as the durable execution record. It contains structured
events such as operation names, durations, request and response sizes, and
errors. It is not a metrics export for reward, KL, GPU utilization, memory,
throughput, tokens per second, or MFU.

Common `job_metrics` line:

```json
{
  "event": "job_metrics",
  "op": "fwd-bwd",
  "duration_s": 0.478,
  "request_bytes": 5089,
  "response_bytes": 117,
  "error": null
}
```

Ops you may see include `create-job`, `fwd-bwd`, `step`, `save-checkpoint`,
`generate`, and `destroy-job`, depending on the workload.

Downloads can be empty if the job ended before the log uploader registered or
no operations ran. Confirm the job status, then retry after the job has reached
training or sampling work.

### Stream live logs

```bash
cortex-training tui JOB_ID
```

Run `cortex-training tui` without a job ID to open a job picker. Useful keys
include `/` to filter, `p` to pause, `s` to save visible logs, `r` to refresh,
and `q` to quit.

Live logs are available while the job is running. After the job ends, download
`execution.jsonl` for the durable record.

---

## Cancel, resume, and retry

### Cancel a job

```bash
cortex-training cancel JOB_ID
```

The job details typically then show `status: CANCELLED` and
`reason: cancelled_by_client`.

### Resume from a checkpoint

To resume at job creation time, set `source_checkpoint_info` in the sub-job
configuration. To load a checkpoint into an existing running training job:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID
```

To load from another job or select a target training sub-job:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID --source-job-id SOURCE_JOB_ID
cortex-training --job-id JOB_ID load CHECKPOINT_ID --target-sub-job-id JOB_ID:training:0
```

Changing data-parallel size (`n_gpus`) relative to the checkpoint requires
`load_optimizer_states=False` at job creation time. Optimizer state cannot be
reshaped at load time. Sampling sub-jobs are not valid load targets.

### Re-run after a failure

There is no one-click retry. Submit a new job with the same or adjusted
configuration. Use a saved checkpoint to continue instead of starting over.

---

## FAQ and notes

**Can I download raw console / pod stdout after I disconnect?**
Not yet. Use `download-log` for `execution.jsonl`, and any files your recipe
wrote under its local `log_path`. Live output is available via
`cortex-training tui` while the job is still running.

**Does `execution.jsonl` include training curves or GPU utilization?**
It records structured execution events (ops, durations, sizes, errors). It is
not a metrics-export file for reward/KL/loss series or GPU util / memory /
throughput / tokens/s / MFU. Training metrics such as loss and learning rate
are logged to Snowflake ML Experiments.

**Is there a one-click retry after infrastructure failure?**
No. Cancel if needed, then submit again (optionally resuming from a checkpoint
as above).

**Why is `download-log` empty?**
Common when the job ended before the log uploader registered, or no ops ran.
Confirm status with `get`, and try a job that reached training or sampling ops.
