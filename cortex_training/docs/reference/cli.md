# Cortex Training CLI and Client Reference

The `cortex-training` command and `cortex_training` Python package provide the
supported command-line and SDK interfaces for Cortex Training.

## Installation

Requires Python 3.8+. Installing the package gives you the `cortex-training` CLI,
the `cortex-training tui` log viewer, and the `cortex_training` Python SDK.

Install straight from the repository:

```bash
pip install "cortex-training @ git+https://github.com/Snowflake-AI-Research/Arctic-Platform.git#subdirectory=cortex_training"
```

Or from a local checkout:

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform.git
cd Arctic-Platform/cortex_training
pip install "."             # add -e for an editable/dev install
```

Verify the install:

```bash
cortex-training --help
cortex-training tui --help
```

## Cortex Training Jobs CLI

`cortex-training` submits and manages Cortex Training jobs through the SNOWAPI endpoint.
The normal workflow is:

1. Create a connection config JSON.
2. Run `cortex-training login --config config.json` once.
3. Use `cortex-training list`, `submit`, `get`, `cancel`, `wait`, and
   `capacity` without passing connection flags every time.

### Connection Config

For Snowflake PAT auth, use `host` for the account hostname. Do not use
`base_url` for Snowflake PAT auth.

```json
{
  "host": "dsa-test.qa6.us-west-2.aws.snowflakecomputing.com",
  "pat": "YOUR_PROGRAMMATIC_ACCESS_TOKEN",
  "database": "CORTEX_TRAINING_DB",
  "schema": "PUBLIC",
  "endpoint": "cortex-training",
  "poll_interval": 0.5,
  "poll_timeout": 1800.0,
  "verify_ssl": true
}
```

If you prefer not to store the PAT in the file, omit `pat` and set it in the
shell instead:

```bash
export CORTEX_TRAINING_PAT='YOUR_PROGRAMMATIC_ACCESS_TOKEN'
```

For a local/mock SNOWAPI server, use `base_url` with an explicit scheme:

```json
{
  "base_url": "http://localhost:8084",
  "database": "MY_DB",
  "schema": "PUBLIC",
  "endpoint": "cortex-training"
}
```

### Login

Login validates the config and stores only the config path, not the config
contents:

```bash
cortex-training login --config config.json
```

The login state is written to `~/.config/cortex-training/login.json` by default,
or `$XDG_CONFIG_HOME/cortex-training/login.json` when `XDG_CONFIG_HOME` is set.

You can bypass login for one command with:

```bash
cortex-training --config config.json list
```

or by setting:

```bash
export CORTEX_TRAINING_CONFIG=/path/to/config.json
```

Explicit CLI flags override config values.

### Commands

```bash
cortex-training list
cortex-training list --status running
cortex-training capacity
cortex-training get JOB_ID
cortex-training checkpoints JOB_ID
cortex-training cancel JOB_ID
cortex-training wait JOB_ID
cortex-training --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
cortex-training --job-id JOB_ID step --lr 1e-4
cortex-training --job-id JOB_ID load CHECKPOINT_ID
cortex-training --job-id JOB_ID generate examples/api/generate.json
cortex-training --job-id JOB_ID weight-sync
cortex-training download-log JOB_ID --output-dir /path/to/dir
```

Global flags must come before the subcommand:

```bash
cortex-training --compact list
cortex-training --config config.json submit examples/api/training.json
```

### Show Current GPU Capacity

Print the caller account's reserved GPU capacity and current usage:

```bash
cortex-training capacity
```

The response includes `has_reservation`, `reserved_gpus`, `in_use_gpus`, and
`available_gpus`.

### Submit A Job

The submit command expects a SNOWAPI CreateJob JSON body:

```json
{
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": "gpt2",
      "inference_config": {
        "max_seq_len": 128,
        "n_gpus": 1
      }
    }
  ]
}
```

Submit it:

```bash
cortex-training submit job.json
cortex-training submit job.json --wait
cortex-training submit job.json --dry-run
```

The repo includes a Prime-RL/Qwen3.6 training example:

```bash
cortex-training submit examples/api/training.json
cortex-training submit examples/api/sampling.json
```

That file creates a training sub-job for `Qwen/Qwen3.6-35B-A3B` with
`training_config.model_provider` set to `prime_rl`.

#### Debug options (internal only)

A CreateJob body may carry a `debug` block — e.g. an `image_tag` override that
pins a job's dynamically-provisioned zone to a specific `Cortex Training backend` build, so
you can test a build without a deployment-configuration change:

```json
{
  "sub_job_configs": [ ... ],
  "debug": { "job": { "image_tag": "release_20260622_<sha>" } }
}
```

This is an internal-only capability. The client refuses to send a request
carrying a `debug` block unless `CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS` is set to a
truthy value (`1`/`true`/`yes`/`on`); otherwise the submit fails fast client-side.
The directives are also gated server-side by the `CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS`
account parameter.

### Run A Forward-Backward Smoke Test

After the training job is running, send one tokenized training batch:

```bash
cortex-training --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
cortex-training --job-id JOB_ID step
```

The fwd-bwd JSON is human-readable: it contains text samples, tokenizer
settings, batch size, sequence length, `position_ids`, and label generation
settings. The CLI tokenizes the text, builds tensor kwargs, serializes
`{"args": (), "kwargs": ...}` with `torch.save`, submits
`forward_backward`, and polls the request by default. Set `"poll": false` in
the JSON to print only the submitted `request_id`.

Text payloads require `transformers` in the client environment. You can also
provide pre-tokenized tensor data directly under `payload.kwargs` for fully
offline use.

Run an optimizer step after fwd-bwd with:

```bash
cortex-training --job-id JOB_ID step
cortex-training --job-id JOB_ID step --lr 2e-5
```

When omitted, `--lr` defaults to `1e-4`.

### Load A Checkpoint Into A Running Job

After a job has already been created and reached running, load a checkpoint into
that existing job with:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID
```

To load from another job's checkpoint store:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID --source-job-id SOURCE_JOB_ID
```

To load into a specific training sub-job (useful for multi-sub-job sessions):

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID --target-sub-job-id JOB_ID:training:0
```

When `--target-sub-job-id` is omitted, the control plane routes the load to the
session's training sub-job. Sampling sub-jobs are not valid targets.

#### Discovering Sub-Job IDs

To find available training sub-jobs in a session:

```python
job = client.get_job(job_id)
for sub_job in job["sub_jobs"]:
    if sub_job["job_type"] == "training":
        sub_job_id = sub_job["sub_job_id"]
        n_gpus = sub_job["training_config"]["n_gpus"]
        print(f"Training sub-job: {sub_job_id} (DP={n_gpus})")
```

Or via CLI:

```bash
cortex-training --job-id JOB_ID get | jq '.sub_jobs[] | select(.job_type=="training") | {sub_job_id, n_gpus: .training_config.n_gpus}'
```

#### When to Use target-sub-job-id

Most sessions have a single training sub-job, so `--target-sub-job-id` can be
omitted. Use it when:

- Your session has multiple training sub-jobs (multi-DP configurations)
- You need to load different checkpoints into different sub-jobs
- You want explicit control over which sub-job receives the checkpoint

#### DP Size Compatibility

If loading a checkpoint into a sub-job with a **different DP size** (different
`n_gpus`) than the checkpoint was saved from, the job **must** have been created
with `load_optimizer_states=False`:

```python
training = SubJobConfig.training_job(
    model_name="...",
    n_gpus=16,  # Different from source checkpoint's DP size
    load_optimizer_states=False,  # REQUIRED for DP size change
    ...
)
```

This setting is configured at **job creation time** and cannot be changed later.
The optimizer states are DP-sharded and cannot be resized. If you forget this,
the load will fail at runtime.

This is the runtime load path. Create-time resume still uses
`source_checkpoint_info` in the submitted sub-job JSON.

### Start Sampling From A Training Checkpoint

Sampling requires a `weights-only` checkpoint. Save one from the training job,
then create a standalone sampling job that references its public checkpoint and
source job ids:

```python
request_id = client.save(training_job_id, checkpoint_type="weights-only")
checkpoint = client.poll_request(training_job_id, request_id)

sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    source_checkpoint_info={
        "checkpoint_id": checkpoint["checkpoint_id"],
        "source_job_id": training_job_id,
    },
)
sampling_job_id = client.create_job(sub_jobs=[sampling])
```

The sampling job is independent: the source training job can be stopped after
the checkpoint has been saved. Resumable DeepSpeed checkpoints contain
optimizer state and are not directly loadable by the sampling runtime.

### Run A Generate Smoke Test

After a sampling job is running, send readable prompts with sampling
parameters:

```bash
cortex-training --job-id JOB_ID generate examples/api/generate.json
```

The generate JSON contains `prompts`, optional `sampling_params`, and optional
`routing_key` / `strict` fields. `sampling_params` may be one object applied to
all prompts or a list of objects/nulls aligned with `prompts`. The CLI submits
`generate` and polls the request by default. Set `"poll": false` to print only
the submitted `request_id`.

### Sync Training Weights

For an RL-style job with one training and one sampling sub-job, sync training
weights into sampling with:

```bash
cortex-training --job-id JOB_ID weight-sync
```

By default this syncs from `JOB_ID:training:0` to `JOB_ID:sampling:0`, routes
the operation through `JOB_ID:training:0`, and polls for completion. Override
sub-job ids when needed:

```bash
cortex-training --job-id JOB_ID weight-sync \
  --source-sub-job-id JOB_ID:training:1 \
  --target-sub-job-id JOB_ID:sampling:0 \
  --target-sub-job-id JOB_ID:sampling:1
```

If a backend needs a different operation routing hint, pass
`--operation-sub-job-id` or `--operation-sub-job-type`.

### Download Execution Logs

Pull every log file the job's experiment run produced. Each sub-job's
`_logs/` directory in S3 may contain multiple files (e.g.
`execution.jsonl`, `server.log`); all of them are downloaded:

```bash
cortex-training download-log JOB_ID --output-dir /path/to/dir
```

Files are written as `<output_dir>/<sub_job_id>/<filename>` so siblings
do not collide. When `--output-dir` is omitted, the current working
directory is used instead. The CLI also prints a JSON summary listing
each `saved_path`.

Programmatic access is `CortexTrainingClient.fetch_execution_logs(job_id)`,
which returns a list of `{sub_job_id, filename, s3_uri, content}` dicts.

### Log TUI

`cortex-training tui` is a read-only terminal UI for tailing a running job's logs
live. It reuses the same connection handling as `cortex-training` — login state,
`--config` /
`CORTEX_TRAINING_CONFIG`, the `CORTEX_TRAINING_*` / `SNOWFLAKE_*` env vars, or explicit
flags. So once you've run `cortex-training login` you can just launch it:

```bash
cortex-training tui                 # opens a job picker
cortex-training tui JOB_ID          # opens that job's logs directly
```

Without login state, pass connection details the same way as the CLI:

```bash
cortex-training tui JOB_ID --config config.json
cortex-training tui JOB_ID --host ACCOUNT.snowflakecomputing.com --pat YOUR_PAT \
  --database CORTEX_TRAINING_DB --schema PUBLIC --endpoint cortex-training
cortex-training tui JOB_ID --base-url http://localhost:8084   # local/mock
```

For example, against the qa6 test account (PAT kept out of the command via an
env var; omit `JOB_ID` to open the job picker):

```bash
cortex-training tui \
  --host dsa-test.qa6.us-west-2.aws.snowflakecomputing.com \
  --pat "$CORTEX_TRAINING_QA6_PAT" \
  --database DSA_TEST_DB --schema PUBLIC
```

The left panel lists the job's sub-jobs; select one to tail its logs (the
zone-manager pod is the Ray head, so a sub-job's worker output is included).
Logs are cached locally so reopening a job replays instantly without
re-fetching from the server — under `~/.cache/cortex-training/` (or
`$XDG_CACHE_HOME`), overridable with `CORTEX_TRAINING_TUI_CACHE_DIR`.

Keys in the log view:

| Key | Action |
|-----|--------|
| `/` | Filter the current source |
| `L` | Cycle minimum log level (INFO / WARNING / ERROR) |
| `p` | Pause / resume auto-scroll |
| `s` | Save the current source to `~/cortex-training-<job>-<source>.log` |
| `y` | Copy the visible log to the clipboard |
| `r` | Refresh the sub-job list |
| `[` / `]` | Narrow / widen the sources panel |
| `b` / `esc` | Back |
| `q` | Quit |

In the job picker, `/` filters by id/status/type and `r` refreshes. The
`--poll-interval` flag (default `1.0s`) is the minimum interval between log
polls per source, biasing toward server reliability over freshness.

### Environment Variables

Connection values can also come from:

```bash
CORTEX_TRAINING_CONFIG
CORTEX_TRAINING_BASE_URL
CORTEX_TRAINING_HOST
SNOWFLAKE_HOST
CORTEX_TRAINING_PAT
SNOWFLAKE_PAT
CORTEX_TRAINING_DATABASE
SNOWFLAKE_DATABASE
CORTEX_TRAINING_SCHEMA
SNOWFLAKE_SCHEMA
CORTEX_TRAINING_ENDPOINT
```

`CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS` (truthy) unlocks sending CreateJob `debug`
options — see [Debug options (internal only)](#debug-options-internal-only).

### Troubleshooting

If you see `provide --base-url for local/mock use, or both --host and --pat`,
the CLI found a `host` but no PAT. Add `"pat": "..."` to `config.json` or set
`CORTEX_TRAINING_PAT`.

If you see `Invalid URL ... No scheme supplied`, the config is using a bare
Snowflake hostname as `base_url`. Use `host` for Snowflake PAT auth, or use a
full local/mock URL such as `http://localhost:8084` for `base_url`.

For server errors, the CLI prints any Snowflake request id and response body
returned by SNOWAPI. Include those details when debugging a `500`.
