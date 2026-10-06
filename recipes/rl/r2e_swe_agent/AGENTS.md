# Working in this repo

## Before you launch or cancel anything: check the run lock

Long RL runs live here. Two complete overfit runs were destroyed by cancels and
relaunches issued from other shells, each costing several hours of collection,
so this is the first thing to check:

```bash
python3 poc/run_lock.py show
```

If that prints a run, **a training run is live right now**. Leave it alone.

Why it is easy to break by accident: every `poc/launch_*.sh` force-deletes all
`r2e-` pods and reaps every sandbox before starting. That is indiscriminate, so
launching a second run kills the first one's in-flight rollouts even though you
never targeted it. A Cortex job id also looks anonymous — nothing about
`d5375d50-...` tells you that hours of collection depend on it.

The guards now refuse these by default:

- `poc/launch_conv4.sh` / `poc/launch_conv15.sh` abort if another run holds the
  lock. Override with `FORCE=1` only when you mean to replace that run.
- `poc/job_cancel.py` refuses the active run's job. Override with `--force`.

If a run really is dead, the lock expires on its own once `run.log` has been
untouched for 30 minutes; `python3 poc/run_lock.py release` clears it sooner.

## Cortex jobs are shared and cost real GPUs

Jobs live in a schema shared with other people, and every job reports
`submitted_by=ADMIN`, so the `comment` field is the only ownership signal.
Launch scripts set `CORTEX_JOB_COMMENT`; keep doing that.

- `poc/job_list.py` lists live jobs without creating one.
- `poc/job_status.py <id>` shows status, failure `reason`, and hardware.
- Never cancel a job you did not start. Check the comment first.

A job holds its GPUs until explicitly cancelled — killing the local driver only
stops the client, and the sub-jobs keep running and billing.

## Credentials

Cortex credentials are at `/data-fast/cortex.env` (pod-local, mode 600). They
are not on shared storage and must not be copied there. Do not read or use
credentials belonging to other users.
