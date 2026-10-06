# R2E-Gym SWE-agent GRPO on Cortex Training

A multi-turn SWE agent trained with GRPO, with Cortex Training as the backend.
The agent runs inside a k3s sandbox and reaches the model only through an
OpenAI-compatible gateway, which records exact prompt and completion token ids
per turn so a trajectory can be packed into one training sequence.

**Start here:** [`SAMPLING_OVERHEAD_DEBUG.md`](SAMPLING_OVERHEAD_DEBUG.md) — how
to reproduce and measure sampler-side overhead, including why a measurement
taken during collection is mostly queue wait.

**Read before launching anything:** [`AGENTS.md`](AGENTS.md). Every
`poc/launch_*.sh` force-deletes all `r2e-` pods and reaps every sandbox on
startup, so launching a second run kills the first one's in-flight rollouts
even though you never targeted it. Check `python3 poc/run_lock.py show` first.

## Layout

| path | what it is |
| --- | --- |
| `poc/r2e_driver.py` | the loop: collect, pack, train, sync |
| `poc/launch_band8.sh` | the 8-prompt overfit configuration |
| `poc/launch_until_placed.sh` | retries an 8-GPU request through placement failures |
| `poc/probe_*.py`, `poc/probe_*_idle.sh` | sampler latency, scaling, prefix cache |
| `poc/gateway_check.py` | proves gateway parameter plumbing with no GPU |
| `poc/job_list.py`, `job_status.py`, `job_cancel.py` | find and release Cortex jobs |
| `poc/run_lock.py` | the guard that stops two runs colliding |
| `rexec.sh` | relays commands to the GPU pod over shared storage |

## Environment

Paths are absolute and pod-specific by design, because the venv and the
credentials are pod-local while the code is on shared storage:

- `/data-fast/ap-venv/bin/python` — the interpreter
- `/data-fast/cortex.env` — Cortex credentials, mode 600, **per person**; do not
  copy anyone else's and do not place yours on shared storage
- `/data-fast/k3s/` — the sandbox cluster's kubeconfig and binaries
- `$R2E_DATASET` — the R2E-Gym task jsonl
- `$CORTEX_JOB_COMMENT` — set it; every job in the shared schema reports
  `submitted_by=ADMIN`, so the comment is the only record of who owns a job

Expect to adjust those for your own pod.
