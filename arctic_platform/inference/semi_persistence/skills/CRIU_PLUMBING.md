# CRIU Plumbing: What Gets Cleaned Up and Why

## Overview

CRIU (Checkpoint/Restore In Userspace) dumps a process tree to disk and
restores it later, potentially on a different GPU.  A vLLM child process
is not a simple program — it has CUDA contexts, PyTorch distributed
backends, io_uring rings, POSIX semaphores, and hundreds of file
descriptors.  CRIU is strict: every FD, memory mapping, and file
reference in the image must be restorable at restore time, or it fails.

The dump is **destructive**: CRIU kills the child process after writing
the image to disk.  Every subsequent use of the model goes through
`load()` which creates a fresh process from the on-disk image.  This
avoids dangling processes from non-destructive (`-R`) dumps and
simplifies the state machine — after `save()`, the model is always
in `"saved"` state with no live process.

This document describes the complications we hit and how each is handled.

---

## Installation

CRIU install instructions (PPA path on Ubuntu 24.04, plus a
build-from-source fallback) have moved to [`INSTALL.md`](./INSTALL.md).
The plugin directory caveat referenced from *Complication 7* below is
covered there.

---

## The Pipeline

```
[Dump side — prepare_criu_dump in vllm_child.py + _worker_criu_save in worker.py]

  (in child)
  1. Drain in-flight engine requests
  2. Destroy PyTorch process group   (NCCL, TCPStore threads)
  3. Wait for store threads to exit  (poll /proc/<pid>/task, up to 2.5s)
  4. Leave stdout/stderr (fd 1, fd 2) alone: they are the pod-log pipe
     (PID 1's stdout), dumped as an external pipe -- see Complication 4
  5. Walk /proc/<pid>/fd and close every FD that does not match the
     keep-list  (see "FD Policy" below); pipe_fd and fd 0/1/2 are skipped
  6. Munmap every "io_uring" region from /proc/<pid>/maps
  7. Remove the /dev/shm/sem.* this tree maps, matched by inode, and each
     rank its own psm_* -- never another replica's (Complication 3)
  8. Audit /proc/<pid>/task for non-"python" threads (informational)

  (back in worker)
  9. Scan child AND descendant fds for socket:[ino] → --external unix[ino]
     (TP>1 worker subprocesses hold the multiproc-executor IPC sockets)
 10. Scan child fds for /dev/nvidia*   → record into meta.json
 11. → criu dump (destructive): image written, child killed

[Every use after dump — load from image]

  1. Pass pipe FD via Unix socket (SCM_RIGHTS) into sudo'd helper
  2. helper dup2's the pipe fd into place, then unshares a fresh PID
     namespace and forks: PID 1 is a reaper that unshares a mount ns with
     a private /proc and execs criu restore *inside* the new namespace
     (see Complication 10 — this frees the image's recorded PIDs)
  3. criu restore --inherit-fd fd[N]:pipe_resource
                  --inherit-fd fd[L]:stdout_resource   (this pod's log pipe)
                  --link-remap --tcp-close        (no --shell-job)
  4. worker finds the restored root's *host* PID via the inherited pipe
  5. cuda-checkpoint restore / driver API restores GPU context
```

---

## FD Policy at Dump Time

The child does not blanket-null its FDs.  `prepare_criu_dump`
walks `/proc/<pid>/fd/` and applies a **keep-list** based on the symlink
target; everything else is `os.close()`'d:

| FD class                | Action                                                     |
|-------------------------|------------------------------------------------------------|
| `fd 0` (stdin)          | left untouched (skipped: `<= 2`)                           |
| `fd 1`, `fd 2`          | skipped: the pod-log pipe, recorded as `stdout_resource`   |
| pipe fd to worker       | preserved (skipped explicitly via `pipe_fd` argument)      |
| `/dev/nvidia*`          | preserved — CUDA driver fds; restored by CRIU CUDA plugin  |
| `/dev/shm/*`            | preserved — POSIX shm, including pinned host buffers       |
| `anon_inode:*`          | preserved — eventfd, epoll, **and io_uring** (rings munmap'd separately) |
| `socket:[…]`            | preserved — worker tags unix sockets `--external unix[ino]`; TCP gets `--tcp-close` plus `SO_LINGER(1,0)` (Complication 12) |
| `socket:[…]` **listening on loopback** | **closed at TP=1 only**, before the keep-list runs — there they are orphaned torch rendezvous listeners whose recorded ephemeral ports criu would otherwise rebind for nothing. At TP>1 they are NCCL RAS's and are **kept** (Complication 15) |
| `pipe:[…]`              | preserved                                                  |
| Everything else         | closed (regular files, Triton `.so` opens, log files, etc.) |

The keep-list lives in `prepare_criu_dump`:

```python
keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:",
                 "socket:", "pipe:")
```

Note that `"socket:"` in that list is why the loopback-listener close
(Complication 15) needs its own earlier pass: the keep-list is prefix-based
and cannot distinguish a live unix socket from a dead TCP listener, so the
narrower rule has to run first and close by socket *state* rather than by
symlink target.

### What survives into the CRIU image

- The Python interpreter, the main thread, and any threads that aren't
  the NCCL/TCPStore set torn down above.
- All Python objects: the `LLM` engine, tokenizer, scheduler, KV cache
  manager, etc.
- Model weights and KV cache on the GPU (handled by the CUDA plugin at
  dump and by `cuCheckpointProcess*` / `cuda-checkpoint` at restore).
- `/dev/nvidia*` FDs (also recorded as `nvidia_fds` in `meta.json`).
- `/dev/shm` FDs and their mappings (anon pages now that `sem.*` files
  are deleted; pinned host buffers remain mapped).
- `anon_inode:` FDs (eventfd, epoll; io_uring FDs too — see Complication 2).
- `socket:` FDs marked `--external unix[<ino>]`, plus the worker pipe FD
  re-inherited via `--inherit-fd`.
- `pipe:` FDs.
- Triton JIT `.so` mappings, including ones already `(deleted)` on disk
  (see Complication 9 — these are intentionally *not* pre-handled and
  rely on `--link-remap` + the destructive dump to keep ghost-remap
  collisions rare).

---

## Complication 1: PyTorch Distributed (NCCL + TCPStore)

**Problem:** `dist.init_process_group()` spawns background threads
(`pt_tcpstore`, `pt_nccl_watchdg`, `pt_nccl_heartbt`) and opens TCP
sockets.  CRIU cannot restore TCP connections or threads that are
blocked on network I/O.

**Fix (dump side):** Call `dist.destroy_process_group()` before dump.
Then poll `/proc/<pid>/task/` until the store threads exit (up to 2.5s).
The TCPStore thread sometimes lingers — the dump proceeds with a
warning, and CRIU handles the remaining thread via `--tcp-close`.

`--tcp-close` covers the *connections*, not the ports they occupied:
CRIU rebinds each recorded local port before closing it, so the sockets
that survive into the image also need `SO_LINGER(1,0)` at dump time.  See
Complication 12.

**Retiring the threads does not retire their listening sockets.** The store
threads exit; their fds stay open, and the FD keep-list preserves them. Those
orphans are what Complication 15 closes, and they are the reason a TP1 image
carried seven loopback listeners on ephemeral ports. Anything reading this
section to reason about what torch leaves behind should read Complication 15
as well — the teardown here is necessary but was not sufficient.

---

## Complication 2: io_uring

**Problem:** Modern PyTorch/libtorch uses `io_uring` for async I/O.
CRIU cannot checkpoint `io_uring` instances — the kernel ring buffers
and submission queues are not serializable.  Both the FDs
(`anon_inode:[io_uring]`) and the memory mappings show up in
`/proc/<pid>/maps`.

**Fix (dump side):** Munmap every `io_uring` region found in
`/proc/<pid>/maps` (scanned line-by-line, `libc.munmap()` via ctypes).
The `anon_inode:[io_uring]` FDs themselves currently fall under the
`anon_inode:` keep-prefix and are *not* explicitly closed — in
practice this has worked because once the rings are unmapped, what
remains is a bare `anon_inode` that CRIU can dump.

> Caveat: if a future PyTorch/libtorch revision triggers a
> dump failure on `anon_inode:[io_uring]`, tighten the keep-list in
> `prepare_criu_dump` to close those FDs explicitly while keeping
> other `anon_inode:` FDs (eventfd, epoll, …).

---

## Complication 3: POSIX Semaphores (`/dev/shm/sem.*`)

**Problem:** Python's `multiprocessing` module creates POSIX named
semaphores in `/dev/shm/`.  These are memory-mapped into the process.
The semaphore file is unlinked shortly after creation (standard POSIX
pattern), so `/proc/<pid>/maps` shows the path as `(deleted)`.  CRIU
then needs to handle the deleted file — either via ghost remaps or
`link_remap`, both of which cause problems at restore time (see
Complication 9).

**Fix (dump side):** In `prepare_criu_dump` (child process), delete
the `/dev/shm/sem.*` files before the dump.  The live process retains
its existing mmap (the kernel keeps the inode alive).  CRIU then
captures the mapping as anonymous memory — no file reference, no ghost,
no `link_remap`.  Each TP rank does the same for vLLM's message-queue
segments (`/dev/shm/psm_*`, `psm2_*`) in `_prepare_worker_dump`.

**Side effect:** At process exit, Python's multiprocessing finalizers
try to `sem_unlink()` the already-removed files, producing harmless
`FileNotFoundError` warnings.  These are cosmetic only.

### Several replicas share `/dev/shm`: unlink only your own tree's files

Until 2026-10-03 both unlinks globbed `/dev/shm`. With one instance per pod
that was harmless; with up to eight replicas dumping at once in one pod it
deletes files a live sibling still needs. A rank attaching to its broadcast
queue, or a worker spawned with a queue whose semaphores it reopens by name,
then gets `FileNotFoundError`.

`_own_shm_paths(pids, prefixes)` in `vllm_child.py` scopes both:

- the child walks its own tree (`_tree_pids`, through
  `/proc/<pid>/task/<tid>/children`) and unlinks the `sem.*` that tree maps;
- each rank unlinks the `psm_*` its own process maps or holds open.

**A mapping's path is not always the file's name, and the first version of
the scoped unlink missed this.** glibc's `sem_open` creates the semaphore under
a temporary `sem.XXXXXX`, hard-links it to the real name
(`sem.loky-<pid>-...`), and unlinks the temporary. So `/proc/<pid>/maps` shows
`sem.XXXXXX (deleted)` while the file is still reachable under its real name.
Matching by path found nothing to unlink, the file stayed linked, and CRIU,
unable to ghost a file that still has a name, link-remapped it: it hard-links
`/dev/shm/link_remap.<id>` in the dumping pod. Replicas whose trees are shaped
alike number their files alike, so concurrent dumps collide on the same
`<id>`. Round 1 of the multi-replica roll lost four of six Qwen dump jobs this
way, at TP=1, 2 and 4:

```
Warn  (criu/files-reg.c:1881): Can't link  -> ./dev/shm/link_remap.508
Error (criu/files-reg.c:1159): Can't link remap to /dev/shm/sem.gSyh4r: File exists
Error (criu/cr-dump.c:1440): Collect mappings (pid: 100296) failed with -1
```

Even without the collision the image would have been unusable elsewhere: a
link-remapped image needs `/dev/shm/link_remap.<id>` at restore, which exists
only in the dumping pod and is consumed by the first restore.

**Fix (`6bf585b`):** collect `(device, inode)` from every matching maps line,
deleted or not (`os.makedev(major, minor)` from the maps device field, which
equals `st_dev`), then add every `/dev/shm` entry with a matching prefix and
`(st_dev, st_ino)`. That catches the real name of a semaphore whose temporary
name is all `maps` shows. Round 2 dumped 8, 4 and 2 replicas concurrently at
TP=1, 2 and 4 on both Qwen models with no link-remap and no failed dump. Unit
tests in `tests/test_pod_log.py` hold both cases against a fake `/proc`.

---

## Complication 4: stdout/stderr (`outt_test` file)

**Problem:** When running with `&>` or `| tee`, the child's fd 1/2
point to a regular file.  CRIU records the file path and size at dump
time.  Between dump and restore, more output gets written to the file,
changing its size.  CRIU's file validation rejects the restore:

```
File outt_test has bad size 17042 (expect 15267)
```

**History.** The first fix pointed fd 1/2 at `/dev/null` before the dump and
passed `--inherit-fd fd[1]:stdout fd[2]:stderr` at restore. Those two
`--inherit-fd` entries were inert: CRIU matches an inherited fd against a
file's *resource name* (a path such as `tmp/inst0.log`, or `pipe:[N]`), and
nothing is named `stdout`. Later the child wrote to `/tmp/inst<N>.log` opened
`O_WRONLY|O_APPEND` -- which CRIU exempts from the size check -- and needed the
path pre-created on a fresh node (`_precreate_dumped_log_paths`), re-pointed
after restore (`rebind_log`), and the TP ranks moved to `/dev/null` and back
(`_rebind_worker_stdio`). None of it reached `kubectl logs`.

**Fix today: the pod log as an external pipe** (2026-10-02).
- *Cold start:* the worker and the child open `/proc/1/fd/1` -- PID 1's stdout,
  a pipe whose reader is the container runtime -- and `dup2` it onto fd 1 and
  2 (`semip_logging.redirect_stdio_to_pod_log`); the TP ranks inherit it. Opened
  `O_NONBLOCK` (a FIFO with no reader would block the open) and set blocking
  again straight after. Not a pipe, or not openable: `/dev/null`, with one
  warning. `SEMIP_LOG_TARGET` overrides the path.
- *Dump:* nothing is redirected. `_worker_criu_save` records the pipe's name,
  `pipe:[N]`, as `stdout_resource` in `meta.json`, and warns if any task's fd
  1/2 points elsewhere. CRIU dumps a pipe with an outside reader without any
  `--external`, the same way runc checkpoints a container's stdio.
- *Restore:* the worker opens this pod's log pipe (before any unshare, since
  `/proc/1` inside the private namespace is the reaper), sends it to the criu
  helper in the same `SCM_RIGHTS` message as the command pipe, and passes
  `--inherit-fd fd[L]:<stdout_resource>`. The restored tree writes straight
  into the new pod's log. An image without `stdout_resource`, or a pod without a
  log pipe, is refused: without the inherit CRIU recreates the pipe with no
  reader and the first write fails with `EPIPE`.

Verified 2026-10-02 at TP=1 and TP=2 on Qwen3-8B (job `8fa62701`): child and
rank lines before and after the restore in `kubectl logs`, outputs matching,
`files.img` naming no log file. Verified at scale 2026-10-03 on image
`dev_20261003_191750_7f806d170f8`: eight TP=1 replicas sharing one pod log
through dump and cross-pod restore, TP=8 GLM-5.3 and Flash-Next, and both
nodes of an `n_gpus=16` job. No `EPIPE`, and lines from different replicas do
not interleave mid-line (pipe writes up to 4 KiB are atomic).

**Reading a shared log.** Lines carry `[r<K> <role>]` (roles `engine`,
`instance`, `worker`, `child`) and vLLM's own lines carry `[r<K>]` through
`VLLM_LOGGING_PREFIX`, so `grep '\[r3 '` isolates one replica. The kubelet
keeps about 10 MiB of current log per container, and an eight-replica restore
writes over 1,000 lines in its first minute, so collect logs promptly.

---

## Complication 5: Pipe FD passing into the criu helper

**Problem:** The worker communicates with the child via a pipe.  CRIU
needs the pipe FD to be inherited by the restored process
(`--inherit-fd fd[N]:resource`).  But the FD does not survive the hop:
`subprocess` closes all FDs >= 3 in the child, and on the namespace path
`sudo` closes them too.

**Original approach:** `sudo -C 1024` (raise the close-from limit).
Failed because the sudoers policy doesn't permit `-C`.

**Fix:** Use a Unix domain socket with `SCM_RIGHTS` to pass the pipe FD:
1. Worker creates a temporary Unix socket and listens
2. A background thread accepts and sends the pipe FD via `SCM_RIGHTS`
3. A Python helper script connects to the socket, receives the FD,
   `dup2`s it into place, then `execvp`s `criu restore`

The helper runs under `sudo` only on the namespace path, and only when
the worker is not already root (`_is_root`) — skipping it there avoids
leaving a long-lived `sudo` between the worker and the reaper, which
holds the terminal for the life of the namespace and leaves the tty in a
bad state when teardown SIGKILLs it.  The lowcap path never uses `sudo`
at all: criu must run at its target's uid (Complication 11).  The
`SCM_RIGHTS` dance stays in every case, because `subprocess` closes
FDs >= 3 regardless.

---

## Complication 6: CUDA Context (driver API)

**Problem:** After CRIU restore, the CUDA context needs to be
re-established on the target GPU.  The `cuda-checkpoint` CLI
(`sudo cuda-checkpoint --action restore`) works but spawns a subprocess
for every lock/checkpoint/restore/unlock call (~2-5s overhead each).

**Fix:** Use `libcuda.so` directly via ctypes, always:
- `cuCheckpointProcessLock(pid, NULL)`
- `cuCheckpointProcessCheckpoint(pid, NULL)`
- `cuCheckpointProcessRestore(pid, args)`  (with GPU UUID mapping for migration)
- `cuCheckpointProcessUnlock(pid, NULL)`

### Correction: this used to be gated on `euid == 0`, and that was wrong

Until 2026-09-08 the driver API was used only when running as root, with
non-root falling back to `sudo cuda-checkpoint`.  **Root was never the
requirement.**  `cuCheckpointProcess*` needs ptrace permission over the
target process, which a parent already has over its own same-uid child.
Measured from an unprivileged (uid 1000) caller against its own CUDA
process, with `ptrace_scope=0`:

```
cuCheckpointProcessLock(pid)        -> 0
cuCheckpointProcessCheckpoint(pid)  -> 0
```

The uid gate was not merely pessimal.  `cuda-checkpoint` is a **separate
binary, not part of the driver**, and is absent from some images — the DSS
pod image among them.  There an unprivileged dump or restore died with
`cuda-checkpoint: command not found` performing an operation the driver
would have done in-process.  Gating a *capability* on a *uid* turned a
working path into a missing one.

The CLI fallback is therefore gone, and with it `_run_cuda_checkpoint` and
`_build_full_device_map` (which existed only to render the device bijection
as a `--device-map` string; the API path builds the same mapping as a
`CUcheckpointRestoreArgs` struct in `_build_restore_args`, so cross-node
migration is unchanged).  One path now, rather than two that differed in
speed *and* availability.

`_is_root` survives, with one job: skipping the `sudo` prefix on the criu
helper when the worker already is root.  That one is genuinely about
privilege — criu needs it.

---

## Destructive Save

The CRIU dump kills the child process after writing the image.  This
simplifies the lifecycle to a single code path:

- **First run (cache miss):** `init → ... → checkpoint → save` (child dies)
  → `load → restore → generate → teardown`
- **Every subsequent run (cache hit):** `load → restore → generate → teardown`

After `save()`, the model is in `"saved"` state with no live process.
The worker exits cleanly.  This eliminates dangling processes from
non-destructive dumps and avoids the POSIX semaphore futex deadlock
that occurred when remapping deleted shared-memory files.

---

## Image Metadata (`meta.json`)

`save()` writes a `meta.json` file alongside the CRIU image containing:

- **CRIU plumbing**: `child_pid`, `pipe_fd`, `pipe_resource`, `nvidia_fds`
- **Placement and hardware identity**: `rank`, `gpus` (the physical GPU
  list; single-element at TP=1), and `gpu_uuids` — the *capture* node's
  GPU UUIDs, which the restore device map needs as its "old" side so the
  image can be restored on a different node
- **Instance metadata**: `vllm_config` (which may include the reserved
  `_env` mapping of per-model env vars), `model_dir`, `total_gpu_bytes`,
  `pinned_cpu_bytes`, `n_gpus`, `max_pinned_bytes_per_worker`

On `criu_restore()`, the instance validates that the image's
`vllm_config` and `model_dir` both match.  A mismatch raises
`RuntimeError` immediately, before any worker is spawned or CRIU restore
attempted.  The `model_dir` check matters because the image bakes
absolute compile-cache paths (see
[semi-p_DESIGN.md](semi-p_DESIGN.md)).

The child's `os.environ` is itself captured inside the CRIU dump and
restored verbatim by `load()`, so env vars set during cold start
(both the hard-coded trio in `vllm_child_loop` and any
`vllm_config["_env"]` applied before `from vllm import LLM`) survive
restore without further work.  `_env` in `meta.json` is consulted only
on cold-start paths (orchestrator-side `register`) and by
client-side dedup; it is *not* re-applied on `load()`.

---

## Complication 7: CRIU Plugin Directory

**Problem:** `--libdir` is CRIU's *plugin* directory (default
`/usr/lib/criu`, which holds `cuda_plugin.so`).  The dump points it at an
empty directory to prevent CRIU loading any plugin, and if that directory
does not exist CRIU aborts at plugin initialization and the dump fails.

```
Error (criu/plugin.c:228): Unable to open directory /usr/lib/criu/empty: No such file or directory
```

**Why the dump wants no plugin.** By the time `criu dump` runs,
`cuda_checkpoint()` has already driven `cuCheckpointProcessLock` /
`cuCheckpointProcessCheckpoint` through the driver API in
`_worker_checkpoint`, and `prepare_criu_dump` has closed every
`/dev/nvidia*` fd — which is why every image records `"nvidia_fds": {}`.
There is nothing left for the CUDA plugin to handle.

**Fix:** `_worker_criu_save` mints a private empty directory per dump with
`tempfile.mkdtemp(prefix="semip-criu-noplugins-")` and removes it in a
`finally` around the criu invocation.  The path was never meaningful — the
whole contract is "exists, holds no `.so`" — and the old fixed location was
the one property that made it root-only, forcing a `sudo mkdir -p` fallback
that no unprivileged pod could satisfy.

`mkdtemp` rather than a fixed `/tmp` path: it is created fresh so it is
guaranteed empty (a stray `.so` in a long-lived directory *would* be loaded,
by a binary carrying file capabilities), its `0700` mode is set in the same
syscall that creates it so there is no world-writable window, and it is
race-free by construction — which matters because a cold start issues
concurrent dumps.

**The asymmetry is load-bearing.** No restore path passes `--libdir`, so
both of them fall back to the real `/usr/lib/criu` and *do* load the CUDA
plugin, which they need.  Verified in both directions: a dump log contains
zero occurrences of "plugin", while the paired restore log carries
`Plugin "cuda_plugin" (version 512 hooks 13)`.  Applying the temp directory
"uniformly" to the restore would hide `cuda_plugin.so` and break it.

Nothing here is TP-dependent — the same argv serves TP=1 and TP>1.

---

## Complication 8: PID Collisions at Restore

**Problem:** CRIU restores every task at its *recorded* PID (via
`clone3(set_tid)`), so the restore fails if that PID is already taken:

```
Can't fork for 47619: File exists
```

**It is task ids, not just PIDs.** A PID is only the id of a thread
group's leader, and leader ids and thread ids are allocated from a single
per-namespace counter.  CRIU restores *every* task, so the set that must
be free is the whole recorded task list — 915 ids across 3 leaders for a
TP2 image — and the kernel reports the two cases through different code
paths, neither of which names the occupant:

```
Can't fork for 47619: File exists              # a thread group leader
pie: 2103: Unable to create a thread: -17      # any other thread (EEXIST)
```

This is also why `ps` is useless for diagnosing it: threads appear only
under `/proc/<pid>/task`, so a 200-thread service can own hundreds of the
ids an image needs while showing a single unrelated PID.

**And an id with no task at all can be taken.** A pid is a refcounted
`struct pid` in the namespace's IDR, freed only when its last reference
goes, and references come from the task itself *and* from every process
using it as a `PGID` or `SID` — a process group is named by its leader's
pid, so each member holds one on the leader. An unreaped zombie keeps
those links. So `clone3(set_tid=N)` can fail `EEXIST` on an id where
`/proc` shows nothing, because `/proc` lists tasks. This is not a corner
case: it is the dev-cluster failure, and it defeats every task-based check.

Five ways they get taken:

1. **A reference from the dumped tree.** The destructive dump kills the
   child, and `_reap_dumped_child` collects the leader — but the leader
   `setsid()`d at startup, so its pid also names its process group and
   session, and anything it forked holds a reference on that pid. Those
   are the worker's *grandchildren*, unreachable by `waitpid` until
   `_set_child_subreaper` adopts them, after which
   `_reap_orphaned_descendants` frees the id. Guaranteed rather than
   incidental: the id at stake is the image's own `child_pid`.
2. **Concurrent restores.** Two images captured on the same node with
   their child trees alive simultaneously carry adjacent/interleaved
   PIDs, so the second restore collides with the first.
3. **An unrelated host process** happening to hold a recorded id — with
   its threads, most likely, and a long-lived service never yields them.
4. **The restore's own launcher.** The counter is sequential, so a driver
   started when it happens to sit below the image's range lands *inside*
   it, together with its `sudo` wrappers and its interpreter's threads.
   Nothing can move an already-running process, so this one is only
   fixable by placing the image's ids elsewhere.
5. **A stray orphaned by anything else in the pod.** Same mechanism as
   (1) but outside the library's subtree, so no reap of ours can reach
   it. Where PID 1 does not `wait()` — `dss-zone-worker` — these
   accumulate for the life of the pod. Only a reaping init or dump-side
   id placement helps.

**Partial fix (retry loop):** `worker_loop` retries the load up to 5
times with 0.5s backoff, keyed on `_is_pid_collision` so it matches the
leader wording, the thread wording and the preflight's own marker alike.
This only helps for case 3 with a short-lived holder.  A zombie under a
live parent never goes away, and neither does a reference held by one, so
cases 1, 2, 4 and 5 defeat it entirely.

**Diagnosis (reduced-capability path):** `_worker_criu_load_lowcap`
preflights `_pid_collision_report` before spawning criu — decode
`pstree.img`, walk `/proc/*/task`, then scan `pgrp` and `session`
(fields 5 and 6 of `/proc/*/stat`) for every recorded id that has no
task — and re-runs it if criu fails with either collision signature,
since an id can be claimed in between.  Task occupancy is grouped by
occupying process, reference occupancy by held id, because the action
differs: kill the process, versus reap the members.  Cost is ~65 ms,
essentially all of it `crit decode`.  `scripts/pidcheck.py` is the same
check standalone, plus `--burn`/`--burn-to`, which advance the counter by
forking so *subsequently started* processes clear the range (it frees
nothing; `ns_last_pid` would do it in one write but is read-only on these
nodes).

Both were task-only until 2026-09-17, which is why they reported "No
collisions" through a restore that then failed `EEXIST` on the first id
it tried. Treat a clean report from an older build as no evidence.

**Real fix:** restore each tree into its own PID namespace, where the
recorded ids are always free.  See Complication 10.

Where that needs capabilities the node won't grant (Complication 11), the
constraint has to be met some other way, and there are exactly three:

1. **Free the contested ids** — reap them.  Only reaches our own subtree, so
   it needs `_set_child_subreaper` to make the holders ours.  This is what
   ships, and it fixes cases 1 and 2.
2. **Have someone reap the rest** — a reaping init in the pod (`tini`, or
   `waitpid(-1, WNOHANG)` in `dss-zone-worker`).  The only clean answer for
   cases 3 and 5, which no reap of ours can reach.
3. **Re-dump instead of restoring.**  For case 4, where the restoring pod's own
   startup sits inside a warm image's recorded range, treat the collision as a
   cache miss and cold-start: the fresh tree records whatever ids are free.
   Self-healing, unprivileged, and the preflight already identifies exactly
   this condition by labelling the holder an ancestor of the restore.
4. **Place the image's ids above the range a fresh pod uses** — burn the
   counter *dump-side*, before the tree exists.  Rejected as automatic policy
   on 2026-09-17 and **adopted on 2026-09-18**, when a durable cache made case
   4 deterministic rather than occasional: a published image is restored in a
   pod whose launcher occupies the same low ids by construction, so (3) would
   cold-start every time and the cache would never be used.  See
   `IMAGE_CACHE.md` §8, which answers the five original objections point by
   point and concedes two of them.

   `_raise_pid_floor` in `server/semip_engine.py` runs before the `Instance`
   that spawns the tree; `SEMIP_PID_FLOOR` (100000, `0` disables) is the knob
   and `meta.json` records the floor reached.  The measured cost is ~10 s, on
   the dump path only: a fork costs ~540 µs on these nodes, but the counter is
   namespace-global, so `SEMIP_PID_BURN_WORKERS` burners advance it in
   parallel until they saturate the pod's CPU quota.  `pidcheck.py --burn-to`
   remains the manual form of the same operation.

---

## Complication 9: Ghost Remap Race (CRIU 4.2)

**Problem:** When a memory-mapped file has been deleted from disk (the
path shows `(deleted)` in `/proc/<pid>/maps`), CRIU uses a *ghost
remap*: it embeds the file content in the image and creates a temporary
`.cr.<id>.ghost` hard link at restore time, then unlinks it afterwards.

With `--link-remap`, CRIU 4.2 has a race condition: when the restored
process has multiple threads (vLLM processes have 400+), each thread
can attempt to unlink the same `.ghost` file.  The first `unlink()`
succeeds; subsequent threads get `ENOENT`, causing CRIU to exit with
`rc=1` even though the restore actually completed:

```
Couldn't unlink remap /tmp/torchinductor_root/.../tmp12345.ghost: No such file or directory
```

Two sources of `(deleted)` mappings were identified:

1. **Triton JIT `.so` files** — Triton compiles kernel `.so` files,
   `dlopen()`s them, then `unlink()`s the file from disk.  Primarily
   affects FP8 models.
2. **POSIX semaphores** — Python's multiprocessing creates and
   immediately unlinks `/dev/shm/sem.*` files.

Tolerating `rc=1` was attempted but led to unstable processes — the
CUDA context was sometimes left in an inconsistent state, and
subsequent `cudaHostRegister` (repin) calls would crash the child.

Recreating deleted files from `/proc/map_files/` (worker side) was
also attempted, but creating a new file at the same path doesn't
clear the `(deleted)` flag on the existing mapping's inode.
Remapping via `mmap(MAP_FIXED)` inside the child corrupted semaphore
futex state, causing deadlocks.

**Fix:** The destructive dump (no `-R`) eliminates the ghost race by
a combination of two approaches:

1. **Semaphores:** Deleted in `prepare_criu_dump` before dump.  CRIU
   captures the mapping as anonymous memory — no file reference at all.
   A semaphore that is missed and stays linked is not ghosted but
   link-remapped, which fails concurrent dumps and pins the image to its
   pod (Complication 3).
2. **Triton `.so` files:** These remain as `(deleted)` mappings, but
   with the destructive dump the number of ghost remaps is typically
   small (only triton kernels, no semaphores).  The race is still
   theoretically possible but far less likely with fewer ghosts.

If ghost remap races resurface, the remaining fix would be to also
handle triton `.so` files in `prepare_criu_dump` by `munmap` +
`mmap(MAP_FIXED)` from a recreated file (safe for `.so` mappings
which are `MAP_PRIVATE`, unlike semaphores which are `MAP_SHARED`).

---

## Complication 10: Per-Restore PID Namespace (and the tty it forced out)

**Problem:** See Complication 8 — the recorded PIDs are frequently
occupied, and CRIU 4.2 explicitly rejects `--join-ns pid:`, so we cannot
ask it to place the tree into a pre-made namespace.

**Fix: run criu inside a fresh PID namespace held open by a reaper.**
The sudo'd restore helper (`_worker_criu_load` in `worker.py`):

1. Receives the inherited pipe fd (SCM_RIGHTS, as in Complication 5),
   then `unshare(CLONE_NEWPID)` and `fork()`.
2. The host-namespace parent publishes **PID 1's host PID** to
   `reaper.pid` (in the image dir) and blocks — this keeps the namespace,
   and the `sudo` handle, alive.
3. **PID 1** unshares a mount namespace, mounts a private `/proc` (so
   criu sees the namespace's PID view), then forks `criu restore -d`
   into the namespace, records criu's exit code to `restore.rc`, and
   reaps forever. A live PID 1 is required both for the namespace to
   persist and for `clone3(set_tid)` of any non-1 PID to be permitted.
4. After criu detaches, the restored tree reparents onto PID 1. The
   worker discovers the restored root's **host** PID via the inherited
   pipe and drives the rest of the lifecycle with it (CUDA
   checkpoint/restore, `/proc` walks, and signals all use host PIDs,
   which are unaffected by the nesting) — `--pidfile` now holds the
   meaningless in-namespace PID.

Teardown SIGKILLs PID 1, which makes the kernel tear down the whole
namespace and the restored tree in one shot. An `atexit` fallback in
`worker_loop` catches Ctrl-C and unhandled exceptions, and a stale-reaper
sweep (reads a leftover `reaper.pid` and kills it) covers `SIGKILL`
deaths that cannot run cleanup.

**The tty this forced out.** A private PID namespace is incompatible
with a `--shell-job` image. The child inherited the interactive shell's
pts on fd 0, so it was captured as a shell job tied to an external
terminal. On restore inside the new namespace the session and pgrp
collapse to `1`, and CRIU's `TIOCSPGRP` on the host terminal fails:

```
tty: Restore inherited group 1
Error (criu/tty.c:689): tty: Failed to set group 1 on 0: Inappropriate ioctl for device
Error (criu/files.c:1221): Unable to open fd=0 id=0xec
```

**Fix (dump side):** the child sheds its controlling terminal at startup
(`vllm_child_loop`): point fd 0 at `/dev/null` and `os.setsid()` so the
captured tree owns its own session and holds no terminal. With no tty in
the image, `--shell-job` is unnecessary and is dropped from both `criu
dump` and `criu restore`. This matches the CRIU maintainers' guidance for
PID-namespace restore. **Images captured before this change (with a tty /
`--shell-job`) must be re-dumped.**

---

## Complication 11: Unprivileged Dump + Restore (`SEMIP_UNPRIVILEGED`)

**Problem:** the default dump and restore paths need `CAP_SYS_ADMIN` in
two places, so they only run on privileged pods:

1. CRIU's kerndat init builds a throwaway *network* namespace to probe
   kernel features; creating a netns needs `CAP_SYS_ADMIN`. Without it
   criu aborts with "Could not initialize kernel features detection" --
   on **both** dump and restore.
2. The restore path additionally wraps criu in a private PID namespace
   (Complication 10): `unshare(CLONE_NEWPID|CLONE_NEWNS)` + a private
   `/proc` mount, which also needs `CAP_SYS_ADMIN`.

Production pods are non-privileged: they grant only
`CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`, never `CAP_SYS_ADMIN`.

**Fix: `SEMIP_UNPRIVILEGED=1` -- one flag, three effects.**  They are
orthogonal, and only the third is about the PID namespace:

- **Dump** (`_worker_criu_save`): append `--unprivileged`, which makes
  CRIU skip the netns kerndat probe. The dump has no namespace machinery
  of its own, so this one flag is the only change it needs; seizing the
  target still uses `CAP_SYS_PTRACE`.
- **The child's capabilities** (`_SEMIP_CHILD_DROP_CAPS`, set by the
  worker only across the child's spawn): the child calls
  `PR_CAPBSET_DROP` + `capset()` before `import torch`, so every task in
  the image records an empty cap set. **This is not a runtime difference
  -- it changes what is written to disk.** The two configurations
  therefore produce materially different images and are not
  interchangeable after the fact; see "image portability is decided at
  dump time" below.
- **Restore**: take `_worker_criu_load_lowcap` instead of
  `_worker_criu_load`. It runs `criu restore -d --unprivileged`
  **directly in the host PID namespace** -- no `unshare`, no reaper, no
  private `/proc`, and no `sudo`. The recorded PID is therefore the host
  PID, so `--pidfile` is authoritative (with a pipe-scan fallback), and
  teardown `os.kill`s the enumerated tree directly (holder
  `{"kind": "tree", "pid": host_pid}`) rather than collapsing a namespace.
  Losing the namespace is what makes recorded task ids contended, which
  is why only this path runs a collision preflight (Complication 8).

Why the asymmetry (a flag on dump, a whole function on restore): the dump
had only dependency (1); the restore had (1) **and** (2), and (2) lives
in the sudo'd wrapper *around* criu, so it cannot be undone by a criu
flag -- it needs a different, wrapper-free restore procedure.

Because `-d` makes criu exit with the restore rc, the sudo process's exit
code *is* criu's rc, so this path needs none of the `reaper.pid` /
`restore.rc` side-channel files the namespace path uses. Its stdout and
stderr must go to `/dev/null` rather than pipes: with `-d` the restored
process inherits fd 1/2 and holds them open after criu exits, so a pipe
would never EOF and draining it would deadlock.

**Capability floor:** `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`, no
`CAP_SYS_ADMIN`. `clone3(set_tid)` at the recorded PIDs in the host PID
namespace is authorized by `CAP_CHECKPOINT_RESTORE`, so it does not need
to write the read-only `ns_last_pid` sysctl. Validate a node with
`sudo criu check --unprivileged` (a residual read-only `ns_last_pid`
complaint from the checker is expected and does not block restore).

That floor holds for the *restore* only because the image records
`no_new_privs`. Without it the restore needs `CAP_SETPCAP` as well, which
no `setcap` can supply on a pod whose bounding set omits it — see
Complication 14, which is the one thing to read before narrowing a pod's
capabilities.

**`SEMIP_UNPRIVILEGED` names the node, not the user.** The flag only
avoids the operations that need `CAP_SYS_ADMIN`; it says nothing about
who you are. Neither the low-cap restore nor the dump shells criu out
under `sudo` any more (see the next subsection). The dumped tree's uid is whatever launched it, and
dropping capabilities does not change it: `_SEMIP_CHILD_DROP_CAPS` calls
`PR_CAPBSET_DROP`, an ambient clear and `capset()`, with no `setuid`
anywhere. A tree launched under `sudo` is therefore `uid 0` with
`cap_eff=[0, 0]` -- which is what every image recorded historically, and
why a root criu could always dump it.

**A root criu cannot dump an unprivileged child.** Discovered 2026-09-08,
the first time a dump was attempted as a non-root user. criu reads every
rlimit of its target with `prlimit()`, and the kernel's
`check_prlimit_permission()` requires either that the caller's *real* uid
and gid match the target's `uid/euid/suid` and `gid/egid/sgid`, or
`CAP_SYS_RESOURCE`. On a cap-prod pod that capability is absent from
`CapEff` **and** `CapBnd`, so nothing can acquire it, and the dump dies
before writing an image:

```
Error (criu/cr-dump.c:389): Can't get rlimit 0: Operation not permitted
Error (criu/cr-dump.c:1584): Dump core (pid: N) failed with -1
Error (criu/cr-dump.c:1975): Dumping FAILED.
```

Measured across all three pairings, which is what makes the rule concrete:

| criu's real uid | target uid | `prlimit` |
|---|---|---|
| unprivileged | same user | OK |
| root | unprivileged | **EPERM** |
| root | root | OK |

So criu must share its target's uid. `_worker_criu_save` therefore
invokes criu directly rather than through `sudo`: the target is the
worker's own child, so not elevating is exactly what makes the uids
match. A root worker is already root and behaves as before; an
unprivileged one takes criu's capabilities from the binary instead:

```bash
sudo setcap cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip \
    /usr/sbin/criu
```

The dump itself needs only the first two; `cap_setpcap` and `cap_setgid`
are what the *restore* needs, and are included here so the binary is
qualified once. See the next subsection.

Verified 2026-09-10 on a cap-prod pod: with the `sudo` gone, a neutrino
dump of two models (27B and 35B, TP1) completed in 6.8 s and 7.2 s and
recorded `uid=1000 gid=1000 cap_eff=[0, 0]`.

Keep the `+e`. Without the effective bit a real-uid-0 `execve` lands with
an empty effective set, which breaks criu for a *root* worker; with it both
callers work,
since `capabilities(7)` treats the file's permitted set as all-ones for
uid 0. Note this is *not* a question of who owns the image files -- the
failure is a permission check on reading the target's limits, and it
happens before any image exists. The setcap is node-local and does not
survive the pod.

**`CAP_SYS_PTRACE` is a separate prerequisite for a cross-uid dump**, and
was missing on the first cap-prod pod tried (`CapEff` and `CapBnd` both
`0x00000100a80425fb`). `ptrace_may_access` takes the same shape as the
rlimit check: same-uid callers pass, cross-uid callers need the
capability, so a root criu dumping a root child never needed it. Its
absence surfaces in compel rather than in criu proper:

```
Warn  (compel/src/lib/infect.c:132): Unable to interrupt task: N (Operation not permitted)
Error (criu/cr-dump.c:1975): Dumping FAILED.
```

Fixing that gap alone is not enough -- it just moves the failure to the
`prlimit` check above, which is the next cross-uid operation in the dump.

**The restore drops its `sudo` too, and for a second reason: pipe
capacity.** CRIU recreates every pipe at its recorded size with
`F_SETPIPE_SZ`, and the kernel refuses to grow a pipe past
`fs.pipe-user-pages-soft` for a user without `CAP_SYS_RESOURCE` --
outside a cap-prod bounding set, so `sudo` cannot supply it. The budget
is **per uid, not per process**, and uid 0 is the account most likely to
be over it, because every root-run tree on the node shares one pool.
Measured 2026-09-10 on a node carrying three stale root vLLM trees:

| uid | fresh pipe | `F_SETPIPE_SZ(64 KiB)` |
|---|---|---|
| 1000 | 65536 | OK (and up to `pipe-max-size`) |
| 0 | **8192** | `EPERM` |

The 8 KiB is the tell: `alloc_pipe_info()` hands out `PIPE_DEF_BUFFERS`
(16 pages) normally but falls back to `PIPE_MIN_DEF_BUFFERS` (2 pages)
once the creating user is over the soft limit, and the same predicate
then fails any growth. A root restore of an ordinary 64 KiB pipe dies
immediately:

```
(00.022224) Restoring size 0x10000 for 0x184ff236
Error (criu/pipes.c:165): Can't restore pipe size: Operation not permitted
Error (criu/files.c:1221): Unable to open fd=3 id=0x1b3
Error (criu/cr-restore.c:2331): Restoring FAILED.
```

Killing the root trees clears the budget, but only until the next
root-run anything; restoring at the image's own uid is the durable fix.
It also removes a credential operation an unprivileged criu could not
perform: with the image's uid equal to criu's, `restore_creds()` has no
`setresuid` to do.

It does **not** remove `PR_CAPBSET_DROP`, which this document asserted
until 2026-09-14 and which cost a full debugging session. See
Complication 14: the drop fires for every capability *absent* from the
recorded `cap_bnd` whether or not it is already absent here, so "the
child's cap-drop left `cap_bnd` equal to the pod's" is the reason it
fires ~39 times, not the reason it is skipped.

**Supported configurations: `root -> root` and `unprivileged ->
unprivileged`. The mix is out of scope and is rejected.** Everything
above is about criu matching *its target's* uid within one dump. The
same identity must also carry across to the restore, for an unrelated
reason: the restored child keeps the uid recorded in the image, so
pairing it with a parent of a different uid leaves both writing the same
files with neither able to write the other's -- `rebind_graphs()` on
`<model_dir>/compilation`, and the restore's own writes into `image/`. (Until
2026-10-02 the child's `/tmp/inst<N>.log` was a third such file.) No file mode
reconciles them, because CRIU
re-validates the recorded mode of every path it re-maps.

Rather than leave that to be discovered as a `PermissionError` at the end
of an expensive restore, `_worker_criu_save` records the dumping `uid`
(and `gid`) in `meta.json`, and `Instance.criu_restore` raises on a
mismatch before it spawns a worker:

```
RuntimeError: uid mismatch: image at <dir> was dumped by uid 0 but this
process is uid 1000; the restored child would keep uid 0 and collide with
this parent over <model_dir>/compilation and <dir>. Restore as
uid 0, or re-dump the image as uid 1000
```

Images captured before that field existed carry no `uid` and are let
through unchecked, so the old hazard still applies to them -- re-dump if
you are unsure which identity wrote one.

**Trade-off (accepted):** without the private PID namespace every
recorded *task* id must be free on the host -- threads included, one
number space, ~900 ids for a TP2 image -- and any unrelated process whose
threads span the image's range blocks the restore. A collision surfaces as
CRIU "File exists" (leader) or `pie: Unable to create a thread: -17`
(thread); the path preflights for both and names the occupants, and the
retry loop recognizes either wording. See Complication 8 for the id
arithmetic and for placing an image's ids out of contention at dump time.

**The binding constraint is id collision, not restore count.** This was
long recorded as "at most one live restore per node", which overstates
it. On 2026-09-08 three images restored and served concurrently on one
node -- 27B TP1, 35B TP1 and 35B TP2+EP, on GPUs 0, 1 and [2,3]. What
makes that work is *how the images were dumped*: the three dump processes
ran concurrently, so the kernel interleaved their allocations and the
recorded id sets came out mutually disjoint (leaders 18255, 18257, and
20135/21735/21736). Their *ranges* overlap heavily while sharing no
individual id, which is all the restore requires.

The practical rules that follow. Dumping images concurrently is what buys
concurrent restores, so do it deliberately rather than as a time saving.
Dumping them serially is the pattern to avoid, since each run starts from
a similar `ns_last_pid` and the images collide by construction. And
`scripts/test_weights.py` is not inherently incompatible with this mode,
as previously stated -- it restores two models concurrently, which
succeeds exactly when its two images hold disjoint ids. Run
`scripts/pidcheck.py <image_dir>` per image to know before spending an
attempt; it reports the required id count, the range, and any occupant.

**Restorable image requires shed capabilities *and* `no_new_privs`.** CRIU's
`restore_creds()` calls `capset()` to reinstate each task's recorded caps; if
the image recorded caps the low-cap node can't grant, restore fails at
`criu/pie/restorer.c` ("Unable to restore capabilities"). Shedding them is
necessary but not sufficient — the bounding-set trim in the same function
fails independently, which is Complication 14. So in
unprivileged mode the vLLM child zeroes its caps *before* `import torch`
-- torch and vLLM spawn many background threads and CRIU records
credentials **per thread**, so dropping first is what makes every task
record an empty cap set. This is **implied by `SEMIP_UNPRIVILEGED=1`**,
not a separate flag: the worker sets an internal
`_SEMIP_CHILD_DROP_CAPS` signal only across the child's spawn, because
the worker itself must keep its caps to run `criu` against that child.

Consequently **image portability is decided at dump time**: an image
dumped without the flag records real caps and cannot be restored on a
low-cap node without re-dumping.

### Precondition: everything the child reads must be world-accessible

The worker tree runs as root, and root bypasses file permission checks
via `CAP_DAC_OVERRIDE`.  Zeroing the capabilities takes that away while
leaving uid 0 in place, so from that instant the child is subject to
ordinary permission checks.

**"World-accessible" overstates it slightly: the child still passes owner
checks as uid 0.** What it loses is the override, not its identity, so
root-owned paths remain reachable at any mode -- `/root` at `drwx------`
is fine, because the child *is* the owner. Only paths owned by somebody
else need `o+x` on every directory and `o+r` on the files. Measured by
dropping all capabilities while staying euid 0: `/root` (mode 0700, root)
lists fine, while `/home/<user>/.cache` (mode 0750, user-owned) gives
`[Errno 13] Permission denied`.

The practical consequence is about where the weights live. A root-run
cold start downloads into `/root/.cache/huggingface` and works with no
mode changes at all. Pointing it at an unprivileged user's existing cache
does not, until that user's home gets `o+x` -- one directory, and never
`chmod -R`, which would rewrite file modes CRIU validates against the
dump.

If the interpreter itself lives behind a private directory, the child
dies immediately after the drop, and the symptom does not look like a
permission problem.  `multiprocessing` spawn runs `spawn_main` ->
`_main` -> unpickle, and unpickling imports the target's module
(`vllm_child`), which is where the drop fires.  The next stdlib import
needed to finish that same unpickle then fails:

```
[semip] dropped capabilities for portable image (capset rc=0)
Traceback (most recent call last):
  File ".../multiprocessing/spawn.py", line 132, in _main
ModuleNotFoundError: No module named 'multiprocessing.popen_spawn_posix'
```

followed by `init FAILED (child pipe broken, ... state=Z (zombie),
exited_code=1)`.  A `ModuleNotFoundError` for a stdlib module right after
the cap-drop line always means this.

Observed with a conda interpreter under a `drwxr-x---` home directory.
Note a venv does not help by itself: it shares its base interpreter's
stdlib, so `/data-fast/myenv/bin/python` built on
`/home/user/miniconda3/envs/dev` still reads its stdlib from under the
private home -- switching between several such venvs changes nothing.
Check the base, not the venv:

```bash
namei -l "$(python -c 'import sys; print(sys.base_prefix)')"
```

and fix by granting traversal (`chmod o+x /home/<user>`, which permits
path traversal without allowing directory listing) or by using an
interpreter under a world-readable prefix such as `/usr/lib/python3.12`.

The operative variable is *where the interpreter lives*, not how it was
installed.  A pre-baked pod image that ships vLLM, CRIU and Python
system-wide satisfies this for free -- `/usr/lib/python3.12` and
`/usr/local/lib/python3.12/dist-packages` are `drwxr-xr-x`, so dropping
capabilities changes nothing that process could already read.  That is
why the mode appears to "just work" on such an image and then fails on a
development pod where the same code runs from a venv on a private home.
A venv built on `/usr/bin/python3` is equally fine; a venv built on a
conda install under `$HOME` is not.

`criu` itself is never implicated: it lives at `/usr/sbin/criu`
(`-rwxr-xr-x`) and runs in the worker, which keeps its capabilities.

**`criu_restore` itself is safe; what runs after it is not.**  The
restore maps an already-initialised process back into memory, so the
child resumes in-memory code, imports nothing, and performs no
DAC-checked read.  The file reads the restore *does* need (reopening
every file-backed mapping) are done by `criu` in the worker, which keeps
its capabilities -- only the child ever drops.

That makes `init` the obvious victim: it spawns a fresh interpreter, so
the child runs Python's import machinery, which is exactly where the drop
fires.  But **any post-restore primitive that spawns a process or imports
a not-yet-loaded module hits the same wall**, and at TP>1 `reinit_nccl`
does:

```
_reinit_nccl -> init_worker_distributed_environment
             -> init_distributed_environment -> _node_count
             -> in_the_same_node_as        (vllm/distributed/parallel_state.py)
             -> shared_memory.SharedMemory(create=True)
             -> multiprocessing.resource_tracker.ensure_running()   # spawns python
```

The tracker spawn cannot read its own stdlib, dies, and the write to its
pipe fails with `BrokenPipeError: [Errno 32] Broken pipe`.

**The symptom is a silent hang, not an error.**  vLLM wraps that block in
`contextlib.suppress(OSError)`, and `BrokenPipeError` *is* an `OSError`,
so nothing is logged at all -- not even the `Error ignored in
is_in_the_same_node` line, which only fires for non-`OSError`.  The
source rank skips its `broadcast_object_list` and proceeds to the
`barrier()` a few lines below, while every other rank waits forever in
the matching broadcast.  `py-spy dump` on the workers shows the two ranks
stopped at *different* lines of `in_the_same_node_as`, which is the
signature.  All you see in the log is `reinit_nccl` never returning and
vLLM repeating:

```
No available shared memory broadcast block found in 60 seconds.
```

So the precondition is not cold-start-only: an image dumped before it was
met keeps *restoring* fine and then deadlocks in `reinit_nccl`.  The fix
is the same (`chmod o+x` the home, or a world-readable interpreter
prefix) and needs no re-dump, since nothing about the image is wrong.

---

## Complication 12: `TIME_WAIT` vs the Recorded Local Port

**Problem:** CRIU records every inet socket's exact local port and
**rebinds it** at restore, *before* `--tcp-close` gets to close it.  The
dump is destructive, so every socket still open at dump time closes with
a FIN when the tree is killed and leaves its tuple in `TIME_WAIT` on the
host.  A restore that follows its own dump immediately then collides
with those tuples:

```
Error (criu/sk-inet.c): Can't bind inet socket (id 0x…): Address already in use
```

The bind cannot step over the tuple because CRIU restores `SO_REUSEADDR`
from the recorded value rather than forcing it on.  Decode an image to
see which sockets are exposed:

```bash
sudo crit decode -i <image_dir>/files.img --pretty \
  | jq -c '.entries[] | select(.type=="INETSK") | .isk
           | {state, src_port, reuseaddr: .opts.reuseaddr}'
```

Listeners — and the sockets accepted from them, which inherit the option
— carry `reuseaddr: true` and rebind fine.  The client-side connected
sockets carry `reuseaddr: false`, and those are the ones that fail.

> **Scope correction.** "Listeners rebind fine" is true *of this
> complication only*, i.e. against a `TIME_WAIT` tuple, which is what
> `SO_REUSEADDR` exists to step over.  A listener still fails flat against a
> **live** `LISTEN` socket on the same port, because that needs
> `SO_REUSEPORT` on both sockets and the recorded value is unset.  That is a
> separate failure wearing this same error string — not time-bound, not
> retryable, and image-bound rather than window-bound.  See **Complication
> 15** before concluding that a "can't bind" failure will drain on its own.

**Deterministic at TP2+, a race at TP1.**  Every rank adds a TCPStore
connection, so at TP>1 there is always a colliding tuple.  A TP1 image
still contains such sockets (the store's self-connection, plus whatever
HTTPS connections the child left open), so TP1 is exposed too — it just
usually wins the race.

**Fix (dump side): `SO_LINGER(1,0)` before the NCCL teardown.**
`_mark_inet_sockets_rst` (`vllm_child.py`) walks the worker's
`/proc/<pid>/fd` and, for every `AF_INET`/`AF_INET6` `SOCK_STREAM`
socket, sets `SO_LINGER` with `l_onoff=1, l_linger=0` so the eventual
close sends an **RST** instead of a FIN and the tuple never enters
`TIME_WAIT`.  It runs at the top of `_destroy_nccl`, before anything is
closed, and it marks through a `dup()` so the wrapper's `close()`
releases only the duplicate: the live fd stays open for CRIU, and
because the option lives on the shared socket the RST fires whenever
teardown or CRIU's kill finally closes it.  `AF_UNIX` and non-stream
sockets are left alone.

**Time-bound, not image-bound — no re-dump needed.**  `TIME_WAIT` is
`TCP_TIMEWAIT_LEN` (60s, hard-coded in the kernel; `tcp_fin_timeout` is
a different timer), so the window closes a minute after the dump that
opened it.  Unlike Complications 10 and 11, **images dumped before this
change do not need re-dumping** — they restore fine once the window has
drained, which is exactly why the failure looks like it needs a
cooldown.

What a fresh dump does buy: CRIU stores the option in the image
(`SkOptsEntry.so_linger`) and replays it at restore, so a tree restored
from a new image also RSTs on its own teardown.  That keeps back-to-back
`criu_restore` → teardown → `criu_restore` cycles on one node clean,
where an old image re-blocks its ports on every teardown.

**Not covered (known, benign today):** the marking sits after the
`tp_size <= 1` early return in `_destroy_nccl`, so TP1 is never marked;
and `_destroy_nccl` is a `collective_rpc` target, so only the workers
are walked — the child's own sockets (e.g. leftover HTTPS connections to
the model hub, bound to the routable IP rather than loopback) still
leave `TIME_WAIT` behind.

---

## Complication 13: Scoping the Teardown Kill (no-namespace path)

**Problem:** the namespace path tears a restored tree down by SIGKILLing
the namespace's PID 1 and letting the kernel collapse everything inside
it — bounded by construction. The lowcap path (Complication 11) has no
namespace, so it has to name its victims. `criu restore -d` detaches the
tree, so tasks reparent onto PID 1 and escape a `_get_descendant_pids`
walk from the root. The original code recovered them with a process-group
kill:

```python
subprocess.run(["sudo", "kill", "-9", f"-{root_pid}"], capture_output=True)
```

That never became a group kill. `/usr/bin/kill` is procps-ng, which parses
arguments with `getopt_long` under `opterr=0`, so a multi-digit negative
pid is consumed as an *option cluster* and never reaches the pid-operand
loop. Its handler for that case derives the target from the first digit
alone (`pid = (long)('0' - optopt)`), silently discarding the rest. So
`sudo kill -9 -1181` ran `kill(-1, SIGKILL)`: as root, every process it is
permitted to signal. It killed the worker mid-sweep and PID 1's only
child, which exits the container with code 137 and burns a `backoffLimit`
retry. One pod exhausted all six and was terminated, returning its 8
H200s to the shared pool. (Long-standing procps bug; Ubuntu #1637026
describes the same failure wiping Hadoop nodes.)

Only the first digit matters, which is why this looked like a TP problem
and is not one: `tp2test` restores at 1181 → `kill(-1, ...)`, fatal, while
`qwen_27b`/`qwen_35b` restore at 3109/3111 → `kill(-3, ...)`, ESRCH on an
empty process group. A TP=1 image restoring at a pid beginning with `1`
would collapse the pod just as reliably.

**Why not scope it in the kernel.** `cgroup.kill` exists on these nodes
and is the ideal mechanism — write `1` and the kernel atomically SIGKILLs
exactly that cgroup, immune to pid reuse. It is unusable here:
`/sys/fs/cgroup` is mounted `ro` and cannot be remounted, because
`CAP_SYS_ADMIN` is absent from the bounding set. That is the same
constraint that forces the lowcap path to exist at all.

**Fix: snapshot identity at restore, verify every victim at teardown.**
`_tree_identity(root_pid)` records each task id with its start time (field
22 of `/proc/<pid>/stat`) while the tree is known-live, and becomes the
holder. `_kill_restored_tree` then kills only positive pids that are not
`<= 1`, not in `_own_ancestry()`, still live, and whose start time still
matches the snapshot — so a recycled pid is never hit. The live descendant
walk that catches tasks forked since the snapshot runs only while the root
itself still matches, since pids it discovers carry no recorded start time
to check against. The victim list is logged *before* the kills, because
the old code logged after and destroyed the evidence along with the
worker.

No re-dump: this reads live `/proc` after a restore has already succeeded,
so image format, `pstree.img`, and `meta.json` are untouched.

Full incident record, invariants, and test plan in
[`TEARDOWN_SCOPING.md`](TEARDOWN_SCOPING.md).

---

## Complication 14: The Bounding-Set Trim Needs `CAP_SETPCAP` Anyway

**Problem:** on a pod granting exactly `CAP_CHECKPOINT_RESTORE +
CAP_SYS_PTRACE` (`CapBnd 0000010000080000`), a dump succeeds and the restore
of that image **hangs forever**. `restore.log` carries, once per task:

```
pie: 34661: Error (criu/pie/restorer.c:317): Unable to drop capability 0: -1
pie: 34661: Error (criu/pie/restorer.c:820): BUG at criu/pie/restorer.c:820
```

Capability 0 is `CAP_CHOWN`, which is a red herring: it is simply the first
of the ~39 capabilities absent from the recorded `cap_bnd`. `-1` is `-EPERM`.

**Why it fires at all.** `restore_creds()` trims the bounding set by calling
`PR_CAPBSET_DROP` for every capability *absent from the image's recorded
`cap_bnd`*, and it never checks whether that capability is already absent
from its own set:

```c
        if (args->cap_bnd[b] & (1 << i))
                /* already set */
                continue;
        ret = sys_prctl(PR_CAPBSET_DROP, i + b * 32, 0, 0, 0);
```

The kernel's `cap_prctl_drop()` tests `ns_capable(CAP_SETPCAP)` *before* it
tests anything else, so a drop of an already-dropped capability still returns
`EPERM`. Restoring into the same pod that produced the image is therefore not
sufficient: the sets match exactly and all ~39 calls still fire and still
fail.

**Why the failure presents as a hang rather than an error.** The restorer
stubs run in the restored task's own address space with no libc and no way to
propagate a return code. `restore_creds()` returning `-1` reaches `BUG()` at
both call sites (`restorer.c:820` in the thread path, `:2292` in the leader
path), and criu's master is left waiting on a restore stage its tasks will
never finish. There is no `Restoring FAILED` line, the rc never arrives, and
`/tmp/inst<N>.log` stops mid-command at `>>> criu_restore`. Observed live 93
minutes after the fact, with 214 half-restored tasks stranded on exactly the
ids the next attempt needs. **A hung `criu restore` with no error is this
bug**; kill the criu and the stranded tree before retrying.

**`setcap` cannot fix this, and makes it worse.** `CAP_SETPCAP` is outside
the bounding set, and `bprm_caps_from_vfs_caps()` computes
`pP' = (pB & fP) | (pI & fI)` and then returns `EPERM` if any file-permitted
bit did not survive — whenever the file's effective bit is set, which the `+e`
required for a root worker does. So adding `cap_setpcap` to the `setcap` line
does not grant it silently-uselessly; it makes `criu` **fail to `execve` at
all**. On a `drop: ["ALL"]` pod the only alternative would be widening the
Gatekeeper policy to hand `CAP_SETPCAP` to every sampling pod.

**Fix (dump side): record `no_new_privs`.** criu already tolerates this exact
`EPERM`, because `NO_NEW_PRIVS` independently prevents inheriting
capabilities outside the bounding set, making the drop unnecessary:

```c
        if (!ce->has_no_new_privs || !ce->no_new_privs || args->cap_prm[b] & (1 << i)) {
                pr_err("Unable to drop capability %d: %d\n", i + b * 32, ret);
                return -1;
        }
        pr_warn("Unable to drop capability %d from bset: %d (but NO_NEW_PRIVS will drop it)\n", ...);
```

All three conditions must hold. The child's `cap_prm` is already `0`, so the
only missing piece was the flag itself, which
`_drop_caps_for_portable_image` (`vllm_child.py`) now sets with
`prctl(PR_SET_NO_NEW_PRIVS, 1)`. It needs no privilege, and it must be set
*before* `import torch` for the same reason the cap-drop is: criu records
credentials per thread, and `NO_NEW_PRIVS` is inherited across `fork` and
preserved across `execve`, so setting it first is what puts it in every
task's `CredsEntry`. No criu patch is needed — the dump side already plumbs
it through (`parasite.c:354` reads `PR_GET_NO_NEW_PRIVS`,
`parasite-syscall.c:120` sets `has_no_new_privs`, `creds.proto:27` carries
it).

It must **not** leak to the worker, which `execve`s a file-capability
`criu`: `NO_NEW_PRIVS` makes the kernel strip those capabilities. The
existing `_SEMIP_CHILD_DROP_CAPS` gating already confines it to the child.

**Verified 2026-09-14** by A/B on a 5-task stand-in rather than a 35B cold
start, since only the credential path is under test:

| image | recorded `no_new_privs` | `criu restore -d --unprivileged` |
|---|---|---|
| without the `prctl` | absent | **hangs** (killed at 45 s); 5 fatal `restorer.c:317` |
| with the `prctl` | `1` | **rc=0**, tree alive; 195/195 `EPERM`s demoted to `NO_NEW_PRIVS will drop it` |

Both recorded `cap_bnd=[524288, 256]`, identical to a real 35B image from the
same pod.

**Re-dump required.** This is a dump-time property, so
`SEMIP_UNPRIVILEGED=1` now changes what is written to disk in *two* ways (the
capability shed and this flag). Any image captured before this can never
restore on a two-capability pod. Confirm the flag landed before spending a
restore:

```bash
grep NoNewPrivs /proc/<child-pid>/status          # 1, while the child lives
crit decode -i <image_dir>/core-<pid>.img | grep -i no_new_privs
```

**Not this bug: `CAP_SETGID`.** `restore_creds()` compares `getgroups()`
against the recorded list and skips `sys_setgroups` when they match, so a
dump and restore in the same pod spec never needs it.

**Not this bug: `TCP_REPAIR`.** The same `restore.log` carries a handful of

```
Error (criu/include/sk-inet.h:80): Failed to turn off repair mode on socket 68: errno 1
```

which is `EPERM` wanting `CAP_NET_ADMIN` and is harmless. `tcp_repair_off()`
returns `void`, and under `--tcp-close` `restore_one_tcp()` only
`shutdown()`s and never enables repair — so `post_open_inet_sk` schedules a
repair-off for a socket that was never in repair mode. criu orders the
repair-off *before* the creds restore precisely because it is privileged.

---

## Complication 15: Recorded Ephemeral Listeners vs a Live Occupant

**Read this together with Complication 12, which reports the same criu error
for a different cause and whose conclusion does not cover this case.**

**Problem:** a dumped image records every listening socket's exact local
port, and criu must `bind()` each one to restore the fd. torch's TCPStore and
Gloo listeners sit on **kernel-assigned ephemeral ports**, so the image
demands a specific set of ephemeral ports back. Any process anywhere in the
restoring pod's network namespace may already hold one, and a single occupied
port fails the whole restore:

```
inet: Restore: family AF_INET type SOCK_STREAM proto IPPROTO_TCP port 42513 state TCP_LISTEN src_addr 127.0.0.1
Error (criu/sk-inet.c:1059): inet: Can't bind inet socket (id 464): Address already in use
Error (criu/files.c:1221): Unable to open fd=70 id=0x1d0
Error (criu/cr-restore.c:2331): Restoring FAILED.
```

The TP1 35B image (`77d95928d2ac`) records **seven** such listeners plus
`0.0.0.0:8000`, so each restore is seven independent chances to collide:

| image | recorded `TCP_LISTEN` on 127.0.0.1 |
|---|---|
| Qwen3.6-35B-A3B TP1 (`77d95928d2ac`) | 34817, 45383, 41455, 42513, 38435, 36607, 40745 |
| Qwen3.8-27B TP1 (`552cc22540cd`) | 38863, 40951, 39629, 41175, 41887, 33013 |

**`SO_REUSEADDR` does not save a listener here, and Complication 12's
"listeners … rebind fine" is only true against `TIME_WAIT`.** The socket that
failed carries `reuseaddr=True`:

```
id=464   LISTEN   port=42513   reuseaddr=True   reuseport=None
```

`SO_REUSEADDR` permits binding over a `TIME_WAIT` tuple. It does **not**
permit binding over a *live* `LISTEN` socket — that needs `SO_REUSEPORT` on
both sockets, and the recorded value is unset. So a listener rebinds fine
against Complication 12's self-inflicted `TIME_WAIT` tuples and fails flat
against a foreign live occupant.

**Not time-bound, and not retryable.** This is the sharpest difference from
Complication 12. The occupant is a live socket that outlives the restore, so
waiting does not clear it: **seven consecutive attempts over 78 s were
observed failing on the identical port** (`inst0` … `inst6`, all
`id=464`/`port=42513`/`fd=70`). Do not add "Address already in use" to
`_PID_COLLISION_SIGNATURES` or any retry gate — a retry rescues a zombie
being reaped, which is Complication 8's case, not this one. The
intermittency is *across pods* (did anything in this pod land on one of the
seven ports?), never across attempts within one pod. Once a pod loses that
coin flip, that pod cannot restore that image at all.

**This is why the failure looks placement-dependent at TP1.** Complication 12
is "deterministic at TP2+, a race at TP1"; this one bites at **every** TP. It
is the first known placement-dependent restore failure at TP1, which had
otherwise been exempt (slot mismatch needs TP2+).

### TP>1: same exposure, different processes, and one extra hazard

The TP8 GLM-5.3 image (`0b69f511e035`) records **17** listeners across 9
processes (driver with 199 threads, plus 8 workers of ~14 each):

| what | count | `reuseaddr` |
|---|---|---|
| `::` (v6 wildcard) `:8000` | 1 | `True` |
| `::1:28028` | 8 (one per worker) | `True` |
| `127.0.0.1:<ephemeral>` — 33115, 33971, 35921, 41739, 46791, 47409, 48269, 50015 | 8 (one per worker) | **`False`** |

Three things follow:

1. **The ephemeral exposure does not shrink with TP** — eight loopback
   ephemeral listeners at TP8 against seven at TP1. The count tracks ranks
   rather than being a fixed per-process cost, so it will grow with TP.
2. **At TP>1 those listeners carry `reuseaddr=False`**, unlike TP1's `True`.
   So the per-worker listeners are exposed to Complication 12's `TIME_WAIT`
   mechanism *as well as* this one, on the very same sockets. That is a second
   way Complication 12's "listeners carry `reuseaddr: true`" generalization
   does not hold.
3. **The driver-side close reaches none of them.** Each rank is its own
   process, so `prepare_criu_dump`'s own sweep only covers the driver — and at
   TP8 the driver owns **zero** recorded listeners. criu's seize list names the
   nine processes (`100015` as the tree root, then `100245`–`100252`), and the
   root never appears as the owning pid of a `TCP_LISTEN` record. Even
   `::`:8000 lives in a worker. So the per-worker close in
   `_prepare_worker_dump` (invoked via `collective_rpc`) is not the larger half
   at TP>1, it is the *only* half. **A TP1-only fix would have closed nothing
   whatsoever on the TP8 image.**

### `::1:28028` is NCCL's RAS listener, and closing it destroys the restore

**This was got wrong once, at the cost of a GLM dump/restore cycle. Do not
close it.** An earlier revision of this section argued that because the port is
a listening socket on `::1` the loopback close covers it, and that doing so also
removed the hazard of two concurrent TP>1 restores both demanding it. Both
halves of that were wrong in the way that matters.

`28028` is NCCL's RAS port. Confirmed from the shipped library:

```bash
$ strings libnccl.so.2 | grep -iE "28028|NCCL_RAS"
NCCL_RAS_ENABLE
localhost:28028
NCCL_RAS_ADDR
```

That also closes the "open lead" this section used to carry about why eight
processes share one port without `SO_REUSEPORT`: they do not share anything.
Each rank runs its own RAS listener on NCCL's default address.

**The RAS thread is not in the torch process group, so no teardown stops it.**
It goes into the image alive, with its socket fd recorded and — after the close
— no socket behind it. On restore it resumes `accept()`, gets `EBADF`, logs a
warning, and retries with no backoff:

```
ras/client_support.cc:203 NCCL WARN Call to accept failed: Bad file descriptor
NCCL INFO ras/rasnet.cc:365 -> 2
```

Measured 2026-09-24 on a GLM-5.3 TP8 restore: **304,269 of those lines in the
first 200 MB, 39.7 GB in about five minutes**, and the restored tree dead. criu
itself reported no error — `criu_restore` succeeded and the child logged
`rebind_log` — so nothing in the restore path names this. The only symptom is a
log that ate the disk.

**The lesson generalizes past NCCL.** The precondition in
`_close_loopback_listeners` was "the process group is down", and that was
satisfied. Being in the process group is not what makes a socket safe to close;
being *nobody's* is.

### RAS has a second listener, and the port-range fix did not cover it

**Restricting the close to `ip_local_port_range` was correct and insufficient.**
RAS holds *two* listeners per rank: the fixed `::1:28028` above, and an
**ephemeral** loopback one. The range check exempted the first and still closed
the second, so the spin survived the fix with a different call site. Measured
2026-09-25 on 35B TP=2, same config either side:

| signature | pre-fix (`69c1de6`) | post-fix (`e326442`) |
|---|---|---|
| `ras/client_support.cc:203 Call to accept failed` | 29,605 | **0** |
| `misc/socket.cc:458 socketTryAccept: Accept failed` | 30,310 | 40,768 |

The first row is the `28028` listener and the fix eliminated it outright. The
second is the ephemeral one and was untouched — on its own it is enough for
~143 MB/s and a restore that never leaves `reinit_nccl`.

### The predicate was inverted: at TP>1 there is nothing of torch's to close

The premise of the whole close — that torch orphans listeners the dump must
clean up — **is false at TP>1.** Measured by mapping every listener in the netns
to its owning process during init and again at dump time:

| | TP=1 | TP>1 (TP=2, 2 ranks) |
|---|---|---|
| ephemeral loopback listeners during init | 7 | 28 |
| still alive at dump time | 7 | **2** (1 per rank) |
| who owns the survivors | torch (`pt_tcpstore` + 14 × `pt_gloo_runloop`) | **NCCL RAS** |
| RAS present at all | **no** — no `::1:28028` anywhere | yes |

At TP>1 torch's own teardown retires 26 of the 28 before the dump prep runs, and
the close then takes the only two that were *not* torch's. At TP=1 the seven are
torch's entirely and NCCL runs no RAS subsystem for a single rank, so the close
there is both correct and worth doing.

Proved by difference, not inference: with `NCCL_RAS_ENABLE=0` and nothing else
changed, the gate reported `closed_listeners=[[], []]` — there was no socket to
close — and the restore ran clean through `rebind_graphs OK` with **zero** NCCL
warnings and a 60 KB restore-side log.

**So the close is a TP=1 tool.** `_prepare_worker_dump` (TP>1 only, gated on
`len(gpus) > 1`) censuses via `_loopback_listeners` and closes nothing;
the driver path in `prepare_criu_dump` (the TP=1 path) still closes. Two tests
hold that split in place, both asserted against the source because a behavioural
version would need a live TP>1 tree.

**This costs no collision exposure that was avoidable.** The TP>1 survivor is
RAS's and RAS needs it recorded, so it was always going into the image; closing
it never reduced the hazard, it only broke the restore. At TP=1 the seven ports
are still removed, which is where the reduction was real.

Fixed sockets are still reported as `skipped_fixed_port` in the TP=1 gate
diagnostics — there are none there today, and that line is what would say so if
RAS ever did appear on that path.

**The residual hazard is accepted, not solved.** Eight ranks each record
`::1:28028`, so two concurrent TP>1 restores in one network namespace would
collide on it deterministically. That is much narrower than it sounds: at TP=8 a
replica takes a whole node, so replicas cannot co-locate (verified 2026-09-24
with 2- and 4-replica jobs, which landed one per node). If it ever does need
solving, the lever is `NCCL_RAS_ENABLE=0` at dump time so the socket is never
created — not closing it underneath a running thread.

The `::` (v6 wildcard) `:8000` listener is deliberately **not** closed — the
filter matches only `127.0.0.1` and `::1`. A wildcard listener is reachable
from off-host and is not the orphaned-rendezvous shape being targeted; the
restored instance is meant to own 8000, so the record belongs in the image.

**But it carries a constraint that is worth stating.** Unlike the ephemeral
ports, 8000 is fixed, so its collision is deterministic rather than a coin
flip: anything holding 8000 in the restoring netns fails that restore every
time, and **two instances from one image cannot coexist in a single network
namespace.** For a system whose purpose is multiplexing, that is a real limit.
It is also the network-side view of the unguarded `num_replicas > 1` problem:
N replicas resolving to one cache key share a netns as well as a pid
namespace, so they collide on 8000 deterministically in addition to colliding
on recorded task ids. `::` with `v6only=0` is dual-stack, so it occupies 8000
in the IPv4 space too.

One address class is absent from our images by construction rather than by
luck. `init` pins `VLLM_HOST_IP=127.0.0.1` and `NCCL_SOCKET_IFNAME`/
`GLOO_SOCKET_IFNAME` to `lo` so images stay node-portable, and all three sit in
`_RESERVED_ENV`, so a job's `_env` cannot repoint them. Without that pinning
the same frameworks bind the routable NIC instead, and an address-keyed filter
would silently close nothing — so the pinning is load-bearing for this fix, not
just for `EADDRNOTAVAIL`.

**Open lead, not verified:** eight sockets sharing `::1:28028` with
`reuseport` unset should not be possible for independent binds, so they are
probably one inherited socket recorded per process rather than eight binds.
Worth confirming, and worth checking against the "one live restore per node"
constraint in `semi-p_DESIGN.md`, which currently has no named mechanism —
this fixed port is a candidate.

**Fix (dump side): close the orphaned listeners in `prepare_criu_dump`.**
Complication 1 already calls `destroy_process_group()` and waits for the
store threads to exit — but the FD keep-list preserves every `socket:` fd, so
the listeners those threads left behind were kept into the image. They are
already logically dead at that point. `prepare_criu_dump` now closes
`AF_INET`/`AF_INET6` sockets that are listening (`SO_ACCEPTCONN`) and bound to
`127.0.0.1` or `::1`, reporting them as `closed_listeners`. A skipped close
reports `would_close` instead, so an empty result cannot be mistaken for a
refused one.

### The gate: `is_initialized()` looked wrong and measured fine

**Measured 2026-09-24, and the concern below did not reproduce.** Both a dense
27B at TP=2 and GLM-5.3 at TP=8 (two coordinators, `tp` and `ep`) reported
`dist_initialized: False` on every rank at dump time, meaning the clean destroy
completed and the original `not is_initialized()` gate would have fired
unaided. The `_PG_ABORTED` condition described here is therefore **defensive,
not a fix** — it has never rescued a case. It is retained because it is an
`or`, so it can only widen when the close fires, and because the diagnostics
beside it make either outcome visible on every dump for free.

Keep the reasoning below for the next person who reads the same code and
reaches the same conclusion: it is a plausible failure mode that the runtime
does not exhibit, and re-deriving it costs a GLM cold start.

The flag is cleared only by the
no-argument `torch.distributed.destroy_process_group()`, which lives solely
inside vLLM's `destroy_distributed_environment()` — the *second* statement of
one `try:` block in `_destroy_nccl`:

```python
try:
    destroy_model_parallel()          # raises on an aborted comm
    destroy_distributed_environment() # never reached, so the flag never flips
except Exception:
    pass
```

`destroy_model_parallel()` walks seven coordinators, each calling
`destroy_process_group(subgroup)` on a communicator `_abort_torch_process_groups`
has just aborted. torch raises `ValueError("Invalid process group specified")`
for any pg already out of `_world.pg_map` — reachable whenever two coordinators
share one pg — and `pg.shutdown()` on an aborted NCCL comm is itself
version-dependent, as torch's own comment about `ncclCommAbort` once being
collective admits. `_force_dist_uninitialized_for_restore`'s docstring states
the consequence outright: after an abort-based teardown the image "can carry
`is_initialized()==True` with dead PGs". That function runs on the **restore**
side only, so nothing repairs the state before the dump.

**The abort is the proof instead.** `_destroy_nccl` sets the module-level
`_PG_ABORTED` immediately after `_abort_torch_process_groups()` and before the
clean destroy is attempted. The abort is unilateral, so once it returns the
rendezvous is dead regardless of what the bookkeeping did. Both conditions are
accepted (`dist_initialized is False or _PG_ABORTED`) because they fail in
opposite directions: a teardown that never reached `_destroy_nccl` flips
`is_initialized()` without setting the flag.

`_PG_ABORTED` is per-process and stays `False` in the driver, because
`_destroy_nccl` is a `collective_rpc` target that only ever runs in the workers.
It is also `False` at TP1, where `_destroy_nccl` returns before the abort. The
driver therefore keeps relying on its own `destroyed_pg`, which is sound there:
TP1 has no prior abort to make its `destroy_process_group()` raise.

**Thread death is *not* usable as the gate.** Gloo keeps its listening socket
for the life of the process (see the `GLOO_SOCKET_IFNAME` note in `init`), so
`pt_gloo_runloop` may never exit and a thread-liveness gate would fail exactly
like `is_initialized()` does. `_wait_store_threads_exit` therefore waits only on
`pt_tcpstore`/`pt_nccl_watchdg`/`pt_nccl_heartbt`, while `_live_store_threads`
*reports* `pt_gloo_runloop` as well. Folding gloo into the wait would convert a
fast poll into a guaranteed 2.5 s timeout and a warning on every dump.

**Nothing reuses the recorded ports — measured, not assumed.** A restored
process listens on seven ports found nowhere in its own image:

| | ports |
|---|---|
| recorded in image | 34817, 45383, 41455, 42513, 38435, 36607, 40745 |
| live after restore | 35527, 39217, 39447, 42751, 43707, 43967, 44025 |

Zero overlap, different fd numbers, and evenly spaced socket inodes — all
seven created in one burst by the `init_process_group` inside `reinit_nccl`.
Thread names on the live process (`pt_tcpstore`, `pt_gloo_runloop`,
`pt_nccl_watchdg`) confirm the owner. So criu was re-binding seven ports,
paying the full collision risk, only for the process to discard them.

**Image-bound, so a re-dump is required.** Unlike Complication 12, the window
does not drain. Images dumped before this change carry their listeners
forever and stay exposed until re-dumped and re-published.

**Restore-side diagnostic:** `_is_port_collision` + `_port_collision_report`
(`worker.py`), the inet analogue of `_pid_collision_report`. criu reports only
the error string and an in-image socket id, neither of which identifies the
occupant, so the report parses the port out of the excerpt and resolves the
holder through `/proc/net/tcp` → socket inode → `/proc/<pid>/fd` → `comm`.
Wired into **both** restore paths, and it says plainly that retrying will not
help and that a post-fix image should record no such port at all.

### Reading an image's listener census

**`crit` is not available in the device-manager container** — verified
2026-09-24: `crit: command not found`, no `pycriu` module, and the image
directory ships `files.img` with no `inetsk.img`. Where `crit` *is* available:

```bash
crit decode -i <image_dir>/files.img \
  | jq -c '.entries[] | select(.type=="INETSK") | .isk
           | select(.state==10) | {src_port, reuseaddr: .opts.reuseaddr}'
```

Otherwise use criu's own `dump.log`, which every image directory carries and
which needs no tooling at all:

```bash
D=/mnt/neutrino/base-models/image-cache/<key>/image
grep -aP 'inet:\s+Dumping:' $D/dump.log | grep TCP_LISTEN \
  | grep -oP 'src_addr \S+' | sort | uniq -c
```

**Match `Dumping:`, not `Collected:`.** The collect phase enumerates every
socket in the netns, co-tenants included, and will badly overcount. (It is
useful for a different question: the collect lines are how you see that the
dump-time netns also held foreign `::` wildcard listeners on ephemeral ports,
which is the collision mechanism from the other side.) Prefix each `fdinfo`
line's pid to attribute a listener to its owning process, and use criu's
`Seized task` lines for the process tree — the `core-*.img` files are
per-thread, so counting them badly overstates the process count (312 files for
9 processes at TP8).

Baseline for the three published keys, measured 2026-09-24, all of which
predate this fix:

| image | `127.0.0.1` | `::1` | `::` |
|---|---|---|---|
| Qwen3.8-27B TP1 (`552cc22540cd`) | 6 | 0 | 1 |
| Qwen3.6-35B-A3B TP1 (`77d95928d2ac`) | 7 | 0 | 1 |
| GLM-5.3 TP8 (`0b69f511e035`) | 8 | 8 | 1 |

The single `::` record is port 8000 in every case. **A post-fix image should
show zero `127.0.0.1` rows and keep the eight `::1:28028` rows and the one
`::`:8000.**

Getting that criterion wrong is how the NCCL RAS failure shipped. A GLM dump on
2026-09-24 produced an image with exactly *one* recorded listener, which was
read as an unusually clean result; the absent `::1:28028` rows were the bug, not
the cure.

**The census is not the verdict, whatever it says.** It proves the image no
longer demands a port. It cannot prove the process still works without one — a
socket closed out from under a live thread makes the census look *better* and
kills the restore. Only a dump/restore cycle settles that, and it costs one job
rather than a publish: the miss path dumps and then restores from the image it
just wrote (`_dump` followed by `_restore_with_port_retry` in `semip_engine.py`),
so a single cold-start job exercises both halves. The `closed_listeners`,
`skipped_fixed_port` and per-rank `gate=` lines then say *why*.

---

## Complication 16: Fabric-Backed Communicator Workspaces (vLLM 0.30)

**The CUDA checkpoint API cannot carry every kind of GPU allocation.** NVIDIA's
unsupported list is "IPC, UVM, RDMA, or fabric handle"
([cuda-checkpoint#14](https://github.com/NVIDIA/cuda-checkpoint/issues/14)), and
the driver "does not attempt to keep the process in a good state if an error is
encountered during checkpoint or restore". That last clause is why one bug wears
two faces: the same unsupported allocation stalls `cuCheckpointProcessCheckpoint`
on one run and fails `cuCheckpointProcessRestore` with `CUresult=801`
(`CUDA_ERROR_NOT_SUPPORTED`) on the next, from an identical payload.

vLLM 0.30 introduced one. `flashinfer_all_reduce.py` builds an all-reduce
norm-fusion workspace and logs

```
[flashinfer_all_reduce.py:216] Initialized FlashInfer Allreduce norm fusion
                               workspace with backend=mnnvl
```

MNNVL is multi-node NVLink; its workspaces are multicast objects over fabric
handles. vLLM 0.26 never built one, which is why this arrived with the upgrade
rather than as a regression in semi-p.

**It takes two factors, which is what made it hard to see.** Measured on one
image, one payload shape per model:

| config | workspace | TP=8 | `cuda_checkpoint` / `cuda_restore` |
|---|---|---|---|
| Qwen TP=1 | no | no | OK |
| Qwen TP=2/4 | **yes** | no | OK |
| 35B-A3B TP=8 | no | **yes** | **OK** (17.7 s / 12.6 s) |
| GLM-5.3 TP=8 | **yes** | **yes** | hang, or 801 |
| Flash-Next TP=8 | **yes** | **yes** | fails |

Neither factor alone breaks anything. The reading that fits is that MNNVL
multicast only becomes a true fabric object when the group spans the whole
NVSwitch domain; below that the workspace exists but is backed by something the
driver can carry.

**What decides whether the workspace exists at all** is not tensor-parallel
width but whether the allreduce-RMS fusion pass runs: it takes the quantized
pattern for fp8 models, so GLM (`quantization=fp8`) builds one where every
bf16 Qwen config does not. Before `9aaaa78` pinned `VLLM_ALLREDUCE_USE_FLASHINFER=0`
the Qwen runs built one too, for the other reason — FlashInfer was still a
selected all-reduce backend. Note that pinning that variable does **not**
suppress the workspace: it only feeds `max_token_num`, while `_get_or_create`
runs unconditionally.

**The fix is vLLM's own API, not an env pin.** 0.30 added
`checkpoint_prepare()` / `checkpoint_restore()` to the device communicator
(`cuda_communicator.py`), covering FlashInfer all-reduce and, via
`all2all_manager`, the MoE all-to-all buffers. FlashInfer implements them as a
**stable-VA detach** — the physical backing is released, the virtual address is
kept:

```python
# flashinfer/comm/allreduce.py
def checkpoint_prepare(self) -> None:
    """Detach physical backing; repeated successful calls are no-ops."""
    ...
    for handle in self.mem_handles:
        handle._unmap_and_release_handles()
    self.mem_handles[0].comm_backend.barrier()
```

Because the VA is preserved, nothing baked into a captured CUDA graph moves and
`ca_graph_rebind` has no extra work — the opposite of custom all-reduce, whose
`meta_ptrs` really do move and are the reason that rebind exists.

`_run_communicator_checkpoint_hook` drives both halves: `checkpoint_prepare` in
`_destroy_nccl` **before** the NCCL abort, and `checkpoint_restore` in
`_reinit_nccl` **after** `init_worker_distributed_environment` rebuilds the
groups. Two details are load-bearing because that trailing barrier is
collective. Targets come from `_communicator_checkpoint_targets`, which sorts by
group name rather than trusting `_groups` insertion order, since a rank visiting
a different set wedges instead of failing. And a raising hook is logged rather
than propagated: every rank runs identical code over an identical set, so a
raise is uniform and nobody reached the barrier, whereas one rank abandoning the
loop while the others continue is a wedge with no diagnostic.

**Two hypotheses tested and rejected**, recorded so they are not re-derived.
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` left total graph memory unchanged at 2.74 GiB,
*raised* the post-sleep residual from 6.03 to 6.71 GiB against 0.26's 5.22, and
produced the hang. And `gpu_memory_utilization: 0.80`, which handoff 27 ranked
first: `r11-glm-warm` and `r14-glm` are clean 0.26 arms at 0.9, so utilization
was never the variable.

**Still unexplained:** 0.30 runs two CUDA graph capture passes where 0.26 ran
one, for 2.74 GiB against 0.73 GiB, independent of breakable cudagraphs. It
accounts for the higher post-sleep residual but not for an 801, which is a
question of what kind of memory rather than how much. The two passes are `FULL`
and `PIECEWISE` under `cudagraph_mode: FULL_AND_PIECEWISE`, and while the extra
memory is still unaccounted for, the *two families* are what Complication 17 is
about.

**Correction, 2026-09-28:** the paragraph above on what decides whether the
workspace exists — the allreduce-RMS fusion pass taking the quantized pattern —
does not survive GLM's own engine config, which reports
`pass_config: {'fuse_allreduce_rms': False}` and `Using ['CUSTOM', 'PYNCCL']`
all-reduce backends, and builds the workspace anyway during CUDA graph capture.
The trigger is **not identified**. The 2×2 above is measured and stands; the
mechanism does not. Do not plan a cheap experiment around forcing a workspace
onto a model that does not build one — there is no known lever, and
`VLLM_ALLREDUCE_USE_FLASHINFER=1` is not it.

**The search space is closed, and it is a contradiction.** Read in a pod on the
same `vllm-0.30.0` wheel the dev image ships. `get_fi_ar_workspace` is the only
function that emits the `flashinfer_all_reduce.py:216` line, and exactly three
files in the vLLM tree reference it. On GLM-5.3 all three are excluded:

1. `compilation/passes/fusion/allreduce_rms_fusion.py:1050`, in
   `AllReduceFusionPass.__init__`. `pass_manager.py:176` builds that pass only
   `if pass_config.fuse_allreduce_rms` — and **semi-p sets that False itself**
   for TP>=2 (`vllm_child.py`, the `_pc["fuse_allreduce_rms"] = False`
   injection), which is why the log's first `pass_config` dump holds that one
   key alone. The field is `None`-until-resolved behind a skip-none validator,
   so an explicit False is not re-resolved by the optimization-level default.
2. `model_executor/layers/fused_allreduce_gemma_rms_norm.py:91`, in
   `_can_use_flashinfer` — a *runtime* lazy create gated only on tensor shape
   and dtype, with no env var and no fusion flag. Worth knowing about: it is a
   live creator for other models. Reachable only through
   `vllm/models/common/ops/fused_allreduce_rms_norm.py`, imported by
   `deepseek_v32`, `kimi_k3` and `minimax_m3`. GLM-5.3 is `vllm/models/glm5next`,
   whose only allreduce mention in the whole package is a comment.
3. `FlashInferAllReduce._ensure_workspace`, built at `cuda_communicator.py:121`
   only when `use_flashinfer_allreduce` and the group name starts with `tp`.
   `envs.py` parses it as `bool(int(os.getenv(...,"1")))`, so the `"0"` pin is
   honoured — no truthy-string bug. Independently, `cuda_communicator.py:277`
   appends `"FLASHINFER"` to the printed dispatch list whenever `fi_ar_comm`
   exists and is enabled, and GLM logs `Using ['CUSTOM', 'PYNCCL']` for `tp`.

So one premise is false: either something reaches the allocator under a name a
text search cannot see, or the worker's effective config differs from the one
the log printed. `install_fi_ar_workspace_probe` in `ca_graph_rebind` settles it
in one run — it wraps `_create_workspace` (not the public getter, which case 2
binds at import) for the capture window and prints the stack on first
allocation.

**A count caveat, corrected.** `logger.info_once` defaults to `scope="local"`,
which is `is_local_first_rank()`, so the "Initialized FlashInfer Allreduce" line
appears **once per node** however many ranks allocated a workspace. One
occurrence in a TP=8 log is the expected output, not evidence that seven were
lost. The probe fires per process and will not agree with that count.

**Correction, 2026-09-28 (second): driving both halves is not enough, because
the restore half cannot find what the prepare half detached.** vLLM matches a
workspace to the group that created it *by object identity*, and the lookup is
keyed on the communicator's own `cpu_group`:

```python
# cuda_communicator.py
def checkpoint_restore(self) -> None:
    checkpoint_restore_fi_ar_workspaces(self.cpu_group)
    if self.all2all_manager is not None:
        self.all2all_manager.checkpoint_restore()

# flashinfer_all_reduce.py
_fi_ar_workspace_groups: dict[int, ProcessGroup] = {}   # id(workspace) -> group
if workspace_group is group:                            # identity, not name
```

`_reinit_nccl` tears the distributed environment down and rebuilds it, so every
`ProcessGroup` on the restore side is a new object and `workspace_group is
group` is False for **every** target — including `tp:0`, which keeps its name.
The lookup returns an empty list, the loop body never runs, and the hook reports
`failed=none`. The workspace keeps its reserved VA with no physical backing and
the first graph replay faults with `cudaErrorIllegalAddress`.

The `ep:0` → `ep:1` rename that shows up in the hook output is a symptom of the
same rebuild, not the cause; aligning the names alone changes nothing.

`_restore_prepared_comm_state` closes it. `checkpoint_prepare`'s inventory —
the workspace objects and the `all2all_manager`, per group role — is stashed on
the worker (`_semip_ckpt_prepared`, strong references, riding the CRIU image the
way `_semip_rank_data_keep` does). On the way back each role is resolved through
`parallel_state`'s getters rather than through `_groups[name]`, and the two
resources are treated oppositely:

* the workspaces are **re-keyed** onto the new `cpu_group` and then left to the
  normal hook, which hands FlashInfer a live `TorchDistBackend(group=...)`. That
  parameter is the point: `checkpoint_prepare` barriers over the backend stored
  at creation, `checkpoint_restore` over whatever it is given, so the API is
  built to be restored against a group other than its creator's. Replaying the
  *old* communicators instead satisfies the identity check and then hangs — the
  re-map's allgather needs a group whose sockets still exist.
* the `all2all_manager` is restored **by hand**, after its `cpu_group` is
  pointed at the live group. The hook drives the rebuilt communicator, whose
  manager is a different object that was never prepared.

Re-keying assigns straight into `_fi_ar_workspace_groups`; vLLM's registration
path raises `"already associated with a different process group"` by design and
that guard is worth leaving intact. Reassign, never delete —
`_fi_ar_workspaces_for_group` raises on a workspace whose entry has gone.

**And count it.** Every step here is a loop over a possibly-empty set, and an
empty loop returns exactly what a full one does. `_assert_checkpoint_state_restored`
compares what `checkpoint_prepare` detached against what `checkpoint_restore`
saw and fails `reinit_nccl` on a mismatch, which is 60 s earlier and far more
legible than an illegal access in the first forward. It runs *after* the hook's
loop, never inside it: the barriers are in there, and a rank that raises early
strands every other rank. Zero prepared stays legal — a bf16 model builds no
workspace and 0.26 has no hooks — so the invariant is equality, not presence.

Do not use `grep -c "Initialized FlashInfer Allreduce"` as the acceptance test.
That message is `logger.info_once` on the *creation* path; `checkpoint_restore`
re-maps existing handles and never re-enters it, and the restore side never
re-runs whatever builds the workspace. A correct restore still prints zero.
Read the `workspaces=` and `all2all=` counts on the `[ckpt-hook]` lines instead.

### RESOLVED, 2026-09-29: the creator is named and the workspace is suppressed

Job `14f13e5a` (GLM-5.3 TP=8, `util 0.9`, `msl 327680`) completed the **full
round trip with zero failures** — the first time on 0.30. `reinit_nccl OK
5.356s` where every prior run had `FAILED`, and `rebind_graphs OK 6.882s`,
which had never executed with a complete graph set.

**Case 2 above was the answer, and the reason it was excluded was wrong.** The
stack, printed on all eight ranks:

```
gpu_worker.py:585                        determine_available_memory
  cudagraph_utils.py:921                   profile_cudagraph_memory
    model_runner.py:1028                     capture_model(profile_only=True)
      ...
deepseek_v2.py:1968                      forward
  models/deepseek_v32/nvidia/model.py:293  forward
    models/deepseek_v32/nvidia/model.py:158  fused_allreduce_rms_norm(
      common/ops/fused_allreduce_rms_norm.py:41  _can_use_flashinfer
        fused_allreduce_gemma_rms_norm.py:91       get_fi_ar_workspace(
```

`glm5next` does not import `deepseek_v32`. It **lands there at runtime**
through `deepseek_v2.forward`. The exclusion was built on an import search, and
an import search cannot see a dispatch. `fp8_ds_mla`, written off above as a
DeepSeek link that only looks plausible, was the tell: `quantization=fp8` is
what routes GLM into that layer family. Every bf16 Qwen config is exempt for
the same reason, which is the real content of the 2×2's "workspace" column.

**Why four probes in a row saw nothing.** The allocation happens inside
`determine_available_memory`, not `compile_or_warm_up_model` — a *memory
profiling* capture (`capture_model(profile_only=True)`) that runs earlier. Any
probe scoped to the warmup hook is structurally blind to it, and its silence
was repeatedly misread as "the allocation did not happen." Install from
`init_device` instead; `_semip_worker` does.

That also closes "Still unexplained: 0.30 runs two capture passes." The first
pass **is** the profiling capture. Both models do it, measured identical —
PIECEWISE, FULL, PIECEWISE, FULL under `cudagraph_mode: FULL_AND_PIECEWISE` —
so it is a 0.30-wide behaviour, not model-specific, and not a memory curiosity.
It is where the allocation was hiding. (The extra 2.74 GiB vs 0.73 GiB is still
unexplained as a *quantity*; it is not a correctness question.)

**The fix: never build it.** `SEMIP_SUPPRESS_FI_AR_WORKSPACE` defaults to `1`
for TP>=2 (`vllm_child.py`, next to the `VLLM_ALLREDUCE_USE_FLASHINFER` pin).
`get_fi_ar_workspace` returning `None` is a supported outcome, not a hack — it
is exactly what every caller gets on a GPU without NVSwitch multicast, and
`_can_use_flashinfer` falls back to the unfused path. `setdefault`, not
assignment: put `"SEMIP_SUPPRESS_FI_AR_WORKSPACE": "0"` in the payload's
`extra_env` to get the old behaviour back, which is what a future driver or
FlashInfer able to rebuild multicast would want.

**What happens to the re-key machinery.** Everything from
`_restore_prepared_comm_state` down is now *dormant, not dead*. It is correct
and it was measured working (`rekey: workspaces=1 problems=none` on all eight
ranks of job `6e386668`); it simply has nothing to re-key once suppression is
on. Keep it, for three reasons:

* It is the whole of the fallback if suppression is ever turned off, and the
  escape hatch is deliberate.
* `_assert_checkpoint_state_restored`'s invariant is equality, not presence, so
  it stays correct at zero and keeps guarding the non-suppressed path.
* Most usefully, the counters **invert into a regression detector**. With
  suppression on, `checkpoint_prepare: ... workspaces=0` is the expected line.
  A non-zero count now means suppression failed — a new creator, a renamed
  getter, an env that did not propagate — and it says so 60 s before an illegal
  access would. Read `workspaces=` as an assertion that the fix is still in
  force, not as a measure of work done.

**The multicast headline was never the thing to fix.** `cuMulticastAddDevice`
returning `CUDA_ERROR_INVALID_DEVICE` after `cuCheckpointProcessRestore` is
real and reproduced (torch's `CUDASymmetricMemory` hits it independently in the
same process, and degrades gracefully where FlashInfer cannot). But it is a
wall you only reach by having built a workspace you did not need. If the NVIDIA
report is filed, note that the semi-p-side device evidence for it was never
actually collected: the probe meant to query `MULTICAST_SUPPORTED` and
`HANDLE_TYPE_FABRIC_SUPPORTED` on the restored process returned `probe={}` for
one bug and then reported the query's *status* rather than its *value* for
another. Both are fixed; neither has been read on a failing run.

---

## Complication 17: Graph Discovery Completeness

**Read with Complication 16 — this is the failure that surfaced once 16's fix
let a TP=8 restore get far enough to fault.**

After `reinit_nccl`, `ca_graph_rebind` rewrites the CustomAllreduce addresses
baked into every captured CUDA graph. It finds those graphs from two places:

1. the **wrapper registries**, `CUDAGraphWrapper._all_instances` and
   `BreakableCUDAGraphWrapper._all_instances`, walking each wrapper's
   `concrete_cudagraph_entries` / `entries`; and
2. the **graph manager**, `model_runner.cudagraph_manager.graphs`.

**Neither source can be assumed to cover its half, and a source that silently
contributes nothing is indistinguishable from one that had nothing to give.**
Measured on GLM-5.3 TP=8 under vLLM 0.30:

```
n_wrappers: 1, n_entries: 51, n_manager_graphs: 51, n_cudagraph_objs: 51
```

`n_cudagraph_objs` is the total and `n_manager_graphs` counts only graphs the
manager added that source 1 had not, so both being 51 means **51 wrapper
entries yielded zero graphs**. The scan was `vars(entry).values()`, and 0.30 had
put the `torch.cuda.CUDAGraph` where **no** attribute walk reaches it — see the
correction at the end of this complication, which retracts the "moved off
`__dict__`" reading. Against 35B-A3B TP=4, which restores cleanly:

```
n_wrappers: 41, n_entries: 2091, n_manager_graphs: 51, n_cudagraph_objs: 2142
```

2091 + 51: both sources contributing, 41 wrappers × 51 capture sizes.

**The failure is asynchronous and lands nowhere near the cause.** The half that
was found is patched correctly; the half that was not keeps its pre-checkpoint
VAs and faults on the first replay:

```
_drive_warmup_ladder("verify", ...) -> engine.step() -> copy_event.synchronize()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Every existing check agreed the rebind was fine**, because every existing
check ranges over the graphs that were discovered. That run reported
`rebind rank=N ok=True` on all eight ranks with `kernel_nodes_patched: 6759`,
`kernel_slots_rewritten: 60831`, `topo_readback ok: True` and
`stale_ca_audit bad: 0`. All true, all about the wrong set. This is the same
shape as the note in that module about E4's `pre_valid_post_invalid=0` being
"true and vacuous" — an inventory taken with the enumerator that just did the
patching cannot testify that the patching was complete.

**Two defences, both in `_find_captured_graphs`:**

- `_entry_cuda_graphs` is shape-agnostic about where the entry keeps its graph:
  a bounded, cycle-guarded walk over `__dict__`, `__slots__` and nested
  containers, with a second pass that reads non-callable descriptors only if
  the first found nothing. The attribute has moved more than once and will
  move again.
- `_discovery_verdict` sets `complete` / `incomplete_why` in the diag, an
  entries-versus-graphs-yielded check that does not depend on knowing which
  source *should* have supplied what. `rewrite_addrs_in_graphs` and
  `_force_recommit_ca_nodes` — the two ways out of the rebind — both refuse to
  report success without it. The gc-unfreeze fallback now also fires on a
  shortfall rather than only on a completely barren scan, which is why it sat
  out this failure: the manager's 51 graphs made the result non-empty.

**The V1/V2 runner split is not a stable property of a model.** The
`_graph_manager_holders` docstring used to list GLM among the V1 models, where
`n_managers == 0` is expected and source 1 covers the FULL graphs. On 0.30
GLM-5.3 reports `n_managers: 1`. `runner_class` does not disambiguate either —
both runners are named `GPUModelRunner` and differ only by module. Rely on the
completeness verdict, not on the allowlist.

**Symptoms that should send you here:** `rebind_graphs FAILED` with an
`illegal memory access` inside the post-restore warmup; any `ok=True` rebind
whose `graph_discovery` shows `n_entries` far larger than the graphs actually
patched; a new vLLM version where `n_cudagraph_objs` drops sharply against the
previous one for the same model and TP width.

**Correction, 2026-09-28: it was never on `__dict__`, and the deeper walk is not
what fixed it.** `CUDAGraphEntry` is a plain dataclass with `cudagraph` as an
ordinary attribute, so `vars()` would have found it. The entries that yielded
nothing were `_BreakableEntry`, and 0.30 keeps their graphs behind **bound
methods**:

```python
# vllm/compilation/breakable_cudagraph.py, _end_segment
self.segments.append(self._current_graph.replay)   # a BOUND METHOD
self._current_graph = None                          # last direct ref dropped
```

`segments` is a `list[Callable]`; the only path to the graph is
`segments[i].__self__`, and `isinstance(bound_method, CUDAGraph)` is False. **No
isinstance-based walk of any depth reaches them** — not the old `vars(entry)`
scan and not `_entry_cuda_graphs`, which still yields 0 on this model. What
recovered the 4029 hidden graphs was the **gc-unfreeze fallback firing on an
incomplete verdict**, and that is the mechanism to protect. Keep the walk — it
costs nothing and covers layouts that *are* attribute-reachable — but do not
mistake it for the fix.

**`_collect_graph_entries` had the same blindness and now shares the same
fallback.** It backs `graph_exec_census`, which the dump-side `COLD IMAGE` check
reads, so while discovery saw 4080 the census was still reporting `exec_ok=51
shapes=51` — enough to call an image warm on 1.2% of the evidence. Recovered
graphs arrive with no entry and therefore no shape, so they are bucketed under
an explicit `<gc-recovered>` key rather than folded into a real one, which would
claim thousands of captures at a batch size that never ran.

---

## Summary Table

| Resource           | Problem at dump time               | Dump-side fix                      | Restore-side fix                     |
|--------------------|-------------------------------------|------------------------------------|--------------------------------------|
| NCCL/TCPStore      | Background threads, TCP sockets; the tree-kill's FINs park the recorded local ports in `TIME_WAIT` | `destroy_process_group()` + poll; `SO_LINGER(1,0)` on every inet TCP socket so the kill RSTs instead | `--tcp-close` — but it closes only *after* the rebind, so it is not sufficient alone |
| io_uring           | Non-serializable kernel state       | Munmap rings (FDs kept via keep-list) | —                                 |
| POSIX semaphores   | `(deleted)` files → ghost/link remap| Delete sem files; captured as anon | —                                    |
| stdout/stderr      | The reader of the pod-log pipe is outside the tree | Record `stdout_resource` (`pipe:[N]`) | `--inherit-fd fd[L]:<stdout_resource>` with this pod's log pipe |
| Pipe FD            | sudo closes FDs >= 3                | —                                  | SCM_RIGHTS via Unix socket           |
| CUDA context       | GPU state not in CRIU image         | CRIU CUDA plugin at dump           | Driver API or cuda-checkpoint        |
| Fabric-backed comm workspaces | vLLM 0.30's FlashInfer all-reduce workspace (`backend=mnnvl`) and the MoE all2all buffers are multicast objects over fabric handles, which `cuCheckpointProcess*` cannot carry. Needs **both** the workspace and TP=8 to bite: `cuda_checkpoint` stalls indefinitely, or `cuda_restore` returns `CUresult=801`. **Built by `fused_allreduce_rms_norm` on the DeepSeek-V3.2 layer path**, which fp8 models reach at runtime through `deepseek_v2.forward` — during `determine_available_memory`'s `capture_model(profile_only=True)`, not the warmup capture. bf16 Qwen never enters that path, which is why it is exempt at any width. `VLLM_ALLREDUCE_USE_FLASHINFER=0` does **not** prevent it; that flag only gates the communicator | **Prevented, not detached.** `SEMIP_SUPPRESS_FI_AR_WORKSPACE` defaults to `1` for TP>=2, so `get_fi_ar_workspace` returns `None` and no workspace is ever built — a supported outcome that `_can_use_flashinfer` handles by falling back to the unfused path. `_run_communicator_checkpoint_hook(ps, "checkpoint_prepare", worker)` still runs in `_destroy_nccl` and now reports `workspaces=0`; a non-zero count means suppression failed (Complication 16) | Nothing to re-map. `_restore_prepared_comm_state` and the re-key are **dormant, not dead** — correct, tested, and the fallback if suppression is turned off; `_assert_checkpoint_state_restored`'s invariant is equality, so it stays valid at zero. Before suppression the restore died here: `cuMulticastAddDevice` returns `CUDA_ERROR_INVALID_DEVICE` in a restored process, so a multicast-backed workspace cannot be rebuilt at all |
| Captured CUDA graphs | The rebind can only patch graphs it finds, and both discovery sources have moved under us: on 0.30 GLM-5.3's 51 wrapper entries yielded 0 graphs, leaving the manager's 51 as the whole set of 4080. The rest kept pre-checkpoint VAs; `topo_readback` and `stale_ca_audit` both ranged over the found 1.2% and reported clean, and the restore died on `cudaErrorIllegalAddress` in the first `engine.step()` | — (discovery runs on the restore side) | `_discovery_verdict` sets `complete`, and an incomplete verdict triggers the gc-unfreeze heap scan — which is what actually recovers 0.30's piecewise segments, since they sit behind bound methods no attribute walk can reach. Both exits of the rebind, and `_collect_graph_entries` behind the census, refuse to report over a partial set (Complication 17) |
| Plugin directory   | `--libdir` path missing             | `_worker_criu_save` mints a per-dump `tempfile.mkdtemp`, removed in a `finally` | — (no `--libdir`: loads the real `cuda_plugin.so`) |
| stdin / tty        | pts captured as `--shell-job`; can't reattach in a PID ns | fd 0 → `/dev/null` + `setsid()` at child start | drop `--shell-job` |
| PID collisions     | Recorded task ids (leaders *and* threads) already taken — by a live task, or with no task at all by a `PGID`/`SID` reference an unreaped zombie still holds: dump leftovers, concurrent restore, unrelated service, the restore's own launcher, or a stray the pod's non-reaping PID 1 kept | `_set_child_subreaper` at worker start + `_reap_orphaned_descendants` after the dump, so the leader's group/session members are collected rather than left to PID 1; a child-side `waitpid` sweep in `prepare_criu_dump` for strays already dead | Restore each tree in its own PID namespace (reaper + private /proc); on the lowcap path a preflight that names both kinds of occupant + `scripts/pidcheck.py`, with dump-side id placement as the durable fix; retry loop as backstop |
| Privileged-only CRIU | dump + restore need `CAP_SYS_ADMIN` (netns kerndat probe; restore PID namespace) | `--unprivileged` (`SEMIP_UNPRIVILEGED=1`) skips the netns probe; caps shed in the child before `import torch` | lowcap path: `criu restore -d --unprivileged` in the host PID ns (no `unshare`), with a snapshot-verified teardown kill (Complication 13) |
| Cross-uid dump | A root `criu` cannot dump an unprivileged child: `ptrace_may_access` needs `CAP_SYS_PTRACE` and `prlimit()` needs `CAP_SYS_RESOURCE`, the latter outside a cap-prod bounding set (`cr-dump.c:389`) | `_worker_criu_save` runs `criu` at the worker's own uid (no `sudo`), so it matches its own child by construction; unprivileged workers need `setcap cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip /usr/sbin/criu` | The last two are for the restore, not the dump |
| Cross-uid restore | The restored child keeps the image's uid, so a parent of a different uid collides with it on `<model_dir>/compilation` and `image/`; no file mode reconciles them | Out of scope: only `root -> root` and `unprivileged -> unprivileged` are supported. `meta.json` records the dumping `uid`; `Instance.criu_restore` raises on a mismatch before spawning a worker | Pre-`uid` images are unchecked — re-dump if unsure |
| Pipe capacity | `F_SETPIPE_SZ` needs `CAP_SYS_RESOURCE` once the *creating uid* is over `fs.pipe-user-pages-soft`; the budget is per uid, and uid 0 pools every root-run tree on the node (`criu/pipes.c:165`) | `_worker_criu_load_lowcap` restores at the worker's own uid (no `sudo`), which is also the image's uid | Killing stale root trees frees the budget, but only until the next root run |
| Ghost remap race   | `(deleted)` .so → ghost race        | Destructive dump + sem deletion    | `--link-remap`                       |
| Teardown scoping   | No namespace to collapse on the lowcap path, and `-d` reparents tasks off the root; a `kill -9 -<pid>` meant as a group kill becomes `kill(-1)` in procps-ng | — | Snapshot task ids + start times at restore (`_tree_identity`); kill only verified positive pids, never a negative one |
| Recorded ephemeral listeners | **At TP=1 only.** torch's TCPStore/Gloo listeners sit on kernel-assigned ephemeral ports and survive `destroy_process_group()` as fds, so the keep-list preserves all 7 into the image; criu rebinds each on every restore and a live occupant fails the whole thing. `SO_REUSEADDR` does not help against a live `LISTEN`, and retrying never clears it. **At TP>1 it does not arise** — torch retires its own before the dump (28 → 2 measured across 2 ranks) and both survivors are NCCL RAS's | `_close_loopback_listeners` closes listening loopback `AF_INET`/`AF_INET6` sockets whose port is inside `ip_local_port_range`, reached **only from the TP=1 driver path** in `prepare_criu_dump`, where the 7 are torch's and NCCL starts no RAS subsystem for a single rank. `_prepare_worker_dump` (TP>1) censuses via `_loopback_listeners` and closes nothing: RAS holds one fixed (`::1:28028`) **and** one ephemeral listener per rank, both behind a thread no teardown stops, and closing either spins `accept()` on `EBADF` after restore at ~143 MB/s while the job reports `RUNNING`. Nothing reuses the TP=1 ports, since `reinit_nccl` rebuilds the rendezvous on fresh ones. **Re-dump required** | `_is_port_collision` + `_port_collision_report` name the holding pid and state that a retry cannot help (Complication 15) |
| Bounding-set trim | `restore_creds()` drops every capability absent from the recorded `cap_bnd` without checking whether it is already absent, and the kernel tests `CAP_SETPCAP` first — so ~39 `EPERM`s fire even restoring into the pod that produced the image, and the `BUG()` that follows **hangs** instead of returning | `prctl(PR_SET_NO_NEW_PRIVS, 1)` in `_drop_caps_for_portable_image`, before `import torch`, which is criu's own licence to demote the `EPERM` to a warning. Re-dump required | — (`setcap cap_setpcap` cannot work: outside the bounding set, it makes `criu` fail to `execve`) |

---

## CRIU-Related Commit History

Chronological summary of CRIU plumbing changes across the branch.

| Commit    | Date       | Summary |
|-----------|------------|---------|
| `586641e` | 2026-03-27 | **Initial semi_persistence package.** GPU sleep/wake lifecycle, `cuda-checkpoint` CLI integration. No CRIU yet. |
| `e8098d6` | 2026-03-31 | **Cross-GPU migration.** Add instance migration with `cuda-checkpoint --device-map`. |
| `b0524b7` | 2026-03-31 | **Orchestrator + multiplexing demo.** Multi-model orchestration, first end-to-end save/load with CRIU dump/restore. |
| `9a04880` | 2026-04-06 | **CRIU save/load, driver API, cross-GPU migration.** Replace `cuda-checkpoint` CLI with `libcuda.so` driver API (`cuCheckpointProcess*`). Add `--link-remap`, `--ext-unix-sk`, `--shell-job`, `--tcp-close`. Add pipe FD passing via SCM_RIGHTS. Add `/dev/shm/sem.*` cleanup pre-dump. Add `--leave-running` for non-destructive save. |
| `fc3a3db` | 2026-04-07 | **State machine ladder.** `saved ↔ checkpoint ↔ up` lifecycle; CRIU load integrated into orchestrator state transitions. |
| `ad31cca` | 2026-04-17 | **CRIU v4.2 build instructions.** Added build-from-source instructions for CRIU 4.2 with CUDA plugin support. |
| `ccaaf81` | 2026-04-20 | **CRIU restore robustness.** Init-time cleanup of stale ghost files and `/dev/shm` leftovers. PID collision retry loop (5 attempts). Orphan process killing on failed restores. Pipe error handling in `_child_thread`. Process settle wait before CUDA restore. `CUresult=401` tolerance for redundant CUDA restores. |
| `df7d82c` | 2026-04-21 | **Destructive dump + ghost remap fixes.** Switch from `--leave-running` to destructive dump (child killed after save). Delete `/dev/shm/sem.*` in `prepare_criu_dump` so CRIU captures semaphores as anonymous memory. Remove `_recover_deleted_mappings()`, `_clear_ghost_files()`, and debug instrumentation. Clean up orchestrator init. All models must be re-dumped. |
| `e0ec41a` | 2026-09-16 | **First dump reap — broken.** Called `_reap_dumped_child` from `_child_thread` where `child_proc` was not in scope, so every dump died with `NameError` 1ms after the image was written, turning a good dump into `criu_dump FAILED`. A static undefined-name check would have caught it in seconds. |
| `e0df4f8` | 2026-09-16 | **Dump reap fixed and made non-fatal.** Passes the `Process` handle through and moves the call out of the handler's `try`. Correct code, but it collects only the leader's own corpse and so does not address the pgid/sid reference — the restore still failed `Can't fork for 1233`. |
| `7f75d33` | 2026-09-17 | **Reap the leader's process group, not just the leader.** `_set_child_subreaper` (`PR_SET_CHILD_SUBREAPER` at worker start) so orphaned grandchildren become reapable, `_reap_orphaned_descendants` sweeping after the dump and at teardown, and a child-side stray sweep in `prepare_criu_dump`. Collision reporting in `_pid_collision_report` and `scripts/pidcheck.py` gains `pgrp`/`session` scanning; `scripts/pidref_test.py` added. First successful semi-p job on the dev cluster (`2469a638`). |
