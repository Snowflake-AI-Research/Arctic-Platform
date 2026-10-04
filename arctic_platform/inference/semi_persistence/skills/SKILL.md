---
name: semi-persistence
description: >-
  Orientation map for the semi_persistence subsystem: CRIU-based checkpoint and
  restore of whole vLLM instances so many models share a small GPU pool without
  cold starts. Covers the process hierarchy (Instance handle -> worker -> vLLM
  child), the saved/checkpoint/sleep/up state ladder, the Slots buddy allocator,
  the per-model Op pipeline, the orchestrator/client split, and the CRIU
  complications. Use when debugging checkpoint or restore failures, tracing a
  model state transition, reasoning about GPU slot allocation or eviction, or
  locating which file owns a given behaviour.
disable-model-invocation: true
---

# semi_persistence: checkpoint/restore multiplexing for vLLM

This subsystem lets a vLLM engine be **checkpointed to CPU and restored onto a
GPU in a second or two**, instead of paying a ~1 minute cold start. That makes
it possible to keep hundreds of models resident and multiplex them over a
handful of GPUs. Per-subsystem detail is in [reference.md](reference.md).

> This is **not** a KV-cache offload feature. Checkpointing saves the whole CUDA
> context (via `cuCheckpointProcess`) and leaves a 0 GB trace on the GPU; the
> on-disk form is a CRIU image of the entire child process tree.

## The one thing to know first

**Import the public names off the package; never import a submodule by its
qualified path.** Every module imports its siblings flat (`import
semip_logging`, `from instance import Instance`), so the package directory
itself has to be on `sys.path` before any of them can load. The package's lazy
`__getattr__` installs that entry on first *attribute* access, which makes the
first two forms below work and the third fail:

```python
from arctic_platform.inference.semi_persistence import Instance   # works
import arctic_platform.inference.semi_persistence                 # works (nothing loaded yet)
import arctic_platform.inference.semi_persistence.instance        # ModuleNotFoundError: semip_logging
```

The third fails because it executes `instance.py` before any attribute access
has installed the `sys.path` entry. `Instance`, `Orchestrator` and
`OrchestratorClient` are the exported names; the class you get back is
`instance.Instance`.

Only the *first* import in a process is decisive, which is what makes this look
flaky. Put line 1 ahead of line 3 and line 3 then succeeds, because the entry
is already on `sys.path` by the time it runs. Test the claim in a fresh
interpreter, never in a REPL you have already imported into.

Code that reaches past the package still needs the directory on `sys.path`
itself, which is why `tests/` and `scripts/` each begin with a
`sys.path.insert(0, <package dir>)` bootstrap. Anything you add there must do
the same.

## Layer map

Read bottom-up; each layer only knows about the one below it.

| Layer | File | Responsibility |
|---|---|---|
| Child | `vllm_child.py` | Owns the GPU. vLLM engine, pinned CPU buffer, async generate loop, CRIU dump prep |
| Worker | `worker.py` | Command loop; `cuCheckpointProcess` via ctypes; CRIU save/load |
| Handle | `instance.py` | `Instance`: chainable non-blocking primitives, one per vLLM config |
| Demux | `demuxer.py` | Sole consumer of the result queue; keeps `Instance` state fresh |
| Slots | `slots.py` | Buddy allocator handing out fractional GPU slots (metadata only) |
| Pipeline | `pipeline.py` | One worker thread + FIFO queue per model, ops as `Op` subclasses |
| Orchestrator | `orchestrator.py` | `model_id` -> `Instance`, the state ladder, eviction policy |
| Serving | `orch_server.py`, `client.py` | HTTP front end and the job-keyed client |
| Observability | `state_server.py`, `dashboard.py` | `/state` endpoint, curses dashboard |
| Support | `abstract.py`, `semip_logging.py` | `InstanceBase` interface, logging |

## Process hierarchy

Both the worker and the vLLM child are **spawned**, never forked — fork can
deadlock on glibc mutexes held by other threads, and spawn gives the child a
clean address space with no inherited CUDA context.

```
Main process
  |-- Instance (handle)
        `-- Worker process (spawn)      <- cuCheckpointProcess ctypes, CRIU
              `-- vLLM child (spawn)    <- owns GPU + pinned CPU buffer
                    `-- EngineCore (in-process, VLLM_ENABLE_V1_MULTIPROCESSING=0)
```

`EngineCore` runs in-process deliberately: as a subprocess it would pickle GPU
tensors across the boundary during `restore_weights`, which fails for models
over 4 GiB and costs ~16 s even for small ones. GPU memory queries use NVML
rather than `torch.cuda.mem_get_info`, to avoid initializing CUDA in the main
process.

## The state ladder

`move()` walks this ladder one rung at a time, in either direction:

```
saved  <->  checkpoint  <->  sleep  <->  up  ( -> running, transient)
```

| State | Image on disk | Live process | Slot held | CUDA context | Weights on GPU |
|---|---|---|---|---|---|
| `saved` | yes | no | no | no | no |
| `checkpoint` | yes | yes | no | no | no |
| `sleep` | yes | yes | usually | yes (small) | no |
| `up` | yes | yes | usually | yes | yes |
| `running` | yes | yes | **always** | yes | yes |

Two states are transient and are **not** valid `move()` targets: `running`
(exists only during a `generate()`), and `wait` (published while blocked in the
`Slots` FIFO during `checkpoint -> sleep`; it is purely a dashboard signal).

After a generate finishes, a model parks in **slotless `up`** — still warm, but
its slot is released, which is what makes it eligible for eviction.

## Slots in one paragraph

`Slots` is a singleton buddy allocator over GPUs, and pure metadata — it does no
CUDA work. A `Slot` is an immutable `(gpu_id, level, index)`; a level-`L` slot
covers `1 / 2^(L-1)` of a GPU, so level 1 is a whole GPU, level 2 a half, level
3 a quarter. A model's level derives from its `gpu_memory_utilization`. Nodes
split on allocation when no exact-size slot is free, and coalesce with their
buddy (`index ^ 1`) on deallocation. Waiters queue FIFO; free GPUs are picked
coldest-first by last-used time.

## Pipeline in one paragraph

Each `model_id` gets one worker thread and one FIFO queue. Operations are `Op`
subclasses — `RegisterOp`, `MoveOp`, `EvictForPeerOp`, `GenerateOp`, `PauseOp`,
`ResumeOp`, `RemoveOp` — and the FIFO is what orders them, replacing an older
implicit future-chaining design. Pause works by `interrupt_now()` plus
`submit_front()`, and long-running ops cooperate via `InterruptFlag`
(`raise_if_set()` / `wait_or_interrupt()`) instead of polling loops.
Cross-model eviction goes through `submit_to_peer_and_wait`, which carries
cycle detection.

## Where things live

```
semi_persistence/
  *.py           library code (orchestrator, instance, worker, client, dashboard, ...)
  tests/         pytest: CPU-only, hermetic, ~2.5s, no GPU
  scripts/       imperative repros: need real GPUs and real vLLM
                 (except imgdiff.py / pidcheck.py, which only need crit + root)
  reproduce/     example_full.py: the saved -> up sweep across TP1/2/4/8
  skills/        this skill (SKILL.md + reference.md) and every design doc
```

Running things:

```bash
cd arctic_inference/semi_persistence
python -m pytest tests/ -q        # 35 tests, no GPU needed
python scripts/test_env.py 0 1    # needs two real GPUs
python dashboard.py               # needs a running orchestrator on :8157
```

`tests/` is the only part runnable in CI. Everything in `scripts/` allocates
real GPUs and loads real weights, except `scripts/imgdiff.py` and
`scripts/pidcheck.py`, which inspect a CRIU image on disk and need neither.

## Gotchas that bite

- **CRIU dump is destructive.** The child is killed once the image is written,
  so after `criu_dump` the model is always `saved` with no live process.
- **A serving image's directory is derived, and the name is the cache key.**
  `restore_and_wrap` computes
  `$SEMIP_IMAGE_CACHE/<sha256(vllm_config [+ device nodes])>_<sha256(image
  digest + driver)>[/replica<K>]` before it knows whether an image exists;
  there is no `semi_p_model_dir` job field, and `SEMIP_IMAGE_CACHE` defaults to
  `/data-fast/image-cache_neutrino`. Both terms are exact identifiers rather than
  samples of the filesystem, which is what makes a miss productive: a collision
  would have nowhere else to put the new image, so the cold start would
  overwrite the image it just rejected and two pods in that state would
  overwrite each other forever. The driver is in the key because it is
  bind-mounted from the host rather than shipped in the image, so the image
  digest alone does not fix it. See `IMAGE_CACHE.md` Section 6.
- **At TP>1 the pod's device allocation is in the key; at TP=1 only with
  several replicas per pod.** A TP>1 image can only be restored where every
  device node it captured can be reopened, so `_device_binding` folds the
  sorted `/dev` set into `cfg12` and "found" becomes the same statement as
  "restorable". A single TP=1 image renumbers onto whatever slot it lands on,
  so binding it would split one universally restorable image into one key per
  GPU. TP=1 with several replicas per pod binds anyway, only so that a 1-GPU
  pod (flat) and an 8-GPU pod (`replica0..7/`) never share a directory;
  `_missing_device_nodes` stays TP>=2 only, because a TP=1 image still
  restores on any GPU. The set is **sorted**,
  because a restore onto the same devices in a different order works
  (`[3, 1, 2, 0]` was measured coming back on `[3, 2, 0, 1]`); hashing the order
  would split an image into up to `TP!` keys. The gain is that dumps on
  different slots now accumulate rather than overwriting one another; the cost
  is that an `env12` bump invalidates every slot's image at once.
- **A miss can be filled from the published mirror, and with one replica per
  pod that never fails a job.** `_materialize_from_source` copies `image/` and
  `compilation/` (bound to the absolute `model_dir`; 6.7 GB at TP=1, 29 GiB at
  TP=4, 207 GiB for Flash-Next at TP=8) out of the read-only mirror
  (`SEMIP_IMAGE_SOURCE`, default `/mnt/neutrino/base-models/image-cache`; `""`
  turns it off) and reads `weight/` (no baked path) in place. Every way it
  declines — an unverified directory, a foreign `model_dir` or uid, a publish
  that withheld the weights, a recorded size mismatch, any I/O failure — is a
  cache miss that cold-starts, so the checks are strict on purpose. See
  `IMAGE_CACHE.md` Section 9.
- **With several replicas per pod it is all-or-none.** A replica that
  cold-starts beside siblings that restore lands its new processes on the task
  ids their images recorded. So the checks every replica answers identically
  stay misses (the whole pod cold-starts), and anything that can fail one
  replica alone (its `replica<K>/` missing, its copy failing) fails the job.
- **Several replicas per pod: per-node slots, isolated dumps, publish-time
  assembly.** `ReplicaPool._node_slots` numbers replicas *within each node*
  from each actor's Ray node ID and passes `SEMIP_REPLICA_ID` (slot) and
  `SEMIP_NUM_REPLICAS` (replicas on that node). With more than one,
  `model_dir` is `<key>/replica<K>`; each replica dumps alone under its own
  lock, `image/`, `compilation/` and `weight/`, and **nothing about the replica
  set is in `meta.json`**. `semip_publish.py <root>/<key>` reconstructs the set
  from the directories, requires a contiguous `replica0..N-1`, refuses unless
  every replica's weights hash the same, uploads one weight copy and writes one
  sentinel. Because slots restart at 0 on every node, an `n_gpus=16` job
  restores `replica0`/`replica1` of a single-node TP=4 dump on each node
  (measured 2026-10-03). One replica per pod (TP=8 at `n_gpus=8`) keeps the
  flat layout. See `IMAGE_CACHE.md` Sections 1 and 11.
- **Never let an `n_gpus=16`/`32` job be the one that dumps.** Dump the same
  spec at `n_gpus=8`, publish it, wait for `--status --wait-verified` to show
  every node, and only then submit the multi-node job. `n_gpus` is not in the
  key, so the larger job resolves the same skeleton. Nothing enforces this: a
  replica sees only its own node, so on a miss every node cold-starts and
  dumps its own identical local copy, and a node the skeleton is not yet
  verified on cold-starts while the others restore.
- **Replicas share `/dev/shm`, so the dump unlinks only its own files.** The
  pre-dump unlink of `sem.*` (child) and `psm_*` (ranks) used to glob the pod,
  deleting segments a live sibling still needed. It is now scoped to the
  tree's own mappings, **matched by inode**, because glibc's `sem_open` leaves
  `maps` showing a deleted temporary name while the real `sem.loky-…` name
  stays linked. Miss that, and CRIU link-remaps the file into
  `/dev/shm/link_remap.<id>`, where concurrent replicas collide (`Can't link
  remap … File exists`, four of six dump jobs in round 1). See CRIU_PLUMBING
  Complication 3.
- **Killing the tree does not free the ids it held.** Dead is not reaped, and
  an unreaped task holds pids two ways: its own task id, and the id naming its
  process group and session. The child `setsid()`s, so that second id is the
  dumped leader's, and *every* process the child forked holds a reference on
  it. Those are the worker's grandchildren, invisible to `waitpid` until
  `PR_SET_CHILD_SUBREAPER` adopts them — so where PID 1 does not reap (a DSS
  zone pod runs `dss-zone-worker`, which never `wait()`s) they pin the leader's
  id for the life of the pod, and the restore fails `Can't fork for <pid>: File
  exists` on an id `/proc` shows as free. The worker sets the subreaper flag at
  startup and sweeps after the dump; the child sweeps its own strays in
  `prepare_criu_dump`. No retry budget can substitute: the holders are
  permanent, not transient. See Complication 8, and `scripts/pidref_test.py`
  for a one-second reproduction.
- **A restore right after its own dump can hit `TIME_WAIT`.** CRIU rebinds
  every local port it recorded, and the destructive dump's own FIN closes park
  those ports for 60s, so the restore dies inside CRIU with `Can't bind inet
  socket … Address already in use`. Dumps now mark the workers' TCP sockets
  `SO_LINGER(1,0)` so the kill RSTs instead; images from before that change
  need the 60s to drain, not a re-dump. See Complication 12. **But do not
  assume every "can't bind" failure drains** — see the next entry.
- **`Can't bind inet socket` has a second, permanent cause.** An image records
  its listening sockets' exact ephemeral ports (the TP1 35B image carries
  seven on loopback) and CRIU must rebind every one. If any live process in
  the restoring pod holds one, the restore fails and **waiting never helps**:
  seven consecutive attempts over 78s were seen failing on the identical port.
  `SO_REUSEADDR` does not rescue a listener from a live `LISTEN`, only from a
  `TIME_WAIT` tuple, so the Complication 12 reasoning does not transfer.
  `prepare_criu_dump` now closes those orphaned listeners, which **does**
  require a re-dump; nothing reused the ports anyway, since `reinit_nccl`
  rebuilds the rendezvous on fresh ones. This is the first placement-dependent
  restore failure that bites at **TP1**, which slot mismatch never did. A
  failure on an old image reports the holding pid via
  `_port_collision_report`. See Complication 15.
- **The close is a TP=1 tool. At TP>1 it must not run at all.** The premise —
  that torch orphans listeners the dump has to clean up — holds only at TP=1,
  where all 7 are torch's (`pt_tcpstore` + 14 × `pt_gloo_runloop` beside them)
  and NCCL starts **no RAS subsystem for a single rank**. At TP>1 torch's own
  teardown retires its listeners first (28 → 2 measured across two ranks), so
  the only ephemeral loopback listener left for the close to find is RAS's. The
  driver owns **none** at TP8, so a driver-only fix would have closed nothing
  there either. `_prepare_worker_dump` now censuses via `_loopback_listeners`
  and closes nothing; two source-level tests hold that split in place.
- **Never close *any* RAS listener — it has two per rank.** `::1:28028` is the
  fixed one and there is an **ephemeral** one beside it, both owned by a thread
  no torch teardown stops. Closing either leaves the restored process spinning
  `accept()` on `EBADF` with no backoff: ~143 MB/s, the chain stuck at
  `reinit_nccl`, criu reporting no error, and at TP=2 the job reporting
  `RUNNING` and serving correct answers throughout. Restricting the close to
  `ip_local_port_range` fixed only the fixed-port half —
  `ras/client_support.cc:203` went to zero while `misc/socket.cc:458` carried
  on alone. Being in the process group is not what makes a socket safe to
  close; being nobody's is, and RAS's are never nobody's. See Complication 15.
- **`NCCL_RAS_ENABLE=0` is the one-variable proof, and a usable stopgap.** With
  it set and nothing else changed, the gate reports `closed_listeners=[[], []]`
  — no socket exists to close — and the restore runs clean through
  `rebind_graphs OK` with zero NCCL warnings and a 60 KB log. It costs RAS
  diagnostics fleet-wide, so it is a lever rather than the fix.
- **A clean census does not mean a working image — it is nearly the opposite.**
  The run that shipped the RAS bug produced an image recording *one* listener
  instead of seventeen and that was read as success. At TP>1 a rank recording
  **no** ephemeral loopback listener is the *failure* shape, because that socket
  is RAS's and belongs in the image; `worker.py` warns on exactly that. The
  census proves what the image demands, never that the process still works —
  only a dump/restore cycle does, and one cold-start job covers both halves.
- **The restore-side log size is the verdict, not the job status.** At TP=2 the
  RAS spin does not fail anything: the job reaches `RUNNING` and answers
  prompts correctly while spewing ~143 MB/s of `accept()` errors. Since
  2026-10-02 every semi-p line goes to the pod log, so the test is the
  device-manager log's growth: `kubectl logs` twice, fifteen seconds apart, or
  `grep -ci 6D7C /proc/net/tcp6` (must equal TP). The dump side shows none of it.
- **`dist.is_initialized()` looked like the wrong gate and measured fine.**
  Both TP=2 dense and TP=8 MoE reported it `False` at dump time, so the
  original gate would have fired unaided; `_PG_ABORTED` is defensive, not a
  fix. Thread death is *not* a usable gate either — rank 0 was observed with a
  live `pt_tcpstore` at dump time, and gloo's listener outlives every teardown,
  so `pt_gloo_runloop` is censused but never waited on. See Complication 15.
- **The dump mints its own empty `--libdir`** with `tempfile.mkdtemp` and
  removes it afterwards, so no plugin loads while dumping and no node needs a
  pre-made directory under `/usr/lib`. The restore passes no `--libdir` at all
  and so does load `/usr/lib/criu/cuda_plugin.so`, which it needs — do not
  "unify" the two. See Complication 7.
- **An image binds to its node's shared libraries byte-for-byte.** CRIU
  re-validates the recorded size of every file-backed mapping at restore, so a
  venv that differs in one compiled extension kills a cross-node restore from
  inside CRIU, with no up-front check. Run `scripts/imgdiff.py <image_dir>`
  before restoring an image captured elsewhere.
- **An image's capability level is fixed at dump time.** Restoring on a pod
  without `CAP_SYS_ADMIN` needs `SEMIP_UNPRIVILEGED=1` set *when the image was
  dumped*, so the child sheds its capabilities and `restore_creds` can reinstate
  them. An image dumped without it cannot be restored there at all. The
  serving adapter now defaults it to `1` (`restore_and_wrap`) and records the
  effective value in `meta.json` as `unprivileged`; a local image dumped under
  the other value refuses to restore, and a published one is a miss. See
  Complication 11.
- **An image's warmth is fixed at dump time too, and nothing records it.** A
  CRIU image captures a process mid-life: whatever it had lazily built by the
  checkpoint is in the image, and whatever it had not, the restore builds. At
  TP>1 this was the 302 s post-restore warmup hang.
  `install_keepgraph_patch` forces `keep_graph=True` so the rebind can read
  graph topology, and torch's `capture_end` instantiates only when
  `keep_graph_` is false — so the graphs would ship uninstantiated, and the
  post-restore warmup would build ~2100 of them across four ranks while
  JIT-compiling the kernels no shape had dispatched to yet.
  **Warming the image is now part of `init` and unconditional** (it was
  `SEMIP_WARM_IMAGE=1` until 2026-09-24); the dump logs `COLD IMAGE` at error
  level if any rank still reports `uninstantiated > 0`, and a key must never be
  published past that. Wave 10: warm 12/12 clean vs cold 1/9, Fisher exact
  p = 5.8e-06. See `tp_DESIGN.md` §5.
- **Cross-uid dump/restore is not supported, and is now rejected.** An image's
  uid is fixed at dump time: `SEMIP_UNPRIVILEGED` drops capabilities without
  changing uid, so a tree dumped under `sudo` is `uid 0` with `cap_eff=[0, 0]`
  -- root in name, subject to ordinary mode bits in fact. Restore that under an
  unprivileged parent and the two identities fight over the same files
  (`<model_dir>/compilation`, `image/`, and until 2026-10-02 the child's
  `/tmp/inst<id>.log`), giving `PermissionError` at the very end of an otherwise
  successful restore. No file mode satisfies both, since CRIU validates the
  recorded mode. **root -> root and unprivileged -> unprivileged are the two
  supported configurations; the mix is not.** `_worker_criu_save` records the
  dumping `uid` in `meta.json` and `Instance.criu_restore` raises on a
  mismatch before spawning a worker, so the failure is now immediate and named
  rather than a late `PermissionError`. Images dumped before that field existed
  carry no `uid` and are let through unchecked. See `dss_integration.md`
  Section 9.
- **A root criu cannot dump an unprivileged child**, which is why the dump no
  longer shells out through `sudo`. criu reads the target's rlimits with
  `prlimit()`, which cross-uid needs `CAP_SYS_RESOURCE` -- outside a cap-prod
  pod's bounding set, so unacquirable. The dump dies at `cr-dump.c:389`
  ("Can't get rlimit 0") before writing anything. `_worker_criu_save` therefore
  invokes criu at the worker's own uid (the child is its own, so the uids match
  by construction); an unprivileged worker needs
  `setcap cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip
  /usr/sbin/criu`. See Complication 11.
- **An unprivileged restore needs two capabilities the dump does not**, so a
  binary qualified by a successful dump can still deadlock the restore.
  `restore_creds()` calls `PR_CAPBSET_DROP` for every capability *absent* from
  the image's recorded `cap_bnd` -- without checking whether it is already
  absent, so a cap-prod bounding set's ~25 gaps mean ~25 drops, each needing
  `cap_setpcap` -- and `setgroups`, which needs `cap_setgid` even when the
  recorded group list is identical to the current one. Both are inside a
  cap-prod bounding set, unlike `cap_sys_resource`. Failure is a deadlock, not
  an error: `Unable to drop capability 2` then `BUG at criu/pie/restorer.c:820`
  per task, the log stops with no `Restoring FAILED`, and the half-restored
  tasks strand at their recorded PIDs. See Complication 11.
- **Everything logs to the pod log, tagged by replica.** Since 2026-10-02 the
  worker, the vLLM child and its TP ranks write to PID 1's stdout
  (`/proc/1/fd/1`), and the parent's `semip.inst` and `semip_engine` records go
  there too, so `kubectl logs <device-manager pod>` is the whole story. Each line
  carries `[r<K> <role>]`, and vLLM's own lines `[r<K>]`, because several
  replicas share one stream. The restore hands the new pod's pipe back with
  `--inherit-fd` (CRIU_PLUMBING Complication 4), so a restored tree needs no
  rebinding. There is no `/tmp/inst<N>.log` any more. An experiment run by hand
  inside a job's pod writes into that job's log; point `SEMIP_LOG_TARGET` at
  your own pipe (`/proc/<shell pid>/fd/1`) to keep it separate. The kubelet
  keeps about 10 MiB of current log, so collect logs promptly.
- **Without the PID namespace, an image's *task ids* have to be free — threads
  included.** PIDs and thread ids share one counter, so a TP2 image needs ~900
  free ids and any 200-thread service (or your own launcher, or the IDE server
  that spawned your shell) can squat on the range; `ps` shows nothing, because
  threads live only under `/proc/<pid>/task`. `criu_restore` preflights this and
  names the occupants; `scripts/pidcheck.py <image>` answers it up front, and
  `--burn-to 200000` before a cold start puts a new image's ids out of
  contention for good. The serving adapter does that burn itself now
  (`SEMIP_PID_FLOOR`, 100000 by default, recorded in `meta.json`), because a
  *published* image is restored in a pod whose launcher holds the low ids by
  construction. See Complication 8 and `IMAGE_CACHE.md` §8.
- **Never shell out to `kill` with a negative pid.** procps-ng `kill(1)` parses
  a multi-digit negative pid as an option cluster and derives its target from
  the first digit alone, so `sudo kill -9 -1181` runs `kill(-1, SIGKILL)` --
  as root, every process it is permitted to signal. On the no-namespace path
  that killed the worker mid-teardown together with PID 1's only child, which
  ends the container and burns a `backoffLimit` retry; it cost a pod and its 8
  H200s. Teardown now enumerates the tree's task ids explicitly and checks each
  victim's `/proc/<pid>/stat` start time. See Complication 13.
- **`SEMIP_UNPRIVILEGED=1` needs a world-readable interpreter.** The child
  drops *all* capabilities, so a uid-0 process loses `CAP_DAC_OVERRIDE` and can
  only read what `other` can. An interpreter behind a private home (`chmod 750`)
  then breaks cold start with a bogus-looking `ModuleNotFoundError` for a stdlib
  module right after the `[semip] dropped capabilities` line. A venv inherits
  its base interpreter's stdlib, so check `sys.base_prefix`, not the venv.
  `criu_restore` itself is unaffected (it imports nothing), but what runs after
  it is: at TP>1 `reinit_nccl` spawns a process underneath vLLM's
  `in_the_same_node_as`, and the failure is a *silent* deadlock because vLLM
  suppresses the resulting `OSError`. No re-dump needed -- fix the permissions.
- **Reserved env keys** (`CUDA_VISIBLE_DEVICES`,
  `VLLM_ENABLE_V1_MULTIPROCESSING`, `USE_LIBUV`, the loopback trio and the
  compile-cache roots) are silently dropped from `vllm_config["_env"]` at apply
  time, but retained on disk in `meta.json`.
- **`criu_restore` does not re-apply `_env`** — the child's environment is baked
  into the CRIU image and restored verbatim.
- **Staging buffers must be freed via `storage().resize_(0)`**, not
  `caching_allocator_delete`, because vLLM's cumem allocator intercepts
  `torch.empty(..., device="cuda")`.

## Design docs

| Doc | Covers |
|---|---|
| [`instance_DESIGN.md`](instance_DESIGN.md) | Primitives table, process hierarchy, pin management, pause/resume |
| [`orchestrator_DESIGN.md`](orchestrator_DESIGN.md) | State machine, slot allocation, public API, Known Issues |
| [`pipeline_DESIGN.md`](pipeline_DESIGN.md) | Op model, interrupts, cross-model eviction, regression plan |
| [`slots_DESIGN.md`](slots_DESIGN.md) | Buddy allocator algorithms, invariants, worked example |
| [`client_DESIGN.md`](client_DESIGN.md) | Job/model two-layer split, calling shapes, session persistence |
| [`CRIU_PLUMBING.md`](CRIU_PLUMBING.md) | The fifteen CRIU complications and the FD keep-list |
| [`CROSS_NODE_RESTORE.md`](CROSS_NODE_RESTORE.md) | Runbook for dumping on node A and restoring on a low-capability node B |
| [`TEARDOWN_SCOPING.md`](TEARDOWN_SCOPING.md) | Why the no-namespace teardown killed the pod, and the bounded kill that replaced it |
| [`tp_DESIGN.md`](tp_DESIGN.md) | Tensor parallelism: the three TP primitives, NCCL teardown/rebuild, graph reuse, and **image warmth** (§5) — why `keep_graph=True` leaves the graphs uninstantiated and why warming the dump is now unconditional |
| [`semi-p_DESIGN.md`](semi-p_DESIGN.md) | The `model_dir` layout, what a re-dump touches, what binds an image — and why an image also has a warmth that nothing records |
| [`async_generate_DETAILS.md`](async_generate_DETAILS.md) | Async generate, IPC protocol, drain points |
| [`dss_integration.md`](dss_integration.md) | Restoring an image from a DSS sampling job: the cross-repo config contract, the worker hook, the adapter |
| [`IMAGE_CACHE.md`](IMAGE_CACHE.md) | The derived two-hash cache key, the node model-cache pipeline an image rides, how a miss is filled from the mirror, the multi-replica layout (§1), and the 2026-10-03 multi-replica roll with timings (§11) |
| [`INSTALL.md`](INSTALL.md) | CRIU install (PPA and from-source), draft model sync |

## Related skills

- `arctic-inference-architecture` — the surrounding vLLM plugin.
