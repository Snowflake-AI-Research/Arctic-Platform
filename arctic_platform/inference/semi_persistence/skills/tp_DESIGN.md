# Tensor Parallelism (TP>1) — Design

How the semi-persistence stack supports tensor-parallel instances, on top
of the single-GPU (TP=1) baseline, including dense TP and TP+EP (expert
parallel) MoE.

The guiding principle is **strict superset**: TP>1 is a set of extra
primitives and a few widened signatures, and every TP-specific step is a
**no-op at TP=1**.  The proven single-GPU path is left untouched, so
running with one GPU reproduces the original behavior exactly.

Scope: dense TP and TP+EP MoE. Ulysses / shift sequence-parallel is out
of scope (`vllm_child._is_arctic_parallel_worker` returns `False`).

---

## 1. Why TP>1 is not free

CRIU cannot snapshot a live distributed engine as-is, and CUDA graphs
bake absolute device addresses that move across a checkpoint/restore.
Three things break at TP>1 that simply do not exist at TP=1:

1. **NCCL communicators / TCPStore.** There is no communicator at TP=1.
   At TP>1, live NCCL comms and the torch `ProcessGroupNCCL` /
   `CustomAllreduce` IPC handles cannot survive a CRIU dump, so they must
   be torn down before checkpoint and rebuilt after restore.
2. **CUDA-graph baked addresses.** At TP=1 the captured decode graphs
   have no cross-rank IPC pointers.  At TP>1 (especially with
   CustomAllreduce) the graphs bake peer buffer addresses that go stale
   after `destroy_nccl` -> `reinit_nccl` reallocates them, so the graphs
   must be rebound (reuse) or recaptured (full).  See `ca_graph_rebind.py`.
3. **Physical GPU placement.** A TP group must span an arbitrary set of
   physical GPUs while keeping all of them visible (the cuda-checkpoint
   physical-GPU addressing requires it), rather than pinning one GPU via
   `CUDA_VISIBLE_DEVICES`.
4. **Where the weight-staging buffer lives.** At TP=1 vLLM uses the
   "uni" executor and the worker *is* the vllm_child process, so a
   buffer allocated there and captured by a `collective_rpc` closure is
   mutated in place.  At TP>1 the multiproc executor cloudpickles that
   closure into N subprocesses: each gets a *copy* of the buffer, writes
   into the copy, and the writes are discarded when the call returns --
   silently.  Each rank also owns a different shard of the parameters,
   so a single child-side buffer is the wrong size regardless.

These map to the three new primitives (`destroy_nccl`, `reinit_nccl`,
`rebind_graphs`), the `SemipGPUWorker` + `SEMIP_GPU_MAP`
placement mechanism, the worker-local staging primitives
(`worker._semip_*`), and a per-worker memory-budget correction.

---

## 2. Lifecycle chains, TP=1 vs TP>1

**TP is selected by the vLLM config**, not by the GPU count: set
`tensor_parallel_size=N` in `Instance(vllm_config)` (default 1 -> TP=1).
The `gpus=[...]` passed to `init` / `cuda_restore` is *physical placement
only* and must have exactly `tensor_parallel_size` entries (validated,
raises otherwise).  `n_gpus == tensor_parallel_size` is fixed at
construction.

The core pipeline is one shared sequence.  TP>1 interleaves NCCL and
graph steps; TP=1 omits them (they would be no-ops anyway).

```
                 TP=1                               TP>1 (tensor_parallel_size=N)
  save:          init(gpu)                          init(gpus=[...])  # len == N
                 generate                            generate
                 attach                              attach
                 [stage] save_weights                [stage] save_weights
                 detach                              detach
                 sleep                               sleep
                 cuda_checkpoint  ─────────────────  cuda_checkpoint
                                                       ├─ destroy_nccl   auto-inserted when n_gpus>1
                                                       └─ cuda_checkpoint
                 criu_dump                           criu_dump

  restore:       criu_restore                        criu_restore
                 cuda_restore(gpu)                   cuda_restore(gpus=[...])
                                                     reinit_nccl          <- NEW
                 wake_up_weights                     attach
                 attach                              load_weights
                 repin                               wake_up_weights
                 load_weights                        repin
                 plan_restore_weights                plan_restore_weights
                 restore_weights                     restore_weights
                 wake_up_kv_cache                    wake_up_kv_cache
                                                     rebind_graphs            <- NEW
                 generate                            generate
```

`destroy_nccl` before the checkpoint and `reinit_nccl`/`rebind_graphs`
after the restore are the only TP-specific caller-visible steps.
`cuda_checkpoint` inserts `destroy_nccl` itself; the caller must invoke
`reinit_nccl` and `rebind_graphs` on the restore side (see
`scripts/test_tp2.py`).

---

## 3. What TP adds to `instance.py`

Every item below is additive or a widened signature; nothing was removed
from the TP=1 path.

### 3.1 Per-instance TP state

`gpus` is the physical GPU list; `n_gpus == tensor_parallel_size`.
Because TP is wired from the config, `n_gpus` is derived from
`vllm_config["tensor_parallel_size"]` at construction (authoritative) and
`gpus` is validated against it — never the other way around.
`max_pinned_bytes_per_worker` is the largest **per-worker** staging shard,
needed because at TP>1 the aggregate `pinned_cpu_bytes` overstates the
per-GPU budget ~N-fold.

### 3.2 `init(gpus=...)`: validate placement, don't infer TP

`init(gpus=None, gpu=None)` is back-compatible: a scalar `gpu=` or
positional `init(0)` still works for TP=1.  TP size is read from the
config, and the supplied GPU list is validated to match; `init` never
infers TP from `len(gpus)` nor injects `tensor_parallel_size` into the
config (the user already set it).  It does inject the internal
`worker_cls` for TP>1.

`total_gpu_bytes` is snapshotted from `gpus[0]` — every GPU in a TP group
has the same capacity, so it stays representative of the per-GPU
`gpu_memory_utilization` budget.

### 3.3 `cuda_checkpoint()`: graph-preserving NCCL teardown prologue

Before the checkpoint, TP>1 preserves the captured graphs and
unilaterally aborts NCCL (`destroy_nccl`).  TP=1 keeps the single
`cuda_checkpoint` send.

### 3.4 Four new primitives

All four short-circuit to a no-op in the child when
`_semip_tp_size(worker) <= 1`, so they are safe to call unconditionally.

| Primitive | When | Child-side |
|---|---|---|
| `destroy_nccl()` | inside `cuda_checkpoint` (TP>1) | `_destroy_nccl` — unilateral `ncclCommAbort`, always graph-preserving; closes CustomAllreduce IPC handles |
| `reinit_nccl()` | after `cuda_restore` | `_reinit_nccl` — fresh TCP port, `init_worker_distributed_environment`, rebind `tp:0`/`world:0` (+ `ep:0`/`dp:0` for MoE) group slots the graphs look up |
| `rebind_graphs()` | after `wake_up_kv_cache` | `_semip_rebind_graphs` — rewrites the preserved graphs' baked CustomAllreduce addresses (`ca_graph_rebind`). Fails if discovery could not account for every captured graph (Complication 17), because patching a subset is indistinguishable from patching all of them until the first replay faults |

**There is one mode.** Captured graphs are always preserved across the
checkpoint, and the restore rewrites their stale device addresses in
place. The `full` alternative — discard and recapture via
`capture_model()`, selectable at dump time through `SEMIP_GRAPH_MODE` —
was retired 2026-09-24, together with the `cleargraph` primitive that
only it could reach and the `graph_mode` argument on everything else.
See section 5 for why reuse-against-a-warm-image is the only method we
trust.

### 3.5 `criu_dump()` / `criu_restore()`: TP shape in the image

`criu_dump` persists `n_gpus` and `max_pinned_bytes_per_worker` into
`meta.json` (alongside `gpus` and `gpu_uuids`, written worker-side).

The restore side has no live `init(gpus)` call.  TP size is authoritative
from this instance's `vllm_config["tensor_parallel_size"]` (already equal
to the image's, per the config mismatch check), so `n_gpus` is derived
from the config with `meta["n_gpus"]` kept only as a legacy fallback.
`gpus` / `max_pinned_bytes_per_worker` are hydrated from `meta.json` (with
a `rank` -> `[rank]` shim for legacy images), and the worker is spawned on
the restored GPU list rather than a hardcoded GPU 0.

### 3.6 `cuda_restore(gpus=...)`: place the group, validate against config

`cuda_restore(gpu=None, gpus=None)`.  With neither argument it falls back
to the `self.gpus` hydrated from `meta.json`, so a restore can re-place
the group on a different physical GPU set.  The GPU count is validated
against the config-derived `n_gpus` (TP size cannot change across a
restore) rather than redefining it.

### 3.7 `plan_restore_weights()`: per-GPU (not aggregate) chunk budget

This is the one correctness-critical numeric change.  The restore chunk
budget must be sized against the **per-GPU** staging shard.  Using the
TP-aggregate `pinned_cpu_bytes` would overstate the per-worker pinned
buffer ~N-fold, shrinking the chunk budget needlessly (or spuriously
failing the `param exceeds chunk_size` check):

```python
pinned = self.max_pinned_bytes_per_worker or self.pinned_cpu_bytes
mb = int(0.9 * min(pinned, allotment - pinned))
```

> **Why the budget matters (both TP=1 and TP>1).** `restore_weights`
> without a cached plan allocates a GPU staging buffer as large as the
> whole per-worker weight buffer, which torch's caching allocator keeps
> resident even after `resize_(0)`. `wake_up_kv_cache` then allocates via
> vLLM's `cumem` allocator straight from the driver and cannot reuse
> torch's cached block -> CUDA OOM. `plan_restore_weights()` bounds the
> staging buffer so enough free driver memory remains for the KV wake-up.
> Always call `plan_restore_weights()` before `restore_weights()`.
> Pass an explicit `max_buffer_bytes` for images dumped before the
> `empty_cache()` fix.

### 3.8 Result bookkeeping

`attach` and `plan_restore_weights` surface `max_pinned_bytes_per_worker`;
`criu_restore` / `cuda_restore` echo the resolved `gpus` back into
instance state; `detach` clears the per-worker figure.

---

## 4. Child-side counterparts

The `instance.py` surface is only the orchestration; the mechanism lives
in the child.  All of it degrades to a no-op at TP=1.

- **`_semip_worker.SemipGPUWorker`** (`vllm_config["worker_cls"]`, TP>1
  only). Two hooks:
  - `init_device` remaps vLLM `local_rank` -> physical GPU via
    `SEMIP_GPU_MAP` (set by the child before spawn), so the group spans
    an arbitrary GPU set with all GPUs visible.
  - `compile_or_warm_up_model` forces the cold-start CUDA-graph capture
    onto the CustomAllreduce copy path with `keep_graph=True`, so the
    preserved graph is reuse-friendly for `ca_graph_rebind`.  It returns
    whatever the base class returns, so it is agnostic to that method's
    return type across vLLM versions.  **`keep_graph=True` also suppresses
    torch's instantiate at capture, which is not obvious and cost nine
    debugging waves — the same hook therefore calls
    `instantiate_captured_graphs()` to undo that side effect, and
    `graph_exec_census()` to make it checkable.  See section 5.**
- **GPU visibility fork** (`vllm_child.vllm_child_loop`): TP>1 clears
  `CUDA_VISIBLE_DEVICES` and sets `SEMIP_GPU_MAP`; TP=1 pins the single
  GPU as device 0 (unchanged).
- **TP-only engine config** (`init`, `_tp >= 2`): disables fused
  allreduce+RMS, NVLS, symmetric-memory allreduce and the MoE
  shared-experts stream, all of which hold state CRIU cannot serialize
  or complicate graph capture.
- **`_destroy_nccl` / `_reinit_nccl`**: NCCL abort/rebuild; both
  early-out when `_semip_tp_size(worker) <= 1`.
- **`_prepare_worker_dump`**: per-worker CRIU dump prep (FD close,
  io_uring/IB munmap, PSM shm unlink), invoked only when `len(gpus) > 1`
  since at TP=1 the "worker" is the driver process itself.
- **`ca_graph_rebind.py`**: the graph-address rebind engine that lets
  preserved decode graphs replay after `destroy_nccl` -> `reinit_nccl`
  without a full recapture (the reuse path). This is the heaviest and
  most fragile piece of the TP work; it is entirely bypassed at TP=1.
- **Worker-local weight staging** (`_semip_attach`, `_semip_stage`,
  `_semip_repin`, `_semip_unpin`, `_semip_plan_load_weights`,
  `_semip_restore_weights`, `_semip_detach`, `_semip_save_weights`,
  `_semip_load_weights`): the staging buffer, param index and chunk plan
  live on the worker (`worker._semip_*`) for the reason in section 1.4.
  Unlike the primitives above this is *not* a TP-only addition -- it is
  the single path both TP sizes take, and the child only aggregates the
  per-worker results. `attach_pinned` is unsupported on it.
- **Per-rank weight shards**: `save_weights`/`load_weights` fan out to
  `weights/rank{R}/` at TP>1; TP=1 keeps the flat `weights/` layout.

### Worker-side (`worker.py`)

- `worker_loop` and `_child_thread` take a `gpus` list (a bare int is
  still accepted); `rank = gpus[0]`.
- `_gpu_migration_permutation` generalises the old 2-GPU swap into a full
  N-GPU bijection pinning `old_gpus[k] -> new_gpus[k]`, completing the
  permutation over the remaining GPUs so `cuCheckpointProcessRestore`
  gets a bijection over every visible device.
- Two-pass restore: restore **all** pids, then unlock **all** pids.  With
  a TP process tree an unlocked worker could otherwise touch a
  still-locked sibling over NCCL/IPC and race.
- The CRIU dump scans **descendant** PIDs (deduped by socket inode), not
  just the child, so the multiproc-executor Unix sockets are declared
  `--external`.

---

## 5. Image warmth: the work the image should not defer

An image has a **warmth**, and it is a property of the image, not of the
restore. A cold image is one whose CUDA graphs have never been
instantiated and whose shape-dependent kernels have never been compiled;
restoring it makes the restore pay for both, on every rank at once, in a
freshly restored CUDA context. That was the cause of the 302 s
post-restore warmup hang (then reached through `recapture_graphs`, now
`rebind_graphs`), and the fix is to do the work before the checkpoint
instead.

### 5.1 How the image went out cold

Two pieces of code, each correct alone:

1. `install_keepgraph_patch()` forces `torch.cuda.CUDAGraph(keep_graph=True)`
   so `raw_cuda_graph()` can serve the rebind.
2. torch's `CUDAGraph::capture_end()` instantiates **only when
   `keep_graph_` is false**:

   ```cpp
   void CUDAGraph::capture_end() {
     capture_end_pre();
     if (!keep_graph_) {
       instantiate();
     }
     capture_end_post();
   }
   ```

So the patch we added for the rebind silently switched vLLM's graphs from
eagerly instantiated at capture to lazily instantiated on first replay.
Measured at the end of `init` on an unwarmed dump: `exec_ok=0,
uninstantiated=2142` — not one exec on any rank.

The dump-time job then ran, touched a single batch shape, and lazily built
execs for that one shape: 41 piecewise wrappers + 1 full-decode wrapper =
42. The image was written holding 42 live execs and 2100 uninstantiated
graphs, plus two kernels (`_zero_kv_blocks_kernel` and the CuTeDSL
`_FullyFusedDeltaRuleSm90`) that no shape it ran had ever dispatched to.

### 5.2 Why that looked like graph corruption for nine waves

The post-restore warmup ladder was therefore not replaying preserved
graphs. It was building execs and JIT-compiling, four ranks at a time, and
it wedged at the one rung that does both (`nreq=16`, ~4.2 s of first-run
work). Every hang was there.

The reason this was misread as CRIU damage is worth recording:
`raw_cuda_graph_exec()` raises on a host-side `TORCH_CHECK(has_graph_exec_)`
— a C++ bool in the process image, which `cuCheckpointProcessCheckpoint`
has no way to reach. `n_uninstantiated=2100` *always* described a state
that predated the dump. Any diagnostic asking "what is corrupt in the
preserved graph?" was asking the wrong question, which is why every one of
them returned clean (see section 5.5).

`full` was never affected because it avoided both halves by accident:
`capture_model()` at restore ran outside the keepgraph patch, so
`capture_end` instantiated everything, and `full` never ran the ladder.
(`full` was retired 2026-09-24 — see section 2.)

### 5.3 The fix: warm the image at dump time

**Unconditional since 2026-09-24.** Warming is part of the `init`
primitive, not a knob: there is no `SEMIP_WARM_IMAGE` and no way to dump
a cold image, because a cold image is not a supported artefact. Two
halves, which fix two different cold spots:

| half | where | what it does |
|---|---|---|
| instantiate | `SemipGPUWorker.compile_or_warm_up_model`, per rank | `instantiate_captured_graphs()` calls `torch.cuda.CUDAGraph.instantiate()` on every captured graph. Pure driver calls, no kernels run. |
| execute | `vllm_child`, in the parent, through the engine | drives every captured batch shape so the shape-dependent JIT compiles before the checkpoint. |

They are not interchangeable. Instantiation builds execs but runs nothing,
so it cannot provoke a JIT; a kernel is only compiled when the runtime
*dispatches* to it, which needs the shape to actually execute.

Use `torch.cuda.CUDAGraph.instantiate()`, never a hand-rolled
`cuGraphInstantiateWithFlags`. The object then owns `graph_exec_` and its
own `replay()` finds it, so there is no handback step, and the flags match
what `capture_end` would have used (`AutoFreeOnLaunch`, plus
`UseNodePriority` on a new enough CUDA). An earlier experiment hand-rolled
the call with `flags=0` *and* skipped every graph whose
`raw_cuda_graph_exec()` raised — which was exactly the set that needed
building — and was recorded as inconclusive for that reason.

The execute half must cover the real capture-size list, not a hand-picked
ladder. Instantiation is shape-blind, but JIT is not, and kernel dispatch
moves with batch size: the CuTeDSL delta-rule kernel first appears between
decode 8 and 16, while the preserved graphs bake the Triton one. Warming a
subset only relocates the first-run compile into live traffic.

### 5.4 What it costs and what it buys

Measured at TP=4 on Qwen3.6-35B-A3B, 2142 graphs over 51 shapes:

| | cost |
|---|---|
| instantiate 2142 execs | ~3.0 s, 0.50 GiB |
| drive 57 rungs (51 shapes, both axes) | ~7.9 s |
| **dump-side total** | **~11 s, once** |

Invisible where it is spent: warm dumps averaged 353 s from slot
assignment to image written, cold 351 s, against a 340–368 s spread on
identical configurations.

On the restore side, per wake:

| | cold (passing) | warm | full |
|---|---|---|---|
| `rebind_graphs` | 6.36 s | **2.01 s** | 19.02 s |
| rung 16 | 4.24 s | **0.035 s** | n/a (no ladder) |

(Measured before the 2026-09-24 rename; the `full` column is the retired
mode, kept because it is what made the case for dropping it.)

**This is the general principle, not a one-off:** work moved from the
restore path to the dump path is paid once and amortised over every wake.
An 11 s dump cost that removes 4.35 s per restore breaks even after three
restores, and semi-persistence exists to restore an image many times.

Wave 10 measured the reliability effect over 42 wave restores (22 matched,
20 lost to unmatched slots) plus 7 dump-site restores: **warm 17 pass /
0 hang, cold 1 pass / 8 hang**, Fisher exact p = 5.8e-06. Every cold failure
carried the identical signature, `recapture_FAILED` with
`stuck_at_nreq=16`. The cold control ran at an 89% hang rate that night,
so the bug was fully present in the build that the warm arm passed on.

### 5.5 The diagnostics, and why they were all negative

Do not re-run these. Each was built to find damage in the preserved graph,
and each returned clean because the graph was never damaged:

| gate | question | verdict |
|---|---|---|
| `SEMIP_CA_E4_INVENTORY` | any stale baked 8-byte pointer? | 308,832 inventoried, 0 stale |
| `SEMIP_E8_NODE_INV` | a node class E4 never walked? | only kernel and memcpy nodes exist |
| `SEMIP_CA_E3_DUMP` | CA `_dp` slots and peer pointers valid? | clean |
| `SEMIP_CA_PAD_PROBE` | CA Signal zero-initialised? | clean, 0 nonzero words |
| `SEMIP_CA_ZERO_SIGNAL` | does zeroing the local CA signal help? | hung (n=1); dead hypothesis |
| `SEMIP_CA_E5_REINSTANTIATE` | does a fresh exec help? | inconclusive — it skipped the uninstantiated graphs |
| `SEMIP_E9_H4` | are the block-table contents out of range? | **never fired** — its hook is gated on descriptors ≤ 8 and the hang is at 16 |
| `SEMIP_E11_VALDIFF` | do non-pointer baked args differ from a fresh capture? | designed, never run |
| `SEMIP_REUSE_DIAG` | which replay surface faults? | all faulting replays had `preserved_exec=False` |

The E9 and E11 rows are there so neither is mistaken for a promising lead
that nobody got to. E9 could not have fired whatever it found, and E11's
premise — that some baked value differs — is undercut by the finding that
nothing in the graph was ever wrong.

The `SEMIP_REUSE_DIAG` row was the answer, printed on every run: every
faulting replay was a fresh lazy instantiate. It read as "fresh execs built
from preserved topology are bad" rather than "these were never built
before".

**Deliberately unresolved: which half of the fix is load-bearing.** Eager
instantiation and JIT prewarm shipped together, and wave 10 cannot say which
one prevents the deadlock — rungs 1–8 instantiate without compiling and never
hang, which hints at the JIT or the combination, but no stack was ever captured
to confirm it (`spins.txt` was empty on every hung probe).

This is a decision, not a gap. Separating them would need one extra arm
(instantiate-only; "execute-only" is not a real arm, because replaying a shape
lazily instantiates its graphs, so driving all 51 shapes builds every exec as a
side effect). It would name the mechanism for an upstream report, but it would
not change what ships: both halves are cheap, the combination is robust at
17/17, and dump-side work is strictly better for restore latency regardless of
which half earns it.

Revisit only if the hang returns despite a warm image. In that case the
instantiate-only arm is the first experiment to run, and tonight's `both` and
`cold` results stand as its controls at TP=4.

### 5.6 The invariant to assert

`graph_exec_census()` reports execs held per shape and is cheap enough to
call on both sides of the checkpoint — `raw_cuda_graph_exec()` is a
host-side bool check, not a driver round trip. Assert **`uninstantiated=0`
at dump time**, not a graph count: 2142 is a property of this model
(41 piecewise wrappers + 1 full-decode wrapper, 51 shapes), the invariant
is not.

Never census *between* warmup rungs on the restore side. It costs a
`collective_rpc`, which drops a synchronisation point immediately before
the rung the hang lives on and perturbs the measurement.

---

## 6. Invariants and gotchas

- **TP is config-driven.** Set `tensor_parallel_size` in the vLLM config;
  `init`/`cuda_restore` take `gpus` as placement only and require
  `len(gpus) == tensor_parallel_size` (raise otherwise). `n_gpus` is
  fixed at construction and never re-derived from a GPU count.
- **Superset invariant.** Any TP>1 primitive is a no-op at TP=1, and the
  TP=1 chain is exactly the original chain. Do not route TP=1 through
  the NCCL/graph machinery.
- **Caller drives NCCL/graph on restore.** `cuda_checkpoint` inserts
  `destroy_nccl` itself, but `reinit_nccl` (right after `cuda_restore`)
  and `rebind_graphs` (right after `wake_up_kv_cache`) are the caller's
  responsibility.
- **`destroy_nccl` frees the ports too, not just the comms.** Before
  tearing the process groups down it marks every inet TCP socket in each
  worker `SO_LINGER(1,0)`, so the destructive dump's kill RSTs them and
  their local ports skip `TIME_WAIT`.  Without that, a restore inside 60s
  of its own dump fails in CRIU with `Address already in use`, because
  CRIU rebinds each recorded local port.  Every rank contributes a
  TCPStore connection, which is why the collision is deterministic at
  TP>1 and only a race at TP=1.  See Complication 12 in
  [`CRIU_PLUMBING.md`](CRIU_PLUMBING.md).
- **`reinit_nccl` ordering.** It must run after `cuda_restore`, and before
  anything that runs the model or replays a captured graph — in practice
  before `rebind_graphs` and `generate`.  `attach` and `load_weights`
  are *not* constrained by it: `collective_rpc` is the executor's
  Unix-socket fan-out, not a device collective, and the work each rank
  does is CPU-only (a host `torch.empty` for `_semip_attach`, then shard
  reads from `<weights_dir>/rankN` for `_semip_load_weights`).
  `test_tp2.py` runs both *before* `cuda_restore` for that reason.
  `restore_weights` is in between: its H2D copies need the restored
  context, but no NCCL.
- **Per-worker budget, not aggregate.** Size the restore chunk plan from
  `max_pinned_bytes_per_worker`; the aggregate `pinned_cpu_bytes` is
  ~N times too large per GPU.
- **Measure the staging budget, never predict it.** The config formula
  (`total × util − weights`) asserts that everything in the allotment
  which is not weights is free, which ignores graph pools, the per-rank
  CUDA contexts mapped on *every* GPU, and activations — ~9 GiB of
  phantom headroom on GLM-5.3 at TP=8, and a late OOM. The worker clamps
  against `torch.cuda.mem_get_info`; the config value is an outer bound
  only. Note the KV cache is *not* resident at that moment, so free
  memory there does not scale with `gpu_memory_utilization`.
- **There is one graph mode.** Graphs are always preserved and always
  rebound. `full` (recapture via `capture_model()`) and `SEMIP_GRAPH_MODE`
  were retired 2026-09-24; there is no fallback if a rebind fails.
- **Dumps are warm by construction.** Warming is part of `init`, not a
  knob, because a reuse image whose graphs were never instantiated makes
  the restore build them and that is where the 302 s hang lived. The dump
  logs `COLD IMAGE` at error level if any rank still reports
  `uninstantiated > 0`; never publish a key past that. Anything that
  changes capture, kernel dispatch, or the dump-time workload can re-cool
  an image, so this is a standing check rather than a bug fixed once.
- **Move work to the dump side.** The dump runs once, in a healthy single
  process; the restore runs on every wake, on all ranks, in a
  freshly restored context. Given the choice, pay at dump time.
- **Silent degradation.** `ca_graph_rebind` is written defensively with
  many guarded imports and `getattr` fallbacks, so a mismatch against a
  newer vLLM tends to *skip* a rebind path rather than raise. When
  bringing up a new vLLM version, check the logs for skipped paths
  instead of assuming success. Graph *discovery* is no longer on the
  honour system: `_find_captured_graphs` and `_collect_graph_entries` both
  report `complete` / `incomplete_why` in their diag, and `rebind_graphs`
  fails rather than reporting a clean rebind over a partial graph set.
  Nothing else in the module has that outside view — `topo_readback` and
  `stale_ca_audit` both range over the graphs that were *found*, so they
  audit a missing family perfectly clean. See CRIU_PLUMBING.md
  Complication 17.
- **A silent no-op reports success.** The same shape three times over in
  waves 28–29: a loop over an empty set, a heap scan that found nothing,
  an audit over exactly the set that was just patched. The communicator
  checkpoint hooks are the worst case — `checkpoint_restore` over zero
  workspaces is byte-identical to a full restore — so `reinit_nccl` now
  counts what `checkpoint_prepare` detached and fails unless the same
  number comes back. If a step can do nothing and return cleanly, make it
  count what it did against an expectation formed somewhere else. See
  CRIU_PLUMBING.md Complication 16.
- **Image portability.** `meta.json` carries `n_gpus`, `gpus`,
  `gpu_uuids` and `max_pinned_bytes_per_worker`; legacy single-GPU images
  fall back via the `rank` -> `[rank]` shim and `n_gpus=1`. Because
  `tensor_parallel_size` lives in the user's `vllm_config`, it is
  persisted in `meta.json` and participates in the `criu_restore`
  mismatch check, so a TP=2 image is not mistaken for a TP=1 one.

---

## 7. Primitive delta summary

```
TP>1 = TP=1 baseline
     + { destroy_nccl, reinit_nccl, rebind_graphs }                   # 3 new primitives
     + TP wired from vllm_config["tensor_parallel_size"]             # config-driven, not GPU-count
     + init(gpus=[...]) / cuda_restore(gpus=[...]) placement + check  # len(gpus) == tp validated
     + cuda_checkpoint auto-inserts destroy_nccl @ TP>1               # behavior change
     + init warms the image (instantiate + execute) @ TP>1            # unconditional, not a knob
     + plan_restore_weights clamps by measured free VRAM              # per-rank, not predicted
     + meta.json carries n_gpus / gpus / max_pinned_bytes_per_worker  # image portability
     + weights/rank{R}/ shard layout                                  # per-rank shards
```

Plus one change that is shared rather than TP-gated: weight staging
lives on the workers (`worker._semip_*`) for both TP sizes, because a
child-side buffer cannot be written by workers at TP>1.

Nothing is removed from the TP=1 path; TP>1 is a strict superset.
