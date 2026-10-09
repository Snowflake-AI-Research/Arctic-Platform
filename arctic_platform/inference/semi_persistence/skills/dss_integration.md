# DSS integration: semi-persistent sampling jobs

Status: **working end to end**, on 2026-09-08, with the whole tree at one uid
(dump under `sudo`, gateway under `sudo` — Section 9). This document was
originally written ahead of the code as a spec, then reconciled against the
working implementation. Where the two disagreed the code won, and the three
places this document was wrong are called out inline so the corrections are not
silently absorbed: Section 2 (unknown config fields do not raise), Section 5
(`plan_restore_weights` needs `attach()` first) and Section 8 (the pool config
fields are not rejected either).

> Naming note (2026-09-24): the incident narratives and quoted errors below
> predate the rename of `recapture_graphs` to `rebind_graphs` — including
> `RuntimeError: command 'recapture_graphs' failed: PermissionError`. Quoted
> verbatim so the record stays accurate. The restore sequence in Section 5 has
> been updated to the current primitive name.

Three jobs reached `RUNNING` and generated coherent text, and all three were
served **concurrently** on one node:

| Job | Model | GPUs | Image leader pid | Result |
|---|---|---|---|---|
| 1 | `Qwen3.8-27B` TP1 | `[0]` | 18255 | RUNNING, coherent text |
| 2 | `Qwen3.6-35B-A3B` TP1 | `[1]` | 18257 | RUNNING, coherent text |
| 3 | `Qwen3.6-35B-A3B` TP2+EP | `[2,3]` | 20135, 21735, 21736 | RUNNING, coherent text |

Job 1's restore took **22s** wall (first `inst0` record to `recapture_graphs
OK`), against the ~1 minute a cold start needs. Four independent signals
distinguish a restore from a silent cold start, and all four held: the child
returns at the image's *recorded* leader pid rather than a fresh one, the full
CRIU sequence appears in the instance log, `attach()` reports the image's pinned
byte counts, and the timing. `rebind_log` completed in 0.000s — the failure that
blocked every earlier attempt — because the parent, the restored child and
`/tmp/inst<N>.log` were all uid 0.

**The adapter alone is separately confirmed at both TP1 and TP2** via
`restore_smoke.py`, which drives `restore_and_wrap` through the worker's own
entry point with no gateway, zone or Ray actor:

| Image | GPUs | Restore | Staging budget | Result |
|---|---|---|---|---|
| `qwen_27b` | `[0]` (dumped `[5]`) | 29.1s | 42.59 GiB, `pinned=50.96` | coherent text, clean teardown |
| `qwen_35b_tp2` | `[0,1]` (dumped `[2,3]`) | 36.0s | 29.45 GiB, `pinned=65.44`, `per_worker=32.72` | coherent text, clean teardown |

Both budgets are non-zero against images recording `pinned_cpu_bytes: 0`, so
Section 5's `attach()`-before-planning contract holds in practice and not just
on paper. Coherent output additionally shows the weights were genuinely
repopulated — a bad plan yields garbage, not sense.

The TP2 row is the more informative one. `29.45 = 0.9 x min(32.72, 98.28 -
32.72)` uses `max_pinned_bytes_per_worker`, **not** the TP-aggregate
`pinned_cpu_bytes` of 65.44 GiB, which is exactly the per-rank preference
Section 6 describes; the aggregate would have sized the buffer at twice the
truth. That run also exercised the two-element GPU list against
`cuda_restore`'s count check, `reinit_nccl()`, the per-rank
`weights/rank<N>/` layout, and the cross-node device map.

The 2026-09-08 re-run of the `qwen_27b` row restored in **21.3s** rather than
29.1s, on a pod whose criu invocation was otherwise identical, so treat these
figures as the right order of magnitude and not a benchmark.

Everything static was re-checked on 2026-09-08 against the installed wheels:
every `Instance` primitive the adapter calls exists and returns `self` (so the
chaining in Section 5 is valid), `attach()` takes no arguments,
`cuda_restore(gpus=...)` accepts the list form, all three job JSONs validate
through `JobConfig` into engine kwargs carrying `semi_p`/`semi_p_model_dir`,
and the adapter covers every `self.llm.*` attribute `worker.py` reaches on a
sampling job. Section 9 lists the pod-level preconditions that turned up while
checking, and is the section to read before a first run on a new node — the
uid rule and the two capability constraints all live there, and each one costs
a full cold start to rediscover.

A DSS sampling job normally brings up its vLLM engine with a cold load: weights
off disk, compile, capture graphs, roughly a minute for a 35B MoE. This
integration lets the same job come up by **restoring a pre-warmed CRIU image**
instead, in a second or two, by routing `InferenceWorker.initialize` to a
`semi_persistence.Instance` rather than to `AsyncLLM`.

The feature spans three repos, and nothing works until all three agree. That
cross-repo contract is the reason this document exists: neither side is
readable on its own.

| Repo | Role |
|---|---|
| `dss-platform` | Job config (`semi_p` flag) and the passthrough into engine kwargs |
| `arcticinference-internal` | The worker hook, the adapter, and `semi_persistence` itself |
| `dss-client` | Example job JSONs that set the flag |

---

## 1. The contract, end to end

```
sampling-tp1-semip.json          inference_config.semi_p = true
  |
  v
dss/ray_dss/models.py            InferenceConfig.semi_p            (pydantic)
  |                                 branch mert/semi-p-integration only
  v
dss/ray_dss/jobs/gpu/sampling.py  kwargs["semi_p"] = True          (coordinator, num_gpus=0)
  |                                 same branch; absent from main
  v
arctic_inference/server/worker.py InferenceWorker.initialize       (GPU pod)
  |                                pops semi_p* before AsyncEngineArgs
  v
arctic_inference/server/semip_engine.py  restore_and_wrap(...)
  |                                _resolve_model_dir derives the path
  v
semi_persistence.Instance         criu_restore -> ... -> up
  |
  v
$SEMIP_IMAGE_CACHE/<cfg12>_<env12>[/replica<K>]  image/ + weight/ + compilation/
```

Four properties of this chain are easy to get wrong:

- **The flag is popped, not passed.** `AsyncEngineArgs` rejects unknown kwargs,
  so `worker.py` must `pop()` every `semi_p*` key before it builds engine args,
  whether or not semi-p is enabled.
- **The config hop crosses a process boundary with no GPU.** `sampling.py`
  runs on the `num_gpus=0` coordinator, which has no baked cache. It only maps
  and forwards; every existence check belongs on the GPU pod.
- **The adapter replaces `self.llm`.** Downstream server code calls `self.llm`
  as if it were an `AsyncLLM`, so `_SemiPEngine` has to satisfy that surface,
  not just `generate`.
- **Teardown is not garbage collection.** The `Instance` owns an
  out-of-process worker and vLLM child, so `InferenceWorker.shutdown` must call
  `close()` on the adapter before dropping the reference.

---

## 2. Configuration surface

DSS side, `dss/ray_dss/models.py` on `InferenceConfig` — on the
`mert/semi-p-integration` branch only, not on `main`; see the note below:

| Field | Default | Meaning |
|---|---|---|
| `semi_p` | `False` | Bring the engine up by restoring a pre-warmed image |
| `extra_env.SEMIP_IMAGE_CACHE` | `/data-fast/image-cache_neutrino` | Root of the image cache, written to |
| `extra_env.SEMIP_IMAGE_SOURCE` | `/mnt/neutrino/base-models/image-cache` | Root of the read-only mirror published images arrive on. `""` turns it off |
| `extra_env.SEMIP_UNPRIVILEGED` | `1` | Capability level of the dump; recorded in `meta.json` as `unprivileged` |

**A job needs only `semi_p: true`** (since 2026-10-03, image
`dev_20261003_191750_7f806d170f8`). The three defaults live in
`server/semip_engine.py` (`_DEFAULT_IMAGE_CACHE`, `_DEFAULT_IMAGE_SOURCE`, and
`os.environ.setdefault("SEMIP_UNPRIVILEGED", "1")` at the top of
`restore_and_wrap`); setting any of them in `extra_env` still overrides it.

arctic_inference side, `arctic_inference/server/config.py` on `ModelConfig`,
mirrors `semi_p`. It is popped by `InferenceWorker.initialize`.

**There is no `semi_p_model_dir` field.** The image directory is derived —
`$SEMIP_IMAGE_CACHE/<config hash>_<environment hash>`, plus `/replica<K>` when
the pod holds several replicas — because the directory name *is* the cache key,
and a job cannot name a key it cannot compute. See
[`IMAGE_CACHE.md`](IMAGE_CACHE.md) Sections 1 and 6 for the derivation, and
Section 6 below for why the path still has to be exact.

The overrides travel in `vllm_config.extra_env`, which
`InferenceWorker.initialize` applies to `os.environ` as its first statement —
before the semi-p branch, which is what makes them readable in
`restore_and_wrap`. `ReplicaPool` adds `SEMIP_REPLICA_ID` and
`SEMIP_NUM_REPLICAS` to the same dict per worker; a job never sets those.

**The two roots default because they are cluster facts.** `/data-fast` is the
`data-nvme` volume the neutrino operator mounts in every device-manager pod,
and the mirror is the operator's `modelCacheMountPath` plus the DaemonSet's
`image-cache` directory. Every pod has to agree on both anyway, or no later run
finds an earlier run's image. `SEMIP_IMAGE_SOURCE=""` is a supported deployment
— there is merely nowhere to *read* a published image from, so a miss
cold-starts — and is the A/B switch. Setting the source to the same path as the
cache root is a misconfiguration and is dropped with a warning: one directory
cannot be materialized from itself.

Neither variable can move the cache key. `extra_env` reaches `initialize` as
its own parameter and never enters `engine_kwargs`, so the same config resolves
to the same directory whether or not a source is configured — which is what
makes turning `SEMIP_IMAGE_SOURCE` on and off an A/B against one published
directory rather than two.

> **Where the dss side lives, and why it is not on `main`.** The dss half of
> this contract is **not on `dss-platform` `main`**, and not at the commit cortex
> pins the `neutrino/dss-backend` submodule to (`630f371`): neither contains the
> string `semi_p` anywhere, so neither would forward the flag at all. It exists
> only on two unmerged branches — `mert/semi-p-integration`, which is what the
> deployed `dss-backend` image is built from (it carries the ray zone bring-up
> fixes too), and `mert/semi-p-config`, the same change without them and the
> sensible base for an eventual PR. Both are tens of commits behind `main`.
>
> `gh search code` is not indexed for that repo and will find nothing. Fetch by
> ref instead:
>
> ```bash
> gh api "repos/<dss-backend repo>/contents/dss/ray_dss/models.py?ref=mert/semi-p-integration" \
>     --jq .content | base64 -d
> ```
>
> **History.** `semi_p_model_dir` was a required job field on both branches, and
> `InferenceConfig._require_semi_p_model_dir` rejected a semi-p job that omitted
> it at `/initialize` — which made a job written against the current contract
> unrunnable, since it was rejected before reaching any of this code. Both
> branches dropped the field and the validator on 2026-09-17 (dss-platform
> `65fc269` / `fa74d7f`), so `semi_p` is now the only key that crosses.
>
> A job JSON that still carries `semi_p_model_dir` stays valid either way:
> `BaseConfig` sets `extra="allow"`, so it is kept as a pydantic extra (with the
> usual `received unknown field(s)` warning) rather than rejected, and
> `worker.py` pops and discards the key unconditionally so it cannot reach
> `AsyncEngineArgs`. The derived path is used regardless.

> **Correction.** An earlier draft of this document claimed that "an
> unrecognized `inference_config` field is a pydantic error". It is not.
> `BaseConfig` sets `extra="allow"` and only logs
> `"%s received unknown field(s): %s"`, keeping the value as a pydantic extra.
> So renaming a field on *either* side leaves the job valid and silently
> cold-starting, with nothing but one warning line to show for it. When a
> semi-p job takes a minute to come up, grep the gateway log for that warning
> first. (A useful side effect: `getattr(cfg.inference_config, "semi_p",
> False)` in `sampling.py` works even against a `dss` that predates these
> fields.)

---

## 3. The worker hook

`InferenceWorker.initialize` in `arctic_inference/server/worker.py` gains an
early-return branch, placed **before** the `AsyncEngineArgs` construction and
before the vLLM plugin force-load:

```python
semi_p = bool(engine_kwargs.pop("semi_p", False))
engine_kwargs.pop("semi_p_model_dir", None)
if semi_p:
    from arctic_platform.inference.server.semip_engine import restore_and_wrap
    self.llm = await asyncio.to_thread(restore_and_wrap, engine_kwargs)
    self.state = WorkerLifecycleState.READY
    self._maybe_init_reasoning_parser(reasoning_parser_name)
    logger.info(...)
    return
```

Both `pop` calls are unconditional, before the branch, because
`AsyncEngineArgs` rejects unknown kwargs on the cold path too. The second one
discards its value: `semi_p_model_dir` is no longer a field here, but the dss
branch actually deployed (`mert/semi-p-integration`) still *requires* it and
forwards it, and `extra="allow"` would otherwise carry it through to
`AsyncEngineArgs`. Discarding rather than honouring it is deliberate — a job
that names a directory is naming a cache key it cannot compute.

The branch sits after `os.environ.update(extra_env)`, which is the first
statement in `initialize` — that ordering is what makes `SEMIP_IMAGE_CACHE`
readable inside `restore_and_wrap`.

`asyncio.to_thread` matters: the restore is a long blocking sequence of
queue round-trips to the worker process, and running it inline would stall the
event loop that the health endpoint answers on.

`shutdown()` gains the matching teardown:

```python
close = getattr(self.llm, "close", None)
if callable(close):
    close()
```

It is written defensively rather than as an `isinstance` check so the same
path is inert for a real `AsyncLLM`.

---

## 4. The adapter

`arctic_inference/server/semip_engine.py` holds two pieces:

- `restore_and_wrap(engine_kwargs, model_dir)` — resolves the physical GPUs,
  reads the baked `vllm_config` from the image, runs the restore sequence, and
  returns the adapter.

  > **Correction.** It no longer only restores. A missing image used to raise
  > `FileNotFoundError`; it now cold-starts vLLM, dumps an image into
  > `model_dir`, and restores that. So `engine_kwargs` reaches a third place
  > besides `_log_config_divergence` and `tokenizer_path` — the dump path
  > projects it into the `vllm_config` it bakes. See
  > [`dss_dump.md`](dss_dump.md), Section 11 for what was built.
- `_SemiPEngine` — stands in for `self.llm`. Forwards `generate` onto
  `Instance.generate`, collects results out of `instance.generate_results`
  keyed by `instance.last_req_id`, and maps the server's lifecycle calls onto
  `pause` / `resume` / `sleep` / `teardown`.

**GPU resolution returns a list, not a scalar.** The `Instance` machinery
enumerates *physical* devices through NVML and ignores `CUDA_VISIBLE_DEVICES`,
while the Ray actor is handed a masked view. So `_resolve_physical_gpus` parses
**every** id out of `CVD`, clears `CVD` so the `Instance`'s child can address
those physical ids directly, and hands the list to `cuda_restore(gpus=...)`.
Reading only the first id — as the prototype did — works at TP1 and breaks
every TP>1 restore, because `cuda_restore` validates the count:

```
cuda_restore: 1 gpu(s) given ([2]) but tensor_parallel_size=2
```

`criu_restore` spawns the worker on the placement recorded in `meta.json`
before `cuda_restore` re-places it, so a mismatch between Ray's assignment and
the dumped placement is expected and supported. TP size itself cannot change
across a restore, so a job's `n_gpus` must equal the image's
`tensor_parallel_size`.

**A cross-node image must not be restored onto the indices it was dumped on.**
This is the one that surprises people, because the *safe*-looking choice is the
broken one. `_worker_restore` builds the `oldUuid=newUuid` device map only when
the placement changed:

```python
migrate = bool(old_gpus and new_gpus and list(old_gpus) != list(new_gpus))
```

Restore an image from another node onto the same indices and no map is built,
so the checkpoint keeps the dumping node's baked GPU UUIDs — which do not exist
here — and `cuCheckpointProcessRestore` fails with a bare
`CUDA_ERROR_INVALID_VALUE`. Zero UUID overlap between the two nodes is the
normal case, not an error; `meta.json` carries the dump node's `gpu_uuids`
precisely so the map can be built.

Under DSS you do not choose the placement — Ray does — so the adapter cannot
fix this, only refuse it. `_check_gpu_placement` compares the image's
`gpu_uuids` against local NVML and raises with the actionable message when the
assignment happens to match the dumped indices and the UUIDs are foreign. If it
fires, place the job elsewhere rather than editing the check. Full background in
[`CROSS_NODE_RESTORE.md`](CROSS_NODE_RESTORE.md).

**On the pod these notes were written against the check is live, not
theoretical.** All three baked images record the same eight `gpu_uuids`,
beginning `GPU-bc969410-…`, and this node's eight H200s are
`GPU-89bd898b-…` and friends — **zero overlap**, so every image here is a
cross-node image. That is the supported case and needs no action, but it does
make three placements unrestorable:

| Image | Dumped on | Ray must not assign |
|---|---|---|
| `qwen_27b` | `[5]` | exactly `[5]` |
| `qwen_35b` | `[7]` | exactly `[7]` |
| `qwen_35b_tp2` | `[2, 3]` | exactly `[2, 3]` |

Ray fills from GPU 0 upward, so a first job is fine and the tp2 image is fine
on `[0, 1]`. A later job that happens to land on the dumped indices will be
refused by `_check_gpu_placement` with the message above rather than dying
inside `cuCheckpointProcessRestore`. Any other placement, including a partial
overlap, builds the device map normally.

> **Retired 2026-10-03: the next four paragraphs describe plumbing that no
> longer exists.** fd 1/2 are now PID 1's stdout pipe, recorded as
> `stdout_resource` and handed back with `--inherit-fd` (CRIU_PLUMBING
> Complication 4). There is no `/tmp/inst<N>.log`, no
> `_precreate_dumped_log_paths` and no `rebind_log`; every line goes to the
> device-manager pod log, tagged `[r<K> <role>]`. Kept for images dumped before
> that build, which cannot restore on it anyway (`env12` changed).

**The dump-time log paths have to exist before CRIU opens them.** CRIU restores
fd 1/2 *by path*, and that path encodes the `instance_id` the model had when it
was dumped — `/tmp/inst<N>.log`, from `semip_logging._INSTANCE_LOG_TEMPLATE` —
not the id this `Instance` was handed now. The library re-`dup2`s onto the
current log immediately afterwards (`rebind_log` in `worker.py`), but CRIU
still has to open the dump-time path first, and on a fresh node nothing has
created it:

```
Error (criu/files-reg.c:2353): Can't open file tmp/inst1.log on
restore: No such file or directory
```

This looks intermittent and is not: an image whose dump-time id happens to
match the one we drew restores fine, because our own `Instance` already created
that file. `_precreate_dumped_log_paths` reads the real paths out of the
image's `files.img` with `crit decode` and touches them, before `Instance` is
constructed.

**Create it, but do not touch its mode.** This path is the dumped process's
fd 1, so CRIU mode-checks it exactly as it does the mmap'd files under
`compilation/`, and the dump recorded `0644`
(`crit decode -i image/files.img` confirms `mode=0o100644`). Widening it to
`0666` — a tempting way to let both the unprivileged parent and the root child
append — fails the restore outright:

```
Error (criu/files-reg.c:2294): File tmp/inst0.log has bad mode 0100666 (expect 0100644)
Error (criu/files.c:1221): Unable to open fd=1 id=0x1b3
```

The default umask already yields `0644`, and the restored child needs no
permission of its own: `criu` opens the fd as full root, and writes through an
already-open fd are not re-checked.

The one case that genuinely breaks is **mixing root and non-root runs**. A
root-run `restore_smoke.py` leaves `/tmp/inst<N>.log` owned by root, and `/tmp`
is sticky, so an unprivileged worker can neither open nor replace it — the
restore then dies with `PermissionError: '/tmp/inst0.log'`. The fix is
`sudo rm -f /tmp/inst*.log` between runs by different users.

**Every job's log lands on `/tmp/inst0.log`, and each one wipes the last.**
`_next_instance_id` is a per-process global starting at 0, and
`Instance.__init__` calls `truncate_instance_file`. Each DSS job gets a fresh
actor process, so each job's single `Instance` is id 0 and truncates the file
the previous job wrote. Three sequential jobs leave one log, belonging to the
third; the first two restores leave no trace. Nothing breaks, but the evidence
you would want after a failure is destroyed by the *next* attempt, so copy the
file aside before starting another job. The same collision hits two dump
scripts run concurrently — both take id 0 and interleave into one file. It shells out to `crit` (which ships with CRIU)
directly, and is best effort throughout — every failure path just
logs and lets the restore produce the CRIU error above, which at least names
the path. `/tmp` is not shared with the dumping node, so on a re-fetched cache
this always has work to do.

**The image's config wins, silently.** `criu_restore` compares the baked and
requested `vllm_config` by exact dict equality and raises on any difference, so
the adapter passes `baked` — read straight from `meta.json` — and it matches by
construction. The consequence is that the job's own `vllm_config` has no effect
at all: ask for `gpu_memory_utilization: 0.8` against the 0.7 image and you
silently get 0.7. `_log_config_divergence` logs the baked config and warns on
every requested key it overrides, which is the only signal you get. Real
validation needs a richer compatibility model and is still deferred.

Importing the library is a module-level one-liner; the prototype's `SEMI_P_SRC`
`sys.path` hack is gone:

```python
from arctic_platform.inference.semi_persistence import Instance
```

The package's lazy `__getattr__` inserts the package directory at `sys.path[0]`
on first attribute access, which is what makes the flat sibling imports inside
the library resolve. Submodule-qualified imports still do **not** work:
`import arctic_platform.inference.semi_persistence.instance` raises
`ModuleNotFoundError: semip_logging`, because that path runs `instance.py`
before any attribute access has installed the `sys.path` entry. The class you
get back is `instance.Instance`. See the corrected note in
[`SKILL.md`](SKILL.md).

---

## 5. The restore sequence

This is the part most likely to rot, because it encodes an ordering contract
across a dozen primitives. The authority is
[`reproduce/example_full.py`](../reproduce/example_full.py), the saved-to-up
sweep that runs at TP1/2/4/8:

```python
inst.criu_restore().wait()
inst.cuda_restore(gpus=gpus).wait()
inst.reinit_nccl().wait()
inst.attach().load_weights().wait()
inst.wake_up_weights().wait()
inst.repin().plan_restore_weights().wait()
inst.restore_weights().wait()
inst.wake_up_kv_cache().wait()
inst.rebind_graphs().wait()
```

The adapter must follow this, not the shorter sequence the July prototype
used — and it must copy the `.wait()` boundaries too, which are part of the
contract rather than formatting. Four differences, each with real consequences:

| Difference | Consequence if skipped |
|---|---|
| `reinit_nccl()` after `cuda_restore` | TP>1 hangs or runs on dead NCCL comms. No-op at TP=1, so TP=1 testing will not reveal it |
| `plan_restore_weights()` before `restore_weights()` | `restore_weights` falls back to a single unbounded chunk, so the staging buffer is sized to the whole weight set and leaves no room for `wake_up_kv_cache` |
| `rebind_graphs()` after `wake_up_kv_cache` | The graphs keep the CustomAllreduce addresses they were captured with, which `reinit_nccl` has just invalidated |
| `attach()` / `load_weights()`, **waited on**, before `plan_restore_weights()` | The plan is computed with no budget and degrades to the unbounded path. See the correction below |

`plan_restore_weights` is the one that bites at 35B, and the reason is subtler
than "call it".

> **Correction.** An earlier draft called the `attach()` position "ordering
> only", and said the orchestrator's alternative placement
> (`criu_restore(image_dir).plan_restore_weights()`, planning once right after
> restore) was equally valid. Both are wrong for any image dumped the way ours
> are, and the failure is silent.

`plan_restore_weights()` computes its budget on the **handle**, synchronously,
at call time:

```python
pinned = self.max_pinned_bytes_per_worker or self.pinned_cpu_bytes
if self.total_gpu_bytes <= 0 or pinned <= 0:
    mb = None                      # the unbounded single-chunk fallback
else:
    allotment = int(self.total_gpu_bytes * gpu_memory_utilization)
    mb = int(0.9 * min(pinned, allotment - pinned))
```

Those inputs reach the handle by two routes, and only one of them yields a
usable number:

1. `criu_restore()` hydrates them from `image/meta.json`, synchronously in its
   own body. **Every image dumped after a `detach()` records zeros**, because
   `Instance._apply_result` zeroes both on `detach` — and detaching before the
   dump is exactly what puts the weights *outside* the CRIU image. All three
   baked images in Section 6 record `pinned_cpu_bytes: 0` and
   `max_pinned_bytes_per_worker: 0`.
2. The `attach` acknowledgement overwrites them with the live child's real
   sizes (`_apply_result`, `elif cmd == "attach":`).

`.wait()` is what makes route 2 visible: `Instance.wait()` blocks on the
demuxer's `wait_idle()`, which guarantees `_apply_result` has already run for
every command in the batch. So `attach().load_weights().wait()` followed by
`repin().plan_restore_weights()` works, and anything that plans before an
`attach` has been waited on gets `mb = None`.

For `qwen_35b` that is the difference between a ~29.6 GiB staging budget and
trying to stage all 65.4 GiB inside a 98.3 GiB allotment, leaving nothing for
`wake_up_kv_cache`. Because it fails as an OOM several steps later, the adapter
calls `_check_staging_budget` immediately before planning: it recomputes the
same budget, logs it, and raises if the pinned sizes are still zero. Do not
remove that guard to "simplify" the sequence.

Pass an explicit `max_buffer_bytes` only for images dumped before the
`empty_cache` fix — never as a workaround for the above. See the primitive
tables in [`reference.md`](reference.md).

`attach()` takes **no arguments**. The prototype called `attach(pool)`, which
is now a `TypeError`.

### Sleep and wake need a weight restore

`_wake_blocking` is not the mirror of `_sleep_blocking`. `sleep()` is
`llm.sleep(level=2)`, which **frees** the weight memory, and
`wake_up_weights()` only re-allocates the parameter tensors without
repopulating them. The wake path therefore needs three calls, matching
`Orchestrator`:

```python
self._inst.wake_up_weights()
self._inst.restore_weights()      # omitting this generates from garbage
self._inst.wake_up_kv_cache()
```

Omit `restore_weights()` and a model DSS slept and woke produces wrong output
with no error anywhere. No `repin()` here: the buffer stays pinned and the plan
cached at restore time still applies — which is another reason the guard above
matters, since a bad plan taints the wake path too. This is reachable in
normal operation: `collective_rpc("sleep")` and `("wake_up")` are both part of
the surface the DSS worker drives.

---

## 6. Where the image lives

`Instance(vllm_config, model_dir)` derives its paths from `model_dir`:

```
$SEMIP_IMAGE_CACHE/<config hash>_<environment hash>/
  image/          CRIU image of the whole child tree
  weights/        shards + weights_meta.json  (save_weights / load_weights)
  compilation/    torch/vLLM compile cache
```

**The image is bound to its `model_dir` and is not relocatable.** `criu_restore`
refuses to run under any other path:

```
model_dir mismatch: instance has X but image at Y was dumped with Z;
the image bakes absolute compile-cache paths, so it must be restored
under the same model_dir
```

Deriving the path from content is what satisfies that binding without anyone
maintaining it. The same config on the same container image and driver hashes
to the same name, and `$SEMIP_IMAGE_CACHE` is a fixed mount point, so the dump
and every later restore agree on an absolute path by construction — where a
hand-assigned path only agreed as long as an operator kept "one directory per
distinct config" true by hand. Do not move or symlink an image directory; let
it be re-derived, and re-dump if it is gone.

**A miss can now be filled by a copy instead of a cold start.** With
`SEMIP_IMAGE_SOURCE` set, `_materialize_from_source` copies `image/` and
`compilation/` out of `$SEMIP_IMAGE_SOURCE/<cfg12>_<env12>` into the derived
`model_dir` — the two directories bound to that absolute path — and leaves
`weights/` on the read-only mirror to be read in place. Every way it declines
is a cache miss that cold-starts, never an error, so nothing about it can fail
a job that would otherwise have run. See [`IMAGE_CACHE.md`](IMAGE_CACHE.md)
Section 9.

Three images are baked today. Note the layouts differ by TP degree:

| `model_dir` | Baked `vllm_config` | TP | Dumped on | Weights |
|---|---|---|---|---|
| `.../qwen_35b` | `Qwen/Qwen3.6-35B-A3B`, util 0.7 | 1 | `[7]` | 65.4 GiB, 33 shards, flat |
| `.../qwen_27b` | `Qwen/Qwen3.8-27B`, util 0.7, `max_num_seqs` 512 | 1 | `[5]` | 51.0 GiB, 26 shards, flat |
| `.../qwen_35b_tp2` | `Qwen/Qwen3.6-35B-A3B`, util 0.7, TP 2, `enable_expert_parallel` | 2 | `[2, 3]` | 65.4 GiB, per-rank |

At TP1, `weights/` holds `shard_*.bin` plus one `weights_meta.json`. At TP>1 it
holds `rank<N>/` subdirectories, each with its own shards and manifest
recording half the aggregate. `Instance._resolve_weights_dir` returns the
`weights/` parent either way and the child appends the rank segment, so
`load_weights()` needs no change — but tooling that reads
`weights/weights_meta.json` directly only works at TP1. The per-rank split is
also why `plan_restore_weights` prefers `max_pinned_bytes_per_worker` over the
TP-aggregate `pinned_cpu_bytes`.

Weights live *outside* the CRIU image here. That is optional in the library —
the orchestrator leaves staged weights inside the image — but the baked cache
is built with them separated, which is why the sequence in Section 5 includes
`load_weights()`, and why `meta.json` records zero pinned bytes. Layout details
are in [`semi-p_DESIGN.md`](semi-p_DESIGN.md).

`/data-fast` is node-local storage, not part of the pod image, so this
directory does not survive a pod change and has to be re-fetched.

This is a different thing from `/mnt/neutrino/base-models/<org>/<model>`, which
is where `dss/model_paths.py` resolves a `model_name` for a **cold** load. A
semi-p job does not read it; a non-semi-p job on the same pod does.

---

## 7. Ray zone bring-up

Semi-p work surfaced a zone bring-up bug that is independent of semi-p but
blocks testing it, so it is recorded here.

The head and its workers share one Ray temp dir, `/tmp/ray-<zone_id>`, and
`tear_down` pkills by matching that path in process cmdlines — so it kills the
head along with the workers. `bring_up_worker` called `tear_down` from its
except branch. On a single-node local zone the head's GCS can take 15 to 20
seconds to come up, so a worker that fails to join a still-starting head would
pkill that head, turning a transient timing failure into a zone that can never
become healthy.

The fix scrubs only worker leftovers, filtering cmdlines on `--address=` and
excluding `--head`, and leaves the head alive for the health-poll loop to
retry. Only `bring_up_worker`'s call site changes; the ones in `bring_up_zone`
and `bring_up_head` legitimately roll back what the same call created.

The timeouts move from 60 to 180 seconds in both `zones/cluster.py` and
`placement/local.py`, because the outer subprocess timeout has to exceed Ray's
own budget (`RAY_gcs_rpc_server_connect_timeout_s`,
`RAY_raylet_start_wait_time_s`) or it fires first and guarantees the failure
path.

Note the shape of this bug: a kill scoped by a shared marker took out more than
its target. That is the same class of mistake as Complication 13 in
[`TEARDOWN_SCOPING.md`](TEARDOWN_SCOPING.md).

**The slow dashboard agent did not reproduce.** A worker's dashboard agent was
once measured at 18 to 25 seconds under DSS against 3 to 5 in a standalone
reproduction that matched topology, ordering, every `RAY_` variable,
`PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, resources and hostname — never
explained, and mitigated rather than fixed by waiting for the head and widening
the budget. On 2026-09-08, with the gateway running as root and the
dashboard-agent deps installed system-wide, three consecutive zones came
healthy in **under a second** (`Waiting for workers: 1 / 1` to `GET /health
200` inside 350ms). Two things differed at once, so this identifies no cause;
record it as evidence that the delay is environmental rather than inherent, and
do not treat the widened budget as load-bearing without re-measuring.

---

## 7a. A job whose engine spans nodes (`tensor_parallel_size` > GPUs per pod)

Supported as of 2026-10-06, and it changes the placement contract rather than
the config surface. A spec asks for it the ordinary way -- `semi_p: true`,
`n_gpus: 16`, `tensor_parallel_size: 16` -- and three things differ:

- **dss builds a different placement group.** `build_inference_pg(...,
  per_node=cfg.inference_config.semi_p)` produces `nnodes` whole-node bundles
  with `STRICT_SPREAD` instead of `world_size` single-GPU bundles with PACK.
  Semi-p places *pods*, not ranks: each node-partition is a CRIU image restored on its
  own pod driving that pod's whole GPU set, so there is no per-rank actor for a
  per-rank bundle to hold -- and PACK could satisfy the group on one node,
  where the node-partitions would contend for the same GPUs and the second node would
  never appear.
- **`ReplicaPool` creates a leader plus agents.** The leader
  `InferenceWorker` takes bundle 0 with `world_size / nnodes` real GPUs (not
  the 0-GPU coordinator the Ray-executor path uses) and each
  `SemipNodeAgent` takes its own bundle. `distributed_executor_backend` stays
  `"mp"`: a semi-p engine is restored rather than constructed, so forcing
  `"ray"` would hand vLLM a backend it never uses *and* change the config
  `criu_restore` compares byte for byte.
- **The dump and restore are joint**, keyed on a `dump_id` every node-partition
  carries. Any disagreement makes all of them cold-start together rather than
  restore a mismatched set, which would deadlock in its first collective.

`NCCL_SOCKET_IFNAME=^lo`, which dss puts in `extra_env` for multi-node jobs,
names no interface; the semi-p child overrides it with a real one, because it
has to bind sockets its dump can account for and close. See
[MULTINODE_TP16.md](MULTINODE_TP16.md).

**Publishing such a dump is not supported yet** (`node<k>/`; IMAGE_CACHE §1).

## 8. Not supported

- **Shared weight pool.** The prototype could restore weights from a pinned
  memory pool (`pool_daemon`) shared across instances. Those six `pool_*`
  modules are not in the published library, and `attach()` no longer accepts a
  pool, so the path is removed rather than carried as dead code.

  > **Correction.** An earlier draft said a job JSON setting `semi_p_pool`
  > "will be rejected by pydantic". It will not, for the reason in Section 2:
  > the key is retained as an extra with a warning, then dropped because
  > nothing pops it. Such a job runs as an ordinary semi-p restore.
  > `sampling-tp1-semip-pool.json` is redundant, not an error case.

- **Weight sync / RL.** `sync_weights`, `sync_weights_broadcast`,
  `sync_spec_weights`, and `close_weight_sync` are deferred; the adapter
  recognizes them as weight-sync RPCs and does not implement them. Semi-p is
  for sampling jobs.
- **LoRA adapters.** `lora_adapter_path` is popped and then ignored: the
  worker's `_load_checkpoint_lora_adapter` call sits *after* the semi-p early
  return, so `self.llm.add_lora` is never reached and the adapter does not
  implement it. A semi-p job that sets the field gets base serving with no
  error. Adding it means teaching the `Instance` child to load an adapter,
  which is the same scope as weight sync.
- **A tokenizer for a model that is only in the image.** `get_tokenizer()`
  loads from `engine_kwargs["model"]`, which dss has already resolved to
  `/mnt/neutrino/base-models/<org>/<model>` — a path a semi-p-only pod has no
  reason to have populated, since avoiding that download is half the point.
  Nothing on the plain sampling path calls it: it is reached only from
  `_maybe_init_reasoning_parser` (inert unless the job sets `reasoning_parser`)
  and from the action-mask builder (inert unless the request sets
  `return_action_masks`). Set either on a pod with an empty `/mnt` and the
  first request raises inside `AutoTokenizer.from_pretrained`, not at
  `/initialize`. Run `local_model_cache.sh` for that model, or fall back to the
  baked HF id, if those features are wanted.

  > **Resolved incidentally, under `DSS_ALLOW_REMOTE_MODELS=1`.** That switch
  > (added for the dump path — see the correction in Section 9) leaves
  > `engine_kwargs["model"]` as the HF id rather than a `/mnt` path, which is
  > something `AutoTokenizer.from_pretrained` can resolve through `HF_HOME`. The
  > limitation stands for a gateway run *without* that switch.
- **`reset_prefix_cache`.** `Instance` had this primitive in the prototype and
  it did not survive into the published layout, while the server still calls it.
  **Resolved: the adapter's method is a no-op returning `False`.** Tracing the
  callers shows this is safe rather than merely convenient — `worker.py`'s
  `reset_prefix_cache` is driven only from `replica_pool.py`, whose own callers
  are both `_reset_prefix_cache_after_weight_sync`, and there is no HTTP route
  for it anywhere. So weight sync is the only live caller, and weight sync
  already raises. If it ever lands, restore the primitive across `instance.py`,
  `worker.py` and `vllm_child.py` instead of relaxing the no-op.
- **TP>1.** Section 5's sequence covers it and `_resolve_physical_gpus` now
  returns the full GPU list, so the count check passes; `qwen_35b_tp2` and
  `sampling-tp2-semip.json` exist to exercise it. Untested through DSS until
  that runs. The library's TP support is described in
  [`tp_DESIGN.md`](tp_DESIGN.md).

---

## 9. Running it

**What does not survive a pod.** `/data-fast` is node-local NVMe, so a fresh
pod starts with none of it: the image cache has to be re-dumped (there is no
backup and none is wanted), the `criu` file capabilities have to be re-applied,
and the system-wide pip installs have to be redone. See
[`INSTALL.md`](INSTALL.md) for the last two. Nothing else needs recreating:
both driver scripts now live beside the job JSONs in `dss-client`, and each
derives its paths from its own location, so a checkout on shared storage is
runnable on a fresh node with no symlinks and no absolute-path edits.

**Check out the integration branch, not a PR branch.** The two `dss-platform`
changes are deliberately independent — one carries the zone bring-up fix, the
other the `semi_p` config — so neither alone can run a job: the ray fix has no
`semi_p` fields, and the config branch has no head-wait. A test-only merge of
the two is what to have checked out. The symptom of getting this wrong is not
an error but a **silent cold start**, so confirm by timing: a restore reaches
`RUNNING` in seconds, a cold load takes about a minute. `grep` the gateway
output for `received unknown field` to confirm the flag was dropped.

Two terminals, against a pod whose node-local cache holds images dumped by
whoever will run the gateway (Section 6 — the cache does not ride in the pod
image). What follows is **configuration B**, everything as your own
unprivileged account, which is what the sampling JSONs here are wired for:
they point at `/data-fast/image-cache_neutrino`. Configuration A follows it.

```bash
# Once per pod: the Ray dashboard-agent deps. PEP 668 makes
# --break-system-packages mandatory on this image. Install them system-wide
# even under configuration B -- that is the one form that serves both, since a
# pip install --user as your own account is invisible to a root gateway.
sudo python3 -m pip install --break-system-packages --index-url https://pypi.org/simple \
    aiohttp-cors opencensus opentelemetry-exporter-prometheus

# Dump as yourself, no sudo, so the image records your uid and matches the
# gateway below. Run the two scripts concurrently -- that is what makes their
# images' task ids disjoint, and so their restores concurrent (CRIU_PLUMBING
# Complication 11).
cd /tmp && PYTHONPATH=/data-fast/dss_image/arcticinference-internal \
  python3 .../semi_persistence/scripts/test_weights.py    # qwen_27b + qwen_35b
cd /tmp && PYTHONPATH=/data-fast/dss_image/arcticinference-internal \
  python3 .../semi_persistence/scripts/test_tp2.py        # qwen_35b_tp2

EX=/data-fast/dss_image/dss-client/examples/local-gateway

# terminal 1 -- no sudo: the parent, the restored child and /tmp/inst<N>.log
# all land at your uid.  The script derives its checkout root from its own
# location, so DSS_IMAGE_ROOT only needs setting to override it, and it
# exports PYTHONPATH and SEMIP_UNPRIVILEGED itself.
bash $EX/run_dss-gateway.sh

# terminal 2 -- plain curl + jq, uid irrelevant
SAMPLING_JSON=$EX/sampling-tp1-semip-27b.json bash $EX/run_local_sampling.sh
```

Verified end to end on 2026-09-10 as `neutrino` (uid 1000) with no `sudo`
anywhere: a TP2+EP dump, then restores of all three models through the gateway.

**Configuration A is the same recipe as root.** Put `sudo` in front of both
dump scripts and in front of `run_dss-gateway.sh`, and point each JSON's
`semi_p_model_dir` at a root-dumped cache such as `/data-fast/image-cache`.
Nothing else changes — in particular `SEMIP_UNPRIVILEGED=1` stays set, because
it is the dump-time capability drop that makes the image restorable at all on
a cap-prod node, and `run_dss-gateway.sh` exports it either way.

Two things make A the less appealing of the two. It is further from production
DSS, where the Ray actor is not root. And on a cap-prod node `sudo` buys no
capability at all — root's bounding set there also lacks `CAP_SYS_ADMIN`, so A
is a way of matching uids, not a way of gaining privilege.

What you must not do is mix them. A root gateway against a neutrino-dumped
image is rejected up front by the `uid` check in `Instance.criu_restore`;
against an image predating that field it instead fails late, in `rebind_log`,
for the reason the blockquote below sets out.

Nothing needs to populate `/mnt/neutrino/base-models` for a semi-p job. The
sampling path never calls `check_model_existence` (only `training.py` does), and
`restore_and_wrap` uses `engine_kwargs` in just two places: `_log_config_divergence`,
which logs rather than enforces, and `tokenizer_path`, which is lazily loaded and
only forced by `reasoning_parser` or `return_action_masks`. So the
`/mnt/neutrino/base-models/...` path that `sampling.py` resolves is never opened —
everything driving the restore comes from the image's `meta.json`. Running
`local_model_cache.sh` is therefore only for a cold-load comparison, and adding a
job that needs either tokenizer-dependent feature is what would change that.

> **Correction — true of a restore, false of a dump.** This paragraph is the
> assumption the gateway dump path breaks, and it is the expensive one to
> rediscover. A semi_p job whose image is *missing* now cold-starts vLLM, and a
> cold start needs the weights. The path `sampling.py` resolves does get opened,
> on exactly the run that has no image to read instead.
>
> On this pod that path cannot be made to exist: `/mnt` is root-owned and the
> `neutrino` account has no sudo entry, so `local_model_cache.sh` cannot run
> (its own header now says so). The way out is `DSS_ALLOW_REMOTE_MODELS=1` plus
> `HF_HOME`, which leaves `resolve_model_path` returning the bare HF id and
> sends vLLM to the hub — the same thing `semi_persistence/scripts/test_*.py`
> have always done, and what keeps gateway-dumped and script-dumped images
> agreeing on the one config key a restore cannot reconcile by filtering.
> `run_dss-gateway.sh` exports both. Production DSS deliberately sets neither,
> so a production dump path needs the baked root populated instead;
> [`dss_dump.md`](dss_dump.md) Section 2 has the trade-off.
>
> Two consequences of that switch, both good: the tokenizer limitation in
> Section 8 below goes away (an HF id is something `AutoTokenizer` can resolve),
> and a re-dump after deleting an image costs a dump but not a download.

> ## Dump and restore under one uid
>
> **The dumped child and the DSS worker must be the same user.** This is the
> single most important operational constraint on this page. It is satisfiable
> two ways, and the mistake to avoid is meeting it halfway.
>
> Cross-uid pairing is **out of scope, not merely discouraged**: `meta.json`
> records the dumping `uid` and `Instance.criu_restore` raises on a mismatch
> before it spawns a worker. Configuration A (`root -> root`) and configuration
> B (`unprivileged -> unprivileged`) below are the two supported shapes.
>
> The restored child keeps the uid recorded in the image, and
> `SEMIP_UNPRIVILEGED=1` leaves it with an empty capability set — it drops
> capabilities and does *not* change uid. So a root-dumped child is `uid 0`
> with **no `CAP_DAC_OVERRIDE`**: root in name, subject to ordinary mode bits
> in fact. Pair that with an unprivileged `Instance` worker and both identities
> write the same files with neither able to write the other's. Concretely,
> `rebind_log` runs *in the child* (`vllm_child.py`) and opens
> `/tmp/inst<current_id>.log`, a file the parent created:
>
> ```
> command 'criu_restore' failed: rebind_log failed:
>   PermissionError: [Errno 13] Permission denied: '/tmp/inst0.log'
> ```
>
> No file mode fixes a split like that. CRIU checks the path's mode against the
> dump and demands `0644`, at which point only the owner may write, and
> widening to `0666` trades the error for `bad mode 0100666 (expect 0100644)`.
> The same squeeze appears on `compilation/` during `recapture_graphs()`. The
> answer is not a mode — it is to stop having two identities.
>
> **Configuration A — everything as root.** Run the dump scripts under `sudo`
> *and* the gateway under `sudo`. Verified end to end on 2026-09-08 for 27B
> TP1, 35B TP1 and 35B TP2+EP: `rebind_log` completes in 0.000s, and none of
> the group-write juggling below is needed, because one identity writes
> everything. Its one non-obvious prerequisite is that the Ray dashboard agent
> deps must be installed **system-wide** — a `pip install --user` as the
> unprivileged account is invisible to root, and PEP 668 on this image needs
> `sudo python3 -m pip install --break-system-packages`. Without them the
> worker raylet aborts in `WaitForDashboardAgentPorts`.
>
> **Configuration B — everything unprivileged.** Preferable, because it is what
> production DSS looks like (the Ray actor is not root), but it needs one more
> thing than you would expect: `criu` itself must run as the target's uid, not
> under `sudo`. criu reads every rlimit of its target with `prlimit()`, and the
> kernel's `check_prlimit_permission()` allows a cross-uid read only to
> `CAP_SYS_RESOURCE`. On a cap-prod pod that capability is outside the
> **bounding** set, so nothing can acquire it, and a root criu dumping an
> unprivileged child dies before writing any image at all:
>
> ```
> Error (criu/cr-dump.c:389): Can't get rlimit 0: Operation not permitted
> Error (criu/cr-dump.c:1584): Dump core (pid: N) failed with -1
> Error (criu/cr-dump.c:1975): Dumping FAILED.
> ```
>
> Note what this is *not*: it is not about who owns the image files. Dropping
> the `sudo` in `_worker_criu_save` is what fixes it, and that is now the
> implemented default for both configurations — criu runs at the worker's own
> uid and takes its capabilities from the binary:
>
> ```bash
> sudo setcap cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip \
>     /usr/sbin/criu
> ```
>
> The dump needs only the first two. `cap_setpcap` and `cap_setgid` are for the
> restore's `restore_creds()`, which otherwise deadlocks — see Complication 11
> in [`CRIU_PLUMBING.md`](CRIU_PLUMBING.md).
>
> Keep the `+e`: without the effective bit a real-uid-0 `execve` would land
> with an empty effective set, breaking criu for a *root* worker. With it,
> both callers work — `capabilities(7)` treats the file's permitted set as
> all-ones for uid 0. Measured this way, an unprivileged criu dumps cleanly
> (`exit=0`, zero error lines) and records `uid=1000 gid=1000 cap_eff=[0, 0]`.
>
> Verify before trusting an image:
>
> ```bash
> sudo crit decode -i <model_dir>/image/core-<pid>.img | python3 -c "
> import json,sys; c=json.load(sys.stdin)['entries'][0]['thread_core']['creds']
> print('uid=%s gid=%s cap_eff=%s' % (c.get('uid'), c.get('gid'), c.get('cap_eff')))"
> ```
>
> The uid must match whoever runs the gateway — `uid=0` under configuration A,
> your own uid under B. `cap_eff=[0, 0]` is required either way: without it the
> cap set was not dropped and restore dies in `criu/pie/restorer.c` with
> *"Unable to restore capabilities"*, fixable only by re-dumping. So keep
> `SEMIP_UNPRIVILEGED=1` set for the dump regardless of configuration.

**The rest of this section is historical.** It describes the mixed
configuration, which is now rejected by the `uid` check in
`Instance.criu_restore` and can no longer be reached with an image that
records a `uid`. Under either supported configuration one identity owns and
writes everything, none of the ownership or mode juggling below is needed, and
the two `find` invariants stop mattering. Read on only to interpret failures in
older logs, or if you are working with a pre-`uid` image that predates the
check.

**Two identities write into the cache, and they are not the same user.** Get
this wrong in either direction and the restore fails late, with a message that
does not mention permissions until you read the traceback.

The first is the **unprivileged parent** — the `Instance` worker, running as
whoever launched the gateway. It unlinks stale pidfiles in `image/` (below).
The second is the **restored vLLM child**, which is `uid 0` but carries an
*empty capability set*: `SEMIP_UNPRIVILEGED=1` makes the child drop all
capabilities before the dump precisely so `restore_creds()` can succeed on a
low-capability node, and `crit decode -i core-<pid>.img` confirms it
(`uid: 0`, `cap_eff: [0, 0]`). Without `CAP_DAC_OVERRIDE` that process is root
in name only: it is subject to ordinary permission checks, and it writes the
Triton/Inductor cache under `compilation/` during `recapture_graphs()`.

So making the tree user-owned — the obvious move after a root-owned download —
breaks the child:

```
RuntimeError: command 'recapture_graphs' failed: PermissionError:
[Errno 13] Permission denied: '<model_dir>/compilation/triton/YBWZ2PX...'
```

`root:<group>` with `g+w` satisfies both: the child writes as owner, the parent
writes as group. Note this failure lands on the *last* call in the restore
sequence, after the weights are already in place, so a run that gets all the
way to `recapture_graphs` and dies is very likely this and not an ordering bug.

**Grant that write on directories only.** A `chmod -R g+w` also rewrites file
modes, and CRIU checks the mode of every file it re-maps against what the dump
recorded, so the very next restore dies far earlier and for an unrelated-looking
reason:

```
Error (criu/files-reg.c:2294): File .../compilation/flashinfer/.../sampling.so
File r/w/x checks can be skipped with the --skip-file-rwx-check option
Error (criu/mem.c:1467): `- Can't open vma
Error (criu/cr-restore.c:2331): Restoring FAILED.
```

Only directories need the bit — the parent unlinks, the child creates. Two
invariants worth asserting before a run, both of which should print 0:

```bash
find /data-fast/image-cache -type f -perm -g+w | wc -l   # files must NOT be g+w
find /data-fast/image-cache -type d ! -perm -g+w | wc -l # dirs must be g+w
```

Expect both invariants to drift, so re-assert them between runs rather than
once. `_worker_criu_load_lowcap` ends with
`sudo chown -R <uid>:<gid> <image_dir>`, so `image/` is re-owned to whoever ran
the last restore — a root-run `restore_smoke.py` leaves it `root:root` and locks
the unprivileged parent out again. Separately, a *successful* restore has the
root child create fresh `compilation/triton/<hash>/` directories under its own
umask, so they arrive `0755` and the next unprivileged run cannot write them.
The repair is the same two lines as the prep:

```bash
sudo chgrp -R "$(id -gn)" /data-fast/image-cache
sudo find /data-fast/image-cache -type d -exec chmod g+w {} +
```

The parent's half matters the moment anything has already restored these images
as root — a `restore_smoke.py` run, say, or a re-download over a cache a
previous root run wrote into. `_worker_criu_load_lowcap` unlinks a *stale*
`restored.pid` and `restore.log` from `<model_dir>/image/` in the
**unprivileged parent**, before it invokes `criu`:

```python
for _stale in (pidfile, os.path.join(image_dir, "restore.log")):
    if os.path.exists(_stale):
        os.remove(_stale)
```

`os.path.exists` succeeds on a root-owned `0755` directory, but `os.remove`
needs write permission on the *directory* and raises `PermissionError` — which
nothing catches, since the retry loop around `_load_fn` only handles
`RuntimeError`. The restore dies there, before `criu` ever runs.

On a cache with no stale files root ownership is harmless: `criu` creates both
files as root and the library immediately runs
`sudo chown -R <uid>:<gid> <image_dir>`, which self-heals the directory for
every later attempt. So the hazard is specifically a root-written cache, and
`ls -l <model_dir>/image/restore*` tells you in one command whether you have
one.

`run_local_sampling.sh` lives only under `/data-fast/dss`; there is no
`dss_image` copy. That is fine — it takes `SAMPLING_JSON` from the environment
and only its `GENERATE_JSON` default stays in the July tree, which is a plain
prompt list — but it is the one path in this document that still points at the
old checkout.

Three job JSONs exist, each naming its own image explicitly because
`semi_p_model_dir` is required. Run them in this order — the cheap ones fail
faster and on different things:

| JSON | Image | What it proves |
|---|---|---|
| `sampling-tp1-semip-27b.json` | `qwen_27b` | The plumbing, on the smallest weight set |
| `sampling-tp1-semip.json` | `qwen_35b` | The staging budget is real; a `None` plan may survive 27B but OOMs here |
| `sampling-tp2-semip.json` | `qwen_35b_tp2` | TP>1: the GPU list, `reinit_nccl()`, the per-rank weights |

To tell "the adapter is wrong" apart from "the zone never came up", run
`/data-fast/dss_image/restore_smoke.py` first. It drives `restore_and_wrap`
through the same entry point and the same `engine_kwargs` shape the worker
uses, with no gateway, no zone and no Ray actor:

```bash
sudo env PYTHONPATH=/data-fast/dss_image/arcticinference-internal \
    SEMIP_UNPRIVILEGED=1 python3 /data-fast/dss_image/restore_smoke.py qwen_27b --gpus 0
```

`run_dss-gateway.sh` carries four things that matter. It puts the source trees
on `PYTHONPATH` ahead of the installed wheels, and propagates that to the Ray
head, worker, and actors via `DSS_PROPAGATE_ENV_EXTRA` — without which the
`InferenceWorker` actor would import a different `arctic_inference` than the
driver. It raises Ray's own connect and raylet-start budgets. It sets
`RAY_enable_open_telemetry=0`, because Ray 2.55 defaults the exporter agent on
while the image ships neither `aiohttp-cors` nor `opencensus`, and the worker
raylet dies in `WaitForDashboardAgentPorts`, collapsing the zone.

> **Correction, observed 2026-09-08.** `RAY_enable_open_telemetry=0` plus
> `--include-dashboard=false` is **not sufficient**, and the residual failure
> looks nothing like a metrics problem. `--include-dashboard=false` is only
> passed to `_ray_start_head`, and on a worker node it would not help anyway:
> the *dashboard agent* starts on every node regardless, and the raylet exits
> if it cannot come up. So the head survives and the worker raylet dies
> milliseconds after starting, which surfaces as:
>
> ```
> Exception: The current node timed out during startup. This could happen
> because some of the raylet failed to startup or the GCS has become overloaded.
> ```
>
> — a message that points at the GCS, which is healthy. The real evidence is
> elsewhere: a zombie raylet parented to `ray start --address=`, a
> `runtime_env_agent.log` that reads `Parent raylet pid is <N>` followed
> immediately by `Raylet is dead!`, and a zone stuck at
> `Waiting for workers: 0 / 2 GPU(s) across 0 node(s)` until it tears down.
>
> The cause is three imports the image is missing. Check them directly rather
> than inferring from the raylet:
>
> ```bash
> cd /tmp && python3 -c "import ray.dashboard.http_server_agent, \
>     ray.dashboard.modules.reporter.reporter_agent"
> ```
>
> On this image that fails on `aiohttp_cors`, then `opencensus`, then
> `opentelemetry.exporter.prometheus`. Installing all three fixes it:
>
> ```bash
> pip install --index-url https://pypi.org/simple \
>     aiohttp-cors opencensus opentelemetry-exporter-prometheus
> ```
>
> The default index is an internal Artifactory mirror that fails SSL
> verification with a hostname mismatch and does not carry these packages, so
> `--index-url` is required; `--trusted-host` alone does not help. They land in
> `~/.local`, which is fine because every Ray daemon runs as the invoking user
> — but note a root-run harness would not see them. No gateway restart is
> needed, since the zone's Ray processes are spawned per job.

And it exports **`SEMIP_UNPRIVILEGED=1`, propagated to the actor**, which on a
pod like this one is not optional. Check the bounding set:

```bash
grep CapEff /proc/self/status; capsh --print | head -3
```

`cap_checkpoint_restore` present, `cap_sys_admin` absent means the
namespace-based restore path cannot run at all, and `SEMIP_UNPRIVILEGED=1`
selects `_worker_criu_load_lowcap` instead (`criu restore -d --unprivileged`,
no `unshare`). It matters that the *same* switch was set at dump time: CRIU
then recorded an empty capability set for every task, so `restore_creds()` has
nothing to reinstate. An image dumped without it dies in
`criu/pie/restorer.c` with *"Unable to restore capabilities"* and can only be
fixed by re-dumping. Because the variable is read inside the `Instance` worker,
it has to reach the Ray actor — hence its presence in
`DSS_PROPAGATE_ENV_EXTRA` and not merely in the driver's environment.

Two things about that check read worse than they are, and both are worth
knowing before someone concludes the node is unsuitable.

**The gateway does not need to run as root, nor to hold sudo**, even though the
whole tree used to. `_worker_criu_load_lowcap` is built for an unprivileged
parent and invokes `criu` at the worker's *own* uid — it must, since criu has
to match its target's uid (Complication 11) — so there is no `sudo` on that
path at all. The helper is still a separate process reached over a Unix socket
with `SCM_RIGHTS`, because `subprocess` strips fds ≥ 3 regardless of privilege.
`crit` likewise runs directly. What the node does need is
`SEMIP_UNPRIVILEGED=1` and the capabilities set on the binary:
`setcap cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip /usr/sbin/criu`.
The Ray actor's uid does matter in one respect: it must equal the uid that
dumped the image.

**The CUDA checkpoint calls are not elevated at all**, as of 2026-09-08.
`_worker_checkpoint`, `_worker_restore` and the inline checkpoint
in `_child_thread` each used to branch on `euid == 0`, calling
`cuCheckpointProcess*` in-process as root and shelling out to
`sudo cuda-checkpoint` otherwise. Root was never the requirement: the driver
needs ptrace permission over the target, which a parent has over its own
same-uid child.

Measured from an unprivileged caller (`euid=1000`) against its own CUDA child,
all four calls return 0 — `Lock`, `Checkpoint`, `Restore` and `Unlock` — and
the side effects confirm they do the work rather than no-op: the child's device
memory disappears on checkpoint and returns on restore, the child is frozen
while checkpointed, and it resumes computing correct results after unlock. The
migrating branch, where `_worker_restore` passes a non-NULL
`CUcheckpointRestoreArgs`, is separately covered by every `restore_smoke.py`
run that restores onto GPUs other than the dumped ones.

That gate mattered here because **`cuda-checkpoint` is a separate binary, not
part of the driver, and this pod image does not ship it** — `find / -xdev`
turns up nothing. So on the old code an unprivileged DSS worker failed with
`cuda-checkpoint: command not found` at `cuda_checkpoint()` during dump and at
`cuda_restore` during restore, for an operation the driver would have performed
in-process. The root-run `restore_smoke.py` never hit it, because root took the
API branch — which is exactly why this stayed hidden until the DSS path got far
enough to reach it.

The CLI fallback is now removed rather than fixed, along with
`_build_full_device_map`, which only existed to render the device bijection as
a `--device-map` string. The API path builds the same mapping as a
`CUcheckpointRestoreArgs` struct via `_build_restore_args`, so cross-node GPU
migration is unaffected. Full reasoning in
[`CRIU_PLUMBING.md`](CRIU_PLUMBING.md) Complication 6.

**`sudo criu check --unprivileged` exits 1 here, and that is a false alarm.**
The only error it reports is:

```
Error (criu/cr-check.c:160): sys/kernel/ns_last_pid sysctl is inaccessible: Read-only file system
```

`/proc/sys` is mounted `ro` in this pod. The low-cap restore never writes that
sysctl: CRIU places each task at its recorded id with `clone3(set_tid)`, which
in the host PID namespace is authorized by `CAP_CHECKPOINT_RESTORE` alone —
the reasoning is spelled out in the comment block above
`_worker_criu_load_lowcap` in `worker.py`. Read the check's *output* rather
than its exit code, and treat a lone `ns_last_pid` line as pass. Note also
that this pod's bounding set has `cap_checkpoint_restore` but **not**
`cap_sys_ptrace`, despite the floor quoted for this path; what actually matters
is that `criu` runs as root via `sudo`, where the bounding set is the effective
set.

One consequence of the low-cap path worth knowing before debugging a failure:
without a private PID namespace, **every task id recorded in the image must be
free on the host**, threads included (a TP2 image needs ~900 ids). That caps
the node at one live restore at a time, and an unrelated process whose threads
span the range blocks it. `criu_restore` preflights this and names the
occupant; `scripts/pidcheck.py <image_dir>` answers it before you spend an
attempt.

### The neutrino zone pod, measured 2026-09-16

Everything above in this section was measured on the research pod, where the
tree runs under `sudo` on a node you provisioned. The **neutrino sampling
zone's device-manager pod** is the production DSS target — semi-p is enabled
on a cluster by setting `--enable-criu-dump` on that cluster's ZMD — and it
differs in ways that change both what you check and what
a correct pod spec looks like. Measured on
`dz-<zone>-s0-device-manager`, kernel `6.12.92-122.166.amzn2023.x86_64`,
criu `4.2.1`.

**The process holds no capabilities; the criu binary does.** This is the single
thing most likely to be got wrong when writing a new pod spec, because a spec
that looks correct fails anyway. The operator renders:

```json
{"allowPrivilegeEscalation": true,
 "capabilities": {"add": ["CHECKPOINT_RESTORE", "SYS_PTRACE"], "drop": ["ALL"]},
 "runAsNonRoot": true, "runAsUser": 1000, "runAsGroup": 1000}
```

and the container then reports:

```
CapPrm: 0000000000000000     CapEff: 0000000000000000
CapBnd: 0000010000080000     NoNewPrivs: 0
```

`capabilities.add` on a **non-root** container populates only the *bounding*
set — bits 19 and 40, `CAP_SYS_PTRACE` and `CAP_CHECKPOINT_RESTORE`. The
process's permitted and effective sets are empty. criu acquires its
capabilities solely from the file capabilities on the binary
(`getcap /usr/sbin/criu` → `cap_sys_ptrace,cap_checkpoint_restore=eip`) at
`execve`, and the kernel grants those only if **both** hold: the caps are
within the bounding set, and `NoNewPrivs` is 0.

So `allowPrivilegeEscalation: true` is **load-bearing**, not a relaxation
someone forgot to tighten. Setting it to `false` sets `NoNewPrivs=1`, which
makes `cap_bprm_creds_from_file()` refuse to honour the file capabilities;
criu then execs with an empty permitted set and fails `EPERM`, with a pod spec
that still lists both capabilities. That it is admissible here is not an
accident either: the `psp-allow-privilege-escalation-container` constraint
carries a `namespaceSelector` excluding namespaces labelled
`psp: permissive-sidecar-injection`, which `neutrino` has.

**The file capabilities must not exceed the bounding set.** The `setcap` quoted
earlier in this section grants four —
`cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip` — and that
binary **cannot exec at all** on this pod. `capabilities(7)`: when the file
effective bit is set, any capability in the file's permitted set that is not in
the process's new permitted set makes `execve` fail with `EPERM`. `cap_setpcap`
and `cap_setgid` are outside this bounding set, so they are masked out and the
exec fails — presenting as `PermissionError: Operation not permitted:
'/usr/sbin/criu'` before criu emits a single line. The neutrino image therefore
ships the narrowed two-cap form, and an image built with the four-cap `setcap`
is unusable here no matter how the pod grants capabilities.

**No `sudo`, and `cap_sys_ptrace` is present.** The note above — that the pod's
bounding set lacks `cap_sys_ptrace` and that what matters is criu running as
root via `sudo` — describes the research pod and does not carry over. Here
criu runs at uid 1000 with no `sudo`, `cap_sys_ptrace` is in the bounding set,
and `criu check --unprivileged` should be run **without** `sudo`: `sudo` on this
pod would only produce a root shell whose bounding set is still the same two
caps. The `ns_last_pid` false alarm above does reproduce exactly — `/proc/sys`
is mounted `ro,nosuid,nodev,noexec`, the sysctl reads fine and cannot be
written, and `clone3(set_tid)` does not need it.

**The pod must carry the Gatekeeper label.** The device-manager pod is labelled
`neutrino.snowflake.com/criu: dump`, and the `NeutrinoCriuLabelIntegrity`
constraint governs who may set it. The label is what the capabilities ride on,
so a device-manager pod *without* it has silently not been opted in — check the
label rather than merely that the pod started, because a pod lacking the label
is admitted trivially, the policy never matching. Note this constraint matches
Deployment, ReplicaSet **and** Pod, so one labelled Deployment produces three
admission reviews under three different creator identities; authorizing only
`neutrino-sa` is not enough, and the ReplicaSet path needed
`k8sc-platform#110367` before any of this could run.

### The dump-then-restore cycle collides with itself

`/data-fast` is node-local NVMe with no backup, so a fresh zone pod has no image
and every cold start takes `restore_and_wrap`'s **miss** path: cold-start, dump,
then restore that image in the same process ~4.8s later, needing back the very
ids the dump just killed. `_dump()` anticipates the overlap and budgets
`time.sleep(2)` for it, commented "so the two do not overlap".

**Two seconds is enough on the research pod and not here, and the difference is
PID 1.** This is worth stating plainly because everything people usually suspect
is identical: `neutrino_prod.yaml` mirrors the DM's `securityContext` exactly
(drop `ALL`, add the same two capabilities, `allowPrivilegeEscalation: true`,
uid 1000), and the pod runs a byte-identical wheel — `worker.py` and
`semip_engine.py` both md5-match the checkout. The same gateway dump/restore
cycle succeeds there.

What differs is who reaps the killed leader:

| | research pod (`neutrino_prod.yaml`) | neutrino zone DM pod |
|---|---|---|
| PID 1 | `/bin/bash -c '… sleep infinity'` | `/usr/bin/python /usr/local/bin/dss-zone-worker` |
| reaps orphans | yes — bash waits with `waitpid(-1)` while running `sleep infinity`, so it reaps anything reparented to it | **no** |

Measured on the DM pod: an orphaned `sleep 3` was still `State: Z (zombie)`,
`PPid: 1` seven seconds later, with six zombies resident. So on the research pod
the dumped leader is reaped within milliseconds of the destructive dump and its
id is free long before `time.sleep(2)` elapses; on the DM pod nothing reaps it,
and its id is still held 9.4s later when the restore's retry budget runs out.

The consequence for anyone reproducing this: **a research pod running `bash` as
PID 1 cannot reproduce the failure, no matter how exactly it mirrors the
capability set.** Reaping behaviour, not capabilities, is the variable.

Observed twice, on two different zones and two different nodes, with
`Qwen3.6-35B-A3B` TP1 images of 51 task ids each:

```
                      zone A                    zone B
CRIU saved, child killed    00:01:55.061              00:20:26.131
dump worker exits           00:01:55.064  (+3ms)      00:20:26.133  (+2ms)
>>> criu_restore            00:01:59.834  (+4.77s)    00:20:30.903  (+4.77s)
task-id collision 1/5 .. 4/5
<<< criu_restore FAILED     00:02:04.531  (4.697s)    00:20:35.550  (4.647s)
Error (cr-restore.c:1237)   Can't fork for 1036       Can't fork for 1263
```

Identical to the millisecond, which is what makes this structural rather than
luck. **The id criu cannot get is the thread group leader the dump just
killed** — but the thing holding that id is not the leader's own corpse, and
getting this wrong cost a merged commit that fixed nothing.

### It is a process-group reference, not the leader's corpse

A pid is a refcounted `struct pid` in the namespace's IDR, freed only when its
last reference goes. References come from the task itself **and** from every
process using it as a `PGID` or `SID`, because a process group is named by its
leader's pid. An unreaped zombie keeps those links. So an id can be unused by
any task and still be unallocatable — `clone3(set_tid=N)`, which is how CRIU
places a restored task, fails `EEXIST` while `/proc` shows nothing at `N`.

The child `setsid()`s at startup (fd 0 → `/dev/null`, so no tty enters the
image), which is what makes its pid a `PGID` and `SID` as well as a pid. Every
process it forks then holds two references on it. Observed on a
device-manager pod:

```
pid    pgid   sid    comm      state
1302   1233   1233   python    Z (zombie)   PPid: 1
1304   1233   1233   python    Z (zombie)   PPid: 1

no task at pid 1233        -- /proc/1233 does not exist
crit: image leader = 1233  -- 51 tasks, ids 1233-1499
criu: Error (criu/cr-restore.c:1237): Can't fork for 1233: File exists
```

Two experiments pin it down. A restore with **no preceding dump**
(`/tmp/isolate_restore.py`, all 51 ids verified free first) fails identically,
so the holders outlive the dump that created them and the collision is
permanent rather than a race. And continuous sampling of `/proc/<pid>/task`
every 2ms across the failing restore logged **zero** occupancy of all 51 ids —
the signature of a reference rather than a task.

### Why the first two fixes did not work

`e0ec41a` and `e0df4f8` both reap the dump worker's own child. That is
necessary — the leader is a zombie for as long as nothing collects it — but
insufficient, because 1302 and 1304 are **grandchildren**. Their parent is
gone, so they reparent to PID 1, and `waitpid` from the worker answers
`ECHILD`. The reap logged `reaped dumped child 1233 (exit -9)` in 2ms and the
restore still failed on 1233.

Nor is the retry loop the answer, though its comment reads as if it should be:
it says retries "only rescue a short-lived holder (a zombie being reaped)",
and `max_retries = 5` at 0.5s gives ~4.7s. Nothing in that pod will ever reap
1302 and 1304, so no budget is long enough.

### The fix that addresses the right object

`_set_child_subreaper` sets `PR_SET_CHILD_SUBREAPER` at worker start, before
any child exists, so orphaned *descendants* reparent to the worker instead of
PID 1 — which is the only thing that brings them within `waitpid`'s reach.
`_reap_orphaned_descendants` then sweeps with `waitpid(-1, WNOHANG)` after the
dump, strictly after `_reap_dumped_child` so the sweep cannot consume the
status `multiprocessing` needs. Each reap drops a reference; the last one frees
the id. Both are bounded, both are outside the handler's `try`, and neither can
fail a dump whose image is already on disk.

`prepare_criu_dump` also sweeps in the **child**, before the tree is frozen.
That covers the other origin: a stray that was already dead at dump time is
not in the image and was never the worker's descendant, so only the child can
collect it.

Verified on a real kernel rather than reasoned about. `scripts/pidref_test.py`
reproduces the state in about a second, unprivileged, with no GPU and no criu:
a session leader with one member, reaped, leaves its id with no task and still
allocated (`killpg` succeeds, `/proc` is empty); the task scan is blind to it
and the pgid/sid scan names the holder; the subreaper adopts the member, the
sweep collects it, and the id frees immediately. It passes on a pod whose PID 1
*does* reap, which is the point — the fix no longer depends on PID 1.

**Confirmed live on the dev cluster**, job `2469a638-2290-44eb-897e-16075046c00a`, the
first semi-p job to complete the cycle:

```
07:35:31.822  CRIU saved to .../qwen_35b/image (child killed by dump)
07:35:31.823  reaped dumped child 1045 (exit -9)
07:35:31.824  reaped 2 orphaned descendant(s): 1074, 1076
07:35:31.824  <<< criu_dump OK (4.950s)
07:35:36.610  >>> criu_restore                                 (+4.79s)
07:35:39.794  CRIU restored pid=1045 (checkpointed, awaiting restore)
```

The restored leader landed at 1045 — the id 1074 and 1076 had been holding.
Two holders, matching the two seen in the failure, and the dump→restore gap is
unchanged from the failing runs, so nothing here was fixed by timing. The zone
manager then reported `"op": "create-job", "duration_s": 298.88, "error": null`
against `284.6` with `HTTPException: 500` the night before.

`reaped_strays=[]` in the same run also settles what those processes are: both
holders were **alive at dump time** and died with the tree, rather than being
zombies that predated it. The subreaper does the work in this path; the
child-side sweep covers the other origin.

What none of this fixes is a stray orphaned by something outside the library's
subtree. The same pod carried two unrelated `python` zombies at 1087 and 1089
with `PPid: 1` for its whole life; they fell outside this image's range and
cost nothing, but one inside it is unclearable by any reap we can perform. The
answer for those is a reaping init — `tini`, or `waitpid(-1, WNOHANG)` in
`dss-zone-worker` — which fixes strays from every source at no per-job cost.
Burning the PID counter dump-side was considered as an alternative and
**rejected** (2026-09-17): it moves a namespace-global counter for the whole
pod, costs 10–15s of fork churn per cold start, protects only what starts
after it, bakes into the image, and still only buys a window because the
counter wraps at `pid_max`. For the restore-side variant — a warm image whose
recorded ids overlap the restoring pod's own startup — the preferred fix is to
treat an unresolvable collision as a cache miss and **re-dump**, which places
the ids wherever the fresh tree lands and needs no privilege, no counter games
and no platform change.

**`pidcheck.py` and `_pid_collision_report` were blind to all of this** until
2026-09-17: both resolved occupancy purely from `/proc/<pid>/task`, so both
reported *"No collisions: every recorded task id is free on this node"* while
the restore failed on the first id it tried. They now also scan `pgrp` and
`session` (fields 5 and 6 of `/proc/*/stat`) and report a recorded id that is
referenced with no task of its own. A clean report from an older build is not
evidence of anything.

The failure is not contained to the restore. It propagates
`semip_engine._restore` → `InferenceWorker.initialize` →
`GPUSamplingJob.create`, and the zone manager fails the job outright —
`HTTPException: 500: Failed to create job` after `duration_s: 284.6`, followed
immediately by `destroy-job`. **`ct get` is not a reliable signal here:** it
still reported `INITIALIZING` sixteen minutes after the zone manager had
destroyed the job. Read the zone manager's log, not the CLI.

When shadowing `arctic_inference` from a source checkout, that tree needs two
things the wheel puts *inside* the package but the repo keeps elsewhere:
`csrc/` (which `op_builder/builder.py` resolves relative to
`Path(__file__).parent.parent`) and the built `suffix_decoding/_C*.so`. A clean
clone on `PYTHONPATH` breaks custom ops and suffix decoding.

---

## 10. Provenance

**The checkout is no longer byte-identical to the wheel.** Until 2026-09-08 every
change lived in the adapter (`arctic_inference/server/`) and `semi_persistence`
itself matched the installed package exactly, which is why shadowing it from a
source tree carried no image-mismatch risk. The driver-API commit changes
`semi_persistence/worker.py`, so the two now differ: anything importing the
library from the wheel still takes the uid-gated `cuda-checkpoint` CLI path,
which fails on a pod that does not ship the binary. `PYTHONPATH` therefore
decides behaviour, not just line numbers — a bare `python3` gets the wheel.

Semi-p rides in the pod image as an ordinary wheel, not an editable install, so
the mapping from image to source is worth recording. For the image these notes
were written against:

| Package | Built from |
|---|---|
| `arctic_inference` 0.1.3 | `arcticinference-internal` `cc1be76`, branch `semi-p/sync` |
| `dss` 0.1.0 | `dss-platform` `c88bd360` |
| `arctic_training` 0.7.3.dev0 | `ArcticTraining-dss` `a92aa61` (`v0.4.8`) |
| `dss-client` 0.0.2 | `dss-client` `a3ea6fe` (`v0.4.4`) |

`arctic_inference` was built from a **branch, not main**: at build time
`semi-p/sync` was 25 commits ahead of `arcticinference-internal/main`, so the
image contained `semi_persistence` while `main` did not. PR #138 has since
merged, which makes that note historical — `main` now carries the library — but
the wheel still predates the merge, so `cc1be76` remains the only commit that
reproduces the installed package byte-for-byte. Whether `cc1be76` is itself
reachable from `main` depends on whether #138 was merged or squashed, which is
not yet verified here.

The `dss` wheel records no VCS provenance at all — it was installed from
`file:///home/neutrino` during the image build, so `direct_url.json` says only
that. `c88bd360` was recovered by blob-matching all 147 wheel-shipped files
against the repo's history. If you need to do that again: hash the installed
files with `git hash-object`, then walk `git rev-list` comparing
`<commit>:<path>`.

The wheel is not a faithful copy of the repo tree. It omits `README.md` files,
`tests/`, `scripts/`, `skills/` (this document included), and `assets/`; it
relocates `csrc/` to inside the package; and it adds two generated protobuf
modules that are not in git. Compare with those exceptions in mind.
