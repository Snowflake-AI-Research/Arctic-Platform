# DSS integration: dumping from the gateway

Status: **implemented and exercised end to end** on 2026-09-13, at TP1 (27B) and
TP2+EP (35B), each dumped then restored through the gateway. Sections 1-10 are
the design as written beforehand and are left as they were; **Section 11 records
what was actually built and Section 12 what the measurements changed**, so where
they disagree with the earlier text, the later sections win. Read
[`dss_integration.md`](dss_integration.md) first -- everything here assumes its
Sections 1-6 and 9.

Headline numbers, first line of the zone log to the first generate:

| Run | Zone | Init | Dump | Restore | Generate | Total |
|---|---|---|---|---|---|---|
| 27B TP1, miss | 124.9s | 246.6s | 44.3s | 20.4s | 2.9s | 445.7s |
| 27B TP1, hit | 123.4s | - | - | 21.3s | 2.9s | 147.0s |
| 35B TP2+EP, miss | 126.5s | 271.0s | 39.7s | 25.1s | 1.6s | 474.2s |
| 35B TP2+EP, hit | 125.5s | - | - | 25.5s | 1.1s | 153.6s |

A warm start is ~3x faster end to end, and the restore replaces 247-271s of cold
init with 20-25s -- ~11x on the part this work controls. The cap is zone
creation, which is 123-127s in every run, 84% of a warm start, and entirely
outside semi-persistence.

Today the miss path is a dead end:

```314:318:arctic_inference/server/semip_engine.py
    meta_path = os.path.join(model_dir, "image", "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"semi_p: no pre-warmed image at {meta_path}; pre-warm this model "
            f"first (init / save / criu_dump offline).")
```

Images are made out of band, by `scripts/test_weights.py` and
`scripts/test_tp2.py`, and the gateway only ever restores. The goal is that
`run_local_sampling.sh` against a `semi_p_model_dir` with no image cold-starts,
dumps, restores, and then serves — so the second run of the same job is a fast
restore and the first run is self-sufficient.

The decision taken up front, because it shapes everything below: **on a miss,
dump and then immediately restore from that dump, and serve from the restore.**
`criu_dump()` is destructive, so the cold-started engine cannot be the one that
serves; and round-tripping the image inside the job that created it is what
makes "the dump worked" and "the image is restorable" the same observation
instead of two separate ones.

**Scope: TP1 and TP2 only**, against the models the image cache already holds —
`qwen_27b` (TP1), `qwen_35b` (TP1), `qwen_35b_tp2` (TP2+EP). TP4 and TP8 are
deliberately out, and Section 3.1 item 2 is the reason it is not a free
extension: the collective-path env flags start mattering above TP2, where a TP2
test cannot show it.

---

## 1. Shape of the change

One new function beside `restore_and_wrap`, and one branch in it:

```
restore_and_wrap(engine_kwargs, semi_p_model_dir)
  |
  +-- image/meta.json exists  -->  (today's path, unchanged)
  |                                read baked config, restore, wrap
  |
  +-- no image                -->  dump_and_wrap(engine_kwargs, model_dir)
                                     build vllm_config from engine_kwargs
                                     Instance(cfg, model_dir).init(gpus=...)
                                     ... dump sequence ...
                                     criu_dump(); teardown
                                     then fall through to the restore path
```

Nothing changes in `worker.py`, `sampling.py`, `models.py`, or the job JSONs.
The flag surface stays `semi_p` + `semi_p_model_dir`, and the miss is detected
with the same `os.path.isfile(<model_dir>/image/meta.json)` test the two scripts
use. That the whole feature fits behind one existing entry point is the main
argument for doing it here rather than as a separate prewarm service.

The cost is honest and should be stated in the log line: a missing image turns a
~20s `/initialize` into a cold start **plus** a dump **plus** a restore. Section
9 has the numbers we need to measure.

---

## 2. Model resolution: the `/mnt` problem

This is the one part of the design that is not about semi-persistence at all.

The two dump scripts hand `Instance` an **HF id**:

```18:19:semi_persistence/scripts/test_weights.py
config_qwen_27b = {"model": "Qwen/Qwen3.8-27B", "gpu_memory_utilization": 0.7, "max_num_seqs": 512}
config_qwen_35b = {"model": "Qwen/Qwen3.6-35B-A3B", "gpu_memory_utilization": 0.7}
```

so vLLM inside the `Instance` child resolves it through the HF hub and cache.
The gateway does not: `_prepare_sampling_engine_kwargs` ends with
`kwargs["model"] = resolve_model_path(cfg.model_name)`, which maps
`Qwen/Qwen3.6-35B-A3B` to `/mnt/neutrino/base-models/Qwen/Qwen3.6-35B-A3B`. A
restore never opens that path (Section 9 of `dss_integration.md`), which is why
this has never mattered before. A **dump** cold-starts vLLM, so it does.

On this pod that path cannot be made to exist:

- `/mnt` is root-owned `0755` and not writable by us; `/mnt/neutrino` does not
  exist, nor does `/data-fast/neutrino`.
- **`neutrino` has no sudo at all.** Not a password prompt — `sudo -n -v`
  answers `Sorry, user neutrino may not run sudo on mert-neutrino-prod-0`.
  `/etc/sudoers.d/` holds only the stock `README`, and our groups are
  `neutrino,users` — no `sudo`, no `wheel`.
- So `local_model_cache.sh` is unrunnable as written: every line that touches
  `/mnt` goes through `sudo`.

`dss/model_paths.py` has two env switches, and either avoids `/mnt` without
privilege. Both are picked up automatically by the zone head, worker, and the
coordinator actor that actually calls `resolve_model_path`, because `"DSS_"` is
already in `propagate_env_prefixes` in `dss/ray_dss/settings.py` — no
`DSS_PROPAGATE_ENV_EXTRA` edit needed.

| Switch | Effect on `resolve_model_path` | What the image bakes as `model` |
|---|---|---|
| `DSS_BAKED_MODEL_ROOT=/data-fast/neutrino/base-models` | maps to that root instead of `/mnt/...` | an absolute local path |
| `DSS_ALLOW_REMOTE_MODELS=1` | returns `model_name` unchanged; also makes `check_model_existence` a no-op | the HF id, exactly as the scripts do |

**Take `DSS_ALLOW_REMOTE_MODELS=1`.** It is the weaker-looking choice and the
better one, for a reason that has nothing to do with downloading:

- It removes the pre-download step entirely. vLLM pulls into the HF cache on the
  cold start, and every later run — dump or restore — is a cache hit. The bytes
  are the same either way; only the explicit step disappears.
- It makes the baked `vllm_config["model"]` an HF id, which is what the three
  existing images in `image-cache_neutrino` already record. Under
  `DSS_BAKED_MODEL_ROOT` the gateway would bake an absolute path instead, and
  `model` is the one key in the config that cannot be reconciled by filtering,
  because `Instance` needs it present. Keeping the two producers agreeing on it
  is worth more than the small honesty of a local path.
- The models are public and reachable from this pod: `Qwen/Qwen3.6-35B-A3B`
  (40 files) and `Qwen/Qwen3.8-27B` (32 files) both answer `200` with
  `gated=false private=false`, and `hf` is on `PATH`.

Set `HF_HOME=/data-fast/hf_home` alongside it. It is unset today, so the cache
would land in `~/.cache/huggingface`; that happens to be on the same 28T
filesystem, so this is about making the location explicit and shared rather than
about space. `HF_HOME` is listed by name in `propagate_env`, and `"HF_"` is a
propagated prefix, so it reaches the actor either way.

Both go in `run_dss-gateway.sh` next to the existing exports. Note this leaves
`local_model_cache.sh` unused and still broken; fixing it is out of scope, and
worth a comment there pointing here.

> **We should not silently rely on remote loading.** `DSS_ALLOW_REMOTE_MODELS=1`
> is exactly the switch production DSS does *not* set — the whole point of the
> baked cache is that a GPU pod never reaches the internet. This is a local
> testing decision. When this moves toward production the dump path needs the
> baked root, and with it the `model`-key divergence above; that is the thing to
> reopen, not the sequence.

---

## 3. The dump sequence

Nine calls, and — unlike the restore — identical at every TP degree. Canonical
form, from `test_weights.py` (`test_tp2.py`'s body is byte-for-byte the same):

```26:35:semi_persistence/scripts/test_weights.py
def init(inst: Instance, gpu=0):
    inst.init(gpu=gpu)
    inst.generate(conversation, sampling_params)
    inst.attach()
    inst.stage()
    inst.save_weights()
    inst.detach()
    inst.sleep()
    inst.cuda_checkpoint()
    inst.criu_dump()  # destroys the instance
```

| Step | What it does | Why it is where it is |
|---|---|---|
| `init(gpus=[...])` | Cold-starts the vLLM child on those physical GPUs; snapshots `total_gpu_bytes` from NVML; points the compile cache at `<model_dir>/compilation` | The list must have exactly `tensor_parallel_size` entries — `n_gpus` comes from the config, the argument is placement only |
| `generate(...)` | One short generation | Smoke test that the cold start actually works, before spending a dump on it |
| `attach()` | Allocates the per-rank CPU buffer sized to that rank's `named_parameters()` | Also what puts real `pinned_cpu_bytes` / `max_pinned_bytes_per_worker` on the handle |
| `stage()` | GPU params -> that buffer | |
| `save_weights()` | Buffer -> `<model_dir>/weights` (flat at TP1, `weights/rank<N>/` at TP>1) | After `stage`, before `detach`, or there is nothing to write |
| `detach()` | Frees the CPU buffer | Keeps the image small. Also zeroes both pinned counters (`_apply_result`), which is why every image records `pinned_cpu_bytes: 0` and why the restore must `attach().wait()` before planning |
| `sleep()` | `llm.sleep(level=2)` — frees weight and KV memory | |
| `cuda_checkpoint()` | `cuCheckpointProcess`; CUDA state to host | At TP>1 inserts `destroy_nccl()` itself |
| `criu_dump()` | Writes `image/` + `meta.json`; **kills the child** | Must follow `cuda_checkpoint` — GPU resources have to be released first |

### 3.1 TP>1: the sequence is the same, the surroundings are not

Checked against [`tp_DESIGN.md`](tp_DESIGN.md) rather than inferred. Its Section
2 lays the two chains side by side, and the save half differs in exactly one
place — inside `cuda_checkpoint`, not in the caller's code:

```71:74:semi_persistence/skills/tp_DESIGN.md
                 cuda_checkpoint  ─────────────────  cuda_checkpoint
                                                       ├─ destroy_nccl   auto-inserted when n_gpus>1
                                                       └─ cuda_checkpoint
                 criu_dump                           criu_dump
```

and Section 7 states the division of labour as an invariant: "`cuda_checkpoint`
inserts `destroy_nccl` itself, but `reinit_nccl` […] and `rebind_graphs` […]
are the caller's responsibility." `test_tp2.py` carries the same note in its
docstring.

**So `dump_and_wrap` needs no TP branch, and one code path serves both.** That
is the opposite of the restore side, where a TP1-only test hides two missing
calls. Write the dump with the list form everywhere — `init(gpus=[...])` accepts
a single-element list at TP1, so there is no reason to keep the scalar `gpu=`
path alive in new code.

Five TP>1 facts that do bear on the surrounding code, none of which is a step:

1. **`init` injects `worker_cls` itself, into a copy.** `instance.py:334-337`
   does `vllm_config = dict(self.vllm_config)` and only then
   `setdefault("worker_cls", "_semip_worker.SemipGPUWorker")`. So the child gets
   it and `self.vllm_config` does not — meaning it is **not** in `meta.json`, and
   the restore neither needs nor sees it. Section 4's projection must not add it.
2. **The TP>1 collective-path flags: covered at TP2, an open item above it.**
   `vllm_child.py:1611-1621` disables `fuse_allreduce_rms` and sets
   `NCCL_NVLS_ENABLE=0`, `VLLM_ALLREDUCE_USE_SYMM_MEM=0`,
   `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` when `tensor_parallel_size >= 2`, and
   in this library version it does so before its own `from vllm import LLM`
   (line 1667), so the ordering looks sufficient on paper.

   **It is not sufficient above TP2.** These flags are reported to matter at
   TP>2 — which is also where NVLS actually gets selected, so a TP2 test cannot
   show it — and that is why `example_full.py` exports them at module level,
   ahead of everything, rather than relying on the child. Treat the child's
   in-handler set as covering our TP1/TP2 scope only, and do **not** record it as
   making `example_full`'s exports redundant.

   When TP4/TP8 images are wanted, the right lever is not the gateway's
   environment but `vllm_config["_env"]` — a reserved mapping the child applies
   to `os.environ` *before* it imports vLLM (lines 1623-1653). All three flags
   are absent from its `_RESERVED_ENV` denylist, so passing them there is legal;
   and because `_env` rides inside `vllm_config`, it is baked into `meta.json`
   and travels with the image instead of depending on how the gateway happened to
   be launched. That is strictly better than an export in
   `run_dss-gateway.sh`, which is why the export is worth *not* adding now.
3. **`len(gpus)` must equal `tensor_parallel_size`, and Ray decides `len(gpus)`.**
   The job's `n_gpus` and its `vllm_config.tensor_parallel_size` are separate
   fields in the JSON that must agree; disagree and `init` raises after the zone
   is already up. Preflight it in `dump_and_wrap` and say which two fields
   conflict.
4. **Per-rank weights are free.** `save_weights()` fans out to
   `weights/rank<N>/` at TP>1 on its own. Only tooling that reads
   `weights/weights_meta.json` directly is TP1-only.
5. **`max_pinned_bytes_per_worker` is the number the restore budget uses**, and
   `detach()` zeroes it, so it is `0` in `meta.json` exactly as at TP1. Nothing
   to do on the dump side; it is the restore's `attach().wait()` that matters.

`save_weights()` is optional in the library — omit it and the staged weights
stay inside the CRIU image, which is what `Orchestrator` does. **We keep it**,
because the restore path in `restore_and_wrap` already calls `load_weights()`
and the baked cache is built that way; changing it would fork the two.

`criu_dump` records `vllm_config`, `model_dir`, `total_gpu_bytes`,
`pinned_cpu_bytes`, `n_gpus`, and `max_pinned_bytes_per_worker` via
`meta_extra`; `_worker_criu_save` adds `rank`, `gpus`, `gpu_uuids`, `uid`, and
`gid`. Those last three are what the uid check and the cross-node device map
read later.

### Two orderings exist in the tree

[`reproduce/example_full.py`](../reproduce/example_full.py) — which
`dss_integration.md` Section 5 treats as the authority for the *restore* — does
the dump differently:

```60:67:semi_persistence/reproduce/example_full.py
    print("== up ==", flush=True)
    inst.init(gpus=gpus).attach().repin().stage()
    inst.generate([PROMPT], SAMPLING).wait()
    print(f"  answer after cold start: {str(inst.last_generate_result)[:80]!r}",
          flush=True)

    print("\n== saved -> up ==", flush=True)
    inst.save_weights().detach().wait()
```

It stages *before* generating, and it calls `repin()` — `cudaHostRegister` on
the staging buffer — which the two scripts omit. The restore always calls
`repin()` itself, so the dump-side one should only change staging DMA
throughput, not correctness, and the three images in `image-cache_neutrino` were
made without it and restore fine.

**Follow the scripts, not `example_full`.** Not because the ordering is better,
but because those are the images we have restored, repeatedly, through this
adapter. `repin()` on the dump side is the one thing worth measuring later as a
possible `save_weights` speedup; it should not ride in on the first change.

---

## 4. Building `vllm_config` from `engine_kwargs`

This is the part most likely to be got wrong, and the one place where a mistake
is silent rather than loud.

`Instance.init` hands `vllm_config` more or less straight to the child, which
constructs the vLLM engine from it. `engine_kwargs` as it arrives from
`sampling.py` is **not** that shape: it carries `extra_engine_kwargs` as a
nested dict (`sampling.py:424-425`), plus whatever else the sampling path has
accumulated. Passing it through unfiltered means the child constructs vLLM with
an unknown `extra_engine_kwargs=` kwarg.

So a projection is needed. The important and slightly surprising property is
that **the projection does not have to be reproducible**:

```485:489:semi_persistence/instance.py
            saved_config = meta.get("vllm_config")
            if saved_config is not None and saved_config != self.vllm_config:
                raise RuntimeError(
                    f"vllm_config mismatch: instance has {self.vllm_config} "
                    f"but image at {filename} was saved with {saved_config}")
```

`criu_restore` compares by exact dict equality — but `restore_and_wrap` already
passes `baked`, read straight out of `meta.json`, so whatever we dump is what we
later restore against, by construction. The projection therefore has exactly one
hard requirement: **the config it produces must be one the child can build an
engine from.** It does not have to match the job JSON, the scripts, or a future
version of itself.

Proposed rule, to live in one function with the reasoning attached:

- Start from `engine_kwargs`.
- Drop `extra_engine_kwargs` and re-merge its contents flat, or drop it
  outright — to be settled against what `vllm_child` actually accepts (Section
  10).
- Drop anything dss-level rather than vLLM-level.
- Keep `model` as resolved (an HF id, per Section 2).
- Assert `tensor_parallel_size == len(gpus)` before calling `init`, so the
  mismatch is named here rather than inside `init`'s own check.

> **Section 11 settles the third bullet, and not as a denylist.** "Drop anything
> dss-level" assumed the set was known in advance; running it showed
> `engine_kwargs` is a serialized `ModelConfig` whose non-vLLM fields
> (`fp32_lm_head`, `forest_cascade_attn_configs`, ...) arrive with defaults no
> job JSON sets. `_strip_kwargs_vllm_rejects` therefore asks vLLM which keys it
> rejects and decides by *value*: a default is dropped, a real request raises.
> Implement from Section 11, not from this list.

One consequence to write down where someone will find it: a gateway-dumped
image will bake a *richer* config than the hand-written ones — the tp1 JSON asks
for `max_model_len`, `dtype`, and `trust_remote_code`, none of which appear in
`qwen_35b`'s baked `{'model': ..., 'gpu_memory_utilization': 0.7}`. That is an
improvement (the job's requested config finally takes effect, and
`_log_config_divergence` stops warning), but it also means script-dumped and
gateway-dumped images are not interchangeable *as configs*. Both remain
restorable through `restore_and_wrap`, which reads `baked`; only the scripts,
which hand-write their config, are picky.

---

## 5. Dump, then restore, in one process

`criu_dump()` destroys the instance, so the restore needs a **fresh `Instance`**
on the same `model_dir`. `example_full` does exactly this — tears the first one
down, waits, and constructs a new one — and the two scripts sidestep it by being
run twice as separate processes.

```python
# sketch, not final
_dump(engine_kwargs, model_dir)          # ends: criu_dump().wait(); teardown()
return _restore(engine_kwargs, model_dir)  # today's restore_and_wrap body
```

Four things about the in-process second half, two reassuring and two not:

**The `/tmp/inst<N>.log` hazard is at its mildest here.** (Retired 2026-10-03:
fd 1/2 are now the pod-log pipe, so this paragraph describes images from
earlier builds only; see CRIU_PLUMBING Complication 4.) Instance ids are a
per-process global from 0 (`_alloc_instance_id`), so the dumping `Instance` is
id 0 and the restoring one is id 1. The image therefore records `/tmp/inst0.log`
as the child's fd 1 — and that file already exists, at the right mode and the
right uid, because our own id-0 `Instance` created it minutes earlier in this
same process. `_precreate_dumped_log_paths` will find nothing to do. Compare the
cross-node case in `dss_integration.md` Section 4, where this is the failure
that blocks everything.

**Same-node, same-index restore is the case that works without a device map.**
`_worker_restore` builds the `oldUuid=newUuid` map only when the placement
changed, and the usual warning is that restoring a *foreign* image onto its
dumped indices skips the map and dies in `cuCheckpointProcessRestore`. Here the
baked `gpu_uuids` are this node's, because we just dumped them, so no map is
needed and none is built. `_check_gpu_placement` will not fire either: it
returns early as soon as the local UUIDs overlap the baked ones. Nothing to do,
but worth knowing so the absence of a map is not read as a bug.

> **Section 12 corrects this item with measurements.** The mechanism below is
> right; the exposure is not. Across four runs the retry never fired, and the
> `DESTROY` → re-run cycle cannot reach the window while zone creation takes
> ~125s. Read Section 12 before acting on any of it -- including the claim that
> TP2 is better protected than TP1, which the images disprove.

**A restore inside 60s of its own dump can fail on port rebind, and our flow is
exactly that.** This is the one finding that argues against the Section 1
decision, so it is worth stating in full. CRIU records every inet socket's local
port and *rebinds it* at restore; `socket:` is on the child's dump keep-list
(`vllm_child.py:859` and `:2033`), so those sockets go into the image. The
destructive dump closes them with a FIN, leaving tuples in `TIME_WAIT` for the
kernel's hard-coded 60s `TCP_TIMEWAIT_LEN`, and the restore collides:

```
Error (criu/sk-inet.c): Can't bind inet socket (id 0x…): Address already in use
```

The library's mitigation is `_mark_inet_sockets_rst`, which sets
`SO_LINGER(1,0)` so the eventual close sends an RST and the tuple never enters
`TIME_WAIT`. It has two coverage gaps, both recorded as "known, benign today" in
Complication 12 of [`CRIU_PLUMBING.md`](CRIU_PLUMBING.md) — benign precisely
because nothing has restored within 60s of its own dump before now:

| Sockets | Marked? |
|---|---|
| TP worker ranks (TCPStore rendezvous) | **Only at TP>1.** The call sits *after* the `tp_size <= 1` early return in `_destroy_nccl` (`vllm_child.py:647`, marking at `:682`) |
| The `vllm_child` process's own | **Never.** `_destroy_nccl` is a `collective_rpc` target, so it walks workers only, and the child's own dump-prep block never calls it |

This inverts the usual intuition. For this flow **TP2 is better protected than
TP1**, because at TP2 the auto-inserted `destroy_nccl` marks each rank's
rendezvous socket — and TP1, the case Section 9 brings up first, has nothing
marked at all. Complication 12 puts it plainly: a TP1 image "still contains such
sockets […] it just usually wins the race."

Worse, Section 2's `DSS_ALLOW_REMOTE_MODELS=1` decision actively creates the
uncovered class. Complication 12 names "leftover HTTPS connections to the model
hub, bound to the routable IP rather than loopback" as exactly the child sockets
that still leave `TIME_WAIT` behind — and a cold start that pulls weights from
the hub is guaranteed to have had them. The two decisions interact, and the
first dump of a freshly downloaded model is the worst case for both.

Proposed handling, in order:

1. **Mark the child's own sockets.** Call `_mark_inet_sockets_rst` from the
   child's dump-prep block, ahead of its fd loop. One call, in the process that
   owns the unmarked sockets, at every TP degree — it closes both gaps at once
   and needs no new primitive or `Instance` surface. This is a change to
   `semi_persistence` rather than to the adapter, so it wants to be a separate,
   deliberately reviewed commit, not a line smuggled in with the dump path.
2. **Retry on `EADDRINUSE` regardless.** Wrap the dump path's `criu_restore` in a
   backoff out past the 60s window. It costs nothing when the marking works, and
   it is the only thing that helps for an image *already* dumped without it.
   Prefer it to a flat `sleep(60)`, which would tax every miss for a collision
   that may not happen.

The redeeming detail: this is **time-bound, not image-bound.** Complication 12 is
explicit that images dumped before the marking existed restore fine once the
window drains, so a first attempt that fails this way does not mean re-dumping —
which is also why the symptom looks like it merely needs a cooldown.

**Task-id collisions are the other residual risk.** The low-cap restore has no private
PID namespace, so every task id in the image must be free on the host — ~900 for
a TP2 image. The ids were vacated seconds earlier by the dump, and Linux
allocates PIDs ascending to `pid_max` before wrapping, so they should still be
free; but this is the one item here that is an argument rather than a
measurement. `scripts/pidcheck.py <image_dir>` answers it directly and should be
what we reach for if the restore half fails with an occupied-id error.

---

## 6. Concurrency, dirty directories, and locking

The scripts get to assume they are the only writer. A gateway does not.

**Two jobs on one `semi_p_model_dir`.** Both miss, both cold-start, both dump
into the same `image/` and `weights/`. The second one's `criu_dump` interleaves
with the first's and the result is a corrupt image that will fail a restore in
an unrecognisable way. Needed: an exclusive lock in `model_dir` (an `O_CREAT |
O_EXCL` lockfile is enough, given both writers are uid 1000 on one node), held
across the whole dump. A job that cannot take the lock should wait for the
holder and then re-check for the image, rather than dump in parallel.

**A failed dump leaves a directory that reads as a miss.** `meta.json` is
written at the end, so an interrupted dump leaves `image/` populated and
`meta.json` absent — which is precisely our miss test. The next job then dumps
*into* that debris. `example_full` handles this by `shutil.rmtree(model_dir)`
before every dump. We should do the same, but only under the lock, and only for
`image/` and `weights/` — blowing away `compilation/` costs a recompile for no
reason. Whether a stale `compilation/` from a *different* config is a hazard is
an open question (Section 10).

**One live restore per node.** Independent of this change, but it becomes
reachable: the low-cap path caps the node at one restore at a time, and the
restore half of a miss now competes with ordinary restores of other jobs.

---

## 7. GPU placement

`_resolve_physical_gpus()` already does the right thing and is reusable
unchanged: it parses every id out of `CUDA_VISIBLE_DEVICES`, clears `CVD` so the
`Instance` child can address physical ids, and returns a list. `init(gpus=...)`
takes the same list shape as `cuda_restore(gpus=...)` and validates the same
count against `tensor_parallel_size`.

So the dump runs on whatever Ray assigned, the image records those indices, and
the immediate restore uses the same ones. The consequence to be explicit about:
**a gateway-dumped image records this node's indices and UUIDs**, so it is a
same-node image. Restoring it later on this node is unconstrained; carrying it
to another node puts it under the whole cross-node story in
[`CROSS_NODE_RESTORE.md`](CROSS_NODE_RESTORE.md), including the rule that it
must not be restored onto the indices it was dumped on.

---

## 8. What we are not doing

- **No new job-config surface.** No `semi_p_dump`, no `semi_p_prewarm`. Dumping
  is what a miss means. If we later want a job that refuses to dump, that is a
  flag then, with a default that keeps this behaviour.
- **No dump on `/destroy`.** Deferring the dump to teardown would keep
  `/initialize` fast, but it dumps an engine that has served real traffic and it
  puts the expensive, failure-prone step where nothing is waiting to observe it.
- **No separate prewarm service.** The one-entry-point property in Section 1 is
  the whole appeal.
- **No re-dump on a config change.** An image whose baked config no longer
  matches the job is today's silent-override behaviour (`dss_integration.md`
  Section 4), and it stays that way. Invalidating a cache on config drift needs
  the compatibility model that document already defers.
- **Nothing about weight sync, LoRA, or `reset_prefix_cache`.** Unchanged, still
  out.

---

## 9. How we will know it worked

Run order matters — the cheap cases fail faster and on different things.

1. **`sampling-tp1-semip-27b.json` into an empty `model_dir`.** Smallest weight
   set (51 GiB). Proves the projection, the sequence, and the in-process
   round-trip.
2. **Same JSON again, image now present.** Must take the existing restore path
   and reach `RUNNING` in ~20s. This is the regression test for the branch: if
   the miss path has quietly changed the config we bake, the hit path is where
   it shows.
3. **`sampling-tp1-semip.json`** (35B, 65 GiB) — the staging budget is real
   here; a bad plan may survive 27B and OOM at 35B.
4. **`sampling-tp2-semip.json`** — TP2+EP: the GPU list, the `destroy_nccl`
   auto-insertion, the per-rank `weights/rank<N>/` layout.

Point each at a **fresh** `model_dir`, not at `image-cache_neutrino/qwen_*` — we
want the existing images intact as a known-good control, and Section 4 means a
gateway dump would bake a different config into them anyway. Something like
`/data-fast/image-cache_gw/<name>`.

Numbers to record, because the whole feature is a latency trade: cold start,
dump, restore, and total `/initialize` wall time, per model. Compare the restore
half against the 21–36s the existing images take.

Two independent confirmations that a dump really happened, rather than an image
appearing by some other route: `meta.json` should record **our** `uid` (1000),
**this** node's `gpu_uuids` (`GPU-89bd898b-…` and friends, not the
`GPU-bc969410-…` set every existing image carries), and the Ray-assigned
`gpus`. And `crit decode` on the core image must show `cap_eff=[0, 0]`, per the
verification snippet in `dss_integration.md` Section 9 — without it the cap set
was not dropped and the image is unrestorable on a cap-prod node, fixable only
by re-dumping.

---

## 10. Open questions

These are the things this design does not settle and should not pretend to.

1. **What exactly does `vllm_child` accept?** Section 4's projection is a rule
   with a hole in it. Needs reading `vllm_child.py`'s engine construction and
   comparing against `worker.py`'s `AsyncEngineArgs` path, then writing the
   whitelist against that rather than against a guess.
2. **Is the `generate()` warm-up load-bearing?** vLLM captures CUDA graphs
   during engine init, so the generation is probably a smoke test rather than a
   capture trigger — but `rebind_graphs` on the restore side rebinds
   *preserved* graphs, and it is worth confirming what state the dump has to
   leave them in before dropping the call. (Since 2026-09-24 the dump's warm
   pass answers most of this: it drives every captured shape deliberately.)
3. **Does anything gateway-side time out during a multi-minute
   `/initialize`?** `run_local_sampling.sh` posts `/initialize` and then polls
   `/job/<id>` for `RUNNING`, which suggests init proceeds in the background and
   only the poll deadline (`INIT_TIMEOUT_S`, 1800s) applies. Needs checking for
   an actor-level or zone-level deadline that a cold start plus dump plus
   restore would now exceed.
4. **Is a stale `compilation/` from a different config a hazard?** Section 6
   proposes keeping it across a re-dump to save a recompile. Cheap to keep,
   possibly wrong.
5. **Where should the lockfile live**, and what should a waiter do if the holder
   dies mid-dump? A stale lock that blocks every future job is worse than the
   race it prevents.
6. **`repin()` on the dump side** — measure whether it speeds up
   `save_weights()` enough to adopt `example_full`'s ordering.
7. **Do we take the `_mark_inet_sockets_rst` change now or ship the retry
   alone?** Section 5 argues for both, but the library change touches
   `vllm_child.py` on the dump path for every caller, including the
   orchestrator. Deciding this needs someone who owns that file, and the retry
   is enough to get the first TP1 bring-up moving without it.
8. **What exactly makes the collective-path env flags matter above TP2?** The
   child sets them before its own vLLM import, so the ordering argument does not
   obviously explain it. Worth pinning down before anyone tries TP4, because the
   answer decides whether `_env` is genuinely sufficient there or whether the
   flags have to be in the environment ahead of the child.

---

## 11. As implemented

Status: **run, at both TP degrees.** Section 9's matrix is done for `qwen_27b`
(TP1) and `qwen_35b_tp2` (TP2+EP): each dumped on a miss and restored on the
following hit, with coherent output and clean teardown every time. `qwen_35b`
(TP1, 35B) is the one row not exercised.

Three things only running it could have found, all fixed here:

1. **`fp32_lm_head` aborted the first attempt.** It is a `ModelConfig` field
   (`server/config.py:50`, default `False`) that dss serializes into
   `engine_kwargs`, not something any job JSON sets. The cold path absorbs it by
   routing to `Fp32LmHeadAsyncEngineArgs`, and pops `forest_cascade_attn_configs`
   separately -- both *after* the semi_p early return, so semi-p had never met
   either. The preflight was right to reject it (the child's plain
   `LLM(**vllm_config)` would have died on it a minute later) and wrong to treat
   a default as fatal. Hence the drop-if-unasked-for rule in
   `_strip_kwargs_vllm_rejects`, made generic rather than a denylist because
   `forest_cascade_attn_configs="{}"` was the very next failure.
2. **`/destroy` leaked the whole GPU.** `_SemiPEngine.close()` sent `teardown()`
   without waiting, and `teardown()` is a non-blocking `_send`. The actor exited
   first, the Instance worker died before draining the command, and the restored
   child survived -- reparented to init, holding 99.5 GiB, and invisible to
   `ps | grep vllm` because a CRIU-restored child's `comm` is plain `python`.
   Restored trees are hit hardest: `criu_restore` sets `child_proc=None`, so
   `worker_loop`'s force-kill fallback is skipped and that handler is the only
   thing that reaps them. `close()` now waits; verified by `<<< teardown OK`
   appearing in the log and the GPUs returning to 0 MiB.
3. **The TP2 path needed no code at all.** `cleargraph OK (0.001s)` and
   `destroy_nccl OK (0.732s)` appear in the TP2 dump log, auto-inserted inside
   `cuda_checkpoint` exactly as `tp_DESIGN.md` Section 2 promises. (That
   `0.001s` is the tell that `cleargraph` never did anything under reuse; the
   primitive was retired 2026-09-24 and only `destroy_nccl` is inserted now.) Per-rank
   `weights/rank0/` and `rank1/` were written and read back without special
   handling. `stage` was *faster* at TP2 (20.7s vs 28.8s) because the two ranks
   copy in parallel.

All of it lives in `arctic_inference/server/semip_engine.py`, plus two shell
scripts. `vllm_child.py`, `worker.py`, `instance.py`, `sampling.py`, `models.py`
and the job JSONs are untouched, so nothing here can reach the orchestrator.

`restore_and_wrap` keeps its signature and becomes a branch over six new
helpers:

| Function | Role |
|---|---|
| `_vllm_config_from_engine_kwargs(engine_kwargs, gpus)` | The projection, plus the TP-vs-placement check |
| `_strip_kwargs_vllm_rejects(cfg)` | `AsyncEngineArgs(**cfg)` as a probe; drops a rejected key left at its default, raises on one holding a real value |
| `_dump_lock(model_dir)` / `_dump_lock_is_stale(path)` | Exclusive `<model_dir>/.dump.lock`, pid-liveness staleness |
| `_dump(cfg, model_dir, gpus)` | The nine-call sequence; destructive |
| `_restore(model_dir, engine_kwargs, gpus)` | The old `restore_and_wrap` body, plus a teardown on failure |
| `_restore_with_port_retry(...)` / `_is_address_in_use(exc)` | The `TIME_WAIT` backoff, wrapping **both** paths |

Concrete values, so they can be argued with rather than rediscovered:

- **Projection.** Drops `extra_engine_kwargs` and re-merges it flat; drops
  `_NON_VLLM_ENGINE_KEYS = {ray_num_gpus, router_replay_max_cache_bytes}`;
  requires `model`; adds no `worker_cls`.
- **Rejected keys.** `_strip_kwargs_vllm_rejects` loops on
  `AsyncEngineArgs(**probe)`, pulling the key named by each
  `unexpected keyword argument` and deciding by value: one still holding a
  `ModelConfig` default -- `_UNSET_LIKE = (None, False, 0, "", "{}")` -- is
  dropped and logged, anything else raises. It is generic rather than a denylist
  because `fp32_lm_head=False` and `forest_cascade_attn_configs="{}"` were the
  first two, and there was no reason to think they were the last. The loop is
  bounded by `len(probe) + 1` since every pass removes exactly one key. An import
  failure, or a constructor complaint that is not a bad kwarg name, is logged and
  left to the child; only a real value the child cannot honour fails the dump.
- **Lock.** `_DUMP_LOCK_WAIT_S = 3600` (a holder may be paying a first-time hub
  download), `_DUMP_LOCK_POLL_S = 5`, `_DUMP_LOCK_GRACE_S = 30` for a lockfile
  that is still empty. Timeout raises and names the file to remove.
- **Retry.** `_TIME_WAIT_S = 60` plus 15s headroom, `_RESTORE_RETRY_SLEEP_S = 10`.
  `_is_address_in_use` tries `worker.py`'s `_is_address_in_use_error` first and
  falls back to matching `"address already in use"` in `str(exc)`, because the
  `Instance` surfaces a worker-side failure as `RuntimeError(...)` carrying
  CRIU's stderr as text with no `__cause__` to walk. Which of the two fires is
  worth checking on the first real collision.
- **Dump smoke generation.** `_DUMP_PROMPT` with `max_tokens=32,
  temperature=0.0, ignore_eos=True`.

Three things the design missed, found while writing it:

1. **`_resolve_physical_gpus()` may be called only once.** It clears
   `CUDA_VISIBLE_DEVICES` as a deliberate side effect, so a second call sees an
   empty CVD and falls back to `[0]`. `restore_and_wrap` now resolves once and
   passes the list to both `_dump` and `_restore`. Calling it inside each would
   have quietly dumped and restored TP1 jobs on GPU 0 regardless of Ray's
   assignment, and broken every TP2 job on the count check.
2. **A failed `_restore` must tear its `Instance` down.** It now does, in an
   `except BaseException` that re-raises. Previously a failed restore leaked the
   worker process and the exception ended the job, so it went unnoticed; with a
   retry loop above it, the leftover worker can hold the very ports and task ids
   the next attempt needs, converting a recoverable collision into a permanent
   one.
3. **Section 6's "add no rmtree" is now verified, not just argued.**
   `_worker_criu_save` rmtree's `image_dir` at `worker.py:351-359` and
   `_semip_save_weights` does the same for its rank dir at
   `vllm_child.py:337-345`, both with a targeted uid-collision message. The
   implementation adds none, and says so in a comment where someone would
   otherwise add one.

Against Section 10's open questions: **3 is closed** -- nothing gateway-side
times out during a long `/initialize`; the 474s TP2 miss completed normally, so
only `run_local_sampling.sh`'s own 1800s poll deadline applies. **5 is closed**
by the implementation (lockfile in `model_dir`, dot-prefixed, pid-liveness with
a grace window for an empty file). **7 is informed by Section 12** rather than
answered: the retry alone has been enough, because the library gap it works
around turns out to be unreachable here. 1, 2, 4, 6 and 8 are untouched --
notably 2, since every run so far has included the `generate()` warm-up and
none has tested dropping it.

`dss_integration.md` Section 8's "tokenizer for a model that is only in the
image" limitation is incidentally resolved by Section 2's
`DSS_ALLOW_REMOTE_MODELS=1`, since `model` stays an HF id that
`AutoTokenizer.from_pretrained` can resolve through `HF_HOME`.

One loose end outside the adapter: `dss-client/examples/local-gateway/vllm_config_trace.py`
(untracked, a local diagnostic) described the old asymmetric divergence check in
a comment and printed a "keys dropped with NO warning" section. Both were
updated to match, since that script exists specifically to explain this
behaviour and would otherwise now mislead.

---

## 12. `TIME_WAIT`: measured, and why it did not fire

Section 5 argued the port-rebind hazard would bite this flow. Four runs later --
27B TP1 and 35B TP2+EP, each a miss then a hit, so four restores -- **the retry
has never fired once.** This section records what was actually measured, because
the reasoning in Section 5 was right about the mechanism and wrong about the
exposure.

### What the images actually contain

`crit decode -i <model_dir>/image/files.img`, counting `INETSK` entries:

| Image | Sockets recorded | `reuseaddr=False` (the ones that fail to rebind) | Any `so_linger`? |
|---|---|---|---|
| `qwen_27b` (TP1) | 11 | 3, incl. two to `dst_port=443` | none |
| `qwen_35b_tp2` (TP2+EP) | 15 | 9 | none |

So the sockets are genuinely in the image, and the risky subset is real and
non-empty. Complication 12's description is accurate.

### Correction: the marked sockets are not the recorded sockets

Section 5 predicted TP2 would be *better* protected than TP1, because
`cuda_checkpoint` auto-inserts `destroy_nccl`, which marks `SO_LINGER(1,0)`.
The TP2 image shows `so_linger` on **nothing** -- and the dump log confirms
`destroy_nccl` did run (`<<< destroy_nccl OK (0.732s)`).

The resolution is that `destroy_nccl` marks those sockets *and then closes
them*. Marking is what makes that close an RST instead of a FIN. By the time
`criu_dump` runs they are gone, so they never enter the image at all. What
remains recorded is a different population entirely: hub HTTPS connections, the
loopback pair on port 8000, and listeners.

That makes the "TP1 is never marked" gap largely beside the point. Neither TP
degree has marked sockets *in the image*, because marked sockets by construction
do not survive to be dumped.

### The ports are not in `TIME_WAIT` anyway

Measured immediately after the TP2 restored tree was torn down, while the kernel
table held **411** `TIME_WAIT` entries: the intersection between those and the
ports the image needs to rebind was **empty**.

The likely reason is that `TIME_WAIT` only burdens the side that closes
*first*. When the tree is killed, most of these connections have already been
closed by the peer, or close with an RST because unread data is sitting in the
receive buffer. Neither path enters `TIME_WAIT`. This is inference from the
measurement rather than something proven, and it is the one part of this section
that could still be wrong.

### The gateway path cannot reach the window

The remaining exposure is the post-dump restore, which follows its own dump by
**5.3s** (27B) and **5.7s** (TP2) -- comfortably inside 60s. It succeeded both
times.

The `DESTROY=1`-then-re-run cycle, which Section 5 called out as the second
exposure, turns out to be unreachable:

```
run 1 teardown   02:48:29.6
run 2 restore    02:51:39.9     gap 190s, against a 60s window
```

Zone creation is why. Measured on one basis (first line of the run's zone log to
the first `Instance` command) it is **123-127s across all four runs**, regardless
of model, TP degree, or hit/miss. A second job simply cannot start restoring
inside a minute of the first one's teardown.

> **Do not delete the retry on the strength of this.** Zone creation is not a
> fixed property of the system -- it is 84% of a warm start (125s of 154s at
> TP2), it is entirely outside semi-persistence, and it is the most obvious
> remaining optimisation target. The moment it drops under ~60s, the
> `DESTROY` → re-run cycle lands *inside* the window and this retry becomes
> load-bearing rather than decorative. The same is true if anything ever shortens
> the gap between dump and restore, or if a future dump leaves a socket open
> that the local side closes first.

### Standing conclusion

Treat the hazard as **documented and plausible, with preconditions present, but
not observed here**. The retry costs nothing when there is no collision, it is
the only thing that helps for an image already dumped without the marking, and
the condition it guards is time-bound rather than baked into the image -- so the
worst case it prevents was always a slow retry rather than a dead image.

Forcing it would take bypassing the gateway: drive `restore_and_wrap` twice in
one process from a `restore_smoke.py`-style harness, where nothing imposes the
125s zone-creation delay. That is the honest test, and it has not been run.
