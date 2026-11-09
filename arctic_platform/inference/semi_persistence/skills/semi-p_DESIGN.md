# Model directory and image semantics

How a saved model is laid out on disk, what each part means, and the
rules an image binds itself to.  Read this before wiring semi-persistence
into a serving stack, or when debugging a restore that fails on paths,
config, or stale code.

For the primitive-by-primitive API see [reference.md](reference.md) and
[instance_DESIGN.md](instance_DESIGN.md); for CRIU internals see
[CRIU_PLUMBING.md](CRIU_PLUMBING.md); for tensor parallelism see
[tp_DESIGN.md](tp_DESIGN.md).

---

## 1. The `model_dir` contract

An `Instance` may be constructed with a per-model directory:

```python
inst = Instance(vllm_config, "/data-fast/image-cache/qwen_35b")
```

It holds everything a saved model needs, so `criu_dump()` and
`criu_restore()` take no path:

```
<model_dir>/
  compilation/   # per-model JIT/compile cache: Triton, torch inductor,
                 # vLLM torch.compile, FlashInfer
  image/         # CRIU image + meta.json
  weights/       # detachable weight shards: shard_NNNN.bin + weights_meta.json
                 # (per-rank subdirs weights/rank{R}/ at TP>1)
```

`model_dir` is optional.  Without it the primitives take explicit paths
(`criu_dump("/path/to/image")`), the weights directory defaults to a
`weights` sibling of the image path, and the compile caches keep their
node-local defaults.  That mode is correct for same-node restore and is
what the orchestrator uses.

`model_dir` threads `init(gpu)` -> `worker_loop` -> `vllm_child_loop`,
where the child points its compile-cache environment variables at
`<model_dir>/compilation` *before* importing vLLM.

**Suggested convention** when you do use it: derive the directory from
the model path, e.g. `f"{resolve_model_path(model_name)}_image"`.  It is
deterministic, colocated with the weights, human-readable, needs no
hashing, and gives CRIU a stable absolute path for the recorded JIT
mappings.

The serving adapter takes a stricter line, and for a reason worth knowing
before you copy the convention above: a path that is merely *stable* still
lets two different configurations share a directory.  `restore_and_wrap`
therefore derives `model_dir` from a hash of the config and of the
environment (`$SEMIP_IMAGE_CACHE/<cfg>_<env>`), which makes the directory
name the cache key rather than a label attached to one.  See
[`IMAGE_CACHE.md`](IMAGE_CACHE.md) Section 6.

**One image and one weights set per `model_dir`.** There are no
per-config subdirectories and no path encoding of TP size, dtype, or
`max_model_len`.  Binding an image to a specific `vllm_config` in the
path is not attempted; the config check below does that job instead.

---

## 2. What a re-dump does to each directory

Verified behavior, and the three parts differ:

| Directory | On re-dump |
|---|---|
| `image/` | `rm -rf`'d and recreated. CRIU will not reliably overwrite a differing file set, so a stale image could otherwise corrupt the new one. |
| `weights/` | Fully overwritten (cleared and rewritten). Shard counts vary with model size, so a partial overwrite would leave a mix the manifest cannot detect. |
| `compilation/` | **Reused, never cleared.** It is a content-keyed cache, so re-dumping into the same directory skips recompilation. |

The `rm -rf` paths raise a named `RuntimeError` if they hit a
`PermissionError`, which means the directory holds files from a run under a
different uid.  Dump and restore must share one uid (enforced by
`Instance.criu_restore`), so such leftovers are cleared by hand as their owner
rather than by elevating.

---

## 3. What binds an image

An image is not portable across arbitrary configurations.  `criu_restore`
checks two things before spawning anything, and both raise immediately
rather than failing later inside CRIU.

**`vllm_config` must match exactly.**  The comparison is a full dict
`!=`, so the config passed to `Instance` must be exactly the baked dict,
with no extra keys.  Because `tensor_parallel_size` lives in the user's
`vllm_config`, it participates in this check — a TP=2 image cannot be
mistaken for a TP=1 one.

**`model_dir` must match.**  CRIU records the compile-cache `.so` and
cubin mappings by absolute path, so an image is bound to the directory it
was dumped under.  `criu_dump` records `model_dir` in `meta.json` and a
mismatch is rejected by name.

`meta.json` also carries the CRIU plumbing (`child_pid`, `pipe_fd`,
`pipe_resource`, `nvidia_fds`), the placement and hardware identity
(`rank`, `gpus`, `gpu_uuids`), and the budget inputs (`total_gpu_bytes`,
`pinned_cpu_bytes`, `n_gpus`, `max_pinned_bytes_per_worker`).

### An image also has a warmth, and nothing records it

Two images can match on every field above and still behave differently,
because an image captures a process *mid-life*: whatever that process had
lazily built by the moment of the checkpoint is in the image, and whatever
it had not, is not. Nothing in `meta.json` describes this.

It is not a curiosity. At TP>1 it was the cause of the 302 s post-restore
hang: `keep_graph=True` stops torch instantiating CUDA graphs at capture,
so an unwarmed image carries execs only for the batch shapes its dump-time
job happened to run, and the restore builds the rest — on every rank, in a
freshly restored context, alongside the JIT compiles for kernels no shape
had yet dispatched to. Warming the dump is unconditional since 2026-09-24,
so an unwarmed image should no longer be producible; `tp_DESIGN.md` §5 has
the mechanism and the fix.

Two things follow for anything built on this stack.

**Prefer paying at dump time.** The dump runs once, in one healthy
process. The restore runs on every wake, on all ranks, in a context that
was just reconstructed. Work moved from the second to the first is paid
once and amortised over every restore — measured here as ~11 s added to a
353 s dump to remove 4.35 s from each restore, which breaks even after
three wakes. Latency on the restore path is the product's whole value
proposition; latency on the dump path is nearly free.

**Assert warmth rather than assuming it.** Laziness is invisible by
construction: a cold image restores, serves, and looks correct right up
until it doesn't. The counterpart is a cheap dump-side census with a
hard gate, so a key is never published from an image that failed to warm.
Whenever you add a lazily-initialised resource to the restore path, ask
what forces it to exist before the checkpoint, and what would tell you if
that ever stopped working.

---

## 4. Constraints worth knowing before you debug

**Cross-node restore needs the same absolute path.** Copy the whole
`model_dir` to the target node at the identical path.  The compile cache
is the reason: its `dlopen`'d artifacts are recorded by absolute path and
must exist at restore.  Two of the other cross-node requirements are
handled automatically — `meta.json` carries the capture node's GPU UUIDs
so the CUDA restore device map can pair them against the local node's
(zero overlap between the two UUID sets is the normal case, not an
error), and the child pins vLLM's rendezvous and the NCCL/gloo bootstrap
sockets to loopback so CRIU does not bake a routable IP into the image.

**Cross-node restore also needs a byte-identical environment, and this
one is not checked up front.**  CRIU records every file-backed mapping by
absolute path *and* size, then re-validates the size when it reopens the
file at restore.  One mapped file of a different length aborts the entire
restore:

```
Error (criu/files-reg.c:2175): File <path> has bad size <local> (expect <image>)
Error (criu/mem.c:1467): `- Can't open vma
Error (criu/cr-restore.c:2331): Restoring FAILED.
```

The venv is where this bites in practice: two nodes that installed the
same requirements at different times end up with different builds of some
compiled extension, and CRIU aborts on the first one it hits rather than
reporting them all.  Unlike the `vllm_config` and `model_dir` checks
above, there is no early raise — the failure surfaces from inside CRIU.
Reinstalling from the same requirements is *not* sufficient, because
identical version specs routinely yield different bytes; copy the tree
itself (`rsync -a`, or a tarball — never `tar -h`, which dereferences the
venv's symlinks and changes sizes) to the identical absolute path.

**Check an image against the node before spending a restore attempt.**
`scripts/imgdiff.py <image_dir>` decodes the image's `files.img` and
reports every recorded mapping whose local size differs or whose file is
missing, so a cross-node environment mismatch takes seconds to diagnose
instead of a failed restore.  It also compares the ELF build-IDs CRIU
recorded, which catches a same-size-but-different-build library that
would pass CRIU's size check and then map the wrong text pages.  The one
entry it always lists and that is never a problem is the `/dev/shm/sem.*`
ghost file: unlinked at dump time, carried inside the image, recreated by
CRIU at restore.

**A restored child runs the code frozen in the image.** Editing
`vllm_child.py` has no effect on an existing image: the child resumes
in-memory code and re-imports nothing from disk.  New behavior only
appears after an offline re-dump with the updated code.  Keep dump-time
code and runtime code the same checkout.  This is a common source of
"my fix did nothing" confusion.

**Root is not required.** CUDA checkpoint/restore always uses the libcuda
driver API, which needs ptrace permission over the target -- something a
parent holds over its own same-uid child -- not root.  CRIU dump and the
low-cap restore both invoke `criu` at the worker's own uid with no `sudo`;
an unprivileged node needs only `SEMIP_UNPRIVILEGED=1` plus the capabilities
set on the binary (see `INSTALL.md`).  `Instance` spawns its worker and vLLM
child with `mp.spawn`, which inherits the uid, so the whole tree runs as
whoever launched it -- and dump and restore must be that same uid.  Only the
default PID-namespace restore path still needs `sudo`, for
`unshare(CLONE_NEWPID)`.

**Modules import their siblings by bare name.** `import semip_logging`,
`from instance import Instance`, and so on, so the package directory must
be on `sys.path`.  The lazy `__getattr__` in `__init__.py` installs that
entry on first attribute access, which is what makes
`from arctic_platform.inference.semi_persistence import Instance` work while still
letting the spawned worker and child import flat.

**Concurrent restores are supported, deliberately.** Each restore runs
inside its own PID namespace, so the recorded PIDs cannot collide with a
sibling restore or with a zombie left by a destructive dump.  See
Complications 8 and 10 in [CRIU_PLUMBING.md](CRIU_PLUMBING.md).  Images
captured before that change carry a tty and must be re-dumped.

**`SEMIP_UNPRIVILEGED=1` trades that concurrency for a lower capability
floor.** It is the single switch for running on pods that grant only
`CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE` (no `CAP_SYS_ADMIN`): the dump
gains `--unprivileged`, the restore takes a path with no private PID
namespace, and the child sheds its capabilities so the image is
restorable where `capset()` cannot grant real ones.  Two consequences
worth planning around:

- **At most one live restore per node**, since the recorded PIDs must be
  free.  `scripts/test_weights.py` restores two models concurrently and
  is therefore incompatible with this mode.
- **Portability is decided at dump time.**  An image dumped *without* the
  flag records real capabilities and cannot be restored on a low-cap node
  at all; you have to re-dump with the flag set.  Setting it on a
  privileged pod is fine and is the normal way to produce a portable
  image -- `--unprivileged` only skips a probe you do not need.

See Complication 11 in [CRIU_PLUMBING.md](CRIU_PLUMBING.md), and validate
a target node with `sudo criu check --unprivileged`.

---

## 5. Known gaps

- **Weight sync is not implemented.** The RL weight-update path is
  deliberately deferred.
- **Multiplexing more than one job per GPU with semi-persistence is
  unvalidated.**
- **Config compatibility is exact, not semantic.** The flat dict equality
  above is strict: it rejects configs that differ only in engine-internal
  or harmless keys. A richer model — bake the full effective engine
  config at dump time and validate a meaningful subset — is the eventual
  fix.
- **`logprobs` are surfaced by the child** but not plumbed through every
  adapter path.
