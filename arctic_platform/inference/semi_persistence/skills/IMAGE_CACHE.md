# The image cache: what it is today, and what a durable one must satisfy

A semi-p job's `model_dir` holds three directories with three different
lifetimes, three different sizes, and three different reasons to exist. Today
all three live on storage that dies with the pod, so **every DSS zone job takes
the miss path** — cold start, dump, restore — and semi-p buys correctness
without yet buying latency.

This document is the map for changing that: what the directories are, what the
existing model-distribution pipeline already does, what a job pod is actually
permitted to do (measured, not assumed), and the five constraints any durable
cache has to meet.

Sections 1-8 are requirements and measurements. **Section 9 is built and in
use**: an image published to the mirror is materialized into the derived
`model_dir` by copying the two path-bound directories and reading the weights
in place. Section 10 is the publish-side measurement that decided its shape.
**Section 11 is the multi-replica roll** (several replicas per pod, one
single-node dump serving `n_gpus=16`), measured end to end on 2026-10-03.

Path binding and the `vllm_config` check are in
[`semi-p_DESIGN.md`](semi-p_DESIGN.md); the DSS wiring is in
[`dss_integration.md`](dss_integration.md) §6; the pid-collision consequence is
Complication 8 in [`CRIU_PLUMBING.md`](CRIU_PLUMBING.md).

---

## 1. The three directories

Locally, under `$SEMIP_IMAGE_CACHE` (default `/data-fast/image-cache_neutrino`),
where a dump writes and a restore reads:

```
<model_dir>/                    = $SEMIP_IMAGE_CACHE/<cfg12>_<env12>
  image/         CRIU image of the child tree + meta.json
  weight/        shard_NNNN.bin + weights_meta.json (rank<N>/ at TP>1)
  compilation/   Triton / inductor / torch.compile / FlashInfer caches
```

**Several replicas in one pod** (`n_gpus=8` at TP=1, 2 or 4) each get their own
`model_dir`, one level down, and dump at the same time without touching each
other's lock, image or compile cache:

```
$SEMIP_IMAGE_CACHE/<cfg12>_<env12>/
  replica0/  image/ weight/ compilation/      <- model_dir of slot 0
  replica1/  ...
```

`K` is the replica's **node-local slot**, assigned by `ReplicaPool._node_slots`
from each actor's Ray node and passed as `SEMIP_REPLICA_ID` with
`SEMIP_NUM_REPLICAS`; a pod holding one replica (TP=8 at `n_gpus=8`) keeps the
flat layout. Because slots restart at 0 on every node, an `n_gpus=16` or `32`
job resolves on each node exactly the layout a single-node dump wrote. That is
the intended workflow: dump and publish at `n_gpus=8`, wait until every node
is verified, then run the multi-node job, which restores and never dumps. It
is a rule, not a check. A replica knows only its own node's slot and count, so
a multi-node miss cold-starts and dumps on every node independently. The
replicas share one key: at TP>1 the pod's device set is in `cfg12` already, and
at TP=1 it is folded in only when the pod holds several replicas, which keeps a
1-GPU and an 8-GPU pod (two shapes) on two keys. Nothing about the replica set
goes into `meta.json`; the publisher reconstructs it from the directories.

**The published tree is shaped differently, and §9.1 is why.** Only `image/` and
`compilation/` are bound to `model_dir`, so only they have to travel together
under a key the engine can derive. The weights are path-free and turn out to be
*identical* across backend images, so they are published once, content-addressed,
and shared:

```
s3://<bucket>/image-cache/<name>/
  skeleton/<cfg12>_<env12>_<wt12>/  image/ compilation/
                                    (or replica<K>/{image,compilation}/, one sentinel)
  weight/<wt12>/                    shards + weights_meta.json (rank<N>/)
```

A multi-replica dump is published with `semip_publish.py <root>/<cfg12>_<env12>`.
The publisher requires a contiguous `replica0..N-1`, each recording its own
`model_dir`, all agreeing on config, image, driver and uid; it hashes every
replica's `weight/`, refuses unless the hashes are equal, and uploads one copy.

> **A node-spanning dump (`node<k>/`) cannot be published yet.** The publisher
> knows the flat and `replica<K>` layouts only. It also cannot simply learn the
> new level: each pod holds just its own half, so there is nothing for one
> invocation to walk. It needs a two-pod rendezvous -- per-node rows under
> `_staging/<key>/<dump_id>/node<k>.json`, a `wt12` over the union, each node
> uploading its own `rank*` and its `node<k>/`, and node 0 writing the sentinel
> last so the DaemonSet still sees one model directory. See
> [MULTINODE_TP16.md](MULTINODE_TP16.md) §10.
The skeleton carries **one** sentinel over every `replica<K>/`, so a node
verifies all replicas at once -- and the engine treats a pod that cannot
materialize every replica as a failed job rather than mix restores with cold
starts, whose new processes would land on the task ids the siblings' images
recorded.

`<name>` is the **last element** of `vllm_config["model"]`, not the whole thing.
dss resolves that field through `resolve_model_path` before the engine sees it, so
it arrives absolute (`/mnt/neutrino/base-models/Qwen/Qwen3.6-35B-A3B`); publishing
it verbatim nested the mirror's own mount inside itself and would fork the shared
weight directory if that mount ever moved. `semip_engine._model_slug` and
`semip_publish.model_slug` both take the last two segments and
`test_publish_layout` cross-checks them, because publish writes where the engine
reads and a disagreement is a permanent, silent miss rather than an error.

The four names -- `image`, `compilation`, `weight`, `skeleton` -- are the same
in both trees. `weight/` was `weights/` locally until the two were unified, and
that rename is not free to repeat: `semip_publish.weights_hash` folds the local
directory name into the weight hash, so every directory published under the
plural has a hash no dump produces again, and each model needs one re-dump and
one re-upload before its skeletons resolve. `tests/test_layout_names.py` pins all
four names across `instance.py`, `vllm_child.py`, `semip_engine.py` and
`semip_publish.py` -- none of which can import the others -- and fails on a bare
literal, which is what keeps this section true.

Sharing a name does **not** remove the scoping the publisher does. A manifest's
paths are relative to the directory that holds it, so `_scoped_manifest` strips
the `weight/` prefix for the weight sentinel regardless of what the local
directory is called. Getting that wrong is §11's 404 loop, not a naming problem.

Both layouts coexist by necessity rather than choice: `model_dir` is fixed at
process startup, because `compilation/` is written under it during engine build,
long before the weights whose hash names them exist. So the weight hash can never
appear in the local directory name, and the local cache stays flat.

The binding lives in the skeleton's **name**. Nothing inside a skeleton
references a weight hash, so there is no pointer to flip, nothing mutable inside
a digest-verified manifest, and no way for the halves to disagree -- the dump that
produced one produced the other. Two weight versions of one config are two
skeleton directories that coexist, which is also the rollback.
`_resolve_published_skeleton` lists `skeleton/` for `<cfg12>_<env12>_*` and
**refuses on more than one match**, cold-starting rather than guessing which
weights a job wants.

Measured on `qwen_35b` (`Qwen/Qwen3.6-35B-A3B`, TP1) in the pod that produced
job `2469a638-2290-44eb-897e-16075046c00a`:

| Directory | Size | What it is | Re-derivable without a cold start? |
|---|---|---|---|
| `weight/` | 66 GB | the flat pinned CPU buffer, byte-for-byte | **no** |
| `image/` | 6.4 GB | CRIU's dump of process memory | no |
| `compilation/` | 306 MB | content-keyed compile artifacts | no (but cheap to re-warm) |

**`weight/` is not a copy of the safetensors already on the node.**
`_semip_save_weights` writes `memoryview(worker._semip_buf.numpy())` in
`shard_bytes`-sized ranges, with a `layout` index of
`[name, offset, nbytes, dtype, shape]`. That buffer is the *post-load* parameter
layout: dtype-converted, fused, quantized, and TP-sharded per rank. Producing it
requires loading the model, which is the cost the cache exists to avoid. So the
73 GB is the price of a durable cache, not an artifact that can be trimmed by
pointing at `/mnt/neutrino/base-models`.

The one real lever is that `compilation/` is **reused and never cleared** on
re-dump (see `semi-p_DESIGN.md` §2), so it is the cheapest thing to make durable
and the only one that helps a cold start rather than replacing it.

---

## 2. Today's lifetime: `/data-fast` is an `emptyDir`

`dss_integration.md` §6 calls `/data-fast` "node-local storage, not part of the
pod image". That is right about the media and misleading about the lifetime, so
state it precisely:

```
vol data-nvme   {'emptyDir': {}}
  mountPath /data-fast              name data-nvme
  mountPath /tmp                    name data-nvme  subPath tmp
```

An `emptyDir` is **pod-scoped**. The bytes land on the node's NVMe RAID
(`/dev/md127`, 28 TB, also where `/mnt/k8s-disks` lives), which is why `df`
inside the pod shows 28 TB and makes the directory look node-persistent. It is
not: kubelet deletes the volume when the pod goes away. A new zone job on the
**same node** gets a fresh, empty `/data-fast`.

Consequence, and it is the whole motivation for this document: the hit path in
`restore_and_wrap` is unreachable as deployed. Every job finds no
`image/meta.json`, dumps its own image, restores it seconds later in the same
pod, and throws it away at teardown.

---

## 3. The distribution pipeline that already exists

Do not design a cache without it — most of the hard parts are built.

```
<model stage>                                (Snowflake stage, source of truth)
   |  sync_cluster_bucket.py   CronJob, every 15 min, additive only
   v
s3://$MODEL_BUCKET                           (per-cluster bucket)
   |  sync_node_cache.py       DaemonSet neutrino-model-cache, every 300 s
   v
/mnt/k8s-disks/0/neutrino                    (hostPath, one copy per GPU node)
   |  mounted readOnly by every device-manager pod
   v
/mnt/neutrino/base-models/<org>/<model>
```

Measured on a dev cluster, 2026-09-17. The DaemonSet runs on 8 nodes
(`workergroup in {neutrino-pool-h200, neutrino-pool-b200}`) with:

```
python3 scripts/sync_node_cache.py --bucket $(MODEL_BUCKET)
  --cache-dir /mnt/k8s-disks/0/neutrino --interval 300
  --serve-port 8081 --parallel 4 --label-node
```

Four properties matter for an image cache:

- **A model dir is whatever holds a `_neutrino_manifest.json` sentinel**, written
  last, carrying per-file SHA-256. `_discover_model_dirs` finds nothing without
  it, so riding this pipeline means producing that manifest.
- **The node sync is a mirror, including deletions.** `_remove_orphans` does
  `shutil.rmtree` on any cached dir absent from the bucket. The cluster CronJob,
  by contrast, is **additive only** — `sync_cluster_bucket.py` contains no
  `delete_object`, and its only `unlink` is of a local scratch file. So an
  object written into the bucket by something other than the CronJob survives
  it, and propagates to every node.
  **Emptying the whole `image-cache/` prefix is not the same as retiring its
  keys one at a time.** S3 has no directories, so with no object left underneath
  it the prefix stops existing at all, and the daemon reads the intermediate
  `image-cache/` directory itself as an orphan and rmtrees it on every node.
  `SEMIP_IMAGE_SOURCE` then names a path that is gone — a supported
  "cold-start on a miss" — so every semi-p job on every node quietly stops
  using the cache, with nothing logged above WARNING, until a publish recreates
  the directory and a sync pass lands. Retire *after* the replacement's
  manifest is in the bucket, never before.
- **Every node pulls every model dir.** There is no per-node selection (the P2P
  seed tree in `model-availability.md` is documented but not built). A 73 GB
  image published this way costs 73 GB on all 8 nodes and 8× the egress.
- **Node labels already gate placement.** As each model reaches `complete` the
  daemon stamps `model.neutrino.snowflake.com/<sanitized>=ready` on its node,
  and the operator turns `spec.requiredModels` into a
  `requiredDuringSchedulingIgnoredDuringExecution` nodeAffinity on
  device-manager pods. **This is the mechanism a durable image cache wants** —
  "schedule me where my image already is" is solved, not open.

---

## 4. What a job pod may actually do

Measured 2026-09-17 from the `device-manager` container of a sampling job's
pod, and independently from a `neutrino-model-cache` pod. Both
assume the same role, so this is a property of the `neutrino-sa` service
account rather than of one workload.

| | |
|---|---|
| Identity | the cluster's `neutrino` S3 role, assumed through the service account |
| Mechanism | IRSA — `AWS_WEB_IDENTITY_TOKEN_FILE`, no static keys |
| Tooling present | `/usr/local/bin/aws`, `boto3`, `s5cmd` (sync image) |

| Operation on `s3://$MODEL_BUCKET` | Result |
|---|---|
| `ListBucket` / `GetObject` / `aws s3 cp` down | OK |
| `PutObject` / `aws s3 cp` up | **OK** |
| `DeleteObject` / `aws s3 rm` | **OK** |
| `ListAllMyBuckets` | denied (no identity-based policy) |

So the answer to "can a job pod publish its image" is **yes, with no new
credentials and no new infrastructure**. `/mnt/neutrino/base-models` itself is
not writable and never will be — it is a `readOnly` hostPath mount — but the
bucket one hop upstream is.

> **Blast radius, worth fixing independently of semi-p.**
> `model-availability.md` §"Cache data access control" asks whether a training
> pod can modify the cache and answers "No — the mount is `readOnly: true`".
> That holds for the mount and not for what the mount mirrors. Every pod with
> `neutrino-sa` can `DeleteObject` the model catalog from the source bucket,
> and `_remove_orphans` propagates that to all 8 nodes within 300 s. A durable
> image cache makes job pods *routine* writers to this bucket, which raises the
> stakes: scope the job-pod policy to a prefix, or give the publisher a
> separate identity, before wiring one up.

---

## 5. Five constraints a durable cache must satisfy

These are established elsewhere; collected here as requirements. Do not
re-derive them.

**1. The image is bound to its `model_dir` and is not relocatable.**
`criu_restore` rejects any other path by name, because the image bakes absolute
compile-cache mappings. *This one is already satisfied, for free.* `/data-fast`
is a fixed `mountPath` in every device-manager pod, so
`/data-fast/image-cache_neutrino/<model>` is the **same absolute path on every
node**. A cache that downloads into that path — rather than restoring in place
from the hostPath mirror — meets the binding by construction. A cache that tries
to restore out of `/mnt/neutrino/base-models/...` cannot.

**2. An image binds to its file-backed mappings byte-for-byte.** CRIU
re-validates the recorded size of every mapping at restore, and a single
differing compiled extension aborts from inside CRIU with no up-front check.
Inside a container these mappings come from two places, and the distinction
decides how portable an image is: the **container image** (deterministic given
the digest — the successful run was
`dss-backend:dev_20260916_235834_7af90823e27 @sha256:9af620dc…`) and
**host-injected driver libraries**, which follow the node. So the container
image digest is certainly part of the cache key; whether the node's driver
version must be too is *not yet measured*. `scripts/imgdiff.py <image_dir>`
answers it in seconds for a specific pair, and should be run before the first
cross-node restore rather than reasoned about.

**3. Capability level and uid are fixed at dump time.** `SEMIP_UNPRIVILEGED=1`
must have been set *when the image was dumped*, and `meta.json` records the
dumping uid, which `Instance.criu_restore` rejects on mismatch. Both belong in
the cache key, not just the model name.

**4. GPU placement is checked.** `meta.json` carries the capture node's
`gpu_uuids`, and `_check_gpu_placement` refuses a foreign-UUID image landing on
the dumped indices. Cross-node restore is supported
([`CROSS_NODE_RESTORE.md`](CROSS_NODE_RESTORE.md)) but not unconditional.

**5. Size.** 73 GB per model per node at the measured sizes, against 20 TB free
on a node that also holds the base-model catalog and every co-located zone's
`emptyDir`. `model-availability.md` already budgets this filesystem carefully
(the training `sizeLimit` was cut from 26 TB to 16 TB to make room for the
catalog); adding an image cache spends from the same budget and needs the same
accounting.

---

## 6. The cache key is the directory name

`model_dir` is **derived, not supplied**. There is no `semi_p_model_dir` job
field; `restore_and_wrap` computes the path before it knows whether an image
exists:

```
$SEMIP_IMAGE_CACHE/<cfg12>_<env12>[/replica<K> | /node<k>]

cfg12 = sha256(vllm_config, sorted keys, values verbatim
               + NUL + sorted device nodes, at TP>1 or with several
                 replicas per pod)[:12]
env12 = sha256(container image digest + nvidia driver version)[:12]
K     = node-local replica slot, present only with several replicas per pod
k     = node rank, present only when one engine spans pods (nnodes > 1)
```

`replica<K>` and `node<k>` are different axes and do not nest in practice: a
node-spanning engine is a single replica by construction, since its placement
group holds the whole `world_size`. So a key carries one or the other.

**`nnodes` is in `cfg12`; the rest of a node's identity is deliberately not.**
The split changes the image -- half of a TP=16 group is not a TP=8 engine -- so
it is hashed. `node_rank`, `master_addr`, `master_port` and the interface travel
in the `MultiNode` parameter instead, because they change on every restore: with
them in `vllm_config` the two halves of one job hash differently and no restored
pair can match its own image. Weights also stay at the key level rather than
under `node<k>/`, since the shards are named by *global* rank
(`weight/rank{0..15}`) and a restore onto a different pod pair has to find all
of them in one place. See [MULTINODE_TP16.md](MULTINODE_TP16.md).

**The third term is discovered, not derived.** A published skeleton is named
`<cfg12>_<env12>_<wt12>`, where `wt12` is a content hash of the staged weight
buffer -- and that is exactly what a restore does not have, since producing it is
the cold start being avoided. So the two derivable halves name a *prefix*, and
`_resolve_published_skeleton` lists the mirror to find the rest. That is the one
place the "derived, never listed" rule is broken, and it is broken deliberately:
the alternative is a mutable pointer file, which reintroduces the disagreement
the name-as-binding eliminates.

`wt12` is composed in `semip_publish.weights_hash` from the per-file SHA-256s the
sentinel manifest already needs, over sorted `(relpath, sha256)` rows. Both
halves of each row matter: sorting makes two nodes agree on the same bytes
whatever order they walk the tree in, and including the path means a TP change
cannot alias onto the layout it replaces, since TP1's flat `shard_NNNN.bin` and
TP>1's `rank<N>/` differ in name before they differ in content.

`cfg12` is taken over the post-strip `vllm_config` — the dict handed to
`Instance(vllm_config, model_dir)` and recorded in `meta.json` — so the config
printed beside an image is the config that named its directory. `env12` covers
what CRIU re-validates: the image digest fixes the ~396 `/usr` mappings, since
layers are content-addressed, and the driver version fixes the one mapped file
the image does *not* ship (see §5 constraint 2).

**The digest term also partitions by GPU architecture, for free.** The gateway
resolves a job's `image_tag` to a hardware-specific variant — `-h` for Hopper,
so `dev_20260917_184643_435d85eab89` runs as
`dev_20260917_184643_435d85eab89-h` — and that variant has its own digest. So
the same config on an H200 and on a B200 lands on different `env12` values and
cannot share a directory, which is what you want: different compiled kernels,
and `cuCheckpointProcessRestore` could not migrate between them regardless. No
hardware term needs adding to the key, and the node `ready` labels partition by
architecture as a consequence.

One practical trap follows: the digest a build reports **may or may not** be the
one that keys the cache, since `_pod_image_ref` reads the pod's own
`containerStatuses[].imageID` — the resolved variant. Observed both ways within
an hour: one build reported `sha256:6c7c2b6f…` while its pod ran
`sha256:5a8fee66…`, and the next reported `sha256:73623a81…` and its pod ran
exactly that. So do not infer the key from a build log. Read `image_ref` out of
the dump's own `meta.json`, which records what the pod actually ran.

**Both terms are exact identifiers rather than samples, and that is the whole
point.** A key derived from a guessed sample of the filesystem can collide, and
a collision has nowhere else to put the new image: the cold start overwrites the
image it just rejected, and two pods in that state overwrite each other
forever. Exactness is what makes every miss productive instead of a livelock,
and it is why there is no fallback — a pod that cannot identify its own
environment fails rather than caching against a weaker name.

The error directions are deliberately asymmetric. Too *fine* a key costs a
cold start, which is recoverable. Too *coarse* aliases two different
environments onto one directory, which is not. So the design errs fine
everywhere it has a choice: `model` keeps its absolute path rather than being
normalized to a bare model name, and a rebuilt container image invalidates
every image cached under it.

### The device allocation is part of `cfg12` at TP>1

A TP>1 image is restorable **only** in a pod that can reopen every device node
its captured state refers to, so the allocation is part of the image's identity
and `_device_binding` folds it into `cfg12`. At TP=1 it contributes nothing, and
must not: a TP=1 image renumbers onto whatever slot it lands on, so binding it
would split one universally restorable image into one key per GPU and turn a hit
rate of 1 into 1/N. The threshold is `_DEVICE_BOUND_MIN_TP`, read by both the key
and `_missing_device_nodes`, because a key that bound the devices where the check
did not would publish images nothing looks up, and the reverse is the placement
lottery this replaced.

**One exception, by design: TP=1 with several replicas per pod binds too**
(`_device_binding(vllm_config, replica_count)`, since 2026-10-03). Not because a
TP=1 image cannot move -- it still renumbers onto any GPU, and
`_missing_device_nodes` stays TP>=2 only -- but because a 1-GPU pod (flat
layout) and an 8-GPU pod (`replica0..7/`) are two shapes of the same config,
and they must not share a directory. Folding in the pod's `/dev` set separates
them, and makes a TP=1 multi-replica key behave like a TP>1 one. A single
TP=1 replica keeps its old key.

**Why this is not "placement leaking into the key".** It reads like the thing a
cache key should never depend on, and until this landed the code said so in as
many words. The correction is that the device set is not placement, it is a
property of the captured state: `SEMIP_GPU_MAP` keeps every group GPU visible at
TP>1 and each rank opens *all* of them, so the dump records device paths a
differently-allocated pod does not have. Excluding it did not make an image
portable, it made it *findable and unusable* — the restore then failed inside
`ncclCommInitRank` as an unrelated-looking "unhandled system error", or, once
`_missing_device_nodes` existed, was declined after the copy had already been
paid for. Putting it in the key makes "found" and "restorable" the same
statement.

Two consequences, both wanted:

- **Dumps accumulate instead of overwriting.** One config on two slots used to
  resolve to one directory, so the second dump destroyed the first and a config
  had exactly one restorable placement however many times it was dumped. Each
  allocation now writes its own directory, and published coverage ratchets up.
- **The exposure is a set, not a sequence.** `cfg12` sorts the device nodes,
  because an image restores onto the same devices in any order —
  `_gpu_migration_permutation` builds the bijection, and one dumped on
  `[3, 1, 2, 0]` was measured coming back on `[3, 2, 0, 1]`. Hashing the order
  would split an image into as many as `TP!` keys a single one already serves.

The GPU *ids* Ray assigns still never enter the hash. They are validated against
`tensor_parallel_size` in `_check_tp_matches_gpus` and otherwise discarded; what
enters is the `/dev` allocation, read by `_visible_device_nodes`.

**The cost is that variants multiply on `env12`.** A backend-image bump
invalidates every allocation's image at once, so a config that had one skeleton
to re-dump now has one per slot it has ever been dumped on, and only the
scheduler decides which it can re-dump. Nothing in the key mitigates that; being
able to *request* a device set would.

**`_visible_device_nodes` is now a key input, and it exists twice** — here and in
`worker.py`, which records the set at dump time. They always had to agree or the
check would refuse a restore that would have worked. Now drift makes every TP>1
job compute a key no publish ever wrote: a permanent silent miss rather than a
loud refusal, which is the `model_slug` failure mode. `test_image_cache_key`
pins the pair.

### What this replaced

Until this landed, `semi_p_model_dir` was the entire cache key: a bare path
from the job JSON, tested with `os.path.isfile` and carrying no config
fingerprint. On a hit the restore handed the image's *own* baked config to the
`Instance`, so a job whose `vllm_config` differed did not fail — it silently
served the baked config, with one `_log_config_divergence` line as the only
evidence. That was survivable only because the image died with its pod. It is
recorded here because the same hazard returns the moment any key stops being a
function of the config.

### Known gap: `extra_env` is in neither hash

`extra_env` reaches `InferenceWorker.initialize` as its own parameter, never
enters `engine_kwargs`, and so reaches neither hash — yet it can carry
`SEMIP_UNPRIVILEGED`, and capability level is fixed at dump time (constraint 3).
Two jobs differing only in that flag would produce different images under one
directory name. Narrowed as of 2026-10-02: `restore_and_wrap` defaults the flag
to `1`, so a job need not set it at all, and `_dump` records the effective value
as `unprivileged` in `meta.json`. A local image under the other value refuses to
restore; a published one is a miss.

Closing it needs an allowlist rather than hashing `extra_env` wholesale: a
device-manager pod was observed carrying
`ARCTIC_ROUTER_REPLAY_SHM_SCOPE=<job's zone name>`, which is job-unique, and
any such value in the key gives every job its own `cfg12` and a permanent miss.
`ARCTIC_INFERENCE_ENABLED` belongs on that allowlist too, since
`arctic_inference_effective_enabled(extra_env)` reads it from there.

---

## 7. Invalidation, and where it is already enforced

A durable cache needs one honest answer to "what makes an image stale". The
enforcement points exist; they are just not joined up:

| Change | Detected by | Today's behaviour |
|---|---|---|
| `vllm_config` | `cfg12` in the directory name | resolves elsewhere — see §6 |
| device allocation (TP>1) | `cfg12` in the directory name | resolves elsewhere; `_missing_device_nodes` is the backstop for images keyed before this |
| container image | `env12` in the directory name | resolves elsewhere |
| driver version | `env12` in the directory name | resolves elsewhere |
| dumping uid | `Instance.criu_restore` | raises before spawning |
| capability level | recorded `cap_bnd`, restore-time | deadlock inside CRIU (Complication 11) |
| GPU identity | `_check_gpu_placement` | raises |
| library bytes | `meta.json`'s `env_files`, or `scripts/imgdiff.py` | recorded at dump; checking it is the hit path's job |
| `extra_env` | nothing | see the known gap in §6 |
| model weights | nothing | — |

The last row is the gap a durable cache introduces. The base catalog already
solves the same problem one layer down — a changed per-file SHA-256 in
`_neutrino_manifest.json` is what re-pulls a model, with no version concept —
and an image cache should borrow that shape rather than invent a version
number: the image's identity is the digest of its inputs.

---

## 8. Durability re-activated the pid-collision cause, and the dump-side floor

Complication 8 lists five ways a recorded pid is already taken at restore.
Four are handled by the subreaper-plus-sweep fix. The fifth — **the restoring
pod's own launcher squatting the image's recorded range** — was unreachable
while the cache was not durable: dump and restore happened in one pod seconds
apart, so every recorded id was one that same pod had allocated moments
earlier, and the only possible holders were its own processes.

Publishing an image ended that, and the cause fired on the **first** cross-pod
restore ever attempted (2026-09-18, job `8bc26249`, 5/5 attempts):

```
criu restore aborted (recorded task-id collision: the image needs 51 task
id(s) in 1105-1340; occupied by pid 765 (ray::InferenceW, an ancestor of this
restore): 1114.
```

It is structural rather than unlucky. Every device-manager container gets a
fresh PID namespace and runs the same workload, so ids land in the same low
range every time: Ray's `InferenceWorker` near 765 with threads past 1114, and
an image dumped in an identically shaped pod records 1105-1340. **The ranges
overlap by construction**, and the occupant cannot be evicted — 1114 is a
*thread of the actor performing the restore*, which is why all five retries
failed identically.

### Why the other three strategies do not reach it

Handoff 4 §5's table still holds; this case simply falls outside every row of
it. **Reaping** only reaches holders in our own subtree, and this holder is an
ancestor, not a descendant. A **private PID namespace** would make every
recorded id free, but needs `CAP_SYS_ADMIN` for `unshare` plus a private
`/proc`, which is exactly what `SEMIP_UNPRIVILEGED=1` does not have.
**Cold-starting on the collision** — the fallback this section used to
prescribe — is not a fix but a surrender: the collision is deterministic, so
every cross-pod restore would take it, and a published image would never once
be used.

**Rewriting the ids in the image is not available either**, and the reason is
worth recording so it is not re-proposed. A pid is not just metadata: the image
files carry it in `pstree.img` and the per-task file names (`core-<pid>.img`,
`mm-<pid>.img`), which is merely tedious — but the dumped process's *own
memory* carries it too, and that is not enumerable. Every glibc `pthread`
descriptor holds its own tid; PI and robust mutexes hold owner tids in the
futex word and in the kernel `robust_list`; NCCL and the CUDA driver cache tids
internally. Rewrite `pstree.img` and a thread runs as 200114 while its own
descriptor says 1114, and the first robust-mutex operation resolves that
badly. Ids also leak into *names* — a real `env_files` here contained
`/dev/shm/sem.b4VAUg`. CRIU offers no pid remapping for these reasons, and
`clone3(set_tid)` wants the exact recorded ids.

That leaves choosing the ids **before the processes exist**, which is the one
strategy that needs no privilege and reaches holders outside our subtree.

### The floor

`_raise_pid_floor` in `server/semip_engine.py` runs in `_dump` immediately
before `Instance(...)`, and advances this PID namespace's counter past a floor
so the tree about to be created records ids far above anything a fresh
container hands out. Cheapest route first: the counter may already be past the
floor; one write to `ns_last_pid` moves it instantly where the namespace allows
one (these containers do not — `CapEff` is `0000000000000000`); otherwise
burners fork throwaway children until the counter clears it.

```
SEMIP_PID_FLOOR          100000   the floor; 0 disables the mechanism entirely
SEMIP_PID_BURN_WORKERS   8        parallel burners
```

Every failure path logs and returns `None`, leaving the dump exactly as it
would have been without this — per the contract that nothing here may fail a
job that would otherwise have run. The floor actually reached is recorded in
`meta.json` as `pid_floor`, so an image that predates this, or was dumped where
the floor could not be placed, says so rather than differing silently.

### Answering handoff 4 §5's rejection

Burning the counter was rejected on 2026-09-17 with five arguments, under the
instruction not to re-propose it without new ones. Each is answered here, and
two are conceded:

| The objection | What changed |
|---|---|
| Moves the counter for the whole pod — zone worker, Ray, sidecars — for a narrow need | The need is no longer narrow: it is *every* cross-pod restore. A high counter costs nothing; with `pid_max` at 4194304, ids are just numbers. |
| 200k ids is 10–15s of fork churn on every cold start | Measured: worse serially than estimated (540 µs/fork on these nodes, **96 s** for 200k), but the counter is namespace-global, so burners parallelise until they saturate the pod's CPU quota — 20k ids cost 10.8 s at 1 burner, 1.9 s at 8. A 100k floor is **~10 s**, and it is paid on the *dump* path, which already spends ~300 s, not on every cold start. |
| Protects only what starts after it — a fragile ordering constraint threaded through engine startup | It is one line in one function, directly before the `Instance(...)` that spawns the entire tree the image records. Nothing else produces ids that reach the image. `test_dump_places_the_floor_before_the_instance_and_records_it` asserts the ordering. |
| Bakes into the image; burned and unburned images are not interchangeable and every existing image needs re-dumping | **Conceded, and true.** `77d95928d2ac_04068da54f0c` had to be re-dumped. The alternative was a cache that is never used cross-pod, so a one-time re-dump is the cheaper side. `meta.json`'s `pid_floor` makes the two kinds distinguishable. |
| Not a guarantee: the counter advances and wraps at `pid_max`, so it buys a window not a proof | **Conceded.** But the window is quantified: a restoring pod has handed out ~1,500 ids by the time `initialize` runs, so ~100,000 process creations would have to precede a restore to threaten the floor, and the restore happens seconds into the pod's life. The preflight still names the collision if it ever happens. |

A second, subtler cause closes for free. `CROSS_NODE_RESTORE.md` notes that
CRIU's own restore helpers spawn into the same number space and can claim a
recorded id in the window *after* the preflight passes. Those helpers are
allocated from the restoring pod's counter, so they now land near 1,500 —
nowhere near the floor, where the previous images' ids sat.

`--burn` / `--burn-to` remain in `pidcheck.py` as the manual form, for a node
being investigated by hand. The automatic path does not call the script, which
does not ship in the image (§8 of handoff 7).

---

## 9. How a miss is filled from the mirror

Two shapes were open here. The deciding number in §10 settled it: **ride the
existing pipeline**, because 300 MB/s per node means pulling on demand in the
job pod trades a 5-minute cold start for a 4-minute download, while the
DaemonSet does the same pull *before* placement and off the job's critical
path. Pulling in the job pod is not built and should not be.

What is built is the consuming half, `_materialize_from_source` in
`server/semip_engine.py`, wired as a third branch between the hit and the cold
start and running inside the existing dump lock:

```python
with _dump_lock(model_dir):
    if os.path.isfile(meta_path):          ...   # another job got there first
    elif _materialize_from_source(source_dir, model_dir): ...   # this section
    else:                                  _dump(...)            # cold start
```

```
SEMIP_IMAGE_CACHE   /data-fast/image-cache_neutrino        writable, pod-local, canonical
SEMIP_IMAGE_SOURCE  /mnt/neutrino/base-models/image-cache  readOnly mirror, outlives pods

<cfg12>_<env12>/
  image/         6.4 GB   COPY   CRIU's -D dir; a restore writes into it
  compilation/   306 MB   COPY   the child's live compile cache
  weight/       ~66 GB    READ IN PLACE
```

Both roots are composed from one key in `_resolve_model_dir`, so they cannot
name different directories. **Both values above are the defaults** as of
2026-10-02, so a job sets only `semi_p: true`: they are cluster facts owned by
the neutrino operator (`/data-fast` is its `data-nvme` volume, and the mirror is
its `modelCacheMountPath` plus the DaemonSet's `image-cache` directory), and
every pod has to agree on them anyway. Either variable still overrides its
default. `SEMIP_IMAGE_SOURCE=""` turns the mirror off -- cold-start on a miss,
the A/B switch -- which is distinct from leaving it unset.

### Why `weight/` is read in place, and the other two are not

`env_files` from the first successful dump has **0 of 416** mappings under
`weight/`, so no absolute path is baked into the shards. The dump order is
why: `save_weights` → `detach` → `sleep` → `cuda_checkpoint` → `criu_dump`, so
the files are outputs and the writer has detached before CRIU runs. The seam
already existed too — the restore calls `inst.attach().load_weights()`, and
`load_weights(weights_dir=...)` takes the override; `_semip_load_weights` is
read-only, reads in parallel over `io_workers`, validates `total_bytes` and
`layout` against the attached model, and appends `rank{N}` itself at TP>1.

So reading in place is strictly better than copying: 66 GB read once instead of
written then read, no second copy per node, and the `_remove_orphans` exposure
window (§5) is the few seconds of `load_weights` rather than the job's lifetime.

**And it is what makes the weights shareable, which is the larger win.** Measured
2026-09-24 against the published manifests' per-file SHA-256s: of the 45 keys in
the cache, 9 config hashes appear under more than one env hash, and **16 of 16
cross-env weight sets are byte-identical**. `6bad1bfa6033` alone is published
under five backend images with the same 70.4 GB of weights each time. A backend
image change does not perturb the staged parameter buffer.

The recurring cost that buys is the thing worth quoting. Every node pulls every
model dir (§3), so under the flat layout a new backend image cost a full weights
copy on all 8 nodes: about 560 GB fleet-wide for a 70 GB model, and ~5.7 TB for
GLM-5.3, whose weights are 759.9 GB across 360 shards. Content-addressing them
collapses that to zero for any config whose weights did not change, which is all
of them so far.

Two axes behave differently and it matters for sizing. The **env** dimension
collapses completely. The **TP** dimension does not collapse at all, and cannot:
the staged buffer is per-rank, so TP1/2/4/8 of one model measured 70.2 / 70.3 /
70.4 / 70.7 GB across 33 / 34 / 36 / 40 shards — growing with rank count from
per-rank shard padding and replicated parameters. Those are four distinct weight
hashes by construction, and `tensor_parallel_size` is in `cfg12` anyway, so they
were never candidates for sharing.

**The hazard the split introduces is the dangling reference**, and it is silent.
Weights are shared, so retiring a hash can orphan a skeleton that still names it;
the restore then declines and cold-starts with nothing above WARNING. Hence
`--unpublish-weights` refuses while `_skeleton_referrers` finds any skeleton
whose name ends in `_<wt12>` — a listing, not a fan-out of file reads, which is
the other reason the binding went into the name.

`compilation/` cannot follow, because it is the one directory whose absolute
paths CRIU baked as mmaps, and CRIU records mount and device for file-backed
mappings. `image/` cannot either: it is CRIU's `-D` dir, and the restore writes
`restore.log` and a pidfile into it. Read-only would need `--work-dir` plumbing
for 6.4 GB of savings.

### Every way it declines, and why none of them is an error

All of these log and return `False`, and the caller cold-starts. **Nothing in
this path can fail a job that would otherwise have run**, which is what lets
the checks be strict.

**Except with several replicas in the pod** (`strict=True`). The replicas
cannot mix: one cold-starting beside siblings that restore puts its new
processes on the task ids their images recorded. So the checks every replica
answers identically -- the verified marker (one, at the skeleton's top, over
every `replica<K>/`), the recorded `model_dir`, uid, weights, environment
libraries, devices, `unprivileged` -- stay misses, and the whole pod
cold-starts together. Anything that can fail one replica alone -- its
`replica<K>/` missing from the skeleton, its `meta.json` unreadable, its copy
failing, its post-copy size check -- raises and fails the job (all-or-none).

| Condition | Why it matters |
|---|---|
| `SEMIP_IMAGE_SOURCE=""`, or the directory absent | Mirror turned off, or nothing published for this key |
| no `.neutrino_verified` | The daemon stamps it only after digest-verifying; without it the directory may be a sync in progress |
| `meta.json` unreadable | Nothing to validate against |
| its `model_dir` is not ours | `image/` and `compilation/` bake that path; `criu_restore` rejects any other by name |
| its `uid` is not ours | `criu_restore` refuses the mix up front |
| the `weight/<wt12>/` the skeleton names holds no `weights_meta.json` (nor `rank0/`), or carries no `.neutrino_verified` | We never copy weights, so the skeleton is unrestorable. Either the hash was unpublished while this skeleton still referenced it, or it has not finished syncing |
| more than one `skeleton/<cfg12>_<env12>_*` | Two weight versions for one config and backend. Which one a job should serve is an operator's decision, so this declines and names the candidates rather than tie-breaking |
| an `env_files` size mismatch outside `model_dir` | This node is not the environment the image was dumped in, despite `env12` agreeing. Run `imgdiff.py` |
| the copy raises | Any I/O failure, disk-full included |
| an `env_files` size mismatch under `model_dir`, after the copy | The copied compile cache is short or wrong; the flipped `compilation/` is removed too, so the cold start does not inherit it |

### The four properties the copy depends on

1. **Gate on `.neutrino_verified`, not on `image/meta.json` existing.** The
   manifest is written last but a partially synced directory can hold any
   subset of the payload until then. This is the check most likely to be
   dropped in review and the one the whole copy's safety rests on.
2. **Flip `image/` last.** Everything is staged under `<model_dir>/.incoming/`
   and `os.replace`d in the order `compilation`, `image`. `image/meta.json` is
   the hit predicate, so a materialize interrupted anywhere leaves a directory
   that reads as a miss. Same discipline as the DaemonSet's manifest.
3. **Preserve mode, then repair it.** `shutil.copytree` defaults to `copy2`,
   which preserves mode -- necessary, because CRIU re-validates the recorded
   mode of every path it re-maps, but not sufficient, because the source is
   downstream of S3 and its modes are already flattened. `_apply_recorded_modes`
   puts them back from `env_files` after each `os.replace`; see "The second
   blocker" below. Do **not** shell out to `cp -a`, which implies `-p`, tries
   to chown as non-root, and returns non-zero after copying fine.
4. **Check `env_files` at both scopes.** Recorded since the first dump and
   never read until now (§7). Outside `model_dir` it is checkable before
   spending the copy; under `model_dir` it verifies the copy before `image/`
   lands. A size mismatch is fatal because size is exactly what CRIU
   re-validates; a missing file is advisory, since ghosted files and `/tmp`
   scratch explain an absence benignly.

### What the first run measured, 2026-09-17

An image dumped on one node, published, and materialized on another. Job
`103cb435` dumped and published `77d95928d2ac_c8dd670c03a8`; job `b364459d`
picked it up on a different node.

| | |
|---|---|
| Publish, 1307 files / 77.3 GB (hash + upload) | 2.5 min, ~680 MB/s upload |
| Node sync, 77.3 GB pulled and digest-verified | ~7 min after the manifest, incl. up to 300 s of interval |
| **Materialize, 6.7 GB into `/data-fast`** | **inside a 26 s job, so ~350-450 MB/s** |
| Cold start it replaces | ~299 s |

> **A publish is not usable when it says it is, and the failure is silent.**
> `semip_publish.py` ends with "expect it on every node within 300 s", but that is
> the daemon's *interval*, not the time to a usable image: the node still has to
> pull the payload and digest-verify it before stamping `.neutrino_verified`, and
> the gate above is on the marker. Measured 2026-09-20 on an ~80 GB TP=2 key, the
> marker appeared **~12 minutes** after the publish reported success. The "< 60 s"
> figure elsewhere in this document was for a 7.1 GB image and does not
> generalise.
>
> What makes this expensive is that a restore attempted inside that window does
> not fail. It reads as a miss, **cold-starts, serves, and reports success** --
> which is correct behaviour and completely indistinguishable from a real restore
> unless you look at the timings. Four experiment runs were scored as successful
> restores before anyone noticed. The tell is a leading `init OK (~217s)` in the
> phase list: a genuine restore never runs `init`. Wait for
> `.neutrino_verified` on the node before believing any restore measurement.

Three things the unit tests could not have told us:

- **The marker gate fired on a real mid-sync directory.** At 43 GB of the 77 GB
  download, `image/meta.json` was already present *and* parsed *and* carried the
  correct `model_dir`, while 34 GB of weights were still in flight. The naive
  hit predicate would have accepted it.
- **The `env_files` fatal/advisory split earned itself immediately.** The
  pre-copy pass checked 397 mappings (416 recorded, less the 19 under
  `model_dir`) and found exactly one absent: `/dev/shm/sem.b4VAUg`, a POSIX
  semaphore. Treating a missing mapping as fatal would have refused a perfectly
  good image on its first use.
- **The key behaved across an image rebuild.** `cfg12` stayed `77d95928d2ac`
  while `env12` moved `1fad1e633806` -> `c8dd670c03a8`, which also orphaned the
  previously published directory, exactly as §6 says it should.

The restore itself was then refused by `_check_gpu_placement` -- see below.

### Still unmeasured

- ~~**Local copy throughput** for `image/` + `compilation/` (~6.7 GB) into
  `/data-fast`~~ — **answered 2026-09-18: ~5 s, ~1.3 GB/s.** Not read off the
  materialize's own log line, which is worker-side INFO and so never reaches the
  pod log; bracketed instead between the pre-copy `_check_env_files` WARNING
  (23:39:21.218) and the start of `criu_restore` (23:39:26.198), a window that
  also contains the flip, the mode repair and the post-copy check. Approximate,
  since Ray's log forwarding adds unknown latency to the first timestamp, but
  enough to settle the question: the copy is not the cost.
- **Read throughput of `weight/` off the hostPath mirror** versus off
  `/data-fast`. `io_workers` is the knob, and needs no new code.
- **Whether `/data-fast` and `/mnt/neutrino/base-models` share a filesystem.**
  If they did, `cp --reflink=auto` would make the copy a metadata-only clone.
  Expect different devices (hostPath under `/mnt/k8s-disks/0` versus the
  `data-nvme` volume), in which case reflink is unavailable. Hardlinks are out
  regardless: links cannot cross mount points.
- **Whether an image travels at all within one workergroup** — one
  `imgdiff.py` run against an image captured on a different node. The
  `env_files` preflight now answers the same question from the recorded sizes,
  before a restore is spent.

### The blocker the first cross-node restore hit

Measured, not predicted. The first materialize worked and the restore was then
refused by `_check_gpu_placement`, because Ray placed the job on GPU 0 — the
index the image had been dumped on, on a different node. One placement in eight
for a TP1 job on an 8-GPU node.

The cause was one line in `worker.py`:

```python
migrate = bool(old_gpus and new_gpus and list(old_gpus) != list(new_gpus))
```

The `oldUuid=newUuid` device map was built only when the *indices* changed,
never when only the *node* changed, so a same-index cross-node restore kept the
capture node's baked UUIDs and `cuCheckpointProcessRestore` failed with
`CUDA_ERROR_INVALID_VALUE`. That condition was sufficient for exactly as long as
an image could not travel; this section is what made it travel.

`_placement_changed` now tests index **or** node, and the refusal is retired
with it. What is left is an image with no recorded `gpu_uuids` restoring onto
its own indices, which warns. See
[`CROSS_NODE_RESTORE.md`](CROSS_NODE_RESTORE.md) §4.

### The second blocker: S3 does not carry POSIX modes

Also measured, and reachable only once the two above were gone -- the task-id
collision had been killing every restore about 0.3 s earlier, so nothing had
ever got as far as opening a mapped file:

```
Error (criu/files-reg.c:2294): File .../flashinfer/.../sampling/sampling.so
  has bad mode 0100644 (expect 0100755)
Error (criu/mem.c:1467): `- Can't open vma
```

S3 objects have no mode, so the publish → bucket → DaemonSet round trip
flattens every file to the syncing daemon's umask. On the same key,
`compilation/` went from 1183×`644` + 1×`755` + 1×`600` on the capture node to
1185×`644` on the mirror. The one file that moved is the one that matters: a
mapped `.so`, so CRIU checks it.

The fix records the mode at dump time and re-applies it at materialize time,
because after the round trip the image is the only place the original survives:

- `_record_env_files` already decodes `files.img`, where the `REG` entry
  carries `mode` (`33261` = `0o100755`) beside `name`, `size` and `build_id`.
  Rows are now `[path, size, build_id, mode]`. `_check_env_files` reads
  `entry[0]`/`entry[1]` positionally, so three-element images stay readable --
  they simply have no mode to apply.
- `_apply_recorded_modes` chmods **only paths under the subtree just copied**.
  That scope is a safety boundary, not an optimisation: `env_files` also lists
  `/usr/lib` and the driver libraries, which never went through S3, have
  nothing to repair, and must never be touched. It is called per subdirectory
  right after each `os.replace`, for the same reason `image/` is flipped last
  -- a trailing fix-up would leave a crash in between as a *hit* with bad
  modes, which fails a restore instead of cold-starting.
- It never raises. A mode it cannot set is logged, and the restore that then
  fails cold-starts, which is where we would have been anyway.

`weight/` needs none of this: it is read in place off the mirror, and 0 of the
416 recorded mappings live under it -- which is the same fact that makes
reading it in place safe.

**Rejected:** CRIU's `--skip-file-rwx-check`. One flag instead of all of the
above, but it drops the check for every restore, and the mode is real
information about whether the file is the file.

---

## 10. It works: measured end to end, 2026-09-17

Publishing a dumped `model_dir` into the cluster bucket distributes it to every
GPU node with **no new infrastructure**. Done with
[`scripts/semip_publish.py`](../scripts/semip_publish.py) from the
device-manager pod of job `bddbebc8-8a90-4447-9f8a-608953058497`:

```
config hash : 308b26d5fae5   (from effective VllmConfig)
image hash  : 77fa3abddc66   (dss-backend@sha256:77fa3abddc667d5a...)
destination : s3://$MODEL_BUCKET/image-cache/qwen_35b/308b26d5fae5-77fa3abddc66
1273 files, 7.1 GB  (image/ + compilation/; weight/ withheld for speed)
```

| Step | Elapsed |
|---|---|
| hash 1273 files + upload 7.1 GB (`aws s3 cp --recursive`) | 53 s |
| DaemonSet pass on another node pulls it (`s5cmd`, `numworkers=4`) | 24 s |
| publish → visible and digest-verified on a different node | < 60 s |

It appeared on the node that ran a *different* job, at the expected path, with
the daemon's own `.neutrino_verified` marker:

```
/mnt/neutrino/base-models/image-cache/qwen_35b/308b26d5fae5-77fa3abddc66/
  image/  6.4G    compilation/  306M    _neutrino_manifest.json    .neutrino_verified
```

The daemon reports it as an ordinary catalog entry
(`image-cache/qwen_35b/308b26d5fae5-77fa3abddc66=complete`, 21 models where
there were 20) and stamps the placement label on every node holding it:

```
model.neutrino.snowflake.com/image-cache.qwen_35b.308b26d5fae5-77fa3abddc66 = ready
```

That label is the whole point: the operator already turns `spec.requiredModels`
into nodeAffinity, so "schedule this job where its image already is" needs
configuration, not code.

**The throughput number that decides §9.** 7.1 GB in 24 s is ~300 MB/s per
node, so a full 73 GB `model_dir` extrapolates to **~4 minutes** — against a
~5 minute cold start (`create-job` measured 298.88 s end to end, of which the
restore proper is 3.2 s, weights 1.4 s, graph recapture 4.4 s). Pulling on
demand in the job pod therefore saves almost nothing: it trades a 5-minute cold
start for a 4-minute download. **Only the label-gated shape is worth
building** — the pull has to happen *before* placement, off the job's critical
path, which is exactly what the DaemonSet already does.

Three caveats on this measurement. `weight/` was withheld, so the 73 GB figure
is extrapolated rather than observed, and s5cmd's per-file parallelism may not
scale identically to 33 shard files of 2 GB each. Nothing here restored from
the published copy — §9 is the consuming side and is unrun. And this
measurement predates the derived key: it was published under
`image-cache/qwen_35b/308b26d5fae5-77fa3abddc66`, a key nothing resolves to
now, so that directory is orphaned and only the *throughput* number carries
over.

Note also that the 73 GB is what §9 no longer moves. Only `image/` and
`compilation/` are copied to the node; the 66 GB of shards are still published
and still distributed, but they are read off the mirror in place rather than
copied a second time into `/data-fast`.

Clean up with `semip_publish.py <model_dir> --unpublish-skeleton` (and
`--unpublish-weights <wt12> --model <org/model>` once no skeleton references
the hash); deletion propagates to every node within 300 s via
`_remove_orphans`.

---

## 11. Several replicas per pod: measured end to end, 2026-10-03

Image `dev_20261003_191750_7f806d170f8-h` (`env12` = `1d9e2753e70c`; pod
digest `sha256:ec0a643973c4…`, driver 580.159.03). Each model was dumped by an
`n_gpus=8` job whose spec set only `semi_p: true`, published from its dump pod
with `semip_publish.py /data-fast/image-cache_neutrino/<cfg12>_<env12>`, and
restored by a fresh job in a different pod once `--status` reported the
skeleton and weights verified on every node.

### What was published

| Model | TP | Replicas per pod | Skeleton `<cfg12>_<env12>_<wt12>` |
|---|---|---|---|
| Qwen3.8-27B | 1 | 8 | `8607f0cc29f1_1d9e2753e70c_64e76626e58c` |
| Qwen3.8-27B | 2 | 4 | `3d87a81c7cf4_1d9e2753e70c_1851f3103661` |
| Qwen3.8-27B | 4 | 2 | `67f11645478e_1d9e2753e70c_d180c26eff00` |
| Qwen3.6-35B-A3B | 1 | 8 | `1592fb35ffb0_1d9e2753e70c_13c5b3a26aae` |
| Qwen3.6-35B-A3B | 2 | 4 | `7c02a4a40b1c_1d9e2753e70c_76c32f3a12af` |
| Qwen3.6-35B-A3B | 4 | 2 | `e302bf221b8f_1d9e2753e70c_55236a8f83e8` |
| Qwen3.8-Flash-Next | 8 | 1 (flat) | `62fb3bf81473_1d9e2753e70c_19cfa2be3773` |
| GLM-5.3 | 8 | 1 (flat) | `503f58cfe44f_1d9e2753e70c_c65f56541db7` |

- **Every replica's weights hashed the same** in every multi-replica dump, so
  each key uploaded one `weight/<wt12>/`.
- **Cross-image weight sharing held again.** GLM-5.3's `c65f56541db7` (760 GB)
  was already published under the previous image's skeleton
  `503f58cfe44f_167e47c4f611_c65f56541db7`, so the publish uploaded only
  `image/` and `compilation/`.

### Restores

Every replica of every key materialized and restored, with no cache miss and
no error. Several things the plan had flagged as unverified are now answered:

- **TP=2 and TP=4 replicas restore onto other GPUs in another pod.** For
  example, 35B TP=4 went from `[5,6,4,7]` to `[7,6,5,4]`, and 27B TP=2
  replica0 from `[6,7]` to `[0,1]`. Several replicas moved to a disjoint GPU
  set. The restored contexts still report dump-time PCI bus IDs (a known
  risk), and nothing tripped on it.
- **Eight concurrent TP=1 dumps and eight concurrent restores in one pod**
  work, once the dump unlinks only its own `/dev/shm` files (CRIU_PLUMBING
  Complication 3, which cost round 1 four of six Qwen dump jobs).
- **One single-node dump serves `n_gpus=16`.** Job `1505acbc` (27B TP=4,
  on two nodes) logged `replica slot 0 of 2` and
  `replica slot 1 of 2` on each node. Each node then resolved and materialized
  `replica0` and `replica1` of skeleton `67f11645478e_1d9e2753e70c_d180c26eff00`,
  and restored both. GPU sets moved here too: node0's replica0 went from
  `[1,2,0,3]` to `[7,4,6,5]`.
- **Cross-pod restore costs about the same as restoring in the dump pod.**
  GLM-5.3 took 104.8 s from a published skeleton, against 100.1 s for its
  in-pod restore right after the dump.

### Timing

The copy (`image/` and `compilation/`) runs at 2.5-3.9 GB/s even with eight
replicas copying at once. **`image/` grows with TP**, because it is per-rank
process memory, so §9's "~6.7 GB" holds only at TP=1:

| Shape | Copied per replica | Copy time |
|---|---|---|
| TP=1 | 6.3-6.7 GiB | 2.4-2.8 s |
| TP=2 | 15.4-16.4 GiB | 5.0-5.5 s |
| TP=4 | 28.0-29.4 GiB | 8.3-11.8 s |
| GLM-5.3 TP=8 | 88.8 GiB | 24.4 s |
| Flash-Next TP=8 | 207.3 GiB | 56.4 s |

The restore that follows the copy (from the `materialized` log line to
`restore complete model_dir`), per replica, as min-max across replicas:

| Job | Replicas | After the copy (s) | CRIU restore (s) | CUDA restore (s) |
|---|---|---|---|---|
| 27B TP=1 | 8 | 28.0-31.8 | 3.3-3.6 | 3.4-5.6 |
| 35B TP=1 | 8 | 32.3-39.7 | 3.5-4.1 | 4.3-5.7 |
| 27B TP=2 | 4 | 66.2-69.4 | 4.8-5.1 | **39.4-40.8** |
| 35B TP=2 | 4 | 25.8-37.1 | 4.8-5.3 | 7.2-9.4 |
| 27B TP=4 | 2 | 33.2-33.8 | 5.3-5.4 | 8.8-9.2 |
| 35B TP=4 | 2 | 34.1-36.1 | 6.0-6.2 | 9.3-9.7 |
| 27B TP=4, `n_gpus=16` | 2 per node | 34.8-36.8 | 5.0-6.0 | 11.4-12.1 |
| Flash-Next TP=8 | 1 | 96.5 | 17.1 | 31.4 |
| GLM-5.3 TP=8 | 1 | 104.8 | 7.9 | 18.7 |

**27B TP=2 is the one outlier, and it is unexplained**: its CUDA restore took
about 40 s on every replica, against 7-9 s for 35B TP=2. It still restored
correctly. GLM-5.3's 104.8 s breaks down as: CRIU restore 7.9 s, CUDA restore
18.7 s, NCCL re-init 5.2 s, weight load 23.0 s, weights wake-up 5.9 s,
re-pinning 29.7 s, weight restore 3.9 s, KV-cache wake-up 0.8 s, CUDA-graph
rebind 7.0 s.

Dump side, per replica: init 223-266 s, CUDA checkpoint 2-15 s and
`criu_dump` 3-10 s for the Qwen models, 273-333 s in total. Flash-Next's init
took 430 s. GLM-5.3's init took 909 s; with a 31 s `criu_dump` it totals
1,099 s.

### Operational notes from the roll

- **Use `--status`, not node labels, to see whether a publish has landed.**
  `semip_publish.py --status --model <name> --key <skeleton> --weight-hash
  <wt12>` reports both directories' verified-node counts. A label name over
  63 characters is truncated to 55 plus `-<7 hex>`, so grepping labels for a
  skeleton's full name finds nothing. One check returned a false 0/10 that way.
- **An `n_gpus=16` job needs two whole nodes free.** Its second pod sat
  `Pending` for about ten minutes until other jobs were cancelled.
- **A mid-job `scale_up` that changes the per-node replica count is out of
  scope**: the new replica gets the lowest free slot on its node, but
  `SEMIP_NUM_REPLICAS` is not recomputed for its siblings.
