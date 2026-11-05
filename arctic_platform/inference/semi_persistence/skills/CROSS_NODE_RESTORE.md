# Cross-node restore with `SEMIP_UNPRIVILEGED=1`

Runbook for dumping a semi-persistence image on node A and restoring it on a
different node B that has only `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`.

Background for every claim here: *Complication 11* and *Complication 7* in
[`CRIU_PLUMBING.md`](CRIU_PLUMBING.md), and *Cross-node restore* in
[`semi-p_DESIGN.md`](semi-p_DESIGN.md). Node qualification is in
[`INSTALL.md`](INSTALL.md).

---

## 0. The one thing that cannot be fixed later

`SEMIP_UNPRIVILEGED=1` must be set **on the dump side too**, not just on the
restore side. It is one switch with two jobs:

- On dump it adds `--unprivileged` to `criu dump` *and* makes the vLLM child
  zero its capabilities before `import torch`, so CRIU records an empty cap set
  for every task in the tree.
- On restore it selects `_worker_criu_load_lowcap` (`criu restore -d
  --unprivileged`, no `unshare`, no private `/proc`).

CRIU's `restore_creds()` reinstates whatever caps the image recorded. An image
dumped **without** the flag records real capabilities that node B cannot grant,
and dies in `criu/pie/restorer.c` with *"Unable to restore capabilities"*. There
is no way to downgrade an existing image — it has to be re-dumped.

So set it before the dump, in the driver process, before `Instance` is used:

```python
import os
os.environ["SEMIP_UNPRIVILEGED"] = "1"
```

and set it again in the restore driver on node B.

---

## 1. Dump side (node A)

```bash
cd <checkout>/arctic_inference/semi_persistence
```

Give the instance an explicit `model_dir` so the image, the weights and the
compile cache land under one tree that can be copied as a unit:

```python
inst = Instance(vllm_config, "/data-fast/image-cache/qwen_35b")
#   <model_dir>/image        CRIU image + meta.json
#   <model_dir>/weights      only if you call save_weights()
#   <model_dir>/compilation  vLLM compile cache (mmapped .so/cubin by abs path)
```

Three rules while dumping:

- **Note which GPU indices you dumped on.** You need to restore onto *different*
  ones (see §4).
- **Do not rebuild the C extensions after the dump.** The image records
  `arctic_inference/*.so` by path and size; rebuilding changes the bytes and
  invalidates it. The repo is on shared Lustre, so a rebuild on *either* node
  breaks images captured by both.
- **Put the tree high in the PID space.** CRIU recreates every task at its
  recorded id and node B must have all of them free (§5), but a container hands
  out low ids to whatever starts first — so a tree captured around PID 1200
  competes forever with node B's own long-lived services. Advance the counter
  before the instance exists with `python3 scripts/pidcheck.py --burn-to
  200000` (namespace-global, so it need not be the same shell). Ids that high
  are unreachable in normal operation — `pid_max` is 4194304 — which retires
  the whole collision class for that image. It is also the *only* fix when the
  occupant on node B turns out to be your own login shell or IDE server, which
  you cannot kill and cannot outrun.

---

## 2. Preflight on node B (before copying anything)

```bash
# a. CRIU 4.2 with the CUDA plugin, and passwordless sudo
criu --version                        # 4.2.x
ls /usr/lib/criu/cuda_plugin.so
sudo -n true && echo "sudo OK"

# b. The kernel-feature probe in the mode the restore actually uses.
#    Plain `criu check` aborts here on a low-cap node; --unprivileged is the
#    real gate. A residual read-only `ns_last_pid` complaint is expected and
#    does NOT block restore.
sudo criu check --unprivileged

# c. Same GPU count and model as node A (the cuda-checkpoint device map is a
#    bijection over EVERY visible GPU, so the counts must match).
nvidia-smi --query-gpu=index,name --format=csv,noheader
```

The `--libdir` plugin directory is only used by `criu dump`, which mints a
private empty one per dump via `tempfile.mkdtemp`.  Restore never passes
`--libdir`, so node B needs nothing here either way.

### The cap-drop precondition (do not skip this on the restore node)

After the cap drop, uid 0 loses `CAP_DAC_OVERRIDE` and can only read what
`other` can. `criu_restore` itself imports nothing and is safe, but **anything
that runs after it may not be** — at TP>1 `reinit_nccl` reaches vLLM's
`in_the_same_node_as`, which spawns a fresh interpreter via
`multiprocessing.resource_tracker`. If that spawn cannot read its own stdlib,
it dies, `BrokenPipeError` is swallowed by vLLM's `contextlib.suppress(OSError)`,
and the restore **hangs silently** with only `No available shared memory
broadcast block found in 60 seconds` in the log.

```bash
# Every component needs o+x, files o+r. Check the BASE prefix: a venv shares
# its base interpreter's stdlib. A 750 home is the usual culprit.
namei -l "$(python3 -c 'import sys; print(sys.base_prefix)')"

# Functional version — makes the exact call that breaks. Must print OK.
sudo python3 -c '
import ctypes
from multiprocessing import shared_memory, resource_tracker
libc = ctypes.CDLL("libc.so.6", use_errno=True)
for c in range(64): libc.prctl(24, c, 0, 0, 0)
libc.prctl(47, 4, 0, 0, 0)
class H(ctypes.Structure): _fields_=[("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class D(ctypes.Structure): _fields_=[("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
libc.capset(ctypes.byref(H(0x20080522,0)), ctypes.byref((D*2)()))
s = shared_memory.SharedMemory(create=True, size=128); print("OK", s.name)
s.close(); s.unlink()'
```

`BrokenPipeError` instead of `OK` means fix the permissions (`chmod o+x` the
home, or serve with a world-readable prefix). No re-dump is needed — nothing
about the image is wrong.

### `arctic_inference` must be importable by whoever runs the tree

> **Correction (2026-09-08).** This section used to say "the whole tree runs as
> root". It does not have to, and under DSS it does not: the `Instance` worker
> is a plain `mp.spawn` child of the caller, so it inherits *your* uid. Under
> `SEMIP_UNPRIVILEGED=1` nothing is elevated at all: `criu` runs at the
> worker's own uid for both dump and restore (it must, to match its target --
> CRIU_PLUMBING Complication 11), and the `cuda-checkpoint` CLI is gone
> entirely, since the driver API works unprivileged (Complication 6).
>
> Prefer to run as the account that will restore. An image records the dumping
> child's uid, and `SEMIP_UNPRIVILEGED=1` leaves it with an empty capability
> set, so a root-dumped image restores as uid 0 *without* `CAP_DAC_OVERRIDE` —
> unable to write files an unprivileged parent created. That breaks
> `rebind_graphs` on `compilation/` (and, before 2026-10-03, `rebind_log` on
> `/tmp/inst<N>.log`, which no longer exists: logs go to the pod log).

If you do run the tree as root, a `pip install --user` is invisible to it.
Verify on node B:

```bash
sudo python3 -c "from arctic_platform.inference.semi_persistence import Instance; print('OK')"
```

If it fails, register the *same* checkout without rebuilding anything — copy the
three editable-install artifacts from node A rather than running `pip install -e .`,
which would recompile the `.so` files in the shared repo and invalidate every
existing image:

```bash
D=/usr/local/lib/python3.12/dist-packages
sudo cp -a $D/__editable__.arctic_inference-0.3.1.dev0.pth \
           $D/__editable___arctic_inference_0_3_1_dev0_finder.py \
           $D/arctic_inference-0.3.1.dev0.dist-info  <node-B>:$D/
```

The finder maps to absolute paths under `<checkout>`,
which is shared Lustre, so node B resolves the identical files.

### …and check that nothing shadows the checkout

If node B has a *real* `dist-packages/arctic_inference/` directory — a plain
`pip install` rather than `-e`, or a hand-copied tree — that wins over the
editable finder, because `PathFinder` is consulted before the finder the `.pth`
appends to `sys.meta_path`. Every edit you make to the checkout is then inert
at runtime, with nothing to show for it but the file paths in tracebacks:

```bash
sudo python3 -c "import arctic_platform.inference.semi_persistence as sp; print(sp.__file__)"
```

If that prints a path under `dist-packages` that is not a symlink, point it at
the checkout instead of syncing files by hand:

```bash
sudo mv $D/arctic_inference/semi_persistence $D/arctic_inference/semi_persistence.bak
sudo ln -sfn <checkout>/arctic_inference/semi_persistence \
             $D/arctic_inference/semi_persistence
```

A symlink is safe where `pip install -e .` is not: it rebuilds nothing, so it
cannot change a recorded `.so` and cannot invalidate an existing image. Note
`scripts/` and `skills/` are not part of an installed copy, so `imgdiff.py` and
`pidcheck.py` are always run from the checkout regardless.

---

## 3. Copy the image tree to the identical absolute path

`/data-fast` is an `emptyDir` on node-local XFS — the image cache travels with
neither the pod nor the node, does **not** ride along with the Lustre repo, and
must be copied. The path must match byte-for-byte, because the
image bakes the compile-cache mappings by absolute path *and* `criu_restore`
rejects a `model_dir` that differs from the one in `meta.json`.

```bash
sudo rsync -aH --info=progress2 \
    /data-fast/image-cache/qwen_35b/ \
    <node-B>:/data-fast/image-cache/qwen_35b/
```

Use `rsync -a` or a plain tarball. **Never `tar -h`** — dereferencing symlinks
changes file sizes and CRIU size-checks every mapping.

### Then verify the environment before spending a restore attempt

CRIU records every file-backed mapping by absolute path and size and
re-validates the size at restore. One mapped file of a different length aborts
the whole restore from inside CRIU, with no up-front check, and it stops at the
first bad mapping instead of reporting them all. "Same requirements installed"
is *not* sufficient — identical version specs routinely yield different bytes.

```bash
# On node B:
cd <checkout>/arctic_inference/semi_persistence
sudo python3 scripts/imgdiff.py /data-fast/image-cache/qwen_35b/image
```

Want `TOTAL PROBLEMS: 0`. It also compares the ELF build-IDs CRIU recorded, so
it catches a same-size-different-build library that would pass CRIU's own check
and then map the wrong text pages. The one entry it always lists and that is
never a problem is the `/dev/shm/sem.*` **GHOST** file — unlinked at dump time,
carried inside the image, recreated by CRIU.

If it reports `SIZE MISMATCH` / `ABSENT` / `BUILD-ID MISMATCH`, it prints the
trees needing sync. Copy those trees from node A to the identical path (an HF
cache under `/root/.cache/huggingface` is a common one if vLLM still had
safetensors mapped at dump time) and re-run until clean.

### Then check the recorded task ids are free — immediately before restoring

```bash
sudo python3 scripts/pidcheck.py /data-fast/image-cache/qwen_35b/image
```

Without a private PID namespace this mode needs *every* recorded task id free,
threads included (§5). Unlike `imgdiff.py` this is a snapshot, not a property
of the image: the answer changes as processes come and go, so run it right
before the restore. `criu_restore` preflights the same check itself and names
the occupant, so this only buys you the answer before spending an attempt.

`--burn` advances the PID counter past the image's highest id so *subsequently
started* processes land clear of it. It frees nothing. Two consequences worth
internalising:

- A process already running keeps its ids. If `pidcheck.py` lists a live
  occupant, burning changes nothing for it — it has to exit.
- If the occupant is an ancestor of your launcher (your shell, `sudo`, the IDE
  server that spawned the terminal), you cannot outrun it either: a burn moves
  only descendants you start afterwards. `criu_restore`'s report flags this case
  as `an ancestor of this restore`. Re-dump per §1 instead.

---

## 4. Restore script for node B

```python
import json, os

os.environ["SEMIP_UNPRIVILEGED"] = "1"          # before Instance is used

from arctic_platform.inference.semi_persistence import Instance

MODEL_DIR = "/data-fast/image-cache/qwen_35b"

# criu_restore compares vllm_config with a full dict !=, so reading it back out
# of the image is the only way to be sure the check passes.
meta = json.load(open(os.path.join(MODEL_DIR, "image", "meta.json")))
print("dumped on GPUs", meta["gpus"], "of", len(meta["gpu_uuids"]), "visible")

inst = Instance(meta["vllm_config"], MODEL_DIR)

inst.criu_restore()                  # -> state 'checkpointed'
inst.attach()                        # only if the image was dumped with
inst.load_weights()                  #   save_weights()/detach(); else drop both

inst.cuda_restore(gpus=[4, 5])       # MUST differ from meta["gpus"] -- see below
inst.reinit_nccl()                   # TP>1 only, immediately after cuda_restore
inst.wake_up_weights()
inst.repin()
inst.restore_weights()
inst.wake_up_kv_cache()
inst.rebind_graphs()                 # TP>1 only, after wake_up_kv_cache
inst.generate(["Hello, world!"], {"temperature": 0.0, "max_tokens": 64})
inst.wait().print_status()
inst.teardown()
```

Run it as root, from this directory (modules import their siblings by bare
name):

```bash
cd <checkout>/arctic_inference/semi_persistence
sudo python3 restore_here.py       # the script sets SEMIP_UNPRIVILEGED itself
```

At TP=1 use `inst.cuda_restore(gpu=3)` and drop `reinit_nccl` /
`rebind_graphs` (they are no-ops anyway).

### Restore onto the dump's own GPU indices, on another node

This was the cross-node-specific trap, and it is now handled;
[`IMAGE_CACHE.md`](IMAGE_CACHE.md) §9 is what forced the fix.

`_worker_restore` used to build the `oldUuid=newUuid` device map **only** when
the indices changed:

```python
migrate = bool(old_gpus and new_gpus and list(old_gpus) != list(new_gpus))
```

That was sufficient for exactly as long as an image could not travel: dump and
restore happened in one pod, so "same indices" implied "same node". Restore an
image onto the same indices on a *different* node and no map was built at all,
leaving the checkpoint's baked node-A UUIDs — which do not exist on node B — and
`cuCheckpointProcessRestore` failed with `CUDA_ERROR_INVALID_VALUE (CUresult=1)`.
With a distributed cache that is a one-in-eight placement, not a corner.

`_placement_changed` now asks whether the placement moved by index **or** by
node, comparing the recorded UUIDs against the local ones when the indices
match. NVML is read only in that one case, so every other path is unchanged.

Two things follow. `meta.json` must carry `gpu_uuids`: an image without them
restoring onto its own indices gets no map and can only work on its capture
node, which `_check_gpu_placement` warns about rather than refusing. And the
visible GPU **count** must match, since the map is a bijection over every GPU;
that is now a named error instead of an `IndexError` while pairing.

The permutation pins `old_gpus[k] -> new_gpus[k]` and pairs the remaining GPUs
in order, so it works even when the index sets overlap (`[4,5,6,7] -> [0,1,4,5]`),
and is the identity when only the node changed.

---

## 4a. A restored context reports its dump-time PCI bus id — why TP>1 needs the same slots

**A CRIU-restored CUDA context keeps answering with the PCI address of the GPU it
was dumped on.** `cuda_restore` migrates compute correctly -- kernels run,
`cudaHostAlloc` succeeds, `cudaGetDeviceCount` is right, the rank-to-device
permutation is applied -- but `cudaDeviceGetPCIBusId` still returns the dumped-on
address. Measured across six mismatched restores on 2026-09-20 with no exception,
at TP=1 and TP=2, on this hardware where the slot-to-PCI map is fixed:

| slots | PCI |
|---|---|
| `0,1` | `59:00.0` `5A:00.0` |
| `2,3` | `72:00.0` `73:00.0` |
| `4,5` | `8B:00.0` `8C:00.0` |
| `6,7` | `A4:00.0` `A5:00.0` |

A TP=2 image dumped on `6,7` and restored in a pod holding `0,1` has ranks that
report `A4`/`A5`. **This is the mechanism behind the rule that a TP>1 image only
restores onto the slots it was dumped on**, and it explains the observations that
previously looked unrelated: TP=1 is immune because nothing reads the address;
TP=8 always works because an 8-GPU pod contains every address; removing the
network entirely (`NCCL_NET_PLUGIN=none`, `NCCL_IB_DISABLE=1`) does not help
because identity is resolved before transport; and `scripts/test_tp2.py` migrates
`[4,6]`→`[5,7]` and passes only because a full node has all the addresses.

At TP>1 NCCL resolves device identity while building the communicator, finds
addresses belonging to no GPU in this pod, and `ncclCommInitRank` returns
`ncclSystemError` -- surfaced as `NCCL error: unhandled system error`, in ~73 ms,
with no `NCCL WARN` because that path does not log one.

**Tested and ruled out as workarounds.** `NCCL_P2P_DISABLE=1` plus
`NCCL_SHM_DISABLE=1`, set at dump time, does **not** rescue a mismatched restore:
the stale identity is in the CUDA context, upstream of NCCL's transport choice, so
disabling the consumers cannot fix the producer. Over-allocating GPUs so every
dumped-on address exists is still a live idea but is not expressible today --
replica count is `n_gpus / tensor_parallel_size`, so `n_gpus: 8` with
`tensor_parallel_size: 2` builds four co-located TP=2 replicas that abort on a
task-id collision rather than one pair in an 8-GPU pod.

The real fix belongs to `cuCheckpointProcessRestore` / `cuda-checkpoint`: a
restore has to update reported device identity, not just the device mapping.
Until then, a TP>1 image is only restorable onto its own slots, and the engine
treats a mismatch as a cache miss and cold-starts (see `_missing_device_nodes`).

The engine now also **keys on the slots** rather than only checking them: at
TP>1 `_device_binding` folds the pod's `/dev` allocation into `cfg12`, so a
mismatch is a plain lookup miss and the same config dumped on two slots keeps
two images instead of the second overwriting the first. That does not make an
image portable — the constraint above is unchanged — it makes the set of slots
a config is restorable on something that grows with each dump, rather than one
lucky placement. See `IMAGE_CACHE.md` §6.

The diagnostic is `_cuda_restore_probe`, which runs in every rank immediately
before `reinit_nccl` and again on failure:

```
cuda probe: {'rank': 0, 'get_device': '0 (no error)', 'device': 1,
             'device_count': 2, 'pci_bus_id': '0000:A5:00.0',
             'cudaHostAlloc': '0 (no error)', 'cudaFreeHost': '0 (no error)',
             'nvml': {'init': '0 (Success)', 'device_count': 2,
                      'p2p_matrix': 'OK (2 pairs)',
                      'local_bus_ids': ['0000:8B:00.0', '0000:8C:00.0'],
                      'stale_bus_id': '0000:A5:00.0',
                      'lookup_stale': '6 (Not Found)',
                      'lookup_local': 'OK -> index 0'}}
```

Compare `pci_bus_id` against `nvidia-smi --query-gpu=index,pci.bus_id` in the
pod. They disagree on any mismatched draw, and that disagreement is the bug.

The `nvml` block narrows *which* line of `commAlloc` dies. NCCL reaches
`ncclNvmlDeviceGetHandleByPciBusId`, which first calls
`ncclNvmlEnsureInitialized` — and that returns a cached `initResult` with no new
warning if NVML init failed earlier, which is why the expected
`nvmlDeviceGetHandleByPciBusId() failed: Not Found` WARN is missing from the
logs. Two candidates, read them in this order:

- `init` or `p2p_matrix` not OK → the **cached-init** path. NCCL's init walks a
  P2P-status matrix over every visible device; one bad pair poisons every later
  NVML call. The bus-ID lookup is then a red herring.
- `init` and `p2p_matrix` OK, `lookup_stale` `Not Found`, `lookup_local` OK →
  the **bus-ID lookup**, confirmed. NVML in this pod cannot resolve the dump
  node's address, `ncclSystemError` follows, and that is the whole failure.
- `lookup_local` also failing → NVML is broken generally; stop reading this as a
  slot problem.

`lookup_*` tries every spelling of a bus id (both domain widths, both cases)
before reporting a miss, so a `Not Found` here is a real absence rather than
CUDA's 8-digit domain failing to match /proc's 4-digit one.

---

## 5. Constraints that stay true in this mode

| Constraint | Consequence |
|---|---|
| **Every recorded task id must be free** | No private PID namespace, so CRIU recreates each task at its recorded id — leaders *and* their threads, which come from one number space (a TP2 image needs ~900 ids across 3 leaders). Any live process whose threads overlap the range blocks the restore. The constraint is collision, not restore count: three images restored and served concurrently on one node once their dumps were run concurrently, which interleaves the allocations into disjoint id sets. Dump serially and they collide by construction. `criu_restore` preflights it and names the occupants; `scripts/pidcheck.py` answers it before you spend an attempt. |
| **`vllm_config` must match exactly** | Full dict `!=`, no extra keys. `tensor_parallel_size` participates, so a TP=2 image cannot be mistaken for a TP=1 one. Load it from `meta.json`. |
| **`model_dir` must match** | Rejected by name up front. |
| **Same visible GPU count** | The device map is a bijection over every visible GPU. |
| **The restored child runs the code frozen in the image** | Editing `vllm_child.py` changes nothing until an offline re-dump. Keep dump-time and runtime code the same checkout. |
| **`criu_restore` does not re-apply `_env`** | The child's environment is baked into the image and restored verbatim. |
| **A dump is destructive** | After `criu_dump` the model is `saved` with no live process. |
| **Don't restore within 60s of a dump on the same node** | Recorded local ports sit in `TIME_WAIT`. Current dumps set `SO_LINGER(1,0)` so the kill RSTs instead; older images need the 60s to drain. Cross-node this does not apply. |

---

## 6. Symptom → cause

| What you see | Cause |
|---|---|
| `Unable to restore capabilities` (`criu/pie/restorer.c`) | Image dumped **without** `SEMIP_UNPRIVILEGED=1`. Re-dump. |
| `Could not initialize kernel features detection` | `--unprivileged` not in play: `SEMIP_UNPRIVILEGED` unset in the restore driver. |
| `File <path> has bad size <local> (expect <image>)` + `Can't open vma` | Environment mismatch. Run `imgdiff.py`, sync the reported trees. |
| `File <path> has bad mode 0100644 (expect 0100755)` + `Can't open vma` | The image reached this node through S3, which carries no POSIX mode, so the publish → sync round trip flattened every file to the syncing daemon's umask. `_apply_recorded_modes` repairs this during a materialize — but only from a recorded mode, so check whether the `env_files` rows in `image/meta.json` have a fourth element. If they do not, the image predates mode recording and must be re-dumped; a hand-patch of the mirror is possible through the `model-sync` container (which mounts the hostPath `rw`, unlike the job pod's `readOnly` mount) but does not survive a re-publish. See `IMAGE_CACHE.md` §9. |
| `CUDA_ERROR_INVALID_VALUE (CUresult=1)` from `cuda_restore` | No device map built. Since `_placement_changed`, the remaining cause is an image whose `meta.json` predates `gpu_uuids` restored onto its own indices off its capture node (§4). |
| `Can't bind inet socket … Address already in use` | Recorded port in `TIME_WAIT`; wait 60s. |
| `criu restore aborted (recorded task-id collision: …)` | The preflight found recorded ids occupied; the message names every occupying process, and flags one that is `an ancestor of this restore`. Check `image/meta.json` for `pid_floor` first: `null` or absent means an image dumped before the dump-side floor existed, whose low ids overlap a fresh pod's launcher by construction — it must be re-dumped, and nothing on the restore side can rescue it (the holder is a thread of the restoring actor and cannot exit). See `IMAGE_CACHE.md` §8. With a floor recorded, this is a genuine collision: free the holders, or `scripts/pidcheck.py <image> --burn` and relaunch. |
| CRIU `Can't fork for <pid>: File exists`, or `pie: Unable to create a thread: -17` | The same collision, claimed in the window after the preflight ran (criu's own helpers spawn into the same number space). `-17` is `EEXIST` on a *thread* id, which is why nothing shows at that number in `ps` — threads live only under `/proc/<pid>/task`. The error appends the resolved occupants. |
| `reinit_nccl` fails in well under a second with `NCCL error: unhandled system error` | Slot mismatch at TP>1. The restored ranks report their dump-time PCI bus ids, which name no GPU in this pod — see §4a. Confirm with the `cuda probe` lines against `nvidia-smi`. Not a network fault despite how it reads, and not fixable from the restore side: this image needs a pod holding the slots it was dumped on. |
| `cuda_restore FAILED … never quiesced within 30.0s … are blocked in the kernel` | Ranks that missed the window **and** are genuinely blocked (`wchan != 0`, or inside a syscall). Since 2026-09-22 a rank merely spinning in userspace is accepted instead, and logged as `N pid(s) accepted as userspace spins` — that is the line to grep for to tell "the fix worked" from "the bug did not happen". Do **not** lower `SEMIP_SETTLE_TIMEOUT_S` — 4.0 s and 4.5 s budgets failed restores that would otherwise have served. |
| `N pid(s) accepted as userspace spins …` (informational) | The clock bug, survived. A restore cannot preserve `CLOCK_MONOTONIC` here — that needs a time namespace, which needs `CAP_SYS_ADMIN`, and the pod runs `CapEff: 0000000000000000` under `SEMIP_UNPRIVILEGED=1`. vLLM's `SpinCondition.wait` compares `time.monotonic()` against `self.last_read + busy_loop_s`, and after a restore `last_read` still holds a reading from the *dump* host, so if the restore host booted later than the dump host by more than the dump-to-restore gap the comparison never turns over and the `sched_yield()` loop never ends. Predicts the failure exactly: `boot(restore) >= boot(dump) + (Tr − Td) − busy_loop_s`, scored 29/29 on 2026-09-22 across rounds 5 and 6. Nothing to do with slots or TP, which is why matched placements hit it too, and why a longer budget never helped. To check a specific pair, read `/proc/uptime` in a pod on each node: boot instant is `now − uptime`. Retrying is pointless — the retry lands on the same host with the same clock, which is why `022eb9d`'s retry rescued 0 of 4 and was removed. |
| `rebind_graphs` hangs ~302 s on **matched** slots, ending `TimeoutError: RPC call to sample_tokens timed out`, last log line `verify nreq=16 toklen=4 start` | **The image was dumped cold.** `install_keepgraph_patch` forces `keep_graph=True` so the rebind can read graph topology, and torch's `capture_end` instantiates only when `keep_graph_` is false — so the graphs are never instantiated at capture, and the post-restore pass is the first code to build them. It wedges at the one rung that also JIT-compiles. Not CRIU damage: `raw_cuda_graph_exec()` raises on a host-side `TORCH_CHECK` of a bool in the process image, which `cuCheckpointProcessCheckpoint` cannot reach. **This should now be unreachable:** warming is part of `init` and unconditional since 2026-09-24, and the dump logs `COLD IMAGE` at error level if any rank reports `uninstantiated > 0`. If you see this hang, the image was published past that error, or something changed capture. Wave 10 measured warm 17/17 clean against cold 1/9, Fisher exact p = 5.8e-06. See `tp_DESIGN.md` §5. (Logged under `recapture_graphs` / `warmup nreq=…` before the rename.) |
| `census[restore:pre-ladder] … exec_ok=42 uninstantiated=2100` (or any non-zero `uninstantiated`) | The image was dumped cold and this restore is about to build 2100 execs on four ranks at once. The count is not damage — it is how many shapes the dump-time job happened to run, times the wrapper count. Re-dump warm rather than debugging the restore. |
| A `reuse` restore succeeds but `warmup nreq=16` costs ~4.2 s | Same cause, survived. That rung is doing first-run instantiate plus two JIT compiles (`_zero_kv_blocks_kernel`, CuTeDSL `_FullyFusedDeltaRuleSm90`); warm it at dump time and the rung drops to ~35 ms. A passing cold restore and a hanging one differ only in whether the race went your way. |
| A restore reports success but its phase list starts with `init OK (~217s)` | It did not restore — it cold-started. Usually the published image was not usable yet: the mirror needs its `.neutrino_verified` marker, which can land ~12 minutes after the publish reports success, not the 300 s the message claims. Confirm the marker before believing any restore result. |
| Silent hang in `reinit_nccl`; log repeats `No available shared memory broadcast block found in 60 seconds` | Cap-drop precondition violated on the restore node. `py-spy dump` shows ranks at *different* lines of `in_the_same_node_as`. Fix permissions; no re-dump. |
| `ModuleNotFoundError` for a stdlib module right after `[semip] dropped capabilities` | Same precondition, hit at cold start instead. |
| `ModuleNotFoundError: No module named 'arctic_inference'` under `sudo` | Installed `--user`; root cannot see `~/.local`. |
