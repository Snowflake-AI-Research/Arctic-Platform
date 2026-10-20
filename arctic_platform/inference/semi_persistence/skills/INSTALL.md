# Installation

Setup steps for running `semi_persistence` on a fresh Ubuntu 24.04 host:
install CRIU (with the CUDA plugin path), then download the speculative
draft models from S3 to `/data-fast/`.

For background on *why* the CRIU plumbing looks the way it does, see
[`CRIU_PLUMBING.md`](./CRIU_PLUMBING.md).

---

## 1. CRIU (Ubuntu 24.04)

CRIU is not in the default Ubuntu repos at a recent-enough version.
Install from the official CRIU PPA:

```bash
# 0. Refresh the package lists first.  On a container whose lists are
#    stale, step 1 fails with 404s on every package because the mirrors
#    have already rotated out the versions the old lists point at.
sudo apt-get update

# 1. Add the CRIU PPA
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:criu/ppa
sudo apt-get update

# 2. Install CRIU (brings in crit, protobuf, etc.)
sudo apt-get install -y criu

# 3. Verify
criu --version                     # should print 4.x (4.2.1 from the PPA)
which crit                         # /usr/bin/crit  (CRIU image tool)
ls /usr/lib/criu/cuda_plugin.so    # shipped by the PPA package
sudo criu check                    # "Looks good."
```

On a **non-privileged** node (only `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`,
no `CAP_SYS_ADMIN`), plain `criu check` aborts at kernel-feature detection
because it tries to create a throwaway network namespace.  Check such a node
with the flag that the `SEMIP_UNPRIVILEGED=1` paths use:

```bash
sudo criu check --unprivileged      # must get past kerndat
```

A residual complaint about a read-only `ns_last_pid` is expected from the
checker and does **not** block restore: `clone3(set_tid)` at the recorded PIDs
is authorized by `CAP_CHECKPOINT_RESTORE`.  See Complication 11 in
[`CRIU_PLUMBING.md`](CRIU_PLUMBING.md).

### Check which capabilities the node actually grants

Several of them decide whether a dump and a restore are possible at all, and
none is implied by `criu check --unprivileged` passing.  Read the **bounding**
set rather than the effective one: `CapBnd` is the ceiling on what a `setcap`
can ever raise, so a capability missing there cannot be granted by any means
short of changing the pod spec.

```bash
grep -E 'CapEff|CapBnd' /proc/self/status   # CapBnd == what can ever be acquired
capsh --decode=$(grep CapBnd /proc/self/status | awk '{print $2}')
```

`cap_checkpoint_restore` is the floor.  Beyond it, what you need depends on
whether the dumped tree will run as root:

- **Dumping under `sudo`** (tree is `uid 0`): nothing further.  Both the ptrace
  seize and the rlimit read are same-uid operations.
- **Dumping as an unprivileged user**: `cap_sys_ptrace` *and* `cap_sys_resource`
  become cross-uid operations.  `cap_sys_resource` is typically absent from
  `CapBnd` on cap-prod pods, so it cannot be acquired at all — the way through
  is to run `criu` at the target's own uid instead, which is what
  `_worker_criu_save` and `_worker_criu_load_lowcap` now do (neither shells out
  through `sudo`).  Both then need their capabilities from the binary:

  ```bash
  sudo setcap cap_sys_ptrace,cap_checkpoint_restore,cap_setpcap,cap_setgid+eip \
      /usr/sbin/criu
  getcap /usr/sbin/criu
  ```

  Keep the `+e`, or a real-uid-0 `execve` lands with an empty effective set and
  a root-run `criu` breaks.  File capabilities are node-local and do not
  survive the pod.  Full reasoning in Complication 11.

  **Grant only what the pod's bounding set contains.**  The four above are
  right for a cap-prod pod; on a `drop: ["ALL"]` pod that adds exactly
  `CHECKPOINT_RESTORE` and `SYS_PTRACE`, use only those two:

  ```bash
  sudo setcap cap_sys_ptrace,cap_checkpoint_restore+eip /usr/sbin/criu
  ```

  This is not a matter of taste, and getting it wrong does not fail soft.  A
  capability outside `CapBnd` is **not** silently dropped at `execve`:
  `bprm_caps_from_vfs_caps()` computes `pP' = (pB & fP) | (pI & fI)` and
  returns `EPERM` if any file-permitted bit did not survive, whenever the
  file's effective bit is set — which the `+e` above sets.  So a
  four-capability `criu` on a two-capability pod fails to `execve` at all.
  Check the bounding set first:

  ```bash
  capsh --decode=$(grep CapBnd /proc/self/status | awk '{print $2}')
  ```

  **The last two are for the restore, not the dump.**  A dump needs only the
  first two, so it is easy to stop there and then lose an expensive restore to
  a hang.  `restore_creds()` runs privileged syscalls that a root restore
  satisfied for free:

  | syscall | capability | when it fires |
  |---|---|---|
  | `PR_CAPBSET_DROP` (`restorer.c:317`) | `cap_setpcap` | criu drops every capability *absent* from the image's recorded `cap_bnd`, without checking whether it is already absent, and the kernel tests `CAP_SETPCAP` first.  So it fires for every gap — ~25 on a cap-prod pod, ~39 on a two-capability one — **unless the image records `no_new_privs`**, which demotes the `EPERM` to a warning.  That flag is how the two-capability pod works at all; see Complication 14. |
  | `setgroups` | `cap_setgid` | Only when the group lists differ.  criu 4.2 compares `getgroups()` against the recorded list and skips the call when they match, so a dump and restore in the same pod spec never needs it. |

  On a cap-prod pod both are inside the bounding set, so `setcap` can grant
  them — unlike `cap_sys_resource`.  On a two-capability pod neither is, and
  neither is needed: `no_new_privs` covers the first and the group lists match
  for the second.

  Getting this wrong does not produce a clean failure.  criu logs
  `Unable to drop capability 2: -1` once per task, then `BUG at
  criu/pie/restorer.c:820`, and **deadlocks** — the log stops mid-stream with
  no `Restoring FAILED`, criu eventually exits, and the restored tasks are left
  stranded at their recorded PIDs, which then have to be killed by hand before
  the next attempt.

### If the gateway will run as root

The Ray dashboard agent needs three packages the image omits, and a
`pip install --user` as an unprivileged account is invisible to root, so the
worker raylet aborts in `WaitForDashboardAgentPorts` and no zone becomes
healthy.  The default index is an internal mirror that fails SSL verification
and does not carry them, and PEP 668 marks this environment externally
managed:

```bash
sudo python3 -m pip install --break-system-packages --index-url https://pypi.org/simple \
    aiohttp-cors opencensus opentelemetry-exporter-prometheus
sudo python3 -c "import ray.dashboard.http_server_agent, \
    ray.dashboard.modules.reporter.reporter_agent; print('ok')"
```

Install them for the unprivileged account too if anything will run the gateway
that way; the two installs are independent.

### Also qualify the interpreter for `SEMIP_UNPRIVILEGED=1`

`criu check` says nothing about the *other* precondition of unprivileged
mode: the child zeroes its capabilities, losing `CAP_DAC_OVERRIDE`, so
from then on uid 0 can only read what `other` can.  Every directory on
the path to the interpreter needs `o+x` and its files `o+r`.  Check the
**base** prefix, since a venv shares its base interpreter's stdlib:

```bash
namei -l "$(python -c 'import sys; print(sys.base_prefix)')"
```

Any component without `o+x` (a `750` home is the usual culprit) fails the
precondition.  A system-wide Python (`/usr/lib/python3.12`,
`/usr/local/lib/python3.12/dist-packages`) satisfies it for free, which is
why pre-baked pod images never hit this.

The functional version of the same check — worth running once per node,
because it exercises the exact call that breaks — drops capabilities and
then does what vLLM's `in_the_same_node_as` does during `reinit_nccl`:

```bash
sudo python3 -c '
import ctypes
from multiprocessing import shared_memory, resource_tracker  # pre-import
libc = ctypes.CDLL("libc.so.6", use_errno=True)
for c in range(64): libc.prctl(24, c, 0, 0, 0)   # PR_CAPBSET_DROP
libc.prctl(47, 4, 0, 0, 0)                       # PR_CAP_AMBIENT_CLEAR_ALL
class H(ctypes.Structure): _fields_=[("version",ctypes.c_uint32),("pid",ctypes.c_int)]
class D(ctypes.Structure): _fields_=[("effective",ctypes.c_uint32),("permitted",ctypes.c_uint32),("inheritable",ctypes.c_uint32)]
libc.capset(ctypes.byref(H(0x20080522,0)), ctypes.byref((D*2)()))
s = shared_memory.SharedMemory(create=True, size=128); print("OK", s.name)
s.close(); s.unlink()'
```

Run it with the interpreter you will actually serve with.  `OK` means the
node is fit; a `BrokenPipeError` means the precondition is violated and a
TP>1 restore will **hang silently** in `reinit_nccl` (Complication 11).
Fix it before dumping: images are otherwise fine, but you will not find
out until a later restore.

Nothing further is needed for the `--libdir` plugin directory the dump
passes.  `_worker_criu_save` mints a private empty one per dump with
`tempfile.mkdtemp` and removes it afterwards, so there is no directory to
pre-create and no root step.  See *Complication 7* in
[`CRIU_PLUMBING.md`](./CRIU_PLUMBING.md) for why it is passed at all.

### Alternative: build from source (when apt mirrors are unreachable)

On hosts where `archive.ubuntu.com` and/or `ppa.launchpadcontent.net`
are blocked or flaky, the PPA path above will fail with connection
timeouts or `404`s.  GitHub is usually still reachable, so the
fallback is to build CRIU 4.2 and its dependencies from source.

Required tooling already on the box: `gcc`, `make`, `autoconf`,
`automake`, `libtool`, `pkg-config`, `protoc` (libprotoc 3.21.x),
`git`, `python3`.

```bash
mkdir -p /tmp/criu-build && cd /tmp/criu-build

# A. Fix-ups for headers/symlinks the system protobuf/libuuid packages
#    omit (only run the ones whose targets are actually missing).
sudo ln -sf /usr/lib/x86_64-linux-gnu/libprotoc.so.32 \
            /usr/lib/x86_64-linux-gnu/libprotoc.so
sudo ln -sf /usr/lib/x86_64-linux-gnu/libuuid.so.1 \
            /usr/lib/x86_64-linux-gnu/libuuid.so

# protobuf compiler headers (needed by protobuf-c) — match installed protoc:
git clone --depth 1 --branch v3.21.12 \
    https://github.com/protocolbuffers/protobuf.git protobuf-src
sudo cp -r protobuf-src/src/google/protobuf/compiler \
           /usr/include/google/protobuf/compiler

# uuid.h header (needed by CRIU)
git clone --depth 1 --branch v2.40.4 \
    https://github.com/util-linux/util-linux.git
sudo mkdir -p /usr/include/uuid
sudo cp util-linux/libuuid/src/uuid.h /usr/include/uuid/uuid.h

# B. Dependencies
git clone --depth 1 --branch libcap-2.73 \
    https://git.kernel.org/pub/scm/libs/libcap/libcap.git
( cd libcap && make -j$(nproc) && sudo make install prefix=/usr )

git clone --depth 1 --branch v1.5.0 \
    https://github.com/protobuf-c/protobuf-c.git
( cd protobuf-c && ./autogen.sh && ./configure --prefix=/usr \
    && make -j$(nproc) && sudo make install )

git clone --depth 1 --branch v1.3 https://github.com/libnet/libnet.git
( cd libnet && ./autogen.sh && ./configure --prefix=/usr \
    && make -j$(nproc) && sudo make install )

# C. CRIU 4.2 itself (also builds cuda_plugin.so)
git clone --depth 1 --branch v4.2 \
    https://github.com/checkpoint-restore/criu.git
cd criu
PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH" \
    make -j$(nproc)
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-criu PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-lib  PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-crit PREFIX=/usr
```

Verify:

```bash
criu --version          # Version: 4.2,  GitID: v4.2
which crit              # /usr/local/bin/crit  (note: not /usr/bin/crit)
```

Notes:

- `crit` lands in `/usr/local/bin/` on the source build (vs `/usr/bin/`
  from the PPA), because it ships as a Python wheel installed by
  `install-crit`.
- The from-source path also produces `cuda_plugin.so` in the CRIU build
  tree — the CRIU CUDA infrastructure picks it up at dump time.  The PPA
  package already ships it as `/usr/lib/criu/cuda_plugin.so`, so the CUDA
  plugin is not on its own a reason to prefer the source build.
- More detailed build notes (and conditional fix-ups for systems missing
  even more headers) live in `instance_DESIGN.md` under
  *"CRIU Installation (v4.2, from source)"*.

---

## 2. Speculative draft models

`register.py` and `register_FCA.py` reference four speculative draft
models, all expected under `/data-fast/`:

- `/data-fast/spec-decode-qwen3-8b-search_r1`
- `/data-fast/spec-decode-qwen3-30b-search_r1`
- `/data-fast/qwen3-32b-bird-4096-3head`
- `/data-fast/qwen3-32b-longcontext-4096-3head`

Sync them from wherever your speculator checkpoints live (`$SPECULATOR_S3`
below stands for that S3 prefix; you need read access to it):

```bash
aws s3 sync \
    $SPECULATOR_S3/spec-decode-qwen3-8b-search_r1 \
    /data-fast/spec-decode-qwen3-8b-search_r1

aws s3 sync \
    $SPECULATOR_S3/spec-decode-qwen3-30b-search_r1 \
    /data-fast/spec-decode-qwen3-30b-search_r1

aws s3 sync \
    $SPECULATOR_S3/qwen3-32b-bird-4096-3head \
    /data-fast/qwen3-32b-bird-4096-3head

aws s3 sync \
    $SPECULATOR_S3/qwen3-32b-longcontext-4096-3head \
    /data-fast/qwen3-32b-longcontext-4096-3head
```

`aws s3 sync` is idempotent — re-running it after a partial download
only fetches missing/changed objects, so it's safe to retry on flaky
connections.

Verify the destinations are non-empty:

```bash
ls /data-fast/spec-decode-qwen3-8b-search_r1
ls /data-fast/spec-decode-qwen3-30b-search_r1
ls /data-fast/qwen3-32b-bird-4096-3head
ls /data-fast/qwen3-32b-longcontext-4096-3head
```
