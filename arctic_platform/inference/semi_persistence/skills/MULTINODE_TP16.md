# Multi-node semi-p (TP=16): one engine across two pods

State as of 2026-10-06. Validated on hardware for the library; the engine layer
has not yet served a job (see §9).

`nnodes > 1` is a TP group whose ranks live on more than one pod. One leader
owns ranks `0..local-1` and serves; a `SemipNodeAgent` owns each further block
and never serves. Every collective is the leader's, because its
`MultiprocExecutor` spans both halves once NCCL is up.

## 0. The one thing to know first

**The gate is `nnodes`, never TP.** What changes behaviour is whether one TP
group crosses a node boundary, not how many GPUs a job uses:

| Job | nnodes per engine | Graphs across a dump |
|---|---|---|
| TP=8, `n_gpus: 8` | 1 | kept, then `rebind_graphs` |
| TP=8, `n_gpus: 16` (two replicas) | 1 **each** | kept, then `rebind_graphs` |
| TP=16, `n_gpus: 16` | 2 | **dropped**, then `recapture_graphs` |

Two TP=8 replicas across two nodes are two *single-node* engines and take the
old path entirely. Only a single engine spanning pods is different.

## 1. Why the graphs cannot survive a multi-node dump

Across nodes vLLM turns custom all-reduce off, so every all-reduce a captured
graph holds is an NCCL kernel. NCCL then refuses to release the communicator.
From NCCL 2.29.7 `src/enqueue.cc`:

```c
if (persistent) {
  comm->sharedRes->persistentRefs += nPlans;
  comm->localPersistentRefs += nPlans;
  NCCLCHECKGOTO(ncclCudaGraphAddDestructor(planner->capturingGraph, persistentDestructor, (void*)planHead), result, failure);
}
```

`ncclCudaGraphAddDestructor` (`src/misc/strongstream.cc`) creates a
`cudaUserObject` and hands its only reference to the graph with
`cudaGraphRetainUserObject(..., cudaGraphUserObjectMove)`. `commDestroySync`
(`src/init.cc`) then waits on exactly that count, and says so:

```c
// And keep polling until all graphs referencing us die.
while (comm->localPersistentRefs != 0) {
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
}
```

That loop is the `ncclCommAbort` hang a graph-mode TP=16 dump used to wedge in
(`commReclaim -> commDestroySync -> ncclCommPollCallbacks`). The count only
falls in `reclaimPlan`, which CUDA triggers when the graph is destroyed.

**Preserving them is blocked twice over**, which is why no amount of patching
gets the rebind path to work here:

1. The refcount cannot be released without destroying the graph. CUDA offers no
   way to enumerate or release a graph's retained user objects from outside.
2. Even forced, the graphs would be wrong. `reclaimPlan` `cudaFree`s
   `plan->workBufPersistent`, the device work buffer the graph's kernels point
   into, and `reinit_nccl` yields a comm with a different channel count (2 on
   sockets at dump time, 16 on EFA). The baked kernel arguments would reference
   freed, differently shaped state.

So the dump destroys the graphs before `destroy_nccl` and the restore captures
them again. **The recapture is not overhead for having been restored** — it is
the same capture a cold start pays, which the restore simply cannot inherit.

## 2. The primitives

`drop_graphs()` and `recapture_graphs()` in `instance.py`, both
`collective_rpc` fan-outs like `rebind_graphs`.

- **`drop_graphs()`** resets every graph `_collect_graph_entries` finds — the
  wrapper entries, the V2 runner's `cudagraph_manager.graphs`, and the
  piecewise segments reachable only through bound `replay` methods (4029 of
  GLM-5.3's 4080 per rank). It then empties the containers, so a later capture
  cannot replay a reset graph, and refreshes the shared graph pool, without
  which the recapture mints private per-size pools and runs out of memory.
  `cuda_checkpoint` inserts it ahead of `destroy_nccl` at `nnodes > 1`.
- **`recapture_graphs()`** calls `model_runner.capture_model()` directly, not
  `compile_or_warm_up_model` (which also redoes one-time setup). Run after
  `reinit_nccl`, the weight restore and `wake_up_kv_cache`.
- **`rebind_graphs()` raises at `nnodes > 1`.** There is nothing to rebind, and
  a quiet no-op would report success and then wedge on the first replay of a
  graph that no longer exists.

Two vLLM 0.30 facts the recapture depends on, both verified by reading the
installed source:

- **`capture_model()` does recapture on a second call.** It gates on
  `cudagraph_manager.needs_capture()`, which is only
  `len(self._capture_descs) > 0`, and `capture()` never clears that list. This
  was the assumption most likely to sink the whole approach.
- **`capture_model()` ends with `lock_workspace()`**, which makes the workspace
  refuse any allocation larger than the current one. The recapture calls
  `unlock_workspace()` first; `capture_model()` re-locks on the way out.
  Without it the recapture runs in a state the cold-start capture never saw.

## 3. The keep-graph machinery is skipped — and that is load-bearing

`install_keepgraph_patch`, `instantiate_captured_graphs`, the warm pass, the
`COLD IMAGE` census, `store_snapshot` / the rank_data reuse patch and
`dump_sp_nccl_nodes` all exist to serve the *rebind* path. At `nnodes > 1` none
of them has a job: the retained topology is never read, and the instantiate
pass would build execs for 4080 graphs per rank that `drop_graphs` destroys
minutes later.

**Skipping them also fixes a numerical difference, which was not the reason for
doing it.** With the machinery on, a cold start captured with `keep_graph=True`
and built its execs in a separate `instantiate_captured_graphs` pass, while the
recapture used stock `capture_model()`. That asymmetry was assumed harmless
"because nothing reads the topology". It was not: Qwen3-8B's restored output
differed from its cold reference by one greedy token, reproducibly at the same
character across two independent cycles. With the machinery skipped both sides
take the same plain-vLLM path and the outputs are bit-identical. GLM-5.3 never
showed the difference either way.

The gate reaches the vLLM worker processes as **`SEMIP_NNODES`** in the
environment, because `_semip_worker.py` runs in processes vLLM spawns: they
import the module fresh and never see the child's state. `vllm_child` exports
it before `LLM(**vllm_config)`.

## 4. `MultiNode`: node identity stays out of the key

`multinode.py` holds a frozen `MultiNode(node_rank, master_addr, master_port,
ifname)`, passed as `Instance(vllm_config, model_dir, multinode=...)`. `None`
is single-node.

**`nnodes` lives in `vllm_config` and is hashed; everything else in `MultiNode`
is not.** The split changes the image — a half of a TP=16 group is not a TP=8
engine — so it belongs in the key. Node identity changes on every restore and
must not: the experiment driver put `node_rank`, `master_addr` and
`master_port` in `vllm_config`, where it is recorded in `meta.json`, compared
by `criu_restore` and hashed into `cfg12`, so the two halves of one job hashed
differently and no restored pair could match its own image.

The child merges node identity into **its own private copy** of the config, so
the dict `criu_dump` records (the parent's) never sees it. That is what lets a
restored pair rendezvous somewhere new.

`MultiNode` reaches the child as a **spawn argument**, not just an `init`
kwarg: the child pins its NCCL/gloo interface, `VLLM_HOST_IP` and the pinned
aws-ofi-nccl values before importing vLLM, which happens before the `init`
command is read off the pipe.

## 5. The cold-start environment, and why it is pinned

At `nnodes > 1` the child sets, from `multinode.py`:

```
NCCL_NET=Socket        NCCL_GIN_ENABLE=0      NCCL_RAS_ENABLE=0
NCCL_BUFFSIZE=8388608            NCCL_P2P_NET_CHUNKSIZE=524288
NCCL_NVLS_CHUNKSIZE=524288       NCCL_NVLSTREE_MAX_CHUNKSIZE=524288
NCCL_NET_FORCE_FLUSH=0           NCCL_NETDEVS_POLICY=max:1
```

`NCCL_NET=Socket` keeps EFA out of the image: EFA state does not survive CRIU,
and the restore's `reinit_nccl` is the first EFA bring-up.

The six `PINNED_OFI_ENV` values are what aws-ofi-nccl 1.21.1 exports on EFA.
**NCCL caches its parameters at the first init in a process**, so a socket cold
start would otherwise cache socket-era defaults and the plugin's EFA values
would arrive at the re-init, too late to take effect — leaving the restored
engine with `P2P_NET_CHUNKSIZE` 131072 instead of 524288 and a different
ring/tree channel order. Pinning them is what makes a restored engine
bit-identical to an EFA cold start. `NCCL_TOPO_FILE` is deliberately **not**
pinned: it is fd-backed, so the plugin must supply it at every init.

`reinit_nccl` reports any drift between the plugin's exports and the pinned set,
so a plugin upgrade that changes a default is noticed rather than silently
breaking bit-identity.

**`_env` with an empty value means *unset*.** For `NCCL_NET`, absent and empty
are different answers — absent is what lets NCCL pick its plugin and come up on
EFA. This is how a cold *reference* run opts out of the Socket pin.

## 6. Why `reinit_nccl` takes kwargs only

`reinit_nccl(master_addr=, port=, ifname=)` reads nothing from the
environment. **The child is a restored process, so its `environ` is the
dump's**; nothing the restoring job sets would be visible there. `port` is what
lets the engine pin one rendezvous for both halves — a port that is free on the
follower says nothing about the node that binds it.

## 7. The joint protocol

Dump:

```
both      init (they rendezvous inside vLLM; neither returns until both arrive)
leader    generate, attach, stage, save_weights, detach, sleep, arm_mq_park
leader    cuda_checkpoint   -> drop_graphs + destroy_nccl across ALL ranks
agents    cuda_checkpoint   (must follow: the leader's tears down their ranks too)
leader    criu_dump         -> parks every rank's queue as its last collective
agents    wait_parked, then criu_dump
```

Restore:

```
both      criu_restore
leader    mq_begin_unpark -> handle
agents    mq_follower_unpark(handle) -> response handles
leader    mq_finish_unpark(handles)
both      cuda_restore
leader    reinit_nccl (first EFA bring-up), attach, load_weights,
          wake_up_weights, repin, plan/restore_weights, wake_up_kv_cache,
          recapture_graphs
```

**Joint or nothing.** A half that cold-starts while the other restores would
rendezvous with a group that does not exist, and the failure is a deadlock in
the first collective rather than an error. Every dump stamps a `dump_id` into
both halves' `meta.json`, and a hit requires every half to carry the same one;
a missing half, a mismatched id or a partial materialize all make both halves
cold-start and dump together.

An **L7 config digest** is exchanged before `init` for the same reason: the
halves profile their shapes independently and then meet in a collective, so a
disagreement deadlocks for the full gloo timeout with nothing in either log
naming the field.

## 8. Layout, placement and timeouts

- **Layout:** `<key>/node<k>/{image,compilation}` per half. Weights stay at
  `<key>/weight/rank{0..15}` — the shards are named by *global* rank, and a
  restore onto a different pod pair has to find all of them in one place.
  `meta.json` gains `nnodes`, `node_rank`, `dump_id`, `dump_ip`.
- **Placement:** dss builds `nnodes` whole-node bundles with `STRICT_SPREAD`
  (`build_inference_pg(..., per_node=True)`). PACK could put both halves on one
  node, where they would contend for the same GPUs and the second node would
  never appear. `ReplicaPool` gives the leader `world_size / nnodes` GPUs in
  bundle 0 and each agent its own bundle, and keeps
  `distributed_executor_backend="mp"`: forcing `"ray"` would both hand vLLM a
  backend it never uses and change the config `criu_restore` compares byte for
  byte.
- **Cold-start timeout 3600 s.** The slowest half's first weight load was 25
  minutes for GLM-5.3 on a cold page cache, and gloo's own rendezvous timeout
  is 1800 s; anything shorter fails jobs that were about to succeed.
- **Restore prefetch:** this node's `weight/rank*` are read in the background
  during the CRIU/CUDA/NCCL steps, which touch no disk. A pod that did not dump
  reads its shards cold, and that is the whole difference between a same-pod
  restore (8 s) and a swapped one (33 s).
- **Teardown fans out.** After the leader is gone the agents hold a process
  tree that can never be driven again — its executor was the leader's — so
  leaving them alive would pin a pod's GPUs for the rest of the job.

## 9. Measured, 2026-10-06, production library on two pods

Graph mode, TP=16, with the pinned env. **Both models restored bit-identical to
a graph-mode EFA cold reference**, which is the correctness criterion.

| | Qwen3-8B | GLM-5.3 (`gmu` 0.75) |
|---|---|---|
| Cold `init` on EFA | 172 s | 1668 s |
| `drop_graphs` | 264/rank, 36 MiB freed | 4080/rank, 724 MiB freed |
| `cuda_checkpoint` (incl. `destroy_nccl`) | 25.7 s | 41.7 s |
| `criu_dump` | 11.4 s | 41.5 s |
| socket census | `[]` both pods | `[]` both pods |
| `recapture_graphs` | 264/264, 8.4 s | 4080/4080, 70.2 s, ~1 GiB/rank |
| Restore total | 35 s | 164 s |

Memory is not a constraint: the recapture cost ~1 GiB per rank against a 4612
MiB cold-start estimate and 13-18 GiB free.

**What this does not cover.** The overlay check drives `Instance` directly, so
the engine layer — `SemipNodeAgent`, the joint `dump_id` hit, the Ray
placement, the mq plane driven through actors — has not served a job yet. The
publisher also does **not** understand `node<k>/` (see §10).

## 10. Known gaps

- **`semip_publish.py` does not handle `node<k>/`.** It knows the flat and
  `replica<K>` layouts only, so a two-node dump cannot be published yet. Each
  pod holds just its own half, so this needs a two-pod staging rendezvous: per-
  node rows under `_staging/<key>/<dump_id>/node<k>.json`, a `wt12` over the
  union, each node uploading its own `rank*` and `node<k>/`, and node 0 writing
  the sentinel last.
- **L5-L11 robustness items are not done**: the subreaper (`PR_SET_CHILD_SUBREAPER`
  plus a `waitpid` sweep, which is what makes a second restore in one pod work),
  the `mq_plane` `ShmRingBuffer.__del__` double close, loud abort timeouts, and
  the CRIU clock audit.
- **Qwen3-8B's one-token difference** is resolved by §3, but it was never root-
  caused beyond "the keep_graph/instantiate asymmetry". If it reappears, that
  is where to look.
