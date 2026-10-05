# Slots Allocator — Design

## Purpose

`Slots` is a singleton bookkeeping facility that hands out **fractional GPU
slots** to instances. It supports recursive halving of a GPU into smaller
slots via a classic **buddy-allocator** algorithm.

It is purely metadata — it does **not** perform any CUDA work, talk to vLLM,
or coordinate process state. It tracks who has reserved how much of which
GPU, and blocks callers until the resources they want become available.

It complements (does not replace) `gpu_pool.py` / `gpu_slot.py`, which
manage exclusive whole-GPU locks. `Slots` is the layer that lets us split a
GPU into halves, quarters, eighths, etc.

## Core concepts

### Slot

A `Slot` is an immutable handle:

```python
@dataclass(frozen=True)
class Slot:
    gpu_id: int
    level: int
    index: int
```

- **`gpu_id`** — which GPU this slot lives on.
- **`level`** — how big the slot is:
  - `level=1` → a whole GPU (one full slot).
  - `level=2` → half of a GPU.
  - `level=3` → a quarter.
  - In general: a level-`L` slot covers `1 / 2^(L-1)` of a GPU.
- **`index`** — position among siblings at this level: `0 ≤ index < 2^(level-1)`.

The triple `(gpu_id, level, index)` uniquely identifies a slot in the
conceptual buddy tree.

### The conceptual buddy tree

For each GPU there's a binary tree:

```
                       L1[0]
                      /      \
                  L2[0]      L2[1]
                  /   \      /   \
              L3[0] L3[1] L3[2] L3[3]
               ...   ...   ...   ...
```

At any moment a node is in **exactly one** of three logical states:

- **`FREE`** — this exact node is available; sits in the level-`L` free-list.
- **`ALLOC`** — this exact node has been handed to a caller; sits in the live set.
- **`SPLIT`** — this node has descendants that are `FREE` or `ALLOC`; the node
  itself can't be handed out as one piece.

### Buddies, splits, merges

For any node `(g, L, i)` with `L > 1`:

- **Buddy**: `(g, L, i ^ 1)` — sibling pairs are `(0,1)`, `(2,3)`, `(4,5)`, ...
- **Parent**: `(g, L-1, i // 2)`.
- **Children**: `(g, L+1, 2i)` and `(g, L+1, 2i+1)`.

Two operations move slots vertically:

- **Split** — `(g, L, i)` becomes `(g, L+1, 2i)` and `(g, L+1, 2i+1)`. Used
  during allocation when no slot of the requested size exists, but a larger
  one does.
- **Merge / coalesce** — if `(g, L, i)` and `(g, L, i^1)` are both `FREE`,
  they collapse back into `(g, L-1, i // 2)`. Used during deallocation to
  recover larger contiguous free regions.

## Bookkeeping (no explicit tree)

We do **not** store the buddy tree as a node graph. The entire state is:

- `_pools: dict[(gpu_id, level), deque[Slot]]` — free-list per `(gpu, level)`.
  Contains exactly the `FREE` nodes.
- `_live: set[Slot]` — contains exactly the `ALLOC` nodes.
- `_last_used: dict[gpu_id, float]` — `perf_counter` of when each GPU was last
  fully released back to a single level-1 root. Drives coldest-first auto-pick.
- `_waiters: deque[Waiter]` — FIFO queue of blocked `allocate` calls.
- `_draining: set[int]` — GPUs being removed from the pool by
  `Orchestrator.sub`. `_try_allocate` skips them so no new placements
  land on a draining GPU; `pop` clears the entry once the GPU is reaped.
- `_lock` / `_cv` — single mutex (`Lock`) wrapped in a `Condition`, guarding
  all of the above and waking waiters.

`SPLIT` nodes are **not** stored — they're inferred: a node `(g, L, i)` is
`SPLIT` iff it's neither in `_pools[(g, L)]` nor in `_live`, but some
descendant of it is. The implicit tree is materialized only by `status()`
for printing — never by `allocate` or `deallocate`.

This is the standard Knuth/Knowlton buddy allocator layout.

## API

```python
class Slots:
    @classmethod
    def init(cls, gpu_ids: list[int]) -> None: ...
    @classmethod
    def allocate(cls, level: int, gpu: int | None = None) -> Slot: ...  # blocks
    @classmethod
    def deallocate(cls, slot: Slot) -> None: ...
    @classmethod
    def add(cls, gpu: int) -> bool: ...                # join pool
    @classmethod
    def pop(cls, gpu: int) -> None:  ...               # leave pool (must be idle)
    @classmethod
    def status(cls) -> None: ...
    @classmethod
    def remove(cls) -> None: ...
```

The class is a **singleton**: state lives on the class, not on instances.
`init` seeds it; `remove` tears it down (and asserts no leaks).

## Algorithms

### `init(gpu_ids)`

Seeds `_pools[(g, 1)] = [Slot(g, 1, 0)]` for every `g` and zeros
`_last_used[g]`. Each GPU starts as one big free slot.

### `_pop_free_in_subtree(g, L)` — non-blocking core (private, lock held)

1. Walk `L' = L, L-1, ..., 1`. Find the first non-empty `_pools[(g, L')]`.
2. If none exists: return `None` (cannot satisfy on this GPU right now).
3. Pop a slot `s` from that pool.
4. While `s.level < L`: split, push the unused half into its level pool,
   keep the surviving half.
5. Return `s` (now at level `L`).

### `_try_allocate(level, gpu)` — coldest-first auto-pick (private, lock held)

If `gpu is not None`: return `None` if `gpu in _draining`, else just call
`_pop_free_in_subtree(gpu, level)`.

If `gpu is None`: iterate GPUs sorted by `(_last_used[g], g)` ascending,
**skipping any GPU in `_draining`**, and try `_pop_free_in_subtree(g,
level)` on each. Return the first success or `None`.

Once auto-pick has chosen a GPU `g`, recursion stays on `g` — we never
mid-flight switch GPUs and fragment a second one.

### `allocate(level, gpu=None)` — blocking with strict FIFO

```python
with _cv:
    me = Waiter(level, gpu)
    _waiters.append(me)
    while True:
        if _waiters[0] is me:
            s = _try_allocate(level, gpu)
            if s is not None:
                _waiters.popleft()
                _live.add(s)
                _cv.notify_all()
                return s
        _cv.wait()
```

**Strict head-of-line FIFO**: only the head of `_waiters` ever runs
`_try_allocate`. A younger waiter whose request happens to be satisfiable
still waits behind the head. This is intentional — when an unsatisfiable
head appears, a later eviction/migration policy will use it as a clear
signal that space must be freed.

### `deallocate(slot)` — coalesce upward

```python
with _cv:
    _live.remove(slot)
    s = slot
    while s.level > 1:
        pool = _pools[(s.gpu_id, s.level)]
        buddy = first x in pool with x.index == s.index ^ 1
        if buddy is None:
            break
        pool.remove(buddy)
        s = parent(s)               # s.level -= 1, s.index //= 2
    _pools[(s.gpu_id, s.level)].append(s)
    if s.level == 1:
        _last_used[s.gpu_id] = perf_counter()
    _cv.notify_all()
```

**Key invariant**: after `deallocate` finishes, no level-`L` pool ever
contains both buddies of the same parent — they would have been merged.
This means the buddy lookup only ever finds 0 or 1 candidate.

`_last_used` is updated **only** when coalescing reaches level 1 (the GPU
is fully free again). Sub-slot releases don't affect coldest-first ordering.

### `status()` — tree-style printout

Walks the implicit tree from each GPU's root. For every visited node:

- If in `_pools` → leaf, label `FREE`.
- Else if in `_live` → leaf, label `ALLOC`.
- Else → internal node, label `SPLIT`, recurse into the two children.

Printed with `tree(1)`-style ASCII (`├──`, `└──`, `│   `, `    `).
Footer prints the waiter queue.

### `add(gpu)` — grow the pool at runtime

Seeds `_pools[(gpu, 1)] = [Slot(gpu, 1, 0)]`, zeros `_last_used[gpu]`,
and clears `gpu` from `_draining` (in case a prior `sub` was racing).
Then `notify_all()` so any FIFO waiter blocked on a head request that's
now satisfiable can retry.

Returns `True` if the GPU was added, `False` if it was already in the
pool. Asserts `_inited` (no implicit init from `add`).

### `pop(gpu)` — shrink the pool at runtime

Inverse of `add`. Removes `gpu` entirely from the bookkeeping after
asserting the GPU is fully idle:

- No live slot references `gpu` (`_live` is clean for `gpu`).
- The level-1 pool for `gpu` has exactly its root.
- All higher-level pools for `gpu` are empty.

These asserts are invariants the *caller* must establish — `Slots.pop`
itself never blocks waiting for residents to clear. The orchestrator
(`Orchestrator._sub_sync`) is the canonical caller: it sets
`_draining` first (so `_try_allocate` stops handing out slots on `gpu`),
walks every resident model down to `checkpoint`, and only then calls
`pop`. Calling `pop` on a non-drained GPU would `AssertionError`.

Removes the per-GPU pools, drops `_last_used[gpu]`, discards `gpu` from
`_draining`, and `notify_all()`.

### `remove()`

- Asserts `_live` is empty (no outstanding slots).
- Asserts every GPU's level-1 pool has exactly its root and all higher-level
  pools are empty.
- Clears all state. A subsequent `init` starts clean.

## Worked example

Start:

```python
Slots.init([2, 3])
```

State:
- `_pools[(2, 1)] = [Slot(2,1,0)]`
- `_pools[(3, 1)] = [Slot(3,1,0)]`
- `_live = {}`

`status()`:

```
Slots:
  GPU 2 (last_used=0.00)
    L1[0] FREE
  GPU 3 (last_used=0.00)
    L1[0] FREE
  Waiters: 0
```

### Step 1 — `s1 = Slots.allocate(level=2, gpu=3)`

For GPU 3: `L'=2` empty, `L'=1` has `Slot(3,1,0)`. Pop. Split once into
`Slot(3,2,0)` (kept) and `Slot(3,2,1)` (pushed to `_pools[(3,2)]`).
Return `Slot(3,2,0)`.

`status()`:

```
  GPU 2:
    L1[0] FREE
  GPU 3:
    L1[0] SPLIT
    ├── L2[0] ALLOC
    └── L2[1] FREE
```

### Step 2 — `s2 = Slots.allocate(level=3)`  (auto)

GPUs sorted by `(_last_used, g)`: `(0.0, 2), (0.0, 3)` → try GPU 2 first.

GPU 2: `L'=3` empty, `L'=2` empty, `L'=1` has `Slot(2,1,0)`. Pop. Split
twice. Halves pushed into `_pools[(2,2)]` (`Slot(2,2,1)`) and
`_pools[(2,3)]` (`Slot(2,3,1)`). Return `Slot(2,3,0)`.

`status()`:

```
  GPU 2:
    L1[0] SPLIT
    ├── L2[0] SPLIT
    │   ├── L3[0] ALLOC
    │   └── L3[1] FREE
    └── L2[1] FREE
  GPU 3:
    L1[0] SPLIT
    ├── L2[0] ALLOC
    └── L2[1] FREE
```

### Step 3 — `Slots.deallocate(s2)`  (`Slot(2,3,0)`)

Remove from `_live`. `s = Slot(2,3,0)`.

- Look for buddy `(2,3,1)` in `_pools[(2,3)]` → found. Remove. `s = Slot(2,2,0)`.
- Look for buddy `(2,2,1)` in `_pools[(2,2)]` → found. Remove. `s = Slot(2,1,0)`.
- `s.level == 1`, stop.

Push `Slot(2,1,0)` into `_pools[(2,1)]`. Update `_last_used[2]`.

GPU 2 is back to a single `FREE` root. GPU 3 is unchanged.

## Concurrency

A single mutex (`_lock`) wrapped in a `Condition` (`_cv`). Every public
method takes it for its full duration. `_cv.notify_all()` is broadcast on
every state change; the FIFO check (`_waiters[0] is me`) ensures only one
waiter actually proceeds per wake-up. Spurious wakeups are harmless — the
wait loop re-checks.

This is sufficient for v1. If the lock becomes hot, the standard refinement
is per-GPU locks plus a small global mutex for `_waiters` and `_last_used`.

## Invariants

These hold whenever no method is mid-execution:

1. Every node in `_pools` is `FREE`; every node in `_live` is `ALLOC`; every
   other node reachable from a root is either `SPLIT` or doesn't exist
   conceptually.
2. For any `(g, L)`, `_pools[(g, L)]` does **not** contain both buddies of
   the same parent. (Coalescing maintains this.)
3. A `Slot` is in at most one of `_pools` ∪ `_live`.
4. After `init([gpus])` and before any `allocate`: `_pools[(g, 1)]` has its
   single root, all other pools are empty, `_live` is empty. Same shape is
   required for `remove()` to succeed.

## Composability with future features

- **Eviction** — when a head request is unsatisfiable, an eviction policy
  can compute precisely which `ALLOC` nodes to free: walk from the
  requested node up to the root; any `ALLOC` descendants of that path must
  go. They get `deallocate`d (with side effects), and the head retries.
- **Migration** — `allocate(new) ; copy_state ; deallocate(old)`. The
  allocator itself is unchanged.
- **Runtime pool resize** — `add(gpu)` / `pop(gpu)` plus the `_draining`
  flag let the orchestrator grow or shrink the pool while the system is
  live. Drain semantics (move residents off, then `pop`) live in
  `Orchestrator.sub`; the allocator only enforces the "no new placements
  on draining GPUs" rule and the "fully-idle on `pop`" assertion.
- **Asymmetric per-GPU root capacities** — if a GPU is "half-size", seed
  `_pools[(g, 2)]` instead of `_pools[(g, 1)]` at init. The buddy math is
  identical; level 1 on that GPU is simply never available.

## Out of scope (v1)

- Eviction / migration policies.
- `snapshot()` returning a structured dict for tests/orchestrator.
- Timeouts on `allocate`.
- Asymmetric per-GPU capacities at init.
- Fairness policies beyond strict FIFO.
