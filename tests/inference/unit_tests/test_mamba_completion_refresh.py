"""CPU completion ordering; real vLLM methods, synthetic storage, no GPU kernels.

VLLM_SOURCE_ROOT may name a vLLM 0.31.0 checkout on CPU-only test hosts.
Otherwise read the installed vLLM package without importing its GPU runtime.
"""

import ast
from collections import defaultdict, deque, namedtuple
import importlib.util
import itertools
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def methods(root, file, cls, names, keep_classmethod=False):
    tree = ast.parse((root / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    result = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(result) == len(names)
    for fn in result:
        fn.decorator_list = [
            d for d in fn.decorator_list if keep_classmethod and isinstance(d, ast.Name) and d.id == "classmethod"
        ]
    return result


@pytest.fixture(autouse=True)
def manager_methods(monkeypatch):
    source = os.environ.get("VLLM_SOURCE_ROOT")
    root = (Path(source) / "vllm" if source else Path(importlib.util.find_spec("vllm").origin).parent) / "v1/core"
    base = ast.ClassDef(
        name="Base",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=methods(
            root,
            "single_type_kv_cache_manager.py",
            "SingleTypeKVCacheManager",
            {"__init__", "pop_blocks_for_free", "free", "remove_skipped_blocks", "_remove_blocks_in_range"},
        ),
    )
    manager = ast.ClassDef(
        name="Manager",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        decorator_list=[],
        body=methods(
            root,
            "single_type_kv_cache_manager.py",
            "MambaManager",
            {
                "__init__",
                "pop_blocks_for_free",
                "_remove_blocks_in_range",
                "remove_skipped_blocks",
                "get_num_skipped_tokens",
            },
        ),
    )
    window = ast.ClassDef(
        name="WindowManager",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        decorator_list=[],
        body=methods(
            root,
            "single_type_kv_cache_manager.py",
            "SlidingWindowManager",
            {"__init__", "_contiguous_blocks_for_hit", "get_num_skipped_tokens"},
            keep_classmethod=True,
        ),
    )
    window_spec = ast.ClassDef(
        name="WindowSpec",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=methods(root.parent, "kv_cache_interface.py", "SlidingWindowSpec", {"max_admission_blocks_per_request"}),
    )
    functions = methods(root, "block_pool.py", "BlockPool", {"free_blocks", "get_new_blocks"})
    functions += methods(root, "sched/scheduler.py", "Scheduler", {"_drain_deferred_frees"})
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            base,
            manager,
            window,
            window_spec,
            *functions,
        ],
        type_ignores=[],
    )
    global ns
    ns = {"defaultdict": defaultdict, "MambaSpec": SimpleNamespace, "cdiv": lambda a, b: (a + b - 1) // b}
    exec(compile(ast.fix_missing_locations(module), str(root), "exec"), ns)  # noqa: S102
    target = ModuleType("vllm.v1.core.single_type_kv_cache_manager")
    target.MambaManager = ns["Manager"]
    target.SlidingWindowManager = ns["WindowManager"]
    monkeypatch.setitem(sys.modules, target.__name__, target)
    interface = ModuleType("vllm.v1.kv_cache_interface")
    interface.SlidingWindowSpec = ns["WindowSpec"]
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    ns["original_window_cap"] = ns["WindowSpec"].max_admission_blocks_per_request
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.31.0"
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    # Loading the patch through the required entry point also covers vanilla vLLM.
    from arctic_platform.inference.vllm import required_patches

    for name, entry in [
        ("config", "ensure_dflash_max_position_patch"),
        ("router_replay", "ensure_router_replay_vllm_patches"),
        ("xgrammar_stop_mask", "ensure_xgrammar_stop_mask_fix"),
        ("dense_prompt_logprobs", "ensure_dense_prompt_logprobs_patch"),
        ("spec_decode_grammar", "ensure_spec_decode_grammar_fix"),
        ("dflash2_nan_fix", "apply_dflash2_nan_fixes"),
        ("glm5next_indexer_page", "apply_glm5next_indexer_page_fix"),
        ("kv_group_size", "ensure_kv_group_size_patch"),
        ("mrope_yarn", "ensure_mrope_yarn_fix"),
        ("glm5next_lora", "ensure_glm5next_lora"),
        ("deepseek_v41_lora", "ensure_deepseek_v41_lora"),
    ]:
        stub = ModuleType(f"arctic_platform.inference.vllm.{name}")
        setattr(stub, entry, lambda: None)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    patch_name = "arctic_platform.inference.vllm.mamba_completion_refresh"
    if importlib.util.find_spec(patch_name):
        from arctic_platform.inference.vllm import mamba_completion_refresh

        monkeypatch.setattr(mamba_completion_refresh, "_APPLIED", False)
    required_patches.apply_required_vllm_patches()
    required_patches.apply_required_vllm_patches()
    return root


class Queue:
    def __init__(self):
        self.blocks = deque()

    def append_n(self, blocks):
        self.blocks.extend(blocks)

    def prepend_n(self, blocks):
        self.blocks.extendleft(reversed(blocks))

    def popleft_n(self, n):
        return [self.blocks.popleft() for _ in range(n)]

    def remove(self, block):
        self.blocks.remove(block)


class Block:
    def __init__(self, name, pool):
        self.name = self.block_hash = name
        self.pool = pool
        self.is_null = False
        self.ref_cnt = 1


def setup(retained, active_last=False):
    queue = Queue()
    mapping = {}
    pool = SimpleNamespace(
        free_block_queue=queue,
        enable_caching=True,
        _reuse_watchers={},
        metrics_collector=None,
        cached_block_hash_to_block=SimpleNamespace(get_one_block=mapping.get),
    )
    pool.free_blocks = lambda blocks: ns["free_blocks"](pool, blocks)
    pool.get_num_free_blocks = lambda: len(queue.blocks)

    def evict(block):
        mapping.pop(block.block_hash, None)
        block.block_hash = None

    pool._maybe_evict_cached_block = evict
    null = Block("NULL", pool)
    null.is_null = True
    blocks = {i: Block(f"M{i}", pool) for i in retained}
    mapping.update((b.block_hash, b) for b in blocks.values())
    pool.null_block = null
    spec = SimpleNamespace(
        block_size=1,
        dcp_sharded=False,
        mamba_cache_mode="align",
        num_speculative_blocks=0,
        num_prefill_checkpoint_blocks=1,
    )
    obj = ns["Manager"](spec, pool, enable_caching=True, kv_cache_group_id=0, scheduler_block_size=1)
    obj.req_to_blocks = {"r": [blocks.get(i, null) for i in range(1, 9)]}
    obj._allocated_block_reqs = {"r"}
    obj.num_cached_block = {"r": 8}
    for i in range(1, 8 if active_last else 9):
        obj._remove_blocks_in_range("r", i - 1, i)
    return obj, pool, blocks, mapping


def finish(obj, deferred):
    pool = obj.block_pool
    if deferred:
        blocks = obj.pop_blocks_for_free("r")
        scheduler = SimpleNamespace(
            deferred_frees=deque([(2, blocks)]), processed_step_seq=1, kv_cache_manager=SimpleNamespace(block_pool=pool)
        )
        snapshot = tuple(pool.free_block_queue.blocks)
        ns["_drain_deferred_frees"](scheduler)
        assert tuple(pool.free_block_queue.blocks) == snapshot
        assert len(scheduler.deferred_frees) == 1
        scheduler.processed_step_seq = 2
        ns["_drain_deferred_frees"](scheduler)
        assert not scheduler.deferred_frees
    else:
        obj.free("r")
    assert "r" not in obj.req_to_blocks
    assert "r" not in getattr(obj, "_retired_checkpoint_hashes", {})


@pytest.mark.parametrize("retained", [(8,), (4, 8), tuple(range(1, 9))])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("active_last", [False, True])
def test_completion_eviction_order(retained, deferred, active_last):
    obj, pool, blocks, mapping = setup(retained, active_last)
    finish(obj, deferred)
    assert pool.get_num_free_blocks() == len(retained)
    assert all(b.ref_cnt == 0 for b in blocks.values())
    order = [b.name for b in pool.free_block_queue.blocks]
    expected = [f"M{i}" for i in reversed(retained)]
    assert order == expected, (retained, deferred, active_last, order, expected)
    hits = [max(retained)]
    for _ in retained:
        ns["get_new_blocks"](pool, 1)
        hits.append(max((i for i, b in blocks.items() if b.block_hash is not None), default=0))
    assert hits == [*reversed(retained), 0], hits


@pytest.mark.parametrize("kind", ["stale_hash", "shared_ref", "alias_duplicate", "moved_hash"])
def test_completion_preserves_residency_and_references(kind):
    obj, pool, blocks, mapping = setup((4, 8))
    old = blocks[4]
    if kind == "stale_hash":
        mapping.pop("M4")
        old.block_hash = "OTHER"
        mapping["OTHER"] = old
    elif kind == "shared_ref":
        pool.free_block_queue.remove(old)
        old.ref_cnt = 1
    elif kind == "alias_duplicate":
        mapping["M4"] = blocks[8]
    elif kind == "moved_hash":
        pool.free_block_queue.remove(old)
        old.block_hash = "OTHER"
        old.ref_cnt = 1
        moved = Block("M4", pool)
        moved.ref_cnt = 0
        mapping["M4"] = moved
        pool.free_block_queue.append_n([moved])
    # The request holds its newest checkpoint until completion releases it.
    newest = blocks[8]
    assert newest.ref_cnt == 1 and newest not in pool.free_block_queue.blocks
    before_count = pool.get_num_free_blocks() + 1
    before_refs = {id(b): b.ref_cnt for b in (*blocks.values(), *mapping.values())}
    before_refs[id(newest)] = 0
    finish(obj, True)
    assert pool.get_num_free_blocks() == before_count
    assert all(b.ref_cnt == before_refs[id(b)] for b in (*blocks.values(), *mapping.values()))
    order = [b.name for b in pool.free_block_queue.blocks]
    assert len({id(b) for b in pool.free_block_queue.blocks}) == len(order)
    if kind in {"stale_hash", "alias_duplicate"}:
        assert pool.free_block_queue.blocks[0] is old
    if kind == "shared_ref":
        assert old not in pool.free_block_queue.blocks
    if kind == "moved_hash":
        assert order == ["M8", "M4"]


def test_skipped_checkpoint_rejoins_fenced_release():
    obj, pool, blocks, mapping = setup((4, 8), active_last=True)
    # The final state can be retired separately after the range cursor passed it.
    obj._num_retired_blocks["r"] = 8
    obj.last_state_block_idx["r"] = 7
    obj.remove_skipped_blocks("r", 9)
    assert blocks[8].ref_cnt == 1  # held by the request until completion
    finish(obj, True)
    assert [b.name for b in pool.free_block_queue.blocks] == ["M8", "M4"]


@pytest.mark.parametrize("attention_first", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_hybrid_completion_retains_a_partial_hit(attention_first, deferred, manager_methods):
    root = manager_methods
    functions = []
    for file, cls, method, name in [
        ("kv_cache_coordinator.py", "KVCacheCoordinator", "free", "coordinator_free"),
        ("kv_cache_coordinator.py", "KVCacheCoordinator", "pop_blocks_for_free", "coordinator_pop"),
        ("single_type_kv_cache_manager.py", "MambaManager", "find_longest_cache_hit", "mamba_hit"),
        ("single_type_kv_cache_manager.py", "FullAttentionManager", "find_longest_cache_hit", "attention_hit"),
    ]:
        [fn] = methods(root, file, cls, {method})
        fn.name = name
        functions.append(fn)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    scope = dict(
        itertools=itertools,
        cdiv=lambda a, b: (a + b - 1) // b,
        MambaSpec=SimpleNamespace,
        FullAttentionSpec=SimpleNamespace,
        ChunkedLocalAttentionSpec=SimpleNamespace,
        resolve_block_hashes=lambda hashes, *args, **kwargs: hashes,
    )
    exec(compile(ast.fix_missing_locations(module), str(root), "exec"), scope)  # noqa: S102
    obj, pool, states, mapping = setup((4, 8))
    attention = [Block(f"A{i}", pool) for i in range(1, 9)]
    mapping.update((block.block_hash, block) for block in attention)
    manager = SimpleNamespace(
        pop_blocks_for_free=lambda request_id: attention, free=lambda request_id: pool.free_blocks(reversed(attention))
    )
    managers = [manager, obj] if attention_first else [obj, manager]
    coordinator = SimpleNamespace(single_type_managers=managers)
    if deferred:
        blocks = scope["coordinator_pop"](coordinator, "r")
        scheduler = SimpleNamespace(
            deferred_frees=deque([(2, blocks)]), processed_step_seq=1, kv_cache_manager=SimpleNamespace(block_pool=pool)
        )
        ns["_drain_deferred_frees"](scheduler)
        assert pool.get_num_free_blocks() == 0
        scheduler.processed_step_seq = 2
        ns["_drain_deferred_frees"](scheduler)
    else:
        scope["coordinator_free"](coordinator, "r")
    pool.hash_block_size = 1
    pool.get_cached_block = lambda h, groups: (
        [mapping[f"{groups[0]}{h + 1}"]] if f"{groups[0]}{h + 1}" in mapping else None
    )
    cls = SimpleNamespace(supports_fine_grained_hash_lookup=True)
    spec = SimpleNamespace(block_size=1)
    hits = []
    while True:
        attention_hit = scope["attention_hit"](cls, range(8), 8, ["A"], pool, spec, False, 1)[1]
        hits.append(scope["mamba_hit"](cls, range(8), attention_hit, ["M"], pool, spec, False, 1)[1])
        if not pool.get_num_free_blocks():
            break
        ns["get_new_blocks"](pool, 1)
    assert hits[0] == 8 and hits[-1] == 0
    assert 4 in hits, hits
    assert hits == sorted(hits, reverse=True)


class FullAttentionSpec(SimpleNamespace):
    pass


class ChunkedLocalAttentionSpec(SimpleNamespace):
    pass


class HybridMambaSpec(SimpleNamespace):
    pass


Group = namedtuple("Group", "spec group_ids manager_cls use_eagle")
PROMPT, DECODED, WINDOW = 8, 14, 3


def hybrid_scope(root):
    functions = []
    for file, cls, method, name in [
        ("kv_cache_coordinator.py", "KVCacheCoordinator", "free", "coordinator_free"),
        ("kv_cache_coordinator.py", "KVCacheCoordinator", "pop_blocks_for_free", "coordinator_pop"),
        ("kv_cache_coordinator.py", "HybridKVCacheCoordinator", "find_longest_cache_hit", "hybrid_hit"),
        ("single_type_kv_cache_manager.py", "FullAttentionManager", "find_longest_cache_hit", "attention_hit"),
        ("single_type_kv_cache_manager.py", "MambaManager", "find_longest_cache_hit", "mamba_hit"),
        ("single_type_kv_cache_manager.py", "MambaManager", "reachable_block_mask", "mamba_mask"),
        ("single_type_kv_cache_manager.py", "SlidingWindowManager", "find_longest_cache_hit", "window_hit"),
        ("single_type_kv_cache_manager.py", "SlidingWindowManager", "reachable_block_mask", "window_mask"),
    ]:
        [fn] = methods(root, file, cls, {method})
        fn.name = name
        functions.append(fn)
    # vLLM 0.32 factored the EAGLE lookahead margin into a module helper.
    utils = ast.parse((root / "kv_cache_utils.py").read_text()).body
    functions += [n for n in utils if isinstance(n, ast.FunctionDef) and n.name == "eagle_proof_margin"]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    scope = dict(
        itertools=itertools,
        cdiv=ns["cdiv"],
        resolve_block_hashes=lambda hashes, *args, **kwargs: hashes,
        FullAttentionSpec=FullAttentionSpec,
        ChunkedLocalAttentionSpec=ChunkedLocalAttentionSpec,
        MambaSpec=HybridMambaSpec,
        SlidingWindowSpec=ns["WindowSpec"],
    )
    exec(compile(ast.fix_missing_locations(module), str(root), "exec"), scope)  # noqa: S102
    return scope


def replay_boundaries(root, coordinator, num_prompt_tokens):
    """vLLM 0.32 computes these in a module function; 0.31 on the coordinator."""
    tree = ast.parse((root / "kv_cache_coordinator.py").read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "replay_boundaries"]
    fn = found[0] if found else methods(root, "kv_cache_coordinator.py", "KVCacheCoordinator", {"get_replay_boundaries"})[0]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[]
    )
    scope = {}
    exec(compile(ast.fix_missing_locations(module), str(root), "exec"), scope)  # noqa: S102
    if found:
        return scope["replay_boundaries"](
            num_prompt_tokens, coordinator.scheduler_block_size, use_eagle=bool(coordinator.eagle_group_ids)
        )
    return scope["get_replay_boundaries"](coordinator, SimpleNamespace(num_prompt_tokens=num_prompt_tokens))


def hybrid_turn(root, *, eagle, window, churn, deferred, resend=False):
    """One finished turn of a hybrid request; returns what the next, longer turn (or an identical resend) can reuse.

    Real vLLM retention masks, retirement, allocation, completion release, per-group hit
    finders and hybrid reconciliation; positions are tokens (block size 1). Decode runs
    past the sliding window, so the boundary's recurrent state and window tail retire
    while the attention blocks stay referenced.
    """
    scope = hybrid_scope(root)
    queue, mapping = Queue(), {}
    pool = SimpleNamespace(
        free_block_queue=queue,
        enable_caching=True,
        _reuse_watchers={},
        metrics_collector=None,
        cached_block_hash_to_block=SimpleNamespace(get_one_block=mapping.get),
        hash_block_size=1,
    )
    pool.free_blocks = lambda blocks: ns["free_blocks"](pool, blocks)
    pool.get_num_free_blocks = lambda: len(queue.blocks)
    pool.get_cached_block = lambda h, gids: (
        [mapping[(g, h)] for g in gids] if all((g, h) in mapping for g in gids) else None
    )

    def evict(block):
        mapping.pop(block.block_hash, None)
        block.block_hash = None

    pool._maybe_evict_cached_block = evict
    pool.null_block = Block("NULL", pool)
    pool.null_block.is_null = True

    def new(key, hashed=True, ref=1):
        block = Block(key, pool)
        block.ref_cnt = ref
        if hashed:
            mapping[key] = block
        else:
            block.block_hash = None
        return block

    others = [new(("other", i), ref=0) for i in range(4)]
    queue.append_n(others)
    attention_cls = type(
        "Attention",
        (),
        {"find_longest_cache_hit": classmethod(scope["attention_hit"]), "supports_fine_grained_hash_lookup": True},
    )
    mamba_cls = type(
        "Mamba", (), {"find_longest_cache_hit": classmethod(scope["mamba_hit"]), "supports_fine_grained_hash_lookup": True}
    )
    window_cls = type(
        "Window",
        (),
        {
            "find_longest_cache_hit": classmethod(scope["window_hit"]),
            "_contiguous_blocks_for_hit": ns["WindowManager"].__dict__["_contiguous_blocks_for_hit"],
            "supports_fine_grained_hash_lookup": False,
        },
    )
    attention_blocks = [new((0, i)) for i in range(DECODED)]
    attention = SimpleNamespace(
        block_size=1,
        block_pool=pool,
        dcp_world_size=1,
        pcp_world_size=1,
        retains_longer_hit=False,
        pop_blocks_for_free=lambda request_id: attention_blocks,
        free=lambda request_id: pool.free_blocks(reversed(attention_blocks)),
    )
    mspec = HybridMambaSpec(
        block_size=1, dcp_sharded=False, mamba_cache_mode="align", num_speculative_blocks=0, num_prefill_checkpoint_blocks=1
    )
    mamba = ns["Manager"](mspec, pool, enable_caching=True, kv_cache_group_id=1, scheduler_block_size=1)
    managers = [attention, mamba]
    groups = [Group(FullAttentionSpec(block_size=1), [0], attention_cls, eagle), Group(mspec, [1], mamba_cls, eagle)]
    if window:
        wspec = ns["WindowSpec"]()
        wspec.block_size, wspec.sliding_window, wspec.extra_retained_tokens, wspec.dcp_sharded = 1, WINDOW, 0, False
        sliding = ns["WindowManager"](wspec, block_pool=pool, enable_caching=True, kv_cache_group_id=2, scheduler_block_size=1)
        sliding.use_eagle = eagle
        managers.append(sliding)
        groups.append(Group(wspec, [2], window_cls, eagle))
    coordinator = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=managers),
        attention_groups=groups,
        single_type_managers=managers,
        hash_block_size=1,
        enable_partial_hash_hits=False,
        _cache_hit_alignment_tokens=1,
        eagle_group_ids=set(range(len(managers))) if eagle else set(),
        scheduler_block_size=1,
    )
    # Retention 0 (the default): only the replay boundary's state and window tail are hashed.
    boundaries = replay_boundaries(root, coordinator, PROMPT)
    states = scope["mamba_mask"](None, 0, DECODED - 1, 1, mspec, eagle, 0, boundaries)
    mamba.req_to_blocks["r"] = [new((1, i)) if keep else pool.null_block for i, keep in enumerate(states)]
    mamba.req_to_blocks["r"].append(new(("running state",), hashed=False))
    mamba._allocated_block_reqs.add("r")
    if window:
        tail = scope["window_mask"](window_cls, 0, DECODED, 1, wspec, eagle, 0, boundaries)
        sliding.req_to_blocks["r"] = [new((2, i), hashed=keep) for i, keep in enumerate(tail)]
    owned = {m.kv_cache_group_id: [b for b in m.req_to_blocks["r"] if not b.is_null] for m in managers[1:]}
    for processed in (PROMPT, DECODED):  # end of prefill, then decode past the window
        for manager in managers[1:]:
            manager.remove_skipped_blocks("r", processed)
    taken = ns["get_new_blocks"](pool, pool.get_num_free_blocks()) if churn else []
    held = {
        m.kv_cache_group_id: sum(
            b.ref_cnt > 0 and b not in taken and not any(b is t for t in m.req_to_blocks["r"])
            for b in owned[m.kv_cache_group_id]
        )
        for m in managers[1:]
    }
    live_window = sum(not b.is_null for b in sliding.req_to_blocks["r"]) if window else 0
    if deferred:
        blocks = scope["coordinator_pop"](coordinator, "r")
        scheduler = SimpleNamespace(
            deferred_frees=deque([(2, blocks)]), processed_step_seq=1, kv_cache_manager=SimpleNamespace(block_pool=pool)
        )
        ns["_drain_deferred_frees"](scheduler)
        scheduler.processed_step_seq = 2
        ns["_drain_deferred_frees"](scheduler)
    else:
        scope["coordinator_free"](coordinator, "r")
    everything = [*others, *attention_blocks, *itertools.chain(*owned.values())]
    assert pool.get_num_free_blocks() == len(everything) - len(taken)
    assert all(b.ref_cnt == 0 for b in everything if b not in taken)
    if resend:  # the identical prompt again: hashes of the prompt only, capped one token short
        hit = scope["hybrid_hit"](coordinator, list(range(PROMPT)), PROMPT - 1)[1]
    else:
        hit = scope["hybrid_hit"](coordinator, list(range(DECODED + 2)), DECODED + 1)[1]
    return SimpleNamespace(
        hit=hit,
        boundary=min(boundaries) if resend else max(boundaries),
        held=held,
        live_window=live_window,
        tail=window_cls._contiguous_blocks_for_hit(WINDOW, 1, eagle),
        window_cap=wspec.max_admission_blocks_per_request(0, DECODED + 2) if window else None,
    )


@pytest.mark.parametrize("eagle,window", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("churn", [False, True])
def test_hybrid_turn_keeps_its_boundary_through_the_run(eagle, window, churn, deferred, manager_methods):
    """Allocation churn while a request decodes must not cost the next turn its prefix."""
    turn = hybrid_turn(manager_methods, eagle=eagle, window=window, churn=churn, deferred=deferred)
    assert turn.hit == turn.boundary, turn


@pytest.mark.parametrize("eagle,window", [(False, False), (False, True), (True, True)])
def test_hybrid_turn_holds_at_most_one_boundary_per_group(eagle, window, manager_methods):
    """The extra pages a running request keeps are one recurrent state and one window tail."""
    turn = hybrid_turn(manager_methods, eagle=eagle, window=window, churn=True, deferred=False)
    assert turn.held[1] <= 1, turn
    if window:
        assert turn.held[2] <= turn.tail, turn
        # vLLM's admission invariant: a request's real-held window blocks stay within its cap.
        assert turn.live_window + turn.held[2] <= turn.window_cap, turn


@pytest.mark.parametrize("eagle,window", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("deferred", [False, True])
def test_identical_resend_hits_on_an_idle_engine(eagle, window, deferred, manager_methods):
    """An identical prompt sent again after the first finished reuses at least its replay boundary."""
    turn = hybrid_turn(manager_methods, eagle=eagle, window=window, churn=False, deferred=deferred, resend=True)
    # vLLM 0.32 moved the EAGLE margin out of the finder, so its resend hit can sit one block higher.
    assert turn.hit >= turn.boundary > 0, turn
