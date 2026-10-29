# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU replay-boundary stops; real vLLM methods, synthetic scheduler, no GPU kernels.

DFlash2 on Qwen3.8-27B has 416-token target attention and GatedDeltaNet blocks and an 832-token sliding-window
drafter group. The Mamba split ends chunks on the 416 grid, while the coordinator's replay boundaries count in the
832-token scheduler block.

VLLM_SOURCE_ROOT may name a vLLM 0.31.0 checkout on CPU-only test hosts.
Otherwise read the installed vLLM package without importing its GPU runtime.
"""

import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

TARGET_BLOCK = 416
SCHEDULER_BLOCK = 832
# The stock split with the replay boundaries added to its own ``stops`` tuple.
INLINE_STOPS = (
    "(*(self.kv_cache_manager.coordinator.get_replay_boundaries(request)"
    " if self.kv_cache_manager.coordinator.eagle_group_ids else ()),)"
)


def source_function(root, file, cls, name):
    tree = ast.parse((root / file).read_text())
    scope = tree if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    [function] = [n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name]
    function.decorator_list = []
    return function


@pytest.fixture(autouse=True)
def vllm_split(monkeypatch):
    source = os.environ.get("VLLM_SOURCE_ROOT")
    spec = importlib.util.find_spec("vllm") if source is None else None
    if source is None and spec is None:
        pytest.skip("set VLLM_SOURCE_ROOT to a vLLM 0.31.0 checkout")
    root = Path(source) / "vllm" if source is not None else Path(spec.origin).parent
    split = source_function(root, "v1/core/sched/scheduler.py", "Scheduler", "_mamba_block_aligned_split")
    inline = copy.deepcopy(split)
    inline.name = "inline_split"
    [stops] = [
        node
        for node in ast.walk(inline)
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["stops"]
    ]
    stops.value.elts.append(ast.parse(INLINE_STOPS, mode="eval").body.elts[0])
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            source_function(root, "v1/kv_cache_interface.py", None, "get_mamba_prefill_checkpoint_position"),
            source_function(root, "v1/kv_cache_interface.py", None, "is_mamba_prefill_checkpoint_valid"),
            source_function(root, "v1/core/kv_cache_coordinator.py", "KVCacheCoordinator", "get_replay_boundaries"),
            split,
            inline,
        ],
        type_ignores=[],
    )
    ns = {"cdiv": lambda a, b: -(-a // b)}
    exec(compile(ast.fix_missing_locations(module), str(root), "exec"), ns)  # noqa: S102

    class Scheduler:
        _mamba_block_aligned_split = ns["_mamba_block_aligned_split"]

    vllm = ModuleType("vllm")
    vllm.__version__ = "0.31.0"
    scheduler_module = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, vllm.__name__, vllm)
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
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
        ("mamba_completion_refresh", "ensure_mamba_completion_refresh"),
    ]:
        stub = ModuleType(f"arctic_platform.inference.vllm.{name}")
        setattr(stub, entry, lambda: None)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    required_patches.apply_required_vllm_patches()
    required_patches.apply_required_vllm_patches()
    return SimpleNamespace(ns=ns, Scheduler=Scheduler, vllm=vllm, scheduler_module=scheduler_module)


def scheduler_stub(ns, *, eagle=True, budget=8192):
    coordinator = SimpleNamespace(
        eagle_group_ids={1} if eagle else set(), scheduler_block_size=SCHEDULER_BLOCK, lookups=0
    )

    def get_replay_boundaries(request):
        coordinator.lookups += 1
        return ns["get_replay_boundaries"](coordinator, request)

    coordinator.get_replay_boundaries = get_replay_boundaries
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=TARGET_BLOCK),
        use_eagle_block_drop=eagle,
        max_num_scheduled_tokens=budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=TARGET_BLOCK,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )


def request_stub(num_prompt_tokens, computed=0, num_tokens=None):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=num_prompt_tokens,
        num_tokens=num_prompt_tokens if num_tokens is None else num_tokens,
        shared_prefix_boundary=0,
    )


def chunk_ends(split, scheduler, num_prompt_tokens, budget):
    request = request_stub(num_prompt_tokens)
    ends = []
    while request.num_computed_tokens < num_prompt_tokens:
        step = split(scheduler, request, min(num_prompt_tokens - request.num_computed_tokens, budget))
        assert step > 0
        request.num_computed_tokens += step
        ends.append(request.num_computed_tokens)
    return ends


@pytest.mark.parametrize("budget", [8192, 16384])
@pytest.mark.parametrize(
    ("num_prompt_tokens", "boundaries"),
    [
        # floor((N - 1) / 416) is odd, even, odd.
        (130_307, (128_960,)),
        (235_529, (234_624,)),
        (226_904, (225_472,)),
        # A block-aligned prompt resumes at a different position when resent than when extended.
        (83_200, (81_536, 82_368)),
    ],
)
def test_prefill_ends_a_chunk_on_every_replay_boundary(vllm_split, num_prompt_tokens, boundaries, budget):
    scheduler = scheduler_stub(vllm_split.ns, budget=budget)
    assert scheduler.kv_cache_manager.coordinator.get_replay_boundaries(request_stub(num_prompt_tokens)) == boundaries

    ends = chunk_ends(vllm_split.Scheduler._mamba_block_aligned_split, scheduler, num_prompt_tokens, budget)

    assert set(boundaries) <= set(ends)


@pytest.mark.parametrize("eagle", [True, False])
def test_split_is_the_stock_split_with_the_boundaries_added_to_its_stops(vllm_split, eagle):
    ns = vllm_split.ns
    patched = vllm_split.Scheduler._mamba_block_aligned_split
    scheduler = scheduler_stub(ns, eagle=eagle)
    coordinator = scheduler.kv_cache_manager.coordinator
    shortened = 0
    for num_prompt_tokens in (1_000, 2_000, 5_000, 83_199, 83_200, 83_201, 130_307, 235_529):
        boundaries = coordinator.get_replay_boundaries(request_stub(num_prompt_tokens))
        starts = {0, *(b + d for b in boundaries for d in (-SCHEDULER_BLOCK, -TARGET_BLOCK, -1, 0, 1, TARGET_BLOCK))}
        for computed in sorted(start for start in starts if 0 <= start < num_prompt_tokens):
            for local, external in [(0, 0), (TARGET_BLOCK, 0), (0, SCHEDULER_BLOCK)]:
                for want in [1, TARGET_BLOCK, SCHEDULER_BLOCK, 2 * SCHEDULER_BLOCK, 8192, num_prompt_tokens]:
                    request = request_stub(num_prompt_tokens, computed)
                    expected = ns["inline_split"](scheduler, request, want, local, external)
                    assert patched(scheduler, request, want, local, external) == expected
                    stock = ns["_mamba_block_aligned_split"](scheduler, request, want, local, external)
                    if not eagle:
                        assert expected == stock
                    shortened += expected != stock
    # Without EAGLE groups no boundary is added; with them some chunk in the grid is shortened.
    assert (shortened > 0) == eagle

    # Decode, then a resumed request replaying its output past the prompt: no boundary is left to stop at.
    for request in [request_stub(5_000, 5_099, 5_100), request_stub(5_000, 5_050, 5_100)]:
        lookups = coordinator.lookups
        result = patched(scheduler, request, 1)
        assert coordinator.lookups == lookups
        assert result == ns["inline_split"](scheduler, request, 1)


def test_patch_wraps_the_split_once(vllm_split):
    split = vllm_split.Scheduler._mamba_block_aligned_split

    assert split.__wrapped__ is vllm_split.ns["_mamba_block_aligned_split"]
    assert split._arctic_replay_boundary_stop is True


def test_patch_refuses_an_unvalidated_vllm_version(vllm_split, monkeypatch):
    from arctic_platform.inference.vllm.mamba_replay_boundary_stop import ensure_mamba_replay_boundary_stop

    stock = vllm_split.ns["_mamba_block_aligned_split"]

    class Unpatched:
        _mamba_block_aligned_split = stock

    monkeypatch.setattr(vllm_split.scheduler_module, "Scheduler", Unpatched)
    monkeypatch.setattr(vllm_split.vllm, "__version__", "0.31.1")

    with pytest.raises(RuntimeError, match="supports vLLM v0.31.0 only"):
        ensure_mamba_replay_boundary_stop()

    assert Unpatched._mamba_block_aligned_split is stock
