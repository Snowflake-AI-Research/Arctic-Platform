"""Exercise route replay from continuation admission to the experts used by the forward."""

import asyncio
from copy import copy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.layer import MoERunner
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import CustomRoutingRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import FusedTopKBiasRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from arctic_platform.inference.server.router_replay import RouterReplayCacheTX
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState
from arctic_platform.inference.vllm import forced_routes
from arctic_platform.inference.vllm.forced_routes import ForcedRoutes, bind_router

E, K, D = 16, 4, 8  # experts, top-k, hidden width


def _router(bias):
    # Grouped top-k with one group: sigmoid scores, a correction bias, renormalised, scaled.
    return GroupedTopKRouter(
        top_k=K, global_num_experts=E, num_expert_group=1, topk_group=1, renormalize=True,
        scoring_func="sigmoid", routed_scaling_factor=2.5, e_score_correction_bias=bias,
    )


@pytest.mark.parametrize("score,norm,dtype", [
    ("sigmoid", True, torch.float32), ("softmax", False, torch.bfloat16), ("sqrtsoftplus", True, torch.float32),
])
def test_masked_rows_take_the_stored_experts_with_the_trainers_replay_weights(monkeypatch, score, norm, dtype):
    histc = torch.histc  # CPU histc takes no integer input; the trainer's per-expert count is not read here
    monkeypatch.setattr(torch, "histc", lambda t, **kw: histc(t.float(), **kw))
    trainer_moe = pytest.importorskip("arctic_platform.model.implementations.moe.layers.moe")
    gen = torch.Generator().manual_seed(0)
    x, gate, bias = torch.randn(6, D, generator=gen), torch.randn(E, D, generator=gen), torch.randn(E, generator=gen)
    x, gate = x.to(dtype), gate.to(dtype)
    logits = F.linear(x, gate)
    stored = torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(3)]).to(torch.int32)

    plain = {
        "sigmoid": GroupedTopKRouter(K, E, 2, 1, norm, score, 2.5, bias),
        "softmax": FusedTopKRouter(K, E, score, norm),
        "sqrtsoftplus": FusedTopKBiasRouter(K, E, bias, norm, 2.5, scoring_func=score),
    }[score]
    if not isinstance(plain, GroupedTopKRouter):
        # The fused routers' unforced choices use GPU kernels. Exercise their select_experts and wrapper on CPU.
        def cpu_topk(self, hidden, logits, indices_type, *, input_ids=None):
            w, ids = torch.topk(logits.float(), K)
            return w, ids.int()

        monkeypatch.setattr(type(plain), "_compute_routing", cpu_topk)
    hooked = copy(plain)
    forced = ForcedRoutes(1, (2, K), torch.zeros((8, 2, K), dtype=torch.int32))
    forced.start[0], forced.end[0] = 11, 14
    forced.rows[0] = np.stack([np.zeros((3, K), np.uint16), stored.numpy().astype(np.uint16)], axis=1)
    # The request computes positions 10-15 in batch rows 0-5; its stored rows cover positions 11-13 (rows 1-3).
    forced.fill(SimpleNamespace(has_prefill=True, idx_mapping_np=np.array([0]),
                                query_start_loc_np=np.array([0, 6]), num_computed_tokens_np=np.array([10])))
    bind_router(hooked, layer_id=1, forced=forced)
    captured = []
    hooked.set_capture_fn(captured.append)

    weights, ids = hooked.select_experts(x, logits)
    plain_weights, plain_ids = plain.select_experts(x, logits)

    assert forced.mask[:6].tolist() == [False, True, True, True, False, False]
    assert torch.equal(ids[1:4], stored)
    other = [0, 4, 5]
    assert torch.equal(ids[other], plain_ids[other]) and torch.equal(weights[other], plain_weights[other])
    assert not torch.equal(ids[1:4], plain_ids[1:4])  # the stored routes are not the router's own choice here
    assert torch.equal(captured[0], ids)  # the capture records the routes the forward used

    trainer = trainer_moe.TokenChoiceTopKRouter(
        dim=D, num_experts=E, top_k=K, score_func=score, route_norm=norm,
        route_scale=1.0 if score == "softmax" else 2.5,
    ).to(dtype)
    trainer.gate.weight.data.copy_(gate)
    trainer_weights, trainer_ids, _ = trainer(x, expert_bias=bias, routed_experts=ids.long())
    assert torch.equal(trainer_ids[1:4], stored.long())
    assert weights.dtype == trainer_weights.dtype == torch.float32
    assert torch.equal(weights[1:4], trainer_weights[1:4])
    monkeypatch.setattr(hooked, "_apply_eplb_mapping", lambda logical: logical + E)
    _, physical = hooked.select_experts(x, logits)
    assert torch.equal(captured[-1], ids) and torch.equal(physical, ids + E)


B, BLOCKS = 4, 2  # prefix-cache block; speculative decoding drops the trailing cached block
L = 3  # MoE layers
STEP_TOKENS, PAD = 5, 3  # per-step token budget (prefills run in chunks); padding rows, as CUDA graphs add


class _MoERunner(MoERunner):
    """The real class, so ``install`` binds it, holding only what ``install`` reads."""

    def __init__(self, layer_id, router, monolithic=False):
        torch.nn.Module.__init__(self)
        self.layer_name = f"model.layers.{layer_id}.mlp.experts"
        self.router = router
        self.routed_experts = SimpleNamespace(quant_method=SimpleNamespace(is_monolithic=monolithic))


class _V2Runner:
    """The Model Runner V2 methods ``patch_model_runner`` wraps, reduced to their request bookkeeping."""

    def __init__(self, model, data_parallel_size=1, num_prefill_lookahead_tokens=1, kv_transfer_config=None):
        self.model, self.device, self.aux_output_connector = model, torch.device("cpu"), None
        parallel = SimpleNamespace(data_parallel_size=data_parallel_size)
        self.vllm_config = SimpleNamespace(parallel_config=parallel, kv_transfer_config=kv_transfer_config,
                                           num_prefill_lookahead_tokens=num_prefill_lookahead_tokens)
        self.max_num_reqs, self.req_states, self.pooling_runner = 8, SimpleNamespace(req_id_to_index={}), None

    def initialize_kv_cache(self):
        capturer = SimpleNamespace(device_buffer=torch.zeros(64, L, K, dtype=torch.int32))
        self.aux_output_connector = SimpleNamespace(_capturer=capturer)

    def add_requests(self, scheduler_output):
        for new_req_data in scheduler_output.scheduled_new_reqs:
            self._remove_request(new_req_data.req_id)
            taken = set(self.req_states.req_id_to_index.values())
            self.req_states.req_id_to_index[new_req_data.req_id] = min(set(range(self.max_num_reqs)) - taken)

    def _remove_request(self, req_id):
        return self.req_states.req_id_to_index.pop(req_id, None) is not None

    def prepare_inputs(self, req_ids, query_start_loc_np, num_computed_tokens_np, has_prefill=True):
        index = self.req_states.req_id_to_index
        return SimpleNamespace(has_prefill=has_prefill, idx_mapping_np=np.array([index[r] for r in req_ids]),
                               query_start_loc_np=query_start_loc_np, num_computed_tokens_np=num_computed_tokens_np)


forced_routes.patch_model_runner(_V2Runner)


def test_tail_intersects_only_recomputed_prompt_positions():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = _V2Runner(torch.nn.ModuleList([_MoERunner(0, _router(None))]))
    runner.initialize_kv_cache()
    rows = np.broadcast_to(np.arange(4, 8, dtype=np.uint8)[:, None, None], (4, L, K))
    params = SamplingParams(extra_args={forced_routes.FORCED_ROUTES_KEY: {
        "start": 4, "dtype": rows.dtype.str, "rows": rows.tobytes()}})
    # The scheduler sends rows to the first three requests only.
    reqs = [SimpleNamespace(req_id=name, prompt_len=n, sampling_params=params if name != "short" else SamplingParams())
            for name, n in [("early", 10), ("late", 10), ("full", 10), ("short", 7)]]
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=reqs[::-1]))
    runner.prepare_inputs([r.req_id for r in reqs], np.array([0, 7, 8, 10, 13, 13]), np.array([2, 6, 8, 4]))
    state = runner._forced_routes
    assert state.mask.nonzero().flatten().tolist() == [2, 3, 4, 5, 7]
    assert state.ids[state.mask, 0, 0].tolist() == [4, 5, 6, 7, 6]
    late = runner.req_states.req_id_to_index["late"]
    assert list(state.rows) == [late]  # early and full ran past their stored rows, which the runner released
    GPUModelRunner.finish_requests(runner, SimpleNamespace(finished_req_ids=set(), preempted_req_ids={"late"}))
    assert not state.rows
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=[reqs[1]]))  # V2 resubmits the request after preemption.
    for first, masked in [(4, [0, 1, 2, 3]), (10, []), (11, [])]:
        # Re-prefill, verify four positions, then verify again after rejecting the last three.
        runner.prepare_inputs(["late"], np.array([0, 4]), np.array([first]), has_prefill=first < 10)
        assert state.mask[:4].nonzero().flatten().tolist() == masked


def test_compiled_router_observes_request_removal_and_reused_padded_rows():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    router = _router(None)
    runner = _V2Runner(torch.nn.ModuleList([_MoERunner(0, router)]))
    runner.initialize_kv_cache()
    state = runner._forced_routes
    logits = torch.arange(E, dtype=torch.float32).expand(3, -1)
    plain = _router(None).select_experts(logits, logits)
    route = torch.compile(router.select_experts, backend="eager", fullgraph=True)
    assert all(torch.equal(a, b) for a, b in zip(route(logits, logits), plain))

    rows = np.broadcast_to(np.arange(K, dtype=np.uint8), (3, L, K))
    payload = {"start": 0, "dtype": rows.dtype.str, "rows": rows.tobytes()}
    req = SimpleNamespace(req_id="reused", prompt_len=4, sampling_params=SamplingParams(
        extra_args={forced_routes.FORCED_ROUTES_KEY: payload}))
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=[req]))
    runner.prepare_inputs([req.req_id], np.array([0, 3]), np.array([0]))
    assert torch.equal(route(logits, logits)[1], torch.arange(K).expand(3, -1))
    GPUModelRunner.finish_requests(runner, SimpleNamespace(finished_req_ids={req.req_id}, preempted_req_ids=set()))
    assert not state.rows

    req.sampling_params = SamplingParams()
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=[req]))
    runner.prepare_inputs([req.req_id], np.array([0, 1]), np.array([0]))  # the two formerly forced rows now pad
    assert not state.mask[:3].any()
    assert all(torch.equal(a, b) for a, b in zip(route(logits, logits), plain))


class _ToyEngine:
    """A vLLM engine at toy width around real routers: block prefix cache with the trailing-block drop, a chunked
    token-budget scheduler, per-row capture (uint16, as for more than 256 experts), and numerics that differ between
    decode and prefill and with the request's length."""

    def __init__(self, replay_window=0, **parallel):
        gen = torch.Generator().manual_seed(0)
        self.logits = torch.randn(L, 64, E, generator=gen)  # by layer and position
        self.decode_shift = 0.5 * torch.randn(L, 64, E, generator=gen)  # a decoded position routes on other numerics
        bias = 0.1 * torch.randn(E, generator=gen)
        self.routers = [_router(bias) for _ in range(L)]
        self.runner = _V2Runner(torch.nn.ModuleList(_MoERunner(i, r) for i, r in enumerate(self.routers)), **parallel)
        self.runner.initialize_kv_cache()
        self.replay_window = replay_window
        self.captured = [None] * L
        for i, router in enumerate(self.routers):
            router.set_capture_fn(partial(self.captured.__setitem__, i))
        self.cache, self.requests, self.submitted = [], [], 0  # cache: (tokens, routes the KV was built with)
        self.finished = {}  # prompt -> finished request
        self.forced_steps, self.params = [], []

    async def generate_once(self, prompt_input, params, request_id, **_):
        # The request crosses into the engine core as vLLM serializes it.
        params = MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
        self.params.append(params)
        prompt = prompt_input["prompt_token_ids"]
        hit, source = 0, None
        for tokens, routes in (() if params.skip_reading_prefix_cache else self.cache):
            common = next((i for i, (a, b) in enumerate(zip(prompt, tokens[: len(routes)])) if a != b), len(routes))
            cached = max(min(common, len(prompt) - 1) // B - (BLOCKS - 1), 0) * B
            if cached > hit:
                hit, source = cached, routes
        request = SimpleNamespace(id=request_id, params=params, prompt=prompt, hit=hit, done=asyncio.Future(),
                                  used=np.zeros((len(prompt) + params.max_tokens - 1, L, K), np.uint8), out=[])
        request.used[:hit] = source[:hit] if hit else 0
        request.computed = max(0, hit - self.replay_window)
        request.prefill_shift = 0.5 * torch.randn(L, 64, E, generator=torch.Generator().manual_seed(len(prompt)))
        request.core = Request(request_id, prompt, params, None)
        request.core.num_computed_tokens = request.computed
        self.requests.append(request)
        self.submitted += 1
        return await request.done

    async def run(self, expected):
        while self.submitted < expected:
            await asyncio.sleep(0)
        added = set()
        while self.requests:
            batch, budget = [], STEP_TOKENS
            for r in self.requests:
                n = min(len(r.prompt) - r.computed, budget) if r.computed < len(r.prompt) else min(1, budget)
                if n > 0:
                    batch.append((r, n))
                    budget -= n
            admissions = [NewRequestData.from_request(r.core, (), r.prompt) for r, _ in batch if r.id not in added]
            self.runner.add_requests(SimpleNamespace(scheduled_new_reqs=admissions))
            added.update(r.id for r, _ in batch)
            self._step(batch)

    def _step(self, batch):
        counts = [n for _, n in batch]
        qsl = np.concatenate([[0], np.cumsum(counts)])
        num_rows = int(qsl[-1]) + PAD
        self.runner.prepare_inputs([r.id for r, _ in batch], qsl, np.array([r.computed for r, _ in batch]))
        forced = self.runner._forced_routes
        mask = forced.mask[:num_rows] if forced is not None else torch.zeros(num_rows, dtype=torch.bool)
        for layer in range(L):
            logits = torch.zeros(num_rows, E)
            for (r, n), at in zip(batch, qsl):
                positions = torch.arange(r.computed, r.computed + n)
                logits[at : at + n] = self.logits[layer, positions]
                shift = self.decode_shift if r.computed >= len(r.prompt) else r.prefill_shift
                logits[at : at + n] += shift[layer, positions]
            hidden = torch.zeros(num_rows, D)
            _, ids = self.routers[layer].select_experts(hidden, logits)
            assert torch.equal(self.captured[layer], ids)
        self.forced_steps.append({})
        captured = torch.stack(self.captured, 1).numpy()
        for (r, n), at in zip(batch, qsl):
            rows = slice(int(at), int(at) + n)
            r.used[r.computed : r.computed + n] = captured[rows]
            forced = mask[rows].nonzero().flatten() + r.computed
            if len(forced):
                self.forced_steps[-1][r.id] = forced.tolist()
            r.computed += n
            if r.computed >= len(r.prompt):
                r.out.append(1000 + len(r.prompt) + len(r.out))
            if len(r.out) == r.params.max_tokens:
                self._finish(r)

    def _finish(self, r):
        self.requests.remove(r)
        self.runner._remove_request(r.id)
        self.cache.append((r.prompt + r.out, r.used))
        self.finished[tuple(r.prompt)] = r
        choice = SimpleNamespace(text="", token_ids=r.out, finish_reason="length", logprobs=None)
        returned = r.used[r.params.routed_experts_prompt_start :].astype(np.uint16)  # the engine skips the sent prefix
        r.done.set_result(SimpleNamespace(outputs=[choice], prompt_token_ids=r.prompt, num_cached_tokens=r.hit,
                                          prompt_logprobs=None, routed_experts=returned))


def _two_turns(engine, between=None, skip_reads=False):
    """Run two continuations and one independent request through InferenceWorker.generate."""
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state, worker._router_replay_tx, worker._replica_label = WorkerLifecycleState.READY, cache, "replica-a"
    forced_routes.ensure_forced_routes_patch()

    async def collective_rpc(method):
        return [getattr(forced_routes, method)(SimpleNamespace(model_runner=engine.runner))]

    async def reset_prefix_cache():
        engine.cache.clear()
        return True

    worker.llm = SimpleNamespace(collective_rpc=collective_rpc, reset_prefix_cache=reset_prefix_cache)
    asyncio.run(worker._init_forced_routes())
    worker._generate_once = engine.generate_once
    prompts = {"a1": [1, 2, 3, 4, 5, 6], "b1": list(range(20, 29)), "c": list(range(40, 47))}

    def request(name, max_tokens, prefix=None):
        extra_args = {} if prefix is None else {"dss_router_replay_prefix_id": f"rr1:{prefix}"}
        params = dict(max_tokens=max_tokens, dss_sample_id=name, dss_router_replay_id=f"rr1:{name}",
                      extra_args=extra_args)
        if skip_reads and prefix is not None:
            params["prompt_logprobs"] = 0
        return worker.generate(prompts[name], params)

    async def turns():
        a1, b1, _ = await asyncio.gather(request("a1", 5), request("b1", 4), engine.run(2))
        if between is not None:
            await between(worker)
        prompts["a2"] = prompts["a1"] + a1["token_ids"] + [50, 51]
        prompts["b2"] = prompts["b1"] + b1["token_ids"] + [60]
        return await asyncio.gather(request("a2", 3, "a1"), request("b2", 3, "b1"), request("c", 3), engine.run(5))

    a2, b2, c, _ = asyncio.run(turns())
    returned = {name: cache.get(f"rr1:{name}").numpy() for name in prompts if f"rr1:{name}" in cache}
    markers = {name: result["router_replay"] for name, result in [("a2", a2), ("b2", b2), ("c", c)]}
    return returned, {name: engine.finished[tuple(prompt)] for name, prompt in prompts.items()}, markers


@pytest.mark.parametrize("mode,window", [("warm", 0), ("warm", 2), ("reset", 0), ("eviction", 0), ("skip", 0),
                                       ("lookahead", 0)])
def test_recomputed_positions_use_the_named_parents_experts(mode, window):
    engine = _ToyEngine(replay_window=window, num_prefill_lookahead_tokens=3 if mode == "lookahead" else 1)

    async def between(worker):
        if mode == "reset":
            await worker.reset_prefix_cache()
        elif mode == "eviction":
            engine.cache.clear()

    returned, ran, markers = _two_turns(engine, between=between, skip_reads=mode == "skip")
    for child, parent, warm_hit in (("a2", "a1", 4), ("b2", "b1", 8)):
        r = ran[child]
        hit = 0 if mode in {"reset", "eviction", "skip"} else warm_hit
        assert r.hit == hit
        positions = [p for step in engine.forced_steps for p in step.get(r.id, [])]
        assert positions == list(range(max(0, hit - window), len(returned[parent])))
        np.testing.assert_array_equal(r.used[:len(returned[parent])], returned[parent])
    assert all(np.array_equal(returned[name], r.used) for name, r in ran.items())
    assert not any(ran["c"].id in step for step in engine.forced_steps)
    if mode == "warm" and window == 0:
        assert any({ran["a2"].id, ran["b2"].id} <= step.keys() for step in engine.forced_steps)
        assert sum(ran["a2"].id in step for step in engine.forced_steps) >= 2


def test_an_unsupported_engine_keeps_its_own_routes_and_splices_the_parent_as_before(caplog):
    engine = _ToyEngine(data_parallel_size=2)
    with caplog.at_level("WARNING"):
        returned, ran, markers = _two_turns(engine)
    assert sum("re-prefilled prefix positions" in r.getMessage() for r in caplog.records) == 1
    assert not any(engine.forced_steps)
    for child, parent in (("a2", "a1"), ("b2", "b1")):
        n = len(returned[parent])
        np.testing.assert_array_equal(returned[child][:n], returned[parent])
        np.testing.assert_array_equal(returned[child][n:], ran[child].used[n:])
    assert np.any(ran["a2"].used[4:10] != returned["a1"][4:10])  # the recomputed tail ran on fresh routes
    assert [markers[name]["prefix_len"] for name in ("a2", "b2", "c")] == [10, 12, 0]


def test_a_continuation_whose_capture_is_gone_keeps_the_routes_it_ran():
    async def drop(worker):  # another replica served the first turns, or the server's cache dropped them
        worker._router_replay_tx.discard(["rr1:a1", "rr1:b1"])

    engine = _ToyEngine()
    returned, ran, markers = _two_turns(engine, between=drop)

    # A missing parent leaves the request tracking only its own accepted rows.
    assert all(p.extra_args[forced_routes.FORCED_ROUTES_KEY] == {} for p in engine.params)
    for name in ("a2", "b2"):
        assert (returned[name] == ran[name].used).all()
        assert markers[name]["prefix_len"] == 0


def test_unsupported_layers_leave_every_router_unhooked():
    def start(routers, monolithic=(), **parallel):
        layers = [_MoERunner(i, r, i in monolithic) for i, r in enumerate(routers)]
        runner = _V2Runner(torch.nn.ModuleList(layers), **parallel)
        runner.initialize_kv_cache()
        assert runner._forced_routes is None and not any("_compute_routing" in vars(r) for r in routers)
        return runner._forced_routes_unsupported

    # Each builds, binds no router (a supported layer 0 included), and says why.
    custom = CustomRoutingRouter(top_k=K, global_num_experts=E, custom_routing_function=None, renormalize=True)
    retry = SimpleNamespace(kv_load_failure_policy="recompute")
    assert "rewind an active request" in start([_router(None)], kv_transfer_config=retry)
    assert "data_parallel_size > 1" in start([_router(None)], data_parallel_size=2)
    assert "layer 1 runs a monolithic MoE kernel" in start([_router(None), _router(None)], monolithic={1})
    assert "layer 1's router CustomRoutingRouter" in start([_router(None), custom])
    shared = FusedTopKBiasRouter(K, E, num_fused_shared_experts=1)
    assert "fused shared experts" in start([_router(None), shared])
    # A dedicated capture source writes logical ids directly without ever calling a MoERunner router.
    source = torch.nn.Module()
    source.layer_id, source.capture_fn = 1, None
    routable = _router(None)
    runner = _V2Runner(torch.nn.ModuleList([_MoERunner(0, routable), source]))
    runner.initialize_kv_cache()
    assert "directly" in runner._forced_routes_unsupported
    assert runner._forced_routes is None and "_compute_routing" not in vars(routable)
