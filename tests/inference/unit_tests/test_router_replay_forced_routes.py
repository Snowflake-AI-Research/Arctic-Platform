"""A continuation re-prefills the end of its previous request's capture under the routes that capture stored."""

import asyncio
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.layer import MoERunner
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import CustomRoutingRouter
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
from vllm.sampling_params import SamplingParams
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


def test_masked_rows_take_the_stored_experts_with_the_trainers_replay_weights(monkeypatch):
    histc = torch.histc  # CPU histc takes no integer input; the trainer's per-expert count is not read here
    monkeypatch.setattr(torch, "histc", lambda t, **kw: histc(t.float(), **kw))
    trainer_moe = pytest.importorskip("arctic_platform.model.implementations.moe.layers.moe")
    gen = torch.Generator().manual_seed(0)
    x, gate, bias = torch.randn(6, D, generator=gen), torch.randn(E, D, generator=gen), torch.randn(E, generator=gen)
    logits = F.linear(x, gate)
    stored = torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(3)]).to(torch.int32)

    plain, hooked = _router(bias), _router(bias)
    forced = ForcedRoutes(max_num_tokens=8, num_layers=2, top_k=K, device=torch.device("cpu"))
    forced.rows["req"] = (11, np.stack([np.zeros((3, K), np.uint16), stored.numpy().astype(np.uint16)], axis=1))
    # The request computes positions 10-15 in batch rows 0-5; its stored rows cover positions 11-13 (rows 1-3).
    forced.fill(["req"], np.array([0, 6]), np.array([10]), num_rows=6)
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
        dim=D, num_experts=E, top_k=K, score_func="sigmoid", route_norm=True, route_scale=2.5, fp32_logits=True
    )
    trainer.gate.weight.data.copy_(gate)
    trainer_weights, trainer_ids, _ = trainer(x[1:4], routed_experts=stored.long())
    assert torch.equal(trainer_ids, stored.long())
    assert torch.equal(weights[1:4], trainer_weights)


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

    def __init__(self, model, data_parallel_size=1, prefill_context_parallel_size=1):
        self.model, self.device, self.aux_output_connector = model, torch.device("cpu"), None
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size, prefill_context_parallel_size=prefill_context_parallel_size))

    def initialize_kv_cache(self):
        capturer = SimpleNamespace(device_buffer=torch.zeros(64, L, K), shape_per_token=(L, K))
        self.aux_output_connector = SimpleNamespace(_capturer=capturer)

    def add_requests(self, scheduler_output):
        for new_req_data in scheduler_output.scheduled_new_reqs:
            self._remove_request(new_req_data.req_id)

    def _remove_request(self, req_id):
        return True

    def prepare_inputs(self, input_batch):
        return input_batch


forced_routes.patch_model_runner(_V2Runner)


class _ToyEngine:
    """A vLLM engine at toy width around real routers: block prefix cache with the trailing-block drop, a chunked
    token-budget scheduler, per-row capture (uint16, as for more than 256 experts), and decode numerics that differ
    from prefill's."""

    def __init__(self, **parallel):
        gen = torch.Generator().manual_seed(0)
        self.logits = torch.randn(L, 64, E, generator=gen)  # by layer and position
        self.decode_shift = 0.5 * torch.randn(L, 64, E, generator=gen)  # a decoded position routes on other numerics
        bias = 0.1 * torch.randn(E, generator=gen)
        self.routers, self.plain = [_router(bias) for _ in range(L)], [_router(bias) for _ in range(L)]
        self.runner = _V2Runner(torch.nn.ModuleList(_MoERunner(i, r) for i, r in enumerate(self.routers)), **parallel)
        self.runner.initialize_kv_cache()
        self.captured = [None] * L
        for i, router in enumerate(self.routers):
            router.set_capture_fn(partial(self.captured.__setitem__, i))
        self.cache, self.requests, self.submitted = [], [], 0  # cache: (tokens, routes the KV was built with)
        self.finished = {}  # prompt -> finished request
        self.forced_steps, self.plain_routes, self.unforced_rows_changed, self.params = [], {}, 0, []

    async def generate_once(self, prompt_input, params, request_id, **_):
        # The request crosses into the engine core as vLLM serializes it.
        params = MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
        self.params.append(params)
        prompt = prompt_input["prompt_token_ids"]
        hit, source = 0, None
        for tokens, routes in self.cache:
            common = next((i for i, (a, b) in enumerate(zip(prompt, tokens[: len(routes)])) if a != b), len(routes))
            cached = max(min(common, len(prompt) - 1) // B - (BLOCKS - 1), 0) * B
            if cached > hit:
                hit, source = cached, routes
        request = SimpleNamespace(id=request_id, params=params, prompt=prompt, hit=hit, done=asyncio.Future(),
                                  used=np.zeros((len(prompt) + params.max_tokens - 1, L, K), np.uint8), out=[])
        request.used[:hit] = source[:hit] if hit else 0
        request.computed = hit
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
            self.runner.add_requests(SimpleNamespace(scheduled_new_reqs=[
                SimpleNamespace(req_id=r.id, sampling_params=r.params) for r, _ in batch if r.id not in added]))
            added.update(r.id for r, _ in batch)
            self._step(batch)

    def _step(self, batch):
        counts = [n for _, n in batch]
        qsl = np.concatenate([[0], np.cumsum(counts)])
        num_rows = int(qsl[-1]) + PAD
        self.runner.prepare_inputs(SimpleNamespace(
            req_ids=[r.id for r, _ in batch], query_start_loc_np=qsl,
            num_computed_tokens_np=np.array([r.computed for r, _ in batch]), num_tokens_after_padding=num_rows))
        forced = self.runner._forced_routes
        mask = forced.mask[:num_rows] if forced is not None else torch.zeros(num_rows, dtype=torch.bool)
        plain = []
        for layer in range(L):
            logits = torch.zeros(num_rows, E)
            for (r, n), at in zip(batch, qsl):
                positions = torch.arange(r.computed, r.computed + n)
                logits[at : at + n] = self.logits[layer, positions]
                if r.computed >= len(r.prompt):
                    logits[at : at + n] += self.decode_shift[layer, positions]
            hidden = torch.zeros(num_rows, D)
            weights, ids = self.routers[layer].select_experts(hidden, logits)
            plain_weights, plain_ids = self.plain[layer].select_experts(hidden, logits)
            changed = ~mask & ((ids != plain_ids).any(1) | (weights != plain_weights).any(1))
            self.unforced_rows_changed += int(changed.sum())
            assert torch.equal(self.captured[layer], ids)
            plain.append(plain_ids)
        self.forced_steps.append({})
        for (r, n), at in zip(batch, qsl):
            rows = slice(int(at), int(at) + n)
            r.used[r.computed : r.computed + n] = torch.stack([c[rows] for c in self.captured], 1).numpy()
            for i, pos in enumerate(range(r.computed, r.computed + n)):
                self.plain_routes[(r.id, pos)] = torch.stack([p[int(at) + i] for p in plain]).numpy()
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
        r.done.set_result(SimpleNamespace(outputs=[choice], prompt_token_ids=r.prompt, num_cached_tokens=r.hit,
                                          prompt_logprobs=None, routed_experts=r.used.astype(np.uint16)))


def _two_turns(engine):
    """Two conversations through ``InferenceWorker.generate`` (each continuation names its first turn), plus a request
    without a prefix id beside the continuations. The server learns at start, as from its engine workers, whether
    they can force stored routes."""
    cache = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state, worker._router_replay_tx, worker._replica_label = WorkerLifecycleState.READY, cache, "replica-a"
    spec = SimpleNamespace(use_eagle_block_drop=lambda: BLOCKS == 2)
    cache_config = SimpleNamespace(block_size=B)

    async def collective_rpc(method):  # two TP ranks, each reporting through the worker RPC
        return [getattr(forced_routes, method)(SimpleNamespace(model_runner=engine.runner))] * 2

    worker.llm = SimpleNamespace(vllm_config=SimpleNamespace(cache_config=cache_config, speculative_config=spec),
                                 collective_rpc=collective_rpc)
    asyncio.run(worker._init_forced_routes())
    worker._generate_once = engine.generate_once
    prompts = {"a1": [1, 2, 3, 4, 5, 6], "b1": list(range(20, 29)), "c": list(range(40, 47))}

    def request(name, max_tokens, prefix=None):
        extra_args = {} if prefix is None else {"dss_router_replay_prefix_id": f"rr1:{prefix}"}
        params = dict(max_tokens=max_tokens, dss_sample_id=name, dss_router_replay_id=f"rr1:{name}",
                      extra_args=extra_args)
        return worker.generate(prompts[name], params)

    async def turns():
        a1, b1, _ = await asyncio.gather(request("a1", 5), request("b1", 4), engine.run(2))
        prompts["a2"] = prompts["a1"] + a1["token_ids"] + [50, 51]
        prompts["b2"] = prompts["b1"] + b1["token_ids"] + [60]
        await asyncio.gather(request("a2", 3, "a1"), request("b2", 3, "b1"), request("c", 3), engine.run(5))

    asyncio.run(turns())
    returned = {name: cache.get(f"rr1:{name}").numpy() for name in prompts}
    return returned, {name: engine.finished[tuple(prompt)] for name, prompt in prompts.items()}


def test_a_continuation_recomputes_the_tail_of_its_previous_capture_under_the_stored_routes():
    engine = _ToyEngine()
    returned, ran = _two_turns(engine)

    # a1's capture ends at 10 and a2's hit stops at 4 (the block at 8 dropped), so positions 4-9 are recomputed;
    # b1's ends at 12 and b2's hit stops at 8: positions 8-11.
    tails = {"a2": ("a1", range(4, 10)), "b2": ("b1", range(8, 12))}

    # The engine's own router picks other experts at some of those positions, so this test can fail.
    assert any((engine.plain_routes[(ran[name].id, p)] != returned[prev][p]).any()
               for name, (prev, positions) in tails.items() for p in positions)

    # The forward routed every recomputed stored position as stored.
    tail = [(name, p) for name, (_, positions) in tails.items() for p in positions]
    tail_mismatch = sum((ran[name].used[p] != returned[tails[name][0]][p]).any() for name, p in tail)
    assert tail_mismatch == 0, f"{tail_mismatch} of {len(tail)} recomputed stored positions routed unlike their storage"

    # Every returned capture is the routes the forward used, at every position.
    capture_mismatch = {name: int((returned[name] != ran[name].used).any(axis=(1, 2)).sum()) for name in ran}
    assert capture_mismatch == dict.fromkeys(ran, 0), f"positions returned unlike the forward: {capture_mismatch}"

    # Exactly those positions ran under stored routes; every other row is routed as an unpatched router routes it, the
    # request without a prefix id included.
    forced = {name: sorted(p for step in engine.forced_steps for p in step.get(ran[name].id, [])) for name in tails}
    assert forced == {name: list(positions) for name, (_, positions) in tails.items()}
    assert engine.unforced_rows_changed == 0
    assert all((returned["c"][p] == engine.plain_routes[(ran["c"].id, p)]).all() for p in range(len(returned["c"])))

    # The tails ran in chunks, and in one step together.
    assert any({ran["a2"].id, ran["b2"].id} <= step.keys() for step in engine.forced_steps)
    assert sum(ran["a2"].id in step for step in engine.forced_steps) >= 2


def test_an_engine_stored_rows_cannot_reach_starts_unhooked_and_routes_as_before(caplog):
    def start(routers, monolithic=(), **parallel):
        layers = [_MoERunner(i, r, i in monolithic) for i, r in enumerate(routers)]
        runner = _V2Runner(torch.nn.ModuleList(layers), **parallel)
        runner.initialize_kv_cache()
        assert runner._forced_routes is None and not any("_compute_routing" in vars(r) for r in routers)
        return runner._forced_routes_unsupported

    # Each builds, binds no router (a supported layer 0 included), and says why.
    custom = CustomRoutingRouter(top_k=K, global_num_experts=E, custom_routing_function=None, renormalize=True)
    assert "data_parallel_size > 1" in start([_router(None)], data_parallel_size=2)
    assert "prefill_context_parallel_size > 1" in start([_router(None)], prefill_context_parallel_size=2)
    assert "layer 1 runs a monolithic MoE kernel" in start([_router(None), _router(None)], monolithic={1})
    assert "layer 1's router CustomRoutingRouter" in start([_router(None), custom])

    # Requests with and without a prefix id behave as before the hook: no rows reach the engine, every row routes as
    # an unpatched router, and the splice returns the stored prefix and the engine's own routes after it.
    engine = _ToyEngine(data_parallel_size=2)
    with caplog.at_level("WARNING"):
        returned, ran = _two_turns(engine)
    warnings = [r.getMessage() for r in caplog.records if "re-prefilled prefix positions" in r.getMessage()]
    assert len(warnings) == 1 and "data_parallel_size > 1" in warnings[0]
    assert not any(forced_routes.FORCED_ROUTES_KEY in (p.extra_args or {}) for p in engine.params)
    assert not any(engine.forced_steps) and engine.unforced_rows_changed == 0
    for name, prev in (("a2", "a1"), ("b2", "b1")):
        n = len(returned[prev])
        assert (returned[name][:n] == returned[prev]).all() and (returned[name][n:] == ran[name].used[n:]).all()
    assert (ran["a2"].used[4:10] != returned["a1"][4:10]).any()  # the tail still ran on the engine's own routes
