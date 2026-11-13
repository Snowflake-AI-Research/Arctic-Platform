from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

from arctic_platform.inference.vllm import pp_routed_experts as pp
from arctic_platform.testing_utils import torch_assert_equal

NUM_LAYERS = 6
TOPK = 2
MAX_TOKENS = 8


def _capturer(output_dtype=torch.uint8):
    return SimpleNamespace(
        device_buffer=torch.zeros((MAX_TOKENS, NUM_LAYERS, TOPK), dtype=torch.int32),
        output_dtype=output_dtype,
    )


def _routes(num_tokens, high=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, high, (num_tokens, NUM_LAYERS, TOPK), dtype=torch.int32, generator=generator
    )


@pytest.mark.parametrize(("dtype", "high"), [(torch.uint8, 256), (torch.uint16, 1024)])
def test_pack_unpack_round_trips_through_bytes(dtype, high):
    sender, receiver = _capturer(dtype), _capturer(dtype)
    sender.device_buffer[:5] = _routes(5, high)
    receiver.device_buffer[:] = 7

    packed = pp.pack_routes(sender, 3)
    assert packed.dtype == torch.uint8
    assert packed.shape == (3, NUM_LAYERS, TOPK * dtype.itemsize)

    pp.unpack_routes(receiver, packed.clone())
    torch_assert_equal(receiver.device_buffer[:3], sender.device_buffer[:3])
    # Rows past the received count are left alone.
    assert bool((receiver.device_buffer[3:] == 7).all())


def test_unpack_rejects_rows_of_another_shape():
    receiver = _capturer()
    wrong = torch.zeros((2, NUM_LAYERS - 1, TOPK), dtype=torch.uint8)
    with pytest.raises(ValueError, match="do not match the capture buffer"):
        pp.unpack_routes(receiver, wrong)


def _runner_class(layers_by_stage, routes):
    """A stand-in for Model Runner V2: its forward captures only its stage's layers."""

    class Runner:
        def __init__(self, stage, num_stages=len(layers_by_stage)):
            self.layers = layers_by_stage[stage]
            self.use_pp = num_stages > 1
            self.is_last_pp_rank = stage == num_stages - 1
            self.capturer = _capturer()
            self.aux_output_connector = SimpleNamespace(_capturer=self.capturer)
            self.execute_model_state = None
            self.seen_keys = None

        def execute_model(
            self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kwargs
        ):
            if intermediate_tensors is not None:
                self.seen_keys = set(intermediate_tensors.tensors)
            n = scheduler_output.num_tokens
            for layer in self.layers:
                self.capturer.device_buffer[:n, layer] = routes[:n, layer]
            self.execute_model_state = SimpleNamespace(
                input_batch=SimpleNamespace(num_tokens_after_padding=n)
            )
            if self.is_last_pp_rank:
                return None
            return IntermediateTensors({"hidden_states": torch.ones(n, 4)})

    pp.patch_execute_model(Runner)
    return Runner


def _run_pipeline(runners, num_tokens, dummy_run=False):
    tensors = None
    step = SimpleNamespace(num_tokens=num_tokens)
    for runner in runners:
        if tensors is not None:
            # What the next stage receives: every tensor, freshly allocated.
            tensors = IntermediateTensors({k: v.clone() for k, v in tensors.items()})
        tensors = runner.execute_model(step, tensors, dummy_run=dummy_run)
    return tensors


@pytest.mark.parametrize(
    "layers_by_stage",
    [
        [range(0, 3), range(3, 6)],
        # Uneven stages, and a layer without MoE (5) that no stage captures.
        [range(0, 1), range(1, 3), range(3, 5)],
    ],
)
def test_last_stage_holds_every_layer_like_pp1(layers_by_stage):
    routes = _routes(MAX_TOKENS, seed=1)
    num_tokens = 5
    one_stage = _runner_class([range(0, 6)], routes)(0)
    if 5 not in layers_by_stage[-1]:
        one_stage.layers = [layer for stage in layers_by_stage for layer in stage]
    _run_pipeline([one_stage], num_tokens)

    cls = _runner_class(layers_by_stage, routes)
    stages = [cls(i) for i in range(len(layers_by_stage))]
    assert _run_pipeline(stages, num_tokens) is None

    last = stages[-1].capturer.device_buffer
    torch_assert_equal(last[:num_tokens], one_stage.capturer.device_buffer[:num_tokens])
    # The relay key never reaches the original runner.
    assert all(pp.ROUTES_KEY not in stage.seen_keys for stage in stages[1:])


def test_routes_accumulate_in_the_sent_tensors():
    routes = _routes(MAX_TOKENS, seed=2)
    cls = _runner_class([range(0, 2), range(2, 4), range(4, 6)], routes)
    first, middle = cls(0), cls(1)
    step = SimpleNamespace(num_tokens=4)

    sent = first.execute_model(step, None)
    assert set(sent.tensors) == {"hidden_states", pp.ROUTES_KEY}
    sent_rows = sent[pp.ROUTES_KEY].view(torch.uint8).to(torch.int32)
    torch_assert_equal(sent_rows[:, :2], routes[:4, :2])
    assert bool((sent_rows[:, 2:] == 0).all())

    relayed = middle.execute_model(step, sent)
    relayed_rows = relayed[pp.ROUTES_KEY].to(torch.int32)
    torch_assert_equal(relayed_rows[:, :4], routes[:4, :4])
    assert bool((relayed_rows[:, 4:] == 0).all())


def test_pp1_and_dummy_runs_are_untouched():
    routes = _routes(MAX_TOKENS, seed=3)
    single = _runner_class([range(0, 6)], routes)(0)
    single.is_last_pp_rank = False  # would return tensors if it were a stage
    out = single.execute_model(SimpleNamespace(num_tokens=3), None)
    assert set(out.tensors) == {"hidden_states"}

    cls = _runner_class([range(0, 3), range(3, 6)], routes)
    first, last = cls(0), cls(1)
    out = first.execute_model(SimpleNamespace(num_tokens=3), None, dummy_run=True)
    assert set(out.tensors) == {"hidden_states"}
    # A dummy run on a later stage does not read the relay key either.
    poisoned = IntermediateTensors(
        {"hidden_states": torch.ones(3, 4), pp.ROUTES_KEY: torch.full((3, 6, 2), 9, dtype=torch.uint8)}
    )
    last.execute_model(SimpleNamespace(num_tokens=3), poisoned, dummy_run=True)
    assert bool((last.capturer.device_buffer[:3, :3] == 0).all())


def test_runner_without_routed_experts_is_untouched():
    routes = _routes(MAX_TOKENS, seed=4)
    cls = _runner_class([range(0, 3), range(3, 6)], routes)
    first = cls(0)
    first.aux_output_connector = None
    out = first.execute_model(SimpleNamespace(num_tokens=3), None)
    assert set(out.tensors) == {"hidden_states"}


def test_patches_apply_once():
    class Runner:
        def execute_model(self, *args, **kwargs):
            return None

    pp.patch_execute_model(Runner)
    patched = Runner.execute_model
    pp.patch_execute_model(Runner)
    assert Runner.execute_model is patched


def test_routes_are_sent_whole_from_each_tp_rank():
    from vllm.distributed.parallel_state import GroupCoordinator

    class Group:
        _should_use_all_gather = GroupCoordinator._should_use_all_gather

    pp.patch_should_use_all_gather(Group)
    tp4 = SimpleNamespace(world_size=4)
    assert Group()._should_use_all_gather("hidden_states", 16, tp4, None)
    assert not Group()._should_use_all_gather("hidden_states", 15, tp4, None)
    assert not Group()._should_use_all_gather(pp.ROUTES_KEY, 16, tp4, None)


class _Store:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _connector_class():
    class Connector:
        stores = []

        def __init__(self, *, vllm_config, model, kv_cache_config):
            self._store = _Store()
            self._buffer = object()
            self.stores.append(self._store)

    return Connector


@pytest.mark.parametrize(
    ("pp_size", "is_last", "keeps"),
    [(1, True, True), (2, True, True), (2, False, False), (4, False, False)],
)
def test_data_plane_lives_on_the_last_stage_only(pp_size, is_last, keeps):
    cls = _connector_class()
    pp.patch_worker_connector(cls, lambda: is_last)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
    )
    connector = cls(vllm_config=config, model=None, kv_cache_config=None)
    if keeps:
        assert connector._store is not None and connector._buffer is not None
    else:
        assert connector._store is None and connector._buffer is None
    assert cls.stores[0].closed is not keeps


def _aux_config(pp_size=2, dcp=1, enabled=True):
    return SimpleNamespace(
        aux_output_config=SimpleNamespace(enabled=enabled),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(runner_type="generate", is_moe=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


def _vllm_config_class():
    from vllm.config import VllmConfig

    verify = VllmConfig._verify_aux_output_compatibility
    verify = getattr(verify, "_orig", verify)

    class Config:
        _verify_aux_output_compatibility = verify

    return Config


def test_pp_refusal_is_vllms_until_patched():
    cls = _vllm_config_class()
    config = _aux_config(pp_size=2)
    with pytest.raises(ValueError, match="pipeline parallelism"):
        cls._verify_aux_output_compatibility(config)


def test_pp_refusal_is_lifted_and_other_checks_stay():
    cls = _vllm_config_class()
    pp.patch_verify_aux_output(cls)

    config = _aux_config(pp_size=2)
    cls._verify_aux_output_compatibility(config)
    assert config.parallel_config.pipeline_parallel_size == 2

    config = _aux_config(pp_size=2, dcp=2)
    with pytest.raises(ValueError, match="context parallelism"):
        cls._verify_aux_output_compatibility(config)
    assert config.parallel_config.pipeline_parallel_size == 2

    config = _aux_config(pp_size=2)
    config.cache_config.enable_prefix_caching = False
    with pytest.raises(ValueError, match="prefix caching"):
        cls._verify_aux_output_compatibility(config)

    # Disabled auxiliary output and PP1 go straight to vLLM's checks.
    cls._verify_aux_output_compatibility(_aux_config(pp_size=2, enabled=False))
    cls._verify_aux_output_compatibility(_aux_config(pp_size=1))


def test_disabled_by_env(monkeypatch):
    monkeypatch.setattr(pp, "_APPLIED", False)
    monkeypatch.setenv("ARCTIC_PP_ROUTED_EXPERTS", "0")
    from vllm.config import VllmConfig

    before = VllmConfig._verify_aux_output_compatibility
    pp.ensure_pp_routed_experts()
    assert VllmConfig._verify_aux_output_compatibility is before
    assert pp._APPLIED is False


def test_received_rows_are_read_after_the_receive_completes():
    from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

    routes = _routes(MAX_TOKENS, seed=5)
    cls = _runner_class([range(0, 3), range(3, 6)], routes)
    first, last = cls(0), cls(1)
    step = SimpleNamespace(num_tokens=4)
    sent = first.execute_model(step, None)

    # The receive buffer holds garbage until its handle is waited on.
    buffers = {k: torch.full_like(v, 255) for k, v in sent.items()}

    class Handle:
        def wait(self):
            for key, tensor in sent.items():
                buffers[key].copy_(tensor)

    received = AsyncIntermediateTensors(buffers, comm_handles=[Handle()])
    last.execute_model(step, received)
    torch_assert_equal(last.capturer.device_buffer[:4], routes[:4])


def test_a_stage_without_relayed_rows_fails_loudly():
    routes = _routes(MAX_TOKENS, seed=6)
    last = _runner_class([range(0, 3), range(3, 6)], routes)(1)
    unrelayed = IntermediateTensors({"hidden_states": torch.ones(4, 4)})
    with pytest.raises(RuntimeError, match="sent no routed-experts rows"):
        last.execute_model(SimpleNamespace(num_tokens=4), unrelayed)
