from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

if "ray" not in sys.modules:
    try:
        __import__("ray")
    except ModuleNotFoundError:
        ray_module = types.ModuleType("ray")
        ray_module.actor = SimpleNamespace(ActorHandle=object)

        def remote(*args, **kwargs):
            if args and len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def decorator(obj):
                return obj

            return decorator

        ray_module.remote = remote
        ray_module.init = lambda *args, **kwargs: None
        ray_module.nodes = lambda: []
        ray_module.get = lambda *args, **kwargs: None
        ray_module.kill = lambda *args, **kwargs: None
        sys.modules["ray"] = ray_module

if "vllm" not in sys.modules:
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_module.__version__ = "0.32.0"
    config_module = types.ModuleType("vllm.config")
    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
    loggers_module = types.ModuleType("vllm.v1.metrics.loggers")
    stats_module = types.ModuleType("vllm.v1.metrics.stats")

    class VllmConfig:
        pass

    class StatLoggerBase:
        pass

    class IterationStats:
        pass

    class SchedulerStats:
        pass

    def check_stop(*args, **kwargs):
        return False

    check_stop._arctic_router_replay_patch = True

    config_module.VllmConfig = VllmConfig
    scheduler_module.check_stop = check_stop
    loggers_module.StatLoggerBase = StatLoggerBase
    stats_module.IterationStats = IterationStats
    stats_module.SchedulerStats = SchedulerStats

    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.config"] = config_module
    sys.modules["vllm.v1"] = types.ModuleType("vllm.v1")
    sys.modules["vllm.v1.core"] = types.ModuleType("vllm.v1.core")
    sys.modules["vllm.v1.core.sched"] = types.ModuleType("vllm.v1.core.sched")
    sys.modules["vllm.v1.core.sched.scheduler"] = scheduler_module
    sys.modules["vllm.v1.metrics"] = types.ModuleType("vllm.v1.metrics")
    sys.modules["vllm.v1.metrics.loggers"] = loggers_module
    sys.modules["vllm.v1.metrics.stats"] = stats_module
    xgrammar_mod = types.ModuleType("vllm.v1.structured_output.backend_xgrammar")

    class XgrammarGrammar:
        def fill_bitmask(self, bitmask, idx):
            return None

    XgrammarGrammar.fill_bitmask._arctic_stop_mask_fix = True
    xgrammar_mod.XgrammarGrammar = XgrammarGrammar
    structured_output = types.ModuleType("vllm.v1.structured_output")
    structured_output.__path__ = []
    sys.modules["vllm.v1.structured_output"] = structured_output
    sys.modules["vllm.v1.structured_output.backend_xgrammar"] = xgrammar_mod
    for name in ("vllm.v1", "vllm.v1.core", "vllm.v1.core.sched", "vllm.v1.metrics"):
        sys.modules[name].__path__ = []

from arctic_platform.inference.server.weight_sync.receiver import (
    WeightSyncExtension,
    _loaded_destination_names,
    _model_parameter_l2,
    _tensor_l2_sq,
)


def test_direct_zero_copy_keeps_non_contiguous_parameter_views(monkeypatch):
    base = torch.zeros((4, 4))
    view = base[:, 1]
    assert not view.is_contiguous()

    class FakeWriter:
        def __init__(self, model, device) -> None:
            pass

        def all_keys(self):
            return ["weight"]

        def get_view(self, name):
            assert name == "weight"
            return view

    class FakeEngine:
        def __init__(self) -> None:
            self.param_views = None

        def receive_weights_direct(self, param_views):
            self.param_views = param_views
            return {"params_loaded": 1}

    import arctic_platform.inference.server.weight_sync.utils as utils

    monkeypatch.setattr(utils, "_DirectParamWriter", FakeWriter)

    fake_engine = FakeEngine()
    loaded = WeightSyncExtension._load_direct_zero_copy(
        SimpleNamespace(device=torch.device("cpu")),
        model=object(),
        engine=fake_engine,
    )

    assert loaded == 1
    assert fake_engine.param_views["weight"] is view


def test_direct_zero_copy_rejects_orphan_sources(monkeypatch):
    class FakeWriter:
        def __init__(self, model, device) -> None:
            pass

        def all_keys(self):
            return ["weight"]

        def get_view(self, name):
            return torch.zeros(1)

    class FakeEngine:
        def receive_weights_direct(self, param_views):
            return {"params_loaded": 1, "orphan": 1}

    import arctic_platform.inference.server.weight_sync.utils as utils

    monkeypatch.setattr(utils, "_DirectParamWriter", FakeWriter)

    with pytest.raises(
        AssertionError,
        match=r"_load_direct_zero_copy.*1/1 tensor",
    ):
        WeightSyncExtension._load_direct_zero_copy(
            SimpleNamespace(device=torch.device("cpu")),
            model=object(),
            engine=FakeEngine(),
        )


def test_direct_param_writer_maps_checkpoint_wrapped_layer_keys(monkeypatch):
    class FakeQKVParallelLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 0
            self.output_sizes = [2, 2, 2]
            self.tp_size = 1
            self.weight = nn.Parameter(torch.arange(18, dtype=torch.float32).reshape(6, 3))

    class FakeMergedColumnParallelLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 0
            self.output_sizes = [2, 4]
            self.tp_size = 1
            self.weight = nn.Parameter(torch.arange(18, dtype=torch.float32).reshape(6, 3))

    linear_module = types.ModuleType("vllm.model_executor.layers.linear")
    linear_module.QKVParallelLinear = FakeQKVParallelLinear
    linear_module.MergedColumnParallelLinear = FakeMergedColumnParallelLinear
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor", types.ModuleType("vllm.model_executor"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_module)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.qkv_proj = FakeQKVParallelLinear()
            self.mlp = nn.Module()
            self.mlp.gate_up_proj = FakeMergedColumnParallelLinear()
            self.mlp.experts = nn.Module()
            self.mlp.experts.routed_experts = nn.Module()
            self.mlp.experts.routed_experts.w13_weight = nn.Parameter(
                torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
            )
            self.mlp.experts.routed_experts.w2_weight = nn.Parameter(
                torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
            )
            self.input_layernorm = nn.LayerNorm(3)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    from arctic_platform.inference.server.weight_sync.utils import _DirectParamWriter

    writer = _DirectParamWriter(Model(), torch.device("cpu"))

    unwrapped_q = writer.get_view("model.layers.0.self_attn.q_proj.weight")
    wrapped_q = writer.get_view(
        "model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.weight"
    )
    unwrapped_gate = writer.get_view("model.layers.0.mlp.gate_proj.weight")
    wrapped_gate = writer.get_view(
        "model.layers.0._checkpoint_wrapped_module.mlp.gate_proj.weight"
    )
    unwrapped_w13 = writer.get_view(
        "model.layers.0.mlp.experts.w13_weight"
    )
    wrapped_w13 = writer.get_view(
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w13_weight"
    )
    unwrapped_w2 = writer.get_view(
        "model.layers.0.mlp.experts.w2_weight"
    )
    wrapped_w2 = writer.get_view(
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w2_weight"
    )
    unwrapped_ln = writer.get_view("model.layers.0.input_layernorm.weight")
    wrapped_ln = writer.get_view(
        "model.layers.0._checkpoint_wrapped_module.input_layernorm.weight"
    )

    assert wrapped_q is unwrapped_q
    assert wrapped_gate is unwrapped_gate
    assert wrapped_w13 is unwrapped_w13
    assert wrapped_w2 is unwrapped_w2
    assert wrapped_ln is unwrapped_ln
    assert unwrapped_q is not None
    assert unwrapped_gate is not None
    assert unwrapped_w13 is not None
    assert unwrapped_w2 is not None
    assert unwrapped_ln is not None
    assert tuple(unwrapped_q.shape) == (2, 3)
    assert tuple(unwrapped_gate.shape) == (2, 3)
    assert tuple(unwrapped_w13.shape) == (2, 3, 3)
    assert tuple(unwrapped_w2.shape) == (2, 3, 3)
    assert tuple(unwrapped_ln.shape) == (3,)

    assert writer.get_destination(
        "model.layers.0._checkpoint_wrapped_module.self_attn.q_proj.weight"
    ) == "model.layers.0.self_attn.qkv_proj.weight"
    assert writer.get_destination(
        "model.layers.0._checkpoint_wrapped_module.mlp.gate_proj.weight"
    ) == "model.layers.0.mlp.gate_up_proj.weight"
    assert writer.get_destination(
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w13_weight"
    ) == "model.layers.0.mlp.experts.routed_experts.w13_weight"
    assert writer.get_destination(
        "model.layers.0._checkpoint_wrapped_module.input_layernorm.weight"
    ) == "model.layers.0.input_layernorm.weight"


class _FakeEngine:
    def __init__(self, items):
        self._items = items

    def receive_weights(self):
        yield from self._items


def test_load_direct_tracks_canonical_destinations_and_fusion(monkeypatch):
    fused = torch.zeros(4)

    class FakeWriter:
        def __init__(self, model, device) -> None:
            self._views = {
                "q.weight": fused[:2],
                "k.weight": fused[2:],
            }

        def get_view(self, name):
            return self._views.get(name)

        def get_destination(self, name):
            return "qkv.weight" if name in self._views else None

    import arctic_platform.inference.server.weight_sync.utils as utils

    monkeypatch.setattr(utils, "_DirectParamWriter", FakeWriter)
    (
        loaded,
        recv_l2_sq,
        applied_source_l2_sq,
        mappings,
        skipped,
    ) = WeightSyncExtension._load_direct(
        SimpleNamespace(device=torch.device("cpu")),
        model=object(),
        engine=_FakeEngine(
            [
                ("q.weight", torch.tensor([1.0, 2.0])),
                ("k.weight", torch.tensor([3.0, 4.0])),
            ]
        ),
    )

    assert loaded == 2
    assert recv_l2_sq == 30.0
    assert applied_source_l2_sq == 30.0
    assert skipped == []
    assert mappings == {
        "q.weight": ["qkv.weight"],
        "k.weight": ["qkv.weight"],
    }
    assert _loaded_destination_names(mappings) == {"qkv.weight"}
    assert torch.equal(fused, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_load_batched_tracks_many_sources_to_one_destination():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fused_weight = nn.Parameter(torch.zeros(4))

        def load_weights(self, weights):
            for name, tensor in weights:
                if name == "part_a":
                    self.fused_weight.data[:2].copy_(tensor)
                elif name == "part_b":
                    self.fused_weight.data[2:].copy_(tensor)
            return {"fused_weight"}

    model = Model()
    (
        loaded,
        recv_l2_sq,
        applied_source_l2_sq,
        mappings,
        skipped,
    ) = WeightSyncExtension._load_batched(
        SimpleNamespace(device=torch.device("cpu")),
        model,
        _FakeEngine(
            [
                ("part_a", torch.tensor([1.0, 4.0])),
                ("part_b", torch.tensor([5.0, 0.0])),
            ]
        ),
    )

    assert loaded == 2
    assert recv_l2_sq == 42.0
    assert applied_source_l2_sq == 42.0
    assert skipped == []
    assert mappings == {
        "part_a": ["fused_weight"],
        "part_b": ["fused_weight"],
    }
    full, loaded_subset, param_l2, loaded_param_l2 = _model_parameter_l2(
        model,
        torch.device("cpu"),
        _loaded_destination_names(mappings),
        collect=True,
    )
    assert full == loaded_subset == 42.0
    assert param_l2 == {"fused_weight": 42.0}
    assert loaded_param_l2 == {"fused_weight": 42.0}


@pytest.mark.parametrize("buffer_dtype", [torch.int64, torch.float32])
def test_load_batched_buffer_bytes_do_not_enter_parameter_norms(buffer_dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))
            self.register_buffer("lookup", torch.zeros(2, dtype=buffer_dtype))

        def load_weights(self, weights):
            destinations = set()
            for name, tensor in weights:
                getattr(self, name).data.copy_(tensor)
                destinations.add(name)
            return destinations

    model = Model()
    source_weight = torch.tensor([3.0, 4.0])
    source_buffer = torch.tensor([100, 200], dtype=buffer_dtype)
    loaded, received, applied, mappings, skipped = WeightSyncExtension._load_batched(
        SimpleNamespace(device=torch.device("cpu")),
        model,
        _FakeEngine([("weight", source_weight), ("lookup", source_buffer)]),
    )
    assert loaded == 2
    assert received == 50025.0
    assert applied == 25.0
    assert mappings == {"weight": ["weight"], "lookup": ["lookup"]}
    assert skipped == []
    assert torch.equal(model.lookup, source_buffer)
    full, subset, parameters, loaded_parameters = _model_parameter_l2(
        model, torch.device("cpu"), _loaded_destination_names(mappings), collect=True
    )
    assert full == subset == applied
    assert parameters == loaded_parameters == {"weight": 25.0}


@pytest.mark.parametrize("orphan", [False, True])
def test_load_direct_uses_multimodal_loader_and_rejects_orphans(monkeypatch, orphan):
    import arctic_platform.inference.server.weight_sync.utils as utils

    def reject_direct_writer(*args):
        pytest.fail("multimodal weights require the model's authoritative loader")

    monkeypatch.setattr(utils, "_DirectParamWriter", reject_direct_writer)

    class Model(nn.Module):
        hf_to_vllm_mapper = object()

        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(2, 1, bias=False)
            self.fused_weight = nn.Parameter(torch.zeros(4))
            self.calls = []

        def load_weights(self, weights):
            destinations = set()
            for name, tensor in weights:
                self.calls.append(name)
                if name == "model.visual.weight":
                    self.visual.weight.data.copy_(tensor)
                    destinations.add("visual.weight")
                elif name == "model.language_model.part_a":
                    self.fused_weight.data[:2].copy_(tensor)
                    destinations.add("fused_weight")
                elif name == "model.language_model.part_b":
                    self.fused_weight.data[2:].copy_(tensor)
                    destinations.add("fused_weight")
            return destinations

    model = Model()
    receiver = WeightSyncExtension()
    receiver.device = torch.device("cpu")
    weights = [
        ("model.visual.weight", torch.tensor([[2.0, 3.0]])),
        ("model.language_model.part_a", torch.tensor([1.0, 4.0])),
        ("model.language_model.part_b", torch.tensor([5.0, 0.0])),
    ]
    if orphan:
        weights.append(("unknown.weight", torch.ones(1)))
        with pytest.raises(AssertionError, match="unknown.weight"):
            receiver._load_direct(model, _FakeEngine(weights))
    else:
        loaded, received_l2, applied_l2, mappings, skipped = receiver._load_direct(model, _FakeEngine(weights))
        assert loaded == 3
        assert received_l2 == applied_l2 == 55.0
        assert skipped == []
        assert mappings == {
            "model.visual.weight": ["visual.weight"],
            "model.language_model.part_a": ["fused_weight"],
            "model.language_model.part_b": ["fused_weight"],
        }
        assert torch.equal(model.visual.weight, torch.tensor([[2.0, 3.0]]))
        assert torch.equal(model.fused_weight, torch.tensor([1.0, 4.0, 5.0, 0.0]))
    assert model.calls == [name for name, _ in weights]


def test_model_parameter_l2_excludes_receiver_only_composite_parameters():
    class CompositeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Parameter(torch.tensor([1.0, 4.0, 5.0]))
            self.vision_tower = nn.Parameter(torch.tensor([3.0, 7.0]))

    full, loaded_subset, param_l2, loaded_param_l2 = _model_parameter_l2(
        CompositeModel(),
        torch.device("cpu"),
        {"language_model"},
        collect=True,
    )

    assert full == 100.0
    assert loaded_subset == 42.0
    assert param_l2 == {
        "language_model": 42.0,
        "vision_tower": 58.0,
    }
    assert loaded_param_l2 == {"language_model": 42.0}


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("parameter_loaded", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_model_parameter_l2_accepts_registered_buffers_without_counting_them(collect, parameter_loaded, persistent):
    model = nn.Module()
    model.weight = nn.Parameter(torch.tensor([3.0, 4.0]))
    model.ple = nn.Module()
    model.ple.register_buffer(
        "layer_multipliers",
        torch.tensor([100, 200], dtype=torch.int64),
        persistent=persistent,
    )
    destinations = {"ple.layer_multipliers"}
    if parameter_loaded:
        destinations.add("weight")

    full, loaded_subset, param_l2, loaded_param_l2 = _model_parameter_l2(
        model, torch.device("cpu"), destinations, collect=collect
    )

    assert full == 25.0
    assert loaded_subset == (25.0 if parameter_loaded else 0.0)
    assert param_l2 == ({"weight": 25.0} if collect else None)
    assert loaded_param_l2 == (({"weight": 25.0} if parameter_loaded else {}) if collect else None)


def test_model_parameter_l2_still_rejects_unknown_names_with_registered_buffers():
    model = nn.Module()
    model.weight = nn.Parameter(torch.tensor([3.0, 4.0]))
    model.register_buffer("layer_multipliers", torch.tensor([100], dtype=torch.int64))

    with pytest.raises(RuntimeError, match="unknown.weight"):
        _model_parameter_l2(
            model,
            torch.device("cpu"),
            {"weight", "layer_multipliers", "unknown.weight"},
        )


def _expected_tensor_l2_sq(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().pow(2).sum().double()


def test_tensor_l2_sq_accepts_noncontiguous_tensor():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not tensor.is_contiguous()
    torch.testing.assert_close(
        _tensor_l2_sq(tensor, chunk_numel=5),
        _expected_tensor_l2_sq(tensor),
    )


@pytest.mark.parametrize(
    "tensor,chunk_numel",
    [
        (torch.empty(0), 1 << 20),
        (torch.tensor(3.0), 1),
        (torch.tensor([1.5], dtype=torch.float32), 1),
        (torch.arange(8, dtype=torch.float32), 4),
        (torch.arange(10, dtype=torch.float32), 3),
        (torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t(), 5),
    ],
)
def test_tensor_l2_sq_matches_single_shot_across_chunk_boundaries(tensor, chunk_numel):
    torch.testing.assert_close(
        _tensor_l2_sq(tensor, chunk_numel=chunk_numel),
        _expected_tensor_l2_sq(tensor),
    )


def test_load_batched_counts_empty_loader_result_as_rank_skip(caplog):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def load_weights(self, weights):
            name, tensor = next(iter(weights))
            if name == "weight":
                self.weight.data.copy_(tensor)
                return {"weight"}
            return set()

    result = WeightSyncExtension._load_batched(
        SimpleNamespace(
            device=torch.device("cpu"),
            parallel_config=SimpleNamespace(enable_expert_parallel=True, pipeline_parallel_size=1),
        ),
        Model(),
        _FakeEngine(
            [
                ("weight", torch.tensor([2.0])),
                ("non_local.weight", torch.tensor([3.0])),
            ]
        )
    )

    loaded, recv_l2_sq, applied_l2_sq, mappings, skipped = result
    assert loaded == 2
    assert recv_l2_sq == 13.0
    assert applied_l2_sq == 4.0
    assert mappings == {"weight": ["weight"]}
    assert skipped == ["non_local.weight"]
    assert [record for record in caplog.records if record.levelname == "WARNING"] == [
        caplog.records[-1]
    ]
    assert "skipped 1/2" in caplog.records[-1].getMessage()


class _PlacesOnlyWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def load_weights(self, weights):
        name, tensor = next(iter(weights))
        if name == "weight":
            self.weight.data.copy_(tensor)
            return {"weight"}
        return set()


@pytest.mark.parametrize(
    "parallel_config", [None, SimpleNamespace(enable_expert_parallel=False, pipeline_parallel_size=1)]
)
def test_load_batched_rejects_sources_without_destination_under_tensor_parallelism(parallel_config):
    receiver = SimpleNamespace(device=torch.device("cpu"), parallel_config=parallel_config)
    engine = _FakeEngine([("weight", torch.tensor([2.0])), ("model.layers.0.extra.weight", torch.tensor([3.0]))])

    with pytest.raises(AssertionError, match="dropped on the floor") as raised:
        WeightSyncExtension._load_batched(receiver, _PlacesOnlyWeight(), engine)

    assert "received 1/2" in str(raised.value)
    assert "model.layers.0.extra.weight" in str(raised.value)
    assert "LoRA" not in str(raised.value)


def test_load_batched_names_lora_route_for_unplaced_adapter_factors():
    receiver = SimpleNamespace(device=torch.device("cpu"))
    adapters = [
        "model.layers.0.self_attn.q_a_proj.lora_A.default.weight",
        "model.layers.0.self_attn.q_a_proj.lora_B.default.weight",
    ]
    engine = _FakeEngine([("weight", torch.tensor([2.0]))] + [(name, torch.ones(1)) for name in adapters])

    with pytest.raises(AssertionError, match="2 of them are LoRA adapter tensors") as raised:
        WeightSyncExtension._load_batched(receiver, _PlacesOnlyWeight(), engine)

    assert "weight_format='lora'" in str(raised.value)
    assert all(name in str(raised.value) for name in adapters)


@pytest.mark.parametrize(
    ("enable_expert_parallel", "pipeline_parallel_size"), [(True, 1), (False, 2)], ids=["expert", "pipeline"]
)
def test_load_batched_returns_rank_skips_when_the_rank_holds_part_of_the_model(
    enable_expert_parallel, pipeline_parallel_size
):
    receiver = SimpleNamespace(
        device=torch.device("cpu"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=enable_expert_parallel, pipeline_parallel_size=pipeline_parallel_size
        ),
    )
    engine = _FakeEngine([("weight", torch.tensor([2.0])), ("other_rank.weight", torch.tensor([3.0]))])

    *_, skipped = WeightSyncExtension._load_batched(receiver, _PlacesOnlyWeight(), engine)

    assert skipped == ["other_rank.weight"]


def test_hf_sync_of_declared_export_buffers_passes_loaded_destination_check(monkeypatch):
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights

    def ngram_model():
        # Qwen3.8 shape: a trained parameter beside constant int64 hash tables that
        # Qwen4ExpTextNGramEmbeddingPrimeRL declares in _dss_weight_export_buffers.
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module()])
        ngram = model.model.layers[0].ngram_embedding = nn.Module()
        ngram._dss_weight_export_buffers = frozenset({"layer_multipliers"})
        ngram.weight = nn.Parameter(torch.zeros(2))
        ngram.register_buffer("layer_multipliers", torch.tensor([23703573157769, 8052911324071]))
        return model

    class Trainer(nn.Module):
        @staticmethod
        def convert_layer_to_hf(layer_state, layer_index):
            return layer_state

    class Sampler(nn.Module):
        def load_weights(self, weights):
            # vLLM's loaders also place persistent buffers and report them as loaded.
            state = self.state_dict(keep_vars=True)
            for name, tensor in weights:
                state[name].data.copy_(tensor)
            return {name for name, _ in weights}

    trainer, sampler = Trainer(), Sampler()
    trainer.model = ngram_model().model
    sampler.model = ngram_model().model
    trainer.model.layers[0].ngram_embedding.weight.data.copy_(torch.tensor([3.0, 4.0]))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    loaded, recv_l2_sq, applied_l2_sq, mappings, skipped = WeightSyncExtension._load_batched(
        SimpleNamespace(device=torch.device("cpu")),
        sampler,
        _FakeEngine(list(build_iter_full_hf_weights(trainer)())),
    )
    _, loaded_l2_sq, _, _ = _model_parameter_l2(
        sampler, torch.device("cpu"), _loaded_destination_names(mappings)
    )

    assert torch.equal(sampler.model.layers[0].ngram_embedding.weight, torch.tensor([3.0, 4.0]))
    assert recv_l2_sq == applied_l2_sq == loaded_l2_sq == 25.0
    assert skipped == []


def test_load_direct_still_rejects_orphan_sources(monkeypatch):
    class FakeWriter:
        def __init__(self, model, device) -> None:
            pass

        def get_view(self, name):
            return None

        def get_destination(self, name):
            return None

    import arctic_platform.inference.server.weight_sync.utils as utils

    monkeypatch.setattr(utils, "_DirectParamWriter", FakeWriter)

    with pytest.raises(AssertionError, match="dropped on the floor"):
        WeightSyncExtension._load_direct(
            SimpleNamespace(device=torch.device("cpu")),
            object(),
            _FakeEngine([("missing.weight", torch.ones(1))]),
        )


def test_load_fp8_tracks_all_received_bytes(monkeypatch):
    class FakeUpdater:
        pending = 0
        unplaced = ()

        def __init__(self, model, dtype, device) -> None:
            self.names = []

        def feed(self, name, tensor):
            self.names.append(name)

    import arctic_platform.inference.server.weight_sync.utils as utils

    monkeypatch.setattr(utils, "_FP8InplaceUpdater", FakeUpdater)
    loaded, recv_l2_sq = WeightSyncExtension._load_fp8(
        SimpleNamespace(
            device=torch.device("cpu"),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        object(),
        _FakeEngine(
            [
                ("first.weight", torch.tensor([1.0, 2.0])),
                ("second.weight", torch.tensor([3.0])),
            ]
        ),
    )

    assert loaded == 2
    assert recv_l2_sq == 14.0


def test_fp8_updater_uses_native_weight_loader_for_tp_sharding(monkeypatch):
    from arctic_platform.inference.server.weight_sync.utils import _FP8InplaceUpdater

    linear_module = types.ModuleType("vllm.model_executor.layers.linear")

    class RowParallelLinear:
        tp_rank = 3
        tp_size = 4

        def weight_loader(self, param, loaded):
            param.data.copy_(loaded.narrow(1, 6, 2))

    linear_module.RowParallelLinear = RowParallelLinear
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.linear", linear_module
    )

    target = nn.Parameter(
        torch.zeros((4, 2), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    target_scale = nn.Parameter(torch.zeros((2, 1)), requires_grad=False)
    module = RowParallelLinear()
    updater = _FP8InplaceUpdater.__new__(_FP8InplaceUpdater)
    updater._params = {
        "layer.down_proj.weight": target,
        "layer.down_proj.weight_scale_inv": target_scale,
    }
    updater._fp8_modules = {"layer.down_proj": module}
    updater._bufs = {}
    updater._shards_left = {}
    updater._dequant = {}

    full = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(
        torch.float8_e4m3fn
    )
    updater.feed("layer.down_proj.weight", full)
    full_scale = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    updater.feed("layer.down_proj.weight_scale_inv", full_scale)

    torch.testing.assert_close(target.data.float(), full[:, 6:8].float())
    torch.testing.assert_close(target_scale, full_scale[:, 3:4])
    assert updater.pending == 0


def _install_fp8_vllm_stubs(monkeypatch):
    class _Linear(nn.Module):
        def __init__(self, weight, *, tp_rank=0, tp_size=1, output_partition_sizes=None):
            super().__init__()
            self.weight = weight
            self.tp_rank = tp_rank
            self.tp_size = tp_size
            if output_partition_sizes is not None:
                self.output_partition_sizes = output_partition_sizes

    class ColumnParallelLinear(_Linear):
        pass

    class MergedColumnParallelLinear(ColumnParallelLinear):
        def weight_loader(self, param, loaded, shard_id=None):
            rows = loaded.shape[0]
            param.data.narrow(0, shard_id * rows, rows).copy_(loaded)

    class QKVParallelLinear(ColumnParallelLinear):
        pass

    class RowParallelLinear(_Linear):
        pass

    class BasevLLMParameter(nn.Parameter):
        pass

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.ColumnParallelLinear = ColumnParallelLinear
    linear.MergedColumnParallelLinear = MergedColumnParallelLinear
    linear.QKVParallelLinear = QKVParallelLinear
    linear.RowParallelLinear = RowParallelLinear
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.BasevLLMParameter = BasevLLMParameter
    custom_ops = types.ModuleType("vllm._custom_ops")
    custom_ops.scaled_fp8_quant = lambda tensor, scale=None: (
        tensor.to(torch.float8_e4m3fn),
        torch.tensor([2.0]),
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.parameter", parameter)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", custom_ops)
    return linear


def _frozen(tensor):
    return nn.Parameter(tensor, requires_grad=False)


@pytest.mark.parametrize("layer", ["row", "column", "merged"])
def test_fp8_updater_shards_scales_from_the_full_tensor(monkeypatch, layer):
    from arctic_platform.inference.server.weight_sync.utils import _FP8InplaceUpdater

    linear = _install_fp8_vllm_stubs(monkeypatch)
    full_scale = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    weight = _frozen(torch.zeros((2, 2), dtype=torch.float8_e4m3fn))
    if layer == "row":
        module = linear.RowParallelLinear(weight, tp_rank=1, tp_size=2)
        expected = full_scale[:, 3:6]
    elif layer == "column":
        module = linear.ColumnParallelLinear(weight, tp_rank=1, tp_size=2)
        expected = full_scale[4:8]
    else:
        module = linear.MergedColumnParallelLinear(
            weight, tp_rank=1, tp_size=2, output_partition_sizes=[2, 2]
        )
        expected = torch.cat([full_scale[2:4], full_scale[6:8]])
    target_scale = _frozen(torch.zeros(expected.T.shape))
    updater = _FP8InplaceUpdater.__new__(_FP8InplaceUpdater)
    updater._params = {"layer.proj.weight_scale_inv": target_scale}
    updater._fp8_modules = {"layer.proj": module}

    updater.feed("layer.proj.weight_scale_inv", full_scale)

    torch.testing.assert_close(target_scale.data, expected.T)


def test_fp8_updater_rejects_scales_that_do_not_split_across_tp(monkeypatch):
    from arctic_platform.inference.server.weight_sync.utils import _FP8InplaceUpdater

    linear = _install_fp8_vllm_stubs(monkeypatch)
    weight = _frozen(torch.zeros((2, 2), dtype=torch.float8_e4m3fn))
    updater = _FP8InplaceUpdater.__new__(_FP8InplaceUpdater)
    updater._params = {"layer.proj.weight_scale_inv": _frozen(torch.zeros(1, 1))}
    updater._fp8_modules = {"layer.proj": linear.RowParallelLinear(weight, tp_size=2)}

    with pytest.raises(ValueError, match="does not split"):
        updater.feed("layer.proj.weight_scale_inv", torch.zeros(1, 3))


def test_fp8_updater_requantizes_bf16_and_copies_unquantized_layers(monkeypatch):
    from arctic_platform.inference.server.weight_sync.utils import _FP8InplaceUpdater

    linear = _install_fp8_vllm_stubs(monkeypatch)
    model = nn.Module()
    model.mlp = nn.Module()
    model.mlp.gate_up_proj = linear.MergedColumnParallelLinear(
        _frozen(torch.zeros((4, 6), dtype=torch.float8_e4m3fn))
    )
    model.mlp.gate_up_proj.weight_scale = _frozen(torch.zeros(1))
    model.mlp.down_proj = linear.RowParallelLinear(
        _frozen(torch.zeros((4, 6), dtype=torch.bfloat16))
    )
    updater = _FP8InplaceUpdater(model, torch.bfloat16, "cpu")
    gate = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    up = gate + 12
    down = torch.ones((4, 6), dtype=torch.bfloat16)

    updater.feed("mlp.gate_proj.weight", gate)
    updater.feed("mlp.up_proj.weight", up)
    updater.feed("mlp.down_proj.weight", down)

    assert updater.pending == 0
    torch.testing.assert_close(
        model.mlp.gate_up_proj.weight.data.float(),
        torch.cat([gate, up]).T.to(torch.float8_e4m3fn).float(),
    )
    torch.testing.assert_close(
        model.mlp.gate_up_proj.weight_scale.data, torch.tensor([2.0])
    )
    torch.testing.assert_close(model.mlp.down_proj.weight.data, down)


class _Fp8RoutedExperts(nn.Module):
    """``RoutedExperts`` with block-FP8 weights; its loader records every call and copies one expert's shard."""

    def __init__(self, experts=2, inter=2, hidden=4):
        super().__init__()
        self.w13_weight = _frozen(torch.zeros((experts, 2 * inter, hidden), dtype=torch.float8_e4m3fn))
        self.w2_weight = _frozen(torch.zeros((experts, hidden, inter), dtype=torch.float8_e4m3fn))
        self.w13_weight_scale_inv = _frozen(torch.zeros((experts, 2, 2)))
        self.w2_weight_scale_inv = _frozen(torch.zeros((experts, 2, 1)))
        self.calls = []

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id

    def weight_loader(self, param, loaded, weight_name, shard_id, expert_id, return_success=False):
        self.calls.append((weight_name, shard_id, expert_id, tuple(loaded.shape)))
        rows = param.shape[1] // 2
        target = param.data[expert_id]
        if shard_id == "w1":
            target = target.narrow(0, 0, rows)
        elif shard_id == "w3":
            target = target.narrow(0, rows, rows)
        target.copy_(loaded)
        return True


def test_fp8_updater_loads_lora_wrapped_fp8_experts_and_scales(monkeypatch):
    from arctic_platform.inference.server.weight_sync.utils import _FP8InplaceUpdater

    _install_fp8_vllm_stubs(monkeypatch)
    model = nn.Module()
    model.mlp = nn.Module()
    model.mlp.experts = nn.Module()
    model.mlp.experts.base_layer = nn.Module()
    experts = model.mlp.experts.base_layer.routed_experts = _Fp8RoutedExperts()
    updater = _FP8InplaceUpdater(model, torch.bfloat16, "cpu")
    w13 = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4).to(torch.float8_e4m3fn)
    w2 = (torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) + 1).to(torch.float8_e4m3fn)
    w13_scale = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 1
    w2_scale = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1) + 1

    for name, tensor in (
        ("mlp.experts.w13_weight", w13),
        ("mlp.experts.w2_weight", w2),
        ("mlp.experts.w13_weight_scale_inv", w13_scale),
        ("mlp.experts.w2_weight_scale_inv", w2_scale),
    ):
        updater.feed(name, tensor)

    assert updater.unplaced == []
    assert [(weight_name, shard_id) for weight_name, shard_id, _, _ in experts.calls] == [
        *[("w13_weight", "w1"), ("w13_weight", "w3")] * 2,
        *[("w2_weight", "w2")] * 2,
        *[("w13_weight_scale_inv", "w1"), ("w13_weight_scale_inv", "w3")] * 2,
        *[("w2_weight_scale_inv", "w2")] * 2,
    ]
    assert torch.equal(experts.w13_weight.view(torch.uint8), w13.view(torch.uint8))
    assert torch.equal(experts.w2_weight.view(torch.uint8), w2.view(torch.uint8))
    torch.testing.assert_close(experts.w13_weight_scale_inv.data, w13_scale)
    torch.testing.assert_close(experts.w2_weight_scale_inv.data, w2_scale)


def test_fused_writer_refines_block_scales_behind_a_lora_modular_quant_method():
    from arctic_platform.inference.server.weight_sync.utils import _ShardAwareFusedWriter

    experts = _Fp8RoutedExperts()
    experts.w13_weight_scale_inv = _frozen(torch.zeros((2, 4, 2)))
    experts.w13_weight_scale_inv.quant_method = "block"
    # vLLM's LoRA wrapper swaps in a modular quant method that keeps the refine factor on the method it wraps.
    experts.quant_method = SimpleNamespace(old_quant_method=SimpleNamespace(weight_scale_refine=(2, 1)))
    model = nn.Module()
    model.mlp = nn.Module()
    model.mlp.experts = experts
    writer = _ShardAwareFusedWriter(model, torch.device("cpu"))
    scale = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    assert writer.feed("mlp.experts.w13_weight_scale_inv", scale)

    torch.testing.assert_close(experts.w13_weight_scale_inv.data, scale.repeat_interleave(2, dim=1))


def test_load_fp8_rejects_tensors_that_reach_no_parameter(monkeypatch):
    class FakeUpdater:
        pending = 0

        def __init__(self, model, dtype, device) -> None:
            self.unplaced = []

        def feed(self, name, tensor):
            self.unplaced.append(name)

    import arctic_platform.inference.server.weight_sync.utils as utils

    monkeypatch.setattr(utils, "_FP8InplaceUpdater", FakeUpdater)
    with pytest.raises(RuntimeError, match="1 of them are LoRA adapter tensors"):
        WeightSyncExtension._load_fp8(
            SimpleNamespace(device=torch.device("cpu"), model_config=SimpleNamespace(dtype=torch.bfloat16)),
            object(),
            _FakeEngine(
                [
                    ("layer.o_proj.lora_A.default.weight", torch.ones(2)),
                    ("layer.missing.weight", torch.ones(1)),
                ]
            ),
        )


def test_fused_writer_reports_canonical_moe_destination():
    class RoutedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = nn.Parameter(torch.zeros(1, 2, 1))
            self.w2_weight = nn.Parameter(torch.zeros(1, 1, 1))

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return expert_id

        def weight_loader(self, *args, **kwargs):
            return True

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = nn.Module()
            self.mlp.experts.routed_experts = RoutedExperts()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    from arctic_platform.inference.server.weight_sync.utils import _ShardAwareFusedWriter

    writer = _ShardAwareFusedWriter(Model(), torch.device("cpu"))

    assert writer.destination_name(
        "model.layers.0.mlp.experts.w13_weight"
    ) == "model.layers.0.mlp.experts.routed_experts.w13_weight"
    assert writer.destination_name(
        "model.layers.0.mlp.experts.w2_weight"
    ) == "model.layers.0.mlp.experts.routed_experts.w2_weight"
    wrapped_w13 = (
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w13_weight"
    )
    assert writer.destination_name(wrapped_w13) == (
        "model.layers.0.mlp.experts.routed_experts.w13_weight"
    )
    assert writer.feed(wrapped_w13, torch.ones(1, 2, 1))


def test_fused_writer_reports_minimax_block_sparse_moe_destination():
    class RoutedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = nn.Parameter(torch.zeros(1, 2, 1))
            self.w2_weight = nn.Parameter(torch.zeros(1, 1, 1))

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return expert_id

        def weight_loader(self, *args, **kwargs):
            return True

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_sparse_moe = nn.Module()
            self.block_sparse_moe.experts = nn.Module()
            self.block_sparse_moe.experts.routed_experts = RoutedExperts()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    from arctic_platform.inference.server.weight_sync.utils import _ShardAwareFusedWriter

    writer = _ShardAwareFusedWriter(Model(), torch.device("cpu"))
    assert writer.destination_name(
        "model.layers.0.block_sparse_moe.experts.w13_weight"
    ) == "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight"


def test_canonicalize_fused_moe_destination_inserts_routed_experts():
    from arctic_platform.inference.server.weight_sync.receiver import (
        _canonicalize_fused_moe_destination,
    )

    params = {
        "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight",
        "model.layers.0.block_sparse_moe.experts.routed_experts.w2_weight",
    }
    assert _canonicalize_fused_moe_destination(
        "model.layers.0.block_sparse_moe.experts.w13_weight", params
    ) == "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight"
    assert _canonicalize_fused_moe_destination(
        "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight", params
    ) == "model.layers.0.block_sparse_moe.experts.routed_experts.w13_weight"


def test_fused_writer_feeds_nongated_w13_as_w1_only():
    calls: list[str] = []

    class RoutedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = nn.Parameter(torch.zeros(2, 4, 1))
            self.w2_weight = nn.Parameter(torch.zeros(2, 1, 1))
            self.moe_config = SimpleNamespace(is_act_and_mul=False)

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return expert_id

        def weight_loader(self, param, tensor, weight_name, shard_id, expert_id,
                          return_success=False):
            calls.append(shard_id)
            assert tensor.shape[0] == 4
            return True

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = nn.Module()
            self.mixer.experts = RoutedExperts()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])

    from arctic_platform.inference.server.weight_sync.utils import _ShardAwareFusedWriter

    writer = _ShardAwareFusedWriter(Model(), torch.device("cpu"))
    assert writer.feed(
        "model.layers.0.mixer.experts.w13_weight", torch.ones(2, 4, 1)
    )
    assert calls == ["w1", "w1"]


@pytest.mark.parametrize(
    ("quantization", "weight_format", "applicable"),
    [(None, "hf", True), ("fp8", "vllm", False)],
)
def test_broadcast_reports_loaded_validation_applicability(
    monkeypatch,
    quantization,
    weight_format,
    applicable,
):
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: 1
    parallel_state.get_world_group = lambda: SimpleNamespace(rank=0)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        parallel_state,
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)

    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.0)
    extension = SimpleNamespace(
        device=torch.device("cpu"),
        model_runner=SimpleNamespace(model=model),
        model_config=SimpleNamespace(quantization=quantization),
        _get_or_create_broadcast_engine=lambda *args: object(),
        _load_batched=lambda model, engine, *, verifier=None: (
            1,
            4.0,
            4.0,
            {"source.weight": ["weight"]},
            [],
        ),
        _load_fp8=lambda model, engine, verifier=None: (1, 4.0),
        _build_result=lambda *args: {"status": "done"},
    )

    result = WeightSyncExtension.sync_weights_broadcast(
        extension,
        "127.0.0.1",
        1234,
        1,
        2,
        weight_format=weight_format,
    )

    assert result["recv_l2_sq"] == 4.0
    assert result["model_l2_sq_after"] == 4.0
    assert result["loaded_destination_validation_applicable"] is applicable
    assert result["loaded_destination_trace_collected"] is False
    assert ("loaded_model_l2_sq_after" in result) is applicable
    assert ("loaded_parameter_count" in result) is applicable
    assert ("applied_source_l2_sq" in result) is applicable


def test_broadcast_distinguishes_full_and_loaded_parameter_traces(monkeypatch):
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: 1
    parallel_state.get_world_group = lambda: SimpleNamespace(rank=0)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        parallel_state,
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setenv("ARCTIC_INFERENCE_DUMP_PARAM_L2", "1")

    model = nn.Linear(1, 1)
    model.weight.data.fill_(2.0)
    model.bias.data.fill_(3.0)
    extension = SimpleNamespace(
        device=torch.device("cpu"),
        model_runner=SimpleNamespace(model=model),
        model_config=SimpleNamespace(quantization=None),
        _get_or_create_broadcast_engine=lambda *args: object(),
        _load_batched=lambda model, engine, *, verifier=None: (
            2,
            13.0,
            4.0,
            {"source.weight": ["weight"]},
            ["receiver_only.bias"],
        ),
        _build_result=lambda *args: {"status": "done"},
    )

    result = WeightSyncExtension.sync_weights_broadcast(
        extension,
        "127.0.0.1",
        1234,
        1,
        2,
        weight_format="hf",
    )

    assert result["param_l2"] == {"weight": 4.0, "bias": 9.0}
    assert result["loaded_param_l2"] == {"weight": 4.0}
    assert result["applied_source_l2_sq"] == 4.0
    assert result["skipped_source_count"] == 1
    assert result["skipped_source_names"] == ["receiver_only.bias"]
    assert result["loaded_destination_trace_collected"] is True


def _ipc_tensor(*args):
    return torch.ones(1)


def _ipc_payload(names):
    import base64
    import pickle

    handles = [{"gpu0": (_ipc_tensor, [None] * 7)} for _ in names]
    return {"names": names, "ipc_handles_pickled": base64.b64encode(pickle.dumps(handles))}


def _run_sync_entry_point(entry_point, extension, tmp_path):
    if entry_point in ("sync_weights", "sync_weights_broadcast", "sync_spec_weights"):
        return getattr(WeightSyncExtension, entry_point)(extension, "127.0.0.1", 1234, 1, 2)
    if entry_point == "load_weights_cuda_ipc":
        return WeightSyncExtension.load_weights_cuda_ipc(extension, _ipc_payload(["w"]))
    if entry_point == "load_weights_cuda_ipc_chunk":
        first = WeightSyncExtension.load_weights_cuda_ipc_chunk(extension, _ipc_payload(["w"]))
        assert "cached_weight_copies_refreshed" not in first
        return WeightSyncExtension.load_weights_cuda_ipc_chunk(extension, _ipc_payload(["w"]), validate_names=["w"])
    path = tmp_path / "weights.pt"
    torch.save([("w", torch.ones(1))], path)
    return WeightSyncExtension.load_weights_from_shm_path(extension, str(path))


@pytest.mark.parametrize(
    "entry_point",
    [
        "sync_weights",
        "sync_weights_broadcast",
        "sync_spec_weights",
        "load_weights_cuda_ipc",
        "load_weights_cuda_ipc_chunk",
        "load_weights_from_shm_path",
    ],
)
def test_sync_refreshes_glm5next_cached_weight_copies(monkeypatch, tmp_path, entry_point):
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: 2
    parallel_state.get_world_group = lambda: SimpleNamespace(rank=0)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel_state)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid="gpu0"))

    # vLLM 0.31.0 Glm5NextLinearAttention: fp32 [C, 1, 4] q/k/v conv weights, and the
    # merged weight its first forward caches under inference mode (kda.py:505-515).
    kda = nn.Module()
    convs = []
    for i, name in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
        conv = nn.Module()
        conv.weight = nn.Parameter(torch.arange(12.0).view(3, 1, 4) + 100 * i, requires_grad=False)
        setattr(kda, name, conv)
        convs.append(conv)

    def merged():
        return torch.cat([c.weight.view(c.weight.size(0), c.weight.size(2)) for c in convs]).contiguous()

    # Glm5NextIndexer: bf16 wk_weights_proj (head_dim key rows, then the head-gate rows), and the
    # fp32 transpose of the head-gate rows its first forward caches (attention.py:338-344).
    indexer = nn.Module()
    indexer.head_dim = 2
    indexer.wk_weights_proj = nn.Module()
    indexer.wk_weights_proj.weight = nn.Parameter(torch.arange(15.0).view(5, 3).bfloat16(), requires_grad=False)

    def head_gate():
        return indexer.wk_weights_proj.weight.data[indexer.head_dim :, :].t().contiguous().float()

    with torch.inference_mode():
        kda._merged_conv_weight = merged()
        indexer._wp_fp32 = head_gate()
    cache_address = kda._merged_conv_weight.data_ptr()
    head_gate_address = indexer._wp_fp32.data_ptr()

    def update_in_place():  # ColumnParallelLinear.weight_loader: param.data.copy_
        for param in (*(conv.weight for conv in convs), indexer.wk_weights_proj.weight):
            param.data.add_(1.0)

    class Model(nn.ModuleDict):
        def load_weights(self, weights):
            names = {name for name, _ in weights}
            update_in_place()
            return names

    def load_batched(model, engine, *, verifier=None):
        update_in_place()
        return 4, 0.0, 0.0, {}, []

    model = Model({"kda": kda, "indexer": indexer})
    extension = SimpleNamespace(
        device=torch.device("cpu"),
        model_runner=SimpleNamespace(model=model, drafter=SimpleNamespace(model=model)),
        model_config=SimpleNamespace(quantization=None),
        _get_or_create_engine=lambda *args, **kwargs: _FakeEngine([("w", torch.ones(1))]),
        _get_or_create_broadcast_engine=lambda *args: object(),
        _load_batched=load_batched,
        _build_result=lambda *args: {"status": "done"},
        _validate_weight_sync_names=lambda *args, **kwargs: None,
    )
    extension._load_cpu_weights_into_model = lambda weights: WeightSyncExtension._load_cpu_weights_into_model(
        extension, weights
    )

    result = _run_sync_entry_point(entry_point, extension, tmp_path)

    assert result["cached_weight_copies_refreshed"] == 2
    assert torch.equal(kda._merged_conv_weight, merged()), "sampler still convolves with pre-sync conv weights"
    assert kda._merged_conv_weight.data_ptr() == cache_address
    assert torch.equal(indexer._wp_fp32, head_gate()), "sparse indexer still gates with pre-sync weights"
    assert indexer._wp_fp32.data_ptr() == head_gate_address


async def _noop_abort_all_streams():
    return None


def test_worker_gathers_full_and_loaded_parameter_traces(monkeypatch):
    from arctic_platform.inference.server.worker import InferenceWorker

    class FakeLLM:
        async def collective_rpc(self, *args, **kwargs):
            return [
                {
                    "loaded_destination_trace_collected": True,
                    "param_l2": {"weight": 4.0, "extra": 9.0},
                    "loaded_param_l2": {"weight": 4.0},
                }
            ]

    monkeypatch.setenv("ARCTIC_INFERENCE_DUMP_PARAM_L2", "1")
    worker_class = InferenceWorker.__ray_metadata__.modified_class
    result = asyncio.run(
        worker_class.sync_weights_broadcast(
            SimpleNamespace(
                llm=FakeLLM(),
                abort_all_streams=_noop_abort_all_streams,
            ),
            "127.0.0.1",
            1234,
            1,
            2,
        )
    )

    assert result["all_rank_param_l2"] == [{"weight": 4.0, "extra": 9.0}]
    assert result["all_rank_loaded_param_l2"] == [{"weight": 4.0}]
    assert "param_l2" not in result
    assert "loaded_param_l2" not in result


def test_load_batched_holds_one_rebuilt_oversized_tensor_at_a_time(monkeypatch):
    """Each rank rebuilds a tensor larger than the bucket whole; no earlier rebuild may still be alive then.

    Flash-Next's 3,355,443,200-byte ``gate_up_proj`` above a 2 GiB bucket measured +2x that on sampler GPU 0.
    The stream covers both ways a previous rebuild stays referenced: a small tensor between two rebuilds
    (``big_a`` -> ``big_b``) and two rebuilds back to back (``big_b`` -> ``big_c``, as GLM's w13 -> w2).
    """
    import weakref

    from arctic_platform.inference.server.weight_sync.engine import NCCLEngine

    class Stream:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(torch.cuda, "Event", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream())

    def chunk(name, shape, tensor_offset, nbytes, total):
        return dict(
            name=name, shape=shape, dtype="torch.uint8", offset=0, nbytes=nbytes,
            tensor_offset=tensor_offset, tensor_nbytes=total,
        )

    small = dict(name="small", shape=[2], dtype="torch.uint8", offset=4, nbytes=2)
    buckets = [
        [chunk("big_a", [8], 0, 4, 8)],
        [chunk("big_a", [8], 4, 4, 8), small],
        [chunk("big_b", [6], 0, 4, 6)],
        [chunk("big_b", [6], 4, 2, 6)],
        [chunk("big_c", [6], 0, 4, 6)],
        [chunk("big_c", [6], 4, 2, 6)],
    ]
    receiver = NCCLEngine.__new__(NCCLEngine)
    receiver.is_sender = False
    receiver.device = torch.device("cpu")
    receiver._nccl_stream = Stream()
    receiver._data_bufs = [torch.zeros(8, dtype=torch.uint8), torch.zeros(8, dtype=torch.uint8)]
    receiver.ensure_staging_buffers = lambda: None
    metas = {}

    def recv_pair(idx, stream):
        metas[idx] = dict(tensors=buckets.pop(0), is_last=not buckets)

    receiver._recv_pair = recv_pair
    receiver._parse_meta = metas.__getitem__

    real_empty, rebuilt, peaks = torch.empty, [], []

    def tracked_empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        rebuilt.append((weakref.ref(tensor.untyped_storage()), tensor.nbytes))
        peaks.append(sum(nbytes for ref, nbytes in rebuilt if ref() is not None))
        return tensor

    class Model(nn.Module):
        def load_weights(self, weights):
            return {name for name, _ in weights}

    monkeypatch.setattr(torch, "empty", tracked_empty)
    loaded = WeightSyncExtension._load_batched(SimpleNamespace(device=receiver.device), Model(), receiver)[0]
    monkeypatch.undo()

    assert loaded == 4
    assert peaks == [8, 6, 6]
