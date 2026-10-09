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
    vllm_module.__version__ = "0.31.0"
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
        SimpleNamespace(device=torch.device("cpu")),
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
        _load_batched=lambda model, engine: (
            1,
            4.0,
            4.0,
            {"source.weight": ["weight"]},
            [],
        ),
        _load_fp8=lambda model, engine: (1, 4.0),
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
        _load_batched=lambda model, engine: (
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
