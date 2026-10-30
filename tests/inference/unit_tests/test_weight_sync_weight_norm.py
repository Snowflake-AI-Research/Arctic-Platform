from __future__ import annotations

import asyncio
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# Only the receiver's pure-Python helpers are under test here, and importing it runs
# `require_supported_vllm_version`, which refuses the vLLM a CPU box usually has. The
# stub has to be in place before that import, so it cannot move into a fixture. A
# vLLM already in `sys.modules` is left alone, so a supported install still wins.
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

import arctic_platform.inference.server.weight_sync.utils as weight_sync_utils
from arctic_platform.inference.server.weight_sync.receiver import WeightSyncExtension


class _Model(nn.Module):
    def __init__(self, norm: torch.Tensor, proj: torch.Tensor) -> None:
        super().__init__()
        self.norm = nn.Parameter(norm)
        self.proj = nn.Parameter(proj)


@pytest.fixture
def tp_world_size(monkeypatch):
    """Serve the one vLLM import the per-rank norm makes, with a settable TP size."""
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    size = {"value": 1}
    parallel_state.get_tensor_model_parallel_world_size = lambda: size["value"]
    monkeypatch.setitem(sys.modules, "vllm.distributed", types.ModuleType("vllm.distributed"))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel_state)
    return size


def _rank_norm(model: nn.Module, quantization: str | None = None) -> dict:
    return WeightSyncExtension.compute_weight_norm(
        SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            device=torch.device("cpu"),
            model_config=SimpleNamespace(quantization=quantization),
        )
    )


def test_rank_norm_squares_every_parameter(tp_world_size):
    model = _Model(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[2.0, 4.0]]))

    result = _rank_norm(model)

    # 1 + 4 + 9 + 4 + 16, accumulated in float64 and read back once.
    assert result == {"sq_sum": 34.0, "num_params": 2, "tp_size": 1}


def test_rank_norm_reports_the_tp_size_it_measured(tp_world_size):
    tp_world_size["value"] = 4

    result = _rank_norm(_Model(torch.tensor([1.0]), torch.tensor([2.0])))

    assert result["tp_size"] == 4


def test_rank_norm_refuses_a_quantized_sampler(tp_world_size):
    """Packed weights and scales would square to a number unrelated to the trainer's."""
    result = _rank_norm(_Model(torch.tensor([1.0]), torch.tensor([2.0])), quantization="fp8")

    assert result["error_code"] == "weight_norm_unavailable"
    assert "not defined for a fp8-quantized sampler" in result["error"]
    assert result["disable"] is True
    assert "sq_sum" not in result


def _aggregate(rank_results: list[dict]) -> dict:
    from arctic_platform.inference.server.worker import InferenceWorker
    from arctic_platform.inference.server.worker import WorkerLifecycleState

    async def _collective_rpc(method):
        assert method == "compute_weight_norm"
        return rank_results

    # `InferenceWorker` is a Ray actor class; this is the plain class behind it.
    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state = WorkerLifecycleState.READY
    worker.llm = SimpleNamespace(collective_rpc=_collective_rpc)
    return asyncio.run(worker.compute_weight_norm())


def test_single_rank_norm_is_the_square_root_of_the_sum():
    result = _aggregate([{"sq_sum": 25.0, "num_params": 3, "tp_size": 1}])

    assert result == {"norm": 5.0, "sq_sum": 25.0, "num_params": 3}


def test_tensor_parallel_norm_is_refused_instead_of_inflated():
    """Summing ranks double-counts anything replicated, so it is not reported."""
    ranks = [
        {"sq_sum": 25.0, "num_params": 3, "tp_size": 2},
        {"sq_sum": 25.0, "num_params": 3, "tp_size": 2},
    ]

    result = _aggregate(ranks)

    assert result["error_code"] == "weight_norm_unavailable"
    assert "exact only at tensor_parallel_size=1" in result["error"]
    assert result["disable"] is True
    assert "norm" not in result


def test_sleeping_sampler_does_not_read_weights():
    from arctic_platform.inference.server.worker import InferenceWorker
    from arctic_platform.inference.server.worker import WorkerLifecycleState

    called = {"n": 0}

    async def _collective_rpc(method):
        called["n"] += 1
        return []

    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state = WorkerLifecycleState.SLEEPING
    worker.llm = SimpleNamespace(collective_rpc=_collective_rpc)

    result = asyncio.run(worker.compute_weight_norm())

    assert result["error_code"] == "weight_norm_unavailable"
    assert "sleeping" in result["error"]
    assert "disable" not in result
    assert called["n"] == 0


def test_rank_error_is_returned_without_a_norm():
    result = _aggregate(
        [{"error_code": "weight_norm_unavailable", "error": "weight norm is not defined for a fp8-quantized sampler"}]
    )

    assert result == {
        "error_code": "weight_norm_unavailable",
        "error": "weight norm is not defined for a fp8-quantized sampler",
    }


def test_undeducible_names_raise_under_strict_mode(monkeypatch):
    def _explode(model):
        raise RuntimeError("unknown fused layout")

    monkeypatch.setattr(weight_sync_utils, "compute_expected_hf_param_names", _explode)
    monkeypatch.delenv("ARCTIC_WEIGHT_SYNC_STRICT_NAMES", raising=False)

    with pytest.raises(RuntimeError, match="could not derive expected param names"):
        WeightSyncExtension._validate_weight_sync_names(
            SimpleNamespace(), object(), ["model.layers.0.q_proj.weight"], context="cuda_ipc"
        )


def test_undeducible_names_warn_when_strict_mode_is_off(monkeypatch, caplog):
    def _explode(model):
        raise RuntimeError("unknown fused layout")

    monkeypatch.setattr(weight_sync_utils, "compute_expected_hf_param_names", _explode)
    monkeypatch.setenv("ARCTIC_WEIGHT_SYNC_STRICT_NAMES", "0")

    with caplog.at_level(
        logging.WARNING, logger="arctic_platform.inference.server.weight_sync.receiver"
    ):
        WeightSyncExtension._validate_weight_sync_names(
            SimpleNamespace(), object(), ["model.layers.0.q_proj.weight"], context="cuda_ipc"
        )

    assert "could not derive expected param names [cuda_ipc]" in caplog.text
    assert "continuing unchecked" in caplog.text


# --- Transport checksum: each load path squares the full source tensors it received. ---


@pytest.fixture
def cpu_cuda(monkeypatch):
    """The load paths' CUDA bookkeeping, served on CPU with one fake GPU."""
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid="GPU-0"))


class _LoadingModel(nn.Module):
    """Records what ``load_weights`` was handed, then zeroes it, so a checksum taken after placement would read 0."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.zeros(2))
        self.loaded: list[tuple[str, list[float]]] = []

    def load_weights(self, weights):
        names = set()
        for name, tensor in weights:
            self.loaded.append((name, tensor.float().tolist()))
            tensor.zero_()
            names.add("proj")
        return names


def _extension(model: nn.Module, quantization: str | None = None) -> WeightSyncExtension:
    ext = object.__new__(WeightSyncExtension)
    ext.model_runner = SimpleNamespace(model=model)
    ext.device = torch.device("cpu")
    ext.model_config = SimpleNamespace(quantization=quantization)
    ext._validate_weight_sync_names = lambda *args, **kwargs: None
    return ext


def _open_ipc_handle(tensor, *args):
    return tensor


def _ipc_payload(weights: list[tuple[str, torch.Tensor]]) -> dict:
    import base64
    import pickle

    handles = [{"GPU-0": (_open_ipc_handle, (tensor, 0, 0, 0, 0, 0, None))} for _, tensor in weights]
    return {
        "names": [name for name, _ in weights],
        "ipc_handles_pickled": base64.b64encode(pickle.dumps(handles)).decode("utf-8"),
    }


def _weights() -> list[tuple[str, torch.Tensor]]:
    # 1 + 4 + 9 + 16, plus bf16 0.25 + 2.25: all exact, so the float64 total is too.
    return [
        ("a", torch.tensor([1.0, 2.0])),
        ("b", torch.tensor([[3.0], [4.0]])),
        ("c", torch.tensor([0.5, 1.5], dtype=torch.bfloat16)),
    ]


def test_cuda_ipc_load_reports_the_received_checksum_before_placement(cpu_cuda):
    model = _LoadingModel()

    result = _extension(model).load_weights_cuda_ipc(_ipc_payload(_weights()))

    assert result["recv_l2_sq"] == 32.5
    assert result["params_loaded"] == 3
    assert [name for name, _ in model.loaded] == ["a", "b", "c"]


def test_cuda_ipc_chunk_reports_only_its_own_tensors(cpu_cuda):
    weights = _weights()
    ext = _extension(_LoadingModel())

    first = ext.load_weights_cuda_ipc_chunk(_ipc_payload(weights[:1]))
    rest = ext.load_weights_cuda_ipc_chunk(_ipc_payload(weights[1:]))

    assert (first["recv_l2_sq"], rest["recv_l2_sq"]) == (5.0, 27.5)


def test_cpu_file_load_reports_the_received_checksum(cpu_cuda):
    result = _extension(_LoadingModel())._load_cpu_weights_into_model(_weights())

    assert result["recv_l2_sq"] == 32.5
    assert result["update_path"] == "cpu_to_gpu"


class _Engine:
    def __init__(self, weights):
        self._weights = weights

    def receive_weights(self):
        yield from self._weights


def _nccl_sync(ext: WeightSyncExtension, engine, **kwargs) -> dict:
    ext._get_or_create_engine = lambda *args, **kw: engine
    return ext.sync_weights("127.0.0.1", 29500, 1, 2, **kwargs)


def test_nccl_batched_sync_reports_the_received_checksum(tp_world_size, cpu_cuda):
    tp_world_size["value"] = 2
    model = _LoadingModel()

    result = _nccl_sync(_extension(model), _Engine(_weights()))

    assert (result["update_path"], result["recv_l2_sq"]) == ("batched", 32.5)
    assert result["applied_source_l2_sq"] == 32.5
    assert result["skipped_source_count"] == 0
    assert [name for name, _ in model.loaded] == ["a", "b", "c"]


def test_nccl_batched_sync_reports_a_skipped_source(tp_world_size, cpu_cuda):
    """A tensor load_weights does not place still counts in recv_l2_sq, and is named as skipped."""
    tp_world_size["value"] = 2

    class _DroppingModel(_LoadingModel):
        def load_weights(self, weights):
            names = set()
            for name, tensor in weights:
                self.loaded.append((name, tensor.float().tolist()))
                tensor.zero_()
                if name != "c":
                    names.add("proj")
            return names

    ext = _extension(_DroppingModel())
    ext.parallel_config = SimpleNamespace(enable_expert_parallel=True, pipeline_parallel_size=1)
    result = _nccl_sync(ext, _Engine(_weights()))

    assert result["recv_l2_sq"] == 32.5
    assert result["applied_source_l2_sq"] == 30.0
    assert result["skipped_source_count"] == 1


def test_nccl_fp8_sync_reports_the_trainers_bf16_checksum(tp_world_size, cpu_cuda):
    ext = _extension(_LoadingModel(), quantization="fp8")
    ext._load_fp8 = lambda model, engine: (3, 32.5)

    result = _nccl_sync(ext, _Engine([]))

    assert (result["update_path"], result["recv_l2_sq"]) == ("fp8", 32.5)


def test_nccl_zero_copy_sync_reports_no_checksum(tp_world_size, cpu_cuda):
    """It writes into possibly padded views without naming what arrived, so it has no honest total."""
    ext = _extension(_LoadingModel())
    ext._load_direct_zero_copy = lambda model, engine: 3

    result = _nccl_sync(ext, _Engine([]), direct_mode=True)

    assert result["update_path"] == "direct_zero_copy"
    assert "recv_l2_sq" not in result


def _worker_load(method: str, rank_results: list[dict], *args) -> dict:
    from arctic_platform.inference.server.worker import InferenceWorker

    async def _collective_rpc(name, args=()):
        assert name == method
        return rank_results

    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.llm = SimpleNamespace(collective_rpc=_collective_rpc)
    return asyncio.run(getattr(worker, method)(*args))


@pytest.mark.parametrize(
    "method, args",
    [
        ("load_weights_cuda_ipc", ({},)),
        ("load_weights_cuda_ipc_chunk", ({}, None)),
        ("load_weights_from_shm_path", ("/dev/shm/x",)),
    ],
)
def test_worker_reports_every_tp_ranks_received_checksum(method, args):
    """Each TP rank receives the whole model, so rank 0's value alone does not vouch for the others."""
    result = _worker_load(method, [{"recv_l2_sq": 32.5, "loaded": 3}, {"recv_l2_sq": 30.0, "loaded": 3}], *args)

    assert result == {"recv_l2_sq": 32.5, "loaded": 3, "rank_recv_l2_sq": [32.5, 30.0]}


def test_worker_marks_a_rank_without_a_checksum():
    result = _worker_load("load_weights_cuda_ipc", [{"recv_l2_sq": 32.5}, {"status": "done"}], {})

    assert result["rank_recv_l2_sq"] == [32.5, None]
