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


def _rank_norm(model: nn.Module) -> dict:
    return WeightSyncExtension.compute_weight_norm(
        SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            device=torch.device("cpu"),
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


def _aggregate(rank_results: list[dict]) -> dict:
    from arctic_platform.inference.server.worker import InferenceWorker

    async def _collective_rpc(method):
        assert method == "compute_weight_norm"
        return rank_results

    # `InferenceWorker` is a Ray actor class; this is the plain class behind it.
    worker = InferenceWorker.__ray_metadata__.modified_class()
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

    with pytest.raises(RuntimeError, match="exact only at tensor_parallel_size=1"):
        _aggregate(ranks)


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
