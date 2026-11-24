"""CPU tests for the CUDA stream used by the vLLM communicator warm-up."""

import logging
import subprocess
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from runpy import run_path
from threading import local
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

vllm = pytest.importorskip("vllm")


@pytest.fixture
def runtime(monkeypatch, request):
    # Some inference tests install process-wide vLLM stubs during collection.
    if getattr(vllm, "__file__", None) is None:
        if PathFinder.find_spec("vllm") is None:
            pytest.skip("vLLM is not installed")
        subprocess.run(
            [sys.executable, "-m", "pytest", "--confcutdir", str(Path(__file__).parents[1]),
             "-p", "no:cacheprovider", "-q", f"{__file__}::{request.node.name}"],
            check=True,
        )
        return None

    from vllm.distributed.device_communicators import pynccl
    from vllm.utils import torch_utils

    # Load the transport without importing the optional Ray server.
    path = Path(__file__).parents[3] / "arctic_platform/inference/server/weight_sync/utils.py"
    init = run_path(str(path))["stateless_init_nccl"]

    def create(*, host, port, rank, world_size):
        return pynccl.StatelessProcessGroup(rank, world_size, torch.distributed.HashStore())

    monkeypatch.setattr(pynccl.StatelessProcessGroup, "create", create)
    monkeypatch.setattr(torch_utils, "_current_stream_tls", local())
    return init, torch_utils, pynccl


@pytest.mark.parametrize("cache", ["empty", "caller", "other", "warmup_failure"])
def test_init_keeps_caller_stream(runtime, monkeypatch, caplog, cache):
    if runtime is None:
        return
    init_nccl, torch_utils, pynccl = runtime
    from vllm import platforms

    device = torch.device("cuda:1")
    other_device, caller, dedicated = [
        Mock(device_index=i, cuda_stream=handle) for i, handle in [(0, 0), (1, 7), (1, 9)]
    ]
    active = SimpleNamespace(index=0)
    streams = [other_device, caller]

    def current_stream(device=None):
        return streams[active.index if device is None else device.index]

    def set_stream(stream):
        active.index = stream.device_index
        streams[active.index] = stream

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: 2, raising=False)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: dedicated)
    monkeypatch.setattr(torch_utils, "prev_set_stream", set_stream)
    monkeypatch.setitem(platforms.__dict__, "current_platform", Mock(is_rocm=lambda: False, is_cuda=lambda: True))
    monkeypatch.setattr(torch.accelerator, "device_index", lambda index: patch.object(active, "index", index))
    monkeypatch.setattr(pynccl, "NCCLLibrary", Mock(return_value=Mock(ncclGetUniqueId=lambda: b"id")))
    monkeypatch.setattr(torch, "zeros", Mock())
    all_reduce = Mock(side_effect=ValueError("warm-up failed") if cache == "warmup_failure" else None)
    monkeypatch.setattr(pynccl.PyNcclCommunicator, "all_reduce", all_reduce)

    expected = dedicated if cache == "other" else caller
    if cache in ("caller", "other"):
        torch_utils._current_stream_tls.value = expected

    caplog.set_level(logging.INFO)
    if cache == "warmup_failure":
        with pytest.raises(ValueError, match="warm-up failed"):
            init_nccl("localhost", 0, 0, 2, device)
    else:
        comm = init_nccl("localhost", 0, 0, 2, device)
        assert comm.device == device

    assert torch.cuda.current_stream() is other_device
    assert torch.cuda.current_stream(device) is caller
    if cache == "warmup_failure":
        assert getattr(torch_utils._current_stream_tls, "value", None) is None
    else:
        assert torch_utils.current_stream() is expected
        expected.synchronize.assert_called_once_with()
    assert sum("CUDA stream before=0x7 after=0x7" in message for message in caplog.messages) == 1


@pytest.mark.parametrize(
    ("device_count", "world_size", "disabled", "device"),
    [(0, 2, False, "cuda:0"), (2, 1, False, "cpu"), (2, 2, True, "cuda:99"), (2, 2, False, "cuda:0")],
    ids=["cpu", "one_rank", "disabled", "missing_nccl"],
)
def test_disabled_communicator_leaves_cache_empty(runtime, monkeypatch, device_count, world_size, disabled, device):
    if runtime is None:
        return
    init_nccl, torch_utils, pynccl = runtime
    if device_count:
        monkeypatch.setattr(torch._C, "_cuda_getDeviceCount", lambda: device_count, raising=False)
    else:
        monkeypatch.delattr(torch._C, "_cuda_getDeviceCount", raising=False)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=Mock(cuda_stream=0)))
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", "1" if disabled else "0")
    monkeypatch.setattr(pynccl, "NCCLLibrary", Mock(side_effect=OSError("no NCCL")))

    assert init_nccl("localhost", 0, 0, world_size, device).disabled
    assert getattr(torch_utils._current_stream_tls, "value", None) is None
    if not device_count or world_size == 1 or disabled:
        torch.cuda.current_stream.assert_not_called()
