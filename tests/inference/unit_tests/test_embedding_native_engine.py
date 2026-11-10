"""Embedding replica lifecycle against the native async engine interface."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def replica(monkeypatch):
    # Generated protobuf bindings are deployment artifacts. Only the engine
    # lifecycle and configuration access are under test here.
    messages = ("EncodeRequest", "EncodeResponse", "AbortRequest", "AbortResponse",
                "ReplicaInfoRequest", "ReplicaInfoResponse", "SingleReplicaInfoResponse",
                "HealthCheckRequest", "HealthCheckResponse")
    proto = importlib.import_module("arctic_platform.inference.embedding.proto")
    bindings = ModuleType(f"{proto.__name__}.python")
    bindings.__path__ = []
    monkeypatch.setitem(sys.modules, bindings.__name__, bindings)
    monkeypatch.setattr(proto, "python", bindings, raising=False)
    for name, attrs in (
        ("inference_pb2", dict.fromkeys(messages, SimpleNamespace)),
        ("inference_pb2_grpc", {"InferenceServiceServicer": object}),
    ):
        module = ModuleType(f"{bindings.__name__}.{name}")
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(bindings, name, module)
    monkeypatch.setenv("VLLM_PLUGINS", "")
    path = Path(__file__).resolve().parents[3] / "arctic_platform/inference/embedding/replica.py"
    spec = importlib.util.spec_from_file_location("embedding_replica_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_embedding_start_info_and_stop_use_native_engine_interface(replica, monkeypatch):
    tokenizer = object()
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="embedding", max_model_len=256,
            dtype="float16", pooler_config=SimpleNamespace(task="embed")),
        parallel_config="parallel", decoding_config="decoding",
        scheduler_config="scheduler", lora_config=None,
    )
    engine = SimpleNamespace(vllm_config=config, get_tokenizer=lambda: tokenizer,
                             shutdown=Mock())
    factory = Mock(return_value=engine)
    monkeypatch.setattr(replica.AsyncLLM, "from_engine_args", factory)
    servicer = replica.InferenceServicer(SimpleNamespace())
    await servicer.start()
    assert servicer.ready
    assert servicer.tokenizer is tokenizer
    assert servicer.max_model_len == 256
    response = await servicer.GetReplicaInfo(SimpleNamespace(), Mock())
    info = response.replica_infos[0]
    assert (info.model_name, info.task, info.dtype) == ("embedding", "embed", "float16")
    assert (info.parallel_config, info.scheduler_config, info.lora_config) == ("parallel", "scheduler", "None")
    await servicer.stop()
    engine.shutdown.assert_called_once_with()
    assert not servicer.ready
