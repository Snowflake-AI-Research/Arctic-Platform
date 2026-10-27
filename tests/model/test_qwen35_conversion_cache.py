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

import builtins
import hashlib
import json
from pathlib import Path

import pytest


def test_conversion_cache_ready_requires_index_and_referenced_shards(tmp_path):
    from arctic_platform.model import conversion_cache_ready

    cache_dir = tmp_path / "prime"
    cache_dir.mkdir()

    assert conversion_cache_ready(cache_dir) is False

    (cache_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    )
    (cache_dir / "model-00001-of-00002.safetensors").write_bytes(b"not-used")

    assert conversion_cache_ready(cache_dir) is False

    (cache_dir / "model-00002-of-00002.safetensors").write_bytes(b"not-used")

    assert conversion_cache_ready(cache_dir) is True


def test_a_state_dict_too_small_to_shard_still_gets_an_index(tmp_path):
    """A converted checkpoint the cache can read, whatever its size.

    ``save_state_dict`` writes the index last, and ``conversion_cache_ready`` treats its presence as the mark of
    a complete cache. Sharding only kicks in above a size threshold, so without an index for the single-file
    case every checkpoint under it converts successfully and is then rejected as incomplete on the next look,
    which fails the load rather than falling back.
    """
    import torch

    from arctic_platform.model import conversion_cache_ready
    from arctic_platform.model import save_exported_state_dict

    save_dir = tmp_path / "prime"
    save_exported_state_dict({"a": torch.zeros(4), "b": torch.ones(2, 2)}, save_dir)

    index_path = save_dir / "model.safetensors.index.json"
    assert index_path.is_file(), f"no index written; directory holds {sorted(p.name for p in save_dir.iterdir())}"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    assert set(weight_map) == {"a", "b"}
    assert all((save_dir / shard).is_file() for shard in set(weight_map.values()))
    assert conversion_cache_ready(save_dir) is True


def test_conversion_cache_scope_detects_data_fast_and_env_override(monkeypatch):
    from arctic_platform.model import conversion_cache_is_node_local

    assert conversion_cache_is_node_local(Path("/data-fast/prime-rl-weight-cache/model/prime")) is True
    assert conversion_cache_is_node_local(Path("/tmp/prime-rl-weight-cache/model/prime")) is True
    assert conversion_cache_is_node_local(Path("/checkpoint/prime-rl-weight-cache/model/prime")) is False

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node-local")
    assert conversion_cache_is_node_local(Path("/checkpoint/prime-rl-weight-cache/model/prime")) is True

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "shared")
    assert conversion_cache_is_node_local(Path("/data-fast/prime-rl-weight-cache/model/prime")) is False
    assert conversion_cache_is_node_local(Path("/tmp/prime-rl-weight-cache/model/prime")) is False


def test_node_local_conversion_cache_uses_lock_and_skips_ready_cache(tmp_path):
    from arctic_platform.model.implementations.qwen35 import conversion_cache

    source_path = tmp_path / "source"
    source_path.mkdir()
    cache_path = tmp_path / "cache" / "prime"
    calls = {"load": 0, "convert": 0, "save": 0}

    def fake_load_state_dict(path):
        assert path == source_path
        calls["load"] += 1
        return {"a": object()}

    def fake_convert(state_dict):
        calls["convert"] += 1
        state_dict["b"] = state_dict.pop("a")

    def fake_save_state_dict(state_dict, save_dir):
        calls["save"] += 1
        save_dir.mkdir(parents=True)
        (save_dir / "model-00001-of-00001.safetensors").write_bytes(b"not-used")
        (save_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"b": "model-00001-of-00001.safetensors"}})
        )

    conversion_cache.ensure_node_local_conversion_cache(
        source_path,
        cache_path,
        fake_convert,
        "HF",
        "PrimeRL",
        fake_load_state_dict,
        fake_save_state_dict,
        rank=0,
        local_rank=0,
    )
    conversion_cache.ensure_node_local_conversion_cache(
        source_path,
        cache_path,
        fake_convert,
        "HF",
        "PrimeRL",
        fake_load_state_dict,
        fake_save_state_dict,
        rank=0,
        local_rank=0,
    )

    assert calls == {"load": 1, "convert": 1, "save": 1}
    assert conversion_cache.conversion_cache_ready(cache_path) is True


def test_failed_node_local_conversion_removes_tmp_cache(tmp_path):
    from arctic_platform.model.implementations.qwen35 import conversion_cache

    source_path = tmp_path / "source"
    source_path.mkdir()
    cache_path = tmp_path / "cache" / "prime"

    def fake_save_state_dict(state_dict, save_dir):
        save_dir.mkdir(parents=True)
        (save_dir / "model-00001-of-00001.safetensors").write_bytes(b"partial")
        raise RuntimeError("disk full")

    try:
        conversion_cache.ensure_node_local_conversion_cache(
            source_path,
            cache_path,
            lambda state_dict: None,
            "HF",
            "PrimeRL",
            lambda path: {"a": object()},
            fake_save_state_dict,
            rank=0,
            local_rank=0,
        )
        raise AssertionError("expected conversion to fail")
    except RuntimeError as exc:
        assert "disk full" in str(exc)

    leftovers = [p.name for p in cache_path.parent.iterdir() if p.name == "prime" or p.name.startswith("prime.")]
    assert leftovers == []


def test_forced_node_local_ready_sibling_does_not_write_or_lock(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.qwen35 import conversion_cache

    source_path = tmp_path / "source"
    source_path.mkdir()
    sibling = conversion_cache.sibling_conversion_cache_path(source_path, "prime")
    _write_ready_cache(sibling)
    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_SCOPE", "node")

    snapshot_path = conversion_cache.resolve_conversion_cache_path(
        SimpleNamespace(weight_conversion_cache_dir=str(tmp_path / "scratch")),
        source_path,
        "prime",
    )
    assert snapshot_path == sibling
    assert conversion_cache.conversion_cache_is_node_local(snapshot_path) is True

    real_open = builtins.open

    def guarded_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"unexpected write attempt: {path}")
        return real_open(path, mode, *args, **kwargs)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("ready immutable sibling must not require writes or locks")

    monkeypatch.setattr(conversion_cache, "open", guarded_open, raising=False)
    monkeypatch.setattr(Path, "mkdir", unexpected_call)
    monkeypatch.setattr(conversion_cache.fcntl, "flock", unexpected_call)

    conversion_cache.ensure_node_local_conversion_cache(
        source_path,
        snapshot_path,
        unexpected_call,
        "HF",
        "PrimeRL",
        unexpected_call,
        unexpected_call,
        rank=0,
        local_rank=0,
    )
    assert not (source_path / ".prime.lock").exists()


def _write_ready_cache(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "model-00001-of-00001.safetensors").write_bytes(b"not-used")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}})
    )


def test_resolve_conversion_cache_prefers_ready_sibling_over_hashed_scratch(tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    sibling = sibling_conversion_cache_path(source, "prime")
    _write_ready_cache(sibling)

    config = SimpleNamespace(weight_conversion_cache_dir=str(tmp_path / "scratch"))
    resolved = resolve_conversion_cache_path(config, source, "prime")
    assert resolved == sibling
    assert resolved != hashed_conversion_cache_path(config.weight_conversion_cache_dir, source, "prime")


def test_resolve_conversion_cache_uses_hashed_scratch_when_sibling_missing(tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    config = SimpleNamespace(weight_conversion_cache_dir=str(tmp_path / "scratch"))
    resolved = resolve_conversion_cache_path(config, source, "prime")
    assert resolved == hashed_conversion_cache_path(config.weight_conversion_cache_dir, source, "prime")
    assert resolved != sibling_conversion_cache_path(source, "prime")


def test_resolve_conversion_cache_falls_back_to_sibling_without_scratch_root(tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    config = SimpleNamespace(weight_conversion_cache_dir="")
    assert resolve_conversion_cache_path(config, source, "prime") == sibling_conversion_cache_path(source, "prime")


def test_conversion_cache_root_precedence_matches_runtime(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.glm52.config import ModelConfig as GlmModelConfig
    from arctic_platform.model.implementations.moe.conversion_cache import DEFAULT_WEIGHT_CONVERSION_CACHE_DIR
    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_root
    from arctic_platform.model.implementations.moe.conversion_cache import sibling_conversion_cache_path
    from arctic_platform.model.implementations.qwen35.config import ModelConfig as QwenModelConfig

    source = tmp_path / "GLM-5.2"
    source.mkdir()
    env_root = tmp_path / "env-cache"
    explicit_root = tmp_path / "config-cache"

    monkeypatch.delenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", raising=False)
    assert resolve_conversion_cache_root() == DEFAULT_WEIGHT_CONVERSION_CACHE_DIR
    for config_type in (QwenModelConfig, GlmModelConfig):
        default_config = config_type()
        assert default_config.weight_conversion_cache_dir is None
        assert resolve_conversion_cache_path(default_config, source, "prime") == hashed_conversion_cache_path(
            DEFAULT_WEIGHT_CONVERSION_CACHE_DIR, source, "prime"
        )

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", "")
    assert resolve_conversion_cache_root() == DEFAULT_WEIGHT_CONVERSION_CACHE_DIR

    monkeypatch.setenv("DSS_WEIGHT_CONVERSION_CACHE_DIR", str(env_root))
    assert resolve_conversion_cache_root() == str(env_root)
    assert resolve_conversion_cache_root(explicit_root) == explicit_root

    for config_type in (QwenModelConfig, GlmModelConfig):
        default_config = config_type()
        assert resolve_conversion_cache_path(default_config, source, "prime") == hashed_conversion_cache_path(
            env_root, source, "prime"
        )

        explicit_config = config_type(weight_conversion_cache_dir=str(explicit_root))
        assert resolve_conversion_cache_path(explicit_config, source, "prime") == hashed_conversion_cache_path(
            explicit_root, source, "prime"
        )

        sibling_config = config_type(weight_conversion_cache_dir="")
        assert resolve_conversion_cache_path(sibling_config, source, "prime") == sibling_conversion_cache_path(
            source, "prime"
        )


_MTP_BASE_NAMES = (
    "fc.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)
_MTP_LAYER_NAMES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
)


def _mtp_names(layer_count=1):
    return tuple(f"mtp.{name}" for name in _MTP_BASE_NAMES) + tuple(
        f"mtp.layers.{layer}.{name}" for layer in range(layer_count) for name in _MTP_LAYER_NAMES
    )


def _mtp_model(layer_count=1):
    from types import SimpleNamespace

    import torch

    model = torch.nn.Linear(3, 2, bias=False)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            mtp_num_hidden_layers=layer_count,
        )
    )
    return model


def _mtp_payload(layer_count=1):
    import torch

    dtypes = (torch.float32, torch.bfloat16, torch.float16)
    payload = {}
    for index, name in enumerate(_mtp_names(layer_count)):
        shape = (index % 3 + 1, index % 4 + 1)
        values = torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.float32).reshape(shape)
        payload[name] = (values + index).to(dtypes[index % len(dtypes)])
    return payload


def _write_mtp_checkpoint(path, payload):
    from safetensors.torch import save_file

    path.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    save_file(payload, path / shard_name, metadata={"format": "pt"})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard_name for name in payload}})
    )


def test_capture_mtp_weights_preserves_exact_cpu_payload_without_registering_state(monkeypatch, tmp_path):
    import torch

    from arctic_platform.model.implementations.qwen35 import mtp_weights

    payload = _mtp_payload()
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    model = _mtp_model()
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    buffer_ids = {name: id(buffer) for name, buffer in model.named_buffers()}
    state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: False)

    mtp_weights.capture_mtp_weights(model, source)
    preserved = dict(mtp_weights.iter_preserved_mtp_weights(model))

    assert tuple(preserved) == tuple(sorted(payload))
    for name, expected in payload.items():
        actual = preserved[name]
        assert actual.device.type == "cpu"
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.is_contiguous()
        assert actual.requires_grad is False
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
    assert {name: id(buffer) for name, buffer in model.named_buffers()} == buffer_ids
    assert set(model.state_dict()) == set(state_before)
    for name, expected in state_before.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


def test_capture_mtp_weights_nonmaster_synchronizes_without_reading_payload(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model = _mtp_model()
    broadcasts = []
    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mtp_weights.dist, "get_rank", lambda: 1)

    def broadcast_object_list(value, src):
        broadcasts.append((tuple(value), src))
        value[0] = None

    monkeypatch.setattr(mtp_weights.dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(mtp_weights, "safe_open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    mtp_weights.capture_mtp_weights(model, tmp_path / "unavailable-on-peer")

    assert broadcasts == [((None,), 0)]
    assert "_ap_preserved_mtp_weights" not in model.__dict__


def test_capture_mtp_weights_propagates_master_failure_to_nonmaster(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model = _mtp_model()
    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mtp_weights.dist, "get_rank", lambda: 1)

    def broadcast_object_list(value, src):
        assert src == 0
        value[0] = "ValueError: original Qwen3.5 MTP inventory mismatch"

    monkeypatch.setattr(mtp_weights.dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(mtp_weights, "safe_open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    with pytest.raises(RuntimeError, match="inventory mismatch"):
        mtp_weights.capture_mtp_weights(model, tmp_path / "unavailable-on-peer")


def test_capture_mtp_weights_is_noop_for_model_without_mtp(monkeypatch, tmp_path):
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model = _mtp_model(layer_count=0)
    model.__dict__["_ap_preserved_mtp_weights"] = ("stale",)
    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mtp_weights.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        mtp_weights.dist,
        "broadcast_object_list",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no collective expected")),
    )
    monkeypatch.setattr(mtp_weights, "safe_open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    mtp_weights.capture_mtp_weights(model, tmp_path / "not-needed")

    assert "_ap_preserved_mtp_weights" not in model.__dict__
    assert list(mtp_weights.iter_preserved_mtp_weights(model)) == []


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload.pop("mtp.fc.weight"), "missing="),
        (
            lambda payload: payload.__setitem__("mtp.unexpected.weight", payload["mtp.fc.weight"].clone()),
            "unexpected=",
        ),
        (
            lambda payload: payload.__setitem__(
                "mtp.layers.1.input_layernorm.weight",
                payload["mtp.layers.0.input_layernorm.weight"].clone(),
            ),
            "unexpected=",
        ),
    ],
)
def test_capture_mtp_weights_rejects_incomplete_or_mismatched_inventory(monkeypatch, tmp_path, mutate, match):
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    payload = _mtp_payload()
    mutate(payload)
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    model = _mtp_model()
    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match=match):
        mtp_weights.capture_mtp_weights(model, source)
    assert "_ap_preserved_mtp_weights" not in model.__dict__


def test_qwen_dcp_capture_uses_original_source_before_converted_cache(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.qwen35 import model_builder

    original = tmp_path / "original"
    converted = tmp_path / "converted" / "prime"
    original.mkdir()
    converted.mkdir(parents=True)
    events = []

    class FakePrimeModel:
        def __init__(self):
            self.config = SimpleNamespace(tie_word_embeddings=False)

        def to_empty(self, *, device):
            events.append(("to_empty", device))

        def state_dict(self):
            return {}

        def is_hf_state_dict(self, state_dict):
            return True

        def is_prime_state_dict(self, state_dict):
            return True

        def convert_to_prime(self, state_dict):
            return state_dict

        def init_buffers_post_meta(self):
            events.append(("init_buffers",))

        def named_buffers(self):
            return ()

    model = FakePrimeModel()
    config = SimpleNamespace(
        name=str(original),
        fsdp_cpu_offload=False,
        debug=SimpleNamespace(random_init=False),
        weight_conversion_cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(model_builder, "PreTrainedModelPrimeRL", FakePrimeModel)
    monkeypatch.setattr(model_builder.torch.distributed, "barrier", lambda: events.append(("barrier",)))

    def capture(captured_model, source, captured_config):
        assert captured_config is config
        events.append(("capture", captured_model, source))

    monkeypatch.setattr(
        model_builder,
        "maybe_capture_reduced_fixture_mtp",
        capture,
    )
    monkeypatch.setattr(
        model_builder, "load_state_dict_keys", lambda path: ["model.layers.0.mlp.experts.gate_up_proj"]
    )
    monkeypatch.setattr(model_builder, "resolve_conversion_cache_path", lambda *args: converted)
    monkeypatch.setattr(model_builder, "conversion_cache_is_node_local", lambda path: False)
    monkeypatch.setattr(model_builder, "conversion_cache_ready", lambda path: True)
    monkeypatch.setattr(model_builder, "get_world", lambda: SimpleNamespace(is_master=True, rank=0, local_rank=0))
    monkeypatch.setattr(model_builder, "HuggingFaceStorageReader", lambda path: ("reader", Path(path)))
    monkeypatch.setattr(model_builder, "maybe_observe_checkpoint_reader", lambda reader, *args: reader)
    monkeypatch.setattr(
        model_builder,
        "dcp_load",
        lambda state_dict, *, storage_reader: events.append(("dcp_load", storage_reader)),
    )

    model_builder.load_dcp_from_hf(model, config, parallel_dims=object())

    assert ("capture", model, original) in events
    assert ("dcp_load", ("reader", converted)) in events
    assert events.index(("capture", model, original)) < events.index(("dcp_load", ("reader", converted)))


def _rewrite_mtp_index(source, mutate):
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    mutate(index["weight_map"])
    index_path.write_text(json.dumps(index))


def _capture_mtp_provenance_locally(monkeypatch, source):
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: False)
    model = _mtp_model()
    mtp_weights.capture_mtp_weights(model, source)
    return dict(mtp_weights.iter_preserved_mtp_weights(model))


def test_capture_mtp_index_provenance_ignores_stale_unreferenced_shard(monkeypatch, tmp_path):
    import torch
    from safetensors.torch import save_file

    payload = _mtp_payload()
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    save_file(
        {"mtp.fc.weight": payload["mtp.fc.weight"] + 1000},
        source / "stale-unreferenced.safetensors",
        metadata={"format": "pt"},
    )

    preserved = _capture_mtp_provenance_locally(monkeypatch, source)

    assert tuple(preserved) == tuple(sorted(payload))
    assert torch.equal(preserved["mtp.fc.weight"], payload["mtp.fc.weight"])


def test_capture_mtp_index_provenance_rejects_missing_index_entry(monkeypatch, tmp_path):
    payload = _mtp_payload()
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    _rewrite_mtp_index(source, lambda weight_map: weight_map.pop("mtp.fc.weight"))

    with pytest.raises(RuntimeError, match="missing="):
        _capture_mtp_provenance_locally(monkeypatch, source)


def test_capture_mtp_index_provenance_rejects_extra_index_entry(monkeypatch, tmp_path):
    payload = _mtp_payload()
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    _rewrite_mtp_index(
        source,
        lambda weight_map: weight_map.__setitem__(
            "mtp.unexpected.weight",
            "model-00001-of-00001.safetensors",
        ),
    )

    with pytest.raises(RuntimeError, match="unexpected="):
        _capture_mtp_provenance_locally(monkeypatch, source)


def test_capture_mtp_index_provenance_rejects_unsafe_shard_path(monkeypatch, tmp_path):
    payload = _mtp_payload()
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    _rewrite_mtp_index(
        source,
        lambda weight_map: weight_map.__setitem__("mtp.fc.weight", "../outside.safetensors"),
    )

    with pytest.raises(RuntimeError, match="shard|path|relative"):
        _capture_mtp_provenance_locally(monkeypatch, source)


def test_capture_mtp_index_provenance_rejects_tensor_missing_from_selected_shard(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    payload = _mtp_payload()
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    selected = payload.copy()
    missing = selected.pop("mtp.fc.weight")
    save_file(selected, source / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    save_file({"mtp.fc.weight": missing}, source / "stale-unreferenced.safetensors", metadata={"format": "pt"})

    with pytest.raises(RuntimeError, match="does not match checkpoint index"):
        _capture_mtp_provenance_locally(monkeypatch, source)


@pytest.mark.parametrize("mutated_path", ["index", "shard"])
def test_capture_mtp_index_provenance_rejects_generation_mutation(monkeypatch, tmp_path, mutated_path):
    from safetensors import safe_open as real_safe_open
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.qwen35 import mtp_weights

    payload = _mtp_payload()
    source = tmp_path / "original"
    _write_mtp_checkpoint(source, payload)
    shard_path = source / "model-00001-of-00001.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    save_file(
        {name: tensor + 1000 for name, tensor in payload.items()},
        replacement,
        metadata={"format": "pt"},
    )
    mutated = False

    def mutate_then_open(path, *args, **kwargs):
        nonlocal mutated
        if not mutated:
            mutated = True
            if mutated_path == "index":
                index_path = source / "model.safetensors.index.json"
                index_path.write_text(index_path.read_text() + "\n")
            else:
                replacement.replace(shard_path)
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(mtp_weights, "safe_open", mutate_then_open)

    with pytest.raises(RuntimeError, match="changed|replaced|identity|generation"):
        _capture_mtp_provenance_locally(monkeypatch, source)
    assert mutated


def _reduced_fixture_mtp_case(tmp_path, *, mtp_payload=None, config_update=None, model_type="qwen3_5_moe_text"):
    from types import SimpleNamespace

    import torch
    from safetensors.torch import save_file

    source = tmp_path / "reduced-fixture"
    source.mkdir()
    source_config = {
        "model_type": "qwen3_5_moe" if model_type == "qwen3_5_moe_text" else "qwen4_exp",
        "text_config": {
            "model_type": model_type,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
        },
        "vision_config": {"depth": 2},
    }
    if config_update is not None:
        config_update(source_config)
    (source / "config.json").write_text(json.dumps(source_config))
    tensors = {"model.embed_tokens.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    tensors.update(mtp_payload or {})
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, source / shard_name, metadata={"format": "pt"})
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard_name for name in tensors}})
    )

    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(source.iterdir())
    ]
    fixture_sha256 = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request = {
        "schema_version": 1,
        "config_id": "qwen35-reduced-fixture",
        "fixture_locator": "ap-correctness/qwen35-reduced",
        "reference_path": str((tmp_path / "reviewed-reference").resolve()),
        "reviewed_hash": "0123456789abcdef",
        "fixture_sha256": fixture_sha256,
        "files": files,
        "source_checkpoint": "Qwen/Qwen3.5-Source",
        "num_hidden_layers": 2,
        "vision_depth": 2,
        "materialization": "pretrained-prefix-reduction",
    }
    model = _mtp_model()
    model.config.text_config.model_type = model_type
    model.config.text_config.num_hidden_layers = 2
    config = SimpleNamespace(
        name=request["fixture_locator"],
        debug=SimpleNamespace(reduced_fixture_mtp=request),
    )
    return model, source, config, request


def test_reduced_fixture_mtp_default_delegates_exact_source_to_strict_capture(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from arctic_platform.debug import reduced_fixture_mtp
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model = _mtp_model()
    source = tmp_path / "original"
    calls = []
    monkeypatch.setattr(
        mtp_weights,
        "capture_mtp_weights",
        lambda captured_model, captured_source: calls.append((captured_model, captured_source)),
    )

    reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(
        model,
        source,
        SimpleNamespace(name="original", debug=SimpleNamespace(reduced_fixture_mtp=None)),
    )

    assert calls == [(model, source)]


@pytest.mark.parametrize("model_type", ["qwen3_5_moe_text", "qwen4_exp_text"])
def test_reduced_fixture_mtp_default_rejects_damaged_original_with_all_mtp_missing(monkeypatch, tmp_path, model_type):
    from types import SimpleNamespace

    from arctic_platform.debug import reduced_fixture_mtp
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model, source, _, _ = _reduced_fixture_mtp_case(tmp_path, model_type=model_type)
    monkeypatch.setattr(mtp_weights.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="missing="):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(
            model,
            source,
            SimpleNamespace(name="unrequested-original", debug=SimpleNamespace(reduced_fixture_mtp=None)),
        )


@pytest.mark.parametrize("model_type", ["qwen3_5_moe_text", "qwen4_exp_text"])
def test_reduced_fixture_mtp_accepts_exact_indexed_absence_but_full_export_remains_strict(
    monkeypatch, tmp_path, model_type
):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen

    model, source, config, _ = _reduced_fixture_mtp_case(tmp_path, model_type=model_type)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)

    assert "_ap_preserved_mtp_weights" not in model.__dict__
    ordinary = torch.tensor([1.0])

    def ordinary_builder(built_model):
        assert built_model is model

        def iterator(*, include_skipped=False):
            assert include_skipped is True
            yield "model.weight", ordinary

        return iterator

    monkeypatch.setattr(qwen, "_build_iter_full_hf_weights", ordinary_builder)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    iterator = qwen._build_iter_full_hf_weights_with_mtp(model)(include_skipped=True)
    assert next(iterator) == ("model.weight", ordinary)
    with pytest.raises(RuntimeError, match="requires captured original MTP weights"):
        next(iterator)


@pytest.mark.parametrize("model_type", ["qwen3_5_moe_text", "qwen4_exp_text"])
def test_reduced_fixture_mtp_rejects_runtime_family_mismatch(monkeypatch, tmp_path, model_type):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp

    model, source, config, _ = _reduced_fixture_mtp_case(tmp_path, model_type=model_type)
    model.config.text_config.model_type = "qwen4_exp_text" if model_type == "qwen3_5_moe_text" else "qwen3_5_moe_text"
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="runtime model differs"):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)


@pytest.mark.parametrize("inventory", ["partial", "unexpected"])
def test_reduced_fixture_mtp_rejects_partial_or_unexpected_mtp_inventory(monkeypatch, tmp_path, inventory):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp

    payload = _mtp_payload()
    if inventory == "partial":
        payload.pop("mtp.fc.weight")
    else:
        payload["mtp.unexpected.weight"] = torch.ones(1)
    model, source, config, _ = _reduced_fixture_mtp_case(tmp_path, mtp_payload=payload)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="partial or unexpected"):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda request, config: request.__setitem__("schema_version", 2), "schema_version|Input should be 1"),
        (lambda request, config: request.__setitem__("config_id", ""), "config_id|at least 1"),
        (lambda request, config: request.__setitem__("fixture_sha256", "0" * 64), "manifest digest differs"),
        (lambda request, config: request.__setitem__("num_hidden_layers", 3), "reduced depths differ"),
        (lambda request, config: setattr(config, "name", "ap-correctness/other"), "locator differs"),
        (
            lambda request, config: request["files"][0].__setitem__("path", "../config.json"),
            "canonical relative file paths",
        ),
    ],
    ids=["schema", "config-id", "digest", "depth", "locator", "path"],
)
def test_reduced_fixture_mtp_rejects_invalid_authorization_fields(monkeypatch, tmp_path, change, match):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp

    model, source, config, request = _reduced_fixture_mtp_case(tmp_path)
    change(request, config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match=match):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)


@pytest.mark.parametrize(
    "change,match",
    [("manifest", "inventory differs"), ("config", "family or reduced depths differ")],
)
def test_reduced_fixture_mtp_rejects_snapshot_manifest_or_config_mismatch(monkeypatch, tmp_path, change, match):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp

    if change == "config":
        model, source, config, _ = _reduced_fixture_mtp_case(
            tmp_path,
            config_update=lambda value: value["text_config"].__setitem__("model_type", "other"),
        )
    else:
        model, source, config, _ = _reduced_fixture_mtp_case(tmp_path)
        (source / "unbound.json").write_text("{}")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match=match):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)


def test_reduced_fixture_mtp_rejects_generation_mutation(monkeypatch, tmp_path):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp

    model, source, config, _ = _reduced_fixture_mtp_case(tmp_path)
    index_path = source / "model.safetensors.index.json"
    replacement = tmp_path / "replacement-index.json"
    replacement.write_bytes(index_path.read_bytes())
    real_file_sha = reduced_fixture_mtp._file_sha
    calls = 0

    def mutate_after_manifest_hash(fd):
        nonlocal calls
        digest = real_file_sha(fd)
        calls += 1
        if calls == 3:
            replacement.replace(index_path)
        return digest

    monkeypatch.setattr(reduced_fixture_mtp, "_file_sha", mutate_after_manifest_hash)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="changed|generation differs"):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)
    assert calls == 3


@pytest.mark.parametrize(
    "peer_result,match",
    [
        (["ValueError: peer fixture bytes differ", False], "peer fixture bytes differ"),
        ([None, False], "decisions differ"),
    ],
)
def test_reduced_fixture_mtp_distributed_peer_failure_or_decision_mismatch_propagates(
    monkeypatch, tmp_path, peer_result, match
):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp

    model, source, config, _ = _reduced_fixture_mtp_case(tmp_path)
    monkeypatch.setattr(reduced_fixture_mtp, "_authenticate_fixture", lambda *args: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def all_gather_object(results, local_result):
        results[:] = [list(local_result), peer_result]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(RuntimeError, match=match):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)


def test_reduced_fixture_mtp_distributed_local_failure_is_gathered_before_raise(monkeypatch, tmp_path):
    import torch

    from arctic_platform.debug import reduced_fixture_mtp

    model, source, config, _ = _reduced_fixture_mtp_case(tmp_path)
    monkeypatch.setattr(
        reduced_fixture_mtp,
        "_authenticate_fixture",
        lambda *args: (_ for _ in ()).throw(ValueError("local fixture bytes differ")),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    gathered = []

    def all_gather_object(results, local_result):
        gathered.append(tuple(local_result))
        results[:] = [list(local_result), [None, False]]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(RuntimeError, match="local fixture bytes differ"):
        reduced_fixture_mtp.maybe_capture_reduced_fixture_mtp(model, source, config)
    assert gathered == [("ValueError: local fixture bytes differ", False)]
