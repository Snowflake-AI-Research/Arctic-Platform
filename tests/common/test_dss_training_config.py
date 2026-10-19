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

"""CPU-only tests of the explicit AP-owned DSS configuration bridge."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from arctic_platform.integrations.dss_training_config import dss_moe_initialize_payload


def job(**controls):
    return {
        "model_name": "qwen",
        "training_config": {
            "model_provider": "prime_rl",
            "ep_size": 8,
            "max_seq_len": 262144,
            **controls,
        },
    }


def test_flat_controls_bridge_without_mutating_source():
    source = job(deepep_token_chunk_size=16384, debug={"gpu_mem_preallocate_gib": 20, "random_init": True})
    original = deepcopy(source)
    transported = dss_moe_initialize_payload(source)
    assert source == original
    assert transported["training_config"]["prime_rl"] == {
        "deepep_token_chunk_size": 16384,
        "seq_len": 262144,
        "debug": {"gpu_mem_preallocate_gib": 20, "random_init": True},
    }
    assert "deepep_token_chunk_size" not in transported["training_config"]
    transported["training_config"]["prime_rl"]["debug"]["random_init"] = False
    assert source == original


def test_preserves_explicit_legacy_sequence_limit_and_equal_chunk():
    transported = dss_moe_initialize_payload(
        job(
            deepep_token_chunk_size=1024,
            prime_rl={"seq_len": 8192, "deepep_token_chunk_size": 1024},
        )
    )
    assert transported["training_config"]["prime_rl"]["seq_len"] == 8192
    assert transported["training_config"]["prime_rl"]["deepep_token_chunk_size"] == 1024


def test_chunk_conflict_is_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        dss_moe_initialize_payload(
            job(
                deepep_token_chunk_size=1024,
                prime_rl={"deepep_token_chunk_size": 4096},
            )
        )


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5, "1024"])
def test_strict_positive_flat_chunk(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        dss_moe_initialize_payload(job(deepep_token_chunk_size=chunk))


@pytest.mark.parametrize(
    "debug,expected",
    [
        (None, {"gpu_mem_preallocate_gib": 2}),
        ({}, {}),
        ({"gpu_mem_preallocate_gib": 20}, {"gpu_mem_preallocate_gib": 20}),
    ],
)
def test_top_debug_overrides_entire_legacy_block(debug, expected):
    transported = dss_moe_initialize_payload(
        job(
            debug=debug,
            prime_rl={"debug": {"gpu_mem_preallocate_gib": 2}},
        )
    )
    assert transported["training_config"]["prime_rl"]["debug"] == expected


@pytest.mark.parametrize("controls", [{"ep_size": 1}, {"model_provider": "huggingface"}])
def test_unsupported_dss_buffer_path_is_not_silently_ignored(controls):
    with pytest.raises(ValueError, match="compatibility bridge"):
        dss_moe_initialize_payload(job(debug={"gpu_mem_preallocate_gib": 20}, **controls))


def test_capacity_guard_is_transported_for_model_construction_validation():
    source = job(debug={"gpu_mem_preallocate_gib": -1})
    original = deepcopy(source)
    transported = dss_moe_initialize_payload(source)
    assert transported["training_config"]["prime_rl"]["debug"] == {"gpu_mem_preallocate_gib": -1}
    assert source == original


def test_default_resolution_excludes_capacity_guard_without_mutating_transport(monkeypatch):
    from arctic_platform.model import loader

    monkeypatch.setattr(loader, "_load_hf_config", lambda _: SimpleNamespace(model_type="qwen3_moe"))
    source = job(attn_implementation=None, debug={"gpu_mem_preallocate_gib": 20})
    original = deepcopy(source)
    transported = dss_moe_initialize_payload(source)
    assert transported["training_config"]["attn_implementation"] == "flash_attention_3"
    assert transported["training_config"]["prime_rl"]["debug"] == {"gpu_mem_preallocate_gib": 20}
    assert source == original


def test_flat_head_controls_reach_loader_channel_without_mutation():
    source = job(
        fused_lm_head_token_chunk_size=1024,
        fused_lm_head_vocab_chunk_size=8192,
        fp32_lm_head=False,
        prime_rl={"fused_cross_entropy": False},
    )
    original = deepcopy(source)
    training = dss_moe_initialize_payload(source)["training_config"]
    assert source == original
    assert training["prime_rl"] == {
        "fused_cross_entropy": False,
        "fused_lm_head_token_chunk_size": 1024,
        "fp32_lm_head": False,
        "seq_len": 262144,
    }
    assert training["fused_lm_head_token_chunk_size"] == 1024
    assert training["fp32_lm_head"] is False


@pytest.mark.parametrize(
    "key,value,nested",
    [
        ("fused_lm_head_token_chunk_size", 1024, 4096),
        ("fused_lm_head_token_chunk_size", 1024, "disabled"),
        ("fused_lm_head_token_chunk_size", 1, True),
        ("fp32_lm_head", True, False),
        ("fp32_lm_head", False, 0),
    ],
)
def test_head_conflicts_rejected(key, value, nested):
    with pytest.raises(ValueError, match=f"prime_rl.{key}"):
        dss_moe_initialize_payload(job(**{key: value, "prime_rl": {key: nested}}))


def test_preserves_equal_and_nested_only_head_options():
    training = dss_moe_initialize_payload(
        job(
            fused_lm_head_token_chunk_size=1024,
            fp32_lm_head=True,
            prime_rl={"fused_lm_head_token_chunk_size": 1024, "fp32_lm_head": True},
        )
    )["training_config"]
    assert training["prime_rl"]["fused_lm_head_token_chunk_size"] == 1024
    assert training["prime_rl"]["fp32_lm_head"] is True
    training = dss_moe_initialize_payload(
        job(
            prime_rl={"fused_lm_head_token_chunk_size": "disabled"},
        )
    )["training_config"]
    assert training["prime_rl"]["fused_lm_head_token_chunk_size"] == "disabled"
    assert "fp32_lm_head" not in training["prime_rl"]


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5, "1024"])
def test_strict_positive_head_chunk(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        dss_moe_initialize_payload(job(fused_lm_head_token_chunk_size=chunk))


@pytest.mark.parametrize("value", [0, 1, "false"])
def test_strict_head_precision_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        dss_moe_initialize_payload(job(fp32_lm_head=value))


@pytest.mark.parametrize("vocab_chunk", [4096, None, True])
def test_unsupported_vocab_chunk_is_not_silently_ignored(vocab_chunk):
    with pytest.raises(ValueError, match="vocab_chunk_size=8192"):
        dss_moe_initialize_payload(
            job(
                fused_lm_head_token_chunk_size=1024,
                fused_lm_head_vocab_chunk_size=vocab_chunk,
            )
        )


def test_nested_head_also_rejects_unsupported_vocab_chunk():
    with pytest.raises(ValueError, match="vocab_chunk_size=8192"):
        dss_moe_initialize_payload(
            job(
                prime_rl={"fused_lm_head_token_chunk_size": 1024},
                fused_lm_head_vocab_chunk_size=4096,
            )
        )


@pytest.mark.parametrize(
    "controls",
    [
        {"fused_lm_head_token_chunk_size": 1024},
        {"fp32_lm_head": True},
    ],
)
def test_unsupported_provider_head_path_rejected(controls):
    with pytest.raises(ValueError, match="compatibility bridge"):
        dss_moe_initialize_payload(job(model_provider="huggingface", **controls))


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("glm5_next", "sparse_mla"),
        ("qwen4_exp", "qsa_flex"),
        ("qwen3_moe", "flash_attention_3"),
    ],
)
def test_null_attention_uses_registered_loader_default_without_source_mutation(monkeypatch, model_type, expected):
    if model_type in {"glm5_next", "qwen4_exp"}:
        pytest.importorskip("transformers.models.qwen3_5_moe")
    from arctic_platform.model import loader

    monkeypatch.setattr(loader, "_load_hf_config", lambda _: SimpleNamespace(model_type=model_type))
    source = job(attn_implementation=None)
    original = deepcopy(source)
    transported = dss_moe_initialize_payload(source)
    assert source == original
    assert transported["training_config"]["attn_implementation"] == expected
    # Simulate the gateway dropping nulls before the next schema parse.
    serialized = {key: value for key, value in transported["training_config"].items() if value is not None}
    assert serialized["attn_implementation"] == expected


@pytest.mark.parametrize("attention", ["flashmla", "qsa_flex"])
def test_explicit_attention_is_not_replaced(monkeypatch, attention):
    from arctic_platform.model import loader

    def unexpected_resolution(_):
        pytest.fail("Explicit attention must not trigger model-default resolution")

    monkeypatch.setattr(loader, "_load_hf_config", unexpected_resolution)
    transported = dss_moe_initialize_payload(job(attn_implementation=attention))
    assert transported["training_config"]["attn_implementation"] == attention


def test_absent_attention_preserves_transport_default(monkeypatch):
    from arctic_platform.model import loader

    monkeypatch.setattr(loader, "_load_hf_config", lambda _: pytest.fail("Absent attention must stay absent"))
    assert "attn_implementation" not in dss_moe_initialize_payload(job())["training_config"]


@pytest.mark.parametrize(
    "model_type,attention",
    [
        ("qwen3_moe", "flash_attention_3"),
        ("glm5_next", "sparse_mla"),
        ("qwen4_exp", "qsa_flex"),
    ],
)
def test_omitted_provider_preserves_ap_moe_loader_without_source_mutation(monkeypatch, model_type, attention):
    if model_type in {"glm5_next", "qwen4_exp"}:
        pytest.importorskip("transformers.models.qwen3_5_moe")
    from arctic_platform.model import loader

    monkeypatch.setattr(loader, "_load_hf_config", lambda _: SimpleNamespace(model_type=model_type))
    source = job(attn_implementation=attention, debug={"gpu_mem_preallocate_gib": 20})
    del source["training_config"]["model_provider"]
    original = deepcopy(source)
    transported = dss_moe_initialize_payload(source)
    assert source == original
    assert "model_provider" not in source["training_config"]
    assert transported["training_config"]["model_provider"] == "prime_rl"
    assert transported["training_config"]["attn_implementation"] == attention
    assert transported["training_config"]["prime_rl"]["debug"]["gpu_mem_preallocate_gib"] == 20


def test_omitted_provider_rejects_unmapped_ap_loader(monkeypatch):
    from arctic_platform.model import loader

    monkeypatch.setattr(loader, "_load_hf_config", lambda _: SimpleNamespace(model_type="dense_model"))
    source = job()
    del source["training_config"]["model_provider"]
    with pytest.raises(ValueError, match="cannot map AP loader 'huggingface'"):
        dss_moe_initialize_payload(source)
    assert "model_provider" not in source["training_config"]


def test_omitted_dense_provider_remains_omitted(monkeypatch):
    from arctic_platform.model import loader

    monkeypatch.setattr(loader, "_load_hf_config", lambda _: pytest.fail("Dense transport must stay unchanged"))
    source = {"model_name": "dense", "training_config": {"ep_size": 1}}
    assert dss_moe_initialize_payload(source) == source


def test_explicit_provider_is_not_auto_replaced(monkeypatch):
    from arctic_platform.model import loader

    monkeypatch.setattr(loader, "_load_hf_config", lambda _: pytest.fail("Explicit provider must stay explicit"))
    source = job(model_provider="huggingface")
    assert dss_moe_initialize_payload(source) == source
