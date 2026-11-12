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
from __future__ import annotations

import importlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from arctic_platform.model.implementations.moe.hf_loader import _SafetensorsReader
from arctic_platform.model.implementations.moe.hf_loader import load_hf_weights
from arctic_platform.model.implementations.moe.hf_loader import supports_direct_hf_loading
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

FAMILIES = [
    "qwen3_moe",
    "qwen3_5_moe",
    "glm4_moe",
    "glm_moe_dsa",
    "minimax_m2",
    "afmoe",
    "nemotron_h",
    "qwen4_exp",
    "glm5_next",
]


def _converter(family):
    modules = {
        "qwen3_moe": ("qwen3_moe.modeling_qwen3_moe", "Qwen3MoeForCausalLM"),
        "qwen3_5_moe": ("qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
        "glm4_moe": ("glm4_moe.modeling_glm4_moe", "Glm4MoeForCausalLM"),
        "glm_moe_dsa": ("glm52.models.glm_moe_dsa.modeling_glm_moe_dsa", "GlmMoeDsaForCausalLM"),
        "minimax_m2": ("minimax_m2.modeling_minimax_m2", "MiniMaxM2ForCausalLM"),
        "afmoe": ("afmoe.modeling_afmoe", "AfmoeForCausalLM"),
        "nemotron_h": ("nemotron_h.modeling_nemotron_h", "NemotronHForCausalLM"),
        "qwen4_exp": ("qwen38.modeling_qwen4_exp", "Qwen4ExpForConditionalGenerationPrimeRL"),
        "glm5_next": ("glm53.modeling_glm5_next", "Glm5NextForConditionalGenerationPrimeRL"),
    }
    module, name = modules[family]
    return getattr(importlib.import_module(f"arctic_platform.model.implementations.{module}"), name)


def _source(family, fused=False, *, fp8=False, num_experts=4):
    state = {}
    generator = torch.Generator().manual_seed(42)

    def add(name, shape, quantized=False):
        value = torch.randn(shape, generator=generator).to(torch.bfloat16)
        if quantized:
            state[name] = value.to(torch.float8_e4m3fn)
            state[f"{name}_scale_inv"] = torch.ones((*shape[:-2], shape[-2] // 4, shape[-1] // 4))
        else:
            state[name] = value

    root = (
        "backbone"
        if family == "nemotron_h"
        else "model.language_model" if family in {"qwen4_exp", "glm5_next"} else "model"
    )
    add(
        f"{root}.{'embeddings' if family == 'nemotron_h' else 'embed_tokens'}.weight",
        (16, 4),
    )
    add(f"{root}.{'norm_f' if family == 'nemotron_h' else 'norm'}.weight", (4,))
    add("lm_head.weight", (16, 4))
    for layer in range(2):
        prefix = f"{root}.layers.{layer}"
        mlp = (
            f"{prefix}.mixer"
            if family == "nemotron_h"
            else f"{prefix}.block_sparse_moe" if family == "minimax_m2" else f"{prefix}.mlp"
        )
        add(f"{prefix}.input_layernorm.weight", (4,))
        add(
            f"{mlp}.{'router.gate' if family == 'afmoe' else 'gate'}.weight",
            (num_experts, 4),
        )
        if family in {"glm4_moe", "glm_moe_dsa", "nemotron_h", "minimax_m2", "glm5_next"}:
            bias = "e_score_correction_bias" if family == "minimax_m2" else "gate.e_score_correction_bias"
            add(f"{mlp}.{bias}", (num_experts,))
        if fused:
            up_name = "up_proj" if family == "nemotron_h" else "gate_up_proj"
            add(
                f"{mlp}.experts.{up_name}",
                (num_experts, 8 if family == "nemotron_h" else 16, 4),
                fp8,
            )
            add(f"{mlp}.experts.down_proj", (num_experts, 4, 8), fp8)
        else:
            projections = (
                [("up_proj", (8, 4)), ("down_proj", (4, 8))]
                if family == "nemotron_h"
                else [
                    ("w1" if family == "minimax_m2" else "gate_proj", (8, 4)),
                    ("w2" if family == "minimax_m2" else "down_proj", (4, 8)),
                    ("w3" if family == "minimax_m2" else "up_proj", (8, 4)),
                ]
            )
            for expert in range(num_experts):
                for projection, shape in projections:
                    add(f"{mlp}.experts.{expert}.{projection}.weight", shape, fp8)
        if family in {"qwen3_5_moe", "qwen4_exp", "glm4_moe", "glm_moe_dsa", "afmoe", "glm5_next"}:
            shared = "shared_expert" if family in {"qwen3_5_moe", "qwen4_exp"} else "shared_experts"
            for projection, shape in [
                ("gate_proj", (8, 4)),
                ("down_proj", (4, 8)),
                ("up_proj", (8, 4)),
            ]:
                add(f"{mlp}.{shared}.{projection}.weight", shape, fp8)
        if family in {"qwen3_5_moe", "qwen4_exp"}:
            add(f"{mlp}.shared_expert_gate.weight", (1, 4))
        if family == "qwen4_exp":
            for shard in range(3):
                add(
                    f"{prefix}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight",
                    (3, 4),
                )
    if family == "nemotron_h":
        add("backbone.layers.2.mixer.q_proj.weight", (4, 4))
        add("backbone.layers.3.mixer.in_proj.weight", (4, 4))
        add("mtp.unused.weight", (4, 4))
    return state


def _write_checkpoint(path, source):
    path.mkdir()
    items = sorted(source.items())
    weight_map = {}
    for shard, tensors in enumerate((items[::2], items[1::2])):
        filename = f"model-{shard}.safetensors"
        save_file({name: value.contiguous() for name, value in tensors}, path / filename)
        weight_map.update({name: filename for name, _ in tensors})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _converted(source, converter):
    state = {name: tensor.clone() for name, tensor in source.items()}
    converter.convert_to_prime(state)
    return state


@pytest.mark.parametrize(
    "family,fused",
    [
        (family, fused)
        for family in FAMILIES
        for fused in (False, True)
        if not (fused and family in {"afmoe", "minimax_m2"})
    ],
)
def test_direct_load_matches_full_conversion(tmp_path, family, fused):
    converter = _converter(family)
    source = _source(family, fused)
    expected = _converted(source, converter)
    expected.pop("mtp.unused.weight", None)
    checkpoint = tmp_path / "hf"
    _write_checkpoint(checkpoint, source)
    # Unindexed files must not shadow the checkpoint selected by its index.
    save_file({"lm_head.weight": torch.zeros(16, 4)}, checkpoint / "unrelated.safetensors")
    targets = {name: torch.empty_like(tensor) for name, tensor in expected.items()}
    pointers = {name: tensor.data_ptr() for name, tensor in targets.items()}
    files_before = sorted(checkpoint.rglob("*"))

    load_hf_weights(targets, checkpoint, converter.convert_layer_to_prime)

    for name, target in targets.items():
        torch_assert_equal(target, expected[name])
        assert target.data_ptr() == pointers[name]
    assert sorted(checkpoint.rglob("*")) == files_before


def _load_sharded(rank, world_size, checkpoint_root, rendezvous):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=45),
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world_size)))
        cases = [
            ("qwen3_moe", False, False, 3),
            ("qwen3_moe", True, False, 4),
            ("qwen3_5_moe", False, False, 4),
            ("glm4_moe", False, False, 2),
            ("glm_moe_dsa", False, True, 4),
            ("glm_moe_dsa", True, True, 4),
            ("nemotron_h", False, False, 4),
            ("nemotron_h", True, False, 4),
            ("qwen4_exp", True, False, 4),
            ("glm5_next", False, True, 4),
            ("glm5_next", True, True, 4),
        ]
        for family, fused, fp8, count in cases:
            converter = _converter(family)
            source = _source(family, fused, fp8=fp8, num_experts=count)
            expected = _converted(source, converter)
            expected.pop("mtp.unused.weight", None)
            targets = {}
            for name, full in expected.items():
                sharded = ".experts.w" in name and full.ndim == 3 or "ngram_embedding.weight" in name
                if sharded:
                    shape, _ = compute_local_shape_and_global_offset(full.shape, mesh, (Shard(0),))
                    targets[name] = DTensor.from_local(
                        torch.empty(shape, dtype=full.dtype),
                        mesh,
                        (Shard(0),),
                        shape=full.shape,
                        stride=full.stride(),
                    )
                else:
                    targets[name] = torch.empty_like(full)
            reads = []
            original = _SafetensorsReader.read

            def track_read(self, name, selection=None, *, _reads=reads, _original=original):
                _reads.append((name, selection))
                return _original(self, name, selection)

            _SafetensorsReader.read = track_read
            try:
                load_hf_weights(
                    targets,
                    Path(checkpoint_root) / f"{family}-{fused}-{fp8}",
                    converter.convert_layer_to_prime,
                )
            finally:
                _SafetensorsReader.read = original
            for name, target in targets.items():
                full = expected[name]
                if isinstance(target, DTensor):
                    local = target.to_local()
                    _, offsets = compute_local_shape_and_global_offset(full.shape, mesh, (Shard(0),))
                    reference = full[offsets[0] : offsets[0] + local.shape[0]]
                    torch_assert_equal(local.float(), reference.float())
                    if not fp8:
                        torch_assert_equal(target.full_tensor(), full)
                else:
                    torch_assert_equal(target.float(), full.float())
            local_count = (count + world_size - 1) // world_size
            start, stop = rank * local_count, min(count, (rank + 1) * local_count)
            for name, selection in reads:
                if "ngram_embedding.shard_" in name:
                    shard_id = int(name.split(".shard_")[1].split(".")[0])
                    assert selection is not None
                    first_row = shard_id * 3 + selection[0].start
                    last_row = shard_id * 3 + selection[0].stop
                    assert rank * 5 <= first_row < last_row <= min(9, (rank + 1) * 5)
                if ".experts." not in name:
                    continue
                part = name.split(".experts.", 1)[1].split(".", 1)[0]
                if part.isdigit():
                    assert start <= int(part) < stop
                elif selection is not None:
                    assert selection[0] == slice(start, stop)
        converter = _converter("qwen3_moe")
        expected = _converted(_source("qwen3_moe"), converter)
        targets = {}
        for name, full in expected.items():
            if ".experts.w" in name:
                shape, _ = compute_local_shape_and_global_offset(full.shape, mesh, (Shard(0),))
                targets[name] = DTensor.from_local(
                    torch.empty(shape, dtype=full.dtype),
                    mesh,
                    (Shard(0),),
                    shape=full.shape,
                    stride=full.stride(),
                )
            else:
                targets[name] = torch.empty_like(full)
        with pytest.raises(CheckpointException) as error:
            load_hf_weights(
                targets,
                Path(checkpoint_root) / "rank-local-failure",
                converter.convert_layer_to_prime,
            )
        assert set(error.value.failures) == {0}
    finally:
        dist.destroy_process_group()


def test_direct_load_reads_only_local_experts_and_ple_rows(tmp_path):
    cases = [
        ("qwen3_moe", False, False, 3),
        ("qwen3_moe", True, False, 4),
        ("qwen3_5_moe", False, False, 4),
        ("glm4_moe", False, False, 2),
        ("glm_moe_dsa", False, True, 4),
        ("glm_moe_dsa", True, True, 4),
        ("nemotron_h", False, False, 4),
        ("nemotron_h", True, False, 4),
        ("qwen4_exp", True, False, 4),
        ("glm5_next", False, True, 4),
        ("glm5_next", True, True, 4),
    ]
    for family, fused, fp8, count in cases:
        _write_checkpoint(
            tmp_path / f"{family}-{fused}-{fp8}",
            _source(family, fused, fp8=fp8, num_experts=count),
        )
    invalid = _source("qwen3_moe")
    invalid["model.layers.0.mlp.experts.0.gate_proj.weight"] = torch.ones(7, 4)
    _write_checkpoint(tmp_path / "rank-local-failure", invalid)
    mp.start_processes(
        _load_sharded,
        args=(2, str(tmp_path), str(tmp_path / "rendezvous")),
        nprocs=2,
        start_method="spawn",
    )


def test_direct_load_vlm_names_and_export_roundtrip(tmp_path):
    converter = _converter("qwen3_5_moe")
    source = {
        name.replace("model.", "model.language_model.", 1): tensor
        for name, tensor in _source("qwen3_5_moe", True).items()
    }
    source["model.visual.patch_embed.weight"] = torch.ones(4, 4)
    expected = _converted(source, converter)
    checkpoint = tmp_path / "vlm"
    _write_checkpoint(checkpoint, source)
    targets = {name: torch.empty_like(tensor) for name, tensor in expected.items()}
    load_hf_weights(targets, checkpoint, converter.convert_layer_to_prime)
    converter.convert_to_hf(targets)
    assert targets.keys() == source.keys()
    for name, tensor in source.items():
        torch_assert_equal(targets[name], tensor)


@pytest.mark.parametrize("builder_name", ["qwen35", "glm52"])
@pytest.mark.parametrize("cache_location", ["missing", "sibling", "hashed", "incomplete"])
def test_builder_load_expert_step_and_native_resume(
    tmp_path, monkeypatch, builder_name, single_rank_group, cache_location
):
    from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
    from arctic_platform.model.implementations.moe.layers.moe import _run_experts_for_loop_impl
    from arctic_platform.model.implementations.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    builder = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.model_builder")
    ModelConfig = importlib.import_module(f"arctic_platform.model.implementations.{builder_name}.config").ModelConfig
    model_cls = _converter("qwen3_moe")
    set_seed(7)
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        use_grouped_mm=False,
    )
    config._attn_implementation = "sdpa"

    def new_model(device="cpu"):
        with torch.device(device):
            model = model_cls(config)
            inject_prime_lm_head(model)
        return model

    reference = new_model()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(7)
        for parameter in reference.parameters():
            parameter.normal_(0, 0.1, generator=generator)
    source = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    model_cls.convert_to_hf(source)
    checkpoint = tmp_path / "hf"
    _write_checkpoint(checkpoint, source)
    from arctic_platform.model.convert_hf_to_primerl import convert_checkpoint
    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path

    config.save_pretrained(checkpoint)
    cache_root = tmp_path / "cache"
    if cache_location == "sibling":
        convert_checkpoint(checkpoint)
    elif cache_location == "hashed":
        convert_checkpoint(checkpoint, dest=hashed_conversion_cache_path(cache_root, checkpoint, "prime"))
    elif cache_location == "incomplete":
        incomplete = checkpoint / "prime"
        incomplete.mkdir()
        (incomplete / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"missing": "missing.safetensors"}})
        )
    files_before = sorted(checkpoint.rglob("*"))
    cache_files_before = sorted(cache_root.rglob("*"))
    direct_calls = []
    direct_load = builder.load_hf_weights

    def track_direct(*args):
        direct_calls.append(True)
        return direct_load(*args)

    def reject_conversion(*args, **kwargs):
        raise AssertionError("Training startup must not create a conversion cache")

    monkeypatch.setattr(builder, "load_hf_weights", track_direct)
    monkeypatch.setattr(builder, "ensure_node_local_conversion_cache", reject_conversion)
    monkeypatch.setattr(builder, "_write_conversion_cache", reject_conversion)
    loaded = new_model("meta")
    # Keep this CPU test independent of the offload path's CUDA buffer placement.
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *_args: None)
    builder.load_dcp_from_hf(
        loaded,
        ModelConfig(name=str(checkpoint), fsdp_cpu_offload=True, weight_conversion_cache_dir=str(cache_root)),
        None,
    )
    for module in loaded.modules():
        if hasattr(module, "tokens_per_expert"):
            module.tokens_per_expert.zero_()
    assert bool(direct_calls) == (cache_location in {"missing", "incomplete"})
    assert sorted(checkpoint.rglob("*")) == files_before
    assert sorted(cache_root.rglob("*")) == cache_files_before
    assert [name for name, _ in loaded.named_parameters()] == [name for name, _ in reference.named_parameters()]
    optim_ref = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    optim_loaded = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
    input_ids = torch.tensor([[1, 3, 5, 7]])

    def step(model, optimizer):
        optimizer.zero_grad()
        hidden = model.model.embed_tokens(input_ids).reshape(-1, config.hidden_size)
        mlp = model.model.layers[0].mlp
        scores = mlp.router.gate(hidden).softmax(-1)
        values, indices = scores.topk(config.num_experts_per_tok)
        routing = torch.zeros_like(scores).scatter(1, indices, values / values.sum(-1, keepdim=True))
        # Run the reference expert kernel without GPU dispatch or histogram kernels.
        expert_output = _run_experts_for_loop_impl(
            mlp.experts.w1,
            mlp.experts.w2,
            mlp.experts.w3,
            hidden.repeat(config.num_experts, 1),
            torch.full((config.num_experts,), hidden.shape[0], dtype=torch.long),
        ).reshape(config.num_experts, hidden.shape[0], config.hidden_size)
        output = torch.einsum("ne,end->nd", routing, expert_output)
        logits = model.lm_head(model.model.norm(output))["logits"]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 16), input_ids.flatten())
        loss.backward()
        grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
        optimizer.step()
        return logits.detach(), loss.detach(), grad_norm.detach()

    torch_assert_equal(step(reference, optim_ref), step(loaded, optim_loaded))
    for name, tensor in reference.state_dict().items():
        torch_assert_equal(loaded.state_dict()[name], tensor)

    native = tmp_path / "native"
    _write_checkpoint(
        native,
        {name: tensor.clone() for name, tensor in reference.state_dict().items()},
    )
    resumed = new_model("meta")
    builder.load_dcp_from_hf(resumed, ModelConfig(name=str(native), fsdp_cpu_offload=True), None)
    for module in resumed.modules():
        if hasattr(module, "tokens_per_expert"):
            module.tokens_per_expert.zero_()
    optim_resumed = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    optimizer_path = tmp_path / "optimizer.pt"
    torch.save(optim_ref.state_dict(), optimizer_path)
    optim_resumed.load_state_dict(torch.load(optimizer_path, weights_only=True))
    torch_assert_equal(step(reference, optim_ref), step(resumed, optim_resumed))
    for name, tensor in reference.state_dict().items():
        torch_assert_equal(resumed.state_dict()[name], tensor)


@pytest.mark.parametrize(
    "model_type,streaming,dequantize",
    [("deepseek_v4", False, True), ("deepseek_v41", True, False), ("future_model", False, False)],
)
def test_builder_retains_conversion_cache_for_other_layouts(
    tmp_path, monkeypatch, single_rank_group, model_type, streaming, dequantize
):
    from transformers import PretrainedConfig

    from arctic_platform.model.implementations.moe.base import PreTrainedModelPrimeRL
    from arctic_platform.model.implementations.qwen35 import model_builder as builder
    from arctic_platform.model.implementations.qwen35.config import ModelConfig

    calls = []

    class CachedModel(PreTrainedModelPrimeRL):
        def __init__(self):
            super().__init__(PretrainedConfig(tie_word_embeddings=False))
            self.config.model_type = model_type
            self.weight = torch.nn.Parameter(torch.empty(2, 2))

        @classmethod
        def is_hf_state_dict(cls, state):
            return "hf.weight" in state

        @classmethod
        def is_prime_state_dict(cls, state):
            return "weight" in state

        @classmethod
        def convert_to_prime(cls, state):
            calls.append("convert")
            state["weight"] = state.pop("hf.weight")
            state.pop("hf.scale", None)
            return state

        def init_buffers_post_meta(self):
            pass

    model = CachedModel()
    assert not supports_direct_hf_loading(model)
    source = tmp_path / "hf"
    _write_checkpoint(source, {"hf.weight": torch.ones(2, 2), "hf.scale": torch.ones(1)})
    if streaming:

        def stream(source_path, output_path):
            assert source_path == source
            calls.append("stream")
            _write_checkpoint(output_path, {"weight": torch.ones(2, 2)})

        model.stream_convert_to_prime = stream
    monkeypatch.setattr(builder, "_move_buffers_to_cuda", lambda *_args: None)
    cache_root = tmp_path / "cache"
    builder.load_dcp_from_hf(
        model,
        ModelConfig(
            name=str(source),
            fsdp_cpu_offload=True,
            weight_conversion_cache_dir=str(cache_root),
            dequantize_checkpoint=dequantize,
        ),
        None,
    )
    torch_assert_equal(model.weight, torch.ones(2, 2))
    assert calls == ["stream" if streaming else "convert"]
    indexes = list(cache_root.rglob("model.safetensors.index.json"))
    assert len(indexes) == 1
    assert indexes[0].parent.name == ("prime-dequantized" if dequantize else "prime")


@pytest.mark.parametrize("failure", ["missing", "shape", "expert_ids", "ple_shards"])
def test_direct_load_rejects_incomplete_or_incompatible_weights(tmp_path, failure):
    family = "qwen4_exp" if failure == "ple_shards" else "qwen3_moe"
    converter = _converter(family)
    source = _source(family, family == "qwen4_exp")
    expected = _converted(source, converter)
    if failure == "missing":
        del source["lm_head.weight"]
    elif failure == "shape":
        source["lm_head.weight"] = torch.ones(15, 4)
    elif failure == "expert_ids":
        for name in list(source):
            if ".experts.2." in name:
                del source[name]
    else:
        del source["model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight"]
    checkpoint = tmp_path / "invalid"
    _write_checkpoint(checkpoint, source)
    targets = {name: torch.empty_like(tensor) for name, tensor in expected.items()}
    with pytest.raises(
        (KeyError, ValueError, CheckpointException),
        match="missing training tensors|shape mismatch|expert ids|Non-contiguous",
    ):
        load_hf_weights(targets, checkpoint, converter.convert_layer_to_prime)


@pytest.mark.parametrize("family", ["qwen3_moe", "glm5_next"])
def test_offline_conversion_preserves_real_family_weights(tmp_path, family):
    from arctic_platform.model.convert_hf_to_primerl import convert_checkpoint
    from arctic_platform.model.implementations.moe.weights import load_state_dict

    source = _source(family)
    expected = _converted(source, _converter(family))
    checkpoint = tmp_path / "hf"
    _write_checkpoint(checkpoint, source)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": family}))
    cache = convert_checkpoint(checkpoint)
    actual = load_state_dict(cache)
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        torch_assert_equal(actual[name], tensor)


def _check_cache_readiness(rank, world_size, root, rendezvous):
    from types import SimpleNamespace

    from arctic_platform.model.implementations.moe.conversion_cache import resolve_ready_conversion_cache_path

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=45),
    )
    try:
        source = Path(root) / f"rank-{rank}"
        source.mkdir()
        config = SimpleNamespace(weight_conversion_cache_dir=str(Path(root) / "cache"))
        if rank == 0:
            _write_checkpoint(source / "prime", {"weight": torch.ones(2, 2)})
        dist.barrier()
        assert resolve_ready_conversion_cache_path(config, source, "prime") is None
        if rank == 1:
            _write_checkpoint(source / "prime", {"weight": torch.ones(2, 2)})
        dist.barrier()
        assert resolve_ready_conversion_cache_path(config, source, "prime") == source / "prime"
    finally:
        dist.destroy_process_group()


def test_prepared_cache_requires_readiness_on_every_rank(tmp_path):
    mp.start_processes(
        _check_cache_readiness,
        args=(2, str(tmp_path), str(tmp_path / "rendezvous")),
        nprocs=2,
        start_method="spawn",
    )
