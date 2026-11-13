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
"""DeepSeek-V4.1 full-weight sync from Arctic's trainer into vLLM 0.33's DeepseekV41ForCausalLM, on the CPU.

One config.json builds both models: the trainer (Arctic's DeepSeek-V4.1, a random published checkpoint, one fake
training step) and two TP ranks of vLLM's model with expert parallelism (gloo). The trainer's HF-named sync
(``build_iter_full_hf_weights``) goes through the receiver into a sampler loaded from the published checkpoint, which
must end equal, bit for bit, to a sampler loaded from the trained checkpoint (the Engram tables, which the sync leaves
out, as published):

- BF16 (``_load_batched``, ``model.load_weights`` per tensor).
- vLLM's online FP8 (``--quantization fp8``, per-tensor scales): ``_load_reload``, vLLM's layerwise reload, which
  requantizes each layer as loading did. Tensors a reload materializes and does not load are poisoned, so a tensor the
  trainer does not send shows up.

The vLLM model is built on the CPU with its CUDA-only constructor calls stubbed (side streams, the SM count, the
Engram n-gram hash's tokenizer) and FP8 quantization done by a reference of vLLM's per-tensor kernel. Weight loading
and the online quantization run vLLM's own code.
"""

from __future__ import annotations

import copy
import json
import os
import types
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.inference.utils import SUPPORTED_VLLM_VERSIONS

try:
    _VLLM_VERSION = version("vllm").split("+", 1)[0]
except PackageNotFoundError:
    _VLLM_VERSION = None
# The installed distribution, not ``import vllm``: other files of the suite put a stub vllm module in sys.modules.
# Only the spawned ranks import vLLM.
pytestmark = pytest.mark.skipif(
    _VLLM_VERSION not in SUPPORTED_VLLM_VERSIONS, reason="needs the vLLM release the receiver supports"
)

TP = 2
TEXT = dict(
    vocab_size=32,
    hidden_size=64,
    moe_intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=512,  # vLLM's compressor kernels fix the head and RoPE dims
    qk_rope_head_dim=64,
    q_lora_rank=32,
    o_lora_rank=16,
    o_groups=2,
    hidden_act="silu",
    swiglu_limit=10.0,
    rms_norm_eps=1e-6,
    max_position_embeddings=64,
    rope_theta=10000,
    compress_rope_theta=160000,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    scoring_func="sqrtsoftplus",
    topk_method="noaux_tc",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    sliding_window=4,
    # Layer 1 is a ratio-2 KV and index source, 2 a ratio-1 one, 3 an index source reusing 2's keys.
    compress_ratios=[0, 2, 1, 1],
    kv_source_layer_ids=[1, 2],
    index_source_layer_ids=[1, 2, 3],
    index_n_heads=2,
    index_head_dim=128,
    index_topk=2,
    candidate_source_layer_id=-1,
    hc_mult=2,
    hc_sinkhorn_iters=2,
    hc_eps=1e-6,
    engram_layer_ids=[0],
    engram_num_embeddings=[30],
    engram_max_ngram_size=2,
    engram_vocab_size=8,
    engram_n_heads=2,  # vLLM splits the hash heads over the TP ranks
    engram_head_dim=32,
    engram_pad_token_id=2,
    engram_compressed_vocab_size=32,
    num_nextn_predict_layers=0,
)


def _write_config(path: str) -> None:
    config = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "pad_token_id": 2,
        "image_token_id": 1,
        "text_config": {"model_type": "deepseek_v41_text", **TEXT},
        "vision_config": {"num_hidden_layers": 1, "hidden_size": 32, "num_attention_heads": 2, "intermediate_size": 64},
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


def _fp8_quant_reference(input, scale=None, num_token_padding=None, scale_ub=None, use_per_token_if_dynamic=False,
                         output=None, group_shape=None):
    """``vllm._custom_ops.scaled_fp8_quant`` per tensor as its CUDA kernels compute it: scale = amax / 448 in FP32, the
    values times the scale's FP32 reciprocal, clamped and rounded to e4m3."""
    assert input.ndim == 2 and scale_ub is None and not use_per_token_if_dynamic and group_shape is None
    finfo = torch.finfo(torch.float8_e4m3fn)
    if scale is None:
        scale = (input.abs().max().float() / torch.tensor(finfo.max)).reshape(1)
    quantized = (input.float() * (1.0 / scale.float().reshape(()))).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    if output is not None:
        output.copy_(quantized)
        quantized = output
    return quantized, scale


class _NoStream:
    def __init__(self, *args, **kwargs):
        pass


def _stub_cuda() -> None:
    import vllm._custom_ops as ops
    import vllm.models.deepseek_v41.nvidia.model as dsv41
    from vllm.platforms import current_platform

    torch.cuda.Stream = torch.cuda.Event = _NoStream
    torch.accelerator.current_device_index = lambda: 0
    torch.cuda.get_device_properties = lambda *args, **kwargs: types.SimpleNamespace(multi_processor_count=132)
    dsv41.NgramHashState = lambda *args, **kwargs: types.SimpleNamespace(lookback_depth=0)
    ops.scaled_fp8_quant = _fp8_quant_reference
    # Collectives over the gloo groups through torch.distributed, not pynccl.
    current_platform.get_device_communicator_cls = lambda: (
        "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"
    )


def _poison_reload_materialization() -> None:
    import vllm.model_executor.model_loader.reload.meta as meta

    materialize = meta.materialize_meta_tensor

    def poisoned(tensor):
        out = materialize(tensor)
        if out.is_floating_point() and out.element_size() > 1:
            out.fill_(float("nan"))
        elif out.element_size() == 1:
            out.view(torch.uint8).fill_(0xAB)
        return out

    meta.materialize_meta_tensor = poisoned


def _vllm_config(path: str, quantization: str | None):
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(
        model=path,
        skip_tokenizer_init=True,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",  # one process per rank, launched here
        quantization=quantization,
        load_format="dummy",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        engram_config={"cpu_offload": False},
    ).create_engine_config()


def _vllm_model(vllm_config, weights):
    """vLLM's model construction and load (model_loader: initialize_model, load_weights, online processing)."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
    from vllm.model_executor.model_loader.utils import initialize_model

    vllm_config.compilation_config.static_forward_context.clear()  # several models in one process
    with set_current_vllm_config(vllm_config), torch.device("cpu"):
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = initialize_model(vllm_config=vllm_config)
        finally:
            torch.set_default_dtype(torch.float32)
        with torch.no_grad():  # what the checkpoint does not carry (the vision tower) starts equal everywhere
            for tensor in [*model.parameters(), *model.buffers()]:
                if not tensor.is_meta and tensor.is_floating_point():
                    tensor.zero_()
        model.load_weights(weights)
        finalize_layerwise_processing(model, vllm_config.model_config)
    return model


def _trainer(path: str):
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41ForCausalLM

    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    config = DeepseekV41Config(
        text_config=raw["text_config"],
        vision_config=raw["vision_config"],
        image_token_id=raw["image_token_id"],
        pad_token_id=0,
        dspark_noise_token_id=1,
        engram_quantized=True,  # the BF16 repo keeps its Engram tables FP8 with UE8M0 scales
    )
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    try:
        return DeepseekV41ForCausalLM(config)
    finally:
        torch.set_default_dtype(torch.float32)


def _published_checkpoint(model) -> dict[str, torch.Tensor]:
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors

    template = model.convert_to_hf({name: tensor.detach() for name, tensor in named_weight_sync_tensors(model)})
    generator = torch.Generator().manual_seed(7)
    checkpoint = {}
    for name, tensor in sorted(template.items()):
        if tensor.dtype == torch.float8_e8m0fnu:
            codes = torch.randint(120, 135, tensor.shape, generator=generator, dtype=torch.uint8)
            checkpoint[name] = codes.view(torch.float8_e8m0fnu)
        elif tensor.dtype == torch.float8_e4m3fn:
            checkpoint[name] = torch.randn(tensor.shape, generator=generator).to(torch.float8_e4m3fn)
        else:
            checkpoint[name] = (0.05 * torch.randn(tensor.shape, generator=generator)).to(tensor.dtype)
    return checkpoint


def _trained_sync(path: str, published: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    """The trainer's sync after one fake step that moves every tensor it sends; every rank builds the stream rank 0
    broadcasts."""
    from arctic_platform.model.implementations.moe.deepspeed_integration import build_iter_full_hf_weights
    from arctic_platform.model.implementations.moe.vllm_weights import named_weight_sync_tensors
    from arctic_platform.model.implementations.moe.vllm_weights import weight_sync_skip_rule

    model = _trainer(path)
    model.load_state_dict(model.convert_to_prime(copy.deepcopy(published)), strict=True)
    skipped = weight_sync_skip_rule(model)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for _, tensor in named_weight_sync_tensors(model, include_export_buffers=False):
            if not skipped(tensor):
                tensor.add_((0.01 * torch.randn(tensor.shape, generator=generator)).to(tensor.dtype))
    rank = dist.get_rank
    dist.get_rank = lambda group=None: 0
    try:
        return [(name, tensor.clone()) for name, tensor in build_iter_full_hf_weights(model)()]
    finally:
        dist.get_rank = rank


def _state(model) -> dict[str, torch.Tensor]:
    state = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
    state.update({f"buffer:{name}": tensor.detach().clone() for name, tensor in model.named_buffers()})
    return state


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor.nan_to_num()


class _Engine:
    def __init__(self, weights):
        self._weights = weights

    def receive_weights(self):
        # The engine yields views into reused receive buffers.
        yield from ((name, tensor.clone()) for name, tensor in self._weights)


def _worker(rank: int, path: str, quantization: str | None, results) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _stub_cuda()
    if quantization is not None:
        _poison_reload_materialization()
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed import initialize_model_parallel

    from arctic_platform.inference.server.weight_sync.receiver import WeightSyncExtension
    from arctic_platform.inference.server.weight_sync.receiver import _quantizes_online
    from arctic_platform.inference.server.weight_sync.receiver import _remote_expert_filter

    vllm_config = _vllm_config(path, quantization)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=TP, rank=rank, local_rank=rank, distributed_init_method=f"file://{path}/store", backend="gloo"
        )
        initialize_model_parallel(tensor_model_parallel_size=TP)
    trainer = _trainer(path)
    published = _published_checkpoint(trainer)
    synced = _trained_sync(path, published)
    trained = dict(published)
    for name, tensor in synced:
        assert name in trained and trained[name].dtype == tensor.dtype, name
        trained[name] = tensor
    left_out = sorted(set(published) - {name for name, _ in synced})
    assert left_out == ["layers.0.engram.embed.scale", "layers.0.engram.embed.weight"], left_out

    want = _state(_vllm_model(vllm_config, trained.items()))
    sampler = _vllm_model(vllm_config, published.items())
    before = _state(sampler)
    receiver = types.SimpleNamespace(device=torch.device("cpu"), model_config=vllm_config.model_config)
    assert _quantizes_online(sampler) == (quantization is not None)
    with set_current_vllm_config(vllm_config), torch.device("cpu"):
        if quantization is None:
            loaded, *_, skipped_sources = WeightSyncExtension._load_batched(receiver, sampler, _Engine(synced))
            assert not skipped_sources
        else:
            remote = _remote_expert_filter(sampler)
            experts = {int(name.split(".")[4]) for name, _ in synced if ".ffn.experts." in name and not remote(name)}
            assert experts == set(range(4 * rank, 4 * rank + 4)), experts  # this EP rank's experts only
            loaded, _ = WeightSyncExtension._load_reload(receiver, sampler, _Engine(synced))
    assert loaded == len(synced)
    got = _state(sampler)
    assert sorted(got) == sorted(want)
    differ = [name for name in want if not torch.equal(_bits(got[name]), _bits(want[name]))]
    moved = [name for name in want if not torch.equal(_bits(before[name]), _bits(want[name]))]
    assert not differ, differ
    assert any(".ffn.experts." in name for name in moved) and any("weight_scale" in name for name in moved) == (
        quantization is not None
    )
    results[rank] = len(moved)


@pytest.mark.parametrize("quantization", [None, "fp8"], ids=["bf16-batched", "online-fp8-reload"])
def test_trained_weights_land_as_a_load_of_the_trained_checkpoint(tmp_path, quantization):
    path = str(tmp_path / "checkpoint")
    _write_config(path)
    results = mp.Manager().dict()
    mp.spawn(_worker, args=(path, quantization, results), nprocs=TP, join=True)
    assert sorted(results) == list(range(TP)) and all(results.values())
