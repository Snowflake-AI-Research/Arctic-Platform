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
"""Gemma 4 on the huggingface route with ``patches.dense_shard`` trains as the replicated model does (8 GPUs, one
node, DeepSpeed ZeRO-2 in BF16 with FP32 masters, built as the job builds them)."""

from __future__ import annotations

import gc
import json
import os

import pytest
import torch
import torch.multiprocessing as mp

WORLD, TOKENS, GAS, STEPS = 8, 1024, 2, 3
VOCAB = 16384

# gemma-4-31B-it's config.json with 6 layers (the 5:1 sliding / full pattern as 2:1), hidden 1024 (5376), 4 query
# heads (32), 2 sliding and 1 full KV heads (16, 4), head sizes kept (256, 512), MLP 4096 (21504), vocabulary 16384
# (262144), a 128-token sliding window (1024) and a one-layer vision tower; the special tokens moved into the
# vocabulary, above the sampled ids.
_GEMMA4 = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "dtype": "bfloat16",
    "audio_config": None,
    "audio_token_id": VOCAB - 7,
    "boa_token_id": VOCAB - 6,
    "boi_token_id": VOCAB - 5,
    "eoa_token_id": VOCAB - 4,
    "eoa_token_index": VOCAB - 4,
    "eoi_token_id": VOCAB - 3,
    "eos_token_id": [1, 106],
    "image_token_id": VOCAB - 2,
    "video_token_id": VOCAB - 1,
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "vision_soft_tokens_per_image": 4,
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_k_eq_v": True,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "enable_moe_block": False,
        "eos_token_id": 1,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "hidden_size_per_layer_input": 0,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"] * 2,
        "max_position_embeddings": 4096,
        "model_type": "gemma4_text",
        "num_attention_heads": 4,
        "num_global_key_value_heads": 1,
        "num_hidden_layers": 6,
        "num_key_value_heads": 2,
        "num_kv_shared_layers": 0,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        "sliding_window": 128,
        "tie_word_embeddings": True,
        "use_bidirectional_attention": "vision",
        "use_cache": True,
        "use_double_wide_mlp": False,
        "vocab_size": VOCAB,
        "vocab_size_per_layer_input": VOCAB,
    },
    "vision_config": {
        "attention_bias": False,
        "dtype": "bfloat16",
        "global_head_dim": 32,
        "head_dim": 32,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 1024,
        "model_type": "gemma4_vision",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
        "standardize": True,
        "use_clipped_linears": False,
    },
}


def write_checkpoint(config: dict, path: str) -> None:
    """A random-init checkpoint of ``config``'s class; key norms and ``layer_scalar`` buffers as in Gemma 4's."""
    import transformers
    from transformers import AutoConfig

    from arctic_platform.testing_utils import set_seed

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    set_seed(0)
    model = getattr(transformers, config["architectures"][0])(AutoConfig.from_pretrained(path)).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
            if name.endswith("self_attn.k_norm.weight") and "language_model" in name:
                parameter.mul_(0.12 if parameter.numel() == 256 else 0.06)
        for name, buffer in model.named_buffers():
            if name.endswith("layer_scalar"):
                buffer.fill_(0.5 + 0.05 * int(name.split(".")[-2]))
    model.save_pretrained(path)


def build_model(model_dir: str, shard: bool):
    """The huggingface loader with the job's patches, on the host, then moved to this rank's GPU."""
    from arctic_platform.model import build_model as build
    from arctic_platform.model import dense_model_spec

    spec = dense_model_spec(
        model_path=model_dir,
        dtype="bfloat16",
        attn_implementation="sdpa",  # deterministic; at SP 1 the job has no other head-512 kernel
        sequence_parallel=1,
        liger=True,
        training_options={
            "activation_checkpointing": True,
            "checkpoint_keep_attention": True,
            "fused_lm_head_token_chunk_size": 512,
            "fused_lm_head_vocab_chunk_size": 4096,
            "shard_dense": shard,
        },
    )
    return build(spec, parallel_groups={"sp_group": None}).model.to(f"cuda:{torch.cuda.current_device()}")


def build_engine(model, rank: int, model_dir: str):
    from arctic_platform.common.deepspeed_engine import OptimizerSettings
    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine

    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            world_size=WORLD,
            gradient_clipping=1.0,
            fp32_masters=True,
            ds_overrides={
                "gradient_accumulation_steps": GAS,
                "train_batch_size": GAS * WORLD,
                "zero_optimization": {"stage": 2},
            },
        )
    )
    engine, _, _ = initialize_training_engine(
        model,
        ds_config,
        optimizer=OptimizerSettings(lr=1e-4, weight_decay=0.01),
        ep_size=1,
        mpu=None,
        model_path=model_dir,
        rank=rank,
    )
    return engine


def micro_batches(rank: int, steps: int) -> list[dict]:
    """Each rank's micro-batches, the same for every run."""
    generator = torch.Generator().manual_seed(1000 + rank)
    out = []
    for _ in range(steps * GAS):
        ids = torch.randint(2, VOCAB - 8, (1, TOKENS), generator=generator)
        out.append({"input_ids": ids, "labels": ids.roll(-1, dims=1), "position_ids": torch.arange(TOKENS)[None]})
    return out


def forward_loss(engine, batch: dict) -> torch.Tensor:
    device = torch.device("cuda", torch.cuda.current_device())
    kwargs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    out = engine(**kwargs, use_cache=False, dss_compute_logprobs=True)
    # Scaled so the gradient norm is above the clipping threshold.
    return -4 * out["logprobs"].float().mean()


def full_weights(model) -> dict[str, torch.Tensor]:
    """Every parameter whole, on the CPU: the sharded ones gathered (collective)."""
    from arctic_platform.model.implementations.moe import dense_shard

    out = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not dense_shard.is_dense_shard(p)}
    for module in model.modules():
        sharded = getattr(module, "_ap_dense_shards", None)
        if sharded is not None:
            out.update({name: t.cpu() for name, t in sharded.full_parameters().items()})
    return out


def sharded_modules(model) -> list:
    return [m._ap_dense_shards for m in model.modules() if getattr(m, "_ap_dense_shards", None) is not None]


def _train(rank: int, model_dir: str, shard: bool) -> dict:
    # Free the previous run's engine and model first, or their cycles may release GPU memory during this load.
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    model = build_model(model_dir, shard)
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated() - before
    sharded = sharded_modules(model)
    assert bool(sharded) == shard
    engine = build_engine(model, rank, model_dir)
    assert engine.has_moe_layers == shard  # the shards are reduced over their expert-data-parallel groups
    fp32_gradients = zero_keeps_fp32_gradients(engine.optimizer)
    initial = full_weights(model)
    losses, norms, weights = [], [], []
    batches = micro_batches(rank, STEPS)
    for step in range(STEPS):
        for batch in batches[step * GAS : (step + 1) * GAS]:
            loss = forward_loss(engine, batch)
            engine.backward(loss)
            engine.step()
            losses.append(loss.item())
        norms.append(float(engine.get_global_grad_norm()))
        weights.append(full_weights(model))
    out = {
        "losses": losses,
        "norms": norms,
        "initial": initial,
        "weights": weights,
        "loaded": loaded,
        "shard_bytes": sum(s.shard_bytes() for s in sharded),
        "full_bytes": sum(s.full_bytes() for s in sharded),
        "modules": len(sharded),
        "fp32_gradients": fp32_gradients,
    }
    engine.destroy()
    del engine, model
    torch.cuda.empty_cache()
    return out


def zero_keeps_fp32_gradients(optimizer) -> bool:
    """Whether ZeRO keeps the BF16 parameters' reduced gradients in FP32 (DeepSpeed #8749 with FP32 accumulation)."""
    uses_fp32_partition = getattr(optimizer, "_use_compact_grad_partition_buffer", None)
    return uses_fp32_partition is not None and any(uses_fp32_partition(i) for i in range(len(optimizer.bit16_groups)))


def _differs(got: dict, want: dict, initial: dict) -> dict:
    """Per weight, the fraction of elements that differ; over all weights, that fraction and the difference's norm
    relative to how far the replicated run moved the weights."""
    fraction = {}
    differ = moved_total = 0.0
    differing = elements = 0
    for name, value in want.items():
        value, other, start = (t.cuda().float() for t in (value, got[name], initial[name]))
        count = int((other != value).sum())
        fraction[name] = count / value.numel()
        differing, elements = differing + count, elements + value.numel()
        difference = float((other - value).norm()) ** 2
        moved = float((value - start).norm()) ** 2
        differ, moved_total = differ + difference, moved_total + moved
    return {"fraction": fraction, "elements": differing / elements, "overall": (differ / moved_total) ** 0.5}


def _worker(rank: int, port: int, model_dir: str, results) -> None:
    import deepspeed
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD),
        LOCAL_RANK=str(rank),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend="nccl")
    replicated = _train(rank, model_dir, False)
    again = _train(rank, model_dir, False)
    sharded = _train(rank, model_dir, True)
    for run in (replicated, again, sharded):
        assert run["weights"][0].keys() == replicated["weights"][0].keys()
    initial = replicated.pop("initial")
    assert all(torch.equal(sharded["initial"][name], value) for name, value in initial.items())
    want = replicated.pop("weights")
    result = {
        "replicated": replicated,
        "again": {k: v for k, v in again.items() if k not in ("weights", "initial")},
        "sharded": {k: v for k, v in sharded.items() if k not in ("weights", "initial")},
        "steps": [
            {"sharded": _differs(got, ref, initial), "again": _differs(rep, ref, initial)}
            for got, rep, ref in zip(sharded["weights"], again["weights"], want)
        ],
    }
    results[rank] = result
    dist.barrier()
    dist.destroy_process_group()


def _spawn(worker, tmp_path) -> dict:
    from arctic_platform.testing_utils import get_unique_port_number
    from arctic_platform.testing_utils import reserve_free_port

    model_dir = str(tmp_path / "model")
    write_checkpoint(_GEMMA4, model_dir)
    port = reserve_free_port(get_unique_port_number() + 3, span=5, reuse_addr=True)
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(worker, args=(port, model_dir, results), nprocs=WORLD, join=True)
        return dict(results)


def _rel(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / abs(y) for x, y in zip(a, b))


def _worst(differs: dict) -> tuple[str, float]:
    return max(differs.items(), key=lambda item: item[1])


@pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs")
def test_sharded_hf_model_trains_as_the_replicated_model(tmp_path):
    results = _spawn(_worker, tmp_path)
    for rank, r in sorted(results.items()):
        rep, again, got = r["replicated"], r["again"], r["sharded"]
        # The first step's micro-batches run the same weights: their forwards are bit-identical.
        assert got["losses"][:GAS] == rep["losses"][:GAS], (rank, got["losses"], rep["losses"])
        assert min(rep["norms"]) > 1.0, rep["norms"]  # the clipping is active
        first = r["steps"][0]
        # The replicated run repeats itself, so only the sharding differs: the reduced gradients' summation order
        # and the shard sum's BF16 rounding, which the replicated run shares only when ZeRO stores BF16 gradients.
        assert again["losses"] == rep["losses"] and again["norms"] == rep["norms"], (rank, again, rep)
        assert all(s["again"]["overall"] == 0 for s in r["steps"]), rank
        fp32_gradients = rep["fp32_gradients"]
        assert got["fp32_gradients"] == fp32_gradients, rank
        first_norm = abs(got["norms"][0] - rep["norms"][0]) / rep["norms"][0]
        assert first_norm < (1e-4 if fp32_gradients else 1e-6), (rank, fp32_gradients, rep["norms"], got["norms"])
        # Seen after the first update: at most 1e-6 of any weight's elements (BF16 storage), 7e-4 of all (FP32).
        if fp32_gradients:
            assert first["sharded"]["elements"] < 1e-2, (rank, first["sharded"]["elements"])
        else:
            assert _worst(first["sharded"]["fraction"])[1] < 1e-4, (rank, _worst(first["sharded"]["fraction"]))
        # Later steps may drift (a last-bit norm difference under clipping), as two non-deterministic runs do.
        assert _rel(got["norms"], rep["norms"]) < 1e-3, (rank, rep["norms"], got["norms"])
        assert _rel(got["losses"], rep["losses"]) < 1e-3, (rank, rep["losses"], got["losses"])
        for step in r["steps"]:
            assert step["sharded"]["overall"] < 0.2, (rank, step["sharded"]["overall"])
        # The GPU holds the bytes the shards save less after the load; the allocator rounds each block up to 2 MiB.
        saved = got["full_bytes"] - got["shard_bytes"]
        assert abs((rep["loaded"] - got["loaded"]) - saved) <= 2 * 2**20 * got["modules"], (rep["loaded"], got)
