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
"""Gemma 4 and Qwen3.8-27B on the huggingface route with ``patches.dense_shard`` train as the replicated models do.

8 GPUs, one node: the shard group is the node (``dense_shard: true``), so each shard's expert-data-parallel group is
its own rank and ZeRO keeps the shard's whole FP32 master and moments there, the 1/8 of the flat FP32 state the
replicated run's ZeRO-2 keeps per rank. Random-init tiny models of the two real architectures, written as checkpoints of
their real classes from the real configs (gemma-4-31B-it, Qwen3.8-27B) with fewer layers, a smaller hidden size and
vocabulary, and loaded through the huggingface loader the training job uses (``dense_model_spec`` -> ``build_model``
on the host, then moved to the GPU) with its patches:

- Gemma 4 (``Gemma4ForConditionalGeneration``, text ``gemma4_text``): sliding layers of head size 256 and full layers
  of head size 512 with ``attention_k_eq_v``, a vision tower (trainable, no gradient in a text step), the embedding
  tied to the LM head (it stays whole) and per-layer ``layer_scalar`` buffers; Liger, full activation checkpointing,
  ``checkpoint_keep_attention``, the chunked LM head with Gemma's final logit softcap, SDPA (FlashAttention 3 has no
  head size 512; at SP 1 the job has no other head-512 kernel).
- Qwen3.8-27B (a ``Qwen3_5ForConditionalGeneration`` checkpoint, which the loader builds as the text-only
  ``Qwen3_5ForCausalLM``, text ``qwen3_5_text``): GatedDeltaNet linear attention with full attention every 4th layer,
  an untied LM head (the embedding is sharded); the fp32 parity forward and the fused GatedDeltaNet the loader
  installs, full activation checkpointing, ``checkpoint_keep_attention``, the chunked LM head, FlashAttention 3, and
  packed rows (``position_ids`` restarting per row, the ``cu_seq_lens`` / ``seq_idx`` kwargs dss passes at SP 1).

DeepSpeed ZeRO-2 in BF16 with FP32 masters, AdamW (lr 1e-4, weight decay), gradient clipping at 1.0, two
micro-batches per step, three steps, built by ``initialize_training_engine`` as the job does; torch's deterministic
algorithms. ZeRO runs without ``overlap_comm``, as the job does. The replicated model runs twice, then the sharded
one. The first step's forwards are bit-identical (the gathered weights are the replicated ones), and so are its local
gradients. Each reduced gradient element is the FP32 sum of the node's eight partials, in another summation order.
The shard's sum is rounded once to BF16 in ``_reduce_scatter``. The replicated run's sum is rounded once to BF16 too
when ZeRO stores gradients in BF16, and stays FP32 when ZeRO keeps them in FP32 (DeepSpeed #8749 with FP32 gradient
accumulation, the default for BF16). Where the replicated run repeats itself (SDPA), the sharded run's first gradient
norm agrees to 1e-6 with BF16 storage, and to the BF16 bound of test_dense_shard.py (1e-4) with FP32 storage. Later
steps drift by no more than two runs of a non-deterministic replicated model do. Gradient clipping turns a last-bit
difference in the global norm, which sums ZeRO's parameter groups, into a rescaled update. With FlashAttention 3
(Qwen3.8's attention in the job, no deterministic backward at head size 256), the sharded run stays within twice the
replicated run's own spread. The patch's log line reports the bytes it saves, which the GPU's allocated memory
confirms.

Before any step, the sharded model's weights-only HF export (``iter_model_weights``, as dss writes a weights-only
checkpoint: persistent buffers included) and its state dict (inside ``full_state_dict``) are the replicated model's,
bit for bit, under the same names in the same order, and rank 0 holds one sharded module's whole weights at a time.

    python -m pytest -q -s tests/model/test_dense_shard_hf.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re

import pytest
import torch
import torch.multiprocessing as mp

WORLD, TOKENS, GAS, STEPS = 8, 1024, 2, 3
VOCAB = 16384
# Qwen3.8's packed micro-batches: three rows of these lengths.
ROWS = (448, 320, 256)

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

# Qwen3.8-27B's config.json with 8 layers (64; full attention every 4th kept), hidden 1024 (5120), 4 query and 1 KV
# heads (24, 4) of size 256 (kept, with the partial rotary and the interleaved M-RoPE sections), 2 / 6 GatedDeltaNet
# key / value heads (16, 48; the 1:3 grouping kept) of size 128, MLP 3584 (17408), vocabulary 16384 (248320) and a
# one-layer vision tower.
_QWEN38 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "language_model_only": False,
    "image_token_id": VOCAB - 3,
    "video_token_id": VOCAB - 2,
    "vision_start_token_id": VOCAB - 5,
    "vision_end_token_id": VOCAB - 4,
    "tie_word_embeddings": False,
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 1,
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3584,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 6,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 4096,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 4,
        "num_hidden_layers": 8,
        "num_key_value_heads": 1,
        "output_gate_type": "swish",
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": VOCAB,
    },
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 1,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "model_type": "qwen3_5",
        "num_heads": 2,
        "num_position_embeddings": 64,
        "out_hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

# (config, attention, Liger, packed rows). The workers run torch's deterministic algorithms, under which SDPA's
# backward is deterministic; FlashAttention 3 (the job's attention for Qwen3.8) has no deterministic backward at head
# size 256, so "qwen38" repeats with SDPA as "qwen38-sdpa" to compare the sharding alone.
CASES = {
    "gemma4": (_GEMMA4, "sdpa", True, False),
    "qwen38": (_QWEN38, "flash_attention_3", False, True),
    "qwen38-sdpa": (_QWEN38, "sdpa", False, True),
}
DETERMINISTIC = {"gemma4", "qwen38-sdpa"}
LOG_LINE = re.compile(
    r"dense_shard: trainable weights of (\d+) modules sharded over (\d+) GPUs of the node: "
    r"([\d.]+) GiB -> ([\d.]+) GiB per GPU"
)


def write_checkpoint(config: dict, path: str) -> None:
    """A random-init checkpoint of ``config``'s class, every parameter (norms included) away from its init value, and
    as in the published Gemma 4 checkpoint its ``layer_scalar`` buffers away from 1 and its key norms at 0.12 (head
    size 256) and 0.06 (512): with unit key norms the attention logits (Gemma 4 does not scale them) have a standard
    deviation of 16-23, every softmax is near one-hot, and a one-ulp change of a weight moves the next step's gradient
    norm by percents."""
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


def build_model(model_dir: str, case: str, shard: bool):
    """The model as the training job builds it: the huggingface loader with the job's patches, on the host, then
    moved to this rank's GPU."""
    from arctic_platform.model import build_model as build
    from arctic_platform.model import dense_model_spec

    _, attention, liger, _ = CASES[case]
    spec = dense_model_spec(
        model_path=model_dir,
        dtype="bfloat16",
        attn_implementation=attention,
        sequence_parallel=1,
        liger=liger,
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


def micro_batches(rank: int, case: str, steps: int) -> list[dict]:
    """Each rank's micro-batches, the same for every run: ``[1, TOKENS]`` token ids and next-token labels; Qwen3.8's
    packed as ``ROWS`` with the varlen kwargs dss derives for an SP-1 call (``_derive_varlen_model_kwargs``)."""
    generator = torch.Generator().manual_seed(1000 + rank)
    packed = CASES[case][3]
    out = []
    for _ in range(steps * GAS):
        ids = torch.randint(2, VOCAB - 8, (1, TOKENS), generator=generator)
        batch = {"input_ids": ids, "labels": ids.roll(-1, dims=1)}
        if packed:
            lens = torch.tensor(ROWS)
            cu = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).to(torch.int32)])
            batch["position_ids"] = torch.cat([torch.arange(n) for n in ROWS])[None]
            batch.update(
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=max(ROWS),
                max_length_k=max(ROWS),
                seq_idx=torch.repeat_interleave(torch.arange(len(ROWS), dtype=torch.int32), lens)[None],
            )
        else:
            batch["position_ids"] = torch.arange(TOKENS)[None]
        out.append(batch)
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


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _train(rank: int, model_dir: str, case: str, shard: bool) -> dict:
    from arctic_platform.model.implementations.moe.logging_utils import get_logger

    lines = _Lines()
    get_logger().addHandler(lines)
    # Free the previous run's engine and model first. Their reference cycles otherwise release GPU memory whenever
    # Python's collector next runs, which can be during this load. Seen: one rank's sharded load measured -700 MiB,
    # so the saved-bytes check below missed by 982 MiB.
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    try:
        model = build_model(model_dir, case, shard)
    finally:
        get_logger().removeHandler(lines)
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated() - before
    sharded = sharded_modules(model)
    assert bool(sharded) == shard
    engine = build_engine(model, rank, model_dir)
    assert engine.has_moe_layers == shard  # the shards are reduced over their expert-data-parallel groups
    fp32_gradients = zero_keeps_fp32_gradients(engine.optimizer)
    initial = full_weights(model)
    losses, norms, weights, peaks = [], [], [], []
    batches = micro_batches(rank, case, STEPS)
    for step in range(STEPS):
        torch.cuda.reset_peak_memory_stats()
        for batch in batches[step * GAS : (step + 1) * GAS]:
            loss = forward_loss(engine, batch)
            engine.backward(loss)
            engine.step()
            losses.append(loss.item())
        norms.append(float(engine.get_global_grad_norm()))
        peaks.append(torch.cuda.max_memory_allocated() - before)
        weights.append(full_weights(model))
    out = {
        "losses": losses,
        "norms": norms,
        "initial": initial,
        "weights": weights,
        "loaded": loaded,
        "peak": max(peaks),
        "log": [line for line in lines.lines if line.startswith("dense_shard:")],
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
    """Whether ZeRO keeps the reduced gradients of the BF16 parameters in FP32. DeepSpeed #8749 ("ZeRO2: preserve fp32
    grad accumulation precision in partition storage") does so when gradient accumulation is FP32, which
    build_deepspeed_config sets for BF16. Before #8749, or with BF16 accumulation, ZeRO stores them in BF16. The probe
    reads #8749's own test of that, a method of the optimizer, and writes nothing."""
    uses_fp32_partition = getattr(optimizer, "_use_compact_grad_partition_buffer", None)
    return uses_fp32_partition is not None and any(uses_fp32_partition(i) for i in range(len(optimizer.bit16_groups)))


def _differs(got: dict, want: dict, initial: dict) -> dict:
    """Over every weight: the fraction of BF16 elements that differ, and the difference's norm relative to how far the
    replicated run moved the weight (0 for a weight it did not move); and both over all weights together."""
    fraction, relative = {}, {}
    differ = moved_total = 0.0
    differing = elements = 0
    for name, value in want.items():
        value, other, start = (t.cuda().float() for t in (value, got[name], initial[name]))
        count = int((other != value).sum())
        fraction[name] = count / value.numel()
        differing, elements = differing + count, elements + value.numel()
        difference = float((other - value).norm()) ** 2
        moved = float((value - start).norm()) ** 2
        relative[name] = (difference / moved) ** 0.5 if moved > 0 else 0.0
        differ, moved_total = differ + difference, moved_total + moved
    return {
        "fraction": fraction,
        "relative": relative,
        "elements": differing / elements,
        "overall": (differ / moved_total) ** 0.5,
    }


def _worker(rank: int, port: int, model_dir: str, case: str, results) -> None:
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
    replicated = _train(rank, model_dir, case, False)
    again = _train(rank, model_dir, case, False)
    sharded = _train(rank, model_dir, case, True)
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


def _export(rank: int, model_dir: str, case: str, shard: bool) -> dict:
    """Engine built, before any step: the weights-only HF export as dss writes it (rank 0 keeps a host copy of each
    tensor and drops the tensor), the GPU memory it adds on rank 0, the weight sync's export, and the state dict."""
    from arctic_platform.model import iter_model_weights
    from arctic_platform.model.implementations.moe import dense_shard

    model = build_model(model_dir, case, shard)
    engine = build_engine(model, rank, model_dir)
    units = [sum(b.total * b.shard.element_size() for b in s.buckets) for s in sharded_modules(model)]
    out = {"units": units}
    for label, kwargs in (("checkpoint", {"include_skipped": True}), ("sync", {"checkpoint_dir": model_dir})):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        exported = []
        for item in iter_model_weights(model, "hf", is_master=rank == 0, is_zero3=False, **kwargs):
            exported.append((item[0], item[1].detach().cpu()))
            del item
        torch.cuda.synchronize()
        out[label] = exported
        out[label + "_peak"] = torch.cuda.max_memory_allocated() - base
    with dense_shard.full_state_dict(model):
        out["state"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    engine.destroy()
    del engine, model
    torch.cuda.empty_cache()
    return out


def _export_worker(rank: int, port: int, model_dir: str, case: str, results) -> None:
    import deepspeed
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend="nccl")
    replicated = _export(rank, model_dir, case, False)
    sharded = _export(rank, model_dir, case, True)
    for label in ("checkpoint", "sync"):
        assert bool(replicated[label]) == (rank == 0) and bool(sharded[label]) == (rank == 0), label
        assert [name for name, _ in sharded[label]] == [name for name, _ in replicated[label]], label
        for (name, got), (_, want) in zip(sharded[label], replicated[label]):
            assert got.dtype == want.dtype and torch.equal(got, want), (label, name)
    assert list(sharded["state"]) == list(replicated["state"])
    assert not any("dense_shard" in key for key in sharded["state"])
    for key, want in replicated["state"].items():
        assert sharded["state"][key].dtype == want.dtype and torch.equal(sharded["state"][key], want), key
    results[rank] = {
        "names": {label: [name for name, _ in replicated[label]] for label in ("checkpoint", "sync")},
        "keys": list(replicated["state"]),
        "peaks": {label: (replicated[label + "_peak"], sharded[label + "_peak"]) for label in ("checkpoint", "sync")},
        "units": sharded["units"],
    }
    dist.barrier()
    dist.destroy_process_group()


def _spawn(worker, tmp_path, case: str) -> dict:
    from arctic_platform.testing_utils import get_unique_port_number
    from arctic_platform.testing_utils import reserve_free_port

    model_dir = str(tmp_path / "model")
    write_checkpoint(CASES[case][0], model_dir)
    port = reserve_free_port(get_unique_port_number() + 3, span=5, reuse_addr=True)
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(worker, args=(port, model_dir, case, results), nprocs=WORLD, join=True)
        return dict(results)


def _rel(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / abs(y) for x, y in zip(a, b))


def _worst(differs: dict) -> tuple[str, float]:
    return max(differs.items(), key=lambda item: item[1])


@pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs")
@pytest.mark.parametrize("case", list(CASES))
def test_sharded_hf_model_trains_as_the_replicated_model(case, tmp_path):
    results = _spawn(_worker, tmp_path, case)
    r = results[0]
    rep, again, got = r["replicated"], r["again"], r["sharded"]
    print(f"\n{case}: losses replicated {rep['losses']}\n{case}: losses sharded    {got['losses']}")
    print(f"{case}: grad norms replicated {rep['norms']} again {again['norms']} sharded {got['norms']}")
    print(f"{case}: log {got['log']}")
    print(
        f"{case}: GPU memory after the load {rep['loaded'] / 2**20:.1f} -> {got['loaded'] / 2**20:.1f} MiB; step "
        f"peak {rep['peak'] / 2**20:.0f} -> {got['peak'] / 2**20:.0f} MiB; sharded modules {got['modules']}, "
        f"{got['full_bytes'] / 2**20:.2f} MiB -> {got['shard_bytes'] / 2**20:.2f} MiB per GPU"
    )
    loss_spread = max(_rel(r["again"]["losses"], r["replicated"]["losses"]) for r in results.values())
    for rank, r in sorted(results.items()):
        rep, again, got = r["replicated"], r["again"], r["sharded"]
        line = "; ".join(
            f"step {i + 1}: elements differ {s['sharded']['elements']:.2e} (again {s['again']['elements']:.2e}), "
            f"most in {_worst(s['sharded']['fraction'])} (again {_worst(s['again']['fraction'])[1]:.2e}), "
            f"weight diff/update {s['sharded']['overall']:.2e} (again "
            f"{s['again']['overall']:.2e}), worst {_worst(s['sharded']['relative'])} (again "
            f"{_worst(s['again']['relative'])[1]:.2e})"
            for i, s in enumerate(r["steps"])
        )
        print(
            f"rank {rank}: loss rel {_rel(got['losses'], rep['losses']):.2e} (again "
            f"{_rel(again['losses'], rep['losses']):.2e}); norm rel {_rel(got['norms'], rep['norms']):.2e} (again "
            f"{_rel(again['norms'], rep['norms']):.2e}); {line}"
        )
        # The first step's micro-batches run the same weights: their forwards are bit-identical.
        assert got["losses"][:GAS] == rep["losses"][:GAS], (rank, got["losses"], rep["losses"])
        assert min(rep["norms"]) > 1.0, rep["norms"]  # the clipping is active
        first = r["steps"][0]
        if case in DETERMINISTIC:
            # The replicated run repeats itself bit for bit, so what differs is the sharding's alone. The first
            # step's local gradients are bit-identical. Each element's reduced gradient is the FP32 sum of the
            # node's eight partials, in another summation order: the reduce-scatter's chunk owners are the shards',
            # not ZeRO's partitions. A parameter that stays whole (Gemma 4's tied embedding) is reduced by ZeRO in
            # both runs, but in another bucket layout. The shard's sum is rounded once to BF16 in _reduce_scatter.
            # When ZeRO stores gradients in BF16, it rounds the replicated run's sum once too, so the two runs
            # differ only where two summation orders round an element differently. With DeepSpeed #8749 and FP32
            # gradient accumulation (zero_keeps_fp32_gradients), ZeRO keeps the replicated run's sum in FP32. The
            # sharded run alone then carries one BF16 rounding, and the first step gets the BF16 bounds of
            # test_dense_shard.py. A missing or misplaced rank's partial changes an element's gradient by about an
            # eighth, and the first norm by far more than either bound.
            assert again["losses"] == rep["losses"] and again["norms"] == rep["norms"], (rank, again, rep)
            assert all(s["again"]["overall"] == 0 for s in r["steps"]), rank
            fp32_gradients = rep["fp32_gradients"]
            assert got["fp32_gradients"] == fp32_gradients, rank
            first_norm = abs(got["norms"][0] - rep["norms"][0]) / rep["norms"][0]
            assert first_norm < (1e-4 if fp32_gradients else 1e-6), (rank, fp32_gradients, rep["norms"], got["norms"])
            # Adam's first update is about lr * sign(g). With BF16 storage, a weight differs only where two summation
            # orders round a gradient differently; seen: at most 1e-6 of any weight's elements. With FP32 storage, it
            # also differs where the extra rounding flipped a nearly cancelling gradient; seen: 7e-4 of all
            # elements. As in the FlashAttention 3 case, that bound is over all elements: a few flipped elements of
            # a 256-element norm weight are 2% of it.
            if fp32_gradients:
                assert first["sharded"]["elements"] < 1e-2, (rank, first["sharded"]["elements"])
            else:
                assert _worst(first["sharded"]["fraction"])[1] < 1e-4, (rank, _worst(first["sharded"]["fraction"]))
            # Later steps need not stay bit-identical. The global gradient norm sums ZeRO's parameter groups, which
            # the shards change, and the gradients it sums can differ in the last bits. So the norm can be a few
            # FP32 ulps apart. With BF16 storage, Gemma 4's first norm was one ulp apart in every run on torch 2.14:
            # its tied embedding's reduced gradient differs. Clipping at 1.0 then scales every gradient by a slightly
            # different factor, and Adam's moments carry the difference forward. With clipping off, Gemma 4's
            # losses and weights stayed bit-identical through all three steps. Qwen3.8 shards its embedding, so
            # with BF16 storage it stays bit-identical with clipping on.
            # Two runs of the replicated model whose backward is not deterministic (FlashAttention 3, below) drift
            # as far: 9.5e-2 of the update by the third step, 6e-4 in the norms, 2e-4 in the losses.
            assert _rel(got["norms"], rep["norms"]) < 1e-3, (rank, rep["norms"], got["norms"])
            assert _rel(got["losses"], rep["losses"]) < 1e-3, (rank, rep["losses"], got["losses"])
            for step in r["steps"]:
                assert step["sharded"]["overall"] < 0.2, (rank, step["sharded"]["overall"])
        else:
            # A second replicated run differs from the first (FlashAttention 3's backward accumulates in a varying
            # order); the sharded run stays within the larger of a fixed bound and twice that spread. Adam's first
            # update is about lr * sign(g): a BF16 weight differs where a gradient that nearly cancels flipped sign,
            # or the update crossed a BF16 rounding boundary (seen: 0.09% of the elements, both runs alike).
            spread = abs(again["norms"][0] - rep["norms"][0]) / rep["norms"][0]
            assert abs(got["norms"][0] - rep["norms"][0]) / rep["norms"][0] < max(1e-4, 2 * spread), rank
            assert first["sharded"]["elements"] < max(1e-2, 2 * first["again"]["elements"]), (rank, first)
            assert _rel(got["norms"], rep["norms"]) < max(1e-3, 2 * _rel(again["norms"], rep["norms"])), rank
            # Each rank's losses are its own micro-batches': compared with the spread over all ranks.
            assert _rel(got["losses"], rep["losses"]) < max(1e-4, 2 * loss_spread), (rank, loss_spread)
            for step in r["steps"]:
                assert step["sharded"]["overall"] < max(0.1, 2 * step["again"]["overall"]), (rank, step)
        # The log line reports the bytes the shards save, and the GPU holds that much less after the load.
        assert len(got["log"]) == 1 and not rep["log"], (got["log"], rep["log"])
        modules, group, full_gib, kept_gib = LOG_LINE.fullmatch(got["log"][0]).groups()
        assert (int(modules), int(group)) == (got["modules"], WORLD)
        assert float(full_gib) == round(got["full_bytes"] / 2**30, 2), (full_gib, got["full_bytes"])
        assert float(kept_gib) == round(got["shard_bytes"] / 2**30, 2), (kept_gib, got["shard_bytes"])
        saved = got["full_bytes"] - got["shard_bytes"]
        # The caching allocator rounds each shard's block up (to 2 MiB for the large ones).
        assert abs((rep["loaded"] - got["loaded"]) - saved) <= 2 * 2**20 * got["modules"], (rep["loaded"], got)


@pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs")
@pytest.mark.parametrize("case", ["gemma4", "qwen38"])
def test_sharded_hf_model_exports_and_saves_the_replicated_weights(case, tmp_path):
    """The HF export as a weights-only checkpoint writes it (persistent buffers included) and as the weight sync
    streams it, and the state dict: the replicated model's, bit for bit, under the same names in the same order; rank 0
    holds one sharded module's whole weights at a time beyond what the replicated export holds."""
    results = _spawn(_export_worker, tmp_path, case)
    r = results[0]
    names, units = r["names"], r["units"]
    print(
        f"\n{case}: {len(names['checkpoint'])} checkpoint tensors ({len(names['sync'])} synced), {len(r['keys'])} "
        f"state dict keys; rank 0 export peaks (replicated, sharded) {r['peaks']} B; largest sharded module "
        f"{max(units)} B of {sum(units)} B"
    )
    buffers = [name for name in names["checkpoint"] if name not in names["sync"]]
    print(f"{case}: buffers in the checkpoint export {buffers}")
    if case == "gemma4":
        assert {f"model.language_model.layers.{i}.layer_scalar" for i in range(6)} <= set(buffers), buffers
        assert "lm_head.weight" not in names["checkpoint"]  # tied: save_pretrained leaves it out too
    else:  # the text-only model exports under the checkpoint's names
        assert all(name.startswith(("model.language_model.", "lm_head.")) for name in names["checkpoint"])
    for label, (replicated, sharded) in r["peaks"].items():
        # The caching allocator rounds each block up (2 MiB for the large ones).
        assert sharded - replicated <= max(units) + 4 * 2**20, (label, r["peaks"], units)
        assert sharded - replicated < sum(units), (label, r["peaks"], units)
