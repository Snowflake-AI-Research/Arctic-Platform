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
"""GLM-5.3 (glm_moe_dsa) with ``shard_dense`` trains as the replicated model does, through the loader.

8 GPUs as two "nodes" of 4 (``shard_dense_group_size=4``), EP8 over DeepEP, a random-init 3-layer model of the real
classes: sparse MLA attention with its DSA indexer (no gradient), a dense MLP layer, two MoE layers with a shared
expert and the router, an IndexShare layer without an indexer, the embedding and the chunked LM head of the job
(``lm_head.token_chunk_size``, no fused cross-entropy). Full activation checkpointing of layers 0 and 1 (layer 2 is
not recomputed), DeepSpeed ZeRO-2 in BF16 with FP32 masters, gradient clipping at 1.0, two micro-batches per step,
three steps, the optimizer built by ``initialize_training_engine``. The first step's forwards are bit-identical and its
gradient norms agree to the BF16 rounding of each node's partial sums (see test_dense_shard.py). The replicated run is
itself not deterministic (the sparse MLA backward adds dKV atomically): two invocations of it differed by 1.2e-5 in
the first gradient norm and 0.4-0.55% in the next two, as much as the sharded run differs from it.

The same model, engine built, before any step: the sharded model's HF export (as a weights-only checkpoint writes it,
GLM's own layer conversion included) and its state dict (inside ``full_state_dict``) are the replicated model's, bit
for bit, and the export holds one unit's whole dense weights at a time on rank 0.

    python -m pytest -q -s tests/model/test_dense_shard_glm.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.multiprocessing as mp

WORLD, NODE, TOKENS = 8, 4, 2048
GAS, STEPS = 2, 3

VOCAB = 4096
_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "dtype": "bfloat16",
    "vocab_size": VOCAB,
    "hidden_size": 1024,
    "intermediate_size": 1024,
    "moe_intermediate_size": 256,
    "num_hidden_layers": 3,
    "first_k_dense_replace": 1,
    "mlp_layer_types": ["dense", "sparse", "sparse"],
    "indexer_types": ["full", "full", "shared"],
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "kv_lora_rank": 512,
    "q_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "index_n_heads": 8,
    "index_head_dim": 128,
    "index_topk": 256,
    "n_routed_experts": 16,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": True,
    "max_position_embeddings": 8192,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "rms_norm_eps": 1e-5,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "tie_word_embeddings": False,
    "num_nextn_predict_layers": 0,
}


def _full_weights(model) -> dict[str, torch.Tensor]:
    """Every trainable weight whole, on the CPU: the sharded ones gathered (collective)."""
    from arctic_platform.model.implementations.moe import dense_shard

    out = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not dense_shard.is_dense_shard(p)}
    for module in model.modules():
        sharded = getattr(module, "_ap_dense_shards", None)
        if sharded is not None:
            out.update({name: t.cpu() for name, t in sharded.full_parameters().items()})
    return out


def _build_engine(rank: int, model_dir: str, shard: bool):
    """The model through the loader and its DeepSpeed engine, as the training job builds them."""
    import deepspeed.utils.groups as ds_groups

    from arctic_platform.common.deepspeed_engine import OptimizerSettings
    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine
    from arctic_platform.model.implementations.glm52 import load_glm_moe_dsa_model
    from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaOptions

    options = GlmMoeDsaOptions(
        seq_len=TOKENS,
        ep_comm_backend="deepep",
        fused_cross_entropy=False,
        fused_lm_head_token_chunk_size=512,
        ac_config={"mode": "full", "layers": [0, 1]},
        debug={"random_init": True, "init_seed": 7},
        shard_dense=shard,
        shard_dense_group_size=NODE,
    )
    model = load_glm_moe_dsa_model(
        model_name=model_dir,
        optimization_dtype="bfloat16",
        attn_implementation="sdpa",
        ep_size=WORLD,
        ep_group=ds_groups._get_expert_parallel_group(f"ep_size_{WORLD}"),
        options=options,
    )
    sharded = [m._ap_dense_shards for m in model.modules() if hasattr(m, "_ap_dense_shards")]
    assert bool(sharded) == shard
    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            world_size=WORLD,
            gradient_clipping=1.0,
            fp32_masters=True,
            ds_overrides={
                "gradient_accumulation_steps": GAS,
                "train_batch_size": GAS * WORLD,
                "zero_optimization": {"stage": 2, "overlap_comm": True},
            },
        )
    )
    engine, _, _ = initialize_training_engine(
        model,
        ds_config,
        optimizer=OptimizerSettings(lr=1e-3),
        ep_size=WORLD,
        mpu=None,
        model_path=model_dir,
        rank=rank,
    )
    torch.cuda.synchronize()
    return model, engine


def _train(rank: int, model_dir: str, shard: bool):
    from arctic_platform.model.implementations.moe import dense_shard

    model, engine = _build_engine(rank, model_dir, shard)
    dense = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
        if getattr(p, "allreduce", True) is not False or dense_shard.is_dense_shard(p)
    )
    generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    position_ids = torch.arange(TOKENS, device="cuda")[None]
    losses, norms, peaks, first = [], [], [], None
    for step in range(STEPS):
        torch.cuda.reset_peak_memory_stats()
        for _ in range(GAS):
            ids = torch.randint(2, VOCAB, (1, TOKENS), device="cuda", generator=generator)
            out = engine(input_ids=ids, position_ids=position_ids, labels=ids.roll(-1, dims=1))
            # Scaled so the gradient norm is above the clipping threshold.
            loss = -4 * out["logprobs"].float().mean()
            engine.backward(loss)
            engine.step()
            losses.append(loss.item())
        norms.append(float(engine.get_global_grad_norm()))
        peaks.append(torch.cuda.max_memory_allocated())
        if step == 0:
            first = _full_weights(model)
    with pytest.raises(RuntimeError, match="sharded over the node") if shard else _nullcontext():
        model.state_dict()
    engine.destroy()
    del engine, model
    torch.cuda.empty_cache()
    return {"losses": losses, "norms": norms, "dense": dense, "peak": max(peaks), "first": first}


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _export(rank: int, model_dir: str, shard: bool) -> dict:
    """With the engine built, before any step: the HF export as a weights-only checkpoint writes it (rank 0 keeps a
    host copy of each tensor and drops the tensor), the GPU memory it adds on rank 0, and the state dict (host)."""
    from arctic_platform.model import iter_model_weights
    from arctic_platform.model.implementations.moe import dense_shard

    model, engine = _build_engine(rank, model_dir, shard)
    units: dict[int, int] = {}  # the export units' whole dense bytes: each layer, and the rest together (-1)
    for sharded in (m._ap_dense_shards for m in model.modules() if hasattr(m, "_ap_dense_shards")):
        name = next(iter(sharded.members)).name
        index = int(name.split(".")[2]) if name.startswith("model.layers.") else -1
        units[index] = units.get(index, 0) + sum(b.total * b.shard.element_size() for b in sharded.buckets)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    exported = []
    for item in iter_model_weights(model, "hf", is_master=rank == 0, is_zero3=False, include_skipped=True):
        exported.append(_host_copy(item))
        del item
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    with dense_shard.full_state_dict(model) if shard else _nullcontext():
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    engine.destroy()
    del engine, model
    torch.cuda.empty_cache()
    return {"exported": exported, "peak": peak, "state": state, "units": units}


def _host_copy(item: tuple) -> tuple:
    name, tensor = item
    return name, tensor.detach().cpu()


def _worker(rank: int, port: int, model_dir: str, results) -> None:
    import deepspeed
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend="nccl")
    from arctic_platform.model.parallel.groups import setup_parallel_groups

    setup_parallel_groups(1, WORLD, world_size=WORLD, gpus_per_node=NODE)
    replicated = _train(rank, model_dir, False)
    sharded = _train(rank, model_dir, True)
    assert sharded["first"].keys() == replicated["first"].keys()
    first_differ = {
        name: float((sharded["first"][name] != want).float().mean()) for name, want in replicated["first"].items()
    }
    for run in (replicated, sharded):
        del run["first"]
    results[rank] = {"replicated": replicated, "sharded": sharded, "first_differ": first_differ}
    dist.barrier()
    dist.destroy_process_group()


def _export_worker(rank: int, port: int, model_dir: str, results) -> None:
    import deepspeed
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(dist_backend="nccl")
    from arctic_platform.model.parallel.groups import setup_parallel_groups

    setup_parallel_groups(1, WORLD, world_size=WORLD, gpus_per_node=NODE)
    replicated = _export(rank, model_dir, False)
    sharded = _export(rank, model_dir, True)
    assert bool(replicated["exported"]) == (rank == 0) and bool(sharded["exported"]) == (rank == 0)
    assert [name for name, _ in sharded["exported"]] == [name for name, _ in replicated["exported"]]
    for (name, got), (_, want) in zip(sharded["exported"], replicated["exported"]):
        assert got.dtype == want.dtype and torch.equal(got, want), name
    assert list(sharded["state"]) == list(replicated["state"])
    assert not any("dense_shard" in key for key in sharded["state"])
    for key, want in replicated["state"].items():
        assert sharded["state"][key].dtype == want.dtype and torch.equal(sharded["state"][key], want), key
    results[rank] = {
        "tensors": len(replicated["exported"]),
        "keys": len(replicated["state"]),
        "peaks": (replicated["peak"], sharded["peak"]),
        "units": sharded["units"],
    }
    dist.barrier()
    dist.destroy_process_group()


def _rel(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / abs(y) for x, y in zip(a, b))


@pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs")
def test_glm_with_sharded_dense_weights_trains_as_the_replicated_model(tmp_path):
    pytest.importorskip("deep_ep")
    from arctic_platform.testing_utils import get_unique_port_number
    from arctic_platform.testing_utils import reserve_free_port

    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    port = reserve_free_port(get_unique_port_number() + 3, span=5)
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_worker, args=(port, str(tmp_path), results), nprocs=WORLD, join=True)
        results = dict(results)
    r = results[0]
    print(f"\nlosses replicated {r['replicated']['losses']}\nlosses sharded    {r['sharded']['losses']}")
    print(f"grad norms replicated {r['replicated']['norms']} sharded {r['sharded']['norms']}")
    print(
        f"dense bytes per GPU {r['replicated']['dense']} -> {r['sharded']['dense']} "
        f"({r['sharded']['dense'] / r['replicated']['dense']:.3f}); step peak {r['replicated']['peak'] / 2**20:.0f} "
        f"-> {r['sharded']['peak'] / 2**20:.0f} MiB"
    )
    for rank, r in results.items():
        want, got = r["replicated"], r["sharded"]
        worst = max(r["first_differ"].items(), key=lambda item: item[1])
        print(
            f"rank {rank}: loss rel {_rel(got['losses'], want['losses']):.2e} norm rel "
            f"{_rel(got['norms'], want['norms']):.2e}; after step 1 the most differing elements: {worst}"
        )
        # The first step's micro-batches run the same weights: their losses are bit-identical, and so is every
        # rank's local gradient. Only the dense reduction differs, by the BF16 rounding of each node's FP32 partial
        # sum (test_dense_shard.py: nothing in FP32, ~1e-5 of the norm in BF16).
        assert got["losses"][:GAS] == want["losses"][:GAS], rank
        assert min(want["norms"]) > 1.0, want["norms"]  # the clipping is active
        assert abs(got["norms"][0] - want["norms"][0]) / want["norms"][0] < 1e-4, (rank, want["norms"], got["norms"])
        # Adam's first update is lr * sign(g) wherever |g| >> eps: a BF16 weight differs where that rounding flipped
        # the sign of a gradient that nearly cancels, or moved the update across a BF16 rounding boundary. Seen:
        # 9.3e-3 of the elements of q_b_proj, less elsewhere; a missing or misplaced partial sum flips about half.
        assert worst[1] < 2e-2, worst
        # Later steps start from those weights: a few flipped elements move some tokens to other experts, so the
        # runs drift apart as two runs of the replicated model do (0.4-0.55% in the norms, 2.8e-4 in the losses).
        assert _rel(got["norms"], want["norms"]) < 5e-2, (rank, want["norms"], got["norms"])
        assert _rel(got["losses"], want["losses"]) < 1e-2, (rank, want["losses"], got["losses"])
        assert got["dense"] < 0.26 * want["dense"], (want["dense"], got["dense"])


@pytest.mark.skipif(torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs")
def test_glm_with_sharded_dense_weights_exports_and_saves_the_replicated_weights(tmp_path):
    """The HF export (as a weights-only checkpoint writes it) and the state dict of the sharded model are the
    replicated model's, bit for bit, under the same names in the same order; rank 0 holds one export unit's whole
    dense weights at a time beyond what the replicated export holds."""
    pytest.importorskip("deep_ep")
    from arctic_platform.testing_utils import get_unique_port_number
    from arctic_platform.testing_utils import reserve_free_port

    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    port = reserve_free_port(get_unique_port_number() + 3, span=5)
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_export_worker, args=(port, str(tmp_path), results), nprocs=WORLD, join=True)
        results = dict(results)
    r = results[0]
    (replicated, sharded), units = r["peaks"], r["units"]
    print(
        f"\n{r['tensors']} exported tensors, {r['keys']} state dict keys; rank 0 export peak {replicated / 2**20:.1f} "
        f"MiB replicated, {sharded / 2**20:.1f} MiB sharded; export units {units} B"
    )
    # The caching allocator rounds each block up (2 MiB for the large ones).
    assert sharded - replicated <= max(units.values()) + 4 * 2**20, (r["peaks"], units)
    assert sharded - replicated < sum(units.values()), (r["peaks"], units)
