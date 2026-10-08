#!/usr/bin/env python3
"""Regression test: TP>1 broadcast weight-sync of the *fused, vLLM-format*
Gated DeltaNet input projections (``in_proj_qkvz`` / ``in_proj_ba``) must load
correctly on the receiver.

These are ``MergedColumnParallelLinear`` modules.  On vLLM 0.26,
``model.load_weights`` re-applies the HF->vLLM stacked mapping to the
*already-fused* name; because ``in_proj_qkv`` is a prefix of ``in_proj_qkvz``
the substring rename yields ``in_proj_qkvzz`` (and ``in_proj_baa``), which are
not real params -> hard ``ValueError`` (HTTP 500 in the gateway).

This test drives the real ``_ShardAwareFusedWriter`` (registration + feed) on a
module tree whose ``linear_attn`` is a genuine ``GatedDeltaNetAttention``
subclass instance (so the writer's parent-type anchor fires), then checks each
rank's sharded param against the ground-truth per-shard narrow.

Run (dss env == vllm 0.26):
    python tests/weight_sync/test_gdn_tp_weight_sync.py
    # or restrict: PROBE_TPS=2,4 python tests/weight_sync/test_gdn_tp_weight_sync.py
"""

import os
import socket
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

HIDDEN = 2048
# Qwen3.5-MoE GDN geometry: qkvz = [key_dim, key_dim, value_dim, value_dim];
# ba = [num_v_heads, num_v_heads].  All divisible by TP in {1,2,4,8}.
KEY_DIM = 2048
VALUE_DIM = 4096
NUM_V_HEADS = 32
GLM_OUTPUT_SIZES = [256, 256, 256, 8, 16, 16]
GLM_REPLICATED_SHARDS = {4, 5}
INDEXER_OUTPUT_SIZES = [256, 8]
DTYPE = torch.bfloat16
SEED = 4321


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _expected_local(full: torch.Tensor, output_sizes, tp_size, tp_rank):
    """Ground truth: concat, over each output shard, of that shard's block
    narrowed to this rank's slice along the output dim."""
    pieces = []
    offset = 0
    for size in output_sizes:
        block = full[offset:offset + size, :]
        local = size // tp_size
        pieces.append(block[tp_rank * local:(tp_rank + 1) * local, :])
        offset += size
    return torch.cat(pieces, dim=0).contiguous()


def _expected_local_with_replicated(
    full: torch.Tensor, output_sizes, replicated_shards, tp_size, tp_rank
):
    pieces = []
    offset = 0
    for shard_id, size in enumerate(output_sizes):
        block = full[offset:offset + size, :]
        if shard_id in replicated_shards:
            pieces.append(block)
        else:
            local = size // tp_size
            pieces.append(block[tp_rank * local:(tp_rank + 1) * local, :])
        offset += size
    return torch.cat(pieces, dim=0).contiguous()


def _build_model(device):
    from torch import nn
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
    from vllm.models.glm5next.nvidia.kda import (
        _Glm5NextMergedColumnParallelLinear,
    )

    qkvz = MergedColumnParallelLinear(
        input_size=HIDDEN,
        output_sizes=[KEY_DIM, KEY_DIM, VALUE_DIM, VALUE_DIM],
        bias=False, prefix="linear_attn.in_proj_qkvz",
    ).to(device)
    ba = MergedColumnParallelLinear(
        input_size=HIDDEN,
        output_sizes=[NUM_V_HEADS, NUM_V_HEADS],
        bias=False, prefix="linear_attn.in_proj_ba",
    ).to(device)
    glm = _Glm5NextMergedColumnParallelLinear(
        input_size=HIDDEN,
        output_sizes=GLM_OUTPUT_SIZES,
        replicated_shard_ids=tuple(GLM_REPLICATED_SHARDS),
        tp_size=torch.distributed.get_world_size(),
        bias=False,
        prefix="linear_attn.in_proj_qkvbfg_a",
    ).to(device)
    indexer = MergedColumnParallelLinear(
        input_size=HIDDEN,
        output_sizes=INDEXER_OUTPUT_SIZES,
        bias=False,
        prefix="self_attn.indexer.wk_weights_proj",
    ).to(device)

    class FakeGDN(GatedDeltaNetAttention):
        # Bypass the heavy GatedDeltaNetAttention.__init__ (needs a full
        # config); we only need a real isinstance so the writer's parent-type
        # anchor recognises the linear-attn block.
        def __init__(self, qkvz, ba, glm):
            nn.Module.__init__(self)
            self.in_proj_qkvz = qkvz
            self.in_proj_ba = ba
            self.in_proj_qkvbfg_a = glm

        def get_state_shape(self):  # abstract in GatedDeltaNetAttention
            return ()

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = FakeGDN(qkvz, ba, glm)
            self.self_attn = nn.Module()
            self.self_attn.indexer = nn.Module()
            self.self_attn.indexer.wk_weights_proj = indexer

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def load_weights(self, weights):
            raise AssertionError(
                "model.load_weights must NOT be called for the fused GDN "
                f"families; got {[n for n, _ in weights]}"
            )

    return Model(), qkvz, ba, glm, indexer


def _worker(rank: int, tp_size: int, port: int, ret: dict):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    init_distributed_environment(
        world_size=tp_size, rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=rank, backend="nccl",
    )

    try:
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=tp_size)
            tp_rank = get_tensor_model_parallel_rank()

            model, qkvz, ba, glm, indexer = _build_model(device)
            with torch.no_grad():
                qkvz.weight.zero_()
                ba.weight.zero_()
                glm.weight.zero_()
                indexer.weight.zero_()

            g = torch.Generator().manual_seed(SEED)
            full_qkvz = torch.randn(
                2 * KEY_DIM + 2 * VALUE_DIM, HIDDEN, generator=g,
                dtype=torch.float32).to(DTYPE).to(device)
            full_ba = torch.randn(
                2 * NUM_V_HEADS, HIDDEN, generator=g,
                dtype=torch.float32).to(DTYPE).to(device)
            full_glm = torch.randn(
                sum(GLM_OUTPUT_SIZES), HIDDEN, generator=g,
                dtype=torch.float32).to(DTYPE).to(device)
            full_indexer = torch.randn(
                sum(INDEXER_OUTPUT_SIZES), HIDDEN, generator=g,
                dtype=torch.float32).to(DTYPE).to(device)

            from arctic_platform.inference.server.weight_sync.utils import (
                _ShardAwareFusedWriter,
            )
            writer = _ShardAwareFusedWriter(model, device)
            h_qkvz = writer.feed(
                "model.layers.0.linear_attn.in_proj_qkvz.weight", full_qkvz)
            h_ba = writer.feed(
                "model.layers.0.linear_attn.in_proj_ba.weight", full_ba)
            h_glm = writer.feed(
                "model.layers.0.linear_attn.in_proj_qkvbfg_a.weight", full_glm)
            h_indexer = writer.feed(
                "model.layers.0.self_attn.indexer.wk_weights_proj.weight",
                full_indexer,
            )

            exp_qkvz = _expected_local(
                full_qkvz, [KEY_DIM, KEY_DIM, VALUE_DIM, VALUE_DIM],
                tp_size, tp_rank)
            exp_ba = _expected_local(
                full_ba, [NUM_V_HEADS, NUM_V_HEADS], tp_size, tp_rank)
            exp_glm = _expected_local_with_replicated(
                full_glm, GLM_OUTPUT_SIZES, GLM_REPLICATED_SHARDS,
                tp_size, tp_rank)
            exp_indexer = _expected_local(
                full_indexer, INDEXER_OUTPUT_SIZES, tp_size, tp_rank)

            ret[rank] = {
                "tp_rank": tp_rank,
                "err": None,
                "handled": bool(h_qkvz and h_ba and h_glm and h_indexer),
                "qkvz_match": bool(qkvz.weight.data.shape == exp_qkvz.shape
                                   and torch.equal(qkvz.weight.data, exp_qkvz)),
                "ba_match": bool(ba.weight.data.shape == exp_ba.shape
                                 and torch.equal(ba.weight.data, exp_ba)),
                "glm_match": bool(glm.weight.data.shape == exp_glm.shape
                                  and torch.equal(glm.weight.data, exp_glm)),
                "indexer_match": bool(
                    indexer.weight.data.shape == exp_indexer.shape
                    and torch.equal(indexer.weight.data, exp_indexer)
                ),
                "qkvz_stale_zero": bool(
                    torch.count_nonzero(qkvz.weight.data) == 0),
            }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        ret[rank] = {"tp_rank": rank, "err": f"{type(e).__name__}: {e}",
                     "handled": False, "qkvz_match": False, "ba_match": False,
                     "glm_match": False,
                     "indexer_match": False,
                     "qkvz_stale_zero": None}


def _run_tp(tp_size: int) -> bool:
    port = _free_port()
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_worker, args=(tp_size, port, ret), nprocs=tp_size, join=True)

    ok = True
    print(f"\n--- TP={tp_size} (receiver path=_ShardAwareFusedWriter) ---")
    for r in sorted(ret.keys()):
        d = ret[r]
        note = "  <-- NOT WRITTEN (still zero)" if d.get("qkvz_stale_zero") else ""
        rank_ok = (
            d["handled"] and d["qkvz_match"] and d["ba_match"]
            and d["glm_match"] and d["indexer_match"]
        )
        print(f"  [rank {d['tp_rank']}] {'PASS' if rank_ok else 'FAIL'}  "
              f"handled={d['handled']} qkvz=={d['qkvz_match']} "
              f"ba=={d['ba_match']} glm=={d['glm_match']} "
              f"indexer=={d['indexer_match']} "
              f"err={d['err']}{note}")
        if not rank_ok:
            ok = False
    print(f"  => TP={tp_size} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    n = torch.cuda.device_count()
    tps = [int(x) for x in os.environ.get("PROBE_TPS", "1,2,4,8").split(",") if x]
    tps = [tp for tp in tps if tp <= n and KEY_DIM % tp == 0
           and VALUE_DIM % tp == 0 and NUM_V_HEADS % tp == 0]
    if not tps:
        print(f"No runnable TP sizes (have {n} GPUs)")
        return 1

    print("=" * 72)
    print("GDN TP weight-sync regression (fused vllm-format in_proj_qkvz/ba)")
    import vllm
    print(f"vllm {vllm.__version__} | GPUs {n} | testing TP={tps}")
    print("=" * 72)

    all_ok = True
    for tp in tps:
        if not _run_tp(tp):
            all_ok = False

    print("\n" + "=" * 72)
    print("RESULT:", "PASSED — TP>1 fused GDN sync loads correctly"
          if all_ok else "FAILED — TP>1 fused GDN sync did NOT match oracle")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
