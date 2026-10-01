#!/usr/bin/env python3
"""Regression test: TP>1 broadcast weight-sync of a *fused, vLLM-format* MoE
weight (``experts.w13_weight`` / ``experts.w2_weight``) must load correctly on
the receiver.

Drives the real receiver seam ``WeightSyncExtension._load_batched`` (which is
what ``sync_weights_broadcast`` selects for non-quantized TP>1 vLLM-format sync)
with the fused, full MoE tensors the DSS trainer broadcasts, then checks each
rank's sharded params against vLLM's own per-expert ``weight_loader`` (oracle).

Failure modes this guards against (vLLM 0.26):
  * SILENT MISLOAD — ``model.load_weights`` does not match the vLLM-internal
    ``w13_weight`` / ``w2_weight`` names (its expert mapping uses HF names
    ``gate_proj`` / ``up_proj`` / ``down_proj``), so the tensors are dropped and
    the MoE experts keep STALE weights with no error raised. Before the fix this
    test is RED (params != oracle). After the fix (slice-before-copy + coverage
    guard) it is GREEN (params == oracle).

Parametrized over TP in {1, 2, 4, 8} (whatever the box has GPUs for):
  * TP=1 is the CONTROL — the receiver takes ``_load_direct`` (copies the full
    tensor into the param view); it must PASS both before and after the fix, and
    guards against TP=1 regressions.
  * TP>1 is the REPRO — the receiver takes ``_load_batched``; RED before the fix.

Run (dss env == vllm 0.26):
    /home/yak/miniconda3/envs/dss/bin/python tests/weight_sync/test_moe_tp_weight_sync.py
    # or restrict: PROBE_TPS=2,4 python tests/weight_sync/test_moe_tp_weight_sync.py
"""

import os
import socket
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "inference"))

NUM_EXPERTS = 8
HIDDEN = 128
INTERMEDIATE = 512          # must be divisible by max TP under test
TOP_K = 2
DTYPE = torch.bfloat16
SEED = 1234


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _ref_weights():
    g = torch.Generator().manual_seed(SEED)
    w1 = torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN, generator=g, dtype=torch.float32).to(DTYPE)
    w3 = torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN, generator=g, dtype=torch.float32).to(DTYPE)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=g, dtype=torch.float32).to(DTYPE)
    return w1, w3, w2


def _build_fused_moe(device, prefix):
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    return FusedMoE(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        params_dtype=DTYPE,
        renormalize=True,
        quant_config=None,
        prefix=prefix,
    ).to(device)


class _FakeBroadcastEngine:
    """Yields the fused, full MoE tensors the trainer broadcasts (vllm format)."""

    def __init__(self, items):
        self._items = items

    def receive_weights(self):
        for name, tensor in self._items:
            yield name, tensor


def _tiny_model_with_experts(experts_layer):
    """Minimal module tree ``model.layers.0.mlp.experts = <FusedMoE>`` whose
    ``load_weights`` uses vLLM's real AutoWeightsLoader — so the pre-fix path
    faithfully reproduces production's silent misload."""
    from torch import nn
    from vllm.model_executor.models.utils import AutoWeightsLoader

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = experts_layer

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Mlp()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights)

    return Model()


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

    err = None
    try:
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=tp_size)
            tp_rank = get_tensor_model_parallel_rank()

            w1, w3, w2 = _ref_weights()
            w1, w3, w2 = w1.to(device), w3.to(device), w2.to(device)

            # ---- ORACLE: vLLM per-expert loader on a reference layer ----------
            oracle = _build_fused_moe(device, "moe_oracle")
            re_oracle = getattr(oracle, "routed_experts", oracle)
            for e in range(NUM_EXPERTS):
                re_oracle.weight_loader(re_oracle.w13_weight, w1[e], "w13_weight", "w1", e)
                re_oracle.weight_loader(re_oracle.w13_weight, w3[e], "w13_weight", "w3", e)
                re_oracle.weight_loader(re_oracle.w2_weight, w2[e], "w2_weight", "w2", e)
            truth_w13 = re_oracle.w13_weight.data.clone()
            truth_w2 = re_oracle.w2_weight.data.clone()

            # ---- SYSTEM UNDER TEST: fresh (zeroed) experts wrapped in a model -
            experts = _build_fused_moe(device, "moe_sut")
            re_sut = getattr(experts, "routed_experts", experts)
            with torch.no_grad():
                re_sut.w13_weight.zero_()
                re_sut.w2_weight.zero_()
            model = _tiny_model_with_experts(experts)

            # The fused, FULL tensors the DSS trainer broadcasts (vllm names):
            full_w13 = torch.cat([w1, w3], dim=1).contiguous()   # [E, 2I, H]
            full_w2 = w2.contiguous()                            # [E, H, I]
            engine = _FakeBroadcastEngine([
                ("model.layers.0.mlp.experts.w13_weight", full_w13),
                ("model.layers.0.mlp.experts.w2_weight", full_w2),
            ])

            from arctic_inference.server.weight_sync.receiver import (
                WeightSyncExtension,
            )

            ext = SimpleNamespace(device=device)
            # Mirror sync_weights_broadcast's branch selection for
            # (weight_format="vllm", non-quantized):
            #   TP==1 -> _load_direct  (control: copies full tensor into the view)
            #   TP >1 -> _load_batched (repro:   silent misload on 0.26)
            if tp_size == 1:
                path = "direct"
                WeightSyncExtension._load_direct(ext, model, engine)
            else:
                path = "batched"
                WeightSyncExtension._load_batched(ext, model, engine)

            got_w13 = re_sut.w13_weight.data.clone()
            got_w2 = re_sut.w2_weight.data.clone()

            ret[rank] = {
                "tp_rank": tp_rank,
                "path": path,
                "err": None,
                "w13_match": bool(got_w13.shape == truth_w13.shape and torch.equal(got_w13, truth_w13)),
                "w2_match": bool(got_w2.shape == truth_w2.shape and torch.equal(got_w2, truth_w2)),
                "w13_stale_zero": bool(torch.count_nonzero(got_w13) == 0),
                "w2_stale_zero": bool(torch.count_nonzero(got_w2) == 0),
            }
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        ret[rank] = {"tp_rank": rank, "path": "?", "err": err, "w13_match": False,
                     "w2_match": False, "w13_stale_zero": None, "w2_stale_zero": None}


def _run_tp(tp_size: int) -> bool:
    port = _free_port()
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_worker, args=(tp_size, port, ret), nprocs=tp_size, join=True)

    ok = True
    role = "control" if tp_size == 1 else "repro"
    path = ret[sorted(ret.keys())[0]].get("path", "?") if ret else "?"
    print(f"\n--- TP={tp_size} ({role}, receiver path=_load_{path}) ---")
    for r in sorted(ret.keys()):
        d = ret[r]
        note = ""
        if d.get("w13_stale_zero"):
            note = "  <-- SILENT MISLOAD (params still zero/stale)"
        rank_ok = d["w13_match"] and d["w2_match"]
        print(f"  [rank {d['tp_rank']}] {'PASS' if rank_ok else 'FAIL'}  "
              f"w13=={d['w13_match']} w2=={d['w2_match']} err={d['err']}{note}")
        if not rank_ok:
            ok = False
    print(f"  => TP={tp_size} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    n = torch.cuda.device_count()
    tps = [int(x) for x in os.environ.get("PROBE_TPS", "1,2,4,8").split(",") if x]
    tps = [tp for tp in tps if tp <= n and INTERMEDIATE % tp == 0]
    if not tps:
        print(f"No runnable TP sizes (have {n} GPUs, INTERMEDIATE={INTERMEDIATE})")
        return 1

    print("=" * 72)
    print("MoE TP weight-sync regression (fused vllm-format experts.w13/w2)")
    import vllm
    print(f"vllm {vllm.__version__} | GPUs {n} | testing TP={tps}")
    print("=" * 72)

    all_ok = True
    for tp in tps:
        if not _run_tp(tp):
            all_ok = False

    print("\n" + "=" * 72)
    print("RESULT:", "PASSED — TP>1 fused MoE sync loads correctly"
          if all_ok else "FAILED — TP>1 fused MoE sync did NOT match oracle")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
