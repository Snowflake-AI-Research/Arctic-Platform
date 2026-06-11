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

#!/usr/bin/env python3
"""
Estimator for gated Deltanet training FLOPs.

Usage examples:
  python3 scripts/gated_deltanet_estimator.py --params 10 --tokens 1e12 --active-fraction 0.25

Model inputs:
  --params: model size in billions (default 10)
  --tokens: total training tokens (default 1e12)
  --per-token-factor: forward+back+overhead factor (default 6)
  --active-fraction: fraction of params active per token due to gating (0..1)
  --ffn-fraction: fraction of params in gated FFN region (optional; if provided with experts/topk, will compute active fraction from them)
  --moe-experts and --moe-topk: alternative way to compute active fraction for an MoE-style gating region
  --gpu-tflops, --num-gpus: optional runtime estimate

The script prints a human-readable summary and a compact JSON blob if --json is provided.
"""
import argparse
import json

UNIT_SUFFIXES = [
    (1e24, "Y"),
    (1e21, "Z"),
    (1e18, "E"),
    (1e15, "P"),
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
]


def human(x):
    if x == 0:
        return "0"
    for v, s in UNIT_SUFFIXES:
        if abs(x) >= v:
            return f"{x/v:.3g}{s}"
    return f"{x:.3g}"


def parse_args():
    p = argparse.ArgumentParser(description="Estimate FLOPs for gated Deltanet training")
    p.add_argument("--params", type=float, default=10.0, help="Model size in billions (default: 10)")
    p.add_argument("--tokens", type=float, default=1e12, help="Total training tokens (default: 1e12)")
    p.add_argument("--per-token-factor", type=float, default=6.0, help="Per-token FLOP factor (default: 6)")
    p.add_argument("--active-fraction", type=float, help="Directly specify fraction of params active per token (0..1)")
    p.add_argument("--ffn-fraction", type=float, default=0.66, help="Fraction of params in FFN region (default: 0.66)")
    p.add_argument("--moe-experts", type=int, help="If provided, compute active fraction as topk/experts")
    p.add_argument("--moe-topk", type=int, default=1, help="Top-k routing used for MoE (default 1)")
    p.add_argument("--gpu-tflops", type=float, help="GPU TFLOPS for runtime estimate (e.g., 100)")
    p.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for runtime estimate")
    p.add_argument("--json", action="store_true", help="Output JSON")
    return p.parse_args()


def compute(params_b, tokens, per_token_factor, active_fraction):
    P = params_b * 1e9
    P_eff = P * active_fraction
    per_token_flops = per_token_factor * P_eff
    total_flops = per_token_flops * tokens
    return {
        "P": P,
        "active_fraction": active_fraction,
        "P_eff": P_eff,
        "per_token_flops": per_token_flops,
        "total_flops": total_flops,
        "tokens": tokens,
    }


def runtime_estimate(total_flops, gpu_tflops, num_gpus):
    if gpu_tflops is None:
        return None
    per_gpu_flops_per_sec = gpu_tflops * 1e12
    seconds = total_flops / (per_gpu_flops_per_sec * max(1, num_gpus))
    return {"seconds": seconds, "days": seconds / 86400.0}


def main():
    args = parse_args()

    # Determine active fraction
    if args.active_fraction is not None:
        a = args.active_fraction
    elif args.moe_experts is not None:
        a = float(args.moe_topk) / float(args.moe_experts)
    else:
        # default: assume gating applies mainly to FFN; if no gating specified, default to 0.25
        a = 0.25

    # If ffn-fraction is given and moe_experts provided, consider that only ffn fraction has sparsity
    # But for a simple gated deltanet, we assume active_fraction applies to whole model unless user specifies otherwise.
    res = compute(args.params, args.tokens, args.per_token_factor, a)
    rt = runtime_estimate(res["total_flops"], args.gpu_tflops, args.num_gpus)

    if args.json:
        out = {
            "result": res,
            "runtime": rt,
            "assumptions": {"per_token_factor": args.per_token_factor, "ffn_fraction": args.ffn_fraction},
        }
        print(json.dumps(out, indent=2))
        return

    print("Gated Deltanet FLOP estimate")
    print("---------------------------")
    print(f"Model params: {args.params}B ({int(res['P']):,})")
    print(f"Total tokens: {int(res['tokens']):,}")
    print(f"Assumed active fraction (a): {res['active_fraction']:.4g}")
    print(
        f"Per-token FLOPs = per_token_factor * P_eff = {args.per_token_factor} * {int(res['P_eff']):,} ="
        f" {res['per_token_flops']:.3g} ({human(res['per_token_flops'])})"
    )
    print(f"Total FLOPs = per-token FLOPs * tokens = {res['total_flops']:.3g} ({human(res['total_flops'])})")
    if rt:
        print(f"Runtime estimate @ {args.gpu_tflops} TFLOPS on {args.num_gpus} GPU(s): {rt['days']:.2f} GPU-days")
    print("\nFormula used: total_FLOPs = per_token_factor * active_fraction * params * tokens")
    print("Assumed per_token_factor accounts for forward+backward+optimizer overhead (typical default = 6)")


if __name__ == "__main__":
    main()
