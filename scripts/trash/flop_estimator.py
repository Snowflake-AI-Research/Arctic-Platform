#!/usr/bin/env python3
"""
Simple FLOP estimator for transformer training runs.

Usage examples:
  python3 scripts/flop_estimator.py --params 80 --tokens 1e12
  python3 scripts/flop_estimator.py --params 80 --tokens 1e12 --moe
  python3 scripts/flop_estimator.py --params 80 --tokens 1e12 --moe --moe-experts 8 --moe-topk 2

All sizes are in billions for parameters (default) and raw tokens (no suffix).
Default assumptions:
  - per-token factor (forward+backward+overhead) = 6.0
  - fraction of params in FFN (where MoE is applied) = 0.66
  - GPU throughput for runtime estimate = 100 TFLOPS

Outputs both dense and effective (MoE) estimates when `--moe` is specified.
"""
import argparse
import math
import json

UNIT_SUFFIXES = [
    (1e24, 'Y'),
    (1e21, 'Z'),
    (1e18, 'E'),
    (1e15, 'P'),
    (1e12, 'T'),
    (1e9, 'G'),
    (1e6, 'M'),
    (1e3, 'k'),
]


def human(x):
    if x == 0:
        return '0'
    for v,s in UNIT_SUFFIXES:
        if abs(x) >= v:
            return f"{x/v:.3g}{s}"
    return f"{x:.3g}"


def parse_args():
    p = argparse.ArgumentParser(description='Estimate FLOPs for transformer training')
    p.add_argument('--params', type=float, default=80.0,
                   help='Model size in billions of parameters (default: 80 => 80e9)')
    p.add_argument('--per-token-factor', type=float, default=6.0,
                   help='FLOPs per token factor times active parameters (default: 6)')
    p.add_argument('--tokens', type=float, default=1e12,
                   help='Total number of training tokens (default: 1e12)')
    p.add_argument('--moe', action='store_true', help='Enable MoE-adjusted estimate')
    p.add_argument('--moe-experts', type=int, default=8, help='Number of experts (default: 8)')
    p.add_argument('--moe-topk', type=int, default=2, help='Top-k routing (default: 2)')
    p.add_argument('--ffn-fraction', type=float, default=0.66,
                   help='Fraction of parameters that belong to FFN/MoE (default: 0.66)')
    p.add_argument('--gpu-tflops', type=float, default=100.0,
                   help='Single GPU sustained TFLOPS for runtime estimate (default: 100 TFLOPS)')
    p.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs for runtime estimate')
    p.add_argument('--json', action='store_true', help='Print machine-readable JSON output')
    return p.parse_args()


def compute(params_b, per_token_factor, tokens, moe=False, moe_experts=8, moe_topk=2, ffn_fraction=0.66):
    # params_b provided in billions -> absolute
    P = params_b * 1e9
    # active fraction for MoE FFN region
    active_fraction = 1.0
    if moe:
        active_fraction = float(moe_topk) / float(moe_experts)
    # Effective active parameter count: dense params outside FFN + active fraction of FFN params
    P_eff = P * ((1.0 - ffn_fraction) + ffn_fraction * active_fraction)
    per_token_flops = per_token_factor * P_eff
    total_flops = per_token_flops * tokens
    return {
        'P': P,
        'P_eff': P_eff,
        'per_token_flops': per_token_flops,
        'total_flops': total_flops,
        'tokens': tokens,
        'active_fraction': active_fraction,
    }


def main():
    args = parse_args()
    dense = compute(args.params, args.per_token_factor, args.tokens, moe=False, ffn_fraction=args.ffn_fraction)
    if args.moe:
        moe_res = compute(args.params, args.per_token_factor, args.tokens, moe=True,
                          moe_experts=args.moe_experts, moe_topk=args.moe_topk, ffn_fraction=args.ffn_fraction)
    else:
        moe_res = None

    def fmt(res):
        return {
            'params': res['P'],
            'effective_params': res['P_eff'],
            'per_token_flops': res['per_token_flops'],
            'total_flops': res['total_flops'],
            'tokens': res['tokens'],
            'active_fraction': res['active_fraction'],
        }

    # runtime estimate
    def runtime_info(total_flops, gpu_tflops, num_gpus):
        # gpu_tflops is TFLOPS (1e12)
        per_gpu_flops_per_sec = gpu_tflops * 1e12
        seconds = total_flops / (per_gpu_flops_per_sec * max(1, num_gpus))
        days = seconds / 86400.0
        return {'seconds': seconds, 'days': days}

    dense_rt = runtime_info(dense['total_flops'], args.gpu_tflops, args.num_gpus)
    moe_rt = runtime_info(moe_res['total_flops'], args.gpu_tflops, args.num_gpus) if moe_res else None

    out = {
        'dense': fmt(dense),
        'moe': fmt(moe_res) if moe_res else None,
        'runtime': {
            'dense': dense_rt,
            'moe': moe_rt,
            'gpu_tflops': args.gpu_tflops,
            'num_gpus': args.num_gpus,
        },
        'assumptions': {
            'per_token_factor': args.per_token_factor,
            'ffn_fraction': args.ffn_fraction,
            'moe_experts': args.moe_experts,
            'moe_topk': args.moe_topk,
        }
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return

    # human output
    print('FLOP estimator results')
    print('----------------------')
    print(f"Model params: {args.params}B ({int(dense['P']):,} params)")
    print(f"Total tokens: {int(args.tokens):,}")
    print('Assumptions:')
    print(f"  per-token factor (fwd+back+overhead): {args.per_token_factor}")
    print(f"  ffn fraction of params: {args.ffn_fraction}")
    if args.moe:
        print(f"  MoE enabled: experts={args.moe_experts}, topk={args.moe_topk}, active_fraction={moe_res['active_fraction']:.3f}")
    else:
        print('  MoE enabled: no')

    print('\nDense estimate:')
    print(f"  effective params active per token: {int(dense['P_eff']):,} ({dense['P_eff']/dense['P']:.3g} of total)")
    print(f"  per-token FLOPs: {dense['per_token_flops']:.3g} ({human(dense['per_token_flops'])})")
    print(f"  total FLOPs: {dense['total_flops']:.3g} ({human(dense['total_flops'])})")
    print(f"  runtime @ {args.gpu_tflops} TFLOPS on {args.num_gpus} GPU(s): {dense_rt['days']:.2f} GPU-days")

    if moe_res:
        print('\nMoE-adjusted estimate (active params per token):')
        print(f"  effective params active per token: {int(moe_res['P_eff']):,} ({moe_res['P_eff']/dense['P']:.3g} of total)")
        print(f"  per-token FLOPs: {moe_res['per_token_flops']:.3g} ({human(moe_res['per_token_flops'])})")
        print(f"  total FLOPs: {moe_res['total_flops']:.3g} ({human(moe_res['total_flops'])})")
        print(f"  runtime @ {args.gpu_tflops} TFLOPS on {args.num_gpus} GPU(s): {moe_rt['days']:.2f} GPU-days")

    print('\nNote: human() prints scaled units (k, M, G, T, P, E, Z, Y).')

if __name__ == '__main__':
    main()
