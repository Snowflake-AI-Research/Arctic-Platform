#!/usr/bin/env python3
"""
CLI wrapper for scripts/flop_estimator.py with convenient presets for Qwen3.

Presets:
  qwen-dense   - 80B params, 1e12 tokens, dense
  qwen-moe     - 80B params, 1e12 tokens, MoE (8 experts, top-2)

This script simply calls the existing estimator script with sensible flags.
"""
import argparse
import shlex
import subprocess
import sys

ESTIMATOR = 'scripts/flop_estimator.py'

PRESETS = {
    'qwen-dense': {
        'params': '80',
        'tokens': '1e12',
        'per-token-factor': '6.0',
    },
    'qwen-moe': {
        'params': '80',
        'tokens': '1e12',
        'per-token-factor': '6.0',
        'moe': True,
        'moe-experts': '8',
        'moe-topk': '2',
        'ffn-fraction': '0.66',
    }
}


def build_command(args, preset_conf=None):
    cmd = [sys.executable, ESTIMATOR]
    if preset_conf:
        for k, v in preset_conf.items():
            if isinstance(v, bool) and v:
                cmd.append(f'--{k}')
            elif not isinstance(v, bool):
                cmd.append(f'--{k}')
                cmd.append(str(v))

    # override with explicit args
    if args.params is not None:
        cmd += ['--params', str(args.params)]
    if args.tokens is not None:
        cmd += ['--tokens', str(args.tokens)]
    if args.per_token_factor is not None:
        cmd += ['--per-token-factor', str(args.per_token_factor)]
    if args.moe:
        cmd.append('--moe')
    if args.moe_experts is not None:
        cmd += ['--moe-experts', str(args.moe_experts)]
    if args.moe_topk is not None:
        cmd += ['--moe-topk', str(args.moe_topk)]
    if args.ffn_fraction is not None:
        cmd += ['--ffn-fraction', str(args.ffn_fraction)]
    if args.gpu_tflops is not None:
        cmd += ['--gpu-tflops', str(args.gpu_tflops)]
    if args.num_gpus is not None:
        cmd += ['--num-gpus', str(args.num_gpus)]
    if args.json:
        cmd.append('--json')
    return cmd


def main():
    p = argparse.ArgumentParser(description='Wrapper CLI with presets for flop_estimator')
    p.add_argument('--preset', choices=list(PRESETS.keys()), help='Choose a preset configuration')
    p.add_argument('--params', type=float, help='Model size in billions of parameters')
    p.add_argument('--per-token-factor', type=float, help='Per-token factor')
    p.add_argument('--tokens', type=float, help='Total tokens')
    p.add_argument('--moe', action='store_true', help='Enable MoE')
    p.add_argument('--moe-experts', type=int, help='Number of MoE experts')
    p.add_argument('--moe-topk', type=int, help='MoE top-k routing')
    p.add_argument('--ffn-fraction', type=float, help='Fraction of params in FFN/MoE')
    p.add_argument('--gpu-tflops', type=float, help='GPU TFLOPS for runtime estimate')
    p.add_argument('--num-gpus', type=int, help='Number of GPUs for runtime estimate')
    p.add_argument('--json', action='store_true', help='Print JSON output (forwarded)')
    p.add_argument('--dry-run', action='store_true', help='Print the underlying command, do not execute')

    args = p.parse_args()

    preset_conf = None
    if args.preset:
        preset_conf = PRESETS[args.preset]

    cmd = build_command(args, preset_conf=preset_conf)

    if args.dry_run:
        print('Dry run:')
        print(' '.join(shlex.quote(x) for x in cmd))
        return

    # Execute
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f'Estimator exited with {e.returncode}', file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
