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

"""reference / target / summarize for one catalog job config on a local cortex-training gateway.

reference: one HF single-GPU forward-backward on the fixture, optionally through a target's exported adapter.
target:    the same batch through the catalog body on the gateway, with per-parameter grad norms.
summarize: the ``single-step-grads`` verdict per config, arm and label, plus an optional tree-vs-tree comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _shape(args) -> tuple[int, int]:
    """--n-gpus/--max-seq-len default to the catalog config, so reference and target build the same arms."""
    from .target import load_config
    from .target import training_config

    tc = training_config(load_config(args.config)) if args.config else {}
    n_gpus = args.n_gpus if args.n_gpus is not None else int(tc.get("n_gpus", 8))
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else int(tc.get("max_seq_len", 65536))
    return n_gpus, max_seq_len


def cmd_reference(args) -> int:
    import torch

    from ..reference import deepseek_v41_official
    from ..reference.hf_single_gpu import run
    from ..reference.model_features import uses_mixer_packing
    from .lora_export import widen_adapter
    from .target import arm_batch

    n_gpus, max_seq_len = _shape(args)
    if args.adapter:
        widen_adapter(args.adapter)
    _, batch = arm_batch(args.fixture, args.arm, n_gpus, max_seq_len)
    t0 = time.time()
    if deepseek_v41_official.is_deepseek_v41(args.fixture):
        result = deepseek_v41_official.run(
            args.fixture, batch, token_budget=args.token_budget, peft_adapter_path=args.adapter, device="cuda"
        )
    else:
        result = run(
            args.fixture,
            batch,
            token_budget=args.token_budget,
            attn_implementation=args.attn_implementation,
            mixer_packing=uses_mixer_packing(args.fixture),
            fused_cross_entropy=args.fused_cross_entropy,
            peft_adapter_path=args.adapter,
            device="cuda",
        )
    out = {
        "kind": "reference",
        "arm": args.arm,
        "n_gpus": n_gpus,
        "max_seq_len": max_seq_len,
        "adapter": args.adapter,
        "loss": result.loss,
        "grad_norms": result.grad_norms,
        "active_tokens": result.active_tokens,
        "microbatches": result.microbatches,
        "peak_gib": result.peak_gib,
        "skipped_no_grad": result.skipped_no_grad,
        "seconds": round(time.time() - t0, 1),
        "device": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(
        f"[reference] arm={args.arm} loss={result.loss:.6f} tensors={len(result.grad_norms)} seconds={out['seconds']}"
    )
    return 0


def cmd_target(args) -> int:
    from .lora_export import zone_url_from_log
    from .target import WARMUP_SEED
    from .target import arm_batch
    from .target import job_body
    from .target import load_config
    from .target import make_client
    from .target import run_target
    from .target import training_config

    n_gpus, max_seq_len = _shape(args)
    config = load_config(args.config)
    body = job_body(
        config,
        args.fixture,
        n_gpus,
        max_seq_len,
        fused_cross_entropy=args.fused_cross_entropy,
        max_tokens_per_mb=args.max_tokens_per_mb,
    )
    _, batch = arm_batch(args.fixture, args.arm, n_gpus, max_seq_len)
    lora = {}
    if training_config(config).get("peft_config"):
        if not (args.zone_url or args.zone_log):
            raise SystemExit("LoRA configs need --zone-url or --zone-log to export the job's adapter")
        out = Path(args.out)
        lora = {
            "warmup_batch": arm_batch(args.fixture, args.arm, n_gpus, max_seq_len, seed=WARMUP_SEED)[1],
            "adapter_dir": out.with_name(out.stem + "-adapter").resolve(),
            "zone_url": args.zone_url or (lambda: zone_url_from_log(args.zone_log)),
            "weights_only_save": not args.skip_weights_only_save,
        }
    result = run_target(make_client(args.gateway), body, batch, **lora)
    record = {
        "kind": "target",
        "label": args.label,
        "arm": args.arm,
        "n_gpus": n_gpus,
        "max_seq_len": max_seq_len,
        "max_tokens_per_mb": training_config(body)["mb_spec"]["max_tokens_per_mb"],
        **result,
    }
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(
        f"[target] {args.label} arm={args.arm} loss={record['loss']:.6f} "
        f"tensors={len(record['grad_norms'])} seconds={record['seconds']}"
    )
    return 0


def cmd_summarize(args) -> int:
    from .compare import summarize

    fixtures = dict(item.split("=", 1) for item in args.fixture)
    rows = summarize(
        args.results,
        fixtures,
        args.labels,
        arms=tuple(args.arms),
        pair=tuple(args.pair) if args.pair else None,
        repeat=args.repeat,
        min_references=args.min_references,
    )
    for row in rows:
        if "head_to_head" in row:
            a, b = row["head_to_head"]
            print(f"   {a}-vs-{b} {json.dumps({k: v for k, v in row.items() if k not in ('config', 'arm')})}")
            continue
        print(
            f"{row['config']:10s} {row['arm']} {row['label']:10s} {row['outcome']} gpus={row['n_gpus']} "
            f"mb={row['max_tokens_per_mb']} calls={row['model_calls']} gate={row['gate']} dloss={row['dloss']:+.2e} "
            f"med={row['rel_median']:.3%} max={row['rel_max']:.3%} over={len(row['over_gate'])}/{row['tensors']} "
            f"ratio={row['ratio_median']:.5f} only_t={len(row['only_target'])}"
            f"(nonzero {len(row['only_target_nonzero'])}) only_r={len(row['only_reference'])}"
        )
        for o in row["over_gate"][:8]:
            ref = o["reference"]
            rel = (o["target"] - ref) / ref if ref else float("nan")
            print(f"      over {o['name']}: t={o['target']:.6g} r={ref:.6g} rel={rel:+.2%}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    from .target import DEFAULT_GATEWAY

    parser = argparse.ArgumentParser(prog="python -m arctic_platform.correctness.gateway")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("reference", "target"):
        p = sub.add_parser(name)
        p.add_argument("--fixture", required=True, help="reduced checkpoint directory")
        p.add_argument("--config", required=name == "target", help="cortex-training create-job JSON")
        p.add_argument("--arm", default="gas1")
        p.add_argument("--n-gpus", type=int, default=None)
        p.add_argument("--max-seq-len", type=int, default=None)
        p.add_argument("--fused-cross-entropy", default="liger")
        p.add_argument("--out", required=True)
    ref = sub.choices["reference"]
    ref.add_argument("--adapter", default=None, help="PEFT adapter exported by the matching LoRA target")
    ref.add_argument("--attn-implementation", default="sdpa")
    ref.add_argument(
        "--token-budget", type=int, default=65536, help="tokens per reference microbatch; eager attention needs less"
    )
    tgt = sub.choices["target"]
    tgt.add_argument("--label", required=True, help="names the tree under test, e.g. pr or base")
    tgt.add_argument("--gateway", default=DEFAULT_GATEWAY)
    tgt.add_argument("--max-tokens-per-mb", type=int, default=None, help="override the config's microbatch")
    tgt.add_argument("--zone-url", default=None, help="training zone v0 server, for LoRA adapter export")
    tgt.add_argument("--zone-log", default=None, help="zone log announcing its server port, if --zone-url is unset")
    tgt.add_argument(
        "--skip-weights-only-save",
        action="store_true",
        help="export the adapter from the resumable save alone, for models without an HF weight export",
    )
    summ = sub.add_parser("summarize")
    summ.add_argument("--results", required=True)
    summ.add_argument("--fixture", nargs="+", required=True, metavar="CONFIG=FIXTURE")
    summ.add_argument("--labels", nargs="+", required=True)
    summ.add_argument("--arms", nargs="+", default=["gas1", "gas4"])
    summ.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"))
    summ.add_argument("--repeat", default=None, help="second run of the first --pair label")
    summ.add_argument("--min-references", type=int, default=3)
    summ.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    return {"reference": cmd_reference, "target": cmd_target, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
