#!/usr/bin/env python3
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

"""Tabulate throughput runs (run_file.sh's $OUT_DIR/results/<name>.json) as a Markdown table.

    report.py [--out-dir DIR] [--names <glob>...] [--baseline DIR] [--markdown FILE]

tok/s is tokens trained per second of step wall time: total tokens / total seconds over steps 3..N. A MoE file's run
(ep_size > 1) of 10 or more steps counts from step 5: routing and the allocator settle later. --baseline sets every run
against the run of the same name in another output dir (e.g. the old trees, run on the same nodes) and applies the
regression bar: new / old >= 0.99 for a dense file, 0.97 for a MoE file.
"""

import argparse
import fnmatch
import json
import os
import statistics
from pathlib import Path

BAR = dict(dense=0.99, moe=0.97)


def steady(path: Path, warmup: int, moe_warmup: int):
    d = json.loads(path.read_text())
    records = d.get("records") or []
    if len(records) == 0:
        return None
    tc = d["training_config"]
    gpus = int(tc["n_gpus"])
    moe = int(tc.get("ep_size") or 1) > 1
    first = moe_warmup if moe and len(records) >= 10 else warmup
    use = [r for r in records if r["step"] >= first] or records
    tps = sum(r["tokens"] for r in use) / sum(r["seconds"] for r in use)
    return dict(
        name=path.stem,
        model="/".join(Path(d["model"]).parts[-2:]),
        gpus=gpus,
        moe=moe,
        seq=d["seq_len"],
        rows=d["batch"],
        tps=tps,
        tps_gpu=tps / gpus,
        median_step_s=statistics.median(r["seconds"] for r in use),
        peak_gib=max(((r.get("peak_memory_gib") or {}).get("gpu_peak", 0) for r in records), default=0),
        steps=len(records),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "throughput-out"))
    ap.add_argument("--names", nargs="*", default=["*"], help="globs over run names")
    ap.add_argument("--baseline", help="another output dir with runs of the same names")
    ap.add_argument("--markdown", help="also write the table here")
    ap.add_argument("--warmup", type=int, default=3, help="first step counted")
    ap.add_argument("--moe-warmup", type=int, default=5, help="first step counted, MoE runs of 10+ steps")
    args = ap.parse_args()

    def load(out_dir):
        runs = {}
        for path in sorted((Path(out_dir) / "results").glob("*.json")):
            if any(fnmatch.fnmatch(path.stem, pattern) for pattern in args.names):
                try:
                    r = steady(path, args.warmup, args.moe_warmup)
                except (KeyError, ValueError):  # a run that did not finish
                    r = None
                if r is not None:
                    runs[r["name"]] = r
        return runs

    runs = load(args.out_dir)
    base = load(args.baseline) if args.baseline else {}
    head = "| run | model | GPUs | seq x rows | tok/s | tok/s/GPU | median step s | peak GiB |"
    rule = "|---|---|---:|---|---:|---:|---:|---:|"
    if args.baseline:
        head += " baseline tok/s/GPU | ratio | verdict |"
        rule += "---:|---:|---|"
    lines = [head, rule]
    for r in runs.values():
        line = (
            f"| {r['name']} | {r['model']} | {r['gpus']} | {r['seq']} x {r['rows']} | {r['tps']:,.0f} | "
            f"{r['tps_gpu']:,.0f} | {r['median_step_s']:.2f} | {r['peak_gib']:.1f} |"
        )
        if args.baseline:
            b = base.get(r["name"])
            if b is None:
                line += " - | - | |"
            else:
                ratio = r["tps_gpu"] / b["tps_gpu"]
                verdict = "ok" if ratio >= BAR["moe" if r["moe"] else "dense"] else "below the bar"
                line += f" {b['tps_gpu']:,.0f} | {ratio:.3f} | {verdict} |"
        lines.append(line)
    out = "\n".join(lines)
    print(out)
    if args.markdown:
        Path(args.markdown).write_text(out + "\n")


if __name__ == "__main__":
    main()
