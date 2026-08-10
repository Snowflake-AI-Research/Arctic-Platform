# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate per-seed summary.json files into a table + bootstrap CI.

Usage::

    python -m arctic_platform.integrations.harbor.aggregate_runs \\
        /tmp/harbor_e2e_seed0/summary.json \\
        /tmp/harbor_e2e_seed1/summary.json \\
        /tmp/harbor_e2e_seed2/summary.json

Prints per-seed baseline/final/delta and a paired bootstrap 95% CI on the
mean delta. Falls back to a plain range if there's only one run.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def _bootstrap_mean_ci(xs: list[float], n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    """Simple non-parametric bootstrap 95% CI on the mean of ``xs``."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(0)
    means = []
    n = len(xs)
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", help="summary.json paths")
    args = ap.parse_args()

    rows = []
    for p in args.summaries:
        s = json.loads(Path(p).read_text())
        rows.append({
            "seed": s.get("seed"),
            "baseline_pass": s.get("baseline_pass_at_1"),
            "final_pass": s.get("final_pass_at_1"),
            "delta_pass": s.get("delta"),
            "baseline_reward": s.get("baseline_mean_reward"),
            "final_reward": s.get("final_mean_reward"),
            "delta_reward": s.get("delta_mean_reward"),
            "iters": s.get("iters"),
            "heldout": s.get("heldout"),
        })

    def _agg(label: str, key_base: str, key_final: str, key_delta: str) -> None:
        vals_b = [r[key_base] for r in rows if r[key_base] is not None]
        vals_f = [r[key_final] for r in rows if r[key_final] is not None]
        deltas = [r[key_delta] for r in rows if r[key_delta] is not None]
        if not deltas:
            print(f"  {label:<22} (no data)")
            return
        print(f"  {label:<22} baseline {statistics.mean(vals_b):.3f}±{statistics.pstdev(vals_b):.3f}  "
              f"final {statistics.mean(vals_f):.3f}±{statistics.pstdev(vals_f):.3f}  "
              f"delta {statistics.mean(deltas):+.3f}±{statistics.pstdev(deltas):.3f}", end="")
        lo, hi = _bootstrap_mean_ci(deltas)
        if lo == lo:
            print(f"  95%CI [{lo:+.3f}, {hi:+.3f}]", end="")
        print()

    # Per-run table
    print(f"{'seed':>4}  {'B pass@1':>8}  {'F pass@1':>8}  {'Δ pass@1':>8}  "
          f"{'B mean r':>8}  {'F mean r':>8}  {'Δ mean r':>8}  iters  heldout")
    for r in rows:
        print(f"{str(r['seed']):>4}  "
              f"{(r['baseline_pass'] or 0):>8.3f}  {(r['final_pass'] or 0):>8.3f}  {(r['delta_pass'] or 0):>+8.3f}  "
              f"{(r['baseline_reward'] or 0):>8.3f}  {(r['final_reward'] or 0):>8.3f}  {(r['delta_reward'] or 0):>+8.3f}  "
              f"{str(r['iters']):>5}  {str(r['heldout']):>7}")

    print()
    print(f"=== aggregate over {len(rows)} runs ===")
    _agg("pass@1",           "baseline_pass",   "final_pass",   "delta_pass")
    _agg("mean held-out r",  "baseline_reward", "final_reward", "delta_reward")


if __name__ == "__main__":
    main()
