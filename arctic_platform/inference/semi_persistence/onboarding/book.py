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

"""The onboarding book: every onboarded model config in one set of tables.

    python -m arctic_platform.inference.semi_persistence.onboarding.book reports=onboarding_reports [out=ONBOARDING.md]

Reads every report JSON under ``reports`` and keeps the latest run for each
(model config, backend image, test copies), so the tables grow as new models
and images are onboarded and a re-run replaces its earlier row. The book holds
no job ids, so it can be committed. ``onboard`` regenerates it after each run.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

from arctic_platform.inference.semi_persistence.onboarding.common import median
from arctic_platform.inference.semi_persistence.onboarding.common import parse_kv_args
from arctic_platform.inference.semi_persistence.onboarding.perf import engine_tok_s
from arctic_platform.inference.semi_persistence.onboarding.perf import ratio
from arctic_platform.inference.semi_persistence.onboarding.report import gated_perf_checks
from arctic_platform.inference.semi_persistence.onboarding.report import verdict

logger = logging.getLogger(__name__)

RESTORE_STEPS = ["criu_restore", "cuda_restore", "reinit_nccl", "load_weights", "wake_up_weights", "repin",
                 "restore_weights", "wake_up_kv_cache", "rebind_graphs", "recapture_graphs"]
VLLM_PHASES = [("load_weights_s", "load weights"), ("model_loading_s", "model loading"),
               ("torch_compile_s", "torch.compile"), ("profiling_s", "profiling"),
               ("graph_capture_s", "graph capture"), ("engine_init_s", "engine init")]


def _f(v, digits: int = 1, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{digits}f}{unit}"


def load_runs(directory: Path) -> list[dict]:
    """Latest report per (model, image, copies), ordered by model name."""
    latest: dict[tuple, dict] = {}
    for path in sorted(directory.rglob("*.json")):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or "model" not in r or "started_at" not in r:
            continue
        key = (r["model"]["name"], r.get("image_tag") or "default", r["model"].get("copies", 1))
        if key not in latest or r["started_at"] > latest[key]["started_at"]:
            latest[key] = r
    return [latest[k] for k in sorted(latest)]


def _layout(r: dict) -> str:
    m = r["model"]
    published = f"{m['n_gpus']} GPU, TP={m['tensor_parallel_size']}, {m['replicas']} rep"
    if m.get("copies", 1) > 1:
        return f"{published} (tested x{m['copies']}: {m['test_n_gpus']} GPU, {m['test_replicas']} rep)"
    return published


def _image(r: dict) -> str:
    return r.get("image_tag") or "default"


def overview(runs: list[dict]) -> list[str]:
    out = ["## Overview", "",
           "| model config | model | layout | backend image | date (UTC) | verdict | correctness | perf | notes |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        v, reasons = verdict(r)
        corr = [c for c in r.get("correctness") or [] if c["required"]]
        perf = gated_perf_checks(r)
        out.append(f"| {r['model']['name']} | `{r['model']['model_name']}` | {_layout(r)} | `{_image(r)}` | "
                   f"{r['started_at'][:16].replace('T', ' ')} | **{v}** | "
                   f"{sum(bool(c['passed']) for c in corr)}/{len(corr)} | "
                   f"{sum(bool(c['passed']) for c in perf)}/{len(perf)} | {'; '.join(reasons)[:200]} |")
    return out + [""]


def plain_cold_start(vl: dict) -> tuple[float | None, bool]:
    """vLLM's model loading plus its engine init (which excludes loading), and
    whether the loading part was found in the logs."""
    init = vl.get("engine_init_s")
    if init is None:
        return None, False
    load = vl.get("model_loading_s") or vl.get("load_weights_s")
    return init + (load or 0.0), load is not None


def startup_table(runs: list[dict]) -> list[str]:
    out = ["## Start-up", "",
           "semi_p times are the slowest replica's, from key resolution to a serving engine. The plain cold "
           "start is vLLM's model loading plus its engine init (profiling, KV cache, compile, graph capture); "
           "`>=` marks rows whose log lacked the loading line. Speedup = plain cold start / semi_p key to ready.", "",
           "| model config | image copy (s) | restore (s) | semi_p key to ready (s) | plain cold start (s) | speedup "
           "| submit to running, semi_p / plain (s) | restored from |",
           "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        st = r.get("startup") or {}
        sp, vl = st.get("semi_p") or {}, st.get("vllm_off") or {}
        reps = [i for i in (sp.get("replicas") or {}).values() if not i.get("partition_agent")]
        copy = max((i.get("materialize_s") or 0.0 for i in reps), default=None)
        restore = max((i.get("restore_s") or 0.0 for i in reps), default=None)
        k2r = sp.get("key_to_ready_s")
        cold, complete = plain_cold_start(vl)
        mark = "" if complete or cold is None else ">= "
        speedup = cold / k2r if k2r and cold else None
        on_t, off_t = st.get("on_timeline") or {}, st.get("off_timeline") or {}
        source = ("published image" if sp.get("restored") else "cold start (cache miss)" if sp.get("cache_miss")
                  else "not confirmed")
        out.append(f"| {r['model']['name']} | {_f(copy)} | {_f(restore)} | {_f(k2r)} | {mark}{_f(cold)} | "
                   f"{mark}{_f(speedup, 1, 'x')} | {_f(on_t.get('submit_to_running_s'), 0)} / "
                   f"{_f(off_t.get('submit_to_running_s'), 0)} | {source} ({len(reps)} replica(s)) |")
    return out + [""]


def restore_steps_table(runs: list[dict]) -> list[str]:
    out = ["## Restore primitives", "",
           "Seconds, slowest replica per step (node-partition agents included). Needs the GPU pods' logs "
           "(`log_files_on`); rows without them show `-`.", "",
           "| model config | " + " | ".join(RESTORE_STEPS) + " |",
           "|---|" + "---|" * len(RESTORE_STEPS)]
    for r in runs:
        steps = ((r.get("startup") or {}).get("semi_p") or {}).get("slowest_replica_steps") or {}
        out.append(f"| {r['model']['name']} | " + " | ".join(_f(steps.get(s)) for s in RESTORE_STEPS) + " |")
    return out + [""]


def cold_start_table(runs: list[dict]) -> list[str]:
    out = ["## Plain vLLM cold start", "", "Seconds, slowest replica.", "",
           "| model config | " + " | ".join(label for _, label in VLLM_PHASES) + " |",
           "|---|" + "---|" * len(VLLM_PHASES)]
    for r in runs:
        vl = (r.get("startup") or {}).get("vllm_off") or {}
        out.append(f"| {r['model']['name']} | " + " | ".join(_f(vl.get(k)) for k, _ in VLLM_PHASES) + " |")
    return out + [""]


def latency_table(runs: list[dict]) -> list[str]:
    out = ["## Latency", "",
           "Server-side p50 at concurrency 1, semi_p / plain. TTFT is a `max_tokens=1` request; TPOT is the "
           "extra latency per token of a long generation. Client overhead is client p50 minus server p50.", "",
           "| model config | TTFT short (ms) | TTFT long (ms) | TPOT (ms) | TTFT gap (ms) | TPOT ratio "
           "| client overhead (ms) |",
           "|---|---|---|---|---|---|---|"]
    for r in runs:
        perf = r.get("perf") or {}
        lat = perf.get("latency") or {}
        if not lat:
            out.append(f"| {r['model']['name']} | - | - | - | - | - | - |")
            continue
        on, off = lat["on"], lat["off"]
        gap = 1000 * max(on["ttft_short_s"] - off["ttft_short_s"], on["ttft_long_s"] - off["ttft_long_s"])
        ovh = perf.get("client_overhead_s") or {}
        out.append(
            f"| {r['model']['name']} | {_f(on['ttft_short_s'] * 1000)} / {_f(off['ttft_short_s'] * 1000)} | "
            f"{_f(on['ttft_long_s'] * 1000)} / {_f(off['ttft_long_s'] * 1000)} | "
            f"{_f(on['tpot_s'] * 1000, 2)} / {_f(off['tpot_s'] * 1000, 2)} | {_f(gap)} | "
            f"{_f(on['tpot_s'] / off['tpot_s'] if off['tpot_s'] else None, 2)} | "
            f"{_f(ovh.get('on', float('nan')) * 1000, 0)} / {_f(ovh.get('off', float('nan')) * 1000, 0)} |")
    return out + [""]


def throughput_table(runs: list[dict]) -> list[str]:
    out = ["## Throughput", "",
           "Tokens/s, semi_p / plain, closed-loop single-prompt requests with `ignore_eos`, on the longest-output "
           "shape at each concurrency level. Engine tok/s is concurrency x output / server p50 (gated, median over "
           "cells with concurrency > 1); client tok/s is tokens over the cell's wall time at the client, API "
           "overhead included (reported, worst cell). Scaling is engine tok/s at concurrency = replicas over "
           "replicas x the concurrency-1 rate.", "",
           "| model config | shape | concurrency: engine tok/s semi_p / plain | concurrency: client tok/s semi_p / plain "
           "| engine tput ratio, median (c>1) | client tput ratio, worst (c>1) | replica scaling, semi_p / plain |",
           "|---|---|---|---|---|---|---|"]
    for r in runs:
        perf = r.get("perf") or {}
        rows = perf.get("rows") or []
        if not rows:
            out.append(f"| {r['model']['name']} | - | - | - | - | - | - |")
            continue
        p, o = max(((x["on"]["prompt"], x["on"]["output"]) for x in rows), key=lambda s: s[1])
        cells = sorted((x for x in rows if (x["on"]["prompt"], x["on"]["output"]) == (p, o)),
                       key=lambda x: x["on"]["concurrency"])
        engine = "<br>".join(f"c={x['on']['concurrency']}: {engine_tok_s(x['on']):.0f} / {engine_tok_s(x['off']):.0f}"
                             for x in cells)
        client = "<br>".join(f"c={x['on']['concurrency']}: {x['on']['tok_s']:.0f} / {x['off']['tok_s']:.0f}"
                             for x in cells)
        loaded = [x for x in rows if x["on"]["concurrency"] > 1] or rows
        engine_median = median([ratio(engine_tok_s(x["on"]), engine_tok_s(x["off"])) for x in loaded])
        client_worst = min(ratio(x["on"]["tok_s"], x["off"]["tok_s"]) for x in loaded)
        sc = perf.get("scaling") or {}
        scaling = f"{_f(sc.get('on'), 2)} / {_f(sc.get('off'), 2)}" if sc else "- (one replica)"
        out.append(f"| {r['model']['name']} | {p}/{o} | {engine} | {client} | {_f(engine_median, 2)} | "
                   f"{_f(client_worst, 2)} | {scaling} |")
    return out + [""]


def _variant_summary(r: dict, variant: str) -> str:
    checks = [c for c in r.get("correctness") or [] if variant in (c.get("variants") or {}) and c["required"]]
    if not checks:
        why = (r.get("variants_skipped") or {}).get(variant)
        return "not run" + (f" ({why.split(':')[0]})" if why else "")
    same = sum(c["variants"][variant]["passed"] for c in checks)
    gated = any(c["variants"][variant]["gated"] for c in checks)
    return f"{same}/{len(checks)} match" + ("" if gated else " *(info)*")


def breakdown_table(runs: list[dict]) -> list[str]:
    """Split of the perf gap into semi_p's collective settings and semi_p itself."""
    out = ["## Where the perf gap comes from", "",
           "Measured where plain with semi_p's collective setup ran the same grid (TP >= 2, one pod). "
           "Settings = plain matched / plain; semi_p = semi_p / plain matched. Latency ratios above 1 and "
           "throughput ratios below 1 are slower; throughput is over cells with concurrency > 1.", "",
           "| model config | settings: TTFT / TPOT | settings: latency median | settings: throughput median / worst "
           "| semi_p: TTFT / TPOT | semi_p: latency median | semi_p: throughput median / worst |",
           "|---|---|---|---|---|---|---|"]
    any_rows = False
    for r in runs:
        b = (r.get("perf") or {}).get("breakdown") or {}
        if not b:
            continue
        any_rows = True
        cells = []
        for part in ("settings", "semi_p"):
            x = b.get(part) or {}
            cells += [f"x{_f(x.get('ttft_ratio'), 2)} ({_f(x.get('ttft_gap_ms'))} ms) / x{_f(x.get('tpot_ratio'), 2)}",
                      f"x{_f(x.get('median_latency_ratio'), 2)}",
                      f"x{_f(x.get('median_throughput_ratio'), 2)} / x{_f(x.get('worst_throughput_ratio'), 2)}"]
        out.append(f"| {r['model']['name']} | " + " | ".join(cells) + " |")
    if not any_rows:
        out.append("| (no run with a plain matched perf job yet) | | | | | | |")
    return out + [""]


def correctness_table(runs: list[dict]) -> list[str]:
    out = ["## Correctness", "",
           "Every job got the same requests, compared token for token with semi_p restored from the published "
           "image. A forced miss and plain-with-semi_p's-collectives must match it (gated); stock plain runs "
           "different collectives at TP >= 2, so its matches are information, and where it differs is in each "
           "run's report.", "",
           "| model config | semi_p forced miss | plain, matched | plain, default |",
           "|---|---|---|---|"]
    for r in runs:
        if not any(c.get("variants") for c in r.get("correctness") or []):
            out.append(f"| {r['model']['name']} | (two-job run) | - | - |")
            continue
        out.append(f"| {r['model']['name']} | {_variant_summary(r, 'on_miss')} | "
                   f"{_variant_summary(r, 'off_matched')} | {_variant_summary(r, 'off_default')} |")
    names = list(dict.fromkeys(c["name"] for r in runs for c in r.get("correctness") or []))
    short = {n: n.split(" (")[0] for n in names}
    out += ["", "Per check (gated result; \"default differs\" marks stock plain disagreeing):", "",
            "| model config | " + " | ".join(short[n] for n in names) + " |",
            "|---|" + "---|" * len(names)]
    for r in runs:
        by = {c["name"]: c for c in r.get("correctness") or []}
        cells = []
        for n in names:
            c = by.get(n)
            if c is None:
                cells.append("-")
                continue
            cell = {True: "PASS", False: "**FAIL**", None: "info"}[c["passed"]]
            default = (c.get("variants") or {}).get("off_default")
            if default is not None and not default["passed"]:
                cell += " (default differs)"
            cells.append(cell)
        legacy = "" if any(c.get("variants") for c in r.get("correctness") or []) else " (two-job run, vs plain)"
        out.append(f"| {r['model']['name']}{legacy} | " + " | ".join(cells) + " |")
    return out + [""]


def build(directory: Path) -> str:
    runs = load_runs(directory)
    head = ["# Semi-persistence onboarding book", "",
            f"{len(runs)} model config(s). Latest run per model config, backend image and test copies; "
            "regenerate with `python -m arctic_platform.inference.semi_persistence.onboarding.book reports=DIR`.", ""]
    sections = [overview, startup_table, restore_steps_table, cold_start_table, latency_table,
                throughput_table, breakdown_table, correctness_table]
    return "\n".join(head + [line for s in sections for line in s(runs)])


def write(directory: str, out: str | None = None) -> Path:
    d = Path(directory).expanduser()
    path = Path(out).expanduser() if out else d / "ONBOARDING.md"
    path.write_text(build(d), encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_kv_args(argv)
    path = write(args.get("reports", "onboarding_reports"), args.get("out"))
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
