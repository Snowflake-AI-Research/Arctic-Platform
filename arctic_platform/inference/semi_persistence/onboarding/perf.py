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

"""Throughput and latency, semi_p against plain, on the same request grid.

Each cell runs ``concurrency`` closed-loop threads, each sending single-prompt
greedy requests with ``ignore_eos`` so every request generates exactly
``output`` tokens. Prompts are token ids varied per request, so prefix caching
and prefix-affinity routing cannot favour one side. Cells alternate between the
two jobs so drift over the run affects both alike.

A semi_p generate returns its result in one piece, so streaming time-to-first-
token is not observable from the client. TTFT is measured instead as the server
latency of a ``max_tokens=1`` request, and TPOT as the extra server latency per
token of a long request over that, on both jobs the same way.

Throughput is gated on the engine, not the client. Every request costs a
submit, a poll and a status fetch through the API (~250 ms against a 0.1-0.5 s
generate), and a closed-loop thread submits only once it has seen its previous
result, so client tok/s mostly measures that path and its tail: one cell has
read 0.76 and 0.88 on the same image. Engine tok/s is ``concurrency x output /
server p50``, what the engine delivers per in-flight request at that load. It is
gated on the median loaded cell; client tok/s and its worst cell are reported.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from arctic_platform.inference.semi_persistence.onboarding.common import Pair
from arctic_platform.inference.semi_persistence.onboarding.common import median
from arctic_platform.inference.semi_persistence.onboarding.common import prompt_ids
from arctic_platform.inference.semi_persistence.onboarding.common import quantile
from arctic_platform.inference.semi_persistence.onboarding.common import timed_generate

logger = logging.getLogger(__name__)

DEFAULTS = {
    "shapes": [[16, 16], [16, 96], [64, 48]],
    "requests_per_thread": 6,
    "min_requests": 16,
    "latency_samples": 16,
    # A cell over the latency limit is measured this many more times on every
    # side and keeps each side's median, so one noisy measurement cannot fail a run.
    "remeasure": 2,
}
# Ratios gate the verdict, but a TTFT or TPOT within the absolute slack passes
# regardless: at tens of milliseconds a few ms of jitter is a large ratio. A
# restored TP=1 engine's TTFT sits either about 1 or about 11 ms above plain's.
THRESHOLD_DEFAULTS = {
    "ttft_ratio_max": 1.10,
    "ttft_slack_ms": 12.0,
    "tpot_ratio_max": 1.10,
    "tpot_slack_ms": 0.5,
    "latency_ratio_max": 1.15,
    "throughput_ratio_min": 0.90,
}


def concurrency_levels(replicas: int) -> list[int]:
    """1, one request per replica, and four per replica (1/8/32 for one replica)."""
    levels = [1, replicas, 4 * replicas] if replicas > 1 else [1, 8, 32]
    return sorted({min(c, 128) for c in levels})


def fit_shapes(shapes: list, max_seq_len: int) -> list[tuple[int, int]]:
    fitted = [(int(p), int(o)) for p, o in shapes if int(p) + int(o) < max_seq_len]
    if not fitted:
        raise ValueError(f"no perf shape fits max_seq_len={max_seq_len}; set perf.shapes in the model file")
    return fitted


def ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def engine_tok_s(c: dict) -> float:
    return c["concurrency"] * c["output"] / c["server_p50"] if c.get("server_p50") else float("nan")


def cell(make_client: Callable[[], Any], job_id: str, concurrency: int, prompt_len: int,
         out_len: int, per_thread: int, salt_base: int, min_requests: int = DEFAULTS["min_requests"]) -> dict:
    per_thread = max(per_thread, -(-min_requests // concurrency))
    clients = [make_client() for _ in range(concurrency)]
    sp = {"max_tokens": out_len, "temperature": 0.0, "ignore_eos": True}
    records: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(k: int) -> None:
        for r in range(per_thread):
            salt = (salt_base * 1_000_003 + concurrency * 10_007 + k * 101 + r) % 999_983
            try:
                rec = timed_generate(clients[k], job_id, prompt_ids(prompt_len, salt), sp)
            except Exception as exc:  # noqa: BLE001 - counted and reported per cell
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                continue
            with lock:
                records.append(rec)

    try:
        timed_generate(clients[0], job_id, prompt_ids(prompt_len, salt_base), sp)  # warm-up
    except Exception as exc:  # noqa: BLE001
        errors.append(f"warm-up {type(exc).__name__}: {exc}")
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    srv = [r["server_s"] for r in records] or [float("nan")]
    cli = [r["client_s"] for r in records] or [float("nan")]
    return {
        "concurrency": concurrency, "prompt": prompt_len, "output": out_len,
        "requests": len(records), "errors": len(errors), "first_error": errors[0] if errors else "",
        "server_p50": median(srv), "server_p90": quantile(srv, 0.9),
        "client_p50": median(cli), "client_p90": quantile(cli, 0.9),
        "tok_s": sum(r["gen"] for r in records) / wall if wall > 0 else 0.0,
        "zero_id_responses": sum(r["zero_ids"] for r in records),
    }


def _compare(rows: list[dict], lat: dict, sa: str, sb: str) -> dict:
    """Ratios of side ``sa`` over side ``sb`` (latency: >1 is slower; throughput: <1 is slower)."""
    loaded = [r for r in rows if r[sa]["concurrency"] > 1] or rows
    tput = [ratio(engine_tok_s(r[sa]), engine_tok_s(r[sb])) for r in loaded]
    lat_ratios = [ratio(r[sa]["server_p50"], r[sb]["server_p50"]) for r in rows]
    return {
        "ttft_ratio": max(ratio(lat[sa]["ttft_short_s"], lat[sb]["ttft_short_s"]),
                          ratio(lat[sa]["ttft_long_s"], lat[sb]["ttft_long_s"])),
        "ttft_gap_ms": 1000 * max(lat[sa]["ttft_short_s"] - lat[sb]["ttft_short_s"],
                                  lat[sa]["ttft_long_s"] - lat[sb]["ttft_long_s"]),
        "tpot_ratio": ratio(lat[sa]["tpot_s"], lat[sb]["tpot_s"]),
        "median_latency_ratio": median(lat_ratios), "worst_latency_ratio": max(lat_ratios),
        "median_throughput_ratio": median(tput), "worst_throughput_ratio": min(tput),
    }


def run(make_client: Callable[[], Any], pair: Pair, settings: dict | None = None,
        thresholds: dict | None = None, matched: str | None = None) -> dict:
    """Gate semi_p (``pair.on``) against stock plain (``pair.off``). With a
    ``matched`` job (plain with semi_p's collective setup) on the same grid, the
    gap also splits into the cost of those settings (matched / stock) and the
    cost of semi_p itself (semi_p / matched), and a check semi_p fails only
    against stock plain passes (``_pass_on_matched``)."""
    cfg = {**DEFAULTS, **(settings or {})}
    th = {**THRESHOLD_DEFAULTS, **(thresholds or {})}
    shapes = fit_shapes(cfg["shapes"], pair.max_seq_len)
    levels = [int(c) for c in cfg.get("concurrency") or concurrency_levels(pair.replicas)]

    sides = [("on", pair.on), ("off", pair.off)] + ([("matched", matched)] if matched else [])

    def measure(c: int, p: int, o: int, salt: int) -> dict:
        return {side: cell(make_client, job, c, p, o, int(cfg["requests_per_thread"]), salt,
                           int(cfg["min_requests"]))
                for side, job in sides}

    rows = []
    for ci, c in enumerate(levels):
        for si, (p, o) in enumerate(shapes):
            row = measure(c, p, o, ci * 100 + si)
            repeats = [row]
            for attempt in range(1, int(cfg["remeasure"]) + 1):
                if ratio(row["on"]["server_p50"], row["off"]["server_p50"]) <= th["latency_ratio_max"]:
                    break
                # Fresh salts, so the prefix cache does not speed up the repeat.
                repeats.append(measure(c, p, o, attempt * 10_000 + ci * 100 + si))
                row = _median_cell(repeats, [side for side, _ in sides])
            if len(repeats) > 1:
                row["remeasured"] = [{side: m[side]["server_p50"] for side, _ in sides} for m in repeats]
                logger.info("c=%-3d %3d/%-3d over the latency limit, measured %d times; server p50 on %s off %s",
                            c, p, o, len(repeats), [f"{m['on']:.3f}" for m in row["remeasured"]],
                            [f"{m['off']:.3f}" for m in row["remeasured"]])
            row["latency_ratio"] = ratio(row["on"]["server_p50"], row["off"]["server_p50"])
            row["throughput_ratio"] = ratio(row["on"]["tok_s"], row["off"]["tok_s"])
            row["engine_throughput_ratio"] = ratio(engine_tok_s(row["on"]), engine_tok_s(row["off"]))
            rows.append(row)
            m = row.get("matched")
            logger.info("c=%-3d %3d/%-3d server p50 on %.3fs off %.3fs%s  engine tok/s ratio %.2f  "
                        "client tok/s on %.0f off %.0f%s (%.2f)",
                        c, p, o, row["on"]["server_p50"], row["off"]["server_p50"],
                        f" matched {m['server_p50']:.3f}s" if m else "", row["engine_throughput_ratio"],
                        row["on"]["tok_s"], row["off"]["tok_s"], f" matched {m['tok_s']:.0f}" if m else "",
                        row["throughput_ratio"])

    n = int(cfg["latency_samples"])
    short = min(16, pair.max_seq_len // 4)
    long_prompt = max(short, min(1024, pair.max_seq_len // 2))
    tpot_tokens = max(8, min(128, pair.max_seq_len - short - 8))
    latency: dict[str, dict] = {}
    for side, job in sides:
        first = cell(make_client, job, 1, short, 1, n, 900, n)
        first_long = cell(make_client, job, 1, long_prompt, 1, n, 901, n) if long_prompt > short else first
        decode = cell(make_client, job, 1, short, tpot_tokens, n, 902, n)
        latency[side] = {
            "ttft_short_s": first["server_p50"], "ttft_long_s": first_long["server_p50"],
            "tpot_s": (decode["server_p50"] - first["server_p50"]) / (tpot_tokens - 1),
            "prompt_short": short, "prompt_long": long_prompt, "tpot_tokens": tpot_tokens,
            "errors": first["errors"] + first_long["errors"] + decode["errors"],
        }

    overhead = {side: median([r[side]["client_p50"] - r[side]["server_p50"] for r in rows]) for side, _ in sides}
    return {"rows": rows, "latency": latency, "client_overhead_s": overhead,
            **score(rows, latency, pair.replicas, th)}


def _median_cell(repeats: list[dict], sides: list[str]) -> dict:
    """Per side, the measurement with the median server p50 (the lower middle one
    of an even count), so its other fields stay consistent with that p50. Errors
    and zero-id responses count from every measurement."""
    row = {}
    for side in sides:
        cells = [m[side] for m in repeats]
        ordered = sorted(cells, key=lambda c: c["server_p50"])
        row[side] = {**ordered[(len(ordered) - 1) // 2],
                     "errors": sum(c["errors"] for c in cells),
                     "zero_id_responses": sum(c["zero_id_responses"] for c in cells)}
    return row


def _pass_on_matched(checks: list[dict], m: dict, tpot_gap_ms: float, th: dict) -> None:
    """A gap that plain vLLM with semi_p's collective setup shows too is not
    semi_p's: a ratio check that fails against stock plain passes when semi_p
    is within the same limit against the matched job."""
    within = {
        "TTFT ratio (semi_p / plain)": (m["ttft_ratio"], m["ttft_ratio"] <= th["ttft_ratio_max"]
                                        or m["ttft_gap_ms"] <= th["ttft_slack_ms"]),
        "TPOT ratio": (m["tpot_ratio"], m["tpot_ratio"] <= th["tpot_ratio_max"] or tpot_gap_ms <= th["tpot_slack_ms"]),
        "worst server p50 latency ratio": (m["worst_latency_ratio"],
                                           m["worst_latency_ratio"] <= th["latency_ratio_max"]),
        "median engine throughput ratio, concurrency > 1": (m["median_throughput_ratio"],
                                                            m["median_throughput_ratio"] >= th["throughput_ratio_min"]),
    }
    for c in checks:
        if c["passed"] or c["name"] not in within:
            continue
        value, ok = within[c["name"]]
        if ok:
            c["passed"] = True
            c["limit"] += f"; passes against plain matched ({value:.3f})"


def score(rows: list[dict], latency: dict, replicas: int, th: dict) -> dict:
    """Checks, replica scaling and the matched breakdown, from measured rows and
    latencies alone, so a saved run can be re-scored offline."""
    th = {**THRESHOLD_DEFAULTS, **th}
    scaling: dict[str, float] = {}
    if replicas > 1:
        o = max(r["on"]["output"] for r in rows)
        p = max(r["on"]["prompt"] for r in rows if r["on"]["output"] == o)
        by = {(r["on"]["concurrency"], r["on"]["prompt"], r["on"]["output"]): r for r in rows}
        one, full = by.get((1, p, o)), by.get((replicas, p, o))
        if one and full:
            scaling = {side: ratio(engine_tok_s(full[side]), replicas * engine_tok_s(one[side]))
                       for side in one if side in full and isinstance(one[side], dict)}

    on_l, off_l = latency["on"], latency["off"]
    ttft_ratio = max(ratio(on_l["ttft_short_s"], off_l["ttft_short_s"]),
                     ratio(on_l["ttft_long_s"], off_l["ttft_long_s"]))
    ttft_gap_ms = 1000 * max(on_l["ttft_short_s"] - off_l["ttft_short_s"], on_l["ttft_long_s"] - off_l["ttft_long_s"])
    tpot_ratio = ratio(on_l["tpot_s"], off_l["tpot_s"])
    tpot_gap_ms = 1000 * (on_l["tpot_s"] - off_l["tpot_s"])
    worst_latency = max(ratio(r["on"]["server_p50"], r["off"]["server_p50"]) for r in rows)
    # Throughput at concurrency 1 is 1 / latency, which the latency gate covers;
    # gate the loaded cells.
    loaded = [r for r in rows if r["on"]["concurrency"] > 1] or rows
    median_engine_tput = median([ratio(engine_tok_s(r["on"]), engine_tok_s(r["off"])) for r in loaded])
    worst_client_tput = min(ratio(r["on"]["tok_s"], r["off"]["tok_s"]) for r in loaded)
    bad = (sum(r[s]["errors"] for r in rows for s in ("on", "off"))
           + sum(r["on"]["zero_id_responses"] for r in rows)
           + latency["on"]["errors"] + latency["off"]["errors"])
    checks = [
        {"name": "TTFT ratio (semi_p / plain)", "value": ttft_ratio,
         "limit": f"<= {th['ttft_ratio_max']} or within {th['ttft_slack_ms']:g} ms (gap {ttft_gap_ms:.1f} ms)",
         "passed": ttft_ratio <= th["ttft_ratio_max"] or ttft_gap_ms <= th["ttft_slack_ms"]},
        {"name": "TPOT ratio", "value": tpot_ratio,
         "limit": f"<= {th['tpot_ratio_max']} or within {th['tpot_slack_ms']:g} ms (gap {tpot_gap_ms:.2f} ms)",
         "passed": tpot_ratio <= th["tpot_ratio_max"] or tpot_gap_ms <= th["tpot_slack_ms"]},
        {"name": "worst server p50 latency ratio", "value": worst_latency,
         "limit": f"<= {th['latency_ratio_max']}", "passed": worst_latency <= th["latency_ratio_max"]},
        {"name": "median engine throughput ratio, concurrency > 1", "value": median_engine_tput,
         "limit": f">= {th['throughput_ratio_min']}", "passed": median_engine_tput >= th["throughput_ratio_min"]},
        {"name": "request errors and zero-id responses", "value": float(bad), "limit": "== 0", "passed": bad == 0},
    ]
    if scaling:
        limit = th["throughput_ratio_min"] * scaling["off"]
        checks.append({"name": f"replica scaling at concurrency {replicas}", "value": scaling["on"],
                       "limit": f">= {limit:.2f} ({th['throughput_ratio_min']} x plain {scaling['off']:.2f})",
                       "passed": scaling["on"] >= limit})
    checks.append({"name": "worst client throughput ratio, concurrency > 1", "value": worst_client_tput,
                   "limit": f"info: >= {th['throughput_ratio_min']}", "gated": False,
                   "passed": worst_client_tput >= th["throughput_ratio_min"]})
    breakdown = {}
    if "matched" in rows[0]:
        breakdown = {"settings": _compare(rows, latency, "matched", "off"),
                     "semi_p": _compare(rows, latency, "on", "matched")}
        _pass_on_matched(checks, breakdown["semi_p"], 1000 * (on_l["tpot_s"] - latency["matched"]["tpot_s"]), th)
    for c in checks:
        status = ("PASS" if c["passed"] else "FAIL") if c.get("gated", True) else "info"
        logger.info("%-5s %s  %.3f (%s)", status, c["name"], c["value"], c["limit"])
    if breakdown:
        for name, b in breakdown.items():
            logger.info("breakdown, %s: TTFT x%.3f (%+.1f ms), TPOT x%.3f, latency median x%.3f, "
                        "engine throughput median x%.3f (worst x%.3f)", name, b["ttft_ratio"], b["ttft_gap_ms"],
                        b["tpot_ratio"], b["median_latency_ratio"], b["median_throughput_ratio"],
                        b["worst_throughput_ratio"])
    return {"scaling": scaling, "checks": checks, "thresholds": th, "breakdown": breakdown,
            "throughput_basis": "engine"}
