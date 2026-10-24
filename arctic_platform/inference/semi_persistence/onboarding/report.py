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

"""Verdict and the Markdown / JSON report for one onboarding run, plus a summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

from arctic_platform.inference.semi_persistence.onboarding.jobs import VARIANT_LABELS
from arctic_platform.inference.semi_persistence.onboarding.perf import engine_tok_s
from arctic_platform.inference.semi_persistence.onboarding.perf import ratio


def verdict(result: dict) -> tuple[str, list[str]]:
    """``PASS``, ``WARN`` or ``FAIL`` with the reasons behind anything but PASS."""
    fails, warns = [], []
    if result.get("error"):
        fails.append(result["error"])
    restore = (result.get("startup") or {}).get("semi_p") or {}
    if restore:
        if restore.get("cache_miss"):
            fails.append("semi_p job cold-started (cache miss): no published image for this config and backend image")
        elif not restore.get("markers_visible"):
            warns.append("restore markers not found in the semi_p job's logs; restore not confirmed")
        elif not restore.get("restored"):
            fails.append("not every semi_p replica restored from the published image")
    for c in result.get("correctness") or []:
        if c["required"] and not c["passed"]:
            fails.append(f"correctness: {c['name']} ({c['detail']})")
    for c in gated_perf_checks(result):
        if not c["passed"]:
            fails.append(f"perf: {c['name']} = {_num(c['value'])} ({c['limit']})")
    if fails:
        return "FAIL", fails + warns
    return ("WARN", warns) if warns else ("PASS", [])


def _num(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _status(passed) -> str:
    return {True: "PASS", False: "FAIL", None: "info"}[passed]


def gated_perf_checks(result: dict) -> list[dict]:
    return [c for c in (result.get("perf") or {}).get("checks") or [] if c.get("gated", True)]


def _perf_status(check: dict) -> str:
    if check.get("gated", True):
        return _status(check["passed"])
    return "info" if check["passed"] else "info (below)"


def replica_table(sp: dict, min_step_s: float = 0.5) -> list[str]:
    """One row per semi_p engine: where it restored from, copy / restore / key-to-ready
    times, then every restore step that took at least ``min_step_s`` on some engine."""
    reps = sp["replicas"]
    steps = [s for s in dict.fromkeys(n for info in reps.values() for n in info.get("steps", {}))
             if max(info.get("steps", {}).get(s, 0.0) for info in reps.values()) >= min_step_s]
    head = ["replica", "source", "copy", "restore", "key to ready"] + steps
    out = [f"semi_p replicas, seconds (restore steps from {sp.get('steps_source', 'logs')}):", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for rep, info in reps.items():
        if info.get("partition_agent"):
            source = "node-partition agent"
        elif info.get("from_published_image"):
            source = "published image"
        else:
            source = "cache miss" if info.get("cache_miss") else "?"
        cells = [rep, source, _num(info.get("materialize_s"), 1), _num(info.get("restore_s"), 1),
                 _num(info.get("key_to_ready_s"), 1)]
        cells += [_num(info.get("steps", {}).get(s), 1) for s in steps]
        failed = info.get("failed_steps") or []
        out.append("| " + " | ".join(cells) + " |" + (f" FAILED: {', '.join(failed)}" if failed else ""))
    return out


def breakdown_table(breakdown: dict) -> list[str]:
    """Where the perf gap comes from: semi_p's collective settings on a plain
    engine (plain matched / plain), and semi_p itself (semi_p / plain matched)."""
    if not breakdown:
        return []
    rows = [("semi_p's collective settings (plain matched / plain)", breakdown.get("settings") or {}),
            ("semi_p itself (semi_p / plain matched)", breakdown.get("semi_p") or {})]
    out = ["", "Where the gap comes from (ratios; latency above 1 and throughput below 1 are slower):", "",
           "| cost of | TTFT | TPOT | server p50 latency, median / worst | throughput c>1, median / worst |",
           "|---|---|---|---|---|"]
    for label, b in rows:
        out.append(f"| {label} | x{_num(b.get('ttft_ratio'))} ({b.get('ttft_gap_ms', float('nan')):+.1f} ms) | "
                   f"x{_num(b.get('tpot_ratio'))} | x{_num(b.get('median_latency_ratio'))} / "
                   f"x{_num(b.get('worst_latency_ratio'))} | x{_num(b.get('median_throughput_ratio'))} / "
                   f"x{_num(b.get('worst_throughput_ratio'))} |")
    return out


def correctness_table(checks: list[dict]) -> list[str]:
    """One row per check; one column per compared job (agreement with the
    reference, and whether that job is gated); the check's overall result."""
    variants = list(dict.fromkeys(v for c in checks for v in c.get("variants") or {}))
    head = ["check", "result"] + [f"{VARIANT_LABELS.get(v, v)} vs reference" for v in variants] + ["reference"]
    out = ["## Correctness", "",
           "Every job got the same requests; each column says whether it matched the reference (semi_p from the "
           "published image) token for token. Gated columns must match; *info* columns are expected to differ "
           "and show where.", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in checks:
        cells = [c["name"], _status(c["passed"])]
        for v in variants:
            r = (c.get("variants") or {}).get(v)
            if r is None:
                cells.append("-")
                continue
            mark = "match" if r["passed"] else "**differs**"
            cells.append(f"{mark}{'' if r['gated'] else ' *(info)*'}: {r['detail']}")
        cells.append(c.get("reference_detail") or ("" if c.get("variants") else c.get("detail", "")))
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in cells) + " |")
    return out


def markdown(result: dict) -> str:
    v, reasons = verdict(result)
    spec = result["model"]
    lines = [f"# Onboarding: {spec['name']}", "",
             f"**Verdict: {v}**", ""]
    lines += [f"- {r}" for r in reasons] + ([""] if reasons else [])
    settings = ((result.get("perf") or {}).get("breakdown") or {}).get("settings")
    if settings:
        lines += [f"Degradation from semi_p's collective settings alone (plain matched / plain), not counted "
                  f"against semi_p: TTFT x{_num(settings.get('ttft_ratio'))}, TPOT x{_num(settings.get('tpot_ratio'))}, "
                  f"server p50 latency median x{_num(settings.get('median_latency_ratio'))}, engine throughput "
                  f"median x{_num(settings.get('median_throughput_ratio'))}.", ""]
    lines += [
        "| | |", "|---|---|",
        f"| Model | `{spec['model_name']}` |",
        f"| Published layout | {spec['n_gpus']} GPUs, TP={spec['tensor_parallel_size']}, {spec['replicas']} replica(s), max_seq_len {spec['max_seq_len']} |",
        f"| Tested as | {spec.get('copies', 1)} copies: {spec.get('test_n_gpus', spec['n_gpus'])} GPUs, "
        f"{spec.get('test_replicas', spec['replicas'])} replica(s) per job |",
        f"| Image tag | `{result.get('image_tag') or 'default'}` |",
        f"| Run | {result.get('started_at')} to {result.get('finished_at')} UTC |",
        "",
    ]
    jobs = result.get("jobs") or {"on_hit": result.get("on_job_id"), "off_default": result.get("off_job_id")}
    notes = result.get("variant_notes") or {}
    lines += ["| job | id | role |", "|---|---|---|"]
    for variant, job_id in jobs.items():
        role = "reference" if variant == "on_hit" else notes.get(variant, "")
        lines.append(f"| {VARIANT_LABELS.get(variant, variant)} | `{job_id}` | {role} |")
    for variant, why in (result.get("variants_skipped") or {}).items():
        lines.append(f"| {VARIANT_LABELS.get(variant, variant)} | not run | {why} |")
    lines.append("")

    st = result.get("startup") or {}
    if st:
        lines += ["## Start-up", "",
                  "| | semi_p | plain |", "|---|---|---|"]
        on_t, off_t = st.get("on_timeline", {}), st.get("off_timeline", {})
        for key, label in (("submit_to_running_s", "submit to running"),
                           ("initializing_to_running_s", "initializing to running")):
            lines.append(f"| {label} | {_num(on_t.get(key), 0)} s | {_num(off_t.get(key), 0)} s |")
        sp, vl = st.get("semi_p") or {}, st.get("vllm_off") or {}
        lines.append(f"| semi_p key to ready (slowest replica) / vLLM engine init | "
                     f"{_num(sp.get('key_to_ready_s'), 0)} s | {_num(vl.get('engine_init_s'), 0)} s |")
        lines.append("")
        if sp.get("replicas"):
            lines += replica_table(sp) + [""]
        if vl:
            lines += ["plain vLLM cold start (slowest replica, seconds): "
                      + ", ".join(f"{k[:-2]} {v:.1f}" for k, v in vl.items()), ""]
        miss = st.get("semi_p_miss") or {}
        if miss.get("replicas"):
            dumps = [i.get("cold_start_and_dump_s") for i in miss["replicas"].values() if i.get("cold_start_and_dump_s")]
            lines += [f"semi_p forced miss: {'cold start + dump ' + _num(max(dumps), 0) + ' s, ' if dumps else ''}"
                      f"key to ready {_num(miss.get('key_to_ready_s'), 0)} s "
                      f"({'cache miss' if miss.get('cache_miss') else 'no cache miss: restored a local image'})", ""]
        if st.get("generate_counts"):
            lines += ["Generate requests served per semi_p replica: "
                      + ", ".join(f"{k} {v}" for k, v in st["generate_counts"].items()), ""]
        lines += [f"Log sources: semi_p {st.get('on_sources') or ['none']}; plain {st.get('off_sources') or ['none']}", ""]

    corr = result.get("correctness") or []
    if corr:
        lines += correctness_table(corr) + [""]

    perf = result.get("perf") or {}
    if perf:
        lines += ["## Throughput and latency", "", "| check | value | limit | result |", "|---|---|---|---|"]
        lines += [f"| {c['name']} | {_num(c['value'])} | {c['limit']} | {_perf_status(c)} |" for c in perf["checks"]]
        lat = perf.get("latency", {})
        if lat:
            lines += ["", "| server latency, concurrency 1 | semi_p | plain |", "|---|---|---|"]
            for key, label in (("ttft_short_s", "TTFT, short prompt"), ("ttft_long_s", "TTFT, long prompt"),
                               ("tpot_s", "TPOT")):
                lines.append(f"| {label} | {_num(lat['on'][key] * 1000, 1)} ms | {_num(lat['off'][key] * 1000, 1)} ms |")
        ovh = perf.get("client_overhead_s")
        if ovh:
            lines += ["", f"Client overhead (median client p50 - server p50 over the grid): "
                          f"semi_p {ovh['on'] * 1000:.0f} ms, plain {ovh['off'] * 1000:.0f} ms"
                          + (f", plain matched {ovh['matched'] * 1000:.0f} ms" if "matched" in ovh else "")]
        lines += breakdown_table(perf.get("breakdown") or {})
        has_m = bool(perf["rows"]) and "matched" in perf["rows"][0]
        lines += ["", "Engine tok/s is concurrency x output / server p50; client tok/s is generated tokens over the "
                      "cell's wall time at the client, API overhead included.", "",
                  "| concurrency | prompt/output | server p50 on / off" + (" / matched" if has_m else "")
                  + " (s) | engine tput ratio | client p50 on / off (s) | client tok/s on / off"
                  + (" / matched" if has_m else "") + " | client tput ratio |",
                  "|---|---|---|---|---|---|---|"]
        for r in perf["rows"]:
            on, off, m = r["on"], r["off"], r.get("matched")
            engine = ratio(engine_tok_s(on), engine_tok_s(off))
            lines.append(f"| {on['concurrency']} | {on['prompt']}/{on['output']} | {_num(on['server_p50'])} / {_num(off['server_p50'])}"
                         + (f" / {_num(m['server_p50'])}" if m else "") + f" | {_num(engine, 2)} | "
                         f"{_num(on['client_p50'])} / {_num(off['client_p50'])} | {on['tok_s']:.0f} / {off['tok_s']:.0f}"
                         + (f" / {m['tok_s']:.0f}" if m else "") + f" | {_num(ratio(on['tok_s'], off['tok_s']), 2)} |")
        lines.append("")
    return "\n".join(lines)


def stem(result: dict) -> str:
    """``<model>-<UTC start>``: the report's file name, and its log directory's prefix."""
    return f"{result['model']['name']}-{result['started_at'].replace(':', '').replace('-', '')[:15]}"


def write(result: dict, out_dir: str) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem_ = stem(result)
    v, reasons = verdict(result)
    result = {**result, "verdict": v, "reasons": reasons}
    (out / f"{stem_}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    md = out / f"{stem_}.md"
    md.write_text(markdown(result), encoding="utf-8")
    return md


def write_summary(results: list[dict], out_dir: str, started_at: str) -> Path:
    lines = ["# Onboarding summary", "",
             "| model | verdict | semi_p key to ready | plain engine init | correctness | perf | report |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        v, reasons = verdict(r)
        st = r.get("startup") or {}
        corr = r.get("correctness") or []
        perf = gated_perf_checks(r)
        req = [c for c in corr if c["required"]]
        lines.append(
            f"| {r['model']['name']} | {v} | {_num((st.get('semi_p') or {}).get('key_to_ready_s'), 0)} s | "
            f"{_num((st.get('vllm_off') or {}).get('engine_init_s'), 0)} s | "
            f"{sum(bool(c['passed']) for c in req)}/{len(req)} | {sum(bool(c['passed']) for c in perf)}/{len(perf)} | "
            f"{Path(r.get('report', '')).name} |")
        if reasons:
            lines.append(f"| | | {reasons[0][:160]} | | | | |")
    out = Path(out_dir).expanduser() / f"summary-{started_at.replace(':', '').replace('-', '')[:15]}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
