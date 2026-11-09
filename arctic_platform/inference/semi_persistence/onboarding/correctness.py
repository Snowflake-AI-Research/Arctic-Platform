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

"""Does every job return exactly what the reference job returns?

The same requests go to every job (semi_p from the published image is the
reference; semi_p forced to cold-start, plain with semi_p's collective setup,
and stock plain are compared against it). Jobs that run the same numerics must
agree token for token, so a difference among them is a defect; stock plain runs
different collectives at TP >= 2 and is reported, with where it first differs,
rather than gated.

No check assumes how the model answers (whether a prompt ends at EOS, for
example). Requests go one prompt at a time unless a check says otherwise:
batch shape changes numerics, so batched greedy output is reported, not gated.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from arctic_platform.inference.semi_persistence.onboarding.common import generate

logger = logging.getLogger(__name__)

TOPICS = ["rivers", "volcanoes", "bees", "jazz", "glaciers", "owls", "tides", "comets",
          "coral", "deserts", "ferns", "rainbows", "octopuses", "bridges", "clocks", "lighthouses"]
QUESTION = "What is the capital of France? Answer in one word."
GUIDED_SCHEMA = {"type": "object", "properties": {"capital": {"type": "string"}}, "required": ["capital"]}
COMPARED_FIELDS = ("token_ids", "text", "finish_reason", "prompt_len", "generation_len")
DEFAULTS = {"greedy_prompts": 8, "max_tokens": 48, "logprob_atol": 1e-3, "batched_prompts": 16}


def answer(result: dict) -> tuple:
    return tuple(tuple(result.get(f) or ()) if f == "token_ids" else result.get(f) for f in COMPARED_FIELDS)


def first_difference(a: dict, b: dict) -> str:
    if "error" in a or "error" in b:
        return f"error: {a.get('error') or b.get('error')!r}"
    for f in COMPARED_FIELDS:
        if a.get(f) != b.get(f):
            if f == "token_ids":
                x, y = a.get(f) or [], b.get(f) or []
                idx = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
                return f"token_ids differ at position {idx}"
            return f"{f}: {a.get(f)!r} vs {b.get(f)!r}"
    return ""


class Checker:
    """Runs each check on every job and records, per job, agreement with the reference."""

    def __init__(self, client: Any, jobs: dict[str, str], reference: str, gated: set[str],
                 replicas: dict[str, int], max_seq_len: int, settings: dict | None = None):
        self.client, self.jobs, self.ref = client, jobs, reference
        self.gated, self.replicas, self.max_seq_len = gated, replicas, max_seq_len
        self.cfg = {**DEFAULTS, **(settings or {})}
        self.max_tokens = max(4, min(int(self.cfg["max_tokens"]), max_seq_len - 40))
        self.checks: list[dict] = []
        self.semi_p_results: list[dict] = []

    def send(self, prompts: list, sp: dict, routing_key: str | None = None) -> dict[str, list[dict]]:
        out = {v: generate(self.client, job, prompts, sp, routing_key) for v, job in self.jobs.items()}
        for v, res in out.items():
            if v.startswith("on_"):
                self.semi_p_results.extend(res)
        return out

    def record(self, name: str, per_variant: dict[str, tuple[bool, str]], required: bool = True,
               reference_detail: str = "") -> None:
        """``per_variant``: variant -> (agrees with the reference, detail)."""
        variants = {v: {"passed": ok, "detail": d, "gated": required and v in self.gated}
                    for v, (ok, d) in per_variant.items()}
        passed = all(r["passed"] for r in variants.values() if r["gated"]) if required else None
        self.checks.append({"name": name, "required": required, "passed": passed,
                            "reference_detail": reference_detail, "variants": variants,
                            "detail": "; ".join(f"{v}: {r['detail']}" for v, r in variants.items()
                                                if not r["passed"] and r["detail"])})

    def compare_answers(self, name: str, results: dict[str, list[dict]], labels: list[str] | None = None,
                        reference_detail: str = "") -> None:
        ref = results[self.ref]
        per = {}
        for v, res in results.items():
            if v == self.ref:
                continue
            bad = [i for i, (a, b) in enumerate(zip(ref, res)) if answer(a) != answer(b) or "error" in b]
            first = ""
            if bad:
                i = bad[0]
                first = f"; {labels[i] if labels else ''}{': ' if labels else ''}{first_difference(ref[i], res[i])}"
            per[v] = (not bad, f"{len(ref) - len(bad)}/{len(ref)} identical{first}")
        self.record(name, per, reference_detail=reference_detail)


def run(client: Any, jobs: dict[str, str], reference: str, gated: set[str], replicas: dict[str, int],
        max_seq_len: int, settings: dict | None = None) -> list[dict]:
    c = Checker(client, jobs, reference, gated, replicas, max_seq_len, settings)
    n = int(c.cfg["greedy_prompts"])

    # Greedy, one prompt at a time, every compared field.
    results: dict[str, list[dict]] = {v: [] for v in jobs}
    for i in range(n):
        for v, res in c.send([f"Write a short paragraph about {TOPICS[i % len(TOPICS)]}."],
                             {"max_tokens": c.max_tokens, "temperature": 0.0}).items():
            results[v].extend(res)
    c.compare_answers("greedy, one prompt at a time", results, [TOPICS[i % len(TOPICS)] for i in range(n)])

    # Fixed length: ignore_eos must produce exactly max_tokens everywhere.
    fixed = min(32, c.max_tokens)
    res = c.send([QUESTION], {"max_tokens": fixed, "temperature": 0.0, "ignore_eos": True})
    c.compare_answers("ignore_eos runs to max_tokens", res,
                      reference_detail=f"reference generation_len {res[reference][0].get('generation_len')} (want {fixed})")
    if res[reference][0].get("generation_len") != fixed:
        c.checks[-1]["passed"] = False
        c.checks[-1]["detail"] = f"reference generated {res[reference][0].get('generation_len')} tokens, want {fixed}"

    # Every replica of every job: one prompt under distinct routing keys; each
    # job must give one answer, and the reference's.
    keys = max(4, 2 * max(replicas.values()))
    seen: dict[str, dict[tuple, int]] = {v: {} for v in jobs}
    for k in range(keys):
        for v, r in c.send([QUESTION], {"max_tokens": 16, "temperature": 0.0}, routing_key=f"onboard-{k}").items():
            seen[v][answer(r[0])] = seen[v].get(answer(r[0]), 0) + 1
    ref_answers = list(seen[reference])
    per = {}
    for v in jobs:
        if v == reference:
            continue
        ok = len(seen[v]) == 1 and len(ref_answers) == 1 and list(seen[v])[0] == ref_answers[0]
        per[v] = (ok, f"{len(seen[v])} distinct answer(s)" + ("" if ok or len(seen[v]) != 1 else ", not the reference's"))
    c.record(f"same answer from every replica ({keys} routing keys)", per,
             reference_detail=f"reference: {len(ref_answers)} distinct answer(s)")
    if len(ref_answers) != 1:
        c.checks[-1]["passed"] = False
        c.checks[-1]["detail"] = f"the reference itself gave {len(ref_answers)} answers across its replicas"

    # Sampled-token logprobs.
    res = c.send([QUESTION], {"max_tokens": 8, "temperature": 0.0, "logprobs": 1, "return_sampled_logprobs_only": True})
    ref_lp = res[reference][0].get("logprobs")
    per = {}
    for v, r in res.items():
        if v == reference:
            continue
        lp = r[0].get("logprobs")
        if ref_lp is None or lp is None or len(lp) != len(ref_lp):
            per[v] = (False, f"lengths {None if lp is None else len(lp)} vs {None if ref_lp is None else len(ref_lp)}")
            continue
        diff = max((abs(float(a) - float(b)) for a, b in zip(ref_lp, lp)), default=0.0)
        per[v] = (diff <= float(c.cfg["logprob_atol"]), f"max |diff| {diff:.4g}")
    c.record("sampled logprobs", per, reference_detail=f"atol {c.cfg['logprob_atol']}")

    # Top-k logprobs: same structure.
    res = c.send([QUESTION], {"max_tokens": 4, "temperature": 0.0, "logprobs": 2})

    def shape(r: dict) -> list | None:
        return [len(p) for p in r["logprobs"]] if r.get("logprobs") else None

    c.record("top-k logprobs shape", {v: (shape(r[0]) is not None and shape(r[0]) == shape(res[reference][0]),
                                          f"{shape(r[0])} vs {shape(res[reference][0])}")
                                      for v, r in res.items() if v != reference})

    # Prompt logprobs.
    res = c.send([QUESTION], {"max_tokens": 4, "temperature": 0.0, "prompt_logprobs": 1})
    ref_plp = res[reference][0].get("prompt_logprobs")
    c.record("prompt logprobs", {
        v: (ref_plp is not None and r[0].get("prompt_logprobs") is not None
            and len(r[0]["prompt_logprobs"]) == len(ref_plp),
            f"{len(r[0].get('prompt_logprobs') or [])} vs {len(ref_plp or [])} positions")
        for v, r in res.items() if v != reference})

    # Guided JSON: identical text, and the reference's parses.
    res = c.send([QUESTION], {"max_tokens": max(16, min(40, c.max_tokens)), "temperature": 0.0,
                              "guided_json": GUIDED_SCHEMA})
    try:
        parses = isinstance(json.loads(res[reference][0].get("text", "")), dict)
    except (TypeError, ValueError):
        parses = False
    c.compare_answers("guided JSON", res, reference_detail=f"reference {'parses' if parses else 'does not parse'}: "
                                                           f"{res[reference][0].get('text', '')[:60]!r}")
    if not parses:
        c.checks[-1]["passed"] = False

    # Stop strings.
    c.compare_answers("stop strings", c.send([f"Write a short paragraph about {TOPICS[0]}."],
                                             {"max_tokens": c.max_tokens, "temperature": 0.0, "stop": [".", "\n\n"]}))

    # Seeded sampling: deterministic across jobs and across repeats of the reference.
    sp = {"max_tokens": 24, "temperature": 1.0, "seed": 1234}
    res = c.send([QUESTION], sp)
    again = generate(client, jobs[reference], [QUESTION], sp)
    c.compare_answers("seeded sampling", res, reference_detail="reference repeat " + (
        "identical" if answer(again[0]) == answer(res[reference][0]) else "DIFFERS"))
    if answer(again[0]) != answer(res[reference][0]):
        c.checks[-1]["passed"] = False

    # Real token ids from every semi_p job.
    bad = [r for r in c.semi_p_results if "error" in r or not r.get("token_ids")
           or all(t == 0 for t in r["token_ids"]) or len(r["token_ids"]) != r.get("generation_len")]
    c.checks.append({"name": "token ids are real", "required": True, "passed": not bad, "variants": {},
                     "reference_detail": "", "detail": f"{len(c.semi_p_results) - len(bad)}/{len(c.semi_p_results)} "
                                                       "semi_p results well-formed"})

    # Batched greedy: reported, not gated.
    nb = int(c.cfg["batched_prompts"])
    res = c.send([f"Write a short paragraph about {TOPICS[i % len(TOPICS)]} (#{i})." for i in range(nb)],
                 {"max_tokens": c.max_tokens, "temperature": 0.0})
    c.record("greedy, batched", {v: (all(answer(a) == answer(b) for a, b in zip(res[reference], r)),
                                     f"{sum(answer(a) == answer(b) for a, b in zip(res[reference], r))}/{nb} identical")
                                 for v, r in res.items() if v != reference}, required=False,
             reference_detail="batch shape changes numerics")

    for check in c.checks:
        status = {True: "PASS", False: "FAIL", None: "INFO"}[check["passed"]]
        per = ", ".join(f"{v} {'ok' if r['passed'] else 'DIFF'}{'' if r['gated'] else ' (info)'}"
                        for v, r in check["variants"].items())
        logger.info("%-5s %s  [%s] %s", status, check["name"], per, check["detail"])
    return c.checks
