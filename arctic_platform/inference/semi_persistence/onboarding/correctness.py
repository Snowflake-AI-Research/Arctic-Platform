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

The requests are fixed in advance (``probes``) and no request depends on
another job's answer, so each job answers all of them on its own (``collect``),
as soon as it runs, and can be cancelled before the others are up; the checks
then compare the stored answers (``compare``).

No check assumes how the model answers (whether a prompt ends at EOS, for
example). Requests go one prompt at a time unless a check says otherwise:
batch shape changes numerics, so batched greedy output is reported, not gated.

When the reference is the only job that answered (a run of on_hit alone), every
check is decided on its own answers where it can be: one answer across its
replicas, ignore_eos length, a seeded repeat, real token ids, logprob lengths and
values, stop strings absent from the text, guided JSON that parses. A check that
needs another job (greedy agreement, batched) is info, and so is guided JSON
that does not parse, since no plain job ran to compare it with.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any

from arctic_platform.inference.semi_persistence.onboarding.common import generate

logger = logging.getLogger(__name__)

TOPICS = ["rivers", "volcanoes", "bees", "jazz", "glaciers", "owls", "tides", "comets",
          "coral", "deserts", "ferns", "rainbows", "octopuses", "bridges", "clocks", "lighthouses"]
QUESTION = "What is the capital of France? Answer in one word."
GUIDED_SCHEMA = {"type": "object", "properties": {"capital": {"type": "string"}}, "required": ["capital"]}
COMPARED_FIELDS = ("token_ids", "text", "finish_reason", "prompt_len", "generation_len")
DEFAULTS = {"greedy_prompts": 8, "max_tokens": 48, "logprob_atol": 1e-3, "batched_prompts": 16}
MATCHED = "off_matched"
TOP_K = 2
STOP = [".", "\n\n"]
ALONE = "only the reference answered, so there is nothing to compare with"
SEEDED = "seeded sampling"
# The reference answers the seeded request a second time, right after the first.
SEEDED_REPEAT = "seeded sampling, repeat"
BATCHED = "greedy, batched"


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


def apply_guided_parse_rule(check: dict, parses: bool) -> None:
    """Fail guided JSON whose reference does not parse, unless plain vLLM with
    semi_p's collective setup gives the same text: a reasoning model's schema
    applies only after its reasoning ends, so a short answer is free text.
    When the reference is the only job that answered, nothing can tell those
    apart, so a reference that does not parse is info."""
    matched = check["variants"].get(MATCHED) or {}
    if parses or matched.get("passed"):
        return
    if not check["variants"]:
        note = "reference does not parse and no plain job ran to compare with: reported, not gated"
        check["passed"], check["required"] = None, False
        check["detail"] = "; ".join(filter(None, [check["detail"], note]))
        return
    check["passed"] = False
    if not matched:
        check["detail"] = "; ".join(filter(None, [check["detail"], f"no {MATCHED} job to compare with"]))


def settings_for(max_seq_len: int, settings: dict | None) -> tuple[dict, int]:
    cfg = {**DEFAULTS, **(settings or {})}
    return cfg, max(4, min(int(cfg["max_tokens"]), max_seq_len - 40))


def routing_keys(replicas: dict[str, int]) -> int:
    return max(4, 2 * max(replicas.values()))


def probes(replicas: dict[str, int], max_seq_len: int, settings: dict | None = None) -> list[tuple]:
    """Every request a job answers, in the order it sends them: ``(name, prompts,
    sampling params, routing key)``. ``replicas`` covers every job, so all of
    them get the same routing keys."""
    cfg, max_tokens = settings_for(max_seq_len, settings)
    greedy = {"max_tokens": max_tokens, "temperature": 0.0}
    out = [(f"greedy {i}", [f"Write a short paragraph about {TOPICS[i % len(TOPICS)]}."], greedy, None)
           for i in range(int(cfg["greedy_prompts"]))]
    out.append(("ignore_eos", [QUESTION], {"max_tokens": min(32, max_tokens), "temperature": 0.0, "ignore_eos": True},
                None))
    out += [(f"replica {k}", [QUESTION], {"max_tokens": 16, "temperature": 0.0}, f"onboard-{k}")
            for k in range(routing_keys(replicas))]
    out += [
        ("sampled logprobs", [QUESTION],
         {"max_tokens": 8, "temperature": 0.0, "logprobs": 1, "return_sampled_logprobs_only": True}, None),
        ("top-k logprobs", [QUESTION], {"max_tokens": 4, "temperature": 0.0, "logprobs": TOP_K}, None),
        ("prompt logprobs", [QUESTION], {"max_tokens": 4, "temperature": 0.0, "prompt_logprobs": 1}, None),
        ("guided JSON", [QUESTION],
         {"max_tokens": max(16, min(40, max_tokens)), "temperature": 0.0, "guided_json": GUIDED_SCHEMA}, None),
        ("stop strings", [f"Write a short paragraph about {TOPICS[0]}."],
         {"max_tokens": max_tokens, "temperature": 0.0, "stop": list(STOP)}, None),
        (SEEDED, [QUESTION], {"max_tokens": 24, "temperature": 1.0, "seed": 1234}, None),
        (BATCHED, [f"Write a short paragraph about {TOPICS[i % len(TOPICS)]} (#{i})."
                   for i in range(int(cfg["batched_prompts"]))], greedy, None),
    ]
    return out


def collect(client: Any, job_id: str, plan: list[tuple], reference: bool = False,
            stop: threading.Event | None = None) -> dict[str, list[dict]] | None:
    """One job's answers to every request in ``plan``, by request name; None if
    ``stop`` is set before it has answered them all."""
    out: dict[str, list[dict]] = {}
    for name, prompts, sp, routing_key in plan:
        if stop is not None and stop.is_set():
            return None
        out[name] = generate(client, job_id, prompts, dict(sp), routing_key)
        if reference and name == SEEDED:
            out[SEEDED_REPEAT] = generate(client, job_id, prompts, dict(sp))
    logger.info("job %s answered %d requests", job_id, len(out))
    return out


class Checker:
    """Records, per check and per job, agreement with the reference, from every job's stored answers."""

    def __init__(self, answers: dict[str, dict[str, list[dict]]], reference: str, gated: set[str],
                 max_seq_len: int, settings: dict | None = None):
        self.answers, self.ref, self.gated = answers, reference, gated
        self.cfg, self.max_tokens = settings_for(max_seq_len, settings)
        self.checks: list[dict] = []

    def get(self, name: str) -> dict[str, list[dict]]:
        """Every job's answer to request ``name``."""
        return {v: a[name] for v, a in self.answers.items()}

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

    def alone(self) -> bool:
        """Only the reference answered: no job to compare it with."""
        return list(self.answers) == [self.ref]

    def own(self, name: str, ok: bool | None, detail: str) -> None:
        """A check decided on the reference's own answers; ``ok`` None (info) when
        it cannot be decided without another job."""
        self.checks.append({"name": name, "required": ok is not None, "passed": ok, "variants": {},
                            "reference_detail": "", "detail": detail})


def _logprob_values_ok(values: list) -> bool:
    """Every value a finite log-probability (at most 0, with float slack)."""
    try:
        return all(math.isfinite(float(x)) and float(x) <= 1e-4 for x in values)
    except (TypeError, ValueError):
        return False


def _own_logprobs(r: dict) -> tuple[bool, str]:
    """Sampled-token logprobs: one per generated token, each a log-probability."""
    lp, n = r.get("logprobs"), r.get("generation_len")
    if not isinstance(lp, list) or not lp or len(lp) != n:
        return False, f"{len(lp) if isinstance(lp, list) else None} logprobs for {n} generated tokens"
    if not _logprob_values_ok(lp):
        return False, f"{len(lp)} logprobs, not all finite and <= 0"
    return True, f"{len(lp)} logprobs for {n} generated tokens, max {max(map(float, lp)):.4g}"


def _own_top_k(r: dict) -> tuple[bool, str]:
    """Top-k logprobs: one entry per generated token, each with 1 to k + 1
    candidates (the sampled token is added when outside the top k)."""
    lp, n = r.get("logprobs"), r.get("generation_len")
    try:
        shape = [len(p) for p in lp] if isinstance(lp, list) else None
    except TypeError:
        shape = None
    ok = (bool(shape) and len(shape) == n and all(1 <= k <= TOP_K + 1 for k in shape)
          and all(_logprob_values_ok(list(p.values())) for p in lp if isinstance(p, dict)))
    return ok, f"shape {shape} for {n} generated tokens, logprobs={TOP_K}"


def _own_prompt_logprobs(r: dict) -> tuple[bool, str]:
    """Prompt logprobs: one position per prompt token (the first may be left out,
    having no context)."""
    plp, n = r.get("prompt_logprobs"), r.get("prompt_len")
    ok = isinstance(plp, list) and bool(n) and len(plp) in (n, n - 1)
    return ok, f"{len(plp) if isinstance(plp, list) else None} positions for a {n}-token prompt"


def _own_stop(r: dict) -> tuple[bool, str]:
    """Stop strings honoured: none of them in the returned text."""
    if "error" in r:
        return False, f"error: {r['error']!r}"
    found = [s for s in STOP if s in (r.get("text") or "")]
    return not found, (f"stop string(s) {found!r} in the text" if found
                       else f"no stop string in the text (finish_reason {r.get('finish_reason')!r})")


def compare(answers: dict[str, dict[str, list[dict]]], reference: str, gated: set[str], replicas: dict[str, int],
            max_seq_len: int, settings: dict | None = None) -> list[dict]:
    """The checks, from ``answers`` (variant -> what ``collect`` returned for it;
    the reference's must be there, jobs that did not answer are left out)."""
    c = Checker(answers, reference, gated, max_seq_len, settings)
    n = int(c.cfg["greedy_prompts"])

    # Greedy, one prompt at a time, every compared field.
    results = {v: [r for i in range(n) for r in a[f"greedy {i}"]] for v, a in answers.items()}
    if c.alone():
        c.own("greedy, one prompt at a time", None, f"{n} prompts answered; {ALONE}")
    else:
        c.compare_answers("greedy, one prompt at a time", results, [TOPICS[i % len(TOPICS)] for i in range(n)])

    # Fixed length: ignore_eos must produce exactly max_tokens everywhere.
    fixed = min(32, c.max_tokens)
    res = c.get("ignore_eos")
    c.compare_answers("ignore_eos runs to max_tokens", res,
                      reference_detail=f"reference generation_len {res[reference][0].get('generation_len')} (want {fixed})")
    if res[reference][0].get("generation_len") != fixed:
        c.checks[-1]["passed"] = False
        c.checks[-1]["detail"] = f"reference generated {res[reference][0].get('generation_len')} tokens, want {fixed}"

    # Every replica of every job: one prompt under distinct routing keys; each
    # job must give one answer, and the reference's.
    keys = routing_keys(replicas)
    seen: dict[str, dict[tuple, int]] = {v: {} for v in answers}
    for k in range(keys):
        for v, r in c.get(f"replica {k}").items():
            seen[v][answer(r[0])] = seen[v].get(answer(r[0]), 0) + 1
    ref_answers = list(seen[reference])
    per = {}
    for v in answers:
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
    res = c.get("sampled logprobs")
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
    if c.alone():
        c.own("sampled logprobs", *_own_logprobs(res[reference][0]))
    else:
        c.record("sampled logprobs", per, reference_detail=f"atol {c.cfg['logprob_atol']}")

    # Top-k logprobs: same structure.
    res = c.get("top-k logprobs")

    def shape(r: dict) -> list | None:
        return [len(p) for p in r["logprobs"]] if r.get("logprobs") else None

    if c.alone():
        c.own("top-k logprobs shape", *_own_top_k(res[reference][0]))
    else:
        c.record("top-k logprobs shape", {v: (shape(r[0]) is not None and shape(r[0]) == shape(res[reference][0]),
                                              f"{shape(r[0])} vs {shape(res[reference][0])}")
                                          for v, r in res.items() if v != reference})

    # Prompt logprobs.
    res = c.get("prompt logprobs")
    ref_plp = res[reference][0].get("prompt_logprobs")
    if c.alone():
        c.own("prompt logprobs", *_own_prompt_logprobs(res[reference][0]))
    else:
        c.record("prompt logprobs", {
            v: (ref_plp is not None and r[0].get("prompt_logprobs") is not None
                and len(r[0]["prompt_logprobs"]) == len(ref_plp),
                f"{len(r[0].get('prompt_logprobs') or [])} vs {len(ref_plp or [])} positions")
            for v, r in res.items() if v != reference})

    # Guided JSON: identical text, and the reference's parses or plain matched gives the same text.
    res = c.get("guided JSON")
    try:
        parses = isinstance(json.loads(res[reference][0].get("text", "")), dict)
    except (TypeError, ValueError):
        parses = False
    c.compare_answers("guided JSON", res, reference_detail=f"reference {'parses' if parses else 'does not parse'}: "
                                                           f"{res[reference][0].get('text', '')[:60]!r}")
    apply_guided_parse_rule(c.checks[-1], parses)

    # Stop strings.
    if c.alone():
        c.own("stop strings", *_own_stop(c.get("stop strings")[reference][0]))
    else:
        c.compare_answers("stop strings", c.get("stop strings"))

    # Seeded sampling: deterministic across jobs and across repeats of the reference.
    res = c.get(SEEDED)
    again = answers[reference][SEEDED_REPEAT]
    c.compare_answers("seeded sampling", res, reference_detail="reference repeat " + (
        "identical" if answer(again[0]) == answer(res[reference][0]) else "DIFFERS"))
    if answer(again[0]) != answer(res[reference][0]):
        c.checks[-1]["passed"] = False

    # Real token ids from every semi_p job.
    semi_p = [r for v, a in answers.items() if v.startswith("on_")
              for name, rs in a.items() if name not in (SEEDED_REPEAT, BATCHED) for r in rs]
    bad = [r for r in semi_p if "error" in r or not r.get("token_ids")
           or all(t == 0 for t in r["token_ids"]) or len(r["token_ids"]) != r.get("generation_len")]
    c.checks.append({"name": "token ids are real", "required": True, "passed": not bad, "variants": {},
                     "reference_detail": "", "detail": f"{len(semi_p) - len(bad)}/{len(semi_p)} "
                                                       "semi_p results well-formed"})

    # Batched greedy: reported, not gated.
    nb = int(c.cfg["batched_prompts"])
    res = c.get(BATCHED)
    c.record("greedy, batched", {v: (all(answer(a) == answer(b) for a, b in zip(res[reference], r)),
                                     f"{sum(answer(a) == answer(b) for a, b in zip(res[reference], r))}/{nb} identical")
                                 for v, r in res.items() if v != reference}, required=False,
             reference_detail="batch shape changes numerics" + (f"; {ALONE}" if c.alone() else ""))

    for check in c.checks:
        status = {True: "PASS", False: "FAIL", None: "INFO"}[check["passed"]]
        per = ", ".join(f"{v} {'ok' if r['passed'] else 'DIFF'}{'' if r['gated'] else ' (info)'}"
                        for v, r in check["variants"].items())
        logger.info("%-5s %s  [%s] %s", status, check["name"], per, check["detail"])
    return c.checks


def run(client: Any, jobs: dict[str, str], reference: str, gated: set[str], replicas: dict[str, int],
        max_seq_len: int, settings: dict | None = None) -> list[dict]:
    """Collect every job's answers in turn, then compare them."""
    plan = probes(replicas, max_seq_len, settings)
    answers = {v: collect(client, job, plan, v == reference) for v, job in jobs.items()}
    return compare(answers, reference, gated, replicas, max_seq_len, settings)
