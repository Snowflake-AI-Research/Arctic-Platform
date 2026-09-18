"""Reproduce R2E-Gym's reward on a live sandbox, before and after the gold patch.

Mirrors ``R2EGymTaskSet._run_tests`` / ``_calculate_reward`` from Boyi's
verifiers checkout: stage ``/r2e_tests`` into the repo, run ``run_tests.sh``,
parse the pytest summary, and require an exact match against
``expected_output_json``. Broken repo must score 0 and the gold patch must
score 1, otherwise the reward signal we train on is not the one he trained on.
"""

from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, "/modeling-code/karthik/abstract-remote-exps/poc")

from sandbox import Sandbox  # noqa: E402

DATASET = (
    "/data/fshu/important/swe_data/r2e_family/"
    "R2E-Gym-Subset_validgold_unique_baseline/train.jsonl"
)
REPO = "/testbed"


def parse_log_pytest(log: str | None) -> dict[str, str]:
    if log is None:
        return {}
    out: dict[str, str] = {}
    if "short test summary info" not in log:
        return out
    for line in log.split("short test summary info")[1].strip().split("\n"):
        if "PASSED" in line:
            out[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            out[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            try:
                name = ".".join(line.split("::")[1:])
            except IndexError:
                name = line
            out[name.split(" - ")[0]] = "ERROR"
    return out


def decolor(d: dict) -> dict:
    return {re.sub(r"\u001b\[\d+m", "", k): v for k, v in d.items()}


def reward(test_output: str, expected_json: str) -> tuple[float, dict, dict]:
    parse = decolor(parse_log_pytest(test_output))
    expected = decolor(json.loads(expected_json))
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse)}
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected)}
    if len(parse) != len(expected):
        return 0.0, parse, expected
    for k in parse:
        if not k:
            continue
        if k not in expected or parse[k] != expected[k]:
            return 0.0, parse, expected
    return 1.0, parse, expected


def run_tests(sb: Sandbox) -> str:
    sb.exec(f"cd {REPO} && rm -f test_output.txt", timeout=60)
    sb.exec(f"cd {REPO} && /bin/bash run_tests.sh > test_output.txt 2>&1", timeout=1800)
    _, out = sb.exec(f"cat {REPO}/test_output.txt", timeout=120)
    return out


def main() -> None:
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    name = sys.argv[2] if len(sys.argv) > 2 else f"r2e-{row}"
    with open(DATASET) as fh:
        for i, line in enumerate(fh):
            if i == row:
                rec = json.loads(line)
                break

    print(f"[grade] {rec['instance_id']}  expecting {rec['n_expected']} tests")

    sb = Sandbox(image=rec["docker_image"], name=name)
    sb._pod = name  # attach to the already-running sandbox from r2e_probe

    rc, _ = sb.exec(f"test -d {REPO}/r2e_tests", timeout=60)
    if rc != 0:
        print("[grade] staging /r2e_tests into repo")
        sb.exec(f"mv /r2e_tests {REPO}/r2e_tests", timeout=120)

    print("[grade] --- BEFORE gold patch (expect reward 0) ---", flush=True)
    before = run_tests(sb)
    r0, got0, exp = reward(before, rec["expected_output_json"])
    print(f"[grade] reward_before = {r0}")
    print(f"[grade] parsed {len(got0)} tests, expected {len(exp)}")
    for k in sorted(exp)[:6]:
        print(f"    {k}: got={got0.get(k, '<missing>')} want={exp[k]}")

    print("\n[grade] applying gold patch ...", flush=True)
    sb.write_file("/tmp/gold.patch", rec["patch"])
    rc, out = sb.exec(f"cd {REPO} && git apply -v /tmp/gold.patch 2>&1", timeout=300)
    print(f"[grade] git apply rc={rc}\n{out.strip()[:600]}")

    print("\n[grade] --- AFTER gold patch (expect reward 1) ---", flush=True)
    after = run_tests(sb)
    r1, got1, _ = reward(after, rec["expected_output_json"])
    print(f"[grade] reward_after = {r1}")
    print(f"[grade] parsed {len(got1)} tests, expected {len(exp)}")
    for k in sorted(exp)[:6]:
        print(f"    {k}: got={got1.get(k, '<missing>')} want={exp[k]}")

    print(f"\n[grade] RESULT before={r0} after={r1} "
          f"{'OK — reward signal reproduced' if (r0 == 0.0 and r1 == 1.0) else 'MISMATCH'}")
    if r1 != 1.0:
        print("\n[grade] tail of after-log:\n" + after[-2500:])


if __name__ == "__main__":
    main()
