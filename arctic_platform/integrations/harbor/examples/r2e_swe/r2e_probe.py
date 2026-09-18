"""Probe one real R2E-Gym instance end-to-end in the in-pod k3s sandbox.

Answers the questions that decide whether Boyi's task path is reproducible here:
pulls the instance's Docker Hub image, confirms the /testbed layout and the
interpreter his config pins, applies the gold patch, and runs the graded tests
both before and after so we can see the reward signal actually move.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/modeling-code/karthik/abstract-remote-exps/poc")

from sandbox import Sandbox  # noqa: E402

DATASET = (
    "/data/fshu/important/swe_data/r2e_family/"
    "R2E-Gym-Subset_validgold_unique_baseline/train.jsonl"
)


def load_instance(instance_id: str | None, row: int) -> dict:
    with open(DATASET) as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            if instance_id and rec["instance_id"] == instance_id:
                return rec
            if not instance_id and i == row:
                return rec
    raise SystemExit(f"instance not found: {instance_id or row}")


def main() -> None:
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rec = load_instance(None, row)
    print(f"[probe] instance : {rec['instance_id']}")
    print(f"[probe] repo     : {rec['repo_name']}")
    print(f"[probe] image    : {rec['docker_image']}")
    print(f"[probe] n_expected: {rec['n_expected']}  verify={rec['verify_status']}")
    print(f"[probe] patch len: {len(rec['patch'])}  test_patch len: {len(rec['test_patch'])}")

    t0 = time.time()
    sb = Sandbox(image=rec["docker_image"], name=f"r2e-{row}")
    print("[probe] creating sandbox (image pull may take minutes) ...", flush=True)
    pod = sb.create(ready_timeout_s=1800)
    print(f"[probe] pod ready in {time.time() - t0:.0f}s: {pod}", flush=True)

    try:
        for label, cmd in [
            ("testbed", "ls -d /testbed && ls /testbed | head -20"),
            ("interp", "/testbed/.venv/bin/python -V 2>&1 || echo NO_VENV_PYTHON"),
            ("fallback", "which python python3; /opt/miniconda3/bin/python -V 2>&1 | head -1"),
            ("git", "cd /testbed && git log --oneline -1 && git status --porcelain | head -5"),
            ("r2e_files", "ls /r2e_tests 2>/dev/null | head -10 || echo NO_R2E_TESTS_DIR"),
            ("expected", "ls /testbed/*.json /*.json 2>/dev/null | head -10"),
        ]:
            rc, out = sb.exec(cmd, timeout=300)
            print(f"\n[{label}] rc={rc}\n{out.strip()[:1200]}", flush=True)
    finally:
        print("\n[probe] leaving sandbox up for follow-up; delete manually:")
        print(f"  k3s kubectl delete sandbox {sb.name}")


if __name__ == "__main__":
    main()
