#!/usr/bin/env python3
"""Time the agent's other per-turn cost: running a command in the sandbox.

Each turn does one ``kubectl exec``, which is a fresh process, a TLS
handshake, an API-server round trip and a stream upgrade before the command
runs at all. None of that scales with the model, and all of it lands on the
driver pod and the API server, so it is a plausible serial ceiling for a loop
that does tens of thousands of turns per step.

Measures a trivial command, so the number is overhead rather than work, both
one at a time and under the concurrency the driver actually uses.

  python3 bench_sandbox.py <pod-name> [concurrency] [samples]
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from sandbox import _kubectl  # noqa: E402

NAMESPACE = "default"


def one(pod: str) -> float:
    t = time.time()
    _kubectl("exec", pod, "-n", NAMESPACE, "--", "bash", "-lc", "true", timeout=120)
    return time.time() - t


def main() -> int:
    pod = sys.argv[1]
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    serial = [one(pod) for _ in range(samples)]
    print(f"serial exec   : median {statistics.median(serial):.2f}s  "
          f"min {min(serial):.2f}s  max {max(serial):.2f}s  (n={samples})")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        conc = list(pool.map(lambda _: one(pod), range(concurrency)))
    wall = time.time() - t0

    print(f"{concurrency} concurrent : wall {wall:.2f}s  "
          f"median {statistics.median(conc):.2f}s  max {max(conc):.2f}s")
    print(f"throughput    : {concurrency / wall:.2f} exec/s")
    print()
    # A turn needs one exec, so this is a hard ceiling on turns per second no
    # matter how fast the sampler is.
    print(f"implies a ceiling of {concurrency / wall:.2f} turns/s from the sandbox alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
