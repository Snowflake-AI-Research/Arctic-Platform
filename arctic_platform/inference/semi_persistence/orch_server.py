"""Long-running orchestrator server.

Boots :class:`Orchestrator` (which starts the dashboard / control HTTP
server inside ``Orchestrator.init``) and parks the process so clients
can drive it remotely via the HTTP control plane in
``state_server.StateHandler``.

Usage:
    python orch_server.py --image-cache /data-fast/image-cache/demo_5 \
                          --gpus 0,1,2,3 --port 8157
"""
from __future__ import annotations

import argparse
import signal
import threading

from orchestrator import Orchestrator


def _parse_gpus(raw: str) -> list[int] | None:
    """Parse a comma-separated GPU list; empty string means 'auto-discover'."""
    raw = raw.strip()
    if not raw:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-running orchestrator server.")
    parser.add_argument(
        "--image-cache", required=True,
        help="Directory holding model images (mirrors Orchestrator.init).")
    parser.add_argument(
        "--gpus", default="",
        help="Comma-separated GPU ids (e.g. '0,1,2,3'). Empty = NVML auto-discover.")
    parser.add_argument(
        "--port", type=int, default=8157,
        help="HTTP port for /state and the control plane (default 8157).")
    args = parser.parse_args()

    gpus = _parse_gpus(args.gpus)
    Orchestrator.init(args.image_cache, gpus, dashboard_port=args.port)
    print(f"[orch_server] listening on :{args.port}  (Ctrl-C to stop)")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    stop.wait()
    print("[orch_server] shutting down")


if __name__ == "__main__":
    main()
