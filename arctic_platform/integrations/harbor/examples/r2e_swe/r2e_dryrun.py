"""Exercise the R2E rollout path with a mock model, before spending Cortex time.

Everything except the model is real here: real R2E images, real sandboxes, the
real harness, the real grader. The mock plays a deliberately incompetent agent
that looks around and submits without fixing anything, so the expected reward
is 0.0 — what this checks is that a full rollout completes, grades, and tears
down, not that anything gets solved.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r2e_driver import load_instances  # noqa: E402
from r2e_driver import run_one_rollout  # noqa: E402

BRIDGE = "10.42.0.1"
PORT = 19302
SUBMIT = "echo MINI_SWE_AGENT_FINAL_OUTPUT"


class Handler(BaseHTTPRequestHandler):
    """Two turns: one look, then submit."""

    state: dict = {}
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        key = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
        with Handler.lock:
            n = Handler.state.get(key, 0)
            Handler.state[key] = n + 1
        cmd = "ls /testbed | head -3" if n == 0 else SUBMIT
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "mock"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": f"Turn {n}.",
                    "tool_calls": [{
                        "id": f"call_{n:024d}",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": "execute_bash",
                            "arguments": json.dumps({"command": cmd}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
                "token_ids": [1, 2, 3],
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "prompt_token_ids": [1, 2, 3, 4],
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    srv = ThreadingHTTPServer((BRIDGE, PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[dry] mock model on http://{BRIDGE}:{PORT}", flush=True)

    out = Path("/data-fast/poc/r2e-dry/transcripts")
    out.mkdir(parents=True, exist_ok=True)

    instances = load_instances(n, seed=42)
    for rec in instances:
        print(f"\n[dry] {rec['instance_id']} ({rec['repo_name']}) "
              f"n_expected={rec['n_expected']}", flush=True)
        t0 = time.time()
        info = run_one_rollout(
            rec,
            rollout_id=f"dry-{rec['instance_id']}",
            base_url=f"http://{BRIDGE}:{PORT}/v1",
            model="Qwen/Qwen3.5-4B",
            max_turns=10,
            command_timeout=60,
            rollout_timeout=1800,
            setup_timeout=1800,
            transcript_dir=out,
        )
        print(f"[dry] -> {json.dumps(info, indent=2)}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
