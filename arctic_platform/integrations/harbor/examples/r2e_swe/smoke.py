#!/usr/bin/env python3
"""Pre-flight for the E2E POC, with no Cortex job and no GPUs.

Stands a fake OpenAI server on the bridge that emits vLLM-style token-id
extensions and canned bash fixes, then runs the real agent inside a real sandbox
against it. Exercises everything except Cortex itself: sandbox lifecycle, the
bridge route, the agent loop, token capture, and reward scoring.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sandbox import BRIDGE_HOST, Sandbox  # noqa: E402
from tasks import TASKS  # noqa: E402

PORT = 19915
CAPTURED: list[tuple[str, int, int]] = []

# Turn 0 fixes the defect, turn 1 declares completion.
REPLIES = [
    "```bash\nprintf 'def add(a, b):\\n    return a + b\\n' > /workspace/calc.py\n```",
    "TASK_DONE",
]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        rollout_id = (self.headers.get("authorization") or "").removeprefix("Bearer ").strip()

        n_assistant = sum(1 for m in body.get("messages", []) if m.get("role") == "assistant")
        content = REPLIES[min(n_assistant, len(REPLIES) - 1)]

        # Mimic the vLLM OpenAI extensions the real router emits, so the capture
        # path under test is the same one production uses.
        prompt_token_ids = list(range(100, 100 + 20 + 5 * n_assistant))
        token_ids = list(range(500, 500 + max(len(content) // 4, 1)))
        CAPTURED.append((rollout_id, len(prompt_token_ids), len(token_ids)))

        payload = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion",
            "model": body.get("model", "smoke"),
            "prompt_token_ids": prompt_token_ids,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "token_ids": token_ids,
                }
            ],
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def main() -> int:
    server = HTTPServer((BRIDGE_HOST, PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[smoke] fake model server on http://{BRIDGE_HOST}:{PORT}/v1", flush=True)

    task = TASKS[0]
    agent_source = (Path(__file__).resolve().parent / "sandbox_agent.py").read_text()
    sandbox = Sandbox()
    try:
        print("[smoke] creating sandbox ...", flush=True)
        pod = sandbox.create()
        print(f"[smoke] sandbox pod {pod} running", flush=True)

        sandbox.write_file("/tmp/sandbox_agent.py", agent_source)
        sandbox.write_file("/tmp/instruction.txt", task.instruction)

        code, out = sandbox.exec(task.setup)
        print(f"[smoke] setup exit={code} {out.strip()[:120]}", flush=True)

        code, out = sandbox.exec(
            "python /tmp/sandbox_agent.py"
            f" --base-url http://{BRIDGE_HOST}:{PORT}/v1"
            " --api-key smoke-rollout-1"
            " --model smoke"
            " --instruction-file /tmp/instruction.txt"
            " --max-turns 3"
        )
        print(f"[smoke] agent exit={code}", flush=True)
        for line in out.splitlines():
            print(f"[smoke]   | {line[:160]}", flush=True)

        code, out = sandbox.exec(task.test)
        reward = 1.0 if code == 0 else 0.0
        print(f"[smoke] test exit={code} reward={reward} {out.strip()[:160]}", flush=True)

        print(f"[smoke] captured {len(CAPTURED)} completions: {CAPTURED}", flush=True)
        ok = reward == 1.0 and len(CAPTURED) >= 2
        print(f"[smoke] {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    finally:
        sandbox.delete()
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
