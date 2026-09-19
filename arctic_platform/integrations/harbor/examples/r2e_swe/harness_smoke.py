"""Run the mini-swe-agent-plus harness against a mock that mimics our gateway.

The question this answers is narrow and worth isolating: does the response shape
our ``openai_compat`` router now emits survive the reference protocol validator? The reference
validator rejects visible content, missing reasoning, more than one call per
turn, unknown tools, and bad ids — any of which zeroes a rollout silently. A
mock costs no GPU time and no Cortex job, so failures here are cheap.

The mock replays what the router produces after parsing Qwen3.5's XML envelope:
``content=None``, ``reasoning_content`` set, one ``tool_calls`` entry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mini_swe_plus  # noqa: E402
from sandbox import Sandbox  # noqa: E402

BRIDGE = "10.42.0.1"
PORT = 19301

SUBMIT = "echo MINI_SWE_AGENT_FINAL_OUTPUT"

# One scripted trajectory: look around, make an edit, then submit with the
# exact command the reference submission policy requires.
SCRIPT = [
    ("execute_bash", {"command": "ls /testbed | head -5"}),
    ("edit_via_str_replace", {
        "path": "/tmp/probe.py",
        "old_str": "return 1",
        "new_str": "return 2",
    }),
    ("execute_bash", {"command": SUBMIT}),
]


class Handler(BaseHTTPRequestHandler):
    turn = 0
    lock = threading.Lock()
    seen_tools: list = []

    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with Handler.lock:
            i = min(Handler.turn, len(SCRIPT) - 1)
            Handler.turn += 1
            if body.get("tools"):
                Handler.seen_tools = [
                    t["function"]["name"] for t in body["tools"]
                ]
        name, args = SCRIPT[i]
        payload = {
            "id": f"chatcmpl-{i}",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "mock"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": f"Step {i}: calling {name}.",
                    "tool_calls": [{
                        "id": f"call_{i:024d}",
                        "type": "function",
                        "index": 0,
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }],
                },
                "finish_reason": "tool_calls",
                "token_ids": [1, 2, 3],
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            "prompt_token_ids": [1, 2, 3, 4, 5],
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "r2e-0"
    srv = ThreadingHTTPServer((BRIDGE, PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[smoke] mock gateway on http://{BRIDGE}:{PORT}", flush=True)

    sb = Sandbox(name=name)
    sb._pod = name  # attach to the live R2E sandbox

    print("[smoke] staging harness ...", flush=True)
    mini_swe_plus.stage(sb, python_command="/testbed/.venv/bin/python")
    # The editor turn needs a real file to rewrite, so the str-replace path is
    # exercised rather than short-circuiting on a missing path.
    sb.write_file("/tmp/probe.py", "def f():\n    return 1\n")

    rc, out = sb.exec(f"ls -la /tmp/verifiers/ && {'/testbed/.venv/bin/python'} -V", timeout=120)
    print(f"[smoke] staged rc={rc}\n{out.strip()[:700]}\n", flush=True)

    print("[smoke] running harness ...", flush=True)
    res = mini_swe_plus.run(
        sb,
        base_url=f"http://{BRIDGE}:{PORT}/v1",
        api_key="mock-key",
        model="Qwen/Qwen3.5-4B",
        task="Print the first few files in /testbed, then submit.",
        working_dir="/testbed",
        timeout=900,
    )
    print(f"[smoke] exit_code = {res['exit_code']}")
    print(f"[smoke] tools offered by harness = {Handler.seen_tools}")
    print(f"[smoke] model turns served = {Handler.turn}")
    print(f"[smoke] stop = {json.dumps(res['stop'], indent=2) if res['stop'] else None}")
    print(f"[smoke] advisories = {res['advisories']}")
    print("\n[smoke] ---- stdout tail ----")
    print(res["stdout"][-3000:])


if __name__ == "__main__":
    main()
