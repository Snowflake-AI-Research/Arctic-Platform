#!/usr/bin/env python3
"""A black-box shell agent, run inside the sandbox.

Deliberately dependency-free (stdlib urllib only) so it can be dropped into any
sandbox image without pip or network egress, following the same pattern as
prime-rl's mini_swe_agent_plus program.

It knows nothing about RL: no token ids, no logprobs, no rollout bookkeeping. It
reads an instruction, then loops "ask the model -> run one bash command -> feed
the output back". Everything needed for training is captured by the gateway on
the other end of --base-url.

The rollout id travels as the bearer token so the gateway can attribute captured
tokens to this trajectory without the agent participating.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

SYSTEM = """You are a software engineer working in a Linux shell.

Respond with EXACTLY ONE bash command in a fenced block, and nothing else:

```bash
your command here
```

The command's output is returned to you. Work in small steps.

Rules:
- Actually edit the file. Write the whole corrected file with a heredoc, e.g.
  cat > /path/file.py <<'EOF' ... EOF
- Inspecting a file does not fix it.
- Run the verification command given in the task before you finish.
- Reply with exactly TASK_DONE only after the verification command has printed
  its success message. Never say TASK_DONE before that.
"""

BLOCK = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.DOTALL)


def complete(base_url, api_key, model, messages, max_tokens, temperature):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    request = urllib.request.Request(
        "{}/chat/completions".format(base_url.rstrip("/")),
        data=body,
        headers={
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def run_bash(command, working_dir, timeout):
    try:
        done = subprocess.run(
            ["bash", "-lc", command],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (done.stdout or "") + (done.stderr or "")
        return "exit={}\n{}".format(done.returncode, out[:4000])
    except subprocess.TimeoutExpired:
        return "exit=timeout after {}s".format(timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True, help="carries the rollout id")
    ap.add_argument("--model", required=True)
    ap.add_argument("--instruction-file", required=True)
    ap.add_argument("--working-dir", default="/workspace")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--command-timeout", type=int, default=60)
    args = ap.parse_args()

    with open(args.instruction_file) as fh:
        instruction = fh.read()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]

    for turn in range(args.max_turns):
        try:
            response = complete(
                args.base_url,
                args.api_key,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            detail = exc.read().decode(errors="replace") if hasattr(exc, "read") else str(exc)
            print("[agent] completion failed: {}".format(detail), file=sys.stderr)
            return 1

        text = (response["choices"][0]["message"].get("content") or "").strip()
        print("[agent] turn {} model: {}".format(turn, text[:300]), flush=True)
        messages.append({"role": "assistant", "content": text})

        if "TASK_DONE" in text:
            print("[agent] model declared done", flush=True)
            return 0

        match = BLOCK.search(text)
        if match is None:
            messages.append(
                {
                    "role": "user",
                    "content": "No bash block found. Reply with exactly one ```bash block, or TASK_DONE.",
                }
            )
            continue

        result = run_bash(match.group(1).strip(), args.working_dir, args.command_timeout)
        print("[agent] turn {} result: {}".format(turn, result[:300]), flush=True)
        messages.append({"role": "user", "content": result})

    print("[agent] out of turns", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
