# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end contract test: the OpenAI Python SDK talking to us over HTTP.

Spins up a live uvicorn process serving ``arctic_platform.openai_compat.router``
with a fake pool + tokenizer under ``app.state`` (so no GPUs, no vLLM, no
Ray), then drives it with the real ``openai`` client. If the router
diverges from the OpenAI wire in a way our unit tests miss — a subtle
SSE frame issue, a missing field, an incompatible content-type — the SDK
will surface it here.

Marked to skip cleanly when ``openai`` isn't installed so CI images that
strip it still pass.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("openai", reason="openai SDK not installed; skipping E2E contract test")
pytest.importorskip("uvicorn", reason="uvicorn not installed; skipping E2E contract test")

from openai import OpenAI  # noqa: E402
import httpx  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(base: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:  # noqa: BLE001 — polling; ignore until ready
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server at {base} never became ready")


_SERVER_SCRIPT = textwrap.dedent(
    """
    import argparse, sys, time
    from fastapi import FastAPI
    import uvicorn

    from arctic_platform.openai_compat import router

    class _Tok:
        chat_template = "<t>"
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "\\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\\nassistant:"

    class _Pool:
        _config = object()
        async def generate(self, prompts, sampling_params=None, model_id=None, routing_key=None, strict=False):
            n = int((sampling_params or {}).get('n', 1))
            per_prompt = []
            for p in prompts:
                per_prompt.append({
                    'text': 'ok',
                    'token_ids': [1, 2],
                    'finish_reason': 'stop',
                    'prompt_len': len(str(p)),
                    'generation_len': 2,
                    'prefix_cache_len': 0,
                })
            return per_prompt

    app = FastAPI()
    app.include_router(router)

    @app.get('/health')
    async def health():
        return {'status': 'OK'}

    app.state.sampling_pool = _Pool()
    app.state.sampling_tokenizer = _Tok()
    app.state.sampling_model_name = 'test-model'
    app.state.sampling_created = int(time.time())

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()

    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
    """
).strip()


@pytest.fixture(scope="module")
def server_base_url():
    port = _pick_port()
    script_path = REPO_ROOT / "tests" / "openai_compat" / "_server_launcher.py"
    script_path.write_text(_SERVER_SCRIPT)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [sys.executable, str(script_path), "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        _wait_ready(base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        try:
            script_path.unlink()
        except FileNotFoundError:
            pass


def test_openai_sdk_chat_completions_non_stream(server_base_url):
    client = OpenAI(base_url=f"{server_base_url}/v1", api_key="not-checked")
    resp = client.chat.completions.create(
        model="test-model",
        messages=[
            {"role": "system", "content": "you are a smoke test"},
            {"role": "user", "content": "say ok"},
        ],
        max_tokens=8,
        temperature=0.0,
    )
    assert resp.choices[0].message.content == "ok"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == "test-model"
    assert resp.usage.completion_tokens == 2


def test_openai_sdk_chat_completions_stream(server_base_url):
    client = OpenAI(base_url=f"{server_base_url}/v1", api_key="not-checked")
    stream = client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    role = None
    content_parts: list[str] = []
    finish = None
    for chunk in stream:
        d = chunk.choices[0].delta
        if getattr(d, "role", None):
            role = d.role
        if getattr(d, "content", None):
            content_parts.append(d.content)
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    assert role == "assistant"
    assert "".join(content_parts) == "ok"
    assert finish == "stop"


def test_openai_sdk_models_list(server_base_url):
    client = OpenAI(base_url=f"{server_base_url}/v1", api_key="not-checked")
    models = list(client.models.list())
    assert [m.id for m in models] == ["test-model"]


def test_openai_sdk_completions(server_base_url):
    client = OpenAI(base_url=f"{server_base_url}/v1", api_key="not-checked")
    resp = client.completions.create(
        model="test-model",
        prompt="2 + 2 =",
        max_tokens=4,
    )
    assert resp.choices[0].text == "ok"
    assert resp.choices[0].finish_reason == "stop"
