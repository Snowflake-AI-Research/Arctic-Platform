# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: Harbor's LiteLLM backend against our OpenAI-compat surface.

This is the client-side E2E that closes the loop on the "any Harbor agent
trains on Cortex" story without needing GPUs. Every Harbor agent that
speaks OpenAI-chat — Terminus 2, LangChain-backed agents, community
``BaseAgent``s — goes through ``harbor.llms.lite_llm.LiteLLM``, which
uses LiteLLM to hit the sampling endpoint. If Harbor + LiteLLM + our
router agree on the wire, any of those agents work.

The test:

* Spawns a live uvicorn server holding ``arctic_platform.openai_compat.router``
  + a fake pool that returns fixed completion tokens.
* Instantiates ``harbor.llms.lite_llm.LiteLLM`` with
  ``collect_rollout_details=True`` — the setting Terminus 2 uses when
  we want to capture ``RolloutDetail`` for GRPO.
* Calls ``llm.call(prompt=...)`` and asserts the returned
  ``LLMResponse`` carries ``prompt_token_ids`` + ``completion_token_ids``
  from our server — i.e. Harbor's rollout capture actually got token ids
  out of the round trip.

If this test passes, the adapter can build a real ``RolloutDataset``
from any Harbor agent's rollouts on top of Cortex, no GPUs required to
prove the wiring.
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

pytest.importorskip("harbor", reason="harbor SDK not installed")
pytest.importorskip("litellm", reason="litellm not installed")
pytest.importorskip("uvicorn", reason="uvicorn not installed")

import httpx  # noqa: E402

from harbor.llms.lite_llm import LiteLLM  # noqa: E402

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
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server at {base} never became ready")


# Server script mirrors ``arctic_platform/rl/http_server`` wiring: mount
# the router + install ``sampling_pool`` / ``sampling_tokenizer`` /
# ``sampling_model_name`` on ``app.state``. We use a deterministic
# encode so the assertions can compare exact ids.
_SERVER_SCRIPT = textwrap.dedent(
    """
    import argparse, sys, time
    from fastapi import FastAPI
    import uvicorn

    from arctic_platform.openai_compat import router

    _COMPLETION_TOKEN_IDS = [11, 22, 33, 44, 55]

    class _Tok:
        chat_template = "<template>"
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "\\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\\nassistant:"
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 100 for c in text]

    class _Pool:
        _config = object()
        async def generate(self, prompts, sampling_params=None, model_id=None, routing_key=None, strict=False):
            out = []
            for p in prompts:
                out.append({
                    'text': 'the answer is 42',
                    'token_ids': list(_COMPLETION_TOKEN_IDS),
                    'finish_reason': 'stop',
                    'prompt_len': len(str(p)),
                    'generation_len': len(_COMPLETION_TOKEN_IDS),
                    'prefix_cache_len': 0,
                })
            return out

    app = FastAPI()
    app.include_router(router)

    @app.get('/health')
    async def health():
        return {'status': 'OK'}

    app.state.sampling_pool = _Pool()
    app.state.sampling_tokenizer = _Tok()
    app.state.sampling_model_name = 'Qwen/Qwen3-0.6B'
    app.state.sampling_created = int(time.time())

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
    """
).strip()


@pytest.fixture(scope="module")
def server_url():
    port = _pick_port()
    script_path = REPO_ROOT / "tests" / "openai_compat" / "_litellm_launcher.py"
    script_path.write_text(_SERVER_SCRIPT)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [sys.executable, str(script_path), "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


@pytest.mark.asyncio
async def test_harbor_litellm_collects_token_ids_from_our_server(server_url):
    """The full Harbor rollout-capture path against our router:

    Harbor LiteLLM  →  LiteLLM  →  HTTP  →  /v1/chat/completions  →  fake pool

    Asserts the returned ``LLMResponse`` has non-empty
    ``prompt_token_ids`` + ``completion_token_ids``. If either is empty
    Harbor's ``RolloutDetail`` would carry no tokens and the adapter
    would silently drop the trial from the GRPO batch — the exact class
    of failure this test guards against.
    """
    llm = LiteLLM(
        model_name="hosted_vllm/qwen3-0p6b",
        api_base=f"{server_url}/v1",
        collect_rollout_details=True,
        temperature=0.0,
        model_info={
            "max_input_tokens": 4096,
            "max_output_tokens": 1024,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    response = await llm.call(prompt="What is 6 * 7?")

    # LLMResponse always carries text.
    assert response.content == "the answer is 42"

    # Harbor's rollout capture: the whole point of the integration.
    assert response.completion_token_ids == [11, 22, 33, 44, 55], (
        "Harbor's LiteLLM._extract_token_ids failed to pull completion "
        "token_ids out of our /v1/chat/completions response — either our "
        "response shape doesn't match vLLM's OpenAI extension, or LiteLLM "
        "isn't passing choices[].token_ids through as provider_specific_fields."
    )
    assert response.prompt_token_ids is not None and len(response.prompt_token_ids) > 0, (
        "response.prompt_token_ids was empty; server didn't include "
        "top-level prompt_token_ids or LiteLLM didn't route it through."
    )
