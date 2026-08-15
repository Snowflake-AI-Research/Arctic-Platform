# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wire-shape tests for the driver-side OpenAI-compat gateway.

The gateway is what makes "any Harbor agent trains on Cortex" work
without a Cortex control-plane change: it re-exposes ``/v1/*`` locally
and forwards each call to ``ArcticRLClient.generate``. These tests
stand up a real uvicorn on ``127.0.0.1``, hit it with the real
``openai`` and ``LiteLLM`` clients, and assert the response shape
matches what those clients expect. The ``ArcticRLClient`` is a fake
that returns deterministic tokens; anything above the client is
exercised for real.
"""

from __future__ import annotations

from typing import Any

import pytest


pytest.importorskip("openai")
pytest.importorskip("transformers")


class _FakeArcticRLClient:
    """Stand-in for ``ArcticRLClient`` that echoes deterministic tokens.

    The real client's ``.generate`` is synchronous and returns
    ``list[dict]`` shaped as ``{text, token_ids, finish_reason,
    prompt_len, generation_len, ...}`` — exactly what the gateway
    forwards to the OpenAI-compat router. Matching that shape is the
    only contract the fake needs to honor.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        prompts: list[Any],
        sampling_params: dict[str, Any],
        routing_key: Any = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append({"prompts": prompts, "sampling_params": sampling_params})
        n = int(sampling_params.get("n", 1))
        max_tokens = int(sampling_params.get("max_tokens", 4))
        results: list[dict[str, Any]] = []
        # One result per prompt (the router calls with a 1-element list
        # and ``n=1``; we still handle the ``n>1``-per-prompt case
        # defensively).
        for _ in range(max(1, len(prompts)) * n):
            token_ids = list(range(1000, 1000 + max_tokens))
            results.append({
                "text": "ok",
                "token_ids": token_ids,
                "finish_reason": "stop",
                "prompt_len": 5,
                "generation_len": max_tokens,
            })
        return results


@pytest.fixture
def gateway_and_client():
    """Boot the gateway with a fake client + a real Qwen chat tokenizer.

    Qwen's tokenizer defines ``chat_template``, which the router
    requires for ``/v1/chat/completions``. Using a real tokenizer here
    means the ``apply_chat_template`` code path is exercised end-to-end
    instead of stubbed out.
    """
    from transformers import AutoTokenizer

    from arctic_platform.integrations.harbor.openai_gateway import DriverOpenAIGateway

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    client = _FakeArcticRLClient()
    gateway = DriverOpenAIGateway(
        client=client, tokenizer=tokenizer, model_name="Qwen/Qwen3-0.6B"
    )
    gateway.start()
    try:
        yield gateway, client
    finally:
        gateway.stop()


def test_models_endpoint(gateway_and_client) -> None:
    import openai

    gateway, _ = gateway_and_client
    oa = openai.OpenAI(base_url=gateway.base_url, api_key="not-used")
    models = oa.models.list()
    ids = [m.id for m in models.data]
    assert ids == ["Qwen/Qwen3-0.6B"]


def test_chat_completion_wire_shape(gateway_and_client) -> None:
    """Real ``openai.OpenAI`` client -> gateway -> fake ``ArcticRLClient``."""
    import openai

    gateway, client = gateway_and_client
    oa = openai.OpenAI(base_url=gateway.base_url, api_key="not-used")
    resp = oa.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.7,
    )

    assert resp.model == "Qwen/Qwen3-0.6B"
    assert len(resp.choices) == 1
    choice = resp.choices[0]
    assert choice.message.content == "ok"
    assert choice.finish_reason == "stop"

    # vLLM's OpenAI extensions: prompt_token_ids (top-level) +
    # per-choice token_ids. Harbor's ``LiteLLM._extract_token_ids``
    # reads these to populate ``RolloutDetail``, so a missing field
    # here would silently blank out the rollout token ids.
    raw = resp.model_dump()
    assert isinstance(raw.get("prompt_token_ids"), list)
    assert raw["prompt_token_ids"], "prompt_token_ids must be non-empty"
    assert raw["choices"][0]["token_ids"] == list(range(1000, 1008))

    # Client-side sanity: the fake saw exactly one call with our
    # sampling params forwarded through the OpenAI -> vLLM translation.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["sampling_params"]["max_tokens"] == 8
    assert call["sampling_params"]["temperature"] == 0.7


def test_chat_completion_n_gt_1(gateway_and_client) -> None:
    """``n>1`` fans out through the router's ``_generate_n``."""
    import openai

    gateway, client = gateway_and_client
    oa = openai.OpenAI(base_url=gateway.base_url, api_key="not-used")
    resp = oa.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        n=3,
    )
    assert len(resp.choices) == 3
    # Router turns ``n=k`` into a single batched call with the prompt
    # repeated k times and ``n=1``, so the sampling sub-job's prefix
    # cache dedups the prompt across samples.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert len(call["prompts"]) == 3
    assert call["sampling_params"]["n"] == 1


@pytest.mark.asyncio
async def test_harbor_litellm_integration(gateway_and_client) -> None:
    """Harbor's LiteLLM backend reads ``token_ids`` off the response.

    Same assertion as ``test_harbor_litellm_integration.py``, applied
    to the driver-side gateway rather than the sub-job router. If this
    passes, any Harbor agent that goes through
    ``LiteLLM(collect_rollout_details=True)`` gets populated
    ``RolloutDetail`` from a Cortex-trained model over the same
    ``ArcticRLClient.generate`` path our native ``CortexRLAgent`` uses.
    """
    lite_llm = pytest.importorskip("harbor.llms.lite_llm")

    gateway, _ = gateway_and_client
    llm = lite_llm.LiteLLM(
        model_name="hosted_vllm/qwen3-0p6b",
        api_base=gateway.base_url,
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

    assert response.completion_token_ids and all(
        tok >= 1000 for tok in response.completion_token_ids
    ), (
        "LiteLLM._extract_token_ids failed to pull completion token_ids from "
        "the gateway response — either the wire shape doesn't match vLLM's "
        "OpenAI extension, or LiteLLM isn't routing choices[].token_ids "
        "through as provider_specific_fields."
    )
    assert response.prompt_token_ids and len(response.prompt_token_ids) > 0, (
        "gateway didn't include top-level prompt_token_ids, or LiteLLM "
        "didn't route it through."
    )
