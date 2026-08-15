# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stock-shape Harbor ``BaseAgent`` that talks OpenAI-chat via ``LiteLLM``.

Serves two purposes:

1. A minimum-viable Harbor chat agent — any user could write this from
   scratch. Nothing here reaches into Arctic-Platform internals; it
   only uses ``harbor.llms.lite_llm.LiteLLM`` and ``BaseAgent``. Points
   ``api_base`` at whatever OpenAI-compat endpoint the trial gets
   (external, vLLM, or our driver-side gateway on Cortex).

2. The training-time counterpart of ``CortexRLAgent`` for the
   OpenAI-compat sampling mode. When Harbor calls the LLM with
   ``collect_rollout_details=True``, the response carries
   ``prompt_token_ids`` and ``completion_token_ids`` from vLLM's
   OpenAI extension (echoed by our gateway or a real vLLM server). We
   surface those as a ``RolloutDetail`` so the adapter can build a
   GRPO batch — the same batch shape ``CortexRLAgent`` produces
   through the native ``ArcticRLClient.generate`` path.

If Harbor lands the ``PostTrainingBackend`` RFC, this class stays
useful outside Arctic-Platform: swap ``api_base`` for any OpenAI-compat
endpoint and any RL backend can drive it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import OutputLengthExceededError
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext
from harbor.models.agent.rollout_detail import RolloutDetail


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _extract_final_answer(text: str) -> str:
    """Pull the model's final answer out of a chain-of-thought reply.

    Preference order: ``\\boxed{...}`` (math benchmarks), ``<answer>...
    </answer>`` (reasoning-gym / open-thought convention), last
    non-empty line. Falls back to the full text if nothing else
    matches, so tasks that consume the raw completion still see it.
    """
    if not text:
        return ""
    m = list(_BOXED.finditer(text))
    if m:
        return m[-1].group(1).strip()
    m2 = _ANSWER_TAG.search(text)
    if m2:
        return m2.group(1).strip()
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().rstrip(".")
        if stripped:
            return stripped
    return text.strip()


class LiteLLMChatAgent(BaseAgent):
    """One-turn chat agent driven by ``harbor.llms.lite_llm.LiteLLM``.

    Constructor kwargs mirror the fields Harbor's ``harbor run``
    populates via ``-m`` / ``--model-base-url`` / ``--ak``: ``api_base``,
    ``model_name``, ``temperature``, ``max_tokens``. Extra ``**kw`` are
    passed through to ``BaseAgent`` so future Harbor releases can add
    fields without breaking the constructor.
    """

    # These flags mirror ``CortexRLAgent``'s: single-turn, no ATIF
    # tool-invocation, no resume, no rolling windows. Harbor infra
    # keys off them.
    SUPPORTS_ATIF = False
    SUPPORTS_RESUME = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path | str,
        model_name: str | None = None,
        *,
        api_base: str | None = None,
        max_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kw: Any,
    ) -> None:
        forward = {k: v for k, v in kw.items() if k in ("logger", "mcp_servers", "skills_dir", "extra_env")}
        super().__init__(logs_dir=Path(logs_dir), model_name=model_name, **forward)
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        self._logs_dir = Path(logs_dir)
        if not api_base:
            raise RuntimeError(
                "LiteLLMChatAgent needs --model-base-url or --ak api_base=... — "
                "pass an OpenAI-compat endpoint (driver-side gateway URL for "
                "Cortex, or a vLLM host for on-prem)."
            )
        if not model_name:
            raise RuntimeError("LiteLLMChatAgent needs -m <model_name>")
        self._api_base = api_base
        self._model_name = model_name
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._top_p = float(top_p)

    @staticmethod
    def name() -> str:
        return "litellm-chat"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment: BaseEnvironment) -> None:  # noqa: ARG002
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        import tempfile
        # ``model_name="hosted_vllm/<slug>"`` is the LiteLLM idiom for
        # "OpenAI-compat vLLM server at ``api_base``". Harbor's
        # ``validate_hosted_vllm_model_config`` requires exactly one
        # ``/`` in the value, so fold ``org/repo`` HF names into a
        # single-segment slug (``Qwen/Qwen3-0.6B`` -> ``Qwen-Qwen3-0.6B``).
        # ``collect_rollout_details=True`` opts into vLLM's
        # ``prompt_token_ids`` / ``completion_token_ids`` extension.
        litellm_slug = self._model_name.replace("/", "-")
        llm = LiteLLM(
            model_name=f"hosted_vllm/{litellm_slug}",
            api_base=self._api_base,
            temperature=self._temperature,
            collect_rollout_details=True,
            model_info={
                "max_input_tokens": 4096,
                "max_output_tokens": 4096,
                # Cost gating in LiteLLM's model_info blocks self-hosted
                # models unless we spell out zero pricing.
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        )
        # Harbor's LiteLLM raises ``OutputLengthExceededError`` on
        # ``finish_reason=length`` unconditionally. For RL sampling
        # that's a bug of Harbor's semantic contract: a truncated
        # rollout is still a valid rollout (the trainer sees the
        # generated tokens as-is and shapes reward from them). Catch
        # it, salvage ``truncated_response`` text, and lose only
        # ``completion_token_ids`` — the trial is then reward=0 but
        # not silently dropped from the batch.
        try:
            response = await llm.call(
                prompt=instruction,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )
            completion_text = response.content or ""
            prompt_token_ids = list(response.prompt_token_ids or [])
            completion_token_ids = list(response.completion_token_ids or [])
        except OutputLengthExceededError as exc:
            completion_text = exc.truncated_response or ""
            prompt_token_ids = []
            completion_token_ids = []

        # Persist locally for post-mortem, then hand the completion to
        # the environment at the two well-known paths Harbor tasks read:
        # * ``/logs/agent/completion.txt`` — the arithmetic verifier in
        #   this package.
        # * ``/workspace/answer.txt`` — reasoning-gym and other
        #   ``open-thought`` derivatives ask the agent to write there.
        # ``environment.upload_file`` routes through the environment's
        # own path translation (HostEnvironment rewrites both prefixes
        # under the trial's host root; container envs put them at the
        # real container paths). Missing prompt or completion token ids
        # would silently drop the trial from the GRPO batch (see
        # ``arctic_platform.integrations.harbor.adapter``).
        (self._logs_dir / "completion.txt").write_text(completion_text)
        (self._logs_dir / "prompt.txt").write_text(instruction)

        # Extract the final integer answer for tasks that expect a
        # short answer (reasoning-gym scoring is exact-match on the
        # last non-empty line, so writing the whole CoT trips scoring).
        answer_short = _extract_final_answer(completion_text)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(answer_short + "\n")
            tmp_path = tmp.name
        try:
            await environment.upload_file(tmp_path, "/workspace/answer.txt")
            await environment.upload_file(tmp_path, "/logs/agent/completion.txt")
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

        detail: RolloutDetail = {
            "prompt_token_ids": [prompt_token_ids],
            "completion_token_ids": [completion_token_ids],
        }
        context.rollout_details = [detail]
