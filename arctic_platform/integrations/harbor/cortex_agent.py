# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""A minimal Harbor BaseAgent that samples from a Cortex sub-job.

Reattaches to an already-running training/sampling pair via
``ArcticRLClient.reconnect_config`` (the exact mechanism SkyRL / verl use for
their driver-side workers), calls the sampling sub-job for one completion,
and writes a Harbor-shaped ``RolloutDetail`` so downstream (adapter -> Arctic
GRPO backend) can consume it 1:1. No LiteLLM proxy, no HTTP hop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.rollout_detail import RolloutDetail


class CortexRLAgent(BaseAgent):
    """One-turn agent: prompt -> sampling sub-job -> RolloutDetail."""

    SUPPORTS_ATIF = False
    SUPPORTS_RESUME = False
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path | str,
        model_name: str | None = None,
        *,
        reconnect_config_path: str | None = None,
        max_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kw: Any,
    ) -> None:
        forward = {k: v for k, v in kw.items() if k in ("logger", "mcp_servers", "skills_dir", "extra_env")}
        super().__init__(logs_dir=Path(logs_dir), model_name=model_name, **forward)
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        self._logs_dir = Path(logs_dir)
        self._reconnect_config_path = reconnect_config_path or os.environ.get(
            "ARCTIC_RECONNECT_CONFIG_PATH"
        )
        if not self._reconnect_config_path:
            raise RuntimeError(
                "CortexRLAgent needs --ak reconnect_config_path=<path> pointing to a "
                "JSON dump of ArcticRLClient.reconnect_config()"
            )
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._top_p = float(top_p)

    @staticmethod
    def name() -> str:
        return "cortex-rl"

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
        from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client

        cfg_dict = json.loads(Path(self._reconnect_config_path).read_text())
        client = create_arctic_rl_client(ArcticRLClientConfig(**cfg_dict))

        # Chat-template the instruction so tokens match how the trainer sees them.
        from transformers import AutoTokenizer

        # Harbor's BaseAgent splits ``Qwen/Qwen3-0.6B`` into provider+name for
        # display; the HuggingFace tokenizer needs the full "org/repo" string.
        tokenizer_name = self.model_name or cfg_dict.get("model_name")
        assert tokenizer_name, "CortexRLAgent needs -m <model_name> so the tokenizer can be loaded"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [{"role": "user", "content": instruction}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # older tokenizers without enable_thinking
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

        sampling_params = {
            "temperature": self._temperature,
            "top_p": self._top_p,
            "n": 1,
            "max_tokens": self._max_tokens,
        }
        results = await client.generate(prompts=[prompt_text], sampling_params=sampling_params)
        # Do NOT call ``client.shutdown()`` here — the shim's shutdown POSTs
        # ``/:cancel`` to the parent Cortex job, which would tear down the
        # shared training + sampling sub-jobs owned by the runner. Sockets
        # get reclaimed by GC when the trial process exits.

        completion = results[0]
        completion_text = completion.get("text", "")
        completion_token_ids = list(completion.get("token_ids", []))

        # Write the completion out so the verifier (which sees the trial dir,
        # not this Python object) can score it.
        (self._logs_dir / "completion.txt").write_text(completion_text)
        (self._logs_dir / "prompt.txt").write_text(prompt_text)

        detail: RolloutDetail = {
            "prompt_token_ids": [list(prompt_token_ids)],
            "completion_token_ids": [completion_token_ids],
            # Cortex generate doesn't return per-token logprobs; the RolloutDetail
            # logprobs field is optional (total=False on the TypedDict) and the
            # server-side GRPO loss defaults old_log_probs to logprobs.detach()
            # when absent, which is correct for single-epoch on-policy training.
        }
        context.rollout_details = [detail]
