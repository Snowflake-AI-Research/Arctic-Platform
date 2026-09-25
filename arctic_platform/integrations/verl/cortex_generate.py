# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cortex generate: chat-template string, not decode of padded ids.

Arctic ``generate`` is string-in. TRL/SkyRL post
``apply_chat_template(..., tokenize=False)``. VeRL Cortex used to decode
already-collated ``prompt_token_ids``, which can inject pad tokens into the
prefix the zone re-tokenizes. Training still concatenates the original ids, so
the sampled continuation is not the sequence ``/fwd-bwd`` scores.

This module is Cortex-only. On-prem VeRL Arctic keeps the historical decode.
"""

from __future__ import annotations

from typing import Any


def strip_pad_ids(ids: list[int], pad_token_id: int | None) -> list[int]:
    out = [int(x) for x in ids]
    if pad_token_id is None:
        return out
    pad = int(pad_token_id)
    while out and out[0] == pad:
        out = out[1:]
    while out and out[-1] == pad:
        out = out[:-1]
    return out


def prompt_text_from_ids(tokenizer: Any, prompt_ids) -> str:
    """Decode unpadded ids with special tokens kept (chat markers must survive)."""
    ids = strip_pad_ids(list(prompt_ids), getattr(tokenizer, "pad_token_id", None))
    return tokenizer.decode(ids, skip_special_tokens=False)


_INSTALLED = False


def install_cortex_string_chat_template() -> None:
    """GSM8K-style loops: template string then encode, matching TRL ``_render_prompt``.

    ``AgentLoopOutput.prompt_ids`` is what train concatenates. If those ids are
    ``encode(text)`` and generate posts ``text``, the zone prefix matches.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopBase
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import normalize_token_ids
    except ImportError:
        return

    orig = AgentLoopBase.apply_chat_template

    async def apply_chat_template_cortex(
        self,
        messages,
        tools=None,
        images=None,
        videos=None,
        remove_system_prompt: bool = False,
    ):
        if getattr(self, "processor", None) is not None or images or videos:
            return await orig(
                self,
                messages,
                tools=tools,
                images=images,
                videos=videos,
                remove_system_prompt=remove_system_prompt,
            )
        kwargs = dict(getattr(self, "apply_chat_template_kwargs", None) or {})

        def _render() -> str:
            return apply_chat_template(
                self.tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )

        text = await self.loop.run_in_executor(None, _render)
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if remove_system_prompt:
            ids = ids[len(self.system_prompt) :]
        return normalize_token_ids(ids)

    AgentLoopBase.apply_chat_template = apply_chat_template_cortex
    _INSTALLED = True

    async def apply_chat_template_cortex(
        self,
        messages,
        tools=None,
        images=None,
        videos=None,
        remove_system_prompt: bool = False,
    ):
        if getattr(self, "processor", None) is not None or images or videos:
            return await orig(
                self,
                messages,
                tools=tools,
                images=images,
                videos=videos,
                remove_system_prompt=remove_system_prompt,
            )
        kwargs = dict(getattr(self, "apply_chat_template_kwargs", None) or {})

        def _render() -> str:
            return apply_chat_template(
                self.tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )

        text = await self.loop.run_in_executor(None, _render)
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if remove_system_prompt:
            ids = ids[len(self.system_prompt) :]
        return normalize_token_ids(ids)

    AgentLoopBase.apply_chat_template = apply_chat_template_cortex
