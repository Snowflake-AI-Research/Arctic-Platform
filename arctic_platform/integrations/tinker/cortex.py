# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Back the Tinker router's verbs with Cortex Training.

:mod:`arctic_platform.integrations.tinker.router` is a pure adapter: it speaks
Tinker's HTTP protocol and calls five injected handlers. This module builds
those five from :class:`~arctic_platform.client.AsyncArcticRLClient`, so the
Tinker surface reaches Cortex without the on-prem server being involved.

Two shape mismatches are handled here rather than in the router, which stays
backend-agnostic:

``{batch, meta, processing}`` → ``{args, kwargs, context, processing}``
    Cortex takes an RPC-style envelope and resolves loss names in
    ArcticTraining-dss, which registers ``causal_cross_entropy``, ``grpo`` and
    ``grpo_echo_v1`` -- not the ``verl_grpo`` the router asks for.
    :func:`~arctic_platform.integrations._cortex_shared.to_cortex_fwd_bwd_payload`
    does the translation and pins ``grpo``, whose PPO shape is what ``ppo`` and
    ``importance_sampling`` both lower to.

Row alignment
    The router lays a row out as ``[pad… prompt][response pad…]``, real tokens
    contiguous in the middle. Cortex's packer requires them in the *leading*
    columns. Aligning is not enough on its own: the log-probs come back in the
    aligned frame while the router's row slices index the original one, so this
    module inverts the permutation on the way back. Aligning without inverting
    shifts every row's log-probs by its own prompt padding -- no error, just a
    wrong number that looks plausible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from arctic_platform.integrations._cortex_shared import to_cortex_fwd_bwd_payload

if TYPE_CHECKING:
    import torch

    from arctic_platform.client import AsyncArcticRLClient

__all__ = ["CortexTinkerBackend", "build_handlers"]

# Cortex zones register `identity` and `compute_logprobs`; the on-prem names the
# router would otherwise ask for (`compute_entropy_and_logprobs`) do not exist
# there and the zone refuses the request before any model call.
_POST_PROCESSORS = ["compute_logprobs"]


def _align_plan(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(order, valid)`` moving each row's real tokens to its leading columns.

    Same construction as ``_cortex_shared._left_align_batch``, returned rather
    than applied so the caller can invert it. ``order`` is a full permutation of
    the width, which is what makes the inverse an exact scatter.
    """
    import torch

    mask = attention_mask.to(torch.bool)
    width = mask.shape[-1]
    lengths = mask.sum(dim=1)
    valid = torch.arange(width, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
    return order, valid


def _align(batch: dict, order: torch.Tensor, valid: torch.Tensor) -> dict:
    """Gather every full-width 2-D tensor through ``order``.

    Everything moves through the *same* index, so ``advantages`` and
    ``response_mask`` stay on the tokens they scored.
    """
    import torch

    width = valid.shape[-1]
    pad_for = {"labels": -100}

    def move(name: str, t: Any) -> Any:
        if not torch.is_tensor(t) or t.dim() != 2 or t.shape[-1] != width:
            return t
        pad = pad_for.get(name, False if t.dtype == torch.bool else 0)
        return torch.where(valid, t.gather(1, order), torch.full_like(t, pad))

    out = {k: move(k, v) for k, v in batch.items()}
    out["attention_mask"] = valid.to(batch["attention_mask"].dtype)
    return out


def _unalign_rows(aligned: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Invert :func:`_align_plan` for one ``[B, width]`` tensor.

    ``aligned[i, j] == original[i, order[i, j]]``, so scattering along ``order``
    puts every value back where the router expects to slice it.
    """
    import torch

    if aligned.dim() != 2 or aligned.shape != order.shape:
        raise ValueError(
            f"cannot un-align log-probs of shape {tuple(aligned.shape)} against an "
            f"alignment plan of shape {tuple(order.shape)}; the server returned a "
            "frame that does not match the batch that was sent"
        )
    out = torch.zeros_like(aligned)
    return out.scatter(1, order, aligned)


def _forward_payload(batch: dict, order: torch.Tensor, valid: torch.Tensor) -> dict:
    """Cortex ``forward`` envelope: model kwargs only, no loss and no context."""
    tensors = _align(dict(batch.get("batch") or batch), order, valid)
    kwargs: dict[str, Any] = {
        "input_ids": tensors["input_ids"],
        "attention_mask": tensors["attention_mask"],
    }
    for key in ("position_ids", "labels"):
        if key in tensors:
            kwargs[key] = tensors[key]
    return {"args": (), "kwargs": kwargs, "processing": {"post": _POST_PROCESSORS}}


def _require_logprobs(response: dict, op: str) -> torch.Tensor:
    """The per-token log-probs, or a loud failure naming what is missing.

    The router's fallback for an absent ``batch.logprobs`` is an empty
    ``loss_fn_outputs`` entry, which surfaces in the cookbook as a bare
    ``KeyError`` several frames away from the cause. Worse, those log-probs feed
    ``compute_kl_sample_train`` -- the sampler-versus-trainer divergence check,
    which is the alarm most likely to catch a new backend's first real bug.
    Substituting zeros would silence exactly that alarm.
    """
    body = response.get("batch") if isinstance(response, dict) else None
    logprobs = body.get("logprobs") if isinstance(body, dict) else None
    if logprobs is None:
        raise RuntimeError(
            f"cortex {op} returned no per-token log-probs. Requested post-processors: "
            f"{_POST_PROCESSORS}; response keys: {sorted(response) if isinstance(response, dict) else type(response)}"
            f"{', batch keys: ' + str(sorted(body)) if isinstance(body, dict) else ''}. "
            "Tinker's forward_backward contract requires them, so this cannot be "
            "defaulted -- they feed the sampler-vs-trainer KL check."
        )
    return logprobs


class CortexTinkerBackend:
    """The five Tinker verbs, lowered onto a Cortex-backed unified client."""

    def __init__(self, client: AsyncArcticRLClient) -> None:
        self.client = client

    async def fwd_bwd(self, batch: dict) -> dict:
        import torch

        body = dict(batch.get("batch") or {})
        meta = dict(batch.get("meta") or {})
        attention_mask = body.get("attention_mask")
        if not torch.is_tensor(attention_mask):
            body = {k: (torch.as_tensor(v) if not torch.is_tensor(v) else v) for k, v in body.items()}
            attention_mask = body.get("attention_mask")
        if attention_mask is None:
            raise ValueError("tinker fwd_bwd batch is missing 'attention_mask'")

        order, valid = _align_plan(attention_mask)
        payload = to_cortex_fwd_bwd_payload(
            {"batch": _align(body, order, valid), "meta": meta},
            processing=batch.get("processing"),
        )
        response = await self.client.fwd_bwd(payload)
        logprobs = _unalign_rows(_require_logprobs(response, "forward-backward"), order)
        return {"batch": {"logprobs": logprobs}, "metrics": response.get("metrics") or {}}

    async def fwd_no_grad(self, batch: dict) -> dict:
        import torch

        body = dict(batch.get("batch") or batch)
        attention_mask = body.get("attention_mask")
        if not torch.is_tensor(attention_mask):
            body = {k: (torch.as_tensor(v) if not torch.is_tensor(v) else v) for k, v in body.items()}
            attention_mask = body.get("attention_mask")
        order, valid = _align_plan(attention_mask)
        response = await self.client.fwd_no_grad(_forward_payload({"batch": body}, order, valid))
        logprobs = _unalign_rows(_require_logprobs(response, "forward"), order)
        return {"batch": {"logprobs": logprobs}, "metrics": response.get("metrics") or {}}

    async def step(self, overrides: dict | None) -> dict:
        # Tinker's AdamParams arrive as optimizer overrides; Cortex's `step`
        # takes a learning rate and nothing else, so the rest is dropped rather
        # than sent to a field that would ignore it silently.
        learning_rate = (overrides or {}).get("lr") or (overrides or {}).get("learning_rate")
        return await self.client.step(learning_rate=learning_rate)

    async def sync_weights(self) -> Any:
        return await self.client.sync_weights()

    async def generate(self, prompt_tokens: list[int], sampling_params: dict) -> dict:
        results = await self.client.generate([prompt_tokens], sampling_params=sampling_params)
        return {"results": results}


def build_handlers(client: AsyncArcticRLClient) -> dict[str, Callable]:
    """Handler kwargs for ``router.init_tinker_state``."""
    backend = CortexTinkerBackend(client)
    return {
        "fwd_bwd_handler": backend.fwd_bwd,
        "fwd_no_grad_handler": backend.fwd_no_grad,
        "step_handler": backend.step,
        "sync_weights_handler": backend.sync_weights,
        "generate_handler": backend.generate,
    }
