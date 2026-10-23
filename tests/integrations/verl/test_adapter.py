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

"""API-surface tests for ``ArcticRLClientWrapper``.

Full integration is covered by the Golden Run E2E smoke; here we pin
the public shape (inheritance, ``from_config`` signature,
``destroy`` idempotency, ``requires_single_forwarder``) without
instantiating the heavy Arctic RL / Ray runtime.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

# adapter.py imports omegaconf + transformers at module top level.
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")


def _adapter(verl_stub):
    """Import the adapter against the shared verl + arctic_platform.rl stubs."""
    return importlib.import_module("arctic_platform.integrations.verl.adapter")


def test_adapter_inherits_from_verl_remote_backend(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    from verl.remote_backend.base import RemoteBackend

    assert issubclass(
        adapter.ArcticRLClientWrapper, RemoteBackend
    ), "ArcticRLClientWrapper must subclass verl's RemoteBackend so verl's trainer can dispatch through the ABC."


def test_from_config_signature(verl_stub) -> None:
    """``from_config`` is the sole public constructor -- its signature is API surface."""
    adapter = _adapter(verl_stub)

    sig = inspect.signature(adapter.ArcticRLClientWrapper.from_config)
    params = sig.parameters

    assert "main_config" in params
    assert "handle" in params
    assert params["handle"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["handle"].default is None


def test_requires_single_forwarder_returns_true(verl_stub) -> None:
    """Arctic owns its own training/sampling parallelism; the verl worker group
    must be a single forwarder. This is the invariant that
    ``RemoteBackendTrainer._enforce_single_forwarder_if_required`` reads.
    """
    adapter = _adapter(verl_stub)

    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)
    assert wrapper.requires_single_forwarder() is True


def test_destroy_is_idempotent(verl_stub) -> None:
    """Second ``destroy()`` call must not raise or re-shutdown the client.

    verl's trainer may call ``destroy()`` on multiple exception paths;
    a second call has to be safe or the trainer leaks async errors on exit.
    """
    adapter = _adapter(verl_stub)

    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)

    shutdown_calls = 0

    class _FakeClient:
        async def shutdown(self):
            nonlocal shutdown_calls
            shutdown_calls += 1

    wrapper._client = _FakeClient()

    async def _run():
        await wrapper.destroy()
        await wrapper.destroy()

    asyncio.run(_run())
    assert shutdown_calls == 1, "destroy() must be idempotent; second call should be a no-op."
    assert wrapper._client is None


def _balance_wrapper(adapter, *, balance_batch: bool, zorro: bool):
    """A wrapper with the fields the balance stamp reads, and no Ray client."""
    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)
    wrapper.balance_batch = balance_batch
    wrapper.zorro_train_enable = zorro
    wrapper.zorro_train_max_rollouts = 2
    wrapper.drop_position_ids = True
    wrapper._max_token_len_per_gpu = 128
    wrapper.logits_optimization = "none"
    wrapper.logits_optimization_peak_mem_size_in_gib = 4
    wrapper.logits_compute_in_fp32 = False
    wrapper.logits_compute_from_fp32_inputs = False
    actor = SimpleNamespace(ppo_mini_batch_size=2, ppo_epochs=1)
    wrapper.config = SimpleNamespace(
        data=SimpleNamespace(train_batch_size=2),
        actor_rollout_ref=SimpleNamespace(actor=actor),
    )
    return wrapper


def _two_row_batch(*, same_prompt: bool, dense_prompts: bool) -> dict:
    prompt_a = [7, 7]
    prompt_b = [7, 7] if same_prompt else [8, 8]
    response_a = [1, 2]
    response_b = [3, 4]
    if dense_prompts:
        prompts = torch.tensor([prompt_a, prompt_b], dtype=torch.long)
        responses = torch.tensor([response_a, response_b], dtype=torch.long)
    else:
        prompts = torch.nested.nested_tensor(
            [torch.tensor(prompt_a, dtype=torch.long), torch.tensor(prompt_b, dtype=torch.long)],
            layout=torch.jagged,
        )
        responses = torch.nested.nested_tensor(
            [torch.tensor(response_a, dtype=torch.long), torch.tensor(response_b, dtype=torch.long)],
            layout=torch.jagged,
        )
    return {
        "input_ids": torch.nested.nested_tensor(
            [
                torch.tensor(prompt_a + response_a, dtype=torch.long),
                torch.tensor(prompt_b + response_b, dtype=torch.long),
            ],
            layout=torch.jagged,
        ),
        "prompts": prompts,
        "responses": responses,
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "response_mask": torch.ones((2, 2), dtype=torch.long),
        "old_log_probs": torch.zeros((2, 2)),
        "advantages": torch.zeros((2, 2)),
        "temperature": 1.0,
    }


class _RecordingClient:
    def __init__(self):
        self.payloads = []

    async def fwd_no_grad(self, payload, reference_model=False):
        self.payloads.append(payload)
        return {"batch": {"logprobs": torch.zeros((2, 4))}, "metrics": {}}

    async def fwd_bwd(self, payload):
        self.payloads.append(payload)
        return {"metrics": {"loss": 1.0}}

    async def step(self):
        return {"metrics": {}}


def test_logprob_stamps_balance_off_like_skyrl(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=True, zorro=True)
    client = _RecordingClient()
    wrapper._client = client

    async def _run():
        await wrapper.compute_log_prob(
            _two_row_batch(same_prompt=True, dense_prompts=False),
            ref=False,
            calculate_entropy=False,
            rollout_n=2,
            temperature=1.0,
            pad_token_id=0,
        )

    asyncio.run(_run())
    assert client.payloads[0]["meta"]["balance_batch"] is False
    assert "prompt_group_ids" not in client.payloads[0]["meta"]
    sent_prompts = client.payloads[0]["batch"]["prompts"]
    assert sent_prompts.ndim == 2 and not sent_prompts.is_nested
    assert torch.equal(sent_prompts, torch.tensor([[7, 7], [7, 7]]))


def test_update_actor_stamps_balance_and_checks_prompt_blocks(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=True, zorro=True)
    client = _RecordingClient()
    wrapper._client = client
    actor_config = SimpleNamespace(calculate_entropy=False, use_kl_loss=False, policy_loss=None)

    async def _run(batch):
        await wrapper.update_actor(batch, actor_config=actor_config, pad_token_id=0, rollout_n=2, temperature=1.0)

    asyncio.run(_run(_two_row_batch(same_prompt=True, dense_prompts=True)))
    assert client.payloads[0]["meta"]["balance_batch"] is True
    assert client.payloads[0]["meta"]["prompt_group_ids"].tolist() == [0, 0]

    with pytest.raises(ValueError, match="share prompt columns"):
        asyncio.run(_run(_two_row_batch(same_prompt=False, dense_prompts=True)))


def test_update_actor_densifies_jagged_prompts(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=True, zorro=True)
    client = _RecordingClient()
    wrapper._client = client
    actor_config = SimpleNamespace(calculate_entropy=False, use_kl_loss=False, policy_loss=None)
    batch = _two_row_batch(same_prompt=True, dense_prompts=False)

    async def _run():
        await wrapper.update_actor(batch, actor_config=actor_config, pad_token_id=0, rollout_n=2, temperature=1.0)

    asyncio.run(_run())
    sent_prompts = client.payloads[0]["batch"]["prompts"]
    assert sent_prompts.ndim == 2 and not sent_prompts.is_nested
    assert torch.equal(sent_prompts, torch.tensor([[7, 7], [7, 7]]))
    assert client.payloads[0]["meta"]["prompt_group_ids"].tolist() == [0, 0]


def test_update_actor_balance_off_skips_the_prompt_block_check(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=False, zorro=True)
    client = _RecordingClient()
    wrapper._client = client
    actor_config = SimpleNamespace(calculate_entropy=False, use_kl_loss=False, policy_loss=None)

    async def _run():
        await wrapper.update_actor(
            _two_row_batch(same_prompt=False, dense_prompts=True),
            actor_config=actor_config,
            pad_token_id=0,
            rollout_n=2,
            temperature=1.0,
        )

    asyncio.run(_run())
    assert client.payloads[0]["meta"]["balance_batch"] is False
    assert "prompt_group_ids" not in client.payloads[0]["meta"]
