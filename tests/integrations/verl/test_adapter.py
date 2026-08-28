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

import pytest

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


def _ds_wrapper(verl_stub, *, mini: int, micro: int, rollout_n: int):
    """A wrapper carrying only what ``_create_ds_config`` reads."""
    from omegaconf import OmegaConf

    adapter = _adapter(verl_stub)
    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)
    wrapper.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 16},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": mini,
                    "ppo_micro_batch_size_per_gpu": micro,
                    "fsdp_config": {},
                },
                "rollout": {"n": rollout_n},
            },
        }
    )
    wrapper._backend_config = OmegaConf.create({"train": {"deepspeed": {"bf16": {"enabled": True}}}})
    return wrapper


def test_ds_config_rejects_indivisible_multi_gpu_triple(verl_stub) -> None:
    """DeepSpeed needs train_batch_size == micro x grad_accum x n_gpus.

    The shipped GSM8K recipe (mini=16, n=5, micro=16) is fine at one GPU but
    indivisible at four: 80 / (16 x 4) truncates to 1, and 16 x 1 x 4 = 64 != 80.
    That used to surface only as an opaque server-side sub-job failure.
    """
    wrapper = _ds_wrapper(verl_stub, mini=16, micro=16, rollout_n=5)

    with pytest.raises(AssertionError, match="divisible by"):
        wrapper._create_ds_config(n_gpus=4)

    # Single GPU is the default path and must stay working.
    assert wrapper._create_ds_config(n_gpus=1)["gradient_accumulation_steps"] == 5


def test_ds_config_multi_gpu_triple_multiplies_out(verl_stub) -> None:
    """With a divisor micro batch, the emitted triple satisfies DeepSpeed."""
    wrapper = _ds_wrapper(verl_stub, mini=16, micro=4, rollout_n=5)

    ds = wrapper._create_ds_config(n_gpus=4)

    assert ds["gradient_accumulation_steps"] == 5
    assert ds["train_batch_size"] == ds["train_micro_batch_size_per_gpu"] * ds["gradient_accumulation_steps"] * 4
