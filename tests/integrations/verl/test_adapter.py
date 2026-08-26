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


def _cortex_config(**overrides):
    """A verl-shaped config tree with GRPO-on-Cortex defaults."""
    from types import SimpleNamespace

    actor = SimpleNamespace(use_kl_loss=False, ppo_epochs=1, policy_loss_fn=None)
    rollout = SimpleNamespace(multi_turn=SimpleNamespace(enable=False))
    algorithm = SimpleNamespace(use_kl_in_reward=False, adv_estimator="grpo")
    for key, value in overrides.items():
        target = {"use_kl_in_reward": algorithm, "adv_estimator": algorithm}.get(key, actor)
        if key == "multi_turn_enable":
            rollout.multi_turn.enable = value
            continue
        setattr(target, key, value)
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(actor=actor, rollout=rollout), algorithm=algorithm
    )


class TestCortexPreflight:
    """Cortex has no `/forward`, so the transport zero-fills log-probs. That is
    only sound while nothing reads them -- otherwise a run trains on zeros and
    still looks healthy. These knobs must be refused before the client is built.
    """

    def _check(self, verl_stub, **overrides):
        adapter = _adapter(verl_stub)
        wrapper = object.__new__(adapter.ArcticRLClientWrapper)
        object.__setattr__(wrapper, "config", _cortex_config(**overrides))
        return wrapper._reject_cortex_incompatible_knobs()

    def test_default_grpo_recipe_is_accepted(self, verl_stub) -> None:
        assert self._check(verl_stub) is None

    @pytest.mark.parametrize(
        "override, expected",
        [
            ({"use_kl_loss": True}, "use_kl_loss"),
            ({"use_kl_in_reward": True}, "use_kl_in_reward"),
            ({"ppo_epochs": 2}, "ppo_epochs"),
            ({"adv_estimator": "gae"}, "adv_estimator"),
            ({"policy_loss_fn": "custom"}, "policy_loss_fn"),
            ({"multi_turn_enable": True}, "multi_turn"),
        ],
    )
    def test_unsupported_knobs_are_refused(self, verl_stub, override, expected) -> None:
        with pytest.raises(NotImplementedError, match=expected):
            self._check(verl_stub, **override)

    def test_error_names_every_offending_knob_at_once(self, verl_stub) -> None:
        """A user fixing these one launch at a time is the surprise we're avoiding."""
        with pytest.raises(NotImplementedError) as excinfo:
            self._check(verl_stub, use_kl_loss=True, ppo_epochs=3)
        assert "use_kl_loss" in str(excinfo.value)
        assert "ppo_epochs" in str(excinfo.value)
