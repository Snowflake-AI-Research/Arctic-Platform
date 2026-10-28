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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import gc
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.model.implementations.moe.distributed import deepep


@pytest.fixture
def dispatched(monkeypatch):
    event_calls = []
    event = SimpleNamespace(current_stream_wait=lambda: event_calls.append("wait"))
    handle = object()
    buffer = SimpleNamespace(
        dispatch=lambda **kwargs: (kwargs["x"], kwargs["topk_idx"], kwargs["topk_weights"], [1], handle, event)
    )
    monkeypatch.setattr(deepep, "_buffer", buffer)
    monkeypatch.setattr(deepep, "_handle_cache", {})
    monkeypatch.setattr(deepep, "_pending_dispatch_events", {})
    monkeypatch.setattr(deepep, "_new_event_overlap", lambda: None)
    args = [torch.ones(1, 1)] * 7
    output = deepep._dispatch_op_impl(*args)
    return SimpleNamespace(output=output, event_calls=event_calls, handle=handle, args=args)


def test_abandoned_dispatch_releases_handle_and_waits_for_event(dispatched):
    key = dispatched.output[-1].item()
    assert deepep._handle_cache[key] is dispatched.handle
    assert key in deepep._pending_dispatch_events
    dispatched.output = None
    gc.collect()
    assert key not in deepep._handle_cache
    assert key not in deepep._pending_dispatch_events
    assert dispatched.event_calls == ["wait"]


def test_live_dispatch_remains_available_and_autograd_owns_its_handle(dispatched):
    key = dispatched.output[-1].item()
    ctx = SimpleNamespace()
    deepep._dispatch_setup_context(ctx, dispatched.args, dispatched.output)
    gc.collect()
    assert deepep._handle_cache[key] is ctx.saved_handle
    assert dispatched.event_calls == []
    dispatched.output = None
    gc.collect()
    assert key not in deepep._handle_cache
    assert ctx.saved_handle is dispatched.handle


def test_consumed_dispatch_is_not_waited_or_released_twice(dispatched):
    key = dispatched.output[-1].item()
    deepep._sync_dispatch(key)
    saved = deepep._handle_cache.pop(key)
    dispatched.output = None
    gc.collect()
    assert saved is dispatched.handle
    assert dispatched.event_calls == ["wait"]
    assert deepep._handle_cache == {}
    assert deepep._pending_dispatch_events == {}
