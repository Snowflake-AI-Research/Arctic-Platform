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

"""CPU tests for ``ArcticWeightTransfer`` (IPC auto-select, unused param iterator)."""

from __future__ import annotations

import types

import pytest
import torch

# Importing weights runs the trl integration package __init__, which imports trl; skip on a minimal image.
pytest.importorskip("trl.experimental.api")

from arctic_platform.integrations.trl.weights import ArcticWeightTransfer  # noqa: E402


def _client(colocate: bool | None):
    """Fake SyncArcticRLClient: exposes config.backend.colocate and records sync_weights kwargs."""
    backend = types.SimpleNamespace() if colocate is None else types.SimpleNamespace(colocate=colocate)
    client = types.SimpleNamespace(config=types.SimpleNamespace(backend=backend))
    client.sync_calls = []
    client.sync_weights = lambda **kw: client.sync_calls.append(kw)
    return client


class TestCudaIpcSelection:
    def test_colocate_true_enables_cuda_ipc(self):
        assert ArcticWeightTransfer(_client(True)).cuda_ipc is True

    def test_colocate_false_disables_cuda_ipc(self):
        assert ArcticWeightTransfer(_client(False)).cuda_ipc is False

    def test_missing_colocate_attr_defaults_off(self):
        # getattr(..., "colocate", False) -> NCCL path when the backend doesn't advertise colocation.
        assert ArcticWeightTransfer(_client(None)).cuda_ipc is False

    @pytest.mark.parametrize("override", [True, False])
    def test_explicit_override_wins_over_colocate(self, override):
        # colocate would imply the opposite; the explicit cuda_ipc argument must take precedence.
        wt = ArcticWeightTransfer(_client(not override), cuda_ipc=override)
        assert wt.cuda_ipc is override


class TestSendWeights:
    def test_delegates_to_sync_weights_with_flags(self):
        client = _client(True)
        wt = ArcticWeightTransfer(client, low_memory=True)
        wt.send_weights(iter([]))
        assert client.sync_calls == [{"cuda_ipc": True, "low_memory": True}]

    def test_does_not_consume_iterator(self):
        client = _client(False)
        consumed = {"hit": False}

        def _params():
            consumed["hit"] = True  # only flips if the generator body runs (i.e. it was iterated)
            yield ("w", torch.zeros(1))

        ArcticWeightTransfer(client).send_weights(_params())
        assert consumed["hit"] is False  # trainer params are NOT the source of truth -> never iterated
        assert client.sync_calls == [{"cuda_ipc": False, "low_memory": False}]

    def test_lifecycle_hooks_are_noops(self):
        # pause/resume/init/destroy exist for the protocol but the whole transaction is sync_weights.
        wt = ArcticWeightTransfer(_client(False))
        wt.init_weight_transfer()
        wt.pause()
        wt.resume()
        wt.destroy()
        assert wt.client.sync_calls == []  # none of the lifecycle no-ops touch the server
