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
"""TRL checkpoint hooks must hit the Arctic server, not return success without saving."""

from __future__ import annotations

import pytest

pytest.importorskip("trl.experimental.async_grpo.training_client")

from arctic_platform.integrations.trl.client import ArcticTrainingClient  # noqa: E402


class _Client:
    def __init__(self):
        self.calls = []

    def save_checkpoint(self, path=None, **kwargs):
        self.calls.append(("save", path, kwargs))
        return {"path": path}

    def load_checkpoint(self, path=None, **kwargs):
        self.calls.append(("load", path, kwargs))
        return {"path": path, "global_step": 3}


def test_save_and_load_delegate_to_the_server():
    client = _Client()
    training = ArcticTrainingClient(client)
    training.save("/tmp/ckpt")
    training.load("/tmp/ckpt")
    assert client.calls == [("save", "/tmp/ckpt", {}), ("load", "/tmp/ckpt", {})]


def test_save_failure_is_not_swallowed():
    class _Broken(_Client):
        def save_checkpoint(self, path=None, **kwargs):
            raise RuntimeError("disk full")

    training = ArcticTrainingClient(_Broken())
    with pytest.raises(RuntimeError, match="disk full"):
        training.save("/tmp/ckpt")
