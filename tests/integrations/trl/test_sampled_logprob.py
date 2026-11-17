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
"""A missing sampled-token logprob must not be replaced by some other token."""

from __future__ import annotations

import pytest

pytest.importorskip("trl.experimental.async_grpo.async_rollout_worker")

from arctic_platform.integrations.trl.rollout import _sampled_logprob  # noqa: E402


def test_int_key():
    assert _sampled_logprob({7: {"logprob": -0.25}}, 7) == -0.25


def test_str_key():
    assert _sampled_logprob({"7": {"logprob": -0.5}}, 7) == -0.5


def test_missing_sampled_token_raises():
    position = {3: {"logprob": -1.0}, "9": {"logprob": -2.0}}
    with pytest.raises(RuntimeError, match="sampled token 7"):
        _sampled_logprob(position, 7)
