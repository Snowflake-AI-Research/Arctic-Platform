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

"""Trainer-local debug configuration must not leak out of test environment scopes."""

import os

import pytest

from arctic_platform.debug import weight_sync


@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_direct_trainer_opt_in_restores_the_prior_environment(monkeypatch, previous):
    monkeypatch.delenv(weight_sync.VERIFY_ENV, raising=False)
    if previous is not None:
        monkeypatch.setenv(weight_sync.VERIFY_ENV, previous)
    with monkeypatch.context() as scope:
        scope.setenv(weight_sync.VERIFY_ENV, "0")
        config = {weight_sync.TRAINING_OPTION: True}
        assert weight_sync.maybe_configure_broadcast_training(config) == {}
        assert config == {weight_sync.TRAINING_OPTION: True}
        assert weight_sync.enabled()
    assert os.environ.get(weight_sync.VERIFY_ENV) == previous
