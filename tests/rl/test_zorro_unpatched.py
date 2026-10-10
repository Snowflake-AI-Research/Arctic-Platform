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

"""ZoRRo fwd_bwd is rejected unless init patched the model."""

import pytest

from arctic_platform.rl.processors.pipeline import _reject_zorro_on_unpatched_model


class _Model:
    pass


class _Engine:
    def __init__(self, model):
        self._model = model

    @property
    def module(self):
        return self._model


def test_zorro_request_requires_the_init_patch():
    model = _Model()
    engine = _Engine(model)
    _reject_zorro_on_unpatched_model(engine, {"zorro_train_enable": False})
    with pytest.raises(ValueError, match="not patched at init"):
        _reject_zorro_on_unpatched_model(engine, {"zorro_train_enable": True})
    model._arctic_zorro_once_patcher = object()
    _reject_zorro_on_unpatched_model(engine, {"zorro_train_enable": True})
