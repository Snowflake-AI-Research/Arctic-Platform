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

"""Classifier for DeepSpeed BF16_Optimizer's zero-norm assert."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

from arctic_platform.testing_utils import TestCasePlus

_FAKE_BF16 = """
def step(*, nested: bool = False, norm: float = 0.0):
    if nested:

        def _unrelated():
            assert False, "unrelated optimizer failure"

        _unrelated()
    all_groups_norm = norm
    assert all_groups_norm > 0.
"""


def _load_fake_bf16(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bf16_optimizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBf16ZeroNormAssert(TestCasePlus):
    def setUp(self):
        super().setUp()
        from arctic_platform.common.utils.bf16_zero_norm import is_bf16_zero_norm_assert

        self._is_match = is_bf16_zero_norm_assert
        self._opt = SimpleNamespace(_global_grad_norm=0.0)
        self._bf16_path = Path(self.get_auto_remove_tmp_dir()) / "bf16_optimizer.py"
        self._bf16_path.write_text(_FAKE_BF16)
        self._bf16 = _load_fake_bf16(self._bf16_path)

    def _raise(self, **kwargs) -> BaseException:
        try:
            self._bf16.step(**kwargs)
        except AssertionError as err:
            return err
        self.fail("expected AssertionError")

    def test_matches_bare_zero_norm_assert(self):
        err = self._raise(norm=0.0)
        self.assertTrue(self._is_match(err, self._opt))

    def test_rejects_nested_assert_under_step(self):
        err = self._raise(nested=True, norm=0.0)
        self.assertFalse(self._is_match(err, self._opt))

    def test_rejects_nonzero_grad_norm(self):
        err = self._raise(norm=0.0)
        self.assertFalse(self._is_match(err, SimpleNamespace(_global_grad_norm=1.0)))

    def test_rejects_unrelated_assert_outside_bf16_step(self):
        try:
            assert False
        except AssertionError as err:
            self.assertFalse(self._is_match(err, self._opt))
