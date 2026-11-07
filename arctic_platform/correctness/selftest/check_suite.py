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

"""Single driver for the correctness selftest library."""

from __future__ import annotations

import importlib

import pytest

pytest.register_assert_rewrite(f"{__package__}.library")
library = importlib.import_module(".library", __package__)

for _name in library._SELFTEST_FIXTURES:
    globals()[_name] = getattr(library, _name)

for _group, _names in library._SELFTEST_GROUPS.items():
    _fixtures = library._SELFTEST_AUTO_FIXTURES[_group]
    for _name in _names:
        _check = getattr(library, _name)
        globals()[_name] = pytest.mark.usefixtures(*_fixtures)(_check) if _fixtures else _check

for _name, _check in vars(library).items():
    if _name.startswith("test_") and callable(_check) and _name not in globals():
        globals()[_name] = _check

del _name, _group, _names, _fixtures, _check
