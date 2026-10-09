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

"""The option registry, the option entries and the profiles import without the training extras.

The pytest process has already imported torch, so the import runs in a fresh interpreter. There the dependency
gate is replaced by a failing stub, and an import hook refuses torch, transformers, vLLM and DeepSpeed.
"""

from __future__ import annotations

import sys

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async

_SCRIPT = """
import importlib.abc
import sys

HEAVY = ("torch", "transformers", "vllm", "deepspeed")


class RefuseHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            raise ImportError(f"{name} must not be imported")
        return None


sys.meta_path.insert(0, RefuseHeavy())

import arctic_platform._dependency_groups as dependency_groups


def fail_gate(*extras):
    raise AssertionError(f"dependency gate called for {extras!r}")


dependency_groups.require_any_dep_group = fail_gate

from arctic_platform.common import model_options  # noqa: F401
from arctic_platform.common import option_registry
from arctic_platform.common import profiles  # noqa: F401

assert len(option_registry.options()) == 20
assert len(option_registry.profiles()) == 6
assert option_registry.check_coverage() == []
loaded = sorted(name for name in sys.modules if name.split(".")[0] in HEAVY)
assert loaded == [], loaded
print("option registry light imports passed")
"""


class TestOptionRegistryLightImports(TestCasePlus):
    def test_registry_imports_do_not_require_training_extras(self):
        result = execute_subprocess_async([sys.executable, "-c", _SCRIPT], env=self.get_env(), echo=False)
        self.assertIn("option registry light imports passed", result.stdout)
