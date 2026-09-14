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

from __future__ import annotations

import sys
import unittest


class TestCommonLightImports(unittest.TestCase):
    def test_common_registry_imports_do_not_require_training_extras(self):
        import arctic_platform
        import arctic_platform._dependency_groups as dependency_groups

        def fail_gate(*extras):
            raise AssertionError(f"dependency gate called for {extras!r}")

        original_gate = dependency_groups.require_any_dep_group
        try:
            sys.modules.pop("arctic_platform.common.registry", None)
            sys.modules.pop("arctic_platform.common", None)
            if hasattr(arctic_platform, "common"):
                delattr(arctic_platform, "common")

            dependency_groups.require_any_dep_group = fail_gate

            import arctic_platform.common as common
            from arctic_platform.common.registry import LOSS_FNS
            from arctic_platform.common.registry import POST_PROCESSORS
            from arctic_platform.common.registry import register_loss_fn
            from arctic_platform.common.registry import register_post_processor

            self.assertIn("DeepSpeedWorker", common.__all__)
            self.assertIsInstance(LOSS_FNS, dict)
            self.assertIsInstance(POST_PROCESSORS, dict)
            self.assertTrue(callable(register_loss_fn))
            self.assertTrue(callable(register_post_processor))
        finally:
            dependency_groups.require_any_dep_group = original_gate
