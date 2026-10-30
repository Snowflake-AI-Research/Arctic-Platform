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

"""Validate Arctic Platform job configs against a single-GPU reference.

See ``arctic_platform/correctness/README.md`` for onboarding and regression usage.
"""

from .harness.registry import TestOutcome
from .harness.registry import TestResult
from .harness.registry import correctness_test
from .harness.registry import registered_tests

__all__ = ["TestOutcome", "TestResult", "correctness_test", "registered_tests"]
