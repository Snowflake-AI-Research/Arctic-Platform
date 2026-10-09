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

"""Model profiles, one per registered loader and named after it. Importing this package registers them.

A cell is ``supports`` only when the loader's own options model (``Patches`` and ``ModelSpec`` for the
``huggingface`` loader) has a field for the option at this commit. ``tests/model/test_option_profiles.py``
checks that against the loaders.
"""

from arctic_platform.common.profiles import generic_moe  # noqa: F401
from arctic_platform.common.profiles import glm5_next  # noqa: F401
from arctic_platform.common.profiles import glm_moe_dsa  # noqa: F401
from arctic_platform.common.profiles import huggingface  # noqa: F401
from arctic_platform.common.profiles import qwen3_5_moe  # noqa: F401
from arctic_platform.common.profiles import qwen4_exp  # noqa: F401
