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

"""Backward-compat shim: verl-shaped loss has moved.

The implementation now lives at
:mod:`arctic_platform.integrations.verl.grpo_loss` so all verl-integration
code sits in one subpackage (see Arctic-Platform#35). This module remains
so existing downstream callers importing::

    from arctic_platform.rl.processors.verl_grpo import verl_grpo_loss

continue to work unchanged. Importing this shim triggers registration of
``"verl_grpo"`` in ``arctic_platform.rl.processors.pipeline.LOSS_FNS``
exactly the same way the old top-level module did.
"""

# noqa: F401,F403 -- deliberate wildcard re-export to preserve public API.
from arctic_platform.integrations.verl.grpo_loss import *  # noqa: F401,F403
