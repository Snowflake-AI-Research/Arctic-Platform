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
"""Public activation-checkpoint recompute detector.

DSS and other clients must import from here rather than
``arctic_platform.model.implementations`` (see DSS ``test_ap_model_ownership``).
The underlying installer is idempotent and stashes the canonical thread-local on
``torch.utils.checkpoint``, so every caller shares one detector.
"""

from __future__ import annotations

from arctic_platform.model.implementations.gpu.router_replay_recompute import (
    _in_recompute as in_recompute,
)
from arctic_platform.model.implementations.gpu.router_replay_recompute import (
    _install_recompute_detector as install_recompute_detector,
)

__all__ = ["in_recompute", "install_recompute_detector"]
