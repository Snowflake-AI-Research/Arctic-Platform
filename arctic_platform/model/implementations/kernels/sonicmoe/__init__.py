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

"""Vendored SonicMoE functional kernels (https://github.com/Dao-AILab/sonic-moe, Apache 2.0, see LICENSE).

Only the functional layer is vendored (``functional/``, ``enums.py``, ``utils.py``). Local change: the up- and
down-projection backwards skip the expert weight gradients when the weights are frozen.
"""

from .enums import ActivationType
from .functional import moe_general_routing_inputs

__all__ = ["ActivationType", "moe_general_routing_inputs"]
