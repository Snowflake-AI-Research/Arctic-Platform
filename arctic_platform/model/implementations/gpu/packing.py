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

"""Model-stack names for the shared token-budget packer.

``IGNORE_INDEX`` and ``cu_seqlens_from_position_ids`` live in
``arctic_platform.common.packing``. This module keeps the import path the model
implementations already use.
"""

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.common.packing import cu_seqlens_from_position_ids

__all__ = [
    "IGNORE_INDEX",
    "cu_seqlens_from_position_ids",
]
