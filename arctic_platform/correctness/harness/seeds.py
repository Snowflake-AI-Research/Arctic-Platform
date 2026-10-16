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

"""The single seed governing every random draw in the harness.

Hardcoded rather than generated so a failing run reproduces exactly on re-run. Both the synthetic model
weights and the batches derive from it, and both are materialized to disk, so a repeat run loads identical
bytes instead of re-drawing.
"""

SEED = 1234
