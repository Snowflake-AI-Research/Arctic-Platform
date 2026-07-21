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
"""V1 (verl-project/verl main) compatibility wrappers for the Arctic backend.

Kept in a separate subpackage so V0 (Snowflake-AI-Research/verl) users can
continue to import ``arctic_platform.integrations.verl`` without paying for
V1's transfer_queue / TransferQueue import overhead, and so V1 users can plug
in via the same plugin bootstrap without pulling in V0-only imports.
"""
