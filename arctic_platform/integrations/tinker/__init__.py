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
"""Implementation behind :mod:`arctic_platform.tinker`.

Nothing is imported here. Open the module that does the work:

- :mod:`arctic_platform.integrations.tinker.convert` — datums, losses, sampling params
- :mod:`arctic_platform.integrations.tinker.job` — GPU count, LoRA, optimizer, sequence isolation
- :mod:`arctic_platform.integrations.tinker.cortex` — forward-backward, optimizer step, sampling
"""
