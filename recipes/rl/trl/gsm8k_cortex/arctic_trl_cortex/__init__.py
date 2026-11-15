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

"""Recipe-local Cortex adapter for the TRL GSM8K driver.

Lives next to ``run_qwen3_1.7b_gsm8k_grpo_cortex.py``, which puts this directory's
parent on ``sys.path``. It is not ``arctic_platform.integrations.trl``: that
package is the on-prem adapter. This one reshapes packed TRL rows for Cortex
and normalizes forward results into ``batch.logprobs``.
"""
