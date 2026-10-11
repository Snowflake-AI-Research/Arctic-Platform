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
"""OpenHands code-localization harness on Cortex.

The agent loop, reward, and chat proxy live here. SkyRL drives GRPO.
Cortex runs the training and sampling sub-jobs. Importing this package does
not import OpenHands or SkyRL.

Adapted from https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2
The Qwen3.5 XML tool-call shape is from that repo at 8184b42.
"""

from __future__ import annotations

from arctic_platform.integrations.openhands.reward import localization_reward
from arctic_platform.integrations.openhands.tool_parser import parse_tool_calls

__all__ = ["localization_reward", "parse_tool_calls"]
