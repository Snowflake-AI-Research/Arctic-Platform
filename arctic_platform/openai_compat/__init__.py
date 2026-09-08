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
"""An OpenAI-compatible ``/v1`` surface over an Arctic sampling job.

Point any OpenAI client at it and change nothing but ``base_url``.
Non-streaming: ``stream=true`` is refused rather than faked. See README.md for
the parameter-by-parameter compatibility table.
"""

from arctic_platform.openai_compat.server import build_app

__all__ = ["build_app"]
