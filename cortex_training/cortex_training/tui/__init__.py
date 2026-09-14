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

"""Read-only terminal UI for Cortex Training logs and scheduling events.

The formatting helpers import cleanly without ``textual``; the app and entry
point use the package's standard TUI dependencies.
"""

from cortex_training.tui.format import format_event  # noqa: F401
from cortex_training.tui.format import format_log_entry
