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
import os


def diagnostics_log_path() -> str:
    """Where the diagnostics lines are appended, so concurrent runs on one node do not share a file.

    The collector reads this file to take the max over ranks. Two runs writing one file interleave into a
    single stream that cannot be split afterwards, which silently mixes configurations. Set
    DSS_MEM_LOG_PATH per run when placing more than one on a node.
    """
    return os.environ.get("DSS_MEM_LOG_PATH", "/tmp/log1")
