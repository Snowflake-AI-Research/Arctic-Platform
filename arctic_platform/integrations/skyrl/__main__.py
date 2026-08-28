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
"""SkyRL launcher that installs the Cortex driver shims first.

python -m arctic_platform.integrations.skyrl <hydra args>
"""

from __future__ import annotations

import runpy
import sys

from arctic_platform.integrations.skyrl import install_cortex_driver_shims

install_cortex_driver_shims()

# Forward argv to SkyRL's own entrypoint. This intentionally re-uses whatever
# SkyRL treats as its main module so we don't fork its argument surface.
if __name__ == "__main__":
    # Drop our own ``-m arctic_platform.integrations.skyrl`` from sys.argv so
    # SkyRL sees the same CLI it would from ``python -m skyrl.train.entrypoints.main_base``.
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    runpy.run_module("skyrl.train.entrypoints.main_base", run_name="__main__", alter_sys=True)
