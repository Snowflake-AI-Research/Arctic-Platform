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
"""Minimal stdlib-logging shim.

Replaces prime-rl's loguru-based logger with a plain ``logging.Logger`` so the
carved-out Qwen3.5 loading path has no ``loguru`` dependency. Only the
``.info``/``.debug``/``.warning``/``.error`` methods are used by this package.
"""

from __future__ import annotations

import logging

_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        logger = logging.getLogger("dss.qwen35")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)7s %(message)s", "%H:%M:%S"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        _LOGGER = logger
    return _LOGGER
