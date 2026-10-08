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

"""The in-process client registers itself as ``tinker`` before a recipe imports it.

Importing ``arctic_platform.tinker`` inside this process would replace
``sys.modules["tinker"]`` for the rest of the suite, so these checks run in a
child interpreter.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("tinker")


def test_import_registers_the_tinker_module() -> None:
    script = """
import arctic_platform.tinker
import tinker
from tinker.types import LossFnType
assert tinker.ServiceClient.__module__ == "arctic_platform.tinker", tinker.ServiceClient
assert "importance_sampling" in LossFnType.__args__
tinker.configure(training_gpus=1, sampling_gpus=1)
try:
    tinker.configure(training_gpus=0, sampling_gpus=1)
except ValueError:
    pass
else:
    raise SystemExit("configure accepted zero training GPUs")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_launcher_requires_gpu_counts() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "arctic_platform.tinker.run", "tinker_cookbook.recipes.math_rl.train"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--training-gpus" in completed.stderr + completed.stdout
