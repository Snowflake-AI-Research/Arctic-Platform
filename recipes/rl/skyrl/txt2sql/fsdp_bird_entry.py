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

"""FSDP-native SkyRL entrypoint that registers ``bird`` / ``bird_sql``.

Stock ``main_base`` doesn't import the Arctic RL integration, so we register
the env in all three interpreters that touch it: the driver
(side-effect-import here), Ray workers (worker-side ``skyrl_entrypoint``
re-imports before running), and any ``multiprocessing.spawn`` child of the
reward scorer (picked up via ``sitecustomize.py`` on the recipe dir).

FSDP-native sibling of ``arctic_rl/entrypoint.py`` — Arctic goes through
``ArcticRLExp``, this one through ``BasePPOExp``.
"""

import os
import sys
from pathlib import Path

import ray

_HERE = Path(__file__).resolve()
_RECIPE_DIR = _HERE.parent
_RECIPE_DIR_STR = str(_RECIPE_DIR)

if _RECIPE_DIR_STR not in sys.path:
    sys.path.insert(0, _RECIPE_DIR_STR)

import arctic_rl  # noqa: E402,F401  register bird / bird_sql on the driver
import skyrl.train.entrypoints.main_base as _mb  # noqa: E402
import skyrl.train.utils.utils as _skyrl_utils  # noqa: E402

# Forward the recipe dir onto Ray workers' PYTHONPATH so worker-side imports of
# ``arctic_rl`` resolve.
_original_prepare = _skyrl_utils.prepare_runtime_environment


def _patched_prepare(cfg):
    env_vars = _original_prepare(cfg)
    existing_pp = env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if _RECIPE_DIR_STR not in existing_pp.split(os.pathsep):
        env_vars["PYTHONPATH"] = _RECIPE_DIR_STR + (os.pathsep + existing_pp if existing_pp else "")
    if "SKYRL_USE_LIGER" in os.environ:
        env_vars["SKYRL_USE_LIGER"] = os.environ["SKYRL_USE_LIGER"]
    return env_vars


_skyrl_utils.prepare_runtime_environment = _patched_prepare


@ray.remote(num_cpus=1)
def _skyrl_entrypoint_with_bird(cfg):
    """Ray worker task that re-registers ``bird`` / ``bird_sql`` before running."""
    import sys as _sys

    if _RECIPE_DIR_STR not in _sys.path:
        _sys.path.insert(0, _RECIPE_DIR_STR)
    import arctic_rl  # noqa: F401,F811  register bird / bird_sql in the worker

    exp = _mb.BasePPOExp(cfg)
    exp.run()


_mb.skyrl_entrypoint = _skyrl_entrypoint_with_bird


if __name__ == "__main__":
    _mb.main()
