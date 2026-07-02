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

"""Recipe-side ``arctic_rl`` package shim.

This package exists for one reason: ensure ``bird`` / ``bird_sql`` are
registered with ``skyrl_gym`` on both the SkyRL driver and inside each Ray
worker that ``skyrl_gym.make()``-s an env.

Registration itself lives upstream — importing ``integrations.arctic_rl.envs``
runs it. This shim gives us a recipe-local module whose *import* triggers that
side-effect, without the recipe launcher ever having to name anything in
``$SKYRL_HOME/integrations/arctic_rl/examples/``.

How the side-effect lands in workers:

1. The recipe launcher passes ``trainer.override_entrypoint=arctic_rl.entrypoint``.
2. SkyRL's ``main_base`` imports that module on the driver, which imports this
   package (here), which imports upstream ``integrations.arctic_rl.envs`` for
   its ``register()`` side-effects.
3. ``arctic_rl.entrypoint`` re-binds Arctic RL's ``@ray.remote`` worker task
   to our package, so the worker re-imports ``arctic_rl.entrypoint`` (and
   therefore this ``__init__.py``) when it deserializes the task — running the
   same registration in the worker's address space.

The actual Arctic RL × SkyRL machinery (config/trainer/generator) is *not*
vendored — it lives in the user's SkyRL checkout at
``$SKYRL_HOME/integrations/arctic_rl/``. The recipe launchers add
``$SKYRL_HOME`` and this directory to ``PYTHONPATH`` so both packages are
importable side-by-side.
"""

import integrations.arctic_rl.envs as _envs  # noqa: F401  register bird / bird_sql

# The BIRD parquet's per-row ``env_class`` column is ``bird_sql`` (from the
# preprocess_bird.py output), but upstream only registers the ``bird`` id.
# Alias ``bird_sql`` -> ``BirdEnv`` so ``skyrl_gym.make("bird_sql", ...)`` works
# without touching the parquet or the upstream registration.
from skyrl_gym.envs.registration import register as _register, registry as _registry

if "bird_sql" not in _registry:
    _register(id="bird_sql", entry_point="integrations.arctic_rl.envs.bird:BirdEnv")
