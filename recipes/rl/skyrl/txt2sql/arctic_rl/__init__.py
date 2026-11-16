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

Registration itself lives upstream in ``integrations.arctic_rl.envs``. This
shim does not import that package: its ``__init__`` also imports the trainer,
which initializes CUDA. Ray actors import this module from ``sitecustomize``
before Ray sets ``CUDA_VISIBLE_DEVICES``, so registration is by entry-point
string only.

How the side-effect lands in workers:

1. ``fsdp_bird_entry`` and ``sitecustomize`` import this package.
2. That registers ``bird`` and ``bird_sql`` without importing
   ``integrations.arctic_rl``.
3. ``skyrl_gym.make`` loads ``BirdEnv`` later, after Ray has assigned a GPU.

The actual Arctic RL × SkyRL machinery (config/trainer/generator) is *not*
vendored — it lives in the user's SkyRL checkout at
``$SKYRL_HOME/integrations/arctic_rl/``. The recipe launchers add
``$SKYRL_HOME`` and this directory to ``PYTHONPATH`` so both packages are
importable side-by-side.
"""

from skyrl_gym.envs.registration import register as _register
from skyrl_gym.envs.registration import registry as _registry

# Entry-point strings only. Importing ``integrations.arctic_rl`` initializes CUDA.
for _id in ("bird", "bird_sql"):
    if _id not in _registry:
        _register(id=_id, entry_point="integrations.arctic_rl.envs.bird:BirdEnv")
