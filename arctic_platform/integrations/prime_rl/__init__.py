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

"""prime-rl integration: train and serve a public prime-rl RL run on Snowflake Cortex.

Unlike the verl adapter, prime-rl exposes no in-process plugin hook, so this is a
*launcher* rather than a registered backend. ``rl.py`` reuses an unmodified, pinned
prime-rl install — its ``orchestrator`` and ``env-server`` console scripts run as
subprocesses exactly as under prime-rl's own ``rl`` entrypoint — while the Cortex
worker (``worker.py``) stands in for the trainer and the vLLM-shaped inference server,
driving the Cortex sampling/training sub-jobs through ``arctic_platform.client``.

prime-rl itself (and its ``verifiers`` runtime) is user-supplied and version-pinned in
the launcher, not a dependency of the ``[prime-rl]`` extra — the same contract the verl
adapter uses for verl. Run with::

    python -m arctic_platform.integrations.prime_rl.rl @ <recipe>.toml [overrides]

See ``README.md`` for the prime-rl / verifiers commit pins and the install recipe.
"""

from arctic_platform._dependency_groups import require_any_dep_group

# The launcher and worker import arctic_platform.client (the Cortex SDK, carried by
# [cortex]); name the extra up front so a missing dep points at the install, not a bare
# ModuleNotFoundError. prime-rl / renderers / verifiers are user-supplied (see README).
require_any_dep_group("prime-rl")

__all__: list[str] = []
