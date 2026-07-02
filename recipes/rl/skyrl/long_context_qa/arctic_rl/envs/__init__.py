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

"""Register the env classes added by this recipe directory.

Only ``long_context_qa`` lives here — there's no upstream home for it yet
(will be PR'd into SkyRL alongside its recipe).

``bird`` / ``bird_sql`` are *not* re-registered here: importing
``integrations.arctic_rl`` from the user's SkyRL clone already runs the
upstream registration as a side-effect (the shim's ``entrypoint.py`` imports
``integrations.arctic_rl.config``, which triggers it). Doing it again here
would raise ``RegistrationError: name already registered``.

Uses ``__name__`` for the entry_point so registration works regardless of the
path this package gets checked out at.
"""

from skyrl_gym.envs.registration import register

register(id="long_context_qa", entry_point=f"{__name__}.long_context_qa:LongContextQAEnv")
