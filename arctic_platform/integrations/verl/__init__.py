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

"""verl integration: Arctic RL as a verl ``trainer.remote_backend``.

Loaded on the verl side via::

    export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register

which registers the ``"arctic"`` backend + rollout replica against verl's
``RemoteBackendRegistry`` / ``RolloutReplicaRegistry`` without importing
heavy dependencies (adapter, DeepSpeed, vLLM) until the trainer actually
resolves the backend at ``fit()`` time.

The submodules here mirror what previously lived inside verl core under
``verl/workers/remote_client/arctic_rl.py`` and
``verl/workers/rollout/remote_rollout/arctic_rollout/``; moving them here
lets Snowflake own both sides of the wire (Arctic runtime +
verl adapter) in a single repo.
"""

__all__: list[str] = []
