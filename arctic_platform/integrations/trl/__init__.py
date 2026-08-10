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

"""Arctic RL backend for TRL.

Unlike the verl integration there is no plugin entry point and no registry to hook. TRL takes the backend as
a constructor argument::

    from arctic_platform.client.client import SyncArcticRLClient
    from arctic_platform.integrations.trl import ArcticOptimizer
    from arctic_platform.integrations.trl import ArcticTrainingClient

    client = SyncArcticRLClient(config)
    trainer = AsyncGRPOTrainer(
        model=model,
        ...,
        training_client=ArcticTrainingClient(client),
        optimizers=(ArcticOptimizer(client, model.parameters()), scheduler),
    )

Importing this package registers ``weighted_logprob_sum`` in ``LOSS_FNS``, which is the only server-side
addition the integration needs.

See ``README.md`` for the design rationale and the open items.
"""

from arctic_platform.integrations.trl.client import ArcticOptimizer
from arctic_platform.integrations.trl.client import ArcticTrainingClient
from arctic_platform.integrations.trl.loss import weighted_logprob_sum

__all__ = ["ArcticOptimizer", "ArcticTrainingClient", "weighted_logprob_sum"]
