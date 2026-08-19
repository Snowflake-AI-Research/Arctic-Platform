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
        rollout_worker=ArcticRolloutWorker(client, dataset, reward_funcs, tokenizer),
        weight_transfer=ArcticWeightTransfer(client),
        optimizers=(ArcticOptimizer(client, model.parameters()), scheduler),
    )

The server registers ``weighted_logprob_sum`` in ``LOSS_FNS`` when it resolves the dotted-path loss name
``arctic_platform.integrations.trl.loss.weighted_logprob_sum`` (importing the ``loss`` submodule runs its
``@register_loss_fn`` decorator). That is the only server-side addition the integration needs.

``ArcticRolloutWorker``, ``ArcticWeightTransfer`` and ``weighted_logprob_sum`` are exposed lazily (PEP 562).
The first two pull in TRL's ``async_grpo`` stack (aiohttp, datasets, ...); ``weighted_logprob_sum`` lives in
``loss``, which imports the server-side ``pipeline`` (and transitively ``deepspeed``). The CPU-only TRL client
imports only ``client`` / ``rollout`` / ``weights`` and passes the loss by dotted-path name, so keeping the
loss import lazy stops ``deepspeed`` from being dragged into the client process. Accessing any of these names
imports the owning submodule on demand.

See ``README.md`` for the design rationale and the open items.
"""

from typing import TYPE_CHECKING

from arctic_platform.integrations.trl.client import ArcticOptimizer
from arctic_platform.integrations.trl.client import ArcticTrainingClient

if TYPE_CHECKING:
    from arctic_platform.integrations.trl.loss import weighted_logprob_sum
    from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker
    from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

__all__ = [
    "ArcticOptimizer",
    "ArcticTrainingClient",
    "ArcticRolloutWorker",
    "ArcticWeightTransfer",
    "weighted_logprob_sum",
]


def __getattr__(name: str):
    # Lazy imports (PEP 562). `rollout`/`weights` pull in the async_grpo stack; `loss` pulls in the server-side
    # `pipeline` (and transitively `deepspeed`). The CPU-only client touches none of these, so importing this
    # package stays free of both dependency stacks until a name is actually accessed.
    if name == "ArcticRolloutWorker":
        from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker

        return ArcticRolloutWorker
    if name == "ArcticWeightTransfer":
        from arctic_platform.integrations.trl.weights import ArcticWeightTransfer

        return ArcticWeightTransfer
    if name == "weighted_logprob_sum":
        from arctic_platform.integrations.trl.loss import weighted_logprob_sum

        return weighted_logprob_sum
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
