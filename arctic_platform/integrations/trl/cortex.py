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

"""Make this integration's client calls legible to a Cortex zone.

The TRL integration was written against the on-prem server and speaks its
dialect: batches go out as ``{"batch", "meta", "processing"}`` and every call
asks for the post-processors ``apply_temperature`` and
``compute_entropy_and_logprobs``. A Cortex zone understands neither. It wants
``{"kwargs", "processing"}``, and the only post-processor it registers is
``compute_logprobs`` -- the other two names do not appear anywhere in
ArcticTraining-dss or dss-platform.

Without translation the recipe fails before it ever reaches the loss: the zone's
``dp_shard_batch`` rejects the outer envelope with ``dispatch requires 2D
input_ids to plan token-budget microbatches``.

:class:`CortexTRLAdapter` sits between :class:`ArcticTrainingClient` /
:class:`ArcticRolloutWorker` and the real client, and translates.

This is a translation layer, not the end state. The two candidates for a real
fix are registering those post-processors in ArcticTraining-dss, or teaching
this integration to emit both dialects directly. Either retires this file. It
exists because the alternative is that the Cortex path cannot run at all.

Two gaps it cannot close, both raised or documented rather than hidden:

* ``apply_temperature`` is a no-op only at ``temperature == 1.0``. Anything else
  is refused in the constructor rather than silently mis-scaled.
* ``compute_entropy_and_logprobs`` returns entropy and ``compute_logprobs`` does
  not, so entropy comes back as zeros. Any entropy metric is therefore
  meaningless and an entropy bonus would silently do nothing. GRPO as configured
  for this path uses neither, which is the only reason this is survivable.

Whether the two post-processors agree on the *frame* of the returned log-probs
is not something this file can assert -- it is a property of the two server
implementations. The check is the reported importance ratio: on policy it must
sit at ~1, and an off-by-one frame blows it up.
"""

from __future__ import annotations

from typing import Any

import torch

_ON_PREM_POST_PROCESSORS = ("apply_temperature", "compute_entropy_and_logprobs")


def next_token_labels(batch: dict) -> torch.Tensor:
    """Next-token targets, with ``-100`` on padding and on each row's last token.

    The zone's ``compute_logprobs`` scores position ``t`` against ``labels[t]``.
    A row's final real token has no successor, so it is not a target -- which is
    also why ``loss_mask`` has to be derived from the labels rather than the
    attention mask. It is one position tighter, and grpo's preflight rejects a
    mask that supervises a position whose label is ``-100``.
    """
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    valid = mask.bool()
    has_next = torch.zeros_like(valid)
    has_next[:, :-1] = valid[:, 1:]
    labels[~(valid & has_next)] = -100
    return labels


class CortexTRLAdapter:
    """Translate on-prem-shaped TRL client calls into Cortex-shaped ones.

    Wrap the client and hand the *wrapper* to both
    :class:`ArcticTrainingClient` and :class:`ArcticRolloutWorker`. Passing the
    unwrapped client to either one reintroduces the dialect mismatch for that
    call path only, which fails at the zone rather than locally.

    Args:
        client: an :class:`~arctic_platform.client.ArcticRLClient` on a Cortex
            backend.
        temperature: must be ``1.0``; see the module docstring.
    """

    def __init__(self, client: Any, *, temperature: float = 1.0):
        if abs(temperature - 1.0) > 1e-9:
            raise ValueError(
                f"temperature={temperature} needs the apply_temperature post-processor, which no "
                "Cortex zone registers. Only 1.0 is safe here, where it is a no-op."
            )
        self._client = client
        self.temperature = temperature

    def __getattr__(self, name: str) -> Any:
        # generate, step, sync_weights, jobs, shutdown, ... all pass through.
        # Only the two calls that carry a batch envelope need translating.
        return getattr(self._client, name)

    def _to_cortex(self, payload: dict) -> tuple[dict, dict]:
        batch = dict(payload["batch"])
        processing = dict(payload.get("processing") or {})
        batch.setdefault("labels", next_token_labels(batch))

        unknown = [p for p in (processing.get("post") or []) if p not in _ON_PREM_POST_PROCESSORS]
        if unknown:
            # Refuse rather than drop: a post-processor we do not recognise may
            # be load-bearing, and silently discarding it would corrupt the run
            # in a way that looks like a modelling problem.
            raise ValueError(
                f"cannot translate post-processors {unknown} for the Cortex path; "
                f"only {list(_ON_PREM_POST_PROCESSORS)} are known to be expressible here"
            )

        cortex: dict[str, Any] = {"post": ["compute_logprobs"], "loss_fn": processing.get("loss_fn")}
        if processing.get("config"):
            cortex["config"] = processing["config"]
        return batch, cortex

    def fwd_no_grad(self, payload: dict) -> dict:
        batch, processing = self._to_cortex(payload)
        result = self._client.fwd_no_grad({"kwargs": batch, "processing": processing})
        if "logprobs" not in result:
            raise RuntimeError(f"cortex forward returned no logprobs (keys={sorted(result)})")
        logprobs = torch.as_tensor(result["logprobs"])
        entropy = result.get("entropy")
        return {
            "batch": {
                "logprobs": logprobs,
                # The zone computes no entropy. Zeros keep the caller's shape
                # contract; see the module docstring for why that is survivable
                # here and what it silently disables.
                "entropy": torch.as_tensor(entropy) if entropy is not None else torch.zeros_like(logprobs),
            }
        }

    def fwd_bwd(self, payload: dict) -> dict:
        batch, processing = self._to_cortex(payload)
        if "loss_mask" in batch:
            # The caller derives loss_mask from attention_mask, which is one
            # position too generous once labels exist. grpo's preflight requires
            # loss_mask == 0 wherever the label is -100.
            batch["loss_mask"] = (batch["labels"] != -100).to(batch["loss_mask"].dtype)
        return self._client.fwd_bwd({"kwargs": batch, "processing": processing})
