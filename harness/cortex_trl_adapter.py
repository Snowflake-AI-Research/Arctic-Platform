"""Make the TRL integration's client calls legible to a Cortex zone.

The TRL integration was written against the on-prem server and speaks its dialect:
batches go out as ``{"batch", "meta", "processing"}`` and every call asks for the
post-processors ``apply_temperature`` and ``compute_entropy_and_logprobs``. A Cortex
zone understands neither. It wants ``{"kwargs", "processing"}``, and the only
post-processor it registers is ``compute_logprobs`` -- grep ``dss/`` for the other
two and you get nothing.

That is why the recipe fails over Cortex before it ever reaches the loss: the zone's
``dp_shard_batch`` rejects the outer envelope with "dispatch requires 2D input_ids".

This adapter sits between ``ArcticTrainingClient`` / ``ArcticRolloutWorker`` and the
real client and translates. It is deliberately a shim and not a fix: the honest fix
is either teaching the Cortex zone those post-processors or teaching the integration
to speak both dialects. It exists so the loss encoding can be exercised end to end
today.

Two things it cannot paper over, both surfaced rather than hidden:

* ``apply_temperature`` is only a no-op at ``temperature == 1.0``. Anything else is
  refused rather than silently mis-scaled.
* ``compute_entropy_and_logprobs`` also returns entropy, and ``compute_logprobs``
  does not. Entropy comes back as zeros, so any entropy metric is meaningless and
  an entropy bonus would silently do nothing. GRPO as configured here uses neither.

Whether the two post-processors agree on the *frame* of the returned log-probs is
not something this file can assert. The check is the reported importance ratio: on
policy it must sit at ~1, and an off-by-one frame blows it up.
"""

from __future__ import annotations

from typing import Any

import torch


def _labels_from(batch: dict) -> torch.Tensor:
    """Next-token targets with -100 on padding and on each row's final position.

    The zone's ``compute_logprobs`` scores position ``t`` against ``labels[t]``. The
    last real token of a row has no successor, so it is not a target -- which is also
    why ``loss_mask`` has to be derived from the labels rather than the attention
    mask, one position tighter.
    """
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    valid = mask.bool()
    nxt = torch.zeros_like(valid)
    nxt[:, :-1] = valid[:, 1:]
    labels[~(valid & nxt)] = -100
    return labels


class CortexTRLAdapter:
    """Translate on-prem-shaped TRL calls into Cortex-shaped ones."""

    def __init__(self, client: Any, *, temperature: float = 1.0):
        if abs(temperature - 1.0) > 1e-9:
            raise ValueError(
                f"temperature={temperature} needs apply_temperature, which no Cortex zone registers. "
                "Only 1.0 is safe here, where it is a no-op."
            )
        self._client = client
        self.temperature = temperature

    def __getattr__(self, name: str) -> Any:
        # generate, step, sync_weights, jobs, shutdown, ... all pass straight through.
        return getattr(self._client, name)

    def _to_cortex(self, payload: dict) -> tuple[dict, dict]:
        batch = dict(payload["batch"])
        processing = dict(payload.get("processing") or {})
        batch.setdefault("labels", _labels_from(batch))

        post = list(processing.get("post") or [])
        unknown = [p for p in post if p not in ("apply_temperature", "compute_entropy_and_logprobs")]
        if unknown:
            raise ValueError(f"unexpected post-processors for the Cortex path: {unknown}")
        cortex_processing: dict[str, Any] = {"post": ["compute_logprobs"], "loss_fn": processing.get("loss_fn")}
        if processing.get("config"):
            cortex_processing["config"] = processing["config"]
        return batch, cortex_processing

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
                # No zone-side entropy exists; zeros keep the caller's shape contract.
                "entropy": torch.as_tensor(entropy) if entropy is not None else torch.zeros_like(logprobs),
            }
        }

    def fwd_bwd(self, payload: dict) -> dict:
        batch, processing = self._to_cortex(payload)
        if "loss_mask" in batch:
            # _surrogate_payload derives loss_mask from attention_mask, which is one
            # position too generous once labels exist; grpo's preflight rejects that.
            batch["loss_mask"] = (batch["labels"] != -100).to(batch["loss_mask"].dtype)
        return self._client.fwd_bwd({"kwargs": batch, "processing": processing})
