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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reshape verl/SkyRL ``{batch, meta, processing}`` into Cortex's
``{args, kwargs, context, processing}`` wire format.

Copied from ``arctic_platform/integrations/_cortex_shared.py`` (PR 55 / SkyRL
Cortex path) into this workspace so TRL can remap without depending on PR 84 AP.

The processing block matches ``arctic_platform/cortex-client/recipes/rl_loop.py``
(Jae's cookbook). ``dp_size`` is deliberately NOT sent: the server treats it
as a loss divisor, which scales the effective LR down by that factor at
multi-GPU DP.
"""

from __future__ import annotations

import os
from typing import Any

from .cortex_forward import _context_meta

_FWD_BWD_DUMP_LEFT = int(os.environ.get("CORTEX_DUMP_FWD_BWD_KEYS", "0"))
_FWD_BWD_COMPARE = os.environ.get("CORTEX_VALIDATE_FWD_BWD_LOGPROBS", "0") not in (
    "0",
    "false",
    "False",
)

# Match Jae's rl_loop.py and verl's actor.yaml defaults; caller-supplied
# processing.config values win.
_DEFAULT_PROC_CONFIG: dict[str, Any] = {
    "eps_clip": 0.2,
    "loss_agg_mode": "token-mean",
    "entropy_coeff": 0.0,
}


def to_cortex_fwd_bwd_payload(batch: dict, *, processing: dict | None = None) -> dict:
    """Reshape a verl/SkyRL fwd_bwd payload into Cortex's wire shape.

    ``old_log_probs_shifted`` is dropped: server-side GRPO defaults it to
    ``logprobs.detach()`` (π_old ≡ π_new), correct for single-epoch on-policy.

    Requires ``loss_mask`` or ``response_mask``; falling back to
    ``attention_mask`` would train on prompt tokens.
    """
    import torch

    payload = dict(batch)
    processing_in = processing or payload.pop("processing", None)
    payload.pop("router_replay", None)

    if "batch" in payload and isinstance(payload["batch"], dict):
        tensors, meta = dict(payload["batch"]), dict(payload.get("meta") or {})
    else:
        tensors = dict(payload)
        meta = dict(tensors.pop("context", None) or {})

    input_ids = tensors.get("input_ids")
    attention_mask = tensors.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("cortex fwd_bwd requires 'input_ids' and 'attention_mask'")

    loss_mask = tensors.pop("loss_mask", None)
    if loss_mask is None:
        loss_mask = tensors.pop("response_mask", None)
    if loss_mask is None:
        raise ValueError(
            "cortex fwd_bwd requires either 'loss_mask' or 'response_mask' in the "
            "batch. Falling back to 'attention_mask' would include prompt tokens "
            "in the loss and silently corrupt the gradient."
        )
    if torch.is_tensor(loss_mask):
        loss_mask = loss_mask.to(torch.bool)
    advantages = tensors.pop("advantages", None)
    if advantages is None:
        raise ValueError("cortex fwd_bwd requires 'advantages' [B, S]")
    tensors.pop("old_log_probs", None)

    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for k in ("position_ids", "labels"):
        if k in tensors:
            kwargs_out[k] = tensors[k]

    caller_config = dict((processing_in or {}).get("config") or {})
    proc_config: dict[str, Any] = {**_DEFAULT_PROC_CONFIG, **caller_config}
    for k in ("global_batch_size", "batch_num_tokens"):
        if k not in proc_config and k in meta:
            proc_config[k] = int(meta[k])

    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": {
            **_context_meta(meta),
            "input_ids": input_ids,
            "advantages": advantages,
            "loss_mask": loss_mask,
        },
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }


def _tensor_note(obj: object) -> str:
    try:
        import torch

        if torch.is_tensor(obj):
            return f"tensor{tuple(obj.shape)} {obj.dtype}"
    except Exception:
        pass
    return type(obj).__name__


def _key_summary(obj: object, *, depth: int = 0) -> object:
    if depth > 4:
        return "..."
    if isinstance(obj, dict):
        return {str(k): _key_summary(v, depth=depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        return [_key_summary(obj[0], depth=depth + 1), f"len={len(obj)}"]
    return _tensor_note(obj)


def _logprobs_from_result(result: dict) -> tuple[str | None, object, object]:
    """Locate logprobs. Mike: Cortex ``post_process_outputs.logprobs`` vs on-prem ``batch.logprobs``."""
    batch = result.get("batch")
    if isinstance(batch, dict):
        lp = batch.get("logprobs", batch.get("log_probs"))
        if lp is not None:
            return "batch", lp, batch.get("entropy", batch.get("entropies"))
    ppo = result.get("post_process_outputs")
    if isinstance(ppo, dict):
        lp = ppo.get("logprobs", ppo.get("log_probs"))
        if lp is not None:
            return "post_process_outputs", lp, ppo.get("entropy", ppo.get("entropies"))
        nested = ppo.get("batch")
        if isinstance(nested, dict):
            lp = nested.get("logprobs", nested.get("log_probs"))
            if lp is not None:
                return "post_process_outputs.batch", lp, nested.get("entropy", nested.get("entropies"))
    lp = result.get("logprobs", result.get("log_probs"))
    if lp is not None:
        return "top", lp, result.get("entropy", result.get("entropies"))
    return None, None, None


def _maybe_decode_wire(result: dict) -> dict:
    import base64

    if not isinstance(result, dict):
        return result
    b64 = result.get("payload_b64")
    if isinstance(b64, str) and result.get("wire_format") is None:
        from arctic_platform import wire

        decoded = wire.loads(base64.b64decode(b64))
        if isinstance(decoded, dict):
            return decoded
    return result


def dump_fwd_bwd_logprob_fields(result: dict) -> None:
    """Print where logprobs sit on a Cortex ``/fwd-bwd`` result (first N dumps)."""
    global _FWD_BWD_DUMP_LEFT
    if _FWD_BWD_DUMP_LEFT <= 0 or not isinstance(result, dict):
        return
    _FWD_BWD_DUMP_LEFT -= 1
    src, lp, ent = _logprobs_from_result(result)
    ppo = result.get("post_process_outputs")
    ppo_keys = sorted(ppo.keys()) if isinstance(ppo, dict) else None
    batch = result.get("batch")
    batch_keys = sorted(batch.keys()) if isinstance(batch, dict) else None
    print(
        "[e2e] /fwd-bwd logprob probe "
        f"top_keys={sorted(result.keys())} "
        f"post_process_outputs_keys={ppo_keys} "
        f"batch_keys={batch_keys} "
        f"logprobs_source={src} "
        f"logprobs={_tensor_note(lp) if lp is not None else None} "
        f"entropy={_tensor_note(ent) if ent is not None else None} "
        f"tree={_key_summary(result)}",
        flush=True,
    )


def normalize_fwd_bwd_result(result: dict) -> dict:
    """Map Cortex ``post_process_outputs.logprobs`` onto on-prem ``batch.logprobs``."""
    result = _maybe_decode_wire(result)
    if not isinstance(result, dict):
        return result
    dump_fwd_bwd_logprob_fields(result)
    src, lp, ent = _logprobs_from_result(result)
    if lp is None:
        return result
    batch = dict(result["batch"]) if isinstance(result.get("batch"), dict) else {}
    if "logprobs" not in batch:
        batch["logprobs"] = lp
        if ent is not None:
            batch["entropy"] = ent
        out = dict(result)
        out["batch"] = batch
        if src and src != "batch":
            print(f"[e2e] /fwd-bwd remapped logprobs from {src} -> batch.logprobs", flush=True)
        return out
    return result


def compare_fwd_bwd_to_forward(fwd_bwd: dict, forward_logprobs: object, response_mask: object) -> None:
    """Same-step ``/forward`` vs remapped ``/fwd-bwd`` logprobs on loss tokens (GSM8K probe)."""
    if not _FWD_BWD_COMPARE:
        return
    import torch

    src, lp, _ent = _logprobs_from_result(fwd_bwd if isinstance(fwd_bwd, dict) else {})
    if lp is None or not torch.is_tensor(lp) or not torch.is_tensor(forward_logprobs):
        print(
            f"[e2e] /fwd-bwd vs /forward: skip compare source={src} "
            f"fwd_bwd={_tensor_note(lp)} forward={_tensor_note(forward_logprobs)}",
            flush=True,
        )
        return
    a = forward_logprobs.detach().float().cpu()
    b = lp.detach().float().cpu()
    if a.shape != b.shape:
        print(
            f"[e2e] /fwd-bwd vs /forward: SHAPE MISMATCH forward={tuple(a.shape)} "
            f"fwd_bwd={tuple(b.shape)} source={src}",
            flush=True,
        )
        return
    mask = None
    if torch.is_tensor(response_mask) and tuple(response_mask.shape) == tuple(a.shape):
        mask = response_mask.detach().cpu().bool()
    delta = (a - b).abs()
    if mask is not None and mask.any():
        delta = delta[mask]
    mean = float(delta.mean()) if delta.numel() else float("nan")
    mx = float(delta.max()) if delta.numel() else float("nan")
    print(
        f"[e2e] /fwd-bwd vs /forward: source={src} n={delta.numel()} abs_diff_mean={mean:.6g} abs_diff_max={mx:.6g}",
        flush=True,
    )
