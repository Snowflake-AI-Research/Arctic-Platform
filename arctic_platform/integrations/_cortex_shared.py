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
"""Reshape SkyRL's ``{batch, meta, processing}`` into Cortex's
``{args, kwargs, context, processing}`` wire format.

The processing block matches ``cortex-client/recipes/rl_loop.py``
(Jae's cookbook). ``dp_size`` is deliberately NOT sent: the server treats it
as a loss divisor, which scales the effective LR down by that factor at
multi-GPU DP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import torch

# Match Jae's rl_loop.py and verl's actor.yaml defaults; caller-supplied
# processing.config values win.
_DEFAULT_PROC_CONFIG: dict[str, Any] = {
    "eps_clip": 0.2,
    "loss_agg_mode": "token-mean",
    "entropy_coeff": 0.0,
}

# verl/SkyRL ``actor_config`` keys the server's ``grpo`` reads under the same name.
_ACTOR_CONFIG_SAME_NAME = ("loss_agg_mode", "entropy_coeff", "use_kl_loss", "kl_loss_coef", "kl_loss_type")


def _actor_loss_config(meta: dict) -> dict[str, Any]:
    """``meta['actor_config']`` lowered onto ``grpo`` config keys.

    verl's PPO clip bounds the ratio to ``[1 - clip_ratio_low, 1 + clip_ratio_high]``,
    each side falling back to ``clip_ratio``; ``grpo`` spells that range
    ``eps_clip`` / ``eps_clip_higher``. verl's dual-clip ``clip_ratio_c`` is
    ``grpo``'s ``c_clip``. Unset (``None``) values are left to the defaults.
    """
    actor = meta.get("actor_config")
    if not isinstance(actor, dict):
        return {}
    clip = actor.get("clip_ratio")
    low, high = actor.get("clip_ratio_low"), actor.get("clip_ratio_high")
    lowered = {
        "eps_clip": clip if low is None else low,
        "eps_clip_higher": clip if high is None else high,
        "c_clip": actor.get("clip_ratio_c"),
        **{key: actor.get(key) for key in _ACTOR_CONFIG_SAME_NAME},
    }
    return {key: value for key, value in lowered.items() if value is not None}


def _resolve_proc_config(meta: dict, processing_in: dict | None) -> dict[str, Any]:
    """Defaults, then the caller's ``actor_config``, then an explicit ``processing.config``."""
    caller_config = dict((processing_in or {}).get("config") or {})
    proc_config: dict[str, Any] = {**_DEFAULT_PROC_CONFIG, **_actor_loss_config(meta), **caller_config}
    for k in ("global_batch_size", "batch_num_tokens"):
        if k not in proc_config and k in meta:
            proc_config[k] = int(meta[k])
    return proc_config


def _resolve_packed(meta: dict, packed: bool | None) -> bool:
    """Whether the request reaches the packed executor.

    ``packed`` is the job's setting (:func:`arctic_platform.client.config.cortex_job_packs`).
    ``None`` keeps the per-call rule: packed unless the meta asks for ZoRRo.
    """
    zorro = bool(meta.get("zorro_train_enable"))
    if packed is None:
        return not zorro
    if packed and zorro:
        raise ValueError(
            "the request asks for ZoRRo (meta zorro_train_enable) but the Cortex training job packs rows; "
            "create the job with ds_worker_config['zorro_train_enable']"
        )
    return packed


def to_cortex_fwd_payload(batch: dict, *, packed: bool | None = None) -> dict:
    """On-prem SkyRL ``{batch, meta}`` → Cortex ``{args, kwargs, context, processing}``.

    The on-prem client measures ``max_prompt_len`` and sends it with
    ``zorro_train_enable``, ``max_response_len``, ``pad_token_id``, and the
    row tensors (including ``prompts``). Those scalars stay on context. ZoRRo
    rows, and every row bound for an unpacked job (``packed=False``), stay in
    the layout the client built: the unpacked forward re-pads its log-probs onto
    those columns, which the fwd_bwd call then sends back as ``old_log_probs``.
    Rows bound for the packed executor are left-aligned, because Cortex's
    packer rejects leading padding.
    """
    import torch

    payload = dict(batch)
    payload.pop("processing", None)
    payload.pop("router_replay", None)
    if "batch" in payload and isinstance(payload["batch"], dict):
        tensors = dict(payload["batch"])
        meta = dict(payload.get("meta") or {})
    else:
        tensors = dict(payload)
        meta = dict(tensors.pop("context", None) or {})

    input_ids = tensors.get("input_ids")
    attention_mask = tensors.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("cortex /forward requires input_ids and attention_mask")

    job_packs = _resolve_packed(meta, packed)
    if meta.get("zorro_train_enable"):
        missing = _missing_zorro_meta_keys(meta)
        if missing:
            raise ValueError("ZoRRo meta is missing " + ", ".join(missing))
        prompts = tensors.get("prompts")
        if not torch.is_tensor(prompts) or prompts.ndim != 2 or int(prompts.shape[1]) == 0:
            raise ValueError("ZoRRo forward requires batch['prompts'] with width max_prompt_len")
    elif not job_packs:
        if "pad_token_id" not in meta:
            raise ValueError("unpacked forward requires meta['pad_token_id']: the worker unpads the rows with it")
    else:
        aligned, _, attention_mask = _left_align_batch(tensors, attention_mask, {})
        tensors = {**aligned, "attention_mask": attention_mask}
        input_ids = tensors["input_ids"]

    kwargs = {key: value for key, value in tensors.items() if torch.is_tensor(value)}
    context = {key: value for key, value in meta.items() if key != "dp_size" and not torch.is_tensor(value)}
    context["input_ids"] = input_ids
    from arctic_platform.common.balance_metadata import apply_balance_metadata

    apply_balance_metadata(context, meta, input_ids, tensors.get("prompts"))
    return {
        "args": (),
        "kwargs": kwargs,
        "context": context,
        "processing": {"post": ["compute_logprobs"], "loss_fn": None},
    }


# Padding sentinel per tensor. `labels` uses HF's ignore_index so a padded
# column contributes no loss even if something downstream reads labels directly.
_PAD_VALUE: dict[str, Any] = {"labels": -100}


def _left_align_batch(tensors: dict, attention_mask, extra: dict) -> tuple[dict, dict, Any]:
    """Move every row's real tokens into its leading columns, padding at the tail.

    SkyRL left-pads each row to the batch's longest sequence; Cortex's packer
    rejects exactly that ("packing requires left-aligned rows: every row's real
    tokens must be its leading columns, with padding at the tail").

    A stable sort on validity gives each row's real columns in order, then its
    pads. Every full-width tensor is gathered through that one index so
    ``advantages`` and ``loss_mask`` stay on the tokens they scored -- moving
    ``input_ids`` alone would shift the loss by a different amount per row, with
    no error and no crash.
    """
    import torch

    mask = attention_mask.to(torch.bool)
    width = mask.shape[-1]
    lengths = mask.sum(dim=1)
    valid = torch.arange(width, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)

    # Already conformant (a batch whose rows all happen to be full width, or a
    # caller that aligned upstream): skip the gather rather than pay for it.
    if torch.equal(mask, valid):
        return tensors, extra, attention_mask

    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)

    def move(name: str, t):
        if not torch.is_tensor(t) or t.dim() != 2 or t.shape[-1] != width:
            return t
        pad = _PAD_VALUE.get(name, False if t.dtype == torch.bool else 0)
        return torch.where(valid, t.gather(1, order), torch.full_like(t, pad))

    return (
        {k: move(k, v) for k, v in tensors.items()},
        {k: move(k, v) for k, v in extra.items()},
        valid.to(attention_mask.dtype),
    )


def _to_predict_next_frame(name: str, t):
    """Move a response-aligned ``[B, S]`` tensor one column left.

    Entry ``i`` scored ``input_ids[i]``; afterwards it weights the prediction
    made at ``i``, which is of ``input_ids[i + 1]``. The last column predicts
    past the row (DSS wraps it to the row's first token), so it is cleared.
    """
    import torch

    if not torch.is_tensor(t) or t.dim() != 2:
        raise ValueError(f"cortex fwd_bwd needs {name!r} as a [B, S] tensor to shift it into the predict-next frame")
    out = torch.zeros_like(t)
    out[:, :-1] = t[:, 1:]
    return out


def _missing_zorro_meta_keys(meta: dict) -> list[str]:
    """Keys the worker indexes directly once ZoRRo is on.

    ``run_pipeline`` reads ``max_prompt_len`` when ``zorro_train_enable`` is set.
    """
    missing: list[str] = []
    if meta.get("zorro_train_enable"):
        if "max_prompt_len" not in meta:
            missing.append("max_prompt_len")
    return missing


def to_cortex_fwd_bwd_payload(
    batch: dict,
    *,
    processing: dict | None = None,
    old_log_probs_shifted: torch.Tensor | None = None,
    response_aligned: bool = False,
    packed: bool | None = None,
) -> dict:
    """Reshape a SkyRL fwd_bwd payload into Cortex's wire shape.

    ``packed`` is whether the training job packs rows (see
    :func:`arctic_platform.client.config.cortex_job_packs`); ``None`` infers it
    from ``meta['zorro_train_enable']``. The caller's ``loss_fn`` and ``post``
    survive only on an unpacked job, whose executor runs on-prem's
    ``run_pipeline(pack=False)``: ZoRRo keeps ``verl_grpo`` (see
    :func:`_to_cortex_zorro_verl_grpo`), and an unpacked job without ZoRRo
    requires ``verl_grpo`` (see :func:`_to_cortex_unpacked_verl_grpo`). The
    packed executor credits log-probs only to the literal post
    ``compute_logprobs`` and scores the predict-next frame, so every packed
    request is rewritten to ``grpo`` as described below.

    Any ``old_log_probs`` in the batch are dropped: server-side GRPO defaults
    π_old to ``logprobs.detach()`` (π_old ≡ π_new), correct for single-epoch
    on-policy. A caller whose ratio is against another policy, such as the
    sampler's, passes ``old_log_probs_shifted`` -- ``[B, S]`` in the frame
    Cortex scores in, entry ``i`` being the log-prob of token ``i + 1`` -- and
    it is sent as π_old.

    Requires ``loss_mask`` or ``response_mask``; falling back to
    ``attention_mask`` would train on prompt tokens.

    Non-ZoRRo rows are left-aligned on the way out (see :func:`_left_align_batch`).
    Cortex's packer rejects any other layout. ZoRRo rows stay in the client
    layout: left-aligning them moves response tokens into ``[:, max_prompt_len]``.
    That path sends ``old_log_probs`` as ``old_log_probs_shifted`` so the
    snapshot from ``fwd_no_grad`` is the ratio's π_old.

    ``response_aligned`` says ``loss_mask`` and ``advantages`` sit on the token
    they score (SkyRL and verl), while Cortex weights position ``i`` by its
    prediction of token ``i + 1``. Non-ZoRRo rows are then shifted one column
    after left alignment, because the frame is defined on the rows Cortex
    receives. ZoRRo is never shifted: its patched forward returns one log-prob
    per response token. ``old_log_probs_shifted`` is already in the Cortex
    frame and is never shifted.
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
    old_log_probs = tensors.pop("old_log_probs", None)

    job_packs = _resolve_packed(meta, packed)
    if meta.get("zorro_train_enable"):
        return _to_cortex_zorro_fwd_bwd(
            tensors,
            meta,
            processing_in,
            input_ids,
            attention_mask,
            advantages,
            loss_mask,
            old_log_probs,
        )
    if not job_packs:
        return _to_cortex_unpacked_verl_grpo(
            tensors, meta, processing_in, input_ids, attention_mask, advantages, loss_mask, old_log_probs
        )

    forwarded = {"input_ids": input_ids}
    for k in ("position_ids", "labels"):
        if k in tensors:
            forwarded[k] = tensors[k]
    scored = {"advantages": advantages, "loss_mask": loss_mask}
    if old_log_probs_shifted is not None:
        scored["old_log_probs_shifted"] = old_log_probs_shifted.to(torch.float32)
    forwarded, scored, attention_mask = _left_align_batch(forwarded, attention_mask, scored)
    if response_aligned:
        for key in ("loss_mask", "advantages"):
            scored[key] = _to_predict_next_frame(key, scored[key])
    input_ids = forwarded["input_ids"]
    context: dict[str, Any] = {
        "input_ids": input_ids,
        **scored,
        **{key: meta[key] for key in ("temperature", "calculate_entropy") if key in meta},
    }

    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for k in ("position_ids", "labels"):
        if k in forwarded:
            kwargs_out[k] = forwarded[k]

    proc_config = _resolve_proc_config(meta, processing_in)

    from arctic_platform.common.balance_metadata import apply_balance_metadata

    apply_balance_metadata(context, meta, input_ids, tensors.get("prompts"))

    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": context,
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }


def _to_cortex_zorro_fwd_bwd(
    tensors: dict,
    meta: dict,
    processing_in: dict | None,
    input_ids,
    attention_mask,
    advantages,
    loss_mask,
    old_log_probs,
) -> dict:
    """ZoRRo wire shape. Rows are not left-aligned.

    The worker slices the response at ``max_prompt_len``. Left-aligning a
    shorter prompt moves response tokens into that slice and drops them from
    the loss.

    ``grpo`` reads π_old only as ``old_log_probs_shifted`` and otherwise uses
    the live log-probs, a ratio of 1. The ``fwd_no_grad`` snapshot is sent
    under that name without a shift: ZoRRo's log-probs, live and snapshot,
    sit on the response token they score, the same columns as ``loss_mask``.
    """
    import torch

    prompts = tensors.get("prompts")
    if not torch.is_tensor(prompts) or prompts.ndim != 2 or int(prompts.shape[1]) == 0:
        raise ValueError("ZoRRo fwd_bwd requires batch['prompts'] with width max_prompt_len")
    if (processing_in or {}).get("loss_fn") == "verl_grpo":
        return _to_cortex_zorro_verl_grpo(
            tensors, meta, processing_in, input_ids, attention_mask, prompts, advantages, loss_mask, old_log_probs
        )
    response_mask = tensors.get("response_mask", loss_mask)
    kwargs_out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": prompts,
        "response_mask": response_mask,
    }
    if torch.is_tensor(tensors.get("position_ids")):
        kwargs_out["position_ids"] = tensors["position_ids"]

    proc_config = _resolve_proc_config(meta, processing_in)
    # The unpacked ZoRRo call flattens every response into one row with no
    # cu_seqlens, so a per-sequence mean would treat the shard as one sequence.
    if proc_config["loss_agg_mode"] != "token-mean":
        raise ValueError(
            f"ZoRRo on Cortex supports only loss_agg_mode='token-mean', got {proc_config['loss_agg_mode']!r}: "
            "its responses reach the loss flattened, without sequence boundaries"
        )

    context: dict[str, Any] = {"input_ids": input_ids, "advantages": advantages, "loss_mask": loss_mask}
    if torch.is_tensor(old_log_probs):
        context["old_log_probs_shifted"] = old_log_probs.to(torch.float32)
    from arctic_platform.common.balance_metadata import apply_balance_metadata

    apply_balance_metadata(context, meta, input_ids, prompts)
    # The unpacked worker unpads with pad_token_id and the patched forward
    # divides by temperature. Balance metadata does not carry either scalar.
    for key in ("pad_token_id", "temperature"):
        if key in meta:
            context[key] = meta[key]
    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": context,
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }


def _to_cortex_zorro_verl_grpo(
    tensors: dict,
    meta: dict,
    processing_in: dict,
    input_ids,
    attention_mask,
    prompts,
    advantages,
    loss_mask,
    old_log_probs,
) -> dict:
    """The on-prem ``verl_grpo`` request on the Cortex wire, for ZoRRo.

    The ZoRRo worker runs the same ``run_pipeline`` as on-prem, which slices
    every ``[B, S]`` loss tensor at ``max_prompt_len`` into one flat response
    row. ``verl_grpo`` scores that row against the patched forward's
    response-aligned log-probs, so the tensors and meta go out in the client's
    layout, unshifted, under their on-prem names.

    ``dp_size`` is left to DSS, which sets it to the shard count alongside the
    global ``batch_num_tokens`` it derives from ``loss_mask``.
    """
    post = list(processing_in.get("post") or ["apply_temperature", "compute_entropy_and_logprobs"])
    return _verl_grpo_wire(
        "ZoRRo",
        tensors,
        meta,
        processing_in,
        post,
        input_ids,
        attention_mask,
        prompts,
        advantages,
        loss_mask,
        old_log_probs,
    )


def _to_cortex_unpacked_verl_grpo(
    tensors: dict,
    meta: dict,
    processing_in: dict | None,
    input_ids,
    attention_mask,
    advantages,
    loss_mask,
    old_log_probs,
) -> dict:
    """The on-prem ``verl_grpo`` request on the Cortex wire, for an unpacked job without ZoRRo.

    DSS runs the stock forward through the same ``run_pipeline(pack=False)``
    call as on-prem. Unlike ZoRRo, nothing here is sliced at
    ``max_prompt_len``: the generic unpad flattens every ``[B, S]`` tensor by
    ``attention_mask``, and ``compute_packing_info_for_batch`` derives each
    row's ``sequence_offsets`` from ``attention_mask`` and its ``response_lens``
    from the ``attention_mask`` columns at or past ``prompts.shape[1]``.
    ``verl_grpo`` then places the predict-next log-probs on the token they
    score inside those windows (``mask_and_pad_for_log_probs``), the column
    frame it reads the unshifted ``response_mask`` / ``advantages`` /
    ``old_log_probs`` in. So ``prompts`` and ``attention_mask`` must
    both be sent, and the rows must keep the client layout: left-aligning
    moves the prompt boundary off ``prompts.shape[1]``, and a one-column shift
    moves every weight off its token.

    DSS refuses a loss that needs token log-probs on the stock forward unless
    ``post`` names ``compute_logprobs``. That post returns existing log-probs
    unchanged, so it is appended after the caller's posts.
    """
    import torch

    if (processing_in or {}).get("loss_fn") != "verl_grpo":
        raise ValueError(
            "an unpacked Cortex training job without ZoRRo runs on-prem's unpacked pipeline, and the client "
            f"defines its wire shape only for loss_fn='verl_grpo'; got {(processing_in or {}).get('loss_fn')!r}. "
            "Send verl_grpo, or create a packed job"
        )
    prompts = tensors.get("prompts")
    if not torch.is_tensor(prompts) or prompts.ndim != 2 or int(prompts.shape[1]) == 0:
        raise ValueError(
            "unpacked verl_grpo fwd_bwd requires batch['prompts'] [B, P]: the worker reads each row's "
            "response length from the attention_mask columns at or past prompts.shape[1]"
        )
    if "pad_token_id" not in meta:
        raise ValueError("unpacked verl_grpo fwd_bwd requires meta['pad_token_id']: the worker unpads the rows with it")
    response_mask = tensors.get("response_mask", loss_mask)
    width = tuple(attention_mask.shape)
    for name, value in (
        ("response_mask", response_mask),
        ("loss_mask", loss_mask),
        ("advantages", advantages),
        ("old_log_probs", old_log_probs),
    ):
        if torch.is_tensor(value) and tuple(value.shape) != width:
            raise ValueError(
                f"unpacked verl_grpo fwd_bwd needs {name!r} with the attention_mask shape {width}, "
                f"got {tuple(value.shape)}: the worker unpads only tensors of that shape"
            )
    scored = response_mask.to(torch.bool)
    prompt_width = int(prompts.shape[1])
    if bool(scored[:, :prompt_width].any()) or bool((scored & ~attention_mask.to(torch.bool)).any()):
        raise ValueError(
            "unpacked verl_grpo fwd_bwd needs every response_mask token to be a valid token at or past "
            f"column prompts.shape[1]={prompt_width}; the loss only scores those columns"
        )
    post = list(processing_in.get("post") or ["apply_temperature", "compute_entropy_and_logprobs"])
    if "compute_logprobs" not in post:
        post.append("compute_logprobs")
    return _verl_grpo_wire(
        "Unpacked training",
        tensors,
        meta,
        processing_in,
        post,
        input_ids,
        attention_mask,
        prompts,
        advantages,
        loss_mask,
        old_log_probs,
    )


def _verl_grpo_wire(
    path: str,
    tensors: dict,
    meta: dict,
    processing_in: dict,
    post: list,
    input_ids,
    attention_mask,
    prompts,
    advantages,
    loss_mask,
    old_log_probs,
) -> dict:
    """``verl_grpo`` tensors and meta under their on-prem names, for an unpacked executor.

    ``dp_size`` is left to DSS, which sets it to the shard count alongside the
    global ``batch_num_tokens`` it derives from ``loss_mask``.
    """
    import torch

    if not torch.is_tensor(old_log_probs):
        raise ValueError(f"{path} verl_grpo fwd_bwd requires batch['old_log_probs'] [B, S]")
    actor = meta.get("actor_config") if isinstance(meta.get("actor_config"), dict) else {}
    loss_agg_mode = actor.get("loss_agg_mode", "token-mean")
    # The unpacked call flattens every response into one row with no
    # cu_seqlens, so a per-sequence mean would treat the shard as one sequence.
    if loss_agg_mode != "token-mean":
        raise ValueError(
            f"{path} on Cortex supports only loss_agg_mode='token-mean', got {loss_agg_mode!r}: "
            "its responses reach the loss flattened, without sequence boundaries"
        )
    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask, "prompts": prompts}
    if torch.is_tensor(tensors.get("position_ids")):
        kwargs_out["position_ids"] = tensors["position_ids"]
    context: dict[str, Any] = {
        key: value for key, value in meta.items() if key != "dp_size" and not torch.is_tensor(value) and value is not None
    }
    context.update(
        input_ids=input_ids,
        response_mask=tensors.get("response_mask", loss_mask),
        loss_mask=loss_mask,
        advantages=advantages,
        old_log_probs=old_log_probs.to(torch.float32),
    )
    from arctic_platform.common.balance_metadata import apply_balance_metadata

    apply_balance_metadata(context, meta, input_ids, prompts)
    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": context,
        "processing": {"post": post, "loss_fn": "verl_grpo", "config": dict(processing_in.get("config") or {})},
    }
