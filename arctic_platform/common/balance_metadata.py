# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Client metadata for rank assignment.

Stamps ``balance_batch`` and ZoRRo prompt-group ids onto a request. The
assignment itself is ``arctic_platform.common.utils.balanced_batch``, which
both the on-prem splitter and Cortex call. This module stays beside
``common.utils`` so importing it does not run that package init.
"""

from __future__ import annotations

LEGACY_BALANCE_KEYS = ("load_balancer", "zorro_load_balancer", "zorro_train_load_balancer")

# Scalars the on-prem splitter reads when ZoRRo balancing is on.
ZORRO_SPLIT_KEYS = (
    "rollout_n",
    "zorro_train_max_rollouts",
    "max_prompt_len",
    "max_response_len",
    "max_token_len_per_gpu",
)


def rollout_prompt_group_ids(input_ids, prompts, rollout_n):
    """Block ids for a ZoRRo request with balancing enabled.

    ``None`` when balancing is off or the row count is not a multiple of
    ``rollout_n``. Each block must share prompt columns.
    """
    import torch

    if rollout_n is None or not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        return None
    rows = int(input_ids.shape[0])
    width = int(rollout_n)
    if width < 1 or rows % width != 0:
        return None
    if not torch.is_tensor(prompts) or prompts.ndim != 2 or int(prompts.shape[0]) != rows:
        raise ValueError("load_balancer prompt_group_ids require batch prompts")
    for start in range(0, rows, width):
        block = prompts[start : start + width]
        if not torch.equal(block, block[:1].expand_as(block)):
            raise ValueError(
                "load_balancer prompt_group_ids require each rollout block to share prompt columns"
            )
    return torch.arange(rows, dtype=torch.int64) // width


def assert_contiguous_rollout_blocks(prompts, rollout_n) -> None:
    """Each contiguous block of ``rollout_n`` prompt rows must be identical.

    A row count that is not a multiple of ``rollout_n`` does not raise.
    """
    import torch

    if not torch.is_tensor(prompts) or prompts.ndim != 2:
        raise ValueError("load_balancer prompt_group_ids require batch prompts")
    rows = int(prompts.shape[0])
    width = int(rollout_n)
    if width < 1 or rows % width != 0:
        return
    for start in range(0, rows, width):
        block = prompts[start : start + width]
        if not torch.equal(block, block[:1].expand_as(block)):
            raise ValueError(
                "load_balancer prompt_group_ids require each rollout block to share prompt columns"
            )


def apply_balance_metadata(context: dict, meta: dict, input_ids, prompts) -> dict:
    """Copy the client balance stamp onto the Cortex context.

    ``True`` and ``False`` are both kept. A missing switch stays missing, and
    the server treats that as balance on. ``prompt_group_ids`` are copied from
    the client stamp; this function does not derive them. Legacy balancer keys
    are removed.
    """
    import torch

    for key in LEGACY_BALANCE_KEYS:
        context.pop(key, None)
    balance_batch = meta.get("balance_batch", None)
    if balance_batch is True or balance_batch is False:
        context["balance_batch"] = balance_batch
    else:
        context.pop("balance_batch", None)
    if meta.get("zorro_train_enable"):
        context["zorro_train_enable"] = True
        for key in ZORRO_SPLIT_KEYS:
            if key in meta:
                context[key] = meta[key]
    group_ids = meta.get("prompt_group_ids")
    if meta.get("balance_batch") is not False and torch.is_tensor(group_ids):
        context["prompt_group_ids"] = group_ids
    else:
        context.pop("prompt_group_ids", None)
    return context


def prompt_rows_for_balance(prompts, input_ids=None, max_prompt_len=None):
    """Dense prompt rows for the ZoRRo block check.

    A 2D ``prompts`` tensor is used as-is. A missing or jagged tensor falls
    back to the leading ``max_prompt_len`` columns of ``input_ids``.
    """
    import torch

    if torch.is_tensor(prompts) and not getattr(prompts, "is_nested", False) and prompts.ndim == 2:
        return prompts
    if (
        torch.is_tensor(input_ids)
        and not getattr(input_ids, "is_nested", False)
        and input_ids.ndim == 2
        and max_prompt_len is not None
    ):
        return input_ids[:, : int(max_prompt_len)]
    return prompts


def stamp_client_balance(
    meta: dict,
    *,
    balance_batch: bool,
    zorro_train_enable: bool,
    rollout_n,
    prompts=None,
    input_ids=None,
    max_prompt_len=None,
) -> None:
    """Stamp one SkyRL or VeRL request.

    ``balance_batch`` false is a positional call: arrival order and no group
    ids. A training call passes the configured flag. ZoRRo group ids are built
    from :func:`prompt_rows_for_balance`.
    """
    rows = prompts
    if balance_batch and zorro_train_enable:
        rows = prompt_rows_for_balance(prompts, input_ids, max_prompt_len)
    stamp_balance_meta(
        meta,
        balance_batch=balance_batch,
        zorro_train_enable=zorro_train_enable,
        prompts=rows,
        rollout_n=rollout_n,
    )


def stamp_balance_meta(meta: dict, *, balance_batch: bool, zorro_train_enable: bool, prompts, rollout_n) -> None:
    """Write the explicit balance switch onto on-prem meta.

    ``False`` keeps arrival order. ``True`` requests balancing and, for ZoRRo,
    checks that each rollout block shares prompt columns. A valid ZoRRo block
    also stores ``prompt_group_ids`` (``row // rollout_n``) so on-prem and Cortex
    group the same rows.
    """
    for key in LEGACY_BALANCE_KEYS:
        meta.pop(key, None)
    meta.pop("prompt_group_ids", None)
    meta["balance_batch"] = True if balance_batch else False
    if not balance_batch or not zorro_train_enable:
        return
    assert_contiguous_rollout_blocks(prompts, rollout_n)
    group_ids = rollout_prompt_group_ids(prompts, prompts, rollout_n)
    if group_ids is not None:
        meta["prompt_group_ids"] = group_ids
