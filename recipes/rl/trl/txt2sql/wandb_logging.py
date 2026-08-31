# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""W&B defaults for the BIRD launchers. ``REPORT_TO=none`` disables logging."""

from __future__ import annotations

import os
from datetime import datetime, timezone


def configure_wandb(
    *,
    config: str,
    max_steps: int,
    per_device_bsz: int,
    grad_accum: int,
    zorro: bool = False,
    seed: int | None = None,
) -> tuple[str, str | None]:
    """Return ``(report_to, run_name)`` and set W&B env defaults in-place."""
    os.environ.setdefault("WANDB_PROJECT", "arctic-trl-bird")
    report_to = (os.environ.get("REPORT_TO") or "wandb").strip() or "wandb"
    if report_to == "none":
        return "none", None

    os.environ.pop("WANDB_MODE", None)
    os.environ.pop("WANDB_SILENT", None)
    os.environ.setdefault("WANDB_RUN_GROUP", f"trl-bird-{max_steps}step")

    run_name = os.environ.get("WANDB_RUN_NAME") or os.environ.get("RUN_NAME") or None
    if not run_name:
        tag = "C3" if (config == "C3" or zorro) else config
        parts = [
            tag,
            f"s{max_steps}",
            f"bsz{per_device_bsz}",
            f"gas{grad_accum}",
        ]
        if zorro:
            parts.append("zorro")
        if seed is not None:
            parts.append(f"seed{seed}")
        parts.append(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        run_name = "_".join(parts)
        os.environ["WANDB_RUN_NAME"] = run_name
    return report_to, run_name
