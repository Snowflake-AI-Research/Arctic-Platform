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

"""W&B defaults for the BIRD launchers. ``REPORT_TO=none`` disables logging."""

from __future__ import annotations

import os
from datetime import datetime
from datetime import timezone


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
