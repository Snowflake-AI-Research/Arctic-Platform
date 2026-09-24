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
"""Opt-in diagnostics for activation-offload slot ownership."""

from __future__ import annotations

import os
import traceback
from typing import Any

from arctic_platform.model.implementations.debug.memory import diagnostics_log_path


def maybe_log_activation_offload_slot(tensor: Any, staged: bool, slot_id: int = -1) -> None:
    """Record the identity and saving frame of an activation-offload slot when inventory logging is enabled."""
    if os.environ.get("DSS_ACT_OFFLOAD_INVENTORY") != "1":
        return
    # torch.autograd.Function.apply is the frame nearest the save and is shared by the checkpoint wrapper and
    # every other custom Function, so it identifies nothing. Keep the innermost frames outside torch and the
    # activation-offload call site; those name the module that saved the tensor.
    skip = ("activation_offload", "torch/autograd", "torch/utils", "torch/nn/functional")
    frames = []
    for frame in reversed(traceback.extract_stack()[:-2]):
        if any(marker in frame.filename.replace(os.sep, "/") for marker in skip):
            continue
        frames.append(f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}")
        if len(frames) == 4:
            break
    origin = " <- ".join(frames) or "?"
    mib = tensor.numel() * tensor.element_size() / (1024 * 1024)
    # The storage address distinguishes one tensor saved twice from two same-shaped tensors saved once.
    line = (
        f"[act-offload-slot] rank={os.environ.get('RANK', '?')} staged={int(staged)} slot={slot_id} "
        f"ptr={tensor.data_ptr():#x} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"size={mib:.1f}MiB from={origin}"
    )
    try:
        with open(diagnostics_log_path(), "a") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)
