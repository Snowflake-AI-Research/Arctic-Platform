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
"""Liger kernel patch."""

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


@register_patch("liger")
def apply_liger(model: nn.Module, ctx: LoaderContext) -> None:
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

    # fused_linear_cross_entropy binds Liger's ``<arch>_lce_forward`` to this instance --
    # the same forward ``AutoLigerKernelForCausalLM`` rebinds at class level. It only
    # fuses when labels/shift_labels reach forward; callers that need logits omit
    # labels and get the ordinary lm_head projection.
    _apply_liger_kernel_to_instance(
        model=model,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rope=True,
        rms_norm=True,
        swiglu=True,
    )
