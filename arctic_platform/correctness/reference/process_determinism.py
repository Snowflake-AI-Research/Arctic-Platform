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

"""What a reference process pins at startup, before CUDA initializes.

Distinct from ``arctic_platform.model.implementations.debug.determinism``, which owns the engine's determinism switches and the
cuBLAS workspace string a caller puts in the environment. This module is only what a single-GPU reference
process does to itself: pin the ATen and cuDNN reduction orders, and ask flash attention for a
deterministic backward when the installed kernel will give one at this model's head dimension.

Shared by every reference entry point. A hundred-step trajectory that was not reduction-order pinned would
drift away from the engine it is compared against under its own optimizer steps, and measure that drift
instead of the engine.
"""

from __future__ import annotations


def pin_reduction_order(model_path: str, attn_implementation: str) -> None:
    """Pin every reduction order this process can pin.

    ``warn_only`` lets the run proceed where no deterministic kernel exists. The orders that can be pinned
    are worth more than the one that cannot, and the comparison's gate covers what is left free.
    """
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from arctic_platform.model.implementations.debug.determinism import pin_fla_gdn_autotuners

    pin_fla_gdn_autotuners(model_path)
    request_flash_attention_determinism(model_path, attn_implementation)


def request_flash_attention_determinism(model_path: str, attn_implementation: str) -> None:
    """Ask flash attention for a deterministic backward when the installed kernel will give one.

    Only the process holding the kernel can ask it, which is why this runs here rather than in the caller
    that sets the rest of the environment. Hugging Face's attention integration reads the variable when it
    calls the kernel, so setting it before the model is built is early enough. A build that refuses at this
    head dimension leaves the attention backward as the one reduction whose order is free, and the
    calibrated tolerance is what covers it.
    """
    import os

    if not attn_implementation.startswith("flash_attention"):
        return
    try:
        if attn_implementation == "flash_attention_4":
            from flash_attn.cute import flash_attn_varlen_func
        else:
            from flash_attn_interface import flash_attn_varlen_func
        from transformers import AutoConfig

        from arctic_platform.model.implementations.debug.determinism import (
            flash_attention_deterministic_backward_refusal,
        )

        model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        inner = getattr(model_cfg, "text_config", model_cfg)
        head_dim = int(getattr(inner, "head_dim", inner.hidden_size // inner.num_attention_heads))
        refusal = flash_attention_deterministic_backward_refusal(flash_attn_varlen_func, head_dim)
    except Exception as exc:  # an unavailable probe is not a reason to fail the run
        print(f"flash attention determinism: not probed ({type(exc).__name__}: {exc})", flush=True)
        return
    if refusal is None:
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        print(f"flash attention determinism: on at head_dim={head_dim}", flush=True)
    else:
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"
        print(f"flash attention determinism: off at head_dim={head_dim}, {refusal}", flush=True)
