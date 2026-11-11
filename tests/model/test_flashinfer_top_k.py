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
"""FlashInfer's top-k module is built at model load and never inside a step (kernels/flashinfer_top_k.py)."""

from __future__ import annotations

import types

import pytest

from arctic_platform.model.implementations.kernels import flashinfer_top_k


class _Stop(Exception):
    pass


def _prepares_before_loading(monkeypatch, loader, **kwargs) -> list:
    """The order of ``loader``'s calls of prepare_flashinfer_top_k and of the shared MoE load (stopped there)."""
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen_ds

    order = []
    monkeypatch.setattr(flashinfer_top_k, "prepare_flashinfer_top_k", lambda: order.append("prepare"))

    def load(*args, **kwargs):
        order.append("load")
        raise _Stop

    monkeypatch.setattr(qwen_ds, "_load_moe_model", load)
    with pytest.raises(_Stop):
        loader(model_name="m", optimization_dtype="bfloat16", ep_size=1, **kwargs)
    return order


def test_the_glm53_flash_loader_prepares_the_module_before_it_loads(monkeypatch):
    from arctic_platform.model.implementations.glm53.deepspeed_integration import load_glm5_next_model

    order = _prepares_before_loading(
        monkeypatch, load_glm5_next_model, attn_implementation="sparse_mla", options=types.SimpleNamespace()
    )
    assert order == ["prepare", "load"]
