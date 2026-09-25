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

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.implementations.qwen35.vlm import freeze_unused_vision_tower
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import apply_patches


class _TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _TinyVision()
        self.language_model = nn.Linear(4, 4)


class _TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.config = type("C", (), {"model_type": "qwen3_5"})()


def test_freeze_unused_vision_tower_disables_visual_grads():
    model = _TinyVLM()
    assert all(p.requires_grad for p in model.model.visual.parameters())
    frozen = freeze_unused_vision_tower(model, rank=0)
    assert frozen == sum(1 for _ in model.model.visual.parameters())
    assert all(not p.requires_grad for p in model.model.visual.parameters())
    assert all(p.requires_grad for p in model.model.language_model.parameters())


def test_freeze_unused_vision_tower_is_noop_without_visual():
    model = nn.Linear(3, 3)
    model.config = type("C", (), {"model_type": "llama"})()
    assert freeze_unused_vision_tower(model, rank=0) == 0
    assert all(p.requires_grad for p in model.parameters())


def test_vision_freeze_patch_is_opt_in():
    import types

    from arctic_platform.model.patches import freeze_unused_vision_tower as _freeze_patch  # noqa: F401

    model = _TinyVLM()
    apply_patches(
        LoadedModel(model=model),
        LoaderContext(spec=types.SimpleNamespace(patches=types.SimpleNamespace(freeze_unused_vision_tower=False))),
    )
    assert all(p.requires_grad for p in model.model.visual.parameters())

    apply_patches(
        LoadedModel(model=model),
        LoaderContext(spec=types.SimpleNamespace(patches=types.SimpleNamespace(freeze_unused_vision_tower=True))),
    )
    assert all(not p.requires_grad for p in model.model.visual.parameters())
