# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from arctic_platform.model.implementations.qwen35.vlm import freeze_unused_vision_tower


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
