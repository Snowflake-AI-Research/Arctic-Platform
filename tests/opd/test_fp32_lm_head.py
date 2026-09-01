# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.rl.processors.pipeline import _maybe_add_chunked_lm_head_kwargs
from arctic_platform.rl.processors.pipeline import collect_model_outputs
from arctic_platform.rl.processors.pipeline import uses_chunked_fp32_lm_head


class _FakeChunkedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_size = 2048
        self.fp32_lm_head = True


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = _FakeChunkedHead()


def test_import_inject_prime_lm_head_avoids_models_package():
    inject = DeepSpeedWorker._import_inject_prime_lm_head()
    assert callable(inject)


def test_collect_model_outputs_from_prime_dict():
    logprobs = torch.zeros(1, 4)
    out = collect_model_outputs({"logprobs": logprobs, "entropy": torch.ones(1, 4), "logits": None})
    assert torch.equal(out["logprobs"], logprobs)
    assert "logits" not in out


def test_uses_chunked_fp32_lm_head_detects_injected_head():
    engine = SimpleNamespace(module=_FakeModel())
    assert uses_chunked_fp32_lm_head(engine)
    assert not uses_chunked_fp32_lm_head(SimpleNamespace(module=nn.Linear(4, 4)))


def test_pipeline_injects_labels_and_temperature_for_chunked_head():
    engine = SimpleNamespace(module=_FakeModel())
    ids = torch.arange(5, dtype=torch.long)
    fwd = _maybe_add_chunked_lm_head_kwargs(engine, {"input_ids": ids})
    assert fwd["input_ids"].shape == (1, 5)
    assert torch.equal(fwd["labels"], torch.roll(fwd["input_ids"], -1, dims=-1))
    assert torch.equal(fwd["temperature"], torch.ones(1, 5))


def test_pipeline_does_not_inject_labels_for_vanilla_head():
    engine = SimpleNamespace(module=nn.Linear(4, 4))
    fwd = _maybe_add_chunked_lm_head_kwargs(engine, {"input_ids": torch.arange(5)})
    assert "labels" not in fwd
    assert "temperature" not in fwd
