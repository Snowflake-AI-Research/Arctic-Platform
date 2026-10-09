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

# !/usr/bin/env python3
"""CPU-only checks for automatic reference sizing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from arctic_platform.correctness.onboarding import calibration as RENDER_MODULE
from arctic_platform.correctness.onboarding import onboard as MODULE
from arctic_platform.correctness.selftest.config_factory import native_config


def _render_config(tmp_path: Path, prime_rl: dict | None, peft_config: dict | None = None) -> str:
    config = tmp_path / "configs/model/h200/workload.config"
    config.parent.mkdir(parents=True)
    training = {
        "n_gpus": 8,
        "sp_size": 8,
        "attn_implementation": "flash_attention_3",
    }
    if prime_rl is not None:
        training["prime_rl"] = prime_rl
    if peft_config is not None:
        training["peft_config"] = peft_config
    config.write_text(json.dumps(native_config(training)))
    args = argparse.Namespace(
        config=config,
        source_checkpoint="source",
        layers=4,
        model_path=tmp_path / "model",
        output_json=tmp_path / "output.json",
        output_script=tmp_path / "driver.py",
        test_spec=tmp_path / "spec.json",
        test_id="single-step-grads",
        tokens=2048,
        reference_token_budget=65536,
    )
    rendered = RENDER_MODULE.render(args)
    compile(rendered, "<generated-reference-calibration>", "exec")
    return rendered


def test_renderer_reads_nested_prime_rl_chunk_size(tmp_path: Path) -> None:
    rendered = _render_config(
        tmp_path,
        {"fused_lm_head_token_chunk_size": 8192, "fused_cross_entropy": False},
    )

    assert "lm_head_token_chunk_size=8192" in rendered


def test_renderer_preserves_runtime_default_without_chunking(tmp_path: Path) -> None:
    rendered = _render_config(tmp_path, None)

    assert "lm_head_token_chunk_size=None" in rendered


def test_renderer_uses_configured_fused_cross_entropy(tmp_path: Path) -> None:
    rendered = _render_config(
        tmp_path,
        {"fused_lm_head_token_chunk_size": "disabled", "fused_cross_entropy": "liger"},
    )

    assert "fused_cross_entropy='liger'" in rendered


def test_renderer_passes_peft_config_to_reference(tmp_path: Path) -> None:
    rendered = _render_config(
        tmp_path,
        None,
        {"peft_type": "Lora", "r": 32, "target_modules": ["q_proj", "v_proj"]},
    )

    assert "peft_config={'peft_type': 'Lora', 'r': 32, 'target_modules': ['q_proj', 'v_proj']}" in rendered


def test_model_shape_uses_first_prefix_containing_every_attention_type(monkeypatch) -> None:
    class TextConfig:
        num_hidden_layers = 8
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2

    class Config:
        text_config = TextConfig()

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: Config())

    assert MODULE._model_shape("unused") == (8, 4, 4)


def test_model_shape_applies_four_layer_floor(monkeypatch) -> None:
    class TextConfig:
        num_hidden_layers = 8
        layer_types = ["full_attention"] * 8

    class Config:
        text_config = TextConfig()

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: Config())

    assert MODULE._model_shape("unused") == (8, 1, 4)


def test_optimizer_reference_model_does_not_replace_test1_model(monkeypatch, tmp_path: Path) -> None:
    test1_path = tmp_path / "Qwen3.8-27B-36L"
    spec = SimpleNamespace(
        model=SimpleNamespace(
            source_checkpoint="Qwen/Qwen3.8-27B",
            cache_path=str(test1_path),
            num_hidden_layers=36,
        )
    )
    monkeypatch.setattr(MODULE, "_model_shape", lambda unused: (64, 4, 4))

    model_path, layers = MODULE._optimizer_reference_model(spec, tmp_path)

    assert model_path == tmp_path / "Qwen3.8-27B-4L"
    assert layers == 4
    assert spec.model.cache_path == str(test1_path)
    assert spec.model.num_hidden_layers == 36


def test_cuda_oom_classification_does_not_swallow_other_failures() -> None:
    assert MODULE._is_cuda_oom(RuntimeError("CUDA out of memory while allocating tensor"))
    assert not MODULE._is_cuda_oom(RuntimeError("shape mismatch"))


def test_final_validation_requires_semantic_pass() -> None:
    report = {"totals": {"pass": 2, "fail": 0, "inapplicable": 0}}

    assert MODULE._validation_outcome(0, "Overall: 2 passed, 0 failed", report) == "pass"


def test_final_validation_classifies_cuda_oom_for_retry() -> None:
    assert MODULE._validation_outcome(1, "torch.OutOfMemoryError: CUDA out of memory", None) == "oom"


def test_final_validation_does_not_retry_correctness_failure() -> None:
    report = {"totals": {"pass": 1, "fail": 1, "inapplicable": 0}}

    assert MODULE._validation_outcome(1, "Overall: 1 passed, 1 failed", report) == "failed"


def test_source_checkpoint_defaults_to_training_model_name() -> None:
    config = argparse.Namespace(model_name="Qwen/Qwen3-8B")

    assert MODULE._select_source_checkpoint(None, config) == "Qwen/Qwen3-8B"


def test_source_checkpoint_override_wins() -> None:
    config = argparse.Namespace(model_name="Qwen/Qwen3-8B")

    assert MODULE._select_source_checkpoint("private/model", config) == "private/model"
