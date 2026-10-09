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

"""Which model implementation a job actually runs.

The carved-out Qwen3.5 implementation is optional at import time, because it needs GPU-only kernels that a
CPU or a partial environment does not have. Tolerating a *missing dependency* is not the same as tolerating a
*broken* implementation, and the difference matters: everything measured about the carved-out model -- its
fused lm_head, its fp32 projection, its requirement for global packed-sequence boundaries -- is a property of
that code and not of the Hugging Face model that stands in for it.
"""

import importlib
import sys
from importlib.abc import MetaPathFinder

import pytest
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

MODELS_PACKAGE = "arctic_platform.model.implementations.qwen35.models"
CARVED_OUT_MODULE = f"{MODELS_PACKAGE}.qwen3_5_moe"


class _ExplodingFinder(MetaPathFinder):
    """Make importing the carved-out model fail the way broken code does, not the way absence does."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == CARVED_OUT_MODULE:
            raise RuntimeError("a kernel this model needs is broken")
        return None


def test_a_broken_carved_out_model_stops_the_job_rather_than_swapping_in_another():
    """A job that quietly runs a different model returns finite losses and green parity numbers for the wrong
    code, and the only trace is one warning line."""
    package = importlib.import_module(MODELS_PACKAGE)
    finder = _ExplodingFinder()
    sys.modules.pop(CARVED_OUT_MODULE, None)
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(RuntimeError, match="kernel this model needs"):
            importlib.reload(package)
    finally:
        sys.meta_path.remove(finder)
        importlib.reload(package)


def test_an_absent_optional_dependency_keeps_hf_imports_available():
    """The package remains importable without GPU kernels, while retaining the error for model selection."""
    package = importlib.import_module(MODELS_PACKAGE)
    sys.modules.pop(CARVED_OUT_MODULE, None)
    # What a failed import leaves behind; importing from it raises ImportError.
    sys.modules[CARVED_OUT_MODULE] = None
    try:
        reloaded = importlib.reload(package)
        assert reloaded._QWEN3_5_CUSTOM_IMPL_AVAILABLE is False
        assert isinstance(reloaded.get_custom_impl_import_error(), ImportError)
    finally:
        sys.modules.pop(CARVED_OUT_MODULE, None)
        importlib.reload(package)


def test_composite_qwen35_config_uses_its_custom_vlm_registration():
    """Composite configs are supported through the VLM map, not the text-only auto mapping."""
    from arctic_platform.model.implementations.qwen35.model_builder import _resolve_model_impl

    assert (
        _resolve_model_impl(
            Qwen3_5MoeConfig(),
            "custom",
            is_vlm_arch=True,
            custom_vlm_cls=type("CustomQwen35", (), {}),
        )
        == "custom"
    )


def test_unavailable_explicit_custom_implementation_raises_with_import_cause(monkeypatch):
    """An explicit custom request must fail at selection rather than load a different model."""
    from arctic_platform.model.implementations.qwen35 import model_builder

    import_error = ImportError("missing custom Qwen3.5 kernel")
    monkeypatch.setattr(model_builder, "get_custom_impl_import_error", lambda: import_error)

    with pytest.raises(RuntimeError, match="explicitly requested") as exc_info:
        model_builder._resolve_model_impl(
            Qwen3_5MoeConfig(),
            "custom",
            is_vlm_arch=True,
            custom_vlm_cls=None,
        )

    assert exc_info.value.__cause__ is import_error


def test_auto_can_still_select_hf_when_custom_is_unavailable():
    from arctic_platform.model.implementations.qwen35.model_builder import _resolve_model_impl

    assert (
        _resolve_model_impl(
            Qwen3_5MoeConfig(),
            "auto",
            is_vlm_arch=True,
            custom_vlm_cls=None,
        )
        == "hf"
    )


def test_strip_lora_from_state_dict_normalizes_nested_peft_wrappers():
    from arctic_platform.model.implementations.qwen35.model_builder import strip_lora_from_state_dict

    fused_weight = object()
    dense_weight = object()
    state = {
        "model.layers.0.mlp.experts.base_layer.base_layer.w1.weight": fused_weight,
        "model.layers.0.self_attn.q_proj.base_layer.weight": dense_weight,
        "model.layers.0.mlp.experts.lora_A.default.weight": object(),
    }

    assert strip_lora_from_state_dict(state) == {
        "model.layers.0.mlp.experts.w1": fused_weight,
        "model.layers.0.self_attn.q_proj.weight": dense_weight,
    }
