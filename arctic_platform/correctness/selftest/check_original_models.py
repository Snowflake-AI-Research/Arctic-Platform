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

"""Original-model metadata extraction preserves admission pins and the runner import."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from transformers.utils import hub

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.models import original
from arctic_platform.correctness.models import validate_local_full_model

QWEN8 = "Qwen/Qwen3-8B"
DENSE27 = "Qwen/Qwen3.8-27B"
FLASH = "Qwen/Qwen3.8-Flash-Next"
DENSE_CONFIG = "qwen3.8-27b-b200-train-sft-8gpus-2k"
FLASH_CONFIG = "qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k"
FLASH_SOURCE = (
    "/checkpoint/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)


def case(model, config_id, source):
    return SimpleNamespace(model_name=model, config_id=config_id), SimpleNamespace(
        model=SimpleNamespace(source_checkpoint=source)
    )


def test_runner_reexports_model_library_validator():
    assert runner.validate_local_full_model is validate_local_full_model
    assert validate_local_full_model is original.validate_local_full_model
    assert validate_local_full_model.__module__ == original.__name__


@pytest.mark.parametrize(
    "model,expected",
    [
        (
            QWEN8,
            (
                "b968826d9c46dd6066d109eabc6255188de91218",
                "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
                None,
                r"model\.layers\.(\d+)\.",
                None,
                None,
                None,
                None,
                None,
                {},
            ),
        ),
        (
            DENSE27,
            (
                "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
                "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
                r"model\.language_model\.layers\.(\d+)\.",
                1199,
                18,
                55562855904,
                (248320, 5120),
                "9c0eb904ed294a62882adb287b81e36fdb48e5fb8e006ee25b084341ac78fd55",
                {"BF16": 2, "F32": 4},
            ),
        ),
        (
            FLASH,
            (
                "de4b8e4d43b917e7706784d8bb445c9af86a3540",
                "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
                "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
                r"model\.language_model\.layers\.(\d+)\.",
                1658,
                131,
                359999963128,
                (248320, 2560),
                "1d55a6d3f64b1b2b004302864c0e74e11871ddf9f3481b10fc21fc8b0ec53d26",
                {"BF16": 2, "I64": 8},
            ),
        ),
    ],
)
def test_reviewed_original_metadata_pins_are_unchanged(model, expected):
    assert set(original.ORIGINAL_MODELS) == {QWEN8, DENSE27, FLASH}
    metadata = original.ORIGINAL_MODELS[model]
    assert metadata.model_name == model
    assert (
        metadata.revision,
        metadata.config_sha256,
        metadata.index_sha256,
        metadata.layer_pattern,
        metadata.tensor_count,
        metadata.shard_count,
        metadata.payload_bytes,
        metadata.vocabulary_shape,
        metadata.schema_sha256,
        dict(metadata.dtype_bytes),
    ) == expected


@pytest.mark.parametrize("model", [QWEN8, DENSE27, FLASH])
def test_registry_and_nested_metadata_are_immutable(model):
    metadata = original.ORIGINAL_MODELS[model]
    with pytest.raises(TypeError):
        original.ORIGINAL_MODELS[model] = metadata
    with pytest.raises(FrozenInstanceError):
        metadata.revision = "0" * 40
    with pytest.raises(TypeError):
        metadata.dtype_bytes["BF16"] = 4
    with pytest.raises(TypeError):
        metadata.source_checkpoints[0] = "unreviewed"
    if metadata.vocabulary_shape is not None:
        with pytest.raises(TypeError):
            metadata.vocabulary_shape[0] = 1


@pytest.mark.parametrize(
    "model,config_id,source",
    [
        (QWEN8, "qwen3-8b-h200-train-sft-full-4gpus-64k", QWEN8),
        (QWEN8, "another-config-using-the-original-qwen3-8b", QWEN8),
        (DENSE27, DENSE_CONFIG, DENSE27),
        (DENSE27, DENSE_CONFIG, "/data-fast/base-models/Qwen/Qwen3.8-27B"),
        (FLASH, FLASH_CONFIG, FLASH),
        (FLASH, FLASH_CONFIG, FLASH_SOURCE),
    ],
)
def test_original_selection_preserves_exact_accepted_sources(model, config_id, source):
    cfg, spec = case(model, config_id, source)
    assert original._original_model(cfg, spec) is original.ORIGINAL_MODELS[model]


@pytest.mark.parametrize(
    "model,config_id,source",
    [
        ("Qwen/unsupported", DENSE_CONFIG, DENSE27),
        (QWEN8, "any-config", "/data-fast/base-models/Qwen/Qwen3-8B"),
        (DENSE27, "unreviewed-config", DENSE27),
        (DENSE27, DENSE_CONFIG, "/data-fast/base-models/synthetic/Qwen3.8-27B-4L"),
        (FLASH, "unreviewed-config", FLASH),
        (FLASH, FLASH_CONFIG, FLASH_SOURCE.replace("de4b8", "00000")),
        (FLASH, FLASH_CONFIG, "/data-fast/base-models/Qwen/Qwen3.8-Flash-Next"),
    ],
)
def test_unreviewed_sources_rejected_before_cache_lookup(monkeypatch, model, config_id, source):
    cfg, spec = case(model, config_id, source)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: pytest.fail("unreviewed cache lookup"))
    with pytest.raises(ValueError, match="^local full model mode requires a supported pinned original source$"):
        validate_local_full_model(CortexConfig(base_url="http://127.0.0.1:8003"), cfg, spec)
