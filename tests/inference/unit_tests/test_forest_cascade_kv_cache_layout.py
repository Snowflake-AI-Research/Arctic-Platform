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

"""CPU tests for Forest Cascade's NHD/HND mapping of resolved KV layouts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.kv_cache_layout import KVCacheLayout

from arctic_platform.inference.vllm.attention.flash_attn_forest_cascade import (
    FlashAttentionBackend,
    get_kv_cache_layout,
)

NHD_STRIDE = (0, 1, 2, 3, 4)
HND_STRIDE = (0, 1, 3, 2, 4)


def _vllm_config_with_layout(layout: KVCacheLayout | None):
    cache_config = SimpleNamespace(kv_cache_layout=None)
    if layout is None:
        def _unresolved():
            raise ValueError(
                "KV cache layout has not been resolved yet; it is resolved once "
                "by the engine core (resolve_kv_cache_layout) unless explicitly "
                "set by the user."
            )

        cache_config.get_resolved_kv_cache_layout = _unresolved
    else:
        cache_config.kv_cache_layout = layout.name
        cache_config.get_resolved_kv_cache_layout = lambda: layout
    return SimpleNamespace(cache_config=cache_config)


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (KVCacheLayout.LBNHC, "NHD"),
        (KVCacheLayout.LBHNC, "HND"),
        (KVCacheLayout.BHLNC, "HND"),
        (KVCacheLayout.BLNHC, "NHD"),
        (KVCacheLayout.BLHNC, "HND"),
    ],
)
def test_get_kv_cache_layout_follows_resolved_layout(layout, expected, monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    with patch(
        "arctic_platform.inference.vllm.attention.flash_attn_forest_cascade.get_current_vllm_config",
        return_value=_vllm_config_with_layout(layout),
    ):
        assert get_kv_cache_layout() == expected


def test_stride_order_follows_resolved_bhlnc_not_env_default(monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    with patch(
        "arctic_platform.inference.vllm.attention.flash_attn_forest_cascade.get_current_vllm_config",
        return_value=_vllm_config_with_layout(KVCacheLayout.BHLNC),
    ):
        assert FlashAttentionBackend.get_kv_cache_stride_order() == HND_STRIDE


def test_resolved_layout_wins_over_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    with patch(
        "arctic_platform.inference.vllm.attention.flash_attn_forest_cascade.get_current_vllm_config",
        return_value=_vllm_config_with_layout(KVCacheLayout.LBNHC),
    ):
        assert get_kv_cache_layout() == "NHD"
        assert FlashAttentionBackend.get_kv_cache_stride_order() == NHD_STRIDE


def test_unresolved_layout_raises(monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    with patch(
        "arctic_platform.inference.vllm.attention.flash_attn_forest_cascade.get_current_vllm_config",
        return_value=_vllm_config_with_layout(None),
    ):
        with pytest.raises(ValueError, match="has not been resolved"):
            get_kv_cache_layout()


def test_unmapped_layout_raises(monkeypatch):
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    with patch(
        "arctic_platform.inference.vllm.attention.flash_attn_forest_cascade.get_current_vllm_config",
        return_value=_vllm_config_with_layout(KVCacheLayout.LHBNC),
    ):
        with pytest.raises(ValueError, match="no NHD/HND mapping"):
            get_kv_cache_layout()
