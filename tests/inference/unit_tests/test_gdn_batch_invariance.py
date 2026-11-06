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

import sys
import types

import pytest

from arctic_platform.inference.vllm import gdn_batch_invariance


class _Autotuner:
    def __init__(self, configs, fn):
        self.configs = list(configs)
        self.cache = {"stale": configs[-1]}
        self.fn = fn


class _Heuristics:
    def __init__(self, fn):
        self.fn = fn


@pytest.fixture
def fake_modules(monkeypatch):
    autotuner_module = types.ModuleType("triton.runtime.autotuner")
    autotuner_module.Autotuner = _Autotuner
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setitem(sys.modules, "triton.runtime", types.ModuleType("triton.runtime"))
    monkeypatch.setitem(sys.modules, "triton.runtime.autotuner", autotuner_module)

    kernels = {}
    prefix = "vllm.third_party.flash_linear_attention.ops"
    for name in ("vllm", "vllm.third_party", "vllm.third_party.flash_linear_attention", prefix):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name in gdn_batch_invariance._FLA_AUTOTUNED_MODULES:
        module = types.ModuleType(f"{prefix}.{name}")
        kernel = _Heuristics(_Autotuner(["a", "b", "c"], fn=object()))
        module.kernel = kernel
        kernels[name] = kernel
        monkeypatch.setitem(sys.modules, f"{prefix}.{name}", module)
    layernorm_guard = types.ModuleType(f"{prefix}.layernorm_guard")
    layernorm_guard.calc_rows_per_block = lambda M, device: 8
    monkeypatch.setitem(sys.modules, f"{prefix}.layernorm_guard", layernorm_guard)
    sys.modules[prefix].layernorm_guard = layernorm_guard

    backend = type("GDNAttentionBackend", (), {"supports_batch_invariance": classmethod(lambda cls: False)})
    gdn_attn = types.ModuleType("vllm.v1.attention.backends.gdn_attn")
    gdn_attn.GDNAttentionBackend = backend
    for name in ("vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.gdn_attn", gdn_attn)

    monkeypatch.setattr(gdn_batch_invariance, "_PINNED", False)
    monkeypatch.setattr(gdn_batch_invariance, "_APPLIED", False)
    return kernels, layernorm_guard, backend


def test_pin_keeps_one_config_through_wrappers_and_one_row_per_block(fake_modules):
    kernels, layernorm_guard, _ = fake_modules
    assert gdn_batch_invariance.pin_gdn_kernels() == len(kernels)
    for kernel in kernels.values():
        assert kernel.fn.configs == ["a"] and kernel.fn.cache == {}
    assert layernorm_guard.calc_rows_per_block(4096, None) == 1


def test_patch_is_opt_in(fake_modules, monkeypatch):
    kernels, _, backend = fake_modules
    monkeypatch.delenv(gdn_batch_invariance.ENV_VAR, raising=False)
    gdn_batch_invariance.ensure_gdn_batch_invariance()
    assert backend.supports_batch_invariance() is False
    assert all(len(kernel.fn.configs) == 3 for kernel in kernels.values())

    monkeypatch.setenv(gdn_batch_invariance.ENV_VAR, "1")
    gdn_batch_invariance.ensure_gdn_batch_invariance()
    assert backend.supports_batch_invariance() is True
    assert all(kernel.fn.configs == ["a"] for kernel in kernels.values())
