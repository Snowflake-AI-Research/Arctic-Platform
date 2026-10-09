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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType
from types import SimpleNamespace

from arctic_platform.correctness.reference import process_determinism
from arctic_platform.model.implementations.debug import determinism


def test_fa4_determinism_probe_uses_fa4_entry_point(monkeypatch):
    fa3_varlen = object()
    fa4_varlen = object()
    flash_attn = ModuleType("flash_attn")
    flash_attn.__path__ = []
    flash_attn_cute = ModuleType("flash_attn.cute")
    flash_attn_cute.flash_attn_varlen_func = fa4_varlen
    flash_attn_interface = ModuleType("flash_attn_interface")
    flash_attn_interface.flash_attn_varlen_func = fa3_varlen
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", flash_attn_cute)
    monkeypatch.setitem(sys.modules, "flash_attn_interface", flash_attn_interface)

    config = SimpleNamespace(text_config=SimpleNamespace(head_dim=128, hidden_size=4096, num_attention_heads=32))
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: config)
    selected = []
    monkeypatch.setattr(
        determinism,
        "flash_attention_deterministic_backward_refusal",
        lambda entry_point, head_dim: selected.append((entry_point, head_dim)),
    )

    process_determinism.request_flash_attention_determinism("model", "flash_attention_4")

    assert selected == [(fa4_varlen, 128)]
