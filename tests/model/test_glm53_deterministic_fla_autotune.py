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
"""Under full determinism GLM-5.3-Flash's FLA kernels take a fixed config, not a timed autotune pick."""

from __future__ import annotations

import types

import pytest
import torch


class _Stop(Exception):
    pass


@pytest.mark.parametrize("deterministic", [False, True])
def test_the_glm53_flash_loader_pins_fla_autotuning_only_under_full_determinism(monkeypatch, deterministic):
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as glm53_ds
    from arctic_platform.model.implementations.kernels import flashinfer_top_k
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen_ds

    calls = []
    monkeypatch.setattr(flashinfer_top_k, "prepare_flashinfer_top_k", lambda: None)
    monkeypatch.setattr(glm53_ds, "pin_fla_autotuning", lambda: calls.append("pin"))

    def load(*args, **kwargs):
        calls.append("load")
        raise _Stop

    monkeypatch.setattr(qwen_ds, "_load_moe_model", load)
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    try:
        with pytest.raises(_Stop):
            glm53_ds.load_glm5_next_model(
                model_name="m",
                optimization_dtype="bfloat16",
                attn_implementation="sparse_mla",
                ep_size=1,
                options=types.SimpleNamespace(),
            )
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    assert calls == (["pin", "load"] if deterministic else ["load"])


def test_pinned_fla_kernels_take_their_first_pruned_config_without_a_benchmark():
    pytest.importorskip("fla")
    from fla.modules import l2norm
    from triton.runtime.autotuner import Autotuner

    from arctic_platform.model.implementations.glm53.context_parallel import pin_fla_autotuning

    tuner = l2norm.l2norm_fwd_kernel
    assert isinstance(tuner, Autotuner) and len(tuner.configs) > 1
    assert pin_fla_autotuning() > 0
    # One config left after pruning: Triton's Autotuner.run caches it without benchmarking any.
    assert tuner.prune_configs({}) == [tuner.configs[0]]
    assert tuner.cache == {}
    assert pin_fla_autotuning() == 0  # already pinned: idempotent
