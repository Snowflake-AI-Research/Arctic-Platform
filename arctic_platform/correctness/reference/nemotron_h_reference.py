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

"""Nemotron-H's HF reference with the Mamba-2 time step the sampler serves it with: dt unfloored.

Transformers 5.20.2's ``NemotronHMamba2Mixer`` floors every Mamba-2 time step at ``config.time_step_min`` (1e-3 for
Nemotron-3-Nano): it sets ``self.time_step_limit = (config.time_step_min, inf)`` (models/nemotron_h/
modeling_nemotron_h.py:414) and passes it as ``dt_limit`` to the fused training kernel (:482) and to the chunk scan
(:594, whose torch path clamps at :295). The checkpoint's own code uses ``time_step_min`` only to initialize dt_bias
(its modeling_nemotron_h.py:1229-1230, as Transformers' at :1031-1032) and limits dt by ``config.time_step_limit``,
(0, inf) by default (modeling_nemotron_h.py:307, :425, configuration_nemotron_h.py:171, :232). vLLM 0.33's MambaMixer2
does the same: ``dt_limit=(0.0, inf)`` on both prefill paths (model_executor/layers/mamba/mamba_mixer2.py:704, :883)
and no limit at all in the decode kernel (ops/mamba_ssm.py:405-420). So the sampler's dt is
``softplus(dt + dt_bias)`` (softplus's threshold of 20 everywhere), never raised to 1e-3.

On Nemotron-3-Nano's layer 0 with the suite's gas1 input, 12.7% of (token, head) pairs fall under the floor (median
unfloored dt 2.3e-4; 4 heads under it on more than 99% of tokens, 6 on 1-99%). Raising them to 1e-3 changes the
layer's output by 6.3% (relative L2, float64) and, at the same upstream gradient, its conv1d.weight gradient by 49%,
dt_bias's by 57% (the floor also zeroes dt's gradient wherever it binds), in_proj's by 38%.

The reference sets each mixer's ``time_step_limit`` to (0, inf); nothing else changes. Arctic's target trains with
the floor too (it builds on Transformers' mixer: nemotron_h/modeling_nemotron_h.py ``_varlen_mamba_forward`` and
layers/cp_mamba.py ``fused_mamba_mixer`` / ``mamba_cp_forward`` pass ``mixer.time_step_limit``); that departure from
the sampler is the target's, reported, not hidden here.
"""

from __future__ import annotations

NEMOTRON_H_MODEL_TYPES = frozenset({"nemotron_h"})
# vLLM's MambaMixer2 dt_limit (mamba_mixer2.py:704, :883), the checkpoint's config.time_step_limit default.
SAMPLER_TIME_STEP_LIMIT = (0.0, float("inf"))


def unfloor_time_step(model) -> int:
    """Give every Transformers Nemotron-H Mamba-2 mixer in ``model`` the sampler's dt limit (0, inf); returns how many
    there were. Refuses a model with none, which would leave nothing aligned."""
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

    mixers = [module for module in model.modules() if isinstance(module, NemotronHMamba2Mixer)]
    if not mixers:
        raise ValueError(f"{type(model).__name__} has no NemotronHMamba2Mixer to give the sampler's unfloored dt")
    for mixer in mixers:
        mixer.time_step_limit = SAMPLER_TIME_STEP_LIMIT
    return len(mixers)


def load_nemotron_h_reference(model_class, model_path, *, dtype, attn_implementation, device):
    """Load the checkpoint, then lift Transformers' dt floor so the Mamba-2 mixers run the sampler's time step."""
    model = model_class.from_pretrained(model_path, dtype=dtype, attn_implementation=attn_implementation)
    unfloor_time_step(model)
    return model.to(device)
