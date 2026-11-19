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

"""The conformance oracle: Hugging Face's own OLMoE implementation, with the trainer model's weights.

It loads them through ``convert_layer_to_hf``, so it also checks the converters against what the checkpoint's names
mean: swapping ``gate_proj`` and ``up_proj``, say, would survive a save and reload but not this comparison.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import OlmoeForCausalLM

from .weights import convert_layer_to_hf


def reference_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits ``[L, vocab]`` of one unpacked sequence ``input_ids`` ``[L]``, in float64 with eager attention."""
    config = model.config.__class__.from_dict(model.config.to_dict())
    config._attn_implementation = "eager"
    reference = OlmoeForCausalLM(config).to(device=input_ids.device, dtype=torch.float64)
    state = convert_layer_to_hf(model.state_dict(), -1, model.config)
    reference.load_state_dict({name: tensor.double() for name, tensor in state.items()})
    with torch.no_grad():
        return reference(input_ids[None]).logits[0]
