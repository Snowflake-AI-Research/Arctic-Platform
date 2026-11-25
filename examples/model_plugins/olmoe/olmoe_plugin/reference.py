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

The weights reach it as a real checkpoint: converted by ``convert_layer_to_hf``, saved, and loaded with
``from_pretrained``, which maps checkpoint names to transformers' own layout. So it also checks the converters
against what the checkpoint's names mean: swapping ``gate_proj`` and ``up_proj``, say, would survive a save and
reload but not this comparison.
"""

from __future__ import annotations

import tempfile

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import OlmoeForCausalLM

from .weights import convert_layer_to_hf


def reference_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits ``[L, vocab]`` of one unpacked sequence ``input_ids`` ``[L]``, in float64 with eager attention and experts."""
    state = convert_layer_to_hf(model.state_dict(), -1, model.config)
    with tempfile.TemporaryDirectory() as path:
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}, f"{path}/model.safetensors"
        )
        model.config.save_pretrained(path)
        reference = OlmoeForCausalLM.from_pretrained(
            path, dtype=torch.float64, attn_implementation="eager", experts_implementation="eager"
        )
    with torch.no_grad():
        return reference.to(input_ids.device)(input_ids[None]).logits[0]
