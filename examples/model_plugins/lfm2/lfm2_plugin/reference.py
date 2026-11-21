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

"""The conformance oracle: Hugging Face's own LFM2 implementation, loaded from a checkpoint of the trainer's weights.

``from_pretrained`` reads the checkpoint as it would a released one, tied embeddings included.
"""

from __future__ import annotations

import tempfile

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import Lfm2ForCausalLM


def reference_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits ``[L, vocab]`` of one unpacked sequence ``input_ids`` ``[L]``, in float64 with eager attention."""
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        state.pop("lm_head.weight")
    with tempfile.TemporaryDirectory() as path:
        save_file(state, f"{path}/model.safetensors")
        model.config.save_pretrained(path)
        reference = Lfm2ForCausalLM.from_pretrained(path, dtype=torch.float64, attn_implementation="eager")
    with torch.no_grad():
        return reference.to(input_ids.device)(input_ids[None]).logits[0]
