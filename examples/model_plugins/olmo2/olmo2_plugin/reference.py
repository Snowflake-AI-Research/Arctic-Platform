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

"""The conformance oracle: Hugging Face's own OLMo-2 implementation, loaded with the trainer model's weights.

Any independent implementation works. For an architecture transformers already implements, that implementation is
the natural one: matching it also means matching the checkpoint's semantics, which vLLM's built-in model follows.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import Olmo2ForCausalLM


def reference_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits ``[L, vocab]`` of one unpacked sequence ``input_ids`` ``[L]``, in float64 with eager attention."""
    config = model.config.__class__.from_dict(model.config.to_dict())
    config._attn_implementation = "eager"
    reference = Olmo2ForCausalLM(config).to(device=input_ids.device, dtype=torch.float64)
    reference.load_state_dict({name: tensor.double() for name, tensor in model.state_dict().items()})
    with torch.no_grad():
        return reference(input_ids[None]).logits[0]
