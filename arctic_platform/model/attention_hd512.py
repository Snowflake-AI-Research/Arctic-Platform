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
"""Head-512 causal GQA attention (Gemma 4's full-attention layers): the public entry to the TileLang kernels of
``implementations/kernels/causal_attention_hd512.py``.

Importing this module loads them and registers the ``arctic_hd512::forward`` / ``::backward`` dispatcher ops, so a
caller that sends head-512 attention here (DSS's heterogeneous Ulysses) can import it before the model loads and let
``patches.checkpoint_keep_attention`` name ``arctic_hd512::forward``.
"""

from arctic_platform.model.implementations.kernels.causal_attention_hd512 import FORWARD_OP
from arctic_platform.model.implementations.kernels.causal_attention_hd512 import causal_gqa_attention_hd512
from arctic_platform.model.implementations.kernels.causal_attention_hd512 import unsupported_reason

__all__ = ["FORWARD_OP", "causal_gqa_attention_hd512", "unsupported_reason"]
