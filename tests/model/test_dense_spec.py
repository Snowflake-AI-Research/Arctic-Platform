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

"""A dense Hugging Face model is one ModelSpec."""

from __future__ import annotations

from arctic_platform.model import ModelSpec
from arctic_platform.model import ParallelismConfig
from arctic_platform.model import Patches


def test_dense_model_spec_records_patches():
    spec = ModelSpec(
        model_path_or_name="Qwen/Qwen3-8B",
        dtype="bfloat16",
        attn_implementation="flash_attention_3",
        loader="huggingface",
        parallelism=ParallelismConfig(sequence_parallel=4),
        patches=Patches(
            liger=True,
            gradient_checkpointing=2,
            activation_offload={"keep_last_n": 3, "use_streams": False},
            compile={"fullgraph": False},
            lm_head={"fp32": True, "token_chunk_size": 128, "vocab_chunk_size": 4096},
            tiled_mlp={"token_chunk_size": 64},
            peft={"peft_type": "Lora", "r": 8, "target_modules": ["q_proj"]},
        ),
    )

    assert spec.loader == "huggingface"
    assert spec.dtype == "bfloat16"
    assert spec.parallelism.sequence_parallel == 4
    assert spec.patches.liger is True
    assert spec.patches.peft["r"] == 8
    assert spec.patches.gradient_checkpointing == 2
    assert spec.patches.activation_offload.keep_last_n == 3
    assert spec.patches.activation_offload.use_streams is False
    assert spec.patches.compile.fullgraph is False
    assert spec.patches.lm_head.fp32 is True
    assert spec.patches.lm_head.token_chunk_size == 128
    assert spec.patches.lm_head.vocab_chunk_size == 4096
    assert spec.patches.tiled_mlp.token_chunk_size == 64


def test_dense_offload_uses_the_typed_patch_default():
    spec = ModelSpec(
        model_path_or_name="Qwen/Qwen3-8B",
        dtype="bfloat16",
        attn_implementation="flash_attention_3",
        loader="huggingface",
        patches=Patches(
            gradient_checkpointing=True,
            activation_offload={"keep_last_n": 3},
        ),
    )

    assert spec.patches.activation_offload.tensor_size_threshold == 1 << 20
    assert spec.patches.activation_offload.use_streams is True
