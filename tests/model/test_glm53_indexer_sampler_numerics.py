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

"""The trainer must attend to the tokens the vLLM 0.31 sampler's GLM-5.3 indexer selected for the same input."""

import pytest
import torch
from torch.nn import functional as F

from arctic_platform.model.implementations.glm53 import context_parallel as cp

configuration = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

HEAD_DIM = 128
KPOOL = 4
POOLS = 4
ROWS = (150, 7, 147)


def _indexer():
    config = configuration.Glm5NextTextConfig(
        vocab_size=64,
        pad_token_id=0,
        hidden_size=64,
        num_hidden_layers=1,
        layer_types=["deepseek_sparse_attention"],
        indexer_types=["full"],
        mlp_layer_types=["dense"],
        q_lora_rank=32,
        qk_rope_head_dim=0,
        index_head_dim=HEAD_DIM,
        index_n_heads=32,
        index_topk=POOLS * KPOOL,
        index_kpool=KPOOL,
        index_kpool_always_select_tail=True,
    )
    indexer = modeling.Glm5NextTextIndexer(config, layer_idx=0)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in indexer.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * parameter.shape[-1] ** -0.5)
    return indexer.to(torch.bfloat16)


@torch.no_grad()
def _vllm_tokens(indexer, hidden, q_resid):
    """Per query of one row, the token set vLLM's glm5next indexer attends to (common/attention.py Indexer.forward,
    nvidia/ops/kpool_compress.py, DeepGEMM fp8 MQA logits, top-k over complete visible pools plus the raw tail)."""
    hadamard = torch.ones(1, 1, dtype=torch.float64)
    while hadamard.shape[0] < HEAD_DIM:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)))

    def rotate_and_quantize(x):
        x = ((x.to(torch.bfloat16).double() @ hadamard).float() * 0.08838834764831845).to(torch.bfloat16).float()
        scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-4) * (1.0 / 448.0))))
        return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).double(), scale.double()

    length = hidden.shape[0]
    q8, q_scale = rotate_and_quantize(indexer.wq_b(q_resid).view(length, -1, HEAD_DIM))
    gate = hidden.float() @ indexer.weights_proj.weight.float().t()
    gate = gate.double() * q_scale[..., 0] * (HEAD_DIM**-0.5 * indexer.n_heads**-0.5)
    pools = length // KPOOL
    keys = indexer.k_norm(indexer.wk(hidden))[: pools * KPOOL].float().view(pools, KPOOL, HEAD_DIM)
    logits = F.linear(hidden, indexer.index_kpool_compress_gate)[: pools * KPOOL].float().view(pools, KPOOL, HEAD_DIM)
    logits = logits + indexer.index_kpool_compress_ape.float()
    weights = torch.exp(logits - logits.amax(1, keepdim=True))
    k8, k_scale = rotate_and_quantize((keys * weights).sum(1) / weights.sum(1))
    scores = (torch.einsum("qhd,pd->qhp", q8, k8).relu() * k_scale[:, 0] * gate[..., None]).sum(1)

    selected = []
    for position in range(length):
        visible = (position + 1) // KPOOL
        chosen = scores[position, :visible].topk(min(POOLS, visible)).indices.tolist()
        tokens = {pool * KPOOL + offset for pool in chosen for offset in range(KPOOL)}
        selected.append(tokens | set(range(visible * KPOOL, position + 1)))
    return selected


def _token_sets(indices):
    return [{int(token) for token in row if token >= 0} for row in indices]


def test_trainer_selects_the_samplers_tokens_in_packed_rows():
    torch.manual_seed(0)
    indexer = _indexer()
    length = sum(ROWS)
    hidden = torch.randn(1, length, 64, dtype=torch.bfloat16)
    q_resid = torch.randn(1, length, 32, dtype=torch.bfloat16)
    boundaries = torch.tensor([0, *torch.tensor(ROWS).cumsum(0).tolist()], dtype=torch.int32)

    trainer = cp._select_sparse_indices(
        indexer,
        hidden,
        q_resid,
        torch.ones(1, length, dtype=torch.bool),
        boundaries,
        cp_group=None,
        cp_rank=0,
    )[0]

    expected = []
    for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        expected += [
            {start + t for t in row} for row in _vllm_tokens(indexer, hidden[0, start:end], q_resid[0, start:end])
        ]
    sparse = sum(max(0, n - (POOLS + 1) * KPOOL + 1) for n in ROWS)
    assert sparse > 200
    assert _token_sets(trainer) == expected

    # The data separates the two numerics: the unquantized bf16 scoring of the HF indexer picks other tokens.
    row = slice(0, ROWS[0])
    unquantized = indexer(hidden[:, row], q_resid[:, row], torch.ones(1, ROWS[0], dtype=torch.bool), None)[0]
    assert _token_sets(unquantized) != expected[row]
