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

"""The four MoE combine paths fold a token's expert rows through one shared implementation.

Token-choice routing sends a token to ``top_k`` experts, so a token's layer output is the sum of several rows of
expert output. That sum was written four times -- once in each expert-parallel backend and once in each of the
two rank-local forwards -- so a change to how it sums, or a check added to it, reached one path and not the
others. The cases here fix what the shared implementation computes and that each expert-parallel backend
reaches it.

Two of those four copies are the ones this change touches, and they are the two that run. The rank-local
folds in ``layers.moe`` are left as they are: ``MoE.forward`` and ``LatentMoE.forward`` return from their
dispatch branch for both values ``EPCommBackend`` allows, and ``GroupedExperts.forward`` raises
``NotImplementedError`` for any other value, so no configuration executes them and no case here could assert
what a converted version computed.
"""

from __future__ import annotations

import importlib

import torch

from arctic_platform.model.implementations.moe.token_combine import sum_rows_by_token

TOKENS = 64
HIDDEN = 32
TOP_K = 4


def _reference_fold(rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Fold row by row in float64, which no device reduction has freedom to reassociate."""
    output = torch.zeros((num_tokens, rows.shape[1]), dtype=torch.float64)
    for row, token in zip(rows.to(torch.float64), token_indices.tolist()):
        output[token] += row
    return output


def test_the_shared_fold_sums_each_tokens_rows():
    """Every row lands on the token its index names, and on no other."""
    torch.manual_seed(0)
    token_indices = torch.arange(TOKENS).repeat_interleave(TOP_K)
    rows = torch.randn((token_indices.shape[0], HIDDEN), dtype=torch.float64)

    torch.testing.assert_close(
        sum_rows_by_token(rows, token_indices, TOKENS),
        _reference_fold(rows, token_indices, TOKENS),
        rtol=0,
        atol=0,
    )


def test_the_shared_fold_drops_rows_the_index_does_not_name():
    """The expert stage can return a padded tail, and those rows belong to no token.

    The scatter iterates over the index rather than over the source, so a source longer than the index keeps
    only the rows the index reaches. Every path relied on that, which makes it a property of the shared fold
    rather than an accident of one caller's shapes.
    """
    rows = torch.cat([torch.ones((2, HIDDEN)), torch.full((3, HIDDEN), 99.0)])
    token_indices = torch.tensor([0, 1], dtype=torch.long)

    folded = sum_rows_by_token(rows, token_indices, 2)
    torch.testing.assert_close(folded, torch.ones((2, HIDDEN)), rtol=0, atol=0)


def test_the_shared_fold_of_no_rows_is_a_zero_block():
    """A rank that received nothing still owes its peers an output block of the full shape."""
    folded = sum_rows_by_token(torch.zeros((0, HIDDEN)), torch.zeros(0, dtype=torch.long), TOKENS)

    assert folded.shape == (TOKENS, HIDDEN)
    assert not folded.any()


def test_the_shared_fold_differentiates():
    """The combine sits on the training forward path, so its backward has to carry the gradient."""
    token_indices = torch.arange(TOKENS).repeat_interleave(TOP_K)
    rows = torch.randn((token_indices.shape[0], HIDDEN), requires_grad=True)
    upstream = torch.randn((TOKENS, HIDDEN))

    (sum_rows_by_token(rows, token_indices, TOKENS) * upstream).sum().backward()

    assert rows.grad is not None
    torch.testing.assert_close(rows.grad, upstream.index_select(0, token_indices), rtol=0, atol=0)


def test_every_expert_parallel_backend_folds_through_the_shared_implementation():
    """Each backend's unpermute reaches the shared fold, which is what having one of them is for.

    Checked by putting a recording fold in the backend's module and calling that backend's own unpermute: a
    backend that kept its inline scatter returns the same value without the recording fold being reached, which
    is the case this rejects. A backend whose transport is absent on the host is skipped rather than asserted
    over, since importing it imports that transport, and the case fails if none could be imported so that the
    requirement is never stated over an empty set.
    """
    reached = {}
    for module_name, fold_name in (
        ("arctic_platform.model.implementations.moe.distributed.token_permute", "unpermute_tokens"),
        ("arctic_platform.model.implementations.moe.distributed.ucclep", "_unpermute_tokens"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        original = module.sum_rows_by_token
        calls = []

        def recording(rows, token_indices, num_tokens, calls=calls):
            calls.append(num_tokens)
            return sum_rows_by_token(rows, token_indices, num_tokens)

        module.sum_rows_by_token = recording
        try:
            getattr(module, fold_name)(torch.ones((2, 4)), torch.zeros(2, dtype=torch.long), 1)
        finally:
            module.sum_rows_by_token = original
        reached[module_name] = calls

    assert reached, "no expert-parallel backend module could be imported, so nothing was checked"
    for module_name, calls in sorted(reached.items()):
        assert calls == [1], (
            f"{module_name}'s unpermute does not fold through models.moe.token_combine.sum_rows_by_token, so a "
            "change to the shared fold reaches every other combine path and not this one"
        )
