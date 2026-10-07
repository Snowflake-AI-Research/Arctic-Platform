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
"""Candidate-token and complement-bucket log probabilities for chunked LM heads.

For ``group_token_ids`` shaped ``[N, M]``, the accumulator returns ``[N, M+1]``:
one log probability per requested token followed by the log probability mass of
every vocabulary token outside the requested group. Computing that complement
with its own logsumexp avoids the precision loss of ``log(1 - sum(p))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def validate_group_temperature(temperature) -> None:
    """Require grouped probabilities from the model's untempered distribution."""
    if temperature is None:
        return
    if torch.is_tensor(temperature):
        received = temperature.detach().reshape(-1)[:8].cpu().tolist()
        valid = (
            temperature.dtype != torch.bool
            and not torch.is_complex(temperature)
            and bool(torch.isfinite(temperature).all().item())
            and bool((temperature == 1).all().item())
        )
    else:
        received = [temperature]
        valid = isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and temperature == 1
    if not valid:
        raise ValueError(
            "grouped LM-head probabilities require temperature 1 at every position; "
            f"received temperature values {received!r}"
        )


def prepare_group_temperature(
    temperature,
    *,
    token_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Validate grouped temperature and broadcast scalar forms to every token."""
    validate_group_temperature(temperature)
    if torch.is_tensor(temperature) and temperature.numel() != 1:
        return temperature.to(device)
    return torch.ones(token_shape, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class _GroupGradientChunk:
    rows: slice
    candidate_gradients: torch.Tensor
    tail_log_normalizer: torch.Tensor


class GroupedLogProbAccumulator:
    """Accumulate grouped logits over a head's token-by-vocabulary tiles."""

    def __init__(
        self,
        group_token_ids: torch.Tensor,
        inv_temperature: torch.Tensor,
        vocab_size: int,
    ):
        if not torch.is_tensor(group_token_ids):
            raise ValueError("group_token_ids must be a tensor")
        if group_token_ids.dtype == torch.bool or group_token_ids.is_floating_point() or group_token_ids.is_complex():
            raise ValueError(f"group_token_ids must contain integer token ids, got {group_token_ids.dtype}")
        n_tokens = int(inv_temperature.numel())
        if group_token_ids.ndim < 2 or n_tokens == 0 or group_token_ids.numel() % n_tokens:
            raise ValueError(
                f"group_token_ids shape {tuple(group_token_ids.shape)} must provide the same "
                f"candidate width at each of {n_tokens} token positions"
            )
        if not bool((inv_temperature == 1).all().item()):
            received = inv_temperature.detach().reciprocal().reshape(-1)[:8].cpu().tolist()
            raise ValueError(
                "grouped LM-head probabilities require temperature 1 at every position; "
                f"received temperature values {received!r}"
            )

        self.token_ids = group_token_ids.reshape(n_tokens, -1).to(inv_temperature.device)
        invalid = (self.token_ids < -1) | (self.token_ids >= vocab_size)
        if bool(invalid.any().item()):
            first_invalid = int(self.token_ids[invalid][0].item())
            raise ValueError(
                f"group_token_ids contains token id {first_invalid} outside the vocabulary "
                f"[0, {vocab_size}); use -1 only for padding"
            )
        self.groups = torch.full(
            (n_tokens, self.token_ids.shape[1] + 1),
            float("-inf"),
            dtype=torch.float32,
            device=inv_temperature.device,
        )
        self.candidate_logits = self.groups[:, :-1]
        self.tail_logits = self.groups[:, -1]

    def _extract_candidates_and_exclude_from_tail(
        self,
        scaled_logits: torch.Tensor,
        rows: slice,
        vocab_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather candidates in this tile and mask them out of its tail logits."""
        token_ids = self.token_ids[rows]
        tile_width = scaled_logits.shape[1]
        in_tile = (token_ids >= vocab_start) & (token_ids < vocab_start + tile_width)
        columns = (token_ids - vocab_start).clamp(0, tile_width - 1)
        candidate_logits = scaled_logits.gather(1, columns)
        # Out-of-tile candidates contribute +inf to amin and leave their
        # clamped columns unchanged.
        scaled_logits.scatter_reduce_(
            1,
            columns,
            torch.where(in_tile, float("-inf"), float("inf")),
            reduce="amin",
        )
        return in_tile, columns, candidate_logits

    def accumulate_forward_tile(
        self,
        scaled_logits: torch.Tensor,
        token_start: int,
        vocab_start: int,
    ) -> None:
        rows = slice(token_start, token_start + scaled_logits.shape[0])
        in_tile, _, candidate_logits = self._extract_candidates_and_exclude_from_tail(scaled_logits, rows, vocab_start)
        self.candidate_logits[rows] = torch.where(in_tile, candidate_logits, self.candidate_logits[rows])
        self.tail_logits[rows] = torch.logaddexp(self.tail_logits[rows], scaled_logits.logsumexp(-1))

    def normalized_log_probs(self, log_normalizer: torch.Tensor) -> torch.Tensor:
        return self.groups - log_normalizer.to(torch.float32)[:, None]

    def combine_output_gradients(
        self,
        logprob_gradients: torch.Tensor,
        group_gradients: torch.Tensor,
        token_start: int,
        token_end: int,
    ) -> tuple[torch.Tensor, _GroupGradientChunk]:
        """Combine all ``-logsumexp`` gradients for one token chunk."""
        rows = slice(token_start, token_end)
        finite_groups = torch.isfinite(self.groups[rows])
        candidate_gradients = torch.where(finite_groups, group_gradients[rows], 0.0)
        normalizer_gradients = logprob_gradients + candidate_gradients.sum(-1)
        state = _GroupGradientChunk(
            rows,
            candidate_gradients,
            torch.where(
                finite_groups[:, -1],
                self.tail_logits[rows],
                0.0,
            ),
        )
        return normalizer_gradients, state

    def add_group_gradients_to_tile(
        self,
        grad_logits: torch.Tensor,
        scaled_logits: torch.Tensor,
        state: _GroupGradientChunk,
        vocab_start: int,
    ) -> None:
        """Add candidate one-hots and complement softmax gradients to a tile."""
        in_tile, columns, _ = self._extract_candidates_and_exclude_from_tail(scaled_logits, state.rows, vocab_start)
        grad_logits.scatter_add_(
            1,
            columns,
            torch.where(in_tile, state.candidate_gradients[:, :-1], 0.0),
        )
        complement_probabilities = scaled_logits.sub(state.tail_log_normalizer[:, None]).exp_()
        grad_logits.addcmul_(
            complement_probabilities,
            state.candidate_gradients[:, -1:],
        )
