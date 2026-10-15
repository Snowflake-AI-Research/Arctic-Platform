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
"""Per-step router health statistics as a trainer ``MetricsCallback``.

Hooks every ``TokenChoiceTopKRouter`` when ``router_stats_sample_stride`` > 0. The load-balancing buffer
``tokens_per_expert`` accumulates for the life of the model and counts router-replay pad rows, which replay
zero-fills, as ``top_k`` tokens on expert 0. These statistics keep their own per-step accumulator instead:

- ``tokens_per_expert``: the router's own histogram, less the pad rows' ``top_k`` on expert 0. A real top-k never
  repeats an expert, so a row of zeros is a pad row.
- ``entropy_sum`` / ``sampled_tokens``: entropy (nats) of the router's normalized scores, on every
  ``stride``-th token from an offset that advances each observed call.
- ``replay_agreement_sum`` / ``replayed_tokens``: on the same sample, when routing was replayed, the fraction of
  the replayed experts the router's own top-k also picks.

Totals are taken on every ``call_stride``-th router call only. Scheduled padding calls and activation-checkpoint
reruns are not counted. Forward-only calls, including reference logprobs, are included; ``post_step`` publishes
local totals with ``Reduce.SUM`` so the driver ``merge_metrics`` sums across ranks (no callback-owned collective).
"""

from __future__ import annotations

from typing import Any
from typing import Mapping

import torch
import torch.nn.functional as F

from arctic_platform.common.callbacks import MetricsCallback
from arctic_platform.common.callbacks import Reduce
from arctic_platform.common.callbacks import Trainer
from arctic_platform.model.implementations.debug.config import router_stats_strides
from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter
from arctic_platform.model.recompute import in_recompute
from arctic_platform.model.recompute import install_recompute_detector

_TOTALS = ("sampled_tokens", "entropy_sum", "replayed_tokens", "replay_agreement_sum")


class RouterStatsCallback(MetricsCallback):
    """Installs forward hooks on every token-choice router and logs local totals at ``post_step``.

    Each rank logs its own totals with ``Reduce.SUM``. The driver merges via ``merge_metrics``; there is no
    callback-owned ``all_reduce``. Homogeneous EP/SP is assumed: every enabled rank hooks the same routers.
    ``on_error`` drops this rank's partial totals (DSS broadcasts ``on_error`` so siblings reset too).
    """

    def __init__(self, training_config: Mapping[str, Any]) -> None:
        super().__init__(training_config)
        self._stride, self._call_stride = router_stats_strides(training_config)
        self._counting = True
        self._layers: list[tuple[str, torch.Tensor]] = []
        self._calls: list[int] = []

    @classmethod
    def enabled(cls, training_config: Mapping[str, Any]) -> bool:
        return bool(router_stats_strides(training_config)[0])

    def post_init(self, trainer: Trainer) -> None:
        for name, router in trainer.model.named_modules():
            if self._stride == 0 or not isinstance(router, TokenChoiceTopKRouter):
                continue
            if router.top_k < 2:
                raise ValueError(f"router_stats cannot read {name}: a top-1 replay pad row looks like expert 0")
            totals = torch.zeros(router.num_experts + len(_TOTALS), device=router.gate.weight.device)
            router.register_forward_hook(self._hook(len(self._layers), totals), with_kwargs=True)
            self._layers.append((name, totals))
            self._calls.append(0)
        install_recompute_detector()

    def pre_fwd_bwd_microbatch(self, trainer: Trainer, is_padding: bool) -> None:
        self._counting = not is_padding

    def pre_fwd_only_microbatch(self, trainer: Trainer, is_padding: bool) -> None:
        self._counting = not is_padding

    def post_fwd_bwd_microbatch(self, trainer: Trainer, is_padding: bool) -> None:
        self._counting = True

    def post_fwd_only_microbatch(self, trainer: Trainer, is_padding: bool) -> None:
        self._counting = True

    def post_step(self, trainer: Trainer) -> None:
        report = self._report()
        if report:
            self.log("router_stats", report, reduce=Reduce.SUM)

    def on_error(self, trainer: Trainer, stage: str, exc: BaseException) -> None:
        # fwd_bwd / forward_only can leave partial totals; clear those so a retry
        # does not double-count. step failures must keep the completed fwd_bwd
        # totals so a step-only retry still publishes them via post_step.
        if stage in {"fwd_bwd", "forward_only"}:
            self._reset()
        else:
            self._counting = True

    def _reset(self) -> None:
        """Drop partial totals and count the next router call."""
        self._counting = True
        self._calls = [0 for _ in self._calls]
        for _, totals in self._layers:
            totals.zero_()

    def _report(self) -> dict[str, dict]:
        """Local per-router totals for this rank; the driver sums them with ``Reduce.SUM``."""
        if not self._layers:
            return {}
        width = len(_TOTALS)
        report: dict[str, dict] = {}
        for name, totals in self._layers:
            row = totals.detach().cpu().tolist()
            report[name] = {
                "tokens_per_expert": row[:-width],
                **dict(zip(_TOTALS, row[-width:])),
            }
            totals.zero_()
        return report

    def _hook(self, layer: int, totals: torch.Tensor):
        @torch.compiler.disable
        def accumulate(router: TokenChoiceTopKRouter, args, kwargs, output) -> None:
            if not self._counting or in_recompute():
                return
            call, self._calls[layer] = self._calls[layer], self._calls[layer] + 1
            if call % self._call_stride:
                return
            offset = call // self._call_stride % self._stride
            x = args[0]
            expert_bias = args[1] if len(args) > 1 else kwargs.get("expert_bias")
            selected, histogram = output[1], output[2]
            experts, top_k = router.num_experts, router.top_k
            with torch.no_grad():
                real = selected[:, 0] != selected[:, 1]
                totals[:experts].add_(histogram)
                totals[0].add_(real.sum(), alpha=top_k).sub_(top_k * len(x))

                picks = torch.arange(min(offset, len(x)), len(x), self._stride, device=x.device)
                sampled = real.index_select(0, picks).to(totals.dtype)
                logits = router.gate(x.index_select(0, picks)).to(torch.float32)
                log_scores = F.logsigmoid(logits) if router.score_func == "sigmoid" else logits
                log_probs = log_scores.log_softmax(-1)
                entropy = -(log_probs.exp() * log_probs).sum(-1)
                sampled_tokens = sampled.sum()
                sums = [sampled_tokens, sampled @ entropy]
                if kwargs.get("routed_experts") is not None:
                    scores = torch.sigmoid(logits) if router.score_func == "sigmoid" else F.softmax(logits, dim=-1)
                    own = torch.topk(scores if expert_bias is None else scores + expert_bias, top_k, dim=-1).indices
                    hits = (own.unsqueeze(-1) == selected.index_select(0, picks).unsqueeze(-2)).any(-1)
                    sums += [sampled_tokens, sampled @ hits.to(totals.dtype).mean(-1)]
                totals[experts : experts + len(sums)].add_(torch.stack(sums))

        return accumulate
